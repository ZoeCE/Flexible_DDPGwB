# ==============================================================================
# test.py — 统一测试脚本（PPO / TD3 / NMPC）
#
# 支持的测试模式：
#   --mode ppo          : 加载 PPOAgent checkpoint，使用 Actor 输出 7D 关节角
#   --mode td3          : 加载 TD3Agent checkpoint，使用 Actor 输出 7D 关节角
#   --mode nmpc         : 仅使用 NMPC + IK 专家控制器（基准测试）
#   --mode manual       : 键盘手动控制（调试用）
#
# 新版与旧版差异：
#   [TEST-NEW-1] 动作空间：7D 关节角
#     - 不再直接执行末端加速度
#     - Actor 输出 → 直接下发给 MuJoCo 位置执行器
#   [TEST-NEW-2] 统一 Agent 加载接口
#     - PPO 和 TD3 均使用 agent.load() 加载 checkpoint
#     - 根据 --mode 自动选择 Agent 类
#   [TEST-NEW-3] 专家控制器测试模式
#     - nmpc 模式：JointSpaceExpert 直接产生关节角并执行
# ==============================================================================
 
import os
import sys
import copy
import time
import argparse
import numpy as np
import torch
 
from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from agent import PPOAgent, TD3Agent
 
 
# ==============================================================================
# 工具函数
# ==============================================================================
 
def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")
 
 
def build_config(args) -> dict:
    """
    从命令行参数构建测试配置。

    n_obstacles 的优先级（高 → 低）：
      1. --obstacles <N> 命令行显式指定（最高）
      2. config["test"]["n_obstacles"]（测试专用默认）
      3. config["scene"]["n_obstacles"]（训练上限，最后兜底）

    约束：最终 n_obstacles 必须 ≤ config["scene"]["n_obstacles"]
          （即 state_dim 上限，超出会导致观测/checkpoint 维度不一致）
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["sim"]["render"] = args.render
    config["scene"]["seed"] = args.seed
    config["train"]["gpu_id"] = args.gpu

    # === n_obstacles 优先级处理 ===
    scene_max = int(config["scene"]["n_obstacles"])        # state_dim 上限
    test_default = int(config.get("test", {}).get("n_obstacles", scene_max))

    if args.obstacles is not None:
        requested = int(args.obstacles)
    else:
        requested = test_default

    # 硬约束：不能超过 scene.n_obstacles（state_dim 依赖）
    if requested > scene_max:
        print(f"[WARN] --obstacles={requested} 超过 scene.n_obstacles={scene_max}（state_dim 上限）")
        print(f"       已自动 clip 到 {scene_max}。如需更多障碍物，请修改 config['scene']['n_obstacles']。")
        requested = scene_max

    # 关键：scene.n_obstacles 保持为上限（决定 state_dim）；
    # 实际生成的障碍物数量通过 env.set_curriculum_n_obstacles(requested) 控制
    # 把 runtime 实际值保存在一个临时 key 里，供 run_test 使用
    config["_runtime_n_obstacles"] = requested

    print(f"[build_config] 障碍物数量：实际生成={requested}, state_dim 上限={scene_max}")

    # 覆盖起终点（如有）
    if args.start_xy:
        config["task"]["default_start_xy"] = list(map(float, args.start_xy.split(",")))
    if args.target_xy:
        config["task"]["default_target_xy"] = list(map(float, args.target_xy.split(",")))

    return config
 
 
def print_summary(mode: str, n_episodes: int, success_count: int,
                  collision_count: int, total_steps_success: int,
                  elapsed: float):
    avg_steps = total_steps_success / max(success_count, 1)
    print("\n" + "="*55)
    print(f"  测试模式：{mode.upper()}")
    print(f"  总回合数：{n_episodes}")
    print(f"  成功率：  {success_count}/{n_episodes} "
          f"({success_count/n_episodes*100:.2f}%)")
    print(f"  碰撞率：  {collision_count}/{n_episodes} "
          f"({collision_count/n_episodes*100:.2f}%)")
    print(f"  成功平均步数：{avg_steps:.1f}")
    print(f"  耗时：{elapsed:.1f}s")
    print("="*55 + "\n")
 
 
# ==============================================================================
# 通用测试循环
# ==============================================================================
 
def run_test(mode: str, config: dict, n_episodes: int,
             ckpt_path: str = None, gpu_id: int = 0,
             save_paths_dir: str = None):
    """
    统一测试主循环。
    mode: "ppo" | "td3" | "nmpc"
    """
    env = CableRobotEnvWithObstacles(config=config)

    # [FIX] 设置运行时实际障碍物数量（不改动 state_dim）
    runtime_n_obs = config.get("_runtime_n_obstacles", config["scene"]["n_obstacles"])
    actual_n = env.set_curriculum_n_obstacles(int(runtime_n_obs))
    print(f"[run_test] 实际生成障碍物数量 = {actual_n}（state_dim 上限 = {env.n_obstacles}）")

    STATE_DIM  = env.state_dim
    ACTION_DIM = config["space"]["action_dim"]
    # env.step() 期望 delta_q，范围应为 ±dq_max
    dq_max     = np.array(config["space"].get("dq_max", [0.1]*ACTION_DIM))
    ACT_LOW    = -dq_max
    ACT_HIGH   =  dq_max
 
    # ── Agent / 专家初始化 ────────────────────────────────────────────────────
    agent  = None
    expert = None
 
    if mode == "ppo":
        print(f"[Test] 加载 PPO checkpoint: {ckpt_path}")
        agent = PPOAgent(None, STATE_DIM, ACTION_DIM, config=config)
        if ckpt_path and os.path.exists(ckpt_path):
            agent.load(ckpt_path, map_location=get_device(gpu_id))
            print("  ✅ 模型加载成功")
        else:
            print(f"  ⚠️  找不到 checkpoint，使用随机初始化策略: {ckpt_path}")
        # 专家仅用于 nmpc 基准对比，ppo 模式不需要
        # 但保留 expert 以便 rollout 时记录 BC 差距（可选）
        expert = JointSpaceExpert(config, env.ik_solver)
 
    elif mode == "td3":
        print(f"[Test] 加载 TD3 checkpoint: {ckpt_path}")
        agent = TD3Agent(None, STATE_DIM, ACTION_DIM, config=config)
        if ckpt_path and os.path.exists(ckpt_path):
            agent.load(ckpt_path, map_location=get_device(gpu_id))
            print("  ✅ 模型加载成功")
        else:
            print(f"  ⚠️  找不到 checkpoint，使用随机初始化策略")
        expert = JointSpaceExpert(config, env.ik_solver)
 
    elif mode == "nmpc":
        print("[Test] 使用 NMPC + IK 专家控制器（纯专家基准）")
        expert = JointSpaceExpert(config, env.ik_solver)
 
    else:
        raise ValueError(f"未知测试模式: {mode}")
 
    if save_paths_dir:
        os.makedirs(save_paths_dir, exist_ok=True)
 
    # ── 测试循环 ──────────────────────────────────────────────────────────────
    success_count        = 0
    collision_count      = 0
    total_steps_success  = 0
    is_ppo               = (mode == "ppo")
    is_td3               = (mode == "td3")
 
    print("*" * 55)
    print(f"  开始测试 [{mode.upper()}] | {n_episodes} 回合 | "
          f"渲染: {'开' if config['sim']['render'] else '关'}")
    print("*" * 55)
 
    t_start = time.time()
 
    # [SKIP-INVALID] 改为 while 循环：路径规划失败的场景跳过，不计入 n_episodes
    # ep_count = 有效回合数（参与统计），attempt_count = 总尝试数
    ep_count = 0
    attempt_count = 0
    max_attempts = n_episodes * 3   # 安全上限，避免死循环

    while ep_count < n_episodes and attempt_count < max_attempts:
        attempt_count += 1
        obs = env.reset()
        current_q = env.data.qpos[:7].copy()

        # [SKIP-INVALID] 路径规划失败 → 跳过，不计入
        planned_path = env.get_planned_path()
        if planned_path is None:
            print(f"  [Skip] 第 {attempt_count} 次尝试：路径规划失败，跳过（不计入）")
            continue

        if expert is not None:
            # [INTERFACE] 与训练保持一致：传 env=env，让 expert 用真实 EE 位置
            expert.reset(obs, current_q, env=env)
            expert.set_path(planned_path)

        # 确认本场景有效，计入回合数
        ep_count += 1
        ep = ep_count - 1   # 用于后续打印 (ep+1) 与原逻辑兼容

        ep_reward = 0.0; step = 0
        ep_success = False; ep_collision = False
 
        # 保存路径（可选）
        trajectory = []
 
        while True:
            # ── 选择动作 ────────────────────────────────────────────────────
            if mode == "nmpc":
                # 纯专家：NMPC → IK → 绝对关节角 → 转 delta_q
                current_q = env.data.qpos[:7].copy().astype(np.float32)
                q_target = expert.compute_joint_target(obs, current_q)
                action = np.clip(q_target - current_q, ACT_LOW, ACT_HIGH)
 
            elif mode == "ppo":
                norm_obs = agent.normalize_obs(obs, update=False)
                action, _, _ = agent.act(norm_obs, deterministic=True)
 
            elif mode == "td3":
                norm_obs = agent.normalize_obs(obs, update=False)
                action, _ = agent.act(norm_obs)
 
            # NaN 保护
            if np.isnan(action).any():
                current_q = env.data.qpos[:7].copy().astype(np.float32)
                if expert:
                    q_target = expert.compute_joint_target(obs, current_q)
                    action = np.clip(q_target - current_q, ACT_LOW, ACT_HIGH)
                else:
                    action = np.zeros(ACTION_DIM, dtype=np.float32)
 
            # ── 环境推进 ────────────────────────────────────────────────────
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
 
            ep_reward += reward; step += 1
 
            if info.get("is_success"):    ep_success   = True
            if info.get("is_collision"):  ep_collision = True
 
            if save_paths_dir:
                trajectory.append(env.data.body("prefab").xpos.copy())
 
            if config["sim"]["render"]:
                time.sleep(0.01)
 
            obs = next_obs
            if done: break
 
        # ── 回合结算 ─────────────────────────────────────────────────────────
        if ep_success:    success_count += 1;   total_steps_success += step
        if ep_collision:  collision_count += 1
 
        status  = "✅ 成功" if ep_success   else "❌ 失败"
        col_str = " (碰撞!)" if ep_collision else ""
        print(f"  Ep {ep+1:3d} | {status}{col_str} | "
              f"总奖励: {ep_reward:7.2f} | 步数: {step:3d}")
 
        # 保存轨迹
        if save_paths_dir and trajectory:
            import csv
            fpath = os.path.join(save_paths_dir, f"ep{ep+1:03d}.csv")
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y", "z"])
                w.writerows(trajectory)
 
    # ── 汇总 ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    skipped = attempt_count - ep_count
    if skipped > 0:
        print(f"\n  [Skip 统计] 跳过 {skipped} 个场景（路径规划失败），"
              f"有效回合 {ep_count}/{n_episodes}，总尝试 {attempt_count}")
    print_summary(mode, ep_count, success_count, collision_count,
                  total_steps_success, elapsed)
 
    env.close()
    return {
        "success_rate":  success_count / max(ep_count, 1),
        "collision_rate": collision_count / max(ep_count, 1),
        "avg_steps":     total_steps_success / max(success_count, 1),
        "episodes_valid": ep_count,
        "episodes_skipped": skipped,
    }
 
 
# ==============================================================================
# 手动控制模式
# ==============================================================================
 
def run_manual(config: dict):
    """键盘手动控制，用于调试。控制 4D 加速度（通过专家内部积分→IK执行）。"""
    config["sim"]["render"] = True
    env = CableRobotEnvWithObstacles(config=config)

    # [FIX] 与 run_test 一致：用 set_curriculum_n_obstacles 设定实际障碍物数量
    runtime_n_obs = config.get("_runtime_n_obstacles", config["scene"]["n_obstacles"])
    actual_n = env.set_curriculum_n_obstacles(int(runtime_n_obs))
    print(f"[run_manual] 实际生成障碍物数量 = {actual_n}（state_dim 上限 = {env.n_obstacles}）")

    expert = JointSpaceExpert(config, env.ik_solver)
 
    key_state = {"paused": True, "acc": np.zeros(4)}  # [ax, ay, az, ayaw]
    STEP = 0.3
 
    def key_cb(keycode):
        KEY_SPACE=32; KEY_R=262; KEY_L=263; KEY_U=264; KEY_D=265
        KEY_UP_Z=46; KEY_DN_Z=44; KEY_YL=91; KEY_YR=93
        if keycode == KEY_SPACE:
            key_state["paused"] = not key_state["paused"]
            print("PAUSED" if key_state["paused"] else "RUNNING"); return
        a = key_state["acc"]
        if keycode == KEY_R: a[0] += STEP
        elif keycode == KEY_L: a[0] -= STEP
        elif keycode == KEY_U: a[1] += STEP
        elif keycode == KEY_D: a[1] -= STEP
        elif keycode == KEY_UP_Z: a[2] += STEP
        elif keycode == KEY_DN_Z: a[2] -= STEP
        elif keycode == KEY_YL: a[3] += 0.5
        elif keycode == KEY_YR: a[3] -= 0.5
 
    env._key_callback = key_cb
    if env.viewer: env.viewer.close()
 
    import mujoco
    env.viewer = mujoco.viewer.launch_passive(
        env.model, env.data, key_callback=key_cb
    )
 
    print("="*50)
    print("手动控制模式")
    print("  方向键 = XY 移动  , / . = Z 升降  [ / ] = 偏航  SPACE = 暂停")
    print("  关闭窗口退出")
    print("="*50)
 
    obs = env.reset()
    current_q = env.data.qpos[:7].copy()
    expert.reset(obs, current_q, env=env)
    planned_path = env.get_planned_path()
    if planned_path: expert.set_path(planned_path)
    print("[重置完成] 已暂停，按 SPACE 开始")
 
    while env.viewer.is_running():
        if key_state["paused"]:
            env.viewer.sync(); time.sleep(0.02); continue
 
        # 用手动输入的 4D 加速度覆盖专家内部 MPC（通过修改 expert 的积分状态）
        acc = key_state["acc"].copy(); key_state["acc"][:] = 0.0
        dt  = expert.dt
        expert._ee_pos += expert._ee_vel * dt + 0.5 * acc[:3] * dt**2
        expert._ee_vel += acc[:3] * dt
        expert._ee_yaw += acc[3] * dt
        if expert._ee_pos[2] < 0.25:
            expert._ee_pos[2] = 0.25; expert._ee_vel[2] = 0.0
 
        current_q = env.data.qpos[:7].copy().astype(np.float32)
        q_target = env.ik_solver.solve_4d(
            current_q, expert._ee_pos[0], expert._ee_pos[1],
            expert._ee_pos[2], expert._ee_yaw
        )
        action = np.clip(q_target - current_q, env.action_space_low, env.action_space_high)
 
        obs, reward, done, _, info = env.step(action)
        if done:
            status = "成功✅" if info.get("is_success") else "结束"
            print(f"  [{status}] 奖励={reward:.2f}")
            obs = env.reset()
            current_q = env.data.qpos[:7].copy()
            expert.reset(obs, current_q, env=env)
            planned_path = env.get_planned_path()
            if planned_path: expert.set_path(planned_path)
            key_state["paused"] = True
            print("[重置完成] 已暂停")
 
        time.sleep(0.02)
 
    print("窗口已关闭。")
    env.close()
 
 
# ==============================================================================
# 命令行入口
# ==============================================================================
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="索驱动机器人测试脚本")
 
    parser.add_argument("--mode",     type=str, default="ppo",
                        choices=["ppo", "td3", "nmpc", "manual"],
                        help="测试模式")
    parser.add_argument("--ckpt",     type=str, default=None,
                        help="checkpoint 路径（ppo/td3 模式必填）")
    parser.add_argument("--log-dir",  type=str, default="saves/ppo_run",
                        help="若 --ckpt 未指定，从此目录自动找 ckpt_latest.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--gpu",      type=int, default=0)
 
    # 场景参数
    # [FIX] default=None：不指定 --obstacles 时，从 config["test"]["n_obstacles"] 读取
    # 显式指定 --obstacles N 时才覆盖，且会 clip 到 scene.n_obstacles 上限
    parser.add_argument("--obstacles",  type=int, default=None,
                        help="实际生成的障碍物数量（默认从 config['test']['n_obstacles'] 读取，"
                             "必须 ≤ config['scene']['n_obstacles']）")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--start-xy",   type=str, default=None, dest="start_xy",
                        help="起始 XY，例如 '0.3,0.15'")
    parser.add_argument("--target-xy",  type=str, default=None, dest="target_xy",
                        help="目标 XY，例如 '-0.3,0.2'")
    parser.add_argument("--save-paths", type=str, default=None, dest="save_paths_dir",
                        help="保存轨迹 CSV 的目录")
 
    args = parser.parse_args()
 
    config = build_config(args)
 
    # 自动推断 checkpoint 路径
    ckpt_path = args.ckpt
    if ckpt_path is None and args.mode in ["ppo", "td3"]:
        ckpt_path = os.path.join(args.log_dir, "ckpt_bc_pretrained.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(args.log_dir, "ckpt_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[Warn] 未找到 checkpoint，将使用随机初始化策略。"
                  f"（搜索路径：{args.log_dir}）")
            ckpt_path = None
 
    if args.mode == "manual":
        run_manual(config)
    else:
        run_test(
            mode=args.mode,
            config=config,
            n_episodes=args.episodes,
            ckpt_path=ckpt_path,
            gpu_id=args.gpu,
            save_paths_dir=args.save_paths_dir,
        )