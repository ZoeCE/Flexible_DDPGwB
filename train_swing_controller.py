# ==============================================================================
# train_swing_controller.py — 底层防摆 RL 控制器训练脚本
#
# 训练流程：
#   阶段 1: BC 暖启动（可选） — 用 JointSpaceExpert 收集数据预训练 Actor
#   阶段 2: PPO 训练 — 在带风力扰动的环境中微调
#
# 风力课程学习：训练初期无风 → 逐步增加到 F_max
# ==============================================================================

import os
import sys
import copy
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from swing_controller import (
    SwingControllerAgent,
    build_swing_obs,
    compute_swing_reward,
    SWING_OBS_DIM,
    OBS_TILT, OBS_PL_VX, OBS_PL_VY,
    OBS_EE_X, OBS_EE_Y, OBS_PL_X, OBS_PL_Y, OBS_PL_Z,
)


def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# BC 暖启动
# ==============================================================================

def bc_pretrain_swing(agent: SwingControllerAgent,
                      config: dict,
                      env: CableRobotEnvWithObstacles,
                      expert: JointSpaceExpert,
                      n_episodes: int = 200,
                      n_epochs: int = 10,
                      lr: float = 1e-3,
                      batch_size: int = 256):
    """
    用专家数据预训练底层 Actor 的 mean_head（冻结 log_std）。

    采集 swing_obs → 专家 Δq 的配对数据，用 MSE 训练。
    """
    print(f"\n{'='*60}")
    print(f"  底层控制器 BC 暖启动 ({n_episodes} 回合, {n_epochs} epochs)")
    print(f"{'='*60}")

    dq_max = agent.dq_max

    # 收集数据
    all_obs = []
    all_dq  = []

    # 初始化 obs_norm
    print("  初始化观测归一化...")
    for _ in range(5):
        obs = env.reset()
        if env.get_planned_path() is None:
            continue
        for _ in range(30):
            action = np.random.uniform(-dq_max, dq_max).astype(np.float32)
            swing_obs, _, _ = build_swing_obs(obs, env, action)
            agent.normalize_obs(swing_obs, update=True)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    print(f"  观测归一化统计量就绪 (n={agent.obs_norm.n})")

    valid_eps = 0
    attempt_cnt = 0
    max_attempts = n_episodes * 3
    last_action = np.zeros(7, dtype=np.float32)
    prev_tilt = 0.0
    prev_yaw  = 0.0

    while valid_eps < n_episodes and attempt_cnt < max_attempts:
        attempt_cnt += 1
        obs = env.reset()
        planned_path = env.get_planned_path()
        if planned_path is None:
            continue

        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        expert.set_path(planned_path)
        valid_eps += 1

        last_action = np.zeros(7, dtype=np.float32)
        prev_tilt = 0.0
        prev_yaw  = 0.0

        while True:
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            bc_dq = expert.compute_delta_q_target(obs, current_q)

            if not np.any(np.isnan(bc_dq)):
                swing_obs, prev_tilt, prev_yaw = build_swing_obs(
                    obs, env, last_action, prev_tilt, prev_yaw)
                norm_obs = agent.normalize_obs(swing_obs, update=True)
                all_obs.append(norm_obs.copy())
                all_dq.append(bc_dq.copy())

            next_obs, _, terminated, truncated, _ = env.step(bc_dq)
            last_action = bc_dq.copy()
            obs = next_obs
            if terminated or truncated:
                break

        if valid_eps % 50 == 0:
            print(f"  收集进度: {valid_eps}/{n_episodes} 回合, "
                  f"{len(all_obs)} 样本")

    print(f"  收集完成: {len(all_obs)} 样本")

    if len(all_obs) < batch_size:
        print("  ⚠ 样本不足，跳过 BC 预训练")
        return

    # 训练 mean_head（冻结 log_std）
    obs_t = torch.tensor(np.array(all_obs), device=agent.device)
    dq_t  = torch.tensor(np.array(all_dq),  device=agent.device)
    n_samples = len(all_obs)

    # 只训练 mean_head 和 backbone，冻结 log_std
    bc_params = [p for name, p in agent.actor.named_parameters()
                 if name != 'log_std']
    bc_optimizer = torch.optim.Adam(bc_params, lr=lr, weight_decay=1e-5)

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = indices[start: start + batch_size]
            obs_b = obs_t[idx]
            dq_b  = dq_t[idx]

            # 直接 MSE（底层不用 tanh，无需 atanh 映射）
            mean, _ = agent.actor._dist(obs_b)
            loss = F.mse_loss(mean, dq_b)

            bc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_params, 1.0)
            bc_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            with torch.no_grad():
                eval_n = min(2000, n_samples)
                pred, _ = agent.actor._dist(obs_t[:eval_n])
                mae = (pred - dq_t[:eval_n]).abs().mean().item()
            print(f"    Epoch {epoch+1:3d} | Loss={avg_loss:.5f} | MAE={mae:.4f}")

    print(f"  BC 暖启动完成\n")


# ==============================================================================
# PPO 训练主循环
# ==============================================================================

def train_swing_controller(config: dict, log_dir: str):
    """底层防摆控制器的完整训练流程。"""
    os.makedirs(log_dir, exist_ok=True)
    set_global_seed(42)

    cfg_swing = config.get("swing_controller", {})
    cfg_wind  = config.get("wind", {})

    TOTAL_STEPS  = int(cfg_swing.get("total_timesteps", 1_000_000))
    N_STEPS      = int(cfg_swing.get("n_steps", 2048))
    EVAL_INTERVAL = int(cfg_swing.get("eval_interval", 20))
    SAVE_INTERVAL = int(cfg_swing.get("save_interval", 50))

    # 风力课程
    wind_cur_start = int(cfg_wind.get("curriculum_start", 0))
    wind_cur_end   = int(cfg_wind.get("curriculum_end", 200_000))

    print("[SwingTrain] 初始化环境...")
    env = CableRobotEnvWithObstacles(config=config)
    env.set_curriculum_n_obstacles(0)  # 底层不需要障碍物

    print("[SwingTrain] 初始化 Agent...")
    agent = SwingControllerAgent(config=config)

    print("[SwingTrain] 初始化专家...")
    expert = JointSpaceExpert(config, env.ik_solver)

    # ── BC 暖启动 ──
    bc_epochs = int(cfg_swing.get("bc_pretrain_epochs", 10))
    if bc_epochs > 0:
        bc_env = CableRobotEnvWithObstacles(config=config)
        bc_env.set_curriculum_n_obstacles(0)
        bc_env.set_wind_curriculum(0.0)  # BC 阶段无风
        bc_expert = JointSpaceExpert(config, bc_env.ik_solver)
        bc_pretrain_swing(
            agent, config, bc_env, bc_expert,
            n_episodes=200,
            n_epochs=bc_epochs,
            lr=float(cfg_swing.get("bc_pretrain_lr", 1e-3)),
        )
        bc_env.close()
        agent.save(os.path.join(log_dir, "ckpt_bc.pt"))

    # ── PPO 训练 ──
    print(f"\n[SwingTrain] 开始 PPO 训练，目标步数 {TOTAL_STEPS}")
    print(f"  风力课程: {wind_cur_start} → {wind_cur_end} 步")

    log_file = os.path.join(log_dir, "swing_train_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "total_steps", "ep_reward", "ep_steps",
            "tilt_rms", "swing_vel_avg", "swing_offset_avg",
            "policy_loss", "value_loss", "entropy",
            "wind_frac",
        ])

    total_steps = 0
    episode = 0
    best_metric = -float('inf')
    t_start = time.time()

    # 滑动窗口统计
    recent_rewards = []
    recent_tilt_rms = []
    WINDOW = 20

    while total_steps < TOTAL_STEPS:
        # 风力课程
        if total_steps < wind_cur_start:
            wind_frac = 0.0
        elif total_steps >= wind_cur_end:
            wind_frac = 1.0
        else:
            wind_frac = (total_steps - wind_cur_start) / max(wind_cur_end - wind_cur_start, 1)
        env.set_wind_curriculum(wind_frac)

        obs = env.reset()
        planned_path = env.get_planned_path()
        if planned_path is None:
            continue

        expert.reset(obs, env.data.qpos[:7].copy(), env=env)
        expert.set_path(planned_path)

        # 回合状态
        last_action = np.zeros(7, dtype=np.float32)
        prev_tilt = 0.0
        prev_yaw  = 0.0
        prev_wp_dist = None
        ep_reward = 0.0
        ep_steps  = 0
        ep_tilts  = []
        ep_swing_vels = []
        ep_swing_offsets = []
        rollout_done = False

        while not rollout_done:
            # 构建底层观测
            swing_obs, prev_tilt, prev_yaw = build_swing_obs(
                obs, env, last_action, prev_tilt, prev_yaw)
            norm_obs = agent.normalize_obs(swing_obs, update=True)

            # 选择动作
            delta_q, log_prob, value = agent.act(norm_obs, deterministic=False)

            # 环境推进
            next_obs, env_reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated

            # 计算底层奖励
            reward, swing_term, rwd_info = compute_swing_reward(
                env, next_obs, delta_q, last_action, prev_wp_dist, config)

            # 使用底层奖励（而非环境原始奖励）
            prev_wp_dist = rwd_info.get("wp_dist", None)
            done = done or swing_term

            # 记录指标
            ep_tilts.append(float(next_obs[OBS_TILT]))
            ep_swing_vels.append(float(np.linalg.norm(
                next_obs[[OBS_PL_VX, OBS_PL_VY]])))
            ep_swing_offsets.append(float(np.linalg.norm(
                next_obs[[OBS_EE_X, OBS_EE_Y]] - next_obs[[OBS_PL_X, OBS_PL_Y]])))

            # Buffer
            agent.buffer.add(norm_obs, delta_q, reward, float(done),
                             value, log_prob)

            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            agent.total_steps = total_steps
            last_action = delta_q.copy()
            obs = next_obs

            # Buffer 满则更新
            if agent.buffer.full:
                if done:
                    last_val = 0.0
                else:
                    with torch.no_grad():
                        ns_norm = agent.normalize_obs(swing_obs, update=False)
                        s_t = torch.tensor(ns_norm, dtype=torch.float32,
                                           device=agent.device).unsqueeze(0)
                        last_val = agent.critic(s_t).item()

                agent.buffer.compute_returns_and_advantages(
                    last_val, agent.gamma, agent.gae_lambda)
                train_info = agent.update()
                rollout_done = True

            if done:
                rollout_done = True

        # ── 回合统计 ──
        tilt_rms = float(np.sqrt(np.mean(np.array(ep_tilts)**2))) if ep_tilts else 0
        swing_vel_avg = float(np.mean(ep_swing_vels)) if ep_swing_vels else 0
        swing_off_avg = float(np.mean(ep_swing_offsets)) if ep_swing_offsets else 0

        recent_rewards.append(ep_reward)
        recent_tilt_rms.append(tilt_rms)
        if len(recent_rewards) > WINDOW:
            recent_rewards.pop(0)
            recent_tilt_rms.pop(0)

        avg_r = float(np.mean(recent_rewards))
        avg_tilt = float(np.mean(recent_tilt_rms))

        print(f"Ep {episode:4d} | R:{ep_reward:7.2f}(avg:{avg_r:6.2f}) | "
              f"Steps:{ep_steps:3d} | TiltRMS:{tilt_rms:.4f} | "
              f"SwVel:{swing_vel_avg:.4f} | SwOff:{swing_off_avg:.4f} | "
              f"Wind:{wind_frac:.2f} | total:{total_steps}")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, total_steps, ep_reward, ep_steps,
                tilt_rms, swing_vel_avg, swing_off_avg,
                train_info.get("policy_loss", 0) if isinstance(train_info, dict) else 0,
                train_info.get("value_loss", 0) if isinstance(train_info, dict) else 0,
                train_info.get("entropy", 0) if isinstance(train_info, dict) else 0,
                wind_frac,
            ])

        # ── 定期评估 ──
        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_swing_controller(agent, config, n_episodes=10)
            print(f"  [Eval] TiltRMS={eval_metrics['tilt_rms']:.4f} | "
                  f"PathErr={eval_metrics['path_error_rms']:.4f} | "
                  f"SwOffset={eval_metrics['swing_offset_avg']:.4f}")

            # 保存最优（基于 tilt_rms 最小）
            metric = -eval_metrics['tilt_rms']  # 取负使其越大越好
            if metric > best_metric:
                best_metric = metric
                agent.save(os.path.join(log_dir, "ckpt_best.pt"))
                print(f"  ★ 新最佳 TiltRMS={eval_metrics['tilt_rms']:.4f}")

        # ── 定期保存 ──
        if episode > 0 and episode % SAVE_INTERVAL == 0:
            agent.save(os.path.join(log_dir, f"ckpt_ep{episode}.pt"))

        episode += 1

    # 最终保存
    agent.save(os.path.join(log_dir, "ckpt_final.pt"))
    elapsed = (time.time() - t_start) / 60
    print(f"\n[SwingTrain] 完成！总步数 {total_steps}，耗时 {elapsed:.1f} 分钟")

    return agent


# ==============================================================================
# 评估函数
# ==============================================================================

def evaluate_swing_controller(agent: SwingControllerAgent,
                               config: dict,
                               n_episodes: int = 10
                               ) -> dict:
    """
    评估底层控制器，返回 tilt_rms / path_error / swing_offset 等指标。
    """
    eval_config = copy.deepcopy(config)
    eval_config["scene"]["seed"] = 42
    eval_config["wind"]["enabled"] = True  # 评估时有风

    env = CableRobotEnvWithObstacles(config=eval_config)
    env.set_curriculum_n_obstacles(0)
    env.set_wind_curriculum(1.0)  # 全风力测试

    all_tilt_rms = []
    all_path_err = []
    all_swing_off = []

    valid_eps = 0
    attempt_cnt = 0

    while valid_eps < n_episodes and attempt_cnt < n_episodes * 3:
        attempt_cnt += 1
        obs = env.reset()
        if env.get_planned_path() is None:
            continue
        valid_eps += 1

        last_action = np.zeros(7, dtype=np.float32)
        prev_tilt = 0.0
        prev_yaw  = 0.0
        ep_tilts = []
        ep_path_errs = []
        ep_swing_offs = []

        while True:
            swing_obs, prev_tilt, prev_yaw = build_swing_obs(
                obs, env, last_action, prev_tilt, prev_yaw)
            norm_obs = agent.normalize_obs(swing_obs, update=False)
            delta_q, _, _ = agent.act(norm_obs, deterministic=True)

            next_obs, _, terminated, truncated, _ = env.step(delta_q)
            last_action = delta_q.copy()

            # 记录指标
            ep_tilts.append(float(next_obs[OBS_TILT]))

            pl_pos = np.array([next_obs[OBS_PL_X], next_obs[OBS_PL_Y],
                               next_obs[OBS_PL_Z]])
            if (env._planned_path is not None and
                    env.current_wp_idx < len(env._planned_path)):
                wp = env._planned_path[env.current_wp_idx]
                ep_path_errs.append(float(np.linalg.norm(pl_pos - wp)))

            ee_xy = next_obs[[OBS_EE_X, OBS_EE_Y]]
            pl_xy = next_obs[[OBS_PL_X, OBS_PL_Y]]
            ep_swing_offs.append(float(np.linalg.norm(ee_xy - pl_xy)))

            obs = next_obs
            if terminated or truncated:
                break

        if ep_tilts:
            all_tilt_rms.append(float(np.sqrt(np.mean(np.array(ep_tilts)**2))))
        if ep_path_errs:
            all_path_err.append(float(np.sqrt(np.mean(np.array(ep_path_errs)**2))))
        if ep_swing_offs:
            all_swing_off.append(float(np.mean(ep_swing_offs)))

    env.close()

    return {
        "tilt_rms":         float(np.mean(all_tilt_rms)) if all_tilt_rms else 0,
        "max_tilt":         float(np.max([np.max(t) for t in all_tilt_rms])) if all_tilt_rms else 0,
        "path_error_rms":   float(np.mean(all_path_err)) if all_path_err else 0,
        "swing_offset_avg": float(np.mean(all_swing_off)) if all_swing_off else 0,
        "n_valid":          valid_eps,
    }


# ==============================================================================
# 主入口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="底层防摆控制器训练")
    parser.add_argument("--log-dir",    type=str, default="saves/swing_ctrl")
    parser.add_argument("--timesteps",  type=int, default=None)
    parser.add_argument("--no-bc",      action="store_true",
                        help="跳过 BC 暖启动")
    parser.add_argument("--wind-fmax",  type=float, default=None)
    parser.add_argument("--gpu",        type=int, default=0)
    args = parser.parse_args()

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["train"]["gpu_id"] = args.gpu

    if args.timesteps:
        config["swing_controller"]["total_timesteps"] = args.timesteps
    if args.no_bc:
        config["swing_controller"]["bc_pretrain_epochs"] = 0
    if args.wind_fmax is not None:
        config["wind"]["F_max"] = args.wind_fmax

    train_swing_controller(config, args.log_dir)