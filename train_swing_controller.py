# ==============================================================================
# train_swing_controller.py — 底层防摆 RL 控制器训练脚本（优化版）
#
# 主要改进：
#   - 奖励尺度重校准（避免全负信号淹没有效梯度）
#   - 风力课程延迟引入（前 50k 步无风）
#   - 保守探索：BC 后冻结 Actor 一段 Critic 预热期
#   - log_std 初始化更低，并早期锁死
#   - 日志增加终止原因与“稳定成功”标记（√/×）
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
# BC 暖启动 (基本不变，仅降低学习率)
# ==============================================================================

def bc_pretrain_swing(agent: SwingControllerAgent,
                      config: dict,
                      env: CableRobotEnvWithObstacles,
                      expert: JointSpaceExpert,
                      n_episodes: int = 200,
                      n_epochs: int = 10,
                      lr: float = 1e-4,   # 更保守
                      batch_size: int = 256):
    """BC 预训练底层 Actor 的 mean_head。"""
    print(f"\n{'='*60}")
    print(f"  底层控制器 BC 暖启动 ({n_episodes} 回合, {n_epochs} epochs, lr={lr})")
    print(f"{'='*60}")

    dq_max = agent.dq_max

    all_obs = []
    all_dq = []

    # 初始化 obs_norm
    print("  初始化观测归一化...")
    warm_env = CableRobotEnvWithObstacles(config=config)
    warm_env.set_curriculum_n_obstacles(0)
    for _ in range(5):
        obs = warm_env.reset()
        if warm_env.get_planned_path() is None:
            continue
        for _ in range(30):
            action = np.random.uniform(-dq_max, dq_max).astype(np.float32)
            swing_obs, _, _ = build_swing_obs(obs, warm_env, action)
            agent.normalize_obs(swing_obs, update=True)
            obs, _, term, trunc, _ = warm_env.step(action)
            if term or trunc:
                break
    warm_env.close()
    print(f"  观测归一化统计量就绪 (n={agent.obs_norm.n})")

    valid_eps = 0
    attempt_cnt = 0
    max_attempts = n_episodes * 3
    last_action = np.zeros(7, dtype=np.float32)
    prev_tilt = 0.0
    prev_yaw = 0.0

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
        prev_yaw = 0.0

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

    obs_t = torch.tensor(np.array(all_obs), device=agent.device)
    dq_t  = torch.tensor(np.array(all_dq),  device=agent.device)
    n_samples = len(all_obs)

    bc_params = [p for name, p in agent.actor.named_parameters()
                 if name != 'log_std']
    bc_optimizer = torch.optim.AdamW(bc_params, lr=lr, weight_decay=1e-5)

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = indices[start: start + batch_size]
            obs_b = obs_t[idx]
            dq_b  = dq_t[idx]

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
    os.makedirs(log_dir, exist_ok=True)
    set_global_seed(42)

    cfg_swing = config.get("swing_controller", {})
    cfg_wind  = config.get("wind", {})

    TOTAL_STEPS   = int(cfg_swing.get("total_timesteps", 1_000_000))
    N_STEPS       = int(cfg_swing.get("n_steps", 2048))
    EVAL_INTERVAL = int(cfg_swing.get("eval_interval", 20))
    SAVE_INTERVAL = int(cfg_swing.get("save_interval", 50))

    # ========== 改进 1：风力课程延迟 ==========
    WIND_CUR_START = int(cfg_wind.get("curriculum_start", 50_000))   # 原 0 → 50k
    WIND_CUR_END   = int(cfg_wind.get("curriculum_end", 300_000))    # 适当延长

    # ========== 改进 2：保守探索与冻结阶段 ==========
    LOG_STD_INIT   = -2.0    # 更小噪声（之前 -1.0）
    FREEZE_ACTOR_STEPS = 30_000   # 前 3 万步冻结 Actor（只训练 Critic）

    # ========== 改进 3：奖励系数调整（覆盖配置） ==========
    rwd_cfg = config.setdefault("swing_controller_reward", {})
    rwd_cfg["step_penalty"]            = -0.001   # 从 -0.01 大幅降低
    rwd_cfg["swing_offset_penalty_coef"] = 0.5    # 从 1.0 降低
    rwd_cfg["swing_vel_penalty_coef"]   = 0.15    # 从 0.3 降低
    rwd_cfg["waypoint_reach_bonus"]     = 2.0     # 增加正向激励
    rwd_cfg["tilt_penalty_coef"]        = 0.3      # 从 0.5 降低

    print("[SwingTrain] 初始化环境...")
    env = CableRobotEnvWithObstacles(config=config)
    env.set_curriculum_n_obstacles(0)

    print("[SwingTrain] 初始化 Agent...")
    # 覆盖 log_std_init
    cfg_swing["log_std_init"] = LOG_STD_INIT
    agent = SwingControllerAgent(config=config)

    print("[SwingTrain] 初始化专家...")
    expert = JointSpaceExpert(config, env.ik_solver)

    # ── BC 暖启动 ──
    bc_epochs = int(cfg_swing.get("bc_pretrain_epochs", 10))
    if bc_epochs > 0:
        bc_env = CableRobotEnvWithObstacles(config=config)
        bc_env.set_curriculum_n_obstacles(0)
        bc_env.set_wind_curriculum(0.0)
        bc_expert = JointSpaceExpert(config, bc_env.ik_solver)
        bc_pretrain_swing(
            agent, config, bc_env, bc_expert,
            n_episodes=200,
            n_epochs=bc_epochs,
            lr=float(cfg_swing.get("bc_pretrain_lr", 1e-4)),
        )
        bc_env.close()
        agent.save(os.path.join(log_dir, "ckpt_bc.pt"))

    # ── 冻结 Actor（均值头），只训练 Critic ──
    print(f"\n[SwingTrain] 冻结 Actor 前 {FREEZE_ACTOR_STEPS} 步，仅预热 Critic...")
    for p in agent.actor.parameters():
        p.requires_grad = False
    agent._freeze_actor = True   # 新增

    # ── PPO 训练循环 ──
    print(f"\n[SwingTrain] 开始 PPO 训练，目标步数 {TOTAL_STEPS}")
    print(f"  风力课程: {WIND_CUR_START} → {WIND_CUR_END} 步")
    print(f"  log_std_init = {LOG_STD_INIT}, 冻结 Actor 前 {FREEZE_ACTOR_STEPS} 步")

    log_file = os.path.join(log_dir, "swing_train_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "total_steps", "ep_reward", "ep_steps",
            "tilt_rms", "swing_vel_avg", "swing_offset_avg",
            "policy_loss", "value_loss", "entropy",
            "wind_frac", "termination",
        ])

    total_steps = 0
    episode = 0
    best_metric = -float('inf')
    t_start = time.time()

    recent_rewards = []
    recent_tilt_rms = []
    WINDOW = 20

    while total_steps < TOTAL_STEPS:
        # ── 风力课程 ──
        if total_steps < WIND_CUR_START:
            wind_frac = 0.0
        elif total_steps >= WIND_CUR_END:
            wind_frac = 1.0
        else:
            wind_frac = (total_steps - WIND_CUR_START) / max(WIND_CUR_END - WIND_CUR_START, 1)
        env.set_wind_curriculum(wind_frac)

        # ── 当步数超过 FREEZE_ACTOR_STEPS 后解冻 Actor ──
        if total_steps >= FREEZE_ACTOR_STEPS and not agent.actor.mean_head.weight.requires_grad:
            print(f"\n[SwingTrain] 解冻 Actor (step {total_steps})")
            for p in agent.actor.parameters():
                p.requires_grad = True
            # 解冻后保留较小的探索噪声
            agent._freeze_actor = False   # 新增
            with torch.no_grad():
                agent.actor.log_std.data.fill_(LOG_STD_INIT)

        obs = env.reset()
        planned_path = env.get_planned_path()
        if planned_path is None:
            continue

        expert.reset(obs, env.data.qpos[:7].copy(), env=env)
        expert.set_path(planned_path)

        last_action = np.zeros(7, dtype=np.float32)
        prev_tilt = 0.0
        prev_yaw  = 0.0
        prev_wp_dist = None
        ep_reward = 0.0
        ep_steps  = 0
        ep_tilts  = []
        ep_swing_vels = []
        ep_swing_offsets = []
        train_info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        termination_reason = "normal"
        rollout_done = False

        while not rollout_done:
            swing_obs, prev_tilt, prev_yaw = build_swing_obs(
                obs, env, last_action, prev_tilt, prev_yaw)
            norm_obs = agent.normalize_obs(swing_obs, update=True)

            # 动作选择
            delta_q, log_prob, value = agent.act(norm_obs, deterministic=False)

            next_obs, env_reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated

            # 计算底层奖励（使用改进后的 reward 函数）
            reward, swing_term, rwd_info = compute_swing_reward(
                env, next_obs, delta_q, last_action, prev_wp_dist, config)

            prev_wp_dist = rwd_info.get("wp_dist", None)
            done = done or swing_term
            if swing_term:
                termination_reason = rwd_info.get("termination", "swing")

            # 记录指标
            ep_tilts.append(float(next_obs[OBS_TILT]))
            ep_swing_vels.append(float(np.linalg.norm(
                next_obs[[OBS_PL_VX, OBS_PL_VY]])))
            ep_swing_offsets.append(float(np.linalg.norm(
                next_obs[[OBS_EE_X, OBS_EE_Y]] - next_obs[[OBS_PL_X, OBS_PL_Y]])))

            # 存入 buffer
            agent.buffer.add(norm_obs, delta_q, reward, float(done),
                             value, log_prob)

            ep_reward += reward
            ep_steps  += 1
            total_steps += 1
            agent.total_steps = total_steps
            last_action = delta_q.copy()
            obs = next_obs

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

        # 判断是否“稳定成功”：未被强制终止，且 tilt_rms < 0.08（可调）
        stable = (termination_reason == "normal") and (tilt_rms < 0.08)
        status_symbol = "√" if stable else "×"

        recent_rewards.append(ep_reward)
        recent_tilt_rms.append(tilt_rms)
        if len(recent_rewards) > WINDOW:
            recent_rewards.pop(0)
            recent_tilt_rms.pop(0)

        avg_r = float(np.mean(recent_rewards))
        avg_tilt = float(np.mean(recent_tilt_rms))

        print(f"Ep {episode:4d} {status_symbol} | R:{ep_reward:7.2f}(avg:{avg_r:6.2f}) | "
              f"Steps:{ep_steps:3d} | TiltRMS:{tilt_rms:.4f} | "
              f"SwVel:{swing_vel_avg:.4f} | SwOff:{swing_off_avg:.4f} | "
              f"Wind:{wind_frac:.2f} | "
              f"Term:{termination_reason[:6]:6s} | total:{total_steps}")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, total_steps, ep_reward, ep_steps,
                tilt_rms, swing_vel_avg, swing_off_avg,
                train_info.get("policy_loss", 0),
                train_info.get("value_loss", 0),
                train_info.get("entropy", 0),
                wind_frac, termination_reason,
            ])

        # ── 定期评估 ──
        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_swing_controller(agent, config, n_episodes=10)
            print(f"  [Eval] TiltRMS={eval_metrics['tilt_rms']:.4f} | "
                  f"PathErr={eval_metrics['path_error_rms']:.4f} | "
                  f"SwOffset={eval_metrics['swing_offset_avg']:.4f}")

            metric = -eval_metrics['tilt_rms']
            if metric > best_metric:
                best_metric = metric
                agent.save(os.path.join(log_dir, "ckpt_best.pt"))
                print(f"  ★ 新最佳 TiltRMS={eval_metrics['tilt_rms']:.4f}")

        if episode > 0 and episode % SAVE_INTERVAL == 0:
            agent.save(os.path.join(log_dir, f"ckpt_ep{episode}.pt"))

        episode += 1

    agent.save(os.path.join(log_dir, "ckpt_final.pt"))
    elapsed = (time.time() - t_start) / 60
    print(f"\n[SwingTrain] 完成！总步数 {total_steps}，耗时 {elapsed:.1f} 分钟")
    return agent


# ==============================================================================
# 评估函数（不变）
# ==============================================================================

def evaluate_swing_controller(agent: SwingControllerAgent,
                               config: dict,
                               n_episodes: int = 10
                               ) -> dict:
    """评估底层控制器，返回 tilt_rms / path_error / swing_offset 等指标。"""
    eval_config = copy.deepcopy(config)
    eval_config["scene"]["seed"] = 42
    eval_config["wind"]["enabled"] = True

    env = CableRobotEnvWithObstacles(config=eval_config)
    env.set_curriculum_n_obstacles(0)
    env.set_wind_curriculum(1.0)

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
# 命令行接口
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="底层防摆控制器训练（优化版）")
    parser.add_argument("--log-dir",    type=str, default="saves/swing_ctrl")
    parser.add_argument("--timesteps",  type=int, default=None)
    parser.add_argument("--no-bc",      action="store_true")
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
        config["wind"]["F_max"] = float(args.wind_fmax)

    # 默认奖励调整已在 train_swing_controller 内完成

    train_swing_controller(config, args.log_dir)