# ==============================================================================
# PPOlearn.py — 两阶段训练框架（旧版 DAgger BC 预训练 + 纯 PPO + v5 性能触发课程）
#
# ══════════════════════════════════════════════════════════════════════════════
# 合并说明
# ══════════════════════════════════════════════════════════════════════════════
#
# [MERGE-P1] 严格保留旧版所有收敛性关键机制：
#   - Phase 1: DAgger β-mixing BC 预训练（β schedule [1.0 → 0.2]，6 轮）
#   - Phase 2: 纯 PPO + BC 软锚 + v5 性能触发课程学习
#   - 课程推进条件：近 30 回合 SR ≥ 0.7 AND avg_reward ≥ 60
#   - 终止原因统计与诊断输出
#
# [MERGE-P2] 对插入任务做的唯一适配：
#   - perf_reward_threshold 可能需要调整（新任务奖励量级不同）
#     但 config 中默认保留 60.0，训练中观察 return 水平后可调
# ══════════════════════════════════════════════════════════════════════════════

import os
import csv
import copy
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from config import DEFAULT_CONFIG
from agent import PPOAgent, TD3Agent
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert


# ==============================================================================
# 工具函数
# ==============================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class EpisodeStats:
    def __init__(self, window: int = 20):
        self.window = window
        self._data: dict = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(float(v))
            if len(self._data[k]) > self.window:
                self._data[k].pop(0)

    def mean(self, key: str) -> float:
        vals = self._data.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def success_rate(self) -> float:
        return self.mean("success")


class Logger:
    def __init__(self, log_dir, project="cable_robot_ppo",
                 run_name=None, use_wandb=True, use_tb=True):
        self._wandb = None; self._writer = None
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                self._wandb.init(project=project,
                                  name=run_name or os.path.basename(log_dir),
                                  dir=log_dir, config={}, resume="allow")
                print("[Logger] wandb 初始化成功。")
            except Exception as e:
                print(f"[Logger] wandb 不可用：{e}")

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(os.path.join(log_dir, "tb"))
                print("[Logger] TensorBoard 初始化成功。")
            except Exception as e:
                print(f"[Logger] TensorBoard 不可用：{e}")

    def update_config(self, cfg):
        if self._wandb:
            flat = {}
            def _flatten(d, prefix=""):
                for k, v in d.items():
                    key = f"{prefix}{k}"
                    if isinstance(v, dict): _flatten(v, key+"/")
                    else: flat[key] = v
            _flatten(cfg); self._wandb.config.update(flat)

    def log(self, step: int, metrics: dict):
        if self._wandb: self._wandb.log(metrics, step=step)
        if self._writer:
            for k, v in metrics.items():
                self._writer.add_scalar(k, float(v), global_step=step)

    def close(self):
        if self._wandb:  self._wandb.finish()
        if self._writer: self._writer.close()


def save_checkpoint(agent, log_dir: str, episode: int, tag: str = ""):
    fname = f"ckpt_{tag}.pt" if tag else f"ckpt_ep{episode}.pt"
    path  = os.path.join(log_dir, fname)
    agent.save(path)
    agent.save(os.path.join(log_dir, "ckpt_latest.pt"))
    return path


# ==============================================================================
# 评估函数（PPO 和 TD3 通用）
# ==============================================================================

def evaluate(agent, env, expert, n_episodes: int = 10,
             deterministic: bool = True, algo: str = "ppo") -> dict:
    rewards, steps, successes = [], [], []
    is_ppo = (algo == "ppo")

    # [SKIP-INVALID] 与 test.py 一致：规划失败的场景跳过，不计入 n_episodes
    ep_count = 0
    attempt_count = 0
    max_attempts = n_episodes * 3

    while ep_count < n_episodes and attempt_count < max_attempts:
        attempt_count += 1
        obs = env.reset()
        planned_path = env.get_planned_path()
        if planned_path is None:
            continue   # 跳过失败场景

        if expert is not None:
            current_q = env.data.qpos[:7].copy()
            # [INTERFACE] 与训练/test 一致：传 env=env
            expert.reset(obs, current_q, env=env)
            expert.set_path(planned_path)

        ep_count += 1
        ep_reward = 0.0; step = 0; ep_success = False

        while True:
            if is_ppo:
                norm_obs = agent.normalize_obs(obs, update=False)
                action, _, _ = agent.act(norm_obs, deterministic=deterministic)
            else:
                norm_obs = agent.normalize_obs(obs, update=False)
                action, _ = agent.act(norm_obs)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward; step += 1
            if info.get("is_success"): ep_success = True
            if terminated or truncated: break

        rewards.append(ep_reward); steps.append(step); successes.append(float(ep_success))

    if ep_count == 0:
        # 所有 attempt 都失败，返回默认值
        return {"success_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0,
                "episodes_valid": 0, "episodes_skipped": attempt_count}
    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward":   float(np.mean(rewards)),
        "avg_steps":    float(np.mean(steps)),
        "episodes_valid":   ep_count,
        "episodes_skipped": attempt_count - ep_count,
    }


# ==============================================================================
# Phase 1: BC 预训练 — DAgger with β-mixing
# ==============================================================================

def pretrain_bc(agent, config: dict, n_episodes: int = 300,
                n_epochs: int = 100, batch_size: int = 256,
                lr: float = 3e-4, algo: str = "ppo"):
    """
    Phase 1: DAgger 预训练（β-混合执行策略）。

    [BC-FIX-2] 核心特性：
      a) obs_norm 在采集完成后冻结，避免训练/测试分布偏移
      b) 使用 bc_forward 在 u 空间做 BC，避免 tanh 饱和
      c) Round 4 崩溃修复：β>0.5 阶段收集的数据更干净，保留权重
      d) 每轮训练时同时冻结 log_std，防止策略方差失控
    """
    print(f"\n{'='*60}")
    print(f"  Phase 1: DAgger 预训练 ({algo.upper()})")
    print(f"{'='*60}")

    # [v3-CURRICULUM] BC 预训练使用 curriculum.bc_n_obstacles（默认 0）
    bc_config = copy.deepcopy(config)
    cur_cfg = config.get("curriculum", {})
    if cur_cfg.get("enabled", False):
        bc_n_obs = int(cur_cfg.get("bc_n_obstacles", 0))
        # 重要：state_dim 仍然基于 config 的 n_obstacles（上限）计算
        bc_config["scene"]["n_obstacles"] = int(config["scene"]["n_obstacles"])
        print(f"  [Curriculum] BC 阶段目标障碍物数 = {bc_n_obs} "
              f"(state_dim 上限 = {config['scene']['n_obstacles']})")
    else:
        bc_n_obs = int(config["scene"]["n_obstacles"])

    env = CableRobotEnvWithObstacles(config=bc_config)
    # 运行时切换到 BC 阶段的障碍物数
    env.set_curriculum_n_obstacles(bc_n_obs)
    expert = JointSpaceExpert(bc_config, env.ik_solver)
    dq_max = np.array(config["space"].get("dq_max", [0.1]*7), dtype=np.float32)
    is_ppo = (algo == "ppo")

    all_obs = []
    all_dq  = []

    # [BC-FIX-2a] BC 只训练 mean_head + backbone，不训练 log_std
    bc_params = []
    if is_ppo:
        for name, p in agent.actor.named_parameters():
            if name != 'log_std':
                bc_params.append(p)
    else:
        bc_params = list(agent.actor.parameters())
    bc_optimizer = torch.optim.Adam(bc_params, lr=lr)

    # DAgger 参数
    # [BC-FIX-3] 更保守的 β 调度：不降到 0，保留少量专家干预
    BETA_SCHEDULE   = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2]
    EPS_PER_ROUND   = [n_episodes, 120, 120, 120, 100, 100]
    EPOCHS_SCHEDULE = [80, 50, 40, 30, 25, 20]
    TARGET_SR       = 0.6

    # [BC-FIX-4] 每轮数据最大容量上限（防止旧数据占比过高）
    MAX_BUFFER_SIZE = 250000

    for rnd, beta in enumerate(BETA_SCHEDULE):
        n_eps = EPS_PER_ROUND[rnd] if rnd < len(EPS_PER_ROUND) else 80
        train_epochs = EPOCHS_SCHEDULE[rnd] if rnd < len(EPOCHS_SCHEDULE) else 20

        print(f"\n[DAgger Round {rnd}] β={beta:.1f} | "
              f"{n_eps} 回合 | {train_epochs} epochs")

        n_new = 0
        n_success = 0

        # [BC-FIX-5] obs_norm update 策略
        update_norm_fraction = 1.0 if rnd == 0 else 0.3

        # [SKIP-INVALID] while 循环：失败场景不占用 n_eps 配额
        valid_eps = 0
        attempt_cnt = 0
        max_attempts = n_eps * 3
        while valid_eps < n_eps and attempt_cnt < max_attempts:
            attempt_cnt += 1
            obs = env.reset()
            current_q = env.data.qpos[:7].copy()
            planned_path = env.get_planned_path()
            if planned_path is None:
                continue
            expert.reset(obs, current_q, env=env)
            expert.set_path(planned_path)
            valid_eps += 1

            ep_success = False
            step_in_ep = 0
            while True:
                do_update = (random.random() < update_norm_fraction)
                norm_obs = agent.normalize_obs(obs, update=do_update)

                # 专家标注当前状态
                current_q = env.data.qpos[:7].copy().astype(np.float32)
                bc_dq = expert.compute_delta_q_target(obs, current_q)

                if not np.any(np.isnan(bc_dq)):
                    all_obs.append(norm_obs.copy())
                    all_dq.append(bc_dq.copy())
                    n_new += 1

                # β-混合：以概率 β 用专家动作，否则用 actor 动作
                if random.random() < beta:
                    action = bc_dq
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(norm_obs.reshape(1, -1),
                                           dtype=torch.float32,
                                           device=agent.device)
                        if is_ppo:
                            action_t, _, _ = agent.actor.get_action(
                                s_t, deterministic=True)
                        else:
                            action_t = agent.actor(s_t)
                        action = action_t.cpu().numpy().flatten()
                        action = np.clip(action, -dq_max, dq_max)

                next_obs, reward, terminated, truncated, info = env.step(action)
                if info.get("is_success"):
                    ep_success = True
                obs = next_obs
                step_in_ep += 1
                if terminated or truncated:
                    break

            if ep_success:
                n_success += 1

        sr = n_success / max(valid_eps, 1)
        skipped = attempt_cnt - valid_eps
        skip_note = f"，跳过 {skipped} 个无效场景" if skipped > 0 else ""
        print(f"  收集完成: +{n_new} 样本（总计 {len(all_obs)}）| "
              f"执行成功率: {sr*100:.1f}%{skip_note}")

        # [BC-FIX-4] 缓冲区裁剪
        if len(all_obs) > MAX_BUFFER_SIZE:
            excess = len(all_obs) - MAX_BUFFER_SIZE
            all_obs = all_obs[excess:]
            all_dq  = all_dq[excess:]
            print(f"  缓冲区裁剪至 {MAX_BUFFER_SIZE} 样本")

        # ── 训练 ─────────────────────────────────────────────────────────
        obs_arr = np.array(all_obs, dtype=np.float32)
        dq_arr  = np.array(all_dq,  dtype=np.float32)
        obs_t = torch.tensor(obs_arr, device=agent.device)
        dq_t  = torch.tensor(dq_arr,  device=agent.device)
        n_samples = len(obs_arr)

        print(f"  训练 {train_epochs} epochs on {n_samples} 样本...")

        for epoch in range(train_epochs):
            indices = np.random.permutation(n_samples)
            total_loss_u  = 0.0
            total_loss_dq = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = indices[start: start + batch_size]
                obs_b = obs_t[idx]
                dq_b  = dq_t[idx]

                if is_ppo:
                    # [BC-FIX-1] 使用 bc_forward，主 loss 在 u 空间
                    bc_loss_u, bc_loss_dq, _ = agent.actor.bc_forward(obs_b, dq_b)
                    # 总 loss：u 空间主导（0.7），Δq 空间辅助（0.3）
                    loss = 0.7 * bc_loss_u + 0.3 * bc_loss_dq
                    total_loss_u  += bc_loss_u.item()
                    total_loss_dq += bc_loss_dq.item()
                else:
                    pred_dq = agent.actor(obs_b)
                    loss = F.mse_loss(pred_dq, dq_b)
                    total_loss_dq += loss.item()

                bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(bc_params, 1.0)
                bc_optimizer.step()

                n_batches += 1

            avg_u  = total_loss_u  / max(n_batches, 1)
            avg_dq = total_loss_dq / max(n_batches, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                with torch.no_grad():
                    if is_ppo:
                        eval_dq, _, _ = agent.actor.get_action(
                            obs_t[:1000], deterministic=True)
                    else:
                        eval_dq = agent.actor(obs_t[:1000])
                    dq_mae = (eval_dq - dq_t[:1000]).abs().mean().item()
                if is_ppo:
                    print(f"    Epoch {epoch+1:3d} | U-MSE: {avg_u:.6f} | "
                          f"DQ-MSE: {avg_dq:.6f} | MAE: {dq_mae:.4f} rad")
                else:
                    print(f"    Epoch {epoch+1:3d} | MSE: {avg_dq:.6f} | "
                          f"MAE: {dq_mae:.4f} rad")

        # [BC-FIX-7] 每轮训练后，将 log_std 重置到配置初值
        if is_ppo:
            with torch.no_grad():
                init_val = float(agent.config["ppo_agent"].get("log_std_init", -1.0))
                agent.actor.log_std.data.fill_(init_val)

        # ── 评估 actor 独立成功率 ─────────────────────────────────────────
        eval_cfg = copy.deepcopy(config)
        eval_cfg["scene"]["seed"] = 42
        eval_env    = CableRobotEnvWithObstacles(config=eval_cfg)
        eval_env.set_curriculum_n_obstacles(bc_n_obs)
        eval_expert = JointSpaceExpert(eval_cfg, eval_env.ik_solver)
        result = evaluate(agent, eval_env, eval_expert,
                          n_episodes=20, deterministic=True, algo=algo)
        eval_env.close()

        print(f"  [Eval] Actor 独立 SR={result['success_rate']*100:.1f}% | "
              f"AvgR={result['avg_reward']:.2f} | "
              f"AvgSteps={result['avg_steps']:.1f}")

        if result["success_rate"] >= TARGET_SR:
            print(f"  ✅ 达到目标 SR {TARGET_SR*100:.0f}%，DAgger 提前退出")
            break

    env.close()
    print(f"{'='*60}\n")
    return result


# ==============================================================================
# Phase 2: 纯 PPO 训练（v5 性能触发课程）
# ==============================================================================

def train_ppo(log_dir: str, config: dict):
    """纯 PPO 训练（两阶段版本：先 BC 预训练，再纯 PPO fine-tune）。"""
    cfg_train  = config["train"]
    cfg_ppo    = config["ppo_agent"]

    TOTAL_STEPS   = int(cfg_train.get("total_timesteps", 5_000_000))
    N_STEPS       = int(cfg_ppo["n_steps"])
    SAVE_INTERVAL = int(cfg_train["save_interval"])
    EVAL_INTERVAL = int(cfg_train.get("eval_interval", 100))
    EVAL_EPS      = int(cfg_train.get("eval_episodes", 10))
    SMOOTH_WIN    = int(cfg_train["log_smooth_win"])
    ACTION_DIM    = int(config["space"]["action_dim"])

    print("[Train-PPO] 初始化环境...")
    env = CableRobotEnvWithObstacles(config=config)
    STATE_DIM  = env.state_dim
    print(f"  state_dim={STATE_DIM}, action_dim={ACTION_DIM}")

    print("[Train-PPO] 初始化 Agent...")
    agent = PPOAgent(log_dir, STATE_DIM, ACTION_DIM, config=config)

    # ── Phase 1: BC 预训练 ────────────────────────────────────────────────
    bc_cfg = config.get("bc_pretrain", {})
    bc_episodes = int(bc_cfg.get("n_episodes", 200))
    bc_epochs   = int(bc_cfg.get("n_epochs", 50))
    bc_lr       = float(bc_cfg.get("lr", 1e-3))
    bc_batch    = int(bc_cfg.get("batch_size", 256))

    pretrain_bc(agent, config,
                n_episodes=bc_episodes, n_epochs=bc_epochs,
                batch_size=bc_batch, lr=bc_lr, algo="ppo")

    # BC 预训练后保存
    save_checkpoint(agent, log_dir, 0, tag="bc_pretrained")

    # [BC-FIX-8 v4] BC 软锚衰减策略
    # 早期（PPO stable 阶段）保持较强 BC 引导，进入课程后快速衰减
    agent.behavior_clone = True
    agent.bc_coef = float(cfg_ppo.get("bc_coef_final", 0.3))   # 起始软锚 0.3
    agent.bc_coef_init = agent.bc_coef
    agent.bc_coef_final = 0.02        # 长期衰减到接近 0
    agent.bc_anneal_steps = 1_500_000  # 1.5M 步内完成退火
    print(f"[Train-PPO] BC 软锚点开启 (coef={agent.bc_coef}，"
          f"退火至 {agent.bc_coef_final} 经过 {agent.bc_anneal_steps} 步) → 进入 PPO 阶段")

    # ── Phase 2: 纯 PPO 训练 ─────────────────────────────────────────────
    expert = JointSpaceExpert(config, env.ik_solver)

    logger = Logger(log_dir, "cable_robot_ppo", os.path.basename(log_dir))
    logger.update_config(config)

    log_file = os.path.join(log_dir, "ppo_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "total_steps", "episode_reward", "avg_reward",
            "success", "success_rate", "steps",
            "policy_loss", "value_loss", "entropy_loss",
            "approx_kl", "clip_fraction",
            "curr_n_obs",   # [v3-CURRICULUM] 记录当前课程难度
        ])

    # ── [v5-CURRICULUM] 基于性能的课程调度器 ────────────────────────────────
    cur_cfg = config.get("curriculum", {})
    use_curriculum = cur_cfg.get("enabled", False)
    max_n_obs      = int(cur_cfg.get("ppo_max_n_obstacles", config["scene"]["n_obstacles"]))
    ramp_mode      = str(cur_cfg.get("ramp_mode", "performance"))
    milestones     = cur_cfg.get("milestones", None)

    # 性能触发参数（v5）
    perf_window         = int(cur_cfg.get("perf_window", 30))
    perf_sr_thresh      = float(cur_cfg.get("perf_sr_threshold", 0.7))
    perf_reward_thresh  = float(cur_cfg.get("perf_reward_threshold", 60.0))
    perf_min_episodes   = int(cur_cfg.get("perf_min_episodes_per_level", 100))
    perf_regression_tol = float(cur_cfg.get("perf_regression_tol", -30.0))
    perf_hard_cap_steps = int(cur_cfg.get("perf_hard_cap_steps", 600_000))

    # 课程状态
    class CurriculumState:
        def __init__(self):
            self.level = 0
            self.level_entered_episode = 0
            self.level_entered_step    = 0
            self.promotions = 0
            self.demotions  = 0
    cur_state = CurriculumState()

    # 兼容旧 step/linear 模式参数
    stable_n_obs   = int(cur_cfg.get("ppo_stable_n_obstacles", 0))
    stable_until   = int(cur_cfg.get("ppo_stable_timesteps", 500_000))
    ramp_start     = int(cur_cfg.get("ppo_ramp_start_timesteps", 500_000))
    ramp_end       = int(cur_cfg.get("ppo_ramp_end_timesteps", 2_000_000))
    ramp_step_size = int(cur_cfg.get("ramp_step_size", 1))

    def compute_curriculum_n_obs_by_timestep(tstep: int) -> int:
        """时间触发的课程 (milestone / step / linear 模式)。"""
        if ramp_mode == "milestone" and milestones is not None:
            n = 0
            for thresh, n_val in milestones:
                if tstep >= thresh:
                    n = n_val
            return int(n)
        if tstep < stable_until:
            return stable_n_obs
        if tstep >= ramp_end:
            return max_n_obs
        frac = (tstep - ramp_start) / max(ramp_end - ramp_start, 1)
        frac = max(0.0, min(1.0, frac))
        if ramp_mode == "linear":
            n = stable_n_obs + int(round(frac * (max_n_obs - stable_n_obs)))
        else:  # step
            n_steps = max(1, (max_n_obs - stable_n_obs) // max(ramp_step_size, 1))
            step_idx = int(frac * n_steps)
            n = stable_n_obs + step_idx * ramp_step_size
            n = min(n, max_n_obs)
        return int(n)

    def compute_curriculum_by_performance(
            cur_level: int,
            recent_sr: float, recent_reward: float,
            episodes_at_level: int, steps_at_level: int) -> int:
        """基于性能的课程推进。
        规则：
          1. 若 (SR 达标 AND reward 达标) 且最少 episodes 已达成 → 推进
          2. 若单级停留步数超过 hard_cap → 强制推进（保险）
        """
        if cur_level >= max_n_obs:
            return cur_level
        if steps_at_level >= perf_hard_cap_steps:
            return cur_level + 1
        if episodes_at_level < perf_min_episodes:
            return cur_level
        if recent_sr >= perf_sr_thresh and recent_reward >= perf_reward_thresh:
            return cur_level + 1
        return cur_level

    def compute_curriculum_n_obs(tstep: int, episode: int,
                                 recent_sr: float = 0.0,
                                 recent_reward: float = 0.0) -> int:
        """综合调度：performance 模式优先，其余回退到 timestep 模式。"""
        if not use_curriculum:
            return int(config["scene"]["n_obstacles"])

        if ramp_mode == "performance":
            eps_at_level = episode - cur_state.level_entered_episode
            steps_at_level = tstep - cur_state.level_entered_step
            return compute_curriculum_by_performance(
                cur_state.level, recent_sr, recent_reward,
                eps_at_level, steps_at_level)
        else:
            return compute_curriculum_n_obs_by_timestep(tstep)

    # 初始化当前课程障碍物数
    current_n_obs = compute_curriculum_n_obs(0, 0, 0.0, 0.0)
    cur_state.level = current_n_obs
    env.set_curriculum_n_obstacles(current_n_obs)
    if use_curriculum:
        if ramp_mode == "performance":
            print(f"[Curriculum v5] Phase 2 开始（performance 模式）:")
            print(f"  触发条件: 近 {perf_window} 回合 SR ≥ {perf_sr_thresh*100:.0f}% "
                  f"且 avg_reward ≥ {perf_reward_thresh}")
            print(f"  每级最少 {perf_min_episodes} 回合，hard_cap = {perf_hard_cap_steps} 步")
            print(f"  最大障碍物数 = {max_n_obs}")
            print(f"  当前 n_obs = {current_n_obs}")
        elif ramp_mode == "milestone":
            print(f"[Curriculum] Phase 2 开始（milestone 模式）:")
            for thresh, n in (milestones or []):
                print(f"  t ≥ {thresh:>8d}: n_obs = {n}")
            print(f"  当前 n_obs = {current_n_obs}")
        else:
            print(f"[Curriculum] Phase 2 开始（{ramp_mode} 模式）：n_obs={current_n_obs}")
        _ = env.reset()
        actual = len(env._obstacles)
        print(f"[Diag] PPO 主 env 首次 reset 后: config.n_obstacles={env.config['scene']['n_obstacles']}, "
              f"实际 len(_obstacles)={actual}")
        if actual != current_n_obs:
            print(f"[WARN] 实际障碍物数与期望不符！检查 set_curriculum_n_obstacles 是否生效")

    stats    = EpisodeStats(window=SMOOTH_WIN)
    cur_stats = EpisodeStats(window=perf_window) if use_curriculum else None
    episode  = 0
    total_steps = 0
    best_sr  = 0.0
    t_start  = time.time()
    last_result = agent._last_result

    # [v3-DIAG] 终止原因统计
    from collections import Counter, deque
    DIAG_WINDOW = 50
    DIAG_INTERVAL = 20
    term_history = deque(maxlen=DIAG_WINDOW)
    last_term_reason = None

    print(f"[Train-PPO] 开始纯 PPO 训练，目标总步数 {TOTAL_STEPS}...")

    while total_steps < TOTAL_STEPS:

        # [v5-CURRICULUM] 每个 episode 检查是否需要调整难度
        if use_curriculum and cur_stats is not None:
            recent_sr = cur_stats.success_rate()
            recent_reward = cur_stats.mean("reward")
        else:
            recent_sr = 0.0
            recent_reward = 0.0

        new_n_obs = compute_curriculum_n_obs(total_steps, episode, recent_sr, recent_reward)

        if new_n_obs != current_n_obs:
            env.set_curriculum_n_obstacles(new_n_obs)
            prev_level = cur_state.level
            cur_state.level = new_n_obs
            cur_state.level_entered_episode = episode
            cur_state.level_entered_step    = total_steps
            if new_n_obs > prev_level:
                cur_state.promotions += 1
            else:
                cur_state.demotions += 1
            if cur_stats is not None:
                cur_stats = EpisodeStats(window=perf_window)

            print(f"\n{'='*70}")
            if new_n_obs > prev_level:
                print(f"  [CURRICULUM PROMOTE] t={total_steps} ep={episode} "
                      f"SR={recent_sr*100:.0f}% avgR={recent_reward:.1f} "
                      f"→ n_obstacles: {prev_level} → {new_n_obs}")
            else:
                print(f"  [CURRICULUM DEMOTE] t={total_steps} ep={episode} "
                      f"SR={recent_sr*100:.0f}% avgR={recent_reward:.1f} "
                      f"→ n_obstacles: {prev_level} → {new_n_obs}")
            print(f"{'='*70}\n")
            current_n_obs = new_n_obs

            logger.log(episode, {
                "curriculum/switch_event": float(new_n_obs),
                "curriculum/switch_total_steps": total_steps,
                "curriculum/promotions": cur_state.promotions,
                "curriculum/demotions":  cur_state.demotions,
            })

        obs = env.reset()
        current_q = env.data.qpos[:7].copy()

        # [SKIP-INVALID] 失败场景不计入 episode，直接重试
        planned_path = env.get_planned_path()
        if planned_path is None:
            # 不增加 episode，直接重新尝试下一场景
            continue
        # expert 用于 BC 监督，不参与 rollout 主策略
        expert.reset(obs, current_q, env=env)
        expert.set_path(planned_path)

        ep_reward = 0.0; ep_steps = 0; ep_success = False
        rollout_done = False

        while not rollout_done:

            norm_obs = agent.normalize_obs(obs, update=True)

            # 采集专家 BC 目标（软锚点监督信号）
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            try:
                bc_delta_q = expert.compute_delta_q_target(obs, current_q)
                if np.any(np.isnan(bc_delta_q)):
                    bc_delta_q = np.zeros(ACTION_DIM, np.float32)
            except Exception:
                bc_delta_q = np.zeros(ACTION_DIM, np.float32)

            delta_q, log_prob, value = agent.act(norm_obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated

            # [v3-DIAG] 捕获终止原因
            if done:
                last_term_reason = info.get("termination_reason", "unknown")

            ep_reward += reward
            ep_steps  += 1
            total_steps += 1
            agent.total_steps = total_steps
            if info.get("is_success"):
                ep_success = True

            agent.buffer.add(norm_obs, delta_q, bc_delta_q,
                             reward, float(done), value, log_prob)
            obs = next_obs

            if agent.buffer.full:
                if done:
                    last_val = 0.0
                else:
                    with torch.no_grad():
                        ns_norm = agent.normalize_obs(next_obs, update=False)
                        s_t = torch.tensor(ns_norm, dtype=torch.float32,
                                           device=agent.device).unsqueeze(0)
                        last_val = agent.critic(s_t).item()

                agent.buffer.compute_returns_and_advantages(
                    last_val, agent.gamma, agent.gae_lambda
                )
                last_result = agent.update()
                rollout_done = True

            if done:
                rollout_done = True

        # ── 回合统计 ──────────────────────────────────────────────────
        stats.update(reward=ep_reward, steps=ep_steps, success=float(ep_success))
        avg_r = stats.mean("reward"); sr = stats.success_rate()
        if cur_stats is not None:
            cur_stats.update(reward=ep_reward, success=float(ep_success))

        # [v3-DIAG] 记录本回合终止原因
        reason_key = (last_term_reason or "unknown").split(":")[0]
        term_history.append(reason_key)

        logger.log(episode, {
            "reward/episode":          ep_reward,
            f"reward/avg{SMOOTH_WIN}": avg_r,
            "steps/episode":           ep_steps,
            "env/success":             float(ep_success),
            "env/success_rate":        sr,
            "loss/policy":             last_result.policy_loss,
            "loss/value":              last_result.value_loss,
            "loss/entropy":            last_result.entropy_loss,
            "loss/bc":                 last_result.bc_loss,
            "ppo/approx_kl":           last_result.approx_kl,
            "ppo/clip_fraction":       last_result.clip_fraction,
            "ppo/bc_coef":             agent.bc_coef,
            "train/total_steps":       total_steps,
            "curriculum/current_n_obs":   float(current_n_obs),
            "curriculum/level":           float(cur_state.level),
            "curriculum/recent_sr":       float(recent_sr),
            "curriculum/recent_reward":   float(recent_reward),
        })

        mark = "✅" if ep_success else "❌"
        print(
            f"Ep {episode:4d} {mark} | "
            f"R:{ep_reward:7.2f}(avg:{avg_r:6.2f}) | "
            f"SR:{sr*100:5.1f}% | Steps:{ep_steps:3d} | "
            f"bc:{agent.bc_coef:.3f} | n_obs:{current_n_obs} | "
            f"Lp:{last_result.policy_loss:.4f} Lv:{last_result.value_loss:.4f} | "
            f"total:{total_steps} | term:{reason_key}"
        )

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, total_steps, ep_reward, avg_r,
                int(ep_success), sr, ep_steps,
                last_result.policy_loss, last_result.value_loss,
                last_result.entropy_loss,
                last_result.approx_kl, last_result.clip_fraction,
                current_n_obs,
            ])

        # 诊断：定期打印终止原因统计
        if episode > 0 and episode % DIAG_INTERVAL == 0 and len(term_history) > 0:
            counter = Counter(term_history)
            top = counter.most_common(5)
            print(f"  [DIAG] 近 {len(term_history)} 回合终止原因 top5: "
                  + ", ".join(f"{k}:{v}" for k, v in top))

        if episode > 0 and episode % SAVE_INTERVAL == 0:
            p = save_checkpoint(agent, log_dir, episode)
            print(f"[Train] Checkpoint → {p}")

        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_cfg = copy.deepcopy(config)
            eval_cfg["scene"]["seed"] = 42
            eval_env    = CableRobotEnvWithObstacles(config=eval_cfg)
            # 评估时按当前课程难度
            eval_env.set_curriculum_n_obstacles(current_n_obs)
            eval_expert = JointSpaceExpert(eval_cfg, eval_env.ik_solver)
            result = evaluate(agent, eval_env, eval_expert,
                              n_episodes=EVAL_EPS, deterministic=True, algo="ppo")
            eval_env.close()

            print(f"  [Eval] SR={result['success_rate']*100:.1f}% | "
                  f"AvgR={result['avg_reward']:.2f} | "
                  f"AvgSteps={result['avg_steps']:.1f}")

            logger.log(episode, {
                "eval/success_rate": result["success_rate"],
                "eval/avg_reward":   result["avg_reward"],
                "eval/avg_steps":    result["avg_steps"],
            })

            if result["success_rate"] > best_sr:
                best_sr = result["success_rate"]
                save_checkpoint(agent, log_dir, episode, tag="best")
                print(f"  [Eval] 新最佳 SR: {best_sr*100:.1f}%")

        episode += 1

    save_checkpoint(agent, log_dir, episode, tag="final")
    elapsed = (time.time() - t_start) / 60
    print(f"\n[Train-PPO] 完成！总步数 {total_steps}，耗时 {elapsed:.1f} 分钟")
    logger.close()
    return agent


# ==============================================================================
# TD3 训练主循环（保留）
# ==============================================================================

def train_td3(log_dir: str, config: dict):
    """TD3 + BC 训练循环（关节空间版本）。"""
    cfg_train  = config["train"]
    cfg_agent  = config["td3_agent"]

    N_EPISODES      = int(cfg_train["n_episodes"])
    WARMUP_EPISODES = int(cfg_train["warmup_episodes"])
    MIN_BUFFER      = int(cfg_train["min_buffer_to_train"])
    GRAD_UPDATES    = int(cfg_train["grad_updates_per_step"])
    SAVE_INTERVAL   = int(cfg_train["save_interval"])
    SMOOTH_WIN      = int(cfg_train["log_smooth_win"])
    EXPLORE_NOISE   = float(cfg_train["explore_noise"])
    ACTION_DIM      = int(config["space"]["action_dim"])

    print("[Train-TD3] 初始化环境...")
    env = CableRobotEnvWithObstacles(config=config)
    STATE_DIM = env.state_dim

    print("[Train-TD3] 初始化 Agent...")
    agent = TD3Agent(log_dir, STATE_DIM, ACTION_DIM, config=config)

    print("[Train-TD3] 初始化专家控制器...")
    expert = JointSpaceExpert(config, env.ik_solver)

    logger = Logger(log_dir, "cable_robot_td3", os.path.basename(log_dir))
    logger.update_config(config)

    log_file = os.path.join(log_dir, "td3_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "frames", "ep_reward", "avg_reward",
            "success", "success_rate", "steps",
            "loss_c", "loss_a", "loss_bc", "q_pred", "q_target",
            "epsilon",
        ])

    stats       = EpisodeStats(window=SMOOTH_WIN)
    frames      = 0
    best_sr     = 0.0
    last_result = None
    t_start     = time.time()

    print(f"[Train-TD3] 开始训练 {N_EPISODES} 回合...")

    for episode in range(N_EPISODES):

        if episode < WARMUP_EPISODES:
            agent.epsilon = 1.0

        obs = env.reset()
        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q)
        planned_path = env.get_planned_path()
        if planned_path is None:
            print(f"[Warn] Ep {episode}: 路径规划失败，跳过。")
            continue
        expert.set_path(planned_path)

        ep_reward = 0.0; ep_steps = 0; ep_success = False

        while True:
            norm_obs = agent.normalize_obs(obs, update=True)

            current_q  = env.data.qpos[:7].copy().astype(np.float32)

            # epsilon-greedy：专家也输出 delta_q
            if random.random() < agent.epsilon:
                current_q = env.data.qpos[:7].copy().astype(np.float32)
                action_expert = expert.compute_joint_target(obs, current_q)
                delta_q = np.clip(action_expert - current_q, -agent.dq_max, agent.dq_max)
            else:
                delta_q, _ = agent.act(norm_obs)
                noise = np.random.normal(0., EXPLORE_NOISE * agent.dq_max)
                delta_q = np.clip(delta_q + noise, -agent.dq_max, agent.dq_max)

            # BC 目标
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            bc_delta_q = expert.compute_delta_q_target(obs, current_q)

            next_obs, reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated

            ep_reward += reward
            ep_steps  += 1
            frames    += 1
            if info.get("is_success"):
                ep_success = True

            norm_obs      = agent.normalize_obs(obs, update=True)
            norm_next_obs = agent.normalize_obs(next_obs, update=False)
            agent.remember(norm_obs, delta_q, bc_delta_q, norm_next_obs, reward, float(done))

            if agent.buffer.size > MIN_BUFFER:
                last_result = agent.train(GRAD_UPDATES)

            if episode >= WARMUP_EPISODES:
                agent.step_epsilon()

            obs = next_obs
            if done: break

        stats.update(reward=ep_reward, steps=ep_steps, success=float(ep_success))
        avg_r = stats.mean("reward"); sr = stats.success_rate()

        lc  = last_result.critic_loss if last_result else 0.0
        la  = last_result.actor_loss  if last_result else 0.0
        lbc = last_result.bc_loss     if last_result else 0.0
        qp  = last_result.q_pred      if last_result else 0.0
        qt  = last_result.q_target    if last_result else 0.0

        logger.log(episode, {
            "reward/episode": ep_reward, f"reward/avg{SMOOTH_WIN}": avg_r,
            "steps/episode": ep_steps, "env/success": float(ep_success),
            "env/success_rate": sr, "loss/critic": lc, "loss/actor": la,
            "loss/bc": lbc, "Q/pred": qp, "Q/target": qt,
            "explore/epsilon": agent.epsilon, "train/frames": frames,
        })

        mark = "✅" if ep_success else "❌"
        print(f"Ep {episode:4d} {mark} | R:{ep_reward:7.2f}(avg:{avg_r:6.2f}) | "
              f"SR:{sr*100:5.1f}% | Steps:{ep_steps:3d} | ε:{agent.epsilon:.4f} | "
              f"Lc:{lc:.4f} | Q:{qp:.2f}/{qt:.2f}")

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, frames, ep_reward, avg_r,
                int(ep_success), sr, ep_steps,
                lc, la, lbc, qp, qt, agent.epsilon,
            ])

        if episode > 0 and episode % SAVE_INTERVAL == 0:
            p = save_checkpoint(agent, log_dir, episode)
            print(f"[Train] Checkpoint → {p}")

    save_checkpoint(agent, log_dir, N_EPISODES, tag="final")
    print(f"\n[Train-TD3] 完成！耗时 {(time.time()-t_start)/60:.1f} 分钟")
    logger.close()
    return agent


# ==============================================================================
# 主入口
# ==============================================================================

def train(log_dir: str, algo: str = "ppo", custom_config: dict = None):
    """
    统一训练入口。

    Args:
        log_dir:       模型和日志保存目录
        algo:          "ppo" 或 "td3"
        custom_config: 局部配置覆盖
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config:
        for key, val in custom_config.items():
            if isinstance(val, dict) and key in config:
                config[key].update(val)
            else:
                config[key] = val

    set_global_seed(42)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[Train] 算法: {algo.upper()} | 保存目录: {log_dir}")

    if algo == "ppo":
        return train_ppo(log_dir, config)
    elif algo == "td3":
        return train_td3(log_dir, config)
    else:
        raise ValueError(f"未知算法: {algo}，请使用 'ppo' 或 'td3'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="索驱动机器人 PPO/TD3 训练")
    parser.add_argument("--algo",     type=str, default="ppo",
                        choices=["ppo", "td3"], help="训练算法")
    parser.add_argument("--log-dir",  type=str, default="saves/ppo_run")
    parser.add_argument("--episodes", type=int, default=None,
                        help="覆盖 n_episodes（TD3 用）")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="覆盖 total_timesteps（PPO 用）")
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--gpu",      type=int, default=0)
    args = parser.parse_args()

    cli_cfg = {}
    if args.render:    cli_cfg.setdefault("sim",   {})["render"]          = True
    if args.gpu != 0:  cli_cfg.setdefault("train", {})["gpu_id"]          = args.gpu
    if args.episodes:  cli_cfg.setdefault("train", {})["n_episodes"]       = args.episodes
    if args.timesteps: cli_cfg.setdefault("train", {})["total_timesteps"]  = args.timesteps

    train(args.log_dir, algo=args.algo, custom_config=cli_cfg or None)