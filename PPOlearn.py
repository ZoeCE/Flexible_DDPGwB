# ==============================================================================
# PPOlearn.py — 两阶段训练框架（BC 预训练 + 纯 RL）
#
# ══════════════════════════════════════════════════════════════════════════════
# 架构重设计说明
# ══════════════════════════════════════════════════════════════════════════════
#
# 旧版问题：PPO+BC 同时训练时，BC 梯度与 policy gradient 方向矛盾，
#   导致 clip_fraction 飙升、bc_loss 上升、reward 停滞。
#
# 新版方案：两阶段训练
#   Phase 1 — BC 预训练（pretrain_bc）
#     用专家执行 rollout → 收集 (obs, delta_q_expert) 对 → 纯监督学习训练 Actor
#     目标：让 Actor 学会模仿专家，达到 ~50%+ 成功率的初始策略
#     仅训练 Actor（mean_head + backbone），不训练 Critic
#     观测归一化在此阶段同步更新（warm up RunningMeanStd）
#
#   Phase 2 — 纯 PPO / TD3 fine-tune
#     关闭 BC，用预训练好的 Actor 做纯 RL 训练
#     Actor 已能完成任务 → advantage 信号清晰 → PPO 可有效优化
#
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

    for _ in range(n_episodes):
        obs = env.reset()
        if expert is not None:
            current_q = env.data.qpos[:7].copy()
            expert.reset(obs, current_q)
            planned_path = env.get_planned_path()
            if planned_path is not None:
                expert.set_path(planned_path)

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

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward":   float(np.mean(rewards)),
        "avg_steps":    float(np.mean(steps)),
    }


# ==============================================================================
# Phase 1: BC 预训练 — DAgger with β-mixing
# ==============================================================================

def pretrain_bc(agent, config: dict, n_episodes: int = 300,
                n_epochs: int = 100, batch_size: int = 256,
                lr: float = 3e-4, algo: str = "ppo"):
    """
    Phase 1: DAgger 预训练（β-混合执行策略）。

    每步以概率 β 执行专家动作、(1-β) 执行 actor 动作，但始终用专家标注。
    β 在每轮 DAgger 中逐渐降低：1.0 → 0.7 → 0.4 → 0.2 → 0.0
    这样轨迹不会过早崩溃，数据覆盖了 actor 偏移后的状态。
    """
    print(f"\n{'='*60}")
    print(f"  Phase 1: DAgger 预训练 ({algo.upper()})")
    print(f"{'='*60}")

    env = CableRobotEnvWithObstacles(config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    dq_max = np.array(config["space"].get("dq_max", [0.1]*7), dtype=np.float32)
    is_ppo = (algo == "ppo")

    all_obs = []
    all_dq  = []

    bc_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=lr)

    # DAgger 参数
    # β 调度：每轮的专家执行概率，逐步降低让 actor 接管
    BETA_SCHEDULE  = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
    EPS_PER_ROUND  = [n_episodes, 100, 100, 100, 100, 80]
    EPOCHS_SCHEDULE = [60, 40, 30, 30, 20, 20]
    TARGET_SR      = 0.5

    for rnd, beta in enumerate(BETA_SCHEDULE):
        n_eps = EPS_PER_ROUND[rnd] if rnd < len(EPS_PER_ROUND) else 80
        train_epochs = EPOCHS_SCHEDULE[rnd] if rnd < len(EPOCHS_SCHEDULE) else 20

        print(f"\n[DAgger Round {rnd}] β={beta:.1f} | "
              f"{n_eps} 回合 | {train_epochs} epochs")

        n_new = 0
        n_success = 0

        for ep in range(n_eps):
            obs = env.reset()
            current_q = env.data.qpos[:7].copy()
            expert.reset(obs, current_q, env=env)
            planned_path = env.get_planned_path()
            if planned_path is None:
                continue
            expert.set_path(planned_path)

            ep_success = False
            while True:
                norm_obs = agent.normalize_obs(obs, update=True)

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
                if terminated or truncated:
                    break

            if ep_success:
                n_success += 1

        sr = n_success / max(n_eps, 1)
        print(f"  收集完成: +{n_new} 样本（总计 {len(all_obs)}）| "
              f"执行成功率: {sr*100:.1f}%")

        # ── 训练 ─────────────────────────────────────────────────────────
        obs_arr = np.array(all_obs, dtype=np.float32)
        dq_arr  = np.array(all_dq,  dtype=np.float32)
        obs_t = torch.tensor(obs_arr, device=agent.device)
        dq_t  = torch.tensor(dq_arr,  device=agent.device)
        n_samples = len(obs_arr)

        print(f"  训练 {train_epochs} epochs on {n_samples} 样本...")

        for epoch in range(train_epochs):
            indices = np.random.permutation(n_samples)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                idx = indices[start: start + batch_size]
                obs_b = obs_t[idx]
                dq_b  = dq_t[idx]

                if is_ppo:
                    pred_dq, _, _ = agent.actor.get_action(
                        obs_b, deterministic=True)
                else:
                    pred_dq = agent.actor(obs_b)

                loss = F.mse_loss(pred_dq, dq_b)

                bc_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
                bc_optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                with torch.no_grad():
                    if is_ppo:
                        eval_dq, _, _ = agent.actor.get_action(
                            obs_t[:1000], deterministic=True)
                    else:
                        eval_dq = agent.actor(obs_t[:1000])
                    dq_mae = (eval_dq - dq_t[:1000]).abs().mean().item()
                print(f"    Epoch {epoch+1:3d} | MSE: {avg_loss:.6f} | "
                      f"MAE: {dq_mae:.4f} rad")

        # ── 评估 actor 独立成功率 ─────────────────────────────────────────
        eval_cfg = copy.deepcopy(config)
        eval_cfg["scene"]["seed"] = 42
        eval_env    = CableRobotEnvWithObstacles(config=eval_cfg)
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
# Phase 2: 纯 PPO 训练（无 BC）
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

    # ── 关闭 BC，切换到纯 PPO ─────────────────────────────────────────────
    agent.behavior_clone = False
    agent.bc_coef = 0.0
    print("[Train-PPO] BC 已关闭，进入纯 PPO fine-tune 阶段")

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
        ])

    stats    = EpisodeStats(window=SMOOTH_WIN)
    episode  = 0
    total_steps = 0
    best_sr  = 0.0
    t_start  = time.time()
    last_result = agent._last_result

    print(f"[Train-PPO] 开始纯 PPO 训练，目标总步数 {TOTAL_STEPS}...")

    while total_steps < TOTAL_STEPS:

        obs = env.reset()
        current_q = env.data.qpos[:7].copy()

        # expert 仅用于评估，不参与 rollout
        expert.reset(obs, current_q, env=env)
        planned_path = env.get_planned_path()
        if planned_path is None:
            episode += 1; continue
        expert.set_path(planned_path)

        ep_reward = 0.0; ep_steps = 0; ep_success = False
        rollout_done = False

        while not rollout_done:

            norm_obs = agent.normalize_obs(obs, update=True)
            delta_q, log_prob, value = agent.act(norm_obs, deterministic=False)

            # BC target 填零（纯 PPO 不用，但 buffer.add 需要占位）
            bc_delta_q = np.zeros(ACTION_DIM, np.float32)

            next_obs, reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated

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

        logger.log(episode, {
            "reward/episode":          ep_reward,
            f"reward/avg{SMOOTH_WIN}": avg_r,
            "steps/episode":           ep_steps,
            "env/success":             float(ep_success),
            "env/success_rate":        sr,
            "loss/policy":             last_result.policy_loss,
            "loss/value":              last_result.value_loss,
            "loss/entropy":            last_result.entropy_loss,
            "ppo/approx_kl":           last_result.approx_kl,
            "ppo/clip_fraction":       last_result.clip_fraction,
            "train/total_steps":       total_steps,
        })

        mark = "✅" if ep_success else "❌"
        print(
            f"Ep {episode:4d} {mark} | "
            f"R:{ep_reward:7.2f}(avg:{avg_r:6.2f}) | "
            f"SR:{sr*100:5.1f}% | Steps:{ep_steps:3d} | "
            f"Lp:{last_result.policy_loss:.4f} Lv:{last_result.value_loss:.4f} | "
            f"total:{total_steps}"
        )

        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, total_steps, ep_reward, avg_r,
                int(ep_success), sr, ep_steps,
                last_result.policy_loss, last_result.value_loss,
                last_result.entropy_loss,
                last_result.approx_kl, last_result.clip_fraction,
            ])

        if episode > 0 and episode % SAVE_INTERVAL == 0:
            p = save_checkpoint(agent, log_dir, episode)
            print(f"[Train] Checkpoint → {p}")

        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_cfg = copy.deepcopy(config)
            eval_cfg["scene"]["seed"] = 42
            eval_env    = CableRobotEnvWithObstacles(config=eval_cfg)
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
# Phase 2: TD3 训练（BC 预训练 + TD3 fine-tune）
# ==============================================================================

def train_td3(log_dir: str, config: dict):
    """TD3 训练（两阶段版本：先 BC 预训练，再 TD3+BC fine-tune）。"""
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

    # ── Phase 1: BC 预训练 ────────────────────────────────────────────────
    bc_cfg = config.get("bc_pretrain", {})
    bc_episodes = int(bc_cfg.get("n_episodes", 200))
    bc_epochs   = int(bc_cfg.get("n_epochs", 50))
    bc_lr       = float(bc_cfg.get("lr", 1e-3))
    bc_batch    = int(bc_cfg.get("batch_size", 256))

    pretrain_bc(agent, config,
                n_episodes=bc_episodes, n_epochs=bc_epochs,
                batch_size=bc_batch, lr=bc_lr, algo="td3")

    save_checkpoint(agent, log_dir, 0, tag="bc_pretrained")

    # ── Phase 2: TD3 训练（保留 BC loss 但 actor 已有好的初始化）────────
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
        expert.reset(obs, current_q, env=env)
        planned_path = env.get_planned_path()
        if planned_path is None:
            print(f"[Warn] Ep {episode}: 路径规划失败，跳过。")
            continue
        expert.set_path(planned_path)

        ep_reward = 0.0; ep_steps = 0; ep_success = False

        while True:
            norm_obs = agent.normalize_obs(obs, update=True)

            # 专家只调用一次
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            bc_delta_q = expert.compute_delta_q_target(obs, current_q)

            # epsilon-greedy
            if random.random() < agent.epsilon:
                delta_q = bc_delta_q.copy()
            else:
                delta_q, _ = agent.act(norm_obs)
                noise = np.random.normal(0., EXPLORE_NOISE * agent.dq_max)
                delta_q = np.clip(delta_q + noise, -agent.dq_max, agent.dq_max)

            next_obs, reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated

            ep_reward += reward
            ep_steps  += 1
            frames    += 1
            if info.get("is_success"):
                ep_success = True

            norm_next_obs = agent.normalize_obs(next_obs, update=False)
            agent.remember(norm_obs, delta_q, bc_delta_q,
                           norm_next_obs, reward, float(done))

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

        EVAL_INTERVAL = int(cfg_train.get("eval_interval", 100))
        EVAL_EPS      = int(cfg_train.get("eval_episodes", 10))
        if episode > 0 and episode % EVAL_INTERVAL == 0:
            eval_cfg = copy.deepcopy(config)
            eval_cfg["scene"]["seed"] = 42
            eval_env    = CableRobotEnvWithObstacles(config=eval_cfg)
            eval_expert = JointSpaceExpert(eval_cfg, eval_env.ik_solver)
            result = evaluate(agent, eval_env, eval_expert,
                              n_episodes=EVAL_EPS, deterministic=True, algo="td3")
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

    save_checkpoint(agent, log_dir, N_EPISODES, tag="final")
    print(f"\n[Train-TD3] 完成！耗时 {(time.time()-t_start)/60:.1f} 分钟")
    logger.close()
    return agent


# ==============================================================================
# 主入口
# ==============================================================================

def train(log_dir: str, algo: str = "ppo", custom_config: dict = None):
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