# ==============================================================================
# PPOlearn.py — PPO + BC 训练主程序
#
# 训练框架说明：
#
# [LOOP-1] 核心数据流
#   每个 n_steps 步的 rollout：
#     for each env step:
#       1. obs → PPOAgent.act() → 7D 关节角动作（含 log_prob, value）
#       2. JointSpaceExpert.compute_joint_target(obs, current_q) → BC 目标关节角
#       3. env.step(action) → next_obs, reward, done
#       4. buffer.add(norm_obs, action, bc_target, reward, done, value, log_prob)
#
#   rollout 结束后：
#     buffer.compute_returns_and_advantages(last_value)
#     agent.update() → PPO + BC 联合梯度更新
#
# [LOOP-2] PPO vs TD3 框架切换
#   config["train"]["algo"] = "ppo" | "td3"
#   运行时通过 --algo 命令行参数覆盖
#
# [LOOP-3] BC 系数退火逻辑
#   - PPO Agent 内部维护 bc_coef（随 total_steps 线性退火）
#   - 训练早期（total_steps < bc_anneal_steps / 2）：BC 主导，快速逼近专家轨迹
#   - 训练后期：RL 主导，在专家基础上自主优化
#
# [LOOP-4] 评估机制
#   每 eval_interval 回合：在固定 seed 场景下运行 eval_episodes 回合，
#   统计成功率和平均回报，保存最优 checkpoint
#
# [LOOP-5] 日志
#   wandb + TensorBoard + CSV 三路写入
# ==============================================================================

# ==============================================================================
# PPOlearn.py — PPO + BC 训练主程序
#
# 训练框架说明：
#
# [LOOP-1] 核心数据流
#   每个 n_steps 步的 rollout：
#     for each env step:
#       1. obs → PPOAgent.act() → 7D 关节角动作（含 log_prob, value）
#       2. JointSpaceExpert.compute_joint_target(obs, current_q) → BC 目标关节角
#       3. env.step(action) → next_obs, reward, done
#       4. buffer.add(norm_obs, action, bc_target, reward, done, value, log_prob)
#
#   rollout 结束后：
#     buffer.compute_returns_and_advantages(last_value)
#     agent.update() → PPO + BC 联合梯度更新
#
# [LOOP-2] PPO vs TD3 框架切换
#   config["train"]["algo"] = "ppo" | "td3"
#   运行时通过 --algo 命令行参数覆盖
#
# [LOOP-3] BC 系数退火逻辑
#   - PPO Agent 内部维护 bc_coef（随 total_steps 线性退火）
#   - 训练早期（total_steps < bc_anneal_steps / 2）：BC 主导，快速逼近专家轨迹
#   - 训练后期：RL 主导，在专家基础上自主优化
#
# [LOOP-4] 评估机制
#   每 eval_interval 回合：在固定 seed 场景下运行 eval_episodes 回合，
#   统计成功率和平均回报，保存最优 checkpoint
#
# [LOOP-5] 日志
#   wandb + TensorBoard + CSV 三路写入
# ==============================================================================

import os
import csv
import copy
import time
import random
import numpy as np
import torch

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
    """
    在固定场景下评估策略性能。
    Returns: {"success_rate", "avg_reward", "avg_steps"}
    """
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
# PPO 训练主循环
# ==============================================================================

def train_ppo(log_dir: str, config: dict):
    """PPO + BC 训练循环（修复版）。"""
    cfg_train  = config["train"]
    cfg_ppo    = config["ppo_agent"]
    cfg_sim    = config["sim"]
 
    TOTAL_STEPS   = int(cfg_train.get("total_timesteps", 5_000_000))
    N_STEPS       = int(cfg_ppo["n_steps"])
    SAVE_INTERVAL = int(cfg_train["save_interval"])
    EVAL_INTERVAL = int(cfg_train.get("eval_interval", 100))
    EVAL_EPS      = int(cfg_train.get("eval_episodes", 10))
    SMOOTH_WIN    = int(cfg_train["log_smooth_win"])
 
    print("[Train-PPO] 初始化环境...")
    env = CableRobotEnvWithObstacles(config=config)
    STATE_DIM  = env.state_dim
    ACTION_DIM = config["space"]["action_dim"]
    print(f"  state_dim={STATE_DIM}, action_dim={ACTION_DIM}")
 
    print("[Train-PPO] 初始化 Agent...")
    agent = PPOAgent(log_dir, STATE_DIM, ACTION_DIM, config=config)
 
    print("[Train-PPO] 初始化专家控制器...")
    expert = JointSpaceExpert(config, env.ik_solver)
 
    logger = Logger(log_dir, "cable_robot_ppo", os.path.basename(log_dir))
    logger.update_config(config)
 
    log_file = os.path.join(log_dir, "ppo_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow([
            "episode", "total_steps", "episode_reward", "avg_reward",
            "success", "success_rate", "steps",
            "policy_loss", "value_loss", "entropy_loss", "bc_loss",
            "approx_kl", "clip_fraction", "bc_coef",
        ])
 
    stats    = EpisodeStats(window=SMOOTH_WIN)
    episode  = 0
    total_steps = 0
    best_sr  = 0.0
    t_start  = time.time()
    last_result = agent._last_result
 
    print(f"[Train-PPO] 开始训练，目标总步数 {TOTAL_STEPS}...")
 
    while total_steps < TOTAL_STEPS:
 
        # ── 每回合开始 ──────────────────────────────────────────────────────
        obs = env.reset()
        current_q = env.data.qpos[:7].copy()
 
        # [LOOP-1] 传入 env，让 expert 直接读取精确 EE 位置
        expert.reset(obs, current_q, env=env)
 
        planned_path = env.get_planned_path()
        if planned_path is None:
            print(f"[Warn] Ep {episode}: 路径规划失败，跳过。")
            episode += 1; continue
        expert.set_path(planned_path)
 
        ep_reward = 0.0; ep_steps = 0; ep_success = False
        rollout_done = False
 
        # ── Rollout 收集 ──────────────────────────────────────────────────
        while not rollout_done:
 
            # Step 1: 选择动作（返回 delta_q）
            norm_obs = agent.normalize_obs(obs, update=True)
            delta_q, log_prob, value = agent.act(norm_obs, deterministic=False)
            
            # Step 2: BC 目标（delta_q_expert = q_expert_next - q_current）
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            bc_delta_q = expert.compute_delta_q_target(obs, current_q)  # [DELTA-C1]
            
            # BC 标签有效性检查
            if np.any(np.isnan(bc_delta_q)):
                bc_delta_q = np.zeros(ACTION_DIM, np.float32)
            
            # Step 3: 执行 delta_q
            next_obs, reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated
            
            # [FIX-L1] 更新回合统计（原版遗漏，导致日志全为 0）
            ep_reward += reward
            ep_steps  += 1
            total_steps += 1
            agent.total_steps = total_steps  # [FIX-L2] 同步 agent 步数，使 BC 退火生效
            if info.get("is_success"):
                ep_success = True
            
            # Step 4: 存入 buffer（delta_q 和 bc_delta_q 量级统一，BC Loss 有意义）
            agent.buffer.add(norm_obs, delta_q, bc_delta_q, reward, float(done), value, log_prob)
 
            obs = next_obs
 
            # buffer 满时触发更新
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
 
        # ── 回合统计 ──────────────────────────────────────────────────────
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
            "loss/bc":                 last_result.bc_loss,
            "ppo/approx_kl":           last_result.approx_kl,
            "ppo/clip_fraction":       last_result.clip_fraction,
            "ppo/bc_coef":             agent.bc_coef,
            "train/total_steps":       total_steps,
        })
 
        mark = "✅" if ep_success else "❌"
        print(
            f"Ep {episode:4d} {mark} | "
            f"R:{ep_reward:7.2f}(avg:{avg_r:6.2f}) | "
            f"SR:{sr*100:5.1f}% | Steps:{ep_steps:3d} | "
            f"bc:{agent.bc_coef:.3f} | "
            f"Lp:{last_result.policy_loss:.4f} Lv:{last_result.value_loss:.4f} | "
            f"total:{total_steps}"
        )
 
        with open(log_file, "a", newline="") as f:
            csv.writer(f).writerow([
                episode, total_steps, ep_reward, avg_r,
                int(ep_success), sr, ep_steps,
                last_result.policy_loss, last_result.value_loss,
                last_result.entropy_loss, last_result.bc_loss,
                last_result.approx_kl, last_result.clip_fraction,
                agent.bc_coef,
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
    print(f"\\n[Train-PPO] 完成！总步数 {total_steps}，耗时 {elapsed:.1f} 分钟")
    logger.close()
    return agent


# ==============================================================================
# TD3 训练主循环
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
    ACT_LOW  = np.array(config["space"]["action_space_low"])
    ACT_HIGH = np.array(config["space"]["action_space_high"])

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
            # 归一化观测
            norm_obs = agent.normalize_obs(obs, update=True)

            # 专家生成 BC 目标关节角
            current_q  = env.data.qpos[:7].copy().astype(np.float32)
            bc_target  = expert.compute_joint_target(obs, current_q)

            # epsilon-greedy：专家也输出 delta_q
            if random.random() < agent.epsilon:
                current_q = env.data.qpos[:7].copy().astype(np.float32)
                # 专家输出绝对关节角，转换为 delta
                action_expert = expert.compute_joint_target(obs, current_q)
                delta_q = np.clip(action_expert - current_q, -agent.dq_max, agent.dq_max)
            else:
                delta_q, _ = agent.act(norm_obs)
                # 探索噪声
                noise = np.random.normal(0., EXPLORE_NOISE * agent.dq_max)
                delta_q = np.clip(delta_q + noise, -agent.dq_max, agent.dq_max)
            
            # BC 目标：delta_q_expert
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            bc_delta_q = expert.compute_delta_q_target(obs, current_q)
            
            # 执行
            next_obs, reward, terminated, truncated, info = env.step(delta_q)
            done = terminated or truncated  # [FIX-L3] done 变量之前未定义
            
            # [FIX-L4] 更新回合统计
            ep_reward += reward
            ep_steps  += 1
            frames    += 1
            if info.get("is_success"):
                ep_success = True
            
            # 存入 buffer
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