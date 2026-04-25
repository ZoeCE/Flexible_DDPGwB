# ==============================================================================
# TD3learn.py — TD3 训练主程序（稳定修复版）
#
# 相对原版的修复：
#
# [LEARN-FIX-1] 消除双重探索噪声（对应 agent.py FIX-1）
#   原版：agent.act() 内部叠加噪声，TD3learn 对 is_network=True 的动作再叠加 EXPLORE_NOISE。
#   修复：agent.act() 只返回纯净 Actor 输出；TD3learn 统一负责：
#         · is_network=True  → action + Gaussian(0, EXPLORE_NOISE)，再 clip
#         · is_network=False → 直接执行专家动作
#
# [LEARN-FIX-2] epsilon 衰减时机
#   原版：epsilon 在 agent.train() 内部衰减，与训练次数（iterations）耦合，
#         导致每 step 调 train(GRAD_UPDATES) 时衰减了 GRAD_UPDATES 倍。
#   修复：epsilon 衰减在每个 env step 结束后调用 agent.step_epsilon() 一次，
#         与梯度更新次数解耦。
#
# [LEARN-FIX-3] remember() 接口更新
#   新版 remember() 移除了 next_base_action 参数（base_boot 已删除），
#   接口更简洁：(state, action, base_action, next_state, reward, done)。
#
# [LEARN-FIX-4] train() 返回值解包安全
#   原版 buffer 不足时返回 3 个值，充足后返回 5 个值，解包失败。
#   新版统一返回 TrainResult namedtuple，始终 5 个字段。
#
# [LEARN-FIX-5] build_agent_from_config 不再重建 ReplayBuffer
#   原版重建了一次 buffer（冗余且危险）；新版 agent.__init__ 直接从 config 读取，
#   build_agent_from_config 只做设备迁移和优化器重建。
# ==============================================================================

import os
import csv
import copy
import time
import random
import numpy as np
import torch

from rich.progress import (
    Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
)

from config import DEFAULT_CONFIG
from agent import WBAgent, np_to_tensor   # 新版 agent 接口


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
        vals = self._data.get("success", [])
        return float(np.mean(vals)) if vals else 0.0


class Logger:
    def __init__(self, log_dir, project="cable_robot", run_name=None,
                 use_wandb=True, use_tb=True):
        self._wandb  = None
        self._writer = None
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                self._wandb.init(
                    project=project,
                    name=run_name or os.path.basename(log_dir),
                    dir=log_dir, config={}, resume="allow",
                )
                print("[Logger] wandb 初始化成功。")
            except Exception as e:
                print(f"[Logger] wandb 不可用：{e}")

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(os.path.join(log_dir, "tb"))
                print(f"[Logger] TensorBoard 初始化成功。")
            except Exception as e:
                print(f"[Logger] TensorBoard 不可用：{e}")

    def update_config(self, cfg):
        if self._wandb is None:
            return
        flat = {}
        def _flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
                if isinstance(v, dict):
                    _flatten(v, key + "/")
                else:
                    flat[key] = v
        _flatten(cfg)
        self._wandb.config.update(flat)

    def log(self, step, metrics):
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, float(v), global_step=step)

    def close(self):
        if self._wandb  is not None: self._wandb.finish()
        if self._writer is not None: self._writer.close()


# ==============================================================================
# 构建 Agent（FIX-5：不再重建 ReplayBuffer）
# ==============================================================================

def build_agent(config: dict, state_dim: int, nmpc_wrapper_func=None, log_dir=None) -> WBAgent:
    cfg_t = config["train"]
    gpu_id = cfg_t.get("gpu_id", 0)
    device = (torch.device(f"cuda:{gpu_id}")
              if torch.cuda.is_available() and gpu_id >= 0
              else torch.device("cpu"))
    print(f"[Agent] 使用设备：{device}")

    agent = WBAgent(
        log_dir=log_dir,
        state_dim=state_dim,
        action_dim=config["space"]["action_dim"],
        config=config,
        base_controller_func=nmpc_wrapper_func,
    )

    # 迁移网络到指定设备
    agent.actor        = agent.actor.to(device)
    agent.target_actor = agent.target_actor.to(device)
    agent.critic       = agent.critic.to(device)
    agent.target_critic= agent.target_critic.to(device)

    # 重建优化器（确保绑定到正确设备上的参数）
    cfg_a = config["agent"]
    agent.optimizer_actor  = torch.optim.Adam(agent.actor.parameters(),  lr=cfg_a.get("lr_actor",  3e-4))
    agent.optimizer_critic = torch.optim.Adam(agent.critic.parameters(), lr=cfg_a.get("lr_critic", 3e-4))

    return agent


# ==============================================================================
# checkpoint 保存
# ==============================================================================

def save_checkpoint(agent: WBAgent, log_dir: str, episode: int, tag: str = ""):
    fname = f"ckpt_{tag}.pt" if tag else f"ckpt_ep{episode}.pt"
    path  = os.path.join(log_dir, fname)
    agent.save(path)
    # 始终覆盖最新
    agent.save(os.path.join(log_dir, "ckpt_latest.pt"))
    return path


# ==============================================================================
# 主训练函数
# ==============================================================================

def train(log_dir: str, custom_config: dict = None):
    """
    TD3 + BC 训练主循环。

    Epsilon-greedy 策略：
      · episode < WARMUP_EPISODES：epsilon=1.0，纯专家预热，填充回放池。
      · episode >= WARMUP_EPISODES：每个 env step 后调用 agent.step_epsilon()，
        线性从 1.0 衰减到 epsilon_min。

    探索噪声（FIX-1）：
      · Actor 执行时（is_network=True）叠加 N(0, EXPLORE_NOISE) 噪声并 clip。
      · 专家执行时（is_network=False）直接使用专家动作，不叠加噪声。
    """

    # ── 0. 合并配置 ──────────────────────────────────────────────────────────
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config:
        for key, val in custom_config.items():
            if isinstance(val, dict) and key in config:
                config[key].update(val)
            else:
                config[key] = val

    cfg_train = config["train"]
    cfg_sim   = config["sim"]
    cfg_ctrl  = config["controller"]
    cfg_agent = config["agent"]

    N_EPISODES      = cfg_train["n_episodes"]
    WARMUP_EPISODES = cfg_train["warmup_episodes"]
    EXPLORE_NOISE   = cfg_train["explore_noise"]       # σ for Actor exploration
    MIN_BUFFER      = cfg_train["min_buffer_to_train"]
    GRAD_UPDATES    = cfg_train["grad_updates_per_step"]
    SAVE_INTERVAL   = cfg_train["save_interval"]
    SMOOTH_WIN      = cfg_train["log_smooth_win"]
    ACTION_DIM      = config["space"]["action_dim"]
    MAX_ACTION      = np.array(config["space"]["action_space_high"], dtype=np.float32)

    # ── 1. 种子与目录 ─────────────────────────────────────────────────────────
    set_global_seed(42)
    os.makedirs(log_dir, exist_ok=True)

    # ── 2. 环境 ───────────────────────────────────────────────────────────────
    print("[Train] 初始化仿真环境...")
    from mujoco_env_new import CableRobotEnvWithObstacles
    env = CableRobotEnvWithObstacles(config=config)
    STATE_DIM = env.state_dim
    print(f"[Train] state_dim={STATE_DIM}, action_dim={ACTION_DIM}")

    # ── 3. NMPC 控制器 ────────────────────────────────────────────────────────
    print("[Train] 初始化 NMPC 控制器...")
    from old.nmpc_controller_new import NMPCTrajectoryTracker
    nmpc = NMPCTrajectoryTracker(
        dt=cfg_ctrl["dt"],
        N=cfg_ctrl["N"],
        L=cfg_ctrl["L"],
        arrival_threshold_xy=cfg_ctrl["arrival_threshold_xy"],
        arrival_threshold_z=cfg_ctrl["arrival_threshold_z"],
    )

    def nmpc_wrapper(state: np.ndarray) -> np.ndarray:
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        return nmpc.compute_action(state, target_yaw=0.0)

    # ── 4. Agent ──────────────────────────────────────────────────────────────
    print("[Train] 初始化 TD3 Agent...")
    agent = build_agent(config, STATE_DIM, nmpc_wrapper, log_dir)

    # ── 5. Logger & CSV ───────────────────────────────────────────────────────
    logger = Logger(log_dir, "cable_robot_td3", os.path.basename(log_dir))
    logger.update_config(config)

    log_file = os.path.join(log_dir, "log.csv")
    csv_header = [
        "episode", "frames_total", "episode_reward", "avg_reward",
        "success", "success_rate", "steps", "expert_ratio",
        "loss_critic", "loss_actor", "loss_bc",
        "q_pred", "q_target", "epsilon", "buffer_size",
    ]
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)

    # ── 6. 统计 ───────────────────────────────────────────────────────────────
    stats        = EpisodeStats(window=SMOOTH_WIN)
    frames_total = 0
    last_result  = type("R", (), {
        "critic_loss": 0.0, "actor_loss": 0.0, "bc_loss": 0.0,
        "q_pred": 0.0, "q_target": 0.0
    })()

    # ── 7. 主循环 ─────────────────────────────────────────────────────────────
    print(f"[Train] 开始训练 {N_EPISODES} 回合，预热 {WARMUP_EPISODES} 回合...")
    t_start = time.time()

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.1f}%"),
        TimeElapsedColumn(), TimeRemainingColumn(),
        auto_refresh=False,
    )

    with progress:
        task_id = progress.add_task(
            "[cyan]Training...",
            total=N_EPISODES * cfg_sim["max_steps"]
        )

        for episode in range(N_EPISODES):

            # ── 预热阶段：纯专家，不衰减 epsilon ────────────────────────────
            if episode < WARMUP_EPISODES:
                agent.epsilon = 1.0

            # ── 环境重置 & 路径同步 ──────────────────────────────────────────
            state = env.reset()
            planned_path = env.get_planned_path()
            if planned_path is None:
                print(f"[Warn] Ep {episode}: 路径规划失败，跳过。")
                continue
            nmpc.set_path(planned_path)

            # ── 回合状态初始化 ───────────────────────────────────────────────
            ep_reward    = 0.0
            step_count   = 0
            expert_count = 0
            ep_success   = False

            # ── 回合内交互 ───────────────────────────────────────────────────
            while True:

                # Step 1: Agent 决策（act() 返回纯净 Actor 输出 + 专家输出）
                actor_action, base_action = agent.act(state)

                # Step 2: Epsilon-greedy 切换（FIX-1：噪声统一在此处叠加）
                if random.random() < agent.epsilon:
                    # 专家执行
                    action_exec  = base_action.copy()
                    expert_count += 1
                else:
                    # Actor + 探索噪声
                    noise       = np.random.normal(0.0, EXPLORE_NOISE * MAX_ACTION)
                    action_exec = np.clip(actor_action + noise, -MAX_ACTION, MAX_ACTION)

                # NaN 保护
                if np.isnan(action_exec).any():
                    action_exec  = base_action.copy()
                    expert_count += 1

                # Step 3: 环境执行
                next_state, reward, terminated, truncated, info = env.step(action_exec)
                done    = terminated or truncated
                success = info.get("is_success", False)
                if success:
                    ep_success = True

                # Step 4: 存入回放池（新接口，无 next_base_action 参数）
                agent.remember(
                    state, action_exec, base_action,
                    next_state, reward, done
                )

                # Step 5: 梯度更新
                if agent.buffer.size > MIN_BUFFER:
                    last_result = agent.train(GRAD_UPDATES)

                # Step 6: Epsilon 衰减（FIX-6：每 step 衰减一次，预热期跳过）
                if episode >= WARMUP_EPISODES:
                    agent.step_epsilon()

                # Step 7: 状态转移
                state        = next_state
                ep_reward   += reward
                step_count  += 1
                frames_total += 1

                progress.update(task_id, advance=1)

                if done:
                    break

            # ── 回合统计 ─────────────────────────────────────────────────────
            expert_ratio = expert_count / max(step_count, 1)
            stats.update(reward=ep_reward, steps=step_count, success=float(ep_success))

            avg_r  = stats.mean("reward")
            avg_s  = stats.mean("steps")
            sr     = stats.success_rate()

            logger.log(episode, {
                "reward/episode":            ep_reward,
                f"reward/avg{SMOOTH_WIN}":   avg_r,
                "steps/episode":             step_count,
                "env/success":               float(ep_success),
                "env/success_rate":          sr,
                "loss/critic":               last_result.critic_loss,
                "loss/actor":                last_result.actor_loss,
                "loss/bc":                   last_result.bc_loss,
                "Q/pred":                    last_result.q_pred,
                "Q/target":                  last_result.q_target,
                "explore/epsilon":           agent.epsilon,
                "explore/expert_ratio":      expert_ratio,
                "train/buffer_size":         agent.buffer.size,
                "train/frames_total":        frames_total,
            })
            progress.refresh()

            mark = "✅" if ep_success else "❌"
            print(
                f"Ep {episode:4d} {mark} | "
                f"R: {ep_reward:7.2f} (avg:{avg_r:7.2f}) | "
                f"SR: {sr*100:5.1f}% | Steps:{step_count:3d} | "
                f"ε:{agent.epsilon:.4f} | "
                f"Lc:{last_result.critic_loss:.4f} | "
                f"Q_pred:{last_result.q_pred:.2f} | "
                f"Buf:{agent.buffer.size}"
            )

            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode, frames_total, ep_reward, avg_r,
                    int(ep_success), sr, step_count, expert_ratio,
                    last_result.critic_loss, last_result.actor_loss, last_result.bc_loss,
                    last_result.q_pred, last_result.q_target,
                    agent.epsilon, agent.buffer.size,
                ])

            if episode > 0 and episode % SAVE_INTERVAL == 0:
                p = save_checkpoint(agent, log_dir, episode)
                print(f"[Train] Checkpoint → {p}")

    # ── 训练结束 ──────────────────────────────────────────────────────────────
    save_checkpoint(agent, log_dir, N_EPISODES, tag="final")
    print(f"\n[Train] 完成！耗时 {(time.time()-t_start)/60:.1f} 分钟")
    logger.close()
    return agent


# ==============================================================================
# 命令行入口
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir",  default="saves/td3_run")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--render",   action="store_true")
    parser.add_argument("--gpu",      type=int, default=0)
    args = parser.parse_args()

    cli_cfg = {}
    if args.render:   cli_cfg.setdefault("sim",   {})["render"]     = True
    if args.gpu != 0: cli_cfg.setdefault("train", {})["gpu_id"]     = args.gpu
    if args.episodes: cli_cfg.setdefault("train", {})["n_episodes"]  = args.episodes

    train(args.log_dir, cli_cfg or None)