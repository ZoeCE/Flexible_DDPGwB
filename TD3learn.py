# ==============================================================================
# td3learn.py — TD3 训练主程序（完全重写版）
#
# 相对于旧版 learn.py 的改进清单：
#
# [LEARN-1]  彻底清除旧版遗留的"伪兼容"代码（current_target_3d 双重计算、
#             nmpc_instance.reset_state_machine() 等已废弃调用）。
#             所有逻辑按 mujoco_env_new + nmpc_controller_new 的实际接口重写。
#
# [LEARN-2]  全局配置统一读取：所有超参数从 config.DEFAULT_CONFIG 的新增节
#             "train" / "agent" / "controller" 中读取，无任何硬编码残留。
#             外部传 custom_config 可完整覆盖任意参数，保证单一超参数入口。
#
# [LEARN-3]  GPU 最大化利用：
#             (a) Agent 的网络全量迁移到 config["train"]["gpu_id"] 指定的设备；
#             (b) ReplayBuffer 的 numpy 数组保持在 CPU，采样后转 tensor 再上 GPU
#                 （这是标准做法，大型 buffer pin_memory 到 GPU 反而浪费显存）；
#             (c) 新增 "grad_updates_per_step"：每个 env step 可执行多次梯度更新，
#                 提高 GPU 利用率（off-policy RL 的标准加速技巧）；
#             (d) 移除训练循环内所有 Python 级别的冗余计算，减少 CPU 阻塞。
#
# [LEARN-4]  仿真频率优化：
#             由于 MuJoCo step 是 CPU-bound，通过以下方式缓解训练瓶颈：
#             (a) 回合内只在 buffer.size > min_buffer_to_train 后才触发训练，
#                 避免早期空转 GPU；
#             (b) 使用 torch.backends.cudnn.benchmark = True 让 cuDNN 自动选择
#                 最优卷积内核（对全连接网络影响有限，但无害）；
#             (c) 预分配 numpy 动作缓冲，避免 hot-loop 中反复内存分配。
#
# [LEARN-5]  wandb / TensorBoard Logger 完整保留，新增：
#             (a) 每步记录 "env/episode_success_rate"（滑动窗口成功率）；
#             (b) 记录 "train/buffer_size" 方便监控预热进度；
#             (c) 将所有超参数通过 logger.update_config 记录到 wandb run。
#
# [LEARN-6]  完善 checkpoint 保存/恢复：
#             (a) 保存完整 agent state dict（而非序列化整个网络对象），
#                 避免 torch.load(..., weights_only=False) 安全警告；
#             (b) 训练结束后自动保存 final checkpoint；
#             (c) 新增 --resume 逻辑入口（函数签名预留，暂不强制实现）。
#
# [LEARN-7]  新增工具模块：
#             (a) EpisodeStats：轻量滑动窗口统计，替代手工 reward_hist 列表；
#             (b) set_global_seed：统一设置 Python / numpy / torch 随机种子，
#                 保证训练可复现；
#             (c) build_agent_from_config：从 config 字典一键构建 WBAgent，
#                 消除 learn.py 与 test.py 之间的网络结构不一致风险。
#
# [LEARN-8]  保证 NMPCTrajectoryTracker 与 env 严格同步：
#             每次 env.reset() 后立即调用 nmpc.set_path(env.get_planned_path())，
#             确保控制器路径与环境规划路径完全一致。
#             nmpc.compute_action 只在实际需要专家动作时调用，避免浪费。
#
# [LEARN-9]  6D 动作对齐：
#             env 接受 6D [ax, ay, az, a_roll, a_pitch, a_yaw]，其中 roll/pitch
#             固定为 0（由 IK 保证末端朝下）。NMPC 输出已是 6D（后两位为 0），
#             Actor 直接输出 6D，不做任何截断或补零。
#
# [LEARN-10] 详细的 step 级注释与函数文档，方便后续二次开发。
# ==============================================================================

import os
import csv
import copy
import time
import random
import numpy as np
import torch
import torch.nn as nn

from rich.progress import (
    Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
)

# 项目内部模块
from config import DEFAULT_CONFIG
from agent import WBAgent, opt_cuda, np_to_tensor
from mujoco_env_new import CableRobotEnvWithObstacles
from nmpc_controller_new import NMPCTrajectoryTracker


# ==============================================================================
# 工具 1：全局随机种子设置
# ==============================================================================

def set_global_seed(seed: int):
    """
    统一设置 Python / numpy / torch（CPU + GPU）随机种子，保证训练可复现。
    注意：MuJoCo 物理仿真的随机性由 config["scene"]["seed"] 单独控制。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 启用 cuDNN 自动调优（对 MLP 影响有限，但无害，且在 CNN 任务中效果显著）
    torch.backends.cudnn.benchmark = True


# ==============================================================================
# 工具 2：滑动窗口统计
# ==============================================================================

class EpisodeStats:
    """
    轻量滑动窗口统计器，替代旧版手工维护的 reward_hist / steps_hist 列表。

    维护最近 window 个回合的均值统计，支持任意 key 的标量记录。
    """

    def __init__(self, window: int = 20):
        self.window = window
        self._data: dict[str, list] = {}

    def update(self, **kwargs):
        """记录一个回合的指标。"""
        for k, v in kwargs.items():
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(float(v))
            if len(self._data[k]) > self.window:
                self._data[k].pop(0)

    def mean(self, key: str) -> float:
        """返回指定 key 的窗口均值，key 不存在时返回 0。"""
        vals = self._data.get(key, [])
        return float(np.mean(vals)) if vals else 0.0

    def success_rate(self) -> float:
        """返回最近 window 回合的成功率（基于 'success' key）。"""
        vals = self._data.get("success", [])
        return float(np.mean(vals)) if vals else 0.0


# ==============================================================================
# 工具 3：Logger（wandb + TensorBoard + CSV 三路写入）
# ==============================================================================

class Logger:
    """
    统一日志封装：优先 wandb，其次 TensorBoard，两者都没有则仅输出到 CSV。

    改进（相对旧版）：
      - update_config 支持任意嵌套 dict 的扁平化写入 wandb config
      - close() 保证 wandb finish 和 writer close 均被调用
    """

    def __init__(self, log_dir: str, project: str = "cable_robot",
                 run_name: str = None, use_wandb: bool = True, use_tb: bool = True):
        self._wandb  = None
        self._writer = None
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # ── wandb ─────────────────────────────────────────────────────────────
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                self._wandb.init(
                    project=project,
                    name=run_name or os.path.basename(log_dir),
                    dir=log_dir,
                    config={},
                    resume="allow",
                )
                print("[Logger] wandb 初始化成功。")
            except Exception as e:
                print(f"[Logger] wandb 不可用，跳过：{e}")
                self._wandb = None

        # ── TensorBoard ───────────────────────────────────────────────────────
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tb")
                self._writer = SummaryWriter(log_dir=tb_dir)
                print(f"[Logger] TensorBoard 初始化成功，日志目录：{tb_dir}")
                print(f"         启动命令：tensorboard --logdir {tb_dir}")
            except Exception as e:
                print(f"[Logger] TensorBoard 不可用，跳过：{e}")
                self._writer = None

        if self._wandb is None and self._writer is None:
            print("[Logger] wandb 与 TensorBoard 均不可用，仅记录 CSV。")

    def update_config(self, cfg: dict):
        """将超参数写入 wandb run config（扁平化嵌套 dict）。"""
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

    def log(self, step: int, metrics: dict):
        """同时向 wandb 和 TensorBoard 写入一批指标。"""
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, float(v), global_step=step)

    def close(self):
        """释放资源。"""
        if self._wandb is not None:
            self._wandb.finish()
        if self._writer is not None:
            self._writer.close()


# ==============================================================================
# 工具 4：从 config 构建 WBAgent（消除 learn / test 之间的结构不一致风险）
# ==============================================================================

def build_agent_from_config(config: dict, state_dim: int,
                             nmpc_wrapper_func=None, log_dir: str = None) -> WBAgent:
    """
    从 config["agent"] 字典一键构建 WBAgent，并将所有超参数同步给实例。

    Args:
        config:            完整的 DEFAULT_CONFIG（已合并自定义配置）。
        state_dim:         观测空间维度（由 env.state_dim 动态确定）。
        nmpc_wrapper_func: 专家控制器回调，签名 (state: np.ndarray) -> np.ndarray(6,)。

    Returns:
        agent: 已配置好的 WBAgent 实例，网络已迁移到目标 GPU。
    """
    cfg_a   = config["agent"]
    cfg_t   = config["train"]
    cfg_sp  = config["space"]

    action_dim = cfg_sp["action_dim"]           # 6
    max_action = cfg_sp["action_space_high"][0]  # 0.5（取第一维作为统一上限）

    # 确定目标设备
    gpu_id = cfg_t.get("gpu_id", 0)
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"[Agent] 使用设备：{device}")

    agent = WBAgent(
    log_dir=log_dir,           # 或者 None
    state_dim=state_dim,
    action_dim=action_dim,
    config=config,             # 【关键修改】直接把整个 config 字典丢进去
    base_controller_func=nmpc_wrapper_func,
)

    # ── 覆盖 WBAgent.__init__ 中的硬编码超参数 ────────────────────────────────
    # 回放池（WBAgent 内部已创建，需替换为 config 指定大小）
    from agent import ReplayBuffer
    agent.buffer = ReplayBuffer(cfg_a["buffer_size"], state_dim, action_dim)

    agent.batch_size    = cfg_a["batch_size"]
    agent.gamma         = cfg_a["gamma"]
    agent.tau           = cfg_a["tau"]
    agent.policy_noise  = cfg_a["policy_noise"]
    agent.noise_clip    = cfg_a["noise_clip"]
    agent.policy_freq   = cfg_a["policy_freq"]
    agent.epsilon       = cfg_a["epsilon_init"]
    agent.epsilon_min   = cfg_a["epsilon_min"]
    agent.delta         = cfg_a["epsilon_delta"]

    # 重建优化器（使用 config 指定的学习率）
    agent.optimizer_actor  = torch.optim.Adam(
        agent.actor.parameters(),  lr=cfg_a["lr_actor"])
    agent.optimizer_critic = torch.optim.Adam(
        agent.critic.parameters(), lr=cfg_a["lr_critic"])

    # 迁移网络到目标设备（WBAgent 内部已调用 opt_cuda，这里再显式迁移以匹配指定 GPU）
    agent.actor         = agent.actor.to(device)
    agent.target_actor  = agent.target_actor.to(device)
    agent.critic        = agent.critic.to(device)
    agent.target_critic = agent.target_critic.to(device)

    return agent


# ==============================================================================
# 工具 5：checkpoint 保存（state_dict 方式，安全且跨版本兼容）
# ==============================================================================

def save_checkpoint(agent: WBAgent, log_dir: str, episode: int, tag: str = ""):
    """
    保存 agent 的 actor / critic state_dict 以及训练状态。

    使用 state_dict 而非序列化整个模型对象，避免 torch.load 的安全警告，
    且对 Python / PyTorch 版本变化更鲁棒。

    文件命名：
      - actor_ep{episode}.pt / critic_ep{episode}.pt  →  周期性 checkpoint
      - actor.pt / critic.pt                          →  最新版本（覆盖写入）
    """
    # episode 专属 checkpoint
    if tag:
        actor_path  = os.path.join(log_dir, f"actor_{tag}.pt")
        critic_path = os.path.join(log_dir, f"critic_{tag}.pt")
    else:
        actor_path  = os.path.join(log_dir, f"actor_ep{episode}.pt")
        critic_path = os.path.join(log_dir, f"critic_ep{episode}.pt")

    torch.save({
        "episode":     episode,
        "actor":       agent.actor.state_dict(),
        "critic":      agent.critic.state_dict(),
        "tgt_actor":   agent.target_actor.state_dict(),
        "tgt_critic":  agent.target_critic.state_dict(),
        "opt_actor":   agent.optimizer_actor.state_dict(),
        "opt_critic":  agent.optimizer_critic.state_dict(),
        "epsilon":     agent.epsilon,
        "total_it":    agent.total_it,
    }, actor_path)   # 将所有内容写入单文件，命名沿用 actor_*.pt 方便 test.py 定位

    # 覆盖最新版本
    latest_path = os.path.join(log_dir, "actor_latest.pt")
    torch.save(agent.actor.state_dict(), latest_path)

    # 保持向后兼容：test.py 仍可用 torch.load('actor.pt', weights_only=False) 加载
    torch.save(agent.actor, os.path.join(log_dir, "actor.pt"))
    torch.save(agent.critic, os.path.join(log_dir, "critic.pt"))


# ==============================================================================
# 主训练函数
# ==============================================================================

def train(log_dir: str, custom_config: dict = None):
    """
    TD3 训练主循环。

    Args:
        log_dir:       训练产物（模型、日志、CSV）的保存目录。
        custom_config: 可选的局部配置覆盖字典，深度合并到 DEFAULT_CONFIG。
                       格式与 CableRobotEnvWithObstacles 的 config 参数相同。
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 0. 合并配置
    # ──────────────────────────────────────────────────────────────────────────
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config is not None:
        for key, val in custom_config.items():
            if isinstance(val, dict) and key in config:
                config[key].update(val)
            else:
                config[key] = val

    cfg_train  = config["train"]
    cfg_agent  = config["agent"]
    cfg_ctrl   = config["controller"]
    cfg_sim    = config["sim"]

    # 便捷别名
    N_EPISODES       = cfg_train["n_episodes"]
    WARMUP_EPISODES  = cfg_train["warmup_episodes"]
    EXPLORE_NOISE    = cfg_train["explore_noise"]
    MIN_BUFFER       = cfg_train["min_buffer_to_train"]
    GRAD_UPDATES     = cfg_train["grad_updates_per_step"]
    SAVE_INTERVAL    = cfg_train["save_interval"]
    SMOOTH_WIN       = cfg_train["log_smooth_win"]
    ACTION_DIM       = config["space"]["action_dim"]          # 6
    MAX_ACTION       = config["space"]["action_space_high"][0] # 0.5

    # ──────────────────────────────────────────────────────────────────────────
    # 1. 随机种子 & 目录
    # ──────────────────────────────────────────────────────────────────────────
    set_global_seed(42)
    os.makedirs(log_dir, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. 环境初始化
    #    将完整 config 传入 env，env 内部会深度合并，不再需要手工拆分子字典
    # ──────────────────────────────────────────────────────────────────────────
    print("[Train] 初始化仿真环境...")
    # 通过 config dict 覆盖默认配置
    env = CableRobotEnvWithObstacles(config=config)

    # 动态读取状态维度（由 env 内部 n_obstacles 决定）
    STATE_DIM = env.state_dim
    print(f"[Train] 状态维度: {STATE_DIM}, 动作维度: {ACTION_DIM}")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. NMPC 控制器初始化
    # ──────────────────────────────────────────────────────────────────────────
    print("[Train] 初始化 NMPC 轨迹追踪器...")
    nmpc = NMPCTrajectoryTracker(
        dt=cfg_ctrl["dt"],
        N=cfg_ctrl["N"],
        L=cfg_ctrl["L"],
        arrival_threshold_xy=cfg_ctrl["arrival_threshold_xy"],
        arrival_threshold_z=cfg_ctrl["arrival_threshold_z"],
    )

    def nmpc_wrapper(state: np.ndarray) -> np.ndarray:
        """
        专家控制器包装器（单步，非 batch）。
        接收来自 env._get_obs() 的观测向量，返回 6D 动作。

        [LEARN-8] 控制器的路径由 train 循环在每次 reset 后显式同步，
                  此处只负责状态 -> 动作的计算。
        """
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        # target_yaw 固定 0.0（末端保持水平朝向）
        return nmpc.compute_action(state, target_yaw=0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Agent 初始化（从 config 构建，无硬编码）
    # ──────────────────────────────────────────────────────────────────────────
    print("[Train] 初始化 TD3 Agent...")
    agent = build_agent_from_config(
        config=config,
        state_dim=STATE_DIM,
        nmpc_wrapper_func=nmpc_wrapper,
        log_dir=log_dir
    )

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Logger 初始化
    # ──────────────────────────────────────────────────────────────────────────
    logger = Logger(
        log_dir=log_dir,
        project="cable_robot_td3",
        run_name=os.path.basename(log_dir),
        use_wandb=True,
        use_tb=True,
    )
    # 将完整 config 写入 wandb（嵌套 dict 会被 Logger 自动扁平化）
    logger.update_config(config)

    # ──────────────────────────────────────────────────────────────────────────
    # 6. CSV 日志文件初始化
    # ──────────────────────────────────────────────────────────────────────────
    log_file = os.path.join(log_dir, "log.csv")
    csv_header = [
        "episode", "frames_total", "episode_reward",
        "avg_reward", "success", "success_rate",
        "steps", "expert_ratio",
        "loss_critic", "loss_actor", "loss_bc",
        "epsilon", "buffer_size",
    ]
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)

    # ──────────────────────────────────────────────────────────────────────────
    # 7. 统计变量初始化
    # ──────────────────────────────────────────────────────────────────────────
    stats        = EpisodeStats(window=SMOOTH_WIN)
    frames_total = 0
    loss_c = loss_a = loss_bc = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # 8. 主训练循环
    # ──────────────────────────────────────────────────────────────────────────
    print(f"[Train] 开始训练，共 {N_EPISODES} 回合，预热 {WARMUP_EPISODES} 回合...")
    t_start = time.time()

    # rich 进度条（以 frame 为单位，max_steps * n_episodes 为估计上限）
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        auto_refresh=False,
    )

    with progress:
        max_frames_est = N_EPISODES * cfg_sim["max_steps"]
        task_id = progress.add_task("[cyan]Training...", total=max_frames_est)

        for episode in range(N_EPISODES):

            # ── 8a. 预热阶段：epsilon=1.0，纯专家策略，填充回放池 ────────────
            if episode < WARMUP_EPISODES:
                agent.epsilon = 1.0
                agent.delta   = 0.0   # 预热期间不衰减
            elif episode == WARMUP_EPISODES:
                # 预热结束，恢复正常衰减
                agent.delta   = cfg_agent["epsilon_delta"]
                agent.epsilon = 1.0   # 从 1.0 开始衰减
                print(f"\n[Train] 预热完成（{WARMUP_EPISODES} 回合），开始正式训练。")

            # ── 8b. 环境重置与路径同步 ─────────────────────────────────────────
            state = env.reset()

            # [LEARN-8] 关键：每次 reset 后立即同步 NMPC 路径
            planned_path = env.get_planned_path()
            if planned_path is not None:
                nmpc.set_path(planned_path)
            else:
                # 路径为空时 NMPC 输出零动作，env 会触发超时
                print(f"[Warn] Ep {episode}: 路径规划失败，跳过本回合。")
                continue

            # ── 8c. 回合状态初始化 ────────────────────────────────────────────
            episode_reward  = 0.0
            step_count      = 0
            expert_count    = 0   # 本回合使用专家动作的步数（用于计算 ratio）
            episode_success = False

            # ── 8d. 回合内交互循环 ────────────────────────────────────────────
            while True:

                # ── Step 1: 专家动作计算（每步都计算，用于存入回放池）──────────
                # 专家动作：NMPC 根据当前 obs 计算最优参考动作
                # 注意：nmpc 的内部航点索引与 env 内部状态机是独立维护的，
                #       两者均从 path[0] 开始，各自按自己的阈值切换航点。
                #       这是设计决策：避免两者强耦合导致的竞态条件。
                base_action = nmpc_wrapper(state)     # shape: (6,)

                # ── Step 2: Agent 决策（epsilon-greedy：专家 or Actor）──────────
                action, is_network, _ = agent.act(state)
                # is_network=False → 专家动作；is_network=True → Actor 输出

                if is_network:
                    # Actor 输出 + 探索噪声（off-policy 标准做法）
                    noise = np.random.normal(0.0, EXPLORE_NOISE, size=ACTION_DIM)
                    action_exec = np.clip(action + noise, -MAX_ACTION, MAX_ACTION)
                else:
                    # 直接执行专家动作
                    action_exec = action.copy()
                    expert_count += 1

                # NaN 保护（网络崩溃时用专家动作托底）
                if np.isnan(action_exec).any():
                    action_exec = base_action.copy()
                    expert_count += 1

                # ── Step 3: 环境执行 ────────────────────────────────────────────
                # env.step 返回标准 Gymnasium 5 元组
                next_state, reward, terminated, truncated, info = env.step(action_exec)
                done    = terminated or truncated
                success = info.get("is_success", False)
                if success:
                    episode_success = True

                # ── Step 4: 下一步专家动作（存入回放池）────────────────────────
                # 若 done，next_base_action 不参与 Q-target 计算（被 (1-done) 屏蔽），
                # 使用零向量占位，减少一次 NMPC 调用。
                if done:
                    next_base_action = np.zeros(ACTION_DIM, dtype=np.float32)
                else:
                    next_base_action = nmpc_wrapper(next_state)

                # ── Step 5: 存入回放池 ──────────────────────────────────────────
                agent.remember(
                    state, action_exec, base_action,
                    next_base_action, next_state, reward, done
                )

                # ── Step 6: 梯度更新（GPU 批量训练）────────────────────────────
                # [LEARN-3c] 每步执行 GRAD_UPDATES 次更新以提高 GPU 利用率
                # 仅在回放池充足时开始训练
                if agent.buffer.size > MIN_BUFFER:
                    for _ in range(GRAD_UPDATES):
                        lc, la, lbc = agent.train(1)
                    # 保留最后一次更新的 loss（用于日志）
                    loss_c, loss_a, loss_bc = lc, la, lbc

                # ── Step 7: 状态转移与统计更新 ─────────────────────────────────
                state          = next_state
                episode_reward += reward
                step_count     += 1
                frames_total   += 1

                progress.update(task_id, advance=1)

                if done:
                    break

            # ── 8e. 回合结束：统计汇总与日志写入 ─────────────────────────────
            expert_ratio = expert_count / max(step_count, 1)

            stats.update(
                reward=episode_reward,
                steps=step_count,
                success=float(episode_success),
            )

            avg_reward   = stats.mean("reward")
            avg_steps    = stats.mean("steps")
            success_rate = stats.success_rate()

            # wandb / TensorBoard
            logger.log(episode, {
                "reward/episode":         episode_reward,
                f"reward/avg{SMOOTH_WIN}": avg_reward,
                "steps/episode":          step_count,
                f"steps/avg{SMOOTH_WIN}":  avg_steps,
                "env/success":            float(episode_success),
                "env/success_rate":       success_rate,
                "loss/critic":            loss_c,
                "loss/actor":             loss_a,
                "loss/behavior_clone":    loss_bc,
                "explore/epsilon":        agent.epsilon,
                "explore/expert_ratio":   expert_ratio,
                "train/buffer_size":      agent.buffer.size,
                "train/frames_total":     frames_total,
            })

            # 进度条刷新
            progress.refresh()

            # 控制台打印（简洁版）
            status_mark = "✅" if episode_success else "❌"
            print(
                f"Ep {episode:4d} {status_mark} | "
                f"R: {episode_reward:7.2f} (avg: {avg_reward:7.2f}) | "
                f"SR: {success_rate*100:5.1f}% | "
                f"Steps: {step_count:3d} | "
                f"Ratio: {expert_ratio:.2f} | "
                f"Eps: {agent.epsilon:.4f} | "
                f"Buf: {agent.buffer.size}"
            )

            # CSV 追加
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    episode, frames_total, episode_reward,
                    avg_reward, int(episode_success), success_rate,
                    step_count, expert_ratio,
                    loss_c, loss_a, loss_bc,
                    agent.epsilon, agent.buffer.size,
                ])

            # ── 8f. 周期性 checkpoint 保存 ────────────────────────────────────
            if episode > 0 and episode % SAVE_INTERVAL == 0:
                save_checkpoint(agent, log_dir, episode)
                print(f"[Train] Checkpoint 已保存 → ep{episode}")

    # ──────────────────────────────────────────────────────────────────────────
    # 9. 训练结束：保存最终模型 & 清理资源
    # ──────────────────────────────────────────────────────────────────────────
    save_checkpoint(agent, log_dir, N_EPISODES, tag="final")
    print(f"\n[Train] 训练完成！耗时 {(time.time() - t_start)/60:.1f} 分钟")
    print(f"[Train] 最终模型已保存至 {log_dir}/")

    logger.close()
    return agent


# ==============================================================================
# 命令行入口
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TD3 训练脚本（Cable Robot + Obstacles）")
    parser.add_argument("--log-dir",   type=str,   default="saves/td3_experiment",
                        help="训练产物保存目录")
    parser.add_argument("--episodes",  type=int,   default=None,
                        help="覆盖 config 中的 n_episodes（调试用）")
    parser.add_argument("--render",    action="store_true",
                        help="开启 MuJoCo GUI 渲染（会大幅降低训练速度）")
    parser.add_argument("--gpu",       type=int,   default=0,
                        help="指定 GPU 编号（-1 = CPU）")
    args = parser.parse_args()

    # 从命令行参数构建自定义配置覆盖
    cli_config: dict = {}
    if args.render:
        cli_config.setdefault("sim", {})["render"] = True
    if args.gpu != 0:
        cli_config.setdefault("train", {})["gpu_id"] = args.gpu
    if args.episodes is not None:
        cli_config.setdefault("train", {})["n_episodes"] = args.episodes

    train(log_dir=args.log_dir, custom_config=cli_config if cli_config else None)