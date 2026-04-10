# ==============================================================================
# agent.py — TD3 + BC Agent（完全重写，稳定版）
#
# 修复清单（针对原版 Q 爆炸 / Critic Loss 飙升 / 训练不稳定问题）：
#
# [FIX-1]  双重探索噪声叠加（根源问题）
#          原版 act() 在 Actor 输出上叠加噪声后返回，TD3learn.py 又对 is_network=True
#          的动作再叠加一次 EXPLORE_NOISE。导致实际执行噪声远超设计值，轨迹混乱，
#          回放池中充斥高方差样本，Critic 直接学到"动作->Q值"的虚假映射，Q发散。
#          修复：act() 返回原始 Actor 输出，噪声注入统一交给调用方（TD3learn.py）。
#
# [FIX-2]  Base Bootstrapping 逻辑错误（核心不稳定源）
#          原版 base_boot 混合目标Q：
#            target_Q = 0.5 * target_Q_actor + 0.5 * target_Q_base
#          问题：target_Q_base 使用 next_base_action（专家在 next_state 处的动作），
#          但 next_base_action 是外部传入的裸向量，未经 target_actor 的 smoothing，
#          且 done=True 时仍然存在 base_action（不为零），导致终止状态的 Q 目标偏高，
#          产生系统性 over-estimation，是Q爆炸的直接触发器之一。
#          修复：彻底移除 base_boot 的混合Q机制，base_action 仅用于 BC Loss。
#
# [FIX-3]  BC Loss 中 lmbda 上限过宽（训练早期崩溃源）
#          原版 lmbda = clamp(0.5 / (q_abs_mean + 1e-5), max=100.0)。
#          训练前期 Q ≈ 0，lmbda 可飙升至 100，完全压制 Q-gradient，Actor 退化为
#          纯行为克隆，失去 RL 改进的能力，之后 Q 不为零时 lmbda 突然变小，
#          Actor 剧烈切换，引发 Critic Loss 飙升。
#          修复：采用 TD3+BC 论文原版公式 lmbda = alpha / E[|Q|]，alpha 固定 2.5，
#          并将 lmbda clamp 到 [0.1, 10.0] 防两端极值。
#
# [FIX-4]  Target Q 未 clamp（Q无限增长的直接通道）
#          原版注释掉了 target_q_abs_max 的 clamp（见原代码第404行注释块）。
#          在稀疏奖励环境中，初期 Q 目标可能因 bootstrap 累积误差而无限增长。
#          修复：重新启用 target_Q clamp，根据环境奖励量级合理设置上限。
#          按照本任务奖励设计：最优累计回报 ≈ N_waypoints*0.05 + 1.0 - 步数*0.05
#          ≈ 约 -3 到 +3 之间，gamma=0.99，理论 Q 上限 ≈ 15。设 clamp ±20。
#
# [FIX-5]  状态归一化器在训练批量中被错误使用
#          原版在 train() 中对采样的整个 batch 调用 normalize()，
#          但 Welford 均值/方差是在 remember() 中单条逐步更新的。
#          训练前期样本量少、方差不稳定，归一化后的状态分布剧烈变化，
#          造成 Critic 学到的 Q 函数在同一个 episode 内跨越多个不同"坐标系"。
#          修复：在 remember() 时同时存入归一化后的状态，训练时直接使用存储的归一化值；
#          或者（更简洁）使用 RunningMeanStd + 固定频率更新，避免在线更新的不一致。
#          本版采用"存归一化值"方案，彻底杜绝训练-推理归一化不一致。
#
# [FIX-6]  Epsilon 衰减颗粒度过细，导致专家占比骤降
#          原版 epsilon_delta=2e-6，每次 train(iterations) 调用只衰减一次，
#          但每个 step 都调用 train()，140步/回合 * iterations次，
#          epsilon 衰减速度和实际 step 数不对应，容易出现过早或过晚切换。
#          修复：epsilon 衰减移到训练循环外（TD3learn.py 每 episode 结束后衰减），
#          或按实际 step 数线性插值。本版在 train() 中不衰减 epsilon，
#          改由调用方（TD3learn.py）在每步末尾调用 step_epsilon()。
#
# [FIX-7]  Critic grad clip 与 Actor grad clip 不一致
#          原版 Critic 使用 config 中的 critic_grad_clip，
#          Actor 硬编码为 1.0（见原代码第503行），忽略 config 中的 actor_grad_clip。
#          修复：统一从 config 读取，两者均可配置。
#
# [FIX-8]  train() 返回值数量不稳定
#          buffer 未满时返回 3 个值，满后返回 5 个值，TD3learn.py 解包崩溃。
#          修复：统一返回 5 个值的 namedtuple，buffer 未满时全部返回 0.0。
#
# [FIX-9]  build_agent_from_config 重复初始化 ReplayBuffer
#          TD3learn.py 中 build_agent_from_config 在 WBAgent.__init__ 创建 buffer 后
#          又用相同参数重建了一次，浪费内存且容易引入 size/ptr 不一致的隐患。
#          修复：WBAgent.__init__ 直接从 config 读取 buffer_size，无需外部重建。
#
# [延伸-1] 奖励归一化（reward normalization）
#          对于稀疏奖励（0.05阶段奖励 + 终止奖励 ±1/±3），Critic 输入的 reward
#          量级差异会造成 Huber loss 的 δ 阈值相对偏移，导致早期 loss 虚高。
#          增加可选的 reward running normalization（PopArt 简化版）。
#
# [延伸-2] 网络权重正交初始化
#          原版使用 PyTorch 默认的 Kaiming Uniform 初始化，对深层 MLP 效果次优。
#          改用 orthogonal init（对 RL 收敛更友好，参见 PPO/SAC 论文）。
#
# [延伸-3] Critic 使用 LayerNorm 稳定中间激活
#          在稀疏奖励环境中，某些状态的 Q 输出会在 relu 后产生"dead neuron"，
#          加入 LayerNorm 可缓解此问题。
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from collections import namedtuple

from config import DEFAULT_CONFIG


# ==============================================================================
# 工具函数
# ==============================================================================

def opt_cuda(t, device=None):
    if torch.cuda.is_available():
        if device is None:
            return t.cuda()
        return t.to(device)
    return t


def np_to_tensor(n, device=None):
    return opt_cuda(torch.as_tensor(n, dtype=torch.float32), device)


def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)


def orthogonal_init(module, gain=1.0):
    """对 Linear 层应用正交初始化（RL 收敛更快）。"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


# 训练返回值（命名元组，防止解包出错）
TrainResult = namedtuple(
    "TrainResult",
    ["critic_loss", "actor_loss", "bc_loss", "q_pred", "q_target"]
)

ZERO_RESULT = TrainResult(0.0, 0.0, 0.0, 0.0, 0.0)


# ==============================================================================
# 在线状态归一化（RunningMeanStd，存归一化值入 buffer）
# ==============================================================================

class RunningMeanStd:
    """
    Welford 在线均值/方差估计。
    与原版 WelfordNormalizer 相同数学，但接口更清晰，
    并增加 warm_start 保护：样本数 < warm_start 时不做归一化，直接返回原值。
    """
    def __init__(self, shape, warm_start: int = 100, clip: float = 10.0):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S    = np.zeros(shape, dtype=np.float64)
        self.warm_start = warm_start
        self.clip = clip

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).flatten()
        self.n += 1
        if self.n == 1:
            self.mean = x.copy()
            self.S    = np.zeros_like(x)
        else:
            old_mean  = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S    = self.S   + (x - old_mean) * (x - self.mean)

    @property
    def var(self):
        return self.S / max(self.n - 1, 1)

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.n < self.warm_start:
            return x   # 数据不足时不归一化，避免方差坍塌
        normed = (x - self.mean.astype(np.float32)) / self.std.astype(np.float32)
        return np.clip(normed, -self.clip, self.clip)


# ==============================================================================
# 奖励 Running Normalization（PopArt 简化版，可选）
# ==============================================================================

class RewardNormalizer:
    """
    对奖励做在线白化（减均值除方差），但保留符号方向。
    适用于稀疏奖励量级差异大的场景。
    注意：只在 remember() 时归一化存入 buffer 的奖励，推理时不影响环境奖励计算。
    """
    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-4, clip: float = 10.0):
        self.gamma   = gamma
        self.epsilon = epsilon
        self.clip    = clip
        self.ret_ms  = RunningMeanStd(shape=(1,), warm_start=1, clip=1e9)
        self._ret    = 0.0   # 折扣回报累计（在线估计方差用）

    def normalize(self, reward: float, done: bool) -> float:
        self._ret = self._ret * self.gamma + reward
        self.ret_ms.update(np.array([self._ret]))
        if done:
            self._ret = 0.0
        normed = reward / (np.sqrt(self.ret_ms.var[0]) + self.epsilon)
        return float(np.clip(normed, -self.clip, self.clip))


# ==============================================================================
# 网络定义
# ==============================================================================

class Actor(nn.Module):
    """
    策略网络（Actor）。
    使用正交初始化 + 两层 256 隐层。
    max_action 支持按维度不同上限（多维张量）。
    """
    def __init__(self, state_dim: int, action_dim: int, max_action, hidden_dim: int = 256):
        super().__init__()

        if isinstance(max_action, (float, int)):
            self.register_buffer('max_action', torch.tensor(max_action, dtype=torch.float32))
        else:
            self.register_buffer('max_action', torch.tensor(max_action, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        # 正交初始化（最后一层 gain 小一点，输出不要太极端）
        self.net.apply(lambda m: orthogonal_init(m, gain=1.0))
        orthogonal_init(self.net[-2], gain=0.01)   # 最后一个 Linear

    def forward(self, s):
        return self.net(s) * self.max_action


class Critic(nn.Module):
    """
    双 Q 网络（Twin Critic）。
    每个 Q 网络加入 LayerNorm 稳定中间层激活，防止 dead-neuron。
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim

        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Q2（结构相同，参数独立）
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q1.apply(lambda m: orthogonal_init(m, gain=1.0))
        self.q2.apply(lambda m: orthogonal_init(m, gain=1.0))

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa)


# ==============================================================================
# 经验回放池（存归一化后的状态——修复 FIX-5）
# ==============================================================================

class ReplayBuffer:
    """
    循环经验回放池。
    存入的 state / next_state 是**归一化后**的值（在 remember() 中完成），
    避免训练时批量归一化造成的"坐标系飘移"问题（FIX-5）。
    """
    def __init__(self, max_size: int, state_dim: int, action_dim: int):
        self.max_size = max_size
        self.ptr  = 0
        self.size = 0

        self.state       = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.next_state  = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.action      = np.zeros((max_size, action_dim), dtype=np.float32)
        self.base_action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward      = np.zeros((max_size, 1),          dtype=np.float32)
        self.done        = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, norm_state, action, base_action, norm_next_state, reward, done):
        self.state[self.ptr]       = norm_state
        self.next_state[self.ptr]  = norm_next_state
        self.action[self.ptr]      = action
        self.base_action[self.ptr] = base_action
        self.reward[self.ptr]      = reward
        self.done[self.ptr]        = done
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[idx],
            self.action[idx],
            self.base_action[idx],
            self.next_state[idx],
            self.reward[idx],
            self.done[idx],
        )


# ==============================================================================
# TD3 + BC Agent（稳定重写版）
# ==============================================================================

class WBAgent:
    """
    TD3 + Behavior Cloning（BC）Agent。

    设计原则：
      - 专家（NMPC）通过 BC Loss 提供初期引导，而非 epsilon-greedy 混合执行。
        （在 epsilon 阶段，调用方 TD3learn.py 决定执行专家还是 Actor，
         Agent 自身 act() 只返回纯净的 Actor 输出，不附加任何噪声。）
      - Q 目标限幅防止 bootstrap 累积发散。
      - target_Q 混合 base_boot 机制已移除（FIX-2）。
      - epsilon 衰减由调用方通过 step_epsilon() 显式控制（FIX-6）。
    """

    def __init__(
        self,
        log_dir,
        state_dim:  int,
        action_dim: int,
        config:     dict = None,
        base_controller_func = None,
    ):
        if config is None:
            config = DEFAULT_CONFIG

        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.config     = config

        agent_cfg = config["agent"]
        space_cfg = config["space"]
        train_cfg = config["train"]

        # ── 动作空间上限 ─────────────────────────────────────────────────────
        self.max_action = np.array(space_cfg["action_space_high"], dtype=np.float32)

        # ── 专家控制器 ───────────────────────────────────────────────────────
        self.base_controller_func = base_controller_func
        self.behavior_clone = bool(agent_cfg.get("behavior_clone", True))

        # ── 超参数读取（全部从 config，不硬编码）────────────────────────────
        hidden_dim             = int(agent_cfg.get("hidden_dim",       256))
        self.batch_size        = int(agent_cfg.get("batch_size",        256))
        self.gamma             = float(agent_cfg.get("gamma",           0.99))
        self.tau               = float(agent_cfg.get("tau",             0.005))
        self.policy_noise      = float(agent_cfg.get("policy_noise",    0.1))
        self.noise_clip        = float(agent_cfg.get("noise_clip",      0.25))
        self.policy_freq       = int(agent_cfg.get("policy_freq",       2))
        self.critic_grad_clip  = float(agent_cfg.get("critic_grad_clip",1.0))
        self.actor_grad_clip   = float(agent_cfg.get("actor_grad_clip", 1.0))   # FIX-7
        self.bc_alpha          = float(agent_cfg.get("bc_alpha",        2.5))   # TD3+BC 论文默认值
        self.target_q_clip     = float(agent_cfg.get("target_q_clip",   20.0))  # FIX-4（重新启用）
        self.use_reward_norm   = bool(agent_cfg.get("use_reward_norm",  False))  # 延伸-1（默认关）

        # ── Epsilon ──────────────────────────────────────────────────────────
        self.epsilon     = float(agent_cfg.get("epsilon_init",  1.0))
        self.epsilon_min = float(agent_cfg.get("epsilon_min",   0.05))
        self.epsilon_delta = float(agent_cfg.get("epsilon_delta", 2e-6))  # 每 step 衰减量

        # critic loss 类型
        self.critic_loss_type = str(agent_cfg.get("critic_loss_type", "huber")).lower()

        # ── 网络初始化 ───────────────────────────────────────────────────────
        self.actor        = opt_cuda(Actor(state_dim, action_dim, self.max_action, hidden_dim))
        self.target_actor = opt_cuda(Actor(state_dim, action_dim, self.max_action, hidden_dim))
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic        = opt_cuda(Critic(state_dim, action_dim, hidden_dim))
        self.target_critic = opt_cuda(Critic(state_dim, action_dim, hidden_dim))
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        # ── 优化器 ───────────────────────────────────────────────────────────
        lr_a = float(agent_cfg.get("lr_actor",  3e-4))
        lr_c = float(agent_cfg.get("lr_critic", 3e-4))
        self.optimizer_actor  = torch.optim.Adam(self.actor.parameters(),  lr=lr_a)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        # ── 回放池 ───────────────────────────────────────────────────────────
        buf_size = int(agent_cfg.get("buffer_size", 200_000))
        self.buffer = ReplayBuffer(buf_size, state_dim, action_dim)

        # ── 状态归一化（FIX-5：推理与训练共用同一归一化器）──────────────────
        self.state_norm = RunningMeanStd(shape=(state_dim,), warm_start=200, clip=10.0)

        # ── 奖励归一化（延伸-1，可选）────────────────────────────────────────
        self.reward_norm = RewardNormalizer(gamma=self.gamma) if self.use_reward_norm else None

        # ── 训练计数 ─────────────────────────────────────────────────────────
        self.total_it = 0

        # ── 缓存最近一次 loss（供日志使用）──────────────────────────────────
        self._last_actor_loss = 0.0
        self._last_bc_loss    = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────────────────

    def _device(self):
        return next(self.actor.parameters()).device

    # ──────────────────────────────────────────────────────────────────────────
    # 动作选择（FIX-1：act() 只返回纯净 Actor 输出，不注入探索噪声）
    # ──────────────────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray):
        """
        返回 Actor 网络的确定性输出（无噪声）。
        探索噪声由 TD3learn.py 的调用方负责叠加。

        Returns:
            actor_action (np.ndarray): Actor 网络输出，已 clip 到 max_action。
            base_action  (np.ndarray): 专家控制器输出（用于调用方决策 epsilon-greedy）。
        """
        with torch.no_grad():
            # 归一化状态（推理用）
            norm_s = self.state_norm.normalize(state)
            s_t    = np_to_tensor(norm_s.reshape(1, -1))
            actor_action = self.actor(s_t).cpu().numpy().flatten()

        # 专家动作（调用方用于 epsilon-greedy 切换与 BC 存储）
        if self.base_controller_func is not None:
            base_action = self.base_controller_func(state)
        else:
            base_action = np.zeros(self.action_dim, dtype=np.float32)

        return actor_action, base_action

    # ──────────────────────────────────────────────────────────────────────────
    # Epsilon 步进衰减（FIX-6：由调用方每 step 显式调用）
    # ──────────────────────────────────────────────────────────────────────────

    def step_epsilon(self):
        """每执行一个环境 step 后调用，线性衰减 epsilon。"""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_delta, self.epsilon_min)

    # ──────────────────────────────────────────────────────────────────────────
    # 经验存储（FIX-5：存归一化后的状态）
    # ──────────────────────────────────────────────────────────────────────────

    def remember(
        self,
        state:       np.ndarray,
        action:      np.ndarray,
        base_action: np.ndarray,
        next_state:  np.ndarray,
        reward:      float,
        done:        bool,
    ):
        """
        将一步转移存入回放池。
        state / next_state 在存入前先更新归一化器，然后归一化后存储（FIX-5）。
        """
        # 更新归一化统计（先更新 state）
        self.state_norm.update(state)

        norm_s  = self.state_norm.normalize(state)
        norm_ns = self.state_norm.normalize(next_state)

        # 可选奖励归一化（延伸-1）
        if self.reward_norm is not None:
            reward = self.reward_norm.normalize(reward, done)

        self.buffer.add(norm_s, action, base_action, norm_ns, reward, done)

    # ──────────────────────────────────────────────────────────────────────────
    # TD3 训练核心
    # ──────────────────────────────────────────────────────────────────────────

    def train(self, iterations: int = 1) -> TrainResult:
        """
        执行 iterations 次 TD3 梯度更新。

        Returns:
            TrainResult: 命名元组，包含 critic_loss / actor_loss / bc_loss /
                         q_pred / q_target（FIX-8：统一返回结构，buffer不足时返回零值）。
        """
        if self.buffer.size < self.batch_size:
            return ZERO_RESULT   # FIX-8

        device = self._device()

        total_lc     = 0.0
        total_q_pred = 0.0
        total_q_tgt  = 0.0

        for _ in range(iterations):
            self.total_it += 1

            # ── 采样 ──────────────────────────────────────────────────────
            (s, a, ba, ns, r, d) = self.buffer.sample(self.batch_size)

            # 转 Tensor（buffer 中已存归一化值，FIX-5）
            si  = np_to_tensor(s,  device).float()   # (B, state_dim)
            ai  = np_to_tensor(a,  device).float()   # (B, action_dim)
            bai = np_to_tensor(ba, device).float()   # (B, action_dim)，BC 用
            nsi = np_to_tensor(ns, device).float()   # (B, state_dim)
            ri  = np_to_tensor(r,  device).float().view(-1, 1)
            di  = np_to_tensor(d,  device).float().view(-1, 1)

            # ================================================================
            # 【TD3 机制 1】Target Policy Smoothing
            # ================================================================
            with torch.no_grad():
                # 目标策略噪声（按动作维度缩放）
                noise_std  = self.policy_noise * self.target_actor.max_action   # shape (action_dim,) broadcast
                noise_clip = self.noise_clip   * self.target_actor.max_action
                noise = (torch.randn_like(ai) * noise_std).clamp(-noise_clip, noise_clip)

                next_a = (self.target_actor(nsi) + noise).clamp(
                    -self.target_actor.max_action,
                     self.target_actor.max_action
                )

                # ============================================================
                # 【TD3 机制 2】Clipped Double-Q
                # ============================================================
                tQ1, tQ2 = self.target_critic(nsi, next_a)
                target_Q  = torch.min(tQ1, tQ2)

                # ── Bellman 目标 ─────────────────────────────────────────────
                yi = ri + self.gamma * (1.0 - di) * target_Q

                # FIX-4：重新启用 target Q clip，防止 bootstrap 无限增长
                yi = yi.clamp(-self.target_q_clip, self.target_q_clip)

            # ================================================================
            # Critic 更新（Huber Loss）
            # ================================================================
            curr_Q1, curr_Q2 = self.critic(si, ai)
            total_q_pred += curr_Q1.mean().item()
            total_q_tgt  += yi.mean().item()

            if self.critic_loss_type == "mse":
                lc = F.mse_loss(curr_Q1, yi) + F.mse_loss(curr_Q2, yi)
            else:
                lc = F.huber_loss(curr_Q1, yi) + F.huber_loss(curr_Q2, yi)

            if not torch.isfinite(lc):
                # 跳过 NaN/Inf 样本，不让梯度污染网络
                continue

            self.optimizer_critic.zero_grad()
            lc.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
            self.optimizer_critic.step()
            total_lc += lc.item()

            # ================================================================
            # 【TD3 机制 3】Delayed Actor + soft update
            # ================================================================
            if self.total_it % self.policy_freq == 0:

                self.optimizer_actor.zero_grad()

                a_pred = self.actor(si)
                q_pred = self.critic.Q1(si, a_pred)

                if self.behavior_clone:
                    # ── TD3+BC（FIX-3：正确的 lmbda 公式与范围限制）────────
                    # BC Loss：MSE 到专家动作
                    lbc = F.mse_loss(a_pred, bai)

                    # 动态 lambda = alpha / E[|Q|]
                    # detach() 防止梯度流回 Critic
                    q_abs_mean = q_pred.abs().mean().detach()
                    lmbda = self.bc_alpha / (q_abs_mean + 1e-5)

                    # FIX-3：clamp 到合理范围，防止训练早期 lambda 飙升
                    # 下限 0.1 保证 Q-gradient 始终存在；上限 10.0 防 BC 完全主导
                    lmbda = lmbda.clamp(0.1, 10.0)

                    # Actor Loss = BC term + 归一化的 Q maximization
                    # 注意：lmbda 乘在 q_pred 上而非 lbc 上（论文原文设计）
                    la = lbc - lmbda * q_pred.mean()

                    self._last_bc_loss = lbc.item()

                else:
                    la = -q_pred.mean()
                    self._last_bc_loss = 0.0

                la.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)   # FIX-7
                self.optimizer_actor.step()
                self._last_actor_loss = la.item()

                # Soft update（Actor 延迟更新后同步 target）
                soft_update(self.target_actor,  self.actor,  self.tau)
                soft_update(self.target_critic, self.critic, self.tau)

        n = max(iterations, 1)
        return TrainResult(
            critic_loss = total_lc     / n,
            actor_loss  = self._last_actor_loss,
            bc_loss     = self._last_bc_loss,
            q_pred      = total_q_pred / n,
            q_target    = total_q_tgt  / n,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 模型保存 / 加载
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        """保存完整训练状态（state_dict 方式，避免 weights_only 安全警告）。"""
        torch.save({
            "actor":       self.actor.state_dict(),
            "target_actor":self.target_actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "target_critic":self.target_critic.state_dict(),
            "opt_actor":   self.optimizer_actor.state_dict(),
            "opt_critic":  self.optimizer_critic.state_dict(),
            "epsilon":     self.epsilon,
            "total_it":    self.total_it,
        }, path)

    def load(self, path: str, map_location=None):
        """加载 checkpoint。"""
        ck = torch.load(path, map_location=map_location or self._device())
        self.actor.load_state_dict(ck["actor"])
        self.target_actor.load_state_dict(ck["target_actor"])
        self.critic.load_state_dict(ck["critic"])
        self.target_critic.load_state_dict(ck["target_critic"])
        self.optimizer_actor.load_state_dict(ck["opt_actor"])
        self.optimizer_critic.load_state_dict(ck["opt_critic"])
        self.epsilon   = ck.get("epsilon",  self.epsilon)
        self.total_it  = ck.get("total_it", self.total_it)


# ==============================================================================
# 纯 RL Agent（无专家依赖，继承 WBAgent 并关闭 BC）
# ==============================================================================

class PureRLAgent(WBAgent):
    """纯 TD3，行为克隆和专家控制器完全禁用。"""

    def __init__(self, log_dir, state_dim, action_dim, config=None, base_controller_func=None):
        super().__init__(log_dir, state_dim, action_dim, config, base_controller_func=None)
        self.behavior_clone        = False
        self.epsilon               = 0.0
        self.epsilon_min           = 0.0
        self.epsilon_delta         = 0.0
        self.base_controller_func  = None

    def act(self, state: np.ndarray):
        with torch.no_grad():
            norm_s = self.state_norm.normalize(state)
            s_t    = np_to_tensor(norm_s.reshape(1, -1))
            action = self.actor(s_t).cpu().numpy().flatten()
        base = np.zeros(self.action_dim, dtype=np.float32)
        return action, base