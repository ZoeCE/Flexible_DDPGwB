# ==============================================================================
# agent.py — PPO + BC 与 TD3 + BC 双框架 Agent（关节空间动作版）
#
# 架构设计：
#
# [AGT-1] 动作空间：7D 关节角目标（替代原 6D 末端加速度）
#   - Actor 输出 7 个关节角目标值（在关节限位范围内）
#   - BC 监督信号：JointSpaceExpert 生成的 IK 求解关节角
#   - 动作 clamp 到每个关节的物理极限范围
#
# [AGT-2] PPO Agent（主框架）
#   核心改进：
#   a) GAE（Generalized Advantage Estimation）优势函数估计
#   b) 动作分布：有界高斯（Gaussian + tanh squashing），
#      保证输出在关节限位内
#   c) BC 系数退火：训练初期 BC 主导（帮助 Actor 进入合理初始区域），
#      随训练进行逐渐让 RL 主导
#   d) 观测 Running Normalization（在线更新，存归一化值到 RolloutBuffer）
#   e) 梯度裁剪 + entropy 正则（PPO 的标准稳定化手段）
#
# [AGT-3] TD3 Agent（保留，用于对比实验）
#   与上一版 agent.py 相同，动作空间已更新为 7D 关节角，
#   max_action 使用关节限位而非加速度边界。
#
# [AGT-4] RolloutBuffer（PPO 专用，替代 TD3 的 ReplayBuffer）
#   - 存储 rollout 期间的 (obs, action, reward, done, value, logprob)
#   - compute_returns_advantages() 在 collect 完成后一次性计算 GAE
#   - 支持 mini-batch 采样（shuffle + split）
# ==============================================================================
 
# ==============================================================================
# agent.py — PPO + BC 与 TD3 + BC 双框架 Agent（关节空间动作版，修复版）
#
# ==============================================================================
# agent.py — delta 关节角动作空间版（完全重写修复版）
#
# ══════════════════════════════════════════════════════════════════════════════
# 核心架构变更：绝对关节角 → 增量关节角（delta-q）
# ══════════════════════════════════════════════════════════════════════════════
#
# [DELTA-1] 为何改用 delta-q（增量动作）
#   原版问题：Actor 输出绝对关节角 q_target，范围约 ±3 rad。
#   a) BC Loss = MSE(q_actor, q_expert)，两者都在 ±3rad 内，初期误差 ~1-3 rad²。
#      但 policy gradient 方向与 BC 方向往往矛盾（Q 刚开始不可信），
#      导致 loss 只有数值上的下降，策略实际行为不变——这正是 BC loss
#      不下降的根本原因。
#   b) 策略初始化接近关节空间中心（mean_head gain=0.01），
#      对应绝对关节角 ≈ 0，但实际运动需要 q ≈ init_q（非零），
#      距离 BC 目标非常远，初期梯度几乎全被 BC 占据但仍无效。
#
#   delta-q 的优势：
#   a) Actor 输出 Δq，范围 ±dq_max（如 ±0.1 rad/step），
#      初始化接近零意味着"不动"，这是合理且安全的初始策略。
#   b) BC 目标变为 Δq_expert = q_expert_next - q_current，量级小（±0.1），
#      bc_loss 的 MSE 初始就小，梯度有效，策略更快被引导。
#   c) 探索噪声直接加在 Δq 上，物理含义清晰（每步最多移动 dq_max）。
#   d) 关节角越界由 env.step 中的 clamp 处理（q + Δq clamp 到关节限位）。
#
# [DELTA-2] PPO Actor 架构变更
#   输出：Δq ∈ [-dq_max, +dq_max]^7（通过 tanh × dq_max 保证范围）
#   log_prob：标准 tanh 高斯对数概率
#   BC Loss：MSE(Δq_actor, Δq_expert)，Δq_expert 由训练循环提供
#
# [DELTA-3] TD3 Actor 架构变更
#   同上，输出 Δq，tanh × dq_max。
#   target policy smoothing 噪声直接加在 Δq 上（量级统一）。
#
# [DELTA-4] PPO 的多个 Bug 修复
#   [BUG-P1] tanh 映射公式颠倒（scale/offset 互换）
#   [BUG-P2] get_minibatches 多 epoch 时原地修改 advantages（破坏后续 epoch）
#   [BUG-P3] GAE 使用 dones[t+1] 而非 dones[t]
#   [BUG-P4] KL 早停在 epoch 循环外计算（实际不起作用）
#   [BUG-P5] BC Loss 在 PPO total_loss 中与 policy_loss 量级严重不匹配
#            → bc_coef 需要足够大才能压住 policy_loss
#   [BUG-P6] Actor 和 Critic 共用同一优化器，Critic 学习过快时会
#            通过共享梯度干扰 Actor
#
# [DELTA-5] TD3 的多个 Bug 修复
#   [BUG-T1] TD3Actor 中 action_scale/offset 互换（与 PPO 同样的 tanh 错误）
#   [BUG-T2] target policy smoothing 用 action_offset（半宽）而非 dq_max
#   [BUG-T3] TD3 训练时 bc_target 存的是绝对关节角而非 delta
#            → 需要在 train() 中把 bc 还原为 delta（已在 learn 循环处理）
#
# ══════════════════════════════════════════════════════════════════════════════

# ==============================================================================
# agent.py — PPO + BC 与 TD3 + BC 双框架 Agent（关节空间动作版）
#
# 架构设计：
#
# [AGT-1] 动作空间：7D 关节角目标（替代原 6D 末端加速度）
#   - Actor 输出 7 个关节角目标值（在关节限位范围内）
#   - BC 监督信号：JointSpaceExpert 生成的 IK 求解关节角
#   - 动作 clamp 到每个关节的物理极限范围
#
# [AGT-2] PPO Agent（主框架）
#   核心改进：
#   a) GAE（Generalized Advantage Estimation）优势函数估计
#   b) 动作分布：有界高斯（Gaussian + tanh squashing），
#      保证输出在关节限位内
#   c) BC 系数退火：训练初期 BC 主导（帮助 Actor 进入合理初始区域），
#      随训练进行逐渐让 RL 主导
#   d) 观测 Running Normalization（在线更新，存归一化值到 RolloutBuffer）
#   e) 梯度裁剪 + entropy 正则（PPO 的标准稳定化手段）
#
# [AGT-3] TD3 Agent（保留，用于对比实验）
#   与上一版 agent.py 相同，动作空间已更新为 7D 关节角，
#   max_action 使用关节限位而非加速度边界。
#
# [AGT-4] RolloutBuffer（PPO 专用，替代 TD3 的 ReplayBuffer）
#   - 存储 rollout 期间的 (obs, action, reward, done, value, logprob)
#   - compute_returns_advantages() 在 collect 完成后一次性计算 GAE
#   - 支持 mini-batch 采样（shuffle + split）
# ==============================================================================
 
# ==============================================================================
# agent.py — PPO + BC 与 TD3 + BC 双框架 Agent（关节空间动作版，修复版）
#
# ==============================================================================
# agent.py — delta 关节角动作空间版（完全重写修复版）
#
# ══════════════════════════════════════════════════════════════════════════════
# 核心架构变更：绝对关节角 → 增量关节角（delta-q）
# ══════════════════════════════════════════════════════════════════════════════
#
# [DELTA-1] 为何改用 delta-q（增量动作）
#   原版问题：Actor 输出绝对关节角 q_target，范围约 ±3 rad。
#   a) BC Loss = MSE(q_actor, q_expert)，两者都在 ±3rad 内，初期误差 ~1-3 rad²。
#      但 policy gradient 方向与 BC 方向往往矛盾（Q 刚开始不可信），
#      导致 loss 只有数值上的下降，策略实际行为不变——这正是 BC loss
#      不下降的根本原因。
#   b) 策略初始化接近关节空间中心（mean_head gain=0.01），
#      对应绝对关节角 ≈ 0，但实际运动需要 q ≈ init_q（非零），
#      距离 BC 目标非常远，初期梯度几乎全被 BC 占据但仍无效。
#
#   delta-q 的优势：
#   a) Actor 输出 Δq，范围 ±dq_max（如 ±0.1 rad/step），
#      初始化接近零意味着"不动"，这是合理且安全的初始策略。
#   b) BC 目标变为 Δq_expert = q_expert_next - q_current，量级小（±0.1），
#      bc_loss 的 MSE 初始就小，梯度有效，策略更快被引导。
#   c) 探索噪声直接加在 Δq 上，物理含义清晰（每步最多移动 dq_max）。
#   d) 关节角越界由 env.step 中的 clamp 处理（q + Δq clamp 到关节限位）。
#
# [DELTA-2] PPO Actor 架构变更
#   输出：Δq ∈ [-dq_max, +dq_max]^7（通过 tanh × dq_max 保证范围）
#   log_prob：标准 tanh 高斯对数概率
#   BC Loss：MSE(Δq_actor, Δq_expert)，Δq_expert 由训练循环提供
#
# [DELTA-3] TD3 Actor 架构变更
#   同上，输出 Δq，tanh × dq_max。
#   target policy smoothing 噪声直接加在 Δq 上（量级统一）。
#
# [DELTA-4] PPO 的多个 Bug 修复
#   [BUG-P1] tanh 映射公式颠倒（scale/offset 互换）
#   [BUG-P2] get_minibatches 多 epoch 时原地修改 advantages（破坏后续 epoch）
#   [BUG-P3] GAE 使用 dones[t+1] 而非 dones[t]
#   [BUG-P4] KL 早停在 epoch 循环外计算（实际不起作用）
#   [BUG-P5] BC Loss 在 PPO total_loss 中与 policy_loss 量级严重不匹配
#            → bc_coef 需要足够大才能压住 policy_loss
#   [BUG-P6] Actor 和 Critic 共用同一优化器，Critic 学习过快时会
#            通过共享梯度干扰 Actor
#
# [DELTA-5] TD3 的多个 Bug 修复
#   [BUG-T1] TD3Actor 中 action_scale/offset 互换（与 PPO 同样的 tanh 错误）
#   [BUG-T2] target policy smoothing 用 action_offset（半宽）而非 dq_max
#   [BUG-T3] TD3 训练时 bc_target 存的是绝对关节角而非 delta
#            → 需要在 train() 中把 bc 还原为 delta（已在 learn 循环处理）
#
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import namedtuple
from typing import Tuple, List

from config import DEFAULT_CONFIG


# ==============================================================================
# 通用工具
# ==============================================================================

def opt_cuda(t, device=None):
    if torch.cuda.is_available():
        return t.to(device) if device else t.cuda()
    return t

def np_to_tensor(n, device=None):
    return opt_cuda(torch.as_tensor(n, dtype=torch.float32), device)

def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1 - tau) + p.data * tau)

def orthogonal_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


# ==============================================================================
# 在线状态归一化
# ==============================================================================

class RunningMeanStd:
    def __init__(self, shape, warm_start=200, clip=10.0):
        self.n    = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S    = np.zeros(shape, dtype=np.float64)
        self.warm_start = warm_start
        self.clip = clip

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).flatten()
        self.n += 1
        if self.n == 1:
            self.mean = x.copy(); self.S = np.zeros_like(x)
        else:
            old = self.mean.copy()
            self.mean = old + (x - old) / self.n
            self.S    = self.S + (x - old) * (x - self.mean)

    @property
    def var(self): return self.S / max(self.n - 1, 1)
    @property
    def std(self): return np.sqrt(self.var + 1e-8)

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.n < self.warm_start:
            return x
        return np.clip(
            (x - self.mean.astype(np.float32)) / self.std.astype(np.float32),
            -self.clip, self.clip
        )

    def state_dict(self):
        return {"n": self.n, "mean": self.mean.copy(), "S": self.S.copy()}

    def load_state_dict(self, d):
        self.n = d["n"]; self.mean = d["mean"].copy(); self.S = d["S"].copy()


# ==============================================================================
# MLP 构建工具
# ==============================================================================

def build_mlp(in_dim, hidden_dim, n_hidden, out_dim,
              activation=nn.ReLU, last_gain=1.0):
    """n_hidden 个隐层 + 1 输出层。"""
    layers: List[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        lin = nn.Linear(d, hidden_dim)
        orthogonal_init(lin, gain=np.sqrt(2))
        layers += [lin, activation()]
        d = hidden_dim
    out = nn.Linear(d, out_dim)
    orthogonal_init(out, gain=last_gain)
    layers.append(out)
    return nn.Sequential(*layers)


# ==============================================================================
# PPO Actor — 输出 delta-q（增量关节角）
# ==============================================================================

class PPOActor(nn.Module):
    """
    策略网络，输出 Δq ∈ [-dq_max, +dq_max]^7。

    [DELTA-2] 动作 = tanh(u) × dq_max，其中 dq_max 从 config 读取。
    初始化时 mean_head 接近零 → Δq ≈ 0 → 机械臂不动 → 安全初始策略。
    """

    def __init__(self, state_dim: int, action_dim: int,
                 dq_max: np.ndarray,
                 hidden_dim: int = 256, n_layers: int = 2,
                 log_std_init: float = -1.0,
                 log_std_min: float = -4.0, log_std_max: float = 0.5):
        super().__init__()

        # [DELTA-2] dq_max 是增量上限（正数），tanh 直接乘以它
        self.register_buffer('dq_max',
                             torch.tensor(dq_max, dtype=torch.float32))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # backbone：n_layers 个隐层
        layers: List[nn.Module] = []
        d = state_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            orthogonal_init(lin, gain=np.sqrt(2))
            layers += [lin, nn.ReLU()]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # 均值头：小增益 → 初始 Δq ≈ 0
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.mean_head, gain=0.01)

        # 状态无关的 log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def _dist(self, s):
        feat     = self.backbone(s)
        mean_raw = self.mean_head(feat)
        log_std  = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std      = log_std.exp().expand_as(mean_raw)
        return mean_raw, std

    def get_action(self, s, deterministic=False):
        mean_raw, std = self._dist(s)
        dist = Normal(mean_raw, std)
        u    = mean_raw if deterministic else dist.rsample()

        u_tanh  = torch.tanh(u)
        delta_q = u_tanh * self.dq_max  # Δq ∈ [-dq_max, +dq_max]

        # log_prob（含 tanh Jacobian）
        log_prob  = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)

        entropy = dist.entropy().sum(-1)
        return delta_q, log_prob, entropy

    def evaluate_actions(self, s, delta_q_taken):
        """PPO 更新时计算已执行 delta_q 的 log_prob。"""
        mean_raw, std = self._dist(s)

        # 反映射：delta_q → u_tanh → u
        u_tanh = (delta_q_taken / (self.dq_max + 1e-8)).clamp(-1 + 1e-6, 1 - 1e-6)
        u      = torch.atanh(u_tanh)

        dist     = Normal(mean_raw, std)
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy

    # ==========================================================================
    # [BC-FIX-1] BC 专用前向：在 u 空间（tanh 前）做 BC，避免 tanh 饱和
    # ==========================================================================
    def bc_forward(self, s, delta_q_target):
        """
        BC 专用前向，返回 (bc_loss_u, bc_loss_dq, delta_q_pred)。

        核心思想：不在 Δq 空间做 MSE（会导致 tanh 饱和，梯度消失），
        而是在 u 空间（tanh 前的线性空间）做 MSE。
        专家的 Δq_target → atanh(Δq_target/dq_max) = u_target
        Actor 的 mean_raw 也是 u 空间的值
        MSE(mean_raw, u_target) 梯度对 mean_head 参数畅通无阻。

        额外：保留一个 Δq 空间的辅助 loss，用于监控或微调。
        """
        mean_raw, std = self._dist(s)  # mean_raw 是 u 空间的值

        # 将专家 Δq 映射到 u 空间（atanh）
        u_tanh_target = (delta_q_target / (self.dq_max + 1e-8)).clamp(-0.999, 0.999)
        u_target      = torch.atanh(u_tanh_target)

        # u 空间 MSE（主 BC loss，梯度健康）
        bc_loss_u  = F.mse_loss(mean_raw, u_target)

        # Δq 空间 MSE（辅助 loss，用于直接对齐输出）
        delta_q_pred = torch.tanh(mean_raw) * self.dq_max
        bc_loss_dq   = F.mse_loss(delta_q_pred, delta_q_target)

        return bc_loss_u, bc_loss_dq, delta_q_pred


# ==============================================================================
# PPO Critic
# ==============================================================================

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, n_layers=2):
        super().__init__()
        layers: List[nn.Module] = []
        d = state_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            orthogonal_init(lin, gain=np.sqrt(2))
            layers += [lin, nn.ReLU()]
            d = hidden_dim
        out = nn.Linear(hidden_dim, 1)
        orthogonal_init(out, gain=1.0)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)


# ==============================================================================
# RolloutBuffer
# ==============================================================================

class RolloutBuffer:
    def __init__(self, n_steps, state_dim, action_dim, device):
        self.n_steps    = n_steps
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = device
        self.clear()

    def clear(self):
        self.obs        = np.zeros((self.n_steps, self.state_dim),  np.float32)
        self.actions    = np.zeros((self.n_steps, self.action_dim), np.float32)  # Δq
        self.bc_targets = np.zeros((self.n_steps, self.action_dim), np.float32)  # Δq_expert
        self.rewards    = np.zeros(self.n_steps, np.float32)
        self.dones      = np.zeros(self.n_steps, np.float32)
        self.values     = np.zeros(self.n_steps, np.float32)
        self.log_probs  = np.zeros(self.n_steps, np.float32)
        self.advantages = np.zeros(self.n_steps, np.float32)
        self.returns    = np.zeros(self.n_steps, np.float32)
        self.ptr  = 0
        self.full = False

    def add(self, obs, delta_q, bc_delta_q, reward, done, value, log_prob):
        i = self.ptr
        self.obs[i]        = obs
        self.actions[i]    = delta_q
        self.bc_targets[i] = bc_delta_q
        self.rewards[i]    = reward
        self.dones[i]      = float(done)
        self.values[i]     = value
        self.log_probs[i]  = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """GAE。[BUG-P3 修复] 使用 dones[t]（不是 dones[t+1]）。"""
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val     = last_value if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]          # [BUG-P3]
            delta        = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae     = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size, normalize_adv=True):
        """[BUG-P2 修复] 归一化用副本，不修改 self.advantages。"""
        assert self.full
        indices = np.random.permutation(self.n_steps)
        adv = self.advantages.copy()
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, self.n_steps, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                np_to_tensor(self.obs[idx],        self.device),
                np_to_tensor(self.actions[idx],    self.device),
                np_to_tensor(self.bc_targets[idx], self.device),
                np_to_tensor(self.returns[idx],    self.device).view(-1, 1),
                np_to_tensor(adv[idx],             self.device),
                np_to_tensor(self.log_probs[idx],  self.device),
            )


# ==============================================================================
# PPO Agent
# ==============================================================================

PPOTrainResult = namedtuple("PPOTrainResult", [
    "policy_loss", "value_loss", "entropy_loss", "bc_loss",
    "approx_kl", "clip_fraction", "total_loss",
])
PPO_ZERO = PPOTrainResult(0., 0., 0., 0., 0., 0., 0.)


class PPOAgent:
    """
    PPO + BC Agent（delta-q 动作空间）。

    动作语义：Δq，每步关节角变化量，由 env 累加到当前关节角后执行。
    BC 目标：Δq_expert = q_expert_t+1 - q_current_t（由训练循环计算并传入）。
    """

    def __init__(self, log_dir, state_dim, action_dim, config=None, expert=None):
        if config is None:
            config = DEFAULT_CONFIG

        self.config     = config
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.expert     = expert

        cfg = config["ppo_agent"]
        sp  = config["space"]

        # [DELTA-2] 增量上限：每步最多移动 dq_max rad
        self.dq_max = np.array(sp.get("dq_max", [0.1] * action_dim), dtype=np.float32)

        # 关节限位（用于 env 侧 clamp，agent 不直接 clamp）
        self.q_low  = np.array(sp["action_space_low"],  dtype=np.float32)
        self.q_high = np.array(sp["action_space_high"], dtype=np.float32)

        # 超参数
        self.gamma           = float(cfg["gamma"])
        self.gae_lambda      = float(cfg["gae_lambda"])
        self.clip_eps        = float(cfg["clip_eps"])
        self.value_loss_coef = float(cfg["value_loss_coef"])
        self.entropy_coef    = float(cfg["entropy_coef"])
        self.max_grad_norm   = float(cfg["max_grad_norm"])
        self.n_steps         = int(cfg["n_steps"])
        self.n_epochs        = int(cfg["n_epochs"])
        self.batch_size      = int(cfg["batch_size"])
        self.norm_adv        = bool(cfg["normalize_advantages"])
        self.target_kl       = float(cfg.get("target_kl", 0.05))

        # BC 退火
        self.behavior_clone  = bool(cfg["behavior_clone"])
        self.bc_coef         = float(cfg["bc_coef_init"])
        self.bc_coef_init    = float(cfg["bc_coef_init"])
        self.bc_coef_final   = float(cfg["bc_coef_final"])
        self.bc_anneal_steps = int(cfg["bc_anneal_steps"])
        self.bc_loss_type    = str(cfg.get("bc_loss_type", "mse"))

        # 观测归一化
        self.use_obs_norm = bool(cfg["use_obs_norm"])
        self.obs_norm     = RunningMeanStd(shape=(state_dim,),
                                           warm_start=200,
                                           clip=float(cfg["obs_norm_clip"]))

        # 设备
        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))

        hidden_dim = int(cfg["hidden_dim"])
        n_layers   = int(cfg["n_layers"])

        self.actor = PPOActor(
            state_dim, action_dim, self.dq_max,
            hidden_dim=hidden_dim, n_layers=n_layers,
            log_std_init=float(cfg.get("log_std_init", -1.0)),
            log_std_min=float(cfg.get("log_std_min", -4.0)),
            log_std_max=float(cfg.get("log_std_max", 0.5)),
        ).to(self.device)

        self.critic = PPOCritic(state_dim, hidden_dim, n_layers).to(self.device)

        # [BUG-P6 修复] Actor 和 Critic 分开优化器
        self.opt_actor  = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg["lr_actor"]), eps=1e-5)
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg.get("lr_critic", cfg["lr_actor"])), eps=1e-5)

        self.buffer      = RolloutBuffer(self.n_steps, state_dim, action_dim, self.device)
        self.total_steps = 0
        self._last_result = PPO_ZERO

    # ──────────────────────────────────────────────────────────────────────────

    def normalize_obs(self, obs, update=True):
        if self.use_obs_norm:
            if update:
                self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        """返回 (delta_q, log_prob, value)。delta_q ∈ [-dq_max, +dq_max]。"""
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        delta_q, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
        value = self.critic(s)

        dq_np  = delta_q.cpu().numpy().flatten()
        lp_np  = log_prob.cpu().item()
        val_np = value.cpu().item()

        # 安全 clamp（理论上 tanh 已保证，防浮点精度）
        dq_np = np.clip(dq_np, -self.dq_max, self.dq_max)
        return dq_np, lp_np, val_np

    def _update_bc_coef(self):
        if not self.behavior_clone or self.bc_anneal_steps <= 0:
            return
        frac = min(self.total_steps / self.bc_anneal_steps, 1.0)
        self.bc_coef = self.bc_coef_init + frac * (self.bc_coef_final - self.bc_coef_init)

    def update(self):
        if not self.buffer.full:
            return PPO_ZERO

        self._update_bc_coef()

        total_pl = total_vl = total_el = total_bl = 0.0
        total_kl = total_cf = 0.0
        n_updates = 0
        stop_early = False

        for epoch in range(self.n_epochs):
            if stop_early:
                break
            for batch in self.buffer.get_minibatches(self.batch_size, self.norm_adv):
                obs_b, dq_b, bc_b, ret_b, adv_b, old_lp_b = batch

                new_lp, entropy = self.actor.evaluate_actions(obs_b, dq_b)
                value            = self.critic(obs_b)

                # PPO clip
                ratio       = (new_lp - old_lp_b).exp()
                surr1       = ratio * adv_b
                surr2       = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss   = F.huber_loss(value, ret_b)

                # Entropy
                entropy_loss = -entropy.mean()

                # BC Loss 占位（仅用于日志显示 rollout 样本上的 BC 误差）
                if self.behavior_clone and self.bc_coef > 0:
                    with torch.no_grad():
                        _, bc_loss_dq_mon, _ = self.actor.bc_forward(obs_b, bc_b)
                        bc_loss = bc_loss_dq_mon
                else:
                    bc_loss = torch.zeros(1, device=self.device)

                # [BUG-P6 修复] 分开更新 Actor 和 Critic
                # Critic 更新
                critic_total = self.value_loss_coef * value_loss
                self.opt_critic.zero_grad()
                critic_total.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                # Actor 更新（重新前向，因为 Critic 已更新但 Actor 参数未变）
                # 重新计算 log_prob 以匹配已执行动作（用于 PPO ratio）
                new_lp2_eval, entropy2_eval = self.actor.evaluate_actions(obs_b, dq_b)
                ratio2       = (new_lp2_eval - old_lp_b).exp()
                surr1_2      = ratio2 * adv_b
                surr2_2      = ratio2.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss2 = -torch.min(surr1_2, surr2_2).mean()
                entropy_loss2 = -entropy2_eval.mean()

                if self.behavior_clone and self.bc_coef > 0:
                    # [BC-FIX-1] 用 u 空间 BC loss，避免 tanh 饱和区梯度消失
                    # 主 loss: u 空间 (0.7)，辅助 loss: Δq 空间 (0.3)
                    bc_loss_u, bc_loss_dq, _ = self.actor.bc_forward(obs_b, bc_b)
                    bc_loss2 = 0.7 * bc_loss_u + 0.3 * bc_loss_dq
                else:
                    bc_loss2 = torch.zeros(1, device=self.device)

                actor_total = (policy_loss2
                               + self.entropy_coef * entropy_loss2
                               + self.bc_coef * bc_loss2)
                self.opt_actor.zero_grad()
                actor_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                with torch.no_grad():
                    approx_kl = (old_lp_b - new_lp2_eval).mean().abs().item()
                    clip_frac = ((ratio2 - 1).abs() > self.clip_eps).float().mean().item()

                total_pl += policy_loss2.item()
                total_vl += value_loss.item()
                total_el += entropy_loss2.item()
                total_bl += bc_loss2.item() if self.behavior_clone else 0.0
                total_kl += approx_kl
                total_cf += clip_frac
                n_updates += 1

                # [BUG-P4 修复] 在 minibatch 级别检查 KL
                if approx_kl > self.target_kl:
                    stop_early = True
                    break

        n = max(n_updates, 1)
        result = PPOTrainResult(
            policy_loss   = total_pl / n,
            value_loss    = total_vl / n,
            entropy_loss  = total_el / n,
            bc_loss       = total_bl / n,
            approx_kl     = total_kl / n,
            clip_fraction = total_cf / n,
            total_loss    = (total_pl + total_vl + total_el + total_bl) / n,
        )
        self._last_result = result
        self.buffer.clear()
        return result

    def save(self, path):
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "opt_actor":   self.opt_actor.state_dict(),
            "opt_critic":  self.opt_critic.state_dict(),
            "total_steps": self.total_steps,
            "bc_coef":     self.bc_coef,
            "obs_norm":    self.obs_norm.state_dict(),
        }, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if "opt_actor" in ck:
            self.opt_actor.load_state_dict(ck["opt_actor"])
        if "opt_critic" in ck:
            self.opt_critic.load_state_dict(ck["opt_critic"])
        self.total_steps = ck.get("total_steps", 0)
        self.bc_coef     = ck.get("bc_coef", self.bc_coef)
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])


# ==============================================================================
# TD3 Actor — delta-q
# ==============================================================================

class TD3Actor(nn.Module):
    """
    TD3 确定性策略，输出 Δq ∈ [-dq_max, +dq_max]。
    [DELTA-3][BUG-T1] 修复：tanh × dq_max，不再是 tanh×half + center。
    """
    def __init__(self, state_dim, action_dim, dq_max, hidden_dim=256):
        super().__init__()
        self.register_buffer('dq_max', torch.tensor(dq_max, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                orthogonal_init(m, gain=np.sqrt(2))
        orthogonal_init(self.net[-2], gain=0.01)

    def forward(self, s):
        return self.net(s) * self.dq_max


class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        in_dim = state_dim + action_dim
        def make_q():
            net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for m in net:
                if isinstance(m, nn.Linear):
                    orthogonal_init(m)
            return net
        self.q1, self.q2 = make_q(), make_q()

    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, s, a):
        return self.q1(torch.cat([s, a], -1))


class ReplayBuffer:
    """
    存储：(norm_s, delta_q, delta_q_expert, norm_ns, r, done)
    [DELTA-3] action 和 bc_target 都是 delta_q，量级统一。
    """
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.state      = np.zeros((max_size, state_dim),  np.float32)
        self.next_state = np.zeros((max_size, state_dim),  np.float32)
        self.action     = np.zeros((max_size, action_dim), np.float32)
        self.bc_target  = np.zeros((max_size, action_dim), np.float32)
        self.reward     = np.zeros((max_size, 1),          np.float32)
        self.done       = np.zeros((max_size, 1),          np.float32)

    def add(self, s, dq, bc_dq, ns, r, d):
        i = self.ptr
        self.state[i]     = s;  self.next_state[i] = ns
        self.action[i]    = dq; self.bc_target[i]  = bc_dq
        self.reward[i]    = r;  self.done[i]       = d
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (self.state[idx], self.action[idx], self.bc_target[idx],
                self.next_state[idx], self.reward[idx], self.done[idx])


TD3Result = namedtuple("TD3Result", ["critic_loss","actor_loss","bc_loss","q_pred","q_target"])
TD3_ZERO  = TD3Result(0.,0.,0.,0.,0.)


class TD3Agent:
    """
    TD3 + BC Agent（delta-q 动作空间）。
    [DELTA-3] 动作 = Δq，target smoothing 噪声直接加在 Δq 上。
    """

    def __init__(self, log_dir, state_dim, action_dim, config=None, expert=None):
        if config is None:
            config = DEFAULT_CONFIG
        self.config     = config
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.expert     = expert

        cfg = config["td3_agent"]
        sp  = config["space"]
        self.dq_max  = np.array(sp.get("dq_max", [0.1]*action_dim), dtype=np.float32)
        self.q_low   = np.array(sp["action_space_low"],  dtype=np.float32)
        self.q_high  = np.array(sp["action_space_high"], dtype=np.float32)

        self.batch_size       = int(cfg["batch_size"])
        self.gamma            = float(cfg["gamma"])
        self.tau              = float(cfg["tau"])
        self.policy_noise     = float(cfg["policy_noise"])    # 相对 dq_max 的噪声比例
        self.noise_clip       = float(cfg["noise_clip"])
        self.policy_freq      = int(cfg["policy_freq"])
        self.critic_grad_clip = float(cfg["critic_grad_clip"])
        self.actor_grad_clip  = float(cfg["actor_grad_clip"])
        self.target_q_clip    = float(cfg["target_q_clip"])
        self.bc_alpha         = float(cfg["bc_alpha"])
        self.behavior_clone   = bool(cfg["behavior_clone"])
        self.critic_loss_type = str(cfg.get("critic_loss_type", "huber"))

        self.epsilon       = float(cfg["epsilon_init"])
        self.epsilon_min   = float(cfg["epsilon_min"])
        self.epsilon_delta = float(cfg["epsilon_delta"])

        self.obs_norm = RunningMeanStd(shape=(state_dim,), warm_start=200)

        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))

        hidden = int(cfg["hidden_dim"])
        self.actor        = TD3Actor(state_dim, action_dim, self.dq_max, hidden).to(self.device)
        self.target_actor = TD3Actor(state_dim, action_dim, self.dq_max, hidden).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()

        self.critic        = TD3Critic(state_dim, action_dim, hidden).to(self.device)
        self.target_critic = TD3Critic(state_dim, action_dim, hidden).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=float(cfg["lr_actor"]))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=float(cfg["lr_critic"]))

        self.buffer   = ReplayBuffer(int(cfg["buffer_size"]), state_dim, action_dim)
        self.total_it = 0
        self._last_al = self._last_bl = 0.0

    def normalize_obs(self, obs, update=True):
        if update:
            self.obs_norm.update(obs)
        return self.obs_norm.normalize(obs)

    @torch.no_grad()
    def act(self, norm_obs):
        """返回 (delta_q, zeros_bc_placeholder)。接受已归一化的观测。"""
        s_t    = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        dq     = self.actor(s_t).cpu().numpy().flatten()
        dq     = np.clip(dq, -self.dq_max, self.dq_max)
        return dq, np.zeros(self.action_dim, np.float32)

    def step_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_delta, self.epsilon_min)

    def remember(self, norm_s, dq, bc_dq, norm_ns, reward, done):
        self.buffer.add(norm_s, dq, bc_dq, norm_ns, reward, done)

    def train(self, iterations=1):
        if self.buffer.size < self.batch_size:
            return TD3_ZERO

        dev = self.device
        total_lc = total_qp = total_qt = 0.0

        for _ in range(iterations):
            self.total_it += 1
            s, a, bc, ns, r, d = self.buffer.sample(self.batch_size)
            si  = np_to_tensor(s,  dev); ai  = np_to_tensor(a,  dev)
            bci = np_to_tensor(bc, dev); nsi = np_to_tensor(ns, dev)
            ri  = np_to_tensor(r,  dev).view(-1, 1)
            di  = np_to_tensor(d,  dev).view(-1, 1)
            dq_max_t = np_to_tensor(self.dq_max, dev)

            with torch.no_grad():
                # [DELTA-3][BUG-T2] 噪声直接加在 Δq 上，与 dq_max 量级一致
                noise = (torch.randn_like(ai) * self.policy_noise * dq_max_t
                         ).clamp(-self.noise_clip * dq_max_t,
                                  self.noise_clip * dq_max_t)
                next_a = (self.target_actor(nsi) + noise
                          ).clamp(-dq_max_t, dq_max_t)
                tQ1, tQ2 = self.target_critic(nsi, next_a)
                yi = (ri + self.gamma * (1 - di) * torch.min(tQ1, tQ2)
                      ).clamp(-self.target_q_clip, self.target_q_clip)

            cQ1, cQ2 = self.critic(si, ai)
            total_qp += cQ1.mean().item(); total_qt += yi.mean().item()

            lc = (F.huber_loss(cQ1, yi) + F.huber_loss(cQ2, yi)
                  if self.critic_loss_type == "huber"
                  else F.mse_loss(cQ1, yi) + F.mse_loss(cQ2, yi))

            if torch.isfinite(lc):
                self.opt_critic.zero_grad(); lc.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
                self.opt_critic.step(); total_lc += lc.item()

            if self.total_it % self.policy_freq == 0:
                a_pred = self.actor(si)
                q_pred = self.critic.Q1(si, a_pred)

                if self.behavior_clone:
                    lbc    = F.mse_loss(a_pred, bci)
                    lmbda  = (self.bc_alpha / (q_pred.abs().mean().detach() + 1e-5)
                              ).clamp(0.1, 10.0)
                    la     = lbc - lmbda * q_pred.mean()
                    self._last_bl = lbc.item()
                else:
                    la = -q_pred.mean(); self._last_bl = 0.0

                self.opt_actor.zero_grad(); la.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
                self.opt_actor.step(); self._last_al = la.item()

                soft_update(self.target_actor,  self.actor,  self.tau)
                soft_update(self.target_critic, self.critic, self.tau)

        n = max(iterations, 1)
        return TD3Result(total_lc/n, self._last_al, self._last_bl,
                         total_qp/n, total_qt/n)

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "epsilon": self.epsilon, "total_it": self.total_it,
            "obs_norm": self.obs_norm.state_dict(),
        }, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.target_actor.load_state_dict(ck["target_actor"])
        self.critic.load_state_dict(ck["critic"])
        self.target_critic.load_state_dict(ck["target_critic"])
        self.opt_actor.load_state_dict(ck["opt_actor"])
        self.opt_critic.load_state_dict(ck["opt_critic"])
        self.epsilon  = ck.get("epsilon",  self.epsilon)
        self.total_it = ck.get("total_it", self.total_it)
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])