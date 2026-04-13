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
# ══════════════════════════════════════════════════════════════════════════════
# 修复清单（相对上传版本）
# ══════════════════════════════════════════════════════════════════════════════
#
# [FIX-A1] PPOActor tanh 映射公式错误（输出超出关节限位）
#   原版：
#     scale  = (high + low) / 2    ← 这是中心点（offset）
#     offset = (high - low) / 2    ← 这是半范围（scale）
#     action = tanh(u) * action_scale + action_offset
#            = tanh(u) * center + half_range
#   当 tanh(u) ∈ (-1, 1) 时：
#     action ∈ (half_range - center, half_range + center)
#   对于 q1 ∈ [-2.967, 2.967]：
#     center = 0, half_range = 2.967
#     action ∈ (2.967 - 0, 2.967 + 0) = (2.967 - 2.967, 2.967 + 2.967)
#            = (0, 5.934)  ← 完全错误！
#
#   正确公式：
#     action = tanh(u) * half_range + center
#            = tanh(u) * (high-low)/2 + (high+low)/2
#   当 tanh(u) = ±1 时 action = high or low ✓
#
#   修复：交换 scale 和 offset 的定义：
#     action_scale  = (high - low) / 2   （乘 tanh 的系数，即半范围）
#     action_center = (high + low) / 2   （偏置，即中心点）
#     action = tanh(u) * action_scale + action_center
#
#   同时修复 evaluate_actions 中的反映射：
#     u_tanh = (action - action_center) / action_scale
#
# [FIX-A2] PPOActor build_mlp 层数语义错误
#   原版：build_mlp(..., n_layers-1, hidden_dim) 生成 n_layers-1 层，
#   最后一层 mean_head 用 nn.Linear 单独定义。
#   但 build_mlp 的 last_gain 参数针对最后一层使用较大增益（sqrt(2)），
#   而 mean_head 被单独初始化为 gain=0.01——这是正确的。
#   问题在于：backbone = build_mlp(state_dim, hidden_dim, n_layers-1, hidden_dim)
#   当 n_layers=3 时，backbone 有 2 层，最后一层输出 hidden_dim，
#   而 PPO 标准是 3 层（2 个隐层 + 1 个输出头）。
#   当 n_layers=2 时，backbone 只有 1 层，几乎等于线性！
#   修复：将 build_mlp 语义明确化——backbone 始终输出 hidden_dim，
#   n_layers 表示 backbone 的隐层数（不含最终输出头），默认 2。
#
# [FIX-A3] RolloutBuffer.get_minibatches 的 normalize_adv 副作用
#   原版在迭代中修改 self.advantages（原地归一化），
#   而 get_minibatches 是一个 generator，多次 epoch 调用时
#   第一个 epoch 已经归一化，第二个 epoch 再次调用 get_minibatches
#   时 advantages 已是归一化后的值，再归一化会产生错误（接近 0 均值但方差可能不对）。
#   修复：归一化使用局部副本，不修改 self.advantages。
#
# [FIX-A4] GAE 计算中 done 标志的语义错误
#   原版计算 GAE 时：
#     if t == n_steps-1: next_non_terminal = 1 - dones[t]
#     else:              next_non_terminal = 1 - dones[t+1]
#   这里 dones[t+1] 是 t+1 时刻的 done，但 GAE 的 δ_t 应该用：
#     δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
#   即应该用 dones[t]（当前步的 done）而不是 dones[t+1]。
#   原版写法将 done 向后偏移了一步，导致 episode 边界处的 bootstrap 错误。
#   修复：统一使用 dones[t]，即"当前这步是否终止"来判断是否 bootstrap。
#
# [FIX-A5] PPO KL 早停逻辑错误
#   原版：
#     if total_kl / max(n_updates, 1) > self.target_kl:
#         break   ← 这是在 epoch 的 minibatch 循环外
#   但 early_stop 变量从未被检查（赋值后丢弃），early_stop = True 没有实际效果。
#   实际行为是：每个 epoch 结束后检查 KL，但 break 只跳出 epoch 循环，
#   检查发生在 epoch 循环体内，实际上根本不会触发早停。
#   修复：在正确的位置检查 KL（每个 minibatch 后），并正确 break。
#
# [FIX-A6] approx_kl 计算符号问题
#   原版：approx_kl = (old_lp_b - new_lp).mean()
#   KL(old||new) ≈ E[log(π_old/π_new)] = E[log_π_old - log_π_new]
#   这在数学上正确，但取值通常为正。
#   然而，当 policy 变化较小时 approx_kl 可能为负（数值误差），
#   用于早停时与 target_kl (正数) 比较会永远不触发。
#   修复：使用 abs(approx_kl) 或取 max(0, approx_kl) 进行早停比较。
#
# [FIX-A7] BC Loss 类型 "nll" 注释与代码不一致
#   注释说 "bc_loss_type=nll 用负对数似然"，实际代码用 smooth_l1_loss。
#   修复：将 "nll" 重命名为 "huber"，与实际损失函数名一致。
#
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import namedtuple
from typing import Optional, Tuple, List

from config import DEFAULT_CONFIG


# ==============================================================================
# 通用工具函数
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
    """Welford 在线均值/方差，warm_start 保护。"""
    def __init__(self, shape, warm_start: int = 200, clip: float = 10.0):
        self.n    = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S    = np.zeros(shape, dtype=np.float64)
        self.warm_start = warm_start
        self.clip = clip

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).flatten()
        self.n += 1
        if self.n == 1:
            self.mean = x.copy(); self.S = np.zeros_like(x)
        else:
            old       = self.mean.copy()
            self.mean = old + (x - old) / self.n
            self.S    = self.S + (x - old) * (x - self.mean)

    @property
    def var(self): return self.S / max(self.n - 1, 1)
    @property
    def std(self): return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
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
        self.n    = d["n"]
        self.mean = d["mean"].copy()
        self.S    = d["S"].copy()


# ==============================================================================
# 共享 MLP 骨干网络
# ==============================================================================

def build_mlp(in_dim: int, hidden_dim: int, n_hidden_layers: int,
              out_dim: int, activation=nn.ReLU,
              last_gain: float = 1.0) -> nn.Sequential:
    """
    构建 n_hidden_layers 个隐层的 MLP。
    [FIX-A2] n_layers 语义明确为隐层数（不含输出层）。
    """
    layers: List[nn.Module] = []
    d = in_dim
    # n_hidden_layers 个隐层
    for _ in range(n_hidden_layers):
        lin = nn.Linear(d, hidden_dim)
        orthogonal_init(lin, gain=np.sqrt(2))
        layers.append(lin)
        layers.append(activation())
        d = hidden_dim
    # 输出层
    out_lin = nn.Linear(d, out_dim)
    orthogonal_init(out_lin, gain=last_gain)
    layers.append(out_lin)
    return nn.Sequential(*layers)


# ==============================================================================
# PPO Actor（有界高斯策略）
# ==============================================================================

class PPOActor(nn.Module):
    """
    PPO 策略网络。
    动作分布：对角高斯 + tanh squashing → 映射到关节限位。

    [FIX-A1] 正确的 tanh 映射：
        action = tanh(u) * action_scale + action_center
        action_scale  = (high - low) / 2   （半范围）
        action_center = (high + low) / 2   （中心）
    """

    def __init__(self, state_dim: int, action_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 hidden_dim: int = 256, n_layers: int = 2,
                 log_std_init: float = -0.7,
                 log_std_min: float = -4.0, log_std_max: float = 1.0):
        super().__init__()

        # [FIX-A1] 正确的映射系数
        # action_scale：tanh 输出的缩放系数（半范围）
        # action_center：关节角空间的中心点
        action_scale  = torch.tensor((action_high - action_low) / 2, dtype=torch.float32)
        action_center = torch.tensor((action_high + action_low) / 2, dtype=torch.float32)
        self.register_buffer('action_scale',  action_scale)
        self.register_buffer('action_center', action_center)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # [FIX-A2] backbone 明确为 n_layers 个隐层，输出 hidden_dim
        self.backbone = nn.Sequential()
        d = state_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            orthogonal_init(lin, gain=np.sqrt(2))
            self.backbone.append(lin)
            self.backbone.append(nn.ReLU())
            d = hidden_dim

        # 均值输出头（小增益，使初始策略接近关节中心，便于 BC 引导）
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.mean_head, gain=0.01)

        # 状态无关的 log_std（可学习）
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def _get_distribution(self, s: torch.Tensor):
        """返回 (mean_raw, std)，mean_raw 是未经 tanh 的原始均值。"""
        feat     = self.backbone(s)
        mean_raw = self.mean_head(feat)
        log_std  = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std      = log_std.exp().expand_as(mean_raw)
        return mean_raw, std

    def get_action(self, s: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样（或确定性取均值）动作。
        Returns: (action_joint, log_prob, entropy)
        """
        mean_raw, std = self._get_distribution(s)
        dist = Normal(mean_raw, std)

        u = mean_raw if deterministic else dist.rsample()

        u_tanh = torch.tanh(u)
        # [FIX-A1] action = tanh(u) * scale + center
        action = u_tanh * self.action_scale + self.action_center

        # log_prob（含 tanh Jacobian 修正）
        log_prob  = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate_actions(self, s: torch.Tensor,
                         action_joint: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对已执行的 action_joint 计算 log_prob 和 entropy（PPO 更新用）。
        [FIX-A1] 反映射使用正确的 action_scale 和 action_center。
        """
        mean_raw, std = self._get_distribution(s)

        # 反映射：action_joint → u_tanh → u
        # u_tanh = (action_joint - action_center) / action_scale
        u_tanh = (action_joint - self.action_center) / (self.action_scale + 1e-8)
        u_tanh = u_tanh.clamp(-1 + 1e-6, 1 - 1e-6)
        u      = torch.atanh(u_tanh)

        dist     = Normal(mean_raw, std)
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        return log_prob, entropy


# ==============================================================================
# PPO Critic
# ==============================================================================

class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        # [FIX-A2] n_layers 个隐层 + 1 输出
        layers: List[nn.Module] = []
        d = state_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            orthogonal_init(lin, gain=np.sqrt(2))
            layers.append(lin)
            layers.append(nn.ReLU())
            d = hidden_dim
        out = nn.Linear(hidden_dim, 1)
        orthogonal_init(out, gain=1.0)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# ==============================================================================
# RolloutBuffer
# ==============================================================================

class RolloutBuffer:
    """PPO Rollout 数据缓冲区。"""

    def __init__(self, n_steps: int, state_dim: int, action_dim: int, device):
        self.n_steps    = n_steps
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = device
        self.clear()

    def clear(self):
        self.obs        = np.zeros((self.n_steps, self.state_dim),  dtype=np.float32)
        self.actions    = np.zeros((self.n_steps, self.action_dim), dtype=np.float32)
        self.bc_targets = np.zeros((self.n_steps, self.action_dim), dtype=np.float32)
        self.rewards    = np.zeros(self.n_steps, dtype=np.float32)
        self.dones      = np.zeros(self.n_steps, dtype=np.float32)
        self.values     = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs  = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns    = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr  = 0
        self.full = False

    def add(self, obs, action, bc_target, reward, done, value, log_prob):
        i = self.ptr
        self.obs[i]        = obs
        self.actions[i]    = action
        self.bc_targets[i] = bc_target
        self.rewards[i]    = reward
        self.dones[i]      = float(done)
        self.values[i]     = value
        self.log_probs[i]  = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float,
                                        gamma: float, gae_lambda: float):
        """
        GAE（Generalized Advantage Estimation）。
        [FIX-A4] 使用 dones[t]（当前步终止标志）决定是否 bootstrap。
        """
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            # [FIX-A4] next_val 与 done 的对应关系：
            # t=n_steps-1 时，next_val 由外部传入 last_value
            # 其他时刻，next_val = V(s_{t+1}) = self.values[t+1]
            # non_terminal 由 dones[t] 决定（当前步结束则不 bootstrap）
            if t == self.n_steps - 1:
                next_val         = last_value
            else:
                next_val         = self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]  # [FIX-A4] 使用 dones[t]

            delta    = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int, normalize_adv: bool = True):
        """
        [FIX-A3] 归一化使用局部副本，不修改 self.advantages。
        """
        assert self.full
        indices = np.random.permutation(self.n_steps)

        # [FIX-A3] 使用副本进行归一化
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
    """PPO + BC Agent（7D 关节角动作空间）。"""

    def __init__(self, log_dir, state_dim: int, action_dim: int,
                 config: dict = None, expert=None):
        if config is None:
            config = DEFAULT_CONFIG

        self.config     = config
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.expert     = expert

        cfg = config["ppo_agent"]
        sp  = config["space"]

        self.action_low  = np.array(sp["action_space_low"],  dtype=np.float32)
        self.action_high = np.array(sp["action_space_high"], dtype=np.float32)

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
        self.target_kl       = float(cfg.get("target_kl", 0.02))

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
            state_dim, action_dim,
            self.action_low, self.action_high,
            hidden_dim=hidden_dim, n_layers=n_layers,
            log_std_init=float(cfg["log_std_init"]),
            log_std_min=float(cfg["log_std_min"]),
            log_std_max=float(cfg["log_std_max"]),
        ).to(self.device)

        self.critic = PPOCritic(state_dim, hidden_dim, n_layers).to(self.device)

        # Actor + Critic 共用优化器（标准 PPO）
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(cfg["lr_actor"]), eps=1e-5
        )

        self.buffer      = RolloutBuffer(self.n_steps, state_dim, action_dim, self.device)
        self.total_steps = 0
        self._last_result = PPO_ZERO

    # ──────────────────────────────────────────────────────────────────────────

    def normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.use_obs_norm:
            if update:
                self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, norm_obs: np.ndarray, deterministic: bool = False
            ) -> Tuple[np.ndarray, float, float]:
        """
        Returns: (action_joint, log_prob, value)
        """
        s       = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        action, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
        value   = self.critic(s)

        action_np   = action.cpu().numpy().flatten()
        log_prob_np = log_prob.cpu().item()
        value_np    = value.cpu().item()
        action_np   = np.clip(action_np, self.action_low, self.action_high)
        return action_np, log_prob_np, value_np

    def _update_bc_coef(self):
        if not self.behavior_clone or self.bc_anneal_steps <= 0:
            return
        frac = min(self.total_steps / self.bc_anneal_steps, 1.0)
        self.bc_coef = self.bc_coef_init + frac * (self.bc_coef_final - self.bc_coef_init)

    def update(self) -> PPOTrainResult:
        """执行 n_epochs 轮 PPO 更新，含 BC Loss。"""
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
                obs_b, act_b, bc_b, ret_b, adv_b, old_lp_b = batch

                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                value            = self.critic(obs_b)

                # PPO Clipped Policy Loss
                ratio       = (new_lp - old_lp_b).exp()
                surr1       = ratio * adv_b
                surr2       = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss  = F.huber_loss(value, ret_b)

                # Entropy Loss（最大化熵 = 最小化负熵）
                entropy_loss = -entropy.mean()

                # BC Loss
                if self.behavior_clone and self.bc_coef > 0:
                    # [FIX-A7] 统一命名："mse" 或 "huber"
                    if self.bc_loss_type == "huber":
                        bc_loss = F.smooth_l1_loss(act_b, bc_b)
                    else:  # 默认 mse
                        bc_loss = F.mse_loss(act_b, bc_b)
                else:
                    bc_loss = torch.zeros(1, device=self.device)

                loss = (policy_loss
                        + self.value_loss_coef * value_loss
                        + self.entropy_coef    * entropy_loss
                        + self.bc_coef         * bc_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    # [FIX-A6] approx_kl 取绝对值用于早停比较
                    approx_kl = ((old_lp_b - new_lp).mean()).abs().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_el += entropy_loss.item()
                total_bl += bc_loss.item()
                total_kl += approx_kl
                total_cf += clip_frac
                n_updates += 1

                # [FIX-A5] 每个 minibatch 后检查 KL，正确触发早停
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

    def save(self, path: str):
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "bc_coef":     self.bc_coef,
            "obs_norm":    self.obs_norm.state_dict(),
        }, path)

    def load(self, path: str, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device,
                        weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.optimizer.load_state_dict(ck["optimizer"])
        self.total_steps = ck.get("total_steps", 0)
        self.bc_coef     = ck.get("bc_coef", self.bc_coef)
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])


# ==============================================================================
# TD3 Agent（关节空间版本，与 PPOAgent 接口对齐）
# ==============================================================================

class TD3Actor(nn.Module):
    """TD3 确定性策略网络，输出关节角目标（tanh + 映射）。"""
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        # [FIX-A1 同步] 正确的映射系数
        action_scale  = torch.tensor((action_high - action_low) / 2, dtype=torch.float32)
        action_center = torch.tensor((action_high + action_low) / 2, dtype=torch.float32)
        self.register_buffer('action_scale',  action_scale)
        self.register_buffer('action_center', action_center)

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
        return self.net(s) * self.action_scale + self.action_center


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
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.state      = np.zeros((max_size, state_dim),  np.float32)
        self.next_state = np.zeros((max_size, state_dim),  np.float32)
        self.action     = np.zeros((max_size, action_dim), np.float32)
        self.bc_target  = np.zeros((max_size, action_dim), np.float32)
        self.reward     = np.zeros((max_size, 1),          np.float32)
        self.done       = np.zeros((max_size, 1),          np.float32)

    def add(self, s, a, bc, ns, r, d):
        i = self.ptr
        self.state[i]     = s;  self.next_state[i] = ns
        self.action[i]    = a;  self.bc_target[i]  = bc
        self.reward[i]    = r;  self.done[i]       = d
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (self.state[idx], self.action[idx], self.bc_target[idx],
                self.next_state[idx], self.reward[idx], self.done[idx])


TD3Result = namedtuple("TD3Result", ["critic_loss","actor_loss","bc_loss","q_pred","q_target"])
TD3_ZERO  = TD3Result(0., 0., 0., 0., 0.)


class TD3Agent:
    """TD3 + BC Agent（关节空间版本）。"""

    def __init__(self, log_dir, state_dim, action_dim, config=None, expert=None):
        if config is None:
            config = DEFAULT_CONFIG
        self.config     = config
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.expert     = expert

        cfg = config["td3_agent"]
        sp  = config["space"]
        self.action_low  = np.array(sp["action_space_low"],  dtype=np.float32)
        self.action_high = np.array(sp["action_space_high"], dtype=np.float32)

        self.batch_size       = int(cfg["batch_size"])
        self.gamma            = float(cfg["gamma"])
        self.tau              = float(cfg["tau"])
        self.policy_noise     = float(cfg["policy_noise"])
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
        self.actor        = TD3Actor(state_dim, action_dim, self.action_low, self.action_high, hidden).to(self.device)
        self.target_actor = TD3Actor(state_dim, action_dim, self.action_low, self.action_high, hidden).to(self.device)
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
    def act(self, raw_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        norm_s = self.normalize_obs(raw_obs, update=False)
        s_t    = np_to_tensor(norm_s.reshape(1, -1), self.device)
        action = self.actor(s_t).cpu().numpy().flatten()
        action = np.clip(action, self.action_low, self.action_high)
        bc     = np.zeros(self.action_dim, dtype=np.float32)
        return action, bc

    def step_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon - self.epsilon_delta, self.epsilon_min)

    def remember(self, norm_s, action, bc_target, norm_ns, reward, done):
        self.buffer.add(norm_s, action, bc_target, norm_ns, reward, done)

    def train(self, iterations=1) -> TD3Result:
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

            with torch.no_grad():
                # Target policy smoothing（关节空间噪声，按半范围缩放）
                half_range = self.target_actor.action_scale
                noise = (torch.randn_like(ai) * self.policy_noise * half_range
                         ).clamp(-self.noise_clip * half_range,
                                  self.noise_clip * half_range)
                next_a = (self.target_actor(nsi) + noise).clamp(
                    torch.tensor(self.action_low,  device=dev),
                    torch.tensor(self.action_high, device=dev)
                )
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

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(), "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "epsilon": self.epsilon, "total_it": self.total_it,
            "obs_norm": self.obs_norm.state_dict(),
        }, path)

    def load(self, path: str, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device,
                        weights_only=False)
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