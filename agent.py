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
        self.n = 0
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
            old = self.mean.copy()
            self.mean = old + (x - old) / self.n
            self.S    = self.S + (x - old) * (x - self.mean)
 
    @property
    def var(self):  return self.S / max(self.n - 1, 1)
    @property
    def std(self):  return np.sqrt(self.var + 1e-8)
 
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
        self.n, self.mean, self.S = d["n"], d["mean"].copy(), d["S"].copy()
 
 
# ==============================================================================
# 共享 MLP 骨干网络
# ==============================================================================
 
def build_mlp(in_dim: int, hidden_dim: int, n_layers: int, out_dim: int,
              activation=nn.ReLU, last_gain: float = 1.0) -> nn.Sequential:
    """构建 n 层 MLP，最后一层用较小的增益初始化（RL 标准做法）。"""
    layers: List[nn.Module] = []
    d = in_dim
    for i in range(n_layers):
        is_last = (i == n_layers - 1)
        out = out_dim if is_last else hidden_dim
        lin = nn.Linear(d, out)
        gain = last_gain if is_last else np.sqrt(2)
        orthogonal_init(lin, gain=gain)
        layers.append(lin)
        if not is_last:
            layers.append(activation())
        d = out
    return nn.Sequential(*layers)
 
 
# ==============================================================================
# PPO Actor（有界高斯策略，输出关节角）
# ==============================================================================
 
class PPOActor(nn.Module):
    """
    PPO 策略网络（Actor）。
    输出：对角高斯分布的均值和对数标准差。
    动作通过 tanh 压缩后映射到关节限位范围。
 
    输出关节角 = tanh(mean) * scale + offset
    其中 scale = (high + low) / 2，offset = (high - low) / 2
    """
 
    def __init__(self, state_dim: int, action_dim: int,
                 action_low: np.ndarray, action_high: np.ndarray,
                 hidden_dim: int = 256, n_layers: int = 3,
                 log_std_init: float = -0.7,
                 log_std_min: float = -4.0, log_std_max: float = 1.0):
        super().__init__()
 
        # 关节限位映射参数（用 register_buffer 确保随模型迁移到 GPU）
        scale  = torch.tensor((action_high + action_low) / 2, dtype=torch.float32)
        offset = torch.tensor((action_high - action_low) / 2, dtype=torch.float32)
        self.register_buffer('action_scale',  scale)
        self.register_buffer('action_offset', offset)
 
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
 
        # 共享特征提取层（最后一层用小增益初始化）
        self.backbone = build_mlp(state_dim, hidden_dim, n_layers - 1, hidden_dim,
                                   last_gain=np.sqrt(2))
        self.act_fn   = nn.ReLU()
 
        # 均值头（用极小增益使初始策略接近零，便于 BC 引导）
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.mean_head, gain=0.01)
 
        # 对数标准差（可学习参数，独立于状态）
        self.log_std = nn.Parameter(
            torch.ones(action_dim) * log_std_init
        )
 
    def forward(self, s: torch.Tensor):
        """返回 (mean_raw, std)，均值未经 tanh 压缩（用于 log_prob 计算）。"""
        feat = self.act_fn(self.backbone(s))
        mean_raw = self.mean_head(feat)
        log_std  = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std      = log_std.exp().expand_as(mean_raw)
        return mean_raw, std
 
    def get_action(self, s: torch.Tensor, deterministic: bool = False):
        """
        采样或取均值动作，返回 (action_joint, log_prob, entropy)。
        action_joint 已映射到关节限位范围。
        """
        mean_raw, std = self.forward(s)
 
        if deterministic:
            u = mean_raw
        else:
            dist = Normal(mean_raw, std)
            u    = dist.rsample()
 
        # tanh squashing → 映射到关节角范围
        u_tanh  = torch.tanh(u)
        action  = u_tanh * self.action_scale + self.action_offset
 
        # log_prob 修正（tanh 雅可比行列式）
        # log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u_i))
        dist_base = Normal(mean_raw, std)
        log_prob  = dist_base.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(dim=-1)
 
        entropy = dist_base.entropy().sum(dim=-1)
 
        return action, log_prob, entropy
 
    def evaluate_actions(self, s: torch.Tensor, action_joint: torch.Tensor):
        """
        对给定动作计算 log_prob 和 entropy（PPO 更新时用）。
        action_joint 是实际执行的关节角（已在限位内）。
        """
        mean_raw, std = self.forward(s)
 
        # 反映射：从关节角恢复 u_tanh，再反 tanh 到 u
        u_tanh = (action_joint - self.action_offset) / (self.action_scale + 1e-8)
        u_tanh = u_tanh.clamp(-1 + 1e-6, 1 - 1e-6)
        u      = torch.atanh(u_tanh)
 
        dist     = Normal(mean_raw, std)
        log_prob = dist.log_prob(u).sum(dim=-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
 
        return log_prob, entropy
 
 
# ==============================================================================
# PPO Critic（状态价值函数 V(s)）
# ==============================================================================
 
class PPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.net = build_mlp(state_dim, hidden_dim, n_layers, 1, last_gain=1.0)
 
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)
 
 
# ==============================================================================
# RolloutBuffer（PPO 专用）
# ==============================================================================
 
class RolloutBuffer:
    """
    PPO Rollout 数据存储与 GAE 计算。
    存储归一化后的 obs（与 ReplayBuffer 保持一致的"存归一化值"策略）。
    """
 
    def __init__(self, n_steps: int, state_dim: int, action_dim: int, device):
        self.n_steps    = n_steps
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.device     = device
        self.clear()
 
    def clear(self):
        self.obs      = np.zeros((self.n_steps, self.state_dim),  dtype=np.float32)
        self.actions  = np.zeros((self.n_steps, self.action_dim), dtype=np.float32)
        self.bc_targets = np.zeros((self.n_steps, self.action_dim), dtype=np.float32)
        self.rewards  = np.zeros(self.n_steps, dtype=np.float32)
        self.dones    = np.zeros(self.n_steps, dtype=np.float32)
        self.values   = np.zeros(self.n_steps, dtype=np.float32)
        self.log_probs = np.zeros(self.n_steps, dtype=np.float32)
        self.advantages = np.zeros(self.n_steps, dtype=np.float32)
        self.returns    = np.zeros(self.n_steps, dtype=np.float32)
        self.ptr = 0
        self.full = False
 
    def add(self, obs, action, bc_target, reward, done, value, log_prob):
        """存入一步数据。"""
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.bc_targets[i]= bc_target
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True
 
    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        """
        GAE（Generalized Advantage Estimation）计算。
        last_value: Critic 对最后一步 next_obs 的价值估计（episode 未完成时为 V(s_T)，完成时为 0）
        """
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_val = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_val = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t + 1]
 
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae
 
        self.returns = self.advantages + self.values
 
    def get_minibatches(self, batch_size: int, normalize_adv: bool = True):
        """
        随机打乱后生成 minibatch 迭代器。
        每个 minibatch 包含 Tensor（已在 device 上）。
        """
        assert self.full, "Buffer 未满，不能采样"
        indices = np.random.permutation(self.n_steps)
 
        if normalize_adv:
            adv = self.advantages
            self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
 
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                np_to_tensor(self.obs[idx],       self.device),
                np_to_tensor(self.actions[idx],    self.device),
                np_to_tensor(self.bc_targets[idx], self.device),
                np_to_tensor(self.returns[idx],    self.device).view(-1, 1),
                np_to_tensor(self.advantages[idx], self.device),
                np_to_tensor(self.log_probs[idx],  self.device),
            )
 
 
# ==============================================================================
# PPO Agent（主框架）
# ==============================================================================
 
PPOTrainResult = namedtuple("PPOTrainResult", [
    "policy_loss", "value_loss", "entropy_loss", "bc_loss",
    "approx_kl", "clip_fraction", "total_loss"
])
PPO_ZERO = PPOTrainResult(0., 0., 0., 0., 0., 0., 0.)
 
 
class PPOAgent:
    """
    PPO + BC Agent，动作空间为 7D 关节角目标。
 
    训练流程：
      1. collect_rollout()：与环境交互 n_steps 步，收集 rollout
      2. update()：用 rollout 数据做 n_epochs 次 PPO 更新（含 BC Loss）
      3. BC 系数随总 env steps 线性退火（从 bc_coef_init 到 bc_coef_final）
    """
 
    def __init__(self, log_dir, state_dim: int, action_dim: int,
                 config: dict = None, expert=None):
        if config is None:
            config = DEFAULT_CONFIG
 
        self.config      = config
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.expert      = expert      # JointSpaceExpert 实例（可为 None）
 
        cfg = config["ppo_agent"]
        sp  = config["space"]
 
        # 动作空间边界
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
        self.use_obs_norm    = bool(cfg["use_obs_norm"])
        self.obs_norm        = RunningMeanStd(shape=(state_dim,),
                                              warm_start=200,
                                              clip=float(cfg["obs_norm_clip"]))
 
        # 设备
        train_cfg = config["train"]
        gpu_id = train_cfg.get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))
 
        # 网络
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
 
        # 优化器（Actor 和 Critic 共用一个，标准 PPO 做法）
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(cfg["lr_actor"]), eps=1e-5
        )
 
        # Rollout Buffer
        self.buffer = RolloutBuffer(self.n_steps, state_dim, action_dim, self.device)
 
        # 训练统计
        self.total_steps = 0
        self._last_result = PPO_ZERO
 
    # ──────────────────────────────────────────────────────────────────────────
    # 观测归一化
    # ──────────────────────────────────────────────────────────────────────────
 
    def normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        """更新归一化统计并返回归一化观测。"""
        if self.use_obs_norm:
            if update:
                self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)
 
    # ──────────────────────────────────────────────────────────────────────────
    # 动作采样（collect_rollout 时调用）
    # ──────────────────────────────────────────────────────────────────────────
 
    @torch.no_grad()
    def act(self, norm_obs: np.ndarray, deterministic: bool = False
            ) -> Tuple[np.ndarray, float, float]:
        """
        Returns:
            action_joint: 7D 关节角，已 clamp 到关节限位
            log_prob: 标量 float
            value: 标量 float（Critic 估计）
        """
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        action, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
        value               = self.critic(s)
 
        action_np   = action.cpu().numpy().flatten()
        log_prob_np = log_prob.cpu().item()
        value_np    = value.cpu().item()
 
        # 安全 clamp（理论上 tanh 已保证范围，但防 float 精度）
        action_np = np.clip(action_np, self.action_low, self.action_high)
        return action_np, log_prob_np, value_np
 
    # ──────────────────────────────────────────────────────────────────────────
    # BC 系数退火
    # ──────────────────────────────────────────────────────────────────────────
 
    def _update_bc_coef(self):
        """根据 total_steps 线性退火 BC 系数。"""
        if not self.behavior_clone or self.bc_anneal_steps <= 0:
            return
        frac = min(self.total_steps / self.bc_anneal_steps, 1.0)
        self.bc_coef = self.bc_coef_init + frac * (self.bc_coef_final - self.bc_coef_init)
 
    # ──────────────────────────────────────────────────────────────────────────
    # PPO 更新核心
    # ──────────────────────────────────────────────────────────────────────────
 
    def update(self) -> PPOTrainResult:
        """
        用当前 rollout buffer 做 n_epochs 次 PPO 更新。
        返回最后一个 epoch 的平均 loss。
        """
        if not self.buffer.full:
            return PPO_ZERO
 
        self._update_bc_coef()
 
        total_pl = total_vl = total_el = total_bl = total_kl = total_cf = 0.0
        n_updates = 0
 
        for epoch in range(self.n_epochs):
            early_stop = False
            for batch in self.buffer.get_minibatches(self.batch_size, self.norm_adv):
                obs_b, act_b, bc_b, ret_b, adv_b, old_lp_b = batch
 
                # ── 前向传播 ──────────────────────────────────────────────
                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                value            = self.critic(obs_b)
 
                # ── PPO Policy Loss ───────────────────────────────────────
                ratio     = (new_lp - old_lp_b).exp()
                surr1     = ratio * adv_b
                surr2     = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
 
                # ── Value Loss（clipped）──────────────────────────────────
                value_loss = F.huber_loss(value, ret_b)
 
                # ── Entropy Loss ──────────────────────────────────────────
                entropy_loss = -entropy.mean()
 
                # ── BC Loss（关节角 MSE 到专家目标）─────────────────────
                if self.behavior_clone and self.bc_coef > 0:
                    if self.bc_loss_type == "mse":
                        bc_loss = F.mse_loss(act_b, bc_b)
                    else:
                        # Soft BC：用 log_prob 形式（更平滑的引导）
                        bc_loss = F.smooth_l1_loss(act_b, bc_b)
                else:
                    bc_loss = torch.tensor(0.0, device=self.device)
 
                # ── Total Loss ────────────────────────────────────────────
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
 
                # 统计
                with torch.no_grad():
                    approx_kl = ((old_lp_b - new_lp).mean()).item()
                    clip_frac  = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
 
                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_el += entropy_loss.item()
                total_bl += bc_loss.item()
                total_kl += approx_kl
                total_cf += clip_frac
                n_updates += 1
 
            # KL 早停
            if total_kl / max(n_updates, 1) > self.target_kl:
                early_stop = True
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
 
        # 清空 buffer 供下次 rollout 使用
        self.buffer.clear()
        return result
 
    # ──────────────────────────────────────────────────────────────────────────
    # 保存 / 加载
    # ──────────────────────────────────────────────────────────────────────────
 
    def save(self, path: str):
        torch.save({
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "bc_coef":    self.bc_coef,
            "obs_norm":   self.obs_norm.state_dict(),
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
# TD3 Agent（保留，关节空间版本，用于对比实验）
# ==============================================================================
 
# ── 网络 ──────────────────────────────────────────────────────────────────────
 
class TD3Actor(nn.Module):
    """TD3 Actor，输出关节角目标（tanh + 缩放）。"""
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        scale  = torch.tensor((action_high + action_low) / 2, dtype=torch.float32)
        offset = torch.tensor((action_high - action_low) / 2, dtype=torch.float32)
        self.register_buffer('action_scale',  scale)
        self.register_buffer('action_offset', offset)
 
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )
        self.net.apply(lambda m: orthogonal_init(m, gain=1.0))
        orthogonal_init(self.net[-2], gain=0.01)
 
    def forward(self, s):
        return self.net(s) * self.action_offset + self.action_scale
 
 
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
            net.apply(lambda m: orthogonal_init(m, gain=1.0))
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
TD3_ZERO  = TD3Result(0.,0.,0.,0.,0.)
 
 
class TD3Agent:
    """TD3 + BC Agent（关节空间版本）。与 PPOAgent 接口对齐，方便在 test.py 中统一加载。"""
 
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
 
        # 超参数
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
 
        # 归一化
        self.obs_norm = RunningMeanStd(shape=(state_dim,), warm_start=200)
 
        # 设备
        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))
 
        hidden = int(cfg["hidden_dim"])
        self.actor        = TD3Actor(state_dim, action_dim, self.action_low, self.action_high, hidden).to(self.device)
        self.target_actor = TD3Actor(state_dim, action_dim, self.action_low, self.action_high, hidden).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict()); self.target_actor.eval()
 
        self.critic        = TD3Critic(state_dim, action_dim, hidden).to(self.device)
        self.target_critic = TD3Critic(state_dim, action_dim, hidden).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict()); self.target_critic.eval()
 
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
        """返回 (actor_action 关节角, bc_target 关节角)。"""
        norm_s = self.normalize_obs(raw_obs, update=False)
        s_t    = np_to_tensor(norm_s.reshape(1, -1), self.device)
        action = self.actor(s_t).cpu().numpy().flatten()
        action = np.clip(action, self.action_low, self.action_high)
 
        # BC 目标由外部 expert 计算后传入 remember，此处返回零占位
        bc = np.zeros(self.action_dim, dtype=np.float32)
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
            ri  = np_to_tensor(r,  dev).view(-1,1)
            di  = np_to_tensor(d,  dev).view(-1,1)
 
            with torch.no_grad():
                # Target policy smoothing（关节空间噪声）
                scale = self.target_actor.action_offset
                n_std  = self.policy_noise * scale
                n_clip = self.noise_clip   * scale
                noise  = (torch.randn_like(ai) * n_std).clamp(-n_clip, n_clip)
                next_a = (self.target_actor(nsi) + noise).clamp(
                    torch.tensor(self.action_low, device=dev),
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
                    lmbda  = (self.bc_alpha / (q_pred.abs().mean().detach() + 1e-5)).clamp(0.1, 10.0)
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
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"]); self.target_actor.load_state_dict(ck["target_actor"])
        self.critic.load_state_dict(ck["critic"]); self.target_critic.load_state_dict(ck["target_critic"])
        self.opt_actor.load_state_dict(ck["opt_actor"]); self.opt_critic.load_state_dict(ck["opt_critic"])
        self.epsilon  = ck.get("epsilon", self.epsilon)
        self.total_it = ck.get("total_it", self.total_it)
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])