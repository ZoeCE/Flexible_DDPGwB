# ==============================================================================
# swing_controller.py — 底层防摆 RL 控制器
#
# 核心职责：平滑跟踪来自路径规划器的航点，保持 payload 姿态稳定，抵抗风力扰动。
# 不负责全局路径规划与避障（那是高层 Planner 的任务）。
#
# 观测维度（修正后 35 维）：
#   payload_pos(3) + payload_vel(3) + ee_payload_offset(3) + ee_vel(3)
#   + tilt(1) + yaw(1) + tilt_rate(1) + yaw_rate(1)
#   + joint_q(7) + last_action(7) + waypoint_err(3) + wind_info(2) = 35
#
# 设计选择：
#   - 不使用 tanh squashing（避免饱和区梯度消失），直接 clip 到 ±dq_max
#   - log_std 状态无关（简化底层策略）
#   - orthogonal_init，输出层 gain=0.01（初始动作接近零）
# ==============================================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional, Dict, Any

from config import DEFAULT_CONFIG


# ==============================================================================
# 工具函数
# ==============================================================================

def _opt_cuda(t: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    if torch.cuda.is_available():
        return t.to(device) if device else t.cuda()
    return t


def _np_to_tensor(n: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    return _opt_cuda(torch.as_tensor(n, dtype=torch.float32), device)


def _orthogonal_init(layer: nn.Module, gain: float = np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


# ==============================================================================
# Running Mean/Std（与 agent.py 中的实现独立，避免循环依赖）
# ==============================================================================

class RunningMeanStd:
    """Welford 在线均值方差估计器。"""

    def __init__(self, shape: tuple, warm_start: int = 200, clip: float = 10.0):
        self.n: int = 0
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
            old = self.mean.copy()
            self.mean = old + (x - old) / self.n
            self.S    = self.S + (x - old) * (x - self.mean)

    @property
    def var(self) -> np.ndarray:
        return self.S / max(self.n - 1, 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.n < self.warm_start:
            return x
        return np.clip(
            (x - self.mean.astype(np.float32)) / self.std.astype(np.float32),
            -self.clip, self.clip
        )

    def state_dict(self) -> dict:
        return {"n": self.n, "mean": self.mean.copy(), "S": self.S.copy()}

    def load_state_dict(self, d: dict):
        self.n = d["n"]
        self.mean = d["mean"].copy()
        self.S = d["S"].copy()


# ==============================================================================
# SwingControllerNetwork — Actor
# ==============================================================================

class SwingControllerActor(nn.Module):
    """
    底层防摆策略网络。

    输出：7维 delta_q_base（线性输出，后续 clip 到 ±dq_max）
    不使用 tanh squashing，避免饱和区梯度消失。
    """

    def __init__(self, obs_dim: int, action_dim: int = 7,
                 hidden_dim: int = 256, n_layers: int = 3,
                 log_std_init: float = -1.0,
                 log_std_min: float = -4.0,
                 log_std_max: float = 0.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # backbone
        layers = []
        d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            _orthogonal_init(lin, gain=np.sqrt(2))
            layers += [lin, nn.ReLU()]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # mean head (gain=0.01 → 初始输出接近零)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        _orthogonal_init(self.mean_head, gain=0.01)

        # 状态无关 log_std
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def _dist(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(s)
        mean = self.mean_head(feat)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, s: torch.Tensor,
                   deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 (action, log_prob, entropy)。
        action 是线性值，需要外部 clip 到 ±dq_max。
        """
        mean, std = self._dist(s)
        dist = Normal(mean, std)
        u = mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(u).sum(-1)
        entropy = dist.entropy().sum(-1)
        return u, log_prob, entropy

    def evaluate_actions(self, s: torch.Tensor,
                         actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PPO 更新时计算已执行 action 的 log_prob 和 entropy。"""
        mean, std = self._dist(s)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy


# ==============================================================================
# SwingControllerCritic
# ==============================================================================

class SwingControllerCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            _orthogonal_init(lin, gain=np.sqrt(2))
            layers += [lin, nn.ReLU()]
            d = hidden_dim
        out = nn.Linear(hidden_dim, 1)
        _orthogonal_init(out, gain=1.0)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


# ==============================================================================
# RolloutBuffer（底层专用，无 bc_target）
# ==============================================================================

class SwingRolloutBuffer:
    def __init__(self, n_steps: int, obs_dim: int, action_dim: int,
                 device: torch.device):
        self.n_steps    = n_steps
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.device     = device
        self.clear()

    def clear(self):
        self.obs        = np.zeros((self.n_steps, self.obs_dim),    np.float32)
        self.actions    = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.rewards    = np.zeros(self.n_steps, np.float32)
        self.dones      = np.zeros(self.n_steps, np.float32)
        self.values     = np.zeros(self.n_steps, np.float32)
        self.log_probs  = np.zeros(self.n_steps, np.float32)
        self.advantages = np.zeros(self.n_steps, np.float32)
        self.returns    = np.zeros(self.n_steps, np.float32)
        self.ptr  = 0
        self.full = False

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            done: float, value: float, log_prob: float):
        i = self.ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.dones[i]     = done
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value: float,
                                       gamma: float, gae_lambda: float):
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val     = last_value if t == self.n_steps - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta        = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae     = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size: int, normalize_adv: bool = True):
        assert self.full
        indices = np.random.permutation(self.n_steps)
        adv = self.advantages.copy()
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, self.n_steps, batch_size):
            idx = indices[start: start + batch_size]
            yield (
                _np_to_tensor(self.obs[idx],       self.device),
                _np_to_tensor(self.actions[idx],   self.device),
                _np_to_tensor(self.returns[idx],   self.device).view(-1, 1),
                _np_to_tensor(adv[idx],            self.device),
                _np_to_tensor(self.log_probs[idx], self.device),
            )


# ==============================================================================
# SwingControllerAgent — 完整的 PPO Agent
# ==============================================================================

class SwingControllerAgent:
    """
    底层防摆 PPO Agent。

    特点：
    - 无 BC 损失（训练时可选 BC 暖启动，但 PPO 阶段纯 RL）
    - 无 tanh squashing（直接 clip）
    - 独立的 obs_norm
    """

    def __init__(self, config: Optional[dict] = None, device: Optional[torch.device] = None):
        if config is None:
            config = DEFAULT_CONFIG
        self.config = config

        cfg = config.get("swing_controller", {})
        sp  = config.get("space", {})

        self.obs_dim    = int(cfg.get("obs_dim", 35))
        self.action_dim = int(sp.get("action_dim", 7))
        self.dq_max     = np.array(sp.get("dq_max", [0.12]*7), dtype=np.float32)

        # 超参数
        self.gamma       = float(cfg.get("gamma", 0.99))
        self.gae_lambda  = float(cfg.get("gae_lambda", 0.95))
        self.clip_eps    = float(cfg.get("clip_eps", 0.2))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.n_steps     = int(cfg.get("n_steps", 2048))
        self.n_epochs    = int(cfg.get("n_epochs", 4))
        self.batch_size  = int(cfg.get("batch_size", 256))

        # 设备
        if device is None:
            gpu_id = config.get("train", {}).get("gpu_id", 0)
            self.device = (torch.device(f"cuda:{gpu_id}")
                           if torch.cuda.is_available() and gpu_id >= 0
                           else torch.device("cpu"))
        else:
            self.device = device

        hidden_dim = int(cfg.get("hidden_dim", 256))
        n_layers   = int(cfg.get("n_layers", 3))

        self.actor = SwingControllerActor(
            self.obs_dim, self.action_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
            log_std_init=float(cfg.get("log_std_init", -1.0)),
            log_std_min=float(cfg.get("log_std_min", -4.0)),
            log_std_max=float(cfg.get("log_std_max", 0.0)),
        ).to(self.device)

        self.critic = SwingControllerCritic(
            self.obs_dim, hidden_dim, n_layers
        ).to(self.device)

        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg.get("lr_actor", 3e-4)), eps=1e-5)
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg.get("lr_critic", 1e-3)), eps=1e-5)

        self.buffer = SwingRolloutBuffer(
            self.n_steps, self.obs_dim, self.action_dim, self.device)

        self.obs_norm = RunningMeanStd(
            shape=(self.obs_dim,), warm_start=200, clip=10.0)
        self.use_obs_norm = bool(cfg.get("use_obs_norm", True))

        self.total_steps = 0

    # ── 观测归一化 ──

    def normalize_obs(self, obs: np.ndarray, update: bool = True) -> np.ndarray:
        if self.use_obs_norm:
            if update:
                self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    # ── 动作选择 ──

    @torch.no_grad()
    def act(self, norm_obs: np.ndarray,
            deterministic: bool = False
            ) -> Tuple[np.ndarray, float, float]:
        """
        返回 (delta_q_base, log_prob, value)。
        delta_q_base 已 clip 到 ±dq_max。
        """
        s = _np_to_tensor(norm_obs.reshape(1, -1), self.device)
        action, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
        value = self.critic(s)

        dq = action.cpu().numpy().flatten()
        dq = np.clip(dq, -self.dq_max, self.dq_max)
        return dq, log_prob.cpu().item(), value.cpu().item()

    # ── PPO 更新 ──

    def update(self) -> Dict[str, float]:
        """标准 PPO 更新（无 BC 损失、无 KL 早停简化版）。"""
        if not self.buffer.full:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0}

        total_pl = total_vl = total_el = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_minibatches(self.batch_size):
                obs_b, act_b, ret_b, adv_b, old_lp_b = batch

                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                value = self.critic(obs_b)

                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = F.huber_loss(value, ret_b)
                entropy_loss = -entropy.mean()

                # Critic 更新
                self.opt_critic.zero_grad()
                (0.5 * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                # Actor 更新
                actor_loss = policy_loss + self.entropy_coef * entropy_loss
                self.opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_el += entropy_loss.item()
                n_updates += 1

        n = max(n_updates, 1)
        self.buffer.clear()
        return {
            "policy_loss": total_pl / n,
            "value_loss":  total_vl / n,
            "entropy":     -total_el / n,
        }

    # ── 保存/加载 ──

    def save(self, path: str):
        torch.save({
            "actor":       self.actor.state_dict(),
            "critic":      self.critic.state_dict(),
            "opt_actor":   self.opt_actor.state_dict(),
            "opt_critic":  self.opt_critic.state_dict(),
            "obs_norm":    self.obs_norm.state_dict(),
            "total_steps": self.total_steps,
        }, path)
        print(f"[SwingController] Saved to {path}")

    def load(self, path: str, map_location: Optional[torch.device] = None):
        ck = torch.load(path, map_location=map_location or self.device,
                        weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if "opt_actor" in ck:
            self.opt_actor.load_state_dict(ck["opt_actor"])
        if "opt_critic" in ck:
            self.opt_critic.load_state_dict(ck["opt_critic"])
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])
        self.total_steps = ck.get("total_steps", 0)
        print(f"[SwingController] Loaded from {path} (steps={self.total_steps})")

    def eval_mode(self):
        """设置为评估模式，冻结参数。"""
        self.actor.eval()
        self.critic.eval()
        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False


# ==============================================================================
# 底层观测构建函数
# ==============================================================================
#
# 修正后观测维度（35维）：
#   [0:3]   payload_pos      — payload 世界位置 (x, y, z)
#   [3:6]   payload_vel      — payload 世界速度 (vx, vy, vz)
#   [6:9]   ee_payload_off   — EE-Payload 相对偏移 (dx, dy, dz)  ★防摆核心信号
#   [9:12]  ee_vel           — EE 世界速度 (vx, vy, vz)
#   [12]    tilt             — payload 倾斜角 (rad)
#   [13]    yaw              — payload 偏航角 (rad)
#   [14]    tilt_rate        — 倾斜角速度 (rad/s)   ★新增
#   [15]    yaw_rate         — 偏航角速度 (rad/s)   ★新增
#   [16:23] joint_q          — 当前关节角 (7D)
#   [23:30] last_action      — 上一步执行的 Δq (7D)
#   [30:33] waypoint_err     — 到当前航点的误差 (dx, dy, dz)
#   [33:35] wind_info        — 风力估计 (F, theta)  ★新增
#
# 设计原则：
#   - 不包含障碍物坐标、钢筋误差、阶段编码（高层专用信息）
#   - EE-Payload 偏移替代了 EE 绝对位置（后者与关节角冗余）
#   - wind_info 作为 privileged information，帮助控制器学习风力补偿
# ==============================================================================

# obs 索引常量（环境完整 obs 的布局，与 controller.py 一致）
# 公开导出，供 train_swing_controller.py / test.py 使用
OBS_EE_X, OBS_EE_Y     = 0, 1
OBS_EE_VX, OBS_EE_VY   = 2, 3
OBS_PL_X, OBS_PL_Y     = 4, 5
OBS_PL_VX, OBS_PL_VY   = 6, 7
OBS_EE_Z   = 19
OBS_EE_VZ  = 20
OBS_PL_Z   = 21
OBS_PL_VZ  = 22
OBS_TILT   = 29          # payload_tilt 在原始 obs 的索引
OBS_YAW    = 30          # payload_yaw

SWING_OBS_DIM = 35


def build_swing_obs(env_obs_raw: np.ndarray,
                    env,
                    last_action: np.ndarray,
                    prev_tilt: float = 0.0,
                    prev_yaw: float = 0.0
                    ) -> Tuple[np.ndarray, float, float]:
    """
    从环境的完整观测中构建底层防摆控制器的专用观测。

    Args:
        env_obs_raw: 环境返回的完整 54 维观测
        env: CableRobotEnvWithObstacles 实例
        last_action: 上一步执行的 Δq（7D），由调用者维护
        prev_tilt: 上一步的 tilt 值（用于计算 tilt_rate）
        prev_yaw: 上一步的 yaw 值（用于计算 yaw_rate）

    Returns:
        (swing_obs, current_tilt, current_yaw)
        swing_obs: 35 维 numpy 数组
        current_tilt, current_yaw: 用于下一步的 rate 计算
    """
    obs = env_obs_raw

    # Payload 位置和速度
    payload_pos = np.array([obs[OBS_PL_X], obs[OBS_PL_Y], obs[OBS_PL_Z]])
    payload_vel = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY], obs[OBS_PL_VZ]])

    # EE 位置和速度
    ee_pos = np.array([obs[OBS_EE_X], obs[OBS_EE_Y], obs[OBS_EE_Z]])
    ee_vel = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]])

    # EE-Payload 相对偏移（防摆核心信号）
    ee_payload_offset = ee_pos - payload_pos

    # 姿态
    tilt = float(obs[OBS_TILT])
    yaw  = float(obs[OBS_YAW])

    # 姿态变化率（数值微分）
    dt = env.dt if hasattr(env, 'dt') else 0.1
    tilt_rate = (tilt - prev_tilt) / dt
    yaw_rate  = (yaw - prev_yaw) / dt

    # 关节角
    joint_q = obs[31:38].copy()  # 7D joint positions

    # 航点误差
    if (env._planned_path is not None and
            env.current_wp_idx < len(env._planned_path)):
        wp = env._planned_path[env.current_wp_idx]
        waypoint_err = wp - payload_pos
    else:
        # 使用最终目标位置作为跟踪点
        target_z = env.cfg_insertion.get("target_payload_z", 0.10)
        target_3d = np.array([env.target_pos[0], env.target_pos[1], target_z])
        waypoint_err = target_3d - payload_pos

    # 风力信息（privileged info）
    if hasattr(env, 'get_wind_state'):
        wind_F, wind_theta = env.get_wind_state()
    else:
        wind_F, wind_theta = 0.0, 0.0

    swing_obs = np.concatenate([
        payload_pos,        # [0:3]
        payload_vel,        # [3:6]
        ee_payload_offset,  # [6:9]   ★防摆核心
        ee_vel,             # [9:12]
        [tilt],             # [12]
        [yaw],              # [13]
        [tilt_rate],        # [14]    ★新增
        [yaw_rate],         # [15]    ★新增
        joint_q,            # [16:23]
        last_action,        # [23:30]
        waypoint_err,       # [30:33]
        [wind_F, wind_theta],  # [33:35] ★新增
    ]).astype(np.float32)

    assert swing_obs.shape[0] == SWING_OBS_DIM, \
        f"swing_obs 维度 {swing_obs.shape[0]} != 预期 {SWING_OBS_DIM}"

    return swing_obs, tilt, yaw


# ==============================================================================
# 底层奖励计算函数
# ==============================================================================

def compute_swing_reward(env,
                         env_obs_raw: np.ndarray,
                         action: np.ndarray,
                         last_action: np.ndarray,
                         prev_wp_dist: Optional[float],
                         config: dict
                         ) -> Tuple[float, bool, Dict[str, float]]:
    """
    计算底层防摆控制器的奖励。

    Args:
        env: 环境实例
        env_obs_raw: 环境完整观测
        action: 本步执行的 Δq
        last_action: 上一步的 Δq
        prev_wp_dist: 上一步到航点的距离（用于 progress 计算）
        config: 配置字典

    Returns:
        (reward, terminated, info_dict)
    """
    cfg_rwd = config.get("swing_controller_reward", {})
    cfg_logic = config.get("step_logic", {})
    obs = env_obs_raw

    # 提取状态
    payload_xy = np.array([obs[OBS_PL_X], obs[OBS_PL_Y]])
    payload_z  = float(obs[OBS_PL_Z])
    payload_vel_xy = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY]])
    ee_xy = np.array([obs[OBS_EE_X], obs[OBS_EE_Y]])
    tilt  = float(obs[OBS_TILT])
    yaw   = float(obs[OBS_YAW])

    reward = 0.0
    terminated = False
    info = {}

    # ── 航点进度奖励 ──
    if (env._planned_path is not None and
            env.current_wp_idx < len(env._planned_path)):
        wp = env._planned_path[env.current_wp_idx]
        payload_pos = np.array([payload_xy[0], payload_xy[1], payload_z])
        current_dist = float(np.linalg.norm(payload_pos - wp))
    else:
        target_z = env.cfg_insertion.get("target_payload_z", 0.10)
        target_3d = np.array([env.target_pos[0], env.target_pos[1], target_z])
        payload_pos = np.array([payload_xy[0], payload_xy[1], payload_z])
        current_dist = float(np.linalg.norm(payload_pos - target_3d))

    if prev_wp_dist is not None:
        progress = prev_wp_dist - current_dist
        r_progress = cfg_rwd.get("waypoint_progress_coef", 3.0) * progress
        reward += r_progress
        info["r_progress"] = r_progress

    info["wp_dist"] = current_dist

    # ── 航点到达奖励 ──
    if getattr(env, '_wp_just_advanced', False):
        bonus = cfg_rwd.get("waypoint_reach_bonus", 1.0)
        reward += bonus
        info["r_wp_reach"] = bonus

    # ── 倾斜惩罚 ──
    r_tilt = -cfg_rwd.get("tilt_penalty_coef", 0.5) * tilt
    reward += r_tilt
    info["r_tilt"] = r_tilt

    # ── 水平摆动速度惩罚 ──
    swing_vel = float(np.linalg.norm(payload_vel_xy))
    r_swing_vel = -cfg_rwd.get("swing_vel_penalty_coef", 0.3) * swing_vel
    reward += r_swing_vel
    info["r_swing_vel"] = r_swing_vel

    # ── EE-Payload 摆幅惩罚（★新增，防摆核心） ──
    swing_offset = float(np.linalg.norm(ee_xy - payload_xy))
    r_swing_off = -cfg_rwd.get("swing_offset_penalty_coef", 1.0) * swing_offset
    reward += r_swing_off
    info["r_swing_offset"] = r_swing_off

    # ── 动作平滑惩罚 ──
    r_smooth = -cfg_rwd.get("action_smooth_coef", 0.02) * float(np.linalg.norm(action - last_action))
    reward += r_smooth
    info["r_smooth"] = r_smooth

    # ── 时间步惩罚 ──
    r_step = cfg_rwd.get("step_penalty", -0.01)
    reward += r_step

    # ── 终止条件检查 ──
    # 碰撞检测
    if hasattr(env, '_check_prefab_collision_with_obstacles'):
        hit_obs, _ = env._check_prefab_collision_with_obstacles()
        if hit_obs:
            reward = cfg_rwd.get("collision_penalty", -10.0)
            terminated = True
            info["termination"] = "collision"
            return reward, terminated, info

    # 机器人基座碰撞
    if float(np.linalg.norm(payload_xy)) < 0.03:
        reward = cfg_rwd.get("collision_penalty", -10.0)
        terminated = True
        info["termination"] = "base_collision"
        return reward, terminated, info

    # 失稳检查
    instab_grace = cfg_logic.get("instability_grace_steps", 50)
    if env.current_step >= instab_grace:
        pl_vel = float(np.linalg.norm(np.array([
            obs[OBS_PL_VX], obs[OBS_PL_VY], obs[OBS_PL_VZ]])))
        if (swing_offset > cfg_logic.get("swing_xy_max", 0.25) or
                pl_vel > cfg_logic.get("payload_vel_max", 2.0) or
                tilt > cfg_logic.get("payload_tilt_max", 1.0)):
            reward = cfg_rwd.get("instability_penalty", -5.0)
            terminated = True
            info["termination"] = "instability"
            return reward, terminated, info

    # 到达最终航点（成功）
    if env.reached_final:
        # 检查姿态条件
        cfg_ins = config.get("insertion", {})
        if (tilt < cfg_ins.get("tilt_tolerance", 0.08) and
                abs(yaw) < cfg_ins.get("yaw_tolerance", 0.06)):
            reward += cfg_rwd.get("success_bonus", 10.0)
            info["success"] = True

    # reward clip
    r_min = cfg_logic.get("reward_clip_min", -5.0)
    r_max = cfg_logic.get("reward_clip_max",  5.0)
    reward = float(np.clip(reward, r_min, r_max))

    return reward, terminated, info