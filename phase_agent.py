# ==============================================================================
# phase_agent.py — 三阶段统一 RL Agent (PPO + SAC) v8 (重构版)
#
# v8 重构变更:
#   - 删除 CruiseDualRLAgent / DescentDualRLAgent (use_dual_rl=False)
#   - 保留 LSTM + 普通 MLP 双架构
#   - 保留 PPO/SAC 主类, 保留 HER (descent SAC 用)
# ==============================================================================

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple, deque

from config import DEFAULT_CONFIG


# ==============================================================================
# 通用工具
# ==============================================================================

def np_to_tensor(n, device=None):
    return torch.as_tensor(n, dtype=torch.float32, device=device)

def soft_update(target, source, tau):
    for tp, p in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1 - tau) + p.data * tau)

def orthogonal_init(layer, gain=np.sqrt(2)):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


class RunningMeanStd:
    def __init__(self, shape, warm_start=5000, clip=10.0):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S = np.zeros(shape, dtype=np.float64)
        self.warm_start = warm_start; self.clip = clip
    def update(self, x):
        x = np.asarray(x, dtype=np.float64).flatten()
        self.n += 1
        if self.n == 1:
            self.mean = x.copy(); self.S = np.zeros_like(x)
        else:
            old = self.mean.copy()
            self.mean = old + (x - old) / self.n
            self.S += (x - old) * (x - self.mean)
    @property
    def var(self): return self.S / max(self.n - 1, 1)
    @property
    def std(self): return np.sqrt(self.var + 1e-8)
    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.n < self.warm_start: return x
        return np.clip(
            (x - self.mean.astype(np.float32)) / self.std.astype(np.float32),
            -self.clip, self.clip)
    def state_dict(self): return {"n": self.n, "mean": self.mean.copy(), "S": self.S.copy()}
    def load_state_dict(self, d):
        self.n = d["n"]; self.mean = d["mean"].copy(); self.S = d["S"].copy()


def delay_mdp_extra_dim(config):
    """Extra policy-observation features for the held-observation Delay-MDP."""
    cfg = config.get("delay_mdp", {})
    if not bool(cfg.get("enabled", False)):
        return 0
    dim = 0
    if bool(cfg.get("include_obs_age", True)):
        dim += 1
    hist_steps = max(0, int(cfg.get("action_history_steps", 4)))
    action_dim = max(0, int(cfg.get("action_dim", 7)))
    dim += hist_steps * action_dim
    return int(dim)


# ==============================================================================
# LSTM 用观测历史缓冲
# ==============================================================================

class ObsHistoryBuffer:
    """维护最近 seq_len 步的观测历史, 供 LSTM 使用。"""
    def __init__(self, obs_dim, seq_len=8):
        self.obs_dim = obs_dim; self.seq_len = seq_len
        self.buffer = deque(maxlen=seq_len)
    def reset(self): self.buffer.clear()
    def push(self, obs): self.buffer.append(obs.copy())
    def get_sequence(self):
        if len(self.buffer) == 0:
            return np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        seq = list(self.buffer)
        while len(seq) < self.seq_len:
            seq.insert(0, np.zeros(self.obs_dim, dtype=np.float32))
        return np.array(seq, dtype=np.float32)


# ==============================================================================
# [v14.0] Cable Encoder — 压缩 240 维绳索 obs 到低维隐特征
# ==============================================================================

class CableEncoder(nn.Module):
    """
    将 240 维原始绳索段状态压缩为低维特征向量.

    架构: LayerNorm → Linear(240, hidden) → ReLU → Linear(hidden, output)
    训练: 与 actor/critic 一起端到端训练 (梯度从 PPO loss 回传)

    目的:
      1. 防止 240 维 raw cable obs 淹没其他 20-40 维的关键信号
      2. 让网络学到绳索状态的紧凑表示 (类似 autoencoder 的 encoder 部分)
      3. 降低 actor/critic 输入维度, 减少过拟合
    """
    def __init__(self, raw_dim=240, hidden_dim=128, output_dim=32,
                 n_layers=2, normalize_input=True):
        super().__init__()
        self.raw_dim = raw_dim
        self.output_dim = output_dim
        self.normalize_input = normalize_input

        if normalize_input:
            self.input_norm = nn.LayerNorm(raw_dim)
        else:
            self.input_norm = nn.Identity()

        layers = []
        d_in = raw_dim
        for i in range(n_layers):
            d_out = hidden_dim if i < n_layers - 1 else output_dim
            lin = nn.Linear(d_in, d_out)
            orthogonal_init(lin, gain=np.sqrt(2) if i < n_layers - 1 else 1.0)
            layers.append(lin)
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            d_in = d_out
        self.encoder = nn.Sequential(*layers)

    def forward(self, cable_raw):
        """
        Args:
            cable_raw: (batch, 240) or (batch, seq_len, 240)
        Returns:
            cable_feat: (batch, output_dim) or (batch, seq_len, output_dim)
        """
        normed = self.input_norm(cable_raw)
        return self.encoder(normed)


def build_wind_obs(env, wind_scale_max=16.5):
    """
    [v14.0] 构建 3 维风观测: [speed_norm, cos(dir), sin(dir)]

    从 env 的内部状态读取当前风力和方向.
    如果是测试恒定风 (_test_wind_mode), 读取测试风力.
    否则读取随机游走风力状态.
    """
    try:
        if hasattr(env, 'get_wind_speed_state'):
            wf, wd = env.get_wind_speed_state()
        elif getattr(env, '_test_wind_mode', False):
            wf = float(getattr(env, '_test_wind_speed', 0.0))
            wd = float(getattr(env, '_test_wind_dir', 0.0))
        else:
            wf = float(getattr(env, 'wind_speed', 0.0))
            wd = float(getattr(env, 'wind_theta', 0.0))
    except Exception:
        wf, wd = 0.0, 0.0

    force_norm = min(wf / max(wind_scale_max, 1e-6), 1.0)
    return np.array([force_norm, np.cos(wd), np.sin(wd)], dtype=np.float32)


# ==============================================================================
# 观测索引
# ==============================================================================
OBS_EE_X, OBS_EE_Y = 0, 1
OBS_EE_VX, OBS_EE_VY = 2, 3
OBS_PL_X, OBS_PL_Y = 4, 5
OBS_PL_VX, OBS_PL_VY = 6, 7
OBS_EE_Z = 19; OBS_EE_VZ = 20
OBS_PL_Z = 21; OBS_PL_VZ = 22
OBS_TILT = 29; OBS_YAW = 30
# [v12.3 修订] obs 末尾结构:
#   index 50-53: rebar_errors (4 维)
#   index 54-293: cable_seg_states (240 维 = 4 根 × 10 段 × 6 维 [rel_pos+lin_vel])
OBS_REBAR_ERR_START = 50
OBS_CABLE_START     = 54
OBS_CABLE_TOTAL     = 4 * 10 * 6   # = 240


# ==============================================================================
# 观测构建 (与原版完全一致)
# ==============================================================================

def _normalize_nmpc_action(base_action, env):
    """Normalize a 4D NMPC EE acceleration action for residual observations."""
    if base_action is None:
        return np.zeros(4, np.float32)
    base = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if base.size < 4:
        base = np.pad(base, (0, 4 - base.size))
    ctrl_cfg = env.config.get("controller", {})
    scale = np.array([
        float(ctrl_cfg.get("u_max_xy", 0.8)),
        float(ctrl_cfg.get("u_max_xy", 0.8)),
        float(ctrl_cfg.get("u_max_z", 2.0)),
        float(ctrl_cfg.get("u_max_yaw", 2.0)),
    ], dtype=np.float32)
    scale = np.maximum(scale, 1e-6)
    return np.clip(base[:4] / scale, -1.5, 1.5).astype(np.float32)


def build_cruise_obs(env_obs, env, target_xy, prev_tilt=0.0, prev_yaw=0.0,
                     wind_obs=None, base_action=None):
    """[v14.0] 返回 (core_obs, cable_raw, wind_obs, tilt, yaw)."""
    obs = env_obs; dt = getattr(env, 'dt', 0.1); n_obs_max = env.n_obstacles
    ee_xy = np.array([obs[OBS_EE_X], obs[OBS_EE_Y]])
    ee_vxy = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY]])
    pl_xy = np.array([obs[OBS_PL_X], obs[OBS_PL_Y]])
    pl_vxy = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY]])
    offset_xy = ee_xy - pl_xy
    tilt = float(obs[OBS_TILT]); yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt; yaw_rate = (yaw - prev_yaw) / dt
    dist = float(np.linalg.norm(pl_xy - target_xy))
    obs_data = obs[10:10 + 3 * n_obs_max].copy()
    joint_q  = obs[31:38].copy()
    z_lock = float(env.config.get("cruise_rl", {}).get("z_lock_height", 0.25))
    pl_z = float(obs[OBS_PL_Z]); pl_vz = float(obs[OBS_PL_VZ]); ee_z = float(obs[OBS_EE_Z])
    z_error = pl_z - z_lock; z_dev_norm = np.clip(z_error / 0.10, -1.0, 1.0)
    # [v14.0] 绳索 raw 数据单独提取
    cable_raw = obs[OBS_CABLE_START:OBS_CABLE_START+OBS_CABLE_TOTAL].copy() \
        if len(obs) >= OBS_CABLE_START + OBS_CABLE_TOTAL \
        else np.zeros(OBS_CABLE_TOTAL, np.float32)
    if wind_obs is None:
        wind_obs = np.zeros(3, np.float32)
    base_nmpc_norm = _normalize_nmpc_action(base_action, env)
    include_base = bool(env.config.get("cruise_rl", {}).get(
        "include_nmpc_action_obs", True))
    core_obs = np.concatenate([
        ee_xy, ee_vxy, pl_xy, pl_vxy, offset_xy,
        [tilt, yaw, tilt_rate, yaw_rate], target_xy, [dist],
        obs_data, joint_q, [pl_z, pl_vz, ee_z, z_error, z_dev_norm],
        base_nmpc_norm if include_base else np.zeros(0, np.float32),
    ]).astype(np.float32)
    return core_obs, cable_raw, wind_obs, tilt, yaw


def _normalize_base_dq(base_action, env):
    """Normalize a 7D base-controller delta_q for residual-policy observations."""
    if base_action is None:
        return np.zeros(7, np.float32)
    base = np.asarray(base_action, dtype=np.float32).reshape(-1)
    if base.size < 7:
        base = np.pad(base, (0, 7 - base.size))
    dq_max = np.asarray(
        env.config.get("space", {}).get("dq_max", [0.12] * 7),
        dtype=np.float32)
    dq_max = np.maximum(dq_max[:7], 1e-6)
    return np.clip(base[:7] / dq_max, -1.5, 1.5).astype(np.float32)


def build_descent_obs(env_obs, env, target_xy, prev_tilt=0.0, prev_yaw=0.0,
                      wind_obs=None, base_action=None):
    """[v14.0] 返回 (core_obs, cable_raw, wind_obs, tilt, yaw)."""
    obs = env_obs; dt = getattr(env, 'dt', 0.1)
    target_pz = float(env.config["insertion"]["target_payload_z"])
    ee_pos = np.array([obs[OBS_EE_X], obs[OBS_EE_Y], obs[OBS_EE_Z]])
    ee_vel = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]])
    pl_pos = np.array([obs[OBS_PL_X], obs[OBS_PL_Y], obs[OBS_PL_Z]])
    pl_vel = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY], obs[OBS_PL_VZ]])
    offset = ee_pos - pl_pos
    tilt = float(obs[OBS_TILT]); yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt; yaw_rate = (yaw - prev_yaw) / dt
    pl_target_xy_err = pl_pos[:2] - target_xy
    z_error = float(pl_pos[2] - target_pz)
    rebar_err = obs[OBS_REBAR_ERR_START:OBS_REBAR_ERR_START+4].copy()
    ins_cfg = env.config.get("insertion", {})
    xy_tol = float(ins_cfg.get("xy_tolerance_train_end",
                               ins_cfg.get("xy_tolerance", 0.005)))
    z_tol = float(ins_cfg.get("success_z_tolerance", 0.020))
    tilt_tol = float(ins_cfg.get("tilt_tolerance_train_end",
                                 ins_cfg.get("tilt_tolerance", 0.05)))
    yaw_tol = float(ins_cfg.get("yaw_tolerance_train_end",
                                ins_cfg.get("yaw_tolerance", 0.08)))
    max_steps = float(env.config.get("descent_rl", {}).get("max_steps", 300))
    step_frac = np.clip(float(getattr(env, "current_step", 0)) /
                        max(max_steps, 1.0), 0.0, 1.0)
    insertion_state = np.array([
        np.clip(np.linalg.norm(pl_target_xy_err) / max(xy_tol, 1e-6), 0.0, 5.0),
        np.clip(abs(z_error) / max(z_tol, 1e-6), 0.0, 5.0),
        np.clip(abs(tilt) / max(tilt_tol, 1e-6), 0.0, 5.0),
        np.clip(abs(yaw) / max(yaw_tol, 1e-6), 0.0, 5.0),
        step_frac,
    ], dtype=np.float32)
    # [v14.0] 绳索 raw 数据单独提取
    cable_raw = obs[OBS_CABLE_START:OBS_CABLE_START+OBS_CABLE_TOTAL].copy() \
        if len(obs) >= OBS_CABLE_START + OBS_CABLE_TOTAL \
        else np.zeros(OBS_CABLE_TOTAL, np.float32)
    if wind_obs is None:
        wind_obs = np.zeros(3, np.float32)
    base_dq_norm = _normalize_base_dq(base_action, env)
    core_obs = np.concatenate([
        ee_pos, ee_vel, pl_pos, pl_vel, offset,
        [tilt, yaw, tilt_rate, yaw_rate], target_xy, [target_pz],
        pl_target_xy_err, [z_error], rebar_err, insertion_state,
        base_dq_norm,
    ]).astype(np.float32)
    return core_obs, cable_raw, wind_obs, tilt, yaw


# ==============================================================================
# LSTM-MLP Actor / Critic
# ==============================================================================

class LSTMPhaseActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale,
                 hidden_dim=256, lstm_dim=128, n_layers=2, seq_len=8,
                 log_std_init=-0.5, log_std_min=-2.0, log_std_max=0.3):
        super().__init__()
        self.log_std_min = log_std_min; self.log_std_max = log_std_max
        self.action_dim = action_dim; self.seq_len = seq_len; self.lstm_dim = lstm_dim
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        orthogonal_init(self.obs_embed[0])
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, num_layers=1, batch_first=True)
        head_layers = []; d = lstm_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            head_layers += [lin, nn.ReLU()]; d = hidden_dim
        self.head = nn.Sequential(*head_layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim); orthogonal_init(self.mean_head, gain=0.01)
        safe_init = float(np.clip(log_std_init, log_std_min, log_std_max))
        self.log_std = nn.Parameter(torch.ones(action_dim) * safe_init)

    def _dist(self, obs_seq, hx=None):
        if obs_seq.dim() == 2: obs_seq = obs_seq.unsqueeze(1)
        embedded = self.obs_embed(obs_seq)
        lstm_out, hx_new = self.lstm(embedded, hx)
        feat = lstm_out[:, -1, :]
        head_feat = self.head(feat)
        mean_raw = self.mean_head(head_feat)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        return mean_raw, log_std.exp().expand_as(mean_raw), hx_new

    def get_action(self, obs_seq, deterministic=False, hx=None):
        mean_raw, std, hx_new = self._dist(obs_seq, hx)
        dist = Normal(mean_raw, std)
        u = mean_raw if deterministic else dist.rsample()
        u_tanh = torch.tanh(u); action = u_tanh * self.action_scale
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        return action, log_prob, dist.entropy().sum(-1), hx_new

    def evaluate_actions(self, obs_seq, actions_taken, hx=None):
        mean_raw, std, _ = self._dist(obs_seq, hx)
        u_tanh = (actions_taken / (self.action_scale + 1e-8)).clamp(-1+1e-6, 1-1e-6)
        u = torch.atanh(u_tanh); dist = Normal(mean_raw, std)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        return log_prob, dist.entropy().sum(-1)

    def get_log_std_per_dim(self):
        return self.log_std.detach().cpu().numpy().copy()


class LSTMPhaseCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, lstm_dim=128, n_layers=2, seq_len=8):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        orthogonal_init(self.obs_embed[0])
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, num_layers=1, batch_first=True)
        layers = []; d = lstm_dim
        for i in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            layers += [lin, nn.ReLU()]; d = hidden_dim
        # [v10] DAWN paper: critic 倒数第二层加 LayerNorm,
        # 让 critic 对 base+residual 的微小输入差异敏感 (恢复 representation sensitivity)
        layers.append(nn.LayerNorm(hidden_dim))
        out = nn.Linear(d, 1); orthogonal_init(out, gain=1.0); layers.append(out)
        self.head = nn.Sequential(*layers)

    def forward(self, obs_seq, hx=None):
        if obs_seq.dim() == 2: obs_seq = obs_seq.unsqueeze(1)
        embedded = self.obs_embed(obs_seq)
        lstm_out, _ = self.lstm(embedded, hx)
        return self.head(lstm_out[:, -1, :])


# ==============================================================================
# 兼容 MLP Actor / Critic
# ==============================================================================

class PhaseActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale,
                 hidden_dim=256, n_layers=3,
                 log_std_init=-0.5, log_std_min=-2.0, log_std_max=0.3):
        super().__init__()
        self.log_std_min = log_std_min; self.log_std_max = log_std_max
        self.action_dim = action_dim
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))
        layers = []; d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            layers += [lin, nn.ReLU()]; d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim); orthogonal_init(self.mean_head, gain=0.01)
        safe_init = float(np.clip(log_std_init, log_std_min, log_std_max))
        self.log_std = nn.Parameter(torch.ones(action_dim) * safe_init)

    def _dist(self, s):
        feat = self.backbone(s); mean_raw = self.mean_head(feat)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        return mean_raw, log_std.exp().expand_as(mean_raw)
    def get_action(self, s, deterministic=False):
        mean_raw, std = self._dist(s)
        dist = Normal(mean_raw, std); u = mean_raw if deterministic else dist.rsample()
        u_tanh = torch.tanh(u); action = u_tanh * self.action_scale
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        return action, log_prob, dist.entropy().sum(-1)
    def evaluate_actions(self, s, actions_taken):
        mean_raw, std = self._dist(s)
        u_tanh = (actions_taken / (self.action_scale + 1e-8)).clamp(-1+1e-6, 1-1e-6)
        u = torch.atanh(u_tanh); dist = Normal(mean_raw, std)
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        return log_prob, dist.entropy().sum(-1)
    def get_log_std_per_dim(self):
        return self.log_std.detach().cpu().numpy().copy()


class PhaseCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = []; d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            layers += [lin, nn.ReLU()]; d = hidden_dim
        # [v10] DAWN: critic LayerNorm 倒数第二层
        layers.append(nn.LayerNorm(hidden_dim))
        out = nn.Linear(d, 1); orthogonal_init(out, gain=1.0); layers.append(out)
        self.net = nn.Sequential(*layers)
    def forward(self, s): return self.net(s)


# ==============================================================================
# [v11 Path 2] HER for PPO Descent — episode-level "final" relabeling
# 来源: Crowder et al. 2024 "Hindsight Experience Replay Accelerates PPO" arXiv:2410.22524
# ==============================================================================

class SequenceRolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, seq_len, device,
                 cable_raw_dim=240):
        self.n_steps = n_steps; self.obs_dim = obs_dim
        self.action_dim = action_dim; self.seq_len = seq_len
        self.device = device
        self._cable_raw_dim = cable_raw_dim  # [v14.1]
        self.clear()
    def clear(self):
        self.obs_seqs   = np.zeros((self.n_steps, self.seq_len, self.obs_dim), np.float32)
        self.actions    = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.rewards    = np.zeros(self.n_steps, np.float32)
        self.dones      = np.zeros(self.n_steps, np.float32)
        self.values     = np.zeros(self.n_steps, np.float32)
        self.next_values= np.full(self.n_steps, np.nan, np.float32)
        self.env_ids    = np.full(self.n_steps, -1, np.int32)
        self.log_probs  = np.zeros(self.n_steps, np.float32)
        self.advantages = np.zeros(self.n_steps, np.float32)
        self.returns    = np.zeros(self.n_steps, np.float32)
        # [v14.1] 存 cable_raw 序列供 PPO update 时 re-encode
        self.cable_raw_seqs = np.zeros(
            (self.n_steps, self.seq_len, self._cable_raw_dim), np.float32)
        self.ptr = 0; self.full = False
    def add(self, obs_seq, action, reward, done, value, log_prob,
            cable_raw_seq=None, next_value=None, env_id=-1):
        if self.ptr >= self.n_steps:
            self.full = True
            return False
        i = self.ptr; self.obs_seqs[i] = obs_seq
        self.actions[i] = action
        self.rewards[i] = reward; self.dones[i] = float(done)
        self.values[i] = value; self.log_probs[i] = log_prob
        if next_value is not None:
            self.next_values[i] = float(next_value)
        self.env_ids[i] = int(env_id)
        if cable_raw_seq is not None:
            self.cable_raw_seqs[i] = cable_raw_seq
        self.ptr += 1
        if self.ptr == self.n_steps: self.full = True
        return True
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        if np.all(np.isfinite(self.next_values)) and np.all(self.env_ids >= 0):
            last_gae_by_env = {}
            for t in reversed(range(self.n_steps)):
                eid = int(self.env_ids[t])
                non_terminal = 1.0 - self.dones[t]
                delta = (self.rewards[t] +
                         gamma * self.next_values[t] * non_terminal -
                         self.values[t])
                carry = last_gae_by_env.get(eid, 0.0)
                adv = delta + gamma * gae_lambda * non_terminal * carry
                self.advantages[t] = adv
                last_gae_by_env[eid] = adv
            self.returns = self.advantages + self.values
            return
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = last_value if t == self.n_steps-1 else self.values[t+1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma*next_val*non_terminal - self.values[t]
            last_gae = delta + gamma*gae_lambda*non_terminal*last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values
    def get_minibatches(self, batch_size, normalize_adv=True):
        assert self.full
        indices = np.random.permutation(self.n_steps)
        adv = self.advantages.copy()
        if normalize_adv: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start+batch_size]
            yield (np_to_tensor(self.obs_seqs[idx], self.device),
                   np_to_tensor(self.actions[idx], self.device),
                   np_to_tensor(self.returns[idx], self.device).view(-1, 1),
                   np_to_tensor(adv[idx], self.device),
                   np_to_tensor(self.log_probs[idx], self.device),
                   np_to_tensor(self.cable_raw_seqs[idx], self.device))  # [v14.1]


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, device,
                 cable_raw_dim=240):
        self.n_steps = n_steps; self.obs_dim = obs_dim
        self.action_dim = action_dim; self.device = device
        self._cable_raw_dim = cable_raw_dim  # [v14.1]
        self.clear()
    def clear(self):
        self.obs        = np.zeros((self.n_steps, self.obs_dim), np.float32)
        self.actions    = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.rewards    = np.zeros(self.n_steps, np.float32)
        self.dones      = np.zeros(self.n_steps, np.float32)
        self.values     = np.zeros(self.n_steps, np.float32)
        self.next_values= np.full(self.n_steps, np.nan, np.float32)
        self.env_ids    = np.full(self.n_steps, -1, np.int32)
        self.log_probs  = np.zeros(self.n_steps, np.float32)
        self.advantages = np.zeros(self.n_steps, np.float32)
        self.returns    = np.zeros(self.n_steps, np.float32)
        self.cable_raws = np.zeros(
            (self.n_steps, self._cable_raw_dim), np.float32)  # [v14.1]
        self.ptr = 0; self.full = False
    def add(self, obs, action, reward, done, value, log_prob,
            cable_raw=None, next_value=None, env_id=-1):
        if self.ptr >= self.n_steps:
            self.full = True
            return False
        i = self.ptr; self.obs[i] = obs; self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done); self.values[i] = value; self.log_probs[i] = log_prob
        if next_value is not None:
            self.next_values[i] = float(next_value)
        self.env_ids[i] = int(env_id)
        if cable_raw is not None:
            self.cable_raws[i] = cable_raw
        self.ptr += 1
        if self.ptr == self.n_steps: self.full = True
        return True
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        if np.all(np.isfinite(self.next_values)) and np.all(self.env_ids >= 0):
            last_gae_by_env = {}
            for t in reversed(range(self.n_steps)):
                eid = int(self.env_ids[t])
                non_terminal = 1.0 - self.dones[t]
                delta = (self.rewards[t] +
                         gamma * self.next_values[t] * non_terminal -
                         self.values[t])
                carry = last_gae_by_env.get(eid, 0.0)
                adv = delta + gamma * gae_lambda * non_terminal * carry
                self.advantages[t] = adv
                last_gae_by_env[eid] = adv
            self.returns = self.advantages + self.values
            return
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = last_value if t == self.n_steps-1 else self.values[t+1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma*next_val*non_terminal - self.values[t]
            last_gae = delta + gamma*gae_lambda*non_terminal*last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values
    def get_minibatches(self, batch_size, normalize_adv=True):
        assert self.full
        indices = np.random.permutation(self.n_steps)
        adv = self.advantages.copy()
        if normalize_adv: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start+batch_size]
            yield (np_to_tensor(self.obs[idx], self.device),
                   np_to_tensor(self.actions[idx], self.device),
                   np_to_tensor(self.returns[idx], self.device).view(-1, 1),
                   np_to_tensor(adv[idx], self.device),
                   np_to_tensor(self.log_probs[idx], self.device),
                   np_to_tensor(self.cable_raws[idx], self.device))  # [v14.1]


# ==============================================================================
# PPO Agent
# ==============================================================================

PPOResult = namedtuple("PPOResult", [
    "policy_loss", "value_loss", "entropy_loss",
    "approx_kl", "clip_fraction", "total_loss", "entropy_coef_used"])
PPO_ZERO = PPOResult(0., 0., 0., 0., 0., 0., 0.)


class PPOPhaseAgent:

    def __init__(self, phase_name, config=None):
        if config is None: config = DEFAULT_CONFIG
        self.config = config; self.phase_name = phase_name
        phase_cfg = config[f"{phase_name}_rl"]; cfg_ppo = config["ppo"]
        self.base_obs_dim = int(phase_cfg["obs_dim"])
        self.delay_mdp_extra_dim = delay_mdp_extra_dim(config)
        self.obs_dim = self.base_obs_dim + self.delay_mdp_extra_dim
        self.action_dim = int(phase_cfg["action_dim"])
        self.gamma = float(cfg_ppo["gamma"])
        self.gae_lambda = float(cfg_ppo["gae_lambda"])
        self.clip_eps = float(cfg_ppo["clip_eps"])
        self.value_loss_coef = float(cfg_ppo["value_loss_coef"])
        self.max_grad_norm = float(cfg_ppo["max_grad_norm"])
        self.n_steps = int(cfg_ppo["n_steps"])
        self.n_epochs = int(cfg_ppo["n_epochs"])
        self.batch_size = int(cfg_ppo["batch_size"])
        self.norm_adv = bool(cfg_ppo["normalize_advantages"])
        self.target_kl = float(cfg_ppo.get("target_kl", 0.03))

        self.use_lstm = bool(cfg_ppo.get("use_lstm", True))
        self.seq_len = int(cfg_ppo.get("seq_len", 8))
        lstm_dim = int(cfg_ppo.get("lstm_dim", 128))

        # [v14.2a] phase-specific entropy annealing (descent 需要更快退火)
        _ent_start_key = f"{phase_name}_entropy_coef_start"
        _ent_end_key   = f"{phase_name}_entropy_coef_end"
        _ent_steps_key = f"{phase_name}_entropy_coef_anneal_steps"
        self.entropy_coef_start = float(phase_cfg.get(
            _ent_start_key, cfg_ppo.get("entropy_coef_start",
                                         cfg_ppo.get("entropy_coef", 0.05))))
        self.entropy_coef_end = float(phase_cfg.get(
            _ent_end_key, cfg_ppo.get("entropy_coef_end", 0.005)))
        self.entropy_coef_anneal_steps = int(phase_cfg.get(
            _ent_steps_key, cfg_ppo.get("entropy_coef_anneal_steps", 600_000)))
        self.entropy_coef = self.entropy_coef_start
        self.use_obs_norm = bool(cfg_ppo["use_obs_norm"])
        self.obs_norm = RunningMeanStd(
            shape=(self.obs_dim,),
            warm_start=int(cfg_ppo.get("obs_norm_warm_start", 5000)),
            clip=float(cfg_ppo["obs_norm_clip"]))
        self._freeze_obs_norm = bool(cfg_ppo.get("freeze_obs_norm", False))
        # [v10] phase-specific log_std_floor: cruise/descent 优先用各自的设置
        _floor_init_key = f"{phase_name}_log_std_floor_init"
        _floor_final_key = f"{phase_name}_log_std_floor_final"
        self._log_std_floor_init  = float(phase_cfg.get(
            _floor_init_key, cfg_ppo.get("log_std_floor_init", 0.0)))
        self._log_std_floor_final = float(phase_cfg.get(
            _floor_final_key, cfg_ppo.get("log_std_floor_final", -1.5)))
        self._log_std_floor_steps = int(cfg_ppo.get("log_std_floor_steps", 800_000))
        self._plasticity_reset_interval = int(cfg_ppo.get("plasticity_reset_interval", 200_000))
        self._last_plasticity_reset = 0

        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))

        # ── 动作尺度 ──────────────────────────────────────────────────────────
        ee_cfg = config.get("ee_control", {})
        cr_cfg = config.get("cruise_rl", {})
        _use_nmpc_base = (phase_name == "cruise" and bool(cr_cfg.get("use_nmpc_base", False)))
        _nmpc_residual = (phase_name == "cruise" and
                          bool(cr_cfg.get("nmpc_residual_mode", _use_nmpc_base)))

        if self.action_dim == 3 and _nmpc_residual:
            axy = float(cr_cfg.get("residual_acc_max_xy_rl", 0.08))
            az = float(cr_cfg.get("residual_acc_max_z_rl", 0.10))
            action_scale = [axy, axy, az]
        elif self.action_dim == 3:
            axy = float(phase_cfg.get("residual_acc_max_xy",
                        phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 0.5))))
            az  = float(phase_cfg.get("residual_acc_max_z",
                        phase_cfg.get("acc_max_z",  ee_cfg.get("acc_max_z",  1.0))))
            action_scale = [axy, axy, az]
        elif _use_nmpc_base:
            # [v8] action_scale = residual_acc_max_xy_rl
            arl = float(cr_cfg.get("residual_acc_max_xy_rl", 0.25))
            action_scale = [arl, arl]
        else:
            axy = float(phase_cfg.get("residual_acc_max_xy",
                        ee_cfg.get("acc_max_xy", 0.60)))
            action_scale = [axy, axy]

        # ── log_std 参数: 支持 phase-specific override ──────────────────────
        _ls_init_key = f"{phase_name}_log_std_init"
        _ls_max_key  = f"{phase_name}_log_std_max"
        _log_std_init = float(phase_cfg.get(
            _ls_init_key, cfg_ppo.get("log_std_init", -0.5)))
        _log_std_min  = float(cfg_ppo.get("log_std_min", -3.0))
        _log_std_max  = float(phase_cfg.get(
            _ls_max_key, cfg_ppo.get("log_std_max", 0.0)))

        # ── 网络构建 ──────────────────────────────────────────────────────────
        if self.use_lstm:
            self.actor = LSTMPhaseActor(
                self.obs_dim, self.action_dim, action_scale,
                hidden_dim=int(cfg_ppo["hidden_dim"]), lstm_dim=lstm_dim,
                n_layers=2, seq_len=self.seq_len,
                log_std_init=_log_std_init,
                log_std_min=_log_std_min,
                log_std_max=_log_std_max).to(self.device)
            self.critic = LSTMPhaseCritic(
                self.obs_dim, hidden_dim=int(cfg_ppo["hidden_dim"]),
                lstm_dim=lstm_dim, n_layers=2, seq_len=self.seq_len).to(self.device)
            self.buffer = SequenceRolloutBuffer(
                self.n_steps, self.obs_dim, self.action_dim, self.seq_len, self.device,
                cable_raw_dim=240)  # [v14.1]
        else:
            self.actor = PhaseActor(
                self.obs_dim, self.action_dim, action_scale,
                hidden_dim=int(cfg_ppo["hidden_dim"]),
                n_layers=int(cfg_ppo["n_layers"]),
                log_std_init=_log_std_init,
                log_std_min=_log_std_min,
                log_std_max=_log_std_max).to(self.device)
            self.critic = PhaseCritic(
                self.obs_dim, hidden_dim=int(cfg_ppo["hidden_dim"]),
                n_layers=int(cfg_ppo["n_layers"])).to(self.device)
            self.buffer = RolloutBuffer(
                self.n_steps, self.obs_dim, self.action_dim, self.device,
                cable_raw_dim=240)  # [v14.1]

        self.obs_history = ObsHistoryBuffer(self.obs_dim, self.seq_len)
        self._lr_actor  = float(cfg_ppo["lr_actor"])
        self._lr_critic = float(cfg_ppo["lr_critic"])

        # ── [v14.0] CableEncoder — 压缩 240 维 cable obs ─────────────────────
        ce_cfg = config.get("cable_encoder", {})
        self._cable_raw_dim = int(ce_cfg.get("raw_dim", 240))
        self._cable_out_dim = int(ce_cfg.get("output_dim", 32))
        self._wind_dim = 3
        self.cable_encoder = CableEncoder(
            raw_dim=self._cable_raw_dim,
            hidden_dim=int(ce_cfg.get("hidden_dim", 128)),
            output_dim=self._cable_out_dim,
            n_layers=int(ce_cfg.get("n_layers", 2)),
            normalize_input=bool(ce_cfg.get("normalize_input", True)),
        ).to(self.device)
        # [v14.2] 冻结 CableEncoder (随机投影)
        # v14.1 尝试解冻端到端训练, 但导致 approx_kl 极端尖峰 (>2.0)
        # 原因: PPO update 时 encoder 参数变化导致 obs re-encode 与 collect 不一致
        # 冻结的随机投影已足够解决 240 维淹没问题, 且训练完全稳定
        for p in self.cable_encoder.parameters():
            p.requires_grad = False
        self.cable_encoder.eval()

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=self._lr_actor,  eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self._lr_critic, eps=1e-5)
        self.total_steps = 0; self._last_result = PPO_ZERO

        # ── [v9] Cruise 残差: actor 输出层硬零初始化, 完全替代 BC ──────────────
        # 来源: Jeon et al. 2025 (Residual MPC), Ankile et al. 2024 (ResiP).
        # 零初始化保证训练初期 RL 残差 ≈ 0, 让 NMPC 单独工作; 梯度仍可正常流动,
        # RL 在 SR 已达基线后自然学到微调.
        if phase_name == "cruise" and _use_nmpc_base:
            self._zero_init_residual_actor()

    def _zero_init_residual_actor(self):
        """[v10] 残差 actor 输出层硬零初始化, 替代 BC."""
        import math
        cruise_cfg = self.config.get("cruise_rl", {})
        init_val = float(cruise_cfg.get("cruise_log_std_init", -2.0))
        with torch.no_grad():
            # mean head 权重 + bias 置零
            self.actor.mean_head.weight.zero_()
            self.actor.mean_head.bias.zero_()
            # [v10] 从 cruise_rl.cruise_log_std_init 读取, 默认 -2.0 (std ≈ 0.135)
            self.actor.log_std.data.fill_(init_val)
        print(f"  [PPO-cruise] v10: actor 输出层零初始化, log_std={init_val:.2f}, "
              f"std≈{math.exp(init_val):.3f}")

    # ── 标准接口 ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_obs(self, core_obs, cable_raw, wind_obs):
        """[v14.0] 将 (core, cable_raw, wind) 编码合并为最终 obs.

        CableEncoder 将 240 维 cable 压缩到 32 维, 然后与 core + wind concat.
        这在 inference 阶段调用 (no_grad), 结果存入 buffer.

        Args:
            core_obs: np.array, 非绳索部分 (20-40 维)
            cable_raw: np.array, 240 维绳索原始数据
            wind_obs: np.array, 3 维风力观测
        Returns:
            final_obs: np.array, shape (obs_dim,)
        """
        cable_arr = np.asarray(cable_raw, dtype=np.float32).reshape(-1)
        if cable_arr.size == self._cable_out_dim:
            cable_feat = cable_arr
        else:
            cable_feat = self.encode_cable(cable_arr)
        return np.concatenate([core_obs, cable_feat, wind_obs]).astype(np.float32)

    def encode_cable(self, cable_raw):
        cable_t = torch.from_numpy(
            np.asarray(cable_raw, dtype=np.float32).reshape(1, -1)).to(self.device)
        with torch.no_grad():
            return self.cable_encoder(cable_t).cpu().numpy().flatten()

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm: update = False
        if self.use_obs_norm:
            if update: self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    def reset_history(self):
        self.obs_history.reset()
        # [v14.1] cable_raw 历史 (LSTM 模式下存最近 seq_len 步的 cable_raw)
        self._cable_raw_history = deque(maxlen=self.seq_len)

    def make_obs_history(self):
        return ObsHistoryBuffer(self.obs_dim, self.seq_len)

    def make_cable_history(self):
        return deque(maxlen=self.seq_len)

    def _get_cable_raw_sequence_from(self, cable_history):
        seq = list(cable_history)
        while len(seq) < self.seq_len:
            seq.insert(0, np.zeros(240, dtype=np.float32))
        return np.asarray(seq, dtype=np.float32)

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        self.obs_history.push(norm_obs)
        if self.use_lstm:
            obs_seq = self.obs_history.get_sequence()
            s = np_to_tensor(obs_seq, self.device).unsqueeze(0)
            action, log_prob, _, _ = self.actor.get_action(s, deterministic=deterministic)
            value = self.critic(s)
            return action.cpu().numpy().flatten(), log_prob.cpu().item(), value.cpu().item()
        else:
            s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
            action, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
            value = self.critic(s)
            return action.cpu().numpy().flatten(), log_prob.cpu().item(), value.cpu().item()

    @torch.no_grad()
    def act_with_history(self, norm_obs, obs_history, deterministic=False):
        """PPO inference with caller-owned history, used by vectorized envs."""
        if obs_history is None:
            return self.act(norm_obs, deterministic=deterministic)
        obs_history.push(norm_obs)
        if self.use_lstm:
            obs_seq = obs_history.get_sequence()
            s = np_to_tensor(obs_seq, self.device).unsqueeze(0)
            action, log_prob, _, _ = self.actor.get_action(s, deterministic=deterministic)
            value = self.critic(s)
            return action.cpu().numpy().flatten(), log_prob.cpu().item(), value.cpu().item()
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        action, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
        value = self.critic(s)
        return action.cpu().numpy().flatten(), log_prob.cpu().item(), value.cpu().item()

    @torch.no_grad()
    def get_value_for_state(self, norm_obs):
        if self.use_lstm:
            obs_seq = self.obs_history.get_sequence()
            s = np_to_tensor(obs_seq, self.device).unsqueeze(0)
            return self.critic(s).cpu().item()
        else:
            s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
            return self.critic(s).cpu().item()

    @torch.no_grad()
    def get_value_for_obs_sequence(self, obs_seq):
        s = np_to_tensor(np.asarray(obs_seq, np.float32), self.device).unsqueeze(0)
        return self.critic(s).cpu().item()

    def get_log_std_per_dim(self): return self.actor.get_log_std_per_dim()

    def push_cable_raw(self, cable_raw):
        """[v14.1] 每 step 存 cable_raw, 与 obs_history 同步."""
        if not hasattr(self, '_cable_raw_history'):
            self._cable_raw_history = deque(maxlen=self.seq_len)
        self._cable_raw_history.append(cable_raw.copy())

    def _get_cable_raw_sequence(self):
        """[v14.1] 获取最近 seq_len 步的 cable_raw 序列."""
        if not hasattr(self, '_cable_raw_history'):
            return np.zeros((self.seq_len, 240), dtype=np.float32)
        seq = list(self._cable_raw_history)
        while len(seq) < self.seq_len:
            seq.insert(0, np.zeros(240, dtype=np.float32))
        return np.array(seq, dtype=np.float32)

    def add_to_buffer(self, norm_obs, action, reward, done, value, log_prob,
                      cable_raw=None):
        """[v14.1] 新增 cable_raw 参数, 存入 buffer 供 update 时 re-encode."""
        if self.use_lstm:
            obs_seq = self.obs_history.get_sequence()
            cable_raw_seq = self._get_cable_raw_sequence()
            self.buffer.add(obs_seq, action, reward, done, value, log_prob,
                            cable_raw_seq=cable_raw_seq)
        else:
            self.buffer.add(norm_obs, action, reward, done, value, log_prob,
                            cable_raw=cable_raw)

    def add_to_buffer_with_history(self, norm_obs, action, reward, done, value,
                                   log_prob, obs_history=None,
                                   cable_history=None, cable_raw=None,
                                   next_value=None, env_id=-1):
        if obs_history is None:
            return self.add_to_buffer(norm_obs, action, reward, done, value,
                                      log_prob, cable_raw=cable_raw)
        if self.use_lstm:
            obs_seq = obs_history.get_sequence()
            cable_seq = (self._get_cable_raw_sequence_from(cable_history)
                         if cable_history is not None else None)
            self.buffer.add(obs_seq, action, reward, done, value, log_prob,
                            cable_raw_seq=cable_seq,
                            next_value=next_value, env_id=env_id)
        else:
            self.buffer.add(norm_obs, action, reward, done, value, log_prob,
                            cable_raw=cable_raw,
                            next_value=next_value, env_id=env_id)

    def _maybe_reset_plasticity(self):
        if self._plasticity_reset_interval <= 0: return
        if self.total_steps - self._last_plasticity_reset < self._plasticity_reset_interval: return
        for opt in [self.opt_actor, self.opt_critic]:
            for group in opt.param_groups:
                for p in group['params']:
                    if p in opt.state and 'exp_avg' in opt.state[p]:
                        opt.state[p]['exp_avg'].zero_()
        self._last_plasticity_reset = self.total_steps
        print(f"  [Plasticity] Adam reset @ step {self.total_steps}")

    def reset_adam_state(self, reason=""):
        """[v10] 强制重置 Adam 一阶 + 二阶矩, 用于课程倒退后清除死局轨迹的影响。

        Adam 的 exp_avg / exp_avg_sq 在死局 episode 上累积过大值, 会让倒退后
        即使梯度方向正确, 学习率被抑制无法快速恢复。
        来源: Ashley et al. 2021 (arXiv:2102.07686), Asadi et al. 2023.
        """
        for opt in [self.opt_actor, self.opt_critic]:
            for group in opt.param_groups:
                for p in group['params']:
                    state = opt.state.get(p, {})
                    for key in ['exp_avg', 'exp_avg_sq']:
                        if key in state and torch.is_tensor(state[key]):
                            state[key].zero_()
                    if 'step' in state:
                        if torch.is_tensor(state['step']):
                            state['step'].zero_()
                        else:
                            state['step'] = 0
        self._last_plasticity_reset = self.total_steps
        print(f"  [Adam reset] @ step {self.total_steps} ({reason})")

    def _update_entropy_coef(self, global_ts=None):
        ts = global_ts if global_ts is not None else self.total_steps
        frac = min(ts / max(self.entropy_coef_anneal_steps, 1), 1.0)
        self.entropy_coef = self.entropy_coef_start + \
            frac * (self.entropy_coef_end - self.entropy_coef_start)

    def update(self, global_ts=None):
        if not self.buffer.full: return PPO_ZERO
        self._maybe_reset_plasticity()
        self._update_entropy_coef(global_ts=global_ts)
        total_pl = total_vl = total_el = total_kl = total_cf = 0.0
        n_updates = 0; stop_early = False

        for epoch in range(self.n_epochs):
            if stop_early: break
            for batch in self.buffer.get_minibatches(self.batch_size, self.norm_adv):
                obs_b, act_b, ret_b, adv_b, old_lp_b, _cable_b = batch
                # [v14.2] encoder 冻结, obs_b 中的 cable_feat 已经是正确的
                # _cable_b 不使用 (保留 buffer 接口兼容性)

                value = self.critic(obs_b)
                value_loss = F.huber_loss(value, ret_b)
                self.opt_critic.zero_grad()
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1-self.clip_eps, 1+self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                actor_total = policy_loss + self.entropy_coef * entropy_loss
                self.opt_actor.zero_grad(); actor_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()
                with torch.no_grad():
                    frac = min(self.total_steps / max(self._log_std_floor_steps, 1), 1.0)
                    floor = self._log_std_floor_init + \
                        frac * (self._log_std_floor_final - self._log_std_floor_init)
                    self.actor.log_std.data.clamp_(min=floor)
                    approx_kl = (old_lp_b - new_lp).mean().abs().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                total_pl += policy_loss.item(); total_vl += value_loss.item()
                total_el += entropy_loss.item()
                total_kl += approx_kl; total_cf += clip_frac; n_updates += 1
                if approx_kl > self.target_kl: stop_early = True; break
        n = max(n_updates, 1)
        result = PPOResult(total_pl/n, total_vl/n, total_el/n,
                           total_kl/n, total_cf/n,
                           (total_pl+total_vl+total_el)/n,
                           self.entropy_coef)
        self._last_result = result; self.buffer.clear(); return result

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "total_steps": self.total_steps,
            "obs_norm": self.obs_norm.state_dict(), "phase_name": self.phase_name,
            "entropy_coef": self.entropy_coef, "use_lstm": self.use_lstm,
            "cable_encoder": self.cable_encoder.state_dict(),  # [v14.0]
            "base_obs_dim": self.base_obs_dim,
            "delay_mdp_extra_dim": self.delay_mdp_extra_dim,
        }, path)

    def _load_module_state_adapt_obs_dim(self, module, state, name):
        target = module.state_dict()
        adapted = {}
        changed_shape = False
        skipped = []
        for key, src in state.items():
            if key not in target:
                continue
            dst = target[key]
            if tuple(src.shape) == tuple(dst.shape):
                adapted[key] = src.to(device=dst.device, dtype=dst.dtype)
                continue
            input_weight_keys = {"obs_embed.0.weight", "backbone.0.weight",
                                 "net.0.weight"}
            if (key in input_weight_keys and src.ndim == 2 and dst.ndim == 2 and
                    src.shape[0] == dst.shape[0]):
                new_tensor = dst.clone()
                cols = min(src.shape[1], dst.shape[1])
                new_tensor[:, :cols] = src[:, :cols].to(
                    device=dst.device, dtype=dst.dtype)
                if dst.shape[1] > cols:
                    new_tensor[:, cols:] = 0.0
                adapted[key] = new_tensor
                changed_shape = True
                continue
            skipped.append(key)
        missing, unexpected = module.load_state_dict(adapted, strict=False)
        if changed_shape or skipped or missing or unexpected:
            print(f"  [PPO load] {name}: adapted checkpoint tensors for "
                  f"obs_dim={self.obs_dim}.")
        return bool(changed_shape or skipped)

    def _load_obs_norm_adapt_obs_dim(self, state):
        mean = np.asarray(state.get("mean", []), dtype=np.float64).reshape(-1)
        S = np.asarray(state.get("S", []), dtype=np.float64).reshape(-1)
        if mean.shape == self.obs_norm.mean.shape and S.shape == self.obs_norm.S.shape:
            self.obs_norm.load_state_dict(state)
            return
        self.obs_norm.n = int(state.get("n", 0))
        self.obs_norm.mean.fill(0.0)
        self.obs_norm.S.fill(max(self.obs_norm.n - 1, 1))
        cols = min(mean.size, self.obs_norm.mean.size)
        if cols > 0:
            self.obs_norm.mean[:cols] = mean[:cols]
        cols = min(S.size, self.obs_norm.S.size)
        if cols > 0:
            self.obs_norm.S[:cols] = S[:cols]
        print(f"  [PPO load] obs_norm adapted from {mean.size} to "
              f"{self.obs_norm.mean.size} dims.")

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        adapted_actor = self._load_module_state_adapt_obs_dim(
            self.actor, ck["actor"], "actor")
        adapted_critic = self._load_module_state_adapt_obs_dim(
            self.critic, ck["critic"], "critic")
        adapted = adapted_actor or adapted_critic
        if not adapted and "opt_actor" in ck:
            try: self.opt_actor.load_state_dict(ck["opt_actor"])
            except Exception as e: print(f"  [PPO load] skipped actor optimizer: {e}")
        if not adapted and "opt_critic" in ck:
            try: self.opt_critic.load_state_dict(ck["opt_critic"])
            except Exception as e: print(f"  [PPO load] skipped critic optimizer: {e}")
        if adapted:
            print("  [PPO load] optimizer state reset because obs_dim changed.")
        for group in self.opt_actor.param_groups:
            group["lr"] = self._lr_actor
        for group in self.opt_critic.param_groups:
            group["lr"] = self._lr_critic
        self.total_steps = ck.get("total_steps", 0)
        if "entropy_coef" in ck:
            self.entropy_coef = float(ck["entropy_coef"])
        if "obs_norm" in ck: self._load_obs_norm_adapt_obs_dim(ck["obs_norm"])
        # [v14.0] cable encoder 加载 (保持 frozen 投影一致性)
        if "cable_encoder" in ck:
            self.cable_encoder.load_state_dict(ck["cable_encoder"])


# ==============================================================================
# SAC Actor / Critic / Buffer / Agent
# ==============================================================================

class SACPhaseActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale,
                 hidden_dim=256, n_layers=3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min; self.log_std_max = log_std_max
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))
        layers = []; d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            layers += [lin, nn.ReLU()]; d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim); orthogonal_init(self.mean_head, gain=0.01)
        self.log_std_head = nn.Linear(hidden_dim, action_dim); orthogonal_init(self.log_std_head, gain=0.01)
    def forward(self, s):
        feat = self.backbone(s)
        return self.mean_head(feat), self.log_std_head(feat).clamp(self.log_std_min, self.log_std_max)
    def sample(self, s):
        mean, log_std = self.forward(s); std = log_std.exp()
        dist = Normal(mean, std); u = dist.rsample(); u_tanh = torch.tanh(u)
        action = u_tanh * self.action_scale
        log_prob = dist.log_prob(u).sum(-1) - torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        return action, log_prob
    def deterministic(self, s):
        mean, _ = self.forward(s); return torch.tanh(mean) * self.action_scale


class SACPhaseCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        in_dim = obs_dim + action_dim
        def _make():
            layers = []; d = in_dim
            for _ in range(n_layers):
                lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
                layers += [lin, nn.ReLU()]; d = hidden_dim
            # [v11 Path 1] DAWN: critic 倒数第二层 LayerNorm, 让 critic 对 base+residual
            # 的微小输入差异敏感. 同 PPO critic, 来源 Ma et al. 2026 arXiv:2602.10539.
            layers.append(nn.LayerNorm(hidden_dim))
            out = nn.Linear(d, 1); orthogonal_init(out, gain=1.0); layers.append(out)
            return nn.Sequential(*layers)
        self.q1 = _make(); self.q2 = _make()
    def forward(self, s, a):
        sa = torch.cat([s, a], -1); return self.q1(sa), self.q2(sa)


class SimpleReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim, **kw):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.obs      = np.zeros((max_size, obs_dim), np.float32)
        self.actions  = np.zeros((max_size, action_dim), np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), np.float32)
        self.rewards  = np.zeros((max_size, 1), np.float32)
        self.dones    = np.zeros((max_size, 1), np.float32)
    def add(self, obs, action, next_obs, reward, done, achieved_pos=None):
        i = self.ptr
        self.obs[i] = obs; self.actions[i] = action; self.next_obs[i] = next_obs
        self.rewards[i] = reward; self.dones[i] = float(done)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    def flush_episode_with_her(self, *a, **kw): pass
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.obs[idx], self.actions[idx], self.next_obs[idx], \
            self.rewards[idx], self.dones[idx]
    @property
    def is_ready(self): return self.size >= 1000


class HERReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim, her_k=4,
                 success_bonus=50.0, her_reward_scale=0.3, **kw):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.her_k = her_k; self.her_reward = success_bonus * her_reward_scale
        self.obs      = np.zeros((max_size, obs_dim), np.float32)
        self.actions  = np.zeros((max_size, action_dim), np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), np.float32)
        self.rewards  = np.zeros((max_size, 1), np.float32)
        self.dones    = np.zeros((max_size, 1), np.float32)
        self._ep = []
    def _store(self, obs, action, next_obs, reward, done):
        i = self.ptr
        self.obs[i] = obs; self.actions[i] = action; self.next_obs[i] = next_obs
        self.rewards[i] = reward; self.dones[i] = float(done)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    def add(self, obs, action, next_obs, reward, done, achieved_pos=None):
        self._store(obs, action, next_obs, reward, done)
        if achieved_pos is not None:
            self._ep.append((obs.copy(), action.copy(), next_obs.copy(),
                             float(reward), bool(done),
                             np.array(achieved_pos, np.float32)))
    def flush_episode_with_her(self, target_xy, target_z, obs_relabel_fn=None):
        ep = self._ep; T = len(ep)
        if T < 2:
            self._ep = []; return
        for t in range(T):
            n_future = min(self.her_k, T-t-1)
            if n_future <= 0: continue
            future_ts = np.random.randint(t+1, T, size=n_future)
            for ft in future_ts:
                achieved = ep[ft][5]
                if obs_relabel_fn is not None:
                    try:
                        her_obs  = obs_relabel_fn(ep[t][0], achieved[:2], float(achieved[2]))
                        her_nobs = obs_relabel_fn(ep[t][2], achieved[:2], float(achieved[2]))
                    except Exception:
                        her_obs = ep[t][0]; her_nobs = ep[t][2]
                else:
                    her_obs = ep[t][0]; her_nobs = ep[t][2]
                self._store(her_obs, ep[t][1], her_nobs, self.her_reward, False)
        self._ep = []
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.obs[idx], self.actions[idx], self.next_obs[idx], \
            self.rewards[idx], self.dones[idx]
    @property
    def is_ready(self): return self.size >= 1000


SACResult = namedtuple("SACResult",
    ["critic_loss", "actor_loss", "alpha_loss", "alpha", "q_mean"])
SAC_ZERO = SACResult(0., 0., 0., 0., 0.)


class SACPhaseAgent:
    def __init__(self, phase_name, config=None):
        if config is None: config = DEFAULT_CONFIG
        self.config = config; self.phase_name = phase_name
        phase_cfg = config[f"{phase_name}_rl"]; cfg_sac = config["sac"]
        self.obs_dim = int(phase_cfg["obs_dim"])
        self.action_dim = int(phase_cfg["action_dim"])
        self.gamma = float(cfg_sac["gamma"]); self.tau = float(cfg_sac["tau"])
        self.batch_size = int(cfg_sac["batch_size"])
        self.warmup_steps = int(cfg_sac["warmup_steps"])
        self.reward_scale = float(cfg_sac.get("reward_scale", 1.0))
        self.update_interval = int(cfg_sac.get("update_interval", 1))
        self.updates_per_step = int(cfg_sac.get("updates_per_step", 1))
        self.critic_grad_clip = float(cfg_sac["critic_grad_clip"])
        self.actor_grad_clip = float(cfg_sac["actor_grad_clip"])
        self.use_obs_norm = bool(cfg_sac["use_obs_norm"])
        self.obs_norm = RunningMeanStd(
            shape=(self.obs_dim,),
            warm_start=int(cfg_sac.get("obs_norm_warm_start", 5000)),
            clip=float(cfg_sac["obs_norm_clip"]))
        self._freeze_obs_norm = False
        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))
        hidden = int(cfg_sac["hidden_dim"]); n_layers = int(cfg_sac["n_layers"])

        # ── 动作尺度 ──────────────────────────────────────────────────────────
        ee_cfg = config.get("ee_control", {})
        cr_cfg = config.get("cruise_rl", {})
        _use_nmpc_base = (phase_name == "cruise" and bool(cr_cfg.get("use_nmpc_base", False)))
        _nmpc_residual = (phase_name == "cruise" and
                          bool(cr_cfg.get("nmpc_residual_mode", _use_nmpc_base)))

        if self.action_dim == 3 and _nmpc_residual:
            axy = float(cr_cfg.get("residual_acc_max_xy_rl", 0.08))
            az = float(cr_cfg.get("residual_acc_max_z_rl", 0.10))
            action_scale = [axy, axy, az]
        elif self.action_dim == 3:
            axy = float(phase_cfg.get("residual_acc_max_xy",
                        phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0))))
            az  = float(phase_cfg.get("residual_acc_max_z",
                        phase_cfg.get("acc_max_z",  ee_cfg.get("acc_max_z",  3.0))))
            action_scale = [axy, axy, az]
        elif _use_nmpc_base:
            # [v11 Path 1] cruise SAC 默认值与 PPO 对齐 (0.25→0.08)
            arl = float(cr_cfg.get("residual_acc_max_xy_rl", 0.08))
            action_scale = [arl, arl]
        else:
            axy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            action_scale = [axy, axy]

        self.actor  = SACPhaseActor(self.obs_dim, self.action_dim, action_scale,
                                     hidden, n_layers).to(self.device)
        self.critic = SACPhaseCritic(self.obs_dim, self.action_dim,
                                      hidden, n_layers).to(self.device)
        self.target_critic = SACPhaseCritic(self.obs_dim, self.action_dim,
                                             hidden, n_layers).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=float(cfg_sac["lr_actor"]))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=float(cfg_sac["lr_critic"]))
        alpha_init = float(cfg_sac["alpha_init"])
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32,
                                       device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=float(cfg_sac["lr_alpha"]))
        self.auto_alpha = bool(cfg_sac["auto_alpha"])
        self.target_entropy = -float(cfg_sac["target_entropy_ratio"]) * self.action_dim
        # [v11.4] log_alpha hard bounds — 防止 alpha 单调发散到 5+ (zero-init actor 下)
        self._log_alpha_min = float(cfg_sac.get("log_alpha_min", -9.2))
        self._log_alpha_max = float(cfg_sac.get("log_alpha_max",  0.0))
        self._last_target_q = 0.0  # [v11.4] target_q diagnostic

        buf_size = int(cfg_sac["buffer_size"])
        sbonus = float(config.get("descent_rl", {}).get("reward", {}).get(
                       "success_bonus", 50.0))
        if phase_name == "descent":
            self.buffer = HERReplayBuffer(
                buf_size, self.obs_dim, self.action_dim,
                her_k=int(cfg_sac.get("her_k", 4)),
                success_bonus=sbonus,
                her_reward_scale=float(cfg_sac.get("her_reward_scale", 0.3)))
            self._use_her = True
        else:
            self.buffer = SimpleReplayBuffer(buf_size, self.obs_dim, self.action_dim)
            self._use_her = False
        self.total_steps = 0; self._last_result = SAC_ZERO

        # ── [v14.2] CableEncoder for SAC (冻结随机投影) ─────────────────────
        ce_cfg = config.get("cable_encoder", {})
        self._cable_raw_dim = int(ce_cfg.get("raw_dim", 240))
        self._cable_out_dim = int(ce_cfg.get("output_dim", 32))
        self._wind_dim = 3
        self.cable_encoder = CableEncoder(
            raw_dim=self._cable_raw_dim,
            hidden_dim=int(ce_cfg.get("hidden_dim", 128)),
            output_dim=self._cable_out_dim,
        ).to(self.device)
        for p in self.cable_encoder.parameters():
            p.requires_grad = False
        self.cable_encoder.eval()

        # ── [v11 Path 1] Cruise SAC 残差: 输出层零初始化, 与 PPO 一致 ───────────
        # SAC 在 cruise 上的优势: off-policy + auto α 自动调整 entropy,
        # 天然规避 PPO 的 entropy collapse / 死局问题.
        if phase_name == "cruise" and _use_nmpc_base:
            self._zero_init_sac_residual_actor()

    def _zero_init_sac_residual_actor(self):
        """[v11 Path 1] SAC 残差 actor 输出层零初始化.

        让训练初期 RL 残差 ≈ 0, NMPC 单独主导; SAC 的 auto α 会自动控制探索强度.
        与 PPO 同样的设计 (Jeon et al. 2025 Residual MPC, Ankile et al. 2024 ResiP).
        """
        import math
        with torch.no_grad():
            self.actor.mean_head.weight.zero_()
            self.actor.mean_head.bias.zero_()
            # SAC log_std 是 state-dependent (log_std_head), 把它初始化到一个小负值
            self.actor.log_std_head.weight.zero_()
            self.actor.log_std_head.bias.fill_(-2.0)  # std ≈ 0.135
        print(f"  [SAC-cruise] v11 Path 1: actor 输出层零初始化, "
              f"log_std bias=-2.0 (std≈0.135)")

    @property
    def alpha(self): return self.log_alpha.exp().item()

    @torch.no_grad()
    def encode_obs(self, core_obs, cable_raw, wind_obs):
        """[v14.0] Same as PPOPhaseAgent.encode_obs."""
        cable_arr = np.asarray(cable_raw, dtype=np.float32).reshape(-1)
        if cable_arr.size == self._cable_out_dim:
            cable_feat = cable_arr
        else:
            cable_feat = self.encode_cable(cable_arr)
        return np.concatenate([core_obs, cable_feat, wind_obs]).astype(np.float32)

    def encode_cable(self, cable_raw):
        cable_t = torch.from_numpy(
            np.asarray(cable_raw, dtype=np.float32).reshape(1, -1)).to(self.device)
        with torch.no_grad():
            return self.cable_encoder(cable_t).cpu().numpy().flatten()

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm: update = False
        if self.use_obs_norm:
            if update: self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        if deterministic:
            action = self.actor.deterministic(s)
        else:
            action = self.actor.sample(s)[0]
        return action.cpu().numpy().flatten()

    def remember(self, norm_obs, action, norm_next_obs, reward, done,
                 achieved_pos=None):
        self.buffer.add(norm_obs, action, norm_next_obs, reward, done, achieved_pos)

    def reset_adam_state(self, reason=""):
        """[v11 vec fix] 强制重置 Adam 一阶 + 二阶矩, 用于课程倒退后清死局的二阶矩.

        与 PPOPhaseAgent.reset_adam_state 完全一致.
        来源: Ashley et al. 2021 (arXiv:2102.07686), Asadi et al. 2023.
        """
        for opt in [self.opt_actor, self.opt_critic, self.opt_alpha]:
            for group in opt.param_groups:
                for p in group['params']:
                    state = opt.state.get(p, {})
                    for key in ['exp_avg', 'exp_avg_sq']:
                        if key in state and torch.is_tensor(state[key]):
                            state[key].zero_()
                    if 'step' in state:
                        if torch.is_tensor(state['step']):
                            state['step'].zero_()
                        else:
                            state['step'] = 0
        print(f"  [SAC Adam reset] @ step {self.total_steps} ({reason})")

    def flush_episode_her(self, target_xy, target_z, obs_relabel_fn=None):
        if self._use_her:
            self.buffer.flush_episode_with_her(target_xy, target_z, obs_relabel_fn)

    def train_step(self):
        if not self.buffer.is_ready: return SAC_ZERO
        dev = self.device
        total_cl = total_al = total_al2 = total_q = 0.0
        total_tq = 0.0  # [v11.4] target_q diagnostic
        for _ in range(self.updates_per_step):
            s, a, ns, r, d = self.buffer.sample(self.batch_size)
            si = np_to_tensor(s, dev); ai = np_to_tensor(a, dev)
            nsi = np_to_tensor(ns, dev)
            ri = np_to_tensor(r, dev) * self.reward_scale
            di = np_to_tensor(d, dev)
            alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                next_a, next_lp = self.actor.sample(nsi)
                tq1, tq2 = self.target_critic(nsi, next_a)
                target_q = ri + self.gamma*(1-di)*(
                    torch.min(tq1,tq2) - alpha*next_lp.unsqueeze(-1))
            q1, q2 = self.critic(si, ai)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.opt_critic.zero_grad(); critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
            self.opt_critic.step()
            new_a, new_lp = self.actor.sample(si)
            q1n, q2n = self.critic(si, new_a); q_min = torch.min(q1n, q2n)
            actor_loss = (alpha*new_lp.unsqueeze(-1) - q_min).mean()
            self.opt_actor.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
            self.opt_actor.step()
            al2 = 0.0
            if self.auto_alpha:
                al_loss = -(self.log_alpha.exp()*(new_lp.detach()+self.target_entropy)).mean()
                self.opt_alpha.zero_grad(); al_loss.backward(); self.opt_alpha.step()
                # [v11.4 关键] log_alpha hard clamp 防止 alpha 发散
                # 原因: zero-init actor 下 log_pi 持续高于 target_entropy → Adam
                # 单调推 log_alpha 上升 → α 爆炸 → Q target 发散.
                with torch.no_grad():
                    self.log_alpha.clamp_(min=self._log_alpha_min,
                                          max=self._log_alpha_max)
                al2 = al_loss.item()
            soft_update(self.target_critic, self.critic, self.tau)
            total_cl += critic_loss.item(); total_al += actor_loss.item()
            total_al2 += al2; total_q += q_min.mean().item()
            total_tq += target_q.mean().item()
        n = self.updates_per_step
        result = SACResult(total_cl/n, total_al/n, total_al2/n, self.alpha, total_q/n)
        # [v11.4] 把 target_q 也存进去供 wandb 日志 (附在 _last_result 上)
        self._last_target_q = total_tq / n
        self._last_result = result; return result

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "total_steps": self.total_steps,
            "obs_norm": self.obs_norm.state_dict(),
            "phase_name": self.phase_name,
            "cable_encoder": self.cable_encoder.state_dict()}, path)  # [v14.0]

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if "actor" in ck:
            try:
                self.actor.load_state_dict(ck["actor"])
            except RuntimeError:
                self.actor.load_state_dict(ck["actor"], strict=False)
                print("  [SAC.load] actor loaded with strict=False")
        if "critic" in ck:
            try:
                self.critic.load_state_dict(ck["critic"])
            except RuntimeError:
                self.critic.load_state_dict(ck["critic"], strict=False)
            if "target_critic" in ck:
                try:
                    self.target_critic.load_state_dict(ck["target_critic"])
                except RuntimeError:
                    self.target_critic.load_state_dict(ck["target_critic"], strict=False)
            else:
                self.target_critic.load_state_dict(self.critic.state_dict())
        if "opt_actor"  in ck:
            try: self.opt_actor.load_state_dict(ck["opt_actor"])
            except Exception: pass
        if "opt_critic" in ck:
            try: self.opt_critic.load_state_dict(ck["opt_critic"])
            except Exception: pass
        if "log_alpha" in ck:
            self.log_alpha.data.copy_(ck["log_alpha"].to(self.device))
        if "opt_alpha" in ck:
            try: self.opt_alpha.load_state_dict(ck["opt_alpha"])
            except Exception: pass
        self.total_steps = ck.get("total_steps", 0)
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])
        if "cable_encoder" in ck:  # [v14.0]
            self.cable_encoder.load_state_dict(ck["cable_encoder"])
