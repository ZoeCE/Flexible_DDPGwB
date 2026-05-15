# ==============================================================================
# phase_agent.py — 三阶段统一 RL Agent (PPO + SAC) v4 (LSTM + Descent 双RL)
#
# v4 升级:
#   [LSTM]        MLP → LSTM-MLP 混合架构 (捕获摆动相位信息)
#   [DESCENT-2RL] Descent 段双 RL 架构:
#                 RL_macro (2D xy) + RL_residual (3D fine-tune)
#   [PLASTICITY]  Lyle et al. NeurIPS 2024
#   [ENT-ANNEAL]  PPO-CMA 退火 entropy 系数
#   [HER]         Andrychowicz 2017 — SAC descent HERReplayBuffer
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
# 工具
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
        if self.n == 1: self.mean = x.copy(); self.S = np.zeros_like(x)
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
    def load_state_dict(self, d): self.n = d["n"]; self.mean = d["mean"].copy(); self.S = d["S"].copy()


# ==============================================================================
# 观测历史缓冲 (LSTM 用)
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
# 观测索引
# ==============================================================================
OBS_EE_X, OBS_EE_Y = 0, 1
OBS_EE_VX, OBS_EE_VY = 2, 3
OBS_PL_X, OBS_PL_Y = 4, 5
OBS_PL_VX, OBS_PL_VY = 6, 7
OBS_EE_Z = 19; OBS_EE_VZ = 20
OBS_PL_Z = 21; OBS_PL_VZ = 22
OBS_TILT = 29; OBS_YAW = 30

# ==============================================================================
# 观测构建
# ==============================================================================

def build_lift_obs(env_obs, env, start_xy, prev_tilt=0.0, prev_yaw=0.0):
    obs = env_obs; dt = getattr(env, 'dt', 0.1)
    z_cruise = float(env.config["planning"]["payload_z_cruise"])
    ee_pos = np.array([obs[OBS_EE_X], obs[OBS_EE_Y], obs[OBS_EE_Z]])
    ee_vel = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]])
    pl_pos = np.array([obs[OBS_PL_X], obs[OBS_PL_Y], obs[OBS_PL_Z]])
    pl_vel = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY], obs[OBS_PL_VZ]])
    offset = ee_pos - pl_pos
    tilt = float(obs[OBS_TILT]); yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt; yaw_rate = (yaw - prev_yaw) / dt
    z_error = float(pl_pos[2] - z_cruise)
    return np.concatenate([ee_pos, ee_vel, pl_pos, pl_vel, offset,
                           [tilt, yaw, tilt_rate, yaw_rate],
                           start_xy[:2], [z_cruise], [z_error]]).astype(np.float32), tilt, yaw


def build_cruise_obs(env_obs, env, target_xy, prev_tilt=0.0, prev_yaw=0.0):
    obs = env_obs; dt = getattr(env, 'dt', 0.1); n_obs_max = env.n_obstacles
    ee_xy = np.array([obs[OBS_EE_X], obs[OBS_EE_Y]])
    ee_vxy = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY]])
    pl_xy = np.array([obs[OBS_PL_X], obs[OBS_PL_Y]])
    pl_vxy = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY]])
    offset_xy = ee_xy - pl_xy
    tilt = float(obs[OBS_TILT]); yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt; yaw_rate = (yaw - prev_yaw) / dt
    dist = float(np.linalg.norm(pl_xy - target_xy))
    obs_data = obs[10:10 + 3 * n_obs_max].copy(); joint_q = obs[31:38].copy()
    z_lock = float(env.config.get("cruise_rl", {}).get("z_lock_height", 0.25))
    pl_z = float(obs[OBS_PL_Z]); pl_vz = float(obs[OBS_PL_VZ]); ee_z = float(obs[OBS_EE_Z])
    z_error = pl_z - z_lock; z_dev_norm = np.clip(z_error / 0.10, -1.0, 1.0)
    return np.concatenate([
        ee_xy, ee_vxy, pl_xy, pl_vxy, offset_xy,
        [tilt, yaw, tilt_rate, yaw_rate], target_xy, [dist],
        obs_data, joint_q, [pl_z, pl_vz, ee_z, z_error, z_dev_norm],
    ]).astype(np.float32), tilt, yaw


def build_descent_obs(env_obs, env, target_xy, prev_tilt=0.0, prev_yaw=0.0):
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
    z_error = float(pl_pos[2] - target_pz); rebar_err = obs[-4:].copy()
    return np.concatenate([
        ee_pos, ee_vel, pl_pos, pl_vel, offset,
        [tilt, yaw, tilt_rate, yaw_rate], target_xy, [target_pz],
        pl_target_xy_err, [z_error], rebar_err,
    ]).astype(np.float32), tilt, yaw


# ==============================================================================
# LSTM-MLP Actor / Critic [v4]
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
        if obs_seq.dim() == 2: obs_seq = obs_seq.unsqueeze(1)  # [B,obs]->[B,1,obs]
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

    def bc_forward(self, obs_seq, action_target, hx=None):
        mean_raw, _, _ = self._dist(obs_seq, hx)
        u_tanh_t = (action_target / (self.action_scale + 1e-8)).clamp(-0.999, 0.999)
        bc_loss = F.mse_loss(mean_raw, torch.atanh(u_tanh_t))
        action_pred = torch.tanh(mean_raw) * self.action_scale
        return bc_loss, F.mse_loss(action_pred, action_target), action_pred

    def get_log_std_per_dim(self):
        return self.log_std.detach().cpu().numpy().copy()


class LSTMPhaseCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, lstm_dim=128, n_layers=2, seq_len=8):
        super().__init__()
        self.obs_embed = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        orthogonal_init(self.obs_embed[0])
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, num_layers=1, batch_first=True)
        layers = []; d = lstm_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            layers += [lin, nn.ReLU()]; d = hidden_dim
        out = nn.Linear(d, 1); orthogonal_init(out, gain=1.0); layers.append(out)
        self.head = nn.Sequential(*layers)

    def forward(self, obs_seq, hx=None):
        if obs_seq.dim() == 2: obs_seq = obs_seq.unsqueeze(1)  # [B,obs]->[B,1,obs]
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
    def bc_forward(self, s, action_target):
        mean_raw, _ = self._dist(s)
        u_tanh_t = (action_target / (self.action_scale + 1e-8)).clamp(-0.999, 0.999)
        bc_loss = F.mse_loss(mean_raw, torch.atanh(u_tanh_t))
        action_pred = torch.tanh(mean_raw) * self.action_scale
        return bc_loss, F.mse_loss(action_pred, action_target), action_pred
    def get_log_std_per_dim(self):
        return self.log_std.detach().cpu().numpy().copy()


class PhaseCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = []; d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            layers += [lin, nn.ReLU()]; d = hidden_dim
        out = nn.Linear(d, 1); orthogonal_init(out, gain=1.0); layers.append(out)
        self.net = nn.Sequential(*layers)
    def forward(self, s): return self.net(s)


# ==============================================================================
# Rollout Buffers (Sequence + MLP)
# ==============================================================================

class SequenceRolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, seq_len, device):
        self.n_steps = n_steps; self.obs_dim = obs_dim
        self.action_dim = action_dim; self.seq_len = seq_len; self.device = device; self.clear()
    def clear(self):
        self.obs_seqs = np.zeros((self.n_steps, self.seq_len, self.obs_dim), np.float32)
        self.actions = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.bc_targets = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.rewards = np.zeros(self.n_steps, np.float32)
        self.dones = np.zeros(self.n_steps, np.float32)
        self.values = np.zeros(self.n_steps, np.float32)
        self.log_probs = np.zeros(self.n_steps, np.float32)
        self.advantages = np.zeros(self.n_steps, np.float32)
        self.returns = np.zeros(self.n_steps, np.float32)
        self.ptr = 0; self.full = False
    def add(self, obs_seq, action, bc_target, reward, done, value, log_prob):
        i = self.ptr; self.obs_seqs[i] = obs_seq
        self.actions[i] = action; self.bc_targets[i] = bc_target
        self.rewards[i] = reward; self.dones[i] = float(done)
        self.values[i] = value; self.log_probs[i] = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps: self.full = True
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
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
                   np_to_tensor(self.bc_targets[idx], self.device),
                   np_to_tensor(self.returns[idx], self.device).view(-1, 1),
                   np_to_tensor(adv[idx], self.device),
                   np_to_tensor(self.log_probs[idx], self.device))


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, device):
        self.n_steps = n_steps; self.obs_dim = obs_dim
        self.action_dim = action_dim; self.device = device; self.clear()
    def clear(self):
        self.obs = np.zeros((self.n_steps, self.obs_dim), np.float32)
        self.actions = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.bc_targets = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.rewards = np.zeros(self.n_steps, np.float32)
        self.dones = np.zeros(self.n_steps, np.float32)
        self.values = np.zeros(self.n_steps, np.float32)
        self.log_probs = np.zeros(self.n_steps, np.float32)
        self.advantages = np.zeros(self.n_steps, np.float32)
        self.returns = np.zeros(self.n_steps, np.float32)
        self.ptr = 0; self.full = False
    def add(self, obs, action, bc_target, reward, done, value, log_prob):
        i = self.ptr; self.obs[i] = obs; self.actions[i] = action
        self.bc_targets[i] = bc_target; self.rewards[i] = reward
        self.dones[i] = float(done); self.values[i] = value; self.log_probs[i] = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps: self.full = True
    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
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
                   np_to_tensor(self.bc_targets[idx], self.device),
                   np_to_tensor(self.returns[idx], self.device).view(-1, 1),
                   np_to_tensor(adv[idx], self.device),
                   np_to_tensor(self.log_probs[idx], self.device))


# ==============================================================================
# PPO Agent v4
# ==============================================================================

PPOResult = namedtuple("PPOResult", [
    "policy_loss", "value_loss", "entropy_loss", "bc_loss",
    "approx_kl", "clip_fraction", "total_loss", "entropy_coef_used"])
PPO_ZERO = PPOResult(0., 0., 0., 0., 0., 0., 0., 0.)


class PPOPhaseAgent:

    def __init__(self, phase_name, config=None):
        if config is None: config = DEFAULT_CONFIG
        self.config = config; self.phase_name = phase_name
        phase_cfg = config[f"{phase_name}_rl"]; cfg_ppo = config["ppo"]
        self.obs_dim = int(phase_cfg["obs_dim"])
        self.action_dim = int(phase_cfg["action_dim"])
        self.gamma = float(cfg_ppo["gamma"]); self.gae_lambda = float(cfg_ppo["gae_lambda"])
        self.clip_eps = float(cfg_ppo["clip_eps"]); self.value_loss_coef = float(cfg_ppo["value_loss_coef"])
        self.max_grad_norm = float(cfg_ppo["max_grad_norm"])
        self.n_steps = int(cfg_ppo["n_steps"]); self.n_epochs = int(cfg_ppo["n_epochs"])
        self.batch_size = int(cfg_ppo["batch_size"]); self.norm_adv = bool(cfg_ppo["normalize_advantages"])
        self.target_kl = float(cfg_ppo.get("target_kl", 0.03))

        self.use_lstm = bool(cfg_ppo.get("use_lstm", True))
        self.seq_len = int(cfg_ppo.get("seq_len", 8))
        lstm_dim = int(cfg_ppo.get("lstm_dim", 128))

        self.entropy_coef_start = float(cfg_ppo.get("entropy_coef_start", cfg_ppo.get("entropy_coef", 0.15)))
        self.entropy_coef_end = float(cfg_ppo.get("entropy_coef_end", 0.02))
        self.entropy_coef_anneal_steps = int(cfg_ppo.get("entropy_coef_anneal_steps", 1_000_000))
        self.entropy_coef = self.entropy_coef_start
        self.use_obs_norm = bool(cfg_ppo["use_obs_norm"])
        self.obs_norm = RunningMeanStd(shape=(self.obs_dim,),
            warm_start=int(cfg_ppo.get("obs_norm_warm_start", 5000)), clip=float(cfg_ppo["obs_norm_clip"]))
        self._freeze_obs_norm = False
        self._log_std_floor_init = float(cfg_ppo.get("log_std_floor_init", 0.0))
        self._log_std_floor_final = float(cfg_ppo.get("log_std_floor_final", -1.5))
        self._log_std_floor_steps = int(cfg_ppo.get("log_std_floor_steps", 800_000))
        self._plasticity_reset_interval = int(cfg_ppo.get("plasticity_reset_interval", 200_000))
        self._last_plasticity_reset = 0

        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))

        ee_cfg = config.get("ee_control", {})
        cr_cfg = config.get("cruise_rl", {})
        _use_nmpc_base = (phase_name == "cruise" and bool(cr_cfg.get("use_nmpc_base", False)))

        if self.action_dim == 3:
            axy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 0.5)))
            az  = float(phase_cfg.get("acc_max_z",  ee_cfg.get("acc_max_z",  1.0)))
            action_scale = [axy, axy, az]
        elif _use_nmpc_base:
            # [v7 KEY] action_scale = residual_acc_max_xy_rl (必须与 clip 范围一致)
            arl = float(cr_cfg.get("residual_acc_max_xy_rl", 0.25))
            action_scale = [arl, arl]
        else:
            axy = float(phase_cfg.get("residual_acc_max_xy",
                        ee_cfg.get("acc_max_xy", 0.60)))
            action_scale = [axy, axy]

        if self.use_lstm:
            self.actor = LSTMPhaseActor(self.obs_dim, self.action_dim, action_scale,
                hidden_dim=int(cfg_ppo["hidden_dim"]), lstm_dim=lstm_dim, n_layers=2, seq_len=self.seq_len,
                log_std_init=float(cfg_ppo["log_std_init"]), log_std_min=float(cfg_ppo["log_std_min"]),
                log_std_max=float(cfg_ppo["log_std_max"])).to(self.device)
            self.critic = LSTMPhaseCritic(self.obs_dim, hidden_dim=int(cfg_ppo["hidden_dim"]),
                lstm_dim=lstm_dim, n_layers=2, seq_len=self.seq_len).to(self.device)
            self.buffer = SequenceRolloutBuffer(self.n_steps, self.obs_dim, self.action_dim, self.seq_len, self.device)
        else:
            self.actor = PhaseActor(self.obs_dim, self.action_dim, action_scale,
                hidden_dim=int(cfg_ppo["hidden_dim"]), n_layers=int(cfg_ppo["n_layers"]),
                log_std_init=float(cfg_ppo["log_std_init"]), log_std_min=float(cfg_ppo["log_std_min"]),
                log_std_max=float(cfg_ppo["log_std_max"])).to(self.device)
            self.critic = PhaseCritic(self.obs_dim, hidden_dim=int(cfg_ppo["hidden_dim"]),
                n_layers=int(cfg_ppo["n_layers"])).to(self.device)
            self.buffer = RolloutBuffer(self.n_steps, self.obs_dim, self.action_dim, self.device)

        self.obs_history = ObsHistoryBuffer(self.obs_dim, self.seq_len)
        self._lr_actor = float(cfg_ppo["lr_actor"]); self._lr_critic = float(cfg_ppo["lr_critic"])
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self._lr_actor, eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self._lr_critic, eps=1e-5)
        self.total_steps = 0; self._last_result = PPO_ZERO; self.bc_coef = 0.0

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm: update = False
        if self.use_obs_norm:
            if update: self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    def reset_history(self): self.obs_history.reset()

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
    def get_value_for_state(self, norm_obs):
        if self.use_lstm:
            obs_seq = self.obs_history.get_sequence()
            s = np_to_tensor(obs_seq, self.device).unsqueeze(0)
            return self.critic(s).cpu().item()
        else:
            s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
            return self.critic(s).cpu().item()

    def get_log_std_per_dim(self): return self.actor.get_log_std_per_dim()

    def add_to_buffer(self, norm_obs, action, bc_target, reward, done, value, log_prob):
        if self.use_lstm:
            obs_seq = self.obs_history.get_sequence()
            self.buffer.add(obs_seq, action, bc_target, reward, done, value, log_prob)
        else:
            self.buffer.add(norm_obs, action, bc_target, reward, done, value, log_prob)

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

    def _update_entropy_coef(self, global_ts=None):
        ts = global_ts if global_ts is not None else self.total_steps
        frac = min(ts / max(self.entropy_coef_anneal_steps, 1), 1.0)
        self.entropy_coef = self.entropy_coef_start + frac * (self.entropy_coef_end - self.entropy_coef_start)

    def update(self, global_ts=None):
        if not self.buffer.full: return PPO_ZERO
        self._maybe_reset_plasticity(); self._update_entropy_coef(global_ts=global_ts)
        total_pl = total_vl = total_el = total_bl = total_kl = total_cf = 0.0
        n_updates = 0; stop_early = False
        for epoch in range(self.n_epochs):
            if stop_early: break
            for batch in self.buffer.get_minibatches(self.batch_size, self.norm_adv):
                obs_b, act_b, bc_b, ret_b, adv_b, old_lp_b = batch
                value = self.critic(obs_b); value_loss = F.huber_loss(value, ret_b)
                self.opt_critic.zero_grad()
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm); self.opt_critic.step()
                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b; surr2 = ratio.clamp(1-self.clip_eps, 1+self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean(); entropy_loss = -entropy.mean()
                bc_loss = torch.zeros(1, device=self.device)
                if self.bc_coef > 0: bc_loss, _, _ = self.actor.bc_forward(obs_b, bc_b)
                actor_total = policy_loss + self.entropy_coef * entropy_loss + self.bc_coef * bc_loss
                self.opt_actor.zero_grad(); actor_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.opt_actor.step()
                with torch.no_grad():
                    frac = min(self.total_steps / max(self._log_std_floor_steps, 1), 1.0)
                    floor = self._log_std_floor_init + frac * (self._log_std_floor_final - self._log_std_floor_init)
                    self.actor.log_std.data.clamp_(min=floor)
                    approx_kl = (old_lp_b - new_lp).mean().abs().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()
                total_pl += policy_loss.item(); total_vl += value_loss.item()
                total_el += entropy_loss.item(); total_bl += bc_loss.item()
                total_kl += approx_kl; total_cf += clip_frac; n_updates += 1
                if approx_kl > self.target_kl: stop_early = True; break
        n = max(n_updates, 1)
        result = PPOResult(total_pl/n, total_vl/n, total_el/n, total_bl/n,
                           total_kl/n, total_cf/n, (total_pl+total_vl+total_el+total_bl)/n, self.entropy_coef)
        self._last_result = result; self.buffer.clear(); return result

    def reset_log_std_for_rl(self):
        """BC→RL: 重置 log_std 到 log_std_max，恢复探索能力。"""
        import math
        with torch.no_grad():
            reset_val = max(float(self.actor.log_std_max), -0.3)
            reset_val = float(np.clip(reset_val, self.actor.log_std_min, self.actor.log_std_max))
            self.actor.log_std.data.fill_(reset_val)
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=self._lr_actor,  eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self._lr_critic, eps=1e-5)
        self._last_plasticity_reset = self.total_steps
        self.entropy_coef = self.entropy_coef_start
        print(f"  [PPO] BC→RL: log_std → {reset_val:.3f} (std={math.exp(reset_val):.3f}), "
              f"entropy_coef → {self.entropy_coef:.4f}")

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "total_steps": self.total_steps, "bc_coef": self.bc_coef,
            "obs_norm": self.obs_norm.state_dict(), "phase_name": self.phase_name,
            "entropy_coef": self.entropy_coef, "use_lstm": self.use_lstm}, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"])
        if "opt_actor" in ck: self.opt_actor.load_state_dict(ck["opt_actor"])
        if "opt_critic" in ck: self.opt_critic.load_state_dict(ck["opt_critic"])
        self.total_steps = ck.get("total_steps", 0); self.bc_coef = ck.get("bc_coef", 0.0)
        if "obs_norm" in ck: self.obs_norm.load_state_dict(ck["obs_norm"])


# ==============================================================================
# SAC Actor / Critic / Buffer / Agent (保持 MLP)
# ==============================================================================

class SACPhaseActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale, hidden_dim=256, n_layers=3, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min; self.log_std_max = log_std_max
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))
        layers = []; d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin); layers += [lin, nn.ReLU()]; d = hidden_dim
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
    def bc_forward(self, s, action_target):
        mean, _ = self.forward(s)
        u_tanh_t = (action_target / (self.action_scale + 1e-8)).clamp(-0.999, 0.999)
        bc_loss = F.mse_loss(mean, torch.atanh(u_tanh_t))
        action_pred = torch.tanh(mean) * self.action_scale
        return bc_loss, F.mse_loss(action_pred, action_target), action_pred

class SACPhaseCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        in_dim = obs_dim + action_dim
        def _make():
            layers = []; d = in_dim
            for _ in range(n_layers):
                lin = nn.Linear(d, hidden_dim); orthogonal_init(lin); layers += [lin, nn.ReLU()]; d = hidden_dim
            out = nn.Linear(d, 1); orthogonal_init(out, gain=1.0); layers.append(out)
            return nn.Sequential(*layers)
        self.q1 = _make(); self.q2 = _make()
    def forward(self, s, a):
        sa = torch.cat([s, a], -1); return self.q1(sa), self.q2(sa)

class SimpleReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim, **kw):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.obs = np.zeros((max_size, obs_dim), np.float32)
        self.actions = np.zeros((max_size, action_dim), np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), np.float32)
        self.rewards = np.zeros((max_size, 1), np.float32)
        self.dones = np.zeros((max_size, 1), np.float32)
    def add(self, obs, action, next_obs, reward, done, achieved_pos=None):
        i = self.ptr; self.obs[i]=obs; self.actions[i]=action; self.next_obs[i]=next_obs
        self.rewards[i]=reward; self.dones[i]=float(done)
        self.ptr = (self.ptr+1) % self.max_size; self.size = min(self.size+1, self.max_size)
    def flush_episode_with_her(self, *a, **kw): pass
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.obs[idx], self.actions[idx], self.next_obs[idx], self.rewards[idx], self.dones[idx]
    @property
    def is_ready(self): return self.size >= 1000

class HERReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim, her_k=4, success_bonus=50.0, her_reward_scale=0.3, **kw):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.her_k = her_k; self.her_reward = success_bonus * her_reward_scale
        self.obs = np.zeros((max_size, obs_dim), np.float32)
        self.actions = np.zeros((max_size, action_dim), np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), np.float32)
        self.rewards = np.zeros((max_size, 1), np.float32)
        self.dones = np.zeros((max_size, 1), np.float32)
        self._ep = []
    def _store(self, obs, action, next_obs, reward, done):
        i = self.ptr; self.obs[i]=obs; self.actions[i]=action; self.next_obs[i]=next_obs
        self.rewards[i]=reward; self.dones[i]=float(done)
        self.ptr=(self.ptr+1)%self.max_size; self.size=min(self.size+1,self.max_size)
    def add(self, obs, action, next_obs, reward, done, achieved_pos=None):
        self._store(obs, action, next_obs, reward, done)
        if achieved_pos is not None:
            self._ep.append((obs.copy(), action.copy(), next_obs.copy(), float(reward), bool(done), np.array(achieved_pos, np.float32)))
    def flush_episode_with_her(self, target_xy, target_z, obs_relabel_fn=None):
        ep = self._ep; T = len(ep)
        if T < 2: self._ep = []; return
        for t in range(T):
            n_future = min(self.her_k, T-t-1)
            if n_future <= 0: continue
            future_ts = np.random.randint(t+1, T, size=n_future)
            for ft in future_ts:
                achieved = ep[ft][5]
                if obs_relabel_fn is not None:
                    try: her_obs = obs_relabel_fn(ep[t][0], achieved[:2], float(achieved[2])); her_nobs = obs_relabel_fn(ep[t][2], achieved[:2], float(achieved[2]))
                    except: her_obs = ep[t][0]; her_nobs = ep[t][2]
                else: her_obs = ep[t][0]; her_nobs = ep[t][2]
                self._store(her_obs, ep[t][1], her_nobs, self.her_reward, False)
        self._ep = []
    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.obs[idx], self.actions[idx], self.next_obs[idx], self.rewards[idx], self.dones[idx]
    @property
    def is_ready(self): return self.size >= 1000


SACResult = namedtuple("SACResult", ["critic_loss", "actor_loss", "alpha_loss", "alpha", "q_mean"])
SAC_ZERO = SACResult(0., 0., 0., 0., 0.)

class SACPhaseAgent:
    def __init__(self, phase_name, config=None):
        if config is None: config = DEFAULT_CONFIG
        self.config = config; self.phase_name = phase_name
        phase_cfg = config[f"{phase_name}_rl"]; cfg_sac = config["sac"]
        self.obs_dim = int(phase_cfg["obs_dim"]); self.action_dim = int(phase_cfg["action_dim"])
        self.gamma = float(cfg_sac["gamma"]); self.tau = float(cfg_sac["tau"])
        self.batch_size = int(cfg_sac["batch_size"]); self.warmup_steps = int(cfg_sac["warmup_steps"])
        self.reward_scale = float(cfg_sac.get("reward_scale", 1.0))
        self.update_interval = int(cfg_sac.get("update_interval", 1))
        self.updates_per_step = int(cfg_sac.get("updates_per_step", 1))
        self.critic_grad_clip = float(cfg_sac["critic_grad_clip"])
        self.actor_grad_clip = float(cfg_sac["actor_grad_clip"])
        self.use_obs_norm = bool(cfg_sac["use_obs_norm"])
        self.obs_norm = RunningMeanStd(shape=(self.obs_dim,),
            warm_start=int(cfg_sac.get("obs_norm_warm_start", 5000)), clip=float(cfg_sac["obs_norm_clip"]))
        self._freeze_obs_norm = False
        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() and gpu_id >= 0 else torch.device("cpu"))
        hidden = int(cfg_sac["hidden_dim"]); n_layers = int(cfg_sac["n_layers"])
        ee_cfg = config.get("ee_control", {})
        if self.action_dim == 3:
            axy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            az = float(phase_cfg.get("acc_max_z", ee_cfg.get("acc_max_z", 3.0)))
            action_scale = [axy, axy, az]
        else:
            axy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            action_scale = [axy, axy]
        self.actor = SACPhaseActor(self.obs_dim, self.action_dim, action_scale, hidden, n_layers).to(self.device)
        self.critic = SACPhaseCritic(self.obs_dim, self.action_dim, hidden, n_layers).to(self.device)
        self.target_critic = SACPhaseCritic(self.obs_dim, self.action_dim, hidden, n_layers).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict()); self.target_critic.eval()
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=float(cfg_sac["lr_actor"]))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=float(cfg_sac["lr_critic"]))
        alpha_init = float(cfg_sac["alpha_init"])
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32, device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=float(cfg_sac["lr_alpha"]))
        self.auto_alpha = bool(cfg_sac["auto_alpha"])
        self.target_entropy = -float(cfg_sac["target_entropy_ratio"]) * self.action_dim
        buf_size = int(cfg_sac["buffer_size"])
        sbonus = float(config.get("descent_rl", {}).get("reward", {}).get("success_bonus", 50.0))
        if phase_name == "descent":
            self.buffer = HERReplayBuffer(buf_size, self.obs_dim, self.action_dim,
                her_k=int(cfg_sac.get("her_k", 4)), success_bonus=sbonus,
                her_reward_scale=float(cfg_sac.get("her_reward_scale", 0.3)))
            self._use_her = True
        else:
            self.buffer = SimpleReplayBuffer(buf_size, self.obs_dim, self.action_dim)
            self._use_her = False
        self.total_steps = 0; self._last_result = SAC_ZERO

    @property
    def alpha(self): return self.log_alpha.exp().item()
    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm: update = False
        if self.use_obs_norm:
            if update: self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)
    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        action = self.actor.deterministic(s) if deterministic else self.actor.sample(s)[0]
        return action.cpu().numpy().flatten()
    def remember(self, norm_obs, action, norm_next_obs, reward, done, achieved_pos=None):
        self.buffer.add(norm_obs, action, norm_next_obs, reward, done, achieved_pos)
    def flush_episode_her(self, target_xy, target_z, obs_relabel_fn=None):
        if self._use_her: self.buffer.flush_episode_with_her(target_xy, target_z, obs_relabel_fn)
    def train_step(self):
        if not self.buffer.is_ready: return SAC_ZERO
        dev = self.device; total_cl = total_al = total_al2 = total_q = 0.0
        for _ in range(self.updates_per_step):
            s, a, ns, r, d = self.buffer.sample(self.batch_size)
            si = np_to_tensor(s, dev); ai = np_to_tensor(a, dev)
            nsi = np_to_tensor(ns, dev); ri = np_to_tensor(r, dev) * self.reward_scale
            di = np_to_tensor(d, dev); alpha = self.log_alpha.exp().detach()
            with torch.no_grad():
                next_a, next_lp = self.actor.sample(nsi)
                tq1, tq2 = self.target_critic(nsi, next_a)
                target_q = ri + self.gamma*(1-di)*(torch.min(tq1,tq2) - alpha*next_lp.unsqueeze(-1))
            q1, q2 = self.critic(si, ai)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.opt_critic.zero_grad(); critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip); self.opt_critic.step()
            new_a, new_lp = self.actor.sample(si)
            q1n, q2n = self.critic(si, new_a); q_min = torch.min(q1n, q2n)
            actor_loss = (alpha*new_lp.unsqueeze(-1) - q_min).mean()
            self.opt_actor.zero_grad(); actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip); self.opt_actor.step()
            al2 = 0.0
            if self.auto_alpha:
                al_loss = -(self.log_alpha.exp()*(new_lp.detach()+self.target_entropy)).mean()
                self.opt_alpha.zero_grad(); al_loss.backward(); self.opt_alpha.step(); al2 = al_loss.item()
            soft_update(self.target_critic, self.critic, self.tau)
            total_cl+=critic_loss.item(); total_al+=actor_loss.item(); total_al2+=al2; total_q+=q_min.mean().item()
        n = self.updates_per_step
        result = SACResult(total_cl/n, total_al/n, total_al2/n, self.alpha, total_q/n)
        self._last_result = result; return result
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(), "opt_alpha": self.opt_alpha.state_dict(),
            "total_steps": self.total_steps, "obs_norm": self.obs_norm.state_dict(), "phase_name": self.phase_name}, path)
    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        # Handle multiple checkpoint formats:
        # 1. Plain SAC ckpt:            actor / critic / target_critic
        # 2. CruiseDualRLAgent ckpt:    planner_actor / planner_critic  (use planner weights)
        # 3. PPO BC ckpt:               actor / critic  (no target_critic)
        if "planner_actor" in ck:
            # CruiseDualRLAgent format → extract planner weights into this SAC agent
            print("  [SAC.load] Detected CruiseDualRLAgent ckpt – loading planner weights")
            try:
                self.actor.load_state_dict(ck["planner_actor"], strict=False)
            except Exception as e:
                print(f"  [SAC.load] actor load warning: {e}")
        elif "actor" in ck:
            try:
                self.actor.load_state_dict(ck["actor"])
            except RuntimeError:
                self.actor.load_state_dict(ck["actor"], strict=False)
                print("  [SAC.load] actor loaded with strict=False")
        if "planner_critic" in ck:
            try:
                self.critic.load_state_dict(ck["planner_critic"], strict=False)
                self.target_critic.load_state_dict(ck["planner_critic"], strict=False)
            except Exception:
                pass
        elif "critic" in ck:
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
        if "opt_actor" in ck:
            try: self.opt_actor.load_state_dict(ck["opt_actor"])
            except Exception: pass
        if "opt_critic" in ck:
            try: self.opt_critic.load_state_dict(ck["opt_critic"])
            except Exception: pass
        if "log_alpha" in ck: self.log_alpha.data.copy_(ck["log_alpha"].to(self.device))
        if "opt_alpha" in ck:
            try: self.opt_alpha.load_state_dict(ck["opt_alpha"])
            except Exception: pass
        self.total_steps = ck.get("total_steps", 0)
        if "obs_norm" in ck: self.obs_norm.load_state_dict(ck["obs_norm"])


# ==============================================================================
# Descent 双 RL Agent (macro + residual)
# ==============================================================================

class DescentDualRLAgent:
    """
    Descent 段双 RL 架构:
      RL_macro:    输出 2D xy acc (宏观防摆+xy对准, LSTM-PPO)
      RL_residual: 输出 3D fine-tune acc (精细修正, LSTM-PPO)
      total_acc_xy = RL_macro + clip(RL_res[:2], ±res_max)
      total_acc_z  = RL_res[2]
    """

    def __init__(self, config=None):
        if config is None: config = DEFAULT_CONFIG
        self.config = config
        phase_cfg = config["descent_rl"]
        self.obs_dim = int(phase_cfg["obs_dim"])

        # RL_macro: 2D
        macro_config = copy.deepcopy(config)
        macro_config["descent_rl"] = copy.deepcopy(phase_cfg)
        macro_config["descent_rl"]["action_dim"] = 2
        self.macro_agent = PPOPhaseAgent("descent", config=macro_config)

        # RL_residual: 3D with smaller acc limits
        res_config = copy.deepcopy(config)
        res_config["descent_rl"] = copy.deepcopy(phase_cfg)
        res_config["descent_rl"]["action_dim"] = 3
        res_config["descent_rl"]["acc_max_xy"] = float(phase_cfg.get("residual_acc_max_xy", 0.15))
        res_config["descent_rl"]["acc_max_z"] = float(phase_cfg.get("residual_acc_max_z", 0.5))
        self.residual_agent = PPOPhaseAgent("descent", config=res_config)

        self.residual_acc_max_xy = float(phase_cfg.get("residual_acc_max_xy", 0.15))
        self.obs_norm = self.macro_agent.obs_norm
        self.use_obs_norm = self.macro_agent.use_obs_norm
        self._freeze_obs_norm = False
        self.obs_history = ObsHistoryBuffer(self.obs_dim, self.macro_agent.seq_len)
        self.total_steps = 0

        # ── 代理属性：让 train_phase.py 无需感知 dual-RL 架构 ────────────────
        self.device      = self.macro_agent.device
        self.action_dim  = 3                           # combined 动作维度
        self.gamma       = self.macro_agent.gamma
        self.gae_lambda  = self.macro_agent.gae_lambda
        self.use_lstm    = self.macro_agent.use_lstm
        self.seq_len     = self.macro_agent.seq_len
        self.bc_coef     = 0.0
        self.actor       = self.macro_agent.actor      # warmup/logstd 监控用
        self.critic      = self.macro_agent.critic
        self.opt_actor   = self.macro_agent.opt_actor
        self.opt_critic  = self.macro_agent.opt_critic
        self.buffer      = self.macro_agent.buffer     # .full 检测触发 update
        self._last_result = self.macro_agent._last_result

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm: update = False
        if self.use_obs_norm:
            if update: self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    def reset_history(self):
        self.obs_history.reset()
        self.macro_agent.obs_history = self.obs_history
        self.residual_agent.obs_history = self.obs_history

    @torch.no_grad()
    def _act_agent(self, agent, norm_obs, deterministic):
        if agent.use_lstm:
            obs_seq = self.obs_history.get_sequence()
            s = np_to_tensor(obs_seq, agent.device).unsqueeze(0)
            action, lp, _, _ = agent.actor.get_action(s, deterministic=deterministic)
            value = agent.critic(s)
            return action.detach().cpu().numpy().flatten(), lp.cpu().item(), value.cpu().item()
        else:
            s = np_to_tensor(norm_obs.reshape(1, -1), agent.device)
            action, lp, _ = agent.actor.get_action(s, deterministic=deterministic)
            value = agent.critic(s)
            return action.detach().cpu().numpy().flatten(), lp.cpu().item(), value.cpu().item()

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        self.obs_history.push(norm_obs)
        self.macro_agent.obs_history = self.obs_history
        self.residual_agent.obs_history = self.obs_history
        macro_act, macro_lp, macro_val = self._act_agent(self.macro_agent, norm_obs, deterministic)
        res_act, res_lp, res_val = self._act_agent(self.residual_agent, norm_obs, deterministic)
        res_xy = np.clip(res_act[:2], -self.residual_acc_max_xy, self.residual_acc_max_xy)
        combined = np.array([macro_act[0]+res_xy[0], macro_act[1]+res_xy[1], res_act[2]], dtype=np.float32)
        return combined, macro_act, res_act, macro_lp, macro_val, res_lp, res_val

    @torch.no_grad()
    def act_simple(self, norm_obs, deterministic=False):
        """统一接口：返回 (combined_act, lp, val)，与 PPOPhaseAgent.act() 签名相同。
        用于 warmup rollout、eval loop 等只需要动作本身的场合。"""
        combined, _, _, macro_lp, macro_val, _, _ = self.act(norm_obs, deterministic=deterministic)
        return combined, macro_lp, macro_val

    def update(self, global_ts=None):
        r1 = self.macro_agent.update(global_ts=global_ts)
        r2 = self.residual_agent.update(global_ts=global_ts)
        self._last_result = r1  # 刷新代理属性，外部读日志无需区分 dual-RL
        return r1, r2

    def add_to_buffers(self, norm_obs, macro_act, res_act, bc_macro, bc_res,
                       reward, done, macro_val, macro_lp, res_val, res_lp):
        obs_seq = self.obs_history.get_sequence()
        if self.macro_agent.use_lstm:
            self.macro_agent.buffer.add(obs_seq, macro_act, bc_macro, reward, done, macro_val, macro_lp)
            self.residual_agent.buffer.add(obs_seq, res_act, bc_res, reward, done, res_val, res_lp)
        else:
            self.macro_agent.buffer.add(norm_obs, macro_act, bc_macro, reward, done, macro_val, macro_lp)
            self.residual_agent.buffer.add(norm_obs, res_act, bc_res, reward, done, res_val, res_lp)

    def get_log_std_per_dim(self):
        return np.concatenate([self.macro_agent.actor.get_log_std_per_dim(),
                               self.residual_agent.actor.get_log_std_per_dim()])

    def save(self, path):
        torch.save({
            "macro_actor": self.macro_agent.actor.state_dict(),
            "macro_critic": self.macro_agent.critic.state_dict(),
            "res_actor": self.residual_agent.actor.state_dict(),
            "res_critic": self.residual_agent.critic.state_dict(),
            "obs_norm": self.obs_norm.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path, map_location=None):
        dev = self.macro_agent.device
        ck = torch.load(path, map_location=map_location or dev, weights_only=False)

        if "macro_actor" in ck:
            # ── dual-RL 格式的 checkpoint ──
            self.macro_agent.actor.load_state_dict(ck["macro_actor"])
            self.macro_agent.critic.load_state_dict(ck["macro_critic"])
            self.residual_agent.actor.load_state_dict(ck["res_actor"])
            self.residual_agent.critic.load_state_dict(ck["res_critic"])
        elif "actor" in ck:
            # ── 标准 PPOPhaseAgent (BC) checkpoint ──
            # BC agent 是 action_dim=3 的单一网络, 需要兼容加载到
            # macro(action_dim=2) 和 residual(action_dim=3)
            bc_actor_sd = ck["actor"]
            bc_critic_sd = ck["critic"]

            # Residual agent: action_dim=3, 与 BC 相同, 直接加载
            try:
                self.residual_agent.actor.load_state_dict(bc_actor_sd)
                print("  [DualRL] residual actor ← BC weights (full)")
            except Exception as e:
                print(f"  [DualRL] residual actor ← BC weights (partial): {e}")
                self.residual_agent.actor.load_state_dict(bc_actor_sd, strict=False)

            # Macro agent: action_dim=2, 只能加载 backbone/embed/lstm 权重
            # mean_head 维度不匹配 (3→2), 跳过
            macro_sd = self.macro_agent.actor.state_dict()
            loaded_keys = []
            for k, v in bc_actor_sd.items():
                if k in macro_sd and macro_sd[k].shape == v.shape:
                    macro_sd[k] = v
                    loaded_keys.append(k)
            self.macro_agent.actor.load_state_dict(macro_sd)
            skipped = set(bc_actor_sd.keys()) - set(loaded_keys)
            print(f"  [DualRL] macro actor ← BC weights: loaded {len(loaded_keys)} params, "
                  f"skipped {len(skipped)} (dim mismatch: {skipped})")

            # Critic: 两个 critic 结构相同 (obs_dim→1), 都可以加载 BC critic
            try:
                self.macro_agent.critic.load_state_dict(bc_critic_sd)
                self.residual_agent.critic.load_state_dict(bc_critic_sd)
                print("  [DualRL] both critics ← BC weights")
            except Exception as e:
                print(f"  [DualRL] critic load partial: {e}")
                self.macro_agent.critic.load_state_dict(bc_critic_sd, strict=False)
                self.residual_agent.critic.load_state_dict(bc_critic_sd, strict=False)
        else:
            print(f"  [DualRL] ⚠️ Unknown checkpoint format, keys: {list(ck.keys())}")

        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])
            self.residual_agent.obs_norm = self.obs_norm
        self.total_steps = ck.get("total_steps", 0)

    def reset_log_std_for_rl(self):
        self.macro_agent.reset_log_std_for_rl()
        self.residual_agent.reset_log_std_for_rl()

# ==============================================================================
# Cruise 双 RL Agent (planner + swing_rl)
# ==============================================================================

class CruiseDualRLAgent:
    """
    Cruise 段双 RL 架构:
      planner_agent:  高层 xy 加速度 (路径规划 + 避障), 使用导航 reward
      swing_agent:    低层残差防摆 (摆动能量最小化), 使用防摆 reward

    组合方式:
      total_acc_xy = planner_acc + clip(swing_residual_acc, ±swing_acc_max)
      两个 agent 使用 *不同 reward* 但 *相同 obs*, 独立更新

    设计思路:
      - planner 学习全局路径规划和避障, 不强调防摆
      - swing_rl 学习如何在当前速度下减小摆动, 不关心到哪里去
      - 两者解耦让各自 reward 信号更清晰, 避免多目标冲突

    与 DescentDualRLAgent 的区别:
      - Cruise 的 swing_rl 是残差叠加 (全局加速度 = planner + swing_res)
      - Descent 的 macro+residual 是不同维度的叠加
    """

    def __init__(self, config=None, algo="ppo"):
        if config is None:
            config = DEFAULT_CONFIG
        self.config = config
        self.algo   = algo

        phase_cfg = config["cruise_rl"]
        self.obs_dim = int(phase_cfg["obs_dim"])

        # ── planner agent (高层, 完整 xy 加速度) ──────────────────────────────
        planner_cfg = copy.deepcopy(config)
        planner_cfg["cruise_rl"] = copy.deepcopy(phase_cfg)
        planner_cfg["cruise_rl"]["action_dim"] = 2

        if algo == "ppo":
            self.planner_agent = PPOPhaseAgent("cruise", config=planner_cfg)
        else:
            self.planner_agent = SACPhaseAgent("cruise", config=planner_cfg)

        # ── swing_rl agent (低层残差防摆, acc 较小) ───────────────────────────
        swing_cfg = copy.deepcopy(config)
        swing_cfg["cruise_rl"] = copy.deepcopy(phase_cfg)
        swing_cfg["cruise_rl"]["action_dim"] = 2
        # 残差 acc 限幅更小: 防摆修正量不应超过导航 acc 的 50%
        swing_acc_max = float(phase_cfg.get("residual_acc_max_xy", 0.60)) * 0.5
        swing_cfg["cruise_rl"]["residual_acc_max_xy"] = swing_acc_max
        swing_cfg["ee_control"] = copy.deepcopy(config.get("ee_control", {}))
        swing_cfg["ee_control"]["acc_max_xy"] = swing_acc_max

        if algo == "ppo":
            self.swing_agent = PPOPhaseAgent("cruise", config=swing_cfg)
        else:
            self.swing_agent = SACPhaseAgent("cruise", config=swing_cfg)

        self.swing_acc_max = swing_acc_max

        # ── 代理属性 ──────────────────────────────────────────────────────────
        self.device     = self.planner_agent.device
        self.action_dim = 2
        self.gamma      = self.planner_agent.gamma
        self.use_lstm   = getattr(self.planner_agent, 'use_lstm', False)
        self.seq_len    = getattr(self.planner_agent, 'seq_len', 8)
        self.bc_coef    = 0.0
        self.actor      = self.planner_agent.actor
        self.critic     = self.planner_agent.critic
        self.buffer     = self.planner_agent.buffer

        if algo == "ppo":
            self.opt_actor  = self.planner_agent.opt_actor
            self.opt_critic = self.planner_agent.opt_critic
            self.gae_lambda = self.planner_agent.gae_lambda

        self._last_result = getattr(self.planner_agent, '_last_result', None)
        self.total_steps  = 0

        # 观测归一化 (共享)
        self.obs_norm      = self.planner_agent.obs_norm
        self.use_obs_norm  = self.planner_agent.use_obs_norm
        self._freeze_obs_norm = False
        self.obs_history   = ObsHistoryBuffer(self.obs_dim, self.seq_len)

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm:
            update = False
        if self.use_obs_norm:
            if update:
                self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    def reset_history(self):
        self.obs_history.reset()
        if hasattr(self.planner_agent, 'obs_history'):
            self.planner_agent.obs_history = self.obs_history
        if hasattr(self.swing_agent, 'obs_history'):
            self.swing_agent.obs_history = self.obs_history

    @torch.no_grad()
    def _act_agent(self, agent, norm_obs, deterministic):
        if self.use_lstm and hasattr(agent, 'obs_history'):
            obs_seq = self.obs_history.get_sequence()
            s = np_to_tensor(obs_seq, agent.device).unsqueeze(0)
            if self.algo == "ppo":
                action, lp, _, _ = agent.actor.get_action(s, deterministic=deterministic)
                value = agent.critic(s)
                return action.detach().cpu().numpy().flatten(), lp.cpu().item(), value.cpu().item()
            else:
                action = (agent.actor.deterministic(s) if deterministic
                          else agent.actor.sample(s)[0])
                return action.detach().cpu().numpy().flatten(), 0.0, 0.0
        else:
            s = np_to_tensor(norm_obs.reshape(1, -1), agent.device)
            if self.algo == "ppo":
                action, lp, _ = agent.actor.get_action(s, deterministic=deterministic)
                value = agent.critic(s)
                return action.detach().cpu().numpy().flatten(), lp.cpu().item(), value.cpu().item()
            else:
                action = (agent.actor.deterministic(s) if deterministic
                          else agent.actor.sample(s)[0])
                return action.detach().cpu().numpy().flatten(), 0.0, 0.0

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        """
        Returns:
            combined_acc (2D):  planner_acc + clipped(swing_res_acc)
            planner_acc (2D):   导航加速度
            swing_acc (2D):     防摆残差加速度
            planner_lp, planner_val: PPO 用
            swing_lp, swing_val:     PPO 用
        """
        self.obs_history.push(norm_obs)
        if hasattr(self.planner_agent, 'obs_history'):
            self.planner_agent.obs_history = self.obs_history
        if hasattr(self.swing_agent, 'obs_history'):
            self.swing_agent.obs_history = self.obs_history

        planner_acc, planner_lp, planner_val = self._act_agent(
            self.planner_agent, norm_obs, deterministic)
        swing_acc, swing_lp, swing_val = self._act_agent(
            self.swing_agent, norm_obs, deterministic)

        # 残差限幅
        swing_res = np.clip(swing_acc, -self.swing_acc_max, self.swing_acc_max)

        # 合并加速度
        combined = planner_acc + swing_res
        # 合并后的总 acc 仍受全局限幅
        acc_max = float(self.config["cruise_rl"].get("residual_acc_max_xy", 0.60))
        norm = float(np.linalg.norm(combined))
        if norm > acc_max and norm > 1e-8:
            combined = (combined / norm * acc_max).astype(np.float32)

        return combined, planner_acc, swing_acc, planner_lp, planner_val, swing_lp, swing_val

    @torch.no_grad()
    def act_simple(self, norm_obs, deterministic=False):
        """统一接口: 返回 (combined_acc, lp, val)."""
        combined, _, _, planner_lp, planner_val, _, _ = self.act(
            norm_obs, deterministic=deterministic)
        return combined, planner_lp, planner_val

    def add_to_buffers(self, norm_obs, planner_act, swing_act,
                       bc_planner, bc_swing,
                       planner_reward, swing_reward,
                       done, planner_val, planner_lp, swing_val, swing_lp):
        """
        向两个 agent 的 buffer 分别添加 (使用不同 reward).
        """
        if self.algo == "ppo":
            if self.use_lstm:
                obs_seq = self.obs_history.get_sequence()
                self.planner_agent.buffer.add(
                    obs_seq, planner_act, bc_planner, planner_reward, done, planner_val, planner_lp)
                self.swing_agent.buffer.add(
                    obs_seq, swing_act, bc_swing, swing_reward, done, swing_val, swing_lp)
            else:
                self.planner_agent.buffer.add(
                    norm_obs, planner_act, bc_planner, planner_reward, done, planner_val, planner_lp)
                self.swing_agent.buffer.add(
                    norm_obs, swing_act, bc_swing, swing_reward, done, swing_val, swing_lp)
        else:
            # SAC: 直接存入各自 buffer
            pass  # 由 train_phase.py 调用 remember() 分别处理

    def update(self, global_ts=None):
        """更新两个 agent."""
        if self.algo == "ppo":
            r1 = self.planner_agent.update(global_ts=global_ts)
            r2 = self.swing_agent.update(global_ts=global_ts)
            self._last_result = r1
            return r1, r2
        return None, None

    def get_value_for_state(self, norm_obs):
        """planner 的 critic value (用于 GAE 计算)."""
        return self.planner_agent.get_value_for_state(norm_obs)

    def get_log_std_per_dim(self):
        p = self.planner_agent.actor.get_log_std_per_dim()
        s = self.swing_agent.actor.get_log_std_per_dim()
        return np.concatenate([p, s])

    def save(self, path):
        torch.save({
            "planner_actor":  self.planner_agent.actor.state_dict(),
            "planner_critic": self.planner_agent.critic.state_dict(),
            "swing_actor":    self.swing_agent.actor.state_dict(),
            "swing_critic":   self.swing_agent.critic.state_dict(),
            "obs_norm":       self.obs_norm.state_dict(),
            "total_steps":    self.total_steps,
            "algo":           self.algo,
        }, path)

    def load(self, path, map_location=None):
        dev = self.planner_agent.device
        ck  = torch.load(path, map_location=map_location or dev, weights_only=False)

        if "planner_actor" in ck:
            self.planner_agent.actor.load_state_dict(ck["planner_actor"])
            self.planner_agent.critic.load_state_dict(ck["planner_critic"])
            self.swing_agent.actor.load_state_dict(ck["swing_actor"])
            self.swing_agent.critic.load_state_dict(ck["swing_critic"])
        elif "actor" in ck:
            # BC checkpoint: 共享加载到两个 agent
            try:
                self.planner_agent.actor.load_state_dict(ck["actor"])
                self.swing_agent.actor.load_state_dict(ck["actor"])
                print("  [CruiseDual] both agents ← BC actor weights")
            except Exception as e:
                self.planner_agent.actor.load_state_dict(ck["actor"], strict=False)
                self.swing_agent.actor.load_state_dict(ck["actor"], strict=False)
                print(f"  [CruiseDual] partial load: {e}")
            try:
                self.planner_agent.critic.load_state_dict(ck["critic"])
                self.swing_agent.critic.load_state_dict(ck["critic"])
            except Exception:
                pass

        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])
            self.swing_agent.obs_norm = self.obs_norm
        self.total_steps = ck.get("total_steps", 0)

    def reset_log_std_for_rl(self):
        if hasattr(self.planner_agent, 'reset_log_std_for_rl'):
            self.planner_agent.reset_log_std_for_rl()
        if hasattr(self.swing_agent, 'reset_log_std_for_rl'):
            self.swing_agent.reset_log_std_for_rl()

    def _update_entropy_coef(self, global_ts=None):
        """PPO entropy coef 退火 - 两个 sub-agent 同步更新."""
        self.planner_agent._update_entropy_coef(global_ts=global_ts)
        self.swing_agent._update_entropy_coef(global_ts=global_ts)

    def remember_dual(self, no, planner_act, swing_act, no2_norm, planner_rw, swing_rw, done):
        """SAC 双 RL: 分别向两个 buffer 存入不同 reward."""
        self.planner_agent.buffer.add(no, planner_act, no2_norm, planner_rw, done)
        self.swing_agent.buffer.add(no, swing_act, no2_norm, swing_rw, done)

    def train_step_dual(self):
        """SAC 双 RL: 两个 agent 独立训练."""
        r1 = (self.planner_agent.train_step()
              if hasattr(self.planner_agent, 'train_step') else None)
        r2 = (self.swing_agent.train_step()
              if hasattr(self.swing_agent, 'train_step') else None)
        return r1, r2