# ==============================================================================
# phase_agent.py — 三阶段统一 RL Agent (PPO + SAC)
#
# 所有阶段共享相同的网络结构, 仅 obs_dim / action_dim / 超参数不同。
# RL 输出: EE 3D 加速度 (tanh * acc_max)
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
from collections import namedtuple

from config import DEFAULT_CONFIG


# ==============================================================================
# 工具函数
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


class RunningMeanStd:
    """Welford's online mean/std estimator."""
    def __init__(self, shape, warm_start=2000, clip=10.0):
        # ★ warm_start 从 200→2000: 需要足够多样本才能得到稳定均值/方差
        # 200 步时 std 估计极度不稳定, 会导致归一化放大噪声
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S = np.zeros(shape, dtype=np.float64)
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
            self.S = self.S + (x - old) * (x - self.mean)

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
            -self.clip, self.clip)

    def state_dict(self):
        return {"n": self.n, "mean": self.mean.copy(), "S": self.S.copy()}
    def load_state_dict(self, d):
        self.n = d["n"]; self.mean = d["mean"].copy(); self.S = d["S"].copy()


# ==============================================================================
# 观测构建函数
# ==============================================================================

# 环境 obs 索引 (与 controller.py 一致)
OBS_EE_X, OBS_EE_Y = 0, 1
OBS_EE_VX, OBS_EE_VY = 2, 3
OBS_PL_X, OBS_PL_Y = 4, 5
OBS_PL_VX, OBS_PL_VY = 6, 7
OBS_EE_Z = 19
OBS_EE_VZ = 20
OBS_PL_Z = 21
OBS_PL_VZ = 22
OBS_TILT = 29
OBS_YAW = 30


def build_lift_obs(env_obs, env, start_xy, prev_tilt=0.0, prev_yaw=0.0):
    """
    提升阶段观测 (23D):
    ee_pos(3) + ee_vel(3) + pl_pos(3) + pl_vel(3) +
    ee_pl_offset(3) + tilt/yaw/tilt_rate/yaw_rate(4) +
    start_xy(2) + target_z(1) + z_error(1)
    """
    obs = env_obs
    dt = env.dt if hasattr(env, 'dt') else 0.1
    z_cruise = float(env.config["planning"]["payload_z_cruise"])

    ee_pos = np.array([obs[OBS_EE_X], obs[OBS_EE_Y], obs[OBS_EE_Z]])
    ee_vel = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]])
    pl_pos = np.array([obs[OBS_PL_X], obs[OBS_PL_Y], obs[OBS_PL_Z]])
    pl_vel = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY], obs[OBS_PL_VZ]])
    offset = ee_pos - pl_pos

    tilt = float(obs[OBS_TILT])
    yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt
    yaw_rate = (yaw - prev_yaw) / dt

    z_error = float(pl_pos[2] - z_cruise)

    lift_obs = np.concatenate([
        ee_pos, ee_vel, pl_pos, pl_vel, offset,
        [tilt, yaw, tilt_rate, yaw_rate],
        start_xy[:2],
        [z_cruise],
        [z_error],
    ]).astype(np.float32)

    return lift_obs, tilt, yaw


def build_cruise_obs(env_obs, env, target_xy, prev_tilt=0.0, prev_yaw=0.0):
    """
    平移阶段观测 (33D for n_obstacles=3):
    ee_xy(2) + ee_vxy(2) + pl_xy(2) + pl_vxy(2) +
    ee_pl_offset_xy(2) + tilt/yaw/tilt_rate/yaw_rate(4) +
    target_xy(2) + target_dist(1) +
    obstacles(3*n_obs_max) + joint_q(7)
    """
    obs = env_obs
    dt = env.dt if hasattr(env, 'dt') else 0.1
    n_obs_max = env.n_obstacles

    ee_xy = np.array([obs[OBS_EE_X], obs[OBS_EE_Y]])
    ee_vxy = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY]])
    pl_xy = np.array([obs[OBS_PL_X], obs[OBS_PL_Y]])
    pl_vxy = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY]])
    offset_xy = ee_xy - pl_xy

    tilt = float(obs[OBS_TILT])
    yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt
    yaw_rate = (yaw - prev_yaw) / dt

    dist = float(np.linalg.norm(pl_xy - target_xy))
    obs_data = obs[10:10 + 3 * n_obs_max].copy()
    joint_q = obs[31:38].copy()

    cruise_obs = np.concatenate([
        ee_xy, ee_vxy, pl_xy, pl_vxy, offset_xy,
        [tilt, yaw, tilt_rate, yaw_rate],
        target_xy,
        [dist],
        obs_data,
        joint_q,
    ]).astype(np.float32)

    return cruise_obs, tilt, yaw


def build_descent_obs(env_obs, env, target_xy, prev_tilt=0.0, prev_yaw=0.0):
    """
    下降阶段观测 (29D):
    ee_pos(3) + ee_vel(3) + pl_pos(3) + pl_vel(3) +
    ee_pl_offset(3) + tilt/yaw/tilt_rate/yaw_rate(4) +
    target_xy(2) + target_z(1) + pl_target_xy_err(2) +
    z_error(1) + rebar_errors(4)
    """
    obs = env_obs
    dt = env.dt if hasattr(env, 'dt') else 0.1
    target_pz = float(env.config["insertion"]["target_payload_z"])

    ee_pos = np.array([obs[OBS_EE_X], obs[OBS_EE_Y], obs[OBS_EE_Z]])
    ee_vel = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]])
    pl_pos = np.array([obs[OBS_PL_X], obs[OBS_PL_Y], obs[OBS_PL_Z]])
    pl_vel = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY], obs[OBS_PL_VZ]])
    offset = ee_pos - pl_pos

    tilt = float(obs[OBS_TILT])
    yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt
    yaw_rate = (yaw - prev_yaw) / dt

    pl_target_xy_err = pl_pos[:2] - target_xy
    z_error = float(pl_pos[2] - target_pz)
    rebar_err = obs[-4:].copy()

    descent_obs = np.concatenate([
        ee_pos, ee_vel, pl_pos, pl_vel, offset,
        [tilt, yaw, tilt_rate, yaw_rate],
        target_xy,
        [target_pz],
        pl_target_xy_err,
        [z_error],
        rebar_err,
    ]).astype(np.float32)

    return descent_obs, tilt, yaw


# ==============================================================================
# PPO Actor / Critic
# ==============================================================================

class PhaseActor(nn.Module):
    """
    通用策略网络。输出 tanh-squashed 连续动作。
    action = tanh(u) * action_scale
    """
    def __init__(self, obs_dim, action_dim, action_scale,
                 hidden_dim=256, n_layers=3,
                 log_std_init=-1.0, log_std_min=-4.0, log_std_max=0.5):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
        self.register_buffer('action_scale',
            torch.tensor(action_scale, dtype=torch.float32))

        layers = []
        d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            orthogonal_init(lin)
            layers += [lin, nn.ReLU()]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.mean_head, gain=0.01)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def _dist(self, s):
        feat = self.backbone(s)
        mean_raw = self.mean_head(feat)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp().expand_as(mean_raw)
        return mean_raw, std

    def get_action(self, s, deterministic=False):
        mean_raw, std = self._dist(s)
        dist = Normal(mean_raw, std)
        u = mean_raw if deterministic else dist.rsample()
        u_tanh = torch.tanh(u)
        action = u_tanh * self.action_scale
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy

    def evaluate_actions(self, s, actions_taken):
        mean_raw, std = self._dist(s)
        u_tanh = (actions_taken / (self.action_scale + 1e-8)).clamp(-1+1e-6, 1-1e-6)
        u = torch.atanh(u_tanh)
        dist = Normal(mean_raw, std)
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    def bc_forward(self, s, action_target):
        """BC 预训练前向。"""
        mean_raw, _ = self._dist(s)
        u_tanh_target = (action_target / (self.action_scale + 1e-8)).clamp(-0.999, 0.999)
        u_target = torch.atanh(u_tanh_target)
        bc_loss = F.mse_loss(mean_raw, u_target)
        action_pred = torch.tanh(mean_raw) * self.action_scale
        bc_loss_a = F.mse_loss(action_pred, action_target)
        return bc_loss, bc_loss_a, action_pred


class PhaseCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = []
        d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            orthogonal_init(lin)
            layers += [lin, nn.ReLU()]
            d = hidden_dim
        out = nn.Linear(d, 1)
        orthogonal_init(out, gain=1.0)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)


# ==============================================================================
# PPO RolloutBuffer
# ==============================================================================

class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, device):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.clear()

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
        self.ptr = 0
        self.full = False

    def add(self, obs, action, bc_target, reward, done, value, log_prob):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.bc_targets[i] = bc_target
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        last_gae = 0.0
        for t in reversed(range(self.n_steps)):
            next_val = last_value if t == self.n_steps - 1 else self.values[t+1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, batch_size, normalize_adv=True):
        assert self.full
        indices = np.random.permutation(self.n_steps)
        adv = self.advantages.copy()
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for start in range(0, self.n_steps, batch_size):
            idx = indices[start:start+batch_size]
            yield (
                np_to_tensor(self.obs[idx], self.device),
                np_to_tensor(self.actions[idx], self.device),
                np_to_tensor(self.bc_targets[idx], self.device),
                np_to_tensor(self.returns[idx], self.device).view(-1, 1),
                np_to_tensor(adv[idx], self.device),
                np_to_tensor(self.log_probs[idx], self.device),
            )


# ==============================================================================
# PPO Agent
# ==============================================================================

PPOResult = namedtuple("PPOResult", [
    "policy_loss", "value_loss", "entropy_loss", "bc_loss",
    "approx_kl", "clip_fraction", "total_loss"])
PPO_ZERO = PPOResult(0., 0., 0., 0., 0., 0., 0.)


class PPOPhaseAgent:
    """PPO agent for a single phase."""

    def __init__(self, phase_name, config=None):
        if config is None:
            config = DEFAULT_CONFIG
        self.config = config
        self.phase_name = phase_name

        phase_cfg = config[f"{phase_name}_rl"]
        cfg_ppo = config["ppo"]

        self.obs_dim = int(phase_cfg["obs_dim"])
        self.action_dim = int(phase_cfg["action_dim"])

        self.gamma = float(cfg_ppo["gamma"])
        self.gae_lambda = float(cfg_ppo["gae_lambda"])
        self.clip_eps = float(cfg_ppo["clip_eps"])
        self.value_loss_coef = float(cfg_ppo["value_loss_coef"])
        self.entropy_coef = float(cfg_ppo["entropy_coef"])
        self.max_grad_norm = float(cfg_ppo["max_grad_norm"])
        self.n_steps = int(cfg_ppo["n_steps"])
        self.n_epochs = int(cfg_ppo["n_epochs"])
        self.batch_size = int(cfg_ppo["batch_size"])
        self.norm_adv = bool(cfg_ppo["normalize_advantages"])
        self.target_kl = float(cfg_ppo.get("target_kl", 0.03))

        self.use_obs_norm = bool(cfg_ppo["use_obs_norm"])
        self.obs_norm = RunningMeanStd(
            shape=(self.obs_dim,),
            # ★ warm_start 设为 BC episode数 * avg_steps ≈ 1000*50=50000
            # 确保 BC 阶段能积累足够样本再开始归一化
            warm_start=int(cfg_ppo.get("obs_norm_warm_start", 5000)),
            clip=float(cfg_ppo["obs_norm_clip"]))
        self._freeze_obs_norm = False

        # ★ log_std floor annealing 参数 (从 config 读取)
        self._log_std_floor_init = float(cfg_ppo.get("log_std_floor_init", -0.5))
        self._log_std_floor_final = float(cfg_ppo.get("log_std_floor_final", -3.0))
        self._log_std_floor_steps = int(cfg_ppo.get("log_std_floor_steps", 800_000))

        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))

        # 动作尺度: EE 加速度范围
        ee_cfg = config.get("ee_control", {})
        if self.action_dim == 3:
            acc_max_xy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            acc_max_z = float(phase_cfg.get("acc_max_z", ee_cfg.get("acc_max_z", 3.0)))
            action_scale = [acc_max_xy, acc_max_xy, acc_max_z]
        else:  # action_dim == 2 (cruise, xy only)
            acc_max_xy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            action_scale = [acc_max_xy, acc_max_xy]

        self.actor = PhaseActor(
            self.obs_dim, self.action_dim, action_scale,
            hidden_dim=int(cfg_ppo["hidden_dim"]),
            n_layers=int(cfg_ppo["n_layers"]),
            log_std_init=float(cfg_ppo["log_std_init"]),
            log_std_min=float(cfg_ppo["log_std_min"]),
            log_std_max=float(cfg_ppo["log_std_max"]),
        ).to(self.device)

        self.critic = PhaseCritic(
            self.obs_dim,
            hidden_dim=int(cfg_ppo["hidden_dim"]),
            n_layers=int(cfg_ppo["n_layers"]),
        ).to(self.device)

        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg_ppo["lr_actor"]), eps=1e-5)
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg_ppo["lr_critic"]), eps=1e-5)

        self.buffer = RolloutBuffer(
            self.n_steps, self.obs_dim, self.action_dim, self.device)

        self.total_steps = 0
        self._last_result = PPO_ZERO
        self.bc_coef = 0.0

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm:
            update = False
        if self.use_obs_norm:
            if update:
                self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        action, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
        value = self.critic(s)
        return (action.cpu().numpy().flatten(),
                log_prob.cpu().item(),
                value.cpu().item())

    def update(self):
        if not self.buffer.full:
            return PPO_ZERO

        total_pl = total_vl = total_el = total_bl = 0.0
        total_kl = total_cf = 0.0
        n_updates = 0
        stop_early = False

        for epoch in range(self.n_epochs):
            if stop_early: break
            for batch in self.buffer.get_minibatches(self.batch_size, self.norm_adv):
                obs_b, act_b, bc_b, ret_b, adv_b, old_lp_b = batch

                # Critic
                value = self.critic(obs_b)
                value_loss = F.huber_loss(value, ret_b)
                self.opt_critic.zero_grad()
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                # Actor
                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                bc_loss = torch.zeros(1, device=self.device)
                if self.bc_coef > 0:
                    bc_loss_u, _, _ = self.actor.bc_forward(obs_b, bc_b)
                    bc_loss = bc_loss_u

                actor_total = (policy_loss
                               + self.entropy_coef * entropy_loss
                               + self.bc_coef * bc_loss)

                self.opt_actor.zero_grad()
                actor_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # ★ log_std floor annealing (防熵过早坍缩)
                # 用 config 驱动的参数: 初始下限较高(允许大探索), 随训练线性降低
                with torch.no_grad():
                    anneal_frac = min(self.total_steps / max(self._log_std_floor_steps, 1), 1.0)
                    log_std_floor = (self._log_std_floor_init +
                                     anneal_frac * (self._log_std_floor_final - self._log_std_floor_init))
                    self.actor.log_std.data.clamp_(min=log_std_floor)

                with torch.no_grad():
                    approx_kl = (old_lp_b - new_lp).mean().abs().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_el += entropy_loss.item()
                total_bl += bc_loss.item()
                total_kl += approx_kl
                total_cf += clip_frac
                n_updates += 1

                if approx_kl > self.target_kl:
                    stop_early = True; break

        n = max(n_updates, 1)
        result = PPOResult(
            total_pl/n, total_vl/n, total_el/n, total_bl/n,
            total_kl/n, total_cf/n, (total_pl+total_vl+total_el+total_bl)/n)
        self._last_result = result
        self.buffer.clear()
        return result

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "total_steps": self.total_steps,
            "bc_coef": self.bc_coef,
            "obs_norm": self.obs_norm.state_dict(),
            "phase_name": self.phase_name,
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
        self.bc_coef = ck.get("bc_coef", 0.0)
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])


# ==============================================================================
# SAC Actor / Critic
# ==============================================================================

class SACPhaseActor(nn.Module):
    """SAC 策略网络 — 输出均值和对数标准差。"""
    def __init__(self, obs_dim, action_dim, action_scale,
                 hidden_dim=256, n_layers=3,
                 log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.register_buffer('action_scale',
            torch.tensor(action_scale, dtype=torch.float32))

        layers = []
        d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim)
            orthogonal_init(lin)
            layers += [lin, nn.ReLU()]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        orthogonal_init(self.mean_head, gain=0.01)
        orthogonal_init(self.log_std_head, gain=0.01)

    def forward(self, s):
        feat = self.backbone(s)
        mean = self.mean_head(feat)
        log_std = self.log_std_head(feat).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, s):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        dist = Normal(mean, std)
        u = dist.rsample()
        u_tanh = torch.tanh(u)
        action = u_tanh * self.action_scale
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1 - u_tanh.pow(2) + 1e-6).sum(-1)
        return action, log_prob

    def deterministic(self, s):
        mean, _ = self.forward(s)
        return torch.tanh(mean) * self.action_scale


class SACPhaseCritic(nn.Module):
    """双 Q 网络。"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        in_dim = obs_dim + action_dim
        def make_q():
            layers = []
            d = in_dim
            for _ in range(n_layers):
                lin = nn.Linear(d, hidden_dim)
                orthogonal_init(lin)
                layers += [lin, nn.ReLU()]
                d = hidden_dim
            out = nn.Linear(d, 1)
            orthogonal_init(out, gain=1.0)
            layers.append(out)
            return nn.Sequential(*layers)
        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, s, a):
        sa = torch.cat([s, a], -1)
        return self.q1(sa), self.q2(sa)


class ReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), np.float32)
        self.actions = np.zeros((max_size, action_dim), np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), np.float32)
        self.rewards = np.zeros((max_size, 1), np.float32)
        self.dones = np.zeros((max_size, 1), np.float32)

    def add(self, obs, action, next_obs, reward, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.next_obs[i] = next_obs
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (self.obs[idx], self.actions[idx], self.next_obs[idx],
                self.rewards[idx], self.dones[idx])


SACResult = namedtuple("SACResult", [
    "critic_loss", "actor_loss", "alpha_loss", "alpha", "q_mean"])
SAC_ZERO = SACResult(0., 0., 0., 0., 0.)


class SACPhaseAgent:
    """SAC agent for a single phase."""

    def __init__(self, phase_name, config=None):
        if config is None:
            config = DEFAULT_CONFIG
        self.config = config
        self.phase_name = phase_name

        phase_cfg = config[f"{phase_name}_rl"]
        cfg_sac = config["sac"]

        self.obs_dim = int(phase_cfg["obs_dim"])
        self.action_dim = int(phase_cfg["action_dim"])

        self.gamma = float(cfg_sac["gamma"])
        self.tau = float(cfg_sac["tau"])
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

        hidden = int(cfg_sac["hidden_dim"])
        n_layers = int(cfg_sac["n_layers"])

        # 动作尺度
        ee_cfg = config.get("ee_control", {})
        if self.action_dim == 3:
            acc_max_xy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            acc_max_z = float(phase_cfg.get("acc_max_z", ee_cfg.get("acc_max_z", 3.0)))
            action_scale = [acc_max_xy, acc_max_xy, acc_max_z]
        else:
            acc_max_xy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            action_scale = [acc_max_xy, acc_max_xy]

        self.actor = SACPhaseActor(
            self.obs_dim, self.action_dim, action_scale,
            hidden_dim=hidden, n_layers=n_layers,
        ).to(self.device)

        self.critic = SACPhaseCritic(
            self.obs_dim, self.action_dim, hidden, n_layers).to(self.device)
        self.target_critic = SACPhaseCritic(
            self.obs_dim, self.action_dim, hidden, n_layers).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg_sac["lr_actor"]))
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=float(cfg_sac["lr_critic"]))

        self.auto_alpha = bool(cfg_sac["auto_alpha"])
        alpha_init = float(cfg_sac["alpha_init"])
        self.log_alpha = torch.tensor(np.log(alpha_init),
                                       dtype=torch.float32, device=self.device,
                                       requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=float(cfg_sac["lr_alpha"]))
        target_ratio = float(cfg_sac["target_entropy_ratio"])
        self.target_entropy = -target_ratio * self.action_dim

        self.buffer = ReplayBuffer(
            int(cfg_sac["buffer_size"]), self.obs_dim, self.action_dim)

        self.total_steps = 0
        self._last_result = SAC_ZERO

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm:
            update = False
        if self.use_obs_norm:
            if update:
                self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        if deterministic:
            action = self.actor.deterministic(s)
        else:
            action, _ = self.actor.sample(s)
        return action.cpu().numpy().flatten()

    def remember(self, norm_obs, action, norm_next_obs, reward, done):
        self.buffer.add(norm_obs, action, norm_next_obs, reward, done)

    def train_step(self):
        if self.buffer.size < max(self.batch_size, self.warmup_steps):
            return SAC_ZERO

        dev = self.device
        total_cl = total_al = total_alpha_l = total_q = 0.0

        for _ in range(self.updates_per_step):
            s, a, ns, r, d = self.buffer.sample(self.batch_size)
            si = np_to_tensor(s, dev)
            ai = np_to_tensor(a, dev)
            nsi = np_to_tensor(ns, dev)
            ri = np_to_tensor(r, dev) * self.reward_scale
            di = np_to_tensor(d, dev)
            alpha = self.log_alpha.exp().detach()

            with torch.no_grad():
                next_a, next_lp = self.actor.sample(nsi)
                tq1, tq2 = self.target_critic(nsi, next_a)
                target_q = ri + self.gamma * (1 - di) * (
                    torch.min(tq1, tq2) - alpha * next_lp.unsqueeze(-1))

            q1, q2 = self.critic(si, ai)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            self.opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
            self.opt_critic.step()

            new_a, new_lp = self.actor.sample(si)
            q1_new, q2_new = self.critic(si, new_a)
            q_min = torch.min(q1_new, q2_new)
            actor_loss = (alpha * new_lp.unsqueeze(-1) - q_min).mean()
            self.opt_actor.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
            self.opt_actor.step()

            alpha_loss_val = 0.0
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha.exp() * (
                    new_lp.detach() + self.target_entropy)).mean()
                self.opt_alpha.zero_grad()
                alpha_loss.backward()
                self.opt_alpha.step()
                alpha_loss_val = alpha_loss.item()

            soft_update(self.target_critic, self.critic, self.tau)
            total_cl += critic_loss.item()
            total_al += actor_loss.item()
            total_alpha_l += alpha_loss_val
            total_q += q_min.mean().item()

        n = self.updates_per_step
        result = SACResult(total_cl/n, total_al/n, total_alpha_l/n,
                           self.alpha, total_q/n)
        self._last_result = result
        return result

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "total_steps": self.total_steps,
            "obs_norm": self.obs_norm.state_dict(),
            "phase_name": self.phase_name,
        }, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.target_critic.load_state_dict(ck["target_critic"])
        if "opt_actor" in ck:
            self.opt_actor.load_state_dict(ck["opt_actor"])
        if "opt_critic" in ck:
            self.opt_critic.load_state_dict(ck["opt_critic"])
        if "log_alpha" in ck:
            self.log_alpha.data.copy_(ck["log_alpha"].to(self.device))
        if "opt_alpha" in ck:
            self.opt_alpha.load_state_dict(ck["opt_alpha"])
        self.total_steps = ck.get("total_steps", 0)
        if "obs_norm" in ck:
            self.obs_norm.load_state_dict(ck["obs_norm"])