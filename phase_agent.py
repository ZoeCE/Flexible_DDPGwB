# ==============================================================================
# phase_agent.py — 三阶段统一 RL Agent (PPO + SAC) v3 (学界全面优化版)
#
# 学界依据与新增特性:
#   [PLASTICITY]  Lyle et al. NeurIPS 2024 "No Representation No Trust"
#                 PPO 多 epoch 非稳态更新导致特征坍塌、表征退化
#                 → 周期性重置 Adam 一阶动量 reset_optimizer_momentum()
#   [ENT-ANNEAL]  PPO-CMA (Hämäläinen 2018) + Unity ML-Agents 工程验证
#                 退火 entropy 系数: 探索期高熵 → 利用期低熵
#                 → entropy_coef 从 0.15 线性退火到 0.02
#   [LOG-STD]     log_std_min=-1.5 硬下界 + floor annealing, std ≥ 0.22
#   [BOOTSTRAP-V] Bootstrapped Reward Shaping (2025)
#                 → get_value_for_state() 供 Bootstrapped PBRS 查询
#   [HER]         Andrychowicz et al. NeurIPS 2017 "Hindsight Experience Replay"
#                 → SAC descent 专用 HERReplayBuffer: 失败轨迹重标注正信号
#   [BUGFIX]      build_lift_obs: pl_vel[2] OBS_PL_Z → OBS_PL_VZ
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from collections import namedtuple

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
# 观测索引 (与 controller.py 同步)
# ==============================================================================
OBS_EE_X,  OBS_EE_Y  = 0, 1
OBS_EE_VX, OBS_EE_VY = 2, 3
OBS_PL_X,  OBS_PL_Y  = 4, 5
OBS_PL_VX, OBS_PL_VY = 6, 7
OBS_EE_Z  = 19; OBS_EE_VZ = 20
OBS_PL_Z  = 21; OBS_PL_VZ = 22   # ★ BUGFIX: 原代码误用 OBS_PL_Z 作速度
OBS_TILT  = 29; OBS_YAW   = 30


# ==============================================================================
# 观测构建
# ==============================================================================

def build_lift_obs(env_obs, env, start_xy, prev_tilt=0.0, prev_yaw=0.0):
    obs = env_obs; dt = getattr(env, 'dt', 0.1)
    z_cruise = float(env.config["planning"]["payload_z_cruise"])
    ee_pos = np.array([obs[OBS_EE_X], obs[OBS_EE_Y], obs[OBS_EE_Z]])
    ee_vel = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY], obs[OBS_EE_VZ]])
    pl_pos = np.array([obs[OBS_PL_X], obs[OBS_PL_Y], obs[OBS_PL_Z]])
    pl_vel = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY], obs[OBS_PL_VZ]])  # ★ FIXED
    offset = ee_pos - pl_pos
    tilt = float(obs[OBS_TILT]); yaw = float(obs[OBS_YAW])
    tilt_rate = (tilt - prev_tilt) / dt; yaw_rate = (yaw - prev_yaw) / dt
    z_error = float(pl_pos[2] - z_cruise)
    return np.concatenate([ee_pos, ee_vel, pl_pos, pl_vel, offset,
                           [tilt, yaw, tilt_rate, yaw_rate],
                           start_xy[:2], [z_cruise], [z_error]]).astype(np.float32), tilt, yaw


def build_cruise_obs(env_obs, env, target_xy, prev_tilt=0.0, prev_yaw=0.0):
    obs = env_obs; dt = getattr(env, 'dt', 0.1); n_obs_max = env.n_obstacles
    ee_xy  = np.array([obs[OBS_EE_X], obs[OBS_EE_Y]])
    ee_vxy = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY]])
    pl_xy  = np.array([obs[OBS_PL_X], obs[OBS_PL_Y]])
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
# PPO Actor / Critic
# ==============================================================================

class PhaseActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_scale,
                 hidden_dim=256, n_layers=3,
                 log_std_init=-0.5, log_std_min=-2.0, log_std_max=0.3):
        super().__init__()
        # [FIX-LOGSTD v4] 硬约束上界 max=0.3 (std≤1.35) 防止 tanh 饱和与熵爆炸
        # 原 max=0.5 (std≤1.65) 导致训练中 logstd 被 entropy bonus 持续推至上界
        # 下界 min=-2.0 (std≥0.135) 保留足够探索能力
        self.log_std_min = log_std_min; self.log_std_max = log_std_max
        self.action_dim = action_dim
        self.register_buffer('action_scale', torch.tensor(action_scale, dtype=torch.float32))
        layers = []; d = obs_dim
        for _ in range(n_layers):
            lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
            layers += [lin, nn.ReLU()]; d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim); orthogonal_init(self.mean_head, gain=0.01)
        # [FIX-LOGSTD v4] 初始值 clamp 到安全范围, 防止外部传入越界值
        safe_init = float(np.clip(log_std_init, log_std_min, log_std_max))
        self.log_std = nn.Parameter(torch.ones(action_dim) * safe_init)

    def _dist(self, s):
        feat = self.backbone(s); mean_raw = self.mean_head(feat)
        # [FIX-LOGSTD v4] 双重保障: clamp 同时作用于 forward (梯度路径上)
        # 使得任何梯度更新都无法将 log_std 推出 [min, max]
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
        """[MON] 各维度 log_std, 供 wandb 分维度记录。"""
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
# PPO Rollout Buffer
# ==============================================================================

class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, device):
        self.n_steps = n_steps; self.obs_dim = obs_dim
        self.action_dim = action_dim; self.device = device; self.clear()

    def clear(self):
        self.obs       = np.zeros((self.n_steps, self.obs_dim),    np.float32)
        self.actions   = np.zeros((self.n_steps, self.action_dim), np.float32)
        self.bc_targets= np.zeros((self.n_steps, self.action_dim), np.float32)
        self.rewards   = np.zeros(self.n_steps, np.float32)
        self.dones     = np.zeros(self.n_steps, np.float32)
        self.values    = np.zeros(self.n_steps, np.float32)
        self.log_probs = np.zeros(self.n_steps, np.float32)
        self.advantages= np.zeros(self.n_steps, np.float32)
        self.returns   = np.zeros(self.n_steps, np.float32)
        self.ptr = 0; self.full = False

    def add(self, obs, action, bc_target, reward, done, value, log_prob):
        i = self.ptr
        self.obs[i]=obs; self.actions[i]=action; self.bc_targets[i]=bc_target
        self.rewards[i]=reward; self.dones[i]=float(done)
        self.values[i]=value; self.log_probs[i]=log_prob
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
            yield (
                np_to_tensor(self.obs[idx],       self.device),
                np_to_tensor(self.actions[idx],   self.device),
                np_to_tensor(self.bc_targets[idx],self.device),
                np_to_tensor(self.returns[idx],   self.device).view(-1, 1),
                np_to_tensor(adv[idx],            self.device),
                np_to_tensor(self.log_probs[idx], self.device),
            )


# ==============================================================================
# PPO Agent v3
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
        self.obs_dim    = int(phase_cfg["obs_dim"])
        self.action_dim = int(phase_cfg["action_dim"])
        self.gamma         = float(cfg_ppo["gamma"])
        self.gae_lambda    = float(cfg_ppo["gae_lambda"])
        self.clip_eps      = float(cfg_ppo["clip_eps"])
        self.value_loss_coef = float(cfg_ppo["value_loss_coef"])
        self.max_grad_norm = float(cfg_ppo["max_grad_norm"])
        self.n_steps   = int(cfg_ppo["n_steps"])
        self.n_epochs  = int(cfg_ppo["n_epochs"])
        self.batch_size= int(cfg_ppo["batch_size"])
        self.norm_adv  = bool(cfg_ppo["normalize_advantages"])
        self.target_kl = float(cfg_ppo.get("target_kl", 0.03))

        # [ENT-ANNEAL] entropy_coef 退火 (PPO-CMA / Unity ML-Agents)
        self.entropy_coef_start = float(cfg_ppo.get("entropy_coef_start",
                                                      cfg_ppo.get("entropy_coef", 0.15)))
        self.entropy_coef_end   = float(cfg_ppo.get("entropy_coef_end", 0.02))
        self.entropy_coef_anneal_steps = int(cfg_ppo.get("entropy_coef_anneal_steps", 1_000_000))
        self.entropy_coef = self.entropy_coef_start

        self.use_obs_norm = bool(cfg_ppo["use_obs_norm"])
        self.obs_norm = RunningMeanStd(
            shape=(self.obs_dim,),
            warm_start=int(cfg_ppo.get("obs_norm_warm_start", 5000)),
            clip=float(cfg_ppo["obs_norm_clip"]))
        self._freeze_obs_norm = False

        # [LOG-STD] floor annealing 参数
        self._log_std_floor_init  = float(cfg_ppo.get("log_std_floor_init",  0.0))
        self._log_std_floor_final = float(cfg_ppo.get("log_std_floor_final", -1.5))
        self._log_std_floor_steps = int(cfg_ppo.get("log_std_floor_steps",  800_000))

        # [PLASTICITY] 周期性 Adam 动量重置 (Lyle et al. NeurIPS 2024)
        self._plasticity_reset_interval = int(cfg_ppo.get("plasticity_reset_interval", 200_000))
        self._last_plasticity_reset = 0

        gpu_id = config["train"].get("gpu_id", 0)
        self.device = (torch.device(f"cuda:{gpu_id}")
                       if torch.cuda.is_available() and gpu_id >= 0
                       else torch.device("cpu"))

        ee_cfg = config.get("ee_control", {})
        if self.action_dim == 3:
            acc_max_xy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            acc_max_z  = float(phase_cfg.get("acc_max_z",  ee_cfg.get("acc_max_z",  3.0)))
            action_scale = [acc_max_xy, acc_max_xy, acc_max_z]
        else:
            acc_max_xy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            action_scale = [acc_max_xy, acc_max_xy]

        self.actor = PhaseActor(
            self.obs_dim, self.action_dim, action_scale,
            hidden_dim=int(cfg_ppo["hidden_dim"]), n_layers=int(cfg_ppo["n_layers"]),
            log_std_init=float(cfg_ppo["log_std_init"]),
            log_std_min=float(cfg_ppo["log_std_min"]),
            log_std_max=float(cfg_ppo["log_std_max"]),
        ).to(self.device)

        self.critic = PhaseCritic(
            self.obs_dim, hidden_dim=int(cfg_ppo["hidden_dim"]),
            n_layers=int(cfg_ppo["n_layers"]),
        ).to(self.device)

        self._lr_actor  = float(cfg_ppo["lr_actor"])
        self._lr_critic = float(cfg_ppo["lr_critic"])
        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=self._lr_actor,  eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self._lr_critic, eps=1e-5)
        self.buffer = RolloutBuffer(self.n_steps, self.obs_dim, self.action_dim, self.device)
        self.total_steps = 0; self._last_result = PPO_ZERO; self.bc_coef = 0.0

    # ── 推理 ──────────────────────────────────────────────────────────────────

    def normalize_obs(self, obs, update=True):
        if self._freeze_obs_norm: update = False
        if self.use_obs_norm:
            if update: self.obs_norm.update(obs)
            return self.obs_norm.normalize(obs)
        return obs.astype(np.float32)

    @torch.no_grad()
    def act(self, norm_obs, deterministic=False):
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        action, log_prob, _ = self.actor.get_action(s, deterministic=deterministic)
        value = self.critic(s)
        return action.cpu().numpy().flatten(), log_prob.cpu().item(), value.cpu().item()

    @torch.no_grad()
    def get_value_for_state(self, norm_obs):
        """[BOOTSTRAP-V] Critic 值函数查询, 供 Bootstrapped PBRS 使用。"""
        s = np_to_tensor(norm_obs.reshape(1, -1), self.device)
        return self.critic(s).cpu().item()

    def get_log_std_per_dim(self):
        """[MON] 各维度 log_std → numpy。"""
        return self.actor.get_log_std_per_dim()

    # ── 训练 ──────────────────────────────────────────────────────────────────

    def _maybe_reset_plasticity(self):
        """[PLASTICITY] 周期性重置 Adam 一阶矩, 防止表征坍塌 (Lyle 2024)。"""
        if self._plasticity_reset_interval <= 0: return
        if self.total_steps - self._last_plasticity_reset < self._plasticity_reset_interval: return
        for opt in [self.opt_actor, self.opt_critic]:
            for group in opt.param_groups:
                for p in group['params']:
                    if p in opt.state and 'exp_avg' in opt.state[p]:
                        opt.state[p]['exp_avg'].zero_()  # 清零一阶矩, 保留二阶矩
        self._last_plasticity_reset = self.total_steps
        print(f"  [Plasticity] Adam 一阶矩重置 @ step {self.total_steps}")

    def _update_entropy_coef(self, global_ts=None):
        """[ENT-ANNEAL] 线性退火 entropy_coef。
        
        使用外部传入的 global_ts (每步实时递增) 而非 self.total_steps
        (按 rollout buffer 批量更新), 确保退火曲线平滑可见。
        """
        ts = global_ts if global_ts is not None else self.total_steps
        frac = min(ts / max(self.entropy_coef_anneal_steps, 1), 1.0)
        self.entropy_coef = (self.entropy_coef_start +
                              frac * (self.entropy_coef_end - self.entropy_coef_start))

    def update(self, global_ts=None):
        """执行一次 PPO 更新。
        
        Args:
            global_ts: 当前全局 env step 数, 用于 entropy_coef 退火。
                       若为 None 则回退到 self.total_steps。
        """
        if not self.buffer.full: return PPO_ZERO
        self._maybe_reset_plasticity()
        self._update_entropy_coef(global_ts=global_ts)

        total_pl = total_vl = total_el = total_bl = 0.0
        total_kl = total_cf = 0.0; n_updates = 0; stop_early = False

        for epoch in range(self.n_epochs):
            if stop_early: break
            for batch in self.buffer.get_minibatches(self.batch_size, self.norm_adv):
                obs_b, act_b, bc_b, ret_b, adv_b, old_lp_b = batch

                # Critic
                value = self.critic(obs_b); value_loss = F.huber_loss(value, ret_b)
                self.opt_critic.zero_grad()
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.opt_critic.step()

                # Actor
                new_lp, entropy = self.actor.evaluate_actions(obs_b, act_b)
                ratio = (new_lp - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b
                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                bc_loss = torch.zeros(1, device=self.device)
                if self.bc_coef > 0:
                    bc_loss, _, _ = self.actor.bc_forward(obs_b, bc_b)

                actor_total = (policy_loss
                               + self.entropy_coef * entropy_loss
                               + self.bc_coef * bc_loss)
                self.opt_actor.zero_grad(); actor_total.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.opt_actor.step()

                # [LOG-STD] floor annealing
                with torch.no_grad():
                    frac = min(self.total_steps / max(self._log_std_floor_steps, 1), 1.0)
                    floor = self._log_std_floor_init + frac * (self._log_std_floor_final - self._log_std_floor_init)
                    self.actor.log_std.data.clamp_(min=floor)

                with torch.no_grad():
                    approx_kl = (old_lp_b - new_lp).mean().abs().item()
                    clip_frac = ((ratio - 1).abs() > self.clip_eps).float().mean().item()

                total_pl += policy_loss.item(); total_vl += value_loss.item()
                total_el += entropy_loss.item(); total_bl += bc_loss.item()
                total_kl += approx_kl; total_cf += clip_frac; n_updates += 1
                if approx_kl > self.target_kl: stop_early = True; break

        n = max(n_updates, 1)
        result = PPOResult(total_pl/n, total_vl/n, total_el/n, total_bl/n,
                           total_kl/n, total_cf/n, (total_pl+total_vl+total_el+total_bl)/n,
                           self.entropy_coef)
        self._last_result = result; self.buffer.clear()
        return result

    def reset_log_std_for_rl(self):
        import math
        with torch.no_grad():
            # [FIX-LOGSTD v4] 确保重置值在硬约束范围内
            # floor_init 来自 config, 可能历史值偏高; 强制 clamp 到 [min, max]
            raw_init = float(self._log_std_floor_init)
            init_val = float(np.clip(raw_init,
                                     self.actor.log_std_min,
                                     self.actor.log_std_max))
            self.actor.log_std.data.fill_(init_val)
        # 重建 optimizer 清零动量, 防止 BC 阶段积累的 Adam 状态影响 RL 初期
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self._lr_actor, eps=1e-5)
        self._last_plasticity_reset = self.total_steps
        print(f"  [PPO] log_std reset → {init_val:.3f} (std={math.exp(init_val):.3f}), "
              f"clamp=[{self.actor.log_std_min:.1f}, {self.actor.log_std_max:.1f}]")
        print(f"  [ENT] entropy_coef anneal: {self.entropy_coef_start:.3f} → "
              f"{self.entropy_coef_end:.3f} over {self.entropy_coef_anneal_steps} steps")

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "total_steps": self.total_steps, "bc_coef": self.bc_coef,
            "obs_norm": self.obs_norm.state_dict(), "phase_name": self.phase_name,
            "entropy_coef": self.entropy_coef,
        }, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"])
        if "opt_actor"  in ck: self.opt_actor.load_state_dict(ck["opt_actor"])
        if "opt_critic" in ck: self.opt_critic.load_state_dict(ck["opt_critic"])
        self.total_steps = ck.get("total_steps", 0); self.bc_coef = ck.get("bc_coef", 0.0)
        if "obs_norm" in ck: self.obs_norm.load_state_dict(ck["obs_norm"])


# ==============================================================================
# SAC Actor / Critic
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
        self.mean_head    = nn.Linear(hidden_dim, action_dim); orthogonal_init(self.mean_head, gain=0.01)
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
                lin = nn.Linear(d, hidden_dim); orthogonal_init(lin)
                layers += [lin, nn.ReLU()]; d = hidden_dim
            out = nn.Linear(d, 1); orthogonal_init(out, gain=1.0); layers.append(out)
            return nn.Sequential(*layers)
        self.q1 = _make(); self.q2 = _make()

    def forward(self, s, a):
        sa = torch.cat([s, a], -1); return self.q1(sa), self.q2(sa)


# ==============================================================================
# Replay Buffers — SimpleReplayBuffer & HERReplayBuffer
# [HER] Andrychowicz et al. NeurIPS 2017 — descent 专用
# ==============================================================================

class SimpleReplayBuffer:
    """标准 SAC replay buffer。"""
    def __init__(self, max_size, obs_dim, action_dim, **kwargs):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.obs     = np.zeros((max_size, obs_dim),    np.float32)
        self.actions = np.zeros((max_size, action_dim), np.float32)
        self.next_obs= np.zeros((max_size, obs_dim),    np.float32)
        self.rewards = np.zeros((max_size, 1),          np.float32)
        self.dones   = np.zeros((max_size, 1),          np.float32)

    def add(self, obs, action, next_obs, reward, done, achieved_pos=None):
        i = self.ptr
        self.obs[i]=obs; self.actions[i]=action; self.next_obs[i]=next_obs
        self.rewards[i]=reward; self.dones[i]=float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def flush_episode_with_her(self, *a, **kw): pass  # no-op

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.obs[idx], self.actions[idx], self.next_obs[idx], self.rewards[idx], self.dones[idx]

    @property
    def is_ready(self): return self.size >= 1000


class HERReplayBuffer:
    """
    [HER] Hindsight Experience Replay buffer (Andrychowicz 2017).
    descent 阶段专用: 以 'future' 策略重标注失败轨迹。

    每条真实 transition 额外添加 her_k 条重标注 transition:
      - 从当前时刻 t 随机采样未来时刻 ft ∈ [t+1, T)
      - 将 ft 时刻实际到达的 payload 位置作为新目标
      - 重标注 reward = her_reward_scale * success_bonus
    这提供了"至少到达了某中间状态"的稠密正信号。
    """
    def __init__(self, max_size, obs_dim, action_dim,
                 her_k=4, success_bonus=50.0, her_reward_scale=0.3, **kwargs):
        self.max_size = max_size; self.ptr = 0; self.size = 0
        self.her_k = her_k
        self.her_reward = success_bonus * her_reward_scale

        self.obs     = np.zeros((max_size, obs_dim),    np.float32)
        self.actions = np.zeros((max_size, action_dim), np.float32)
        self.next_obs= np.zeros((max_size, obs_dim),    np.float32)
        self.rewards = np.zeros((max_size, 1),          np.float32)
        self.dones   = np.zeros((max_size, 1),          np.float32)

        # episode 临时缓存: (obs, action, next_obs, reward, done, pl_pos_3d)
        self._ep  = []

    def _store(self, obs, action, next_obs, reward, done):
        i = self.ptr
        self.obs[i]=obs; self.actions[i]=action; self.next_obs[i]=next_obs
        self.rewards[i]=reward; self.dones[i]=float(done)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def add(self, obs, action, next_obs, reward, done, achieved_pos=None):
        self._store(obs, action, next_obs, reward, done)
        if achieved_pos is not None:
            self._ep.append((obs.copy(), action.copy(), next_obs.copy(),
                             float(reward), bool(done),
                             np.array(achieved_pos, np.float32)))

    def flush_episode_with_her(self, target_xy, target_z, obs_relabel_fn=None):
        """
        [HER] episode 结束时调用 future 目标重标注。

        obs_relabel_fn: callable(base_obs, new_goal_xy, new_goal_z) → new_obs
          若为 None, 使用默认替换 (obs 原样返回, 仅 reward 修改)
        """
        ep = self._ep; T = len(ep)
        if T < 2: self._ep = []; return

        for t in range(T):
            n_future = min(self.her_k, T - t - 1)
            if n_future <= 0: continue
            future_ts = np.random.randint(t+1, T, size=n_future)
            for ft in future_ts:
                achieved = ep[ft][5]  # payload xyz at time ft
                if obs_relabel_fn is not None:
                    try:
                        her_obs  = obs_relabel_fn(ep[t][0], achieved[:2], float(achieved[2]))
                        her_nobs = obs_relabel_fn(ep[t][2], achieved[:2], float(achieved[2]))
                    except Exception:
                        her_obs  = ep[t][0]; her_nobs = ep[t][2]
                else:
                    her_obs  = ep[t][0]; her_nobs = ep[t][2]
                self._store(her_obs, ep[t][1], her_nobs, self.her_reward, False)

        self._ep = []

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return self.obs[idx], self.actions[idx], self.next_obs[idx], self.rewards[idx], self.dones[idx]

    @property
    def is_ready(self): return self.size >= 1000


# ==============================================================================
# SAC Agent v3
# ==============================================================================

SACResult = namedtuple("SACResult", ["critic_loss", "actor_loss", "alpha_loss", "alpha", "q_mean"])
SAC_ZERO = SACResult(0., 0., 0., 0., 0.)


class SACPhaseAgent:

    def __init__(self, phase_name, config=None):
        if config is None: config = DEFAULT_CONFIG
        self.config = config; self.phase_name = phase_name

        phase_cfg = config[f"{phase_name}_rl"]; cfg_sac = config["sac"]
        self.obs_dim    = int(phase_cfg["obs_dim"])
        self.action_dim = int(phase_cfg["action_dim"])
        self.gamma      = float(cfg_sac["gamma"]); self.tau = float(cfg_sac["tau"])
        self.batch_size = int(cfg_sac["batch_size"])
        self.warmup_steps     = int(cfg_sac["warmup_steps"])
        self.reward_scale     = float(cfg_sac.get("reward_scale", 1.0))
        self.update_interval  = int(cfg_sac.get("update_interval", 1))
        self.updates_per_step = int(cfg_sac.get("updates_per_step", 1))
        self.critic_grad_clip = float(cfg_sac["critic_grad_clip"])
        self.actor_grad_clip  = float(cfg_sac["actor_grad_clip"])

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
        ee_cfg = config.get("ee_control", {})
        if self.action_dim == 3:
            axy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            az  = float(phase_cfg.get("acc_max_z",  ee_cfg.get("acc_max_z",  3.0)))
            action_scale = [axy, axy, az]
        else:
            axy = float(phase_cfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
            action_scale = [axy, axy]

        self.actor = SACPhaseActor(self.obs_dim, self.action_dim, action_scale,
                                    hidden, n_layers).to(self.device)
        self.critic = SACPhaseCritic(self.obs_dim, self.action_dim, hidden, n_layers).to(self.device)
        self.target_critic = SACPhaseCritic(self.obs_dim, self.action_dim, hidden, n_layers).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict()); self.target_critic.eval()

        self.opt_actor  = torch.optim.Adam(self.actor.parameters(),  lr=float(cfg_sac["lr_actor"]))
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=float(cfg_sac["lr_critic"]))
        alpha_init = float(cfg_sac["alpha_init"])
        self.log_alpha = torch.tensor(np.log(alpha_init), dtype=torch.float32,
                                       device=self.device, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=float(cfg_sac["lr_alpha"]))
        self.auto_alpha = bool(cfg_sac["auto_alpha"])
        self.target_entropy = -float(cfg_sac["target_entropy_ratio"]) * self.action_dim

        # [HER] descent 专用 HERReplayBuffer, 其他阶段用 SimpleReplayBuffer
        buf_size = int(cfg_sac["buffer_size"])
        sbonus   = float(config.get("descent_rl", {}).get("reward", {}).get("success_bonus", 50.0))
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
        """添加 transition; achieved_pos (3,) 用于 HER (仅 descent)。"""
        self.buffer.add(norm_obs, action, norm_next_obs, reward, done, achieved_pos)

    def flush_episode_her(self, target_xy, target_z, obs_relabel_fn=None):
        """[HER] episode 结束时调用重标注 (仅 descent 生效)。"""
        if self._use_her:
            self.buffer.flush_episode_with_her(target_xy, target_z, obs_relabel_fn)

    def train_step(self):
        if not self.buffer.is_ready: return SAC_ZERO
        dev = self.device
        total_cl = total_al = total_al2 = total_q = 0.0
        for _ in range(self.updates_per_step):
            s, a, ns, r, d = self.buffer.sample(self.batch_size)
            si  = np_to_tensor(s,  dev); ai = np_to_tensor(a, dev)
            nsi = np_to_tensor(ns, dev); ri = np_to_tensor(r, dev) * self.reward_scale
            di  = np_to_tensor(d,  dev); alpha = self.log_alpha.exp().detach()

            with torch.no_grad():
                next_a, next_lp = self.actor.sample(nsi)
                tq1, tq2 = self.target_critic(nsi, next_a)
                target_q = ri + self.gamma*(1-di)*(torch.min(tq1,tq2) - alpha*next_lp.unsqueeze(-1))

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
                al2 = al_loss.item()

            soft_update(self.target_critic, self.critic, self.tau)
            total_cl+=critic_loss.item(); total_al+=actor_loss.item()
            total_al2+=al2; total_q+=q_min.mean().item()

        n = self.updates_per_step
        result = SACResult(total_cl/n, total_al/n, total_al2/n, self.alpha, total_q/n)
        self._last_result = result; return result

    def save(self, path):
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(), "opt_alpha": self.opt_alpha.state_dict(),
            "total_steps": self.total_steps, "obs_norm": self.obs_norm.state_dict(),
            "phase_name": self.phase_name,
        }, path)

    def load(self, path, map_location=None):
        ck = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"]); self.critic.load_state_dict(ck["critic"])
        self.target_critic.load_state_dict(ck["target_critic"])
        if "opt_actor"  in ck: self.opt_actor.load_state_dict(ck["opt_actor"])
        if "opt_critic" in ck: self.opt_critic.load_state_dict(ck["opt_critic"])
        if "log_alpha"  in ck: self.log_alpha.data.copy_(ck["log_alpha"].to(self.device))
        if "opt_alpha"  in ck: self.opt_alpha.load_state_dict(ck["opt_alpha"])
        self.total_steps = ck.get("total_steps", 0)
        if "obs_norm" in ck: self.obs_norm.load_state_dict(ck["obs_norm"])