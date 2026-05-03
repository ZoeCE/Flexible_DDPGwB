# ==============================================================================
# train_phase.py — 三阶段独立训练框架 v2
#
# v2 变更:
#   - 三阶段独立物理初始化 (不依赖专家 rollout)
#   - BC 预训练仅用于 descent 阶段
#   - 静默环境噪声输出
#   - 优化终端日志格式
# ==============================================================================

import os
import sys
import csv
import copy
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, deque

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from phase_agent import (
    PPOPhaseAgent, SACPhaseAgent,
    build_lift_obs, build_cruise_obs, build_descent_obs,
    PPO_ZERO, SAC_ZERO,
    OBS_EE_X, OBS_EE_Y, OBS_EE_VX, OBS_EE_VY,
    OBS_PL_X, OBS_PL_Y, OBS_PL_VX, OBS_PL_VY,
)
from phase_reward import (
    compute_lift_reward, compute_cruise_reward, compute_descent_reward,
    LiftRewardState, CruiseRewardState, DescentRewardState,
)
from ee_acc_controller import EEAccController, CruiseZYawPID

import mujoco


# ==============================================================================
# 工具
# ==============================================================================

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpisodeStats:
    def __init__(self, window=20):
        self.window = window
        self._data = {}
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._data: self._data[k] = []
            self._data[k].append(float(v))
            if len(self._data[k]) > self.window: self._data[k].pop(0)
    def mean(self, key):
        vals = self._data.get(key, [])
        return float(np.mean(vals)) if vals else 0.0
    def success_rate(self):
        return self.mean("success")


class Logger:
    def __init__(self, log_dir, project="phase_rl", run_name=None):
        self._wandb = None
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        try:
            import wandb
            self._wandb = wandb
            self._wandb.init(project=project, name=run_name or os.path.basename(log_dir),
                              dir=log_dir, config={}, resume="allow")
        except Exception: pass
    def update_config(self, cfg):
        if self._wandb:
            flat = {}
            def _f(d, pre=""):
                for k, v in d.items():
                    key = f"{pre}{k}"
                    if isinstance(v, dict): _f(v, key+"/")
                    else: flat[key] = v
            _f(cfg)
            self._wandb.config.update(flat, allow_val_change=True)
    def log(self, step, metrics):
        if self._wandb: self._wandb.log(metrics, step=step)
    def close(self):
        if self._wandb: self._wandb.finish()


def save_checkpoint(agent, log_dir, episode, tag=""):
    fname = f"ckpt_{tag}.pt" if tag else f"ckpt_ep{episode}.pt"
    path = os.path.join(log_dir, fname)
    agent.save(path)
    agent.save(os.path.join(log_dir, "ckpt_latest.pt"))
    return path


# ==============================================================================
# 观测 / Reward 调度
# ==============================================================================

REWARD_FNS = {
    "lift": compute_lift_reward,
    "cruise": compute_cruise_reward,
    "descent": compute_descent_reward,
}
REWARD_STATES = {
    "lift": LiftRewardState,
    "cruise": CruiseRewardState,
    "descent": DescentRewardState,
}

def build_phase_obs(phase, env_obs, env, start_xy, target_xy, prev_tilt, prev_yaw):
    if phase == "lift":
        return build_lift_obs(env_obs, env, start_xy, prev_tilt, prev_yaw)
    elif phase == "cruise":
        return build_cruise_obs(env_obs, env, target_xy, prev_tilt, prev_yaw)
    elif phase == "descent":
        return build_descent_obs(env_obs, env, target_xy, prev_tilt, prev_yaw)
    raise ValueError(f"Unknown phase: {phase}")


# ==============================================================================
# 三阶段独立物理初始化
# ==============================================================================

def _sync_env_internal_state(env):
    """
    在直接修改物理状态后, 同步 env 内部所有跟踪变量。
    必须在 mj_forward 之后调用。
    """
    # 步数计数器
    env.current_step = 0
    env.current_wp_idx = 0
    env.reached_final = False
    env.last_dist = None
    env.last_wp_idx = -1
    env._wp_just_advanced = False
    env._termination_reason = None

    # 关节角跟踪
    env._prev_q = env.data.qpos[:7].copy().astype(np.float32)
    env._prev_delta_q = np.zeros(env.action_dim, dtype=np.float32)

    # 插入阶段状态机
    env._insertion_hold_counter = 0
    env._in_insertion_phase = False
    env._best_insertion_z = 10.0

    # 势能差分奖励状态
    env._prev_goal_potential = None
    env._prev_phi_z = None
    env._prev_descent_depth = 0.0
    env._prev_ref_dist = None

    # EE 速度缓存
    env._prev_ee_pos = env._get_ee_pos().copy()
    env._ee_vel_cache = np.zeros(3)
    mat = env._get_ee_mat()
    from scipy.spatial.transform import Rotation as R
    env._prev_ee_euler = R.from_matrix(mat).as_euler('xyz').copy()
    env._ee_euler_vel_cache = np.zeros(3)

    # 动作队列
    init_q = env.data.qpos[:7].copy().astype(np.float32)
    env.action_queue.clear()
    for _ in range(max(1, env.latency_steps + 1)):
        env.action_queue.append(init_q.copy())


def reset_for_phase(env, phase, config):
    """
    为指定阶段直接初始化物理状态。不依赖专家 rollout。

      - Lift:    payload 在 start_xy, z=0.1 (标准 env.reset)
      - Cruise:  payload 在 start_xy, z=z_cruise, 直接设定
      - Descent: payload 在 target_xy, z=z_cruise, 直接设定
    """
    # 静默 A*/env 输出
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        obs = env.reset()
        planned_path = env.get_planned_path()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout

    if planned_path is None:
        return None, None

    phase_cfg = config.get(f"{phase}_rl", {})
    rng = np.random.default_rng()

    if phase == "lift":
        pass  # env.reset() 已经正确初始化

    elif phase == "cruise":
        z_cruise = float(config["planning"]["payload_z_cruise"])
        start_xy = env.default_start_xy.copy()
        rope_L = float(config["controller"].get("L", 0.5))
        ee_z = z_cruise + rope_L
        seed_q = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_q = env.ik_solver.solve_4d(seed_q, float(start_xy[0]), float(start_xy[1]), ee_z, 0.0)
        if init_q is None or np.any(np.isnan(init_q)): init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx = env.model.jnt_dofadr[pref_jnt]
        noise_xy = rng.uniform(-phase_cfg.get("init_xy_range", 0.02), phase_cfg.get("init_xy_range", 0.02), 2)
        env.data.qpos[qpos_addr]   = start_xy[0] + noise_xy[0]
        env.data.qpos[qpos_addr+1] = start_xy[1] + noise_xy[1]
        env.data.qpos[qpos_addr+2] = z_cruise
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        # 稳定化 warmup: 保持 arm 静止, 让绳索张紧自然
        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        for _ in range(30):
            env.data.qpos[:7] = init_q; env.data.qvel[:7] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer:
                env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        noise_vel = rng.uniform(-phase_cfg.get("init_vel_range", 0.02), phase_cfg.get("init_vel_range", 0.02), 2)
        env.data.qvel[dof_idx:dof_idx+2] += noise_vel
        mujoco.mj_forward(env.model, env.data)

        _sync_env_internal_state(env)
        obs = env._get_obs()

    elif phase == "descent":
        z_cruise = float(config["planning"]["payload_z_cruise"])
        target_xy = env.target_pos.copy()
        rope_L = float(config["controller"].get("L", 0.5))
        ee_z = z_cruise + rope_L
        seed_q = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_q = env.ik_solver.solve_4d(seed_q, float(target_xy[0]), float(target_xy[1]), ee_z, 0.0)
        if init_q is None or np.any(np.isnan(init_q)): init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx = env.model.jnt_dofadr[pref_jnt]
        noise_xy = rng.uniform(-phase_cfg.get("init_xy_range", 0.015), phase_cfg.get("init_xy_range", 0.015), 2)
        env.data.qpos[qpos_addr]   = target_xy[0] + noise_xy[0]
        env.data.qpos[qpos_addr+1] = target_xy[1] + noise_xy[1]
        env.data.qpos[qpos_addr+2] = z_cruise
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        # 稳定化 warmup: 保持 arm 静止, 让绳索张紧自然
        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        for _ in range(30):
            env.data.qpos[:7] = init_q; env.data.qvel[:7] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer:
                env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        noise_vel = rng.uniform(-phase_cfg.get("init_vel_range", 0.015), phase_cfg.get("init_vel_range", 0.015), 3)
        env.data.qvel[dof_idx:dof_idx+3] += noise_vel
        mujoco.mj_forward(env.model, env.data)

        _sync_env_internal_state(env)
        obs = env._get_obs()

    return obs, planned_path


# ==============================================================================
# 专家 BC 标签 (仅 descent)
# ==============================================================================

def collect_expert_acc(expert, env, obs, current_q, phase, config):
    _ = expert.compute_joint_target(obs, current_q)
    target_vel = expert._ee_vel.copy()
    dt = float(config.get("ee_control", {}).get("integrator_dt", 0.1))
    real_vel = env._ee_vel_cache.copy()
    acc = (target_vel - real_vel) / max(dt, 1e-6)
    if phase == "cruise": acc = acc[:2]
    else: acc = acc[:3]
    ee_cfg = config.get("ee_control", {})
    pcfg = config.get(f"{phase}_rl", {})
    axy = float(pcfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
    az = float(pcfg.get("acc_max_z", ee_cfg.get("acc_max_z", 3.0)))
    if len(acc) >= 2: acc[0] = np.clip(acc[0], -axy, axy); acc[1] = np.clip(acc[1], -axy, axy)
    if len(acc) >= 3: acc[2] = np.clip(acc[2], -az, az)
    return acc.astype(np.float32)

def _advance_expert_to_nearest_wp(expert, planned_path, pl_pos):
    if planned_path is None or len(planned_path) == 0: return
    dists = [np.linalg.norm(pl_pos - wp) for wp in planned_path]
    expert.tracker.current_idx = int(np.argmin(dists))


# ==============================================================================
# 课程学习
# ==============================================================================

class CurriculumManager:
    def __init__(self, config, phase):
        cur = config.get("curriculum", {})
        self.enabled = cur.get("enabled", False)
        self.wind_start = float(cur.get("wind_start_frac", 0.0))
        self.wind_end = float(cur.get("wind_end_frac", 1.0))
        self.wind_anneal = int(cur.get("wind_anneal_steps", 500_000))
        self.use_obs_cur = (phase == "cruise" and cur.get("obstacle_enabled", False))
        self.current_n_obs = int(cur.get("obstacle_start_n", 0))
        self.obs_max = int(cur.get("obstacle_max_n", 3))
        self.perf_window = int(cur.get("perf_window", 30))
        self.perf_sr = float(cur.get("perf_sr_threshold", 0.5))
        self.perf_rwd = float(cur.get("perf_reward_threshold", 10.0))
        self.perf_min = int(cur.get("perf_min_episodes_per_level", 100))
        self.perf_cap = int(cur.get("perf_hard_cap_steps", 600_000))
        self.lvl_step = 0; self.lvl_ep = 0
        self.stats = EpisodeStats(window=self.perf_window)

    def wind_frac(self, t):
        if not self.enabled: return 0.0  # 关闭时无风
        if self.wind_anneal <= 0: return self.wind_end
        f = min(t / self.wind_anneal, 1.0)
        return self.wind_start + f * (self.wind_end - self.wind_start)

    def update_obs(self, env, ep, t, r, s):
        if not self.use_obs_cur: return False
        self.stats.update(reward=r, success=float(s))
        if self.current_n_obs >= self.obs_max: return False
        if ((t - self.lvl_step >= self.perf_cap) or
            (ep - self.lvl_ep >= self.perf_min and
             self.stats.success_rate() >= self.perf_sr and
             self.stats.mean("reward") >= self.perf_rwd)):
            self.current_n_obs = min(self.current_n_obs + 1, self.obs_max)
            env.set_curriculum_n_obstacles(self.current_n_obs)
            self.lvl_ep = ep; self.lvl_step = t
            self.stats = EpisodeStats(window=self.perf_window)
            return True
        return False


# ==============================================================================
# PPO 训练
# ==============================================================================

def train_ppo(phase, log_dir, config, bc_ckpt=None):
    T = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W = int(config["train"]["log_smooth_win"])

    print(f"\n{'='*60}\n  PPO | {phase.upper()} | {T} steps | {log_dir}\n{'='*60}\n")

    env = CableRobotEnvWithObstacles(config=config)
    agent = PPOPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    ectl = EEAccController(config, env.ik_solver)
    # ★ Cruise 段 PID 控制器
    z_pid = CruiseZYawPID(config) if phase == "cruise" else None
    cur = CurriculumManager(config, phase)
    if cur.use_obs_cur: env.set_curriculum_n_obstacles(cur.current_n_obs)
    if bc_ckpt and os.path.exists(bc_ckpt): agent.load(bc_ckpt); print(f"  Loaded: {bc_ckpt}")

    logger = Logger(log_dir, f"{phase}_ppo", f"{phase}_ppo"); logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time()

    lf = os.path.join(log_dir, f"{phase}_ppo_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode","total_steps","ep_reward","avg_reward","sr","steps","pl","vl","kl","wind"])

    while ts < T:
        wf = cur.wind_frac(ts); env.set_wind_curriculum(wf)
        obs, pp = reset_for_phase(env, phase, config)
        if obs is None: continue
        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp)
            if phase != "lift": _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())
        ectl.reset(env._get_ee_pos(), cq)
        # ★ 初始化 cruise PID
        if z_pid is not None:
            from scipy.spatial.transform import Rotation as _Rpid
            _pl_z = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(_Rpid.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)
        sxy = env.default_start_xy.copy(); txy = env.target_pos.copy()
        pt, py = 0.0, 0.0; rs = REWARD_STATES[phase]()
        # ★ 修复 bug: PPO 路径也需要同步 total_steps_global
        if phase == "cruise" and hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        # ★ descent 阶段同步渐进容差用的全局步数
        if phase == "descent" and hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        er = 0.0; es = 0; suc = False; rd = False
        term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])

        while not rd:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt, py = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
            no = agent.normalize_obs(po, update=True)
            act, lp, val = agent.act(no)
            bct = np.zeros(agent.action_dim, np.float32)
            if phase == "descent":
                try: bct = collect_expert_acc(expert, env, obs, cq, phase, config)
                except: pass
            ree = env._get_ee_pos()
            if phase == "cruise" and z_pid is not None:
                # ★ 读取 payload 物理状态
                _pl_pos = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz = float(env.data.qvel[_dof_idx + 2])
                _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                from scipy.spatial.transform import Rotation as _Rpid
                _pl_euler = _Rpid.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) if _dof_idx + 5 < len(env.data.qvel) else 0.0
                # ★ PID 计算
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                # ★ 坠落检测 → 提前终止
                if _falling:
                    rw = -5.0; done = True
                    agent.buffer.add(no, act, bct, rw, 1.0, val, lp)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts
                    rd = True; break
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr,
                    target_yaw=_tgt_yaw)
            elif phase == "cruise":
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True, z_lock_height=float(config["cruise_rl"]["z_lock_height"]))
            else:
                dq = ectl.compute_delta_q(act, cq, ree)
            no2, _, _, _, ei = env.step(dq)
            rw, dn, sc, ri = REWARD_FNS[phase](env, no2, config, rs)
            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1: done = True; ri.setdefault("termination", "timeout")
            if sc: suc = True
            # ★ 记录终止原因
            if done and ri.get("termination"):
                term_reason = ri["termination"]
            agent.buffer.add(no, act, bct, rw, float(done), val, lp)
            er += rw; es += 1; ts += 1; agent.total_steps = ts; obs = no2
            if agent.buffer.full:
                if done: lv = 0.0
                else:
                    ns = build_phase_obs(phase, no2, env, sxy, txy, pt, py)[0]
                    nn = agent.normalize_obs(ns, update=False)
                    with torch.no_grad(): lv = agent.critic(torch.tensor(nn, dtype=torch.float32, device=agent.device).unsqueeze(0)).item()
                agent.buffer.compute_returns_and_advantages(lv, agent.gamma, agent.gae_lambda)
                agent.update(); rd = True
            if done: rd = True

        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()
        up = cur.update_obs(env, ep, ts, er, suc)
        r = agent._last_result
        m = "✅" if suc else "❌"
        o = f" obs→{cur.current_n_obs}" if up else ""
        # ★ 改善 terminal 输出: 加入终止原因、obs_norm 状态、log_std
        log_std_val = agent.actor.log_std.mean().item()
        norm_n = agent.obs_norm.n
        print(f"Ep{ep:4d} [{ts:7d}] {m} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} | PL:{r.policy_loss:6.3f} VL:{r.value_loss:6.3f} "
              f"KL:{r.approx_kl:.3f} CF:{r.clip_fraction:.2f} | "
              f"logstd:{log_std_val:.2f} norm_n:{norm_n} w:{wf:.2f}{o} | {term_reason}")
        logger.log(ep, {f"{phase}/reward":er, f"{phase}/avg_reward":ar, f"{phase}/sr":sr, f"{phase}/steps":es, f"{phase}/wind":wf, f"ppo/pl":r.policy_loss, f"ppo/vl":r.value_loss, f"ppo/ent":r.entropy_loss, f"ppo/kl":r.approx_kl, f"ppo/cf":r.clip_fraction})
        with open(lf, "a", newline="") as f: csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es, f"{r.policy_loss:.5f}", f"{r.value_loss:.5f}", f"{r.approx_kl:.5f}", f"{wf:.3f}"])
        if ep > 0 and ep % SI == 0: save_checkpoint(agent, log_dir, ep)
        if ep > 0 and ep % EI == 0 and sr > best: best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-PPO] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, best_sr={best*100:.0f}%")
    logger.close(); env.close(); return agent


# ==============================================================================
# SAC 训练
# ==============================================================================

def train_sac(phase, log_dir, config, bc_ckpt=None):
    T = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W = int(config["train"]["log_smooth_win"])
    WU = int(config["sac"].get("warmup_steps", 5000))

    print(f"\n{'='*60}\n  SAC | {phase.upper()} | {T} steps | warmup={WU} | {log_dir}\n{'='*60}\n")

    env = CableRobotEnvWithObstacles(config=config)
    agent = SACPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    wex = JointSpaceExpert(config, env.ik_solver)
    ectl = EEAccController(config, env.ik_solver)
    # ★ Cruise 段 PID 控制器
    z_pid = CruiseZYawPID(config) if phase == "cruise" else None
    cur = CurriculumManager(config, phase)
    if cur.use_obs_cur: env.set_curriculum_n_obstacles(cur.current_n_obs)
    if bc_ckpt and os.path.exists(bc_ckpt): agent.load(bc_ckpt); print(f"  Loaded: {bc_ckpt}")

    logger = Logger(log_dir, f"{phase}_sac", f"{phase}_sac"); logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time(); onf = False

    lf = os.path.join(log_dir, f"{phase}_sac_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode","total_steps","ep_reward","avg_reward","sr","steps","cl","al","alpha","wind"])

    while ts < T:
        wf = cur.wind_frac(ts); env.set_wind_curriculum(wf)
        obs, pp = reset_for_phase(env, phase, config)
        if obs is None: continue
        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env); wex.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp); wex.set_path(pp)
            if phase != "lift":
                plp = env.data.body('prefab').xpos.copy()
                _advance_expert_to_nearest_wp(expert, pp, plp)
                _advance_expert_to_nearest_wp(wex, pp, plp)
        ectl.reset(env._get_ee_pos(), cq)
        # ★ 初始化 cruise PID
        if z_pid is not None:
            from scipy.spatial.transform import Rotation as _Rpid
            _pl_z = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(_Rpid.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)
        sxy = env.default_start_xy.copy(); txy = env.target_pos.copy()
        pt, py = 0.0, 0.0; rs = REWARD_STATES[phase]()
        # ★ 修复 bug: SAC 路径也要同步 total_steps_global
        if phase == "cruise" and hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        if phase == "descent" and hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        er = 0.0; es = 0; suc = False
        term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])

        while True:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt, py = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
            no = agent.normalize_obs(po, update=not onf)
            if ts < WU:
                try: act = collect_expert_acc(wex, env, obs, cq, phase, config); act += np.random.normal(0, 0.1, len(act)).astype(np.float32)
                except: act = np.zeros(agent.action_dim, np.float32)
            else: act = agent.act(no, deterministic=False)
            if ts == WU and not onf: agent._freeze_obs_norm = True; onf = True
            ree = env._get_ee_pos()
            if phase == "cruise" and z_pid is not None:
                _pl_pos = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz = float(env.data.qvel[_dof_idx + 2])
                _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                from scipy.spatial.transform import Rotation as _Rpid
                _pl_euler = _Rpid.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                if _falling:
                    rw = -5.0; done = True
                    npo, _, _ = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
                    nn = agent.normalize_obs(npo, update=False)
                    agent.remember(no, act, nn, rw, 1.0)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts
                    break
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr,
                    target_yaw=_tgt_yaw)
            elif phase == "cruise":
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True, z_lock_height=float(config["cruise_rl"]["z_lock_height"]))
            else: dq = ectl.compute_delta_q(act, cq, ree)
            no2, _, _, _, ei = env.step(dq)
            rw, dn, sc, ri = REWARD_FNS[phase](env, no2, config, rs)
            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1: done = True; ri.setdefault("termination", "timeout")
            if sc: suc = True
            if done and ri.get("termination"):
                term_reason = ri["termination"]
            npo, _, _ = build_phase_obs(phase, no2, env, sxy, txy, pt, py)
            nn = agent.normalize_obs(npo, update=False)
            agent.remember(no, act, nn, rw, float(done))
            if ts >= WU and ts % agent.update_interval == 0: agent.train_step()
            er += rw; es += 1; ts += 1; agent.total_steps = ts; obs = no2
            if done: break

        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()
        up = cur.update_obs(env, ep, ts, er, suc)
        r = agent._last_result
        m = "✅" if suc else "❌"
        o = f" obs→{cur.current_n_obs}" if up else ""
        norm_n = agent.obs_norm.n
        print(f"Ep{ep:4d} [{ts:7d}] {m} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} | CL:{r.critic_loss:6.3f} AL:{r.actor_loss:6.3f} "
              f"α:{agent.alpha:.3f} | norm_n:{norm_n} w:{wf:.2f}{o} | {term_reason}")
        logger.log(ep, {f"{phase}/reward":er, f"{phase}/avg_reward":ar, f"{phase}/sr":sr, f"{phase}/steps":es, f"sac/alpha":agent.alpha, f"sac/cl":r.critic_loss, f"sac/al":r.actor_loss})
        with open(lf, "a", newline="") as f: csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es, f"{r.critic_loss:.5f}", f"{r.actor_loss:.5f}", f"{agent.alpha:.5f}", f"{wf:.3f}"])
        if ep > 0 and ep % SI == 0: save_checkpoint(agent, log_dir, ep)
        if ep > 0 and ep % EI == 0 and sr > best: best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-SAC] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, best_sr={best*100:.0f}%")
    logger.close(); env.close(); return agent


# ==============================================================================
# ==============================================================================
# BC 预训练 (cruise + descent)
# ==============================================================================

def pretrain_bc_cruise(agent, config):
    """
    巡航段 BC 预训练 v3:
      - 大幅增加数据量 (n_episodes 从 300→1000)
      - Epsilon 退火: 训练前期加入动作噪声模拟 DAgger, 提高鲁棒性
      - 独立 eval: 每 eval_interval epoch 在 held-out 数据上评估, patience 早停
      - 详细 terminal 输出: 显示 train/eval loss、BC action 误差
    """
    bc = config.get("bc_pretrain", {})
    if not bc.get("enabled", True):
        return

    nep        = int(bc.get("n_episodes", 1000))
    nepoch     = int(bc.get("n_epochs", 200))
    lr         = float(bc.get("lr", 3e-4))
    lr_decay   = float(bc.get("lr_decay", 0.5))
    lr_decay_interval = int(bc.get("lr_decay_interval", 50))
    bs         = int(bc.get("batch_size", 512))
    eval_interval  = int(bc.get("eval_interval", 20))
    eval_episodes  = int(bc.get("eval_episodes", 30))
    patience       = int(bc.get("patience", 5))
    loss_thresh    = float(bc.get("loss_threshold", 0.02))
    eps_start  = float(bc.get("epsilon_start", 0.3))
    eps_end    = float(bc.get("epsilon_end", 0.0))
    acc_max    = float(config.get("ee_control", {}).get("acc_max_xy", 0.5))

    print(f"\n{'='*60}")
    print(f"  [BC-cruise] 开始预训练")
    print(f"  目标: {nep} episodes, {nepoch} epochs, batch={bs}")
    print(f"  epsilon 退火: {eps_start:.1f} → {eps_end:.1f}")
    print(f"  早停: patience={patience}, loss_thresh={loss_thresh:.3f}")
    print(f"{'='*60}")

    env = CableRobotEnvWithObstacles(config=config)
    expert = JointSpaceExpert(config, env.ik_solver)

    # ── 收集训练数据 ──
    aobs_train = []; aact_train = []
    aobs_eval  = []; aact_eval  = []
    ve = 0; att = 0; succ = 0; total_steps_collected = 0

    while ve < nep and att < nep * 3:
        att += 1
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs, pp = reset_for_phase(env, "cruise", config)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        if obs is None:
            continue

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp)
            _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())

        ve += 1
        is_eval_ep = (ve % max(nep // eval_episodes, 1) == 0)
        txy = env.target_pos.copy()
        pt2, py2 = 0.0, 0.0
        mx = int(config["cruise_rl"]["max_steps"])
        ep_steps = 0

        for s in range(mx):
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt2, py2 = build_cruise_obs(obs, env, txy, pt2, py2)
            no = agent.normalize_obs(po, update=True)

            # ── BC 加速度标签 (P-D 控制器) ──
            pl_xy  = np.array([obs[OBS_PL_X], obs[OBS_PL_Y]])
            pl_vxy = np.array([obs[OBS_PL_VX], obs[OBS_PL_VY]])
            ee_vxy = np.array([obs[OBS_EE_VX], obs[OBS_EE_VY]])
            diff = txy - pl_xy
            dist = float(np.linalg.norm(diff))

            if dist > 0.005:
                direction = diff / dist
                vel_proj  = float(np.dot(ee_vxy, direction))
                kp = 0.8; kd = 0.5
                acc_mag = kp * min(dist, 0.15) / 0.15 - kd * vel_proj
                acc_mag = np.clip(acc_mag, -1.0, 1.0)
                acc_label = direction * acc_mag * acc_max
            else:
                acc_label = -ee_vxy * 2.0
            acc_label = np.clip(acc_label, -acc_max, acc_max).astype(np.float32)

            if is_eval_ep:
                aobs_eval.append(no.copy())
                aact_eval.append(acc_label.copy())
            else:
                aobs_train.append(no.copy())
                aact_train.append(acc_label.copy())

            # 专家执行
            dq = expert.compute_delta_q_target(obs, cq)
            obs, _, t, tr, _ = env.step(dq)
            ep_steps += 1
            total_steps_collected += 1
            if t or tr:
                break

        if ep_steps > 30:
            succ += 1

        if ve % 100 == 0 or ve == nep:
            print(f"  [BC-cruise] 收集进度: {ve}/{nep} eps | "
                  f"train={len(aobs_train)} eval={len(aobs_eval)} steps | "
                  f"成功率={succ/max(ve,1)*100:.0f}%")

    env.close()

    if len(aobs_train) < 500:
        print(f"  [BC-cruise] ⚠ 样本不足 ({len(aobs_train)}), 跳过 BC")
        return

    print(f"\n  [BC-cruise] 数据收集完成:")
    print(f"    训练集: {len(aobs_train)} samples ({ve} episodes, {succ} valid)")
    print(f"    验证集: {len(aobs_eval)} samples")
    print(f"    开始 {nepoch} epochs 训练...")

    # ── 转 tensor ──
    ot = torch.tensor(np.array(aobs_train), device=agent.device)
    at = torch.tensor(np.array(aact_train), device=agent.device)
    oe = torch.tensor(np.array(aobs_eval),  device=agent.device) if aobs_eval else None
    ae = torch.tensor(np.array(aact_eval),  device=agent.device) if aobs_eval else None
    ns = len(aobs_train)

    # 只训练 mean head
    bp = [p for n, p in agent.actor.named_parameters() if 'log_std' not in n]
    opt = torch.optim.AdamW(bp, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_interval, gamma=lr_decay)

    best_eval_loss = float('inf')
    best_state = None
    patience_count = 0

    for e in range(nepoch):
        # epsilon 退火 (动作层面加噪声 → 模拟 DAgger, 提高鲁棒性)
        eps = eps_start + (eps_end - eps_start) * min(e / max(nepoch - 1, 1), 1.0)

        # ── 训练 ──
        agent.actor.train()
        idx = np.random.permutation(ns)
        tl = 0.0; tl_a = 0.0; nb = 0
        for st in range(0, ns, bs):
            i = idx[st:st+bs]
            obs_b = ot[i]
            act_b = ae_b = at[i]
            if eps > 0:
                # epsilon 退火: 以 eps 概率用随机动作替换 BC 标签
                mask = torch.rand(len(i), device=agent.device) < eps
                noise = torch.rand_like(act_b) * 2 - 1  # U(-1,1) * acc_max
                act_b = torch.where(mask.unsqueeze(1), noise * acc_max, act_b)
            loss, loss_a, _ = agent.actor.bc_forward(obs_b, act_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            opt.step()
            tl += loss.item(); tl_a += loss_a.item(); nb += 1
        scheduler.step()

        train_loss = tl / max(nb, 1)
        train_loss_a = tl_a / max(nb, 1)

        # ── 独立 Eval ──
        if (e + 1) % eval_interval == 0 or e == nepoch - 1:
            agent.actor.eval()
            with torch.no_grad():
                if oe is not None and len(oe) > 0:
                    el, el_a, _ = agent.actor.bc_forward(oe, ae)
                    eval_loss = el.item(); eval_loss_a = el_a.item()
                else:
                    eval_loss = train_loss; eval_loss_a = train_loss_a

            cur_lr = opt.param_groups[0]['lr']
            print(f"  Epoch {e+1:3d}/{nepoch} | "
                  f"train_loss(u)={train_loss:.5f} act={train_loss_a:.5f} | "
                  f"eval_loss(u)={eval_loss:.5f} act={eval_loss_a:.5f} | "
                  f"eps={eps:.3f} lr={cur_lr:.2e} | "
                  f"patience={patience_count}/{patience}")

            # 早停逻辑
            if eval_loss < best_eval_loss - 1e-5:
                best_eval_loss = eval_loss
                best_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                print(f"  [BC-cruise] 早停: eval loss 连续 {patience} 次未提升")
                break

            if eval_loss < loss_thresh:
                print(f"  [BC-cruise] 目标达成: eval_loss={eval_loss:.5f} < {loss_thresh}")
                break
        elif (e + 1) % 10 == 0:
            cur_lr = opt.param_groups[0]['lr']
            print(f"  Epoch {e+1:3d}/{nepoch} | train_loss={train_loss:.5f} | eps={eps:.3f} lr={cur_lr:.2e}")

    # 恢复最佳权重
    if best_state is not None:
        agent.actor.load_state_dict(best_state)
        print(f"  [BC-cruise] ✅ 恢复最佳权重 (eval_loss={best_eval_loss:.5f})")

    print(f"  [BC-cruise] 完成\n")


def pretrain_bc_descent(agent, config):
    """
    下降段 BC 预训练 v2:
      - 改进 BC 标签: 使用几何方法直接计算目标加速度方向, 替代噪声大的速度差分
      - 同样加入 epsilon 退火 + 独立 eval + 早停
      - 详细 terminal 输出
    """
    bc = config.get("bc_pretrain", {})
    if not bc.get("enabled", True):
        return

    nep        = int(bc.get("n_episodes", 1000))
    nepoch     = int(bc.get("n_epochs", 200))
    lr         = float(bc.get("lr", 3e-4))
    lr_decay   = float(bc.get("lr_decay", 0.5))
    lr_decay_interval = int(bc.get("lr_decay_interval", 50))
    bs         = int(bc.get("batch_size", 512))
    eval_interval  = int(bc.get("eval_interval", 20))
    eval_episodes  = int(bc.get("eval_episodes", 30))
    patience       = int(bc.get("patience", 5))
    loss_thresh    = float(bc.get("loss_threshold", 0.02))
    eps_start  = float(bc.get("epsilon_start", 0.3))
    eps_end    = float(bc.get("epsilon_end", 0.0))

    ee_cfg   = config.get("ee_control", {})
    dcfg     = config.get("descent_rl", {})
    acc_max_xy = float(dcfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 0.5)))
    acc_max_z  = float(dcfg.get("acc_max_z",  ee_cfg.get("acc_max_z", 1.0)))

    print(f"\n{'='*60}")
    print(f"  [BC-descent] 开始预训练")
    print(f"  目标: {nep} episodes, {nepoch} epochs, batch={bs}")
    print(f"  acc_max: xy={acc_max_xy}, z={acc_max_z}")
    print(f"{'='*60}")

    env = CableRobotEnvWithObstacles(config=config)
    expert = JointSpaceExpert(config, env.ik_solver)

    aobs_train = []; aact_train = []
    aobs_eval  = []; aact_eval  = []
    ve = 0; att = 0; succ = 0

    while ve < nep and att < nep * 3:
        att += 1
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs, pp = reset_for_phase(env, "descent", config)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
        if obs is None:
            continue

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp)
            _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())

        ve += 1
        is_eval_ep = (ve % max(nep // eval_episodes, 1) == 0)
        txy = env.target_pos.copy()
        target_pz = float(config["insertion"]["target_payload_z"])
        pt2, py2 = 0.0, 0.0
        mx = int(config["descent_rl"]["max_steps"])
        ep_steps = 0

        for s in range(mx):
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt2, py2 = build_descent_obs(obs, env, txy, pt2, py2)
            no = agent.normalize_obs(po, update=True)

            # ── 改进的 BC 标签: 几何 P-D 控制器, 避免速度差分噪声 ──
            pl_pos = env.data.body('prefab').xpos.copy()
            pl_xy  = pl_pos[:2]
            pl_z   = float(pl_pos[2])
            dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
            pl_vel  = env.data.qvel[dof_idx:dof_idx+3].copy()

            # XY 对准加速度
            diff_xy = txy - pl_xy
            dist_xy = float(np.linalg.norm(diff_xy))
            if dist_xy > 0.002:
                dir_xy   = diff_xy / dist_xy
                vel_proj = float(np.dot(pl_vel[:2], dir_xy))
                kp_xy = 0.6; kd_xy = 0.8
                acc_mag_xy = kp_xy * min(dist_xy, 0.05) / 0.05 - kd_xy * vel_proj
                acc_xy = dir_xy * np.clip(acc_mag_xy, -1.0, 1.0) * acc_max_xy
            else:
                acc_xy = -pl_vel[:2] * 1.5  # 已对准, 制动
            acc_xy = np.clip(acc_xy, -acc_max_xy, acc_max_xy)

            # Z 下降加速度 (只在 xy 对准时下降)
            align_factor = np.exp(-dist_xy / 0.01)  # 1cm 为特征距离
            z_error = pl_z - target_pz   # 正=还需下降
            kp_z = 0.5; kd_z = 0.3
            if z_error > 0.005 and align_factor > 0.3:
                acc_z = -(kp_z * min(z_error, 0.15) + kd_z * max(float(pl_vel[2]), 0)) * align_factor
            else:
                acc_z = -float(pl_vel[2]) * 1.0  # 制动
            acc_z = np.clip(acc_z, -acc_max_z, acc_max_z)

            acc_label = np.array([acc_xy[0], acc_xy[1], acc_z], dtype=np.float32)

            if is_eval_ep:
                aobs_eval.append(no.copy())
                aact_eval.append(acc_label.copy())
            else:
                aobs_train.append(no.copy())
                aact_train.append(acc_label.copy())

            # 专家执行
            dq = expert.compute_delta_q_target(obs, cq)
            obs, _, t, tr, _ = env.step(dq)
            ep_steps += 1
            if t or tr:
                break

        if ep_steps > 20:
            succ += 1

        if ve % 100 == 0 or ve == nep:
            print(f"  [BC-descent] 收集进度: {ve}/{nep} eps | "
                  f"train={len(aobs_train)} eval={len(aobs_eval)} | "
                  f"成功率={succ/max(ve,1)*100:.0f}%")

    env.close()

    if len(aobs_train) < 200:
        print(f"  [BC-descent] ⚠ 样本不足 ({len(aobs_train)}), 跳过 BC")
        return

    print(f"\n  [BC-descent] 数据收集完成: train={len(aobs_train)}, eval={len(aobs_eval)}")
    print(f"  开始 {nepoch} epochs 训练...")

    ot = torch.tensor(np.array(aobs_train), device=agent.device)
    at = torch.tensor(np.array(aact_train), device=agent.device)
    oe = torch.tensor(np.array(aobs_eval),  device=agent.device) if aobs_eval else None
    ae = torch.tensor(np.array(aact_eval),  device=agent.device) if aobs_eval else None
    ns = len(aobs_train)

    bp = [p for n, p in agent.actor.named_parameters() if 'log_std' not in n]
    opt = torch.optim.AdamW(bp, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_interval, gamma=lr_decay)

    best_eval_loss = float('inf')
    best_state = None
    patience_count = 0

    for e in range(nepoch):
        eps = eps_start + (eps_end - eps_start) * min(e / max(nepoch - 1, 1), 1.0)

        agent.actor.train()
        idx = np.random.permutation(ns)
        tl = 0.0; tl_a = 0.0; nb = 0
        for st in range(0, ns, bs):
            i = idx[st:st+bs]
            act_b = at[i]
            if eps > 0:
                mask = torch.rand(len(i), device=agent.device) < eps
                noise = torch.rand_like(act_b) * 2 - 1
                # 只对 xy 加噪
                noise[:, 2] *= 0.3
                act_b = torch.where(mask.unsqueeze(1), noise * acc_max_xy, act_b)
            loss, loss_a, _ = agent.actor.bc_forward(ot[i], act_b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            opt.step()
            tl += loss.item(); tl_a += loss_a.item(); nb += 1
        scheduler.step()

        train_loss = tl / max(nb, 1)
        train_loss_a = tl_a / max(nb, 1)

        if (e + 1) % eval_interval == 0 or e == nepoch - 1:
            agent.actor.eval()
            with torch.no_grad():
                if oe is not None and len(oe) > 0:
                    el, el_a, _ = agent.actor.bc_forward(oe, ae)
                    eval_loss = el.item(); eval_loss_a = el_a.item()
                else:
                    eval_loss = train_loss; eval_loss_a = train_loss_a

            cur_lr = opt.param_groups[0]['lr']
            print(f"  Epoch {e+1:3d}/{nepoch} | "
                  f"train={train_loss:.5f}/{train_loss_a:.5f} | "
                  f"eval={eval_loss:.5f}/{eval_loss_a:.5f} | "
                  f"eps={eps:.3f} lr={cur_lr:.2e} | patience={patience_count}/{patience}")

            if eval_loss < best_eval_loss - 1e-5:
                best_eval_loss = eval_loss
                best_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1

            if patience_count >= patience:
                print(f"  [BC-descent] 早停: patience={patience}")
                break
            if eval_loss < loss_thresh:
                print(f"  [BC-descent] 目标达成: eval_loss={eval_loss:.5f}")
                break
        elif (e + 1) % 10 == 0:
            cur_lr = opt.param_groups[0]['lr']
            print(f"  Epoch {e+1:3d}/{nepoch} | train_loss={train_loss:.5f} | eps={eps:.3f} lr={cur_lr:.2e}")

    if best_state is not None:
        agent.actor.load_state_dict(best_state)
        print(f"  [BC-descent] ✅ 恢复最佳权重 (eval_loss={best_eval_loss:.5f})")

    print(f"  [BC-descent] 完成\n")


# ==============================================================================
# 入口
# ==============================================================================

def train(phase, log_dir, algo="ppo", custom_config=None, bc_ckpt=None, skip_bc=False):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config:
        for k, v in custom_config.items():
            if isinstance(v, dict) and k in config: config[k].update(v)
            else: config[k] = v
    set_global_seed(config["train"].get("seed", 42))
    os.makedirs(log_dir, exist_ok=True)
    
    # ★ BC 预训练: cruise 和 descent 阶段都支持
    if phase in ("cruise", "descent") and not skip_bc and bc_ckpt is None:
        ag = PPOPhaseAgent(phase, config=config) if algo == "ppo" else SACPhaseAgent(phase, config=config)
        if phase == "cruise":
            pretrain_bc_cruise(ag, config)
        else:
            pretrain_bc_descent(ag, config)
        bp = os.path.join(log_dir, "ckpt_bc.pt"); ag.save(bp); bc_ckpt = bp
    
    if algo == "ppo": return train_ppo(phase, log_dir, config, bc_ckpt)
    elif algo == "sac": return train_sac(phase, log_dir, config, bc_ckpt)
    else: raise ValueError(f"Unknown algo: {algo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三阶段 RL 训练")
    parser.add_argument("--phase", type=str, required=True, choices=["lift", "cruise", "descent"])
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bc-ckpt", type=str, default=None)
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wind", action="store_true", help="启用风力")
    parser.add_argument("--curriculum", action="store_true", help="启用课程学习(风力+障碍物渐进)")
    parser.add_argument("--obstacles", type=int, default=None, help="固定障碍物数量")
    args = parser.parse_args()
    ld = args.log_dir or f"saves/{args.phase}_{args.algo}"
    cc = {}
    if args.render: cc.setdefault("sim", {})["render"] = True
    if args.gpu != 0: cc.setdefault("train", {})["gpu_id"] = args.gpu
    if args.timesteps: cc.setdefault("train", {})["total_timesteps"] = args.timesteps
    cc.setdefault("train", {})["seed"] = args.seed
    if args.wind: cc.setdefault("wind", {})["enabled"] = True
    if args.curriculum: cc.setdefault("curriculum", {})["enabled"] = True
    if args.obstacles is not None:
        cc.setdefault("curriculum", {})["obstacle_enabled"] = False
        cc.setdefault("scene", {})["n_obstacles"] = args.obstacles
    train(args.phase, ld, algo=args.algo, custom_config=cc or None, bc_ckpt=args.bc_ckpt, skip_bc=args.skip_bc)