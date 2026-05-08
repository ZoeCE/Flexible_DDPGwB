# ==============================================================================
# train_phase.py — 三阶段独立训练框架 v3
#
# v3 主要变更:
#
# [ARCH-CRUISE] Cruise 段架构: RL 直接输出完整 xy 加速度
#   - 移除 SwingDampingController 的 residual_mode 叠加
#   - RL acc 直接送入 EEAccController (非残差)
#   - SwingDampingController 保留仅作监控 (swing_energy 日志)
#   - CruiseZYawPID 保留 (Z/Yaw 仍由 PID 控制)
#
# [ARCH-DESCENT] Descent 段架构: PID base + RL residual delta_q
#   - JointSpaceExpert.compute_delta_q_target() 生成 base delta_q (PID)
#   - RL PPOPhaseAgent 生成 residual acc (3D), 转为 residual delta_q
#   - 最终 delta_q = pid_dq + clip(rl_dq, residual_dq_scale * |pid_dq|)
#   - 废弃 DescentDualRLAgent (use_dual_rl=False)
#
# [CUR] 课程学习:
#   Cruise: cruise_dist_curriculum=False (全程), obstacle_enabled=False (0 障碍物)
#   Descent: descent_init_xy_start=2mm, tol_mult=3.0
#
# [BC] BC 预训练:
#   cruise: n_epochs_cruise (默认 40)
#   descent: n_epochs (默认 20)
#
# [REW] 奖励: 使用 phase_reward.py v3 (无变更需要在此处处理)
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
    OBS_EE_Z, OBS_PL_Z, OBS_PL_VZ,
)
from phase_reward import (
    compute_lift_reward, compute_cruise_reward, compute_descent_reward,
    LiftRewardState, CruiseRewardState, DescentRewardState,
)
from ee_acc_controller import EEAccController, CruiseZYawPID, SwingDampingController, compute_swing_energy

import mujoco
from scipy.spatial.transform import Rotation as R


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
            if k not in self._data:
                self._data[k] = []
            self._data[k].append(float(v))
            if len(self._data[k]) > self.window:
                self._data[k].pop(0)

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
        except Exception:
            pass

    def update_config(self, cfg):
        if self._wandb:
            flat = {}
            def _f(d, pre=""):
                for k, v in d.items():
                    key = f"{pre}{k}"
                    if isinstance(v, dict):
                        _f(v, key + "/")
                    else:
                        flat[key] = v
            _f(cfg)
            self._wandb.config.update(flat, allow_val_change=True)

    def log(self, step, metrics):
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def close(self):
        if self._wandb:
            self._wandb.finish()


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
    "lift":    compute_lift_reward,
    "cruise":  compute_cruise_reward,
    "descent": compute_descent_reward,
}
REWARD_STATES = {
    "lift":    LiftRewardState,
    "cruise":  CruiseRewardState,
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
    env.current_step = 0
    env.current_wp_idx = 0
    env.reached_final = False
    env.last_dist = None
    env.last_wp_idx = -1
    env._wp_just_advanced = False
    env._termination_reason = None
    env._prev_q = env.data.qpos[:7].copy().astype(np.float32)
    env._prev_delta_q = np.zeros(env.action_dim, dtype=np.float32)
    env._insertion_hold_counter = 0
    env._in_insertion_phase = False
    env._best_insertion_z = 10.0
    env._prev_goal_potential = None
    env._prev_phi_z = None
    env._prev_descent_depth = 0.0
    env._prev_ref_dist = None
    env._prev_ee_pos = env._get_ee_pos().copy()
    env._ee_vel_cache = np.zeros(3)
    mat = env._get_ee_mat()
    env._prev_ee_euler = R.from_matrix(mat).as_euler('xyz').copy()
    env._ee_euler_vel_cache = np.zeros(3)
    init_q = env.data.qpos[:7].copy().astype(np.float32)
    env.action_queue.clear()
    for _ in range(max(1, env.latency_steps + 1)):
        env.action_queue.append(init_q.copy())


def reset_for_phase(env, phase, config, override_init_xy_range=None,
                    override_init_vel_range=None, override_init_tilt_range=None,
                    cruise_dist_frac=1.0):
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
        pass

    elif phase == "cruise":
        z_cruise = float(config["planning"]["payload_z_cruise"])
        start_xy = env.default_start_xy.copy()
        target_xy = env.target_pos.copy()

        # [v3] cruise_dist_curriculum=False 时 cruise_dist_frac=1.0, 全程训练
        if cruise_dist_frac < 1.0:
            lerp_start = target_xy + cruise_dist_frac * (start_xy - target_xy)
            noise_range = float(phase_cfg.get("init_xy_range", 0.01))
            lerp_start += rng.uniform(-noise_range, noise_range, 2)
            start_xy = lerp_start

        rope_L = float(config["controller"].get("L", 0.5))
        ee_z = z_cruise + rope_L
        seed_q = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_q = env.ik_solver.solve_4d(seed_q, float(start_xy[0]), float(start_xy[1]), ee_z, 0.0)
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx = env.model.jnt_dofadr[pref_jnt]

        xy_range = override_init_xy_range if override_init_xy_range is not None \
                   else float(phase_cfg.get("init_xy_range", 0.01))
        noise_xy = rng.uniform(-xy_range, xy_range, 2)
        env.data.qpos[qpos_addr]   = start_xy[0] + noise_xy[0]
        env.data.qpos[qpos_addr+1] = start_xy[1] + noise_xy[1]
        env.data.qpos[qpos_addr+2] = z_cruise
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        prefab_hold_xy = env.data.qpos[qpos_addr:qpos_addr+2].copy()
        for _ in range(60):
            env.data.qpos[:7] = init_q
            env.data.qvel[:7] = 0.0
            env.data.qpos[qpos_addr]   = prefab_hold_xy[0]
            env.data.qpos[qpos_addr+1] = prefab_hold_xy[1]
            env.data.qpos[qpos_addr+2] = z_cruise
            env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            env.data.qvel[dof_idx:dof_idx+6] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer:
                env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        vel_range = override_init_vel_range if override_init_vel_range is not None \
                    else float(phase_cfg.get("init_vel_range", 0.0))
        if vel_range > 0:
            noise_vel = rng.uniform(-vel_range, vel_range, 2)
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
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx = env.model.jnt_dofadr[pref_jnt]

        xy_range = override_init_xy_range if override_init_xy_range is not None \
                   else float(phase_cfg.get("init_xy_range", 0.010))
        noise_xy = rng.uniform(-xy_range, xy_range, 2)
        env.data.qpos[qpos_addr]   = target_xy[0] + noise_xy[0]
        env.data.qpos[qpos_addr+1] = target_xy[1] + noise_xy[1]
        env.data.qpos[qpos_addr+2] = z_cruise
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        prefab_hold_xy2 = env.data.qpos[qpos_addr:qpos_addr+2].copy()
        for _ in range(60):
            env.data.qpos[:7] = init_q
            env.data.qvel[:7] = 0.0
            env.data.qpos[qpos_addr]   = prefab_hold_xy2[0]
            env.data.qpos[qpos_addr+1] = prefab_hold_xy2[1]
            env.data.qpos[qpos_addr+2] = z_cruise
            env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            env.data.qvel[dof_idx:dof_idx+6] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer:
                env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        # 加噪声
        vel_range = override_init_vel_range if override_init_vel_range is not None \
                    else float(phase_cfg.get("init_vel_range", 0.0))
        if vel_range > 0:
            env.data.qvel[dof_idx:dof_idx+2] += rng.uniform(-vel_range, vel_range, 2)

        tilt_range = override_init_tilt_range if override_init_tilt_range is not None \
                     else float(phase_cfg.get("init_tilt_range", 0.005))
        if tilt_range > 0:
            tilt_noise = rng.uniform(-tilt_range, tilt_range, 2)
            env.data.qpos[qpos_addr+3] = 1.0
            env.data.qpos[qpos_addr+4] = tilt_noise[0] * 0.5
            env.data.qpos[qpos_addr+5] = tilt_noise[1] * 0.5
            env.data.qpos[qpos_addr+6] = 0.0
            qnorm = np.linalg.norm(env.data.qpos[qpos_addr+3:qpos_addr+7])
            env.data.qpos[qpos_addr+3:qpos_addr+7] /= qnorm
        mujoco.mj_forward(env.model, env.data)
        _sync_env_internal_state(env)
        obs = env._get_obs()

    return obs, planned_path


# ==============================================================================
# OmniReset for Descent
# ==============================================================================

def reset_for_phase_omnireset(env, phase, config, cur=None, ts=0, ep=0, suc=False):
    """Descent 段 OmniReset: 有概率从近目标处初始化。"""
    assert phase == "descent"
    cur_cfg = config.get("curriculum", {})
    omnireset_enabled = bool(cur_cfg.get("omnireset_enabled", True))
    near_prob = float(cur_cfg.get("omnireset_near_goal_prob", 0.10))

    _desc_init = None
    if cur is not None and hasattr(cur, 'get_descent_init_params'):
        _desc_init = cur.get_descent_init_params(ts, ep, suc)

    if omnireset_enabled and np.random.rand() < near_prob:
        near_xy = float(cur_cfg.get("omnireset_near_goal_xy", 0.012))
        near_z_offset = float(cur_cfg.get("omnireset_near_goal_z_offset", 0.05))
        return reset_for_phase(
            env, "descent", config,
            override_init_xy_range=near_xy,
            override_init_vel_range=0.005,
            override_init_tilt_range=0.001,
        )
    else:
        xy_r = _desc_init["xy_range"]   if _desc_init else None
        v_r  = _desc_init["vel_range"]  if _desc_init else None
        t_r  = _desc_init["tilt_range"] if _desc_init else None
        return reset_for_phase(
            env, "descent", config,
            override_init_xy_range=xy_r,
            override_init_vel_range=v_r,
            override_init_tilt_range=t_r,
        )


# ==============================================================================
# 专家辅助
# ==============================================================================

def _advance_expert_to_nearest_wp(expert, planned_path, pl_pos):
    if planned_path is None or len(planned_path) == 0:
        return
    dists = [np.linalg.norm(pl_pos - wp) for wp in planned_path]
    nearest_idx = int(np.argmin(dists))
    expert.tracker.current_idx = nearest_idx


def collect_expert_acc(expert, env, obs, current_q, phase, config):
    """收集专家 EE 加速度 (用于 BC 标签)。"""
    if phase == "cruise":
        action_4d = expert.tracker.compute_ee_acceleration(obs, target_yaw=0.0)
        acc_max_xy = float(config["ee_control"].get("acc_max_xy", 0.8))
        return np.clip(
            np.array([float(action_4d[0]), float(action_4d[1])], dtype=np.float32),
            -acc_max_xy, acc_max_xy)
    elif phase == "descent":
        # [v3] descent expert: 返回 3D acc 标签 (用于 BC 预训练)
        # PID + residual 架构下, BC 标签是残差为 0 时 PID 本身对应的 acc
        target_xy = env.target_pos.copy()
        target_pz = float(config["insertion"]["target_payload_z"])
        pl_pos = env.data.body('prefab').xpos.copy()
        pl_xy  = pl_pos[:2]
        pl_z   = float(pl_pos[2])
        dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
        pl_vel  = env.data.qvel[dof_idx:dof_idx+3].copy()
        acc_max_xy = float(config["descent_rl"].get("acc_max_xy", 0.5))
        acc_max_z  = float(config["descent_rl"].get("acc_max_z", 1.0))
        diff_xy = target_xy - pl_xy
        dist_xy = float(np.linalg.norm(diff_xy))
        if dist_xy > 0.002:
            dir_xy = diff_xy / dist_xy
            vel_proj = float(np.dot(pl_vel[:2], dir_xy))
            acc_mag = 0.6 * min(dist_xy, 0.05) / 0.05 - 0.8 * vel_proj
            acc_xy = dir_xy * np.clip(acc_mag, -1.0, 1.0) * acc_max_xy
        else:
            acc_xy = -pl_vel[:2] * 1.5
        acc_xy = np.clip(acc_xy, -acc_max_xy, acc_max_xy)
        align_factor = np.exp(-dist_xy / 0.01)
        z_error = pl_z - target_pz
        if z_error > 0.005 and align_factor > 0.3:
            acc_z = -(0.5 * min(z_error, 0.15) + 0.3 * max(float(pl_vel[2]), 0)) * align_factor
        else:
            acc_z = -float(pl_vel[2]) * 1.0
        acc_z = np.clip(acc_z, -acc_max_z, acc_max_z)
        return np.array([acc_xy[0], acc_xy[1], acc_z], dtype=np.float32)
    else:
        return np.zeros(config[f"{phase}_rl"]["action_dim"], dtype=np.float32)


# ==============================================================================
# [v3 ARCH-DESCENT] PID base + RL residual delta_q 计算
# ==============================================================================

def _apply_descent_pid_residual(expert, rl_act, obs, env, config, current_q):
    """
    Descent 段 PID + RL residual 动作合并.

    接收已经采样好的 rl_act (PPO/SAC 主循环中采样一次), 与 PID delta_q 合并.

    Bug fix (v3.1): 原 compute_descent_pid_residual_dq() 内部调用 agent.act()
    导致每步采样两次, PPO buffer 中存储的 lp/val 与实际执行的 act 不匹配,
    破坏 on-policy 约束, 引起 SR 大幅振荡. 本函数只做合并, 不做采样.

    Args:
        rl_act: 已采样的 RL 残差加速度 (shape (3,), 来自主循环的 agent.act())
        obs:    环境原始 obs (供 expert 使用)
        current_q: 当前关节角

    Returns:
        dq_total: 最终关节角增量 (shape (7,))
        pid_dq:   PID 输出的 delta_q (shape (7,), 供日志使用)
    """
    # 1. PID base delta_q
    try:
        pid_dq = expert.compute_delta_q_target(obs, current_q.astype(np.float64))
    except Exception:
        pid_dq = np.zeros(7, dtype=np.float32)
    pid_dq = np.asarray(pid_dq, dtype=np.float32)

    # 2. RL acc → residual delta_q
    residual_dq_scale = float(config["descent_rl"].get("residual_dq_scale", 0.30))
    dq_max     = np.array(config["space"].get("dq_max", [0.12]*7), dtype=np.float32)
    acc_max_xy = float(config["descent_rl"].get("acc_max_xy", 0.5))
    acc_max_z  = float(config["descent_rl"].get("acc_max_z",  1.0))

    # 归一化到 [-1, 1]
    rl_norm_xy = rl_act[:2] / max(acc_max_xy, 1e-6)
    rl_norm_z  = float(rl_act[2]) / max(acc_max_z, 1e-6) if len(rl_act) > 2 else 0.0

    # 残差量级上限 = residual_dq_scale × max(|pid_dq|, dq_mean*0.1)
    pid_dq_norm      = float(np.linalg.norm(pid_dq))
    max_residual_norm = residual_dq_scale * max(pid_dq_norm, float(np.mean(dq_max)) * 0.1)

    # 映射到前 3 个关节 (xyz 方向), 其余关节残差为 0
    rl_dq    = np.zeros(7, dtype=np.float32)
    rl_dq[0] = rl_norm_xy[0] * max_residual_norm / np.sqrt(3)
    rl_dq[1] = rl_norm_xy[1] * max_residual_norm / np.sqrt(3)
    rl_dq[2] = rl_norm_z      * max_residual_norm / np.sqrt(3)
    rl_dq    = np.clip(rl_dq, -dq_max * residual_dq_scale, dq_max * residual_dq_scale)

    # 3. 合并并整体限幅
    dq_total = np.clip(pid_dq + rl_dq, -dq_max, dq_max)
    return dq_total, pid_dq


# ==============================================================================
# 课程管理器 [v3]
# ==============================================================================

class CurriculumManager:
    def __init__(self, config, phase):
        cur = config.get("curriculum", {})
        self.enabled = bool(cur.get("enabled", True))
        self.phase   = phase
        self._rl_started = False

        # 风力课程
        self.wind_start = float(cur.get("wind_start_frac", 0.0))
        self.wind_end   = float(cur.get("wind_end_frac",   1.0))
        self.wind_steps = int(cur.get("wind_anneal_steps",  200_000))

        # ── Cruise 课程 [v3] ─────────────────────────────────────────────────
        # 障碍物: 从 0 个真实障碍物开始, 渐进到 obstacle_max_n 个
        # 0 真实障碍物阶段: 使用 shadow_obstacles 参与 reward (不参与物理碰撞)
        self.use_obs_cur   = (phase == "cruise" and
                              bool(cur.get("obstacle_enabled", True)) and
                              self.enabled)
        self.current_n_obs = int(cur.get("obstacle_start_n", 0))
        self.obs_max_n     = int(cur.get("obstacle_max_n", 3))
        self.obs_warmup    = int(cur.get("obstacle_level_warmup_eps", 500))
        self.obs_perf_win  = int(cur.get("perf_window", 300))
        self.obs_sr_thresh = float(cur.get("perf_sr_threshold", 0.65))
        self.obs_min_eps   = int(cur.get("perf_min_episodes_per_level", 3000))
        self.obs_stats     = EpisodeStats(window=self.obs_perf_win)
        self._obs_lvl_ep   = 0
        self._obs_lvl_ts   = 0
        # Shadow obstacle 配置 (0 真实障碍物时用于 reward)
        self.shadow_enabled = bool(cur.get("shadow_obstacle_enabled", True))
        self.shadow_n       = int(cur.get("shadow_obstacle_n", 3))
        self.shadow_r_min   = float(cur.get("shadow_obstacle_r_min", 0.006))
        self.shadow_r_max   = float(cur.get("shadow_obstacle_r_max", 0.015))

        # 距离课程: 禁用 (全程训练)
        self.use_cruise_dist_cur = False   # [v3] 关闭
        self.current_cruise_dist_frac = 1.0

        # 课程统计 (保留用于日志)
        self.cruise_dist_stats = EpisodeStats(window=150)

        # ── Descent 课程 [v3] ────────────────────────────────────────────────
        self.use_descent_init_cur = (phase == "descent" and
                                     bool(cur.get("descent_init_curriculum", True)) and
                                     self.enabled)

        if self.use_descent_init_cur:
            # [v3.4] per-level 查表, 不再用 start/end 线性插值
            # 优先读取 per-level 列表; 若不存在则退回旧的线性插值
            _xy_ranges   = cur.get("descent_init_xy_ranges",   None)
            _vel_ranges  = cur.get("descent_init_vel_ranges",  None)
            _tilt_ranges = cur.get("descent_init_tilt_ranges", None)
            _xy_tols     = cur.get("descent_init_xy_tols",     None)
            n_levels     = int(cur.get("descent_cur_levels", 5))

            if _xy_ranges is not None:
                # 新格式: per-level 列表
                assert len(_xy_ranges) == n_levels, "descent_init_xy_ranges 长度须等于 descent_cur_levels"
                self.descent_levels = []
                for i in range(n_levels):
                    self.descent_levels.append({
                        "xy_range":   float(_xy_ranges[i]),
                        "vel_range":  float(_vel_ranges[i])  if _vel_ranges  else 0.0,
                        "tilt_range": float(_tilt_ranges[i]) if _tilt_ranges else 0.001,
                        "xy_tol":     float(_xy_tols[i])     if _xy_tols     else 0.010,
                    })
            else:
                # 旧格式兼容: 线性插值 (xy_tol 用 tol_mult=3.0)
                xy_start   = float(cur.get("descent_init_xy_start",  0.002))
                xy_end     = float(cur.get("descent_init_xy_end",    0.030))
                vel_start  = float(cur.get("descent_init_vel_start", 0.000))
                vel_end    = float(cur.get("descent_init_vel_end",   0.020))
                tilt_start = float(cur.get("descent_init_tilt_start", 0.001))
                tilt_end   = float(cur.get("descent_init_tilt_end",   0.008))
                self.descent_levels = []
                for i in range(n_levels):
                    t = i / max(n_levels - 1, 1)
                    xy_r = xy_start + t * (xy_end - xy_start)
                    self.descent_levels.append({
                        "xy_range":   xy_r,
                        "vel_range":  vel_start + t * (vel_end  - vel_start),
                        "tilt_range": tilt_start + t * (tilt_end - tilt_start),
                        "xy_tol":     max(xy_r * 3.0, 0.005),  # 旧逻辑
                    })

            self.current_descent_level = 0
            self.descent_lvl_ep = 0
            self.descent_lvl_ts = 0
            self.descent_cur_min_eps   = int(cur.get("descent_cur_min_eps",    200))
            self.descent_cur_sr_thresh = float(cur.get("descent_cur_sr_threshold", 0.50))
            self.descent_cur_hard_cap  = int(cur.get("descent_cur_hard_cap",   300_000))
            _stats_win = int(cur.get("descent_cur_stats_window", 50))  # [v3.4] 30→50
            self.descent_stats = EpisodeStats(window=_stats_win)

    def wind_frac(self, ts):
        if not self.enabled:
            return 0.0
        frac = min(ts / max(self.wind_steps, 1), 1.0)
        return self.wind_start + frac * (self.wind_end - self.wind_start)

    def update_obs_curriculum(self, env, ep, ts, reward, success):
        """
        障碍物课程: 0→1→2→3 个真实障碍物渐进晋级.
        [v3 SHADOW] 0 真实障碍物阶段: shadow_obstacles 参与 reward (不参与碰撞).
        晋级到 >=1 后: 真实障碍物接管, shadow_obstacles 不再使用.
        """
        if not self.use_obs_cur or not self._rl_started:
            return False, self.current_n_obs

        if success:
            self.obs_stats.update(success=1.0)
        else:
            self.obs_stats.update(success=0.0)

        changed = False
        if self.current_n_obs < self.obs_max_n:
            sr  = self.obs_stats.success_rate()
            eps = ep - self._obs_lvl_ep
            advance = (eps >= self.obs_min_eps and sr >= self.obs_sr_thresh)
            if advance:
                self.current_n_obs = min(self.current_n_obs + 1, self.obs_max_n)
                env.set_curriculum_n_obstacles(self.current_n_obs)
                self._obs_lvl_ep = ep
                self._obs_lvl_ts = ts
                self.obs_stats = EpisodeStats(window=self.obs_perf_win)
                changed = True
                print(f"  [Curriculum] Obstacle level → {self.current_n_obs} 个真实障碍物")
        return changed, self.current_n_obs

    def sample_shadow_obstacles(self, start_xy, target_xy):
        """
        [v3 SHADOW] 为 0 真实障碍物阶段采样影子障碍物.
        影子障碍物随机分布在起终点连线附近, 不参与物理碰撞, 仅用于 reward.
        当真实障碍物 >= 1 时返回空列表 (真实障碍物接管).
        """
        if not self.shadow_enabled or self.current_n_obs >= 1:
            return []

        rng = np.random.default_rng()
        obstacles = []
        direction = np.asarray(target_xy) - np.asarray(start_xy)
        L = float(np.linalg.norm(direction))
        if L < 1e-6:
            return []
        direction /= L
        perp = np.array([-direction[1], direction[0]])

        path_width = 0.6   # 同 config planning.path_width
        workspace_r = 0.50
        attempts = 0
        max_attempts = self.shadow_n * 200
        base_r2 = 0.10   # 机械臂底座半径 (排除区域)
        min_clr = 0.075 + 0.03  # payload_radius + planning_margin

        while len(obstacles) < self.shadow_n and attempts < max_attempts:
            attempts += 1
            t = rng.uniform(0.15, 0.85)
            s = rng.uniform(-path_width / 2, path_width / 2)
            center = np.asarray(start_xy) + t * L * direction + s * perp
            r = rng.uniform(self.shadow_r_min, self.shadow_r_max)
            # 工作空间约束
            if float(np.linalg.norm(center)) + r > workspace_r - 0.02:
                continue
            # 离起终点和底座保持安全距离
            if (np.linalg.norm(center - np.asarray(start_xy))  < r + min_clr or
                np.linalg.norm(center - np.asarray(target_xy)) < r + min_clr or
                np.linalg.norm(center) < r + base_r2):
                continue
            # 影子障碍物互相不重叠
            if all(np.linalg.norm(center - np.array([ox, oy])) >= r + or_ + 0.01
                   for (ox, oy, or_) in obstacles):
                obstacles.append((float(center[0]), float(center[1]), float(r)))

        return obstacles

    def get_cruise_dist_frac(self, ts, ep, success):
        """距离课程已禁用, 始终返回 1.0。"""
        return 1.0

    def get_descent_init_params(self, ts, ep, success):
        if not self.use_descent_init_cur:
            return None

        if success:
            self.descent_stats.update(success=1.0)
        else:
            self.descent_stats.update(success=0.0)

        level = self.current_descent_level
        max_level = len(self.descent_levels) - 1

        if level < max_level and self._rl_started:
            sr  = self.descent_stats.success_rate()
            eps = ep - self.descent_lvl_ep
            t   = ts - self.descent_lvl_ts
            advance = ((t >= self.descent_cur_hard_cap) or
                       (eps >= self.descent_cur_min_eps and sr >= self.descent_cur_sr_thresh))
            if advance:
                self.current_descent_level = min(level + 1, max_level)
                self.descent_lvl_ep = ep
                self.descent_lvl_ts = ts
                self.descent_stats = EpisodeStats(window=30)
                print(f"  [Curriculum] Descent level {level}→{self.current_descent_level} "
                      f"(xy={self.descent_levels[self.current_descent_level]['xy_range']*1000:.1f}mm)")

        return self.descent_levels[self.current_descent_level]

    def mark_rl_start(self):
        self._rl_started = True
        self.cruise_dist_stats = EpisodeStats(window=150)
        print(f"  [Curriculum] RL 开始, 课程 SR 统计重置")

    def get_current_success_radius(self, rcfg):
        """
        Cruise 成功半径: 随障碍物课程晋级逐步收紧.

        收紧逻辑 (与障碍物 level 联动):
          n_obs=0: success_radius_start = 0.20m (宽松, 快速建立导航能力)
          n_obs=1: 插值到 0.15m
          n_obs=2: 插值到 0.12m
          n_obs=3: success_radius_end = 0.10m (接近真实 pipeline 切换条件 0.06m)

        这样训练目标随课程逐步向真实需求靠拢,
        测试时用 success_radius_end (0.10m) 作为最终评判标准.
        """
        r_start = float(rcfg.get("success_radius_start", 0.20))
        r_end   = float(rcfg.get("success_radius_end",   0.10))
        obs_max = max(self.obs_max_n, 1)
        # 线性插值: n_obs=0 → r_start, n_obs=obs_max → r_end
        frac = min(self.current_n_obs / obs_max, 1.0)
        return r_start + frac * (r_end - r_start)

    def get_curriculum_info(self):
        info = {}
        if self.use_obs_cur:
            info["cur/n_obstacles"] = self.current_n_obs
            # 日志: 是否在使用影子障碍物
            info["cur/shadow_active"] = float(
                self.shadow_enabled and self.current_n_obs == 0)
        if self.use_descent_init_cur:
            lvl = self.descent_levels[self.current_descent_level]
            info["cur/descent_level"]    = self.current_descent_level
            info["cur/descent_xy_range"] = lvl["xy_range"]
            info["cur/descent_xy_tol"]   = lvl.get("xy_tol", 0.010)  # [v3.4]
        return info


# ==============================================================================
# PPO 训练 [v3]
# ==============================================================================

def train_ppo(phase, log_dir, config, bc_ckpt=None):
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W  = int(config["train"]["log_smooth_win"])

    print(f"\n{'='*60}\n  PPO | {phase.upper()} | {T} steps | {log_dir}\n{'='*60}\n")

    # [v3] Descent 段不再使用 dual-RL, 始终用单 agent
    env    = CableRobotEnvWithObstacles(config=config)
    agent  = PPOPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    ectl   = EEAccController(config, env.ik_solver)

    # Cruise: z_pid 保留, swing_d 仅监控 (不参与控制)
    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None

    cur = CurriculumManager(config, phase)
    if cur.use_obs_cur:
        env.set_curriculum_n_obstacles(cur.current_n_obs)

    if bc_ckpt and os.path.exists(bc_ckpt):
        agent.load(bc_ckpt)
        print(f"  Loaded BC ckpt: {bc_ckpt}")
        if hasattr(agent, 'reset_log_std_for_rl'):
            agent.reset_log_std_for_rl()

    logger = Logger(log_dir, f"{phase}_ppo", f"{phase}_ppo")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time()
    _ppo_update_count = 0
    _ppo_warmup_updates = 10

    cur.mark_rl_start()

    lf = os.path.join(log_dir, f"{phase}_ppo_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                 "sr", "steps", "pl", "vl", "kl", "cf", "wind"])

    # 判断 descent pid_residual 模式
    _descent_pid_residual = (phase == "descent" and
                              bool(config.get("descent_rl", {}).get("pid_residual_mode", True)))
    if _descent_pid_residual:
        print(f"  [v3] Descent PID+residual RL 架构 (residual_dq_scale="
              f"{config['descent_rl'].get('residual_dq_scale', 0.30):.2f})")

    while ts < T:
        wf = cur.wind_frac(ts)
        env.set_wind_curriculum(wf)

        cruise_dist_frac = 1.0   # [v3] 距离课程已禁用
        _desc_init = None
        if phase == "descent":
            _desc_init = cur.get_descent_init_params(ts, ep, False)

        # ── 初始化 episode ────────────────────────────────────────────────────
        if phase == "descent":
            obs, pp = reset_for_phase_omnireset(env, phase, config, cur=cur, ts=ts, ep=ep, suc=False)
        else:
            obs, pp = reset_for_phase(env, phase, config, cruise_dist_frac=cruise_dist_frac)
        if obs is None:
            continue

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp)
            if phase != "lift":
                _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())

        ectl.reset(env._get_ee_pos(), cq)

        if z_pid is not None:
            _pl_z   = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)

        sxy = env.default_start_xy.copy(); txy = env.target_pos.copy()
        pt, py = 0.0, 0.0
        rs = REWARD_STATES[phase]()
        if hasattr(agent, 'reset_history'):
            agent.reset_history()
        if hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts

        # 注入 descent 课程信息到 reward_state
        if phase == "descent" and _desc_init is not None:
            rs.current_xy_range = _desc_init["xy_range"]
            rs.current_xy_tol   = _desc_init.get("xy_tol", None)  # [v3.4] per-level tol
            rs.current_descent_level = cur.current_descent_level
            rs.descent_n_levels      = len(cur.descent_levels)
            rs.steps_at_max_level    = ts - cur.descent_lvl_ts \
                if cur.current_descent_level >= len(cur.descent_levels) - 1 else 0

        # 注入 cruise 成功半径 & shadow_obstacles
        if phase == "cruise" and hasattr(rs, 'current_success_radius'):
            rs.current_success_radius = cur.get_current_success_radius(
                config["cruise_rl"]["reward"])
        # [v3 SHADOW] 注入影子障碍物 (0 真实障碍物阶段参与 reward, 不参与碰撞)
        if phase == "cruise" and hasattr(rs, 'shadow_obstacles'):
            rs.shadow_obstacles = cur.sample_shadow_obstacles(sxy, txy)
            rs.n_real_obstacles  = len(env._obstacles) if hasattr(env, '_obstacles') else 0

        er = 0.0; es = 0; suc = False; rd = False
        term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])

        # ── Episode 主循环 ────────────────────────────────────────────────────
        while not rd:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt, py = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
            no = agent.normalize_obs(po, update=True)
            act, lp, val = agent.act(no)

            ree = env._get_ee_pos()

            # ── Cruise: RL 直接输出完整 xy 加速度 [v3] ───────────────────────
            if phase == "cruise" and z_pid is not None:
                _pl_pos  = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw  = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) \
                    if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                if _falling:
                    rw = -5.0
                    agent.add_to_buffer(no, act, np.zeros(2, np.float32), rw, 1.0, val, lp)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts
                    rd = True; break

                # [v3] RL 直接输出完整 xy 加速度 (不再是残差)
                # swing_d 仅用于监控, 不参与 acc 叠加
                if swing_d is not None:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)   # 仅更新监控缓存

                a3 = np.array([act[0], act[1], 0.0])
                # [v3] residual_mode=False: RL acc 为完整输出, 不叠加 base
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=None, residual_mode=False)

            elif phase == "cruise":
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True,
                                          z_lock_height=float(config["cruise_rl"]["z_lock_height"]))

            # ── Descent: PID base + RL residual [v3] ─────────────────────────
            elif phase == "descent" and _descent_pid_residual:
                # Bug fix: act/lp/val 已在主循环开头采样, 直接传入不重新采样
                # compute_descent_pid_residual_dq_from_act 只做 PID + residual 合并
                dq, _pid_dq = _apply_descent_pid_residual(
                    expert, act, obs, env, config, cq)

            else:
                # lift 或 descent 非 pid_residual 模式
                dq = ectl.compute_delta_q(act, cq, ree)

            # 专家 acc (BC 标签) — Bug fix: descent pid 分支单独处理, 不被覆盖
            bct = np.zeros(agent.action_dim, np.float32)
            if phase in ("cruise", "lift"):
                try:
                    bct = collect_expert_acc(expert, env, obs, cq, phase, config)
                except Exception:
                    pass
            elif phase == "descent":
                try:
                    bct = collect_expert_acc(expert, env, obs, cq, phase, config)
                except Exception:
                    pass

            no2, _, _, _, ei = env.step(dq)
            rw, dn, sc, ri = REWARD_FNS[phase](env, no2, config, rs)
            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1:
                done = True; ri.setdefault("termination", "timeout")
            if sc:
                suc = True
            if done and ri.get("termination"):
                term_reason = ri["termination"]

            agent.add_to_buffer(no, act, bct, rw, float(done), val, lp)
            er += rw; es += 1; ts += 1; agent.total_steps = ts
            obs = no2
            agent._update_entropy_coef(global_ts=ts)

            if agent.buffer.full:
                if done:
                    lv = 0.0
                else:
                    ns_ = build_phase_obs(phase, no2, env, sxy, txy, pt, py)[0]
                    nn_ = agent.normalize_obs(ns_, update=False)
                    lv  = agent.get_value_for_state(nn_)

                agent.buffer.compute_returns_and_advantages(lv, agent.gamma, agent.gae_lambda)
                _is_warmup = _ppo_update_count < _ppo_warmup_updates
                if _is_warmup:
                    _freeze_mean = _ppo_update_count < 5
                    if _freeze_mean:
                        for _name, _param in agent.actor.named_parameters():
                            if 'log_std' not in _name:
                                _param.requires_grad_(False)
                    else:
                        for _param in agent.actor.parameters():
                            _param.requires_grad_(True)
                        _orig_lr = agent.opt_actor.param_groups[0]['lr']
                        for _pg in agent.opt_actor.param_groups:
                            _pg['lr'] = _orig_lr * 0.05
                agent.update(global_ts=ts)
                if _is_warmup:
                    if _freeze_mean:
                        for _param in agent.actor.parameters():
                            _param.requires_grad_(True)
                    else:
                        for _pg in agent.opt_actor.param_groups:
                            _pg['lr'] = _orig_lr
                _ppo_update_count += 1
                if _ppo_update_count == _ppo_warmup_updates:
                    print(f"  [Warmup] {_ppo_warmup_updates} 次保护更新完成, 切换到正常 PPO 训练")
                rd = True
            if done:
                rd = True

        # ── 更新课程 & 日志 ───────────────────────────────────────────────────
        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()

        cur.update_obs_curriculum(env, ep, ts, er, suc)
        if phase == "descent":
            cur.get_descent_init_params(ts, ep, suc)
        cur_info = cur.get_curriculum_info()

        r = agent._last_result
        m = "✅" if suc else "❌"

        log_std_dims = agent.actor.log_std.detach().cpu().numpy().copy()
        log_std_mean = float(np.mean(log_std_dims))
        log_std_str  = "/".join(f"{v:.3f}" for v in log_std_dims)
        _ls_ok = all(-2.05 <= v <= 0.35 for v in log_std_dims)
        _ls_warn = "" if _ls_ok else " ⚠️LOGSTD_OOB"

        _swing_str = ""
        _swing_energy = 0.0
        if phase == "cruise" and swing_d is not None:
            _swing_energy = swing_d.last_energy
            _angle_deg    = swing_d.last_angle_deg
            _swing_str    = f" | E:{_swing_energy*1000:.1f}mJ θ:{_angle_deg:.1f}°"

        pl_pos_now   = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) \
            if phase in ("cruise", "descent") else 0.0
        cur_str = " | ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in cur_info.items())

        print(f"Ep{ep:4d} [{ts:7d}] {m} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm{_swing_str}")
        print(f"       PPO PL:{r.policy_loss:7.4f} VL:{r.value_loss:6.3f} "
              f"KL:{r.approx_kl:.4f} CF:{r.clip_fraction:.2f} ent:{agent.entropy_coef:.4f}")
        print(f"       logstd[{log_std_str}]{_ls_warn} | {cur_str} | {term_reason}")

        log_metrics = {
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal": dist_to_goal,
            f"{phase}/wind":         wf,
            "ppo/pl":                r.policy_loss,
            "ppo/vl":                r.value_loss,
            "ppo/ent":               r.entropy_loss,
            "ppo/kl":                r.approx_kl,
            "ppo/cf":                r.clip_fraction,
            "ppo/logstd_mean":       log_std_mean,
            "ppo/entropy_coef":      r.entropy_coef_used,
            "ppo/logstd_in_bounds":  float(_ls_ok),
        }
        dim_names = ["ax", "ay", "az"] if agent.action_dim == 3 else ["ax", "ay"]
        for i, name in enumerate(dim_names[:len(log_std_dims)]):
            log_metrics[f"ppo/logstd_{name}"] = float(log_std_dims[i])
        if phase == "cruise" and swing_d is not None:
            log_metrics["cruise/swing_energy_mJ"] = _swing_energy * 1000
            log_metrics["cruise/swing_angle_deg"] = swing_d.last_angle_deg
            log_metrics["cruise/damp_gain"]       = swing_d._gain_scale
            log_metrics["cur/success_radius"] = cur.get_current_success_radius(
                config["cruise_rl"]["reward"])
        log_metrics.update(cur_info)
        logger.log(ep, log_metrics)

        with open(lf, "a", newline="") as f:
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                    f"{r.policy_loss:.5f}", f"{r.value_loss:.5f}",
                                    f"{r.approx_kl:.5f}", f"{r.clip_fraction:.3f}", f"{wf:.3f}"])
        if ep > 0 and ep % SI == 0:
            save_checkpoint(agent, log_dir, ep)
        if ep > 0 and ep % EI == 0 and sr > best:
            best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-PPO] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    logger.close(); env.close(); return agent


# ==============================================================================
# SAC 训练 [v3]
# ==============================================================================

def train_sac(phase, log_dir, config, bc_ckpt=None):
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W  = int(config["train"]["log_smooth_win"])
    WU = int(config["sac"].get("warmup_steps", 5000))

    print(f"\n{'='*60}\n  SAC | {phase.upper()} | {T} steps | warmup={WU} | {log_dir}\n{'='*60}\n")

    env    = CableRobotEnvWithObstacles(config=config)
    agent  = SACPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    wex    = JointSpaceExpert(config, env.ik_solver)
    ectl   = EEAccController(config, env.ik_solver)
    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    cur = CurriculumManager(config, phase)
    if cur.use_obs_cur:
        env.set_curriculum_n_obstacles(cur.current_n_obs)

    if bc_ckpt and os.path.exists(bc_ckpt):
        agent.load(bc_ckpt); print(f"  Loaded BC ckpt: {bc_ckpt}")

    _descent_pid_residual = (phase == "descent" and
                              bool(config.get("descent_rl", {}).get("pid_residual_mode", True)))

    logger = Logger(log_dir, f"{phase}_sac", f"{phase}_sac")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time(); onf = False

    lf = os.path.join(log_dir, f"{phase}_sac_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                 "sr", "steps", "cl", "al", "alpha", "wind"])

    while ts < T:
        wf = cur.wind_frac(ts)
        env.set_wind_curriculum(wf)

        _desc_init = None
        if phase == "descent":
            _desc_init = cur.get_descent_init_params(ts, ep, False)

        if phase == "descent":
            obs, pp = reset_for_phase_omnireset(env, phase, config, cur=cur, ts=ts, ep=ep, suc=False)
        else:
            obs, pp = reset_for_phase(env, phase, config, cruise_dist_frac=1.0)
        if obs is None:
            continue

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env); wex.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp); wex.set_path(pp)
            if phase != "lift":
                plp = env.data.body('prefab').xpos.copy()
                _advance_expert_to_nearest_wp(expert, pp, plp)
                _advance_expert_to_nearest_wp(wex, pp, plp)
        ectl.reset(env._get_ee_pos(), cq)

        if z_pid is not None:
            _pl_z   = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)

        sxy = env.default_start_xy.copy(); txy = env.target_pos.copy()
        pt, py = 0.0, 0.0
        rs = REWARD_STATES[phase]()
        if hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        if phase == "descent" and _desc_init is not None:
            rs.current_xy_range = _desc_init["xy_range"]
            rs.current_xy_tol   = _desc_init.get("xy_tol", None)  # [v3.4] per-level tol
            rs.current_descent_level = cur.current_descent_level
            rs.descent_n_levels      = len(cur.descent_levels)
            rs.steps_at_max_level    = ts - cur.descent_lvl_ts \
                if cur.current_descent_level >= len(cur.descent_levels) - 1 else 0
        if phase == "cruise" and hasattr(rs, 'current_success_radius'):
            rs.current_success_radius = cur.get_current_success_radius(
                config["cruise_rl"]["reward"])
        # [v3 SHADOW] 注入影子障碍物
        if phase == "cruise" and hasattr(rs, 'shadow_obstacles'):
            rs.shadow_obstacles = cur.sample_shadow_obstacles(sxy, txy)
            rs.n_real_obstacles  = len(env._obstacles) if hasattr(env, '_obstacles') else 0

        er = 0.0; es = 0; suc = False; term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])

        while True:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt, py = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
            no = agent.normalize_obs(po, update=not onf)

            if ts < WU:
                try:
                    act = collect_expert_acc(wex, env, obs, cq, phase, config)
                    act += np.random.normal(0, 0.1, len(act)).astype(np.float32)
                except Exception:
                    act = np.zeros(agent.action_dim, np.float32)
            else:
                act = agent.act(no, deterministic=False)
            if ts == WU and not onf:
                agent._freeze_obs_norm = True; onf = True

            ree = env._get_ee_pos()

            if phase == "cruise" and z_pid is not None:
                _pl_pos  = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw  = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) \
                    if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                if _falling:
                    rw = -5.0
                    npo, _, _ = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
                    nn_ = agent.normalize_obs(npo, update=False)
                    agent.remember(no, act, nn_, rw, 1.0)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts; break

                # [v3] 监控 swing_d 但不叠加到控制
                if swing_d is not None:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)

                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=None, residual_mode=False)

            elif phase == "cruise":
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True,
                                          z_lock_height=float(config["cruise_rl"]["z_lock_height"]))

            elif phase == "descent" and _descent_pid_residual:
                # SAC 版本: act 直接用 SAC 输出 (已在主循环采样, 不重复采样)
                dq, _pid_dq = _apply_descent_pid_residual(
                    expert, act, obs, env, config, cq)
            else:
                dq = ectl.compute_delta_q(act, cq, ree)

            no2, _, _, _, ei = env.step(dq)
            rw, dn, sc, ri = REWARD_FNS[phase](env, no2, config, rs)
            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1:
                done = True; ri.setdefault("termination", "timeout")
            if sc: suc = True
            if done and ri.get("termination"): term_reason = ri["termination"]

            npo, _, _ = build_phase_obs(phase, no2, env, sxy, txy, pt, py)
            nn_ = agent.normalize_obs(npo, update=False)
            _pl_pos_now = env.data.body('prefab').xpos.copy()
            agent.remember(no, act, nn_, rw, float(done), achieved_pos=_pl_pos_now)
            if ts >= WU and ts % agent.update_interval == 0:
                agent.train_step()
            er += rw; es += 1; ts += 1; agent.total_steps = ts; obs = no2
            if done:
                if phase == "descent":
                    agent.flush_episode_her(txy, float(config["insertion"]["target_payload_z"]))
                break

        # ── 更新课程 & 日志 ───────────────────────────────────────────────────
        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()

        cur.update_obs_curriculum(env, ep, ts, er, suc)
        if phase == "descent":
            cur.get_descent_init_params(ts, ep, suc)
        cur_info = cur.get_curriculum_info()

        r = agent._last_result
        m = "✅" if suc else "❌"
        pl_pos_now   = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) \
            if phase in ("cruise", "descent") else 0.0

        _swing_str = ""
        _swing_energy = 0.0
        if phase == "cruise" and swing_d is not None:
            _swing_energy = swing_d.last_energy
            _angle_deg    = swing_d.last_angle_deg
            _swing_str    = f" | E:{_swing_energy*1000:.1f}mJ θ:{_angle_deg:.1f}°"

        cur_str = " | ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in cur_info.items())
        print(f"Ep{ep:4d} [{ts:7d}] {m} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm{_swing_str}")
        print(f"       SAC CL:{r.critic_loss:7.4f} AL:{r.actor_loss:7.4f} "
              f"α:{agent.alpha:.4f} | {cur_str} | {term_reason}")

        log_metrics = {
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal": dist_to_goal,
            f"{phase}/wind":         wf,
            "sac/alpha":             agent.alpha,
            "sac/cl":                r.critic_loss,
            "sac/al":                r.actor_loss,
        }
        if phase == "cruise" and swing_d is not None:
            log_metrics["cruise/swing_energy_mJ"] = _swing_energy * 1000
            log_metrics["cruise/swing_angle_deg"] = swing_d.last_angle_deg
            log_metrics["cruise/damp_gain"]       = swing_d._gain_scale
            log_metrics["cur/success_radius"] = cur.get_current_success_radius(
                config["cruise_rl"]["reward"])
        log_metrics.update(cur_info)
        logger.log(ep, log_metrics)

        with open(lf, "a", newline="") as f:
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                    f"{r.critic_loss:.5f}", f"{r.actor_loss:.5f}",
                                    f"{agent.alpha:.5f}", f"{wf:.3f}"])
        if ep > 0 and ep % SI == 0: save_checkpoint(agent, log_dir, ep)
        if ep > 0 and ep % EI == 0 and sr > best: best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-SAC] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    logger.close(); env.close(); return agent


# ==============================================================================
# BC 预训练 — Cruise [v3: n_epochs 使用 n_epochs_cruise]
# ==============================================================================

def pretrain_bc_cruise(agent, config):
    bc = config.get("bc_pretrain", {})
    if not bc.get("enabled", True): return

    nep               = int(bc.get("n_episodes",      1000))
    # [v3] cruise 使用专用 epoch 数 (比 descent 多但不过拟合)
    nepoch            = int(bc.get("n_epochs_cruise",  40))
    lr                = float(bc.get("lr",             3e-4))
    lr_decay          = float(bc.get("lr_decay",       0.5))
    lr_decay_interval = int(bc.get("lr_decay_interval", 20))
    bs                = int(bc.get("batch_size",       512))
    eval_interval     = int(bc.get("eval_interval",    5))   # [v3]
    eval_episodes     = int(bc.get("eval_episodes",    30))
    patience          = int(bc.get("patience",         3))   # [v3]
    loss_thresh       = float(bc.get("loss_threshold", 0.30))
    eps_start         = float(bc.get("epsilon_start",  0.3))
    eps_end           = float(bc.get("epsilon_end",    0.0))
    n_dagger_rounds   = int(bc.get("n_dagger_rounds",  2))

    acc_max   = float(config["ee_control"].get("acc_max_xy", 0.8))
    max_steps = int(config["cruise_rl"]["max_steps"])

    import copy as _copy
    from phase_reward import CruiseRewardState, compute_cruise_reward
    bc_config = _copy.deepcopy(config)
    rope_L = float(config["controller"].get("L", 0.5))
    bc_config["step_logic"]["instability_grace_steps"] = max_steps
    bc_config["step_logic"]["swing_xy_max"] = rope_L * 1.2

    print(f"\n{'='*60}")
    print(f"  [BC-cruise] {nep} eps | {nepoch} epochs | DAgger={n_dagger_rounds}")
    print(f"{'='*60}")

    env    = CableRobotEnvWithObstacles(config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    aobs_tr, aact_tr = [], []
    aobs_ev, aact_ev = [], []
    ve = 0; att = 0; succ = 0

    while ve < nep and att < nep * 3:
        att += 1
        sys.stdout = open(os.devnull, 'w')
        try:
            obs, pp = reset_for_phase(env, "cruise", config)
        finally:
            sys.stdout.close(); sys.stdout = sys.__stdout__
        if obs is None or pp is None: continue

        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        if pp is not None:
            expert.set_path(pp)
            _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())

        txy = env.target_pos.copy()
        pt2, py2 = 0.0, 0.0
        ve += 1
        is_eval = (ve % max(nep // max(eval_episodes, 1), 1) == 0)
        ep_succ = False
        rstate = CruiseRewardState(); rstate.total_steps_global = 0

        for _s in range(max_steps):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            action_4d = expert.tracker.compute_ee_acceleration(obs, target_yaw=0.0)
            acc_label = np.clip(
                np.array([float(action_4d[0]), float(action_4d[1])], dtype=np.float32),
                -acc_max, acc_max)
            po, pt2, py2 = build_cruise_obs(obs, env, txy, pt2, py2)
            no = agent.normalize_obs(po, update=True)
            if is_eval:
                aobs_ev.append(no.copy()); aact_ev.append(acc_label.copy())
            else:
                aobs_tr.append(no.copy()); aact_tr.append(acc_label.copy())
            delta_q = expert.compute_delta_q_target(obs, current_q)
            obs, _r, env_term, env_trunc, env_info = env.step(delta_q)
            _rb, r_done, r_success, _ = compute_cruise_reward(env, obs, bc_config, rstate)
            if r_success: ep_succ = True
            if r_done or env_term or env_trunc or env_info.get("nan_detected", False): break

        if ep_succ: succ += 1
        if ve % 50 == 0 or ve == nep:
            print(f"  [BC-cruise] {ve:4d}/{nep} | tr={len(aobs_tr):6d} ev={len(aobs_ev):5d} "
                  f"| 到达={succ/max(ve,1)*100:.0f}%")

    env.close()
    if len(aobs_tr) < 500:
        print(f"  [BC-cruise] ⚠ 样本不足({len(aobs_tr)})，跳过"); return

    ot = torch.tensor(np.array(aobs_tr), device=agent.device)
    at = torch.tensor(np.array(aact_tr), device=agent.device)
    oe = torch.tensor(np.array(aobs_ev), device=agent.device) if aobs_ev else None
    ae = torch.tensor(np.array(aact_ev), device=agent.device) if aobs_ev else None
    ns = len(aobs_tr)

    bp  = [p for n, p in agent.actor.named_parameters() if 'log_std' not in n]
    opt = torch.optim.AdamW(bp, lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_interval, gamma=lr_decay)
    best_eval = float('inf'); best_state = None

    for dagger_round in range(n_dagger_rounds):
        if dagger_round > 0:
            dagger_eps = nep // n_dagger_rounds
            print(f"  [BC-cruise] DAgger 轮{dagger_round+1}: 策略rollout({dagger_eps}eps)")
            env_d = CableRobotEnvWithObstacles(config=config)
            exp_d = JointSpaceExpert(config, env_d.ik_solver)
            dobs_new, dact_new = [], []
            dv = 0
            while dv < dagger_eps:
                sys.stdout = open(os.devnull, 'w')
                try: d_obs, d_pp = reset_for_phase(env_d, "cruise", config)
                finally: sys.stdout.close(); sys.stdout = sys.__stdout__
                if d_obs is None or d_pp is None: continue
                d_cq = env_d.data.qpos[:7].copy()
                exp_d.reset(d_obs, d_cq, env=env_d)
                if d_pp is not None:
                    exp_d.set_path(d_pp)
                    _advance_expert_to_nearest_wp(exp_d, d_pp, env_d.data.body('prefab').xpos.copy())
                d_txy = env_d.target_pos.copy(); d_pt2, d_py2 = 0.0, 0.0
                dv += 1
                d_rs = CruiseRewardState(); d_rs.total_steps_global = 0
                for _ds in range(max_steps):
                    d_a4 = exp_d.tracker.compute_ee_acceleration(d_obs, target_yaw=0.0)
                    d_lbl = np.clip(np.array([float(d_a4[0]), float(d_a4[1])], dtype=np.float32),
                                    -acc_max, acc_max)
                    d_po, d_pt2, d_py2 = build_cruise_obs(d_obs, env_d, d_txy, d_pt2, d_py2)
                    d_no = agent.normalize_obs(d_po, update=False)
                    dobs_new.append(d_no.copy()); dact_new.append(d_lbl.copy())
                    d_cq_now = env_d.data.qpos[:7].copy().astype(np.float32)
                    d_dq = exp_d.compute_delta_q_target(d_obs, d_cq_now)
                    d_obs, _, d_et, d_etr, d_ei = env_d.step(d_dq)
                    _, d_rd, d_rs_ok, _ = compute_cruise_reward(env_d, d_obs, bc_config, d_rs)
                    if d_rd or d_et or d_etr or d_rs_ok: break
            env_d.close()
            if dobs_new:
                ot = torch.cat([ot, torch.tensor(np.array(dobs_new), device=agent.device)], dim=0)
                at = torch.cat([at, torch.tensor(np.array(dact_new), device=agent.device)], dim=0)
                ns = len(ot)
                print(f"  [BC-cruise] DAgger +{len(dobs_new)} 样本, 总计 {ns}")

        round_nepoch = nepoch if dagger_round == 0 else nepoch // 2
        pat = 0
        for e in range(round_nepoch):
            eps = max(eps_end, eps_start + (eps_end - eps_start) * min(e / max(round_nepoch - 1, 1), 1.0))
            agent.actor.train()
            idx = np.random.permutation(ns)
            tl = 0.0; nb = 0
            for st in range(0, ns, bs):
                i = idx[st:st+bs]
                ab = at[i]
                if eps > 0:
                    mask = torch.rand(len(i), device=agent.device) < eps
                    noise = (torch.rand_like(ab)*2 - 1) * acc_max
                    ab = torch.where(mask.unsqueeze(1), noise, ab)
                loss, _, _ = agent.actor.bc_forward(ot[i], ab)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(bp, 1.0)
                opt.step()
                tl += loss.item(); nb += 1
            sch.step(); tl /= max(nb, 1)

            if (e+1) % eval_interval == 0 or e == round_nepoch-1:
                agent.actor.eval()
                with torch.no_grad():
                    el = agent.actor.bc_forward(oe, ae)[0].item() if oe is not None else tl
                print(f"  R{dagger_round+1} Ep{e+1:3d}/{round_nepoch} tr={tl:.4f} ev={el:.4f} "
                      f"eps={eps:.2f} pat={pat}/{patience}")
                if el < best_eval - 1e-5:
                    best_eval = el
                    best_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                    pat = 0
                else:
                    pat += 1
                if pat >= patience: print(f"  [BC-cruise] 早停"); break
                if el < loss_thresh: print(f"  [BC-cruise] loss 达标"); break

    if best_state:
        agent.actor.load_state_dict(best_state)
        print(f"  [BC-cruise] ✅ 最佳权重 ev={best_eval:.4f}")
    if hasattr(agent, 'reset_log_std_for_rl'):
        agent.reset_log_std_for_rl()
    print("  [BC-cruise] 完成\n")


# ==============================================================================
# BC 预训练 — Descent [v3: n_epochs=20, 防过拟合]
# ==============================================================================

def pretrain_bc_descent(agent, config):
    bc = config.get("bc_pretrain", {})
    if not bc.get("enabled", True): return

    nep       = int(bc.get("n_episodes",    1000))
    # [v3] 大幅削减 epochs: 60→20, 防止过拟合导致熵崩塌
    nepoch    = int(bc.get("n_epochs",      20))
    lr        = float(bc.get("lr",          3e-4))
    lr_decay  = float(bc.get("lr_decay",    0.5))
    lr_decay_interval = int(bc.get("lr_decay_interval", 20))
    bs        = int(bc.get("batch_size",    512))
    eval_interval = int(bc.get("eval_interval", 5))
    eval_episodes = int(bc.get("eval_episodes", 30))
    patience  = int(bc.get("patience",      3))   # [v3] 5→3
    loss_thresh = 0.10
    eps_start = float(bc.get("epsilon_start", 0.3))
    eps_end   = float(bc.get("epsilon_end",   0.0))

    ee_cfg     = config.get("ee_control", {})
    dcfg       = config.get("descent_rl", {})
    acc_max_xy = float(dcfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 0.5)))
    acc_max_z  = float(dcfg.get("acc_max_z",  ee_cfg.get("acc_max_z", 1.0)))

    print(f"\n{'='*60}")
    print(f"  [BC-descent v3] {nep} eps | {nepoch} epochs (防过拟合: patience={patience})")
    print(f"  acc_max: xy={acc_max_xy}, z={acc_max_z} | loss_thresh={loss_thresh}")
    print(f"  [v3 NOTE] BC 标签是残差为 0 时 PID 对应的 acc, 初始化 actor 接近零残差")
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
            sys.stdout.close(); sys.stdout = old_stdout
        if obs is None: continue

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp)
            _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())

        txy = env.target_pos.copy()
        target_pz = float(config["insertion"]["target_payload_z"])
        pt2, py2 = 0.0, 0.0
        ve += 1
        is_eval_ep = (ve % max(nep // eval_episodes, 1) == 0)
        mx = int(config["descent_rl"]["max_steps"])

        for s in range(mx):
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt2, py2 = build_descent_obs(obs, env, txy, pt2, py2)
            no = agent.normalize_obs(po, update=True)

            # [v3] BC 标签: 残差为 0 (目标是让 RL 从零残差开始学习)
            # 这里标签全为 0, 让 actor 初始化接近零输出
            # 注: 如果需要更好的初始化, 可以用专家 acc 作为目标
            #     但零残差初始化与 PID+residual 架构更匹配
            acc_label = np.zeros(3, dtype=np.float32)

            if is_eval_ep:
                aobs_eval.append(no.copy()); aact_eval.append(acc_label.copy())
            else:
                aobs_train.append(no.copy()); aact_train.append(acc_label.copy())

            # 用 expert PID 执行动作 (收集 obs 分布)
            dq = expert.compute_delta_q_target(obs, cq)
            obs, _, t, tr, _ = env.step(dq)
            if t or tr: break

        succ += 1
        if ve % 100 == 0 or ve == nep:
            print(f"  [BC-descent] {ve}/{nep} | train={len(aobs_train)} eval={len(aobs_eval)}")

    env.close()
    if len(aobs_train) < 100:
        print(f"  [BC-descent] ⚠ 样本不足({len(aobs_train)}), 跳过"); return

    ot = torch.tensor(np.array(aobs_train), device=agent.device)
    at = torch.tensor(np.array(aact_train), device=agent.device)
    oe = torch.tensor(np.array(aobs_eval),  device=agent.device) if aobs_eval else None
    ae = torch.tensor(np.array(aact_eval),  device=agent.device) if aobs_eval else None
    ns = len(aobs_train)

    bp = [p for n, p in agent.actor.named_parameters() if 'log_std' not in n]
    opt = torch.optim.AdamW(bp, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_interval, gamma=lr_decay)
    best_eval_loss = float('inf'); best_state = None; patience_count = 0

    for e in range(nepoch):
        eps = eps_start + (eps_end - eps_start) * min(e / max(nepoch - 1, 1), 1.0)
        agent.actor.train()
        idx = np.random.permutation(ns)
        tl = 0.0; nb = 0
        for st in range(0, ns, bs):
            i = idx[st:st+bs]
            act_b = at[i]
            if eps > 0:
                mask = torch.rand(len(i), device=agent.device) < eps
                noise = torch.rand_like(act_b) * 2 - 1
                act_b = torch.where(mask.unsqueeze(1), noise * 0.1, act_b)   # 小幅噪声
            loss, loss_a, _ = agent.actor.bc_forward(ot[i], act_b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            opt.step()
            tl += loss.item(); nb += 1
        scheduler.step(); tl /= max(nb, 1)

        if (e + 1) % eval_interval == 0 or e == nepoch - 1:
            agent.actor.eval()
            with torch.no_grad():
                el = agent.actor.bc_forward(oe, ae)[0].item() if oe is not None else tl
            print(f"  Ep{e+1:3d}/{nepoch} train={tl:.5f} eval={el:.5f} "
                  f"eps={eps:.3f} pat={patience_count}/{patience}")
            if el < best_eval_loss - 1e-5:
                best_eval_loss = el
                best_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= patience: print(f"  [BC-descent] 早停"); break
            if el < loss_thresh: print(f"  [BC-descent] 目标达成 {el:.5f}"); break

    if best_state:
        agent.actor.load_state_dict(best_state)
        print(f"  [BC-descent] ✅ 最佳权重 eval={best_eval_loss:.5f}")
    if hasattr(agent, 'reset_log_std_for_rl'):
        agent.reset_log_std_for_rl()
    print(f"  [BC-descent] 完成\n")


# ==============================================================================
# 入口
# ==============================================================================

def train(phase, log_dir, algo="ppo", custom_config=None, bc_ckpt=None, skip_bc=False):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config:
        for k, v in custom_config.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    set_global_seed(config["train"].get("seed", 42))
    os.makedirs(log_dir, exist_ok=True)

    if phase in ("cruise", "descent") and not skip_bc and bc_ckpt is None:
        ag = PPOPhaseAgent(phase, config=config) if algo == "ppo" \
             else SACPhaseAgent(phase, config=config)
        if phase == "cruise":
            pretrain_bc_cruise(ag, config)
        else:
            pretrain_bc_descent(ag, config)
        bp = os.path.join(log_dir, "ckpt_bc.pt"); ag.save(bp); bc_ckpt = bp

    if algo == "ppo":
        return train_ppo(phase, log_dir, config, bc_ckpt)
    elif algo == "sac":
        return train_sac(phase, log_dir, config, bc_ckpt)
    else:
        raise ValueError(f"Unknown algo: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三阶段 RL 训练 v3")
    parser.add_argument("--phase",      type=str, required=True, choices=["lift", "cruise", "descent"])
    parser.add_argument("--algo",       type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--log-dir",    type=str, default=None)
    parser.add_argument("--timesteps",  type=int, default=None)
    parser.add_argument("--render",     action="store_true")
    parser.add_argument("--gpu",        type=int, default=0)
    parser.add_argument("--bc-ckpt",    type=str, default=None)
    parser.add_argument("--skip-bc",    action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--wind",       action="store_true")
    parser.add_argument("--no-curriculum", action="store_true", help="禁用课程学习")
    parser.add_argument("--obstacles",  type=int, default=None,
                        help="手动设置障碍物数量 (0=无障碍物训练, 1-3=引入障碍物)")
    args = parser.parse_args()

    ld = args.log_dir or f"saves/{args.phase}_{args.algo}"
    cc = {}
    if args.render:      cc.setdefault("sim", {})["render"] = True
    if args.gpu != 0:    cc.setdefault("train", {})["gpu_id"] = args.gpu
    if args.timesteps:   cc.setdefault("train", {})["total_timesteps"] = args.timesteps
    cc.setdefault("train", {})["seed"] = args.seed
    if args.wind:        cc.setdefault("wind", {})["enabled"] = True
    if args.no_curriculum: cc.setdefault("curriculum", {})["enabled"] = False
    if args.obstacles is not None:
        # 手动指定障碍物数量 (用于加入障碍物的第二阶段训练)
        if args.obstacles == 0:
            cc.setdefault("curriculum", {})["obstacle_enabled"] = False
            cc.setdefault("scene", {})["n_obstacles"] = 0
        else:
            cc.setdefault("curriculum", {})["obstacle_enabled"] = True
            cc.setdefault("curriculum", {})["obstacle_start_n"] = args.obstacles
            cc.setdefault("scene", {})["n_obstacles"] = args.obstacles

    train(args.phase, ld, algo=args.algo, custom_config=cc or None,
          bc_ckpt=args.bc_ckpt, skip_bc=args.skip_bc)