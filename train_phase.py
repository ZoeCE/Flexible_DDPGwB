# ==============================================================================
# train_phase.py — 三阶段独立训练框架 v3 (优化版)
#
# 主要变更:
#   [OPT-CUR]  CurriculumManager 全面重设计:
#     - Cruise: 距离课程 (agent 从近处目标开始, 渐进到全程)
#     - Descent: 初始化范围课程 (xy/vel/tilt 渐进扩大)
#     - 两个阶段的障碍物课程逻辑保留并修正 perf_reward_threshold
#   [OPT-ENT]  train_ppo/train_sac: 移除动态 entropy_coef 覆盖
#              (entropy_coef 已在 config 和 agent 中正确配置)
#   [OPT-MON]  新增 wandb 监控:
#     - 各阶段分维度 log_std (logstd_ax, logstd_ay, logstd_az)
#     - success_rate 独立曲线
#     - dist_to_goal (cruise/descent)
#     - curriculum_level
#   [OPT-BC]   BC 预训练: n_epochs 从 config 读取 (60, 防过拟合)
#   [OPT-INIT] Descent reset_for_phase 支持动态 init_xy_range (课程注入)
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
    """在直接修改物理状态后, 同步 env 内部所有跟踪变量。"""
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
    """
    为指定阶段直接初始化物理状态。

    新增参数 (用于课程学习):
        override_init_xy_range:   覆盖 config 中的 init_xy_range
        override_init_vel_range:  覆盖 config 中的 init_vel_range
        override_init_tilt_range: 覆盖 config 中的 init_tilt_range (descent 专用)
        cruise_dist_frac:         巡航起点距终点的比例 [0.3, 1.0] (cruise 专用)
    """
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

        # [OPT-CUR] 距离课程: 起点从靠近终点处开始, 渐进扩大到真实起点
        if cruise_dist_frac < 1.0:
            # 在 start_xy 到 target_xy 的连线上插值
            lerp_start = target_xy + cruise_dist_frac * (start_xy - target_xy)
            noise_range = float(phase_cfg.get("init_xy_range", 0.01))
            lerp_start += rng.uniform(-noise_range, noise_range, 2)
            start_xy = lerp_start

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
            if has_viewer: env.viewer.sync()
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
        if init_q is None or np.any(np.isnan(init_q)): init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx = env.model.jnt_dofadr[pref_jnt]

        # [OPT-CUR] 支持课程注入 init_xy_range
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
            if has_viewer: env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        # [OPT-CUR] 支持课程注入 init_vel_range
        vel_range = override_init_vel_range if override_init_vel_range is not None \
                    else float(phase_cfg.get("init_vel_range", 0.0))
        if vel_range > 0:
            noise_vel = rng.uniform(-vel_range, vel_range, 3)
            env.data.qvel[dof_idx:dof_idx+3] += noise_vel

        # [OPT-CUR] 支持课程注入 init_tilt_range (通过微小初始倾斜)
        tilt_range = override_init_tilt_range if override_init_tilt_range is not None \
                     else float(phase_cfg.get("init_tilt_range", 0.005))
        if tilt_range > 0:
            # 微小的 payload 角速度模拟初始倾斜
            tilt_noise = rng.uniform(-tilt_range, tilt_range, 2)
            env.data.qvel[dof_idx+3:dof_idx+5] += tilt_noise * 2.0

        mujoco.mj_forward(env.model, env.data)
        _sync_env_internal_state(env)
        obs = env._get_obs()

    return obs, planned_path


def reset_for_phase_omnireset(env, phase, config, cur=None, ts=0, ep=0, suc=False):
    """
    [OMNIRESET] OmniReset-style 重置入口 (descent 专用增强版).
    依据: Weirdlab 2025 "Emergent Dexterity via Diverse Resets"

    逻辑:
      - 以 omnireset_near_goal_prob 的概率从"目标附近"初始化
        (覆盖"已对准,只需下降"的高价值状态, 提供最直接的成功路径)
      - 其余概率走正常课程初始化 (xy_range 从小到大)
    """
    if phase != "descent" or cur is None:
        # 非 descent 阶段或无课程管理器: 走标准路径
        desc_params = cur.get_descent_init_params(ts, ep, suc) if cur else None
        return reset_for_phase(
            env, phase, config,
            override_init_xy_range=desc_params["xy_range"]   if desc_params else None,
            override_init_vel_range=desc_params["vel_range"]  if desc_params else None,
            override_init_tilt_range=desc_params["tilt_range"] if desc_params else None,
        )

    cur_cfg = config.get("curriculum", {})
    omnireset_enabled = cur_cfg.get("omnireset_enabled", True)
    near_goal_prob    = float(cur_cfg.get("omnireset_near_goal_prob", 0.25))
    near_goal_xy      = float(cur_cfg.get("omnireset_near_goal_xy", 0.008))
    near_goal_z_off   = float(cur_cfg.get("omnireset_near_goal_z_offset", 0.05))

    desc_params = cur.get_descent_init_params(ts, ep, suc)

    # 判断是否走 near-goal 初始化
    if omnireset_enabled and cur.enabled and np.random.random() < near_goal_prob:
        # OmniReset near-goal: 覆盖 xy_range 为极小值 + 从目标上方 z_offset 开始
        return reset_for_phase(
            env, phase, config,
            override_init_xy_range=near_goal_xy,
            override_init_vel_range=0.005,   # 接近静止
            override_init_tilt_range=0.002,  # 接近水平
        )
    else:
        # 正常课程初始化
        return reset_for_phase(
            env, phase, config,
            override_init_xy_range=desc_params["xy_range"]    if desc_params else None,
            override_init_vel_range=desc_params["vel_range"]   if desc_params else None,
            override_init_tilt_range=desc_params["tilt_range"] if desc_params else None,
        )


# ==============================================================================
# 专家 BC 标签
# ==============================================================================

def collect_expert_acc(expert, env, obs, current_q, phase, config):
    """BC 标签提取。"""
    ee_cfg = config.get("ee_control", {})
    pcfg   = config.get(f"{phase}_rl", {})
    axy    = float(pcfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 2.0)))
    az     = float(pcfg.get("acc_max_z",  ee_cfg.get("acc_max_z",  3.0)))

    if phase == "cruise":
        action_4d = expert.tracker.compute_ee_acceleration(obs, target_yaw=0.0)
        acc = np.array([float(action_4d[0]), float(action_4d[1])], dtype=np.float32)
        acc[0] = np.clip(acc[0], -axy, axy)
        acc[1] = np.clip(acc[1], -axy, axy)
        return acc

    _ = expert.compute_joint_target(obs, current_q)
    target_vel = expert._ee_vel.copy()
    dt = float(config.get("ee_control", {}).get("integrator_dt", 0.1))
    real_vel = env._ee_vel_cache.copy()
    acc = (target_vel - real_vel) / max(dt, 1e-6)
    acc = acc[:3]
    acc[0] = np.clip(acc[0], -axy, axy)
    acc[1] = np.clip(acc[1], -axy, axy)
    acc[2] = np.clip(acc[2], -az,  az)
    return acc.astype(np.float32)

def _advance_expert_to_nearest_wp(expert, planned_path, pl_pos):
    if planned_path is None or len(planned_path) == 0: return
    dists = [np.linalg.norm(pl_pos - wp) for wp in planned_path]
    expert.tracker.current_idx = int(np.argmin(dists))


# ==============================================================================
# 课程学习 [OPT-CUR] 全面重设计
# ==============================================================================

class CurriculumManager:
    """
    统一课程学习管理器。

    支持:
      - 风力课程 (所有阶段)
      - Cruise 障碍物课程 (0→3 障碍物)
      - Cruise 距离课程 (起点从靠近终点处渐进到真实起点)
      - Descent 初始化范围课程 (xy/vel/tilt 渐进扩大)
    """

    def __init__(self, config, phase):
        cur = config.get("curriculum", {})
        self.enabled = cur.get("enabled", True)
        self.phase = phase
        self.config = config  # [FIX2] 保存config供 update_obs_curriculum 读取解锁条件

        # ── 风力课程 ──────────────────────────────────────────────────────────
        self.wind_start = float(cur.get("wind_start_frac", 0.0))
        self.wind_end = float(cur.get("wind_end_frac", 1.0))
        self.wind_anneal = int(cur.get("wind_anneal_steps", 500_000))

        # ── Cruise 障碍物课程 ──────────────────────────────────────────────────
        self.use_obs_cur = (phase == "cruise" and cur.get("obstacle_enabled", False))
        self.current_n_obs = int(cur.get("obstacle_start_n", 0))
        self.obs_max = int(cur.get("obstacle_max_n", 3))
        self.perf_window = int(cur.get("perf_window", 30))
        self.perf_sr = float(cur.get("perf_sr_threshold", 0.40))
        self.perf_rwd = float(cur.get("perf_reward_threshold", -10.0))
        self.perf_min = int(cur.get("perf_min_episodes_per_level", 100))
        self.perf_cap = int(cur.get("perf_hard_cap_steps", 600_000))
        self.obs_lvl_step = 0; self.obs_lvl_ep = 0
        self.obs_stats = EpisodeStats(window=self.perf_window)

        # ── Cruise 距离课程 [OPT-CUR] ─────────────────────────────────────────
        self.use_cruise_dist_cur = (phase == "cruise" and
                                    cur.get("cruise_dist_curriculum", False))
        self.cruise_dist_start = float(cur.get("cruise_dist_start_frac", 0.3))
        self.cruise_dist_end = float(cur.get("cruise_dist_end_frac", 1.0))
        self.cruise_dist_anneal = int(cur.get("cruise_dist_anneal_steps", 400_000))
        self.cruise_dist_sr_thresh = float(cur.get("cruise_dist_sr_threshold", 0.35))
        self.current_cruise_dist_frac = self.cruise_dist_start
        self.cruise_dist_stats = EpisodeStats(window=30)
        self.cruise_dist_level_ts = 0
        self._rl_started = False          # ★ BC 阶段不更新距离课程

        # ── Descent 初始化范围课程 [OPT-CUR] ──────────────────────────────────
        self.use_descent_init_cur = (phase == "descent" and
                                     cur.get("descent_init_curriculum", False))
        n_levels = int(cur.get("descent_cur_levels", 5))
        # xy_range 课程: start → end, 分 n_levels 级
        xy_start = float(cur.get("descent_init_xy_start", 0.005))
        xy_end = float(cur.get("descent_init_xy_end", 0.030))
        # vel_range 课程
        vel_start = float(cur.get("descent_init_vel_start", 0.000))
        vel_end = float(cur.get("descent_init_vel_end", 0.030))
        # tilt_range 课程
        tilt_start = float(cur.get("descent_init_tilt_start", 0.002))
        tilt_end = float(cur.get("descent_init_tilt_end", 0.010))
        # 生成各级别的参数
        self.descent_levels = [
            {
                "xy_range": xy_start + i / max(n_levels - 1, 1) * (xy_end - xy_start),
                "vel_range": vel_start + i / max(n_levels - 1, 1) * (vel_end - vel_start),
                "tilt_range": tilt_start + i / max(n_levels - 1, 1) * (tilt_end - tilt_start),
            }
            for i in range(n_levels)
        ]
        self.current_descent_level = 0
        self.descent_cur_min_eps = int(cur.get("descent_cur_min_eps", 150))
        self.descent_cur_sr_thresh = float(cur.get("descent_cur_sr_threshold", 0.35))
        self.descent_cur_hard_cap = int(cur.get("descent_cur_hard_cap", 400_000))
        self.descent_lvl_ep = 0; self.descent_lvl_ts = 0
        self.descent_stats = EpisodeStats(window=30)

    def wind_frac(self, t):
        # [v4] 风力完全解耦: 只有在 config["wind"]["enabled"]=True 时才增加风力
        # 主训练阶段默认 enabled=False, 鲁棒微调阶段再开启
        wind_enabled = self.config.get("wind", {}).get("enabled", False)
        if not wind_enabled or not self.enabled:
            return 0.0
        if self.wind_anneal <= 0:
            return self.wind_end
        f = min(t / self.wind_anneal, 1.0)
        return self.wind_start + f * (self.wind_end - self.wind_start)

    # ── Cruise 障碍物课程 ──────────────────────────────────────────────────────
    def update_obs_curriculum(self, env, ep, t, r, s):
        """
        更新障碍物课程。返回 (是否晋级, 当前障碍物数)。

        [FIX2] 新增解锁条件: dist_frac 必须达到 obstacle_unlock_dist_frac 才开始引入障碍物.
        解锁后: SR+ep驱动晋级, ep保底 (不用steps保底, 避免过快推进).
        """
        if not self.use_obs_cur: return False, self.current_n_obs

        # [FIX2] 解锁检查: dist_frac 未达阈值时障碍物锁定为 0
        cur_cfg = self.config.get("curriculum", {}) if hasattr(self, 'config') else {}
        unlock_frac = float(cur_cfg.get("obstacle_unlock_dist_frac", 0.50))
        if self.current_cruise_dist_frac < unlock_frac:
            if self.current_n_obs > 0:
                # 极端情况: dist_frac回退时也锁定 (理论上不会)
                self.current_n_obs = 0
                env.set_curriculum_n_obstacles(0)
            return False, 0  # 未解锁, 维持 0 障碍物

        self.obs_stats.update(reward=r, success=float(s))
        if self.current_n_obs >= self.obs_max: return False, self.current_n_obs

        sr = self.obs_stats.success_rate()
        eps_at_level = ep - self.obs_lvl_ep
        steps_at_level = t - self.obs_lvl_step

        # [v8] 改用 gradient_step 计数作为 hard_cap
        # 原因: episode计数不准确 (每次gradient update≈14ep, 6000ep≈430 updates 太少)
        # 目标: 每个障碍物级别至少3000次梯度更新 = ~43000 episodes
        hard_cap_steps = int(cur_cfg.get("obstacle_hard_cap_grad_steps", 3000))

        # [v8] 新增: level切换后warmup期内不允许晋级
        # 防止切换后obs_stats中旧数据污染导致立即再次晋级
        warmup_eps = int(cur_cfg.get("obstacle_level_warmup_eps", 200))
        in_warmup = eps_at_level < warmup_eps

        advance = (not in_warmup) and (
            (steps_at_level >= hard_cap_steps) or
            (eps_at_level >= self.perf_min and
             sr >= self.perf_sr and
             self.obs_stats.mean("reward") >= self.perf_rwd))
        if advance:
            old_n = self.current_n_obs
            self.current_n_obs = min(self.current_n_obs + 1, self.obs_max)
            env.set_curriculum_n_obstacles(self.current_n_obs)
            self.obs_lvl_ep = ep; self.obs_lvl_step = t
            self.obs_stats = EpisodeStats(window=self.perf_window)
            print(f"  [Curriculum] n_obstacles {old_n}→{self.current_n_obs} "
                  f"(dist_frac={self.current_cruise_dist_frac:.2f}, SR={sr:.2f}, eps={eps_at_level})")
            return True, self.current_n_obs
        return False, self.current_n_obs

    # ── Cruise 距离课程 [OPT-CUR] ─────────────────────────────────────────────
    def get_cruise_dist_frac(self, t, ep, s):
        """
        返回当前 cruise 距离系数 [0.3, 1.0]。

        课程逻辑: SR 达标后渐进扩大起点距离。
        同时支持基于时间步数的平滑退火作为备用。
        """
        if not self.use_cruise_dist_cur or not self.enabled:
            return 1.0
        # ★ BC 阶段 (RL 未开始) 不更新 SR 统计, 防止高 SR 过早触发晋级
        if not self._rl_started:
            time_frac = min(t / max(self.cruise_dist_anneal, 1), 1.0)
            return self.cruise_dist_start + time_frac * (
                self.cruise_dist_end - self.cruise_dist_start)
        self.cruise_dist_stats.update(success=float(s))

        # 基于 SR 的晋级 — 每次步进 0.05, 避免单次大跳变
        sr = self.cruise_dist_stats.success_rate()
        eps_at_level = ep - self.cruise_dist_level_ts
        if (eps_at_level >= 150 and sr >= self.cruise_dist_sr_thresh and  # [FIX] 80→150: 更保守晋级
                self.current_cruise_dist_frac < self.cruise_dist_end):
            step = 0.05   # ★ 固定 5% 步进, 原来是 (1.0-0.3)/5 = 14% 一跳
            old_frac = self.current_cruise_dist_frac
            self.current_cruise_dist_frac = min(
                self.current_cruise_dist_frac + step, self.cruise_dist_end)
            self.cruise_dist_level_ts = ep
            self.cruise_dist_stats = EpisodeStats(window=30)
            print(f"  [Curriculum] cruise_dist_frac {old_frac:.2f}→{self.current_cruise_dist_frac:.2f} (SR={sr:.2f})")

        # [FIX] 纯 SR 驱动, 无时间退火
        # 保底: 每 hard_cap_eps=500 个 episode 强制晋一级 (防死锁, 不依赖时间步)
        hard_cap_eps = 800  # [FIX] 500→800: 保底晋级更保守
        if (ep - self.cruise_dist_level_ts >= hard_cap_eps and
                self.current_cruise_dist_frac < self.cruise_dist_end):
            old_frac = self.current_cruise_dist_frac
            self.current_cruise_dist_frac = min(
                self.current_cruise_dist_frac + 0.05, self.cruise_dist_end)
            self.cruise_dist_level_ts = ep
            self.cruise_dist_stats = EpisodeStats(window=30)
            print(f"  [Curriculum] 保底晋级 (hard_cap_eps={hard_cap_eps}): "
                  f"dist_frac {old_frac:.2f}→{self.current_cruise_dist_frac:.2f}")

        return self.current_cruise_dist_frac

    # ── Descent 初始化范围课程 [OPT-CUR] ──────────────────────────────────────
    def get_descent_init_params(self, t, ep, s):
        """
        返回当前 descent 初始化参数 dict: {xy_range, vel_range, tilt_range}。

        课程逻辑: SR 达标后晋升到下一级 (更大随机范围)。
        """
        if not self.use_descent_init_cur or not self.enabled:
            return None  # None 表示使用 config 默认值

        self.descent_stats.update(success=float(s))
        level = self.current_descent_level
        max_level = len(self.descent_levels) - 1

        if level < max_level:
            sr = self.descent_stats.success_rate()
            eps_at_level = ep - self.descent_lvl_ep
            time_at_level = t - self.descent_lvl_ts
            advance = ((time_at_level >= self.descent_cur_hard_cap) or
                       (eps_at_level >= self.descent_cur_min_eps and
                        sr >= self.descent_cur_sr_thresh))
            if advance:
                self.current_descent_level = min(level + 1, max_level)
                self.descent_lvl_ep = ep
                self.descent_lvl_ts = t
                self.descent_stats = EpisodeStats(window=30)

        return self.descent_levels[self.current_descent_level]

    def mark_rl_start(self):
        """RL 训练开始时调用: 重置课程 SR 统计, 防止 BC 阶段高 SR 污染。"""
        self._rl_started = True
        self.cruise_dist_stats = EpisodeStats(window=30)
        self.cruise_dist_level_ts = 0    # 从 ep=0 重新计算晋级条件
        print(f"  [Curriculum] RL 开始, 课程 SR 统计重置, _rl_started=True")

    def get_current_success_radius(self, rcfg):
        """
        [BUG-A FIX] success_radius 联动计算, 保证初始位置不在成功区域内.

        安全约束: success_radius < (1-dist_frac) × full_dist × safety_margin
        其中 full_dist ≈ 0.41m (从 default_start_xy 到 target_xy 的距离)
        """
        r_start = float(rcfg.get("success_radius_start", 0.12))
        r_end   = float(rcfg.get("success_radius_end",   0.10))
        if not self.use_cruise_dist_cur:
            return r_start

        d_start = self.cruise_dist_start
        d_end   = self.cruise_dist_end
        if d_end <= d_start:
            return r_start
        frac = (self.current_cruise_dist_frac - d_start) / (d_end - d_start)
        frac = float(np.clip(frac, 0.0, 1.0))
        table_sr = r_start + frac * (r_end - r_start)

        # [BUG-A FIX] 安全上界: 初始距离=(1-dist_frac)×full_dist, 留15%余量
        # full_dist 从 config 中估算 (两点之间的距离)
        full_dist = float(rcfg.get("estimated_full_dist_m", 0.41))
        safety_margin = 0.80  # success_radius 最多是初始距离的80%
        safety_cap = (1.0 - self.current_cruise_dist_frac) * full_dist * safety_margin

        final_sr = min(table_sr, safety_cap)
        return max(final_sr, 0.05)  # 最小5cm, 避免过于严格

    def get_curriculum_info(self):
        """返回用于日志的课程状态 dict。"""
        info = {}
        if self.use_obs_cur:
            info["cur/n_obstacles"] = self.current_n_obs
        if self.use_cruise_dist_cur:
            info["cur/cruise_dist_frac"] = self.current_cruise_dist_frac
        if self.use_descent_init_cur:
            lvl = self.descent_levels[self.current_descent_level]
            info["cur/descent_level"] = self.current_descent_level
            info["cur/descent_xy_range"] = lvl["xy_range"]
        return info



def _bc_rl_warmup_rollout(env, agent, expert, ectl, z_pid, phase, config, cur, bc_ckpt):
    """
    [FIX-KL32] BC结束后、RL开始前的热身Rollout.
    
    目的:
      1. 用BC策略(确定性)收集一个完整rollout(n_steps步)
      2. 在此过程中更新obs_norm统计 (让归一化更准确)
      3. 用真实的RL return训练Critic (让VL从25降到合理范围)
      4. 不更新Actor (保护BC权重)
    
    效果: 第一次完整PPO更新时KL从32降到合理范围(预期<2)
    """
    import torch.nn.functional as F
    
    if bc_ckpt is None:
        return False  # 无BC ckpt则跳过
    
    n_warmup_steps = int(config["ppo"]["n_steps"])
    max_steps_phase = int(config[f"{phase}_rl"]["max_steps"])
    
    from phase_reward import LiftRewardState, CruiseRewardState, DescentRewardState
    RSTATE_CLS = {"lift": LiftRewardState, "cruise": CruiseRewardState, "descent": DescentRewardState}
    
    obs_buf = []; act_buf = []; ret_buf = []; val_buf = []
    ts_warmup = 0
    
    while ts_warmup < n_warmup_steps:
        # 初始化episode
        if phase == "descent":
            obs, pp = reset_for_phase_omnireset(env, phase, config, cur=cur, ts=0, ep=0, suc=False)
        else:
            dist_frac = cur.get_cruise_dist_frac(0, 0, False) if phase == "cruise" else 1.0
            obs, pp = reset_for_phase(env, phase, config,
                                       cruise_dist_frac=dist_frac)
        if obs is None:
            continue
        
        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp)
            _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())
        ectl.reset(env._get_ee_pos(), cq)
        if z_pid is not None:
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(float(env.data.body('prefab').xpos[2]), _pl_yaw)
        
        sxy = env.default_start_xy.copy(); txy = env.target_pos.copy()
        pt, py = 0.0, 0.0
        rs = RSTATE_CLS[phase]()
        
        for _ in range(max_steps_phase):
            if ts_warmup >= n_warmup_steps:
                break
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt, py = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
            # 用BC策略确定性动作 (不随机)
            no = agent.normalize_obs(po, update=True)
            act, lp, val = agent.act(no, deterministic=True)
            
            ree = env._get_ee_pos()
            if phase == "cruise" and z_pid is not None:
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz = float(env.data.qvel[_dof_idx + 2])
                _pl_mat2 = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat2).as_euler('xyz')
                _pl_yaw2 = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(env.data.body('prefab').xpos[2]), _pl_vz, _pl_yaw2, _pl_yaw_rate)
                if _falling:
                    break
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True,
                                           z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                                           z_pid_correction=_z_corr, target_yaw=_tgt_yaw)
            elif phase == "cruise":
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True,
                                           z_lock_height=float(config["cruise_rl"]["z_lock_height"]))
            else:
                dq = ectl.compute_delta_q(act, cq, ree)
            
            no2, _, _, _, ei = env.step(dq)
            rw, dn, sc, ri = REWARD_FNS[phase](env, no2, config, rs)
            done = dn or ei.get("nan_detected", False)
            if ts_warmup + 1 >= n_warmup_steps: done = True
            
            obs_buf.append(no.copy())
            val_buf.append(val)
            ret_buf.append(rw)  # 简化: 用step reward近似return
            ts_warmup += 1
            obs = no2
            if done:
                break
    
    if len(obs_buf) < 100:
        return False
    
    # 计算粗略return (倒序GAE简化版)
    gamma = float(config["ppo"]["gamma"])
    returns = []
    G = 0.0
    for r in reversed(ret_buf):
        G = r + gamma * G
        returns.insert(0, G)
    
    # 只训练Critic, 不更新Actor
    device = agent.device
    obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
    
    critic_warmup_epochs = 30  # [FIX2] 5→30: VL=150根因, 需要更充分预热
    batch_sz = int(config["ppo"]["batch_size"])
    for _ep in range(critic_warmup_epochs):
        idx = np.random.permutation(len(obs_buf))
        for start in range(0, len(obs_buf), batch_sz):
            b = idx[start:start+batch_sz]
            v = agent.critic(obs_t[b])
            loss = F.huber_loss(v, ret_t[b])
            agent.opt_critic.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),
                                           float(config["ppo"]["max_grad_norm"]))
            agent.opt_critic.step()
    
    print(f"  [Warmup] Critic预热: {critic_warmup_epochs} epochs × {len(obs_buf)} samples")
    return True


# ==============================================================================
# PPO 训练 [OPT-CUR] [OPT-MON]
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
    z_pid   = CruiseZYawPID(config)         if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    cur = CurriculumManager(config, phase)
    if cur.use_obs_cur: env.set_curriculum_n_obstacles(cur.current_n_obs)

    if bc_ckpt and os.path.exists(bc_ckpt):
        agent.load(bc_ckpt)
        print(f"  Loaded BC ckpt: {bc_ckpt}")
        if hasattr(agent, 'reset_log_std_for_rl'):
            agent.reset_log_std_for_rl()

    logger = Logger(log_dir, f"{phase}_ppo", f"{phase}_ppo")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time()
    # descent init params (课程第一级)
    _desc_init = None
    _ppo_update_count = 0  # [FIX] 记录 PPO 更新次数, 前几次用更小 lr 防 KL 爆炸
    _ppo_warmup_updates = 10  # [FIX] 3→10: 更多次warmup防KL=32爆炸

    # [BOOTSTRAP-V] Bootstrapped PBRS 参数
    _cur_cfg = config.get("curriculum", {})
    _pbrs_enabled = _cur_cfg.get("bootstrapped_pbrs_enabled", True) and phase == "descent"
    _pbrs_coef    = float(_cur_cfg.get("bootstrapped_pbrs_coef", 0.5))
    _pbrs_gamma   = float(config["ppo"].get("gamma", 0.99))

    # ★ BUG5: BC 阶段结束, RL 开始前重置课程 SR 统计
    cur.mark_rl_start()

    # [FIX-KL32] BC后热身Rollout: 用BC策略收集数据, 只训练Critic和obs_norm
    # 让Critic先适应RL的return分布, 防止第一次PPO更新KL=32
    _warmup_done = _bc_rl_warmup_rollout(
        env, agent, expert, ectl, z_pid, phase, config, cur, bc_ckpt)
    if _warmup_done:
        print(f"  [Warmup] BC后热身Rollout完成, Critic预热完毕")

    lf = os.path.join(log_dir, f"{phase}_ppo_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode","total_steps","ep_reward","avg_reward",
                                 "sr","steps","pl","vl","kl","cf","wind"])

    while ts < T:
        wf = cur.wind_frac(ts)
        env.set_wind_curriculum(wf)

        # ── 获取课程参数 ─────────────────────────────────────────────────────
        cruise_dist_frac = cur.get_cruise_dist_frac(ts, ep, False) if phase == "cruise" else 1.0
        if phase == "descent":
            _desc_init = cur.get_descent_init_params(ts, ep, False)

        # ── 初始化 episode [OMNIRESET] ───────────────────────────────────────
        if phase == "descent":
            obs, pp = reset_for_phase_omnireset(env, phase, config, cur=cur, ts=ts, ep=ep, suc=False)
        else:
            obs, pp = reset_for_phase(
                env, phase, config,
                override_init_xy_range=None,
                override_init_vel_range=None,
                override_init_tilt_range=None,
                cruise_dist_frac=cruise_dist_frac,
            )
        if obs is None: continue

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            expert.set_path(pp)
            if phase != "lift": _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())
        ectl.reset(env._get_ee_pos(), cq)

        if z_pid is not None:
            _pl_z = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)

        sxy = env.default_start_xy.copy(); txy = env.target_pos.copy()
        pt, py = 0.0, 0.0; rs = REWARD_STATES[phase]()
        if hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts
        # [FIX] 注入当前 success_radius (由课程管理器控制)
        if phase == "cruise" and hasattr(rs, 'current_success_radius'):
            rcfg_cur = config["cruise_rl"]["reward"]
            rs.current_success_radius = cur.get_current_success_radius(rcfg_cur)
        er = 0.0; es = 0; suc = False; rd = False
        term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])

        # ── Episode 主循环 ────────────────────────────────────────────────────
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
                _pl_pos  = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw  = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                if _falling:
                    rw = -5.0
                    agent.buffer.add(no, act, bct, rw, 1.0, val, lp)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts
                    rd = True; break

                # ── 底层防摆控制器 + RL 残差模式 ──────────────────────────────
                _pl_vel = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                _ee_vel = getattr(env, '_ee_vel_cache', np.zeros(3))
                _base_acc, _damp_info = swing_d.compute(
                    _pl_pos, ree, _pl_vel, _ee_vel)

                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=_base_acc[:2], residual_mode=True)
            elif phase == "cruise":
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True,
                                           z_lock_height=float(config["cruise_rl"]["z_lock_height"]))
            else:
                dq = ectl.compute_delta_q(act, cq, ree)

            no2, _, _, _, ei = env.step(dq)
            rw, dn, sc, ri = REWARD_FNS[phase](env, no2, config, rs)
            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1: done = True; ri.setdefault("termination", "timeout")
            if sc: suc = True
            if done and ri.get("termination"):
                term_reason = ri["termination"]

            # [BOOTSTRAP-V] Bootstrapped PBRS (2025)
            # F(s,s') = γ·V(s') - V(s): 用 Critic 值函数作 potential
            # 理论保证: policy-invariant (Ng 1999 + Wiewiora 2003)
            if (_pbrs_enabled and hasattr(agent, 'get_value_for_state') and
                    not done and phase == "descent"):
                try:
                    ns_po, _, _ = build_phase_obs(phase, no2, env, sxy, txy, pt, py)
                    ns_no = agent.normalize_obs(ns_po, update=False)
                    v_next = agent.get_value_for_state(ns_no)
                    pbrs_bonus = _pbrs_gamma * v_next - val
                    rw += _pbrs_coef * float(np.clip(pbrs_bonus, -1.0, 1.0))
                except Exception: pass

            agent.buffer.add(no, act, bct, rw, float(done), val, lp)
            er += rw; es += 1; ts += 1; agent.total_steps = ts; obs = no2

            # ★ BUG6: 每步实时退火 entropy_coef, 不等 buffer 满
            agent._update_entropy_coef(global_ts=ts)

            if agent.buffer.full:
                if done: lv = 0.0
                else:
                    ns_ = build_phase_obs(phase, no2, env, sxy, txy, pt, py)[0]
                    nn_ = agent.normalize_obs(ns_, update=False)
                    with torch.no_grad():
                        lv = agent.critic(
                            torch.tensor(nn_, dtype=torch.float32, device=agent.device).unsqueeze(0)
                        ).item()
                agent.buffer.compute_returns_and_advantages(lv, agent.gamma, agent.gae_lambda)
                # [FIX] PPO warmup: 前N次更新保护BC权重
                _is_warmup = _ppo_update_count < _ppo_warmup_updates
                if _is_warmup:
                    # 阶段1 (前5次): 冻结 mean_head, 只更新 log_std+Critic
                    # 阶段2 (5-10次): mean_head 解冻但 lr=0.01×
                    _orig_lr = agent.opt_actor.param_groups[0]['lr']
                    _freeze_mean = _ppo_update_count < 5
                    if _freeze_mean:
                        # 冻结 backbone 和 mean_head
                        for _name, _param in agent.actor.named_parameters():
                            if 'log_std' not in _name:
                                _param.requires_grad_(False)
                        _warmup_lr_scale = 0.0  # 冻结时lr无意义
                    else:
                        for _param in agent.actor.parameters():
                            _param.requires_grad_(True)
                        for _pg in agent.opt_actor.param_groups:
                            _pg['lr'] = _orig_lr * 0.05  # 阶段2: 5%lr
                agent.update(global_ts=ts)
                if _is_warmup:
                    if _freeze_mean:
                        # 解冻参数
                        for _param in agent.actor.parameters():
                            _param.requires_grad_(True)
                    else:
                        for _pg in agent.opt_actor.param_groups:
                            _pg['lr'] = _orig_lr
                _ppo_update_count += 1
                if _ppo_update_count == _ppo_warmup_updates:
                    print(f"  [Warmup] {_ppo_warmup_updates}次保护更新完成, 切换到正常PPO训练")
                rd = True
            if done: rd = True

        # ── 更新课程 & 日志 ──────────────────────────────────────────────────
        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()

        obs_adv, n_obs = cur.update_obs_curriculum(env, ep, ts, er, suc)
        _ = cur.get_cruise_dist_frac(ts, ep, suc)
        if phase == "descent": _ = cur.get_descent_init_params(ts, ep, suc)
        cur_info = cur.get_curriculum_info()

        r = agent._last_result
        m = "✅" if suc else "❌"

        # ── logstd 监控 ───────────────────────────────────────────────────────
        log_std_dims = agent.actor.log_std.detach().cpu().numpy().copy()
        log_std_mean = float(np.mean(log_std_dims))
        log_std_str  = "/".join(f"{v:.3f}" for v in log_std_dims)
        # [v4] 硬约束 [−2.0, 0.3]: 超出范围说明 clamp 未生效
        _ls_ok = all(-2.05 <= v <= 0.35 for v in log_std_dims)
        _ls_warn = "" if _ls_ok else " ⚠️LOGSTD_OOB"

        # ── 防摆能量监控 (cruise 专用) ────────────────────────────────────────
        _swing_str = ""
        _swing_energy = 0.0
        if phase == "cruise" and swing_d is not None:
            _swing_energy = swing_d.last_energy
            _angle_deg    = swing_d.last_angle_deg
            _swing_str    = f" | E:{_swing_energy*1000:.1f}mJ θ:{_angle_deg:.1f}°"

        # ── dist_to_goal ──────────────────────────────────────────────────────
        pl_pos_now   = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) if phase in ("cruise", "descent") else 0.0

        cur_str = " | ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in cur_info.items())

        # ── Terminal 输出 (详细版) ────────────────────────────────────────────
        print(f"Ep{ep:4d} [{ts:7d}] {m} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm{_swing_str}")
        print(f"       PPO PL:{r.policy_loss:7.4f} VL:{r.value_loss:6.3f} "
              f"KL:{r.approx_kl:.4f} CF:{r.clip_fraction:.2f} "
              f"ent:{agent.entropy_coef:.4f}")
        print(f"       logstd[{log_std_str}]{_ls_warn} | {cur_str} | {term_reason}")

        # ── wandb 指标 ────────────────────────────────────────────────────────
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
            # logstd 硬约束监控: 值应始终在 [-2.0, 0.3]
            "ppo/logstd_in_bounds":  float(_ls_ok),
        }
        # 分维度 logstd
        dim_names = ["ax", "ay", "az"] if agent.action_dim == 3 else ["ax", "ay"]
        for i, name in enumerate(dim_names[:len(log_std_dims)]):
            log_metrics[f"ppo/logstd_{name}"] = float(log_std_dims[i])
        # cruise 专属: 防摆能量监控
        if phase == "cruise" and swing_d is not None:
            log_metrics["cruise/swing_energy_mJ"] = _swing_energy * 1000
            log_metrics["cruise/swing_angle_deg"] = swing_d.last_angle_deg
            log_metrics["cruise/damp_gain"]       = swing_d._gain_scale
            if cur_info:
                log_metrics["cur/success_radius"]  = cur.get_current_success_radius(
                    config["cruise_rl"]["reward"])
        log_metrics.update(cur_info)
        logger.log(ep, log_metrics)

        with open(lf, "a", newline="") as f:
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                     f"{r.policy_loss:.5f}", f"{r.value_loss:.5f}",
                                     f"{r.approx_kl:.5f}", f"{r.clip_fraction:.3f}", f"{wf:.3f}"])
        if ep > 0 and ep % SI == 0: save_checkpoint(agent, log_dir, ep)
        if ep > 0 and ep % EI == 0 and sr > best: best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-PPO] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, best_sr={best*100:.0f}%")
    logger.close(); env.close(); return agent


# ==============================================================================
# SAC 训练 [OPT-CUR] [OPT-MON]
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
    z_pid   = CruiseZYawPID(config)         if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    cur = CurriculumManager(config, phase)
    if cur.use_obs_cur: env.set_curriculum_n_obstacles(cur.current_n_obs)

    if bc_ckpt and os.path.exists(bc_ckpt):
        agent.load(bc_ckpt)
        print(f"  Loaded BC ckpt: {bc_ckpt}")

    logger = Logger(log_dir, f"{phase}_sac", f"{phase}_sac")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time(); onf = False
    _desc_init = None

    lf = os.path.join(log_dir, f"{phase}_sac_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode","total_steps","ep_reward","avg_reward",
                                 "sr","steps","cl","al","alpha","wind"])

    while ts < T:
        wf = cur.wind_frac(ts)
        env.set_wind_curriculum(wf)

        cruise_dist_frac = cur.get_cruise_dist_frac(ts, ep, False) if phase == "cruise" else 1.0
        if phase == "descent":
            _desc_init = cur.get_descent_init_params(ts, ep, False)

        if phase == "descent":
            obs, pp = reset_for_phase_omnireset(env, phase, config, cur=cur, ts=ts, ep=ep, suc=False)
        else:
            obs, pp = reset_for_phase(
                env, phase, config,
                override_init_xy_range=None,
                override_init_vel_range=None,
                override_init_tilt_range=None,
                cruise_dist_frac=cruise_dist_frac,
            )
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

        if z_pid is not None:
            _pl_z = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)

        sxy = env.default_start_xy.copy(); txy = env.target_pos.copy()
        pt, py = 0.0, 0.0; rs = REWARD_STATES[phase]()
        if hasattr(rs, 'total_steps_global'): rs.total_steps_global = ts
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
                _pl_pos  = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw  = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                if _falling:
                    rw = -5.0
                    npo, _, _ = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
                    nn_ = agent.normalize_obs(npo, update=False)
                    agent.remember(no, act, nn_, rw, 1.0)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts
                    break
                # ── 底层防摆 + RL 残差 ─────────────────────────────────────
                _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                _base_acc, _damp_info = swing_d.compute(
                    _pl_pos, ree, _pl_vel, _ee_vel)
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=_base_acc[:2], residual_mode=True)
            elif phase == "cruise":
                a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(a3, cq, ree, lock_z=True,
                                           z_lock_height=float(config["cruise_rl"]["z_lock_height"]))
            else: dq = ectl.compute_delta_q(act, cq, ree)

            no2, _, _, _, ei = env.step(dq)
            rw, dn, sc, ri = REWARD_FNS[phase](env, no2, config, rs)
            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1: done = True; ri.setdefault("termination", "timeout")
            if sc: suc = True
            if done and ri.get("termination"): term_reason = ri["termination"]

            npo, _, _ = build_phase_obs(phase, no2, env, sxy, txy, pt, py)
            nn_ = agent.normalize_obs(npo, update=False)
            # [HER] 传入 achieved_pos (payload 当前 xyz) 供 HER 重标注
            _pl_pos_now = env.data.body('prefab').xpos.copy()
            agent.remember(no, act, nn_, rw, float(done), achieved_pos=_pl_pos_now)
            if ts >= WU and ts % agent.update_interval == 0: agent.train_step()
            er += rw; es += 1; ts += 1; agent.total_steps = ts; obs = no2
            if done:
                # [HER] episode 结束时触发 future 重标注 (仅 descent)
                if phase == "descent":
                    agent.flush_episode_her(txy, float(config["insertion"]["target_payload_z"]))
                break

        # ── 更新课程 & 日志 ──────────────────────────────────────────────────
        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()

        cur.update_obs_curriculum(env, ep, ts, er, suc)
        _ = cur.get_cruise_dist_frac(ts, ep, suc)
        if phase == "descent": _ = cur.get_descent_init_params(ts, ep, suc)
        cur_info = cur.get_curriculum_info()

        r = agent._last_result
        m = "✅" if suc else "❌"
        pl_pos_now   = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) if phase in ("cruise", "descent") else 0.0

        # ── 防摆能量监控 (cruise 专用) ────────────────────────────────────────
        _swing_str    = ""
        _swing_energy = 0.0
        if phase == "cruise" and swing_d is not None:
            _swing_energy = swing_d.last_energy
            _angle_deg    = swing_d.last_angle_deg
            _swing_str    = f" | E:{_swing_energy*1000:.1f}mJ θ:{_angle_deg:.1f}°"

        cur_str = " | ".join(f"{k.split('/')[-1]}={v:.3f}" for k, v in cur_info.items())

        # ── Terminal 输出 (详细版) ────────────────────────────────────────────
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
            _rcfg_log = config["cruise_rl"]["reward"]
            log_metrics["cur/success_radius"] = cur.get_current_success_radius(_rcfg_log)
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
    print(f"\n[{phase.upper()}-SAC] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, best_sr={best*100:.0f}%")
    logger.close(); env.close(); return agent


# ==============================================================================
# BC 预训练 — Cruise [OPT-BC]
# ==============================================================================

def pretrain_bc_cruise(agent, config):
    """巡航段 BC 预训练 v8 (保持不变, 从 config 读取 n_epochs)。"""
    bc = config.get("bc_pretrain", {})
    if not bc.get("enabled", True): return

    nep               = int(bc.get("n_episodes",        1000))
    nepoch            = int(bc.get("n_epochs",           60))   # [OPT-BC] 默认 60
    lr                = float(bc.get("lr",               3e-4))
    lr_decay          = float(bc.get("lr_decay",         0.5))
    lr_decay_interval = int(bc.get("lr_decay_interval",  20))   # [OPT-BC] 默认 20
    bs                = int(bc.get("batch_size",         512))
    eval_interval     = int(bc.get("eval_interval",      10))   # [OPT-BC] 默认 10
    eval_episodes     = int(bc.get("eval_episodes",      30))
    patience          = int(bc.get("patience",           5))    # [OPT-BC] 默认 5
    loss_thresh       = float(bc.get("loss_threshold",   0.30))
    eps_start         = float(bc.get("epsilon_start",    0.3))
    eps_end           = float(bc.get("epsilon_end",      0.0))
    n_dagger_rounds   = int(bc.get("n_dagger_rounds",    2))

    acc_max   = float(config["ee_control"].get("acc_max_xy", 0.8))
    max_steps = int(config["cruise_rl"]["max_steps"])

    import copy as _copy
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
    ve = 0; att = 0; succ = 0; total_steps = 0

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
            total_steps += 1
            _rb, r_done, r_success, _ = compute_cruise_reward(env, obs, bc_config, rstate)
            if r_success: ep_succ = True
            if r_done or env_term or env_trunc or env_info.get("nan_detected", False): break

        if ep_succ: succ += 1
        if ve % 50 == 0 or ve == nep:
            print(f"  [BC-cruise] {ve:4d}/{nep} | tr={len(aobs_tr):6d} ev={len(aobs_ev):5d} | 到达={succ/max(ve,1)*100:.0f}%")

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
                    d_lbl = np.clip(np.array([float(d_a4[0]), float(d_a4[1])], dtype=np.float32), -acc_max, acc_max)
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
                print(f"  R{dagger_round+1} Ep{e+1:3d}/{round_nepoch} tr={tl:.4f} ev={el:.4f} eps={eps:.2f} pat={pat}/{patience}")
                if el < best_eval - 1e-5:
                    best_eval = el; best_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}; pat = 0
                else: pat += 1
                if pat >= patience: print(f"  [BC-cruise] 早停"); break
                if el < loss_thresh: print(f"  [BC-cruise] loss达标"); break

    if best_state: agent.actor.load_state_dict(best_state); print(f"  [BC-cruise] ✅ 最佳权重 ev={best_eval:.4f}")
    if hasattr(agent, 'reset_log_std_for_rl'): agent.reset_log_std_for_rl()
    print("  [BC-cruise] 完成\n")


# ==============================================================================
# BC 预训练 — Descent [OPT-BC]
# ==============================================================================

def pretrain_bc_descent(agent, config):
    """下降段 BC 预训练 v3 (减少 epochs, 防过拟合)。"""
    bc = config.get("bc_pretrain", {})
    if not bc.get("enabled", True): return

    nep        = int(bc.get("n_episodes", 1000))
    nepoch     = int(bc.get("n_epochs", 60))     # [OPT-BC] 200→60
    lr         = float(bc.get("lr", 3e-4))
    lr_decay   = float(bc.get("lr_decay", 0.5))
    lr_decay_interval = int(bc.get("lr_decay_interval", 20))  # [OPT-BC] 50→20
    bs         = int(bc.get("batch_size", 512))
    eval_interval  = int(bc.get("eval_interval", 10))         # [OPT-BC] 20→10
    eval_episodes  = int(bc.get("eval_episodes", 30))
    patience       = int(bc.get("patience", 5))               # [OPT-BC] 8→5
    loss_thresh    = 0.05  # [OPT-BC] 0.02→0.05: 更早停止, 防过拟合
    eps_start  = float(bc.get("epsilon_start", 0.3))
    eps_end    = float(bc.get("epsilon_end", 0.0))

    ee_cfg   = config.get("ee_control", {})
    dcfg     = config.get("descent_rl", {})
    acc_max_xy = float(dcfg.get("acc_max_xy", ee_cfg.get("acc_max_xy", 0.5)))
    acc_max_z  = float(dcfg.get("acc_max_z",  ee_cfg.get("acc_max_z", 1.0)))

    print(f"\n{'='*60}")
    print(f"  [BC-descent] {nep} eps | {nepoch} epochs (防过拟合: patience={patience})")
    print(f"  acc_max: xy={acc_max_xy}, z={acc_max_z} | loss_thresh={loss_thresh}")
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
        ep_steps = 0

        for s in range(mx):
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt2, py2 = build_descent_obs(obs, env, txy, pt2, py2)
            no = agent.normalize_obs(po, update=True)

            pl_pos = env.data.body('prefab').xpos.copy()
            pl_xy  = pl_pos[:2]
            pl_z   = float(pl_pos[2])
            dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
            pl_vel  = env.data.qvel[dof_idx:dof_idx+3].copy()

            diff_xy = txy - pl_xy
            dist_xy = float(np.linalg.norm(diff_xy))
            if dist_xy > 0.002:
                dir_xy = diff_xy / dist_xy
                vel_proj = float(np.dot(pl_vel[:2], dir_xy))
                kp_xy = 0.6; kd_xy = 0.8
                acc_mag_xy = kp_xy * min(dist_xy, 0.05) / 0.05 - kd_xy * vel_proj
                acc_xy = dir_xy * np.clip(acc_mag_xy, -1.0, 1.0) * acc_max_xy
            else:
                acc_xy = -pl_vel[:2] * 1.5
            acc_xy = np.clip(acc_xy, -acc_max_xy, acc_max_xy)

            align_factor = np.exp(-dist_xy / 0.01)
            z_error = pl_z - target_pz
            kp_z = 0.5; kd_z = 0.3
            if z_error > 0.005 and align_factor > 0.3:
                acc_z = -(kp_z * min(z_error, 0.15) + kd_z * max(float(pl_vel[2]), 0)) * align_factor
            else:
                acc_z = -float(pl_vel[2]) * 1.0
            acc_z = np.clip(acc_z, -acc_max_z, acc_max_z)
            acc_label = np.array([acc_xy[0], acc_xy[1], acc_z], dtype=np.float32)

            if is_eval_ep:
                aobs_eval.append(no.copy()); aact_eval.append(acc_label.copy())
            else:
                aobs_train.append(no.copy()); aact_train.append(acc_label.copy())

            dq = expert.compute_delta_q_target(obs, cq)
            obs, _, t, tr, _ = env.step(dq)
            ep_steps += 1
            if t or tr: break

        if ep_steps > 20: succ += 1
        if ve % 100 == 0 or ve == nep:
            print(f"  [BC-descent] {ve}/{nep} | train={len(aobs_train)} eval={len(aobs_eval)} | 成功={succ/max(ve,1)*100:.0f}%")

    env.close()
    if len(aobs_train) < 200:
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
                noise[:, 2] *= 0.3
                act_b = torch.where(mask.unsqueeze(1), noise * acc_max_xy, act_b)
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
            print(f"  Ep{e+1:3d}/{nepoch} train={tl:.5f} eval={el:.5f} eps={eps:.3f} pat={patience_count}/{patience}")
            if el < best_eval_loss - 1e-5:
                best_eval_loss = el
                best_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                patience_count = 0
            else: patience_count += 1
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
            if isinstance(v, dict) and k in config: config[k].update(v)
            else: config[k] = v
    set_global_seed(config["train"].get("seed", 42))
    os.makedirs(log_dir, exist_ok=True)

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
    parser = argparse.ArgumentParser(description="三阶段 RL 训练 v3")
    parser.add_argument("--phase", type=str, required=True, choices=["lift", "cruise", "descent"])
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bc-ckpt", type=str, default=None)
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--no-curriculum", action="store_true", help="禁用课程学习")
    parser.add_argument("--obstacles", type=int, default=None)
    args = parser.parse_args()

    ld = args.log_dir or f"saves/{args.phase}_{args.algo}"
    cc = {}
    if args.render: cc.setdefault("sim", {})["render"] = True
    if args.gpu != 0: cc.setdefault("train", {})["gpu_id"] = args.gpu
    if args.timesteps: cc.setdefault("train", {})["total_timesteps"] = args.timesteps
    cc.setdefault("train", {})["seed"] = args.seed
    if args.wind: cc.setdefault("wind", {})["enabled"] = True
    if args.curriculum: cc.setdefault("curriculum", {})["enabled"] = True
    if args.no_curriculum: cc.setdefault("curriculum", {})["enabled"] = False
    if args.obstacles is not None:
        cc.setdefault("curriculum", {})["obstacle_enabled"] = False
        cc.setdefault("scene", {})["n_obstacles"] = args.obstacles
    train(args.phase, ld, algo=args.algo, custom_config=cc or None,
          bc_ckpt=args.bc_ckpt, skip_bc=args.skip_bc)