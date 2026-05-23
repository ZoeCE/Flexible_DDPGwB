# ==============================================================================
# train_phase.py — 三阶段独立训练框架 v9 (BC 完全移除版)
#
# v9 重构核心:
#   1. [BC 完全移除] cruise 残差 actor 在 PPOPhaseAgent.__init__ 中自动零初始化,
#      pretrain_bc_cruise / pretrain_bc_descent 函数和相关 CLI 选项全部删除.
#      来源: Jeon et al. 2025 (Residual MPC), Ankile et al. 2024 (ResiP).
#   2. 缩小 cruise 残差幅度: 0.25 → 0.08 (base 31% → 13%, 符合文献最佳实践)
#   3. 缩小课程扰动: wind 2N → 0.5N, 三类噪声 ÷3 (匹配 RL 物理修正能力)
#   4. Descent 课程 6 级 → 5 级 (跳变 3× 改 2×), 加入倒退机制 (SR < 20% 回退一级)
#   5. wandb 日志精简: PPO 训练 + 课程 + 训练表现 三类
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
from collections import deque

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from phase_agent import (
    PPOPhaseAgent, SACPhaseAgent,
    build_lift_obs, build_cruise_obs, build_descent_obs,
    build_wind_obs, CableEncoder,
    PPO_ZERO, SAC_ZERO,
)
from phase_reward import (
    compute_lift_reward, compute_cruise_reward, compute_descent_reward,
    LiftRewardState, CruiseRewardState, DescentRewardState,
    RewardComponentTracker,
)
from ee_acc_controller import EEAccController, CruiseZYawPID, SwingDampingController
from stability_metrics import StabilityMetrics  # [v12.6] 训练时输出细粒度评估指标

import mujoco
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# 工具
# ==============================================================================

def set_global_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpisodeStats:
    def __init__(self, window=50):
        self.window = window
        self._data = {}
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._data:
                self._data[k] = deque(maxlen=self.window)
            self._data[k].append(float(v))
    def mean(self, key):
        d = self._data.get(key)
        return float(np.mean(d)) if d else 0.0
    def success_rate(self):
        return self.mean("success")


class Logger:
    """轻量 wandb 包装器, 容错: 没装 wandb 也能跑。"""
    def __init__(self, log_dir, project="phase_rl", run_name=None):
        self._wandb = None; self.log_dir = log_dir
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
                    if isinstance(v, dict): _f(v, key + "/")
                    else: flat[key] = v
            _f(cfg)
            try: self._wandb.config.update(flat, allow_val_change=True)
            except Exception: pass
    def log(self, step, metrics):
        if self._wandb:
            try: self._wandb.log(metrics, step=step)
            except Exception: pass
    def close(self):
        if self._wandb:
            try: self._wandb.finish()
            except Exception: pass


def save_checkpoint(agent, log_dir, episode, tag=""):
    fname = f"ckpt_{tag}.pt" if tag else f"ckpt_ep{episode}.pt"
    path = os.path.join(log_dir, fname)
    agent.save(path)
    agent.save(os.path.join(log_dir, "ckpt_latest.pt"))
    return path


# ==============================================================================
# Reward / Obs 调度
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


def build_phase_obs(phase, env_obs, env, start_xy, target_xy, prev_tilt, prev_yaw,
                    wind_obs=None, base_action=None):
    """[v14.0] 返回 (core_obs, cable_raw, wind_obs, tilt, yaw)."""
    if phase == "lift":
        return build_lift_obs(env_obs, env, start_xy, prev_tilt, prev_yaw, wind_obs)
    elif phase == "cruise":
        return build_cruise_obs(env_obs, env, target_xy, prev_tilt, prev_yaw, wind_obs)
    elif phase == "descent":
        return build_descent_obs(env_obs, env, target_xy, prev_tilt, prev_yaw,
                                 wind_obs, base_action=base_action)
    raise ValueError(f"Unknown phase: {phase}")


# ==============================================================================
# 课程管理器 [v8] 统一噪声/风力 + descent 初始化精度
# ==============================================================================

class CurriculumManager:
    """
    统一课程管理器, 每个阶段独立 level 计数.

    每个 level 注入 4 类扰动:
      - obs_noise:   normalize_obs 后的高斯噪声 σ
      - act_noise:   delta_q 上的高斯噪声 σ (rad/step)
      - force_noise: payload 上随机方向力 σ (N)
      - wind_max:    风力上限 (N), 每 episode 从 [0, wind_max] 均匀采样

    descent 额外字段: init_xy / init_vel / init_tilt / xy_tol

    晋级条件:
      (sr >= sr_threshold) AND (eps_at_level >= min_eps)
      OR  eps_at_level >= hard_cap_eps (防卡死)

    [v14.2] 连续课程模式:
      wind_max 不再跳变, 而是在每个 episode 根据 SR 平滑增长:
      - 当 SR >= sr_threshold: wind_max 按 ramp_rate 线性增长
      - 当 SR < sr_threshold: wind_max 暂停增长 (等待 RL 适应)
      - wind_max 不会回退 (避免 PPO on-policy 分布混乱)
      旧 level 结构仅用于 descent 的精度参数, wind 完全由连续 ramp 控制
    """
    def __init__(self, config, phase):
        cur = config.get("curriculum", {})
        self.enabled = bool(cur.get("enabled", True))
        self.phase   = phase

        self.levels      = list(cur.get(f"{phase}_levels", []))
        self.sr_thresh   = float(cur.get(f"{phase}_sr_threshold", 0.7))
        self.min_eps     = int(cur.get(f"{phase}_min_eps",        150))
        self.hard_cap    = int(cur.get(f"{phase}_hard_cap_eps",   1000))
        self.stats_win   = int(cur.get(f"{phase}_stats_window",   50))

        # [v9] 倒退机制 (连续模式下无效, 但保留接口)
        self.regression_enabled = False  # [v14.2] 连续模式不需要倒退
        self.regression_sr_thresh = float(cur.get(f"{phase}_regression_sr_threshold", 0.20))
        self.regression_min_eps   = int(cur.get(f"{phase}_regression_min_eps",         50))
        self.regression_stay_min_eps = int(cur.get(
            f"{phase}_regression_stay_min_eps", 0))
        self._regressed_recently = False

        self.level_idx    = 0
        self.eps_at_level = 0
        self._sr_window   = deque(maxlen=self.stats_win)

        # 上一 episode 采样的风力 (供日志)
        self._last_wind_force = 0.0
        self._last_wind_dir   = 0.0

        # ── [v14.2] 连续课程参数 ─────────────────────────────────────────────
        # wind_max 从 levels[0] 的值开始, 按 ramp_rate 每 episode 线性增长
        _wind_levels = [float(lv.get("wind_max", 0.0)) for lv in self.levels] if self.levels else [0.0]
        _configured_min = _wind_levels[0] if _wind_levels else 0.0
        self._wind_min  = max(_configured_min, float(cur.get(
            f"{phase}_ramp_min_wind", _configured_min)))
        self._wind_max  = _wind_levels[-1] if _wind_levels else 0.0
        self._wind_cur  = self._wind_min   # 当前 wind_max (平滑增长)
        # ramp_rate: 每成功 episode 增加的 wind_max 幅度
        # 设计: 从 min 到 max 需要 ~500-800 成功 episode
        _total_range = self._wind_max - self._wind_min
        _target_success_eps = int(cur.get(f"{phase}_ramp_episodes", 600))
        self._wind_ramp_rate = _total_range / max(_target_success_eps, 1)
        # SR 低于此值时暂停增长
        self._ramp_sr_thresh = float(cur.get(f"{phase}_ramp_sr_threshold",
                                              self.sr_thresh * 0.8))
        self._ramp_warmup_sr_thresh = float(cur.get(
            f"{phase}_ramp_warmup_sr_threshold", self._ramp_sr_thresh))
        self._ramp_warmup_scale = float(cur.get(
            f"{phase}_ramp_warmup_scale", 0.0))
        # 最低 SR 时减速 (不回退, 但增速降到 0)
        self._ramp_pause_count = 0  # 连续暂停的 episode 数

    # ── 当前 level 信息 ──────────────────────────────────────────────────────
    @property
    def n_levels(self):
        return max(1, len(self.levels))
    @property
    def current_level(self):
        if not self.levels:
            return {"obs_noise": 0.0, "act_noise": 0.0, "force_noise": 0.0, "wind_max": 0.0}
        # [v14.2] 用连续 wind_cur 替换 level 中的 wind_max
        idx = min(self.level_idx, len(self.levels) - 1)
        lvl = dict(self.levels[idx])  # copy
        lvl["wind_max"] = self._wind_cur
        return lvl
    @property
    def is_max_level(self):
        return self.level_idx >= len(self.levels) - 1
    @property
    def wind_current(self):
        """[v14.2] 当前连续 wind_max 值."""
        return self._wind_cur

    # ── 每 episode 采样 ──────────────────────────────────────────────────────
    def sample_episode_perturbations(self):
        if not self.enabled:
            return {"obs_noise": 0.0, "act_noise": 0.0,
                    "force_noise": 0.0, "wind_force": 0.0, "wind_dir": 0.0}
        lvl = self.current_level
        # [v14.2] wind_max 来自连续 ramp
        wind_max = self._wind_cur
        wf = float(np.random.uniform(0.0, wind_max)) if wind_max > 0 else 0.0
        wd = float(np.random.uniform(0.0, 2 * np.pi))
        self._last_wind_force = wf
        self._last_wind_dir   = wd
        return {
            "obs_noise":   float(lvl.get("obs_noise",   0.0)),
            "act_noise":   float(lvl.get("act_noise",   0.0)),
            "force_noise": float(lvl.get("force_noise", 0.0)),
            "wind_force":  wf, "wind_dir":  wd,
        }

    # ── descent 专属: 初始化精度参数 ────────────────────────────────────────
    def get_descent_init(self):
        if self.phase != "descent" or not self.levels:
            return None
        lvl = self.current_level
        return {
            "xy_range":   float(lvl.get("init_xy",   0.030)),
            "vel_range":  float(lvl.get("init_vel",  0.020)),
            "tilt_range": float(lvl.get("init_tilt", 0.008)),
            "xy_tol":     float(lvl.get("xy_tol",    0.005)),
        }

    # ── [v14.2] 连续推进逻辑 ────────────────────────────────────────────────
    def update(self, success):
        """每 episode 末调用. 连续课程: 根据 SR 平滑增长 wind_max."""
        if not self.enabled:
            return False
        self.eps_at_level += 1
        self._sr_window.append(float(success))

        # 统计 SR
        if len(self._sr_window) >= min(20, self.stats_win):
            sr = float(np.mean(self._sr_window))
        else:
            sr = 0.5  # 初始窗口不够时假定中等 SR

        # ── 连续 wind ramp ──
        ramp_scale = 0.0
        if sr >= self._ramp_sr_thresh:
            ramp_scale = 1.0
        elif sr >= self._ramp_warmup_sr_thresh:
            ramp_scale = max(0.0, self._ramp_warmup_scale)

        if ramp_scale > 0.0:
            self._wind_cur = min(
                self._wind_cur + self._wind_ramp_rate * ramp_scale,
                self._wind_max)
            self._ramp_pause_count = 0
        else:
            # SR 太低 → 暂停增长 (不回退)
            self._ramp_pause_count += 1

        # ── 阶梯 level 推进 (仅用于 descent 精度参数) ──
        # 根据 wind_cur 位置推断当前应属哪个 level
        if self.levels:
            _wind_levels = [float(lv.get("wind_max", 0.0)) for lv in self.levels]
            _new_idx = 0
            for j, wl in enumerate(_wind_levels):
                if self._wind_cur >= wl * 0.9:  # 90% 阈值
                    _new_idx = j
            if _new_idx > self.level_idx:
                self.level_idx = _new_idx
                self._just_promoted = True
                print(f"  [Curriculum-{self.phase}] → L{self.level_idx} "
                      f"(wind={self._wind_cur:.3f}N, SR={sr:.0%})")

        # 定期打印
        if self.eps_at_level % 100 == 0:
            print(f"  [Curriculum-{self.phase}] ep={self.eps_at_level} "
                  f"wind={self._wind_cur:.3f}/{self._wind_max:.1f}N "
                  f"SR={sr:.0%} pause={self._ramp_pause_count}")

        return False

    def consume_promotion_flag(self):
        flag = getattr(self, '_just_promoted', False)
        self._just_promoted = False
        return flag

    def _regress(self, reason):
        """[v9] 保留接口兼容性, 连续模式下不使用."""
        self._just_regressed = True
        self._regressed_recently = True
        return True

    def consume_regression_flag(self):
        """[v10] 训练循环每 episode 末调用, 如果刚倒退则返回 True。"""
        flag = getattr(self, '_just_regressed', False)
        self._just_regressed = False
        return flag

    # ── 日志辅助 ─────────────────────────────────────────────────────────────
    def info(self):
        """返回当前课程信息字典 (供 wandb 日志)。"""
        lvl = self.current_level
        sr = float(np.mean(self._sr_window)) if self._sr_window else 0.0
        return {
            f"cur/{self.phase}/level":       self.level_idx,
            f"cur/{self.phase}/eps_at_lvl":  self.eps_at_level,
            f"cur/{self.phase}/sr_window":   sr,
            f"cur/{self.phase}/obs_noise":   float(lvl.get("obs_noise",   0.0)),
            f"cur/{self.phase}/act_noise":   float(lvl.get("act_noise",   0.0)),
            f"cur/{self.phase}/force_noise": float(lvl.get("force_noise", 0.0)),
            f"cur/{self.phase}/wind_max":    float(lvl.get("wind_max",    0.0)),
            f"cur/{self.phase}/last_wind":   self._last_wind_force,
        }


# ==============================================================================
# 三阶段独立物理初始化 (移除 cruise_dist_curriculum, ORCA 相关)
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


def reset_for_phase(env, phase, config,
                    override_init_xy_range=None,
                    override_init_vel_range=None,
                    override_init_tilt_range=None):
    """统一三阶段物理初始化 (移除 cruise_dist_curriculum, ORCA)。"""
    max_xml_retries = 5
    obs = None; planned_path = None
    for _xml_retry in range(max_xml_retries):
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs = env.reset()
            planned_path = env.get_planned_path()
            sys.stdout.close(); sys.stdout = old_stdout
            break
        except (ValueError, RuntimeError) as _xml_err:
            sys.stdout.close(); sys.stdout = old_stdout
            _err_str = str(_xml_err)
            if any(s in _err_str.lower() for s in ["empty file", "xml error", "xml"]):
                if _xml_retry < max_xml_retries - 1:
                    time.sleep(0.05 * (_xml_retry + 1)); continue
            raise
        except Exception:
            sys.stdout.close(); sys.stdout = old_stdout; raise

    if obs is None or planned_path is None:
        return None, None

    phase_cfg = config.get(f"{phase}_rl", {})
    rng = np.random.default_rng()

    if phase == "lift":
        _xy_range  = override_init_xy_range if override_init_xy_range is not None \
                     else float(phase_cfg.get("init_xy_range", 0.01))
        _z_range   = float(phase_cfg.get("init_z_range",  0.01))
        _vel_range = float(phase_cfg.get("init_vel_range", 0.0))

        rope_L      = float(config["controller"].get("L", 0.5))
        seed_q      = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_pref_z = float(config["reset"]["init_qpos_prefab"][2])

        pref_jnt  = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx   = env.model.jnt_dofadr[pref_jnt]

        base_xy = env.data.qpos[qpos_addr:qpos_addr+2].copy()
        base_z  = float(env.data.qpos[qpos_addr+2])
        noise_xy = rng.uniform(-_xy_range, _xy_range, 2)
        noise_z  = rng.uniform(-_z_range,  _z_range)
        new_pref_xy = base_xy + noise_xy
        new_pref_z  = float(np.clip(base_z + noise_z, init_pref_z * 0.5, init_pref_z * 1.5))
        ee_z_target = new_pref_z + rope_L
        init_q = env.ik_solver.solve_4d(
            seed_q, float(new_pref_xy[0]), float(new_pref_xy[1]),
            ee_z_target, 0.0)
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q
        env.data.qpos[qpos_addr]   = new_pref_xy[0]
        env.data.qpos[qpos_addr+1] = new_pref_xy[1]
        env.data.qpos[qpos_addr+2] = new_pref_z
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        for _ in range(60):
            env.data.qpos[:7] = init_q
            env.data.qvel[:7] = 0.0
            env.data.qpos[qpos_addr]   = new_pref_xy[0]
            env.data.qpos[qpos_addr+1] = new_pref_xy[1]
            env.data.qpos[qpos_addr+2] = new_pref_z
            env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            env.data.qvel[dof_idx:dof_idx+6] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer: env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        if _vel_range > 0:
            env.data.qvel[dof_idx:dof_idx+2] += rng.uniform(-_vel_range, _vel_range, 2)
            mujoco.mj_forward(env.model, env.data)
        _sync_env_internal_state(env)
        obs = env._get_obs()

    elif phase == "cruise":
        # [v13.0 合并 lift] cruise 现在从低空 (z ≈ 0.11) 起步, 接管 lift 任务
        # NMPC 自动处理 lift→cruise 边界 (controller.py tracker 已识别 lift WP)
        z_cruise   = float(config["planning"]["payload_z_cruise"])
        init_pref_z = float(config["reset"]["init_qpos_prefab"][2])  # 默认 0.10
        start_xy   = env.default_start_xy.copy()
        rope_L = float(config["controller"].get("L", 0.5))
        # EE 起步: 比 payload 高一个 rope_L
        ee_z   = init_pref_z + rope_L
        seed_q = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_q = env.ik_solver.solve_4d(seed_q, float(start_xy[0]), float(start_xy[1]),
                                         ee_z, 0.0)
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt  = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx   = env.model.jnt_dofadr[pref_jnt]

        xy_range = override_init_xy_range if override_init_xy_range is not None \
                   else float(phase_cfg.get("init_xy_range", 0.01))
        noise_xy = rng.uniform(-xy_range, xy_range, 2)
        env.data.qpos[qpos_addr]   = start_xy[0] + noise_xy[0]
        env.data.qpos[qpos_addr+1] = start_xy[1] + noise_xy[1]
        env.data.qpos[qpos_addr+2] = init_pref_z   # [v13.0] 低空起步
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
            env.data.qpos[qpos_addr+2] = init_pref_z   # [v13.0]
            env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            env.data.qvel[dof_idx:dof_idx+6] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer: env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        vel_range = override_init_vel_range if override_init_vel_range is not None \
                    else float(phase_cfg.get("init_vel_range", 0.0))
        if vel_range > 0:
            env.data.qvel[dof_idx:dof_idx+2] += rng.uniform(-vel_range, vel_range, 2)
        mujoco.mj_forward(env.model, env.data)
        _sync_env_internal_state(env)
        obs = env._get_obs()

    elif phase == "descent":
        z_cruise = float(config["planning"]["payload_z_cruise"])
        target_xy = env.target_pos.copy()
        rope_L = float(config["controller"].get("L", 0.5))
        ee_z   = z_cruise + rope_L
        seed_q = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_q = env.ik_solver.solve_4d(seed_q,
            float(target_xy[0]), float(target_xy[1]), ee_z, 0.0)
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7] = init_q
        env.data.qvel[:7] = 0.0
        env.data.ctrl[:7] = init_q

        pref_jnt  = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx   = env.model.jnt_dofadr[pref_jnt]

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


def reset_for_descent_with_cur(env, config, cur):
    """Descent 段 reset, 含课程注入 + OmniReset 概率。"""
    cur_cfg = config.get("curriculum", {})
    omnireset_enabled = bool(cur_cfg.get("omnireset_enabled", True))
    near_prob = float(cur_cfg.get("omnireset_near_goal_prob", 0.10))

    desc_init = cur.get_descent_init() if cur is not None else None

    if omnireset_enabled and np.random.rand() < near_prob:
        near_xy       = float(cur_cfg.get("omnireset_near_goal_xy", 0.012))
        return reset_for_phase(env, "descent", config,
            override_init_xy_range=near_xy,
            override_init_vel_range=0.005,
            override_init_tilt_range=0.001)
    if desc_init is not None:
        return reset_for_phase(env, "descent", config,
            override_init_xy_range=desc_init["xy_range"],
            override_init_vel_range=desc_init["vel_range"],
            override_init_tilt_range=desc_init["tilt_range"])
    return reset_for_phase(env, "descent", config)


# ==============================================================================
# 专家辅助
# ==============================================================================

def _advance_expert_to_nearest_wp(expert, planned_path, pl_pos):
    if planned_path is None or len(planned_path) == 0:
        return
    dists = [np.linalg.norm(pl_pos - wp) for wp in planned_path]
    nearest_idx = int(np.argmin(dists))
    expert.tracker.current_idx = nearest_idx


def _truncate_path_for_lift(planned_path, config):
    """[v12.2] 为 lift phase 截短 path, 只保留垂直上升段.

    问题: full path = [lift wps (xy=start, z 渐升)] + [cruise wps (xy 移动, z=z_cruise)]
        + [descent wps]. NMPC tracker 有 10cm look-ahead, 会在 lift 段尚未完成时
        把参考点拉向 cruise xy → expert 提前平移 → lift 不能达到 cruise 高度 → SR=0.

    修复: 训练 lift 时只把 lift 段 + 一个虚拟"悬停"航点喂给 tracker.
        悬停航点 = (start_xy, z_cruise). NMPC 到达后会在该位置悬停, 不会平移.

    Args:
        planned_path: (N, 3) ndarray, env 生成的全段 path
        config: dict, 用于读取 z_cruise

    Returns:
        truncated_path: (M, 3) ndarray, M ≤ N, 只含 lift waypoints
    """
    if planned_path is None or len(planned_path) == 0:
        return planned_path
    pp = np.asarray(planned_path, dtype=np.float64)
    z_cruise = float(config.get("planning", {}).get("payload_z_cruise", 0.25))
    # lift 段定义: z 上升期 (z < z_cruise) + 第一个 z=z_cruise 的航点 (= lift target).
    # 之后的 cruise/descent 航点全部丢弃.
    keep_idx = []
    reached_cruise_z = False
    for i, wp in enumerate(pp):
        wp_z = float(wp[2]) if len(wp) >= 3 else 0.3
        if wp_z < z_cruise - 0.001:
            keep_idx.append(i)            # lift 渐升段
        elif not reached_cruise_z:
            keep_idx.append(i)            # 首个 cruise 高度航点 = lift target
            reached_cruise_z = True
        else:
            break                         # 后续 cruise 段不要
    if not keep_idx:
        # fallback: 至少保留首点
        return pp[:1].copy()
    return pp[keep_idx].copy()


def collect_expert_acc(expert, env, obs, current_q, phase, config):
    """收集专家 EE 加速度 (用于 BC 标签 [descent] 或 SAC warmup [cruise/descent])。"""
    if phase == "cruise":
        action_4d = expert.tracker.compute_ee_acceleration(obs, target_yaw=0.0)
        acc_max_xy = float(config["ee_control"].get("acc_max_xy", 0.8))
        return np.clip(
            np.array([float(action_4d[0]), float(action_4d[1])], dtype=np.float32),
            -acc_max_xy, acc_max_xy)
    elif phase == "descent":
        target_xy = env.target_pos.copy()
        target_pz = float(config["insertion"]["target_payload_z"])
        pl_pos = env.data.body('prefab').xpos.copy()
        pl_xy  = pl_pos[:2]; pl_z = float(pl_pos[2])
        dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
        pl_vel  = env.data.qvel[dof_idx:dof_idx+3].copy()
        acc_max_xy = float(config["descent_rl"].get(
            "residual_acc_max_xy", config["descent_rl"].get("acc_max_xy", 0.5)))
        acc_max_z  = float(config["descent_rl"].get(
            "residual_acc_max_z", config["descent_rl"].get("acc_max_z",  1.0)))
        diff_xy = target_xy - pl_xy
        dist_xy = float(np.linalg.norm(diff_xy))
        if dist_xy > 0.002:
            dir_xy   = diff_xy / dist_xy
            vel_proj = float(np.dot(pl_vel[:2], dir_xy))
            acc_mag  = 0.6 * min(dist_xy, 0.05) / 0.05 - 0.8 * vel_proj
            acc_xy   = dir_xy * np.clip(acc_mag, -1.0, 1.0) * acc_max_xy
        else:
            acc_xy = -pl_vel[:2] * 1.5
        acc_xy = np.clip(acc_xy, -acc_max_xy, acc_max_xy)
        align_factor = float(np.exp(-dist_xy / 0.01))
        z_error = pl_z - target_pz
        if z_error > 0.005 and align_factor > 0.3:
            acc_z = -(0.5 * min(z_error, 0.15) + 0.3 * max(float(pl_vel[2]), 0)) * align_factor
        else:
            acc_z = -float(pl_vel[2]) * 1.0
        acc_z = np.clip(acc_z, -acc_max_z, acc_max_z)
        return np.array([acc_xy[0], acc_xy[1], acc_z], dtype=np.float32)
    return np.zeros(config[f"{phase}_rl"]["action_dim"], dtype=np.float32)


# ==============================================================================
# Descent: PID base + RL residual delta_q 合并
# ==============================================================================

def _apply_descent_pid_residual(expert, rl_act, obs, env, config, current_q,
                                pid_dq=None):
    """Descent 段 PID + RL residual 动作合并.

    [v12.3 关键修复] 修复 RL authority 随 PID 收敛而消失的 bug:

    旧逻辑 (有 bug):
        max_residual_norm = residual_dq_scale * max(|pid_dq|, dq_max_avg * 0.1)
        # 当 pid_dq 很小 (PID 收敛到稳态时), max_residual_norm 也很小
        # → RL 残差只有 pid_dq 的 30%, 即 0.3-1mm
        # 但 descent 最后 1-2mm 的精度修正恰恰需要 RL 介入
        # → RL 没有"力量"做最后修正 → SR 卡在课程过渡处

    新逻辑 (v12.3):
        max_residual_norm = residual_dq_scale * dq_max_avg
        # 让 RL 残差 始终 有充分 authority (默认 30% × 0.12 = 0.036, 约 36mm)
        # 这样 RL 残差能在任何时候做出 ~5mm 级别的精度修正
        # 不会因 PID 收敛而 RL "失去能量"

    来源:
        Ankile et al. 2024 (ResiP, arXiv:2407.16677): residual 应有恒定且
        足够的 authority, 不能与 base 输出耦合.
        Alakuijala 2021 (arXiv:2106.08050): residual scale 应独立设定.
    """
    if pid_dq is None:
        try:
            pid_dq = expert.compute_delta_q_target(obs, current_q.astype(np.float64))
        except Exception:
            pid_dq = np.zeros(7, dtype=np.float32)
    pid_dq = np.asarray(pid_dq, dtype=np.float32)

    residual_dq_scale = float(config["descent_rl"].get("residual_dq_scale", 0.30))
    dq_max     = np.array(config["space"].get("dq_max", [0.12]*7), dtype=np.float32)
    acc_max_xy = float(config["descent_rl"].get(
        "residual_acc_max_xy", config["descent_rl"].get("acc_max_xy", 0.20)))
    acc_max_z  = float(config["descent_rl"].get(
        "residual_acc_max_z",  config["descent_rl"].get("acc_max_z",  0.40)))

    # rl_act 来自 actor, 范围 [-acc_max_xy, +acc_max_xy] (xy) 和 [-acc_max_z, +acc_max_z] (z)
    rl_norm_xy = rl_act[:2] / max(acc_max_xy, 1e-6)   # 归一化到 [-1, 1]
    rl_norm_z  = float(rl_act[2]) / max(acc_max_z, 1e-6) if len(rl_act) > 2 else 0.0

    # [v12.3 修复] max_residual_norm 与 PID 输出解耦
    # 旧: max_residual_norm = residual_dq_scale * max(|pid_dq|, dq_max_avg * 0.1)
    #     → 当 PID 收敛 (|pid_dq| 小) 时, max_residual_norm 也变小, RL 失去 authority
    # 新: max_residual_norm = residual_dq_scale * dq_max_avg
    #     → RL authority 与 PID 输出无关, 始终有足够"力量"做精度修正
    dq_max_avg = float(np.mean(dq_max))                          # 默认 0.12
    max_residual_norm = residual_dq_scale * dq_max_avg           # 默认 0.036

    rl_dq = np.zeros(7, dtype=np.float32)
    rl_dq[0] = rl_norm_xy[0] * max_residual_norm / np.sqrt(3)
    rl_dq[1] = rl_norm_xy[1] * max_residual_norm / np.sqrt(3)
    rl_dq[2] = rl_norm_z      * max_residual_norm / np.sqrt(3)
    rl_dq = np.clip(rl_dq, -dq_max * residual_dq_scale, dq_max * residual_dq_scale)

    dq_total = np.clip(pid_dq + rl_dq, -dq_max, dq_max)
    return dq_total, pid_dq


# ==============================================================================
# 噪声注入辅助
# ==============================================================================

def _add_obs_noise(norm_obs, sigma):
    if sigma <= 0: return norm_obs
    return (norm_obs + np.random.normal(0, sigma, norm_obs.shape).astype(np.float32))

def _add_act_noise(dq, sigma):
    if sigma <= 0: return dq
    return (dq + np.random.normal(0, sigma, dq.shape).astype(dq.dtype))


def _apply_episode_wind_env(env, pert):
    wf = float(pert.get("wind_force", 0.0))
    wd = float(pert.get("wind_dir", 0.0))
    if wf > 0.0 and hasattr(env, 'set_wind_force'):
        env.set_wind_force(wf, wd)
    else:
        if hasattr(env, 'clear_wind_force'):
            env.clear_wind_force()
        elif hasattr(env, 'set_wind_force'):
            env.set_wind_force(0.0, 0.0)
        else:
            try: env.set_wind_curriculum(0.0)
            except Exception: pass


def _apply_episode_wind_vec(vec, idx, pert):
    wf = float(pert.get("wind_force", 0.0))
    wd = float(pert.get("wind_dir", 0.0))
    if wf > 0.0:
        vec.set_wind_force(idx, wf, wd)
    else:
        if hasattr(vec, 'clear_wind_force'):
            vec.clear_wind_force(idx)
        else:
            try: vec.set_wind_curriculum(idx, 0.0)
            except Exception: pass


# ==============================================================================
# PPO 训练
# ==============================================================================

def make_phase_env_and_controllers(phase, config, worker_id=0):
    """[v11 Path 3] 工厂函数: 构造 env + 所有 base controllers (按 phase).

    供单进程 train_ppo / SubprocVecEnv worker 共用. 每次调用返回独立实例.

    Args:
        phase:     "lift" / "cruise" / "descent"
        config:    完整 config dict (deepcopy 后)
        worker_id: 用于 seed 偏移 (0 = main, >0 = subproc worker)
    Returns:
        (env, controllers_dict, config) — controllers 中含 'phase' 键, worker 可读
    """
    import copy as _copy
    cfg = _copy.deepcopy(config) if worker_id > 0 else config
    if worker_id > 0:
        cfg["train"]["seed"] = int(cfg["train"].get("seed", 42)) + worker_id * 1000
        set_global_seed(cfg["train"]["seed"])
    env = CableRobotEnvWithObstacles(config=cfg)
    controllers = {
        "phase":   phase,                       # [v11 fix] worker 读取 phase 用
        "expert":  JointSpaceExpert(cfg, env.ik_solver),
        "ee_ctrl": EEAccController(cfg, env.ik_solver),
        "z_pid":   CruiseZYawPID(cfg)          if phase == "cruise" else None,
        "swing_d": SwingDampingController(cfg) if phase == "cruise" else None,
    }
    return env, controllers, cfg


def train_ppo(phase, log_dir, config, resume_ckpt=None):
    """PPO 训练入口. [v11] n_envs > 1 自动 dispatch 到 train_ppo_vec."""
    n_envs = int(config["train"].get("n_envs", 1))
    if n_envs > 1:
        # [v11 Path 3] 多环境并行
        return train_ppo_vec(phase, log_dir, config, resume_ckpt=resume_ckpt,
                             n_envs=n_envs)
    return _train_ppo_single(phase, log_dir, config, resume_ckpt=resume_ckpt)


def _train_ppo_single(phase, log_dir, config, resume_ckpt=None):
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W  = int(config["train"]["log_smooth_win"])

    print(f"\n{'='*60}\n  PPO | {phase.upper()} | {T} steps | {log_dir}\n{'='*60}\n")

    env    = CableRobotEnvWithObstacles(config=config)
    agent  = PPOPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    ectl   = EEAccController(config, env.ik_solver)
    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    _cruise_nmpc_base = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get("use_nmpc_base", False)))
    _descent_pid_residual = (phase == "descent" and
        bool(config.get("descent_rl", {}).get("pid_residual_mode", True)))
    # [v12] lift 也支持 NMPC base + RL 残差
    _lift_nmpc_base = (phase == "lift" and
        bool(config.get("lift_rl", {}).get("use_nmpc_base", False)))

    cur = CurriculumManager(config, phase)

    # [v9] resume_ckpt 用于断点续训, 不再用于加载 BC 权重
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed from ckpt: {resume_ckpt}")

    # [v11 Path 2] HER for PPO descent
    _ppo_her_enabled = (phase == "descent" and bool(
        config.get("descent_rl", {}).get("ppo_her_enabled", False)))
    her_recorder = None
    if _ppo_her_enabled:
        from phase_agent import HEREpisodeRecorder
        max_steps = int(config["descent_rl"].get("max_steps", 500))
        her_recorder = HEREpisodeRecorder(agent.action_dim, max_steps=max_steps)
        _her_xy_tol_relabel = float(config["descent_rl"].get(
            "ppo_her_xy_tol_relabel", 0.015))
        _her_min_disp = float(config["descent_rl"].get(
            "ppo_her_min_displacement", 0.005))
        _her_max_eps = int(config["descent_rl"].get("ppo_her_max_episodes", 5))
        _her_eps_added = 0
        _her_xy_align_coef = float(config["descent_rl"]["reward"].get(
            "xy_align_coef", 4.0))
        _her_success_bonus = float(config["descent_rl"]["reward"].get(
            "success_bonus", 50.0))
        print(f"\n  [v11 Path 2] PPO-HER 启用 (descent): "
              f"tol={_her_xy_tol_relabel*1000:.0f}mm, max_eps_per_rollout={_her_max_eps}")

    logger = Logger(log_dir, project=f"phase_rl_v9", run_name=f"{phase}_ppo")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time()
    _ppo_update_count = 0

    lf = os.path.join(log_dir, f"{phase}_ppo_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                "sr", "steps", "pol_loss", "val_loss", "ent",
                                "lvl", "wind", "wind_max", "cur_sr", "cur_eps"])

    while ts < T:
        # ── 每 episode: 采样课程扰动 ───────────────────────────────────────────
        pert = cur.sample_episode_perturbations()

        # ── 物理初始化 ────────────────────────────────────────────────────────
        if phase == "descent":
            obs, pp = reset_for_descent_with_cur(env, config, cur)
        else:
            obs, pp = reset_for_phase(env, phase, config)
        if obs is None:
            continue
        env.set_force_noise(pert["force_noise"])
        _apply_episode_wind_env(env, pert)

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            # [v12.2] lift 时只把"lift 段"喂给 tracker, 防止 look-ahead 跨段拉走 xy
            _pp_for_expert = _truncate_path_for_lift(pp, config) if phase == "lift" else pp
            expert.set_path(_pp_for_expert)
            if phase != "lift":
                plp = env.data.body('prefab').xpos.copy()
                _advance_expert_to_nearest_wp(expert, pp, plp)
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
        if phase == "lift" and hasattr(rs, 'start_xy'):
            rs.start_xy = env.data.body('prefab').xpos[:2].copy()
        if phase == "descent":
            _di = cur.get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = cur.level_idx
                rs.descent_n_levels      = cur.n_levels

        if hasattr(agent, 'reset_history'):
            agent.reset_history()

        rew_tracker = RewardComponentTracker(phase)
        # [v12.6] 细粒度 RL 评估指标 tracker
        stab = StabilityMetrics()
        er = 0.0; es = 0; suc = False; term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])

        # ── Episode 主循环 ───────────────────────────────────────────────────
        # [v14.0] 获取 wind obs (每 episode 更新一次, 因为 wind 可能在 step 间变化)
        _wind_cfg = config.get("wind_obs", {})
        _wf_max = float(_wind_cfg.get("wind_force_max", 2.0))

        rd = False
        while not rd:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            base_dq_for_obs = None
            if phase == "descent" and _descent_pid_residual:
                try:
                    base_dq_for_obs = expert.compute_delta_q_target(
                        obs, cq.astype(np.float64))
                except Exception:
                    base_dq_for_obs = np.zeros(7, dtype=np.float32)
            # [v14.0] 构建 obs: (core, cable_raw, wind, tilt, yaw)
            wobs = build_wind_obs(env, _wf_max)
            core, cable_raw, wobs, pt, py = build_phase_obs(
                phase, obs, env, sxy, txy, pt, py, wind_obs=wobs,
                base_action=base_dq_for_obs)
            # [v14.0] 通过 CableEncoder 编码后合并
            po = agent.encode_obs(core, cable_raw, wobs)
            no = agent.normalize_obs(po, update=True)
            no_noisy = _add_obs_noise(no, pert["obs_noise"])

            ree = env._get_ee_pos()

            # ── Cruise [v13.0 合并 lift]: 根据 payload 高度切换两种模式 ──────
            #   低空 (payload_z < z_cruise - 0.03): lift 模式 — expert.compute_delta_q_target
            #     (单一积分器路径, 与 test 一致, NMPC 自动垂直上升)
            #   高空 (payload_z 接近 z_cruise): cruise 模式 — 原 lock_z + xy 残差
            if phase == "cruise" and z_pid is not None:
                _pl_pos  = env.data.body('prefab').xpos
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                _pl_yaw  = float(_pl_euler[2])
                _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) \
                    if _dof_idx + 5 < len(env.data.qvel) else 0.0
                _z_cruise = float(config["cruise_rl"].get("target_z_cruise", 0.25))
                _is_lift_phase = float(_pl_pos[2]) < _z_cruise - 0.03

                # cruise 模式需要 z_pid 反馈, lift 模式不需要 (NMPC 自处理 z)
                if not _is_lift_phase:
                    _z_corr, _tgt_yaw, _falling = z_pid.compute(
                        float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                    if _falling:
                        rw = -5.0
                        if agent.use_lstm:
                            agent.obs_history.push(no_noisy)
                            agent.push_cable_raw(cable_raw)
                            val = agent.get_value_for_obs_sequence(
                                agent.obs_history.get_sequence())
                        else:
                            val = agent.get_value_for_state(no_noisy)
                        agent.add_to_buffer(no_noisy, np.zeros(agent.action_dim, np.float32),
                                            rw, 1.0, val, 0.0,
                                            cable_raw=cable_raw)
                        er += rw; es += 1; ts += 1; agent.total_steps = ts
                        rd = True; break
                else:
                    _z_corr, _tgt_yaw = 0.0, 0.0

                if swing_d is not None and not _is_lift_phase:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)

                act, lp, val = agent.act(no_noisy)
                _act_arr = np.asarray(act, np.float32)

                if _is_lift_phase:
                    # ── Lift 模式: 用 expert.compute_delta_q_target + 3D residual_acc
                    # 与 test_phase / v12.1 lift fix 完全一致路径
                    _rm_xy = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                    _rm_z  = float(config["cruise_rl"].get("residual_acc_max_z_rl",  0.10))
                    # action_dim=3, 索引 [0,1,2] = xy_residual + z_residual
                    _res3 = np.array([
                        float(np.clip(_act_arr[0], -_rm_xy, _rm_xy)),
                        float(np.clip(_act_arr[1], -_rm_xy, _rm_xy)),
                        float(np.clip(_act_arr[2], -_rm_z,  _rm_z)) if len(_act_arr) > 2 else 0.0,
                    ], np.float64)
                    dq = expert.compute_delta_q_target(obs, cq, residual_acc=_res3)
                else:
                    # ── Cruise 模式: NMPC 输出 + xy 残差 + lock_z
                    if _cruise_nmpc_base:
                        try:
                            _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                            _base = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                        except Exception:
                            _base = np.zeros(2, np.float32)
                        _res_max = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                        _rl_clip = np.clip(_act_arr[:2], -_res_max, _res_max)
                        _comb = _base + _rl_clip
                        _amax = float(config["cruise_rl"].get("residual_acc_max_xy", 0.60))
                        _cn = float(np.linalg.norm(_comb))
                        if _cn > _amax: _comb = _comb / _cn * _amax
                        a3 = np.array([_comb[0], _comb[1], 0.0])
                    else:
                        a3 = np.array([_act_arr[0], _act_arr[1], 0.0])

                    dq = ectl.compute_delta_q(
                        a3, cq, ree, lock_z=True,
                        z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                        z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                        base_acc_xy=None, residual_mode=False)

                dq = _add_act_noise(dq, pert["act_noise"])

                no2, _, _, _, ei = env.step(dq)
                # [v13.0] rl_action 传完整 3D 维度
                rw, dn, sc, ri = compute_cruise_reward(env, no2, config, rs,
                                                       tracker=rew_tracker,
                                                       rl_action=_act_arr)
                rew_tracker.step()

            # ── Descent: PID base + RL residual ──────────────────────────────
            elif phase == "descent" and _descent_pid_residual:
                act, lp, val = agent.act(no_noisy)
                dq, _pid_dq = _apply_descent_pid_residual(
                    expert, act, obs, env, config, cq,
                    pid_dq=base_dq_for_obs)

                dq = _add_act_noise(dq, pert["act_noise"])
                no2, _, _, _, ei = env.step(dq)
                # [v11.3] 传 rl_action 给 descent reward (action_magnitude/smoothness penalty)
                rw, dn, sc, ri = compute_descent_reward(env, no2, config, rs,
                                                        tracker=rew_tracker,
                                                        rl_action=act)
                rew_tracker.step()

            # ── Lift v12: NMPC vertical base + RL residual 3D ────────────────
            # [v12 fix] 用 expert.compute_delta_q_target(..., residual_acc=...)
            # 这样积分器/速度限制/锚定/IK 都和 test_phase expert-only 完全一致.
            # 修复了之前 expert.tracker + EEAccController 拼接的"双积分器漂移"问题.
            elif phase == "lift" and _lift_nmpc_base:
                act, lp, val = agent.act(no_noisy)
                # RL 残差 clip 到小范围 (Olesen 2026 推荐: ≤ base 的 10-20%)
                _rm_xy = float(config["lift_rl"].get("residual_acc_max_xy", 0.08))
                _rm_z  = float(config["lift_rl"].get("residual_acc_max_z",  0.10))
                _res3 = np.array([
                    float(np.clip(act[0], -_rm_xy, _rm_xy)),
                    float(np.clip(act[1], -_rm_xy, _rm_xy)),
                    float(np.clip(act[2], -_rm_z,  _rm_z)),
                ], np.float64)
                # 调用 expert (与 test_phase 同一函数, 仅多传 residual_acc)
                dq = expert.compute_delta_q_target(obs, cq, residual_acc=_res3)
                dq = _add_act_noise(dq, pert["act_noise"])
                no2, _, _, _, ei = env.step(dq)
                rw, dn, sc, ri = compute_lift_reward(env, no2, config, rs,
                                                     tracker=rew_tracker,
                                                     rl_action=act)
                rew_tracker.step()

            # ── Lift fallback: 纯 RL 输出 EE acc (旧架构, 不推荐) ──────────────
            else:  # phase == "lift" without NMPC base, or other fallback
                act, lp, val = agent.act(no_noisy)
                dq = ectl.compute_delta_q(act, cq, ree)
                dq = _add_act_noise(dq, pert["act_noise"])
                no2, _, _, _, ei = env.step(dq)
                rw, dn, sc, ri = compute_lift_reward(env, no2, config, rs,
                                                     tracker=rew_tracker,
                                                     rl_action=act)
                rew_tracker.step()

            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1:
                done = True
                if not ri.get("termination"):
                    if phase == "descent":
                        rw += float(config["descent_rl"]["reward"].get(
                            "timeout_penalty", 0.0))
                    ri["termination"] = "timeout"
            if sc: suc = True
            if done and ri.get("termination"):
                term_reason = ri["termination"]

            # [v12.6] 细粒度 RL 评估指标更新 (no2 是 env._get_obs() raw 输出)
            stab.update_step(no2, config, env=env, rl_action=act)

            # [v14.1] push cable_raw to history (for LSTM seq buffer)
            agent.push_cable_raw(cable_raw)
            agent.add_to_buffer(no_noisy, act, rw, float(done), val, lp,
                                cable_raw=cable_raw)

            # [v11 Path 2] descent 步骤记录到 HER recorder
            if her_recorder is not None and phase == "descent":
                _pl_pos_now = env.data.body('prefab').xpos
                _pl_xy_now = np.array([float(_pl_pos_now[0]),
                                       float(_pl_pos_now[1])], np.float32)
                # 从 tracker 取本步的 xy_align_reward (用于 relabel 时替换)
                _xy_align_r = float(rew_tracker.get_last_step_value("xy_align_reward")) \
                    if hasattr(rew_tracker, "get_last_step_value") else 0.0
                her_recorder.record(
                    raw_obs=po, norm_obs=no_noisy, action=act,
                    reward=rw, done=done, value=val, log_prob=lp,
                    pl_xy=_pl_xy_now, xy_align_reward=_xy_align_r)

            er += rw; es += 1; ts += 1; agent.total_steps = ts
            obs = no2
            agent._update_entropy_coef(global_ts=ts)

            if agent.buffer.full:
                if done:
                    lv = 0.0
                else:
                    # [v14.0] 重新构建 obs
                    _wobs2 = build_wind_obs(env, _wf_max)
                    _base2 = None
                    if phase == "descent" and _descent_pid_residual:
                        try:
                            _cq2 = env.data.qpos[:7].copy().astype(np.float32)
                            _base2 = expert.compute_delta_q_target(
                                no2, _cq2.astype(np.float64))
                        except Exception:
                            _base2 = np.zeros(7, dtype=np.float32)
                    _core2, _cable2, _wobs2, _, _ = build_phase_obs(
                        phase, no2, env, sxy, txy, pt, py, wind_obs=_wobs2,
                        base_action=_base2)
                    ns_ = agent.encode_obs(_core2, _cable2, _wobs2)
                    nn_ = agent.normalize_obs(ns_, update=False)
                    nn_noisy = _add_obs_noise(nn_, pert["obs_noise"])
                    if agent.use_lstm:
                        seq = list(agent.obs_history.buffer)
                        seq.append(nn_noisy.copy())
                        while len(seq) > agent.seq_len:
                            seq.pop(0)
                        while len(seq) < agent.seq_len:
                            seq.insert(0, np.zeros(agent.obs_dim, dtype=np.float32))
                        lv = agent.get_value_for_obs_sequence(seq)
                    else:
                        lv = agent.get_value_for_state(nn_noisy)
                agent.buffer.compute_returns_and_advantages(
                    lv, agent.gamma, agent.gae_lambda)
                agent.update(global_ts=ts)
                _ppo_update_count += 1
                rd = True
            if done:
                rd = True

        # ── Episode 末: 更新统计 / 课程 / 日志 ───────────────────────────────
        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()
        cur.update(suc)
        # [v10] 课程倒退则清 Adam 状态, 避免死局轨迹累积的二阶矩阻碍恢复
        if cur.consume_regression_flag() and hasattr(agent, 'reset_adam_state'):
            agent.reset_adam_state(reason=f"{phase}_curriculum_regression")

        # [v14.0] 课程晋级时 entropy boost: 临时提高 entropy_coef 给新分布探索空间
        if cur.consume_promotion_flag():
            _boost_val = agent.entropy_coef_start * 0.6
            agent.entropy_coef = max(agent.entropy_coef, _boost_val)
            print(f"  [v14.0 entropy boost] Level {cur.level_idx}: "
                  f"entropy_coef → {agent.entropy_coef:.4f}")

        # [v11 Path 2] PPO-HER: 失败 episode 用 "final" 策略 relabel, 加入 buffer
        if (her_recorder is not None and phase == "descent"
                and not suc and her_recorder.has_data()
                and _her_eps_added < _her_max_eps
                and not agent.buffer.full):
            # 取最后一步的 payload xy 作为 relabel 的 achieved goal
            achieved_xy = her_recorder.pl_xy_history[-1].copy()
            relabeled = her_recorder.generate_relabeled_transitions(
                agent, achieved_xy,
                xy_tol_relabel=_her_xy_tol_relabel,
                xy_align_coef=_her_xy_align_coef,
                success_bonus=_her_success_bonus,
                min_displacement=_her_min_disp)
            if relabeled is not None:
                n_added = agent.ingest_her_transitions(relabeled)
                _her_eps_added += 1
                if ep % 20 == 0:
                    print(f"  [HER] Ep{ep} relabel: achieved=({achieved_xy[0]*100:.1f}, "
                          f"{achieved_xy[1]*100:.1f})cm, +{n_added} transitions")
        # 重置 HER recorder for 下一 episode
        if her_recorder is not None:
            her_recorder.reset()
        # rollout buffer 刚 update 完, HER 计数归零
        if her_recorder is not None and not agent.buffer.full and agent.buffer.ptr == 0:
            _her_eps_added = 0

        r = agent._last_result
        mark = "✅" if suc else "❌"
        pl_pos_now = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) \
            if phase in ("cruise", "descent") else 0.0

        cur_info = cur.info()
        cur_str = f"L{cur.level_idx}/{cur.n_levels-1} ep{cur.eps_at_level}"

        print(f"Ep{ep:4d} [{ts:7d}] {mark} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm "
              f"W:{pert['wind_force']:.3f}N "
              f"[{cur_str}] | {term_reason}")
        print(f"       PPO PL:{r.policy_loss:6.3f} VL:{r.value_loss:6.3f} "
              f"E:{r.entropy_loss:6.3f} KL:{r.approx_kl:.4f}")

        # ── wandb 日志 ──────────────────────────────────────────────────────
        # [v10] 诊断指标: actor std + RL action 量级
        try:
            _std_per_dim = agent.actor.get_log_std_per_dim()
            _actor_std_mean = float(np.exp(_std_per_dim).mean())
            _actor_std_max  = float(np.exp(_std_per_dim).max())
        except Exception:
            _actor_std_mean = _actor_std_max = 0.0

        log_metrics = {
            # 训练表现
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal_cm": dist_to_goal * 100,
            # PPO 训练指标
            "ppo/pol_loss":    r.policy_loss,
            "ppo/val_loss":    r.value_loss,
            "ppo/ent_loss":    r.entropy_loss,
            "ppo/approx_kl":   r.approx_kl,
            "ppo/clip_frac":   r.clip_fraction,
            "ppo/ent_coef":    r.entropy_coef_used,
            # [v10] 诊断指标
            f"diag/{phase}/actor_std_mean": _actor_std_mean,
            f"diag/{phase}/actor_std_max":  _actor_std_max,
        }
        # 课程指标
        log_metrics.update(cur_info)
        # reward 分项
        log_metrics.update(rew_tracker.episode_summary())
        # [v12.6] 细粒度 RL 评估指标 (anti-sway quality + RL intervention)
        log_metrics.update(stab.summary_for_wandb(phase, prefix="stab"))
        logger.log(ep, log_metrics)

        with open(lf, "a", newline="") as f:
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                    f"{r.policy_loss:.4f}", f"{r.value_loss:.4f}",
                                    f"{r.entropy_loss:.4f}",
                                    cur.level_idx, f"{pert['wind_force']:.4f}",
                                    f"{cur_info['cur/%s/wind_max' % phase]:.4f}",
                                    f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                                    cur.eps_at_level])
        if ep > 0 and ep % SI == 0:
            save_checkpoint(agent, log_dir, ep)
        if sr > best:
            best = sr
            save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-PPO] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    logger.close(); env.close()
    return agent


# ==============================================================================
# [v11 Path 3] PPO 训练 — 向量化版本 (SubprocVecEnv 并行)
# ==============================================================================

def train_ppo_vec(phase, log_dir, config, resume_ckpt=None, n_envs=4):
    """[v11 Path 3] PPO 训练 with SubprocVecEnv.

    每个 worker 子进程独立运行 env + base controllers (NMPC / PID / Z-PID 等).
    主进程负责: agent inference, PPO update, 课程, wandb, HER (descent).

    与 _train_ppo_single 的差异:
    - 每 step 通过 vec.rl_step(idx, payload) 远程调用 worker
    - n_envs 个 episode 同时进行 (各自独立 curriculum 状态/reward tracker)
    - rollout buffer 接受来自所有 envs 的 transitions

    注意:
    - HER 暂不支持 vec 模式 (n_envs > 1 时自动禁用)
    - 推荐 n_envs ≤ CPU 物理核数 - 1
    """
    from vec_env import make_vec_env
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W  = int(config["train"]["log_smooth_win"])
    start_method = config["train"].get("vec_env_start_method", "forkserver")

    print(f"\n{'='*60}\n  PPO [VEC n_envs={n_envs}] | {phase.upper()} | "
          f"{T} steps | {log_dir}\n{'='*60}\n")

    # ── 主进程: agent only, 不持有 env ────────────────────────────────────────
    agent = PPOPhaseAgent(phase, config=config)
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed: {resume_ckpt}")

    # Vec PPO uses per-env GAE. HER relabeling would need per-env next_value
    # reconstruction, so keep HER in single-env PPO and disable it here.
    _ppo_her_enabled = False
    her_recorders = None
    if _ppo_her_enabled:
        from phase_agent import HEREpisodeRecorder
        max_steps_her = int(config["descent_rl"].get("max_steps", 500))
        her_recorders = [HEREpisodeRecorder(agent.action_dim, max_steps=max_steps_her)
                         for _ in range(n_envs)]
        _her_xy_tol_relabel = float(config["descent_rl"].get(
            "ppo_her_xy_tol_relabel", 0.015))
        _her_min_disp = float(config["descent_rl"].get(
            "ppo_her_min_displacement", 0.005))
        _her_max_eps = int(config["descent_rl"].get("ppo_her_max_episodes", 5))
        _her_eps_added = 0
        _her_xy_align_coef = float(config["descent_rl"]["reward"].get(
            "xy_align_coef", 4.0))
        _her_success_bonus = float(config["descent_rl"]["reward"].get(
            "success_bonus", 50.0))
        print(f"  [v11 vec] PPO-HER 启用: per-worker recorders × {n_envs}, "
              f"tol={_her_xy_tol_relabel*1000:.0f}mm")

    # ── 启动 n_envs 个 worker (每个独立持有 env + controllers) ───────────────
    def _make_one(wid):
        return make_phase_env_and_controllers(phase, config, worker_id=wid)
    vec = make_vec_env(_make_one, n_envs=n_envs, start_method=start_method)

    # Vectorized training shares one curriculum so the ramp follows global
    # episode outcomes instead of eight slow per-worker windows.
    shared_cur = CurriculumManager(config, phase)
    curs = [shared_cur for _ in range(n_envs)]
    ep_rewards = [0.0] * n_envs; ep_steps = [0] * n_envs
    ep_suc     = [False] * n_envs
    ep_term    = ["running"] * n_envs
    obs_list   = [None] * n_envs
    sxy_list   = [None] * n_envs; txy_list = [None] * n_envs
    pt_list    = [0.0] * n_envs;  py_list  = [0.0] * n_envs
    rstate_list= [None] * n_envs
    obs_histories = [agent.make_obs_history() for _ in range(n_envs)]
    cable_histories = [agent.make_cable_history() for _ in range(n_envs)]

    # ── 课程 + 物理初始化 (每 worker) ─────────────────────────────────────────
    def _reset_one_env(i, max_retries=10):
        for _retry in range(max_retries):
            pert = curs[i].sample_episode_perturbations()
            # reset
            if phase == "descent":
                _di = curs[i].get_descent_init()
                if _di is not None:
                    obs, pp = vec.reset_for_descent(i, cur_init=_di)
                else:
                    obs, pp = vec.reset_for_descent(i, cur_init=None)
            else:
                obs, pp = vec.reset(i)
            if obs is not None:
                break  # reset 成功
        else:
            raise RuntimeError(f"Worker {i}: reset_for_phase 重试 {max_retries} 次仍失败")
        vec.set_force_noise(i, pert["force_noise"])
        _apply_episode_wind_vec(vec, i, pert)
        # [v11 KEY FIX] reset worker 内的 expert / ee_ctrl / z_pid.
        # 之前漏掉, 导致 NMPC tracker / PID 保留上 episode 状态, base action 错误,
        # 是 vec 模式 SR=0 的根本原因.
        cq = vec.get_qpos(i)
        vec.reset_controllers(i, obs, cq, pp)
        # start_xy / target_xy
        if phase == "lift":
            # [v12 fix] pp is path_3d (N, 3), pp[0] = [x, y, z]; need [x, y] only
            _pp_arr = np.asarray(pp, np.float32)
            sxy = np.array([float(_pp_arr[0, 0]), float(_pp_arr[0, 1])], np.float32)
            txy = sxy
        else:
            sxy = None
            tp_attr = vec.env_attr(i, "target_pos")
            txy = np.asarray(tp_attr, np.float32)[:2]
        # rstate: 按 phase 选用对应的 RewardState 类 ([v11 fix])
        rs = REWARD_STATES[phase]()
        if phase == "descent":
            _di = curs[i].get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = curs[i].level_idx
                rs.descent_n_levels      = curs[i].n_levels
        return obs, sxy, txy, rs, pert

    # 初始化所有 worker
    perts = [None] * n_envs
    for i in range(n_envs):
        obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
            _reset_one_env(i)
        pt_list[i] = 0.0; py_list[i] = 0.0
        obs_histories[i].reset(); cable_histories[i].clear()
        ep_rewards[i] = 0.0; ep_steps[i] = 0
        ep_suc[i] = False; ep_term[i] = "running"

    logger = Logger(log_dir, project=f"phase_rl_v11_vec", run_name=f"{phase}_ppo_vec")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ts = 0; ep_count = 0; t0 = time.time(); best = 0.0

    lf = os.path.join(log_dir, f"{phase}_ppo_vec_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                "sr", "steps", "pol_loss", "val_loss", "ent",
                                "lvl", "wind", "wind_max", "cur_sr", "cur_eps",
                                "worker_id"])

    try:
        # 首次为每个 worker 通过 worker 端 build_phase_obs (因为主进程没有 env)
        phase_obs_cache = [None] * n_envs
        for i in range(n_envs):
            phase_obs_cache[i] = vec.build_phase_obs_remote(
                i, phase, obs_list[i], sxy_list[i], txy_list[i],
                pt_list[i], py_list[i])

        while ts < T:
            # ── 主进程: 用缓存的 phase obs 做 RL inference ──────────────────
            actions_list = []; lps_list = []; vals_list = []; obs_noisy_list = []
            for i in range(n_envs):
                # [v15] phase_obs_cache[i] = (core, cable_raw, wind, tilt, yaw, base_dq)
                _core_i, _cable_i, _wind_i, _, _, _base_i = phase_obs_cache[i]
                po = agent.encode_obs(_core_i, _cable_i, _wind_i)
                no = agent.normalize_obs(po, update=True)
                no_noisy = _add_obs_noise(no, perts[i]["obs_noise"])
                act, lp, val = agent.act_with_history(
                    no_noisy, obs_histories[i])
                cable_histories[i].append(_cable_i.copy())
                obs_noisy_list.append(no_noisy)
                actions_list.append(act); lps_list.append(lp); vals_list.append(val)

            # ── 并发发送 rl_step 到所有 worker ────────────────────────────────
            payloads = []
            for i in range(n_envs):
                cq = vec.get_qpos(i)
                payloads.append({
                    'phase': phase,
                    'rl_action': actions_list[i],
                    'obs': obs_list[i],
                    'current_q': cq,
                    'start_xy': sxy_list[i],
                    'target_xy': txy_list[i],
                    'prev_tilt': pt_list[i],
                    'prev_yaw':  py_list[i],
                    'rstate':    rstate_list[i],
                    'act_noise': perts[i]["act_noise"],
                    'base_dq':    phase_obs_cache[i][5],
                })

            # 异步发送 (SubprocVecEnv); DummyVecEnv 内联
            if hasattr(vec, 'remotes'):
                for i, p in enumerate(payloads):
                    vec.remotes[i].send(('rl_step', p))
                results = [vec._check_recv(vec.remotes[i].recv(), i)
                           for i in range(n_envs)]
            else:
                results = [vec.rl_step(i, payloads[i]) for i in range(n_envs)]

            # ── 处理每个 env 的结果 ─────────────────────────────────────────
            for i, res in enumerate(results):
                rw = res['reward']; done = res['done']; suc_step = res['success']
                obs_list[i] = res['new_obs']
                rstate_list[i] = res['rstate']

                # 终止: episode 满 max_steps?
                mx = int(config[f"{phase}_rl"]["max_steps"])
                if ep_steps[i] >= mx - 1:
                    done = True
                    if res['termination'] == 'running':
                        if phase == "descent":
                            rw += float(config["descent_rl"]["reward"].get(
                                "timeout_penalty", 0.0))
                        res['termination'] = 'timeout'
                if suc_step: ep_suc[i] = True
                if done and res['termination']:
                    ep_term[i] = res['termination']

                # [v14.0] 加入 buffer (用刚才 RL inference 时的 phase obs)
                _core_prev, _cable_prev, _wind_prev, _, _, _ = phase_obs_cache[i]
                po_prev = agent.encode_obs(_core_prev, _cable_prev, _wind_prev)
                no_buf = agent.normalize_obs(po_prev, update=False)
                no_buf_noisy = obs_noisy_list[i]
                next_val = 0.0
                if (not done) and res.get('new_core_obs') is not None:
                    next_po = agent.encode_obs(
                        res['new_core_obs'], res['new_cable_raw'],
                        res['new_wind_obs'])
                    next_no = agent.normalize_obs(next_po, update=False)
                    next_no_noisy = _add_obs_noise(next_no, perts[i]["obs_noise"])
                    if agent.use_lstm:
                        seq = list(obs_histories[i].buffer)
                        seq.append(next_no_noisy.copy())
                        while len(seq) > agent.seq_len:
                            seq.pop(0)
                        while len(seq) < agent.seq_len:
                            seq.insert(0, np.zeros(agent.obs_dim, dtype=np.float32))
                        next_val = agent.get_value_for_obs_sequence(seq)
                    else:
                        next_val = agent.get_value_for_state(next_no_noisy)
                if not agent.buffer.full:
                    agent.add_to_buffer_with_history(
                        no_buf_noisy, actions_list[i], rw, float(done),
                        vals_list[i], lps_list[i],
                        obs_history=obs_histories[i],
                        cable_history=cable_histories[i],
                        cable_raw=_cable_prev,
                        next_value=next_val, env_id=i)

                # [v11 vec fix] HER recorder 记录本步
                if her_recorders is not None and phase == "descent":
                    _pl_xy_now = np.asarray(res.get('pl_xy', np.zeros(2)), np.float32)
                    _xy_align_r = float(res.get('xy_align_r', 0.0))
                    her_recorders[i].record(
                        raw_obs=po_prev, norm_obs=no_buf_noisy,
                        action=actions_list[i], reward=rw, done=done,
                        value=vals_list[i], log_prob=lps_list[i],
                        pl_xy=_pl_xy_now, xy_align_reward=_xy_align_r)

                ep_rewards[i] += rw; ep_steps[i] += 1; ts += 1
                agent.total_steps = ts

                # [v14.0] 更新 phase_obs_cache: 用 worker 返回的新 obs
                if not done:
                    phase_obs_cache[i] = (
                        res['new_core_obs'], res['new_cable_raw'],
                        res['new_wind_obs'], res['new_tilt'], res['new_yaw'],
                        res.get('new_base_dq'))
                    pt_list[i] = res['new_tilt']; py_list[i] = res['new_yaw']

                # ── Episode 结束: log + 重置 ──────────────────────────────
                if done:
                    # [v11 vec fix] HER: 失败 episode 用 'final' relabel
                    if (her_recorders is not None and phase == "descent"
                            and not ep_suc[i] and her_recorders[i].has_data()
                            and _her_eps_added < _her_max_eps
                            and not agent.buffer.full):
                        ach_xy = her_recorders[i].pl_xy_history[-1].copy()
                        relabeled = her_recorders[i].generate_relabeled_transitions(
                            agent, ach_xy,
                            xy_tol_relabel=_her_xy_tol_relabel,
                            xy_align_coef=_her_xy_align_coef,
                            success_bonus=_her_success_bonus,
                            min_displacement=_her_min_disp)
                        if relabeled is not None:
                            n_added = agent.ingest_her_transitions(relabeled)
                            _her_eps_added += 1
                            if ep_count % 20 == 0:
                                print(f"  [HER] w{i} Ep{ep_count} +{n_added} relabel")
                    # 重置该 worker 的 HER recorder
                    if her_recorders is not None:
                        her_recorders[i].reset()
                    # buffer 重置时也清 HER episode 计数器
                    if (her_recorders is not None and not agent.buffer.full
                            and agent.buffer.ptr == 0):
                        _her_eps_added = 0

                    stats.update(reward=ep_rewards[i], steps=ep_steps[i],
                                 success=float(ep_suc[i]))
                    curs[i].update(ep_suc[i])
                    if curs[i].consume_regression_flag() and hasattr(agent, 'reset_adam_state'):
                        agent.reset_adam_state(reason=f"{phase}_w{i}_regression")

                    ar = stats.mean("reward"); sr = stats.success_rate()
                    r = agent._last_result
                    mark = "✅" if ep_suc[i] else "❌"
                    cur_str = f"L{curs[i].level_idx}/{curs[i].n_levels-1} ep{curs[i].eps_at_level}"
                    print(f"[w{i}] Ep{ep_count:4d} [{ts:7d}] {mark} R:{ep_rewards[i]:6.2f}"
                          f"({ar:5.2f}) SR:{sr*100:4.0f}% S:{ep_steps[i]:3d} "
                          f"W:{perts[i]['wind_force']:.3f}N [{cur_str}] | {ep_term[i]}")

                    cur_info = curs[i].info()
                    try:
                        _std = agent.actor.get_log_std_per_dim()
                        _std_mean = float(np.exp(_std).mean())
                    except Exception:
                        _std_mean = 0.0
                    log_metrics = {
                        f"{phase}/reward":   ep_rewards[i],
                        f"{phase}/avg_reward": ar,
                        f"{phase}/sr":       sr,
                        f"{phase}/steps":    ep_steps[i],
                        "ppo/pol_loss":      r.policy_loss,
                        "ppo/val_loss":      r.value_loss,
                        "ppo/ent_loss":      r.entropy_loss,
                        "ppo/approx_kl":     r.approx_kl,
                        "ppo/clip_frac":     r.clip_fraction,
                        "ppo/ent_coef":      r.entropy_coef_used,
                        f"diag/{phase}/actor_std_mean": _std_mean,
                        "diag/vec/worker_id": i,
                    }
                    log_metrics.update(cur_info)
                    # [v12.6] vec 模式 stab: worker 在 done 时返回 stab_summary, 直接喂 wandb
                    _stab_sum = res.get('stab_summary')
                    if _stab_sum:
                        log_metrics.update({
                            f"stab/{phase}/{k}": v for k, v in _stab_sum.items()
                        })
                    logger.log(ep_count, log_metrics)

                    with open(lf, "a", newline="") as f:
                        csv.writer(f).writerow([ep_count, ts, f"{ep_rewards[i]:.3f}",
                            f"{ar:.3f}", f"{sr:.3f}", ep_steps[i],
                            f"{r.policy_loss:.4f}", f"{r.value_loss:.4f}",
                            f"{r.entropy_loss:.4f}",
                            curs[i].level_idx, f"{perts[i]['wind_force']:.4f}",
                            f"{cur_info['cur/%s/wind_max' % phase]:.4f}",
                            f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                            curs[i].eps_at_level, i])

                    ep_count += 1
                    if ep_count > 0 and ep_count % SI == 0:
                        save_checkpoint(agent, log_dir, ep_count)
                    if sr > best:
                        best = sr
                        save_checkpoint(agent, log_dir, ep_count, tag="best")

                    # 重置此 worker 的 env + 状态
                    obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                        _reset_one_env(i)
                    pt_list[i] = 0.0; py_list[i] = 0.0
                    obs_histories[i].reset(); cable_histories[i].clear()
                    ep_rewards[i] = 0.0; ep_steps[i] = 0
                    ep_suc[i] = False; ep_term[i] = "running"
                    # 重新 build phase obs (新 episode 起点)
                    phase_obs_cache[i] = vec.build_phase_obs_remote(
                        i, phase, obs_list[i], sxy_list[i], txy_list[i],
                        pt_list[i], py_list[i])

                agent._update_entropy_coef(global_ts=ts)

            # ── Buffer 满? 触发 PPO update ─────────────────────────────────
            if agent.buffer.full:
                agent.buffer.compute_returns_and_advantages(
                    0.0, agent.gamma, agent.gae_lambda)
                agent.update(global_ts=ts)
    finally:
        vec.close()
        try: logger.close()
        except Exception: pass

    save_checkpoint(agent, log_dir, ep_count, tag="final")
    print(f"\n[{phase.upper()}-PPO-VEC] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    return agent


# ==============================================================================
# [v11 Path 3] SAC 训练 — 向量化版本 (SubprocVecEnv 并行)
# ==============================================================================

def train_sac_vec(phase, log_dir, config, resume_ckpt=None, n_envs=4):
    """[v11 Path 3] SAC 训练 with SubprocVecEnv.

    设计:
    - N 个 worker 子进程独立运行 env + base controllers
    - 主进程: SAC agent (actor + 2 critics + replay buffer)
    - 每步: 主进程为 N 个 worker 各生成 1 action, 并发发送 rl_step,
            收回 N 个 transitions, 全部加入共享 replay buffer
    - 每 update_interval 步: SAC train_step (off-policy 优化)
    - Warmup: 用 actor 输出 + 高斯噪声 (cruise actor 零初始化, 起步即接近 NMPC)

    与 train_ppo_vec 的差异:
    - 不需要 phase_obs_cache (SAC act 是 stateless)
    - HER 仅在 descent 单 env 时支持; vec descent 仅记 episode-level HER (worker 末 flush)
    - update 频率独立于 buffer 满 (off-policy)
    """
    from vec_env import make_vec_env
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W  = int(config["train"]["log_smooth_win"])
    WU = int(config["sac"].get("warmup_steps", 5000))
    start_method = config["train"].get("vec_env_start_method", "forkserver")

    print(f"\n{'='*60}\n  SAC [VEC n_envs={n_envs}] | {phase.upper()} | "
          f"{T} steps | warmup={WU} | {log_dir}\n{'='*60}\n")

    # ── 主进程: 仅 agent ─────────────────────────────────────────────────────
    agent = SACPhaseAgent(phase, config=config)
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed: {resume_ckpt}")

    # ── 启动 worker (每个独立持有 env + controllers) ─────────────────────────
    def _make_one(wid):
        return make_phase_env_and_controllers(phase, config, worker_id=wid)
    vec = make_vec_env(_make_one, n_envs=n_envs, start_method=start_method)

    # Share one curriculum across vectorized workers for a true global ramp.
    shared_cur = CurriculumManager(config, phase)
    curs = [shared_cur for _ in range(n_envs)]
    ep_rewards = [0.0] * n_envs; ep_steps = [0] * n_envs
    ep_suc     = [False] * n_envs
    ep_term    = ["running"] * n_envs
    obs_list   = [None] * n_envs
    sxy_list   = [None] * n_envs; txy_list = [None] * n_envs
    pt_list    = [0.0] * n_envs;  py_list  = [0.0] * n_envs
    rstate_list= [None] * n_envs

    def _reset_one_env(i, max_retries=10):
        for _retry in range(max_retries):
            pert = curs[i].sample_episode_perturbations()
            if phase == "descent":
                _di = curs[i].get_descent_init()
                obs, pp = vec.reset_for_descent(i, cur_init=_di)
            else:
                obs, pp = vec.reset(i)
            if obs is not None:
                break
        else:
            raise RuntimeError(f"Worker {i}: reset_for_phase 重试 {max_retries} 次仍失败")
        vec.set_force_noise(i, pert["force_noise"])
        _apply_episode_wind_vec(vec, i, pert)
        # [v11 KEY FIX] reset worker 内的 expert / ee_ctrl / z_pid (同 PPO vec)
        cq = vec.get_qpos(i)
        vec.reset_controllers(i, obs, cq, pp)
        if phase == "lift":
            # [v12 fix] pp is path_3d (N, 3), pp[0] = [x, y, z]; need [x, y] only
            _pp_arr = np.asarray(pp, np.float32)
            sxy = np.array([float(_pp_arr[0, 0]), float(_pp_arr[0, 1])], np.float32)
            txy = sxy
        else:
            sxy = None
            tp_attr = vec.env_attr(i, "target_pos")
            txy = np.asarray(tp_attr, np.float32)[:2]
        # [v11 fix] 按 phase 选用对应的 RewardState 类
        rs = REWARD_STATES[phase]()
        if phase == "descent":
            _di = curs[i].get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = curs[i].level_idx
                rs.descent_n_levels      = curs[i].n_levels
        return obs, sxy, txy, rs, pert

    perts = [None] * n_envs
    for i in range(n_envs):
        obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
            _reset_one_env(i)
        pt_list[i] = 0.0; py_list[i] = 0.0

    logger = Logger(log_dir, project=f"phase_rl_v11_vec", run_name=f"{phase}_sac_vec")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ts = 0; ep_count = 0; t0 = time.time(); best = 0.0; onf = False
    update_interval = int(config["sac"].get("update_interval", 1))

    lf = os.path.join(log_dir, f"{phase}_sac_vec_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                "sr", "steps", "cl", "al", "alpha", "lvl", "wind",
                                "wind_max", "cur_sr", "cur_eps", "worker_id"])

    try:
        # 首轮 build phase obs (远程, 因主进程无 env)
        phase_obs_cache = [None] * n_envs
        for i in range(n_envs):
            phase_obs_cache[i] = vec.build_phase_obs_remote(
                i, phase, obs_list[i], sxy_list[i], txy_list[i],
                pt_list[i], py_list[i])

        while ts < T:
            # ── 主进程: 为每个 worker 生成 action ─────────────────────────────
            actions_list = []; obs_noisy_list = []
            for i in range(n_envs):
                # [v14.0]
                _core_i, _cable_i, _wind_i, _, _, _ = phase_obs_cache[i]
                po = agent.encode_obs(_core_i, _cable_i, _wind_i)
                no = agent.normalize_obs(po, update=not onf)
                no_noisy = _add_obs_noise(no, perts[i]["obs_noise"])
                if ts < WU:
                    # Warmup: actor zero-init + 较大噪声 (cruise residual ~0 + noise)
                    act_base = agent.act(no_noisy, deterministic=False)
                    noise = np.random.normal(0, 0.1, len(act_base)).astype(np.float32)
                    act = (act_base + noise).astype(np.float32)
                else:
                    act = agent.act(no_noisy, deterministic=False)
                obs_noisy_list.append(no_noisy)
                actions_list.append(act)
            if ts >= WU and not onf:
                agent._freeze_obs_norm = True; onf = True

            # ── 并发发送 rl_step ──────────────────────────────────────────────
            payloads = []
            for i in range(n_envs):
                cq = vec.get_qpos(i)
                payloads.append({
                    'phase': phase,
                    'rl_action': actions_list[i],
                    'obs': obs_list[i],
                    'current_q': cq,
                    'start_xy': sxy_list[i],
                    'target_xy': txy_list[i],
                    'prev_tilt': pt_list[i],
                    'prev_yaw':  py_list[i],
                    'rstate':    rstate_list[i],
                    'act_noise': perts[i]["act_noise"],
                    'base_dq':    phase_obs_cache[i][5],
                })

            if hasattr(vec, 'remotes'):
                for i, p in enumerate(payloads):
                    vec.remotes[i].send(('rl_step', p))
                results = [vec._check_recv(vec.remotes[i].recv(), i)
                           for i in range(n_envs)]
            else:
                results = [vec.rl_step(i, payloads[i]) for i in range(n_envs)]

            # ── 处理每个 worker 的 transition ─────────────────────────────────
            for i, res in enumerate(results):
                rw = res['reward']; done = res['done']; suc_step = res['success']
                obs_list[i] = res['new_obs']
                rstate_list[i] = res['rstate']

                mx = int(config[f"{phase}_rl"]["max_steps"])
                if ep_steps[i] >= mx - 1:
                    done = True
                    if res['termination'] == 'running':
                        res['termination'] = 'timeout'
                if suc_step: ep_suc[i] = True
                if done and res['termination']:
                    ep_term[i] = res['termination']

                # [v14.0] 计算 normalized obs: SAC remember 用 norm_obs + norm_next_obs
                _core_prev, _cable_prev, _wind_prev, _, _, _ = phase_obs_cache[i]
                po_prev = agent.encode_obs(_core_prev, _cable_prev, _wind_prev)
                no_prev_noisy = obs_noisy_list[i]
                if res.get('new_core_obs') is not None:
                    new_po = agent.encode_obs(
                        res['new_core_obs'], res['new_cable_raw'], res['new_wind_obs'])
                    new_no = agent.normalize_obs(new_po, update=False)
                    new_no = _add_obs_noise(new_no, perts[i]["obs_noise"])
                else:
                    new_no = no_prev_noisy  # falling 等异常 case
                pl_xy_now = res.get('pl_xy', np.zeros(2, np.float32))
                achieved_pos = np.array([pl_xy_now[0], pl_xy_now[1],
                                          float(config.get("insertion", {}).get(
                                              "target_payload_z", 0.10))], np.float32)
                agent.remember(no_prev_noisy, actions_list[i], new_no,
                               rw, float(done), achieved_pos=achieved_pos)

                ep_rewards[i] += rw; ep_steps[i] += 1; ts += 1
                agent.total_steps = ts

                if not done:
                    phase_obs_cache[i] = (
                        res['new_core_obs'], res['new_cable_raw'],
                        res['new_wind_obs'], res['new_tilt'], res['new_yaw'],
                        res.get('new_base_dq'))
                    pt_list[i] = res['new_tilt']; py_list[i] = res['new_yaw']

                # Episode 结束
                if done:
                    # HER flush (descent)
                    if phase == "descent":
                        agent.flush_episode_her(
                            txy_list[i],
                            float(config["insertion"]["target_payload_z"]))

                    stats.update(reward=ep_rewards[i], steps=ep_steps[i],
                                 success=float(ep_suc[i]))
                    curs[i].update(ep_suc[i])
                    # [v11 vec fix] 倒退时 reset Adam state, 同 PPO vec.
                    if curs[i].consume_regression_flag() and hasattr(agent, 'reset_adam_state'):
                        agent.reset_adam_state(reason=f"{phase}_w{i}_regression")

                    ar = stats.mean("reward"); sr = stats.success_rate()
                    r = agent._last_result
                    mark = "✅" if ep_suc[i] else "❌"
                    cur_str = f"L{curs[i].level_idx}/{curs[i].n_levels-1} ep{curs[i].eps_at_level}"
                    print(f"[w{i}] Ep{ep_count:4d} [{ts:7d}] {mark} R:{ep_rewards[i]:6.2f}"
                          f"({ar:5.2f}) SR:{sr*100:4.0f}% S:{ep_steps[i]:3d} "
                          f"W:{perts[i]['wind_force']:.3f}N [{cur_str}] | {ep_term[i]}")
                    print(f"       SAC CL:{r.critic_loss:.4f} AL:{r.actor_loss:.4f} "
                          f"α:{agent.alpha:.4f}")

                    cur_info = curs[i].info()
                    log_metrics = {
                        f"{phase}/reward":   ep_rewards[i],
                        f"{phase}/avg_reward": ar,
                        f"{phase}/sr":       sr,
                        f"{phase}/steps":    ep_steps[i],
                        "sac/critic_loss":   r.critic_loss,
                        "sac/actor_loss":    r.actor_loss,
                        "sac/alpha_loss":    r.alpha_loss,
                        "sac/alpha":         r.alpha,
                        "sac/q_mean":        r.q_mean,
                        # [v11.4] Q-divergence 早期检测
                        "sac/target_q_mean": getattr(agent, '_last_target_q', 0.0),
                        "sac/q_target_gap":  r.q_mean - getattr(agent, '_last_target_q', 0.0),
                        "diag/vec/worker_id": i,
                    }
                    log_metrics.update(cur_info)
                    # [v12.6] vec 模式 stab: worker 返回 stab_summary
                    _stab_sum = res.get('stab_summary')
                    if _stab_sum:
                        log_metrics.update({
                            f"stab/{phase}/{k}": v for k, v in _stab_sum.items()
                        })
                    logger.log(ep_count, log_metrics)

                    with open(lf, "a", newline="") as f:
                        csv.writer(f).writerow([ep_count, ts, f"{ep_rewards[i]:.3f}",
                            f"{ar:.3f}", f"{sr:.3f}", ep_steps[i],
                            f"{r.critic_loss:.5f}", f"{r.actor_loss:.5f}",
                            f"{agent.alpha:.5f}",
                            curs[i].level_idx, f"{perts[i]['wind_force']:.4f}",
                            f"{cur_info['cur/%s/wind_max' % phase]:.4f}",
                            f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                            curs[i].eps_at_level, i])

                    ep_count += 1
                    if ep_count > 0 and ep_count % SI == 0:
                        save_checkpoint(agent, log_dir, ep_count)
                    if sr > best:
                        best = sr; save_checkpoint(agent, log_dir, ep_count, tag="best")

                    # Reset this worker
                    obs_list[i], sxy_list[i], txy_list[i], rstate_list[i], perts[i] = \
                        _reset_one_env(i)
                    pt_list[i] = 0.0; py_list[i] = 0.0
                    ep_rewards[i] = 0.0; ep_steps[i] = 0
                    ep_suc[i] = False; ep_term[i] = "running"
                    phase_obs_cache[i] = vec.build_phase_obs_remote(
                        i, phase, obs_list[i], sxy_list[i], txy_list[i],
                        pt_list[i], py_list[i])

            # ── SAC train_step (off-policy update) ────────────────────────────
            # [v11 vec fix] 每个外层 step 收集 n_envs 个 transitions; 必须做
            # n_envs 次 update 才能 keep up, 否则 buffer 涨太快、critic 训练量
            # 落后, 表现为 critic_loss 涨到 100-300 (单进程仅 5-10).
            # 来源: SAC 原论文 (Haarnoja 2018) 推荐 1 update per 1 env step.
            if ts >= WU and ts % update_interval == 0:
                _n_updates = n_envs  # vec 模式: n_envs updates per outer step
                for _ in range(_n_updates):
                    agent.train_step()

    finally:
        vec.close()
        try: logger.close()
        except Exception: pass

    save_checkpoint(agent, log_dir, ep_count, tag="final")
    print(f"\n[{phase.upper()}-SAC-VEC] Done: {ts} steps, "
          f"{(time.time()-t0)/60:.1f} min, best_sr={best*100:.0f}%")
    return agent


# ==============================================================================
# SAC 训练 (单进程)
# ==============================================================================

def train_sac(phase, log_dir, config, resume_ckpt=None):
    T  = int(config["train"].get("total_timesteps", 2_000_000))
    SI = int(config["train"]["save_interval"])
    EI = int(config["train"].get("eval_interval", 100))
    W  = int(config["train"]["log_smooth_win"])
    WU = int(config["sac"].get("warmup_steps", 5000))

    print(f"\n{'='*60}\n  SAC | {phase.upper()} | {T} steps | warmup={WU} | "
          f"{log_dir}\n{'='*60}\n")

    env    = CableRobotEnvWithObstacles(config=config)
    agent  = SACPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    ectl   = EEAccController(config, env.ik_solver)
    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    _cruise_nmpc_base = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get("use_nmpc_base", False)))
    _descent_pid_residual = (phase == "descent" and
        bool(config.get("descent_rl", {}).get("pid_residual_mode", True)))
    # [v12] lift 也支持 NMPC base + RL 残差
    _lift_nmpc_base = (phase == "lift" and
        bool(config.get("lift_rl", {}).get("use_nmpc_base", False)))

    cur = CurriculumManager(config, phase)

    # [v9] resume_ckpt 用于断点续训
    if resume_ckpt and os.path.exists(resume_ckpt):
        agent.load(resume_ckpt); print(f"  Resumed from ckpt: {resume_ckpt}")

    logger = Logger(log_dir, project=f"phase_rl_v9", run_name=f"{phase}_sac")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time(); onf = False

    lf = os.path.join(log_dir, f"{phase}_sac_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                "sr", "steps", "cl", "al", "alpha", "lvl", "wind",
                                "wind_max", "cur_sr", "cur_eps"])

    while ts < T:
        pert = cur.sample_episode_perturbations()

        if phase == "descent":
            obs, pp = reset_for_descent_with_cur(env, config, cur)
        else:
            obs, pp = reset_for_phase(env, phase, config)
        if obs is None:
            continue
        env.set_force_noise(pert["force_noise"])
        _apply_episode_wind_env(env, pert)

        cq = env.data.qpos[:7].copy()
        expert.reset(obs, cq, env=env)
        if pp is not None:
            # [v12.2] lift 时只把"lift 段"喂给 tracker, 防止 look-ahead 跨段拉走 xy
            _pp_for_expert = _truncate_path_for_lift(pp, config) if phase == "lift" else pp
            expert.set_path(_pp_for_expert)
            if phase != "lift":
                plp = env.data.body('prefab').xpos.copy()
                _advance_expert_to_nearest_wp(expert, pp, plp)
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
        if phase == "lift" and hasattr(rs, 'start_xy'):
            rs.start_xy = env.data.body('prefab').xpos[:2].copy()
        if phase == "descent":
            _di = cur.get_descent_init()
            if _di is not None:
                rs.current_xy_range = _di["xy_range"]
                rs.current_xy_tol   = _di["xy_tol"]
                rs.current_descent_level = cur.level_idx
                rs.descent_n_levels      = cur.n_levels

        rew_tracker = RewardComponentTracker(phase)
        # [v12.6] 细粒度 RL 评估指标 tracker
        stab = StabilityMetrics()
        er = 0.0; es = 0; suc = False; term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])
        cached_base_dq = None

        while True:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            base_dq_for_obs = cached_base_dq
            if phase == "descent" and _descent_pid_residual and base_dq_for_obs is None:
                try:
                    base_dq_for_obs = expert.compute_delta_q_target(
                        obs, cq.astype(np.float64))
                except Exception:
                    base_dq_for_obs = np.zeros(7, dtype=np.float32)
            cached_base_dq = None
            # [v14.0]
            _wobs = build_wind_obs(env, float(config.get("wind_obs", {}).get("wind_force_max", 2.0)))
            core, cable_raw, _wobs, pt, py = build_phase_obs(
                phase, obs, env, sxy, txy, pt, py, wind_obs=_wobs,
                base_action=base_dq_for_obs)
            po = agent.encode_obs(core, cable_raw, _wobs)
            no = agent.normalize_obs(po, update=not onf)
            no_noisy = _add_obs_noise(no, pert["obs_noise"])

            if ts < WU:
                try:
                    act = collect_expert_acc(expert, env, obs, cq, phase, config)
                    act = act + np.random.normal(0, 0.1, len(act)).astype(np.float32)
                except Exception:
                    act = np.zeros(agent.action_dim, np.float32)
            else:
                act = agent.act(no_noisy, deterministic=False)
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
                    _wobs2 = build_wind_obs(env, float(config.get("wind_obs", {}).get("wind_force_max", 2.0)))
                    _c2, _cb2, _w2, _, _ = build_phase_obs(phase, obs, env, sxy, txy, pt, py, wind_obs=_wobs2)
                    npo = agent.encode_obs(_c2, _cb2, _w2)
                    nn_ = agent.normalize_obs(npo, update=False)
                    nn_ = _add_obs_noise(nn_, pert["obs_noise"])
                    agent.remember(no_noisy, act, nn_, rw, 1.0)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts; break

                if swing_d is not None:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)

                if _cruise_nmpc_base:
                    try:
                        _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                        _ba = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                    except Exception:
                        _ba = np.zeros(2, np.float32)
                    # [v11 Path 1] SAC cruise 默认与 PPO 对齐
                    _rm = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                    _cm = _ba + np.clip(act[:2], -_rm, _rm)
                    _am = float(config["cruise_rl"].get("residual_acc_max_xy", 0.60))
                    _cn = float(np.linalg.norm(_cm))
                    if _cn > _am: _cm = _cm / _cn * _am
                    a3 = np.array([_cm[0], _cm[1], 0.0])
                else:
                    a3 = np.array([act[0], act[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=None, residual_mode=False)

            elif phase == "descent" and _descent_pid_residual:
                dq, _pid_dq = _apply_descent_pid_residual(
                    expert, act, obs, env, config, cq,
                    pid_dq=base_dq_for_obs)

            elif phase == "lift" and _lift_nmpc_base:
                # [v12 fix] 用 expert.compute_delta_q_target(..., residual_acc=...)
                # 与 test_phase expert-only 路径完全一致.
                _rm_xy = float(config["lift_rl"].get("residual_acc_max_xy", 0.08))
                _rm_z  = float(config["lift_rl"].get("residual_acc_max_z",  0.10))
                _res3 = np.array([
                    float(np.clip(act[0], -_rm_xy, _rm_xy)),
                    float(np.clip(act[1], -_rm_xy, _rm_xy)),
                    float(np.clip(act[2], -_rm_z,  _rm_z)),
                ], np.float64)
                dq = expert.compute_delta_q_target(obs, cq, residual_acc=_res3)

            else:
                dq = ectl.compute_delta_q(act, cq, ree)

            dq = _add_act_noise(dq, pert["act_noise"])
            no2, _, _, _, ei = env.step(dq)

            if phase == "lift":
                # [v12] 传 rl_action 给 lift reward
                rw, dn, sc, ri = compute_lift_reward(env, no2, config, rs,
                                                     tracker=rew_tracker,
                                                     rl_action=act)
            elif phase == "cruise":
                # [v11.2] 传 rl_action 给 cruise reward (action_magnitude/smoothness penalty)
                rw, dn, sc, ri = compute_cruise_reward(env, no2, config, rs,
                                                       tracker=rew_tracker,
                                                       rl_action=act[:2])
            elif phase == "descent":
                # [v11.3] 传 rl_action 给 descent reward
                rw, dn, sc, ri = compute_descent_reward(env, no2, config, rs,
                                                        tracker=rew_tracker,
                                                        rl_action=act)
            else:
                rw, dn, sc, ri = compute_lift_reward(env, no2, config, rs,
                                                     tracker=rew_tracker,
                                                     rl_action=act)
            rew_tracker.step()

            done = dn or ei.get("nan_detected", False)
            if es >= mx - 1:
                done = True; ri.setdefault("termination", "timeout")
            if sc: suc = True
            if done and ri.get("termination"): term_reason = ri["termination"]

            # [v12.6] 细粒度 RL 评估指标更新
            stab.update_step(no2, config, env=env, rl_action=act)

            _wobs3 = build_wind_obs(env, float(config.get("wind_obs", {}).get("wind_force_max", 2.0)))
            _base3 = None
            if phase == "descent" and _descent_pid_residual and not done:
                try:
                    _cq3 = env.data.qpos[:7].copy().astype(np.float32)
                    _base3 = expert.compute_delta_q_target(
                        no2, _cq3.astype(np.float64))
                except Exception:
                    _base3 = np.zeros(7, dtype=np.float32)
                cached_base_dq = _base3
            _c3, _cb3, _w3, _, _ = build_phase_obs(
                phase, no2, env, sxy, txy, pt, py, wind_obs=_wobs3,
                base_action=_base3)
            npo = agent.encode_obs(_c3, _cb3, _w3)
            nn_ = agent.normalize_obs(npo, update=False)
            nn_ = _add_obs_noise(nn_, pert["obs_noise"])
            _pl_pos_now = env.data.body('prefab').xpos.copy()
            agent.remember(no_noisy, act, nn_, rw, float(done),
                           achieved_pos=_pl_pos_now)
            if ts >= WU and ts % agent.update_interval == 0:
                agent.train_step()
            er += rw; es += 1; ts += 1; agent.total_steps = ts; obs = no2
            if done:
                if phase == "descent":
                    agent.flush_episode_her(txy,
                        float(config["insertion"]["target_payload_z"]))
                break

        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()
        cur.update(suc)
        # [v10] 课程倒退后清理 (SAC 没有 reset_adam_state, 只清标志)
        if cur.consume_regression_flag():
            print(f"  [v10] SAC: 课程倒退检测到 (无 actor reset 处理)")
        r = agent._last_result

        mark = "✅" if suc else "❌"
        pl_pos_now = env.data.body('prefab').xpos
        dist_to_goal = float(np.linalg.norm(pl_pos_now[:2] - txy)) \
            if phase in ("cruise", "descent") else 0.0
        cur_info = cur.info()
        print(f"Ep{ep:4d} [{ts:7d}] {mark} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm "
              f"W:{pert['wind_force']:.3f}N L{cur.level_idx} | {term_reason}")
        print(f"       SAC CL:{r.critic_loss:7.4f} AL:{r.actor_loss:7.4f} "
              f"α:{agent.alpha:.4f}")

        log_metrics = {
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal_cm": dist_to_goal * 100,
            "sac/critic_loss": r.critic_loss,
            "sac/actor_loss":  r.actor_loss,
            "sac/alpha_loss":  r.alpha_loss,
            "sac/alpha":       r.alpha,
            "sac/q_mean":      r.q_mean,
            # [v11.4] Q-divergence 早期检测 (q_mean 应 ≈ target_q, gap 大说明 critic 落后)
            "sac/target_q_mean": getattr(agent, '_last_target_q', 0.0),
            "sac/q_target_gap":  r.q_mean - getattr(agent, '_last_target_q', 0.0),
        }
        log_metrics.update(cur_info)
        log_metrics.update(rew_tracker.episode_summary())
        # [v12.6] 细粒度 RL 评估指标
        log_metrics.update(stab.summary_for_wandb(phase, prefix="stab"))
        logger.log(ep, log_metrics)

        with open(lf, "a", newline="") as f:
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                    f"{r.critic_loss:.5f}", f"{r.actor_loss:.5f}",
                                    f"{agent.alpha:.5f}",
                                    cur.level_idx, f"{pert['wind_force']:.4f}",
                                    f"{cur_info['cur/%s/wind_max' % phase]:.4f}",
                                    f"{cur_info['cur/%s/sr_window' % phase]:.3f}",
                                    cur.eps_at_level])
        if ep > 0 and ep % SI == 0: save_checkpoint(agent, log_dir, ep)
        if ep > 0 and ep % EI == 0 and sr > best:
            best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-SAC] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    logger.close(); env.close()
    return agent



# ==============================================================================
# 训练入口
# ==============================================================================

def train(phase, log_dir, algo="ppo", custom_config=None, resume_ckpt=None):
    """[v11] BC 完全移除. cruise 残差用 actor 输出层零初始化 (自动); descent 用 PID 残差.

    Args:
        phase:        "lift" / "cruise" / "descent"
        log_dir:      日志和 checkpoint 目录
        algo:         "ppo" 或 "sac"
        custom_config: 覆盖 DEFAULT_CONFIG 的字段 (CLI 参数)
        resume_ckpt:   断点续训用的 checkpoint 路径 (可选)
    """
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config:
        for k, v in custom_config.items():
            if isinstance(v, dict) and k in config:
                config[k].update(v)
            else:
                config[k] = v
    set_global_seed(config["train"].get("seed", 42))
    os.makedirs(log_dir, exist_ok=True)

    n_envs = int(config["train"].get("n_envs", 1))
    if algo == "ppo":
        # [v11 Path 3] n_envs > 1 时启用 SubprocVecEnv 并行版本
        if n_envs > 1:
            print(f"\n  [v11 Path 3] 启用 vectorized PPO with n_envs={n_envs}")
            return train_ppo_vec(phase, log_dir, config,
                                 resume_ckpt=resume_ckpt, n_envs=n_envs)
        return train_ppo(phase, log_dir, config, resume_ckpt=resume_ckpt)
    elif algo == "sac":
        # [v11 Path 3] SAC 也支持 vec
        if n_envs > 1:
            print(f"\n  [v11 Path 3] 启用 vectorized SAC with n_envs={n_envs}")
            return train_sac_vec(phase, log_dir, config,
                                 resume_ckpt=resume_ckpt, n_envs=n_envs)
        return train_sac(phase, log_dir, config, resume_ckpt=resume_ckpt)
    else:
        raise ValueError(f"Unknown algo: {algo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[v13.0] 两阶段 RL 训练: cruise (NMPC 抬升+平移) + descent (PID 下降)")
    parser.add_argument("--phase",     type=str, required=True,
                        choices=["cruise", "descent", "lift"],
                        help="[v13.0] 推荐 'cruise' 或 'descent'. 'lift' 已被合并到 "
                             "cruise, 输入 'lift' 会自动重定向到 cruise.")
    parser.add_argument("--algo",      type=str, default="ppo",
                        choices=["ppo", "sac"])
    parser.add_argument("--log-dir",   type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--render",    action="store_true")
    parser.add_argument("--gpu",       type=int, default=0)
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="[v11] 断点续训 checkpoint 路径 (替代旧 --bc-ckpt)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-curriculum", action="store_true",
                        help="禁用课程学习 (level 永远停在 0)")
    parser.add_argument("--n-envs",    type=int, default=None,
                        help="[v11 Path 3] 并行环境数 (默认 1 = 单进程, "
                             ">1 = SubprocVecEnv 多进程)")
    parser.add_argument("--disable-her", action="store_true",
                        help="[v11 Path 2] 禁用 PPO descent 的 HER (默认开启)")
    args = parser.parse_args()

    # [v13.0] lift 已合并进 cruise, 自动重定向
    if args.phase == "lift":
        print("\n" + "=" * 70)
        print("[v13.0 提示] lift 阶段已合并进 cruise (NMPC 抬升+平移统一段)")
        print("             自动重定向 --phase lift → --phase cruise")
        print("=" * 70 + "\n")
        args.phase = "cruise"

    ld = args.log_dir or f"saves/{args.phase}_{args.algo}"
    cc = {}
    if args.render:    cc.setdefault("sim", {})["render"]   = True
    if args.gpu != 0:  cc.setdefault("train", {})["gpu_id"] = args.gpu
    if args.timesteps: cc.setdefault("train", {})["total_timesteps"] = args.timesteps
    cc.setdefault("train", {})["seed"] = args.seed
    if args.no_curriculum:
        cc.setdefault("curriculum", {})["enabled"] = False
    if args.n_envs is not None:
        cc.setdefault("train", {})["n_envs"] = args.n_envs
    if args.disable_her:
        cc.setdefault("descent_rl", {})["ppo_her_enabled"] = False

    train(args.phase, ld, algo=args.algo, custom_config=cc or None,
          resume_ckpt=args.resume_ckpt)
