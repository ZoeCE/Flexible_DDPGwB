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
    PPOPhaseAgent, SACPhaseAgent, CruiseDualRLAgent,
    build_lift_obs, build_cruise_obs, build_descent_obs,
    PPO_ZERO, SAC_ZERO,
    OBS_EE_X, OBS_EE_Y, OBS_EE_VX, OBS_EE_VY,
    OBS_PL_X, OBS_PL_Y, OBS_PL_VX, OBS_PL_VY,
    OBS_EE_Z, OBS_PL_Z, OBS_PL_VZ,
)
from phase_reward import (
    compute_lift_reward, compute_cruise_reward, compute_descent_reward,
    compute_lift_reward_tracked, compute_cruise_reward_planner,
    compute_cruise_reward_swing_rl, compute_descent_reward_tracked,
    LiftRewardState, CruiseRewardState, DescentRewardState,
    RewardComponentTracker,
)
from ee_acc_controller import EEAccController, CruiseZYawPID, SwingDampingController, compute_swing_energy
from orca_expert import CruiseORCAExpert, get_cruise_expert_acc

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


class StabilityMetrics:
    """
    全过程稳定性指标追踪器 — 用于定量衡量 RL 相对传统算法的优化效果.

    记录每个 episode 的:
      - swing_ke_curve: 摆动动能时序 (mJ), 用于绘制曲线
      - avg_swing_ke: 全过程平均摆动动能 (mJ)
      - max_swing_ke: 全过程最大摆动动能 (mJ)
      - avg_swing_angle: 全过程平均摆角 (deg)
      - max_swing_angle: 全过程最大摆角 (deg)
      - max_acc: 全过程最大 EE 加速度 (m/s²)
      - avg_acc: 全过程平均 EE 加速度 (m/s²)
      - acc_curve: EE 加速度时序
    """
    def __init__(self):
        self.reset_episode()

    def reset_episode(self):
        self.swing_ke_curve = []   # mJ
        self.acc_curve = []        # m/s²
        self.angle_curve = []      # deg
        self._prev_ee_pos = None

    def update_step(self, env, obs, config, dt=0.1):
        """每步调用, 提取摆动动能、摆角、加速度。"""
        try:
            rope_L = float(config.get("controller", {}).get("L", 0.5))
            mass   = float(config.get("prefab", {}).get("mass", 1.0))
            g      = 9.81

            # --- 摆动动能 ---
            pl_vxy = np.array([obs[6], obs[7]], np.float64)
            ee_vxy = np.array([obs[2], obs[3]], np.float64)
            rel_v  = pl_vxy - ee_vxy
            ke_J   = 0.5 * mass * float(np.dot(rel_v, rel_v))

            # --- 摆角 (via 相对位移) ---
            pl_xy  = np.array([obs[4], obs[5]], np.float64)
            ee_xy  = np.array([obs[0], obs[1]], np.float64)
            delta  = pl_xy - ee_xy
            sin_th = float(np.linalg.norm(delta)) / max(rope_L, 0.01)
            angle_deg = np.degrees(np.arcsin(min(sin_th, 1.0)))

            self.swing_ke_curve.append(ke_J * 1000)   # → mJ
            self.angle_curve.append(angle_deg)

            # --- EE 加速度 (差分估计) ---
            ee_pos = np.array([obs[0], obs[1]], np.float64)
            if self._prev_ee_pos is not None:
                v_now  = (ee_pos - self._prev_ee_pos) / max(dt, 1e-6)
                # 只取 xy 分量做近似
                acc_mag = float(np.linalg.norm(v_now))
                self.acc_curve.append(acc_mag)
            self._prev_ee_pos = ee_pos.copy()

        except Exception:
            pass

    def episode_summary(self, phase_prefix=""):
        """返回本 episode 的稳定性指标字典 (供 wandb 日志)。"""
        if not self.swing_ke_curve:
            return {}
        prefix = f"{phase_prefix}/" if phase_prefix else ""
        ke_arr  = np.array(self.swing_ke_curve)
        ang_arr = np.array(self.angle_curve)
        acc_arr = np.array(self.acc_curve) if self.acc_curve else np.zeros(1)
        return {
            f"{prefix}stability/avg_swing_ke_mJ":   float(np.mean(ke_arr)),
            f"{prefix}stability/max_swing_ke_mJ":   float(np.max(ke_arr)),
            f"{prefix}stability/avg_swing_angle_deg": float(np.mean(ang_arr)),
            f"{prefix}stability/max_swing_angle_deg": float(np.max(ang_arr)),
            f"{prefix}stability/avg_ee_acc_ms2":    float(np.mean(acc_arr)),
            f"{prefix}stability/max_ee_acc_ms2":    float(np.max(acc_arr)),
            f"{prefix}stability/p95_swing_ke_mJ":   float(np.percentile(ke_arr, 95)),
            f"{prefix}stability/p95_swing_angle_deg": float(np.percentile(ang_arr, 95)),
        }

    def print_summary(self, phase=""):
        """控制台打印稳定性摘要。"""
        if not self.swing_ke_curve:
            return ""
        ke_arr  = np.array(self.swing_ke_curve)
        ang_arr = np.array(self.angle_curve)
        return (f"KE:{np.mean(ke_arr):.1f}/{np.max(ke_arr):.1f}mJ "
                f"θ:{np.mean(ang_arr):.1f}°/{np.max(ang_arr):.1f}°")


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
    # [v3.6] 增加 XML 文件竞争重试机制
    # WSL/Windows 跨文件系统上 MuJoCo include 文件可能出现竞争写入 → Empty file 错误
    # 最多重试 5 次
    max_xml_retries = 5
    obs = None; planned_path = None
    for _xml_retry in range(max_xml_retries):
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs = env.reset()
            planned_path = env.get_planned_path()
            sys.stdout.close()
            sys.stdout = old_stdout
            break   # 成功
        except (ValueError, RuntimeError) as _xml_err:
            sys.stdout.close()
            sys.stdout = old_stdout
            _err_str = str(_xml_err)
            if "Empty file" in _err_str or "XML Error" in _err_str or "xml" in _err_str.lower():
                if _xml_retry < max_xml_retries - 1:
                    import time as _time
                    _time.sleep(0.05 * (_xml_retry + 1))  # 指数退避
                    continue
            raise   # 其他错误直接抛出
        except Exception:
            sys.stdout.close()
            sys.stdout = old_stdout
            raise

    if obs is None or planned_path is None:
        return None, None

    phase_cfg = config.get(f"{phase}_rl", {})
    rng = np.random.default_rng()

    if phase == "lift":
        # [v3.6] Lift 专属初始化 — 之前为 pass, agent 始终从固定起点出发, 无法泛化
        #
        # 设计:
        #   1. env.reset() 已将 prefab 放在 default_start_xy + noise 处 (noise=init_position_range)
        #      但该 noise 由 env 内部随机, 与 config["lift_rl"]["init_xy_range"] 无关
        #   2. 这里在 env.reset() 结果基础上追加 lift_rl 专属扰动:
        #      - prefab xy: ± init_xy_range (可叠加到 env 自身的随机)
        #      - prefab z:  默认 = init_qpos_prefab[2], 加 ± init_z_range
        #      - prefab vel: 从零开始 (init_vel_range=0.0)
        #   3. 臂形用 IK 跟随新的 prefab 位置
        #   4. warmup 60步让物理稳定后再开始训练
        #
        _lift_cfg  = config.get("lift_rl", {})
        _xy_range  = override_init_xy_range  if override_init_xy_range  is not None                      else float(_lift_cfg.get("init_xy_range", 0.01))
        _z_range   = float(_lift_cfg.get("init_z_range",  0.01))
        _vel_range = float(_lift_cfg.get("init_vel_range", 0.0))

        rope_L    = float(config["controller"].get("L", 0.5))
        seed_q    = np.array(config["reset"]["init_qpos_arm"], np.float64)
        init_pref_z = float(config["reset"]["init_qpos_prefab"][2])

        pref_jnt  = env.model.body("prefab").jntadr[0]
        qpos_addr = env.model.jnt_qposadr[pref_jnt]
        dof_idx   = env.model.jnt_dofadr[pref_jnt]

        # env.reset() 后的 prefab 位置作为基础起点
        base_xy = env.data.qpos[qpos_addr:qpos_addr+2].copy()
        base_z  = float(env.data.qpos[qpos_addr+2])

        # 在基础起点上叠加 lift 专属扰动
        noise_xy = rng.uniform(-_xy_range, _xy_range, 2)
        noise_z  = rng.uniform(-_z_range,  _z_range)
        new_pref_xy = base_xy + noise_xy
        new_pref_z  = float(np.clip(base_z + noise_z, init_pref_z * 0.5, init_pref_z * 1.5))

        # 臂形: IK 让 EE 悬停在 prefab 正上方 rope_L 处
        ee_z_target = new_pref_z + rope_L
        init_q = env.ik_solver.solve_4d(
            seed_q, float(new_pref_xy[0]), float(new_pref_xy[1]), ee_z_target, 0.0)
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()

        env.data.qpos[:7]          = init_q
        env.data.qvel[:7]          = 0.0
        env.data.ctrl[:7]          = init_q
        env.data.qpos[qpos_addr]   = new_pref_xy[0]
        env.data.qpos[qpos_addr+1] = new_pref_xy[1]
        env.data.qpos[qpos_addr+2] = new_pref_z
        env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]  # 单位四元数
        env.data.qvel[dof_idx:dof_idx+6] = 0.0

        # 物理稳定 (夹持prefab位置)
        has_viewer = (getattr(env, 'render_mode', False)
                      and getattr(env, 'viewer', None) is not None)
        for _ in range(60):
            env.data.qpos[:7]          = init_q
            env.data.qvel[:7]          = 0.0
            env.data.qpos[qpos_addr]   = new_pref_xy[0]
            env.data.qpos[qpos_addr+1] = new_pref_xy[1]
            env.data.qpos[qpos_addr+2] = new_pref_z
            env.data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            env.data.qvel[dof_idx:dof_idx+6] = 0.0
            mujoco.mj_step(env.model, env.data)
            if has_viewer:
                env.viewer.sync()
        mujoco.mj_forward(env.model, env.data)

        # 初始速度扰动 (vel_range 默认 0)
        if _vel_range > 0:
            env.data.qvel[dof_idx:dof_idx+2] += rng.uniform(-_vel_range, _vel_range, 2)
            mujoco.mj_forward(env.model, env.data)

        _sync_env_internal_state(env)
        obs = env._get_obs()

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


def collect_expert_acc(expert, env, obs, current_q, phase, config, orca_expert=None):
    """收集专家 EE 加速度 (用于 BC 标签)。"""
    if phase == "cruise":
        # [v4] 优先使用 ORCA expert (更真实的避障导航)
        if orca_expert is not None:
            try:
                txy = env.target_pos.copy()
                return get_cruise_expert_acc(env, obs, txy, config, orca_expert)
            except Exception:
                pass
        # fallback: 原有 tracker-based expert
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

        # ── 随机风力课程 [v5] ────────────────────────────────────────────────
        self.init_wind_curriculum(config)

        # ── Cruise 课程 [v5] ─────────────────────────────────────────────────
        # 取消障碍物课程渐进, 直接使用 3 个真实障碍物
        self.use_obs_cur   = False   # [v5] 关闭课程渐进
        self.current_n_obs = int(config.get("scene", {}).get("n_obstacles", 3))
        self.obs_max_n     = int(cur.get("obstacle_max_n", 3))
        self.obs_warmup    = int(cur.get("obstacle_level_warmup_eps", 500))
        self.obs_perf_win  = int(cur.get("perf_window", 300))
        self.obs_sr_thresh = float(cur.get("perf_sr_threshold", 0.65))
        self.obs_min_eps   = int(cur.get("perf_min_episodes_per_level", 3000))
        self.obs_hard_cap_eps = int(cur.get("perf_hard_cap_eps", 999_999))  # [v4] ep硬上限
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
        """旧接口兼容 — 不再使用线性插值模式, 返回 0.0 (由新接口控制)。"""
        return 0.0

    # ── 新接口: 随机风力课程 ────────────────────────────────────────────────
    def init_wind_curriculum(self, config):
        """初始化随机风力课程状态机。"""
        cur = config.get("curriculum", {})
        wind_cfg = config.get("wind", {})

        self.wind_enabled     = bool(cur.get("wind_enabled", False)) or \
                                bool(wind_cfg.get("enabled", False))
        self.wind_unlocked    = False
        self.wind_level       = 0
        self.wind_force_levels = list(cur.get("wind_max_force_levels", [0.5, 1.0, 2.0, 3.0]))
        self.wind_unlock_sr   = float(cur.get("wind_unlock_sr_thresh", 0.65))
        self.wind_unlock_min_eps = int(cur.get("wind_unlock_min_eps", 500))
        self.wind_level_sr    = float(cur.get("wind_level_sr_thresh", 0.55))
        self.wind_level_min_eps = int(cur.get("wind_level_min_eps", 300))
        self.wind_random_dir  = bool(cur.get("wind_random_dir", True))
        self._wind_level_ep   = 0
        self._wind_unlock_ep  = 0
        self._wind_stats      = EpisodeStats(window=100)
        self._wind_max_now    = 0.0  # 当前最大风力 (N)

    def sample_wind(self, ep, success, stats):
        """
        根据课程状态采样本 episode 的风力 (N) 和方向.
        返回 (force_magnitude, direction_xy) — force=0 表示无风.
        每 episode 开始时调用一次, 采样结果固定整个 episode.
        """
        if not self.wind_enabled:
            return 0.0, np.zeros(2)

        # 更新解锁统计
        if not self.wind_unlocked:
            self._wind_stats.update(success=float(success))
            eps_since_start = ep - self._wind_unlock_ep
            if (eps_since_start >= self.wind_unlock_min_eps and
                    self._wind_stats.success_rate() >= self.wind_unlock_sr):
                self.wind_unlocked = True
                self._wind_level_ep = ep
                self._wind_stats = EpisodeStats(window=100)
                self._wind_max_now = self.wind_force_levels[0] if self.wind_force_levels else 0.5
                print(f"  [Wind] ✅ 解锁风力训练! max_force={self._wind_max_now:.1f}N")
            return 0.0, np.zeros(2)

        # 已解锁: 更新晋级统计
        self._wind_stats.update(success=float(success))
        eps_at_level = ep - self._wind_level_ep
        if (self.wind_level < len(self.wind_force_levels) - 1 and
                eps_at_level >= self.wind_level_min_eps and
                self._wind_stats.success_rate() >= self.wind_level_sr):
            self.wind_level += 1
            self._wind_level_ep = ep
            self._wind_stats = EpisodeStats(window=100)
            self._wind_max_now = self.wind_force_levels[self.wind_level]
            print(f"  [Wind] ↑ 风力等级 → {self.wind_level} "
                  f"(max_force={self._wind_max_now:.1f}N)")

        # 随机采样: 均匀分布在 [0, max_force]
        force = np.random.uniform(0.0, self._wind_max_now)
        if self.wind_random_dir:
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        else:
            direction = np.array([1.0, 0.0])
        return float(force), direction

    def get_wind_curriculum_info(self):
        """返回风力课程状态, 用于日志。"""
        return {
            "wind/enabled":  float(self.wind_enabled),
            "wind/unlocked": float(getattr(self, 'wind_unlocked', False)),
            "wind/level":    float(getattr(self, 'wind_level', 0)),
            "wind/max_force": float(getattr(self, '_wind_max_now', 0.0)),
        }

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
            # [v4] hard_cap: 超过eps上限强制晋级 (防止永远卡在某个level)
            hard_capped = (eps >= self.obs_hard_cap_eps)
            sr_advance  = (eps >= self.obs_min_eps and sr >= self.obs_sr_thresh)
            advance = sr_advance or hard_capped
            if advance:
                reason = "hard_cap" if hard_capped and not sr_advance else f"sr={sr*100:.0f}%"
                self.current_n_obs = min(self.current_n_obs + 1, self.obs_max_n)
                env.set_curriculum_n_obstacles(self.current_n_obs)
                self._obs_lvl_ep = ep
                self._obs_lvl_ts = ts
                self.obs_stats = EpisodeStats(window=self.obs_perf_win)
                changed = True
                print(f"  [Curriculum] Obstacle level → {self.current_n_obs} 个真实障碍物 ({reason})")
        return changed, self.current_n_obs

    def sample_shadow_obstacles(self, start_xy, target_xy):
        """
        [v3.6 SHADOW] 始终补充 shadow obstacles 到 obs_max_n 个.
        无论真实障碍物有多少, shadow 补足剩余数量.
        真实障碍物 + shadow = obs_max_n (始终保持 3 个障碍物的 reward 信号)
        这样 critic 在任何课程阶段都见过 3 个障碍物, 晋级时无需重新适应.
        """
        if not self.shadow_enabled:
            return []
        # [v3.6] 补足数量: shadow_n = obs_max_n - current_n_obs
        n_shadow_needed = max(0, self.obs_max_n - self.current_n_obs)
        if n_shadow_needed == 0:
            return []

        rng = np.random.default_rng()
        obstacles = []
        direction = np.asarray(target_xy) - np.asarray(start_xy)
        L = float(np.linalg.norm(direction))
        if L < 1e-6:
            return []
        direction /= L
        perp = np.array([-direction[1], direction[0]])

        path_width = 0.6
        workspace_r = 0.50
        attempts = 0
        max_attempts = n_shadow_needed * 200
        base_r2 = 0.10   # 机械臂底座半径 (排除区域)
        min_clr = 0.075 + 0.03  # payload_radius + planning_margin

        while len(obstacles) < n_shadow_needed and attempts < max_attempts:
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

    def get_descent_init_params(self, ts, ep, success=None):
        """
        Get current descent level init params.
        Pass success=None (or omit) at episode START for read-only access.
        Pass success=True/False at episode END to update stats and check advance.
        """
        if not self.use_descent_init_cur:
            return None

        # Only update stats and check advance when success is explicitly provided
        if success is not None:
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
                    _stats_win = self.descent_stats.window
                    self.descent_stats = EpisodeStats(window=_stats_win)
                    print(f"  [Curriculum] Descent level {level}→{self.current_descent_level} "
                          f"(xy={self.descent_levels[self.current_descent_level]['xy_range']*1000:.1f}mm, "
                          f"eps={eps}, sr={sr*100:.0f}%)")

        return self.descent_levels[self.current_descent_level]

    def get_descent_stats_info(self):
        """Return current descent curriculum stats for logging."""
        if not self.use_descent_init_cur:
            return {}
        sr = self.descent_stats.success_rate()
        level = self.current_descent_level
        max_level = len(self.descent_levels) - 1
        return {
            "cur/descent_level": level,
            "cur/descent_sr_tracked": sr,
            "cur/descent_sr_pct": sr * 100,
            "cur/descent_sr_thresh": self.descent_cur_sr_thresh,
            "cur/descent_xy_mm": self.descent_levels[level]["xy_range"] * 1000,
            "cur/descent_at_max": float(level >= max_level),
        }

    def mark_rl_start(self):
        self._rl_started = True
        self.cruise_dist_stats = EpisodeStats(window=150)
        if hasattr(self, '_wind_unlock_ep'):
            self._wind_unlock_ep = 0   # reset from current ep (will be set properly at call time)
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
            info["cur/shadow_active"] = float(
                self.shadow_enabled and self.current_n_obs == 0)
        elif hasattr(self, 'current_n_obs'):
            info["cur/n_obstacles"] = self.current_n_obs
        if self.use_descent_init_cur:
            lvl = self.descent_levels[self.current_descent_level]
            info["cur/descent_level"]    = self.current_descent_level
            info["cur/descent_xy_range"] = lvl["xy_range"]
            info["cur/descent_xy_tol"]   = lvl.get("xy_tol", 0.010)
        # 风力信息直接用 get_wind_curriculum_info()
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

    # [v7] NMPC base + RL 残差 (ORCA 已废弃)
    _cruise_dual = (phase == "cruise" and
                    config.get("cruise_rl", {}).get("use_dual_rl", False))
    _cruise_nmpc_base = (phase == "cruise" and
                         bool(config.get("cruise_rl", {}).get("use_nmpc_base", False)) and
                         not _cruise_dual)
    _cruise_residual_orca = False  # ORCA 已废弃

    env    = CableRobotEnvWithObstacles(config=config)
    if _cruise_dual:
        agent = CruiseDualRLAgent(config=config, algo="ppo")
        print("  [v4] Cruise 双 RL: planner + swing_rl")
    else:
        agent  = PPOPhaseAgent(phase, config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    ectl   = EEAccController(config, env.ik_solver)
    orca_expert = None   # [v7] ORCA 已废弃，保留变量名兼容旧代码

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

    # ── 随机风力课程初始化 ────────────────────────────────────────────────────
    _wind_force_ep = 0.0   # 本 episode 风力大小 (N), 0=无风
    _wind_dir_ep   = np.zeros(2)
    _prev_suc      = False  # 上一 episode 成功标记 (用于 sample_wind 更新统计)

    # 多 episode 稳定性汇总统计 (滑动窗口)
    _stability_window = 50
    _stab_ke_list   = []   # 最近 N ep 的平均摆动动能 (mJ)
    _stab_ang_list  = []   # 最近 N ep 的平均摆角 (deg)
    _stab_acc_list  = []   # 最近 N ep 的平均 EE 加速度

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
        # ── 随机风力采样 [v5] ─────────────────────────────────────────────
        _wind_force_ep, _wind_dir_ep = cur.sample_wind(ep, _prev_suc, stats)
        wf = _wind_force_ep   # 兼容旧日志字段
        if _wind_force_ep > 0 and hasattr(env, 'set_wind_force'):
            env.set_wind_force(_wind_force_ep, _wind_dir_ep)
        else:
            env.set_wind_curriculum(0.0)


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
        if orca_expert is not None:
            orca_expert.reset()
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

        # Reward states
        rs = REWARD_STATES[phase]()
        rs_swing = CruiseRewardState() if _cruise_dual else None  # swing RL 独立 reward state

        if hasattr(agent, 'reset_history'):
            agent.reset_history()
        if hasattr(rs, 'total_steps_global'):
            rs.total_steps_global = ts

        # reward 分项追踪器
        rew_tracker = RewardComponentTracker(phase)
        swing_rew_tracker = RewardComponentTracker("cruise_swing") if _cruise_dual else None

        if phase == "lift" and hasattr(rs, 'start_xy'):
            rs.start_xy = env.data.body('prefab').xpos[:2].copy()

        if phase == "descent" and _desc_init is not None:
            rs.current_xy_range = _desc_init["xy_range"]
            rs.current_xy_tol   = _desc_init.get("xy_tol", None)
            rs.current_descent_level = cur.current_descent_level
            rs.descent_n_levels      = len(cur.descent_levels)
            rs.steps_at_max_level    = ts - cur.descent_lvl_ts \
                if cur.current_descent_level >= len(cur.descent_levels) - 1 else 0

        if phase == "cruise" and hasattr(rs, 'current_success_radius'):
            rs.current_success_radius = cur.get_current_success_radius(
                config["cruise_rl"]["reward"])
        if phase == "cruise" and hasattr(rs, 'shadow_obstacles'):
            rs.shadow_obstacles = cur.sample_shadow_obstacles(sxy, txy)
            rs.n_real_obstacles  = len(env._obstacles) if hasattr(env, '_obstacles') else 0
        if rs_swing is not None:
            rs_swing.current_success_radius = getattr(rs, 'current_success_radius', 0.20)
            rs_swing.shadow_obstacles = getattr(rs, 'shadow_obstacles', [])

        er = 0.0; es = 0; suc = False; rd = False
        ep_swing_reward = 0.0  # swing rl episode reward 追踪
        term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])
        stab_metrics = StabilityMetrics()   # [v5] 稳定性指标追踪

        # ── Episode 主循环 ────────────────────────────────────────────────────
        while not rd:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt, py = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
            no = agent.normalize_obs(po, update=True)

            ree = env._get_ee_pos()

            # ── Cruise: Dual RL (planner + swing_rl) [v4] ─────────────────────
            if _cruise_dual and z_pid is not None:
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
                    # push obs to history first (needed for LSTM buffer seq)
                    agent.obs_history.push(no)
                    agent.planner_agent.obs_history = agent.obs_history
                    agent.swing_agent.obs_history = agent.obs_history
                    agent.planner_agent.add_to_buffer(
                        no, np.zeros(2, np.float32), np.zeros(2, np.float32),
                        rw, 1.0, 0.0, 0.0)
                    agent.swing_agent.add_to_buffer(
                        no, np.zeros(2, np.float32), np.zeros(2, np.float32),
                        rw, 1.0, 0.0, 0.0)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts
                    rd = True; break

                # 监控 swing_d
                if swing_d is not None:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)

                # Dual RL act
                combined, planner_act, swing_act, planner_lp, planner_val, swing_lp, swing_val = \
                    agent.act(no)

                a3 = np.array([combined[0], combined[1], 0.0])
                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=None, residual_mode=False)

                # BC 标签
                bct_planner = np.zeros(2, np.float32)
                bct_swing   = np.zeros(2, np.float32)
                try:
                    bct_combined = collect_expert_acc(expert, env, obs, cq, phase, config, orca_expert)
                    bct_planner  = bct_combined
                    bct_swing    = np.zeros(2, np.float32)
                except Exception:
                    pass

                no2, _, _, _, ei = env.step(dq)
                stab_metrics.update_step(env, no2, config)  # [v5]

                # planner reward (导航为主)
                rw_plan, dn, sc, ri = compute_cruise_reward_planner(
                    env, no2, config, rs, tracker=rew_tracker)
                # swing reward (防摆为主)
                rw_swing, _ = compute_cruise_reward_swing_rl(
                    env, no2, config, rs_swing, tracker=swing_rew_tracker)

                done = dn or ei.get("nan_detected", False)
                if es >= mx - 1:
                    done = True; ri.setdefault("termination", "timeout")
                if sc:
                    suc = True
                if done and ri.get("termination"):
                    term_reason = ri["termination"]

                # 分别加入各自 buffer
                agent.add_to_buffers(
                    no, planner_act, swing_act, bct_planner, bct_swing,
                    rw_plan, rw_swing, float(done),
                    planner_val, planner_lp, swing_val, swing_lp)

                er += rw_plan; ep_swing_reward += rw_swing
                es += 1; ts += 1; agent.total_steps = ts
                obs = no2

                # 两个 agent 同步触发 update
                if agent.planner_agent.buffer.full:
                    if done:
                        lv_plan = 0.0; lv_swing = 0.0
                    else:
                        ns_ = build_phase_obs(phase, no2, env, sxy, txy, pt, py)[0]
                        nn_ = agent.normalize_obs(ns_, update=False)
                        lv_plan  = agent.planner_agent.get_value_for_state(nn_)
                        lv_swing = agent.swing_agent.get_value_for_state(nn_)

                    agent.planner_agent.buffer.compute_returns_and_advantages(
                        lv_plan, agent.gamma, agent.gae_lambda)
                    agent.swing_agent.buffer.compute_returns_and_advantages(
                        lv_swing, agent.gamma, agent.gae_lambda)
                    agent.update(global_ts=ts)
                    _ppo_update_count += 1
                    rd = True
                if done:
                    rd = True

            # ── 非 Cruise Dual 分支 ────────────────────────────────────────────
            # ── [v5] ORCA base + 残差 RL (cruise, 非双RL分支) ─────────────────
            elif phase == "cruise" and z_pid is not None:
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
                    agent.add_to_buffer(no, np.zeros(2, np.float32),
                                        np.zeros(2, np.float32), rw, 1.0,
                                        agent.get_value_for_state(no), 0.0)
                    er += rw; es += 1; ts += 1; agent.total_steps = ts
                    rd = True; break

                if swing_d is not None:
                    _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                    swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)

                act, lp, val = agent.act(no)

                # [v7] NMPC base + RL 残差 xy acc
                if _cruise_nmpc_base:
                    try:
                        _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                        _base = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                    except Exception:
                        _base = np.zeros(2, np.float32)
                    _res_max = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.25))
                    bct = np.clip(_base, -_res_max * 0.95, _res_max * 0.95)
                    _rl_clip = np.clip(act[:2], -_res_max, _res_max)
                    _comb = _base + _rl_clip
                    _amax = float(config["cruise_rl"].get("residual_acc_max_xy", 0.80))
                    _cn = float(np.linalg.norm(_comb))
                    if _cn > _amax: _comb = _comb / _cn * _amax
                    a3 = np.array([_comb[0], _comb[1], 0.0])
                else:
                    # 旧版 or 无 base
                    a3  = np.array([act[0], act[1], 0.0])
                    bct = np.zeros(2, np.float32)
                    try:
                        bct = collect_expert_acc(expert, env, obs, cq, phase, config, orca_expert)
                    except Exception:
                        pass

                dq = ectl.compute_delta_q(
                    a3, cq, ree, lock_z=True,
                    z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                    z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                    base_acc_xy=None, residual_mode=False)

                no2, _, _, _, ei = env.step(dq)
                stab_metrics.update_step(env, no2, config)  # [v5]
                rw, dn, sc, ri = compute_cruise_reward_planner(
                    env, no2, config, rs, tracker=rew_tracker)
                done = dn or ei.get("nan_detected", False)
                if es >= mx - 1:
                    done = True; ri.setdefault("termination", "timeout")
                if sc: suc = True
                if done and ri.get("termination"): term_reason = ri["termination"]

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
                    agent.update(global_ts=ts)
                    _ppo_update_count += 1
                    rd = True
                if done:
                    rd = True


            # ── Descent: PID base + RL residual [v3] ─────────────────────────
            elif phase == "descent" and _descent_pid_residual:
                act, lp, val = agent.act(no)
                dq, _pid_dq = _apply_descent_pid_residual(
                    expert, act, obs, env, config, cq)

                bct = np.zeros(agent.action_dim, np.float32)
                try:
                    bct = collect_expert_acc(expert, env, obs, cq, phase, config)
                except Exception:
                    pass

                no2, _, _, _, ei = env.step(dq)
                stab_metrics.update_step(env, no2, config)  # [v5]
                rw, dn, sc, ri = compute_descent_reward_tracked(
                    env, no2, config, rs, tracker=rew_tracker)
                done = dn or ei.get("nan_detected", False)
                if es >= mx - 1:
                    done = True; ri.setdefault("termination", "timeout")
                if sc: suc = True
                if done and ri.get("termination"): term_reason = ri["termination"]

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
                    rd = True
                if done:
                    rd = True

            else:
                # lift 或其他
                act, lp, val = agent.act(no)
                dq = ectl.compute_delta_q(act, cq, ree)

                bct = np.zeros(agent.action_dim, np.float32)
                if phase == "lift":
                    try:
                        bct = collect_expert_acc(expert, env, obs, cq, phase, config)
                    except Exception:
                        pass

                no2, _, _, _, ei = env.step(dq)
                stab_metrics.update_step(env, no2, config)  # [v5]
                rw, dn, sc, ri = compute_lift_reward_tracked(
                    env, no2, config, rs, tracker=rew_tracker)
                done = dn or ei.get("nan_detected", False)
                if es >= mx - 1:
                    done = True; ri.setdefault("termination", "timeout")
                if sc: suc = True
                if done and ri.get("termination"): term_reason = ri["termination"]

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
                    rd = True
                if done:
                    rd = True

        # ── 更新课程 & 日志 ───────────────────────────────────────────────────
        stats.update(reward=er, steps=es, success=float(suc))
        ar = stats.mean("reward"); sr = stats.success_rate()
        _prev_suc = suc   # [v5] 传递给风力课程

        cur.update_obs_curriculum(env, ep, ts, er, suc)
        if phase == "descent":
            cur.get_descent_init_params(ts, ep, suc)
        cur_info = cur.get_curriculum_info()

        # [v5] 稳定性指标汇总
        stab_summary = stab_metrics.episode_summary(phase)
        stab_str = stab_metrics.print_summary(phase)
        if stab_summary:
            _stab_ke_list.append(stab_summary.get(f"{phase}/stability/avg_swing_ke_mJ", 0))
            _stab_ang_list.append(stab_summary.get(f"{phase}/stability/avg_swing_angle_deg", 0))
            _stab_acc_list.append(stab_summary.get(f"{phase}/stability/avg_ee_acc_ms2", 0))
            if len(_stab_ke_list) > _stability_window:
                _stab_ke_list.pop(0); _stab_ang_list.pop(0); _stab_acc_list.pop(0)

        # agent last result (planner 为主)
        if _cruise_dual:
            r = agent.planner_agent._last_result
        else:
            r = agent._last_result
        m = "✅" if suc else "❌"

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

        # log_std (planner 的)
        if _cruise_dual:
            log_std_dims = agent.planner_agent.actor.log_std.detach().cpu().numpy().copy()
        elif hasattr(agent.actor, 'log_std'):
            log_std_dims = agent.actor.log_std.detach().cpu().numpy().copy()
        else:
            log_std_dims = np.array([-1.0])
        log_std_mean = float(np.mean(log_std_dims))
        log_std_str  = "/".join(f"{v:.3f}" for v in log_std_dims)
        _ls_ok = all(-2.05 <= v <= 0.35 for v in log_std_dims)
        _ls_warn = "" if _ls_ok else " ⚠️LOGSTD_OOB"

        print(f"Ep{ep:4d} [{ts:7d}] {m} R:{er:6.2f}({ar:5.2f}) SR:{sr*100:4.0f}% "
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm{_swing_str} W:{wf:.1f}N {stab_str}")
        if _cruise_dual:
            r2 = agent.swing_agent._last_result
            print(f"       PPO plan PL:{r.policy_loss:7.4f} VL:{r.value_loss:6.3f} "
                  f"KL:{r.approx_kl:.4f} | swing PL:{r2.policy_loss:7.4f}")
        else:
            if hasattr(r, 'policy_loss'):
                print(f"       PPO PL:{r.policy_loss:7.4f} VL:{r.value_loss:6.3f} "
                      f"KL:{r.approx_kl:.4f} CF:{r.clip_fraction:.2f} ent:{r.entropy_coef_used:.4f}")
        print(f"       logstd[{log_std_str}]{_ls_warn} | {cur_str} | {term_reason}")

        # ── wandb 日志 ────────────────────────────────────────────────────────
        log_metrics = {
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal": dist_to_goal,
            f"{phase}/wind_force":   wf,   # [v5] 随机风力大小
        }
        # [v5] 稳定性指标 (wandb)
        log_metrics.update(stab_summary)
        log_metrics.update(cur.get_wind_curriculum_info())
        if _stab_ke_list:
            log_metrics[f"{phase}/stability/win_avg_ke_mJ"]    = float(np.mean(_stab_ke_list))
            log_metrics[f"{phase}/stability/win_avg_angle_deg"] = float(np.mean(_stab_ang_list))
            log_metrics[f"{phase}/stability/win_avg_acc_ms2"]  = float(np.mean(_stab_acc_list))

        if _cruise_dual:
            log_metrics[f"{phase}/swing_reward"] = ep_swing_reward
            log_metrics["ppo_planner/pl"]  = r.policy_loss
            log_metrics["ppo_planner/vl"]  = r.value_loss
            log_metrics["ppo_planner/kl"]  = r.approx_kl
            log_metrics["ppo_planner/ent"] = r.entropy_loss
            r2 = agent.swing_agent._last_result
            if hasattr(r2, 'policy_loss'):
                log_metrics["ppo_swing/pl"]  = r2.policy_loss
                log_metrics["ppo_swing/vl"]  = r2.value_loss
                log_metrics["ppo_swing/kl"]  = r2.approx_kl
        elif hasattr(r, 'policy_loss'):
            log_metrics["ppo/pl"]  = r.policy_loss
            log_metrics["ppo/vl"]  = r.value_loss
            log_metrics["ppo/kl"]  = r.approx_kl
            log_metrics["ppo/cf"]  = r.clip_fraction
            log_metrics["ppo/ent"] = r.entropy_loss
            log_metrics["ppo/entropy_coef"] = r.entropy_coef_used
            log_metrics["ppo/logstd_mean"]  = log_std_mean

        # 摆动能量
        if phase == "cruise" and swing_d is not None:
            log_metrics["cruise/swing_energy_mJ"] = _swing_energy * 1000
            log_metrics["cruise/swing_angle_deg"] = swing_d.last_angle_deg
            log_metrics["cruise/damp_gain"]       = swing_d._gain_scale
            log_metrics["cur/success_radius"] = cur.get_current_success_radius(
                config["cruise_rl"]["reward"])

        # reward 分项 (详细分析)
        rew_summary = rew_tracker.episode_summary()
        log_metrics.update(rew_summary)
        if swing_rew_tracker is not None:
            swing_summary = swing_rew_tracker.episode_summary()
            log_metrics.update(swing_summary)

        log_metrics.update(cur_info)
        # [v4] descent 课程追踪 SR (实际统计SR, 非滑动窗口SR)
        if phase == "descent":
            log_metrics.update(cur.get_descent_stats_info())
        logger.log(ep, log_metrics)

        # logstd per dim
        dim_names = ["ax", "ay", "az"] if not _cruise_dual and \
            agent.action_dim == 3 else ["ax", "ay"]
        for i, name in enumerate(dim_names[:len(log_std_dims)]):
            log_metrics[f"ppo/logstd_{name}"] = float(log_std_dims[i])

        with open(lf, "a", newline="") as f:
            pl_val = r.policy_loss if hasattr(r, 'policy_loss') else 0.0
            vl_val = r.value_loss  if hasattr(r, 'value_loss')  else 0.0
            kl_val = r.approx_kl  if hasattr(r, 'approx_kl')   else 0.0
            cf_val = r.clip_fraction if hasattr(r, 'clip_fraction') else 0.0
            csv.writer(f).writerow([ep, ts, f"{er:.3f}", f"{ar:.3f}", f"{sr:.3f}", es,
                                    f"{pl_val:.5f}", f"{vl_val:.5f}",
                                    f"{kl_val:.5f}", f"{cf_val:.3f}", f"{wf:.3f}"])
        if ep > 0 and ep % SI == 0:
            save_checkpoint(agent, log_dir, ep)
        if ep > 0 and ep % EI == 0 and sr > best:
            best = sr; save_checkpoint(agent, log_dir, ep, tag="best")
        ep += 1

    save_checkpoint(agent, log_dir, ep, tag="final")
    print(f"\n[{phase.upper()}-PPO] Done: {ts} steps, {(time.time()-t0)/60:.1f} min, "
          f"best_sr={best*100:.0f}%")
    logger.close(); env.close(); return agent
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
    _cruise_nmpc_base_sac = (phase == "cruise" and
                             bool(config.get("cruise_rl", {}).get("use_nmpc_base", False)))
    orca_expert = None  # [v7] ORCA 已废弃
    cur = CurriculumManager(config, phase)
    if cur.use_obs_cur:
        env.set_curriculum_n_obstacles(cur.current_n_obs)

    if bc_ckpt and os.path.exists(bc_ckpt):
        agent.load(bc_ckpt); print(f"  Loaded BC ckpt: {bc_ckpt}")

    _descent_pid_residual = (phase == "descent" and
                              bool(config.get("descent_rl", {}).get("pid_residual_mode", True)))
    # NOTE: cruise dual-RL (CruiseDualRLAgent) is PPO-only.
    # SAC cruise uses single SACPhaseAgent (use_dual_rl ignored for SAC).
    cur.mark_rl_start()   # 必须在 RL 开始前调用，否则 descent 课程永远不推进

    logger = Logger(log_dir, f"{phase}_sac", f"{phase}_sac")
    logger.update_config(config)
    stats = EpisodeStats(window=W)
    ep = 0; ts = 0; best = 0.0; t0 = time.time(); onf = False

    # [v5] 随机风力 & 稳定性追踪初始化
    _prev_suc = False
    _stability_window = 50
    _stab_ke_list = []; _stab_ang_list = []; _stab_acc_list = []

    lf = os.path.join(log_dir, f"{phase}_sac_log.csv")
    with open(lf, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "total_steps", "ep_reward", "avg_reward",
                                 "sr", "steps", "cl", "al", "alpha", "wind"])

    while ts < T:
        # ── 随机风力采样 [v5] ─────────────────────────────────────────────
        _wind_force_ep, _wind_dir_ep = cur.sample_wind(ep, _prev_suc, stats)
        wf = _wind_force_ep
        if _wind_force_ep > 0 and hasattr(env, 'set_wind_force'):
            env.set_wind_force(_wind_force_ep, _wind_dir_ep)
        else:
            env.set_wind_curriculum(0.0)


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

        # [v3.6] 注入 lift start_xy
        if phase == "lift" and hasattr(rs, 'start_xy'):
            rs.start_xy = env.data.body('prefab').xpos[:2].copy()

        er = 0.0; es = 0; suc = False; term_reason = "running"
        mx = int(config[f"{phase}_rl"]["max_steps"])
        rew_tracker = RewardComponentTracker(phase)  # reward 分项追踪器
        stab_metrics = StabilityMetrics()   # [v5]

        while True:
            cq = env.data.qpos[:7].copy().astype(np.float32)
            po, pt, py = build_phase_obs(phase, obs, env, sxy, txy, pt, py)
            no = agent.normalize_obs(po, update=not onf)

            if ts < WU:
                try:
                    act = collect_expert_acc(wex, env, obs, cq, phase, config, orca_expert)
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

                # [v7] NMPC base + RL 残差 (SAC)
                if _cruise_nmpc_base_sac:
                    try:
                        _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                        _ba = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                    except Exception:
                        _ba = np.zeros(2, np.float32)
                    _rm = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.25))
                    _cm = _ba + np.clip(act[:2], -_rm, _rm)
                    _am = float(config["cruise_rl"].get("residual_acc_max_xy", 0.80))
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
            stab_metrics.update_step(env, no2, config)  # [v5]
            # 使用带追踪的 reward 函数
            if phase == "lift":
                rw, dn, sc, ri = compute_lift_reward_tracked(env, no2, config, rs, tracker=rew_tracker)
            elif phase == "cruise":
                rw, dn, sc, ri = compute_cruise_reward_planner(env, no2, config, rs, tracker=rew_tracker)
            elif phase == "descent":
                rw, dn, sc, ri = compute_descent_reward_tracked(env, no2, config, rs, tracker=rew_tracker)
            else:
                rw, dn, sc, ri = compute_lift_reward_tracked(env, no2, config, rs, tracker=rew_tracker)
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
        _prev_suc = suc   # [v5] 传递给风力课程

        cur.update_obs_curriculum(env, ep, ts, er, suc)
        if phase == "descent":
            cur.get_descent_init_params(ts, ep, suc)
        cur_info = cur.get_curriculum_info()

        # [v5] 稳定性指标
        stab_summary = stab_metrics.episode_summary(phase)
        stab_str = stab_metrics.print_summary(phase)
        if stab_summary:
            _stab_ke_list.append(stab_summary.get(f"{phase}/stability/avg_swing_ke_mJ", 0))
            _stab_ang_list.append(stab_summary.get(f"{phase}/stability/avg_swing_angle_deg", 0))
            _stab_acc_list.append(stab_summary.get(f"{phase}/stability/avg_ee_acc_ms2", 0))
            if len(_stab_ke_list) > _stability_window:
                _stab_ke_list.pop(0); _stab_ang_list.pop(0); _stab_acc_list.pop(0)

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
              f"S:{es:3d} dist:{dist_to_goal*100:.1f}cm{_swing_str} W:{wf:.1f}N {stab_str}")
        print(f"       SAC CL:{r.critic_loss:7.4f} AL:{r.actor_loss:7.4f} "
              f"α:{agent.alpha:.4f} | {cur_str} | {term_reason}")

        log_metrics = {
            f"{phase}/reward":       er,
            f"{phase}/avg_reward":   ar,
            f"{phase}/sr":           sr,
            f"{phase}/steps":        es,
            f"{phase}/dist_to_goal": dist_to_goal,
            f"{phase}/wind_force":   wf,   # [v5]
            "sac/alpha":             agent.alpha,
            "sac/cl":                r.critic_loss,
            "sac/al":                r.actor_loss,
        }
        # [v5] 稳定性指标
        log_metrics.update(stab_summary)
        log_metrics.update(cur.get_wind_curriculum_info())
        if _stab_ke_list:
            log_metrics[f"{phase}/stability/win_avg_ke_mJ"]    = float(np.mean(_stab_ke_list))
            log_metrics[f"{phase}/stability/win_avg_angle_deg"] = float(np.mean(_stab_ang_list))
            log_metrics[f"{phase}/stability/win_avg_acc_ms2"]  = float(np.mean(_stab_acc_list))
        if phase == "cruise" and swing_d is not None:
            log_metrics["cruise/swing_energy_mJ"] = _swing_energy * 1000
            log_metrics["cruise/swing_angle_deg"] = swing_d.last_angle_deg
            log_metrics["cruise/damp_gain"]       = swing_d._gain_scale
            log_metrics["cur/success_radius"] = cur.get_current_success_radius(
                config["cruise_rl"]["reward"])
        # reward 分项追踪
        rew_summary = rew_tracker.episode_summary()
        log_metrics.update(rew_summary)
        log_metrics.update(cur_info)
        if phase == "descent":
            log_metrics.update(cur.get_descent_stats_info())
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
# BC 预训练 — Cruise [v7: NMPC base, 执行者=标签者, 量纲正确]
# ==============================================================================

def pretrain_bc_cruise(agent, config):
    """
    BC 预训练 v7: NMPC 执行 + NMPC tracker xy acc 标签

    关键修复:
      acc_max = action_scale = residual_acc_max_xy_rl = 0.25 m/s²
      BC 标签限幅到 ±0.95×acc_max 防止 atanh 饱和
    """
    bc = config.get("bc_pretrain", {})
    if not bc.get("enabled", True): return

    nep               = int(bc.get("n_episodes",     800))
    nepoch            = int(bc.get("n_epochs_cruise", 30))
    lr                = float(bc.get("lr",            3e-4))
    lr_decay          = float(bc.get("lr_decay",      0.5))
    lr_decay_interval = int(bc.get("lr_decay_interval", 15))
    bs                = int(bc.get("batch_size",      512))
    eval_interval     = int(bc.get("eval_interval",   5))
    eval_episodes     = int(bc.get("eval_episodes",   30))
    patience          = int(bc.get("patience",        4))
    loss_thresh       = float(bc.get("loss_threshold", 0.015))
    eps_start         = float(bc.get("epsilon_start", 0.2))
    eps_end           = float(bc.get("epsilon_end",   0.0))
    n_dagger_rounds   = int(bc.get("n_dagger_rounds", 2))

    # [v7 KEY] action_scale = residual_acc_max_xy_rl
    acc_max   = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.25))
    max_steps = int(config["cruise_rl"]["max_steps"])

    import copy as _copy
    from phase_reward import CruiseRewardState, compute_cruise_reward
    bc_config = _copy.deepcopy(config)
    rope_L = float(config["controller"].get("L", 0.5))
    bc_config["step_logic"]["instability_grace_steps"] = max_steps
    bc_config["step_logic"]["swing_xy_max"] = rope_L * 1.5
    _sr_real = float(config["cruise_rl"]["reward"].get("success_radius_end", 0.10))

    print(f"\n{'='*60}")
    print(f"  [BC-cruise v7] NMPC base | {nep} eps | {nepoch} epochs")
    print(f"  action_scale={acc_max:.3f} m/s² | 真实SR≈100%")
    print(f"{'='*60}")

    env    = CableRobotEnvWithObstacles(config=config)
    expert = JointSpaceExpert(config, env.ik_solver)
    aobs_tr, aact_tr = [], []
    aobs_ev, aact_ev = [], []
    ve = 0; att = 0; succ = 0

    while ve < nep and att < nep * 2:
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
        pt2, py2 = 0.0, 0.0; ve += 1
        is_eval = (ve % max(nep // max(eval_episodes, 1), 1) == 0)
        ep_succ = False
        rstate = CruiseRewardState()
        rstate.current_success_radius = _sr_real

        for _s in range(max_steps):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            # [v7] 标签: NMPC tracker xy acc, 限幅到 0.95×acc_max 防 atanh 饱和
            try:
                _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=0.0)
                acc_label = np.clip(
                    np.array([float(_a4[0]), float(_a4[1])], np.float32),
                    -acc_max * 0.95, acc_max * 0.95)
            except Exception:
                acc_label = np.zeros(2, np.float32)

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
        if ve % 100 == 0 or ve == nep:
            print(f"  [BC-cruise] {ve:4d}/{nep} | tr={len(aobs_tr):6d} ev={len(aobs_ev):5d} "
                  f"| NMPC SR={succ/max(ve,1)*100:.0f}%")

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

    for dagger_round in range(max(1, n_dagger_rounds)):
        if dagger_round > 0:
            dagger_eps = nep // max(n_dagger_rounds, 1)
            print(f"  [BC-cruise] DAgger 轮{dagger_round+1}: {dagger_eps}eps")
            env_d  = CableRobotEnvWithObstacles(config=config)
            exp_d  = JointSpaceExpert(config, env_d.ik_solver)
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
                d_txy = env_d.target_pos.copy(); d_pt2, d_py2 = 0.0, 0.0; dv += 1
                d_rs = CruiseRewardState(); d_rs.current_success_radius = _sr_real
                for _ds in range(max_steps):
                    try:
                        _da4 = exp_d.tracker.compute_ee_acceleration(d_obs, target_yaw=0.0)
                        d_lbl = np.clip(np.array([float(_da4[0]), float(_da4[1])], np.float32),
                                        -acc_max * 0.95, acc_max * 0.95)
                    except Exception:
                        d_lbl = np.zeros(2, np.float32)
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
                ns = len(ot); print(f"  DAgger +{len(dobs_new)}, 总计 {ns}")

        round_nepoch = nepoch if dagger_round == 0 else nepoch // 2
        pat = 0
        for e in range(round_nepoch):
            eps = max(eps_end, eps_start * (1.0 - e / max(round_nepoch - 1, 1)))
            agent.actor.train()
            idx = np.random.permutation(ns); tl = 0.0; nb = 0
            for st in range(0, ns, bs):
                i = idx[st:st+bs]; ab = at[i]
                if eps > 0:
                    mask  = torch.rand(len(i), device=agent.device) < eps
                    noise = (torch.rand_like(ab)*2 - 1) * acc_max * 0.3
                    ab    = torch.where(mask.unsqueeze(1), ab + noise, ab)
                loss, _, _ = agent.actor.bc_forward(ot[i], ab)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(bp, 1.0)
                opt.step(); tl += loss.item(); nb += 1
            sch.step(); tl /= max(nb, 1)
            if (e+1) % eval_interval == 0 or e == round_nepoch - 1:
                agent.actor.eval()
                with torch.no_grad():
                    el = agent.actor.bc_forward(oe, ae)[0].item() if oe is not None else tl
                print(f"  R{dagger_round+1} Ep{e+1:3d}/{round_nepoch} "
                      f"tr={tl:.5f} ev={el:.5f} eps={eps:.2f} pat={pat}/{patience}")
                if el < best_eval - 1e-6:
                    best_eval = el
                    best_state = {k: v.clone() for k, v in agent.actor.state_dict().items()}
                    pat = 0
                else:
                    pat += 1
                if pat >= patience: print(f"  早停"); break
                if el < loss_thresh: print(f"  loss 达标"); break

    if best_state:
        agent.actor.load_state_dict(best_state)
        print(f"  [BC-cruise] ✅ ev={best_eval:.5f}")
    if hasattr(agent, 'reset_log_std_for_rl'):
        agent.reset_log_std_for_rl()
    print(f"  [BC-cruise v7] 完成 | NMPC SR={succ/max(ve,1)*100:.0f}%\n")
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
        # [v4] cruise BC:
        #   PPO → CruiseDualRLAgent: trains planner, saves planner_actor/swing_actor format
        #   SAC → plain SACPhaseAgent: trains single agent, saves actor/critic format
        #         (SAC dual-RL not supported; single SAC agent is the RL agent)
        _ppo_dual = (phase == "cruise" and algo == "ppo" and
                     config.get("cruise_rl", {}).get("use_dual_rl", True))

        if _ppo_dual:
            ag = CruiseDualRLAgent(config=config, algo="ppo")
            pretrain_bc_cruise(ag.planner_agent, config)
            try:
                ag.swing_agent.actor.load_state_dict(
                    ag.planner_agent.actor.state_dict(), strict=False)
            except Exception:
                pass
        elif algo == "ppo":
            ag = PPOPhaseAgent(phase, config=config)
            if phase == "cruise":
                pretrain_bc_cruise(ag, config)
            else:
                pretrain_bc_descent(ag, config)
        else:  # SAC
            ag = SACPhaseAgent(phase, config=config)
            if phase == "cruise":
                # For SAC BC: train a temporary PPO agent, then extract actor weights
                _tmp_ppo = PPOPhaseAgent(phase, config=config)
                pretrain_bc_cruise(_tmp_ppo, config)
                # Transfer BC actor weights (shared MLP backbone compatible)
                try:
                    ag.actor.load_state_dict(_tmp_ppo.actor.state_dict(), strict=False)
                    print("  [BC] PPO→SAC actor weight transfer")
                except Exception as e:
                    print(f"  [BC] actor transfer warning: {e}")
                if hasattr(_tmp_ppo, 'obs_norm'):
                    ag.obs_norm.load_state_dict(_tmp_ppo.obs_norm.state_dict())
                del _tmp_ppo
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
    if args.wind:
        cc.setdefault("curriculum", {})["wind_enabled"] = True
        cc.setdefault("wind", {})["enabled"] = True
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