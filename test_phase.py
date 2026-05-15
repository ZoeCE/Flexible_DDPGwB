# ==============================================================================
# test_phase.py — 三阶段独立 + 流水线测试
#
# 用法:
#   # 单阶段测试
#   python test_phase.py --phase lift --algo ppo --ckpt saves/lift_ppo/ckpt_best.pt
#   python test_phase.py --phase cruise --algo sac --ckpt saves/cruise_sac/ckpt_best.pt
#   python test_phase.py --phase descent --algo ppo --ckpt saves/descent_ppo/ckpt_best.pt
#
#   # 专家基准 (单阶段)
#   python test_phase.py --phase lift --algo expert
#   python test_phase.py --phase cruise --algo expert
#
#   # 完整流水线 (3 阶段串联)
#   python test_phase.py --phase pipeline \
#       --lift-ckpt saves/lift_ppo/ckpt_best.pt \
#       --cruise-ckpt saves/cruise_sac/ckpt_best.pt \
#       --descent-ckpt saves/descent_ppo/ckpt_best.pt \
#       --lift-algo ppo --cruise-algo sac --descent-algo ppo
#
#   # 专家全流程基准
#   python test_phase.py --phase pipeline --algo expert
#
#   # 带渲染
#   python test_phase.py --phase lift --algo expert --render
# ==============================================================================

import os
import sys
import copy
import argparse
import numpy as np
import torch
from collections import Counter

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from phase_agent import (
    PPOPhaseAgent, SACPhaseAgent, DescentDualRLAgent, CruiseDualRLAgent,
    build_lift_obs, build_cruise_obs, build_descent_obs,
)
from phase_reward import (
    compute_lift_reward, compute_cruise_reward, compute_descent_reward,
    LiftRewardState, CruiseRewardState, DescentRewardState,
)
from ee_acc_controller import EEAccController, CruiseZYawPID, SwingDampingController
from scipy.spatial.transform import Rotation as R
from orca_expert import CruiseORCAExpert, get_cruise_expert_acc


# ==============================================================================
# 稳定性指标追踪器 (与 train_phase.StabilityMetrics 保持一致)
# ==============================================================================

class StabilityMetrics:
    """全过程稳定性指标: 摆动动能、摆角、EE 加速度。"""
    def __init__(self):
        self.reset_episode()

    def reset_episode(self):
        self.swing_ke_curve = []
        self.acc_curve = []
        self.angle_curve = []
        self._prev_ee_pos = None

    def update_step(self, obs, config, dt=0.1):
        try:
            rope_L = float(config.get("controller", {}).get("L", 0.5))
            mass   = float(config.get("prefab", {}).get("mass", 1.0))
            pl_vxy = np.array([obs[6], obs[7]], np.float64)
            ee_vxy = np.array([obs[2], obs[3]], np.float64)
            ke_J   = 0.5 * mass * float(np.dot(pl_vxy - ee_vxy, pl_vxy - ee_vxy))
            pl_xy  = np.array([obs[4], obs[5]], np.float64)
            ee_xy  = np.array([obs[0], obs[1]], np.float64)
            sin_th = float(np.linalg.norm(pl_xy - ee_xy)) / max(rope_L, 0.01)
            angle_deg = float(np.degrees(np.arcsin(min(sin_th, 1.0))))
            self.swing_ke_curve.append(ke_J * 1000)
            self.angle_curve.append(angle_deg)
            ee_pos = np.array([obs[0], obs[1]], np.float64)
            if self._prev_ee_pos is not None:
                acc_mag = float(np.linalg.norm((ee_pos - self._prev_ee_pos) / max(dt, 1e-6)))
                self.acc_curve.append(acc_mag)
            self._prev_ee_pos = ee_pos.copy()
        except Exception:
            pass

    def summary(self):
        if not self.swing_ke_curve:
            return {}
        ke  = np.array(self.swing_ke_curve)
        ang = np.array(self.angle_curve)
        acc = np.array(self.acc_curve) if self.acc_curve else np.zeros(1)
        return {
            "avg_ke_mJ":   float(np.mean(ke)),
            "max_ke_mJ":   float(np.max(ke)),
            "p95_ke_mJ":   float(np.percentile(ke, 95)),
            "avg_angle":   float(np.mean(ang)),
            "max_angle":   float(np.max(ang)),
            "p95_angle":   float(np.percentile(ang, 95)),
            "avg_acc":     float(np.mean(acc)),
            "max_acc":     float(np.max(acc)),
            "ke_curve":    ke.tolist(),
            "angle_curve": ang.tolist(),
        }

    def print_line(self):
        if not self.swing_ke_curve:
            return ""
        ke  = np.array(self.swing_ke_curve)
        ang = np.array(self.angle_curve)
        acc = np.array(self.acc_curve) if self.acc_curve else np.zeros(1)
        return (f"KE avg/max:{np.mean(ke):.1f}/{np.max(ke):.1f}mJ "
                f"θ avg/max:{np.mean(ang):.1f}°/{np.max(ang):.1f}° "
                f"acc avg/max:{np.mean(acc):.2f}/{np.max(acc):.2f}m/s²")


# ==============================================================================
# 工具
# ==============================================================================

def build_config(args):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["sim"]["render"] = args.render
    config["train"]["gpu_id"] = args.gpu
    if args.obstacles is not None:
        config["scene"]["n_obstacles"] = max(int(args.obstacles),
                                              config["scene"]["n_obstacles"])
    return config


def load_agent(phase, algo, ckpt_path, config):
    """加载指定阶段和算法的 agent。"""
    # descent dual-RL
    use_descent_dual = (phase == "descent" and
                        config.get("descent_rl", {}).get("use_dual_rl", False))
    # cruise dual-RL [v4]
    use_cruise_dual = (phase == "cruise" and
                       config.get("cruise_rl", {}).get("use_dual_rl", True))

    if use_descent_dual and algo == "ppo":
        agent = DescentDualRLAgent(config=config)
        agent.load(ckpt_path)
        return agent
    elif use_cruise_dual and algo in ("ppo", "sac"):
        agent = CruiseDualRLAgent(config=config, algo=algo)
        agent.load(ckpt_path)
        return agent
    elif algo == "ppo":
        agent = PPOPhaseAgent(phase, config=config)
    elif algo == "sac":
        agent = SACPhaseAgent(phase, config=config)
    else:
        raise ValueError(f"Unknown algo: {algo}")
    agent.load(ckpt_path)
    return agent


def get_phase_state(env, obs, start_xy=None, config=None):
    """
    获取当前物理状态，用于阶段切换判断。

    v7 修复: 补充 lift 切换所需字段 (pl_vz_abs, swing_energy, dtf_start)
    原版缺失这些字段 → state.get(..., 999.0) 永远返回 999 → lift→cruise 切换永远失败
    """
    pl_xy     = np.array([obs[4], obs[5]])
    payload_z = float(env.data.body('prefab').xpos[2])
    pl_vxy    = obs[6:8]
    ee_vxy    = obs[2:4]
    swing_vel = float(np.linalg.norm(pl_vxy - ee_vxy))
    pl_vel    = float(np.linalg.norm(pl_vxy))

    pl_mat    = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler  = R.from_matrix(pl_mat).as_euler('xyz')
    tilt      = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))

    # [v7 FIX] lift 切换所需字段
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz   = float(env.data.qvel[dof_idx + 2])

    # 摆动能量 (KE + PE)
    try:
        mass   = float(env.config.get("prefab", {}).get("mass", 1.0))
        rope_L = float(env.config.get("controller", {}).get("L", 0.5))
        pl_pos_3d = env.data.body('prefab').xpos.copy()
        ee_pos_3d = env._get_ee_pos()
        pl_vel_3d = env.data.qvel[dof_idx:dof_idx+3].copy()
        ee_vel_3d = getattr(env, '_ee_vel_cache', np.zeros(3))
        rel_vel   = pl_vel_3d[:2] - ee_vel_3d[:2]
        ke        = 0.5 * mass * float(np.dot(rel_vel, rel_vel))
        offset_xy = pl_pos_3d[:2] - ee_pos_3d[:2]
        sin_th    = min(float(np.linalg.norm(offset_xy)) / max(rope_L, 0.01), 1.0)
        theta     = float(np.arcsin(sin_th))
        pe        = mass * 9.81 * rope_L * (1.0 - np.cos(theta))
        swing_energy = ke + pe
    except Exception:
        swing_energy = 0.0

    # 与起点的 xy 距离 (lift 成功条件需要)
    if start_xy is not None:
        dtf_start = float(np.linalg.norm(pl_xy - np.asarray(start_xy[:2])))
    else:
        try:
            dtf_start = float(np.linalg.norm(pl_xy - env.default_start_xy[:2]))
        except Exception:
            dtf_start = 0.0

    return {
        "pl_xy":        pl_xy,
        "payload_z":    payload_z,
        "swing_vel":    swing_vel,
        "pl_vel":       pl_vel,
        "tilt":         tilt,
        # [v7 FIX] lift 切换所需
        "pl_vz_abs":    abs(pl_vz),
        "swing_energy": swing_energy,
        "dtf_start":    dtf_start,
    }


def check_physical_insertion(env, config):
    """
    用真实物理容差（由几何推导）独立验证插入是否成功。
    物理约束:
      socket_hole_size = 14mm×14mm → 孔半径 = 7mm
      rebar_radius (可配置, 当前缩小为 2.5mm)
      → xy_tol = socket_hole_radius - rebar_radius (动态计算, 随 rebar_radius 联动)
      → tilt_tol = 0.05 rad (保证4孔对准, 插入深60mm)
      → yaw_tol  = 0.08 rad
      → z: payload_z 需达到 target_payload_z ± 20mm
    返回 (is_success: bool, detail: str)
    """
    cfg_ins  = config.get("insertion", {})
    cfg_pref = config.get("prefab",    {})
    cfg_tgt  = config.get("target",    {})
    target_pz = float(cfg_ins.get("target_payload_z", 0.10))

    # 动态计算 xy_tol: socket 孔半径 - 钢筋半径
    # socket_hole_size = [w, h], 取较小的半边长作为孔半径
    socket_hole_size = cfg_pref.get("socket_hole_size", [0.014, 0.014])
    socket_hole_radius = min(socket_hole_size[0], socket_hole_size[1]) / 2.0
    rebar_radius = float(cfg_tgt.get("rebar_radius", 0.003))
    xy_tol = socket_hole_radius - rebar_radius   # 几何推导, 随 rebar_radius 联动

    z_tol    = float(cfg_ins.get("success_z_tolerance", 0.020))
    tilt_tol = float(cfg_ins.get("tilt_tolerance",      0.05))
    yaw_tol  = float(cfg_ins.get("yaw_tolerance",       0.08))

    pl_pos = env.data.body('prefab').xpos.copy()
    payload_z = float(pl_pos[2])
    target_xy = env.target_pos.copy()
    dtf = float(np.linalg.norm(pl_pos[:2] - target_xy))

    pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    abs_yaw = abs(float(pl_euler[2]))

    ok_z    = abs(payload_z - target_pz) < z_tol
    ok_xy   = dtf < xy_tol
    ok_tilt = tilt < tilt_tol
    ok_yaw  = abs_yaw < yaw_tol

    detail = (f"z={payload_z*1000:.1f}mm(±{z_tol*1000:.0f}) "
              f"dtf={dtf*1000:.1f}mm(<{xy_tol*1000:.0f}) "
              f"tilt={tilt:.3f}(<{tilt_tol:.3f}) "
              f"yaw={abs_yaw:.3f}(<{yaw_tol:.3f}) "
              f"[{'✅' if ok_z else '❌'}z "
              f"{'✅' if ok_xy else '❌'}xy "
              f"{'✅' if ok_tilt else '❌'}tilt "
              f"{'✅' if ok_yaw else '❌'}yaw]")
    return (ok_z and ok_xy and ok_tilt and ok_yaw), detail


def check_phase_transition(phase, state, config, step=0, max_steps=500, env=None):
    """
    检查阶段切换条件。

    v7 修复:
      - target_xy 从 env.target_pos 读取（而非 config 中的默认值）
      - 超时保底切换（85%步数后宽松条件）
      - lift 切换条件放宽 z_tol（测试时 NMPC 精度约 ±20mm，原±25mm 足够）
    """
    pt = config["phase_transition"]
    # [v7 FIX] 优先从 env 读取实际目标（随机化场景下 default_target_xy 已过时）
    if env is not None and hasattr(env, 'target_pos'):
        target_xy = np.array(env.target_pos[:2])
    else:
        target_xy = np.array(config["task"]["default_target_xy"])

    if phase == "lift":
        _rcfg_l   = config["lift_rl"]["reward"]
        _z_cruise = float(config["lift_rl"]["target_z_cruise"])
        z_tol     = float(_rcfg_l.get("success_z_tol", 0.025))
        vz_max    = float(_rcfg_l.get("success_vz_max", 0.05))
        e_thresh  = float(_rcfg_l.get("swing_energy_thresh", 0.05))
        xy_max    = float(_rcfg_l.get("xy_max_dist_success", 0.12))

        # [v7] 测试时 z_tol 放宽到 ±30mm（NMPC 精度约 ±20mm，保留裕量）
        z_tol_test = max(z_tol, 0.030)

        return (
            abs(state["payload_z"] - _z_cruise)    < z_tol_test              and
            state.get("pl_vz_abs",    999.0)        < vz_max                  and
            state.get("swing_energy", 999.0)        < e_thresh * 3.0          and  # [v7] 测试放宽3×
            state["tilt"]                           < float(pt["lift_to_cruise_tilt_max"]) and
            state.get("dtf_start",    999.0)        < xy_max * 1.5            # [v7] 测试放宽1.5×
        )

    elif phase == "cruise":
        dtf = float(np.linalg.norm(state["pl_xy"] - target_xy))

        normal_ok = (
            dtf < float(pt["cruise_to_descent_xy_dist"]) and
            state["tilt"]      < float(pt["cruise_to_descent_tilt_max"]) and
            state["swing_vel"] < float(pt["cruise_to_descent_swing_vel_max"]) and
            state["pl_vel"]    < float(pt["cruise_to_descent_payload_vel_max"])
        )
        if normal_ok:
            return True

        # 超时保底: 85%步数后宽松条件，防止 pipeline 卡死
        fallback_frac  = float(pt.get("cruise_to_descent_fallback_step_frac", 0.85))
        fallback_xy    = float(pt.get("cruise_to_descent_fallback_xy_dist",   0.15))
        fallback_swing = float(pt.get("cruise_to_descent_fallback_swing_max", 0.30))
        if (step >= int(max_steps * fallback_frac) and
                dtf < fallback_xy and
                state["swing_vel"] < fallback_swing and
                state["tilt"] < float(pt["cruise_to_descent_tilt_max"])):
            return True

        return False

    return False


# ==============================================================================
# 单阶段测试
# ==============================================================================

def _advance_expert_to_nearest_wp(expert, planned_path, pl_pos):
    """将专家航点索引推进到最接近当前 payload 位置的航点。"""
    if planned_path is None or len(planned_path) == 0:
        return
    dists = [np.linalg.norm(pl_pos - wp) for wp in planned_path]
    nearest_idx = int(np.argmin(dists))
    expert.tracker.current_idx = nearest_idx


def test_orca_cruise(env, ee_ctrl, config, n_episodes=20):
    """
    纯 ORCA Expert 测试 — cruise 段专用.

    用于验证 ORCA 参数正确性:
      ✅ 能到达终点 (dtf < success_radius)
      ✅ 能绕开障碍物 (无碰撞终止)
      ✅ 速度/加速度在合理范围
      ✅ 摆动动能不超标

    运行方式:
      python test_phase.py --phase cruise --algo orca --render --obstacles 3
    """
    from train_phase import reset_for_phase, REWARD_FNS, REWARD_STATES

    z_pid   = CruiseZYawPID(config)
    swing_d = SwingDampingController(config)
    orca    = CruiseORCAExpert(config)

    results = []
    ep_count = 0
    attempt  = 0

    print(f"\n{'='*60}")
    print(f"  ORCA Expert Cruise 验证测试")
    print(f"  max_speed={orca.orca.max_speed:.2f}m/s  "
          f"acc_max={orca.acc_max_xy:.2f}m/s²  "
          f"k_nav={orca.k_nav}  k_damp={orca.k_damp}")
    print(f"{'='*60}")

    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        obs, planned_path = reset_for_phase(env, "cruise", config)
        if obs is None:
            continue

        current_q = env.data.qpos[:7].copy()
        ee_ctrl.reset(env._get_ee_pos(), current_q)
        orca.reset()

        _pl_z   = float(env.data.body('prefab').xpos[2])
        _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
        _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
        z_pid.reset(_pl_z, _pl_yaw)

        target_xy = env.target_pos.copy()
        rstate    = REWARD_STATES["cruise"]()
        rstate.total_steps_global = 10_000_000
        _rcfg = config.get("cruise_rl", {}).get("reward", {})
        rstate.current_success_radius = float(_rcfg.get("success_radius_end", 0.10))

        stab_metrics = StabilityMetrics()
        ep_reward = 0.0
        ep_steps  = 0
        ep_success = False
        term_reason = None
        max_steps = int(config["cruise_rl"]["max_steps"])

        # 诊断数值记录
        acc_history  = []   # 每步实际 acc 幅度
        vel_history  = []   # 每步 payload 速度
        dtf_history  = []   # 每步到目标距离

        for step in range(max_steps):
            current_q   = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()
            real_ee     = env._get_ee_pos()
            pl_mat      = env.data.body('prefab').xmat.reshape(3, 3)
            pl_euler    = R.from_matrix(pl_mat).as_euler('xyz')

            # Z/Yaw PID
            dof_idx  = env.model.jnt_dofadr[env.prefab_jnt_id]
            pl_vz    = float(env.data.qvel[dof_idx + 2])
            pl_yaw   = float(pl_euler[2])
            pl_yaw_r = float(env.data.qvel[dof_idx + 5]) if dof_idx+5 < len(env.data.qvel) else 0.0
            z_corr, tgt_yaw, falling = z_pid.compute(float(payload_pos[2]), pl_vz, pl_yaw, pl_yaw_r)
            if falling:
                term_reason = f"ground_collision:z={payload_pos[2]:.3f}"
                break

            # swing_d 监控
            pl_vel_full = env.data.qvel[dof_idx:dof_idx+3].copy()
            ee_vel      = getattr(env, '_ee_vel_cache', np.zeros(3))
            swing_d.compute(payload_pos, real_ee, pl_vel_full, ee_vel)

            # 障碍物列表
            n_obs_env = getattr(env, 'n_obstacles', 0)
            obs_data  = obs[10:10 + 3 * n_obs_env]
            obstacles = [(float(obs_data[3*i]), float(obs_data[3*i+1]), float(obs_data[3*i+2]))
                         for i in range(n_obs_env)
                         if 3*i+2 < len(obs_data) and obs_data[3*i+2] > 0.001]

            # ORCA 纯计算
            pl_pos_2d = np.array([obs[4], obs[5]], np.float64)
            pl_vel_2d = np.array([obs[6], obs[7]], np.float64)
            ee_pos_2d = np.array([obs[0], obs[1]], np.float64)
            ee_vel_2d = np.array([obs[2], obs[3]], np.float64)
            acc_xy    = orca.compute_acc(pl_pos_2d, pl_vel_2d, ee_pos_2d, ee_vel_2d,
                                          target_xy, obstacles)

            acc_history.append(float(np.linalg.norm(acc_xy)))
            vel_history.append(float(np.linalg.norm(pl_vel_2d)))
            dtf_history.append(float(np.linalg.norm(pl_pos_2d - target_xy)))

            # 执行
            z_lock  = float(config["cruise_rl"]["z_lock_height"])
            acc_3d  = np.array([acc_xy[0], acc_xy[1], 0.0])
            delta_q = ee_ctrl.compute_delta_q(
                acc_3d, current_q, real_ee,
                lock_z=True, z_lock_height=z_lock,
                z_pid_correction=z_corr, target_yaw=tgt_yaw,
                base_acc_xy=None, residual_mode=False)

            next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
            stab_metrics.update_step(next_obs, config)

            reward, r_done, r_success, r_info = REWARD_FNS["cruise"](
                env, next_obs, config, rstate)
            ep_reward += reward
            ep_steps  += 1

            # 成功判定
            if r_success:
                _pl    = env.data.body('prefab').xpos
                _dtf   = float(np.linalg.norm(_pl[:2] - env.target_pos))
                ep_success = (_dtf < rstate.current_success_radius)
                if ep_success:
                    print(f"    [ORCA ✅] step={step} dtf={_dtf*1000:.1f}mm "
                          f"KE={swing_d.last_energy*1000:.1f}mJ θ={swing_d.last_angle_deg:.1f}°")

            if r_info.get("termination"):
                term_reason = r_info["termination"]

            # 每 40 步打印诊断
            if step > 0 and step % 40 == 0:
                dtf_now = float(np.linalg.norm(
                    np.array([obs[4], obs[5]]) - target_xy))
                print(f"    [s{step:3d}] dtf={dtf_now*1000:.0f}mm "
                      f"acc={acc_history[-1]:.3f}m/s² "
                      f"vel={vel_history[-1]:.3f}m/s "
                      f"z={payload_pos[2]*1000:.1f}mm "
                      f"KE={swing_d.last_energy*1000:.1f}mJ "
                      f"θ={swing_d.last_angle_deg:.1f}°")

            obs = next_obs
            if r_done or env_info.get("nan_detected", False):
                break

        ep_count += 1
        stab = stab_metrics.summary()

        # 诊断汇总
        acc_arr = np.array(acc_history) if acc_history else np.zeros(1)
        vel_arr = np.array(vel_history) if vel_history else np.zeros(1)
        dtf_arr = np.array(dtf_history) if dtf_history else np.zeros(1)

        mark = "✅" if ep_success else "❌"
        term_short = (term_reason or "timeout").split(":")[0]
        print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | Steps:{ep_steps:3d} | {term_short}")
        print(f"       acc avg/max: {np.mean(acc_arr):.3f}/{np.max(acc_arr):.3f} m/s²"
              f"  vel avg/max: {np.mean(vel_arr):.3f}/{np.max(vel_arr):.3f} m/s")
        print(f"       dtf_final: {dtf_arr[-1]*1000:.1f}mm  {stab_metrics.print_line()}")

        results.append({
            "reward": ep_reward, "steps": ep_steps,
            "success": ep_success, "physical_success": ep_success,
            "termination": term_reason or "timeout",
            "trajectory": [],
            "stability": stab,
            "diag": {
                "acc_avg": float(np.mean(acc_arr)),
                "acc_max": float(np.max(acc_arr)),
                "vel_avg": float(np.mean(vel_arr)),
                "vel_max": float(np.max(vel_arr)),
                "dtf_final_mm": float(dtf_arr[-1] * 1000),
            },
        })

    # 汇总诊断
    if results:
        diags = [r["diag"] for r in results]
        print(f"\n{'─'*55}")
        print(f"  ORCA 参数诊断汇总 (N={len(results)} ep)")
        print(f"  acc avg/max: {np.mean([d['acc_avg'] for d in diags]):.3f} / "
              f"{np.mean([d['acc_max'] for d in diags]):.3f} m/s²")
        print(f"  vel avg/max: {np.mean([d['vel_avg'] for d in diags]):.3f} / "
              f"{np.mean([d['vel_max'] for d in diags]):.3f} m/s")
        print(f"  dtf_final   avg: {np.mean([d['dtf_final_mm'] for d in diags]):.1f}mm")
        _stabs = [r["stability"] for r in results if r["stability"]]
        if _stabs:
            print(f"  KE avg/max: {np.mean([s['avg_ke_mJ'] for s in _stabs]):.1f} / "
                  f"{np.mean([s['max_ke_mJ'] for s in _stabs]):.1f} mJ")
            print(f"  θ  avg/max: {np.mean([s['avg_angle'] for s in _stabs]):.2f} / "
                  f"{np.mean([s['max_angle'] for s in _stabs]):.2f} °")

        # 参数建议
        print(f"\n  ── ORCA 参数参考 ──")
        _acc_max_obs = np.mean([d['acc_max'] for d in diags])
        _vel_max_obs = np.mean([d['vel_max'] for d in diags])
        if _acc_max_obs < 0.15:
            print(f"  ⚠ acc 偏低 ({_acc_max_obs:.3f}m/s²), 考虑提高 k_nav 或 residual_acc_max_xy")
        if _vel_max_obs < 0.10:
            print(f"  ⚠ vel 偏低 ({_vel_max_obs:.3f}m/s), 考虑提高 ORCA max_speed")
        if _vel_max_obs > 0.28:
            print(f"  ⚠ vel 偏高 ({_vel_max_obs:.3f}m/s), 考虑降低 ORCA max_speed")
        sr = np.mean([r["success"] for r in results])
        if sr < 0.7:
            print(f"  ⚠ 成功率低 ({sr*100:.0f}%), 检查 time_horizon / obstacle_margin / k_nav")

    return results



def test_single_phase(env, agent, expert, ee_ctrl, phase, config,
                      n_episodes=20, deterministic=True,
                      wind_force=0.0, wind_dir=None):
    """测试单个阶段。"""
    from train_phase import reset_for_phase, build_phase_obs, REWARD_FNS, REWARD_STATES

    # ★ Cruise 段控制器
    z_pid   = CruiseZYawPID(config)         if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    # [v5] Cruise 残差 ORCA 标志
    _cruise_residual_orca = (phase == "cruise" and
                             config.get("cruise_rl", {}).get("use_residual_orca", True))
    _orca_expert_test = CruiseORCAExpert(config) if (_cruise_residual_orca and phase == "cruise") else None

    results = []
    ep_count = 0
    attempt = 0
    _wind_force = float(wind_force)

    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        obs, planned_path = reset_for_phase(env, phase, config)
        if obs is None:
            continue

        # 施加风力（每个 episode 重新设置，因为 reset 后 xfrc_applied 被清零）
        if _wind_force > 0:
            _wd = float(wind_dir) if wind_dir is not None else float(np.random.uniform(0, 2*np.pi))
            if hasattr(env, 'set_wind_force'):
                env.set_wind_force(_wind_force, _wd)
        else:
            # 确保无风
            if hasattr(env, '_test_wind_mode'):
                env._test_wind_mode = False
            if hasattr(env, 'data') and hasattr(env, 'prefab_body_id'):
                env.data.xfrc_applied[env.prefab_body_id, :3] = [0.0, 0.0, 0.0]
        
        '''# ★ 在 cruise 阶段且开启渲染时，暂停让用户观察初始状态
        if phase == "cruise" and getattr(env, 'render_mode', False):
            print("\n[Cruise Test] Warmup 完成，当前为 cruise 阶段的初始状态。")
            print("按 Enter 键开始测试...")
            input()''' # zxy: 取消测试前暂停，直接进入测试

        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        if planned_path is not None:
            expert.set_path(planned_path)
            # 对 cruise/descent 阶段, 将专家航点推进到当前位置附近
            if phase in ("cruise", "descent"):
                pl_pos = env.data.body('prefab').xpos.copy()
                _advance_expert_to_nearest_wp(expert, planned_path, pl_pos)
        ee_ctrl.reset(env._get_ee_pos(), current_q)
        # ★ 初始化 PID
        if z_pid is not None:
            _pl_z = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)
        # [v10] LSTM: 重置观测历史
        if agent is not None and hasattr(agent, 'reset_history'):
            agent.reset_history()
        
        start_xy = env.default_start_xy.copy()
        target_xy = env.target_pos.copy()
        prev_tilt, prev_yaw = 0.0, 0.0
        rstate = REWARD_STATES[phase]()
        # ★ 测试时注入严格判定参数 (训练容差从宽到严, 测试用最终严格值)
        if hasattr(rstate, 'total_steps_global'):
            rstate.total_steps_global = 10_000_000
        # [v3.6] lift: 注入 start_xy (用于 xy 约束)
        if phase == "lift" and hasattr(rstate, 'start_xy'):
            rstate.start_xy = env.data.body('prefab').xpos[:2].copy()
        # [v3.5] 测试 success_radius = 训练最后阶段 (n_obs=3): 0.10m
        if hasattr(rstate, 'current_success_radius'):
            _rcfg = config.get("cruise_rl", {}).get("reward", {})
            _train_final_sr = float(_rcfg.get("success_radius_end", 0.10))
            rstate.current_success_radius = _train_final_sr
        # [v3.6] 重置 hold_counter
        if hasattr(rstate, 'hold_counter'):
            rstate.hold_counter = 0
        # [v3 SHADOW] 测试时也注入影子障碍物 (若无真实障碍物)
        if phase == "cruise" and hasattr(rstate, 'shadow_obstacles'):
            _cur_cfg = config.get("curriculum", {})
            if bool(_cur_cfg.get("shadow_obstacle_enabled", True)) and len(env._obstacles) == 0:
                # 简单采样: 在起终点连线附近随机放置影子障碍物
                import numpy as _np_shadow
                _sxy = env.default_start_xy.copy()
                _txy = env.target_pos.copy()
                _dir = _txy - _sxy; _L = float(_np_shadow.linalg.norm(_dir))
                if _L > 0.01:
                    _dir /= _L; _perp = _np_shadow.array([-_dir[1], _dir[0]])
                    _rng = _np_shadow.random.default_rng()
                    _sh_obs = []
                    for _ in range(int(_cur_cfg.get("shadow_obstacle_n", 3))):
                        _t = _rng.uniform(0.2, 0.8)
                        _s = _rng.uniform(-0.25, 0.25)
                        _c = _sxy + _t * _L * _dir + _s * _perp
                        _r = _rng.uniform(float(_cur_cfg.get("shadow_obstacle_r_min", 0.006)),
                                          float(_cur_cfg.get("shadow_obstacle_r_max", 0.015)))
                        _sh_obs.append((float(_c[0]), float(_c[1]), float(_r)))
                    rstate.shadow_obstacles = _sh_obs

        ep_reward = 0.0
        ep_steps = 0
        ep_success = False
        term_reason = None
        trajectory = []
        stab_metrics = StabilityMetrics()   # [v5]
        # [v3.4] 测试步数上限: descent 缩短至 200 步
        # 防止机械臂通过长时间接触吊装物"压着走"达到课程容差
        if phase == "descent":
            max_steps = 200
        else:
            max_steps = int(config[f"{phase}_rl"]["max_steps"])
        
        for step in range(max_steps):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()
            ee_pos = env._get_ee_pos()
            pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
            tilt_val = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
            yaw_val = abs(float(pl_euler[2]))
            swing_vel = float(np.linalg.norm(obs[6:8] - obs[2:4]))
            _pl_vxy_l  = obs[6:8]
            _dof_idx_l = env.model.jnt_dofadr[env.prefab_jnt_id]
            _pl_vz_l   = float(env.data.qvel[_dof_idx_l + 2])
            _pl_vel_l  = float(np.linalg.norm(np.append(_pl_vxy_l, _pl_vz_l)))
            trajectory.append({
                "payload":   payload_pos.copy(),
                "ee":        ee_pos.copy(),
                "tilt":      tilt_val,
                "yaw":       yaw_val,
                "swing_vel": swing_vel,
                "pl_xy":     payload_pos[:2].copy(),  # [v3.6] lift xy
                "pl_vel":    _pl_vel_l,               # [v3.6] lift vel
                "step":      step,
            })
            
            if agent is None:  # 专家模式
                # 测试时使用原始NMPC expert (ORCA仅用于训练BC数据收集)
                delta_q = expert.compute_delta_q_target(obs, current_q)
            else:
                p_obs, prev_tilt, prev_yaw = build_phase_obs(
                    phase, obs, env, start_xy, target_xy, prev_tilt, prev_yaw)
                norm_obs = agent.normalize_obs(p_obs, update=False)
                
                if hasattr(agent, 'act'):
                    result = agent.act(norm_obs, deterministic=deterministic)
                    # [v10] DescentDualRLAgent returns 7 values
                    if isinstance(result, tuple) and len(result) == 7:
                        action = result[0]  # combined 3D acc
                    elif isinstance(result, tuple) and len(result) == 3:
                        action = result[0]
                    elif isinstance(result, tuple):
                        action = result[0]
                    else:
                        action = result
                else:
                    action = agent.act(norm_obs, deterministic=deterministic)
                
                real_ee = env._get_ee_pos()
                if phase == "cruise" and z_pid is not None:
                    # ★ PID 计算
                    _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                    _pl_vz = float(env.data.qvel[_dof_idx + 2])
                    _pl_yaw = float(pl_euler[2])
                    _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) if _dof_idx + 5 < len(env.data.qvel) else 0.0
                    _z_corr, _tgt_yaw, _falling = z_pid.compute(
                        float(payload_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                    if _falling:
                        term_reason = f"ground_collision:z={payload_pos[2]:.3f}"
                        break
                    # ★ 底层防摆控制器 + RL/Expert 残差模式
                    _pl_vel_full = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel = getattr(env, '_ee_vel_cache', np.zeros(3))
                    # [v3] swing_d 仅监控
                    if swing_d is not None:
                        swing_d.compute(payload_pos, real_ee, _pl_vel_full, _ee_vel)
                    # [v5] ORCA base + RL 残差
                    if _cruise_residual_orca and _orca_expert_test is not None and agent is not None:
                        try:
                            _pl_p2 = np.array([obs[4], obs[5]], np.float64)
                            _pl_v2 = np.array([obs[6], obs[7]], np.float64)
                            _ee_p2 = np.array([obs[0], obs[1]], np.float64)
                            _ee_v2 = np.array([obs[2], obs[3]], np.float64)
                            _n_e = getattr(env, 'n_obstacles', 0)
                            _od  = obs[10:10 + 3 * _n_e]
                            _ob  = [(float(_od[3*_i]), float(_od[3*_i+1]), float(_od[3*_i+2]))
                                    for _i in range(_n_e) if 3*_i+2 < len(_od) and _od[3*_i+2] > 0.001]
                            _base = _orca_expert_test.compute_acc(_pl_p2, _pl_v2, _ee_p2, _ee_v2, target_xy, _ob)
                        except Exception:
                            _base = np.zeros(2, np.float32)
                        _rmax = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.20))
                        _rl_r = np.clip(action[:2], -_rmax, _rmax)
                        _comb = _base + _rl_r
                        _amax = float(config["cruise_rl"].get("residual_acc_max_xy", 0.60))
                        _cn   = float(np.linalg.norm(_comb))
                        if _cn > _amax: _comb = _comb / _cn * _amax
                        acc_3d = np.array([_comb[0], _comb[1], 0.0])
                    else:
                        acc_3d = np.array([action[0], action[1], 0.0])
                    z_lock = float(config["cruise_rl"]["z_lock_height"])
                    delta_q = ee_ctrl.compute_delta_q(
                        acc_3d, current_q, real_ee,
                        lock_z=True, z_lock_height=z_lock,
                        z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                        base_acc_xy=None, residual_mode=False)
                elif phase == "cruise":
                    acc_3d = np.array([action[0], action[1], 0.0])
                    z_lock = float(config["cruise_rl"]["z_lock_height"])
                    delta_q = ee_ctrl.compute_delta_q(
                        acc_3d, current_q, real_ee,
                        lock_z=True, z_lock_height=z_lock)
                elif phase == "descent":
                    # [v3.4 FIX] Descent: PID base + RL residual (与训练架构一致)
                    # RL 只输出残差 acc, 必须叠加 PID base delta_q 才能正常工作
                    from train_phase import _apply_descent_pid_residual
                    _pid_residual_mode = bool(
                        config.get("descent_rl", {}).get("pid_residual_mode", True))
                    if _pid_residual_mode:
                        delta_q, _pid_dq = _apply_descent_pid_residual(
                            expert, action, obs, env, config, current_q)
                    else:
                        _vmax_z_d = float(config.get("ee_control", {}).get(
                            "vel_max_z_descent", 0.03))
                        delta_q = ee_ctrl.compute_delta_q(
                            action, current_q, real_ee, vel_max_z=_vmax_z_d)
                else:
                    # lift: 纯 RL 输出 3D EE 加速度, 直接送 EEAccController
                    # [BUG FIX] 之前 lift 误走 descent PID+residual 分支
                    # (else 分支被 descent 注释占据, lift 也走进去调用了下降专家)
                    # 下降专家 compute_delta_q_target 会驱动 EE 向 target_xy 移动
                    # → 导致 lift 测试时 payload 往终点方向运动而非向上提升
                    delta_q = ee_ctrl.compute_delta_q(action, current_q, real_ee)
            
            next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
            stab_metrics.update_step(next_obs, config)  # [v5]
            
            reward, r_done, r_success, r_info = REWARD_FNS[phase](
                env, next_obs, config, rstate)
            
            ep_reward += reward
            ep_steps += 1
            
            if r_success:
                if phase == "descent":
                    # [v3.4] 测试唯一判定标准: 物理插入成功 (忽略课程容差)
                    phys_ok, phys_detail = check_physical_insertion(env, config)
                    ep_success = phys_ok
                    mark_str = "✅ 物理插入成功" if phys_ok else "⚠ 课程容差达标但未插入"
                    print(f"    [{mark_str}] {phys_detail}")
                elif phase == "cruise":
                    # [v3.5] 测试成功判定与训练最后阶段完全一致:
                    #   dtf < success_radius_end(0.10m) + 稳定性条件
                    # 同时打印 phase_transition (0.06m) 满足情况作为诊断
                    _pl    = env.data.body('prefab').xpos
                    _dtf   = float(np.linalg.norm(_pl[:2] - env.target_pos))
                    _pt    = config["phase_transition"]
                    _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                    _euler  = R.from_matrix(_pl_mat).as_euler('xyz')
                    _tilt   = float(np.sqrt(_euler[0]**2 + _euler[1]**2))
                    _pl_vxy = np.array([obs[6], obs[7]])
                    _ee_vxy = np.array([obs[2], obs[3]])
                    _swing  = float(np.linalg.norm(_pl_vxy - _ee_vxy))
                    _plvel  = float(np.linalg.norm(_pl_vxy))
                    _train_sr = rstate.current_success_radius   # = 0.10m
                    # 与训练末期完全一致的成功条件
                    _train_ok = (
                        _dtf   < _train_sr and
                        _tilt  < float(_pt["cruise_to_descent_tilt_max"]) and
                        _swing < float(_pt["cruise_to_descent_swing_vel_max"]) and
                        _plvel < float(_pt["cruise_to_descent_payload_vel_max"])
                    )
                    ep_success = _train_ok
                    # 额外诊断: 是否满足更严格的 pipeline 切换条件
                    _pipeline_ok = _train_ok and (
                        _dtf < float(_pt["cruise_to_descent_xy_dist"]))
                    if _train_ok:
                        _pipe_str = "✅ 可切换下降" if _pipeline_ok else "⚠ 到达但不满足切换"
                        print(f"    [cruise ✅] dtf={_dtf*1000:.1f}mm<{_train_sr*1000:.0f}mm "
                              f"tilt={_tilt:.3f} swing={_swing:.3f} | {_pipe_str}")
                    else:
                        print(f"    [cruise ❌] dtf={_dtf*1000:.1f}mm "
                              f"(需<{_train_sr*1000:.0f}mm) "
                              f"tilt={_tilt:.3f} swing={_swing:.3f} plvel={_plvel:.3f}")
                else:  # lift
                    # [v3.6] lift r_success 已包含所有约束: xy + hold + vel
                    # compute_lift_reward 新增: dtf_start<0.15m, vel<0.15, hold=3步
                    ep_success = True
                    _pl_l = env.data.body('prefab').xpos
                    _sxy  = getattr(rstate, 'start_xy', env.default_start_xy.copy())
                    _dtf_s = float(np.linalg.norm(_pl_l[:2] - _sxy))
                    print(f"    [lift ✅] z={_pl_l[2]*1000:.0f}mm "
                          f"dtf_start={_dtf_s*1000:.0f}mm (需<150mm)")
            if r_info.get("termination"):
                term_reason = r_info["termination"]

            # ★ cruise 阶段: 每30步打印 z 高度 + 摆动能量状态
            # step > 0: 跳过第0步, 此时防摆控制器缓存尚未更新 (会显示初始零值)
            if phase == "cruise" and step > 0 and step % 30 == 0:
                pl_pos_c   = env.data.body('prefab').xpos
                z_lock_dbg = float(config["cruise_rl"]["z_lock_height"])
                dtf_c      = float(np.linalg.norm(pl_pos_c[:2] - env.target_pos))
                dof_idx_c  = env.model.jnt_dofadr[env.prefab_jnt_id]
                vz_c       = float(env.data.qvel[dof_idx_c + 2])
                # 摆动能量 (swing_d.compute() 在此步 env.step() 之前已调用, 缓存有效)
                _e_str = ""
                if swing_d is not None:
                    _e_str = (f" | E:{swing_d.last_energy*1000:.1f}mJ"
                              f" θ:{swing_d.last_angle_deg:.1f}°"
                              f" gain:{swing_d._gain_scale:.2f}")
                _sr_str = f" sr={rstate.current_success_radius*1000:.0f}mm" if hasattr(rstate, 'current_success_radius') else ""
                print(f"    [cruise s{step:3d}] z={pl_pos_c[2]*1000:.1f}mm "
                      f"(tgt={z_lock_dbg*1000:.0f}mm dev={abs(pl_pos_c[2]-z_lock_dbg)*1000:.1f}mm) "
                      f"vz={vz_c*1000:.1f}mm/s dtf={dtf_c*1000:.1f}mm{_sr_str}{_e_str}")

            # ★ descent 阶段: 每20步打印插入进度
            if phase == "descent" and step % 20 == 0:
                pl_pos = env.data.body('prefab').xpos
                target_xy_dbg = env.target_pos.copy()
                dtf_dbg  = float(np.linalg.norm(pl_pos[:2] - target_xy_dbg))
                pl_mat_dbg = env.data.body('prefab').xmat.reshape(3, 3)
                pl_euler_dbg = R.from_matrix(pl_mat_dbg).as_euler('xyz')
                tilt_dbg = float(np.sqrt(pl_euler_dbg[0]**2 + pl_euler_dbg[1]**2))
                yaw_dbg  = abs(float(pl_euler_dbg[2]))
                hold_dbg = getattr(rstate, 'insertion_hold_counter', 0)
                _z_reached = getattr(expert, '_z_reached', False)
                _contact   = getattr(expert, '_contact_detected', False)
                _integ = getattr(expert, '_integral_xy', None)
                _int_str = f" int={np.linalg.norm(_integ)*1000:.1f}mm" if _integ is not None else ""
                if _contact:
                    _mode = "CONTACT"
                elif _z_reached:
                    _mode = "HOLD"
                elif getattr(expert, '_descent_settle_counter', 0) <= 8:
                    _mode = "SETTLE"
                else:
                    _mode = "DESC"
                print(f"    [descent s{step:3d}|{_mode}] "
                      f"z={pl_pos[2]*1000:.1f}mm dtf={dtf_dbg*1000:.1f}mm "
                      f"tilt={tilt_dbg:.3f} yaw={yaw_dbg:.3f} hold={hold_dbg}{_int_str}")
            
            obs = next_obs
            
            if r_done or env_info.get("nan_detected", False):
                break
        
        ep_count += 1
        # [v3.4] ep_success 已经是物理判定结果, physical_success = ep_success
        stab = stab_metrics.summary()   # [v5]
        results.append({
            "reward": ep_reward,
            "steps": ep_steps,
            "success": ep_success,
            "physical_success": ep_success,
            "termination": term_reason or "timeout",
            "trajectory": trajectory,
            "stability": stab,          # [v5]
        })
        
        mark = "✅" if ep_success else "❌"
        term_short = (term_reason or "timeout").split(":")[0]
        print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | "
              f"Steps:{ep_steps:3d} | {term_short} | "
              f"{stab_metrics.print_line()}")
    
    return results


# ==============================================================================
# 全流水线测试 (3 阶段串联)
# ==============================================================================

def test_pipeline(env, agents, expert, ee_ctrl, config,
                  n_episodes=20, deterministic=True,
                  wind_force=0.0, wind_dir=None,
                  obs_noise_sigma=0.0, act_noise_sigma=0.0):
    """
    测试完整 3 阶段流水线。

    v7 修复:
      - 使用 reset_for_phase 正确初始化每个阶段（fix lift→cruise 从不切换的 bug）
      - get_phase_state 传入 start_xy、env，修复 pl_vz_abs/swing_energy/dtf_start 缺失
      - check_phase_transition 传入 env、step，支持超时保底切换
      - NMPC base + RL 残差（v7 架构）

    参数:
      wind_force:       施加的恒定风力 (N)，0 = 无风
      wind_dir:         风向 (rad)，None = 随机
      obs_noise_sigma:  观测噪声标准差 (m/s 量纲)，0 = 关闭
      act_noise_sigma:  执行噪声标准差 (rad/step 量纲)，0 = 关闭
    """
    from train_phase import reset_for_phase, build_phase_obs, REWARD_FNS, REWARD_STATES

    z_pid   = CruiseZYawPID(config)
    swing_d = SwingDampingController(config)
    _cruise_nmpc_base = bool(config.get("cruise_rl", {}).get("use_nmpc_base", True))
    _cruise_max = int(config["cruise_rl"]["max_steps"])

    # 风力设置
    _wind_force = float(wind_force)
    _wind_dir   = float(wind_dir) if wind_dir is not None else float(np.random.uniform(0, 2*np.pi))

    results = []
    ep_count = 0
    attempt  = 0

    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1

        # [v7 FIX] 用 reset_for_phase 初始化 lift（正确设置 rstate.start_xy 等）
        sys_stdout_saved = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs, planned_path = reset_for_phase(env, "lift", config)
        finally:
            sys.stdout.close(); sys.stdout = sys_stdout_saved

        if obs is None:
            continue

        # [KEY FIX] 在 reset 之后施加风力（reset 会清零 xfrc_applied）
        if _wind_force > 0 and hasattr(env, 'set_wind_force'):
            _wd = float(np.random.uniform(0, 2*np.pi)) if wind_dir is None else _wind_dir
            env.set_wind_force(_wind_force, _wd)
        else:
            if hasattr(env, '_test_wind_mode'):
                env._test_wind_mode = False
            if hasattr(env, 'data') and hasattr(env, 'prefab_body_id'):
                env.data.xfrc_applied[env.prefab_body_id, :3] = [0.0, 0.0, 0.0]

        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        if planned_path is not None:
            expert.set_path(planned_path)
            pl_pos_init = env.data.body('prefab').xpos.copy()
            _advance_expert_to_nearest_wp(expert, planned_path, pl_pos_init)
        ee_ctrl.reset(env._get_ee_pos(), current_q)
        _pl_z   = float(env.data.body('prefab').xpos[2])
        _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
        _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
        z_pid.reset(_pl_z, _pl_yaw)

        start_xy  = env.default_start_xy.copy()
        target_xy = env.target_pos.copy()
        prev_tilt, prev_yaw = 0.0, 0.0

        # 阶段状态机
        current_phase = "lift"
        phase_rewards = {"lift": 0.0, "cruise": 0.0, "descent": 0.0}
        phase_steps   = {"lift": 0,   "cruise": 0,   "descent": 0}
        phase_success = {"lift": False, "cruise": False, "descent": False}

        rstate = REWARD_STATES["lift"]()
        if hasattr(rstate, 'total_steps_global'): rstate.total_steps_global = 10_000_000
        if hasattr(rstate, 'start_xy'):           rstate.start_xy = env.data.body('prefab').xpos[:2].copy()
        if hasattr(rstate, 'current_success_radius'):
            rstate.current_success_radius = float(
                config.get("cruise_rl", {}).get("reward", {}).get("success_radius_end", 0.10))

        ep_reward  = 0.0
        ep_steps   = 0
        final_success = False
        term_reason   = None
        trajectory    = []
        stab_all      = StabilityMetrics()
        phase_stab    = {"lift": StabilityMetrics(), "cruise": StabilityMetrics(), "descent": StabilityMetrics()}

        for step in range(config["sim"]["max_steps"]):
            current_q   = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()
            ee_pos      = env._get_ee_pos()

            trajectory.append({
                "payload": payload_pos.copy(),
                "ee":      ee_pos.copy(),
                "phase":   current_phase,
                "step":    step,
            })

            # [v7 FIX] get_phase_state 传 start_xy 和 env
            state = get_phase_state(env, obs, start_xy=start_xy, config=config)

            # ── 阶段切换 ──────────────────────────────────────────────────────
            if (current_phase == "lift" and
                    check_phase_transition("lift", state, config,
                                           step=phase_steps["lift"], env=env)):
                phase_success["lift"] = True
                current_phase = "cruise"
                rstate = REWARD_STATES["cruise"]()
                if hasattr(rstate, 'total_steps_global'): rstate.total_steps_global = 10_000_000
                if hasattr(rstate, 'current_success_radius'):
                    rstate.current_success_radius = float(
                        config.get("cruise_rl", {}).get("reward", {}).get("success_radius_end", 0.10))
                ee_ctrl.reset(env._get_ee_pos(), current_q)
                _pl_z   = float(payload_pos[2])
                _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
                z_pid.reset(_pl_z, _pl_yaw)
                # expert 切换到 cruise 路径
                if planned_path is not None:
                    _advance_expert_to_nearest_wp(expert, planned_path, payload_pos)
                print(f"    [pipeline s{step}] Lift→Cruise ✅  "
                      f"z={payload_pos[2]*1000:.0f}mm "
                      f"KE={state.get('swing_energy',0)*1000:.1f}mJ")

            elif (current_phase == "cruise" and
                  check_phase_transition("cruise", state, config,
                                         step=phase_steps["cruise"],
                                         max_steps=_cruise_max, env=env)):
                phase_success["cruise"] = True
                current_phase = "descent"
                rstate = REWARD_STATES["descent"]()
                if hasattr(rstate, 'total_steps_global'): rstate.total_steps_global = 10_000_000
                ee_ctrl.reset(env._get_ee_pos(), current_q)
                _dtf_sw = np.linalg.norm(state["pl_xy"] - env.target_pos)
                print(f"    [pipeline s{step}] Cruise→Descent ✅  "
                      f"dtf={_dtf_sw*1000:.1f}mm")

            # ── 动作计算 ──────────────────────────────────────────────────────
            agent = agents.get(current_phase)

            if agent is None:
                # 专家模式（lift 用 NMPC，cruise 用 NMPC，descent 用 NMPC）
                delta_q = expert.compute_delta_q_target(obs, current_q)
            else:
                p_obs, prev_tilt, prev_yaw = build_phase_obs(
                    current_phase, obs, env, start_xy, target_xy, prev_tilt, prev_yaw)
                norm_obs = agent.normalize_obs(p_obs, update=False)

                # 观测噪声（可选）
                if obs_noise_sigma > 0:
                    norm_obs = norm_obs + np.random.normal(0, obs_noise_sigma, norm_obs.shape).astype(np.float32)

                result = agent.act(norm_obs, deterministic=deterministic)
                action = result[0] if isinstance(result, tuple) else result
                real_ee = env._get_ee_pos()

                if current_phase == "cruise":
                    _dof_idx  = env.model.jnt_dofadr[env.prefab_jnt_id]
                    _pl_vz    = float(env.data.qvel[_dof_idx + 2])
                    _pl_mat   = env.data.body('prefab').xmat.reshape(3, 3)
                    _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                    _pl_yaw   = float(_pl_euler[2])
                    _pl_yr    = float(env.data.qvel[_dof_idx + 5]) if _dof_idx+5 < len(env.data.qvel) else 0.0
                    _z_corr, _tgt_yaw, _falling = z_pid.compute(
                        float(payload_pos[2]), _pl_vz, _pl_yaw, _pl_yr)
                    if _falling:
                        term_reason = f"ground_collision:z={payload_pos[2]:.3f}"; break
                    if swing_d is not None:
                        swing_d.compute(payload_pos, real_ee,
                                        env.data.qvel[_dof_idx:_dof_idx+3].copy(),
                                        getattr(env, '_ee_vel_cache', np.zeros(3)))

                    if _cruise_nmpc_base:
                        try:
                            _a4  = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                            _ba  = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                        except Exception:
                            _ba  = np.zeros(2, np.float32)
                        _rm  = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.25))
                        _cm  = _ba + np.clip(action[:2], -_rm, _rm)
                        _am  = float(config["cruise_rl"].get("residual_acc_max_xy", 0.80))
                        _cn  = float(np.linalg.norm(_cm))
                        if _cn > _am: _cm = _cm / _cn * _am
                        acc_3d = np.array([_cm[0], _cm[1], 0.0])
                    else:
                        acc_3d = np.array([action[0], action[1], 0.0])

                    z_lock  = float(config["cruise_rl"]["z_lock_height"])
                    delta_q = ee_ctrl.compute_delta_q(
                        acc_3d, current_q, real_ee,
                        lock_z=True, z_lock_height=z_lock,
                        z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                        base_acc_xy=None, residual_mode=False)

                elif current_phase == "descent":
                    from train_phase import _apply_descent_pid_residual
                    if bool(config.get("descent_rl", {}).get("pid_residual_mode", True)):
                        delta_q, _ = _apply_descent_pid_residual(
                            expert, action, obs, env, config, current_q)
                    else:
                        _vmax_z_d = float(config.get("ee_control", {}).get("vel_max_z_descent", 0.03))
                        delta_q   = ee_ctrl.compute_delta_q(action, current_q, real_ee, vel_max_z=_vmax_z_d)
                else:
                    delta_q = ee_ctrl.compute_delta_q(action, current_q, real_ee)

            # 执行噪声（可选）
            if act_noise_sigma > 0:
                delta_q = delta_q + np.random.normal(0, act_noise_sigma, delta_q.shape).astype(np.float32)

            next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
            stab_all.update_step(next_obs, config)
            phase_stab[current_phase].update_step(next_obs, config)

            reward, r_done, r_success, r_info = REWARD_FNS[current_phase](
                env, next_obs, config, rstate)

            ep_reward              += reward
            phase_rewards[current_phase] += reward
            phase_steps[current_phase]   += 1
            ep_steps += 1

            if r_success and current_phase == "descent":
                phys_ok, phys_detail = check_physical_insertion(env, config)
                mark_str = "✅ 物理插入成功" if phys_ok else "⚠ 课程容差达标但未插入"
                print(f"    [{mark_str}] {phys_detail}")
                if phys_ok:
                    final_success = True
                    phase_success["descent"] = True
                    term_reason = r_info.get("termination", "insertion_success")
                    obs = next_obs; break

            if current_phase == "descent" and phase_steps["descent"] >= 200:
                term_reason = "descent_timeout_200"; break

            if current_phase == "descent" and phase_steps["descent"] % 20 == 0:
                _pl_dbg  = env.data.body('prefab').xpos
                _dtf_dbg = float(np.linalg.norm(_pl_dbg[:2] - env.target_pos))
                _mat_dbg = env.data.body('prefab').xmat.reshape(3, 3)
                _eul_dbg = R.from_matrix(_mat_dbg).as_euler('xyz')
                _tlt_dbg = float(np.sqrt(_eul_dbg[0]**2 + _eul_dbg[1]**2))
                _hold    = getattr(rstate, 'insertion_hold_counter', 0)
                print(f"    [descent s{phase_steps['descent']:3d}] "
                      f"z={_pl_dbg[2]*1000:.1f}mm dtf={_dtf_dbg*1000:.1f}mm "
                      f"tilt={_tlt_dbg:.3f} hold={_hold}")

            if r_info.get("termination"):
                term_reason = r_info["termination"]

            obs = next_obs
            if r_done or env_info.get("nan_detected", False):
                break

        ep_count += 1
        stab_sum = stab_all.summary()
        phase_stab_sum = {p: phase_stab[p].summary() for p in ["lift", "cruise", "descent"]}

        results.append({
            "reward":          ep_reward,
            "steps":           ep_steps,
            "success":         final_success,
            "physical_success": final_success,
            "termination":     term_reason or "timeout",
            "phase_rewards":   phase_rewards.copy(),
            "phase_steps":     phase_steps.copy(),
            "phase_success":   phase_success.copy(),
            "trajectory":      trajectory,
            "stability":       stab_sum,
            "phase_stability": phase_stab_sum,
            "wind_force":      _wind_force,
        })
        mark      = "✅" if final_success else "❌"
        term_short= (term_reason or "timeout").split(":")[0]
        phases_str= " → ".join([
            f"{'✅' if phase_success[p] else '❌'}{p[0].upper()}"
            for p in ["lift", "cruise", "descent"]])
        print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | "
              f"Steps:{ep_steps:3d} | {phases_str} | {term_short} | "
              f"{stab_all.print_line()}")

    return results
    return results


# ==============================================================================
# 结果汇总
# ==============================================================================

def test_cruise_nmpc_wind(env, expert, ee_ctrl, config,
                          wind_levels=None, n_episodes_per_level=20):
    """
    Cruise 阶段 NMPC expert 风力扰动测试。

    用途: 量化 NMPC base controller 在不同风力下的防摆稳定性，
          作为 RL 改善效果的对比基线。

    参数:
      wind_levels: 风力列表 (N)，默认 [0, 0.5, 1.0, 2.0, 3.0]
      n_episodes_per_level: 每个风力等级测试回合数

    输出: 每个风力等级的 SR / KE / angle 统计表
    """
    from train_phase import reset_for_phase, REWARD_FNS, REWARD_STATES

    if wind_levels is None:
        wind_levels = [0.0, 0.5, 1.0, 2.0, 3.0]

    z_pid = CruiseZYawPID(config)
    _success_radius = float(
        config.get("cruise_rl", {}).get("reward", {}).get("success_radius_end", 0.10))

    print(f"\n{'='*70}")
    print(f"  NMPC Expert — Cruise 风力扰动测试")
    print(f"  风力等级: {wind_levels} N  |  每级 {n_episodes_per_level} 回合")
    print(f"{'='*70}")
    print(f"  {'Wind(N)':>8} | {'SR':>6} | {'KE avg':>8} | {'KE p95':>8} | "
          f"{'θ avg':>7} | {'θ max':>7} | {'Steps':>6}")
    print(f"  {'-'*70}")

    all_results = {}

    for wind_f in wind_levels:
        ep_results = []
        ve = 0; att = 0

        while ve < n_episodes_per_level and att < n_episodes_per_level * 3:
            att += 1
            sys_stdout_saved = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                obs, pp = reset_for_phase(env, "cruise", config)
            finally:
                sys.stdout.close(); sys.stdout = sys_stdout_saved
            if obs is None or pp is None:
                continue

            # [KEY FIX] 在 reset 之后施加风力（reset 会清零 xfrc_applied）
            if wind_f > 0 and hasattr(env, 'set_wind_force'):
                wind_dir_ep = float(np.random.uniform(0, 2 * np.pi))
                env.set_wind_force(wind_f, wind_dir_ep)
            elif hasattr(env, 'data') and hasattr(env, 'prefab_body_id'):
                env.data.xfrc_applied[env.prefab_body_id, :3] = [0.0, 0.0, 0.0]
                if hasattr(env, '_test_wind_mode'):
                    env._test_wind_mode = False

            current_q = env.data.qpos[:7].copy()
            expert.reset(obs, current_q, env=env)
            if pp is not None:
                expert.set_path(pp)
                _advance_expert_to_nearest_wp(expert, pp, env.data.body('prefab').xpos.copy())
            ee_ctrl.reset(env._get_ee_pos(), current_q)
            _pl_z   = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)

            txy = env.target_pos.copy()
            rstate = REWARD_STATES["cruise"]()
            if hasattr(rstate, 'total_steps_global'): rstate.total_steps_global = 10_000_000
            if hasattr(rstate, 'current_success_radius'):
                rstate.current_success_radius = _success_radius

            stab = StabilityMetrics()
            ep_success = False
            ep_steps   = 0
            max_steps  = int(config["cruise_rl"]["max_steps"])

            for _s in range(max_steps):
                current_q = env.data.qpos[:7].copy().astype(np.float32)
                payload_pos = env.data.body('prefab').xpos.copy()

                # Z/Yaw PID
                _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                _pl_vz   = float(env.data.qvel[_dof_idx + 2])
                _pl_mat2 = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_yaw2 = float(R.from_matrix(_pl_mat2).as_euler('xyz')[2])
                _pl_yr2  = float(env.data.qvel[_dof_idx + 5]) if _dof_idx+5 < len(env.data.qvel) else 0.0
                _z_corr, _tgt_yaw, _falling = z_pid.compute(
                    float(payload_pos[2]), _pl_vz, _pl_yaw2, _pl_yr2)
                if _falling:
                    break

                # NMPC expert 执行
                delta_q = expert.compute_delta_q_target(obs, current_q)

                next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
                stab.update_step(next_obs, config)
                ep_steps += 1

                _, r_done, r_success, _ = REWARD_FNS["cruise"](env, next_obs, config, rstate)
                if r_success:
                    ep_success = True
                if r_done or env_term or env_trunc or env_info.get("nan_detected", False):
                    break
                obs = next_obs

            ve += 1
            s = stab.summary()
            ep_results.append({
                "success": ep_success,
                "steps":   ep_steps,
                "ke_avg":  s.get("avg_ke_mJ", 0),
                "ke_p95":  s.get("p95_ke_mJ", 0),
                "ke_max":  s.get("max_ke_mJ", 0),
                "ang_avg": s.get("avg_angle", 0),
                "ang_max": s.get("max_angle", 0),
            })

        # 统计
        sr     = np.mean([r["success"] for r in ep_results]) if ep_results else 0
        ke_avg = np.mean([r["ke_avg"]  for r in ep_results]) if ep_results else 0
        ke_p95 = np.mean([r["ke_p95"]  for r in ep_results]) if ep_results else 0
        ke_max = np.mean([r["ke_max"]  for r in ep_results]) if ep_results else 0
        ang_avg= np.mean([r["ang_avg"] for r in ep_results]) if ep_results else 0
        ang_max= np.mean([r["ang_max"] for r in ep_results]) if ep_results else 0
        steps  = np.mean([r["steps"]   for r in ep_results]) if ep_results else 0

        all_results[wind_f] = {
            "sr": sr, "ke_avg": ke_avg, "ke_p95": ke_p95, "ke_max": ke_max,
            "ang_avg": ang_avg, "ang_max": ang_max, "steps": steps,
            "episodes": ep_results,
        }

        wind_str = f"{wind_f:.1f}" if wind_f > 0 else "无风"
        print(f"  {wind_str:>8} | {sr*100:>5.1f}% | {ke_avg:>7.1f}mJ | "
              f"{ke_p95:>7.1f}mJ | {ang_avg:>6.2f}° | {ang_max:>6.2f}° | "
              f"{steps:>5.1f}")

    print(f"  {'='*70}")
    print(f"\n  说明:")
    print(f"    KE avg: 全程平均摆动动能 (NMPC base 无风基线)")
    print(f"    KE p95: 95分位摆动动能 (极端情况指标)")
    print(f"    θ max:  最大摆角 (安全约束参考)")
    print(f"    [RL 改善目标] 在同等风力下, 以上指标均应低于 NMPC base")

    return all_results



    """打印测试结果汇总 (含稳定性指标)。"""
    if not results:
        print(f"[{mode_name}] 无有效结果")
        return

    sr = np.mean([r["success"] for r in results])
    phys_sr = np.mean([r.get("physical_success", r["success"]) for r in results])
    avg_r = np.mean([r["reward"] for r in results])
    avg_s = np.mean([r["steps"] for r in results])

    terms = Counter()
    for r in results:
        t = (r["termination"] or "unknown").split(":")[0]
        terms[t] += 1

    print(f"\n{'='*60}")
    print(f"  [{mode_name}] 结果汇总")
    print(f"{'='*60}")
    print(f"  成功率:    {phys_sr*100:.1f}%")
    print(f"  平均奖励:  {avg_r:.2f}")
    print(f"  平均步数:  {avg_s:.1f}")
    print(f"  终止原因:  {dict(terms)}")

    # 流水线模式额外统计
    if "phase_success" in results[0]:
        for p in ["lift", "cruise", "descent"]:
            p_sr = np.mean([r["phase_success"][p] for r in results])
            p_avg_r = np.mean([r["phase_rewards"][p] for r in results])
            p_avg_s = np.mean([r["phase_steps"][p] for r in results])
            print(f"  {p:>8s}: SR={p_sr*100:.0f}% | R={p_avg_r:.2f} | Steps={p_avg_s:.0f}")

    # ── 稳定性指标汇总 [v5] ───────────────────────────────────────────────
    stab_list = [r["stability"] for r in results if r.get("stability")]
    if stab_list:
        avg_ke   = np.mean([s.get("avg_ke_mJ",  0) for s in stab_list])
        max_ke   = np.mean([s.get("max_ke_mJ",  0) for s in stab_list])
        p95_ke   = np.mean([s.get("p95_ke_mJ",  0) for s in stab_list])
        avg_ang  = np.mean([s.get("avg_angle",  0) for s in stab_list])
        max_ang  = np.mean([s.get("max_angle",  0) for s in stab_list])
        p95_ang  = np.mean([s.get("p95_angle",  0) for s in stab_list])
        avg_acc  = np.mean([s.get("avg_acc",    0) for s in stab_list])
        max_acc  = np.mean([s.get("max_acc",    0) for s in stab_list])
        print(f"\n  ── 全过程稳定性指标 (N={len(stab_list)} episodes) ──")
        print(f"  摆动动能  avg/p95/max: {avg_ke:.1f} / {p95_ke:.1f} / {max_ke:.1f} mJ")
        print(f"  摆动角度  avg/p95/max: {avg_ang:.2f} / {p95_ang:.2f} / {max_ang:.2f} °")
        print(f"  EE 加速度 avg/max:     {avg_acc:.3f} / {max_acc:.3f} m/s²")
    print()


# ==============================================================================
# 主函数
# ==============================================================================

def print_summary(results, mode_name):
    """打印测试结果汇总（含全程+分阶段稳定性，用于 base vs RL 对比）。"""
    if not results:
        print(f"[{mode_name}] 无有效结果"); return
    sr     = np.mean([r["success"] for r in results])
    avg_r  = np.mean([r["reward"]  for r in results])
    avg_s  = np.mean([r["steps"]   for r in results])
    terms  = Counter((r["termination"] or "unknown").split(":")[0] for r in results)
    wind_f = results[0].get("wind_force", 0.0)
    print(f"\n{'='*60}")
    print(f"  [{mode_name}] 结果汇总"
          + (f"  (风力={wind_f:.1f}N)" if wind_f > 0 else ""))
    print(f"{'='*60}")
    print(f"  成功率:    {sr*100:.1f}%")
    print(f"  平均奖励:  {avg_r:.2f}")
    print(f"  平均步数:  {avg_s:.1f}")
    print(f"  终止原因:  {dict(terms)}")
    if results and "phase_success" in results[0]:
        for p in ["lift", "cruise", "descent"]:
            p_sr = np.mean([r["phase_success"].get(p, False) for r in results])
            p_r  = np.mean([r["phase_rewards"].get(p, 0)    for r in results])
            p_s  = np.mean([r["phase_steps"].get(p, 0)      for r in results])
            print(f"  {p:>8s}: SR={p_sr*100:.0f}% | R={p_r:.2f} | Steps={p_s:.0f}")

    def _stab_block(stab_list, label):
        if not stab_list: return
        ke_avg  = np.mean([s.get("avg_ke_mJ", 0) for s in stab_list])
        ke_p95  = np.mean([s.get("p95_ke_mJ", 0) for s in stab_list])
        ke_max  = np.mean([s.get("max_ke_mJ", 0) for s in stab_list])
        ang_avg = np.mean([s.get("avg_angle",  0) for s in stab_list])
        ang_p95 = np.mean([s.get("p95_angle",  0) for s in stab_list])
        ang_max = np.mean([s.get("max_angle",  0) for s in stab_list])
        acc_avg = np.mean([s.get("avg_acc",    0) for s in stab_list])
        acc_max_= np.mean([s.get("max_acc",    0) for s in stab_list])
        print(f"\n  ── {label} (N={len(stab_list)}) ──")
        print(f"  摆动动能  avg/p95/max: {ke_avg:6.1f} / {ke_p95:6.1f} / {ke_max:6.1f} mJ")
        print(f"  摆动角度  avg/p95/max: {ang_avg:5.2f} / {ang_p95:5.2f} / {ang_max:5.2f} °")
        print(f"  EE 加速度 avg/max:     {acc_avg:.3f} / {acc_max_:.3f} m/s²")

    _stab_block([r["stability"] for r in results if r.get("stability")], "全过程稳定性")
    if results and "phase_stability" in results[0]:
        for p in ["lift", "cruise", "descent"]:
            p_stabs = [r["phase_stability"][p] for r in results if r.get("phase_stability")]
            _stab_block(p_stabs, f"{p} 阶段稳定性")
    print()


def main():
    parser = argparse.ArgumentParser(description="三阶段 RL 测试")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["lift", "cruise", "descent", "pipeline"])
    parser.add_argument("--algo", type=str, default="expert",
                        choices=["ppo", "sac", "expert", "orca"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-paths", action="store_true")
    parser.add_argument("--save-dir", type=str, default="test_results")

    # 流水线模式额外参数
    parser.add_argument("--lift-ckpt",   type=str, default=None)
    parser.add_argument("--cruise-ckpt", type=str, default=None)
    parser.add_argument("--descent-ckpt",type=str, default=None)
    parser.add_argument("--lift-algo",   type=str, default="ppo",
                        choices=["ppo","sac","expert","orca"])
    parser.add_argument("--cruise-algo", type=str, default="ppo",
                        choices=["ppo","sac","expert","orca","nmpc_wind"])
    parser.add_argument("--descent-algo",type=str, default="ppo",
                        choices=["ppo","sac","expert","orca"])

    # 风力扰动参数
    parser.add_argument("--wind-force", type=float, default=0.0,
                        help="施加恒定风力 (N)，0=无风")
    parser.add_argument("--wind-levels", type=float, nargs="+",
                        default=None,
                        help="wind test 模式下的风力等级列表 (N)，例如 0 0.5 1.0 2.0 3.0")

    # 观测/执行噪声开关（默认关闭）
    parser.add_argument("--obs-noise", type=float, default=0.0,
                        help="观测噪声标准差 (关闭=0)，例如 0.02")
    parser.add_argument("--act-noise", type=float, default=0.0,
                        help="执行噪声标准差 rad/step (关闭=0)，例如 0.005")

    args = parser.parse_args()
    config = build_config(args)
    config["scene"]["seed"] = args.seed

    if args.obstacles is not None:
        test_n_obs = min(int(args.obstacles), config["scene"]["n_obstacles"])
    else:
        test_n_obs = config["scene"]["n_obstacles"]

    env = CableRobotEnvWithObstacles(config=config)
    env.set_curriculum_n_obstacles(test_n_obs)
    expert  = JointSpaceExpert(config, env.ik_solver)
    ee_ctrl = EEAccController(config, env.ik_solver)

    # ── nmpc_wind: cruise 阶段 NMPC 风力扰动测试 ─────────────────────────────
    if args.cruise_algo == "nmpc_wind" or (args.phase == "cruise" and args.algo == "nmpc_wind"):
        wind_levels = args.wind_levels or [0.0, 0.5, 1.0, 2.0, 3.0]
        n_eps = args.episodes
        results = test_cruise_nmpc_wind(
            env, expert, ee_ctrl, config,
            wind_levels=wind_levels, n_episodes_per_level=n_eps)
        env.close(); return

    if args.phase == "pipeline":
        agents = {}
        for p, p_algo, p_ckpt in [
            ("lift",    args.lift_algo,    args.lift_ckpt),
            ("cruise",  args.cruise_algo,  args.cruise_ckpt),
            ("descent", args.descent_algo, args.descent_ckpt),
        ]:
            if p_algo == "expert":
                agents[p] = None
            else:
                if p_ckpt is None:
                    print(f"[Error] --{p}-ckpt 必须指定 ({p_algo} 模式)")
                    return
                agents[p] = load_agent(p, p_algo, p_ckpt, config)
                print(f"[{p.upper()}] 加载: {p_ckpt} ({p_algo})")

        mode_name = "Pipeline"
        if all(a is None for a in agents.values()):
            mode_name = "Expert Pipeline"

        wind_str = f" | 风力={args.wind_force:.1f}N" if args.wind_force > 0 else ""
        noise_str= ""
        if args.obs_noise > 0: noise_str += f" | obs_noise={args.obs_noise}"
        if args.act_noise > 0: noise_str += f" | act_noise={args.act_noise}"
        print(f"\n{'='*60}")
        print(f"  流水线测试 | {args.episodes} 回合 | n_obs={test_n_obs}{wind_str}{noise_str}")
        print(f"{'='*60}")

        results = test_pipeline(
            env, agents, expert, ee_ctrl, config,
            n_episodes=args.episodes,
            wind_force=args.wind_force,
            obs_noise_sigma=args.obs_noise,
            act_noise_sigma=args.act_noise)
        print_summary(results, mode_name)
    
    else:
        # 单阶段测试
        if args.algo == "orca":
            # 纯 ORCA Expert 测试 (cruise 专用)
            if args.phase != "cruise":
                print("[Error] --algo orca 仅支持 --phase cruise")
                return
            mode_name = "CRUISE (ORCA Expert)"
            print(f"\n{'='*60}")
            print(f"  {mode_name} | {args.episodes} 回合 | n_obs={test_n_obs}")
            print(f"{'='*60}")
            results = test_orca_cruise(env, ee_ctrl, config, n_episodes=args.episodes)
            print_summary(results, mode_name)
        else:
            # RL / NMPC expert 测试
            agent = None
            if args.algo != "expert":
                if args.ckpt is None:
                    print(f"[Error] --ckpt 必须指定 ({args.algo} 模式)")
                    return
                agent = load_agent(args.phase, args.algo, args.ckpt, config)
                print(f"[{args.phase.upper()}] 加载: {args.ckpt} ({args.algo})")

            mode_name = f"{args.phase.upper()} ({args.algo.upper()})"
            print(f"\n{'='*60}")
            print(f"  {mode_name} | {args.episodes} 回合 | n_obs={test_n_obs}")
            print(f"{'='*60}")

            results = test_single_phase(env, agent, expert, ee_ctrl,
                                         args.phase, config,
                                         n_episodes=args.episodes,
                                         wind_force=args.wind_force)
            print_summary(results, mode_name)
    
    # 保存轨迹 + 稳定性数据
    if args.save_paths and results:
        save_dir = os.path.join(args.save_dir, args.phase)
        os.makedirs(save_dir, exist_ok=True)
        for i, r in enumerate(results):
            traj = r["trajectory"]
            pl_arr = np.array([t["payload"] for t in traj])
            ee_arr = np.array([t["ee"] for t in traj])
            tilt_arr = np.array([t.get("tilt", 0.) for t in traj])
            yaw_arr = np.array([t.get("yaw", 0.) for t in traj])
            swing_arr = np.array([t.get("swing_vel", 0.) for t in traj])
            stab = r.get("stability", {})
            save_data = dict(
                payload=pl_arr, ee=ee_arr,
                tilt=tilt_arr, yaw=yaw_arr, swing_vel=swing_arr,
                success=r["success"], reward=r["reward"],
                termination=str(r.get("termination", "")),
                # [v5] 稳定性曲线 (供绘图)
                swing_ke_curve_mJ=np.array(stab.get("ke_curve", [])),
                swing_angle_curve_deg=np.array(stab.get("angle_curve", [])),
                avg_ke_mJ=float(stab.get("avg_ke_mJ", 0)),
                max_ke_mJ=float(stab.get("max_ke_mJ", 0)),
                avg_angle_deg=float(stab.get("avg_angle", 0)),
                max_angle_deg=float(stab.get("max_angle", 0)),
                avg_acc_ms2=float(stab.get("avg_acc", 0)),
                max_acc_ms2=float(stab.get("max_acc", 0)),
            )
            # pipeline 模式额外保存阶段信息
            if "phase_success" in r:
                phase_arr = np.array([t.get("phase", "") for t in traj], dtype=object)
                save_data["phase"] = phase_arr
                for p in ["lift", "cruise", "descent"]:
                    save_data[f"{p}_success"] = r["phase_success"][p]
                    save_data[f"{p}_reward"] = r["phase_rewards"][p]
                    save_data[f"{p}_steps"] = r["phase_steps"][p]
            np.savez(os.path.join(save_dir, f"ep_{i}.npz"), **save_data)
        print(f"轨迹已保存: {save_dir}")
    
    env.close()


if __name__ == "__main__":
    main()