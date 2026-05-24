# ==============================================================================
# test_phase.py — 三阶段独立 + 流水线测试 v8 (重构版)
#
# 用法:
#   # 单阶段测试 (默认无噪声无风)
#   python test_phase.py --phase lift --algo ppo --ckpt saves/lift_ppo/ckpt_best.pt
#   python test_phase.py --phase cruise --algo ppo --ckpt saves/cruise_ppo/ckpt_best.pt
#   python test_phase.py --phase descent --algo ppo --ckpt saves/descent_ppo/ckpt_best.pt
#
#   # 专家基准 (无 RL)
#   python test_phase.py --phase lift --algo expert
#
#   # 噪声/风力扫描
#   python test_phase.py --phase cruise --algo ppo --ckpt ... \
#       --wind-force 1.5 --obs-noise 0.02 --act-noise 0.003 --force-noise 0.15
#
#   # 完整流水线
#   python test_phase.py --phase pipeline \
#       --lift-ckpt saves/lift_ppo/ckpt_best.pt \
#       --cruise-ckpt saves/cruise_ppo/ckpt_best.pt \
#       --descent-ckpt saves/descent_ppo/ckpt_best.pt
#
# v8 变更:
#   - 删除 ORCA expert, DescentDualRLAgent, CruiseDualRLAgent, test_cruise_nmpc_wind
#   - 所有噪声/风力参数 CLI 默认 0
#   - test 噪声/风力作用一致: 用 set_force_noise / set_wind_force + obs/act 高斯噪声
# ==============================================================================

import os
import sys
import copy
import argparse
import csv
import json
import time
import numpy as np
import torch
from collections import Counter

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from phase_agent import (
    PPOPhaseAgent, SACPhaseAgent,
    build_lift_obs, build_cruise_obs, build_descent_obs,
    build_wind_obs, CableEncoder,
)
from phase_reward import (
    compute_lift_reward, compute_cruise_reward, compute_descent_reward,
    LiftRewardState, CruiseRewardState, DescentRewardState,
)
from ee_acc_controller import EEAccController, CruiseZYawPID, SwingDampingController
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# 稳定性指标 (用于 base vs RL 对比)
# ==============================================================================

from stability_metrics import StabilityMetrics  # [v12.6] 共享模块, 同时供 train 用


# ==============================================================================
# 配置 / Agent 加载
# ==============================================================================

def build_config(args):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["sim"]["render"] = args.render
    config["train"]["gpu_id"] = args.gpu
    config.setdefault("test", {})["wait_for_space_start"] = bool(
        getattr(args, "wait_for_space", False))
    # Keep the trained descent z-gating in evaluation. Strict controller defaults
    # can lock z at high payload height when the learned residual keeps XY just
    # outside the old 15mm hard gate.
    if args.obstacles is not None:
        config["scene"]["n_obstacles"] = max(int(args.obstacles),
                                              config["scene"]["n_obstacles"])
    if getattr(args, "phase", None) == "descent":
        insert_target_z = float(getattr(args, "insert_target_z", 0.10))
        insert_depth_min = float(getattr(args, "insert_depth_min", 0.025))
        physical_target_z = float(getattr(args, "physical_insert_target_z", 0.090))
        config["insertion"]["target_payload_z"] = insert_target_z
        config["planning"]["target_z_descent"] = insert_target_z
        config["insertion"]["physical_insert_depth_min"] = insert_depth_min
        config["insertion"]["physical_insert_target_z"] = physical_target_z
        config["insertion"]["success_by_floor_contact"] = True
    if getattr(args, "phase", None) == "pipeline":
        # Pipeline test policy: lift + translation are pure NMPC. Descent keeps
        # the trained 100mm alignment target, then a short PID finishing move
        # continues downward only after alignment is already achieved.
        insert_target_z = float(getattr(args, "pipeline_insert_target_z", 0.10))
        insert_depth_min = float(getattr(args, "pipeline_insert_depth_min", 0.025))
        physical_target_z = float(getattr(
            args, "pipeline_physical_insert_target_z", 0.090))
        config["insertion"]["target_payload_z"] = insert_target_z
        config["planning"]["target_z_descent"] = insert_target_z
        config["insertion"]["physical_insert_depth_min"] = insert_depth_min
        config["insertion"]["physical_insert_target_z"] = physical_target_z
        config["insertion"]["success_by_floor_contact"] = True
        pipeline_cruise_z = float(getattr(args, "pipeline_cruise_z", 0.25))
        config["planning"]["payload_z_cruise"] = pipeline_cruise_z
        config["cruise_rl"]["z_lock_height"] = pipeline_cruise_z
        config["cruise_rl"]["target_z_cruise"] = pipeline_cruise_z
        config["planning"]["disable_descent_segment"] = True
        # Match the handoff state to the standalone descent reset:
        # payload starts near target XY at planning.payload_z_cruise.
        descent_xy_range = float(config.get("descent_rl", {}).get(
            "init_xy_range", 0.030))
        descent_z_range = float(config.get("descent_rl", {}).get(
            "init_z_range", 0.005))
        descent_handoff_xy = max(min(descent_xy_range, 0.020), 0.005)
        config["phase_transition"]["cruise_to_descent_xy_dist"] = min(
            float(config["phase_transition"].get(
                "cruise_to_descent_xy_dist", descent_handoff_xy)),
            descent_handoff_xy)
        config["phase_transition"]["cruise_to_descent_z_tol"] = max(
            0.020, descent_z_range * 4.0)
        config["phase_transition"]["cruise_to_descent_payload_vel_max"] = min(
            float(config["phase_transition"].get(
                "cruise_to_descent_payload_vel_max", 0.25)),
            0.12)
        config["phase_transition"]["cruise_to_descent_swing_vel_max"] = min(
            float(config["phase_transition"].get(
                "cruise_to_descent_swing_vel_max", 0.20)),
            0.12)
        config["phase_transition"]["cruise_to_descent_fallback_xy_dist"] = (
            config["phase_transition"]["cruise_to_descent_xy_dist"])
        config["phase_transition"]["cruise_to_descent_fallback_swing_max"] = (
            config["phase_transition"]["cruise_to_descent_swing_vel_max"])
        config["phase_transition"]["cruise_to_descent_min_z"] = max(
            0.0, pipeline_cruise_z - max(0.010, descent_z_range * 2.0))
        config["phase_transition"]["cruise_to_descent_safe_xy_dist"] = max(
            descent_handoff_xy * 1.25, 0.025)
        config["phase_transition"]["cruise_to_descent_safe_tilt_max"] = 0.20
        config["phase_transition"]["cruise_to_descent_safe_swing_vel_max"] = 0.30
        config["phase_transition"]["cruise_to_descent_safe_payload_vel_max"] = 0.25

        # Keep pipeline cruise NMPC identical to standalone
        # `--phase cruise --algo expert`. Only insertion/descent targets and
        # handoff gates are adjusted above; controller/ee_control/planning
        # tracking parameters must not be overridden here, otherwise the same
        # expert enters a different control regime in pipeline tests.

        cruise_steps = int(config.get("cruise_rl", {}).get("max_steps", 700))
        descent_steps = int(config.get("descent_rl", {}).get("max_steps", 300))
        pipeline_steps = cruise_steps + descent_steps + 50
        config.setdefault("pipeline", {})["max_steps"] = pipeline_steps
        config["sim"]["max_steps"] = max(int(config["sim"]["max_steps"]),
                                          pipeline_steps)
    return config


def install_space_start_callback(env, enabled):
    """Install a passive-viewer space-key callback for render-start gating."""
    if not enabled:
        return
    env._wait_start_key_pressed = False
    prev_cb = getattr(env, "_key_callback", None)

    def _key_callback(keycode):
        if prev_cb is not None:
            try:
                prev_cb(keycode)
            except Exception:
                pass
        if int(keycode) == 32:  # GLFW KEY_SPACE
            env._wait_start_key_pressed = True

    env._key_callback = _key_callback


def wait_for_space_start(env, config, label="episode"):
    """Pause rendered tests after initialization until Space is pressed."""
    if not bool(config.get("test", {}).get("wait_for_space_start", False)):
        return
    if not getattr(env, "render_mode", False) or getattr(env, "viewer", None) is None:
        return

    env._wait_start_key_pressed = False
    try:
        env.viewer.sync()
    except Exception:
        return

    print("\n" + "=" * 60)
    print(f"  {label} 初始化完成，点击 MuJoCo 渲染窗口后按 [Space] 开始执行")
    print("=" * 60)

    while not bool(getattr(env, "_wait_start_key_pressed", False)):
        viewer = getattr(env, "viewer", None)
        if viewer is None:
            break
        is_running = getattr(viewer, "is_running", None)
        if callable(is_running):
            try:
                if not is_running():
                    break
            except Exception:
                pass
        try:
            viewer.sync()
        except Exception:
            break
        time.sleep(0.03)

    env._wait_start_key_pressed = False
    print("  开始执行\n")


def load_agent(phase, algo, ckpt_path, config):
    """加载指定阶段和算法的 agent (v8: 只支持 PPO/SAC, 无 dual-RL)。"""
    if algo == "ppo":
        agent = PPOPhaseAgent(phase, config=config)
    elif algo == "sac":
        agent = SACPhaseAgent(phase, config=config)
    elif algo == "expert":
        return None
    else:
        raise ValueError(f"Unknown algo: {algo}")
    agent.load(ckpt_path)
    return agent


# ==============================================================================
# 噪声/风力应用辅助
# ==============================================================================

def apply_perturbations(env, wind_force, wind_dir, force_noise):
    """统一施加风力 + 噪声力 (在每 episode reset 之后调用)。"""
    if force_noise > 0 and hasattr(env, 'set_force_noise'):
        env.set_force_noise(force_noise)
    elif hasattr(env, 'set_force_noise'):
        env.set_force_noise(0.0)

    if wind_force > 0 and hasattr(env, 'set_wind_force'):
        _wd = float(wind_dir) if wind_dir is not None else float(np.random.uniform(0, 2*np.pi))
        env.set_wind_force(float(wind_force), _wd)
    else:
        if hasattr(env, 'clear_wind_force'):
            env.clear_wind_force()
        else:
            # 显式关闭风力
            if hasattr(env, '_test_wind_mode'):
                env._test_wind_mode = False
            if hasattr(env, 'data') and hasattr(env, 'prefab_body_id'):
                env.data.xfrc_applied[env.prefab_body_id, :3] = [0.0, 0.0, 0.0]


def _add_obs_noise(norm_obs, sigma):
    if sigma <= 0: return norm_obs
    return norm_obs + np.random.normal(0, sigma, norm_obs.shape).astype(np.float32)


def _add_act_noise(dq, sigma):
    if sigma <= 0: return dq
    return dq + np.random.normal(0, sigma, dq.shape).astype(dq.dtype)


def get_descent_eval_init(config, level="max"):
    levels = list(config.get("curriculum", {}).get("descent_levels", []))
    if not levels:
        dcfg = config.get("descent_rl", {})
        return {
            "xy_range": float(dcfg.get("init_xy_range", 0.030)),
            "vel_range": float(dcfg.get("init_vel_range", 0.030)),
            "tilt_range": float(dcfg.get("init_tilt_range", 0.010)),
            "xy_tol": float(config.get("insertion", {}).get(
                "xy_tolerance_train_end", 0.005)),
            "level_idx": 0,
            "wind_max": 0.0,
        }
    if level == "max":
        idx = len(levels) - 1
    else:
        idx = int(level)
        idx = max(0, min(idx, len(levels) - 1))
    lv = levels[idx]
    return {
        "xy_range": float(lv.get("init_xy", 0.030)),
        "vel_range": float(lv.get("init_vel", 0.030)),
        "tilt_range": float(lv.get("init_tilt", 0.010)),
        "xy_tol": float(lv.get("xy_tol", config.get("insertion", {}).get(
            "xy_tolerance_train_end", 0.005))),
        "level_idx": idx,
        "wind_max": float(lv.get("wind_max", 0.0)),
    }


def print_curriculum_hardest_task(config):
    task = get_descent_eval_init(config, "max")
    ins = config["insertion"]
    print("\n[Descent curriculum hardest task]")
    print(f"  wind_max: {task['wind_max']:.3f} N, sampled uniformly per episode")
    print(f"  init_xy: +/-{task['xy_range']*1000:.1f} mm")
    print(f"  init_payload_xy_vel: +/-{task['vel_range']:.3f} m/s")
    print(f"  init_tilt_noise: +/-{task['tilt_range']:.3f} rad")
    print(f"  success_xy_tol: {task['xy_tol']*1000:.1f} mm")
    print(f"  target_payload_z: {float(ins.get('target_payload_z', 0.10))*1000:.0f} mm")
    print(f"  z_tol/tilt_tol/yaw_tol: "
          f"{float(ins.get('success_z_tolerance', 0.020))*1000:.0f} mm / "
          f"{float(ins.get('tilt_tolerance', 0.05)):.3f} rad / "
          f"{float(ins.get('yaw_tolerance', 0.08)):.3f} rad")


def print_descent_eval_settings(config, eval_init):
    dcfg = config.get("descent_rl", {})
    ecfg = config.get("ee_control", {})
    print("\n[Descent train/test control settings]")
    print(f"  pid_residual_mode: {bool(dcfg.get('pid_residual_mode', True))}")
    print(f"  residual_dq_scale: {float(dcfg.get('residual_dq_scale', 0.0)):.3f}")
    print(f"  residual_acc_max_xy/z: "
          f"{float(dcfg.get('residual_acc_max_xy', 0.0)):.3f} / "
          f"{float(dcfg.get('residual_acc_max_z', 0.0)):.3f}")
    print(f"  base vel_max_z_descent: "
          f"{float(ecfg.get('vel_max_z_descent', 0.03)):.3f} m/s")
    print(f"  max_steps: {int(dcfg.get('max_steps', 300))}")
    print(f"  eval init: level={int(eval_init['level_idx'])}, "
          f"xy=+/-{float(eval_init['xy_range'])*1000:.1f}mm, "
          f"vel=+/-{float(eval_init['vel_range']):.3f}m/s, "
          f"tilt=+/-{float(eval_init['tilt_range']):.3f}rad, "
          f"xy_tol={float(eval_init['xy_tol'])*1000:.1f}mm")


# ==============================================================================
# 物理状态 + 阶段切换 (pipeline 用)
# ==============================================================================

def get_phase_state(env, obs, start_xy=None, config=None):
    """获取当前物理状态, 用于阶段切换判断。"""
    pl_xy     = np.array([obs[4], obs[5]])
    payload_z = float(env.data.body('prefab').xpos[2])
    pl_vxy    = obs[6:8]; ee_vxy = obs[2:4]
    swing_vel = float(np.linalg.norm(pl_vxy - ee_vxy))
    pl_vel    = float(np.linalg.norm(pl_vxy))

    pl_mat    = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler  = R.from_matrix(pl_mat).as_euler('xyz')
    tilt      = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))

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
        "pl_vz_abs":    abs(pl_vz),
        "swing_energy": swing_energy,
        "dtf_start":    dtf_start,
    }


def check_physical_insertion(env, config):
    """
    用真实物理容差验证插入成功。
    返回 (is_success: bool, detail: str)
    """
    cfg_ins  = config.get("insertion", {})
    cfg_pref = config.get("prefab",    {})
    cfg_tgt  = config.get("target",    {})
    target_pz = float(cfg_ins.get("target_payload_z", 0.10))

    socket_hole_size = cfg_pref.get("socket_hole_size", [0.014, 0.014])
    socket_hole_radius = min(socket_hole_size[0], socket_hole_size[1]) / 2.0
    rebar_radius = float(cfg_tgt.get("rebar_radius", 0.003))
    xy_tol = socket_hole_radius - rebar_radius

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
    insert_depth, hole_depth = _estimate_rebar_insertion_depth(env, config)
    min_insert_depth = float(cfg_ins.get(
        "physical_insert_depth_min", min(0.025, max(hole_depth, 0.0) * 0.5)))
    ok_insert = insert_depth >= min_insert_depth
    floor_contact = _check_payload_floor_contact(env, config)
    ok_contact_success = (
        bool(cfg_ins.get("success_by_floor_contact", False)) and
        floor_contact)

    detail = (f"z={payload_z*1000:.1f}mm(±{z_tol*1000:.0f}) "
              f"dtf={dtf*1000:.1f}mm(<{xy_tol*1000:.0f}) "
              f"tilt={tilt:.3f}(<{tilt_tol:.3f}) "
              f"yaw={abs_yaw:.3f}(<{yaw_tol:.3f}) "
              f"[{'✅' if ok_z else '❌'}z "
              f"{'✅' if ok_xy else '❌'}xy "
              f"{'✅' if ok_tilt else '❌'}tilt "
              f"{'✅' if ok_yaw else '❌'}yaw]")
    detail += (f" insert={insert_depth*1000:.1f}mm"
               f"(>={min_insert_depth*1000:.0f}) "
               f"[{'OK' if ok_insert else 'NO'} insert]")
    detail += f" floor={'OK' if floor_contact else 'NO'}"
    return (ok_z and ok_xy and ok_tilt and ok_yaw and
            (ok_insert or ok_contact_success)), detail


def _cruise_handoff_status(state, config, step=0, max_steps=500, env=None):
    pt = config["phase_transition"]
    if env is not None and hasattr(env, 'target_pos'):
        target_xy = np.array(env.target_pos[:2])
    else:
        target_xy = np.array(config["task"]["default_target_xy"])

    _rcfg_c = config["cruise_rl"]["reward"]
    z_lock = float(config["cruise_rl"].get("z_lock_height", 0.25))
    z_success_tol = float(pt.get("cruise_to_descent_z_tol", z_lock * 0.24))
    xy_strict = float(pt.get(
        "cruise_to_descent_xy_dist",
        _rcfg_c.get("success_radius", 0.12)))
    tilt_strict = float(pt.get(
        "cruise_to_descent_tilt_max",
        _rcfg_c.get("success_tilt_max", 0.20)))
    swing_strict = float(pt.get(
        "cruise_to_descent_swing_vel_max",
        _rcfg_c.get("success_swing_vel_max", 0.30)))
    vel_strict = float(pt.get(
        "cruise_to_descent_payload_vel_max",
        _rcfg_c.get("success_payload_vel_max", 0.35)))

    payload_z = float(state.get("payload_z", state.get("pl_z", 0.0)))
    dtf = float(np.linalg.norm(state["pl_xy"] - target_xy))
    tilt = float(state["tilt"])
    swing_vel = float(state["swing_vel"])
    pl_vel = float(state["pl_vel"])

    strict_ok = (
        dtf < xy_strict and
        abs(payload_z - z_lock) < z_success_tol and
        tilt < tilt_strict and
        swing_vel < swing_strict and
        pl_vel < vel_strict)
    if strict_ok:
        detail = (f"strict dtf={dtf*1000:.1f}mm z={payload_z*1000:.1f}mm "
                  f"tilt={tilt:.3f} swing={swing_vel:.3f} plV={pl_vel:.3f}")
        return True, "strict", detail

    safe_min_z = float(pt.get("cruise_to_descent_min_z",
                              max(0.20, z_lock - 0.06)))
    safe_xy = float(pt.get(
        "cruise_to_descent_safe_xy_dist",
        max(xy_strict * 2.0, 0.08)))
    safe_tilt = float(pt.get(
        "cruise_to_descent_safe_tilt_max",
        max(tilt_strict, 0.35)))
    safe_swing = float(pt.get(
        "cruise_to_descent_safe_swing_vel_max",
        max(swing_strict, 0.45)))
    safe_vel = float(pt.get(
        "cruise_to_descent_safe_payload_vel_max",
        max(vel_strict, 0.50)))
    safe_ok = (
        dtf < safe_xy and
        payload_z >= safe_min_z and
        tilt < safe_tilt and
        swing_vel < safe_swing and
        pl_vel < safe_vel)
    if safe_ok:
        detail = (f"safe dtf={dtf*1000:.1f}mm(<{safe_xy*1000:.0f}) "
                  f"z={payload_z*1000:.1f}mm(>={safe_min_z*1000:.0f}) "
                  f"tilt={tilt:.3f} swing={swing_vel:.3f} plV={pl_vel:.3f}")
        return True, "safe", detail

    detail = (f"wait dtf={dtf*1000:.1f}mm strict<{xy_strict*1000:.0f}/"
              f"safe<{safe_xy*1000:.0f}, z={payload_z*1000:.1f}mm "
              f"target={z_lock*1000:.0f}, min_z={safe_min_z*1000:.0f}, "
              f"tilt={tilt:.3f}, swing={swing_vel:.3f}, plV={pl_vel:.3f}")
    return False, "wait", detail


def _estimate_rebar_insertion_depth(env, config):
    cfg_pref = config.get("prefab", {})
    cfg_tgt = config.get("target", {})

    socket_half_size = cfg_pref.get("socket_half_size", [0.05, 0.05, 0.10])
    socket_half_z = float(socket_half_size[2]) if len(socket_half_size) >= 3 else 0.10
    hole_depth = float(cfg_pref.get("socket_hole_depth", 0.06))
    rebar_half_h = float(cfg_tgt.get("rebar_half_height", 0.01))

    payload_z = float(env.data.body('prefab').xpos[2])
    try:
        target_base_z = float(env.data.body('target').xpos[2])
    except Exception:
        target_base_z = 0.0

    rebar_top_z = target_base_z + 2.0 * rebar_half_h
    socket_bottom_z = payload_z - socket_half_z
    raw_depth = rebar_top_z - socket_bottom_z
    return float(np.clip(raw_depth, 0.0, max(hole_depth, 0.0))), hole_depth


def _check_payload_floor_contact(env, config):
    if hasattr(env, "_check_prefab_floor_contact"):
        try:
            if bool(env._check_prefab_floor_contact()):
                return True
        except Exception:
            pass

    try:
        cfg_pref = config.get("prefab", {})
        socket_half_size = cfg_pref.get("socket_half_size", [0.05, 0.05, 0.10])
        socket_half_z = float(socket_half_size[2]) if len(socket_half_size) >= 3 else 0.10
        z_tol = float(config.get("insertion", {}).get(
            "floor_contact_z_tolerance", 0.008))
        payload_z = float(env.data.body('prefab').xpos[2])
        return payload_z <= socket_half_z + z_tol
    except Exception:
        return False


def _get_physical_insert_target_z(config):
    cfg_ins = config.get("insertion", {})
    if "physical_insert_target_z" in cfg_ins:
        return float(cfg_ins["physical_insert_target_z"])

    cfg_pref = config.get("prefab", {})
    cfg_tgt = config.get("target", {})
    socket_half_size = cfg_pref.get("socket_half_size", [0.05, 0.05, 0.10])
    socket_half_z = float(socket_half_size[2]) if len(socket_half_size) >= 3 else 0.10
    rebar_half_h = float(cfg_tgt.get("rebar_half_height", 0.01))
    min_depth = float(cfg_ins.get("physical_insert_depth_min", 0.025))
    # target body is placed at z=0 in generate_scene_and_trajectory.
    return max(0.02, 2.0 * rebar_half_h + socket_half_z - min_depth - 0.005)


def _activate_physical_insert_finish(expert, config):
    if expert is None:
        return None
    final_z = _get_physical_insert_target_z(config)
    expert._target_payload_z = float(final_z)
    expert._z_reached_thresh = float(final_z)
    expert._z_reached = False
    expert._final_hold_counter = 0
    expert._contact_detected = False
    if hasattr(expert, "tracker"):
        expert.tracker._is_descending = True
    return final_z


def check_phase_transition(phase, state, config, step=0, max_steps=500, env=None):
    """检查阶段切换条件 (pipeline 用)。"""
    pt = config["phase_transition"]
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
        z_tol_test = max(z_tol, 0.030)
        return (
            abs(state["payload_z"] - _z_cruise)    < z_tol_test           and
            state.get("pl_vz_abs",    999.0)        < vz_max                  and
            state.get("swing_energy", 999.0)        < e_thresh * 3.0          and
            state["tilt"]                           < float(pt["lift_to_cruise_tilt_max"]) and
            state.get("dtf_start",    999.0)        < xy_max * 1.5
        )

    elif phase == "cruise":
        ok, _, _ = _cruise_handoff_status(
            state, config, step=step, max_steps=max_steps, env=env)
        return ok
        # [v13.1] train (compute_cruise_reward) 和 test_pipeline (此函数) 使用一致的判定:
        # 读相同 config keys (cruise_rl.reward 中的 success_*).
        # 用户反馈 "确保 train 和 test 的标准一致".
        # For pipeline handoff, prefer the stricter phase_transition gates so
        # PID only takes over once the payload is actually above the target.
        _rcfg_c  = config["cruise_rl"]["reward"]
        z_lock   = float(config["cruise_rl"].get("z_lock_height", 0.25))
        z_success_tol = float(pt.get("cruise_to_descent_z_tol",
                                     z_lock * 0.24))
        success_radius = float(pt.get(
            "cruise_to_descent_xy_dist",
            _rcfg_c.get("success_radius", 0.12)))
        # success_* keys 比 cruise_to_descent_* 宽松 (按 v13.1 设计):
        tilt_max   = float(pt.get(
            "cruise_to_descent_tilt_max",
            _rcfg_c.get("success_tilt_max", 0.20)))
        swing_max  = float(pt.get(
            "cruise_to_descent_swing_vel_max",
            _rcfg_c.get("success_swing_vel_max", 0.30)))
        vel_max    = float(pt.get(
            "cruise_to_descent_payload_vel_max",
            _rcfg_c.get("success_payload_vel_max", 0.35)))

        payload_z = float(state.get("payload_z", state.get("pl_z", 0.0)))
        dtf = float(np.linalg.norm(state["pl_xy"] - target_xy))
        normal_ok = (
            dtf < success_radius and
            abs(payload_z - z_lock) < z_success_tol and
            state["tilt"]      < tilt_max and
            state["swing_vel"] < swing_max and
            state["pl_vel"]    < vel_max)
        if normal_ok: return True

        # Fallback: 超时前最后 15% step 内, 标准更宽松 (避免 pipeline 卡死)
        # 旧的 cruise_to_descent_fallback_* 保留, 但不再用 cruise_to_descent_xy_dist
        fallback_frac  = float(pt.get("cruise_to_descent_fallback_step_frac", 0.85))
        fallback_xy    = float(pt.get("cruise_to_descent_fallback_xy_dist",   0.15))
        fallback_swing = float(pt.get("cruise_to_descent_fallback_swing_max", 0.30))
        if (step >= int(max_steps * fallback_frac) and
                dtf < fallback_xy and
                abs(payload_z - z_lock) < z_success_tol and
                state["swing_vel"] < fallback_swing and
                state["pl_vel"] < vel_max and
                state["tilt"] < tilt_max):
            return True
        return False
    return False


def _advance_expert_to_nearest_wp(expert, planned_path, pl_pos):
    if planned_path is None or len(planned_path) == 0:
        return
    dists = [np.linalg.norm(pl_pos - wp) for wp in planned_path]
    nearest_idx = int(np.argmin(dists))
    expert.tracker.current_idx = nearest_idx


def _truncate_path_for_lift(planned_path, config):
    """[v12.2] 为 lift phase 截短 path, 只保留垂直上升段.
    与 train_phase.py 中同名函数行为一致.
    """
    if planned_path is None or len(planned_path) == 0:
        return planned_path
    pp = np.asarray(planned_path, dtype=np.float64)
    z_cruise = float(config.get("planning", {}).get("payload_z_cruise", 0.25))
    keep_idx = []
    reached_cruise_z = False
    for i, wp in enumerate(pp):
        wp_z = float(wp[2]) if len(wp) >= 3 else 0.3
        if wp_z < z_cruise - 0.001:
            keep_idx.append(i)
        elif not reached_cruise_z:
            keep_idx.append(i)
            reached_cruise_z = True
        else:
            break
    if not keep_idx:
        return pp[:1].copy()
    return pp[keep_idx].copy()


def _truncate_path_for_pipeline_cruise(planned_path, config):
    """Keep lift + cruise waypoints only; remove descent waypoints."""
    if planned_path is None or len(planned_path) == 0:
        return planned_path
    pp = np.asarray(planned_path, dtype=np.float64)
    z_cruise = float(config.get("planning", {}).get("payload_z_cruise", 0.25))
    keep_idx = []
    reached_cruise_z = False
    for i, wp in enumerate(pp):
        wp_z = float(wp[2]) if len(wp) >= 3 else z_cruise
        if reached_cruise_z and wp_z < z_cruise - 0.01:
            break
        keep_idx.append(i)
        if wp_z >= z_cruise - 0.005:
            reached_cruise_z = True
    if not keep_idx:
        return pp[:1].copy()
    return pp[keep_idx].copy()


# ==============================================================================
# 单阶段测试
# ==============================================================================

def test_single_phase(env, agent, expert, ee_ctrl, phase, config,
                      n_episodes=20, deterministic=True,
                      wind_force=0.0, wind_dir=None,
                      obs_noise=0.0, act_noise=0.0, force_noise=0.0,
                      eval_cur_init=None, wind_force_sampler=None,
                      wind_dir_sampler=None, reset_seed_sampler=None,
                      verbose=True, progress_every=0, progress_prefix=""):
    """测试单个阶段, 支持噪声/风力扰动。"""
    from train_phase import (reset_for_phase, build_phase_obs,
                             REWARD_FNS, REWARD_STATES,
                             _apply_descent_pid_residual,
                             clip_cruise_residual, get_last_nmpc_action)

    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    _cruise_nmpc_base = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get("use_nmpc_base", True)))
    _cruise_nmpc_residual = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get(
            "nmpc_residual_mode", _cruise_nmpc_base)))

    results = []
    ep_count = 0; attempt = 0

    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        reset_kw = {}
        if phase == "descent" and eval_cur_init is not None:
            reset_kw = {
                "override_init_xy_range": eval_cur_init["xy_range"],
                "override_init_vel_range": eval_cur_init["vel_range"],
                "override_init_tilt_range": eval_cur_init["tilt_range"],
            }
        if reset_seed_sampler is not None:
            reset_kw["rng_seed"] = int(reset_seed_sampler(ep_count))
        obs, planned_path = reset_for_phase(env, phase, config, **reset_kw)
        if obs is None:
            continue

        # ── 施加扰动 ──────────────────────────────────────────────────────────
        ep_wind_force = (float(wind_force_sampler(ep_count))
                         if wind_force_sampler is not None else float(wind_force))
        ep_wind_dir = (float(wind_dir_sampler(ep_count))
                       if wind_dir_sampler is not None else wind_dir)
        apply_perturbations(env, ep_wind_force, ep_wind_dir, force_noise)

        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        if planned_path is not None:
            # [v12.2] lift 时只把"lift 段"喂给 tracker (test_phase 同步修复)
            _pp_for_expert = _truncate_path_for_lift(planned_path, config) \
                if phase == "lift" else planned_path
            expert.set_path(_pp_for_expert)
            if phase in ("cruise", "descent"):
                pl_pos = env.data.body('prefab').xpos.copy()
                _advance_expert_to_nearest_wp(expert, planned_path, pl_pos)
        ee_ctrl.reset(env._get_ee_pos(), current_q)
        if z_pid is not None:
            _pl_z   = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)
        if agent is not None and hasattr(agent, 'reset_history'):
            agent.reset_history()
        wait_for_space_start(env, config, label=f"{phase} Ep {ep_count + 1}")

        start_xy = env.default_start_xy.copy()
        target_xy = env.target_pos.copy()
        prev_tilt, prev_yaw = 0.0, 0.0
        rstate = REWARD_STATES[phase]()
        if hasattr(rstate, 'total_steps_global'):
            rstate.total_steps_global = 10_000_000
        if phase == "lift" and hasattr(rstate, 'start_xy'):
            rstate.start_xy = env.data.body('prefab').xpos[:2].copy()
        if phase == "descent":
            # 测试用严格判定: xy_tol = config 中的 train_end 值
            if eval_cur_init is not None:
                rstate.current_xy_tol = float(eval_cur_init["xy_tol"])
                rstate.current_descent_level = int(eval_cur_init["level_idx"])
                rstate.descent_n_levels = max(1, len(config.get(
                    "curriculum", {}).get("descent_levels", [])))
            else:
                rstate.current_xy_tol = float(config["insertion"].get(
                    "xy_tolerance_train_end", 0.005))
                rstate.current_descent_level = 99
                rstate.descent_n_levels = 1

        ep_reward = 0.0; ep_steps = 0; ep_success = False
        term_reason = None
        physical_finish_active = False
        stab_metrics = StabilityMetrics()

        max_steps = int(config[f"{phase}_rl"]["max_steps"])

        for step in range(max_steps):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()
            _cruise_reward_base = None
            _cruise_reward_action = None

            if agent is None:
                delta_q = expert.compute_delta_q_target(obs, current_q)
            else:
                # [v14.0] 构建 obs: (core, cable_raw, wind, tilt, yaw)
                _wobs = build_wind_obs(env,
                    float(config.get("wind_obs", {}).get("wind_force_max", 2.0)))
                base_dq_for_obs = None
                if phase == "descent" and bool(config.get("descent_rl", {}).get("pid_residual_mode", True)):
                    try:
                        base_dq_for_obs = expert.compute_delta_q_target(
                            obs, current_q.astype(np.float64))
                    except Exception:
                        base_dq_for_obs = np.zeros(7, dtype=np.float32)
                elif phase == "cruise" and _cruise_nmpc_residual:
                    base_dq_for_obs = get_last_nmpc_action(expert)
                core, cable_raw, _wobs, prev_tilt, prev_yaw = build_phase_obs(
                    phase, obs, env, start_xy, target_xy, prev_tilt, prev_yaw,
                    wind_obs=_wobs, base_action=base_dq_for_obs)
                p_obs = agent.encode_obs(core, cable_raw, _wobs)
                norm_obs = agent.normalize_obs(p_obs, update=False)
                norm_obs = _add_obs_noise(norm_obs, obs_noise)

                result = agent.act(norm_obs, deterministic=deterministic)
                if isinstance(result, tuple):
                    action = result[0]
                else:
                    action = result

                real_ee = env._get_ee_pos()
                if phase == "cruise" and _cruise_nmpc_residual:
                    _res3 = clip_cruise_residual(action, config)
                    delta_q = expert.compute_delta_q_target(
                        obs, current_q.astype(np.float64),
                        residual_acc=_res3)
                    _cruise_reward_base = get_last_nmpc_action(expert)
                    _cruise_reward_action = _res3
                elif phase == "cruise" and z_pid is not None:
                    _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                    _pl_vz = float(env.data.qvel[_dof_idx + 2])
                    _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                    _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                    _pl_yaw = float(_pl_euler[2])
                    _pl_yr = float(env.data.qvel[_dof_idx + 5]) \
                        if _dof_idx + 5 < len(env.data.qvel) else 0.0
                    _z_corr, _tgt_yaw, _falling = z_pid.compute(
                        float(payload_pos[2]), _pl_vz, _pl_yaw, _pl_yr)
                    if _falling:
                        term_reason = f"ground_collision:z={payload_pos[2]:.3f}"
                        break
                    if swing_d is not None:
                        swing_d.compute(payload_pos, real_ee,
                            env.data.qvel[_dof_idx:_dof_idx+3].copy(),
                            getattr(env, '_ee_vel_cache', np.zeros(3)))
                    if _cruise_nmpc_base:
                        try:
                            _a4 = expert.tracker.compute_ee_acceleration(obs,
                                target_yaw=_tgt_yaw)
                            _ba = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                        except Exception:
                            _ba = np.zeros(2, np.float32)
                        _cruise_reward_base = np.array(
                            [_ba[0], _ba[1], 0.0, 0.0], dtype=np.float32)
                        _rm = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.25))
                        _cm = _ba + np.clip(action[:2], -_rm, _rm)
                        _am = float(config["cruise_rl"].get("residual_acc_max_xy", 0.80))
                        _cn = float(np.linalg.norm(_cm))
                        if _cn > _am: _cm = _cm / _cn * _am
                        acc_3d = np.array([_cm[0], _cm[1], 0.0])
                    else:
                        acc_3d = np.array([action[0], action[1], 0.0])
                    z_lock = float(config["cruise_rl"]["z_lock_height"])
                    delta_q = ee_ctrl.compute_delta_q(
                        acc_3d, current_q, real_ee,
                        lock_z=True, z_lock_height=z_lock,
                        z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                        base_acc_xy=None, residual_mode=False)
                elif phase == "descent":
                    if bool(config.get("descent_rl", {}).get("pid_residual_mode", True)):
                        delta_q, _ = _apply_descent_pid_residual(
                            expert, action, obs, env, config, current_q,
                            pid_dq=base_dq_for_obs)
                    else:
                        _vmax_z_d = float(config.get("ee_control", {}).get(
                            "vel_max_z_descent", 0.03))
                        delta_q = ee_ctrl.compute_delta_q(
                            action, current_q, real_ee, vel_max_z=_vmax_z_d)
                else:  # lift
                    delta_q = ee_ctrl.compute_delta_q(action, current_q, real_ee)

            delta_q = _add_act_noise(delta_q, act_noise)
            next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
            # [v12.6] 传 env (用于 cable_ke) + rl_action (用于 rl_action_mag)
            stab_metrics.update_step(next_obs, config, env=env,
                                     rl_action=(action if agent is not None else None))

            setattr(rstate, "phase_step", int(ep_steps))
            if phase == "cruise":
                reward, r_done, r_success, r_info = compute_cruise_reward(
                    env, next_obs, config, rstate,
                    rl_action=(_cruise_reward_action
                               if _cruise_reward_action is not None
                               else (action if agent is not None else None)),
                    base_action=_cruise_reward_base)
            else:
                reward, r_done, r_success, r_info = REWARD_FNS[phase](
                    env, next_obs, config, rstate,
                    rl_action=(action if agent is not None else None))

            ep_reward += reward; ep_steps += 1

            if r_success:
                if phase == "descent":
                    phys_ok, phys_detail = check_physical_insertion(env, config)
                    ep_success = phys_ok
                    mark_str = "✅ 物理插入成功" if phys_ok else "⚠ 课程容差达标但未插入"
                    if verbose and (phys_ok or not physical_finish_active):
                        print(f"    [{mark_str}] {phys_detail}")
                else:
                    ep_success = True
                if phase == "descent" and not ep_success:
                    if not physical_finish_active:
                        final_z = _activate_physical_insert_finish(expert, config)
                        if verbose and final_z is not None:
                            print(f"    [physical insert finish] PID target_z={final_z*1000:.1f}mm")
                        physical_finish_active = True
                    r_done = False
                    r_success = False
                    r_info = dict(r_info)
                    r_info.pop("termination", None)
                    term_reason = "reward_success_waiting_physical_insert"
                else:
                    term_reason = r_info.get("termination", "success")
                    obs = next_obs; break

            if phase == "descent" and physical_finish_active and not ep_success:
                phys_ok, phys_detail = check_physical_insertion(env, config)
                if phys_ok:
                    ep_success = True
                    term_reason = "physical_insertion_success"
                    if verbose:
                        print(f"    [✅ 物理插入成功] {phys_detail}")
                    obs = next_obs; break

            if r_info.get("termination"):
                term_reason = r_info["termination"]
            obs = next_obs
            if r_done or env_info.get("nan_detected", False):
                break

        ep_count += 1
        results.append({
            "reward":      ep_reward,
            "steps":       ep_steps,
            "success":     ep_success,
            "termination": term_reason or "timeout",
            "stability":   stab_metrics.summary(),
            "wind_force":  ep_wind_force,
            "wind_dir":    ep_wind_dir,
        })
        if verbose:
            mark = "✅" if ep_success else "❌"
            term_short = (term_reason or "timeout").split(":")[0]
            print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | "
                  f"Steps:{ep_steps:3d} | W:{ep_wind_force:.3f}N | "
                  f"{term_short} | {stab_metrics.print_line()}")
        elif progress_every and (ep_count % progress_every == 0 or
                                 ep_count == n_episodes):
            sr_now = float(np.mean([r["success"] for r in results])) if results else 0.0
            avg_steps_now = float(np.mean([r["steps"] for r in results])) if results else 0.0
            print(f"{progress_prefix} progress {ep_count}/{n_episodes} "
                  f"SR={sr_now*100:.1f}% avg_steps={avg_steps_now:.1f}",
                  flush=True)

    return results


# ==============================================================================
# 流水线测试 (3 阶段串联)
# ==============================================================================

def test_pipeline(env, agents, expert, ee_ctrl, config,
                  n_episodes=20, deterministic=True,
                  wind_force=0.0, wind_dir=None,
                  obs_noise=0.0, act_noise=0.0, force_noise=0.0):
    """测试完整 3 阶段流水线。"""
    from train_phase import (reset_for_phase, build_phase_obs,
                             REWARD_FNS, REWARD_STATES,
                             _apply_descent_pid_residual,
                             clip_cruise_residual, get_last_nmpc_action)

    z_pid   = CruiseZYawPID(config)
    swing_d = SwingDampingController(config)
    _cruise_nmpc_base = bool(config.get("cruise_rl", {}).get("use_nmpc_base", True))
    _cruise_nmpc_residual = bool(config.get("cruise_rl", {}).get(
        "nmpc_residual_mode", _cruise_nmpc_base))
    _cruise_max = int(config["cruise_rl"]["max_steps"])
    _descent_max = int(config.get("descent_rl", {}).get("max_steps", 300))
    _pipeline_max = int(config.get("pipeline", {}).get(
        "max_steps", _cruise_max + _descent_max + 50))

    results = []
    ep_count = 0; attempt = 0

    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        sys_stdout_saved = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs, planned_path = reset_for_phase(env, "cruise", config)
        finally:
            sys.stdout.close(); sys.stdout = sys_stdout_saved
        if obs is None: continue

        apply_perturbations(env, wind_force, wind_dir, force_noise)

        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        if planned_path is not None:
            # [v13.0] 完整 path (lift+cruise+descent), cruise 段 NMPC 自处理 lift→cruise 边界
            cruise_path = _truncate_path_for_pipeline_cruise(planned_path, config)
            expert.set_path(cruise_path)
            pl_pos_init = env.data.body('prefab').xpos.copy()
            _advance_expert_to_nearest_wp(expert, cruise_path, pl_pos_init)
        ee_ctrl.reset(env._get_ee_pos(), current_q)
        _pl_z   = float(env.data.body('prefab').xpos[2])
        _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
        _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
        z_pid.reset(_pl_z, _pl_yaw)
        for _agent in agents.values():
            if _agent is not None and hasattr(_agent, 'reset_history'):
                _agent.reset_history()
        wait_for_space_start(env, config, label=f"pipeline Ep {ep_count + 1}")

        start_xy  = env.default_start_xy.copy()
        target_xy = env.target_pos.copy()
        prev_tilt, prev_yaw = 0.0, 0.0

        # [v13.0] 2 阶段: cruise (合并 lift) + descent
        current_phase = "cruise"
        phase_rewards = {"cruise": 0.0, "descent": 0.0}
        phase_steps   = {"cruise": 0,   "descent": 0}
        phase_success = {"cruise": False, "descent": False}

        rstate = REWARD_STATES["cruise"]()
        if hasattr(rstate, 'total_steps_global'):
            rstate.total_steps_global = 10_000_000

        ep_reward = 0.0; ep_steps = 0
        final_success = False; term_reason = None
        physical_finish_active = False
        stab_all = StabilityMetrics()
        phase_stab = {p: StabilityMetrics() for p in ["cruise", "descent"]}

        for step in range(_pipeline_max):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()

            state = get_phase_state(env, obs, start_xy=start_xy, config=config)

            # ── 阶段切换 [v13.0] 只有 cruise→descent ─────────────────────────
            _handoff_ok = False
            _handoff_mode = "wait"
            _handoff_detail = ""
            if current_phase == "cruise":
                _handoff_ok, _handoff_mode, _handoff_detail = (
                    _cruise_handoff_status(
                        state, config, step=phase_steps["cruise"],
                        max_steps=_cruise_max, env=env))
            if current_phase == "cruise" and _handoff_ok:
                phase_success["cruise"] = True
                current_phase = "descent"
                rstate = REWARD_STATES["descent"]()
                if hasattr(rstate, 'total_steps_global'):
                    rstate.total_steps_global = 10_000_000
                rstate.current_xy_tol = float(config["insertion"].get(
                    "xy_tolerance_train_end", 0.005))
                rstate.current_descent_level = 99; rstate.descent_n_levels = 1
                expert.reset(obs, current_q, env=env)
                if planned_path is not None:
                    expert.set_path(planned_path)
                    _advance_expert_to_nearest_wp(expert, planned_path, payload_pos)
                expert.tracker._is_descending = True
                ee_ctrl.reset(env._get_ee_pos(), current_q)
                prev_tilt, prev_yaw = 0.0, 0.0
                desc_agent = agents.get("descent")
                if desc_agent is not None and hasattr(desc_agent, 'reset_history'):
                    desc_agent.reset_history()
                _dtf_sw = np.linalg.norm(state["pl_xy"] - env.target_pos)
                _target_pz = float(config["insertion"].get(
                    "target_payload_z", 0.10))
                _handoff_xy = float(config.get("phase_transition", {}).get(
                    "cruise_to_descent_safe_xy_dist", 0.020))
                _z_cruise = float(config.get("planning", {}).get(
                    "payload_z_cruise", 0.25))
                print(f"    [pipeline s{step}] NMPC->PID+RL "
                      f"({_handoff_mode}) {_handoff_detail} "
                      f"dtf={_dtf_sw*1000:.1f}mm "
                      f"(<= {_handoff_xy*1000:.0f}) "
                      f"z={state['payload_z']*1000:.1f}mm "
                      f"(target {_z_cruise*1000:.0f}) "
                      f"insert_z={_target_pz*1000:.1f}mm")

            # ── 动作计算 ──────────────────────────────────────────────────────
            # Cruise can optionally use NMPC + residual RL; descent keeps the
            # PID + residual RL path.
            agent = agents.get(current_phase)
            action = None
            _cruise_reward_base = None
            _cruise_reward_action = None
            if agent is None:
                delta_q = expert.compute_delta_q_target(obs, current_q)
            else:
                # [v14.0] 构建 obs
                _wobs = build_wind_obs(env,
                    float(config.get("wind_obs", {}).get("wind_force_max", 2.0)))
                base_dq_for_obs = None
                if current_phase == "cruise" and _cruise_nmpc_residual:
                    base_dq_for_obs = get_last_nmpc_action(expert)
                elif current_phase == "descent" and bool(config.get("descent_rl", {}).get("pid_residual_mode", True)):
                    try:
                        base_dq_for_obs = expert.compute_delta_q_target(
                            obs, current_q.astype(np.float64))
                    except Exception:
                        base_dq_for_obs = np.zeros(7, dtype=np.float32)
                core, cable_raw, _wobs, prev_tilt, prev_yaw = build_phase_obs(
                    current_phase, obs, env, start_xy, target_xy, prev_tilt, prev_yaw,
                    wind_obs=_wobs, base_action=base_dq_for_obs)
                p_obs = agent.encode_obs(core, cable_raw, _wobs)
                norm_obs = agent.normalize_obs(p_obs, update=False)
                norm_obs = _add_obs_noise(norm_obs, obs_noise)

                result = agent.act(norm_obs, deterministic=deterministic)
                action = result[0] if isinstance(result, tuple) else result
                real_ee = env._get_ee_pos()

                if current_phase == "cruise":
                    if _cruise_nmpc_residual:
                        _res3 = clip_cruise_residual(action, config)
                        delta_q = expert.compute_delta_q_target(
                            obs, current_q.astype(np.float64),
                            residual_acc=_res3)
                        _cruise_reward_base = get_last_nmpc_action(expert)
                        _cruise_reward_action = _res3
                    else:
                        _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                        _pl_vz = float(env.data.qvel[_dof_idx + 2])
                        _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                        _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                        _pl_yaw = float(_pl_euler[2])
                        _pl_yr = float(env.data.qvel[_dof_idx + 5]) \
                            if _dof_idx + 5 < len(env.data.qvel) else 0.0
                        _z_corr, _tgt_yaw, _falling = z_pid.compute(
                            float(payload_pos[2]), _pl_vz, _pl_yaw, _pl_yr)
                        if _falling:
                            term_reason = f"ground_collision:z={payload_pos[2]:.3f}"
                            break
                        if swing_d is not None:
                            swing_d.compute(payload_pos, real_ee,
                                env.data.qvel[_dof_idx:_dof_idx+3].copy(),
                                getattr(env, '_ee_vel_cache', np.zeros(3)))
                        if _cruise_nmpc_base:
                            try:
                                _a4 = expert.tracker.compute_ee_acceleration(obs,
                                    target_yaw=_tgt_yaw)
                                _ba = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                            except Exception:
                                _ba = np.zeros(2, np.float32)
                            _cruise_reward_base = np.array(
                                [_ba[0], _ba[1], 0.0, 0.0], dtype=np.float32)
                            _rm = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.25))
                            _cm = _ba + np.clip(action[:2], -_rm, _rm)
                            _am = float(config["cruise_rl"].get("residual_acc_max_xy", 0.80))
                            _cn = float(np.linalg.norm(_cm))
                            if _cn > _am: _cm = _cm / _cn * _am
                            acc_3d = np.array([_cm[0], _cm[1], 0.0])
                        else:
                            acc_3d = np.array([action[0], action[1], 0.0])
                        z_lock = float(config["cruise_rl"]["z_lock_height"])
                        delta_q = ee_ctrl.compute_delta_q(
                            acc_3d, current_q, real_ee,
                            lock_z=True, z_lock_height=z_lock,
                            z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                            base_acc_xy=None, residual_mode=False)
                elif current_phase == "descent":
                    if bool(config.get("descent_rl", {}).get("pid_residual_mode", True)):
                        delta_q, _ = _apply_descent_pid_residual(
                            expert, action, obs, env, config, current_q,
                            pid_dq=base_dq_for_obs)
                    else:
                        _vmax_z_d = float(config.get("ee_control", {}).get(
                            "vel_max_z_descent", 0.03))
                        delta_q = ee_ctrl.compute_delta_q(
                            action, current_q, real_ee, vel_max_z=_vmax_z_d)
                else:
                    delta_q = ee_ctrl.compute_delta_q(action, current_q, real_ee)

            delta_q = _add_act_noise(delta_q, act_noise)
            next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
            # [v12.6] 传 env (用于 cable_ke) + rl_action (用于 rl_action_mag)
            _rl_a = action if agent is not None else None
            stab_all.update_step(next_obs, config, env=env, rl_action=_rl_a)
            phase_stab[current_phase].update_step(next_obs, config, env=env,
                                                   rl_action=_rl_a)

            setattr(rstate, "phase_step", int(phase_steps[current_phase]))
            if current_phase == "cruise":
                reward, r_done, r_success, r_info = compute_cruise_reward(
                    env, next_obs, config, rstate,
                    rl_action=(_cruise_reward_action
                               if _cruise_reward_action is not None else _rl_a),
                    base_action=_cruise_reward_base)
            else:
                reward, r_done, r_success, r_info = REWARD_FNS[current_phase](
                    env, next_obs, config, rstate, rl_action=_rl_a)

            ep_reward += reward
            phase_rewards[current_phase] += reward
            phase_steps[current_phase] += 1
            ep_steps += 1

            if current_phase == "cruise" and (r_success or r_done):
                obs = next_obs
                state_after = get_phase_state(
                    env, obs, start_xy=start_xy, config=config)
                can_handoff, handoff_mode, handoff_detail = (
                    _cruise_handoff_status(
                        state_after, config, step=phase_steps["cruise"],
                        max_steps=_cruise_max, env=env))
                if can_handoff:
                    phase_success["cruise"] = True
                    current_phase = "descent"
                    rstate = REWARD_STATES["descent"]()
                    if hasattr(rstate, 'total_steps_global'):
                        rstate.total_steps_global = 10_000_000
                    rstate.current_xy_tol = float(config["insertion"].get(
                        "xy_tolerance_train_end", 0.005))
                    rstate.current_descent_level = 99; rstate.descent_n_levels = 1

                    current_q_after = env.data.qpos[:7].copy().astype(np.float32)
                    payload_pos_after = env.data.body('prefab').xpos.copy()
                    expert.reset(obs, current_q_after, env=env)
                    if planned_path is not None:
                        expert.set_path(planned_path)
                        _advance_expert_to_nearest_wp(
                            expert, planned_path, payload_pos_after)
                    expert.tracker._is_descending = True
                    ee_ctrl.reset(env._get_ee_pos(), current_q_after)
                    prev_tilt, prev_yaw = 0.0, 0.0
                    desc_agent = agents.get("descent")
                    if desc_agent is not None and hasattr(desc_agent, 'reset_history'):
                        desc_agent.reset_history()

                    _dtf_sw = np.linalg.norm(state_after["pl_xy"] - env.target_pos)
                    _target_pz = float(config["insertion"].get(
                        "target_payload_z", 0.10))
                    _handoff_xy = float(config.get("phase_transition", {}).get(
                        "cruise_to_descent_safe_xy_dist", 0.020))
                    _z_cruise = float(config.get("planning", {}).get(
                        "payload_z_cruise", 0.25))
                    print(f"    [pipeline s{step}] NMPC->PID+RL "
                          f"({handoff_mode}) {handoff_detail} "
                          f"dtf={_dtf_sw*1000:.1f}mm "
                          f"(<= {_handoff_xy*1000:.0f}) "
                          f"z={state_after['payload_z']*1000:.1f}mm "
                          f"(target {_z_cruise*1000:.0f}) "
                          f"insert_z={_target_pz*1000:.1f}mm")
                    continue

                # Single-phase cruise reward is intentionally loose. In the
                # pipeline it is only a progress signal; keep NMPC cruising
                # until the strict descent-init-matched handoff gate is met.
                if r_done and not r_success:
                    term_reason = (
                        f"{r_info.get('termination', 'cruise_done')};"
                        f"handoff={handoff_detail}")
                    break
                _dtf_wait = np.linalg.norm(state_after["pl_xy"] - env.target_pos)
                if phase_steps["cruise"] >= _cruise_max:
                    term_reason = (
                        f"cruise_handoff_timeout:dtf={_dtf_wait*1000:.1f}mm,"
                        f"z={state_after['payload_z']*1000:.1f}mm;"
                        f"handoff={handoff_detail}")
                    break
                if env_info.get("nan_detected", False):
                    term_reason = "nan_detected"
                    break
                term_reason = None
                continue

            if r_success and current_phase == "descent":
                phys_ok, phys_detail = check_physical_insertion(env, config)
                mark_str = "✅ 物理插入成功" if phys_ok else "⚠ 课程容差达标但未插入"
                if phys_ok or not physical_finish_active:
                    print(f"    [{mark_str}] {phys_detail}")
                if phys_ok:
                    final_success = True
                    phase_success["descent"] = True
                    term_reason = r_info.get("termination", "insertion_success")
                    obs = next_obs; break
                else:
                    if not physical_finish_active:
                        final_z = _activate_physical_insert_finish(expert, config)
                        if final_z is not None:
                            print(f"    [physical insert finish] PID target_z={final_z*1000:.1f}mm")
                        physical_finish_active = True
                    r_done = False
                    r_success = False
                    r_info = dict(r_info)
                    r_info.pop("termination", None)
                    term_reason = "reward_success_waiting_physical_insert"

            if current_phase == "descent" and physical_finish_active and not final_success:
                phys_ok, phys_detail = check_physical_insertion(env, config)
                if phys_ok:
                    final_success = True
                    phase_success["descent"] = True
                    term_reason = "physical_insertion_success"
                    print(f"    [✅ 物理插入成功] {phys_detail}")
                    obs = next_obs; break

            descent_max_steps = _descent_max
            if current_phase == "descent" and phase_steps["descent"] >= descent_max_steps:
                term_reason = f"descent_timeout_{descent_max_steps}"; break

            if r_info.get("termination"):
                term_reason = r_info["termination"]
            obs = next_obs
            if r_done or env_info.get("nan_detected", False):
                break

        ep_count += 1
        results.append({
            "reward":          ep_reward,
            "steps":           ep_steps,
            "success":         final_success,
            "physical_success": final_success,
            "termination":     term_reason or "timeout",
            "phase_rewards":   phase_rewards.copy(),
            "phase_steps":     phase_steps.copy(),
            "phase_success":   phase_success.copy(),
            "stability":       stab_all.summary(),
            "phase_stability": {p: phase_stab[p].summary()
                                for p in ["cruise", "descent"]},
            "wind_force":      wind_force,
        })
        mark = "✅" if final_success else "❌"
        term_short = (term_reason or "timeout").split(":")[0]
        phases_str = " → ".join([
            f"{'✅' if phase_success[p] else '❌'}{p[0].upper()}"
            for p in ["cruise", "descent"]])
        print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | "
              f"Steps:{ep_steps:3d} | {phases_str} | {term_short} | "
              f"{stab_all.print_line()}")

    return results


# ==============================================================================
# 汇总
# ==============================================================================

def print_summary(results, mode_name):
    """打印测试结果汇总。"""
    if not results:
        print(f"[{mode_name}] 无有效结果"); return
    sr     = np.mean([r["success"] for r in results])
    avg_r  = np.mean([r["reward"]  for r in results])
    avg_s  = np.mean([r["steps"]   for r in results])
    terms  = Counter((r["termination"] or "unknown").split(":")[0] for r in results)
    wind_f = results[0].get("wind_force", 0.0)
    print(f"\n{'='*60}")
    print(f"  [{mode_name}] 结果汇总" +
          (f"  (风力={wind_f:.1f}N)" if wind_f > 0 else ""))
    print(f"{'='*60}")
    print(f"  成功率:    {sr*100:.1f}%")
    print(f"  平均奖励:  {avg_r:.2f}")
    print(f"  平均步数:  {avg_s:.1f}")
    print(f"  终止原因:  {dict(terms)}")
    if results and "phase_success" in results[0]:
        for p in ["cruise", "descent"]:
            p_sr = np.mean([r["phase_success"].get(p, False) for r in results])
            p_r  = np.mean([r["phase_rewards"].get(p, 0)    for r in results])
            p_s  = np.mean([r["phase_steps"].get(p, 0)      for r in results])
            print(f"  {p:>8s}: SR={p_sr*100:.0f}% | R={p_r:.2f} | Steps={p_s:.0f}")

    def _stab_block(stab_list, label):
        if not stab_list: return
        valid = [s for s in stab_list if s]
        if not valid: return
        ke_avg = np.mean([s.get("avg_ke_mJ", 0) for s in valid])
        ke_p95 = np.mean([s.get("p95_ke_mJ", 0) for s in valid])
        ke_max = np.mean([s.get("max_ke_mJ", 0) for s in valid])
        ang_avg = np.mean([s.get("avg_angle", 0) for s in valid])
        ang_p95 = np.mean([s.get("p95_angle", 0) for s in valid])
        ang_max = np.mean([s.get("max_angle", 0) for s in valid])
        acc_avg = np.mean([s.get("avg_acc",    0) for s in valid])
        acc_max = np.mean([s.get("max_acc",    0) for s in valid])
        print(f"\n  ── {label} (N={len(valid)}) ──")
        print(f"  摆动动能  avg/p95/max: {ke_avg:6.1f} / {ke_p95:6.1f} / {ke_max:6.1f} mJ")
        print(f"  摆动角度  avg/p95/max: {ang_avg:5.2f} / {ang_p95:5.2f} / {ang_max:5.2f} °")
        print(f"  EE 加速度 avg/max:     {acc_avg:.3f} / {acc_max:.3f} m/s²")

    _stab_block([r.get("stability") for r in results], "全过程稳定性")
    if results and "phase_stability" in results[0]:
        for p in ["cruise", "descent"]:
            p_stabs = [r["phase_stability"].get(p) for r in results
                       if r.get("phase_stability")]
            _stab_block(p_stabs, f"{p} 阶段稳定性")
    print()


def summarize_results(results):
    if not results:
        return {}
    terms = Counter((r["termination"] or "unknown").split(":")[0]
                    for r in results)
    summary = {
        "episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "termination_counts": dict(terms),
    }
    stab_keys = [
        "avg_ke_mJ", "p95_ke_mJ", "max_ke_mJ", "rms_ke_mJ",
        "integral_ke_mJs", "avg_angle", "p95_angle", "max_angle",
        "rms_angle", "avg_acc", "max_acc", "pl_vel_peak", "pl_vel_rms",
        "cable_ke_peak", "cable_ke_avg", "cable_ke_integral",
        "rl_action_mag_mean", "rl_action_mag_peak",
    ]
    stabs = [r.get("stability") or {} for r in results]
    for key in stab_keys:
        vals = [float(s[key]) for s in stabs if key in s]
        if vals:
            summary[key] = float(np.mean(vals))
    return summary


def parse_wind_bins(spec):
    bins = []
    for raw in str(spec).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "-" not in raw:
            val = float(raw)
            bins.append((val, val))
        else:
            lo, hi = raw.split("-", 1)
            bins.append((float(lo), float(hi)))
    if not bins:
        raise ValueError("No valid wind bins provided")
    return bins


def run_wind_benchmark(args, config):
    if args.phase != "descent":
        raise ValueError("--compare-wind-bins currently targets --phase descent")
    if args.ckpt is None:
        raise ValueError("--ckpt is required for Residual RL benchmark")
    if args.algo == "expert":
        raise ValueError("--compare-wind-bins requires --algo ppo or --algo sac")

    config["sim"]["render"] = False
    eval_init = get_descent_eval_init(config, args.eval_curriculum_level)
    bins = parse_wind_bins(args.wind_bins)
    episodes_per_bin = int(args.episodes_per_bin)
    if episodes_per_bin <= 0:
        raise ValueError("--episodes-per-bin must be positive")
    out_path = args.benchmark_out or os.path.join(
        "test_results",
        f"descent_wind_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print_curriculum_hardest_task(config)
    print_descent_eval_settings(config, eval_init)
    print(f"\n[Benchmark] wind bins={bins}, episodes_per_bin={episodes_per_bin}")
    print(f"[Benchmark] output={out_path}\n", flush=True)

    rows = []
    for bin_id, (lo, hi) in enumerate(bins):
        case_seed = int(args.seed) + 1009 * bin_id
        rng = np.random.default_rng(case_seed)
        if hi <= lo:
            wind_forces = np.full(episodes_per_bin, float(lo), dtype=np.float64)
        else:
            wind_forces = rng.uniform(float(lo), float(hi), episodes_per_bin)
        wind_dirs = rng.uniform(0.0, 2.0 * np.pi, episodes_per_bin)
        reset_seeds = rng.integers(0, np.iinfo(np.int32).max,
                                   episodes_per_bin, dtype=np.int64)

        for mode in ("expert", "residual_rl"):
            np.random.seed(case_seed)
            try:
                torch.manual_seed(case_seed)
            except Exception:
                pass
            t_mode = time.perf_counter()
            print(f"[Benchmark] preparing bin {bin_id+1}/{len(bins)} "
                  f"{lo:.3f}-{hi:.3f}N | {mode} ...", flush=True)
            t0 = time.perf_counter()
            env = CableRobotEnvWithObstacles(config=config)
            try:
                print(f"[Benchmark] env ready in {time.perf_counter()-t0:.1f}s",
                      flush=True)
                if hasattr(env, 'set_curriculum_n_obstacles'):
                    env.set_curriculum_n_obstacles(config["scene"]["n_obstacles"])
                expert = JointSpaceExpert(config, env.ik_solver)
                ee_ctrl = EEAccController(config, env.ik_solver)
                print("[Benchmark] controllers ready", flush=True)
                agent = None
                if mode == "residual_rl":
                    t0 = time.perf_counter()
                    agent = load_agent("descent", args.algo, args.ckpt, config)
                    print(f"[Benchmark] ckpt loaded in {time.perf_counter()-t0:.1f}s",
                          flush=True)

                print(f"[Benchmark] running bin {bin_id+1}/{len(bins)} "
                      f"{lo:.3f}-{hi:.3f}N | {mode} ...", flush=True)
                prog_every = int(getattr(args, "benchmark_progress_every", 32))
                if prog_every <= 0:
                    prog_every = 0
                results = test_single_phase(
                    env, agent, expert, ee_ctrl, "descent", config,
                    n_episodes=episodes_per_bin,
                    wind_force_sampler=lambda k, wf=wind_forces: wf[k],
                    wind_dir_sampler=lambda k, wd=wind_dirs: wd[k],
                    reset_seed_sampler=lambda k, rs=reset_seeds: rs[k],
                    obs_noise=args.obs_noise, act_noise=args.act_noise,
                    force_noise=args.force_noise,
                    eval_cur_init=eval_init,
                    verbose=False,
                    progress_every=prog_every,
                    progress_prefix=(f"  bin {bin_id+1}/{len(bins)} {mode}"))
            finally:
                env.close()

            summary = summarize_results(results)
            row = {
                "bin_id": bin_id,
                "wind_low": float(lo),
                "wind_high": float(hi),
                "mode": mode,
                **summary,
                "termination_counts": json.dumps(
                    summary.get("termination_counts", {}),
                    ensure_ascii=False, sort_keys=True),
            }
            rows.append(row)
            print(f"  -> SR={summary.get('success_rate', 0)*100:.1f}% "
                  f"steps={summary.get('avg_steps', 0):.1f} "
                  f"KE={summary.get('avg_ke_mJ', 0):.1f}mJ "
                  f"p95KE={summary.get('p95_ke_mJ', 0):.1f}mJ "
                  f"elapsed={time.perf_counter()-t_mode:.1f}s",
                  flush=True)

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n[Benchmark Summary]")
    for row in rows:
        print(f"  bin {row['bin_id']} {row['wind_low']:.3f}-{row['wind_high']:.3f}N "
              f"{row['mode']:>11s}: SR={row['success_rate']*100:5.1f}% "
              f"steps={row['avg_steps']:6.1f} "
              f"avgKE={row.get('avg_ke_mJ', 0):6.1f}mJ "
              f"p95KE={row.get('p95_ke_mJ', 0):6.1f}mJ "
              f"plV_rms={row.get('pl_vel_rms', 0):.3f}")
    print(f"\nSaved benchmark CSV: {out_path}")


# ==============================================================================
# 入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="[v13.0] 两阶段 RL 测试: cruise (NMPC 抬升+平移) + descent (PID 下降) + pipeline")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["cruise", "descent", "pipeline", "lift"],
                        help="[v13.0] 'lift' 自动重定向到 cruise (lift 已合并)")
    parser.add_argument("--algo", type=str, default="expert",
                        choices=["ppo", "sac", "expert"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--wait-for-space", action="store_true",
                        help="render only: pause after each episode reset until Space is pressed in the MuJoCo viewer")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--insert-target-z", type=float, default=0.10,
                        help="Descent only: trained alignment target payload COM z (m)")
    parser.add_argument("--insert-depth-min", type=float, default=0.025,
                        help="Descent only: minimum visible rebar insertion depth (m)")
    parser.add_argument("--physical-insert-target-z", type=float, default=0.090,
                        help="Descent only: PID finishing target z after alignment success (m)")

    # pipeline 模式 [v13.0] 只有 2 个 ckpt
    parser.add_argument("--cruise-ckpt",  type=str, default=None)
    parser.add_argument("--descent-ckpt", type=str, default=None)
    parser.add_argument("--cruise-algo",  type=str, default="expert",
                        choices=["ppo", "sac", "expert"])
    parser.add_argument("--descent-algo", type=str, default="ppo",
                        choices=["ppo", "sac", "expert"])
    parser.add_argument("--pipeline-insert-target-z", type=float, default=0.10,
                        help="Pipeline only: trained alignment target payload COM z (m)")
    parser.add_argument("--pipeline-insert-depth-min", type=float, default=0.025,
                        help="Pipeline only: minimum visible rebar insertion depth (m)")
    parser.add_argument("--pipeline-physical-insert-target-z", type=float, default=0.090,
                        help="Pipeline only: PID finishing target z after alignment success (m)")
    parser.add_argument("--pipeline-cruise-z", type=float, default=0.25,
                        help="Pipeline only: NMPC terminal payload z above the rebars (m)")
    # 兼容: 旧 --lift-ckpt 参数仍允许, 但会被映射成 cruise-ckpt
    parser.add_argument("--lift-ckpt",    type=str, default=None,
                        help="[v13.0 兼容] 自动映射到 --cruise-ckpt")
    parser.add_argument("--lift-algo",    type=str, default=None,
                        choices=["ppo", "sac", "expert", None],
                        help="[v13.0 兼容] 自动映射到 --cruise-algo")

    # ── 噪声/风力扰动 (默认全部 0) [v8] ──────────────────────────────────────
    parser.add_argument("--wind-force",  type=float, default=0.0,
                        help="施加恒定风力 (N), 0=无风")
    parser.add_argument("--wind-dir",    type=float, default=None,
                        help="风向 (rad), None=随机")
    parser.add_argument("--obs-noise",   type=float, default=0.0,
                        help="观测噪声标准差 (加到 normalized obs)")
    parser.add_argument("--act-noise",   type=float, default=0.0,
                        help="执行噪声标准差 (加到 delta_q, rad/step)")
    parser.add_argument("--force-noise", type=float, default=0.0,
                        help="环境噪声力标准差 (N, 加在 payload 上)")
    parser.add_argument("--eval-curriculum-level", type=str, default="max",
                        help="descent eval init level: 'max' or a numeric curriculum level")
    parser.add_argument("--compare-wind-bins", action="store_true",
                        help="run expert vs Residual RL descent benchmark over wind bins")
    parser.add_argument("--episodes-per-bin", type=int, default=512,
                        help="episodes per wind bin for --compare-wind-bins")
    parser.add_argument("--wind-bins", type=str,
                        default="0.00-0.02,0.02-0.05,0.05-0.10,0.10-0.20,0.20-0.35",
                        help="comma separated wind bins in N, e.g. 0-0.02,0.02-0.05")
    parser.add_argument("--benchmark-out", type=str, default=None,
                        help="CSV output path for --compare-wind-bins")
    parser.add_argument("--benchmark-progress-every", type=int, default=32,
                        help="print benchmark progress every N episodes; 0 disables it")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-episode logs in single-phase tests")

    args = parser.parse_args()
    np.random.seed(int(args.seed))
    try:
        torch.manual_seed(int(args.seed))
    except Exception:
        pass

    # [v13.0] lift 重定向到 cruise
    if args.phase == "lift":
        print("\n[v13.0 提示] lift 已合并进 cruise, 自动重定向 --phase lift → cruise\n")
        args.phase = "cruise"
    # 兼容: 旧的 --lift-ckpt 映射到 --cruise-ckpt
    if args.lift_ckpt and not args.cruise_ckpt:
        print(f"[v13.0 兼容] --lift-ckpt {args.lift_ckpt} → --cruise-ckpt")
        args.cruise_ckpt = args.lift_ckpt
    if args.lift_algo and args.lift_algo != "ppo":
        args.cruise_algo = args.lift_algo

    config = build_config(args)
    config["scene"]["seed"] = args.seed

    if args.compare_wind_bins:
        run_wind_benchmark(args, config)
        return

    if args.obstacles is not None:
        test_n_obs = min(int(args.obstacles), config["scene"]["n_obstacles"])
    else:
        test_n_obs = config["scene"]["n_obstacles"]

    env = CableRobotEnvWithObstacles(config=config)
    install_space_start_callback(
        env,
        bool(args.wait_for_space and args.render and not args.compare_wind_bins))
    if args.wait_for_space and not args.render:
        print("[wait-for-space] 未开启 --render，等待空格设置已忽略")
    if hasattr(env, 'set_curriculum_n_obstacles'):
        env.set_curriculum_n_obstacles(test_n_obs)
    expert  = JointSpaceExpert(config, env.ik_solver)
    ee_ctrl = EEAccController(config, env.ik_solver)

    print(f"\n{'='*60}")
    print(f"  测试配置: phase={args.phase} algo={args.algo} eps={args.episodes}")
    print(f"  扰动: wind={args.wind_force:.1f}N, force_noise={args.force_noise:.2f}, "
          f"obs_noise={args.obs_noise:.3f}, act_noise={args.act_noise:.3f}")
    print(f"{'='*60}\n")

    if args.phase == "pipeline":
        # [v13.0] 只 2 阶段: cruise (合并 lift+cruise) + descent
        agents = {"cruise": None}
        if args.cruise_algo == "expert":
            print("[CRUISE] pure NMPC/expert control (no residual RL)")
        else:
            if args.cruise_ckpt is None:
                print(f"[Error] --cruise-ckpt 必须指定 ({args.cruise_algo} 模式)"); return
            agents["cruise"] = load_agent("cruise", args.cruise_algo,
                                          args.cruise_ckpt, config)
            print(f"[CRUISE] 加载 NMPC residual: "
                  f"{args.cruise_ckpt} ({args.cruise_algo})")

        if args.descent_algo == "expert":
            agents["descent"] = None
            print("[DESCENT] PID expert control (no residual RL)")
        else:
            if args.descent_ckpt is None:
                print(f"[Error] --descent-ckpt 必须指定 ({args.descent_algo} 模式)"); return
            agents["descent"] = load_agent("descent", args.descent_algo,
                                           args.descent_ckpt, config)
            print(f"[DESCENT] 加载: {args.descent_ckpt} ({args.descent_algo})")
        results = test_pipeline(
            env, agents, expert, ee_ctrl, config,
            n_episodes=args.episodes,
            wind_force=args.wind_force, wind_dir=args.wind_dir,
            obs_noise=args.obs_noise, act_noise=args.act_noise,
            force_noise=args.force_noise)
        print_summary(
            results,
            f"Pipeline-cruise_{args.cruise_algo}-descent_{args.descent_algo}")
    else:
        eval_cur_init = None
        if args.phase == "descent":
            eval_cur_init = get_descent_eval_init(
                config, args.eval_curriculum_level)
            print_curriculum_hardest_task(config)
            print_descent_eval_settings(config, eval_cur_init)

        if args.algo == "expert":
            agent = None
        else:
            if args.ckpt is None:
                print(f"[Error] --ckpt 必须指定 ({args.algo} 模式)"); return
            agent = load_agent(args.phase, args.algo, args.ckpt, config)
            print(f"[{args.phase.upper()}] 加载: {args.ckpt} ({args.algo})")
        results = test_single_phase(
            env, agent, expert, ee_ctrl, args.phase, config,
            n_episodes=args.episodes,
            wind_force=args.wind_force, wind_dir=args.wind_dir,
            obs_noise=args.obs_noise, act_noise=args.act_noise,
            force_noise=args.force_noise,
            eval_cur_init=eval_cur_init,
            verbose=not args.quiet)
        print_summary(results, f"{args.phase}-{args.algo}")

    env.close()


if __name__ == "__main__":
    main()
