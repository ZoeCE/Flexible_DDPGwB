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
import numpy as np
import torch
from collections import Counter

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from phase_agent import (
    PPOPhaseAgent, SACPhaseAgent,
    build_lift_obs, build_cruise_obs, build_descent_obs,
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
    if args.obstacles is not None:
        config["scene"]["n_obstacles"] = max(int(args.obstacles),
                                              config["scene"]["n_obstacles"])
    return config


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
        # [v13.1] train (compute_cruise_reward) 和 test_pipeline (此函数) 使用一致的判定:
        # 读相同 config keys (cruise_rl.reward 中的 success_*).
        # 用户反馈 "确保 train 和 test 的标准一致".
        _rcfg_c  = config["cruise_rl"]["reward"]
        z_lock   = float(config["cruise_rl"].get("z_lock_height", 0.25))
        z_tol_frac    = float(_rcfg_c.get("z_success_tol_frac", 0.40))
        z_success_tol = z_lock * z_tol_frac
        success_radius = float(_rcfg_c.get("success_radius", 0.12))
        # success_* keys 比 cruise_to_descent_* 宽松 (按 v13.1 设计):
        tilt_max   = float(_rcfg_c.get("success_tilt_max",         0.20))
        swing_max  = float(_rcfg_c.get("success_swing_vel_max",    0.30))
        vel_max    = float(_rcfg_c.get("success_payload_vel_max",  0.35))

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
                state["swing_vel"] < fallback_swing and
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


# ==============================================================================
# 单阶段测试
# ==============================================================================

def test_single_phase(env, agent, expert, ee_ctrl, phase, config,
                      n_episodes=20, deterministic=True,
                      wind_force=0.0, wind_dir=None,
                      obs_noise=0.0, act_noise=0.0, force_noise=0.0):
    """测试单个阶段, 支持噪声/风力扰动。"""
    from train_phase import (reset_for_phase, build_phase_obs,
                             REWARD_FNS, REWARD_STATES,
                             _apply_descent_pid_residual)

    z_pid   = CruiseZYawPID(config)          if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    _cruise_nmpc_base = (phase == "cruise" and
        bool(config.get("cruise_rl", {}).get("use_nmpc_base", True)))

    results = []
    ep_count = 0; attempt = 0

    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        obs, planned_path = reset_for_phase(env, phase, config)
        if obs is None:
            continue

        # ── 施加扰动 ──────────────────────────────────────────────────────────
        apply_perturbations(env, wind_force, wind_dir, force_noise)

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
            rstate.current_xy_tol = float(config["insertion"].get(
                "xy_tolerance_train_end", 0.005))
            rstate.current_descent_level = 99   # 不限速
            rstate.descent_n_levels = 1

        ep_reward = 0.0; ep_steps = 0; ep_success = False
        term_reason = None
        stab_metrics = StabilityMetrics()

        max_steps = (200 if phase == "descent"
                     else int(config[f"{phase}_rl"]["max_steps"]))

        for step in range(max_steps):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()

            if agent is None:
                delta_q = expert.compute_delta_q_target(obs, current_q)
            else:
                p_obs, prev_tilt, prev_yaw = build_phase_obs(
                    phase, obs, env, start_xy, target_xy, prev_tilt, prev_yaw)
                norm_obs = agent.normalize_obs(p_obs, update=False)
                norm_obs = _add_obs_noise(norm_obs, obs_noise)

                result = agent.act(norm_obs, deterministic=deterministic)
                if isinstance(result, tuple):
                    action = result[0]
                else:
                    action = result

                real_ee = env._get_ee_pos()
                if phase == "cruise" and z_pid is not None:
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
                            expert, action, obs, env, config, current_q)
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

            reward, r_done, r_success, r_info = REWARD_FNS[phase](
                env, next_obs, config, rstate)

            ep_reward += reward; ep_steps += 1

            if r_success:
                if phase == "descent":
                    phys_ok, phys_detail = check_physical_insertion(env, config)
                    ep_success = phys_ok
                    mark_str = "✅ 物理插入成功" if phys_ok else "⚠ 课程容差达标但未插入"
                    print(f"    [{mark_str}] {phys_detail}")
                else:
                    ep_success = True
                term_reason = r_info.get("termination", "success")
                if phase != "descent" or ep_success:
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
            "wind_force":  wind_force,
        })
        mark = "✅" if ep_success else "❌"
        term_short = (term_reason or "timeout").split(":")[0]
        print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | Steps:{ep_steps:3d} | "
              f"{term_short} | {stab_metrics.print_line()}")

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
                             _apply_descent_pid_residual)

    z_pid   = CruiseZYawPID(config)
    swing_d = SwingDampingController(config)
    _cruise_nmpc_base = bool(config.get("cruise_rl", {}).get("use_nmpc_base", True))
    _cruise_max = int(config["cruise_rl"]["max_steps"])

    results = []
    ep_count = 0; attempt = 0

    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        sys_stdout_saved = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            obs, planned_path = reset_for_phase(env, "lift", config)
        finally:
            sys.stdout.close(); sys.stdout = sys_stdout_saved
        if obs is None: continue

        apply_perturbations(env, wind_force, wind_dir, force_noise)

        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        if planned_path is not None:
            # [v13.0] 完整 path (lift+cruise+descent), cruise 段 NMPC 自处理 lift→cruise 边界
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
        stab_all = StabilityMetrics()
        phase_stab = {p: StabilityMetrics() for p in ["cruise", "descent"]}

        for step in range(config["sim"]["max_steps"]):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()

            state = get_phase_state(env, obs, start_xy=start_xy, config=config)

            # ── 阶段切换 [v13.0] 只有 cruise→descent ─────────────────────────
            if (current_phase == "cruise" and
                  check_phase_transition("cruise", state, config,
                                         step=phase_steps["cruise"],
                                         max_steps=_cruise_max, env=env)):
                phase_success["cruise"] = True
                current_phase = "descent"
                rstate = REWARD_STATES["descent"]()
                if hasattr(rstate, 'total_steps_global'):
                    rstate.total_steps_global = 10_000_000
                rstate.current_xy_tol = float(config["insertion"].get(
                    "xy_tolerance_train_end", 0.005))
                rstate.current_descent_level = 99; rstate.descent_n_levels = 1
                ee_ctrl.reset(env._get_ee_pos(), current_q)
                _dtf_sw = np.linalg.norm(state["pl_xy"] - env.target_pos)
                print(f"    [pipeline s{step}] Cruise→Descent ✅  "
                      f"dtf={_dtf_sw*1000:.1f}mm")

            # ── 动作计算 ──────────────────────────────────────────────────────
            agent = agents.get(current_phase)
            if agent is None:
                delta_q = expert.compute_delta_q_target(obs, current_q)
            else:
                p_obs, prev_tilt, prev_yaw = build_phase_obs(
                    current_phase, obs, env, start_xy, target_xy, prev_tilt, prev_yaw)
                norm_obs = agent.normalize_obs(p_obs, update=False)
                norm_obs = _add_obs_noise(norm_obs, obs_noise)

                result = agent.act(norm_obs, deterministic=deterministic)
                action = result[0] if isinstance(result, tuple) else result
                real_ee = env._get_ee_pos()

                if current_phase == "cruise":
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
                            expert, action, obs, env, config, current_q)
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

            reward, r_done, r_success, r_info = REWARD_FNS[current_phase](
                env, next_obs, config, rstate)

            ep_reward += reward
            phase_rewards[current_phase] += reward
            phase_steps[current_phase] += 1
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
                                for p in ["lift", "cruise", "descent"]},
            "wind_force":      wind_force,
        })
        mark = "✅" if final_success else "❌"
        term_short = (term_reason or "timeout").split(":")[0]
        phases_str = " → ".join([
            f"{'✅' if phase_success[p] else '❌'}{p[0].upper()}"
            for p in ["lift", "cruise", "descent"]])
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
        for p in ["lift", "cruise", "descent"]:
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
        for p in ["lift", "cruise", "descent"]:
            p_stabs = [r["phase_stability"].get(p) for r in results
                       if r.get("phase_stability")]
            _stab_block(p_stabs, f"{p} 阶段稳定性")
    print()


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
    parser.add_argument("--gpu", type=int, default=0)

    # pipeline 模式 [v13.0] 只有 2 个 ckpt
    parser.add_argument("--cruise-ckpt",  type=str, default=None)
    parser.add_argument("--descent-ckpt", type=str, default=None)
    parser.add_argument("--cruise-algo",  type=str, default="ppo",
                        choices=["ppo", "sac", "expert"])
    parser.add_argument("--descent-algo", type=str, default="ppo",
                        choices=["ppo", "sac", "expert"])
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

    args = parser.parse_args()

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

    if args.obstacles is not None:
        test_n_obs = min(int(args.obstacles), config["scene"]["n_obstacles"])
    else:
        test_n_obs = config["scene"]["n_obstacles"]

    env = CableRobotEnvWithObstacles(config=config)
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
        agents = {}
        for p, p_algo, p_ckpt in [
            ("cruise",  args.cruise_algo,  args.cruise_ckpt),
            ("descent", args.descent_algo, args.descent_ckpt),
        ]:
            if p_algo == "expert":
                agents[p] = None
            else:
                if p_ckpt is None:
                    print(f"[Error] --{p}-ckpt 必须指定 ({p_algo} 模式)"); return
                agents[p] = load_agent(p, p_algo, p_ckpt, config)
                print(f"[{p.upper()}] 加载: {p_ckpt} ({p_algo})")
        results = test_pipeline(
            env, agents, expert, ee_ctrl, config,
            n_episodes=args.episodes,
            wind_force=args.wind_force, wind_dir=args.wind_dir,
            obs_noise=args.obs_noise, act_noise=args.act_noise,
            force_noise=args.force_noise)
        print_summary(results, f"Pipeline-{args.lift_algo}_{args.cruise_algo}_{args.descent_algo}")
    else:
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
            force_noise=args.force_noise)
        print_summary(results, f"{args.phase}-{args.algo}")

    env.close()


if __name__ == "__main__":
    main()