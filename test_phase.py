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
import copy
import argparse
import numpy as np
import torch
from collections import Counter

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from controller import JointSpaceExpert
from phase_agent import (
    PPOPhaseAgent, SACPhaseAgent, DescentDualRLAgent,
    build_lift_obs, build_cruise_obs, build_descent_obs,
)
from phase_reward import (
    compute_lift_reward, compute_cruise_reward, compute_descent_reward,
    LiftRewardState, CruiseRewardState, DescentRewardState,
)
from ee_acc_controller import EEAccController, CruiseZYawPID, SwingDampingController
from scipy.spatial.transform import Rotation as R


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
    # [v10] descent 段检查是否使用 dual-RL
    use_dual_rl = (phase == "descent" and
                   config.get("descent_rl", {}).get("use_dual_rl", False))
    if use_dual_rl and algo == "ppo":
        agent = DescentDualRLAgent(config=config)
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


def get_phase_state(env, obs):
    """获取当前物理状态, 用于阶段切换判断。"""
    pl_xy = np.array([obs[4], obs[5]])
    payload_z = float(env.data.body('prefab').xpos[2])
    pl_vxy = obs[6:8]
    ee_vxy = obs[2:4]
    swing_vel = float(np.linalg.norm(pl_vxy - ee_vxy))
    pl_vel = float(np.linalg.norm(pl_vxy))
    
    pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    
    return {
        "pl_xy": pl_xy,
        "payload_z": payload_z,
        "swing_vel": swing_vel,
        "pl_vel": pl_vel,
        "tilt": tilt,
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


def check_phase_transition(phase, state, config):
    """检查是否满足阶段切换条件 (用于 pipeline 测试)。"""
    pt = config["phase_transition"]
    target_xy = np.array(config["task"]["default_target_xy"])

    if phase == "lift":
        return (state["payload_z"] >= pt["lift_to_cruise_z_threshold"] and
                state["tilt"] < pt["lift_to_cruise_tilt_max"] and
                state["swing_vel"] < pt["lift_to_cruise_swing_vel_max"])

    elif phase == "cruise":
        dtf = float(np.linalg.norm(state["pl_xy"] - target_xy))
        return (dtf < pt["cruise_to_descent_xy_dist"] and
                state["tilt"] < pt["cruise_to_descent_tilt_max"] and
                state["swing_vel"] < pt["cruise_to_descent_swing_vel_max"] and
                state["pl_vel"] < pt["cruise_to_descent_payload_vel_max"])

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


def test_single_phase(env, agent, expert, ee_ctrl, phase, config,
                      n_episodes=20, deterministic=True):
    """测试单个阶段。"""
    from train_phase import reset_for_phase, build_phase_obs, REWARD_FNS, REWARD_STATES
    
    # ★ Cruise 段控制器
    z_pid   = CruiseZYawPID(config)         if phase == "cruise" else None
    swing_d = SwingDampingController(config) if phase == "cruise" else None
    
    results = []
    ep_count = 0
    attempt = 0
    
    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        obs, planned_path = reset_for_phase(env, phase, config)
        if obs is None:
            continue
        
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
        # [v3.5] 测试 success_radius = 课程最后阶段的值 (n_obs=3 对应的 radius)
        # 训练末期: get_current_success_radius(n_obs=3) = success_radius_end = 0.10m
        # 测试判定与训练最后阶段完全一致: dtf < 0.10m + 稳定性条件
        if hasattr(rstate, 'current_success_radius'):
            _rcfg = config.get("cruise_rl", {}).get("reward", {})
            _train_final_sr = float(_rcfg.get("success_radius_end", 0.10))
            rstate.current_success_radius = _train_final_sr
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
            trajectory.append({
                "payload": payload_pos.copy(),
                "ee": ee_pos.copy(),
                "tilt": tilt_val,
                "yaw": yaw_val,
                "swing_vel": swing_vel,
                "step": step,
            })
            
            if agent is None:  # 专家模式
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
                else:
                    # [v3.4 FIX] Descent: PID base + RL residual (与训练架构一致)
                    # 训练中 RL 只输出残差 acc, 必须叠加 PID base delta_q 才能正常工作
                    # 直接用纯 RL 会"完全乱跑": residual acc 很小, 没有 PID 驱动下降
                    from train_phase import _apply_descent_pid_residual
                    _pid_residual_mode = bool(
                        config.get("descent_rl", {}).get("pid_residual_mode", True))
                    if _pid_residual_mode:
                        delta_q, _pid_dq = _apply_descent_pid_residual(
                            expert, action, obs, env, config, current_q)
                    else:
                        # 纯 RL 模式 (兼容旧训练)
                        _vmax_z_d = float(config.get("ee_control", {}).get(
                            "vel_max_z_descent", 0.03))
                        delta_q = ee_ctrl.compute_delta_q(
                            action, current_q, real_ee, vel_max_z=_vmax_z_d)
            
            next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
            
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
                else:
                    ep_success = True
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
        results.append({
            "reward": ep_reward,
            "steps": ep_steps,
            "success": ep_success,
            "physical_success": ep_success,
            "termination": term_reason or "timeout",
            "trajectory": trajectory,
        })
        
        mark = "✅" if ep_success else "❌"
        term_short = (term_reason or "timeout").split(":")[0]
        print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | "
              f"Steps:{ep_steps:3d} | {term_short}")
    
    return results


# ==============================================================================
# 全流水线测试 (3 阶段串联)
# ==============================================================================

def test_pipeline(env, agents, expert, ee_ctrl, config,
                  n_episodes=20, deterministic=True):
    """测试完整 3 阶段流水线。"""
    from train_phase import build_phase_obs, REWARD_FNS, REWARD_STATES
    
    # ★ Cruise 段控制器
    z_pid   = CruiseZYawPID(config)
    swing_d = SwingDampingController(config)
    
    results = []
    ep_count = 0
    attempt = 0
    
    while ep_count < n_episodes and attempt < n_episodes * 5:
        attempt += 1
        obs = env.reset()
        planned_path = env.get_planned_path()
        if planned_path is None:
            continue
        
        current_q = env.data.qpos[:7].copy()
        expert.reset(obs, current_q, env=env)
        expert.set_path(planned_path)
        ee_ctrl.reset(env._get_ee_pos(), current_q)
        # ★ 初始化 PID
        _pl_z = float(env.data.body('prefab').xpos[2])
        _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
        _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
        z_pid.reset(_pl_z, _pl_yaw)
        
        start_xy = env.default_start_xy.copy()
        target_xy = env.target_pos.copy()
        prev_tilt, prev_yaw = 0.0, 0.0
        
        # 阶段状态机
        current_phase = "lift"
        phase_rewards = {"lift": 0.0, "cruise": 0.0, "descent": 0.0}
        phase_steps = {"lift": 0, "cruise": 0, "descent": 0}
        phase_success = {"lift": False, "cruise": False, "descent": False}
        rstate = REWARD_STATES[current_phase]()
        # ★ 测试时严格判定参数
        if hasattr(rstate, 'total_steps_global'):
            rstate.total_steps_global = 10_000_000
        if hasattr(rstate, 'current_success_radius'):
            _rcfg_p = config.get("cruise_rl", {}).get("reward", {})
            rstate.current_success_radius = float(_rcfg_p.get("success_radius_test", 0.06))

        ep_reward = 0.0
        ep_steps = 0
        final_success = False
        term_reason = None
        trajectory = []
        
        for step in range(config["sim"]["max_steps"]):
            current_q = env.data.qpos[:7].copy().astype(np.float32)
            payload_pos = env.data.body('prefab').xpos.copy()
            ee_pos = env._get_ee_pos()
            trajectory.append({
                "payload": payload_pos.copy(),
                "ee": ee_pos.copy(),
                "phase": current_phase,
            })
            
            # 阶段切换检查
            state = get_phase_state(env, obs)
            if current_phase == "lift" and check_phase_transition("lift", state, config):
                phase_success["lift"] = True
                current_phase = "cruise"
                rstate = REWARD_STATES[current_phase]()
                if hasattr(rstate, 'total_steps_global'):
                    rstate.total_steps_global = 10_000_000
                if hasattr(rstate, 'current_success_radius'):
                    _rcfg_p = config.get("cruise_rl", {}).get("reward", {})
                    # [v3.4] pipeline 测试用训练末期半径 (0.10m) 而非 0.06m
                    rstate.current_success_radius = float(_rcfg_p.get("success_radius_end", 0.10))
                ee_ctrl.reset(env._get_ee_pos(), current_q)
                # ★ 切换到 cruise 时重置控制器
                _pl_z = float(payload_pos[2])
                _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
                z_pid.reset(_pl_z, _pl_yaw)
                print(f"    [pipeline s{step}] Lift→Cruise ✅")
            elif current_phase == "cruise" and check_phase_transition("cruise", state, config):
                phase_success["cruise"] = True
                current_phase = "descent"
                rstate = REWARD_STATES[current_phase]()
                # ★ 测试时注入严格容差
                if hasattr(rstate, 'total_steps_global'):
                    rstate.total_steps_global = 10_000_000
                ee_ctrl.reset(env._get_ee_pos(), current_q)
                print(f"    [pipeline s{step}] Cruise→Descent ✅ "
                      f"dtf={np.linalg.norm(state['pl_xy'] - env.target_pos)*1000:.1f}mm")
            
            agent = agents.get(current_phase)
            
            if agent is None:  # 专家模式
                delta_q = expert.compute_delta_q_target(obs, current_q)
            else:
                p_obs, prev_tilt, prev_yaw = build_phase_obs(
                    current_phase, obs, env, start_xy, target_xy,
                    prev_tilt, prev_yaw)
                norm_obs = agent.normalize_obs(p_obs, update=False)
                
                result = agent.act(norm_obs, deterministic=deterministic)
                action = result[0] if isinstance(result, tuple) else result
                
                real_ee = env._get_ee_pos()
                if current_phase == "cruise":
                    # ★ Z/Yaw PID
                    _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                    _pl_euler = R.from_matrix(_pl_mat).as_euler('xyz')
                    _dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
                    _pl_vz = float(env.data.qvel[_dof_idx + 2])
                    _pl_yaw = float(_pl_euler[2])
                    _pl_yaw_rate = float(env.data.qvel[_dof_idx + 5]) if _dof_idx + 5 < len(env.data.qvel) else 0.0
                    _z_corr, _tgt_yaw, _falling = z_pid.compute(
                        float(payload_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                    if _falling:
                        term_reason = f"ground_collision:z={payload_pos[2]:.3f}"
                        break
                    # ★ 底层防摆 + RL 残差
                    _pl_vel_full = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                    _ee_vel = getattr(env, '_ee_vel_cache', np.zeros(3))
                    if swing_d is not None:
                        swing_d.compute(payload_pos, real_ee, _pl_vel_full, _ee_vel)
                    acc_3d = np.array([action[0], action[1], 0.0])
                    z_lock = float(config["cruise_rl"]["z_lock_height"])
                    delta_q = ee_ctrl.compute_delta_q(
                        acc_3d, current_q, real_ee,
                        lock_z=True, z_lock_height=z_lock,
                        z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                        base_acc_xy=None, residual_mode=False)
                else:
                    # [v3.4 FIX] Descent pipeline: PID base + RL residual
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
            
            next_obs, _, env_term, env_trunc, env_info = env.step(delta_q)
            
            reward, r_done, r_success, r_info = REWARD_FNS[current_phase](
                env, next_obs, config, rstate)
            
            ep_reward += reward
            phase_rewards[current_phase] += reward
            phase_steps[current_phase] += 1
            ep_steps += 1
            
            if r_success and current_phase == "descent":
                # [v3.4] 仅物理插入成功才算真正完成
                phys_ok, phys_detail = check_physical_insertion(env, config)
                mark_str = "✅ 物理插入成功" if phys_ok else "⚠ 课程容差达标但未插入"
                print(f"    [{mark_str}] {phys_detail}")
                if phys_ok:
                    final_success = True
                    phase_success["descent"] = True
                    term_reason = r_info.get("termination", "insertion_success")
                    obs = next_obs
                    break
                # 未满足物理标准: 继续执行, 不终止 episode

            # [v3.4] Descent 阶段最多 200 步 (防止长时间接触走成)
            if current_phase == "descent" and phase_steps["descent"] >= 200:
                term_reason = "descent_timeout_200"
                break

            # ★ pipeline descent 阶段调试输出
            if current_phase == "descent" and phase_steps["descent"] % 20 == 0:
                pl_pos_dbg = env.data.body('prefab').xpos
                dtf_dbg = float(np.linalg.norm(pl_pos_dbg[:2] - env.target_pos))
                pl_mat_dbg = env.data.body('prefab').xmat.reshape(3, 3)
                pl_euler_dbg = R.from_matrix(pl_mat_dbg).as_euler('xyz')
                tilt_dbg = float(np.sqrt(pl_euler_dbg[0]**2 + pl_euler_dbg[1]**2))
                hold_dbg = getattr(rstate, 'insertion_hold_counter', 0)
                print(f"    [descent s{phase_steps['descent']:3d}] "
                      f"z={pl_pos_dbg[2]*1000:.1f}mm dtf={dtf_dbg*1000:.1f}mm "
                      f"tilt={tilt_dbg:.3f} hold={hold_dbg}")
            
            if r_info.get("termination"):
                term_reason = r_info["termination"]
            
            obs = next_obs
            
            if r_done or env_info.get("nan_detected", False):
                break
        
        ep_count += 1
        phys_success = False
        if final_success:
            phys_success, _ = check_physical_insertion(env, config)
        results.append({
            "reward": ep_reward,
            "steps": ep_steps,
            "success": final_success,
            "physical_success": phys_success,
            "termination": term_reason or "timeout",
            "phase_rewards": phase_rewards.copy(),
            "phase_steps": phase_steps.copy(),
            "phase_success": phase_success.copy(),
            "trajectory": trajectory,
        })
        
        mark = "✅" if final_success else "❌"
        term_short = (term_reason or "timeout").split(":")[0]
        phases_str = " → ".join([
            f"{'✅' if phase_success[p] else '❌'}{p[0].upper()}"
            for p in ["lift", "cruise", "descent"]])
        print(f"  Ep {ep_count:3d} {mark} | R:{ep_reward:7.2f} | "
              f"Steps:{ep_steps:3d} | {phases_str} | {term_short}")
    
    return results


# ==============================================================================
# 结果汇总
# ==============================================================================

def print_summary(results, mode_name):
    """打印测试结果汇总。"""
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

    print(f"\n{'='*55}")
    print(f"  [{mode_name}] 结果汇总")
    print(f"{'='*55}")
    # [v3.4] 测试指标:
    #   descent: 物理插入成功率 (socket 物理容差)
    #   cruise:  pipeline 切换成功率 (phase_transition 条件)
    #   lift:    课程成功率 (提升到巡航高度)
    print(f"  成功率: {phys_sr*100:.1f}%")
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
    print()


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="三阶段 RL 测试")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["lift", "cruise", "descent", "pipeline"])
    parser.add_argument("--algo", type=str, default="expert",
                        choices=["ppo", "sac", "expert"])
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save-paths", action="store_true")
    parser.add_argument("--save-dir", type=str, default="test_results")
    
    # 流水线模式的额外参数
    parser.add_argument("--lift-ckpt", type=str, default=None)
    parser.add_argument("--cruise-ckpt", type=str, default=None)
    parser.add_argument("--descent-ckpt", type=str, default=None)
    parser.add_argument("--lift-algo", type=str, default="ppo",
                        choices=["ppo", "sac", "expert"])
    parser.add_argument("--cruise-algo", type=str, default="ppo",
                        choices=["ppo", "sac", "expert"])
    parser.add_argument("--descent-algo", type=str, default="ppo",
                        choices=["ppo", "sac", "expert"])
    
    args = parser.parse_args()
    config = build_config(args)
    config["scene"]["seed"] = args.seed
    
    if args.obstacles is not None:
        test_n_obs = min(int(args.obstacles), config["scene"]["n_obstacles"])
    else:
        test_n_obs = config["scene"]["n_obstacles"]
    
    env = CableRobotEnvWithObstacles(config=config)
    env.set_curriculum_n_obstacles(test_n_obs)
    expert = JointSpaceExpert(config, env.ik_solver)
    ee_ctrl = EEAccController(config, env.ik_solver)
    
    if args.phase == "pipeline":
        # 流水线模式
        agents = {}
        for p, p_algo, p_ckpt in [
            ("lift", args.lift_algo, args.lift_ckpt),
            ("cruise", args.cruise_algo, args.cruise_ckpt),
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
        
        print(f"\n{'='*60}")
        print(f"  流水线测试 | {args.episodes} 回合 | n_obs={test_n_obs}")
        print(f"{'='*60}")
        
        results = test_pipeline(env, agents, expert, ee_ctrl, config,
                                n_episodes=args.episodes)
        print_summary(results, mode_name)
    
    else:
        # 单阶段测试
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
                                     n_episodes=args.episodes)
        print_summary(results, mode_name)
    
    # 保存轨迹
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
            save_data = dict(
                payload=pl_arr, ee=ee_arr,
                tilt=tilt_arr, yaw=yaw_arr, swing_vel=swing_arr,
                success=r["success"], reward=r["reward"],
                termination=str(r.get("termination", "")),
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