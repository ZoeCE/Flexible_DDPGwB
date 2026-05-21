# ==============================================================================
# vec_env.py — [v11 Path 3] Vectorized environment wrappers (CPU-parallel)
#
# 用 Python multiprocessing 并行运行多个 CableRobotEnv 实例, 每个子进程独立持有:
#   env + JointSpaceExpert + EEAccController + (CruiseZYawPID + SwingDampingController)
#
# 设计理念:
# - 不重写 MuJoCo 环境 (避免 MJX/Isaac Gym 大改)
# - 每个子进程通信: action_vec 进, (obs, reward, done, info) 出
# - 主进程: PPO/SAC 推理 + 优化
# - 默认 n_envs=1 用 DummyVecEnv (单进程, 向后兼容)
#
# 来源: Stable-Baselines3 SubprocVecEnv 设计 (Hill et al. 2018)
# 关于并行 PPO/SAC: Crowder et al. 2024 (PPO-HER), Schulman 2017 (PPO 原论文 N_envs)
# ==============================================================================

import os
import sys
import time
import multiprocessing as mp
from multiprocessing import Pipe
import numpy as np
import traceback


# ==============================================================================
# Worker 进程: 在子进程中运行一个 env + 所有 base controllers
# ==============================================================================

def _vec_env_worker(remote, parent_remote, env_fn_pickle, worker_id, init_lock=None):
    """子进程主循环.

    Protocol (来自主进程的命令):
        ('reset', None)                 → 重置 env, 返回 (obs, pp)
        ('reset_for_descent', cur_init) → descent 用 reset_for_descent_with_cur
        ('step', dq)                    → env.step(dq), 返回 (obs, reward, done, info)
        ('rl_step', payload)            → [v11 Path 3] 高级: 给 RL action,
                                          worker 内部计算 base controller + reward
        ('set_perturbations', pert)     → 设置噪声/风扰
        ('get_state', None)             → 返回 internal env state snapshot
        ('close', None)                 → 关闭, 退出循环

    `rl_step` payload (dict):
        {
            'phase':        'lift' / 'cruise' / 'descent',
            'rl_action':    np.ndarray,           # RL 输出
            'obs':          np.ndarray (env_obs), # 当前 env obs (来自 main 缓存)
            'current_q':    np.ndarray (7,),     # 关节角
            'start_xy':     np.ndarray (2,) or None,
            'target_xy':    np.ndarray (2,),
            'prev_tilt':    float,
            'prev_yaw':     float,
            'rstate':       dict (compute_*_reward 用的 state),
            'act_noise':    float,
        }
    返回 dict:
        {
            'new_obs':      env_obs after step,
            'new_phase_obs': phase obs (build_phase_obs 输出),
            'reward':       float,
            'done':         bool,
            'success':      bool,
            'info':         dict,
            'new_tilt':     float,
            'new_yaw':      float,
            'pl_xy':        np.ndarray (2,),   # for HER (descent)
            'xy_align_r':   float,             # for HER reward replay
            'rstate':       dict (updated),
            'termination':  str,
        }
    """
    parent_remote.close()
    # [v11 fix] 在 worker 顶部就 import traceback, 避免后续 except 中 UnboundLocalError
    import traceback
    # 在子进程内 unpickle env_fn (cloudpickle 处理 lambda)
    try:
        import cloudpickle
        env_fn = cloudpickle.loads(env_fn_pickle)
    except ImportError:
        import pickle
        env_fn = pickle.loads(env_fn_pickle)

    # 设置 random seed 每个 worker 不同
    np.random.seed(int(time.time() * 1000) % 2**32 + worker_id * 1000)
    try:
        import random
        random.seed(int(time.time() * 1000) % 2**32 + worker_id * 1000)
    except Exception:
        pass

    # 在子进程中构造 env + controllers + reward 函数引用
    # [v11 fix] 用 init_lock 序列化: gen_rope() 写共享 XML 文件, 多个 worker
    # 同时调用会导致 XML parse error. Lock 序列化后, env.reset() 才用 tempfile.
    try:
        if init_lock is not None:
            with init_lock:
                env, controllers, config = env_fn()
        else:
            env, controllers, config = env_fn()
    except Exception as e:
        remote.send(('error', f"Worker {worker_id} init failed: {e}\n{traceback.format_exc()}"))
        remote.close()
        return

    expert = controllers.get("expert")
    ectl   = controllers.get("ee_ctrl")
    z_pid  = controllers.get("z_pid")
    swing_d = controllers.get("swing_d")
    # [v11 fix] phase 从 controllers 取出, 用于 reset 调用 reset_for_phase
    worker_phase = controllers.get("phase", "lift")

    # Import worker-side helpers (lazy, 在 subprocess 内部)
    from train_phase import (build_phase_obs, _add_act_noise,
                             _apply_descent_pid_residual, reset_for_phase,
                             _advance_expert_to_nearest_wp as _advance_expert_to_nearest_wp_local,
                             _truncate_path_for_lift as _truncate_path_for_lift_local)
    from phase_reward import (compute_lift_reward, compute_cruise_reward,
                              compute_descent_reward, RewardComponentTracker)
    from stability_metrics import StabilityMetrics
    from scipy.spatial.transform import Rotation as R
    REWARD_FNS = {
        "lift":    compute_lift_reward,
        "cruise":  compute_cruise_reward,
        "descent": compute_descent_reward,
    }
    # [v12.6] worker-local stability tracker (持久跨 step, 在 episode done 时 summary 后 reset)
    stab = StabilityMetrics()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'reset':
                # [v11 fix] env.reset() 只返回 obs (1 个值), 必须用 reset_for_phase
                # 才能得到 planned_path. 单进程 train_ppo 也是这么做的.
                obs, pp = reset_for_phase(env, worker_phase, config)
                # [v12.6] 新 episode 开始, 重置 stability tracker
                stab.reset_episode()
                remote.send((obs, pp))
            elif cmd == 'reset_controllers':
                # [v11 KEY FIX] 重置 expert / ee_ctrl / z_pid / swing_d.
                # 单进程 train_ppo 每个 episode 开始都会调这些; vec 模式之前完全漏掉,
                # 导致 NMPC tracker 使用旧 episode 的路径, descent PID 也用旧目标点,
                # 所有 base action 都是垃圾, RL 残差无法补偿 → SR=0.
                # data = {'obs', 'cq', 'planned_path', 'pl_pos_xy_z'}
                payload = data
                obs_r   = np.asarray(payload['obs'], np.float32)
                cq_r    = np.asarray(payload['cq'], np.float32)
                pp_r    = payload.get('planned_path')
                try:
                    expert.reset(obs_r, cq_r, env=env)
                    if pp_r is not None:
                        # [v12.2] lift 时只把"lift 段"喂给 tracker (防 look-ahead 跨段)
                        if worker_phase == "lift":
                            _pp_for_expert = _truncate_path_for_lift_local(pp_r, config)
                        else:
                            _pp_for_expert = pp_r
                        expert.set_path(_pp_for_expert)
                        if worker_phase != "lift":
                            plp = env.data.body('prefab').xpos.copy()
                            _advance_expert_to_nearest_wp_local(expert, pp_r, plp)
                    ectl.reset(env._get_ee_pos(), cq_r)
                    if z_pid is not None:
                        _pl_z   = float(env.data.body('prefab').xpos[2])
                        _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
                        _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
                        z_pid.reset(_pl_z, _pl_yaw)
                    remote.send('ok')
                except Exception as e:
                    import traceback
                    remote.send(('error', f"reset_controllers failed: {e}\n{traceback.format_exc()}"))
            elif cmd == 'reset_for_descent':
                # descent override: data = dict with xy_range/vel_range/tilt_range
                from train_phase import reset_for_phase
                kw = {}
                if data is not None:
                    if "xy_range" in data: kw["override_init_xy_range"] = data["xy_range"]
                    if "vel_range" in data: kw["override_init_vel_range"] = data["vel_range"]
                    if "tilt_range" in data: kw["override_init_tilt_range"] = data["tilt_range"]
                obs, pp = reset_for_phase(env, "descent", config, **kw)
                # [v12.6] 新 episode 开始, 重置 stability tracker
                stab.reset_episode()
                remote.send((obs, pp))
            elif cmd == 'step':
                dq = data
                no2, r_dummy, dn_dummy, _, ei = env.step(dq)
                remote.send((no2, r_dummy, dn_dummy, ei))
            elif cmd == 'rl_step':
                # [v11 Path 3] 完整 RL step: 主进程提供 RL action, worker 内做 base + step + reward
                payload = data
                phase = payload['phase']
                rl_act = np.asarray(payload['rl_action'], np.float32)
                obs    = np.asarray(payload['obs'], np.float32)
                cq     = np.asarray(payload['current_q'], np.float32)
                sxy    = payload.get('start_xy')
                txy    = np.asarray(payload['target_xy'], np.float32)
                pt     = float(payload.get('prev_tilt', 0.0))
                py     = float(payload.get('prev_yaw', 0.0))
                rstate = payload['rstate']
                act_noise = float(payload.get('act_noise', 0.0))

                ree = env._get_ee_pos()
                term_reason = "running"
                done = False; reward = 0.0; success = False; ei = {}
                tracker = RewardComponentTracker(phase)
                # 这里我们不返回完整 tracker, 只返回 xy_align_r (HER 用)

                if phase == "cruise":
                    # [v13.0 合并 lift] 双模: payload 低空时走 lift 模式, 高空走 cruise 模式
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

                    if not _is_lift_phase:
                        _z_corr, _tgt_yaw, _falling = z_pid.compute(
                            float(_pl_pos[2]), _pl_vz, _pl_yaw, _pl_yaw_rate)
                        if _falling:
                            try:
                                _stab_sum = stab.summary()
                            except Exception:
                                _stab_sum = None
                            remote.send({
                                'falling': True, 'reward': -5.0, 'done': True,
                                'success': False, 'termination': 'falling',
                                'new_obs': obs, 'new_phase_obs': None,
                                'new_tilt': pt, 'new_yaw': py,
                                'pl_xy': _pl_pos[:2].copy(), 'xy_align_r': 0.0,
                                'rstate': rstate,
                                'stab_summary': _stab_sum,
                            })
                            continue
                        if swing_d is not None:
                            _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
                            _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
                            swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)
                    else:
                        _z_corr, _tgt_yaw = 0.0, 0.0

                    _rl_arr = np.asarray(rl_act, np.float32)

                    if _is_lift_phase:
                        # Lift 模式: expert.compute_delta_q_target + 3D residual
                        _rm_xy = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                        _rm_z  = float(config["cruise_rl"].get("residual_acc_max_z_rl",  0.10))
                        _res3 = np.array([
                            float(np.clip(_rl_arr[0], -_rm_xy, _rm_xy)),
                            float(np.clip(_rl_arr[1], -_rm_xy, _rm_xy)),
                            float(np.clip(_rl_arr[2], -_rm_z,  _rm_z)) if len(_rl_arr) > 2 else 0.0,
                        ], np.float64)
                        dq = expert.compute_delta_q_target(obs, cq, residual_acc=_res3)
                    else:
                        # Cruise 模式: NMPC + xy 残差 + lock_z
                        _use_nmpc = bool(config.get("cruise_rl", {}).get("use_nmpc_base", False))
                        if _use_nmpc:
                            try:
                                _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                                _base = np.array([float(_a4[0]), float(_a4[1])], np.float32)
                            except Exception:
                                _base = np.zeros(2, np.float32)
                            _res_max = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
                            _rl_clip = np.clip(_rl_arr[:2], -_res_max, _res_max)
                            _comb = _base + _rl_clip
                            _amax = float(config["cruise_rl"].get("residual_acc_max_xy", 0.60))
                            _cn = float(np.linalg.norm(_comb))
                            if _cn > _amax: _comb = _comb / _cn * _amax
                            a3 = np.array([_comb[0], _comb[1], 0.0])
                        else:
                            a3 = np.array([_rl_arr[0], _rl_arr[1], 0.0])
                        dq = ectl.compute_delta_q(
                            a3, cq, ree, lock_z=True,
                            z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
                            z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
                            base_acc_xy=None, residual_mode=False)

                    dq = _add_act_noise(dq, act_noise)
                    no2, _, _, _, ei = env.step(dq)
                    # [v13.0] rl_action 传完整 3D
                    reward, done, success, ri = compute_cruise_reward(
                        env, no2, config, rstate, tracker=tracker,
                        rl_action=_rl_arr)
                    term_reason = ri.get('termination', 'running')

                elif phase == "descent":
                    dq, _pid_dq = _apply_descent_pid_residual(
                        expert, rl_act, obs, env, config, cq)
                    dq = _add_act_noise(dq, act_noise)
                    no2, _, _, _, ei = env.step(dq)
                    # [v11.3] 传 rl_action
                    reward, done, success, ri = compute_descent_reward(
                        env, no2, config, rstate, tracker=tracker,
                        rl_action=rl_act)
                    term_reason = ri.get('termination', 'running')

                else:  # lift
                    _use_lift_nmpc = bool(config.get("lift_rl", {}).get("use_nmpc_base", False))
                    if _use_lift_nmpc:
                        # [v12 fix] 用 expert.compute_delta_q_target(..., residual_acc=...)
                        # 与 test_phase expert-only 路径完全一致.
                        _rm_xy = float(config["lift_rl"].get("residual_acc_max_xy", 0.08))
                        _rm_z  = float(config["lift_rl"].get("residual_acc_max_z",  0.10))
                        _res3 = np.array([
                            float(np.clip(rl_act[0], -_rm_xy, _rm_xy)),
                            float(np.clip(rl_act[1], -_rm_xy, _rm_xy)),
                            float(np.clip(rl_act[2], -_rm_z,  _rm_z)),
                        ], np.float64)
                        dq = expert.compute_delta_q_target(obs, cq, residual_acc=_res3)
                    else:
                        dq = ectl.compute_delta_q(rl_act, cq, ree)
                    dq = _add_act_noise(dq, act_noise)
                    no2, _, _, _, ei = env.step(dq)
                    # [v12] 传 rl_action
                    reward, done, success, ri = compute_lift_reward(
                        env, no2, config, rstate, tracker=tracker,
                        rl_action=rl_act)
                    term_reason = ri.get('termination', 'running')

                # 构造 phase obs (供下一步 RL inference)
                new_phase_obs, new_tilt, new_yaw = build_phase_obs(
                    phase, no2, env, sxy, txy, pt, py)
                pl_xy_now = env.data.body('prefab').xpos[:2].copy().astype(np.float32)
                xy_align_r = float(tracker.get_last_step_value("xy_align_reward")) \
                    if hasattr(tracker, 'get_last_step_value') else 0.0

                if ei.get('nan_detected', False):
                    done = True; term_reason = 'nan_detected'

                # [v12.6] 更新 stability tracker, 在 episode done 时一起返回 summary
                try:
                    stab.update_step(no2, config, env=env, rl_action=rl_act)
                except Exception:
                    pass
                stab_summary = stab.summary() if done else None

                remote.send({
                    'falling': False,
                    'new_obs': no2, 'new_phase_obs': new_phase_obs,
                    'reward': float(reward), 'done': bool(done), 'success': bool(success),
                    'info': ei, 'new_tilt': float(new_tilt), 'new_yaw': float(new_yaw),
                    'pl_xy': pl_xy_now, 'xy_align_r': xy_align_r,
                    'rstate': rstate, 'termination': term_reason,
                    'stab_summary': stab_summary,    # [v12.6] None or dict
                })

            elif cmd == 'build_phase_obs':
                # data = (phase, env_obs, start_xy, target_xy, prev_tilt, prev_yaw)
                from train_phase import build_phase_obs
                phase_b, eo, sxy_b, txy_b, pt_b, py_b = data
                po, t, y = build_phase_obs(phase_b, eo, env, sxy_b, txy_b, pt_b, py_b)
                remote.send((po, t, y))

            elif cmd == 'get_pl_pos':
                pp = env.data.body('prefab').xpos.copy()
                remote.send(pp)
            elif cmd == 'get_qpos':
                remote.send(env.data.qpos[:7].copy().astype(np.float32))
            elif cmd == 'get_ee_pos':
                remote.send(env._get_ee_pos())
            elif cmd == 'set_force_noise':
                env.set_force_noise(data)
                remote.send('ok')
            elif cmd == 'set_wind_force':
                f, d = data
                if hasattr(env, 'set_wind_force'):
                    env.set_wind_force(f, d)
                remote.send('ok')
            elif cmd == 'set_wind_curriculum':
                if hasattr(env, 'set_wind_curriculum'):
                    env.set_wind_curriculum(data)
                remote.send('ok')
            elif cmd == 'env_attr':
                # 取 env 上的属性 (例如 target_pos)
                attr_name = data
                val = getattr(env, attr_name, None)
                # numpy 数组可序列化
                if val is not None and hasattr(val, 'copy'):
                    val = val.copy()
                remote.send(val)
            elif cmd == 'close':
                env.close()
                remote.send('closed')
                break
            else:
                remote.send(('error', f"Unknown cmd: {cmd}"))
    except Exception as e:
        remote.send(('error', f"Worker {worker_id} crashed: {e}\n{traceback.format_exc()}"))
    finally:
        try: env.close()
        except Exception: pass
        remote.close()


# ==============================================================================
# DummyVecEnv: 单进程版本 (默认, 向后兼容)
# ==============================================================================

class DummyVecEnv:
    """同步, 单进程, n_envs=1 时使用。接口与 SubprocVecEnv 对齐。

    使用模式:
        vec = DummyVecEnv([lambda: make_env_and_controllers(config)])
        obs, pp = vec.reset(idx=0)
        ...

    单进程版主要为了让训练代码统一接口, 实际计算和不开并行一致.
    """
    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.envs = []
        self.controllers_list = []
        self.configs = []
        for fn in env_fns:
            env, ctrls, cfg = fn()
            self.envs.append(env)
            self.controllers_list.append(ctrls)
            self.configs.append(cfg)
        self.closed = False

    def reset(self, idx):
        # [v11 fix] env.reset() 只返回 obs (1 个值); 用 reset_for_phase 取 (obs, pp)
        from train_phase import reset_for_phase
        env = self.envs[idx]; config = self.configs[idx]
        phase = self.controllers_list[idx].get("phase", "lift")
        return reset_for_phase(env, phase, config)

    def reset_controllers(self, idx, obs, cq, planned_path):
        """[v11 KEY FIX] 每个 episode 开始时重置 expert / ee_ctrl / z_pid.
        必须 reset, 否则 NMPC tracker / PID 保留上 episode 的内部状态 → SR=0.
        """
        from train_phase import (_advance_expert_to_nearest_wp,
                                 _truncate_path_for_lift)
        from scipy.spatial.transform import Rotation as R
        env = self.envs[idx]; ctrls = self.controllers_list[idx]
        phase = ctrls.get("phase", "lift")
        expert = ctrls.get("expert"); ectl = ctrls.get("ee_ctrl")
        z_pid = ctrls.get("z_pid")
        expert.reset(obs, cq, env=env)
        if planned_path is not None:
            # [v12.2] lift 时只把"lift 段"喂给 tracker
            _pp_for_expert = (_truncate_path_for_lift(planned_path, self.configs[idx])
                              if phase == "lift" else planned_path)
            expert.set_path(_pp_for_expert)
            if phase != "lift":
                plp = env.data.body('prefab').xpos.copy()
                _advance_expert_to_nearest_wp(expert, planned_path, plp)
        ectl.reset(env._get_ee_pos(), cq)
        if z_pid is not None:
            _pl_z   = float(env.data.body('prefab').xpos[2])
            _pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
            _pl_yaw = float(R.from_matrix(_pl_mat).as_euler('xyz')[2])
            z_pid.reset(_pl_z, _pl_yaw)
        return 'ok'

    def reset_for_descent(self, idx, cur_init=None):
        """Descent 用 reset_for_phase with overrides."""
        from train_phase import reset_for_phase
        env = self.envs[idx]; config = self.configs[idx]
        kw = {}
        if cur_init is not None:
            if "xy_range" in cur_init: kw["override_init_xy_range"] = cur_init["xy_range"]
            if "vel_range" in cur_init: kw["override_init_vel_range"] = cur_init["vel_range"]
            if "tilt_range" in cur_init: kw["override_init_tilt_range"] = cur_init["tilt_range"]
        return reset_for_phase(env, "descent", config, **kw)

    def step(self, idx, dq):
        return self.envs[idx].step(dq)

    def rl_step(self, idx, payload):
        """[v11 Path 3] 完整 RL step (DummyVecEnv 内联版, 单进程直接运行).

        与 SubprocVecEnv worker 中 'rl_step' 命令逻辑完全一致.
        """
        return _run_rl_step_inline(self.envs[idx], self.controllers_list[idx],
                                   self.configs[idx], payload)

    def get_pl_pos(self, idx):
        return self.envs[idx].data.body('prefab').xpos.copy()

    def get_qpos(self, idx):
        return self.envs[idx].data.qpos[:7].copy().astype(np.float32)

    def get_ee_pos(self, idx):
        return self.envs[idx]._get_ee_pos()

    def set_force_noise(self, idx, val):
        self.envs[idx].set_force_noise(val)

    def set_wind_force(self, idx, f, d):
        if hasattr(self.envs[idx], 'set_wind_force'):
            self.envs[idx].set_wind_force(f, d)

    def set_wind_curriculum(self, idx, w):
        if hasattr(self.envs[idx], 'set_wind_curriculum'):
            self.envs[idx].set_wind_curriculum(w)

    def env_attr(self, idx, attr_name):
        val = getattr(self.envs[idx], attr_name, None)
        if val is not None and hasattr(val, 'copy'):
            val = val.copy()
        return val

    def build_phase_obs_remote(self, idx, phase, env_obs, start_xy,
                               target_xy, prev_tilt, prev_yaw):
        """主进程调用 build_phase_obs (DummyVecEnv 直接调本地 env)."""
        from train_phase import build_phase_obs
        return build_phase_obs(phase, env_obs, self.envs[idx], start_xy,
                               target_xy, prev_tilt, prev_yaw)

    def get_env(self, idx):
        return self.envs[idx]

    def get_controllers(self, idx):
        return self.controllers_list[idx]

    def close(self):
        if self.closed: return
        for env in self.envs:
            try: env.close()
            except Exception: pass
        self.closed = True


def _run_rl_step_inline(env, controllers, config, payload):
    """共享逻辑: DummyVecEnv.rl_step 直接调用; SubprocVecEnv worker 也内联同样的代码."""
    from train_phase import (build_phase_obs, _add_act_noise,
                             _apply_descent_pid_residual)
    from phase_reward import (compute_lift_reward, compute_cruise_reward,
                              compute_descent_reward, RewardComponentTracker)
    from scipy.spatial.transform import Rotation as R

    expert  = controllers.get("expert")
    ectl    = controllers.get("ee_ctrl")
    z_pid   = controllers.get("z_pid")
    swing_d = controllers.get("swing_d")

    phase = payload['phase']
    rl_act = np.asarray(payload['rl_action'], np.float32)
    obs    = np.asarray(payload['obs'], np.float32)
    cq     = np.asarray(payload['current_q'], np.float32)
    sxy    = payload.get('start_xy')
    txy    = np.asarray(payload['target_xy'], np.float32)
    pt     = float(payload.get('prev_tilt', 0.0))
    py     = float(payload.get('prev_yaw', 0.0))
    rstate = payload['rstate']
    act_noise = float(payload.get('act_noise', 0.0))

    ree = env._get_ee_pos()
    tracker = RewardComponentTracker(phase)

    if phase == "cruise":
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
            return {
                'falling': True, 'reward': -5.0, 'done': True,
                'success': False, 'termination': 'falling',
                'new_obs': obs, 'new_phase_obs': None,
                'new_tilt': pt, 'new_yaw': py,
                'pl_xy': _pl_pos[:2].copy().astype(np.float32),
                'xy_align_r': 0.0, 'rstate': rstate, 'info': {},
            }
        if swing_d is not None:
            _pl_vel  = env.data.qvel[_dof_idx:_dof_idx+3].copy()
            _ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
            swing_d.compute(_pl_pos, ree, _pl_vel, _ee_vel)
        _use_nmpc = bool(config.get("cruise_rl", {}).get("use_nmpc_base", False))
        if _use_nmpc:
            try:
                _a4 = expert.tracker.compute_ee_acceleration(obs, target_yaw=_tgt_yaw)
                _base = np.array([float(_a4[0]), float(_a4[1])], np.float32)
            except Exception:
                _base = np.zeros(2, np.float32)
            _res_max = float(config["cruise_rl"].get("residual_acc_max_xy_rl", 0.08))
            _rl_clip = np.clip(rl_act[:2], -_res_max, _res_max)
            _comb = _base + _rl_clip
            _amax = float(config["cruise_rl"].get("residual_acc_max_xy", 0.60))
            _cn = float(np.linalg.norm(_comb))
            if _cn > _amax: _comb = _comb / _cn * _amax
            a3 = np.array([_comb[0], _comb[1], 0.0])
        else:
            a3 = np.array([rl_act[0], rl_act[1], 0.0])
        dq = ectl.compute_delta_q(
            a3, cq, ree, lock_z=True,
            z_lock_height=float(config["cruise_rl"]["z_lock_height"]),
            z_pid_correction=_z_corr, target_yaw=_tgt_yaw,
            base_acc_xy=None, residual_mode=False)
        dq = _add_act_noise(dq, act_noise)
        no2, _, _, _, ei = env.step(dq)
        # [v11.2] rl_action 用于 action_magnitude_penalty + smoothness
        reward, done, success, ri = compute_cruise_reward(
            env, no2, config, rstate, tracker=tracker, rl_action=rl_act[:2])
    elif phase == "descent":
        dq, _pid_dq = _apply_descent_pid_residual(
            expert, rl_act, obs, env, config, cq)
        dq = _add_act_noise(dq, act_noise)
        no2, _, _, _, ei = env.step(dq)
        # [v11.3] 传 rl_action
        reward, done, success, ri = compute_descent_reward(
            env, no2, config, rstate, tracker=tracker, rl_action=rl_act)
    else:  # lift
        _use_lift_nmpc = bool(config.get("lift_rl", {}).get("use_nmpc_base", False))
        if _use_lift_nmpc:
            # [v12 fix] 用 expert.compute_delta_q_target(..., residual_acc=...)
            # 与 test_phase expert-only 路径完全一致.
            _rm_xy = float(config["lift_rl"].get("residual_acc_max_xy", 0.08))
            _rm_z  = float(config["lift_rl"].get("residual_acc_max_z",  0.10))
            _res3 = np.array([
                float(np.clip(rl_act[0], -_rm_xy, _rm_xy)),
                float(np.clip(rl_act[1], -_rm_xy, _rm_xy)),
                float(np.clip(rl_act[2], -_rm_z,  _rm_z)),
            ], np.float64)
            dq = expert.compute_delta_q_target(obs, cq, residual_acc=_res3)
        else:
            dq = ectl.compute_delta_q(rl_act, cq, ree)
        dq = _add_act_noise(dq, act_noise)
        no2, _, _, _, ei = env.step(dq)
        # [v12] 传 rl_action
        reward, done, success, ri = compute_lift_reward(
            env, no2, config, rstate, tracker=tracker, rl_action=rl_act)

    new_phase_obs, new_tilt, new_yaw = build_phase_obs(
        phase, no2, env, sxy, txy, pt, py)
    pl_xy_now = env.data.body('prefab').xpos[:2].copy().astype(np.float32)
    xy_align_r = float(tracker.get_last_step_value("xy_align_reward")) \
        if hasattr(tracker, 'get_last_step_value') else 0.0
    if ei.get('nan_detected', False):
        done = True

    return {
        'falling': False, 'new_obs': no2, 'new_phase_obs': new_phase_obs,
        'reward': float(reward), 'done': bool(done), 'success': bool(success),
        'info': ei, 'new_tilt': float(new_tilt), 'new_yaw': float(new_yaw),
        'pl_xy': pl_xy_now, 'xy_align_r': xy_align_r,
        'rstate': rstate, 'termination': ri.get('termination', 'running'),
    }


# ==============================================================================
# SubprocVecEnv: 多进程版本
# ==============================================================================

class SubprocVecEnv:
    """多进程 vectorized env. 每个 env 在独立子进程中运行.

    依赖: cloudpickle (用于 pickle lambda/closure env_fn)

    使用示例:
        env_fns = [lambda i=i: make_env_and_controllers(config, seed=i)
                   for i in range(n_envs)]
        vec = SubprocVecEnv(env_fns, start_method='spawn')

    注意:
        - n_envs 不应超过 CPU 核心数
        - 主进程需要在 if __name__ == '__main__' 保护下创建 (spawn/forkserver 启动方式)
        - 子进程内 env 实例完全独立, 不共享状态
    """
    def __init__(self, env_fns, start_method=None):
        self.n_envs = len(env_fns)
        if start_method is None:
            # macOS 默认 spawn, Linux 默认 fork. fork 更快但 MuJoCo 不一定线程安全
            start_method = 'spawn' if sys.platform == 'darwin' else 'forkserver'
        ctx = mp.get_context(start_method)

        try:
            import cloudpickle
            pickler = cloudpickle
        except ImportError:
            import pickle
            pickler = pickle

        # [v11 fix] 全局 init lock: 序列化 env_fn 调用, 避免并发写共享 XML 文件
        # mujoco_env_new.CableRobotEnvWithObstacles.__init__ 会调 gen_rope() 写
        # /assets/iiwa14_four_cables_with_plate.xml 等共享文件; 多个 worker 同时
        # 调用会导致一个 worker 读到另一个写到一半的 XML → "XML parse error 7".
        # 用 Lock 序列化, 之后 env.reset() 用 tempfile 是 worker-local 安全的.
        self._init_lock = ctx.Lock()

        self.remotes = []
        self.processes = []
        for i, fn in enumerate(env_fns):
            parent_remote, worker_remote = Pipe()
            args = (worker_remote, parent_remote, pickler.dumps(fn), i, self._init_lock)
            p = ctx.Process(target=_vec_env_worker, args=args, daemon=True)
            p.start()
            worker_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(p)
        self.closed = False
        print(f"[VecEnv] SubprocVecEnv 启动 {self.n_envs} 个 worker (method={start_method})")

    def _check_recv(self, x, idx):
        # [v11 fix] 必须保证 x[0] 是 str 才能比较 'error', 否则 numpy array 触发
        # ValueError: The truth value of an array with more than one element is ambiguous
        if (isinstance(x, tuple) and len(x) >= 2 and
                isinstance(x[0], str) and x[0] == 'error'):
            raise RuntimeError(f"Worker {idx} error: {x[1]}")
        return x

    def reset(self, idx):
        self.remotes[idx].send(('reset', None))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def reset_controllers(self, idx, obs, cq, planned_path):
        """[v11 KEY FIX] 每个 episode 开始时重置 worker 内的 expert / ee_ctrl / z_pid.
        必须 reset, 否则 NMPC tracker / PID 保留上 episode 状态 → 所有 base action 错误.
        """
        self.remotes[idx].send(('reset_controllers', {
            'obs': obs, 'cq': cq, 'planned_path': planned_path,
        }))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def reset_for_descent(self, idx, cur_init=None):
        self.remotes[idx].send(('reset_for_descent', cur_init))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def step(self, idx, dq):
        self.remotes[idx].send(('step', dq))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def rl_step(self, idx, payload):
        """[v11 Path 3] 完整 RL step via subprocess."""
        self.remotes[idx].send(('rl_step', payload))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def get_pl_pos(self, idx):
        self.remotes[idx].send(('get_pl_pos', None))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def get_qpos(self, idx):
        self.remotes[idx].send(('get_qpos', None))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def get_ee_pos(self, idx):
        self.remotes[idx].send(('get_ee_pos', None))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def set_force_noise(self, idx, val):
        self.remotes[idx].send(('set_force_noise', val))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def set_wind_force(self, idx, f, d):
        self.remotes[idx].send(('set_wind_force', (f, d)))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def set_wind_curriculum(self, idx, w):
        self.remotes[idx].send(('set_wind_curriculum', w))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def env_attr(self, idx, attr_name):
        self.remotes[idx].send(('env_attr', attr_name))
        return self._check_recv(self.remotes[idx].recv(), idx)

    def build_phase_obs_remote(self, idx, phase, env_obs, start_xy,
                               target_xy, prev_tilt, prev_yaw):
        self.remotes[idx].send(('build_phase_obs',
                                (phase, env_obs, start_xy, target_xy,
                                 prev_tilt, prev_yaw)))
        return self._check_recv(self.remotes[idx].recv(), idx)

    # ── 兼容接口: 不能直接返回 env / controllers (它们在子进程) ─────────────
    # 主进程不能持有 worker 的 env. 但目前 train 代码在 step 之后会读
    # env.data.body / env.target_pos 等. SubprocVecEnv 模式下需要换用
    # env_attr/get_pl_pos 等远程调用. 完整 PPO 训练循环迁移到 vec 模式需要
    # 额外重构 (见 README.md "Path 3" 部分).

    def get_env(self, idx):
        raise NotImplementedError(
            "SubprocVecEnv 模式下不能直接访问 env 实例. "
            "使用 env_attr/get_pl_pos/get_qpos 等远程调用.")

    def get_controllers(self, idx):
        raise NotImplementedError(
            "SubprocVecEnv 模式下 controllers 在 worker 进程内, 主进程不可访问. "
            "训练循环需将 base controller (NMPC/PID) 计算也移至 worker.")

    def close(self):
        if self.closed: return
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except Exception: pass
        for p in self.processes:
            try: p.join(timeout=5)
            except Exception: pass
            if p.is_alive():
                try: p.terminate()
                except Exception: pass
        self.closed = True
        print(f"[VecEnv] SubprocVecEnv 关闭 {self.n_envs} 个 worker")


# ==============================================================================
# Factory: 根据 config.train.n_envs 选择 Dummy / Subproc
# ==============================================================================

def make_vec_env(make_one_fn, n_envs=1, start_method=None):
    """工厂函数, 根据 n_envs 选择实现.

    参数:
        make_one_fn: 接受 worker_id, 返回 (env, controllers_dict, config) 的函数
        n_envs:     并行环境数 (1 → DummyVecEnv, 否则 SubprocVecEnv)
        start_method: 'spawn' / 'fork' / 'forkserver'

    使用:
        def make_one(wid):
            from config import DEFAULT_CONFIG
            from mujoco_env_new import CableRobotEnvWithObstacles
            from controller import JointSpaceExpert
            ...
            env = CableRobotEnvWithObstacles(config=DEFAULT_CONFIG)
            ctrls = {
                "expert":  JointSpaceExpert(DEFAULT_CONFIG, env.ik_solver),
                "ee_ctrl": EEAccController(DEFAULT_CONFIG, env.ik_solver),
                ...
            }
            return env, ctrls, DEFAULT_CONFIG

        vec = make_vec_env(make_one, n_envs=4)
    """
    env_fns = [lambda wid=i: make_one_fn(wid) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=start_method)