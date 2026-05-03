# ==============================================================================
# phase_reward.py — 三阶段独立奖励函数
#
# 每阶段 reward 简洁可靠:
#   Lift:    z 接近奖励 + 动能惩罚
#   Cruise:  双势能 + 动能惩罚 + 碰撞/成功
#   Descent: xy 对准 + z 下降 + 动能惩罚 + 姿态 + 成功
# ==============================================================================

import numpy as np
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# 通用安全检查
# ==============================================================================

def _check_instability(env, obs, config, grace_steps=50):
    """检查是否失稳。返回 (unstable: bool, reason: str)。"""
    cfg_logic = config.get("step_logic", {})
    current_step = getattr(env, 'current_step', 0)
    instab_grace = int(cfg_logic.get("instability_grace_steps", grace_steps))

    if current_step < instab_grace:
        return False, ""

    ee_xy = np.array([obs[0], obs[1]])
    pl_xy = np.array([obs[4], obs[5]])
    swing_xy = float(np.linalg.norm(ee_xy - pl_xy))

    pl_vxy = obs[6:8]
    payload_z = env.data.body('prefab').xpos[2]
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz = float(env.data.qvel[dof_idx + 2])
    pl_vel = float(np.linalg.norm(np.append(pl_vxy, pl_vz)))

    pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    abs_yaw = abs(float(pl_euler[2]))

    parts = []
    if swing_xy > cfg_logic.get("swing_xy_max", 0.25):
        parts.append(f"swing={swing_xy:.3f}")
    if pl_vel > cfg_logic.get("payload_vel_max", 2.0):
        parts.append(f"vel={pl_vel:.2f}")
    if tilt > cfg_logic.get("payload_tilt_max", 1.0):
        parts.append(f"tilt={tilt:.2f}")
    if abs_yaw > cfg_logic.get("payload_yaw_max", 1.2):
        parts.append(f"yaw={abs_yaw:.2f}")

    if parts:
        return True, "instability:" + ",".join(parts)
    return False, ""


def _check_collision(env, config):
    """检查 payload 是否碰撞障碍物。"""
    use_mjc = config.get("reward", {}).get("use_mujoco_contact", True)
    if use_mjc:
        hit_obs, _ = env._check_prefab_collision_with_obstacles()
        return hit_obs
    else:
        pl_xy = np.array([env.data.body('prefab').xpos[0],
                          env.data.body('prefab').xpos[1]])
        for (ox, oy, orad) in env._obstacles:
            if float(np.linalg.norm(pl_xy - np.array([ox, oy]))) < (orad + env.payload_radius):
                return True
    return False


def _get_swing_ke(env, obs):
    """计算 payload 相对 EE 的摆动动能 (xy平面)。"""
    ee_vxy = np.array([obs[2], obs[3]])
    pl_vxy = np.array([obs[6], obs[7]])
    v_rel = np.linalg.norm(pl_vxy - ee_vxy)
    mass = float(env.config.get("prefab", {}).get("mass", 1.0))
    return 0.5 * mass * v_rel ** 2


# ==============================================================================
# Phase 1: Lift Reward
# ==============================================================================

class LiftRewardState:
    def __init__(self):
        self.prev_z = None

    def reset(self):
        self.prev_z = None


def compute_lift_reward(env, obs, config, rstate):
    """
    提升阶段奖励:
      - z 接近巡航高度: 差分奖励
      - 摆动动能惩罚
      - 成功/失败终端奖励
    """
    rcfg = config["lift_rl"]["reward"]
    z_cruise = float(config["lift_rl"]["target_z_cruise"])

    payload_z = float(env.data.body('prefab').xpos[2])
    reward = 0.0
    done = False
    success = False
    info = {}

    # 安全检查
    unstable, reason = _check_instability(env, obs, config, grace_steps=30)
    if unstable:
        return float(rcfg["instability_penalty"]), True, False, {"termination": reason}

    # 坠毁检查
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz = float(env.data.qvel[dof_idx + 2])
    if payload_z < 0.03 and pl_vz < -0.5:
        return float(rcfg["crash_penalty"]), True, False, {"termination": "crash"}

    # z 接近奖励 (差分)
    if rstate.prev_z is not None:
        z_delta = payload_z - rstate.prev_z  # 越往上越正
        z_error = abs(payload_z - z_cruise)
        prev_error = abs(rstate.prev_z - z_cruise)
        reward += rcfg["z_approach_coef"] * (prev_error - z_error)
    rstate.prev_z = payload_z

    # 摆动动能惩罚
    ke = _get_swing_ke(env, obs)
    reward -= rcfg["swing_ke_coef"] * min(ke, 0.5)

    # 时间步惩罚
    reward += rcfg["step_penalty"]

    # 成功判定: payload 到达巡航高度
    phase_cfg = config["phase_transition"]
    pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    pl_vxy = obs[6:8]
    ee_vxy = obs[2:4]
    swing_vel = float(np.linalg.norm(pl_vxy - ee_vxy))

    if (payload_z >= phase_cfg["lift_to_cruise_z_threshold"] and
            tilt < phase_cfg["lift_to_cruise_tilt_max"] and
            swing_vel < phase_cfg["lift_to_cruise_swing_vel_max"]):
        reward += rcfg["success_bonus"]
        success = True
        done = True
        info["termination"] = "lift_success"

    # 超时
    max_steps = int(config["lift_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        info["termination"] = "timeout"
        reward -= 1.0

    return reward, done, success, info


# ==============================================================================
# Phase 2: Cruise Reward
# ==============================================================================

class CruiseRewardState:
    def __init__(self):
        self.prev_dist = None
        self.prev_potential = None
        self._near_given = False
        self.total_steps_global = 0
        self.initial_dist = None

    def reset(self):
        self.prev_dist = None
        self.prev_potential = None
        self._near_given = False
        self.initial_dist = None


def compute_cruise_reward(env, obs, config, rstate):
    """
    平移阶段奖励 v3 — 针对收敛问题重写:

    问题诊断:
      旧版 direction_coef 是速度投影奖励, agent 只要有一点朝目标的速度
      就持续拿奖励, 即使实际位移很小。这导致 agent 学会"缓慢漂移"策略,
      reward plateau 在 ~8 但 success_rate=0。

    v3 修改:
      1. 删除 direction_reward (罪魁祸首)
      2. 改用 PBRS 势能差分作为唯一距离驱动
      3. 加入渐进成功半径 (20cm→6cm)
      4. 降低 step_penalty, 加入 alive_bonus
    """
    rcfg = config["cruise_rl"]["reward"]
    target_xy = env.target_pos.copy()

    pl_xy = np.array([obs[4], obs[5]])
    pl_vxy = np.array([obs[6], obs[7]])
    dtf = float(np.linalg.norm(pl_xy - target_xy))
    reward = 0.0
    done = False
    success = False
    info = {}

    # 记录初始距离
    if rstate.initial_dist is None:
        rstate.initial_dist = dtf

    # 安全检查
    unstable, reason = _check_instability(env, obs, config, grace_steps=15)
    if unstable:
        return float(rcfg["instability_penalty"]), True, False, {"termination": reason}

    # payload 坠落检测
    payload_z_cruise = float(env.data.body('prefab').xpos[2])
    pid_cfg = config.get("cruise_z_pid", {})
    floor_z = float(pid_cfg.get("floor_z_threshold", 0.04))
    z_lock = float(config.get("cruise_rl", {}).get("z_lock_height", 0.25))
    if payload_z_cruise < floor_z:
        return float(rcfg.get("collision_penalty", -10.0)), True, False, {
            "termination": f"ground_collision:z={payload_z_cruise:.3f}"}
    if payload_z_cruise < z_lock * 0.4:
        return float(rcfg.get("instability_penalty", -5.0)), True, False, {
            "termination": f"payload_too_low:z={payload_z_cruise:.3f}"}

    # 碰撞检查
    if _check_collision(env, config):
        return float(rcfg["collision_penalty"]), True, False, {"termination": "collision"}

    # 基座碰撞
    if float(np.linalg.norm(pl_xy)) < 0.03:
        return float(rcfg["collision_penalty"]), True, False, {"termination": "collision_base"}

    # ═══════════════════════════════════════════════════════════════════
    # PBRS 势能差分 (唯一的距离驱动力)
    # Φ(s) = -dist,  r_shape = γ·Φ(s') - Φ(s)
    # 当 agent 靠近目标: Φ 增大, r_shape > 0
    # 当 agent 远离目标: Φ 减小, r_shape < 0
    # 理论保证不改变最优策略 (Ng et al., 1999)
    # ═══════════════════════════════════════════════════════════════════
    gamma = float(rcfg.get("pbrs_gamma", 0.99))
    k_pbrs = float(rcfg.get("pbrs_coef", 15.0))
    current_potential = -dtf
    if rstate.prev_potential is not None:
        pbrs = gamma * current_potential - rstate.prev_potential
        reward += k_pbrs * pbrs
    rstate.prev_potential = current_potential

    # ═══════════════════════════════════════════════════════════════════
    # 障碍物排斥势
    # ═══════════════════════════════════════════════════════════════════
    d0 = float(rcfg.get("obs_repulse_d0", 0.12))
    k_obs = float(rcfg.get("obs_repulse_coef", 2.0))
    for (ox, oy, orad) in env._obstacles:
        d = max(float(np.linalg.norm(pl_xy - np.array([ox, oy]))) - orad, 0.005)
        if d < d0:
            reward -= k_obs * (1.0/d - 1.0/d0) ** 2 * 0.5
    reward = max(reward, -5.0)

    # ═══════════════════════════════════════════════════════════════════
    # 摆动动能惩罚 (低权重, 不阻碍移动)
    # ═══════════════════════════════════════════════════════════════════
    ke = _get_swing_ke(env, obs)
    reward -= float(rcfg.get("swing_ke_coef", 0.3)) * min(ke, 0.5)

    # ═══════════════════════════════════════════════════════════════════
    # alive_bonus + step_penalty (净效果微正, 鼓励存活)
    # ═══════════════════════════════════════════════════════════════════
    reward += float(rcfg.get("alive_bonus", 0.01))
    reward += float(rcfg.get("step_penalty", -0.005))

    # ═══════════════════════════════════════════════════════════════════
    # 渐进成功半径: 0.20m → 0.06m
    # 前期容易拿 success_bonus, 后期逐渐收紧
    # ═══════════════════════════════════════════════════════════════════
    r_start = float(rcfg.get("success_radius_start", 0.20))
    r_end = float(rcfg.get("success_radius_end", 0.06))
    anneal = int(rcfg.get("success_radius_anneal_steps", 500_000))
    global_ts = getattr(rstate, 'total_steps_global', 0)
    frac = min(global_ts / max(anneal, 1), 1.0)
    success_radius = r_start + frac * (r_end - r_start)

    # 成功判定
    phase_cfg = config["phase_transition"]
    pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    ee_vxy = np.array([obs[2], obs[3]])
    swing_vel = float(np.linalg.norm(pl_vxy - ee_vxy))
    pl_vel = float(np.linalg.norm(pl_vxy))

    # 使用渐进半径
    tilt_max = phase_cfg["cruise_to_descent_tilt_max"]
    swing_max = phase_cfg["cruise_to_descent_swing_vel_max"]
    vel_max = phase_cfg["cruise_to_descent_payload_vel_max"]

    if (dtf < success_radius and
            tilt < tilt_max and
            swing_vel < swing_max and
            pl_vel < vel_max):
        reward += float(rcfg.get("success_bonus", 50.0))
        success = True
        done = True
        info["termination"] = f"cruise_success:r={success_radius:.3f},d={dtf:.3f}"

    # 里程碑: 进入 15cm 范围
    milestone_r = float(rcfg.get("milestone_radius", 0.15))
    if not success and dtf < milestone_r and not getattr(rstate, '_near_given', False):
        reward += float(rcfg.get("milestone_bonus", 5.0))
        rstate._near_given = True

    # 超时
    max_steps = int(config["cruise_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        info["termination"] = "timeout"
        if rstate.initial_dist is not None and rstate.initial_dist > 0.01:
            progress_frac = max(0.0, 1.0 - dtf / rstate.initial_dist)
        else:
            progress_frac = max(0.0, 1.0 - dtf / 0.3)
        reward += progress_frac * 10.0

    return reward, done, success, info


# ==============================================================================
# Phase 3: Descent Reward
# ==============================================================================

class DescentRewardState:
    def __init__(self):
        self.prev_z = None
        self.prev_dtf = None         # 用于 xy 差分奖励; None = 第一步跳过
        self.insertion_hold_counter = 0
        self.total_steps_global = 0  # 用于渐进容差; 由训练循环在每 episode 开始时注入

    def reset(self):
        # 注意: total_steps_global 不在此重置, 由训练循环维护
        self.prev_z = None
        self.prev_dtf = None
        self.insertion_hold_counter = 0


def compute_descent_reward(env, obs, config, rstate):
    """
    下降阶段奖励 v2:
      - xy 对准: 差分奖励 (越近越正) 替代绝对值惩罚, 给 agent 更清晰的梯度信号
      - z 下降: 差分奖励, align_discount 特征尺度从 0.02→0.05 (更宽松)
      - 摆动动能惩罚
      - 姿态惩罚
      - 成功/失败终端奖励
    """
    rcfg = config["descent_rl"]["reward"]
    cfg_ins = config.get("insertion", {})
    target_xy = env.target_pos.copy()
    target_pz = float(cfg_ins.get("target_payload_z", 0.10))

    pl_xy = np.array([env.data.body('prefab').xpos[0],
                      env.data.body('prefab').xpos[1]])
    payload_z = float(env.data.body('prefab').xpos[2])
    dtf = float(np.linalg.norm(pl_xy - target_xy))

    pl_mat = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    abs_yaw = abs(float(pl_euler[2]))

    reward = 0.0
    done = False
    success = False
    info = {}

    # 安全检查
    unstable, reason = _check_instability(env, obs, config, grace_steps=20)
    if unstable:
        return float(rcfg["instability_penalty"]), True, False, {"termination": reason}

    # 坠毁检查
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz = float(env.data.qvel[dof_idx + 2])
    if payload_z < 0.03 and pl_vz < -0.3:
        return float(rcfg["crash_penalty"]), True, False, {"termination": "crash"}

    # ★ xy 对准: 差分奖励 (越靠近目标越有正奖励)
    # Φ = -dtf, r_diff = prev_dtf - dtf (正 = 靠近)
    # 第一步 prev_dtf 为 None, 跳过差分项, 只给连续惩罚
    if rstate.prev_dtf is not None:
        xy_diff_reward = (rstate.prev_dtf - dtf) * rcfg["xy_align_coef"] * 10.0
        reward += xy_diff_reward
    # 连续小惩罚: 防止 agent 原地打转 (每步都承担距离代价)
    xy_penalty = -min(dtf, 0.05) * rcfg["xy_align_coef"] * 0.3
    reward += xy_penalty
    rstate.prev_dtf = dtf

    # z 下降奖励 (差分)
    if rstate.prev_z is not None:
        z_delta = rstate.prev_z - payload_z  # 向下为正
        # ★ align_discount 特征尺度 0.02→0.05: 5cm 内才激活 (原来 2cm 太严苛)
        align_discount = np.exp(-dtf / 0.05)
        reward += rcfg["z_descent_coef"] * z_delta * align_discount
    rstate.prev_z = payload_z

    # 摆动动能惩罚
    ke = _get_swing_ke(env, obs)
    reward -= rcfg["swing_ke_coef"] * min(ke, 0.25)

    # 姿态惩罚
    reward -= rcfg["tilt_coef"] * min(tilt, 0.3)

    # 时间步惩罚
    reward += rcfg["step_penalty"]

    # ★ 成功判定: 读取可能的渐进容差
    total_ts = getattr(rstate, 'total_steps_global', 0)
    xy_tol_start  = float(cfg_ins.get("xy_tolerance_train_start", cfg_ins.get("xy_tolerance", 0.015)))
    xy_tol_end    = float(cfg_ins.get("xy_tolerance_train_end", cfg_ins.get("xy_tolerance", 0.010)))
    anneal_steps  = int(cfg_ins.get("xy_tolerance_anneal_steps", 1_000_000))
    frac = min(total_ts / max(anneal_steps, 1), 1.0)
    xy_tol = xy_tol_start + frac * (xy_tol_end - xy_tol_start)

    z_tol     = float(cfg_ins.get("success_z_tolerance", 0.030))
    tilt_tol  = float(cfg_ins.get("tilt_tolerance", 0.12))
    yaw_tol   = float(cfg_ins.get("yaw_tolerance", 0.10))
    hold_steps = int(cfg_ins.get("hold_steps", 2))

    on_target = (abs(payload_z - target_pz) < z_tol and
                 dtf < xy_tol and
                 tilt < tilt_tol and abs_yaw < yaw_tol)

    if on_target:
        rstate.insertion_hold_counter += 1
    else:
        rstate.insertion_hold_counter = 0

    if rstate.insertion_hold_counter >= hold_steps:
        reward += rcfg["success_bonus"]
        success = True
        done = True
        info["termination"] = (
            f"insertion_success:z={payload_z*1000:.0f}mm,"
            f"dtf={dtf*1000:.1f}mm,tilt={tilt:.3f},yaw={abs_yaw:.3f},"
            f"xy_tol={xy_tol*1000:.1f}mm")
        return reward, done, success, info

    # 超时 → 软成功
    max_steps = int(config["descent_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        dist_frac = max(0.0, 1.0 - dtf / 0.05)
        z_frac = max(0.0, 1.0 - abs(payload_z - target_pz) / 0.15)
        pose_frac = max(0.0, 1.0 - float(np.sqrt(tilt**2 + abs_yaw**2)) / 0.3)
        combined = 0.4 * dist_frac + 0.3 * z_frac + 0.3 * pose_frac
        reward += rcfg["soft_success_bonus"] * combined
        info["termination"] = (
            f"timeout:soft={combined:.2f},"
            f"dtf={dtf*1000:.1f}mm,z={payload_z*1000:.0f}mm")

    return reward, done, success, info