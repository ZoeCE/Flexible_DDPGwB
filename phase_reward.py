# ==============================================================================
# phase_reward.py — 三阶段独立奖励函数 v3
#
# v3 主要变更:
#
# [CRUISE-REW]
#   z_dev_coef/vz_penalty_coef → 0 (PID 负责 Z, reward 不重复惩罚)
#   swing_energy_penalty 加硬上限 swing_energy_penalty_max=0.5/step
#   alive_bonus → 0 (完全去除存活奖励)
#   RL 直接输出完整 xy 加速度, reward 逻辑不变 (架构变更在 train_phase.py)
#
# [DESCENT-REW]
#   z_unconditional_frac → 0 (移除无条件下降激励)
#   rebar_sdf 严格门控: dtf < rebar_sdf_xy_gate(8mm) AND z_reached
#   near_goal_focus: 仅 z_reached AND dtf<5mm, 上限 0.10/step
#   step_penalty -0.005 → -0.02 (逼迫快速完成)
#   软成功奖励完全去除 (超时不给额外 reward)
# ==============================================================================

import numpy as np
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# 通用安全检查
# ==============================================================================

def _check_instability(env, obs, config, grace_steps=50):
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
    ee_vxy = np.array([obs[2], obs[3]])
    pl_vxy = np.array([obs[6], obs[7]])
    v_rel = np.linalg.norm(pl_vxy - ee_vxy)
    mass = float(env.config.get("prefab", {}).get("mass", 1.0))
    return 0.5 * mass * v_rel ** 2


# ==============================================================================
# Phase 1: Lift Reward (不变)
# ==============================================================================

class LiftRewardState:
    def __init__(self):
        self.prev_z = None

    def reset(self):
        self.prev_z = None


def compute_lift_reward(env, obs, config, rstate):
    rcfg = config["lift_rl"]["reward"]
    z_cruise = float(config["lift_rl"]["target_z_cruise"])

    payload_z = float(env.data.body('prefab').xpos[2])
    reward = 0.0
    done = False
    success = False
    info = {}

    unstable, reason = _check_instability(env, obs, config, grace_steps=30)
    if unstable:
        return float(rcfg["instability_penalty"]), True, False, {"termination": reason}

    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz = float(env.data.qvel[dof_idx + 2])
    if payload_z < 0.03 and pl_vz < -0.5:
        return float(rcfg["crash_penalty"]), True, False, {"termination": "crash"}

    if rstate.prev_z is not None:
        z_error = abs(payload_z - z_cruise)
        prev_error = abs(rstate.prev_z - z_cruise)
        reward += rcfg["z_approach_coef"] * (prev_error - z_error)
    rstate.prev_z = payload_z

    ke = _get_swing_ke(env, obs)
    reward -= rcfg["swing_ke_coef"] * min(ke, 0.5)
    reward += rcfg["step_penalty"]

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

    max_steps = int(config["lift_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        info["termination"] = "timeout"
        reward -= 1.0

    return reward, done, success, info


# ==============================================================================
# Phase 2: Cruise Reward [v3]
#
# 核心变更:
#   1. z_dev_coef=0, vz_penalty_coef=0 → PID 负责 Z, reward 不重复
#   2. swing_energy_penalty 加硬上限 (max 0.5/step), 不压倒 PBRS (max ~0.5/step)
#   3. alive_bonus=0 完全去除
#   4. 架构: RL 直接输出完整 xy 加速度 (SwingDamping 仅监控, 不参与控制)
# ==============================================================================

class CruiseRewardState:
    def __init__(self):
        self.prev_potential   = None
        self._near_given      = False
        self.total_steps_global = 0
        self.initial_dist     = None
        self.current_success_radius = 0.20   # [v3] 默认 0.20m (更松)
        self.prev_swing_energy = None
        # [v3 SHADOW] 影子障碍物列表: list of (ox, oy, orad)
        # 当真实障碍物为空 (n_real=0) 时使用, 参与 reward 但不参与物理碰撞
        # 由 CurriculumManager.sample_shadow_obstacles() 在每 episode 开始时注入
        self.shadow_obstacles = []   # list[(ox, oy, orad)]
        self.n_real_obstacles = 0    # 当前真实障碍物数量 (来自 env._obstacles)

    def reset(self):
        self.prev_potential    = None
        self._near_given       = False
        self.initial_dist      = None
        self.prev_swing_energy = None
        # shadow_obstacles 由外部在 reset 后注入, 不在这里清空


def _get_swing_energy_cruise(env, obs, config):
    from ee_acc_controller import compute_swing_energy
    pl_pos = env.data.body('prefab').xpos.copy()
    ee_pos = env._get_ee_pos()
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vel  = env.data.qvel[dof_idx:dof_idx+3].copy()
    ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))
    mass    = float(env.config.get("prefab",     {}).get("mass", 1.0))
    rope_L  = float(env.config.get("controller", {}).get("L",    0.5))
    energy, ke, pe, angle = compute_swing_energy(
        pl_pos, ee_pos, pl_vel, ee_vel, mass, rope_L)
    return energy, ke, pe, angle


def compute_cruise_reward(env, obs, config, rstate):
    """
    Cruise 段奖励 v3.

    架构说明 (v3):
      RL 直接输出完整 xy 加速度, SwingDampingController 不再参与控制.
      reward 层不再有残差相关逻辑, 保持纯粹的任务奖励设计.

    奖励设计原则:
      PBRS 导航信号 (20 × dist_diff) 是绝对主导.
      摆动惩罚上限 0.5/step ≤ PBRS 典型值, 不压倒导航.
      Z/Vz 完全由 PID 控制, reward 不重复惩罚.
      无存活奖励, 净步长效果为负.
    """
    rcfg      = config["cruise_rl"]["reward"]
    target_xy = env.target_pos.copy()
    pl_xy  = np.array([obs[4], obs[5]])
    pl_vxy = np.array([obs[6], obs[7]])
    dtf    = float(np.linalg.norm(pl_xy - target_xy))

    reward  = 0.0
    done    = False
    success = False
    info    = {}

    if rstate.initial_dist is None:
        rstate.initial_dist = dtf

    # ── 安全检查 ──────────────────────────────────────────────────────────────
    unstable, reason = _check_instability(env, obs, config, grace_steps=15)
    if unstable:
        return float(rcfg["instability_penalty"]), True, False, {"termination": reason}

    payload_z = float(env.data.body('prefab').xpos[2])
    z_lock    = float(config.get("cruise_rl", {}).get("z_lock_height", 0.25))
    pid_cfg   = config.get("cruise_z_pid", {})

    if payload_z < float(pid_cfg.get("floor_z_threshold", 0.04)):
        return float(rcfg.get("collision_penalty", -5.0)), True, False, {
            "termination": f"ground_collision:z={payload_z:.3f}"}
    if payload_z < z_lock * 0.20:
        return float(rcfg.get("instability_penalty", -3.0)), True, False, {
            "termination": f"payload_too_low:z={payload_z*1000:.0f}mm"}

    # ── 碰撞检查 ──────────────────────────────────────────────────────────────
    if _check_collision(env, config):
        return float(rcfg["collision_penalty"]), True, False, {"termination": "collision"}
    if float(np.linalg.norm(pl_xy)) < 0.03:
        return float(rcfg["collision_penalty"]), True, False, {"termination": "collision_base"}

    # ── 摆动能量约束 [v3: 加硬上限 max_penalty] ──────────────────────────────
    try:
        swing_energy, swing_ke, swing_pe, swing_angle = _get_swing_energy_cruise(
            env, obs, config)
    except Exception:
        swing_energy, swing_ke, swing_pe, swing_angle = 0., 0., 0., 0.

    energy_thresh       = float(rcfg.get("swing_energy_thresh",       0.15))
    energy_penalty_coef = float(rcfg.get("swing_energy_penalty_coef", 1.0))
    # [v3] 硬上限: 摆动惩罚最多 0.5/step, 保证不超过 PBRS 导航信号
    energy_penalty_max  = float(rcfg.get("swing_energy_penalty_max",  0.50))
    if swing_energy > energy_thresh:
        excess = swing_energy - energy_thresh
        swing_pen = energy_penalty_coef * (excess + 0.5 * excess ** 2)
        reward -= min(swing_pen, energy_penalty_max)
    reward = max(reward, -5.0)

    # ── PBRS 势能差分 (导航主导) ──────────────────────────────────────────────
    gamma   = float(rcfg.get("pbrs_gamma", 0.99))
    k_pbrs  = float(rcfg.get("pbrs_coef",  20.0))
    current_potential = -dtf
    if rstate.prev_potential is not None:
        pbrs   = gamma * current_potential - rstate.prev_potential
        reward += k_pbrs * pbrs
    rstate.prev_potential = current_potential

    # ── 障碍物排斥势 [v3 SHADOW] ──────────────────────────────────────────────
    # 优先使用真实障碍物 (env._obstacles); 若为空则使用影子障碍物 (shadow_obstacles)
    # 影子障碍物在 0 真实障碍物阶段提供 reward 信号, 让 critic 提前学习障碍物特征
    # 真实障碍物晋级到 >=1 后, 影子障碍物自动退出
    _real_obs = list(env._obstacles) if hasattr(env, '_obstacles') else []
    _obs_for_reward = _real_obs if len(_real_obs) > 0 else getattr(rstate, 'shadow_obstacles', [])
    d0    = float(rcfg.get("obs_repulse_d0",     0.10))
    k_obs = float(rcfg.get("obs_repulse_coef",   0.3))
    use_linear = bool(rcfg.get("obs_repulse_linear", True))
    for (ox, oy, orad) in _obs_for_reward:
        d = max(float(np.linalg.norm(pl_xy - np.array([ox, oy]))) - orad, 0.001)
        if d < d0:
            if use_linear:
                reward -= k_obs * (1.0 - d / d0)
            else:
                reward -= k_obs * (1.0/d - 1.0/d0) ** 2 * 0.5
    reward = max(reward, -5.0)

    # ── Z 高度 [v3: 归零, PID 负责 Z] ────────────────────────────────────────
    # z_dev_coef=0 和 vz_penalty_coef=0 意味着这两行不产生 reward
    z_dev = abs(payload_z - z_lock)
    reward -= float(rcfg.get("z_dev_coef", 0.0)) * min(z_dev, 0.15)

    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz   = float(env.data.qvel[dof_idx + 2])
    if pl_vz < 0:
        reward -= float(rcfg.get("vz_penalty_coef", 0.0)) * min(abs(pl_vz), 0.3)

    # ── step penalty [v3: alive_bonus=0] ─────────────────────────────────────
    # alive_bonus=0 且 step_penalty=-0.01 → 净效果 -0.01/step
    reward += float(rcfg.get("alive_bonus",    0.0))
    reward += float(rcfg.get("step_penalty", -0.01))

    # ── 近目标稠密奖励 ────────────────────────────────────────────────────────
    near_r = float(rcfg.get("near_goal_radius", 0.10))
    near_b = float(rcfg.get("near_goal_bonus",  0.05))
    if dtf < near_r:
        reward += near_b * (1.0 - dtf / near_r)

    # ── 成功判定 ──────────────────────────────────────────────────────────────
    phase_cfg = config["phase_transition"]
    pl_mat    = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler  = R.from_matrix(pl_mat).as_euler('xyz')
    tilt      = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    ee_vxy    = np.array([obs[2], obs[3]])
    swing_vel = float(np.linalg.norm(pl_vxy - ee_vxy))
    pl_vel    = float(np.linalg.norm(pl_vxy))

    swing_disp     = float(np.linalg.norm(pl_xy - np.array([obs[0], obs[1]])))
    swing_disp_max = float(config.get("step_logic", {}).get("swing_xy_max", 0.35)) * 0.5
    z_tol_frac     = float(rcfg.get("z_success_tol_frac", 0.15))
    z_success_tol  = z_lock * z_tol_frac
    success_radius = getattr(rstate, 'current_success_radius', 0.20)
    _min_steps     = int(config.get("cruise_rl", {}).get("min_steps_for_success", 5))
    _current_step  = getattr(env, 'current_step', 0)

    tilt_max  = phase_cfg["cruise_to_descent_tilt_max"]
    swing_max = phase_cfg["cruise_to_descent_swing_vel_max"]
    vel_max   = phase_cfg["cruise_to_descent_payload_vel_max"]

    if (_current_step >= _min_steps and
            abs(payload_z - z_lock) < z_success_tol and
            dtf < success_radius and
            tilt < tilt_max and
            swing_vel < swing_max and
            swing_disp < swing_disp_max and
            pl_vel < vel_max):
        reward  += float(rcfg.get("success_bonus", 8.0))
        success  = True
        done     = True
        info["termination"] = (
            f"cruise_success:r={success_radius:.3f},"
            f"d={dtf*1000:.1f}mm,z={payload_z*1000:.1f}mm,"
            f"E={swing_energy*1000:.1f}mJ")

    # ── 里程碑奖励 ────────────────────────────────────────────────────────────
    milestone_r = float(rcfg.get("milestone_radius", 0.15))
    if not success and dtf < milestone_r and not getattr(rstate, '_near_given', False):
        reward += float(rcfg.get("milestone_bonus", 1.5))
        rstate._near_given = True

    # ── 超时 ──────────────────────────────────────────────────────────────────
    max_steps = int(config["cruise_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        info["termination"] = "timeout"
        if rstate.initial_dist and rstate.initial_dist > 0.01:
            progress_frac = max(0.0, 1.0 - dtf / rstate.initial_dist)
        else:
            progress_frac = max(0.0, 1.0 - dtf / 0.3)
        reward += progress_frac * 5.0

    return reward, done, success, info


# ==============================================================================
# Phase 3: Descent Reward [v3]
#
# 核心变更:
#   1. z_unconditional_frac=0: 完全移除无条件下降激励
#      → agent 必须先对准 (dtf<30mm) 才能获得 z 下降奖励
#   2. rebar_sdf 严格门控: dtf < 8mm AND z_reached
#      → 防止在未对准时给予高斯对准奖励 (exploit)
#   3. near_goal_focus 上限 0.10/step
#      → 500步 × 0.10 = 50 = success_bonus, 但同时有 step_penalty × 500 = -10
#      → 净效果: 悬停不如成功合算
#   4. step_penalty -0.02: 逼迫快速插入
#   5. 超时不给 soft_success_bonus: 成功唯一途径是真正插入
#   6. 架构: PID base + RL residual (reward 逻辑不变, 架构变更在 train_phase.py)
# ==============================================================================

class DescentRewardState:
    def __init__(self):
        self.prev_z = None
        self.prev_dtf = None
        self.insertion_hold_counter = 0
        self.total_steps_global = 0
        # 课程注入字段
        self.current_xy_range      = None
        self.current_xy_tol        = None   # [v3.4] per-level tol (由训练循环注入)
        self.current_descent_level = 0
        self.descent_n_levels      = 5
        self.steps_at_max_level    = 0

    def reset(self):
        self.prev_z = None
        self.prev_dtf = None
        self.insertion_hold_counter = 0


def compute_descent_reward(env, obs, config, rstate):
    """
    Descent 段奖励 v3.2 — reward/SR 反相关修复版.

    根因分析:
      xy_gauss_coef=1.5 在 z 到位后每步给 ~1.49 的持续奖励.
      500步失败 episode 累积高斯奖励 ~650 >> success_bonus=50.
      episode 越长 reward 越高 → SR↑(episode变短)时 avg_reward 反而下降.

    修复原则:
      1. 完全移除 xy_gauss per-step 持续奖励 (根本原因)
      2. 完全移除 rebar_sdf per-step 持续奖励 (同类问题)
      3. 完全移除 near_goal_focus per-step 持续奖励 (同类问题)
      4. 保留纯差分奖励 (xy_align_diff, z_descent_diff): 有界, 不随 episode 长度累积
      5. 保留稳定性惩罚: 有界, 不随 episode 长度累积 (失败ep摆动大→惩罚更重)
      6. success_bonus=50 作为唯一的大额正向奖励
      7. 终止时一次性对准奖励代替 per-step 高斯 (仅成功时给)

    量级验证 (修复后):
      成功ep(150步): 稳定(-20) + z_diff(+1) + xy_diff(+1) + pen(-3) + bonus(+50) = +29
      失败ep(500步): 稳定(-81) + z_diff(+2) + xy_diff(+1) + pen(-10) + NO_bonus  = -88
      → reward 与 SR 正相关 ✅
    """
    rcfg    = config["descent_rl"]["reward"]
    cfg_ins = config.get("insertion", {})
    target_xy = env.target_pos.copy()
    target_pz = float(cfg_ins.get("target_payload_z", 0.10))

    pl_xy     = np.array([env.data.body('prefab').xpos[0],
                           env.data.body('prefab').xpos[1]])
    payload_z = float(env.data.body('prefab').xpos[2])
    dtf       = float(np.linalg.norm(pl_xy - target_xy))

    pl_mat   = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt     = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    abs_yaw  = abs(float(pl_euler[2]))

    reward  = 0.0
    done    = False
    success = False
    info    = {}

    # ── 安全检查 ──────────────────────────────────────────────────────────────
    unstable, reason = _check_instability(env, obs, config, grace_steps=20)
    if unstable:
        return float(rcfg["instability_penalty"]), True, False, {"termination": reason}

    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz   = float(env.data.qvel[dof_idx + 2])
    if payload_z < 0.03 and pl_vz < -0.3:
        return float(rcfg["crash_penalty"]), True, False, {"termination": "crash"}

    # ── [P1] 稳定性惩罚 (有界, 失败ep摆动大→惩罚更重 → 正相关方向) ───────────
    ke = _get_swing_ke(env, obs)
    reward -= float(rcfg.get("swing_ke_coef", 3.0)) * min(ke, 0.25)
    reward -= float(rcfg.get("tilt_coef", 1.5))     * min(tilt,    0.3)
    reward -= float(rcfg.get("yaw_coef",  2.0))     * min(abs_yaw, 0.3)

    # ── [P2] XY 对准奖励 (纯差分, 有界: ±clip×coef/step) ─────────────────────
    # 对准后 delta_dtf ≈ 0 → 该项趋近 0, 不随 episode 长度无限累积
    z_not_reached = abs(payload_z - target_pz) >= 0.025
    if rstate.prev_dtf is not None:
        delta_dtf_m       = rstate.prev_dtf - dtf
        delta_dtf_clipped = np.clip(delta_dtf_m, -0.010, 0.010)
        # z 未到位时给一半系数 (鼓励先对准)
        coef_mult = 0.5 if z_not_reached else 1.0
        reward += delta_dtf_clipped * float(rcfg.get("xy_align_coef", 4.0)) * coef_mult
    rstate.prev_dtf = dtf

    # [v3.2] 移除 xy_gauss per-step 持续奖励 (根本原因: 每步+1.49, 随 episode 长度累积)
    # 原代码: reward += xy_gauss_coef * exp(-0.5*(dtf/sigma)^2)  ← 已删除

    # [v3.2] 移除 rebar_sdf per-step 持续奖励 (同类问题)
    # 原代码: reward += sdf_coef * mean(gaussians)               ← 已删除

    # ── [P3] Z 下降奖励 (纯差分, 有界) ───────────────────────────────────────
    # z_delta 每步典型 0.001-0.002m → 6.0×0.002 = 0.012/step (量级可控)
    # 对准门控: agent 需先把 dtf 降到 xy_gate 以内才能获得 z 奖励
    if rstate.prev_z is not None:
        z_delta    = rstate.prev_z - payload_z   # 向下为正
        xy_aligned = dtf < float(rcfg.get("z_descent_xy_gate", 0.030))

        if xy_aligned and z_delta > 0 and payload_z > target_pz:
            reward += float(rcfg.get("z_descent_coef", 6.0)) * z_delta

        # 上升惩罚 (PID 已阻止激进上升, 此惩罚为兜底)
        if z_delta < 0 and payload_z < 0.60:
            z_rise        = abs(z_delta)
            rise_penalty  = float(rcfg.get("z_rise_penalty_coef", 2.0)) * z_rise
            rise_penalty  = min(rise_penalty, float(rcfg.get("z_rise_max_penalty", 0.2)))
            reward       -= rise_penalty

    rstate.prev_z = payload_z

    # ── step penalty ──────────────────────────────────────────────────────────
    reward += float(rcfg.get("step_penalty", -0.02))

    # [v3.2] 移除 near_goal_focus per-step 持续奖励 (同类问题)
    # 原代码: reward += focus_coef * (1.0 - dtf/focus_xy)       ← 已删除

    # ── 成功判定 ──────────────────────────────────────────────────────────────
    cur_descent_level     = getattr(rstate, 'current_descent_level', 0)
    cur_descent_n_levels  = getattr(rstate, 'descent_n_levels',      5)
    steps_at_max_level    = getattr(rstate, 'steps_at_max_level',    0)
    xy_tol_final          = float(cfg_ins.get("xy_tolerance_train_end", 0.005))

    # [v3.4] xy_tol 优先从 rstate.current_xy_tol 读取 (per-level 查表)
    # 退回顺序: current_xy_tol → current_xy_range×tol_mult → 全局退火
    _injected_tol = getattr(rstate, 'current_xy_tol', None)
    cur_descent_xy_range  = getattr(rstate, 'current_xy_range', None)

    if _injected_tol is not None:
        # 新逻辑: per-level 固定 tol
        # 最高 level 时, 在 fine_anneal_steps 内从 level_tol 退火到 xy_tol_final
        xy_tol = float(_injected_tol)
        if cur_descent_level >= cur_descent_n_levels - 1:
            fine_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 500_000))
            fine_frac  = min(steps_at_max_level / max(fine_steps, 1), 1.0)
            xy_tol     = xy_tol + fine_frac * (xy_tol_final - xy_tol)
    elif cur_descent_xy_range is not None:
        # 旧逻辑兼容: tol_mult 计算
        tol_mult = float(cfg_ins.get("xy_tol_range_multiplier", 3.0))
        xy_tol   = max(cur_descent_xy_range * tol_mult, xy_tol_final)
    else:
        # 全局退火 (无课程时)
        total_ts     = getattr(rstate, 'total_steps_global', 0)
        anneal_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 500_000))
        frac         = min(total_ts / max(anneal_steps, 1), 1.0)
        xy_tol = (float(cfg_ins.get("xy_tolerance_train_start", 0.040)) +
                  frac * (xy_tol_final - float(cfg_ins.get("xy_tolerance_train_start", 0.040))))

    total_ts = getattr(rstate, 'total_steps_global', 0)
    anneal_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 500_000))
    frac = min(total_ts / max(anneal_steps, 1), 1.0)

    tilt_tol = (float(cfg_ins.get("tilt_tolerance_train_start", 0.12)) +
                frac * (float(cfg_ins.get("tilt_tolerance_train_end", 0.05)) -
                        float(cfg_ins.get("tilt_tolerance_train_start", 0.12))))
    yaw_tol  = (float(cfg_ins.get("yaw_tolerance_train_start", 0.15)) +
                frac * (float(cfg_ins.get("yaw_tolerance_train_end", 0.08)) -
                        float(cfg_ins.get("yaw_tolerance_train_start", 0.15))))

    z_tol      = float(cfg_ins.get("success_z_tolerance", 0.020))
    hold_steps = int(cfg_ins.get("hold_steps", 3))

    on_target = (abs(payload_z - target_pz) < z_tol and
                 dtf < xy_tol and
                 tilt < tilt_tol and abs_yaw < yaw_tol)

    if on_target:
        rstate.insertion_hold_counter += 1
    else:
        rstate.insertion_hold_counter = 0

    if rstate.insertion_hold_counter >= hold_steps:
        reward  += float(rcfg["success_bonus"])
        # [v3.2] 一次性对准质量奖励: 替代被删除的 per-step xy_gauss
        # 鼓励更精准的插入, 但只在成功时给一次 (不影响 episode 长度)
        precision_bonus_coef = float(rcfg.get("precision_bonus_coef", 10.0))
        sigma_prec = float(rcfg.get("precision_bonus_sigma", 0.005))  # 5mm
        reward += precision_bonus_coef * float(np.exp(-0.5 * (dtf / sigma_prec) ** 2))
        success  = True
        done     = True
        info["termination"] = (
            f"insertion_success:z={payload_z*1000:.0f}mm,"
            f"dtf={dtf*1000:.1f}mm,tilt={tilt:.3f},yaw={abs_yaw:.3f},"
            f"tols=xy{xy_tol*1000:.1f}mm/tilt{tilt_tol:.3f}/yaw{yaw_tol:.3f}")
        return reward, done, success, info

    # ── 超时: 无额外奖励 ──────────────────────────────────────────────────────
    max_steps = int(config["descent_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        info["termination"] = (
            f"timeout:dtf={dtf*1000:.1f}mm,z={payload_z*1000:.0f}mm,"
            f"tilt={tilt:.3f},yaw={abs_yaw:.3f}")

    return reward, done, success, info