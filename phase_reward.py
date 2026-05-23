# ==============================================================================
# phase_reward.py — 三阶段独立奖励函数 v8 (重构版)
#
# v8 重构核心:
#   1. 删除 compute_cruise_reward 函数内重复的 v3 旧版函数体
#   2. 删除 compute_cruise_reward_planner / compute_cruise_reward_swing_rl (dual-RL)
#   3. 删除 shadow_obstacles 逻辑 (障碍物课程已废弃)
#   4. 保留三阶段核心 reward: lift / cruise / descent
#   5. 保留 RewardComponentTracker (用于 wandb 分项), 但仅记录关键分项
#
# 各阶段成功条件保持不变 (训练表现已验证). 噪声/风力鲁棒性通过课程层注入,
# 不修改 reward 函数本身.
# ==============================================================================

import numpy as np
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# [v12.3 新增] 绳索能量帮助函数
# 文献依据: Kotaru 2017 (arXiv:1711.04895), FLARE 2025 (arXiv:2508.09797)
# 我们的 4 根绳, 每根 10 段, 用各段 com 线速度的平方和作为动能代理
# (不乘 mass 是因为各段 mass 相同, 系数可调)
# ==============================================================================

def _get_cable_kinetic_energy(env):
    """
    返回所有绳所有段 (4×10=40 段) 的总动能代理 = sum_i ||v_i||²

    返回值量级:
      静止: ~0
      轻微摆动: 0.001-0.01
      明显摆动: 0.1-1.0
      剧烈摆动: > 1.0

    优势 (vs 仅看 payload swing_energy):
      cable 段在 payload 还没开始大幅摆动时已有动能 (因为它们更轻)
      所以能更早检测到摆动趋势, 让 RL 提前介入
    """
    try:
        return float(env._get_cable_energy())
    except Exception:
        return 0.0


# ==============================================================================
# 通用安全/碰撞检查
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
    pl_vxy   = obs[6:8]
    payload_z = env.data.body('prefab').xpos[2]
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz   = float(env.data.qvel[dof_idx + 2])
    pl_vel  = float(np.linalg.norm(np.append(pl_vxy, pl_vz)))
    pl_mat  = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    abs_yaw = abs(float(pl_euler[2]))

    parts = []
    if swing_xy > cfg_logic.get("swing_xy_max",   0.25): parts.append(f"swing={swing_xy:.3f}")
    if pl_vel   > cfg_logic.get("payload_vel_max", 2.0): parts.append(f"vel={pl_vel:.2f}")
    if tilt     > cfg_logic.get("payload_tilt_max", 1.0): parts.append(f"tilt={tilt:.2f}")
    if abs_yaw  > cfg_logic.get("payload_yaw_max",  1.2): parts.append(f"yaw={abs_yaw:.2f}")

    if parts:
        return True, "instability:" + ",".join(parts)
    return False, ""


def _check_collision(env, config):
    use_mjc = config.get("reward", {}).get("use_mujoco_contact", True)
    if use_mjc:
        hit_obs, _ = env._check_prefab_collision_with_obstacles()
        return hit_obs
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


def _get_swing_energy(env, obs, config):
    """统一摆动能量 (KE + PE), 用于 lift/cruise/descent。"""
    from ee_acc_controller import compute_swing_energy
    pl_pos = env.data.body('prefab').xpos.copy()
    ee_pos = env._get_ee_pos()
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vel = env.data.qvel[dof_idx:dof_idx+3].copy()
    ee_vel = getattr(env, '_ee_vel_cache', np.zeros(3))
    mass   = float(env.config.get("prefab",     {}).get("mass", 1.0))
    rope_L = float(env.config.get("controller", {}).get("L",    0.5))
    return compute_swing_energy(pl_pos, ee_pos, pl_vel, ee_vel, mass, rope_L)


# ==============================================================================
# Phase 1: Lift Reward
# ==============================================================================

class LiftRewardState:
    def __init__(self):
        self.prev_z         = None
        self.prev_dtf_start = None
        self.hold_counter   = 0
        self.start_xy       = None
        self.total_steps_global = 0
        # [v12] 残差 RL 用: action smoothness penalty
        self.prev_rl_action = None
        # [v12.4] 合并 cruise reward: swing_improve 差分需要 prev_swing_energy
        self.prev_swing_energy = None

    def reset(self):
        self.prev_z         = None
        self.prev_dtf_start = None
        self.hold_counter   = 0
        self.prev_rl_action = None
        self.prev_swing_energy = None


def compute_lift_reward(env, obs, config, rstate, tracker=None, rl_action=None):
    """
    Lift 段奖励 v12.4 — 与 cruise 统一: 使用相同的防摆 reward 主体, 只差成功判定.

    [v12.4 重大改动] 合并 lift+cruise reward 设计:
      Lift 与 cruise 物理任务几乎相同 (末端动, payload 跟随防摆), 之前分两套
      reward 在过渡处不连续, RL 学不稳. 现在统一:
        共用部分 (防摆主体):
          - swing_energy_penalty (核心, 强惩罚)
          - cable_ke_penalty (v12.3 新增)
          - action_magnitude_penalty + action_smoothness_penalty (Olesen/CAPS)
          - tilt_penalty (防 payload 翻倒)
        Lift 特有:
          - z_approach 差分 (引导上升, 主导)
          - 成功条件 = 到达 z_cruise 高度 + xy 不漂 + 摆动小
        Cruise 特有:
          - swing_improve 差分 (主动减摆)
          - calm_bonus (持续低摆)
          - 成功条件 = 到达 target_xy

    返回与原版本一致: (reward, done, success, info)
    """
    rcfg     = config["lift_rl"]["reward"]
    z_cruise = float(config["lift_rl"]["target_z_cruise"])

    pl_pos    = env.data.body('prefab').xpos.copy()
    payload_z = float(pl_pos[2])
    pl_xy     = pl_pos[:2].copy()
    reward = 0.0; done = False; success = False; info = {}

    # ── 安全检查 (与 cruise 一致) ─────────────────────────────────────────────
    unstable, reason = _check_instability(env, obs, config, grace_steps=30)
    if unstable:
        r = float(rcfg["instability_penalty"])
        if tracker: tracker.add("instability_penalty", r)
        return r, True, False, {"termination": reason}

    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz = float(env.data.qvel[dof_idx + 2])
    if payload_z < 0.03 and pl_vz < -0.5:
        r = float(rcfg["crash_penalty"])
        if tracker: tracker.add("crash_penalty", r)
        return r, True, False, {"termination": "crash"}

    _sxy = getattr(rstate, 'start_xy', None)
    start_xy = _sxy if _sxy is not None else env.default_start_xy.copy()
    dtf_start = float(np.linalg.norm(pl_xy - start_xy))

    # ── [v12.4 统一] 摆动能量计算 (与 cruise 一致) ──────────────────────────
    try:
        swing_energy, swing_ke, swing_pe, swing_angle = _get_swing_energy(env, obs, config)
    except Exception:
        swing_energy, swing_ke, swing_pe, swing_angle = 0., 0., 0., 0.

    # ── [v12.4 统一防摆主体] 与 cruise 完全相同 ─────────────────────────────
    # 1. swing_energy_penalty: 绝对惩罚 (主信号)
    e_thresh = float(rcfg.get("swing_energy_thresh",       0.010))
    e_coef   = float(rcfg.get("swing_energy_penalty_coef", 4.0))
    e_max    = float(rcfg.get("swing_energy_penalty_max",  0.10))
    r_swing = 0.0
    if swing_energy > e_thresh:
        r_swing = -min(e_coef * (swing_energy - e_thresh), e_max)
        reward += r_swing
    if tracker:
        tracker.add("swing_energy_penalty", r_swing)
        tracker.add("swing_energy_J", swing_energy)

    # 2. [v12.4 加入] swing_improve 差分: 鼓励主动减摆 (与 cruise 一致)
    k_improve  = float(rcfg.get("swing_improve_coef", 20.0))
    imp_max    = float(rcfg.get("swing_improve_max",   0.04))
    worsen_max = float(rcfg.get("swing_worsen_max",    0.04))
    r_improve = 0.0
    if rstate.prev_swing_energy is not None:
        delta_energy = rstate.prev_swing_energy - swing_energy
        if delta_energy > 0:
            r_improve = min(k_improve * delta_energy, imp_max)
        else:
            r_improve = -min(k_improve * abs(delta_energy), worsen_max)
        reward += r_improve
    rstate.prev_swing_energy = swing_energy
    if tracker: tracker.add("swing_improve", r_improve)

    # 3. [v12.3] cable_ke_penalty (绳索动能, 防 cable 高频振动)
    cable_ke = _get_cable_kinetic_energy(env)
    ce_thresh = float(rcfg.get("cable_ke_thresh",      0.05))
    ce_coef   = float(rcfg.get("cable_ke_penalty_coef", 1.0))
    ce_max    = float(rcfg.get("cable_ke_penalty_max",  0.08))
    r_cable_ke = 0.0
    if cable_ke > ce_thresh:
        r_cable_ke = -min(ce_coef * (cable_ke - ce_thresh), ce_max)
        reward += r_cable_ke
    if tracker:
        tracker.add("cable_ke_penalty", r_cable_ke)
        tracker.add("cable_ke", cable_ke)

    # 4. Tilt 惩罚 (有界, lift 段防 payload 翻倒)
    pl_euler = R.from_matrix(env.data.body('prefab').xmat.reshape(3, 3)).as_euler('xyz')
    tilt     = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    tilt_max_pen = float(rcfg.get("tilt_max_for_penalty", 0.3))
    r_tilt = -float(rcfg.get("tilt_coef", 0.5)) * min(tilt, tilt_max_pen)
    reward += r_tilt
    if tracker: tracker.add("tilt_penalty", r_tilt)

    # 5. action_magnitude_penalty (Olesen 2026)
    r_act_mag = 0.0
    if rl_action is not None:
        a_norm_sq = float(np.mean(np.square(np.asarray(rl_action, np.float32))))
        k_act_mag = float(rcfg.get("action_magnitude_coef", 0.02))
        r_act_mag = -k_act_mag * a_norm_sq
        reward += r_act_mag
    if tracker: tracker.add("action_magnitude_penalty", r_act_mag)

    # 6. action_smoothness_penalty (Mysore 2021 CAPS)
    r_act_smooth = 0.0
    if rl_action is not None and rstate.prev_rl_action is not None:
        diff = np.asarray(rl_action, np.float32) - np.asarray(rstate.prev_rl_action, np.float32)
        smooth_sq = float(np.mean(np.square(diff)))
        k_smooth = float(rcfg.get("action_smoothness_coef", 0.04))
        r_act_smooth = -k_smooth * smooth_sq
        reward += r_act_smooth
    if rl_action is not None:
        rstate.prev_rl_action = np.asarray(rl_action, np.float32).copy()
    if tracker: tracker.add("action_smoothness_penalty", r_act_smooth)

    # ── [Lift 特有] z 接近差分 (引导上升) + xy 漂移惩罚 ─────────────────────
    # 这部分是 lift vs cruise 的唯一差异
    r_z = 0.0
    if rstate.prev_z is not None:
        r_z = float(rcfg.get("z_approach_coef", 2.0)) * (
            abs(rstate.prev_z - z_cruise) - abs(payload_z - z_cruise))
        reward += r_z
    rstate.prev_z = payload_z
    if tracker: tracker.add("z_approach", r_z)

    r_xy = 0.0
    xy_coef = float(rcfg.get("xy_drift_coef", 1.0))
    if rstate.prev_dtf_start is not None:
        delta = np.clip(rstate.prev_dtf_start - dtf_start, -0.010, 0.010)
        r_xy = xy_coef * float(delta)
        reward += r_xy
    rstate.prev_dtf_start = dtf_start
    if tracker: tracker.add("xy_drift_reward", r_xy)

    # ── 成功判定 (Lift 特有: 达到 z_cruise + 稳定) ────────────────────────────
    phase_cfg = config["phase_transition"]
    z_tol      = float(rcfg.get("success_z_tol",                0.025))
    vz_max     = float(rcfg.get("success_vz_max",               0.08))
    s_e_thresh = float(rcfg.get("success_swing_energy_thresh",  0.020))
    tilt_max   = float(phase_cfg["lift_to_cruise_tilt_max"])
    xy_max     = float(rcfg.get("xy_max_dist_success", 0.12))
    hold_steps = int(rcfg.get("hold_steps", 3))

    in_zone = (abs(payload_z - z_cruise) < z_tol and
               abs(pl_vz) < vz_max  and
               swing_energy < s_e_thresh and
               tilt < tilt_max and
               dtf_start < xy_max)
    rstate.hold_counter = (rstate.hold_counter + 1) if in_zone else 0

    if rstate.hold_counter >= hold_steps:
        r_bonus = float(rcfg["success_bonus"])
        reward += r_bonus
        if tracker: tracker.add("success_bonus", r_bonus)
        success = True; done = True
        info["termination"] = (
            f"lift_success:z={payload_z*1000:.0f}mm,"
            f"vz={pl_vz*1000:.0f}mm/s,E={swing_energy*1000:.0f}mJ,"
            f"dtf={dtf_start*1000:.0f}mm")
        return reward, done, success, info

    if getattr(env, 'current_step', 0) >= int(config["lift_rl"]["max_steps"]) - 1:
        done = True
        info["termination"] = (
            f"timeout:z={payload_z*1000:.0f}mm,"
            f"E={swing_energy*1000:.0f}mJ,dtf={dtf_start*1000:.0f}mm")

    reward = float(np.clip(reward, -2.0, 6.0))
    return reward, done, success, info


# ==============================================================================
# Phase 2: Cruise Reward (v8 — NMPC base + RL 残差专用)
#
# 核心思路:
#   NMPC base 已保证: 100% 到达 + 优秀防摆 (KE≈4mJ)
#   RL 残差任务: 在 NMPC 基础上进一步改善鲁棒性 (噪声/风力下保持稳定)
#
# 防摆差分奖励是核心: reward = k * (prev_E - curr_E)
#   摆动减少 → 正奖励 (RL 学习有效防摆)
#   摆动增加 → 负惩罚 (非对称, 不对称防止 RL 学到"先制造摆动再消除")
# ==============================================================================

class CruiseRewardState:
    def __init__(self):
        self.prev_potential   = None
        self._near_given      = False
        self.total_steps_global = 0
        self.initial_dist     = None
        self.prev_swing_energy = None
        self.hold_counter     = 0
        # [v11.2] 残差 RL 用: 跟踪上一步 RL action (供 action smoothness penalty)
        self.prev_rl_action   = None
        # [v13.0] 合并 lift 段后, z_approach 差分需要 prev_z
        self.prev_z           = None

    def reset(self):
        self.prev_potential    = None
        self._near_given       = False
        self.initial_dist      = None
        self.prev_swing_energy = None
        self.hold_counter      = 0
        self.prev_rl_action    = None
        self.prev_z            = None


def compute_cruise_reward(env, obs, config, rstate, tracker=None, rl_action=None):
    """
    Cruise 段奖励 v11.2 — 基于 residual RL 文献彻底重设计.

    设计原则 (文献依据):
    1. Olesen et al. 2026 (arXiv:2602.05895, Residual RL for Crane Container Lifting):
       与本任务几乎相同的场景. 用 action_magnitude_penalty 让 residual policy
       默认接近 0, 仅在必要时介入.
    2. Jeon et al. 2025 (arXiv:2510.12717, Residual MPC): residual RL 的 reward
       应与 base MPC 目标对齐, 避免 redundant 信号 (如让 RL 加速到目标 → 与 MPC 冲突).
    3. Mysore et al. 2021 (arXiv:2012.06644, CAPS): temporal action smoothness penalty
       (||a_t - a_{t-1}||²) 防止 RL 抖动破坏底层 controller.
    4. Alakuijala et al. 2021 (arXiv:2106.08050, Residual RL from Demos): residual
       formulation 应该用简单 reward, 让 RL 学小 action.

    与 v8 的差异 (全部基于诊断报告):
      ❌ 删除 pbrs_nav:        NMPC 已导航, 给 RL 加速信号是 redundant 且 harmful
      ❌ 删除 step_penalty:    与 pbrs_nav 同样鼓励"加速", 与 base 冲突
      ❌ 删除 milestone_bonus: 一次性 spike, 高方差信号
      ❌ 删除 obs_repulsion:   NMPC 已避障, 残差不该介入
      ✅ 大幅强化 swing_energy 绝对惩罚: threshold 50mJ → 10mJ, coef 2 → 8
      ✅ 新增 action_magnitude_penalty: -λ_a * ||a_res||² (强制残差小)
      ✅ 新增 action_smoothness_penalty: -λ_s * ||a_res - a_res_prev||² (防抖动)
      ✅ 扩大 calm_bonus 触发范围: near_goal 不必, 任何时候 swing < threshold 都奖
      ✅ success_bonus 提升: 8 → 20 (让 RL 清楚 "成功 > 任何 collision")
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

    # ── 安全检查 (保留, 仅终止信号) ───────────────────────────────────────────
    unstable, reason = _check_instability(env, obs, config, grace_steps=15)
    if unstable:
        r = float(rcfg["instability_penalty"])
        if tracker: tracker.add("instability_penalty", r)
        return r, True, False, {"termination": reason}

    payload_z = float(env.data.body('prefab').xpos[2])
    z_lock    = float(config.get("cruise_rl", {}).get("z_lock_height", 0.25))
    pid_cfg   = config.get("cruise_z_pid", {})
    if payload_z < float(pid_cfg.get("floor_z_threshold", 0.04)):
        r = float(rcfg.get("collision_penalty", -10.0))
        if tracker: tracker.add("ground_collision_penalty", r)
        return r, True, False, {"termination": f"ground_collision:z={payload_z:.3f}"}
    if payload_z < z_lock * 0.20:
        r = float(rcfg.get("instability_penalty", -3.0))
        if tracker: tracker.add("instability_penalty", r)
        return r, True, False, {"termination": f"payload_too_low:z={payload_z*1000:.0f}mm"}
    if _check_collision(env, config):
        r = float(rcfg["collision_penalty"])
        if tracker: tracker.add("collision_penalty", r)
        return r, True, False, {"termination": "collision"}
    if float(np.linalg.norm(pl_xy)) < 0.03:
        r = float(rcfg["collision_penalty"])
        if tracker: tracker.add("collision_penalty", r)
        return r, True, False, {"termination": "collision_base"}

    # ── 摆动能量 (核心物理量) ────────────────────────────────────────────────
    try:
        swing_energy, swing_ke, swing_pe, swing_angle = _get_swing_energy(env, obs, config)
    except Exception:
        swing_energy, swing_ke, swing_pe, swing_angle = 0., 0., 0., 0.

    # ── 1. [v11.2 强化] swing_energy 绝对惩罚 — 保护 NMPC 稳定性 ──────────────
    # 用户核心需求: "残差 RL 不能破坏 NMPC 稳定性"
    # 阈值降到 NMPC normal (4 mJ) 的 2.5×, 让任何让 swing 增大的 action 都被惩罚
    energy_thresh  = float(rcfg.get("swing_energy_thresh",       0.010))  # 50mJ → 10mJ
    energy_coef    = float(rcfg.get("swing_energy_penalty_coef", 8.0))    # 2.0 → 8.0
    energy_pen_max = float(rcfg.get("swing_energy_penalty_max",  0.50))   # 0.30 → 0.50
    r_swing_abs = 0.0
    if swing_energy > energy_thresh:
        excess = swing_energy - energy_thresh
        r_swing_abs = -min(energy_coef * excess, energy_pen_max)
        reward += r_swing_abs
    if tracker:
        tracker.add("swing_energy_penalty", r_swing_abs)
        tracker.add("swing_energy_J", swing_energy)
        tracker.add("swing_angle_deg", float(np.degrees(swing_angle)))

    # ── 1b. [v12.3] 绳索动能惩罚 (用户要求, 三段统一)
    # 防止 RL 让绳索快速振动. 即使 payload 看似稳定但 cable 在抖动.
    # 来源: FLARE 2025 (arXiv:2508.09797), Kotaru 2017
    cable_ke = _get_cable_kinetic_energy(env)
    ce_thresh = float(rcfg.get("cable_ke_thresh",      0.05))
    ce_coef   = float(rcfg.get("cable_ke_penalty_coef", 1.0))
    ce_max    = float(rcfg.get("cable_ke_penalty_max",  0.10))
    r_cable_ke = 0.0
    if cable_ke > ce_thresh:
        r_cable_ke = -min(ce_coef * (cable_ke - ce_thresh), ce_max)
        reward += r_cable_ke
    if tracker:
        tracker.add("cable_ke_penalty", r_cable_ke)
        tracker.add("cable_ke", cable_ke)

    # ── 2. [v11.2 保留 + 改进] swing_improve 差分奖励 ────────────────────────
    # 鼓励 RL 主动减摆动, 但 worsen 系数与 improve 对称 (不再 0.5×)
    k_improve  = float(rcfg.get("swing_improve_coef", 50.0))
    imp_max    = float(rcfg.get("swing_improve_max",   0.20))
    worsen_max = float(rcfg.get("swing_worsen_max",    0.20))   # 与 imp_max 对称
    r_improve = 0.0
    if rstate.prev_swing_energy is not None:
        delta_energy = rstate.prev_swing_energy - swing_energy
        if delta_energy > 0:
            r_improve = min(k_improve * delta_energy, imp_max)
        else:
            r_improve = -min(k_improve * abs(delta_energy), worsen_max)
        reward += r_improve
    rstate.prev_swing_energy = swing_energy
    if tracker: tracker.add("swing_improve", r_improve)

    # ── 3. [v11.2 新增] action_magnitude_penalty ─────────────────────────────
    # Olesen 2026: residual policy 应学到 "默认 0, 必要时介入"
    # 来源: arXiv:2602.05895 §III.C
    # 量级设计: 一 ep cap = -2.5 (与 success_bonus 比例 1:8, 不会主导)
    r_act_mag = 0.0
    if rl_action is not None:
        # rl_action 是 RL 输出, 在 [-1, 1] 内 (squashed); 平方平均
        a_norm_sq = float(np.mean(np.square(np.asarray(rl_action, np.float32))))
        k_act_mag = float(rcfg.get("action_magnitude_coef", 0.05))
        r_act_mag = -k_act_mag * a_norm_sq
        reward += r_act_mag
    if tracker: tracker.add("action_magnitude_penalty", r_act_mag)

    # ── 4. [v11.2 新增] action_smoothness_penalty (CAPS) ─────────────────────
    # Mysore 2021: 防止 RL 抖动, ||a_t - a_{t-1}||²
    # 来源: arXiv:2012.06644 §III
    r_act_smooth = 0.0
    if rl_action is not None and rstate.prev_rl_action is not None:
        diff = np.asarray(rl_action, np.float32) - np.asarray(rstate.prev_rl_action, np.float32)
        smooth_sq = float(np.mean(np.square(diff)))
        k_smooth = float(rcfg.get("action_smoothness_coef", 0.10))
        r_act_smooth = -k_smooth * smooth_sq
        reward += r_act_smooth
    if rl_action is not None:
        rstate.prev_rl_action = np.asarray(rl_action, np.float32).copy()
    if tracker: tracker.add("action_smoothness_penalty", r_act_smooth)

    # ── [v13.0 合并 lift 段] z_approach 差分 + tilt 惩罚 ──────────────────────
    # cruise 段现在合并了 lift, payload 起步在 z=0.11, 需要引导到 z_cruise=0.25.
    # 用差分 reward, 仅在 z < z_cruise - 0.02 (低空段) 启用.
    z_cruise = float(rcfg.get("target_z_cruise",
                              config["cruise_rl"].get("target_z_cruise", 0.25)))
    r_z = 0.0
    if payload_z < z_cruise - 0.02:  # 低空段, 启用 z 引导
        if rstate.prev_z is not None:
            r_z = float(rcfg.get("z_approach_coef", 2.0)) * (
                abs(rstate.prev_z - z_cruise) - abs(payload_z - z_cruise))
            reward += r_z
    rstate.prev_z = payload_z
    if tracker: tracker.add("z_approach", r_z)

    # tilt 惩罚 (lift 段防 payload 翻倒)
    pl_euler = R.from_matrix(env.data.body('prefab').xmat.reshape(3, 3)).as_euler('xyz')
    tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    tilt_max_pen = float(rcfg.get("tilt_max_for_penalty", 0.3))
    r_tilt = -float(rcfg.get("tilt_coef", 0.5)) * min(tilt, tilt_max_pen)
    reward += r_tilt
    if tracker: tracker.add("tilt_penalty", r_tilt)

    # ── 5. [v11.2 改进] calm_bonus — 任何时刻低摆动都奖, 强信号 ──────────────
    # 之前: 仅在 near_goal (10cm 内) + 低摆动同时满足才给
    # 现在: 任何时刻 swing < calm_thresh 就给, 在 near_goal 时给额外 bonus
    # 这是引导 RL 学 "持续保持低摆动" 的最强信号
    calm_thresh   = float(rcfg.get("calm_energy_thresh", 0.005))   # 0.015 → 0.005 (更严)
    base_calm     = float(rcfg.get("calm_base_bonus",    0.03))    # 全程低摆奖励
    near_r        = float(rcfg.get("near_goal_radius",   0.10))
    near_calm     = float(rcfg.get("near_goal_calm_bonus", 0.05))  # 近目标 + 低摆 额外
    r_calm = 0.0
    if swing_energy < calm_thresh:
        r_calm = base_calm * (1.0 - swing_energy / calm_thresh)
        # 近目标 + 低摆 双重满足时再加 bonus
        if dtf < near_r:
            r_calm += near_calm * (1.0 - dtf / near_r) * (1.0 - swing_energy / calm_thresh)
    reward += r_calm
    if tracker: tracker.add("calm_bonus", r_calm)

    # ── [已删除 v11.2]
    #   pbrs_nav        — NMPC 已导航, 给 RL 加速信号是 redundant + harmful
    #   step_penalty    — 与 pbrs_nav 配合鼓励"加速到目标", 与 base 冲突
    #   milestone_bonus — 一次性 spike, 高方差信号, 实际无用
    #   obs_repulsion   — NMPC 已避障, RL 残差不该介入

    # ── 成功判定 (success_bonus 大幅放大) ────────────────────────────────────
    phase_cfg = config["phase_transition"]
    pl_mat    = env.data.body('prefab').xmat.reshape(3, 3)
    pl_euler  = R.from_matrix(pl_mat).as_euler('xyz')
    tilt      = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
    ee_vxy    = np.array([obs[2], obs[3]])
    swing_vel = float(np.linalg.norm(pl_vxy - ee_vxy))
    pl_vel    = float(np.linalg.norm(pl_vxy))
    swing_disp     = float(np.linalg.norm(pl_xy - np.array([obs[0], obs[1]])))
    swing_disp_max = float(config.get("step_logic", {}).get("swing_xy_max", 0.35)) * 0.5
    z_tol_frac     = float(rcfg.get("z_success_tol_frac", 0.40))
    z_success_tol  = z_lock * z_tol_frac
    success_radius = float(rcfg.get("success_radius", 0.12))
    _min_steps     = int(config.get("cruise_rl", {}).get("min_steps_for_success", 5))
    _current_step  = getattr(env, 'current_step', 0)

    # [v13.1] cruise reward 用专用阈值, 不再借用 phase_transition 里更严的 cruise_to_descent_*
    # 用户反馈: expert 到达终点上方但没成功率 → success 标准应比 transition 标准更宽松,
    # 让 expert NMPC + (少量) RL 残差能稳定达成成功条件.
    tilt_max  = float(rcfg.get("success_tilt_max",        0.20))    # 之前 0.15
    swing_max = float(rcfg.get("success_swing_vel_max",   0.30))    # 之前 0.20
    vel_max   = float(rcfg.get("success_payload_vel_max", 0.35))    # 之前 0.25
    _hold_req = int(rcfg.get("success_hold_steps", 2))
    in_zone = (
        _current_step >= _min_steps and
        abs(payload_z - z_lock) < z_success_tol and
        dtf < success_radius and
        tilt < tilt_max and
        swing_vel < swing_max and
        swing_disp < swing_disp_max and
        pl_vel < vel_max)

    rstate.hold_counter = (rstate.hold_counter + 1) if in_zone else 0

    if rstate.hold_counter >= _hold_req:
        r_bonus = float(rcfg.get("success_bonus", 20.0))   # 8 → 20
        reward += r_bonus
        if tracker: tracker.add("success_bonus", r_bonus)
        success = True; done = True
        info["termination"] = (
            f"cruise_success:d={dtf*1000:.1f}mm,E={swing_energy*1000:.1f}mJ")

    # ── 超时 ──────────────────────────────────────────────────────────────────
    max_steps = int(config["cruise_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        info["termination"] = "timeout"

    # clip 范围放宽 (从 [-10, 15] → [-12, 25]), 让 success_bonus=20 不被截
    reward = float(np.clip(reward, -12.0, 25.0))
    return reward, done, success, info


# ==============================================================================
# Phase 3: Descent Reward
# ==============================================================================

class DescentRewardState:
    def __init__(self):
        self.prev_z   = None
        self.prev_dtf = None
        self.prev_insert_error = None
        self.insertion_hold_counter = 0
        self.total_steps_global = 0
        # 课程注入字段
        self.current_xy_range      = None
        self.current_xy_tol        = None
        self.current_descent_level = 0
        self.descent_n_levels      = 5
        self.steps_at_max_level    = 0
        # [v11.3] 残差 RL 用: action smoothness penalty
        self.prev_rl_action        = None
        # [v14.2] 早停追踪
        self._z_near_counter       = 0     # payload 在 target_z 附近的连续步数
        self._z_reached            = False  # 是否曾经到达 target_z 附近

    def reset(self):
        self.prev_z = None
        self.prev_dtf = None
        self.prev_insert_error = None
        self.insertion_hold_counter = 0
        self.prev_rl_action = None
        self._z_near_counter = 0
        self._z_reached = False


def compute_descent_reward(env, obs, config, rstate, tracker=None, rl_action=None):
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

    # Terminate unsafe episodes early. The task reward remains centered on
    # swing energy, cable energy, alignment, and success; residual intervention
    # is handled below as a small regularizer.
    unstable, reason = _check_instability(env, obs, config, grace_steps=20)
    if unstable:
        return 0.0, True, False, {"termination": reason}

    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz = float(env.data.qvel[dof_idx + 2])
    if payload_z < 0.03 and pl_vz < -0.3:
        return 0.0, True, False, {"termination": "crash"}

    # 1. Payload swing energy: KE + PE, bounded penalty.
    try:
        swing_energy, swing_ke, swing_pe, swing_angle = _get_swing_energy(env, obs, config)
    except Exception:
        swing_energy = swing_ke = swing_pe = swing_angle = 0.0
    e_thresh = float(rcfg.get("swing_energy_thresh", 0.008))
    e_coef   = float(rcfg.get("swing_energy_coef", 6.0))
    e_max    = float(rcfg.get("swing_energy_penalty_max", 0.40))
    r_swing_energy = -min(e_coef * max(0.0, swing_energy - e_thresh), e_max)
    reward += r_swing_energy
    if tracker:
        tracker.add("swing_energy_penalty", r_swing_energy)
        tracker.add("swing_energy_J", swing_energy)
        tracker.add("swing_ke_J", swing_ke)
        tracker.add("swing_pe_J", swing_pe)
        tracker.add("swing_angle_deg", float(np.degrees(swing_angle)))
        tracker.add("xy_dist_mm", dtf * 1000)
        tracker.add("z_dist_mm", (payload_z - target_pz) * 1000)

    # 2. Cable vibration energy penalty.
    cable_ke = _get_cable_kinetic_energy(env)
    ce_thresh = float(rcfg.get("cable_ke_thresh",      0.05))
    ce_coef   = float(rcfg.get("cable_ke_penalty_coef", 0.25))
    ce_max    = float(rcfg.get("cable_ke_penalty_max",  0.04))
    r_cable_ke = 0.0
    if cable_ke > ce_thresh:
        r_cable_ke = -min(ce_coef * (cable_ke - ce_thresh), ce_max)
        reward += r_cable_ke
    if tracker:
        tracker.add("cable_ke_penalty", r_cable_ke)
        tracker.add("cable_ke", cable_ke)

    cur_level   = getattr(rstate, 'current_descent_level', 0)
    cur_nlevels = getattr(rstate, 'descent_n_levels',      5)
    steps_at_max = getattr(rstate, 'steps_at_max_level',   0)
    xy_tol_final = float(cfg_ins.get("xy_tolerance_train_end", 0.005))

    _inj_tol = getattr(rstate, 'current_xy_tol', None)
    if _inj_tol is not None:
        xy_tol = float(_inj_tol)
        if cur_level >= cur_nlevels - 1:
            fine_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 500_000))
            fine_frac  = min(steps_at_max / max(fine_steps, 1), 1.0)
            xy_tol = xy_tol + fine_frac * (xy_tol_final - xy_tol)
    else:
        xy_tol = xy_tol_final

    total_ts = getattr(rstate, 'total_steps_global', 0)
    anneal_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 500_000))
    frac = min(total_ts / max(anneal_steps, 1), 1.0)

    tilt_tol = (float(cfg_ins.get("tilt_tolerance_train_start", 0.12)) +
                frac * (float(cfg_ins.get("tilt_tolerance_train_end", 0.05)) -
                        float(cfg_ins.get("tilt_tolerance_train_start", 0.12))))
    yaw_tol  = (float(cfg_ins.get("yaw_tolerance_train_start", 0.15)) +
                frac * (float(cfg_ins.get("yaw_tolerance_train_end", 0.08)) -
                        float(cfg_ins.get("yaw_tolerance_train_start", 0.15))))
    z_tol = float(cfg_ins.get("success_z_tolerance", 0.020))
    hold_steps = int(cfg_ins.get("hold_steps", 3))

    z_abs_err = abs(payload_z - target_pz)
    insert_error = float(np.sqrt(
        (dtf / max(xy_tol, 1e-6)) ** 2 +
        (z_abs_err / max(z_tol, 1e-6)) ** 2 +
        0.5 * (tilt / max(tilt_tol, 1e-6)) ** 2 +
        0.5 * (abs_yaw / max(yaw_tol, 1e-6)) ** 2))

    # 3. Alignment reward tied to the actual success tube. The previous broad
    # XY Gaussian paid timeout episodes almost as much as true insertions.
    xy_sigma = max(float(rcfg.get("alignment_xy_sigma_tol", 0.75)) * xy_tol, 1e-6)
    z_sigma = max(float(rcfg.get("alignment_z_sigma_tol", 0.75)) * z_tol, 1e-6)
    tilt_sigma = max(float(rcfg.get("alignment_tilt_sigma_tol", 0.90)) * tilt_tol, 1e-6)
    yaw_sigma = max(float(rcfg.get("alignment_yaw_sigma_tol", 0.90)) * yaw_tol, 1e-6)

    xy_score = float(np.exp(-0.5 * (dtf / xy_sigma) ** 2))
    z_score = float(np.exp(-0.5 * (z_abs_err / z_sigma) ** 2))
    tilt_score = float(np.exp(-0.5 * (tilt / tilt_sigma) ** 2))
    yaw_score = float(np.exp(-0.5 * (abs_yaw / yaw_sigma) ** 2))
    insert_score = xy_score * z_score * tilt_score * yaw_score
    r_proximity = float(rcfg.get("alignment_success_coef", 0.30)) * insert_score

    r_progress = 0.0
    if rstate.prev_insert_error is not None:
        delta_err = np.clip(
            rstate.prev_insert_error - insert_error,
            -float(rcfg.get("alignment_progress_clip", 0.25)),
            float(rcfg.get("alignment_progress_clip", 0.25)))
        r_progress = float(rcfg.get("alignment_progress_coef", 0.30)) * delta_err
    rstate.prev_insert_error = insert_error
    rstate.prev_dtf = dtf

    r_z_progress = 0.0
    if rstate.prev_z is not None:
        z_delta = max(0.0, rstate.prev_z - payload_z)
        if dtf < float(rcfg.get("alignment_z_gate", 0.030)) and payload_z > target_pz:
            r_z_progress = (float(rcfg.get("alignment_z_progress_coef", 4.0)) *
                            xy_score * z_score * z_delta)
    rstate.prev_z = payload_z

    r_height_penalty = -float(rcfg.get("alignment_height_penalty_coef", 0.04)) * (
        1.0 - z_score) * xy_score

    r_align = r_proximity + r_progress + r_z_progress + r_height_penalty
    reward += r_align
    if tracker:
        tracker.add("alignment_reward", r_align)
        tracker.add("alignment_proximity", r_proximity)
        tracker.add("alignment_insert_score", insert_score)
        tracker.add("alignment_z_score", z_score)
        tracker.add("alignment_tilt_score", tilt_score)
        tracker.add("alignment_yaw_score", yaw_score)
        tracker.add("insertion_error_norm", insert_error)
        tracker.add("alignment_height_penalty", r_height_penalty)
        tracker.add("xy_align_reward", r_progress)  # kept for HER relabeling
        tracker.add("z_descent_reward", r_z_progress)

    # Residual intervention regularizer. It is intentionally a soft band, not a
    # hard clamp, so the policy can still spend large residuals when insertion
    # accuracy needs them.
    r_action = 0.0
    r_smooth = 0.0
    action_rms = 0.0
    if rl_action is not None:
        a = np.asarray(rl_action, np.float32).reshape(-1)
        drl = config.get("descent_rl", {})
        scale = np.array([
            float(drl.get("residual_acc_max_xy", drl.get("acc_max_xy", 0.60))),
            float(drl.get("residual_acc_max_xy", drl.get("acc_max_xy", 0.60))),
            float(drl.get("residual_acc_max_z",  drl.get("acc_max_z",  0.90))),
        ], dtype=np.float32)[:len(a)]
        scale = np.maximum(scale, 1e-6)
        a_norm = np.clip(a / scale, -1.5, 1.5)
        action_rms = float(np.sqrt(np.mean(np.square(a_norm))))
        free = float(rcfg.get("action_rms_free", 0.30))
        excess = max(0.0, action_rms - free)
        r_action = -min(float(rcfg.get("action_magnitude_coef", 0.28)) * excess * excess,
                        float(rcfg.get("action_penalty_max", 0.08)))
        reward += r_action

        if rstate.prev_rl_action is not None:
            prev = np.asarray(rstate.prev_rl_action, np.float32).reshape(-1)[:len(a)]
            diff_norm = (a - prev) / scale
            smooth = float(np.mean(np.square(diff_norm)))
            r_smooth = -min(float(rcfg.get("action_smoothness_coef", 0.035)) * smooth,
                            float(rcfg.get("action_penalty_max", 0.08)))
            reward += r_smooth
        rstate.prev_rl_action = a.copy()
    if tracker:
        tracker.add("action_rms_norm", action_rms)
        tracker.add("action_magnitude_penalty", r_action)
        tracker.add("action_smoothness_penalty", r_smooth)

    # 4. Success reward.
    cur_level   = getattr(rstate, 'current_descent_level', 0)
    cur_nlevels = getattr(rstate, 'descent_n_levels',      5)
    steps_at_max = getattr(rstate, 'steps_at_max_level',   0)
    xy_tol_final = float(cfg_ins.get("xy_tolerance_train_end", 0.005))

    _inj_tol = getattr(rstate, 'current_xy_tol', None)
    if _inj_tol is not None:
        xy_tol = float(_inj_tol)
        if cur_level >= cur_nlevels - 1:
            fine_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 500_000))
            fine_frac  = min(steps_at_max / max(fine_steps, 1), 1.0)
            xy_tol = xy_tol + fine_frac * (xy_tol_final - xy_tol)
    else:
        xy_tol = xy_tol_final

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
    floor_contact_success = False
    if bool(cfg_ins.get("success_by_floor_contact", False)) and on_target:
        try:
            floor_contact_success = bool(env._check_prefab_floor_contact())
        except Exception:
            floor_contact_success = False

    rstate.insertion_hold_counter = (rstate.insertion_hold_counter + 1) if on_target else 0

    if floor_contact_success or rstate.insertion_hold_counter >= hold_steps:
        r_bonus = float(rcfg["success_bonus"])
        reward += r_bonus
        if tracker:
            tracker.add("success_bonus", r_bonus)
        success = True; done = True
        suffix = ",floor_contact=1" if floor_contact_success else ""
        info["termination"] = (
            f"insertion_success:z={payload_z*1000:.0f}mm,"
            f"dtf={dtf*1000:.1f}mm,tilt={tilt:.3f},yaw={abs_yaw:.3f},"
            f"xy_tol={xy_tol*1000:.1f}mm{suffix}")
        return reward, done, success, info

    def _failure_miss_penalty():
        miss_clip = float(rcfg.get("failure_miss_error_clip", 3.0))
        miss = min(insert_error, miss_clip)
        return -min(float(rcfg.get("failure_miss_penalty_coef", 10.0)) * miss,
                    float(rcfg.get("failure_miss_penalty_max", 25.0)))

    current_step = getattr(env, 'current_step', 0)
    max_steps = int(config["descent_rl"]["max_steps"])
    late_start_frac = float(rcfg.get("late_step_penalty_start_frac", 0.70))
    late_start = int(max_steps * np.clip(late_start_frac, 0.0, 0.95))
    r_late = 0.0
    if current_step >= late_start:
        denom = max(max_steps - late_start, 1)
        late_frac = np.clip((current_step - late_start + 1) / denom, 0.0, 1.0)
        r_late = -min(float(rcfg.get("late_step_penalty_coef", 0.02)) * late_frac,
                      float(rcfg.get("late_step_penalty_max", 0.05)))
        reward += r_late
    if tracker:
        tracker.add("late_step_penalty", r_late)

    # ── [v14.2] 早停: payload 已到达 z 但 xy 偏差太大 ────────────────────────
    # PID 控制下 payload 很快下降到 target_z 附近, 如果 xy 没对准,
    # 后续步数全在积累无效负 reward, 浪费训练预算且产生误导梯度
    _es_cfg = config.get("descent_rl", {})
    _es_enabled = bool(_es_cfg.get("early_stop_enabled", False))
    if _es_enabled and not done:
        _es_z_near    = float(_es_cfg.get("early_stop_z_near",    0.030))
        _es_xy_fail   = float(_es_cfg.get("early_stop_xy_fail",   0.025))
        _es_patience  = int(_es_cfg.get("early_stop_patience",    20))
        _es_penalty   = float(_es_cfg.get("early_stop_penalty",   -1.0))

        # 检查 payload 是否在 target_z 附近
        if abs(payload_z - target_pz) < _es_z_near:
            rstate._z_reached = True
            rstate._z_near_counter += 1
        elif rstate._z_reached:
            # payload 曾经到达但又升回去了 (异常)
            rstate._z_near_counter += 1

        # 到达 z 后超过 patience 步且 xy 偏差仍然太大 → 提前终止
        if (rstate._z_reached and
                rstate._z_near_counter >= _es_patience and
                dtf > _es_xy_fail):
            r_fail = _es_penalty + _failure_miss_penalty()
            reward += r_fail
            if tracker:
                tracker.add("early_stop_penalty", _es_penalty)
                tracker.add("failure_miss_penalty", r_fail - _es_penalty)
            done = True
            info["termination"] = (
                f"early_stop:dtf={dtf*1000:.1f}mm>xy_fail={_es_xy_fail*1000:.0f}mm,"
                f"z={payload_z*1000:.0f}mm,patience={rstate._z_near_counter}")
            return reward, done, success, info

    if current_step >= max_steps - 1 and not done:
        r_timeout = float(rcfg.get("timeout_penalty", 0.0))
        r_miss = _failure_miss_penalty()
        reward += r_timeout + r_miss
        if tracker:
            tracker.add("timeout_penalty", r_timeout)
            tracker.add("failure_miss_penalty", r_miss)
        done = True
        info["termination"] = (
            f"timeout:dtf={dtf*1000:.1f}mm,z={payload_z*1000:.0f}mm,"
            f"tilt={tilt:.3f},yaw={abs_yaw:.3f}")

    return reward, done, success, info


# ==============================================================================
# Reward 分项追踪器 (用于 wandb)
#
# 简化版: 每个 episode 汇总各分项的累积值和平均值, 训练循环负责上报.
# ==============================================================================

class RewardComponentTracker:
    """轻量分项 reward 追踪器, 每 episode 汇总累积值 + 平均值。
    [v11 Path 2] 增加 last-step value 跟踪, 供 PPO-HER relabel 用。
    """
    def __init__(self, phase):
        self.phase = phase
        self._sums  = {}
        self._last_step = {}   # [v11] 本步各项最新值, step() 会清空给下一步
        self._count = 0

    def add(self, name, value):
        if name not in self._sums:
            self._sums[name] = 0.0
        v = float(value)
        self._sums[name] += v
        self._last_step[name] = v
        # count 只在主 reward 项 (step_penalty) 累加, 防止重复
        # 这里取消 count 跟踪, 改为外部统计 episode 步数

    def step(self):
        """每 step 调用一次, 累计 step 计数。"""
        self._count += 1

    def get_last_step_value(self, name, default=0.0):
        """[v11] 取本步最新 add 的值; HER relabel 重算时调用。"""
        return self._last_step.get(name, default)

    def episode_summary(self):
        """返回 {key: total_or_per_step}, 用于 wandb logging。"""
        out = {}
        for k, v in self._sums.items():
            out[f"{self.phase}/rew/{k}"] = v
            if self._count > 0:
                out[f"{self.phase}/rew/{k}_per_step"] = v / max(self._count, 1)
        return out
