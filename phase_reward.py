# ==============================================================================
# phase_reward.py — 三阶段独立奖励函数 v2 (优化版)
#
# 主要变更:
#   [OPT-REW-C] Cruise reward v4:
#     - alive_bonus 0.08→0.02, step_penalty -0.002→-0.01 (净负值消除存活套利)
#     - pbrs_coef 10→20 (从 config 读取, 更强到达驱动)
#     - 新增 near_goal_bonus (距离 <0.10m 时持续正奖励)
#   [OPT-REW-D] Descent reward v3:
#     - z_descent 无条件权重从 0.3→0.5 (从 config 读取 z_unconditional_frac)
#     - align_factor 特征距离 0.08→0.12 (更宽松对准窗口)
#     - step_penalty -0.001→-0.005 (从 config 读取)
#     - 新增 near_target_bonus (xy+z 均接近时稳定奖励)
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
# Phase 1: Lift Reward (不变)
# ==============================================================================

class LiftRewardState:
    def __init__(self):
        self.prev_z = None

    def reset(self):
        self.prev_z = None


def compute_lift_reward(env, obs, config, rstate):
    """提升阶段奖励 (v1 不变)。"""
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
# Phase 2: Cruise Reward [v5 — 能量约束 + 残差RL架构适配]
# ==============================================================================

class CruiseRewardState:
    def __init__(self):
        self.prev_potential   = None
        self._near_given      = False
        self.total_steps_global = 0
        self.initial_dist     = None
        self.current_success_radius = 0.25
        # 能量监控
        self.prev_swing_energy = None

    def reset(self):
        self.prev_potential    = None
        self._near_given       = False
        self.initial_dist      = None
        self.prev_swing_energy = None


def _get_swing_energy_cruise(env, obs, config):
    """
    计算 cruise 段摆动能量 (KE + PE)。
    使用 compute_swing_energy 统一接口。
    """
    from ee_acc_controller import compute_swing_energy
    pl_pos = env.data.body('prefab').xpos.copy()
    ee_pos = env._get_ee_pos()
    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vel  = env.data.qvel[dof_idx:dof_idx+3].copy()
    ee_vel  = getattr(env, '_ee_vel_cache', np.zeros(3))

    mass   = float(env.config.get("prefab",     {}).get("mass", 1.0))
    rope_L = float(env.config.get("controller", {}).get("L",    0.5))
    energy, ke, pe, angle = compute_swing_energy(
        pl_pos, ee_pos, pl_vel, ee_vel, mass, rope_L)
    return energy, ke, pe, angle


def compute_cruise_reward(env, obs, config, rstate):
    """
    Cruise 段奖励 v5 (能量约束 + 残差RL架构适配)。

    设计原则:
    ─────────
    1. 只对摆动总能量 (KE + PE) 做约束, 不对相对位移/速度硬性惩罚
       - 允许 RL 利用 payload 惯性 (如: 借助摆动惯性辅助转向)
       - 能量约束 = 软上界 + 超阈值指数惩罚, 不直接强制 swing=0

    2. PBRS 驱动导航 (势能差分)
       - 仍用 -dist 势能, 保证最优策略不变性 (Ng 1999)

    3. 底层防摆控制器 (SwingDampingController) 负责物理阻尼
       - Reward 层无需重复惩罚已由控制器处理的低频摆动
       - 只惩罚超过能量阈值的「失控摆动」

    4. 去掉 alive_bonus, 保留轻微 step_penalty (净负值, 防存活套利)
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

    # ── [ENERGY] 摆动能量约束 ─────────────────────────────────────────────────
    # 只对能量做软约束, 不直接强制 swing=0
    # 控制器负责物理阻尼, reward 只惩罚「控制器也压不住」的剧烈摆动
    try:
        swing_energy, swing_ke, swing_pe, swing_angle = _get_swing_energy_cruise(
            env, obs, config)
    except Exception:
        swing_energy, swing_ke, swing_pe, swing_angle = 0., 0., 0., 0.

    energy_thresh = float(rcfg.get("swing_energy_thresh",  0.08))  # J, 正常摆动允许值
    energy_penalty_coef = float(rcfg.get("swing_energy_penalty_coef", 3.0))
    if swing_energy > energy_thresh:
        # 超阈值部分: 指数惩罚 (比线性更陡, 强烈反对失控摆动)
        excess = swing_energy - energy_thresh
        reward -= energy_penalty_coef * (excess + 0.5 * excess ** 2)
    reward = max(reward, -5.0)

    # ── PBRS 势能差分 (导航) ──────────────────────────────────────────────────
    gamma   = float(rcfg.get("pbrs_gamma",  0.99))
    k_pbrs  = float(rcfg.get("pbrs_coef",  10.0))
    current_potential = -dtf
    if rstate.prev_potential is not None:
        pbrs   = gamma * current_potential - rstate.prev_potential
        reward += k_pbrs * pbrs
    rstate.prev_potential = current_potential

    # ── 障碍物排斥势 [v6 线性化] ──────────────────────────────────────────────
    # [v6] 从指数排斥改为线性软排斥:
    #   旧: k*(1/d-1/d0)²*0.5 → d=5cm时penalty=60-300, 掩盖导航信号300倍
    #   新: k*max(0,1-d/d0)   → d=5cm时penalty≤0.5, 量级合理
    # 硬碰撞由 collision_penalty=-5 + episode终止负责
    d0    = float(rcfg.get("obs_repulse_d0",     0.10))
    k_obs = float(rcfg.get("obs_repulse_coef",   0.5))
    use_linear = bool(rcfg.get("obs_repulse_linear", True))
    for (ox, oy, orad) in env._obstacles:
        d = max(float(np.linalg.norm(pl_xy - np.array([ox, oy]))) - orad, 0.001)
        if d < d0:
            if use_linear:
                # 线性: 最大惩罚k_obs (d=0时), 线性衰减到0 (d=d0时)
                reward -= k_obs * (1.0 - d / d0)
            else:
                # 保留旧指数模式(兼容)
                reward -= k_obs * (1.0/d - 1.0/d0) ** 2 * 0.5
    reward = max(reward, -5.0)

    # ── Z 高度偏差 + Vz 惩罚 (PID 漂移辅助) ─────────────────────────────────
    z_dev = abs(payload_z - z_lock)
    reward -= float(rcfg.get("z_dev_coef", 1.0)) * min(z_dev, 0.15)

    dof_idx = env.model.jnt_dofadr[env.prefab_jnt_id]
    pl_vz   = float(env.data.qvel[dof_idx + 2])
    if pl_vz < 0:
        reward -= float(rcfg.get("vz_penalty_coef", 1.5)) * min(abs(pl_vz), 0.3)

    # ── step penalty (净负, 防存活套利) ──────────────────────────────────────
    reward += float(rcfg.get("step_penalty", -0.01))

    # ── 近目标连续稠密奖励 ────────────────────────────────────────────────────
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
    success_radius = getattr(rstate, 'current_success_radius', 0.25)
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
# Phase 3: Descent Reward [OPT-REW-D]
# ==============================================================================

class DescentRewardState:
    def __init__(self):
        self.prev_z = None
        self.prev_dtf = None
        self.insertion_hold_counter = 0
        self.total_steps_global = 0

    def reset(self):
        self.prev_z = None
        self.prev_dtf = None
        self.insertion_hold_counter = 0


def compute_descent_reward(env, obs, config, rstate):
    """
    Descent 段奖励 v4 (优先级修正: 稳定 > 对准 > 下降)。

    核心变更 vs v3:
      1. [优先级修正] z 无条件下降权重 0.5 → 0.05 (几乎禁用)
         z 下降奖励严格门控: 只在 dtf < 10mm 时激活
         防止 agent 在未对准时就下降 (任务逻辑倒置)

      2. [稳定优先] swing_ke_coef 权重提升 ×3, 加入 tilt_rate 惩罚
         稳定是插入成功的前提, 权重必须高于 xy_align

      3. [精度聚焦] xy 奖励改为高斯聚焦 (dtf < 15mm 以内才有强信号)
         5mm 精度要求需要聚焦激励, 差分奖励在大误差区间信噪比低

      4. [软成功修正] 超时仅保留 xy 分量, 去掉 z 分量
         z 维的软奖励会误导 agent 在未对准时下降

      5. [hold_steps] 提升到 5 步 (更严格的插入验证)
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

    # ══════════════════════════════════════════════════════════════════════════
    # [P1] 稳定性惩罚 (最高优先级, 权重最重)
    # ══════════════════════════════════════════════════════════════════════════
    # 摆动总能量惩罚 (KE, 简化版: 不依赖 ee_vel_cache 的鲁棒实现)
    ke = _get_swing_ke(env, obs)
    # [修正] 权重 ×3 vs 旧版, 明确体现稳定优先
    reward -= float(rcfg.get("swing_ke_coef", 3.0)) * min(ke, 0.25)

    # 姿态惩罚 (tilt + yaw)
    reward -= float(rcfg.get("tilt_coef", 1.5)) * min(tilt, 0.3)
    reward -= float(rcfg.get("yaw_coef",  0.5)) * min(abs_yaw, 0.3)

    # ══════════════════════════════════════════════════════════════════════════
    # [P2] XY 对准奖励 (中等优先级) [v9 重设计]
    # ══════════════════════════════════════════════════════════════════════════
    # [v9 BUG FIX] 去掉错误的 *10.0 系数
    # 原 xy_diff = delta_dtf * coef * 10.0 → 绳摆振荡300mm→100mm时给+9/step极端奖励
    # agent学会激励大幅绳摆振荡获得奖励，导致测试时dtf=100-380mm
    # [v9] 两阶段策略: z未到位时轻微xy修正，到位后全力对准
    # 防止下降过程中大幅横向运动激励绳摆旋转模式
    z_not_reached = abs(payload_z - target_pz) >= 0.025
    if rstate.prev_dtf is not None:
        delta_dtf_m = rstate.prev_dtf - dtf   # 正=靠近(m)
        # clip: 单步最大±10mm，防止大振荡给极端reward
        delta_dtf_clipped = np.clip(delta_dtf_m, -0.010, 0.010)
        if z_not_reached:
            # z未到位: 轻微xy奖励，主要任务是稳定下降
            xy_diff = delta_dtf_clipped * float(rcfg.get("xy_align_coef", 4.0)) * 0.5
        else:
            # z已到位: 正常xy对准奖励
            xy_diff = delta_dtf_clipped * float(rcfg.get("xy_align_coef", 4.0))
        reward += xy_diff
    # 高斯聚焦奖励: 只在z已到位时生效，防止下降阶段横向激励
    if not z_not_reached:
        sigma_xy = float(rcfg.get("xy_gauss_sigma", 0.020))
        reward += float(rcfg.get("xy_gauss_coef", 1.5)) * np.exp(
            -0.5 * (dtf / sigma_xy) ** 2)
    rstate.prev_dtf = dtf

    # ══════════════════════════════════════════════════════════════════════════
    # [P3] Z 下降奖励 (最低优先级, 严格门控)
    # ══════════════════════════════════════════════════════════════════════════
    if rstate.prev_z is not None:
        z_delta = rstate.prev_z - payload_z  # 向下为正 (下降 > 0)

        # [v5] 关键修复: 明确惩罚上升行为
        # 测试日志显示 payload 从 250mm 上升到 600mm+ — agent 学到了「上升」
        # 这是因为之前下降激励太弱（z_unconditional_frac=0.05）
        # 现在双管齐下：增加下降奖励 + 惩罚上升行为
        if z_delta < 0 and payload_z < 0.60:  # 上升行为且未超过安全高度
            z_rise = abs(z_delta)  # 上升量 (正值)
            rise_penalty = float(rcfg.get("z_rise_penalty_coef", 5.0)) * z_rise
            rise_penalty = min(rise_penalty, float(rcfg.get("z_rise_max_penalty", 0.3)))
            reward -= rise_penalty

        # [FIX v4.1] 门控放宽: dtf < 30mm 时才激活条件下降奖励
        # 原门控 10mm 过严 — 在实际训练中 agent 几乎永远无法触发此奖励
        # 造成 z 轴完全没有学习信号, 表现为: 有下降趋势但精度不足
        xy_aligned = dtf < float(rcfg.get("z_descent_xy_gate", 0.030))
        if xy_aligned and z_delta > 0 and payload_z > target_pz:
            reward += float(rcfg.get("z_descent_coef", 10.0)) * z_delta

        # [v5] 无条件下降激励提升 0.05→0.25: agent必须主动下降才能获得足够收益
        z_unconditional_frac = float(rcfg.get("z_unconditional_frac", 0.25))
        if z_delta > 0 and payload_z > target_pz:
            reward += float(rcfg.get("z_descent_coef", 10.0)) * z_delta * z_unconditional_frac
    rstate.prev_z = payload_z

    # ── step penalty ──────────────────────────────────────────────────────────
    reward += float(rcfg.get("step_penalty", -0.005))

    # ── 近目标稠密奖励 [v7 重设计] ─────────────────────────────────────────────
    # 核心问题：z到位(120mm)后agent停止xy运动，dtf停在10-20mm不再收敛
    # 原因：z到位后z_descent奖励=0，而xy对准奖励相对step_penalty不够强
    # 解决：z到位后专门给予持续的精细对准奖励，使agent有强动力完成最后一公里
    z_reached = abs(payload_z - target_pz) < 0.025  # z已到达目标区域(±25mm)

    if z_reached:
        # [v7] z已到位时，精细xy对准是唯一目标，给予强密集奖励
        # 奖励与dtf成反比：距离越小，奖励越大，形成吸引势阱
        precision_bonus = float(rcfg.get("precision_coef", 3.0)) * max(0.0, 1.0 - dtf / 0.030)
        # 额外：dtf<10mm时叠加高精度奖励
        if dtf < 0.010:
            precision_bonus += float(rcfg.get("near_target_bonus", 1.0)) * (1.0 - dtf / 0.010)
        reward += precision_bonus
        # [v7] z到位后step_penalty减半，降低agent保持不动的代价
        # 原step_penalty=-0.005强迫agent快速行动，但在精细对准阶段是错误的
        reward += 0.0025  # 补回一半step_penalty，使精细对准的净cost更小
    else:
        # z未到位时，保持原有的near_target逻辑
        xy_near = dtf < 0.010
        z_near  = abs(payload_z - target_pz) < 0.020
        stable  = ke < 0.02 and tilt < 0.05
        if xy_near and z_near and stable:
            reward += float(rcfg.get("near_target_bonus", 1.0))
        elif xy_near:
            reward += float(rcfg.get("near_target_bonus", 1.0)) * 0.3

    # ── 成功判定 [v7b 关键修复: 容差与课程level动态耦合] ─────────────────────
    # 根本问题: xy_tolerance基于全局时间退化, 与descent课程level完全脱节
    # 症状: level2(xy_range=17.5mm)时xy_tol=14mm → 成功在物理上不可能 → SR骤降0
    # 修复: xy_tol动态跟随当前level的init_xy_range, 最高level后才做精细退化
    cur_descent_xy_range  = getattr(rstate, 'current_xy_range',     None)
    cur_descent_level     = getattr(rstate, 'current_descent_level', 0)
    cur_descent_n_levels  = getattr(rstate, 'descent_n_levels',      5)
    steps_at_max_level    = getattr(rstate, 'steps_at_max_level',    0)

    xy_tol_final = float(cfg_ins.get("xy_tolerance_train_end", 0.005))

    if cur_descent_xy_range is not None:
        # level联动容差: 容差 = init_xy_range * multiplier (确保成功物理可达)
        tol_mult  = float(cfg_ins.get("xy_tol_range_multiplier", 2.0))
        xy_tol    = max(cur_descent_xy_range * tol_mult, xy_tol_final)
        # 到达最高level后, 对容差做进一步精细退化
        if cur_descent_level >= cur_descent_n_levels - 1:
            fine_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 80_000))
            fine_frac  = min(steps_at_max_level / max(fine_steps, 1), 1.0)
            xy_tol     = xy_tol + fine_frac * (xy_tol_final - xy_tol)
    else:
        # 兼容回退: 使用全局时间退化
        total_ts     = getattr(rstate, 'total_steps_global', 0)
        anneal_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 80_000))
        frac         = min(total_ts / max(anneal_steps, 1), 1.0)
        xy_tol = (float(cfg_ins.get("xy_tolerance_train_start", 0.015)) +
                  frac * (xy_tol_final - float(cfg_ins.get("xy_tolerance_train_start", 0.015))))

    total_ts = getattr(rstate, 'total_steps_global', 0)
    anneal_steps = int(cfg_ins.get("xy_tolerance_anneal_steps", 80_000))
    frac = min(total_ts / max(anneal_steps, 1), 1.0)

    tilt_tol = (float(cfg_ins.get("tilt_tolerance_train_start", 0.10)) +
                frac * (float(cfg_ins.get("tilt_tolerance_train_end",   0.05)) -
                        float(cfg_ins.get("tilt_tolerance_train_start", 0.10))))

    yaw_tol = (float(cfg_ins.get("yaw_tolerance_train_start", 0.12)) +
               frac * (float(cfg_ins.get("yaw_tolerance_train_end",   0.08)) -
                       float(cfg_ins.get("yaw_tolerance_train_start", 0.12))))

    z_tol      = float(cfg_ins.get("success_z_tolerance", 0.020))
    # [修正] hold_steps 3 → 5: 更严格的插入验证
    hold_steps = int(cfg_ins.get("hold_steps", 5))

    on_target = (abs(payload_z - target_pz) < z_tol and
                 dtf < xy_tol and
                 tilt < tilt_tol and abs_yaw < yaw_tol)

    if on_target:
        rstate.insertion_hold_counter += 1
    else:
        rstate.insertion_hold_counter = 0

    if rstate.insertion_hold_counter >= hold_steps:
        reward  += float(rcfg["success_bonus"])
        success  = True
        done     = True
        info["termination"] = (
            f"insertion_success:z={payload_z*1000:.0f}mm,"
            f"dtf={dtf*1000:.1f}mm,tilt={tilt:.3f},yaw={abs_yaw:.3f},"
            f"tols=xy{xy_tol*1000:.1f}mm/tilt{tilt_tol:.3f}/yaw{yaw_tol:.3f}")
        return reward, done, success, info

    # ── 超时 → 软成功 (只保留 xy 分量) ──────────────────────────────────────
    max_steps = int(config["descent_rl"]["max_steps"])
    if getattr(env, 'current_step', 0) >= max_steps - 1 and not done:
        done = True
        # [修正] 只保留 dist 和 pose 分量, 去掉 z 分量
        # z 分量的软奖励会误导 agent 在未对准时下降
        dist_frac = max(0.0, 1.0 - dtf / 0.05)
        pose_frac = max(0.0, 1.0 - float(np.sqrt(tilt**2 + abs_yaw**2)) / 0.3)
        combined  = 0.6 * dist_frac + 0.4 * pose_frac
        reward   += float(rcfg.get("soft_success_bonus", 25.0)) * combined
        info["termination"] = (
            f"timeout:soft={combined:.2f},"
            f"dtf={dtf*1000:.1f}mm,z={payload_z*1000:.0f}mm")

    return reward, done, success, info