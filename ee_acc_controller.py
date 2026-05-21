# ==============================================================================
# ee_acc_controller.py — EE 加速度控制器 (v4)
#
# v4 变更:
#   1. 新增 SwingDampingController: cruise 段底层防摆控制器
#      - 基于能量耗散原理的比例阻尼法
#      - 可单独验证: RL输出=0时检验摆动衰减曲线
#      - 与 RL 残差叠加, 前后效果均可独立量化
#
#   2. 新增 compute_swing_energy: 统一摆动能量计算 (KE + PE)
#      用于 reward 设计和监控, 替代仅用 KE 的旧方案
#
#   3. EEAccController.compute_delta_q 新增 base_acc_xy / residual_mode 参数
#      total_acc = base_acc (底层防摆) + residual_acc (RL 残差)
#      两者独立限幅后叠加, 向下完全兼容旧调用
#
#   4. CruiseZYawPID 保持 v3 不变
# ==============================================================================

import numpy as np


# ==============================================================================
# 工具: 摆动能量
# ==============================================================================

def compute_swing_energy(pl_pos, ee_pos, pl_vel, ee_vel, mass, rope_L, g=9.81):
    """
    计算 payload 摆动总机械能 (动能 + 势能)。

    KE = 0.5 * m * |v_pl_xy - v_ee_xy|^2
    PE = m * g * L * (1 - cos(theta)),  theta = arctan(|offset_xy| / L)

    Returns:
        (energy_total, ke, pe, swing_angle_rad)
    """
    pl_pos = np.asarray(pl_pos, dtype=np.float64)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    pl_vel = np.asarray(pl_vel, dtype=np.float64)
    ee_vel = np.asarray(ee_vel, dtype=np.float64)

    offset_xy  = pl_pos[:2] - ee_pos[:2]
    rel_vel_xy = pl_vel[:2] - ee_vel[:2]

    dist_xy     = float(np.linalg.norm(offset_xy))
    swing_angle = np.arctan2(dist_xy, max(float(rope_L), 0.01))

    ke = 0.5 * float(mass) * float(np.dot(rel_vel_xy, rel_vel_xy))
    pe = float(mass) * g * float(rope_L) * (1.0 - np.cos(swing_angle))

    return float(ke + pe), float(ke), float(pe), float(swing_angle)


# ==============================================================================
# 底层防摆控制器 (cruise 专用)
# ==============================================================================

class SwingDampingController:
    """
    Cruise 段底层防摆控制器 v1。

    控制律:
        acc_damp = kd_vel * (v_pl - v_ee)_xy   # 速度阻尼 (主)
                 + kp_pos * (p_pl - p_ee)_xy    # 位移恢复 (辅, kp_pos 默认小)

    物理直觉:
        EE 跟随 payload 速度方向运动 → 等效为摆动系统引入粘性阻尼 → 耗散动能
        EE 轻微朝 payload 位移方向移动 → 减小摆角 → 耗散势能

    自适应增益:
        摆动能量越大 → gain_scale 越大 → 阻尼越强

    独立验证方法:
        令 RL 残差输出 = 0, 仅运行此控制器, 观察:
        1. swing_energy 是否单调下降 (有阻尼振荡)
        2. 摆角是否在数步内收敛到 < 2°

    config 字段 (cruise_swing_damping):
        kd_vel:         速度阻尼增益     (default 2.0)
        kp_pos:         位移恢复增益     (default 0.3)
        acc_max:        最大防摆修正量   (default 0.4 m/s^2)
        adaptive:       自适应增益开关   (default True)
        energy_ref:     参考能量阈值(J)  (default 0.05)
        gain_max_scale: 最大增益倍数     (default 2.5)
    """

    def __init__(self, config):
        cfg  = config.get("cruise_swing_damping", {})
        ec   = config.get("ee_control", {})

        self.kd_vel         = float(cfg.get("kd_vel",         2.0))
        self.kp_pos         = float(cfg.get("kp_pos",         0.3))
        self.acc_max        = float(cfg.get("acc_max",         0.4))
        self.adaptive       = bool(cfg.get("adaptive",         True))
        self.energy_ref     = float(cfg.get("energy_ref",      0.05))
        self.gain_max_scale = float(cfg.get("gain_max_scale",  2.5))

        self.mass   = float(config.get("prefab",     {}).get("mass", 1.0))
        self.rope_L = float(config.get("controller", {}).get("L",    0.5))

        # 监控缓存
        self._energy      = 0.0
        self._ke          = 0.0
        self._pe          = 0.0
        self._angle_deg   = 0.0
        self._gain_scale  = 1.0
        self._acc_norm    = 0.0

    def reset(self):
        self._energy     = 0.0
        self._ke         = 0.0
        self._pe         = 0.0
        self._angle_deg  = 0.0
        self._gain_scale = 1.0
        self._acc_norm   = 0.0

    def compute(self, pl_pos, ee_pos, pl_vel, ee_vel):
        """
        Returns:
            acc_damp (np.float32, shape (2,)): EE 防摆修正加速度 (ax, ay)
            info dict: 用于日志/reward 的监控量
        """
        pl_pos = np.asarray(pl_pos, dtype=np.float64)
        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        pl_vel = np.asarray(pl_vel, dtype=np.float64)
        ee_vel = np.asarray(ee_vel, dtype=np.float64)

        offset_xy  = pl_pos[:2] - ee_pos[:2]
        rel_vel_xy = pl_vel[:2] - ee_vel[:2]

        # 摆动能量
        energy, ke, pe, angle = compute_swing_energy(
            pl_pos, ee_pos, pl_vel, ee_vel, self.mass, self.rope_L)
        self._energy    = energy
        self._ke        = ke
        self._pe        = pe
        self._angle_deg = float(np.degrees(angle))

        # 自适应增益
        gain_scale = 1.0
        if self.adaptive and energy > 1e-6:
            gain_scale = min(
                1.0 + energy / max(self.energy_ref, 1e-8),
                self.gain_max_scale)
        self._gain_scale = gain_scale

        # 控制律
        acc_damp = self.kd_vel * gain_scale * rel_vel_xy
        if self.kp_pos > 0:
            acc_damp += self.kp_pos * gain_scale * offset_xy

        # 限幅
        norm = float(np.linalg.norm(acc_damp))
        if norm > self.acc_max and norm > 1e-8:
            acc_damp *= self.acc_max / norm
        self._acc_norm = float(np.linalg.norm(acc_damp))

        info = {
            "damp/energy":     energy,
            "damp/ke":         ke,
            "damp/pe":         pe,
            "damp/angle_deg":  self._angle_deg,
            "damp/gain_scale": gain_scale,
            "damp/acc_norm":   self._acc_norm,
        }
        return acc_damp.astype(np.float32), info

    @property
    def last_energy(self):
        return self._energy

    @property
    def last_angle_deg(self):
        return self._angle_deg


# ==============================================================================
# CruiseZYawPID (v3, 不变)
# ==============================================================================

class CruiseZYawPID:
    """巡航段 Z/Yaw 解耦 PID 控制器 (v3)。"""

    def __init__(self, config):
        ee_cfg  = config.get("ee_control", {})
        pid_cfg = config.get("cruise_z_pid", {})

        # [BUGFIX-CRITICAL] PID 控制 payload 高度, 输出 EE 修正量
        # z_target 是 payload 目标高度 (0.25m), 不是 EE 高度
        # EE 的绝对高度 = payload_z + rope_L, 由 EEAccController.lock_z 负责
        self.z_target = float(config.get("cruise_rl", {}).get(
            "z_lock_height", config["planning"]["payload_z_cruise"]))

        self.kp_z            = float(pid_cfg.get("kp_z",            2.0))
        self.ki_z            = float(pid_cfg.get("ki_z",            0.5))
        self.kd_z            = float(pid_cfg.get("kd_z",            0.3))
        self.z_integral_max  = float(pid_cfg.get("z_integral_max",  0.05))
        self.z_correction_max = float(pid_cfg.get("z_correction_max", 0.08))

        self.kp_yaw             = float(pid_cfg.get("kp_yaw",             1.0))
        self.kd_yaw             = float(pid_cfg.get("kd_yaw",             0.2))
        self.yaw_correction_max = float(pid_cfg.get("yaw_correction_max", 0.3))

        self.floor_z_threshold  = float(pid_cfg.get("floor_z_threshold",  0.04))
        self.floor_vz_threshold = float(pid_cfg.get("floor_vz_threshold", -0.15))

        self._z_integral   = 0.0
        self._prev_z_error = 0.0
        self._prev_yaw     = 0.0
        self._dt = float(ee_cfg.get("integrator_dt", 0.1))

    def reset(self, payload_z, payload_yaw=0.0):
        self._z_integral   = 0.0
        self._prev_z_error = float(payload_z) - self.z_target
        self._prev_yaw     = float(payload_yaw)

    def compute(self, payload_z, payload_vz, payload_yaw, payload_yaw_rate):
        dt      = self._dt
        z_error = self.z_target - float(payload_z)
        self._z_integral += z_error * dt
        self._z_integral  = np.clip(
            self._z_integral, -self.z_integral_max, self.z_integral_max)
        z_deriv = -float(payload_vz)
        ee_z_correction = np.clip(
            self.kp_z * z_error + self.ki_z * self._z_integral + self.kd_z * z_deriv,
            -self.z_correction_max, self.z_correction_max)
        self._prev_z_error = z_error

        yaw_error  = -float(payload_yaw)
        yaw_deriv  = -float(payload_yaw_rate)
        target_yaw = np.clip(
            self.kp_yaw * yaw_error + self.kd_yaw * yaw_deriv,
            -self.yaw_correction_max, self.yaw_correction_max)
        self._prev_yaw = float(payload_yaw)

        payload_falling = (
            float(payload_z) < self.floor_z_threshold and
            float(payload_vz) < self.floor_vz_threshold)

        return float(ee_z_correction), float(target_yaw), payload_falling

    def check_floor_collision(self, payload_z):
        return float(payload_z) < self.floor_z_threshold


# ==============================================================================
# EEAccController (v4)
# ==============================================================================

class EEAccController:
    """
    EE 加速度控制器 v4。

    cruise 模式 (residual_mode=True):
        total_acc_xy = base_acc_xy (SwingDampingController) + residual_acc_xy (RL)
        RL 只需学习「路径规划 + 精调」的残差部分, 防摆由底层保底

    其他模式 (旧行为, 向下兼容):
        total_acc = acc_3d (完整输出)
    """

    def __init__(self, config, ik_solver):
        ee_cfg = config.get("ee_control", {})
        sp     = config["space"]

        self.dt         = float(ee_cfg.get("integrator_dt", 0.1))
        self.vel_max_xy = float(ee_cfg.get("vel_max_xy",    0.3))
        self.vel_max_z  = float(ee_cfg.get("vel_max_z",     0.3))

        self.ik_solver = ik_solver
        self.dq_max  = np.array(sp.get("dq_max", [0.1]*7),    dtype=np.float64)
        self.q_low   = np.array(sp["action_space_low"],        dtype=np.float64)
        self.q_high  = np.array(sp["action_space_high"],       dtype=np.float64)

        self._ee_pos       = np.zeros(3, np.float64)
        self._ee_vel       = np.zeros(3, np.float64)
        self._last_q       = None
        # [v12 fix] 锚定强度从 config 读取 (与 JointSpaceExpert 同步, 默认 0.10)
        # 之前硬编码 0.08, 与 expert 的 0.10 不同 → train vs test 行为不一致.
        self._anchor_alpha = float(ee_cfg.get("anchor_alpha", 0.10))

        # [BUGFIX-CRITICAL] 存储绳长, 用于 lock_z 时正确计算 EE 目标高度
        rope_cfg = config.get("rope", {})
        n_seg   = int(rope_cfg.get("num_segments", 10))
        seg_len = float(rope_cfg.get("segment_length", 0.04))
        hook    = float(rope_cfg.get("hook_offset", 0.05))
        self._rope_L = n_seg * seg_len + hook  # 实际绳长 (默认 0.45m)

        # cruise 残差上限: RL 残差部分的独立限幅
        _cr  = config.get("cruise_rl", {})
        _axy = float(ee_cfg.get("acc_max_xy", 0.8))
        self._residual_acc_max_xy = float(
            _cr.get("residual_acc_max_xy", _axy * 0.5))

    def reset(self, ee_pos, current_q):
        self._ee_pos = ee_pos.astype(np.float64).copy()
        self._ee_vel = np.zeros(3, np.float64)
        self._last_q = current_q.astype(np.float64).copy()

    def compute_delta_q(self, acc_3d, current_q, real_ee_pos,
                        vel_max_xy=None, vel_max_z=None,
                        lock_z=False, z_lock_height=None,
                        z_pid_correction=0.0, target_yaw=0.0,
                        base_acc_xy=None, residual_mode=False):
        """
        将加速度指令转换为 delta_q。

        新增参数 (v4):
            base_acc_xy (array-like, shape (2,)):
                底层防摆控制器输出。仅在 residual_mode=True 时生效。
            residual_mode (bool):
                True  → acc_3d[:2] 为 RL 残差, 与 base_acc_xy 叠加
                False → acc_3d 为完整指令 (旧行为)
        """
        acc     = np.asarray(acc_3d, dtype=np.float64)
        vmax_xy = vel_max_xy if vel_max_xy is not None else self.vel_max_xy
        vmax_z  = vel_max_z  if vel_max_z  is not None else self.vel_max_z

        # ── 残差叠加 ──────────────────────────────────────────────────────────
        if residual_mode and base_acc_xy is not None:
            base = np.asarray(base_acc_xy, dtype=np.float64)
            # RL 残差独立限幅
            res  = acc[:2].copy()
            rnorm = float(np.linalg.norm(res))
            if rnorm > self._residual_acc_max_xy and rnorm > 1e-8:
                res *= self._residual_acc_max_xy / rnorm
            acc = np.array([base[0] + res[0], base[1] + res[1], acc[2]],
                           dtype=np.float64)

        # ── 积分 ──────────────────────────────────────────────────────────────
        self._ee_vel += acc * self.dt
        vxy = float(np.linalg.norm(self._ee_vel[:2]))
        if vxy > vmax_xy and vxy > 1e-8:
            self._ee_vel[:2] *= vmax_xy / vxy
        self._ee_vel[2] = np.clip(self._ee_vel[2], -vmax_z, vmax_z)
        self._ee_pos   += self._ee_vel * self.dt

        # ── Z 锁定 ────────────────────────────────────────────────────────────
        if lock_z and z_lock_height is not None:
            # [BUGFIX-CRITICAL] z_lock_height 是 payload 目标高度.
            # EE 必须在 payload 上方 rope_L 处. 之前直接用 payload_z=0.25,
            # 导致 EE 与 payload 处于同一高度, 绳子张力为零, payload 立即坠地.
            ee_z_target = z_lock_height + self._rope_L + z_pid_correction
            self._ee_pos[2] = ee_z_target
            self._ee_vel[2] = 0.0

        # ── 软锚定 ────────────────────────────────────────────────────────────
        alpha = self._anchor_alpha
        ree   = real_ee_pos.astype(np.float64)
        if lock_z:
            self._ee_pos[:2] = (1 - alpha) * self._ee_pos[:2] + alpha * ree[:2]
        else:
            self._ee_pos     = (1 - alpha) * self._ee_pos     + alpha * ree

        # ── 工作空间约束 ──────────────────────────────────────────────────────
        dxy = float(np.linalg.norm(self._ee_pos[:2]))
        if dxy > 0.48:
            self._ee_pos[:2] *= 0.48 / max(dxy, 1e-6)
        self._ee_pos[2] = max(self._ee_pos[2], 0.02)

        # ── IK ────────────────────────────────────────────────────────────────
        q_start = (self._last_q if self._last_q is not None
                   else current_q.astype(np.float64))
        q_target = self.ik_solver.solve_4d(
            current_q=q_start,
            target_x=float(self._ee_pos[0]),
            target_y=float(self._ee_pos[1]),
            target_z=float(self._ee_pos[2]),
            target_yaw=float(target_yaw),
        )
        if q_target is None or np.any(np.isnan(q_target)):
            q_target = current_q.astype(np.float64).copy()

        q_target     = np.clip(q_target, self.q_low, self.q_high)
        self._last_q = q_target.copy()

        delta_q = np.clip(
            q_target - current_q.astype(np.float64),
            -self.dq_max, self.dq_max)
        return delta_q.astype(np.float32)