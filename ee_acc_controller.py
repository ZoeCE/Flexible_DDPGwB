# ==============================================================================
# ee_acc_controller.py — EE 加速度控制器 (v3)
#
# v3 变更:
#   1. 新增 CruiseZYawPID: 环境端 PID 控制器, 用于巡航段解耦控制
#      - Z 轴: 读取真实 payload_z, PID 输出 EE z 速度修正
#      - Yaw 轴: 读取真实 payload_yaw, PID 输出 EE yaw 目标修正
#   2. compute_delta_q 增加 z_pid_correction / target_yaw 参数
#   3. 软锚定在 z-lock 模式下不再作用于 z 分量
#   4. 新增 payload 坠落检测
# ==============================================================================

import numpy as np


class CruiseZYawPID:
    """
    巡航段 Z 轴 / Yaw 轴 PID 控制器。

    解耦设计:
      - RL agent 只控制 EE 的 xy 加速度 (路径规划 + 防摆)
      - 本 PID 控制器独立维持:
          1. Payload z ≈ z_cruise (通过调整 EE z 位置)
          2. Payload yaw ≈ 0 (通过调整 IK 的 target_yaw)

    物理原理:
      - Payload 挂在绳索下方, EE z 变化会间接影响 payload z
      - EE 抬高 → 绳索拉紧 → payload z 升高; 反之亦然
      - 因此 PID 输出 ee_z_correction 来补偿 payload z 的偏移
    """

    def __init__(self, config):
        ee_cfg = config.get("ee_control", {})
        pid_cfg = config.get("cruise_z_pid", {})

        self.z_target = float(config.get("cruise_rl", {}).get(
            "z_lock_height", config["planning"]["payload_z_cruise"]))

        # Z-axis PID gains
        self.kp_z = float(pid_cfg.get("kp_z", 2.0))
        self.ki_z = float(pid_cfg.get("ki_z", 0.5))
        self.kd_z = float(pid_cfg.get("kd_z", 0.3))
        self.z_integral_max = float(pid_cfg.get("z_integral_max", 0.05))
        self.z_correction_max = float(pid_cfg.get("z_correction_max", 0.08))

        # Yaw-axis PD gains
        self.kp_yaw = float(pid_cfg.get("kp_yaw", 1.0))
        self.kd_yaw = float(pid_cfg.get("kd_yaw", 0.2))
        self.yaw_correction_max = float(pid_cfg.get("yaw_correction_max", 0.3))

        # Floor collision detection
        self.floor_z_threshold = float(pid_cfg.get("floor_z_threshold", 0.04))
        self.floor_vz_threshold = float(pid_cfg.get("floor_vz_threshold", -0.15))

        # Internal state
        self._z_integral = 0.0
        self._prev_z_error = 0.0
        self._prev_yaw = 0.0
        self._dt = float(ee_cfg.get("integrator_dt", 0.1))

    def reset(self, payload_z, payload_yaw=0.0):
        """重置 PID 内部状态。"""
        self._z_integral = 0.0
        self._prev_z_error = float(payload_z) - self.z_target
        self._prev_yaw = float(payload_yaw)

    def compute(self, payload_z, payload_vz, payload_yaw, payload_yaw_rate):
        """
        计算 Z 轴和 Yaw 轴的修正量。

        Returns:
            ee_z_correction: EE z 位置修正 (加到 EE 目标 z 上)
            target_yaw: IK 的 yaw 目标
            payload_falling: 是否检测到 payload 坠落
        """
        dt = self._dt

        # ── Z-axis PID ──
        z_error = self.z_target - float(payload_z)  # 正 = 需要升高

        # 积分项 (anti-windup)
        self._z_integral += z_error * dt
        self._z_integral = np.clip(
            self._z_integral, -self.z_integral_max, self.z_integral_max)

        # 微分项 (用速度, 更平滑)
        z_deriv = -float(payload_vz)  # payload 下降时 vz<0, deriv 应为正

        ee_z_correction = (
            self.kp_z * z_error +
            self.ki_z * self._z_integral +
            self.kd_z * z_deriv
        )
        ee_z_correction = np.clip(
            ee_z_correction, -self.z_correction_max, self.z_correction_max)

        self._prev_z_error = z_error

        # ── Yaw-axis PD ──
        yaw_error = 0.0 - float(payload_yaw)  # 目标 yaw = 0
        yaw_deriv = -float(payload_yaw_rate)

        target_yaw = self.kp_yaw * yaw_error + self.kd_yaw * yaw_deriv
        target_yaw = np.clip(
            target_yaw, -self.yaw_correction_max, self.yaw_correction_max)

        self._prev_yaw = float(payload_yaw)

        # ── 坠落检测 ──
        payload_falling = (
            float(payload_z) < self.floor_z_threshold and
            float(payload_vz) < self.floor_vz_threshold
        )

        return float(ee_z_correction), float(target_yaw), payload_falling

    def check_floor_collision(self, payload_z):
        """简单地面碰撞检测。"""
        return float(payload_z) < self.floor_z_threshold


class EEAccController:
    """
    EE 加速度控制器。

    每步:
      ee_vel += acc * dt
      ee_pos += ee_vel * dt
      q_target = IK(ee_pos)
      delta_q = clip(q_target - q_current, dq_max)
    """

    def __init__(self, config, ik_solver):
        ee_cfg = config.get("ee_control", {})
        sp = config["space"]

        self.dt = float(ee_cfg.get("integrator_dt", 0.1))
        self.vel_max_xy = float(ee_cfg.get("vel_max_xy", 0.3))
        self.vel_max_z = float(ee_cfg.get("vel_max_z", 0.3))

        self.ik_solver = ik_solver
        self.dq_max = np.array(sp.get("dq_max", [0.1]*7), dtype=np.float64)
        self.q_low = np.array(sp["action_space_low"], dtype=np.float64)
        self.q_high = np.array(sp["action_space_high"], dtype=np.float64)

        # 内部状态
        self._ee_pos = np.zeros(3, np.float64)
        self._ee_vel = np.zeros(3, np.float64)
        self._last_q = None

        # 锚定参数
        self._anchor_alpha = 0.08

    def reset(self, ee_pos, current_q):
        """重置内部状态。"""
        self._ee_pos = ee_pos.astype(np.float64).copy()
        self._ee_vel = np.zeros(3, np.float64)
        self._last_q = current_q.astype(np.float64).copy()

    def compute_delta_q(self, acc_3d, current_q, real_ee_pos,
                         vel_max_xy=None, vel_max_z=None,
                         lock_z=False, z_lock_height=None,
                         z_pid_correction=0.0, target_yaw=0.0):
        """
        将 3D 加速度转换为 delta_q。

        新增参数:
            z_pid_correction: 来自 CruiseZYawPID 的 z 修正量
            target_yaw: IK 目标 yaw (来自 PID 或固定值)
        """
        acc = np.asarray(acc_3d, dtype=np.float64)
        dt = self.dt
        vmax_xy = vel_max_xy if vel_max_xy is not None else self.vel_max_xy
        vmax_z = vel_max_z if vel_max_z is not None else self.vel_max_z

        # 积分
        self._ee_vel += acc * dt

        # 速度限幅
        vxy = np.linalg.norm(self._ee_vel[:2])
        if vxy > vmax_xy and vxy > 1e-8:
            self._ee_vel[:2] *= vmax_xy / vxy
        self._ee_vel[2] = np.clip(self._ee_vel[2], -vmax_z, vmax_z)

        # 位置更新
        self._ee_pos += self._ee_vel * dt

        # ★ z 锁定 (cruise 阶段) — 加入 PID 修正
        if lock_z and z_lock_height is not None:
            # 基础高度 + PID 修正
            self._ee_pos[2] = z_lock_height + z_pid_correction
            self._ee_vel[2] = 0.0

        # 软锚定到真实 EE (防止积分漂移)
        alpha = self._anchor_alpha
        if lock_z:
            # ★ z-lock 模式: 只锚定 xy, z 由 PID 控制
            self._ee_pos[:2] = (
                (1 - alpha) * self._ee_pos[:2] +
                alpha * real_ee_pos[:2].astype(np.float64)
            )
        else:
            self._ee_pos = (
                (1 - alpha) * self._ee_pos +
                alpha * real_ee_pos.astype(np.float64)
            )

        # 工作空间约束
        dist_xy = np.linalg.norm(self._ee_pos[:2])
        ws_r = 0.48
        if dist_xy > ws_r:
            self._ee_pos[:2] *= ws_r / max(dist_xy, 1e-6)

        # z 下限保护
        self._ee_pos[2] = max(self._ee_pos[2], 0.02)

        # IK 求解 — ★ 使用 target_yaw 参数
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

        q_target = np.clip(q_target, self.q_low, self.q_high)
        self._last_q = q_target.copy()

        # delta_q
        delta_q = np.clip(
            q_target - current_q.astype(np.float64),
            -self.dq_max, self.dq_max
        )
        return delta_q.astype(np.float32)