# ==============================================================================
# orca_nmpc_controller.py — ORCA Planner + NMPC Executor 分层控制器
#
# 架构设计:
#   Layer 1 (规划): ORCA
#     - 输入: payload当前位置/速度, 最终目标, 障碍物列表
#     - 输出: 中间目标位置 p_mid_xy (避障后的下一步期望到达点)
#     - 特点: 时间视野短(0.5s), 只负责"去哪"
#
#   Layer 2 (执行): NMPC (NMPCController4D)
#     - 输入: 完整12维状态(EE+payload), 中间目标 p_mid_xy
#     - 输出: EE加速度 [ax, ay, az, ayaw]
#     - 特点: 代价函数包含防摆项(Q_swing), 同时解决导航+防摆
#     - 不修改NMPC内部代码, 只替换传入的 P_ref
#
#   Layer 3 (残差RL, 可选):
#     - 在NMPC输出的delta_q基础上叠加小量残差
#     - 专注于补偿NMPC无法处理的细节 (非线性摆动/模型误差)
#
# 关键洞察:
#   原版NMPC使用A*规划的路径点作为P_ref, 不感知障碍物
#   → 当实际轨迹偏离A*路径时(因摆动/扰动), NMPC仍然追原路径点
#   → payload可能撞到障碍物
#
#   新版: ORCA实时重规划P_ref
#   → 每步根据当前payload位置计算避障目标
#   → NMPC追这个实时目标, 既避障又防摆
#   → 更鲁棒: 摆动导致偏离时ORCA自动重规划绕回
# ==============================================================================

import numpy as np
from scipy.spatial.transform import Rotation as R


# ==============================================================================
# ORCA 规划层 (精简版, 专注为NMPC提供中间目标)
# ==============================================================================

class ORCAMidpointPlanner:
    """
    为 NMPC 提供实时中间目标点的 ORCA 规划器。

    与 orca_expert.py 的区别:
      orca_expert: 直接输出EE加速度 (完整控制器)
      ORCAMidpointPlanner: 只输出"下一步目标位置" (规划层)
        → 不关心如何执行, 只告诉NMPC"去哪"

    输出格式: p_mid_xy (2D位置), 单位 m
      - 在当前位置和最终目标之间
      - 满足障碍物约束
      - 距当前位置约 lookahead_dist
    """

    def __init__(self, config):
        orca_cfg = config.get("orca_nmpc", config.get("orca", {}))

        # ORCA 参数
        self.max_speed       = float(orca_cfg.get("max_speed",        0.18))
        self.time_horizon    = float(orca_cfg.get("time_horizon",      1.5))  # 更短，反应更快
        self.obstacle_margin = float(orca_cfg.get("obstacle_margin",   0.07))
        self.soft_margin     = float(orca_cfg.get("soft_margin",       0.04))

        # 中间目标前瞻距离: NMPC的参考点距当前位置的距离
        # 太近: NMPC参考点跳动快, 解不稳定
        # 太远: 避障不够及时
        self.lookahead_dist  = float(orca_cfg.get("lookahead_dist",   0.12))

        # 分段速度规划
        self.d_decel_start   = float(orca_cfg.get("d_decel_start",    0.20))  # 开始减速
        self.d_stop          = float(orca_cfg.get("d_stop",           0.05))  # 精停区
        self.creep_speed     = float(orca_cfg.get("creep_speed",      0.04))

        self._prev_v = None

    def reset(self):
        self._prev_v = None

    def _speed_ref(self, dist_to_goal):
        """分段速度规划: 根据到目标距离决定参考速度。"""
        if dist_to_goal > self.d_decel_start:
            return self.max_speed
        elif dist_to_goal > self.d_stop:
            t = (dist_to_goal - self.d_stop) / (self.d_decel_start - self.d_stop)
            return self.creep_speed + t * t * (self.max_speed - self.creep_speed)
        else:
            return self.creep_speed * (dist_to_goal / max(self.d_stop, 1e-6))

    def compute_midpoint(self, pl_pos_2d, pl_vel_2d, goal_xy, obstacles):
        """
        计算下一步的中间目标位置。

        Args:
            pl_pos_2d:  payload 当前 2D 位置
            pl_vel_2d:  payload 当前 2D 速度
            goal_xy:    最终目标 2D 位置
            obstacles:  [(ox, oy, radius), ...]

        Returns:
            p_mid_xy (np.ndarray, shape(2,)): NMPC 的中间参考点
            v_ref    (float): 当前参考速度 (用于调整NMPC权重)
        """
        pl_pos_2d = np.asarray(pl_pos_2d, np.float64)
        pl_vel_2d = np.asarray(pl_vel_2d, np.float64)
        goal_xy   = np.asarray(goal_xy,   np.float64)

        diff_to_goal = goal_xy - pl_pos_2d
        dist_to_goal = float(np.linalg.norm(diff_to_goal))

        # 已到达: 返回目标本身
        if dist_to_goal < self.d_stop * 0.5:
            return goal_xy.copy(), 0.0

        # 参考速度
        v_ref = self._speed_ref(dist_to_goal)

        # ORCA 计算最优速度方向 (满足避障约束)
        v_orca = self._orca_velocity(pl_pos_2d, pl_vel_2d, goal_xy, obstacles, v_ref)

        # 中间目标 = 当前位置 + v_orca 方向的 lookahead_dist
        v_norm = float(np.linalg.norm(v_orca))
        if v_norm > 1e-6:
            direction  = v_orca / v_norm
            lookahead  = min(self.lookahead_dist, dist_to_goal)
            p_mid_xy   = pl_pos_2d + direction * lookahead
        else:
            # 速度为零: 中间目标就是最终目标 (让NMPC稳定)
            p_mid_xy = goal_xy.copy()

        return p_mid_xy, v_ref

    def _orca_velocity(self, pos, vel, goal, obstacles, v_ref_mag):
        """ORCA计算满足约束的最优速度向量。"""
        diff = goal - pos
        dist = float(np.linalg.norm(diff))
        if dist < 1e-4:
            return np.zeros(2)

        # 期望速度
        v_pref = (diff / dist) * v_ref_mag

        if not obstacles:
            return self._smooth(v_pref, v_ref_mag)

        # 紧急逃脱: 已在障碍物内
        escape = np.zeros(2)
        deeply_inside = False
        for (ox, oy, orad) in obstacles:
            obs_pos    = np.array([ox, oy], np.float64)
            rel        = pos - obs_pos
            combined_r = orad + self.obstacle_margin
            d_obs      = float(np.linalg.norm(rel))
            if d_obs < combined_r and d_obs > 1e-6:
                escape += (rel / d_obs) * (combined_r - d_obs + 0.02) * 3.0
                deeply_inside = True
        if deeply_inside:
            v_e = vel + escape
            n = float(np.linalg.norm(v_e))
            if n > v_ref_mag: v_e = v_e / n * v_ref_mag
            return self._smooth(v_e, v_ref_mag)

        # 构建半平面约束
        half_planes = []
        tau = self.time_horizon
        for (ox, oy, orad) in obstacles:
            obs_pos    = np.array([ox, oy], np.float64)
            rel_pos    = obs_pos - pos
            combined_r = orad + self.obstacle_margin
            d_obs      = float(np.linalg.norm(rel_pos))
            if d_obs < 1e-6: continue

            center = rel_pos / tau
            r_tau  = combined_r / tau
            w      = vel - center
            w_len  = float(np.linalg.norm(w))

            if w_len < r_tau:
                n = (w / w_len) if w_len > 1e-8 else (-rel_pos / d_obs)
                u = (r_tau - w_len + 1e-5) * n
                half_planes.append((n, float(np.dot(n, vel + u))))
            else:
                r_soft = (combined_r + self.soft_margin) / tau
                if w_len < r_soft:
                    n = w / w_len
                    soft_u = (r_soft - w_len) * 0.3 * n
                    half_planes.append((n, float(np.dot(n, vel + soft_u))))

        if not half_planes:
            return self._smooth(v_pref, v_ref_mag)

        # 迭代投影
        v_opt = v_pref.copy()
        n_v = float(np.linalg.norm(v_opt))
        if n_v > v_ref_mag: v_opt = v_opt / n_v * v_ref_mag

        for _ in range(60):
            violated = False
            for (n, b) in half_planes:
                proj = float(np.dot(n, v_opt))
                if proj < b - 1e-8:
                    v_opt += (b - proj) * n
                    nv = float(np.linalg.norm(v_opt))
                    if nv > v_ref_mag: v_opt = v_opt / nv * v_ref_mag
                    violated = True
            if not violated: break

        return self._smooth(v_opt, v_ref_mag)

    def _smooth(self, v, v_max):
        """速度平滑 + 限幅。"""
        v = np.asarray(v, np.float64)
        n = float(np.linalg.norm(v))
        if n > v_max: v = v / n * v_max
        if self._prev_v is not None:
            v = 0.70 * v + 0.30 * self._prev_v
            n2 = float(np.linalg.norm(v))
            if n2 > v_max: v = v / n2 * v_max
        self._prev_v = v.copy()
        return v


# ==============================================================================
# ORCA+NMPC 分层控制器 (Cruise 段核心)
# ==============================================================================

class ORCANMPCCruiseController:
    """
    ORCA Planner + NMPC Executor 分层 Cruise 控制器。

    调用方式 (每步):
        delta_q = ctrl.compute_delta_q(obs, current_q, goal_xy, obstacles)

    内部流程:
        1. ORCA 规划: p_mid = planner.compute_midpoint(pl_pos, pl_vel, goal, obs)
        2. 构建 NMPC 状态: state_12d = [ee_pos/vel, pl_pos/vel]
        3. NMPC 求解: action_4d = nmpc.get_action(state_12d, P_ref=[p_mid, z_cruise, 0])
        4. EE 积分: ee_pos += ee_vel*dt + 0.5*acc*dt², ee_vel += acc*dt
        5. Z 锁定: ee_z = z_cruise + rope_L + z_pid_correction
        6. IK + delta_q
    """

    def __init__(self, config, ik_solver):
        from controller import NMPCController4D

        ctrl_cfg  = config["controller"]
        plan_cfg  = config.get("planning", {})
        ee_cfg    = config.get("ee_control", {})
        sp        = config["space"]
        orca_cfg  = config.get("orca_nmpc", config.get("orca", {}))

        self.dt        = float(ctrl_cfg["dt"])
        self.z_cruise  = float(plan_cfg.get("payload_z_cruise", 0.25))
        self.ik_solver = ik_solver

        # 绳长 (EE需在payload上方rope_L处)
        rope_cfg    = config.get("rope", {})
        n_seg       = int(rope_cfg.get("num_segments", 10))
        seg_len     = float(rope_cfg.get("segment_length", 0.04))
        hook        = float(rope_cfg.get("hook_offset", 0.05))
        self._rope_L = n_seg * seg_len + hook   # 默认 0.45m

        # 关节限位
        self.dq_max = np.array(sp.get("dq_max", [0.1]*7), dtype=np.float64)
        self.q_low  = np.array(sp["action_space_low"],     dtype=np.float64)
        self.q_high = np.array(sp["action_space_high"],    dtype=np.float64)

        # NMPC (使用现有 NMPCController4D, 不做任何修改)
        self._nmpc = NMPCController4D(
            dt      = self.dt,
            N       = int(ctrl_cfg.get("N", 20)),
            L       = float(ctrl_cfg["L"]),
            u_max_xy= float(ctrl_cfg.get("u_max_xy",  1.2)),
            u_max_z = float(ctrl_cfg.get("u_max_z",   2.5)),
            u_max_yaw=float(ctrl_cfg.get("u_max_yaw", 2.0)),
        )

        # ORCA 规划器
        self._planner = ORCAMidpointPlanner(config)

        # EE 积分状态 (与 EEAccController 相同逻辑)
        self._ee_pos      = np.zeros(3, np.float64)
        self._ee_vel      = np.zeros(3, np.float64)
        self._ee_yaw      = 0.0
        self._ee_yaw_vel  = 0.0
        self._last_q      = None
        self._anchor_alpha= float(orca_cfg.get("anchor_alpha", 0.10))

        # 速度限制
        self._vel_max_xy  = float(ee_cfg.get("vel_max_xy", 0.25))
        self._vel_max_z   = float(ee_cfg.get("vel_max_z",  0.15))

        # 诊断
        self._last_p_mid     = None
        self._last_v_ref     = 0.0
        self._last_nmpc_acc  = np.zeros(4)

    def reset(self, ee_pos, current_q, obs=None):
        """每个 episode 开始时调用。"""
        self._ee_pos     = np.asarray(ee_pos, np.float64).copy()
        self._ee_vel     = np.zeros(3, np.float64)
        self._ee_yaw     = 0.0
        self._ee_yaw_vel = 0.0
        self._last_q     = np.asarray(current_q, np.float64).copy()
        self._nmpc.last_sol = None
        self._nmpc.last_az  = 0.0
        self._planner.reset()
        self._last_p_mid    = None
        self._last_v_ref    = 0.0
        self._last_nmpc_acc = np.zeros(4)

        # 从 obs 初始化 EE 状态
        if obs is not None:
            self._ee_pos[0] = float(obs[0])
            self._ee_pos[1] = float(obs[1])
            self._ee_pos[2] = float(obs[19]) if len(obs) > 19 else self._ee_pos[2]
            self._ee_vel[0] = float(obs[2])
            self._ee_vel[1] = float(obs[3])

    def compute_delta_q(self, obs, current_q, goal_xy,
                        obstacles,
                        z_pid_correction=0.0,
                        target_yaw=0.0,
                        real_ee_pos=None):
        """
        主控制接口: 每步调用一次。

        Args:
            obs:              环境原始 obs (用于提取EE/payload状态)
            current_q:        当前关节角 (7维)
            goal_xy:          最终目标 2D 位置
            obstacles:        [(ox, oy, r), ...]
            z_pid_correction: Z轴PID修正量 (来自CruiseZYawPID)
            target_yaw:       目标偏航角
            real_ee_pos:      真实EE位置 (用于软锚定, None则从obs读取)

        Returns:
            delta_q (np.ndarray, shape(7,)): 关节角增量
        """
        obs        = np.asarray(obs, np.float64)
        current_q  = np.asarray(current_q, np.float64)
        goal_xy    = np.asarray(goal_xy, np.float64)

        # ── 从 obs 提取状态 ────────────────────────────────────────────────
        ee_x,  ee_y   = float(obs[0]),  float(obs[1])
        ee_vx, ee_vy  = float(obs[2]),  float(obs[3])
        pl_x,  pl_y   = float(obs[4]),  float(obs[5])
        pl_vx, pl_vy  = float(obs[6]),  float(obs[7])
        ee_z          = float(obs[19]) if len(obs) > 19 else self._ee_pos[2]
        ee_vz         = float(obs[20]) if len(obs) > 20 else 0.0
        ee_yaw        = float(obs[27]) if len(obs) > 27 else 0.0
        ee_yaw_v      = float(obs[28]) if len(obs) > 28 else 0.0

        if real_ee_pos is None:
            real_ee_pos = np.array([ee_x, ee_y, ee_z], np.float64)
        else:
            real_ee_pos = np.asarray(real_ee_pos, np.float64)

        # ── Layer 1: ORCA 规划 → 中间目标位置 ────────────────────────────
        pl_pos_2d = np.array([pl_x, pl_y], np.float64)
        pl_vel_2d = np.array([pl_vx, pl_vy], np.float64)
        p_mid_xy, v_ref = self._planner.compute_midpoint(
            pl_pos_2d, pl_vel_2d, goal_xy, obstacles)

        self._last_p_mid = p_mid_xy.copy()
        self._last_v_ref = v_ref

        # ── Layer 2: NMPC 执行 ────────────────────────────────────────────
        # 构建 NMPC 12维状态: [ee_x, ee_y, ee_z, ee_yaw, ee_vx, ee_vy, ee_vz, ee_yaw_v, pl_x, pl_y, pl_vx, pl_vy]
        state_12d = np.array([
            ee_x, ee_y, ee_z, ee_yaw,
            ee_vx, ee_vy, ee_vz, ee_yaw_v,
            pl_x, pl_y, pl_vx, pl_vy,
        ], dtype=np.float64)

        # NMPC 参考点: 中间目标位置 + 巡航高度
        # z_ref 是 payload 目标高度, NMPC内部会加上 estimated_L
        # 这里传 p_mid 的高度 = z_cruise (payload巡航高度)
        P_ref = np.array([
            float(p_mid_xy[0]),
            float(p_mid_xy[1]),
            float(self.z_cruise),   # payload 目标 z (NMPC加rope_L得EE_z)
            float(target_yaw),
        ], dtype=np.float64)

        action_4d = self._nmpc.get_action(state_12d, P_ref)
        self._last_nmpc_acc = action_4d.copy()

        # ── 积分: acc → vel → pos ────────────────────────────────────────
        a_xyz = action_4d[:3]
        a_yaw = float(action_4d[3])

        self._ee_vel += a_xyz * self.dt
        vxy = float(np.linalg.norm(self._ee_vel[:2]))
        if vxy > self._vel_max_xy and vxy > 1e-8:
            self._ee_vel[:2] *= self._vel_max_xy / vxy
        self._ee_vel[2] = np.clip(self._ee_vel[2], -self._vel_max_z, self._vel_max_z)
        self._ee_pos += self._ee_vel * self.dt

        self._ee_yaw_vel += a_yaw * self.dt
        self._ee_yaw     += self._ee_yaw_vel * self.dt

        # ── Z 锁定 (与 EEAccController 完全一致) ─────────────────────────
        ee_z_target = self.z_cruise + self._rope_L + z_pid_correction
        self._ee_pos[2] = ee_z_target
        self._ee_vel[2] = 0.0

        # ── 软锚定 ────────────────────────────────────────────────────────
        alpha = self._anchor_alpha
        self._ee_pos[:2] = ((1-alpha)*self._ee_pos[:2] + alpha*real_ee_pos[:2])

        # ── 工作空间约束 ──────────────────────────────────────────────────
        dxy = float(np.linalg.norm(self._ee_pos[:2]))
        if dxy > 0.48:
            self._ee_pos[:2] *= 0.48 / max(dxy, 1e-6)
        self._ee_pos[2] = max(self._ee_pos[2], 0.02)

        # ── IK ────────────────────────────────────────────────────────────
        q_start = self._last_q if self._last_q is not None else current_q
        q_target = self.ik_solver.solve_4d(
            current_q = q_start,
            target_x  = float(self._ee_pos[0]),
            target_y  = float(self._ee_pos[1]),
            target_z  = float(self._ee_pos[2]),
            target_yaw= float(self._ee_yaw),
        )
        if q_target is None or np.any(np.isnan(q_target)):
            q_target = current_q.copy()

        q_target     = np.clip(q_target, self.q_low, self.q_high)
        self._last_q = q_target.copy()

        delta_q = np.clip(
            q_target - current_q,
            -self.dq_max, self.dq_max)
        return delta_q.astype(np.float32)

    @property
    def diag(self):
        """返回诊断信息 (用于日志)。"""
        return {
            "orca_nmpc/p_mid_x":   float(self._last_p_mid[0]) if self._last_p_mid is not None else 0.0,
            "orca_nmpc/p_mid_y":   float(self._last_p_mid[1]) if self._last_p_mid is not None else 0.0,
            "orca_nmpc/v_ref":     float(self._last_v_ref),
            "orca_nmpc/nmpc_ax":   float(self._last_nmpc_acc[0]),
            "orca_nmpc/nmpc_ay":   float(self._last_nmpc_acc[1]),
            "orca_nmpc/nmpc_acc_norm": float(np.linalg.norm(self._last_nmpc_acc[:2])),
        }


# ==============================================================================
# 残差 RL 接口: ORCA+NMPC base + RL 残差 delta_q
# ==============================================================================

class ORCANMPCResidualController:
    """
    ORCA+NMPC base 控制器的残差 RL 接口。

    RL 输出: residual_dq (shape (7,), 在NMPC delta_q基础上的修正量)
    总输出:  delta_q = clip(nmpc_dq + residual_dq, -dq_max, dq_max)

    这比原版"残差加速度→积分→delta_q"更直接:
      - NMPC已经在关节空间给出了最优delta_q
      - RL直接在delta_q空间做修正, 物理意义更清晰
      - 避免残差acc通过积分放大的问题
    """

    def __init__(self, config, ik_solver):
        self._base = ORCANMPCCruiseController(config, ik_solver)
        sp = config["space"]
        self.dq_max = np.array(sp.get("dq_max", [0.1]*7), dtype=np.float64)
        # RL 残差限幅: 不超过 dq_max 的一定比例
        rl_cfg = config.get("cruise_rl", {})
        self._residual_scale = float(rl_cfg.get("residual_dq_scale_cruise", 0.30))

    def reset(self, ee_pos, current_q, obs=None):
        self._base.reset(ee_pos, current_q, obs)

    def compute_delta_q_base(self, obs, current_q, goal_xy, obstacles,
                              z_pid_correction=0.0, target_yaw=0.0, real_ee_pos=None):
        """计算 base NMPC delta_q (不含RL残差)。"""
        return self._base.compute_delta_q(
            obs, current_q, goal_xy, obstacles,
            z_pid_correction, target_yaw, real_ee_pos)

    def apply_residual(self, base_dq, rl_residual_dq):
        """
        叠加 RL 残差。

        Args:
            base_dq:         NMPC 输出的 delta_q (shape 7)
            rl_residual_dq:  RL 输出的残差 delta_q (shape 7)

        Returns:
            total_dq (np.ndarray, shape 7)
        """
        base_dq = np.asarray(base_dq, np.float64)
        res_dq  = np.asarray(rl_residual_dq, np.float64)

        # 残差限幅: 不超过 dq_max × residual_scale
        res_max = self.dq_max * self._residual_scale
        res_dq  = np.clip(res_dq, -res_max, res_max)

        total_dq = np.clip(base_dq + res_dq, -self.dq_max, self.dq_max)
        return total_dq.astype(np.float32)

    @property
    def diag(self):
        return self._base.diag


# ==============================================================================
# BC 标签生成器: 使用 ORCA+NMPC 生成 delta_q BC 标签
# ==============================================================================

def collect_orca_nmpc_bc_label(ctrl, obs, current_q, goal_xy, obstacles,
                                z_pid_correction=0.0, target_yaw=0.0, real_ee_pos=None):
    """
    生成 ORCA+NMPC 的 BC 标签: base delta_q (不含残差)。

    这个函数在 BC 数据收集阶段调用:
      - obs 来自 ORCA+NMPC 执行的真实轨迹 (状态分布对齐)
      - 标签 = NMPC 在当前 obs 下计算的 delta_q
      - RL 学习的是在此基础上的残差 (初始化为零残差)

    Returns:
        bc_label_dq (np.ndarray, shape 7)
    """
    # ctrl 内部状态已经积分过一步, 所以这里计算的是"当前观测下NMPC的决策"
    # 注意: 这会让 ctrl 内部状态前进一步, 调用者不应再调用 ctrl.compute_delta_q
    return ctrl.compute_delta_q_base(
        obs, current_q, goal_xy, obstacles,
        z_pid_correction, target_yaw, real_ee_pos)