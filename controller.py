# ==============================================================================
# controller.py — NMPC 专家控制器（BC 标签生成器）
#
# 架构职责（新版）：
#   此模块在新架构中承担"专家标签生成器"的角色，而非直接参与控制链。
#
# 数据流：
#   env.obs (物理状态)
#     → NMPCTrajectoryTracker.compute_action()  → 4D 末端加速度
#     → 内部积分 + NativeIKSolver.solve_4d()    → 7 关节角目标
#     → 作为 BC Loss 的监督信号（q_target_joints）
#
# 与旧版 nmpc_controller_new.py 的差异：
#   [CTRL-NEW-1] 新增 JointSpaceExpert 类
#     - 封装"NMPC → 积分 → IK"完整链路
#     - 输出 7D 关节角（直接作为 BC 目标）
#     - 保留完整的 NMPC 计算逻辑（NMPCController4D + NMPCTrajectoryTracker）
#
#   [CTRL-NEW-2] 内部维护虚拟 EE 位姿积分状态
#     - 与 env 中的 current_mocap_pos 独立，避免耦合
#     - 每次 reset() 时同步 env 初始状态
#
#   [CTRL-NEW-3] compute_joint_target() 接口
#     - 输入：env.obs（与 _get_obs() 对齐）+ 当前真实关节角
#     - 输出：7D 关节角目标（BC 监督信号）
#     - 若 IK 失败，返回当前关节角（保持原地不动，不产生不合理标签）
#
# 保留内容：
#   - NMPCController4D（完整 NMPC 求解器）
#   - NMPCTrajectoryTracker（航点追踪器）
#   - 所有修复日志（CTRL-BUG-1 ~ CTRL-BUG-7 和 CTRL-IMPROVE-1~2）
# ==============================================================================

import numpy as np
import casadi as ca


# ==============================================================================
# NMPCController4D — 4D 末端加速度求解器（保持完整，与 nmpc_controller_new.py 对齐）
# ==============================================================================

class NMPCController4D:
    """
    4D NMPC 防摆控制器（ax, ay, az, ayaw）。
    物理模型：3D 小角近似悬吊摆，Z 轴独立通道，偏航直接控制。
    """

    def __init__(self, dt=0.1, N=15, L=0.45,
                 u_max_xy=0.5, u_max_z=2.0, u_max_yaw=2.0):
        self.dt  = dt
        self.N   = N
        self.L   = L
        self.g   = 9.81
        self.nx  = 12
        self.nu  = 4
        self.last_az = 0.0

        x      = ca.SX.sym('x', self.nx)
        u      = ca.SX.sym('u', self.nu)
        az_prev = ca.SX.sym('az_prev')

        damping = 0.01
        p_x, p_y, p_z, p_yaw     = x[0], x[1], x[2], x[3]
        v_px, v_py, v_pz, v_pyaw = x[4], x[5], x[6], x[7]
        q_x, q_y                 = x[8], x[9]
        v_qx, v_qy               = x[10], x[11]
        ax, ay, az, ayaw         = u[0], u[1], u[2], u[3]

        omega_sq_eff = (self.g + az_prev) / self.L

        dx = ca.vertcat(
            v_px, v_py, v_pz, v_pyaw,
            ax, ay, az, ayaw,
            v_qx, v_qy,
            -omega_sq_eff * (q_x - p_x) - damping * v_qx,
            -omega_sq_eff * (q_y - p_y) - damping * v_qy
        )
        f_dyn = ca.Function('f', [x, u, az_prev], [dx])

        X      = ca.SX.sym('X', self.nx, self.N + 1)
        U      = ca.SX.sym('U', self.nu, self.N)
        P_ref  = ca.SX.sym('P_ref', 4)
        X_init = ca.SX.sym('X_init', self.nx)
        Az_lin = ca.SX.sym('Az_lin')

        cost        = 0
        constraints = []

        Q_pos   = np.array([10.0, 10.0, 20.0, 5.0])
        Q_swing = np.array([100.0, 100.0])
        Q_vel   = 2.0
        R_acc   = np.array([0.2, 0.2, 0.2, 0.3])

        constraints.append(X[:, 0] - X_init)
        mocap_target_z = P_ref[2] + self.L

        for k in range(self.N):
            k1 = f_dyn(X[:, k], U[:, k], Az_lin)
            k2 = f_dyn(X[:, k] + dt/2 * k1, U[:, k], Az_lin)
            k3 = f_dyn(X[:, k] + dt/2 * k2, U[:, k], Az_lin)
            k4 = f_dyn(X[:, k] + dt    * k3, U[:, k], Az_lin)
            x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            constraints.append(X[:, k+1] - x_next)

            cost += Q_pos[0] * (X[8,  k] - P_ref[0])**2
            cost += Q_pos[1] * (X[9,  k] - P_ref[1])**2
            cost += Q_pos[2] * (X[2,  k] - mocap_target_z)**2
            cost += Q_pos[3] * (X[3,  k] - P_ref[3])**2
            cost += Q_swing[0] * (X[8, k] - X[0, k])**2
            cost += Q_swing[1] * (X[9, k] - X[1, k])**2
            cost += Q_vel * (X[4,k]**2 + X[5,k]**2 + X[6,k]**2 + X[7,k]**2)
            cost += (R_acc[0]*U[0,k]**2 + R_acc[1]*U[1,k]**2
                   + R_acc[2]*U[2,k]**2 + R_acc[3]*U[3,k]**2)

        cost += 10 * (Q_pos[0]*(X[8, self.N]-P_ref[0])**2
                    + Q_pos[1]*(X[9, self.N]-P_ref[1])**2)
        cost += 10 * (Q_pos[2]*(X[2, self.N]-mocap_target_z)**2
                    + Q_pos[3]*(X[3, self.N]-P_ref[3])**2)

        nlp = {
            'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            'f': cost,
            'g': ca.vertcat(*constraints),
            'p': ca.vertcat(X_init, P_ref, Az_lin)
        }
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.max_iter': 50, 'ipopt.tol': 1e-2, 'ipopt.acceptable_tol': 1e-1,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = self.nx * (self.N + 1) + self.nu * self.N
        self.lbx = -ca.inf * np.ones(n_vars)
        self.ubx =  ca.inf * np.ones(n_vars)

        u_start_idx = self.nx * (self.N + 1)
        for i in range(self.N):
            idx = u_start_idx + i * self.nu
            self.lbx[idx:idx+self.nu] = [-u_max_xy, -u_max_xy, -u_max_z, -u_max_yaw]
            self.ubx[idx:idx+self.nu] = [ u_max_xy,  u_max_xy,  u_max_z,  u_max_yaw]

        self.lbg = np.zeros(self.nx * (self.N + 1))
        self.ubg = np.zeros(self.nx * (self.N + 1))
        self.last_sol = None

    def get_action(self, state_12d, target_4d):
        """
        Args:
            state_12d: [mocap_x,y,z,yaw, mocap_vx,vy,vz,yaw_vel, payload_x,y, payload_vx,vy]
            target_4d: [target_payload_x, target_payload_y, target_payload_z, target_yaw]
        Returns:
            u_opt: (4,) [ax, ay, az, ayaw]
        """
        p_val   = np.concatenate([state_12d, target_4d, [self.last_az]])
        x0_guess = self.last_sol if self.last_sol is not None else np.zeros(self.lbx.shape)
        try:
            sol = self.solver(
                x0=x0_guess, p=p_val,
                lbg=self.lbg, ubg=self.ubg,
                lbx=self.lbx, ubx=self.ubx
            )
            self.last_sol = sol['x']
            u_start = self.nx * (self.N + 1)
            u_opt = np.array(sol['x'][u_start: u_start + self.nu]).flatten()
            self.last_az = float(u_opt[2])
        except Exception:
            self.last_sol = None
            u_opt = np.zeros(self.nu)
            self.last_az = 0.0
        return u_opt


# ==============================================================================
# NMPCTrajectoryTracker — 航点追踪器（保留，内部使用）
# ==============================================================================

class NMPCTrajectoryTracker:
    def __init__(self, dt=0.1, N=15, L=0.445,
                 arrival_threshold_xy=0.05, arrival_threshold_z=0.20):
        self.mpc = NMPCController4D(dt=dt, N=N, L=L)
        self.path = None
        self.current_idx = 0
        self.arrival_threshold_xy = arrival_threshold_xy
        self.arrival_threshold_z  = arrival_threshold_z
        self.estimated_L = L

    def set_path(self, path):
        if path is None or len(path) == 0:
            self.path = None
            self.current_idx = 0
            return
        self.path = np.array(path, dtype=np.float64)
        self.current_idx = 0
        self.mpc.last_sol = None
        self.mpc.last_az  = 0.0

    def compute_ee_acceleration(self, obs, target_yaw=0.0):
        """
        计算末端 4D 加速度（[ax, ay, az, ayaw]）。
        返回: np.ndarray (4,)
        """
        if self.path is None:
            return np.zeros(4)

        mocap_x, mocap_y  = obs[0], obs[1]
        mocap_vx, mocap_vy = obs[2], obs[3]
        payload_x, payload_y = obs[4], obs[5]
        payload_vx, payload_vy = obs[6], obs[7]

        # [修复] 替换错误的负向索引
        mocap_z    = float(obs[-26])  # 原为 -12
        mocap_vz   = float(obs[-25])  # 原为 -11
        payload_z  = float(obs[-24])  # 原为 -10
        payload_vz = float(obs[-23])  # 原为 -9
        mocap_yaw     = float(obs[-4])
        mocap_yaw_vel = float(obs[-3])

        actual_L = mocap_z - payload_z
        self.estimated_L = 0.9 * self.estimated_L + 0.1 * actual_L

        curr_payload_xy = np.array([payload_x, payload_y])
        target_wp = self.path[self.current_idx]
        target_wp_3d = np.array([
            target_wp[0], target_wp[1],
            target_wp[2] if len(target_wp) >= 3 else 0.3
        ])

        dist_xy = np.linalg.norm(curr_payload_xy - target_wp_3d[:2])
        dist_z  = abs(payload_z - target_wp_3d[2])

        if (dist_xy < self.arrival_threshold_xy and
                dist_z < self.arrival_threshold_z and
                self.current_idx < len(self.path) - 1):
            self.current_idx += 1
            target_wp = self.path[self.current_idx]
            target_wp_3d = np.array([
                target_wp[0], target_wp[1],
                target_wp[2] if len(target_wp) >= 3 else 0.3
            ])
            self.mpc.last_sol = None
            self.mpc.last_az  = 0.0

        state_12d = np.array([
            mocap_x, mocap_y, mocap_z, mocap_yaw,
            mocap_vx, mocap_vy, mocap_vz, mocap_yaw_vel,
            payload_x, payload_y, payload_vx, payload_vy
        ], dtype=np.float64)

        compensated_target_z = target_wp_3d[2] + (self.estimated_L - self.mpc.L)
        P_ref = np.array([
            target_wp_3d[0], target_wp_3d[1],
            compensated_target_z, target_yaw
        ], dtype=np.float64)

        return self.mpc.get_action(state_12d, P_ref)


# ==============================================================================
# JointSpaceExpert — BC 标签生成器
# ==============================================================================

class JointSpaceExpert:
    """
    [CTRL-NEW-1] BC 标签生成器：NMPC → 积分 → IK → 关节角目标

    这个类封装了完整的"专家控制链路"，输出 7D 关节角目标作为 BC 监督信号。
    它不直接控制机器人，只负责生成"如果专家来操作会设置什么关节角"的标签。

    使用方法：
        expert = JointSpaceExpert(config, ik_solver)
        expert.reset(env_obs, init_q)          # 每 episode 开始时同步状态
        expert.set_path(path)                   # 设置路径

        # 每步调用
        q_target = expert.compute_joint_target(env_obs, current_q)
    """

    def __init__(self, config: dict, ik_solver):
        """
        Args:
            config: 全局 DEFAULT_CONFIG
            ik_solver: NativeIKSolver 实例（来自 env，共享同一个 MuJoCo 模型）
        """
        ctrl_cfg = config["controller"]
        self.tracker = NMPCTrajectoryTracker(
            dt=ctrl_cfg["dt"],
            N=ctrl_cfg["N"],
            L=ctrl_cfg["L"],
            arrival_threshold_xy=ctrl_cfg["arrival_threshold_xy"],
            arrival_threshold_z=ctrl_cfg["arrival_threshold_z"],
        )
        self.ik_solver = ik_solver
        self.dt = ctrl_cfg["dt"]

        # 内部虚拟 EE 位姿状态（独立于 env，避免耦合）
        self._ee_pos  = np.zeros(3)
        self._ee_vel  = np.zeros(3)
        self._ee_yaw  = 0.0
        self._ee_yaw_vel = 0.0

        # 关节角缓存（IK 迭代起点）
        self._last_q = None

    def reset(self, env_obs: np.ndarray, init_q: np.ndarray):
        """
        每回合开始时调用，同步 env 的初始状态。

        Args:
            env_obs: env.reset() 返回的初始观测
            init_q:  机械臂初始关节角 (7,)
        """
        # 从 obs 中读取初始 EE 位姿（与 env.current_mocap_pos 对齐）
        # [修复] 正确读取 X, Y, Z (不要直接用 [:3]，因为 env_obs[2] 是 vx)
        ee_x = float(env_obs[0])
        ee_y = float(env_obs[1])
        ee_z = float(env_obs[-26])  # 新版 ee_z 索引
        self._ee_pos = np.array([ee_x, ee_y, ee_z], dtype=np.float32)
        # [修复] 正确读取 vx, vy, vz
        self._ee_vel = np.array([env_obs[2], env_obs[3], env_obs[-25]], dtype=np.float32)
        # [修复] 正确读取 yaw 相关的索引
        self._ee_yaw = float(env_obs[-18])
        self._ee_yaw_vel = float(env_obs[-17])
        self._last_q  = init_q.copy()

        self.tracker.mpc.last_sol = None
        self.tracker.mpc.last_az  = 0.0

    def set_path(self, path):
        """加载规划路径。"""
        self.tracker.set_path(path)

    def compute_joint_target(self, env_obs: np.ndarray, current_q: np.ndarray,
                              target_yaw: float = 0.0) -> np.ndarray:
        """
        [CTRL-NEW-3] 核心接口：根据当前观测计算 BC 目标关节角。

        Args:
            env_obs:   env._get_obs() 返回的原始观测（未归一化）
            current_q: 当前真实关节角 (7,)
            target_yaw: 目标偏航角

        Returns:
            q_target: 7D 关节角目标 (7,)，作为 BC 监督信号
        """
        # 1. NMPC 计算末端 4D 加速度
        action_4d = self.tracker.compute_ee_acceleration(env_obs, target_yaw)
        a_xyz = action_4d[:3]
        a_yaw = action_4d[3]

        # 2. 积分更新内部虚拟 EE 状态（二阶 Euler，与 env.step 对齐）
        dt = self.dt
        self._ee_pos += self._ee_vel * dt + 0.5 * a_xyz * dt**2
        self._ee_vel += a_xyz * dt
        self._ee_yaw += self._ee_yaw_vel * dt + 0.5 * a_yaw * dt**2
        self._ee_yaw_vel += a_yaw * dt

        # 地板夹紧保护
        if self._ee_pos[2] < 0.25:
            self._ee_pos[2] = 0.25
            self._ee_vel[2] = 0.0

        # 3. IK 求解：虚拟 EE 目标位姿 → 7 关节角
        q_target = self.ik_solver.solve_4d(
            current_q=self._last_q if self._last_q is not None else current_q,
            target_x=self._ee_pos[0],
            target_y=self._ee_pos[1],
            target_z=self._ee_pos[2],
            target_yaw=self._ee_yaw
        )

        # IK 返回原始 current_q 表示失败，用真实关节角代替以保持标签合理性
        if q_target is None or np.any(np.isnan(q_target)):
            q_target = current_q.copy()

        # 更新缓存（下次 IK 的迭代起点）
        self._last_q = q_target.copy()
        return q_target.astype(np.float32)


# ==============================================================================
# 工具函数
# ==============================================================================

def obs_xy(obs, idx):
    """安全取 obs 前几维的正向索引值（不受障碍物偏移影响）。"""
    return float(obs[idx])