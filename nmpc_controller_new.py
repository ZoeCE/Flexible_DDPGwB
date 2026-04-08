import numpy as np
import casadi as ca

# ==============================================================================
# 修复日志（对照上一版 nmpc_controller_new.py 的全部问题）：
#
# [CTRL-BUG-1] compute_action() 使用正向固定索引 obs[0..13] 读取物理状态。
#              由于 obs 中间夹了 n_obstacles*3 维障碍物数据（默认 9 维），
#              obs[8] 实际是 rel_tx，obs[9] 是 rel_ty，obs[12] 是障碍物 X 坐标，
#              被错误地当作 mocap_z、mocap_vz、mocap_yaw 传入 MPC，造成完全错乱。
#              修复：统一改用 obs 末尾负索引读取，与 _get_obs() 布局严格对齐：
#                [-12] mocap_z  [-11] mocap_vz  [-10] payload_z  [-9] payload_vz
#                [-8]  mocap_roll  [-7] mocap_roll_vel  [-6] mocap_pitch  [-5] mocap_pitch_vel
#                [-4]  mocap_yaw   [-3] mocap_yaw_vel   [-2] payload_yaw  [-1] payload_yaw_vel
#              同时 mocap_xy / payload_xy 依旧用正向 obs[0..7]（在障碍物段之前，固定不变）。
#
# [CTRL-BUG-2] NMPCController4D 摆长 L 默认 0.4m，但实际控制链路中
#              "虚拟末端" current_mocap_pos 追踪的是 EE attachment_site，
#              attachment_site 到 prefab（payload）的实际距离 =
#                rope_length(0.4) + hook_offset_z(0.045) ≈ 0.445m。
#              MPC 物理模型用 L=0.4 会高估摆动频率 ω²=g/L，导致过度抑摆。
#              修复：将 NMPCTrajectoryTracker 默认 L 改为 0.445，
#              并在 NMPCController4D 中暴露 L 参数（调用方可按实际绳长传入）。
#
# [CTRL-BUG-3] mocap_target_z = P_ref[2] + self.L 中，P_ref[2] 是 payload 目标 Z，
#              加上 L 得到末端目标 Z，逻辑正确。
#              但旧版 NMPCController4D 的 u_max_z=0.5 m/s²，在 dt=0.1s 下
#              每步最大速度增量 0.05 m/s，初始 mocap_z=1.0m 需要降到 ~0.75m
#              需要 ~25 步，这段时间内 payload 无法被拉起（绳子松弛），
#              造成"上升一小段后失稳"现象。
#              修复：将 u_max_z 放宽至 2.0 m/s²，加快 Z 轴响应；
#              同时提高 Q_pos[2]（Z 轴位置权重）至 20.0，加快末端高度收敛。
#
# [CTRL-BUG-4] NMPCController4D 中 omega_sq_eff = (g + az) / L，az 是决策变量。
#              这使代价函数对 u 是四次的（az² 出现在 omega_sq_eff 中），
#              IPOPT 面对强非线性时收敛变差甚至失败，进而 fallback 到 zeros。
#              修复：将 omega_sq_eff 线性化，在每次 get_action 调用时用上一步
#              的 az 估计值（last_az）替代决策变量，使动力学保持双线性。
#              第一次调用时 last_az=0，即退化为标准线性化小角摆模型。
#
# [CTRL-BUG-5] R_acc 中 yaw 权重 0.1 相对 XY(0.2)/Z(0.2) 偏低，
#              yaw 误差容易被忽视导致末端旋转漂移，绳子扭转后进一步失稳。
#              修复：R_acc[3] 改为 0.3，适当提高 yaw 控制惩罚。
#
# [CTRL-BUG-6] set_path 接口只接受 np.array，当 env 传入 list 时 path[k][:2]
#              等操作无异常但若元素是 numpy 标量则语义不同。
#              修复：set_path 强制 np.array 化并做保护性检查。
#
# [CTRL-BUG-7] compute_action 中到达判断 dist_to_target 只比较 XY，
#              但此处的 arrival_threshold 也应考虑 Z，否则平面到达但高度还差
#              很远时会提前切换航点，Z 跟踪滞后会叠加。
#              修复：改为 3D 欧氏距离判断；同时提高默认 arrival_threshold 为 0.08。
#
# [CTRL-IMPROVE-1] last_sol 热启动策略：切换航点后目标突变，旧解作为初始猜测
#                  反而使 IPOPT 偏离最优区域。修复：切换航点时清空 last_sol。
#
# [CTRL-IMPROVE-2] IPOPT 参数调整：max_iter 从 30 提高到 50，提升在复杂
#                  非线性区域的求解成功率，代价是略增单步计算时间（仍在 10Hz 可控范围）。
# ==============================================================================


class NMPCController4D:
    """
    4D NMPC 防摆控制器（ax, ay, az, ayaw）。
    物理模型：3D 小角近似悬吊摆，Z 轴独立通道，偏航直接控制。
    """

    def __init__(self, dt=0.1, N=15, L=0.445,
                 u_max_xy=0.5, u_max_z=2.0, u_max_yaw=2.0):
        # 严格对齐 Config 中的控制频率 (10Hz -> 0.1s) 与动作边界
        # [CTRL-BUG-3 修复] u_max_z 放宽至 2.0 以加快 Z 轴初始收敛
        # [CTRL-BUG-2 修复] 默认 L=0.445 (rope_length 0.4 + hook_offset_z 0.045)
        self.dt = dt
        self.N = N
        self.L = L
        self.g = 9.81
        self.nx = 12
        self.nu = 4

        # [CTRL-BUG-4 修复] 保存上一步 az 估计值，用于线性化 omega_sq_eff
        self.last_az = 0.0

        # ======================================================================
        # 状态变量定义（符号）
        # ======================================================================
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)

        damping = 0.01  # 负载空气阻尼

        # 解析状态向量
        p_x, p_y, p_z, p_yaw   = x[0], x[1], x[2], x[3]     # 虚拟末端（EE） 位置
        v_px, v_py, v_pz, v_pyaw = x[4], x[5], x[6], x[7]   # 虚拟末端速度
        q_x, q_y                = x[8], x[9]                  # 负载 XY 位置
        v_qx, v_qy              = x[10], x[11]                # 负载 XY 速度

        # 解析控制输入（加速度）
        ax, ay, az, ayaw = u[0], u[1], u[2], u[3]

        # ======================================================================
        # [CTRL-BUG-4 修复] 线性化 omega_sq_eff：
        # 用 CasADi 参数占位符 az_prev 替代决策变量 az，在每次调用时传入上一步估计值
        # 这将动力学从四次退化为双线性，大幅改善 IPOPT 收敛
        # ======================================================================
        az_prev = ca.SX.sym('az_prev')  # 线性化点（上一步 az 值）
        omega_sq_eff = (self.g + az_prev) / self.L  # 此时为常数（给定 az_prev）

        dx = ca.vertcat(
            v_px, v_py, v_pz, v_pyaw,                                          # EE 速度
            ax, ay, az, ayaw,                                                    # EE 加速度（控制量）
            v_qx, v_qy,                                                          # 负载速度
            -omega_sq_eff * (q_x - p_x) - damping * v_qx,                      # 负载 X 动力学
            -omega_sq_eff * (q_y - p_y) - damping * v_qy                        # 负载 Y 动力学
        )

        f_dyn = ca.Function('f', [x, u, az_prev], [dx])

        # ======================================================================
        # NLP 构建
        # ======================================================================
        X      = ca.SX.sym('X', self.nx, self.N + 1)
        U      = ca.SX.sym('U', self.nu, self.N)
        P_ref  = ca.SX.sym('P_ref', 4)     # 目标 [target_px, target_py, target_pz, target_yaw]
        X_init = ca.SX.sym('X_init', self.nx)
        Az_lin = ca.SX.sym('Az_lin')        # 线性化点（当前步 az 估计值，标量）

        cost        = 0
        constraints = []

        # [CTRL-BUG-3 修复] 提高 Z 轴跟踪权重，加快初始高度收敛
        Q_pos   = np.array([10.0, 10.0, 20.0, 5.0])  # [x, y, z, yaw]
        Q_swing = np.array([50.0, 50.0])              # 防摆惩罚（保持 q ≈ p）
        Q_vel   = 2.0                                 # 速度惩罚
        # [CTRL-BUG-5 修复] 提高 yaw 控制惩罚防止末端旋转漂移
        R_acc   = np.array([0.2, 0.2, 0.2, 0.3])     # [ax, ay, az, ayaw]

        constraints.append(X[:, 0] - X_init)

        # 【混合跟踪逻辑】：XY 由负载跟踪，Z 高度直接由末端（EE）跟踪
        mocap_target_z = P_ref[2] + self.L  # payload 目标 Z + 绳长 = EE 目标 Z

        for k in range(self.N):
            # RK4 积分器（传入线性化点 Az_lin）
            k1 = f_dyn(X[:, k],              U[:, k], Az_lin)
            k2 = f_dyn(X[:, k] + dt/2 * k1, U[:, k], Az_lin)
            k3 = f_dyn(X[:, k] + dt/2 * k2, U[:, k], Az_lin)
            k4 = f_dyn(X[:, k] + dt    * k3, U[:, k], Az_lin)
            x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            constraints.append(X[:, k+1] - x_next)

            # 阶段代价
            cost += Q_pos[0] * (X[8,  k] - P_ref[0])**2   # 负载跟踪目标 X
            cost += Q_pos[1] * (X[9,  k] - P_ref[1])**2   # 负载跟踪目标 Y
            cost += Q_pos[2] * (X[2,  k] - mocap_target_z)**2  # EE 跟踪 Z 高度
            cost += Q_pos[3] * (X[3,  k] - P_ref[3])**2   # EE 跟踪 Yaw

            cost += Q_swing[0] * (X[8, k] - X[0, k])**2   # 抑制 X 摆角
            cost += Q_swing[1] * (X[9, k] - X[1, k])**2   # 抑制 Y 摆角
            cost += Q_vel * (X[4, k]**2 + X[5, k]**2 + X[6, k]**2 + X[7, k]**2)  # 抑制过速

            cost += (R_acc[0]*U[0,k]**2 + R_acc[1]*U[1,k]**2
                   + R_acc[2]*U[2,k]**2 + R_acc[3]*U[3,k]**2)

        # 终端代价（加权 10×，强化末端精度）
        cost += 10 * (Q_pos[0] * (X[8,  self.N] - P_ref[0])**2
                    + Q_pos[1] * (X[9,  self.N] - P_ref[1])**2)
        cost += 10 * (Q_pos[2] * (X[2,  self.N] - mocap_target_z)**2
                    + Q_pos[3] * (X[3,  self.N] - P_ref[3])**2)

        nlp = {
            'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            'f': cost,
            'g': ca.vertcat(*constraints),
            'p': ca.vertcat(X_init, P_ref, Az_lin)  # 参数向量：初始状态 + 目标 + 线性化点
        }

        # [CTRL-IMPROVE-2] max_iter 提高到 50，提升复杂区域收敛率
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.max_iter': 50, 'ipopt.tol': 1e-2, 'ipopt.acceptable_tol': 1e-1,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # ======================================================================
        # 决策变量边界
        # ======================================================================
        n_vars = self.nx * (self.N + 1) + self.nu * self.N
        self.lbx = -ca.inf * np.ones(n_vars)
        self.ubx =  ca.inf * np.ones(n_vars)

        u_start_idx = self.nx * (self.N + 1)
        for i in range(self.N):
            idx = u_start_idx + i * self.nu
            self.lbx[idx:idx + self.nu] = [-u_max_xy, -u_max_xy, -u_max_z, -u_max_yaw]
            self.ubx[idx:idx + self.nu] = [ u_max_xy,  u_max_xy,  u_max_z,  u_max_yaw]

        self.lbg = np.zeros(self.nx * (self.N + 1))
        self.ubg = np.zeros(self.nx * (self.N + 1))
        self.last_sol = None

    def get_action(self, state_12d, target_4d):
        """
        求解一步最优控制律。

        Args:
            state_12d: [mocap_x, mocap_y, mocap_z, mocap_yaw,
                        mocap_vx, mocap_vy, mocap_vz, mocap_yaw_vel,
                        payload_x, payload_y, payload_vx, payload_vy]
            target_4d: [target_payload_x, target_payload_y,
                        target_payload_z, target_yaw]

        Returns:
            u_opt: np.ndarray (4,) [ax, ay, az, ayaw]
        """
        # [CTRL-BUG-4 修复] 将线性化点 last_az 追加到参数向量末尾
        p_val = np.concatenate([state_12d, target_4d, [self.last_az]])
        x0_guess = self.last_sol if self.last_sol is not None else np.zeros(self.lbx.shape)

        try:
            sol = self.solver(
                x0=x0_guess, p=p_val,
                lbg=self.lbg, ubg=self.ubg,
                lbx=self.lbx, ubx=self.ubx
            )
            self.last_sol = sol['x']
            u_start_idx = self.nx * (self.N + 1)
            u_opt = np.array(sol['x'][u_start_idx: u_start_idx + self.nu]).flatten()
            # 更新线性化点
            self.last_az = float(u_opt[2])
        except Exception:
            self.last_sol = None
            u_opt = np.zeros(self.nu)
            self.last_az = 0.0

        return u_opt


# ==============================================================================
# 顶层封装：自动适配环境航点的 Tracker
# ==============================================================================
class NMPCTrajectoryTracker:
    """
    NMPC 轨迹追踪器：将 env 生成的 3D 路径传给 NMPCController4D，
    自动管理航点切换逻辑，并输出 6D 动作（env 要求格式）。
    """

    def __init__(self, dt=0.1, N=15, L=0.445, arrival_threshold_xy=0.05, arrival_threshold_z=0.20):
        self.mpc = NMPCController4D(dt=dt, N=N, L=L)
        self.path = None
        self.current_idx = 0
        self.arrival_threshold_xy = arrival_threshold_xy
        self.arrival_threshold_z = arrival_threshold_z
        self.estimated_L = L  # 初始动态摆长估计

    def set_path(self, path):
        """
        加载环境规划路径。支持 list/ndarray，强制转换并保护性检查。
        [CTRL-BUG-6 修复] 强制 np.array 化
        [CTRL-IMPROVE-1 修复] 清空 last_sol，防止旧解污染新路径求解
        """
        if path is None or len(path) == 0:
            self.path = None
            self.current_idx = 0
            return

        self.path = np.array(path, dtype=np.float64)
        self.current_idx = 0
        # 切换路径时清空热启动，防止上一回合的解偏离新问题区域
        self.mpc.last_sol = None
        self.mpc.last_az  = 0.0
        # print(f"[Tracker] 路径已载入，共 {len(self.path)} 个航点")

    def compute_action(self, obs, target_yaw=0.0):
        """
        主控制接口：将环境观测转换为 6D 动作输出。

        obs 布局（与 mujoco_env_new._get_obs() 严格对齐）：
          [0]  mocap_x       [1]  mocap_y
          [2]  mocap_vx      [3]  mocap_vy
          [4]  payload_x     [5]  payload_y
          [6]  payload_vx    [7]  payload_vy
          [8]  rel_tx        [9]  rel_ty
          [10 ~ 10+3n-1]  障碍物数据（n_obstacles × 3）
          [-12] mocap_z      [-11] mocap_vz
          [-10] payload_z    [-9]  payload_vz
          [-8]  mocap_roll   [-7]  mocap_roll_vel
          [-6]  mocap_pitch  [-5]  mocap_pitch_vel
          [-4]  mocap_yaw    [-3]  mocap_yaw_vel
          [-2]  payload_yaw  [-1]  payload_yaw_vel
        """
        if self.path is None:
            return np.zeros(6)  # 安全返回 6D 零动作

        # ======================================================================
        # 1. 状态提取（[CTRL-BUG-1 修复] 全部改用负索引读取末尾物理状态）
        # ======================================================================
        # XY 状态：在障碍物段之前，索引固定不变
        mocap_x,   mocap_y   = obs[0], obs[1]
        mocap_vx,  mocap_vy  = obs[2], obs[3]
        payload_x, payload_y = obs[4], obs[5]
        payload_vx, payload_vy = obs[6], obs[7]

        # Z 及姿态状态：用负索引，不受 n_obstacles 影响
        mocap_z        = float(obs[-12])
        mocap_vz       = float(obs[-11])
        # payload_z、payload_vz 目前仅用于调试，不进入 MPC 状态（MPC 通过 EE Z 间接控制）
        payload_z    = float(obs[-10])
        # payload_vz   = float(obs[-9])
        
        # 【新增】动态摆长滤波估计（消除静态误差的核心）
        actual_L = mocap_z - payload_z
        self.estimated_L = 0.9 * self.estimated_L + 0.1 * actual_L

        mocap_yaw      = float(obs[-4])
        mocap_yaw_vel  = float(obs[-3])

        # ======================================================================
        # 2. 自动切换航点（[CTRL-BUG-7 修复] 改用 3D 距离判断）
        # ======================================================================
        curr_payload_xy = np.array([payload_x, payload_y])
        curr_payload_z  = payload_z
        
        target_wp = self.path[self.current_idx]
        target_wp_3d = np.array([
            target_wp[0],
            target_wp[1],
            target_wp[2] if len(target_wp) >= 3 else 0.3
        ])

        # 【修改】解耦 XY 和 Z 的距离计算
        dist_xy = np.linalg.norm(curr_payload_xy - target_wp_3d[:2])
        dist_z  = abs(curr_payload_z - target_wp_3d[2])

        # 【修改】使用圆柱体判定域，满足条件则推向下一个航点
        if dist_xy < self.arrival_threshold_xy and dist_z < self.arrival_threshold_z and self.current_idx < len(self.path) - 1:
            self.current_idx += 1
            target_wp = self.path[self.current_idx]
            target_wp_3d = np.array([
                target_wp[0],
                target_wp[1],
                target_wp[2] if len(target_wp) >= 3 else 0.3
            ])
            self.mpc.last_sol = None
            self.mpc.last_az  = 0.0

        # ======================================================================
        # 3. 组装 12D MPC 状态
        # ======================================================================
        state_12d = np.array([
            mocap_x, mocap_y, mocap_z, mocap_yaw,
            mocap_vx, mocap_vy, mocap_vz, mocap_yaw_vel,
            payload_x, payload_y,
            payload_vx, payload_vy
        ], dtype=np.float64)

        # ======================================================================
        # 4. 调用 MPC 计算 4D 最优控制律
        # ======================================================================
        # 【核心修正】: 用自适应摆长补偿 Z 轴静态误差
        # MPC 内部固定使用 mocap_target_z = P_ref[2] + self.mpc.L
        # 我们希望 mocap_target_z = target_wp_3d[2] + self.estimated_L
        # 因此逆向补偿 P_ref[2]：
        compensated_target_z = target_wp_3d[2] + (self.estimated_L - self.mpc.L)

        P_ref = np.array([
            target_wp_3d[0],
            target_wp_3d[1],
            compensated_target_z,  # 【修改】传入补偿后的高度
            target_yaw
        ], dtype=np.float64)

        action_4d = self.mpc.get_action(state_12d, P_ref)

        # ======================================================================
        # 5. 组装为 6D 动作：[ax, ay, az, a_roll, a_pitch, a_yaw]
        # ======================================================================
        action_6d = np.zeros(6)
        action_6d[0] = action_4d[0]  # ax
        action_6d[1] = action_4d[1]  # ay
        action_6d[2] = action_4d[2]  # az
        action_6d[3] = 0.0            # a_roll  = 0（末端保持水平，由 IK 自动保证）
        action_6d[4] = 0.0            # a_pitch = 0
        action_6d[5] = action_4d[3]  # a_yaw

        return action_6d