import numpy as np
import casadi as ca

class NMPCController:
    def __init__(self, dt=0.02, N=20, L=0.6, u_max=0.5):
        self.dt = dt
        self.N = N
        self.L = L
        self.g = 9.81
        self.u_max = u_max
        self.nx = 8
        self.nu = 2 
        
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        
        damping = 0.01
        omega_sq = self.g / self.L
        
        p_x, p_y = x[0], x[1]
        v_px, v_py = x[2], x[3]
        q_x, q_y = x[4], x[5]
        v_qx, v_qy = x[6], x[7]
        ax, ay = u[0], u[1]
        
        dx = ca.vertcat(
            v_px, v_py, ax, ay, v_qx, v_qy,
            -omega_sq * (q_x - p_x) - damping * v_qx,
            -omega_sq * (q_y - p_y) - damping * v_qy
        )
        
        f_dyn = ca.Function('f', [x, u], [dx])
        X = ca.SX.sym('X', self.nx, self.N + 1)
        U = ca.SX.sym('U', self.nu, self.N)
        
        cost = 0
        constraints = []
        
        Q_pos = np.array([10.0, 80.0])
        Q_swing = np.array([20.0, 10.0])
        Q_vel = 1.0
        R_acc = 0.1
        
        P_ref = ca.SX.sym('P_ref', 2)
        X_init = ca.SX.sym('X_init', self.nx)
        
        constraints.append(X[:, 0] - X_init)
        
        for k in range(self.N):
            k1 = f_dyn(X[:, k], U[:, k])
            k2 = f_dyn(X[:, k] + dt/2 * k1, U[:, k])
            k3 = f_dyn(X[:, k] + dt/2 * k2, U[:, k])
            k4 = f_dyn(X[:, k] + dt * k3, U[:, k])
            x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            constraints.append(X[:, k+1] - x_next)
            
            cost += Q_pos[0] * (X[4, k] - P_ref[0])**2 + Q_pos[1] * (X[5, k] - P_ref[1])**2
            cost += Q_swing[0] * (X[4, k] - X[0, k])**2 + Q_swing[1] * (X[5, k] - X[1, k])**2
            cost += Q_vel * (X[2, k]**2 + X[3, k]**2)
            cost += R_acc * (U[0, k]**2 + U[1, k]**2)
            
        cost += 10 * (Q_pos[0] * (X[4, self.N] - P_ref[0])**2 + Q_pos[1] * (X[5, self.N] - P_ref[1])**2)
        
        nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
               'f': cost,
               'g': ca.vertcat(*constraints),
               'p': ca.vertcat(X_init, P_ref)}
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 20,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-1,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        n_vars = self.nx * (self.N + 1) + self.nu * self.N
        self.lbx = -ca.inf * np.ones(n_vars)
        self.ubx =  ca.inf * np.ones(n_vars)
        u_start_idx = self.nx * (self.N + 1)
        self.lbx[u_start_idx:] = -self.u_max
        self.ubx[u_start_idx:] =  self.u_max
        
        self.last_sol = None

    def get_action(self, state, target_pos):
        p_val = np.concatenate([state, target_pos])
        x0_guess = np.zeros(self.lbx.shape)
        if self.last_sol is not None:
             x0_guess = self.last_sol
        
        try:
            sol = self.solver(x0=x0_guess, p=p_val, lbg=0, ubg=0, lbx=self.lbx, ubx=self.ubx)
            self.last_sol = sol['x']
            u_start_idx = self.nx * (self.N + 1)
            u_opt = sol['x'][u_start_idx : u_start_idx + self.nu]
            return np.array(u_opt).flatten()
        except:
            self.last_sol = None
            return np.zeros(2)


# ---------------------------------------------------------------------------
# 带障碍物避碰的 NMPC 控制器
# ---------------------------------------------------------------------------

class NMPCTrajectoryTracker:
    """
    纯轨迹跟踪 NMPC：不包含任何避障逻辑。
    专注于动力学预测、防摆控制以及基于前瞻 (Look-ahead) 的密集轨迹跟踪。
    """

    def __init__(self, dt=0.02, N=20, L=0.6, u_max=0.5):
        self.dt = dt
        self.N = N       # 预测步数
        self.L = L       # 绳长
        self.g = 9.81
        self.u_max = u_max
        self.nx = 8      # 状态维度
        self.nu = 2      # 动作维度 (ax, ay)
        
        # --- 轨迹跟踪相关状态 ---
        self.trajectory_xy = []
        self.current_wp_idx = 0
        self.reached_final = False

        # --- NMPC 求解器构建 ---
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        damping = 0.01
        omega_sq = self.g / self.L
        
        # 状态变量映射
        p_x, p_y = x[0], x[1]     # 台车位置
        v_px, v_py = x[2], x[3]   # 台车速度
        q_x, q_y = x[4], x[5]     # 负载位置
        v_qx, v_qy = x[6], x[7]   # 负载速度
        ax, ay = u[0], u[1]       # 控制输入：台车加速度
        
        # 连续时间动力学方程
        dx = ca.vertcat(
            v_px, v_py, ax, ay, v_qx, v_qy,
            -omega_sq * (q_x - p_x) - damping * v_qx,
            -omega_sq * (q_y - p_y) - damping * v_qy
        )
        f_dyn = ca.Function('f', [x, u], [dx])

        X = ca.SX.sym('X', self.nx, self.N + 1)
        U = ca.SX.sym('U', self.nu, self.N)
        cost = 0
        eq_constraints = []

        # 代价函数权重
        Q_pos = np.array([10.0, 80.0])   # 目标位置跟踪权重
        Q_swing = np.array([20.0, 10.0]) # 防摆权重
        Q_vel = 1.0                      # 速度抑制权重
        R_acc = 0.1                      # 控制输入平滑权重

        P_ref = ca.SX.sym('P_ref', 2)    # 当前追踪的局部目标点
        X_init = ca.SX.sym('X_init', self.nx)
        
        # 参数仅包含初始状态和目标点，不再有 obstacles
        p_sym = ca.vertcat(X_init, P_ref)

        # 初始状态等式约束
        eq_constraints.append(X[:, 0] - X_init)

        # RK4 离散化与多重打靶法 (Multiple Shooting)
        for k in range(self.N):
            k1 = f_dyn(X[:, k], U[:, k])
            k2 = f_dyn(X[:, k] + dt/2 * k1, U[:, k])
            k3 = f_dyn(X[:, k] + dt/2 * k2, U[:, k])
            k4 = f_dyn(X[:, k] + dt * k3, U[:, k])
            x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            eq_constraints.append(X[:, k+1] - x_next)

            # 累加预测步代价
            cost += Q_pos[0] * (X[4, k] - P_ref[0])**2 + Q_pos[1] * (X[5, k] - P_ref[1])**2
            cost += Q_swing[0] * (X[4, k] - X[0, k])**2 + Q_swing[1] * (X[5, k] - X[1, k])**2
            cost += Q_vel * (X[2, k]**2 + X[3, k]**2)
            cost += R_acc * (U[0, k]**2 + U[1, k]**2)

        # 终端代价 (Terminal Cost)
        cost += 10 * (Q_pos[0] * (X[4, self.N] - P_ref[0])**2 + Q_pos[1] * (X[5, self.N] - P_ref[1])**2)

        # 组合约束：仅保留等式约束
        g_eq = ca.vertcat(*eq_constraints)
        n_eq = g_eq.size1()

        nlp = {
            'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            'f': cost,
            'g': g_eq,
            'p': p_sym
        }
        
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.max_iter': 50, 'ipopt.tol': 1e-2, 'ipopt.acceptable_tol': 1e-1,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = self.nx * (self.N + 1) + self.nu * self.N
        self.lbx = -ca.inf * np.ones(n_vars)
        self.ubx = ca.inf * np.ones(n_vars)
        
        # 动作界限约束 u_max
        u_start_idx = self.nx * (self.N + 1)
        self.lbx[u_start_idx:] = -self.u_max
        self.ubx[u_start_idx:] = self.u_max

        # 约束边界全为 0 (等式约束)
        self.lbg = np.zeros(n_eq)
        self.ubg = np.zeros(n_eq)

        self.last_sol = None

    def get_action(self, state, target_pos):
        """
        底层 NMPC 动作求解 (无障碍物版本)
        """
        p_val = np.concatenate([state, target_pos])

        x0_guess = np.zeros(self.lbx.shape)
        if self.last_sol is not None:
            x0_guess = self.last_sol

        try:
            sol = self.solver(
                x0=x0_guess, p=p_val,
                lbg=self.lbg, ubg=self.ubg,
                lbx=self.lbx, ubx=self.ubx
            )
            self.last_sol = sol['x']
            u_start_idx = self.nx * (self.N + 1)
            u_opt = sol['x'][u_start_idx : u_start_idx + self.nu]
            return np.array(u_opt).flatten()
        except Exception:
            self.last_sol = None
            return np.zeros(2)
    
    def set_trajectory(self, path_array):
        """
        传入 2D 轨迹 (N, 2)
        """
        if path_array is None or len(path_array) == 0:
            self.trajectory_xy = []
            self.current_wp_idx = 0
            self.reached_final = False
            return
            
        self.trajectory_xy = path_array
        self.current_wp_idx = 0
        self.reached_final = False

    def get_tracking_action(self, state):
        """
        高级轨迹调度：基于密集路点的前瞻控制 (Look-ahead)
        去除人为动作衰减，完全信任 NMPC 底层的防摆与定点能力。
        """
        if self.trajectory_xy is None or len(self.trajectory_xy) == 0:
            return self.get_action(state, state[4:6])

        # 1. 获取当前状态与目标点
        target_xy = self.trajectory_xy[self.current_wp_idx]
        q_x, q_y = state[4], state[5]
        dist = np.linalg.norm([q_x - target_xy[0], q_y - target_xy[1]])
        total_wps = len(self.trajectory_xy)

        # 2. 计算距离终点的剩余路点数
        rem_wps = (total_wps - 1) - self.current_wp_idx
        brake_zone = 10  # 定义最后 5 个点为刹车区

        # 3. 动态前瞻与步进逻辑 (空间减速机制保留)
        if not self.reached_final:
            if rem_wps <= brake_zone:
                # 【刹车区】：收紧前瞻距离，停止跳点，强制系统慢速精细逼近
                look_ahead_dist = 0.01
                max_step = 1
            else:
                # 【巡航区】：较宽的前瞻距离，允许跳点以保持高速和流畅
                look_ahead_dist = 0.05
                max_step = 1

            if dist < look_ahead_dist:
                step = min(max_step, rem_wps)
                self.current_wp_idx += step
                
                # 更新状态
                if self.current_wp_idx >= total_wps - 1:
                    self.current_wp_idx = total_wps - 1
                    self.reached_final = True
                
                # 目标点已切换，更新 target_xy
                target_xy = self.trajectory_xy[self.current_wp_idx]

        # 4. 调用底层 NMPC 求解当前目标点的动作
        # 【修改核心】：与 Base Controller 保持一致，直接返回完整的 NMPC 输出！
        # 依赖其内部的 Q_pos 和 Q_swing 权重自然地实现柔性定点和防摆。
        return self.get_action(state, target_xy)