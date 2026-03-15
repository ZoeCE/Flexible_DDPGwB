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
    优化后的纯轨迹跟踪 NMPC
    修复了权重不对称导致的甩尾现象，以及使用负载位置判定跳点带来的共振发散问题。
    """

    def __init__(self, dt=0.02, N=20, L=0.6, u_max=0.5):
        self.dt = dt
        self.N = N       
        self.L = L       
        self.g = 9.81
        self.u_max = u_max
        self.nx = 8      
        self.nu = 2      
        
        self.trajectory_xy = []
        self.current_wp_idx = 0
        self.reached_final = False

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
        eq_constraints = []

        # 【核心优化 1：恢复对称的代价函数】
        # XY方向的惩罚必须完全一致，否则在二维平面斜向运动时会导致剧烈的扭矩和甩尾
        Q_pos = np.array([20.0, 20.0])   # 适度降低极端的跟踪需求，给予系统缓冲
        Q_swing = np.array([20.0, 20.0]) # 强化对称的防摆权重
        Q_vel = 1.0                      
        R_acc = 0.1                      

        P_ref = ca.SX.sym('P_ref', 2)    
        X_init = ca.SX.sym('X_init', self.nx)
        
        p_sym = ca.vertcat(X_init, P_ref)

        eq_constraints.append(X[:, 0] - X_init)

        for k in range(self.N):
            k1 = f_dyn(X[:, k], U[:, k])
            k2 = f_dyn(X[:, k] + dt/2 * k1, U[:, k])
            k3 = f_dyn(X[:, k] + dt/2 * k2, U[:, k])
            k4 = f_dyn(X[:, k] + dt * k3, U[:, k])
            x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            eq_constraints.append(X[:, k+1] - x_next)

            cost += Q_pos[0] * (X[4, k] - P_ref[0])**2 + Q_pos[1] * (X[5, k] - P_ref[1])**2
            cost += Q_swing[0] * (X[4, k] - X[0, k])**2 + Q_swing[1] * (X[5, k] - X[1, k])**2
            cost += Q_vel * (X[2, k]**2 + X[3, k]**2)
            cost += R_acc * (U[0, k]**2 + U[1, k]**2)

        cost += 10 * (Q_pos[0] * (X[4, self.N] - P_ref[0])**2 + Q_pos[1] * (X[5, self.N] - P_ref[1])**2)

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
        
        u_start_idx = self.nx * (self.N + 1)
        self.lbx[u_start_idx:] = -self.u_max
        self.ubx[u_start_idx:] = self.u_max

        self.lbg = np.zeros(n_eq)
        self.ubg = np.zeros(n_eq)

        self.last_sol = None

    def get_action(self, state, target_pos):
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
        if path_array is None or len(path_array) == 0:
            self.trajectory_xy = []
            self.current_wp_idx = 0
            self.reached_final = False
            return
            
        self.trajectory_xy = path_array
        self.current_wp_idx = 0
        self.reached_final = False

    def get_tracking_action(self, state):
        if self.trajectory_xy is None or len(self.trajectory_xy) == 0:
            return self.get_action(state, state[4:6])

        target_xy = self.trajectory_xy[self.current_wp_idx]
        
        # 【核心优化 2：改用台车(推车)的坐标进行距离判断】
        # state[0], state[1] 分别对应物理系统的 p_x 和 p_y (即台车XY坐标)
        p_x, p_y = state[0], state[1]
        
        # 阻断正反馈：使用绝对刚性的台车位置判断，摆锤再怎么晃动，也不会引发系统误判而错误加速
        dist = np.linalg.norm([p_x - target_xy[0], p_y - target_xy[1]])
        total_wps = len(self.trajectory_xy)
        rem_wps = (total_wps - 1) - self.current_wp_idx
        brake_zone = 5  

        if not self.reached_final:
            if rem_wps <= brake_zone:
                look_ahead_dist = 0.02
                max_step = 1
            else:
                # 巡航区使用更宽泛的判定，保证台车流畅拉着负载走
                look_ahead_dist = 0.04
                max_step = 1

            if dist < look_ahead_dist:
                step = min(max_step, rem_wps)
                self.current_wp_idx += step
                
                if self.current_wp_idx >= total_wps - 1:
                    self.current_wp_idx = total_wps - 1
                    self.reached_final = True
                
                target_xy = self.trajectory_xy[self.current_wp_idx]

        return self.get_action(state, target_xy)