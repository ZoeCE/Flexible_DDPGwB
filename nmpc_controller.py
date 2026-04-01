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
        
        Q_pos = np.array([10.0, 10.0])
        Q_swing = np.array([50.0, 50.0]) 
        Q_vel = 2.0                      
        R_acc = 0.2                      
        
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
# 带障碍物避碰的 NMPC 控制器 (兼容 3D 轨迹)
# ---------------------------------------------------------------------------

class NMPCTrajectoryTracker:
    """
    优化后的轨迹跟踪 NMPC
    全面升级为 3D 轨迹跟随模式，解耦 XY 的 NMPC 防摆与 Z 轴的动态下潜。
    """

    def __init__(self, dt=0.02, N=20, L=0.6, u_max=0.5):
        self.dt = dt
        self.N = N       
        self.L = L       
        self.g = 9.81
        self.u_max = u_max
        self.nx = 8      
        self.nu = 2      
        
        self.trajectory = []
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

        # [中和修改]：兼顾速度与防摆的折中方案
        Q_pos = np.array([20.0, 20.0])   
        Q_swing = np.array([50.0, 50.0]) # 中度防摆
        Q_vel = 2.0                      # 中度限制速度
        R_acc = 0.2                      # 中度平滑加速度

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

    def reset_state_machine(self):
        """每局环境 reset 时调用，重置状态"""
        self.current_wp_idx = 0
        self.reached_final = False

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
            self.trajectory = []
            self.reset_state_machine()
            return
            
        self.trajectory = path_array
        self.reset_state_machine()

    def get_tracking_action(self, full_obs, env_wp_idx=None):
        # ---------------------------------------------------------
        # 1. 适配新版 Env 的状态解析 (关键修复)
        # ---------------------------------------------------------
        nmpc_s = full_obs[:8]  
        q_x, q_y = full_obs[4], full_obs[5] 
        v_qx, v_qy = full_obs[6], full_obs[7]
        
        # ❗因为末尾增加了7维姿态数据，Z轴相关数据的索引必须往前推7位
        mocap_z    = full_obs[-11]
        mocap_vz   = full_obs[-10]
        payload_z  = full_obs[-9]
        payload_vz = full_obs[-8] 
        
        # 提取新加入的姿态数据
        prefab_quat   = full_obs[-7:-3] # [w, x, y, z]
        prefab_angvel = full_obs[-3:]   # [wx, wy, wz]

        # ---------------------------------------------------------
        # 2. 目标点与步进逻辑
        # ---------------------------------------------------------
        if self.trajectory is None or len(self.trajectory) == 0:
            target_xy = [0.0, 0.0] 
            target_z = 0.4
            action_2d = self.get_action(nmpc_s, target_xy)
        else:
            total_wps = len(self.trajectory)
            if env_wp_idx is not None:
                self.current_wp_idx = min(env_wp_idx, total_wps - 1)
                if self.current_wp_idx >= total_wps - 1:
                    self.reached_final = True
            else:
                target_wp_current = self.trajectory[self.current_wp_idx]
                current_pos_3d = np.array([q_x, q_y, payload_z])
                dist_3d = np.linalg.norm(current_pos_3d - target_wp_current)
                
                if not self.reached_final:
                    rem_wps = total_wps - 1 - self.current_wp_idx
                    look_ahead_dist = 0.06 if rem_wps <= 2 else 0.10 
                    if dist_3d < look_ahead_dist:
                        step = min(1, rem_wps)
                        self.current_wp_idx += step
                        if self.current_wp_idx >= total_wps - 1:
                            self.current_wp_idx = total_wps - 1
                            self.reached_final = True
            
            target_wp = self.trajectory[self.current_wp_idx]
            target_xy = target_wp[:2]
            target_z = target_wp[2] if len(target_wp) >= 3 else 0.4
            action_2d = self.get_action(nmpc_s, target_xy)

        # ---------------------------------------------------------
        # 3. Z 轴双重阻尼控制器 (防止砸地)
        # ---------------------------------------------------------
        Kp_z = 4.0 
        Kd_mocap = 2.0
        Kd_payload = 3.5 
        az = Kp_z * (target_z - payload_z) - Kd_mocap * mocap_vz - Kd_payload * payload_vz
        az = np.clip(az, -self.u_max, self.u_max)

        # ---------------------------------------------------------
        # 4. ✅ 新增：Z 轴旋转 (Yaw) 闭环反馈控制器
        # ---------------------------------------------------------
        from scipy.spatial.transform import Rotation as R
        
        # MuJoCo 的四元数格式是 [w, x, y, z]，SciPy 期望的是 [x, y, z, w]
        quat_scipy = [prefab_quat[1], prefab_quat[2], prefab_quat[3], prefab_quat[0]]
        
        try:
            r = R.from_quat(quat_scipy)
            # zyx 顺序，返回的第一个元素就是绕 Z 轴的欧拉角 (Yaw)
            current_yaw = r.as_euler('zyx', degrees=False)[0] 
        except Exception:
            current_yaw = 0.0 # 出现异常时默认不旋转
            
        current_yaw_vel = prefab_angvel[2] # 绕 Z 轴的角速度
        
        target_yaw = 0.0 # 目标航向角（套圆柱任务中最好保持 0 以防扭摆）
        Kp_yaw = 3.0
        Kd_yaw = 1.0
        
        # 计算 Yaw 轴角加速度
        a_yaw = Kp_yaw * (target_yaw - current_yaw) - Kd_yaw * current_yaw_vel
        a_yaw = np.clip(a_yaw, -1.0, 1.0) # 旋转速度限幅可以稍微放宽一点

        # ---------------------------------------------------------
        # 5. 合并为 4D 动作输出
        # ---------------------------------------------------------
        action_2d = np.clip(action_2d, -self.u_max, self.u_max)
        base_action = np.array([action_2d[0], action_2d[1], az, a_yaw], dtype=np.float32)

        return base_action