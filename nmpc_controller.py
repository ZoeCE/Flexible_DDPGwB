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

import numpy as np
import casadi as ca

class NMPCTrajectoryTracker:
    """
    优化后的轨迹跟踪 NMPC
    全面升级为 3D 轨迹跟随模式，解耦 XY 的 NMPC 防摆与 Z 轴及 3D 姿态的主动控制。
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

        # --- NMPC CasADi 优化器初始化 (针对 XY 平面平移防摆) ---
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

        Q_pos = np.array([20.0, 20.0])   
        Q_swing = np.array([50.0, 50.0]) 
        Q_vel = 2.0                      
        R_acc = 0.2                      

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
        """底层的 2D NMPC 优化求解器"""
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
        """加载环境规划出的 3D 轨迹点"""
        if path_array is None or len(path_array) == 0:
            self.trajectory = []
            self.reset_state_machine()
            return
            
        self.trajectory = path_array
        self.reset_state_machine()

    def get_tracking_action(self, full_obs, env_wp_idx=None):
        """
        核心控制接口：融合 2D NMPC 防摆、Z轴下潜 与 3D 姿态控制
        """
        # ---------------------------------------------------------
        # 1. 状态提取 (期待 Env 更新后的观测布局)
        # ---------------------------------------------------------
        # 前 8 维：无缝喂给 NMPC [mocap_x/y, mocap_vx/vy, payload_x/y, payload_vx/vy]
        nmpc_s = full_obs[:8]  
        q_x, q_y = nmpc_s[4], nmpc_s[5] 
        
        # 为了兼容未来的 Env 升级，采用负索引提取最后的 12 维物理状态：
        # [-12] mocap_z,      [-11] mocap_vz,   [-10] payload_z, [-9] payload_vz
        # [-8]  mocap_roll,   [-7]  mocap_roll_vel, [-6] mocap_pitch, [-5] mocap_pitch_vel
        # [-4]  mocap_yaw,    [-3]  mocap_yaw_vel,  [-2] payload_yaw, [-1] payload_yaw_vel
        
        # 为了防止当前版本 Env 尚未提供 roll/pitch 导致数组越界，做个柔性兼容：
        if len(full_obs) >= 20: 
            mocap_vz   = full_obs[-11]
            payload_z  = full_obs[-10]
            payload_vz = full_obs[-9]
            
            mocap_roll      = full_obs[-8]
            mocap_roll_vel  = full_obs[-7]
            mocap_pitch     = full_obs[-6]
            mocap_pitch_vel = full_obs[-5]
            mocap_yaw       = full_obs[-4] 
            mocap_yaw_vel   = full_obs[-3] 
        else:
            # 兼容老版本 4D 环境（待 Env 更新后该分支将不再触发）
            mocap_vz   = full_obs[-7]
            payload_z  = full_obs[-6]
            payload_vz = full_obs[-5]
            mocap_roll, mocap_roll_vel, mocap_pitch, mocap_pitch_vel = 0.0, 0.0, 0.0, 0.0
            mocap_yaw       = full_obs[-4] 
            mocap_yaw_vel   = full_obs[-3] 

        # ---------------------------------------------------------
        # 2. 目标点与 2D 步进逻辑 (降维打击防死锁)
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
                
                # 仅算 XY 距离，让 Z 轴作为独立通道慢慢跟，防止卡死
                current_pos_2d = np.array([q_x, q_y])
                target_wp_2d = target_wp_current[:2]
                dist_2d = np.linalg.norm(current_pos_2d - target_wp_2d)
                
                if not self.reached_final:
                    rem_wps = total_wps - 1 - self.current_wp_idx
                    look_ahead_dist = 0.08 if rem_wps <= 2 else 0.15 
                    if dist_2d < look_ahead_dist:
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
        # 3. Z 轴双重阻尼控制器 (解除限幅)
        # ---------------------------------------------------------
        Kp_z = 8.0     
        Kd_mocap = 3.0
        Kd_payload = 4.0 
        az = Kp_z * (target_z - payload_z) - Kd_mocap * mocap_vz - Kd_payload * payload_vz
        az = np.clip(az, -2.0, 2.0) 

        # ---------------------------------------------------------
        # 4. 【核心升级】完整的 3D 姿态 (Roll, Pitch, Yaw) 主动稳定器
        # ---------------------------------------------------------
        # 强制目标姿态保持绝对水平且不发生偏航转动
        target_roll, target_pitch, target_yaw = 0.0, 0.0, 0.0
        
        # 姿态控制刚度 (保持水平通常需要较强的刚性反馈)
        Kp_attitude = 5.0
        Kd_attitude = 1.5

        # 计算三轴主动角加速度反馈 (PD 闭环)
        a_roll  = Kp_attitude * (target_roll - mocap_roll)   - Kd_attitude * mocap_roll_vel
        a_pitch = Kp_attitude * (target_pitch - mocap_pitch) - Kd_attitude * mocap_pitch_vel
        a_yaw   = Kp_attitude * (target_yaw - mocap_yaw)     - Kd_attitude * mocap_yaw_vel

        # 对角加速度限幅，防止动作剧烈造成数值爆炸或翻车
        a_roll  = np.clip(a_roll, -2.0, 2.0)
        a_pitch = np.clip(a_pitch, -2.0, 2.0)
        a_yaw   = np.clip(a_yaw, -2.0, 2.0)

        # ---------------------------------------------------------
        # 5. 合并为完整的 6D 动作输出
        # ---------------------------------------------------------
        action_2d = np.clip(action_2d, -self.u_max, self.u_max)
        
        # 返回格式: [a_x, a_y, a_z, a_roll, a_pitch, a_yaw]
        base_action = np.array([
            action_2d[0], action_2d[1], az, 
            a_roll, a_pitch, a_yaw
        ], dtype=np.float32)

        return base_action