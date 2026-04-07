import numpy as np
import casadi as ca

class NMPCController4D:
    def __init__(self, dt=0.1, N=15, L=0.6, u_max_xy=0.5, u_max_z=0.5, u_max_yaw=2.0):
        # 注意：这里的 dt 改为 0.1，因为 env 的控制频率为 10Hz
        self.dt = dt
        self.N = N
        self.L = L
        self.g = 9.81
        self.nx = 12
        self.nu = 4 
        
        # 状态变量: mocap 和 payload 的位置与速度
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)
        
        damping = 0.01  # 负载空气阻尼
        
        # 解析状态向量
        p_x, p_y, p_z, p_yaw = x[0], x[1], x[2], x[3]
        v_px, v_py, v_pz, v_pyaw = x[4], x[5], x[6], x[7]
        q_x, q_y = x[8], x[9]
        v_qx, v_qy = x[10], x[11]
        
        # 解析控制输入 (加速度)
        ax, ay, az, ayaw = u[0], u[1], u[2], u[3]
        
        # ---------------------------------------------------------
        # 【核心物理】3D 小角近似方程：有效重力加速度受到 Z 轴加速度的耦合影响
        # omega_sq_eff = (g + az) / L
        # ---------------------------------------------------------
        omega_sq_eff = (self.g + az) / self.L
        
        dx = ca.vertcat(
            v_px, v_py, v_pz, v_pyaw,   # mocap 速度
            ax, ay, az, ayaw,           # mocap 加速度 (控制量)
            v_qx, v_qy,                 # 负载 速度
            -omega_sq_eff * (q_x - p_x) - damping * v_qx,  # 负载 X 轴小角近似动力学
            -omega_sq_eff * (q_y - p_y) - damping * v_qy   # 负载 Y 轴小角近似动力学
        )
        
        f_dyn = ca.Function('f', [x, u], [dx])
        X = ca.SX.sym('X', self.nx, self.N + 1)
        U = ca.SX.sym('U', self.nu, self.N)
        
        cost = 0
        constraints = []
        
        # ---------------------------------------------------------
        # 继承并适配原版的超参
        # ---------------------------------------------------------
        Q_pos   = np.array([10.0, 10.0, 10.0, 5.0]) # 目标位置惩罚 [x, y, z, yaw]
        Q_swing = np.array([50.0, 50.0])            # 防摆惩罚 (保持 q_x 贴近 p_x)
        Q_vel   = 2.0                               # 速度惩罚
        R_acc   = np.array([0.2, 0.2, 0.2, 0.1])    # 控制输出惩罚
        
        P_ref  = ca.SX.sym('P_ref', 4)  # 外部输入的目标 [target_x, target_y, target_z, target_yaw]
        X_init = ca.SX.sym('X_init', self.nx)
        
        constraints.append(X[:, 0] - X_init)
        
        for k in range(self.N):
            # RK4 积分器
            k1 = f_dyn(X[:, k], U[:, k])
            k2 = f_dyn(X[:, k] + dt/2 * k1, U[:, k])
            k3 = f_dyn(X[:, k] + dt/2 * k2, U[:, k])
            k4 = f_dyn(X[:, k] + dt * k3, U[:, k])
            x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            constraints.append(X[:, k+1] - x_next)
            
            # Stage cost (阶段代价)
            cost += Q_pos[0] * (X[8, k] - P_ref[0])**2 + Q_pos[1] * (X[9, k] - P_ref[1])**2 # 负载 XY 跟踪
            cost += Q_pos[2] * (X[2, k] - P_ref[2])**2                                      # 动捕点 Z 轴跟踪
            cost += Q_pos[3] * (X[3, k] - P_ref[3])**2                                      # 动捕点 Yaw 跟踪
            
            cost += Q_swing[0] * (X[8, k] - X[0, k])**2 + Q_swing[1] * (X[9, k] - X[1, k])**2 # 最小化摆角
            
            cost += Q_vel * (X[4, k]**2 + X[5, k]**2 + X[6, k]**2 + X[7, k]**2)             # 抑制过度提速
            
            cost += R_acc[0]*U[0, k]**2 + R_acc[1]*U[1, k]**2 + R_acc[2]*U[2, k]**2 + R_acc[3]*U[3, k]**2
            
        # Terminal cost (终端代价，增加权重保证稳定收敛)
        cost += 10 * (Q_pos[0] * (X[8, self.N] - P_ref[0])**2 + Q_pos[1] * (X[9, self.N] - P_ref[1])**2)
        cost += 10 * (Q_pos[2] * (X[2, self.N] - P_ref[2])**2 + Q_pos[3] * (X[3, self.N] - P_ref[3])**2)
        
        nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
               'f': cost,
               'g': ca.vertcat(*constraints),
               'p': ca.vertcat(X_init, P_ref)}
        
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 30, # 4D解算稍微增加上限
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-1,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # 边界与约束设定
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

    def compute_action_from_obs(self, obs, target_4d):
        """
        根据最新 env 的观测布局解包，并进行 NMPC 求解
        obs: env 返回的观测值 (18 + 3*n_obs 维)
        target_4d: 目标 [x, y, z, yaw]
        返回 6D action [ax, ay, az, a_roll, a_pitch, a_yaw]
        """
        # 1. 对齐新版 mujoco_env_new.py 的索引
        mocap_x = obs[0]; mocap_y = obs[1]
        mocap_vx = obs[2]; mocap_vy = obs[3]
        payload_x = obs[4]; payload_y = obs[5]
        payload_vx = obs[6]; payload_vy = obs[7]
        
        mocap_z = obs[-8]; mocap_vz = obs[-7]
        mocap_yaw = obs[-4]; mocap_yaw_vel = obs[-3]
        
        # 2. 组装 12D 内部状态
        state_12d = np.array([
            mocap_x, mocap_y, mocap_z, mocap_yaw,
            mocap_vx, mocap_vy, mocap_vz, mocap_yaw_vel,
            payload_x, payload_y,
            payload_vx, payload_vy
        ])
        
        p_val = np.concatenate([state_12d, target_4d])
        x0_guess = np.zeros(self.lbx.shape)
        if self.last_sol is not None:
             x0_guess = self.last_sol
        
        try:
            sol = self.solver(x0=x0_guess, p=p_val, lbg=self.lbg, ubg=self.ubg, lbx=self.lbx, ubx=self.ubx)
            self.last_sol = sol['x']
            u_start_idx = self.nx * (self.N + 1)
            u_opt = np.array(sol['x'][u_start_idx : u_start_idx + self.nu]).flatten()
        except:
            self.last_sol = None
            u_opt = np.zeros(self.nu)
            
        # 3. 输出包装为 6D action，环境的 roll 和 pitch 置为 0 (完全交由 MPC 管理)
        action_6d = np.zeros(6)
        action_6d[0] = u_opt[0]  # ax
        action_6d[1] = u_opt[1]  # ay
        action_6d[2] = u_opt[2]  # az
        action_6d[5] = u_opt[3]  # a_yaw
        
        return action_6d

# ---------------------------------------------------------------------------
# 顶层封装：自动适配环境航点的 Tracker
# ---------------------------------------------------------------------------
class NMPCTrajectoryTracker:
    def __init__(self, dt=0.1, N=15, L=0.6):
        # 实例化底层的 4D MPC
        self.mpc = NMPCController4D(dt=dt, N=N, L=L)
        
    def compute_action(self, obs, target_z=0.7, target_yaw=0.0):
        """
        极其纯净的接口：由于新的 env 内部自动管理了 A* 航点切换，
        我们只需要从 obs 中提取 rel_tx 和 rel_ty，就能算出当前的局部追踪目标！
        """
        # 从 obs 解析相对距离 (对应环境中的 obs[8], obs[9])
        payload_x = obs[4]
        payload_y = obs[5]
        rel_tx = obs[8]
        rel_ty = obs[9]
        
        # 计算当前目标点
        target_x = payload_x + rel_tx
        target_y = payload_y + rel_ty
        
        target_4d = np.array([target_x, target_y, target_z, target_yaw])
        
        # 交给 4D MPC 求解
        return self.mpc.compute_action_from_obs(obs, target_4d)