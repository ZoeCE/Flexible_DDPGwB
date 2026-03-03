# -*- coding: utf-8 -*-
"""
带障碍物避碰约束的缆索机器人 NMPC 控制器。

在原有 NMPC 基础上增加：路径上的圆形障碍物，约束负载 (q_x, q_y) 与每个障碍物
保持不小于 (障碍物半径 + 安全裕度) 的距离，避免吊装物体碰撞。
不修改原有 nmpc_controller.py，本文件独立使用。
"""

import numpy as np
import casadi as ca


class NMPCControllerObstacles:
    """
    带障碍物避碰的 NMPC 控制器。

    动力学与代价与 NMPCController 一致，额外增加不等式约束：
    对预测时域内每一步 k、每个障碍物 j，负载到障碍物中心距离 >= r_safe_j。
    障碍物为圆形，r_safe_j = 障碍物半径 + 安全裕度。
    """

    def __init__(self, dt=0.1, N=20, L=0.6, u_max=0.5, n_obstacles_max=5,
                 obstacle_margin=0.05):
        """
        构建带障碍物约束的 NMPC 求解器。

        Args:
            dt: 控制步长（秒）。
            N: 预测时域步数。
            L: 缆索长度（米）。
            u_max: 加速度上下界（m/s^2）。
            n_obstacles_max: 最大障碍物数量，NLP 构造时固定，实际可少传。
            obstacle_margin: 安全裕度（米），负载与障碍物表面最小间隔。
        """
        self.dt = dt
        self.N = N
        self.L = L
        self.g = 9.81
        self.u_max = u_max
        self.n_obstacles_max = n_obstacles_max
        self.obstacle_margin = obstacle_margin
        self.nx = 8
        self.nu = 2

        # 符号变量
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

        Q_pos = np.array([10.0, 80.0])
        Q_swing = np.array([20.0, 10.0])
        Q_vel = 1.0
        R_acc = 0.1

        # 参数：X_init(8), P_ref(2), 然后每个障碍物 (ox, oy, r_safe) 共 3*n_obstacles_max
        n_p = 8 + 2 + 3 * self.n_obstacles_max
        P_ref = ca.SX.sym('P_ref', 2)
        X_init = ca.SX.sym('X_init', self.nx)
        # 障碍物参数块 (ox, oy, r_safe) 每个障碍物
        obs_params = ca.SX.sym('obs', 3 * self.n_obstacles_max)
        p_sym = ca.vertcat(X_init, P_ref, obs_params)

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

        # 障碍物不等式约束：对每个时刻（含终端）、每个障碍物，d_sq >= r_safe^2
        # 负载位置 X[4,k], X[5,k]；障碍物 j: ox=obs_params[3*j], oy=obs_params[3*j+1], r_safe=obs_params[3*j+2]
        # 约束 g_ineq = (qx - ox)^2 + (qy - oy)^2 - r_safe^2 >= 0
        ineq_list = []
        for k in range(self.N + 1):
            qx_k = X[4, k]
            qy_k = X[5, k]
            for j in range(self.n_obstacles_max):
                ox = obs_params[3*j]
                oy = obs_params[3*j+1]
                r_safe = obs_params[3*j+2]
                d_sq = (qx_k - ox)**2 + (qy_k - oy)**2
                ineq_list.append(d_sq - r_safe**2)

        g_eq = ca.vertcat(*eq_constraints)
        g_ineq = ca.vertcat(*ineq_list)
        n_eq = g_eq.size1()
        n_ineq = g_ineq.size1()
        g_all = ca.vertcat(g_eq, g_ineq)

        nlp = {
            'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            'f': cost,
            'g': g_all,
            'p': p_sym
        }
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 50,
            'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-1,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = self.nx * (self.N + 1) + self.nu * self.N
        self.lbx = -ca.inf * np.ones(n_vars)
        self.ubx = ca.inf * np.ones(n_vars)
        u_start_idx = self.nx * (self.N + 1)
        self.lbx[u_start_idx:] = -self.u_max
        self.ubx[u_start_idx:] = self.u_max

        self.lbg = np.concatenate([np.zeros(n_eq), np.zeros(n_ineq)])
        self.ubg = np.concatenate([np.zeros(n_eq), np.full(n_ineq, np.inf)])

        self.n_eq = n_eq
        self.n_ineq = n_ineq
        self.last_sol = None

    def get_action(self, state, target_pos, obstacles):
        """
        根据当前状态、目标位置和障碍物列表求解 NMPC，返回第一步加速度。

        Args:
            state: 长度 8 的数组 [p_x, p_y, v_px, v_py, q_x, q_y, v_qx, v_qy]。
            target_pos: 长度 2 [target_x, target_y]。
            obstacles: list of (x, y, radius)，每个障碍物圆心 (x,y) 及半径（米）。
                       不足 n_obstacles_max 时用 (0,0,0) 填充，r_safe=0 表示不约束。

        Returns:
            长度 2 的数组 [ax, ay]；求解失败返回 [0, 0]。
        """
        # 构建参数：10 + 3*n_obstacles_max
        obs_flat = []
        for j in range(self.n_obstacles_max):
            if j < len(obstacles):
                ox, oy, r = obstacles[j]
                r_safe = float(r) + self.obstacle_margin
            else:
                ox, oy, r_safe = 0.0, 0.0, 0.0
            obs_flat.extend([ox, oy, r_safe])
        p_val = np.concatenate([state, target_pos, np.array(obs_flat)])

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
