# ==============================================================================
# controller.py — NMPC 专家控制器（下降段专项优化 v3）
#
# ══════════════════════════════════════════════════════════════════════════════
# 下降段晃动根因分析与修复
# ══════════════════════════════════════════════════════════════════════════════
#
# 症状：上升段和平移段稳定，进入下降段后 payload 严重晃动
#
# 根因 1：NMPC 航点推进时清空 last_sol（冷启动）
#   下降段有 15 个航点，间距仅 8.7mm，每推进 1 个航点就冷启动 NMPC，
#   导致 NMPC 解跳变，EE 加速度不连续 → 激发 payload 摆动。
#   修复：下降段航点推进时保留 NMPC 热启动。
#
# 根因 2：NMPC Q_pos_xy 远大于 Q_swing → 正反馈摆动
#   下降时 payload 自然微摆，NMPC 以高权重 Q_pos(=800) 追踪 payload XY，
#   激进移动 EE → 激发更大摆动 → 正反馈循环。
#   修复：下降段绕过 NMPC 的 XY 输出，改用直接位置控制。
#
# 根因 3：积分器在下降段累积 NMPC XY 加速度
#   NMPC 输出 XY 加速度，积分器持续累积放大 → EE 偏移 → 晃动。
#   修复：下降段不用积分器做 XY，直接设 EE XY = target 上方 + PD 微调。
#
# 根因 4：软锚定 alpha 太小 (0.1)
#   下降段每步仅移 ~0.01mm，积分器误差相对运动量巨大。
#   修复：下降段 Z 锚定 0.7，XY 直接位置控制不需要积分器。
#
# 整体策略：下降段 EE XY 直接锁定在目标正上方，只做 payload PD 微调。
#   EE Z 仍用 NMPC 的 Z 输出控制下降速度。
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import casadi as ca
import mujoco
import math

# obs 索引常量
# [V3-FIX] 改为正向索引，不受obs尾部新增维度影响
# obs布局 (n_obstacles=3, tl=9):
#   [0:10]  ee_x, ee_y, ee_vx, ee_vy, pl_x, pl_y, pl_vx, pl_vy, rel_tx, rel_ty
#   [10:19] obstacle_data (3*3)
#   [19:31] ee_z, ee_vz, pl_z, pl_vz, ee_roll, ee_roll_v, ee_pitch, ee_pitch_v,
#           ee_yaw, ee_yaw_v, payload_tilt, payload_yaw
#   [31:38] joint_q[0:7]
#   [38:45] joint_dq[0:7]
#   [45:54] phase_encode(3), progress(1), z_error(1), rebar_errors(4)
OBS_EE_X, OBS_EE_Y   = 0, 1
OBS_EE_VX, OBS_EE_VY = 2, 3
OBS_PL_X, OBS_PL_Y   = 4, 5
OBS_PL_VX, OBS_PL_VY = 6, 7
OBS_EE_Z       = 19
OBS_EE_VZ      = 20
OBS_PL_Z       = 21
OBS_PL_VZ      = 22
OBS_EE_YAW     = 27
OBS_EE_YAW_V   = 28


# ==============================================================================
# NMPCController4D
# ==============================================================================

class NMPCController4D:
    def __init__(self, dt=0.1, N=25,
                 L=0.445,
                 u_max_xy=0.8, u_max_z=2.0, u_max_yaw=2.0):
        self.dt  = dt
        self.N   = N
        self.L   = L
        self.g   = 9.81
        self.nx  = 12
        self.nu  = 4
        self.last_az = 0.0

        x       = ca.SX.sym('x', self.nx)
        u       = ca.SX.sym('u', self.nu)
        az_prev = ca.SX.sym('az_prev')

        damping = 0.08
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
            -omega_sq_eff * (q_y - p_y) - damping * v_qy,
        )
        f_dyn = ca.Function('f', [x, u, az_prev], [dx])

        X      = ca.SX.sym('X', self.nx, self.N + 1)
        U      = ca.SX.sym('U', self.nu, self.N)
        P_ref  = ca.SX.sym('P_ref', 4)
        X_init = ca.SX.sym('X_init', self.nx)
        Az_lin = ca.SX.sym('Az_lin')

        cost = 0; constraints = []

        Q_pos       = np.array([500.0, 500.0, 40.0, 1.0])
        Q_swing     = np.array([500.0, 500.0])
        Q_swing_vel = 200.0
        Q_vel       = 15.0
        R_acc       = np.array([0.03, 0.03, 0.15, 0.3])
        R_jerk      = 0.15

        constraints.append(X[:, 0] - X_init)
        mocap_target_z = P_ref[2] + self.L

        for k in range(self.N):
            k1 = f_dyn(X[:, k],              U[:, k], Az_lin)
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
            cost += Q_swing_vel * (X[10, k] - X[4, k])**2
            cost += Q_swing_vel * (X[11, k] - X[5, k])**2
            cost += Q_vel * (X[4,k]**2 + X[5,k]**2 + X[6,k]**2 + X[7,k]**2)
            cost += (R_acc[0]*U[0,k]**2 + R_acc[1]*U[1,k]**2
                   + R_acc[2]*U[2,k]**2 + R_acc[3]*U[3,k]**2)
            if k > 0:
                for j in range(self.nu):
                    cost += R_jerk * (U[j, k] - U[j, k-1])**2

        Qf = 25.0
        cost += Qf * (Q_pos[0]*(X[8, self.N]-P_ref[0])**2
                    + Q_pos[1]*(X[9, self.N]-P_ref[1])**2)
        cost += Qf * (Q_pos[2]*(X[2, self.N]-mocap_target_z)**2
                    + Q_pos[3]*(X[3, self.N]-P_ref[3])**2)
        cost += Qf * (Q_swing[0]*(X[8, self.N]-X[0, self.N])**2
                    + Q_swing[1]*(X[9, self.N]-X[1, self.N])**2)
        cost += Qf * Q_swing_vel * (X[10, self.N]-X[4, self.N])**2
        cost += Qf * Q_swing_vel * (X[11, self.N]-X[5, self.N])**2
        cost += Qf * Q_vel * (X[4,self.N]**2 + X[5,self.N]**2
                             + X[6,self.N]**2 + X[7,self.N]**2)

        nlp = {
            'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            'f': cost, 'g': ca.vertcat(*constraints),
            'p': ca.vertcat(X_init, P_ref, Az_lin),
        }
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.max_iter': 80, 'ipopt.tol': 5e-3,
            'ipopt.acceptable_tol': 5e-2,
            'ipopt.warm_start_init_point': 'yes',
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = self.nx * (self.N + 1) + self.nu * self.N
        self.lbx = -ca.inf * np.ones(n_vars)
        self.ubx =  ca.inf * np.ones(n_vars)
        u_start = self.nx * (self.N + 1)
        for i in range(self.N):
            idx = u_start + i * self.nu
            self.lbx[idx:idx+self.nu] = [-u_max_xy, -u_max_xy, -u_max_z, -u_max_yaw]
            self.ubx[idx:idx+self.nu] = [ u_max_xy,  u_max_xy,  u_max_z,  u_max_yaw]

        self.lbg = np.zeros(self.nx * (self.N + 1))
        self.ubg = np.zeros(self.nx * (self.N + 1))
        self.last_sol = None

    def get_action(self, state_12d, target_4d):
        p_val    = np.concatenate([state_12d, target_4d, [self.last_az]])
        x0_guess = self.last_sol if self.last_sol is not None \
                   else np.zeros(self.lbx.shape)
        try:
            sol = self.solver(x0=x0_guess, p=p_val,
                              lbg=self.lbg, ubg=self.ubg,
                              lbx=self.lbx, ubx=self.ubx)
            self.last_sol = sol['x']
            u_start = self.nx * (self.N + 1)
            u_opt   = np.array(sol['x'][u_start: u_start+self.nu]).flatten()
            self.last_az = float(u_opt[2])
        except Exception:
            self.last_sol = None
            u_opt = np.zeros(self.nu)
            self.last_az = 0.0
        return u_opt


# ==============================================================================
# NMPCTrajectoryTracker
# ==============================================================================

class NMPCTrajectoryTracker:
    def __init__(self, dt=0.1, N=25, L=0.445,
                 arrival_threshold_xy=0.05, arrival_threshold_z=0.05,
                 z_cruise=0.2):
        self.mpc = NMPCController4D(dt=dt, N=N, L=L)
        self.path = None
        self.current_idx = 0
        self.arrival_threshold_xy = arrival_threshold_xy
        self.arrival_threshold_z  = arrival_threshold_z
        self.estimated_L = L
        self.z_cruise = z_cruise
        self._is_descending = False

    def set_path(self, path):
        if path is None or len(path) == 0:
            self.path = None; self.current_idx = 0; return
        self.path = np.array(path, dtype=np.float64)
        self.current_idx = 0
        self._is_descending = False
        self.mpc.last_sol = None; self.mpc.last_az = 0.0

    def compute_ee_acceleration(self, obs, target_yaw=0.0):
        if self.path is None:
            return np.zeros(4)

        ee_x  = float(obs[OBS_EE_X]);  ee_y  = float(obs[OBS_EE_Y])
        ee_vx = float(obs[OBS_EE_VX]); ee_vy = float(obs[OBS_EE_VY])
        pl_x  = float(obs[OBS_PL_X]);  pl_y  = float(obs[OBS_PL_Y])
        pl_vx = float(obs[OBS_PL_VX]); pl_vy = float(obs[OBS_PL_VY])
        ee_z  = float(obs[OBS_EE_Z]);  ee_vz = float(obs[OBS_EE_VZ])
        pl_z  = float(obs[OBS_PL_Z])
        ee_yaw   = float(obs[OBS_EE_YAW])
        ee_yaw_v = float(obs[OBS_EE_YAW_V])

        actual_L = ee_z - pl_z
        if 0.1 < actual_L < 1.0:
            self.estimated_L = 0.9 * self.estimated_L + 0.1 * actual_L

        curr_pl_xy = np.array([pl_x, pl_y])
        wp  = self.path[self.current_idx]
        wp3 = np.array([wp[0], wp[1], wp[2] if len(wp) >= 3 else 0.3])

        dist_xy = np.linalg.norm(curr_pl_xy - wp3[:2])
        dist_z  = abs(pl_z - wp3[2])

        is_current_descent_wp = (wp3[2] < self.z_cruise - 0.01)

        if is_current_descent_wp:
            xy_thresh_desc = 0.025
            z_thresh_desc  = 0.010
            advance = (dist_xy < xy_thresh_desc and
                       dist_z  < z_thresh_desc and
                       self.current_idx < len(self.path) - 1)
        else:
            advance = (dist_xy < self.arrival_threshold_xy and
                       dist_z  < self.arrival_threshold_z and
                       self.current_idx < len(self.path) - 1)

        if advance:
            self.current_idx += 1
            wp  = self.path[self.current_idx]
            wp3 = np.array([wp[0], wp[1], wp[2] if len(wp) >= 3 else 0.3])
            # 下降段不清空 NMPC 解（保留热启动，避免解跳变）
            if not is_current_descent_wp:
                self.mpc.last_sol = None
                self.mpc.last_az = 0.0

        # 下降检测
        if not self._is_descending:
            in_cruise_height = (pl_z > self.z_cruise - 0.05)
            target_is_desc_wp = (wp3[2] < self.z_cruise - 0.01)
            if in_cruise_height and target_is_desc_wp:
                self._is_descending = True
        else:
            if pl_z > self.z_cruise + 0.03:
                self._is_descending = False

        # 航点前瞻（仅巡航段）
        ref_xy = wp3[:2].copy()
        ref_z  = wp3[2]
        if (not self._is_descending
                and self.current_idx < len(self.path) - 1
                and dist_xy < 0.10):
            next_wp = self.path[self.current_idx + 1]
            next_wp3 = np.array([next_wp[0], next_wp[1],
                                 next_wp[2] if len(next_wp) >= 3 else 0.3])
            blend = max(0.0, 1.0 - dist_xy / 0.10) * 0.4
            ref_xy = (1 - blend) * ref_xy + blend * next_wp3[:2]
            ref_z  = (1 - blend) * ref_z  + blend * next_wp3[2]

        state_12d = np.array([
            ee_x, ee_y, ee_z, ee_yaw,
            ee_vx, ee_vy, ee_vz, ee_yaw_v,
            pl_x, pl_y, pl_vx, pl_vy,
        ], dtype=np.float64)

        compensated_z = ref_z + (self.estimated_L - self.mpc.L)
        P_ref = np.array([ref_xy[0], ref_xy[1], compensated_z, target_yaw],
                         dtype=np.float64)
        return self.mpc.get_action(state_12d, P_ref)


# ==============================================================================
# NativeIKSolver
# ==============================================================================

class NativeIKSolver:
    def __init__(self, mj_model, mj_data):
        self.model  = mj_model
        self.data   = mj_data
        self.scratch = mujoco.MjData(mj_model)
        self.target_frame = "attachment_site"
        self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                          self.target_frame)
        self.is_site = True
        if self.obj_id == -1:
            self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                              self.target_frame)
            self.is_site = False
        if self.obj_id == -1:
            raise ValueError(f"找不到 '{self.target_frame}'")
        self.damping        = 5e-3
        self.nullspace_gain = 0.05
        self.w_pos          = 1.0
        self.w_rot_base     = 0.7
        self.max_iters      = 25
        self.dq_clip        = 0.2
        self._update_limits()

    def _update_limits(self):
        self.q_min       = self.model.jnt_range[:7, 0].copy()
        self.q_max       = self.model.jnt_range[:7, 1].copy()
        self.jnt_limited = self.model.jnt_limited[:7].copy()
        self.q_margin    = 0.15 * (self.q_max - self.q_min)

    def update_model(self, mj_model, mj_data):
        self.model   = mj_model
        self.data    = mj_data
        self.scratch = mujoco.MjData(mj_model)
        self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                          self.target_frame)
        self.is_site = True
        if self.obj_id == -1:
            self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                              self.target_frame)
            self.is_site = False
        self._update_limits()

    def solve_4d(self, current_q, target_x, target_y, target_z, target_yaw):
        roll, pitch, yaw = np.pi, 0.0, target_yaw
        cx, sx = np.cos(roll),  np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw),   np.sin(yaw)
        R_x = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        R_y = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        R_z = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        Rmat = R_z @ R_y @ R_x
        target_pos  = np.array([target_x, target_y, target_z])
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, Rmat.flatten())
        self.scratch.qpos[:] = self.data.qpos[:]
        self.scratch.qvel[:] = self.data.qvel[:]
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        q_guess = current_q.copy()
        for _ in range(self.max_iters):
            self.scratch.qpos[:7] = q_guess
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)
            if self.is_site:
                cp = self.scratch.site_xpos[self.obj_id].copy()
                cm = self.scratch.site_xmat[self.obj_id].reshape(3,3).copy()
            else:
                cp = self.scratch.xpos[self.obj_id].copy()
                cm = self.scratch.xmat[self.obj_id].reshape(3,3).copy()
            cq = np.zeros(4); mujoco.mju_mat2Quat(cq, cm.flatten())
            pe = target_pos - cp
            re = np.zeros(3); nq = np.zeros(4); eq = np.zeros(4)
            mujoco.mju_negQuat(nq, cq)
            mujoco.mju_mulQuat(eq, target_quat, nq)
            if eq[0] < 0: eq = -eq
            mujoco.mju_quat2Vel(re, eq, 1.0)
            pn = np.linalg.norm(pe); rn = np.linalg.norm(re)
            if pn < 3e-4 and rn < 3e-3: break
            wr = self.w_rot_base * (0.08 / (0.08 + pn))
            if pn > 0.08: pe = (pe/pn)*0.08
            if rn > 0.15: re = (re/rn)*0.15
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            if self.is_site:
                mujoco.mj_jacSite(self.model, self.scratch, jacp, jacr, self.obj_id)
            else:
                mujoco.mj_jacBody(self.model, self.scratch, jacp, jacr, self.obj_id)
            J = np.vstack([jacp, jacr])[:, :7]
            J_w   = J.copy(); J_w[:3] *= self.w_pos; J_w[3:] *= wr
            err_w = np.concatenate([pe*self.w_pos, re*wr])
            JJT_w = J_w @ J_w.T
            diag  = (self.damping**2) * np.eye(6)
            dq    = J_w.T @ np.linalg.solve(JJT_w + diag, err_w)
            grad = np.zeros(7)
            for j in range(7):
                if self.jnt_limited[j]:
                    if q_guess[j] > self.q_max[j] - self.q_margin[j]:
                        grad[j] = (self.q_max[j]-self.q_margin[j]) - q_guess[j]
                    elif q_guess[j] < self.q_min[j] + self.q_margin[j]:
                        grad[j] = (self.q_min[j]+self.q_margin[j]) - q_guess[j]
            if np.any(grad != 0):
                J_inv = J.T @ np.linalg.solve(J@J.T+diag, np.eye(6))
                dq   += (np.eye(7) - J_inv@J) @ (self.nullspace_gain * grad)
            dq = np.clip(dq, -self.dq_clip, self.dq_clip)
            q_guess += dq
            for j in range(7):
                if self.jnt_limited[j]:
                    q_guess[j] = np.clip(q_guess[j], self.q_min[j], self.q_max[j])
        return q_guess


# ==============================================================================
# JointSpaceExpert — 下降段直接位置控制 (针对 4 根钢筋高精度插入优化)
# ==============================================================================

class JointSpaceExpert:
    """BC 标签生成器：巡航段用 NMPC，下降段用平滑速度闭环（无跳变，高精度）。"""

    def __init__(self, config: dict, ik_solver: NativeIKSolver):
        ctrl_cfg = config["controller"]
        plan_cfg = config.get("planning", {})
        self.z_cruise = plan_cfg.get("payload_z_cruise", 0.2)
        
        # 允许的最低插入高度
        self.insertion_z_limit = plan_cfg.get("insertion_z_limit", 0.02) 

        self.tracker = NMPCTrajectoryTracker(
            dt=ctrl_cfg["dt"],
            N=ctrl_cfg.get("N", 25),
            L=ctrl_cfg["L"],
            arrival_threshold_xy=ctrl_cfg["arrival_threshold_xy"],
            arrival_threshold_z=ctrl_cfg["arrival_threshold_z"],
            z_cruise=self.z_cruise,
        )
        self.ik_solver = ik_solver
        self.dt = ctrl_cfg["dt"]

        sp = config["space"]
        self.dq_max = np.array(sp.get("dq_max", [0.1]*7), dtype=np.float64)
        self.q_low  = np.array(sp["action_space_low"],  dtype=np.float64)
        self.q_high = np.array(sp["action_space_high"], dtype=np.float64)

        self._target_xy = np.array(config["task"]["default_target_xy"], dtype=np.float64)

        self._ee_pos     = np.zeros(3, np.float64)
        self._ee_vel     = np.zeros(3, np.float64)
        self._ee_yaw     = 0.0
        self._ee_yaw_vel = 0.0
        self._last_q     = None

        # 巡航段参数
        self._v_max_xy_normal  = 0.3
        self._v_max_z_normal   = 0.2
        self._anchor_alpha_normal  = 0.3

        # =========================================================
        # 下降段：3 项防摇控制 + 稳态误差消除 (终极版)
        # =========================================================
        self._descent_K_target = 1.0   # 宏观引力 (拉向终点)
        self._descent_Ki       = 0.3   # 宏观积分 (消除没对准的稳态误差!)
        self._descent_K_swing  = 2.5   # 微观虚拟重力 (保持在吊载正上方，防打转)
        self._descent_K_catch  = 0.5   # 微观主动阻尼 (顺势接住晃动，吸能)
        
        self._integral_xy          = np.zeros(2, np.float64) 
        
        self._v_max_xy_descent = 0.20  # XY 允许足够速度去追赶
        self._v_max_z_descent  = -0.02 
        self._v_max_yaw_descent= 0.2

    def reset(self, env_obs, init_q, env=None):
        if env is not None:
            self._ee_pos = env._get_ee_pos().astype(np.float64)
        else:
            self._ee_pos = np.array([
                float(env_obs[OBS_EE_X]),
                float(env_obs[OBS_EE_Y]),
                float(env_obs[OBS_EE_Z]),
            ], np.float64)
        self._ee_vel     = np.zeros(3, np.float64)
        self._ee_yaw     = float(env_obs[OBS_EE_YAW])
        self._ee_yaw_vel = 0.0
        self._last_q     = init_q.copy().astype(np.float64)
        self.tracker.mpc.last_sol = None
        self.tracker.mpc.last_az  = 0.0 
        real_pl_z = float(env_obs[OBS_PL_Z])
        self.tracker.estimated_L = float(self._ee_pos[2]) - real_pl_z

    def set_path(self, path):
        self.tracker.set_path(path)

    def compute_joint_target(self, env_obs, current_q, target_yaw=0.0):
        is_desc = self.tracker._is_descending

        action_4d = self.tracker.compute_ee_acceleration(env_obs, target_yaw)

        real_ee = np.array([float(env_obs[OBS_EE_X]),
                             float(env_obs[OBS_EE_Y]),
                             float(env_obs[OBS_EE_Z])], np.float64)
        real_vel = np.array([float(env_obs[OBS_EE_VX]),
                              float(env_obs[OBS_EE_VY]),
                              float(env_obs[OBS_EE_VZ])], np.float64)

        dt = self.dt

        if is_desc:
            # ==============================================================
            # 下降段：无静差工业防摇控制
            # ==============================================================
            ee_xy = self._ee_pos[:2]
            pl_x  = float(env_obs[OBS_PL_X])
            pl_y  = float(env_obs[OBS_PL_Y])
            pl_vx = float(env_obs[OBS_PL_VX])
            pl_vy = float(env_obs[OBS_PL_VY])

            pl_xy = np.array([pl_x, pl_y])
            pl_vel_xy = np.array([pl_vx, pl_vy])
            yaw_err   = target_yaw - self._ee_yaw
            
            # --- 宏观对准 (Targeting) ---
            pl_xy_err = self._target_xy - pl_xy
            self._integral_xy += pl_xy_err * dt
            self._integral_xy = np.clip(self._integral_xy, -0.05, 0.05) # 抗积分饱和
            
            # 1. 目标引力：拉动吊载走向终点 (带积分，保证最终误差为 0)
            v_target = self._descent_K_target * pl_xy_err + self._descent_Ki * self._integral_xy
            
            # --- 微观消摆 (Anti-Swing) ---
            # 2. 虚拟重力：拉动 EE 保持在吊载正上方 (打破画圈极限环)
            v_swing  = self._descent_K_swing * (pl_xy - ee_xy)
            # 3. 主动阻尼：顺着吊载速度移动进行吸能
            v_catch  = self._descent_K_catch * pl_vel_xy
            
            # 综合速度指令
            target_v_xy = v_target + v_swing + v_catch
            
            v_norm = np.linalg.norm(target_v_xy)
            if v_norm > self._v_max_xy_descent:
                target_v_xy *= self._v_max_xy_descent / v_norm
                
            # 直接下发位置积分，[核心修复] 坚决不再使用 alpha_desc 软锚定！
            # 让控制器闭环自己处理物理误差
            self._ee_vel[:2] = target_v_xy
            self._ee_pos[:2] += self._ee_vel[:2] * dt

            # 2. Yaw 轴对准
            target_v_yaw = 1.0 * yaw_err
            target_v_yaw = np.clip(target_v_yaw, -self._v_max_yaw_descent, self._v_max_yaw_descent)
            self._ee_yaw_vel = target_v_yaw
            self._ee_yaw += self._ee_yaw_vel * dt

            # 3. Z 轴门控下降
            xy_err_norm = np.linalg.norm(pl_xy_err)
            
            # 误差要求 < 1.5cm，速度要求 < 1.5cm/s
            is_aligned = (xy_err_norm < 0.015) and (abs(yaw_err) < 0.05) 
            is_stable  = (np.linalg.norm(pl_vel_xy) < 0.015) 

            if is_aligned and is_stable:
                target_v_z = self._v_max_z_descent
            else:
                target_v_z = 0.0

            # Z轴保留平滑滤波
            self._ee_vel[2] = 0.8 * self._ee_vel[2] + 0.2 * target_v_z
            self._ee_pos[2] += self._ee_vel[2] * dt

            # [核心修复2] 注意：这里彻底移除了对 self._ee_pos 的 real_ee 状态锚定。
            # 让内部生成的完美轨迹顺畅滑入 IK 求解器，积分器才能有效克服外部阻力！

        else:
            # ==============================================================
            # 巡航/上升段：NMPC 逻辑
            # ==============================================================
            a_xyz = action_4d[:3].astype(np.float64)
            a_yaw = float(action_4d[3])

            self._ee_pos += self._ee_vel * dt + 0.5 * a_xyz * dt**2
            self._ee_vel += a_xyz * dt
            self._ee_yaw     += self._ee_yaw_vel * dt + 0.5 * a_yaw * dt**2
            self._ee_yaw_vel += a_yaw * dt

            vxy = np.linalg.norm(self._ee_vel[:2])
            if vxy > self._v_max_xy_normal and vxy > 1e-8:
                self._ee_vel[:2] *= self._v_max_xy_normal / vxy
            self._ee_vel[2] = np.clip(self._ee_vel[2],
                                       -self._v_max_z_normal, self._v_max_z_normal)

            # 巡航段保留软锚定，防止 NMPC 轨迹偏离实际太远
            alpha = self._anchor_alpha_normal
            self._ee_pos = (1 - alpha) * self._ee_pos + alpha * real_ee
            vel_alpha = alpha * 0.5
            self._ee_vel = (1 - vel_alpha) * self._ee_vel + vel_alpha * real_vel
            

        # ==============================================================
        # 插入深度保护
        # ==============================================================
        if self._ee_pos[2] < self.insertion_z_limit:
            self._ee_pos[2] = self.insertion_z_limit
            self._ee_vel[2] = max(self._ee_vel[2], 0.0)

        # IK 求解
        q_start  = self._last_q if self._last_q is not None else current_q
        q_target = self.ik_solver.solve_4d(
            current_q=q_start.astype(np.float64),
            target_x=float(self._ee_pos[0]),
            target_y=float(self._ee_pos[1]),
            target_z=float(self._ee_pos[2]),
            target_yaw=float(self._ee_yaw),
        )
        if q_target is None or np.any(np.isnan(q_target)):
            q_target = current_q.copy().astype(np.float64)

        q_target = np.clip(q_target, self.q_low, self.q_high)
        self._last_q = q_target.copy()
        return q_target.astype(np.float32)

    def compute_delta_q_target(self, env_obs, current_q, target_yaw=0.0):
        q_target = self.compute_joint_target(env_obs, current_q, target_yaw)
        delta_q  = q_target.astype(np.float64) - current_q.astype(np.float64)
        delta_q  = np.clip(delta_q, -self.dq_max, self.dq_max)
        return delta_q.astype(np.float32)