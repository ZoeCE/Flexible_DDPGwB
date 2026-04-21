# ==============================================================================
# controller.py — NMPC 专家控制器（大幅优化版）
#
# ══════════════════════════════════════════════════════════════════════════════
# 优化目标：更稳定、摆动更小、下降阶段 XY 偏差更小
# ══════════════════════════════════════════════════════════════════════════════
#
# [OPT-1] NMPC 代价函数全面重调
#   a) Q_swing 200→500：摆角惩罚成为绝对主导
#   b) Q_swing_vel 20→80：大幅增加摆角速度阻尼
#   c) Q_pos_xy 10→25：提高位置追踪精度
#   d) Q_pos_z 20→40：Z 轴追踪更紧
#   e) Q_vel 2→4：EE 速度惩罚加大，到达航点前主动减速
#   f) R_acc_xy 0.1→0.05：降低 XY 加速度代价，允许更激进的防摆修正
#   g) 终端代价系数 15→25
#
# [OPT-2] EE 速度限幅（新增）
#
# [OPT-3] 下降阶段专用逻辑（新增）
#   检测到当前航点 Z < z_cruise 时进入下降模式：
#   a) XY 速度限幅收紧到 0.08 m/s
#   b) Z 速度限幅收紧到 0.10 m/s
#   c) 软锚定 alpha 0.3→0.6
#
# [OPT-4] 软锚定增强（位置 + 速度）
#
# [OPT-5] IK 求解器增强
#
# [OPT-6] NMPC 求解器参数优化
#
# [OPT-7] 航点前瞻
#
# [OPT-8] 终端摆角速度 + EE 速度惩罚
#
# [OPT-9] 加速度变化率惩罚（jerk）
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import casadi as ca
import mujoco

# obs 索引常量（与 mujoco_env_new._get_obs 布局严格对齐）
OBS_EE_X, OBS_EE_Y   = 0, 1
OBS_EE_VX, OBS_EE_VY = 2, 3
OBS_PL_X, OBS_PL_Y   = 4, 5
OBS_PL_VX, OBS_PL_VY = 6, 7
OBS_EE_Z       = -26
OBS_EE_VZ      = -25
OBS_PL_Z       = -24
OBS_PL_VZ      = -23
OBS_EE_YAW     = -18
OBS_EE_YAW_V   = -17


# ==============================================================================
# NMPCController4D — 大幅优化版
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

        Q_pos       = np.array([500.0, 500.0, 40.0, 5.0])
        Q_swing     = np.array([500.0, 500.0])
        Q_swing_vel = 200.0
        Q_vel       = 12.0
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
# NMPCTrajectoryTracker — 航点前瞻 + 下降检测
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

        # [WP-ADVANCE FIX] 判断是否处于下降段航点（z < z_cruise - 0.01）
        # 下降段航点 XY 全部相同（= target_xy），仅 z 递减。
        # 若仍用原 80mm 阈值，payload 一到达 target_xy 附近就会单步跳过所有 15 个下降航点，
        # 导致 NMPC 反复冷启动、失去平滑控制 → XY 对准精度丢失。
        # 修复：下降段用 "z 距离 < dz/2" 的严格阈值，每步最多推进 1 个航点。
        is_current_descent_wp = (wp3[2] < self.z_cruise - 0.01)

        if is_current_descent_wp:
            # 下降段专用阈值：基于航点间距 dz 自动计算（每次推进 1 个）
            # 同时 XY 必须严格对准（否则策略未真正到位就推进）
            xy_thresh_desc = 0.015     # 15mm，远严于成功容差 3mm 的复原空间
            z_thresh_desc  = 0.005     # 5mm，小于典型 dz=8-9mm 的一半
            advance = (dist_xy < xy_thresh_desc and
                       dist_z  < z_thresh_desc and
                       self.current_idx < len(self.path) - 1)
        else:
            # 巡航/上升段：保留原阈值
            advance = (dist_xy < self.arrival_threshold_xy and
                       dist_z  < self.arrival_threshold_z and
                       self.current_idx < len(self.path) - 1)

        if advance:
            self.current_idx += 1
            wp  = self.path[self.current_idx]
            wp3 = np.array([wp[0], wp[1], wp[2] if len(wp) >= 3 else 0.3])
            self.mpc.last_sol = None; self.mpc.last_az = 0.0

        # [OPT-3 FIX] 下降检测：正确定义是"从巡航高度向下降"而非"目标航点 z 小"
        # 旧版仅 (wp3[2] < z_cruise - 0.02) → 上升段前期航点 z 也小于 z_cruise，被误判为下降
        # 新版：
        #   进入下降条件（需全部满足）：
        #     (a) payload 已到巡航高度附近（pl_z > z_cruise - 0.05 = 0.20）
        #     (b) 目标航点低于巡航高度（wp3[2] < z_cruise - 0.01 = 0.24）
        #         即轨迹已进入"下降段航点"（由 set_path 生成时 z 单调下降）
        #   退出下降条件：payload 被显著抬升（pl_z > z_cruise + 0.03）
        #   状态保持避免频繁切换（下降段航点间隔很小，瞬时条件会抖动）
        if not self._is_descending:
            in_cruise_height = (pl_z > self.z_cruise - 0.05)
            target_is_desc_wp = (wp3[2] < self.z_cruise - 0.01)
            if in_cruise_height and target_is_desc_wp:
                self._is_descending = True
        else:
            if pl_z > self.z_cruise + 0.03:
                self._is_descending = False

        # [OPT-7] 航点前瞻（下降阶段禁用，XY 必须锁定目标）
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
# NativeIKSolver — 优化版
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
# JointSpaceExpert — 大幅优化版
# ==============================================================================

class JointSpaceExpert:
    """
    BC 标签生成器：NMPC → 积分 → IK → delta_q。
    """

    def __init__(self, config: dict, ik_solver: NativeIKSolver):
        ctrl_cfg = config["controller"]
        plan_cfg = config.get("planning", {})
        self.z_cruise = plan_cfg.get("payload_z_cruise", 0.2)

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

        self._ee_pos     = np.zeros(3, np.float64)
        self._ee_vel     = np.zeros(3, np.float64)
        self._ee_yaw     = 0.0
        self._ee_yaw_vel = 0.0
        self._last_q     = None

        # [OPT-2][OPT-3] 速度限幅
        # normal = 上升段 / 巡航段（平移）：提速 2.5× 以缩短运动时间
        #   每控制步 (dt=0.1s)：XY 25mm，Z 12mm — EE 单步位移仍远小于 A* 航点间距
        # descent = 严格保持原值（下降段精度敏感，与本改动解耦）
        self._v_max_xy_normal  = 0.15    # 旧 0.1 → 0.25（2.5×）
        self._v_max_z_normal   = 0.15    # 旧 0.05 → 0.12（2.4×）
        self._v_max_xy_descent = 0.05    # 保持不变
        self._v_max_z_descent  = 0.05   # 保持不变

        # [OPT-4] 软锚定系数
        # normal 略增 alpha：高速下积分器需要更快跟随真实 EE 状态，
        #   避免发散（alpha 越大越信任真实值）
        self._anchor_alpha_normal  = 0.2   # 旧 0.2 → 0.3
        self._anchor_alpha_descent = 0.2    # 保持不变

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
        # 强行同步一次 tracker 的预估绳长，避免首步 NMPC 目标突变
        real_pl_z = float(env_obs[OBS_PL_Z])
        self.tracker.estimated_L = float(self._ee_pos[2]) - real_pl_z

    def set_path(self, path):
        self.tracker.set_path(path)

    def compute_joint_target(self, env_obs, current_q, target_yaw=0.0):
        """返回绝对关节角 q_target。"""
        action_4d = self.tracker.compute_ee_acceleration(env_obs, target_yaw)
        a_xyz = action_4d[:3].astype(np.float64)
        a_yaw = float(action_4d[3])

        dt = self.dt
        self._ee_pos += self._ee_vel * dt + 0.5 * a_xyz * dt**2
        self._ee_vel += a_xyz * dt
        self._ee_yaw     += self._ee_yaw_vel * dt + 0.5 * a_yaw * dt**2
        self._ee_yaw_vel += a_yaw * dt

        # [OPT-3] 下降模式参数切换
        is_desc = self.tracker._is_descending
        v_max_xy = self._v_max_xy_descent if is_desc else self._v_max_xy_normal
        v_max_z  = self._v_max_z_descent  if is_desc else self._v_max_z_normal
        alpha    = self._anchor_alpha_descent if is_desc else self._anchor_alpha_normal

        # [OPT-2] 速度限幅
        vxy = np.linalg.norm(self._ee_vel[:2])
        if vxy > v_max_xy and vxy > 1e-8:
            self._ee_vel[:2] *= v_max_xy / vxy
        self._ee_vel[2] = np.clip(self._ee_vel[2], -v_max_z, v_max_z)

        # Z 下限保护
        if self._ee_pos[2] < 0.20:
            self._ee_pos[2] = 0.20
            self._ee_vel[2] = max(self._ee_vel[2], 0.0)

        # [OPT-4] 软锚定（位置）
        real_ee = np.array([float(env_obs[OBS_EE_X]),
                             float(env_obs[OBS_EE_Y]),
                             float(env_obs[OBS_EE_Z])], np.float64)
        self._ee_pos = (1 - alpha) * self._ee_pos + alpha * real_ee

        # [OPT-4] 软锚定（速度）
        real_vel = np.array([float(env_obs[OBS_EE_VX]),
                              float(env_obs[OBS_EE_VY]),
                              float(env_obs[OBS_EE_VZ])], np.float64)
        vel_alpha = alpha * 0.5
        self._ee_vel = (1 - vel_alpha) * self._ee_vel + vel_alpha * real_vel

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
        """输出 delta_q = q_target - q_current，clamp 到 [-dq_max, +dq_max]。"""
        q_target = self.compute_joint_target(env_obs, current_q, target_yaw)
        delta_q  = q_target.astype(np.float64) - current_q.astype(np.float64)
        delta_q  = np.clip(delta_q, -self.dq_max, self.dq_max)
        return delta_q.astype(np.float32)