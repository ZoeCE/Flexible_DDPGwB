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
OBS_TILT       = 29   # payload tilt (sqrt(roll²+pitch²))
OBS_YAW        = 30   # payload yaw


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

        # ── 航点推进（仅单步推进，不跳跃） ──
        if is_current_descent_wp:
            advance = (dist_xy < 0.025 and dist_z < 0.010 and
                       self.current_idx < len(self.path) - 1)
        else:
            advance = (dist_xy < self.arrival_threshold_xy and
                       dist_z  < self.arrival_threshold_z and
                       self.current_idx < len(self.path) - 1)

        if advance:
            self.current_idx += 1
            wp  = self.path[self.current_idx]
            wp3 = np.array([wp[0], wp[1], wp[2] if len(wp) >= 3 else 0.3])
            # 保留 NMPC 热启动，不清空 last_sol

        # ── 下降检测 (单向锁定: 一旦进入下降模式就不再退出)
        # 原逻辑: pl_z > z_cruise+30mm 时退出下降 → 摆动时 payload 升高会反复触发 settling+积分清零
        if not self._is_descending:
            in_cruise_height   = (pl_z > self.z_cruise - 0.05)
            target_is_desc_wp  = (wp3[2] < self.z_cruise - 0.01)
            if in_cruise_height and target_is_desc_wp:
                self._is_descending = True
        # 不再有 else 退出逻辑: 进入下降后锁定, 避免摆动引起的模式切换

        # ══════════════════════════════════════════════════════════════
        # [FIX-A] 路径切线参考点：沿路径前方固定距离处取参考
        #
        # 原问题：单航点跟踪 + 简单前瞻混合，导致：
        #   1. NMPC 参考点跳跃：航点推进时参考从 wp[i] 跳到 wp[i+1]
        #   2. 短切路径：前瞻把参考拉向下一航点，payload 走直线抄近路
        #      撞到路径弯道处的障碍物
        #   3. 速度不均匀：密集航点区参考点近→NMPC加速小→慢
        #      稀疏航点区参考点远→NMPC加速大→快
        #
        # 修复：沿路径向前取固定弧长距离（cruise_ref_dist）处作为参考
        # 这样参考点始终在路径上，不会短切，且移动速度一致
        # ══════════════════════════════════════════════════════════════
        ref_xy = wp3[:2].copy()
        ref_z  = wp3[2]

        if not self._is_descending and self.current_idx < len(self.path) - 1:
            # 沿路径向前取固定弧长距离处的点作为参考
            # [OPT] 增大前瞻距离, 让 NMPC 有更长预览窗口, 减少弯道急转
            cruise_ref_dist = 0.10  # 0.06→0.10, 沿路径前方 10cm 处作为 NMPC 参考
            accum_dist = 0.0
            ref_idx = self.current_idx
            prev_pt = wp3[:2].copy()

            for k in range(self.current_idx + 1, len(self.path)):
                next_wp = self.path[k]
                next_pt = np.array([next_wp[0], next_wp[1]])
                next_z  = next_wp[2] if len(next_wp) >= 3 else 0.3

                # 如果下一个是下降航点，停止向前延伸
                if next_z < self.z_cruise - 0.01:
                    break

                seg_len = np.linalg.norm(next_pt - prev_pt)
                if accum_dist + seg_len >= cruise_ref_dist:
                    # 在这段上插值
                    remain = cruise_ref_dist - accum_dist
                    frac = remain / max(seg_len, 1e-6)
                    ref_xy = prev_pt + frac * (next_pt - prev_pt)
                    ref_z  = wp3[2]  # 巡航段 z 保持不变
                    ref_idx = k
                    break
                accum_dist += seg_len
                prev_pt = next_pt
                ref_idx = k
            else:
                # 路径剩余不足 cruise_ref_dist，用最后一个巡航航点
                last_cruise = self.path[ref_idx]
                ref_xy = np.array([last_cruise[0], last_cruise[1]])
                ref_z  = last_cruise[2] if len(last_cruise) >= 3 else 0.3

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
        # [OPT] 降低锚定强度减少位置拖拽, 提高速度一致性
        self._v_max_xy_normal  = 0.15       # 0.18→0.15, 更慢更稳
        self._v_max_z_normal   = 0.2
        self._anchor_alpha_normal  = 0.10   # 0.15→0.10, 减少积分器与真实位置的拖拽冲突

        # =========================================================
        # 下降段：精准插入控制器 (v9 — 最小修改原则)
        # ─────────────────────────────────────────────────────────
        # 原始 v2 框架已验证: 防摆好, 最终 dtf=5mm 差 1mm
        # v9 只做两处最小修改, 其余全部保持原始值:
        #
        # [修改1] Ki 0.3→0.6 (只翻倍, 不过激)
        #   原始 Ki=0.3: 积分力 = 0.3×5mm = 1.5mm/s (太弱, 被 v_swing 残差抵消)
        #   v9   Ki=0.6: 积分力 = 0.6×5mm = 3.0mm/s (足够推过 1mm 残差)
        #   不用更大的 Ki: Ki>1.0 会在大摆动时积分过快累积方向错误
        #
        # [修改2] 积分上限固定 0.006m (不用动态, 简单可控)
        #   原始: 0.05m (太大, 大摆动时饱和后方向错误)
        #   v9: 0.006m ≈ 6mm (只允许积分修正 6mm 以内的稳态误差)
        #   在 dtf>6mm 时积分上限=dtf, 积分随误差正比增长不会过冲
        #   在 dtf<6mm 时积分上限固定, 防止小区间过积分
        #
        # 其余参数全部保持原始 v2 值 (已验证稳定)
        # ─────────────────────────────────────────────────────────
        self._descent_K_target = 1.0   # 原始值
        self._descent_Ki       = 0.6   # 0.3→0.6: 仅翻倍
        self._descent_K_swing  = 2.5   # 原始值
        self._descent_K_catch  = 0.5   # 原始值

        self._integral_xy  = np.zeros(2, np.float64)
        self._integral_max = 0.006   # 固定 6mm 上限 (简单可控)

        # Z 软门控 — 从 config 读取, 保持原始默认值 (v14.2)
        # 训练时可通过 config["descent_rl"] 放宽门控, 让 RL 探索期间 z 也能下降
        # 测试时保持原始严格值 (15mm/5mm) 确保精度
        _descent_cfg = config.get("descent_rl", {})
        self._v_max_xy_descent = float(_descent_cfg.get("base_v_max_xy", 0.20))
        self._v_max_z_descent  = -abs(float(_descent_cfg.get("base_v_max_z", 0.020)))
        self._v_max_yaw_descent= float(_descent_cfg.get("base_v_max_yaw", 0.2))
        self._z_hard_gate      = float(_descent_cfg.get("z_hard_gate",      0.015))
        self._z_soft_gate_full = float(_descent_cfg.get("z_soft_gate_full", 0.005))
        self._z_gate_vel       = float(_descent_cfg.get("z_gate_vel",       0.015))
        self._z_gate_yaw       = float(_descent_cfg.get("z_gate_yaw",       0.05))
        self._z_trickle_enabled = bool(_descent_cfg.get("z_trickle_enabled", True))
        self._z_trickle_xy_gate = float(_descent_cfg.get("z_trickle_xy_gate", 0.080))
        self._z_min_speed_frac  = float(_descent_cfg.get("z_min_speed_frac", 0.25))
        self._z_vel_slowdown    = float(_descent_cfg.get("z_vel_slowdown", 0.70))
        self._z_yaw_slowdown    = float(_descent_cfg.get("z_yaw_slowdown", 0.70))
        self._z_near_slowdown_margin = float(_descent_cfg.get(
            "z_near_slowdown_margin", 0.040))

        # z 到位判定
        _rope_L    = config.get("controller", {}).get("L", 0.5)
        _target_pz = config.get("insertion", {}).get("target_payload_z", 0.10)
        self._target_payload_z = float(_target_pz)
        self._z_reached_thresh = float(_target_pz) + 0.015
        self._rope_L_config    = float(_rope_L)

        self._was_descending     = False
        self._descent_settle_counter = 0
        self._descent_settle_steps   = 8
        self._z_reached          = False
        self._final_hold_counter = 0
        self._contact_detected   = False   # 接触检测标志

    def reset(self, env_obs, init_q, env=None):
        if env is not None:
            self._ee_pos = env._get_ee_pos().astype(np.float64)
            # [FIX] 每次 reset 从 env 更新实际目标位置
            # _target_xy 原来只在 __init__ 从 config["task"]["default_target_xy"] 读取一次
            # 但每个 episode 的 env.target_pos 是随机化的, 必须每次同步
            if hasattr(env, 'target_pos'):
                self._target_xy = np.array(env.target_pos[:2], dtype=np.float64)
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
        self._was_descending = False
        self._descent_settle_counter = 0
        self._integral_xy[:] = 0.0
        self._z_reached = False
        self._final_hold_counter = 0
        self._contact_detected = False
        real_pl_z = float(env_obs[OBS_PL_Z])
        self.tracker.estimated_L = float(self._ee_pos[2]) - real_pl_z

    def set_path(self, path):
        self.tracker.set_path(path)

    def compute_joint_target(self, env_obs, current_q, target_yaw=0.0,
                             residual_acc=None):
        """
        计算 joint target. 与 test_phase 单独 expert 完全相同的积分器路径.

        [v12 新增] residual_acc 可选参数:
            形状 (3,) 的 EE 加速度残差, 加在 NMPC 输出 action_4d[:3] 上.
            注入点: 在 action_4d 输出之后, 在积分器之前.
            这样 NMPC 预测、积分器、速度限制、锚定、IK 全部与 test_phase 一致,
            残差 RL 真正变成 "在 expert 内部的小幅微调".
        """
        is_desc = self.tracker._is_descending

        action_4d = self.tracker.compute_ee_acceleration(env_obs, target_yaw)

        # [v12 关键修复] 在 NMPC 输出后, 积分器前, 注入 RL 残差
        # 这样: NMPC 在"真实 ee state"上规划 → 积分器用"NMPC + RL 残差"演化
        # → 与 test_phase 中 expert-only (residual=0) 路径完全一致.
        if residual_acc is not None:
            _res = np.asarray(residual_acc, dtype=np.float64)
            if _res.shape == (3,):
                action_4d = np.array([
                    float(action_4d[0]) + _res[0],
                    float(action_4d[1]) + _res[1],
                    float(action_4d[2]) + _res[2],
                    float(action_4d[3]),  # yaw 不加残差
                ], dtype=np.float64)

        real_ee = np.array([float(env_obs[OBS_EE_X]),
                             float(env_obs[OBS_EE_Y]),
                             float(env_obs[OBS_EE_Z])], np.float64)
        real_vel = np.array([float(env_obs[OBS_EE_VX]),
                              float(env_obs[OBS_EE_VY]),
                              float(env_obs[OBS_EE_VZ])], np.float64)

        dt = self.dt

        if is_desc:
            # ==============================================================
            # 下降段：精准插入控制器 v7 (回到原始框架)
            # ==============================================================

            if not self._was_descending:
                self._ee_pos[:] = real_ee
                self._ee_vel[:] = real_vel * 0.3
                self._integral_xy[:] = 0.0
                self._descent_settle_counter = 0
                self._z_reached = False
                self._final_hold_counter = 0
            self._was_descending = True
            self._descent_settle_counter += 1

            pl_x  = float(env_obs[OBS_PL_X])
            pl_y  = float(env_obs[OBS_PL_Y])
            pl_vx = float(env_obs[OBS_PL_VX])
            pl_vy = float(env_obs[OBS_PL_VY])
            pl_z  = float(env_obs[OBS_PL_Z])
            pl_tilt = float(env_obs[OBS_TILT])   # payload 姿态 (index 29)

            pl_xy     = np.array([pl_x, pl_y])
            pl_vel_xy = np.array([pl_vx, pl_vy])
            ee_xy     = self._ee_pos[:2].copy()
            yaw_err   = target_yaw - self._ee_yaw

            in_settling = (self._descent_settle_counter <= self._descent_settle_steps)

            # ── 接触检测 ──────────────────────────────────────────────────
            # 现象: z=140mm 时 tilt 从 0.011 突增到 0.021, 同时 dtf 从 3.8mm 跳到 8.5mm
            # 物理: payload 底面接触钢筋顶端, 约束力使质心侧移 + tilt 增大
            # 识别: z 接近目标 (< entry_z) 且 tilt 超过正常摆动阈值
            # 应对: 检测到接触时锁定 EE xy 位置, 不再追 payload 的接触后位移
            #       接触后 payload 被钢筋导向, EE 只需保持不动让绳索张力自然对准
            _entry_z  = 0.16    # 进入插入区域的 z 高度 (来自 config insertion.entry_z)
            _tilt_contact_thresh = 0.018  # 接触检测 tilt 阈值 (正常摆动 < 0.015)
            _in_contact = (pl_z < _entry_z and pl_tilt > _tilt_contact_thresh)
            if _in_contact and not self._z_reached:
                # 首次检测到接触: 锁定当前 EE xy 位置, 清零积分防止过冲
                # 不设 z_reached (z 还没到位), 但冻结 xy 控制
                if not getattr(self, '_contact_detected', False):
                    self._contact_detected = True
                    self._integral_xy[:] = 0.0   # 清零积分, 防止历史积分导致 EE 偏移
            elif not _in_contact:
                self._contact_detected = False

            # z 到位判定 (单向锁定)
            if pl_z <= self._z_reached_thresh:
                self._z_reached = True
            # z 到位后计数: 用于 v_swing 平滑淡出
            if self._z_reached:
                self._final_hold_counter += 1

            # ── XY 控制 ──────────────────────────────────────────────────
            pl_xy_err   = self._target_xy - pl_xy
            xy_err_norm = float(np.linalg.norm(pl_xy_err))

            # 积分: 抗饱和 — 上限 = min(当前误差, 最大值)
            # 防止历史积分在 dtf 已经很小时仍然推 EE 过冲
            if not in_settling and not getattr(self, '_contact_detected', False):
                dynamic_clip = min(xy_err_norm, self._integral_max)
                self._integral_xy += pl_xy_err * dt
                self._integral_xy = np.clip(self._integral_xy,
                                            -dynamic_clip, dynamic_clip)

            # 原始控制律: 目标引力 + 弹簧(EE追payload) + 速度阻尼 + 积分
            v_target = self._descent_K_target * pl_xy_err
            v_int    = self._descent_Ki * self._integral_xy
            v_swing  = self._descent_K_swing * (pl_xy - ee_xy)
            v_catch  = self._descent_K_catch * pl_vel_xy

            # 接触检测到时: 冻结 EE xy (不再追 payload 的接触后位移)
            # payload 被钢筋端约束后会自然沿钢筋导向, EE 保持不动让绳索张力自动对准
            if getattr(self, '_contact_detected', False):
                target_v_xy = np.zeros(2)
            elif in_settling:
                target_v_xy = np.zeros(2)
            else:
                # z 到位后: v_swing 线性淡出 (15步内从1.0→0.0)
                if self._z_reached:
                    _fadeout_steps = 15
                    _swing_alpha = max(0.0, 1.0 - self._final_hold_counter / _fadeout_steps)
                    v_swing = v_swing * _swing_alpha
                target_v_xy = v_target + v_int + v_swing + v_catch

            # 速度上限: z 到位后平滑收紧
            if self._z_reached:
                _fadeout_steps = 15
                _speed_alpha = max(0.4, 1.0 - self._final_hold_counter / _fadeout_steps * 0.6)
            else:
                _speed_alpha = 1.0
            v_max = self._v_max_xy_descent * _speed_alpha
            v_norm = float(np.linalg.norm(target_v_xy))
            if v_norm > v_max:
                target_v_xy *= v_max / v_norm

            self._ee_vel[:2] = target_v_xy
            self._ee_pos[:2] += self._ee_vel[:2] * dt

            # ── Yaw 对准 ─────────────────────────────────────────────────
            self._ee_yaw_vel = np.clip(yaw_err,
                                       -self._v_max_yaw_descent, self._v_max_yaw_descent)
            self._ee_yaw += self._ee_yaw_vel * dt

            # ── Z 软门控下降 ──────────────────────────────────────────────
            z_above_target = max(0.0, pl_z - self._target_payload_z)
            if self._z_reached or in_settling or z_above_target <= 0.0:
                target_v_z = 0.0
            else:
                pl_speed = float(np.linalg.norm(pl_vel_xy))
                if xy_err_norm >= self._z_hard_gate:
                    z_speed_frac = 0.0
                    if (self._z_trickle_enabled and
                            xy_err_norm < self._z_trickle_xy_gate):
                        span = max(self._z_trickle_xy_gate - self._z_hard_gate, 1e-6)
                        taper = 1.0 - (xy_err_norm - self._z_hard_gate) / span
                        z_speed_frac = self._z_min_speed_frac * np.clip(taper, 0.0, 1.0)
                elif xy_err_norm <= self._z_soft_gate_full:
                    z_speed_frac = 1.0
                else:
                    z_speed_frac = 1.0 - ((xy_err_norm - self._z_soft_gate_full) /
                                          (self._z_hard_gate - self._z_soft_gate_full))
                if pl_speed >= self._z_gate_vel:
                    z_speed_frac *= self._z_vel_slowdown
                if abs(yaw_err) >= self._z_gate_yaw:
                    z_speed_frac *= self._z_yaw_slowdown
                if z_above_target < self._z_near_slowdown_margin:
                    z_speed_frac *= max(0.25, z_above_target / max(
                        self._z_near_slowdown_margin, 1e-6))
                target_v_z = self._v_max_z_descent * z_speed_frac

            self._ee_vel[2] = 0.7 * self._ee_vel[2] + 0.3 * target_v_z
            self._ee_pos[2] += self._ee_vel[2] * dt
        else:
            # ==============================================================
            # 巡航/上升段：NMPC 逻辑
            # ==============================================================
            self._was_descending = False  # [FIX-C]

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

            # 巡航段软锚定
            # [OPT] 位置锚定用较低 alpha, 速度锚定独立且更柔和
            alpha = self._anchor_alpha_normal
            self._ee_pos = (1 - alpha) * self._ee_pos + alpha * real_ee
            vel_alpha = 0.05  # 速度锚定更弱, 避免速度突变
            self._ee_vel = (1 - vel_alpha) * self._ee_vel + vel_alpha * real_vel
            

        # ==============================================================
        # EE 高度下限保护
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

    def compute_delta_q_target(self, env_obs, current_q, target_yaw=0.0,
                               residual_acc=None):
        """计算 delta_q. [v12] 加 residual_acc 可选参数."""
        q_target = self.compute_joint_target(env_obs, current_q, target_yaw,
                                             residual_acc=residual_acc)
        delta_q  = q_target.astype(np.float64) - current_q.astype(np.float64)
        delta_q  = np.clip(delta_q, -self.dq_max, self.dq_max)
        return delta_q.astype(np.float32)
