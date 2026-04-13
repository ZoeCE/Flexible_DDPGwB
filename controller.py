# ==============================================================================
# controller.py — NMPC 专家控制器（BC 标签生成器）
#
# 架构职责（新版）：
#   此模块在新架构中承担"专家标签生成器"的角色，而非直接参与控制链。
#
# 数据流：
#   env.obs (物理状态)
#     → NMPCTrajectoryTracker.compute_action()  → 4D 末端加速度
#     → 内部积分 + NativeIKSolver.solve_4d()    → 7 关节角目标
#     → 作为 BC Loss 的监督信号（q_target_joints）
#
# 与旧版 nmpc_controller_new.py 的差异：
#   [CTRL-NEW-1] 新增 JointSpaceExpert 类
#     - 封装"NMPC → 积分 → IK"完整链路
#     - 输出 7D 关节角（直接作为 BC 目标）
#     - 保留完整的 NMPC 计算逻辑（NMPCController4D + NMPCTrajectoryTracker）
#
#   [CTRL-NEW-2] 内部维护虚拟 EE 位姿积分状态
#     - 与 env 中的 current_mocap_pos 独立，避免耦合
#     - 每次 reset() 时同步 env 初始状态
#
#   [CTRL-NEW-3] compute_joint_target() 接口
#     - 输入：env.obs（与 _get_obs() 对齐）+ 当前真实关节角
#     - 输出：7D 关节角目标（BC 监督信号）
#     - 若 IK 失败，返回当前关节角（保持原地不动，不产生不合理标签）
#
# 保留内容：
#   - NMPCController4D（完整 NMPC 求解器）
#   - NMPCTrajectoryTracker（航点追踪器）
#   - 所有修复日志（CTRL-BUG-1 ~ CTRL-BUG-7 和 CTRL-IMPROVE-1~2）
# ==============================================================================

# ==============================================================================
# controller.py — NMPC 专家控制器（BC 标签生成器）修复版
#
# ══════════════════════════════════════════════════════════════════════════════
# 修复清单（对照当前上传版本逐项排查）
# ══════════════════════════════════════════════════════════════════════════════
#
# [FIX-C1] IK 末端垂直约束错误（最根本的稳定性问题）
#   原版 solve_4d 中旋转矩阵：roll=pi, pitch=0, yaw=target_yaw
#   构造方式为 R_z @ R_y @ R_x（ZYX 外旋/XYZ 内旋），
#   但 KUKA iiwa14 末端朝下约定通常需要 roll=pi（绕 X 翻转），
#   实际上这等价于 "末端 Z 轴朝下"。问题在于：
#     a) 当 target_yaw 非零时，R_z 被放在最外层，旋转正确。
#        但传入 IK 的 target_quat 在 MuJoCo 约定中 (w,x,y,z) 顺序
#        而 mju_mat2Quat 输出也是 (w,x,y,z)。✓ 顺序一致，无问题。
#     b) 真正的问题：旋转权重 w_rot_base=0.15 且带动态衰减，
#        在平移误差较大时 w_rot≈0，IK 几乎不追踪旋转目标。
#        这导致末端姿态漂移，不能保持垂直。
#   修复：
#     - 将 w_rot_base 从 0.15 提升到 0.5。
#     - 调整动态衰减公式，确保位置误差收敛后旋转权重仍足够大。
#     - 增大 IK 迭代次数从 5 → 15，给姿态控制更多修正机会。
#     - 减小 dq 截断从 ±0.1 → ±0.15，允许更大的姿态修正步长。
#
# [FIX-C2] NMPCController4D 默认摆长 L=0.45（应为 0.445）
#   原版 NMPCController4D.__init__ L=0.45，
#   但 NMPCTrajectoryTracker 传入 L=0.445（从 config 读取）。
#   两者不一致导致 MPC 计算的 mocap_target_z = P_ref[2] + 0.45
#   而 tracker 传入的 compensated_target_z 基于 L=0.445。
#   修复：统一默认值为 0.445。
#
# [FIX-C3] obs 索引硬编码为 -4/-3（yaw/yaw_vel）
#   新版 env _get_obs 布局末尾是 [..., joint_q×7, joint_dq×7]（14维）
#   因此 ee_yaw 实际在 obs[-18]（已在 reset() 中修正），
#   但 compute_ee_acceleration 中仍使用 obs[-4] 和 obs[-3]！
#   obs[-4] 实际上是 joint_q[3]（第4关节角），obs[-3] 是 joint_q[4]——
#   把关节角当成末端 yaw 传入 MPC 状态，这是控制完全失效的根源。
#   修复：定义一套清晰的索引常量，compute_ee_acceleration 统一使用。
#
# [FIX-C4] JointSpaceExpert.reset() 初始 EE 位姿来源不准确
#   reset() 直接从 obs 读取初始 EE 位姿，但此时 env 刚 reset 完毕，
#   _prev_ee_pos 尚未初始化，obs 中的速度项为 0/inf，
#   更重要的是：ee_vz 来自 obs[-25]，这是数值微分值，
#   reset 第一帧时该值为 0（_prev_ee_pos 被设为当前 ee_pos），正确。
#   但问题在于：如果 env reset 时 _prev_ee_pos 与实际 EE 位置有偏差，
#   第一步 ee_vel 会出现一个大的虚假初速度冲击，导致积分爆炸。
#   修复：reset 时直接从 env.data 读取精确的初始 EE 位置和速度，
#   而不是依赖 obs 的数值微分速度（第一帧速度一律置 0）。
#   同时 JointSpaceExpert.reset 新增 env 参数（可选），直接读取物理状态。
#
# [FIX-C5] NMPCTrajectoryTracker L 初始化不一致
#   __init__ 中 self.estimated_L = L（由 config 传入 0.445），
#   但 NMPCController4D 由 L=0.45 构造（FIX-C2 之前），
#   导致 compensated_target_z 计算中：
#     compensated_target_z = target_wp_z + (estimated_L - mpc.L)
#                          = target_wp_z + (0.445 - 0.45)
#                          = target_wp_z - 0.005（仅偏5mm，影响较小）
#   FIX-C2 修复后 mpc.L=0.445，estimated_L 初始也=0.445，补偿为0，正确。
#
# [FIX-C6] IK backup/restore 策略污染主仿真状态（间歇性抖动根源）
#   原版 solve_4d 中：
#     backup_qpos = self.data.qpos.copy()
#     ... 迭代修改 self.data.qpos[:7] ...
#     self.data.qpos[:] = backup_qpos   # restore
#     mujoco.mj_kinematics(...)         # re-forward with backup
#   这看起来安全，但 mj_kinematics 只更新运动学，不更新约束/接触状态。
#   在 env.step() 的 mj_step 循环中间调用 IK（controller 每步都调用），
#   若 IK 在物理步进完成后但奖励计算前调用，restore 的 forward 会干扰
#   下一个 mj_step 的初始状态缓存。
#   修复：将 IK 的 qpos 操作完全移到一个独立的 MjData 副本中，
#   彻底不触碰 self.data（主仿真数据）。
#   注意：NativeIKSolver 初始化时创建 self.scratch_data，之后仅用副本操作。
#
# [FIX-C7] JointSpaceExpert 积分状态与 env 实际 EE 位置发散（长期抖动根源）
#   随着 episode 进行，expert 内部积分出的虚拟 EE 位置
#   与 env 中真实的 EE 位置（由 MuJoCo 物理决定）会逐渐发散。
#   这是因为：真实机械臂有伺服延迟、物理阻尼、惯性，
#   而 expert 的积分是理想的质点积分，没有这些。
#   发散后 IK 的目标位置与机械臂当前位置差距越来越大，
#   IK 步长被 dq clip 限制，无法追上，BC 标签变成不可达目标→噪声。
#   修复：在每步 compute_joint_target 调用时，
#   将内部积分 EE 位置向真实 EE 位置做软对齐（soft anchor）：
#     self._ee_pos = (1-alpha) * self._ee_pos + alpha * real_ee_pos
#   alpha=0.1 时，约 20 步内收敛，阻止长期发散。
#   real_ee_pos 通过新增的 env_obs 解析获得（直接用 obs 中的 ee_x/y/z）。
#
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import casadi as ca
import mujoco


# ==============================================================================
# 观测索引常量（与 mujoco_env_new._get_obs 布局严格对齐）
# 布局：[0-1] ee_xy | [2-3] ee_vxy | [4-5] payload_xy | [6-7] payload_vxy
#        [8-9] rel_t | [10..10+3n-1] obstacles
#        末尾26维（不受障碍物影响）：
#          [-26] ee_z  [-25] ee_vz  [-24] payload_z  [-23] payload_vz
#          [-22] ee_roll  [-21] ee_roll_vel  [-20] ee_pitch  [-19] ee_pitch_vel
#          [-18] ee_yaw   [-17] ee_yaw_vel   [-16] payload_yaw [-15] payload_yaw_vel
#          [-14...-8] joint_q (7)   [-7...-1] joint_dq (7)
# ==============================================================================

# 固定正向索引（在障碍物段之前，始终有效）
OBS_EE_X       = 0
OBS_EE_Y       = 1
OBS_EE_VX      = 2
OBS_EE_VY      = 3
OBS_PL_X       = 4
OBS_PL_Y       = 5
OBS_PL_VX      = 6
OBS_PL_VY      = 7

# 末尾固定负索引（不受障碍物数量影响）
OBS_EE_Z       = -26
OBS_EE_VZ      = -25
OBS_PL_Z       = -24
OBS_PL_VZ      = -23
OBS_EE_ROLL    = -22
OBS_EE_ROLL_V  = -21
OBS_EE_PITCH   = -20
OBS_EE_PITCH_V = -19
OBS_EE_YAW     = -18   # [FIX-C3] 修正，原版错用 -4
OBS_EE_YAW_V   = -17   # [FIX-C3] 修正，原版错用 -3
# joint_q:  obs[-14] ~ obs[-8]  → indices -14,-13,-12,-11,-10,-9,-8
# joint_dq: obs[-7]  ~ obs[-1]  → indices -7,-6,-5,-4,-3,-2,-1


# ==============================================================================
# NMPCController4D
# ==============================================================================

class NMPCController4D:
    """4D NMPC 防摆控制器（ax, ay, az, ayaw）。"""

    def __init__(self, dt=0.1, N=15, L=0.445,          # [FIX-C2] 默认 L=0.445
                 u_max_xy=0.5, u_max_z=2.0, u_max_yaw=2.0):
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

        damping = 0.01
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

        Q_pos   = np.array([10.0, 10.0, 20.0, 5.0])
        Q_swing = np.array([50.0, 50.0])
        Q_vel   = 2.0
        R_acc   = np.array([0.2, 0.2, 0.2, 0.3])

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
            cost += Q_vel * (X[4,k]**2 + X[5,k]**2 + X[6,k]**2 + X[7,k]**2)
            cost += (R_acc[0]*U[0,k]**2 + R_acc[1]*U[1,k]**2
                   + R_acc[2]*U[2,k]**2 + R_acc[3]*U[3,k]**2)

        cost += 10 * (Q_pos[0]*(X[8, self.N]-P_ref[0])**2
                    + Q_pos[1]*(X[9, self.N]-P_ref[1])**2)
        cost += 10 * (Q_pos[2]*(X[2, self.N]-mocap_target_z)**2
                    + Q_pos[3]*(X[3, self.N]-P_ref[3])**2)

        nlp = {
            'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
            'f': cost, 'g': ca.vertcat(*constraints),
            'p': ca.vertcat(X_init, P_ref, Az_lin),
        }
        opts = {
            'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes',
            'ipopt.max_iter': 50, 'ipopt.tol': 1e-2,
            'ipopt.acceptable_tol': 1e-1,
            'ipopt.warm_start_init_point': 'yes',
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        n_vars = self.nx * (self.N + 1) + self.nu * self.N
        self.lbx = -ca.inf * np.ones(n_vars)
        self.ubx =  ca.inf * np.ones(n_vars)
        u_start  = self.nx * (self.N + 1)
        for i in range(self.N):
            idx = u_start + i * self.nu
            self.lbx[idx:idx+self.nu] = [-u_max_xy, -u_max_xy, -u_max_z, -u_max_yaw]
            self.ubx[idx:idx+self.nu] = [ u_max_xy,  u_max_xy,  u_max_z,  u_max_yaw]

        self.lbg = np.zeros(self.nx * (self.N + 1))
        self.ubg = np.zeros(self.nx * (self.N + 1))
        self.last_sol = None

    def get_action(self, state_12d: np.ndarray, target_4d: np.ndarray) -> np.ndarray:
        """
        state_12d: [ee_x,y,z,yaw, ee_vx,vy,vz,yaw_vel, pl_x,y, pl_vx,vy]
        target_4d: [target_pl_x, target_pl_y, target_pl_z, target_yaw]
        Returns: (4,) [ax, ay, az, ayaw]
        """
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
    def __init__(self, dt=0.1, N=15, L=0.445,
                 arrival_threshold_xy=0.05, arrival_threshold_z=0.20):
        self.mpc = NMPCController4D(dt=dt, N=N, L=L)  # [FIX-C2] L 对齐
        self.path = None
        self.current_idx = 0
        self.arrival_threshold_xy = arrival_threshold_xy
        self.arrival_threshold_z  = arrival_threshold_z
        self.estimated_L = L      # [FIX-C5] 与 mpc.L 一致，初始补偿为 0

    def set_path(self, path):
        if path is None or len(path) == 0:
            self.path = None; self.current_idx = 0; return
        self.path = np.array(path, dtype=np.float64)
        self.current_idx = 0
        self.mpc.last_sol = None
        self.mpc.last_az  = 0.0

    def compute_ee_acceleration(self, obs: np.ndarray,
                                 target_yaw: float = 0.0) -> np.ndarray:
        """
        从 obs 提取状态并调用 NMPC，返回 4D EE 加速度。
        [FIX-C3] 全部通过常量索引读取，彻底消除旧版 -4/-3 错误。
        """
        if self.path is None:
            return np.zeros(4)

        # 固定正向索引（障碍物段之前）
        ee_x  = float(obs[OBS_EE_X]);  ee_y  = float(obs[OBS_EE_Y])
        ee_vx = float(obs[OBS_EE_VX]); ee_vy = float(obs[OBS_EE_VY])
        pl_x  = float(obs[OBS_PL_X]);  pl_y  = float(obs[OBS_PL_Y])
        pl_vx = float(obs[OBS_PL_VX]); pl_vy = float(obs[OBS_PL_VY])

        # 末尾负索引（不受障碍物数量影响）
        ee_z      = float(obs[OBS_EE_Z])
        ee_vz     = float(obs[OBS_EE_VZ])
        pl_z      = float(obs[OBS_PL_Z])
        ee_yaw    = float(obs[OBS_EE_YAW])    # [FIX-C3] 原版错误地用 obs[-4]
        ee_yaw_v  = float(obs[OBS_EE_YAW_V])  # [FIX-C3] 原版错误地用 obs[-3]

        # 动态摆长估计（软滤波）
        actual_L = ee_z - pl_z
        if 0.1 < actual_L < 1.0:  # 合理范围保护（绳松弛时跳过）
            self.estimated_L = 0.9 * self.estimated_L + 0.1 * actual_L

        # 航点切换
        curr_pl_xy = np.array([pl_x, pl_y])
        wp = self.path[self.current_idx]
        wp3 = np.array([wp[0], wp[1], wp[2] if len(wp) >= 3 else 0.3])

        dist_xy = np.linalg.norm(curr_pl_xy - wp3[:2])
        dist_z  = abs(pl_z - wp3[2])

        if (dist_xy < self.arrival_threshold_xy and
                dist_z < self.arrival_threshold_z and
                self.current_idx < len(self.path) - 1):
            self.current_idx += 1
            wp  = self.path[self.current_idx]
            wp3 = np.array([wp[0], wp[1], wp[2] if len(wp) >= 3 else 0.3])
            self.mpc.last_sol = None
            self.mpc.last_az  = 0.0

        state_12d = np.array([
            ee_x, ee_y, ee_z, ee_yaw,
            ee_vx, ee_vy, ee_vz, ee_yaw_v,
            pl_x, pl_y, pl_vx, pl_vy,
        ], dtype=np.float64)

        # 补偿目标 Z：使 MPC 期望的 mocap_target_z = wp3[2] + estimated_L
        compensated_z = wp3[2] + (self.estimated_L - self.mpc.L)
        P_ref = np.array([wp3[0], wp3[1], compensated_z, target_yaw],
                         dtype=np.float64)

        return self.mpc.get_action(state_12d, P_ref)


# ==============================================================================
# NativeIKSolver（修复版）
# ==============================================================================

class NativeIKSolver:
    """
    MuJoCo 原生微分 IK 求解器。

    [FIX-C6] 使用独立的 scratch_data 副本进行 IK 迭代，
             彻底不修改主仿真 self.data。
    [FIX-C1] 提升旋转权重 w_rot_base，增加迭代次数，
             确保末端垂直约束被充分追踪。
    """

    def __init__(self, mj_model, mj_data):
        self.model = mj_model
        self.data  = mj_data  # 主仿真数据（只读，IK 不修改）

        # [FIX-C6] 专用于 IK 迭代的独立数据副本
        self.scratch = mujoco.MjData(mj_model)

        self.target_frame = "link7"
        self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                          self.target_frame)
        self.is_site = True
        if self.obj_id == -1:
            self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                              self.target_frame)
            self.is_site = False
        if self.obj_id == -1:
            raise ValueError(f"找不到 '{self.target_frame}'，请检查 XML 中的 site/body 名称")

        self.damping        = 1e-2          # 阻尼（适当降低以加快收敛）
        self.nullspace_gain = 0.05
        self.w_pos          = 1.0
        self.w_rot_base     = 0.5           # [FIX-C1] 从 0.15 提升到 0.5
        self.max_iters      = 15            # [FIX-C1] 从 5 提升到 15
        self.dq_clip        = 0.15          # [FIX-C1] 从 0.1 放宽到 0.15

        self._update_limits()

    def _update_limits(self):
        """从当前模型读取关节限位（每次模型更新后调用）。"""
        self.q_min       = self.model.jnt_range[:7, 0].copy()
        self.q_max       = self.model.jnt_range[:7, 1].copy()
        self.jnt_limited = self.model.jnt_limited[:7].copy()
        self.q_margin    = 0.15 * (self.q_max - self.q_min)

    def update_model(self, mj_model, mj_data):
        """每次 env reset 后更新模型引用（场景重载）。"""
        self.model   = mj_model
        self.data    = mj_data
        self.scratch = mujoco.MjData(mj_model)  # [FIX-C6] 重建副本

        # 重新查找 ID
        self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE,
                                          self.target_frame)
        self.is_site = True
        if self.obj_id == -1:
            self.obj_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                              self.target_frame)
            self.is_site = False
        self._update_limits()

    def solve_4d(self, current_q: np.ndarray,
                 target_x: float, target_y: float,
                 target_z: float, target_yaw: float) -> np.ndarray:
        """
        求解 7 关节角使末端到达 (target_x, target_y, target_z, target_yaw)，
        末端保持垂直朝下（roll=π, pitch=0）。

        [FIX-C6] 完全在 self.scratch 副本上操作，不污染主仿真。
        [FIX-C1] 旋转权重提升，迭代次数增加。

        Returns:
            q_result: (7,) 关节角（rad）
        """
        # 构建目标旋转矩阵：末端垂直朝下 + target_yaw 偏航
        # ZYX 内旋：先绕 X 翻转 π（朝下），再绕 Z 旋转 target_yaw
        roll, pitch, yaw = np.pi, 0.0, target_yaw
        cx, sx = np.cos(roll),  np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw),   np.sin(yaw)
        R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx,  cx]])
        R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        Rmat = R_z @ R_y @ R_x  # 外旋 ZYX = 内旋 XYZ

        target_pos  = np.array([target_x, target_y, target_z])
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, Rmat.flatten())

        # [FIX-C6] 将主仿真当前状态复制到 scratch，作为 IK 起点
        self.scratch.qpos[:] = self.data.qpos[:]
        self.scratch.qvel[:] = self.data.qvel[:]
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)

        q_guess = current_q.copy()

        for iteration in range(self.max_iters):  # [FIX-C1]
            self.scratch.qpos[:7] = q_guess
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)

            if self.is_site:
                cp = self.scratch.site_xpos[self.obj_id].copy()
                cm = self.scratch.site_xmat[self.obj_id].reshape(3, 3).copy()
            else:
                cp = self.scratch.xpos[self.obj_id].copy()
                cm = self.scratch.xmat[self.obj_id].reshape(3, 3).copy()

            # 计算误差
            cq = np.zeros(4)
            mujoco.mju_mat2Quat(cq, cm.flatten())
            pe = target_pos - cp
            re = np.zeros(3); nq = np.zeros(4); eq = np.zeros(4)
            mujoco.mju_negQuat(nq, cq)
            mujoco.mju_mulQuat(eq, target_quat, nq)
            if eq[0] < 0: eq = -eq
            mujoco.mju_quat2Vel(re, eq, 1.0)

            pn = np.linalg.norm(pe)
            rn = np.linalg.norm(re)

            # 收敛判定（旋转精度更严格）
            if pn < 5e-4 and rn < 5e-3:
                break

            # [FIX-C1] 动态旋转权重：位置误差大时适度保留旋转追踪能力
            # 原版 (0.02 / (0.02 + pn)) 在 pn=0.1m 时衰减到 0.16×w_rot_base
            # 新版用更温和的衰减，确保旋转始终得到关注
            wr = self.w_rot_base * (0.05 / (0.05 + pn))  # [FIX-C1]

            # 误差截断
            if pn > 0.05: pe = (pe / pn) * 0.05
            if rn > 0.15: re = (re / rn) * 0.15

            # 雅可比矩阵
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            if self.is_site:
                mujoco.mj_jacSite(self.model, self.scratch, jacp, jacr, self.obj_id)
            else:
                mujoco.mj_jacBody(self.model, self.scratch, jacp, jacr, self.obj_id)
            J = np.vstack([jacp, jacr])[:, :7]

            J_w   = J.copy()
            J_w[:3] *= self.w_pos
            J_w[3:] *= wr
            err_w = np.concatenate([pe * self.w_pos, re * wr])
            JJT_w = J_w @ J_w.T
            diag  = (self.damping ** 2) * np.eye(6)
            dq    = J_w.T @ np.linalg.solve(JJT_w + diag, err_w)

            # 零空间：关节极限拉回
            grad = np.zeros(7)
            for j in range(7):
                if self.jnt_limited[j]:
                    if q_guess[j] > self.q_max[j] - self.q_margin[j]:
                        grad[j] = (self.q_max[j] - self.q_margin[j]) - q_guess[j]
                    elif q_guess[j] < self.q_min[j] + self.q_margin[j]:
                        grad[j] = (self.q_min[j] + self.q_margin[j]) - q_guess[j]
            if np.any(grad != 0):
                J_sq  = J @ J.T + diag
                J_inv = J.T @ np.linalg.solve(J_sq, np.eye(6))
                dq   += (np.eye(7) - J_inv @ J) @ (self.nullspace_gain * grad)

            dq = np.clip(dq, -self.dq_clip, self.dq_clip)  # [FIX-C1]
            q_guess += dq
            for j in range(7):
                if self.jnt_limited[j]:
                    q_guess[j] = np.clip(q_guess[j], self.q_min[j], self.q_max[j])

        # [FIX-C6] 不 restore scratch（它是独立副本，不影响主仿真）
        return q_guess


# ==============================================================================
# JointSpaceExpert — BC 标签生成器（修复版）
# ==============================================================================

class JointSpaceExpert:
    """
    BC 标签生成器：NMPC → 积分 → IK → 7D 关节角目标。

    修复要点：
    [FIX-C3] obs 索引通过常量访问，彻底消除 yaw 读取错误
    [FIX-C4] reset 时速度置 0（防初始冲击）
    [FIX-C7] 每步对积分状态做软锚定（防长期发散）
    """

    def __init__(self, config: dict, ik_solver: NativeIKSolver):
        ctrl_cfg = config["controller"]
        self.tracker = NMPCTrajectoryTracker(
            dt=ctrl_cfg["dt"],
            N=ctrl_cfg["N"],
            L=ctrl_cfg["L"],
            arrival_threshold_xy=ctrl_cfg["arrival_threshold_xy"],
            arrival_threshold_z=ctrl_cfg["arrival_threshold_z"],
        )
        self.ik_solver = ik_solver
        self.dt = ctrl_cfg["dt"]

        # 内部虚拟 EE 积分状态
        self._ee_pos     = np.zeros(3, dtype=np.float64)
        self._ee_vel     = np.zeros(3, dtype=np.float64)
        self._ee_yaw     = 0.0
        self._ee_yaw_vel = 0.0
        self._last_q     = None

        # [FIX-C7] 软锚定系数（每步将积分拉向真实 EE 位置的比例）
        self._anchor_alpha = 0.1

    def reset(self, env_obs: np.ndarray, init_q: np.ndarray,
              env=None):
        """
        每回合开始时调用，同步初始状态。

        [FIX-C4] 速度置零（第一帧数值微分不可靠）。
        [FIX-C3] 通过常量索引读取 EE 位姿。

        Args:
            env_obs: env.reset() 返回的初始观测
            init_q:  机械臂初始关节角 (7,)
            env:     可选，传入 env 实例以直接从物理状态读取精确 EE 位置
        """
        if env is not None:
            # 优先从物理引擎直接读取，最准确
            ee_pos = env._get_ee_pos()
            self._ee_pos = ee_pos.astype(np.float64)
        else:
            # fallback：从 obs 读取
            self._ee_pos = np.array([
                float(env_obs[OBS_EE_X]),
                float(env_obs[OBS_EE_Y]),
                float(env_obs[OBS_EE_Z]),
            ], dtype=np.float64)

        # [FIX-C4] 速度一律置 0，避免数值微分带来的初始冲击
        self._ee_vel     = np.zeros(3, dtype=np.float64)
        self._ee_yaw     = float(env_obs[OBS_EE_YAW])  # [FIX-C3]
        self._ee_yaw_vel = 0.0
        self._last_q     = init_q.copy().astype(np.float64)

        self.tracker.mpc.last_sol = None
        self.tracker.mpc.last_az  = 0.0

    def set_path(self, path):
        self.tracker.set_path(path)

    def compute_joint_target(self, env_obs: np.ndarray,
                              current_q: np.ndarray,
                              target_yaw: float = 0.0) -> np.ndarray:
        """
        计算本步的 BC 目标关节角。

        [FIX-C3] obs 通过常量索引读取
        [FIX-C7] 积分状态软锚定到真实 EE 位置
        """
        # ── 1. NMPC 计算 4D EE 加速度 ──────────────────────────────────────
        action_4d = self.tracker.compute_ee_acceleration(env_obs, target_yaw)
        a_xyz = action_4d[:3].astype(np.float64)
        a_yaw = float(action_4d[3])

        # ── 2. 二阶 Euler 积分 ───────────────────────────────────────────────
        dt = self.dt
        self._ee_pos += self._ee_vel * dt + 0.5 * a_xyz * dt ** 2
        self._ee_vel += a_xyz * dt
        self._ee_yaw     += self._ee_yaw_vel * dt + 0.5 * a_yaw * dt ** 2
        self._ee_yaw_vel += a_yaw * dt

        # 地板夹紧保护
        if self._ee_pos[2] < 0.25:
            self._ee_pos[2] = 0.25
            self._ee_vel[2] = max(self._ee_vel[2], 0.0)

        # ── 3. [FIX-C7] 软锚定：将积分 EE 向真实 EE 位置拉近 ────────────────
        real_ee_x = float(env_obs[OBS_EE_X])
        real_ee_y = float(env_obs[OBS_EE_Y])
        real_ee_z = float(env_obs[OBS_EE_Z])
        real_ee   = np.array([real_ee_x, real_ee_y, real_ee_z], dtype=np.float64)

        alpha = self._anchor_alpha
        self._ee_pos = (1.0 - alpha) * self._ee_pos + alpha * real_ee

        # ── 4. IK 求解 ──────────────────────────────────────────────────────
        q_start = self._last_q if self._last_q is not None else current_q
        q_target = self.ik_solver.solve_4d(
            current_q=q_start.astype(np.float64),
            target_x=float(self._ee_pos[0]),
            target_y=float(self._ee_pos[1]),
            target_z=float(self._ee_pos[2]),
            target_yaw=float(self._ee_yaw),
        )

        # IK 失败保护
        if q_target is None or np.any(np.isnan(q_target)):
            q_target = current_q.copy()

        self._last_q = q_target.copy()
        return q_target.astype(np.float32)