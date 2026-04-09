# ==============================================================================
# 修复日志（对照上一版 mujoco_env_new.py 的全部问题逐条处理）：
#
# [BUG-1] __init__ 中 self.config["obstacle"] → 改为 self.config["scene"]
#         Config 中对应键名为 "scene"，原版写错导致实例化即 KeyError。
#
# [BUG-2] generate_scene_and_trajectory 缺少 @staticmethod，裸调用触发 NameError。
#         添加装饰器，并在 reset() 调用处改为 CableRobotEnvWithObstacles.generate_scene_and_trajectory(...)
#
# [BUG-3] reset() 传入 config=self.config["planning"]，但函数还需要 "scene" 中的
#         radius_range / n_obstacles / path_width / obstacle_z_center /
#         obstacle_halfheight / endpoint_z_offset 共 6 个键，全部 KeyError。
#         修复：传入合并后的 scene_plan_config = {**scene_cfg, **plan_cfg}。
#
# [BUG-4] _get_obs() 中 obs[4]/obs[5] 索引对齐了新的观测布局（mocap 8维在前），
#         但 reset() / step() / _compute_reward() 里仍用旧版索引 obs[4]/obs[5]
#         取负载 XY。修复：统一改为读取 payload 真实 xpos，不再依赖 obs 索引。
#
# [BUG-5] 运动学积分顺序与旧版不一致：
#         旧版：pos += vel*dt + 0.5*a*dt²；vel += a*dt （显式 Euler 二阶修正）
#         新版：vel += a*dt；pos += vel*dt       （半隐式 Euler，动力学不同）
#         修复：恢复旧版积分公式，保持与预训练策略的一致性。
#
# [BUG-6] 缺少 mocap_pos[2] < 0.4 地板夹紧保护，动捕点可穿地导致物理爆炸。
#         修复：在积分后加回地板夹紧逻辑。
#
# [BUG-7] 缺少 NaN 检测保护（旧版在物理步进后有 isnan 检查）。
#         修复：在 mj_step 循环后加入 NaN 检测，发现即提前终止并惩罚。
#
# [BUG-8] current_step 在 step 最开头递增（旧版在末尾）。
#         超时判定 >= max_steps 可能比旧版提前 1 步触发。
#         修复：移至 step 末尾递增，与旧版对齐。
#
# [BUG-9] 航点切换后 last_dist 未清零（缺少 last_wp_idx 重置逻辑），
#         导致切换航点第一步产生虚假大进展奖励。
#         修复：引入 self.last_wp_idx，切换航点时清空 last_dist。
#
# [BUG-10] look_ahead_dist 使用固定配置值，丢失旧版接近终点时收紧为 0.06 的
#          自适应逻辑。修复：恢复动态 look_ahead（尾部 2 个航点用较小阈值）。
#
# [BUG-11] 成功条件被极度简化，仅靠 reached_final 判定，丢失旧版的
#          速度 / 高度 / 水平距离多条件门控。
#          修复：恢复旧版成功判定（dist<0.15, vel_xy<0.2, abs(vz)<0.5）。
#
# [BUG-12] 成功奖励 +15.0 和超时惩罚 -5.0 在新版中未实现。
#          修复：加回旧版对应的奖励项（值可通过 Config 调节）。
#
# [BUG-13] action_buffer 与 action_queue 并存（两个延迟队列变量冗余）。
#          修复：__init__ 中直接初始化 action_queue，删除 action_buffer。
#
# [IMPROVE-1] reset() 中 tempfile 删除时机改进：
#             先完成模型加载，再在 finally 中删除，确保 XML 在加载期间存在。
#             （原版 finally 立即删除在某些平台可能有竞争，已改为加载成功后单独删除）
#
# [IMPROVE-2] sim_steps 在 reset() 重载模型后重新计算，防止外部修改 physics_dt 时错乱。
#
# [IMPROVE-3] _get_obs() 的 obs 索引重新设计以与旧版保持兼容：
#             保留旧版的 obs[4]/obs[5] = payload XY 语义（通过调整字段顺序实现）。
#             观测布局：[mocap_x, mocap_y, mocap_vx, mocap_vy,
#                        payload_x, payload_y, payload_vx, payload_vy,
#                        rel_tx, rel_ty,                           ← 旧版10维
#                        obstacle_data...,                         ← n_obs*3
#                        mocap_z, mocap_vz, payload_z, payload_vz, ← Z轴4维
#                        mocap_roll, mocap_roll_vel,               ← 新增姿态4维
#                        mocap_pitch, mocap_pitch_vel,
#                        mocap_yaw, mocap_yaw_vel,                 ← 新增旋转2维
#                        payload_yaw, payload_yaw_vel]             ← 新增旋转2维
#             总计: 10 + n_obstacles*3 + 4 + 8 = 22 + n_obstacles*3
#             负索引：[-12]=mocap_z  [-11]=mocap_vz  [-10]=payload_z  [-9]=payload_vz
#                     [-8]=mocap_roll [-7]=mocap_roll_vel [-6]=mocap_pitch [-5]=mocap_pitch_vel
#                     [-4]=mocap_yaw  [-3]=mocap_yaw_vel  [-2]=payload_yaw [-1]=payload_yaw_vel
#
# ──────────────────────────────────────────────────────────────────────────────
# [ENV-BUG-A] reset() 中 initial_ee_pos 在 warmup 前读取（mj_forward 后），
#             warmup 完成再次 mj_forward 但未更新 initial_ee_pos，
#             导致 current_mocap_pos 轻微偏离实际 EE 位置。
#             修复：warmup 后重新读取 EE 位置再赋给 current_mocap_pos。
#
# [ENV-BUG-B] current_mocap_pos[2] 初始化为真实 EE Z（通常 ~1.0m），
#             而 path_3d[0] 中 payload 巡航高度仅 0.30m，
#             MPC 需令 mocap_z 从 1.0 降至 0.70（0.30+L），
#             幅度 0.30m，以 u_max_z=0.5 m/s² 需 ~25 步，
#             这期间绳子因末端下降而松弛，payload 无法被提升，
#             是"上升一段后失稳"的根本原因。
#             修复：与 nmpc_controller_new 联动，将 u_max_z 放宽至 2.0 m/s²
#             并在 NMPCController4D 提高 Q_pos[2] 至 20.0。
#             （env 侧无需改代码，由 controller 参数决定）
#
# [ENV-BUG-C] ground_clamp: 地板夹紧阈值 0.4m 过高。
#             若 payload 巡航高度目标 0.30m、末端目标 0.70m（0.3+L=0.3+0.4=0.7），
#             则末端在下降过程中不会触发夹紧。但若某些场景 payload_z_cruise 配置更低，
#             末端目标 Z 可能接近 0.4m 导致夹紧提前激活，产生速度突变。
#             修复：地板夹紧改为更保守的 0.25m（EE 在此高度以下绳子必然碰地）。
# ==============================================================================
 
import os
import re
import heapq
import tempfile
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R  # 【新增】处理欧拉角与四元数转换
import pinocchio
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.ik.nullspace_components import joint_limit_nullspace_component
 
# 引入独立配置文件
from config import DEFAULT_CONFIG
 
 
class CableRobotEnvWithObstacles:
    """
    配置驱动的索驱动机器人物理仿真环境（带动态障碍物）。
    动作空间: 6D [a_x, a_y, a_z, a_roll, a_pitch, a_yaw]
 
    观测布局（共 22 + 3*n_obstacles 维，末尾 12 维用负索引访问）：
      [0]  mocap_x        动捕点（虚拟末端）X 位置
      [1]  mocap_y        动捕点 Y 位置
      [2]  mocap_vx       动捕点 X 速度
      [3]  mocap_vy       动捕点 Y 速度
      [4]  payload_x      负载 X 位置（与旧版 obs[4] 对齐）
      [5]  payload_y      负载 Y 位置（与旧版 obs[5] 对齐）
      [6]  payload_vx     负载 X 速度（与旧版 obs[6] 对齐）
      [7]  payload_vy     负载 Y 速度（与旧版 obs[7] 对齐）
      [8]  rel_tx         目标相对负载的 X 距离（与旧版 obs[8] 对齐）
      [9]  rel_ty         目标相对负载的 Y 距离（与旧版 obs[9] 对齐）
      [10 ~ 10+3*n-1]     障碍物信息 (ox, oy, r) * n_obstacles
      ── 末尾 12 维，用负索引访问，不受 n_obstacles 影响 ──
      [-12] mocap_z       动捕点 Z 位置
      [-11] mocap_vz      动捕点 Z 速度
      [-10] payload_z     负载 Z 位置
      [-9]  payload_vz    负载 Z 速度
      [-8]  mocap_roll    动捕点 roll
      [-7]  mocap_roll_vel
      [-6]  mocap_pitch   动捕点 pitch
      [-5]  mocap_pitch_vel
      [-4]  mocap_yaw     动捕点 yaw
      [-3]  mocap_yaw_vel
      [-2]  payload_yaw   负载 yaw（占位，当前为 0）
      [-1]  payload_yaw_vel（占位，当前为 0）
 
    ⚠️ 控制器读取末尾物理状态时必须使用负索引，正向索引会被障碍物数据偏移！
    """
 
    def __init__(self, config: dict = None):
        # ======================================================================
        # 1. 合并配置项（支持用户传入部分覆盖）
        # ======================================================================
        # 深拷贝防止修改全局 DEFAULT_CONFIG
        import copy
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        if config is not None:
            for key, val in config.items():
                if isinstance(val, dict) and key in self.config:
                    self.config[key].update(val)
                else:
                    self.config[key] = val
 
        cfg_sim   = self.config["sim"]
        cfg_space = self.config["space"]
        cfg_task  = self.config["task"]
        # [BUG-1 修复] 原版写的是 "obstacle"，Config 中实际键名为 "scene"
        cfg_scene = self.config["scene"]
        cfg_plan  = self.config["planning"]
        cfg_noise = self.config["noise"]

        # [核心更新] 提取 reward 和 step_logic 字典为成员变量，替代魔法数字
        self.cfg_reward = self.config.get("reward", {})
        self.cfg_logic  = self.config.get("step_logic", {})
 
        # ======================================================================
        # 2. 解析时间与控制参数
        # ======================================================================
        self.physics_dt      = cfg_sim["physics_dt"]        # 物理步长 (500 Hz)
        self.control_freq_hz = cfg_sim["control_freq_hz"]   # 控制频率 (10 Hz)
        self.control_dt      = 1.0 / self.control_freq_hz
        self.dt              = self.control_dt               # 外部惯用别名
        self.sim_steps       = int(self.dt / self.physics_dt)  # 每控制步循环次数 (50)
        self.max_steps       = cfg_sim["max_steps"]          # 回合最大步数
        self.current_step    = 0
        self.substep_skip    = cfg_sim.get("substep_skip", 1) # 物理奖励子步跳过
 
        # ======================================================================
        # 3. 任务与几何参数预加载
        # ======================================================================
        # 动作边界支持逐维不同（4D 向量）
        self.action_space_high = np.array(cfg_space["action_space_high"])
        self.action_dim        = cfg_space["action_dim"]
 
        self.start_pos_mocap  = np.array(cfg_task["start_pos_mocap"])
        # [新增] 动捕点初始偏航四元数 (w, x, y, z)
        self.start_quat_mocap = np.array(cfg_task["start_quat_mocap"])
 
        self.default_start_xy   = np.array(cfg_task["default_start_xy"])
        self.default_target     = np.array(cfg_task["default_target_xy"])
        self.target_pos         = self.default_target.copy()
 
        self.init_position_range = cfg_task["init_position_range"]
        self.init_velocity_scale = cfg_task["init_velocity_scale"]  # 保留，当前未注入（与旧版一致）
 
        # [BUG-1 修复] 从正确的 "scene" 键读取障碍物参数
        self.n_obstacles          = cfg_scene["n_obstacles"]
        self.obstacle_radius_range = cfg_scene["radius_range"]
        self._obstacle_rng        = np.random.default_rng(cfg_scene["seed"])
 
        # 规划参数
        self.path_width       = cfg_scene["path_width"]   # 障碍物横向分布宽度，来自 scene
        self.payload_radius   = cfg_plan["payload_radius"]
        self.planning_margin  = cfg_plan["planning_margin"]
        self.planning_grid_res = cfg_plan["planning_grid_res"]
 
        # ======================================================================
        # 4. 延迟与噪声参数
        # ======================================================================
        self.latency_steps    = cfg_noise["latency_steps"]
        self.force_noise_level = cfg_noise["force_noise_level"]   # 保留，当前未注入（与旧版一致）
        # [BUG-13 修复] 删除冗余的 action_buffer，直接初始化唯一的延迟队列 action_queue
        self.action_queue = deque(maxlen=max(1, self.latency_steps + 1))
        for _ in range(self.latency_steps):
            self.action_queue.append(np.zeros(self.action_dim))
 
        # ======================================================================
        # 5. 状态维度计算
        # ======================================================================
        # 10（旧版基础 XY 段）+ n_obstacles*3（障碍物）+ 4（Z轴）+ 8（完整3D姿态：roll/pitch/yaw + 速度）
        # 末尾共 12 维，通过负索引访问（[-12] ~ [-1]）
        self.state_dim = 10 + (self.n_obstacles * 3) + 4 + 8
 
        # ======================================================================
        # 6. XML 资产管理与模型初始化
        # ======================================================================
        current_dir   = os.path.dirname(os.path.abspath(__file__))
        self._assets_dir = os.path.join(current_dir, "assets")
 
        # Auto-regenerate rope XML from config before loading
        from assets.generate_four_cables_with_plate import main as generate_rope_xml
        generate_rope_xml()
 
        base_xml_path = os.path.join(
            self._assets_dir,
            "demo_fourCable_withSteel_withSensor_cylinder.xml"
        )
        if not os.path.exists(base_xml_path):
            raise FileNotFoundError(f"Base XML not found: {base_xml_path}")
 
        with open(base_xml_path, "r", encoding="utf-8") as f:
            self._base_xml_content = f.read()
 
        self.model = mujoco.MjModel.from_xml_path(base_xml_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = self.physics_dt
 
        # ----------------------------------------------------------------------
        # 【核心修复】：让 Pinocchio 只加载纯净的机械臂模型
        # ----------------------------------------------------------------------
        import pinocchio as pin
        
        # 方案：你需要在这里指定一个【只有 KUKA 机械臂】的模型文件。
        # 请根据你 assets 文件夹下的实际文件名进行修改，比如 "iiwa14.xml" 或 "iiwa14.urdf"
        pure_arm_model_name = "iiwa14.xml" 
        
        arm_model_path = os.path.join(self._assets_dir, pure_arm_model_name)
        
        if not os.path.exists(arm_model_path):
            raise FileNotFoundError(f"找不到纯机械臂模型文件: {arm_model_path}，请提供仅包含机械臂的 URDF 或 XML。")
 
        try:
            # 如果是 URDF 文件，用 buildModelFromUrdf
            if arm_model_path.endswith(".urdf"):
                self.model_pin = pin.buildModelFromUrdf(arm_model_path)
            # 如果是纯 XML 文件，用 buildModelFromMjcf
            else:
                self.model_pin = pin.buildModelFromMJCF(arm_model_path)
                
            self.data_pin  = self.model_pin.createData()
            
            # 实例化你的 IK 求解器
            self.ik_solver = MPCPinocchioIKSolver(self.model_pin, self.data_pin)
            print("✅ IK Solver (Pinocchio) 初始化成功！")
            
        except Exception as e:
            print(f"❌ Pinocchio 加载机械臂模型失败: {e}")
            raise e
 
        # ======================================================================
        # 7. MuJoCo 对象 ID 寻址
        # ======================================================================
        self._reresolve_ids()
 
        # ======================================================================
        # 8. 内部状态占位符初始化
        # ======================================================================
        self._obstacles      = []
        self._temp_xml_path  = None
        self._planned_path   = None
 
        # 【修改】完全废弃实体 mocap，但保留变量名 current_mocap_pos/euler 
        # 作为 IK 求解器的“虚拟目标 (Virtual Target)”，以免破坏其他代码的兼容性
        self.current_mocap_pos = np.zeros(3)
        self.current_mocap_vel = np.zeros(3)
        self.current_mocap_euler = np.zeros(3)
        self.current_mocap_euler_vel = np.zeros(3)
        self.current_mocap_yaw     = 0.0
        self.current_mocap_yaw_vel = 0.0
 
        self.current_wp_idx = 0
        self.reached_final  = False
        self.last_dist      = None
        self.last_wp_idx    = -1
        # ======================================================================
        # 9. 渲染器初始化
        # ======================================================================
        self.render_mode = cfg_sim["render"]
        self.viewer      = None
        if self.render_mode:
            kw = {}
            if hasattr(self, '_key_callback') and self._key_callback is not None:
                kw['key_callback'] = self._key_callback
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, **kw)
 
    # ==========================================================================
    # 辅助：重新解析 MuJoCo ID（每次重载 XML 后调用）
    # ==========================================================================
    def _reresolve_ids(self):
        """重新绑定 MuJoCo 体/关节 ID，在每次 XML 重载后必须调用。"""
        self.prefab_jnt_id  = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("target").id
        self.ee_site_id     = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
 
    # ==========================================================================
    # 一站式场景生成器（静态方法，无需实例）
    # ==========================================================================
    # [BUG-2 修复] 添加 @staticmethod，消除"第一个参数被当成 self"的错误；
    # [BUG-3 修复] 函数签名接收合并后的 scene_plan_config（包含 scene+planning 两部分的键），
    #             而非单独的 "planning" 子 dict。
    @staticmethod
    def generate_scene_and_trajectory(start_xy, target_xy, base_xml_content,
                                      scene_plan_config, rng=None):
        """
        一站式环境生成器：随机生成障碍物、执行 2D A* 规划、生成 3D 轨迹、并组装 XML。
        """
        if rng is None:
            rng = np.random.default_rng()

        start_xy  = np.asarray(start_xy,  dtype=float).reshape(2)
        target_xy = np.asarray(target_xy, dtype=float).reshape(2)

        # 统一读取来自合并 dict 的参数
        p_radius    = scene_plan_config["payload_radius"]
        p_margin    = scene_plan_config["planning_margin"]
        min_clearance = p_radius + p_margin   # 起终点防撞保护圈半径

        # 【新增】：定义机械臂基座为虚拟障碍物
        base_xy = np.array([0.0, 0.0])
        base_radius = 0.15  # 假设机械臂基座和第一关节占据的物理半径约为 0.15 米

        # ======================================================================
        # 步骤 1：在两点连线附近随机生成障碍物
        # ======================================================================
        delta     = target_xy - start_xy
        length    = np.linalg.norm(delta)
        direction = np.array([1.0, 0.0]) if length < 1e-6 else delta / length
        normal    = np.array([-direction[1], direction[0]])   # 垂直于连线方向

        obstacles = []
        r_min, r_max = scene_plan_config["radius_range"]

        for _ in range(scene_plan_config["n_obstacles"]):
            center = None
            r      = None
            for _ in range(500):   # 最多尝试 500 次找到合法位置
                t      = rng.uniform(0.15, 0.85)   # 限制在连线中间段，避免压在起终点
                along  = start_xy + t * delta
                off    = rng.uniform(-scene_plan_config["path_width"],
                                      scene_plan_config["path_width"])
                center = along + off * normal
                r      = rng.uniform(r_min, r_max)

                d_start  = np.linalg.norm(center - start_xy)
                d_target = np.linalg.norm(center - target_xy)
                # 【修改】：计算与机械臂基座的距离，确保生成的障碍物不会和基座重叠
                d_base   = np.linalg.norm(center - base_xy)
                
                if (d_start >= (min_clearance + r) and 
                    d_target >= (min_clearance + r) and 
                    d_base >= (min_clearance + r + base_radius)):
                    obstacles.append((float(center[0]), float(center[1]), float(r)))
                    break
            else:
                # 兜底：500 次均未找到合法位置，强行放入最后一个采样位置
                if center is not None:
                    obstacles.append((float(center[0]), float(center[1]), float(r)))

        # ======================================================================
        # 步骤 2：2D A* 路径规划
        # ======================================================================
        # 【修改】：将基座加入专门用于寻路的“规划障碍物”列表
        planning_obstacles = obstacles + [(0.0, 0.0, base_radius)]

        grid_res = scene_plan_config["planning_grid_res"]
        xs = [start_xy[0], target_xy[0]]
        ys = [start_xy[1], target_xy[1]]
        
        # 【修改】：使用 planning_obstacles 计算网格边界
        for (ox, oy, r) in planning_obstacles:
            r_eff = r + min_clearance
            xs.extend([ox - r_eff, ox + r_eff])
            ys.extend([oy - r_eff, oy + r_eff])

        x_min = min(xs) - scene_plan_config["bounds_margin"]
        x_max = max(xs) + scene_plan_config["bounds_margin"]
        y_min = min(ys) - scene_plan_config["bounds_margin"]
        y_max = max(ys) + scene_plan_config["bounds_margin"]
        nx = max(2, int(np.ceil((x_max - x_min) / grid_res)))
        ny = max(2, int(np.ceil((y_max - y_min) / grid_res)))

        def world_to_grid(x, y):
            i = max(0, min(nx - 1, int((x - x_min) / grid_res)))
            j = max(0, min(ny - 1, int((y - y_min) / grid_res)))
            return i, j

        def grid_to_world(i, j):
            return x_min + (i + 0.5) * grid_res, y_min + (j + 0.5) * grid_res

        # 构建占据栅格
        occ = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                wx, wy = grid_to_world(i, j)
                # 【修改】：使用 planning_obstacles 进行碰撞检测
                for (ox, oy, r) in planning_obstacles:
                    if (wx - ox) ** 2 + (wy - oy) ** 2 < (r + min_clearance) ** 2:
                        occ[i, j] = True
                        break

        def find_nearest_free(i0, j0, search_radius=5):
            if not occ[i0, j0]:
                return i0, j0
            best, best_d2 = None, None
            for di in range(-search_radius, search_radius + 1):
                for dj in range(-search_radius, search_radius + 1):
                    ni, nj = i0 + di, j0 + dj
                    if 0 <= ni < nx and 0 <= nj < ny and not occ[ni, nj]:
                        d2 = di * di + dj * dj
                        if best is None or d2 < best_d2:
                            best, best_d2 = (ni, nj), d2
            return best

        start_ij = find_nearest_free(*world_to_grid(*start_xy)) or world_to_grid(*start_xy)
        goal_ij  = find_nearest_free(*world_to_grid(*target_xy)) or world_to_grid(*target_xy)

        # A* 搜索
        open_heap = []
        g_cost    = {start_ij: 0.0}
        parent    = {}
        import heapq
        heapq.heappush(open_heap, (
            float(np.hypot(grid_to_world(*start_ij)[0] - target_xy[0],
                           grid_to_world(*start_ij)[1] - target_xy[1])),
            start_ij
        ))
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        closed    = set()
        found     = False
        expansions = 0

        while open_heap and expansions < scene_plan_config["max_expansions"]:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_ij:
                found = True
                break
            closed.add(current)
            expansions += 1
            for di, dj in neighbors:
                ni, nj = current[0] + di, current[1] + dj
                if not (0 <= ni < nx and 0 <= nj < ny):
                    continue
                if occ[ni, nj]:
                    continue
                step_cost = grid_res if (di == 0 or dj == 0) else grid_res * 1.414
                new_g     = g_cost[current] + step_cost
                neighbor  = (ni, nj)
                if neighbor not in g_cost or new_g < g_cost[neighbor]:
                    g_cost[neighbor]  = new_g
                    parent[neighbor]  = current
                    h = float(np.hypot(grid_to_world(ni, nj)[0] - target_xy[0],
                                       grid_to_world(ni, nj)[1] - target_xy[1]))
                    heapq.heappush(open_heap, (new_g + h, neighbor))

        if not found:
            # A* 失败时回退到直线
            path_2d = np.vstack([start_xy, target_xy])
        else:
            path_idx = []
            node     = goal_ij
            while node != start_ij:
                path_idx.append(node)
                node = parent.get(node)
                if node is None:
                    break
            path_idx.append(start_ij)
            path_idx.reverse()
            path_2d = np.array([grid_to_world(i, j) for (i, j) in path_idx], dtype=float)

        # ======================================================================
        # 步骤 3：生成真实 3D 轨迹
        # ======================================================================
        z_cruise = scene_plan_config["payload_z_cruise"]
        path_3d  = [[pt[0], pt[1], z_cruise] for pt in path_2d]

        # 终点正上方垂直下降段
        last_xy    = path_2d[-1]
        descent_zs = np.linspace(
            z_cruise,
            scene_plan_config["target_z_descent"],
            scene_plan_config["num_descent_steps"] + 1
        )[1:]
        for z in descent_zs:
            path_3d.append([float(last_xy[0]), float(last_xy[1]), float(z)])

        path_3d = np.array(path_3d)

        # ======================================================================
        # 步骤 4：组装 XML（注入障碍物几何体、轨迹球、起终点标记）
        # ======================================================================
        xml = base_xml_content

        if obstacles:
            xml = xml.replace(
                '  </asset>',
                '    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>\n'
                '  </asset>',
                1
            )

        # 【注】：这里渲染 XML 时仍然使用原本的 obstacles 列表，不包含机械臂基座
        obs_z   = scene_plan_config["obstacle_z_center"]
        obs_hh  = scene_plan_config["obstacle_halfheight"]
        obstacle_bodies = "".join([
            f'    <body name="obstacle_{i}" pos="{x} {y} {obs_z}">\n'
            f'      <geom type="cylinder" size="{r} {obs_hh}" pos="0 0 0" '
            f'material="obstacle" contype="1" conaffinity="1"/>\n'
            f'    </body>\n'
            for i, (x, y, r) in enumerate(obstacles)
        ])

        path_bodies = "".join([
            f'    <body name="path_pt_{i}" pos="{p[0]} {p[1]} {p[2]}">\n'
            f'      <geom type="sphere" size="0.01" rgba="0 0 1 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n'
            for i, p in enumerate(path_3d)
        ])

        endpoint_z = obs_z + obs_hh + scene_plan_config["endpoint_z_offset"]
        endpoint_bodies = (
            f'    <body name="path_start" pos="{start_xy[0]} {start_xy[1]} {endpoint_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n'
            f'    <body name="path_goal" pos="{target_xy[0]} {target_xy[1]} {endpoint_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n'
        )

        replacement = (
            '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
            + obstacle_bodies + path_bodies + endpoint_bodies + '    '
        )
        xml = xml.replace(
            '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    ',
            replacement, 1
        )

        xml = re.sub(
            r'<body name="target" pos="[^"]+">',
            f'<body name="target" pos="{target_xy[0]} {target_xy[1]} 0">',
            xml, count=1
        )

        return {
            "obstacles":   obstacles,
            "path_3d":     path_3d,
            "xml_content": xml,
        }
 
    # ==========================================================================
    # reset()
    # ==========================================================================
    def reset(self):
        rng = self._obstacle_rng
 
        # === 1. Task: start / target positions ================================
        noise = self.init_position_range
        start_x = self.default_start_xy[0] + rng.uniform(-noise, noise)
        start_y = self.default_start_xy[1] + rng.uniform(-noise, noise)
        start_xy = np.array([start_x, start_y])
        target_xy = self.default_target.copy()
        self.target_pos = target_xy
 
        # === 2. Scene: obstacles + 2D path + 3D trajectory + XML ==============
        scene_plan_cfg = {**self.config["scene"], **self.config["planning"]}
        scene_data = CableRobotEnvWithObstacles.generate_scene_and_trajectory(
            start_xy=start_xy, target_xy=target_xy,
            base_xml_content=self._base_xml_content,
            scene_plan_config=scene_plan_cfg, rng=rng,
        )
        self._obstacles = scene_data["obstacles"]
        self._planned_path = scene_data["path_3d"]
 
        # === 3. Reload MuJoCo model from generated XML ========================
        fd, path = tempfile.mkstemp(suffix=".xml", dir=self._assets_dir, prefix="obstacles_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(scene_data["xml_content"])
            self.model = mujoco.MjModel.from_xml_path(path)
            self.data = mujoco.MjData(self.model)
            self.model.opt.timestep = self.physics_dt
            self.sim_steps = int(self.dt / self.physics_dt)
            self._reresolve_ids()
            if self.render_mode:
                if self.viewer is not None:
                    try: self.viewer.close()
                    except Exception: pass
                kw = {}
                if hasattr(self, '_key_callback') and self._key_callback is not None:
                    kw['key_callback'] = self._key_callback
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data, **kw)
        finally:
            try:
                if os.path.exists(path): os.remove(path)
            except Exception: pass
 
        # === 4. Set qpos: all joints to a clean initial state =================
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
 
        for j in range(self.model.njnt):
            adr = self.model.jnt_qposadr[j]
            jtype = self.model.jnt_type[j]
            if jtype == mujoco.mjtJoint.mjJNT_BALL:
                self.data.qpos[adr:adr + 4] = [1, 0, 0, 0]
            elif jtype == mujoco.mjtJoint.mjJNT_FREE:
                self.data.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
 
        # Prefab free joint
        init_prefab = np.array(self.config["reset"]["init_qpos_prefab"], dtype=np.float64)
        init_prefab[0] = start_x
        init_prefab[1] = start_y
        self.data.qpos[-7:] = init_prefab
        prefab_z = init_prefab[2]
 
        # Arm joints: IK or default
        if self.config["reset"].get("ik_enabled", False):
            self.data.qpos[:7] = self._solve_ik(
                target_xy=np.array([start_x, start_y]), prefab_z=prefab_z)
        else:
            self.data.qpos[:7] = np.array(self.config["reset"]["init_qpos_arm"])
 
        # Target body position
        self.model.body_pos[self.target_body_id][:2] = target_xy
 
        # === 5. Sync derived quantities (Mocap 逻辑彻底移除) ==================
        ee_body_id = self.model.body("link7").id
        mujoco.mj_forward(self.model, self.data)
        
        # 获取当前机械臂末端的真实位置，作为 IK 控制的起点
        initial_ee_pos = self.data.xpos[ee_body_id].copy()
 
        # === 6. Warmup: let ropes settle, lock arm + prefab ===================
        arm_qpos_hold = self.data.qpos[:7].copy()
        prefab_qpos_hold = self.data.qpos[-7:].copy()
        
        # 【核心新增】将初始姿态立刻下发给电机，防止刚开始物理模拟时机械臂瘫软
        self.data.ctrl[:7] = arm_qpos_hold
 
        for _ in range(self.config["reset"]["warmup_steps"]):
            self.data.qpos[:7] = arm_qpos_hold
            self.data.qvel[:7] = 0.0
            self.data.qpos[-7:] = prefab_qpos_hold
            self.data.qvel[-6:] = 0.0
            mujoco.mj_step(self.model, self.data)
 
        self.data.qpos[:7] = arm_qpos_hold
        self.data.qvel[:7] = 0.0
        self.data.qpos[-7:] = prefab_qpos_hold
        self.data.qvel[-6:] = 0.0
        mujoco.mj_forward(self.model, self.data)
 
        # [ENV-BUG-A 修复] warmup 完成后重新读取 EE site 位置，
        # 确保虚拟控制目标与真实末端位置严格对齐（而非 warmup 前的旧值）
        initial_ee_pos = self.data.site_xpos[self.ee_site_id].copy()
 
        # === 7. Initialize tracking state =====================================
        self.current_step = 0
        # 将末端（warmup后真实）位置赋给虚拟控制目标
        self.current_mocap_pos = initial_ee_pos.copy()
        self.current_mocap_vel = np.zeros(3)
        self.current_mocap_euler = np.zeros(3)
        self.current_mocap_euler_vel = np.zeros(3)
        self.current_mocap_yaw = 0.0
        self.current_mocap_yaw_vel = 0.0
 
        self.action_queue.clear()
        for _ in range(self.latency_steps):
            self.action_queue.append(np.zeros(self.action_dim))
 
        self.current_wp_idx = 0
        self.reached_final = False
        self.last_dist = None
        self.last_wp_idx = -1
 
        # === 8. Debug output ==================================================
        ee_pos = self.data.site_xpos[self.ee_site_id]
        prefab_pos = self.data.body('prefab').xpos
        rope_len = self.config["rope"]["num_segments"] * self.config["rope"]["segment_length"]
        # print(f"[RESET] EE:     {np.round(ee_pos, 4)}")
        # print(f"[RESET] Prefab: {np.round(prefab_pos, 4)}")
        # print(f"[RESET] Target: {np.round(self.current_mocap_pos, 4)} (Virtual Target)")
        # print(f"[RESET] dz={ee_pos[2] - prefab_pos[2]:.4f}  dxy={np.linalg.norm(ee_pos[:2] - prefab_pos[:2]):.4f}  rope={rope_len:.4f}")
 
        # === 9. Viewer sync + return obs ======================================
        if self.render_mode and self.viewer is not None:
            self.viewer.sync()
 
        obs = self._get_obs()
        if self._planned_path is not None and len(self._planned_path) > 0:
            payload_z = self.data.body('prefab').xpos[2]
            current_pos = np.array([obs[4], obs[5], payload_z])
            self.last_dist = np.linalg.norm(current_pos - self._planned_path[0])
            self.last_wp_idx = 0
 
        return obs
 
    # ==========================================================================
    # step()
    # ==========================================================================
    def step(self, action):
        """
        物理推演与状态更新：
          - 动作延迟
          - 二阶运动学积分（与旧版完全对齐）
          - 地板夹紧（防穿地）
          - NaN 检测保护
          - 3D 航点追踪状态机
          - 奖励计算与终止判定
        返回格式（Gymnasium 兼容）：obs, reward, done, truncated, info
        """
        # ======================================================================
        # 1. 动作裁剪与延迟队列
        # ======================================================================
        action_high = np.array(self.config["space"]["action_space_high"])
        action      = np.clip(action, -action_high, action_high)
 
        # 压入当前动作，弹出延迟后的有效动作
        self.action_queue.append(action)
        effective_action = self.action_queue.popleft()
 
        # 解析 4D/6D 动作 (保留原代码的重复解析，防止遗漏)
        a_xyz   = effective_action[:3]    # 平移加速度 [ax, ay, az]
        a_euler = effective_action[3:6]   # [a_roll, a_pitch, a_yaw]
 
        # ======================================================================
        # 2. 运动学积分（显式 Euler + 二阶修正）
        # ======================================================================
        # 解析 6D 动作
        a_xyz = effective_action[:3]
        a_euler = effective_action[3:6] # [a_roll, a_pitch, a_yaw]
        
        dt = self.dt
        
        # (1) 平移部分积分：先用旧速度更新（带 0.5*a*dt² 修正项），再更新速度
        self.current_mocap_pos += self.current_mocap_vel * dt + 0.5 * a_xyz * dt ** 2
        self.current_mocap_vel += a_xyz * dt
        
        # (2) 【修复】旋转部分积分：与平移部分保持严格相同的二阶 Euler，并彻底弃用 alpha_z
        self.current_mocap_euler += self.current_mocap_euler_vel * dt + 0.5 * a_euler * dt ** 2
        self.current_mocap_euler_vel += a_euler * dt
 
        # [ENV-BUG-C 修复] 地板夹紧阈值从 0.4m 降为 0.25m，
        # 防止在末端正常下降过程中（目标 Z ≈ 0.70m）意外触发夹紧产生速度突变。
        # 0.25m 以下 EE 必然与地面冲突，此时截断是合理的。
        if self.current_mocap_pos[2] < 0.25:
            self.current_mocap_pos[2] = 0.25
            self.current_mocap_vel[2] = 0.0
            
 
        # ======================================================================
        # 3. 注入 MuJoCo 并执行物理步进 (🔥🔥🔥 已使用 IK Solver 升级 🔥🔥🔥)
        # ======================================================================
        # 剥离旧版的 Mocap 直接修改（Scipy 四元数乘法等逻辑已被移除）
        
        # 3.1 获取当前机械臂的 7 维真实关节角度（作为 IK 迭代起点保证平滑连续）
        current_q = self.data.qpos[:7].copy()
        
        # 3.2 调用 IK Solver，将积分得到的 4D 目标位姿转换为 7 个关节角度
        # (注意: 这里传入 euler 的第3维即 Yaw 作为偏航角目标)
        target_q = self.ik_solver.solve_4d(
            current_q=current_q,
            target_x=self.current_mocap_pos[0],
            target_y=self.current_mocap_pos[1],
            target_z=self.current_mocap_pos[2],
            target_yaw=self.current_mocap_euler[2] 
        )
        
        # 3.3 将目标关节角下发给 MuJoCo 的位置执行器 (Actuators)
        # 前提是你的 XML 已经把 mocap 换成了 7 个 position actuators
        self.data.ctrl[:7] = target_q
        
        # 循环步进物理引擎 (保持你原有的逻辑，用电机平滑追踪)
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
 
        # [BUG-7 修复] NaN 检测：物理爆炸时立即终止并施加大惩罚
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            obs = self._get_obs()
            return obs, -10.0, True, False, {"is_success": False, "nan_detected": True}
 
        if self.render_mode and self.viewer is not None:
            self.viewer.sync()
 
        # ======================================================================
        # 4. 获取观测与负载真实 3D 坐标
        # ======================================================================
        obs = self._get_obs()
        # xyc: 重新确定prefab的z-value
        payload_z           = self.data.body('prefab').xpos[2]
        current_payload_pos = np.array([obs[4], obs[5], payload_z])
 
        # ======================================================================
        # 5. 航点追踪状态机
        # ======================================================================
        if self._planned_path is not None and not self.reached_final:
            target_wp   = self._planned_path[self.current_wp_idx]
            dist_to_wp  = np.linalg.norm(current_payload_pos - target_wp)
            
 
            # [BUG-9 修复] 切换航点时清空 last_dist，避免跨航点产生虚假进展奖励
            if self.last_wp_idx != self.current_wp_idx:
                self.last_dist   = None
                self.last_wp_idx = self.current_wp_idx
 
            # 保存当前距离供 _compute_reward 使用
            self.last_dist = dist_to_wp
 
            # [BUG-10 修复] 自适应 look_ahead：
            # 接近终点（剩余 ≤ 2 个航点）时用更小阈值 (0.06)，防止过早触发 reached_final
            total_wps  = len(self._planned_path)
            rem_wps    = total_wps - 1 - self.current_wp_idx
            look_ahead = 0.04 if rem_wps <= 2 else self.config["step_logic"]["look_ahead_dist"]
 
            if dist_to_wp < look_ahead:
                if self.current_wp_idx < total_wps - 1:
                    self.current_wp_idx += 1
                    self._wp_just_advanced = True
                else:
                    self.reached_final = True
 
        # ======================================================================
        # 6. 奖励计算与终止判定
        # ======================================================================
        reward, done, success = self._compute_reward(effective_action)
        # print(reward) # zxy
 
        # [BUG-8 修复] current_step 在步末递增，与旧版对齐
        self.current_step += 1
 
        # 超时判定（在递增之后比较，保证与旧版 >= max_steps 语义一致）
        if self.current_step >= self.config["sim"]["max_steps"]:
            if not done:
                # [BUG-12 修复] 超时未成功给予惩罚（与旧版 -5.0 对齐）
                reward += self.config["reward"]["timeout_penalty"]
            done = True
 
        info = {
            "is_success":      success,
            "current_wp_idx":  self.current_wp_idx,
            "reached_final":   self.reached_final,
        }
 
        # Gymnasium 标准格式：obs, reward, terminated, truncated, info
        return obs, reward, done, False, info
 
    # ==========================================================================
    # _compute_reward()
    # ==========================================================================
    # ==============================================================================
# reward_patch.py — mujoco_env_new.py 的 _compute_reward 替换补丁
#
# 使用方法：
#   将本文件中的 _compute_reward 方法粘贴替换掉 mujoco_env_new.py 中对应的方法。
#   无需修改其他代码。
#
# 奖励重设计说明（对应 config.py [CFG-3]）：
#
# 旧版问题分析：
#   - progress_coef=50, clip=±2 → 单步最大 progress ≈ ±2
#   - success_bonus=15
#   - step_penalty=-0.02, action_penalty≈-0.01 (per step)
#   → 一个 200 步回合：total shaping ≈ ±400，而 success 仅 +15
#   → progress 完全主导梯度，success 信号几乎被淹没
#   → agent 学会的策略：靠近航点最大化 progress，而不是"完成任务"
#
# 新版设计原则：
#   ① success_bonus = +10（基准）
#   ② 每步 shaping 量级 ≈ ±0.3（约为 success 的 3%/步）
#   ③ 中间航点里程碑 waypoint_bonus = +0.3（鼓励早期正向探索）
#   ④ swing_penalty：惩罚绳摆角（EE XY 与 payload XY 的偏差），
#      与 NMPC 防摆目标对齐，鼓励 RL 维持绳子竖直
#   ⑤ 终止惩罚统一降低到 -5（旧版 crash/collision/oob 均为 -8~-10），
#      保持与 success_bonus 的量级差距约 2×，避免 agent 过于保守
#
# 量级校验（200 步满血回合）：
#   最优轨迹（快速到达无碰撞）：+10（success）+ ~8×0.3（waypoints）- ~5（steps+action）
#                               ≈ +7.4
#   超时失败：-3（timeout）- ~10（steps+action） ≈ -13
#   碰撞失败：-5（collision）- ~3（steps）        ≈ -8
#   → success/timeout 量级比 ≈ 7.4 / 13 ≈ 0.57，信号均衡，success 仍是最优选择
# ==============================================================================


    def _compute_reward(self, action):
        """
        奖励计算与终止状态机（双重势能场 + 动态阶段里程碑版）。
        
        奖励组成：
        Per-step（每步）：
            + global_progress   ← 全局：迈向最终目标的势能奖励（兜底防迷路）
            + local_progress    ← 局部：迈向当前航点的势能奖励（引导避障）
            - step_penalty      ← 生存惩罚（鼓励尽快完成）
            - action_penalty    ← 动作 L2 正则（抑制抖动）
            - velocity_penalty  ← 速度过载惩罚（防甩动）
            - swing_penalty     ← 绳摆角惩罚（EE 与 payload XY 偏差）
            
        Milestone（航点切换时）：
            + waypoint_bonus    ← 基础过点奖励
            + stage_bonus       ← 【新增】基于 (当前进度/总航点数) 的动态阶段性奖励
            
        Terminal（终止时）：
            + success_bonus     ← 成功降落
            - timeout_penalty   ← 超时 (在 step 中计算)
            - collision_penalty ← 碰撞障碍物
            - crash_penalty     ← 坠毁（砸地/甩机）
            - out_of_bounds     ← 出界
        """
        reward  = 0.0
        done    = False
        success = False

        cfg_rwd   = self.config["reward"]
        cfg_logic = self.config["step_logic"]

        # ======================================================================
        # 1. 获取物理状态
        # ======================================================================
        obs = self._get_obs()
        payload_xy  = obs[4:6]
        payload_vxy = obs[6:8]

        payload_z  = self.data.body('prefab').xpos[2]
        dof_idx    = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vz = self.data.qvel[dof_idx + 2]

        current_payload_pos = np.array([payload_xy[0], payload_xy[1], payload_z])
        payload_vel_norm    = np.linalg.norm(np.append(payload_vxy, payload_vz))

        ee_xy = self.current_mocap_pos[:2]

        '''# ======================================================================
        # 2. 连续性惩罚（每步）
        # ======================================================================
        reward += cfg_rwd.get("step_penalty", -0.005)
        reward += cfg_rwd.get("action_smooth_penalty", -0.001) * float(np.sum(np.square(action)))

        vel_pen = cfg_rwd.get("velocity_penalty_coef", 0.005) * payload_vel_norm
        reward -= float(np.clip(vel_pen, 0.0, 0.5))

        swing_offset = float(np.linalg.norm(ee_xy - payload_xy))
        reward -= cfg_rwd.get("swing_penalty_coef", 0.02) * float(np.clip(swing_offset, 0.0, 0.1))'''


        # ======================================================================
        # 4. 终止条件判定（优先级：成功 > 碰撞 > 砸地 > 出界）
        # ======================================================================
        if self.reached_final:
            vel_xy        = float(np.linalg.norm(payload_vxy))
            dist_to_final = float(np.linalg.norm(payload_xy - self.target_pos))

            if dist_to_final < 0.03 and vel_xy < 0.1 and abs(payload_vz) < 0.2:
                reward  += cfg_rwd.get("success_bonus", 1.0)
                success  = True
                done     = True
                return reward, done, success
            else:
                reward += cfg_rwd.get("crash_penalty", -1.0)
                done    = True
                return reward, done, success

        '''payload_radius = self.config["planning"]["payload_radius"]
        for (ox, oy, orad) in self._obstacles:
            dist_to_obs = float(np.linalg.norm(payload_xy - np.array([ox, oy])))
            if dist_to_obs < (orad + payload_radius):
                reward += cfg_rwd.get("collision_penalty", -10.0)
                done    = True
                return reward, done, success

        if (payload_z < cfg_logic["crash_z_threshold"] and
                payload_vz < cfg_logic["crash_vz_threshold"]):
            reward += cfg_rwd.get("crash_penalty", -10.0)
            done    = True
            return reward, done, success

        dist_to_target_xy = float(np.linalg.norm(payload_xy - self.target_pos))
        if dist_to_target_xy > cfg_logic["out_of_bounds_dist"]:
            reward += cfg_rwd.get("out_of_bounds_penalty", -10.0)
            done    = True
            return reward, done, success'''

        # ======================================================================
        # 5. 【修改核心】航点里程碑奖励（基础保留 + 动态比例阶段叠加）
        # ======================================================================
        if getattr(self, '_wp_just_advanced', True):
            # 5.1 基础航点奖励 (可选)
            reward += cfg_rwd.get("waypoint_bonus", 0.05)
            
            '''# 5.2 均分总奖池作为过点奖励
            if self._planned_path is not None:
                total_wps = len(self._planned_path)
                total_stage_reward = cfg_rwd.get("total_stage_reward", 1.0)
                
                stage_bonus = total_stage_reward / max(1, total_wps)
                reward += stage_bonus'''

            self._wp_just_advanced = False

        return reward, done, success
 
    # ==========================================================================
    # _get_obs()
    # ==========================================================================
    def _get_obs(self):
        """
        观测空间整合（末尾 12 维保持固定负索引布局，不受障碍物数量影响）：
 
        布局：
          [0-1]   mocap_x, mocap_y               ← 固定
          [2-3]   mocap_vx, mocap_vy             ← 固定
          [4-5]   payload_x, payload_y           ← 固定，与旧版 obs[4]/obs[5] 对齐
          [6-7]   payload_vx, payload_vy         ← 固定，与旧版 obs[6]/obs[7] 对齐
          [8-9]   rel_tx, rel_ty                 ← 固定，与旧版 obs[8]/obs[9] 对齐
          [10 ~ 10+3*n-1]  障碍物 (ox, oy, r) * n_obstacles  ← 可变长度
          ── 末尾 12 维（负索引），不受 n_obstacles 影响 ──
          [-12] mocap_z     [-11] mocap_vz
          [-10] payload_z   [-9]  payload_vz
          [-8]  mocap_roll  [-7]  mocap_roll_vel
          [-6]  mocap_pitch [-5]  mocap_pitch_vel
          [-4]  mocap_yaw   [-3]  mocap_yaw_vel
          [-2]  payload_yaw [-1]  payload_yaw_vel
 
        ⚠️ 控制器读取末尾物理状态时必须用负索引，正向索引会被障碍物数据错位偏移！
        [BUG-4 修复] 负载 XY/速度使用真实物理值（xpos/qvel），不依赖 obs 索引自取。
        """
        # ------------------------------------------------------------------
        # 动捕点平移状态（来自内部追踪变量）
        # ------------------------------------------------------------------
        mocap_x,  mocap_y  = self.current_mocap_pos[0], self.current_mocap_pos[1]
        mocap_vx, mocap_vy = self.current_mocap_vel[0], self.current_mocap_vel[1]
        mocap_z            = self.current_mocap_pos[2]
        mocap_vz           = self.current_mocap_vel[2]
 
        # ------------------------------------------------------------------
        # 负载真实物理状态（xyc 保留的核心机制）
        # ------------------------------------------------------------------
        # xyc: 重新确定prefab的x/y-value
        payload_x = self.data.body('prefab').xpos[0]
        payload_y = self.data.body('prefab').xpos[1]
        # xyc: 重新确定prefab的z-value
        payload_z = self.data.body('prefab').xpos[2]
 
        dof_idx    = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vx = self.data.qvel[dof_idx]       # 与旧版一致
        payload_vy = self.data.qvel[dof_idx + 1]   # 与旧版一致
        payload_vz = self.data.qvel[dof_idx + 2]   # xyc 钩子
 
        # ------------------------------------------------------------------
        # 目标相对距离（与旧版 obs[8:10] = target_pos - payload_xy 对齐）
        # ------------------------------------------------------------------
        rel_tx = self.target_pos[0] - payload_x
        rel_ty = self.target_pos[1] - payload_y
 
        # ------------------------------------------------------------------
        # 障碍物信息（固定长度，不足时补零）
        # ------------------------------------------------------------------
        obs_data = []
        for (ox, oy, r) in self._obstacles:
            obs_data.extend([ox, oy, r])
        target_len = self.n_obstacles * 3
        while len(obs_data) < target_len:
            obs_data.append(0.0)
 
        # ------------------------------------------------------------------
        # 旋转状态（提取完整的 Roll, Pitch, Yaw 及其速度）
        # ------------------------------------------------------------------
        mocap_roll      = self.current_mocap_euler[0]
        mocap_pitch     = self.current_mocap_euler[1]
        mocap_yaw       = self.current_mocap_euler[2]
        
        mocap_roll_vel  = self.current_mocap_euler_vel[0]
        mocap_pitch_vel = self.current_mocap_euler_vel[1]
        mocap_yaw_vel   = self.current_mocap_euler_vel[2]
        
        # 负载姿态 (目前为 0 占位，如需更精确的防摆可从 self.data.qpos 中解算)
        payload_yaw     = 0.0  
        payload_yaw_vel = 0.0  
 
        # ------------------------------------------------------------------
        # 拼接（末尾 12 维通过负索引访问，布局与 NMPC 控制器约定严格对齐）
        # 负索引映射：
        #   [-12]=mocap_z [-11]=mocap_vz [-10]=payload_z [-9]=payload_vz
        #   [-8]=mocap_roll [-7]=mocap_roll_vel [-6]=mocap_pitch [-5]=mocap_pitch_vel
        #   [-4]=mocap_yaw  [-3]=mocap_yaw_vel  [-2]=payload_yaw [-1]=payload_yaw_vel
        # ------------------------------------------------------------------
        return np.array(
            [mocap_x, mocap_y, mocap_vx, mocap_vy,
             payload_x, payload_y, payload_vx, payload_vy,
             rel_tx, rel_ty]
            + obs_data[:target_len]
            + [
                mocap_z, mocap_vz, payload_z, payload_vz,                      # [-12] ~ [-9]
                mocap_roll, mocap_roll_vel, mocap_pitch, mocap_pitch_vel,       # [-8]  ~ [-5]
                mocap_yaw, mocap_yaw_vel, payload_yaw, payload_yaw_vel         # [-4]  ~ [-1]
              ],
            dtype=np.float32
        )
 
    # ==========================================================================
    # IK 求解：使末端在指定位置上方、垂直朝下
    # ==========================================================================
    def _solve_ik(self, target_xy, prefab_z):
        """
        雅可比伪逆 IK：求解机械臂 7 关节角，使 attachment_site 到达
        (target_xy[0], target_xy[1], prefab_z + height_above)，姿态垂直朝下。
 
        返回: np.ndarray (7,) 关节角；若未收敛则返回 config 中的默认值。
        """
        cfg_ik = self.config["reset"]
        cfg_rope = self.config["rope"]
        # IK 高度差 = 绳长 + hook_attachment 偏移 (0.045m)
        # 实际链: EE → hook_attachment(0.045m) → rope → prefab
        rope_length = cfg_rope["num_segments"] * cfg_rope["segment_length"]
        hook_offset_z = 0.045  # hook_attachment body pos in link7 local Z
        height = rope_length + hook_offset_z
        tgt_quat = np.array(cfg_ik["ik_target_quat"], dtype=np.float64)
        max_iter = cfg_ik["ik_max_iter"]
        tol_pos  = cfg_ik["ik_tol_pos"]
        tol_rot  = cfg_ik["ik_tol_rot"]
 
        target_pos = np.array([target_xy[0], target_xy[1], prefab_z + height])
 
        # 在独立的 data 副本上求解，避免污染主仿真状态
        data_ik = mujoco.MjData(self.model)
        n_joints = 7
        site_id = self.ee_site_id
 
        # 初始猜测：优先使用上次 IK 成功结果（热启动），否则用默认值
        if hasattr(self, '_last_ik_qpos') and self._last_ik_qpos is not None:
            data_ik.qpos[:n_joints] = self._last_ik_qpos.copy()
        else:
            data_ik.qpos[:n_joints] = np.array(
                self.config["reset"]["init_qpos_arm"], dtype=np.float64
            )
 
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        step_pos, step_rot = 0.5, 0.3
 
        converged = False
        for i in range(max_iter):
            mujoco.mj_forward(self.model, data_ik)
 
            pos_err = target_pos - data_ik.site_xpos[site_id]
            # 姿态误差：四元数差 → 角轴
            q_cur = np.zeros(4)
            mujoco.mju_mat2Quat(q_cur, data_ik.site_xmat[site_id])
            q_cur_inv = q_cur.copy()
            q_cur_inv[1:] *= -1
            q_err = np.zeros(4)
            mujoco.mju_mulQuat(q_err, tgt_quat, q_cur_inv)
            if q_err[0] < 0:
                q_err *= -1
            rot_err = 2.0 * q_err[1:]
 
            if np.linalg.norm(pos_err) < tol_pos and np.linalg.norm(rot_err) < tol_rot:
                converged = True
                break
 
            mujoco.mj_jacSite(self.model, data_ik, jacp, jacr, site_id)
            Jp = jacp[:, :n_joints]
            Jr = jacr[:, :n_joints]
            dq = step_pos * Jp.T @ pos_err + step_rot * Jr.T @ rot_err
            data_ik.qpos[:n_joints] += dq
 
            # 关节限位裁剪
            for j in range(n_joints):
                lo, hi = self.model.jnt_range[j]
                if lo < hi:
                    data_ik.qpos[j] = np.clip(data_ik.qpos[j], lo, hi)
 
        if not converged:
            print(f"[IK] NOT converged (iter={max_iter}), using default qpos")
            return np.array(self.config["reset"]["init_qpos_arm"], dtype=np.float64)
        result = data_ik.qpos[:n_joints].copy()
        self._last_ik_qpos = result
        # print(f"[IK] Converged! qpos={np.round(result, 4)}")
        return result
 
    # ==========================================================================
    # 辅助调试工具（保留自旧版）
    # ==========================================================================
    def _print_qpos_name(self):
        """遍历所有关节，打印 qpos 索引与关节名称的对应关系（调试用）。"""
        for idx in range(self.model.njnt):
            jnt_name  = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, idx)
            qpos_idx  = self.model.jnt_qposadr[idx]
            jnt_type  = self.model.jnt_type[idx]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                print(f"qpos[{qpos_idx}~{qpos_idx+6}] = {jnt_name} (free, 7维)")
            elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                print(f"qpos[{qpos_idx}~{qpos_idx+3}] = {jnt_name} (ball, 4维)")
            else:
                print(f"qpos[{qpos_idx}] = {jnt_name} (1维)")
 
    def _print_state(self):
        """打印机械臂关键状态（调试用）。"""
        print(f"Robot state: {self.data.qpos[:7]}")
        print(f"MocapPos   : {self.data.mocap_pos}")
        print(f"EEPos      : {self.data.xpos[self.model.body('link7').id]}")
        print(f"EEQuat     : {self.data.xquat[self.model.body('link7').id]}")
 
    def get_obstacles(self):
        """返回当前回合的障碍物列表（供外部可视化使用）。"""
        return list(self._obstacles)
 
    def get_planned_path(self):
        """返回当前回合的 3D 规划路径（供外部可视化使用）。"""
        if self._planned_path is None:
            return None
        return np.array(self._planned_path, copy=True)
 
 
class MPCPinocchioIKSolver:
    def __init__(self, model_roboplan, data_roboplan, collision_model=None):
        """
        初始化 IK 求解器
        :param model_roboplan: 通过 pyroboplan.models 载入的 KUKA iiwa14 模型
        :param data_roboplan: 对应的 data
        """
        self.model = model_roboplan
        self.data = data_roboplan
        self.collision_model = collision_model
        
        # 末端执行器的名字（请根据你 KUKA XML 中的名字修改，通常是 link7 所在的 frame）
        self.target_frame = "link7"  
        
        # ---------------------------------------------------------
        # 关键优化 1：针对 MPC 实时追踪的 IK 参数配置
        # ---------------------------------------------------------
        options = DifferentialIkOptions(
            max_iters=100,         # MPC 连续追踪不需要像 demo 里的 200 次，50 次足够
            max_retries=0,        # 实时控制中禁止重试跳跃，必须保持连续性
            damping=1e-2,         # 阻尼项 (DLS)，防止奇异点引发大幅度跳动
            min_step_size=0.01,
            max_step_size=0.2,
            ignore_joint_indices=[], # 如果你有夹爪，把夹爪的 index 放在这里忽略掉
            rng_seed=None,
        )
        
        self.ik = DifferentialIk(
            self.model,
            data=self.data,
            collision_model=self.collision_model,
            options=options,
            visualizer=None,
        )
        
        # ---------------------------------------------------------
        # 关键优化 2：零空间投影 (解决 7 轴冗余)
        # ---------------------------------------------------------
        # 我们保留学长 demo 中的关节限位保护，防止机械臂扭死。
        # 如果你的 MPC 速度要求极高，可以暂时去掉避障的 nullspace component
        self.nullspace_components = [
            lambda m, q: joint_limit_nullspace_component(m, q, gain=1.0, padding=0.025)
        ]
 
    def solve_4d(self, current_q, target_x, target_y, target_z, target_yaw):
        """
        根据 MPC 给出的 4D 目标，计算 KUKA 的 7 关节角度
        :param current_q: 当前机械臂的 7 维真实关节角度（作为迭代起点保证平滑！）
        :param target_x, target_y, target_z: 目标平移
        :param target_yaw: 目标偏航角
        :return: 7 维 numpy array (目标关节角度)
        """
        # 1. 固定 Roll 和 Pitch (使机械臂末端垂直向下)
        # 注意：这里假设 np.pi 的 roll 能让你的末端朝下，请根据实际 URDF 坐标系调整
        roll = np.pi  
        pitch = 0.0
        yaw = target_yaw
 
        # 2. 构建旋转矩阵 (Z-Y-X 欧拉角转旋转矩阵)
        R_x = np.array([[1, 0, 0], 
                        [0, np.cos(roll), -np.sin(roll)], 
                        [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], 
                        [0, 1, 0], 
                        [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], 
                        [np.sin(yaw), np.cos(yaw), 0], 
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x
 
        # 3. 生成 Pinocchio 的 SE3 目标位姿 (等同于学长 demo 的 target_tform)
        target_tform = pinocchio.SE3(R, np.array([target_x, target_y, target_z]))
 
        # 4. 调用 Differential IK 进行求解
        # 【极其重要】：init_state 必须传入 current_q，这样 IK 才会只在前一帧的基础上微调
        q_sol = self.ik.solve(
            self.target_frame,
            target_tform,
            init_state=current_q,
            nullspace_components=self.nullspace_components,
            verbose=False # 关闭打印以防刷屏
        )
 
        if q_sol is None:
            # 计算当前 EE 的实际位置
            pinocchio.forwardKinematics(self.model, self.data, current_q)
            pinocchio.updateFramePlacements(self.model, self.data)
            curr_ee = self.data.oMf[self.model.getFrameId(self.target_frame)].translation
            
            '''print(f"[IK 失败警告] 当前 EE 位置: {curr_ee}")
            print(f"[IK 失败警告] 试图求解的死点目标: X={target_x:.3f}, Y={target_y:.3f}, Z={target_z:.3f}")
            print(f"[IK 失败警告] 欧氏距离跳变: {np.linalg.norm(np.array([target_x, target_y, target_z]) - curr_ee):.4f}m")'''
            return current_q
 
        # 如果你的模型包含夹爪等额外自由度，确保只返回前 7 个 arm 关节
        return q_sol[:7]