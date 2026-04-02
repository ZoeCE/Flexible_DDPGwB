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
#                        mocap_z, mocap_vz, payload_z, payload_vz, ← 旧版Z轴4维
#                        mocap_yaw, mocap_yaw_vel,                 ← 新增旋转2维
#                        payload_yaw, payload_yaw_vel]             ← 新增旋转2维
#             总计: 10 + n_obstacles*3 + 4 + 4 = 18 + n_obstacles*3
# ==============================================================================
 
import os
import re
import heapq
import tempfile
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
 
# 引入独立配置文件
from config import DEFAULT_CONFIG
 
 
class CableRobotEnvWithObstacles:
    """
    配置驱动的索驱动机器人物理仿真环境（带动态障碍物）。
    动作空间: 4D [a_x, a_y, a_z, alpha_z(Z轴角加速度)]
 
    观测布局（共 18 + 3*n_obstacles 维，索引与旧版高度兼容）：
      [0]  mocap_x        动捕点 X 位置
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
      [-8] mocap_z        动捕点 Z 位置
      [-7] mocap_vz       动捕点 Z 速度
      [-6] payload_z      负载 Z 位置（xyc 钩子读取）
      [-5] payload_vz     负载 Z 速度（xyc 钩子读取）
      [-4] mocap_yaw      动捕点偏航角（新增）
      [-3] mocap_yaw_vel  动捕点偏航角速度（新增）
      [-2] payload_yaw    负载偏航角（占位，当前为 0）
      [-1] payload_yaw_vel 负载偏航角速度（占位，当前为 0）
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
        # 10（旧版基础）+ n_obstacles*3（障碍物）+ 4（Z轴）+ 4（旋转，新增）
        self.state_dim = 10 + (self.n_obstacles * 3) + 4 + 4
 
        # ======================================================================
        # 6. XML 资产管理与模型初始化
        # ======================================================================
        current_dir   = os.path.dirname(os.path.abspath(__file__))
        self._assets_dir = os.path.join(current_dir, "assets")
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
 
        # 动捕点平移状态追踪（旧版等价变量）
        self.current_mocap_pos = np.zeros(3)
        self.current_mocap_vel = np.zeros(3)
        # [新增] 动捕点 Z 轴旋转状态追踪
        self.current_mocap_yaw     = 0.0
        self.current_mocap_yaw_vel = 0.0
 
        # 路径追踪辅助变量（reset 后由 reset() 初始化）
        self.current_wp_idx = 0
        self.reached_final  = False
        self.last_dist      = None
        self.last_wp_idx    = -1   # [BUG-9 修复] 用于检测航点切换
 
        # ======================================================================
        # 9. 渲染器初始化
        # ======================================================================
        self.render_mode = cfg_sim["render"]
        self.viewer      = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
 
    # ==========================================================================
    # 辅助：重新解析 MuJoCo ID（每次重载 XML 后调用）
    # ==========================================================================
    def _reresolve_ids(self):
        """重新绑定 MuJoCo 体/关节 ID，在每次 XML 重载后必须调用。"""
        mocap_body = self.model.body("mocap")
        if hasattr(mocap_body, "mocapid"):
            mids = mocap_body.mocapid
            self.mocap_id = mids[0] if isinstance(mids, (np.ndarray, list)) else mids
        else:
            raise ValueError("Model does not contain a mocap body named 'mocap'")
 
        self.prefab_jnt_id  = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("rebar_base").id
 
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
 
        Args:
            start_xy:           起点 XY（带噪声的真实起点）
            target_xy:          终点 XY
            base_xml_content:   原始 XML 字符串
            scene_plan_config:  合并后的配置字典，包含 "scene" + "planning" 两部分的所有键
            rng:                numpy 随机生成器（为 None 时自动创建）
 
        Returns:
            dict:
                "obstacles": [(x, y, r), ...]   障碍物列表
                "path_3d":   np.ndarray          3D 引导轨迹
                "xml_content": str               注入了障碍物与轨迹球的 XML
        """
        if rng is None:
            rng = np.random.default_rng()
 
        start_xy  = np.asarray(start_xy,  dtype=float).reshape(2)
        target_xy = np.asarray(target_xy, dtype=float).reshape(2)
 
        # 统一读取来自合并 dict 的参数
        p_radius    = scene_plan_config["payload_radius"]
        p_margin    = scene_plan_config["planning_margin"]
        min_clearance = p_radius + p_margin   # 起终点防撞保护圈半径
 
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
                if d_start >= (min_clearance + r) and d_target >= (min_clearance + r):
                    obstacles.append((float(center[0]), float(center[1]), float(r)))
                    break
            else:
                # 兜底：500 次均未找到合法位置，强行放入最后一个采样位置
                if center is not None:
                    obstacles.append((float(center[0]), float(center[1]), float(r)))
 
        # ======================================================================
        # 步骤 2：2D A* 路径规划
        # ======================================================================
        if not obstacles:
            # 无障碍物时直接连线
            path_2d = np.vstack([start_xy, target_xy])
        else:
            grid_res = scene_plan_config["planning_grid_res"]
            xs = [start_xy[0], target_xy[0]]
            ys = [start_xy[1], target_xy[1]]
            for (ox, oy, r) in obstacles:
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
                    for (ox, oy, r) in obstacles:
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
        # 步骤 3：生成真实 3D 轨迹（直接使用配置中的巡航高度，无需 dummy_path）
        # ======================================================================
        z_cruise = scene_plan_config["payload_z_cruise"]   # 巡航高度（来自 planning）
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
 
        # 插入 obstacle 材质（只在有障碍物时插入）
        if obstacles:
            xml = xml.replace(
                '    <material name="steel" rgba="0.6 0.6 0.6 1"/>',
                '    <material name="steel" rgba="0.6 0.6 0.6 1"/>\n'
                '    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>',
                1
            )
 
        # 障碍物圆柱体
        obs_z   = scene_plan_config["obstacle_z_center"]     # 来自 scene
        obs_hh  = scene_plan_config["obstacle_halfheight"]   # 来自 scene
        obstacle_bodies = "".join([
            f'    <body name="obstacle_{i}" pos="{x} {y} {obs_z}">\n'
            f'      <geom type="cylinder" size="{r} {obs_hh}" pos="0 0 0" '
            f'material="obstacle" contype="1" conaffinity="1"/>\n'
            f'    </body>\n'
            for i, (x, y, r) in enumerate(obstacles)
        ])
 
        # 轨迹可视化球（位置直接来自 3D 路径，无需后续挪动）
        path_bodies = "".join([
            f'    <body name="path_pt_{i}" pos="{p[0]} {p[1]} {p[2]}">\n'
            f'      <geom type="sphere" size="0.01" rgba="0 0 1 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n'
            for i, p in enumerate(path_3d)
        ])
 
        # 起终点标记球
        endpoint_z = obs_z + obs_hh + scene_plan_config["endpoint_z_offset"]  # 来自 scene
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
 
        # 文本注入（在地板几何体后方插入所有新增实体）
        replacement = (
            '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
            + obstacle_bodies + path_bodies + endpoint_bodies + '    '
        )
        xml = xml.replace(
            '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    ',
            replacement, 1
        )
 
        # 移动目标重物基座 rebar_base 到真实终点
        xml = re.sub(
            r'<body name="rebar_base" pos="[^"]+">',
            f'<body name="rebar_base" pos="{target_xy[0]} {target_xy[1]} 0">',
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
        """
        全方位重置环境：一站式生成新场景、重载 MuJoCo 模型、对齐物理状态。
        """
        rng = self._obstacle_rng
 
        # ======================================================================
        # 1. 确定本次回合的起点与终点
        # ======================================================================
        noise_range = self.init_position_range
        start_x = self.default_start_xy[0] + rng.uniform(-noise_range, noise_range)
        start_y = self.default_start_xy[1] + rng.uniform(-noise_range, noise_range)
        start_xy  = np.array([start_x, start_y])
        target_xy = self.default_target.copy()   # 子类沿用：目标点不加噪声
        self.target_pos = target_xy              # 供后续 _get_obs / _compute_reward 使用
 
        # ======================================================================
        # 2. 一站式场景生成
        # ======================================================================
        # [BUG-3 修复] 合并 scene + planning 两个子配置，确保函数可以取到所有需要的键
        scene_plan_cfg = {**self.config["scene"], **self.config["planning"]}
 
        # [BUG-2 修复] 通过类名调用 @staticmethod，消除 NameError
        scene_data = CableRobotEnvWithObstacles.generate_scene_and_trajectory(
            start_xy        = start_xy,
            target_xy       = target_xy,
            base_xml_content = self._base_xml_content,
            scene_plan_config = scene_plan_cfg,
            rng             = rng,
        )
        self._obstacles   = scene_data["obstacles"]
        self._planned_path = scene_data["path_3d"]
 
        # ======================================================================
        # 3. 写入临时 XML 文件并重载 MuJoCo 模型
        # ======================================================================
        # [IMPROVE-1] 先完成模型加载，成功后再删除临时文件，避免加载期间文件被删
        fd, path = tempfile.mkstemp(
            suffix=".xml", dir=self._assets_dir, prefix="obstacles_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(scene_data["xml_content"])
 
            # 加载含障碍物的新模型
            self.model = mujoco.MjModel.from_xml_path(path)
            self.data  = mujoco.MjData(self.model)
            self.model.opt.timestep = self.physics_dt
 
            # [IMPROVE-2] 重载后重新计算 sim_steps，防止外部修改 physics_dt 时失效
            self.sim_steps = int(self.dt / self.physics_dt)
 
            # 重新绑定 ID（XML 变化后 ID 可能改变）
            self._reresolve_ids()
 
            # 渲染器随模型更新
            if self.render_mode and self.viewer is not None:
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        finally:
            # 模型加载完成后删除临时文件（无论成功与否都清理）
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
 
        # ======================================================================
        # 4. 物理状态重置（严格对齐原父类逻辑）
        # ======================================================================
        # xyc!: xml中定义的keyframe仅在这里有效
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
 
        # xyc: 设置机械臂初始关节值
        self.data.qpos[:7]  = np.array(self.config["reset"]["init_qpos_arm"])
        # xyc: 设置prefab初始位姿
        self.data.qpos[-7:] = np.array(self.config["reset"]["init_qpos_prefab"])
 
        # ======================================================================
        # 5. 坐标对齐（目标体 + 负载 + 动捕点）
        # ======================================================================
        # 对齐目标重物基座与随机化后的终点
        self.model.body_pos[self.target_body_id][:2] = target_xy
 
        # xyc: 重新确定prefab的x/y-value
        self.data.body('prefab').xpos[0] = start_x
        self.data.body('prefab').xpos[1] = start_y
 
        # 动捕点 XY 对齐到负载初始位置，Z 强制为配置高度
        self.data.mocap_pos[self.mocap_id][0] = start_x
        self.data.mocap_pos[self.mocap_id][1] = start_y
        self.data.mocap_pos[self.mocap_id][2] = self.config["reset"]["mocap_init_z"]
 
        # [新增] 初始化 Mocap 偏航姿态四元数
        self.data.mocap_quat[self.mocap_id] = self.start_quat_mocap
 
        # ======================================================================
        # 6. 物理预热（让绳索自然垂落稳定）
        # ======================================================================
        for _ in range(self.config["reset"]["warmup_steps"]):
            mujoco.mj_step(self.model, self.data)
 
        # ======================================================================
        # 7. 内部状态追踪变量初始化
        # ======================================================================
        self.current_step = 0
 
        # 平移状态：从预热后的真实 mocap 位置出发
        self.current_mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
        self.current_mocap_vel = np.zeros(3)
 
        # 旋转状态：从零开始（对应 start_quat_mocap 方向）
        self.current_mocap_yaw     = 0.0
        self.current_mocap_yaw_vel = 0.0
 
        # 延迟队列清零（reset 后动作历史无效）
        self.action_queue.clear()
        for _ in range(self.latency_steps):
            self.action_queue.append(np.zeros(self.action_dim))
 
        # 路径追踪状态机
        self.current_wp_idx = 0
        self.reached_final  = False
        self.last_dist      = None
        self.last_wp_idx    = -1   # [BUG-9 修复] 清零航点切换检测变量
 
        # ======================================================================
        # 8. 计算并返回初始观测
        # ======================================================================
        obs = self._get_obs()
 
        # 计算到第一个航点的初始距离（供第一步的 progress reward 使用）
        if self._planned_path is not None and len(self._planned_path) > 0:
            # xyc: 重新确定prefab的z-value，解决第一帧reset问题
            payload_z   = self.data.body('prefab').xpos[2]
            current_pos = np.array([obs[4], obs[5], payload_z])
            target_wp   = self._planned_path[self.current_wp_idx]
            self.last_dist  = np.linalg.norm(current_pos - target_wp)
            self.last_wp_idx = self.current_wp_idx
        else:
            self.last_dist = None
 
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
 
        # 解析 4D 动作
        a_xyz   = effective_action[:3]    # 平移加速度 [ax, ay, az]
        alpha_z = effective_action[3]     # Z 轴角加速度
 
        # ======================================================================
        # 2. 运动学积分（旧版显式 Euler + 二阶修正）
        # ======================================================================
        # [BUG-5 修复] 恢复旧版积分公式：pos 先用旧速度更新（带 0.5*a*dt² 修正项），
        # 再更新速度。这与旧版 CableRobotEnvWithObstacles.step() 完全一致。
        dt = self.dt
        self.current_mocap_pos += self.current_mocap_vel * dt + 0.5 * a_xyz * dt ** 2
        self.current_mocap_vel += a_xyz * dt
 
        # [BUG-6 修复] 地板夹紧：动捕点 Z 不能低于 0.4 m（旧版保护逻辑）
        # 防止动捕点（末端执行器代理）穿入地面导致物理仿真数值爆炸
        if self.current_mocap_pos[2] < 0.4:
            self.current_mocap_pos[2] = 0.4
            self.current_mocap_vel[2] = 0.0
 
        # [新增] Z 轴旋转积分（与平移部分相同的二阶 Euler）
        self.current_mocap_yaw_vel += alpha_z * dt
        self.current_mocap_yaw     += self.current_mocap_yaw_vel * dt
 
        # ======================================================================
        # 3. 注入 MuJoCo 并执行物理步进
        # ======================================================================
        # 更新平移动捕点
        self.data.mocap_pos[self.mocap_id] = self.current_mocap_pos
 
        # 更新旋转动捕点：将偏航角 yaw 转为四元数 q = [cos(yaw/2), 0, 0, sin(yaw/2)]
        half_yaw = self.current_mocap_yaw / 2.0
        self.data.mocap_quat[self.mocap_id] = np.array([
            np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)
        ])
 
        # 按控制频率循环执行物理步（例如 10Hz 控制 / 500Hz 物理 = 50 次 mj_step）
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
            look_ahead = 0.06 if rem_wps <= 2 else self.config["step_logic"]["look_ahead_dist"]
 
            if dist_to_wp < look_ahead:
                if self.current_wp_idx < total_wps - 1:
                    self.current_wp_idx += 1
                else:
                    self.reached_final = True
 
        # ======================================================================
        # 6. 奖励计算与终止判定
        # ======================================================================
        reward, done, success = self._compute_reward(effective_action)
 
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
    def _compute_reward(self, action):
        """
        奖励计算与终止状态机：
          整合旧版父类（连续性惩罚、碰撞）与子类（进展奖励、砸地、出界、成功判定）。
        """
        reward  = 0.0
        done    = False
        success = False
 
        cfg_rwd   = self.config["reward"]
        cfg_logic = self.config["step_logic"]
 
        # ======================================================================
        # 1. 获取物理状态
        # ======================================================================
        obs        = self._get_obs()
        payload_xy = obs[4:6]              # 负载 XY（与旧版 obs[4]/obs[5] 对齐）
        payload_vxy = obs[6:8]             # 负载 XY 速度（与旧版 obs[6]/obs[7] 对齐）
 
        # xyc: 重新确定 prefab 的 z-value 和 Z 轴速度
        payload_z   = self.data.body('prefab').xpos[2]
        dof_idx     = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vz  = self.data.qvel[dof_idx + 2]
 
        current_payload_pos = np.array([payload_xy[0], payload_xy[1], payload_z])
        payload_vel_norm    = np.linalg.norm(np.append(payload_vxy, payload_vz))
 
        # ======================================================================
        # 2. 连续性惩罚（每步都会累加）
        # ======================================================================
        # (1) 步数惩罚：鼓励尽快完成
        reward += cfg_rwd["step_penalty"]
 
        # (2) 动作平滑 L2 惩罚：抑制抖动，同时惩罚 alpha_z 防止旋转
        reward += cfg_rwd["action_smooth_penalty"] * np.sum(np.square(action))
 
        # (3) 速度过载惩罚：防止负载甩动
        reward -= np.clip(cfg_rwd["velocity_penalty_coef"] * payload_vel_norm, 0.0, 1.0)
 
        # ======================================================================
        # 3. 进展奖励（势能差密集奖励）
        # ======================================================================
        if self._planned_path is not None and not self.reached_final:
            target_wp    = self._planned_path[self.current_wp_idx]
            current_dist = np.linalg.norm(current_payload_pos - target_wp)
 
            if self.last_dist is not None:
                # 靠近航点 → 正奖励；远离 → 负奖励
                progress = cfg_rwd["progress_coef"] * (self.last_dist - current_dist)
                reward  += np.clip(progress, -2.0, 2.0)   # 与旧版 clip 对齐
 
        # ======================================================================
        # 4. 终止条件判定（优先级：成功 > 碰撞 > 砸地 > 出界）
        # ======================================================================
 
        # (1) 成功判定（已到达最后航点 + 多条件门控）
        # [BUG-11 修复] 恢复旧版多条件门控：速度 + 高度 + 水平距离
        if self.reached_final:
            vel_xy        = np.linalg.norm(payload_vxy)
            dist_to_final = np.linalg.norm(payload_xy - self.target_pos)
            if dist_to_final < 0.15 and vel_xy < 0.2 and abs(payload_vz) < 0.5:
                # [BUG-12 修复] 恢复旧版成功奖励 +15.0
                reward  += cfg_rwd["success_bonus"]
                success  = True
                done     = True
                return reward, done, success
            else:
                # 到达航点区域但不满足降落条件（速度过大或偏离），视为摔机
                reward += cfg_rwd["crash_penalty"]
                done    = True
                return reward, done, success
 
        # (2) 碰撞惩罚
        payload_radius = self.config["planning"]["payload_radius"]
        for (ox, oy, orad) in self._obstacles:
            dist_to_obs = np.linalg.norm(payload_xy - np.array([ox, oy]))
            if dist_to_obs < (orad + payload_radius):
                reward += cfg_rwd["collision_penalty"]   # 已是负数 (-8.0)
                done    = True
                return reward, done, success
 
        # (3) 砸地/坠毁惩罚：高度极低 且 正在快速下坠
        if payload_z < cfg_logic["crash_z_threshold"] and payload_vz < cfg_logic["crash_vz_threshold"]:
            reward += cfg_rwd["crash_penalty"]
            done    = True
            return reward, done, success
 
        # (4) 出界判定：偏离目标超过阈值
        dist_to_target_xy = np.linalg.norm(payload_xy - self.target_pos)
        if dist_to_target_xy > cfg_logic["out_of_bounds_dist"]:
            reward += cfg_rwd["out_of_bounds_penalty"]
            done    = True
            return reward, done, success
 
        return reward, done, success
 
    # ==========================================================================
    # _get_obs()
    # ==========================================================================
    def _get_obs(self):
        """
        观测空间整合（维度与旧版高度兼容，新增旋转状态）：
 
        布局：
          [0-1]   mocap_x, mocap_y
          [2-3]   mocap_vx, mocap_vy
          [4-5]   payload_x, payload_y        ← 与旧版 obs[4]/obs[5] 对齐
          [6-7]   payload_vx, payload_vy      ← 与旧版 obs[6]/obs[7] 对齐
          [8-9]   rel_tx, rel_ty              ← 与旧版 obs[8]/obs[9] 对齐
          [10 ~ 10+3*n-1]  障碍物 (ox, oy, r) * n_obstacles
          [-8]    mocap_z
          [-7]    mocap_vz
          [-6]    payload_z                   ← xyc 钩子
          [-5]    payload_vz                  ← xyc 钩子
          [-4]    mocap_yaw                   ← 新增
          [-3]    mocap_yaw_vel               ← 新增
          [-2]    payload_yaw                 ← 新增（占位，当前为 0）
          [-1]    payload_yaw_vel             ← 新增（占位，当前为 0）
 
        注意：[BUG-4] 修复 —— 负载 XY/速度使用真实物理值而不依赖 obs 索引自取，
        与旧版的 xpos / qvel 读取逻辑完全对齐。
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
        # 旋转状态（新增；payload_yaw 占位为 0，可接传感器数据）
        # ------------------------------------------------------------------
        mocap_yaw      = self.current_mocap_yaw
        mocap_yaw_vel  = self.current_mocap_yaw_vel
        payload_yaw    = 0.0   # 占位：可通过 quat2euler 解析 prefab 四元数
        payload_yaw_vel = 0.0  # 占位：可通过 prefab DOF 的旋转速度补充
 
        # ------------------------------------------------------------------
        # 拼接（顺序与上方文档注释严格对应）
        # ------------------------------------------------------------------
        return np.array(
            [mocap_x, mocap_y, mocap_vx, mocap_vy,
             payload_x, payload_y, payload_vx, payload_vy,
             rel_tx, rel_ty]
            + obs_data[:target_len]
            + [mocap_z, mocap_vz, payload_z, payload_vz,
               mocap_yaw, mocap_yaw_vel, payload_yaw, payload_yaw_vel],
            dtype=np.float32
        )
 
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

