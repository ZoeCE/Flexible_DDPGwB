import mujoco
import mujoco.viewer
import numpy as np
import os
import re
import tempfile
import heapq
from collections import deque

class CableRobotEnv:
    def __init__(self, render=False):
        # --- 1. 加载模型 ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "assets2/demo_fourCable_withSteel_withSensor_cylinder.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML file not found at: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # --- 2. 系统参数 ---
        self.dt = 0.02  # 50Hz
        self.model.opt.timestep = 0.002 
        self.sim_steps = int(self.dt / self.model.opt.timestep)
        
        # --- 3. 获取对象 ID ---
        mocap_body = self.model.body("mocap")
        if hasattr(mocap_body, 'mocapid'):
            mids = mocap_body.mocapid
            if isinstance(mids, np.ndarray) or isinstance(mids, list):
                self.mocap_id = mids[0]
            else:
                self.mocap_id = mids
        else:
            raise ValueError("Model does not contain a mocap body named 'mocap'")

        self.prefab_jnt_id = self.model.body("prefab").jntadr[0]
        self.target_body_id = self.model.body("rebar_base").id
        
        # --- 4. 状态与动作 ---
        self.state_dim = 10 
        self.action_dim = 2 
        self.action_space_high = 0.5 
        
        # --- 5. 任务参数 ---
        self.start_pos_mocap = np.array([0.2, 0.3, 1.0])
        self.default_target = np.array([-0.2, 0.3])
        self.target_pos = self.default_target.copy()
        self.max_steps = 500 
        self.current_step = 0
        
        # --- 6. 渲染 ---
        self.render_mode = render
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        
        # 1. 随机化目标
        noise_target = np.random.uniform(-0.1, 0.1, size=2)
        self.target_pos = self.default_target + noise_target
        self.model.body_pos[self.target_body_id][:2] = self.target_pos
        
        # 2. 随机化负载初始位置
        start_x = 0.2 + np.random.uniform(-0.05, 0.05)
        start_y = 0.3 + np.random.uniform(-0.05, 0.05)
        
        q_idx = self.prefab_jnt_id
        self.data.qpos[q_idx] = start_x
        self.data.qpos[q_idx+1] = start_y
        
        # 3. Mocap 对齐
        self.data.mocap_pos[self.mocap_id][0] = start_x
        self.data.mocap_pos[self.mocap_id][1] = start_y
        self.data.mocap_pos[self.mocap_id][2] = 1.0
        
        # 4. 预热
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            
        self.current_mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
        self.current_mocap_vel = np.zeros(3)
        
        self.current_step = 0
        return self._get_obs()

    def step(self, action):
        # 获取当前误差状态 (用于判断是否自动下降)
        obs_tmp = self._get_obs()
        dist_xy = np.linalg.norm(obs_tmp[8:10]) # 目标距离
        vel_xy = np.linalg.norm(obs_tmp[6:8])   # 摆动速度
        
        # --- 动作处理逻辑 ---
        if len(action) == 2:
            # Case A: RL 训练模式 (2D 动作)
            # 启用【自动下降逻辑】
            action = np.clip(action, -self.action_space_high, self.action_space_high)
            ax, ay = action
            
            # 逻辑：如果水平对准了(<5cm) 且 摆动很小(<0.15m/s)，就开始下降
            if dist_xy < 0.05 and vel_xy < 0.15:
                # 简单的 P 控制向下
                target_vz = -0.2
                current_vz = self.current_mocap_vel[2]
                az = 2.0 * (target_vz - current_vz)
            else:
                # 否则保持高度 Z=1.0
                target_z = 1.0
                current_z = self.current_mocap_pos[2]
                current_vz = self.current_mocap_vel[2]
                # PD 控制保持高度
                az = 5.0 * (target_z - current_z) - 2.0 * current_vz
                
        else:
            # Case B: NMPC 测试模式 (3D 动作)
            # 直接执行输入的 Z 轴指令
            action = np.clip(action, -self.action_space_high, self.action_space_high)
            ax, ay, az = action

        # --- 动力学积分 (3D) ---
        self.current_mocap_pos[0] += self.current_mocap_vel[0] * self.dt + 0.5 * ax * self.dt**2
        self.current_mocap_pos[1] += self.current_mocap_vel[1] * self.dt + 0.5 * ay * self.dt**2
        self.current_mocap_pos[2] += self.current_mocap_vel[2] * self.dt + 0.5 * az * self.dt**2
        
        self.current_mocap_vel[0] += ax * self.dt
        self.current_mocap_vel[1] += ay * self.dt
        self.current_mocap_vel[2] += az * self.dt
        
        # 地面碰撞保护
        if self.current_mocap_pos[2] < 0.4: 
             self.current_mocap_pos[2] = 0.4
             self.current_mocap_vel[2] = 0

        self.data.mocap_pos[self.mocap_id] = self.current_mocap_pos
        
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
            
        if self.render_mode and self.viewer:
            self.viewer.sync()
            
        self.current_step += 1
        obs = self._get_obs()
        reward, done, success = self._compute_reward(obs)
        
        if self.current_step >= self.max_steps:
            done = True
            
        return obs, reward, done, success

    def _get_obs(self):
        px, py = self.current_mocap_pos[0], self.current_mocap_pos[1]
        vpx, vpy = self.current_mocap_vel[0], self.current_mocap_vel[1]
        
        q_idx = self.prefab_jnt_id
        qx = self.data.qpos[q_idx]
        qy = self.data.qpos[q_idx+1]
        
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
        vqx = self.data.qvel[dof_idx]
        vqy = self.data.qvel[dof_idx+1]
        
        tx = self.target_pos[0] - qx
        ty = self.target_pos[1] - qy
        
        return np.array([px, py, vpx, vpy, qx, qy, vqx, vqy, tx, ty], dtype=np.float32)

    def _compute_reward(self, obs):
        dist_xy = np.linalg.norm(obs[8:10])
        payload_vel = np.linalg.norm(obs[6:8])
        q_z = self.data.qpos[self.prefab_jnt_id + 2]
        
        success = False
        reward = 0.0
        
        # 成功判据：XY对准 + 不摆 + Z到位
        if dist_xy < 0.03 and payload_vel < 0.1 and q_z < 0.15:
            reward = 1.0
            success = True
            
        return reward, success, success


# ---------------------------------------------------------------------------
# 带路径障碍物的环境：临时 XML、2D 路径规划、CableRobotEnvWithObstacles
# ---------------------------------------------------------------------------

OBSTACLE_Z_CENTER = 0.45
OBSTACLE_HALFHEIGHT = 0.2


def _sample_obstacles_on_path(start_xy, target_xy, n_obstacles, radius_range,
                              path_width=0.15, rng=None,
                              min_clearance_from_endpoints=0.0):
    """
    在起点到目标的路径附近采样圆形障碍物（仅在 XY 平面）。
    约束：障碍物圆心到起点、终点的距离均不少于 min_clearance_from_endpoints + 障碍物半径，
    以保证目标物（负载）在起点和终点处有足够空间，不与障碍物过近。
    """
    if rng is None:
        rng = np.random.default_rng()
    r_min, r_max = radius_range
    delta = target_xy - start_xy
    length = np.linalg.norm(delta)
    if length < 1e-6:
        direction = np.array([1.0, 0.0])
    else:
        direction = delta / length
    normal = np.array([-direction[1], direction[0]])
    obstacles = []
    max_attempts = 500
    for _ in range(n_obstacles):
        for _ in range(max_attempts):
            t = rng.uniform(0.15, 0.85)
            along = start_xy + t * delta
            off = rng.uniform(-path_width, path_width)
            center = along + off * normal
            r = rng.uniform(r_min, r_max)
            d_start = np.linalg.norm(center - start_xy)
            d_target = np.linalg.norm(center - target_xy)
            need = min_clearance_from_endpoints + r
            if d_start >= need and d_target >= need:
                obstacles.append((float(center[0]), float(center[1]), float(r)))
                break
        else:
            obstacles.append((float(center[0]), float(center[1]), float(r)))
    return obstacles


def _build_xml_with_obstacles(base_xml_content, obstacles,
                              path_points=None,
                              start_xy=None,
                              goal_xy=None):
    """在基础 XML 中插入障碍物与路径可视化，并设置 rebar_base 的 xy 与 goal_xy 一致。"""
    if obstacles:
        material_line = '    <material name="steel" rgba="0.6 0.6 0.6 1"/>'
        insert = (
            '    <material name="steel" rgba="0.6 0.6 0.6 1"/>\n'
            '    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>'
        )
        xml = base_xml_content.replace(material_line, insert, 1)
    else:
        xml = base_xml_content

    obstacle_bodies = []
    for i, (x, y, r) in enumerate(obstacles):
        body = (
            f'    <!-- 路径障碍物 {i} (静态) -->\n'
            f'    <body name="obstacle_{i}" pos="{x} {y} {OBSTACLE_Z_CENTER}">\n'
            f'      <geom type="cylinder" size="{r} {OBSTACLE_HALFHEIGHT}" pos="0 0 0" '
            f'material="obstacle" contype="1" conaffinity="1"/>\n'
            f'    </body>\n\n'
        )
        obstacle_bodies.append(body)
    obstacles_block = '\n'.join(obstacle_bodies) if obstacle_bodies else ''

    path_bodies = []
    if path_points is not None:
        path_z = OBSTACLE_Z_CENTER + OBSTACLE_HALFHEIGHT + 0.02
        for i, p in enumerate(path_points):
            px, py = float(p[0]), float(p[1])
            body = (
                f'    <!-- 规划路径点 {i} (可视化) -->\n'
                f'    <body name="path_pt_{i}" pos="{px} {py} {path_z}">\n'
                f'      <geom type="sphere" size="0.01" rgba="0 0 1 1" '
                f'contype="0" conaffinity="0"/>\n'
                f'    </body>\n\n'
            )
            path_bodies.append(body)
    path_block = '\n'.join(path_bodies) if path_bodies else ''

    endpoint_bodies = []
    endpoint_z = OBSTACLE_Z_CENTER + OBSTACLE_HALFHEIGHT + 0.025
    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        body = (
            f'    <!-- 规划起点 (可视化) -->\n'
            f'    <body name="path_start" pos="{sx} {sy} {endpoint_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n\n'
        )
        endpoint_bodies.append(body)
    if goal_xy is not None:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        body = (
            f'    <!-- 规划终点 (可视化) -->\n'
            f'    <body name="path_goal" pos="{gx} {gy} {endpoint_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n\n'
        )
        endpoint_bodies.append(body)
    endpoint_block = '\n'.join(endpoint_bodies) if endpoint_bodies else ''
    replacement = (
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
        + (obstacles_block if obstacles_block else '')
        + (path_block if path_block else '')
        + (endpoint_block if endpoint_block else '')
        + '    <!--  固定钢筋 -->'
    )
    xml = xml.replace(
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    <!--  固定钢筋 -->',
        replacement,
        1,
    )
    if goal_xy is not None:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        xml = re.sub(
            r'<body name="rebar_base" pos="[^"]+">',
            f'<body name="rebar_base" pos="{gx} {gy} 0">',
            xml,
            count=1,
        )
    return xml


def plan_path_2d(start_xy, target_xy, obstacles,
                 grid_res=0.02,
                 payload_radius=0.06,
                 safety_margin=0.02,
                 bounds_margin=0.3,
                 max_expansions=100000):
    """在 XY 平面上使用栅格 A* 规划从 start_xy 到 target_xy 的避障路径。"""
    start_xy = np.asarray(start_xy, dtype=float).reshape(2)
    target_xy = np.asarray(target_xy, dtype=float).reshape(2)
    if not obstacles:
        return np.vstack([start_xy, target_xy])

    xs = [start_xy[0], target_xy[0]]
    ys = [start_xy[1], target_xy[1]]
    for (ox, oy, r) in obstacles:
        r_eff = r + payload_radius + safety_margin
        xs.extend([ox - r_eff, ox + r_eff])
        ys.extend([oy - r_eff, oy + r_eff])
    x_min = min(xs) - bounds_margin
    x_max = max(xs) + bounds_margin
    y_min = min(ys) - bounds_margin
    y_max = max(ys) + bounds_margin
    nx = max(2, int(np.ceil((x_max - x_min) / grid_res)))
    ny = max(2, int(np.ceil((y_max - y_min) / grid_res)))

    def world_to_grid(x, y):
        i = int((x - x_min) / grid_res)
        j = int((y - y_min) / grid_res)
        i = max(0, min(nx - 1, i))
        j = max(0, min(ny - 1, j))
        return int(i), int(j)

    def grid_to_world(i, j):
        x = x_min + (i + 0.5) * grid_res
        y = y_min + (j + 0.5) * grid_res
        return x, y

    occ = np.zeros((nx, ny), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            x, y = grid_to_world(i, j)
            for (ox, oy, r) in obstacles:
                r_eff = r + payload_radius + safety_margin
                if (x - ox) ** 2 + (y - oy) ** 2 < r_eff ** 2:
                    occ[i, j] = True
                    break

    def find_nearest_free(i0, j0, search_radius=5):
        if not occ[i0, j0]:
            return i0, j0
        best = None
        best_d2 = None
        for di in range(-search_radius, search_radius + 1):
            for dj in range(-search_radius, search_radius + 1):
                i = i0 + di
                j = j0 + dj
                if 0 <= i < nx and 0 <= j < ny and not occ[i, j]:
                    d2 = di * di + dj * dj
                    if best is None or d2 < best_d2:
                        best = (i, j)
                        best_d2 = d2
        return best

    start_ij = world_to_grid(start_xy[0], start_xy[1])
    goal_ij = world_to_grid(target_xy[0], target_xy[1])
    start_ij = find_nearest_free(*start_ij) or start_ij
    goal_ij = find_nearest_free(*goal_ij) or goal_ij

    def heuristic(i, j):
        x, y = grid_to_world(i, j)
        return float(np.hypot(x - target_xy[0], y - target_xy[1]))

    open_heap = []
    g_cost = {start_ij: 0.0}
    parent = {}
    heapq.heappush(open_heap, (heuristic(*start_ij), start_ij))
    closed = set()
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
    found = False
    expansions = 0

    while open_heap and expansions < max_expansions:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_ij:
            found = True
            break
        closed.add(current)
        ci, cj = current
        expansions += 1
        for di, dj in neighbors:
            ni, nj = ci + di, cj + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if occ[ni, nj]:
                continue
            step_cost = grid_res if di == 0 or dj == 0 else grid_res * np.sqrt(2.0)
            new_g = g_cost[current] + step_cost
            neighbor = (ni, nj)
            if neighbor in g_cost and new_g >= g_cost[neighbor]:
                continue
            g_cost[neighbor] = new_g
            parent[neighbor] = current
            heapq.heappush(open_heap, (new_g + heuristic(ni, nj), neighbor))

    if not found:
        return np.vstack([start_xy, target_xy])
    path_idx = []
    node = goal_ij
    while node != start_ij:
        path_idx.append(node)
        node = parent.get(node)
        if node is None:
            break
    path_idx.append(start_ij)
    path_idx.reverse()
    pts = [grid_to_world(i, j) for (i, j) in path_idx]
    return np.array(pts, dtype=float)


class CableRobotEnvWithObstacles(CableRobotEnv):
    """
    带路径障碍物的环境：每次 reset 时生成带静态障碍物的临时 XML 并加载，
    同时进行 2D 路径规划；通过 get_obstacles() / get_planned_path() 暴露给控制层。
    """

    def __init__(self, n_obstacles=3, obstacle_radius_range=(0.001, 0.005),
                 path_width=0.12, obstacle_seed=None,
                 payload_radius=0.06, planning_margin=0.02,
                 planning_grid_res=0.02,
                 default_start_xy=None, default_target_xy=None,
                 **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self._assets2_dir = os.path.join(current_dir, "assets2")
        base_xml_path = os.path.join(
            self._assets2_dir,
            "demo_fourCable_withSteel_withSensor_cylinder.xml",
        )
        if not os.path.exists(base_xml_path):
            raise FileNotFoundError(f"Base XML not found: {base_xml_path}")
        with open(base_xml_path, "r", encoding="utf-8") as f:
            self._base_xml_content = f.read()

        self.model = mujoco.MjModel.from_xml_path(base_xml_path)
        self.data = mujoco.MjData(self.model)
        self.physics_dt = 0.002
        self.control_freq_hz = kwargs.get("control_freq_hz", 10)
        self.control_dt = 1.0 / self.control_freq_hz
        self.dt = self.control_dt
        self.model.opt.timestep = self.physics_dt
        self.sim_steps = int(self.dt / self.model.opt.timestep)

        mocap_body = self.model.body("mocap")
        if hasattr(mocap_body, "mocapid"):
            mids = mocap_body.mocapid
            self.mocap_id = mids[0] if isinstance(mids, (np.ndarray, list)) else mids
        else:
            raise ValueError("Model does not contain a mocap body named 'mocap'")
        self.prefab_jnt_id = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("rebar_base").id

        self.state_dim = 10
        self.action_dim = 2
        self.action_space_high = 0.5
        self.start_pos_mocap = np.array([0.2, 0.3, 1.0])
        _start = np.array(default_start_xy if default_start_xy is not None else [0.2, 0.3])
        _target = np.array(default_target_xy if default_target_xy is not None else [-0.2, 0.3])
        self.default_start_xy = _start
        self.default_target = _target
        self.target_pos = self.default_target.copy()
        self.max_steps = 500
        self.current_step = 0

        self.latency_steps = kwargs.get("latency_steps", 1)
        self.action_buffer = deque(maxlen=self.latency_steps + 1)
        self.force_noise_level = kwargs.get("force_noise_level", 0.1)
        self.init_velocity_scale = kwargs.get("init_velocity_scale", 0.15)
        self.init_position_range = kwargs.get("init_position_range", 0.08)
        self.render_mode = kwargs.get("render", False)
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.n_obstacles = n_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.path_width = path_width
        self._obstacle_rng = np.random.default_rng(obstacle_seed)
        self._obstacles = []
        self._temp_xml_path = None
        self.payload_radius = float(payload_radius)
        self.planning_margin = float(planning_margin)
        self.planning_grid_res = float(planning_grid_res)
        self._planned_path = None

    def _reresolve_ids(self):
        mocap_body = self.model.body("mocap")
        if hasattr(mocap_body, "mocapid"):
            mids = mocap_body.mocapid
            self.mocap_id = mids[0] if isinstance(mids, (np.ndarray, list)) else mids
        self.prefab_jnt_id = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("rebar_base").id

    def _reload_model_with_obstacles(self, obstacles, path_points=None, start_xy=None, goal_xy=None):
        xml_content = _build_xml_with_obstacles(
            self._base_xml_content, obstacles,
            path_points=path_points, start_xy=start_xy, goal_xy=goal_xy,
        )
        fd, path = tempfile.mkstemp(suffix=".xml", dir=self._assets2_dir, prefix="obstacles_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(xml_content)
            self._temp_xml_path = path
        except Exception:
            os.close(fd)
            if os.path.exists(path):
                os.remove(path)
            raise
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.physics_dt
        self.sim_steps = int(self.dt / self.model.opt.timestep)
        self._reresolve_ids()
        if self.render_mode and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        self._temp_xml_path = None

    def reset(self):
        start_xy = self.default_start_xy.copy()
        target_xy = self.default_target.copy()
        min_clearance = self.payload_radius + self.planning_margin
        self._obstacles = _sample_obstacles_on_path(
            start_xy, target_xy, self.n_obstacles, self.obstacle_radius_range,
            self.path_width, self._obstacle_rng,
            min_clearance_from_endpoints=min_clearance,
        )
        self._planned_path = plan_path_2d(
            start_xy=start_xy, target_xy=target_xy, obstacles=self._obstacles,
            grid_res=self.planning_grid_res, payload_radius=self.payload_radius,
            safety_margin=self.planning_margin,
        )
        self._reload_model_with_obstacles(
            self._obstacles, self._planned_path, start_xy=start_xy, goal_xy=target_xy,
        )
        obs = super().reset()
        rng = self._obstacle_rng
        start_x = self.default_start_xy[0] + rng.uniform(-self.init_position_range, self.init_position_range)
        start_y = self.default_start_xy[1] + rng.uniform(-self.init_position_range, self.init_position_range)
        q_idx = self.prefab_jnt_id
        self.data.qpos[q_idx] = start_x
        self.data.qpos[q_idx + 1] = start_y
        self.data.mocap_pos[self.mocap_id][0] = start_x
        self.data.mocap_pos[self.mocap_id][1] = start_y
        self.data.mocap_pos[self.mocap_id][2] = 1.0
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.current_mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
        self.current_mocap_vel = np.zeros(3)
        return self._get_obs()

    def step(self, action):
        return super().step(action)

    def get_obstacles(self):
        return list(self._obstacles)

    def get_planned_path(self):
        if self._planned_path is None:
            return None
        return np.array(self._planned_path, copy=True)