import mujoco
import mujoco.viewer
import numpy as np
import os
import re
import tempfile
import heapq
from collections import deque

from ipdb import set_trace as xxxx

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
        # self.data.qpos[q_idx] = start_x ## orig
        # self.data.qpos[q_idx+1] = start_y ## orig
        self.data.body('prefab').xpos[0] = start_x ## xyc: 重新确定prefab的x-value
        self.data.body('prefab').xpos[1] = start_y ## xyc: 重新确定prefab的y-value
        
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
        obs_tmp = self._get_obs()
        dist_xy = np.linalg.norm(obs_tmp[8:10]) 
        vel_xy = np.linalg.norm(obs_tmp[6:8])   
        
        if len(action) == 2:
            action = np.clip(action, -self.action_space_high, self.action_space_high)
            ax, ay = action
            
            if dist_xy < 0.05 and vel_xy < 0.15:
                target_vz = -0.2
                current_vz = self.current_mocap_vel[2]
                az = 2.0 * (target_vz - current_vz)
            else:
                target_z = 1.0
                current_z = self.current_mocap_pos[2]
                current_vz = self.current_mocap_vel[2]
                az = 5.0 * (target_z - current_z) - 2.0 * current_vz
                
        else:
            action = np.clip(action, -self.action_space_high, self.action_space_high)
            ax, ay, az = action

        self.current_mocap_pos[0] += self.current_mocap_vel[0] * self.dt + 0.5 * ax * self.dt**2
        self.current_mocap_pos[1] += self.current_mocap_vel[1] * self.dt + 0.5 * ay * self.dt**2
        self.current_mocap_pos[2] += self.current_mocap_vel[2] * self.dt + 0.5 * az * self.dt**2
        
        self.current_mocap_vel[0] += ax * self.dt
        self.current_mocap_vel[1] += ay * self.dt
        self.current_mocap_vel[2] += az * self.dt
        
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
        # qx = self.data.qpos[q_idx] ## orig
        # qy = self.data.qpos[q_idx+1] ## orig
        qx = self.data.body('prefab').xpos[0] ## xyc: 重新确定prefab的x-value
        qy = self.data.body('prefab').xpos[1] ## xyc: 重新确定prefab的y-value
        
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
        vqx = self.data.qvel[dof_idx]
        vqy = self.data.qvel[dof_idx+1]
        
        tx = self.target_pos[0] - qx
        ty = self.target_pos[1] - qy
        
        return np.array([px, py, vpx, vpy, qx, qy, vqx, vqy, tx, ty], dtype=np.float32)

    def _compute_reward(self, obs):
        dist_xy = np.linalg.norm(obs[8:10])
        payload_vel = np.linalg.norm(obs[6:8])
        # q_z = self.data.qpos[self.prefab_jnt_id + 2] ## orig
        q_z = self.data.body('prefab').xpos[2] ## xyc: 重新确定prefab的z-value
        
        success = False
        reward = 0.0
        
        if dist_xy < 0.03 and payload_vel < 0.1 and q_z < 0.15:
            reward = 1.0
            success = True
            
        return reward, success, success


# ---------------------------------------------------------------------------
# 带路径障碍物的环境：临时 XML、2D 路径规划、CableRobotEnvWithObstacles
# ---------------------------------------------------------------------------

OBSTACLE_Z_CENTER = 0.25
OBSTACLE_HALFHEIGHT = 0.2

def _sample_obstacles_on_path(start_xy, target_xy, n_obstacles, radius_range,
                              path_width=0.15, rng=None,
                              min_clearance_from_endpoints=0.0):
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
            f'    \n'
            f'    <body name="obstacle_{i}" pos="{x} {y} {OBSTACLE_Z_CENTER}">\n'
            f'      <geom type="cylinder" size="{r} {OBSTACLE_HALFHEIGHT}" pos="0 0 0" '
            f'material="obstacle" contype="1" conaffinity="1"/>\n'
            f'    </body>\n\n'
        )
        obstacle_bodies.append(body)
    obstacles_block = '\n'.join(obstacle_bodies) if obstacle_bodies else ''

    path_bodies = []
    if path_points is not None:
        for i, p in enumerate(path_points):
            px, py = float(p[0]), float(p[1])
            # [核心修改 1]：让可视化点兼容 3D 坐标输入
            pz = float(p[2]) if len(p) >= 3 else (OBSTACLE_Z_CENTER + OBSTACLE_HALFHEIGHT + 0.02)
            body = (
                f'    \n'
                f'    <body name="path_pt_{i}" pos="{px} {py} {pz}">\n'
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
            f'    \n'
            f'    <body name="path_start" pos="{sx} {sy} {endpoint_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" '
            f'contype="0" conaffinity="0"/>\n'
            f'    </body>\n\n'
        )
        endpoint_bodies.append(body)
    if goal_xy is not None:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        body = (
            f'    \n'
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
        + '    '
    )
    xml = xml.replace(
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    ',
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

        self.n_obstacles = n_obstacles 
        self.state_dim = 10 + (self.n_obstacles * 3) + 4 
        self.action_dim = 3

        self.action_space_high = 0.5
        self.start_pos_mocap = np.array([0.2, 0.3, 1.0])
        _start = np.array(default_start_xy if default_start_xy is not None else [0.2, 0.3])
        _target = np.array(default_target_xy if default_target_xy is not None else [-0.2, 0.3])
        self.default_start_xy = _start
        self.default_target = _target
        self.target_pos = self.default_target.copy()
        self.max_steps = 150
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
        # load xml with obstacles
        xml_content = _build_xml_with_obstacles(
            self._base_xml_content, obstacles,
            path_points=path_points, start_xy=start_xy, goal_xy=goal_xy,
        )
        # write xml to file
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
        # load model from xml file
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.physics_dt
        self.sim_steps = int(self.dt / self.model.opt.timestep)
        self._reresolve_ids()
        # launch viewer if render mode is enabled
        if self.render_mode and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            ## xyc: 增加初始化时的预热，检查初始化时的模型是否正确
            for _ in range(500):
                mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            # xyc?: 发现此时，机械臂末端并没有跟踪mocap的位置，而是自然掉落，需确认是否是跟踪mocap的控制失效
            ## xyc: 增加初始化时的预热，检查初始化时的模型是否正确
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
        
        # 1. 障碍物及 2D 路径规划
        self._obstacles = _sample_obstacles_on_path(
            start_xy, target_xy, self.n_obstacles, self.obstacle_radius_range,
            self.path_width, self._obstacle_rng,
            min_clearance_from_endpoints=min_clearance,
        )
        path_2d = plan_path_2d(
            start_xy=start_xy, target_xy=target_xy, obstacles=self._obstacles,
            grid_res=self.planning_grid_res, payload_radius=self.payload_radius,
            safety_margin=self.planning_margin,
        )
        
        # [核心修改 2]：预构造占位 3D 轨迹点（包含下降段），以确保编译的 XML 包含正确数量的球体用于渲染
        num_descent_steps = 6
        dummy_path = []
        for pt in path_2d:
            dummy_path.append([pt[0], pt[1], 1.0])
        last_xy = path_2d[-1]
        for _ in range(num_descent_steps):
            dummy_path.append([last_xy[0], last_xy[1], 0.5])
        dummy_path_np = np.array(dummy_path)

        # 加载带实体的 XML 模型
        self._reload_model_with_obstacles(
            self._obstacles, dummy_path_np, start_xy=start_xy, goal_xy=target_xy,
        )
        
        obs = super().reset() # 父类 reset 会再次随机化初始位置和目标

        self.target_pos = target_xy.copy()
        self.model.body_pos[self.target_body_id][:2] = target_xy

        rng = self._obstacle_rng
        start_x = self.default_start_xy[0] + rng.uniform(-self.init_position_range, self.init_position_range)
        start_y = self.default_start_xy[1] + rng.uniform(-self.init_position_range, self.init_position_range)
        q_idx = self.prefab_jnt_id
        # self.data.qpos[q_idx] = start_x ## orig
        # self.data.qpos[q_idx + 1] = start_y ## orig
        self.data.body('prefab').xpos[0] = start_x ## xyc: 重新确定prefab的x-value
        self.data.body('prefab').xpos[1] = start_y ## xyc: 重新确定prefab的y-value
        self.data.mocap_pos[self.mocap_id][0] = start_x
        self.data.mocap_pos[self.mocap_id][1] = start_y
        self.data.mocap_pos[self.mocap_id][2] = 1.0
        
        # 再次预热以让绳索和负载自然下垂稳定
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            
        self.current_mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
        self.current_mocap_vel = np.zeros(3)
        
        # [核心修改 3]：预热完毕后获取实际悬停高度，构建真正的 3D 轨迹！
        # payload_z_cruise = self.data.body('prefab').xpos[2] ## xyc: confirm z-value of planned path
        payload_z_cruise = 0.35 ## xyc?: confirm z-value of planned path, 固定一个值方便测试
        
        true_path_3d = []
        # 前半段：保持悬停高度的平移
        for pt in path_2d:
            true_path_3d.append([pt[0], pt[1], payload_z_cruise])
            
        # 后半段：目标点正上方的垂直下降（下潜至 0.12 米）
        descent_zs = np.linspace(payload_z_cruise, 0.12, num_descent_steps + 1)[1:] 
        for z in descent_zs:
            true_path_3d.append([last_xy[0], last_xy[1], z])
            
        self._planned_path = np.array(true_path_3d)

        # 动态更新物理引擎中轨迹球的位置，完成完美的可视化对接
        for i, pt in enumerate(self._planned_path):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"path_pt_{i}")
            if body_id != -1:
                self.model.body_pos[body_id] = pt

        self.current_wp_idx = 0
        self.reached_final = False
        
        obs = self._get_obs()
        
        if self._planned_path is not None and len(self._planned_path) > 0:
            # 初始化变为 3D 距离计算
            payload_z = self.data.body('prefab').xpos[2] ## xyc: 重新确定prefab的z-value，解决第一帧reset问题
            current_pos = np.array([obs[4], obs[5], payload_z])
            target_wp = self._planned_path[self.current_wp_idx]
            self.last_dist = np.linalg.norm(current_pos - target_wp)
        else:
            self.last_dist = None
            
        return obs

    def step(self, action):
        action = np.clip(action, -self.action_space_high, self.action_space_high)
        
        if len(action) == 3:
            ax, ay, az = action
        elif len(action) == 2:
            ax, ay = action
            az = 0.0
        else:
            ax, ay, az = action[:3]

        self.current_mocap_pos[0] += self.current_mocap_vel[0] * self.dt + 0.5 * ax * self.dt**2
        self.current_mocap_pos[1] += self.current_mocap_vel[1] * self.dt + 0.5 * ay * self.dt**2
        self.current_mocap_pos[2] += self.current_mocap_vel[2] * self.dt + 0.5 * az * self.dt**2
        
        self.current_mocap_vel[0] += ax * self.dt
        self.current_mocap_vel[1] += ay * self.dt
        self.current_mocap_vel[2] += az * self.dt
        
        if self.current_mocap_pos[2] < 0.4: 
             self.current_mocap_pos[2] = 0.4
             self.current_mocap_vel[2] = 0

        self.data.mocap_pos[self.mocap_id] = self.current_mocap_pos


        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)
            
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return self._get_obs(), -10.0, True, False

        if self.render_mode and hasattr(self, 'viewer') and self.viewer:
            self.viewer.sync()
            
        self.current_step += 1
        obs = self._get_obs()
        
        if self._planned_path is not None and len(self._planned_path) > 0 and not self.reached_final:
            # [核心修改 4]：推演状态机，将目标捕捉的距离测算升级为 3D
            payload_z = self.data.body('prefab').xpos[2] ## xyc: 重新确定prefab的z-value，解决第一帧reset问题
            current_pos = np.array([obs[4], obs[5], payload_z])
            target_wp = self._planned_path[self.current_wp_idx]
            dist_to_wp = np.linalg.norm(current_pos - target_wp)
            
            total_wps = len(self._planned_path)
            rem_wps = total_wps - 1 - self.current_wp_idx
            # 引入 Z 轴后容差稍微放大，确保平滑过渡而不卡死在航点上
            look_ahead_dist = 0.06 if rem_wps <= 2 else 0.10
                
            if dist_to_wp < look_ahead_dist:
                step_idx = min(1, rem_wps)
                self.current_wp_idx += step_idx
                if self.current_wp_idx >= total_wps - 1:
                    self.current_wp_idx = total_wps - 1
                    self.reached_final = True
        
        reward, done, success = self._compute_reward(obs, action)
        
        payload_z = self.data.body('prefab').xpos[2] ## xyc: 重新确定prefab的z-value，解决第一帧reset问题
        dist_to_final = np.linalg.norm(obs[8:10])

        if payload_z <= 0.12: 
            done = True
            vel_xy = np.linalg.norm(obs[6:8])   
            payload_vz = obs[-1]
            # 【优化】：加入对下降速度的限制 (如绝对值小于 0.5 m/s，防止砸地)
            if dist_to_final < 0.15 and vel_xy < 0.2 and abs(payload_vz) < 0.5:
                success = True
                reward += 15.0 # 【优化】：拉开成功奖励与失败惩罚的差距
            else:
                reward -= 5.0  # 摔机或偏离
        
        if self.current_step >= self.max_steps:
            done = True
            if not success:
                reward -= 5.0   
            
        if dist_to_final > 2.0: 
            reward -= 10.0      
            done = True

        reward = float(np.clip(reward, -15.0, 15.0))
            
        return obs, reward, done, success

    def _compute_reward(self, obs, action):
        done = False
        success = False
        reward = 0.0
        
        current_xy = obs[4:6]
        payload_vel = np.linalg.norm(obs[6:8])
        
        if self._planned_path is not None and len(self._planned_path) > 0:
            # [核心修改 5]：势能导向函数升级为引导 3D 趋近，自动鼓励垂直下落
            payload_z = self.data.body('prefab').xpos[2] ## xyc: 重新确定prefab的z-value，解决第一帧reset问题
            current_pos = np.array([current_xy[0], current_xy[1], payload_z])
            target_wp = self._planned_path[self.current_wp_idx]
            
            dist_t = np.linalg.norm(current_pos - target_wp)
            
            if getattr(self, 'last_wp_idx', -1) != self.current_wp_idx:
                self.last_dist = None
                self.last_wp_idx = self.current_wp_idx

            if self.last_dist is not None:
                step_progress_reward = (self.last_dist - dist_t) * 50.0
                reward += np.clip(step_progress_reward, -2.0, 2.0)
            
            reward -= np.clip(dist_t * 0.5, 0.0, 2.0)
            self.last_dist = dist_t

        if hasattr(self, '_obstacles') and self._obstacles:
            payload_radius = getattr(self, 'payload_radius', 0.1) 
            for (ox, oy, orad) in self._obstacles:
                d = np.linalg.norm(current_xy - np.array([ox, oy]))
                if d < (orad + payload_radius):
                    reward -= 8.0  
                    done = True    
                    return reward, done, success

        reward -= 0.02 * np.sum(np.square(action))
        reward -= np.clip(0.1 * payload_vel, 0.0, 1.0)
        reward -= 0.02
            
        return reward, done, success
    
    def _get_obs(self):
        base_obs = super()._get_obs()
        
        obs_data = []
        if hasattr(self, '_obstacles') and self._obstacles:
            for (ox, oy, r) in self._obstacles:
                obs_data.extend([ox, oy, r])
                
        target_len = self.n_obstacles * 3
        while len(obs_data) < target_len:
            obs_data.append(0.0)
            
        mocap_z = self.current_mocap_pos[2]
        mocap_vz = self.current_mocap_vel[2]
        
        payload_z = self.data.body('prefab').xpos[2] ## xyc: 重新确定prefab的z-value
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vz = self.data.qvel[dof_idx + 2]
            
        return np.concatenate([base_obs, obs_data[:target_len], [mocap_z, mocap_vz, payload_z, payload_vz]], dtype=np.float32)
    
    def get_obstacles(self):
        return list(self._obstacles)

    def get_planned_path(self):
        if self._planned_path is None:
            return None
        return np.array(self._planned_path, copy=True)