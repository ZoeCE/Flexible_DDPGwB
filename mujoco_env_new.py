# ==============================================================================
# mujoco_env_new.py — 关节空间控制版索驱动机器人环境
#
# 架构变更（相对原版）：
#
# [ENV-NEW-1] 动作空间重定义：7D 关节角目标
#   旧版：action = 6D 末端加速度 [ax, ay, az, a_roll, a_pitch, a_yaw]
#   新版：action = 7D 关节角目标 [q1, q2, q3, q4, q5, q6, q7]
#   执行机制：
#     - env.step(action) 直接将 7D 关节角下发给 MuJoCo 位置执行器
#     - 内部不再维护 current_mocap_pos 积分（去除了"虚拟末端"概念）
#     - EE 位置由 MuJoCo FK 自动计算（data.site_xpos）
#
# [ENV-NEW-2] 观测空间更新（新增关节状态）
#   旧版观测：22 + 3*n 维（不包含关节状态）
#   新版观测：22 + 3*n + 14 维
#   新增末尾 14 维（关节角 7 + 关节角速度 7）：
#     [-14] ~ [-8]  q1..q7    当前关节角（rad）
#     [-7]  ~ [-1]  dq1..dq7  当前关节角速度（rad/s）
#   总计：22 + 3*n_obstacles + 14 = 36 + 3*n_obstacles
#   末尾固定负索引：
#     [-12] mocap_z  ...（原有 12 维）
#     [-26] ~ [-13] 关节状态 14 维（在原 12 维前面，共 26 维末尾）
#   为避免索引混乱，观测末尾布局统一为：
#     [*basic_10] + [*obstacle_3n] + [*ee_z_vel_12] + [*joints_14]
#   因此新的末尾关节角索引为：[-14] ~ [-8] 关节角，[-7] ~ [-1] 关节速度
#
# [ENV-NEW-3] 奖励函数新增关节空间惩罚
#   - joint_limit_penalty：关节角接近极限时的惩罚
#   - joint_smooth_penalty：相邻步关节角变化量惩罚（防抖）
#
# [ENV-NEW-4] 内部状态管理简化
#   - 移除 current_mocap_pos/vel 的手动积分
#   - EE 位置直接从 data.site_xpos['attachment_site'] 读取
#   - 为了兼容 controller.py 中的 obs 索引约定，
#     _get_obs 中 mocap_x/y/z/vx/vy/vz 均直接从 MuJoCo 真实 EE 状态读取
#
# 保留内容（与原版完全一致）：
#   - A* 路径规划与场景生成（generate_scene_and_trajectory）
#   - 航点追踪状态机（step() 中的 wp_idx 逻辑）
#   - 终止条件判定（碰撞、坠毁、出界、成功）
#   - NativeIKSolver 类
#   - 所有 BUG 修复标注（BUG-1 ~ BUG-13 + IMPROVE）
# ==============================================================================

import os
import copy
import heapq
import tempfile
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R

from config import DEFAULT_CONFIG


class CableRobotEnvWithObstacles:
    """
    关节空间控制版索驱动机器人环境。

    动作空间：7D 关节角目标 [q1..q7]（rad）
    观测布局：
      [0-1]   ee_x, ee_y               末端执行器 XY 位置（FK 计算）
      [2-3]   ee_vx, ee_vy             末端 XY 速度（数值微分）
      [4-5]   payload_x, payload_y     负载 XY 位置
      [6-7]   payload_vx, payload_vy   负载 XY 速度
      [8-9]   rel_tx, rel_ty           目标相对负载距离
      [10 ~ 10+3n-1]  障碍物 (ox,oy,r)*n
      ── 末尾固定维（不受 n_obstacles 影响）──
      [-26] ee_z          末端 Z 位置
      [-25] ee_vz         末端 Z 速度
      [-24] payload_z     负载 Z 位置
      [-23] payload_vz    负载 Z 速度
      [-22] ee_roll       末端 roll
      [-21] ee_roll_vel
      [-20] ee_pitch      末端 pitch
      [-19] ee_pitch_vel
      [-18] ee_yaw        末端 yaw
      [-17] ee_yaw_vel
      [-16] payload_yaw
      [-15] payload_yaw_vel
      [-14] ~ [-8]  q1..q7    当前关节角
      [-7]  ~ [-1]  dq1..dq7  当前关节角速度
    总计：10 + 3*n_obstacles + 26
    """

    def __init__(self, config: dict = None):
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
        cfg_scene = self.config["scene"]
        cfg_plan  = self.config["planning"]
        cfg_noise = self.config["noise"]
        self.cfg_reward = self.config.get("reward", {})
        self.cfg_logic  = self.config.get("step_logic", {})

        # 时间参数
        self.physics_dt      = cfg_sim["physics_dt"]
        self.control_freq_hz = cfg_sim["control_freq_hz"]
        self.control_dt      = 1.0 / self.control_freq_hz
        self.dt              = self.control_dt
        self.sim_steps       = int(self.dt / self.physics_dt)
        self.max_steps       = cfg_sim["max_steps"]
        self.current_step    = 0

        # [ENV-NEW-1] 动作空间：7D 关节角
        self.action_dim        = cfg_space["action_dim"]   # 7
        self.action_space_high = np.array(cfg_space["action_space_high"])
        self.action_space_low  = np.array(cfg_space["action_space_low"])

        # 任务参数
        self.default_start_xy = np.array(cfg_task["default_start_xy"])
        self.default_target   = np.array(cfg_task["default_target_xy"])
        self.target_pos       = self.default_target.copy()
        self.init_position_range = cfg_task["init_position_range"]

        # 场景参数
        self.n_obstacles          = cfg_scene["n_obstacles"]
        self.obstacle_radius_range = cfg_scene["radius_range"]
        self._obstacle_rng        = np.random.default_rng(cfg_scene["seed"])

        self.path_width       = cfg_scene["path_width"]
        self.payload_radius   = cfg_plan["payload_radius"]
        self.planning_margin  = cfg_plan["planning_margin"]
        self.planning_grid_res = cfg_plan["planning_grid_res"]

        # 延迟
        self.latency_steps = cfg_noise["latency_steps"]
        self.action_queue  = deque(maxlen=max(1, self.latency_steps + 1))
        for _ in range(self.latency_steps):
            self.action_queue.append(np.zeros(self.action_dim))

        # 状态维度（10 + 3n + 26）
        self.state_dim = 10 + (self.n_obstacles * 3) + 26

        # 关节角初始值（从 config 读取）
        self._init_q = np.array(self.config["reset"]["init_qpos_arm"], dtype=np.float32)

        # 上一步关节角（用于 smooth_penalty）
        self._prev_q = self._init_q.copy()

        # 关节限位（rad），用于 limit_penalty
        self._q_low  = self.action_space_low.copy()
        self._q_high = self.action_space_high.copy()
        self._q_margin_ratio = float(self.cfg_reward.get("joint_limit_margin", 0.1))

        # XML 与 MuJoCo 初始化
        current_dir      = os.path.dirname(os.path.abspath(__file__))
        self._assets_dir = os.path.join(current_dir, "assets")

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

        # IK Solver（供 controller.py 的 JointSpaceExpert 共享）
        self.ik_solver = NativeIKSolver(self.model, self.data)
        print("✅ IK Solver (MuJoCo Native) 初始化成功！")

        self._reresolve_ids()

        # 内部状态
        self._obstacles     = []
        self._planned_path  = None
        self.current_wp_idx = 0
        self.reached_final  = False
        self.last_dist      = None
        self.last_wp_idx    = -1
        self._wp_just_advanced = False

        # EE 速度数值微分
        self._prev_ee_pos   = np.zeros(3)

        # 渲染
        self.render_mode = cfg_sim["render"]
        self.viewer      = None
        if self.render_mode:
            kw = {}
            if hasattr(self, '_key_callback') and self._key_callback is not None:
                kw['key_callback'] = self._key_callback
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, **kw)

    # ==========================================================================
    # 辅助：ID 解析
    # ==========================================================================

    def _reresolve_ids(self):
        self.prefab_jnt_id  = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("target").id
        self.ee_site_id     = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

    # ==========================================================================
    # 一站式场景生成（保持与原版完全一致）
    # ==========================================================================

    @staticmethod
    def generate_scene_and_trajectory(start_xy, target_xy, base_xml_content,
                                       scene_plan_config, rng=None):
        """A* 路径规划 + 3D 轨迹生成 + XML 注入（与原版逻辑完全一致）。"""
        if rng is None:
            rng = np.random.default_rng()

        start_xy  = np.asarray(start_xy,  dtype=float).reshape(2)
        target_xy = np.asarray(target_xy, dtype=float).reshape(2)

        p_radius  = scene_plan_config["payload_radius"]
        p_margin  = scene_plan_config["planning_margin"]
        min_clearance = p_radius + p_margin

        base_xy      = np.array([0.0, 0.0])
        base_radius1 = 0.20

        n_obs        = scene_plan_config["n_obstacles"]
        r_min, r_max = scene_plan_config["radius_range"]
        path_width   = scene_plan_config["path_width"]

        obstacles = []
        midpoint  = (start_xy + target_xy) / 2
        direction = target_xy - start_xy
        L_path    = np.linalg.norm(direction)

        if L_path > 1e-6:
            direction /= L_path
            perp = np.array([-direction[1], direction[0]])
            attempts = 0
            while len(obstacles) < n_obs and attempts < 500:
                attempts += 1
                t = rng.uniform(0.2, 0.8)
                s = rng.uniform(-path_width / 2, path_width / 2)
                center = midpoint + t * L_path * direction + s * perp
                r      = rng.uniform(r_min, r_max)
                if (np.linalg.norm(center - start_xy)  < r + min_clearance or
                        np.linalg.norm(center - target_xy) < r + min_clearance or
                        np.linalg.norm(center - base_xy)   < r + base_radius1):
                    continue
                ok = True
                for (ox, oy, or_) in obstacles:
                    if np.linalg.norm(center - np.array([ox, oy])) < r + or_ + 0.02:
                        ok = False; break
                if ok:
                    obstacles.append((float(center[0]), float(center[1]), float(r)))

        # A* 规划
        planning_obstacles = obstacles + [(0.0, 0.0, base_radius1)]
        grid_res = scene_plan_config["planning_grid_res"]

        xs = [start_xy[0], target_xy[0]]
        ys = [start_xy[1], target_xy[1]]
        for (ox, oy, r) in planning_obstacles:
            r_eff = r + min_clearance
            xs.extend([ox-r_eff, ox+r_eff]); ys.extend([oy-r_eff, oy+r_eff])

        x_min = min(xs) - scene_plan_config["bounds_margin"]
        x_max = max(xs) + scene_plan_config["bounds_margin"]
        y_min = min(ys) - scene_plan_config["bounds_margin"]
        y_max = max(ys) + scene_plan_config["bounds_margin"]
        nx = max(2, int(np.ceil((x_max - x_min) / grid_res)))
        ny = max(2, int(np.ceil((y_max - y_min) / grid_res)))

        def w2g(x, y):
            return (max(0, min(nx-1, int((x-x_min)/grid_res))),
                    max(0, min(ny-1, int((y-y_min)/grid_res))))
        def g2w(i, j):
            return x_min+(i+.5)*grid_res, y_min+(j+.5)*grid_res

        occ = np.zeros((nx, ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                wx, wy = g2w(i, j)
                for (ox, oy, r) in planning_obstacles:
                    if (wx-ox)**2 + (wy-oy)**2 < (r+min_clearance)**2:
                        occ[i, j] = True; break

        def nearest_free(i0, j0, rad=5):
            if not occ[i0, j0]: return i0, j0
            best, bd = None, None
            for di in range(-rad, rad+1):
                for dj in range(-rad, rad+1):
                    ni, nj = i0+di, j0+dj
                    if 0<=ni<nx and 0<=nj<ny and not occ[ni,nj]:
                        d = di*di+dj*dj
                        if best is None or d < bd: best, bd = (ni,nj), d
            return best

        si = nearest_free(*w2g(*start_xy))  or w2g(*start_xy)
        gi = nearest_free(*w2g(*target_xy)) or w2g(*target_xy)

        open_h = []; g_cost = {si: 0.0}; parent = {}
        heapq.heappush(open_h, (float(np.hypot(*(np.array(g2w(*si))-target_xy))), si))
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        closed = set(); found = False; expansions = 0

        while open_h and expansions < scene_plan_config["max_expansions"]:
            _, cur = heapq.heappop(open_h)
            if cur in closed: continue
            if cur == gi: found = True; break
            closed.add(cur); expansions += 1
            for di, dj in neighbors:
                ni, nj = cur[0]+di, cur[1]+dj
                if not (0<=ni<nx and 0<=nj<ny) or occ[ni,nj]: continue
                step = grid_res if (di==0 or dj==0) else grid_res*1.414
                ng = g_cost[cur] + step; nb = (ni, nj)
                if nb not in g_cost or ng < g_cost[nb]:
                    g_cost[nb] = ng; parent[nb] = cur
                    h = float(np.hypot(*(np.array(g2w(ni,nj))-target_xy)))
                    heapq.heappush(open_h, (ng+h, nb))

        if not found:
            path_2d = np.vstack([start_xy, target_xy])
        else:
            path_idx = []; node = gi
            while node != si:
                path_idx.append(node); node = parent.get(node)
                if node is None: break
            path_idx.append(si); path_idx.reverse()
            path_2d = np.array([g2w(i,j) for (i,j) in path_idx])

        z_cruise = scene_plan_config["payload_z_cruise"]
        path_3d  = [[pt[0], pt[1], z_cruise] for pt in path_2d]
        last_xy  = path_2d[-1]
        for z in np.linspace(z_cruise, scene_plan_config["target_z_descent"],
                             scene_plan_config["num_descent_steps"]+1)[1:]:
            path_3d.append([float(last_xy[0]), float(last_xy[1]), float(z)])
        path_3d = np.array(path_3d)

        # XML 注入
        xml = base_xml_content
        if obstacles:
            xml = xml.replace('  </asset>',
                '    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>\n  </asset>', 1)

        obs_z  = scene_plan_config["obstacle_z_center"]
        obs_hh = scene_plan_config["obstacle_halfheight"]
        obs_bodies = "".join([
            f'    <body name="obstacle_{i}" pos="{x} {y} {obs_z}">\n'
            f'      <geom type="cylinder" size="{r} {obs_hh}" material="obstacle" '
            f'contype="1" conaffinity="1"/>\n    </body>\n'
            for i,(x,y,r) in enumerate(obstacles)
        ])
        path_bodies = "".join([
            f'    <body name="path_pt_{i}" pos="{p[0]} {p[1]} {p[2]}">\n'
            f'      <geom type="sphere" size="0.01" rgba="0 0 1 1" '
            f'contype="0" conaffinity="0"/>\n    </body>\n'
            for i,p in enumerate(path_3d)
        ])
        ep_z = obs_z + obs_hh + scene_plan_config["endpoint_z_offset"]
        ep_bodies = (
            f'    <body name="path_start" pos="{start_xy[0]} {start_xy[1]} {ep_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n    </body>\n'
            f'    <body name="path_goal" pos="{target_xy[0]} {target_xy[1]} {ep_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n    </body>\n'
        )
        repl = ('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
                + obs_bodies + path_bodies + ep_bodies + '    ')
        xml  = xml.replace(
            '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    ',
            repl, 1
        )
        return obstacles, path_3d, xml

    # ==========================================================================
    # reset()
    # ==========================================================================

    def reset(self):
        """重置环境，返回初始观测。"""
        cfg_task  = self.config["task"]
        cfg_scene = self.config["scene"]
        cfg_plan  = self.config["planning"]
        cfg_reset = self.config["reset"]

        # 随机化起始 XY
        noise = self._obstacle_rng.uniform(
            -self.init_position_range, self.init_position_range, size=2
        )
        start_xy  = self.default_start_xy + noise
        target_xy = np.array(cfg_task["default_target_xy"])
        self.target_pos = target_xy.copy()

        # 场景生成
        scene_plan_cfg = {**cfg_scene, **cfg_plan}
        obstacles, path_3d, new_xml = self.generate_scene_and_trajectory(
            start_xy, target_xy, self._base_xml_content, scene_plan_cfg, self._obstacle_rng
        )
        self._obstacles    = obstacles
        self._planned_path = path_3d

        # 重载 XML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml',
                                          delete=False, encoding='utf-8') as f:
            f.write(new_xml)
            tmp_path = f.name
        try:
            self.model = mujoco.MjModel.from_xml_path(tmp_path)
            self.data  = mujoco.MjData(self.model)
            self.model.opt.timestep = self.physics_dt
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # 更新 IK Solver 的模型引用
        self.ik_solver.model = self.model
        self.ik_solver.data  = self.data
        # 重新查找 site/body ID（新模型）
        self.ik_solver.obj_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ik_solver.target_frame
        )
        self.ik_solver.is_site = True
        if self.ik_solver.obj_id == -1:
            self.ik_solver.obj_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.ik_solver.target_frame
            )
            self.ik_solver.is_site = False
        self.ik_solver.q_min = self.model.jnt_range[:7, 0]
        self.ik_solver.q_max = self.model.jnt_range[:7, 1]
        self.ik_solver.jnt_limited = self.model.jnt_limited[:7]
        self.ik_solver.q_margin = 0.15 * (self.ik_solver.q_max - self.ik_solver.q_min)

        self._reresolve_ids()

        # 初始化 MuJoCo 状态
        mujoco.mj_resetData(self.model, self.data)
        init_q = np.array(cfg_reset["init_qpos_arm"], dtype=np.float64)
        self.data.qpos[:7] = init_q

        # 负载初始位姿
        pref_qpos = np.array(cfg_reset["init_qpos_prefab"], dtype=np.float64)
        pref_jnt  = self.model.body("prefab").jntadr[0]
        dof_addr  = self.model.jnt_dofadr[pref_jnt]
        qpos_addr = self.model.jnt_qposadr[pref_jnt]
        self.data.qpos[qpos_addr:qpos_addr+7] = pref_qpos

        # 物理预热
        for _ in range(cfg_reset["warmup_steps"]):
            self.data.ctrl[:7] = init_q
            mujoco.mj_step(self.model, self.data)

        # 状态重置
        self.current_step    = 0
        self.current_wp_idx  = 0
        self.reached_final   = False
        self.last_dist       = None
        self.last_wp_idx     = -1
        self._wp_just_advanced = False
        self._prev_q         = self.data.qpos[:7].copy().astype(np.float32)

        # 初始化 EE 速度估计
        ee_pos = self._get_ee_pos()
        self._prev_ee_pos = ee_pos.copy()

        # 延迟队列重置
        self.action_queue.clear()
        for _ in range(self.latency_steps):
            self.action_queue.append(init_q.copy())

        # 渲染器重建
        if self.render_mode:
            if self.viewer is not None:
                try: self.viewer.close()
                except Exception: pass
            kw = {}
            if hasattr(self, '_key_callback') and self._key_callback is not None:
                kw['key_callback'] = self._key_callback
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, **kw)

        return self._get_obs()

    # ==========================================================================
    # step()
    # ==========================================================================

    def step(self, action: np.ndarray):
        """
        [ENV-NEW-1] 直接执行 7D 关节角目标。

        Args:
            action: (7,) 关节角目标（rad），已 clamp 到关节限位
        Returns:
            Gymnasium 标准 5 元组 (obs, reward, terminated, truncated, info)
        """
        # 安全 clamp
        action = np.clip(action, self.action_space_low, self.action_space_high)

        # 动作延迟队列
        self.action_queue.append(action.copy())
        effective_q = self.action_queue[0].copy()

        # 直接下发关节角目标到位置执行器
        self.data.ctrl[:7] = effective_q

        # 物理步进
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        # NaN 检测
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            obs = self._get_obs()
            return obs, -10.0, True, False, {"is_success": False, "nan_detected": True}

        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

        # 获取观测
        obs = self._get_obs()

        # 负载 3D 位置
        payload_z           = self.data.body('prefab').xpos[2]
        payload_xy          = np.array([obs[4], obs[5]])
        current_payload_pos = np.array([payload_xy[0], payload_xy[1], payload_z])

        # 航点追踪状态机
        if self._planned_path is not None and not self.reached_final:
            target_wp  = self._planned_path[self.current_wp_idx]
            dist_to_wp = np.linalg.norm(current_payload_pos - target_wp)

            if self.last_wp_idx != self.current_wp_idx:
                self.last_dist   = None
                self.last_wp_idx = self.current_wp_idx

            self.last_dist = dist_to_wp

            total_wps  = len(self._planned_path)
            rem_wps    = total_wps - 1 - self.current_wp_idx
            look_ahead = 0.04 if rem_wps <= 2 else self.config["step_logic"]["look_ahead_dist"]

            if dist_to_wp < look_ahead:
                if self.current_wp_idx < total_wps - 1:
                    self.current_wp_idx  += 1
                    self._wp_just_advanced = True
                else:
                    self.reached_final = True

        # 奖励计算
        current_q = self.data.qpos[:7].copy().astype(np.float32)
        reward, done, success, is_collision = self._compute_reward(
            action, current_q, self._prev_q
        )
        self._prev_q = current_q

        self.current_step += 1
        if self.current_step >= self.config["sim"]["max_steps"]:
            if not done:
                reward += self.config["reward"]["timeout_penalty"]
            done = True

        info = {
            "is_success":     success,
            "current_wp_idx": self.current_wp_idx,
            "reached_final":  self.reached_final,
            "is_collision":   is_collision,
        }
        return obs, reward, done, False, info

    # ==========================================================================
    # _compute_reward()
    # ==========================================================================

    def _compute_reward(self, action: np.ndarray, current_q: np.ndarray, prev_q: np.ndarray):
        """
        奖励计算（新增关节空间惩罚）。
        """
        reward = 0.0; done = False; success = False; is_collision = False
        cfg_rwd = self.config["reward"]

        obs        = self._get_obs()
        payload_xy = obs[4:6]
        payload_vxy = obs[6:8]
        payload_z  = self.data.body('prefab').xpos[2]
        dof_idx    = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vz = self.data.qvel[dof_idx + 2]
        payload_vel_norm = np.linalg.norm(np.append(payload_vxy, payload_vz))

        ee_xy = self._get_ee_pos()[:2]

        # ── 每步惩罚 ───────────────────────────────────────────────────────
        reward += float(cfg_rwd.get("step_penalty", -0.005))

        # 速度惩罚
        vel_pen = cfg_rwd.get("velocity_penalty_coef", 0.005) * payload_vel_norm
        reward -= float(np.clip(vel_pen, 0, 0.3))

        # 摆角惩罚
        swing = float(np.linalg.norm(ee_xy - payload_xy))
        reward -= cfg_rwd.get("swing_penalty_coef", 0.02) * float(np.clip(swing, 0, 0.1))

        # [ENV-NEW-3] 关节角平滑惩罚（防抖）
        joint_delta = float(np.linalg.norm(current_q - prev_q))
        reward += cfg_rwd.get("joint_smooth_penalty", -0.002) * joint_delta

        # [ENV-NEW-3] 关节角极限惩罚
        q_range = self._q_high - self._q_low
        q_margin = self._q_margin_ratio * q_range
        n_near_limit = 0
        for j in range(7):
            if (current_q[j] > self._q_high[j] - q_margin[j] or
                    current_q[j] < self._q_low[j] + q_margin[j]):
                n_near_limit += 1
        if n_near_limit > 0:
            reward += cfg_rwd.get("joint_limit_penalty", -0.05) * n_near_limit

        # ── 终止条件（优先级：成功 > 碰撞 > 坠毁 > 出界）──────────────────
        if self.reached_final:
            vel_xy        = float(np.linalg.norm(payload_vxy))
            dist_to_final = float(np.linalg.norm(payload_xy - self.target_pos))
            if dist_to_final < 0.03 and vel_xy < 0.1 and abs(payload_vz) < 0.2:
                reward += cfg_rwd.get("success_bonus", 3.0)
                success = True; done = True
            else:
                reward += cfg_rwd.get("crash_penalty", -3.0)
                done = True
            return reward, done, success, is_collision

        # 碰撞检测
        for (ox, oy, orad) in self._obstacles:
            if float(np.linalg.norm(payload_xy - np.array([ox, oy]))) < (orad + self.payload_radius):
                reward += cfg_rwd.get("collision_penalty", -3.0)
                done = True; is_collision = True
                return reward, done, success, is_collision

        # 近机械臂基座
        if float(np.linalg.norm(payload_xy - np.array([0, 0]))) < 0.03:
            reward += cfg_rwd.get("collision_penalty", -3.0)
            done = True; is_collision = True
            return reward, done, success, is_collision

        # 坠毁
        cfg_logic = self.config["step_logic"]
        if (payload_z < cfg_logic["crash_z_threshold"] and
                payload_vz < cfg_logic["crash_vz_threshold"]):
            reward += cfg_rwd.get("crash_penalty", -3.0)
            done = True
            return reward, done, success, is_collision

        # 航点里程碑奖励
        if getattr(self, '_wp_just_advanced', False):
            reward += cfg_rwd.get("waypoint_bonus", 0.1)
            self._wp_just_advanced = False

        return reward, done, success, is_collision

    # ==========================================================================
    # _get_obs()
    # ==========================================================================

    def _get_obs(self) -> np.ndarray:
        """
        [ENV-NEW-2] 新增末尾 14 维关节状态。
        总计：10 + 3*n_obstacles + 26 维
        """
        # EE 状态（从 MuJoCo FK 读取）
        ee_pos  = self._get_ee_pos()
        ee_vel  = (ee_pos - self._prev_ee_pos) / self.dt
        self._prev_ee_pos = ee_pos.copy()
        ee_x, ee_y, ee_z   = ee_pos
        ee_vx, ee_vy, ee_vz = ee_vel

        # EE 姿态（从 site xmat 提取欧拉角）
        site_mat  = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        r_obj     = R.from_matrix(site_mat)
        ee_euler  = r_obj.as_euler('xyz')
        ee_roll, ee_pitch, ee_yaw = ee_euler

        # 负载状态
        payload_x = self.data.body('prefab').xpos[0]
        payload_y = self.data.body('prefab').xpos[1]
        payload_z = self.data.body('prefab').xpos[2]
        dof_idx   = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vx = self.data.qvel[dof_idx]
        payload_vy = self.data.qvel[dof_idx + 1]
        payload_vz = self.data.qvel[dof_idx + 2]

        rel_tx = self.target_pos[0] - payload_x
        rel_ty = self.target_pos[1] - payload_y

        # 障碍物数据
        obs_data = []
        for (ox, oy, r) in self._obstacles:
            obs_data.extend([ox, oy, r])
        target_len = self.n_obstacles * 3
        while len(obs_data) < target_len:
            obs_data.append(0.0)

        # [ENV-NEW-2] 关节状态（7 关节角 + 7 关节角速度）
        joint_q  = self.data.qpos[:7].copy().astype(np.float32)
        joint_dq = self.data.qvel[:7].copy().astype(np.float32)

        return np.array(
            [ee_x, ee_y, ee_vx, ee_vy,
             payload_x, payload_y, payload_vx, payload_vy,
             rel_tx, rel_ty]
            + obs_data[:target_len]
            + [ee_z, ee_vz, payload_z, payload_vz,
               ee_roll, 0.0, ee_pitch, 0.0,
               ee_yaw, 0.0, 0.0, 0.0]
            + list(joint_q)
            + list(joint_dq),
            dtype=np.float32
        )

    # ==========================================================================
    # 辅助工具
    # ==========================================================================

    def _get_ee_pos(self) -> np.ndarray:
        """从 MuJoCo FK 读取末端执行器位置（比积分更准确）。"""
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_planned_path(self):
        return self._planned_path

    def close(self):
        if self.viewer is not None:
            try: self.viewer.close()
            except Exception: pass


# ==============================================================================
# NativeIKSolver（与原版完全一致，供 controller.py 使用）
# ==============================================================================

class NativeIKSolver:
    def __init__(self, mj_model, mj_data):
        self.model = mj_model
        self.data  = mj_data
        self.target_frame = "link7"

        self.obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.target_frame)
        self.is_site = True
        if self.obj_id == -1:
            self.obj_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_frame)
            self.is_site = False
        if self.obj_id == -1:
            raise ValueError(f"找不到 '{self.target_frame}'")

        self.damping        = 2e-2
        self.nullspace_gain = 0.05
        self.w_pos          = 1.0
        self.w_rot_base     = 0.15
        self.q_min          = self.model.jnt_range[:7, 0]
        self.q_max          = self.model.jnt_range[:7, 1]
        self.jnt_limited    = self.model.jnt_limited[:7]
        self.q_margin       = 0.15 * (self.q_max - self.q_min)

    def solve_4d(self, current_q, target_x, target_y, target_z, target_yaw):
        roll, pitch, yaw = np.pi, 0.0, target_yaw
        cx, sx = np.cos(roll), np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw), np.sin(yaw)
        R_x = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        R_y = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        R_z = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        Rmat = R_z @ R_y @ R_x

        target_pos  = np.array([target_x, target_y, target_z])
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, Rmat.flatten())

        backup_qpos = self.data.qpos.copy()
        q_guess     = current_q.copy()

        for _ in range(5):
            self.data.qpos[:7] = q_guess
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            if self.is_site:
                cp = self.data.site_xpos[self.obj_id]
                cm = self.data.site_xmat[self.obj_id].reshape(3,3)
            else:
                cp = self.data.xpos[self.obj_id]
                cm = self.data.xmat[self.obj_id].reshape(3,3)

            cq = np.zeros(4); mujoco.mju_mat2Quat(cq, cm.flatten())
            pe = target_pos - cp
            re = np.zeros(3); nq = np.zeros(4); eq = np.zeros(4)
            mujoco.mju_negQuat(nq, cq)
            mujoco.mju_mulQuat(eq, target_quat, nq)
            if eq[0] < 0: eq = -eq
            mujoco.mju_quat2Vel(re, eq, 1.0)

            pn = np.linalg.norm(pe); rn = np.linalg.norm(re)
            if pn < 1e-3 and rn < 1e-2: break

            wr = self.w_rot_base * (0.02 / (0.02 + pn))
            if pn > 0.05: pe = (pe/pn)*0.05
            if rn > 0.15: re = (re/rn)*0.15

            jacp = np.zeros((3, self.model.nv)); jacr = np.zeros((3, self.model.nv))
            if self.is_site:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.obj_id)
            else:
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.obj_id)
            J = np.vstack([jacp, jacr])[:, :7]

            J_w     = J.copy(); J_w[:3] *= self.w_pos; J_w[3:] *= wr
            err_w   = np.concatenate([pe*self.w_pos, re*wr])
            JJT_w   = J_w @ J_w.T
            diag    = (self.damping**2) * np.eye(6)
            dq      = J_w.T @ np.linalg.solve(JJT_w+diag, err_w)

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

            dq = np.clip(dq, -0.1, 0.1)
            q_guess += dq
            for j in range(7):
                if self.jnt_limited[j]:
                    q_guess[j] = np.clip(q_guess[j], self.q_min[j], self.q_max[j])

        self.data.qpos[:] = backup_qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return q_guess


# ==============================================================================
# 便捷函数：从 config 构建环境
# ==============================================================================

def make_env(config: dict = None) -> CableRobotEnvWithObstacles:
    return CableRobotEnvWithObstacles(config=config)