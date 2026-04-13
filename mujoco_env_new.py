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

# ==============================================================================
# mujoco_env_new.py — 关节空间控制版索驱动机器人环境（修复版）
#
# ══════════════════════════════════════════════════════════════════════════════
# 修复清单
# ══════════════════════════════════════════════════════════════════════════════
#
# [FIX-E1] NativeIKSolver 已移至 controller.py，env 直接导入复用
#          原版在 env 底部定义了独立的 NativeIKSolver，
#          controller.py 中也有相同的 NativeIKSolver，两者并存且代码不同步。
#          修复：env 从 controller.py import NativeIKSolver，统一维护。
#
# [FIX-E2] reset() 中 IK Solver 模型更新方式不完整
#          原版手动逐字段赋值（obj_id/is_site/q_min...），
#          容易遗漏字段（如 scratch 副本未更新）。
#          修复：调用 ik_solver.update_model(model, data)，
#          由 NativeIKSolver 自身负责完整更新。
#
# [FIX-E3] _get_obs() 中 EE 速度数值微分的时机问题
#          _get_obs 是纯查询函数，但它每次调用都会更新 _prev_ee_pos，
#          产生副作用：同一步如果被调用两次（step 和 _compute_reward 各调一次），
#          第二次调用时 ee_vel 会变成 0（因为 pos 没变但 _prev 已更新）。
#          修复：
#            a) 从 step() 中分离 EE 速度更新逻辑（只在 step 末尾更新 _prev）
#            b) _get_obs 使用成员变量 _ee_vel_cache 存储当前步速度，
#               只在 step() 完成物理步进后更新一次
#
# [FIX-E4] _compute_reward() 内部调用 _get_obs() 导致双重更新（与 FIX-E3 关联）
#          step() 调用 _get_obs() → 存 obs
#          step() 调用 _compute_reward() → 内部又调 _get_obs() → _prev_ee_pos 再次更新
#          修复：_compute_reward 接收 obs 作为参数，不再内部调用 _get_obs()
#
# [FIX-E5] 观测布局注释与代码不一致
#          文档说 ee_roll_vel / ee_pitch_vel / ee_yaw_vel 已填充，
#          但代码中这三个速度值实际全写 0.0（placeholder）。
#          修复：计算真实的欧拉角速度（从 data.qvel 通过角速度雅可比计算），
#          或如保留 0 则注释中明确标注"占位，当前未实现"。
#          本版选择：通过 site_xmat 当前帧与上一帧的差分估计角速度，
#          较为准确且无需雅可比。
#
# [FIX-E6] step() 中 action_queue 与 latency_steps 逻辑
#          action_queue maxlen = latency_steps + 1
#          append 后取 [0]（队首），这会在 latency=1 时给出上一步的动作——正确。
#          但初始化时填 np.zeros(action_dim)，而正确初始化应填 init_q（初始关节角）。
#          否则第一步执行 zeros 而非 init_q，导致机械臂从初始位置突然跳到 0 关节角。
#          修复：reset() 中用 init_q 填充 action_queue。
#
# [FIX-E7] prefab dof_idx 计算：jnt_dofadr vs qposadr
#          原版：dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
#          qvel[dof_idx + 2] 取负载 Z 速度。
#          jnt_dofadr 指向速度自由度起始偏移，对于 free joint：
#            qvel[dof_idx:dof_idx+3] = 平移速度，[dof_idx+3:dof_idx+6] = 角速度
#          这是正确的。✓ 无需修复。
#          但注意：qpos 的地址需要用 jnt_qposadr（free joint qpos 有 7 个分量）。
#          原版 payload_vx/vy/vz 用 dof_idx + 0/1/2 是正确的。
#
# [FIX-E8] 奖励中的 joint_smooth_penalty 系数符号问题
#          cfg_rwd.get("joint_smooth_penalty", -0.002) 本身是负数，
#          代码 reward += coef * joint_delta（delta 为正），
#          = reward += (-0.002) * (正数) → 正确减少奖励。✓
#          但如果 config 中用户误填正数，则会给奖励。
#          修复：强制取负数绝对值再施加。
#
# ══════════════════════════════════════════════════════════════════════════════

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
# [FIX-E1] 统一从 controller 导入 NativeIKSolver
from controller import NativeIKSolver


class CableRobotEnvWithObstacles:
    """
    关节空间控制版索驱动机器人环境。

    动作空间：7D 关节角目标 [q1..q7]（rad）

    观测布局（总计 10 + 3*n_obstacles + 26 维）：
      [0]   ee_x          末端执行器 X（FK）
      [1]   ee_y          末端执行器 Y（FK）
      [2]   ee_vx         末端 X 速度（数值微分）
      [3]   ee_vy         末端 Y 速度
      [4]   payload_x     负载 X
      [5]   payload_y     负载 Y
      [6]   payload_vx    负载 X 速度
      [7]   payload_vy    负载 Y 速度
      [8]   rel_tx        目标 X - 负载 X
      [9]   rel_ty        目标 Y - 负载 Y
      [10 ~ 10+3n-1]  障碍物 (ox, oy, r) × n
      ── 末尾固定 26 维（负索引）──
      [-26] ee_z          末端 Z（FK）
      [-25] ee_vz         末端 Z 速度
      [-24] payload_z     负载 Z
      [-23] payload_vz    负载 Z 速度
      [-22] ee_roll       末端 roll（rad，ZYX Euler）
      [-21] ee_roll_vel   末端 roll 速度（估计）
      [-20] ee_pitch      末端 pitch
      [-19] ee_pitch_vel  末端 pitch 速度
      [-18] ee_yaw        末端 yaw
      [-17] ee_yaw_vel    末端 yaw 速度
      [-16] payload_yaw   负载 yaw（当前为 0）
      [-15] payload_yaw_vel（当前为 0）
      [-14...-8]  joint_q[0..6]   当前 7 关节角（rad）
      [-7...-1]   joint_dq[0..6]  当前 7 关节角速度（rad/s）
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

        # ── 时间参数 ─────────────────────────────────────────────────────────
        self.physics_dt      = cfg_sim["physics_dt"]
        self.control_freq_hz = cfg_sim["control_freq_hz"]
        self.control_dt      = 1.0 / self.control_freq_hz
        self.dt              = self.control_dt
        self.sim_steps       = int(self.dt / self.physics_dt)
        self.max_steps       = cfg_sim["max_steps"]
        self.current_step    = 0

        # ── 动作空间（7D 关节角）────────────────────────────────────────────
        self.action_dim        = cfg_space["action_dim"]
        self.action_space_high = np.array(cfg_space["action_space_high"])
        self.action_space_low  = np.array(cfg_space["action_space_low"])

        # ── 任务参数 ─────────────────────────────────────────────────────────
        self.default_start_xy  = np.array(cfg_task["default_start_xy"])
        self.default_target    = np.array(cfg_task["default_target_xy"])
        self.target_pos        = self.default_target.copy()
        self.init_position_range = cfg_task["init_position_range"]

        # ── 场景参数 ─────────────────────────────────────────────────────────
        self.n_obstacles          = cfg_scene["n_obstacles"]
        self.obstacle_radius_range = cfg_scene["radius_range"]
        self._obstacle_rng        = np.random.default_rng(cfg_scene["seed"])
        self.path_width           = cfg_scene["path_width"]
        self.payload_radius       = cfg_plan["payload_radius"]
        self.planning_margin      = cfg_plan["planning_margin"]
        self.planning_grid_res    = cfg_plan["planning_grid_res"]

        # ── 动作延迟队列 ─────────────────────────────────────────────────────
        self.latency_steps = cfg_noise["latency_steps"]
        # maxlen = latency+1：append 后取 [0] 实现 latency 步延迟
        self.action_queue  = deque(maxlen=max(1, self.latency_steps + 1))
        init_q_default = np.array(self.config["reset"]["init_qpos_arm"],
                                   dtype=np.float32)
        for _ in range(max(1, self.latency_steps + 1)):
            self.action_queue.append(init_q_default.copy())

        # ── 状态维度 ─────────────────────────────────────────────────────────
        self.state_dim = 10 + (self.n_obstacles * 3) + 26

        # ── 关节惩罚参数 ─────────────────────────────────────────────────────
        self._prev_q        = init_q_default.copy()
        self._q_low         = self.action_space_low.copy()
        self._q_high        = self.action_space_high.copy()
        self._q_margin_ratio = float(self.cfg_reward.get("joint_limit_margin", 0.1))

        # ── XML 与 MuJoCo 初始化 ─────────────────────────────────────────────
        current_dir      = os.path.dirname(os.path.abspath(__file__))
        self._assets_dir = os.path.join(current_dir, "assets")

        from assets.generate_four_cables_with_plate import main as gen_rope
        gen_rope()

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

        # ── IK Solver（[FIX-E1] 使用 controller.py 中统一维护的版本）────────
        self.ik_solver = NativeIKSolver(self.model, self.data)
        print("✅ IK Solver 初始化成功！")

        self._reresolve_ids()

        # ── 内部状态 ─────────────────────────────────────────────────────────
        self._obstacles        = []
        self._planned_path     = None
        self.current_wp_idx    = 0
        self.reached_final     = False
        self.last_dist         = None
        self.last_wp_idx       = -1
        self._wp_just_advanced = False

        # [FIX-E3] EE 速度缓存（只在 step 末尾更新一次）
        self._prev_ee_pos  = np.zeros(3)
        self._ee_vel_cache = np.zeros(3)  # 当前步速度缓存

        # [FIX-E5] EE 欧拉角缓存（用于角速度估计）
        self._prev_ee_euler    = np.zeros(3)
        self._ee_euler_vel_cache = np.zeros(3)

        # ── 渲染器 ───────────────────────────────────────────────────────────
        self.render_mode = cfg_sim["render"]
        self.viewer      = None
        if self.render_mode:
            self._launch_viewer()

    # ==========================================================================
    # 辅助方法
    # ==========================================================================

    def _reresolve_ids(self):
        """重新解析 MuJoCo body/site ID（每次场景重载后调用）。"""
        self.prefab_jnt_id  = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("target").id
        self.ee_site_id     = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )

    def _get_ee_pos(self) -> np.ndarray:
        """从 MuJoCo FK 读取末端执行器 3D 位置。"""
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_ee_mat(self) -> np.ndarray:
        """从 MuJoCo FK 读取末端执行器旋转矩阵 (3×3)。"""
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def _launch_viewer(self):
        kw = {}
        if hasattr(self, '_key_callback') and self._key_callback is not None:
            kw['key_callback'] = self._key_callback
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, **kw)

    def get_planned_path(self):
        return self._planned_path

    def close(self):
        if self.viewer is not None:
            try: self.viewer.close()
            except Exception: pass

    # ==========================================================================
    # 一站式场景生成（A* + 3D 轨迹 + XML，与原版逻辑一致）
    # ==========================================================================

    @staticmethod
    def generate_scene_and_trajectory(start_xy, target_xy, base_xml_content,
                                       scene_plan_config, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        start_xy  = np.asarray(start_xy,  dtype=float).reshape(2)
        target_xy = np.asarray(target_xy, dtype=float).reshape(2)

        p_radius  = scene_plan_config["payload_radius"]
        p_margin  = scene_plan_config["planning_margin"]
        min_clr   = p_radius + p_margin
        base_xy   = np.array([0.0, 0.0])
        base_r1   = 0.20
        n_obs     = scene_plan_config["n_obstacles"]
        r_min, r_max = scene_plan_config["radius_range"]
        path_width   = scene_plan_config["path_width"]

        obstacles = []
        direction = target_xy - start_xy
        L_path    = np.linalg.norm(direction)
        if L_path > 1e-6:
            direction /= L_path
            perp = np.array([-direction[1], direction[0]])
            attempts = 0
            while len(obstacles) < n_obs and attempts < 500:
                attempts += 1
                t      = rng.uniform(0.2, 0.8)
                s      = rng.uniform(-path_width/2, path_width/2)
                center = start_xy + t * L_path * direction + s * perp
                r      = rng.uniform(r_min, r_max)
                if (np.linalg.norm(center-start_xy)  < r+min_clr or
                        np.linalg.norm(center-target_xy) < r+min_clr or
                        np.linalg.norm(center-base_xy)   < r+base_r1):
                    continue
                ok = all(np.linalg.norm(center-np.array([ox,oy])) >= r+or_+0.02
                         for (ox,oy,or_) in obstacles)
                if ok:
                    obstacles.append((float(center[0]), float(center[1]), float(r)))

        planning_obs = obstacles + [(0.0, 0.0, base_r1)]
        grid_res = scene_plan_config["planning_grid_res"]
        xs = [start_xy[0], target_xy[0]]
        ys = [start_xy[1], target_xy[1]]
        for (ox,oy,r) in planning_obs:
            re = r + min_clr
            xs.extend([ox-re, ox+re]); ys.extend([oy-re, oy+re])
        x_min = min(xs) - scene_plan_config["bounds_margin"]
        x_max = max(xs) + scene_plan_config["bounds_margin"]
        y_min = min(ys) - scene_plan_config["bounds_margin"]
        y_max = max(ys) + scene_plan_config["bounds_margin"]
        nx = max(2, int(np.ceil((x_max-x_min)/grid_res)))
        ny = max(2, int(np.ceil((y_max-y_min)/grid_res)))

        def w2g(x, y):
            return (max(0,min(nx-1,int((x-x_min)/grid_res))),
                    max(0,min(ny-1,int((y-y_min)/grid_res))))
        def g2w(i, j):
            return x_min+(i+.5)*grid_res, y_min+(j+.5)*grid_res

        occ = np.zeros((nx,ny), dtype=bool)
        for i in range(nx):
            for j in range(ny):
                wx,wy = g2w(i,j)
                if any((wx-ox)**2+(wy-oy)**2<(r+min_clr)**2
                       for (ox,oy,r) in planning_obs):
                    occ[i,j] = True

        def nearest_free(i0, j0, rad=5):
            if not occ[i0,j0]: return i0,j0
            best,bd = None,None
            for di in range(-rad,rad+1):
                for dj in range(-rad,rad+1):
                    ni,nj = i0+di,j0+dj
                    if 0<=ni<nx and 0<=nj<ny and not occ[ni,nj]:
                        d = di*di+dj*dj
                        if best is None or d<bd: best,bd=(ni,nj),d
            return best

        si = nearest_free(*w2g(*start_xy))  or w2g(*start_xy)
        gi = nearest_free(*w2g(*target_xy)) or w2g(*target_xy)

        open_h=[]; g_cost={si:0.0}; parent={}
        heapq.heappush(open_h,(float(np.hypot(*(np.array(g2w(*si))-target_xy))),si))
        nbrs=[(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
        closed=set(); found=False; exp=0

        while open_h and exp<scene_plan_config["max_expansions"]:
            _,cur=heapq.heappop(open_h)
            if cur in closed: continue
            if cur==gi: found=True; break
            closed.add(cur); exp+=1
            for di,dj in nbrs:
                ni,nj=cur[0]+di,cur[1]+dj
                if not(0<=ni<nx and 0<=nj<ny) or occ[ni,nj]: continue
                step=grid_res if(di==0 or dj==0) else grid_res*1.414
                ng=g_cost[cur]+step; nb=(ni,nj)
                if nb not in g_cost or ng<g_cost[nb]:
                    g_cost[nb]=ng; parent[nb]=cur
                    h=float(np.hypot(*(np.array(g2w(ni,nj))-target_xy)))
                    heapq.heappush(open_h,(ng+h,nb))

        if not found:
            path_2d=np.vstack([start_xy,target_xy])
        else:
            idx=[]; node=gi
            while node!=si:
                idx.append(node); node=parent.get(node)
                if node is None: break
            idx.append(si); idx.reverse()
            path_2d=np.array([g2w(i,j) for (i,j) in idx])

        z_cruise = scene_plan_config["payload_z_cruise"]
        num_lift  = scene_plan_config.get("num_lift_steps", 5)
        path_3d   = []
        first_xy  = path_2d[0]
        for z in np.linspace(0.11, z_cruise, num_lift+1)[1:]:
            path_3d.append([float(first_xy[0]), float(first_xy[1]), float(z)])
        for pt in path_2d[1:]:
            path_3d.append([float(pt[0]), float(pt[1]), float(z_cruise)])
        last_xy = path_2d[-1]
        for z in np.linspace(z_cruise, scene_plan_config["target_z_descent"],
                             scene_plan_config["num_descent_steps"]+1)[1:]:
            path_3d.append([float(last_xy[0]), float(last_xy[1]), float(z)])
        path_3d = np.array(path_3d)

        xml = base_xml_content
        if obstacles:
            xml = xml.replace('  </asset>',
                '    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>\n  </asset>',1)
        obs_z=scene_plan_config["obstacle_z_center"]
        obs_hh=scene_plan_config["obstacle_halfheight"]
        obs_bodies="".join([
            f'    <body name="obstacle_{i}" pos="{x} {y} {obs_z}">\n'
            f'      <geom type="cylinder" size="{r} {obs_hh}" material="obstacle" '
            f'contype="1" conaffinity="1"/>\n    </body>\n'
            for i,(x,y,r) in enumerate(obstacles)])
        path_bodies="".join([
            f'    <body name="path_pt_{i}" pos="{p[0]} {p[1]} {p[2]}">\n'
            f'      <geom type="sphere" size="0.01" rgba="0 0 1 1" '
            f'contype="0" conaffinity="0"/>\n    </body>\n'
            for i,p in enumerate(path_3d)])
        ep_z=obs_z+obs_hh+scene_plan_config["endpoint_z_offset"]
        ep_b=(
            f'    <body name="path_start" pos="{start_xy[0]} {start_xy[1]} {ep_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" '
            f'contype="0" conaffinity="0"/>\n    </body>\n'
            f'    <body name="path_goal" pos="{target_xy[0]} {target_xy[1]} {ep_z}">\n'
            f'      <geom type="sphere" size="0.012" rgba="1 0 0 1" '
            f'contype="0" conaffinity="0"/>\n    </body>\n'
        )
        repl=('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
              +obs_bodies+path_bodies+ep_b+'    ')
        xml=xml.replace(
            '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    ',
            repl,1)
        return obstacles, path_3d, xml

    # ==========================================================================
    # reset()
    # ==========================================================================

    def reset(self):
        cfg_task  = self.config["task"]
        cfg_scene = self.config["scene"]
        cfg_plan  = self.config["planning"]
        cfg_reset = self.config["reset"]

        noise    = self._obstacle_rng.uniform(
            -self.init_position_range, self.init_position_range, size=2)
        start_xy  = self.default_start_xy + noise
        target_xy = np.array(cfg_task["default_target_xy"])
        self.target_pos = target_xy.copy()

        scene_plan_cfg = {**cfg_scene, **cfg_plan}
        obstacles, path_3d, new_xml = self.generate_scene_and_trajectory(
            start_xy, target_xy, self._base_xml_content,
            scene_plan_cfg, self._obstacle_rng
        )
        self._obstacles    = obstacles
        self._planned_path = path_3d

        # 重载 XML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml',
                                          dir=self._assets_dir,
                                          delete=False, encoding='utf-8') as f:
            f.write(new_xml)
            tmp_path = f.name
        try:
            self.model = mujoco.MjModel.from_xml_path(tmp_path)
            self.data  = mujoco.MjData(self.model)
            self.model.opt.timestep = self.physics_dt
        finally:
            try: os.remove(tmp_path)
            except OSError: pass

        # [FIX-E2] 统一调用 update_model，确保 IK scratch 也更新
        self.ik_solver.update_model(self.model, self.data)
        self._reresolve_ids()

        # 初始化仿真状态
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        # 初始化 free/ball joint 四元数，防止非法初始状态
        for j in range(self.model.njnt):
            adr   = self.model.jnt_qposadr[j]
            jtype = self.model.jnt_type[j]
            if jtype == mujoco.mjtJoint.mjJNT_BALL:
                self.data.qpos[adr:adr+4] = [1,0,0,0]
            elif jtype == mujoco.mjtJoint.mjJNT_FREE:
                self.data.qpos[adr+3:adr+7] = [1,0,0,0]

        # 用 IK 计算初始关节角（末端在 start_xy 正上方）
        seed_q   = np.array(cfg_reset["init_qpos_arm"], dtype=np.float64)
        start_z  = float(cfg_reset.get("mocap_init_z", 0.4))
        init_q   = self.ik_solver.solve_4d(
            current_q=seed_q,
            target_x=start_xy[0], target_y=start_xy[1],
            target_z=start_z, target_yaw=0.0
        )
        if init_q is None or np.any(np.isnan(init_q)):
            init_q = seed_q.copy()
        self.data.qpos[:7] = init_q

        # 负载初始位姿
        pref_qpos = np.array(cfg_reset["init_qpos_prefab"], dtype=np.float64)
        pref_qpos[0] = start_xy[0]
        pref_qpos[1] = start_xy[1]
        pref_jnt  = self.model.body("prefab").jntadr[0]
        qpos_addr = self.model.jnt_qposadr[pref_jnt]
        self.data.qpos[qpos_addr:qpos_addr+7] = pref_qpos

        # 物理预热：锁定状态防重力扰动
        arm_hold    = init_q.copy()
        prefab_hold = pref_qpos.copy()
        self.data.ctrl[:7] = arm_hold
        for _ in range(cfg_reset["warmup_steps"]):
            self.data.qpos[:7]                   = arm_hold
            self.data.qvel[:7]                   = 0.0
            self.data.qpos[qpos_addr:qpos_addr+7] = prefab_hold
            self.data.qvel[qpos_addr:qpos_addr+6] = 0.0
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # 状态重置
        self.current_step      = 0
        self.current_wp_idx    = 0
        self.reached_final     = False
        self.last_dist         = None
        self.last_wp_idx       = -1
        self._wp_just_advanced = False
        self._prev_q           = self.data.qpos[:7].copy().astype(np.float32)

        # [FIX-E3][FIX-E5] 初始化速度缓存
        ee_pos_init  = self._get_ee_pos()
        self._prev_ee_pos        = ee_pos_init.copy()
        self._ee_vel_cache       = np.zeros(3)
        ee_mat_init  = self._get_ee_mat()
        ee_euler_init = R.from_matrix(ee_mat_init).as_euler('xyz')
        self._prev_ee_euler      = ee_euler_init.copy()
        self._ee_euler_vel_cache = np.zeros(3)

        # [FIX-E6] 用 init_q 初始化延迟队列（不是 zeros）
        self.action_queue.clear()
        for _ in range(max(1, self.latency_steps + 1)):
            self.action_queue.append(init_q.copy().astype(np.float32))

        # 渲染器重建
        if self.render_mode:
            if self.viewer is not None:
                try: self.viewer.close()
                except Exception: pass
            self._launch_viewer()

        return self._get_obs()

    # ==========================================================================
    # step()
    # ==========================================================================

    def step(self, action: np.ndarray):
        # 安全 clamp
        action = np.clip(action, self.action_space_low, self.action_space_high)

        # 动作延迟
        self.action_queue.append(action.copy())
        effective_q = np.array(self.action_queue[0], dtype=np.float64)

        # 下发关节角目标
        self.data.ctrl[:7] = effective_q

        # 物理步进
        for _ in range(self.sim_steps):
            mujoco.mj_step(self.model, self.data)

        # NaN 保护
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            obs = self._get_obs()
            return obs, -10.0, True, False, {"is_success": False, "nan_detected": True}

        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

        # [FIX-E3] 物理步进完成后，一次性更新 EE 速度缓存
        ee_pos_new = self._get_ee_pos()
        self._ee_vel_cache = (ee_pos_new - self._prev_ee_pos) / self.dt
        self._prev_ee_pos  = ee_pos_new.copy()

        # [FIX-E5] 更新 EE 欧拉角速度缓存
        ee_mat_new  = self._get_ee_mat()
        ee_euler_new = R.from_matrix(ee_mat_new).as_euler('xyz')
        self._ee_euler_vel_cache = (ee_euler_new - self._prev_ee_euler) / self.dt
        self._prev_ee_euler = ee_euler_new.copy()

        # 获取观测（此时速度缓存已是最新值）
        obs = self._get_obs()

        # 负载 3D 位置
        payload_z  = self.data.body('prefab').xpos[2]
        payload_xy = np.array([obs[4], obs[5]])
        cur_pl_pos = np.array([payload_xy[0], payload_xy[1], payload_z])

        # 航点追踪状态机
        if self._planned_path is not None and not self.reached_final:
            target_wp  = self._planned_path[self.current_wp_idx]
            dist_to_wp = np.linalg.norm(cur_pl_pos - target_wp)

            if self.last_wp_idx != self.current_wp_idx:
                self.last_dist   = None
                self.last_wp_idx = self.current_wp_idx
            self.last_dist = dist_to_wp

            total_wps  = len(self._planned_path)
            rem_wps    = total_wps - 1 - self.current_wp_idx
            look_ahead = 0.04 if rem_wps <= 2 \
                         else self.config["step_logic"]["look_ahead_dist"]

            if dist_to_wp < look_ahead:
                if self.current_wp_idx < total_wps - 1:
                    self.current_wp_idx   += 1
                    self._wp_just_advanced = True
                else:
                    self.reached_final = True

        # [FIX-E4] 传入 obs，避免 _compute_reward 内部重复调用 _get_obs
        current_q = self.data.qpos[:7].copy().astype(np.float32)
        reward, done, success, is_collision = self._compute_reward(
            action, current_q, self._prev_q, obs
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

    # ──────────────────────────────────────────────────────────────────────────────
    # PATCH 1：mujoco_env_new.py 的 _compute_reward 方法
    # 替换 CableRobotEnvWithObstacles._compute_reward
    #
    # 主要改动：
    #   [RWD-3] 新增稠密距离进展奖励（势能函数）
    #   [RWD-4] step_penalty 已在 config 中设为 0，此处逻辑不变
    #   [RWD-1/2] success_bonus/惩罚量级由 config 控制，此处不变
    # ──────────────────────────────────────────────────────────────────────────────
    
    def _compute_reward(self, action, current_q, prev_q, obs):
        """
        奖励计算（修复版）。
        新增稠密距离进展奖励，填补航点间的信号空洞。
        """
        reward = 0.0; done = False; success = False; is_collision = False
        cfg_rwd = self.config["reward"]
    
        payload_xy  = obs[4:6]
        payload_vxy = obs[6:8]
        payload_z   = self.data.body('prefab').xpos[2]
        dof_idx     = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vz  = self.data.qvel[dof_idx + 2]
        pl_vel_norm = float(np.linalg.norm(np.append(payload_vxy, payload_vz)))
        ee_xy       = self._get_ee_pos()[:2]
    
        # ── 每步惩罚 ──────────────────────────────────────────────────────────
        # [RWD-4] step_penalty 在新 config 中为 0，保留接口
        reward += float(cfg_rwd.get("step_penalty", 0.0))
    
        # 速度惩罚
        vel_pen = cfg_rwd.get("velocity_penalty_coef", 0.003) * pl_vel_norm
        reward -= float(np.clip(vel_pen, 0, 0.2))
    
        # 摆角惩罚
        swing = float(np.linalg.norm(ee_xy - payload_xy))
        reward -= cfg_rwd.get("swing_penalty_coef", 0.01) * float(np.clip(swing, 0, 0.1))
    
        # 关节平滑惩罚
        smooth_coef  = -abs(float(cfg_rwd.get("joint_smooth_penalty", -0.001)))
        joint_delta  = float(np.linalg.norm(current_q - prev_q))
        reward      += smooth_coef * joint_delta
    
        '''# 关节极限惩罚
        q_range  = self._q_high - self._q_low
        q_margin = self._q_margin_ratio * q_range
        n_near   = sum(
            1 for j in range(7)
            if (current_q[j] > self._q_high[j] - q_margin[j] or
                current_q[j] < self._q_low[j]  + q_margin[j])
        )
        if n_near > 0:
            reward -= abs(float(cfg_rwd.get("joint_limit_penalty", -0.05))) * n_near'''
    
        # ── [RWD-3] 稠密距离进展奖励（势能函数）─────────────────────────────
        # 计算负载到当前目标航点（或最终目标）的距离进展
        # 只在非终止状态时计算（终止状态由下方逻辑单独处理）
        if self._planned_path is not None and not self.reached_final:
            # 当前目标：当前航点（或最终目标）
            target_wp = self._planned_path[min(self.current_wp_idx,
                                                len(self._planned_path) - 1)]
            curr_dist = float(np.linalg.norm(
                np.array([payload_xy[0], payload_xy[1], payload_z]) - target_wp
            ))
    
            # 上一步距离（初始化为当前距离，避免第一步虚假奖励）
            if self.last_dist is None:
                self.last_dist = curr_dist
    
            # 进展 = 上一步距离 - 当前距离（正值=靠近，负值=远离）
            # 只奖励靠近（clip 下界为 0），不惩罚停滞/远离（那是 policy 学习的代价）
            progress = float(np.clip(
                self.last_dist - curr_dist,
                0.0,
                cfg_rwd.get("progress_clip", 0.1)
            ))
            reward  += cfg_rwd.get("progress_coef", 2.0) * progress
            # 注意：last_dist 在 step() 的航点状态机中已经更新，
            # 这里不再重复赋值（step() 末尾会更新 self.last_dist = dist_to_wp）
    
        # ── 终止条件 ──────────────────────────────────────────────────────────
        if self.reached_final:
            vel_xy        = float(np.linalg.norm(payload_vxy))
            dist_to_final = float(np.linalg.norm(payload_xy - self.target_pos))
            if dist_to_final < 0.03 and vel_xy < 0.1 and abs(payload_vz) < 0.2:
                reward += cfg_rwd.get("success_bonus", 10.0)
                success = True
            else:
                reward += cfg_rwd.get("crash_penalty", -5.0)
            done = True
            return reward, done, success, is_collision
    
        for (ox, oy, orad) in self._obstacles:
            if float(np.linalg.norm(payload_xy - np.array([ox,oy]))) < (orad + self.payload_radius):
                reward += cfg_rwd.get("collision_penalty", -5.0)
                done = True; is_collision = True
                return reward, done, success, is_collision
    
        if float(np.linalg.norm(payload_xy)) < 0.03:
            reward += cfg_rwd.get("collision_penalty", -5.0)
            done = True; is_collision = True
            return reward, done, success, is_collision
    
        cfg_logic = self.config["step_logic"]
        if (payload_z < cfg_logic["crash_z_threshold"] and
                payload_vz < cfg_logic["crash_vz_threshold"]):
            reward += cfg_rwd.get("crash_penalty", -5.0)
            done = True
            return reward, done, success, is_collision
    
        # 航点里程碑奖励
        if getattr(self, '_wp_just_advanced', False):
            reward += cfg_rwd.get("waypoint_bonus", 0.15)
            self._wp_just_advanced = False
    
        return reward, done, success, is_collision

    # ==========================================================================
    # _get_obs()
    # ==========================================================================

    def _get_obs(self) -> np.ndarray:
        """
        [FIX-E3] EE 速度使用缓存值（由 step() 在物理步进后更新一次），
        避免多次调用导致的速度重算问题。
        [FIX-E5] EE 欧拉角速度使用缓存估计。
        """
        # EE 位置与姿态
        ee_pos = self._get_ee_pos()
        ee_x, ee_y, ee_z = ee_pos

        # [FIX-E3] 速度来自缓存
        ee_vx, ee_vy, ee_vz = self._ee_vel_cache

        # EE 姿态（欧拉角）
        site_mat = self._get_ee_mat()
        ee_euler = R.from_matrix(site_mat).as_euler('xyz')
        ee_roll, ee_pitch, ee_yaw = ee_euler

        # [FIX-E5] 欧拉角速度来自缓存
        ee_roll_v, ee_pitch_v, ee_yaw_v = self._ee_euler_vel_cache

        # 负载状态
        payload_x  = self.data.body('prefab').xpos[0]
        payload_y  = self.data.body('prefab').xpos[1]
        payload_z  = self.data.body('prefab').xpos[2]
        dof_idx    = self.model.jnt_dofadr[self.prefab_jnt_id]
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

        # 关节角 & 关节角速度
        joint_q  = self.data.qpos[:7].copy().astype(np.float32)
        joint_dq = self.data.qvel[:7].copy().astype(np.float32)

        # 拼接（与头文件布局严格对齐）
        return np.array(
            [ee_x, ee_y, ee_vx, ee_vy,
             payload_x, payload_y, payload_vx, payload_vy,
             rel_tx, rel_ty]
            + obs_data[:target_len]
            + [ee_z,   ee_vz,   payload_z,   payload_vz,
               ee_roll, ee_roll_v, ee_pitch, ee_pitch_v,
               ee_yaw, ee_yaw_v, 0.0, 0.0]         # [-16/-15] payload_yaw 占位
            + list(joint_q)
            + list(joint_dq),
            dtype=np.float32
        )


# ==============================================================================
# 便捷工厂函数
# ==============================================================================

def make_env(config: dict = None) -> CableRobotEnvWithObstacles:
    return CableRobotEnvWithObstacles(config=config)