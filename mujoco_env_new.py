# ==============================================================================
# mujoco_env_new.py — 合并版（旧版 v3/v4 奖励 + 新版 socket/rebar 任务）
#
# ══════════════════════════════════════════════════════════════════════════════
# 合并核心变更（相对两个输入版本）
# ══════════════════════════════════════════════════════════════════════════════
#
# [MERGE-E1] 保留新版 XML 动态生成（socket prefab + rebar target）
#   - generate_scene_and_trajectory 末尾用 re.sub 动态改写 target body 位置
#   - 导入 re 模块
#
# [MERGE-E2] 奖励函数（_compute_reward）全面恢复旧版 v3/v4 结构 + 插入任务扩展
#   保留旧版：
#     - 失稳早停（instability_check + grace_steps）
#     - 负指数连续惩罚（_neg_exp）
#     - 吊装物姿态复合惩罚（tilt/yaw 线性 + 指数）
#     - APF 障碍物排斥势（含 rho_0 / d_min / apf_max）
#     - 稠密进展奖励（progress_coef）
#     - 终点判定 / 碰撞 / crash grace / 单步 reward clip
#     - 终止原因记录（_termination_reason）
#   新增插入任务专属：
#     - [INS-1] 钢筋对准引导奖励（激励 XY 和姿态同时对齐）
#     - [INS-2] 插入成功精确判定（target_payload_z + tilt/yaw/xy tolerance）
#     - [INS-3] 基于 MuJoCo data.contact 的真实碰撞检测（prefab vs obstacle/rebar）
#     - [INS-4] payload_radius 改为 socket 外接圆半径
#
# [MERGE-E3] 保留旧版课程学习 runtime 接口
#   - set_curriculum_n_obstacles() 方法
#
# [MERGE-E4] 保留旧版所有 FIX-E* 修复 + ENV-DELTA-* delta-q 动作空间
# ══════════════════════════════════════════════════════════════════════════════

import os
import re
import copy
import heapq
import tempfile
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R

from config import DEFAULT_CONFIG
from controller import NativeIKSolver


class CableRobotEnvWithObstacles:
    """
    delta-q 动作空间版索驱动机器人环境（钢筋插入任务版）。
    action = Δq ∈ [-dq_max, +dq_max]^7
    env 内部：q_cmd = clip(q_current + Δq, q_low, q_high)

    新任务：将带 4 方孔的 socket 吊装物准确插入地面 4 根钢筋桩。
    成功条件：tilt/yaw/XY/z 同时满足容差（insertion 节配置）。
    """

    def __init__(self, config=None):
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
        self.cfg_insertion = self.config.get("insertion", {})

        self.physics_dt      = cfg_sim["physics_dt"]
        self.control_freq_hz = cfg_sim["control_freq_hz"]
        self.control_dt      = 1.0 / self.control_freq_hz
        self.dt              = self.control_dt
        self.sim_steps       = int(self.dt / self.physics_dt)
        self.max_steps       = cfg_sim["max_steps"]
        self.current_step    = 0

        # [ENV-DELTA-2] 关节限位（用于 q_cmd clamp）
        self.q_low  = np.array(cfg_space["action_space_low"],  dtype=np.float32)
        self.q_high = np.array(cfg_space["action_space_high"], dtype=np.float32)
        self.dq_max = np.array(cfg_space.get("dq_max", [0.1]*7), dtype=np.float32)
        self.action_dim = cfg_space["action_dim"]
        self.action_space_high = self.dq_max.copy()
        self.action_space_low  = -self.dq_max.copy()

        self.default_start_xy   = np.array(cfg_task["default_start_xy"])
        self.default_target     = np.array(cfg_task["default_target_xy"])
        self.target_pos         = self.default_target.copy()
        self.init_position_range = cfg_task["init_position_range"]

        self.n_obstacles          = cfg_scene["n_obstacles"]
        self.obstacle_radius_range = cfg_scene["radius_range"]
        self._obstacle_rng        = np.random.default_rng(cfg_scene["seed"])
        # [WS] path_width 从 planning 节读（新版），兼容旧 scene 节
        self.path_width           = cfg_plan.get("path_width", cfg_scene.get("path_width", 0.4))
        self.payload_radius       = cfg_plan["payload_radius"]
        self.planning_margin      = cfg_plan["planning_margin"]
        self.planning_grid_res    = cfg_plan["planning_grid_res"]
        # [WS] 机械臂工作空间硬约束半径
        self.workspace_radius     = cfg_plan.get("workspace_radius", 0.50)

        self.latency_steps = cfg_noise["latency_steps"]
        init_q_default     = np.array(self.config["reset"]["init_qpos_arm"], np.float32)
        self.action_queue  = deque(maxlen=max(1, self.latency_steps + 1))
        for _ in range(max(1, self.latency_steps + 1)):
            self.action_queue.append(init_q_default.copy())

        # [V3-OBS] 状态维度（10 + 3n + 26 + 9）
        # 新增9维: phase_encode(3) + progress(1) + z_error(1) + rebar_errors(4)
        # 注: payload_tilt/yaw 替换了原来的两个0.占位，不增加维度
        self.state_dim = 10 + (self.n_obstacles * 3) + 26 + 9

        self._prev_q        = init_q_default.copy()
        self._q_margin_ratio = float(self.cfg_reward.get("joint_limit_margin", 0.1))

        # ── XML 初始化 ─────────────────────────────────────────────────────────
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

        self.ik_solver = NativeIKSolver(self.model, self.data)
        print("✅ IK Solver 初始化成功！")
        self._reresolve_ids()

        self._obstacles        = []
        self._planned_path     = None
        self.current_wp_idx    = 0
        self.reached_final     = False
        self.last_dist         = None
        self.last_wp_idx       = -1
        self._wp_just_advanced = False

        self._prev_ee_pos        = np.zeros(3)
        self._ee_vel_cache       = np.zeros(3)
        self._prev_ee_euler      = np.zeros(3)
        self._ee_euler_vel_cache = np.zeros(3)
        self._termination_reason = None     # [v3-DIAG]

        # ── [WIND] 风力扰动初始化 ─────────────────────────────────────────────
        self.cfg_wind = self.config.get("wind", {})
        wind_seed = self.cfg_wind.get("seed", 123)
        self.wind_rng   = np.random.default_rng(wind_seed)
        self.wind_theta = 0.0          # 风向角 (rad)
        self.wind_F     = 0.0          # 风力大小 (N)
        self._wind_curriculum_frac = 1.0  # 训练时的风力倍率（0→1）

        self.render_mode = cfg_sim["render"]
        self.viewer      = None
        if self.render_mode:
            self._launch_viewer()

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def _reresolve_ids(self):
        self.prefab_jnt_id  = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("target").id
        self.ee_site_id     = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # [INS-3] 解析 obstacle 和 rebar body id，用于真实碰撞检测
        self._obstacle_body_ids = []
        for i in range(self.n_obstacles):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
            if bid >= 0:
                self._obstacle_body_ids.append(bid)

        # prefab 子树的所有 geom id（用于 contact 归属判断）
        # 注意：MuJoCo 的 contact 记录是 geom1/geom2，需要查 geom → body 映射
        self._prefab_geom_ids = set()
        for gid in range(self.model.ngeom):
            bid = self.model.geom_bodyid[gid]
            # 追溯到 prefab body 或其子 body
            cur = bid
            while cur > 0:
                if cur == self.prefab_body_id:
                    self._prefab_geom_ids.add(gid)
                    break
                cur = self.model.body_parentid[cur]

        # [INS-3] 重建 obstacle/rebar geom_id 缓存（reset 后需要更新）
        # 使用 hasattr 避免 __init__ 首次调用时出错
        self._obstacle_geom_ids = set()
        self._rebar_geom_ids    = set()
        self._build_geom_id_sets()

    def _get_ee_pos(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def _get_ee_mat(self):
        return self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()

    def _launch_viewer(self):
        kw = {}
        if hasattr(self, '_key_callback') and self._key_callback is not None:
            kw['key_callback'] = self._key_callback
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, **kw)

    def get_planned_path(self):
        return self._planned_path

    # ── [WIND] 风力扰动方法 ──────────────────────────────────────────────────

    def _update_wind(self):
        """缓慢随机游走更新风向和风力大小（每个物理子步调用）。"""
        if not self.cfg_wind.get("enabled", False):
            return
        dt = self.physics_dt
        theta_std = self.cfg_wind.get("theta_rate_std", 0.15)
        force_std = self.cfg_wind.get("force_rate_std", 0.1)
        F_max     = self.cfg_wind.get("F_max", 1.0)

        self.wind_theta += dt * self.wind_rng.normal(0, theta_std)
        self.wind_F     += dt * self.wind_rng.normal(0, force_std)
        self.wind_F      = float(np.clip(self.wind_F, 0, F_max))

    def _apply_wind_force(self):
        """将风力作为外力施加到 payload body 上（每个物理子步调用）。"""
        if not self.cfg_wind.get("enabled", False):
            return
        effective_F = self.wind_F * self._wind_curriculum_frac
        fx = effective_F * np.cos(self.wind_theta)
        fy = effective_F * np.sin(self.wind_theta)
        self.data.xfrc_applied[self.prefab_body_id, :3] = [fx, fy, 0.0]

    def set_wind_curriculum(self, frac: float):
        """设置风力课程学习倍率，0.0=无风，1.0=全风力。"""
        self._wind_curriculum_frac = float(np.clip(frac, 0.0, 1.0))

    def get_wind_state(self):
        """返回当前风力状态 (wind_F, wind_theta)，供观测构建使用。"""
        effective_F = self.wind_F * self._wind_curriculum_frac
        return float(effective_F), float(self.wind_theta)

    # ── [v3-CURRICULUM] 运行时动态设置障碍物数 ────────────────────────────────
    def set_curriculum_n_obstacles(self, n: int):
        """
        课程学习 / 测试：运行时修改实际生成的障碍物数量。

        约束：
          - n 必须 ∈ [0, self.n_obstacles]（初始化时的上限）
          - state_dim 已在 __init__ 时固定为 10 + 3*self.n_obstacles + 26，不可改
          - 若 n < self.n_obstacles，_get_obs 会用 0 填充缺失的障碍物维度

        返回：实际生效的 n（clip 到合法范围后的值）。

        警告：如果 n > self.n_obstacles，函数会 clip 并打印警告，
             因为超出上限的障碍物维度无法塞入已固定的 observation。
        """
        n_orig = int(n)
        n = int(max(0, min(n_orig, self.n_obstacles)))
        if n != n_orig:
            print(f"[WARN] set_curriculum_n_obstacles: 请求 n={n_orig} 超出上限 {self.n_obstacles}，"
                  f"已 clip 至 {n}。若要更多障碍物，需增大 config['scene']['n_obstacles'] "
                  f"并重新创建环境。")
        self.config["scene"]["n_obstacles"] = n
        return n

    def close(self):
        if self.viewer is not None:
            try: self.viewer.close()
            except Exception: pass

    # ── 场景生成 ────────────────────────────────────────────────────────────

    @staticmethod
    def generate_scene_and_trajectory(start_xy, target_xy, base_xml_content,
                                       scene_plan_config, rng=None):
        """A* + 3D 轨迹 + XML（含新版动态 target 位置改写）。"""
        if rng is None:
            rng = np.random.default_rng()
        start_xy  = np.asarray(start_xy,  float).reshape(2)
        target_xy = np.asarray(target_xy, float).reshape(2)

        p_radius = scene_plan_config["payload_radius"]
        p_margin = scene_plan_config["planning_margin"]
        min_clr  = p_radius + p_margin
        base_r1  = 0.13
        base_r2  = 0.1
        n_obs    = scene_plan_config["n_obstacles"]
        r_min, r_max = scene_plan_config["radius_range"]
        # [WS] path_width 从 planning 节读，兼容旧 scene 节
        path_width   = scene_plan_config.get("path_width", 0.4)
        # [WS] 机械臂工作空间硬约束半径（包含 payload 外接圆）
        workspace_r  = scene_plan_config.get("workspace_radius", 0.50)
        # [WS] 路径单侧约束：强制 A* 和障碍物都在 y >= y_min_corridor 侧
        # 当前起点 y=0.15, 终点 y=0.2，路径天然在 y>0 区间，强制障碍物也在 y>0
        y_min_corridor = scene_plan_config.get("y_min_corridor", 0.0)

        obstacles = []
        direction = target_xy - start_xy
        L_path    = np.linalg.norm(direction)
        if L_path > 1e-6:
            direction /= L_path
            perp = np.array([-direction[1], direction[0]])
            attempts = 0
            max_attempts = max(500, n_obs * 200)
            while len(obstacles) < n_obs and attempts < max_attempts:
                attempts += 1
                t = rng.uniform(0.15, 0.85)
                s = rng.uniform(-path_width/2, path_width/2)
                center = start_xy + t*L_path*direction + s*perp
                r = rng.uniform(r_min, r_max)
                # [WS] 障碍物中心 + 半径必须完全在工作空间内
                if np.linalg.norm(center) + r > workspace_r - 0.02:
                    continue
                # [WS] 障碍物必须完全在 y >= y_min_corridor 侧（同路径走廊约束）
                if center[1] - r < y_min_corridor:
                    continue
                if (np.linalg.norm(center-start_xy)<r+min_clr or
                        np.linalg.norm(center-target_xy)<r+min_clr or
                        np.linalg.norm(center)<r+base_r2): continue
                if all(np.linalg.norm(center-np.array([ox,oy]))>=r+or_+0.01
                       for (ox,oy,or_) in obstacles):
                    obstacles.append((float(center[0]),float(center[1]),float(r)))

        planning_obs = obstacles + [(0.0,0.0,base_r1)]
        grid_res = scene_plan_config["planning_grid_res"]
        xs = [start_xy[0], target_xy[0]]; ys = [start_xy[1], target_xy[1]]
        for (ox,oy,r) in planning_obs:
            re_=r+min_clr; xs.extend([ox-re_,ox+re_]); ys.extend([oy-re_,oy+re_])
        x_min=min(xs)-scene_plan_config["bounds_margin"]; x_max=max(xs)+scene_plan_config["bounds_margin"]
        y_min=min(ys)-scene_plan_config["bounds_margin"]; y_max=max(ys)+scene_plan_config["bounds_margin"]
        nx=max(2,int(np.ceil((x_max-x_min)/grid_res))); ny=max(2,int(np.ceil((y_max-y_min)/grid_res)))

        def w2g(x,y): return (max(0,min(nx-1,int((x-x_min)/grid_res))),max(0,min(ny-1,int((y-y_min)/grid_res))))
        def g2w(i,j): return x_min+(i+.5)*grid_res, y_min+(j+.5)*grid_res

        # 端点附近不受 y_min_corridor 约束（保证起终点可达）
        endpoint_buf = 0.08  # 8cm
        start_pt = np.asarray(start_xy, float)
        target_pt = np.asarray(target_xy, float)

        occ = np.zeros((nx,ny),bool)
        ws_r_eff = workspace_r - p_radius
        for i in range(nx):
            for j in range(ny):
                wx,wy=g2w(i,j)
                # 工作空间硬约束
                if wx*wx + wy*wy > ws_r_eff*ws_r_eff:
                    occ[i,j]=True
                    continue
                # [WS] Y 单侧约束（端点附近保留可达性）
                if wy < y_min_corridor:
                    d_start = np.hypot(wx - start_pt[0], wy - start_pt[1])
                    d_target = np.hypot(wx - target_pt[0], wy - target_pt[1])
                    if d_start > endpoint_buf and d_target > endpoint_buf:
                        occ[i,j] = True
                        continue
                # 障碍物占据判定
                if any((wx-ox)**2+(wy-oy)**2<(r+min_clr)**2 for (ox,oy,r) in planning_obs):
                    occ[i,j]=True

        def nf(i0,j0,rad=5):
            if not occ[i0,j0]: return i0,j0
            best,bd=None,None
            for di in range(-rad,rad+1):
                for dj in range(-rad,rad+1):
                    ni,nj=i0+di,j0+dj
                    if 0<=ni<nx and 0<=nj<ny and not occ[ni,nj]:
                        d=di*di+dj*dj
                        if best is None or d<bd: best,bd=(ni,nj),d
            return best

        si=nf(*w2g(*start_xy)) or w2g(*start_xy)
        gi=nf(*w2g(*target_xy)) or w2g(*target_xy)
        open_h=[]; g_cost={si:0.}; parent={}
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
                    heapq.heappush(open_h,(ng+float(np.hypot(*(np.array(g2w(ni,nj))-target_xy))),nb))

        if not found:
            # [PATH-FAIL] A* 失败 → 不使用 fallback，返回 None
            # 上层 (env.reset) 接到 None 会选择重试或标记该回合无效
            # 这避免了"兜底路径"让坏场景混入训练/测试造成污染
            print(f"[A*] 规划失败（n_obs={n_obs}, obstacles 层布置导致无可通行路径）→ 返回 None")

            # 生成 XML（即便路径失败，scene 本身仍可用于调试）
            xml = base_xml_content
            if obstacles:
                xml = xml.replace('  </asset>',
                    '    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>\n  </asset>', 1)
            obs_z = scene_plan_config["obstacle_z_center"]
            obs_hh = scene_plan_config["obstacle_halfheight"]
            obs_b = "".join([
                f'    <body name="obstacle_{i}" pos="{x} {y} {obs_z}">\n      '
                f'<geom type="cylinder" size="{r} {obs_hh}" material="obstacle" '
                f'contype="1" conaffinity="1"/>\n    </body>\n'
                for i, (x, y, r) in enumerate(obstacles)])
            repl = ('<geom name="floor" size="0 0 0.05" type="plane" '
                    'material="groundplane"/>\n\n' + obs_b + '    ')
            xml = xml.replace(
                '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    ',
                repl, 1)
            xml = re.sub(
                r'(<body\s+name="target"\s+pos=")[^"]*(")',
                rf'\g<1>{target_xy[0]} {target_xy[1]} 0\2',
                xml, count=1)

            return obstacles, None, xml   # 返回 path_3d=None 表示失败
        else:
            idx=[]; node=gi
            while node!=si: idx.append(node); node=parent.get(node); (node is None) and idx.append(si) or None
            idx.append(si); idx.reverse()
            path_2d=np.array([g2w(i,j) for (i,j) in idx])
            path_2d[0]  = start_xy
            path_2d[-1] = target_xy

        z_cruise=scene_plan_config["payload_z_cruise"]
        num_lift=scene_plan_config.get("num_lift_steps",5)
        path_3d=[]
        for z in np.linspace(0.11,z_cruise,num_lift+1)[1:]:
            path_3d.append([float(start_xy[0]),float(start_xy[1]),float(z)])
        for pt in path_2d[1:]:
            path_3d.append([float(pt[0]),float(pt[1]),float(z_cruise)])
        for z in np.linspace(z_cruise,scene_plan_config["target_z_descent"],
                             scene_plan_config["num_descent_steps"]+1)[1:]:
            path_3d.append([float(target_xy[0]),float(target_xy[1]),float(z)])
        path_3d=np.array(path_3d)

        xml=base_xml_content
        if obstacles:
            xml=xml.replace('  </asset>','    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>\n  </asset>',1)
        obs_z=scene_plan_config["obstacle_z_center"]; obs_hh=scene_plan_config["obstacle_halfheight"]
        obs_b="".join([f'    <body name="obstacle_{i}" pos="{x} {y} {obs_z}">\n      <geom type="cylinder" size="{r} {obs_hh}" material="obstacle" contype="1" conaffinity="1"/>\n    </body>\n' for i,(x,y,r) in enumerate(obstacles)])
        pb="".join([f'    <body name="path_pt_{i}" pos="{p[0]} {p[1]} {p[2]}">\n      <geom type="sphere" size="0.01" rgba="0 0 1 1" contype="0" conaffinity="0"/>\n    </body>\n' for i,p in enumerate(path_3d)])
        epz=obs_z+obs_hh+scene_plan_config["endpoint_z_offset"]
        epb=(f'    <body name="path_start" pos="{start_xy[0]} {start_xy[1]} {epz}">\n      <geom type="sphere" size="0.012" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n    </body>\n'
             f'    <body name="path_goal" pos="{target_xy[0]} {target_xy[1]} {epz}">\n      <geom type="sphere" size="0.012" rgba="1 0 0 1" contype="0" conaffinity="0"/>\n    </body>\n')
        repl='<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'+obs_b+pb+epb+'    '
        xml=xml.replace('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    ',repl,1)

        # [MERGE-E1] 动态更新 target body 位置，使 rebar 与路径终点一致
        xml = re.sub(
            r'(<body\s+name="target"\s+pos=")[^"]*(")',
            rf'\g<1>{target_xy[0]} {target_xy[1]} 0\2',
            xml, count=1)

        return obstacles, path_3d, xml

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        cfg_task=self.config["task"]; cfg_scene=self.config["scene"]
        cfg_plan=self.config["planning"]; cfg_reset=self.config["reset"]

        # [PATH-FAIL] 场景生成重试机制：A* 失败时用新 seed 再生成，最多 max_retries 次
        max_retries = 3
        obstacles = None; path_3d = None; new_xml = None
        start_xy = None; target_xy = None

        for retry in range(max_retries + 1):
            noise    = self._obstacle_rng.uniform(-self.init_position_range, self.init_position_range, 2)
            start_xy = self.default_start_xy + noise
            target_xy= np.array(cfg_task["default_target_xy"]); self.target_pos=target_xy.copy()

            spCfg={**cfg_scene,**cfg_plan}
            obstacles,path_3d,new_xml=self.generate_scene_and_trajectory(
                start_xy,target_xy,self._base_xml_content,spCfg,self._obstacle_rng)

            if path_3d is not None:
                break  # 规划成功
            if retry < max_retries:
                print(f"[env.reset] 第 {retry+1} 次场景生成失败，重试...")

        self._obstacles=obstacles; self._planned_path=path_3d

        with tempfile.NamedTemporaryFile(mode='w',suffix='.xml',dir=self._assets_dir,delete=False,encoding='utf-8') as f:
            f.write(new_xml); tmp_path=f.name
        try:
            self.model=mujoco.MjModel.from_xml_path(tmp_path)
            self.data=mujoco.MjData(self.model)
            self.model.opt.timestep=self.physics_dt
        finally:
            try: os.remove(tmp_path)
            except OSError: pass

        self.ik_solver.update_model(self.model,self.data)
        self._reresolve_ids()

        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[:]=0.; self.data.qvel[:]=0.
        for j in range(self.model.njnt):
            adr=self.model.jnt_qposadr[j]; jtype=self.model.jnt_type[j]
            if jtype==mujoco.mjtJoint.mjJNT_BALL: self.data.qpos[adr:adr+4]=[1,0,0,0]
            elif jtype==mujoco.mjtJoint.mjJNT_FREE: self.data.qpos[adr+3:adr+7]=[1,0,0,0]

        seed_q=np.array(cfg_reset["init_qpos_arm"],np.float64)
        start_z=float(cfg_reset.get("mocap_init_z",0.4))
        init_q=self.ik_solver.solve_4d(seed_q,start_xy[0],start_xy[1],start_z,0.0)
        if init_q is None or np.any(np.isnan(init_q)): init_q=seed_q.copy()
        self.data.qpos[:7]=init_q

        pref_qpos=np.array(cfg_reset["init_qpos_prefab"],np.float64)
        pref_qpos[0]=start_xy[0]; pref_qpos[1]=start_xy[1]
        pref_jnt=self.model.body("prefab").jntadr[0]
        qpos_addr=self.model.jnt_qposadr[pref_jnt]
        self.data.qpos[qpos_addr:qpos_addr+7]=pref_qpos

        arm_hold=init_q.copy(); prefab_hold=pref_qpos.copy()
        self.data.ctrl[:7]=arm_hold
        for _ in range(cfg_reset["warmup_steps"]):
            self.data.qpos[:7]=arm_hold; self.data.qvel[:7]=0.
            self.data.qpos[qpos_addr:qpos_addr+7]=prefab_hold
            self.data.qvel[qpos_addr:qpos_addr+6]=0.
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)

        self.current_step=0; self.current_wp_idx=0; self.reached_final=False
        self.last_dist=None; self.last_wp_idx=-1; self._wp_just_advanced=False
        self._prev_q=self.data.qpos[:7].copy().astype(np.float32)
        self._termination_reason = None

        # [INS-NEW] 插入阶段状态机：
        # hold_counter = 连续满足插入条件的步数，达到 hold_steps 时判定成功
        # in_insertion = payload_z 已进入 entry_z 以下（进入插入阶段）
        self._insertion_hold_counter = 0
        self._in_insertion_phase     = False
        self._best_insertion_z       = 10.0   # 记录 payload 到达过的最低 z

        # [POT-NEW] 势能差分奖励状态（第一步置 None，用"当前即初始"值）
        self._prev_goal_potential = None
        # [V3] 新增势能状态
        self._prev_phi_z = None
        self._prev_descent_depth = 0.0

        # [STABLE] 抖动抑制：记录上一步 Δq，用于 action rate 惩罚
        self._prev_delta_q = np.zeros(self.action_dim, dtype=np.float32)

        ee_init=self._get_ee_pos()
        self._prev_ee_pos=ee_init.copy(); self._ee_vel_cache=np.zeros(3)
        mat_init=self._get_ee_mat()
        self._prev_ee_euler=R.from_matrix(mat_init).as_euler('xyz').copy()
        self._ee_euler_vel_cache=np.zeros(3)

        # 新奖励需要的状态变量
        self._prev_ref_dist = None
        self._prev_descent_depth = 0.0
        self._best_insertion_z = 10.0   # 记录 payload 曾达到的最低 z (值越小越好)

        self.action_queue.clear()
        for _ in range(max(1, self.latency_steps+1)):
            self.action_queue.append(init_q.copy().astype(np.float32))

        # ── [WIND] 风力状态重置 ──────────────────────────────────────────────
        self.wind_theta = float(self.wind_rng.uniform(0, 2 * np.pi))
        F_max = self.cfg_wind.get("F_max", 1.0)
        self.wind_F = 0.2 * F_max

        if self.render_mode:
            if self.viewer is not None:
                try: self.viewer.close()
                except Exception: pass
            self._launch_viewer()
            '''# ==========================================================
            # 新增逻辑：在开启渲染界面时，先同步画面，然后阻塞等待空格键
            # ==========================================================
            if hasattr(self, 'viewer') and self.viewer is not None:
                self.viewer.sync()  # 确保加载出第一帧初始画面，避免黑屏
            
            print("\n" + "="*50)
            # 使用原生 input 阻塞程序，等待终端的回车键
            input("⏸️  环境初始状态已加载，请在当前终端按下 [Enter 回车键] 开始运动...")
            print("="*50)
            print("▶️  开始执行！")
            # =========================================================='''

        return self._get_obs()

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, delta_q: np.ndarray):
        """[ENV-DELTA-1] 接受 delta_q，累加到当前关节角后执行。
        [STABLE] 插入阶段 (_in_insertion_phase=True) 自动将 Δq 幅度乘 dq_scale_insertion。
        """
        # 保存原始 actor 输出（用于 reward 的 action_rate 惩罚与 BC 对齐）
        delta_q_raw = np.clip(
            np.asarray(delta_q, dtype=np.float32),
            self.action_space_low, self.action_space_high
        )

        # 插入阶段自动缩减 Δq 幅度（减小抖动）
        if getattr(self, '_in_insertion_phase', False):
            scale = float(self.config["space"].get("dq_scale_insertion", 0.3))
            delta_q_exec = delta_q_raw * scale
        else:
            delta_q_exec = delta_q_raw

        q_current = self.data.qpos[:7].copy().astype(np.float32)
        q_cmd     = np.clip(q_current + delta_q_exec, self.q_low, self.q_high)

        self.action_queue.append(q_cmd.copy())
        effective_q = np.array(self.action_queue[0], np.float64)

        self.data.ctrl[:7] = effective_q

        for _ in range(self.sim_steps):
            self._update_wind()
            self._apply_wind_force()
            mujoco.mj_step(self.model, self.data)

        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            obs = self._get_obs()
            return obs, -10.0, True, False, {"is_success": False, "nan_detected": True}

        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

        # 更新速度缓存
        ee_new = self._get_ee_pos()
        self._ee_vel_cache = (ee_new - self._prev_ee_pos) / self.dt
        self._prev_ee_pos  = ee_new.copy()
        mat_new = self._get_ee_mat()
        euler_new = R.from_matrix(mat_new).as_euler('xyz')
        self._ee_euler_vel_cache = (euler_new - self._prev_ee_euler) / self.dt
        self._prev_ee_euler = euler_new.copy()

        obs = self._get_obs()

        payload_z  = self.data.body('prefab').xpos[2]
        payload_xy = np.array([obs[4], obs[5]])
        cur_pl_pos = np.array([payload_xy[0], payload_xy[1], payload_z])

        if self._planned_path is not None and not self.reached_final:
            target_wp  = self._planned_path[self.current_wp_idx]
            dist_to_wp = np.linalg.norm(cur_pl_pos - target_wp)
            if self.last_wp_idx != self.current_wp_idx:
                self.last_dist=None; self.last_wp_idx=self.current_wp_idx
            self.last_dist = dist_to_wp
            total_wps=len(self._planned_path); rem=total_wps-1-self.current_wp_idx
            look_ahead=0.04 if rem<=2 else self.config["step_logic"]["look_ahead_dist"]
            if dist_to_wp < look_ahead:
                if self.current_wp_idx < total_wps-1:
                    self.current_wp_idx+=1; self._wp_just_advanced=True
                else:
                    self.reached_final=True

        current_q = self.data.qpos[:7].copy().astype(np.float32)
        reward, done, success, is_collision = self._compute_reward(
            delta_q_raw, current_q, self._prev_q, obs)
        self._prev_q = current_q
        # [STABLE] 保存本步 Δq 用于下一步 action_rate 惩罚
        self._prev_delta_q = delta_q_raw.copy()

        self.current_step += 1
        if self.current_step >= self.config["sim"]["max_steps"]:
            if not done:
                reward += self.config["reward"]["timeout_penalty"]
                if getattr(self, '_termination_reason', None) is None:
                    self._termination_reason = "timeout"
            done = True

        return obs, reward, done, False, {
            "is_success": success, "current_wp_idx": self.current_wp_idx,
            "reached_final": self.reached_final, "is_collision": is_collision,
            "termination_reason": getattr(self, '_termination_reason', None),
        }

    # ── [INS-3] MuJoCo 真实接触检测辅助 ───────────────────────────────────────

    def _build_geom_id_sets(self):
        """
        缓存 obstacle_geom_ids、rebar_geom_ids、floor_geom_ids。
        在 _reresolve_ids 之后调用一次；reset 后需要重新调用。
        """
        self._obstacle_geom_ids = set()
        for bid in self._obstacle_body_ids:
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == bid:
                    self._obstacle_geom_ids.add(gid)

        self._rebar_geom_ids = set()
        if hasattr(self, 'target_body_id') and self.target_body_id >= 0:
            for gid in range(self.model.ngeom):
                bid = self.model.geom_bodyid[gid]
                cur = bid
                while cur > 0:
                    if cur == self.target_body_id:
                        self._rebar_geom_ids.add(gid)
                        break
                    cur = self.model.body_parentid[cur]

        # [INS-NEW] floor geom id（用于地面接触检测，作为成功判定的一部分）
        self._floor_geom_ids = set()
        fid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if fid >= 0:
            self._floor_geom_ids.add(fid)

    def _check_prefab_collision_with_obstacles(self):
        """
        检查 prefab 子树任意 geom 与其他物体的 MuJoCo 真实接触。

        用户需求：
          - prefab × obstacle → 视为"碰撞"，立即 reset
          - prefab × rebar    → 允许接触（插入过程自然摩擦），不终止
          - prefab × floor    → 允许接触（成功判定一部分）
          - prefab × robot    → 允许接触（cable 连接）

        返回：(hit_obstacle, hit_rebar)
        """
        if not hasattr(self, '_obstacle_geom_ids'):
            self._build_geom_id_sets()

        hit_obstacle = False
        hit_rebar    = False

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2

            p_hit = (g1 in self._prefab_geom_ids) or (g2 in self._prefab_geom_ids)
            if not p_hit:
                continue

            other = g2 if g1 in self._prefab_geom_ids else g1
            if other in self._obstacle_geom_ids:
                hit_obstacle = True
                break
            elif other in self._rebar_geom_ids:
                hit_rebar = True

        return hit_obstacle, hit_rebar

    def _check_prefab_floor_contact(self):
        """
        [INS-NEW] 检查 payload 底部是否真实触碰到地面。
        成功判定的一部分：用户要求"平稳接触地面则任务成功"。
        """
        if not hasattr(self, '_floor_geom_ids'):
            self._build_geom_id_sets()
        if not self._floor_geom_ids:
            return False  # 没有 floor geom 就无法检测

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            g1, g2 = con.geom1, con.geom2
            p_hit = (g1 in self._prefab_geom_ids) or (g2 in self._prefab_geom_ids)
            if not p_hit:
                continue
            other = g2 if g1 in self._prefab_geom_ids else g1
            if other in self._floor_geom_ids:
                return True
        return False

    # ── _compute_reward V3 ──────────────────────────────────────────────────

    @staticmethod
    def _neg_exp(value, coef, scale):
        """负指数惩罚：coef * (1 - exp(-value/scale))。"""
        return coef * (1.0 - np.exp(-value / max(scale, 1e-8)))

    @staticmethod
    def _log_potential(d, k, eps):
        """对数势能: Φ(d) = k * log(d + eps)"""
        return k * np.log(d + eps)
    
    def _compute_rebar_errors(self, payload_xy, payload_mat):
        """计算4根钢筋与对应方孔的XY偏差。"""
        rebar_pos = np.array([
            [ 0.035,  0.035], [ 0.035, -0.035],
            [-0.035,  0.035], [-0.035, -0.035]], dtype=np.float64)
        R_pl = payload_mat[:2, :2]
        errors = np.zeros(4)
        for i in range(4):
            hole_w = payload_xy + R_pl @ rebar_pos[i]
            rebar_w = self.target_pos + rebar_pos[i]
            errors[i] = float(np.linalg.norm(hole_w - rebar_w))
        return errors, float(np.max(errors)), float(np.mean(errors))

    def _compute_reward(self, action, current_q, prev_q, obs):
        """
        简化奖励 V5.1：
          - 增加吊装物姿态软约束（tilt/yaw 微小惩罚）
          - 其余同 V5（差分距离追踪 + 指数折扣下降奖励 + 终端奖励独立）
        """
        reward = 0.0; done = False; success = False; is_collision = False
        cfg_rwd   = self.config["reward"]
        cfg_logic = self.config["step_logic"]
        cfg_ins   = self.cfg_insertion

        payload_xy = obs[4:6].copy(); payload_vxy = obs[6:8].copy()
        payload_z  = self.data.body('prefab').xpos[2]
        dof_idx    = self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vz = self.data.qvel[dof_idx + 2]
        pl_vel     = float(np.linalg.norm(np.append(payload_vxy, payload_vz)))
        ee_pos     = self._get_ee_pos()
        ee_xy = ee_pos[:2]
        dtf = float(np.linalg.norm(payload_xy - self.target_pos))

        # 提前计算姿态（供后续使用）
        pl_mat   = self.data.body('prefab').xmat.reshape(3, 3)
        pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
        tilt     = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
        abs_yaw  = abs(float(pl_euler[2]))

        entry_z = cfg_ins.get("entry_z", 0.16)
        if not self._in_insertion_phase and payload_z <= entry_z:
            self._in_insertion_phase = True

        # ---------- SAFETY (unchanged) ----------
        instab_grace = cfg_logic.get("instability_grace_steps", 50)
        if cfg_logic.get("instability_check", True) and self.current_step >= instab_grace:
            unstable = False; reason_detail = []
            swing_xy = float(np.linalg.norm(ee_xy - payload_xy))
            if swing_xy  > cfg_logic.get("swing_xy_max", 0.25):
                unstable = True; reason_detail.append(f"swing={swing_xy:.3f}")
            if pl_vel    > cfg_logic.get("payload_vel_max", 2.0):
                unstable = True; reason_detail.append(f"vel={pl_vel:.2f}")
            if tilt      > cfg_logic.get("payload_tilt_max", 1.0):
                unstable = True; reason_detail.append(f"tilt={tilt:.2f}")
            if abs_yaw   > cfg_logic.get("payload_yaw_max", 1.2):
                unstable = True; reason_detail.append(f"yaw={abs_yaw:.2f}")
            if unstable:
                reward = float(cfg_rwd.get("instability_penalty", -5.0))
                self._termination_reason = "instability:" + ",".join(reason_detail)
                return reward, True, False, False

        use_mjc_contact = cfg_rwd.get("use_mujoco_contact", True)
        if use_mjc_contact:
            hit_obs, _ = self._check_prefab_collision_with_obstacles()
            if hit_obs:
                reward = float(cfg_rwd.get("collision_penalty", -10.0))
                self._termination_reason = "collision_obstacle"
                return reward, True, False, True
        else:
            for (ox, oy, orad) in self._obstacles:
                if float(np.linalg.norm(payload_xy - np.array([ox, oy]))) < \
                        (orad + self.payload_radius):
                    reward = float(cfg_rwd.get("collision_penalty", -10.0))
                    self._termination_reason = "collision_obstacle"
                    return reward, True, False, True

        if float(np.linalg.norm(payload_xy)) < 0.03:
            reward = float(cfg_rwd.get("collision_penalty", -10.0))
            self._termination_reason = "collision_base"
            return reward, True, False, True

        # ---------- PROGRESS REWARD ----------
        reward += -0.003   # step penalty

        # 差分距离奖励
        if self._planned_path is not None and not self.reached_final:
            max_look = min(3, len(self._planned_path) - self.current_wp_idx - 1)
            best_dist = float('inf')
            for i in range(max_look + 1):
                wp = self._planned_path[self.current_wp_idx + i]
                d = np.linalg.norm(payload_xy - wp[:2]) + 0.5 * abs(payload_z - wp[2])
                if d < best_dist:
                    best_dist = d
            ref_dist = best_dist
        else:
            target_pz = cfg_ins.get("target_payload_z", 0.10)
            ref_dist = dtf + 0.5 * abs(payload_z - target_pz)

        prev_ref = getattr(self, '_prev_ref_dist', None)
        if prev_ref is not None:
            progress = prev_ref - ref_dist
            reward += 4.0 * progress
        self._prev_ref_dist = ref_dist

        if getattr(self, '_wp_just_advanced', False):
            reward += 1.0
            self._wp_just_advanced = False

        # 姿态软约束（新增，防止剧烈摆动）
        reward -= 0.02 * tilt
        reward -= 0.02 * abs_yaw

        # 插入阶段下降奖励（指数折扣）
        if self._in_insertion_phase:
            target_pz = cfg_ins.get("target_payload_z", 0.10)
            cur_depth = max(0.0, entry_z - payload_z)
            prev_depth = getattr(self, '_prev_descent_depth', 0.0)
            depth_delta = cur_depth - prev_depth
            discount = np.exp(-dtf / 0.02)
            reward += 6.0 * depth_delta * discount
            self._prev_descent_depth = cur_depth

        # ---------- SUCCESS / SOFT SUCCESS / CRASH ----------
        z_tol    = cfg_ins.get("success_z_tolerance", 0.020)
        xy_tol   = cfg_ins.get("xy_tolerance",  0.005)
        tilt_tol = cfg_ins.get("tilt_tolerance", 0.08)
        yaw_tol  = cfg_ins.get("yaw_tolerance",  0.06)
        hold_steps = int(cfg_ins.get("hold_steps", 3))

        on_target = (abs(payload_z - target_pz) < z_tol and
                     dtf < xy_tol and
                     tilt < tilt_tol and abs_yaw < yaw_tol)

        if on_target:
            self._insertion_hold_counter += 1
        else:
            self._insertion_hold_counter = 0

        if self._insertion_hold_counter >= hold_steps:
            reward += cfg_rwd.get("success_bonus", 50.0)
            success = True; done = True
            self._termination_reason = (
                f"success:z={payload_z*1000:.0f}mm,dtf={dtf*1000:.1f}mm,"
                f"tilt={tilt:.3f},yaw={abs_yaw:.3f}")
            return reward, done, success, is_collision

        max_steps = self.config["sim"]["max_steps"]
        near_timeout = (self.current_step >= max_steps - 10)
        if self.reached_final and near_timeout:
            dist_frac  = max(0.0, 1.0 - dtf / 0.05)
            best_z = getattr(self, '_best_insertion_z', 0.10)
            depth_frac = max(0.0, min(1.0,
                (entry_z - best_z) / max(entry_z - target_pz, 1e-6)))
            pose_err = float(np.sqrt(tilt**2 + abs_yaw**2))
            pose_frac = max(0.0, 1.0 - pose_err / 0.3)
            rebar_frac = 0.0
            if dtf < 0.05:
                _, worst_err, _ = self._compute_rebar_errors(
                    payload_xy.astype(np.float64), pl_mat)
                rebar_frac = max(0.0, 1.0 - worst_err / 0.01)
            combined = 0.3*dist_frac + 0.2*depth_frac + 0.2*pose_frac + 0.3*rebar_frac
            reward += cfg_rwd.get("soft_success_bonus", 20.0) * combined
            self._termination_reason = (
                f"soft_success:score={combined:.2f},dtf={dtf*1000:.1f}mm,"
                f"best_z={best_z*1000:.0f}mm")
            done = True
            return reward, done, success, is_collision

        grace_steps = cfg_logic.get("crash_grace_steps", 30)
        if self.current_step >= grace_steps:
            if payload_z < cfg_logic["crash_z_threshold"] and \
                    payload_vz < cfg_logic["crash_vz_threshold"]:
                reward = float(cfg_rwd.get("crash_penalty", -8.0))
                self._termination_reason = f"crash:z={payload_z:.3f},vz={payload_vz:.2f}"
                return reward, True, False, False

        # ---------- 普通步骤裁剪 ----------
        r_min = cfg_logic.get("reward_clip_min", -5.0)
        r_max = cfg_logic.get("reward_clip_max",  5.0)
        reward = float(np.clip(reward, r_min, r_max))
        return reward, done, success, is_collision

    # ── _get_obs ───────────────────────────────────────────────────────────────

    def _get_obs(self):
        ee_pos=self._get_ee_pos(); ee_x,ee_y,ee_z=ee_pos
        ee_vx,ee_vy,ee_vz=self._ee_vel_cache
        mat=self._get_ee_mat(); ee_euler=R.from_matrix(mat).as_euler('xyz')
        ee_roll,ee_pitch,ee_yaw=ee_euler
        ee_roll_v,ee_pitch_v,ee_yaw_v=self._ee_euler_vel_cache

        payload_x=self.data.body('prefab').xpos[0]; payload_y=self.data.body('prefab').xpos[1]
        payload_z=self.data.body('prefab').xpos[2]
        dof_idx=self.model.jnt_dofadr[self.prefab_jnt_id]
        payload_vx=self.data.qvel[dof_idx]; payload_vy=self.data.qvel[dof_idx+1]
        payload_vz=self.data.qvel[dof_idx+2]
        rel_tx=self.target_pos[0]-payload_x; rel_ty=self.target_pos[1]-payload_y

        obs_data=[]
        for (ox,oy,r) in self._obstacles: obs_data.extend([ox,oy,r])
        tl=self.n_obstacles*3
        while len(obs_data)<tl: obs_data.append(0.0)

        joint_q=self.data.qpos[:7].copy().astype(np.float32)
        joint_dq=self.data.qvel[:7].copy().astype(np.float32)

        # [V3-OBS] payload姿态（替换原来的两个0.占位）
        pl_mat = self.data.body('prefab').xmat.reshape(3, 3)
        pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
        payload_tilt = float(np.sqrt(pl_euler[0]**2 + pl_euler[1]**2))
        payload_yaw_val = float(pl_euler[2])

        # [V3-OBS] 阶段编码 (one-hot 3维)
        entry_z = self.cfg_insertion.get("entry_z", 0.16)
        phase_encode = [0.0, 0.0, 0.0]
        if payload_z <= entry_z or getattr(self, '_in_insertion_phase', False):
            phase_encode[2] = 1.0   # INSERTION
        elif self.reached_final:
            phase_encode[1] = 1.0   # ALIGN
        else:
            phase_encode[0] = 1.0   # CRUISE

        # [V3-OBS] 航点进度
        if self._planned_path is not None and len(self._planned_path) > 0:
            progress = float(self.current_wp_idx) / len(self._planned_path)
        else:
            progress = 0.0

        # [V3-OBS] Z误差
        target_pz = self.cfg_insertion.get("target_payload_z", 0.10)
        z_error = float(payload_z - target_pz)

        # [V3-OBS] Per-rebar偏差 (4维)
        rebar_errors = [0.0, 0.0, 0.0, 0.0]
        payload_xy_arr = np.array([payload_x, payload_y])
        dtf_obs = float(np.linalg.norm(payload_xy_arr - self.target_pos))
        if dtf_obs < 0.05:
            rebar_pos = np.array([
                [ 0.035,  0.035], [ 0.035, -0.035],
                [-0.035,  0.035], [-0.035, -0.035]], dtype=np.float64)
            R_pl = pl_mat[:2, :2]
            for i in range(4):
                hole_w = payload_xy_arr + R_pl @ rebar_pos[i]
                rebar_w = self.target_pos + rebar_pos[i]
                rebar_errors[i] = float(np.linalg.norm(hole_w - rebar_w))

        return np.array(
            [ee_x,ee_y,ee_vx,ee_vy,payload_x,payload_y,payload_vx,payload_vy,rel_tx,rel_ty]
            +obs_data[:tl]
            +[ee_z,ee_vz,payload_z,payload_vz,
              ee_roll,ee_roll_v,ee_pitch,ee_pitch_v,
              ee_yaw,ee_yaw_v,
              payload_tilt, payload_yaw_val]
            +list(joint_q)+list(joint_dq)
            +phase_encode
            +[progress, z_error]
            +rebar_errors,
            dtype=np.float32)


def make_env(config=None):
    return CableRobotEnvWithObstacles(config=config)