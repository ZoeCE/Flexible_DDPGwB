# -*- coding: utf-8 -*-
"""
带路径障碍物的缆索机器人环境包装。

每次 reset 前在代码中生成带静态障碍物的临时 XML（障碍物样式与 rebar_base 一致：
固定圆柱、可碰撞、较长），写入 assets2 临时文件后加载，再跑本局模拟。
不修改原有 mujoco_env.py，本文件独立使用。
"""

import os
import tempfile
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque

from mujoco_env import CableRobotEnv


# 障碍物圆柱：中心高度与半高（米），覆盖负载运动范围
OBSTACLE_Z_CENTER = 0.45
OBSTACLE_HALFHEIGHT = 0.2


def _sample_obstacles_on_path(start_xy, target_xy, n_obstacles, radius_range,
                              path_width=0.15, rng=None):
    """
    在起点到目标的路径附近采样圆形障碍物。
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
    for _ in range(n_obstacles):
        t = rng.uniform(0.15, 0.85)
        along = start_xy + t * delta
        off = rng.uniform(-path_width, path_width)
        center = along + off * normal
        r = rng.uniform(r_min, r_max)
        obstacles.append((float(center[0]), float(center[1]), float(r)))
    return obstacles


def _build_xml_with_obstacles(base_xml_content, obstacles):
    """
    在基础 XML 中插入障碍物：asset 中加 obstacle 材质，worldbody 中在 floor 后插入
    若干静态圆柱 body（与 rebar_base 同风格：无 joint，geom cylinder，contype/conaffinity=1）。

    Args:
        base_xml_content: 原始 demo XML 全文。
        obstacles: list of (x, y, radius)。

    Returns:
        插入障碍物后的完整 XML 字符串。
    """
    # 1) 在 steel 材质后插入 obstacle 材质（仅当有障碍物时）
    if obstacles:
        material_line = '    <material name="steel" rgba="0.6 0.6 0.6 1"/>'
        insert = (
            '    <material name="steel" rgba="0.6 0.6 0.6 1"/>\n'
            '    <material name="obstacle" rgba="0.9 0.45 0.1 1"/>'
        )
        xml = base_xml_content.replace(material_line, insert, 1)
    else:
        xml = base_xml_content

    # 2) 在 floor 与「固定钢筋」之间插入障碍物 body
    obstacle_bodies = []
    for i, (x, y, r) in enumerate(obstacles):
        # 与 rebar_base 一致：静态 body，圆柱 geom，contype/conaffinity=1；body 中心在 (x,y,z)，圆柱竖直
        body = (
            f'    <!-- 路径障碍物 {i} (静态) -->\n'
            f'    <body name="obstacle_{i}" pos="{x} {y} {OBSTACLE_Z_CENTER}">\n'
            f'      <geom type="cylinder" size="{r} {OBSTACLE_HALFHEIGHT}" pos="0 0 0" '
            f'material="obstacle" contype="1" conaffinity="1"/>\n'
            f'    </body>\n\n'
        )
        obstacle_bodies.append(body)
    obstacles_block = '\n'.join(obstacle_bodies) if obstacle_bodies else ''
    replacement = (
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n'
        + (obstacles_block if obstacles_block else '')
        + '    <!--  固定钢筋 -->'
    )
    xml = xml.replace(
        '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>\n\n    <!--  固定钢筋 -->',
        replacement,
        1,
    )
    return xml


class CableRobotEnvWithObstacles(CableRobotEnv):
    """
    每次 reset 时根据本局障碍物在代码中生成临时 XML（静态障碍物，rebar_base 风格），
    写入 assets2 下的临时文件并加载，再执行父类 reset 完成本局初始化。
    """

    def __init__(self, n_obstacles=3, obstacle_radius_range=(0.04, 0.08),
                 path_width=0.12, obstacle_seed=None, **kwargs):
        """
        Args:
            n_obstacles: 每次 reset 在路径上生成的障碍物数量。
            obstacle_radius_range: (r_min, r_max) 障碍物半径范围（米）。
            path_width: 障碍物在路径两侧的采样半宽（米）。
            obstacle_seed: 随机种子，None 则每次不同。
            其余 **kwargs 传给 CableRobotEnv（如 render, control_freq_hz 等）。
        """
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

        # 先用基础 XML 加载一次，保证有合法 model/data（首局 reset 会再替换）
        self.model = mujoco.MjModel.from_xml_path(base_xml_path)
        self.data = mujoco.MjData(self.model)

        self.physics_dt = 0.002
        self.control_freq_hz = kwargs.get("control_freq_hz", 10)
        self.control_dt = 1.0 / self.control_freq_hz
        self.frame_skip = int(self.control_dt / self.physics_dt)
        self.dt = self.control_dt
        self.model.opt.timestep = self.physics_dt

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
        self.default_target = np.array([-0.2, 0.3])
        self.target_pos = self.default_target.copy()
        self.max_steps = 200
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

    def _reresolve_ids(self):
        """在替换 model/data 后重新解析 body/joint 与 mocap id。"""
        mocap_body = self.model.body("mocap")
        if hasattr(mocap_body, "mocapid"):
            mids = mocap_body.mocapid
            self.mocap_id = mids[0] if isinstance(mids, (np.ndarray, list)) else mids
        self.prefab_jnt_id = self.model.body("prefab").jntadr[0]
        self.prefab_body_id = self.model.body("prefab").id
        self.target_body_id = self.model.body("rebar_base").id

    def _reload_model_with_obstacles(self, obstacles):
        """根据障碍物列表生成临时 XML、加载新 model/data，并更新 viewer。"""
        xml_content = _build_xml_with_obstacles(self._base_xml_content, obstacles)
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
        # 用默认起点/目标采样障碍物（与父类 reset 的随机化无关，仅用于本局障碍物布局）
        start_xy = np.array([0.2, 0.3])
        target_xy = self.default_target.copy()
        self._obstacles = _sample_obstacles_on_path(
            start_xy, target_xy,
            self.n_obstacles,
            self.obstacle_radius_range,
            self.path_width,
            self._obstacle_rng,
        )
        self._reload_model_with_obstacles(self._obstacles)
        return super().reset()

    def step(self, action):
        return super().step(action)

    def get_obstacles(self):
        """返回当前 episode 的障碍物列表，每项 (x, y, radius)。"""
        return list(self._obstacles)
