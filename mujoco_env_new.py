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
import time
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R

from config import DEFAULT_CONFIG
from controller import NativeIKSolver
from vision_rgbd import CameraFrame, OpenCVRGBDPoseEstimator, make_camera_matrix


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
        self.episode_start_xy   = self.default_start_xy.copy()
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
        # [v12.3] 缓存绳索 body id, 用于 cable obs / energy 计算
        self._cache_cable_body_ids()
        self._cache_rope_marker_site_ids()

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
        self.wind_speed = 0.0          # user-facing wind speed (m/s)
        self._wind_curriculum_frac = 1.0  # 训练时的风力倍率（0→1）
        self._force_noise_sigma    = 0.0  # [v8] 环境噪声力 σ (N)

        # ── [VISION] 三 RGB-D 相机的轻量测量模型 ─────────────────────────────
        self.cfg_vision = self.config.get("vision", {})
        vision_seed = self.cfg_vision.get("seed", None)
        if vision_seed is None:
            vision_seed = (wind_seed + 17) if wind_seed is not None else None
        self.vision_rng = np.random.default_rng(vision_seed)
        self._vision_queue = deque()
        self._vision_current = None
        self._vision_last_generated = None
        self._vision_last_pos = None
        self._vision_last_vel = None
        self._vision_last_update_step = None
        self._vision_episode_bias = {}
        self._vision_estimator = None
        self._vision_renderer = None
        self._vision_scene_option = None
        self._vision_camera_ids = {}
        self._vision_frame_cache = {}
        self._vision_last_fused_pose = None
        self._vision_debug_dump_count = 0
        self._vision_last_timing = {}
        self._vision_last_camera_statuses = []

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

    # ────────────────────────────────────────────────────────────────────────
    # [v12.3 修订] 绳索每个 segment 的运动状态观测 (用户要求)
    #
    # 文献依据:
    #   - Kotaru et al. 2017 (arXiv:1711.04895): 多段绳建模, 每段 unit vector + omega
    #   - Goodarzi et al. 2014 (arXiv:1407.8164): geometric control of flexible cable
    #   - FLARE 2025 (arXiv:2508.09797): cable state in observation
    #
    # 我们的结构 (来自 generate_four_cables_with_plate.py):
    #   4 根绳 × 10 段 = 40 个 body (链接点)
    #   绳命名: rope_fl, rope_fr, rope_rl, rope_rr
    #   段命名: rope_<NAME>_root, rope_<NAME>_1, ..., rope_<NAME>_9
    #   (root 是顶段, 其余按序号)
    #   每段有 ball joint (3 DoF rotation)
    #
    # 每段观测: rel_pos(3) + lin_vel(3) = 6 维
    #   rel_pos: 该段 com 相对于"该绳上端锚点 ee 上的 attachment_site"的位置 (世界系)
    #   lin_vel: 该段 com 的线速度 (世界系, 从 cvel[3:6] 读)
    # 加速度: 不入 obs (RL 可通过相邻帧 vel 差分隐式推断, LSTM actor 尤其擅长)
    #
    # 总维度: 4 × 10 × 6 = 240 维
    # ────────────────────────────────────────────────────────────────────────
    CABLE_NAMES = ("rope_fl", "rope_fr", "rope_rl", "rope_rr")
    CABLE_OBS_PER_SEG = 6   # rel_pos(3) + lin_vel(3)
    CABLE_OBS_TOTAL = 4 * 10 * 6  # = 240 (4 ropes × 10 segs × 6 dims)

    def _cache_cable_body_ids(self):
        """缓存绳索分段 body id (在 _reresolve_ids 后调用)."""
        n_segs = int(self.config.get("rope", {}).get("num_segments", 10))
        self._cable_n_segs = n_segs

        self._cable_body_ids = []   # [n_cables][n_segs] — body ids
        for cname in self.CABLE_NAMES:
            this_cable = []
            # rope_<NAME>_root 是第 0 段, rope_<NAME>_1 ... rope_<NAME>_{n_segs-1}
            for si in range(n_segs):
                if si == 0:
                    bname = f"{cname}_root"
                else:
                    bname = f"{cname}_{si}"
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
                this_cable.append(bid)  # 即使 -1 也保留, 后续会检查
            self._cable_body_ids.append(this_cable)

        # 检查: 至少第一根第一段必须存在
        if (self._cable_body_ids and
            len(self._cable_body_ids[0]) > 0 and
            self._cable_body_ids[0][0] >= 0):
            self._cable_obs_available = True
        else:
            self._cable_obs_available = False
            # 仅首次警告 (避免每次 reset 都打)
            if not getattr(self, '_cable_warned', False):
                print(f"[v12.3] 警告: 找不到绳索 body (期望命名: rope_fl_root, rope_fl_1, ...). "
                      f"cable obs 将全 0, 不影响其他训练. 检查 generate_four_cables_with_plate.py 是否一致.")
                self._cable_warned = True

    def _cache_rope_marker_site_ids(self):
        cfg = self.config.get("rope_markers", {})
        self._rope_marker_sites = []
        self._rope_marker_sites_available = False
        if not bool(cfg.get("enabled", False)):
            return
        n_markers = max(0, int(cfg.get("markers_per_rope", 5)))
        for cname in self.CABLE_NAMES:
            for mi in range(n_markers):
                sname = f"{cname}_marker_{mi}"
                sid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, sname)
                self._rope_marker_sites.append({
                    "name": sname,
                    "rope": cname,
                    "marker_index": mi,
                    "site_id": int(sid),
                })
        self._rope_marker_sites_available = any(
            m["site_id"] >= 0 for m in self._rope_marker_sites)

    def get_rope_marker_world_positions(self):
        """Return visual rope-marker center positions in lab/world frame."""
        markers = []
        for meta in getattr(self, "_rope_marker_sites", []):
            sid = int(meta.get("site_id", -1))
            if sid < 0:
                continue
            markers.append({
                "name": str(meta["name"]),
                "rope": str(meta["rope"]),
                "marker_index": int(meta["marker_index"]),
                "pos": self.data.site_xpos[sid].copy().astype(np.float64),
            })
        return markers

    def _rope_marker_feature_source(self):
        cfg_pred = self.config.get("cable_latent_predictor", {})
        cfg_marker = self.config.get("rope_markers", {})
        source = str(cfg_pred.get(
            "rope_marker_feature_source",
            cfg_marker.get("feature_source", "site")) or "").strip().lower()
        if not source:
            source = str(cfg_marker.get("feature_source", "site")).lower()
        return source

    def _rope_marker_feature_reference_position(self, reference=None):
        cfg = self.config.get("cable_latent_predictor", {})
        reference = str(reference or cfg.get(
            "rope_marker_feature_reference", "ee")).lower()
        if reference == "payload":
            return self.data.body('prefab').xpos.copy().astype(np.float64)
        return self._get_ee_pos().astype(np.float64)

    def _rope_marker_noise_vec(self, key, default=0.0):
        cfg = self.config.get("rope_markers", {})
        val = cfg.get(key, default)
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=np.float64).reshape(-1)
            if arr.size == 1:
                return np.full(3, float(arr[0]), dtype=np.float64)
            if arr.size >= 3:
                return arr[:3].astype(np.float64)
        return np.full(3, float(val), dtype=np.float64)

    def _postprocess_rope_marker_position(self, p_lab, valid):
        """Apply deployable marker measurement noise and dropout."""
        cfg = self.config.get("rope_markers", {})
        if not valid:
            return None, False, "missing"
        if self.vision_rng.random() < float(cfg.get("feature_dropout_prob", 0.0)):
            return None, False, "dropout"
        out = np.asarray(p_lab, dtype=np.float64).reshape(3).copy()
        sigma = self._rope_marker_noise_vec("feature_pos_noise_std", 0.0)
        if np.any(sigma > 0.0):
            out += self.vision_rng.normal(0.0, sigma, size=3)
        if self.vision_rng.random() < float(cfg.get("feature_outlier_prob", 0.0)):
            out += self.vision_rng.normal(
                0.0, float(cfg.get("feature_outlier_std", 0.0)), size=3)
        return out, True, "ok"

    def _rope_marker_rgbd_world_estimates(self):
        """Estimate marker centers using RGB-D depth back-projection.

        The current simulator uses an ideal marker-center detector: MuJoCo site
        positions define the image pixel to sample, while the 3D position fed to
        the policy is reconstructed from rendered depth and camera calibration.
        This keeps the training path deployable in geometry/noise/masking terms,
        while explicit image-level marker detection remains a later hardware
        integration step.
        """
        markers = self.get_rope_marker_world_positions()
        cfg_marker = self.config.get("rope_markers", {})
        depth_tol = float(cfg_marker.get(
            "visibility_depth_tolerance", 0.035))
        depth_radius = int(cfg_marker.get("visibility_depth_window", 2))
        quantize_px = bool(cfg_marker.get("feature_quantize_px", False))
        pixel_noise = float(cfg_marker.get("feature_pixel_noise_std", 0.0))
        width, height = self._vision_resolution()

        fused = {}
        in_fov = set()
        camera_est_count = 0
        camera_count = 0
        for cam_cfg in list(self.cfg_vision.get("cameras", [])):
            cam_name = str(cam_cfg.get("name", "rgbd"))
            if self._vision_camera_id(cam_name) < 0:
                continue
            rendered = self._render_rgbd_camera(cam_cfg)
            if rendered is None:
                continue
            _rgb, depth = rendered
            if depth is None:
                continue
            camera_count += 1
            K = self._vision_camera_matrix(cam_cfg)
            T_lab_cam = self._vision_camera_lab_transform(cam_cfg)
            R_lab_cam = T_lab_cam[:3, :3]
            t_lab_cam = T_lab_cam[:3, 3]
            max_range = float(cam_cfg.get("max_range", np.inf))

            for marker in markers:
                name = str(marker["name"])
                p_lab = np.asarray(marker["pos"], dtype=np.float64).reshape(3)
                p_cam = R_lab_cam.T @ (p_lab - t_lab_cam)
                z_true = float(p_cam[2])
                if z_true <= 1e-6 or z_true > max_range:
                    continue
                u = float(K[0, 0] * p_cam[0] / z_true + K[0, 2])
                v = float(K[1, 1] * p_cam[1] / z_true + K[1, 2])
                if not (0.0 <= u < width and 0.0 <= v < height):
                    continue
                in_fov.add(name)
                u_det = round(u) if quantize_px else u
                v_det = round(v) if quantize_px else v
                if pixel_noise > 0.0:
                    u_det += float(self.vision_rng.normal(0.0, pixel_noise))
                    v_det += float(self.vision_rng.normal(0.0, pixel_noise))
                z_depth = self._sample_depth_patch(
                    depth, u_det, v_det, radius=depth_radius)
                if z_depth is None:
                    continue
                z_depth = float(z_depth)
                if abs(z_depth - z_true) > depth_tol:
                    continue
                x = (float(u_det) - K[0, 2]) * z_depth / K[0, 0]
                y = (float(v_det) - K[1, 2]) * z_depth / K[1, 1]
                p_est_lab = R_lab_cam @ np.array([x, y, z_depth]) + t_lab_cam
                fused.setdefault(name, []).append(p_est_lab)
                camera_est_count += 1

        estimates = {
            name: np.mean(np.asarray(points, dtype=np.float64), axis=0)
            for name, points in fused.items()
            if len(points) > 0
        }
        self._rope_marker_feature_last_diag = {
            "source": "rgbd",
            "markers_total": len(markers),
            "in_fov": len(in_fov),
            "visible": len(estimates),
            "camera_estimates": int(camera_est_count),
            "cameras": int(camera_count),
        }
        return estimates

    @staticmethod
    def _rope_marker_hue_mask(hue_img, target_hue, tolerance):
        hue = np.asarray(hue_img, dtype=np.int16)
        target = int(target_hue) % 180
        diff = np.abs(hue - target)
        diff = np.minimum(diff, 180 - diff)
        return diff <= int(tolerance)

    def _cluster_rope_marker_image_candidates(self, candidates):
        cfg = self.config.get("rope_markers", {})
        radius = max(1.0, float(cfg.get("opencv_cluster_px", 16.0)))
        out = []
        by_rope = {}
        for cand in candidates:
            by_rope.setdefault(str(cand.get("rope", "")), []).append(cand)
        for rope, rope_candidates in by_rope.items():
            clusters = []
            for cand in sorted(
                    rope_candidates,
                    key=lambda c: float(c.get("area", 0.0)),
                    reverse=True):
                uv = np.array([float(cand["u"]), float(cand["v"])],
                              dtype=np.float64)
                best_i = -1
                best_d = radius
                for ci, cluster in enumerate(clusters):
                    c_uv = np.array([cluster["u"], cluster["v"]],
                                    dtype=np.float64)
                    dist = float(np.linalg.norm(uv - c_uv))
                    if dist <= best_d:
                        best_d = dist
                        best_i = ci
                if best_i < 0:
                    clusters.append({
                        "rope": rope,
                        "u": float(uv[0]),
                        "v": float(uv[1]),
                        "area": float(cand.get("area", 1.0)),
                        "members": [cand],
                    })
                else:
                    cluster = clusters[best_i]
                    cluster["members"].append(cand)
                    weights = np.asarray([
                        max(1e-6, float(m.get("area", 1.0)))
                        for m in cluster["members"]], dtype=np.float64)
                    us = np.asarray([float(m["u"]) for m in cluster["members"]],
                                    dtype=np.float64)
                    vs = np.asarray([float(m["v"]) for m in cluster["members"]],
                                    dtype=np.float64)
                    cluster["area"] = float(np.sum(weights))
                    cluster["u"] = float(np.average(us, weights=weights))
                    cluster["v"] = float(np.average(vs, weights=weights))
            out.extend(clusters)
        return out

    def _detect_rope_marker_blobs_opencv(self, rgb):
        """Detect colored rope marker blobs from an RGB image with OpenCV."""
        try:
            import cv2
        except Exception as exc:
            self._rope_marker_opencv_last_error = f"opencv_import_failed:{exc}"
            return []

        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] < 3:
            return []
        if img.dtype != np.uint8:
            if np.nanmax(img) <= 1.5:
                img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)
        img = img[:, :, :3]
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        cfg = self.config.get("rope_markers", {})
        hue_tol = int(cfg.get("opencv_hue_tolerance", 12))
        sat_min = int(cfg.get("opencv_min_saturation", 70))
        val_min = int(cfg.get("opencv_min_value", 70))
        min_area = float(cfg.get("opencv_min_area_px", 4.0))
        max_area = float(cfg.get("opencv_max_area_px", 2000.0))
        kernel_size = int(cfg.get("opencv_morph_kernel", 3))
        kernel = None
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        candidates = []
        colors = cfg.get("colors", {}) or {}
        for rope in self.CABLE_NAMES:
            rgba = np.asarray(
                colors.get(rope, [1.0, 0.85, 0.05, 1.0]),
                dtype=np.float64).reshape(-1)
            rgb_u8 = np.clip(rgba[:3] * 255.0, 0, 255).astype(np.uint8)
            target_hsv = cv2.cvtColor(
                rgb_u8.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
            mask = self._rope_marker_hue_mask(
                hue, int(target_hsv[0]), hue_tol)
            mask = mask & (sat >= sat_min) & (val >= val_min)
            mask_u8 = (mask.astype(np.uint8) * 255)
            if kernel is not None:
                mask_u8 = cv2.morphologyEx(
                    mask_u8, cv2.MORPH_OPEN, kernel)
                mask_u8 = cv2.morphologyEx(
                    mask_u8, cv2.MORPH_CLOSE, kernel)
            contours, _hier = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < min_area or area > max_area:
                    continue
                moment = cv2.moments(contour)
                if abs(float(moment.get("m00", 0.0))) < 1e-9:
                    continue
                u = float(moment["m10"] / moment["m00"])
                v = float(moment["m01"] / moment["m00"])
                x, y, w, h = cv2.boundingRect(contour)
                local = np.zeros((h, w), dtype=np.uint8)
                shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
                cv2.drawContours(local, [shifted], -1, 255, thickness=-1)
                yy, xx = np.nonzero(local)
                pixels = np.column_stack((xx + x, yy + y)).astype(np.float64)
                candidates.append({
                    "rope": rope,
                    "u": u,
                    "v": v,
                    "area": area,
                    "pixels": pixels,
                })
        return self._cluster_rope_marker_image_candidates(candidates)

    def _backproject_rope_marker_cluster(self, cluster, depth, K,
                                         R_lab_cam, t_lab_cam):
        if depth is None:
            return None
        arr = np.asarray(depth, dtype=np.float64)
        h, w = arr.shape[:2]
        cfg = self.config.get("rope_markers", {})
        quantize_px = bool(cfg.get("feature_quantize_px", False))
        pixel_noise = float(cfg.get("feature_pixel_noise_std", 0.0))
        depth_radius = int(cfg.get(
            "opencv_depth_window",
            cfg.get("visibility_depth_window", 2)))
        points = []
        for member in list(cluster.get("members", [])):
            pixels = np.asarray(member.get("pixels", []),
                                dtype=np.float64).reshape(-1, 2)
            if pixels.size > 0:
                if pixels.shape[0] > 80:
                    step = max(1, int(np.ceil(pixels.shape[0] / 80.0)))
                    pixels = pixels[::step]
                uu = pixels[:, 0].copy()
                vv = pixels[:, 1].copy()
                if pixel_noise > 0.0:
                    uu += self.vision_rng.normal(0.0, pixel_noise,
                                                 size=uu.shape)
                    vv += self.vision_rng.normal(0.0, pixel_noise,
                                                 size=vv.shape)
                if quantize_px:
                    uu = np.round(uu)
                    vv = np.round(vv)
                ui = np.rint(uu).astype(np.int64)
                vi = np.rint(vv).astype(np.int64)
                ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
                if np.any(ok):
                    z = arr[vi[ok], ui[ok]]
                    good = np.isfinite(z) & (z > 1e-6)
                    if np.any(good):
                        uu_g = uu[ok][good]
                        vv_g = vv[ok][good]
                        z_g = z[good]
                        x = (uu_g - K[0, 2]) * z_g / K[0, 0]
                        y = (vv_g - K[1, 2]) * z_g / K[1, 1]
                        cam_pts = np.column_stack((x, y, z_g))
                        lab_pts = (R_lab_cam @ cam_pts.T).T + t_lab_cam
                        points.append(np.mean(lab_pts, axis=0))
                        continue

            u_det = float(member.get("u", cluster.get("u", 0.0)))
            v_det = float(member.get("v", cluster.get("v", 0.0)))
            if pixel_noise > 0.0:
                u_det += float(self.vision_rng.normal(0.0, pixel_noise))
                v_det += float(self.vision_rng.normal(0.0, pixel_noise))
            if quantize_px:
                u_det = round(u_det)
                v_det = round(v_det)
            z_depth = self._sample_depth_patch(
                depth, u_det, v_det, radius=depth_radius)
            if z_depth is None:
                continue
            x = (u_det - K[0, 2]) * float(z_depth) / K[0, 0]
            y = (v_det - K[1, 2]) * float(z_depth) / K[1, 1]
            points.append(
                R_lab_cam @ np.array([x, y, float(z_depth)]) + t_lab_cam)

        if not points:
            return None
        return np.mean(np.asarray(points, dtype=np.float64), axis=0)

    def _cluster_rope_marker_world_points(self, world_points, n_markers):
        cfg = self.config.get("rope_markers", {})
        cluster_dist = max(1e-4, float(cfg.get("opencv_world_cluster_m", 0.025)))
        estimates = {}
        world_cluster_count = 0
        support_count = 0
        for rope in self.CABLE_NAMES:
            entries = list(world_points.get(rope, []))
            clusters = []
            for entry in entries:
                p = np.asarray(entry["pos"], dtype=np.float64).reshape(3)
                best_i = -1
                best_d = cluster_dist
                for ci, cluster in enumerate(clusters):
                    center = np.asarray(cluster["center"], dtype=np.float64)
                    dist = float(np.linalg.norm(p - center))
                    if dist <= best_d:
                        best_d = dist
                        best_i = ci
                if best_i < 0:
                    clusters.append({
                        "points": [p],
                        "weights": [max(1e-6, float(entry.get("area", 1.0)))],
                        "cameras": {str(entry.get("camera", ""))},
                        "center": p.copy(),
                    })
                else:
                    cluster = clusters[best_i]
                    cluster["points"].append(p)
                    cluster["weights"].append(
                        max(1e-6, float(entry.get("area", 1.0))))
                    cluster["cameras"].add(str(entry.get("camera", "")))
                    cluster["center"] = np.average(
                        np.asarray(cluster["points"], dtype=np.float64),
                        axis=0,
                        weights=np.asarray(cluster["weights"],
                                           dtype=np.float64))

            clusters.sort(key=lambda c: float(c["center"][2]), reverse=True)
            world_cluster_count += len(clusters)
            for mi, cluster in enumerate(clusters[:n_markers]):
                estimates[f"{rope}_marker_{mi}"] = np.asarray(
                    cluster["center"], dtype=np.float64).copy()
                support_count += len(cluster["points"])
        return estimates, world_cluster_count, support_count

    def _rope_marker_opencv_rgbd_world_estimates(self):
        """Estimate rope markers from RGB color blobs and RGB-D back-projection."""
        t_total0 = time.perf_counter()
        render_time_s = 0.0
        cfg_marker = self.config.get("rope_markers", {})
        n_markers = max(0, int(cfg_marker.get("markers_per_rope", 5)))
        markers_total = len(getattr(self, "_rope_marker_sites", []))
        world_points = {rope: [] for rope in self.CABLE_NAMES}
        detected_blobs = 0
        image_clusters = 0
        camera_est_count = 0
        camera_count = 0
        camera_names = []
        error = ""

        for cam_cfg in list(self.cfg_vision.get("cameras", [])):
            cam_name = str(cam_cfg.get("name", "rgbd"))
            if self._vision_camera_id(cam_name) < 0:
                continue
            t_render0 = time.perf_counter()
            rendered = self._render_rgbd_camera(cam_cfg)
            render_time_s += time.perf_counter() - t_render0
            if rendered is None:
                continue
            rgb, depth = rendered
            if depth is None:
                continue
            camera_count += 1
            camera_names.append(cam_name)
            clusters = self._detect_rope_marker_blobs_opencv(rgb)
            image_clusters += len(clusters)
            detected_blobs += sum(len(c.get("members", [])) for c in clusters)
            if not clusters and getattr(
                    self, "_rope_marker_opencv_last_error", ""):
                error = str(self._rope_marker_opencv_last_error)
            K = self._vision_camera_matrix(cam_cfg)
            T_lab_cam = self._vision_camera_lab_transform(cam_cfg)
            R_lab_cam = T_lab_cam[:3, :3]
            t_lab_cam = T_lab_cam[:3, 3]
            for cluster in clusters:
                p_lab = self._backproject_rope_marker_cluster(
                    cluster, depth, K, R_lab_cam, t_lab_cam)
                if p_lab is None:
                    continue
                rope = str(cluster.get("rope", ""))
                if rope not in world_points:
                    continue
                world_points[rope].append({
                    "pos": p_lab,
                    "area": float(cluster.get("area", 1.0)),
                    "camera": cam_name,
                })
                camera_est_count += 1

        estimates, world_clusters, support_count = (
            self._cluster_rope_marker_world_points(world_points, n_markers))
        total_s = time.perf_counter() - t_total0
        algo_s = max(0.0, total_s - render_time_s)
        self._rope_marker_feature_last_diag = {
            "source": "opencv_rgbd",
            "markers_total": markers_total,
            "in_fov": int(image_clusters),
            "visible": len(estimates),
            "camera_estimates": int(camera_est_count),
            "cameras": int(camera_count),
            "detected_blobs": int(detected_blobs),
            "image_clusters": int(image_clusters),
            "world_clusters": int(world_clusters),
            "world_cluster_support": int(support_count),
            "camera_names": camera_names,
            "total_ms": float(1000.0 * total_s),
            "render_ms": float(1000.0 * render_time_s),
            "algo_ms": float(1000.0 * algo_s),
        }
        if error:
            self._rope_marker_feature_last_diag["error"] = error
        return estimates

    def get_rope_marker_feature_vector(self, reference=None):
        """Return deployable rope-marker features for CableLatPred.

        The feature order is fixed by CABLE_NAMES and marker index:
        [rel_x, rel_y, rel_z, valid_mask] for each marker. In real tests the
        same vector should be filled by multi-camera marker triangulation.
        """
        source = self._rope_marker_feature_source()
        ref = self._rope_marker_feature_reference_position(reference)
        rgbd_estimates = None
        if source in ("rgbd", "vision", "backproject", "backprojection"):
            rgbd_estimates = self._rope_marker_rgbd_world_estimates()
        elif source in ("opencv_rgbd", "opencv", "color_rgbd", "color"):
            rgbd_estimates = self._rope_marker_opencv_rgbd_world_estimates()
        else:
            self._rope_marker_feature_last_diag = {
                "source": "site",
                "markers_total": len(getattr(self, "_rope_marker_sites", [])),
                "in_fov": 0,
                "visible": len(getattr(self, "_rope_marker_sites", [])),
                "camera_estimates": 0,
                "cameras": 0,
            }
        out = []
        n_valid = 0
        n_dropout = 0
        for meta in getattr(self, "_rope_marker_sites", []):
            sid = int(meta.get("site_id", -1))
            name = str(meta.get("name", ""))
            if sid < 0:
                p_raw = None
            elif rgbd_estimates is not None:
                p_raw = rgbd_estimates.get(name, None)
            else:
                p_raw = self.data.site_xpos[sid].copy().astype(np.float64)
            p_est, valid, reason = self._postprocess_rope_marker_position(
                p_raw, p_raw is not None)
            if not valid:
                if reason == "dropout":
                    n_dropout += 1
                out.extend([0.0, 0.0, 0.0, 0.0])
            else:
                rel = p_est - ref
                out.extend([float(rel[0]), float(rel[1]), float(rel[2]), 1.0])
                n_valid += 1
        diag = getattr(self, "_rope_marker_feature_last_diag", {}) or {}
        diag["valid_after_noise"] = int(n_valid)
        diag["dropout"] = int(n_dropout)
        self._rope_marker_feature_last_diag = diag
        return np.asarray(out, dtype=np.float32)

    def get_rope_marker_feature_debug(self):
        return dict(getattr(self, "_rope_marker_feature_last_diag", {}) or {})

    def get_cable_segment_states(self):
        """
        返回每根绳每段的 (相对位置, 线速度).

        Returns:
            seg_obs: np.array shape (4*10*6,) = (240,), float32
                每 6 维: [rel_x, rel_y, rel_z, vx, vy, vz]
                顺序: rope_fl seg0..9, rope_fr seg0..9, rope_rl seg0..9, rope_rr seg0..9
                rel_pos: seg com 减去 attachment_site 的世界位置
                lin_vel: cvel[3:6] (世界系)
        """
        if not getattr(self, '_cable_obs_available', False):
            return np.zeros(self.CABLE_OBS_TOTAL, np.float32)

        # 上端锚点位置 (4 根绳共用 attachment_site, 即 ee 上的连接点)
        # 注: 实际 root_pos 是相对于 attachment_site 的偏移 (h, h, 0) 等,
        #     但绝对位置我们用 attachment_site (在 _get_ee_pos 上方)
        anchor_world = self._get_ee_pos()   # 简化: 用 ee site 作 anchor

        out = np.zeros(self.CABLE_OBS_TOTAL, np.float32)
        idx = 0
        for ci in range(4):
            for si in range(self._cable_n_segs):
                bid = self._cable_body_ids[ci][si]
                if bid < 0:
                    idx += 6
                    continue
                # 段 com 位置 (世界系)
                p_seg = self.data.body(bid).xpos.copy()
                rel = p_seg - anchor_world
                out[idx:idx+3] = rel.astype(np.float32)
                # 段 com 线速度 (cvel: 6=angular(3)+linear(3), 世界系)
                cvel = self.data.cvel[bid]
                lin_vel = cvel[3:6]
                out[idx+3:idx+6] = lin_vel.astype(np.float32)
                idx += 6
        return out

    def _get_cable_energy(self):
        """
        返回所有绳所有段的总动能 (近似), 用作 reward 防摆项.

        cable_kinetic_energy = sum over segments: 0.5 * m_seg * ||v_seg||²
        但因为各段 mass 相同 (default 0.01 kg), 我们直接用 v² 总和, mass 通过 coef 调.

        Returns:
            cable_kinetic_energy: float, 单位 m²/s² (没乘 mass)
        """
        if not getattr(self, '_cable_obs_available', False):
            return 0.0
        total_v_sq = 0.0
        for ci in range(4):
            for si in range(self._cable_n_segs):
                bid = self._cable_body_ids[ci][si]
                if bid < 0:
                    continue
                v = self.data.cvel[bid][3:6]
                total_v_sq += float(v[0]**2 + v[1]**2 + v[2]**2)
        return total_v_sq

    def _launch_viewer(self):
        kw = {}
        if hasattr(self, '_key_callback') and self._key_callback is not None:
            kw['key_callback'] = self._key_callback
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, **kw)
        self._configure_viewer_visuals()

    def _configure_viewer_visuals(self):
        if self.viewer is None or not bool(self.cfg_vision.get(
                "show_camera_models", True)):
            return
        group = int(self.cfg_vision.get("camera_model_geom_group", 5))
        try:
            opt = getattr(self.viewer, "opt", None)
            if opt is not None and 0 <= group < len(opt.geomgroup):
                opt.geomgroup[group] = 1
        except Exception:
            pass

    def get_planned_path(self):
        return self._planned_path

    # ── [WIND] 风力扰动方法 ──────────────────────────────────────────────────

    def _wind_speed_to_force(self, speed_mps: float) -> float:
        """Convert wind speed (m/s) to horizontal force (N)."""
        v = max(0.0, float(speed_mps))
        rho = float(self.cfg_wind.get("air_density", 1.225))
        cd = float(self.cfg_wind.get("drag_coefficient", 1.30))
        area = float(self.cfg_wind.get("projected_area", 0.020))
        force = 0.5 * rho * cd * area * v * v
        return float(min(force, float(self.cfg_wind.get("F_max", force))))

    def _update_wind(self):
        """缓慢随机游走更新风向和风力大小（每个物理子步调用）。"""
        if getattr(self, '_test_wind_mode', False):
            return
        if not self.cfg_wind.get("enabled", False):
            return
        dt = self.physics_dt
        theta_std = self.cfg_wind.get("theta_rate_std", 0.15)
        self.wind_theta += dt * self.wind_rng.normal(0, theta_std)
        speed_std = self.cfg_wind.get("speed_rate_std", 0.50)
        speed_max = self.cfg_wind.get("speed_max", 16.5)
        self.wind_speed += dt * self.wind_rng.normal(0, speed_std)
        self.wind_speed = float(np.clip(self.wind_speed, 0, speed_max))
        self.wind_F = self._wind_speed_to_force(self.wind_speed)

    def _apply_wind_load(self):
        """将风力作为外力施加到 payload body 上（每个物理子步调用）。"""
        if not self.cfg_wind.get("enabled", False):
            return
        effective_F = self.wind_F * self._wind_curriculum_frac
        fx = effective_F * np.cos(self.wind_theta)
        fy = effective_F * np.sin(self.wind_theta)
        self.data.xfrc_applied[self.prefab_body_id, :3] = [fx, fy, 0.0]

    def set_wind_curriculum(self, frac: float):
        """设置风力课程学习倍率，0.0=无风，1.0=全风力。"""
        self._test_wind_mode = False
        self._wind_curriculum_frac = float(np.clip(frac, 0.0, 1.0))
        if self._wind_curriculum_frac <= 0.0:
            self.wind_F = 0.0
            self.wind_speed = 0.0
            if hasattr(self, 'data') and hasattr(self, 'prefab_body_id'):
                self.data.xfrc_applied[self.prefab_body_id, :3] = [0.0, 0.0, 0.0]

    def set_wind_speed(self, speed_mps: float, direction_rad: float = 0.0):
        """Apply a fixed wind speed (m/s), converting it to payload force."""
        if abs(float(speed_mps)) <= 1e-12:
            self.clear_wind()
            return

        self.wind_speed = float(speed_mps)
        self.wind_F     = self._wind_speed_to_force(self.wind_speed)
        self.wind_theta = float(direction_rad)
        self._test_wind_speed = float(speed_mps)
        self._test_wind_dir = float(direction_rad)
        self._test_wind_initial_speed = float(speed_mps)
        self._test_wind_initial_dir = float(direction_rad)
        self._wind_curriculum_frac = 1.0
        self._test_wind_mode = True
        self._apply_test_wind()

    def _apply_test_wind(self):
        """在 step 中持续施加测试风力（不受随机游走影响）。"""
        if not getattr(self, '_test_wind_mode', False):
            return
        if not (hasattr(self, 'data') and hasattr(self, 'prefab_body_id')):
            return
        speed = float(getattr(self, '_test_wind_speed', self.wind_speed))
        theta = float(getattr(self, '_test_wind_dir', self.wind_theta))
        if bool(self.cfg_wind.get("test_wind_variable", False)):
            dt = float(getattr(self, "physics_dt", 0.002))
            init_speed = float(getattr(
                self, '_test_wind_initial_speed', speed))
            speed_band = max(
                float(self.cfg_wind.get("test_speed_band_abs", 0.50)),
                abs(init_speed) * float(self.cfg_wind.get(
                    "test_speed_band_frac", 0.15)))
            speed_pull = float(self.cfg_wind.get(
                "test_speed_mean_reversion", 0.80))
            speed_std = float(self.cfg_wind.get("test_speed_rate_std", 0.25))
            speed += speed_pull * (init_speed - speed) * dt
            speed += speed_std * np.sqrt(max(dt, 1e-9)) * self.wind_rng.normal()
            speed_max = float(self.cfg_wind.get("speed_max", 16.5))
            speed = float(np.clip(
                speed,
                max(0.0, init_speed - speed_band),
                min(speed_max, init_speed + speed_band)))

            init_theta = float(getattr(
                self, '_test_wind_initial_dir', theta))
            dir_pull = float(self.cfg_wind.get(
                "test_dir_mean_reversion", 0.50))
            dir_std = float(self.cfg_wind.get("test_dir_rate_std", 0.08))
            dir_band = float(self.cfg_wind.get("test_dir_band_rad", 0.35))
            wrap = lambda a: (a + np.pi) % (2.0 * np.pi) - np.pi
            theta += dir_pull * wrap(init_theta - theta) * dt
            theta += dir_std * np.sqrt(max(dt, 1e-9)) * self.wind_rng.normal()
            theta = init_theta + float(np.clip(
                wrap(theta - init_theta), -dir_band, dir_band))
            self._test_wind_speed = speed
            self._test_wind_dir = theta
        force = self._wind_speed_to_force(speed)
        self.wind_speed = speed
        self.wind_F = force
        self.wind_theta = theta
        fx = force * np.cos(theta)
        fy = force * np.sin(theta)
        self.data.xfrc_applied[self.prefab_body_id, :3] = [fx, fy, 0.0]

    def clear_wind(self):
        """Disable externally forced wind and clear any residual xfrc."""
        self._test_wind_mode = False
        self._test_wind_speed = 0.0
        self._test_wind_dir = 0.0
        self._wind_curriculum_frac = 0.0
        self.wind_F = 0.0
        self.wind_speed = 0.0
        self.wind_theta = 0.0
        if hasattr(self, 'data') and hasattr(self, 'prefab_body_id'):
            self.data.xfrc_applied[self.prefab_body_id, :3] = [0.0, 0.0, 0.0]
    def get_wind_speed_state(self):
        """返回当前风速状态 (m/s, direction)，供观测构建和日志使用。"""
        if getattr(self, '_test_wind_mode', False):
            return float(self.wind_speed), float(self.wind_theta)
        effective_speed = self.wind_speed * self._wind_curriculum_frac
        return float(effective_speed), float(self.wind_theta)

    # ── [VISION] RGB-D 多相机测量模型 ───────────────────────────────────────
    def _vision_enabled(self):
        cfg = getattr(self, "cfg_vision", self.config.get("vision", {}))
        return bool(cfg.get("enabled", False))

    def _vision_total_delay_steps(self):
        cfg = getattr(self, "cfg_vision", self.config.get("vision", {}))
        return max(0, int(cfg.get("latency_steps", 0))) + \
            max(0, int(cfg.get("processing_delay_steps", 0)))

    @staticmethod
    def _as_vec3(value, default):
        arr = np.asarray(value if value is not None else default,
                         dtype=np.float64).reshape(-1)
        if arr.size < 3:
            arr = np.pad(arr, (0, 3 - arr.size))
        return arr[:3]

    def _vision_source(self):
        cfg = getattr(self, "cfg_vision", self.config.get("vision", {}))
        return str(cfg.get("source", "opencv_rgbd")).lower()

    def _reset_vision_runtime_after_model_change(self):
        if getattr(self, "_vision_renderer", None) is not None:
            try: self._vision_renderer.close()
            except Exception: pass
        self._vision_renderer = None
        self._vision_scene_option = None
        self._vision_camera_ids = {}
        self._vision_frame_cache = {}

    def _ensure_vision_estimator(self):
        if self._vision_estimator is None:
            opencv_cfg = copy.deepcopy(self.cfg_vision.get("opencv", {}))
            self._vision_estimator = OpenCVRGBDPoseEstimator(opencv_cfg)
        return self._vision_estimator

    def _vision_resolution(self):
        return (
            int(self.cfg_vision.get("render_width", 640)),
            int(self.cfg_vision.get("render_height", 480)),
        )

    def _vision_camera_id(self, name):
        if name not in self._vision_camera_ids:
            self._vision_camera_ids[name] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return int(self._vision_camera_ids[name])

    def _vision_camera_matrix(self, cam_cfg):
        width, height = self._vision_resolution()
        K_cfg = cam_cfg.get("K", None)
        if K_cfg is not None:
            return np.asarray(K_cfg, dtype=np.float64).reshape(3, 3)
        return make_camera_matrix(width, height, float(cam_cfg.get("fovy", 70.0)))

    def _vision_camera_distortion(self, cam_cfg):
        return np.asarray(cam_cfg.get("dist", [0, 0, 0, 0, 0]),
                          dtype=np.float64).reshape(-1)

    def _vision_camera_lab_transform(self, cam_cfg):
        name = str(cam_cfg.get("name", ""))
        cam_id = self._vision_camera_id(name)
        T = np.eye(4, dtype=np.float64)
        # MuJoCo/OpenGL camera frame: x right, y up, forward -z.
        # OpenCV camera frame: x right, y down, forward +z.
        R_mj_from_cv = np.diag([1.0, -1.0, -1.0])
        if cam_id >= 0:
            T[:3, 3] = self.data.cam_xpos[cam_id].copy()
            R_lab_mj = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
            T[:3, :3] = R_lab_mj @ R_mj_from_cv
            return T
        T[:3, 3] = self._as_vec3(cam_cfg.get("pos"), [0, 0, 1])
        R_lab_mj = R.from_euler(
            "xyz", self._as_vec3(cam_cfg.get("euler"), [0, 0, 0])).as_matrix()
        T[:3, :3] = R_lab_mj @ R_mj_from_cv
        return T

    def _ensure_vision_renderer(self):
        width, height = self._vision_resolution()
        if self._vision_renderer is None:
            self._vision_renderer = mujoco.Renderer(
                self.model, height=height, width=width)
        return self._vision_renderer

    def _vision_render_scene_option(self):
        if self._vision_scene_option is None:
            opt = mujoco.MjvOption()
            group = int(self.cfg_vision.get("camera_model_geom_group", 5))
            if 0 <= group < len(opt.geomgroup):
                opt.geomgroup[group] = 0
            if bool(self.cfg_vision.get("hide_sites_in_vision", True)):
                for i in range(len(opt.sitegroup)):
                    opt.sitegroup[i] = 0
            if bool(self.cfg_vision.get("hide_tendons_in_vision", True)):
                for attr in ("tendongroup", "jointgroup", "actuatorgroup"):
                    groups = getattr(opt, attr, None)
                    if groups is not None:
                        for i in range(len(groups)):
                            groups[i] = 0
            self._vision_scene_option = opt
        return self._vision_scene_option

    def _update_vision_renderer_scene(self, renderer, camera_name):
        scene_option = self._vision_render_scene_option()
        camera_refs = [camera_name]
        camera_id = self._vision_camera_id(camera_name)
        if camera_id >= 0:
            camera_refs.append(camera_id)

        last_exc = None
        for camera_ref in camera_refs:
            try:
                renderer.update_scene(
                    self.data, camera=camera_ref, scene_option=scene_option)
                return
            except TypeError:
                try:
                    renderer.update_scene(self.data, camera=camera_ref)
                    return
                except Exception as exc:
                    last_exc = exc
            except Exception as exc:
                last_exc = exc
        if last_exc is not None:
            raise last_exc

    def _render_rgbd_camera(self, cam_cfg):
        name = str(cam_cfg.get("name", ""))
        if self._vision_camera_id(name) < 0:
            return None
        cache_enabled = bool(self.cfg_vision.get(
            "reuse_rendered_rgbd_frames", True))
        cache_key = None
        if cache_enabled:
            cache_key = (name, round(float(getattr(self.data, "time", 0.0)), 9))
            cached = getattr(self, "_vision_frame_cache", {}).get(cache_key)
            if cached is not None:
                return cached
        renderer = self._ensure_vision_renderer()
        try:
            if hasattr(renderer, "disable_depth_rendering"):
                renderer.disable_depth_rendering()
            self._update_vision_renderer_scene(renderer, name)
            rgb = renderer.render().copy()
            depth = None
            if hasattr(renderer, "enable_depth_rendering"):
                renderer.enable_depth_rendering()
                self._update_vision_renderer_scene(renderer, name)
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
            rendered = (rgb, depth)
            if cache_enabled and cache_key is not None:
                cache = getattr(self, "_vision_frame_cache", {})
                if len(cache) > 16:
                    cache.clear()
                cache[cache_key] = rendered
                self._vision_frame_cache = cache
            return rendered
        except Exception as exc:
            print(f"[vision] RGB-D render failed for camera {name}: {exc}")
            return None

    def _dump_vision_debug_frame(self, frame, estimate):
        dump_dir = str(self.cfg_vision.get("debug_dump_dir", "") or "").strip()
        limit = int(self.cfg_vision.get("debug_dump_frames", 0))
        if not dump_dir or limit <= 0:
            return
        if int(getattr(self, "_vision_debug_dump_count", 0)) >= limit:
            return
        try:
            import cv2
            os.makedirs(dump_dir, exist_ok=True)
            step = int(getattr(self, "current_step", 0))
            start_step = int(self.cfg_vision.get("debug_dump_start_step", 0))
            end_step = int(self.cfg_vision.get("debug_dump_end_step", -1))
            if step < start_step or (end_step >= 0 and step > end_step):
                return
            idx = int(getattr(self, "_vision_debug_dump_count", 0))
            stem = f"{idx:04d}_step{step:04d}_{frame.name}_{'ok' if estimate is not None else 'fail'}"
            rgb = np.asarray(frame.rgb)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb
            cv2.imwrite(os.path.join(dump_dir, stem + "_rgb.png"), bgr)

            overlay = bgr.copy()
            try:
                estimator = self._ensure_vision_estimator()
                corners, ids = estimator._detect_markers(rgb)
                if len(corners) > 0:
                    cv2.aruco.drawDetectedMarkers(overlay, corners, ids.reshape(-1, 1))
            except Exception:
                pass
            status = "ok" if estimate is not None else "fail"
            cv2.putText(
                overlay,
                f"step={step} cam={frame.name} status={status}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(os.path.join(dump_dir, stem + "_detect.png"), overlay)

            if frame.depth is not None:
                depth = np.asarray(frame.depth, dtype=np.float32)
                finite = depth[np.isfinite(depth) & (depth > 0)]
                if finite.size > 0:
                    lo, hi = np.percentile(finite, [1, 99])
                    denom = max(float(hi - lo), 1e-6)
                    depth_u8 = np.clip((depth - lo) / denom * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(dump_dir, stem + "_depth.png"), depth_u8)
            self._vision_debug_dump_count = idx + 1
        except Exception as exc:
            if int(getattr(self, "_vision_debug_dump_count", 0)) == 0:
                print(f"[vision] debug frame dump failed: {exc}")

    def _estimate_payload_pose_from_rgbd(self):
        t_total0 = time.perf_counter()
        render_time_s = 0.0
        estimator = self._ensure_vision_estimator()
        estimates = []
        statuses = []
        for cam_cfg in list(self.cfg_vision.get("cameras", [])):
            cam_name = str(cam_cfg.get("name", "rgbd"))
            t_render0 = time.perf_counter()
            rendered = self._render_rgbd_camera(cam_cfg)
            render_time_s += time.perf_counter() - t_render0
            if rendered is None:
                statuses.append({
                    "name": cam_name,
                    "valid": False,
                    "reason": "render_failed",
                    "marker_ids": [],
                })
                continue
            rgb, depth = rendered
            frame = CameraFrame(
                name=cam_name,
                rgb=rgb,
                depth=depth,
                K=self._vision_camera_matrix(cam_cfg),
                dist=self._vision_camera_distortion(cam_cfg),
                T_lab_cam=self._vision_camera_lab_transform(cam_cfg),
            )
            est = estimator.estimate_frame(frame)
            self._dump_vision_debug_frame(frame, est)
            if est is not None:
                statuses.append({
                    "name": cam_name,
                    "valid": True,
                    "reason": "",
                    "marker_ids": list(est.marker_ids),
                    "reprojection_error": float(est.reprojection_error),
                    "depth_rmse": float(est.depth_rmse),
                    "depth_support": int(est.depth_support),
                })
                estimates.append(est)
            else:
                marker_ids = []
                reason = "opencv_no_marker_pose"
                try:
                    _corners, ids = estimator._detect_markers(rgb)
                    marker_ids = [int(v) for v in np.asarray(ids).reshape(-1)]
                    if marker_ids:
                        reason = "pose_rejected"
                except Exception as exc:
                    reason = f"detect_error:{exc}"
                statuses.append({
                    "name": cam_name,
                    "valid": False,
                    "reason": reason,
                    "marker_ids": marker_ids,
                })
        self._vision_last_camera_statuses = copy.deepcopy(statuses)
        fused = estimator.fuse(estimates)
        total_s = time.perf_counter() - t_total0
        timing = {
            "total_ms": float(1000.0 * total_s),
            "render_ms": float(1000.0 * render_time_s),
            "algo_ms": float(1000.0 * max(0.0, total_s - render_time_s)),
        }
        self._vision_last_timing = timing
        if fused is None:
            return None
        fused["camera_statuses"] = copy.deepcopy(statuses)
        fused.update(timing)
        return fused

    def _nominal_vision_measurement(self, reason="no_detection"):
        if self._vision_last_generated is not None:
            held = copy.deepcopy(self._vision_last_generated)
            held["valid"] = False
            held["age_steps"] = int(held.get("age_steps", 0)) + 1
            held["failure_reason"] = reason
            held["camera_statuses"] = copy.deepcopy(getattr(
                self, "_vision_last_camera_statuses", []))
            held.update(getattr(self, "_vision_last_timing", {}) or {})
            return held
        start_xy = np.asarray(getattr(self, "episode_start_xy",
                                      self.default_start_xy), dtype=np.float64)
        z0 = float(self.cfg_vision.get(
            "initial_payload_z",
            self.config.get("planning", {}).get("payload_z_cruise", 0.25)))
        return {
            "pos": np.array([start_xy[0], start_xy[1], z0], dtype=np.float64),
            "vel": np.zeros(3, dtype=np.float64),
            "tilt": 0.0,
            "yaw": 0.0,
            "valid": False,
            "active_cameras": 0,
            "active_camera_names": [],
            "marker_ids": [],
            "age_steps": 0,
            "source_step": int(getattr(self, "current_step", 0)),
            "source": self._vision_source(),
            "failure_reason": reason,
            "camera_statuses": copy.deepcopy(getattr(
                self, "_vision_last_camera_statuses", [])),
            "total_ms": float(getattr(
                self, "_vision_last_timing", {}).get("total_ms", 0.0)),
            "render_ms": float(getattr(
                self, "_vision_last_timing", {}).get("render_ms", 0.0)),
            "algo_ms": float(getattr(
                self, "_vision_last_timing", {}).get("algo_ms", 0.0)),
        }

    def _opencv_pose_to_measurement(self, fused, period):
        pos = np.asarray(fused["pos"], dtype=np.float64).reshape(3)
        R_lab_payload = np.asarray(fused["R"], dtype=np.float64).reshape(3, 3)
        euler = R.from_matrix(R_lab_payload).as_euler("xyz")
        tilt = float(np.sqrt(euler[0] ** 2 + euler[1] ** 2))
        yaw = float(euler[2])
        if self._vision_last_pos is not None:
            raw_vel = (pos - self._vision_last_pos) / max(self.dt * period, 1e-6)
        else:
            raw_vel = np.zeros(3, dtype=np.float64)
        alpha = float(np.clip(self.cfg_vision.get(
            "velocity_lowpass_alpha", 0.55), 0.0, 1.0))
        if self._vision_last_vel is None:
            vel = raw_vel
        else:
            vel = alpha * raw_vel + (1.0 - alpha) * self._vision_last_vel
        self._vision_last_pos = pos.copy()
        self._vision_last_vel = vel.copy()
        meas = {
            "pos": pos,
            "vel": vel,
            "tilt": tilt,
            "yaw": yaw,
            "valid": True,
            "active_cameras": int(fused.get("active_cameras", 0)),
            "active_camera_names": list(fused.get("active_camera_names", [])),
            "marker_ids": list(fused.get("marker_ids", [])),
            "reprojection_error": float(fused.get("reprojection_error", 0.0)),
            "depth_rmse": float(fused.get("depth_rmse", 0.0)),
            "depth_support": int(fused.get("depth_support", 0)),
            "age_steps": 0,
            "source_step": int(getattr(self, "current_step", 0)),
            "source": self._vision_source(),
            "failure_reason": "",
            "camera_statuses": copy.deepcopy(fused.get(
                "camera_statuses",
                getattr(self, "_vision_last_camera_statuses", []))),
            "total_ms": float(fused.get("total_ms", 0.0)),
            "render_ms": float(fused.get("render_ms", 0.0)),
            "algo_ms": float(fused.get("algo_ms", 0.0)),
        }
        self._vision_last_fused_pose = fused
        return meas

    def reset_vision_state(self):
        """Reset per-episode RGB-D biases and fill the latency queue."""
        if not self._vision_enabled():
            self._vision_queue.clear()
            self._vision_current = None
            self._vision_last_generated = None
            self._vision_last_pos = None
            self._vision_last_vel = None
            self._vision_last_update_step = None
            self._vision_last_fused_pose = None
            self._vision_last_camera_statuses = []
            self._vision_debug_dump_count = 0
            self._vision_last_timing = {}
            return

        cfg = self.cfg_vision
        self._vision_episode_bias = {
            "pos": self.vision_rng.normal(
                0.0, self._as_vec3(cfg.get("position_bias_std"), [0, 0, 0])),
            "vel": self.vision_rng.normal(
                0.0, self._as_vec3(cfg.get("velocity_bias_std"), [0, 0, 0])),
            "tilt": float(self.vision_rng.normal(
                0.0, float(cfg.get("tilt_bias_std", 0.0)))),
            "yaw": float(self.vision_rng.normal(
                0.0, float(cfg.get("yaw_bias_std", 0.0)))),
        }
        delay = self._vision_total_delay_steps()
        self._vision_queue = deque(maxlen=delay + 1)
        self._vision_current = None
        self._vision_last_generated = None
        self._vision_last_pos = None
        self._vision_last_vel = None
        self._vision_last_update_step = None
        self._vision_last_fused_pose = None
        self._vision_last_camera_statuses = []
        self._vision_debug_dump_count = 0
        self._vision_last_timing = {}

        first = self._generate_vision_measurement(force=True)
        for _ in range(delay):
            self._vision_queue.append(copy.deepcopy(first))
        self._vision_current = copy.deepcopy(first)

    def _payload_state_for_vision(self):
        pl_pos = self.data.body('prefab').xpos.copy().astype(np.float64)
        dof_idx = self.model.jnt_dofadr[self.prefab_jnt_id]
        pl_vel = self.data.qvel[dof_idx:dof_idx + 3].copy().astype(np.float64)
        pl_mat = self.data.body('prefab').xmat.reshape(3, 3)
        pl_euler = R.from_matrix(pl_mat).as_euler('xyz')
        tilt = float(np.sqrt(pl_euler[0] ** 2 + pl_euler[1] ** 2))
        yaw = float(pl_euler[2])
        return {"pos": pl_pos, "vel": pl_vel, "tilt": tilt, "yaw": yaw}

    def _camera_occlusion_scale(self, cam_pos, payload_pos):
        cfg = self.cfg_vision
        penalty = float(cfg.get("occlusion_penalty", 1.0))
        if penalty >= 0.999 or not getattr(self, "_obstacles", None):
            return 1.0
        a = np.asarray(cam_pos[:2], dtype=np.float64)
        b = np.asarray(payload_pos[:2], dtype=np.float64)
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-9:
            return 1.0
        obs_z = float(self.config.get("scene", {}).get("obstacle_z_center", 0.15))
        obs_hh = float(self.config.get("scene", {}).get("obstacle_halfheight", 0.15))
        if float(payload_pos[2]) > obs_z + obs_hh + 0.05:
            return 1.0
        for ox, oy, radius in self._obstacles:
            c = np.array([ox, oy], dtype=np.float64)
            t = float(np.clip(np.dot(c - a, ab) / denom, 0.0, 1.0))
            closest = a + t * ab
            if float(np.linalg.norm(c - closest)) < float(radius) + 0.02:
                return max(0.05, penalty)
        return 1.0

    def _generate_vision_measurement(self, force=False):
        cfg = self.cfg_vision
        period = max(1, int(cfg.get("measurement_period_steps", 1)))
        step = int(getattr(self, "current_step", 0))
        if (not force and self._vision_last_generated is not None and
                step % period != 0):
            held = copy.deepcopy(self._vision_last_generated)
            held["age_steps"] = int(held.get("age_steps", 0)) + 1
            return held

        source = self._vision_source()
        if source == "truth_noise":
            if not bool(cfg.get("allow_truth_fallback", False)):
                raise RuntimeError(
                    "vision.source='truth_noise' requires "
                    "vision.allow_truth_fallback=True because it uses MuJoCo "
                    "payload ground truth.")
            return self._generate_truth_noise_vision_measurement(force=force)
        if source != "opencv_rgbd":
            raise ValueError(f"Unknown vision.source: {source}")

        fused = self._estimate_payload_pose_from_rgbd()
        if fused is None:
            meas = self._nominal_vision_measurement("opencv_no_marker_pose")
        else:
            meas = self._opencv_pose_to_measurement(fused, period)
        self._vision_last_generated = copy.deepcopy(meas)
        return meas

    def _generate_truth_noise_vision_measurement(self, force=False):
        cfg = self.cfg_vision
        true = self._payload_state_for_vision()
        period = max(1, int(cfg.get("measurement_period_steps", 1)))
        step = int(getattr(self, "current_step", 0))
        if (not force and self._vision_last_generated is not None and
                step % period != 0):
            held = copy.deepcopy(self._vision_last_generated)
            held["age_steps"] = int(held.get("age_steps", 0)) + 1
            return held

        pos_sigma = self._as_vec3(cfg.get("position_noise_std"),
                                  [0.003, 0.003, 0.0045])
        vel_sigma = self._as_vec3(cfg.get("velocity_noise_std"),
                                  [0.010, 0.010, 0.014])
        depth_per_m = float(cfg.get("depth_noise_per_m", 0.0))
        dropout_prob = float(cfg.get("dropout_prob", 0.0))
        min_active = max(1, int(cfg.get("min_active_cameras", 1)))
        fused = []
        weights = []
        active_names = []

        for cam in list(cfg.get("cameras", [])):
            if self.vision_rng.random() < dropout_prob:
                continue
            cam_pos = self._as_vec3(cam.get("pos"), [0, 0, 1])
            rel = true["pos"] - cam_pos
            dist = max(float(np.linalg.norm(rel)), 1e-6)
            if dist > float(cam.get("max_range", 10.0)):
                continue
            quality = 1.0 / (1.0 + dist * dist)
            quality *= self._camera_occlusion_scale(cam_pos, true["pos"])
            if quality <= 0.0:
                continue
            sigma = pos_sigma.copy()
            sigma[2] += depth_per_m * dist
            sigma = sigma / max(np.sqrt(quality), 1e-3)
            cam_bias = self._as_vec3(cam.get("bias"), [0, 0, 0])
            meas = (true["pos"] + self._vision_episode_bias.get("pos", 0.0) +
                    cam_bias + self.vision_rng.normal(0.0, sigma))
            fused.append(meas)
            weights.append(max(quality, 1e-3))
            active_names.append(str(cam.get("name", "rgbd")))

        valid = len(fused) >= min_active
        if fused:
            pos_meas = np.average(
                np.asarray(fused), axis=0,
                weights=np.asarray(weights, dtype=np.float64))
        elif self._vision_last_generated is not None:
            pos_meas = self._vision_last_generated["pos"].copy()
        else:
            pos_meas = true["pos"].copy()

        if self.vision_rng.random() < float(cfg.get("outlier_prob", 0.0)):
            pos_meas = pos_meas + self.vision_rng.normal(
                0.0, float(cfg.get("outlier_pos_std", 0.03)), size=3)
            valid = False

        if self._vision_last_pos is not None:
            raw_vel = (pos_meas - self._vision_last_pos) / max(self.dt * period, 1e-6)
        else:
            raw_vel = true["vel"].copy()
        raw_vel = raw_vel + self._vision_episode_bias.get("vel", 0.0) + \
            self.vision_rng.normal(0.0, vel_sigma)
        alpha = float(np.clip(cfg.get("velocity_lowpass_alpha", 0.55), 0.0, 1.0))
        if self._vision_last_vel is None:
            vel_meas = raw_vel
        else:
            vel_meas = alpha * raw_vel + (1.0 - alpha) * self._vision_last_vel

        tilt_meas = true["tilt"] + self._vision_episode_bias.get("tilt", 0.0) + \
            float(self.vision_rng.normal(0.0, float(cfg.get("tilt_noise_std", 0.01))))
        yaw_meas = true["yaw"] + self._vision_episode_bias.get("yaw", 0.0) + \
            float(self.vision_rng.normal(0.0, float(cfg.get("yaw_noise_std", 0.012))))

        self._vision_last_pos = pos_meas.copy()
        self._vision_last_vel = vel_meas.copy()
        meas = {
            "pos": pos_meas.astype(np.float64),
            "vel": vel_meas.astype(np.float64),
            "tilt": float(tilt_meas),
            "yaw": float(yaw_meas),
            "valid": bool(valid),
            "active_cameras": int(len(fused)),
            "active_camera_names": active_names,
            "age_steps": 0,
            "source_step": step,
        }
        self._vision_last_generated = copy.deepcopy(meas)
        return meas

    def _current_vision_measurement(self):
        if not self._vision_enabled():
            return None
        step = int(getattr(self, "current_step", 0))
        if self._vision_last_update_step == step and self._vision_current is not None:
            return self._vision_current
        if self._vision_last_generated is None and not self._vision_queue:
            self.reset_vision_state()
        generated = self._generate_vision_measurement()
        self._vision_queue.append(copy.deepcopy(generated))
        delay = self._vision_total_delay_steps()
        if delay <= 0:
            self._vision_current = copy.deepcopy(generated)
        elif len(self._vision_queue) > delay:
            self._vision_current = copy.deepcopy(self._vision_queue.popleft())
        elif self._vision_queue:
            self._vision_current = copy.deepcopy(self._vision_queue[0])
        else:
            self._vision_current = copy.deepcopy(generated)
        self._vision_last_update_step = step
        return self._vision_current

    def _apply_vision_to_obs(self, obs):
        meas = self._current_vision_measurement()
        if meas is None:
            return obs
        out = np.asarray(obs, dtype=np.float32).copy()
        pos = np.asarray(meas["pos"], dtype=np.float32)
        vel = np.asarray(meas["vel"], dtype=np.float32)
        out[4], out[5] = pos[0], pos[1]
        out[6], out[7] = vel[0], vel[1]
        out[8] = float(self.target_pos[0] - pos[0])
        out[9] = float(self.target_pos[1] - pos[1])
        out[21], out[22] = pos[2], vel[2]
        out[29] = float(meas["tilt"])
        out[30] = float(meas["yaw"])
        if out.size > 49:
            target_pz = float(self.cfg_insertion.get("target_payload_z", 0.10))
            out[49] = float(pos[2] - target_pz)
        if out.size >= 54:
            entry_z = float(self.cfg_insertion.get("entry_z", 0.16))
            out[45:48] = 0.0
            if pos[2] <= entry_z or getattr(self, '_in_insertion_phase', False):
                out[47] = 1.0
            elif self.reached_final:
                out[46] = 1.0
            else:
                out[45] = 1.0
            rebar_errors = np.zeros(4, dtype=np.float32)
            if float(np.linalg.norm(pos[:2] - self.target_pos)) < 0.05:
                local = np.array([
                    [0.035, 0.035], [0.035, -0.035],
                    [-0.035, 0.035], [-0.035, -0.035]], dtype=np.float32)
                cy = float(np.cos(meas["yaw"]))
                sy = float(np.sin(meas["yaw"]))
                R2 = np.array([[cy, -sy], [sy, cy]], dtype=np.float32)
                for i in range(4):
                    hole_w = pos[:2] + R2 @ local[i]
                    rebar_w = self.target_pos.astype(np.float32) + local[i]
                    rebar_errors[i] = float(np.linalg.norm(hole_w - rebar_w))
            out[50:54] = rebar_errors
        return out

    def get_vision_debug(self):
        meas = getattr(self, "_vision_current", None)
        if not meas:
            return {}
        return {
            "valid": bool(meas.get("valid", False)),
            "active_cameras": int(meas.get("active_cameras", 0)),
            "active_camera_names": list(meas.get("active_camera_names", [])),
            "marker_ids": list(meas.get("marker_ids", [])),
            "age_steps": int(meas.get("age_steps", 0)) +
                         self._vision_total_delay_steps(),
            "source_step": int(meas.get("source_step", -1)),
            "source": str(meas.get("source", self._vision_source())),
            "reprojection_error": float(meas.get("reprojection_error", 0.0)),
            "depth_rmse": float(meas.get("depth_rmse", 0.0)),
            "depth_support": int(meas.get("depth_support", 0)),
            "failure_reason": str(meas.get("failure_reason", "")),
            "camera_statuses": copy.deepcopy(meas.get("camera_statuses", [])),
            "total_ms": float(meas.get("total_ms", 0.0)),
            "render_ms": float(meas.get("render_ms", 0.0)),
            "algo_ms": float(meas.get("algo_ms", 0.0)),
        }

    def get_vision_measurement(self):
        meas = self._current_vision_measurement()
        return copy.deepcopy(meas) if meas is not None else None

    @staticmethod
    def _sample_depth_patch(depth, u, v, radius=2):
        if depth is None:
            return None
        arr = np.asarray(depth, dtype=np.float64)
        h, w = arr.shape[:2]
        ui = int(round(float(u)))
        vi = int(round(float(v)))
        if ui < 0 or ui >= w or vi < 0 or vi >= h:
            return None
        x0 = max(0, ui - int(radius))
        x1 = min(w, ui + int(radius) + 1)
        y0 = max(0, vi - int(radius))
        y1 = min(h, vi + int(radius) + 1)
        patch = arr[y0:y1, x0:x1].reshape(-1)
        patch = patch[np.isfinite(patch) & (patch > 1e-6)]
        if patch.size == 0:
            return None
        return float(np.median(patch))

    def get_rope_marker_camera_diagnostics(self, use_depth=True):
        """Project rope markers into each RGB-D camera and summarize coverage."""
        markers = self.get_rope_marker_world_positions()
        width, height = self._vision_resolution()
        cfg_marker = self.config.get("rope_markers", {})
        depth_tol = float(cfg_marker.get(
            "visibility_depth_tolerance", 0.035))
        depth_radius = int(cfg_marker.get("visibility_depth_window", 2))
        total = len(markers)
        union_fov = set()
        union_depth = set()
        cameras = []
        for cam_cfg in list(self.cfg_vision.get("cameras", [])):
            cam_name = str(cam_cfg.get("name", "rgbd"))
            K = self._vision_camera_matrix(cam_cfg)
            T_lab_cam = self._vision_camera_lab_transform(cam_cfg)
            R_lab_cam = T_lab_cam[:3, :3]
            t_lab_cam = T_lab_cam[:3, 3]
            max_range = float(cam_cfg.get("max_range", np.inf))
            depth = None
            if use_depth and self._vision_camera_id(cam_name) >= 0:
                rendered = self._render_rgbd_camera(cam_cfg)
                if rendered is not None:
                    _rgb, depth = rendered
            cam_rows = []
            in_fov_count = 0
            depth_visible_count = 0
            distances = []
            for marker in markers:
                p_lab = np.asarray(marker["pos"], dtype=np.float64).reshape(3)
                p_cam = R_lab_cam.T @ (p_lab - t_lab_cam)
                z = float(p_cam[2])
                dist = float(np.linalg.norm(p_lab - t_lab_cam))
                distances.append(dist)
                in_front = z > 1e-6
                u = np.nan
                v = np.nan
                if in_front:
                    u = float(K[0, 0] * p_cam[0] / z + K[0, 2])
                    v = float(K[1, 1] * p_cam[1] / z + K[1, 2])
                in_range = bool(in_front and z <= max_range)
                in_fov = bool(
                    in_range and 0.0 <= u < width and 0.0 <= v < height)
                depth_value = None
                depth_visible = None
                if in_fov:
                    in_fov_count += 1
                    union_fov.add(marker["name"])
                    if depth is not None:
                        depth_value = self._sample_depth_patch(
                            depth, u, v, radius=depth_radius)
                        if depth_value is not None:
                            depth_visible = bool(abs(depth_value - z) <= depth_tol)
                            if depth_visible:
                                depth_visible_count += 1
                                union_depth.add(marker["name"])
                    elif not use_depth:
                        depth_visible = True
                        depth_visible_count += 1
                        union_depth.add(marker["name"])
                cam_rows.append({
                    "name": marker["name"],
                    "rope": marker["rope"],
                    "marker_index": int(marker["marker_index"]),
                    "pixel": [u, v],
                    "camera_z": z,
                    "distance": dist,
                    "in_fov": in_fov,
                    "depth": depth_value,
                    "depth_visible": depth_visible,
                })
            cameras.append({
                "name": cam_name,
                "markers_total": total,
                "in_fov": in_fov_count,
                "depth_visible": depth_visible_count,
                "distance_min": float(min(distances)) if distances else 0.0,
                "distance_max": float(max(distances)) if distances else 0.0,
                "markers": cam_rows,
            })
        visible_union = union_depth if (use_depth and cameras) else union_fov
        return {
            "markers_total": total,
            "cameras_total": len(cameras),
            "union_in_fov": len(union_fov),
            "union_depth_visible": len(union_depth),
            "union_visible": len(visible_union),
            "coverage_fov": (len(union_fov) / max(total, 1)),
            "coverage_visible": (len(visible_union) / max(total, 1)),
            "use_depth": bool(use_depth),
            "cameras": cameras,
        }

    # ── [v8] 环境噪声力 (force_noise) ────────────────────────────────────────
    def set_force_noise(self, sigma_n: float):
        """
        设置环境噪声力标准差 (N).
        每个物理子步施加 N(0, sigma) 力到 payload, 随机方向.
        与风力可叠加 (wind 是确定性, force_noise 是高斯).
        """
        self._force_noise_sigma = float(max(0.0, sigma_n))

    def _apply_force_noise(self):
        """在 step 子循环中调用, 施加随机噪声力到 payload。"""
        sigma = getattr(self, '_force_noise_sigma', 0.0)
        if sigma <= 0:
            return
        if not (hasattr(self, 'data') and hasattr(self, 'prefab_body_id')):
            return
        # 随机方向 2D 力 (z=0, 模拟空气扰动而非冲击)
        fx, fy = np.random.normal(0.0, sigma, 2)
        # 注意: xfrc_applied 已被 wind/test_wind 占用, 这里叠加而不覆盖
        cur = self.data.xfrc_applied[self.prefab_body_id, :3].copy()
        cur[0] += fx
        cur[1] += fy
        self.data.xfrc_applied[self.prefab_body_id, :3] = cur

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
        if getattr(self, "_vision_renderer", None) is not None:
            try: self._vision_renderer.close()
            except Exception: pass
            self._vision_renderer = None

    def _target_xy_candidate_valid(self, target_xy, start_xy, cfg_task):
        target_xy = np.asarray(target_xy, dtype=np.float64).reshape(2)
        start_xy = np.asarray(start_xy, dtype=np.float64).reshape(2)

        min_start_dist = float(cfg_task.get("target_xy_min_start_dist", 0.0))
        if min_start_dist > 0.0:
            if np.linalg.norm(target_xy - start_xy) < min_start_dist:
                return False

        min_norm = float(cfg_task.get("target_xy_min_norm", 0.0))
        if min_norm > 0.0 and np.linalg.norm(target_xy) < min_norm:
            return False

        margin = float(cfg_task.get("target_xy_workspace_margin", 0.0))
        max_norm = float(cfg_task.get(
            "target_xy_max_norm",
            max(0.0, self.workspace_radius - margin)))
        if max_norm > 0.0 and np.linalg.norm(target_xy) > max_norm:
            return False

        for zone in cfg_task.get("target_xy_dead_zones", []) or []:
            if isinstance(zone, dict):
                center = zone.get("center", [0.0, 0.0])
                radius = zone.get("radius", 0.0)
            else:
                if len(zone) < 3:
                    continue
                center = zone[:2]
                radius = zone[2]
            center = np.asarray(center, dtype=np.float64).reshape(2)
            if np.linalg.norm(target_xy - center) < float(radius):
                return False
        return True

    def _sample_episode_target_xy(self, start_xy, cfg_task):
        default_target = np.asarray(
            cfg_task.get("default_target_xy", self.default_target),
            dtype=np.float64).reshape(2)
        if not bool(cfg_task.get("target_xy_randomize", False)):
            return default_target.copy()

        span = np.asarray(
            cfg_task.get("target_xy_range", [0.0, 0.0]),
            dtype=np.float64).reshape(-1)
        if span.size == 1:
            span = np.repeat(span[0], 2)
        else:
            span = span[:2]
        span = np.maximum(span, 0.0)
        max_tries = max(1, int(cfg_task.get("target_xy_max_tries", 64)))

        last_candidate = default_target.copy()
        for _ in range(max_tries):
            candidate = default_target + self._obstacle_rng.uniform(-span, span)
            last_candidate = candidate
            if self._target_xy_candidate_valid(candidate, start_xy, cfg_task):
                return candidate.astype(np.float64)

        if self._target_xy_candidate_valid(default_target, start_xy, cfg_task):
            return default_target.copy()
        return last_candidate.astype(np.float64)

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

            # [v5] 障碍物间最小净空: payload直径 + 双侧planning_margin
            # 这是中心距 >= r_a + r_b + min_gap 里的 min_gap
            min_gap = p_radius * 2 + p_margin * 2   # ≈ 0.23m (不含障碍物半径)

            # [v5] 分段放置: 路径均分为 n_obs 段，每段放1个
            # 保证障碍物沿路径均匀分布，避免扎堆
            n_placed = 0
            for seg_idx in range(n_obs):
                t_lo = 0.10 + seg_idx * (0.80 / max(n_obs, 1))
                t_hi = t_lo + (0.80 / max(n_obs, 1)) * 0.85
                t_hi = min(t_hi, 0.90)

                placed = False
                for _ in range(300):   # 每段最多尝试300次
                    t = rng.uniform(t_lo, t_hi)
                    # [v5] 偏向走廊两侧: 为payload保留中央通道
                    # 最小侧偏 = payload_radius + planning_margin，确保中央可通行
                    s_min = p_radius + p_margin + 0.01   # ≈ 0.12m
                    s_max = path_width / 2
                    if s_min >= s_max:
                        s_min = s_max * 0.3
                    s_side = rng.choice([-1, 1]) * rng.uniform(s_min, s_max)
                    center = start_xy + t * L_path * direction + s_side * perp
                    r = rng.uniform(r_min, r_max)

                    # 工作空间约束
                    if np.linalg.norm(center) + r > workspace_r - 0.02:
                        continue
                    # y走廊约束
                    if center[1] - r < y_min_corridor:
                        continue
                    # 起终点 & 底座安全距离
                    if (np.linalg.norm(center - start_xy) < r + min_clr or
                            np.linalg.norm(center - target_xy) < r + min_clr or
                            np.linalg.norm(center) < r + base_r2):
                        continue
                    # 与已放置障碍物的净空: 两圆心距 >= r_a + r_b + min_gap
                    if not all(np.linalg.norm(center - np.array([ox, oy])) >= r + or_ + min_gap
                               for (ox, oy, or_) in obstacles):
                        continue

                    obstacles.append((float(center[0]), float(center[1]), float(r)))
                    placed = True
                    break

                if not placed:
                    # 该段放置失败，不强制，继续尝试下一段（保证A*能规划通）
                    pass

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
        for _ in range(int(scene_plan_config.get("num_cruise_target_hover_steps", 0))):
            path_3d.append([float(target_xy[0]),float(target_xy[1]),float(z_cruise)])
        if not bool(scene_plan_config.get("disable_descent_segment", False)):
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
            self.episode_start_xy = start_xy.copy()
            target_xy = self._sample_episode_target_xy(start_xy, cfg_task)
            self.target_pos = target_xy.copy()

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
        # [v12.3] scene 切换后重新缓存绳索 body id
        self._cache_cable_body_ids()
        self._cache_rope_marker_site_ids()
        self._reset_vision_runtime_after_model_change()

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

        # [TEST-WIND] reset 时清除测试风力（不影响训练）
        # 若测试需要保留风速扰动，应在 reset 后重新调用 set_wind_speed
        if not getattr(self, '_test_wind_mode', False):
            self.data.xfrc_applied[self.prefab_body_id, :3] = [0.0, 0.0, 0.0]

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
        speed_max = self.cfg_wind.get("speed_max", 16.5)
        self.wind_speed = 0.2 * speed_max
        self.wind_F = self._wind_speed_to_force(self.wind_speed)

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

        self.reset_vision_state()
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
            self._apply_wind_load()
            self._apply_test_wind()   # 测试风力：覆盖随机游走，施加恒定风
            self._apply_force_noise() # [v8] 环境噪声力
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

        rope_marker_features = None
        if bool(self.config.get("rope_markers", {}).get("enabled", False)):
            rope_marker_features = self.get_rope_marker_feature_vector()
        return obs, reward, done, False, {
            "is_success": success, "current_wp_idx": self.current_wp_idx,
            "reached_final": self.reached_final, "is_collision": is_collision,
            "termination_reason": getattr(self, '_termination_reason', None),
            "vision": self.get_vision_debug(),
            "rope_marker_features": rope_marker_features,
            "rope_marker_feature_debug": self.get_rope_marker_feature_debug(),
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

    def _prefab_floor_clearance(self):
        """Return the lowest prefab geometry clearance above the z=0 floor."""
        if not hasattr(self, '_prefab_geom_ids'):
            self._reresolve_ids()

        min_bottom_z = np.inf
        for gid in self._prefab_geom_ids:
            gtype = int(self.model.geom_type[gid])
            size = self.model.geom_size[gid]
            xpos = self.data.geom_xpos[gid]
            xmat = self.data.geom_xmat[gid].reshape(3, 3)

            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                support_z = float(np.dot(np.abs(xmat[2, :]), size[:3]))
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                support_z = float(size[0])
            elif gtype in (
                    int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                radius = float(size[0])
                half_len = float(size[1])
                support_z = (abs(float(xmat[2, 2])) * half_len +
                             radius * float(np.linalg.norm(xmat[2, :2])))
            else:
                support_z = float(np.max(size))

            min_bottom_z = min(min_bottom_z, float(xpos[2]) - support_z)

        if not np.isfinite(min_bottom_z):
            return np.inf
        return float(min_bottom_z)

    def _check_prefab_floor_contact(self, allow_z_fallback=None):
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

        cfg_ins = self.config.get("insertion", {})
        if allow_z_fallback is None:
            allow_z_fallback = bool(
                cfg_ins.get("floor_contact_allow_z_fallback", True))
        if bool(allow_z_fallback):
            tol = float(cfg_ins.get("floor_contact_z_tolerance", 0.004))
            try:
                return self._prefab_floor_clearance() <= tol
            except Exception:
                return False
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
        target_pz = cfg_ins.get("target_payload_z", 0.10)
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
        floor_contact_success = (
            bool(cfg_ins.get("success_by_floor_contact", False)) and
            on_target and
            self._check_prefab_floor_contact())

        if on_target:
            self._insertion_hold_counter += 1
        else:
            self._insertion_hold_counter = 0

        if floor_contact_success or self._insertion_hold_counter >= hold_steps:
            reward += cfg_rwd.get("success_bonus", 50.0)
            success = True; done = True
            suffix = ",floor_contact=1" if floor_contact_success else ""
            self._termination_reason = (
                f"success:z={payload_z*1000:.0f}mm,dtf={dtf*1000:.1f}mm,"
                f"tilt={tilt:.3f},yaw={abs_yaw:.3f}{suffix}")
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

    def _get_obs_raw(self):
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

        # [v12.3 修订] 绳索每段运动状态 (240 维): 4 根 × 10 段 × 6 维 (rel_pos + lin_vel)
        # 用户原话: "把四根绳索的每个链接点的运动状态(相对位移、速度、加速度)加入观测层"
        # 文献: Kotaru 2017 (arXiv:1711.04895), Goodarzi 2014 (arXiv:1407.8164), FLARE 2025
        # 加速度: 由 LSTM actor 通过相邻帧 vel 差分隐式推断 (避免维度爆炸)
        try:
            cable_obs = list(self.get_cable_segment_states())
        except Exception:
            cable_obs = [0.0] * (4 * 10 * 6)   # = 240

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
            +rebar_errors
            +cable_obs,    # [v12.3] 240 维: 4 根绳 × 10 段 × (rel_pos+lin_vel)
            dtype=np.float32)

    def _get_obs(self):
        obs = self._get_obs_raw()
        if (self._vision_enabled() and
                bool(self.cfg_vision.get("apply_to_env_obs", True))):
            return self._apply_vision_to_obs(obs)
        return obs


def make_env(config=None):
    return CableRobotEnvWithObstacles(config=config)
