#!/usr/bin/env python3
"""Capture paper-figure-ready snapshots from the MuJoCo simulation.

The main entry point for later code is ``capture_snapshot(env, out_dir, ...)``.
It uses the environment's existing RGB-D camera configuration and renderer, so
figure assets stay aligned with the actual vision pipeline.

Examples:
    python scripts/capture_simulation_snapshots.py --action-mode policy \
        --steps 0,20,60 \
        --out-dir test_results/figure_assets/sim_snapshots_clean

    from scripts.capture_simulation_snapshots import capture_snapshot
    capture_snapshot(env, "test_results/figure_assets/sim_snapshots",
                     label="handoff", cameras=["rgbd_front"])
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from vision_rgbd import CameraFrame


DEFAULT_MAINLINE_CKPT = (
    REPO_ROOT
    / "saves"
    / "descent_ablation_fullobs_mlp_trainable8_seed270627_20260705"
    / "ckpt_best.pt"
)

GENERATED_XML_PATHS = (
    REPO_ROOT / "assets" / "demo_fourCable_withSteel_withSensor_cylinder.xml",
    REPO_ROOT / "assets" / "iiwa14_four_cables_with_plate.xml",
)


def _deep_update(dst, src):
    out = copy.deepcopy(dst)
    for key, val in (src or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "snap"


def _parse_steps(text):
    steps = set()
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            fields = [int(x) for x in part.split(":")]
            if len(fields) == 2:
                start, stop = fields
                stride = 1
            elif len(fields) == 3:
                start, stop, stride = fields
            else:
                raise ValueError(f"Bad step range: {part}")
            if stride <= 0:
                raise ValueError(f"Step stride must be positive: {part}")
            steps.update(range(start, stop + 1, stride))
        else:
            steps.add(int(part))
    return {s for s in steps if s >= 0}


def _load_resolved_config(config_json=None, ckpt_path=None):
    """Load the training-time resolved config when available."""
    candidates = []
    if config_json:
        candidates.append(Path(config_json))
    if ckpt_path:
        candidates.append(Path(ckpt_path).resolve().parent / "resolved_config.json")
    for path in candidates:
        if not path:
            continue
        path = Path(path)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = data.get("config", data)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config JSON has no config object: {path}")
        return copy.deepcopy(cfg), path
    return copy.deepcopy(DEFAULT_CONFIG), None


def _snapshot_generated_xmls():
    snapshot = {}
    for path in GENERATED_XML_PATHS:
        path = Path(path)
        snapshot[path] = path.read_bytes() if path.exists() else None
    return snapshot


def _restore_generated_xmls(snapshot):
    for path, data in (snapshot or {}).items():
        path = Path(path)
        if data is None:
            if path.exists():
                path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def _camera_cfgs(env, cameras=None):
    wanted = None
    if cameras:
        wanted = {str(c).strip() for c in cameras if str(c).strip()}
    out = []
    for cam_cfg in list(env.cfg_vision.get("cameras", [])):
        name = str(cam_cfg.get("name", ""))
        if wanted is None or name in wanted:
            out.append(cam_cfg)
    return out


def _write_rgb(path, rgb):
    import cv2

    arr = np.asarray(rgb)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), arr)


def _depth_to_u8(depth):
    arr = np.asarray(depth, dtype=np.float32)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    denom = max(float(hi - lo), 1e-6)
    return np.clip((arr - lo) / denom * 255.0, 0, 255).astype(np.uint8)


def _write_depth(path, depth, colorize=False):
    import cv2

    depth_u8 = _depth_to_u8(depth)
    if colorize:
        depth_u8 = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(path), depth_u8)


def _hide_clean_render_debug_geoms(env):
    """Hide visual-only planning/debug geoms for clean figure snapshots.

    This mutates only the in-memory MuJoCo model used by this capture process.
    It does not edit the generated XML or the main training/evaluation config.
    """
    try:
        import mujoco
    except Exception:
        return []

    prefixes = (
        "path_pt_",
        "path_start",
        "path_goal",
        "rgbd_front_",
        "rgbd_left_",
        "rgbd_right_",
    )
    hidden = []
    model = getattr(env, "model", None)
    if model is None:
        return hidden
    for gid in range(int(getattr(model, "ngeom", 0))):
        try:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        except Exception:
            name = ""
        if (name.startswith(prefixes) or
                name.startswith("vision_marker_") or
                "_marker_" in name):
            try:
                model.geom_rgba[gid, 3] = 0.0
                hidden.append(name)
            except Exception:
                pass
    return hidden


def _show_only_connection_sites(env):
    """Keep only payload lift-site dots visible in offscreen snapshots."""
    try:
        import mujoco
    except Exception:
        return []

    model = getattr(env, "model", None)
    if model is None:
        return []
    keep = {"lift_fl", "lift_fr", "lift_rl", "lift_rr"}
    hidden = []
    for sid in range(int(getattr(model, "nsite", 0))):
        try:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        except Exception:
            name = ""
        try:
            if name in keep:
                model.site_rgba[sid, 3] = 1.0
            else:
                model.site_rgba[sid, 3] = 0.0
                hidden.append(name)
        except Exception:
            pass
    return hidden


def prepare_clean_snapshot_visuals(env, show_connection_sites=False):
    """Configure an existing env for clean offscreen figure snapshots.

    Use this before ``capture_snapshot`` when the env was created from a normal
    XML that still contains tag/rope-marker/debug geoms.
    """
    cfg_vision = getattr(env, "cfg_vision", None)
    if isinstance(cfg_vision, dict):
        cfg_vision["render_markers"] = False
        cfg_vision["show_camera_models"] = False
        cfg_vision["hide_sites_in_vision"] = not bool(show_connection_sites)
        cfg_vision["hide_tendons_in_vision"] = True
    config = getattr(env, "config", None)
    if isinstance(config, dict):
        config.setdefault("vision", {}).update({
            "render_markers": False,
            "show_camera_models": False,
            "hide_sites_in_vision": not bool(show_connection_sites),
            "hide_tendons_in_vision": True,
        })
        config.setdefault("rope_markers", {})["enabled"] = False
    try:
        env._vision_scene_option = None
    except Exception:
        pass
    if show_connection_sites:
        _show_only_connection_sites(env)
    return _hide_clean_render_debug_geoms(env)


def _estimate_and_overlay(env, cam_cfg, rgb, depth, label_text):
    import cv2

    bgr = cv2.cvtColor(np.asarray(rgb)[:, :, :3], cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    info = {
        "detector_ok": False,
        "estimate_ok": False,
        "marker_ids": [],
    }
    try:
        estimator = env._ensure_vision_estimator()
        corners, ids = estimator._detect_markers(rgb)
        if ids is not None and len(ids) > 0:
            ids_arr = np.asarray(ids).reshape(-1, 1)
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids_arr)
            info["detector_ok"] = True
            info["marker_ids"] = [int(x) for x in ids_arr.reshape(-1)]

        frame = CameraFrame(
            name=str(cam_cfg.get("name", "rgbd")),
            rgb=rgb,
            depth=depth,
            K=env._vision_camera_matrix(cam_cfg),
            dist=env._vision_camera_distortion(cam_cfg),
            T_lab_cam=env._vision_camera_lab_transform(cam_cfg),
        )
        estimate = estimator.estimate_frame(frame)
        if estimate is not None:
            info.update({
                "estimate_ok": True,
                "reprojection_error": float(estimate.reprojection_error),
                "depth_rmse": float(estimate.depth_rmse),
                "depth_support": int(estimate.depth_support),
                "position": np.asarray(estimate.position).tolist(),
            })
    except Exception as exc:
        info["error"] = str(exc)

    cv2.putText(
        overlay,
        label_text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay, info


def capture_snapshot(env, out_dir, label=None, cameras=None, prefix="sim",
                     include_depth=True, include_detection=False,
                     colorize_depth=False, include_metadata=True,
                     extra_metadata=None):
    """Save RGB-D camera snapshots from the current simulation state.

    Args:
        env: A ``CableRobotEnvWithObstacles`` instance.
        out_dir: Directory where image files are written.
        label: Human-readable label used in filenames. Defaults to current step.
        cameras: Optional iterable of camera names. Defaults to all RGB-D cams.
        prefix: Filename prefix.
        include_depth: Save normalized depth PNGs when depth is available.
        include_detection: Save AprilTag detection overlay PNGs.
        colorize_depth: Save depth with a colormap instead of grayscale.
        include_metadata: Save one JSON file per camera.

    Returns:
        A list of written file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = int(getattr(env, "current_step", 0))
    sim_time = float(getattr(getattr(env, "data", None), "time", 0.0))
    label = _safe_name(label or f"step{step:04d}")
    written = []

    for cam_cfg in _camera_cfgs(env, cameras):
        cam_name = _safe_name(cam_cfg.get("name", "rgbd"))
        rendered = env._render_rgbd_camera(cam_cfg)
        stem = f"{_safe_name(prefix)}_{label}_{cam_name}"
        meta = {
            "step": step,
            "sim_time": sim_time,
            "camera": str(cam_cfg.get("name", "")),
            "camera_config": copy.deepcopy(cam_cfg),
            "files": {},
            "render_ok": rendered is not None,
        }
        if extra_metadata:
            meta.update(copy.deepcopy(extra_metadata))
        if rendered is None:
            if include_metadata:
                json_path = out_dir / f"{stem}_meta.json"
                json_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
                written.append(json_path)
            continue

        rgb, depth = rendered
        rgb_path = out_dir / f"{stem}_rgb.png"
        _write_rgb(rgb_path, rgb)
        meta["files"]["rgb"] = str(rgb_path)
        written.append(rgb_path)

        if include_depth and depth is not None:
            depth_path = out_dir / f"{stem}_depth.png"
            _write_depth(depth_path, depth, colorize=colorize_depth)
            meta["files"]["depth"] = str(depth_path)
            written.append(depth_path)

        if include_detection:
            label_text = f"step={step} t={sim_time:.3f}s cam={cam_name}"
            overlay, det_info = _estimate_and_overlay(env, cam_cfg, rgb, depth, label_text)
            detect_path = out_dir / f"{stem}_detect.png"
            import cv2
            cv2.imwrite(str(detect_path), overlay)
            meta["files"]["detect"] = str(detect_path)
            meta["detection"] = det_info
            written.append(detect_path)

        try:
            meta["payload_pos"] = env.data.body("prefab").xpos.copy()
            meta["target_pos_xy"] = getattr(env, "target_pos", None)
            meta["vision_debug"] = env.get_vision_debug()
            meta["rope_marker_debug"] = env.get_rope_marker_feature_debug()
        except Exception as exc:
            meta["metadata_warning"] = str(exc)

        if include_metadata:
            json_path = out_dir / f"{stem}_meta.json"
            json_path.write_text(json.dumps(_jsonable(meta), indent=2), encoding="utf-8")
            written.append(json_path)

    return written


def build_capture_config(args):
    ckpt_path = getattr(args, "policy_ckpt", None)
    cfg, loaded_config_path = _load_resolved_config(
        getattr(args, "config_json", None),
        ckpt_path if getattr(args, "action_mode", "") == "policy" else None,
    )
    args._loaded_config_path = str(loaded_config_path) if loaded_config_path else ""
    cfg = _deep_update(cfg, {
        "sim": {
            "render": False,
            "max_steps": int(args.max_steps),
        },
        "scene": {
            "seed": int(args.seed),
        },
        "vision": {
            "enabled": False,
            "source": "opencv_rgbd",
            "apply_to_env_obs": False,
            "allow_truth_fallback": False,
            "render_width": int(args.render_width),
            "render_height": int(args.render_height),
            "reuse_rendered_rgbd_frames": False,
            "render_markers": False,
            "show_camera_models": False,
            "hide_sites_in_vision": not bool(args.show_connection_sites),
            "hide_tendons_in_vision": True,
            "debug_dump_frames": 0,
        },
        "rope_markers": {
            "enabled": False,
        },
        "observation_predictor": {
            "enabled": False,
        },
        "cable_latent_predictor": {
            "enabled": False,
        },
    })
    if args.obstacles is not None:
        cfg["scene"]["n_obstacles"] = int(args.obstacles)
    return cfg


def _make_phase_base_expert(phase, config, ik_solver):
    from controller import JointSpaceExpert

    if phase == "descent":
        kind = str(config.get("descent_rl", {}).get(
            "base_expert", "joint_space")).strip().lower()
        if kind in ("mpc", "traditional_mpc", "descent_mpc"):
            from traditional_experts import make_traditional_expert
            return make_traditional_expert("mpc", config, ik_solver)
        if kind in ("traditional_pid", "paper_pid"):
            from traditional_experts import make_traditional_expert
            return make_traditional_expert("pid", config, ik_solver)
        if kind in ("damped_pd", "traditional_damped_pd"):
            from traditional_experts import make_traditional_expert
            return make_traditional_expert("damped_pd", config, ik_solver)
    return JointSpaceExpert(config, ik_solver)


def _reset_env_for_capture(env, args, cfg):
    if args.action_mode == "policy":
        from train_phase import reset_for_phase

        obs, planned_path = reset_for_phase(
            env, "descent", cfg, rng_seed=int(args.seed))
    else:
        obs = env.reset()
        planned_path = env.get_planned_path()
    if not bool(args.show_debug_geoms):
        hidden = prepare_clean_snapshot_visuals(
            env, show_connection_sites=bool(args.show_connection_sites))
    else:
        hidden = []
    return obs, planned_path, hidden


def _make_policy_runtime(env, cfg, ckpt_path, obs, planned_path):
    from ee_acc_controller import EEAccController
    from phase_agent import PPOPhaseAgent
    from train_phase import _advance_expert_to_nearest_wp

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {ckpt_path}")

    agent = PPOPhaseAgent("descent", config=cfg)
    agent.load(str(ckpt_path))
    agent.reset_history()

    expert = _make_phase_base_expert("descent", cfg, env.ik_solver)
    current_q = env.data.qpos[:7].copy()
    expert.reset(obs, current_q, env=env)
    if planned_path is not None:
        expert.set_path(planned_path)
        pl_pos = env.data.body("prefab").xpos.copy()
        _advance_expert_to_nearest_wp(
            expert, planned_path, pl_pos, force_descent=True)

    ee_ctrl = EEAccController(cfg, env.ik_solver)
    ee_ctrl.reset(env._get_ee_pos(), current_q)

    return {
        "agent": agent,
        "expert": expert,
        "ee_ctrl": ee_ctrl,
        "start_xy": np.asarray(getattr(
            env, "episode_start_xy", env.default_start_xy),
            dtype=np.float32).copy(),
        "target_xy": env.target_pos.copy(),
        "prev_tilt": 0.0,
        "prev_yaw": 0.0,
        "ckpt_path": str(ckpt_path),
    }


def _policy_delta_q(env, cfg, obs, runtime, deterministic=True):
    from phase_agent import build_wind_obs
    from train_phase import (
        _apply_descent_pid_residual,
        build_phase_obs,
        compute_descent_base_delta_q,
        wind_obs_scale,
    )

    agent = runtime["agent"]
    expert = runtime["expert"]
    current_q = env.data.qpos[:7].copy().astype(np.float32)

    wind_obs = build_wind_obs(env, wind_obs_scale(cfg))
    base_dq = None
    if (bool(cfg.get("descent_rl", {}).get("pid_residual_mode", True)) or
            bool(cfg.get("descent_rl", {}).get("include_pid_base_obs", False))):
        try:
            base_dq = compute_descent_base_delta_q(
                expert, obs, current_q, env=env)
        except Exception:
            base_dq = np.zeros(7, dtype=np.float32)

    core, cable_raw, wind_obs, prev_tilt, prev_yaw = build_phase_obs(
        "descent",
        obs,
        env,
        runtime["start_xy"],
        runtime["target_xy"],
        runtime["prev_tilt"],
        runtime["prev_yaw"],
        wind_obs=wind_obs,
        base_action=base_dq,
    )
    runtime["prev_tilt"] = prev_tilt
    runtime["prev_yaw"] = prev_yaw

    p_obs = agent.encode_obs(core, cable_raw, wind_obs)
    norm_obs = agent.normalize_obs(p_obs, update=False)
    result = agent.act(norm_obs, deterministic=deterministic)
    policy_action = result[0] if isinstance(result, tuple) else result

    if bool(cfg.get("descent_rl", {}).get("pid_residual_mode", True)):
        delta_q, pid_dq = _apply_descent_pid_residual(
            expert, policy_action, obs, env, cfg, current_q, pid_dq=base_dq)
    else:
        vel_max_z = float(cfg.get("ee_control", {}).get(
            "vel_max_z_descent", 0.03))
        no_upward_z = bool(cfg.get("descent_rl", {}).get(
            "no_pid_no_upward_z", True))
        delta_q = runtime["ee_ctrl"].compute_delta_q(
            policy_action, current_q, env._get_ee_pos(),
            vel_max_z=vel_max_z, no_upward_z=no_upward_z)
        pid_dq = np.zeros(7, dtype=np.float32)

    runtime["last_policy_action"] = np.asarray(
        policy_action, dtype=np.float32).copy()
    runtime["last_pid_dq"] = np.asarray(pid_dq, dtype=np.float32).copy()
    runtime["last_delta_q"] = np.asarray(delta_q, dtype=np.float32).copy()
    return np.asarray(delta_q, dtype=np.float32)


def _runtime_metadata(args, runtime=None, hidden_debug_geoms=None):
    meta = {
        "action_mode": str(args.action_mode),
        "config_json": getattr(args, "_loaded_config_path", ""),
        "clean_render": {
            "render_markers": False,
            "rope_markers_enabled": False,
            "show_camera_models": False,
            "hide_sites_in_vision": not bool(args.show_connection_sites),
            "show_connection_sites": bool(args.show_connection_sites),
            "hide_tendons_in_vision": True,
            "hidden_debug_geoms": list(hidden_debug_geoms or []),
        },
    }
    if runtime is not None:
        meta["policy"] = {
            "phase": "descent",
            "checkpoint": runtime.get("ckpt_path", ""),
        }
        for key in ("last_policy_action", "last_pid_dq", "last_delta_q"):
            if key in runtime:
                meta[key] = np.asarray(runtime[key]).tolist()
    return meta


def rollout_and_capture(args):
    steps = _parse_steps(args.steps)
    if args.capture_every is not None and args.capture_every > 0:
        steps.update(range(0, int(args.max_steps) + 1, int(args.capture_every)))
    if not steps:
        steps.add(0)

    cfg = build_capture_config(args)
    xml_snapshot = None
    if bool(args.preserve_generated_xml):
        xml_snapshot = _snapshot_generated_xmls()
    env = None
    try:
        env = CableRobotEnvWithObstacles(cfg)
        obs, planned_path, hidden_debug_geoms = _reset_env_for_capture(env, args, cfg)
        if args.wind_speed is not None:
            env.set_wind_speed(float(args.wind_speed), float(args.wind_dir))
        runtime = None
        if args.action_mode == "policy":
            runtime = _make_policy_runtime(
                env, cfg, args.policy_ckpt, obs, planned_path)

        rng = np.random.default_rng(int(args.seed))
        cameras = None
        if args.cameras:
            cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]

        captured = []
        max_step = max(steps)
        while int(getattr(env, "current_step", 0)) <= max_step:
            step = int(getattr(env, "current_step", 0))
            if step in steps:
                captured.extend(capture_snapshot(
                    env,
                    args.out_dir,
                    label=f"step{step:04d}",
                    cameras=cameras,
                    prefix=args.prefix,
                    include_depth=not args.no_depth,
                    include_detection=not args.no_detection,
                    colorize_depth=bool(args.colorize_depth),
                    extra_metadata=_runtime_metadata(
                        args, runtime=runtime,
                        hidden_debug_geoms=hidden_debug_geoms),
                ))
            if step >= max_step:
                break
            if args.action_mode == "policy":
                action = _policy_delta_q(
                    env, cfg, obs, runtime,
                    deterministic=bool(args.deterministic_policy))
            elif args.action_mode == "random":
                scale = np.asarray(env.action_space_high, dtype=np.float32)
                action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
                action = action * scale * float(args.random_action_scale)
            else:
                action = np.zeros(env.action_dim, dtype=np.float32)
            obs, _reward, done, _truncated, _info = env.step(action)
            if done and not args.continue_after_done:
                break
        return captured
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if xml_snapshot is not None:
            _restore_generated_xmls(xml_snapshot)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture RGB/depth/detection snapshots from the simulation.")
    parser.add_argument("--out-dir", type=str,
                        default="test_results/figure_assets/sim_snapshots")
    parser.add_argument("--steps", type=str, default="0,20,40,60,80,100",
                        help="comma list/ranges, e.g. 0,20,60 or 0:100:10")
    parser.add_argument("--capture-every", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--prefix", type=str, default="sim")
    parser.add_argument("--cameras", type=str, default=None,
                        help="comma-separated camera names, default all RGB-D cameras")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-detection", action="store_true", default=True)
    parser.add_argument("--with-detection", dest="no_detection",
                        action="store_false")
    parser.add_argument("--colorize-depth", action="store_true")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--wind-speed", type=float, default=None)
    parser.add_argument("--wind-dir", type=float, default=0.0)
    parser.add_argument("--action-mode", choices=["zero", "random", "policy"],
                        default="policy")
    parser.add_argument("--policy-ckpt", type=str,
                        default=str(DEFAULT_MAINLINE_CKPT))
    parser.add_argument("--config-json", type=str, default=None,
                        help="resolved_config.json; defaults to policy checkpoint sibling")
    parser.add_argument("--deterministic-policy", action="store_true",
                        default=True)
    parser.add_argument("--stochastic-policy", dest="deterministic_policy",
                        action="store_false")
    parser.add_argument("--random-action-scale", type=float, default=0.15)
    parser.add_argument("--show-debug-geoms", action="store_true",
                        help="keep path/camera/debug marker geoms visible")
    parser.add_argument("--show-connection-sites", action="store_true",
                        help="show red payload lift-site dots at cable anchors")
    parser.add_argument("--hide-connection-sites",
                        dest="show_connection_sites", action="store_false")
    parser.add_argument("--preserve-generated-xml", action="store_true",
                        default=True,
                        help="restore generated XML files after capture")
    parser.add_argument("--leave-generated-xml", dest="preserve_generated_xml",
                        action="store_false",
                        help="leave the clean capture XML on disk for debugging")
    parser.add_argument("--continue-after-done", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    written = rollout_and_capture(args)
    print(f"[capture] wrote {len(written)} files to {args.out_dir}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
