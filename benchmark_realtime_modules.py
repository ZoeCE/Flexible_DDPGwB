import argparse
import copy
import csv
import os
import time

import numpy as np

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from obs_predictor import build_cable_latent_predictor
from phase_agent import (
    PPOPhaseAgent,
    build_cruise_obs,
    build_descent_obs,
    build_wind_obs,
)


def _parse_xy_range(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return [v, v]
    text = str(value).strip()
    if not text:
        return None
    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) == 1:
        v = float(parts[0])
        return [v, v]
    return [float(parts[0]), float(parts[1])]


def build_benchmark_config(args):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.setdefault("sim", {})["render"] = False
    config.setdefault("train", {})["gpu_id"] = int(args.gpu)
    if args.obstacles is not None:
        config.setdefault("scene", {})["n_obstacles"] = int(args.obstacles)

    vision = config.setdefault("vision", {})
    vision["enabled"] = not bool(args.disable_vision)
    vision["source"] = "opencv_rgbd"
    if args.render_width is not None:
        vision["render_width"] = int(args.render_width)
    if args.render_height is not None:
        vision["render_height"] = int(args.render_height)
    if args.camera_names:
        wanted = {
            name.strip() for name in str(args.camera_names).split(",")
            if name.strip()
        }
        if wanted:
            vision["cameras"] = [
                cam for cam in list(vision.get("cameras", []))
                if str(cam.get("name", "")) in wanted
            ]
    if args.disable_render_cache:
        vision["reuse_rendered_rgbd_frames"] = False
    vision["measurement_period_steps"] = max(1, int(args.vision_period))
    vision["latency_steps"] = max(0, int(args.vision_latency_steps))
    vision["processing_delay_steps"] = max(
        0, int(args.vision_processing_delay_steps))

    markers = config.setdefault("rope_markers", {})
    markers["enabled"] = True
    markers["feature_source"] = str(args.rope_marker_feature_source).lower()
    if args.rope_marker_pos_noise is not None:
        sigma = max(0.0, float(args.rope_marker_pos_noise))
        markers["feature_pos_noise_std"] = [sigma, sigma, sigma]
    if args.rope_marker_pixel_noise is not None:
        markers["feature_pixel_noise_std"] = max(
            0.0, float(args.rope_marker_pixel_noise))
    if args.rope_marker_dropout is not None:
        markers["feature_dropout_prob"] = min(
            1.0, max(0.0, float(args.rope_marker_dropout)))
    if args.rope_marker_outlier_prob is not None:
        markers["feature_outlier_prob"] = min(
            1.0, max(0.0, float(args.rope_marker_outlier_prob)))
    if args.rope_marker_outlier_std is not None:
        markers["feature_outlier_std"] = max(
            0.0, float(args.rope_marker_outlier_std))
    if args.rope_marker_quantize_px:
        markers["feature_quantize_px"] = True

    task = config.setdefault("task", {})
    if args.target_xy_randomize:
        task["target_xy_randomize"] = True
    xy_range = _parse_xy_range(args.target_xy_range)
    if xy_range is not None:
        task["target_xy_range"] = xy_range
        if max(xy_range) > 0.0:
            task["target_xy_randomize"] = True

    if args.cable_latent_predictor or args.cable_latent_predictor_ckpt:
        clp = config.setdefault("cable_latent_predictor", {})
        clp["enabled"] = True
        clp["train_enabled"] = False
        clp["use_rope_marker_features"] = bool(
            args.cable_latent_use_rope_markers)
        clp["rope_marker_feature_source"] = markers["feature_source"]
        if args.cable_latent_predictor_ckpt:
            clp["checkpoint"] = str(args.cable_latent_predictor_ckpt)
    return config


def _phase_obs_from_env(args, env, obs, base_action=None):
    target_xy = np.asarray(
        getattr(env, "target_pos", env.config["task"].get(
            "default_target_xy", [-0.3, 0.2])),
        dtype=np.float32).reshape(-1)[:2]
    wind_obs = build_wind_obs(
        env,
        wind_scale_max=float(env.config.get(
            "wind_obs", {}).get("wind_speed_max", 16.5)))
    if args.phase == "cruise":
        return build_cruise_obs(
            obs, env, target_xy, wind_obs=wind_obs, base_action=base_action)
    return build_descent_obs(
        obs, env, target_xy, wind_obs=wind_obs, base_action=base_action)


def _visible_no_cable_vector(phase_obs, rope_marker_features=None):
    core = np.asarray(phase_obs[0], dtype=np.float32).reshape(-1)
    wind = np.asarray(phase_obs[2], dtype=np.float32).reshape(-1)
    parts = [core, wind]
    if rope_marker_features is not None:
        parts.append(np.asarray(
            rope_marker_features, dtype=np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def _measure(label, func, iterations, warmup):
    for _ in range(max(0, int(warmup))):
        func()
    times = []
    last = None
    for _ in range(max(1, int(iterations))):
        t0 = time.perf_counter()
        last = func()
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=np.float64)
    avg = float(np.mean(arr))
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    mx = float(np.max(arr))
    return {
        "module": label,
        "calls": int(arr.size),
        "avg_ms": avg * 1000.0,
        "p50_ms": p50 * 1000.0,
        "p95_ms": p95 * 1000.0,
        "max_ms": mx * 1000.0,
        "hz_avg": 1.0 / max(avg, 1e-12),
        "last": last,
    }


def _print_rows(rows):
    print("\nRealtime module benchmark")
    print("-" * 88)
    print(f"{'module':34s} {'avg ms':>10s} {'p95 ms':>10s} "
          f"{'max ms':>10s} {'Hz(avg)':>10s} {'calls':>7s}")
    print("-" * 88)
    for row in rows:
        print(f"{row['module']:34s} {row['avg_ms']:10.2f} "
              f"{row['p95_ms']:10.2f} {row['max_ms']:10.2f} "
              f"{row['hz_avg']:10.2f} {row['calls']:7d}")
    print("-" * 88)


def _write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ["module", "calls", "avg_ms", "p50_ms", "p95_ms", "max_ms",
            "hz_avg"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark max compute frequency of vision/RL modules.")
    parser.add_argument("--phase", choices=["cruise", "descent"],
                        default="descent")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="optional PPO checkpoint for actor timing")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--disable-vision", action="store_true")
    parser.add_argument("--vision-period", type=int, default=1)
    parser.add_argument("--vision-latency-steps", type=int, default=1)
    parser.add_argument("--vision-processing-delay-steps", type=int, default=1)
    parser.add_argument("--render-width", type=int, default=None)
    parser.add_argument("--render-height", type=int, default=None)
    parser.add_argument("--camera-names", type=str, default="",
                        help="comma-separated camera subset, e.g. rgbd_front,rgbd_right")
    parser.add_argument("--disable-render-cache", action="store_true",
                        help="disable same-frame RGB-D reuse inside combined benchmark")
    parser.add_argument("--include-cached-frame-rows", action="store_true",
                        help="also time OpenCV processing when RGB-D frames are already available")
    parser.add_argument("--rope-marker-feature-source", type=str,
                        default="opencv_rgbd",
                        choices=["site", "rgbd", "vision", "backproject",
                                 "backprojection", "opencv_rgbd", "opencv",
                                 "color_rgbd", "color"])
    parser.add_argument("--rope-marker-pos-noise", type=float, default=None)
    parser.add_argument("--rope-marker-pixel-noise", type=float, default=None)
    parser.add_argument("--rope-marker-dropout", type=float, default=None)
    parser.add_argument("--rope-marker-outlier-prob", type=float, default=None)
    parser.add_argument("--rope-marker-outlier-std", type=float, default=None)
    parser.add_argument("--rope-marker-quantize-px", action="store_true")
    parser.add_argument("--cable-latent-predictor", action="store_true")
    parser.add_argument("--cable-latent-predictor-ckpt", type=str,
                        default=None)
    parser.add_argument("--cable-latent-use-rope-markers",
                        action="store_true")
    parser.add_argument("--target-xy-randomize", action="store_true")
    parser.add_argument("--target-xy-range", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default=None)
    args = parser.parse_args()

    config = build_benchmark_config(args)
    env = CableRobotEnvWithObstacles(config=config)
    obs = env.reset()
    zero_action = np.zeros(env.action_dim, dtype=np.float32)
    phase_obs = _phase_obs_from_env(args, env, obs, base_action=zero_action)

    agent = None
    obs_history = None
    if args.ckpt:
        agent = PPOPhaseAgent(args.phase, config=config)
        agent.load(args.ckpt)
        agent.actor.eval()
        agent.critic.eval()
        obs_history = agent.make_obs_history()

    rope_features = None
    if bool(config.get("cable_latent_predictor", {}).get(
            "use_rope_marker_features", False)):
        rope_features = env.get_rope_marker_feature_vector()
    visible = _visible_no_cable_vector(phase_obs, rope_features)

    cable_pred = None
    if bool(config.get("cable_latent_predictor", {}).get("enabled", False)):
        cable_pred = build_cable_latent_predictor(
            config, args.phase, visible_dim=visible.size,
            latent_dim=int(config.get("cable_latent_predictor", {}).get(
                "latent_dim", 32)),
            action_dim=int(config.get("cable_latent_predictor", {}).get(
                "action_dim", 7)),
            device=getattr(agent, "device", None))
    if cable_pred is not None and args.cable_latent_predictor_ckpt:
        cable_pred.load(args.cable_latent_predictor_ckpt)
        cable_pred.net.eval()

    def clear_rgbd_cache():
        if hasattr(env, "_vision_frame_cache"):
            env._vision_frame_cache = {}

    def payload_vision(clear_cache=True):
        if clear_cache:
            clear_rgbd_cache()
        meas = env._generate_vision_measurement(force=True)
        env._vision_current = copy.deepcopy(meas)
        env._vision_last_update_step = int(getattr(env, "current_step", 0))
        return meas

    def rope_marker(clear_cache=True):
        if clear_cache:
            clear_rgbd_cache()
        features = env.get_rope_marker_feature_vector()
        return {
            "features": features,
            "debug": env.get_rope_marker_feature_debug(),
        }

    cable_action_dim = int(config.get(
        "cable_latent_predictor", {}).get("action_dim", 7))
    cable_zero_action = np.zeros(cable_action_dim, dtype=np.float32)

    def cable_latpred():
        if cable_pred is None:
            return None
        return cable_pred.predict_and_update(
            visible, action=cable_zero_action, phase=args.phase)[0]

    encoded_obs = None
    if agent is not None:
        cable_component = (
            cable_latpred() if cable_pred is not None else phase_obs[1])
        encoded_obs = agent.encode_obs(
            phase_obs[0], cable_component, phase_obs[2])
        encoded_obs = agent.normalize_obs(encoded_obs, update=False)

    def ppo_actor():
        if agent is None or encoded_obs is None:
            return None
        return agent.act_with_history(
            encoded_obs, obs_history, deterministic=True)[0]

    combo_history = agent.make_obs_history() if agent is not None else None

    def combined_chain(clear_cache=True):
        meas = payload_vision(clear_cache=clear_cache)
        local_obs = env._get_obs()
        local_phase_obs = _phase_obs_from_env(
            args, env, local_obs, base_action=zero_action)
        local_rope = None
        if bool(config.get("cable_latent_predictor", {}).get(
                "use_rope_marker_features", False)):
            local_rope = env.get_rope_marker_feature_vector()
        cable_component = local_phase_obs[1]
        if cable_pred is not None:
            local_visible = _visible_no_cable_vector(
                local_phase_obs, local_rope)
            cable_component = cable_pred.predict_and_update(
                local_visible, action=cable_zero_action,
                phase=args.phase)[0]
        action = None
        if agent is not None:
            local_encoded = agent.encode_obs(
                local_phase_obs[0], cable_component, local_phase_obs[2])
            local_encoded = agent.normalize_obs(local_encoded, update=False)
            action = agent.act_with_history(
                local_encoded, combo_history, deterministic=True)[0]
        return {
            "vision": meas,
            "rope": env.get_rope_marker_feature_debug(),
            "action": action,
        }

    rows = []
    if not args.disable_vision:
        rows.append(_measure("payload_apriltag_opencv_rgbd",
                             payload_vision, args.iterations, args.warmup))
    rows.append(_measure(
        f"rope_marker_{args.rope_marker_feature_source}",
        rope_marker, args.iterations, args.warmup))
    if args.include_cached_frame_rows:
        payload_vision(clear_cache=True)
        if not args.disable_vision:
            rows.append(_measure(
                "payload_apriltag_cached_frames",
                lambda: payload_vision(clear_cache=False),
                args.iterations, args.warmup))
        rows.append(_measure(
            f"rope_marker_cached_{args.rope_marker_feature_source}",
            lambda: rope_marker(clear_cache=False),
            args.iterations, args.warmup))
    if cable_pred is not None:
        rows.append(_measure("cable_latent_predictor",
                             cable_latpred, args.iterations, args.warmup))
    if agent is not None:
        rows.append(_measure("ppo_actor",
                             ppo_actor, args.iterations, args.warmup))
    rows.append(_measure("combined_obs_to_action_chain",
                         combined_chain, args.iterations, args.warmup))
    if args.include_cached_frame_rows:
        payload_vision(clear_cache=True)
        rows.append(_measure(
            "combined_cached_frame_chain",
            lambda: combined_chain(clear_cache=False),
            args.iterations, args.warmup))

    _print_rows(rows)
    _write_csv(args.out_csv, rows)

    vision_debug = env.get_vision_debug()
    rope_debug = env.get_rope_marker_feature_debug()
    if vision_debug:
        print("Vision last:",
              f"valid={vision_debug.get('valid')}",
              f"cams={vision_debug.get('active_cameras')}",
              f"reproj={vision_debug.get('reprojection_error', 0.0):.3f}px",
              f"depth={vision_debug.get('depth_rmse', 0.0)*1000.0:.2f}mm",
              f"reason={vision_debug.get('failure_reason', '') or '-'}")
    if rope_debug:
        print("Rope marker last:",
              f"src={rope_debug.get('source')}",
              f"visible={rope_debug.get('visible')}/"
              f"{rope_debug.get('markers_total')}",
              f"valid={rope_debug.get('valid_after_noise')}",
              f"cams={rope_debug.get('cameras')}",
              f"cam_est={rope_debug.get('camera_estimates')}",
              f"clusters={rope_debug.get('world_clusters', '-')}")
    if args.out_csv:
        print(f"Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
