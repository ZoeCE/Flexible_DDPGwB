#!/usr/bin/env python3
"""Evaluate RGB-D rope-marker coordinate accuracy against MuJoCo truth.

This script uses an ideal marker detector: marker centers are projected from the
simulator truth to each RGB-D image, then the rendered depth image is sampled and
back-projected to the lab frame. The resulting error is therefore an upper-bound
check of camera geometry, depth rendering, occlusion, and marker placement. It
does not measure color/marker image-detection robustness.
"""

import argparse
import copy
import csv
import os

import numpy as np

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles


def _parse_xy_range(text):
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) == 1:
        v = max(0.0, float(parts[0]))
        return [v, v]
    if len(parts) == 2:
        return [max(0.0, float(parts[0])), max(0.0, float(parts[1]))]
    raise ValueError("--target-xy-range expects one value or x,y")


def _deep_update(dst, src):
    for key, val in (src or {}).items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], val)
        else:
            dst[key] = val
    return dst


def _percentile(values, q):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def _summarize_mm(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return "n/a"
    return (
        f"avg={np.mean(arr) * 1000.0:.2f}mm "
        f"p50={np.percentile(arr, 50) * 1000.0:.2f}mm "
        f"p95={np.percentile(arr, 95) * 1000.0:.2f}mm "
        f"max={np.max(arr) * 1000.0:.2f}mm"
    )


def _estimate_marker_positions(env, quantize_px=False, pixel_noise_std=0.0):
    """Return per-camera and fused marker estimates from RGB-D depth."""
    width, height = env._vision_resolution()
    markers = env.get_rope_marker_world_positions()
    cfg_marker = env.config.get("rope_markers", {})
    depth_tol = float(cfg_marker.get("visibility_depth_tolerance", 0.035))
    depth_radius = int(cfg_marker.get("visibility_depth_window", 2))

    marker_truth = {m["name"]: np.asarray(m["pos"], dtype=np.float64)
                    for m in markers}
    marker_meta = {m["name"]: m for m in markers}
    per_camera = []
    fused_by_marker = {}
    in_fov = set()
    depth_visible = set()

    for cam_cfg in list(env.cfg_vision.get("cameras", [])):
        cam_name = str(cam_cfg.get("name", "rgbd"))
        if env._vision_camera_id(cam_name) < 0:
            continue
        rendered = env._render_rgbd_camera(cam_cfg)
        if rendered is None:
            continue
        _rgb, depth = rendered
        if depth is None:
            continue

        K = env._vision_camera_matrix(cam_cfg)
        T_lab_cam = env._vision_camera_lab_transform(cam_cfg)
        R_lab_cam = T_lab_cam[:3, :3]
        t_lab_cam = T_lab_cam[:3, 3]
        max_range = float(cam_cfg.get("max_range", np.inf))
        cam_estimates = []

        for marker in markers:
            name = marker["name"]
            p_lab = marker_truth[name]
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
            if pixel_noise_std > 0.0:
                u_det += np.random.normal(0.0, pixel_noise_std)
                v_det += np.random.normal(0.0, pixel_noise_std)

            z_depth = env._sample_depth_patch(
                depth, u_det, v_det, radius=depth_radius)
            if z_depth is None:
                continue
            z_depth = float(z_depth)
            if abs(z_depth - z_true) > depth_tol:
                continue

            depth_visible.add(name)
            x = (float(u_det) - K[0, 2]) * z_depth / K[0, 0]
            y = (float(v_det) - K[1, 2]) * z_depth / K[1, 1]
            p_est_lab = R_lab_cam @ np.array([x, y, z_depth]) + t_lab_cam
            err = p_est_lab - p_lab
            norm = float(np.linalg.norm(err))
            row = {
                "camera": cam_name,
                "marker": name,
                "rope": marker_meta[name]["rope"],
                "marker_index": int(marker_meta[name]["marker_index"]),
                "u": float(u),
                "v": float(v),
                "z_true": z_true,
                "z_depth": z_depth,
                "err_x": float(err[0]),
                "err_y": float(err[1]),
                "err_z": float(err[2]),
                "err_norm": norm,
                "p_est": p_est_lab,
                "p_true": p_lab,
            }
            per_camera.append(row)
            cam_estimates.append(row)
            fused_by_marker.setdefault(name, []).append(p_est_lab)

    fused = []
    for name, estimates in fused_by_marker.items():
        p_est = np.mean(np.asarray(estimates, dtype=np.float64), axis=0)
        p_true = marker_truth[name]
        err = p_est - p_true
        fused.append({
            "marker": name,
            "rope": marker_meta[name]["rope"],
            "marker_index": int(marker_meta[name]["marker_index"]),
            "n_cameras": len(estimates),
            "err_x": float(err[0]),
            "err_y": float(err[1]),
            "err_z": float(err[2]),
            "err_norm": float(np.linalg.norm(err)),
        })

    return {
        "markers_total": len(markers),
        "in_fov": len(in_fov),
        "depth_visible": len(depth_visible),
        "per_camera": per_camera,
        "fused": fused,
    }


def _append_csv(path, step, result):
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "step", "kind", "camera", "marker", "rope", "marker_index",
                "n_cameras", "err_x_m", "err_y_m", "err_z_m", "err_norm_m",
                "u", "v", "z_true_m", "z_depth_m",
            ])
        for row in result["per_camera"]:
            writer.writerow([
                step, "camera", row["camera"], row["marker"], row["rope"],
                row["marker_index"], 1, f"{row['err_x']:.8f}",
                f"{row['err_y']:.8f}", f"{row['err_z']:.8f}",
                f"{row['err_norm']:.8f}", f"{row['u']:.3f}",
                f"{row['v']:.3f}", f"{row['z_true']:.8f}",
                f"{row['z_depth']:.8f}",
            ])
        for row in result["fused"]:
            writer.writerow([
                step, "fused", "", row["marker"], row["rope"],
                row["marker_index"], row["n_cameras"], f"{row['err_x']:.8f}",
                f"{row['err_y']:.8f}", f"{row['err_z']:.8f}",
                f"{row['err_norm']:.8f}", "", "", "", "",
            ])


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RGB-D rope-marker reconstruction accuracy.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--step-skip", type=int, default=1)
    parser.add_argument("--phase", choices=["descent", "raw"], default="descent")
    parser.add_argument("--obstacles", type=int, default=0)
    parser.add_argument("--target-xy-randomize", action="store_true")
    parser.add_argument("--target-xy-range", type=str, default=None)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--quantize-px", action="store_true",
                        help="round projected marker center to integer pixel")
    parser.add_argument("--pixel-noise-std", type=float, default=0.0,
                        help="optional marker-center detection noise in pixels")
    args = parser.parse_args()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    _deep_update(cfg, {
        "sim": {"render": False},
        "scene": {"n_obstacles": int(args.obstacles), "seed": int(args.seed)},
        "vision": {"enabled": True},
        "rope_markers": {"enabled": True},
    })
    xy_range = _parse_xy_range(args.target_xy_range)
    if xy_range is not None:
        cfg.setdefault("task", {})["target_xy_range"] = xy_range
        cfg.setdefault("task", {})["target_xy_randomize"] = (
            bool(args.target_xy_randomize) or max(xy_range) > 0.0)
    elif args.target_xy_randomize:
        cfg.setdefault("task", {})["target_xy_randomize"] = True

    env = CableRobotEnvWithObstacles(config=cfg)
    all_camera_err = []
    all_fused_err = []
    all_fused_cam_counts = []
    total_markers = None
    total_slots = 0
    total_in_fov = 0
    total_visible = 0
    try:
        if args.phase == "descent":
            from train_phase import CurriculumManager, reset_for_descent_with_cur
            cur = CurriculumManager(cfg, "descent")
            obs, _planned = reset_for_descent_with_cur(env, cfg, cur)
        else:
            obs = env.reset()
        if obs is None:
            raise RuntimeError("env.reset() returned None")

        zero = np.zeros(env.action_dim, dtype=np.float32)
        for step in range(max(1, int(args.steps))):
            if step > 0:
                for _ in range(max(1, int(args.step_skip))):
                    env.step(zero)
            result = _estimate_marker_positions(
                env,
                quantize_px=bool(args.quantize_px),
                pixel_noise_std=max(0.0, float(args.pixel_noise_std)),
            )
            total_markers = int(result["markers_total"])
            total_slots += total_markers
            total_in_fov += int(result["in_fov"])
            total_visible += int(result["depth_visible"])
            all_camera_err.extend(row["err_norm"] for row in result["per_camera"])
            all_fused_err.extend(row["err_norm"] for row in result["fused"])
            all_fused_cam_counts.extend(row["n_cameras"] for row in result["fused"])
            if args.csv:
                _append_csv(args.csv, step, result)
            print(
                f"[step {step:03d}] markers={total_markers} "
                f"fov={result['in_fov']}/{total_markers} "
                f"visible={result['depth_visible']}/{total_markers} "
                f"camera_est={len(result['per_camera'])} "
                f"fused={len(result['fused'])}/{total_markers}"
            )

        print("\n=== Rope marker RGB-D accuracy ===")
        print(f"steps: {max(1, int(args.steps))}")
        print(f"markers per step: {total_markers}")
        print(
            f"FOV coverage: {total_in_fov}/{total_slots} "
            f"({100.0 * total_in_fov / max(total_slots, 1):.1f}%)")
        print(
            f"Depth-visible coverage: {total_visible}/{total_slots} "
            f"({100.0 * total_visible / max(total_slots, 1):.1f}%)")
        print(f"Single-camera position error: {_summarize_mm(all_camera_err)}")
        print(f"Fused position error:         {_summarize_mm(all_fused_err)}")
        if all_fused_cam_counts:
            counts = np.asarray(all_fused_cam_counts, dtype=np.float64)
            print(
                f"Active cameras per fused marker: "
                f"avg={np.mean(counts):.2f}, "
                f"p05={_percentile(counts, 5):.0f}, "
                f"p95={_percentile(counts, 95):.0f}")
        if args.csv:
            print(f"Saved CSV: {args.csv}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
