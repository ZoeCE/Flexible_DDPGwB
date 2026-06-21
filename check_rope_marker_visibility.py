#!/usr/bin/env python3
"""Check whether the RGB-D camera layout can see rope markers."""

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


def _marker_sets(diag):
    in_fov = set()
    visible = set()
    for cam in diag.get("cameras", []):
        for row in cam.get("markers", []):
            if row.get("in_fov", False):
                in_fov.add(row["name"])
            if bool(row.get("depth_visible", False)):
                visible.add(row["name"])
    return in_fov, visible


def _print_diag(step, diag):
    total = int(diag.get("markers_total", 0))
    in_fov, visible = _marker_sets(diag)
    print(
        f"[step {step:03d}] markers={total} "
        f"fov={len(in_fov)}/{total} ({diag.get('coverage_fov', 0.0)*100:.1f}%) "
        f"visible={len(visible)}/{total} "
        f"({diag.get('coverage_visible', 0.0)*100:.1f}%)"
    )
    for cam in diag.get("cameras", []):
        print(
            f"  {cam['name']:<10s} "
            f"fov={int(cam.get('in_fov', 0)):2d}/{total} "
            f"visible={int(cam.get('depth_visible', 0)):2d}/{total} "
            f"dist={cam.get('distance_min', 0.0):.3f}-"
            f"{cam.get('distance_max', 0.0):.3f}m"
        )
    missing = sorted({row["name"] for cam in diag.get("cameras", [])
                      for row in cam.get("markers", [])} - visible)
    if missing:
        print("  missing_visible:", ",".join(missing[:12]) +
              ("..." if len(missing) > 12 else ""))


def _append_csv(path, step, diag):
    new_file = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "step", "camera", "marker", "rope", "marker_index",
                "in_fov", "depth_visible", "u", "v", "camera_z",
                "distance", "depth",
            ])
        for cam in diag.get("cameras", []):
            for row in cam.get("markers", []):
                u, v = row.get("pixel", [np.nan, np.nan])
                writer.writerow([
                    step, cam.get("name", ""), row.get("name", ""),
                    row.get("rope", ""), row.get("marker_index", -1),
                    int(bool(row.get("in_fov", False))),
                    "" if row.get("depth_visible", None) is None
                    else int(bool(row.get("depth_visible", False))),
                    f"{float(u):.3f}" if np.isfinite(u) else "",
                    f"{float(v):.3f}" if np.isfinite(v) else "",
                    f"{float(row.get('camera_z', np.nan)):.6f}",
                    f"{float(row.get('distance', np.nan)):.6f}",
                    "" if row.get("depth", None) is None
                    else f"{float(row.get('depth')):.6f}",
                ])


def main():
    parser = argparse.ArgumentParser(
        description="Check RGB-D camera coverage of visual rope markers.")
    parser.add_argument("--steps", type=int, default=1,
                        help="number of control steps to sample")
    parser.add_argument("--phase", type=str, default="descent",
                        choices=["descent", "raw"],
                        help="descent uses the training descent initializer")
    parser.add_argument("--step-skip", type=int, default=1,
                        help="env.step calls between diagnostics")
    parser.add_argument("--no-depth", action="store_true",
                        help="only check FOV/range; skip depth occlusion")
    parser.add_argument("--csv", type=str, default=None,
                        help="optional per-marker CSV output")
    parser.add_argument("--render", action="store_true",
                        help="open MuJoCo viewer")
    parser.add_argument("--obstacles", type=int, default=0,
                        help="number of obstacles in scene")
    parser.add_argument("--target-xy-randomize", action="store_true")
    parser.add_argument("--target-xy-range", type=str, default=None)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    _deep_update(cfg, {
        "sim": {"render": bool(args.render)},
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
    try:
        if args.phase == "descent":
            from train_phase import CurriculumManager, reset_for_descent_with_cur
            cur = CurriculumManager(cfg, "descent")
            obs, _planned = reset_for_descent_with_cur(env, cfg, cur)
        else:
            obs = env.reset()
        if obs is None:
            raise RuntimeError("env.reset() returned None")
        diag = env.get_rope_marker_camera_diagnostics(
            use_depth=not bool(args.no_depth))
        _print_diag(0, diag)
        if args.csv:
            _append_csv(args.csv, 0, diag)
        zero = np.zeros(env.action_dim, dtype=np.float32)
        for step in range(1, max(1, int(args.steps))):
            for _ in range(max(1, int(args.step_skip))):
                env.step(zero)
            diag = env.get_rope_marker_camera_diagnostics(
                use_depth=not bool(args.no_depth))
            _print_diag(step, diag)
            if args.csv:
                _append_csv(args.csv, step, diag)
    finally:
        env.close()


if __name__ == "__main__":
    main()
