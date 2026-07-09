#!/usr/bin/env python3
"""Interactive MuJoCo viewer for manual paper-figure screenshots.

This script does not use the RGB-D simulation cameras.  It opens the normal
MuJoCo passive viewer so you can rotate/zoom the scene yourself and capture the
screen with your OS screenshot tool.  By default it starts the payload above
the target/rebar endpoint and hides sensing markers for clean paper figures.

Keyboard controls in the MuJoCo window:
    Space : start / pause action execution
    N     : execute one action step while paused
    R     : reset scene and pause
    Q/Esc : quit
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_CONFIG
from controller import JointSpaceExpert
from mujoco_env_new import CableRobotEnvWithObstacles


KEY_SPACE = 32
KEY_ESCAPE = 256
KEY_N = ord("N")
KEY_Q = ord("Q")
KEY_R = ord("R")


class ViewerRunControl:
    def __init__(self, start_running=False):
        self.paused = not bool(start_running)
        self.single_step = False
        self.reset_requested = False
        self.quit_requested = False

    def key_callback(self, keycode):
        keycode = int(keycode)
        if keycode == KEY_SPACE:
            self.paused = not self.paused
            print(f"[viewer] {'paused' if self.paused else 'running'}")
        elif keycode == KEY_N:
            self.single_step = True
            self.paused = True
            print("[viewer] single step")
        elif keycode == KEY_R:
            self.reset_requested = True
            self.paused = True
            print("[viewer] reset requested")
        elif keycode in (KEY_Q, KEY_ESCAPE):
            self.quit_requested = True
            print("[viewer] quit requested")


def deep_update(dst, src):
    out = copy.deepcopy(dst)
    for key, val in (src or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def build_config(args):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    target_xy = list(cfg.get("task", {}).get("default_target_xy", [-0.3, 0.2]))
    payload_z_cruise = float(cfg.get("planning", {}).get("payload_z_cruise", 0.25))
    rope_cfg = cfg.get("rope", {})
    prefab_cfg = cfg.get("prefab", {})
    rope_total_length = (
        int(rope_cfg.get("num_segments", 10)) *
        float(rope_cfg.get("segment_length", 0.04))
    )
    lift_site_offset = float(prefab_cfg.get("lift_site_offset", 0.10))
    mocap_init_z = float(cfg.get("reset", {}).get("mocap_init_z", 0.6))
    init_prefab = list(cfg.get("reset", {}).get(
        "init_qpos_prefab", [target_xy[0], target_xy[1], payload_z_cruise, 1.0, 0.0, 0.0, 0.0]))
    if len(init_prefab) >= 3 and not args.keep_default_start:
        init_prefab[0] = float(target_xy[0])
        init_prefab[1] = float(target_xy[1])
        init_prefab[2] = float(payload_z_cruise)
        # Keep the visual rope endpoints aligned with payload lift sites.
        # Default reset is consistent because 0.10 + 0.10 + 0.40 = 0.60.
        mocap_init_z = payload_z_cruise + lift_site_offset + rope_total_length

    cfg = deep_update(cfg, {
        "sim": {
            # Delay viewer launch until the key callback is installed.
            "render": False,
            "max_steps": int(args.max_steps),
            "control_freq_hz": float(args.control_freq_hz),
        },
        "task": {
            # Screenshot mode defaults to a descent-style pose over the endpoint.
            "default_start_xy": target_xy if not args.keep_default_start
            else list(cfg.get("task", {}).get("default_start_xy", target_xy)),
            "target_xy_randomize": False,
        },
        "reset": {
            "init_qpos_prefab": init_prefab,
            "mocap_init_z": mocap_init_z,
        },
        "scene": {
            "seed": int(args.seed),
        },
        "rope_markers": {
            # Hide colored rope marker spheres/sites in clean screenshot mode.
            "enabled": bool(args.show_sensing_markers),
        },
        "vision": {
            # Keep vision disabled; this tool is for manual viewer screenshots.
            "enabled": False,
            # Hide payload AprilTag/marker board unless explicitly requested.
            "render_markers": bool(args.show_sensing_markers),
            "show_camera_models": bool(args.show_camera_models),
        },
    })
    if args.obstacles is not None:
        cfg["scene"]["n_obstacles"] = int(args.obstacles)
    if args.no_random_target:
        cfg.setdefault("task", {})["target_xy_randomize"] = False
    return cfg


def reset_scene(env, expert, args):
    obs = env.reset()
    apply_viewer_clean_visuals(env, args)
    if args.wind_speed > 0:
        env.set_wind_speed(float(args.wind_speed), float(args.wind_dir))
    elif hasattr(env, "clear_wind"):
        env.clear_wind()

    current_q = env.data.qpos[:7].copy()
    try:
        expert.reset(obs, current_q, env=env)
    except TypeError:
        expert.reset(obs, current_q)

    planned_path = env.get_planned_path() if hasattr(env, "get_planned_path") else None
    if planned_path is not None and hasattr(expert, "set_path"):
        expert.set_path(planned_path)

    if getattr(env, "viewer", None) is not None:
        env.viewer.sync()
    print(
        f"[viewer] reset done: step={env.current_step}, "
        f"start_xy={np.asarray(getattr(env, 'episode_start_xy', [0, 0]))}, "
        f"target_xy={np.asarray(getattr(env, 'target_pos', [0, 0]))}, "
        f"payload_z={float(env.data.body('prefab').xpos[2]):.3f}"
    )
    return obs


def apply_viewer_clean_visuals(env, args):
    """Hide diagnostic sites in the interactive viewer for clean screenshots."""
    viewer = getattr(env, "viewer", None)
    opt = getattr(viewer, "opt", None) if viewer is not None else None
    if opt is None:
        return
    try:
        if bool(args.hide_sites):
            for i in range(len(opt.sitegroup)):
                opt.sitegroup[i] = 0
        elif not bool(args.show_all_sites):
            # Keep the payload lift / rope-end connection points visible so
            # screenshots read as connected, while hiding EE/debug site groups.
            for i in range(len(opt.sitegroup)):
                opt.sitegroup[i] = 1 if i == 0 else 0
    except Exception:
        pass


def compute_action(env, obs, expert, args, rng):
    if args.action_source == "zero":
        return np.zeros(env.action_dim, dtype=np.float32)
    if args.action_source == "random":
        scale = np.asarray(env.action_space_high, dtype=np.float32)
        action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
        return action * scale * float(args.random_action_scale)

    current_q = env.data.qpos[:7].copy().astype(np.float32)
    try:
        return expert.compute_delta_q_target(obs, current_q, env=env)
    except TypeError:
        return expert.compute_delta_q_target(obs, current_q)


def viewer_is_running(env):
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return False
    is_running = getattr(viewer, "is_running", None)
    if callable(is_running):
        try:
            return bool(is_running())
        except Exception:
            return True
    return True


def run(args):
    cfg = build_config(args)
    control = ViewerRunControl(start_running=args.start_running)
    env = CableRobotEnvWithObstacles(cfg)
    rng = np.random.default_rng(int(args.seed))

    try:
        env._key_callback = control.key_callback
        env.render_mode = True
        env.config.setdefault("sim", {})["render"] = True
        expert = JointSpaceExpert(env.config, env.ik_solver)
        obs = reset_scene(env, expert, args)

        print("\nInteractive screenshot viewer")
        print("  Mouse: rotate / pan / zoom in MuJoCo viewer")
        print("  Space: start/pause action")
        print("  N: one step while paused")
        print("  R: reset and pause")
        print("  Q or Esc: quit")
        print("  Screenshot: use your OS screenshot tool when paused\n")

        status_every = max(1, int(args.status_every))
        while viewer_is_running(env) and not control.quit_requested:
            if control.reset_requested:
                control.reset_requested = False
                obs = reset_scene(env, expert, args)

            should_step = (not control.paused) or control.single_step
            if should_step:
                control.single_step = False
                t0 = time.perf_counter()
                action = compute_action(env, obs, expert, args, rng)
                obs, _reward, done, _truncated, info = env.step(action)
                if int(env.current_step) % status_every == 0:
                    payload = env.data.body("prefab").xpos.copy()
                    print(
                        f"[viewer] step={env.current_step:04d} "
                        f"payload=({payload[0]:+.3f},{payload[1]:+.3f},{payload[2]:+.3f}) "
                        f"done={done} success={bool(info.get('is_success', False))}"
                    )
                if done:
                    print("[viewer] episode done; paused. Press R to reset.")
                    control.paused = True
                if args.realtime:
                    elapsed = time.perf_counter() - t0
                    target_dt = max(float(env.dt) / max(float(args.speed), 1e-6), 0.0)
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)
            else:
                viewer = getattr(env, "viewer", None)
                if viewer is not None:
                    try:
                        viewer.sync()
                    except Exception:
                        break
                time.sleep(0.03)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manual MuJoCo viewer with keyboard start/pause control.")
    parser.add_argument("--action-source", choices=["expert", "zero", "random"],
                        default="expert",
                        help="action source used when running")
    parser.add_argument("--start-running", action="store_true",
                        help="start immediately instead of opening paused")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--control-freq-hz", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--obstacles", type=int, default=None)
    parser.add_argument("--wind-speed", type=float, default=0.0)
    parser.add_argument("--wind-dir", type=float, default=0.0)
    parser.add_argument("--random-action-scale", type=float, default=0.15)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="real-time playback multiplier")
    parser.add_argument("--no-realtime", dest="realtime", action="store_false")
    parser.set_defaults(realtime=True)
    parser.add_argument("--show-camera-models", action="store_true")
    parser.add_argument("--show-sensing-markers", action="store_true",
                        help="show rope marker spheres and payload AprilTag board")
    parser.add_argument("--show-all-sites", action="store_true",
                        help="show all MuJoCo diagnostic site groups")
    parser.add_argument("--hide-sites", action="store_true",
                        help="hide all MuJoCo sites, including lift connection points")
    parser.add_argument("--keep-default-start", action="store_true",
                        help="use config default_start_xy instead of target_xy")
    parser.add_argument("--no-random-target", action="store_true")
    parser.add_argument("--status-every", type=int, default=20)
    return parser.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
