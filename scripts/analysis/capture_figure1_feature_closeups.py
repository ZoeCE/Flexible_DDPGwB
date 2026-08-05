#!/usr/bin/env python3
"""Capture non-evaluation Figure 1 close-ups from the actual simulator.

The three captures deliberately isolate the task characteristics requested for
the paper illustration: near-insertion geometry, intermittent wind loading,
and cable deformation after a bounded joint-space excitation.  They never
write the training configuration or checkpoint assets.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from scripts.capture_simulation_snapshots import (
    _make_policy_runtime, _policy_delta_q, _render_scene_camera,
    _snapshot_generated_xmls, _restore_generated_xmls,
    prepare_clean_snapshot_visuals,
)
from train_phase import reset_for_phase
from PIL import Image

CKPT = ROOT / "saves" / "descent_ppo_paper_trueobs_residual_pretrain_trainable8_seed270627_20260704" / "ckpt_best_l8.pt"
OUT = ROOT / "outputs" / "figure1_complete" / "closeups"


def render(env, name, camera, metadata):
    rgb, _ = _render_scene_camera(env, {"_scene_camera": camera}, False)
    image_path = OUT / f"{name}.png"
    Image.fromarray(rgb).save(image_path)
    metadata.update({"camera": camera, "image": str(image_path)})
    (OUT / f"{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(image_path)


def set_arm_visible(env, visible):
    """Toggle only KUKA base/link geoms; rope bodies remain visible."""
    import mujoco
    arm_ids = set()
    for name in ("base", "link1", "link2", "link3", "link4", "link5", "link6", "link7"):
        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            arm_ids.add(body_id)
    for geom_id in range(env.model.ngeom):
        if int(env.model.geom_bodyid[geom_id]) in arm_ids:
            env.model.geom_rgba[geom_id, 3] = 1.0 if visible else 0.0


def main():
    if not CKPT.is_file():
        raise FileNotFoundError(CKPT)
    OUT.mkdir(parents=True, exist_ok=True)
    xml = _snapshot_generated_xmls()
    env = None
    try:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["sim"].update({"render": False, "max_steps": 260})
        cfg["scene"]["seed"] = 270909
        cfg["vision"].update({"enabled": False, "render_width": 640, "render_height": 480,
                              "render_markers": False, "show_camera_models": False})
        cfg["rope_markers"]["enabled"] = False
        cfg["wind"].update({"enabled": True, "projected_area": 0.15})
        cfg["descent_rl"]["early_stop_enabled"] = False
        env = CableRobotEnvWithObstacles(cfg)
        obs, planned = reset_for_phase(env, "descent", cfg, rng_seed=270909)
        prepare_clean_snapshot_visuals(env, show_connection_sites=True)
        runtime = _make_policy_runtime(env, cfg, CKPT, obs, planned)

        target = np.asarray(env.target_pos, dtype=float)
        # Aim below the payload center so the entry hole and rebars occupy the
        # lower half of the close-up instead of being hidden by the payload.
        near_cam = {"lookat": [float(target[0]), float(target[1]), 0.055], "distance": 0.52,
                    "azimuth": 235.0, "elevation": -10.0}
        # Level side view: make horizontal payload excursion readable without
        # perspective compression from an overhead camera.
        wind_cam = {"lookat": [float(target[0] - 0.04), float(target[1]), 0.29], "distance": 0.82,
                    "azimuth": 270.0, "elevation": 0.0}
        rope_cam = {"lookat": [float(target[0] - 0.04), float(target[1]), 0.30], "distance": 0.72,
                    "azimuth": 278.0, "elevation": -6.0}

        saved = set()
        for step in range(220):
            # A is captured before this point; B/C then exclude the arm body.
            if step == 64:
                set_arm_visible(env, False)
            # The wind panel uses repeated 10 m/s gusts with quiet intervals.
            gust = 64 <= step < 116
            if gust:
                # Rapidly changing direction produces a visible swing envelope.
                phase = (step - 64) // 13
                env.set_wind_speed(16.5, (phase % 4) * (np.pi / 2.0))
            else:
                env.clear_wind()

            action = _policy_delta_q(env, cfg, obs, runtime, deterministic=True)
            # A bounded end-effector excitation makes rope compliance visible.
            if 122 <= step < 148:
                action = action.copy()
                phase = step - 122
                action[1] += 0.30 * np.sin(phase * 0.9)
                action[3] -= 0.24 * np.sin(phase * 0.9)
                action[5] += 0.18 * np.cos(phase * 0.9)
            obs, _, done, _, info = env.step(action)

            if step == 60 and "insertion" not in saved:
                render(env, "insertion_closeup", near_cam, {
                    "mode": "policy", "step": step, "event": info.get("event"),
                    "purpose": "pre-insertion close-up showing rebar and payload opening",
                    "checkpoint": str(CKPT),
                })
                saved.add("insertion")
            if step in (76, 91, 106):
                name = f"wind_gust_closeup_{len([x for x in saved if x.startswith('wind')])+1}"
                render(env, name, wind_cam, {
                    "mode": "policy", "step": step, "wind_speed_mps": 16.5,
                    "wind_pattern": "direction switches every 13 control steps",
                    "purpose": "three-frame swing silhouette under rapidly varying wind",
                    "checkpoint": str(CKPT),
                })
                saved.add(name)
            if step == 138 and "rope" not in saved:
                render(env, "rope_excitation_closeup", rope_cam, {
                    "mode": "policy_plus_z_shake", "step": step,
                    "joint_pulse": {"joint_2": "0.30*sin", "joint_4": "-0.24*sin", "joint_6": "0.18*cos", "steps": [122, 147]},
                    "purpose": "cable deformation after end-effector z-direction excitation",
                    "checkpoint": str(CKPT),
                })
                saved.add("rope")
            # This is a visual stress-test capture, not an evaluation.  Keep
            # integrating after an early terminal flag so every requested
            # disturbance state can be rendered from the same rollout.
    finally:
        if env is not None:
            env.close()
        _restore_generated_xmls(xml)


if __name__ == "__main__":
    main()
