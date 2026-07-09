#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_phase import build_config


PROFILES = [
    ("ctrl-c0-base", 0.0),
    ("ctrl-c1-init", 0.0),
    ("ctrl-c2-rope10", 0.0),
    ("ctrl-c3-wind2", 2.0),
    ("ctrl-c4-wind6", 6.0),
    ("ctrl-c4-wind8", 8.0),
    ("ctrl-c5-ratio4-w6", 6.0),
    ("ctrl-c5-ratio4-w8", 8.0),
]

VARIANTS = ["PID", "DampedPD", "MPC", "Mainline", "NoCable", "MainlineP2"]


def base_args(profile, wind):
    return dict(
        phase="descent",
        algo="ppo",
        ckpt=None,
        render=False,
        wait_for_space=False,
        gpu=0,
        task_profile=profile,
        wind_speed=wind,
        wind_speed_max=10.0,
        wind_dir=None,
        variable_wind=False,
        wind_speed_band_abs=None,
        wind_speed_band_frac=None,
        wind_speed_rate_std=None,
        wind_dir_band_rad=None,
        wind_dir_rate_std=None,
        obs_noise=0.0,
        act_noise=0.0,
        force_noise=0.0,
        obstacles=0,
        insert_target_z=0.10,
        insert_depth_min=0.025,
        control_freq_hz=None,
        keep_step_budget=False,
        scale_limits_with_control_dt=False,
        target_xy_randomize=False,
        disable_target_xy_randomize=False,
        target_xy_range=None,
        obs_predictor=False,
        disable_obs_predictor=True,
        obs_predictor_ckpt=None,
        obs_predictor_target_mode=None,
        obs_period=None,
        cable_latent_predictor=False,
        disable_cable_latent_predictor=True,
        cable_latent_predictor_ckpt=None,
        cable_latent_use_rope_markers=False,
        rope_marker_feature_source=None,
        rope_marker_pos_noise=None,
        rope_marker_pixel_noise=None,
        rope_marker_dropout=None,
        rope_marker_outlier_prob=None,
        rope_marker_outlier_std=None,
        rope_marker_quantize_px=False,
        vision=False,
        vision_shadow_eval=False,
        disable_vision=True,
        vision_latency_steps=None,
        vision_processing_delay_steps=None,
        vision_period=None,
        vision_pos_noise=None,
        vision_dropout=None,
        vision_debug_dump_dir=None,
        vision_debug_dump_frames=None,
        vision_debug_dump_start_step=None,
        vision_debug_dump_end_step=None,
        zero_cable_obs=False,
        disable_cable_obs=False,
        descent_base_expert=None,
        disable_descent_pid_base=False,
        keep_descent_base_obs=False,
        descent_z_soft_gate_full=None,
        descent_z_hard_gate=None,
        descent_z_min_speed_frac=None,
        descent_z_trickle_xy_gate=None,
        descent_base_v_max_z=None,
        descent_alignment_z_gate=None,
        descent_premature_descent_xy_gate=None,
        descent_action_rms_free=None,
        descent_action_magnitude_coef=None,
    )


def variant_args(args, variant):
    args = dict(args)
    if variant == "PID":
        args["algo"] = "expert"
        args["descent_base_expert"] = "traditional_pid"
    elif variant == "DampedPD":
        args["algo"] = "expert"
        args["descent_base_expert"] = "damped_pd"
    elif variant == "MPC":
        args["algo"] = "expert"
        args["descent_base_expert"] = "mpc"
    elif variant == "NoCable":
        args["disable_cable_obs"] = True
    elif variant == "MainlineP2":
        args["obs_predictor"] = True
        args["disable_obs_predictor"] = False
        args["obs_period"] = 2
        args["obs_predictor_target_mode"] = "non_cable_latent"
        args["obs_predictor_ckpt"] = (
            "saves/descent_ablation_fullobs_obspred_p2_ft_20260629_145921/"
            "ckpt_latest_obs_predictor.pt")
    return args


def flatten_config(profile, wind, variant, cfg):
    ins = cfg.get("insertion", {})
    drl = cfg.get("descent_rl", {})
    rope = cfg.get("rope", {})
    task = cfg.get("task", {})
    prefab = cfg.get("prefab", {})
    curr = cfg.get("curriculum", {}).get("descent_levels", [{}])[0]
    wind_cfg = cfg.get("wind", {})
    obs_pred = cfg.get("observation_predictor", {})
    vision = cfg.get("vision", {})
    ce = cfg.get("cable_encoder", {})
    return {
        "profile": profile,
        "variant": variant,
        "target_xy": json.dumps(task.get("default_target_xy")),
        "start_xy": json.dumps(task.get("default_start_xy")),
        "target_xy_randomize": bool(task.get("target_xy_randomize", False)),
        "actual_obstacles_runtime": 0,
        "obstacle_slots_config": cfg.get("scene", {}).get("n_obstacles"),
        "control_hz": cfg.get("sim", {}).get("control_freq_hz"),
        "payload_mass": prefab.get("mass"),
        "hole_size_m": (prefab.get("socket_hole_size") or [""])[0],
        "rebar_radius_m": cfg.get("target", {}).get("rebar_radius"),
        "rope_segments": rope.get("num_segments"),
        "rope_segment_length": rope.get("segment_length"),
        "rope_damping": rope.get("damping"),
        "rope_segment_mass": rope.get("segment_mass"),
        "init_xy": curr.get("init_xy"),
        "init_vel": curr.get("init_vel"),
        "init_tilt": curr.get("init_tilt"),
        "wind_arg": wind,
        "wind_max_curriculum": curr.get("wind_max"),
        "wind_speed_max_norm": cfg.get("wind_obs", {}).get("wind_speed_max"),
        "time_variable_wind": bool(wind_cfg.get("test_wind_variable", False)),
        "xy_tol_curriculum": curr.get("xy_tol"),
        "xy_tol_obs_norm": ins.get("xy_tolerance_train_end"),
        "physical_rebar_xy_tol": ins.get("physical_rebar_xy_tolerance"),
        "z_tol": ins.get("success_z_tolerance"),
        "tilt_tol_obs_norm": ins.get("tilt_tolerance_train_end"),
        "yaw_tol_obs_norm": ins.get("yaw_tolerance_train_end"),
        "pid_residual_mode": bool(drl.get("pid_residual_mode", True)),
        "base_expert": drl.get("base_expert", "joint_space"),
        "include_pid_base_obs": bool(drl.get("include_pid_base_obs", False)),
        "obs_dim": drl.get("obs_dim"),
        "cable_encoder_enabled": bool(ce.get("enabled", False)),
        "cable_encoder_output_dim": ce.get("output_dim"),
        "vision_enabled": bool(vision.get("enabled", False)),
        "obs_predictor_enabled": bool(obs_pred.get("enabled", False)),
        "obs_period": obs_pred.get("measurement_period_steps"),
        "obs_predictor_target_mode": obs_pred.get("target_mode"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_results/paper_controlled_ladder/config_audit.csv")
    args = parser.parse_args()
    out = Path(args.out)
    rows = []
    for profile, wind in PROFILES:
        for variant in VARIANTS:
            ns = SimpleNamespace(**variant_args(base_args(profile, wind), variant))
            cfg = build_config(ns)
            rows.append(flatten_config(profile, wind, variant, cfg))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved config audit CSV: {out}")


if __name__ == "__main__":
    main()
