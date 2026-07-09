#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DEFAULT_CONFIG
from mujoco_env_new import CableRobotEnvWithObstacles
from ee_acc_controller import EEAccController
from test_phase import (
    build_config,
    get_descent_eval_init,
    make_phase_base_expert,
    summarize_results,
    test_single_phase,
)


PROFILES = [
    ("ctrl-c0-base", 0.0),
    ("ctrl-c1-init", 0.0),
    ("ctrl-c2-rope10", 0.0),
    ("ctrl-c3-wind2", 2.0),
]


COMMON_BASE = {
    "anchor_alpha": 0.12,
    "vel_max_xy": 0.12,
    "vel_max_z": 0.022,
    "terminal_vel_scale": 0.60,
    "terminal_vel_min_xy": 0.0,
    "attitude_z_gate": True,
    "attitude_gate_min": 0.20,
    "terminal_align_enabled": True,
    "terminal_align_z_margin": 0.10,
    "terminal_align_xy": 0.11,
    "terminal_align_kp_xy": 2.8,
    "terminal_align_kd_xy": 0.90,
    "terminal_align_blend": 1.0,
    "terminal_yaw_boost_enabled": True,
    "terminal_yaw_boost_z_margin": 0.12,
    "terminal_yaw_kp_scale": 2.0,
    "terminal_yaw_vel_scale": 2.5,
}

COMMON_PARAM_KEYS = set(COMMON_BASE.keys())


PID_CANDIDATES = [
    {
        "name": "pid_low_i_align",
        "params": {
            **COMMON_BASE,
            "kp_xy": 2.4, "ki_xy": 0.20, "kd_payload_xy": 0.75,
            "k_swing_xy": 0.38, "k_rel_vel_xy": 0.24,
            "integral_limit": 0.012, "integral_active_xy": 0.08,
            "swing_fade_xy": 0.055, "swing_fade_z_margin": 0.045,
            "z_kp": 0.75, "z_kd": 0.50,
            "z_gate": 0.028, "z_full_gate": 0.010,
            "z_trickle_gate": 0.075, "z_min_gate_frac": 0.18,
            "yaw_kp": 1.20, "yaw_kd": 0.30, "vel_max_yaw": 0.70,
        },
    },
    {
        "name": "pid_no_i_strong_damp",
        "params": {
            **COMMON_BASE,
            "kp_xy": 2.0, "ki_xy": 0.0, "kd_payload_xy": 0.95,
            "k_swing_xy": 0.55, "k_rel_vel_xy": 0.32,
            "integral_limit": 0.0, "integral_active_xy": 0.0,
            "swing_fade_xy": 0.050, "swing_fade_z_margin": 0.040,
            "z_kp": 0.70, "z_kd": 0.55,
            "z_gate": 0.030, "z_full_gate": 0.010,
            "z_trickle_gate": 0.085, "z_min_gate_frac": 0.12,
            "yaw_kp": 1.35, "yaw_kd": 0.35, "vel_max_yaw": 0.80,
        },
    },
    {
        "name": "pid_slow_precision",
        "params": {
            **COMMON_BASE,
            "vel_max_xy": 0.09, "vel_max_z": 0.018,
            "kp_xy": 1.7, "ki_xy": 0.12, "kd_payload_xy": 1.05,
            "k_swing_xy": 0.62, "k_rel_vel_xy": 0.36,
            "integral_limit": 0.008, "integral_active_xy": 0.07,
            "swing_fade_xy": 0.045, "swing_fade_z_margin": 0.035,
            "terminal_align_kp_xy": 3.2, "terminal_align_kd_xy": 1.20,
            "z_kp": 0.60, "z_kd": 0.60,
            "z_gate": 0.024, "z_full_gate": 0.008,
            "z_trickle_gate": 0.065, "z_min_gate_frac": 0.08,
            "yaw_kp": 1.50, "yaw_kd": 0.45, "vel_max_yaw": 0.90,
        },
    },
    {
        "name": "pid_fast_align",
        "params": {
            **COMMON_BASE,
            "vel_max_xy": 0.16, "vel_max_z": 0.024,
            "kp_xy": 2.9, "ki_xy": 0.18, "kd_payload_xy": 0.75,
            "k_swing_xy": 0.35, "k_rel_vel_xy": 0.22,
            "integral_limit": 0.010, "integral_active_xy": 0.09,
            "swing_fade_xy": 0.060, "swing_fade_z_margin": 0.050,
            "terminal_align_kp_xy": 3.0, "terminal_align_kd_xy": 0.85,
            "z_kp": 0.80, "z_kd": 0.45,
            "z_gate": 0.032, "z_full_gate": 0.012,
            "z_trickle_gate": 0.090, "z_min_gate_frac": 0.20,
            "yaw_kp": 1.20, "yaw_kd": 0.28, "vel_max_yaw": 0.80,
        },
    },
]


PD_CANDIDATES = [
    {
        "name": "pd_strong_damp",
        "params": {
            **COMMON_BASE,
            "kp_xy": 2.0, "ki_xy": 0.0, "kd_payload_xy": 1.05,
            "k_swing_xy": 0.58, "k_rel_vel_xy": 0.36,
            "swing_fade_xy": 0.050, "swing_fade_z_margin": 0.040,
            "z_kp": 0.62, "z_kd": 0.60,
            "z_gate": 0.030, "z_full_gate": 0.010,
            "z_trickle_gate": 0.080, "z_min_gate_frac": 0.10,
            "yaw_kp": 1.40, "yaw_kd": 0.38, "vel_max_yaw": 0.85,
        },
    },
    {
        "name": "pd_slow_precision",
        "params": {
            **COMMON_BASE,
            "vel_max_xy": 0.085, "vel_max_z": 0.018,
            "kp_xy": 1.55, "ki_xy": 0.0, "kd_payload_xy": 1.20,
            "k_swing_xy": 0.72, "k_rel_vel_xy": 0.45,
            "swing_fade_xy": 0.045, "swing_fade_z_margin": 0.035,
            "terminal_align_kp_xy": 3.4, "terminal_align_kd_xy": 1.35,
            "z_kp": 0.55, "z_kd": 0.65,
            "z_gate": 0.024, "z_full_gate": 0.008,
            "z_trickle_gate": 0.065, "z_min_gate_frac": 0.08,
            "yaw_kp": 1.60, "yaw_kd": 0.50, "vel_max_yaw": 0.95,
        },
    },
    {
        "name": "pd_fast_terminal",
        "params": {
            **COMMON_BASE,
            "vel_max_xy": 0.15, "vel_max_z": 0.024,
            "kp_xy": 2.7, "ki_xy": 0.0, "kd_payload_xy": 0.85,
            "k_swing_xy": 0.45, "k_rel_vel_xy": 0.28,
            "swing_fade_xy": 0.060, "swing_fade_z_margin": 0.050,
            "terminal_align_kp_xy": 3.1, "terminal_align_kd_xy": 0.95,
            "z_kp": 0.78, "z_kd": 0.48,
            "z_gate": 0.032, "z_full_gate": 0.012,
            "z_trickle_gate": 0.090, "z_min_gate_frac": 0.18,
            "yaw_kp": 1.25, "yaw_kd": 0.32, "vel_max_yaw": 0.85,
        },
    },
    {
        "name": "pd_xy_integratorless_align",
        "params": {
            **COMMON_BASE,
            "vel_max_xy": 0.11, "vel_max_z": 0.020,
            "kp_xy": 2.3, "ki_xy": 0.0, "kd_payload_xy": 0.95,
            "k_swing_xy": 0.50, "k_rel_vel_xy": 0.30,
            "swing_fade_xy": 0.052, "swing_fade_z_margin": 0.040,
            "terminal_align_kp_xy": 3.6, "terminal_align_kd_xy": 1.10,
            "z_kp": 0.70, "z_kd": 0.55,
            "z_gate": 0.028, "z_full_gate": 0.010,
            "z_trickle_gate": 0.075, "z_min_gate_frac": 0.10,
            "yaw_kp": 1.45, "yaw_kd": 0.40, "vel_max_yaw": 0.90,
        },
    },
]


def make_args(profile):
    return SimpleNamespace(
        phase="descent", algo="expert", ckpt=None, episodes=0,
        obstacles=0, seed=0, task_profile=profile, render=False,
        wait_for_space=False, gpu=0, insert_target_z=0.10,
        insert_depth_min=0.025, control_freq_hz=None, keep_step_budget=False,
        scale_limits_with_control_dt=False, wind_speed=0.0,
        wind_speed_max=10.0, wind_dir=None, variable_wind=False,
        wind_speed_band_abs=None, wind_speed_band_frac=None,
        wind_speed_rate_std=None, wind_dir_band_rad=None,
        wind_dir_rate_std=None, obs_noise=0.0, act_noise=0.0,
        force_noise=0.0, eval_curriculum_level="max",
        target_xy_randomize=False, disable_target_xy_randomize=False,
        target_xy_range=None, compare_wind_bins=False,
        compare_multi_rl_wind_bins=False, compare_pred_true_expert=False,
        true_obs_ckpt=None, pred_ckpt=None, pred_obs_ckpt=None,
        pred_obs_period=2, multi_rl_ckpts=None, obs_predictor=False,
        disable_obs_predictor=True, obs_predictor_ckpt=None,
        obs_predictor_target_mode=None, obs_period=None,
        cable_latent_predictor=False, disable_cable_latent_predictor=True,
        cable_latent_predictor_ckpt=None, cable_latent_use_rope_markers=False,
        rope_marker_feature_source=None, rope_marker_pos_noise=None,
        rope_marker_pixel_noise=None, rope_marker_dropout=None,
        rope_marker_outlier_prob=None, rope_marker_outlier_std=None,
        rope_marker_quantize_px=False, vision=False,
        vision_shadow_eval=False, vision_shadow_csv=None,
        summary_out=None, disable_vision=True, vision_latency_steps=None,
        vision_processing_delay_steps=None, vision_period=None,
        vision_pos_noise=None, vision_dropout=None,
        vision_debug_dump_dir=None, vision_debug_dump_frames=None,
        vision_debug_dump_start_step=None, vision_debug_dump_end_step=None,
        zero_cable_obs=False, disable_cable_obs=False,
        descent_base_expert=None, descent_z_soft_gate_full=None,
        descent_z_hard_gate=None, descent_z_min_speed_frac=None,
        descent_z_trickle_xy_gate=None, descent_base_v_max_z=None,
        disable_descent_pid_base=False, keep_descent_base_obs=False,
        descent_alignment_z_gate=None, descent_premature_descent_xy_gate=None,
        descent_action_rms_free=None, descent_action_magnitude_coef=None,
        quiet=True,
    )


def evaluate(expert_name, candidate, profiles, episodes, seed_base, out_dir):
    rows = []
    for pidx, (profile, wind) in enumerate(profiles):
        args = make_args(profile)
        args.wind_speed = wind
        args.seed = seed_base + pidx * 1000
        args.descent_base_expert = (
            "traditional_pid" if expert_name == "pid" else "damped_pd")
        config = build_config(args)
        config["scene"]["seed"] = args.seed
        common_params = {
            key: val for key, val in candidate["params"].items()
            if key in COMMON_PARAM_KEYS
        }
        expert_params = {
            key: val for key, val in candidate["params"].items()
            if key not in COMMON_PARAM_KEYS
        }
        texperts = config.setdefault("traditional_experts", {})
        texperts.setdefault("common", {}).update(copy.deepcopy(common_params))
        texperts.setdefault(expert_name, {}).update(copy.deepcopy(expert_params))

        env = CableRobotEnvWithObstacles(config=config)
        try:
            if hasattr(env, "set_curriculum_n_obstacles"):
                env.set_curriculum_n_obstacles(0)
            expert = make_phase_base_expert("descent", config, env.ik_solver)
            ee_ctrl = EEAccController(config, env.ik_solver)
            eval_init = get_descent_eval_init(config, "max")
            t0 = time.perf_counter()
            results = test_single_phase(
                env, None, expert, ee_ctrl, "descent", config,
                n_episodes=episodes,
                wind_speed=wind,
                eval_cur_init=eval_init,
                verbose=False,
                progress_every=0,
            )
            elapsed = time.perf_counter() - t0
        finally:
            env.close()
        summary = summarize_results(results)
        row = {
            "expert": expert_name,
            "candidate": candidate["name"],
            "profile": profile,
            "wind": wind,
            "episodes": episodes,
            "seed": args.seed,
            "elapsed_s": elapsed,
            "strict_success_rate": summary.get("strict_success_rate", 0.0),
            "broad_success_rate": summary.get("broad_success_rate", 0.0),
            "avg_steps": summary.get("avg_steps", 0.0),
            "avg_angle": summary.get("avg_angle", 0.0),
            "avg_ke_mJ": summary.get("avg_ke_mJ", 0.0),
            "terminal_dtf_m_mean": summary.get("terminal_dtf_m_mean", ""),
            "terminal_tilt_rad_mean": summary.get("terminal_tilt_rad_mean", ""),
            "terminal_worst_rebar_err_m_mean": summary.get(
                "terminal_worst_rebar_err_m_mean", ""),
            "terminal_insert_depth_m_mean": summary.get(
                "terminal_insert_depth_m_mean", ""),
            "termination_counts": json.dumps(
                summary.get("termination_counts", {}), sort_keys=True),
        }
        rows.append(row)
        print(
            f"[{expert_name}] {candidate['name']} {profile}: "
            f"SR={row['strict_success_rate']*100:.1f}% "
            f"bSR={row['broad_success_rate']*100:.1f}% "
            f"steps={row['avg_steps']:.1f}",
            flush=True,
        )
    return rows


def score_candidate(rows):
    # Prioritize a nonzero/healthy C0 success rate, then robustness into C1-C3.
    sr = {r["profile"]: float(r["broad_success_rate"]) for r in rows}
    steps = np.mean([float(r["avg_steps"]) for r in rows])
    stability = np.mean([float(r["avg_ke_mJ"]) for r in rows])
    return (
        4.0 * sr.get("ctrl-c0-base", 0.0) +
        2.0 * sr.get("ctrl-c1-init", 0.0) +
        1.5 * sr.get("ctrl-c2-rope10", 0.0) +
        1.0 * sr.get("ctrl-c3-wind2", 0.0) -
        0.0005 * steps -
        0.0001 * stability
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=270802)
    parser.add_argument("--out-dir", default="test_results/paper_controlled_ladder/tuning_pid_pd_20260702")
    parser.add_argument("--profiles", default="ctrl-c0-base,ctrl-c1-init,ctrl-c2-rope10,ctrl-c3-wind2")
    args = parser.parse_args()

    selected = {p.strip() for p in args.profiles.split(",") if p.strip()}
    profiles = [p for p in PROFILES if p[0] in selected]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    best = {}
    for expert_name, candidates in [
        ("pid", PID_CANDIDATES),
        ("damped_pd", PD_CANDIDATES),
    ]:
        best_score = -1e9
        best_candidate = None
        for cidx, candidate in enumerate(candidates):
            rows = evaluate(
                expert_name, candidate, profiles, args.episodes,
                args.seed + cidx * 10000 + (0 if expert_name == "pid" else 100000),
                out_dir,
            )
            score = score_candidate(rows)
            for row in rows:
                row["score"] = score
                row["params_json"] = json.dumps(candidate["params"], sort_keys=True)
            all_rows.extend(rows)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        best[expert_name] = {
            "name": best_candidate["name"],
            "score": best_score,
            "params": best_candidate["params"],
        }

    csv_path = out_dir / "tuning_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    json_path = out_dir / "best_params.json"
    json_path.write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved tuning CSV: {csv_path}")
    print(f"Saved best params: {json_path}")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
