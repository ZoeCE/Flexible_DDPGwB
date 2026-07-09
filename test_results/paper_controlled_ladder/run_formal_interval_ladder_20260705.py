#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_phase import (  # noqa: E402
    CableRobotEnvWithObstacles,
    EEAccController,
    build_config,
    get_descent_eval_init,
    load_agent,
    make_phase_base_expert,
    summarize_results,
    test_single_phase,
)


TRADITIONAL_OVERRIDES = (
    "test_results/paper_controlled_ladder/"
    "tuned_traditional_experts_20260702.json"
)

CKPTS = {
    "Full8D": (
        "saves/descent_ppo_paper_trueobs_residual_pretrain_trainable8_"
        "seed270627_20260704/ckpt_best_l8.pt"
    ),
    "NoCable": (
        "saves/descent_ablation_trueobs_no_cable_latent_pretrain_l8best_"
        "20260702_parallel_l8best/ckpt_best_l8.pt"
    ),
    "Full16D": (
        "saves/descent_ppo_paper_trueobs_residual_pretrain_trainable16_"
        "seed270627_20260703/ckpt_best_l8.pt"
    ),
    "MLP8D": (
        "saves/descent_ablation_fullobs_mlp_trainable8_seed270627_20260705/"
        "ckpt_best_l8.pt"
    ),
}

MISSING_OR_NOT_FORMAL = {
    "MainlineP2": (
        "No ckpt_best_l8.pt is registered for the ObsPred P2 finetune. "
        "Do not substitute ckpt_latest.pt for the formal L8-best protocol."
    ),
    "NoPID": (
        "No successful L8-best no-PID/full-action checkpoint exists; current "
        "no-PID runs are recorded as failed evidence only."
    ),
    "LegacyMLP32": (
        "The historical A5 MLP checkpoint used a frozen 32D cable encoder. "
        "It is preserved in old JSON files but excluded from the formal "
        "paper summary after the matched trainable-8D MLP run became available."
    ),
}

LEVELS = [
    {
        "id": "C0",
        "label": "C0 base",
        "profile": "ctrl-c0-base",
        "wind_low": 0.0,
        "wind_high": 2.0,
        "added_factor": "base",
        "hole_ratio": 8.0,
        "rope_segments": 4,
        "init_xy_m": 0.0,
    },
    {
        "id": "C1",
        "label": "C1 + init randomization",
        "profile": "ctrl-c1-init",
        "wind_low": 0.0,
        "wind_high": 2.0,
        "added_factor": "init",
        "hole_ratio": 8.0,
        "rope_segments": 4,
        "init_xy_m": 0.016,
    },
    {
        "id": "C2",
        "label": "C2 + 10-segment rope",
        "profile": "ctrl-c2-rope10",
        "wind_low": 0.0,
        "wind_high": 2.0,
        "added_factor": "rope10",
        "hole_ratio": 8.0,
        "rope_segments": 10,
        "init_xy_m": 0.016,
    },
    {
        "id": "C3",
        "label": "C3 + high wind",
        "profile": "ctrl-c4-wind8",
        "wind_low": 6.0,
        "wind_high": 10.0,
        "added_factor": "wind6to10",
        "hole_ratio": 8.0,
        "rope_segments": 10,
        "init_xy_m": 0.016,
    },
    {
        "id": "C4",
        "label": "C4 + ratio 4 precision",
        "profile": "ctrl-c5-ratio4-w8",
        "wind_low": 6.0,
        "wind_high": 10.0,
        "added_factor": "ratio4",
        "hole_ratio": 4.0,
        "rope_segments": 10,
        "init_xy_m": 0.016,
    },
]

MAIN_VARIANTS = ["PID", "DampedPD", "MPC", "Full8D"]
ABLATION_VARIANTS = ["Full8D", "NoCable", "Full16D", "MLP8D"]
FORMAL_VARIANTS = set(MAIN_VARIANTS) | set(ABLATION_VARIANTS)
DISPLAY = {
    "PID": "PID",
    "DampedPD": "Damped-PD",
    "MPC": "MPC",
    "Full8D": "Residual RL Full-8D",
    "NoCable": "w/o Cable",
    "Full16D": "Full-16D",
    "MLP8D": "w/o LSTM (MLP, trainable 8D)",
}

def _base_args(level, variant, episodes):
    return SimpleNamespace(
        phase="descent",
        algo="ppo",
        ckpt=CKPTS.get(variant),
        episodes=int(episodes),
        render=False,
        wait_for_space=False,
        gpu=0,
        task_profile=level["profile"],
        wind_speed=0.0,
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
        pipeline_insert_target_z=0.10,
        pipeline_insert_depth_min=0.025,
        pipeline_cruise_z=0.25,
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
        cable_encoder_output_dim=None,
        trainable_cable_encoder=False,
        frozen_cable_encoder=False,
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
        traditional_expert_overrides_json=TRADITIONAL_OVERRIDES,
        eval_curriculum_level="max",
        summary_out=None,
        quiet=True,
    )


def _apply_variant_args(args, variant):
    if variant == "PID":
        args.algo = "expert"
        args.ckpt = None
        args.descent_base_expert = "traditional_pid"
    elif variant == "DampedPD":
        args.algo = "expert"
        args.ckpt = None
        args.descent_base_expert = "damped_pd"
    elif variant == "MPC":
        args.algo = "expert"
        args.ckpt = None
        args.descent_base_expert = "mpc"
    elif variant == "Full8D":
        args.cable_encoder_output_dim = 8
        args.trainable_cable_encoder = True
    elif variant == "Full16D":
        args.cable_encoder_output_dim = 16
        args.trainable_cable_encoder = True
    elif variant == "NoCable":
        args.disable_cable_obs = True
    elif variant == "MLP8D":
        args.cable_encoder_output_dim = 8
        args.trainable_cable_encoder = True
    else:
        raise ValueError(f"unknown variant: {variant}")
    return args


def _apply_variant_config(config, variant):
    if variant == "MLP8D":
        config.setdefault("ppo", {})["use_lstm"] = False
    return config


def _schedule_path(out_dir, level_id, episodes, seed):
    return out_dir / "schedules" / f"{level_id}_n{episodes}_seed{seed}.npz"


def get_or_create_schedule(out_dir, level, episodes, seed):
    path = _schedule_path(out_dir, level["id"], episodes, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = np.load(path)
        return data["wind_speeds"], data["wind_dirs"], data["reset_seeds"], path
    rng = np.random.default_rng(int(seed))
    lo = float(level["wind_low"])
    hi = float(level["wind_high"])
    if hi <= lo:
        wind_speeds = np.full(int(episodes), lo, dtype=np.float64)
    else:
        wind_speeds = rng.uniform(lo, hi, int(episodes))
    wind_dirs = rng.uniform(0.0, 2.0 * np.pi, int(episodes))
    reset_seeds = rng.integers(
        0, np.iinfo(np.int32).max, int(episodes), dtype=np.int64)
    np.savez(path, wind_speeds=wind_speeds, wind_dirs=wind_dirs,
             reset_seeds=reset_seeds)
    return wind_speeds, wind_dirs, reset_seeds, path


def precheck(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Formal Interval Ladder Precheck",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Required L8-Best Checkpoints",
        "",
    ]
    ok = True
    for name, rel in CKPTS.items():
        path = REPO_ROOT / rel
        exists = path.exists()
        ok = ok and exists
        lines.append(f"- {name}: `{'OK' if exists else 'MISSING'}` `{rel}`")
    overrides_ok = (REPO_ROOT / TRADITIONAL_OVERRIDES).exists()
    ok = ok and overrides_ok
    lines.append("")
    lines.append(
        f"- Traditional overrides: `{'OK' if overrides_ok else 'MISSING'}` "
        f"`{TRADITIONAL_OVERRIDES}`")
    lines.append("")
    lines.append("## Skipped Formal Candidates")
    lines.append("")
    for name, reason in MISSING_OR_NOT_FORMAL.items():
        lines.append(f"- {name}: {reason}")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Wind is sampled uniformly per episode from the level interval.")
    lines.append("- All variants in one level reuse the same wind, wind-direction, and reset-seed schedule.")
    lines.append("- Main comparison: PID, Damped-PD, MPC, Full8D Residual RL.")
    lines.append("- High-wind ablation: Full8D, NoCable, Full16D, matched MLP8D on C4 only.")
    lines.append("- The formal runner is single-environment and serial to avoid shared MuJoCo XML races.")
    report = out_dir / "PRECHECK_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not ok:
        raise FileNotFoundError(f"precheck failed; see {report}")
    return report


def flatten_row(section, level, variant, summary, ckpt, schedule_path,
                wind_speeds):
    term = summary.get("termination_counts", {})
    if not isinstance(term, dict):
        term = {}
    row = {
        "section": section,
        "level_id": level["id"],
        "level_label": level["label"],
        "task_profile": level["profile"],
        "added_factor": level["added_factor"],
        "hole_ratio": level["hole_ratio"],
        "rope_segments_design": level["rope_segments"],
        "init_xy_m_design": level["init_xy_m"],
        "wind_low": level["wind_low"],
        "wind_high": level["wind_high"],
        "wind_mean_sampled": float(np.mean(wind_speeds)),
        "wind_std_sampled": float(np.std(wind_speeds)),
        "variant": variant,
        "method": DISPLAY.get(variant, variant),
        "checkpoint": ckpt or "",
        "schedule": str(schedule_path),
        "episodes": summary.get("episodes", summary.get("episodes_requested", "")),
        "broad_success_rate": summary.get("broad_success_rate", summary.get("success_rate", 0.0)),
        "strict_success_rate": summary.get("strict_success_rate", summary.get("success_rate", 0.0)),
        "lucky_insert_rate": summary.get("lucky_insert_rate", 0.0),
        "avg_reward": summary.get("avg_reward", ""),
        "avg_steps": summary.get("avg_steps", ""),
        "avg_ke_mJ": summary.get("avg_ke_mJ", ""),
        "p95_ke_mJ": summary.get("p95_ke_mJ", ""),
        "max_ke_mJ": summary.get("max_ke_mJ", ""),
        "avg_angle": summary.get("avg_angle", ""),
        "p95_angle": summary.get("p95_angle", ""),
        "max_angle": summary.get("max_angle", ""),
        "avg_ee_acc": summary.get("avg_ee_acc", ""),
        "max_ee_acc": summary.get("max_ee_acc", ""),
        "termination_counts": json.dumps(term, ensure_ascii=False, sort_keys=True),
    }
    return row


def write_summary_csv(out_dir, rows):
    path = out_dir / "formal_interval_summary.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_summary_md(out_dir, rows):
    def pct(v):
        try:
            return f"{100.0 * float(v):.1f}"
        except Exception:
            return "NA"

    by_key = {(r["section"], r["level_id"], r["variant"]): r for r in rows}
    lines = ["# Formal Interval Ladder Summary", ""]
    lines.append("Cell format: broad success / strict success (%).")
    lines.append("")
    lines.append("## Main Controlled Comparison")
    lines.append("")
    lines.append("| Level | Wind interval | " + " | ".join(DISPLAY[v] for v in MAIN_VARIANTS) + " |")
    lines.append("|---|---:|" + "|".join(["---:"] * len(MAIN_VARIANTS)) + "|")
    for level in LEVELS:
        cells = []
        for variant in MAIN_VARIANTS:
            r = by_key.get(("main", level["id"], variant))
            cells.append("pending" if r is None else f"{pct(r['broad_success_rate'])} / {pct(r['strict_success_rate'])}")
        lines.append(f"| {level['label']} | {level['wind_low']:.0f}-{level['wind_high']:.0f} m/s | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## High-Wind Highest-Difficulty Ablation")
    lines.append("")
    lines.append("| Variant | Broad / strict success (%) | Avg KE (mJ) | Avg angle (deg) |")
    lines.append("|---|---:|---:|---:|")
    for variant in ABLATION_VARIANTS:
        r = by_key.get(("ablation", "C4", variant))
        if r is None:
            lines.append(f"| {DISPLAY[variant]} | pending | pending | pending |")
        else:
            lines.append(
                f"| {DISPLAY[variant]} | {pct(r['broad_success_rate'])} / {pct(r['strict_success_rate'])} "
                f"| {float(r['avg_ke_mJ']):.1f} | {float(r['avg_angle']):.2f} |")
    path = out_dir / "formal_interval_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_one(out_dir, section, level, variant, episodes, seed, resume=True):
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{section}__{level['id']}__{variant}.json"
    if resume and json_path.exists() and json_path.stat().st_size > 0:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        row = data.get("_flat_row")
        if row:
            print(f"[SKIP] {section}/{level['id']}/{variant}: {json_path}", flush=True)
            return row

    wind_speeds, wind_dirs, reset_seeds, schedule_path = get_or_create_schedule(
        out_dir, level, episodes, seed)
    args = _apply_variant_args(_base_args(level, variant, episodes), variant)
    config = build_config(args)
    config = _apply_variant_config(config, variant)
    config.setdefault("sim", {})["render"] = False
    config.setdefault("wind", {})["speed_max"] = 10.0
    config.setdefault("wind_obs", {})["wind_speed_max"] = 10.0
    eval_init = get_descent_eval_init(config, "max")

    np.random.seed(int(seed))
    try:
        torch.manual_seed(int(seed))
    except Exception:
        pass

    t0 = time.perf_counter()
    print(
        f"\n[RUN] {section}/{level['id']}/{variant} "
        f"episodes={episodes} wind={level['wind_low']:.1f}-{level['wind_high']:.1f} "
        f"profile={level['profile']}",
        flush=True,
    )
    env = CableRobotEnvWithObstacles(config=config)
    try:
        if hasattr(env, "set_curriculum_n_obstacles"):
            env.set_curriculum_n_obstacles(0)
        expert = make_phase_base_expert("descent", config, env.ik_solver)
        ee_ctrl = EEAccController(config, env.ik_solver)
        agent = None
        ckpt = CKPTS.get(variant)
        if args.algo != "expert":
            agent = load_agent("descent", "ppo", ckpt, config)
        results = test_single_phase(
            env,
            agent,
            expert,
            ee_ctrl,
            "descent",
            config,
            n_episodes=int(episodes),
            deterministic=True,
            wind_speed_sampler=lambda k, ws=wind_speeds: ws[k],
            wind_dir_sampler=lambda k, wd=wind_dirs: wd[k],
            reset_seed_sampler=lambda k, rs=reset_seeds: rs[k],
            obs_noise=0.0,
            act_noise=0.0,
            force_noise=0.0,
            eval_cur_init=eval_init,
            verbose=False,
            progress_every=32,
            progress_prefix=f"  {section}/{level['id']}/{variant}",
        )
    finally:
        env.close()

    summary = summarize_results(results)
    row = flatten_row(section, level, variant, summary, ckpt, schedule_path,
                      wind_speeds)
    payload = dict(summary)
    payload.update({
        "section": section,
        "level": level,
        "variant": variant,
        "method": DISPLAY.get(variant, variant),
        "checkpoint": ckpt or "",
        "schedule": str(schedule_path),
        "wind_speed_low": float(level["wind_low"]),
        "wind_speed_high": float(level["wind_high"]),
        "wind_speed_mean": float(np.mean(wind_speeds)),
        "wind_speed_std": float(np.std(wind_speeds)),
        "elapsed_s": float(time.perf_counter() - t0),
        "_flat_row": row,
    })
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

    print(
        f"[DONE] {section}/{level['id']}/{variant}: "
        f"bSR={100*float(row['broad_success_rate']):.1f}% "
        f"strict={100*float(row['strict_success_rate']):.1f}% "
        f"avgKE={row['avg_ke_mJ']}",
        flush=True,
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=270705)
    parser.add_argument("--section", choices=["all", "main", "ablation"],
                        default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-filter", default="")
    parser.add_argument("--variant-filter", default="")
    args = parser.parse_args()

    out_dir = Path(args.out)
    report = precheck(out_dir)
    print(f"[precheck] {report}", flush=True)

    profile_filter = {x.strip() for x in args.profile_filter.split(",") if x.strip()}
    variant_filter = {x.strip() for x in args.variant_filter.split(",") if x.strip()}

    rows = []
    existing = sorted((out_dir / "json").glob("*.json")) if (out_dir / "json").exists() else []
    for path in existing:
        try:
            with path.open("r", encoding="utf-8") as f:
                flat = json.load(f).get("_flat_row")
            if flat and flat.get("variant") in FORMAL_VARIANTS:
                rows.append(flat)
        except Exception:
            pass

    if args.section in ("all", "main"):
        for i, level in enumerate(LEVELS):
            if profile_filter and level["id"] not in profile_filter and level["profile"] not in profile_filter:
                continue
            level_seed = int(args.seed) + 1009 * i
            for variant in MAIN_VARIANTS:
                if variant_filter and variant not in variant_filter:
                    continue
                row = run_one(out_dir, "main", level, variant, int(args.episodes),
                              level_seed, resume=args.resume)
                rows = [r for r in rows if not (
                    r["section"] == row["section"] and
                    r["level_id"] == row["level_id"] and
                    r["variant"] == row["variant"])]
                rows.append(row)
                write_summary_csv(out_dir, rows)
                write_summary_md(out_dir, rows)

    if args.section in ("all", "ablation"):
        level = LEVELS[-1]
        if not profile_filter or level["id"] in profile_filter or level["profile"] in profile_filter:
            level_seed = int(args.seed) + 1009 * 100
            for variant in ABLATION_VARIANTS:
                if variant_filter and variant not in variant_filter:
                    continue
                row = run_one(out_dir, "ablation", level, variant,
                              int(args.episodes), level_seed, resume=args.resume)
                rows = [r for r in rows if not (
                    r["section"] == row["section"] and
                    r["level_id"] == row["level_id"] and
                    r["variant"] == row["variant"])]
                rows.append(row)
                write_summary_csv(out_dir, rows)
                write_summary_md(out_dir, rows)

    csv_path = write_summary_csv(out_dir, rows)
    md_path = write_summary_md(out_dir, rows)
    print(f"[DONE] summary csv: {csv_path}", flush=True)
    print(f"[DONE] summary md:  {md_path}", flush=True)


if __name__ == "__main__":
    main()
