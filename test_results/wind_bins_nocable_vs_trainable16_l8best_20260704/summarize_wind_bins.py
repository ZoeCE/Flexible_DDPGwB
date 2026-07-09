#!/usr/bin/env python3
import csv
import json
from pathlib import Path


OUT = Path("test_results/wind_bins_nocable_vs_trainable16_l8best_20260704")
SOURCES = [
    ("NoCable", OUT / "nocable_l8best_wind_bins.csv"),
    ("FullTrainable16", OUT / "full_trainable16_l8best_wind_bins.csv"),
]


KEEP = [
    "model",
    "wind_bin",
    "wind_speed_mean",
    "episodes",
    "broad_success_rate",
    "strict_success_rate",
    "success_rate",
    "lucky_insert_rate",
    "avg_reward",
    "avg_steps",
    "avg_ke_mJ",
    "p95_ke_mJ",
    "max_ke_mJ",
    "avg_angle",
    "p95_angle",
    "max_angle",
    "avg_acc",
    "max_acc",
    "pl_vel_rms",
    "pl_vel_peak",
    "cable_ke_avg",
    "cable_ke_peak",
    "terminal_dtf_m_mean",
    "terminal_dtf_m_p95",
    "terminal_tilt_rad_mean",
    "terminal_yaw_rad_mean",
    "terminal_ok_xy_rate",
    "terminal_ok_tilt_rate",
    "terminal_ok_yaw_rate",
    "termination_counts",
]


def f(row, key, default=""):
    return row.get(key, default)


def pct(v):
    try:
        return f"{100.0 * float(v):.1f}"
    except Exception:
        return ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    pretty_rows = []
    for model, path in SOURCES:
        if not path.exists():
            print(f"missing: {path}")
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                lo = float(row["wind_speed_low"])
                hi = float(row["wind_speed_high"])
                merged = {"model": model, "wind_bin": f"{lo:g}-{hi:g}"}
                merged.update(row)
                rows.append(merged)
                pretty_rows.append({
                    "model": model,
                    "wind_bin_mps": f"{lo:g}-{hi:g}",
                    "episodes": f(row, "episodes"),
                    "main_success_broad_pct": pct(f(row, "broad_success_rate")),
                    "perfect_success_strict_pct": pct(f(row, "strict_success_rate") or f(row, "success_rate")),
                    "lucky_insert_pct": pct(f(row, "lucky_insert_rate")),
                    "avg_steps": f(row, "avg_steps"),
                    "avg_reward": f(row, "avg_reward"),
                    "avg_swing_ke_mJ": f(row, "avg_ke_mJ"),
                    "p95_swing_ke_mJ": f(row, "p95_ke_mJ"),
                    "avg_swing_angle_deg": f(row, "avg_angle"),
                    "p95_swing_angle_deg": f(row, "p95_angle"),
                    "avg_ee_acc_mps2": f(row, "avg_acc"),
                    "max_ee_acc_mps2": f(row, "max_acc"),
                    "payload_vel_rms_mps": f(row, "pl_vel_rms"),
                    "termination_counts": f(row, "termination_counts"),
                })

    if rows:
        combined = OUT / "combined_wind_bins_process_metrics.csv"
        with combined.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=KEEP, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        pretty = OUT / "combined_wind_bins_process_metrics_readable.csv"
        with pretty.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(pretty_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pretty_rows)
        summary_json = OUT / "combined_wind_bins_process_metrics.json"
        summary_json.write_text(json.dumps(pretty_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote: {combined}")
        print(f"wrote: {pretty}")
        print(f"wrote: {summary_json}")
    else:
        print("no rows written yet")


if __name__ == "__main__":
    main()
