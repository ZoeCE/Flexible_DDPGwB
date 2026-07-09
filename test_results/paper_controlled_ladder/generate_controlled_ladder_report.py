#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROFILES = [
    ("ctrl-c0-base", "C0 base", "shared", "base", 8.0, 4, 0, 0.0),
    ("ctrl-c1-init", "C1 init", "shared", "init", 8.0, 4, 0, 0.0),
    ("ctrl-c2-rope10", "C2 rope10", "shared", "rope 10", 8.0, 10, 0, 0.016),
    ("ctrl-c3-wind2", "C3 wind2", "shared", "wind 2", 8.0, 10, 2, 0.016),
    ("ctrl-c4-wind6", "C4 wind6", "w6", "wind 6", 8.0, 10, 6, 0.016),
    ("ctrl-c4-wind8", "C4 wind8", "w8", "wind 8", 8.0, 10, 8, 0.016),
    ("ctrl-c5-ratio4-w6", "C5 ratio4 w6", "w6", "ratio 4", 4.0, 10, 6, 0.016),
    ("ctrl-c5-ratio4-w8", "C5 ratio4 w8", "w8", "ratio 4", 4.0, 10, 8, 0.016),
]
PROFILE_META = {p[0]: p for p in PROFILES}
VARIANTS = ["PID", "DampedPD", "MPC", "Mainline", "NoCable", "MainlineP2"]
DISPLAY = {
    "PID": "PID",
    "DampedPD": "Damped-PD",
    "MPC": "MPC",
    "Mainline": "Mainline",
    "NoCable": "NoCable",
    "MainlineP2": "Mainline P2",
}


def pct(x):
    try:
        return 100.0 * float(x)
    except Exception:
        return math.nan


def read_rows(out: Path):
    rows = []
    json_dir = out / "json"
    for profile, *_ in PROFILES:
        for variant in VARIANTS:
            path = json_dir / f"{profile}__{variant}.json"
            if not path.exists():
                rows.append({
                    "profile": profile,
                    "variant": variant,
                    "missing": True,
                })
                continue
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            key, label, branch, added, ratio, rope, wind, init_xy = PROFILE_META[profile]
            term = data.get("termination_counts", {})
            if not isinstance(term, dict):
                term = {}
            rows.append({
                "profile": key,
                "profile_label": label,
                "branch": branch,
                "added_factor": added,
                "hole_ratio": ratio,
                "rope_segments_design": rope,
                "wind_mps_design": wind,
                "init_xy_m_design": init_xy,
                "variant": variant,
                "method": DISPLAY.get(variant, variant),
                "missing": False,
                "episodes": data.get("episodes", data.get("episodes_requested", "")),
                "broad_success_pct": pct(data.get("broad_success_rate", 0.0)),
                "strict_success_pct": pct(data.get("strict_success_rate", data.get("success_rate", 0.0))),
                "lucky_insert_pct": pct(data.get("lucky_insert_rate", 0.0)),
                "avg_reward": data.get("avg_reward", ""),
                "avg_steps": data.get("avg_steps", ""),
                "avg_ke_mJ": data.get("avg_ke_mJ", ""),
                "p95_ke_mJ": data.get("p95_ke_mJ", ""),
                "avg_angle": data.get("avg_angle", ""),
                "p95_angle": data.get("p95_angle", ""),
                "termination_counts": json.dumps(term, sort_keys=True),
                "json_path": str(path),
            })
    return rows


def write_csv(rows, out: Path):
    csv_path = out / "controlled_ladder_summary.csv"
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def write_markdown(rows, out: Path):
    by_key = {(r["profile"], r["variant"]): r for r in rows if not r.get("missing")}
    md = []
    md.append("# Controlled-Variable Ladder Summary")
    md.append("")
    md.append("Cell format: broad success / strict success (%). Broad includes lucky insertions.")
    md.append("")
    md.append("| Profile | Added factor | Wind | Rope | Ratio | " + " | ".join(DISPLAY[v] for v in VARIANTS) + " |")
    md.append("|---|---|---:|---:|---:|" + "|".join(["---:"] * len(VARIANTS)) + "|")
    for profile, label, _branch, added, ratio, rope, wind, _init in PROFILES:
        cells = []
        for variant in VARIANTS:
            r = by_key.get((profile, variant))
            if not r:
                cells.append("pending")
            else:
                cells.append(f"{r['broad_success_pct']:.1f} / {r['strict_success_pct']:.1f}")
        md.append(f"| `{profile}` | {added} | {wind} | {rope} | {ratio:.1f} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("Generated files:")
    md.append("")
    md.append("- `controlled_ladder_summary.csv`")
    md.append("- `controlled_ladder_success.png`")
    md.append("- `controlled_ladder_heatmap_broad.png`")
    md.append("- `controlled_ladder_failure_composition.png`")
    path = out / "controlled_ladder_summary.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def plot(rows, out: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip plots: {exc}")
        return []

    good = [r for r in rows if not r.get("missing")]
    by_key = {(r["profile"], r["variant"]): r for r in good}
    colors = {
        "PID": "#7f8fa6",
        "DampedPD": "#4c78a8",
        "MPC": "#f58518",
        "Mainline": "#54a24b",
        "NoCable": "#b279a2",
        "MainlineP2": "#72b7b2",
    }
    saved = []

    branches = [
        ("6 m/s branch", ["ctrl-c0-base", "ctrl-c1-init", "ctrl-c2-rope10", "ctrl-c3-wind2", "ctrl-c4-wind6", "ctrl-c5-ratio4-w6"]),
        ("8 m/s branch", ["ctrl-c0-base", "ctrl-c1-init", "ctrl-c2-rope10", "ctrl-c3-wind2", "ctrl-c4-wind8", "ctrl-c5-ratio4-w8"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, (title, profs) in zip(axes, branches):
        x = np.arange(len(profs))
        labels = [PROFILE_META[p][1].replace(" ", "\n") for p in profs]
        for variant in VARIANTS:
            y = [by_key.get((p, variant), {}).get("broad_success_pct", np.nan) for p in profs]
            ys = [by_key.get((p, variant), {}).get("strict_success_pct", np.nan) for p in profs]
            ax.plot(x, y, marker="o", linewidth=2.0, label=DISPLAY[variant], color=colors[variant])
            ax.plot(x, ys, marker=".", linewidth=1.0, linestyle="--", color=colors[variant], alpha=0.45)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(-2, 102)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("Success rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.suptitle("Controlled-Variable Descent Ladder: Broad Solid, Strict Dashed", y=1.04)
    fig.tight_layout()
    path = out / "controlled_ladder_success.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    matrix = np.full((len(VARIANTS), len(PROFILES)), np.nan, dtype=float)
    for i, variant in enumerate(VARIANTS):
        for j, (profile, *_rest) in enumerate(PROFILES):
            matrix[i, j] = by_key.get((profile, variant), {}).get("broad_success_pct", np.nan)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_yticks(np.arange(len(VARIANTS)))
    ax.set_yticklabels([DISPLAY[v] for v in VARIANTS])
    ax.set_xticks(np.arange(len(PROFILES)))
    ax.set_xticklabels([p[1].replace(" ", "\n") for p in PROFILES], fontsize=8)
    for i in range(len(VARIANTS)):
        for j in range(len(PROFILES)):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Broad success (%)")
    ax.set_title("Broad Success Heatmap")
    fig.tight_layout()
    path = out / "controlled_ladder_heatmap_broad.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)

    final_profiles = ["ctrl-c5-ratio4-w6", "ctrl-c5-ratio4-w8"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, profile in zip(axes, final_profiles):
        bottoms = np.zeros(len(VARIANTS))
        categories = ["success", "lucky_rebar_insert_failure", "timeout", "early_stop", "instability", "stuck_on_rebar", "other"]
        cat_colors = {
            "success": "#54a24b",
            "lucky_rebar_insert_failure": "#eeca3b",
            "timeout": "#9aa6b8",
            "early_stop": "#ff9da6",
            "instability": "#e45756",
            "stuck_on_rebar": "#b279a2",
            "other": "#bab0ac",
        }
        vals_by_cat = {cat: [] for cat in categories}
        for variant in VARIANTS:
            r = by_key.get((profile, variant))
            if not r:
                for cat in categories:
                    vals_by_cat[cat].append(0.0)
                continue
            episodes = float(r.get("episodes") or 1.0)
            try:
                terms = json.loads(r.get("termination_counts", "{}") or "{}")
            except Exception:
                terms = {}
            success_pct = float(r.get("strict_success_pct", 0.0))
            lucky_pct = float(r.get("lucky_insert_pct", 0.0))
            known = success_pct + lucky_pct
            vals_by_cat["success"].append(success_pct)
            vals_by_cat["lucky_rebar_insert_failure"].append(lucky_pct)
            for cat in ["timeout", "early_stop", "instability", "stuck_on_rebar"]:
                vals_by_cat[cat].append(100.0 * float(terms.get(cat, 0)) / max(episodes, 1.0))
                known += vals_by_cat[cat][-1]
            vals_by_cat["other"].append(max(0.0, 100.0 - known))
        x = np.arange(len(VARIANTS))
        for cat in categories:
            vals = np.asarray(vals_by_cat[cat], dtype=float)
            ax.bar(x, vals, bottom=bottoms, label=cat, color=cat_colors[cat])
            bottoms += vals
        ax.set_title(PROFILE_META[profile][1])
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[v] for v in VARIANTS], rotation=25, ha="right")
        ax.set_ylim(0, 100)
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_ylabel("Episode composition (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Final-Condition Outcome Composition", y=1.04)
    fig.tight_layout()
    path = out / "controlled_ladder_failure_composition.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    saved.append(path)
    plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    rows = read_rows(out)
    csv_path = write_csv(rows, out)
    md_path = write_markdown(rows, out)
    plot_paths = plot(rows, out)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD: {md_path}")
    for path in plot_paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
