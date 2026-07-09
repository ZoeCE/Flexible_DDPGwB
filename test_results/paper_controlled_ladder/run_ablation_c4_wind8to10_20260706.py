#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_results.paper_controlled_ladder import run_formal_interval_ladder_20260705 as formal

ABLATION_WIND8TO10_VARIANTS = ["Full8D", "NoCable", "MLP8D"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=270706)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    level = dict(formal.LEVELS[-1])
    level.update({
        "id": "C4W8",
        "label": "C4 + ratio 4 precision, wind 8-10",
        "wind_low": 8.0,
        "wind_high": 10.0,
        "added_factor": "ratio4_wind8to10",
    })

    report = formal.precheck(out_dir)
    print(f"[precheck] {report}", flush=True)

    rows = []
    existing = sorted((out_dir / "json").glob("*.json")) if (out_dir / "json").exists() else []
    for path in existing:
        try:
            with path.open("r", encoding="utf-8") as f:
                flat = json.load(f).get("_flat_row")
            if flat and flat.get("variant") in ABLATION_WIND8TO10_VARIANTS:
                rows.append(flat)
        except Exception:
            pass

    for variant in ABLATION_WIND8TO10_VARIANTS:
        row = formal.run_one(
            out_dir,
            "ablation_wind8to10",
            level,
            variant,
            int(args.episodes),
            int(args.seed),
            resume=args.resume,
        )
        rows = [r for r in rows if not (
            r["section"] == row["section"] and
            r["level_id"] == row["level_id"] and
            r["variant"] == row["variant"])]
        rows.append(row)
        formal.write_summary_csv(out_dir, rows)
        formal.write_summary_md(out_dir, rows)

    csv_path = formal.write_summary_csv(out_dir, rows)
    md_path = formal.write_summary_md(out_dir, rows)
    print(f"[DONE] summary csv: {csv_path}", flush=True)
    print(f"[DONE] summary md:  {md_path}", flush=True)


if __name__ == "__main__":
    main()
