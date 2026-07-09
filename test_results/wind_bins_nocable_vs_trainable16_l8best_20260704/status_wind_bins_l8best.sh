#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

OUT="test_results/wind_bins_nocable_vs_trainable16_l8best_20260704"
PID_FILE="$OUT/wind_bins_l8best.pid"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  echo "PID: $PID"
  ps -p "$PID" -o pid,ppid,stat,etime,cmd || true
else
  echo "PID file missing: $PID_FILE"
fi

echo
echo "--- output files ---"
ls -lh "$OUT"/*.csv "$OUT"/*.json "$OUT"/*.log 2>/dev/null || true

echo
echo "--- current benchmark tail ---"
tail -n 80 "$OUT/wind_bins_l8best.stdout.log" 2>/dev/null || true

echo
echo "--- partial combined summary ---"
python "$OUT/summarize_wind_bins.py" 2>/dev/null || true
if [[ -f "$OUT/combined_wind_bins_process_metrics_readable.csv" ]]; then
  python - "$OUT/combined_wind_bins_process_metrics_readable.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    print(
        f"{r['model']:>15s} wind={r['wind_bin_mps']:>5s} "
        f"mainSR={r['main_success_broad_pct']:>5s}% "
        f"strict={r['perfect_success_strict_pct']:>5s}% "
        f"avgKE={r['avg_swing_ke_mJ']} p95KE={r['p95_swing_ke_mJ']} "
        f"avgAngle={r['avg_swing_angle_rad']} p95Angle={r['p95_swing_angle_rad']} "
        f"avgAcc={r['avg_ee_acc_mps2']} terms={r['termination_counts']}"
    )
PY
else
  echo "combined readable CSV not available yet"
fi
