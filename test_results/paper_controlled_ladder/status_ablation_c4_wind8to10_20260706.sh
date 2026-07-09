#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

ENV_FILE="${ENV_FILE:-test_results/paper_controlled_ladder/latest_ablation_wind8to10.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "No run env: $ENV_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

echo "OUT: $OUT"
if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
  echo "PID: $PID (running)"
else
  echo "PID: ${PID:-NA} (not running)"
fi

CSV="$OUT/formal_interval_summary.csv"
if [[ -f "$CSV" ]]; then
  echo
  echo "--- completed rows ---"
  python - "$CSV" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline="")))
print(f"{len(rows)} rows")
for r in rows:
    print(
        f"{r['variant']:8s} "
        f"broad={100*float(r['broad_success_rate']):5.1f}% "
        f"strict={100*float(r['strict_success_rate']):5.1f}% "
        f"KE={float(r['avg_ke_mJ']):6.1f} "
        f"angle={float(r['avg_angle']):4.2f}"
    )
PY
fi

echo
echo "--- log tail ---"
tail -n 80 "$OUT/controller.log" 2>/dev/null || true
