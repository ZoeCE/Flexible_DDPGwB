#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

ENV_FILE="test_results/paper_controlled_ladder/latest_run.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
else
  OUT="${OUT:-}"
fi

if [[ -z "${OUT:-}" ]]; then
  OUT="$(ls -dt test_results/paper_controlled_ladder/controlled_variable_* 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${OUT:-}" || ! -d "$OUT" ]]; then
  echo "No controlled-variable run directory found."
  exit 1
fi

pid_file="$OUT/controller.pid"
pid=""
if [[ -f "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
fi

echo "OUT: $OUT"
if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "PID: $pid (running)"
else
  echo "PID: ${pid:-unknown} (not running)"
fi

total=48
done_json="$(find "$OUT/json" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "Completed JSON summaries: $done_json/$total"

if [[ -f "$OUT/controller.log" ]]; then
  echo
  echo "--- controller.log tail ---"
  tail -n 50 "$OUT/controller.log"
fi

if [[ "${1:-}" == "--report" ]]; then
  if [[ -f /home/xyzha/miniconda3/etc/profile.d/conda.sh ]]; then
    # shellcheck disable=SC1091
    source /home/xyzha/miniconda3/etc/profile.d/conda.sh
    conda activate vsdrl_env_5060
  fi
  python test_results/paper_controlled_ladder/generate_controlled_ladder_report.py --out "$OUT"
fi
