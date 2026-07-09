#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

ENV_FILE="test_results/paper_controlled_ladder/latest_formal_interval_run.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
OUT="${OUT:-$(ls -dt test_results/paper_controlled_ladder/formal_interval_* 2>/dev/null | head -n 1 || true)}"
if [[ -z "${OUT:-}" || ! -d "$OUT" ]]; then
  echo "No formal interval run directory found."
  exit 1
fi

pid_file="$OUT/controller.pid"
pid="${PID:-}"
if [[ -z "$pid" && -f "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
fi

echo "OUT: $OUT"
if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
  echo "PID: $pid (running)"
else
  echo "PID: ${pid:-unknown} (not running)"
fi

done_json="$(find "$OUT/json" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
echo "Completed JSON summaries: $done_json/24"

if [[ -f "$OUT/formal_interval_summary.md" ]]; then
  echo
  echo "--- summary ---"
  cat "$OUT/formal_interval_summary.md"
fi

if [[ -f "$OUT/controller.log" ]]; then
  echo
  echo "--- controller.log tail ---"
  tail -n 80 "$OUT/controller.log"
fi
