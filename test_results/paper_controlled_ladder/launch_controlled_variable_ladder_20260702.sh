#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

EPISODES="${EPISODES:-256}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-test_results/paper_controlled_ladder/controlled_variable_${RUN_ID}_n${EPISODES}}"
mkdir -p "$OUT"

export REPO EPISODES RUN_ID OUT
nohup bash test_results/paper_controlled_ladder/run_controlled_variable_ladder_20260702.sh \
  > "$OUT/controller.log" 2>&1 &
pid=$!
echo "$pid" > "$OUT/controller.pid"

cat > test_results/paper_controlled_ladder/latest_run.env <<EOF
RUN_ID=$RUN_ID
EPISODES=$EPISODES
OUT=$OUT
PID=$pid
EOF

echo "Started controlled-variable ladder."
echo "  PID: $pid"
echo "  OUT: $OUT"
echo
echo "Observe with:"
echo "  bash test_results/paper_controlled_ladder/status_controlled_variable_ladder_20260702.sh"

