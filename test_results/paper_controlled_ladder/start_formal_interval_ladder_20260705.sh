#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

EPISODES="${EPISODES:-512}"
RUN_ID="${RUN_ID:-20260705_interval_wind_n${EPISODES}}"
OUT="${OUT:-test_results/paper_controlled_ladder/formal_interval_${RUN_ID}}"
SECTION="${SECTION:-all}"
mkdir -p "$OUT"

cat > "$OUT/controller.env" <<EOF
REPO=$REPO
EPISODES=$EPISODES
RUN_ID=$RUN_ID
OUT=$OUT
SECTION=$SECTION
EOF

nohup bash test_results/paper_controlled_ladder/launch_formal_interval_ladder_20260705.sh \
  > "$OUT/controller.log" 2>&1 &
echo $! > "$OUT/controller.pid"

cat > test_results/paper_controlled_ladder/latest_formal_interval_run.env <<EOF
OUT=$OUT
PID=$(cat "$OUT/controller.pid")
EOF

echo "[STARTED] PID=$(cat "$OUT/controller.pid") OUT=$OUT"

