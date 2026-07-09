#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

OUT="test_results/wind_bins_nocable_vs_trainable16_l8best_20260704"
mkdir -p "$OUT"

nohup bash "$OUT/run_wind_bins_l8best.sh" >/dev/null 2>&1 &
echo $! > "$OUT/wind_bins_l8best.pid"
echo "Started wind-bin L8-best benchmark PID=$(cat "$OUT/wind_bins_l8best.pid")"
