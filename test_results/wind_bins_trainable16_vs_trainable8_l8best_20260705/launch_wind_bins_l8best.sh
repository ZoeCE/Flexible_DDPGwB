#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

OUT="test_results/wind_bins_trainable16_vs_trainable8_l8best_20260705"
mkdir -p "$OUT"

nohup bash "$OUT/run_wind_bins_l8best.sh" >/dev/null 2>&1 &
echo $! > "$OUT/wind_bins_l8best.pid"
echo "Started FullTrainable16 vs FullTrainable8 wind-bin benchmark PID=$(cat "$OUT/wind_bins_l8best.pid")"
