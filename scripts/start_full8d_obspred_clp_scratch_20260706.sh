#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

PID_FILE=test_results/full8d_obspred_clp_scratch_20260706.pid
LOG=test_results/full8d_obspred_clp_scratch_20260706.master.log

mkdir -p test_results
nohup bash scripts/run_full8d_obspred_clp_scratch_20260706.sh > "$LOG" 2>&1 &
echo $! > "$PID_FILE"

echo "Started Full8D scratch ObsPred+CableLatPred chain PID $(cat "$PID_FILE")"
echo "Master log: $LOG"
echo "Status: bash scripts/status_full8d_obspred_clp_scratch_20260706.sh"

