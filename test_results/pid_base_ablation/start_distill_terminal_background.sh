#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

OUT="test_results/pid_base_ablation"
LOG="$OUT/distill_dagger_terminal_launcher.log"
PID_FILE="$OUT/distill_dagger_terminal.pid"
mkdir -p "$OUT"

(
  cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
  exec bash test_results/pid_base_ablation/run_distill_trueobs_no_pid.sh
) > "$LOG" 2>&1 &

pid=$!
printf '%s\n' "$pid" > "$PID_FILE"
printf 'started pid=%s log=%s\n' "$pid" "$LOG"
