#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

OUT="test_results/pid_base_ablation"
LOG="$OUT/distill_dagger_terminal_launcher.log"
mkdir -p "$OUT"

exec bash test_results/pid_base_ablation/run_distill_trueobs_no_pid.sh > "$LOG" 2>&1
