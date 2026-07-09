#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

DIR=saves/descent_ablation_full8d_trueobs_5hzctrl_5hzobs_pretrain_20260706
PID_FILE="$DIR/pid.txt"
LOG="$DIR/train.log"
CSV="$DIR/descent_ppo_vec_log.csv"
echo "== Full8D 5Hz full pretrain process =="
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "PID $pid: running"
  else
    echo "PID $pid: not running"
  fi
else
  echo "no pid file"
fi
ps -eo pid,ppid,stat,etime,cmd | grep -E 'descent_ablation_full8d_trueobs_5hzctrl_5hzobs_pretrain_20260706|train_phase.py' | grep -v grep || true

echo
echo "== checkpoints =="
find "$DIR" -maxdepth 1 -type f \( -name 'ckpt*.pt' -o -name '*_log.csv' -o -name 'resolved_config*.json' \) -printf '  %f\n' | sort || true

echo
echo "== CSV tail =="
if [[ -f "$CSV" ]]; then
  tail -n 5 "$CSV"
else
  echo "CSV not created yet"
fi

echo
echo "== log tail =="
if [[ -f "$LOG" ]]; then
  tail -n 50 "$LOG"
else
  echo "log not created yet"
fi
