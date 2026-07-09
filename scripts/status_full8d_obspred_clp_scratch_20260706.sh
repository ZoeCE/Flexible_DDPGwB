#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

PID_FILE=test_results/full8d_obspred_clp_scratch_20260706.pid
LOG=test_results/full8d_obspred_clp_scratch_20260706.master.log
INIT_DIR=saves/descent_ablation_full8d_scratch_init_seed270716_20260706
OBSP_DIR=saves/descent_ablation_obspred_full8d_scratchinit_pretrain_20260706
CLP_DIR=saves/descent_ablation_cable_latpred_full8d_scratchinit_pretrain_20260706
PPO_DIR=saves/descent_ablation_full8d_obspred_clp_p2_10hzctrl_5hzobs_ppo_20260706

echo "== process =="
if [[ -f "$PID_FILE" ]]; then
  pid=$(cat "$PID_FILE")
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "chain PID $pid: running"
  else
    echo "chain PID $pid: not running"
  fi
else
  echo "no pid file"
fi
ps -eo pid,ppid,stat,etime,cmd | grep -E 'run_full8d_obspred_clp_scratch|pretrain_cable_latent_predictor|descent_ablation_obspred_full8d_scratchinit|descent_ablation_full8d_obspred_clp_p2|train_phase.py' | grep -v grep || true

echo
echo "== checkpoints =="
for d in "$INIT_DIR" "$OBSP_DIR" "$CLP_DIR" "$PPO_DIR"; do
  echo "-- $d"
  find "$d" -maxdepth 1 -type f \( -name 'ckpt*.pt' -o -name '*_log.csv' -o -name 'resolved_config*.json' \) -printf '  %f\n' | sort || true
done

echo
echo "== progress tails =="
for f in \
  "$OBSP_DIR/descent_obs_pred_pretrain_log.csv" \
  "$CLP_DIR/descent_cable_latpred_pretrain_log.csv" \
  "$PPO_DIR/descent_ppo_vec_log.csv"; do
  if [[ -f "$f" ]]; then
    echo "-- $f"
    tail -n 3 "$f"
  fi
done

echo
echo "== master log tail =="
if [[ -f "$LOG" ]]; then
  tail -n 60 "$LOG"
else
  echo "no master log"
fi

