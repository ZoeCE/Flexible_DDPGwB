#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

PY=/home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
BASE_LOG_DIR="saves/descent_ppo_variable_wind_true_obs_50hz_scratch"
RUN_LOG="test_results/descent_ppo_variable_wind_true_obs_50hz_scratch_${STAMP}.log"

"${PY}" train_phase.py \
  --phase descent \
  --algo ppo \
  --control-freq-hz 50 \
  --scale-limits-with-control-dt \
  --variable-wind \
  --wind-speed-max 10 \
  --n-envs 8 \
  --log-dir "${BASE_LOG_DIR}" \
  --timesteps 35000000 \
  2>&1 | tee "${RUN_LOG}"

echo
echo "Log: ${RUN_LOG}"
echo "Checkpoints: ${BASE_LOG_DIR}_YYYYMMDD_HHMMSS"
