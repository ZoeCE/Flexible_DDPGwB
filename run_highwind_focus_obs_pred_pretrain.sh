#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

PY=/home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
BASE_LOG_DIR="saves/descent_obs_pred_pretrain_highwind_focus_variable_wind_nclatent"
RUN_LOG="test_results/descent_obs_pred_pretrain_highwind_focus_variable_wind_nclatent_${STAMP}.log"

POLICY_CKPT="saves/descent_ppo_variable_wind_true_obs_finetune_highwind_focus_20260531_231923/ckpt_latest.pt"

"${PY}" train_phase.py \
  --phase descent \
  --algo obs_pred \
  --resume-ckpt "${POLICY_CKPT}" \
  --obs-period 2 \
  --obs-predictor-target-mode non_cable_latent \
  --obs-pred-pretrain-wind-min 3.0 \
  --variable-wind \
  --wind-speed-max 10 \
  --wind-speed-band-abs 0.50 \
  --wind-speed-band-frac 0.15 \
  --wind-speed-rate-std 0.25 \
  --wind-dir-band-rad 0.35 \
  --wind-dir-rate-std 0.08 \
  --n-envs 8 \
  --log-dir "${BASE_LOG_DIR}" \
  --timesteps 1200000 \
  2>&1 | tee "${RUN_LOG}"

echo
echo "Log: ${RUN_LOG}"
echo "Checkpoints: ${BASE_LOG_DIR}_YYYYMMDD_HHMMSS"
