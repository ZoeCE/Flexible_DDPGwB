#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

PY=/home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="test_results/descent_highwind_focus_pred_vs_true_variable_wind_0to10_8bin_128_${STAMP}.csv"
LOG="${OUT%.csv}.log"

RL_CKPT="saves/descent_ppo_variable_wind_true_obs_finetune_highwind_focus_20260531_231923/ckpt_latest.pt"
PRED_CKPT="saves/descent_obs_pred_pretrain_20260528_221230/ckpt_final_obs_predictor.pt"

"${PY}" test_phase.py \
  --phase descent \
  --algo ppo \
  --compare-pred-true-expert \
  --true-obs-ckpt "${RL_CKPT}" \
  --pred-ckpt "${RL_CKPT}" \
  --pred-obs-ckpt "${PRED_CKPT}" \
  --pred-obs-period 2 \
  --obs-predictor-target-mode raw \
  --episodes-per-bin 128 \
  --wind-bins "0-1.25,1.25-2.5,2.5-3.75,3.75-5,5-6.25,6.25-7.5,7.5-8.75,8.75-10" \
  --variable-wind \
  --wind-speed-max 10 \
  --wind-speed-band-abs 0.50 \
  --wind-speed-band-frac 0.15 \
  --wind-speed-rate-std 0.25 \
  --wind-dir-band-rad 0.35 \
  --wind-dir-rate-std 0.08 \
  --eval-curriculum-level max \
  --benchmark-progress-every 32 \
  --plot-benchmark \
  --benchmark-out "${OUT}" \
  2>&1 | tee "${LOG}"

echo
echo "CSV: ${OUT}"
echo "LOG: ${LOG}"
echo "Plots: ${OUT%.csv}_plots"
