#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

OUT="test_results/wind_bins_trainable16_vs_trainable8_l8best_20260705"
PY="/home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python"

FULL16_CKPT="saves/descent_ppo_paper_trueobs_residual_pretrain_trainable16_seed270627_20260703/ckpt_best_l8.pt"
FULL8_CKPT="saves/descent_ppo_paper_trueobs_residual_pretrain_trainable8_seed270627_20260704/ckpt_best_l8.pt"

mkdir -p "$OUT/plots_full16" "$OUT/plots_full8"

COMMON_ARGS=(
  --phase descent
  --algo ppo
  --compare-wind-bins
  --benchmark-rl-only
  --episodes-per-bin 256
  --seed 280705
  --eval-curriculum-level max
  --wind-bins "0-2,2-4,4-6,6-8,8-10"
  --wind-speed-max 10
  --variable-wind
  --wind-speed-band-abs 0.5
  --wind-speed-band-frac 0.15
  --wind-speed-rate-std 0.25
  --wind-dir-band-rad 0.35
  --wind-dir-rate-std 0.08
  --disable-vision
  --disable-obs-predictor
  --disable-cable-latent-predictor
  --trainable-cable-encoder
  --benchmark-progress-every 32
  --plot-benchmark
)

{
  echo "=== FullTrainable16 vs FullTrainable8 wind-bin L8-best test started at $(date -Is) ==="
  echo "Full16 ckpt: $FULL16_CKPT"
  echo "Full8 ckpt:  $FULL8_CKPT"
  echo "episodes_per_bin=256, seed=280705"
  echo "metrics: broad/strict SR, lucky insert, reward/steps, swing KE, swing angle, EE acceleration, payload velocity, terminal rates, termination counts"
  echo

  echo "=== FullTrainable16 ==="
  "$PY" -u test_phase.py \
    "${COMMON_ARGS[@]}" \
    --ckpt "$FULL16_CKPT" \
    --cable-encoder-output-dim 16 \
    --benchmark-out "$OUT/full_trainable16_l8best_wind_bins.csv" \
    --plot-out-dir "$OUT/plots_full16"

  echo
  echo "=== FullTrainable8 ==="
  "$PY" -u test_phase.py \
    "${COMMON_ARGS[@]}" \
    --ckpt "$FULL8_CKPT" \
    --cable-encoder-output-dim 8 \
    --benchmark-out "$OUT/full_trainable8_l8best_wind_bins.csv" \
    --plot-out-dir "$OUT/plots_full8"

  echo
  echo "=== Merge summaries ==="
  "$PY" "$OUT/summarize_wind_bins.py"
  echo "=== FullTrainable16 vs FullTrainable8 wind-bin L8-best test finished at $(date -Is) ==="
} > "$OUT/wind_bins_l8best.stdout.log" 2>&1
