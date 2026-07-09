#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

PY=/home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python
DIR=saves/descent_ablation_full8d_trueobs_5hzctrl_5hzobs_pretrain_20260706
LOG="$DIR/train.log"
PID_FILE="$DIR/pid.txt"

mkdir -p "$DIR"

nohup env \
  WANDB_PROJECT=phase_rl_v11_ablation \
  WANDB_NAME=full8d_trueobs_5hzctrl_5hzobs_ppo8m_seed270727_20260706 \
  "$PY" -u train_phase.py \
    --profile descent_variable_wind_true_obs \
    --log-dir "$DIR" \
    --resume-in-place \
    --timesteps 8000000 \
    --n-envs 8 \
    --seed 270727 \
    --control-freq-hz 5 \
    --scale-limits-with-control-dt \
    --disable-vision \
    --disable-obs-predictor \
    --disable-cable-latent-predictor \
    --cable-encoder-output-dim 8 \
    --trainable-cable-encoder \
    --cable-encoder-lr 1e-4 \
    > "$LOG" 2>&1 &

echo $! > "$PID_FILE"
echo "Started Full8D 5Hz full pretrain PID $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Status: bash scripts/status_full8d_5hz_pretrain_20260706.sh"
