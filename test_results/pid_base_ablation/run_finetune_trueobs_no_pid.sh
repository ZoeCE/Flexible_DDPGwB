#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
source /home/xyzha/miniconda3/etc/profile.d/conda.sh
conda activate vsdrl_env_5060

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

OUT="test_results/pid_base_ablation"
mkdir -p "$OUT"

if [[ -n "${DISTILL_CKPT:-}" ]]; then
  CKPT="$DISTILL_CKPT"
else
  DISTILL_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_noup_trueobs_* 2>/dev/null | head -n 1)"
  if [[ -z "$DISTILL_DIR" ]]; then
    DISTILL_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_nolift_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$DISTILL_DIR" ]]; then
    DISTILL_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$DISTILL_DIR" ]]; then
    DISTILL_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_distill_dagger_stable_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$DISTILL_DIR" ]]; then
    DISTILL_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_distill_fullact_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$DISTILL_DIR" ]]; then
    echo "No distill run found. Run run_distill_trueobs_no_pid.sh first." >&2
    exit 1
  fi
  CKPT="$DISTILL_DIR/ckpt_best.pt"
  [[ -f "$CKPT" ]] || CKPT="$DISTILL_DIR/ckpt_final.pt"
fi

python -u train_phase.py \
  --phase descent --algo ppo \
  --resume-ckpt "$CKPT" \
  --log-dir saves/descent_pidbase_ablation_no_pid_finetune_dagger_terminal_noup_trueobs \
  --timesteps "${FINETUNE_STEPS:-1000000}" \
  --n-envs "${N_ENVS:-4}" \
  --seed "${SEED:-260627}" \
  --reset-optimizer-on-resume \
  --ppo-lr-actor "${LR_ACTOR:-1e-5}" \
  --ppo-lr-critic "${LR_CRITIC:-5e-5}" \
  --descent-action-rms-free "${ACTION_RMS_FREE:-1.0}" \
  --descent-action-magnitude-coef "${ACTION_MAG_COEF:-0.02}" \
  --disable-descent-pid-base \
  --disable-vision \
  --disable-obs-predictor \
  --disable-cable-latent-predictor \
  --wind-speed-max "${WIND_MAX:-2}" \
  2>&1 | tee "$OUT/finetune_dagger_terminal_noup_trueobs_no_pid.log"
