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

TEACHER="${TEACHER:-saves/descent_ppo_variable_wind_true_obs_finetune_20260531_110809/ckpt_best.pt}"

python -u train_phase.py \
  --phase descent --algo distill \
  --teacher-ckpt "$TEACHER" \
  --log-dir saves/descent_pidbase_ablation_no_pid_distill_dagger_terminal_noup_trueobs \
  --timesteps "${DISTILL_STEPS:-300000}" \
  --n-envs "${N_ENVS:-4}" \
  --seed "${SEED:-260626}" \
  --distill-batch-size "${DISTILL_BATCH:-1024}" \
  --distill-action-target-mode "${DISTILL_TARGET_MODE:-descent_stable_acc_plus_actor}" \
  --distill-teacher-residual-coef "${DISTILL_RESIDUAL_COEF:-0.35}" \
  --distill-student-rollin-prob "${DISTILL_STUDENT_ROLLIN_PROB:-0.02}" \
  --distill-student-rollin-start-prob "${DISTILL_STUDENT_ROLLIN_START:-0.0}" \
  --distill-student-rollin-warmup-steps "${DISTILL_STUDENT_ROLLIN_WARMUP:-80000}" \
  --distill-student-rollin-ramp-steps "${DISTILL_STUDENT_ROLLIN_RAMP:-160000}" \
  --distill-student-rollin-max-steps-per-episode "${DISTILL_STUDENT_ROLLIN_MAX_EP:-3}" \
  --distill-student-rollin-recovery-steps "${DISTILL_STUDENT_ROLLIN_RECOVERY:-12}" \
  --distill-student-rollin-min-episode-step "${DISTILL_STUDENT_ROLLIN_MIN_EP_STEP:-12}" \
  --disable-descent-pid-base \
  --disable-vision \
  --disable-obs-predictor \
  --disable-cable-latent-predictor \
  --wind-speed-max "${WIND_MAX:-0}" \
  2>&1 | tee "$OUT/distill_dagger_terminal_noup_trueobs_no_pid.log"
