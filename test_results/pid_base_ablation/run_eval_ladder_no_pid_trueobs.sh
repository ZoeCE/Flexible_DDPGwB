#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
source /home/xyzha/miniconda3/etc/profile.d/conda.sh
conda activate vsdrl_env_5060

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

OUT="test_results/pid_base_ablation/eval_ladder_no_pid_trueobs"
mkdir -p "$OUT"

if [[ -n "${POLICY:-}" ]]; then
  CKPT="$POLICY"
else
  FT_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_finetune_dagger_terminal_noup_trueobs_* 2>/dev/null | head -n 1)"
  if [[ -z "$FT_DIR" ]]; then
    FT_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_finetune_dagger_terminal_nolift_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$FT_DIR" ]]; then
    FT_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_finetune_dagger_terminal_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$FT_DIR" ]]; then
    FT_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_finetune_dagger_stable_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$FT_DIR" ]]; then
    FT_DIR="$(ls -td saves/descent_pidbase_ablation_no_pid_finetune_fullact_trueobs_* 2>/dev/null | head -n 1)"
  fi
  if [[ -z "$FT_DIR" ]]; then
    echo "No finetune run found. Run run_finetune_trueobs_no_pid.sh first." >&2
    exit 1
  fi
  CKPT="$FT_DIR/ckpt_best.pt"
  [[ -f "$CKPT" ]] || CKPT="$FT_DIR/ckpt_latest.pt"
  [[ -f "$CKPT" ]] || CKPT="$FT_DIR/ckpt_final.pt"
fi

levels=(paper-l1 paper-l2 paper-l3 paper-l4 paper-l5)
winds=(0 0 2 4 6)

for i in "${!levels[@]}"; do
  lv="${levels[$i]}"
  wind="${winds[$i]}"
  seed=$((260725 + i * 100))
  python -u test_phase.py \
    --phase descent --algo ppo \
    --ckpt "$CKPT" \
    --task-profile "$lv" \
    --episodes "${EPISODES:-32}" \
    --seed "$seed" \
    --eval-curriculum-level max \
    --wind-speed "$wind" \
    --wind-speed-max 8 \
    --obstacles 0 \
    --disable-descent-pid-base \
    --disable-vision \
    --disable-obs-predictor \
    --disable-cable-latent-predictor \
    --summary-out "$OUT/${lv}_wind${wind}_n${EPISODES:-32}_summary.json" \
    2>&1 | tee "$OUT/${lv}_wind${wind}_n${EPISODES:-32}.log"
done
