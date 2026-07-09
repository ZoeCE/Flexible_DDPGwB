#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

PY=/home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python
INIT_DIR=saves/descent_ablation_full8d_scratch_init_seed270716_20260706
INIT="$INIT_DIR/ckpt_init.pt"
OBSP_DIR=saves/descent_ablation_obspred_full8d_scratchinit_pretrain_20260706
CLP_DIR=saves/descent_ablation_cable_latpred_full8d_scratchinit_pretrain_20260706
PPO_DIR=saves/descent_ablation_full8d_obspred_clp_p2_10hzctrl_5hzobs_ppo_20260706

mkdir -p "$INIT_DIR" "$OBSP_DIR" "$CLP_DIR" "$PPO_DIR" test_results

echo "[$(date '+%F %T')] Stage 0/3: creating untrained Full8D init checkpoint"
"$PY" scripts/make_full8d_scratch_init.py \
  --out "$INIT" \
  --seed 270716 \
  2>&1 | tee "$INIT_DIR/create_init.log"

echo "[$(date '+%F %T')] Stage 1/3: pretraining ObsPred from scratch-init encoder"
WANDB_PROJECT=phase_rl_v11_ablation \
WANDB_NAME=full8d_scratchinit_obspred_p2_pretrain_seed270706_20260706 \
"$PY" -u train_phase.py \
  --profile descent_ablation_obs_pred_fullobs_pretrain \
  --log-dir "$OBSP_DIR" \
  --resume-in-place \
  --resume-ckpt "$INIT" \
  --timesteps 1000000 \
  --n-envs 8 \
  --seed 270706 \
  --disable-vision \
  --disable-cable-latent-predictor \
  --cable-encoder-output-dim 8 \
  --trainable-cable-encoder \
  --cable-encoder-lr 1e-4 \
  --obs-predictor \
  --obs-period 2 \
  --obs-predictor-target-mode non_cable_latent \
  2>&1 | tee "$OBSP_DIR/train.log"

OBSP="$OBSP_DIR/ckpt_final_obs_predictor.pt"
if [[ ! -f "$OBSP" ]]; then
  echo "ObsPred pretrain did not finish; missing final checkpoint: $OBSP" >&2
  exit 2
fi
echo "[$(date '+%F %T')] ObsPred pretrain complete: $OBSP"

echo "[$(date '+%F %T')] Stage 2/3: pretraining CableLatPred from scratch-init encoder"
WANDB_PROJECT=phase_rl_v11_ablation \
WANDB_NAME=full8d_scratchinit_cable_latpred_pretrain_seed270707_20260706 \
"$PY" -u scripts/pretrain_cable_latent_predictor.py \
  --log-dir "$CLP_DIR" \
  --resume-ckpt "$INIT" \
  --timesteps 1000000 \
  --n-envs 8 \
  --seed 270707 \
  --cable-encoder-output-dim 8 \
  --cable-encoder-lr 1e-4 \
  --use-rope-markers \
  --rope-marker-source site \
  2>&1 | tee "$CLP_DIR/train.log"

CLP="$CLP_DIR/ckpt_final_cable_latent_predictor.pt"
if [[ ! -f "$CLP" ]]; then
  echo "CableLatPred pretrain did not finish; missing final checkpoint: $CLP" >&2
  exit 3
fi
echo "[$(date '+%F %T')] CableLatPred pretrain complete: $CLP"

echo "[$(date '+%F %T')] Stage 3/3: starting scratch PPO with ObsPred + CableLatPred"
WANDB_PROJECT=phase_rl_v11_ablation \
WANDB_NAME=full8d_scratch_obspred_clp_p2_10hzctrl_5hzobs_ppo8m_seed270716_20260706 \
"$PY" -u train_phase.py \
  --profile descent_variable_wind_true_obs \
  --log-dir "$PPO_DIR" \
  --resume-in-place \
  --resume-ckpt "$INIT" \
  --obs-predictor-ckpt "$OBSP" \
  --cable-latent-predictor-ckpt "$CLP" \
  --timesteps 8000000 \
  --n-envs 8 \
  --seed 270716 \
  --disable-vision \
  --cable-encoder-output-dim 8 \
  --trainable-cable-encoder \
  --cable-encoder-lr 1e-4 \
  --obs-predictor \
  --obs-period 2 \
  --obs-predictor-target-mode non_cable_latent \
  --cable-latent-predictor \
  --cable-latent-use-rope-markers \
  --rope-marker-feature-source site \
  2>&1 | tee "$PPO_DIR/train.log"

echo "[$(date '+%F %T')] Full8D scratch ObsPred+CableLatPred chain done"
