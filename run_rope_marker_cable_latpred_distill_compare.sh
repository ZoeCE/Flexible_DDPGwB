#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
mkdir -p test_results

LOG="test_results/train_rope_marker_cable_latpred_distill_compare_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="test_results/train_rope_marker_cable_latpred_distill_compare.pid"

setsid -f bash -lc "cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB && exec nice -n 10 /home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python -u train_phase.py --profile descent_10hz_vision_pred_small_endpoint_fast_zgate_ft --algo distill --teacher-ckpt saves/descent_ppo_10hz_vision_pred_small_endpoint_fast_zgate_ft_continue_20260608_092016/ckpt_best.pt --log-dir saves/descent_distill_10hz_vision_rope_marker_cable_latpred_l0_compare --resume-in-place --timesteps 600000 --n-envs 4 --vision --vision-period 1 --vision-latency-steps 1 --vision-processing-delay-steps 1 --disable-obs-predictor --cable-latent-predictor --cable-latent-use-rope-markers --cable-latent-predictor-lr 3e-4 --target-xy-randomize --target-xy-range 0.005 --curriculum-start-level 0 --curriculum-start-wind 0.0 --distill-batch-size 2048 > '$LOG' 2>&1"

sleep 1
PID=$(pgrep -f "train_phase.py .*rope_marker_cable_latpred_l0_compare" | tail -n 1 || true)
echo "$PID" > "$PID_FILE"
echo "PID=$PID"
echo "LOG=$LOG"
