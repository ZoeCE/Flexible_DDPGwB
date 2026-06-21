#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB
mkdir -p test_results

ts=$(date +%Y%m%d_%H%M%S)
log="test_results/windbin_opencv_marker_64_${ts}.log"
csv="test_results/windbin_opencv_marker_64_${ts}.csv"
pidfile="test_results/windbin_opencv_marker_64_${ts}.pid"

echo "$$" > "$pidfile"
echo "PID=$$"
echo "LOG=$log"
echo "CSV=$csv"
echo "PIDFILE=$pidfile"

exec env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  /home/xyzha/miniconda3/envs/vsdrl_env_5060/bin/python -u test_phase.py \
  --phase descent \
  --algo ppo \
  --ckpt saves/descent_ppo_10hz_vision_rope_marker_rgbd_cable_latpred_l0_to_l8_20260618_010735/ckpt_latest.pt \
  --compare-wind-bins \
  --benchmark-rl-only \
  --episodes-per-bin 64 \
  --wind-bins 0-1.25,1.25-2.5,2.5-3.75,3.75-5,5-6.25,6.25-7.5,7.5-8.75,8.75-10 \
  --benchmark-out "$csv" \
  --vision \
  --vision-period 1 \
  --vision-latency-steps 1 \
  --vision-processing-delay-steps 1 \
  --cable-latent-predictor \
  --cable-latent-predictor-ckpt saves/descent_ppo_10hz_vision_rope_marker_rgbd_cable_latpred_l0_to_l8_20260618_010735/ckpt_latest_cable_latent_predictor.pt \
  --cable-latent-use-rope-markers \
  --rope-marker-feature-source opencv_rgbd \
  --target-xy-randomize \
  --target-xy-range 0.01 \
  --eval-curriculum-level max \
  --obstacles 0 \
  --benchmark-progress-every 8 \
  > "$log" 2>&1
