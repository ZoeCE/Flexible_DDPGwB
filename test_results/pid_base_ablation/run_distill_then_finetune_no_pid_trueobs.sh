#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/ResearchProject/DDPG/Flexible_DDPGwB

bash test_results/pid_base_ablation/run_distill_trueobs_no_pid.sh
bash test_results/pid_base_ablation/run_finetune_trueobs_no_pid.sh
