#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

if [[ -f /home/xyzha/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/xyzha/miniconda3/etc/profile.d/conda.sh
  conda activate vsdrl_env_5060
fi

EPISODES="${EPISODES:-512}"
SEED="${SEED:-270706}"
OUT="${OUT:-test_results/paper_controlled_ladder/ablation_c4_wind8to10_20260706_n${EPISODES}}"
LOG="$OUT/controller.log"
PIDFILE="$OUT/controller.pid"

mkdir -p "$OUT"
nohup python -u test_results/paper_controlled_ladder/run_ablation_c4_wind8to10_20260706.py \
  --out "$OUT" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --resume \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
cat > "$OUT/run.env" <<EOF
OUT=$OUT
LOG=$LOG
PID=$(cat "$PIDFILE")
EPISODES=$EPISODES
SEED=$SEED
EOF

cat > test_results/paper_controlled_ladder/latest_ablation_wind8to10.env <<EOF
OUT=$OUT
PID=$(cat "$PIDFILE")
EOF

echo "[STARTED] PID=$(cat "$PIDFILE") OUT=$OUT"
