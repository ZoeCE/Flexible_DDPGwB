#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

if [[ -f /home/xyzha/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/xyzha/miniconda3/etc/profile.d/conda.sh
  conda activate vsdrl_env_5060
fi

OUT="${OUT:-test_results/paper_controlled_ladder/formal_interval_20260705_interval_wind_n512}"
EPISODES="${EPISODES:-512}"
SEED="${SEED:-270705}"
LOG="$OUT/mlp8d_replacement.log"
PIDFILE="$OUT/mlp8d_replacement.pid"

mkdir -p "$OUT"
nohup python -u test_results/paper_controlled_ladder/run_formal_interval_ladder_20260705.py \
  --out "$OUT" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --section ablation \
  --profile-filter C4 \
  --variant-filter MLP8D \
  --resume \
  > "$LOG" 2>&1 &

echo $! > "$PIDFILE"
cat > "$OUT/mlp8d_replacement.env" <<EOF
OUT=$OUT
LOG=$LOG
PID=$(cat "$PIDFILE")
EPISODES=$EPISODES
SEED=$SEED
EOF

echo "[STARTED] PID=$(cat "$PIDFILE") LOG=$LOG"
