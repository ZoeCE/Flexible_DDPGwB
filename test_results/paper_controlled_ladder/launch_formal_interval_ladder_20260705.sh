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
RUN_ID="${RUN_ID:-20260705_interval_wind_n${EPISODES}}"
OUT="${OUT:-test_results/paper_controlled_ladder/formal_interval_${RUN_ID}}"
SECTION="${SECTION:-all}"
RESUME="${RESUME:-1}"
SEED="${SEED:-270705}"

mkdir -p "$OUT"
cat > "$OUT/run_config.env" <<EOF
RUN_ID=$RUN_ID
EPISODES=$EPISODES
OUT=$OUT
SECTION=$SECTION
RESUME=$RESUME
SEED=$SEED
EOF

python -u test_results/paper_controlled_ladder/run_formal_interval_ladder_20260705.py \
  --out "$OUT" \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --section "$SECTION" \
  $(if [[ "$RESUME" == "1" ]]; then echo "--resume"; fi)

