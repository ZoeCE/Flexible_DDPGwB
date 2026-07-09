#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/d/ResearchProject/DDPG/Flexible_DDPGwB}"
cd "$REPO"

if [[ -f /home/xyzha/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/xyzha/miniconda3/etc/profile.d/conda.sh
  conda activate vsdrl_env_5060
fi

EPISODES="${EPISODES:-256}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-test_results/paper_controlled_ladder/controlled_variable_${RUN_ID}_n${EPISODES}}"
RESUME="${RESUME:-1}"
PROFILE_FILTER="${PROFILE_FILTER:-}"
VARIANT_FILTER="${VARIANT_FILTER:-}"
TRADITIONAL_OVERRIDES="${TRADITIONAL_OVERRIDES:-test_results/paper_controlled_ladder/tuned_traditional_experts_20260702.json}"
mkdir -p "$OUT/logs" "$OUT/json"

cat > "$OUT/run_config.env" <<EOF
RUN_ID=$RUN_ID
EPISODES=$EPISODES
OUT=$OUT
RESUME=$RESUME
PROFILE_FILTER=$PROFILE_FILTER
VARIANT_FILTER=$VARIANT_FILTER
TRADITIONAL_OVERRIDES=$TRADITIONAL_OVERRIDES
EOF
cp test_results/paper_controlled_ladder/README.md "$OUT/README_controlled_ladder.md"

MAINLINE="saves/descent_ppo_paper_trueobs_residual_pretrain_online_20260627_010320/ckpt_latest.pt"
NOCABLE="saves/descent_ablation_trueobs_no_cable_latent_pretrain_20260627_224522/ckpt_latest.pt"
P2_POLICY="saves/descent_ablation_fullobs_obspred_p2_ft_20260629_145921/ckpt_latest.pt"
P2_OBSP="saves/descent_ablation_fullobs_obspred_p2_ft_20260629_145921/ckpt_latest_obs_predictor.pt"

for path in "$MAINLINE" "$NOCABLE" "$P2_POLICY" "$P2_OBSP"; do
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing checkpoint: $path" >&2
    exit 2
  fi
done
if [[ ! -f "$TRADITIONAL_OVERRIDES" ]]; then
  echo "[ERROR] Missing traditional expert override file: $TRADITIONAL_OVERRIDES" >&2
  exit 2
fi

profiles=(
  ctrl-c0-base
  ctrl-c1-init
  ctrl-c2-rope10
  ctrl-c3-wind2
  ctrl-c4-wind6
  ctrl-c4-wind8
  ctrl-c5-ratio4-w6
  ctrl-c5-ratio4-w8
)
winds=(0 0 0 2 6 8 6 8)
variants=(PID DampedPD MPC Mainline NoCable MainlineP2)

list_has_item() {
  local list="$1"
  local item="$2"
  [[ -z "$list" ]] && return 0
  local padded=",${list},"
  [[ "$padded" == *",${item},"* ]]
}

run_one() {
  local profile="$1"
  local wind="$2"
  local variant="$3"
  local seed="$4"
  local json="$OUT/json/${profile}__${variant}.json"
  local log="$OUT/logs/${profile}__${variant}.log"

  if [[ "$RESUME" == "1" && -s "$json" ]]; then
    echo "[SKIP] $profile / $variant already has $json"
    return 0
  fi

  echo
  echo "============================================================"
  echo "[RUN] profile=$profile variant=$variant episodes=$EPISODES seed=$seed wind=$wind"
  echo "      json=$json"
  echo "============================================================"

  local common=(
    python -u test_phase.py
    --phase descent
    --task-profile "$profile"
    --episodes "$EPISODES"
    --seed "$seed"
    --eval-curriculum-level max
    --wind-speed "$wind"
    --wind-speed-max 10
    --obstacles 0
    --quiet
    --disable-vision
    --disable-cable-latent-predictor
    --summary-out "$json"
  )

  case "$variant" in
    PID)
      "${common[@]}" --disable-obs-predictor \
        --traditional-expert-overrides-json "$TRADITIONAL_OVERRIDES" \
        --algo expert --descent-base-expert traditional_pid 2>&1 | tee "$log"
      ;;
    DampedPD)
      "${common[@]}" --disable-obs-predictor \
        --traditional-expert-overrides-json "$TRADITIONAL_OVERRIDES" \
        --algo expert --descent-base-expert damped_pd 2>&1 | tee "$log"
      ;;
    MPC)
      "${common[@]}" --disable-obs-predictor --algo expert --descent-base-expert mpc 2>&1 | tee "$log"
      ;;
    Mainline)
      "${common[@]}" --disable-obs-predictor --algo ppo --ckpt "$MAINLINE" 2>&1 | tee "$log"
      ;;
    NoCable)
      "${common[@]}" --disable-obs-predictor --algo ppo --ckpt "$NOCABLE" --disable-cable-obs 2>&1 | tee "$log"
      ;;
    MainlineP2)
      "${common[@]}" \
        --algo ppo --ckpt "$P2_POLICY" \
        --obs-predictor \
        --obs-predictor-ckpt "$P2_OBSP" \
        --obs-predictor-target-mode non_cable_latent \
        --obs-period 2 \
        2>&1 | tee "$log"
      ;;
    *)
      echo "[ERROR] unknown variant: $variant" >&2
      exit 3
      ;;
  esac
}

total=0
for i in "${!profiles[@]}"; do
  profile="${profiles[$i]}"
  list_has_item "$PROFILE_FILTER" "$profile" || continue
  for j in "${!variants[@]}"; do
    variant="${variants[$j]}"
    list_has_item "$VARIANT_FILTER" "$variant" || continue
    total=$((total + 1))
  done
done
done_count=0
for i in "${!profiles[@]}"; do
  profile="${profiles[$i]}"
  list_has_item "$PROFILE_FILTER" "$profile" || continue
  wind="${winds[$i]}"
  for j in "${!variants[@]}"; do
    variant="${variants[$j]}"
    list_has_item "$VARIANT_FILTER" "$variant" || continue
    seed=$((270702000 + i * 10000 + j * 1000))
    done_count=$((done_count + 1))
    echo "[PROGRESS] $done_count/$total"
    run_one "$profile" "$wind" "$variant" "$seed"
  done
done

python test_results/paper_controlled_ladder/generate_controlled_ladder_report.py --out "$OUT"
echo "[DONE] controlled-variable ladder complete: $OUT"
