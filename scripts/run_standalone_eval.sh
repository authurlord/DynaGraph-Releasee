#!/bin/bash
# Standalone Phase-2 (adaptive K) evaluation — used for fast iteration of
# aggregation logic without an online server. Outputs NDCG / F1 / avgK /
# token reduction.
#
# Usage:
#   bash scripts/run_standalone_eval.sh [DATASET] [AGG_MODE] [EXP_ID]

set -euo pipefail

DATASET=${1:-2wikimultihopqa}
AGG_MODE=${2:-joint}
EXP_ID=${3:-C5_standalone_${DATASET}}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

declare -A SOURCE_EXP_IDS=(
  [2wikimultihopqa]="baseline_v3"
  [hotpotqa]="baseline_full_1000q"
  [musique]="1"
)
src="${SOURCE_EXP_IDS[$DATASET]}"

python src/eval_phase2_standalone.py \
  --dataset "${DATASET}" \
  --source_exp_id "${src}" \
  --questions 1000 \
  --agg_mode "${AGG_MODE}" \
  --k_min 10 --k_max 20 \
  --bp_k_max 19 --bp_threshold 0.7 \
  --k_mapping linear \
  --exp_id "${EXP_ID}"
