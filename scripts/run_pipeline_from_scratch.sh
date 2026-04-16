#!/bin/bash
# End-to-end pipeline: rebuild every retrieval/KG artifact from only
# data/{dataset}.json and data/{dataset}_corpus.json.
#
# Required running servers (see README):
#   - LLM       at :8000
#   - Embedding at :8001
#   - Reranker  at :8002
#
# Usage:
#   bash scripts/run_pipeline_from_scratch.sh [MODEL_NAME] [TRIPLE_SIZE]

set -euo pipefail

MODEL_NAME=${1:-qwen3.5-4b}
TRIPLE_SIZE=${2:-4B}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

declare -A EXP_IDS=(
  [2wikimultihopqa]="baseline_v3"
  [hotpotqa]="baseline_full_1000q"
  [musique]="1"
)

for dataset in 2wikimultihopqa hotpotqa musique; do
  exp_id="${EXP_IDS[$dataset]}"
  echo "==== [${dataset}] Stage 1/3: KG extraction (${TRIPLE_SIZE}) ===="
  python src/extract_kg_standalone.py \
    --dataset "${dataset}" \
    --model_name "${MODEL_NAME}"

  echo "==== [${dataset}] Stage 2/3: retrieval (exp_id=${exp_id}) ===="
  python src/pipeline_baseline.py \
    --dataset "${dataset}" \
    --model_name "${MODEL_NAME}" \
    --exp_id "${exp_id}" \
    --triple_size "${TRIPLE_SIZE}" \
    --kg_match_type exact

  echo "==== [${dataset}] Stage 3/3: graph features ===="
  python src/graph_feature_extractor.py \
    --dataset "${dataset}" \
    --source_exp_id "${exp_id}" \
    --triple_size "${TRIPLE_SIZE}" \
    --questions 1000 \
    --output graph_features_1000q.json
done

echo "Pipeline complete. Now run scripts/run_dynagraph.sh."
