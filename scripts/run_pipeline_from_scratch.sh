#!/bin/bash
# From-scratch pipeline that rebuilds the intermediate artifacts for all
# three datasets starting from only data/{dataset}.json + corpus.
#
# Stages:
#   1. KG extraction   → tmp_data/{ds}/triple_extraction_{size}.npy
#   2. Rewrite + embed + retrieve + rerank
#                      → tmp_data/{ds}/retrieve_result_{size}_{exp_id}.npy
#   3. Graph features  → tmp_data/{ds}/graph_features_1000q.json
#
# Required running servers:
#   - LLM (Qwen3.5-4B or Qwen3.5-35B) at :8000
#   - Embedding (Qwen3-Embedding-0.6B) at :8001
#   - Reranker (bge-reranker-v2-m3)    at :8002
#
# See README §"Reproduce from scratch" for a walkthrough.
#
# Usage:
#   bash scripts/run_pipeline_from_scratch.sh [MODEL_NAME] [TRIPLE_SIZE]
#
# Example:
#   bash scripts/run_pipeline_from_scratch.sh qwen3.5-4b 35B

set -euo pipefail

MODEL_NAME=${1:-qwen3.5-4b}
TRIPLE_SIZE=${2:-35B}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

declare -A EXP_IDS=(
  [2wikimultihopqa]="baseline_v3"
  [hotpotqa]="baseline_full_1000q"
  [musique]="1"
)

for dataset in 2wikimultihopqa hotpotqa musique; do
  exp_id="${EXP_IDS[$dataset]}"
  echo "==== [${dataset}] stage 1/3 KG extraction (${TRIPLE_SIZE}) ===="
  python src/extract_kg_standalone.py \
    --dataset "${dataset}" \
    --model_name "qwen3.5-${TRIPLE_SIZE,,}" || \
  python src/extract_kg_standalone.py \
    --dataset "${dataset}" \
    --model_name "qwen3-${TRIPLE_SIZE,,}"

  echo "==== [${dataset}] stage 2/3 retrieval pipeline (exp_id=${exp_id}) ===="
  python src/pipeline_baseline.py \
    --dataset "${dataset}" \
    --model_name "${MODEL_NAME}" \
    --exp_id "${exp_id}" \
    --triple_size "${TRIPLE_SIZE}" \
    --kg_match_type exact

  echo "==== [${dataset}] stage 3/3 graph feature extraction ===="
  python src/graph_feature_extractor.py \
    --dataset "${dataset}" \
    --source_exp_id "${exp_id}" \
    --triple_size "${TRIPLE_SIZE}" \
    --num_queries 1000 \
    --output graph_features_1000q.json
done

echo "Pipeline complete. Now run scripts/run_c5_all_datasets.sh."
