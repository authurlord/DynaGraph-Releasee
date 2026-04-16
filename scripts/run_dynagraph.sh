#!/bin/bash
# Run the DynaGraph online benchmark across the three bundled datasets.
# Main setting: Zipf alpha=1.0, 30 QPS, 1000 unique queries, 3000 total requests.
#
# Usage:
#   bash scripts/run_dynagraph.sh [MODEL_TAG] [PORT] [API_KEY]

set -euo pipefail

MODEL_TAG=${1:-qwen3.5-4b}
PORT=${2:-8000}
API_KEY=${3:-api-key-qwen3}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

declare -A SOURCE_EXP_IDS=(
  [2wikimultihopqa]="baseline_v3"
  [hotpotqa]="baseline_full_1000q"
  [musique]="1"
)

if ! curl -fs -H "Authorization: Bearer ${API_KEY}" \
     "http://localhost:${PORT}/v1/models" > /dev/null; then
  echo "ERROR: LLM server not reachable at localhost:${PORT}."
  echo "Start it first (see README \"Main setting\")."
  exit 1
fi

for dataset in 2wikimultihopqa hotpotqa musique; do
  src="${SOURCE_EXP_IDS[$dataset]}"
  if [[ ! -f "tmp_data/${dataset}/retrieve_result_4B_${src}.npy" ]]; then
    echo "SKIP ${dataset}: missing retrieval artifact (unzip dynagraph_intermediate_artifacts.zip first)"
    continue
  fi

  echo ">>> ${dataset}"
  python src/benchmark_e2e_joint.py \
    --dataset "${dataset}" \
    --source_exp_id "${src}" \
    --config DynaGraph \
    --arrival zipf --zipf_alpha 1.0 --rate 30 \
    --questions 1000 --total_requests 3000 \
    --concurrency 64 --slo_ttft_ms 2000 \
    --k_min 10 --k_max 20 \
    --bp_k_max 19 --bp_threshold 0.7 \
    --k_mapping linear --bp_decay exponential \
    --engine vllm \
    --port "${PORT}" --api_key "${API_KEY}" \
    --model_name_tag "${MODEL_TAG}" \
    --exp_id "dynagraph_${MODEL_TAG}_${dataset}"
done

echo "Done. Reports saved under tmp_data/{dataset}/."
