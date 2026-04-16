#!/bin/bash
# Warm up the LLM server with realistic full-length QA prompts before
# running the benchmark. This amortises the CUDA-graph capture cost,
# populates the prefix cache with the shared few-shot prefix, and lets
# the priority scheduler reach steady state.
#
# Required because the first ~200 requests otherwise show large TTFT
# outliers that distort SLO@2s and QA-Goodput numbers.
#
# Usage:
#   bash scripts/warmup_server.sh [MODEL_TAG] [PORT] [API_KEY]

set -euo pipefail

MODEL_TAG=${1:-qwen3.5-4b}
PORT=${2:-8000}
API_KEY=${3:-api-key-qwen3}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "tmp_data/2wikimultihopqa/retrieve_result_4B_baseline_v3.npy" ]]; then
  echo "ERROR: missing retrieval artifact — unpack dynagraph_intermediate_artifacts.zip first."
  exit 1
fi

echo "Warming up ${MODEL_TAG} at localhost:${PORT} with 300 realistic requests..."

python src/benchmark_e2e_joint.py \
  --dataset 2wikimultihopqa \
  --source_exp_id baseline_v3 \
  --config DynaGraph \
  --arrival zipf --zipf_alpha 1.0 --rate 30 \
  --questions 300 --total_requests 300 \
  --concurrency 64 --slo_ttft_ms 2000 \
  --k_min 10 --k_max 20 \
  --bp_k_max 19 --bp_threshold 0.7 \
  --k_mapping linear --bp_decay exponential \
  --engine vllm \
  --port "${PORT}" --api_key "${API_KEY}" \
  --model_name_tag "${MODEL_TAG}" \
  --exp_id _warmup_discard > /dev/null 2>&1 || true

# Remove the discardable warmup report so it is not confused with real results
rm -f tmp_data/2wikimultihopqa/e2e__warmup_discard_report.json

echo "Warmup done."
