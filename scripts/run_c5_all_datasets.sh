#!/bin/bash
# DynaGraph (C5) online benchmark — paper main setting.
#
# Runs Zipf-arrival online replay across 2WikiMultihopQA / HotpotQA / MuSiQue
# against a running vLLM server (default http://localhost:8000).
#
# Expects the intermediate artifacts to be present at:
#   tmp_data/{dataset}/retrieve_result_4B_{source_exp_id}.npy
#   tmp_data/{dataset}/graph_features_1000q.json
#
# If you do not have them, either:
#   (a) download the intermediate artifacts zip (see README §"Reproducibility
#       options") and unpack into tmp_data/, OR
#   (b) run scripts/run_pipeline_from_scratch.sh first.
#
# Usage:
#   bash scripts/run_c5_all_datasets.sh [MODEL_TAG] [PORT] [CONCURRENCY] [RATE] [TOTAL] [API_KEY]
#
# Example (main paper setting, Qwen3.5-4B TP2):
#   bash scripts/run_c5_all_datasets.sh qwen3.5-4b 8000 64 30 3000 api-key-qwen3

set -euo pipefail

MODEL_TAG=${1:-qwen3.5-4b}
PORT=${2:-8000}
CONCURRENCY=${3:-64}
RATE=${4:-30.0}
TOTAL_REQUESTS=${5:-3000}
API_KEY=${6:-api-key-qwen3}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

declare -A SOURCE_EXP_IDS=(
  [2wikimultihopqa]="baseline_v3"
  [hotpotqa]="baseline_full_1000q"
  [musique]="1"
)

if ! curl -fs -H "Authorization: Bearer ${API_KEY}" \
     "http://localhost:${PORT}/v1/models" > /dev/null; then
  echo "ERROR: LLM server not reachable at localhost:${PORT}"
  echo "Start it first, e.g.:"
  echo "  MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_vllm_qwen35_4b_tuned.sh"
  exit 1
fi

echo "========================================================"
echo "  DynaGraph C5 — 3 datasets (${MODEL_TAG})"
echo "  zipf(alpha=1.0), rate=${RATE}qps, total=${TOTAL_REQUESTS}"
echo "  concurrency=${CONCURRENCY}, slo=2000ms, k_min=10, k_max=20"
echo "========================================================"

for dataset in 2wikimultihopqa hotpotqa musique; do
  src="${SOURCE_EXP_IDS[$dataset]}"
  rr_path="tmp_data/${dataset}/retrieve_result_4B_${src}.npy"
  gf_path="tmp_data/${dataset}/graph_features_1000q.json"

  if [[ ! -f "${rr_path}" ]]; then
    echo "SKIP ${dataset}: missing ${rr_path}" ; continue
  fi
  if [[ ! -f "${gf_path}" ]]; then
    echo "SKIP ${dataset}: missing ${gf_path}" ; continue
  fi

  exp_id="C5_${MODEL_TAG}_${dataset}_zipf${RATE%.*}_${TOTAL_REQUESTS}"
  echo ">>> ${dataset}  exp=${exp_id}"

  python src/benchmark_e2e_joint.py \
    --dataset "${dataset}" \
    --source_exp_id "${src}" \
    --config C5 \
    --arrival zipf --zipf_alpha 1.0 \
    --rate "${RATE}" \
    --questions 1000 \
    --total_requests "${TOTAL_REQUESTS}" \
    --concurrency "${CONCURRENCY}" \
    --slo_ttft_ms 2000 \
    --k_min 10 --k_max 20 \
    --bp_k_max 19 --bp_threshold 0.7 \
    --k_mapping linear --bp_decay exponential \
    --engine vllm \
    --port "${PORT}" --api_key "${API_KEY}" \
    --model_name_tag "${MODEL_TAG}" \
    --exp_id "${exp_id}"
done

echo "Done. Per-request reports saved under tmp_data/{dataset}/."
