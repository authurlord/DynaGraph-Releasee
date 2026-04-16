#!/bin/bash
# Qwen3-4B-Instruct-2507 TP2 — used for historical Qwen3 comparisons and
# as the model compatible with LMCache / CacheBlend.
#
# This model is NOT a VL model; no `--language-model-only` flag needed.
# It uses the standard Qwen3 chat template (no enable_thinking switch).
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 bash scripts/start_vllm_qwen3_4b.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_SERVER_DEV_MODE=1

MODEL="${MODEL_PATH:?Please set MODEL_PATH=/absolute/path/to/Qwen3-4B-Instruct-2507}"
SERVED="${SERVED_NAME:-qwen3-4b}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-23000}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
MAX_SEQS="${MAX_SEQS:-64}"
API_KEY="${VLLM_API_KEY:-api-key-qwen3}"

echo "========================================================"
echo "  Qwen3-4B-Instruct-2507 TP2"
echo "  model=${MODEL}  served=${SERVED}  port=${PORT}"
echo "========================================================"

vllm serve "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --tensor-parallel-size 2 \
  --attention-backend FLASHINFER \
  --dtype auto \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_BATCHED}" \
  --max-num-seqs "${MAX_SEQS}" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --disable-custom-all-reduce
