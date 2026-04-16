#!/bin/bash
# DynaGraph default LLM server: Qwen3.5-4B on two GPUs (TP=2),
# with priority scheduling enabled (required for Phase 3).
#
# Download the model: https://huggingface.co/Qwen/Qwen3.5-4B
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_vllm_qwen35_4b.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_SERVER_DEV_MODE=1
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_PATH:?Set MODEL_PATH=/absolute/path/to/Qwen3.5-4B}"
SERVED="${SERVED_NAME:-qwen3.5-4b}"
PORT="${PORT:-8000}"
API_KEY="${VLLM_API_KEY:-api-key-qwen3}"

vllm serve "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 --port "${PORT}" --api-key "${API_KEY}" \
  --tensor-parallel-size 2 \
  --attention-backend FLASHINFER \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --max-model-len 23000 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --async-scheduling \
  --scheduling-policy priority \
  --performance-mode throughput \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce
