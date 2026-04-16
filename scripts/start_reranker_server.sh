#!/bin/bash
# bge-reranker-v2-m3 server on port 8002. Used by the retrieval stage.
#
# Usage:
#   MODEL_PATH=/path/to/bge-reranker-v2-m3 bash scripts/start_reranker_server.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_SERVER_DEV_MODE=1

MODEL="${MODEL_PATH:?Please set MODEL_PATH=/absolute/path/to/bge-reranker-v2-m3}"
PORT="${PORT:-8002}"
API_KEY="${VLLM_API_KEY:-reranker-key-m3}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --runner pooling \
  --tensor-parallel-size 1 \
  --api-key "${API_KEY}" \
  --gpu-memory-utilization 0.18 \
  --max-model-len 8192
