#!/bin/bash
# Qwen3-Embedding-0.6B server on port 8001. Used by the retrieval stage
# (scripts/run_pipeline_from_scratch.sh). Not required if you use the
# pre-computed intermediate artifacts bundle.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-Embedding-0.6B bash scripts/start_embedding_server.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_SERVER_DEV_MODE=1

MODEL="${MODEL_PATH:?Please set MODEL_PATH=/absolute/path/to/Qwen3-Embedding-0.6B}"
PORT="${PORT:-8001}"
API_KEY="${VLLM_API_KEY:-embed-key-m3}"

vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --gpu-memory-utilization 0.30 \
  --max-model-len 13000 \
  --hf-overrides '{"matryoshka_dimensions":[32,128,256,512,1024]}' \
  --enable-sleep-mode
