#!/bin/bash
# Embedding server for the end-to-end pipeline: Qwen3-Embedding-0.6B on GPU 0.
# Only needed when regenerating retrieval artifacts from scratch.
#
# Download: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-Embedding-0.6B bash scripts/start_embedding_server.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_SERVER_DEV_MODE=1

MODEL="${MODEL_PATH:?Set MODEL_PATH=/absolute/path/to/Qwen3-Embedding-0.6B}"
PORT="${PORT:-8001}"
API_KEY="${VLLM_API_KEY:-embed-key-m3}"

vllm serve "${MODEL}" \
  --host 0.0.0.0 --port "${PORT}" --api-key "${API_KEY}" \
  --gpu-memory-utilization 0.30 \
  --max-model-len 13000 \
  --hf-overrides '{"matryoshka_dimensions":[32,128,256,512,1024]}'
