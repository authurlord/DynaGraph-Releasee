#!/bin/bash
# DynaGraph C5 anchor server: Qwen3.5-4B TP2, TTFT-tuned.
# This is the configuration used for the main experimental results.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_vllm_qwen35_4b_tuned.sh
#
# Optional env overrides:
#   PORT=8000 GPU_MEM=0.85 MAX_MODEL_LEN=23000
#   MAX_BATCHED=8192 MAX_SEQS=64 PERF_MODE=throughput
#   CUDA_VISIBLE_DEVICES=0,1

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_SERVER_DEV_MODE=1
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_PATH:?Please set MODEL_PATH=/absolute/path/to/Qwen3.5-4B}"
SERVED="${SERVED_NAME:-qwen3.5-4b}"
PORT="${PORT:-8000}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-23000}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
MAX_SEQS="${MAX_SEQS:-64}"
PERF_MODE="${PERF_MODE:-throughput}"
API_KEY="${VLLM_API_KEY:-api-key-qwen3}"

echo "========================================================"
echo "  DynaGraph C5 anchor: Qwen3.5-4B TP2 TTFT-tuned"
echo "  model=${MODEL}  served=${SERVED}  port=${PORT}"
echo "  gpu_memory_utilization=${GPU_MEM}  max_model_len=${MAX_MODEL_LEN}"
echo "  max_num_batched_tokens=${MAX_BATCHED}  max_num_seqs=${MAX_SEQS}"
echo "  performance_mode=${PERF_MODE}  api_key=${API_KEY}"
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
  --enable-sleep-mode \
  --async-scheduling \
  --performance-mode "${PERF_MODE}" \
  --reasoning-parser qwen3 \
  --disable-custom-all-reduce
