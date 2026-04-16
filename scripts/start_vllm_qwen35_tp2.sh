#!/bin/bash
# Unified Qwen3.5 TP2 launcher (4B / 9B / 35B).
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.5-9B bash scripts/start_vllm_qwen35_tp2.sh 9B [PORT]
#   MODEL_PATH=/path/to/Qwen3.5-35B-A3B-FP8 bash scripts/start_vllm_qwen35_tp2.sh 35B
#
# 35B requires MoE model files (Qwen3.5-35B-A3B-FP8). On dual 4090 it must use
# max_num_seqs=8 and performance-mode=interactivity for stability; this is a
# characterisation config, not the main-paper setting.

set -euo pipefail

MODEL_SIZE=${1:-9B}
PORT=${2:-8000}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_SERVER_DEV_MODE=1
export SAFETENSORS_FAST_GPU=1

MODEL="${MODEL_PATH:?Please set MODEL_PATH=/absolute/path/to/model}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-23000}"
MAX_BATCHED="${MAX_BATCHED:-23000}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
API_KEY="${VLLM_API_KEY:-api-key-qwen3}"

SERVED=""
MAX_SEQS="64"
PERF_MODE="balanced"
EXTRA_ARGS=()

case "${MODEL_SIZE}" in
  35B|35b)
    SERVED="${SERVED_NAME:-qwen3.5-35b}"
    MAX_SEQS="${MAX_SEQS:-8}"
    PERF_MODE="${PERF_MODE:-interactivity}"
    EXTRA_ARGS=(
      --kv-cache-dtype "${KV_CACHE_DTYPE}"
      --enable-prefix-caching
      --enable-chunked-prefill
      --enable-sleep-mode
      --async-scheduling
      --enable-expert-parallel
      --all2all-backend allgather_reducescatter
      --performance-mode "${PERF_MODE}"
      --reasoning-parser qwen3
      --disable-custom-all-reduce
    )
    ;;
  9B|9b)
    SERVED="${SERVED_NAME:-qwen3.5-9b}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-10000}"
    MAX_BATCHED="${MAX_BATCHED:-10000}"
    MAX_SEQS="${MAX_SEQS:-64}"
    PERF_MODE="${PERF_MODE:-balanced}"
    EXTRA_ARGS=(
      --kv-cache-dtype fp8
      --enable-prefix-caching
      --enable-chunked-prefill
      --enable-sleep-mode
      --async-scheduling
      --performance-mode "${PERF_MODE}"
      --reasoning-parser qwen3
      --disable-custom-all-reduce
    )
    ;;
  4B|4b)
    SERVED="${SERVED_NAME:-qwen3.5-4b}"
    MAX_SEQS="${MAX_SEQS:-64}"
    PERF_MODE="${PERF_MODE:-balanced}"
    EXTRA_ARGS=(
      --kv-cache-dtype fp8
      --enable-prefix-caching
      --enable-chunked-prefill
      --enable-sleep-mode
      --async-scheduling
      --performance-mode "${PERF_MODE}"
      --reasoning-parser qwen3
      --disable-custom-all-reduce
    )
    ;;
  *)
    echo "Unsupported model size: ${MODEL_SIZE}"
    echo "Use one of: 4B, 9B, 35B"
    exit 1
    ;;
esac

echo "========================================================"
echo "  Qwen3.5 ${MODEL_SIZE} TP2"
echo "  model=${MODEL}  served=${SERVED}  port=${PORT}"
echo "  max_num_seqs=${MAX_SEQS}  performance_mode=${PERF_MODE}"
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
  "${EXTRA_ARGS[@]}"
