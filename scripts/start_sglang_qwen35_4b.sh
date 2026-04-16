#!/bin/bash
# SGLang alternative (>= 0.5.9): Qwen3.5-4B on two GPUs (TP=2),
# with priority scheduling enabled (required for Phase 3).
#
# Download the model: https://huggingface.co/Qwen/Qwen3.5-4B
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_sglang_qwen35_4b.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL="${MODEL_PATH:?Set MODEL_PATH=/absolute/path/to/Qwen3.5-4B}"
SERVED="${SERVED_NAME:-qwen3.5-4b}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-api-key-qwen3}"

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host 0.0.0.0 --port "${PORT}" --api-key "${API_KEY}" \
  --tp 2 \
  --mem-fraction-static 0.85 \
  --context-length 23000 \
  --chunked-prefill-size 8192 \
  --enable-priority-scheduling \
  --schedule-low-priority-values-first \
  --disable-custom-all-reduce
