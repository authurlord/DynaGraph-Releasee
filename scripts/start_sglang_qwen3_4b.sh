#!/bin/bash
# SGLang alternative: Qwen3-4B-Instruct-2507 TP2.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507 bash scripts/start_sglang_qwen3_4b.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL="${MODEL_PATH:?Please set MODEL_PATH=/absolute/path/to/Qwen3-4B-Instruct-2507}"
SERVED="${SERVED_NAME:-qwen3-4b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-23000}"
API_KEY="${API_KEY:-api-key-qwen3}"

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --tp 2 \
  --mem-fraction-static 0.85 \
  --context-length "${MAX_MODEL_LEN}" \
  --chunked-prefill-size 8192 \
  --disable-custom-all-reduce
