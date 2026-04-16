#!/bin/bash
# SGLang alternative: Qwen3.5-4B TP2
# Use this only if you cannot install vLLM 0.17+. Note: SGLang support for
# the Qwen3.5 GDN / hybrid attention layers is still maturing; for the main
# paper results we recommend the vLLM path.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.5-4B bash scripts/start_sglang_qwen35_4b.sh

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL="${MODEL_PATH:?Please set MODEL_PATH=/absolute/path/to/Qwen3.5-4B}"
SERVED="${SERVED_NAME:-qwen3.5-4b}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-23000}"
API_KEY="${API_KEY:-api-key-qwen3}"

echo "========================================================"
echo "  SGLang Qwen3.5-4B TP2"
echo "  model=${MODEL}  served=${SERVED}  port=${PORT}"
echo "========================================================"

python -m sglang.launch_server \
  --model-path "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --tp 2 \
  --mem-fraction-static 0.85 \
  --context-length "${MAX_MODEL_LEN}" \
  --enable-torch-compile \
  --chunked-prefill-size 8192 \
  --disable-custom-all-reduce
