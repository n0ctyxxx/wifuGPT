#!/usr/bin/env bash
set -euo pipefail

# Load .env
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE" >&2
    exit 1
fi

# From .env
MODEL="${VLLM_MODEL:?Set VLLM_MODEL in .env}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

# 4x H100 configuration
TENSOR_PARALLEL_SIZE=4
GPU_MEMORY_UTILIZATION=0.92
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=512

echo "Starting vLLM server..."
echo "  Model: $MODEL"
echo "  GPUs: $TENSOR_PARALLEL_SIZE x H100"
echo "  Max model length: $MAX_MODEL_LEN"
echo "  Max concurrent sequences: $MAX_NUM_SEQS"
echo "  Endpoint: http://$HOST:$PORT/v1"

vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests
