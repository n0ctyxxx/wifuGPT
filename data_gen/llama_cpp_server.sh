#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# llama-cpp-python OpenAI-compatible server for GGUF models
# Serves the model with GPU offloading across all available GPUs
# ============================================================================

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

# Defaults
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
MODEL_PATH="${GGUF_MODEL_PATH:?Set GGUF_MODEL_PATH in .env to the path of your .gguf file}"

# Detect GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo "0")
TENSOR_SPLIT="$(python3 -c "print(','.join(['1']*${GPU_COUNT}))" 2>/dev/null || echo "1")"

echo "Starting llama-cpp-python server..."
echo "  Model:        $MODEL_PATH"
echo "  Host:         $HOST:$PORT"
echo "  GPUs:         $GPU_COUNT"
echo "  Tensor split: $TENSOR_SPLIT"
echo "  Context:      8192"
echo "  Endpoint:     http://$HOST:$PORT/v1"

python3 -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --n_gpu_layers -1 \
    --n_ctx 8192 \
    --n_batch 2048 \
    --chat_format chatml \
    --split_mode 1 \
    --tensor_split $TENSOR_SPLIT
