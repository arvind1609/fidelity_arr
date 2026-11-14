#!/bin/bash

# --- Configuration ---
MODEL_NAME="google/gemma-4-E4B-it"
PORT=8000
MAX_LEN=128000
MEM_UTIL=0.85
NUM_GPUS=2

echo "Starting single vLLM instance for $MODEL_NAME across $NUM_GPUS GPUs (tensor-parallel)..."

# Activate the environment
if [ -d ".venv-vllm" ]; then
    source .venv-vllm/bin/activate
fi

mkdir -p logs
LOG="logs/vllm_${PORT}.log"

CUDA_VISIBLE_DEVICES=1,2 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --tensor-parallel-size $NUM_GPUS \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $MEM_UTIL \
    --dtype bfloat16 \
    --port $PORT \
    --enable-chunked-prefill \
    --trust-remote-code 2>&1 | tee "$LOG"
