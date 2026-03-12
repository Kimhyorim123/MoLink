#!/usr/bin/env bash
set -euo pipefail

# Run vLLM OpenAI API server with chunked prefill enabled (0.11.2).

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-molinkexp/vllm:0.11.2}
NAME=${NAME:-vllm0112_chunked_api}
DETACH=${DETACH:-0}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
RAY_ADDRESS=${RAY_ADDRESS:-192.168.79.4:6380}

RAY_TMPDIR=${RAY_TMPDIR:-/home/sslab/ray_tmp_docker}
HF_CACHE=${HF_CACHE:-/home/sslab/hf_cache}

# NCCL settings (multi-node communication).
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp7s0}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_DEBUG=${NCCL_DEBUG:-}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-}

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

RUN_ARGS=(--name "$NAME" --network host --gpus all)
if [[ "$DETACH" == "1" ]]; then
  RUN_ARGS=(-d "${RUN_ARGS[@]}")
else
  RUN_ARGS=(--rm "${RUN_ARGS[@]}")
fi

exec $DOCKER run "${RUN_ARGS[@]}" \
  -v "$HF_CACHE:/hf_cache" \
  -v "$RAY_TMPDIR:/ray_tmp" \
  -e HF_HOME=/hf_cache \
  -e TRANSFORMERS_CACHE=/hf_cache \
  -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
  -e HF_MODULES_CACHE=/hf_cache/modules \
  -e RAY_TMPDIR=/ray_tmp \
  -e RAY_ADDRESS="$RAY_ADDRESS" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  -e NCCL_DEBUG="$NCCL_DEBUG" \
  -e NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS" \
  "$IMAGE" \
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --distributed-executor-backend ray \
    --pipeline-parallel-size 3 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-chunked-prefill
