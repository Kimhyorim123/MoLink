#!/usr/bin/env bash
set -euo pipefail

# vLLM chunked prefill baseline: 3-GPU pipeline parallel (PP=3) over Ray.

HF_HOME=${HF_HOME:-/home/sslab/hf_cache}
TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
HF_MODULES_CACHE=${HF_MODULES_CACHE:-$HF_HOME/modules}

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}

NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/sslab/nccl/lib}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp7s0}
RAY_ADDRESS=${RAY_ADDRESS:-192.168.79.4:6379}

NCCL_NET=${NCCL_NET:-Socket}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}

mkdir -p "$HF_HOME"

# Make HuggingFace remote-code dynamic modules importable in this process too.
export PYTHONPATH="$HF_MODULES_CACHE:${PYTHONPATH:-}"

echo "[vllm] starting chunked PP=3 on $HOST:$PORT model=$MODEL"
HF_HOME="$HF_HOME" TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
  HF_MODULES_CACHE="$HF_MODULES_CACHE" \
  LD_LIBRARY_PATH="$NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}" \
  NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  NCCL_IB_DISABLE=1 \
  NCCL_NET="$NCCL_NET" \
  NCCL_DEBUG="$NCCL_DEBUG" \
  NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS" \
  RAY_ADDRESS="$RAY_ADDRESS" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --enable-chunked-prefill \
  --distributed-executor-backend ray \
  --pipeline-parallel-size 3 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL"
