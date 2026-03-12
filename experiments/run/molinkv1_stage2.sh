#!/usr/bin/env bash
set -euo pipefail

# MoLink(v1) distributed pipeline stage2
# Run this on node6 (192.168.79.22)

VENV_PY=${VENV_PY:-"/home/sslab/MoLink/.venv-molink/bin/python"}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8002}

# Qwen2.5 default max context can be very large (e.g., 32768). Cap it to avoid
# KV cache init failures on smaller GPUs.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

GRPC_PORT=${GRPC_PORT:-50063}
INITIAL_PEER=${INITIAL_PEER:-"192.168.79.9:50061"}

START_LAYER=${START_LAYER:-22}
END_LAYER=${END_LAYER:--1}

CHUNKED_PREFILL=${CHUNKED_PREFILL:-0}

export HF_HOME=${HF_HOME:-"/home/sslab/hf_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/home/sslab/hf_cache"}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-"${HF_HOME}/modules"}
export PYTHONPATH="${HF_MODULES_CACHE}:${PYTHONPATH:-}"

# Avoid permission issues with default ~/.cache/vllm paths on some nodes.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-"${HF_HOME}/vllm_cache"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${HF_HOME}"}

mkdir -p "/home/sslab/MoLink/results" "$VLLM_CACHE_ROOT"

CHUNKED_FLAG="--no-enable-chunked-prefill"
if [[ "$CHUNKED_PREFILL" == "1" ]]; then
  CHUNKED_FLAG="--enable-chunked-prefill"
fi

exec "$VENV_PY" -m molinkv1.entrypoints.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --molink-enabled \
  --molink-grpc-port "$GRPC_PORT" \
  --molink-start-layer "$START_LAYER" \
  --molink-end-layer "$END_LAYER" \
  --molink-initial-peer "$INITIAL_PEER" \
  $CHUNKED_FLAG \
  2>&1 | tee -a "/home/sslab/MoLink/results/_server_molink_stage2_${PORT}.log"
