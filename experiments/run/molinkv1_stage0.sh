#!/usr/bin/env bash
set -euo pipefail

# MoLink(v1) distributed pipeline stage0 (HTTP ingress)
# Run this on node5 (192.168.79.4) as the head stage

VENV_PY=${VENV_PY:-"/home/sslab/MoLink/.venv-molink/bin/python"}
ROOT_DIR=${ROOT_DIR:-"/home/sslab/MoLink-deploy"}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}

# Qwen2.5 default max context can be very large (e.g., 32768). Cap it to avoid
# KV cache init failures on smaller GPUs.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

# gRPC port for pipeline communication
GRPC_PORT=${GRPC_PORT:-50061}

# Layer partition (end is exclusive; -1 means "to end")
START_LAYER=${START_LAYER:-0}
END_LAYER=${END_LAYER:-11}

# For fair comparison vs vLLM baseline, default chunked-prefill OFF.
# Set CHUNKED_PREFILL=1 to enable.
CHUNKED_PREFILL=${CHUNKED_PREFILL:-0}

# HF caches (avoid root-owned defaults)
export HF_HOME=${HF_HOME:-"/home/sslab/hf_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"/home/sslab/hf_cache"}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-"${HF_HOME}/modules"}
# Prefer the checked-out runtime tree over any editable install in the venv.
export PYTHONPATH="${ROOT_DIR}:${HF_MODULES_CACHE}:${PYTHONPATH:-}"

# Avoid permission issues with default ~/.cache/vllm paths on some nodes.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-"${ROOT_DIR}/.runtime/vllm_cache"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${HF_HOME}"}
export TMPDIR=${TMPDIR:-"${ROOT_DIR}/.runtime/tmp"}

RESULTS_DIR=${RESULTS_DIR:-"${ROOT_DIR}/results"}
mkdir -p "$RESULTS_DIR" "$VLLM_CACHE_ROOT" "$TMPDIR"

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
  $CHUNKED_FLAG \
  2>&1 | tee -a "$RESULTS_DIR/_server_molink_stage0_${PORT}.log"
