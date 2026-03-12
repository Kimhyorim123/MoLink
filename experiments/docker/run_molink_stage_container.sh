#!/usr/bin/env bash
set -euo pipefail

# Generic MoLink(v1) stage runner in Docker.
# You typically run:
#  - node4: STAGE=0 ...
#  - node5: STAGE=1 ...
#  - node6: STAGE=2 ...

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-molinkexp/molink:0.1-vllm0.11.2}
NAME=${NAME:-molink_stage}
DETACH=${DETACH:-0}

STAGE=${STAGE:?set STAGE=0|1|2}
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
HOST=${HOST:-0.0.0.0}

# Qwen2.5 default max seq len can be very large (e.g., 32768) and may exceed
# available KV cache memory on smaller GPUs. Our experiments don't require that
# much context, so cap it by default.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

HF_CACHE=${HF_CACHE:-/home/sslab/hf_cache}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/hf_cache/vllm_cache}

# Defaults per stage
if [[ "$STAGE" == "0" ]]; then
  PORT=${PORT:-8000}
  GRPC_PORT=${GRPC_PORT:-50061}
  START_LAYER=${START_LAYER:-0}
  END_LAYER=${END_LAYER:-11}
  INITIAL_PEER=""
elif [[ "$STAGE" == "1" ]]; then
  PORT=${PORT:-8001}
  GRPC_PORT=${GRPC_PORT:-50062}
  START_LAYER=${START_LAYER:-11}
  END_LAYER=${END_LAYER:-22}
  INITIAL_PEER=${INITIAL_PEER:-"192.168.79.9:50061"}
elif [[ "$STAGE" == "2" ]]; then
  PORT=${PORT:-8002}
  GRPC_PORT=${GRPC_PORT:-50063}
  START_LAYER=${START_LAYER:-22}
  END_LAYER=${END_LAYER:--1}
  INITIAL_PEER=${INITIAL_PEER:-"192.168.79.9:50061"}
else
  echo "Invalid STAGE: $STAGE" >&2
  exit 1
fi

CHUNKED_PREFILL=${CHUNKED_PREFILL:-0}
CHUNKED_FLAG="--no-enable-chunked-prefill"
if [[ "$CHUNKED_PREFILL" == "1" ]]; then
  CHUNKED_FLAG="--enable-chunked-prefill"
fi

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

ARGS=(
  python3 -m molinkv1.entrypoints.api_server
  --host "$HOST"
  --port "$PORT"
  --model "$MODEL"
  --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN"
  --molink-enabled
  --molink-grpc-port "$GRPC_PORT"
  --molink-start-layer "$START_LAYER"
  --molink-end-layer "$END_LAYER"
  $CHUNKED_FLAG
)
if [[ -n "$INITIAL_PEER" ]]; then
  ARGS+=(--molink-initial-peer "$INITIAL_PEER")
fi

RUN_ARGS=(--name "$NAME" --network host --gpus all)
if [[ "$DETACH" == "1" ]]; then
  RUN_ARGS=(-d "${RUN_ARGS[@]}")
else
  RUN_ARGS=(--rm "${RUN_ARGS[@]}")
fi

exec $DOCKER run "${RUN_ARGS[@]}" \
  -v "$HF_CACHE:/hf_cache" \
  -e HF_HOME=/hf_cache \
  -e TRANSFORMERS_CACHE=/hf_cache \
  -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
  -e HF_MODULES_CACHE=/hf_cache/modules \
  -e XDG_CACHE_HOME=/hf_cache \
  -e VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  "$IMAGE" "${ARGS[@]}"
