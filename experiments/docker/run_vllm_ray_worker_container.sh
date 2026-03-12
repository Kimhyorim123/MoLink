#!/usr/bin/env bash
set -euo pipefail

# Start a Ray worker inside the vLLM 0.6.3 image container.

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-molinkexp/vllm:0.6.3}
NAME=${NAME:-rayworker_vllm063}
NODE_IP=${NODE_IP:?set NODE_IP}
HEAD=${HEAD:-192.168.79.4:6380}
RAY_TMPDIR=${RAY_TMPDIR:-/home/sslab/ray_tmp_docker}

HF_CACHE=${HF_CACHE:-/home/sslab/hf_cache}

mkdir -p "$RAY_TMPDIR"
mkdir -p "$HF_CACHE"

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

exec $DOCKER run -d --name "$NAME" \
  --network host \
  --gpus all \
  -v "$RAY_TMPDIR:/ray_tmp" \
  -v "$HF_CACHE:/hf_cache" \
  -e HF_HOME=/hf_cache \
  -e TRANSFORMERS_CACHE=/hf_cache \
  -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
  -e HF_MODULES_CACHE=/hf_cache/modules \
  "$IMAGE" bash -lc "ray stop -f >/dev/null 2>&1 || true; \
    export RAY_TMPDIR=/ray_tmp; \
    ray start --address='$HEAD' --node-ip-address='$NODE_IP'; \
    tail -f /dev/null"
