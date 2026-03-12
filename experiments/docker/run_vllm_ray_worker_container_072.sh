#!/usr/bin/env bash
set -euo pipefail

# Start a Ray worker inside the vLLM 0.7.2 image container.

DOCKER=${DOCKER:-docker}
IMAGE=${IMAGE:-molinkexp/vllm:0.7.2}
NAME=${NAME:-rayworker_vllm072}
NODE_IP=${NODE_IP:?set NODE_IP}
HEAD=${HEAD:-192.168.79.4:6380}
RAY_TMPDIR=${RAY_TMPDIR:-/home/sslab/ray_tmp_docker}
SHM_SIZE=${SHM_SIZE:-10g}

# NCCL settings (multi-node communication).
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp7s0}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_DEBUG=${NCCL_DEBUG:-}
NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-}

HF_CACHE=${HF_CACHE:-/home/sslab/hf_cache}

# vLLM cache root (torch.compile, etc). Keep it off the container overlay FS.
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/hf_cache/vllm_cache}

mkdir -p "$RAY_TMPDIR"
mkdir -p "$HF_CACHE"

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

exec $DOCKER run -d --name "$NAME" \
  --network host \
  --shm-size "$SHM_SIZE" \
  --gpus all \
  -v "$RAY_TMPDIR:/ray_tmp" \
  -v "$HF_CACHE:/hf_cache" \
  -e HF_HOME=/hf_cache \
  -e TRANSFORMERS_CACHE=/hf_cache \
  -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
  -e HF_MODULES_CACHE=/hf_cache/modules \
  -e VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" \
  -e NCCL_DEBUG="$NCCL_DEBUG" \
  -e NCCL_DEBUG_SUBSYS="$NCCL_DEBUG_SUBSYS" \
  "$IMAGE" bash -lc "set -euo pipefail; \
    ray stop -f >/dev/null 2>&1 || true; \
    export RAY_TMPDIR=/ray_tmp; \
    ray start --address='$HEAD' --node-ip-address='$NODE_IP' --block"
