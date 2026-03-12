#!/usr/bin/env bash
set -euo pipefail

# Build both images on the current node.
# Use DOCKER='sudo -n docker' if docker.sock permissions require it.

DOCKER=${DOCKER:-docker}

ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)

VLLM_TAG=${VLLM_TAG:-molinkexp/vllm:0.7.2}
VLLM_DOCKERFILE=${VLLM_DOCKERFILE:-}
MOLINK_TAG=${MOLINK_TAG:-molinkexp/molink:0.1-vllm0.11.2}

echo "[build] root=$ROOT_DIR"

echo "[build] vLLM image: $VLLM_TAG"
if [[ -z "$VLLM_DOCKERFILE" ]]; then
	case "$VLLM_TAG" in
		*":0.6.3"*) VLLM_DOCKERFILE="$ROOT_DIR/experiments/docker/vllm063/Dockerfile";;
		*":0.7.2"*) VLLM_DOCKERFILE="$ROOT_DIR/experiments/docker/vllm072/Dockerfile";;
		*":0.11.2"*) VLLM_DOCKERFILE="$ROOT_DIR/experiments/docker/vllm0112/Dockerfile";;
		*)
			echo "[build] unknown VLLM_TAG=$VLLM_TAG; set VLLM_DOCKERFILE explicitly" >&2
			exit 2
			;;
	esac
fi

$DOCKER build -t "$VLLM_TAG" -f "$VLLM_DOCKERFILE" "$ROOT_DIR"

echo "[build] MoLink image: $MOLINK_TAG"
$DOCKER build -t "$MOLINK_TAG" -f "$ROOT_DIR/experiments/docker/molink/Dockerfile" "$ROOT_DIR"

echo "[build] done"
