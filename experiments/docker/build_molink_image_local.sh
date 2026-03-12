#!/usr/bin/env bash
set -euo pipefail

# Build ONLY the MoLink image on the current node.
# Use DOCKER='sudo -n docker' if docker.sock permissions require it.

DOCKER=${DOCKER:-docker}
ROOT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
MOLINK_TAG=${MOLINK_TAG:-molinkexp/molink:0.1-vllm0.11.2}

echo "[build] root=$ROOT_DIR"
echo "[build] MoLink image: $MOLINK_TAG"

$DOCKER build -t "$MOLINK_TAG" -f "$ROOT_DIR/experiments/docker/molink/Dockerfile" "$ROOT_DIR"

echo "[build] done"
