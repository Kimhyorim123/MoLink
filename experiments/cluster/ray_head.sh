#!/usr/bin/env bash
set -euo pipefail

# Start Ray head on this node.
# Run this on node5 (192.168.79.4).

usage() {
  cat <<'EOF'
Usage: ray_head.sh --node-ip IP [--port 6379]

Example:
  ./ray_head.sh --node-ip 192.168.79.4 --port 6379
EOF
}

NODE_IP=""
PORT="6379"

RAY_BIN=${RAY_BIN:-}
if [[ -z "$RAY_BIN" ]]; then
  if [[ -x "$HOME/.local/bin/ray" ]]; then
    RAY_BIN="$HOME/.local/bin/ray"
  else
    RAY_BIN="ray"
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --node-ip) NODE_IP="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$NODE_IP" ]]; then
  echo "--node-ip is required"
  exit 1
fi

# Ensure all Ray worker processes inherit a consistent NCCL.
# Put the same libnccl.so.2 on every node under /home/sslab/nccl/lib.
NCCL_LIB_DIR=${NCCL_LIB_DIR:-/home/sslab/nccl/lib}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp7s0}
RAY_TMPDIR=${RAY_TMPDIR:-/home/sslab/ray_tmp}

# Ensure Ray workers can import HuggingFace remote-code dynamic modules
# (e.g., `transformers_modules.*`) during Ray object deserialization.
HF_HOME=${HF_HOME:-/home/sslab/hf_cache}
HF_MODULES_CACHE=${HF_MODULES_CACHE:-$HF_HOME/modules}

mkdir -p "$RAY_TMPDIR"
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:${LD_LIBRARY_PATH:-}"
export NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
export NCCL_IB_DISABLE=1
export RAY_TMPDIR="$RAY_TMPDIR"

mkdir -p "$HF_HOME"
export HF_HOME="$HF_HOME"
export HF_MODULES_CACHE="$HF_MODULES_CACHE"
export PYTHONPATH="$HF_MODULES_CACHE:${PYTHONPATH:-}"

"$RAY_BIN" stop -f >/dev/null 2>&1 || true

echo "[ray] starting head: $NODE_IP:$PORT"
"$RAY_BIN" start --head --node-ip-address="$NODE_IP" --port="$PORT" --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "[ray] status:"
"$RAY_BIN" status || true
