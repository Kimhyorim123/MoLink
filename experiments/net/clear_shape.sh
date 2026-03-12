#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: clear_shape.sh [--iface IFACE]

Removes the root qdisc on IFACE (undoes shape_peers.sh).
EOF
}

IFACE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface) IFACE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$IFACE" ]]; then
  IFACE=$(ip route show default 0.0.0.0/0 | awk '{print $5; exit}')
fi

if [[ -z "$IFACE" ]]; then
  echo "Could not determine default interface; pass --iface"
  exit 1
fi

echo "[clear] removing qdisc root on $IFACE"
(tc qdisc del dev "$IFACE" root 2>/dev/null || true)
echo "[clear] done"