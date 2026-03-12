#!/usr/bin/env bash
set -euo pipefail

# Shape egress traffic to a set of peer IPs only.
#
# Adds:
# - HTB root qdisc
# - One shaped class (rate limit + netem delay)
# - One fast default class for everything else
# - u32 filters matching dst IPs to the shaped class
#
# This is meant to be run on *each node* so that delay is applied in both
# directions. If you want RTT=30ms, use one-way delay 15ms here.
#
# Usage:
#   sudo ./shape_peers.sh --peers 192.168.0.4,192.168.0.6 \
#        --rate-mbps 100 --delay-ms 15 [--iface eth0]
#

usage() {
  cat <<'EOF'
Usage: shape_peers.sh --peers ip1,ip2,... --rate-mbps N --delay-ms N [--iface IFACE] [--dry-run]

Args:
  --peers       Comma-separated peer IPs to shape (dst match).
  --rate-mbps   Egress bandwidth cap in Mbps (e.g., 100).
  --delay-ms    One-way delay in ms to add (e.g., 15 for ~30ms RTT when applied on both ends).
  --iface       Network interface to apply qdisc on. If omitted, uses the default route interface.
  --dry-run     Print commands without executing.

Notes:
  - This shapes only egress. Apply on all nodes for symmetric RTT.
  - Run ./clear_shape.sh to remove.
EOF
}

PEERS=""
RATE_MBPS=""
DELAY_MS=""
IFACE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --peers) PEERS="$2"; shift 2;;
    --rate-mbps) RATE_MBPS="$2"; shift 2;;
    --delay-ms) DELAY_MS="$2"; shift 2;;
    --iface) IFACE="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$PEERS" || -z "$RATE_MBPS" || -z "$DELAY_MS" ]]; then
  usage
  exit 1
fi

if [[ -z "$IFACE" ]]; then
  IFACE=$(ip route show default 0.0.0.0/0 | awk '{print $5; exit}')
fi

if [[ -z "$IFACE" ]]; then
  echo "Could not determine default interface; pass --iface"
  exit 1
fi

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "+ $*"
  else
    eval "$@"
  fi
}

echo "[shape] iface=$IFACE peers=$PEERS rate=${RATE_MBPS}mbit delay=${DELAY_MS}ms"

# Clean any existing qdisc we own (ignore errors).
run "tc qdisc del dev '$IFACE' root 2>/dev/null || true"

# Root HTB with default class 1:30 (fast).
run "tc qdisc add dev '$IFACE' root handle 1: htb default 30"

# Fast class (not shaped) - set very high rate.
run "tc class add dev '$IFACE' parent 1: classid 1:30 htb rate 10000mbit ceil 10000mbit"

# Shaped class 1:10.
run "tc class add dev '$IFACE' parent 1: classid 1:10 htb rate ${RATE_MBPS}mbit ceil ${RATE_MBPS}mbit"

# Attach netem to shaped class.
run "tc qdisc add dev '$IFACE' parent 1:10 handle 10: netem delay ${DELAY_MS}ms"

# Filters: match dst IPs to shaped class.
IFS=',' read -r -a peer_arr <<< "$PEERS"
for peer in "${peer_arr[@]}"; do
  peer_trim=$(echo "$peer" | xargs)
  if [[ -z "$peer_trim" ]]; then
    continue
  fi
  run "tc filter add dev '$IFACE' protocol ip parent 1: prio 1 u32 match ip dst ${peer_trim}/32 flowid 1:10"
  echo "[shape] matched dst ${peer_trim} -> class 1:10"
done

echo "[shape] done. Verify with: tc -s qdisc show dev $IFACE && tc -s class show dev $IFACE"