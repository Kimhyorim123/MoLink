#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"/home/sslab/MoLink"}
REMOTE=${REMOTE:-"origin"}
BRANCH=${BRANCH:-""}
COMMIT=${COMMIT:-""}
LOCAL_NODE=${LOCAL_NODE:-"node5"}
NODES=${NODES:-"node2,node5,node6"}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes"}

usage() {
  cat <<'EOF'
Sync the MoLink repo across node2/node5/node6.

Usage:
  ./experiments/run/sync_nodes.sh [--branch <name>] [--commit <sha>] [--remote <name>]

Behavior:
  - If --commit is given, all nodes checkout that exact commit in detached HEAD.
  - Otherwise, all nodes checkout --branch and pull --ff-only from the remote.

Examples:
  ./experiments/run/sync_nodes.sh --branch feat/phase-metadata
  ./experiments/run/sync_nodes.sh --commit 78aaabc
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      BRANCH=${2:-}
      shift 2
      ;;
    --commit)
      COMMIT=${2:-}
      shift 2
      ;;
    --remote)
      REMOTE=${2:-}
      shift 2
      ;;
    --nodes)
      NODES=${2:-}
      shift 2
      ;;
    --local-node)
      LOCAL_NODE=${2:-}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$BRANCH" && -n "$COMMIT" ]]; then
  echo "Use either --branch or --commit, not both." >&2
  exit 1
fi

if [[ -z "$BRANCH" && -z "$COMMIT" ]]; then
  BRANCH=$(git -C "$ROOT_DIR" branch --show-current)
fi

IFS=',' read -r -a NODE_LIST <<< "$NODES"

sync_branch_cmd() {
  local branch=$1
  cat <<EOF
set -euo pipefail
cd "$ROOT_DIR"
git fetch "$REMOTE"
git checkout "$branch"
git pull --ff-only "$REMOTE" "$branch"
printf "%s %s %s\n" "\$(hostname)" "\$(git branch --show-current)" "\$(git rev-parse --short HEAD)"
EOF
}

sync_commit_cmd() {
  local commit=$1
  cat <<EOF
set -euo pipefail
cd "$ROOT_DIR"
git fetch "$REMOTE"
git checkout "$commit"
printf "%s DETACHED %s\n" "\$(hostname)" "\$(git rev-parse --short HEAD)"
EOF
}

run_on_node() {
  local node=$1
  local cmd=$2
  echo "[sync] $node"
  if [[ "$node" == "$LOCAL_NODE" ]]; then
    bash -lc "$cmd"
  else
    ssh $SSH_OPTS "$node" "bash -lc '$cmd'"
  fi
}

echo "[sync] root=$ROOT_DIR remote=$REMOTE"
if [[ -n "$COMMIT" ]]; then
  echo "[sync] target commit=$COMMIT"
  for node in "${NODE_LIST[@]}"; do
    run_on_node "$node" "$(sync_commit_cmd "$COMMIT")"
  done
else
  echo "[sync] target branch=$BRANCH"
  for node in "${NODE_LIST[@]}"; do
    run_on_node "$node" "$(sync_branch_cmd "$BRANCH")"
  done
fi
