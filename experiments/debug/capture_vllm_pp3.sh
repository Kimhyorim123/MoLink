#!/usr/bin/env bash
# Capture evidence for vLLM PP=3 bottlenecks across node4/node5/node6.
# Collects:
# - NIC tx/rx byte deltas (enp7s0)
# - GPU util via nvidia-smi dmon (35s)
# - ss snapshots
# - vLLM API container tail logs

set -Eeuo pipefail

IFACE="${IFACE:-enp7s0}"
NODE_LOCAL="${NODE_LOCAL:-node5}"
NODE4="${NODE4:-node4}"
NODE6="${NODE6:-node6}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:8000/v1/completions}"
VLLM_CONTAINER="${VLLM_CONTAINER:-vllm072_api}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
USE_LOADGEN="${USE_LOADGEN:-0}"
PROMPT_TOKENS="${PROMPT_TOKENS:-}"
MAX_TOKENS="${MAX_TOKENS:-512}"

OUT_DIR="${1:-results/_debug_vllmpp3_capture_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

log() {
  printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT_DIR/run.log" >/dev/null
}

on_err() {
  local exit_code=$?
  log "ERROR exit_code=$exit_code line=$1 cmd=$2"
  exit "$exit_code"
}
trap 'on_err "$LINENO" "$BASH_COMMAND"' ERR

die() {
  log "FATAL: $*"
  exit 1
}

read_bytes_local() {
  local direction=$1
  cat "/sys/class/net/${IFACE}/statistics/${direction}_bytes" 2>/dev/null || echo 0
}

read_bytes_remote() {
  local node=$1 direction=$2
  ssh -o BatchMode=yes -o ConnectTimeout=3 "$node" "cat /sys/class/net/${IFACE}/statistics/${direction}_bytes" 2>/dev/null || echo 0
}

write_prepost() {
  local node=$1 tag=$2
  local tx rx
  if [[ "$node" == "$NODE_LOCAL" ]]; then
    tx=$(read_bytes_local tx)
    rx=$(read_bytes_local rx)
  else
    tx=$(read_bytes_remote "$node" tx)
    rx=$(read_bytes_remote "$node" rx)
  fi
  printf '%s\n' "$tx" > "$OUT_DIR/${node}_tx_${tag}.txt"
  printf '%s\n' "$rx" > "$OUT_DIR/${node}_rx_${tag}.txt"
}

start_dmon() {
  local node=$1
  local out_txt="$OUT_DIR/${node}_dmon_u.txt"
  local err_txt="$OUT_DIR/${node}_dmon_err.txt"
  : > "$out_txt"; : > "$err_txt"

  if [[ "$node" == "$NODE_LOCAL" ]]; then
    timeout 35 nvidia-smi dmon -s u -d 1 >"$out_txt" 2>"$err_txt" &
  else
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$node" "timeout 35 nvidia-smi dmon -s u -d 1" >"$out_txt" 2>"$err_txt" &
  fi
  echo $!
}

capture_ss() {
  local node=$1
  local out_txt="$OUT_DIR/${node}_ss.txt"
  local err_txt="$OUT_DIR/${node}_ss_err.txt"
  : > "$out_txt"; : > "$err_txt"

  local cmd="sudo ss -tpn | egrep 'ESTAB|192\\.168\\.79\\.(9|22|4)' | head -n 200"
  if [[ "$node" == "$NODE_LOCAL" ]]; then
    bash -lc "$cmd" >"$out_txt" 2>"$err_txt" || true
  else
    ssh -o BatchMode=yes -o ConnectTimeout=3 "$node" "$cmd" >"$out_txt" 2>"$err_txt" || true
  fi
}

log "OUT_DIR=$OUT_DIR"
log "Pre network bytes"
write_prepost node4 pre
write_prepost node5 pre
write_prepost node6 pre

log "Start GPU dmon captures (35s)"
P4=$(start_dmon node4)
P5=$(start_dmon node5)
P6=$(start_dmon node6)

log "Trigger one streaming request (~30s)"
if [[ "$USE_LOADGEN" == "1" ]]; then
  # Use the project's loadgen to generate an exact token-count prompt (if requested)
  # and to produce TTFT/TPOT/E2E numbers aligned with Table2 settings.
  LOADGEN_OUT="$OUT_DIR/loadgen.csv"
  LOADGEN_ARGS=(
    --system vllm
    --mode openai-completions
    --base-url http://127.0.0.1:8000
    --model "$MODEL"
    --rate-rps 1
    --num-requests 1
    --max-tokens "$MAX_TOKENS"
    --temperature 0
    --timeout-s 180
    --out-csv "$LOADGEN_OUT"
  )
  if [[ -n "$PROMPT_TOKENS" ]]; then
    LOADGEN_ARGS+=(--prompt-tokens "$PROMPT_TOKENS" --tokenizer "$MODEL")
  else
    LOADGEN_ARGS+=(--prompt "Write a detailed 500-word explanation of what pipeline parallelism is.")
  fi

  python3 experiments/loadgen/openai_stream_loadgen.py "${LOADGEN_ARGS[@]}" | tee "$OUT_DIR/loadgen.stdout" >/dev/null || true
else
  REQ_JSON=$(python3 - <<PY
import json, os
print(json.dumps({
  'model': os.environ.get('MODEL','Qwen/Qwen2.5-7B-Instruct'),
  'prompt': 'Write a detailed 500-word explanation of what pipeline parallelism is.',
  'max_tokens': int(os.environ.get('MAX_TOKENS','512')),
  'temperature': 0,
  'stream': True,
}))
PY
)
  printf '%s' "$REQ_JSON" > "$OUT_DIR/request.json"
  (timeout 60 curl --max-time 60 -sS -N "$VLLM_URL" -H 'Content-Type: application/json' -d "$REQ_JSON" >/dev/null) || true
fi

log "Capture ss snapshots"
capture_ss node5
capture_ss node4
capture_ss node6

log "Wait GPU captures"
wait "$P4" 2>/dev/null || true
wait "$P5" 2>/dev/null || true
wait "$P6" 2>/dev/null || true

log "Post network bytes"
write_prepost node4 post
write_prepost node5 post
write_prepost node6 post

python3 - <<PY
import pathlib
p=pathlib.Path("$OUT_DIR")
for n in ['node4','node5','node6']:
    tx0=int((p/f"{n}_tx_pre.txt").read_text().strip() or 0)
    rx0=int((p/f"{n}_rx_pre.txt").read_text().strip() or 0)
    tx1=int((p/f"{n}_tx_post.txt").read_text().strip() or 0)
    rx1=int((p/f"{n}_rx_post.txt").read_text().strip() or 0)
    (p/f"{n}_net_delta.txt").write_text(f"tx_delta={tx1-tx0}\nrx_delta={rx1-rx0}\n")
print('ok')
PY

(docker logs --tail 500 "$VLLM_CONTAINER" > "$OUT_DIR/${VLLM_CONTAINER}_tail500.log") 2> "$OUT_DIR/${VLLM_CONTAINER}_tail500_err.txt" || true

log "DONE"
echo "$OUT_DIR"
