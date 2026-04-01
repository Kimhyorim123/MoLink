#!/usr/bin/env bash
set -euo pipefail

# Run a 3-way MoLink comparison with identical workload settings:
#   1) baseline
#   2) fixed 2MB chunk
#   3) jit+fallback
#
# The script:
# - syncs a target commit to /home/sslab/MoLink-deploy on node2/node5/node6
# - restarts the 3 pipeline stages
# - optionally shapes the inter-node network
# - runs the existing loadgen sweep
# - writes per-system CSVs
# - produces combined summary.csv and p95 plots for each scenario

ROOT_DIR=${ROOT_DIR:-"/home/sslab/MoLink-deploy"}
DEV_ROOT=${DEV_ROOT:-"/home/sslab/MoLink"}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
VENV_PY=${VENV_PY:-"/home/sslab/MoLink/.venv-molink/bin/python"}

NODE5_IP=${NODE5_IP:-"192.168.79.4"}
NODE2_IP=${NODE2_IP:-"192.168.79.20"}
NODE6_IP=${NODE6_IP:-"192.168.79.22"}

BASELINE_COMMIT=${BASELINE_COMMIT:-"78aaabc"}
FIXED_COMMIT=${FIXED_COMMIT:-"d39d54e"}
JIT_COMMIT=${JIT_COMMIT:-"c204ea8"}

RATES=${RATES:-"0.1,0.2,0.3,0.7"}
NUM_REQUESTS=${NUM_REQUESTS:-"50"}
PROMPT_TOKENS=${PROMPT_TOKENS:-"512"}
MAX_TOKENS=${MAX_TOKENS:-"128"}
TEMP=${TEMP:-"0"}
MAX_IN_FLIGHT=${MAX_IN_FLIGHT:-"32"}
VARIANTS=${VARIANTS:-"baseline,fixed,jit"}
SCENARIOS=${SCENARIOS:-"unshaped,shaped30"}

SHAPED_BW_MBPS=${SHAPED_BW_MBPS:-"100"}
SHAPED_RTT_MS=${SHAPED_RTT_MS:-"30"}
UNSHAPED_BW_MBPS=${UNSHAPED_BW_MBPS:-"0"}
UNSHAPED_RTT_MS=${UNSHAPED_RTT_MS:-"0"}

OUT_BASE=${OUT_BASE:-"$ROOT_DIR/results/molink_compare_3way_$(date +%Y%m%d_%H%M%S)"}
SSH_OPTS=${SSH_OPTS:-"-o BatchMode=yes"}

usage() {
  cat <<'EOF'
Usage: molink_compare_3way.sh [--out-base DIR]

Environment overrides:
  BASELINE_COMMIT=78aaabc
  FIXED_COMMIT=d39d54e
  JIT_COMMIT=c204ea8
  RATES=0.1,0.2,0.3,0.7
  NUM_REQUESTS=50
  PROMPT_TOKENS=512
  MAX_TOKENS=128
  TEMP=0
  VARIANTS=baseline,fixed,jit
  SCENARIOS=unshaped,shaped30
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-base)
      OUT_BASE="$2"
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

mkdir -p "$OUT_BASE"

log() {
  printf '[compare] %s\n' "$*"
}

wait_for_http_200() {
  local url="$1"
  local msg="$2"
  log "wait $msg: $url"
  for _ in $(seq 1 240); do
    local code
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$url" || true)
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "[error] timeout waiting for $url" >&2
  return 1
}

stop_stages() {
  log "stopping existing molink stages"
  pkill -f "molinkv1.entrypoints.api_server.*--port 8000" || true
  ssh $SSH_OPTS node2 "pkill -f 'molinkv1.entrypoints.api_server.*--port 8001' || true" || true
  ssh $SSH_OPTS node6 "pkill -f 'molinkv1.entrypoints.api_server.*--port 8002' || true" || true
  sleep 3
}

sync_commit() {
  local commit="$1"
  log "sync commit=$commit"
  ROOT_DIR="$ROOT_DIR" "$DEV_ROOT/experiments/run/sync_nodes.sh" --commit "$commit"
}

shape_on_all_nodes() {
  local rate_mbps="$1"
  local delay_ms="$2"
  log "apply shaping rate=${rate_mbps}mbps delay=${delay_ms}ms(one-way)"
  sudo "$DEV_ROOT/experiments/net/shape_peers.sh" \
    --peers "$NODE2_IP,$NODE6_IP" --rate-mbps "$rate_mbps" --delay-ms "$delay_ms"
  ssh $SSH_OPTS node2 "sudo '$DEV_ROOT/experiments/net/shape_peers.sh' --peers '$NODE5_IP,$NODE6_IP' --rate-mbps '$rate_mbps' --delay-ms '$delay_ms'"
  ssh $SSH_OPTS node6 "sudo '$DEV_ROOT/experiments/net/shape_peers.sh' --peers '$NODE5_IP,$NODE2_IP' --rate-mbps '$rate_mbps' --delay-ms '$delay_ms'"
}

clear_shape_on_all_nodes() {
  log "clear shaping"
  sudo "$DEV_ROOT/experiments/net/clear_shape.sh" || true
  ssh $SSH_OPTS node2 "sudo '$DEV_ROOT/experiments/net/clear_shape.sh' || true" || true
  ssh $SSH_OPTS node6 "sudo '$DEV_ROOT/experiments/net/clear_shape.sh' || true" || true
}

start_stages() {
  local log_dir="$1"
  mkdir -p "$log_dir"
  log "starting stage0/stage1/stage2"

  nohup env \
    VENV_PY="$VENV_PY" \
    ROOT_DIR="$ROOT_DIR" \
    RESULTS_DIR="$log_dir" \
    HF_HOME="$ROOT_DIR/.runtime/hf_cache" \
    TRANSFORMERS_CACHE="$ROOT_DIR/.runtime/hf_cache" \
    HF_MODULES_CACHE="$ROOT_DIR/.runtime/hf_cache/modules" \
    VLLM_CACHE_ROOT="$ROOT_DIR/.runtime/vllm_cache" \
    XDG_CACHE_HOME="$ROOT_DIR/.runtime/hf_cache" \
    TMPDIR="$ROOT_DIR/.runtime/tmp" \
    PORT=8000 GRPC_PORT=50061 \
    bash "$ROOT_DIR/experiments/run/molinkv1_stage0.sh" \
    >"$log_dir/stage0_launcher.log" 2>&1 &

  ssh $SSH_OPTS node2 "mkdir -p '$log_dir'" >/dev/null 2>&1 || true
  ssh $SSH_OPTS node2 "nohup env \
    VENV_PY='$VENV_PY' \
    ROOT_DIR='$ROOT_DIR' \
    RESULTS_DIR='$log_dir' \
    HF_HOME='/tmp/molink-node2-runtime/hf_cache' \
    TRANSFORMERS_CACHE='/tmp/molink-node2-runtime/hf_cache' \
    HF_MODULES_CACHE='/tmp/molink-node2-runtime/hf_cache/modules' \
    VLLM_CACHE_ROOT='/tmp/molink-node2-runtime/vllm_cache' \
    XDG_CACHE_HOME='/tmp/molink-node2-runtime/hf_cache' \
    TMPDIR='/tmp/molink-node2-runtime/tmp' \
    PORT=8001 GRPC_PORT=50062 INITIAL_PEER='$NODE5_IP:50061' \
    bash '$ROOT_DIR/experiments/run/molinkv1_stage1.sh' \
    >'$log_dir/stage1_launcher.log' 2>&1 &" || true

  ssh $SSH_OPTS node6 "mkdir -p '$log_dir'" >/dev/null 2>&1 || true
  ssh $SSH_OPTS node6 "nohup env \
    VENV_PY='$VENV_PY' \
    ROOT_DIR='$ROOT_DIR' \
    RESULTS_DIR='$log_dir' \
    HF_HOME='/tmp/molink-node6-runtime/hf_cache' \
    TRANSFORMERS_CACHE='/tmp/molink-node6-runtime/hf_cache' \
    HF_MODULES_CACHE='/tmp/molink-node6-runtime/hf_cache/modules' \
    VLLM_CACHE_ROOT='/tmp/molink-node6-runtime/vllm_cache' \
    XDG_CACHE_HOME='/tmp/molink-node6-runtime/hf_cache' \
    TMPDIR='/tmp/molink-node6-runtime/tmp' \
    PORT=8002 GRPC_PORT=50063 INITIAL_PEER='$NODE5_IP:50061' \
    bash '$ROOT_DIR/experiments/run/molinkv1_stage2.sh' \
    >'$log_dir/stage2_launcher.log' 2>&1 &" || true

  wait_for_http_200 "http://127.0.0.1:8000/health" "stage0"
  wait_for_http_200 "http://$NODE2_IP:8001/health" "stage1"
  wait_for_http_200 "http://$NODE6_IP:8002/health" "stage2"
}

run_one_sweep() {
  local system_name="$1"
  local out_dir="$2"
  local bandwidth_mbps="$3"
  local rtt_ms="$4"

  mkdir -p "$out_dir"
  "$DEV_ROOT/experiments/loadgen/run_sweep.sh" \
    --system "$system_name" \
    --mode generate \
    --base-url "http://$NODE5_IP:8000" \
    --model "$MODEL" \
    --rates "$RATES" \
    --num-requests "$NUM_REQUESTS" \
    --poisson \
    --prompt-tokens "$PROMPT_TOKENS" \
    --tokenizer "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    --bandwidth-mbps "$bandwidth_mbps" \
    --rtt-ms "$rtt_ms" \
    --max-in-flight "$MAX_IN_FLIGHT" \
    --out-dir "$out_dir"
}

summarize_scenario() {
  local scenario_dir="$1"
  "$VENV_PY" "$DEV_ROOT/experiments/analysis/summarize_plot.py" \
    --in "$scenario_dir" \
    --out-dir "$scenario_dir" \
    --baseline-system molink_baseline \
    --systems molink_baseline molink_fixed2mb molink_jit \
    --emit-table2 \
    --stat p95
}

run_variant() {
  local commit="$1"
  local system_name="$2"
  local scenario="$3"
  local bandwidth_mbps="$4"
  local rtt_ms="$5"
  local scenario_dir="$OUT_BASE/$scenario"
  local variant_dir="$scenario_dir/$system_name"
  local log_dir="$variant_dir/_logs"

  mkdir -p "$variant_dir" "$log_dir"

  sync_commit "$commit"
  stop_stages
  start_stages "$log_dir"
  run_one_sweep "$system_name" "$variant_dir" "$bandwidth_mbps" "$rtt_ms"
  stop_stages
}

run_scenario() {
  local scenario="$1"
  local bandwidth_mbps="$2"
  local rtt_ms="$3"
  local one_way_delay_ms="$4"

  log "===== scenario=$scenario ====="
  mkdir -p "$OUT_BASE/$scenario"

  if [[ "$scenario" == "shaped30" ]]; then
    shape_on_all_nodes "$bandwidth_mbps" "$one_way_delay_ms"
  else
    clear_shape_on_all_nodes
  fi

  IFS=',' read -r -a variant_arr <<< "$VARIANTS"
  for variant in "${variant_arr[@]}"; do
    variant=$(echo "$variant" | xargs)
    case "$variant" in
      baseline)
        run_variant "$BASELINE_COMMIT" "molink_baseline" "$scenario" "$bandwidth_mbps" "$rtt_ms"
        ;;
      fixed)
        run_variant "$FIXED_COMMIT" "molink_fixed2mb" "$scenario" "$bandwidth_mbps" "$rtt_ms"
        ;;
      jit)
        run_variant "$JIT_COMMIT" "molink_jit" "$scenario" "$bandwidth_mbps" "$rtt_ms"
        ;;
      *)
        echo "[error] unknown variant: $variant" >&2
        exit 1
        ;;
    esac
  done

  summarize_scenario "$OUT_BASE/$scenario"

  if [[ "$scenario" == "shaped30" ]]; then
    clear_shape_on_all_nodes
  fi
}

trap 'clear_shape_on_all_nodes; stop_stages' EXIT

log "out_base=$OUT_BASE"
IFS=',' read -r -a scenario_arr <<< "$SCENARIOS"
for scenario in "${scenario_arr[@]}"; do
  scenario=$(echo "$scenario" | xargs)
  case "$scenario" in
    unshaped)
      run_scenario "unshaped" "$UNSHAPED_BW_MBPS" "$UNSHAPED_RTT_MS" "0"
      ;;
    shaped30)
      run_scenario "shaped30" "$SHAPED_BW_MBPS" "$SHAPED_RTT_MS" "15"
      ;;
    *)
      echo "[error] unknown scenario: $scenario" >&2
      exit 1
      ;;
  esac
done
log "done -> $OUT_BASE"
