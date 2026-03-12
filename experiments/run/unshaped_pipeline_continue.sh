#!/usr/bin/env bash
set -euo pipefail

# Continue the unshaped resweep after the baseline vLLM sweep finishes.
# Runs on node5.

cd /home/sslab/MoLink

OUT_BASE=$(cat results/_last_unshaped_dir.txt)
LOG_DIR="$OUT_BASE/_logs"
mkdir -p "$LOG_DIR"

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
RATES=${RATES:-0.1,0.2,0.3,0.7}
NUM_REQUESTS=${NUM_REQUESTS:-50}
PROMPT_TOKENS=${PROMPT_TOKENS:-512}
MAX_TOKENS=${MAX_TOKENS:-128}
TEMP=${TEMP:-0}
BW_MBPS=${BW_MBPS:-0}
RTT_MS=${RTT_MS:-0}

wait_for_file() {
  local path="$1"
  local msg="$2"
  echo "[wait] $msg: $path"
  while [[ ! -f "$path" ]]; do
    sleep 5
  done
}

wait_for_http_200() {
  local url="$1"
  local msg="$2"
  echo "[wait] $msg: $url"
  for _ in $(seq 1 240); do
    code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$url" || true)
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep 2
  done
  echo "[error] timeout waiting for $url" >&2
  return 1
}

stop_molink_stages() {
  echo "[molink] stopping any existing stages"
  ssh -o BatchMode=yes node4 "pkill -f 'molinkv1.entrypoints.api_server.*--port 8000' || true"
  pkill -f "molinkv1.entrypoints.api_server.*--port 8001" || true
  ssh -o BatchMode=yes node6 "pkill -f 'molinkv1.entrypoints.api_server.*--port 8002' || true"
}

stop_vllm_containers() {
  docker rm -f vllm072_api >/dev/null 2>&1 || true
  docker rm -f vllm072_chunked_api >/dev/null 2>&1 || true
}

cleanup() {
  set +e
  stop_molink_stages
  docker rm -f vllm072_chunked_api >/dev/null 2>&1 || true
}
trap cleanup EXIT

# 1) Wait baseline sweep to finish (last rate CSV exists).
wait_for_file "$OUT_BASE/vllm/vllm_rate0.7.csv" "vLLM baseline sweep done"

# 2) vLLM chunked sweep (restart API container).
echo "[vllm] switching to chunked prefill API"
stop_vllm_containers
DETACH=1 ./experiments/docker/run_vllm_chunked_api_container_072.sh >"$LOG_DIR/vllm_chunked_container.log" 2>&1
wait_for_http_200 "http://127.0.0.1:8000/v1/models" "vLLM chunked server ready"

./experiments/loadgen/run_sweep.sh \
  --system vllm_chunked --mode openai-completions \
  --base-url http://127.0.0.1:8000 --model "$MODEL" \
  --rates "$RATES" --num-requests "$NUM_REQUESTS" \
  --prompt-tokens "$PROMPT_TOKENS" --tokenizer "$MODEL" \
  --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
  --bandwidth-mbps "$BW_MBPS" --rtt-ms "$RTT_MS" \
  --out-dir "$OUT_BASE/vllm_chunked" \
  >"$LOG_DIR/vllm_chunked_sweep.log" 2>&1

docker rm -f vllm072_chunked_api >/dev/null 2>&1 || true

# 3) MoLink sweep (start stages on node4/node5/node6).
echo "[molink] starting stages"
stop_molink_stages

ssh -o BatchMode=yes node4 "cd /home/sslab/MoLink && nohup ./experiments/run/molinkv1_stage0.sh >'$LOG_DIR/molink_stage0.log' 2>&1 &" 
nohup ./experiments/run/molinkv1_stage1.sh >"$LOG_DIR/molink_stage1.log" 2>&1 &
ssh -o BatchMode=yes node6 "cd /home/sslab/MoLink && nohup ./experiments/run/molinkv1_stage2.sh >'$LOG_DIR/molink_stage2.log' 2>&1 &" 

wait_for_http_200 "http://192.168.79.9:8000/health" "MoLink stage0 ready"
wait_for_http_200 "http://127.0.0.1:8001/health" "MoLink stage1 ready"
wait_for_http_200 "http://192.168.79.22:8002/health" "MoLink stage2 ready"

./experiments/loadgen/run_sweep.sh \
  --system molink --mode generate \
  --base-url http://192.168.79.9:8000 --model "$MODEL" \
  --rates "$RATES" --num-requests "$NUM_REQUESTS" \
  --prompt-tokens "$PROMPT_TOKENS" --tokenizer "$MODEL" \
  --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
  --bandwidth-mbps "$BW_MBPS" --rtt-ms "$RTT_MS" \
  --out-dir "$OUT_BASE/molink" \
  >"$LOG_DIR/molink_sweep.log" 2>&1

echo "[done] unshaped sweeps complete: $OUT_BASE"
