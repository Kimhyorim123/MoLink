#!/usr/bin/env bash
set -euo pipefail

cd /home/sslab/MoLink

OUT_BASE=$(cat results/_last_unshaped_dir.txt)
LOG_DIR="$OUT_BASE/_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/monitor.log"
PID_FILE="$LOG_DIR/monitor.pid"

INTERVAL_S=${INTERVAL_S:-30}

files_to_watch=(
  "$OUT_BASE/vllm/vllm_rate0.1.csv"
  "$OUT_BASE/vllm/vllm_rate0.2.csv"
  "$OUT_BASE/vllm/vllm_rate0.3.csv"
  "$OUT_BASE/vllm/vllm_rate0.7.csv"
  "$OUT_BASE/vllm_chunked/vllm_chunked_rate0.1.csv"
  "$OUT_BASE/vllm_chunked/vllm_chunked_rate0.2.csv"
  "$OUT_BASE/vllm_chunked/vllm_chunked_rate0.3.csv"
  "$OUT_BASE/vllm_chunked/vllm_chunked_rate0.7.csv"
  "$OUT_BASE/molink/molink_rate0.1.csv"
  "$OUT_BASE/molink/molink_rate0.2.csv"
  "$OUT_BASE/molink/molink_rate0.3.csv"
  "$OUT_BASE/molink/molink_rate0.7.csv"
)

status_line() {
  local path="$1"
  if [[ -f "$path" ]]; then
    local sz
    sz=$(stat -c '%s' "$path" 2>/dev/null || echo '?')
    echo "OK size=$sz"
  else
    echo "WAIT"
  fi
}

http_code() {
  local url="$1"
  curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$url" 2>/dev/null || echo "000"
}

{
  echo "[$(date '+%F %T')] monitor start OUT_BASE=$OUT_BASE interval=${INTERVAL_S}s"
} >>"$LOG_FILE"

echo $$ >"$PID_FILE"

while true; do
  ts="[$(date '+%F %T')]"
  {
    echo "$ts heartbeat"
    echo "$ts http vllm_models=$(http_code http://127.0.0.1:8000/v1/models) stage0=$(http_code http://192.168.79.9:8000/health) stage1=$(http_code http://127.0.0.1:8001/health) stage2=$(http_code http://192.168.79.22:8002/health)"

    procs=$(ps -eo pid,etimes,cmd | grep -E 'openai_stream_loadgen.py|run_sweep.sh|unshaped_pipeline_continue.sh|vllm.entrypoints.openai.api_server|molinkv1.entrypoints.api_server' | grep -v grep || true)
    if [[ -n "$procs" ]]; then
      echo "$ts procs:\n$procs"
    else
      echo "$ts procs: (none)"
    fi

    for f in "${files_to_watch[@]}"; do
      echo "$ts file $(basename "$f"): $(status_line "$f")"
    done

    echo "$ts ----"
  } >>"$LOG_FILE"

  sleep "$INTERVAL_S"
done
