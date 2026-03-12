#!/usr/bin/env bash
set -euo pipefail

# HuggingFace cache path fix:
# Some nodes/images have /home/sslab/.cache/huggingface owned by root, which breaks
# tokenizer/model downloads. Prefer a user-writable cache if available.
if [[ -z "${HF_HOME:-}" && -d "/home/sslab/hf_cache" ]]; then
  export HF_HOME="/home/sslab/hf_cache"
fi
if [[ -z "${TRANSFORMERS_CACHE:-}" && -d "/home/sslab/hf_cache" ]]; then
  export TRANSFORMERS_CACHE="/home/sslab/hf_cache"
fi

# Run a rate sweep for a single system+endpoint.
#
# Example:
#   ./run_sweep.sh \
#     --system vllm --mode openai-completions \
#     --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct \
#     --rates 0.1,0.2,0.3,0.7 --num-requests 50 \
#     --max-tokens 128 --temperature 0 \
#     --tokenizer Qwen/Qwen2.5-7B-Instruct \
#     --out-dir results/vllm

usage() {
  cat <<'EOF'
Usage: run_sweep.sh --system NAME --mode MODE --base-url URL --model MODEL \
  --rates r1,r2,... --num-requests N --out-dir DIR [other args]

Required:
  --system        vllm | vllm_chunked | molink
  --mode          openai-completions | openai-chat | generate
  --base-url      e.g., http://127.0.0.1:8000
  --model         e.g., Qwen/Qwen2.5-7B-Instruct
  --rates         comma-separated rates (req/s), e.g. 0.1,0.2,0.3,0.7
  --num-requests  requests per rate
  --out-dir       output directory

Optional (passed to loadgen):
  --bandwidth-mbps N (default 100)
  --rtt-ms N (default 30)
  --poisson
  --prompt FILE or --prompt-text TEXT
  --prompt-tokens N (requires --tokenizer)
  --prompt-seed-text TEXT
  --max-tokens N (default 128)
  --temperature X (default 0)
  --api-key KEY (default EMPTY)
  --timeout-s S (default 600)
  --seed N (default 0)
  --max-in-flight N (default 32)
  --tokenizer NAME_OR_PATH (recommended)
EOF
}

SYSTEM=""; MODE=""; BASE_URL=""; MODEL=""; RATES=""; NUM_REQUESTS=""; OUT_DIR=""
POISSON=""; PROMPT_FILE=""; PROMPT_TEXT=""; MAX_TOKENS="128"; TEMPERATURE="0"
API_KEY="EMPTY"; TIMEOUT_S="600"; SEED="0"; MAX_IN_FLIGHT="32"; TOKENIZER=""
PROMPT_TOKENS=""; PROMPT_SEED_TEXT=""
BW_MBPS="100"; RTT_MS="30"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system) SYSTEM="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --base-url) BASE_URL="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --rates) RATES="$2"; shift 2;;
    --num-requests) NUM_REQUESTS="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;

    --bandwidth-mbps) BW_MBPS="$2"; shift 2;;
    --rtt-ms) RTT_MS="$2"; shift 2;;

    --poisson) POISSON="--poisson"; shift 1;;
    --prompt) PROMPT_FILE="$2"; shift 2;;
    --prompt-text) PROMPT_TEXT="$2"; shift 2;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2;;
    --prompt-seed-text) PROMPT_SEED_TEXT="$2"; shift 2;;
    --max-tokens) MAX_TOKENS="$2"; shift 2;;
    --temperature) TEMPERATURE="$2"; shift 2;;
    --api-key) API_KEY="$2"; shift 2;;
    --timeout-s) TIMEOUT_S="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --max-in-flight) MAX_IN_FLIGHT="$2"; shift 2;;
    --tokenizer) TOKENIZER="$2"; shift 2;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$SYSTEM" || -z "$MODE" || -z "$BASE_URL" || -z "$MODEL" || -z "$RATES" || -z "$NUM_REQUESTS" || -z "$OUT_DIR" ]]; then
  usage
  exit 1
fi

mkdir -p "$OUT_DIR"

PROMPT_ARGS=()
if [[ -n "$PROMPT_FILE" ]]; then
  PROMPT_ARGS+=(--prompt-file "$PROMPT_FILE")
elif [[ -n "$PROMPT_TEXT" ]]; then
  PROMPT_ARGS+=(--prompt "$PROMPT_TEXT")
fi

if [[ -n "$PROMPT_TOKENS" ]]; then
  PROMPT_ARGS+=(--prompt-tokens "$PROMPT_TOKENS")
  if [[ -n "$PROMPT_SEED_TEXT" ]]; then
    PROMPT_ARGS+=(--prompt-seed-text "$PROMPT_SEED_TEXT")
  fi
fi

TOKENIZER_ARGS=()
if [[ -n "$TOKENIZER" ]]; then
  TOKENIZER_ARGS+=(--tokenizer "$TOKENIZER")
fi

IFS=',' read -r -a rate_arr <<< "$RATES"
for rate in "${rate_arr[@]}"; do
  rate_trim=$(echo "$rate" | xargs)
  [[ -z "$rate_trim" ]] && continue

  out_csv="$OUT_DIR/${SYSTEM}_rate${rate_trim}.csv"
  echo "[sweep] system=$SYSTEM mode=$MODE rate=$rate_trim -> $out_csv"

  python3 "$(dirname "$0")/openai_stream_loadgen.py" \
    --system "$SYSTEM" \
    --mode "$MODE" \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    --bandwidth-mbps "$BW_MBPS" \
    --rtt-ms "$RTT_MS" \
    --rate-rps "$rate_trim" \
    --num-requests "$NUM_REQUESTS" \
    $POISSON \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --api-key "$API_KEY" \
    --timeout-s "$TIMEOUT_S" \
    --seed "$SEED" \
    --max-in-flight "$MAX_IN_FLIGHT" \
    "${PROMPT_ARGS[@]}" \
    "${TOKENIZER_ARGS[@]}" \
    --out-csv "$out_csv"
done

echo "[sweep] done: $OUT_DIR"