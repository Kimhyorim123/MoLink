#!/usr/bin/env bash
set -euo pipefail

RAY_BIN=${RAY_BIN:-}
if [[ -z "$RAY_BIN" ]]; then
	if [[ -x "$HOME/.local/bin/ray" ]]; then
		RAY_BIN="$HOME/.local/bin/ray"
	else
		RAY_BIN="ray"
	fi
fi

echo "[ray] stopping (if running)"
"$RAY_BIN" stop -f || true
