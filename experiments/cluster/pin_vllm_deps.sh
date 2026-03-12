#!/usr/bin/env bash
set -euo pipefail

# Pin Python deps to versions compatible with vLLM 0.6.3.post1 and common extras.
# Run this on every node that will participate in the Ray/vLLM PP=3 cluster.
#
# Fixes common resolver conflicts like:
# - vllm requires numpy<2
# - outlines requires numpy<2
# - numba requires numpy<2.1
# - mistral-common requires pillow<11
# - datasets requires fsspec<=2024.9.0

PY=${PY:-python3}
PIP_ARGS=("--user" "--upgrade" "--force-reinstall")

# Known-good pins for Python 3.10 + torch 2.4.0/cu121 era.
NUMPY_VER=${NUMPY_VER:-1.26.4}
PILLOW_VER=${PILLOW_VER:-10.3.0}
FSSPEC_VER=${FSSPEC_VER:-2024.9.0}

# vLLM 0.6.3.post1 requires transformers>=4.45.2
TRANSFORMERS_SPEC=${TRANSFORMERS_SPEC:-transformers==4.45.2}

# Some environments have PyNaCl installed without cffi; install cffi to satisfy pip check.
CFFI_SPEC=${CFFI_SPEC:-cffi}

echo "[pin] Using: numpy==${NUMPY_VER}, pillow==${PILLOW_VER}, fsspec==${FSSPEC_VER}"

"$PY" -m pip install "${PIP_ARGS[@]}" \
  "numpy==${NUMPY_VER}" \
  "pillow==${PILLOW_VER}" \
  "fsspec==${FSSPEC_VER}" \
  "${TRANSFORMERS_SPEC}" \
  "${CFFI_SPEC}"

echo "[pin] Versions after pinning:"
"$PY" - <<'PY'
import numpy
import PIL
import fsspec
import transformers
print('numpy', numpy.__version__)
print('pillow', PIL.__version__)
print('fsspec', fsspec.__version__)
print('transformers', transformers.__version__)
PY

echo "[pin] Done. If you still see conflicts, paste: python3 -m pip check"

if "$PY" -m pip show tensorrt-llm >/dev/null 2>&1; then
  cat <<'EOF'
[pin] Note: tensorrt-llm is installed in this environment.
[pin] It commonly conflicts with the torch/transformers pins required by vLLM.
[pin] If you do NOT need tensorrt-llm for these experiments, you can remove it:
[pin]   python3 -m pip uninstall -y tensorrt-llm
EOF
fi
