FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git curl ca-certificates \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

# Install GPU PyTorch (cu121) compatible with prior vLLM 0.6.3.post1 experiments.
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# Pin commonly conflicting deps and install vLLM.
# Keep Ray aligned with host control-plane Ray (2.39.0) for fewer surprises.
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 pillow==10.3.0 fsspec==2024.9.0 \
    transformers==4.45.2 \
    click==8.1.7 ray==2.39.0 \
    vllm==0.6.3.post1

ENV HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache \
    HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    HF_MODULES_CACHE=/hf_cache/modules

WORKDIR /workspace
