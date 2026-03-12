FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    git curl ca-certificates \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

# Install GPU PyTorch (cu128) to match MoLink image.
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.0 torchvision==0.24.0

# Install vLLM + Ray for PP via distributed-executor-backend=ray.
# Keep click pinned (historically helps Ray CLI compatibility).
RUN python3 -m pip install --no-cache-dir \
    click==8.1.7 \
    ray==2.48.0 \
    vllm==0.11.2

ENV HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache \
    HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    HF_MODULES_CACHE=/hf_cache/modules

WORKDIR /workspace
