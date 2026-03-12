FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    git curl ca-certificates \
    build-essential ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel

# Install GPU PyTorch (cu121). vLLM 0.7.x is compatible with Torch 2.4.x.
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.0 torchvision==0.19.0

# Install vLLM + Ray for PP via distributed-executor-backend=ray.
# Keep click pinned (often helps with Ray CLI compatibility).
RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 pillow==10.3.0 fsspec==2024.9.0 \
    transformers==4.48.2 \
    click==8.1.7 \
    ray==2.39.0 \
    vllm==0.7.2

ENV HF_HOME=/hf_cache \
    TRANSFORMERS_CACHE=/hf_cache \
    HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    HF_MODULES_CACHE=/hf_cache/modules

WORKDIR /workspace
