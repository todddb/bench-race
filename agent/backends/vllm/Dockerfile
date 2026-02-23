# =============================================================================
# NOTICE: This Dockerfile is no longer used for the primary vLLM build.
#
# bench-race now builds vLLM from source using the upstream vLLM repository
# Dockerfile to support Blackwell GPUs (RTX Pro 6000, RTX 5090, sm_120).
#
# To build the vLLM image:
#   ./scripts/install_x64_vllm.sh
#
# This produces image: bench-race/vllm:blackwell (also tagged :latest)
#
# The built image runs vLLM's native OpenAI-compatible API server directly
# on port 8000. No wrapper is needed.
#
# Original wrapper-based Dockerfile preserved below for reference only.
# =============================================================================

# --- ARCHIVED (reference only, not built) ---
# FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#
# ENV DEBIAN_FRONTEND=noninteractive \
#     PIP_NO_CACHE_DIR=1 \
#     VLLM_HOST=0.0.0.0 \
#     VLLM_PORT=8010 \
#     VLLM_OPENAI_BASE=http://127.0.0.1:8001
#
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3 python3-pip python3-venv curl && \
#     rm -rf /var/lib/apt/lists/*
#
# WORKDIR /app
# COPY requirements.txt /app/requirements.txt
# RUN python3 -m pip install --upgrade pip && \
#     python3 -m pip install -r /app/requirements.txt && \
#     python3 -m pip install vllm==0.6.2
#
# COPY wrapper.py /app/wrapper.py
# COPY start.sh /app/start.sh
# RUN chmod +x /app/start.sh
#
# EXPOSE 8010
# HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:${VLLM_PORT}/health || exit 1
#
# CMD ["/app/start.sh"]
