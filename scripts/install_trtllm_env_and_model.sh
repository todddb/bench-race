#!/usr/bin/env bash
set -euo pipefail
# install_trtllm_env_and_model.sh
# Usage: ./scripts/install_trtllm_env_and_model.sh
# Edit the variables below to change model/image/paths.

PROJECT_ROOT="${HOME}/projects/bench-race"
VENV_DIR="${PROJECT_ROOT}/.venv"
SECRETS_FILE="${HOME}/.bench-race-secrets"

# model/checkpoint/engine layout (change if you prefer different layout)
BACKEND_ROOT="${PROJECT_ROOT}/backends/tensorrt-llm"
MODEL_DIR="${BACKEND_ROOT}/models/opt-125m"        # HF model snapshot location
CHECKPOINT_DIR="${BACKEND_ROOT}/checkpoints/opt/125M/trt_ckpt/fp16/1-gpu"  # optional trt_ckpt
ENGINE_OUT="${BACKEND_ROOT}/engines/opt125m_engine"

# container image & build params
CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3"
MAX_BATCH=8
MAX_INPUT_LEN=924
MAX_SEQ_LEN=1024
WORKERS=1

# ensure dirs exist
mkdir -p "$PROJECT_ROOT" "$MODEL_DIR" "$CHECKPOINT_DIR" "$ENGINE_OUT" "$(dirname "$VENV_DIR")"

# load secrets
if [ ! -f "$SECRETS_FILE" ]; then
  echo "Secrets file missing: $SECRETS_FILE"
  echo "Create it with HUGGINGFACE_TOKEN and NGC_API_KEY variables (see instructions)."
  exit 1
fi
# shellcheck source=/dev/null
source "$SECRETS_FILE"

if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
  echo "HUGGINGFACE_TOKEN is empty in $SECRETS_FILE"
  exit 1
fi

# create & activate venv
if [ ! -d "$VENV_DIR" ]; then
  echo ">>> Creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools
python -m pip install --upgrade huggingface_hub

# Download HF model snapshot (idempotent)
echo ">>> Downloading HF model snapshot to: $MODEL_DIR"
python - <<PY
from huggingface_hub import snapshot_download
import os,sys
out_dir = os.path.expanduser("${MODEL_DIR}")
repo_id = "facebook/opt-125m"
token = os.getenv("HUGGINGFACE_TOKEN")
print("snapshot_download(repo_id={0}, out_dir={1})".format(repo_id, out_dir))
snapshot_download(repo_id=repo_id, local_dir=out_dir, local_dir_use_symlinks=False, token=token)
print("download complete")
PY

# verify model weight presence
echo ">>> Verifying model artifacts..."
ls -la "${MODEL_DIR}" | egrep 'pytorch_model.bin|flax_model.msgpack|tf_model.h5|config.json' || true

# pull the container image
echo ">>> Pulling container image: $CONTAINER_IMAGE"
docker pull "$CONTAINER_IMAGE"

# Run trtllm-build inside container
# Prefer using a checkpoint dir if you have one (convert flow). If you built trt_ckpt already, point --checkpoint_dir there.
# If you have full HF model dir with pytorch_model.bin, we mount that as /checkpoint as well.
echo ">>> Running trtllm-build (containerized) -> output: $ENGINE_OUT"
docker run --rm --gpus all -it \
  -v "${MODEL_DIR}:/checkpoint:ro" \
  -v "${ENGINE_OUT}:/output:rw" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface:rw" \
  --ipc=host \
  "${CONTAINER_IMAGE}" \
  /bin/bash -lc "\
    set -euo pipefail; \
    echo 'Inside container: trtllm-build --checkpoint_dir /checkpoint --max_batch_size ${MAX_BATCH} --max_input_len ${MAX_INPUT_LEN} --max_seq_len ${MAX_SEQ_LEN} --output_dir /output --workers ${WORKERS}'; \
    trtllm-build --checkpoint_dir /checkpoint \
      --gemm_plugin float16 \
      --max_batch_size ${MAX_BATCH} \
      --max_input_len ${MAX_INPUT_LEN} \
      --max_seq_len ${MAX_SEQ_LEN} \
      --output_dir /output \
      --workers ${WORKERS} ; \
    echo 'trtllm-build finished (exit).' \
"

echo ">>> Build finished. Engine dir: $ENGINE_OUT"
echo ">>> Deactivate venv and you're done for this step."
deactivate || true
