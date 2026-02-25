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
MODEL_ID="distilgpt2"                             # intentionally small model for quick validation
MODEL_DIR="${BACKEND_ROOT}/models/${MODEL_ID}"    # HF model snapshot location
CHECKPOINT_DIR="${BACKEND_ROOT}/checkpoints/${MODEL_ID}/fp16/1-gpu"
ENGINE_OUT="${BACKEND_ROOT}/engines/${MODEL_ID}_engine"

# container image & build params
CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3"
MAX_BATCH=4
MAX_INPUT_LEN=256
MAX_SEQ_LEN=512
WORKERS=1
NUM_GPUS=1
DTYPE="float16"

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
echo ">>> Downloading HF model snapshot (${MODEL_ID}) to: $MODEL_DIR"
python - <<PY
from huggingface_hub import snapshot_download
import os,sys
out_dir = os.path.expanduser("${MODEL_DIR}")
repo_id = "${MODEL_ID}"
token = os.getenv("HUGGINGFACE_TOKEN")
print("snapshot_download(repo_id={0}, out_dir={1})".format(repo_id, out_dir))
snapshot_download(
    repo_id=repo_id,
    local_dir=out_dir,
    local_dir_use_symlinks=False,
    token=token,
    ignore_patterns=["*.h5", "*.msgpack", "rust_model.ot", "*.tflite"],
)
print("download complete")
PY

# verify model weight presence
echo ">>> Verifying model artifacts..."
ls -la "${MODEL_DIR}" | egrep 'pytorch_model.bin|model.safetensors|config.json|tokenizer.json|vocab.json|merges.txt' || true

# pull the container image
echo ">>> Pulling container image: $CONTAINER_IMAGE"
docker pull "$CONTAINER_IMAGE"

# Convert from HF checkpoint to TensorRT-LLM checkpoint, then build engine.
# NOTE: We do not use NVFP4 here. NVFP4 is optional and requires an explicit quantization pass.
echo ">>> Converting HF model -> TensorRT-LLM checkpoint -> engine"
docker run --rm --gpus all -it \
  -v "${MODEL_DIR}:/model:ro" \
  -v "${CHECKPOINT_DIR}:/checkpoint:rw" \
  -v "${ENGINE_OUT}:/output:rw" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface:rw" \
  --ipc=host \
  "${CONTAINER_IMAGE}" \
  /bin/bash -lc "\
    set -euo pipefail; \
    CONVERT_SCRIPT=''; \
    for p in /app/tensorrt_llm/examples/models/core/gpt/convert_checkpoint.py /app/tensorrt_llm/examples/gpt/convert_checkpoint.py; do \
      if [ -f \"$p\" ]; then CONVERT_SCRIPT=\"$p\"; break; fi; \
    done; \
    if [ -z \"$CONVERT_SCRIPT\" ]; then \
      echo 'Could not find GPT convert_checkpoint.py inside image'; \
      exit 1; \
    fi; \
    echo \"Using convert script: $CONVERT_SCRIPT\"; \
    python \"$CONVERT_SCRIPT\" \
      --model_dir /model \
      --output_dir /checkpoint \
      --dtype ${DTYPE} \
      --tp_size ${NUM_GPUS}; \
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

echo ">>> Build finished."
echo "    TensorRT-LLM checkpoint: $CHECKPOINT_DIR"
echo "    Engine output: $ENGINE_OUT"
echo ">>> Deactivate venv and you're done for this step."
deactivate || true
