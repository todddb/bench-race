#!/usr/bin/env bash
# ============================================================
# Bench-Race Universal Model Sync
# Apple + NVIDIA safe
# Self-contained venv
# Idempotent + continues on failure
# ============================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REGISTRY_PATH="${REPO_ROOT}/config/registry/models.json"
MACHINES_PATH="${REPO_ROOT}/config/machines.yaml"
SECRETS_FILE="${HOME}/.bench-race-secrets"

VENV_DIR="${SCRIPT_DIR}/.sync_venv"
PYTHON_BIN="${VENV_DIR}/bin/python"

COMFY_DIR="${REPO_ROOT}/agent/models/comfy"
MLX_DIR="${REPO_ROOT}/agent/models/mlx"
TRT_DIR="${REPO_ROOT}/agent/models/trtllm"

mkdir -p "$COMFY_DIR" "$MLX_DIR" "$TRT_DIR"

echo "==== Bench-Race Model Sync ===="

# ============================================================
# 1️⃣ Setup isolated venv
# ============================================================

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating sync virtual environment..."
    python3 -m venv "$VENV_DIR" || exit 1
fi

source "${VENV_DIR}/bin/activate"

pip install --upgrade pip >/dev/null 2>&1
pip install huggingface_hub pyyaml >/dev/null 2>&1

# ============================================================
# 2️⃣ Load secrets
# ============================================================

if [[ -f "$SECRETS_FILE" ]]; then
    source "$SECRETS_FILE"
fi

export HF_TOKEN="${HUGGINGFACE_TOKEN:-}"
export AGENT_ID="${AGENT_ID:-}"

# ============================================================
# 3️⃣ Select machine if needed
# ============================================================

if [[ -z "${AGENT_ID}" ]]; then
    echo
    echo "Select this machine:"
    echo

    "$PYTHON_BIN" <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
for i,m in enumerate(data.get("machines",[]),1):
    print(f"{i}. {m['machine_id']} ({m.get('gpu',{}).get('type')})")
PY

    echo
    read -p "Enter number: " IDX

    AGENT_ID=$("$PYTHON_BIN" <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
idx=int("$IDX")-1
machines=data.get("machines",[])
print(machines[idx]["machine_id"] if 0<=idx<len(machines) else "")
PY
)

    [[ -z "$AGENT_ID" ]] && echo "Invalid selection." && exit 1

    echo "Persisting AGENT_ID=$AGENT_ID"
    echo "AGENT_ID=\"$AGENT_ID\"" >> "$SECRETS_FILE"
fi

echo "Using AGENT_ID: $AGENT_ID"

# ============================================================
# 4️⃣ Determine architecture
# ============================================================

ARCH=$("$PYTHON_BIN" <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
for m in data.get("machines",[]):
    if m["machine_id"]=="$AGENT_ID":
        print(m.get("gpu",{}).get("type"))
        break
PY
)

[[ "$ARCH" != "apple" && "$ARCH" != "nvidia" ]] && \
    echo "Could not determine architecture." && exit 1

echo "Detected architecture: $ARCH"

# ============================================================
# 5️⃣ Validate models.json
# ============================================================

"$PYTHON_BIN" - <<PY
import json
json.load(open("$REGISTRY_PATH"))
PY

[[ $? -ne 0 ]] && echo "models.json invalid." && exit 1

# ============================================================
# 6️⃣ Tracking arrays
# ============================================================

declare -a SUCCESS
declare -a FAILED

# ============================================================
# 7️⃣ Registry reader
# ============================================================

read_registry() {
"$PYTHON_BIN" <<PY
import json
data=json.load(open("$REGISTRY_PATH"))

for m in data.get("ollama",[]):
    tag=m.get("$ARCH")
    if tag:
        print("OLLAMA", tag)

for m in data.get("custom",[]):
    if "$ARCH"=="apple":
        val=m.get("mlx_hf_id")
        if val:
            print("MLX", val)
    else:
        hf=m.get("trt-llm_hf_id")
        eng=m.get("trt-llm_engine_dir")
        if hf and eng:
            print("TRT", hf, eng)

for c in data.get("comfyui",[]):
    print("COMFY", "stabilityai/stable-diffusion-xl-base-1.0")
PY
}

# ============================================================
# 8️⃣ Sync loop
# ============================================================

while read kind a b; do

case "$kind" in

OLLAMA)
echo "[Ollama] $a"
if command -v ollama >/dev/null 2>&1; then
    ollama pull "$a" && SUCCESS+=("Ollama:$a") || FAILED+=("Ollama:$a")
else
    FAILED+=("Ollama:$a (missing ollama)")
fi
;;

MLX)
echo "[MLX] $a"
target="$MLX_DIR/$(basename "$a")"

if [[ -d "$target" ]]; then
    SUCCESS+=("MLX:$a (exists)")
else
    "$PYTHON_BIN" <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="$a",
    local_dir="$target",
    token=os.environ.get("HF_TOKEN"),
    resume_download=True
)
PY
    [[ $? -eq 0 ]] && SUCCESS+=("MLX:$a") || FAILED+=("MLX:$a")
fi
;;

TRT)
echo "[TRT] $a"
target="$TRT_DIR/$b"

if [[ -d "$target" ]]; then
    SUCCESS+=("TRT:$a (exists)")
else
    "$PYTHON_BIN" <<PY
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id="$a",
    local_dir="$target",
    token=os.environ.get("HF_TOKEN"),
    resume_download=True
)
PY
    [[ $? -eq 0 ]] && SUCCESS+=("TRT:$a") || FAILED+=("TRT:$a")
fi
;;

COMFY)
echo "[ComfyUI] SDXL Base 1.0"
"$PYTHON_BIN" <<PY
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id="$a",
    filename="sd_xl_base_1.0.safetensors",
    local_dir="$COMFY_DIR",
    token=os.environ.get("HF_TOKEN"),
    resume_download=True
)
PY
[[ $? -eq 0 ]] && SUCCESS+=("Comfy:SDXL") || FAILED+=("Comfy:SDXL")
;;

esac

done < <(read_registry)

# ============================================================
# 9️⃣ Summary
# ============================================================

echo
echo "==== Sync Summary ===="

for s in "${SUCCESS[@]}"; do
    echo "✔ $s"
done

for f in "${FAILED[@]}"; do
    echo "✘ $f"
done

echo "======================"
