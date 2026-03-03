#!/usr/bin/env bash
# ============================================================
# Bench-Race Universal Model Sync
# Self-contained venv + architecture aware
# Safe for Apple + NVIDIA
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
# 1. Setup isolated venv (auto create if missing)
# ============================================================

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating sync virtual environment..."
    python3 -m venv "$VENV_DIR" || {
        echo "Failed to create venv."
        exit 1
    }
fi

source "${VENV_DIR}/bin/activate"

pip install --upgrade pip >/dev/null 2>&1
pip install huggingface_hub pyyaml >/dev/null 2>&1

# ============================================================
# 2. Load secrets
# ============================================================

if [[ -f "$SECRETS_FILE" ]]; then
    source "$SECRETS_FILE"
fi

export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
export NGC_API_KEY="${NGC_API_KEY:-}"
export AGENT_ID="${AGENT_ID:-}"

# ============================================================
# 3. Select machine if not stored
# ============================================================

if [[ -z "${AGENT_ID}" ]]; then
    echo
    echo "Select this machine:"
    echo

    $PYTHON_BIN - <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
for i,m in enumerate(data.get("machines",[]),1):
    print(f"{i}. {m['machine_id']} ({m.get('gpu',{}).get('type')})")
PY

    echo
    read -p "Enter number: " IDX

    AGENT_ID=$($PYTHON_BIN - <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
idx=int("$IDX")-1
machines=data.get("machines",[])
if idx<0 or idx>=len(machines):
    print("")
else:
    print(machines[idx]["machine_id"])
PY
)

    if [[ -z "$AGENT_ID" ]]; then
        echo "Invalid selection."
        exit 1
    fi

    echo "Persisting AGENT_ID=$AGENT_ID"
    echo "AGENT_ID=\"$AGENT_ID\"" >> "$SECRETS_FILE"
fi

echo "Using AGENT_ID: $AGENT_ID"

# ============================================================
# 4. Determine architecture
# ============================================================

ARCH=$($PYTHON_BIN - <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
for m in data.get("machines",[]):
    if m["machine_id"]=="$AGENT_ID":
        print(m.get("gpu",{}).get("type"))
        break
PY
)

if [[ "$ARCH" != "apple" && "$ARCH" != "nvidia" ]]; then
    echo "Could not determine architecture."
    exit 1
fi

echo "Detected architecture: $ARCH"

# ============================================================
# 5. Validate JSON
# ============================================================

$PYTHON_BIN - <<PY
import json
json.load(open("$REGISTRY_PATH"))
PY

if [[ $? -ne 0 ]]; then
    echo "models.json invalid."
    exit 1
fi

# ============================================================
# 6. Tracking
# ============================================================

declare -a SUCCESS
declare -a FAILED

# ============================================================
# 7. Registry reader
# ============================================================

read_registry() {
$PYTHON_BIN - <<PY
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
    url=c.get("download_url")
    if url:
        print("COMFY", url)
PY
}

# ============================================================
# 8. Sync loop
# ============================================================

while read kind a b; do

    case "$kind" in

    OLLAMA)
        echo "[Ollama] $a"
        if command -v ollama >/dev/null 2>&1; then
            if ollama pull "$a"; then
                SUCCESS+=("Ollama:$a")
            else
                FAILED+=("Ollama:$a")
            fi
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
            if huggingface-cli download "$a" \
                --local-dir "$target" \
                --token "$HUGGINGFACE_TOKEN" ; then
                SUCCESS+=("MLX:$a")
            else
                FAILED+=("MLX:$a")
            fi
        fi
        ;;

    TRT)
        hf="$a"
        eng_dir="$TRT_DIR/$b"

        echo "[TRT] $hf"

        if [[ -d "$eng_dir" ]]; then
            SUCCESS+=("TRT:$hf (exists)")
        else
            if command -v trt-convert >/dev/null 2>&1; then
                if trt-convert --model "$hf" --engine-dir "$eng_dir"; then
                    SUCCESS+=("TRT:$hf")
                else
                    FAILED+=("TRT:$hf")
                fi
            else
                FAILED+=("TRT:$hf (trt-convert missing)")
            fi
        fi
        ;;

    COMFY)
        url="$a"
        fname=$(basename "$url")
        out="$COMFY_DIR/$fname"

        echo "[ComfyUI] $fname"

        if [[ -f "$out" ]]; then
            SUCCESS+=("Comfy:$fname (exists)")
        else
            if curl -L --fail -o "$out" "$url"; then
                SUCCESS+=("Comfy:$fname")
            else
                FAILED+=("Comfy:$fname")
            fi
        fi
        ;;

    esac

done < <(read_registry)

# ============================================================
# 9. Summary
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
