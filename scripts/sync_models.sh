#!/usr/bin/env bash
# ============================================================
# Bench-Race Robust Model Sync
# Safe for Apple + NVIDIA
# ============================================================

set -u  # NO set -e (we handle errors manually)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

REGISTRY_PATH="${REPO_ROOT}/config/registry/models.json"
MACHINES_PATH="${REPO_ROOT}/config/machines.yaml"
SECRETS_FILE="${HOME}/.bench-race-secrets"

COMFY_DIR="${REPO_ROOT}/agent/models/comfy"
MLX_DIR="${REPO_ROOT}/agent/models/mlx"
TRT_DIR="${REPO_ROOT}/agent/models/trtllm"

mkdir -p "$COMFY_DIR" "$MLX_DIR" "$TRT_DIR"

echo "==== Bench-Race Model Sync ===="

# ============================================================
# Load secrets
# ============================================================

if [[ -f "$SECRETS_FILE" ]]; then
    source "$SECRETS_FILE"
fi

export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}"
export NGC_API_KEY="${NGC_API_KEY:-}"
export AGENT_ID="${AGENT_ID:-}"

# ============================================================
# Select agent if not stored
# ============================================================

if [[ -z "${AGENT_ID}" ]]; then
    echo
    echo "Select this machine from machines.yaml:"
    echo

    python3 - <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
for i,m in enumerate(data.get("machines",[]),1):
    print(f"{i}. {m.get('machine_id')} ({m.get('gpu',{}).get('type')})")
PY

    echo
    read -p "Enter number: " IDX

    AGENT_ID=$(python3 - <<PY
import yaml,sys
data=yaml.safe_load(open("$MACHINES_PATH"))
idx=int("$IDX")-1
machines=data.get("machines",[])
if idx<0 or idx>=len(machines):
    print("")
else:
    print(machines[idx].get("machine_id"))
PY
)

    if [[ -z "$AGENT_ID" ]]; then
        echo "Invalid selection."
        exit 1
    fi

    echo
    echo "Persisting AGENT_ID=$AGENT_ID to $SECRETS_FILE"
    echo "AGENT_ID=\"$AGENT_ID\"" >> "$SECRETS_FILE"
fi

echo "Using AGENT_ID: $AGENT_ID"

# ============================================================
# Determine architecture from machines.yaml
# ============================================================

ARCH=$(python3 - <<PY
import yaml
data=yaml.safe_load(open("$MACHINES_PATH"))
for m in data.get("machines",[]):
    if m.get("machine_id")=="$AGENT_ID":
        print(m.get("gpu",{}).get("type"))
        break
PY
)

if [[ "$ARCH" != "apple" && "$ARCH" != "nvidia" ]]; then
    echo "Could not determine architecture for $AGENT_ID"
    exit 1
fi

echo "Detected architecture: $ARCH"

# ============================================================
# Validate JSON
# ============================================================

python3 - <<PY
import json
json.load(open("$REGISTRY_PATH"))
PY

if [[ $? -ne 0 ]]; then
    echo "models.json invalid. Fix before running."
    exit 1
fi

# ============================================================
# Tracking arrays
# ============================================================

declare -a SUCCESS
declare -a FAILED

# ============================================================
# Read registry
# ============================================================

read_registry() {
python3 - <<PY
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
# Sync Loop
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
            FAILED+=("Ollama:$a (ollama missing)")
        fi
        ;;

    MLX)
        echo "[MLX] $a"
        target="$MLX_DIR/$(basename $a)"

        if [[ -d "$target" ]]; then
            SUCCESS+=("MLX:$a (exists)")
        else
            if command -v huggingface-cli >/dev/null 2>&1; then
                if huggingface-cli download "$a" --local-dir "$target"; then
                    SUCCESS+=("MLX:$a")
                else
                    FAILED+=("MLX:$a")
                fi
            else
                FAILED+=("MLX:$a (hf-cli missing)")
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
            if command -v curl >/dev/null 2>&1; then
                if curl -L --fail -o "$out" "$url"; then
                    SUCCESS+=("Comfy:$fname")
                else
                    FAILED+=("Comfy:$fname")
                fi
            else
                FAILED+=("Comfy:$fname (curl missing)")
            fi
        fi
        ;;

    esac

done < <(read_registry)

# ============================================================
# Summary
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
