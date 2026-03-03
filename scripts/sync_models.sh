#!/usr/bin/env bash
# Sync models from config/registry/models.json for local agent runtime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGISTRY_PATH="${REGISTRY_PATH:-${REPO_ROOT}/config/registry/models.json}"
COMFY_DIR="${COMFY_DIR:-${REPO_ROOT}/agent/models/comfy}"

log() { printf '[sync_models] %s\n' "$*"; }
warn() { printf '[sync_models][warn] %s\n' "$*" >&2; }

if [[ ! -f "$REGISTRY_PATH" ]]; then
  echo "Registry not found: $REGISTRY_PATH" >&2
  exit 1
fi

if [[ -f /proc/driver/nvidia/version ]] || command -v nvidia-smi >/dev/null 2>&1; then
  ARCH="nvidia"
else
  ARCH="apple"
fi
log "Detected architecture: ${ARCH}"

read_registry() {
  python3 - "$REGISTRY_PATH" "$ARCH" <<'PY'
import json,sys
p=sys.argv[1]
arch=sys.argv[2]
data=json.load(open(p,'r',encoding='utf-8'))
for m in data.get('ollama',[]):
    tag=(m.get(arch) or m.get('apple') or m.get('nvidia') or '').strip()
    if tag:
        print('OLLAMA\t'+tag)
for m in data.get('custom',[]):
    mid=(m.get('id') or '').strip()
    if not mid:
        continue
    if arch=='apple':
        val=(m.get('mlx_hf_id') or '').strip()
        if val:
            print('CUSTOM_MLX\t'+mid+'\t'+val)
    else:
        hf=(m.get('trt-llm_hf_id') or '').strip()
        eng=(m.get('trt-llm_engine_dir') or '').strip()
        if hf or eng:
            print('CUSTOM_TRT\t'+mid+'\t'+hf+'\t'+eng)
for c in data.get('comfyui',[]):
    url=(c.get('download_url') or '').strip()
    cid=(c.get('id') or '').strip()
    if url:
        print('COMFY\t'+cid+'\t'+url)
PY
}

mkdir -p "$COMFY_DIR"

while IFS=$'\t' read -r kind a b c; do
  [[ -n "${kind:-}" ]] || continue
  case "$kind" in
    OLLAMA)
      if command -v ollama >/dev/null 2>&1; then
        log "ollama pull $a"
        ollama pull "$a"
      else
        warn "ollama missing; skipping tag $a"
      fi
      ;;
    CUSTOM_MLX)
      if command -v python3 >/dev/null 2>&1; then
        log "Custom MLX model $a ($b)"
        if python3 -m auto_llm --help >/dev/null 2>&1; then
          python3 -m auto_llm download "$b" || warn "auto_llm download failed for $b"
        else
          warn "python3 -m auto_llm unavailable; skipping $b"
        fi
      fi
      ;;
    CUSTOM_TRT)
      log "Custom TRT model $a (hf=${b:-n/a}, engine_dir=${c:-n/a})"
      if command -v trt-convert >/dev/null 2>&1; then
        trt-convert --model "${b}" --engine-dir "${c}" || warn "trt-convert failed for $a"
      else
        warn "trt-convert unavailable; skipping conversion for $a"
      fi
      ;;
    COMFY)
      fname="$(basename "$b")"
      out="${COMFY_DIR}/${fname}"
      if [[ -f "$out" ]]; then
        log "Comfy checkpoint exists: $out"
      elif command -v curl >/dev/null 2>&1; then
        log "Downloading comfy checkpoint $a"
        curl -L --fail --retry 3 -o "$out" "$b"
      elif command -v wget >/dev/null 2>&1; then
        log "Downloading comfy checkpoint $a"
        wget -O "$out" "$b"
      else
        warn "Neither curl nor wget available; skipping $b"
      fi
      ;;
  esac
done < <(read_registry)

log "Model sync complete"
