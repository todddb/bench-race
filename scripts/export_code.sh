#!/usr/bin/env bash
set -euo pipefail

# scripts/export_code.sh
# Focused exporter (git-tracked first), macOS-safe (bash 3.2+)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/export"
TS="$(date +"%Y%m%dT%H%M%S")"
OUT_FILE="${OUT_DIR}/bench-race-code-export-${TS}.txt"

mkdir -p "$OUT_DIR"

# Skip huge files (bytes). Adjust if needed.
MAX_BYTES="${MAX_BYTES:-1048576}"  # 1 MiB

# Directories we never want to traverse (fallback mode).
PRUNE_DIRS=(
  ".git" ".svn" ".hg"
  "__pycache__" ".pytest_cache" ".mypy_cache" ".ruff_cache" ".tox"
  ".venv" "venv" "env"
  "node_modules"
  "dist" "build" "coverage"
  "logs" "log" "tmp" "temp"
  "ollama" "models"
  ".idea" ".vscode" ".cache"
)

# File extensions we consider "code" (fallback mode).
CODE_EXTS=(
  "py" "js" "ts" "tsx" "jsx"
  "sh" "bash" "zsh"
  "json" "yaml" "yml" "toml"
  "md" "txt"
  "sql" "ini" "cfg"
  "go" "rs" "java" "kt" "c" "h" "cpp" "hpp"
)

is_text_file() {
  local f="$1"
  local size
  size=$(wc -c < "$f" 2>/dev/null || echo 0)

  # Empty files are fine (treat as text)
  if [[ "$size" -eq 0 ]]; then
    return 0
  fi

  # grep -I: treat binary as no match; -q: quiet
  # If it's text, this usually returns 0 quickly.
  LC_ALL=C grep -Iq . "$f"
}

write_file_block() {
  local rel="$1"
  local f="${ROOT_DIR}/${rel}"

  [[ -f "$f" ]] || return 0

  local size
  size=$(wc -c < "$f" 2>/dev/null || echo 0)

  echo "----- FILE: ${rel} -----" >> "$OUT_FILE"

  if [[ "$size" -gt "$MAX_BYTES" ]]; then
    echo "SKIPPED: file too large (${size} bytes > ${MAX_BYTES})" >> "$OUT_FILE"
    echo >> "$OUT_FILE"
    return 0
  fi

  if ! is_text_file "$f"; then
    echo "SKIPPED: non-text/binary" >> "$OUT_FILE"
    echo >> "$OUT_FILE"
    return 0
  fi

  echo "Size: ${size} bytes" >> "$OUT_FILE"
  echo >> "$OUT_FILE"
  cat "$f" >> "$OUT_FILE"
  echo >> "$OUT_FILE"
  echo >> "$OUT_FILE"
}

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

cd "$ROOT_DIR"

{
  echo "bench-race code export"
  echo "Generated: $(date)"
  echo "Root: $ROOT_DIR"
  echo "Mode: git-tracked preferred; fallback find"
  echo "MAX_BYTES: $MAX_BYTES"
  echo
} > "$OUT_FILE"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Mode: git-tracked files only"

  {
    git ls-files
    git diff --name-only --cached
  } | awk 'NF' | sort -u > "$tmp_list"
else
  echo "Mode: fallback (extensions allowlist + directory pruning)"

  # Build prune expression for find
  PRUNE_EXPR=()
  for d in "${PRUNE_DIRS[@]}"; do
    PRUNE_EXPR+=( -name "$d" -o )
  done
  unset 'PRUNE_EXPR[${#PRUNE_EXPR[@]}-1]'

  # Build extension expression
  EXT_EXPR=()
  for ext in "${CODE_EXTS[@]}"; do
    EXT_EXPR+=( -iname "*.${ext}" -o )
  done
  unset 'EXT_EXPR[${#EXT_EXPR[@]}-1]'

  find "$ROOT_DIR" \
    \( -type d \( "${PRUNE_EXPR[@]}" \) -prune \) -o \
    \( -type f \( "${EXT_EXPR[@]}" \) -print \) \
    | sed "s|^$ROOT_DIR/||" \
    | sort -u > "$tmp_list"
fi

count=0
skipped=0

while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue

  # Extra guardrails even in git mode
  case "$rel" in
    */.venv/*|*/venv/*|*/node_modules/*|*/logs/*|*/ollama/*|*/models/*|*/dist/*|*/build/*)
      continue
      ;;
  esac

  if [[ -f "$ROOT_DIR/$rel" ]]; then
    before_size=$(wc -c < "$OUT_FILE" 2>/dev/null || echo 0)
    write_file_block "$rel"
    after_size=$(wc -c < "$OUT_FILE" 2>/dev/null || echo 0)
    if [[ "$after_size" -gt "$before_size" ]]; then
      count=$((count + 1))
    else
      skipped=$((skipped + 1))
    fi
  fi
done < "$tmp_list"

echo
echo "✅ Export complete"
echo "   Files included blocks: $count"
echo "   Output: $OUT_FILE"

