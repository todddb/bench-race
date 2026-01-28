#!/usr/bin/env bash
set -euo pipefail

# Drop-in replacement for scripts/export_code.sh
# Focuses on *your* code, avoids venv/site-packages/node_modules/logs/models/etc.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/export"
TS="$(date +"%Y%m%dT%H%M%S")"
OUT_FILE="${OUT_DIR}/bench-race-code-export-${TS}.txt"

mkdir -p "$OUT_DIR"

echo "Exporting code from: $ROOT_DIR"
echo "Output file: $OUT_FILE"
echo

# -------- config --------
# Directories we never want to traverse (fallback mode).
PRUNE_DIRS=(
  ".git"
  ".svn"
  ".hg"
  ".DS_Store"
  "__pycache__"
  ".pytest_cache"
  ".mypy_cache"
  ".ruff_cache"
  ".tox"
  ".venv"
  "venv"
  "node_modules"
  "dist"
  "build"
  "coverage"
  ".coverage"
  "logs"
  "log"
  "tmp"
  "temp"
  "ollama"
  "models"
  ".idea"
  ".vscode"
  ".cache"
)

# File extensions we consider "code" (fallback mode).
CODE_EXTS=(
  "py" "js" "ts" "tsx" "jsx"
  "sh" "bash" "zsh"
  "json" "yaml" "yml" "toml"
  "md" "txt"
  "sql"
  "ini" "cfg"
  "Dockerfile"
  "Makefile"
  "go" "rs" "java" "kt" "c" "h" "cpp" "hpp"
)

# Skip huge files (bytes). Adjust if needed.
MAX_BYTES=$(( 1024 * 1024 ))  # 1 MiB

# -------- helpers --------
is_binary() {
  # Heuristic: treat as binary if it contains NUL byte
  LC_ALL=C grep -q $'\x00' "$1" 2>/dev/null
}

write_file_block() {
  local f="$1"

  # Skip non-regular
  [[ -f "$f" ]] || return 0

  # Skip huge
  local size
  size=$(wc -c < "$f" 2>/dev/null || echo 0)
  if [[ "$size" -gt "$MAX_BYTES" ]]; then
    echo "----- FILE: ${f#"$ROOT_DIR"/} -----" >> "$OUT_FILE"
    echo "SKIPPED: file too large (${size} bytes > ${MAX_BYTES})" >> "$OUT_FILE"
    echo >> "$OUT_FILE"
    return 0
  fi

  # Skip binaries
  if is_binary "$f"; then
    echo "----- FILE: ${f#"$ROOT_DIR"/} -----" >> "$OUT_FILE"
    echo "SKIPPED: binary file" >> "$OUT_FILE"
    echo >> "$OUT_FILE"
    return 0
  fi

  echo "----- FILE: ${f#"$ROOT_DIR"/} -----" >> "$OUT_FILE"
  # Basic metadata (portable-ish)
  echo "Size: ${size} bytes" >> "$OUT_FILE"
  echo >> "$OUT_FILE"
  cat "$f" >> "$OUT_FILE"
  echo >> "$OUT_FILE"
  echo >> "$OUT_FILE"
}

# -------- build file list --------
tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

cd "$ROOT_DIR"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Mode: git-tracked files only"
  echo

  # Only tracked + staged (in case you haven't committed yet)
  {
    git ls-files
    git diff --name-only --cached
  } | awk 'NF' | sort -u > "$tmp_list"
else
  echo "Mode: fallback (extensions allowlist + directory pruning)"
  echo

  # Build prune expression for find
  PRUNE_EXPR=()
  for d in "${PRUNE_DIRS[@]}"; do
    PRUNE_EXPR+=( -name "$d" -o )
  done
  # remove trailing -o
  unset 'PRUNE_EXPR[${#PRUNE_EXPR[@]}-1]'

  # Build extension expression
  EXT_EXPR=()
  for ext in "${CODE_EXTS[@]}"; do
    if [[ "$ext" == "Dockerfile" || "$ext" == "Makefile" ]]; then
      EXT_EXPR+=( -name "$ext" -o )
    else
      EXT_EXPR+=( -iname "*.${ext}" -o )
    fi
  done
  unset 'EXT_EXPR[${#EXT_EXPR[@]}-1]'

  # Find code-ish files, excluding pruned dirs
  find "$ROOT_DIR" \
    \( -type d \( "${PRUNE_EXPR[@]}" \) -prune \) -o \
    \( -type f \( "${EXT_EXPR[@]}" \) -print \) \
    | sed "s|^$ROOT_DIR/||" \
    | sort -u > "$tmp_list"
fi

# -------- write output --------
{
  echo "bench-race code export"
  echo "Generated: $(date)"
  echo "Root: $ROOT_DIR"
  echo
} > "$OUT_FILE"

count=0
while IFS= read -r rel; do
  [[ -n "$rel" ]] || continue

  # Extra guardrails even in git mode
  case "$rel" in
    */.venv/*|*/venv/*|*/node_modules/*|*/logs/*|*/ollama/*|*/models/*|*/dist/*|*/build/*)
      continue
      ;;
  esac

  f="$ROOT_DIR/$rel"
  if [[ -f "$f" ]]; then
    write_file_block "$f"
    count=$((count + 1))
  fi
done < "$tmp_list"

echo "Done. Exported $count files."
echo "Wrote: $OUT_FILE"

