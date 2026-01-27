#!/usr/bin/env bash
set -euo pipefail

# Prefer an explicit PYTHON=... override.
if [ -n "${PYTHON:-}" ]; then
  echo "$PYTHON"
  exit 0
fi

# Prefer Homebrew python if present, then fallback to system python3.
for p in /opt/homebrew/bin/python3 python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null 2>&1; then
    echo "$p"
    exit 0
  fi
done

echo "ERROR: No python3 found." >&2
exit 1
