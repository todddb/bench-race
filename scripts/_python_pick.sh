#!/usr/bin/env bash
set -euo pipefail

# _python_pick.sh – Selects a stable Python interpreter for venv creation.
#
# Environment variables:
#   BENCH_RACE_PYTHON              – Force a specific interpreter (path or command)
#   BENCH_RACE_ALLOW_PRERELEASE_PYTHON=1 – Allow Python 3.14+ (prerelease/bleeding-edge)
#
# Output:
#   stdout: the interpreter command/path (for capture)
#   stderr: informational messages about selection

MAX_MINOR=13  # Reject Python >= 3.14 by default (wheels often lag)

# Check if a Python interpreter's version is acceptable.
# Returns 0 if acceptable, 1 if too new (and prerelease not allowed).
check_version() {
  local py="$1"
  local version_output
  version_output="$("$py" --version 2>&1)" || return 1

  # Extract version: "Python 3.12.1" -> "3.12.1"
  local version="${version_output#Python }"
  local minor
  minor="$(echo "$version" | cut -d. -f2)"

  if [[ "$minor" -gt "$MAX_MINOR" ]]; then
    if [[ "${BENCH_RACE_ALLOW_PRERELEASE_PYTHON:-0}" == "1" ]]; then
      echo "⚠ Warning: Using Python $version (prerelease allowed via BENCH_RACE_ALLOW_PRERELEASE_PYTHON)" >&2
      return 0
    else
      echo "⚠ Skipping $py ($version): Python >= 3.14 may lack wheel support for pinned deps." >&2
      echo "  Set BENCH_RACE_ALLOW_PRERELEASE_PYTHON=1 to override." >&2
      return 1
    fi
  fi
  return 0
}

# Select and validate the chosen interpreter.
select_python() {
  local py="$1"
  if ! command -v "$py" >/dev/null 2>&1; then
    return 1
  fi
  if ! check_version "$py"; then
    return 1
  fi
  # Print info to stderr, path to stdout
  local version_output
  version_output="$("$py" --version 2>&1)"
  echo "✓ Selected: $py ($version_output)" >&2
  echo "$py"
  exit 0
}

# 1. Honor explicit BENCH_RACE_PYTHON override
if [[ -n "${BENCH_RACE_PYTHON:-}" ]]; then
  if command -v "$BENCH_RACE_PYTHON" >/dev/null 2>&1; then
    if check_version "$BENCH_RACE_PYTHON"; then
      version_output="$("$BENCH_RACE_PYTHON" --version 2>&1)"
      echo "✓ Selected (BENCH_RACE_PYTHON): $BENCH_RACE_PYTHON ($version_output)" >&2
      echo "$BENCH_RACE_PYTHON"
      exit 0
    fi
  else
    echo "ERROR: BENCH_RACE_PYTHON='$BENCH_RACE_PYTHON' not found or not executable." >&2
    exit 1
  fi
fi

# 2. Prefer stable, widely-supported Python versions (newest stable first)
for p in python3.12 python3.11 python3.10 python3.9 python3; do
  select_python "$p" 2>/dev/null || true
done

# 3. If we get here, no suitable Python found
echo "" >&2
echo "ERROR: No suitable Python interpreter found." >&2
echo "       Requires Python 3.9–3.13 (3.14+ rejected by default due to wheel lag)." >&2
echo "" >&2
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Suggestion (macOS): brew install python@3.12" >&2
else
  echo "Suggestion: Install Python 3.12 via your package manager." >&2
fi
echo "" >&2
echo "Alternatively, set BENCH_RACE_PYTHON=/path/to/python3.x" >&2
echo "Or set BENCH_RACE_ALLOW_PRERELEASE_PYTHON=1 to use Python 3.14+." >&2
exit 1
