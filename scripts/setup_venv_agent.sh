#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/agent"

PYTHON="$("$SCRIPT_DIR/_python_pick.sh")"
echo "Using: $PYTHON ($($PYTHON --version))"

$PYTHON -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

echo "Agent venv ready."
echo "Run: cd $ROOT_DIR/agent && . .venv/bin/activate && uvicorn agent_app:app --host 0.0.0.0 --port 9001"
