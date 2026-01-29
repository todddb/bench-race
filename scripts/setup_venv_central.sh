#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR/central"

# Select Python interpreter (rejects Python 3.14+ by default)
if ! PYTHON="$("$SCRIPT_DIR/_python_pick.sh")"; then
  echo "" >&2
  echo "Failed to find a suitable Python interpreter for central venv." >&2
  exit 1
fi

$PYTHON -m venv .venv
source .venv/bin/activate

python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

echo ""
echo "Central venv ready at: $ROOT_DIR/central/.venv"
echo "Run: cd $ROOT_DIR/central && . .venv/bin/activate && python app.py"
