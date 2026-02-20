#!/usr/bin/env bash
set -euo pipefail
# Sanitize a model name for filesystem use.
# Usage: ./agent/scripts/sanitize_name.sh <model-id>
input="$1"
echo "$input" | sed -E 's/[^A-Za-z0-9._-]/_/g' | sed -E 's/_+/_/g' | sed -E 's/^_+//; s/_+$//' | cut -c1-90
