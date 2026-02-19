#!/usr/bin/env bash
# Robust model-sanitizer:
# - Usage: ./scripts/model_sanitize.sh <model-id>
# - Or: echo "meta-llama/Llama-2-8b-chat-hf" | ./scripts/model_sanitize.sh
set -euo pipefail

# Read input: arg preferred, else stdin if piped
if [[ $# -ge 1 ]]; then
  input="$1"
else
  # If nothing on stdin and no arg, show usage
  if [ -t 0 ]; then
    cat >&2 <<'USAGE'
Usage: model_sanitize.sh <model-id>
Example: ./scripts/model_sanitize.sh "meta-llama/Llama-2-8b-chat-hf"

You can also pipe a model id:
  echo "meta-llama/Llama-2-8b-chat-hf" | ./scripts/model_sanitize.sh
USAGE
    exit 2
  fi
  # read piped stdin
  input="$(cat -)"
fi

# sanitize: replace any char not in A-Za-z0-9_.- with underscore,
# collapse multiple underscores, trim leading/trailing underscores, limit length.
echo "$input" \
  | sed -E 's/[^A-Za-z0-9_.-]/_/g' \
  | sed -E 's/_+/_/g' \
  | sed -E 's/^_+//; s/_+$//' \
  | cut -c1-90
