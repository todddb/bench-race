#!/usr/bin/env bash
set -euo pipefail

REQUESTED="${1:-}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if [[ -z "${REQUESTED}" ]]; then
  echo "Usage: $0 <requested-model-id>" >&2
  exit 1
fi

mapfile -t TAGS < <(curl -fsS "${OLLAMA_HOST}/api/tags" | python3 -c 'import json,sys; data=json.load(sys.stdin); [print(m.get("name","")) for m in data.get("models",[]) if m.get("name")]')

resolve_tag() {
  local requested="$1"
  shift
  local tags=("$@")

  for tag in "${tags[@]}"; do
    [[ "$tag" == "$requested" ]] && { echo "$tag"; return 0; }
  done

  local candidates=()
  for tag in "${tags[@]}"; do
    [[ "$tag" == "$requested"* ]] && candidates+=("$tag")
  done
  if [[ ${#candidates[@]} -eq 0 ]]; then
    return 1
  fi

  IFS=$'\n' read -r -d '' -a sorted < <(printf '%s\n' "${candidates[@]}" | sort && printf '\0')
  for tag in "${sorted[@]}"; do
    if [[ "${tag,,}" == *q4* ]]; then
      echo "$tag"
      return 0
    fi
  done

  printf '%s\n' "${sorted[@]}" | awk '{ print length, $0 }' | sort -n -k1,1 -k2,2 | head -n1 | cut -d' ' -f2-
}

if ! RESOLVED="$(resolve_tag "$REQUESTED" "${TAGS[@]}")"; then
  echo "No ollama tag resolved for model '${REQUESTED}'." >&2
  exit 2
fi

echo "Mapped requested model '${REQUESTED}' -> '${RESOLVED}'"
exec ollama pull "${RESOLVED}"
