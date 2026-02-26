#!/usr/bin/env bash
set -euo pipefail

BASE="${WRAPPER_BASE_URL:-http://127.0.0.1:9002}"
MLX_MODEL="${SMOKE_MLX_MODEL:-mlx-community/Meta-Llama-3-8B-Instruct-4bit}"
TRT_MODEL="${SMOKE_TRT_MODEL:-distilgpt2}"

echo "[1/6] GET /v1/models"
curl -fsS "$BASE/v1/models" | jq .

echo "[2/6] POST /v1/models/start (MLX)"
curl -fsS -X POST "$BASE/v1/models/start" \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\": \"$MLX_MODEL\"}" | jq .

echo "[3/6] POST /v1/infer/stream"
curl -fsS -N -X POST "$BASE/v1/infer/stream" \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\": \"$MLX_MODEL\", \"prompt\": \"Say hello in one short sentence\", \"max_tokens\": 32}" | sed -n '1,80p'

echo "[4/6] POST /v1/models/switch (TRT)"
curl -fsS -X POST "$BASE/v1/models/switch" \
  -H 'Content-Type: application/json' \
  -d "{\"model_id\": \"$TRT_MODEL\"}" | jq .

echo "[5/6] GET /v1/health"
curl -fsS "$BASE/v1/health" | jq .

echo "[6/6] POST /v1/models/stop"
curl -fsS -X POST "$BASE/v1/models/stop" \
  -H 'Content-Type: application/json' \
  -d '{"backend": "trt"}' | jq .

echo "Smoke tests completed."
