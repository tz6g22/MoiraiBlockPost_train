#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${QWEN3_FORMAL_CONFIG:-$ROOT/qwen3_1.7b_config.yaml}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing Python executable: %s\n' "$PYTHON" >&2
  exit 1
fi

printf '[formal] config=%s\n' "$CONFIG"
printf '[formal] default stage is validate; pass --stage explicitly for later stages.\n'
exec "$PYTHON" -m src.formal.pipeline --config "$CONFIG" "$@"
