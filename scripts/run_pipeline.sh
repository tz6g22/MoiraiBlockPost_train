#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="${QWEN3_FORMAL_CONFIG:-$ROOT/qwen3_14b_config.yaml}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing Python executable: %s\n' "$PYTHON" >&2
  exit 1
fi

printf '[formal] config=%s\n' "$CONFIG"
printf '[formal] no formal training is started by this entry point.\n'
exec "$PYTHON" -m src.formal.pipeline --config "$CONFIG" "$@"
