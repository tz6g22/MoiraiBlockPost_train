#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_PATH="${LOG_PATH:-$ROOT/outputs/logs/iridis-4xl4-$RUN_ID.log}"
PID_PATH="${PID_PATH:-$ROOT/outputs/logs/iridis-4xl4-$RUN_ID.pid}"
mkdir -p "$(dirname "$LOG_PATH")" "$(dirname "$PID_PATH")"

nohup setsid env \
  NPROC_PER_NODE=4 \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$ROOT/scripts/run_pipeline.sh" "$@" \
  >"$LOG_PATH" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_PATH"
printf 'PID=%s\nLOG=%s\nPID_FILE=%s\n' "$pid" "$LOG_PATH" "$PID_PATH"
