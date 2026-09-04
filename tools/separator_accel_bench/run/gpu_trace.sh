#!/usr/bin/env bash
# Does the warm restore actually use the GPU, or only hold a context?
#
# Sample utilisation while one arm runs, then read the trace against the phase
# durations the tool already reports. The phases are strictly sequential:
#   imports -> build/load -> warmup forward #1 -> warmup forward #2 -> chunks
# so a long near-zero stretch followed by a busy one locates the restore.
set -e
MAIN=${FINESUB_DATA_ROOT:-.}
export PYTHONPATH="$(pwd)/src"
export FINESUB_SEPARATOR_ACCEL=0
PY="$MAIN/tmp/venv-torch211/Scripts/python.exe"
ACCEL="$(cd "$MAIN/cache/separator-accel/v1-2.11.0+cu128-cuda12.8-sm120-6a790594" && pwd)"
export TORCHINDUCTOR_CACHE_DIR="$ACCEL/inductor"
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/accel
LABEL="$1"; shift

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
  --format=csv,noheader -lms 200 -f "$OUT/$LABEL-gpu.csv" &
SMI=$!
sleep 1
"$PY" -m tools.separator_benchmark tmp/native/BV1kYLR6AEXv.wav "$OUT/$LABEL.flac" \
  --mode amp --gpu-tier entry --time-forwards --probe-compile-timing \
  --result "$OUT/$LABEL.json" "$@" 2>&1 \
  | grep -E 'Load model duration|Separation duration|"elapsed_sec":' || true
kill $SMI 2>/dev/null || true
wait $SMI 2>/dev/null || true
echo "trace -> $OUT/$LABEL-gpu.csv"
