#!/usr/bin/env bash
# Headline throughput: the same three arms with NO forward instrumentation.
#
# The per-forward CUDA sync costs ~3.3s over 34 chunks on the short material --
# it serialises the H2D copy against the previous chunk's CPU overlap-add. Fine
# for attributing time, wrong for reporting it. Model load is still separated,
# because separator_build_sec is plain wall clock around the build call.
set -eo pipefail
MAIN=${FINESUB_DATA_ROOT:-.}
export PYTHONPATH="$(pwd)/src"
export FINESUB_SEPARATOR_ACCEL=0
PY="$MAIN/tmp/venv-torch211/Scripts/python.exe"
ACCEL="$(cd "$MAIN/cache/separator-accel/v1-2.11.0+cu128-cuda12.8-sm120-6a790594" && pwd)"
export TORCHINDUCTOR_CACHE_DIR="$ACCEL/inductor"
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/accel
mkdir -p "$OUT"

run() {
  local label="$1"; shift
  local material="$1"; shift
  local profile="$1"; shift
  echo "=== $label"
  "$PY" -m tools.separator_benchmark "$material" "$OUT/$label.flac" \
    --mode amp --gpu-tier "$profile" \
    --result "$OUT/$label.json" "$@" 2>&1 | grep -E '"elapsed_sec":'
}

SHORT=tmp/native/BV1kYLR6AEXv.wav
LONG=$MAIN/assets/bilibili/BV1ojjc6MEAs.ogg

run "c-short-eager" "$SHORT" 4
run "c-short-jit"   "$SHORT" 4 --torch-compile --compile-scope all
run "c-short-aoti"  "$SHORT" 4 --aoti-transformer-dir "$ACCEL/aoti"
run "c-long-eager"  "$LONG"  8
run "c-long-jit"    "$LONG"  8 --torch-compile --compile-scope all
run "c-long-aoti"   "$LONG"  8 --aoti-transformer-dir "$ACCEL/aoti"
echo "clean bench done"
