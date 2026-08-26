#!/usr/bin/env bash
# eager / torch.compile / AOTI under torch 2.11.0+cu128, warm caches only.
#
# All three arms go through the benchmark's own instrumentation with production
# accel switched off (FINESUB_SEPARATOR_ACCEL=0), so nothing is measured twice:
# left on, run_vocal_separation would pick AOTI by itself and the compiled arms
# would stack on top of it.
#
# --time-forwards makes the eager arm pay the same per-forward CUDA sync the
# compiled arms always pay; without it the baseline is measured on a different
# instrument than the things it is being compared against.
set -eo pipefail
MAIN=${FINESUB_DATA_ROOT:-.}
export PYTHONPATH="$(pwd)/src"
export FINESUB_SEPARATOR_ACCEL=0
PY="$MAIN/tmp/venv-torch211/Scripts/python.exe"
ACCEL="$(cd "$MAIN/cache/separator-accel/v1-2.11.0+cu128-cuda12.8-sm120-6a790594" && pwd)"
export TORCHINDUCTOR_CACHE_DIR="$ACCEL/inductor"
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/accel
mkdir -p "$OUT"

run() {  # label material profile extra...
  local label="$1"; shift
  local material="$1"; shift
  local profile="$1"; shift
  echo "=== $label"
  "$PY" -m tools.separator_benchmark "$material" "$OUT/$label.flac" \
    --mode amp --gpu-budget-gb "$profile" --time-forwards --probe-compile-timing \
    --result "$OUT/$label.json" "$@" 2>&1 \
    | grep -E '"elapsed_sec":|Load model duration|Separation duration'
}

SHORT=tmp/native/BV1kYLR6AEXv.wav          # 269.9s, 44.1k stereo
LONG=$MAIN/assets/bilibili/BV1ojjc6MEAs.ogg  # 2014.8s

for pass in cold warm; do
  run "short-jit-$pass"  "$SHORT" 4 --torch-compile --compile-scope all
done
run "short-eager" "$SHORT" 4
run "short-aoti"  "$SHORT" 4 --aoti-transformer-dir "$ACCEL/aoti"

for pass in cold warm; do
  run "long-jit-$pass" "$LONG" 8 --torch-compile --compile-scope all
done
run "long-eager" "$LONG" 8
run "long-aoti"  "$LONG" 8 --aoti-transformer-dir "$ACCEL/aoti"

echo "bench211 done"
