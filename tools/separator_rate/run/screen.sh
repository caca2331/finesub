#!/usr/bin/env bash
# Separation-only screen: three rates over several materials, no ASR yet.
set -e
export PYTHONPATH="$(pwd)/src"
ASSETS=${FINESUB_DATA_ROOT:-.}/assets/bilibili
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/screen
mkdir -p "$OUT"

for id in BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  for rate in 44100 22050 16000; do
    target="$OUT/${id}-${rate}.flac"
    if [ -f "$target" ]; then
      echo "skip $target"
      continue
    fi
    extra=""
    if [ "$rate" != "44100" ]; then
      extra="--model-sample-rate $rate"
    fi
    echo "=== $id @ $rate"
    python -m tools.separator_benchmark "$ASSETS/$id.ogg" "$target" \
      --mode amp --gpu-budget-gb 4 $extra \
      --result "$OUT/${id}-${rate}.json" 2>&1 | grep -E '"elapsed_sec":'
  done
done
echo "screen done"
