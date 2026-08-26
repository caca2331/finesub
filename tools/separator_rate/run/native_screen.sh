#!/usr/bin/env bash
# Redo the rate screen under the condition production actually runs today:
# native-rate stereo input, not the stale 16 kHz mono .ogg cache.
#
# Separation only. The failure mode that condemns 16 kHz (whole passages masked
# to the noise floor) is visible on the vocal track alone, so this half answers
# the decisive question without paying for ASR.
set -e
export PYTHONPATH="$(pwd)/src"
SRC=tmp/native
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/native
mkdir -p "$OUT"

for id in BV1cqLR6hEp3 BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  for rate in 44100 32000 22050 16000; do
    target="$OUT/${id}-${rate}.flac"
    if [ -f "$target" ]; then
      echo "skip $target"
      continue
    fi
    extra=""
    if [ "$rate" != "44100" ]; then
      extra="--model-sample-rate $rate"
    fi
    echo "=== $id @ $rate (native input)"
    python -m tools.separator_benchmark "$SRC/$id.wav" "$target" \
      --mode amp --gpu-budget-gb 4 $extra \
      --result "$OUT/${id}-${rate}.json" 2>&1 | grep -E '"elapsed_sec":'
  done
done
echo "native screen done"
