#!/usr/bin/env bash
# A fourth interior point at 44100/1.5 = 29400, to test the straight line with
# three points between the endpoints instead of one.
set -e
export PYTHONPATH="$(pwd)/src"
ASSETS=${FINESUB_DATA_ROOT:-.}/assets/bilibili
SCREEN=${FINESUB_RATE_WORK:-out/separator-rate}/screen
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/down

for id in BV1cqLR6hEp3 BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  flac="$SCREEN/${id}-29400.flac"
  if [ ! -f "$flac" ]; then
    echo "=== separate $id @ 29400"
    python -m tools.separator_benchmark "$ASSETS/$id.ogg" "$flac" \
      --mode amp --gpu-budget-gb 4 --model-sample-rate 29400 \
      --result "$SCREEN/${id}-29400.json" 2>&1 | grep -E '"elapsed_sec":'
  fi
  dir="$OUT/${id}-29400"
  if [ -f "$dir/${id}-raw.srt" ]; then
    echo "skip $dir"
    continue
  fi
  mkdir -p "$dir"
  cp "$flac" "$dir/${id}-vocal.flac"
  echo "=== downstream $id @ 29400"
  python -m finesub.pipeline "$ASSETS/$id.ogg" -o "$dir/${id}.srt" \
    --stage raw-srt --language ja 2>&1 | grep -E "语音识别摘要|字幕稳定化摘要"
done
echo "29400 done"
