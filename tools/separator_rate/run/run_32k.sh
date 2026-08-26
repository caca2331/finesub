#!/usr/bin/env bash
# Add the 32 kHz rate on the same five materials, same layout as the screen.
set -e
export PYTHONPATH="$(pwd)/src"
ASSETS=${FINESUB_DATA_ROOT:-.}/assets/bilibili
SCREEN=${FINESUB_RATE_WORK:-out/separator-rate}/screen
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/down
mkdir -p "$SCREEN"

for id in BV1cqLR6hEp3 BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  flac="$SCREEN/${id}-32000.flac"
  if [ ! -f "$flac" ]; then
    echo "=== separate $id @ 32000"
    python -m tools.separator_benchmark "$ASSETS/$id.ogg" "$flac" \
      --mode amp --gpu-budget-gb 4 --model-sample-rate 32000 \
      --result "$SCREEN/${id}-32000.json" 2>&1 | grep -E '"elapsed_sec":'
  fi
  dir="$OUT/${id}-32000"
  if [ -f "$dir/${id}-raw.srt" ]; then
    echo "skip $dir"
    continue
  fi
  mkdir -p "$dir"
  cp "$flac" "$dir/${id}-vocal.flac"
  echo "=== downstream $id @ 32000"
  python -m finesub.pipeline "$ASSETS/$id.ogg" -o "$dir/${id}.srt" \
    --stage raw-srt --language ja 2>&1 | grep -E "语音识别摘要|字幕稳定化摘要"
done

# BV1cqLR6hEp3's other rates live in pipeA/pipeC/pipeB; mirror the 44.1k baseline
# into the screen layout so the interval analysis can address every arm the same way.
if [ ! -f "$SCREEN/BV1cqLR6hEp3-44100.flac" ]; then
  cp ${FINESUB_RATE_WORK:-out/separator-rate}/A3.flac "$SCREEN/BV1cqLR6hEp3-44100.flac"
  cp ${FINESUB_RATE_WORK:-out/separator-rate}/C1.flac "$SCREEN/BV1cqLR6hEp3-22050.flac"
  cp ${FINESUB_RATE_WORK:-out/separator-rate}/B3.flac "$SCREEN/BV1cqLR6hEp3-16000.flac"
fi
echo "32k done"
