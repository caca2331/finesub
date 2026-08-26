#!/usr/bin/env bash
# Run VAD+ASR+stabilize+SRT on the screened vocal tracks.
set -e
export PYTHONPATH="$(pwd)/src"
ASSETS=${FINESUB_DATA_ROOT:-.}/assets/bilibili
SCREEN=${FINESUB_RATE_WORK:-out/separator-rate}/screen
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/down

for id in BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  for rate in 44100 22050 16000; do
    dir="$OUT/${id}-${rate}"
    if [ -f "$dir/${id}-raw.srt" ]; then
      echo "skip $dir"
      continue
    fi
    mkdir -p "$dir"
    cp "$SCREEN/${id}-${rate}.flac" "$dir/${id}-vocal.flac"
    echo "=== $id @ $rate"
    python -m finesub.pipeline "$ASSETS/$id.ogg" -o "$dir/${id}.srt" \
      --stage raw-srt --language ja 2>&1 | grep -E "语音识别摘要|字幕稳定化摘要"
  done
done
echo "downstream done"
