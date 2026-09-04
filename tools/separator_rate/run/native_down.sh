#!/usr/bin/env bash
# Native-input rerun of the two rates that matter, plus the null control.
#
# The null arm (same 44.1 kHz, block grid moved) is not optional: without it the
# CER against a 44.1k-derived reference has no scale, and the ±0.04 band it
# draws is what decided every call in E12.
set -e
export PYTHONPATH="$(pwd)/src"
SRC=tmp/native
SCREEN=${FINESUB_RATE_WORK:-out/separator-rate}/native
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/native-down
mkdir -p "$OUT"

for id in BV1cqLR6hEp3 BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  # null: separate again at 44.1k with the grid moved
  nulldir="$OUT/${id}-null"
  if [ ! -f "$nulldir/${id}-raw.srt" ]; then
    mkdir -p "$nulldir"
    if [ ! -f "$nulldir/${id}-vocal.flac" ]; then
      echo "=== separate $id null (44100, block-seconds 120, native)"
      python -m tools.separator_benchmark "$SRC/$id.wav" "$nulldir/${id}-vocal.flac" \
        --mode amp --gpu-tier entry --block-seconds 120 \
        --result "$nulldir/${id}-sep.json" 2>&1 | grep -E '"elapsed_sec":'
    fi
    echo "=== downstream $id null"
    python -m finesub.pipeline "$SRC/$id.wav" -o "$nulldir/${id}.srt" \
      --stage raw-srt --language ja 2>&1 | grep -E "语音识别摘要|字幕稳定化摘要"
  fi

  for rate in 44100 22050; do
    dir="$OUT/${id}-${rate}"
    if [ -f "$dir/${id}-raw.srt" ]; then
      echo "skip $dir"
      continue
    fi
    mkdir -p "$dir"
    cp "$SCREEN/${id}-${rate}.flac" "$dir/${id}-vocal.flac"
    echo "=== downstream $id @ $rate (native)"
    python -m finesub.pipeline "$SRC/$id.wav" -o "$dir/${id}.srt" \
      --stage raw-srt --language ja 2>&1 | grep -E "语音识别摘要|字幕稳定化摘要"
  done
done
echo "native downstream done"
