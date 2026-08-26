#!/usr/bin/env bash
# Null control: same 44.1 kHz rate, only the block grid moves.
#
# Production already accepts that block edges depend on the worker count and
# that Roformer is chunk-sensitive even with the pad (docs/gpu-profiles.md), so
# this perturbs the waveform without any claim of quality change. Whatever CER
# penalty it draws against the reference is the metric's own floor.
set -e
export PYTHONPATH="$(pwd)/src"
ASSETS=${FINESUB_DATA_ROOT:-.}/assets/bilibili
OUT=${FINESUB_RATE_WORK:-out/separator-rate}/null

for id in BV1cqLR6hEp3 BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  dir="$OUT/$id"
  if [ -f "$dir/${id}-raw.srt" ]; then
    echo "skip $dir"
    continue
  fi
  mkdir -p "$dir"
  echo "=== $id null (44100, block-seconds 120)"
  python -m tools.separator_benchmark "$ASSETS/$id.ogg" "$dir/${id}-vocal.flac" \
    --mode amp --gpu-budget-gb 4 --block-seconds 120 \
    --result "$dir/${id}-sep.json" 2>&1 | grep -E '"elapsed_sec":'
  python -m finesub.pipeline "$ASSETS/$id.ogg" -o "$dir/${id}.srt" \
    --stage raw-srt --language ja 2>&1 | grep -E "语音识别摘要|字幕稳定化摘要"
done
echo "null arm done"
