#!/usr/bin/env bash
# Pull the native-rate stereo audio out of each .mp4.
#
# The tracked assets/bilibili/*.ogg are 16 kHz mono -- an artefact of the old
# download path, which re-encoded before separation ever saw the audio.
# download_audio no longer does that (media/source.py keeps whatever yt-dlp
# gave it), so the .ogg is stale cache and every arm measured on it was
# measured under the wrong acoustic condition.
set -e
ASSETS=${FINESUB_DATA_ROOT:-.}/assets/bilibili
OUT=tmp/native
mkdir -p "$OUT"

for id in BV1cqLR6hEp3 BV1kYLR6AEXv BV1UBjq6fEgb BV1ySjz6FEzD BV1dwjP6LECU; do
  target="$OUT/$id.wav"
  if [ -f "$target" ]; then
    echo "skip $target"
    continue
  fi
  ffmpeg -v error -y -i "$ASSETS/$id.mp4" -vn -acodec pcm_s16le "$target"
  ffprobe -v error -show_entries stream=sample_rate,channels -show_entries format=duration \
    -of csv=p=0 "$target"
done
echo "extract done"
