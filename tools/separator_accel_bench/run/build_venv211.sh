#!/usr/bin/env bash
# Build the torch 2.11 benchmark venv from the cached wheels.
#
# Lives in the MAIN checkout so it outlives this worktree. The torch trio comes
# from the local cache with --no-index, so this step is offline; only the small
# packages below touch the network.
set -e
MAIN=${FINESUB_DATA_ROOT:-.}
WHEELS="$(cd "$MAIN/cache/wheels/torch-2.11.0-cu128" && pwd)"
VENV="$MAIN/tmp/venv-torch211"

if [ ! -f "$VENV/Scripts/python.exe" ]; then
  python -m venv "$VENV"
fi
PY="$VENV/Scripts/python.exe"

"$PY" -m pip install --quiet --upgrade pip

echo "--- torch trio from the local cache (offline)"
"$PY" -m pip install --no-index --find-links "$WHEELS" \
  torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0

echo "--- everything else"
# triton-windows is pinned to the torch build (pyproject: it declares no torch
# constraint of its own, so a range lets the resolver pair them wrongly).
"$PY" -m pip install --quiet \
  "triton-windows==3.6.0.post26" \
  "audio-separator==0.44.3" \
  "librosa==0.11.0" \
  "audioread==3.1.0" \
  onnxruntime psutil numba soundfile

echo "--- versions"
"$PY" -c "import torch, torchaudio; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__)"
"$PY" -c "import triton; print('triton', triton.__version__)"
"$PY" -c "import audio_separator, onnxruntime; print('audio-separator ok, ort', onnxruntime.__version__)"
