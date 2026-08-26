#!/usr/bin/env bash
# Fetch the pinned torch trio as wheels into a durable cache, then build a venv
# from that cache alone.
#
# Both live in the MAIN checkout, not this worktree, so they survive the
# worktree being removed. Re-creating the venv later is offline:
#   python -m venv tmp/venv-torch211
#   tmp/venv-torch211/Scripts/python -m pip install --no-index \
#       --find-links cache/wheels/torch-2.11.0-cu128 torch torchaudio torchvision
#
# repo-install.md is explicit that the torch trio must come from the cu128 index
# with NO --extra-index-url in the same command: PyPI's Windows torch has no
# CUDA, and pip gives no guarantee about which index wins when both are offered.
set -e
MAIN=${FINESUB_DATA_ROOT:-.}
WHEELS=$MAIN/cache/wheels/torch-2.11.0-cu128

python -m pip download torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  -d "$WHEELS"

echo "--- wheels cached:"
ls -la "$WHEELS"
