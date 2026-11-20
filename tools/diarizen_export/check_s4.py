"""S3+S4: clustering and assembly, against the frozen final timeline.

Clustering has no cached intermediate of its own, so it is scored together with the assembly
against `w16_default.npz` -- the rasterised annotation every published DiariZen reading used.

⚠ Fed with the CACHED local segmentation so this measures the second half of the port only. Our
own segmentation differs on about 0.02% of frames and would mix two readings into one.

    python tools/diarizen_export/check_s4.py --ids hina-32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

CACHE = REPO / "tmp" / "diarization-dz3"
EXPORT = REPO / "tmp" / "diarizen-export" / "diarizen_embedding.ts"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"


def align(ours, theirs):
    """Match our cluster columns to theirs by best overlap, so a permutation is not a difference."""
    from scipy.optimize import linear_sum_assignment

    width = max(ours.shape[1], theirs.shape[1])

    def pad(grid):
        if grid.shape[1] >= width:
            return grid
        return np.hstack([grid, np.zeros((len(grid), width - grid.shape[1]), dtype=grid.dtype)])

    a, b = pad(ours), pad(theirs)
    frames = min(len(a), len(b))
    table = a[:frames].T.astype(np.int64) @ b[:frames].astype(np.int64)
    rows, columns = linear_sum_assignment(-table)
    mapped = np.zeros_like(a)
    for source, target in zip(rows, columns):
        mapped[:, target] |= a[:, source]
    return mapped[:frames], b[:frames]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", default=["hina-32"])
    args = parser.parse_args()

    import soundfile
    import torch

    import accuracy_lab as lab
    from finesub.speech.diarization import diarizen_assemble as assembler
    from frame_grid import to_frame_grid
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_embed as embedder
    from finesub.speech.diarization import diarizen_runner as runner

    for ident in args.ids:
        local = np.load(CACHE / ident / "w16" / "diarizen-local.npz")["data"]
        want = np.load(CACHE / ident / "w16_default.npz")["probs"]
        signal, rate = soundfile.read(lab.source(ident), dtype="float32", always_2d=True)
        assert rate == runner.SAMPLE_RATE
        waveform = torch.from_numpy(signal.mean(axis=1))[None]

        filtered = runner.median_filtered(local)
        embeddings = embedder.embed(EXPORT, waveform, filtered)
        hard = clusterer.cluster(embeddings, filtered, PLDA).hard
        native = assembler.assemble(filtered, hard)
        # ⚠ The cache is `raster()`d onto the 80 ms grid the whole lab uses; `assemble` returns the
        # model's own 20 ms receptive field. Comparing them directly lines up our first quarter
        # against their whole file -- which is what made the first reading 92.9%.
        ours = to_frame_grid(native, target_step=0.08)

        print(f"{ident}: ours {ours.shape} (speakers {ours.shape[1]})  "
              f"cache {want.shape} (speakers {want.shape[1]})")
        mine, theirs = align(ours.astype(bool), want.astype(bool))
        either = mine.any(axis=1) | theirs.any(axis=1)
        if either.any():
            moved = float((mine[either] != theirs[either]).any(axis=1).mean())
            print(f"   frames where either speaks: {int(either.sum())}; "
                  f"leading assignment differs on {moved:.4%}")
        speech_ours = float(mine.any(axis=1).mean())
        speech_theirs = float(theirs.any(axis=1).mean())
        print(f"   speech share: ours {speech_ours:.4%} vs cache {speech_theirs:.4%}")


if __name__ == "__main__":
    main()
