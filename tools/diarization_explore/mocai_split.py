"""Split 魔裁's big mixed tracks: does re-diarizing a track's own audio pull its people apart?

Plan 5r.2 / 5r.3. On an 11-hour 魔裁 part the model does not give the streamer a slot: it isolates distinctive
minor voices into small, pure tracks (input1 s2: 0.32 h, 81% Warden) and dumps everyone else into two large
ones (s0 4.68 h, s1 3.36 h) that are ~40% outside every character cue -- the streamer -- and a mix of a dozen
characters inside. Any gain for >4 speakers has to come from those two.

The over4 recursive split is built for exactly this: concatenate the frames one track claims into a single
stream and run Sortformer on it again. People who shared a slot meet again inside one continuous stream, so
their identities are consistent by construction and need no cross-window association.

For each chosen parent track this reports how the sub-pass divides it:
  - per sub-track: its share of the parent, how much of it lies OUTSIDE every character cue (the streamer
    signal -- a sub-track that is mostly outside is the streamer being pulled out), and its top characters;
  - parent vs children purity on the character frames -- did splitting make anyone cleaner?

    python tools/diarization_explore/mocai_split.py 1 --tracks 0 1

⚠ Uses raw probabilities thresholded at 0.5 (over4's convention), not production's hysteresis smoothing,
so absolute numbers are not production's; the parent-vs-children comparison uses the same rule on both sides.
⚠ Concatenating non-contiguous runs puts hard joins in the audio -- inherent to the method (over4 does the same).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WT / "src"))

import vram_guard  # noqa: E402  -- before anything touches CUDA
import mocai_long as ML  # noqa: E402
import over4_v2 as V  # noqa: E402


def describe(label, frames_mask, truth, people):
    """Share outside every cue, and character purity, over the frames in `frames_mask`."""
    n = int(frames_mask.sum())
    if not n:
        return None
    inside = frames_mask & (truth >= 0)
    counts = np.bincount(truth[inside], minlength=len(people))
    tot = int(counts.sum())
    top = sorted(((c, people[i]) for i, c in enumerate(counts) if c), reverse=True)[:3]
    purity = (top[0][0] / tot) if tot else float("nan")
    return {
        "label": label, "hours": n * V.FRAME / 3600, "outside": 1 - int(inside.sum()) / n,
        "purity": purity, "top": [(name, c / tot) for c, name in top] if tot else [],
    }


def show(d, indent=""):
    top = ", ".join(f"{name} {share:.0%}" for name, share in d["top"])
    print(f"{indent}{d['label']:<10}{d['hours']:>7.2f} h   outside cues {d['outside']:>6.1%}   "
          f"purity {d['purity']:>6.1%}   {top}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", type=int)
    parser.add_argument("--tracks", type=int, nargs="+",
                        help="parent tracks to split (default: the two with the most active time -- the mixed ones)")
    args = parser.parse_args()

    vram_guard.install()
    import soundfile
    import torch
    from finesub.speech.diarization.runner import SortformerRunner
    from finesub.speech.diarization.sortformer_backend import resolve_artifacts

    tracks, cues, people, truth, track_of = ML.build(args.part)
    if not args.tracks:
        args.tracks = [int(k) for k in np.argsort(-track_of.sum(axis=0))[:2]]
    vocal = WT / "out" / f"mocai-in{args.part}" / f"mocai-in{args.part}-vocal.ogg"
    signal, _ = soundfile.read(str(vocal), dtype="float32", always_2d=True)
    waveform = signal.mean(axis=1)
    del signal
    runner = SortformerRunner(resolve_artifacts(), device="cuda")
    saved = {}

    for k in args.tracks:
        frames = np.flatnonzero(track_of[:, k])
        parent = describe(f"s{k}", track_of[:, k], truth, people)
        print(f"\n=== input{args.part} parent track ===")
        show(parent)
        audio, runs = V._concat(waveform, frames)
        probs = runner.diarize(torch.from_numpy(audio)).float().cpu().numpy()
        probs = probs.reshape(-1, probs.shape[-1])
        back = V._back(probs, runs)                                  # (len(frames), 4), original order
        active = (back > V.THRESHOLD).any(axis=1)
        assign = np.where(active, back.argmax(axis=1), -1)
        saved[f"frames_{k}"], saved[f"assign_{k}"] = frames, assign
        print(f"  sub-pass on {len(audio) / 16000 / 3600:.2f} h of its own audio "
              f"(guard peak {vram_guard.peak_gib():.2f} GiB):")
        children = []
        for j in range(back.shape[1]):
            mask = np.zeros(len(truth), dtype=bool)
            mask[frames[assign == j]] = True
            d = describe(f"  s{k}.{j}", mask, truth, people)
            if d and d["hours"] > 0.02:
                children.append(d)
                show(d, "  ")
        silent = int((assign < 0).sum()) * V.FRAME / 3600
        print(f"    (no sub-track above threshold: {silent:.2f} h)")
        if children:
            # did splitting make the character frames cleaner? weight each child by its character frames
            w = np.array([c["hours"] * (1 - c["outside"]) for c in children])
            child_purity = np.average([c["purity"] for c in children], weights=w) if w.sum() else float("nan")
            print(f"  purity on character frames: parent {parent['purity']:.1%} -> children (weighted) {child_purity:.1%}")
        torch.cuda.empty_cache()

    # Frame-level child assignment, so attribution can be scored (mocai_compose.py) without another GPU pass.
    out = WT / "out" / f"mocai-in{args.part}" / f"mocai-in{args.part}-split.npz"
    np.savez_compressed(out, tracks=np.array(args.tracks), **saved)
    print(f"\nsaved frame assignments -> {out.relative_to(WT)}")


if __name__ == "__main__":
    main()
