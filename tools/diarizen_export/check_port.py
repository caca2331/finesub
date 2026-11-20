"""The number that decides: the whole port against the research path, per subtitle.

Everything before this scored one stage at a time on the cached input of the stage before it,
which is how you find where a difference comes from. This one runs the port end to end -- our
segmentation, our embeddings, our clustering, our assembly -- and asks the only question the
pre-registration says matters: how often does a subtitle line end up with a different speaker.

⚠ Per SUBTITLE, not per frame. A frame-level rate is not comparable to the band it has to be read
against: DiariZen's own perturbation readings (sigma=1e-7 -> 1.36%, PCM16 -> 1.14%, one sample of
shift -> 2.84%) are all per-subtitle. Comparing 2.4% of frames to 1.36% of labels would be
comparing two different quantities that happen to be printed the same way.

⚠ The comparison is permutation-invariant. Cluster ids carry no meaning across two runs, so the
timelines are matched by best overlap first -- otherwise a pure renumbering reads as total
disagreement.

    python tools/diarizen_export/check_port.py --ids hina-32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

CACHE = REPO / "tmp" / "diarization-dz3"
EXPORT_DIR = REPO / "tmp" / "diarizen-export"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"
LAB_FRAME = 0.08


def to_lab_grid(native, step: float):
    """The model's own frame rate down to the 80 ms grid every lab reading uses.

    ⚠ The beds' shared reduction (`frame_grid.py`), not a second copy of it -- a checker with its
    own arithmetic would be comparing something the other readings do not compute.
    """
    from frame_grid import to_frame_grid

    return to_frame_grid(native, source_step=step, target_step=LAB_FRAME)


def source_for(ident: str):
    """The vocal track for an ident, whichever corpus it belongs to.

    ⚠ `accuracy_lab.source` hardcodes `out/refined-2609/<ident>/`, which is where the hina set
    lives. The onetonton tracks are at `out/<ident>/` and were nearly written off as missing
    because of it -- `data/index.md` records that corpus as subtitles only, which is true of
    `data/refined-multi/` and says nothing about `out/`.
    """
    import accuracy_lab as lab

    first = lab.source(ident)
    if first.exists():
        return first
    return REPO / "out" / ident / f"{ident}-vocal.ogg"


def episode_for(ident: str):
    """`(stable, grid, starts, reference, people)` for hina or onetonton, or None.

    The two corpora are loaded differently in `backend_cost` -- hina through `hina_episode`,
    onetonton through `ev.episode` plus `refined_starts` -- and their frozen DiariZen references
    sit in different trees. Everything downstream only wants those four things.
    """
    import backend_cost as bc

    if ident.startswith("onetonton-"):
        number = int(ident.split("-")[1])
        data = bc.ev.episode(number)
        cached = bc.V2_OUT / ident / "diarizen.npz"
        if data is None or not cached.exists():
            return None
        stable, _, grid, people, _ = data
        return stable, grid, bc.refined_starts(number), np.load(cached)["probs"], people

    loaded = bc.hina_episode(ident)
    reference = CACHE / ident / f"w16_default.npz"
    if loaded is None or not reference.exists():
        return None
    (stable, _, grid, people, _), _, starts = loaded
    return stable, grid, starts, np.load(reference)["probs"], people


def run_port(ident: str):
    import soundfile
    import torch

    import accuracy_lab as lab
    from finesub.speech.diarization import diarizen_assemble as assembler
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_embed as embedder
    from finesub.speech.diarization import diarizen_runner as runner

    signal, rate = soundfile.read(source_for(ident), dtype="float32", always_2d=True)
    assert rate == runner.SAMPLE_RATE
    waveform = torch.from_numpy(signal.mean(axis=1))[None]

    local = runner.segment(EXPORT_DIR / "diarizen_segmentation.ts", waveform).data
    filtered = runner.median_filtered(local)
    embeddings = embedder.embed(EXPORT_DIR / "diarizen_embedding.ts", waveform, filtered)
    hard = clusterer.cluster(embeddings, filtered, PLDA).hard
    native = assembler.assemble(filtered, hard)
    return to_lab_grid(native.astype(np.float32), assembler.FRAME_STEP)


def label_change(ours, theirs, stable, grid, *, with_rows: bool = False):
    """Share of reference-covered subtitle rows whose leading speaker moves."""
    from scipy.optimize import linear_sum_assignment

    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    width = max(ours.shape[1], theirs.shape[1])

    def pad(data):
        if data.shape[1] >= width:
            return data
        return np.hstack([data, np.zeros((len(data), width - data.shape[1]), data.dtype)])

    a, b = pad(ours), pad(theirs)
    frames = min(len(a), len(b))
    a, b = a[:frames], b[:frames]
    table = (a > 0).T.astype(np.int64) @ (b > 0).astype(np.int64)
    rows, columns = linear_sum_assignment(-table)
    aligned = np.zeros_like(a)
    for source, target in zip(rows, columns):
        aligned[:, target] = np.maximum(aligned[:, target], a[:, source])

    first = [s.track for s in assign_all(stable, segments_from_activity(aligned.tolist()))]
    second = [s.track for s in assign_all(stable, segments_from_activity(b.tolist()))]
    changed, total = [], 0
    for index, (row, one, two) in enumerate(zip(stable, first, second)):
        low = int(row["start"] / LAB_FRAME)
        high = max(low + 1, int(np.ceil(row["end"] / LAB_FRAME)))
        if not grid[low:high].any():
            continue
        total += 1
        if one != two:
            changed.append(index)
    share = 100.0 * len(changed) / max(total, 1)
    # the row identities, not just the count: three noise draws on hina-27 all read 0.369%, and a
    # rate alone cannot tell "the same knife-edge rows flip every time" from "different rows,
    # same count" -- only the row SET can.
    return (share, changed) if with_rows else share


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="*", default=["hina-32"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    import backend_cost as bc

    rows = []
    for ident in args.ids:
        loaded = episode_for(ident)
        if loaded is None:
            print(f"{ident}: no refined axis or no reference, skipped")
            continue
        stable, grid, _, reference, _ = loaded
        ours = run_port(ident)
        moved = label_change(ours, reference, stable, grid)
        rows.append({"ident": ident, "label_change_pct": moved,
                     "speakers_ours": int(ours.shape[1]),
                     "speakers_reference": int(reference.shape[1])})
        print(f"{ident}: subtitle speaker labels differ on {moved:.3f}%  "
              f"(speakers {ours.shape[1]} vs {reference.shape[1]})", flush=True)

    if rows:
        values = [row["label_change_pct"] for row in rows]
        print(f"\nport vs research path, {len(rows)} files: "
              f"median {np.median(values):.3f}%  max {max(values):.3f}%")
        print("  DiariZen's own band, for reference: sigma=1e-7 1.36%, PCM16 1.14%, "
              "one-sample shift 2.84% (all n=1, probe 7.5)")
    if args.write and rows:
        (EXPORT_DIR / "port-results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
