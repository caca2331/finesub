"""Is centroid distance weak because the evidence is weak? Re-embed the tracks and compare.

Both Q11 arms failed at the same question -- "are these two tracks one person" -- and the second
one failed *with* the centroids in hand: ungated, 49.1% of its suggestions would merge two
different people. The hypothesis this tests is that the centroid is the problem rather than the
idea: it is a gamma-weighted average of 16 s chunk embeddings, which is a much thinner thing than
an embedding taken over all of a track's exclusive speech.

So: for every track, concatenate the audio where it sounds ALONE, embed that once, and compare
tracks in that space. Same model, same PLDA transform, a far longer observation.

⚠ The comparison is against the same truth and the same pairs, so the two arms differ in exactly
one thing. Anything else would make the result unreadable.

⚠⚠ **This bed cannot answer its own question, because its truth is not about people**
(2026-09-20). It reported centroid 0.798 vs 0.845 and re-embedded 0.815 vs 0.838 -- the two
populations lying on top of each other -- and that was read as "merging by acoustic similarity is
closed off". **Withdraw that.** hina's cast list holds role placeholders and game characters, so
its "same person" pairs are partly two different humans wearing one label; breaking the population
down by label leaves almost nothing but replayed game audio.

The question was re-asked on 姐姐咚咚, three named humans across fifteen episodes
(`onetonton_identity_bed.py`), and the answer there is the opposite:

    same person, two tracks   n=  4  median 0.330  max 0.354
    two different people      n= 53  median 1.011  MIN 0.456     -> a gap of +0.102

⚠ Self-selection is REAL and it is measurable -- the same person's tracks where the clustering did
NOT split them sit at 0.050, so a split pair really is ~6.6x farther apart. It just does not reach
the different-person population. ⇒ The mechanism was right and the conclusion drawn from it was
wrong: the space can still tell the two cases apart.

⚠ n=4. That is four pairs from four episodes, all consistent, and it is not a calibration. What
would settle it is material that over-splits often AND names people rather than parts.

    python tools/diarizen_export/track_identity_bed.py --ids hina-20 hina-23
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

from eval_contract import require_people_truth  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = REPO / "tmp" / "diarization-dz3"
EXPORT_DIR = REPO / "tmp" / "diarizen-export"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"

#: A track with less exclusive speech than this cannot be embedded meaningfully.
MIN_SECONDS = 8.0

#: How much of a track's exclusive audio to use. Longer is better evidence but slower; this is
#: already several times what a single 16 s chunk gives the centroid.
MAX_SECONDS = 60.0


def track_audio(mono, rate, native, step, track, *, max_seconds=MAX_SECONDS):
    """The audio where `track` sounds and nothing else does, concatenated."""
    alone = (native[:, track] > 0) & (native.sum(axis=1) == 1)
    if not alone.any():
        return None
    edges = np.flatnonzero(np.diff(np.concatenate(([0], alone.view(np.int8), [0]))))
    pieces, total = [], 0.0
    for begin, stop in zip(edges[0::2], edges[1::2]):
        start, end = begin * step, stop * step
        if end - start < 0.4:
            continue
        pieces.append(mono[int(start * rate):int(end * rate)])
        total += end - start
        if total >= max_seconds:
            break
    if total < MIN_SECONDS:
        return None
    return np.concatenate(pieces)


def embed_audio(model, waveform, device="cuda"):
    """One embedding for a whole waveform, through the exported WeSpeaker resnet."""
    import torch

    from finesub.speech.diarization.diarizen_features import compute_fbank
    from finesub.speech.diarization.diarizen_runtime import jit_fusion_disabled, tf32_disabled

    with jit_fusion_disabled(), tf32_disabled(), torch.inference_mode():
        # ⚠ `(batch, channel, samples)`: `compute_fbank` vmaps over the batch and hands each
        # element to kaldi.fbank, which wants a channel axis. Production passes `(1, window)`
        # crops, so one long span is a batch of one.
        audio = torch.from_numpy(waveform)[None, None].to(device)
        fbank = compute_fbank(audio)
        weights = torch.ones((1, fbank.shape[1]), device=device)
        return model(fbank, weights).cpu().numpy()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", default=["hina-20", "hina-23", "hina-25", "hina-27"])
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-bed" / "track-identity.json"))
    args = parser.parse_args()

    # ⚠ This bed produced "merging by acoustic similarity is closed off", which was withdrawn
    # for exactly the reason the gate states.
    require_people_truth(
        "hina" if str(args.ids[0]).startswith("hina") else "onetonton",
        "are two tracks of the same person acoustically closer than two people",
        roles_ok=("the hina reading is kept as the contrast case: it is what this "
                  "measurement looks like when the labels are roles")
        if str(args.ids[0]).startswith("hina") else None)

    import soundfile
    import torch

    import backend_cost as bc
    from check_port import source_for
    from finesub.speech.diarization import diarizen_assemble as assembler
    from frame_grid import to_frame_grid
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner
    from finesub.speech.diarization import diarizen_runtime as runtime
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import FRAME_SECONDS, segments_from_activity

    xvec_tf, plda_tf, _ = clusterer.load_plda(PLDA)
    rows = []

    with runtime.loaded_model(EXPORT_DIR / "diarizen_embedding.ts", "cuda") as model:
        for ident in args.ids:
            local_path = CACHE / ident / "w16" / "diarizen-local.npz"
            embed_path = CACHE / ident / "w16" / "embeddings.npy"
            loaded = bc.hina_episode(ident)
            if loaded is None or not (local_path.exists() and embed_path.exists()):
                print(f"{ident}: cache incomplete, skipped")
                continue
            (stable, _, grid, people, _), _, _ = loaded
            filtered = runner.median_filtered(np.load(local_path)["data"])
            base = clusterer.cluster(np.load(embed_path), filtered, PLDA)
            native = assembler.assemble(filtered, base.hard).astype(np.float32)
            probs = to_frame_grid(native, target_step=0.08)

            # Who each track really is, by the refined axis.
            tracks = segments_from_activity(probs.tolist())
            owners: dict[int, Counter] = {}
            for row, speaker in zip(stable, assign_all(stable, tracks)):
                cell = speaker.to_dict()
                held = cell.get("tracks") or ([cell["track"]] if cell.get("track") is not None
                                              else [])
                if not held:
                    continue
                low = int(float(row["start"]) / FRAME_SECONDS)
                high = max(low + 1, int(np.ceil(float(row["end"]) / FRAME_SECONDS)))
                span = grid[low:high]
                counts = {people[p]: int(span[:, p].sum()) for p in range(span.shape[1])
                          if span[:, p].any()}
                if counts:
                    owners.setdefault(int(held[0]), Counter())[max(counts, key=counts.get)] += \
                        max(0.0, float(row["end"]) - float(row["start"]))
            owner = {t: c.most_common(1)[0][0] for t, c in owners.items()}

            signal, rate = soundfile.read(source_for(ident), dtype="float32", always_2d=True)
            mono = signal.mean(axis=1)
            fresh = {}
            for track in sorted(owner):
                audio = track_audio(mono, rate, native, 0.02, track)
                if audio is None:
                    continue
                fresh[track] = embed_audio(model, audio)
            if len(fresh) < 2:
                print(f"{ident}: fewer than two embeddable tracks, skipped")
                continue

            mapping = np.asarray(base.centroid_of_track)
            keys = sorted(fresh)
            transformed = plda_tf(xvec_tf(np.stack([fresh[k] for k in keys])))
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    same = owner[a] == owner[b]
                    j = keys.index(b)
                    re_d = float(1 - np.dot(transformed[i], transformed[j]) /
                                 (np.linalg.norm(transformed[i]) * np.linalg.norm(transformed[j])))
                    if mapping[a] >= 0 and mapping[b] >= 0:
                        ca, cb = base.centroids[mapping[a]], base.centroids[mapping[b]]
                        cen_d = float(1 - np.dot(ca, cb) /
                                      (np.linalg.norm(ca) * np.linalg.norm(cb)))
                    else:
                        cen_d = None
                    rows.append({"ident": ident, "a": a, "b": b, "same": same,
                                 "centroid": cen_d, "reembedded": re_d})
            print(f"{ident}: {len(keys)} tracks embeddable, "
                  f"{sum(1 for r in rows if r['ident'] == ident)} pairs", flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")

    if not rows:
        return
    print(f"\n{len(rows)} pairs, {sum(r['same'] for r in rows)} of them the same person\n")
    for field in ("centroid", "reembedded"):
        same = [r[field] for r in rows if r["same"] and r[field] is not None]
        diff = [r[field] for r in rows if not r["same"] and r[field] is not None]
        if not same or not diff:
            continue
        print(f"{field:>11}: same-person median {np.median(same):.3f}  "
              f"different median {np.median(diff):.3f}  "
              f"separable? max(same)={max(same):.3f} vs min(diff)={min(diff):.3f} -> "
              f"{'YES' if max(same) < min(diff) else 'overlapping'}")


if __name__ == "__main__":
    main()
