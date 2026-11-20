"""The acoustic identity questions, re-asked on truth that names PEOPLE.

Three readings from this round were taken on hina, and all three lean on one thing being true:
that two tracks the truth gives the same label really are one human. ⚠ On hina that is not safe.
`docs/diarization-findings.md` 11.3 already records that its cast list holds role placeholders --
`男主`, `嘉宾`, `派蒙` -- so one label can span several voices, and a different episode's `嘉宾` is
a different person entirely. Every "same person" population built from it is contaminated, and
each of the three conclusions was in the direction contamination pushes:

    Q11 arm 2   "49.1% of merge suggestions would join two different people"
    identity    "same-person centroid distance 0.798 vs different-person 0.845 -- inseparable"
    Q12         "cross-episode matching works" (87% of the pairs were replayed game audio)

姐姐咚咚 (`onetonton`) has none of that: fifteen episodes, the same three named humans throughout,
line-level truth. The local segmentation is already cached from the accuracy work, so the only new
compute is one embedding pass per episode, cached beside it.

⚠ This bed does not re-ask whether the CLUSTERING is good -- that is scored elsewhere. It asks the
narrow question all three readings actually depend on: **in this embedding space, is same-person
closer than different-person?**

    python tools/diarizen_export/onetonton_identity_bed.py            # all 15
    python tools/diarizen_export/onetonton_identity_bed.py --ids 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))

from eval_contract import require_people_truth  # noqa: E402

V2 = REPO / "tmp" / "diarization-accuracy-v2"
EXPORT_DIR = REPO / "tmp" / "diarizen-export"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"

#: A track below this share of the episode's labelled speech is not an identity to reason about.
MIN_SHARE = 0.05

#: For the re-embedding arm: how much exclusive audio to use, and the least that is worth using.
MAX_SECONDS = 60.0
MIN_SECONDS = 8.0


def audio_path(key: int) -> Path:
    return REPO / "out" / f"onetonton-{key:02d}" / f"onetonton-{key:02d}-vocal.ogg"


def load_waveform(path: Path):
    """The vocal track as the model wants it: 16 kHz mono, `(1, samples)`."""
    import soundfile
    import torch

    from finesub.speech.preprocessing.audio import TARGET_SR, resample_if_needed

    signal, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(signal.mean(axis=1)).unsqueeze(0)
    waveform, _ = resample_if_needed(waveform, rate, TARGET_SR)
    return waveform


def embeddings_for(key: int, filtered, *, device: str):
    """Per-(chunk, speaker) embeddings, computed once and cached beside the segmentation."""
    from finesub.speech.diarization import diarizen_embed

    cache = V2 / f"onetonton-{key:02d}" / "embeddings.npy"
    if cache.exists():
        return np.load(cache)
    waveform = load_waveform(audio_path(key))
    found = diarizen_embed.embed(EXPORT_DIR / "diarizen_embedding.ts", waveform, filtered,
                                 device=device)
    np.save(cache, found)
    return found


def track_owners(probs, stable, grid, people):
    """`({track: person}, {track: seconds})` by the refined axis -- who each track really is."""
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import FRAME_SECONDS, segments_from_activity

    tracks = segments_from_activity(probs.tolist())
    owners: dict[int, Counter] = {}
    seconds: Counter = Counter()
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        cell = speaker.to_dict()
        held = cell.get("tracks") or ([cell["track"]] if cell.get("track") is not None else [])
        if not held:
            continue
        low = int(float(row["start"]) / FRAME_SECONDS)
        high = max(low + 1, int(np.ceil(float(row["end"]) / FRAME_SECONDS)))
        span = grid[low:high]
        counts = {people[p]: int(span[:, p].sum())
                  for p in range(span.shape[1]) if span[:, p].any()}
        if not counts:
            continue
        duration = max(0.0, float(row["end"]) - float(row["start"]))
        seconds[int(held[0])] += duration
        owners.setdefault(int(held[0]), Counter())[max(counts, key=counts.get)] += duration
    return {t: c.most_common(1)[0][0] for t, c in owners.items()}, seconds


def exclusive_audio(mono, rate, native, step, track, *, max_seconds=MAX_SECONDS):
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
    return np.concatenate(pieces) if total >= MIN_SECONDS else None


def embed_span(model, waveform, device="cuda"):
    """One embedding for a whole waveform, through the same exported WeSpeaker resnet."""
    import torch

    from finesub.speech.diarization.diarizen_features import compute_fbank
    from finesub.speech.diarization.diarizen_runtime import jit_fusion_disabled, tf32_disabled

    with jit_fusion_disabled(), tf32_disabled(), torch.inference_mode():
        # `(batch, channel, samples)`: compute_fbank vmaps over the batch.
        audio = torch.from_numpy(waveform)[None, None].to(device)
        fbank = compute_fbank(audio)
        weights = torch.ones((1, fbank.shape[1]), device=device)
        return model(fbank, weights).cpu().numpy()[0]


def cosine(a, b) -> float:
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def report(name, same, different) -> None:
    """The one table every arm here prints: do the two populations come apart?"""
    if not same or not different:
        print(f"{name:>14}: not enough pairs ({len(same)} same, {len(different)} different)")
        return
    same_a, diff_a = np.array(same), np.array(different)
    overlapping = max(same_a) >= min(diff_a)
    print(f"{name:>14}: same {np.median(same_a):.3f} "
          f"[{np.percentile(same_a, 10):.3f}, {np.percentile(same_a, 90):.3f}]  "
          f"different {np.median(diff_a):.3f} "
          f"[{np.percentile(diff_a, 10):.3f}, {np.percentile(diff_a, 90):.3f}]  "
          f"n={len(same_a)}/{len(diff_a)}  -> "
          f"{'OVERLAPPING' if overlapping else 'SEPARABLE'}")
    # ⚠ The number that decides whether a threshold is usable, which a median never does.
    cut = float(np.percentile(same_a, 90))
    print(f"{'':>14}  at a cut keeping 90% of same-person pairs ({cut:.3f}): "
          f"{float((diff_a <= cut).mean()):.1%} of different-person pairs come with them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", type=int, default=list(range(1, 16)))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reembed", action="store_true",
                        help="also embed each track's exclusive audio (slower, needs the audio)")
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-bed" / "onetonton-identity.json"))
    args = parser.parse_args()

    # The whole reason this bed exists: 姐姐咚咚 truth names people, so the question is
    # answerable here and was not on hina. Asserted rather than assumed.
    require_people_truth("onetonton", "is same-person closer than different-person")

    import soundfile

    import relabel_eval as ev
    from finesub.speech.diarization import diarizen_assemble as assembler
    from frame_grid import to_frame_grid
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner
    from finesub.speech.diarization import diarizen_runtime as runtime

    xvec_tf, plda_tf, _ = clusterer.load_plda(PLDA)
    within_same, within_diff = [], []
    reembed_same, reembed_diff = [], []
    suggestions = Counter()
    by_distance = []
    per_person_centroids: dict[str, list] = {}
    rows = []

    for key in args.ids:
        local = V2 / f"onetonton-{key:02d}" / "diarizen-local.npz"
        data = ev.episode(key)
        if data is None or not local.exists():
            print(f"onetonton-{key:02d}: cache incomplete, skipped")
            continue
        stable, _, grid, people, _ = data
        filtered = runner.median_filtered(np.load(local)["data"])
        embeddings = embeddings_for(key, filtered, device=args.device)

        base = clusterer.cluster(embeddings, filtered, PLDA)
        native = assembler.assemble(filtered, base.hard).astype(np.float32)
        probs = to_frame_grid(native, target_step=0.08)
        owner, seconds = track_owners(probs, stable, grid, people)
        total = sum(seconds.values()) or 1.0
        mapping = np.asarray(base.centroid_of_track)
        # ⚠⚠ The MEAN, not the sum `cluster()` stores. `xvec_tf` starts by subtracting `mean1`,
        # and a translation is not scale-invariant -- so feeding the raw centroid puts how much
        # the track SPOKE into every distance below. Measured: the same direction at 100x the
        # mass reads a PLDA cosine distance of 0.227 from itself (2026-09-20 audit, finding 5).
        # Production's own cosine is unaffected and still uses `base.centroids`; this arm is the
        # one that cannot.
        means = clusterer.mean_centroids(base)

        # --- arm A: do centroids of the same person sit closer together? ------------------
        usable = [t for t in sorted(owner)
                  if seconds[t] / total >= MIN_SHARE and t < len(mapping) and mapping[t] >= 0]
        for a, b in combinations(usable, 2):
            pair = plda_tf(xvec_tf(np.stack([means[mapping[a]], means[mapping[b]]])))
            distance = cosine(pair[0], pair[1])
            (within_same if owner[a] == owner[b] else within_diff).append(distance)
        for track in usable:
            per_person_centroids.setdefault(owner[track], []).append(
                (key, means[mapping[track]]))

        # --- arm B: Q11's merge suggestions, scored against people rather than roles -------
        sizes = (native > 0).sum(axis=0)
        hints = clusterer.merge_suggestions(base.centroids, base.centroid_of_track, sizes,
                                            keep=len(people))
        for hint in hints:
            a, b = int(hint["track"]), int(hint["like"])
            if a not in owner or b not in owner:
                suggestions["unknown (a track that leads no line)"] += 1
                continue
            same = owner[a] == owner[b]
            suggestions["correct: same person" if same else "WRONG: two different people"] += 1
            by_distance.append((float(hint["distance"]), same))

        # --- arm C: is the centroid the weak part? re-embed the exclusive audio ------------
        if args.reembed and len(usable) >= 2:
            signal, rate = soundfile.read(audio_path(key), dtype="float32", always_2d=True)
            mono = signal.mean(axis=1)
            with runtime.loaded_model(EXPORT_DIR / "diarizen_embedding.ts", args.device) as model:
                fresh = {}
                for track in usable:
                    clip = exclusive_audio(mono, rate, native, 0.02, track)
                    if clip is not None:
                        fresh[track] = embed_span(model, clip, args.device)
            keys = sorted(fresh)
            if len(keys) >= 2:
                moved = plda_tf(xvec_tf(np.stack([fresh[k] for k in keys])))
                for i, a in enumerate(keys):
                    for j, b in list(enumerate(keys))[i + 1:]:
                        (reembed_same if owner[a] == owner[b] else reembed_diff).append(
                            cosine(moved[i], moved[j]))

        rows.append({"episode": key, "tracks": len(usable), "people": len(people),
                     "suggestions": len(hints),
                     "owners": {str(t): owner[t] for t in usable}})
        print(f"onetonton-{key:02d}: {len(usable)} usable tracks for {len(people)} people, "
              f"{len(hints)} suggestions", flush=True)

    if not rows:
        return

    print("\n" + "=" * 96)
    print("A. WITHIN an episode -- two tracks, same person or not? (PLDA cosine between centroids)")
    report("centroid", within_same, within_diff)
    if args.reembed:
        report("re-embedded", reembed_same, reembed_diff)
        print("   ⚠ If re-embedding does not separate either, the evidence was never the problem.")

    print("\nB. Q11 arm 2 -- are the merge suggestions right, judged against NAMED people?")
    total = sum(suggestions.values())
    for name, count in suggestions.most_common():
        print(f"   {name:>34}: {count:4d} = {count / max(total, 1):5.1%}")
    if by_distance:
        by_distance.sort()
        for cut in (0.2, 0.3, 0.4, 0.5, 1.0):
            kept = [(d, s) for d, s in by_distance if d <= cut]
            if kept:
                right = sum(1 for _, s in kept if s)
                print(f"   distance <= {cut:.1f}: {len(kept):3d} kept, "
                      f"{right / len(kept):5.1%} correct")

    print("\nC. Q12 -- ACROSS episodes, does a centroid identify a person? (same cast throughout)")
    cross_same, cross_diff = [], []
    everyone = [(person, key, centroid)
                for person, found in per_person_centroids.items() for key, centroid in found]
    for (person_a, key_a, centroid_a), (person_b, key_b, centroid_b) in combinations(everyone, 2):
        if key_a == key_b:
            continue  # within-episode pairs are arm A; this is the cross-episode question
        pair = plda_tf(xvec_tf(np.stack([centroid_a, centroid_b])))
        (cross_same if person_a == person_b else cross_diff).append(cosine(pair[0], pair[1]))
    report("cross-episode", cross_same, cross_diff)
    print("   ⚠ Every pair here is one of three named humans -- no replayed game audio, no role")
    print("     placeholder. This is the reading hina could not give.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "within_same": within_same, "within_different": within_diff,
         "reembed_same": reembed_same, "reembed_different": reembed_diff,
         "cross_same": cross_same, "cross_different": cross_diff,
         "suggestions": dict(suggestions), "by_distance": by_distance}, indent=1,
        ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
