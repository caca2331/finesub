"""Q12's potential, estimated rather than built: does a centroid identify a person ACROSS episodes?

The owner asked for data, not an implementation. The question is one number: on a series, is the
distance between two episodes' centroids for the SAME person reliably smaller than for different
people? If yes, naming becomes a lookup -- name once, match by cosine forever, no quota. If no,
Q12 is not worth building.

⚠ This is a different pairing from the within-episode one that just failed, and the difference
matters. There, "same person on two tracks" meant the clustering had separated them because they
sounded different -- self-selection that guaranteed the distance was large. Here the two centroids
come from different recordings of the same person with no such filter, so the failure mode does
not carry over and the question is genuinely open.

⚠ CPU only: the centroids come from the frozen embeddings.

⚠⚠ **On hina this bed cannot answer the question, and its headline number is a trap**
(2026-09-20). It reports same-person median 0.134 against different-person 0.869 -- 6.5x apart,
3.5% leakage -- which reads as "build it". Broken down by who:

    派蒙 (game character)   n=26  median 0.145
    钟离 (game character)   n=15  median 0.140
    男主 (role placeholder) n= 3  median 0.054
    温迪 (game character)   n= 1  median 0.043
    嘉宾 (an actual human)  n= 2  median 1.002

**87% of the same-person pairs are game characters**, whose "voice" is the same recorded clips
replayed every episode -- matching them across episodes is matching a file to itself. The only
human pairs read 1.002, farther apart than the different-person median.

⚠ And these labels are ROLES, not identities: one episode's 嘉宾 ("guest") need not be the next
one's, so the same-person set is contaminated in both directions. The breakdown prints on every
run so the headline cannot be read alone again.

⇒ **Q12 was re-asked on material whose truth names people** (`onetonton_identity_bed.py`, 姐姐咚咚,
three named humans over fifteen episodes) and there it separates cleanly:

    same person, different episodes   n=374  median 0.050  max 0.395
    different people                  n=745  median 1.012  MIN 0.417

Zero different-person pairs pass a cut that keeps 90% of same-person ones. ⚠ **One series, one
recording setup** -- channel and microphone are constant across those episodes, so some of that
separation may be the room rather than the voice. Naming-by-lookup is worth prototyping on this
evidence; it is not yet proven to survive a change of recording.

    python tools/diarizen_export/cross_episode_bed.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = REPO / "tmp" / "diarization-dz3"
PLDA = REPO / "tmp" / "diarization-models" / "diarizen-v2" / "plda"

#: Only tracks that hold real speech: a centroid built from a few seconds is not an identity.
MIN_SHARE = 0.05


def episode_centroids(ident):
    """`[(person, centroid)]` for one episode, tracks below MIN_SHARE dropped."""
    import backend_cost as bc

    from finesub.speech.diarization import diarizen_assemble as assembler
    from frame_grid import to_frame_grid
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import FRAME_SECONDS, segments_from_activity

    local_path = CACHE / ident / "w16" / "diarizen-local.npz"
    embed_path = CACHE / ident / "w16" / "embeddings.npy"
    loaded = bc.hina_episode(ident)
    if loaded is None or not (local_path.exists() and embed_path.exists()):
        return []
    (stable, _, grid, people, _), _, _ = loaded
    filtered = runner.median_filtered(np.load(local_path)["data"])
    base = clusterer.cluster(np.load(embed_path), filtered, PLDA)
    native = assembler.assemble(filtered, base.hard).astype(np.float32)
    probs = to_frame_grid(native, target_step=0.08)

    tracks = segments_from_activity(probs.tolist())
    seconds: Counter = Counter()
    owners: dict[int, Counter] = {}
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        cell = speaker.to_dict()
        held = cell.get("tracks") or ([cell["track"]] if cell.get("track") is not None else [])
        if not held:
            continue
        low = int(float(row["start"]) / FRAME_SECONDS)
        high = max(low + 1, int(np.ceil(float(row["end"]) / FRAME_SECONDS)))
        span = grid[low:high]
        counts = {people[p]: int(span[:, p].sum()) for p in range(span.shape[1]) if span[:, p].any()}
        if not counts:
            continue
        duration = max(0.0, float(row["end"]) - float(row["start"]))
        seconds[int(held[0])] += duration
        owners.setdefault(int(held[0]), Counter())[max(counts, key=counts.get)] += duration

    total = sum(seconds.values()) or 1.0
    mapping = np.asarray(base.centroid_of_track)
    found = []
    for track, counts in owners.items():
        if seconds[track] / total < MIN_SHARE or track >= len(mapping) or mapping[track] < 0:
            continue
        found.append((counts.most_common(1)[0][0], base.centroids[mapping[track]]))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+",
                        default=[f"hina-{n:02d}" for n in range(10, 35)])
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-bed" / "cross-episode.json"))
    args = parser.parse_args()

    # ⚠ The gate that would have caught this bed's first headline. Run on hina it read
    # 0.134 against 0.869 and 3.5% leakage, which looked like "worth building" -- until the
    # same-person population turned out to be 87% replayed game audio. The reading is still
    # worth having; calling it identity is what was wrong.
    require_people_truth(
        "hina" if str(args.ids[0]).startswith("hina") else "onetonton",
        "does a centroid identify a person across episodes",
        roles_ok=("characterising what this measurement looks like on role labels -- the "
                  "breakdown below prints who is in each class, which is the point")
        if str(args.ids[0]).startswith("hina") else None)

    from finesub.speech.diarization import diarizen_cluster as clusterer

    xvec_tf, plda_tf, _ = clusterer.load_plda(PLDA)
    per_episode = {}
    for ident in args.ids:
        found = episode_centroids(ident)
        if found:
            per_episode[ident] = found
            print(f"{ident}: {len(found)} tracks -> "
                  f"{', '.join(sorted({p for p, _ in found}))}", flush=True)

    if len(per_episode) < 2:
        print("fewer than two usable episodes")
        return

    same, different, by_person = [], [], {}
    for a, b in combinations(sorted(per_episode), 2):
        for (person_a, centroid_a) in per_episode[a]:
            for (person_b, centroid_b) in per_episode[b]:
                pair = plda_tf(xvec_tf(np.stack([centroid_a, centroid_b])))
                distance = float(1 - np.dot(pair[0], pair[1]) /
                                 (np.linalg.norm(pair[0]) * np.linalg.norm(pair[1])))
                if person_a == person_b:
                    same.append(distance)
                    by_person.setdefault(person_a, []).append(distance)
                else:
                    different.append(distance)

    print(f"\ncross-episode pairs: {len(same)} same person, {len(different)} different")
    if not same or not different:
        print("⚠ no shared cast across these episodes -- the question cannot be asked here")
        return
    same_a, diff_a = np.array(same), np.array(different)
    print(f"  same person:  median {np.median(same_a):.3f}  "
          f"[{np.percentile(same_a, 10):.3f}, {np.percentile(same_a, 90):.3f}]")
    print(f"  different:    median {np.median(diff_a):.3f}  "
          f"[{np.percentile(diff_a, 10):.3f}, {np.percentile(diff_a, 90):.3f}]")

    # ⚠ The number that decides whether a lookup is possible: at the threshold that keeps 90% of
    # same-person pairs, how many different-person pairs come with them?
    cut = float(np.percentile(same_a, 90))
    leak = float((diff_a <= cut).mean())
    print(f"\n  at a threshold keeping 90% of same-person pairs ({cut:.3f}), "
          f"{leak:.1%} of different-person pairs also pass")
    # ⚠ Never read the headline without this. On hina, 87% of the same-person pairs turned out to
    # be game characters, whose "voice" is the same clips replayed every episode -- matching those
    # across episodes is matching a file to itself, and says nothing about recognising a person.
    print("\n  same-person pairs by who -- ⚠ read the headline only after this:")
    for who, values in sorted(by_person.items(), key=lambda kv: -len(kv[1])):
        print(f"    {who:>8}: n={len(values):3d}  median {np.median(values):.3f}")
    print("  ⇒ the separation above is about PEOPLE only if the rows above are people. Recurring")
    print("    game audio, or a role placeholder that different humans share, makes it empty.")

    print("  ⇒ " + ("the two populations separate -- BUT see the breakdown above first"
                    if leak < 0.1 else
                    "NOT worth building on this evidence: the populations overlap too much"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"same": same, "different": different, "threshold": cut, "leak": leak}, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
