"""Does a track change owner mid-file, and can the clustering's OWN state say so?

Owner 2026-09-20: "轨中途换主人 就没有什么置信度之类的信号吗？不需要很多换主人的正例，
而是通过算法逻辑不就能推导出来了？"

That is the right question and the answer is yes. `cluster()` already computes, for every
(chunk, local speaker), the distance to EVERY global centroid -- `soft` -- and then throws the
ranking away, keeping only the Hungarian assignment in `hard`. The two disagree exactly where the
per-chunk constraint ("one cluster per local speaker") overrode the acoustics. So:

    for each chunk assigned to track T, is T's own centroid actually the NEAREST one?

A track that is one person throughout should be nearest to its own centroid nearly everywhere,
and where it is not, that should be scattered noise. A track that changes owner should have a
**contiguous stretch** where some OTHER centroid is consistently nearest -- and, more sharply,
the other centroid that wins should DIFFER between the track's first and second half.

⚠ **No positive examples are needed to build this**, which is the owner's point. They are needed
only to check it, and they come free: onetonton has per-line speaker truth, so "this track's
majority person in its first half differs from its second half" is derivable for all 15 episodes
rather than hand-annotated.

⚠ What this does NOT do: repair anything. It is a hint -- the sidecar half the plan promised in
§0a and never built. Splitting a track on it would be a partition change, which the role layer is
not allowed to make.

    python tools/diarizen_export/track_drift_bed.py
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

EXPORT_DIR = REPO / "tmp" / "diarizen-export"
V2 = REPO / "tmp" / "diarization-accuracy-v2"

#: Chunk geometry, so a chunk index becomes a time. Its centre, because a 16 s window labelled by
#: its start would put every reading eight seconds early.
CHUNK_STEP = 1.6
CHUNK_DURATION = 16.0

#: A track with fewer lines than this has no halves worth comparing.
MIN_LINES = 10

#: And fewer chunks than this is not a time series.
MIN_CHUNKS = 20


#: How many consecutive blocks a track's lines are cut into when counting owner changes.
BLOCKS = 6


def truth_impurity(times_people) -> dict:
    """What the reference says about one track: how much of it is NOT its main person.

    ⚠⚠ **This replaced "did the owner change", and the reason matters.** The first version split
    a track in half and compared the majority person -- the same shape the plan's own 换主人 count
    uses. On 51 tracks it found ZERO positives, because the plan itself records that DiariZen's
    single case was a **swing and not a handover** (§0a), and a swing A->B->A has the same
    majority in both halves. A validation with no positives validates nothing.

    Impurity has none of those problems: it is continuous, it is defined for every track, it
    needs no positive examples at all, and it is the quantity the owner actually named as the
    expensive failure -- one person's lines landing in another person's track. `changes` keeps a
    count of majority flips across consecutive blocks, which does see swings.
    """
    ordered = sorted(times_people)
    people = Counter(person for _, person in ordered)
    main, main_count = people.most_common(1)[0]

    size = max(1, len(ordered) // BLOCKS)
    majorities = []
    for start in range(0, len(ordered), size):
        block = Counter(person for _, person in ordered[start:start + size])
        if block:
            majorities.append(block.most_common(1)[0][0])
    changes = sum(1 for a, b in zip(majorities, majorities[1:]) if a != b)

    return {"main": main, "impurity": 1.0 - main_count / len(ordered),
            "changes": changes, "distinct": len(people)}


def drift_signal(soft, hard, track: int, column: int) -> dict | None:
    """What the clustering's own ranking says about one track, over time.

    `foreign_rate` -- how often some other centroid was nearer than this track's own.
    `halves_differ` -- and whether the OTHER centroid that wins differs between the first and
    second half of the track, which is the shape an owner change makes and scattered noise
    does not.
    """
    chunks, speakers = np.where(hard == track)
    if len(chunks) < MIN_CHUNKS:
        return None
    nearest = np.argmax(soft[chunks, speakers, :], axis=1)
    times = chunks * CHUNK_STEP + 0.5 * CHUNK_DURATION
    foreign = nearest != column

    order = np.argsort(times)
    nearest, foreign = nearest[order], foreign[order]
    half = len(nearest) // 2
    winners = []
    for piece in (slice(0, half), slice(half, None)):
        others = Counter(int(value) for value, is_foreign
                         in zip(nearest[piece], foreign[piece]) if is_foreign)
        winners.append(others.most_common(1)[0][0] if others else None)

    return {
        "chunks": int(len(nearest)),
        "foreign_rate": float(foreign.mean()),
        "first_half_foreign": winners[0],
        "second_half_foreign": winners[1],
        # ⚠ Both halves must actually HAVE a foreign winner: "one half has none" is a track that
        # is simply clean for a while, not a track that changed hands.
        "halves_differ": bool(winners[0] is not None and winners[1] is not None
                              and winners[0] != winners[1]),
    }


def episode_rows(key: int) -> list[dict]:
    import backend_cost as bc
    import relabel_eval as ev
    from finesub.speech.diarization import diarizen_assemble as assembler
    from frame_grid import to_frame_grid
    from finesub.speech.diarization import diarizen_cluster as clusterer
    from finesub.speech.diarization import diarizen_runner as runner
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    local = V2 / f"onetonton-{key:02d}" / "diarizen-local.npz"
    embeds = V2 / f"onetonton-{key:02d}" / "embeddings.npy"
    data = bc.ev.episode(key)
    if data is None or not (local.exists() and embeds.exists()):
        return []
    stable, _, grid, people, _ = data

    filtered = runner.median_filtered(np.load(local)["data"])
    result = clusterer.cluster(np.load(embeds), filtered, EXPORT_DIR / "plda")
    native = assembler.assemble(filtered, result.hard).astype(np.float32)
    probs = to_frame_grid(native, target_step=0.08)[: len(grid)]

    tracks = list(segments_from_activity(probs.tolist()))
    lines: dict[int, list] = {}
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        if speaker.track is None:
            continue
        cover, _ = ev.span_cover({**row, "speaker": speaker.to_dict()}, grid)
        if not cover.any():
            continue
        lines.setdefault(int(speaker.track), []).append(
            (float(row["start"]), people[int(cover.argmax())]))

    mapping = np.asarray(result.centroid_of_track)
    rows = []
    for track, times_people in sorted(lines.items()):
        if len(times_people) < MIN_LINES or track >= len(mapping) or mapping[track] < 0:
            continue
        signal = drift_signal(result.soft, result.hard, track, int(mapping[track]))
        if signal is None:
            continue
        rows.append({"episode": key, "track": track, "lines": len(times_people),
                     **truth_impurity(times_people), **signal})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", nargs="+", type=int, default=list(range(1, 16)))
    parser.add_argument("--out", default=str(REPO / "tmp" / "diarizen-bed" / "track-drift.json"))
    args = parser.parse_args()

    rows = []
    for key in args.ids:
        found = episode_rows(key)
        rows.extend(found)
        for row in found:
            print(f"  ep{row['episode']:02d} s{row['track']} lines={row['lines']:3d} "
                  f"impurity={row['impurity']:.3f} ({row['distinct']}p, {row['changes']} flips)"
                  f"  ->  foreign={row['foreign_rate']:.3f} "
                  f"halves_differ={row['halves_differ']}")

    if rows:
        impurity = np.array([r["impurity"] for r in rows])
        foreign = np.array([r["foreign_rate"] for r in rows])
        print()
        print(f"tracks scored: {len(rows)}   "
              f"impurity median {np.median(impurity):.3f} [{impurity.min():.3f}, {impurity.max():.3f}]")
        if impurity.std() > 0 and foreign.std() > 0:
            r = float(np.corrcoef(foreign, impurity)[0, 1])
            print(f"  Pearson r(foreign_rate, impurity) = {r:+.3f}")
        # ⚠ The question a hint has to answer is not "do they correlate" but "if I flag the worst
        # ones, am I pointing at the mixed tracks" -- so report it as a lookup, not a coefficient.
        order = np.argsort(-foreign)
        for top in (5, 10):
            if len(rows) < top:
                continue
            flagged = impurity[order[:top]]
            rest = impurity[order[top:]]
            print(f"  top {top:2d} by foreign_rate: impurity median {np.median(flagged):.3f} "
                  f"vs {np.median(rest):.3f} for the rest")
        worst = sorted(rows, key=lambda r: -r["impurity"])[:5]
        print("  the five most mixed tracks, and what the signal said about them:")
        for r in worst:
            rank = int(np.where(order == rows.index(r))[0][0]) + 1
            print(f"    ep{r['episode']:02d} s{r['track']} impurity {r['impurity']:.3f} "
                  f"({r['distinct']} people, {r['changes']} block flips) -> "
                  f"foreign {r['foreign_rate']:.3f}, rank {rank}/{len(rows)}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
