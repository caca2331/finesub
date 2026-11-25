"""W1: the two rulers that map to what the owner actually asked for, beside `fixed` and B-cubed F1.

Plan section 1.3 asks for the fewest human edits and at most about three track-level decisions per
video. Three rounds of model comparison reported neither, so "DiariZen is the most accurate" was
established while "DiariZen is the cheapest to fix up" was never measured (audit 2.3).

Two things live here, both pure functions over artifacts already on disk:

`naming_cost` -- how many tracks a person has to make a decision about. Raw track count is the
wrong number: it counts tracks that never win a line and so never reach the `spk` column. Three
counts instead, cheapest first: tracks that ever LEAD a reference-covered row (what a proofreader
actually sees), tracks holding at least `floor_seconds` of speech, and how many tracks it takes to
cover `share` of all track activity.

`edit_cost` -- the owner's 2026-09-14/15 pricing (delete 1, move 2, a missed second speaker 2 on
strong evidence only, and 3 for the merge itself), reached through `refined_multi.strong_cost` so
that the number is the SAME ruler `relabel_eval.py --strong` uses. It is not reimplemented here;
this only assembles the arguments for an arbitrary timeline, which is what lets two backends be
priced against each other.

⚠ Both are per-file readings. Aggregation and intervals are the caller's job, and
`accuracy_report.bootstrap_mean_ci` refuses to produce an interval below five files.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import accuracy_lab as lab  # noqa: E402
from eval_contract import DEFAULT_POLICY  # noqa: E402


def track_seconds(probs, *, frame_seconds: float | None = None) -> dict[int, float]:
    """Seconds of activity per track, from a frame-by-track activity or probability map.

    `frame_seconds` is the grid `probs` is on (default: the lab's 80 ms), as everywhere below.
    """
    from finesub.speech.diarization.segments import segments_from_activity

    step = lab.FRAME if frame_seconds is None else frame_seconds
    out: dict[int, float] = {}
    for segment in segments_from_activity(probs.tolist(), frame_seconds=step):
        out[segment.track] = out.get(segment.track, 0.0) + (segment.end - segment.start)
    return out


def leading_tracks(probs, stable, grid, *, frame_seconds: float | None = None) -> set[int]:
    """Tracks that lead at least one subtitle row the reference covers."""
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    step = lab.FRAME if frame_seconds is None else frame_seconds
    tracks = segments_from_activity(probs.tolist(), frame_seconds=step)
    out = set()
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        lo = int(row["start"] / lab.FRAME)
        hi = max(lo + 1, int(np.ceil(row["end"] / lab.FRAME)))
        if grid[lo:hi].any() and speaker.track is not None:
            out.add(int(speaker.track))
    return out


def leading_counts(probs, stable, grid, thresholds=(1, 3, 10), *,
                   frame_seconds: float | None = None) -> dict[int, int]:
    """How many tracks lead at least N reference-covered rows, for several N.

    `leading` counts a track that wins ONE row the same as one that carries the episode, and a
    naming decision on a track with a single line has almost no evidence behind it. Without this
    sweep "8.63 tracks to name" cannot be told apart from "3 tracks to name plus 5 one-line
    accidents" -- and which of those it is decides whether the tail rule is even the right lever
    (fable 2026-09-19).
    """
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    counts: Counter = Counter()
    step = lab.FRAME if frame_seconds is None else frame_seconds
    tracks = segments_from_activity(probs.tolist(), frame_seconds=step)
    for row, speaker in zip(stable, assign_all(stable, tracks)):
        lo = int(row["start"] / lab.FRAME)
        hi = max(lo + 1, int(np.ceil(row["end"] / lab.FRAME)))
        if grid[lo:hi].any() and speaker.track is not None:
            counts[int(speaker.track)] += 1
    return {n: sum(1 for value in counts.values() if value >= n) for n in thresholds}


def naming_cost(probs, stable, grid, *, floor_seconds: float = 30.0, share: float = 0.95,
                frame_seconds: float | None = None) -> dict:
    """How many tracks somebody has to decide about, three ways (cheapest reading first).

    `leading` is the one that reaches the product: a track that never leads a row never appears in
    the `spk` column, so it costs no decision there however long it is. `covering` is the opposite
    end -- how many tracks hold `share` of the activity -- and is the number the role layer would
    have to name if it named everything.
    """
    seconds = track_seconds(probs, frame_seconds=frame_seconds)
    ordered = sorted(seconds.values(), reverse=True)
    total = sum(ordered)
    covering, running = 0, 0.0
    for value in ordered:
        if running >= share * total:
            break
        running += value
        covering += 1
    return {
        "tracks": len(seconds),
        "leading": len(leading_tracks(probs, stable, grid, frame_seconds=frame_seconds)),
        "over_floor": sum(1 for value in seconds.values() if value >= floor_seconds),
        "covering": covering,
    }


def owner_of_tracks(probs, grid, people, *, frame_seconds: float | None = None) -> dict[int, str]:
    """Each track's majority reference person -- the mapping `strong_cost` prices against.

    `frame_seconds` is the grid of `probs` (default: the lab's 80 ms); segments come back in
    seconds and index the 80 ms truth grid either way.

    Keys are `probs` column numbers, not raw track ids: see `backend_cost.timeline_probs`. Looking up raw ids silently
    reads the neighbouring track's owner; before a new script reports numbers, check it on a file whose track ids
    do not start at 0.
    """
    from finesub.speech.diarization.segments import segments_from_activity

    step = lab.FRAME if frame_seconds is None else frame_seconds
    counts: dict[int, Counter] = {}
    for segment in segments_from_activity(probs.tolist(), frame_seconds=step):
        lo = int(segment.start / lab.FRAME)
        hi = min(int(np.ceil(segment.end / lab.FRAME)), len(grid))
        if hi <= lo:
            continue
        span = grid[lo:hi].sum(axis=0)
        bucket = counts.setdefault(segment.track, Counter())
        for index, value in enumerate(span):
            bucket[people[index]] += int(value)
    return {track: bucket.most_common(1)[0][0] for track, bucket in counts.items() if sum(bucket.values())}


def cost_terms(labelled, grid, people, owner, starts, *, add: float = DEFAULT_POLICY.add,
               merge: float = DEFAULT_POLICY.merge, slack: float = 0.3,
               price_unlabelled: bool = DEFAULT_POLICY.price_unlabelled) -> dict:
    """`refined_multi.strong_cost` split into the owner's four priced actions.

    W5 asked "are the extra second labels worth it" with a minutes-level estimate, which cannot
    answer it: the reference axis under-annotates overlap, and that bias falls on the MORE sensitive
    detector (fable 2026-09-19). The edit cost already prices a spurious second label -- at 1, as a
    delete -- so the answer was inside W2b all along, just not broken out. This is that breakdown.

    ⚠ `strong_cost` holds the minimisation of record; this walks the same one and reports which
    term each unit came from. `test_cost_metrics.py` pins the two together on every episode, so a
    drift here cannot quietly restate a published total -- which means `price_unlabelled` has to
    be passed through to both, and it is.
    """
    import bisect

    import refined_multi as rm

    keys = [a for a, _ in starts]
    terms = {"move": 0.0, "delete": 0.0, "add": 0.0, "duplicate": 0.0, "merge": 0.0}
    for s in labelled:
        tr = rm.ev.tracks_of(s)
        # ⚠ Mirrors `strong_cost`'s flag exactly. Both now default to skipping (owner 2026-09-20:
        # an unlabelled row is an interjection nobody cares about, so it costs 0) -- which is why
        # `edit_cost` reports `abstained` beside every total: this ruler can be lowered by
        # labelling less, and only that count says whether it was.
        if not tr and not price_unlabelled:
            continue
        cover, _ = rm.ev.span_cover(s, grid)
        if not cover.any():
            continue
        human = people[int(cover.argmax())]
        lo, hi = bisect.bisect_left(keys, s["start"] - slack), bisect.bisect_left(keys, s["end"])
        target = {human} | {w for _, w in starts[lo:hi]}
        labels = [owner.get(t) for t in tr]
        extra, missing = len(set(labels) - target), len(target - set(labels))
        best = min(range(min(extra, missing) + 1),
                   key=lambda k: 2 * k + (extra - k) + add * (missing - k))
        terms["move"] += 2 * best
        terms["delete"] += extra - best
        terms["add"] += add * (missing - best)
        terms["duplicate"] += len(labels) - len(set(labels))
        terms["merge"] += merge * (len(target) - 1)
    terms["total"] = sum(terms.values())
    return terms


def edit_cost(probs, stable, grid, people, starts, *, add: float = DEFAULT_POLICY.add,
              merge: float = DEFAULT_POLICY.merge,
              price_unlabelled: bool = DEFAULT_POLICY.price_unlabelled,
              frame_seconds: float | None = None) -> dict:
    """The owner's pricing for one timeline, through `refined_multi.strong_cost`.

    `starts` is the episode's (start, person) list that decides strong evidence; build it the way
    `relabel_eval.strong` does. Returns the total, the merge component (a function of the segments,
    not the labels) and the label component, because only the last one moves between two backends
    scored on the SAME segments.

    ⚠ **`abstained` is reported and is not optional to look at.** It is how many rows the
    reference covers that this timeline left with no label at all. With `price_unlabelled` those
    rows cost what the reference says is missing from them; with it off they cost nothing, which
    is the shape that let a backend buy a lower total by labelling less (2026-09-20 audit). Either
    way the count has to travel with the total, because "cheap" and "silent" are answers to
    different questions and this number is the only thing that tells them apart.
    """
    import refined_multi as rm
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    step = lab.FRAME if frame_seconds is None else frame_seconds
    tracks = segments_from_activity(probs.tolist(), frame_seconds=step)
    labelled = [
        {**row, "speaker": speaker.to_dict()} for row, speaker in zip(stable, assign_all(stable, tracks))
    ]
    owner = owner_of_tracks(probs, grid, people, frame_seconds=step)
    total, merges = rm.strong_cost(labelled, grid, people, owner, starts, add, merge=merge,
                                   price_unlabelled=price_unlabelled)
    abstained = sum(1 for row in labelled
                    if not rm.ev.tracks_of(row) and rm.ev.span_cover(row, grid)[0].any())
    return {"total": float(total), "merges": int(merges), "labels": float(total - merge * merges),
            "abstained": int(abstained)}


def labelled_segments(probs, stable, *, top_one: bool = False):
    """Production assignment for one timeline, optionally stripped to the leading label only.

    The `top_one` arm is the ablation W5 needed and did not have. Comparing DiariZen's total edit
    cost against Sortformer's answers "which backend is cheaper", not "are the extra second labels
    worth it": 80% of that gap is the `move` term, which is the LEADING track being the right
    person and has nothing to do with second labels (fable 2026-09-19). Holding the backend fixed
    and removing only the extra labels is the comparison that does answer it.
    """
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import segments_from_activity

    rows = []
    for row, speaker in zip(stable, assign_all(stable, segments_from_activity(probs.tolist()))):
        payload = speaker.to_dict()
        if top_one and payload.get("tracks"):
            # ⚠ TOP level. `SegmentSpeaker.to_dict` flattens `extras` into the payload
            # (`payload.update(self.extras)`), and `relabel_eval.tracks_of` reads `speaker["tracks"]`
            # there -- writing a nested `extras` dict instead is a silent no-op, which is how the
            # first run of this ablation produced a table of exact zeros.
            payload = {**payload, "tracks": payload["tracks"][:1]}
        rows.append({**row, "speaker": payload})
    return rows


def owner_of_tracks_folded(probs, grid, people, *, fold: int, folds: int = 2,
                           seconds: float = DEFAULT_POLICY.fold_seconds):
    """`owner_of_tracks`, but fitted on the OTHER folds of a time-interleaved split.

    `owner_of_tracks` picks each track's majority person on the very rows it is then scored on, so
    a backend that emits more tracks gets more chances to find a flattering mapping. `fixed`
    already carries `crossfit` for exactly this; the edit cost never had it, and W2/W2b quoted a
    -300 that contains some unknown amount of in-sample fitting (fable 2026-09-19).

    Interleaving by minute rather than splitting the file in half: speakers come and go over a
    recording, so a front/back split would hand the two halves different populations and measure
    that instead of the fitting.

    ⚠⚠ **The hold-out is applied at FRAME level, not by which fold a segment starts in.** Deciding
    per segment and then reading the whole of `grid[lo:hi]` leaks: a segment running 59-65 s is
    assigned to the training fold by its start and then carries five seconds of held-out truth
    into the mapping. Demonstrated by changing only the held-out truth and watching a track's
    fitted owner flip from B to A -- and it is not a constructed case, 36 to 39 segments cross a
    minute boundary on a real episode (2026-09-20 audit, finding 4). Masking the frames keeps the
    boundary-crossing segment, and uses only the part of it that is in the training fold.
    """
    from finesub.speech.diarization.segments import segments_from_activity

    # Which frames belong to the held-out fold. Built once: the same answer for every segment,
    # and it is the thing that has to agree with how `cost_with` splits the rows it scores.
    frames = np.arange(len(grid))
    held_out = ((frames * lab.FRAME) // seconds).astype(np.int64) % folds == fold

    counts: dict[int, Counter] = {}
    for segment in segments_from_activity(probs.tolist()):
        lo = int(segment.start / lab.FRAME)
        hi = min(int(np.ceil(segment.end / lab.FRAME)), len(grid))
        if hi <= lo:
            continue
        usable = ~held_out[lo:hi]
        if not usable.any():
            continue
        span = grid[lo:hi][usable].sum(axis=0)
        bucket = counts.setdefault(segment.track, Counter())
        for index, value in enumerate(span):
            bucket[people[index]] += int(value)
    return {track: bucket.most_common(1)[0][0] for track, bucket in counts.items() if sum(bucket.values())}


def time_block_split(probs, *, pieces: int = 2):
    """Cut the FILE into `pieces` equal spans and give each track a separate id in each span.

    ⚠ The spans are equal in *file* time, not in each track's own activity, so a track that is only
    active inside one span comes out uncut and the arm is a weaker null than `pieces` suggests
    (fable 2026-09-19). That is the conservative direction -- it understates the free lunch -- but
    it is why the sweep over `pieces` matters rather than a single value.

    The null hypothesis findings 7 used and W2/W2b did not: any partition of a set cannot
    lower its weighted majority share, so "more tracks scores better" is partly arithmetic. This
    arm adds tracks WITHOUT adding information, so whatever it gains is the free lunch that a real
    backend comparison has to beat.
    """
    frames, tracks = probs.shape
    out = np.zeros((frames, tracks * pieces), dtype=probs.dtype)
    edges = [round(frames * index / pieces) for index in range(pieces + 1)]
    for track in range(tracks):
        for piece in range(pieces):
            low, high = edges[piece], edges[piece + 1]
            out[low:high, track * pieces + piece] = probs[low:high, track]
    return out
