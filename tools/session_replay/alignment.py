"""Carry a merge/drop gold onto a re-segmented window.

A gold names boundaries and rows by source index, and an index means nothing
once the window is cut differently: `BV1ojjc6MEAs-0001` was audited at 286
sources and the 2026-08-05 VAD change re-cut the same audio into 303 finer
ones, so every id in the gold pointed somewhere else and scoring stopped
working entirely.

What survives a re-cut is **time**. A boundary the gold judged is a moment in
the audio where one subtitle ended and the next began; if the new segmentation
still breaks there, the judgment still applies to whatever pair of rows now
meets at that moment.

The part that must not be fudged is the other direction. The gold's default is
`must_not_merge`, and it earns that by having audited *every* boundary in its
own window. A finer re-cut introduces boundaries nobody audited -- and those
are exactly the ones a good answer is most likely to join, since they are the
splits that cut phrases in half. Applying the default to them would punish
correct work. So they are a third class here: **unaudited**, scored neither
way, and reported as coverage so nobody reads a partial score as a full one.

A re-cut that *joins* sources raises the same question on the row side: a new
row built from one source the gold says to drop and one it says to keep has no
right answer, so it is demoted to `may_drop` rather than inheriting either
verdict.

That demotion is *not* the unaudited class, and the difference is the reason
coverage is two numbers rather than one. Such a row is **covered** -- the gold
does reach it -- but its verdict is permissive, and the scorer subtracts
`may_merge` and `may_drop` from both sides, so it cannot move the score
whatever the answer does. `*_covered` is how much of the window the gold
reaches; `*_discriminating` is how much of it can actually be got wrong. A
single count standing for both reads as the stricter one and is the looser.
Mixed rows are listed on their own besides, because a permissive verdict this
module manufactured is worth telling apart from one a person wrote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: How far apart two cut points may sit and still be the same moment. The old
#: and new pipelines round differently and pad edges slightly; 0.15s is well
#: inside the shortest subtitle and comfortably outside that jitter.
BOUNDARY_TOLERANCE_SECONDS = 0.15

#: How much of a new row a gold row must actually cover before its judgment
#: carries there. Two rows that merely touch at a cut point overlap by nothing;
#: two rows whose edges drifted by a millisecond overlap by a millisecond, and
#: reading that as "audited" would hand a `must_drop` to the neighbour of the
#: row that was judged. Capped at half the shorter row so a genuinely tiny
#: subtitle can still be covered.
MIN_OVERLAP_SECONDS = 0.05


@dataclass(frozen=True)
class AlignedBenchmark:
    """A gold restated in the current window's ids, plus what got lost."""

    benchmark: Mapping[str, Any]
    #: Boundaries in the current window that no gold entry covers.
    unaudited_boundaries: frozenset[str]
    #: Source ids in the current window that no gold row covers.
    unaudited_sources: frozenset[str]
    #: New rows built from gold rows that disagree -- one droppable, one that
    #: must be kept. They are demoted to `may_drop`. Listed separately because
    #: a permissive verdict a re-cut manufactured is worth telling apart from
    #: one a person wrote.
    mixed_sources: frozenset[str]
    #: Gold entries whose boundary or row no longer exists at all.
    lost_must_merge: int
    lost_may_merge: int
    lost_must_drop: int
    lost_may_drop: int
    aligned: bool = True

    def _permissive(self, *keys: str) -> frozenset[str]:
        """Gold entries whose verdict cannot move the score either way.

        `may_merge` and `may_drop` are subtracted from both sides of the
        scorer: doing the thing is not an error and not doing it is not a
        miss. A row holding one is covered by the gold and still tells you
        nothing about whether the model was right.
        """

        section = self.benchmark.get(keys[0])
        if not isinstance(section, Mapping):
            return frozenset()
        return frozenset(str(item) for item in section.get(keys[1]) or ())

    def _counts(
        self, total: int, unaudited: frozenset[str], permissive: frozenset[str]
    ) -> tuple[int, int]:
        """`(covered, discriminating)` for one axis.

        Two numbers rather than one because they answer different questions,
        and a single "scored" conflated them: covered is how much of the
        window the gold reaches at all, discriminating is how much of it can
        actually be got wrong.
        """

        covered = total - len(unaudited)
        return covered, max(0, covered - len(permissive - unaudited))

    def coverage_note(self, boundary_total: int, source_total: int) -> str:
        boundaries, hard_boundaries = self._counts(
            boundary_total, self.unaudited_boundaries, self._permissive("merge", "may_merge")
        )
        sources, hard_sources = self._counts(
            source_total, self.unaudited_sources, self._permissive("drop", "may_drop")
        )
        return (
            f"gold aligned onto a re-segmented window: "
            f"{boundaries}/{boundary_total} boundaries and "
            f"{sources}/{source_total} sources are covered by a verdict, of "
            f"which {hard_boundaries} and {hard_sources} can change the score "
            f"-- the rest hold a permissive one that is subtracted from both "
            f"sides ({len(self.mixed_sources)} of those source verdicts are "
            f"permissive only because the re-cut merged a droppable row into "
            f"one that must be kept; lost from gold: "
            f"{self.lost_must_merge} must_merge, {self.lost_may_merge} "
            f"may_merge, {self.lost_must_drop} must_drop, {self.lost_may_drop} "
            f"may_drop)"
        )

    def as_json(self, boundary_total: int, source_total: int) -> dict[str, Any]:
        """The same coverage, for a caller that is not printing prose.

        `--json` used to carry the score alone, which said nothing about the
        score being a floor over a partial gold -- exactly the reading this
        module exists to prevent, and the one an automated comparison is most
        likely to make.
        """

        boundaries, hard_boundaries = self._counts(
            boundary_total, self.unaudited_boundaries, self._permissive("merge", "may_merge")
        )
        sources, hard_sources = self._counts(
            source_total, self.unaudited_sources, self._permissive("drop", "may_drop")
        )
        return {
            "aligned": self.aligned,
            "boundaries_total": boundary_total,
            # `*_covered` is "the gold reaches this one at all"; the score can
            # only move on `*_discriminating`. One number called "scored" used
            # to stand for both, which reads as the stricter of the two and is
            # the looser.
            "boundaries_covered": boundaries,
            "boundaries_discriminating": hard_boundaries,
            "boundaries_unaudited": sorted(self.unaudited_boundaries),
            "sources_total": source_total,
            "sources_covered": sources,
            "sources_discriminating": hard_sources,
            "sources_unaudited": sorted(self.unaudited_sources),
            "sources_mixed": sorted(self.mixed_sources),
            "lost_must_merge": self.lost_must_merge,
            "lost_may_merge": self.lost_may_merge,
            "lost_must_drop": self.lost_must_drop,
            "lost_may_drop": self.lost_may_drop,
            "note": self.coverage_note(boundary_total, source_total),
        }


def _pair(left: Any, right: Any) -> str:
    return f"{left}-{right}"


def _cut_points(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, str, str]]:
    """(time, left id, right id) for every internal boundary, in order."""

    points = []
    for left, right in zip(rows, rows[1:]):
        end = float(left["start"]) + float(left["duration"])
        points.append((end, str(left["id"]), str(right["id"])))
    return points


def _rows_from_segments(segments: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(segment.id),
            "start": float(segment.start),
            "duration": float(segment.end) - float(segment.start),
        }
        for segment in segments
    ]


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _covers(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    """Does a gold row cover a new row enough to speak for it?

    Bare `> 0` made two rows that merely drifted apart by a millisecond read
    as one -- and a `must_drop` next to a `must_keep` is precisely where that
    costs something.
    """

    overlap = _overlaps(a_start, a_end, b_start, b_end)
    if overlap <= 0.0:
        return False
    floor = min(MIN_OVERLAP_SECONDS, 0.5 * min(a_end - a_start, b_end - b_start))
    return overlap > floor


#: How far a row may move and still be called the same row. Far below the
#: boundary tolerance on purpose: this answers "was this window cut the way it
#: was audited", where anything but rounding noise means no.
SAME_CUT_TOLERANCE_SECONDS = 0.005


def same_cut(gold_rows: Sequence[Mapping[str, Any]], source_segments: Sequence[Any]) -> bool:
    """Is this window segmented exactly as the gold was audited?

    The question `prepare_benchmark` has to answer before it aligns anything.
    A gold's fingerprint covers the ASR text as well as the cut, so it trips
    on a decode change that moved no boundary at all -- and aligning *that*
    would drop the only signal saying the gold has gone stale. Alignment is
    for a re-cut; this is how a re-cut is told from everything else.
    """

    new_rows = _rows_from_segments(source_segments)
    if len(new_rows) != len(gold_rows):
        return False
    for gold, new in zip(gold_rows, new_rows):
        if str(gold["id"]) != new["id"]:
            return False
        if abs(float(gold["start"]) - new["start"]) > SAME_CUT_TOLERANCE_SECONDS:
            return False
        if abs(float(gold["duration"]) - new["duration"]) > SAME_CUT_TOLERANCE_SECONDS:
            return False
    return True


def align_benchmark(
    benchmark: Mapping[str, Any],
    source_segments: Sequence[Any],
    *,
    tolerance: float = BOUNDARY_TOLERANCE_SECONDS,
) -> AlignedBenchmark:
    """Restate `benchmark` in the ids of `source_segments`.

    Requires the gold to carry the `sources` it was audited against; without
    them there is nothing to align by and the caller should keep demanding an
    exact match.
    """

    gold_rows = list(benchmark.get("sources") or ())
    if not gold_rows:
        raise ValueError(
            "Benchmark cannot be aligned: it does not carry the source rows it "
            "was audited against (add a 'sources' list)."
        )
    new_rows = _rows_from_segments(source_segments)

    gold_cuts = _cut_points(gold_rows)
    new_cuts = _cut_points(new_rows)

    # Boundary -> boundary, nearest cut point within tolerance, each used once.
    remap: dict[str, str] = {}
    taken: set[int] = set()
    for time, left, right in gold_cuts:
        best, best_gap = None, tolerance
        for index, (new_time, _new_left, _new_right) in enumerate(new_cuts):
            if index in taken:
                continue
            gap = abs(new_time - time)
            if gap <= best_gap:
                best, best_gap = index, gap
        if best is not None:
            taken.add(best)
            _t, new_left, new_right = new_cuts[best]
            remap[_pair(left, right)] = _pair(new_left, new_right)

    def carry(names: Sequence[str]) -> tuple[list[str], int]:
        kept = [remap[name] for name in names if name in remap]
        return kept, len(names) - len(kept)

    must_merge, lost_must = carry(list(benchmark.get("merge", {}).get("must_merge") or ()))
    may_merge, lost_may = carry(list(benchmark.get("merge", {}).get("may_merge") or ()))

    audited = {remap[name] for name in remap}
    all_boundaries = {_pair(left, right) for _t, left, right in new_cuts}
    unaudited_boundaries = all_boundaries - audited

    # Rows: a gold row's judgment carries to every new row it covers in time --
    # but only a row whose sources *agree* inherits a verdict. A re-cut that
    # joined a droppable source to one that must be kept produces a row for
    # which neither answer is right, and calling it `must_drop` would reward
    # deleting content the gold demands be kept. Those go neutral, and are
    # reported, for the same reason unaudited boundaries do.
    may_drop_gold = {str(name) for name in benchmark.get("drop", {}).get("may_drop") or ()}
    must_drop_gold = {str(name) for name in benchmark.get("drop", {}).get("must_drop") or ()}
    by_id = {str(row["id"]): row for row in gold_rows}

    def verdict(gold_id: str) -> str:
        if gold_id in must_drop_gold:
            return "must_drop"
        return "may_drop" if gold_id in may_drop_gold else "keep"

    provenance: dict[str, set[str]] = {row["id"]: set() for row in new_rows}
    reached: dict[str, list[str]] = {}
    for row in gold_rows:
        gold_id = str(row["id"])
        start = float(row["start"])
        end = start + float(row["duration"])
        hits = [
            new["id"]
            for new in new_rows
            if _covers(
                start, end, float(new["start"]), float(new["start"]) + float(new["duration"])
            )
        ]
        reached[gold_id] = hits
        for hit in hits:
            provenance[hit].add(verdict(gold_id))

    # A gold row named in a drop set is lost when nothing it judged survives --
    # including the case where the set names an id the gold's own sources never
    # had, which was already unscoreable and is counted rather than swallowed.
    lost_must_drop = sum(1 for gold_id in must_drop_gold if not reached.get(gold_id))
    lost_may_drop = sum(1 for gold_id in may_drop_gold if not reached.get(gold_id))

    must_drop: set[str] = set()
    may_drop: set[str] = set()
    mixed_sources: set[str] = set()
    for new_id, verdicts in provenance.items():
        if verdicts == {"must_drop"}:
            must_drop.add(new_id)
        elif "must_drop" in verdicts or "may_drop" in verdicts:
            may_drop.add(new_id)
            if "keep" in verdicts:
                mixed_sources.add(new_id)
    covered = {new_id for new_id, verdicts in provenance.items() if verdicts}
    unaudited_sources = {row["id"] for row in new_rows} - covered

    aligned_benchmark = dict(benchmark)
    aligned_benchmark["source_count"] = len(new_rows)
    aligned_benchmark.pop("source_fingerprint_sha256", None)
    # Unaudited boundaries and rows ride in the neutral classes: the scorer
    # already charges nothing either way for those, which is precisely the
    # semantics wanted, and no scoring code has to learn a third class.
    # One consequence to know: the soft/hard length surcharge only fires on
    # segments that joined a non-neutral boundary, so a row built entirely
    # from unaudited splits escapes it however long it grows. That is the
    # conservative direction -- silence about a boundary must not become a
    # penalty -- but it does mean the aligned score is a floor, not a verdict.
    aligned_benchmark["merge"] = {
        **dict(benchmark.get("merge") or {}),
        "must_merge": sorted(set(must_merge)),
        "may_merge": sorted(set(may_merge) | unaudited_boundaries),
    }
    aligned_benchmark["drop"] = {
        **dict(benchmark.get("drop") or {}),
        "must_drop": sorted(must_drop),
        "may_drop": sorted(may_drop | unaudited_sources),
    }
    return AlignedBenchmark(
        benchmark=aligned_benchmark,
        unaudited_boundaries=frozenset(unaudited_boundaries),
        unaudited_sources=frozenset(unaudited_sources),
        mixed_sources=frozenset(mixed_sources),
        lost_must_merge=lost_must,
        lost_may_merge=lost_may,
        lost_must_drop=lost_must_drop,
        lost_may_drop=lost_may_drop,
    )
