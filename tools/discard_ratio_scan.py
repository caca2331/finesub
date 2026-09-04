"""How much of a correction window does a *correct* reply discard?

`output_protocol.MAX_DISCARD_RATIO` rejects a reply that discards most of its
window -- the shape a model takes when it never saw the window text at all.
Picking that threshold, and deciding which windows it may be applied to, both
rest on one measurement: the discard ratio of replies that were *right*.

    python -m tools.discard_ratio_scan out            # or any dirs

This scans archived `correction-windows.jsonl` for it, and -- because the
production gate deliberately does not apply to split leaves -- also replays
what each window would look like split, by calling the production
`chunking.split_window_in_half`. That matters: the split point is the
*reasonable boundary* nearest the middle, not the middle, and the second half
re-includes an overlap tail, so halves are neither equal nor disjoint. An
equal-parts approximation reads the worst quarter as 39/64; the real geometry
reads it as 39/63.

Every archived reply is treated as correct: these runs were reviewed and
shipped. That is the point -- the threshold has to sit above what correct
output does, so the false-positive side is what needs measuring.

**Identity.** One row is one reply to one chunk. A ledger file is not a run:
the same chunk can appear twice under two `task_fingerprint`s (the same source
corrected again under a different configuration), and those are two independent
observations, both of which count. A repeat of the *same* `(fingerprint,
chunk_id)` is one run re-emitting its own row; only the last is kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.llm.chunking import (  # noqa: E402
    CorrectionBudget,
    SubtitleSegment,
    SubtitleWindow,
    load_segments_from_stable_json,
    split_window_in_half,
)
from finesub.llm.output_protocol import _row_is_voided  # noqa: E402
from finesub.llm.output_tags import find_top_level_tag_blocks  # noqa: E402
from finesub.llm.stages.correction.context import WindowGeometry  # noqa: E402
from finesub.llm.token_budget import HeuristicTokenCounter  # noqa: E402

#: Production's own cap, not a copy: a window is split at most this many times.
MAX_SPLITS = WindowGeometry.MAX_SPLITS

#: Depth -> the name the record and the code comments use for it.
DEPTH_NAMES = {0: "whole", 1: "half", 2: "quarter"}

#: Offline and deterministic: the scan only needs segment counts, and the
#: 3-tier default counter would reach for a binary or the countTokens endpoint.
_COUNTER = HeuristicTokenCounter()

_EMPTY_BUDGET = CorrectionBudget(
    input_tokens=0,
    subtitle_input_tokens=0,
    estimated_output_tokens=0,
    token_counter_source="scan",
)


#: Row kinds carrying a position cell: `sub|…`, `discard|…`, and the default
#: kind, whose `type` cell is empty (`|3|…`).
_ROW_PREFIXES = ("sub|", "discard|", "|")


def _rows(content: str) -> list[str]:
    """The reply's CSV rows: inside `<translated>`, and not self-retracted.

    Both halves match the validator. `<reasoning>` and `<singles>` are sibling
    blocks that can hold row-shaped lines of their own, and a row ending in
    `<void>` is treated as if it were never written (`output_protocol`), so a
    voided `discard` is not a discard.
    """

    return [
        row
        for block in find_top_level_tag_blocks(content, "translated")
        for raw in block.splitlines()
        if (row := raw.strip()).startswith(_ROW_PREFIXES) and not _row_is_voided(row)
    ]


def _positions(rows: list[str], kinds: tuple[str, ...]) -> set[str]:
    """Position cells of the named row kinds, one entry per merged source.

    A row may answer several sources at once (`sub|3,4|...`), and so may a
    discard, so the cell is comma-separated -- reading it whole silently drops
    every merged row.
    """

    found: set[str] = set()
    for row in rows:
        if not row.startswith(kinds):
            continue
        fields = row.split("|")
        if len(fields) < 2:
            continue
        for part in fields[1].split(","):
            if part.strip():
                found.add(part.strip())
    return found


def discarded_source_ids(content: str, sources: list[str]) -> set[str] | None:
    """The window sources this reply discarded, as stable global ids.

    ⚠ **The archive holds two id namespaces.** Today the model is handed
    window-local positions `1..N` and `output_protocol.remap_validation_source_
    ids` maps them back; older runs were prompted with the global ids straight
    from the stable JSON, and their ledgers record those. Which one a row uses
    cannot be assumed -- for the first window of any task the two namespaces
    are literally the same numbers, which is exactly why reading it wrong stays
    invisible until a later window.

    So decide per row, from the whole reply (`sub` rows included -- a reply
    covers every source, so its positions span the namespace it is written in).
    Returns `None` when neither namespace fits, rather than guessing.
    """

    rows = _rows(content)
    positions = _positions(rows, _ROW_PREFIXES)
    discards = _positions(rows, ("discard|",))
    if not discards:
        return set()

    as_global = positions <= set(sources)
    as_local = all(
        part.isdigit() and 1 <= int(part) <= len(sources) for part in positions
    )
    if as_global:
        # Identical namespaces (a task's first window) land here too, and there
        # the two readings agree, so preferring global costs nothing.
        return discards & set(sources)
    if as_local:
        return {sources[int(part) - 1] for part in discards}
    return None


def _window(chunk_id: str, segments: list[SubtitleSegment]) -> SubtitleWindow:
    return SubtitleWindow(
        chunk_id=chunk_id,
        segments=segments,
        overlap_segments=[],
        boundary_reason="scan",
        budget=_EMPTY_BUDGET,
        clip_start=segments[0].start if segments else 0.0,
        clip_end=segments[-1].end if segments else 0.0,
    )


def leaves(window: SubtitleWindow, depth: int) -> list[SubtitleWindow]:
    """The windows a `depth`-times-split parent would actually be validated as.

    Production geometry, via `split_window_in_half`: the cut lands on the
    reasonable boundary nearest the middle, and the second half re-includes the
    overlap tail -- so the two halves overlap and rarely have equal length.
    """

    if depth == 0:
        return [window]
    halves = split_window_in_half(window, counter=_COUNTER)
    if halves is None:
        return [window]
    return [leaf for half in halves for leaf in leaves(half, depth - 1)]


def _stable_json(ledger: Path) -> Path | None:
    """The `*-stable.json` this ledger's windows were planned from.

    The artifacts directory has gone by three names over time (`<stem>.llm-
    artifacts`, `<stem>-artifacts`, plain `llm-artifacts`), so walk up instead
    of deriving a stem: the run directory is the nearest ancestor holding
    exactly one stable JSON.
    """

    for parent in list(ledger.parents)[:3]:
        found = sorted(parent.glob("*-stable.json"))
        if len(found) == 1:
            return found[0]
    return None


def scan(roots: list[Path], skipped: list[str] | None = None) -> list[dict]:
    """One record per (run, task_fingerprint, chunk_id), last row winning."""

    skipped = [] if skipped is None else skipped
    records: dict[tuple[str, str, str], dict] = {}
    for root in roots:
        for ledger in sorted(root.rglob("correction-windows.jsonl")):
            stable = _stable_json(ledger)
            if stable is None:
                skipped.append(f"{ledger.parent}: no stable JSON kept")
                continue
            run = stable.parent.name
            by_id = {seg.id: seg for seg in load_segments_from_stable_json(stable)}
            for line in ledger.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                # The ledger also carries bookkeeping rows (`parallel_entry_set`
                # headers, `split_into` markers). A reply has both a body and
                # the sources it answered.
                if not row.get("content") or not row.get("source_ids"):
                    continue
                chunk_id = str(row.get("chunk_id"))
                sources = [str(s) for s in row["source_ids"]]
                segments = [by_id[sid] for sid in sources if sid in by_id]
                if len(segments) != len(sources):
                    skipped.append(
                        f"{run} {chunk_id}: "
                        f"{len(sources) - len(segments)} of {len(sources)} "
                        "source(s) not in stable JSON"
                    )
                    continue
                discarded = discarded_source_ids(row["content"], sources)
                if discarded is None:
                    skipped.append(
                        f"{run} {chunk_id}: reply positions match neither the "
                        "window-local nor the global id namespace"
                    )
                    continue
                fingerprint = str(row.get("task_fingerprint"))
                records[(run, fingerprint, chunk_id)] = {
                    "run": run,
                    "fingerprint": fingerprint,
                    "chunk_id": chunk_id,
                    "segments": segments,
                    "discarded": discarded,
                }
    return list(records.values())


def _ratios(records: list[dict], depth: int) -> list[tuple[float, int, int, dict, str]]:
    """Every leaf that would sit at absolute split depth `depth`.

    Absolute, not relative: production splits a window at most `MAX_SPLITS`
    times *in total*, so a ledger row that is already `0001-a` contributes
    itself to depth 1 and one more round to depth 2 -- never three rounds of
    its own. Reading it the other way manufactures windows production cannot
    produce.
    """

    out = []
    for record in records:
        parent = _window(record["chunk_id"], record["segments"])
        remaining = depth - parent.split_depth
        if remaining < 0 or depth > MAX_SPLITS:
            continue
        for leaf in leaves(parent, remaining):
            if leaf.split_depth != depth:
                continue
            hit = sum(1 for seg in leaf.segments if seg.id in record["discarded"])
            out.append(
                (hit / len(leaf.segments), hit, len(leaf.segments), record, leaf.chunk_id)
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args(argv)

    skipped: list[str] = []
    records = scan(args.roots, skipped)
    sources = {r["run"] for r in records}
    configs = {(r["run"], r["fingerprint"]) for r in records}
    print(
        f"{len(records)} replies across {len(configs)} "
        f"(source, fingerprint) runs from {len(sources)} sources; "
        f"{len(skipped)} skipped"
    )
    for line in skipped:
        print(f"  skipped: {line}", file=sys.stderr)

    for depth in range(MAX_SPLITS + 1):
        ratios = sorted(_ratios(records, depth), key=lambda row: -row[0])
        if not ratios:
            continue
        values = [row[0] for row in ratios]
        p50 = values[len(values) // 2]
        p95 = values[int(len(values) * 0.05)]
        worst, hit, total, record, chunk_id = ratios[0]
        print(
            f"  {DEPTH_NAMES[depth]:<8} n={len(values):<5} p50={p50:.3f} "
            f"p95={p95:.3f} max={worst:.3f} "
            f"({record['run']} {chunk_id}: {hit}/{total})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
