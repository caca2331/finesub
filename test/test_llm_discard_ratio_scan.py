"""The scan behind `MAX_DISCARD_RATIO`'s calibration (`bench-baselines.md` 二十五).

Named `test_llm_*` because that prefix is what puts a file in the `llm` marker
(`test_packaging.test_the_domain_markers_cover_every_test_file`).

A measurement tool earns its numbers only if it measures the thing production
does. Three ways this one can quietly stop doing that, all of them found the
hard way:

- reading the reply's positions in the wrong id namespace (the model is handed
  window-local `1..N`; older archives carry global ids);
- approximating the split as equal parts (production cuts on a *reasonable
  boundary* and the second half re-includes an overlap tail);
- splitting a window that is already a leaf as if it were whole (production
  caps the total at `MAX_SPLITS`);
- counting ledger files as runs (one file can hold two `task_fingerprint`s).

⚠ Every fixture below keeps the two namespaces **apart** (global ids start at
101). They coincide for a task's first window, and a fixture that copies that
coincidence cannot see the first bug at all -- which is exactly how it shipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from finesub.llm.chunking import SubtitleSegment
from tools.discard_ratio_scan import (
    MAX_SPLITS,
    _ratios,
    _window,
    discarded_source_ids,
    leaves,
    scan,
)

#: Global ids never start at 1 in these fixtures -- see the module docstring.
FIRST_GLOBAL_ID = 101


def _translated(rows: list[str]) -> str:
    """A reply body. Rows only count inside `<translated>`, as in production."""

    return "\n".join(["<translated>", *rows, "</translated>"])


def _segments(count: int, *, sentence_end_at: set[int] = frozenset()) -> list:
    """20s lines 0.5s apart; only `sentence_end_at` ends a sentence.

    Two constants shape this. Gaps stay under `is_reasonable_boundary`'s 0.8s,
    so punctuation is the only thing that can make a cut point reasonable --
    that is how the test puts the boundary somewhere other than the middle. And
    lines are long relative to the 30s overlap window, so the overlap is a
    couple of lines rather than the whole first half (which `split_window_in_
    half` would clamp back to zero to keep the halves making progress).
    """

    return [
        SubtitleSegment(
            id=str(FIRST_GLOBAL_ID + index),
            start=index * 20.5,
            end=index * 20.5 + 20.0,
            text=("台词。" if index in sentence_end_at else "台词"),
        )
        for index in range(count)
    ]


def test_the_scan_splits_where_production_splits_not_down_the_middle() -> None:
    """Off-centre cut, and halves that overlap -- so they are not `n/2` each."""

    segments = _segments(10, sentence_end_at={2})
    first, second = leaves(_window("0001", segments), 1)

    # The only reasonable boundary is after index 2 -- the middle is 4.5.
    assert [seg.id for seg in first.segments] == ["101", "102", "103"]
    # And the second half re-includes the overlap tail, so the halves are
    # neither equal nor disjoint: an equal-parts model reads 5 and 5, and the
    # denominator the gate would divide by is wrong in both.
    assert [seg.id for seg in second.segments] == [
        str(FIRST_GLOBAL_ID + i) for i in range(2, 10)
    ]
    assert len(first.segments) + len(second.segments) > len(segments)
    assert (first.chunk_id, second.chunk_id) == ("0001-a", "0001-b")


def test_depth_is_absolute_and_stops_at_the_production_cap() -> None:
    """A ledger row that is already `0001-a` belongs to the half population.

    Splitting it as if it were whole manufactures windows production cannot
    produce -- and how many it may produce is `MAX_SPLITS`, read from
    `WindowGeometry` rather than copied, so lowering the cap (2 -> 1 on
    2026-09-03) narrows the scan in the same breath.
    """

    segments = _segments(12, sentence_end_at={5})
    record = {
        "run": "r",
        "fingerprint": "f",
        "chunk_id": "0001-a",
        "segments": segments,
        "discarded": set(),
    }

    assert [row[4] for row in _ratios([record], 0)] == []
    assert [row[4] for row in _ratios([record], 1)] == ["0001-a"]
    assert _ratios([record], MAX_SPLITS + 1) == []

    whole = {**record, "chunk_id": "0001"}
    assert [row[4] for row in _ratios([whole], 0)] == ["0001"]
    assert [row[4] for row in _ratios([whole], 1)] == ["0001-a", "0001-b"]


def test_a_local_position_reply_is_mapped_not_intersected() -> None:
    """The model answers in window-local `1..N`; the ledger lists global ids.

    Intersecting the two namespaces is not merely imprecise, it is off by an
    order of magnitude: real window `BV1ojjc6MEAs` 0002 (sources 296..605)
    discards 39 sources, and the intersection happened to keep 1.
    """

    sources = [str(FIRST_GLOBAL_ID + i) for i in range(10)]
    reply = _translated(
        ["discard|1", "discard|3,4", "sub|2|1|1|a|a|high|1|"]
        + [f"sub|{i}|1|1|a|a|high|1|" for i in range(5, 11)]
    )

    assert discarded_source_ids(reply, sources) == {"101", "103", "104"}


def test_a_global_id_reply_is_read_as_written() -> None:
    """Older archives were prompted with the stable ids, and recorded those."""

    sources = [str(FIRST_GLOBAL_ID + i) for i in range(10)]
    reply = _translated(
        ["discard|101", "discard|103,104"]
        + [f"sub|{sid}|1|1|a|a|high|1|" for sid in sources[1:]]
    )

    assert discarded_source_ids(reply, sources) == {"101", "103", "104"}


def test_a_reply_in_neither_namespace_is_reported_not_guessed() -> None:
    """Returning `None` is the point: a wrong reading is invisible downstream."""

    sources = [str(FIRST_GLOBAL_ID + i) for i in range(10)]
    reply = _translated(["discard|999", "sub|998|1|1|a|a|high|1|"])
    assert discarded_source_ids(reply, sources) is None


def test_only_translated_rows_count_and_void_rows_do_not() -> None:
    """The scan reads what the validator reads.

    `<reasoning>` and `<singles>` are sibling blocks that can carry row-shaped
    lines of their own, and a row ending in `<void>` is treated as never
    written -- so a voided `discard` is not a discard.
    """

    sources = [str(FIRST_GLOBAL_ID + i) for i in range(6)]
    reply = "\n".join(
        ["<reasoning>", "discard|2", "</reasoning>"]  # row-shaped, outside
    ) + _translated(
        [
            "discard|1",
            "discard|3 <void>",
            "sub|4|1|1|a|a|high|1|",
            "|5|1|1|a|a|high|1|",
            "sub|6|1|1|a|a|high|1|",
        ]
    )

    assert discarded_source_ids(reply, sources) == {"101"}


def _ledger(directory: Path, rows: list[dict], segments: list) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "x-stable.json").write_text(
        json.dumps(
            {
                "segments": [
                    {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
                    for s in segments
                ]
            }
        ),
        encoding="utf-8",
    )
    artifacts = directory / "x.llm-artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / "correction-windows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )


def test_two_fingerprints_in_one_ledger_are_two_replies(tmp_path: Path) -> None:
    """A ledger file is not a run.

    The same source corrected again under a different configuration writes a
    second row for the same chunk; both are independent observations of what a
    correct reply discards. A repeat of the *same* fingerprint is one run
    re-emitting its own row, and only the last of those counts.
    """

    segments = _segments(4)
    sources = [s.id for s in segments]
    row = {
        "chunk_id": "0001",
        "source_ids": sources,
        "content": _translated(["sub|1|1|1|a|a|high|1|"]),
    }
    _ledger(
        tmp_path / "run",
        [
            {"parallel_entry_set": [], "task_fingerprint": "sha256:aaa"},
            {**row, "task_fingerprint": "sha256:aaa"},
            {**row, "task_fingerprint": "sha256:bbb"},
            {
                **row,
                "task_fingerprint": "sha256:bbb",
                "content": _translated(["discard|1", "sub|2|1|1|a|a|high|1|"]),
            },
        ],
        segments,
    )

    records = scan([tmp_path])

    assert [r["fingerprint"] for r in records] == ["sha256:aaa", "sha256:bbb"]
    # Last row wins within a fingerprint, the bookkeeping row is not a reply,
    # and the reply's local `1` comes back as the global id it names.
    assert [sorted(r["discarded"]) for r in records] == [[], [str(FIRST_GLOBAL_ID)]]


def test_a_run_without_its_stable_json_is_reported_not_guessed(tmp_path: Path) -> None:
    """Segment text and timing decide where the split lands, so a run that did
    not keep its stable JSON cannot be scanned -- it is named, never estimated."""

    artifacts = tmp_path / "run" / "x.llm-artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "correction-windows.jsonl").write_text(
        json.dumps({"chunk_id": "0001", "source_ids": ["1"], "content": "x"}),
        encoding="utf-8",
    )

    skipped: list[str] = []
    assert scan([tmp_path], skipped) == []
    assert len(skipped) == 1 and "no stable JSON" in skipped[0]
