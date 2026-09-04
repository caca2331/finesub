"""The one property the threshold sweep rests on.

Not collected by the default suite (`tools/` is maintained on demand); run with
`python -m pytest tools/bench/test_ghost_threshold.py`.

A sweep that reimplements the rule measures a rule that is not the one
shipping, and the copy stays right only until someone edits one of the two.
`_candidates` therefore drives the real `drop_ghost_duplicate_segments` with
its module constants swapped -- so the thing worth pinning is that at the
CURRENT thresholds it returns exactly what production drops, and that the
swap does not leak.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from finesub.speech.recognition import segments as segment_ops
from tools.bench import probe_ghost_threshold as probe


def _ghost(text: str, at: float, span: float = 0.04) -> dict:
    return {
        "start": at,
        "end": at + span,
        "text": text,
        "words": [{"word": text, "start": at, "end": at + span}],
        "alignment_events": [{"type": "zero_duration_chunk_tail"}],
    }


def _real(text: str, start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": [{"word": text, "start": start, "end": end}],
    }


def _bed() -> list[dict]:
    return [
        _real("どうしてロザリンまで", 100.0, 101.2),
        _ghost("どうしてロザリンまで", 101.5),
        _real("まったく別の話です", 140.0, 141.4),
        _ghost("まったく", 148.0),          # too far for the current context
        _ghost("あ", 160.0),                # one char: under min_chars
    ]


def test_at_current_thresholds_it_is_the_production_rule() -> None:
    """Same input, same answer -- otherwise the sweep's baseline row is a
    different rule than the one that ships."""

    segments = _bed()
    _, dropped = segment_ops.drop_ghost_duplicate_segments(segments)
    expected = [int(record["index"]) for record in dropped]

    got = probe._candidates(
        {"segments": segments},
        max_span=segment_ops.GHOST_SEGMENT_MAX_SPAN_SEC,
        min_chars=segment_ops.GHOST_SEGMENT_MIN_CHARS,
        context=segment_ops.GHOST_SEGMENT_CONTEXT_SEC,
    )

    assert got == expected
    assert got, "the bed must actually exercise a drop, or this proves nothing"


def test_the_constant_swap_does_not_leak() -> None:
    """A sweep runs dozens of grid points in one process. If a swapped
    constant survived the call, every later point -- and anything else in that
    process -- would silently score against the wrong rule."""

    before = (
        segment_ops.GHOST_SEGMENT_MAX_SPAN_SEC,
        segment_ops.GHOST_SEGMENT_MIN_CHARS,
        segment_ops.GHOST_SEGMENT_CONTEXT_SEC,
    )

    probe._candidates(
        {"segments": _bed()}, max_span=0.4, min_chars=1, context=12.0
    )

    assert (
        segment_ops.GHOST_SEGMENT_MAX_SPAN_SEC,
        segment_ops.GHOST_SEGMENT_MIN_CHARS,
        segment_ops.GHOST_SEGMENT_CONTEXT_SEC,
    ) == before


def test_it_restores_the_constants_even_when_the_rule_raises(monkeypatch) -> None:
    before = segment_ops.GHOST_SEGMENT_MAX_SPAN_SEC

    def explode(_segments):
        raise RuntimeError("boom")

    monkeypatch.setattr(segment_ops, "drop_ghost_duplicate_segments", explode)
    try:
        probe._candidates({"segments": []}, max_span=9.9, min_chars=1, context=99.0)
    except RuntimeError:
        pass
    else:  # pragma: no cover - the stub always raises
        raise AssertionError("the stub should have raised")

    assert segment_ops.GHOST_SEGMENT_MAX_SPAN_SEC == before


def test_loosening_can_only_add() -> None:
    """The sweep's table is read as "loosening buys this much". If a looser
    grid point could DROP a candidate the current one selects, that reading is
    wrong and the refusal criterion would be scoring the wrong set."""

    segments = _bed()
    current = set(probe._candidates(
        {"segments": segments},
        max_span=segment_ops.GHOST_SEGMENT_MAX_SPAN_SEC,
        min_chars=segment_ops.GHOST_SEGMENT_MIN_CHARS,
        context=segment_ops.GHOST_SEGMENT_CONTEXT_SEC,
    ))
    loose = set(probe._candidates(
        {"segments": segments}, max_span=0.4, min_chars=1, context=12.0
    ))

    assert current <= loose
