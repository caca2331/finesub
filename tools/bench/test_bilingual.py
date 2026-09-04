"""Guards for the synthetic bilingual harness.

Both of these exist because the first version of this experiment produced
numbers that could not be reproduced from the artifacts it committed:

* the generator advanced its source cursor by summed *speech* duration while
  selecting segments by *source time*, so pauses between segments let the next
  block re-use audio already in the mix -- one run had 27 spans but only 15
  distinct texts, and every per-span rate computed from it counted that audio
  twice;
* a Japanese+Japanese control was generated before `--b-lang` existed, so half
  its truth claimed `en`, and the committed scorer reproduced 58.6% where the
  report said 100%.

So: the generator must prove it never re-uses a source range, and the scorer
must refuse a truth file it cannot trust rather than print a number from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bench import make_bilingual, score_bilingual


# --- the generator must not re-use source audio ------------------------------


def _span(source: str, begin: float, end: float) -> dict:
    return {"source": source, "source_start": begin, "source_end": end}


def test_reuse_of_a_source_range_is_detected() -> None:
    spans = [_span("a", 0.0, 5.0), _span("a", 4.0, 9.0)]

    assert make_bilingual._overlapping_reuse(spans) == 1


def test_adjacent_ranges_are_not_reuse() -> None:
    spans = [_span("a", 0.0, 5.0), _span("a", 5.0, 9.0)]

    assert make_bilingual._overlapping_reuse(spans) == 0


def test_the_same_range_from_a_different_source_is_not_reuse() -> None:
    """Two assets legitimately both have a 0-5s region."""

    spans = [_span("a", 0.0, 5.0), _span("b", 0.0, 5.0)]

    assert make_bilingual._overlapping_reuse(spans) == 0


def test_every_repeat_of_one_source_is_counted() -> None:
    spans = [_span("a", 0.0, 5.0), _span("a", 1.0, 6.0), _span("a", 2.0, 7.0)]

    assert make_bilingual._overlapping_reuse(spans) == 2


# --- the scorer must refuse rather than score a truth it cannot trust --------


def _run_dir(tmp_path: Path, truth: list[dict], segments: list[dict]) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "bilingual-truth.json").write_text(
        json.dumps(truth, ensure_ascii=False), encoding="utf-8"
    )
    (run / "aligned.json").write_text(
        json.dumps(
            {"segments": segments, "metadata": {"asr_align": {}}}, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return run


def _truth(language: str, begin: float, end: float) -> dict:
    return {
        "start": begin,
        "end": end,
        "language": language,
        "source": "asset",
        "source_start": begin,
        "source_end": end,
    }


def test_an_unknown_language_label_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    """The ja+ja control's bug: half the truth said `en` for Japanese audio."""

    run = _run_dir(
        tmp_path,
        [_truth("ja", 0.0, 5.0), _truth("klingon", 5.0, 10.0)],
        [{"start": 0.0, "end": 5.0, "lang": "ja"}],
    )
    monkeypatch.setattr("sys.argv", ["score_bilingual", str(run)])

    assert score_bilingual.main() == 2
    assert "REFUSED" in capsys.readouterr().out


def test_truth_without_a_source_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    run = _run_dir(
        tmp_path,
        [{"start": 0.0, "end": 5.0, "language": "ja"}],
        [{"start": 0.0, "end": 5.0, "lang": "ja"}],
    )
    monkeypatch.setattr("sys.argv", ["score_bilingual", str(run)])

    assert score_bilingual.main() == 2
    assert "REFUSED" in capsys.readouterr().out


def test_a_segment_straddling_a_switch_is_not_scored(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """It belongs to neither side; scoring it against one is a coin flip."""

    run = _run_dir(
        tmp_path,
        [_truth("ja", 0.0, 10.0), _truth("en", 10.0, 20.0)],
        [
            {"start": 0.0, "end": 9.0, "lang": "ja"},      # cleanly inside ja
            {"start": 8.0, "end": 12.0, "lang": "en"},     # straddles the switch
        ],
    )
    monkeypatch.setattr("sys.argv", ["score_bilingual", str(run)])

    assert score_bilingual.main() == 0
    out = capsys.readouterr().out
    assert "skipped 1 segment" in out
    assert "1/1 = 100.0%" in out


def test_a_clean_run_scores(tmp_path: Path, monkeypatch, capsys) -> None:
    run = _run_dir(
        tmp_path,
        [_truth("ja", 0.0, 10.0), _truth("en", 10.0, 20.0)],
        [
            {"start": 0.0, "end": 9.0, "lang": "ja"},
            {"start": 11.0, "end": 19.0, "lang": "ja"},  # wrong on purpose
        ],
    )
    monkeypatch.setattr("sys.argv", ["score_bilingual", str(run)])

    assert score_bilingual.main() == 0
    out = capsys.readouterr().out
    assert "1/2 = 50.0%" in out
    assert "WRONG" in out
