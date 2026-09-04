"""Guards for the P8 answer-rate reader.

Its whole claim is that it keeps **"not recorded" apart from "zero"**, because
conflating them is exactly the mistake P8 exists to catch: a detector that was
never measured looks identical to one that answered zero times, and both read
as "this feature is doing nothing".

The claim was only half true when written. The block-level check (does this
artifact carry `qwen_verify` at all?) was honest, but the per-field sums used
`r[field] or 0`, which turns a *missing counter* into a zero and folds it into
the total -- and the ranked listing crashed outright on one. A historical
artifact can carry the block and still predate one of its counters, so both
paths are real.

Run explicitly (`tools/` tests are not in the default suite):

    python -m pytest tools/bench/test_detector_answer_rate.py -o addopts=""
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bench import detector_answer_rate as reader


def _artifact(path: Path, segments: int, **align) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "segments": [{"start": float(i), "end": i + 1.0} for i in range(segments)],
                "metadata": {"asr_align": align},
            }
        ),
        encoding="utf-8",
    )


# --- tally: the unit where "unmeasured" must survive -------------------------


def test_a_missing_counter_is_not_counted_as_zero() -> None:
    rows = [
        {"suspects": 4},
        {"suspects": 0},
        {},  # an older artifact: block present, this counter absent
    ]

    stat = reader.tally(rows, "suspects")

    assert stat["total"] == 4
    assert stat["runs"] == 2, "the run that never recorded it must not join the denominator"
    assert stat["runs_missing"] == 1
    assert stat["runs_nonzero"] == 1


def test_a_recorded_zero_still_counts_as_a_measurement() -> None:
    """The other half: a real zero is data and must not be dropped."""

    stat = reader.tally([{"gaps_probed": 0}, {"gaps_probed": 0}], "gaps_probed")

    assert stat["runs"] == 2
    assert stat["runs_missing"] == 0
    assert stat["runs_nonzero"] == 0


def test_a_non_numeric_counter_is_treated_as_missing() -> None:
    """Artifacts are read from disk; a corrupt field must not become a number."""

    stat = reader.tally([{"suspects": "many"}, {"suspects": None}, {"suspects": 2}], "suspects")

    assert stat["total"] == 2
    assert stat["runs"] == 1
    assert stat["runs_missing"] == 2


# --- end to end over a directory of artifacts --------------------------------


def test_an_artifact_missing_one_counter_does_not_crash_the_report(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """The ranked listing used to format the missing counter and raise."""

    _artifact(
        tmp_path / "old" / "old-aligned.json",
        segments=100,
        qwen_verify={"suspects": 3},  # no gaps_probed: predates that counter
    )
    _artifact(
        tmp_path / "new" / "new-aligned.json",
        segments=200,
        qwen_verify={"suspects": 1, "gaps_probed": 4, "gaps_recovered": 3},
        lang_redecode={"triggers": 0, "adopted": 0},
    )

    monkeypatch.setattr("sys.argv", ["detector_answer_rate", str(tmp_path)])
    assert reader.main() == 0

    out = capsys.readouterr().out
    assert "NOT RECORDED by 1" in out, "the missing counter must be named, not absorbed"
    assert "n/r" in out, "the ranked listing must render a missing counter, not crash"
    # 4 probed, 3 recovered -- the yield must come from recorded runs only.
    assert "3/4" in out


def test_lang_redecode_absent_everywhere_reads_as_unmeasured(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    _artifact(
        tmp_path / "a" / "a-aligned.json",
        segments=10,
        qwen_verify={"suspects": 0, "gaps_probed": 0, "gaps_recovered": 0},
    )

    monkeypatch.setattr("sys.argv", ["detector_answer_rate", str(tmp_path)])
    assert reader.main() == 0

    out = capsys.readouterr().out
    assert "NOT RECORDED in any scanned artifact" in out
    assert "'unmeasured', not 'zero'" in out


def test_an_empty_directory_fails_rather_than_reporting_zeroes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("sys.argv", ["detector_answer_rate", str(tmp_path)])

    assert reader.main() == 2


def test_an_unreadable_artifact_is_skipped_not_fatal(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "bad-aligned.json").write_text("{ not json", encoding="utf-8")
    _artifact(
        tmp_path / "ok" / "ok-aligned.json", segments=5, qwen_verify={"suspects": 1}
    )

    monkeypatch.setattr("sys.argv", ["detector_answer_rate", str(tmp_path)])
    assert reader.main() == 0
