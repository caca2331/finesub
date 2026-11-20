"""The separator-wipe note: strict on purpose, so most of these are reasons it stays silent."""

from __future__ import annotations

import numpy as np
import pytest

from finesub.speech.verification import separator_wipe as sw

SR = 16000


def _tone(db: float, seconds: float) -> np.ndarray:
    """White noise at `db` dBFS mean power."""

    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * SR)) * 10 ** (db / 20)).astype(np.float32)


def _segments(covered: list[float]) -> list[dict]:
    """One VAD segment per 30 s window, covering that share of it."""

    return [{"start": i * 30.0, "end": i * 30.0 + share * 30.0} for i, share in enumerate(covered) if share > 0]


def test_window_coverage_splits_segments_across_windows_and_drops_the_partial_tail() -> None:
    coverage = sw.window_coverage([{"start": 20.0, "end": 40.0}], duration=75.0)
    assert coverage == pytest.approx([10 / 30, 10 / 30])


def test_prescreen_needs_both_the_share_and_the_count() -> None:
    assert sw.prescreen([0.0] * 4 + [0.5] * 9)  # 4 of 13 >= 30%
    assert not sw.prescreen([0.0] * 3 + [0.5] * 7)  # 30%, but only 3 windows
    assert not sw.prescreen([0.0] * 4 + [0.5] * 10)  # 4 windows, 28.6%


def test_a_window_is_wiped_only_when_quiet_in_the_mix_and_attenuated() -> None:
    mix = [-40.0, -40.0, -20.0, -65.0, -40.0]
    vocal = [-75.0, -50.0, -90.0, -99.0, -75.0]
    coverage = [0.0, 0.0, 0.0, 0.0, 0.2]
    # 0: wiped. 1: only 10 dB taken. 2: loud mix -- a song the separator emptied, rightly.
    # 3: the mix is silence. 4: the VAD still found speech there.
    assert sw.wiped_windows(mix, vocal, coverage) == [0]


def test_probes_spread_over_the_wiped_windows() -> None:
    assert sw.probe_windows([1, 2]) == [1, 2]
    assert sw.probe_windows(list(range(10))) == [1, 5, 8]


class _Referee:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.clips: list[np.ndarray] = []

    def transcribe_batch(self, clips):
        self.clips = list(clips)
        return [(text, "English") for text in self.texts[: len(clips)]]


def _run(monkeypatch, mix_db: list[float], vocal_db: list[float], covered: list[float], texts: list[str]):
    tracks = {
        "mix": np.concatenate([_tone(db, 30) for db in mix_db]),
        "vocal": np.concatenate([_tone(db, 30) for db in vocal_db]),
    }
    monkeypatch.setattr(sw, "_decode", lambda path: tracks[str(path)])
    referee = _Referee(texts)
    record = sw.check("mix", "vocal", segments=_segments(covered), duration=30.0 * len(mix_db), referee=referee)
    return record, referee


def test_a_wiped_whisper_passes_all_three_steps(monkeypatch) -> None:
    record, referee = _run(
        monkeypatch,
        mix_db=[-40.0] * 8 + [-30.0] * 2,
        vocal_db=[-75.0] * 8 + [-30.5] * 2,
        covered=[0.0] * 8 + [0.6] * 2,
        texts=["once upon a time", "the wolf took the shortest way", "um"],
    )
    assert record == {"windows": 10, "wiped": True, "wiped_windows": 8, "probed": 3, "heard": 2}
    assert [clip.size for clip in referee.clips] == [30 * SR] * 3
    assert "约 4 分钟" in sw.note_message(record)  # 8 windows of 30 s


def test_ordinary_content_stops_at_the_prescreen_without_decoding(monkeypatch) -> None:
    monkeypatch.setattr(sw, "_decode", lambda path: pytest.fail("decoded on ordinary content"))
    record = sw.check("mix", "vocal", segments=_segments([0.7] * 10), duration=300.0, referee=None)
    assert record == {"windows": 10, "wiped": False, "stopped_at": "prescreen"}


def test_a_loud_song_the_separator_emptied_stops_at_the_level_step(monkeypatch) -> None:
    record, _ = _run(monkeypatch, mix_db=[-18.0] * 10, vocal_db=[-90.0] * 10, covered=[0.0] * 10, texts=[])
    assert record["stopped_at"] == "level" and not record["wiped"]


def test_quiet_ambience_with_no_words_stops_at_the_referee(monkeypatch) -> None:
    record, _ = _run(monkeypatch, mix_db=[-45.0] * 10, vocal_db=[-80.0] * 10, covered=[0.0] * 10,
                     texts=["", "Thank you.", ""])
    assert record["stopped_at"] == "referee" and record["heard"] == 0 and not record["wiped"]
