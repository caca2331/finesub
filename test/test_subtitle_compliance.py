"""`subtitles/compliance.py`: the whole-file compliance table."""

from __future__ import annotations

from pathlib import Path

import pytest

from finesub.subtitles import compliance as C
from finesub.subtitles.model import SrtSegment, render_srt


def cue(start, end, text, key=None):
    return SrtSegment(0, start, end, text, key=key)


def value(label_start: str, cues, **limits):
    (row,) = [row for row in C.rows(**limits) if row.label.startswith(label_start)]
    return row.measure(cues)


def test_reading_speed_holds_only_lines_of_ten_characters_or_more_to_the_limit():
    cues = [
        cue(0.0, 1.0, "一二三四五六七八九十"),  # 10 chars in 1 s: over
        cue(2.0, 3.0, "一二三四五六七八九"),  # 9 chars in 1 s: too short to hold to it
        cue(4.0, 6.0, "一二三四五六七八九十"),  # 5 chars/s
        cue(7.0, 8.0, "abcdefghijklmnopqrst"),  # 20 half-width = 10 weighted in 1 s: over 9
    ]
    assert value("over 9 chars/s", cues) == pytest.approx(0.5)
    assert value("over 4 chars/s", cues, cps_limit=4.0) == pytest.approx(0.75)


def test_the_longest_line_of_a_cue_is_what_counts_and_durations_are_bucketed():
    cues = [cue(0.0, 0.4, "短"), cue(1.0, 1.8, "一" * 17 + "\n二"), cue(3.0, 10.5, "长")]
    assert value("longest line over 16", cues) == pytest.approx(1 / 3)
    assert value("longest line over 20", cues) == 0
    assert value("shorter than 0.5", cues) == pytest.approx(1 / 3)
    assert value("shorter than 0.833", cues) == pytest.approx(2 / 3)
    assert value("longer than 7", cues) == pytest.approx(1 / 3)
    assert value("more than 2 lines", [cue(0.0, 2.0, "一\n二\n三"), cue(3.0, 4.0, "一\n二")]) == 0.5



def test_styling_markup_counts_toward_no_length():
    cues = [cue(0.0, 1.0, "<i>一二三四五六七八</i>"), cue(2.0, 3.0, "{\\an8}<font color=\"#fff\">一二三四五六七八九</font>")]
    assert value("over 9 chars/s", cues) == 0  # 8 and 9 shown characters: under the 10 the rule needs
    assert value("longest line over 16", [cue(0.0, 2.0, "<b>" + "一" * 16 + "</b>")]) == 0

def test_gaps_and_overlaps_are_taken_by_start_within_each_file():
    cues = [
        cue(3.0, 4.0, "c", key="a"), cue(0.0, 2.0, "a", key="a"), cue(2.2, 3.1, "b", key="a"),  # gaps 0.2, -0.1
        cue(0.0, 1.0, "x", key="b"),  # another file: no gap with file a
    ]
    assert value("gaps in", cues) == pytest.approx(0.5)
    assert value("overlapping", cues) == pytest.approx(0.5)
    assert value("gaps in", cues[3:]) is None


def test_a_pair_gets_the_second_minus_the_first(tmp_path: Path, capsys):
    first, second = tmp_path / "a.srt", tmp_path / "b.srt"
    first.write_text(render_srt([cue(0.0, 0.4, "一"), cue(1.0, 3.0, "二")]), encoding="utf-8")
    second.write_text(render_srt([cue(0.0, 2.0, "一"), cue(2.5, 4.0, "二"), cue(5.0, 6.0, " ")]), encoding="utf-8")

    assert C.main([str(first), str(second)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"|  | {first} | {second} | Δ (2nd − 1st; shares in points) |"
    assert "| cues | 2 | 2 | +0 |" in lines  # the cue with no text is not counted
    assert "| shorter than 0.5 s | 50.0% | 0.0% | -50.0 |" in lines


def test_a_directory_is_one_pool_of_its_files(tmp_path: Path):
    for name, start in (("one.srt", 0.0), ("sub/two.srt", 10.0)):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(render_srt([cue(start, start + 1.0, "一"), cue(start + 1.2, start + 2.0, "二")]),
                        encoding="utf-8")
    cues = C.read_cues(tmp_path)
    assert len(cues) == 4 and len({c.key for c in cues}) == 2
    assert value("gaps in", cues) == 1.0  # the two in-file gaps, not the one between the files
