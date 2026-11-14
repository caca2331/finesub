"""The speaker judge's LLM step: windows, prompts, replies, the sidecar it writes and resumes from.

No model is called: a fake client answers. What the reader then makes of the sidecar is
`test_speaker_judge_results.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from finesub.llm.chunking import Seam, SubtitleSegment, SubtitleWindow
from finesub.llm.client import LLMCallResult
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.speaker_judge import answers, prompt, run as judge_run, windows
from finesub.speech.diarization import judgments as jd


def correction(*spans):
    """Correction windows whose core rows are the given 1-based id ranges."""
    out = []
    for number, (first, last) in enumerate(spans, start=1):
        segments = [SubtitleSegment(str(i), float(i), i + 1.0, f"line {i}") for i in range(first, last + 1)]
        out.append(SubtitleWindow(f"{number:04d}", segments, [], "even", None))
    return out


def test_a_judge_window_is_a_correction_window():
    ids = [str(i) for i in range(1, 401)]
    cut = windows.split_windows(ids, correction((1, 350), (351, 400)))
    assert [(w.id, w.chunk, w.low, w.high) for w in cut] == [("0001", "0001", 0, 350), ("0002", "0002", 350, 400)]


def test_judge_windows_tile_even_where_correction_cores_do_not_start_at_the_top():
    cut = windows.split_windows([str(i) for i in range(1, 11)], correction((3, 10)))
    assert [(w.low, w.high) for w in cut] == [(0, 2), (2, 10)]


def test_a_judge_window_starts_at_the_correction_windows_first_core_row():
    """A correction window re-reads a few rows of the one before (its overlap); its own rows start after them."""
    rows = [SubtitleSegment(str(i), float(i), i + 1.0, f"line {i}") for i in range(1, 21)]
    first = SubtitleWindow("0001", rows[:12], [], "even", None)
    second = SubtitleWindow("0002", rows[8:], rows[8:12], "even", None)
    cut = windows.split_windows([str(i) for i in range(1, 21)], [first, second])
    assert [(w.id, w.low, w.high) for w in cut] == [("0001", 0, 12), ("0002", 12, 20)]


def test_shown_rows_carry_the_correction_overlap_as_context_on_each_side():
    """As much context as the correction carries over a cut: the rows starting within 30 s of it."""
    from types import SimpleNamespace

    starts = [*range(0, 100, 10), 200, *range(210, 300, 10)]  # a 100 s pause before row 10
    rows = [SimpleNamespace(start=float(s), end=s + 5.0, text="x") for s in starts]
    assert windows.JudgeWindow("0002", "0002", 5, 8).shown(rows) == range(2, 10), "three before, two after"
    assert windows.JudgeWindow("0002", "0002", 5, 10).shown(rows) == range(2, 10), "nothing after, across the pause"
    assert windows.JudgeWindow("0003", "0003", 10, 15).shown(rows) == range(10, 18), "nothing before, across the pause; after, 30 s from row 14"
    assert windows.JudgeWindow("0001", "0001", 0, 20).shown(rows) == range(0, 20), "the whole file: no context"


def test_a_window_is_halved_at_the_reasonable_boundary_nearest_the_middle():
    from types import SimpleNamespace

    rows = [SimpleNamespace(text=f"line {i}" + ("。" if i == 6 else ""), start=float(i), end=i + 0.9) for i in range(10)]
    assert windows.halves(rows, 0, 10) == [(0, 7), (7, 10)], "the sentence end after row 6 is the nearest boundary"
    plain = [SimpleNamespace(text="line", start=float(i), end=i + 0.9) for i in range(10)]
    assert windows.halves(plain, 0, 10) == [(0, 5), (5, 10)], "no boundary: the middle"
    assert windows.halves(plain, 3, 4) == []


# --- rows, cells, profiles ------------------------------------------------------------------------


def speaker(track, *others, coverage=0.9, mark=None):
    payload = {"track": track, "tracks": [track, *others], "block": 0, "labels": [{"role": "unknown", "conf": "high"}],
               "coverage": {str(track): coverage, **{str(o): 0.3 for o in others}}}
    if mark:
        payload["mark"] = mark
        payload["review"] = mark.get("tier") == "low"
    return payload


def segments_of(specs):
    return [{"id": str(i), "start": float(2 * i), "end": 2 * i + float(seconds), "text": text,
             **({"speaker": sp} if sp is not None else {})}
            for i, (sp, seconds, text) in enumerate(specs, start=1)]


def test_cells_follow_round_eight_wording_with_marks_and_round_nine_without():
    rows = prompt.rows_of(segments_of([
        (speaker(5, 4), 2, "a"),
        (speaker(3, mark={"tier": "low", "alternative": 5}), 2, "b"),
        (speaker(3, mark={"tier": "low"}), 2, "c"),
        ({"block": 0, "labels": [], "mark": {"tier": "filled", "track": 4}}, 2, "d"),
        (None, 2, "e"),
    ]))
    assert [prompt.cell(r, marks=True) for r in rows] == ["s5+s4", "s3（低，备选 s5）", "s3（低）", "s4（补）", "??"]
    assert [prompt.cell(r, marks=False) for r in rows] == ["s5+s4", "s3", "s3", "??", "??"]


def test_profiles_prefer_lines_outside_the_shown_rows_then_coverage():
    specs = [(speaker(0, coverage=0.5), 3, "零号的一句足够长的话甲")] * 3 + [(speaker(1), 3, "一号的一句足够长的话乙")] * 20
    specs[1] = (speaker(0, coverage=0.99), 3, "零号最有把握的那一句话")
    specs[2] = (speaker(0, mark={"tier": "low"}), 3, "零号没把握的那一句很长")
    rows = [replace(row, start=40.0 * i, end=40.0 * i + 3) for i, row in enumerate(prompt.rows_of(segments_of(specs)))]
    window = windows.JudgeWindow("0001.1", "0001", 15, 18)  # rows 40 s apart: no context rows
    shown = prompt.profiles(rows, window)
    assert "零号最有把握的那一句话" in shown.text and "没把握" not in shown.text
    assert shown.text.index("- s1") < shown.text.index("- s0"), "tracks in order of speech share"
    assert all(i not in window.shown(rows) for i in shown.samples[1]), "outside the shown rows first"
    assert len(shown.samples[1]) == 2


def test_the_audio_layout_puts_samples_first_then_the_recording():
    rows = [replace(row, start=10.0 * i, end=10.0 * i + 3)
            for i, row in enumerate(prompt.rows_of(segments_of([(speaker(0), 3, "x")] * 12)))]
    window = windows.JudgeWindow("0001.1", "0001", 0, 2)  # context: the rows starting within 30 s after row 1, the last judged
    layout = prompt.audio_layout(rows, window, {0: [10, 11], 1: []})
    assert layout.pieces[:2] == [(rows[10].start, rows[10].end), (rows[11].start, rows[11].end)]
    assert layout.offset == pytest.approx(2 * (3 + prompt.SAMPLE_GAP))
    assert layout.pieces[-1] == (pytest.approx(0.0), pytest.approx(rows[4].end + 0.5)), "the edge pad stops at 0"
    assert "没有声音样本的轨：s1" in layout.table
    speech = layout.speech(rows)
    assert speech[:2] == [(0.0, 3.0), (pytest.approx(3 + prompt.SAMPLE_GAP), pytest.approx(6 + prompt.SAMPLE_GAP))]
    assert speech[2:] == [(pytest.approx(layout.offset + 10.0 * i), pytest.approx(layout.offset + 10.0 * i + 3))
                          for i in range(5)], "the recording's rows, in file seconds"


def test_the_judge_prompt_marks_the_judged_rows_and_renders_every_placeholder():
    specs = [(speaker(0), 2, f"line {i}") for i in range(1, 21)]
    specs[9] = (speaker(0, mark={"tier": "low"}), 2, "line 10")
    rows = prompt.rows_of(segments_of(specs))
    window = windows.JudgeWindow("0001.2", "0001", 8, 12)
    profile = prompt.profiles(rows, window)
    system, user, ids = prompt.judge_prompt(rows, window, title="某节目", summary="两位主播聊游戏", cast=[prompt.Candidate("佐藤", ("さとう",))],
                                            marks=True, profile=profile)
    assert ids == ["q9", "q10", "q11", "q12"]
    assert "▶ q9 18.0|s0|line 9" in user and "\n16.0|s0|line 8" in user, "context rows carry no id"
    assert "两位主播聊游戏" in system and "不据此认定谁开口" in system
    assert "轨号后面的括号是声学层对这一行的把握" in system, "the marks note is in"
    unmarked = prompt.rows_of(segments_of([(speaker(0), 2, f"line {i}") for i in range(1, 21)]))
    assert "轨号后面的括号" not in prompt.judge_prompt(unmarked, window, title="t", summary="", cast=[], marks=True,
                                                    profile=prompt.profiles(unmarked, window))[0], \
        "no window row shows a mark, so the note is left out"
    assert "这一轮附带了一个音频文件" not in system
    assert "字幕可能校对过" not in system, "only true of corrected text"
    assert "$" not in system + user
    plain, _, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False, profile=profile)
    assert "素材概要" not in plain and "轨号后面的括号" not in plain
    assert "候选人物" in user and "佐藤（さとう）" in user
    assert "认不出名字不影响判轨号）：（无）" in prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False,
                                                    profile=profile)[1]


def test_audio_rows_are_timed_inside_the_audio_file():
    rows = prompt.rows_of(segments_of([(speaker(0), 3, f"line {i}") for i in range(1, 21)]))
    window = windows.JudgeWindow("0001.2", "0001", 10, 12)
    profile = prompt.profiles(rows, window)
    layout = prompt.audio_layout(rows, window, profile.samples)
    system, user, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False,
                                          profile=profile, audio=layout)
    first = rows[window.shown(rows)[0]]
    assert f"\n{layout.offset + 0.5:.1f}|s0|{first.text}" in user
    assert "这一轮附带了一个音频文件" in system and "音频布局" in user
    assert "又说不清时，`keep` 或 `abstain`。" in system, "the plain judge keeps both answers"
    shaped, _, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False,
                                       profile=profile, audio=layout, shape=True)
    assert "又说不清时，答 `keep`。" in shaped, "the shape section keeps abstain for a possible change of speaker"


def test_candidates_once_each_and_capped():
    pool = prompt.candidates([{"name": "さとう", "zh": "佐藤", "type": "人物"}, {"name": "東京", "type": "地点"},
                              {"name": "ひな", "zh": "雏", "type": "主播"}],
                             [("ひな", ("Hina",), "intro")])
    assert [c.display for c in pool] == ["ひな（Hina）", "さとう（佐藤）"]
    many = prompt.candidates([{"name": f"人物{i}", "type": "人物"} for i in range(50)], [], [{"name": "主持"}])
    assert len(many) == prompt.MAX_CANDIDATES and many[0].name == "主持", "the speakers come first, the cap cuts the tail"


def test_a_reply_left_open_reads_as_cut_off():
    assert answers.cut_off('<judge>\n{"id":"q1","track":"keep"}\n{"id":"q2","tra')
    assert not answers.cut_off('<judge>\n{"id":"q1","track":"keep"}\n</judge>'), "a closed block is an answer or malformed"
    assert not answers.cut_off("thinking...")


# --- replies ------------------------------------------------------------------------------------


def test_a_judge_reply_is_read_and_copied_cells_count_as_their_main_track():
    text = ('<judge>\n{"id":"q1","track":"keep","why":"a"}\n{"id":"q2","track":"s5+s4","why":"b"}\n'
            '{"id":"q3","track":"@甲","why":"c"}\n{"id":"q4","track":"abstain","why":"d"}\n{"id":"q5","track":"??","why":"e"}\n'
            '</judge>\n<tracks>\ns5 = 佐藤\n@甲 = 不明\n</tracks>')
    got = answers.parse_judge(text, ["q1", "q2", "q3", "q4", "q5"], [4, 5])
    assert [(v.kind, v.track, v.label) for v in got.values()] == [
        ("keep", None, None), ("track", 5, None), ("new", None, "甲"), ("abstain", None, None), ("keep", None, None)]
    assert answers.parse_tracks(text) == {"s5": "佐藤", "@甲": "不明"}


@pytest.mark.parametrize("body", [
    '{"id":"q1","track":"keep","why":""}',                                    # misses q2
    '{"id":"q1","track":"keep"}\n{"id":"q1","track":"keep"}',                # repeats q1
    '{"id":"q1","track":"keep"}\n{"id":"q2","track":"s9"}',                  # s9 is no track
    '{"id":"q1","track":"keep"}\nnot json',
])
def test_a_reply_that_is_not_an_answer_is_refused(body):
    with pytest.raises(answers.AnswerError):
        answers.parse_judge(f"<judge>\n{body}\n</judge>", ["q1", "q2"], [0, 1])


def test_results_keep_only_what_changed_and_name_where_it_came_from():
    rows = prompt.rows_of(segments_of([(speaker(0), 2, "a"), (speaker(1), 2, "b"), (None, 2, "c"),
                                       ({"block": 0, "labels": [], "mark": {"tier": "filled", "track": 1}}, 2, "d")]))
    window = windows.JudgeWindow("0001.1", "0001", 0, 4)
    verdicts = {"q1": answers.Verdict("track", "x", track=0), "q2": answers.Verdict("track", "y", track=0),
                "q3": answers.Verdict("new", "z", label="甲"), "q4": answers.Verdict("track", "w", track=1)}
    assert answers.results(rows, window, verdicts) == [
        {"row": "2", "from": 1, "why": "y", "track": 0},
        {"row": "3", "from": None, "why": "z", "label": "甲"},
    ]


# --- the run --------------------------------------------------------------------------------------


class FakeClient:
    """Answers the judge by moving every judged row to s1."""

    def __init__(self, *, judge=None, error=None):
        self.calls = []
        self.judge = judge
        self.error = error

    def complete(self, role, messages, **kwargs):
        self.calls.append(kwargs.get("task_group"))
        user = messages[1]["content"]
        if self.error is not None:
            raise self.error
        ids = re.findall(r"▶ (q\d+) ", user)
        content = self.judge(ids) if self.judge else "<judge>\n" + "\n".join(
            json.dumps({"id": i, "track": "s1", "why": "w"}) for i in ids) + "\n</judge>\n<tracks>\ns1 = 佐藤\n</tracks>"
        return LLMCallResult(content=content, role=role, model="fake", fallback_used=False, raw_response={})


def judge_files(tmp_path, rows=8):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": segments_of([(speaker(i % 2), 2, f"line {i}") for i in range(1, rows + 1)])}),
                      encoding="utf-8")
    return stable


def run(stable, client, tmp_path, cut=((1, 4), (5, 8)), media="text", audio=None, **options):
    return judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction(*cut), profile=resolve_profile("text", "none", "quality"),
        client=client, inputs=judge_run.JudgeInputs("t", pool=(prompt.Candidate("さとう"),)), media=media,
        audio=audio, task_artifact_dir=tmp_path / "art", task_id="t", **options)


def test_a_run_writes_the_sidecar_and_the_reader_applies_it(tmp_path):
    stable = judge_files(tmp_path)
    client = FakeClient()
    report = run(stable, client, tmp_path)
    assert (report.windows, report.judged, report.failed) == (2, 2, 0)
    assert client.calls == ["correction-text"] * 2, "one call a window, on the text group"
    judged = jd.read_judgments(stable)
    assert judged.moved == {str(i): 1 for i in range(2, 9, 2)}, "only the s0 rows (even ids) changed"
    # what tools/session_replay's `speaker-judge` session rebuilds a unit's prompt from
    dumped = json.loads((tmp_path / "art" / judge_run.INPUT_FILENAME).read_text(encoding="utf-8"))
    assert dumped["pool"] == [{"name": "さとう", "others": []}] and dumped["media"] == "text"
    assert dumped["stable_json"] == str(stable) and dumped["shape"] is None


def test_each_window_is_told_its_own_summary_and_the_dump_keeps_each(tmp_path):
    """A chunked preparation task's summary differs by window (`summary_of`: own chunk first)."""
    systems = []

    class Client(FakeClient):
        def complete(self, role, messages, **kwargs):
            systems.append(messages[0]["content"])
            return super().complete(role, messages, **kwargs)

    stable = judge_files(tmp_path)
    judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction((1, 4), (5, 8)), profile=resolve_profile("text", "none", "quality"),
        client=Client(), inputs=judge_run.JudgeInputs("t", summary=lambda ids: f"概要{ids[0]}到{ids[-1]}"), media="text",
        audio=None, task_artifact_dir=tmp_path / "art", task_id="t")
    assert sorted(summary for system in systems for summary in ("概要1到4", "概要5到8") if summary in system) == [
        "概要1到4", "概要5到8"]
    dumped = json.loads((tmp_path / "art" / judge_run.INPUT_FILENAME).read_text(encoding="utf-8"))
    assert dumped["summaries"] == {"0001": "概要1到4", "0002": "概要5到8"}


def test_the_audio_judge_may_not_move_a_short_row_off_its_cell(tmp_path, monkeypatch):
    import finesub.media.ffmpeg as ffmpeg

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": segments_of([
        (speaker(0), 1.0, "short, onto a track its cell does not show"),
        (speaker(1), 1.5, "1.5 s is not short"),
        (speaker(0, 1), 0.8, "short, onto its cell's second track"),
        (speaker(1), 0.6, "short, onto a new track"),
        (speaker(2), 1.0, "line 5"), (speaker(2), 2, "line 6"),
    ])}), encoding="utf-8")
    monkeypatch.setattr(ffmpeg, "extract_spliced_audio", lambda source, pieces, gap, out: out)
    monkeypatch.setattr(judge_run._Session, "upload", lambda self, path, **speech: None)
    moves = {"q1": "s2", "q2": "s2", "q3": "s1", "q4": "@甲"}

    def judge(ids):
        return "<judge>\n" + "\n".join(json.dumps({"id": i, "track": moves.get(i, "keep"), "why": f"w{i}"}, ensure_ascii=False)
                                       for i in ids) + "\n</judge>"

    def entry(media):
        jd.sidecar_path(stable).unlink(missing_ok=True)
        run(stable, FakeClient(judge=judge), tmp_path, cut=((1, 6),), media=media, audio=tmp_path / "clip-vocal.ogg")
        return json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]

    heard = entry("audio")
    assert [r["row"] for r in heard["results"]] == ["2", "3", "4"]
    assert heard["gated"] == [{"row": "1", "track": 2, "why": "wq1"}]
    assert jd.read_judgments(stable).moved == {"2": 2, "3": 1, "4": 3}, "row 1 keeps s0; row 4 opens s3"
    read = entry("text")
    assert [r["row"] for r in read["results"]] == ["1", "2", "3", "4"] and "gated" not in read, "text is not gated"


def test_a_short_row_move_a_repair_round_patches_in_is_gated_too(tmp_path, monkeypatch):
    """The gate reads the answer that stands, whichever call delivered it: here a `<patch>` the repair round
    lays over a first reply that missed the row."""
    import finesub.media.ffmpeg as ffmpeg

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": segments_of([
        (speaker(0), 1.0, "short, onto a track its cell does not show"), (speaker(1), 2, "line 2"),
        (speaker(2), 2, "line 3"), (speaker(2), 2, "line 4"),
    ])}), encoding="utf-8")
    monkeypatch.setattr(ffmpeg, "extract_spliced_audio", lambda source, pieces, gap, out: out)
    monkeypatch.setattr(judge_run._Session, "upload", lambda self, path, **speech: None)
    replies = []

    def judge(ids):
        line = lambda i, track: json.dumps({"id": i, "track": track, "why": f"w{i}"})
        replies.append(ids)
        if len(replies) == 1:
            return "<judge>\n" + "\n".join(line(i, "keep") for i in ids[1:]) + "\n</judge>"
        return "<patch>\n" + line(ids[0], "s2") + "\n</patch>"

    run(stable, FakeClient(judge=judge), tmp_path, cut=((1, 4),), media="audio", audio=tmp_path / "clip-vocal.ogg")
    entry = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]
    assert len(replies) == 2 and entry["repairs"] == 1
    assert entry["gated"] == [{"row": "1", "track": 2, "why": "wq1"}]
    assert "1" not in jd.read_judgments(stable).moved


def test_the_short_row_gate_spares_carried_tracks_and_a_rows_own_filled_track(monkeypatch):
    # marks hidden: a filled row's cell is then empty, its own (filled) track must still not be gated
    monkeypatch.setenv(judge_run.MARKS_SWITCH, "0")
    rows = prompt.rows_of(segments_of([
        (speaker(0), 1.0, "onto a carried track"),
        ({"block": 0, "labels": [], "mark": {"tier": "filled", "track": 1}}, 1.0, "onto the track its mark filled"),
        (speaker(0, 1), 1.0, "onto its cell's second track"),
        (speaker(0), 1.0, "onto a track outside its cell"),
    ]))
    window = windows.JudgeWindow("0001", "0001", 0, 4)
    verdicts = {"q1": answers.Verdict("track", "a", track=5), "q2": answers.Verdict("track", "b", track=1),
                "q3": answers.Verdict("track", "c", track=1), "q4": answers.Verdict("track", "d", track=1)}
    taken, gated = judge_run._short_row_gate(rows, window, verdicts, {5: ("0000", "甲", "x")})
    assert gated == [{"row": "4", "track": 1, "why": "d"}]
    assert taken["q4"].kind == "keep" and all(taken[k] == verdicts[k] for k in ("q1", "q2", "q3"))


def test_the_judge_may_answer_a_track_the_prompt_shows_that_never_leads_a_line(tmp_path):
    """An alternative (or a filled or second track) is in the cells the judge reads though no profile is
    written for it: naming it is an answer, not a malformed reply."""
    stable = tmp_path / "clip-stable.json"
    rows = [(speaker(i % 2), 2, f"line {i}") for i in range(1, 9)]
    rows[2] = (speaker(0, mark={"tier": "low", "alternative": 7}), 2, "line 3")
    stable.write_text(json.dumps({"segments": segments_of(rows)}), encoding="utf-8")
    client = FakeClient(judge=lambda ids: "<judge>\n" + "\n".join(
        json.dumps({"id": i, "track": "s7" if i == "q3" else "keep", "why": "w"}) for i in ids) + "\n</judge>")
    report = run(stable, client, tmp_path, cut=((1, 8),))
    assert (report.judged, report.failed) == (1, 0)
    assert jd.read_judgments(stable).moved == {"3": 7}


def test_a_resumed_run_judges_only_what_is_missing_and_keeps_the_window_list(tmp_path):
    stable = judge_files(tmp_path)
    run(stable, FakeClient(), tmp_path)
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    payload["windows"][1]["status"] = "failed"
    jd.sidecar_path(stable).write_text(json.dumps(payload), encoding="utf-8")
    client = FakeClient()
    report = run(stable, client, tmp_path, cut=((1, 8),))  # the correction re-planned: the sidecar's list stands
    assert (report.reused, report.judged) == (1, 1) and len(client.calls) == 1
    assert [w["id"] for w in json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]] == ["0001", "0002"]


def test_changed_rows_recut_the_windows(tmp_path):
    stable = judge_files(tmp_path)
    run(stable, FakeClient(), tmp_path)
    payload = json.loads(stable.read_text(encoding="utf-8"))
    payload["segments"][0]["text"] = "re-transcribed"
    stable.write_text(json.dumps(payload), encoding="utf-8")
    report = run(stable, FakeClient(), tmp_path, cut=((1, 8),))
    assert report.windows == 1 and report.judged == 1


def test_a_reply_that_never_parses_fails_the_window_after_four_calls(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    client = FakeClient(judge=lambda ids: "<judge>\nnope\n</judge>")
    report = run(stable, client, tmp_path, cut=((1, 4),))
    assert report.failed == 1 and client.calls.count("correction-text") == judge_run.MAX_ATTEMPTS
    assert jd.read_judgments(stable).moved == {}


def test_a_quota_error_stops_the_judge_and_leaves_the_rest_pending(tmp_path):
    stable = judge_files(tmp_path)
    report = run(stable, FakeClient(error=RuntimeError("429 RESOURCE_EXHAUSTED quota")), tmp_path)
    assert report.stopped and report.pending == 2 and report.judged == 0


def test_a_request_with_an_unknown_judge_mode_is_refused(tmp_path):
    from dataclasses import replace

    from finesub.llm.correction_request import CorrectionRequestError, resolved_request

    with pytest.raises(CorrectionRequestError, match="text"):
        resolved_request(replace(task_request(tmp_path, judge_files(tmp_path)), speaker_judge="on"))


def test_a_provider_402_stops_the_judge_like_a_429(tmp_path):
    import httpx

    from finesub.llm.provider_transports import ProviderHTTPError

    stable = judge_files(tmp_path)
    error = ProviderHTTPError("HTTP 402 Insufficient Balance", httpx.Response(402, request=httpx.Request("POST", "http://x")))
    report = run(stable, FakeClient(error=error), tmp_path)
    assert report.stopped and report.pending == 2 and report.failed == 0


def test_no_labels_skips(tmp_path):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": segments_of([(None, 2, "a")])}), encoding="utf-8")
    assert run(stable, FakeClient(), tmp_path).skipped == "no speaker labels"


# --- inside a correction task ------------------------------------------------------------------


def task_request(tmp_path, stable, **overrides):
    from finesub.llm.correction_request import CorrectionRequest

    return CorrectionRequest(stable_json=stable, output_path=tmp_path / "out" / "clip.srt", postprocess_profile=None,
                             knowledge="none", profile=resolve_profile("text", "none", "quality"), test_profile=True,
                             speaker_judge="text", **overrides)


def fast_whole_file(monkeypatch, orchestration, rows=8):
    window = correction((1, rows))[0]
    monkeypatch.setattr(orchestration, "decide_fast_mode", lambda **kwargs: orchestration.FastDecision(
        mode="auto", enabled=True, reason="test", window=window))


def test_the_judge_runs_before_the_correction_windows(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration

    stable = judge_files(tmp_path)
    fast_whole_file(monkeypatch, orchestration)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())
    seen = {}

    def fake_execute(**kwargs):
        seen["sidecar"] = jd.sidecar_path(stable).exists()
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    assert seen == {"sidecar": True}
    report = (tmp_path / "out" / "clip.llm-artifacts" / "task-report.md")
    assert "## Speaker Judge" in report.read_text(encoding="utf-8")


def test_the_correction_windows_carry_the_turns_the_judge_left(tmp_path, monkeypatch):
    """The windows are planned on the acoustic speakers; the judge then moves rows, and the correction must
    read the turns of the judged speakers, not the planned ones."""
    import finesub.llm.correction_translation as orchestration
    from finesub.llm.chunking import load_segments_from_stable_json

    stable = judge_files(tmp_path)
    planned = load_segments_from_stable_json(stable)
    assert [s.seam for s in planned] == [Seam.NONE] + [Seam.CHANGE] * 7, "acoustic tracks alternate"
    monkeypatch.setattr(orchestration, "decide_fast_mode", lambda **kwargs: orchestration.FastDecision(
        mode="auto", enabled=True, reason="test", window=SubtitleWindow("0001", planned, [], "even", None)))
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())  # every row to s1
    seen = {}

    def fake_execute(**kwargs):
        seen["turns"] = [s.seam for window in kwargs["plan"].windows for s in window.segments]
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    assert seen["turns"] == [Seam.NONE] * 8, "one judged speaker: no turn left"


def test_a_finished_task_gets_the_judge_and_only_its_spk_column_rewritten(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration

    stable = judge_files(tmp_path)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())
    monkeypatch.setattr(orchestration, "execute_correction_windows", lambda **kwargs: pytest.fail("never re-corrected"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip-translated.srt").write_text("kept", encoding="utf-8")
    from finesub.llm.output_protocol import ANNOTATED_CSV_HEADER

    (out / "clip-annotated.csv").write_text(
        ANNOTATED_CSV_HEADER + "\nsub|1,2|1.0|0.0|甲|甲|high|1|s1+s0|n|o\nsub|3|1.0|0.0|乙|乙|high|1|s1|\n", encoding="utf-8")
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    rows = (out / "clip-annotated.csv").read_text(encoding="utf-8").splitlines()
    assert rows[1] == "sub|1,2|1.0|0.0|甲|甲|high|1|s1<s1~+s0|n|o", "the note's raw pipes stay"
    assert rows[2] == "sub|3|1.0|0.0|乙|乙|high|1|s1|", "row 3 was s1 already"
    assert (out / "clip-translated.srt").read_text(encoding="utf-8") == "kept"
    assert jd.covers(stable)
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))  # covered: nothing to do


def test_a_finished_task_without_its_saved_plan_is_judged_in_correction_sized_windows(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    from finesub.llm.stages.correction import run as correction_run

    stable = judge_files(tmp_path)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient())
    monkeypatch.setattr(orchestration, "execute_correction_windows", lambda **kwargs: pytest.fail("never re-corrected"))
    monkeypatch.setattr(correction_run, "plan_correction_windows", lambda segments, **kwargs: correction((1, 4), (5, 8)))
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip-translated.srt").write_text("kept", encoding="utf-8")
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    assert [(w["first"], w["last"]) for w in windows] == [("1", "4"), ("5", "8")], "planned afresh, not the whole file"
    assert not list(tmp_path.rglob("correction-window-plan.json")), "a finished task's plan is not written"


def test_a_finished_task_whose_windows_cannot_be_planned_keeps_its_products(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    from finesub.llm.stages.correction import run as correction_run

    stable = judge_files(tmp_path)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: pytest.fail("no judge call"))
    monkeypatch.setattr(orchestration, "execute_correction_windows", lambda **kwargs: pytest.fail("never re-corrected"))

    def unplannable(segments, **kwargs):
        raise ValueError("a row does not fit the window envelope")

    monkeypatch.setattr(correction_run, "plan_correction_windows", unplannable)
    warnings = []
    monkeypatch.setattr(orchestration, "_judge_warning", warnings.append)
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip-translated.srt").write_text("kept", encoding="utf-8")
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    assert (out / "clip-translated.srt").read_text(encoding="utf-8") == "kept"
    assert any("排窗失败" in warning for warning in warnings)


def test_the_dry_run_writes_the_judge_prompts(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    from finesub.llm.prompt_artifacts import write_prompt_artifacts

    stable = judge_files(tmp_path)
    fast_whole_file(monkeypatch, orchestration)
    shown = {}
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="plan", on_plan=shown.update,
                                                   extra_info="某节目 主播A"))
    assert shown["speaker_judge"]["windows"] == 1 and shown["speaker_judge"]["input_tokens"] > 0
    write_prompt_artifacts(shown, tmp_path / "prompts")
    judge = (tmp_path / "prompts" / "speaker-judge-0001.txt").read_text(encoding="utf-8")
    assert "候选人物" in judge and "（无）" in judge and "▶ q1 " in judge
    assert "speaker_judge_messages" not in json.loads((tmp_path / "prompts" / "plan.json").read_text(encoding="utf-8"))


def test_a_file_the_judge_cannot_run_on_counts_as_covered(tmp_path):
    """Else a finished task with the judge on but no labels would leave the finished-task shortcut on every run."""
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": segments_of([(None, 2, "a")])}), encoding="utf-8")
    assert jd.covers(stable)
    assert not jd.covers(judge_files(tmp_path))


def test_a_verdict_from_an_older_prompt_is_reused_not_judged_again(tmp_path, monkeypatch):
    stable = judge_files(tmp_path)
    run(stable, FakeClient(), tmp_path)
    before = jd.read_judgments(stable).moved
    monkeypatch.setattr(judge_run, "PROMPT_VERSION", "speaker-judge-next")
    client = FakeClient()
    report = run(stable, client, tmp_path)
    assert client.calls == [] and (report.reused, report.judged) == (2, 0)
    assert jd.read_judgments(stable).moved == before


def test_after_a_prompt_bump_only_the_owed_window_is_judged_by_the_new_prompt(tmp_path, monkeypatch):
    stable = judge_files(tmp_path)
    run(stable, FakeClient(), tmp_path)
    first = judge_run.PROMPT_VERSION
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    payload["windows"][1]["status"] = "failed"
    jd.sidecar_path(stable).write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(judge_run, "PROMPT_VERSION", "speaker-judge-next")
    client = FakeClient()
    run(stable, client, tmp_path)
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    assert len(client.calls) == 1 and [w["prompt_version"] for w in windows] == [first, "speaker-judge-next"]


def test_empty_replies_in_a_row_stop_the_judge_once_two_units_saw_them(tmp_path):
    stable = judge_files(tmp_path, rows=12)
    client = FakeClient(judge=lambda ids: "")
    report = judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction((1, 4), (5, 8), (9, 12)),
        profile=resolve_profile("text", "none", "quality"), client=client, inputs=judge_run.JudgeInputs("t"),
        media="text", audio=None, task_artifact_dir=tmp_path / "art", task_id="t")
    assert "empty replies" in report.stopped
    assert len(client.calls) == judge_run.MAX_ATTEMPTS + 1, "one unit's empties alone are not a frozen quota"
    assert report.pending == 2 and report.failed == 1
    assert "parts" not in json.dumps(json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))), \
        "free empty replies say nothing about size: the window is not cut for good"


def test_empty_replies_that_cost_output_are_not_a_frozen_quota(tmp_path):
    stable = judge_files(tmp_path, rows=12)

    class Thinker(FakeClient):
        def complete(self, role, messages, **kwargs):
            self.calls.append(1)
            return LLMCallResult(content="", role=role, model="m", fallback_used=False,
                                 raw_response={"usageMetadata": {"thoughtsTokenCount": 3000}})

    client = Thinker()
    report = run(stable, client, tmp_path, cut=((1, 4), (5, 8), (9, 12)))
    assert not report.stopped and report.failed == 3 and len(client.calls) == 3 * judge_run.MAX_ATTEMPTS


def test_labels_in_several_blocks_skip_with_a_warning(tmp_path):
    stable = tmp_path / "clip-stable.json"
    second = {**speaker(1), "block": 1}
    stable.write_text(json.dumps({"segments": segments_of([(speaker(0), 2, "a"), (second, 2, "b")])}), encoding="utf-8")
    warnings = []
    report = judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction((1, 2)), profile=resolve_profile("text", "none", "quality"),
        client=FakeClient(), inputs=judge_run.JudgeInputs("t"), media="text", audio=None,
        task_artifact_dir=tmp_path / "art", task_id="t", warn=warnings.append)
    assert report.skipped and warnings and not jd.sidecar_path(stable).exists()


def test_no_candidates_still_one_call(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    client = FakeClient()
    judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction((1, 4)), profile=resolve_profile("text", "none", "quality"),
        client=client, inputs=judge_run.JudgeInputs("t"), media="text", audio=None, task_artifact_dir=tmp_path / "art",
        task_id="t")
    assert len(client.calls) == 1
    assert "cast" not in json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]


def test_the_judge_hears_the_vocals_else_the_source_with_a_warning(tmp_path):
    from types import SimpleNamespace

    from finesub.llm.stages.speaker_judge.task import judge_audio

    stable = judge_files(tmp_path)
    request = SimpleNamespace(vocal_audio=None, stable_json=stable, audio_path=tmp_path / "source.wav")
    warnings = []
    assert judge_audio(request, "audio", warnings.append) == tmp_path / "source.wav" and warnings
    (tmp_path / "clip-vocal.flac").write_bytes(b"")
    assert judge_audio(request, "audio", warnings.append) == tmp_path / "clip-vocal.flac"
    assert judge_audio(request, "text", warnings.append) is None


def test_the_modality_is_the_switchs_and_falls_back_to_text_without_an_audio_model(monkeypatch):
    from types import SimpleNamespace

    import finesub.llm.stages.speaker_judge.task as task

    request = SimpleNamespace(speaker_judge="audio", profile=resolve_profile("text", "none", "quality"))
    monkeypatch.setattr(task, "audio_route_usable", lambda difficulty: True)
    assert task.judge_media(request, print) == "audio", "the correction reading text does not decide the judge's"
    warnings = []
    monkeypatch.setattr(task, "audio_route_usable", lambda difficulty: False)
    assert task.judge_media(request, warnings.append) == "text" and "纯文本" in warnings[0]
    assert task.judge_media(SimpleNamespace(speaker_judge="text"), print) == "text"
    assert judge_run.judge_group("audio") == "correction-mm" and judge_run.judge_group("text") == "correction-text"


def test_a_broken_judge_never_fails_the_correction(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    import finesub.llm.stages.speaker_judge.task as judge_task

    stable = judge_files(tmp_path)
    fast_whole_file(monkeypatch, orchestration)
    monkeypatch.setattr(judge_task, "run_speaker_judge", lambda **kwargs: (_ for _ in ()).throw(PermissionError("locked")))
    ran = []

    def fake_execute(**kwargs):
        ran.append(True)
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    assert ran == [True]


def test_the_judge_reads_a_chunked_preparation_read_back_from_disk(tmp_path, monkeypatch):
    """A resumed task or `--context-file` hands the judge a pack without the source order; a chunked one must
    still place each judge window in its chunk."""
    import finesub.llm.correction_translation as orchestration
    import finesub.llm.stages.speaker_judge.task as judge_task

    stable = judge_files(tmp_path)
    context = tmp_path / "research-context.json"
    context.write_text(json.dumps({"context_pack": {
        "general_context": {"c01": {"global_summary": "a"}, "c02": {"global_summary": "b"}},
        "general_chunks": [{"chunk_id": "c01", "first_source_id": "1", "last_source_id": "4"},
                           {"chunk_id": "c02", "first_source_id": "5", "last_source_id": "8"}],
    }}), encoding="utf-8")
    summaries = []

    def fake_judge(**kwargs):
        summaries.append(kwargs["inputs"].summary(["6"]))
        raise PermissionError("stop here")

    monkeypatch.setattr(judge_task, "run_speaker_judge", fake_judge)

    def fake_execute(**kwargs):
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute", fast="off", context_file=context))
    assert summaries == ["b\na"]


def test_research_mode_and_a_non_fast_dry_run_leave_no_judge_trace(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration

    stable = judge_files(tmp_path)
    monkeypatch.setattr(orchestration, "run_research_stage", lambda **kwargs: orchestration.ContextPack())
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: pytest.fail("no judge call in research mode"))
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="research", fast="off"))
    assert not jd.sidecar_path(stable).exists()
    shown = {}
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="plan", fast="off", on_plan=shown.update,
                                                   task_artifact_dir=tmp_path / "plan-art"))
    assert shown["speaker_judge"]["windows"] >= 1
    assert not (tmp_path / "plan-art" / "correction-window-plan.json").exists(), "a dry run writes no plan"


# --- who speaks: the preparation task, and tracks carried between windows ----------------------------


def test_the_preparation_task_asks_for_speakers_only_with_the_judge():
    from finesub.llm.prepare_prompts import build_prepare_task_messages

    plain = build_prepare_task_messages(transcript="t")[0]["content"]
    asked = build_prepare_task_messages(transcript="t", speakers=True)[0]["content"]
    assert "speakers" not in plain and "$" not in plain
    assert '"speakers": [{"name"' in asked and "不为这一项另外检索" in asked
    unsearched = build_prepare_task_messages(transcript="t", speakers=True, emits_queries=False,
                                             emits_entries=False)[0]["content"]
    assert "不逐句细查" in unsearched and "不为这一项另外检索" not in unsearched, "nothing to look up, nothing to forbid"
    assert asked.replace(asked[asked.index("\n6. 列出"):asked.index("\n", asked.index("\n6. 列出") + 1)], "").replace(
        asked[asked.index('\n    "speakers"'):asked.index("\n", asked.index('\n    "speakers"') + 1)], "") == plain


def test_the_correction_never_sees_the_speakers():
    from finesub.llm.prompts import ContextPack

    pack = ContextPack.from_dict({"general_context": {"global_summary": "s", "speakers": [{"name": "佐藤"}]}})
    chunked = ContextPack.from_dict({
        "general_context": {"c01": {"global_summary": "a", "speakers": [{"name": "x"}]},
                            "c02": {"global_summary": "b", "speakers": [{"name": "y"}]}},
        "general_chunks": [{"chunk_id": "c01", "first_source_id": "1", "last_source_id": "1"},
                           {"chunk_id": "c02", "first_source_id": "2", "last_source_id": "2"}],
    }).with_source_order(["1", "2"])
    assert "佐藤" not in pack.general_text_for_source_ids(["1"])
    assert "speakers" not in chunked.general_text_for_source_ids(["1"]) and "speakers" not in chunked.general_text_whole()
    assert judge_run.speakers_of(pack) == [{"name": "佐藤"}]
    # the candidates come from every chunk; a window's summary puts its own chunk first
    assert judge_run.speakers_of(chunked) == [{"name": "x"}, {"name": "y"}]
    assert judge_run.summary_of(chunked, ["2"]) == "b\na" and judge_run.summary_of(chunked, ["1"]) == "a\nb"


def test_speakers_lead_the_candidates():
    pool = prompt.candidates([{"name": "さとう", "type": "人物"}], [("ひな", (), "")], [{"name": "主持人甲"}])
    assert [c.name for c in pool] == ["主持人甲", "ひな", "さとう"]


def carrying_judge(ids, user):
    """Opens @甲 on the first row of the first window; later windows put their first row on the carried track."""
    carried = re.search(r"- (s\d+)：前面的窗口里新建的轨", user)
    first = carried.group(1) if carried else "@甲"
    return "<judge>\n" + "\n".join(json.dumps({"id": i, "track": first if n == 0 else "keep", "why": "w"})
                                   for n, i in enumerate(ids)) + "\n</judge>\n<tracks>\n" + (
        f"{first} = 佐藤" if carried else "@甲 = 佐藤") + "\n</tracks>"


class CarryClient(FakeClient):
    def complete(self, role, messages, **kwargs):
        self.calls.append(messages[1]["content"])
        ids = re.findall(r"▶ (q\d+) ", messages[1]["content"])
        return LLMCallResult(content=carrying_judge(ids, messages[1]["content"]), role=role, model="fake",
                             fallback_used=False, raw_response={})


def carry_run(stable, client, tmp_path, continuity):
    from dataclasses import replace

    return judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction((1, 4), (5, 8)),
        profile=replace(resolve_profile("text", "none", "quality"), continuity=continuity), client=client,
        inputs=judge_run.JudgeInputs("t"), media="text", audio=None, task_artifact_dir=tmp_path / "art", task_id="t",
        parallel=2)


def test_windows_judged_in_order_carry_new_tracks_on_and_may_merge_into_them(tmp_path):
    stable = judge_files(tmp_path)
    client = CarryClient()
    carry_run(stable, client, tmp_path, "serial")
    assert "前面的窗口里新建的轨（前面的窗口猜是佐藤，未确认）" in client.calls[1] and "line 1" in client.calls[1]
    windows = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"]
    assert windows[1]["results"][0]["carried"] == {"window": "0001", "label": "甲", "of": jd.verdict_of(windows[0])}
    judged = jd.read_judgments(stable)
    assert judged.moved == {"1": 2, "5": 2} and judged.new_tracks == (2,), "one track across both windows"


def test_windows_judged_in_parallel_open_their_own_tracks(tmp_path):
    stable = judge_files(tmp_path)
    client = CarryClient()
    carry_run(stable, client, tmp_path, "parallel")
    assert not any("前面的窗口里新建的轨" in call for call in client.calls)
    judged = jd.read_judgments(stable)
    assert judged.new_tracks == (2, 3) and judged.moved["1"] != judged.moved["5"]


def test_a_rejudge_is_not_shown_its_own_earlier_answers_as_carried_evidence(tmp_path):

    stable = judge_files(tmp_path)
    carry_run(stable, CarryClient(), tmp_path, "serial")
    client = CarryClient()
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    payload["windows"][1]["status"] = "failed"
    jd.sidecar_path(stable).write_text(json.dumps(payload), encoding="utf-8")
    carry_run(stable, client, tmp_path, "serial")
    assert len(client.calls) == 1 and "line 5" not in client.calls[0].split("这一段字幕")[0]


def test_a_carry_whose_opener_was_judged_again_is_not_handed_on(tmp_path):
    """Window 1 judged anew (same rows, another verdict): window 2's carry of its old 甲 is void, so window 3
    must not be shown that track as one window 2 carried."""
    from dataclasses import replace

    stable = judge_files(tmp_path, rows=12)
    profile = replace(resolve_profile("text", "none", "quality"), continuity="serial")

    def judge(client):
        judge_run.run_speaker_judge(
            stable_json=stable, correction_windows=correction((1, 4), (5, 8), (9, 12)), profile=profile, client=client,
            inputs=judge_run.JudgeInputs("t"), media="text", audio=None, task_artifact_dir=tmp_path / "art", task_id="t")

    judge(CarryClient())
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    payload["windows"][0]["results"] = [{"row": "2", "from": 0, "label": "甲", "why": "w"}]
    payload["windows"][2] = {key: payload["windows"][2][key] for key in ("id", "chunk", "first", "last")} | {"status": "pending"}
    jd.sidecar_path(stable).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    client = CarryClient()
    judge(client)
    assert len(client.calls) == 1 and "前面的窗口里新建的轨" not in client.calls[0]


# --- a window whose answer or prompt does not fit: halved once, as the correction's split ---------


def judge_up_to(fits, *, cut_off=True):
    """Answers windows of at most `fits` rows; a longer one is left open after `fits` rows (as the output
    ceiling would) or, with `cut_off=False`, comes back malformed."""
    def judge(ids):
        if len(ids) <= fits:
            return "<judge>\n" + "\n".join(json.dumps({"id": i, "track": "keep", "why": "w"}) for i in ids) + "\n</judge>"
        if not cut_off:
            return "（想了很久）"
        return "<judge>\n" + "\n".join(json.dumps({"id": i, "track": "keep", "why": "w"}) for i in ids[:fits]) + '\n{"id'
    return judge


class Ceiling(FakeClient):
    """An API whose usage says a window longer than `fits` rows hit the output ceiling."""

    def __init__(self, fits):
        super().__init__()
        self.fits = fits

    def complete(self, role, messages, **kwargs):
        ids = re.findall(r"▶ (q\d+) ", messages[1]["content"])
        self.calls.append(len(ids))
        if len(ids) > self.fits:
            return LLMCallResult(content="", role=role, model="m", fallback_used=False, requested_output_tokens=65536,
                                 raw_response={"limited": True, "usageMetadata": {"thoughtsTokenCount": 65500}})
        return LLMCallResult(content=judge_up_to(self.fits)(ids), role=role, model="m", fallback_used=False,
                             raw_response={})


@pytest.fixture
def usage_ceiling(monkeypatch):
    monkeypatch.setattr(judge_run, "is_likely_output_limited", lambda raw, max_tokens: raw.get("limited", False))


def test_an_output_ceiling_hit_halves_the_window_at_once(tmp_path, usage_ceiling):
    stable = judge_files(tmp_path, rows=40)
    client = Ceiling(25)
    report = run(stable, client, tmp_path, cut=((1, 40),))
    assert client.calls == [40, 20, 20], "no retry of a window the ceiling cut"
    assert (report.judged, report.failed, report.cut) == (2, 0, 1)
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    assert [(p["id"], p["first"], p["last"]) for p in payload["windows"][0]["parts"]] == [
        ("0001-a", "1", "20"), ("0001-b", "21", "40")]
    assert "row_limit" not in payload and jd.covers(stable)


def test_a_half_that_still_hits_the_ceiling_fails_and_is_not_halved_again(tmp_path, usage_ceiling):
    stable = judge_files(tmp_path, rows=40)
    client = Ceiling(10)
    report = run(stable, client, tmp_path, cut=((1, 40),))
    assert client.calls == [40, 20, 20] and (report.judged, report.failed) == (0, 2)


def test_only_the_window_that_did_not_fit_is_halved(tmp_path, usage_ceiling):
    stable = judge_files(tmp_path, rows=60)
    client = Ceiling(25)
    run(stable, client, tmp_path, cut=((1, 40), (41, 60)))
    assert client.calls == [40, 20, 20, 20], "nothing learned: the next window is sent whole"


def test_a_reply_left_open_halves_the_window_once_the_retries_are_spent(tmp_path):
    """An agent's usage is the whole session's, so only the reply's shape can say "cut off"."""
    stable = judge_files(tmp_path, rows=40)
    client = FakeClient(judge=judge_up_to(25))
    report = run(stable, client, tmp_path, cut=((1, 40),))
    assert len(client.calls) == judge_run.MAX_ATTEMPTS + 2 and (report.judged, report.cut) == (2, 1)


def test_a_half_still_left_open_after_its_retries_fails(tmp_path):
    stable = judge_files(tmp_path, rows=40)
    client = FakeClient(judge=judge_up_to(10))
    report = run(stable, client, tmp_path, cut=((1, 40),))
    assert len(client.calls) == 3 * judge_run.MAX_ATTEMPTS and (report.failed, report.cut) == (2, 1)


def test_only_the_last_reply_left_open_halves_the_window(tmp_path):
    """As the correction reads its final attempt: earlier open replies followed by a malformed one do not."""
    stable = judge_files(tmp_path, rows=40)
    replies = iter(["<judge>\n{\"id"] * (judge_run.MAX_ATTEMPTS - 1) + ["（想了很久）"])
    report = run(stable, FakeClient(judge=lambda ids: next(replies)), tmp_path, cut=((1, 40),))
    assert (report.failed, report.cut) == (1, 0)


def test_malformed_replies_fail_the_window_without_halving_it(tmp_path):
    stable = judge_files(tmp_path, rows=40)
    report = run(stable, FakeClient(judge=judge_up_to(25, cut_off=False)), tmp_path, cut=((1, 40),))
    assert (report.judged, report.failed, report.cut) == (0, 1, 0)


def test_calls_that_all_fail_do_not_cut_the_window(tmp_path):
    stable = judge_files(tmp_path, rows=40)
    client = FakeClient(error=OSError("Codex CLI exited with status 1"))
    report = run(stable, client, tmp_path, cut=((1, 40),))
    assert (report.failed, report.cut) == (1, 0) and len(client.calls) == judge_run.MAX_ATTEMPTS


def input_check(per_row):
    from types import SimpleNamespace

    limits = SimpleNamespace(prompt_input_limit=300, output_limit=65536, audio_tokens_per_second=32)
    return {"limits": limits, "counter": SimpleNamespace(count_text=lambda text: text.count("▶") * per_row)}


def test_a_prompt_past_the_routes_input_is_halved_before_any_call(tmp_path):
    stable = judge_files(tmp_path, rows=40)
    client = FakeClient()
    report = run(stable, client, tmp_path, cut=((1, 40),), **input_check(10))
    assert len(client.calls) == 2 and report.cut == 1 and report.judged == 2


def test_a_half_whose_prompt_is_still_too_long_fails_without_a_call(tmp_path):
    stable = judge_files(tmp_path, rows=40)
    client = FakeClient()
    report = run(stable, client, tmp_path, cut=((1, 40),), **input_check(20))
    assert client.calls == [] and (report.failed, report.cut) == (2, 1)


def test_a_resumed_run_judges_the_half_still_owed(tmp_path, usage_ceiling):
    stable = judge_files(tmp_path, rows=40)
    run(stable, Ceiling(25), tmp_path, cut=((1, 40),))
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    half = payload["windows"][0]["parts"][1]
    payload["windows"][0]["parts"][1] = {key: half[key] for key in ("id", "first", "last")} | {"status": "pending"}
    jd.sidecar_path(stable).write_text(json.dumps(payload), encoding="utf-8")
    client = Ceiling(25)
    run(stable, client, tmp_path, cut=((1, 40),))
    assert client.calls == [20]


def test_halves_judged_in_order_hand_the_first_halfs_new_track_to_the_second(tmp_path, usage_ceiling):
    from dataclasses import replace

    stable = judge_files(tmp_path, rows=40)

    class Halves(CarryClient):
        def complete(self, role, messages, **kwargs):
            if len(re.findall(r"▶ (q\d+) ", messages[1]["content"])) > 25:
                self.calls.append(messages[1]["content"])
                return LLMCallResult(content="", role=role, model="m", fallback_used=False, requested_output_tokens=65536,
                                     raw_response={"limited": True, "usageMetadata": {"thoughtsTokenCount": 65500}})
            return super().complete(role, messages, **kwargs)

    judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction((1, 40)),
        profile=replace(resolve_profile("text", "none", "quality"), continuity="serial"), client=Halves(),
        inputs=judge_run.JudgeInputs("t"), media="text", audio=None, task_artifact_dir=tmp_path / "art", task_id="t")
    halves = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]["parts"]
    assert halves[1]["results"][0]["carried"] == {"window": "0001-a", "label": "甲", "of": jd.verdict_of(halves[0])}
    assert jd.read_judgments(stable).new_tracks == (2,), "one track across both halves"


def test_a_one_row_window_at_the_ceiling_fails_rather_than_halving(tmp_path, usage_ceiling):
    stable = judge_files(tmp_path, rows=5)
    client = Ceiling(0)
    report = run(stable, client, tmp_path, cut=((1, 4), (5, 5)))
    assert client.calls == [4, 2, 2, 1] and (report.failed, report.cut) == (3, 1), "the one-row window: one call, failed"


def test_a_halved_window_whose_halves_failed_is_judged_again_by_halves(tmp_path, usage_ceiling):
    stable = judge_files(tmp_path, rows=40)
    run(stable, Ceiling(10), tmp_path, cut=((1, 40),))
    client = Ceiling(25)
    report = run(stable, client, tmp_path, cut=((1, 40),))
    assert client.calls == [20, 20] and report.judged == 2, "the halves stand: the whole window is not sent again"


def test_a_failed_judge_call_leaves_its_attempts_for_the_token_books(tmp_path):
    """The task report books what a failed call spent from this record (`token_books`)."""

    from finesub import jsonl
    from finesub.llm.knowledge.base import TASK_ARTIFACT_FILENAME

    error = RuntimeError("agent left without an answer")
    error._harness_execution_attempts = [{"backend": "local_agent", "usage": {"input_tokens": 9}}]
    error._harness_route_decision = {"effective_chain": []}
    run(judge_files(tmp_path), FakeClient(error=error), tmp_path)
    failed = [r["payload"] for r in jsonl.iter_records(tmp_path / "art" / TASK_ARTIFACT_FILENAME)
              if r["kind"] == "speaker_judge_call_error"]
    assert failed and failed[0]["execution_attempts"] == error._harness_execution_attempts
    assert failed[0]["route_decision"] == {"effective_chain": []} and failed[0]["call"].startswith("speaker_judge")
