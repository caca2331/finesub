"""Three passes' pass 1: the speaker judge with the shape section (`stages/speaker_judge`, the sidecar's `acts` in
`speech.diarization.judgments`)."""

from __future__ import annotations

import json

import pytest

from finesub.llm.stages.speaker_judge import answers, prompt, windows
from finesub.llm.stages.speaker_judge.task import judge_task
from finesub.speech.diarization import judgments as jd

from .test_speaker_judge_stage import FakeClient, judge_files, run, segments_of, speaker, task_request


def shown(rows=20):
    rows = prompt.rows_of(segments_of([(speaker(0), 2, f"line {i}") for i in range(1, rows + 1)]))
    return rows, windows.JudgeWindow("0001", "0001", 8, 12)


def test_the_shaped_prompt_adds_the_shape_section_and_the_rows_timing():
    rows, window = shown()
    profile = prompt.profiles(rows, window)
    system, user, ids = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False,
                                            profile=profile, shape=True)
    assert ids == ["q9", "q10", "q11", "q12"]
    assert "## 定字幕的形状" in system and "后面「定字幕的形状」一节还有三个可选字段" in system
    assert "大约 4 秒" in system, "the join length is the shared threshold"
    assert '"track":"keep","why":"一直在自言自语' in system, "the tracked example answers a track"
    assert "▶ q9 18.0|2.0|0.0|s0|line 9" in user and "`开始秒|时长|gap|轨号|文字`" in user
    assert "$" not in system + user
    plain, plain_user, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False,
                                               profile=profile)
    assert "定字幕的形状" not in plain and "▶ q9 18.0|s0|line 9" in plain_user
    assert "`<judge>` 与 `<tracks>` 两个块" in plain_user


def test_a_shaped_rows_gap_is_the_pause_before_it_and_an_overlap_reads_zero():
    # rows start every 2 s: the first lasts 3 s (1 s into the second), the second 2.04 s (just into the third)
    rows = prompt.rows_of(segments_of([(speaker(0), 3, "a"), (speaker(0), 2.04, "b"), (speaker(0), 2, "c")]))
    assert [prompt._timing(rows, i) for i in range(3)] == ["3.0|", "2.0|0.0", "2.0|0.0"], "never -0.0"


def test_the_shape_fields_are_read_and_the_first_rows_join_is_set_aside():
    text = ('<judge>\n{"id":"q1","track":"keep","why":"a","join":true}\n'
            '{"id":"q2","track":"keep","why":"b","join":true}\n{"id":"q3","track":"keep","why":"c","may_join":true}\n'
            '{"id":"q4","track":"keep","why":"d","drop":true,"join":true}\n{"id":"q5","track":"keep","why":"e","drop":false}\n'
            '</judge>')
    got = answers.parse_acts(text, ["q1", "q2", "q3", "q4", "q5"])
    assert got.acts == {"2": {"act": "join", "why": "b"}, "3": {"act": "may_join", "why": "c"}}
    assert got.ignored == [{"row": "1", "act": "join", "why": "first row"},
                           {"row": "4", "act": "join+drop", "why": "several acts"}]
    assert answers.parse_acts('<judge>\n{"id":"q1","why":"垫词","drop":true}\n</judge>', ["q1", "q2"]).acts == {
        "1": {"act": "drop", "why": "垫词"}}, "a first row may be dropped"
    with pytest.raises(answers.AnswerError):
        answers.parse_acts('<judge>\n{"id":"q9","why":"x","drop":true}\n</judge>', ["q1", "q2"])


def shaping(acts):
    """A judge reply moving nobody, with `acts` (`{q id: act}`) on the rows it names."""
    def judge(ids):
        return "<judge>\n" + "\n".join(json.dumps({"id": i, "track": "keep", "why": "w", **({acts[i]: True} if i in acts else {})})
                                       for i in ids) + "\n</judge>\n<tracks>\n</tracks>"
    return judge


def test_a_shaped_run_keeps_the_acts_and_a_plain_verdict_is_judged_again(tmp_path):
    stable = judge_files(tmp_path)
    run(stable, FakeClient(), tmp_path)
    assert jd.covers(stable) and not jd.covers(stable, shape=jd.SHAPE_TRACKED)
    client = FakeClient(judge=shaping({"q2": "join", "q5": "drop", "q6": "may_join"}))
    report = run(stable, client, tmp_path, shape=jd.SHAPE_TRACKED)
    assert report.judged == 2 and len(client.calls) == 2, "verdicts without acts are owed again"
    assert jd.covers(stable, shape=jd.SHAPE_TRACKED) and jd.covers(stable), "a single pass reuses them"
    shape = jd.read_acts(stable, shape=jd.SHAPE_TRACKED)
    assert shape.acts == {"2": "join", "5": "drop", "6": "may_join"}
    assert shape.dropped == {"5": "w"} and not shape.missing_rows and not shape.missing_windows
    entry = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][1]
    assert entry["ignored"] == [] and "results" in entry
    assert run(stable, FakeClient(), tmp_path, shape=jd.SHAPE_TRACKED).judged == 0


def test_a_window_whose_pass_1_failed_is_missing(tmp_path):
    stable = judge_files(tmp_path)
    run(stable, FakeClient(judge=shaping({})), tmp_path, shape=jd.SHAPE_TRACKED)
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    payload["windows"][1]["status"] = "failed"
    jd.sidecar_path(stable).write_text(json.dumps(payload), encoding="utf-8")
    shape = jd.read_acts(stable, shape=jd.SHAPE_TRACKED)
    assert shape.missing_windows == {"0002"} and shape.missing_rows == {"5", "6", "7", "8"}


def test_a_unit_answered_without_tracks_moves_nobody_and_is_judged_again(tmp_path):
    """A sidecar from a pass 1 that answered the shape alone holds units marked `trackless`: acts, no results."""
    stable = judge_files(tmp_path)
    run(stable, FakeClient(judge=shaping({"q3": "join"})), tmp_path, shape=jd.SHAPE_TRACKED)
    path = jd.sidecar_path(stable)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload["windows"]:
        entry.pop("results")
        entry["trackless"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    stale = []
    assert jd.read_judgments(stable, warn=stale.append).moved == {} and not stale, "skipped, not stale"
    assert not jd.covers(stable, shape=jd.SHAPE_TRACKED) and not jd.covers(stable)
    assert run(stable, FakeClient(judge=shaping({})), tmp_path, shape=jd.SHAPE_TRACKED).judged == 2


def test_the_judge_report_lists_the_rows_pass_1_dropped(tmp_path):
    stable = judge_files(tmp_path)
    from .test_speaker_judge_stage import correction

    judge_task(task_request(tmp_path, stable), correction((1, 4), (5, 8)), None,
               client=FakeClient(judge=shaping({"q3": "drop", "q6": "join"})), artifact_dir=tmp_path / "art",
               task_id="t", knowledge_root=None, fallback_title="clip", warn=lambda message: None,
               shape=jd.SHAPE_TRACKED)
    records = [json.loads(line) for line in (tmp_path / "art" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    report = next(record["payload"] for record in records if record["kind"] == "speaker_judge_report")
    assert report["dropped"] == [{"row": "3", "why": "w"}]
    assert report["acts"] == {"join": 1, "may_join": 0, "drop": 1} and report["without_pass1"] == []


def test_the_dry_run_preview_shows_the_pass_1_it_will_run(tmp_path):
    from finesub.llm.stages.speaker_judge.task import preview_task

    from .test_llm_speaker_turns import FakeTokenCounter
    from .test_speaker_judge_stage import correction

    stable = judge_files(tmp_path)
    request = task_request(tmp_path, stable)
    shown = preview_task(request, correction((1, 8)), knowledge_root=None, fallback_title="clip",
                         counter=FakeTokenCounter(), shape=jd.SHAPE_TRACKED)
    assert "▶ q1 2.0|2.0||s1|line 1" in shown["speaker_judge_messages"][0][1]["content"]


# --- split: only a judge that hears the audio ---------------------------------------------------------


def test_the_audio_shaped_prompt_offers_split_and_the_text_one_does_not():
    rows, window = shown()
    profile = prompt.profiles(rows, window)
    layout = prompt.audio_layout(rows, window, profile.samples)
    system, _, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False, profile=profile,
                                       audio=layout, shape=True)
    assert '`"split":"<切点之前的原文>"`' in system and "按「定字幕的形状」里的 `split` 切开" in system
    assert "还有几个可选字段" in system and "$" not in system
    text, _, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False, profile=profile,
                                     shape=True)
    assert "split" not in text
    single, _, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False, profile=profile,
                                       audio=layout)
    assert "split" not in single and "判这一行**大部分是谁说的**。" in single


def test_a_split_names_who_speaks_after_the_cut_and_a_bad_one_is_set_aside():
    text = ('<judge>\n{"id":"q1","track":"keep","why":"a","split":"そうだね","track2":"s1"}\n'
            '{"id":"q2","track":"keep","why":"b","split":"うん","track2":"@甲"}\n'
            '{"id":"q3","track":"keep","why":"c","split":"はい","track2":"keep"}\n</judge>')
    splits, ignored = answers.parse_splits(text, ["q1", "q2", "q3"], [0, 1])
    assert {row: (split.prefix, split.second.kind, split.second.track, split.second.label)
            for row, split in splits.items()} == {"1": ("そうだね", "track", 1, None), "2": ("うん", "new", None, "甲")}
    assert ignored == [{"row": "3", "act": "split", "why": "track2 'keep'"}]


def worded(*words):
    return {"words": [{"word": word, "start": float(n), "end": n + 0.9} for n, word in enumerate(words)]}


def test_a_split_snaps_to_the_nearest_word_boundary_of_the_row():
    row = worded("そう", "だね。", "で", "も")
    assert answers.split_at(row, "そうだね") == (2, "")
    assert answers.split_at(row, "そう だ") == (1, ""), "nearest boundary: そう|だね。"
    assert answers.split_at(row, "ちがう") == (None, "prefix does not open the row")
    assert answers.split_at(row, "そうだね。でも") == (None, "prefix is the whole row")
    assert answers.split_at(worded("ＡＢ", "cd"), "ab") == (1, ""), "compared in one form"
    assert answers.split_at(worded("そうだね"), "そう") == (None, "one word")
    untimed = worded("そう", "だね")
    untimed["words"][1]["start"] = None
    assert answers.split_at(untimed, "そう") == (None, "a word has no time")


def test_splits_are_placed_on_word_boundaries_and_refused_on_a_dropped_row_or_the_same_speaker():
    from finesub.llm.stages.speaker_judge import run as judge_run

    segments = [{"id": str(n), "start": 2.0 * n, "end": 2.0 * n + 1, "text": "そうだね。でも", "speaker": speaker(0),
                 **worded("そう", "だね。", "でも")} for n in (1, 2, 3, 4)]
    rows = prompt.rows_of(segments)
    unit = windows.JudgeWindow("0001", "0001", 0, 4)
    keep = answers.Verdict("keep")
    verdicts = {f"q{n}": keep for n in (1, 2, 3, 4)}
    splits = {"1": answers.Split("そうだね", answers.Verdict("track", track=1)),
              "2": answers.Split("そうだね", answers.Verdict("track", track=1)),
              "3": answers.Split("そうだね", answers.Verdict("track", track=0)),
              "4": answers.Split("そうだね", answers.Verdict("new", label="甲"))}
    placed, refused = judge_run._placed_splits(segments, rows, unit, verdicts, {"2": {"act": "drop", "why": "x"}},
                                               splits, {})
    assert placed == [{"row": "1", "prefix": "そうだね", "at": 2, "from": 0, "track": 1},
                      {"row": "4", "prefix": "そうだね", "at": 2, "from": 0, "label": "甲"}]
    assert refused == [{"row": "2", "act": "split", "why": "the row is dropped"},
                       {"row": "3", "act": "split", "why": "second half is the first half's speaker"}]


def test_the_reader_numbers_a_second_halfs_new_label_after_the_units_results(tmp_path):
    stable = judge_files(tmp_path, rows=4)
    payload = json.loads(stable.read_text(encoding="utf-8"))
    for segment in payload["segments"]:
        segment.update(worded("li", "ne"))
    stable.write_text(json.dumps(payload), encoding="utf-8")
    run(stable, FakeClient(judge=shaping({})), tmp_path, cut=((1, 4),), shape=jd.SHAPE_TRACKED)
    path = jd.sidecar_path(stable)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["windows"][0]
    entry["results"] = [{"row": "2", "from": 0, "why": "w", "label": "甲"}]  # rows alternate s1, s0
    entry["splits"] = [{"row": "1", "prefix": "li", "at": 1, "from": 1, "label": "乙"},
                       {"row": "3", "prefix": "li", "at": 1, "from": 1, "label": "甲"},
                       {"row": "4", "prefix": "li", "at": 1, "from": 99, "track": 0}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    judged = jd.read_judgments(stable)
    assert judged.moved == {"2": 2}
    assert judged.halves == {"1": (1, 3), "3": (1, 2)}, "甲 is the result's track; 乙 comes after; a stale `from` is not read"
    entry["splits"] = [{"row": "1", "prefix": "li", "at": 2, "from": 1, "track": 0}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert jd.read_judgments(stable).halves == {}, "a cut past the row's words is not read"
    entry["splits"] = [{"row": "1", "prefix": "li", "at": 0, "from": 1, "track": 0}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not jd.covers(stable, shape=jd.SHAPE_TRACKED), "a split that does not read voids the unit"


def test_an_audio_shaped_run_keeps_the_splits_it_could_place(tmp_path, monkeypatch):
    import finesub.media.ffmpeg as ffmpeg
    from finesub.llm.stages.speaker_judge import run as judge_run

    stable = tmp_path / "clip-stable.json"
    lines = [{"id": str(n), "start": 3.0 * n, "end": 3.0 * n + 2.5, "text": "そうだね。でも", "speaker": speaker(n % 2),
              **worded("そう", "だね。", "でも")} for n in range(1, 5)]
    stable.write_text(json.dumps({"segments": lines}), encoding="utf-8")
    monkeypatch.setattr(ffmpeg, "extract_spliced_audio", lambda source, pieces, gap, out: out)
    monkeypatch.setattr(judge_run._Session, "upload", lambda self, path, **_: None)

    def judge(ids):
        rows = [{"id": i, "track": "keep", "why": "w"} for i in ids]
        rows[1].update(split="そうだね", track2="s1")   # row 2 is s0: a change of speaker
        rows[2].update(split="ちがう", track2="s0")     # not the row's opening
        return "<judge>\n" + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n</judge>"

    report = run(stable, FakeClient(judge=judge), tmp_path, cut=((1, 4),), shape=jd.SHAPE_TRACKED,
                 media="audio", audio=tmp_path / "clip-vocal.ogg")
    assert report.judged == 1
    entry = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]
    assert entry["media"] == "audio"
    assert entry["splits"] == [{"row": "2", "prefix": "そうだね", "at": 2, "from": 0, "track": 1}]
    assert entry["ignored"] == [{"row": "3", "act": "split", "why": "prefix does not open the row"}]
    assert jd.read_judgments(stable).halves == {"2": (2, 1)}


def test_a_split_of_a_gated_short_row_is_checked_against_the_track_it_keeps_and_the_one_refused(tmp_path, monkeypatch):
    import finesub.media.ffmpeg as ffmpeg
    from finesub.llm.stages.speaker_judge import run as judge_run

    stable = tmp_path / "clip-stable.json"
    lines = [{"id": str(n), "start": 3.0 * n, "end": 3.0 * n + (1.0 if n % 2 == 0 else 2.5), "text": "そうだね。でも",
              "speaker": speaker(n % 2), **worded("そう", "だね。", "でも")} for n in range(1, 5)]
    stable.write_text(json.dumps({"segments": lines}), encoding="utf-8")
    monkeypatch.setattr(ffmpeg, "extract_spliced_audio", lambda source, pieces, gap, out: out)
    monkeypatch.setattr(judge_run._Session, "upload", lambda self, path, **speech: None)

    def judge(ids):
        rows = [{"id": i, "track": "keep", "why": "w"} for i in ids]
        rows[1].update(track="s1", split="そうだね", track2="s0")  # row 2: 1 s on s0, moved outside its cell
        rows[3].update(track="s1", split="そうだね", track2="s1")  # row 4: the same, its second half on that track
        return "<judge>\n" + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n</judge>"

    run(stable, FakeClient(judge=judge), tmp_path, cut=((1, 4),), shape=jd.SHAPE_TRACKED,
        media="audio", audio=tmp_path / "clip-vocal.ogg")
    entry = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]
    assert entry["gated"] == [{"row": "2", "track": 1, "why": "w"}, {"row": "4", "track": 1, "why": "w"}]
    assert entry.get("splits", []) == []
    assert entry["ignored"] == [{"row": "2", "act": "split", "why": "second half is the first half's speaker"},
                                {"row": "4", "act": "split", "why": "second half is the track the short-row gate refused"}]


# --- fill and insert: what the audio judge heard the ASR miss ----------------------------------------


def test_the_audio_shaped_prompt_offers_fill_and_insert_and_points_the_audio_note_at_insert():
    rows, window = shown()
    profile = prompt.profiles(rows, window)
    layout = prompt.audio_layout(rows, window, profile.samples)
    system, _, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False, profile=profile,
                                       audio=layout, shape=True)
    assert '`"fill":"<补全之后这一行的原文>"`' in system and '`"insert":{"start":<开始秒>' in system
    assert "还可以做三件只有听才能做的事" in system
    assert "不用为它们找行；有内容、听得清的话整段漏写，按「定字幕的形状」里的 `insert` 补一行。" in system
    single, _, _ = prompt.judge_prompt(rows, window, title="t", summary="", cast=[], marks=False, profile=profile,
                                       audio=layout)
    assert "不用为它们找行。" in single and "insert" not in single and "fill" not in single


def test_fills_and_inserts_are_read_and_a_bad_insert_is_set_aside():
    text = ('<judge>\n{"id":"q1","track":"keep","why":"a","fill":" そうだね、でもさ "}\n'
            '{"id":"q2","track":"keep","why":"b","insert":{"start":3.5,"end":"4.5","text":"ほんとに？","track":"@甲"}}\n'
            '{"id":"q3","track":"keep","why":"c","insert":{"start":5,"text":"うん","track":"s1"}}\n'
            '{"id":"q4","track":"keep","why":"d","insert":{"start":true,"text":"x","track":"s1"}}\n'
            '{"id":"q5","track":"keep","why":"e","insert":{"start":1,"text":" ","track":"s1"}}\n'
            '{"id":"q6","track":"keep","why":"f","insert":{"start":1,"text":"x","track":"keep"}}\n'
            '{"id":"q7","track":"keep","why":"g","insert":"x","fill":""}\n'
            '{"id":"q8","track":"keep","why":"h","insert":null}\n</judge>')
    ids = [f"q{n}" for n in range(1, 9)]
    assert answers.parse_fills(text, ids) == {"1": "そうだね、でもさ"}
    inserts, ignored = answers.parse_inserts(text, ids, [0, 1])
    assert {row: (insert.start, insert.end, insert.text, insert.who.kind) for row, insert in inserts.items()} == {
        "2": (3.5, 4.5, "ほんとに？", "new"), "3": (5.0, None, "うん", "track")}
    assert ignored == [{"row": "4", "act": "insert", "why": "no start"},
                       {"row": "5", "act": "insert", "why": "no text"},
                       {"row": "6", "act": "insert", "why": "track 'keep'"},
                       {"row": "7", "act": "insert", "why": "not an object"}]


def test_a_fill_must_keep_the_rows_own_text_whole_and_add_to_it():
    assert answers.fill_ends("だね。", "そう、だね。でも") == ((2, 2), "")
    assert answers.fill_ends("だね", "そうだよね") == (None, "changes the row's own text")
    assert answers.fill_ends("だね", "だね。") == (None, "adds nothing")


def test_fills_are_refused_on_a_dropped_or_split_row_and_inserts_land_on_the_source_timeline():
    from types import SimpleNamespace

    from finesub.llm.stages.speaker_judge import run as judge_run

    segments = [{"id": str(n), "start": 10.0 * n, "end": 10.0 * n + 4, "text": "だね", "speaker": speaker(0)}
                for n in (1, 2, 3, 4)]
    rows = prompt.rows_of(segments)
    fills = {"1": "そうだね", "2": "そうだね", "3": "そうだね", "4": "だよね"}
    placed, refused = judge_run._placed_fills(rows, {"2": {"act": "drop", "why": "x"}}, fills, {"3"})
    assert placed == [{"row": "1", "text": "そうだね", "head": 2, "tail": 0}]
    assert refused == [{"row": "2", "act": "fill", "why": "the row is dropped"},
                       {"row": "3", "act": "fill", "why": "the row is split"},
                       {"row": "4", "act": "fill", "why": "changes the row's own text"}]
    layout = SimpleNamespace(offset=3.0, origin=10.0)  # the audio file's 3.0 s is the source's 10.0 s
    keep = answers.Verdict("track", track=1)
    inserts = {"1": answers.Insert(8.0, 9.0, "ほんと", answers.Verdict("new", label="甲")),
               "2": answers.Insert(0.0, None, "ほんとにそう", keep),  # before its row: held at the row's start
               "3": answers.Insert(40.0, 99.0, "うん", keep),         # past the next row's end: held there
               "4": answers.Insert(40.0, None, "うん", keep)}         # the file's last row: 5 s past it at most
    assert judge_run._placed_inserts(rows, inserts, layout, {}) == [
        {"row": "1", "start": 15.0, "end": 16.0, "text": "ほんと", "label": "甲"},
        {"row": "2", "start": 20.0, "end": 20.857, "text": "ほんとにそう", "track": 1},
        {"row": "3", "start": 44.0, "end": 44.8, "text": "うん", "track": 1},
        {"row": "4", "start": 47.0, "end": 47.8, "text": "うん", "track": 1}]


def rows_at(*spans):
    return [{"id": str(n), "start": start, "end": end, "text": "x"} for n, (start, end) in enumerate(spans, start=1)]


def test_a_fill_grows_into_the_silence_and_never_over_its_neighbours():
    segments = rows_at((0.0, 2.0), (2.5, 4.0), (6.0, 8.0), (8.2, 9.0))
    fills = {"1": {"text": "a", "head": 7, "tail": 7},    # 1 s each side: the head stops at 0, the tail at row 2
             "2": {"text": "b", "head": 7, "tail": 7},    # its head finds row 1 already grown to its start
             "3": {"text": "c", "head": 0, "tail": 14}}   # 2 s of tail, stopped at row 4
    grown, _ = jd.placed(segments, fills, [])
    assert {key: (fill.start, fill.end) for key, fill in grown.items()} == {
        "1": (0.0, 2.5), "2": (2.5, 5.0), "3": (6.0, 8.2)}


def test_an_insert_shows_in_the_room_the_fills_leave_and_is_deferred_without_room():
    segments = rows_at((0.0, 2.0), (5.0, 6.0), (6.3, 7.0), (10.0, 11.0))
    fills = {"2": {"text": "b", "head": 7, "tail": 0}}  # row 2 now starts at 4.0
    inserts = [({"row": "1", "start": 1.0, "end": 4.0, "text": "p"}, 1),   # heard over row 1: shown after it
               ({"row": "2", "start": 6.0, "end": 6.2, "text": "q"}, 1),   # 0.3 s of room: deferred
               ({"row": "3", "start": 7.5, "end": 7.6, "text": "r"}, 1),   # heard short: shown 0.5 s
               ({"row": "4", "start": 11.0, "end": 13.0, "text": "s"}, 2)]  # after the file's last row
    _, shown = jd.placed(segments, fills, inserts)
    assert [(insert.after, insert.heard, insert.shown) for insert in shown] == [
        ("1", (1.0, 4.0), (2.0, 4.0)), ("2", (6.0, 6.2), None), ("3", (7.5, 7.6), (7.5, 8.0)),
        ("4", (11.0, 13.0), (11.0, 13.0))]


def test_the_reader_places_an_insert_against_the_next_units_fill_and_numbers_its_label(tmp_path):
    stable = tmp_path / "clip-stable.json"
    lines = [{"id": str(n), "start": 4.0 * n, "end": 4.0 * n + 2, "text": "だね", "speaker": speaker(n % 2)}
             for n in range(1, 5)]
    stable.write_text(json.dumps({"segments": lines}), encoding="utf-8")
    run(stable, FakeClient(judge=shaping({})), tmp_path, cut=((1, 2), (3, 4)), shape=jd.SHAPE_TRACKED)
    path = jd.sidecar_path(stable)
    payload = json.loads(path.read_text(encoding="utf-8"))
    first, second = payload["windows"]
    first["results"] = [{"row": "1", "from": 1, "why": "w", "label": "甲"}]
    first["inserts"] = [{"row": "2", "start": 10.0, "end": 12.0, "text": "ほんと", "label": "乙"},
                        {"row": "3", "start": 13.0, "end": 14.0, "text": "outside its unit", "track": 0}]
    second["fills"] = [{"row": "3", "text": "そうだね", "head": 7, "tail": 0}]  # row 3 now starts at 11.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    judged = jd.read_judgments(stable)
    assert judged.fills == {"3": jd.Filled("そうだね", 11.0, 14.0)}
    assert judged.inserts == (jd.Inserted("2", "ほんと", 3, (10.0, 12.0), (10.0, 11.0)),), \
        "乙 numbered after the unit's 甲; the next row is the next unit's, as it grew"
    first["inserts"] = [{"row": "2", "start": 12.0, "end": 10.0, "text": "x", "track": 0}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not jd.covers(stable, shape=jd.SHAPE_TRACKED), "an insert that does not read voids the unit"


def test_an_audio_shaped_run_keeps_its_fills_and_inserts_and_the_report_counts_them(tmp_path, monkeypatch):
    import finesub.media.ffmpeg as ffmpeg
    from finesub.llm.stages.speaker_judge import run as judge_run

    from .test_speaker_judge_stage import correction

    stable = tmp_path / "clip-stable.json"
    lines = [{"id": str(n), "start": 3.0 * n, "end": 3.0 * n + 2.8, "text": "だね", "speaker": speaker(n % 2)}
             for n in range(1, 5)]
    stable.write_text(json.dumps({"segments": lines}), encoding="utf-8")
    monkeypatch.setattr(ffmpeg, "extract_spliced_audio", lambda source, pieces, gap, out: out)
    monkeypatch.setattr(judge_run._Session, "upload", lambda self, path, **_: None)

    def judge(ids):
        rows = [{"id": i, "track": "keep", "why": "w"} for i in ids]
        rows[0].update(fill="そうだね")
        rows[1].update(fill="ちがう")
        rows[2].update(insert={"start": 0, "text": "ほんと", "track": "s1"})  # held at row 3's start
        return "<judge>\n" + "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n</judge>"

    from finesub.llm.stages.speaker_judge import task as judge_task_module

    monkeypatch.setattr(judge_task_module, "judge_media", lambda request, warn: "audio")
    monkeypatch.setattr(judge_task_module, "judge_audio", lambda request, media, warn: tmp_path / "clip-vocal.ogg")
    judge_task(task_request(tmp_path, stable), correction((1, 4)), None, client=FakeClient(judge=judge), artifact_dir=tmp_path / "art",
               task_id="t", knowledge_root=None, fallback_title="clip", warn=lambda message: None,
               shape=jd.SHAPE_TRACKED)
    entry = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))["windows"][0]
    assert entry["fills"] == [{"row": "1", "text": "そうだね", "head": 2, "tail": 0}]
    assert entry["inserts"] == [{"row": "3", "start": 9.0, "end": 9.8, "text": "ほんと", "track": 1}]
    assert entry["ignored"] == [{"row": "2", "act": "fill", "why": "changes the row's own text"}]
    records = [json.loads(line) for line in (tmp_path / "art" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    report = next(record["payload"] for record in records if record["kind"] == "speaker_judge_report")
    assert (report["fills"], report["inserts"], report["deferred"]) == (1, 1, 1), "row 3 to row 4: 0.2 s of room"
