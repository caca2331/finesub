"""capableC-tracked: a single-judged task's single pass, which also names who speaks. Its input adds the acoustic
track column, read off the judge's layer; its rows write `spk`, which the task writes as the judge's sidecar.

Every other set's prompt is held byte for byte by the goldens (`test_llm_prompt_goldens.py`), which these changes
leave as they were."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from finesub.llm.chunking import CorrectionBudget, Seam, SubtitleWindow, load_product_segments
from finesub.llm.output_protocol import validate_correction_window_output
from finesub.llm.prompt_compose import PromptAssemblyError
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.prompts import ContextPack, build_correction_csv_messages
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.correction.commit import window_prompt_version
from finesub.llm.stages.correction.request_turn import checkpoint_set
from finesub.llm.window_speakers import speaker_material, with_speakers

TRACKED = "capableC-tracked"
HEADER = "type|position|spk|corrected_text|translation|conf|note"


def _speaker(track, *others, mark=None):
    payload = {"track": track, "tracks": [track, *others], "block": 0, "labels": [{"role": "unknown", "conf": "high"}],
               "coverage": {str(track): 0.9}}
    if mark:
        payload["mark"] = mark
    return payload


def _stable(tmp_path):
    """Four rows: s0, s0, s1 (low, alternative s0), s1+s0."""
    rows = [(_speaker(0), "それでさ、昨日の配信で"), (_speaker(0), "ボスに負けちゃって"),
            (_speaker(1, mark={"tier": "low", "alternative": 0}), "えー本当に?"), (_speaker(1, 0), "うんマジで")]
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"id": str(i), "start": 2.0 * i, "end": 2.0 * i + 1.5, "text": text, "speaker": sp}
        for i, (sp, text) in enumerate(rows, start=1)]}, ensure_ascii=False), encoding="utf-8")
    return stable


def _window(stable, *, tracked=True):
    segments, _layer = load_product_segments(stable)
    window = SubtitleWindow("0001", segments, [], "even", CorrectionBudget(0, 0, 0, "test"), clip_start=0.0,
                            clip_end=10.0, length_scale=1.0)
    if not tracked:
        return window
    material = speaker_material(stable, ContextPack(), knowledge_root=None, title="テスト配信")
    return with_speakers([window], material)[0]


def _messages(window, variant=TRACKED):
    return build_correction_csv_messages(window=window, profile=resolve_profile("text", "none", "intermediate"),
                                         variant=variant, knowledge_enabled=False)


def test_a_tracked_windows_fill_turn_asks_for_the_tracks_table_first(tmp_path):
    """Production passes the cell's set (`capableC` or none); the fill turn names the delivery of the set the
    window is served, so a window with speakers is asked for `<tracks>` before `<translated>` there too."""
    from finesub.llm.request_prompts import build_window_task_messages

    def fill(window):
        return build_window_task_messages(
            request_channel=True, request_reply="<requests>\n</requests>", window=window,
            profile=resolve_profile("text", "none", "intermediate"), variant="capableC",
            knowledge_enabled=False)[-1]["content"]

    stable = _stable(tmp_path)
    assert "先 `<tracks>`、再 `<translated>` 完整终稿" in fill(_window(stable))
    assert "`<tracks>`" not in fill(_window(stable, tracked=False))


def test_the_tracked_set_renders_whole(tmp_path):
    system, user = (message["content"] for message in _messages(_window(_stable(tmp_path))))
    assert "local_id|start|duration|gap|turn|track|text" in system and HEADER in system
    assert "`track` 列" in system and "`spk`" in system and "（低，备选 s5）" in system, "the marks are explained"
    assert "认人用的材料" in user and "テスト配信" in user and "- s0：占语音" in user
    assert "local_id|start|duration|gap|turn|track|text" in user
    assert "|换人|s1（低，备选 s0）|えー本当に?" in user and "|s1+s0|うんマジで" in user
    assert HEADER in user
    # v2: name who speaks on each track first, the four kinds of answer shown, the acoustic `turn` said as such
    assert "说话人复核" in system and "认人从点名开始" in system and "`spk` 写 `@山田花子`" in system
    assert "声学层按轨号推出的标记" in system and "说话人分析判定" not in system
    assert "轨号对就照抄" in user
    # v3: the naming written out as a `<tracks>` table before `<translated>`, and a row that leaves its track's
    # main says in its note who it is and why
    assert "规定顺序为：`<tracks>` → `<translated>` → 其它允许块。`<tracks>` 是本窗的轨人表，写法见「认人与定形状」。" in system
    assert "<tracks>\ns1 = 佐藤一郎" in system and "note 写一句短结论：是谁、凭什么" in system
    assert "交付终稿时以 `<tracks>` 轨人表开头、接着写 `<translated>`" in user and "note 写一句凭据" in user
    assert resolve_variant(TRACKED).fill_delivery.startswith("先 `<tracks>`")
    assert resolve_variant(TRACKED + "-polish").fill_delivery == resolve_variant(TRACKED).fill_delivery


def test_the_untracked_sets_keep_their_role_and_turn_column(tmp_path):
    system, user = (message["content"] for message in _messages(_window(_stable(tmp_path), tracked=False),
                                                                 variant="capableC"))
    assert "说话人复核" not in system and "认人从点名开始" not in system
    assert "说话人分析判定" in system and "声学层按轨号推出的标记" not in system
    assert "规定顺序为：`<translated>` → 其它允许块。`<translated>` 里不写" in system and "<tracks>" not in system
    assert "交付终稿时以 `<translated>` 开头，" in user


def test_the_window_decides_the_tracked_set(tmp_path):
    """capableC's input is as before on a window without material; a window with it is served the tracked set under
    the cell's name (as a grouped window is), and the tracked set named for a window without it is refused. A set
    with no tracked version (a single-judged task resumed on efficiency) serves capableC's."""
    stable = _stable(tmp_path)
    system, user = (message["content"] for message in _messages(_window(stable, tracked=False), "capableC"))
    assert "local_id|start|duration|gap|turn|text" in user and "|track|" not in user and "认人用的材料" not in user
    assert "spk" not in system
    assert _messages(_window(stable), "capableC") == _messages(_window(stable), TRACKED)
    with pytest.raises(PromptAssemblyError):
        _messages(_window(stable, tracked=False), TRACKED)
    assert _messages(_window(stable), "basicA") == _messages(_window(stable), TRACKED)


def test_the_windows_read_the_acoustic_labels_whatever_sidecar_is_there(tmp_path):
    """The invariant: neither the material nor the lines read a sidecar (the task's own, or one left before)."""
    from finesub.speech.diarization import judgments

    stable = _stable(tmp_path)
    before = _window(stable)
    ids = ["1", "2", "3", "4"]
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    tracks = judgments.profile_tracks([judgments.speaker_of(segment) for segment in segments])
    judgments.sidecar_path(stable).write_text(json.dumps({
        "version": judgments.SIDECAR_VERSION, "route": "single-pass", "rows_hash": judgments.rows_hash(segments),
        "windows": [{"id": "0001", "chunk": "0001", "first": "1", "last": "4", "status": "success",
                     "results": [{"row": "3", "from": 1, "track": 0}],
                     "key": judgments.window_key(segments, ids, tracks)}]}), encoding="utf-8")
    assert load_product_segments(stable)[0][2].seam is Seam.NONE, "the products read the verdict"
    assert load_product_segments(stable, verdicts=False)[0][2].seam is Seam.CHANGE
    segments_now, _layer = load_product_segments(stable, verdicts=False)
    again = with_speakers([SubtitleWindow("0001", segments_now, [], "even", CorrectionBudget(0, 0, 0, "test"),
                                          clip_start=0.0, clip_end=10.0, length_scale=1.0)],
                          speaker_material(stable, ContextPack(), knowledge_root=None, title="テスト配信"))[0]
    assert again == before
    # on a resume the plan's rows were read with the verdicts (the task's own): its windows go back on the labels
    from dataclasses import dataclass

    from finesub.llm.correction_translation import _on_product_lines

    @dataclass
    class Plan:
        windows: list

    planned = SubtitleWindow("0001", load_product_segments(stable)[0], [], "even", CorrectionBudget(0, 0, 0, "test"),
                             clip_start=0.0, clip_end=10.0, length_scale=1.0)
    [window] = _on_product_lines(Plan([planned]), stable, verdicts=False).windows
    assert window.segments[2].seam is Seam.CHANGE


def _validate(window, rows):
    text = "<translated>\n" + "\n".join([HEADER, *rows]) + "\n</translated>"
    return validate_correction_window_output(text, window, variant=resolve_variant(TRACKED))


def test_spk_legal_missing_and_illegal(tmp_path):
    window = _window(_stable(tmp_path))
    ok = _validate(window, ["sub|1|s0|a|甲|high|", "sub|2|@佐藤|b|乙|high|", "sub|3|?|c|丙|high|", "discard|4|附和"])
    assert ok.ok, ok.errors
    assert [segment.spk for segment in ok.segments] == ["s0", "@佐藤", "?"]
    for rows, needle in (
        (["sub|1|a|甲|high|"], "no spk cell"),
        (["sub|1||a|甲|high|"], "empty spk"),
        (["sub|1|s9|a|甲|high|"], "neither the profiles list"),
        (["sub|1|佐藤|a|甲|high|"], "no speaker"),
        (["sub|1|@|a|甲|high|"], "no speaker"),
    ):
        result = _validate(window, [*rows, "sub|2|s0|b|乙|high|", "sub|3|s1|c|丙|high|", "sub|4|s1|d|丁|high|"])
        assert not result.ok and any(needle in error and "Row 1" in error for error in result.errors), (rows, result)
        assert result.rows[0].errors, "the error belongs to its row, so a patch repair can send it back"


def test_a_merged_row_writes_one_spk_and_a_merge_over_a_turn_is_not_a_violation(tmp_path):
    window = _window(_stable(tmp_path))
    assert window.segments[2].seam is Seam.CHANGE
    result = _validate(window, ["sub|1,2|s0|ab|甲乙|high|", "sub|3,4|s1|cd|丙丁|high|"])
    assert result.ok, result.errors
    assert [(segment.source_ids, segment.spk) for segment in result.segments] == [(("1", "2"), "s0"), (("3", "4"), "s1")]
    across = _validate(window, ["sub|1,2,3|s0|abc|甲乙丙|high|", "sub|4|s1|d|丁|high|"])
    assert across.ok and across.turn_violation_rows == () and across.turn_notes == ()
    # a track only row 4's cell shows besides the profiles: allowed there (s0 is in the profiles anyway)
    assert _validate(window, ["sub|1,2,3|@甲|abc|甲乙丙|high|", "sub|4|s0|d|丁|high|"]).ok


def test_a_tracked_window_names_its_own_prompt_version(tmp_path):
    stable = _stable(tmp_path)
    assert window_prompt_version(_window(stable)) == "zh-subtitle-correction-tracked-v3"
    assert window_prompt_version(_window(stable, tracked=False)).startswith("zh-subtitle-correction-csv-")


def test_a_tool_session_validator_gets_the_speaker_material(tmp_path):
    from finesub.llm.agent.agent_validators import deserialize_window, serialize_window

    window = _window(_stable(tmp_path))
    again = deserialize_window(json.loads(json.dumps(serialize_window(window))))
    assert again.speakers == window.speakers
    assert "speakers" not in serialize_window(_window(_stable(tmp_path), tracked=False))


def test_the_request_turn_checkpoint_is_composed_under_the_windows_own_set(tmp_path):
    stable = _stable(tmp_path)
    tracked, plain = _window(stable), _window(stable, tracked=False)
    assert checkpoint_set(tracked) == ("", "zh-subtitle-correction-tracked-v3")
    assert checkpoint_set(plain)[0] == "" and checkpoint_set(plain)[1].startswith("zh-subtitle-correction-csv-")
    assert _messages(tracked, checkpoint_set(tracked)[0]) == _messages(tracked)  # the default set, served tracked


def test_spk_unchecked_takes_bad_cells_and_a_missing_column(tmp_path):
    """Owner Q10: out of repairs with only `spk` wrong, the window is taken; its bad cells, or the whole missing
    column, keep the acoustic labels."""
    window = _window(_stable(tmp_path))
    rows = ["sub|1|s9|a|甲|high|", "sub|2||b|乙|high|", "sub|3|s1|c|丙|high|", "sub|4|s1|d|丁|high|"]
    text = "<translated>\n" + "\n".join([HEADER, *rows]) + "\n</translated>"
    assert not validate_correction_window_output(text, window, variant=resolve_variant("capableC")).ok
    loose = validate_correction_window_output(text, window, variant=resolve_variant("capableC"), spk_unchecked=True)
    assert loose.ok and [segment.spk for segment in loose.segments] == ["s9", "", "s1", "s1"]
    bare = "<translated>\ntype|position|corrected_text|translation|conf|note\n" + "\n".join(
        f"sub|{n}|{c}|{t}|high|" for n, c, t in ((1, "a", "甲"), (2, "b", "乙"), (3, "c", "丙"), (4, "d", "丁"))) \
        + "\n</translated>"
    no_column = validate_correction_window_output(bare, window, variant=resolve_variant("capableC"), spk_unchecked=True)
    assert no_column.ok and {segment.spk for segment in no_column.segments} == {""}
    broken = text.replace("sub|3|s1|c|丙|high|", "sub|3|s1||丙|high|")
    assert not validate_correction_window_output(broken, window, variant=resolve_variant("capableC"),
                                                 spk_unchecked=True).ok, "anything else wrong still fails"


def _two_windows(stable):
    """Rows 1-2 and 3-4, the second re-reading row 2 as its overlap."""
    segments, _layer = load_product_segments(stable, verdicts=False)
    budget = CorrectionBudget(0, 0, 0, "test")
    first = SubtitleWindow("0001", segments[:2], [], "even", budget, clip_start=0.0, clip_end=4.0, length_scale=1.0)
    second = SubtitleWindow("0002", segments[1:], segments[1:2], "even", budget, clip_start=2.0, clip_end=8.0,
                            length_scale=1.0)
    material = speaker_material(stable, ContextPack(), knowledge_root=None, title="テスト配信")
    return with_speakers([first, second], material), material


def _rows(window, cells, notes=()):
    """Validated rows of ``window`` with these `spk` cells (and notes), one row per core source (harness ids)."""
    notes = list(notes) + [""] * (len(cells) - len(notes))
    lines = [f"sub|{n}|{cell}|t{n}|译{n}|high|{note}" for n, (cell, note) in enumerate(zip(cells, notes), start=1)]
    result = _validate(window, lines)
    assert result.ok, result.errors
    return result.segments


def test_the_book_writes_the_windows_answers_as_the_judges_sidecar(tmp_path):
    from finesub.llm.stages.correction.speaker_book import SpeakerBook, foreign_sidecar
    from finesub.speech.diarization import judgments

    stable = _stable(tmp_path)
    (first, second), material = _two_windows(stable)
    book = SpeakerBook(stable, [first, second], material)
    book.record(first, _rows(first, ["s0", "@佐藤"]))
    read = judgments.read_judgments(stable)
    assert read.moved == {"2": 2} and read.new_tracks == (2,), "row 2 opened a new track; row 1 kept s0"
    # the overlap row (2) is the first window's: the second writes rows 3 and 4 only
    book.record(second, _rows(second, ["s1", "s0", "?"]))
    read = judgments.read_judgments(stable)
    assert read.moved == {"2": 2, "3": 0} and read.abstained == {"4"}
    assert foreign_sidecar(stable) is None, "the task's own sidecar"
    payload = json.loads(judgments.sidecar_path(stable).read_text(encoding="utf-8"))
    assert payload["route"] == "single-pass" and {unit["source"] for unit in payload["windows"]} == {"single-pass"}
    assert {unit["media"] for unit in payload["windows"]} == {"text"}
    SpeakerBook(stable, [first, second], material, media="audio").record(first, _rows(first, ["s0", "s1"]))
    payload = json.loads(judgments.sidecar_path(stable).read_text(encoding="utf-8"))
    assert payload["windows"][0]["media"] == "audio", "the windows heard what --llm-media gave them"


def test_the_book_takes_each_moved_rows_note_as_its_reason_and_the_tracks_table_as_the_units_names(tmp_path):
    from finesub.llm.stages.correction.speaker_book import SpeakerBook

    stable = _stable(tmp_path)
    (first, second), material = _two_windows(stable)
    book = SpeakerBook(stable, [first, second], material)
    reply = "<tracks>\ns0 = 健太\n@佐藤 = 佐藤\n</tracks>\n<translated>\n...\n</translated>"
    book.record(first, _rows(first, ["s0", "@佐藤"], ["", "被叫作佐藤的人在回答"]), reply)
    unit = json.loads(book.path.read_text(encoding="utf-8"))["windows"][0]
    assert unit["names"] == {"s0": "健太", "@佐藤": "佐藤"}
    assert [(r["row"], r["label"], r["why"]) for r in unit["results"]] == [("2", "佐藤", "被叫作佐藤的人在回答")]
    carried = book.with_carried(second)
    assert "前面的窗口猜是佐藤" in carried.speakers.profiles, "the next window is shown who the track is"
    book.record(carried, _rows(carried, ["s2", "s2", "s1"]))
    unit = json.loads(book.path.read_text(encoding="utf-8"))["windows"][1]
    assert "names" not in unit, "a reply without a table names nobody"


def test_a_patch_repair_keeps_the_tracks_table_unless_it_sends_a_new_one(tmp_path):
    from finesub.llm.correction_patch import apply_patch, repair_base
    from finesub.llm.stages.speaker_judge.answers import parse_tracks

    variant, window = resolve_variant(TRACKED), _window(_stable(tmp_path))
    rows = ["sub|1|s0|a|甲|high|", "sub|2|s0|b|乙|high|", "sub|3|s1|c|丙|high|", "sub|4||d|丁|high|"]
    answer = "<tracks>\ns0 = 健太\ns1 = 不明\n</tracks>\n<translated>\n" + "\n".join([HEADER, *rows]) + "\n</translated>"
    result = validate_correction_window_output(answer, window, variant=variant)
    assert not result.ok
    base = repair_base(answer, result, window, variant)
    patch = "<patch>\n" + HEADER + "\nsub|4|s1|d|丁|high|\n</patch>"
    merged = apply_patch(base, patch)
    assert validate_correction_window_output(merged.text, window, variant=variant).ok
    assert parse_tracks(merged.text) == {"s0": "健太", "s1": "不明"}
    renamed = apply_patch(base, patch + "\n<tracks>\ns0 = 健太\ns1 = 美香\n</tracks>")
    assert parse_tracks(renamed.text) == {"s0": "健太", "s1": "美香"}


def test_a_task_resumed_on_a_cell_with_no_tracked_set_keeps_the_tracked_set(tmp_path, monkeypatch):
    """Switched to efficiency when the quota ran out, the task follows its record: its windows are still served
    capableC's tracked set (the only one), and the run says so once."""
    from finesub.llm.prompt_variants import for_window
    from finesub.llm.stages import three_pass
    from .test_llm_three_pass import request

    stable = _stable(tmp_path)
    (first, _second), _material = _two_windows(stable)
    assert for_window(resolve_variant("basicA"), first).name == TRACKED
    art = tmp_path / "art"
    art.mkdir()
    three_pass.record_choice(art, three_pass.PassChoice("single-judged", False, "intermediate, judge text", "text"))
    monkeypatch.setattr(three_pass, "tracked_prompts", lambda profile: False)
    warned = []
    choice = three_pass.task_choice(request(tmp_path, stable, profile=resolve_profile("text", "none", "efficiency"),
                                            speaker_judge="text"), art, [], warn=warned.append)
    assert choice.mode == "single-judged" and sum("capableC-tracked" in message for message in warned) == 1


def test_a_committed_answer_naming_a_carried_track_replays_before_the_track_is_carried(tmp_path):
    """A resume's refit, or a serial task resumed in parallel, asks the ledger before any carried track is on the
    window: its `spk` cells passed when committed and are not checked again; the book takes only what it may."""
    from finesub.llm.stages.correction.context import ResumeLedger, _window_input_hash

    stable = _stable(tmp_path)
    (first, _second), _material = _two_windows(stable)
    content = "<translated>\n" + "\n".join([HEADER, "sub|1|s0|a|甲|high|", "sub|2|s7|b|乙|high|"]) + "\n</translated>"
    ledger = ResumeLedger(enabled=True, path=None, task_fingerprint="t", records={first.chunk_id: {
        "input_hash_core": _window_input_hash(first), "content": content, "variant": TRACKED}})
    assert ledger.replay(first) is not None
    bare = "<translated>\n" + "\n".join(["type|position|corrected_text|translation|conf|note", "sub|1|a|甲|high|",
                                          "sub|2|b|乙|high|"]) + "\n</translated>"
    ledger.records[first.chunk_id]["content"] = bare
    assert ledger.replay(first) is None, "a missing `spk` column replays only on a record accepted with it unfixed"
    ledger.records[first.chunk_id]["spk_unchecked"] = True
    assert ledger.replay(first) is not None

def test_a_serial_window_is_handed_the_track_the_one_before_opened(tmp_path):
    from finesub.llm.stages.correction.speaker_book import SpeakerBook
    from finesub.speech.diarization import judgments

    stable = _stable(tmp_path)
    (first, second), material = _two_windows(stable)
    book = SpeakerBook(stable, [first, second], material)
    book.record(first, _rows(first, ["s0", "@佐藤"]))
    carried = book.with_carried(second)
    assert 2 in carried.speakers.profile_tracks and "- s2：" in carried.speakers.profiles
    assert second.speakers.profile_tracks == (0, 1), "the window as planned does not know it"
    book.record(carried, _rows(carried, ["s2", "s2", "s1"]))
    read = judgments.read_judgments(stable)
    assert read.moved == {"2": 2, "3": 2} and read.new_tracks == (2,), "one person, one track across the windows"
    half = replace(second, chunk_id="0002-a")  # split on an earlier run, re-cut on resume
    assert 2 in book.with_carried(half).speakers.profile_tracks, "a half gets its unit's carried tracks"


def test_the_task_report_says_what_the_windows_judged(tmp_path):
    from types import SimpleNamespace

    from finesub.llm.stages.correction.run import _record_speaker_book
    from finesub.llm.stages.correction.speaker_book import SpeakerBook
    from finesub.llm.task_report import write_task_report

    stable = _stable(tmp_path)
    (first, second), material = _two_windows(stable)
    book = SpeakerBook(stable, [first, second], material)
    book.record(first, _rows(first, ["s0", "@佐藤"]))
    artifacts = tmp_path / "artifacts"
    # the first window replayed from an earlier run's ledger, unchecked; the second never reached
    run = SimpleNamespace(speaker_book=book, task_artifact_dir=artifacts, task_id="t", progress=None,
                          ledger=SimpleNamespace(records={"0001": {"spk_unchecked": True}}, committed={}))
    _record_speaker_book(run)
    text = write_task_report(artifacts, task_id="t").read_text(encoding="utf-8")
    line = text.split("## Speaker Judge", 1)[1].strip().splitlines()[0]
    assert line.startswith("- in the correction windows (one pass), 2 window(s): judged 1, failed 0 and pending 1")
    assert "reused" not in line and "1 line(s) moved; new tracks: s2" in line
    assert line.endswith("1 window(s) accepted with the spk column unchecked")
    book.record(second, _rows(second, ["s1", "s0", "?"]))
    run.ledger = SimpleNamespace(records={}, committed={"0001": {}})
    _record_speaker_book(run)
    line = write_task_report(artifacts, task_id="t").read_text(encoding="utf-8").split("## Speaker Judge", 1)[1]
    assert "judged 2, failed 0 and pending 0" in line and "1 line(s) abstained" in line and "unchecked" not in line


def test_a_window_whose_spk_did_not_come_through_is_a_failed_unit(tmp_path):
    from finesub.llm.stages.correction.speaker_book import SpeakerBook

    stable = _stable(tmp_path)
    (first, second), material = _two_windows(stable)
    book = SpeakerBook(stable, [first, second], material)
    loose = validate_correction_window_output(
        "<translated>\n" + "\n".join([HEADER, "sub|1|s9|a|甲|high|", "sub|2|x|b|乙|high|"]) + "\n</translated>",
        first, variant=resolve_variant("capableC"), spk_unchecked=True)
    book.record(first, loose.segments)
    assert book.entries[0]["status"] == "failed" and book.entries[1]["status"] == "pending"


def _long_tracked_window(tmp_path, count: int = 40):
    """``count`` rows on one track, each source's words its own (`test_llm_serial_shift._line`)."""
    from .test_llm_serial_shift import _line

    stable = tmp_path / "long-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"id": str(i), "start": 2.0 * i, "end": 2.0 * i + 1.5, "text": _line(i), "speaker": _speaker(0)}
        for i in range(1, count + 1)]}, ensure_ascii=False), encoding="utf-8")
    return _window(stable)


def test_a_tracked_answer_whose_rows_slid_goes_back_as_a_patch_that_carries_spk(tmp_path):
    """The shift check reads `corrected_text` alone: a `capableC-tracked` answer is checked as the others are, and
    its repair base keeps the `spk` column, so the patch the model sends back carries it."""
    from finesub.llm.correction_patch import apply_patch, repair_base
    from .test_llm_serial_shift import _line, _shifted

    variant, window = resolve_variant(TRACKED), _long_tracked_window(tmp_path)

    def answer(texts):
        rows = [f"sub|{i}|s0|{text}|译文{i}|high|" for i, text in enumerate(texts, start=1)]
        return "<translated>\n" + "\n".join([HEADER, *rows]) + "\n</translated>\n"

    texts = _shifted(40, 20, 6)
    result = validate_correction_window_output(answer(texts), window, variant=variant)
    assert not result.ok and any("从源 20 起" in error for error in result.errors), result.errors
    base = repair_base(answer(texts), result, window, variant)
    assert base is not None and base.header == HEADER and base.pending == tuple(str(i) for i in range(20, 26))
    patch = "<patch>\n" + "\n".join([HEADER, *(f"sub|{i}|s0|{_line(i)}|译文{i}|high|" for i in range(20, 26))])
    merged = apply_patch(base, patch + "\n</patch>")
    assert merged.mode == "patch" and validate_correction_window_output(merged.text, window, variant=variant).ok


def test_a_window_accepted_with_spk_unfixed_records_its_translation_shift_in_source_ids(tmp_path, monkeypatch):
    """A single-judged window out of repairs with only `spk` wrong is accepted (`_accept_unfixed_speakers`) through
    the loose check, which still finds translations that slid; the record names them by stable id, not position."""
    from finesub.llm.client import LLMCallResult
    from finesub.llm.routing.config import LLMRole
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages.correction import execute_correction_windows
    from .conftest import setattr_correction
    from .support.fakes import FakeTokenCounter
    from .test_llm_translation_shift import _slid, _source

    count = 30
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"metadata": {}, "segments": [
        {"id": str(100 + i), "start": i * 2.0, "end": i * 2.0 + 1.0, "text": _source(i),
         "speaker": _speaker(i % 2)} for i in range(1, count + 1)]}, ensure_ascii=False), encoding="utf-8")
    translations = _slid(count, 6, 10)
    answer = "<translated>\n" + "\n".join(
        [HEADER, *(f"sub|{i}|s9|{_source(i)}|{translations[i - 1]}|high|" for i in range(1, count + 1))]
    ) + "\n</translated>\n"

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(content=answer, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                                 raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
                               max_retries_per_window=1, max_replacements_per_window=0,
                               profile=resolve_profile("text", "none", "intermediate"),
                               task_artifact_dir=tmp_path / "artifacts", length_scale=1.0, single_judged=True)
    ledger = [json.loads(line) for line in (tmp_path / "artifacts" / "correction-windows.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    record = ledger[-1]
    assert record["spk_unchecked"] is True and record["variant"] == TRACKED
    assert (record["translation_shift"]["start"], record["translation_shift"]["end"]) == ("106", "115")


# --- The correct-only derivation (a Chinese source: corrected_text and polished, no translation) ---

TRACKED_POLISH = "capableC-tracked-polish"
POLISH_HEADER = "type|position|spk|corrected_text|polished|conf|note"


def _zh_stable(tmp_path):
    """Four rows: s0, s0, s1, s1; the last only filler."""
    rows = [(_speaker(0), "然后昨天直播的时候"), (_speaker(0), "被那个boss打败了"), (_speaker(1), "啊真的假的"),
            (_speaker(1), "呃")]
    stable = tmp_path / "zh-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"id": str(i), "start": 2.0 * i, "end": 2.0 * i + 1.5, "text": text, "speaker": sp}
        for i, (sp, text) in enumerate(rows, start=1)]}, ensure_ascii=False), encoding="utf-8")
    return stable


def test_the_correct_only_set_names_who_speaks_and_resolves_by_name():
    from finesub.llm.prompt_variants import has_tracked_variant, tracked_variant

    polish = resolve_variant(TRACKED_POLISH)
    assert polish.tracked and polish.correct_only and polish.examples_name == "capableC"
    assert has_tracked_variant("capableC-polish")
    assert tracked_variant(resolve_variant("capableC-polish")) is polish
    # a set with no tracked version falls back in its own mode, never to the translating set
    assert tracked_variant(resolve_variant("basicA-polish")) is polish
    assert tracked_variant(resolve_variant("basicA")).name == TRACKED


def test_the_correct_only_set_shares_the_tracked_prompt_version(tmp_path):
    """The version is the window's (it carries speaker material), whichever mode: the translating render did not
    change, and no correct-only tracked window was committed before."""
    window = _window(_zh_stable(tmp_path))
    assert window_prompt_version(window) == "zh-subtitle-correction-tracked-v3"


def test_the_correct_only_role_and_work_words_come_from_the_mode(tmp_path):
    system = _messages(_window(_zh_stable(tmp_path)), TRACKED_POLISH)[0]["content"]
    assert "ASR 纠错、说话人复核与中文字幕模型" in system and "翻译" not in system
    assert "`spk` 和纠错、局部编辑一样是要交付的结果" in system and POLISH_HEADER in system


def test_polished_and_spk_validate_together_and_a_filler_row_keeps_its_speaker(tmp_path):
    from finesub.llm.stages.correction.speaker_book import SpeakerBook
    from finesub.speech.diarization import judgments

    stable = _zh_stable(tmp_path)
    window = _window(stable)
    rows = ["sub|1,2|s0|然后昨天直播的时候，被那个boss打败了|然后昨天直播的时候，被那个boss打败了|high|",
            "sub|3|@小明|啊，真的假的？|啊，真的假的？|high|", "sub|4|@小明|呃||high|整行只有垫词"]
    text = "<translated>\n" + "\n".join([POLISH_HEADER, *rows]) + "\n</translated>"
    result = validate_correction_window_output(text, window, variant=resolve_variant(TRACKED_POLISH))
    assert result.ok, result.errors
    assert [(segment.spk, segment.translation) for segment in result.segments][-1] == ("@小明", "")
    material = speaker_material(stable, ContextPack(), knowledge_root=None, title="测试直播")
    SpeakerBook(stable, [window], material).record(window, result.segments)
    read = judgments.read_judgments(stable)
    assert read.moved == {"3": 2, "4": 2}, "the filler row, out of the subtitle, is still judged"
    translated = text.replace(POLISH_HEADER, HEADER)
    assert not validate_correction_window_output(translated, window, variant=resolve_variant(TRACKED_POLISH)).ok


def test_a_correct_only_task_with_the_judge_folds_it_at_intermediate(tmp_path):
    from finesub.llm.stages import three_pass
    from finesub.subtitles.stable_rows import read_rows
    from .test_llm_three_pass import request

    stable = _zh_stable(tmp_path)
    rows = read_rows(stable)
    profile = resolve_profile("text", "none", "intermediate", correct_only=True)
    warned = []
    choice = three_pass.task_choice(request(tmp_path, stable, profile=profile, speaker_judge="text"), None, rows,
                                    warn=warned.append)
    assert choice.mode == "single-judged" and choice.correct_only and not choice.inline_proofread
    assert len(warned) == 1 and "不如三个 pass" in warned[0] and "--llm-difficulty quality" in warned[0]
    quality = resolve_profile("text", "none", "quality", correct_only=True)
    warned.clear()
    assert three_pass.task_choice(request(tmp_path, stable, profile=quality, speaker_judge="text"), None,
                                  rows, warn=warned.append).mode == "three-pass"
    assert warned == []
    # a Chinese task recorded as three passes goes on with three passes
    art = tmp_path / "art"
    art.mkdir()
    three_pass.record_choice(art, three_pass.PassChoice("three-pass", False, "recorded", "text", correct_only=True,
                                                        source_language="zh"))
    assert three_pass.task_choice(request(tmp_path, stable, profile=profile, speaker_judge="text"), art,
                                  rows).mode == "three-pass"
    # resumed as single-judged, it is told again; a translating task is not
    art2 = tmp_path / "art2"
    art2.mkdir()
    three_pass.record_choice(art2, three_pass.PassChoice("single-judged", False, "recorded", "text",
                                                         correct_only=True, source_language="zh"))
    three_pass.task_choice(request(tmp_path, stable, profile=profile, speaker_judge="text"), art2, rows,
                           warn=warned.append)
    assert sum("不如三个 pass" in message for message in warned) == 1
    warned.clear()
    translating = resolve_profile("text", "none", "intermediate")
    assert three_pass.task_choice(request(tmp_path, stable, profile=translating, speaker_judge="text"), None, rows,
                                  warn=warned.append).mode == "single-judged"
    assert not any("不如三个 pass" in message for message in warned)


def test_a_correct_only_task_resumed_without_a_tracked_set_names_the_correct_only_fallback(tmp_path, monkeypatch):
    from finesub.llm.stages import three_pass
    from .test_llm_three_pass import request

    stable = _zh_stable(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    three_pass.record_choice(art, three_pass.PassChoice("single-judged", False, "recorded", "text", correct_only=True,
                                                        source_language="zh"))
    monkeypatch.setattr(three_pass, "tracked_prompts", lambda profile: False)
    warned = []
    efficiency = resolve_profile("text", "none", "efficiency", correct_only=True)
    three_pass.task_choice(request(tmp_path, stable, profile=efficiency, speaker_judge="text"), art, [],
                           warn=warned.append)
    assert sum(TRACKED_POLISH in message for message in warned) == 1


def test_the_worked_examples_label_the_same_rows_in_both_languages():
    """One track table serves both example directories (`example_tracks_v1.toml`): each example it names has the
    same row labels in the Japanese and the Chinese material, and every label it gives a track exists there."""
    import tomllib

    from finesub.llm.example_builder import EXAMPLE_TRACKS_FILE, EXAMPLES_DIR, EXAMPLES_ZH_DIR, load_examples
    from finesub.llm.prompt_tables import PROMPT_TEMPLATE_DIR

    table = tomllib.loads((PROMPT_TEMPLATE_DIR / EXAMPLE_TRACKS_FILE).read_text(encoding="utf-8"))
    by_dir = [{example.id: example for example in load_examples(directory)} for directory in (EXAMPLES_DIR,
                                                                                               EXAMPLES_ZH_DIR)]
    for example_id, cells in table.items():
        labels = [[row.label for block in examples[example_id].blocks for row in block.inputs] for examples in by_dir]
        assert labels[0] == labels[1], example_id
        assert set(cells) <= set(labels[0]), example_id
