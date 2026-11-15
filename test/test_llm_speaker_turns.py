"""The correction's speaker turns: where a line's speaker differs from the line before it, the correction
must not merge the two (`chunking.SubtitleSegment.seam`, docs/llm_harness_behavior.md 「Prompt 信息」)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace

from finesub.llm.agent.agent_validators import build_validator, deserialize_window, serialize_window
from finesub.llm.chunking import (
    Seam,
    SubtitleSegment,
    load_segments_from_stable_json,
    plan_correction_windows,
    render_window_preceding_as_csv,
    render_window_segments_as_csv,
    product_windows,
)
from finesub.llm.client import LLMCallResult
from finesub.llm.output_protocol import validate_correction_window_output
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.routing.config import CapabilityTier, LLMRole
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.stages.correction.commit import _window_input_hash
from finesub.reporting import reporting_to
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization.labels import SegmentSpeaker, different_speakers, speaker_of
from finesub.speech.diarization.product import WHOLE, Line, Product
from finesub.subtitles.stable_rows import overlay_for, write_overlay

from .conftest import setattr_correction

HEADER = "type|position|corrected_text|translation|conf|note"


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def speaker(track, block=0, **extra):
    return {"track": track, "tracks": [track], "block": block, "coverage": {str(track): 0.9}, "labels": [], **extra}


def labelled_stable(tmp_path, speakers, texts=None):
    """A stable whose lines carry `speakers` (None: no track) through a rows file, as the diarization stage writes."""
    lines = [{"id": str(index), "start": index * 2.0, "end": index * 2.0 + 1.0,
              "text": (texts[index - 1] if texts else f"line {index}")}
             for index in range(1, len(speakers) + 1)]
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"metadata": {}, "segments": lines}), encoding="utf-8")
    write_overlay(stable, overlay_for(lines, [{"speaker": s} if s is not None else {} for s in speakers],
                                      speaker={"marks": None}))
    return stable


def changed(segment):
    """The seam before a line as the old two-way turn: None without labels."""
    return None if segment.seam is None else segment.seam is Seam.CHANGE


def turns(stable):
    return [changed(segment) for segment in load_segments_from_stable_json(stable)]


def test_two_lines_are_two_people_when_their_main_tracks_differ_in_one_block():
    one, other = SegmentSpeaker(track=1), SegmentSpeaker(track=2)
    assert different_speakers(one, other)
    assert not different_speakers(one, SegmentSpeaker(track=1, review=True))
    # A `?` does not soften it: the rule reads the main track only.
    assert different_speakers(one, SegmentSpeaker(track=2, review=True))
    assert not different_speakers(one, SegmentSpeaker())
    assert not different_speakers(SegmentSpeaker(), other)
    # Track ids are only comparable within a block.
    assert not different_speakers(one, SegmentSpeaker(track=2, block=1))


def test_the_loader_marks_each_line_whose_speaker_differs_from_the_line_before(tmp_path):
    stable = labelled_stable(tmp_path, [speaker(1), speaker(1), speaker(2), None, speaker(1), speaker(3, block=1)])
    assert turns(stable) == [False, False, True, False, False, False]


def test_a_task_without_labels_has_no_turns_at_all(tmp_path):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "a"},
                                               {"id": "2", "start": 2.0, "end": 3.0, "text": "b"}]}),
                      encoding="utf-8")
    assert turns(stable) == [None, None]


def test_adjacent_means_adjacent_among_the_lines_the_loader_keeps(tmp_path):
    # Line 2 has no text, so the loader drops it and lines 1 and 3 are neighbours.
    stable = labelled_stable(tmp_path, [speaker(1), speaker(2), speaker(1)], texts=["a", "", "c"])
    segments = load_segments_from_stable_json(stable)
    assert [(segment.id, changed(segment)) for segment in segments] == [("1", False), ("3", False)]


def judge(stable, results):
    """A sidecar whose single window holds `results`, for the rows the stable reads now."""
    from finesub.subtitles.stable_rows import read_rows

    segments = read_rows(stable)
    tracks = jd.profile_tracks([speaker_of(s) for s in segments])
    window = {"id": "0001", "first": "1", "last": str(len(segments)), "status": "success",
              "key": jd.window_key(segments, jd.segment_ids(segments), tracks), "results": results}
    jd.sidecar_path(stable).write_text(json.dumps(
        {"version": jd.SIDECAR_VERSION, "route": "separate", "prompt_version": "t",
         "rows_hash": jd.rows_hash(segments), "windows": [window]}), encoding="utf-8")


def test_turns_follow_the_judges_verdicts(tmp_path):
    stable = labelled_stable(tmp_path, [speaker(1), speaker(2, review=True), speaker(1)])
    assert turns(stable) == [False, True, True]
    judge(stable, [{"row": "2", "track": 1, "from": 2}])
    assert turns(stable) == [False, False, False]


def window_of(segments):
    return dataclasses.replace(plan_correction_windows(segments, counter=FakeTokenCounter())[0], length_scale=1.0)


SEGMENTS = [
    SubtitleSegment("1", 0.0, 1.0, "一。", seam=Seam.NONE),
    SubtitleSegment("2", 1.5, 2.5, "二。", seam=Seam.CHANGE),
    SubtitleSegment("3", 3.0, 4.0, "三。", seam=Seam.NONE),
]


def test_only_the_window_input_carries_the_turn_column():
    window = window_of(SEGMENTS)
    assert render_window_segments_as_csv(window).splitlines() == [
        "1|0.0|1.0|||一。", "2|1.5|1.0|0.5|换人|二。", "3|3.0|1.0|0.5||三。"]
    plain = window_of([replace(segment, seam=None) for segment in SEGMENTS])
    assert render_window_segments_as_csv(plain).splitlines()[1] == "2|1.5|1.0|0.5|二。"
    preceding = replace(window, preceding_segments=[SubtitleSegment("9", -2.0, -1.0, "前。", seam=Seam.CHANGE)])
    assert "换人" not in render_window_preceding_as_csv(preceding)


def test_turns_restamped_after_the_judge_reach_every_copy_of_a_line():
    window = window_of([replace(segment, seam=Seam.NONE) for segment in SEGMENTS])
    window = replace(window, preceding_segments=[SubtitleSegment("2", 1.5, 2.5, "二。", seam=Seam.NONE)])
    layer = Product(tuple(Line({"id": segment.id}, segment.id, WHOLE) for segment in SEGMENTS))
    (stamped,) = product_windows([window], SEGMENTS, layer)
    assert [changed(segment) for segment in stamped.segments] == [False, True, False]
    assert stamped.preceding_segments[0].seam is Seam.CHANGE


def test_a_changed_turn_changes_the_window_cache_identity():
    window = window_of(SEGMENTS)
    other = window_of([replace(segment, seam=Seam.NONE) for segment in SEGMENTS])
    assert _window_input_hash(window) != _window_input_hash(other)


CAPABLE_C = resolve_variant("capableC", CapabilityTier.CAPABLE)


def reply(*rows):
    return "<translated>\n" + HEADER + "\n" + "\n".join(rows) + "\n</translated>"


def grouped_reply(*rows):
    """An answer in the six columns three passes' pass 2 writes (`capableC-grouped`), as capableC does."""
    return reply(*rows)


def test_turn_notes_speak_the_model_s_local_ids_and_the_rows_the_source_ids():
    segments = [SubtitleSegment(str(n), n * 2.0, n * 2.0 + 1.0, f"{n}。", seam=Seam.CHANGE if n == 12 else Seam.NONE) for n in (11, 12, 13)]
    result = validate_correction_window_output(
        reply("sub|1,2|十一。十二。|十一十二|high|", "sub|3|十三。|十三|high|"),
        window_of(segments), variant=CAPABLE_C)
    assert result.turn_violation_rows == (("11", "12"),)
    assert result.turn_notes == ("顺带：position 为 1,2 的合并行里，源 2 的 turn 列标了换人，拆回各自的行",)


def test_a_merge_across_a_turn_is_reported_apart_from_the_errors():
    window = window_of(SEGMENTS)
    across = validate_correction_window_output(
        reply("sub|1,2|一。二。|一二|high|", "sub|3|三。|三|high|"), window, variant=CAPABLE_C)
    assert across.ok and not across.errors
    assert across.turn_violation_rows == (("1", "2"),)
    within = validate_correction_window_output(
        reply("sub|1|一。|一|high|", "sub|2,3|二。三。|二三|high|"), window, variant=CAPABLE_C)
    assert within.ok and within.turn_violation_rows == ()


def test_the_agent_side_sees_the_turns_and_accepts_a_merge_across_one():
    window = window_of(SEGMENTS)
    payload = json.loads(json.dumps(serialize_window(window)))
    assert [changed(segment) for segment in deserialize_window(payload).segments] == [False, True, False]
    validate = build_validator("correction-window", {"metadata": {"validator": {"window": payload},
                                                                  "variant": "capableC"}})
    result = validate(reply("sub|1,2|一。二。|一二|high|", "sub|3|三。|三|high|"), {})
    assert result.status == "accepted" and not result.errors
    broken = validate(reply("sub|1,2|一。二。|一二|high|", "sub|3|三。|三|high|x|y"), {})
    assert broken.status == "repairable"
    assert "顺带：position 为 1,2 的合并行里，源 2 的 turn 列标了换人，拆回各自的行" in broken.errors


def test_a_merge_across_a_turn_is_accepted_with_a_warning_and_never_sent_back(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1), speaker(2)], texts=["一。", "二。"])
    merged = reply("sub|1,2|一。二。|一二|high|")
    calls: list[list[str]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            calls.append(list(kwargs.get("validation_errors", ())))
            assert "local_id|start|duration|gap|turn|text" in messages[-1]["content"]
            return LLMCallResult(content=merged, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                                 raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    warnings: list[str] = []

    class Reporter:
        def warning(self, code, message, **_kwargs):
            warnings.append(code)

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    with reporting_to(Reporter()):
        execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt",
                                   token_counter=FakeTokenCounter(), max_retries_per_window=1,
                                   max_replacements_per_window=0, profile=resolve_profile("text", "none", "quality"),
                                   task_artifact_dir=tmp_path / "artifacts", length_scale=1.0)
    assert calls == [[]]
    assert "correction-turn-merge" in warnings
    ledger = [json.loads(line) for line in (tmp_path / "artifacts" / "correction-windows.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert ledger[-1]["turn_violations"] == [["1", "2"]]


def test_a_window_sent_back_for_a_structural_error_also_splits_its_merges_across_turns(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1), speaker(2), speaker(2)], texts=["一。", "二。", "三。"])
    broken = reply("sub|1,2|一。二。|一二|high|", "sub|3|三。|三|high|x|y")
    fixed = reply("sub|1|一。|一|high|", "sub|2|二。|二|high|", "sub|3|三。|三|high|")
    calls: list[list[str]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            calls.append(list(kwargs.get("validation_errors", ())))
            return LLMCallResult(content=broken if len(calls) == 1 else fixed, role=LLMRole.AUDIO_MULTIMODAL,
                                 model="fake", fallback_used=False,
                                 raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt",
                               token_counter=FakeTokenCounter(), max_retries_per_window=1,
                               max_replacements_per_window=0, profile=resolve_profile("text", "none", "quality"),
                               task_artifact_dir=tmp_path / "artifacts", length_scale=1.0)
    assert len(calls) == 2 and calls[0] == []
    assert any("7 cells" in error for error in calls[1])
    assert "顺带：position 为 1,2 的合并行里，源 2 的 turn 列标了换人，拆回各自的行" in calls[1]
    ledger = [json.loads(line) for line in (tmp_path / "artifacts" / "correction-windows.jsonl")
              .read_text(encoding="utf-8").splitlines()]
    assert ledger[-1]["turn_violations"] == []


def test_a_merge_skipping_lines_is_checked_against_every_line_it_spans():
    from finesub.llm.output_protocol import TranslatedCsvSegment, _turn_violation_rows

    def row(*ids):
        return TranslatedCsvSegment(source_ids=ids, start=0.0, end=1.0, corrected_text="x", translation="x")

    # Line 2 starts another speaker and line 3 carries on with that one: joining 1 and 3 still crosses the turn.
    assert _turn_violation_rows([row("1", "3")], SEGMENTS) == (("1", "3"),)
    assert _turn_violation_rows([row("2", "3")], SEGMENTS) == ()


def test_parallel_windows_with_merges_across_turns_are_accepted_on_their_first_answer(tmp_path, monkeypatch):
    """No retry, no replacement chain, nothing for the parallel breaker to count."""
    segments = [SubtitleSegment(str(index), index * 2.0, index * 2.0 + 1.0, f"第{index}段。", seam=Seam.CHANGE if index > 1 else Seam.NONE)
                for index in range(1, 9)]
    windows = [replace(window_of(segments[low:low + 2]), chunk_id=f"{low // 2 + 1:04d}") for low in range(0, 8, 2)]
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": s.id, "start": s.start, "end": s.end, "text": s.text}
                                               for s in segments]}), encoding="utf-8")
    calls: list[str] = []

    class MergingClient:
        def complete(self, role, messages, **kwargs):
            prompt = (messages("capableC") if callable(messages) else messages)[-1]["content"]
            block = prompt.split("<asr_result>")[-1].split("</asr_result>")[0]
            count = sum(1 for line in block.splitlines() if line[:1].isdigit())
            calls.append(prompt)
            rows = ["sub|1,2|合。|合|high|", *(f"sub|{n}|单。|单|high|" for n in range(3, count + 1))]
            return LLMCallResult(content=reply(*rows), role=role, model="fake", fallback_used=False,
                                 raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", lambda **_: MergingClient())
    execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
                               profile=resolve_profile("text", "none", "quality", "parallel"),
                               knowledge_root=tmp_path / "kb", knowledge_enabled=False, windows_override=windows,
                               max_retries_per_window=0, max_replacements_per_window=1, parallel_window_limit=1, length_scale=1.0)
    assert len(calls) == len(windows)
