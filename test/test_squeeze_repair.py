"""Squeezed rows stretched into the VAD speech after them (docs/asr-stabilize.md, "坍缩行" under "补救：坍缩行与空档").

The geometry, run as stabilize runs a recall repair (`stabilize_payload(repair=...)`) with every row the plain
repair would stretch let through (`_geometric`): which of them production stretches is the referee's call
(`postprocessing/recall_repair`), tested there."""

from __future__ import annotations

from finesub.speech.postprocessing import squeeze_repair
from finesub.speech.postprocessing import stabilization as asr_stabilize
from finesub.speech.postprocessing.segmentation import ANCHOR_UNCERTAIN_TAG, WHISPER_SEGMENT_WORD_TAG, split_params_of


def _word(text: str, start: float, end: float, **extra: object) -> dict[str, object]:
    return {"start": start, "end": end, "word": text, "space_before": False, "confidence": 0.9, **extra}


def _row(tokens: list[str], start: float, step: float, **extra: object) -> dict[str, object]:
    words = [_word(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(tokens)]
    words[0][WHISPER_SEGMENT_WORD_TAG] = True
    row = {
        "start": start,
        "end": words[-1]["end"],
        "text": "".join(tokens),
        "words": words,
        "confidence": 0.9,
        "vad_weighted_energy_db": 5.0,
    }
    row.update(extra)
    return row


def _payload(*rows: dict[str, object], speech: list[tuple[float, float]]) -> dict[str, object]:
    return {
        "segments": list(rows),
        "vad_timeline": {"intervals": [{"start": a, "end": b} for a, b in speech]},
        "metadata": {"asr_align": {"model": "test"}},
    }


#: 11.5 weighted characters in 0.24 s -- 48 per second, far past the criterion.
SQUEEZED = ["させて", "いただき", "ましょう", "それでは"]
NEED = 15.0 / squeeze_repair.TARGET_CHARS_PER_SEC   # weighted chars of SQUEEZED / target rate


def _geometric(payload, rows):
    """Every row the plain repair would stretch, stretched."""
    intervals = (payload.get("vad_timeline") or {}).get("intervals") or []
    rows, n = squeeze_repair.stretch_squeezed(rows, intervals, leave=asr_stabilize.is_closing_shape,
                                              params=split_params_of(payload))
    return rows, {"stretched": n}


def _stabilize(payload, **kwargs):
    return asr_stabilize.stabilize_payload(payload, repair=_geometric, **kwargs)


def _stretched(out) -> int:
    return out["metadata"]["asr_stabilize"].get("recall", {}).get("stretched", 0)


def _stable(payload):
    return _stabilize(payload)[0]


def test_a_squeezed_row_grows_its_end_into_the_speech_after_it() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    out = _stable(_payload(row, _row(["次です"], 20.0, 1.0), speech=[(10.0, 15.0), (20.0, 21.0)]))
    got = out["segments"][0]
    assert got["start"] == 10.0
    assert abs(got["end"] - (10.0 + NEED)) < 1e-6
    # The span is still its words' span: the timeline contract has nothing to bounce back.
    assert got["words"][0]["start"] == got["start"] and got["words"][-1]["end"] == got["end"]
    assert all(w["start"] < w["end"] for w in got["words"])
    assert got[squeeze_repair.SOURCE_FIELD] == {"start": 10.0, "end": 10.24}
    assert "segment_time_repairs" not in out["metadata"].get("asr_stabilize", {})


def test_rows_outside_the_criterion_are_untouched() -> None:
    slow = _row(["ゆっくり", "話して", "います"], 10.0, 0.5)
    other = _row(["次です"], 20.0, 1.0)
    payload = _payload(slow, other, speech=[(10.0, 15.0), (20.0, 21.0)])
    assert _stable(payload)["segments"] == _stabilize(dict(payload, vad_timeline={"intervals": []}))[0]["segments"]


def test_the_end_stops_at_the_next_row() -> None:
    out = _stable(_payload(_row(SQUEEZED, 10.0, 0.06), _row(["次です"], 10.8, 0.4), speech=[(10.0, 15.0)]))
    assert out["segments"][0]["end"] == 10.8
    assert out["segments"][1]["start"] == 10.8


def test_no_unclaimed_speech_after_the_row_changes_nothing() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    payload = _payload(row, _row(["次です"], 20.0, 1.0), speech=[(9.0, 10.24), (20.0, 21.0)])
    out = _stable(payload)
    assert {k: v for k, v in out["segments"][0].items() if k != "tags"} == row


def test_a_row_whose_own_span_holds_no_speech_is_not_stretched() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    payload = _payload(row, _row(["次です"], 20.0, 1.0), speech=[(10.5, 15.0), (20.0, 21.0)])
    assert _stable(payload)["segments"][0]["end"] == 10.24


def test_the_end_does_not_cross_a_long_silence() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    gap = squeeze_repair.MAX_SILENCE_BRIDGE_SEC + 0.05
    payload = _payload(row, _row(["次です"], 20.0, 1.0), speech=[(10.0, 10.24), (10.24 + gap, 15.0), (20.0, 21.0)])
    assert _stable(payload)["segments"][0]["end"] == 10.24


def test_the_silence_is_measured_from_the_row_own_last_speech() -> None:
    # Within the cap from the row's end, past it from where its speech stopped.
    row = _row(SQUEEZED, 10.0, 0.06)
    after = 10.24 + squeeze_repair.MAX_SILENCE_BRIDGE_SEC - 0.1
    payload = _payload(row, _row(["次です"], 20.0, 1.0), speech=[(9.9, 10.05), (after, 15.0), (20.0, 21.0)])
    assert _stable(payload)["segments"][0]["end"] == 10.24


def test_pauses_under_the_cap_are_bridged() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    speech = [(10.0, 10.24), (10.8, 11.0), (11.6, 11.8), (12.4, 15.0)]   # 0.56 / 0.6 / 0.6 s pauses
    got = _stable(_payload(row, _row(["次です"], 20.0, 1.0), speech=speech + [(20.0, 21.0)]))["segments"][0]
    assert abs(got["end"] - (12.4 + NEED - 0.64)) < 1e-6


def test_the_grown_part_stops_before_a_long_pause_and_no_word_spans_it() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    gap = squeeze_repair.MAX_SILENCE_BRIDGE_SEC + 0.05
    speech = [(10.0, 10.24), (10.8, 11.0), (11.0 + gap, 15.0)]   # reached speech, then a pause past the cap
    out = _stable(_payload(row, _row(["次です"], 20.0, 1.0), speech=speech + [(20.0, 21.0)]))["segments"]
    assert len(out) == 2   # nothing grown past the pause to cut off as a piece of its own
    assert out[0]["end"] == 11.0 and all(w["end"] <= 11.0 for w in out[0]["words"])


def test_a_row_an_earlier_row_runs_past_is_not_stretched() -> None:
    """Partly covered: the speech after the earlier row's end is unclaimed, but
    stretching would deepen the overlap the two already have."""

    earlier = _row(["ずっと", "話して", "いる", "行です"], 5.0, 1.75)   # 5.0 - 12.0
    row = _row(SQUEEZED, 10.0, 0.06)
    rows, stretched = squeeze_repair.stretch_squeezed(
        [earlier, row], [{"start": 5.0, "end": 15.0}], leave=lambda text: False
    )
    assert rows == [earlier, row] and stretched == 0


def test_speech_an_earlier_row_already_covers_is_not_unclaimed() -> None:
    long_row = _row(["ずっと", "話して", "いる", "長い", "行です"], 5.0, 2.0)   # 5.0 - 15.0
    rows, _ = squeeze_repair.stretch_squeezed(
        [long_row, _row(SQUEEZED, 10.0, 0.06)], [{"start": 5.0, "end": 15.0}], leave=lambda text: False
    )
    assert rows[1]["end"] == 10.24


def test_a_row_already_overlapping_the_next_is_not_stretched_further() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    nxt = _row(["次です"], 10.2, 0.5)
    rows, stretched = squeeze_repair.stretch_squeezed(
        [row, nxt], [{"start": 10.0, "end": 15.0}], leave=lambda text: False
    )
    assert rows == [row, nxt] and stretched == 0


def test_stock_phrase_shapes_are_left_alone() -> None:
    for tokens in (["ご視聴", "ありがと"], ["ありがとう", "ございました"]):
        row = _row(tokens, 10.0, 0.05)
        assert asr_stabilize.is_closing_shape(row["text"])
        rows, stretched = squeeze_repair.stretch_squeezed(
            [row], [{"start": 10.0, "end": 15.0}], leave=asr_stabilize.is_closing_shape
        )
        assert rows == [row] and stretched == 0


def test_a_row_echoing_a_nearby_row_is_left_alone() -> None:
    copy = _row(["濃い", "感じ", "にね"], 10.0, 0.05)
    source = _row(["濃い", "感じ", "にやろう", "ぜ"], 4.0, 0.5)
    rows, stretched = squeeze_repair.stretch_squeezed(
        [source, copy], [{"start": 4.0, "end": 15.0}], leave=lambda text: False
    )
    assert stretched == 0


def test_an_empty_text_row_nearby_is_no_echo_source() -> None:
    blank = {"start": 8.0, "end": 9.0, "text": "…", "words": [_word("…", 8.0, 9.0)]}
    rows, stretched = squeeze_repair.stretch_squeezed(
        [blank, _row(SQUEEZED, 10.0, 0.06)], [{"start": 10.0, "end": 15.0}], leave=lambda text: False
    )
    assert stretched == 1


def test_squeezed_coordinates_are_not_restored_from_raw_times() -> None:
    row = _row(SQUEEZED, 10.0, 0.06)
    row["words"][1].update(raw_start=10.0, raw_end=10.01, split_adjust_case="case3")
    got = _stable(_payload(row, speech=[(10.0, 15.0)]))["segments"][0]
    assert not any("raw_start" in w or "split_adjust_case" in w for w in got["words"])
    assert got["words"][1]["start"] > 10.06


def _two_lines() -> dict[str, object]:
    """Two sentences crushed into 0.28 s; their speech has a pause between."""

    tokens = ["させて", "いただき", "ましょう。", "それでは", "次に", "行き", "ましょう"]
    row = _row(tokens, 10.0, 0.04, tags=["custom"], qwen_verify={"text": "x"},
               alignment_events=[{"type": "alignment_stack", "time": 10.05}])
    before = _row(["前の", "行です"], 5.0, 1.0)
    after = _row(["次", "です"], 20.0, 0.5)
    # The pause must be long enough for the splitter to cut here (about 0.66 s)
    # and short enough for the stretch to bridge it.
    assert 12.1 - 11.4375 <= squeeze_repair.MAX_SILENCE_BRIDGE_SEC
    return _payload(before, row, after, speech=[(5.0, 7.0), (10.0, 11.4375), (12.1, 14.1), (20.0, 21.0)])


def test_a_stretched_row_is_split_locally_and_traceably() -> None:
    payload = _two_lines()
    out = _stable(payload)
    before, first, second, after = out["segments"]
    assert before == payload["segments"][0] and after == payload["segments"][2]
    assert first["text"] == "させていただきましょう。" and second["text"] == "それでは次に行きましょう"
    assert first["start"] == 10.0 and first["end"] <= second["start"]
    for piece in (first, second):
        assert piece["start"] < piece["end"]
        assert piece["words"][0]["start"] >= piece["start"] and piece["words"][-1]["end"] <= piece["end"]
        assert "custom" in piece["tags"] and piece["qwen_verify"] == {"text": "x"}
    assert first["words"][0].get(WHISPER_SEGMENT_WORD_TAG) is True
    assert first[squeeze_repair.SOURCE_FIELD] == second[squeeze_repair.SOURCE_FIELD] == {"start": 10.0, "end": 10.28}
    events = [e for piece in (first, second) for e in piece.get("alignment_events") or []]
    assert events == payload["segments"][1]["alignment_events"]


def test_the_first_piece_keeps_a_mid_segment_row_unmarked() -> None:
    payload = _two_lines()
    payload["segments"][1]["words"][0].pop(WHISPER_SEGMENT_WORD_TAG)
    first = _stable(payload)["segments"][1]
    assert WHISPER_SEGMENT_WORD_TAG not in first["words"][0]


def test_the_row_seam_tag_stays_on_the_first_piece() -> None:
    payload = _two_lines()
    payload["segments"][1]["tags"] = ["custom", ANCHOR_UNCERTAIN_TAG]
    _before, first, second, _after = _stable(payload)["segments"]
    assert ANCHOR_UNCERTAIN_TAG in first["tags"] and ANCHOR_UNCERTAIN_TAG not in second["tags"]
    assert "custom" in second["tags"]


def test_a_second_stabilize_is_a_no_op() -> None:
    once = _stable(_two_lines())
    assert _stable(once)["segments"] == once["segments"]


def test_a_row_echoing_a_cut_row_stays_protected_on_a_second_stabilize() -> None:
    # Neither piece covers the copy alone; the row they came from does.
    row = _row(["させて", "いただき", "ましょう。", "それでは", "次に", "行き", "ましょう"], 10.0, 0.04)
    copy = _row(["ましょう", "それでは"], 16.0, 0.05)
    once = _stable(_payload(row, copy, speech=[(10.0, 11.4375), (12.1, 14.1), (16.0, 19.0)]))
    assert [s["text"] for s in once["segments"]] == ["させていただきましょう。", "それでは次に行きましょう", "ましょうそれでは"]
    assert once["segments"][2]["end"] == 16.1
    assert _stable(once)["segments"] == once["segments"]


def test_a_cut_row_keeps_its_start_and_its_echo_radius_on_a_second_stabilize() -> None:
    # The row's own speech starts 0.05 s late, so its first piece's first word
    # does too; the copy sits at the edge of the radius from the row's start.
    row = _row(["させて", "いただき", "ましょう。", "それでは", "次に", "行き", "ましょう"], 10.0, 0.04)
    copy = _row(["ましょう", "それでは"], 0.0, 0.05)
    once = _stable(_payload(copy, row, speech=[(0.0, 3.0), (10.05, 11.4875), (12.15, 14.15)]))
    copy_once, first, second = once["segments"]
    assert copy_once["end"] == 0.1 and first["start"] == 10.0 and second["start"] > first["end"] - 1e-9
    assert _stable(once)["segments"] == once["segments"]


def test_the_cut_follows_the_recorded_length_scale() -> None:
    tokens = ["今日は", "いい", "天気", "ですね", "それでは", "始めましょう", "最初の", "コーナーは",
              "こちら", "です", "お便りを", "読みます", "ラジオネーム", "ジュンさん", "から", "いただきました"]

    def pieces(scale):
        payload = _payload(_row(tokens, 10.0, 0.015), _row(["次です"], 30.0, 1.0), speech=[(10.0, 25.0), (30.0, 31.0)])
        if scale is not None:
            payload["metadata"]["asr_align"]["segment_split"] = {"length_scale": scale}
        return len(_stable(payload)["segments"]) - 1

    assert pieces(None) == pieces(1.0) == 2
    assert pieces(1.6) == 1


def test_pieces_keep_the_verdict_of_their_row() -> None:
    """A low-confidence row the ladder kept: its one-character piece would look
    like a filler to the legs, but its confidence belongs to the whole row."""

    tokens = ["させて", "いただき", "ましょう。", "ん?"]
    row = _row(tokens, 10.0, 0.06, confidence=0.1)
    for word in row["words"]:
        word["confidence"] = 0.1
    payload = _payload(row, speech=[(10.0, 11.1), (12.0, 13.0)])
    once, _ = _stabilize(payload)
    assert _stretched(once) == 1
    assert "".join(s["text"] for s in once["segments"]) == "".join(tokens)
    assert _stable(once)["segments"] == once["segments"]


def test_a_stretched_row_is_not_judged_on_fields_from_its_squeezed_span() -> None:
    """Kept today because 0.08 s is too short for the duration-gated noise leg;
    the energy was measured on those 0.08 s, so the new length must not trip it."""

    row = _row(["ちゃんと", "出会って", "ました"], 10.0, 0.08 / 3, confidence=0.9)
    row["vad_weighted_energy_db"] = -25.0
    for word in row["words"]:
        word["confidence"] = 0.5
    without_room, _ = _stabilize(_payload(row, speech=[(10.0, 10.08)]))
    assert [s["text"] for s in without_room["segments"]] == [row["text"]]
    once, _ = _stabilize(_payload(row, speech=[(10.0, 15.0)]))
    assert _stretched(once) == 1
    assert [s["text"] for s in once["segments"]] == [row["text"]] and once["segments"][0]["end"] > 10.5
    assert _stable(once)["segments"] == once["segments"]


def test_importing_stabilize_loads_no_torch() -> None:
    """It is the light postprocessing path, and it now calls the splitter."""

    import os
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")   # this checkout, not an installed copy
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]))
    code = "import sys, finesub.speech.postprocessing.stabilization; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "False"


def test_profiles_other_than_the_default_run_no_repair() -> None:
    payload = _payload(_row(SQUEEZED, 10.0, 0.06), speech=[(10.0, 15.0)])
    for profile in (1, 2):
        out, _ = _stabilize(payload, profile=profile)
        assert out["segments"][0]["end"] == 10.24 and "recall" not in out["metadata"].get("asr_stabilize", {})


def test_without_a_repair_stabilize_stretches_nothing() -> None:
    out, _ = asr_stabilize.stabilize_payload(_payload(_row(SQUEEZED, 10.0, 0.06), speech=[(10.0, 15.0)]))
    assert out["segments"][0]["end"] == 10.24 and "recall" not in out["metadata"].get("asr_stabilize", {})


def test_the_plan_names_the_rows_a_pass_stretches_and_only_picks_among_them() -> None:
    first, second = _row(SQUEEZED, 10.0, 0.06), _row(SQUEEZED[::-1], 30.0, 0.06)
    rows = [first, _row(["次です"], 20.0, 1.0), second]
    speech = [{"start": a, "end": b} for a, b in [(10.0, 15.0), (20.0, 21.0), (30.0, 35.0)]]
    keep = lambda text: False
    plan = squeeze_repair.stretch_plan(rows, speech, leave=keep)
    assert set(plan) == {0, 2} and abs(plan[0] - (10.0 + NEED)) < 1e-6
    every, n = squeeze_repair.stretch_squeezed(rows, speech, leave=keep)
    gated, m = squeeze_repair.stretch_squeezed(rows, speech, leave=keep, only=lambda i: i == 2)
    assert (n, m) == (2, 1)
    assert gated[0] == first and gated[1:] == every[1:]


def test_the_repairs_second_timeline_pass_records_only_what_it_changed() -> None:
    # Two rows overlapping before any repair: the first pass records the overlap, the pass after a repair that
    # changes nothing must not record it again as if the repair had made it.
    payload = _payload(_row(["おはよう"], 10.0, 1.0), _row(["こんにちは"], 10.5, 1.0), speech=[(10.0, 12.0)])
    out, _ = asr_stabilize.stabilize_payload(payload, repair=lambda p, rows: (rows, {}))
    meta = out["metadata"]["asr_stabilize"]
    assert [r["kind"] for r in meta["segment_time_repairs"]] == ["overlap"]
    assert "segment_time_repairs" not in meta["recall"]
    # The same when the rows came out of order and the repair re-sorts them: the scan then names the other row.
    unordered = _payload(_row(["こんにちは"], 10.5, 1.0), _row(["おはよう"], 10.0, 1.0), speech=[(10.0, 12.0)])
    by_time = lambda p, rows: (sorted(rows, key=lambda r: r["start"]), {})
    out, _ = asr_stabilize.stabilize_payload(unordered, repair=by_time)
    meta = out["metadata"]["asr_stabilize"]
    assert [r["text"] for r in meta["segment_time_repairs"]] == ["おはよう"]
    assert [r["text"] for r in out["segments"]] == ["おはよう", "こんにちは"]
    assert "segment_time_repairs" not in meta["recall"]


def test_a_row_on_the_overlap_lane_stays_over_the_next_one_through_stabilize_and_the_raw_srt() -> None:
    # The collapse repair hands back a row stretched over the next one (docs/asr-stabilize.md, "坍缩行"): the second
    # timeline pass keeps the lane and the overlap, the rows come back by start, and the raw SRT shows both cues.
    from finesub.subtitles import line_groups, rendering

    def over(payload, rows):
        first = {**rows[0], "end": 13.0, line_groups.LANE_FIELD: True,
                 squeeze_repair.SOURCE_FIELD: {"start": rows[0]["start"], "end": rows[0]["end"]}}
        return [rows[1], first, *rows[2:]], {}

    payload = _payload(_row(["おはよう"], 10.0, 0.5), _row(["うん"], 11.0, 1.0), _row(["次です"], 20.0, 1.0),
                       speech=[(10.0, 13.0), (20.0, 21.0)])
    out, _ = asr_stabilize.stabilize_payload(payload, repair=over)
    rows = out["segments"]
    assert [r["text"] for r in rows] == ["おはよう", "うん", "次です"]
    assert rows[0][line_groups.LANE_FIELD] and rows[0]["end"] == 13.0 and (rows[1]["start"], rows[1]["end"]) == (11.0, 12.0)
    cues = rendering.segment_cues(rows)
    assert [(c[0], c[1]) for c in cues[:2]] == [(10.0, 13.0), (11.0, 12.0)]
