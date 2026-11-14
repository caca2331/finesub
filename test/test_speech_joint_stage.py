"""Cloud ASR in production: the lines (`speech/joint/lines.py`), MOSS Pro's timing (`moss_timing.py`), the
"cloud task" helper and the task's speaker rows, and the pipeline's cloud branch end to end with the service
replaced. The adapters themselves are `test_speech_joint.py`."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from finesub.speech.joint import lines as L
from finesub.speech.joint import moss_timing as T
from finesub.speech.joint import stage as joint_stage
from finesub.speech.joint.model import Transcript, Unit, turns_from_units
from finesub.speech.postprocessing import segmentation as seg

PARAMS = seg.split_params_for_length_scale(1.0)

def unit(start, end, text, speaker="A"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


# --- lines ---------------------------------------------------------------------------------------------------


def test_units_keep_a_latin_word_or_number_inside_cjk_text_whole():
    assert L.units_of("第一个bug。拿20分，3.5倍 Hide Together") == [
        "第", "一", "个", "bug。", "拿", "20", "分，", "3.5", "倍", "Hide", "Together"]


def test_dp_words_fold_punctuation_and_space_latin_words_only():
    words = L.dp_words([unit(0.0, 0.4, "Hello"), unit(0.4, 0.45, "!"), unit(0.5, 0.9, "world"),
                        unit(1.0, 1.2, "こん"), unit(1.2, 1.4, "ばんは")], duration=10)
    assert [w["word"] for w in words] == ["Hello!", "world", "こん", "ばんは"]
    assert [w["space_before"] for w in words] == [False, True, False, False]


def test_ungrid_pulls_abutting_words_apart_and_leaves_a_step_back_alone():
    out = L.ungrid([{"start": 0.0, "end": 0.5}, {"start": 0.5, "end": 1.0}, {"start": 0.2, "end": 0.3}])
    assert out[0]["end"] == pytest.approx(0.5 - L.GRID_GAP)
    assert out[1]["end"] == 1.0


def test_word_lines_keep_every_word_once_and_never_mix_speakers():
    units, t = [], 0.0
    for k in range(40):
        speaker = "A" if (k // 7) % 2 == 0 else "B"
        units.append(unit(t, t + 0.3, f"w{k}", speaker))
        t += 0.35 + (1.2 if k % 11 == 10 else 0.0)
    lines, vad = L.word_lines(units, duration=t + 1, params=PARAMS, language="en")
    words = [w["word"] for line in lines for w in line["words"]]
    assert words == [u["text"] for u in units]
    for line in lines:
        assert {u["speaker"] for u in units if u["text"] in {w["word"] for w in line["words"]}} == {line[L.LINE_SPEAKER]}
        assert all(key not in w for w in line["words"] for key in ("speaker", "seam"))
    assert vad and all(a["end"] < b["start"] for a, b in zip(vad, vad[1:]))


def test_speaker_runs_follow_the_order_given():
    runs = L.speaker_runs([unit(0, 1, "a", "A"), unit(5, 6, "b", "A"), unit(2, 3, "c", "B"),
                           unit(3, 4, "d", None), unit(4, 5, "e", None), unit(7, 8, "f", "A")])
    assert [[w["text"] for w in r] for r in runs] == [["a", "b"], ["c"], ["d", "e"], ["f"]]


def _talk(speaker, start, n, step=0.35):
    return [unit(start + k * step, start + k * step + 0.3, f"{speaker}{start:g}.{k}", speaker) for k in range(n)]


def test_35t_turns_each_run_their_own_dp_and_overlap_rather_than_interleave():
    # 35t's output order: A's long turn, then B's backchannel that was said during it, then A again.
    a, b, a2 = _talk("A", 0.0, 30), _talk("B", 3.0, 2), _talk("A", 12.0, 8)
    lines, _vad = L.word_lines(a + b + a2, duration=20, params=PARAMS, language="en", per_turn=True)
    turn_of = {u["text"]: k for k, turn in enumerate((a, b, a2)) for u in turn}
    for line in lines:
        assert len({turn_of[w["word"]] for w in line["words"]}) == 1  # no line takes words from two turns
    assert [line["start"] for line in lines] == sorted(line["start"] for line in lines)
    b_line = next(line for line in lines if line[L.LINE_SPEAKER] == "B")
    assert (b_line["start"], [w["word"] for w in b_line["words"]]) == (3.0, ["B3.0", "B3.1"])
    assert any(line[L.LINE_SPEAKER] == "A" and line["start"] < b_line["start"] < line["end"] for line in lines)
    assert [w["word"] for line in lines if line[L.LINE_SPEAKER] == "A" for w in line["words"]] == \
        [u["text"] for u in a + a2]


def test_mai2_is_one_dp_by_start():
    a, b = _talk("A", 0.0, 6), _talk("B", 2.2, 6)
    lines, _vad = L.word_lines(a + b, duration=10, params=PARAMS, language="en")
    words = [w["word"] for line in lines for w in line["words"]]
    assert words == [u["text"] for u in sorted(a + b, key=lambda u: u["start"])]  # interleaved by time


#: MAI2-shaped words (start, end, text, speaker): two speakers abutting within 0.03 s, a zero-length word inside a
#: line, a pause, a tail.
MAI2_WORDS = [
    (0.00, 0.40, "Hello", "S01"), (0.45, 0.80, "there", "S01"), (0.85, 1.00, "my", "S01"), (1.05, 1.50, "friend.", "S01"),
    (1.53, 1.70, "Hi", "S02"), (1.75, 1.90, "how", "S02"), (1.90, 1.90, "are", "S02"), (1.95, 2.40, "you?", "S02"),
    (2.42, 2.70, "Fine,", "S01"), (2.75, 3.20, "thanks.", "S01"),
    (5.00, 5.30, "Good", "S02"), (5.35, 5.45, "to", "S02"), (5.50, 6.00, "hear.", "S02"),
    (6.02, 6.30, "Yes.", "S01"),
]


def test_mai2_comes_out_unchanged_across_the_overlap_rules(tmp_path):
    """MAI2's words never overlap, so the per-speaker passes must leave its products as the code before the overlap
    rules (24cc1c1d) wrote them: the stable lines and both raw SRTs below are that code's output."""
    from finesub import stages
    from finesub.speech.postprocessing import stabilization
    from finesub.subtitles.model import parse_srt

    units = [unit(s, e, t, k) for s, e, t, k in MAI2_WORDS]
    segments, vad = L.word_lines(units, duration=8.0, params=PARAMS, language="en")
    paths = stages.default_pipeline_paths(tmp_path / "x.wav", tmp_path / "x.srt")
    paths.aligned_json.write_text(json.dumps({
        "segments": segments, "vad_timeline": {"intervals": vad, "pause_hints": {"scorer": [], "padding": []}},
        "metadata": {"asr_align": {"backend": "mai2", "language": "en", "audio_duration": 8.0}}}), encoding="utf-8")
    stabilization.run_asr_stabilize(paths.aligned_json, output_path=paths.stable_json, profile=-1)
    stable = json.loads(paths.stable_json.read_text(encoding="utf-8"))["segments"]
    assert [(s["start"], s["end"], s["text"], s["line_speaker"], [(w["start"], w["end"]) for w in s["words"]])
            for s in stable] == [
        (0.0, 1.5, "Hello there my friend.", "S01", [(0.0, 0.4), (0.45, 0.8), (0.85, 1.0), (1.05, 1.5)]),
        (1.53, 2.4, "Hi how are you?", "S02", [(1.53, 1.7), (1.75, 1.9), (1.9, 1.91), (1.95, 2.4)]),
        (2.42, 3.2, "Fine, thanks.", "S01", [(2.42, 2.7), (2.75, 3.2)]),
        (5.0, 6.0, "Good to hear.", "S02", [(5.0, 5.3), (5.35, 5.45), (5.5, 6.0)]),
        (6.02, 6.3, "Yes.", "S01", [(6.02, 6.3)]),
    ]

    def raw(word):
        if paths.raw_srt.exists():
            paths.raw_srt.unlink()
        stages._write_raw_srt(paths, word=word, postprocess_profile=0, empty_cues=False)
        return [(c.start, c.end, c.text) for c in parse_srt(paths.raw_srt.read_text(encoding="utf-8"))]

    assert raw(False) == [(0.0, 1.53, "Hello there my friend."), (1.53, 2.42, "Hi how are you?"),
                          (2.42, 3.5, "Fine, thanks."), (5.0, 6.02, "Good to hear."), (6.02, 6.6, "Yes.")]
    assert raw(True) == [
        (0.0, 0.45, "Hello"), (0.45, 0.85, "there"), (0.85, 1.05, "my"), (1.05, 1.53, "friend."), (1.53, 1.75, "Hi"),
        (1.75, 1.9, "how"), (1.9, 1.95, "are"), (1.95, 2.42, "you?"), (2.42, 2.75, "Fine,"), (2.75, 3.5, "thanks."),
        (5.0, 5.35, "Good"), (5.35, 5.5, "to"), (5.5, 6.02, "hear."), (6.02, 6.6, "Yes."),
    ]


def test_mai2_lines_widen_into_the_gaps_only():
    def line(start, end):
        return {"start": start, "end": end, "text": "x", "words": [{"start": start, "end": (start + end) / 2},
                                                                    {"start": (start + end) / 2, "end": end}]}

    # 0.5 s to the next line, then 0.15 s (the end takes 0.1, the start the 0.05 left), then abutting, then the audio end.
    lines = [line(0.05, 1.0), line(1.5, 2.0), line(2.15, 3.0), line(3.0, 3.95)]
    out = L.widened(lines, pad=0.1, duration=4.0)
    assert [(l["start"], l["end"]) for l in out] == [(0.0, 1.1), (1.4, 2.1), (2.1, 3.0), (3.0, 4.0)]
    assert [(l["words"][0]["start"], l["words"][-1]["end"]) for l in out] == [(l["start"], l["end"]) for l in out]
    assert out[0]["words"][0]["end"] == 0.525  # inner word edges stay
    assert lines[0]["start"] == 0.05  # the input is not touched


def test_widening_never_shrinks_a_line_or_adds_words():
    # Overlapping lines keep their overlapping edges; a line already past the audio's end keeps its end.
    out = L.widened([{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 0.9, "end": 4.2, "text": "b"}],
                    pad=0.1, duration=4.0)
    assert [(l["start"], l["end"]) for l in out] == [(0.0, 1.0), (0.9, 4.2)]
    assert all("words" not in l for l in out)
    assert L.widened([{"start": 1.0, "end": 2.0, "text": "c", "words": []}], pad=0.1, duration=4.0) == [
        {"start": 0.9, "end": 2.1, "text": "c", "words": []}]


def test_only_mai2_lines_are_widened(cloud_source, tmp_path, monkeypatch):
    from finesub.stages import run_pipeline

    cloud_source, _calls = cloud_source
    seen = []
    real = L.widened
    monkeypatch.setattr(L, "widened", lambda lines, **kw: seen.append(len(lines)) or real(lines, **kw))
    records = {}
    for backend in ("mai2", "gemini-35t"):
        paths = run_pipeline(cloud_source, output_path=tmp_path / backend / "talk.srt", stage="raw-srt", asr=backend,
                             language="en")
        aligned = json.loads(paths.aligned_json.read_text(encoding="utf-8"))
        records[backend] = aligned["metadata"][joint_stage.METADATA_KEY]
        if backend == "mai2":  # the derived VAD is the widened words'
            vad = aligned["vad_timeline"]["intervals"]
            for s in aligned["segments"]:
                assert any(v["start"] <= s["start"] and s["end"] <= v["end"] for v in vad), s
    assert len(seen) == 1
    assert records["mai2"]["edge_pad"] == L.MAI2_EDGE_PAD and "edge_pad" not in records["gemini-35t"]


def test_moss_lines_never_span_two_merged_segments():
    segs = [
        {"start": 0.0, "end": 3.0, "text": "今日はいい天気ですね散歩に", "speaker": "S01"},
        {"start": 3.1, "end": 6.0, "text": "行きましょうか公園まで歩いて", "speaker": "S01"},
        {"start": 1.0, "end": 1.8, "text": "そうですね", "speaker": "S02"},
    ]
    timed = [dict(s, units=L.interpolated(s)) for s in segs]
    lines, _vad = L.segment_lines(timed, duration=10, params=PARAMS, language="ja")
    assert all(any(line["text"] in s["text"] for s in segs) for line in lines)
    s02 = next(line for line in lines if line[L.LINE_SPEAKER] == "S02")
    assert (s02["start"], s02["end"]) == (1.0, 1.8)  # inside S01's line, kept whole


def test_line_speaker_is_who_holds_most_of_the_time():
    assert L.line_speaker([unit(0, 0.2, "a", "A"), unit(0.2, 1.0, "b", "B"), unit(1.0, 1.1, "c", "A")]) == "B"
    assert L.line_speaker([unit(0, 1, "a", None)]) == "unknown"


# --- MOSS Pro timing -----------------------------------------------------------------------------------------


SEGMENT = {"start": 10.0, "end": 16.0, "text": "今日はいい天気ですね。散歩に行きましょう。", "speaker": "S01"}


@pytest.mark.parametrize("spans", [
    [],  # nothing heard: interpolated
    [(10.4, 15.5)],  # one run, trimmed edges
    [(10.2, 12.6), (13.9, 15.8)],  # a pause near the full stop
    [(10.2, 15.0), (15.9, 15.95)],  # a trailing blip
], ids=["silent", "one-run", "pause", "blip"])
def test_warp_keeps_every_unit_once_in_order_and_within_bounds(spans):
    out = T.warp(SEGMENT, spans)
    assert [u["text"] for u in out] == L.units_of(SEGMENT["text"])
    starts = [u["start"] for u in out]
    assert starts == sorted(starts)
    assert all(u["end"] > u["start"] for u in out)
    assert out[0]["start"] <= SEGMENT["start"] + T.TRIM_MAX_START + 1e-9
    assert out[-1]["end"] >= SEGMENT["end"] - T.TRIM_MAX_END - 1e-9
    assert out[0]["start"] >= SEGMENT["start"] - 1e-9 and out[-1]["end"] <= SEGMENT["end"] + 1e-9


def test_warp_puts_the_sentence_end_at_the_pause():
    out = T.warp(SEGMENT, [(10.2, 12.6), (13.9, 15.8)])
    full_stop = next(k for k, u in enumerate(out) if u["text"].endswith("。"))
    assert out[full_stop]["end"] <= 12.6 and out[full_stop + 1]["start"] >= 13.9


def test_warp_interpolates_when_too_little_of_the_segment_is_heard():
    heard = [(10.0, 10.0 + 0.5 * T.MIN_VOICED * (SEGMENT["end"] - SEGMENT["start"]))]
    assert T.warp(SEGMENT, heard) == L.interpolated(SEGMENT)


def test_timed_takes_the_aligner_only_for_long_segments():
    short = {"start": 0.0, "end": 1.0, "text": "はい。", "speaker": "S01"}
    long = {"start": 2.0, "end": 30.0, "text": "あ" * 60 + "。", "speaker": "S02"}
    seen = []

    def align(long_ones, every):
        seen.append([s["text"] for s in long_ones])
        return {0: [unit(2.0, 30.0, long["text"], "S02")]}

    out, counts = T.timed([long, short], np.zeros(1000, np.float32), duration=40, params=PARAMS, align=align)
    assert seen == [[long["text"]]]
    assert [s["text"] for s in out] == ["はい。", long["text"]]  # by start
    assert counts == {"segments": 2, "aligned": 1}
    assert out[1]["units"] == [unit(2.0, 30.0, long["text"], "S02")]


def test_a_segment_the_aligner_gave_up_on_is_timed_by_silero():
    long = {"start": 2.0, "end": 30.0, "text": "あ" * 60 + "。", "speaker": "S02"}
    out, counts = T.timed([long], np.zeros(1000, np.float32), duration=40, params=PARAMS, align=lambda *_: {})
    assert counts["aligned"] == 0
    assert [u["text"] for u in out[0]["units"]] == L.units_of(long["text"])


def test_segment_lines_merge_short_same_speaker_segments_and_keep_every_unit():
    segs = [
        {"start": 0.0, "end": 0.6, "text": "ええ。", "speaker": "S01"},
        {"start": 0.8, "end": 2.4, "text": "そうですね。", "speaker": "S01"},
        {"start": 2.6, "end": 4.0, "text": "わかりました。", "speaker": "S02"},
    ]
    timed = [dict(s, units=L.interpolated(s)) for s in segs]
    lines, _vad = L.segment_lines(timed, duration=10, params=PARAMS, language="ja")
    assert "".join(line["text"] for line in lines) == "".join(s["text"] for s in segs)
    assert [line[L.LINE_SPEAKER] for line in lines] == ["S01", "S02"]


def same_key_overlaps(lines):
    """Pairs of one speaker's lines that overlap in time (the timeline contract allows none)."""
    return [(a["text"], b["text"]) for i, a in enumerate(lines) for b in lines[i + 1:]
            if a[L.LINE_SPEAKER] == b[L.LINE_SPEAKER] and a["start"] < b["end"] and b["start"] < a["end"]]


def test_one_speakers_overlapping_lines_are_pulled_back_and_two_speakers_kept():
    segs = [
        {"start": 0.0, "end": 4.0, "text": "今日はとても良い天気ですね本当に", "speaker": "S01"},
        {"start": 3.6, "end": 7.0, "text": "それでは始めましょうかみなさん", "speaker": "S01"},
        {"start": 1.0, "end": 2.0, "text": "うん", "speaker": "S02"},
    ]
    timed = [dict(s, units=L.interpolated(s)) for s in segs]
    lines, _vad = L.segment_lines(timed, duration=10, params=PARAMS, language="ja")
    assert not same_key_overlaps(lines)
    assert "".join(line["text"] for line in lines if line[L.LINE_SPEAKER] == "S01") == segs[0]["text"] + segs[1]["text"]
    nod = next(line for line in lines if line[L.LINE_SPEAKER] == "S02")
    assert any(line["start"] < nod["end"] and nod["start"] < line["end"] for line in lines if line is not nod)


def test_aligner_pad_stops_at_a_neighbour_and_collapsed_units_are_caught():
    from finesub.speech.joint import aligner as A

    a = {"start": 1.0, "end": 2.0}
    b = {"start": 2.2, "end": 4.0}
    c = {"start": 4.1, "end": 5.0}
    assert A._span([a, b, c], b, 10.0) == (2.0, 4.1)  # never into a neighbour it does not overlap
    assert A._span([b], b, 4.3) == pytest.approx((2.2 - A.PAD, 4.3))  # PAD, clipped to the audio
    squeezed = [{"text": "あ", "start": 1.0, "end": 1.0 + 0.01 * k} for k in range(8)]
    assert A._squeezed(squeezed)
    assert not A._squeezed([{"text": "あいう", "start": 0.0, "end": 1.0}, {"text": "えお", "start": 1.0, "end": 2.0}])
    gaps = [{"text": "a"}, {"text": "b", "start": 1.0, "end": 1.5}, {"text": "c"}]
    A._fill(gaps)
    assert [(u["start"], u["end"]) for u in gaps] == [(1.0, 1.0), (1.0, 1.5), (1.5, 1.5)]


# --- the cloud task ------------------------------------------------------------------------------------------


def _products(path: Path, backend: str | None) -> Path:
    metadata = {joint_stage.METADATA_KEY: {"backend": backend}} if backend else {}
    path.write_text(json.dumps({"segments": [], "metadata": metadata}), encoding="utf-8")
    return path


def test_the_task_backend_is_what_its_products_record(tmp_path):
    stable, aligned = tmp_path / "x-stable.json", tmp_path / "x-aligned.json"
    assert joint_stage.task_backend(stable, aligned, "moss") == "moss-pro"  # nothing yet: the request
    _products(aligned, "mai2")
    assert joint_stage.task_backend(stable, aligned, "local") == "mai2"
    _products(stable, None)  # a local stable wins over a cloud aligned
    assert joint_stage.task_backend(stable, aligned, "mai2") == "local"


def test_cloud_rows_number_tracks_by_first_appearance(tmp_path):
    from finesub.speech.diarization.stage import write_joint_rows

    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"start": 0, "end": 1, "text": "a", "line_speaker": "S02"},
        {"start": 1, "end": 2, "text": "b", "line_speaker": "S01"},
        {"start": 2, "end": 3, "text": "c", "line_speaker": "S02"},
    ]}), encoding="utf-8")
    rows = json.loads(write_joint_rows(stable, backend="moss-pro", model="m").read_text(encoding="utf-8"))
    assert rows["speaker"]["backend"] == "joint:moss-pro"
    tracks = [row["speaker"]["track"] for row in rows["rows"]]
    assert tracks == [0, 1, 0]


def test_a_cloud_tasks_speaker_ass_shows_two_people_at_once(tmp_path):
    """The overlap a cloud task keeps reaches the speaker export: one style per speaker, two events at once."""
    from types import SimpleNamespace

    from finesub.speech.diarization.stage import write_joint_rows
    from finesub.stages import _run_speaker_export_stage

    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"id": "1", "start": 0.0, "end": 10.0, "text": "長い話", "line_speaker": "S01"},
        {"id": "2", "start": 2.0, "end": 3.0, "text": "うん", "line_speaker": "S02"},
    ]}), encoding="utf-8")
    write_joint_rows(stable, backend="moss-pro", model="m")
    _run_speaker_export_stage(SimpleNamespace(stable_json=stable), speaker_tracks="ass", empty_cues=False, announce=lambda *a, **k: None)
    ass = (tmp_path / "x-speakers" / "x-speakers.ass").read_text(encoding="utf-8")
    assert "Style: speaker0," in ass and "Style: speaker1," in ass
    assert "Dialogue: 0,0:00:00.00,0:00:10.00,speaker0," in ass
    assert "Dialogue: 0,0:00:02.00,0:00:03.00,speaker1," in ass


# --- the pipeline's cloud branch -----------------------------------------------------------------------------


@pytest.fixture
def cloud_source(tmp_path, monkeypatch) -> Path:
    import soundfile as sf

    sr = 16000
    t = np.arange(int(sr * 8.0)) / sr
    path = tmp_path / "talk.wav"
    sf.write(path, (0.1 * np.sin(2 * math.pi * 220 * t)).astype(np.float32), sr)
    words = [Unit(0.5 + 0.4 * k, 0.8 + 0.4 * k, f"word{k}" + ("." if k % 5 == 4 else ""), "A" if k < 8 else "B")
             for k in range(15)]
    calls = []

    def transcribe(backend, source, *, cache_root, language, duration, repair, aligner, gpu_tier, device, speech):
        calls.append(backend)
        return Transcript(backend, "fake-model", duration, "word", turns_from_units(words)), {}

    from finesub.media import ffmpeg

    monkeypatch.setattr(ffmpeg, "probe_media_duration", lambda src: 8.0)  # CI's Linux job has no ffprobe
    monkeypatch.setattr(joint_stage, "transcribe", transcribe)

    def no_local(*args, **kwargs):
        raise AssertionError("a cloud task ran a local stage")

    from finesub.speech.preprocessing.separator import separation
    from finesub.speech.recognition import vad_asr_stage

    for name in ("run_vocal_separation", "encode_asr_delivery"):
        monkeypatch.setattr(separation, name, no_local)
    monkeypatch.setattr(vad_asr_stage, "run_vad_asr", no_local)
    from finesub.speech.verification import recall_decode

    monkeypatch.setattr(recall_decode, "recall_repair", no_local)  # the stable step's repair is Whisper's too
    from finesub.speech.joint import sbd

    def no_sentence_ends(words):
        # 35t's sentence-boundary model is a large Hugging Face download, and on Linux into a HOME the suite
        # isolates, so every run; these tests are about the stage, not the model's probabilities.
        for word in words:
            word[seg.SENTENCE_END_BONUS_KEY] = 0.0

    monkeypatch.setattr(sbd, "sentence_ends", no_sentence_ends)
    return path, calls


def test_a_cloud_task_skips_the_local_stages_and_keeps_its_backend(cloud_source, tmp_path, capsys):
    from finesub.stages import run_pipeline

    cloud_source, calls = cloud_source
    output = tmp_path / "out" / "talk.srt"
    paths = run_pipeline(cloud_source, output_path=output, stage="raw-srt", asr="mai2", language="en",
                         squeeze_repair="on", gap_fill="on")
    stable = json.loads(paths.stable_json.read_text(encoding="utf-8"))
    aligned = json.loads(paths.aligned_json.read_text(encoding="utf-8"))
    assert stable["segments"] == aligned["segments"]  # profile -1, no repair: a verbatim copy
    assert not same_key_overlaps(stable["segments"])
    assert stable["metadata"][joint_stage.METADATA_KEY]["backend"] == "mai2"
    assert not paths.resolve_vocal_audio().exists()
    rows = json.loads(paths.diar_rows_json.read_text(encoding="utf-8"))
    assert rows["speaker"]["backend"] == "joint:mai2"
    assert paths.raw_srt.read_text(encoding="utf-8").count("-->") == len(stable["segments"])
    from finesub.speech.postprocessing.stabilization import recorded_audio_end

    assert recorded_audio_end(paths.stable_json) == pytest.approx(8.0, abs=0.05)

    # Stable only, and asked for local this time: still the task's own cloud products, nothing re-asked.
    paths.aligned_json.unlink()
    paths.raw_srt.unlink()
    paths.diar_rows_json.unlink()
    again = run_pipeline(cloud_source, output_path=output, stage="raw-srt", asr="local", language="en")
    assert calls == ["mai2"]
    assert again.raw_srt.exists() and again.diar_rows_json.exists()
    assert "mai2" in capsys.readouterr().err


def test_a_cloud_aligned_alone_resumed_local_with_acoustics_stays_a_cloud_task(cloud_source, tmp_path, monkeypatch):
    from finesub import stages
    from finesub.speech.diarization import stage as diarization
    from finesub.stages import run_pipeline

    cloud_source, calls = cloud_source
    output = tmp_path / "out" / "talk.srt"
    paths = run_pipeline(cloud_source, output_path=output, stage="raw-srt", asr="mai2", language="en")
    for product in (paths.stable_json, paths.raw_srt, paths.diar_rows_json):
        product.unlink()
    monkeypatch.setattr(diarization, "run_diarization", lambda *a, **k: pytest.fail("acoustic diarization ran"))
    monkeypatch.setattr(stages.PipelinePaths, "resolve_vocal_audio",
                        lambda self: pytest.fail("a cloud task looked for a vocal track"))
    again = run_pipeline(cloud_source, output_path=output, stage="raw-srt", asr="local", language="en",
                         diarize="acoustics")
    assert calls == ["mai2"]
    stable = json.loads(again.stable_json.read_text(encoding="utf-8"))
    assert stable["metadata"][joint_stage.METADATA_KEY]["backend"] == "mai2"
    assert not same_key_overlaps(stable["segments"])
    assert json.loads(again.diar_rows_json.read_text(encoding="utf-8"))["speaker"]["backend"] == "joint:mai2"


def test_the_aligner_is_only_a_note_for_another_backend(cloud_source, tmp_path, capsys):
    from finesub.stages import run_pipeline

    cloud_source, _calls = cloud_source
    output = tmp_path / "out" / "talk.srt"
    for _ in range(2):  # on the run that makes the aligned and on a resumed one
        run_pipeline(cloud_source, output_path=output, stage="raw-srt", asr="mai2", language="en", asr_aligner=True)
        assert "--asr-aligner only applies to moss-pro, moss-local and gemini-35t" in capsys.readouterr().err


def test_the_repair_level_reaches_the_backend_and_only_a_repairing_run_records_it(cloud_source, tmp_path, monkeypatch):
    from finesub.stages import run_pipeline

    cloud_source, _calls = cloud_source
    fake, seen = joint_stage.transcribe, []

    def transcribe(backend, source, **kwargs):
        seen.append(kwargs["repair"])
        return fake(backend, source, **kwargs)

    monkeypatch.setattr(joint_stage, "transcribe", transcribe)
    records = []
    for name, repair in (("a", None), ("b", "off")):
        paths = run_pipeline(cloud_source, output_path=tmp_path / name / "talk.srt", stage="raw-srt", asr="gemini-35t",
                             language="en", asr_repair=repair)
        records.append(json.loads(paths.aligned_json.read_text(encoding="utf-8"))["metadata"][joint_stage.METADATA_KEY])
    assert seen == ["on", "off"]
    assert records[0]["repair"] == "on" and "repair" not in records[1]


def test_local_products_under_a_cloud_request_say_so(tmp_path, capsys):
    from finesub.stages import _note_task_backend, default_pipeline_paths

    paths = default_pipeline_paths(tmp_path / "talk.wav", tmp_path / "out" / "talk.srt")
    paths.stable_json.parent.mkdir(parents=True)
    _products(paths.stable_json, None)
    backend = joint_stage.task_backend(paths.stable_json, paths.aligned_json, "moss")
    assert backend == "local"
    _note_task_backend(paths, backend, "moss", None)
    assert "came from local" in capsys.readouterr().err


def test_the_speaker_judge_of_a_cloud_task_listens_to_the_mix(tmp_path):
    from finesub.stages import _correction_request, default_pipeline_paths

    source = tmp_path / "talk.wav"
    paths = default_pipeline_paths(source, tmp_path / "out" / "talk.srt")
    for cloud, heard in ((True, source), (False, paths.resolve_vocal_audio())):
        request = _correction_request(paths, source, "final-srt", axes=("text", "none", "quality", "serial", 1.0),
                                      llm_video=None, refined_srt=None, task_id="", postprocess_profile=0,
                                      joint=cloud)
        assert request.vocal_audio == heard


def test_upload_notice_is_one_reporter_warning_before_the_first_call(monkeypatch):
    warned = []

    class Reporter:
        def warning(self, code, message, **_):
            warned.append(code)

    monkeypatch.setattr(joint_stage, "current_reporter", lambda: Reporter())
    calls = []
    call = joint_stage._announcing("moss-pro", lambda *a, **k: calls.append(warned[:]) or "ok")
    assert call() == "ok" and call() == "ok"
    assert calls == [["cloud-asr-upload"], ["cloud-asr-upload"]]  # said before the first upload, and only once


def test_moss_pro_and_its_line_timing_decode_the_source_once(tmp_path, monkeypatch):
    """MOSS Pro reads the samples for its gap check and the stage again for the timing: one decode for both."""
    from finesub.media import ffmpeg
    from finesub.speech.joint import mosspro

    source = tmp_path / "talk.wav"
    source.write_bytes(b"RIFF")
    decodes = []
    monkeypatch.setattr(ffmpeg, "decode_pcm16k", lambda src, **_: decodes.append(src) or bytes(8 * 16000 * 2))
    monkeypatch.setattr(ffmpeg, "probe_media_duration", lambda src: 8.0)
    monkeypatch.setattr(T, "frame_probabilities", lambda pcm: np.full(len(pcm) // T.HOP, 0.9, np.float32))
    monkeypatch.setattr(mosspro, "Caller", lambda: object())

    def transcribe(source, *, speech, duration, **_):
        assert len(speech.pcm()) == 8 * 16000
        units = [Unit(0.5, 3.0, "こんにちは。", "S01")]
        return Transcript("moss-pro", "m", duration, "segment", turns_from_units(units))

    monkeypatch.setattr(mosspro, "transcribe", transcribe)
    joint_stage.run(source, backend="moss-pro", output_path=tmp_path / "a-aligned.json", cache_root=tmp_path / "c",
                    language="ja")
    assert decodes == [source]
