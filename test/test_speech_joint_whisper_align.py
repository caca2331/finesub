"""Whisper re-timing the forced aligner's unsure boundaries (`whisper_align`) and its place in `aligned_units`.
No model is loaded: the decode is stubbed, the combination and the hand-over between the two models are real."""

from __future__ import annotations

import numpy as np
import pytest

from finesub.speech.joint import aligner as A
from finesub.speech.joint import whisper_align as W


def _words(*spans):
    return [{"word": w, "start": a, "end": b} for w, a, b in spans]


def test_unit_times_spread_each_word_over_its_characters():
    # Japanese: words without spaces, a punctuation unit taking no time of its own
    words = _words(("こん", 0.0, 0.4), ("にちは", 0.4, 1.0), ("。", 1.0, 1.0))
    got = W._unit_times(words, "こんにちは。", ["こ", "ん", "に", "ち", "は"])
    assert got == pytest.approx([0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0])


def test_unit_times_read_english_words_without_their_spaces_and_skip_the_disfluency_marker():
    words = _words(("[*]", 0.0, 0.3), ("Hello", 0.3, 0.8), ("world", 1.0, 1.5))
    got = W._unit_times(words, "Hello world", ["Hello", "world"])
    assert got == pytest.approx([0.3, 0.8, 1.0, 1.5])


def test_unit_times_take_a_last_word_that_runs_on_past_the_text_and_refuse_other_text():
    # the decoder went on after the forced text: the text's share of the word's characters keeps its share of time
    run_on = _words(("行", 0.0, 0.2), ("って！:", 0.2, 0.8))
    assert W._unit_times(run_on, "行って", ["行", "っ", "て"]) == pytest.approx([0.0, 0.2, 0.2, 0.35, 0.35, 0.5])
    assert W._unit_times(_words(("違う", 0.0, 0.4)), "行って", ["行", "っ", "て"]) is None
    assert W._unit_times(_words(("行", 0.0, 0.4)), "行って", ["行", "っ", "て"]) is None  # stopped short


def test_unit_times_read_the_teacher_forced_words_with_their_leading_spaces():
    # `find_alignment` words carry the space before them; the last unit's end is the last word's end
    words = _words(("Hello", 0.2, 0.7), (" big", 0.7, 1.1), (" world", 1.3, 1.9))
    got = W._unit_times(words, "Hello big world", ["Hello", "big", "world"])
    assert got == pytest.approx([0.2, 0.7, 0.7, 1.1, 1.3, 1.9])


class _Stub:
    """Just what `retime` asks of the model, with the decodes stubbed in `_stubbed`."""

    hf_tokenizer = None


def _stubbed(monkeypatch, *, detected="ja", trace_ok=True):
    fwt = pytest.importorskip("faster_whisper.tokenizer")

    monkeypatch.setattr(W, "_model", lambda: _Stub())
    monkeypatch.setattr(W, "_encode", lambda model, clip: ("enc", 100))
    monkeypatch.setattr(W, "_detected", lambda model, enc: detected)
    monkeypatch.setattr(fwt, "Tokenizer", lambda *a, **k: type("T", (), {"encode": lambda self, text: [1]})())
    words = _words(("あ", 0.1, 0.5), ("い", 0.5, 0.9))
    monkeypatch.setattr(W, "_trace", lambda *a: words if trace_ok else None)
    _Stub.find_alignment = lambda self, tok, tokens, enc, frames: [words]


def test_retime_skips_long_windows_english_under_auto_and_failed_decodes(monkeypatch):
    def job(a, b):
        return W.Job(text="あい", kept_units=["あ", "い"], a=a, b=b, times=np.array([a, a + .5, a + .5, a + 1.0]),
                     peaks=np.array([0.9, 0.3, 0.3, 0.9]))

    _stubbed(monkeypatch)
    out, record = W.retime([job(0.0, 2.0), job(10.0, 10.0 + W.WINDOW + 1)], np.zeros(16000 * 60, np.float32),
                           W.Retime("ja", False))
    assert set(out) == {0} and record["skipped_long"] == 1
    assert out[0] == pytest.approx([0.0, 0.5, 0.5, 1.0])  # sure edges kept; Whisper's 0.5 there less no offset
    _stubbed(monkeypatch, detected="en")
    out, record = W.retime([job(0.0, 2.0)], np.zeros(16000 * 5, np.float32), W.Retime(None, False))
    assert out == {} and record["skipped_language"] == 1
    out, _ = W.retime([job(0.0, 2.0)], np.zeros(16000 * 5, np.float32), W.Retime(None, True))
    assert set(out) == {0}  # `on`: English too
    _stubbed(monkeypatch, trace_ok=False)
    out, record = W.retime([job(0.0, 2.0)], np.zeros(16000 * 5, np.float32), W.Retime("ja", False))
    assert out == {} and record["skipped_decode"] == 1

    def raising(*a):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(W, "_trace", raising)  # every segment fails: one failure for the caller to warn about
    with pytest.raises(RuntimeError, match="out of memory"):
        W.retime([job(0.0, 2.0), job(3.0, 5.0)], np.zeros(16000 * 6, np.float32), W.Retime("ja", False))


def _job(times, peaks, wt=None, wh=None):
    return W.Job(text="", kept_units=[], a=0.0, b=10.0, times=np.array(times, float), peaks=np.array(peaks, float),
                 wt=None if wt is None else np.array(wt, float), wh=None if wh is None else np.array(wh, float))


def test_whisper_replaces_only_the_unsure_boundaries_after_its_offset_is_taken_off():
    # Whisper runs 0.05 s late on every boundary the aligner is sure of: that is its offset
    sure = _job([0.0, 1.0, 1.0, 2.0], [0.9] * 4, wt=[0.05, 1.05, 1.05, 2.05], wh=[0.1, 1.1, 1.1, 2.1])
    # internal boundaries at 0.7 keep the aligner's time (>= KEEP_INTERNAL); the start at 0.7 does not (< KEEP_EDGE)
    edges = _job([3.0, 4.0, 4.0, 5.0, 5.0, 6.0], [0.7, 0.7, 0.5, 0.9, 0.9, 0.5],
                 wt=[3.45, 4.5, 4.35, 5.5, 5.5, 7.0], wh=[3.5, 4.5, 4.5, 5.5, 5.5, 6.4])
    untimed = _job([7.0, 8.0], [0.1, 0.1])  # Whisper could not decode it: left as it is
    out, record = W.combined([sure, edges, untimed])
    assert set(out) == {0, 1}
    assert out[0] == pytest.approx(sure.times)
    # start and the internal 0.5 from the trace less 0.05; the end from the teacher-forced alignment less its 0.1
    assert out[1] == pytest.approx([3.4, 4.0, 4.3, 5.0, 5.0, 6.3])
    assert record == {"segments": 2, "boundaries": 3, "offset_ms": 50, "end_offset_ms": 100}


def test_whispers_times_stay_inside_the_aligners_window():
    # the window ends at the source's end (10 s); Whisper's last start 9.98 s less a -0.03 s offset would be 10.01 s
    job = W.Job(text="", kept_units=[], a=8.0, b=10.0, times=np.array([8.0, 9.0, 9.0, 9.99]),
                peaks=np.array([0.9, 0.9, 0.1, 0.1]), wt=np.array([7.97, 8.97, 9.98, 10.0]), wh=np.array([8.0, 9.0, 9.0, 10.02]))
    out, record = W.combined([job])
    assert out[0] == pytest.approx([8.0, 9.0, 9.0, 9.99]) and record["boundaries"] == 0  # the aligner's, both


def test_without_whisper_s_end_alignment_the_segment_keeps_the_aligners_end():
    job = _job([0.0, 1.0], [0.1, 0.1], wt=[0.2, 1.5])
    out, record = W.combined([job])
    assert out[0] == pytest.approx([0.2, 1.0]) and record["offset_ms"] == 0  # no sure boundary: no offset


def _timed_stub(monkeypatch, events):
    """`aligned_units` with the aligner's decode stubbed: one segment, the aligner unsure of its middle."""
    seconds = np.array([0.0, 0.5, 0.5, 1.0])
    peaks = np.array([0.9, 0.3, 0.3, 0.9])
    monkeypatch.setattr(A, "_kept", lambda unit: unit)
    monkeypatch.setattr(A, "_aligned", lambda segments, todo, all_segments, wav, total, device: (
        events.append(("aligner", device)) or {0: A._Timed(1.0, 3.0, seconds.copy(), peaks)}))
    monkeypatch.setattr(A, "_drop", lambda: events.append("aligner released"))
    monkeypatch.setattr(W, "release", lambda: events.append("whisper released"))
    from finesub.speech.runtime import gpu_stage_gate

    class Gate:
        def acquire(self, family, **_):
            events.append(("gate", family))
            return type("Lease", (), {"release": lambda self: events.append("gate released")})()

    monkeypatch.setattr(gpu_stage_gate, "GPU_STAGE_GATE", Gate())
    return {"start": 1.0, "end": 2.0, "text": "あい"}


def test_the_aligner_is_released_before_whisper_re_times_and_whisper_after_it(monkeypatch):
    events: list = []
    seg = _timed_stub(monkeypatch, events)

    def retime(jobs, wav, how):
        events.append(("whisper", how))
        assert jobs[0].times == pytest.approx([1.0, 1.5, 1.5, 2.0]) and jobs[0].text == "あい"
        return {0: np.array([1.0, 1.7, 1.7, 2.0])}, {"segments": 1}

    monkeypatch.setattr(W, "retime", retime)
    record: dict = {}
    how = W.Retime("ja", False)
    got = A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16), device="cpu", retime=how,
                          record=record)
    assert events == [("aligner", "cpu"), "aligner released", ("gate", W.GATE_FAMILY), ("whisper", how),
                      "whisper released", "gate released"]
    assert [t for u in got[0] for t in (u["start"], u["end"])] == pytest.approx([1.0, 1.7, 1.7, 2.0])
    assert record == {"aligner_whisper": {"segments": 1}}


def test_a_failing_whisper_leaves_the_aligners_times_with_a_warning(monkeypatch):
    events: list = []
    seg = _timed_stub(monkeypatch, events)

    def broken(jobs, wav, how):
        raise RuntimeError("CTranslate2 was not built with the WT refine trace extension")

    warned = []

    class Reporter:
        def warning(self, code, message, **_):
            warned.append(code)

    from finesub import reporting

    monkeypatch.setattr(W, "retime", broken)
    monkeypatch.setattr(reporting, "current_reporter", lambda: Reporter())
    record: dict = {}
    got = A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16), retime=W.Retime(None, True),
                          record=record)
    assert [t for u in got[0] for t in (u["start"], u["end"])] == pytest.approx([1.0, 1.5, 1.5, 2.0])
    assert warned == ["joint-aligner-whisper"] and "WT refine trace" in record["aligner_whisper_error"]
    assert events[-2:] == ["whisper released", "gate released"]


def test_with_no_segment_inside_whispers_window_whisper_is_not_loaded(monkeypatch):
    events: list = []
    seg = _timed_stub(monkeypatch, events)
    monkeypatch.setattr(A, "_aligned", lambda segments, todo, all_segments, wav, total, device: {
        0: A._Timed(1.0, 1.0 + W.WINDOW + 1, np.array([0.0, 0.5, 0.5, 1.0]), np.array([0.9, 0.3, 0.3, 0.9]))})
    monkeypatch.setattr(W, "retime", lambda *a: pytest.fail("Whisper loaded for nothing it could take"))
    record: dict = {}
    A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(16000 * 40, np.int16), retime=W.Retime("ja", False),
                    record=record)
    assert events == ["aligner released"] and record == {"aligner_whisper": {"segments": 0, "skipped_long": 1}}


def test_without_retime_the_aligner_stays_loaded_for_the_next_call(monkeypatch):
    events: list = []
    seg = _timed_stub(monkeypatch, events)
    A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16))
    assert events == [("aligner", "cuda")]  # 35t's repairs align spot by spot; the stage releases it at the end


def test_the_whisper_step_waits_for_another_items_local_whisper_to_leave_the_card(monkeypatch):
    """The real gate: a batch item's local Whisper ASR (family `wt`) holds it; the network lane's re-timing waits."""
    import threading

    from finesub.speech.runtime import gpu_stage_gate

    events: list = []
    seg = _timed_stub(monkeypatch, events)
    gate = gpu_stage_gate.GpuStageGate()
    monkeypatch.setattr(gpu_stage_gate, "GPU_STAGE_GATE", gate)
    monkeypatch.setattr(W, "retime", lambda jobs, wav, how: events.append("whisper") or ({}, {}))
    asr = gate.acquire("wt")
    worker = threading.Thread(target=lambda: A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16),
                                                             retime=W.Retime("ja", False)))
    worker.start()
    worker.join(0.3)
    assert worker.is_alive() and "whisper" not in events
    asr.release()
    worker.join(5)
    assert not worker.is_alive() and "whisper" in events
    with pytest.raises(RuntimeError):  # and gave its lease back
        gate.release(W.GATE_FAMILY)


def test_a_task_asking_for_another_device_gets_its_own_aligner_not_the_cached_one(monkeypatch):
    """The network lane's two workers may run a CPU task after a GPU one: the cached model is keyed by the device."""
    pytest.importorskip("transformers")
    from finesub.speech.runtime import device, hf_weights

    loads: list[str] = []
    monkeypatch.setattr(hf_weights, "prepare", lambda manifest_id: None)
    monkeypatch.setattr(hf_weights, "offline_first", lambda fetch, plan, what: loads.append("load") or (object(), None))
    monkeypatch.setattr(device, "resolve_device", lambda requested, context=None: requested)
    monkeypatch.setattr(A, "_LOADED", None)
    monkeypatch.setattr(A, "_LOADED_FOR", None)
    first = A._model("cuda")
    assert A._model("cuda") is first and loads == ["load"]  # the same device: kept
    second = A._model("cpu")
    assert second is not first and loads == ["load", "load"] and A._LOADED_FOR == "cpu"
    assert A._model("cuda") is not second and A._LOADED_FOR == "cuda"


def test_a_failing_whisper_frees_what_its_frames_held_before_the_gate_opens(monkeypatch):
    """The traceback of a failed decode holds `retime`'s frames and the model in them: the gate opens only after
    the exception is gone and the model collected."""
    import gc
    import weakref

    from finesub import reporting

    events: list = []
    seg = _timed_stub(monkeypatch, events)

    class Model:
        pass

    refs = []

    def broken(jobs, wav, how):
        model = Model()
        refs.append(weakref.ref(model))
        raise RuntimeError("CUDA out of memory")

    alive = []

    def release():
        gc.collect()
        alive.append(refs[0]() is not None)

    monkeypatch.setattr(W, "retime", broken)
    monkeypatch.setattr(W, "release", release)
    monkeypatch.setattr(reporting, "current_reporter",
                        lambda: type("R", (), {"warning": lambda self, *a, **k: None})())
    A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16), retime=W.Retime("ja", False))
    assert alive == [False] and events[-1] == "gate released"


def test_whisper_is_not_fetched_where_ctranslate2_cannot_use_cuda(monkeypatch):
    from finesub.speech.recognition import vad_asr_stage
    from finesub.speech.runtime import device

    monkeypatch.setattr(W, "_LOADED", None)
    monkeypatch.setattr(device, "ct2_cuda_unusable_reason", lambda: "a CPU-only CTranslate2 wheel")
    monkeypatch.setattr(vad_asr_stage, "ensure_asr_weights", lambda *a, **k: pytest.fail("fetched Whisper's weights"))
    with pytest.raises(RuntimeError, match="CPU-only"):
        W._model()


def test_ordering_that_would_push_a_unit_past_the_audio_keeps_the_aligners_times_and_every_character(monkeypatch):
    """The window ends at the audio's end (3 s); Whisper's times inside it, but its last unit 2.99-3.0 s ordered to
    the 20 ms floor ends at 3.01 s: the segment keeps the aligner's times and the line DP keeps both characters."""
    from finesub.speech.joint import lines as L

    events: list = []
    seg = _timed_stub(monkeypatch, events)
    monkeypatch.setattr(W, "retime", lambda jobs, wav, how: (
        {0: np.array([1.0, 2.99, 2.99, 3.0])}, {"segments": 1, "boundaries": 3}))
    record: dict = {}
    got = A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16), retime=W.Retime("ja", False),
                          record=record)
    assert [t for u in got[0] for t in (u["start"], u["end"])] == pytest.approx([1.0, 1.5, 1.5, 2.0])
    assert record["aligner_whisper"] == {"segments": 0, "boundaries": 0, "outside_window": 1}
    assert "".join(w["word"] for w in L.dp_words(got[0], duration=3.0)) == "あい"


def test_a_stop_inside_whisper_still_releases_it_before_the_gate_and_goes_on_up(monkeypatch):
    events: list = []
    seg = _timed_stub(monkeypatch, events)

    def stopped(jobs, wav, how):
        raise SystemExit(2)

    monkeypatch.setattr(W, "retime", stopped)
    with pytest.raises(SystemExit):
        A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16), retime=W.Retime("ja", False))
    assert events[-2:] == ["whisper released", "gate released"]


@pytest.mark.parametrize("mark", ["%", "+"])
def test_a_mark_with_no_time_of_its_own_at_the_audios_end_stays_with_its_word(monkeypatch, mark):
    """Whisper's last unit ends at the window's end, which is the audio's (3 s): the mark after it takes that end
    (`_fill`) and the line DP folds it into the word before instead of dropping it."""
    from finesub.speech.joint import lines as L

    events: list = []
    seg = dict(_timed_stub(monkeypatch, events), text=f"上涨 12 {mark}")  # units 上 / 涨 / 12 / mark, the last untimed
    monkeypatch.setattr(A, "_kept", lambda unit: "" if unit == mark else unit)
    monkeypatch.setattr(A, "_aligned", lambda segments, todo, all_segments, wav, total, device: {
        0: A._Timed(1.0, 3.0, np.array([0.0, 0.5, 0.5, 1.0, 1.0, 1.5]), np.array([0.9, 0.3, 0.3, 0.3, 0.3, 0.3]))})
    monkeypatch.setattr(W, "retime", lambda jobs, wav, how: (
        {0: np.array([1.0, 1.5, 1.5, 2.5, 2.5, 3.0])}, {"segments": 1, "boundaries": 4}))
    got = A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16), retime=W.Retime("zh", False))
    assert got[0][-1]["text"] == mark and got[0][-1]["start"] == pytest.approx(3.0)
    assert "".join(w["word"] for w in L.dp_words(got[0], duration=3.0)).replace(" ", "") == f"上涨12{mark}"


def test_past_the_cut_a_mark_after_a_dropped_word_goes_with_it_not_onto_the_word_before():
    from finesub.speech.joint import lines as L

    units = [{"start": 2.0, "end": 2.5, "text": "12"}, {"start": 3.1, "end": 3.4, "text": "34"},
             {"start": 3.4, "end": 3.4, "text": "%"}]
    assert [w["word"] for w in L.dp_words(units, duration=3.0)] == ["12"]
    timed_mark = [{"start": 2.0, "end": 3.0, "text": "12"}, {"start": 3.0, "end": 3.2, "text": "!"}]
    assert [w["word"] for w in L.dp_words(timed_mark, duration=3.0)] == ["12"]  # a mark with time of its own past it
    spanning = [{"start": 2.0, "end": 3.4, "text": "12"}, {"start": 3.1, "end": 3.2, "text": "!"}]
    assert [w["word"] for w in L.dp_words(spanning, duration=3.0)] == ["12"]  # inside the word before, still timed


def test_a_stop_raised_while_handling_a_decode_error_frees_what_its_chain_held(monkeypatch):
    import gc
    import weakref

    events: list = []
    seg = _timed_stub(monkeypatch, events)

    class Model:
        pass

    refs = []

    def stopped(jobs, wav, how):
        model = Model()
        refs.append(weakref.ref(model))
        try:
            raise RuntimeError("decode failed")
        except RuntimeError:
            raise SystemExit(2)  # its __context__ is the decode error, whose frame holds `model`

    alive = []

    def release():
        gc.collect()
        alive.append(refs[0]() is not None)

    monkeypatch.setattr(W, "retime", stopped)
    monkeypatch.setattr(W, "release", release)
    with pytest.raises(SystemExit):
        A.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(48000, np.int16), retime=W.Retime("ja", False))
    assert alive == [False] and events[-1] == "gate released"
