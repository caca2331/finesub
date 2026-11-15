"""How the recall repair asks its models (`speech/verification/recall_decode`): what is transcribed, in which order,
and when Whisper is loaded at all. The models are stand-ins; the rules' own decisions are `test_recall_repair.py`'s."""

from __future__ import annotations

import numpy as np
import pytest

from finesub.speech.postprocessing import row_fill as rf
from finesub.speech.verification import recall_decode as rd
from .test_recall_repair import _payload, _row, needs_readings

LINE = "今日はいい天気ですね"


def _models(said):
    """A referee answering `said(span)` and a Whisper echoing the referee, both logging their calls."""
    log = []

    def referee(spans, label):
        log.append(("referee", list(spans)))
        return [said(s) for s in spans]

    def whisper():
        log.append(("load whisper", None))

        def transcribe(spans):
            log.append(("whisper", list(spans)))
            return [(said(s)[0], "ja") for s in spans]
        return transcribe
    return referee, whisper, log


@needs_readings
def test_whisper_is_loaded_last_and_hears_only_the_gaps_that_need_it() -> None:
    payload = _payload([_row("おはようございます", 0.0, 1.5)], [(0.0, 1.5), (10.0, 12.0), (20.0, 21.5)])
    referee, whisper, log = _models(lambda s: (LINE, "Japanese") if s == (10.0, 12.0) else ("うん", "Japanese"))
    rows, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=False, gap=True,
                             referee=referee, whisper=whisper)
    assert [kind for kind, _ in log] == ["referee", "referee", "load whisper", "whisper"]
    assert log[0][1] == [(10.0, 12.0), (20.0, 21.5)] and log[-1][1] == [(10.0, 12.0)]
    assert [r["text"] for r in rows] == ["おはようございます", LINE]
    assert [(g["start"], g["whisper"], g["heard"]) for g in record["gaps"]] == [(10.0, LINE, True), (20.0, None, False)]
    assert (record["inserted"], record["moved"]) == (1, 0)


def test_without_a_gap_worth_asking_whisper_is_never_loaded() -> None:
    payload = _payload([_row("おはようございます", 0.0, 1.5)], [(0.0, 1.5), (10.0, 12.0)])
    referee, whisper, log = _models(lambda s: ("うん", "Japanese"))
    rows, _ = rd.repair(payload, payload["segments"], lang="ja", squeeze=False, gap=True,
                        referee=referee, whisper=whisper)
    assert "load whisper" not in [kind for kind, _ in log]
    assert rows == payload["segments"]


@needs_readings
def test_gaps_are_taken_after_the_collapse_repair() -> None:
    text = "させていただきましょうそれでは"
    payload = _payload([_row(text, 10.0, 10.24), _row("次です", 20.0, 21.0)], [(10.0, 15.0), (20.0, 21.0)])
    # The referee hears the squeezed row's text everywhere after its start: the stretch goes through.
    referee, whisper, log = _models(lambda s: (text, "Japanese") if s[0] >= 10.0 else ("", None))
    rows, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=True, gap=True,
                             referee=referee, whisper=whisper)
    assert record["squeezed"][0]["branch"] == "stretch" and record["stretched"] == 1
    stretched_end = rows[0]["end"]
    gap_spans = log[1][1]
    assert all(a >= stretched_end - 1e-6 for a, _ in gap_spans)


def test_only_clips_past_the_audio_are_left_out_of_a_batch() -> None:
    class Reader:
        def read(self, start, end):
            return np.zeros(int(max(0.0, min(end, 30.0) - start) * 16000), dtype=np.float32)

    asked = []
    transcribe = rd._transcriber(Reader(), lambda clips: asked.extend(clips) or [("x", "ja")] * len(clips))
    # A 10 ms row is still in the batch: the batch's members change the referee's text.
    assert transcribe([(1.0, 2.0), (5.0, 5.01), (30.5, 31.0)]) == [("x", "ja"), ("x", "ja"), ("", None)]
    assert len(asked) == 2


def _slow_start(row):
    """`row` with its first word drawn out over a second and every confidence low: what the row fill's gate keeps."""
    words = [dict(w, confidence=0.3) for w in row["words"]]
    words[0]["end"] = words[1]["start"] = row["start"] + 1.0
    return {**row, "words": words}


@needs_readings
def test_the_row_fill_is_heard_before_whisper_loads_once_for_gaps_and_rows() -> None:
    row = _slow_start(_row("おはようございます", 0.0, 1.5))
    payload = _payload([row], [(0.0, 1.5), (10.0, 12.0)])
    said = {(0.0, 1.5): ("おはようございますまたあした", "Japanese"), (10.0, 12.0): (LINE, "Japanese")}
    referee, whisper, log = _models(lambda s: said[s])
    rows, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=False, gap=True, row_gate=rf.GATE_THRESHOLD,
                             referee=referee, whisper=whisper)
    assert [kind for kind, _ in log] == ["referee", "referee", "referee", "load whisper", "whisper"]
    assert log[2][1] == [(0.0, 1.5)] and log[-1][1] == [(10.0, 12.0), (0.0, 1.5)]
    assert [r["text"] for r in rows] == ["おはようございますまたあした", LINE]
    assert {k: record["row_fill"][k] for k in ("gate", "rows", "whisper", "moved", "filled", "pieces")} == {
        "gate": rf.GATE_THRESHOLD, "rows": 1, "whisper": 1, "moved": 0, "filled": 1, "pieces": 1}


@needs_readings
def test_the_row_fill_reads_the_gap_fill_s_lines_as_neighbours_and_skips_the_rows_it_moved() -> None:
    a, b = _row("おはようございます", 0.0, 1.5), _row("こんばんは", 4.0, 5.0)
    filling = {k: {"start": r["start"], "end": r["end"], "qwen": r["text"] + "またあした", "language": "Japanese",
                   "whisper": r["text"] + "またあした"} for k, r in enumerate((a, b))}
    line = {"start": 1.6, "end": 3.0, "text": "またあしたね", "words": [], "gap_fill": True, "whisper": "またあしたね"}
    moved = {**b, "start": 3.5, "gap_retime": {"start": 4.0, "end": 5.0}}
    rows, record = rd._fill_rows([a, b], [a, line, moved], filling, "ja")
    assert rows == [a, line, moved]
    assert (record["moved"], record["filled"], record["pieces_neighbour"]) == (1, 0, 1)


@needs_readings
def test_the_row_fill_finds_its_rows_again_when_the_gap_fill_sorts_rows_stabilize_left_out_of_order() -> None:
    a, b = _slow_start(_row("おはようございます", 2.0, 3.5)), _slow_start(_row("こんばんは", 1.0, 1.9))
    payload = _payload([a, b], [(1.0, 3.5), (10.0, 12.0)])
    said = {(2.0, 3.5): ("おはようございますまたあした", "Japanese"), (1.0, 1.9): ("こんばんは", "Japanese"),
            (10.0, 12.0): (LINE, "Japanese")}
    referee, whisper, _ = _models(lambda s: said[s])
    rows, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=False, gap=True, row_gate=rf.GATE_THRESHOLD,
                             referee=referee, whisper=whisper)
    assert [r["text"] for r in rows] == ["こんばんは", "おはようございますまたあした", LINE]
    assert record["row_fill"]["filled"] == 1


@needs_readings
def test_the_row_fill_hears_the_rows_the_collapse_repair_left_and_not_the_stretched_ones() -> None:
    text = "させていただきましょうそれでは"
    slow = _slow_start(_row("おはようございます", 30.0, 31.5))
    payload = _payload([_row(text, 10.0, 10.24), slow], [(10.0, 15.0), (30.0, 31.5)])
    referee, whisper, log = _models(lambda s: (text, "Japanese") if 10.0 <= s[0] < 30.0 else ("おはようございます", "Japanese"))
    rows, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=True, gap=False, row_gate=rf.GATE_THRESHOLD,
                             referee=referee, whisper=whisper)
    assert record["stretched"] == 1 and log[-1] == ("referee", [(30.0, 31.5)])
    assert record["row_fill"]["rows"] == 1


@needs_readings
def test_the_row_fill_s_auto_gate_hears_only_the_likelier_rows() -> None:
    likely = _slow_start(_row("おはようございます", 0.0, 1.5))  # gate score 0.63
    middling = _row("こんばんは", 3.0, 4.5)  # 0.31: over the full gate, under auto's
    middling["words"] = [dict(w, confidence=0.5) for w in middling["words"]]
    middling["words"][0]["end"] = middling["words"][1]["start"] = 3.6
    payload = _payload([likely, middling], [(0.0, 1.5), (3.0, 4.5)])
    for gate, heard in ((rf.GATE_THRESHOLD, [(0.0, 1.5), (3.0, 4.5)]), (rf.AUTO_THRESHOLD, [(0.0, 1.5)])):
        referee, whisper, log = _models(lambda s: ("おはようございます" if s[0] < 2 else "こんばんは", "Japanese"))
        _, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=False, gap=False, row_gate=gate,
                              referee=referee, whisper=whisper)
        assert log == [("referee", heard)] and record["row_fill"]["gate"] == gate


class _Reporter:
    def __init__(self):
        self.warnings = []

    def warning(self, code, message, **kw):
        self.warnings.append(code)

    def progress(self, *a, **kw):
        pass


def _no_devices():
    raise AssertionError("a skipped repair asked for devices (that imports CTranslate2)")


def _hook(monkeypatch, tmp_path, *, squeeze="auto", gap="auto", qwen_verify="auto", vocals=True, devices=_no_devices,
          row_fill="off", vad_prefix=None):
    reporter = _Reporter()
    monkeypatch.setattr(rd, "current_reporter", lambda: reporter)
    audio = tmp_path / "x-vocal.ogg"
    if vocals:
        audio.write_bytes(b"")
    return rd.recall_repair(audio_path=audio, squeeze=squeeze, gap=gap, qwen_verify=qwen_verify,
                            devices=devices, row_fill=row_fill, vad_prefix=vad_prefix), reporter


def _ja_payload():
    payload = _payload([{**_row("おはようございます", 0.0, 1.5), "lang": "ja"}], [(0.0, 1.5), (10.0, 12.0)])
    payload["metadata"]["asr_align"]["language"] = "None"  # detected: the rows say ja
    return payload


def test_switches_off_build_no_hook_and_auto_follows_qwen_verify_off(monkeypatch, tmp_path) -> None:
    assert _hook(monkeypatch, tmp_path, squeeze="off", gap="off")[0] is None
    assert _hook(monkeypatch, tmp_path, qwen_verify="off")[0] is None
    assert _hook(monkeypatch, tmp_path, squeeze="on", gap="off", qwen_verify="off")[0] is not None
    assert _hook(monkeypatch, tmp_path, squeeze="off", gap="off", row_fill="auto", qwen_verify="off")[0] is None
    assert _hook(monkeypatch, tmp_path, squeeze="off", gap="off", row_fill="auto")[0] is not None


def test_auto_skips_without_vocals_and_warns_while_on_raises(monkeypatch, tmp_path) -> None:
    payload = _ja_payload()
    hook, reporter = _hook(monkeypatch, tmp_path, vocals=False)
    rows, record = hook(payload, payload["segments"])
    assert rows == payload["segments"] and record["skipped"] == {"squeeze": "no vocals", "gap": "no vocals"}
    assert reporter.warnings == ["recall-repair-skipped"]
    hook, _ = _hook(monkeypatch, tmp_path, gap="on", vocals=False)
    with pytest.raises(RuntimeError, match="--gap-fill on"):
        hook(payload, payload["segments"])


def test_the_record_keeps_the_switches_in_effect(monkeypatch, tmp_path) -> None:
    payload = _ja_payload()
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="on", gap="auto", qwen_verify="off", vocals=False)
    with pytest.raises(RuntimeError, match="--squeeze-repair on"):
        hook(payload, payload["segments"])
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="auto", gap="off", vocals=False)
    _, record = hook(payload, payload["segments"])
    assert record["switches"] == {"squeeze": "auto", "gap": "off"} and record["skipped"] == {"squeeze": "no vocals"}
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="off", gap="off", row_fill="auto", vocals=False)
    _, record = hook(payload, payload["segments"])
    assert record["switches"] == {"squeeze": "off", "gap": "off", "row_fill": "auto"}
    assert record["skipped"] == {"row_fill": "no vocals"}


def test_an_old_aligned_without_a_vad_timeline_is_skipped_quietly(monkeypatch, tmp_path) -> None:
    payload = _ja_payload()
    del payload["vad_timeline"]
    hook, reporter = _hook(monkeypatch, tmp_path)
    _, record = hook(payload, payload["segments"])
    assert record["skipped"]["gap"] == "no vad_timeline" and reporter.warnings == []


def test_rows_naming_no_language_skip_even_under_on(monkeypatch, tmp_path) -> None:
    payload = _payload([_row("あいう", 0.0, 1.0)], [(0.0, 1.0)])
    monkeypatch.setattr(rd, "_missing", lambda payload, audio: None)
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="on", gap="on")
    rows, record = hook(payload, payload["segments"])
    assert rows == payload["segments"] and record["skipped"] == {"squeeze": "no language", "gap": "no language"}


def _models_that(monkeypatch, *, load_fails: bool):
    class Referee:
        effective_device = "cuda"

        def __init__(self, *a, **kw):
            pass

        def warm(self):
            if load_fails:
                raise OSError("weights unavailable")

        def transcribe_batch(self, clips, **kw):
            raise RuntimeError("CUDA out of memory")

        def close(self):
            pass

    class Reader:
        def __init__(self, path):
            pass

        def read(self, start, end):
            return np.zeros(int((end - start) * 16000), dtype=np.float32)

    monkeypatch.setattr(rd, "_missing", lambda payload, audio: None)
    monkeypatch.setattr(rd, "_whisper_weights", lambda model: None)
    monkeypatch.setattr(rd, "QwenReferee", Referee)
    monkeypatch.setattr(rd, "SpanReader", Reader)


CUDA = lambda: ("cuda", "cuda")


def test_a_referee_that_will_not_load_skips_under_auto_and_raises_under_on(monkeypatch, tmp_path) -> None:
    _models_that(monkeypatch, load_fails=True)
    payload = _ja_payload()
    hook, reporter = _hook(monkeypatch, tmp_path, devices=CUDA)
    rows, record = hook(payload, payload["segments"])
    assert rows == payload["segments"] and record["skipped"] == {"squeeze": "load failed", "gap": "load failed"}
    assert reporter.warnings == ["recall-repair-skipped"]
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="on", devices=CUDA)
    with pytest.raises(RuntimeError, match="weights unavailable"):
        hook(payload, payload["segments"])


def test_a_decode_that_fails_raises_under_auto_too(monkeypatch, tmp_path) -> None:
    _models_that(monkeypatch, load_fails=False)
    payload = _ja_payload()
    hook, _ = _hook(monkeypatch, tmp_path, devices=CUDA)
    with pytest.raises(RuntimeError, match="out of memory"):
        hook(payload, payload["segments"])


def test_without_whisper_s_model_only_the_gap_fill_is_skipped(monkeypatch, tmp_path) -> None:
    _models_that(monkeypatch, load_fails=False)
    squeezed = {**_row("させていただきましょうそれでは", 10.0, 10.24), "lang": "ja"}
    payload = _payload([squeezed], [(10.0, 15.0), (20.0, 22.0)])
    del payload["metadata"]["asr_align"]["model"]
    hook, _ = _hook(monkeypatch, tmp_path, devices=CUDA)
    with pytest.raises(RuntimeError, match="out of memory"):  # the collapse repair went on to decode
        hook(payload, payload["segments"])
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="off", gap="auto")
    _, record = hook(payload, payload["segments"])
    assert record["skipped"] == {"gap": "no asr model"}


def test_nothing_to_hear_loads_no_model(monkeypatch, tmp_path) -> None:
    _models_that(monkeypatch, load_fails=True)
    payload = _payload([{**_row("おはようございます", 0.0, 1.5), "lang": "ja"}], [(0.0, 1.5)])
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="on", gap="on")
    rows, record = hook(payload, payload["segments"])
    assert rows == payload["segments"] and "skipped" not in record


def test_the_stage_builds_the_hook_only_for_profile_0_with_a_switch_on(monkeypatch, tmp_path) -> None:
    from finesub import pipeline, stages

    paths = stages.default_pipeline_paths(tmp_path / "x.wav", tmp_path / "out" / "x.srt")
    kw = dict(qwen_verify="auto", device="cpu", gpu_tier="cpu")
    assert stages._recall_repair(paths, squeeze="on", gap="on", profile=2, row_fill="off", **kw) is None
    assert stages._recall_repair(paths, squeeze="off", gap="off", profile=0, row_fill="off", **kw) is None
    assert stages._recall_repair(paths, squeeze="auto", gap="off", profile=0, row_fill="off", **kw) is not None
    assert stages._recall_repair(paths, squeeze="off", gap="off", row_fill="on", profile=0, **kw) is not None
    # the ladder reads the ASR step's VAD prefix of this task
    from finesub.speech.verification import recall_decode
    seen = {}
    monkeypatch.setattr(recall_decode, "recall_repair", lambda **k: seen.update(k) or "hook")
    stages._recall_repair(paths, squeeze="auto", gap="off", profile=0, row_fill="off", **kw)
    assert seen["vad_prefix"] == paths.vad_json
    args = pipeline.build_parser().parse_args(["x.wav", "--squeeze-repair", "off", "--gap-fill", "on",
                                               "--row-fill", "auto"])
    assert (args.squeeze_repair, args.gap_fill, args.row_fill) == ("off", "on", "auto")
    assert pipeline.build_parser().parse_args(["x.wav"]).row_fill is None
    assert {"squeeze_repair", "gap_fill", "row_fill"} <= pipeline._PIPELINE_PARAMS


def test_with_both_switches_off_the_stable_step_writes_what_dev_wrote_before_the_repair(tmp_path) -> None:
    """The fixture: two minutes of a real aligned (three squeezed rows in it) and what the stable step of dev made of
    it at the commit this work branched from (7346ffe7)."""
    from pathlib import Path

    from finesub import stages
    from finesub.speech.postprocessing import stabilization

    fixtures = Path(__file__).resolve().parent / "fixtures"
    paths = stages.default_pipeline_paths(tmp_path / "x.wav", tmp_path / "x.srt")
    hook = stages._recall_repair(paths, squeeze="off", gap="off", qwen_verify="auto", device="cpu", gpu_tier="cpu",
                                 profile=0, row_fill="off")
    out, _ = stabilization.stabilize_json_file(fixtures / "recall-off-aligned.json", output_path=tmp_path / "s.json",
                                               repair=hook)
    assert out.read_bytes() == (fixtures / "recall-off-stable.json").read_bytes()


def test_a_row_fill_asked_for_raises_without_its_prerequisites(monkeypatch, tmp_path) -> None:
    payload = _ja_payload()
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="off", gap="off", row_fill="on", vocals=False)
    with pytest.raises(RuntimeError, match="--row-fill on"):
        hook(payload, payload["segments"])


def test_a_row_fill_with_no_row_through_the_gate_loads_no_model(monkeypatch, tmp_path) -> None:
    _models_that(monkeypatch, load_fails=True)
    payload = _payload([{**_row("おはようございます", 0.0, 1.5), "lang": "ja"}], [(0.0, 1.5)])
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="off", gap="off", row_fill="on")
    rows, record = hook(payload, payload["segments"])
    assert rows == payload["segments"] and "skipped" not in record


def test_without_whisper_s_model_a_row_fill_asked_for_names_only_itself(monkeypatch, tmp_path) -> None:
    _models_that(monkeypatch, load_fails=False)
    payload = _ja_payload()
    del payload["metadata"]["asr_align"]["model"]
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="off", gap="auto", row_fill="on")
    with pytest.raises(RuntimeError, match="^--row-fill on："):
        hook(payload, payload["segments"])


def test_whisper_s_weights_failing_raises_for_a_row_fill_asked_for(monkeypatch, tmp_path) -> None:
    _models_that(monkeypatch, load_fails=False)

    def fail(model):
        raise OSError("no weights")
    monkeypatch.setattr(rd, "_whisper_weights", fail)
    payload = _payload([_slow_start({**_row("おはようございます", 0.0, 1.5), "lang": "ja"})], [(0.0, 1.5)])
    hook, _ = _hook(monkeypatch, tmp_path, squeeze="off", gap="off", row_fill="on")
    with pytest.raises(RuntimeError, match="--row-fill.*no weights"):
        hook(payload, payload["segments"])


SQUEEZED = "させていただきましょうそれでは"
#: Whisper's decode of the squeezed row's `pad` window: the row's text over 10.0-12.2.
PAD_WORDS = [("させて", 10.0, 10.6), ("いただき", 10.6, 11.2), ("ましょう", 11.2, 11.7), ("それでは", 11.7, 12.2)]


def _ladder_case(deaf: tuple | None = None):
    """A squeezed row at 10.0, the decoded `pad` window, and a referee hearing the row's text over every span that
    starts in 10-20 s but `deaf`."""
    from finesub.speech.postprocessing import collapse_ladder as cl

    payload = _payload([_row(SQUEEZED, 10.0, 10.24), _row("次です", 20.0, 21.0)], [(10.0, 15.0), (20.0, 21.0)])
    pad = next(w for w in cl.windows(payload, payload["segments"]) if w["kind"] == "pad")
    decoded = {("pad", 0): {**pad, "segments": [{"words": [{"word": a, "start": s, "end": e} for a, s, e in PAD_WORDS]}]}}
    referee, whisper, log = _models(lambda s: (SQUEEZED, "Japanese") if 10.0 <= s[0] < 20.0 and s != deaf
                                    else ("", None))
    return payload, decoded, referee, whisper, log


@needs_readings
def test_the_ladder_takes_whisper_s_pad_span_once_the_referee_hears_the_row_there() -> None:
    payload, decoded, referee, whisper, log = _ladder_case()
    rows, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=True, gap=False,
                             referee=referee, whisper=whisper, decoded=decoded)
    # two referee batches (the collapse windows, then every distinct proposal), and no plain Whisper
    assert [kind for kind, _ in log] == ["referee", "referee"]
    assert (10.0, 12.2) in log[1][1]
    assert record["squeezed"] == [{"index": 0, "start": 10.0, "end": 10.24, "branch": "pad", "span": [10.0, 12.2],
                                   "guards": {}}]
    assert record["stretched"] == 1 and (rows[0]["start"], rows[0]["end"]) == (10.0, 12.2)


@needs_readings
def test_a_proposal_the_referee_does_not_hear_leaves_the_row_to_the_next_proposer() -> None:
    payload, decoded, referee, whisper, _ = _ladder_case(deaf=(10.0, 12.2))
    rows, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=True, gap=False,
                             referee=referee, whisper=whisper, decoded=decoded)
    verdict = record["squeezed"][0]
    assert verdict["branch"] == "S" and verdict["guards"] == {"pad": "check"}
    assert rows[0]["start"] == 10.0 and rows[0]["end"] < 12.0


def _ladder_hook(monkeypatch, *, prefix=(object(), None)):
    """The hook's models as loggers: the ladder's decode, the referee's load and its batches."""
    log = []

    class Referee:
        effective_device = "cuda"

        def __init__(self, *a, **kw):
            log.append("referee load")

        def warm(self):
            pass

        def transcribe_batch(self, clips, **kw):
            return [("", None)] * len(clips)

        def close(self):
            log.append("referee close")

    class Reader:
        def __init__(self, path):
            pass

        def read(self, start, end):
            return np.zeros(int((end - start) * 16000), dtype=np.float32)

    def decode(payload, rows, prefix, **kw):
        log.append(("ladder decode", kw["model_name"]))
        return {}
    monkeypatch.setattr(rd, "_missing", lambda payload, audio: None)
    monkeypatch.setattr(rd, "_whisper_weights", lambda model: None)
    monkeypatch.setattr(rd, "QwenReferee", Referee)
    monkeypatch.setattr(rd, "SpanReader", Reader)
    monkeypatch.setattr(rd, "_ladder_prefix", lambda payload, path: prefix)
    monkeypatch.setattr(rd, "_ladder_decode", decode)
    payload = _payload([{**_row(SQUEEZED, 10.0, 10.24), "lang": "ja"}], [(10.0, 15.0)])
    return payload, log


@needs_readings
def test_the_ladder_s_whisper_runs_before_the_referee_loads(monkeypatch, tmp_path) -> None:
    payload, log = _ladder_hook(monkeypatch)
    hook, reporter = _hook(monkeypatch, tmp_path, gap="off", devices=CUDA)
    _, record = hook(payload, payload["segments"])
    assert log[:2] == [("ladder decode", "test"), "referee load"]
    assert record["collapse"] == {"repair": "ladder", "windows": 0} and reporter.warnings == []


def test_in_any_language_the_ladder_and_the_row_fill_run(monkeypatch, tmp_path) -> None:
    _, log = _ladder_hook(monkeypatch)
    payload = _payload([{**_row("안녕하세요여러분오늘도반갑습니다", 10.0, 10.24), "lang": "ko"}], [(10.0, 15.0)])
    hook, reporter = _hook(monkeypatch, tmp_path, gap="off", devices=CUDA, row_fill="on")
    _, record = hook(payload, payload["segments"])
    assert record["collapse"]["repair"] == "ladder" and "skipped" not in record and reporter.warnings == []
    assert log[0] == ("ladder decode", "test")


@needs_readings
def test_the_collapse_repair_falls_back_to_the_stretch_without_a_matching_vad_prefix(monkeypatch, tmp_path) -> None:
    payload, log = _ladder_hook(monkeypatch, prefix=(None, "vad prefix of other audio"))
    hook, reporter = _hook(monkeypatch, tmp_path, gap="off", devices=CUDA)
    _, record = hook(payload, payload["segments"])
    assert record["collapse"] == {"repair": "stretch", "fallback": "vad prefix of other audio"}
    assert "referee load" in log and not any(isinstance(x, tuple) for x in log)
    assert reporter.warnings == ["collapse-ladder-fallback"]


def test_the_ladder_reads_the_vad_prefix_only_when_it_is_the_aligned_s(tmp_path) -> None:
    import torch

    from finesub.speech.preprocessing.energy import VadEnergyTrack
    from finesub.speech.recognition.vad_asr_stage import VadPrefix, write_vad_prefix

    source = tmp_path / "x.wav"
    source.write_bytes(b"x")
    track = VadEnergyTrack(energy_db=torch.zeros(4), hop_sec=0.01, frame_sec=0.02, energy_mode="weighted")
    prefix = VadPrefix(raw_segments=[], segments=[{"start": 10.0, "end": 15.0}], vad_meta={}, audio_duration=20.0,
                       timing={}, energy_track=track)
    path = tmp_path / "x-vad.json"
    write_vad_prefix(path, prefix, source_path=source, vad_silero_assist=True)
    same = _payload([], [(10.0, 15.0)])
    got, why = rd._ladder_prefix(same, path)
    assert why is None and got.segments == prefix.segments and got.audio_duration == 20.0
    assert rd._ladder_prefix(_payload([], [(10.0, 14.0)]), path) == (None, "vad prefix of other audio")
    assert rd._ladder_prefix(same, tmp_path / "missing-vad.json") == (None, "no vad prefix")
    assert rd._ladder_prefix(same, None) == (None, "no vad prefix")
    path.with_name("x-vad-energy.npz").unlink()  # the prefix's energy sidecar
    assert rd._ladder_prefix(same, path)[1].startswith("vad prefix unreadable")


def test_the_ladder_decodes_pad_first_then_iso_and_ctx_for_the_rows_pad_missed_in_one_load(monkeypatch) -> None:
    pytest.importorskip("faster_whisper", reason="[asr] extra not installed")
    from finesub.speech.recognition import fw_refine_backend, transcribe, vad_asr_stage

    calls = []

    class Pool:
        def __init__(self, model_name, **kw):
            calls.append(("pool", model_name))

        def lease(self):
            import contextlib

            @contextlib.contextmanager
            def lease():
                calls.append("lease")
                yield "model"
            return lease()

        def close(self):
            calls.append("pool closed")

    class Loader:
        def __init__(self, path, **kw):
            pass

        def close(self):
            calls.append("loader closed")

    def decode(intervals, *, model, gap_sec, **kw):
        calls.append(("decode", intervals[0]["start"], gap_sec))
        if intervals[0]["start"] == 10.0:  # row 0's windows hear it; row 2's hear nothing
            return [{"words": [{"word": a, "start": s, "end": e} for a, s, e in PAD_WORDS]}]
        return []
    monkeypatch.setattr(fw_refine_backend, "FwRefineModelPool", Pool)
    monkeypatch.setattr(transcribe, "AudioBlockLoader", Loader)
    monkeypatch.setattr(transcribe, "decode_intervals", decode)
    monkeypatch.setattr(vad_asr_stage, "finish_decoded", lambda segments, **kw: segments)
    monkeypatch.setattr(rd, "current_reporter", lambda: _Reporter())
    rows = [_row(SQUEEZED, 10.0, 10.24), _row("次です", 20.0, 21.0), _row(SQUEEZED, 30.0, 30.24)]
    payload = _payload(rows, [(10.0, 15.0), (20.0, 21.0), (30.0, 35.0)])
    payload["metadata"]["asr_align"]["gap_sec"] = 0.5
    prefix = vad_asr_stage.VadPrefix(raw_segments=[], segments=[], vad_meta={}, audio_duration=40.0, timing={},
                                     energy_track=None)
    out = rd._ladder_decode(payload, rows, prefix, audio_path="x.ogg", model_name="m", device="cuda", lang="ja",
                            weights=None)
    assert set(out) == {("pad", 0), ("pad", 2), ("iso", 2), ("ctx", 2)}
    assert calls[:2] == [("pool", "m"), "lease"] and calls.count("lease") == 1
    # pad, pad, then row 2's iso and ctx (its wide window reaches back to the interval at 20 s)
    assert [c[1] for c in calls if c[0] == "decode"] == [10.0, 30.0, 30.0, 20.0]
    assert {c[2] for c in calls if c[0] == "decode"} == {0.5}  # the ASR step's gap, as the aligned recorded it
    assert calls[-2:] == ["loader closed", "pool closed"]

    def fails(intervals, **kw):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(transcribe, "decode_intervals", fails)
    calls.clear()
    with pytest.raises(RuntimeError, match="out of memory"):
        rd._ladder_decode(payload, rows, prefix, audio_path="x.ogg", model_name="m", device="cuda", lang="ja",
                          weights=None)
    assert calls[-2:] == ["loader closed", "pool closed"]


@needs_readings
def test_whisper_s_weights_failing_falls_back_to_the_stretch_even_under_on(monkeypatch, tmp_path) -> None:
    payload, log = _ladder_hook(monkeypatch)

    def fail(model):
        raise OSError("no weights")
    monkeypatch.setattr(rd, "_whisper_weights", fail)
    for mode in ("auto", "on"):
        hook, reporter = _hook(monkeypatch, tmp_path, squeeze=mode, gap="off", devices=CUDA)
        _, record = hook(payload, payload["segments"])
        assert record["collapse"] == {"repair": "stretch", "fallback": "load failed: OSError: no weights"}
        assert reporter.warnings == ["collapse-ladder-fallback"] and "skipped" not in record


@needs_readings
def test_an_empty_set_of_decodes_is_still_the_ladder() -> None:
    payload, _, referee, whisper, log = _ladder_case()
    _, record = rd.repair(payload, payload["segments"], lang="ja", squeeze=True, gap=False,
                          referee=referee, whisper=whisper, decoded={})
    assert record["squeezed"][0]["branch"] == "S" and len(log) == 2  # the stretch's proposal, checked
