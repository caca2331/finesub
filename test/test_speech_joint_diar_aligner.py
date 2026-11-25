"""`--asr moss-diarization-aligner`: a word backend's lines, each line's speaker chosen by local MOSS's forced decode.
Light: the decode is faked (`moss_local.label_rows`), and the word backend's transcript too."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from finesub import settings
from finesub.speech.joint import lines as L
from finesub.speech.joint import moss_local
from finesub.speech.joint import stage as joint_stage
from finesub.speech.joint.model import UNKNOWN_SPEAKER
from finesub.speech.postprocessing import segmentation as seg

from .test_speech_joint_stage import cloud_source  # noqa: F401 -- the fixture

ALIGNER = "moss-diarization-aligner"


def _fake_labels(monkeypatch, speakers=None, margins=None):
    seen = []

    def label_rows(source, lines, *, language, gpu_tier, device, speech):
        seen.append(len(lines))
        n = len(lines)
        who = speakers or [f"S{1 + (i % 2):02d}" for i in range(n)]
        sure = margins or [None] + [1.5] * (n - 1)
        return moss_local.Labels(list(who[:n]), list(sure[:n]),
                                 [{"lo": 0.0, "hi": 8.0, "kv": "bf16", "rows": []}], [], 0, "rev1")

    monkeypatch.setattr(moss_local, "label_rows", label_rows)
    monkeypatch.setattr(moss_local, "unusable_reason", lambda backend, tier, dev: None)
    return seen


def test_the_lines_keep_their_text_and_take_moss_speakers(cloud_source, tmp_path, monkeypatch, capsys):  # noqa: F811
    from finesub.stages import run_pipeline

    source, calls = cloud_source
    seen = _fake_labels(monkeypatch)
    paths = run_pipeline(source, output_path=tmp_path / "talk.srt", stage="raw-srt", asr=ALIGNER, asr_rows="mai2",
                         language="en", diarize="off")
    assert calls == ["mai2"] and seen
    aligned = json.loads(paths.aligned_json.read_text(encoding="utf-8"))
    record = aligned["metadata"]["joint_asr"]
    assert (record["backend"], record["rows"], record["rows_model"], record["rows_source"]) == \
        (ALIGNER, "mai2", "fake-model", "cache")
    assert record["model"] == moss_local.MODEL_NAME and record["revision"] == "rev1"
    assert record["parts"] == [[0.0, 8.0, "bf16"]] and record["unlabelled"] == 0 and "overlap_cues" in record
    stable = json.loads(paths.stable_json.read_text(encoding="utf-8"))["segments"]
    assert {line["asr_speaker"] for line in stable} <= {"A", "B"}
    assert {line["line_speaker"] for line in stable} <= {"S01", "S02"}
    assert any(L.SPEAKER_MARGIN in line for line in stable)
    rows = json.loads(paths.diar_rows_json.read_text(encoding="utf-8"))
    assert any(L.SPEAKER_MARGIN in json.dumps(r) for r in rows["rows"])
    kept = json.loads(paths.moss_diar_json.read_text(encoding="utf-8"))
    assert set(kept) == {"rows", "parts", "voice_merges", "labels", "overlap_cues"}
    assert len(kept["labels"]) == len(kept["rows"]) and all("words" in r for r in kept["rows"])
    err = capsys.readouterr().err
    assert err.count("--diarize off does not apply") == 1
    # resumed: the aligned is reused, nothing decodes, no second warning
    run_pipeline(source, output_path=tmp_path / "talk.srt", stage="raw-srt", diarize="off")
    assert len(seen) == 1 and "--diarize off does not apply" not in capsys.readouterr().err


def test_no_rows_backend_is_an_error_when_the_aligned_is_made(cloud_source, tmp_path, monkeypatch):  # noqa: F811
    from finesub.stages import run_pipeline

    _fake_labels(monkeypatch)
    with pytest.raises(joint_stage.JointAsrError, match="needs --asr-rows"):
        run_pipeline(cloud_source[0], output_path=tmp_path / "talk.srt", stage="raw-srt", asr=ALIGNER)
    with pytest.raises(joint_stage.JointAsrError, match="--asr-rows must be one of"):
        joint_stage.rows_backend("moss-pro")
    assert joint_stage.rows_backend("gemini") == "gemini-35t"


def test_rows_options_given_to_another_backend_are_refused_but_the_config_key_is_not(cloud_source, tmp_path,  # noqa: F811
                                                                                     monkeypatch):
    from finesub.stages import run_pipeline

    for given in ({"asr_rows": "mai2"}, {"asr_transcript": str(tmp_path / "x.json")}):
        with pytest.raises(ValueError, match="only applies to --asr moss-diarization-aligner") as refused:
            run_pipeline(cloud_source[0], output_path=tmp_path / "a" / "talk.srt", stage="raw-srt", asr="mai2",
                         **given)
        assert "products" not in str(refused.value)  # a new task has none
    monkeypatch.setattr(settings, "ASR_ROWS", replace(settings.ASR_ROWS, default="scribe"))  # as `[asr] rows`
    run_pipeline(cloud_source[0], output_path=tmp_path / "b" / "talk.srt", stage="raw-srt", asr="mai2")


def test_a_transcript_directory_of_another_recording_or_backend_is_refused(tmp_path):
    source = tmp_path / "talk.wav"
    source.write_bytes(b"x" * 100)
    cloud = tmp_path / "old-cloud"
    with pytest.raises(joint_stage.JointAsrError, match="does not exist"):
        joint_stage.rows_input(cloud, "mai2", source)
    cloud.mkdir()
    with pytest.raises(joint_stage.JointAsrError, match="holds no answers of mai2"):
        joint_stage.rows_input(cloud, "mai2", source)
    (cloud / "mai2").mkdir(parents=True)
    (cloud / "mai2" / "identity.json").write_text(json.dumps({"audio": "another"}), encoding="utf-8")
    with pytest.raises(joint_stage.JointAsrError, match="another recording"):
        joint_stage.rows_input(cloud, "mai2", source)
    from finesub.speech.joint.cache import audio_fingerprint

    (cloud / "mai2" / "identity.json").write_text(json.dumps({"audio": audio_fingerprint(source)}), encoding="utf-8")
    for given in (cloud, cloud / "mai2"):
        got = joint_stage.rows_input(given, "mai2", source)
        assert got.offline and got.cache_root == cloud and got.source == {"dir": str(given)}
    assert sorted(p.name for p in (cloud / "mai2").iterdir()) == ["identity.json"]  # read only


def test_a_transcript_file_is_one_answer_read_by_the_backend(tmp_path):
    source = tmp_path / "talk.wav"
    source.write_bytes(b"x" * 100)
    bad = tmp_path / "list.json"
    bad.write_text("[]", encoding="utf-8")
    with pytest.raises(joint_stage.JointAsrError, match="not a JSON object"):
        joint_stage.rows_input(bad, "scribe", source)
    good = tmp_path / "scribe.json"
    good.write_text(json.dumps({"words": [{"text": "hi", "start": 0.1, "end": 0.4}]}), encoding="utf-8")
    got = joint_stage.rows_input(good, "scribe", source)
    assert got.answer and not got.offline and got.source["file"] == str(good)
    from finesub.speech.joint import scribe

    assert scribe.answer_units(got.answer["words"])[0][0]["text"] == "hi"  # no `type`: a word


def test_a_transcript_file_with_no_words_stops(tmp_path, monkeypatch):
    from finesub.media import ffmpeg

    source = tmp_path / "talk.wav"
    source.write_bytes(b"x" * 100)
    empty = tmp_path / "mai2.json"
    empty.write_text(json.dumps({"words": []}), encoding="utf-8")
    monkeypatch.setattr(ffmpeg, "probe_media_duration", lambda src: 8.0)
    _fake_labels(monkeypatch)
    with pytest.raises(joint_stage.JointAsrError, match="holds no mai2 words"):
        joint_stage.run(source, backend=ALIGNER, output_path=tmp_path / "a.json", cache_root=tmp_path / "c",
                        language="en", rows="mai2", transcript=empty)
    # Scribe's shape given as MAI2's: MAI2 reads `word`, so every word has no text
    scribe_shaped = tmp_path / "scribe.json"
    scribe_shaped.write_text(json.dumps({"words": [{"text": "hi", "start": 0.1, "end": 0.4}]}), encoding="utf-8")
    with pytest.raises(joint_stage.JointAsrError, match="is --asr-rows the backend"):
        joint_stage.run(source, backend=ALIGNER, output_path=tmp_path / "a.json", cache_root=tmp_path / "c",
                        language="en", rows="mai2", transcript=scribe_shaped)


def test_a_mai2_transcript_file_leaves_the_tasks_cache_alone(tmp_path, monkeypatch):
    """The task's `<stem>-cloud/mai2/` holding another recording's answers is neither read, checked nor discarded
    when the rows come from a file."""
    from finesub.reporting import current_reporter
    from finesub.speech.joint import mai2

    source = tmp_path / "talk.wav"
    source.write_bytes(b"x" * 100)
    held = tmp_path / "c" / "mai2"
    held.mkdir(parents=True)
    (held / "identity.json").write_text(json.dumps({"audio": "another", "via": "azure"}), encoding="utf-8")
    (held / "w-0.0-8.0.raw.json").write_text("{}", encoding="utf-8")
    warned = []
    monkeypatch.setattr(current_reporter(), "warning", lambda *a, **k: warned.append(a))
    answer = {"words": [{"word": "hi", "start": 0.1, "end": 0.4, "speaker": 1}]}
    got = mai2.transcribe(source, cache_root=tmp_path / "c", language="en", via="", repair="off", duration=8.0,
                          answer=answer, speech=object())
    assert [u.text for u in got.units()] == ["hi"] and not warned
    assert sorted(p.name for p in held.iterdir()) == ["identity.json", "w-0.0-8.0.raw.json"]


def test_a_transcript_directory_missing_an_answer_stops_without_calling(tmp_path, monkeypatch):
    from finesub.media import ffmpeg
    from finesub.speech.joint.cache import audio_fingerprint

    source = tmp_path / "talk.wav"
    source.write_bytes(b"x" * 100)
    cloud = tmp_path / "old-cloud"
    (cloud / "scribe").mkdir(parents=True)
    (cloud / "scribe" / "identity.json").write_text(json.dumps({"audio": audio_fingerprint(source)}),
                                                    encoding="utf-8")
    monkeypatch.setattr(ffmpeg, "probe_media_duration", lambda src: 8.0)
    _fake_labels(monkeypatch)
    with pytest.raises(joint_stage.JointAsrError, match="lacks an answer this run needs"):
        joint_stage.run(source, backend=ALIGNER, output_path=tmp_path / "a.json", cache_root=tmp_path / "c",
                        language="en", rows="scribe", transcript=cloud)
    assert sorted(p.name for p in (cloud / "scribe").iterdir()) == ["identity.json"]  # nothing written there


def test_rows_fetched_through_the_api_are_recorded_as_such(cloud_source, tmp_path, monkeypatch):  # noqa: F811
    from finesub.stages import run_pipeline

    source, calls = cloud_source
    fake = joint_stage.transcribe

    def uploading(backend, src, *, uploads=None, **kwargs):
        uploads(lambda: None)()  # one request went out
        return fake(backend, src, **kwargs)

    monkeypatch.setattr(joint_stage, "transcribe", uploading)
    _fake_labels(monkeypatch)
    paths = run_pipeline(source, output_path=tmp_path / "talk.srt", stage="aligned", asr=ALIGNER, asr_rows="mai2",
                         language="en")
    record = json.loads(paths.aligned_json.read_text(encoding="utf-8"))["metadata"]["joint_asr"]
    assert record["rows_source"] == "api" and calls == ["mai2"]


def _line(start, end, text, who, margin=None, word=None):
    line = {"start": start, "end": end, "text": text, L.LINE_SPEAKER: who,
            "words": [{"start": start, "end": end, "word": word or text, "space_before": False}]}
    if margin is not None:
        line[L.SPEAKER_MARGIN] = margin
    return line


PARAMS = seg.DEFAULT_SPLIT_PARAMS


def test_an_overlap_cue_joins_one_person_around_an_interjection():
    lines = [_line(0.0, 2.0, "so I was", "S01", 2.0, "was"), _line(2.05, 2.4, "yeah", "S02"),
             _line(2.45, 4.0, "going home", "S01", 0.5, "going")]
    out, merges = L.overlap_cues(lines, params=PARAMS)
    assert merges == [[0, 2]]
    cue = next(x for x in out if x[L.LINE_SPEAKER] == "S01")
    assert (cue["start"], cue["end"], cue[L.SPEAKER_MARGIN]) == (0.0, 4.0, 0.5)
    assert cue["words"][1]["space_before"] and cue["text"] == "was going"  # English: a space at the seam
    assert any(x["text"] == "yeah" for x in out)  # the interjection stays, over the cue


@pytest.mark.parametrize("ending, joined", [("。", False), ("?", False), (".", False), ("...", True), ("、", True)])
def test_a_cue_ending_a_sentence_takes_no_further_piece(ending, joined):
    lines = [_line(0.0, 2.0, "そう" + ending, "S01"), _line(2.05, 2.4, "うん", "S02"), _line(2.45, 4.0, "だね", "S01")]
    assert bool(L.overlap_cues(lines, params=PARAMS)[1]) == joined


def test_unknown_lines_join_nothing_and_the_limits_scale_with_the_length():
    lines = [_line(0.0, 2.0, "aa", UNKNOWN_SPEAKER), _line(2.05, 2.4, "b", "S02"), _line(2.45, 4.0, "cc", UNKNOWN_SPEAKER)]
    assert L.overlap_cues(lines, params=PARAMS)[1] == []
    long = [_line(0.0, 5.0, "aa", "S01"), _line(5.05, 5.4, "b", "S02"), _line(5.45, 9.0, "cc", "S01")]
    assert L.overlap_cues(long, params=PARAMS)[1] == []  # 9 s > dur_ok_hi
    assert L.overlap_cues(long, params=seg.split_params_for_length_scale(1.25))[1] == [[0, 2]]


def test_relabelled_keeps_the_asr_speaker_and_redoes_the_timeline_per_speaker():
    lines = [_line(0.0, 2.0, "a", "A"), _line(1.5, 3.0, "b", "B")]  # two people's, overlapping
    out = L.relabelled(lines, ["S01", "S01"], [None, 0.7], duration=10.0)
    assert [x["asr_speaker"] for x in out] == ["A", "B"]
    assert out[0]["end"] <= out[1]["start"]  # now one person's: pulled apart
    assert L.SPEAKER_MARGIN not in out[0] and out[1][L.SPEAKER_MARGIN] == 0.7


def test_a_line_no_window_keeps_comes_from_the_one_holding_it():
    from finesub.speech.joint.stitch import stitch

    left = [{"row": 0, "start": 10.0, "end": 12.0, "speaker": "1"},
            {"row": 1, "start": 60.0, "end": 75.0, "speaker": "1"}]  # only the left part holds row 1
    right = [{"row": 2, "start": 80.0, "end": 82.0, "speaker": "1"}]
    got = stitch([(0.0, 76.0, left), (61.0, 140.0, right)], segments=False, orphans="row")
    assert sorted(u["row"] for u in got) == [0, 1, 2]
    without = stitch([(0.0, 76.0, left), (61.0, 140.0, right)], segments=False)
    assert len(without) <= len(got)


def test_label_rows_names_speakers_in_order_and_leaves_lines_no_part_held_unknown(tmp_path, monkeypatch):
    from finesub.speech.joint import relink
    from finesub.speech.joint.moss_local import forced

    lines = [{"start": 1.0, "end": 2.0, "text": "a"}, {"start": 3.0, "end": 4.0, "text": "b"},
             {"start": 5.0, "end": 900.0, "text": "long"}]
    parts = [forced.Part(0.0, 600.0, "bf16", [(0, 7, {7: -0.1}), (1, 3, {7: -2.0, 3: -0.5})])]
    monkeypatch.setattr(moss_local, "_prepare", lambda *a: (8.0, type("Plan", (), {"revision": "r"})()))
    monkeypatch.setattr(moss_local, "_decode", lambda source: None)
    monkeypatch.setattr(moss_local, "_run", lambda plan, budget, work: work(None, None))
    monkeypatch.setattr(forced, "label", lambda *a, **k: parts)
    monkeypatch.setattr(relink, "merge_by_voice", lambda *a, **k: ([], []))

    class Speech:
        def spans(self):
            return []

        def pcm(self):
            return None

    got = moss_local.label_rows(tmp_path / "x.wav", lines, language="ja", gpu_tier=None, device=None, speech=Speech())
    assert got.speakers == ["S01", "S02", UNKNOWN_SPEAKER] and got.unlabelled == 1
    assert got.margins == [None, 1.5, None]


def test_the_decoder_keeps_bf16_when_asked_whatever_moss_fast_kv_says(monkeypatch):
    from finesub.speech.joint.moss_local import decoder as D

    monkeypatch.setenv("MOSS_FAST_KV", "fp8")
    monkeypatch.setattr(D, "fits", lambda model, prompt, budget: {"bf16": 1000, "fp8": 2000})
    assert D.plan_kv(None, 100, 5000, 8.0, "bf16") == ("bf16", 1000)


def test_a_diarization_aligner_task_takes_the_gpu_lane_and_is_stopped_without_its_card(tmp_path, monkeypatch):
    from finesub.pipeline import build_item
    from finesub.scheduler import GPU_LANE
    from finesub.speech.joint.moss_local import decoder as D
    from finesub.speech.runtime import device as device_module

    monkeypatch.setattr(D, "check_transformers", lambda backend: None)  # the root CI suite has no [asr]
    source = tmp_path / "talk.wav"
    source.write_bytes(b"")
    row = {"source": str(source), "stage": "raw-srt", "output": str(tmp_path / "o" / "talk.srt"), "asr": ALIGNER,
           "asr_rows": "mai2"}
    asked = []
    monkeypatch.setattr(device_module, "moss_local_unusable_reason",
                        lambda tier, dev, backend: asked.append(backend) or None)
    assert build_item(row).asr_lane(build_item(row).payload) == GPU_LANE
    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev, backend: f"--asr {backend} X")
    with pytest.raises(ValueError, match=f"--asr {ALIGNER} X"):
        build_item(row)
    assert asked and set(asked) == {ALIGNER}


def test_transformers_that_cannot_decode_stop_the_task_before_its_rows_are_asked_for(cloud_source, tmp_path,  # noqa: F811
                                                                                     monkeypatch):
    """The card is fine but transformers is not: refused when the task is built, and, called directly, before
    step 1 asks the API for rows."""
    from finesub.pipeline import build_item
    from finesub.speech.joint.moss_local import decoder as D
    from finesub.speech.runtime import device as device_module
    from finesub.stages import run_pipeline

    source, calls = cloud_source
    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev, backend: None)

    def broken(backend="moss-local"):
        raise RuntimeError(f"--asr {backend} does not run on transformers 9.0")

    monkeypatch.setattr(D, "check_transformers", broken)
    monkeypatch.setattr(moss_local, "label_rows", lambda *a, **k: pytest.fail("decoded"))
    row = {"source": str(source), "stage": "raw-srt", "output": str(tmp_path / "o" / "talk.srt"), "asr": ALIGNER,
           "asr_rows": "mai2"}
    with pytest.raises(ValueError, match="transformers 9.0"):
        build_item(row)
    with pytest.raises(joint_stage.JointAsrError, match="transformers 9.0"):
        run_pipeline(source, output_path=tmp_path / "talk.srt", stage="raw-srt", asr=ALIGNER, asr_rows="mai2")
    assert calls == []


def test_the_last_part_takes_a_line_ending_past_the_decoded_audio_and_a_tiny_budget_says_so(monkeypatch):
    """Lines are clamped to the container's duration, which can be a few ms longer than the decoded samples."""
    import numpy as np

    from finesub.speech.joint.moss_local import decoder as D
    from finesub.speech.joint.moss_local import forced

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [0] * len(text)

    processor = type("P", (), {"tokenizer": Tok()})()
    monkeypatch.setattr(D, "longest_pass", lambda *a, **k: 600.0)
    monkeypatch.setattr(D, "fits", lambda model, prompt, budget: {"bf16": 10 ** 9})
    monkeypatch.setattr(forced, "decode_part", lambda model, proc, audio, rows, lo, budget, language:
                        [(1, {1: 0.0})] * len(rows))
    audio = np.zeros(160010, dtype=np.float32)  # 10.000625 s
    lines = [{"start": 1.0, "end": 2.0, "text": "a"}, {"start": 9.0, "end": 10.001, "text": "b"}]
    parts = forced.label(None, processor, audio, lines, [], "ja", 8.0, 600.0)
    assert [i for p in parts for i, _, _ in p.picks] == [0, 1]
    monkeypatch.setattr(D, "longest_pass", lambda *a, **k: 60.0)
    with pytest.raises(RuntimeError, match="the VRAM budget holds no part"):
        forced.label(None, processor, np.zeros(16000 * 200, dtype=np.float32), lines, [], "ja", 8.0, 600.0)


def test_step_one_holds_no_gpu_lock_and_the_aligner_is_released_before_the_decode(cloud_source, tmp_path,  # noqa: F811
                                                                                     monkeypatch):
    """`joint.GPU_LOCK` is not reentrant and 35t's repair aligner takes it itself: step 1 runs outside it, and the
    aligner gives its memory back before local MOSS counts the card."""
    from finesub.speech import joint
    from finesub.speech.joint import aligner as A
    from finesub.stages import run_pipeline

    source, calls = cloud_source
    order = []
    rows = joint_stage.transcribe

    def transcribe(*a, **k):
        assert not joint.GPU_LOCK.locked()
        order.append("rows")
        return rows(*a, **k)

    monkeypatch.setattr(joint_stage, "transcribe", transcribe)
    monkeypatch.setattr(A, "release", lambda: order.append("release"))
    _fake_labels(monkeypatch)
    decode = moss_local.label_rows

    def label_rows(*a, **k):
        assert not joint.GPU_LOCK.locked()
        order.append("decode")
        return decode(*a, **k)

    monkeypatch.setattr(moss_local, "label_rows", label_rows)
    run_pipeline(source, output_path=tmp_path / "talk.srt", stage="raw-srt", asr=ALIGNER, asr_rows="gemini-35t",
                 asr_aligner="on", language="ja")
    assert order == ["rows", "release", "decode"]


def test_a_task_made_by_another_backend_says_how_to_switch(cloud_source, tmp_path, monkeypatch):  # noqa: F811
    from finesub.stages import run_pipeline

    source, _ = cloud_source
    _fake_labels(monkeypatch)
    run_pipeline(source, output_path=tmp_path / "talk.srt", stage="raw-srt", asr="mai2", language="en")
    with pytest.raises(ValueError, match="recognition is mai2 .*delete its -aligned.json"):
        run_pipeline(source, output_path=tmp_path / "talk.srt", stage="raw-srt", asr=ALIGNER, asr_rows="mai2",
                     language="en")


def _part_fakes(monkeypatch, room):
    from finesub.speech.joint.moss_local import decoder as D
    from finesub.speech.joint.moss_local import forced

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [0] * len(text)

    decoded = []
    monkeypatch.setattr(D, "longest_pass", lambda *a, **k: 600.0)
    monkeypatch.setattr(D, "fits", lambda model, prompt, budget: {"bf16": prompt + room + 64 + D.BUCKET})
    monkeypatch.setattr(forced, "decode_part", lambda model, proc, audio, rows, lo, budget, language:
                        decoded.append(lo) or [(1, {1: 0.0})] * len(rows))
    return forced, type("P", (), {"tokenizer": Tok()})(), decoded


def test_a_part_too_dense_for_its_kv_is_shortened_and_the_next_one_is_not(monkeypatch):
    import numpy as np

    forced, processor, _ = _part_fakes(monkeypatch, room=6000)
    # 150 lines of 50 tokens in the first 300 s (7500 tokens), none after
    lines = [{"start": 2.0 * k, "end": 2.0 * k + 1.5, "text": "a" * (50 - forced.ROW_TOKENS)} for k in range(150)]
    parts = forced.label(None, processor, np.zeros(16000 * 1500, dtype=np.float32), lines, [], "ja", 8.0, 600.0)
    assert parts[0].hi - parts[0].lo < 600 * forced.SHRINK
    assert parts[1].hi - parts[1].lo == pytest.approx(600.0)
    assert {i for p in parts for i, _, _ in p.picks} == set(range(150))


def test_a_part_with_no_lines_is_not_decoded(monkeypatch):
    import numpy as np

    forced, processor, decoded = _part_fakes(monkeypatch, room=10 ** 6)
    lines = [{"start": s, "end": s + 2.0, "text": "a"} for s in (10.0, 30.0, 1700.0, 1780.0)]
    parts = forced.label(None, processor, np.zeros(16000 * 1800, dtype=np.float32), lines, [], "ja", 8.0, 600.0)
    empty = [p for p in parts if not p.picks]
    assert empty and all(p.kv is None for p in empty)
    assert len(decoded) == len(parts) - len(empty) == 2


def test_a_shared_transcript_is_refused_for_several_sources():
    from finesub import pipeline

    assert "asr_transcript" in pipeline._PER_SOURCE_KEYS



@pytest.mark.heavy_resource
def test_the_block_forward_matches_the_decoders_step():
    """`blocks.Blocks` is a second hand-written forward beside `Decoder`'s step: fed the same tokens they give the
    same next-token log-probabilities up to bf16 rounding, and K/V a block wrote read back by steps as their own.
    (2026-11-24: KL 0.002 / 0.0003; with the block's key RoPE sign flipped, 10.95.)"""
    import numpy as np
    import torch

    from finesub.speech.runtime import hf_weights
    from finesub.speech.joint.moss_local import decoder as D
    from finesub.speech.joint.moss_local import forced
    from finesub.speech.runtime.device import moss_local_unusable_reason

    reason = moss_local_unusable_reason(None, None)
    if reason is not None:
        pytest.skip(reason)
    model, processor = D.load(hf_weights.prepare(D.MANIFEST_ID))
    rng = np.random.default_rng(0)
    t = np.arange(D.SR * 12) / D.SR
    pcm = (0.05 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(t.size))
    pcm = pcm.astype(np.float32)
    s = forced._Stream(model, processor, pcm, 512, 4.0, "ja")
    tok = processor.tokenizer
    head = tok.encode("[0.52][S01]こんにちは、今日はいい天気ですね。[2.40][2.61][S02]", add_special_tokens=False)
    tail = tok.encode("そうですね、散歩に行きましょうか？[4.80]", add_special_tokens=False)
    start = s.dec.pos

    def lp(x):
        return torch.log_softmax(x.float(), -1)

    def run(block: bool):
        s.dec.pos = start
        out = s.blocks.feed(head).clone() if block else [s.dec.logits(i).clone() for i in head][-1]
        after = [s.dec.logits(i).clone() for i in tail][-1]  # steps over what the first half wrote
        return lp(out), lp(after)

    stepped, blocked = run(False), run(True)
    for a, b in zip(stepped, blocked):
        top = torch.topk(a, 20).indices
        kl = float((a.exp() * (a - b)).sum())
        assert int(a.argmax()) == int(b.argmax()) and kl < 0.01
        # bf16 logits of magnitude 16-32 are 0.125 apart: two ticks; a wrong RoPE or mask is off by whole nats
        assert float((a[top] - b[top]).abs().max()) < 0.5
