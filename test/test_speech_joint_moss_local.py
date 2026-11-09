"""Local MOSS (`--asr moss-local`) without a GPU or the model: the rescue's pure rules, the adapter's bookkeeping and
the process state it has to give back, the device criterion, and its wiring into the stage and the batch."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from finesub.speech import joint
from finesub.speech.joint import moss_local as ML
from finesub.speech.joint import stage as joint_stage
from finesub.speech.joint.model import Transcript, Unit, segment_turns
from finesub.speech.joint.moss_local import decoder as D
from finesub.speech.joint.moss_local import rescue
from finesub.speech.joint.moss_local import rules as R
from finesub.speech.runtime import device as device_module
from finesub.speech.runtime import hf_weights
from finesub.subtitles.metrics import weighted_char_count


def seg(start, end, text, speaker="S01"):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


# --- rules ---------------------------------------------------------------------------------------------------

ENGLISH = "so we went out to the park and it was just like the best day of the year for all of us " * 2
GERMAN = ("wir sind heute in den Park gegangen und es war wirklich der schönste Tag des Jahres für uns alle "
          "zusammen mit Freunden und Familie draußen in der Sonne bei gutem Wetter")


def test_english_is_told_from_another_latin_language_by_its_common_words():
    assert R.wrong_language(ENGLISH, "en") is False
    assert R.wrong_language(GERMAN, "en") is True
    assert R.wrong_language("too short to tell", "en") is None


def test_japanese_is_told_from_chinese_by_kana_and_chinese_from_japanese():
    japanese = "今日はみんなで公園に行きました。とても楽しかったですね、また行きたいと思います。" * 2
    chinese = "我们今天一起去了公园，天气非常好，大家都玩得很开心，希望下次还能再一起去那里玩。" * 2
    assert R.wrong_language(japanese, "ja") is False
    assert R.wrong_language(chinese, "ja") is True
    assert R.wrong_language(chinese, "zh") is False
    assert R.wrong_language(japanese, "zh") is True


def test_a_language_it_cannot_test_or_none_passes():
    for language in (None, "", "auto", "sw"):
        assert R.wrong_language(GERMAN, language) is False


def test_a_loop_is_a_short_period_over_the_last_span():
    assert R.periodic([1, 2, 3] * 16)
    assert not R.periodic(list(range(R.LOOP_SPAN)))
    assert not R.periodic([7] * (R.LOOP_SPAN - 1))


def test_gaps_are_the_uncovered_stretches_up_to_the_end():
    segs = [seg(0, 5, "a"), seg(40, 50, "b"), seg(45, 52, "c")]
    assert R.gaps(segs, 100.0) == [(5, 40), (52, 100.0)]
    assert R.gaps(segs, 60.0, least=3.0) == [(5, 40), (52, 60.0)]


def test_a_window_drops_lines_it_heard_again_and_keeps_new_ones_in_the_gap():
    have = [seg(10.0, 12.0, "hello there my friend"), seg(60.0, 62.0, "see you later then")]
    heard_again = seg(10.4, 12.3, "hello there my friend")
    new = seg(20.0, 24.0, "something nobody said before")
    kept, anchors, off = R.kept_from_window([heard_again, new], have, 9.0, 61.0, 12.0, 60.0, short=False)
    assert kept == [new] and anchors == 1 and off == pytest.approx(0.4)


def test_a_window_whose_clock_is_off_adds_nothing():
    have = [seg(10.0, 12.0, "hello there my friend")]
    late = seg(13.5, 15.0, "hello there my friend")
    new = seg(20.0, 24.0, "something nobody said before")
    kept, anchors, off = R.kept_from_window([late, new], have, 9.0, 61.0, 12.0, 60.0, short=False)
    assert kept == [] and anchors == 1 and off > R.ANCHOR_TOL


def test_a_short_gaps_window_keeps_only_what_is_centred_in_the_gap():
    inside = seg(21.0, 23.0, "in the gap")
    neighbour = seg(25.5, 28.0, "the next line")  # overlaps the gap but is centred after it
    kept, _, _ = R.kept_from_window([inside, neighbour], [], 10.0, 35.0, 20.0, 25.0, short=True)
    assert kept == [inside]


def test_a_late_end_is_pulled_in_but_not_shorter_than_its_text_takes():
    text = "あいうえおかきくけこ"
    w = weighted_char_count(text)
    paced = [seg(float(k), k + 2.0, text) for k in (0, 3, 6)]  # ending on their speech: the source's pace
    spans = [(0.0, 2.0), (3.0, 5.0), (6.0, 8.0), (10.0, 11.0)]
    long = seg(10.0, 20.0, text)
    out, trimmed = R.trim_tails([*paced, long], spans)
    floor = 10.0 + R.TAIL_SLACK * w / (w / 2.0)
    assert out[-1]["end"] == pytest.approx(min(20.0, floor))
    assert trimmed == [[10.0, 20.0, round(min(20.0, floor), 2)]]
    assert out[:3] == paced


def test_a_segment_without_speech_keeps_its_end():
    segs = [seg(0.0, 2.0, "abc"), seg(5.0, 9.0, "def")]
    out, trimmed = R.trim_tails(segs, [(0.0, 2.0)])
    assert out == segs and trimmed == []


def _centroids(vectors):
    def centroids(segs):
        return {s["speaker"]: np.asarray(vectors[s["speaker"]], float) for s in segs}

    return centroids


def test_labels_of_one_voice_merge_into_the_one_with_more_speech():
    segs = [seg(0, 10, "a", "S01"), seg(20, 22, "b", "S02"), seg(30, 40, "c", "S03")]
    s = math.sqrt(0.19)
    merges = R.merge_labels(segs, _centroids({"S01": [1, 0, 0], "S02": [0.9, s, 0], "S03": [0, 0, 1]}), 0.8)
    assert merges == [("S02", "S01", 0.9)]
    assert [x["speaker"] for x in segs] == ["S01", "S01", "S03"]


def test_two_labels_talking_over_each_other_stay_two_people():
    segs = [seg(0, 10, "a", "S01"), seg(2, 6, "b", "S02")]
    assert R.merge_labels(segs, _centroids({"S01": [1, 0], "S02": [1, 0]}), 0.8) == []
    assert {x["speaker"] for x in segs} == {"S01", "S02"}


# --- the adapter -----------------------------------------------------------------------------------------------


def test_event_counts_walk_parts_and_windows_and_skip_the_kv_records():
    events = [{"kind": "kv"}, {"kind": "skip"},
              {"kind": "fill", "events": [{"kind": "kv"}, {"kind": "loop-penalty"}]},
              {"kind": "part", "events": [{"kind": "kv"}, {"kind": "skip"}, {"kind": "short", "events": []}]},
              {"kind": "merge"}]
    assert ML.event_counts(events) == {"fill": 1, "loop-penalty": 1, "merge": 1, "short": 1, "skip": 2}


def test_prompts_follow_the_language_and_default_to_upstreams():
    assert D.prompt_for("en").startswith("Transcribe the audio.")
    assert D.prompt_for("EN") == D.prompt_for("en")
    for language in (None, "ja", "zh", "de"):
        assert D.prompt_for(language) == D.DEFAULT_PROMPT


@pytest.fixture
def fake_gpu(monkeypatch, tmp_path):
    """Everything the adapter touches on the card, recorded: the allocator cap, the model load, the decode."""
    import torch

    state = {"fractions": [], "loaded": 0, "prepared": 0, "decode": None}
    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev: None)
    monkeypatch.setattr(device_module, "total_vram_gib", lambda: 16.0)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction",
                        lambda fraction, card=None: state["fractions"].append(round(fraction, 4)))
    monkeypatch.setattr(D, "check_transformers", lambda: None)

    def prepare(manifest_id):
        state["prepared"] += 1
        return hf_weights.HfLoad("rev123", True)

    def load(plan):
        state["loaded"] += 1
        return "model", "processor"

    def transcribe(model, processor, audio, language, budget_gib):
        state["budget"] = budget_gib
        if state["decode"] is not None:
            raise state["decode"]
        assert joint.GPU_LOCK.locked()
        return rescue.Result([seg(1.0, 2.0, "hello", "S01"), seg(2.5, 2.5, "", "S02"), seg(3.0, 5.0, "bye", "S02")],
                             [{"kind": "kv"}, {"kind": "eos"}], [{"start": 0.0, "kv": "bf16", "positions": 8192}])

    monkeypatch.setattr(hf_weights, "prepare", prepare)
    monkeypatch.setattr(D, "load", load)
    monkeypatch.setattr(rescue, "transcribe", transcribe)
    monkeypatch.setattr(ML, "_decode", lambda source: np.zeros(16000, np.float32))
    monkeypatch.setattr(ML, "_quiet_transformers", nullcontext)
    monkeypatch.setattr(ML, "_triton_cache_in_checkout", nullcontext)
    return state


def test_the_adapter_hands_back_segments_and_what_the_metadata_records(fake_gpu, tmp_path):
    transcript, route = ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="standard",
                                      device=None)
    assert (transcript.provider, transcript.unit, transcript.duration) == ("moss-local", "segment", 6.0)
    assert [(u.start, u.end, u.speaker) for u in transcript.units()] == [(1.0, 2.0, "S01"), (3.0, 5.0, "S02")]
    assert route == {"revision": "rev123", "kv": [{"start": 0.0, "kv": "bf16", "positions": 8192}],
                     "events": {"eos": 1}}
    from finesub.speech.runtime.resources import get_resource_profile

    assert fake_gpu["budget"] == pytest.approx(get_resource_profile("standard").usable_gib - D.CONTEXT_GIB)
    assert fake_gpu["fractions"][-1] == 1.0 and fake_gpu["fractions"][0] < 1.0
    assert not joint.GPU_LOCK.locked()


def test_a_failed_decode_gives_back_the_cap_and_the_lock(fake_gpu, tmp_path):
    fake_gpu["decode"] = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="standard", device=None)
    assert fake_gpu["fractions"] == [fake_gpu["fractions"][0], 1.0]
    assert not joint.GPU_LOCK.locked()


@pytest.mark.parametrize("chained", [False, True])
def test_a_failed_decode_frees_the_model_before_the_lock_goes(fake_gpu, tmp_path, monkeypatch, chained):
    import weakref

    class Model:
        pass

    box = [Model()]
    gone = weakref.ref(box[0])
    monkeypatch.setattr(D, "load", lambda plan: (box.pop(), "processor"))

    def deeper(model):
        held = [model]  # a frame local, as `Pass` holds the model and its K/V buffers  # noqa: F841
        raise RuntimeError("CUDA out of memory")

    def transcribe(model, processor, audio, language, budget_gib):
        if not chained:
            deeper(model)
        try:
            deeper(model)
        except RuntimeError as e:  # a handler that re-raises: the first one's frames ride along as its cause
            raise RuntimeError("CUDA out of memory, again") from e

    monkeypatch.setattr(rescue, "transcribe", transcribe)
    import torch

    at_release = []  # whether the model was still alive when the step gave the cache back, still holding the lock
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: at_release.append((gone() is not None,
                                                                              joint.GPU_LOCK.locked())))
    with pytest.raises(RuntimeError, match="out of memory"):
        ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="standard", device=None)
    assert at_release == [(False, True)]


def test_a_run_that_cannot_use_it_stops_before_any_download(fake_gpu, tmp_path, monkeypatch):
    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev: "--asr moss-local needs X")
    with pytest.raises(joint_stage.JointAsrError, match="needs X"):
        ML.transcribe(tmp_path / "a.wav", language="ja", duration=6.0, gpu_tier="cpu", device=None)
    assert fake_gpu["prepared"] == 0 and fake_gpu["loaded"] == 0 and fake_gpu["fractions"] == []


def test_without_a_language_it_says_what_that_costs(fake_gpu, tmp_path, capsys):
    ML.transcribe(tmp_path / "a.wav", language=None, duration=6.0, gpu_tier="standard", device=None)
    assert "Warning: --asr moss-local without --language" in capsys.readouterr().err


def test_the_sdpa_swap_lasts_only_while_the_prefill_runs():
    hf_sdpa = pytest.importorskip("transformers.integrations.sdpa_attention")
    original = hf_sdpa.use_gqa_in_sdpa
    with pytest.raises(ValueError):
        with D._expanded_kv_sdpa():
            assert hf_sdpa.use_gqa_in_sdpa(None, None) is False
            raise ValueError("prefill failed")
    assert hf_sdpa.use_gqa_in_sdpa is original


def test_triton_compiles_into_the_checkout_cache_only_during_the_step(monkeypatch, tmp_path):
    import os

    import finesub.paths

    monkeypatch.setattr(finesub.paths, "resolve_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    with ML._triton_cache_in_checkout():
        assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "cache" / "moss-local-triton")
    assert "TRITON_CACHE_DIR" not in os.environ  # Inductor reads it too: the separator's kernels go elsewhere
    monkeypatch.setenv("TRITON_CACHE_DIR", "mine")
    with ML._triton_cache_in_checkout():
        assert os.environ["TRITON_CACHE_DIR"] == "mine"
    assert os.environ["TRITON_CACHE_DIR"] == "mine"


class _Config:
    num_hidden_layers, num_key_value_heads, head_dim, max_position_embeddings = 28, 8, 128, 131072


class _Model:
    class model:
        class language_model:
            config = _Config


@pytest.mark.parametrize("tier, minutes", [("standard", 33), ("high", 68)])
def test_a_pass_is_as_long_as_the_tiers_vram_holds(monkeypatch, tier, minutes):
    import torch

    from finesub.speech.runtime.resources import get_resource_profile

    monkeypatch.delenv("MOSS_FAST_KV", raising=False)
    monkeypatch.setattr(D, "fp8_kernels", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: int(1.69 * 2**30))  # the weights
    budget = get_resource_profile(tier).usable_gib - D.CONTEXT_GIB
    assert D.longest_pass(_Model, rescue.TOKENS_PER_SECOND, budget) == 60.0 * minutes


def test_the_entry_tier_holds_no_minute_so_the_check_refuses_it_up_front(monkeypatch, nvidia):
    import torch

    from finesub.speech.runtime.resources import get_resource_profile

    monkeypatch.setattr(D, "fp8_kernels", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: int(1.69 * 2**30))
    with pytest.raises(RuntimeError, match="holds no minute"):
        D.longest_pass(_Model, rescue.TOKENS_PER_SECOND, get_resource_profile("entry").usable_gib - D.CONTEXT_GIB)
    assert "GPU tier (entry) gives it 3 GiB" in device_module.moss_local_unusable_reason("entry", None)


# --- the device criterion ----------------------------------------------------------------------------------


@pytest.fixture
def nvidia(monkeypatch):
    import torch

    from finesub.speech.runtime import resources

    card = {"capability": (8, 9), "unusable": None, "rocm": False}
    monkeypatch.setattr(resources, "detect_gpu_tier", lambda **k: "standard")
    monkeypatch.setattr(device_module, "cuda_unusable_reason", lambda: card["unusable"])
    monkeypatch.setattr(device_module, "is_rocm", lambda: card["rocm"])
    monkeypatch.setattr(device_module, "gpu_name", lambda index=None: "Test GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: card["capability"])
    return card


def test_an_ampere_or_newer_nvidia_card_runs_it(nvidia):
    assert device_module.moss_local_unusable_reason(None, None) is None
    assert device_module.moss_local_unusable_reason("auto", "cuda") is None
    nvidia["capability"] = (8, 0)
    assert device_module.moss_local_unusable_reason("standard", None) is None


@pytest.mark.parametrize("tier, device, card, says", [
    (None, "cpu", {}, "--device cpu"),
    ("cpu", None, {}, "--gpu-tier cpu"),
    (None, None, {"unusable": "it is unavailable"}, "the GPU cannot be used (it is unavailable)"),
    (None, None, {"rocm": True}, "Test GPU is an AMD card"),
    (None, None, {"capability": (7, 5)}, "Test GPU has compute capability 7.5"),
])
def test_anything_else_is_refused_with_the_reason_and_the_way_out(nvidia, tier, device, card, says):
    nvidia.update(card)
    reason = device_module.moss_local_unusable_reason(tier, device)
    assert says in reason
    assert reason.startswith("--asr moss-local needs an NVIDIA card of compute capability 8.0")
    assert "--asr local" in reason


# --- the stage and the batch ----------------------------------------------------------------------------------


@pytest.fixture
def talk(tmp_path, monkeypatch) -> Path:
    import soundfile as sf

    from finesub.media import ffmpeg

    # CI's Linux job has no ffmpeg; the speech probabilities are stubbed in the test, so silence will do
    monkeypatch.setattr(ffmpeg, "probe_media_duration", lambda src: 8.0)
    monkeypatch.setattr(ffmpeg, "decode_pcm16k", lambda src, **_: bytes(8 * 16000 * 2))

    sr = 16000
    t = np.arange(int(sr * 8.0)) / sr
    path = tmp_path / "talk.wav"
    sf.write(path, (0.1 * np.sin(2 * math.pi * 220 * t)).astype(np.float32), sr)
    return path


def test_the_stage_builds_its_lines_as_moss_pros_and_records_the_run(talk, tmp_path, monkeypatch):
    from finesub.speech.joint import moss_timing as T

    seen = {}

    def transcribe(source, *, language, duration, gpu_tier, device):
        seen.update(gpu_tier=gpu_tier, device=device)
        units = [Unit(0.5, 3.0, "こんにちは、今日はいい天気ですね。", "S01"), Unit(3.5, 6.0, "そうですね。", "S02")]
        return (Transcript("moss-local", ML.MODEL_NAME, duration, "segment", segment_turns(units)),
                {"revision": "rev", "kv": [{"start": 0.0, "kv": "bf16", "positions": 4096}], "events": {"eos": 1}})

    monkeypatch.setattr(ML, "transcribe", transcribe)
    monkeypatch.setattr(T, "frame_probabilities", lambda pcm: np.full(len(pcm) // T.HOP, 0.9, np.float32))
    out = joint_stage.run(talk, backend="moss-local", output_path=tmp_path / "a-aligned.json",
                          cache_root=tmp_path / "cloud", language="ja", repair="on", gpu_tier="high", device="cuda")
    payload = json.loads(out.read_text(encoding="utf-8"))
    record = payload["metadata"][joint_stage.METADATA_KEY]
    assert {k: record[k] for k in ("backend", "model", "revision", "kv", "events")} == {
        "backend": "moss-local", "model": ML.MODEL_NAME, "revision": "rev",
        "kv": [{"start": 0.0, "kv": "bf16", "positions": 4096}], "events": {"eos": 1}}
    assert record["timing"] == "S5b" and "repair" not in record  # MOSS Pro's line building, no repair
    assert seen == {"gpu_tier": "high", "device": "cuda"}
    assert "".join(s["text"] for s in payload["segments"]) == "こんにちは、今日はいい天気ですね。そうですね。"


def test_a_local_moss_task_takes_the_gpu_lane(tmp_path, monkeypatch):
    from finesub.pipeline import build_item
    from finesub.scheduler import GPU_LANE

    monkeypatch.setattr(device_module, "moss_local_unusable_reason", lambda tier, dev: None)
    source = tmp_path / "talk.wav"
    source.write_bytes(b"")
    item = build_item({"source": str(source), "stage": "raw-srt", "output": str(tmp_path / "o" / "talk.srt"),
                       "asr": "moss-local"})
    assert item.asr_lane(item.payload) == GPU_LANE


def test_a_run_that_cannot_decode_with_it_stops_when_the_task_is_built(tmp_path):
    from finesub.pipeline import build_item
    from finesub.stages import default_pipeline_paths

    source = tmp_path / "talk.wav"
    source.write_bytes(b"")
    output = tmp_path / "o" / "talk.srt"
    row = {"source": str(source), "stage": "raw-srt", "output": str(output), "asr": "moss-local", "gpu_tier": "cpu"}
    with pytest.raises(ValueError, match="--gpu-tier cpu keeps this run off the GPU"):
        build_item(row)
    build_item({**row, "stage": "vocal"})  # nothing to decode at this stage
    build_item({**row, "asr": "local"})
    aligned = default_pipeline_paths(source, output_path=output).aligned_json
    aligned.parent.mkdir(parents=True)
    aligned.write_text(json.dumps({"segments": [], "metadata": {"joint_asr": {"backend": "moss-local"}}}),
                       encoding="utf-8")
    build_item(row)  # decoded already: an LLM run of it needs no card
    aligned.unlink()
    paths = default_pipeline_paths(source, output_path=output)
    paths.stable_json.write_text(json.dumps({"segments": [], "metadata": {"joint_asr": {"backend": "moss-local"}}}),
                                 encoding="utf-8")
    build_item(row)  # past aligned already
    with pytest.raises(ValueError, match="--gpu-tier cpu"):
        build_item({**row, "stage": "aligned"})  # asked for the aligned again: it decodes


def test_a_linked_source_is_checked_where_the_run_will_look(tmp_path, monkeypatch):
    import os

    from finesub import stages
    from finesub.pipeline import _refuse_moss_local_without_its_card

    monkeypatch.setattr(stages, "default_output_path", lambda p: tmp_path / "out" / p.stem / f"{p.stem}.srt")
    target = tmp_path / "talk.wav"
    target.write_bytes(b"")
    link = tmp_path / "alias.wav"
    try:
        os.symlink(target, link)
    except OSError:
        pytest.skip("no symlinks here")
    stable = stages.default_pipeline_paths(target).stable_json
    stable.parent.mkdir(parents=True)
    stable.write_text(json.dumps({"segments": [], "metadata": {"joint_asr": {"backend": "moss-local"}}}),
                      encoding="utf-8")
    _refuse_moss_local_without_its_card(str(link), None, {"asr": "moss-local", "gpu_tier": "cpu"}, "raw-srt")


def test_a_url_without_an_output_counts_as_decoding_with_this_runs_backend():
    from finesub.pipeline import _refuse_moss_local_without_its_card

    shared = {"asr": "moss-local", "gpu_tier": "cpu"}
    with pytest.raises(ValueError, match="--gpu-tier cpu"):
        _refuse_moss_local_without_its_card("https://example.com/v", None, shared, "raw-srt")
    _refuse_moss_local_without_its_card("https://example.com/v", None, {**shared, "asr": "moss-pro"}, "raw-srt")


def test_the_aligner_applies_to_local_moss(tmp_path, capsys):
    from finesub.stages import _note_task_backend, default_pipeline_paths

    paths = default_pipeline_paths(tmp_path / "talk.wav", tmp_path / "out" / "talk.srt")
    _note_task_backend(paths, "moss-local", "moss-local", True)
    assert "--asr-aligner" not in capsys.readouterr().err
