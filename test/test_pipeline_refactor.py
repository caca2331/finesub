from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tomllib

import pipeline
import vad_asr


def test_vad_asr_default_output_path_uses_aligned_suffix() -> None:
    assert vad_asr.default_output_path(Path("out/input-vocal.flac")) == Path(
        "out/input-vocal-aligned.json"
    )


def test_pipeline_default_paths_nest_under_out_stem_dir() -> None:
    paths = pipeline.default_pipeline_paths(Path("data/input.wav"))
    assert paths.final_srt == Path("out/input/input.srt")
    assert paths.vocal_audio == Path("out/input/input-vocal.ogg")
    assert paths.aligned_json == Path("out/input/input-aligned.json")
    assert paths.stable_json == Path("out/input/input-stable.json")
    assert paths.raw_srt == Path("out/input/input-raw.srt")
    assert paths.translated_srt == Path("out/input/input-translated.srt")
    assert paths.task_artifact_dir == Path("out/input/input.llm-artifacts")
    assert paths.srt == paths.final_srt


def test_pipeline_output_path_drives_intermediate_names() -> None:
    paths = pipeline.default_pipeline_paths(Path("data/input.wav"), Path("results/final.srt"))
    assert paths.vocal_audio == Path("results/final-vocal.ogg")
    assert paths.aligned_json == Path("results/final-aligned.json")
    assert paths.stable_json == Path("results/final-stable.json")
    assert paths.raw_srt == Path("results/final-raw.srt")
    assert paths.translated_srt == Path("results/final-translated.srt")
    assert paths.final_srt == Path("results/final.srt")


def test_use_or_create_commits_output_atomically(tmp_path) -> None:
    target = tmp_path / "result.json"
    observed: list[Path] = []

    def create(path: Path) -> Path:
        observed.append(path)
        path.write_text("complete", encoding="utf-8")
        assert not target.exists()
        return path

    assert pipeline._use_or_create(target, "test", create) == target
    assert target.read_text(encoding="utf-8") == "complete"
    assert observed == [tmp_path / ".result.part.json"]


def test_use_or_create_removes_partial_output_after_failure(tmp_path) -> None:
    target = tmp_path / "result.json"
    temporary = tmp_path / ".result.part.json"

    def fail(path: Path) -> Path:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        pipeline._use_or_create(target, "test", fail)

    assert not target.exists()
    assert not temporary.exists()


def test_pipeline_passes_parameters_to_each_stage(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_separate(input_path, **kwargs):
        calls.append(("separate", {"input_path": input_path, **kwargs}))
        Path(kwargs["output_path"]).write_bytes(b"vocal")
        return Path(kwargs["output_path"])

    def fake_vad_asr(input_path, **kwargs):
        calls.append(("vad_asr", {"input_path": input_path, **kwargs}))
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def fake_stabilize(input_path, **kwargs):
        calls.append(("asr_stabilize", {"input_path": input_path, **kwargs}))
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def fake_to_srt(input_path, **kwargs):
        calls.append(("to_srt", {"input_path": input_path, **kwargs}))
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(pipeline.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(pipeline.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(pipeline.asr_stabilize, "run_asr_stabilize", fake_stabilize)
    monkeypatch.setattr(pipeline.to_srt, "convert_json_to_srt", fake_to_srt)

    output = tmp_path / "out" / "final.srt"
    paths = pipeline.run_pipeline(
        source,
        output_path=output,
        model_name="large-v3-turbo",
        device="cuda",
        language="en",
        gap_sec=0.5,
        gpu_budget_gb=12,
        word=True,
        asr_stabilize_profile=2,
    )

    assert paths.final_srt == output
    assert calls[0] == (
        "separate",
        {
            "input_path": source.resolve(),
            "output_path": output.with_name(".final-vocal.part.ogg"),
            "gpu_budget_gb": 12,
        },
    )
    assert calls[1] == (
        "vad_asr",
        {
            "input_path": output.with_name("final-vocal.ogg"),
            "output_path": output.with_name(".final-aligned.part.json"),
            "model_name": "large-v3-turbo",
            "device": "cuda",
            "language": "en",
            "gap_sec": 0.5,
            "gpu_budget_gb": 12,
        },
    )
    assert calls[2] == (
        "asr_stabilize",
        {
            "input_path": output.with_name("final-aligned.json"),
            "output_path": output.with_name(".final-stable.part.json"),
            "profile": 2,
        },
    )
    assert calls[3] == (
        "to_srt",
        {
            "input_path": output.with_name("final-stable.json"),
            "output_path": output.with_name(".final-raw.part.srt"),
            "word": True,
        },
    )


def test_pipeline_skips_existing_step_outputs(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.srt.parent.mkdir(parents=True)
    paths.vocal_audio.write_bytes(b"existing vocal")
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")
    calls: list[str] = []

    def fail_separate(*args, **kwargs):
        raise AssertionError("vocal separation should be skipped")

    def fail_vad_asr(*args, **kwargs):
        raise AssertionError("VAD-ASR should be skipped")

    def fail_to_srt(*args, **kwargs):
        raise AssertionError("raw SRT export should be skipped")

    monkeypatch.setattr(pipeline.vocal_separation, "run_vocal_separation", fail_separate)
    monkeypatch.setattr(pipeline.vad_asr, "run_vad_asr", fail_vad_asr)
    monkeypatch.setattr(pipeline.to_srt, "convert_json_to_srt", fail_to_srt)

    assert pipeline.run_pipeline(source, output_path=output) == paths
    assert calls == []


def test_pipeline_skips_all_default_steps_when_raw_output_exists(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.srt.parent.mkdir(parents=True)
    paths.vocal_audio.write_bytes(b"existing vocal")
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        pipeline.vocal_separation,
        "run_vocal_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("separation should be skipped")),
    )
    monkeypatch.setattr(
        pipeline.vad_asr,
        "run_vad_asr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VAD-ASR should be skipped")),
    )
    monkeypatch.setattr(
        pipeline.to_srt,
        "convert_json_to_srt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SRT export should be skipped")),
    )

    assert pipeline.run_pipeline(source, output_path=output) == paths


def test_pipeline_skips_vocal_separation_when_stable_json_exists(tmp_path, monkeypatch) -> None:
    # stable.json present but vocal audio missing, targeting a later stage:
    # vocal separation must be skipped (its only consumer is already satisfied).
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.srt.parent.mkdir(parents=True)
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    to_srt_calls: list[dict] = []

    def fail_separate(*args, **kwargs):
        raise AssertionError("vocal separation should be skipped")

    def fail_vad_asr(*args, **kwargs):
        raise AssertionError("VAD-ASR should be skipped")

    def fake_to_srt(input_path, **kwargs):
        to_srt_calls.append({"input_path": input_path, **kwargs})
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(pipeline.vocal_separation, "run_vocal_separation", fail_separate)
    monkeypatch.setattr(pipeline.vad_asr, "run_vad_asr", fail_vad_asr)
    monkeypatch.setattr(pipeline.to_srt, "convert_json_to_srt", fake_to_srt)

    pipeline.run_pipeline(source, output_path=output, stage="raw-srt")

    assert not paths.vocal_audio.exists()
    assert to_srt_calls and to_srt_calls[0]["input_path"] == paths.stable_json


def test_pipeline_reuses_aligned_json_when_stable_is_missing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.aligned_json.write_text('{"segments":[]}', encoding="utf-8")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        pipeline.vocal_separation,
        "run_vocal_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("separation should be skipped")
        ),
    )
    monkeypatch.setattr(
        pipeline.vad_asr,
        "run_vad_asr",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("VAD-ASR should be skipped")
        ),
    )

    def fake_stabilize(input_path, **kwargs):
        calls.append({"input_path": input_path, **kwargs})
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(pipeline.asr_stabilize, "run_asr_stabilize", fake_stabilize)

    pipeline.run_pipeline(
        source,
        output_path=output,
        stage="stable",
        asr_stabilize_profile=-1,
    )

    assert not paths.vocal_audio.exists()
    assert calls == [
        {
            "input_path": paths.aligned_json,
            "output_path": paths.stable_json.with_name(".final-stable.part.json"),
            "profile": -1,
        }
    ]


def test_explicit_aligned_stage_is_not_satisfied_by_existing_stable(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    calls: list[str] = []

    def fake_separate(input_path, **kwargs):
        calls.append("vocal")
        Path(kwargs["output_path"]).write_bytes(b"vocal")
        return Path(kwargs["output_path"])

    def fake_vad_asr(input_path, **kwargs):
        calls.append("aligned")
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(pipeline.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(pipeline.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(
        pipeline.asr_stabilize,
        "run_asr_stabilize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("stabilization should not run for aligned stage")
        ),
    )

    pipeline.run_pipeline(source, output_path=output, stage="aligned")

    assert calls == ["vocal", "aligned"]
    assert paths.aligned_json.exists()


def test_pipeline_final_stage_reuses_translated_srt_for_postprocess_only(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.vocal_audio.write_bytes(b"existing vocal")
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")
    paths.translated_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\n你好。\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        pipeline.vocal_separation,
        "run_vocal_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("separation should be skipped")),
    )
    monkeypatch.setattr(
        pipeline.vad_asr,
        "run_vad_asr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VAD-ASR should be skipped")),
    )
    monkeypatch.setattr(
        pipeline.to_srt,
        "convert_json_to_srt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw SRT should be skipped")),
    )

    assert pipeline.run_pipeline(source, output_path=output, stage="final-srt") == paths
    assert output.exists()
    assert "你好" in output.read_text(encoding="utf-8")


def test_pipeline_passes_llm_profile_args_through(tmp_path, monkeypatch) -> None:
    import llm.correction_translation as ct

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run_full_correction(**kwargs):
        seen.update(kwargs)
        return paths.translated_srt

    monkeypatch.setattr(ct, "run_full_correction", fake_run_full_correction)

    pipeline.run_pipeline(
        source,
        output_path=output,
        stage="translated-srt",
        llm_route="text",
        llm_level="low",
        llm_fast="off",
        llm_output_scale=1.5,
    )

    assert seen["profile"].profile_id == "text-low"
    assert seen["profile"].output_scale == 1.5
    assert seen["fast"] == "off"
    assert seen["video_path"] is None


def test_pipeline_url_input_downloads_audio_and_uses_video_id_default_paths(
    tmp_path, monkeypatch
) -> None:
    import llm.media_source as media_source

    monkeypatch.chdir(tmp_path)
    audio = tmp_path / "out" / "vid1" / "vid1.ogg"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"fake")
    calls: list[str] = []

    monkeypatch.setattr(media_source, "resolve_video_id", lambda url, data_dir: "vid1")
    monkeypatch.setattr(
        media_source,
        "download_audio",
        lambda url, data_dir, **kwargs: calls.append(
            f"audio:{Path(kwargs['target_dir']).name}"
        )
        or ("vid1", audio),
    )
    monkeypatch.setattr(
        media_source,
        "download_video",
        lambda url, data_dir, **kwargs: (_ for _ in ()).throw(
            AssertionError("video download should not be used for raw-srt")
        ),
    )

    def fake_separate(input_path, **kwargs):
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"vocal")
        return Path(kwargs["output_path"])

    def fake_vad_asr(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def fake_to_srt(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(pipeline.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(pipeline.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(pipeline.to_srt, "convert_json_to_srt", fake_to_srt)

    paths = pipeline.run_pipeline("https://example.com/watch?v=1")

    assert calls == ["audio:vid1"]
    assert paths.final_srt == Path("out/vid1/vid1.srt")
    assert paths.raw_srt == Path("out/vid1/vid1-raw.srt")


def test_pipeline_url_input_mm_high_downloads_video_for_llm(tmp_path, monkeypatch) -> None:
    import llm.correction_translation as ct
    import llm.media_source as media_source

    audio = tmp_path / "out" / "final-dir" / "final-dir.ogg"
    video = tmp_path / "out" / "final-dir" / "final-dir.mp4"
    video.parent.mkdir(parents=True)
    audio.write_bytes(b"audio")
    video.write_bytes(b"video")
    seen: dict[str, object] = {}

    monkeypatch.setattr(media_source, "resolve_video_id", lambda url, data_dir: "vid1")
    monkeypatch.setattr(
        media_source,
        "download_video",
        lambda url, data_dir, **kwargs: ("vid1", video),
    )
    monkeypatch.setattr(media_source, "extract_audio_from_video", lambda video_path: audio)

    def fake_separate(input_path, **kwargs):
        seen["source_for_separation"] = input_path
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_bytes(b"vocal")
        return Path(kwargs["output_path"])

    def fake_vad_asr(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def fake_to_srt(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return Path(kwargs["output_path"])

    def fake_correction(**kwargs):
        seen["correction"] = kwargs
        return Path(kwargs["output_path"])

    monkeypatch.setattr(pipeline.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(pipeline.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(pipeline.to_srt, "convert_json_to_srt", fake_to_srt)
    monkeypatch.setattr(ct, "run_full_correction", fake_correction)

    output = tmp_path / "out" / "final-dir" / "final.srt"
    pipeline.run_pipeline(
        "https://example.com/watch?v=1",
        output_path=output,
        stage="translated-srt",
        llm_level="high",
    )

    assert seen["source_for_separation"] == audio.resolve()
    assert Path(seen["correction"]["video_path"]) == video
    assert "https://example.com/watch?v=1" in seen["correction"]["extra_info"]
    assert str(audio) in seen["correction"]["extra_info"]
    assert str(video) in seen["correction"]["extra_info"]


def test_pipeline_validates_llm_video_flag(tmp_path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = pipeline.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")

    try:
        pipeline.run_pipeline(
            source,
            output_path=output,
            stage="translated-srt",
            llm_video=str(tmp_path / "v.mp4"),  # audio input auto-downgrades to med; explicit video then conflicts
        )
    except ValueError as exc:
        assert "--llm-video only applies" in str(exc)
    else:
        raise AssertionError("expected ValueError for --llm-video outside mm-high")


def test_llm_level_untouched_before_llm_stages(tmp_path) -> None:
    # A plain raw-srt run never reaches the LLM stages, so the mm-high default
    # must not be rewritten (and must stay silent) for audio-only input.
    source = tmp_path / "input.wav"
    level, video, notice = pipeline.resolve_llm_level_for_source(
        source, stage="raw-srt", llm_route="mm", llm_level="high", llm_video=None
    )
    assert (level, video, notice) == ("high", None, "")


def test_llm_level_downgrades_for_audio_only_llm_run(tmp_path) -> None:
    source = tmp_path / "input.wav"
    level, video, notice = pipeline.resolve_llm_level_for_source(
        source, stage="translated-srt", llm_route="mm", llm_level="high", llm_video=None
    )
    assert level == "med"
    assert video is None
    assert "audio-only" in notice


def test_llm_level_keeps_high_and_defaults_video_for_video_input(tmp_path) -> None:
    source = tmp_path / "input.mp4"
    level, video, notice = pipeline.resolve_llm_level_for_source(
        source, stage="final-srt", llm_route="mm", llm_level="high", llm_video=None
    )
    assert (level, video, notice) == ("high", source, "")

    explicit = tmp_path / "other.mkv"
    level, video, notice = pipeline.resolve_llm_level_for_source(
        source, stage="final-srt", llm_route="mm", llm_level="high", llm_video=explicit
    )
    assert (level, video, notice) == ("high", explicit, "")


def test_llm_level_untouched_for_text_route(tmp_path) -> None:
    source = tmp_path / "input.wav"
    level, video, notice = pipeline.resolve_llm_level_for_source(
        source, stage="final-srt", llm_route="text", llm_level="high", llm_video=None
    )
    assert (level, video, notice) == ("high", None, "")


def test_name_output_path_maps_to_out_dir() -> None:
    assert pipeline.resolve_name_output_path("四月一看PV") == Path("out/四月一看PV/四月一看PV.srt")
    assert pipeline.resolve_name_output_path("  spaced  ") == Path("out/spaced/spaced.srt")


@pytest.mark.parametrize("bad", ["a/b", "a\\b", "../escape", "..", ".", "", "   "])
def test_name_output_path_rejects_separators(bad: str) -> None:
    with pytest.raises(ValueError, match="--name must be a bare name"):
        pipeline.resolve_name_output_path(bad)


def test_vad_asr_empty_vad_output_keeps_aligned_json_schema(tmp_path, monkeypatch) -> None:
    source = tmp_path / "vocal.flac"
    source.write_bytes(b"fake")
    output = tmp_path / "aligned.json"

    monkeypatch.setitem(sys.modules, "whisper_timestamped", types.SimpleNamespace())
    monkeypatch.setattr(vad_asr.asr_align, "print_peak_resource_usage", lambda *args: None)
    monkeypatch.setattr(vad_asr.asr_align, "reset_peak_gpu_memory_stats_for_run", lambda *args: None)
    monkeypatch.setattr(vad_asr, "resolve_device", lambda device, context="VAD-ASR": "cpu")
    monkeypatch.setattr(
        vad_asr,
        "load_and_detect_segments",
        lambda input_path: ([], [], {"vad": {"backend": "test"}}, 0.0, {}, object()),
    )

    assert vad_asr.run_vad_asr(source, output_path=output, device="cpu") == output.resolve()
    assert '"segments": []' in output.read_text(encoding="utf-8")
    assert '"backend": "test"' in output.read_text(encoding="utf-8")


def _reconstruct_via_block_loader(path: Path, *, block_seconds: float, pad_seconds: float, step: float) -> np.ndarray:
    """Rebuild the whole 16k timeline through AudioBlockLoader, as run_vad_asr does."""
    import asr_align
    import vad_energy
    from utils.audio import get_audio_info

    sr, frames = get_audio_info(str(path))
    dur = frames / float(sr)
    loader = asr_align.AudioBlockLoader(
        str(path),
        target_sr=vad_energy.TARGET_SR,
        block_seconds=block_seconds,
        pad_seconds=pad_seconds,
        preprocess=False,
    )
    parts = []
    s = 0.0
    while s < dur:
        e = min(dur, s + step)
        parts.append(loader.get_slice(s, e).copy())
        s = e
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def test_streaming_loader_matches_inmemory_audio_no_resample(tmp_path, monkeypatch) -> None:
    """The alignment streaming path must feed Whisper the same 16k samples the old
    in-memory path did. At 16k (no resample) the two must be bit-identical."""
    import soundfile as sf
    import torch

    import vad_energy

    monkeypatch.setattr(vad_energy, "BLOCK_LENGTH", 2.0)  # force multiple block boundaries
    sr = vad_energy.TARGET_SR
    n = int(7.3 * sr)
    t = np.arange(n) / sr
    mono = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    src = tmp_path / "clip16k.wav"
    sf.write(str(src), mono, sr, subtype="PCM_16")

    with torch.inference_mode():
        old = vad_energy._load_asr_audio_streamed(str(src)).detach().cpu().numpy().astype(np.float32)
    new = _reconstruct_via_block_loader(src, block_seconds=4.0, pad_seconds=1.0, step=3.0)

    assert old.shape == new.shape
    assert np.array_equal(old, new)


def test_streaming_loader_matches_inmemory_audio_with_resample(tmp_path, monkeypatch) -> None:
    """At 44.1k the two paths resample in different block layouts; differences must
    stay negligible and confined to block boundaries (Whisper output is unaffected)."""
    import soundfile as sf
    import torch

    import vad_energy

    monkeypatch.setattr(vad_energy, "BLOCK_LENGTH", 2.0)
    src_sr = 44100
    n = int(7.3 * src_sr)
    t = np.arange(n) / src_sr
    mono = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    src = tmp_path / "clip44k.wav"
    sf.write(str(src), stereo, src_sr, subtype="PCM_16")

    with torch.inference_mode():
        old = vad_energy._load_asr_audio_streamed(str(src)).detach().cpu().numpy().astype(np.float32)
    new = _reconstruct_via_block_loader(src, block_seconds=4.0, pad_seconds=1.0, step=3.0)

    assert abs(old.shape[0] - new.shape[0]) <= 1
    m = min(old.shape[0], new.shape[0])
    diff = np.abs(old[:m] - new[:m])
    # Negligible and rare: only a few samples of resampler ringing at boundaries.
    assert diff.max() < 5e-2
    assert np.count_nonzero(diff > 1e-4) < 0.001 * m


def test_block_loader_slice_crossing_boundary_is_not_truncated(tmp_path) -> None:
    """A slice that straddles a block boundary by more than pad_seconds must
    still return the whole requested range (regression guard for the loader)."""
    import asr_align
    import soundfile as sf

    sr = 16000
    n = int(9.0 * sr)
    ramp = (np.arange(n, dtype=np.float32) / n)  # strictly increasing -> position-identifiable
    src = tmp_path / "ramp16k.wav"
    sf.write(str(src), ramp, sr, subtype="PCM_16")

    loader = asr_align.AudioBlockLoader(
        str(src), target_sr=sr, block_seconds=4.0, pad_seconds=1.0, preprocess=False
    )
    # [3.0, 7.0] crosses the block-1 boundary (4.0) by 3s, far more than pad=1s.
    clip = loader.get_slice(3.0, 7.0)
    expected = int(round((7.0 - 3.0) * sr))
    assert abs(clip.shape[0] - expected) <= 1
    # PCM_16 round-trip tolerance; values must track the source ramp, not stop early.
    assert clip[0] == pytest.approx(3.0 / 9.0, abs=1e-3)
    assert clip[-1] == pytest.approx(7.0 / 9.0, abs=1e-3)


def test_pyproject_references_only_current_entrypoints() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    assert scripts["asr-pipeline"] == "pipeline:main"
    assert data["project"]["requires-python"] == ">=3.12"
    assert "torch~=2.8.0" in data["project"]["optional-dependencies"]["asr"]
    assert "torchaudio~=2.8.0" in data["project"]["optional-dependencies"]["asr"]
    modules = set(data["tool"]["setuptools"]["py-modules"])
    assert scripts["asr-stabilize"] == "asr_stabilize:main"
    assert scripts["vad-asr"] == "vad_asr:main"
    assert "asr-playground-main" not in scripts
    assert "main" not in modules
    assert {"asr_stabilize", "pipeline", "subtitle_metrics", "vad_asr"}.issubset(modules)
