from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import tomllib

from finesub import config as app_config
from finesub import pipeline, stages
from finesub.reporting import NullReporter, reporting_to
from finesub.speech.postprocessing import segmentation
from finesub.speech.recognition import vad_asr_stage as vad_asr
from finesub.speech.runtime import hf_weights


def _with_config(tmp_path, monkeypatch, body: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()


def test_split_length_scale_defaults_to_the_calibrated_params(
    tmp_path, monkeypatch
) -> None:
    _with_config(tmp_path, monkeypatch, "[providers]\ntavily = false\n")

    assert vad_asr.resolve_split_params() is segmentation.DEFAULT_SPLIT_PARAMS


def test_split_length_scale_comes_from_config_when_not_given(
    tmp_path, monkeypatch
) -> None:
    _with_config(tmp_path, monkeypatch, "[segmentation]\nlength_scale = 0.8\n")

    params = vad_asr.resolve_split_params()

    assert params.dur_ok_hi == pytest.approx(6.4)


def test_explicit_split_length_scale_beats_the_config(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[segmentation]\nlength_scale = 0.8\n")

    params = vad_asr.resolve_split_params(1.2)

    assert params.dur_ok_hi == pytest.approx(9.6)


@pytest.mark.parametrize(
    ("explicit", "body", "expected_source"),
    [
        (None, "[segmentation]\nlength_scale = 4.0\n", "config.toml"),
        (4.0, "[providers]\ntavily = false\n", "--split-length-scale"),
    ],
)
def test_out_of_range_split_length_scale_names_where_it_came_from(
    tmp_path, monkeypatch, explicit, body, expected_source
) -> None:
    _with_config(tmp_path, monkeypatch, body)

    with pytest.raises(ValueError, match=expected_source):
        vad_asr.resolve_split_params(explicit)


def test_the_silero_assist_defaults_to_on_in_the_backend(
    tmp_path, monkeypatch
) -> None:
    """The backend owns the answer; the CLI passes "the user did not say"."""

    _with_config(tmp_path, monkeypatch, "[providers]\ntavily = false\n")

    assert vad_asr.resolve_vad_silero_assist() is True
    assert vad_asr.resolve_vad_silero_assist(None) is True


def test_the_config_file_can_turn_the_silero_assist_off(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[vad]\nsilero_assist = false\n")

    assert vad_asr.resolve_vad_silero_assist() is False


def test_an_explicit_silero_switch_beats_the_config(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[vad]\nsilero_assist = false\n")

    assert vad_asr.resolve_vad_silero_assist(True) is True


def test_a_non_boolean_silero_setting_is_refused_rather_than_coerced(
    tmp_path, monkeypatch
) -> None:
    """`silero_assist = 1` is a typo, not a True -- say so instead of guessing."""

    _with_config(tmp_path, monkeypatch, "[vad]\nsilero_assist = 1\n")

    with pytest.raises(ValueError, match="true or false"):
        vad_asr.resolve_vad_silero_assist()


def test_the_cli_can_still_turn_the_silero_assist_off(monkeypatch) -> None:
    """Flipping a `store_true` default would have removed the off switch.

    `BooleanOptionalAction` keeps both directions and leaves "unset" as None,
    which is what lets the backend resolver see that the user said nothing.
    """

    import sys

    def parsed(*extra: str):
        monkeypatch.setattr(sys, "argv", ["finesub", "clip.wav", *extra])
        return pipeline.parse_args().vad_silero_assist

    assert parsed() is None
    assert parsed("--vad-silero-assist") is True
    assert parsed("--no-vad-silero-assist") is False


def test_asr_prefetch_skips_a_model_the_manifest_does_not_describe(
    monkeypatch,
) -> None:
    """`--model tiny` must not first download the default model's 1.6 GB."""

    from finesub_bootstrap import model_ensure

    calls: list[str] = []
    monkeypatch.setattr(
        model_ensure,
        "ensure_hf_model",
        lambda model_id, **_kwargs: calls.append(model_id),
    )

    assert vad_asr.ensure_asr_weights("tiny") == hf_weights.UNMANAGED
    assert calls == []


def test_asr_prefetch_hands_back_the_pinned_revision(monkeypatch) -> None:
    """The loader must load the snapshot that was verified, not today's `main`."""

    from finesub_bootstrap import model_ensure

    monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
    monkeypatch.setattr(model_ensure, "ensure_hf_model", lambda *_a, **_k: None)
    monkeypatch.setattr(
        model_ensure, "pinned_snapshot_loadable", lambda _id: False
    )

    assert vad_asr.ensure_asr_weights(
        vad_asr.asr_align.DEFAULT_MODEL
    ) == hf_weights.HfLoad("abc123", False)


def test_asr_prefetch_covers_every_listed_alternative(monkeypatch) -> None:
    """A listed alternative gets the same routing as the default model.

    The manifest is what makes mirror routing and pinned-revision verification
    possible, so every model it describes must reach `ensure_hf_model` under
    its own manifest id -- a model wired into the catalog but not into this
    lookup would silently keep the bare lazy download.
    """

    from finesub_bootstrap import model_ensure
    from finesub_bootstrap.model_caches import WHISPER_JA_REPO_ID, WHISPER_REPO_ID

    calls: list[str] = []
    monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
    monkeypatch.setattr(
        model_ensure,
        "ensure_hf_model",
        lambda model_id, **_kwargs: calls.append(model_id),
    )
    monkeypatch.setattr(
        "finesub.paths.resolve_managed_app_paths",
        lambda: SimpleNamespace(data_root=Path("."), models=Path(".")),
    )

    for model_name in (
        vad_asr.asr_align.DEFAULT_MODEL,
        WHISPER_REPO_ID,
        WHISPER_JA_REPO_ID,
    ):
        assert vad_asr.ensure_asr_weights(model_name).revision == "abc123"
    assert calls == ["whisper", "whisper", "whisper-ja"]


def test_aligned_json_keeps_observations_out_of_metadata(tmp_path) -> None:
    intervals = [{"start": 0.0, "end": 1.5}, {"start": 2.0, "end": 3.25}]
    vad_meta = {
        "vad": {"energy_mode": "weighted"},
        "pause_hints": {"scorer": [1.46], "padding": []},
    }
    output = tmp_path / "clip-aligned.json"

    vad_asr.write_aligned_json(
        output,
        [],
        vad_meta=vad_meta,
        align_meta={"model": "test"},
        vad_timeline=vad_asr.build_vad_timeline(intervals, vad_meta),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    # The splitter's yardstick is data, and it now sits where a later pass can
    # read it without re-running the VAD.
    assert payload["vad_timeline"]["intervals"] == intervals
    assert payload["vad_timeline"]["pause_hints"] == {"scorer": [1.46], "padding": []}
    # metadata stays the invocation record.
    assert payload["metadata"]["vad"] == {"energy_mode": "weighted"}


def test_vad_asr_default_output_path_uses_aligned_suffix() -> None:
    assert vad_asr.default_output_path(Path("out/input-vocal.flac")) == Path(
        "out/input-vocal-aligned.json"
    )


def test_pipeline_default_paths_nest_under_out_stem_dir() -> None:
    paths = stages.default_pipeline_paths(Path("data/input.wav"))
    assert paths.final_srt == Path("out/input/input.srt")
    assert paths.vocal_audio == Path("out/input/input-vocal.ogg")
    assert paths.aligned_json == Path("out/input/input-aligned.json")
    assert paths.stable_json == Path("out/input/input-stable.json")
    assert paths.raw_srt == Path("out/input/input-raw.srt")
    assert paths.translated_srt == Path("out/input/input-translated.srt")
    assert paths.task_artifact_dir == Path("out/input/input.llm-artifacts")
    assert paths.metadata_json == Path("out/input/input-metadata.json")
    assert paths.srt == paths.final_srt


def test_pipeline_output_path_drives_intermediate_names() -> None:
    paths = stages.default_pipeline_paths(Path("data/input.wav"), Path("results/final.srt"))
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

    assert stages._use_or_create(target, "test", create) == target
    assert target.read_text(encoding="utf-8") == "complete"
    assert observed == [tmp_path / ".result.part.json"]


def test_use_or_create_removes_partial_output_after_failure(tmp_path) -> None:
    target = tmp_path / "result.json"
    temporary = tmp_path / ".result.part.json"

    def fail(path: Path) -> Path:
        path.write_text("partial", encoding="utf-8")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        stages._use_or_create(target, "test", fail)

    assert not target.exists()
    assert not temporary.exists()


def _stage_fakes(monkeypatch) -> None:
    """Stand in for every stage that writes an artifact."""

    def separate(input_path, **kwargs):
        Path(kwargs["output_path"]).write_bytes(b"vocal")
        return Path(kwargs["output_path"])

    def vad_asr(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def stabilize(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    def to_srt(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text("", encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", vad_asr)
    monkeypatch.setattr(stages.asr_stabilize, "run_asr_stabilize", stabilize)
    monkeypatch.setattr(stages.to_srt, "convert_json_to_srt", to_srt)


class _StageRecorder(NullReporter):
    """Records stage transitions, ignoring everything else a run reports."""

    def __init__(self) -> None:
        self.announced: list[tuple[str, str]] = []
        self.planned_stages: list[str] = []

    def planned(self, stages) -> None:
        self.planned_stages = list(stages)

    def stage_started(self, stage, *, reused=False, detail="") -> None:
        self.announced.append((stage, "reused" if reused else "running"))


def test_every_stage_reports_itself_once(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    _stage_fakes(monkeypatch)
    recorder = _StageRecorder()

    with reporting_to(recorder):
        pipeline.run_pipeline(
            source,
            output_path=tmp_path / "out" / "final.srt",
        )

    assert recorder.announced == [
        ("vocal", "running"),
        ("aligned", "running"),
        ("stable", "running"),
        ("raw-srt", "running"),
    ]
    # The denominator comes from the run's own plan, not a fixed list.
    assert recorder.planned_stages == ["vocal", "aligned", "stable", "raw-srt"]


def test_stages_skipped_outright_still_report_themselves(tmp_path, monkeypatch) -> None:
    # The branches that skip a stage wholesale never reach `_use_or_create`:
    # with an aligned JSON in place, separation is not merely cached but
    # unnecessary, and VAD-ASR takes an early-out of its own. Reporting from
    # inside that helper would leave both silent -- and a silent stage is
    # exactly what let the progress list tick everything at once.
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    output.parent.mkdir(parents=True)
    (output.parent / "final-aligned.json").write_text(
        '{"segments":[]}', encoding="utf-8"
    )
    _stage_fakes(monkeypatch)
    recorder = _StageRecorder()

    with reporting_to(recorder):
        pipeline.run_pipeline(source, output_path=output)

    assert recorder.announced == [
        ("vocal", "reused"),
        ("aligned", "reused"),
        ("stable", "running"),
        ("raw-srt", "running"),
    ]


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

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(stages.asr_stabilize, "run_asr_stabilize", fake_stabilize)
    monkeypatch.setattr(stages.to_srt, "convert_json_to_srt", fake_to_srt)

    output = tmp_path / "out" / "final.srt"
    paths = pipeline.run_pipeline(
        source,
        output_path=output,
        model_name="large-v3-turbo",
        device="cuda",
        language="en",
        gap_sec=0.5,
        gpu_tier="high",
        word=True,
        asr_stabilize_profile=2,
    )

    assert paths.final_srt == output
    assert calls[0] == (
        "separate",
        {
            "input_path": source.resolve(),
            "output_path": output.with_name(".final-vocal.part.ogg"),
            "gpu_tier": "high",
            # The user's request, so `--device cpu` reaches this stage too.
            "device": "cuda",
            # None means "the model's own rate"; the switch only carries a
            # value when the caller asked for one of the lower rungs.
            "separator_sample_rate": None,
            "metadata_sink": {},
            # Where to write down a decoded copy of the input, so a run that
            # dies still leaves something able to name the file it created.
            "run_metadata_path": paths.metadata_json,
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
            "gpu_tier": "high",
            # None reaches the stage, which resolves it. The pipeline no
            # longer carries a copy of the answer.
            "vad_silero_assist": None,
            "qwen_verify": "auto",
            "lang_redecode": "auto",
            # None = follow config.toml, then the code default. The stage owns
            # that resolution so every front end lands on the same answer.
            "split_length_scale": None,
            # Same shape: None reaches the stage, which asks the GPU profile.
            # Recorded as provenance; the assembly exists (transcribe.DecodePrefetch)
            # but the tier table is 1 everywhere (bench-baselines 二十二).
            "asr_decode_batch": None,
            # The knowledge names, already resolved to text by the pipeline:
            # `speech` must not import `llm`, so what crosses is a string.
            "asr_context": "",
            "run_metadata_path": paths.metadata_json,
            # Its own stage artifact, on the final path rather than a temporary
            # one: a run that dies in recognition should leave the VAD pass
            # behind for the next one to reuse.
            "vad_prefix_path": paths.vad_json,
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
            # The raw export writes into this `.part` file up to three times
            # (render, then one pass per timeline profile). Validating in each
            # pass reported one finding three times, naming a path the reader
            # cannot open -- so the passes stay quiet and the export validates
            # the finished artifact once, under its real name.
            "validate": False,
        },
    )


def test_pipeline_skips_existing_step_outputs(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
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

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fail_separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fail_vad_asr)
    monkeypatch.setattr(stages.to_srt, "convert_json_to_srt", fail_to_srt)

    assert pipeline.run_pipeline(source, output_path=output) == paths
    assert calls == []


def test_the_decoded_temporary_is_removed_on_success_and_kept_on_failure(
    tmp_path, monkeypatch
) -> None:
    """`ensure_decodable_input`'s contract, and `run_vocal_separation`'s shape.

    Both halves have been got wrong once each: deleting in a `finally` threw
    away the decode a retry wanted, and not deleting at all left a lossless
    copy of an hour of video in the task directory forever. The real function
    is called here -- the pipeline-level tests replace it wholesale, which is
    exactly why neither mistake showed up there.
    """

    from finesub.speech.preprocessing.separator import separation

    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    decoded = tmp_path / "input-decoded.flac"

    def fake_decode(input_path, workdir):
        decoded.write_bytes(b"decoded")
        return decoded, decoded

    monkeypatch.setattr(separation, "ensure_decodable_input", fake_decode)
    monkeypatch.setattr(
        separation,
        "_encode_asr_delivery",
        lambda merged, output: Path(output).write_bytes(b"ogg"),
    )

    separation.encode_asr_delivery(source, tmp_path / "out" / "input-vocal.ogg")
    assert not decoded.exists()

    def fail_encode(merged, output):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(separation, "_encode_asr_delivery", fail_encode)
    with pytest.raises(RuntimeError, match="interrupted"):
        separation.encode_asr_delivery(source, tmp_path / "out" / "input-vocal.ogg")
    assert decoded.exists(), "a failed run keeps it so the retry skips the decode"


def test_a_lossless_vocal_track_is_reused_instead_of_separating_again(
    tmp_path,
    monkeypatch,
) -> None:
    """`.flac` is separation's other delivery, not a stale artifact.

    Every reader below already takes it (`resolve_vocal_audio`), so a run that
    holds one must not pay for the most expensive GPU stage a second time only
    to end up with the 16 kHz copy of what it had. The skip check used to look
    for the `.ogg` this stage happens to write, which is not the same question.
    """

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
    paths.srt.parent.mkdir(parents=True)
    lossless = paths.vocal_audio.with_suffix(".flac")
    lossless.write_bytes(b"lossless vocal")

    def fail_separate(*args, **kwargs):
        raise AssertionError("vocal separation should be skipped")

    recognized: list[Path] = []

    def fake_vad_asr(input_path, **kwargs):
        recognized.append(Path(input_path))
        target = Path(kwargs["output_path"])
        target.write_text('{"segments":[]}', encoding="utf-8")
        return target

    monkeypatch.setattr(
        stages.vocal_separation, "run_vocal_separation", fail_separate
    )
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fake_vad_asr)

    pipeline.run_pipeline(source, output_path=output, stage="aligned")

    assert recognized == [lossless]
    assert not paths.vocal_audio.exists()


def test_pipeline_hands_a_local_video_to_separation_unconverted(
    tmp_path,
    monkeypatch,
) -> None:
    """No lossy generation before separation: the source goes in as it is.

    Separation decodes the container itself (losslessly, keeping rate and
    channels), so narrowing the audio here would only cost quality the
    44.1 kHz stereo separator model is trained on.
    """

    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake video")
    output = tmp_path / "out" / "final.srt"
    separation_inputs: list[Path] = []

    def fake_separate(input_path, **kwargs):
        separation_inputs.append(Path(input_path))
        target = Path(kwargs["output_path"])
        target.write_bytes(b"vocal")
        return target

    monkeypatch.setattr(
        stages.vocal_separation,
        "run_vocal_separation",
        fake_separate,
    )

    paths = pipeline.run_pipeline(source, output_path=output, stage="vocal")

    assert separation_inputs == [source.resolve()]
    assert list(output.parent.glob("*.ogg")) == [paths.vocal_audio]
    assert paths.vocal_audio.read_bytes() == b"vocal"


@pytest.mark.parametrize(
    "requested, handed",
    [("cpu", "cpu"), ("cuda", "cuda"), (None, "cuda")],
)
def test_pipeline_hands_the_requested_device_to_separation(
    tmp_path, monkeypatch, requested, handed
) -> None:
    """`--device cpu` must reach separation, not just VAD-ASR.

    The stage decides for itself (torch) while the ASR stage asks
    CTranslate2, so what the pipeline passes is the user's *request*, and it
    passes it to both. `None` is the desktop's "not chosen" and is the code
    default, cuda -- normalised once in `run_pipeline` so no stage ever reads
    None as anything else (review 2026-09-02).
    """

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    received: list[object] = []

    def fake_separate(input_path, **kwargs):
        received.append(kwargs["device"])
        target = Path(kwargs["output_path"])
        target.write_bytes(b"vocal")
        return target

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fake_separate)

    pipeline.run_pipeline(source, output_path=output, stage="vocal", device=requested)

    assert received == [handed]


def test_pipeline_skips_all_default_steps_when_raw_output_exists(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
    paths.srt.parent.mkdir(parents=True)
    paths.vocal_audio.write_bytes(b"existing vocal")
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        stages.vocal_separation,
        "run_vocal_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("separation should be skipped")),
    )
    monkeypatch.setattr(
        stages.vad_asr,
        "run_vad_asr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VAD-ASR should be skipped")),
    )
    monkeypatch.setattr(
        stages.to_srt,
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
    paths = stages.default_pipeline_paths(source, output)
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

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fail_separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fail_vad_asr)
    monkeypatch.setattr(stages.to_srt, "convert_json_to_srt", fake_to_srt)

    pipeline.run_pipeline(source, output_path=output, stage="raw-srt")

    assert not paths.vocal_audio.exists()
    assert to_srt_calls and to_srt_calls[0]["input_path"] == paths.stable_json


def test_pipeline_applies_timeline_only_profile_to_raw_srt(tmp_path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 0.3, "text": "hello"},
                    {"start": 2.0, "end": 2.5, "text": "world"},
                ]
            }
        ),
        encoding="utf-8",
    )

    pipeline.run_pipeline(
        source,
        output_path=output,
        stage="raw-srt",
        postprocess_profile=0,
    )

    raw = paths.raw_srt.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:00,600" in raw
    assert "hello" in raw


def test_pipeline_resolves_raw_srt_overlaps_before_extending(tmp_path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 2.0, "text": "hello"},
                    {"start": 1.5, "end": 3.0, "text": "world"},
                ]
            }
        ),
        encoding="utf-8",
    )

    pipeline.run_pipeline(
        source,
        output_path=output,
        stage="raw-srt",
        postprocess_profile=0,
    )

    raw = paths.raw_srt.read_text(encoding="utf-8")
    # Overlap trimmed to the next start rather than silently capped by the
    # duration step; the last cue still takes the full +0.3s pad.
    assert "00:00:00,000 --> 00:00:01,500" in raw
    assert "00:00:01,500 --> 00:00:03,300" in raw


def test_pipeline_writes_core_timing_and_worker_metadata(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"

    def fake_separate(input_path, **kwargs):
        kwargs["metadata_sink"].update(
            {"profile_limit": 2, "effective": 1, "device": "cuda"}
        )
        Path(kwargs["output_path"]).write_bytes(b"vocal")
        return Path(kwargs["output_path"])

    def fake_vad_asr(input_path, **kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "segments": [],
                    "metadata": {
                        "asr_align": {}
                    },
                }
            ),
            encoding="utf-8",
        )
        return Path(kwargs["output_path"])

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fake_vad_asr)

    paths = pipeline.run_pipeline(
        source,
        output_path=output,
        stage="aligned",
        gpu_tier="standard",
    )

    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["timing"]["stages"]["vocal_separation"]["status"] == "executed"
    assert metadata["timing"]["stages"]["asr"]["status"] == "executed"
    assert metadata["timing"]["total_sec"] >= 0
    assert metadata["workers"]["vocal_separation"]["effective"] == 1
    # ASR no longer reports workers: it always runs one.
    assert "asr" not in metadata["workers"]


def test_pipeline_reuses_aligned_json_when_stable_is_missing(tmp_path, monkeypatch) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.aligned_json.write_text('{"segments":[]}', encoding="utf-8")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        stages.vocal_separation,
        "run_vocal_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("separation should be skipped")
        ),
    )
    monkeypatch.setattr(
        stages.vad_asr,
        "run_vad_asr",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("VAD-ASR should be skipped")
        ),
    )

    def fake_stabilize(input_path, **kwargs):
        calls.append({"input_path": input_path, **kwargs})
        Path(kwargs["output_path"]).write_text('{"segments":[]}', encoding="utf-8")
        return Path(kwargs["output_path"])

    monkeypatch.setattr(stages.asr_stabilize, "run_asr_stabilize", fake_stabilize)

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
    paths = stages.default_pipeline_paths(source, output)
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

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(
        stages.asr_stabilize,
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
    paths = stages.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.vocal_audio.write_bytes(b"existing vocal")
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")
    paths.translated_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\n你好。\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        stages.vocal_separation,
        "run_vocal_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("separation should be skipped")),
    )
    monkeypatch.setattr(
        stages.vad_asr,
        "run_vad_asr",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("VAD-ASR should be skipped")),
    )
    monkeypatch.setattr(
        stages.to_srt,
        "convert_json_to_srt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw SRT should be skipped")),
    )

    assert pipeline.run_pipeline(source, output_path=output, stage="final-srt") == paths
    assert output.exists()
    assert "你好" in output.read_text(encoding="utf-8")


def test_pipeline_uses_custom_artifact_dir_for_summary_and_report(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
    paths.final_srt.parent.mkdir(parents=True)
    paths.stable_json.write_text('{"segments":[]}', encoding="utf-8")
    paths.raw_srt.write_text("", encoding="utf-8")
    paths.translated_srt.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\n你好。\n",
        encoding="utf-8",
    )
    custom_artifacts = tmp_path / "custom-artifacts"
    custom_artifacts.mkdir()
    (custom_artifacts / "task-artifacts.jsonl").write_text(
        json.dumps(
            {
                "kind": "correction_window_response",
                "created_at": "2026-01-01T00:00:02+00:00",
                "payload": {
                    "chunk_id": "0001",
                    "validation_ok": True,
                    "output_limited": False,
                    "api_attempts": [
                        {
                            "started_at": "2026-01-01T00:00:00+00:00",
                            "returned_at": "2026-01-01T00:00:01+00:00",
                            "elapsed_sec": 1.0,
                        }
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        stages.vocal_separation,
        "run_vocal_separation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("separation should be skipped")
        ),
    )
    monkeypatch.setattr(
        stages.vad_asr,
        "run_vad_asr",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("VAD-ASR should be skipped")
        ),
    )

    returned_paths = pipeline.run_pipeline(
        source,
        output_path=output,
        stage="final-srt",
        task_artifact_dir=custom_artifacts,
    )

    assert returned_paths.task_artifact_dir == custom_artifacts
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert metadata["llm_rounds"][0]["round"] == "correction-0001-answer"
    assert (custom_artifacts / "task-report.md").exists()
    assert not paths.task_artifact_dir.exists()


def test_pipeline_passes_llm_profile_args_through(tmp_path, monkeypatch) -> None:
    import finesub.llm.correction_translation as ct

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    paths = stages.default_pipeline_paths(source, output)
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
        llm_media="text",
        llm_retrieval="none",
        llm_difficulty="efficiency",
        llm_fast="off",
        llm_output_scale=1.5,
    )

    assert seen["profile"].profile_id == (
        "correction_media=text,planning_media=text,retrieval=none,"
        "difficulty=efficiency,continuity=serial"
    )
    assert seen["profile"].output_scale == 1.5
    assert seen["fast"] == "off"
    assert seen["video_path"] is None


def test_pipeline_url_input_takes_the_audio_path_only_when_asked(
    tmp_path, monkeypatch
) -> None:
    from finesub.media import source as media_source

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
            AssertionError("--no-download-video must not fetch the video")
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

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(stages.to_srt, "convert_json_to_srt", fake_to_srt)

    paths = pipeline.run_pipeline(
        "https://example.com/watch?v=1", download_video_source=False
    )

    assert calls == ["audio:vid1"]
    assert paths.final_srt == Path("out/vid1/vid1.srt")
    assert paths.raw_srt == Path("out/vid1/vid1-raw.srt")


def test_pipeline_url_input_mm_high_downloads_video_for_llm(tmp_path, monkeypatch) -> None:
    import finesub.llm.correction_translation as ct
    from finesub.media import source as media_source

    video = tmp_path / "out" / "final-dir" / "final-dir.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    seen: dict[str, object] = {}

    monkeypatch.setattr(media_source, "resolve_video_id", lambda url, data_dir: "vid1")
    monkeypatch.setattr(
        media_source,
        "download_video",
        lambda url, data_dir, **kwargs: ("vid1", video),
    )

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

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fake_separate)
    monkeypatch.setattr(stages.vad_asr, "run_vad_asr", fake_vad_asr)
    monkeypatch.setattr(stages.to_srt, "convert_json_to_srt", fake_to_srt)
    monkeypatch.setattr(ct, "run_full_correction", fake_correction)

    output = tmp_path / "out" / "final-dir" / "final.srt"
    pipeline.run_pipeline(
        "https://example.com/watch?v=1",
        output_path=output,
        stage="translated-srt",
        llm_media="video",
    )

    # The download is the source as-is; separation makes its own copy if it
    # needs one, so no narrowed audio track is derived up front.
    assert seen["source_for_separation"] == video.resolve()
    assert Path(seen["correction"]["video_path"]) == video
    assert "https://example.com/watch?v=1" in seen["correction"]["extra_info"]
    assert str(video) in seen["correction"]["extra_info"]


def test_explicit_llm_video_satisfies_a_video_switch_on_audio_input(
    tmp_path,
) -> None:
    """`--llm-video` exists exactly for "transcribe the .wav, show the .mp4".

    Availability is a property of the *video source*, not of the ASR input's
    suffix -- checking only the suffix rejected the very combination the flag
    was added for.
    """

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    video = tmp_path / "source.mp4"
    video.write_bytes(b"fake")

    media, resolved_video, notice = stages.resolve_llm_media_for_source(
        source_path=source,
        stage="translated-srt",
        llm_media="video",
        llm_video=str(video),
        llm_correction_media="video",
    )

    assert media == "video"
    assert str(resolved_video) == str(video)
    assert notice == ""


def test_video_switch_without_any_video_source_is_an_error(tmp_path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")

    with pytest.raises(ValueError, match="--llm-video"):
        stages.resolve_llm_media_for_source(
            source_path=source,
            stage="translated-srt",
            llm_media="video",
            llm_video=None,
            llm_correction_media="video",
        )


def test_convenience_video_default_still_downgrades_on_audio_only(tmp_path) -> None:
    """Only the *explicit* per-task override is an error; the convenience
    default keeps downgrading with a notice (plan v2 D20)."""

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")

    media, resolved_video, notice = stages.resolve_llm_media_for_source(
        source_path=source,
        stage="translated-srt",
        llm_media="video",
        llm_video=None,
    )

    assert media == "audio"
    assert resolved_video is None
    assert notice


def test_llm_media_untouched_before_llm_stages(tmp_path) -> None:
    # A plain raw-srt run never reaches the LLM stages, so the mm-high default
    # must not be rewritten (and must stay silent) for audio-only input.
    source = tmp_path / "input.wav"
    media, video, notice = stages.resolve_llm_media_for_source(
        source, stage="raw-srt", llm_media="video", llm_video=None
    )
    assert (media, video, notice) == ("video", None, "")


def test_llm_media_downgrades_for_audio_only_llm_run(tmp_path) -> None:
    source = tmp_path / "input.wav"
    media, video, notice = stages.resolve_llm_media_for_source(
        source, stage="translated-srt", llm_media="video", llm_video=None
    )
    assert media == "audio"
    assert video is None
    assert "audio-only" in notice


def test_llm_media_keeps_video_and_defaults_video_path_for_video_input(tmp_path) -> None:
    source = tmp_path / "input.mp4"
    media, video, notice = stages.resolve_llm_media_for_source(
        source, stage="final-srt", llm_media="video", llm_video=None
    )
    assert (media, video, notice) == ("video", source, "")

    explicit = tmp_path / "other.mkv"
    media, video, notice = stages.resolve_llm_media_for_source(
        source, stage="final-srt", llm_media="video", llm_video=explicit
    )
    assert (media, video, notice) == ("video", explicit, "")


def test_llm_media_untouched_when_not_video(tmp_path) -> None:
    source = tmp_path / "input.wav"
    media, video, notice = stages.resolve_llm_media_for_source(
        source, stage="final-srt", llm_media="text", llm_video=None
    )
    assert (media, video, notice) == ("text", None, "")


def test_name_output_path_maps_to_out_dir() -> None:
    from finesub.paths import resolve_name_output_path

    assert resolve_name_output_path("四月一看PV") == Path("out/四月一看PV/四月一看PV.srt")
    assert resolve_name_output_path("  spaced  ") == Path("out/spaced/spaced.srt")


@pytest.mark.parametrize("bad", ["a/b", "a\\b", "../escape", "..", ".", "", "   "])
def test_name_output_path_rejects_separators(bad: str) -> None:
    from finesub.paths import resolve_name_output_path

    with pytest.raises(ValueError, match="--name must be a bare name"):
        resolve_name_output_path(bad)


def test_vad_asr_empty_vad_output_keeps_aligned_json_schema(tmp_path, monkeypatch) -> None:
    source = tmp_path / "vocal.flac"
    # A real FLAC: the stage probes its input before any of the mocks below run.
    sf.write(str(source), np.zeros((1000, 1), dtype="float32"), 16000)
    output = tmp_path / "aligned.json"

    monkeypatch.setattr(
        vad_asr.asr_align,
        "print_peak_resource_usage",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(vad_asr.asr_align, "reset_peak_gpu_memory_stats_for_run", lambda *args: None)
    monkeypatch.setattr(
        vad_asr, "resolve_asr_device", lambda device, *, gpu_allowed=True: "cpu"
    )
    monkeypatch.setattr(
        vad_asr.vad_detection,
        "detect_segments",
        lambda input_path, observer=None: (
            [], {"vad": {"backend": "test"}}, 0.0, {}, object()
        ),
    )

    # Explicitly off: this is about the empty-VAD schema, and the assist is
    # on by default now, which would pull silero in for no reason here.
    assert (
        vad_asr.run_vad_asr(
            source, output_path=output, device="cpu", vad_silero_assist=False
        )
        == output.resolve()
    )
    assert '"segments": []' in output.read_text(encoding="utf-8")
    assert '"backend": "test"' in output.read_text(encoding="utf-8")

    fw_output = tmp_path / "fw-aligned.json"
    vad_asr.run_vad_asr(
        source,
        output_path=fw_output,
        device="cpu",
        vad_silero_assist=False,
    )
    fw_metadata = json.loads(fw_output.read_text(encoding="utf-8"))["metadata"][
        "asr_align"
    ]
    assert fw_metadata["fw_refine"] == {
        "detect_disfluencies": True,
        "collect_path_signals": True,
        "collect_boundary_signals": False,
        "event_field": "alignment_events",
    }


def _reconstruct_via_block_loader(path: Path, *, block_seconds: float, pad_seconds: float, step: float) -> np.ndarray:
    """Rebuild the whole 16k timeline through AudioBlockLoader, as run_vad_asr does."""
    from finesub.speech.recognition import transcribe as asr_align
    from finesub.speech.preprocessing import energy as vad_energy
    from finesub.speech.preprocessing.audio import get_audio_info

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

    from finesub.speech.preprocessing import energy as vad_energy

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

    from finesub.speech.preprocessing import energy as vad_energy

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
    from finesub.speech.recognition import transcribe as asr_align
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


def test_pyproject_pins_the_stack_the_pipeline_needs() -> None:
    """What the distribution declares, minus the scripts it no longer has.

    The console-script table went away in 2026-08 (`finesub` comes from the CLI
    wheel), so what used to be checked here -- that each speech stage has a
    script pointing at the right module -- is now
    `test_packaging.MODULE_ENTRY_POINTS`, which additionally proves each module
    can be run with `python -m`. That is strictly more than this file asserted.
    """

    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert "scripts" not in data["project"]
    assert data["project"]["requires-python"] == ">=3.12"
    asr_deps = data["project"]["optional-dependencies"]["asr"]
    # Exact, and they move together: torchaudio stops at 2.11, triton declares
    # no torch constraint, and the patched CT2 needs cuBLAS from CUDA 12.
    assert "torch==2.11.0" in asr_deps
    assert "torchaudio==2.11.0" in asr_deps
    assert "torchvision==0.26.0" in asr_deps
    assert (
        "triton-windows==3.6.0.post26 ; platform_system == 'Windows'" in asr_deps
    )
    discovery = data["tool"]["setuptools"]["packages"]["find"]
    assert "py-modules" not in data["tool"]["setuptools"]
    assert "finesub*" in discovery["include"]
    assert "utils*" not in discovery["include"]
    # The top-level `llm` package was folded into `finesub` in 2026-08; leaving
    # its pattern here would silently re-publish a generic top-level name the
    # moment someone recreated the directory.
    assert "llm*" not in discovery["include"]


# ---------------------------------------------------------------------------
# knowledge switch: one resolution rule, shared by every front end
# ---------------------------------------------------------------------------


def test_an_unset_knowledge_switch_resolves_against_the_difficulty() -> None:
    """`efficiency` disables knowledge by construction.

    Resolving only in the pipeline CLI left `run_pipeline` handing "collect"
    to an efficiency run, which the LLM layer refuses with a ValueError --
    *after* ASR has already finished, the most expensive place to find out.
    The desktop worker passes `request.knowledge` straight through, and the
    batch runner copies every CLI default into every item, so both were
    exposed to exactly that.
    """

    from finesub.stages import resolve_knowledge_switch

    assert resolve_knowledge_switch(None, "quality") == "collect"
    assert resolve_knowledge_switch(None, "intermediate") == "collect"
    assert resolve_knowledge_switch(None, "efficiency") == "none"
    # An explicit switch is a user statement and passes through untouched --
    # including the contradiction, which stays the LLM layer's hard error.
    assert resolve_knowledge_switch("update", "quality") == "update"
    assert resolve_knowledge_switch("collect", "efficiency") == "collect"
    assert resolve_knowledge_switch("none", "quality") == "none"


def test_every_front_end_leaves_the_switch_unset_by_default(monkeypatch) -> None:
    """A concrete default anywhere upstream would shadow the shared rule.

    The multi-source path is the one that bit: `_defaults_from_args` copies
    each CLI default into every item, so a `--knowledge collect` default there
    was always truthy by the time an item was built and the efficiency fallback
    became dead code.
    """

    import inspect
    import sys

    from finesub import pipeline as pipeline_mod
    from finesub import stages as stages_mod

    assert inspect.signature(pipeline_mod.run_pipeline).parameters["knowledge"].default is None

    monkeypatch.setattr(sys, "argv", ["asr-pipeline", "input.wav"])
    args = pipeline_mod.parse_args()
    assert args.knowledge is None
    # And what an item inherits keeps it unset.
    assert "knowledge" not in pipeline_mod._defaults_from_args(args, single=True)


def test_lang_redecode_defaults_to_auto_everywhere(monkeypatch) -> None:
    import inspect
    import sys

    from finesub import pipeline as pipeline_mod
    from finesub import stages as stages_mod
    from finesub.speech.recognition import vad_asr_stage as vad_asr_mod

    assert (
        inspect.signature(pipeline_mod.run_pipeline)
        .parameters["lang_redecode"]
        .default
        == "auto"
    )
    assert (
        inspect.signature(vad_asr_mod.run_vad_asr)
        .parameters["lang_redecode"]
        .default
        == "auto"
    )

    # The main CLI no longer restates it: an absent flag reaches the runner as
    # `None`, is dropped from the row, and the signature above is what answers.
    # Asserting "auto" here again is what made the value exist in two places.
    monkeypatch.setattr(sys, "argv", ["asr-pipeline", "input.wav"])
    args = pipeline_mod.parse_args()
    assert args.lang_redecode is None
    assert "lang_redecode" not in pipeline_mod._defaults_from_args(args, single=True)

    monkeypatch.setattr(sys, "argv", ["vad-asr", "input.wav"])
    assert vad_asr_mod.parse_args().lang_redecode == "auto"


def test_a_url_item_defaults_to_fetching_the_video(monkeypatch) -> None:
    """The subtitles usually end up burned into it, so the file is wanted.

    Opting out is `--no-download-video`; the media switch is about what a model
    is shown, not about what lands on disk.
    """

    import inspect
    import sys

    from finesub import pipeline as pipeline_mod
    from finesub import stages as stages_mod

    monkeypatch.setattr(sys, "argv", ["asr-pipeline", "https://example.com/a"])
    args = pipeline_mod.parse_args()
    # Not restated on the CLI any more: absent reaches the runner as `None`,
    # is dropped from the row, and `run_pipeline`'s signature is what says
    # True. One place, so "the default" cannot mean two things.
    assert args.download_video_source is None
    assert "download_video_source" not in pipeline_mod._defaults_from_args(
        args, single=True
    )
    assert (
        inspect.signature(stages_mod.run_pipeline)
        .parameters["download_video_source"]
        .default
        is True
    )
    monkeypatch.setattr(
        sys, "argv", ["asr-pipeline", "https://example.com/a", "--no-download-video"]
    )
    assert pipeline_mod.parse_args().download_video_source is False


def test_no_separate_delivers_the_track_without_running_separation(
    tmp_path, monkeypatch
) -> None:
    """`--no-separate` skips the algorithm, not the stage's contract.

    The artifact still lands at the path everything downstream reads, which is
    why no consumer, and neither the existence-based skip nor resume, needs a
    case for "there is no vocal file".
    """

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    output = tmp_path / "out" / "final.srt"
    _stage_fakes(monkeypatch)
    encoded: list[tuple[Path, Path]] = []

    def fail_separate(*args, **kwargs):
        raise AssertionError("separation must not run under --no-separate")

    def encode(source_path, output_path, **kwargs):
        encoded.append((Path(source_path), Path(output_path)))
        Path(output_path).write_bytes(b"transcoded vocal")
        return Path(output_path)

    monkeypatch.setattr(stages.vocal_separation, "run_vocal_separation", fail_separate)
    monkeypatch.setattr(stages.vocal_separation, "encode_asr_delivery", encode)

    paths = pipeline.run_pipeline(source, output_path=output, separate=False)

    assert paths.vocal_audio.read_bytes() == b"transcoded vocal"
    assert [call[0] for call in encoded] == [source]
    record = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    vocal = record["timing"]["stages"]["vocal_separation"]
    # Not `reused`: something ran, and a bug report is read for exactly this
    # field to find out whether separation was part of it.
    assert vocal["status"] == "skipped"
    assert vocal.get("elapsed_sec") is not None


def test_a_skipped_separation_is_not_announced_as_reused(
    tmp_path, monkeypatch
) -> None:
    """`reused` means nothing ran; the detail says which route did."""

    source = tmp_path / "input.wav"
    source.write_bytes(b"fake")
    _stage_fakes(monkeypatch)

    def encode(source_path, output_path, **kwargs):
        Path(output_path).write_bytes(b"v")
        return Path(output_path)

    monkeypatch.setattr(stages.vocal_separation, "encode_asr_delivery", encode)
    seen: list[tuple[str, bool, str]] = []

    class _Detail(NullReporter):
        def stage_started(self, stage, *, reused=False, detail="") -> None:
            seen.append((stage, reused, detail))

    with reporting_to(_Detail()):
        pipeline.run_pipeline(
            source, output_path=tmp_path / "out" / "final.srt", separate=False
        )

    vocal = [entry for entry in seen if entry[0] == "vocal"]
    assert vocal and vocal[0][1] is False
    assert vocal[0][2], "the skipped route has to say what it did instead"


def test_separate_is_on_unless_something_says_otherwise(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[providers]\ntavily = false\n")

    assert stages.vocal_separation.resolve_separate() is True
    assert stages.vocal_separation.resolve_separate(False) is False


def test_separate_comes_from_config_when_not_given(tmp_path, monkeypatch) -> None:
    _with_config(tmp_path, monkeypatch, "[separator]\nenabled = false\n")

    assert stages.vocal_separation.resolve_separate() is False
    # An explicit flag still outranks the file -- the chain is CLI, then config,
    # then the default, and only the first layer is a statement about this run.
    assert stages.vocal_separation.resolve_separate(True) is True
