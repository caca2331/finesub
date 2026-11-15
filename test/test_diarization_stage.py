"""The diarization stage's file-level behaviour, without loading a model.

The split this file guards: the expensive half is cached because it cannot be
reproduced, and the cheap half rewrites stable.json and *can* be redone. Getting
that backwards would either recompute a non-reproducible answer (losing labels a
human already corrected downstream) or cache a join that should follow the data.
"""

from __future__ import annotations

import json

import pytest

from finesub.speech.diarization.backend import ARTIFACT_ENV
from finesub.speech.diarization.segments import TrackSegment
from finesub.speech.diarization.sortformer_backend import resolve_artifacts
from finesub.speech.diarization.stage import (
    BACKENDS,
    DiarizationResult,
    DiarizationUnavailable,
    labels_for,
    read_timeline,
    rows_current,
    write_rows,
    write_timeline,
)
from finesub.subtitles.stable_rows import overlay_path, read_overlay, read_rows

TIMELINE = DiarizationResult(
    tracks=(
        TrackSegment(track=0, start=0.0, end=2.0),
        TrackSegment(track=1, start=2.0, end=4.0),
    ),
    frame_seconds=0.08,
    overlap=0.0,
)


def stable_payload():
    return {
        "metadata": {"asr": {"model": "large-v3-turbo"}},
        "segments": [
            {"start": 0.0, "end": 1.8, "text": "first"},
            {"start": 2.1, "end": 3.9, "text": "second"},
            {"start": 9.0, "end": 9.5, "text": "after the timeline ends"},
        ],
    }


def test_missing_artifacts_name_the_way_out(tmp_path, monkeypatch):
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    with pytest.raises(DiarizationUnavailable, match="export_sortformer"):
        resolve_artifacts(tmp_path / "nowhere")


def test_the_lossless_vocal_delivery_is_brought_to_16k_mono(tmp_path, monkeypatch):
    # `resolve_vocal_audio` falls back to the separator's `.flac`, which keeps the
    # model's 44.1 kHz stereo. The stage used to refuse anything but 16 kHz, so
    # `--diarize` could not run on a task that only had the lossless delivery.
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")
    soundfile = pytest.importorskip("soundfile")
    import numpy as np

    from finesub.speech.diarization import runner as runner_module
    from finesub.speech.diarization.stage import run_diarization

    seconds, rate = 2.0, 44100
    tone = np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(seconds * rate))).astype(np.float32) * 0.1
    flac = tmp_path / "clip-vocal.flac"
    soundfile.write(str(flac), np.stack([tone, tone], axis=1), rate)
    (tmp_path / "sortformer_core.ts").write_bytes(b"")
    seen = {}

    class FakeRunner:
        """⚠ Stands in for the whole protocol, not just `diarize`.

        The real runner holds about 300 MB of GPU modules and is closed by the caller, so a double
        that only answers `diarize` stops the caller at the `with`. Asserting `closed` here is what
        keeps the release from being dropped later without a test noticing.
        """

        def __init__(self, directory, device="cuda", geometry=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exception):
            self.close()

        def close(self):
            seen["closed"] = True

        def diarize(self, waveform):
            seen["shape"] = tuple(waveform.shape)
            return torch.zeros((int(waveform.shape[-1] / 1280), 4))

    monkeypatch.setattr(runner_module, "SortformerRunner", FakeRunner)
    run_diarization(flac, artifacts=tmp_path, device="cpu", backend="sortformer")
    assert seen["shape"][0] == 1, "mono, one waveform"
    assert abs(seen["shape"][1] - int(seconds * 16000)) <= 2, "resampled to 16 kHz"
    assert seen.get("closed"), "the runner's GPU modules are handed back, not left to the allocator"


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine whose card and model files a test can switch off one at a time."""
    from finesub.speech.runtime import device as device_module

    (tmp_path / "sortformer_core.ts").write_bytes(b"")
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path))
    state = {"cuda": True}
    monkeypatch.setattr(device_module, "cuda_usable", lambda: state["cuda"])
    # An Intel card answers `resolve_device("cuda")` with "xpu", and this stage
    # refuses it (see `_diarization_device`). Pinned off so the fixture describes
    # the machine it says it does rather than the one running the suite.
    monkeypatch.setattr(device_module, "xpu_usable", lambda: False)
    return state


def _paths(tmp_path, *, timeline: bool):
    """The members of `PipelinePaths` the stage touches."""

    class Paths:
        stable_json = tmp_path / "clip-stable.json"
        raw_srt = tmp_path / "clip-raw.srt"
        translated_srt = tmp_path / "clip-translated.srt"
        final_srt = tmp_path / "clip.srt"
        speaker_judge_json = tmp_path / "clip-speaker-judge.json"
        task_artifact_dir = tmp_path / "clip.llm-artifacts"
        diar_json = tmp_path / "clip-diar.json"
        diar_rows_json = tmp_path / "clip-diar-rows.json"

        @staticmethod
        def resolve_vocal_audio():
            raise AssertionError("labelling from a cached timeline must not reach the model")

    Paths.stable_json.write_text(json.dumps(stable_payload()), encoding="utf-8")
    if timeline:
        write_timeline(TIMELINE, Paths.diar_json)
    return Paths


def test_labels_from_a_cached_timeline_need_neither_the_card_nor_the_model(tmp_path, machine):
    # Applying a timeline is JSON to JSON: a machine that cannot diarize can still
    # write the labels an earlier run's timeline carries.
    from finesub.stages import _run_diarization_stage

    machine["cuda"] = False
    paths = _paths(tmp_path, timeline=True)
    _run_diarization_stage(paths, diarize="acoustics", device=None, gpu_tier="entry", announce=lambda *a, **k: None,
                           diarize_split=True)
    assert rows_current(paths.stable_json)
    assert read_overlay(paths.stable_json)["split"]["cuts"] == 0, "the row splitting is decided with the labels"


def test_the_stable_is_never_rewritten_and_the_rows_are_written_once(tmp_path, machine):
    """Every task file is written once by its own step (owner 2026-10-15): the stage reads `-stable.json`
    and writes `-diar-rows.json`, and a later run reuses it. Only a missing file, or one made for other
    lines (the stable was redone), is written again."""
    from finesub.stages import _run_diarization_stage

    paths = _paths(tmp_path, timeline=True)
    stable_bytes = paths.stable_json.read_bytes()
    seen = []

    def run():
        _run_diarization_stage(paths, diarize="acoustics", device=None, gpu_tier="entry",
                               announce=lambda stage, status, **k: seen.append(status), diarize_split=True)

    run()
    rows_bytes = paths.diar_rows_json.read_bytes()
    run()
    assert seen == ["running", "reused"]
    assert paths.stable_json.read_bytes() == stable_bytes, "the stage never writes the stable"
    assert paths.diar_rows_json.read_bytes() == rows_bytes
    paths.diar_rows_json.unlink()
    run()
    assert seen[-1] == "running" and paths.diar_rows_json.read_bytes() == rows_bytes
    fixed = stable_payload()
    fixed["segments"][0]["text"] = "first, fixed by hand"
    paths.stable_json.write_text(json.dumps(fixed), encoding="utf-8")
    run()
    assert seen[-1] == "reused", "a hand-fixed text keeps the rows (they count times, not words)"
    redone = stable_payload()
    redone["segments"][0]["end"] = 1.7  # transcribed again
    paths.stable_json.write_text(json.dumps(redone), encoding="utf-8")
    assert not rows_current(paths.stable_json), "made for other lines"
    run()
    assert seen[-1] == "running" and rows_current(paths.stable_json)


def test_on_gets_a_stage_line():
    from finesub.stages import PIPELINE_STAGE_ORDER, _planned_stages

    everything = max(PIPELINE_STAGE_ORDER.values())
    assert "diarize" in _planned_stages(everything, diarize="acoustics")
    assert "diarize" not in _planned_stages(everything, diarize="off")


def four_busy_tracks(seconds=25.0):
    return DiarizationResult(
        tracks=tuple(TrackSegment(track=k, start=k * seconds, end=(k + 1) * seconds) for k in range(4))
        + (TrackSegment(track=1, start=10.0, end=15.0),),
        frame_seconds=0.08,
        overlap=0.0,
    )


def test_the_timeline_carries_the_acoustic_hints(tmp_path):
    path = write_timeline(four_busy_tracks(), tmp_path / "clip-diar.json")
    hints = json.loads(path.read_text(encoding="utf-8"))["metadata"]["hints"]
    # s1's extra 5 s sit inside s0's 25 s; s0 (25 s) is the shorter of the two (s1: 30 s)
    assert hints["co_activity"]["s0-s1"] == pytest.approx(5 / 25, abs=1e-4)
    assert "not_same" not in hints, "plan 5q: co-activity is reported as a number, never as a verdict"
    # 30 / 25 / 25 / 25 s of track over 100 s of speech
    assert hints["fourth_track_share"] == pytest.approx(0.25)
    assert "capacity_warning" not in hints, "plan 5o.5: no threshold told >=5 speakers from four"


def test_a_short_track_inside_a_long_one_reports_its_share_and_no_verdict(tmp_path):
    # The share is relative to the shorter track, so a 5 s track made of moments the model
    # held one voice on two slots reads 1.0. That is why it stays a number (plan 5q).
    result = DiarizationResult(
        tracks=(TrackSegment(0, 0.0, 100.0), TrackSegment(1, 40.0, 45.0)), frame_seconds=0.08, overlap=0.0
    )
    assert result.hints["co_activity"] == {"s0-s1": 1.0}
    assert set(result.hints) == {"co_activity", "fourth_track_share"}


def test_fewer_than_four_tracks_hold_no_fourth_share(tmp_path):
    assert TIMELINE.hints["fourth_track_share"] == 0.0


def test_the_rows_get_the_same_hints(tmp_path):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    timeline = write_timeline(TIMELINE, tmp_path / "clip-diar.json")
    write_rows(stable, timeline)
    hints = read_overlay(stable)["speaker"]["hints"]
    assert hints == {"co_activity": {"s0-s1": 0.0}, "fourth_track_share": 0.0}


def test_the_rows_and_the_timeline_carry_identical_hints(tmp_path):
    # Plan 5o.1 K2: the timeline is written at millisecond precision; hints derived
    # from the finer numbers of the run would differ from those of a later read.
    #
    # ⚠ The fixture carries EVIDENCE, and that is the point of it. This test existed with an
    # evidence-free result, so it stayed green through the whole window in which
    # `from_payload` dropped `single_speaker_log_bayes` and `samples` -- while asserting, by its
    # own name, the invariant that was broken. A test whose fixture cannot express the failure
    # is not covering it.
    fine = DiarizationResult(
        tracks=(TrackSegment(0, 0.0, 30.00049), TrackSegment(1, 9.00049, 42.0)),
        frame_seconds=0.08,
        overlap=0.0,
        backend="diarizen",
        single_speaker_log_bayes=63.14159,
        log_bayes_embeddings=20,
        samples=({"track": 0, "start": 1.0, "end": 6.0, "closeness": 0.2},),
    )
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    timeline = write_timeline(fine, tmp_path / "clip-diar.json")
    write_rows(stable, timeline)
    written = json.loads(timeline.read_text(encoding="utf-8"))["metadata"]["hints"]
    assert "single_speaker_log_bayes" in written and "samples" in written, "the fixture must bite"
    assert read_overlay(stable)["speaker"]["hints"] == written
    assert fine.hints == written, "what the run holds is what the file says"
    # And a third time, because the product path is run -> write -> read -> apply: a value that
    # survives the write but not the read is dropped on the FIRST run, not only on resume.
    assert read_timeline(timeline).hints == written


def test_the_rows_record_the_backend_that_actually_made_the_timeline(tmp_path):
    """⚠ `-diar-rows.json` is a RECORD artifact: a wrong model name there is never cleaned up.

    `to_payload` resolved the model through the backend registry while the label writer
    hardcoded Sortformer's checkpoint, so every DiariZen run wrote the wrong provenance into a
    file that is kept forever.
    """
    result = DiarizationResult(tracks=(TrackSegment(0, 0.0, 10.0),), frame_seconds=0.08,
                               overlap=0.0, backend="diarizen")
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    write_rows(stable, write_timeline(result, tmp_path / "clip-diar.json"))
    speaker = read_overlay(stable)["speaker"]
    assert speaker["backend"] == "diarizen"
    assert speaker["model"] == BACKENDS["diarizen"].model
    assert "sortformer" not in speaker["model"].lower()


def test_the_environment_variable_is_honoured(tmp_path, monkeypatch):
    (tmp_path / "sortformer_core.ts").write_bytes(b"")
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path))
    assert resolve_artifacts() == tmp_path


def test_an_explicit_path_wins_over_the_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    (explicit / "sortformer_core.ts").write_bytes(b"")
    other = tmp_path / "other"
    other.mkdir()
    (other / "sortformer_core.ts").write_bytes(b"")
    monkeypatch.setenv(ARTIFACT_ENV, str(other))
    assert resolve_artifacts(explicit) == explicit


def test_the_timeline_round_trips(tmp_path):
    path = write_timeline(TIMELINE, tmp_path / "x-diar.json", artifacts_revision="rev1")
    again = read_timeline(path)
    assert again.tracks == TIMELINE.tracks
    assert again.frame_seconds == pytest.approx(TIMELINE.frame_seconds)


def test_the_timeline_records_that_it_is_not_reproducible(tmp_path):
    path = write_timeline(TIMELINE, tmp_path / "x-diar.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["reproducible"] is False


def test_labels_follow_the_timeline_and_stop_where_it_stops():
    speakers = labels_for(stable_payload()["segments"], TIMELINE)
    assert [speaker.track for speaker in speakers] == [0, 1, None]


def test_write_rows_writes_the_labels_beside_the_stable_and_leaves_it_alone(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    before = stable.read_bytes()
    timeline = write_timeline(TIMELINE, tmp_path / "x-diar.json")

    assert write_rows(stable, timeline) == overlay_path(stable)

    assert stable.read_bytes() == before
    lines = read_rows(stable)
    assert lines[0]["speaker"]["track"] == 0
    assert "speaker" not in lines[2], "no track there, so no label"
    assert [line["text"] for line in lines] == [line["text"] for line in stable_payload()["segments"]]
    assert read_overlay(stable)["speaker"]["track_count"] == 2


def test_writing_twice_gives_the_same_bytes(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    timeline = write_timeline(TIMELINE, tmp_path / "x-diar.json")

    first = write_rows(stable, timeline).read_text(encoding="utf-8")
    assert write_rows(stable, timeline).read_text(encoding="utf-8") == first, "deterministic"


def test_no_temporary_file_is_left_behind(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    write_rows(stable, write_timeline(TIMELINE, tmp_path / "x-diar.json"))
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".")] == []


def test_a_plain_stable_has_no_current_rows(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    assert rows_current(stable) is False, "the stable exists long before anyone diarizes it"


def test_the_rows_are_current_after_writing(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    write_rows(stable, write_timeline(TIMELINE, tmp_path / "x-diar.json"))
    assert rows_current(stable) is True


def test_a_corrupt_rows_file_is_not_current(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps(stable_payload()), encoding="utf-8")
    overlay_path(stable).write_text("{not json", encoding="utf-8")
    assert rows_current(stable) is False


def test_an_unreadable_rows_file_is_left_alone(tmp_path, machine):
    """It may hold names a person wrote, and a file written in its place would record the split choice again,
    possibly against the products that already count its lines: the run says so and writes nothing."""
    from finesub.stages import _run_diarization_stage

    paths = _paths(tmp_path, timeline=True)
    broken = '{"rows": [{"speaker": {"labels": [{"name": "Aki"}]}},]}'
    paths.diar_rows_json.write_text(broken, encoding="utf-8")
    for _ in range(2):
        _run_diarization_stage(paths, diarize="acoustics", device=None, gpu_tier="entry",
                               announce=lambda *a, **k: None, diarize_split=True)
        assert paths.diar_rows_json.read_text(encoding="utf-8") == broken
    assert not rows_current(paths.stable_json)
    paths.diar_rows_json.unlink()
    _run_diarization_stage(paths, diarize="acoustics", device=None, gpu_tier="entry",
                           announce=lambda *a, **k: None, diarize_split=True)
    assert rows_current(paths.stable_json), "deleting it is how to have it written again"


def test_no_stable_has_no_current_rows(tmp_path):
    assert rows_current(tmp_path / "missing-stable.json") is False


@pytest.fixture
def warnings_said(monkeypatch):
    from finesub import reporting
    import finesub.stages as stages_module

    said = []

    class Recorder(reporting.NullReporter):
        def warning(self, code, message, *, impact="", action=""):
            said.append(code)

    monkeypatch.setattr(reporting, "current_reporter", lambda: Recorder())
    monkeypatch.setattr(stages_module, "current_reporter", lambda: Recorder())
    return said


def test_diarization_follows_a_usable_card(monkeypatch, warnings_said):
    from finesub.speech.runtime import device as device_module
    from finesub.stages import _diarization_device

    # Stated, not inherited: this is an NVIDIA machine's answer. A real Intel
    # card in the test box would otherwise resolve the same request to xpu.
    monkeypatch.setattr(device_module, "cuda_usable", lambda: True)
    monkeypatch.setattr(device_module, "xpu_usable", lambda: False)
    monkeypatch.setattr(device_module, "cuda_unusable_reason", lambda: None)
    assert _diarization_device(None, "entry") == "cuda"
    assert "diarization-on-cpu" not in warnings_said


def test_diarization_does_not_follow_an_intel_card(monkeypatch, warnings_said):
    """Diarization has no XPU backend, so the honest answer is the CPU.

    The request resolves to the Intel GPU (`resolve_device` answers with the
    only accelerator present), and this stage has to say no anyway -- dropping
    to the CPU it actually runs, and announcing the cost instead of silently
    putting the model somewhere it cannot compute.
    """

    from finesub.speech.runtime import device as device_module
    from finesub.stages import _diarization_device

    monkeypatch.setattr(device_module, "cuda_usable", lambda: False)
    monkeypatch.setattr(device_module, "xpu_usable", lambda: True)
    monkeypatch.setattr(device_module, "cuda_unusable_reason", lambda: None)

    assert _diarization_device(None, "entry") == "cpu"
    assert "diarization-on-cpu" in warnings_said


@pytest.mark.parametrize(
    "device, tier", [("cpu", "entry"), (None, "cpu")], ids=["--device cpu", "--gpu-tier cpu"]
)
def test_diarization_runs_on_cpu_when_the_run_does_and_says_how_slow(
    monkeypatch, warnings_said, device, tier
):
    """Both backends run on CPU now, so a run without a GPU still gets labels -- and is told
    the cost, because an hour of audio then takes about fifty minutes."""
    from finesub.speech.runtime import device as device_module
    from finesub.stages import _diarization_device

    monkeypatch.setattr(device_module, "cuda_unusable_reason", lambda: None)
    assert _diarization_device(device, tier) == "cpu"
    assert "diarization-on-cpu" in warnings_said
