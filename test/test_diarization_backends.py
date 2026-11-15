"""The backend seam: what must hold before a second diarization model can enter.

Everything downstream of the timeline reads only `(frames, speakers)` activity and `TrackSegment`,
so adding a backend should touch `stage.py` and nothing else. These pin the four things that make
that true, and the artifact rule the owner set on 2026-09-19: a cached timeline is NOT invalidated
by asking for a different backend -- backend identity is provenance, it warns.
"""

from __future__ import annotations

import json

import pytest

from finesub.speech.diarization import diarizen_backend
from finesub.speech.diarization.backend import ARTIFACT_ENV
from finesub.speech.diarization import stage as diarization
from finesub.speech.diarization.segments import TrackSegment


def _result(backend: str) -> diarization.DiarizationResult:
    return diarization.DiarizationResult(
        tracks=(TrackSegment(0, 0.0, 1.0), TrackSegment(1, 1.0, 2.0)),
        frame_seconds=0.08,
        overlap=0.0,
        backend=backend,
    )


# --- the registry ---------------------------------------------------------------------------

def test_the_default_backend_lives_in_the_stage_and_not_in_argparse():
    """`README_DEV.md` 开发原则: one copy of a default, in the backend.

    The front end passes `None`; this is what answers. A second copy in argparse is what
    `test_option_defaults` exists to stop, and it would silently win over config.
    """
    from finesub.pipeline import build_parser

    assert build_parser().parse_args(["clip.wav"]).diarize_backend is None
    assert diarization.resolve_backend(None).name == diarization.DEFAULT_BACKEND


def test_an_unknown_backend_names_the_ones_that_exist():
    with pytest.raises(ValueError) as error:
        diarization.resolve_backend("whisper-x")
    assert all(name in str(error.value) for name in ("nemotron3", "diarizen", "sortformer"))


def test_every_backend_declares_its_own_frame_rate():
    """Each backend says what grid its activity is on, and they no longer agree.

    ⚠ Until 2026-09-20 both sat at 80 ms and this test pinned the SET to `{0.08}`, which read as
    a property of the format. It was a fact about two models: DiariZen now segments at its native
    20 ms (A3, pre-registered and measured as a net gain), so the pin is per backend. The schema
    survives it because `-diar.json` records `frame_seconds` and `from_payload` reads it back --
    which is the reason this field is on `Backend` at all.
    """
    for backend in diarization.BACKENDS.values():
        assert backend.frame_seconds > 0
        assert backend.model, f"{backend.name} must name what produced the timeline"
    assert diarization.BACKENDS["sortformer"].frame_seconds == 0.08
    assert diarization.BACKENDS["diarizen"].frame_seconds == 0.02
    assert diarization.BACKENDS["nemotron3"].frame_seconds == 0.02


def test_diarizen_hands_over_the_grid_it_declares(monkeypatch):
    """⚠ The declared rate and the returned array must agree, or every boundary is mis-timed.

    `run_diarization` segments `activity`'s output with `backend.frame_seconds`. If the backend
    declared 20 ms and still returned the 80 ms reduction -- or the reverse -- every track would
    land at a quarter (or four times) of its real time, and nothing would raise. So the array
    length is checked against the declared step on a synthetic run.
    """
    import numpy as np
    import torch

    from finesub.speech.diarization import (
        diarizen_assemble, diarizen_backend, diarizen_cluster, diarizen_embed, diarizen_runner,
    )
    from finesub.speech.diarization.diarizen_cluster import ClusterResult

    frames_native = 800   # 16 s of native 20 ms frames
    segmentation = np.zeros((1, frames_native, 4), dtype=np.float32)
    segmentation[0, :400, 0] = 1.0
    monkeypatch.setattr(diarizen_runner, "segment",
                        lambda *a, **k: diarizen_runner.LocalSegmentation(segmentation, 16.0, 1.6))
    monkeypatch.setattr(diarizen_runner, "median_filtered", lambda data: data)
    monkeypatch.setattr(diarizen_embed, "embed", lambda *a, **k: np.zeros((1, 4, 8)))
    monkeypatch.setattr(diarizen_cluster, "cluster", lambda *a, **k: ClusterResult(
        hard=np.zeros((1, 4), dtype=np.int8), soft=np.ones((1, 4, 1)),
        centroids=np.zeros((1, 8)), centroid_of_track=np.zeros(1, dtype=np.int16),
        single_speaker_log_bayes=0.0, log_bayes_embeddings=0,
        centroid_weights=np.ones(1)))

    output = diarizen_backend.activity(torch.zeros(1, 16000 * 16), handle=".", device="cuda")
    expected = len(diarizen_assemble.assemble(segmentation, np.zeros((1, 4), dtype=np.int8)))
    assert output.activity.shape[0] == expected, "the array is on the native grid"
    seconds = output.activity.shape[0] * diarizen_backend.BACKEND.frame_seconds
    assert 15.0 < seconds < 17.5, f"at the declared step it spans {seconds:.1f} s of a 16 s input"


# --- the artifact carries which backend made it ----------------------------------------------

def test_the_timeline_records_the_backend_and_its_model():
    payload = _result("diarizen").to_payload(artifacts_revision="r1", parameters={})
    assert payload["metadata"]["backend"] == "diarizen"
    assert payload["metadata"]["model"] == diarization.BACKENDS["diarizen"].model
    assert diarization.DiarizationResult.from_payload(payload).backend == "diarizen"


def test_a_timeline_written_before_backends_existed_still_reads():
    """Those files have no `backend` key and were all Sortformer's."""
    legacy = {
        "metadata": {"model": diarization.MODEL_NAME, "parameters": {}},
        "tracks": [{"track": 0, "start": 0.0, "end": 1.0}],
    }
    assert diarization.DiarizationResult.from_payload(legacy).backend == "sortformer"


# --- the owner's rule: a cached timeline is reused, with a warning --------------------------

def test_a_cached_timeline_from_another_backend_is_reused_and_warned_about(tmp_path, capsys):
    """Owner 2026-09-19: asking for a different backend does not expire the artifact.

    It is neither corrupt nor unreadable, so by the repository's reuse rule it stays -- the way to
    recompute is to delete it, as for every other stage. The warning is the whole point: serving a
    Sortformer timeline to a run whose command line said `diarizen` is the part nobody would spot.
    """
    from finesub import stages

    path = tmp_path / "clip-diar.json"
    path.write_text(
        json.dumps(_result("sortformer").to_payload(artifacts_revision="", parameters={})),
        encoding="utf-8",
    )

    stages._warn_on_backend_mismatch(path, "diarizen", diarization)
    warning = capsys.readouterr().err
    assert "sortformer" in warning and "diarizen" in warning
    assert "Delete it" in warning, "the warning has to say how to get the other backend's answer"

    stages._warn_on_backend_mismatch(path, "sortformer", diarization)
    assert capsys.readouterr().err == "", "no warning when the cache is what was asked for"

    stages._warn_on_backend_mismatch(path, None, diarization)
    assert capsys.readouterr().err == "", "nobody asked for a backend, so nothing to disagree with"


def test_an_unreadable_timeline_does_not_make_the_warning_the_error(tmp_path, capsys):
    """The consumer reports a broken cache properly; this probe must not pre-empt it."""
    from finesub import stages

    path = tmp_path / "broken-diar.json"
    path.write_text("{not json", encoding="utf-8")
    stages._warn_on_backend_mismatch(path, "diarizen", diarization)
    assert capsys.readouterr().err == ""


def test_a_cache_from_the_other_backend_is_quiet_unless_a_backend_was_named(tmp_path, capsys):
    """⚠ The default is environment-dependent, so an implicit mismatch is not a warning.

    Export DiariZen once and the default flips for every later run. If that printed a warning,
    every older task's resume would say "produced by sortformer, not diarizen" for a run where
    nothing on the command line ever said diarizen -- which is exactly what the resume rule covers:
    current parameters do not retroactively redefine finished stages.

    An earlier version of this test asserted the opposite, on the reasoning that a default is an
    answer too. That was right while the default was a constant and wrong once it became a probe.
    """
    from finesub import stages

    path = tmp_path / "clip-diar.json"
    result = diarization.DiarizationResult(tracks=(), frame_seconds=0.08, overlap=0.0,
                                           backend="diarizen")
    diarization.write_timeline(result, path)

    stages._warn_on_backend_mismatch(path, None, diarization)
    assert capsys.readouterr().err == "", "nobody asked for sortformer; the default did"

    stages._warn_on_backend_mismatch(path, "sortformer", diarization)
    captured = capsys.readouterr().err
    assert "diarizen" in captured and "sortformer" in captured, "a NAMED backend still warns"
    assert "Delete it to recompute" in captured


# --- resolving the artifacts ------------------------------------------------------------------

def _hosted(monkeypatch, *, fetched: bool, directory):
    """Stand in for the hosted-build fetch: record which build was asked for, return `directory`."""
    from finesub.speech.runtime import hf_weights

    asked = []
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    monkeypatch.setattr(diarizen_backend, "_already_fetched", lambda build: fetched)

    def prepare_directory(build):
        asked.append(build)
        return directory

    monkeypatch.setattr(hf_weights, "prepare_directory", prepare_directory)
    return asked


@pytest.mark.parametrize("device, build", [("cuda", "diarizen-ts-cuda"), ("cpu", "diarizen-ts-cpu")])
def test_diarizen_fetches_the_build_traced_for_its_device(tmp_path, monkeypatch, device, build):
    """⚠ A traced graph keeps the device it was traced on, in both directions (measured
    2026-09-21: the CUDA build raises inside TorchScript on CPU, the CPU build on CUDA). So the
    device picks the build, before anything is loaded."""
    asked = _hosted(monkeypatch, fetched=True, directory=tmp_path)
    assert diarization.BACKENDS["diarizen"].ensure(None, device) == tmp_path
    assert asked == [build]


def test_a_local_export_wins_over_the_hosted_build(tmp_path, monkeypatch):
    asked = _hosted(monkeypatch, fetched=True, directory=None)
    _complete_diarizen_export(tmp_path)
    assert diarization.BACKENDS["diarizen"].ensure(tmp_path, "cuda") == tmp_path
    assert asked == [], "a developer's own export is not second-guessed by a download"


def _traced_segmentation(root, device):
    """A stand-in segmentation archive whose recorded code carries what a trace on `device` does."""
    import zipfile

    constant = b"torch.device('cuda:0')" if device == "cuda" else b'torch.device("cpu")'
    with zipfile.ZipFile(root / "diarizen_segmentation.ts", "w") as archive:
        archive.writestr("diarizen_segmentation/code/__torch__/model.py", b"x = " + constant)


@pytest.mark.parametrize("traced_on, device", [("cuda", "cpu"), ("cpu", "cuda")])
def test_a_local_export_for_the_other_device_is_refused_before_anything_loads(
    tmp_path, monkeypatch, traced_on, device
):
    """⚠ Loading it would raise a bare RuntimeError from inside TorchScript, which the stage does
    not catch -- the run would end after the ASR. Read from the graph itself, so an export made
    before anyone recorded its device is covered too (Codex review 2026-09-21: the old CUDA
    exports carry `cuda:0` constants and no metadata saying so)."""
    _hosted(monkeypatch, fetched=True, directory=None)
    _complete_diarizen_export(tmp_path)
    _traced_segmentation(tmp_path, traced_on)
    with pytest.raises(diarization.DiarizationUnavailable) as error:
        diarization.BACKENDS["diarizen"].ensure(tmp_path, device)
    assert f"--device {device}" in str(error.value), "it says how to get the right one"

    _traced_segmentation(tmp_path, device)
    assert diarization.BACKENDS["diarizen"].ensure(tmp_path, device) == tmp_path


def test_a_local_export_whose_device_cannot_be_read_is_not_gambled_on(tmp_path, monkeypatch):
    _hosted(monkeypatch, fetched=True, directory=None)
    _complete_diarizen_export(tmp_path)
    (tmp_path / "diarizen_segmentation.ts").write_bytes(b"")  # not an archive
    with pytest.raises(diarization.DiarizationUnavailable) as error:
        diarization.BACKENDS["diarizen"].ensure(tmp_path, "cpu")
    assert "unknown device" in str(error.value)


def test_a_corrupt_download_costs_the_labels_not_the_run(monkeypatch):
    """`VerificationMismatch` is raised on purpose -- a model known to be wrong must not load --
    but it is not what the stage catches. It has to arrive as `DiarizationUnavailable`."""
    from finesub.speech.runtime import hf_weights
    from finesub_bootstrap.hf_verify import VerificationMismatch

    _hosted(monkeypatch, fetched=True, directory=None)

    def corrupt(build):
        raise VerificationMismatch("sha256 differs")

    monkeypatch.setattr(hf_weights, "prepare_directory", corrupt)
    with pytest.raises(diarization.DiarizationUnavailable) as error:
        diarization.BACKENDS["diarizen"].ensure(None, "cuda")
    assert "failed verification" in str(error.value)


def test_the_licence_is_said_when_a_fetch_is_needed_and_only_then(tmp_path, monkeypatch):
    from finesub import reporting

    said = []

    class Recorder(reporting.NullReporter):
        def warning(self, code, message, *, impact="", action=""):
            said.append((code, message))

    # `ensure` looks the reporter up at call time, from the module.
    monkeypatch.setattr(reporting, "current_reporter", lambda: Recorder())

    _hosted(monkeypatch, fetched=False, directory=tmp_path)
    diarization.BACKENDS["diarizen"].ensure(None, "cuda")
    assert [code for code, _ in said] == ["diarizen-download"]
    assert "CC BY-NC 4.0" in said[0][1] and "非商业" in said[0][1]

    said.clear()
    _hosted(monkeypatch, fetched=True, directory=tmp_path)
    diarization.BACKENDS["diarizen"].ensure(None, "cuda")
    assert said == [], "a cached build is used silently"


def test_a_failed_fetch_costs_the_labels_with_a_way_out(monkeypatch):
    """No weights and no network: the stage catches `DiarizationUnavailable`, so the run goes on."""
    _hosted(monkeypatch, fetched=True, directory=None)
    with pytest.raises(diarization.DiarizationUnavailable) as error:
        diarization.BACKENDS["diarizen"].ensure(None, "cpu")
    message = str(error.value)
    assert "diarizen-ts-cpu" in message, "which build it tried"
    assert ARTIFACT_ENV in message and "diarizen_export" in message, "the local way out"
    assert "default backend" in message, "and the way back to the default"


def _complete_diarizen_export(root, device="cuda"):
    """Every file a run opens; the segmentation one is a stand-in archive traced on `device`."""
    for name in diarizen_backend.ARTIFACT_NAMES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    _traced_segmentation(root, device)
    return root


def test_diarizen_resolves_explicit_and_environment_directories(tmp_path, monkeypatch):
    """Both backends answer 'where are the artifacts' the same way, from the same variable."""
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    _complete_diarizen_export(tmp_path)

    assert diarization.BACKENDS["diarizen"].ensure(tmp_path) == tmp_path
    monkeypatch.setenv(ARTIFACT_ENV, str(tmp_path))
    assert diarization.BACKENDS["diarizen"].ensure(None) == tmp_path


@pytest.mark.parametrize("missing", diarizen_backend.ARTIFACT_NAMES)
def test_diarizen_refuses_a_half_exported_directory(tmp_path, monkeypatch, missing):
    """⚠ Every file the run opens, not just the first one.

    Checking only the segmentation model let `auto` through, ran minutes of segmentation, and then
    died inside torch or numpy with an exception the stage does not catch -- so the whole pipeline
    fell over after the ASR was already done. A missing model must cost the labels, not the run.
    """
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    _complete_diarizen_export(tmp_path)
    (tmp_path / missing).unlink()

    with pytest.raises(diarization.DiarizationUnavailable):
        diarization.BACKENDS["diarizen"].ensure(tmp_path)


def test_diarizen_does_not_accept_a_sortformer_directory(tmp_path, monkeypatch):
    """⚠ One variable, two backends: the directory that satisfies one must not satisfy the other.

    Both read `FINESUB_DIARIZATION_ARTIFACTS`, so a developer with Sortformer exported and
    DiariZen not would otherwise get a load failure deep inside torch instead of the message
    that tells them which export to run.
    """
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    (tmp_path / "sortformer_core.ts").write_bytes(b"")
    with pytest.raises(diarization.DiarizationUnavailable):
        diarization.BACKENDS["diarizen"].ensure(tmp_path)


# --- the frame grid ---------------------------------------------------------------------------

def test_reduction_to_the_coarse_grid_keeps_a_speaker_who_spoke_at_all():
    """⚠ `max`, not mean: 20 ms of speech inside an 80 ms frame is speech in that frame."""
    import numpy as np

    from finesub.speech.diarization.diarizen_assemble import to_frame_grid

    native = np.zeros((8, 2), dtype=np.float32)
    native[1, 0] = 1.0          # one 20 ms frame of speaker 0, inside the first coarse frame
    native[4:8, 1] = 1.0        # all of the second coarse frame for speaker 1

    coarse = to_frame_grid(native, source_step=0.02, target_step=0.08)
    assert coarse.shape == (2, 2)
    assert coarse[0, 0] == 1.0, "a speaker active for part of the frame is active in it"
    assert coarse[0, 1] == 0.0
    assert list(coarse[1]) == [0.0, 1.0]


def test_reduction_drops_a_trailing_partial_frame_rather_than_padding_it():
    """Padding would claim the speaker stopped at a boundary the audio never reached."""
    import numpy as np

    from finesub.speech.diarization.diarizen_assemble import to_frame_grid

    native = np.ones((9, 1), dtype=np.float32)
    assert len(to_frame_grid(native, source_step=0.02, target_step=0.08)) == 2


def test_reduction_to_the_same_rate_is_the_identity():
    """A backend already at the target rate must not be rewritten on the way past."""
    import numpy as np

    from finesub.speech.diarization.diarizen_assemble import to_frame_grid

    native = np.ones((4, 3), dtype=np.float32)
    assert to_frame_grid(native, source_step=0.08, target_step=0.08) is native
