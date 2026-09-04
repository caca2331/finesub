"""The VAD stage's artifact: what it stores, and when it refuses to be reused.

The point of splitting the prefix out is that deleting `-aligned.json` to
re-split, re-verify or re-recognize no longer re-runs the VAD pass that would
produce the same intervals. That is only true if what comes back is what went
in -- bit for bit, because everything downstream compares timestamps against
these numbers -- and only safe if a prefix that belongs to different audio is
refused rather than silently reused.

Identity and provenance are separate questions here. The audio's name, size and
mtime decide whether a stored prefix *is* this run's prefix; a run parameter
like `--vad-silero-assist` only says how it was produced, so a mismatch is
reported and the prefix is still used.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from finesub.reporting import NullReporter, reporting_to
from finesub.speech.preprocessing import energy as vad_energy
from finesub.speech.recognition import vad_asr_stage


class _Warnings(NullReporter):
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        self.entries.append((code, message))


def _audio(tmp_path: Path) -> Path:
    source = tmp_path / "clip-vocal.ogg"
    source.write_bytes(b"not really audio, but it has a size and an mtime")
    return source


def _prefix() -> vad_asr_stage.VadPrefix:
    return vad_asr_stage.VadPrefix(
        raw_segments=[{"start": 0.123456, "end": 1.5}],
        segments=[{"start": 0.123456, "end": 1.5, "index": 0}],
        vad_meta={"vad": {"mode": "energy"}, "pause_hints": {"scorer": [0.5]}},
        audio_duration=1.5,
        timing={"vad_sec": 0.25, "energy_sec": 0.125},
        energy_track=vad_energy.VadEnergyTrack(
            energy_db=torch.tensor([-40.5, -12.25, -3.75], dtype=torch.float32),
            hop_sec=0.01,
            frame_sec=0.025,
            energy_mode="weighted",
            frame_dbfs=torch.tensor([-41.0, -13.0, -4.0], dtype=torch.float32),
        ),
    )


def _write(tmp_path: Path, *, assist: bool = False) -> tuple[Path, Path]:
    source = _audio(tmp_path)
    artifact = tmp_path / "clip-vad.json"
    vad_asr_stage.write_vad_prefix(
        artifact, _prefix(), source_path=source, vad_silero_assist=assist
    )
    return source, artifact


def test_a_stored_prefix_comes_back_exactly(tmp_path: Path) -> None:
    source, artifact = _write(tmp_path)

    restored = vad_asr_stage.read_vad_prefix(
        artifact, source_path=source, vad_silero_assist=False
    )

    original = _prefix()
    assert restored is not None
    assert restored.raw_segments == original.raw_segments
    assert restored.segments == original.segments
    assert restored.vad_meta == original.vad_meta
    assert restored.audio_duration == original.audio_duration
    assert restored.timing == original.timing
    # Exact equality, not allclose: a rounded energy track would move the
    # boundary frames the word-start clamp picks.
    assert torch.equal(restored.energy_track.energy_db, original.energy_track.energy_db)
    assert restored.energy_track.energy_db.dtype == torch.float32
    assert torch.equal(
        restored.energy_track.frame_dbfs, original.energy_track.frame_dbfs
    )
    assert restored.energy_track.hop_sec == original.energy_track.hop_sec
    assert restored.energy_track.frame_sec == original.energy_track.frame_sec
    assert restored.energy_track.energy_mode == original.energy_track.energy_mode


def test_a_track_without_frame_dbfs_round_trips_as_absent(tmp_path: Path) -> None:
    source = _audio(tmp_path)
    artifact = tmp_path / "clip-vad.json"
    original = _prefix()
    vad_asr_stage.write_vad_prefix(
        artifact,
        vad_asr_stage.VadPrefix(
            raw_segments=original.raw_segments,
            segments=original.segments,
            vad_meta=original.vad_meta,
            audio_duration=original.audio_duration,
            timing=original.timing,
            energy_track=vad_energy.VadEnergyTrack(
                energy_db=original.energy_track.energy_db,
                hop_sec=0.01,
                frame_sec=0.025,
                energy_mode="weighted",
            ),
        ),
        source_path=source,
        vad_silero_assist=False,
    )

    restored = vad_asr_stage.read_vad_prefix(
        artifact, source_path=source, vad_silero_assist=False
    )

    assert restored is not None
    assert restored.energy_track.frame_dbfs is None


def test_a_prefix_for_other_audio_is_refused(tmp_path: Path) -> None:
    source, artifact = _write(tmp_path)
    source.write_bytes(b"the separator ran again and wrote different bytes")

    assert (
        vad_asr_stage.read_vad_prefix(
            artifact, source_path=source, vad_silero_assist=False
        )
        is None
    )


def test_a_prefix_computed_under_other_switches_is_reused_and_reported(
    tmp_path: Path,
) -> None:
    """A run parameter is provenance, not identity (README_DEV -> 复用的依据是任务身份).

    The assist does change the intervals, but a prefix produced without it is
    still a complete prefix of this audio: resuming neither errors nor loses
    data. So it is reused and the mismatch is warned about. Regenerating
    everything under new parameters is what a new task is for.

    This replaces an earlier test that pinned the opposite -- the switch used
    to sit in the compatibility key, so flipping it silently discarded the
    prefix mid-task.
    """

    source, artifact = _write(tmp_path, assist=False)
    reporter = _Warnings()

    with reporting_to(reporter):
        prefix = vad_asr_stage.read_vad_prefix(
            artifact, source_path=source, vad_silero_assist=True
        )

    assert prefix is not None
    assert [code for code, _ in reporter.entries] == ["vad-prefix-provenance"]


def test_a_legacy_prefix_survives_the_provenance_split(tmp_path: Path) -> None:
    """The migration hazard: whole-dict equality would discard every old file.

    Prefixes written before the split carry `vad_silero_assist` inside
    `source`. Comparing the stored dict for equality would see four keys where
    three are expected and recompute every prefix on disk at the first upgrade
    -- the behaviour this change removes, just happening once instead of
    every time.
    """

    source, artifact = _write(tmp_path, assist=False)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    # Put it back the way the old writer laid it out.
    payload["source"]["vad_silero_assist"] = False
    payload.pop("provenance", None)
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    reporter = _Warnings()
    with reporting_to(reporter):
        same = vad_asr_stage.read_vad_prefix(
            artifact, source_path=source, vad_silero_assist=False
        )
        flipped = vad_asr_stage.read_vad_prefix(
            artifact, source_path=source, vad_silero_assist=True
        )

    assert same is not None, "an old prefix must survive the upgrade"
    assert flipped is not None, "and still be reused when the switch differs"
    # Only the flipped read has anything to say.
    assert [code for code, _ in reporter.entries] == ["vad-prefix-provenance"]


def test_an_unreadable_or_incomplete_artifact_recomputes_rather_than_raises(
    tmp_path: Path,
) -> None:
    source, artifact = _write(tmp_path)
    read = lambda: vad_asr_stage.read_vad_prefix(  # noqa: E731
        artifact, source_path=source, vad_silero_assist=False
    )

    assert read() is not None

    # The sidecar the document points at went away.
    vad_asr_stage.vad_prefix_energy_path(artifact).unlink()
    assert read() is None

    # The sidecar is there but its zip structure is damaged: `np.load` raises
    # `zipfile.BadZipFile`, which subclasses `Exception` directly rather than
    # `OSError` or `ValueError`.
    _write(tmp_path)
    vad_asr_stage.vad_prefix_energy_path(artifact).write_bytes(
        b"PK\x03\x04 a zip magic over what is not a zip"
    )
    assert read() is None

    # A future schema, and then something that is not even JSON.
    _write(tmp_path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["schema"] = vad_asr_stage.VAD_PREFIX_SCHEMA + 1
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert read() is None

    artifact.write_text("{ truncated", encoding="utf-8")
    assert read() is None

    artifact.unlink()
    assert read() is None


def test_the_artifact_is_named_from_the_aligned_stem() -> None:
    aligned = Path("out/clip/clip-aligned.json")

    assert vad_asr_stage.default_vad_prefix_path(aligned) == Path(
        "out/clip/clip-vad.json"
    )
    assert vad_asr_stage.vad_prefix_energy_path(
        vad_asr_stage.default_vad_prefix_path(aligned)
    ) == Path("out/clip/clip-vad-energy.npz")


def test_the_pipeline_names_the_vad_artifact_beside_the_others() -> None:
    from finesub.stages import default_pipeline_paths

    paths = default_pipeline_paths("data/clip.wav", "out/clip/clip.srt")

    assert paths.vad_json == Path("out/clip/clip-vad.json")
    assert paths.vad_json.parent == paths.aligned_json.parent
    # The pipeline lists the sidecar for cleanup, so its two derivations of the
    # name -- here and in the stage that writes it -- have to agree.
    assert paths.vad_energy_npz == vad_asr_stage.vad_prefix_energy_path(
        paths.vad_json
    )


def test_the_vad_artifact_is_removable_after_a_run() -> None:
    """It is reproducible from the vocal track by one CPU pass, so it goes."""

    from finesub_bootstrap import artifacts

    assert "-vad.json" in artifacts.REMOVABLE_SUFFIXES
    assert "-vad-energy.npz" in artifacts.REMOVABLE_SUFFIXES


# --- the second opinion has to survive reuse --------------------------------


def test_the_voiced_fraction_round_trips_with_the_prefix(tmp_path: Path) -> None:
    """The assist only computes it while it runs, so it is stored in the
    artifact; a reused prefix must carry it back out unchanged."""

    source = _audio(tmp_path)
    artifact = tmp_path / "clip-vad.json"
    prefix = _prefix()
    prefix.vad_meta["vad"]["silero_voiced_fraction"] = 0.42
    vad_asr_stage.write_vad_prefix(
        artifact, prefix, source_path=source, vad_silero_assist=True
    )

    restored = vad_asr_stage.read_vad_prefix(
        artifact, source_path=source, vad_silero_assist=True
    )

    assert restored is not None
    assert restored.vad_meta["vad"]["silero_voiced_fraction"] == 0.42
