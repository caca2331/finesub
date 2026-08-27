"""The prefix artifact: what it promises the transcription half, and what it refuses.

`prepare_vad_asr` writes the state `run_vad_asr` would otherwise have derived
itself, so everything here is about the seam between them -- that a restored
run sees the same values, that a stale or foreign artifact is refused rather
than transcribed against, and that reading one cannot execute what is in it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from finesub.speech.preprocessing import energy as vad_energy
from finesub.speech.recognition import vad_asr_stage


def _track() -> vad_energy.VadEnergyTrack:
    return vad_energy.VadEnergyTrack(
        energy_db=torch.tensor([-30.0, -12.0, -40.0, -11.0]),
        frame_dbfs=torch.tensor([-31.0, -13.0, -41.0, -12.0]),
        hop_sec=0.01,
        frame_sec=0.02,
        energy_mode="weighted",
    )


def _prepare(monkeypatch, tmp_path: Path, *, segments) -> tuple[Path, Path]:
    """Run `prepare_vad_asr` over a stand-in for the decode and the VAD.

    Neither is what this file is about: the decode wants ffmpeg and the VAD
    wants real audio, and the seam being tested is what gets written and read
    back afterwards.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    audio = tmp_path / "vocal.wav"
    audio.write_bytes(b"RIFF-not-really-audio")
    prepared = tmp_path / "prepared.pt"

    monkeypatch.setattr(
        vad_asr_stage,
        "ensure_decodable_input",
        lambda source, _scratch: (source, None),
    )
    monkeypatch.setattr(
        vad_asr_stage,
        "detect_vad_prefix",
        lambda _source, **_kwargs: (
            list(segments),
            {"vad": {"mode": "energy"}},
            12.5,
            {"vad_sec": 0.4},
            _track(),
        ),
    )
    monkeypatch.setattr(
        vad_asr_stage.asr_align,
        "normalize_vad_segments",
        lambda raw, _duration: list(raw),
    )

    vad_asr_stage.prepare_vad_asr(audio, prepared_path=prepared)
    return audio, prepared


def test_the_transcription_half_restores_exactly_what_the_prefix_computed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    segments = [{"start": 0.0, "end": 1.5}, {"start": 3.0, "end": 4.25}]
    audio, prepared = _prepare(monkeypatch, tmp_path, segments=segments)

    payload = vad_asr_stage._load_prepared_vad(audio, prepared)

    assert payload["segments"] == segments
    assert payload["audio_duration"] == 12.5
    assert payload["vad_meta"] == {"vad": {"mode": "energy"}}
    restored = vad_asr_stage._energy_track_from_payload(payload["energy_track"])
    original = _track()
    assert torch.equal(restored.energy_db, original.energy_db)
    assert torch.equal(restored.frame_dbfs, original.frame_dbfs)
    assert restored.energy_mode == original.energy_mode
    assert (restored.hop_sec, restored.frame_sec) == (
        original.hop_sec,
        original.frame_sec,
    )


def test_an_artifact_for_different_audio_is_refused_not_transcribed_against(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The failure this digest exists to prevent is a silent one.

    Segmentation from one file applied to another produces an aligned artifact
    that is wrong everywhere and looks fine.
    """

    _audio, prepared = _prepare(monkeypatch, tmp_path, segments=[{"start": 0.0}])
    other = tmp_path / "other.wav"
    other.write_bytes(b"RIFF-a-different-file")

    with pytest.raises(RuntimeError, match="does not match"):
        vad_asr_stage._load_prepared_vad(other, prepared)

    assert not vad_asr_stage.prepared_vad_matches(other, prepared)


def test_reading_an_artifact_cannot_execute_what_is_in_it(tmp_path: Path) -> None:
    """The artifact travels: an earlier run, another machine, shared scratch.

    `weights_only=True` is what keeps reading one from reconstructing whatever
    objects it names. The digest cannot do this job -- comparing it happens
    after the payload has already been rebuilt.
    """

    hostile = tmp_path / "hostile.pt"
    torch.save(
        {
            "schema": vad_asr_stage.PREPARED_VAD_SCHEMA,
            "input_sha256": "whatever",
            "segments": [],
            "payload": vad_energy.VadEnergyTrack(
                energy_db=torch.zeros(1),
                frame_dbfs=None,
                hop_sec=0.01,
                frame_sec=0.02,
                energy_mode="weighted",
            ),
        },
        hostile,
    )

    # Any class the loader would have had to construct is enough to show it is
    # not constructing classes.
    with pytest.raises(Exception) as refused:
        vad_asr_stage._read_prepared_vad(hostile)
    assert "weights_only" in str(refused.value) or "Unsupported" in str(refused.value)


def test_a_schema_that_does_not_match_is_recomputed_rather_than_migrated(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "stale.pt"
    torch.save(
        {"schema": vad_asr_stage.PREPARED_VAD_SCHEMA + 1, "segments": [{"start": 0.0}]},
        stale,
    )

    with pytest.raises(RuntimeError, match="schema mismatch"):
        vad_asr_stage._read_prepared_vad(stale)
    assert not vad_asr_stage.prepared_vad_has_speech(stale)


def test_silence_and_absence_both_read_as_no_speech(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A caller deciding whether the GPU is needed at all asks this."""

    _audio, prepared = _prepare(monkeypatch, tmp_path, segments=[])

    assert not vad_asr_stage.prepared_vad_has_speech(prepared)
    assert not vad_asr_stage.prepared_vad_has_speech(tmp_path / "never-written.pt")

    _audio, with_speech = _prepare(
        monkeypatch,
        tmp_path / "second",
        segments=[{"start": 0.0, "end": 1.0}],
    )
    assert vad_asr_stage.prepared_vad_has_speech(with_speech)


def _aligned(path: Path, align_meta: dict) -> Path:
    path.write_text(
        json.dumps(
            {
                "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
                "vad_timeline": {"intervals": [{"start": 0.0, "end": 1.0}]},
                "metadata": {"asr_align": align_meta},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_finalizing_an_artifact_that_already_has_the_evidence_changes_nothing(
    tmp_path: Path,
) -> None:
    """What lets a resumed run call this without checking first."""

    aligned = _aligned(tmp_path / "aligned.json", {"qwen_verify": {"suspects": 2}})
    before = aligned.read_text(encoding="utf-8")

    vad_asr_stage.finalize_qwen_verification(tmp_path / "vocal.wav", aligned)

    assert aligned.read_text(encoding="utf-8") == before


def test_finalizing_with_verification_off_is_a_no_op(tmp_path: Path) -> None:
    aligned = _aligned(tmp_path / "aligned.json", {})
    before = aligned.read_text(encoding="utf-8")

    vad_asr_stage.finalize_qwen_verification(
        tmp_path / "vocal.wav", aligned, qwen_verify="off"
    )

    assert aligned.read_text(encoding="utf-8") == before


def test_an_unusable_verification_mode_fails_before_anything_is_read(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported qwen verification mode"):
        vad_asr_stage.finalize_qwen_verification(
            tmp_path / "vocal.wav",
            tmp_path / "does-not-exist.json",
            qwen_verify="yes",
        )
