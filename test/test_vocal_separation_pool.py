from __future__ import annotations

import concurrent.futures as cf
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from finesub.speech.preprocessing.separator import separation as vocal_separation


class _FakeModelInstance:
    def __init__(self) -> None:
        self.model_run = object()
        self.output_dir = "master"
        self.output_format = "ogg"
        self.output_single_stem = "Vocals"
        self.batch_size = 99
        self.cached_sources_map = {"master": object()}
        self.audio_file_path = "master.wav"
        self.audio_file_base = "master"
        self.primary_source = object()
        self.secondary_source = object()
        self.primary_stem_output_path = "master-vocal"
        self.secondary_stem_output_path = "master-instrumental"

    def clear_file_specific_paths(self) -> None:
        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None
        self.secondary_source = None
        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None


def _fake_separator() -> SimpleNamespace:
    return SimpleNamespace(
        model_instance=_FakeModelInstance(),
        output_dir="master",
        output_format="ogg",
        output_single_stem="Vocals",
    )


def test_shared_separator_clone_isolates_state_but_reuses_model() -> None:
    master = _fake_separator()

    worker = vocal_separation._clone_separator_with_shared_model(
        master,
        output_dir="worker",
        output_format="flac",
        batch_size=1,
    )

    assert worker is not master
    assert worker.model_instance is not master.model_instance
    assert worker.model_instance.model_run is master.model_instance.model_run
    assert worker.model_instance.cached_sources_map == {}
    assert worker.model_instance.cached_sources_map is not master.model_instance.cached_sources_map
    assert worker.model_instance.audio_file_path is None
    assert worker.model_instance.output_dir == "worker"
    assert worker.model_instance.output_format == "flac"
    assert master.model_instance.audio_file_path == "master.wav"
    assert master.model_instance.output_dir == "master"


def test_shared_separator_pool_loads_once_for_concurrent_leases(monkeypatch) -> None:
    build_count = 0
    warmup_modes: list[bool] = []
    build_lock = threading.Lock()
    barrier = threading.Barrier(3)
    pool = vocal_separation._SharedSeparatorPool()

    def fake_build(output_dir: str, output_format: str, batch_size: int):
        nonlocal build_count
        with build_lock:
            build_count += 1
        time.sleep(0.05)
        return _fake_separator()

    monkeypatch.setattr(vocal_separation, "_build_separator", fake_build)
    monkeypatch.setattr(
        vocal_separation,
        "_warm_up_shared_roformer",
        lambda model_instance, *, use_amp: warmup_modes.append(use_amp),
    )

    def acquire(index: int):
        barrier.wait()
        return pool.acquire(f"worker-{index}", "flac", 1, use_amp=True)

    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(acquire, index) for index in range(2)]
        barrier.wait()
        leases = [future.result(timeout=2) for future in futures]

    assert build_count == 1
    assert warmup_modes == [True]
    assert leases[0].separator is not leases[1].separator
    assert (
        leases[0].separator.model_instance.model_run
        is leases[1].separator.model_instance.model_run
    )
    assert pool._active_leases == 2

    leases[0].release()
    assert pool._master is not None
    leases[1].release()
    assert pool._master is None
    assert pool._active_leases == 0


def test_non_cuda_acquire_keeps_independent_model(monkeypatch) -> None:
    built = _fake_separator()

    monkeypatch.setattr(vocal_separation, "cuda_usable", lambda: False)
    monkeypatch.setattr(
        vocal_separation,
        "_build_separator",
        lambda output_dir, output_format, batch_size: built,
    )

    lease = vocal_separation._acquire_separator(
        "worker",
        "flac",
        1,
        use_amp=False,
    )

    assert lease.separator is built
    assert built.use_autocast is False
    assert vocal_separation._SHARED_SEPARATOR_POOL._master is None
    lease.release()
    assert lease.separator is None


def test_acquire_pins_autocast_on_the_pooled_clone(monkeypatch) -> None:
    """The clone inherits the master's flag, so acquisition must overwrite it."""

    master = _fake_separator()
    master.use_autocast = False
    pool = vocal_separation._SharedSeparatorPool()

    monkeypatch.setattr(vocal_separation, "cuda_usable", lambda: True)
    monkeypatch.setattr(vocal_separation.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(vocal_separation, "_SHARED_SEPARATOR_POOL", pool)
    monkeypatch.setattr(
        vocal_separation,
        "_build_separator",
        lambda output_dir, output_format, batch_size: master,
    )
    monkeypatch.setattr(
        vocal_separation,
        "_warm_up_shared_roformer",
        lambda model_instance, *, use_amp: None,
    )

    lease = vocal_separation._acquire_separator("worker", "flac", 1, use_amp=True)

    assert lease.separator is not master
    assert lease.separator.use_autocast is True
    lease.release()


def _install_counting_separator(monkeypatch, state: dict, *, barrier_parties: int):
    """Fake separator recording concurrency, block order and requested format."""

    counter_lock = threading.Lock()
    started = threading.Barrier(barrier_parties)

    class FakeSeparator:
        def __init__(self, output_dir: str) -> None:
            self.output_dir = Path(output_dir)

        def separate(self, input_file: str, output_names: dict):
            with counter_lock:
                call_index = state["calls"]
                state["calls"] += 1
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            try:
                if call_index < barrier_parties:
                    started.wait(timeout=2)
                block_index = int(Path(input_file).stem.rsplit("_", 1)[-1])
                # Block 0 finishes last, so an implementation that appended on
                # completion order instead of block order would scramble output.
                time.sleep(0.06 if block_index == 0 else 0.01)
                data, sr = sf.read(input_file, dtype="float32", always_2d=True)
                output_file = self.output_dir / f"{output_names['Vocals']}.wav"
                sf.write(output_file, data, sr, subtype="PCM_16")
                return [str(output_file)]
            finally:
                with counter_lock:
                    state["active"] -= 1

    class FakeLease:
        accel_backend = "eager"

        def __init__(self, separator) -> None:
            self.separator = separator

        def release(self) -> None:
            self.separator = None

    monkeypatch.setattr(vocal_separation, "cuda_usable", lambda: True)
    monkeypatch.setattr(vocal_separation.torch.cuda, "empty_cache", lambda: None)

    def fake_acquire(
        output_dir,
        output_format,
        batch_size,
        *,
        use_amp,
        accel_backend="eager",
    ):
        state.setdefault("formats", []).append(output_format)
        return FakeLease(FakeSeparator(output_dir))

    monkeypatch.setattr(vocal_separation, "_acquire_separator", fake_acquire)
    monkeypatch.setattr(
        vocal_separation,
        "reset_peak_gpu_memory_stats_for_run",
        lambda device: None,
    )
    monkeypatch.setattr(
        vocal_separation,
        "print_peak_resource_usage",
        lambda device, profile, sampler=None: None,
    )


def _write_striped_source(path: Path, sample_rate: int) -> np.ndarray:
    source = np.concatenate(
        [
            np.full(800, -0.5, dtype=np.float32),
            np.zeros(800, dtype=np.float32),
            np.full(800, 0.5, dtype=np.float32),
        ]
    )
    sf.write(path, source, sample_rate, subtype="PCM_16")
    return source


def test_short_input_is_gated_to_one_worker_by_the_duration_ladder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.flac"
    _write_striped_source(input_path, sample_rate)

    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=1)
    meta: dict = {}

    vocal_separation.run_vocal_separation(
        input_path,
        output_path=output_path,
        block_seconds=0.1,
        pad_seconds=0,
        gpu_budget_gb=16,
        metadata_sink=meta,
    )

    # 0.3s of audio: the 300s ladder allows exactly one worker whatever the
    # 16GB profile permits, so the block pool is never created.
    assert meta["profile_limit"] == 4
    assert meta["duration_limit"] == 1
    assert meta["effective"] == 1
    assert meta["amp"] is True
    assert state["peak"] == 1


def test_parallel_blocks_are_merged_in_source_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.flac"
    source = _write_striped_source(input_path, sample_rate)

    # The ladder is exercised above; patch it here so a fixture short enough to
    # stay fast can still drive the real block pool.
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 4)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=4)
    meta: dict = {}

    vocal_separation.run_vocal_separation(
        input_path,
        output_path=output_path,
        block_seconds=0.1,
        pad_seconds=0,
        gpu_budget_gb=16,
        metadata_sink=meta,
    )

    # Blocks are now sized to the workers, so all four run at once.
    assert meta["effective"] == 4
    assert state["peak"] == 4
    assert state["calls"] % 4 == 0     # a whole number of rounds

    actual, actual_sr = sf.read(output_path, dtype="float32")
    assert actual_sr == sample_rate
    np.testing.assert_allclose(actual, source, atol=1e-4)
    assert np.allclose(actual, source, atol=1 / 32768)


def test_blocks_are_separated_as_flac_whatever_the_delivered_format(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A block is a temporary file, so it never carries the lossy format.

    Blocks are written by the separator and read straight back into the final
    track; asking for the delivered format meant encoding the separated vocals
    to Vorbis twice for a file that is deleted moments later.
    """

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.ogg"
    _write_striped_source(input_path, sample_rate)

    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 2)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=2)

    vocal_separation.run_vocal_separation(
        input_path,
        output_path=output_path,
        block_seconds=0.1,
        pad_seconds=0,
        gpu_budget_gb=16,
    )

    assert state["formats"]
    assert set(state["formats"]) == {"flac"}
    delivered = sf.info(output_path)
    assert delivered.format == "OGG"
    # ...and the delivery is the shape every reader of it resamples to anyway.
    assert (delivered.samplerate, delivered.channels) == (
        vocal_separation.ASR_TARGET_SR,
        1,
    )


def test_windowed_resampling_matches_a_whole_file_resample(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The seam test: windows must not be visible in the delivered signal.

    Windows are shrunk so a short fixture spans a couple of dozen of them, with
    the production context width kept -- that is the ratio the argument for
    seamlessness rests on. The reference is the same signal resampled in one
    piece, which is what a naive implementation would fail to reproduce at
    every window boundary.
    """

    rng = np.random.default_rng(20260818)
    src_sr = 44100
    # Broadband and stereo: a seam shows up as a click, which is broadband, and
    # the delivery downmixes -- so both halves of the transform are exercised.
    source = rng.standard_normal((src_sr * 2, 2)).astype(np.float32) * 0.25
    merged = tmp_path / "merged.flac"
    sf.write(merged, source, src_sr, subtype="PCM_16")

    monkeypatch.setattr(vocal_separation, "_RESAMPLE_WINDOW_STEPS", 20)

    streamed = np.concatenate(list(vocal_separation.stream_asr_frames(merged)))

    stored, _ = sf.read(merged, dtype="float32", always_2d=True)
    whole = vocal_separation.resample_if_needed(
        torch.from_numpy(stored.mean(axis=1)).unsqueeze(0),
        src_sr,
        vocal_separation.ASR_TARGET_SR,
    )[0].squeeze(0).numpy()

    assert streamed.shape == whole.shape
    assert float(np.max(np.abs(streamed - whole))) < 1e-6


def test_the_asr_delivery_rate_is_the_rate_every_reader_resamples_to() -> None:
    """The delivery is only worth its shape if it lands on the readers' rate.

    Separation takes the rate from `preprocessing.audio`, which recognition
    reads too. The energy VAD keeps its own copy of the number -- predating
    this and not the separator's to fix -- so that one is pinned here: if the
    two ever diverge, the delivery would be resampled again on the way in and
    the whole point of encoding at 16 kHz would be gone.
    """

    from finesub.speech.preprocessing.audio import TARGET_SR
    from finesub.speech.preprocessing.energy import TARGET_SR as VAD_TARGET_SR

    assert vocal_separation.ASR_TARGET_SR is TARGET_SR
    assert VAD_TARGET_SR == TARGET_SR


def test_output_mode_comes_from_the_suffix_and_rejects_anything_else() -> None:
    assert vocal_separation.output_mode_for(Path("a-vocal.ogg")) == vocal_separation.ASR_MODE
    assert (
        vocal_separation.output_mode_for(Path("a-vocal.flac"))
        == vocal_separation.LOSSLESS_MODE
    )
    # The pipeline writes through an atomic temp whose name keeps the real
    # suffix last; resolving the mode from it must not fall over the dot.
    assert (
        vocal_separation.output_mode_for(Path(".a-vocal.part.ogg"))
        == vocal_separation.ASR_MODE
    )
    with pytest.raises(SystemExit):
        vocal_separation.output_mode_for(Path("a-vocal.wav"))


def test_separator_block_limiter_caps_nested_sessions_globally() -> None:
    limiter = vocal_separation._SeparatorBlockLimiter()
    first = limiter.acquire(2)
    second = limiter.acquire(2)
    third_acquired = threading.Event()

    def acquire_third() -> None:
        lease = limiter.acquire(2)
        third_acquired.set()
        lease.release()

    thread = threading.Thread(target=acquire_third)
    thread.start()
    assert not third_acquired.wait(timeout=0.05)

    first.release()
    assert third_acquired.wait(timeout=1)
    second.release()
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_vocal_separation_releases_shared_lease_after_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.wav"
    # A real WAV: the stage now probes the input and would otherwise fail while
    # converting a placeholder, before reaching the separator this test is about.
    sf.write(str(input_path), np.zeros((1000, 1), dtype="float32"), 16000)
    released = False

    class FailingSeparator:
        @staticmethod
        def separate(input_path: str, output_names: dict):
            raise RuntimeError("boom")

    class FakeLease:
        accel_backend = "eager"
        separator = FailingSeparator()

        def release(self) -> None:
            nonlocal released
            released = True
            self.separator = None

    monkeypatch.setattr(
        vocal_separation,
        "_acquire_separator",
        lambda output_dir, output_format, batch_size, *, use_amp, accel_backend="eager": (
            FakeLease()
        ),
    )
    monkeypatch.setattr(
        vocal_separation,
        "reset_peak_gpu_memory_stats_for_run",
        lambda device: None,
    )
    monkeypatch.setattr(
        vocal_separation,
        "print_peak_resource_usage",
        lambda device, profile, sampler=None: None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        vocal_separation.run_vocal_separation(
            input_path,
            output_path=tmp_path / "output.flac",
            block_seconds=0,
        )

    assert released is True


def test_block_output_stem_strips_pipeline_part_leading_dot() -> None:
    part = Path(".mt8g-cIgoqAy-vocal.part.ogg")
    assert (
        vocal_separation._block_output_stem(part, 2)
        == "mt8g-cIgoqAy-vocal.part-block00002"
    )


def test_find_output_file_resolves_relative_paths_and_dot_stems(
    tmp_path: Path,
) -> None:
    stem = "mt8g-cIgoqAy-vocal.part-block00002"
    written = tmp_path / f"{stem}.ogg"
    written.write_bytes(b"ogg")

    found = vocal_separation._find_output_file(
        [f"{stem}.ogg"],
        f".{stem}",
        tmp_path,
    )
    assert found == written
    assert found.exists()
