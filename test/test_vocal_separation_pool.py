from __future__ import annotations

import concurrent.futures as cf
import sys
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


def _fake_gpu(monkeypatch, family: str | None = "cuda") -> None:
    """Pin the stage's view of the machine to this test's answer.

    ``separation`` imports the usability probes by name, so the family the
    stage resolves is read from these copies; `device_type` and
    `is_gpu_device` read only the string they are handed. ``family=None``
    means "no usable GPU".
    """

    monkeypatch.setattr(vocal_separation, "cuda_usable", lambda: family == "cuda")
    monkeypatch.setattr(vocal_separation, "xpu_usable", lambda: family == "xpu")


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

    def fake_build(output_dir: str, output_format: str, batch_size: int, *, device: str):
        nonlocal build_count
        with build_lock:
            build_count += 1
        time.sleep(0.05)
        return _fake_separator()

    _fake_gpu(monkeypatch)
    monkeypatch.setattr(vocal_separation, "_build_separator", fake_build)
    monkeypatch.setattr(
        vocal_separation,
        "_warm_up_shared_roformer",
        lambda model_instance, *, use_amp: warmup_modes.append(use_amp),
    )

    def acquire(index: int):
        barrier.wait()
        return pool.acquire(f"worker-{index}", "flac", 1, use_amp=True, device="cuda")

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


def test_cpu_acquire_keeps_independent_model(monkeypatch) -> None:
    built = _fake_separator()

    _fake_gpu(monkeypatch, family=None)
    monkeypatch.setattr(
        vocal_separation,
        "_build_separator",
        lambda output_dir, output_format, batch_size, *, device: built,
    )

    lease = vocal_separation._acquire_separator(
        "worker",
        "flac",
        1,
        use_amp=False,
        device="cpu",
    )

    assert lease.separator is built
    assert built.use_autocast is False
    assert vocal_separation._SHARED_SEPARATOR_POOL._master is None
    lease.release()
    assert lease.separator is None


def test_the_cpu_tier_reaches_the_weights_even_when_the_card_works(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`--gpu-tier cpu` is a policy, and a policy that stops at the tier table
    is a label.

    The separator never goes through `resolve_device` -- it asks the device
    helpers itself -- so the tier has to be folded into that answer before it
    reaches `_build_separator`, which is what actually pins audio-separator away
    from the GPU. Asserted with a **working** card (`cuda_usable()` returns
    True): that is exactly the case a capability-only check gets wrong, and it
    would pass with the fold deleted if the card were merely absent.
    """

    seen: list[str] = []
    def recording_acquire(*args, device: str, **kwargs):
        seen.append(device)
        return inner_acquire(*args, device=device, **kwargs)

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    _write_striped_source(input_path, sample_rate)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=1)
    # Captured after the fixture, not before it: this recorder only observes
    # the `device` token it is handed, so what it wraps has to be the fake the
    # fixture installs. Bound earlier it wraps the genuine `_acquire_separator`,
    # which imports audio-separator and builds real weights -- green only on a
    # machine carrying the [asr] extra, and red everywhere else.
    inner_acquire = vocal_separation._acquire_separator
    _fake_gpu(monkeypatch)
    monkeypatch.setattr(vocal_separation, "_acquire_separator", recording_acquire)
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 1)

    meta: dict = {}
    vocal_separation.run_vocal_separation(
        input_path,
        output_path=tmp_path / "out.flac",
        block_seconds=0.1,
        pad_seconds=0,
        gpu_tier="cpu",
        metadata_sink=meta,
    )

    assert seen and all(device == "cpu" for device in seen), "the cpu tier did not reach the separator"
    assert meta["device"] == "cpu"

    # The control: the same fixture on a GPU tier does ask for the card, so the
    # assertion above cannot be passing because nothing ever asks.
    seen.clear()
    vocal_separation.run_vocal_separation(
        input_path,
        output_path=tmp_path / "out2.flac",
        block_seconds=0.1,
        pad_seconds=0,
        gpu_tier="standard",
        metadata_sink={},
    )
    assert seen and all(device == "cuda" for device in seen)


def test_an_explicit_cpu_request_keeps_separation_off_a_working_card(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """`--device cpu` is a promise about the WHOLE run, and separation is its
    first and heaviest stage.

    The pipeline used to hand this stage only the tier, so `--gpu-tier
    standard --device cpu` still separated on the GPU -- the card the user
    said to leave alone (review 2026-09-02). Asserted with a working card and
    a GPU tier: the only two things that could still say yes.
    """

    from finesub.reporting import NullReporter, reporting_to

    class _Collect(NullReporter):
        def __init__(self) -> None:
            self.codes: list[str] = []

        def warning(self, code, message, **kwargs) -> None:  # noqa: D102
            self.codes.append(code)

    seen: list[str] = []
    def recording_acquire(*args, device: str, **kwargs):
        seen.append(device)
        return inner_acquire(*args, device=device, **kwargs)

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    _write_striped_source(input_path, sample_rate)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=1)
    # Captured after the fixture, not before it: this recorder only observes
    # the `device` token it is handed, so what it wraps has to be the fake the
    # fixture installs. Bound earlier it wraps the genuine `_acquire_separator`,
    # which imports audio-separator and builds real weights -- green only on a
    # machine carrying the [asr] extra, and red everywhere else.
    inner_acquire = vocal_separation._acquire_separator
    _fake_gpu(monkeypatch)
    monkeypatch.setattr(vocal_separation, "_acquire_separator", recording_acquire)
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 1)

    reporter = _Collect()
    meta: dict = {}
    with reporting_to(reporter):
        vocal_separation.run_vocal_separation(
            input_path,
            output_path=tmp_path / "out.flac",
            block_seconds=0.1,
            pad_seconds=0,
            gpu_tier="standard",
            device="cpu",
            metadata_sink=meta,
        )

    assert seen and all(device == "cpu" for device in seen), "an explicit cpu request reached the GPU"
    assert meta["device"] == "cpu"
    # A choice is not a fallback: no "CUDA is unavailable" for a card that is fine.
    assert "cpu-fallback" not in reporter.codes

    # The control: the same fixture with the default request does use the card.
    seen.clear()
    vocal_separation.run_vocal_separation(
        input_path,
        output_path=tmp_path / "out2.flac",
        block_seconds=0.1,
        pad_seconds=0,
        gpu_tier="standard",
        metadata_sink={},
    )
    assert seen and all(device == "cuda" for device in seen)


def test_an_omitted_device_on_a_machine_without_a_usable_card_says_it_fell_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The default request is the accelerator, so losing it is a fallback.

    The bare CLI and the library call leave `device` out. Read as "the CPU was
    chosen", a GPU tier on a box whose card is unusable would run the heaviest
    stage on the CPU without the warning that says why.
    """

    from finesub.reporting import NullReporter, reporting_to

    class _Collect(NullReporter):
        def __init__(self) -> None:
            self.codes: list[str] = []

        def warning(self, code, message, **kwargs) -> None:  # noqa: D102
            self.codes.append(code)

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    _write_striped_source(input_path, sample_rate)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=1)
    _fake_gpu(monkeypatch, family=None)
    monkeypatch.setattr(vocal_separation, "cuda_unusable_reason", lambda: "no CUDA driver")
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 1)

    reporter = _Collect()
    meta: dict = {}
    with reporting_to(reporter):
        vocal_separation.run_vocal_separation(
            input_path,
            output_path=tmp_path / "out.flac",
            block_seconds=0.1,
            pad_seconds=0,
            gpu_tier="standard",
            metadata_sink=meta,
        )

    assert meta["device"] == "cpu"
    assert reporter.codes == ["cpu-fallback"]


def test_an_intel_gpu_is_named_as_the_separation_device(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The token has to name the family that runs, not the default one.

    The default request, ``"cuda"``, means "the accelerator", and on this
    machine that is the Intel GPU. Recording `"cuda"` would make the artefact metadata -- and the
    support notice derived from it -- describe a CUDA run that never happened.
    """

    seen: list[str] = []
    def recording_acquire(*args, device: str, **kwargs):
        seen.append(device)
        return inner_acquire(*args, device=device, **kwargs)

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    _write_striped_source(input_path, sample_rate)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=1)
    inner_acquire = vocal_separation._acquire_separator
    _fake_gpu(monkeypatch, family="xpu")
    # support_levels answers from a real probe; pin the placement it is handed
    # so the assertion is about the stage rather than about the machine.
    monkeypatch.setattr(
        vocal_separation, "note_accelerated_stage", lambda stage, device: device
    )
    monkeypatch.setattr(vocal_separation, "_acquire_separator", recording_acquire)
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 1)

    meta: dict = {}
    vocal_separation.run_vocal_separation(
        input_path,
        output_path=tmp_path / "out.flac",
        block_seconds=0,
        pad_seconds=0,
        gpu_tier="standard",
        metadata_sink=meta,
    )

    assert seen and all(device == "xpu" for device in seen)
    assert meta["device"] == "xpu"
    # fp16 autocast is pending evidence on this card, so the run is fp32: the
    # metadata must not promise a precision the stage did not use.
    assert meta["amp"] is False


def test_an_unusable_intel_gpu_falls_back_to_cpu_with_the_intel_reason(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A requested Intel GPU that cannot run is a fallback, and says so.

    The wording matters: "XPU is the default ..." sends the user looking for a
    product called XPU. The subject is the card they asked for, and the reason
    is the XPU reason, not the CUDA one.
    """

    from finesub.reporting import NullReporter, reporting_to

    class _Collect(NullReporter):
        def __init__(self) -> None:
            self.messages: list[tuple[str, str]] = []

        def warning(self, code, message, **kwargs) -> None:  # noqa: D102
            self.messages.append((code, message))

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    _write_striped_source(input_path, sample_rate)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=1)
    _fake_gpu(monkeypatch, family=None)
    monkeypatch.setattr(
        vocal_separation,
        "xpu_unusable_reason",
        lambda: "the driver is missing",
    )
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 1)

    reporter = _Collect()
    meta: dict = {}
    with reporting_to(reporter):
        vocal_separation.run_vocal_separation(
            input_path,
            output_path=tmp_path / "out.flac",
            block_seconds=0,
            pad_seconds=0,
            gpu_tier="standard",
            device="xpu",
            metadata_sink=meta,
        )

    assert meta["device"] == "cpu"
    assert [code for code, _ in reporter.messages] == ["cpu-fallback"]
    code, message = reporter.messages[0]
    assert message == (
        "the Intel GPU was requested for vocal separation but the driver is "
        "missing; falling back to CPU."
    )


def test_a_cuda_box_asked_for_the_intel_gpu_falls_back_to_cpu(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Asking for one family must not quietly run the other.

    `--device xpu` on an NVIDIA machine used to mean "any GPU": the stage took
    the CUDA card and recorded an Intel run. The contract allows the silent
    substitution only the other way round -- `cuda` is the default and may land
    on Intel -- so this is a fallback with a warning, not a reprogramming of
    the run behind the user's back.
    """

    from finesub.reporting import NullReporter, reporting_to

    class _Collect(NullReporter):
        def __init__(self) -> None:
            self.messages: list[tuple[str, str]] = []

        def warning(self, code, message, **kwargs) -> None:  # noqa: D102
            self.messages.append((code, message))

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    _write_striped_source(input_path, sample_rate)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=1)
    _fake_gpu(monkeypatch, family="cuda")
    monkeypatch.setattr(
        vocal_separation,
        "xpu_unusable_reason",
        lambda: "this machine has no Intel GPU",
    )
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 1)

    reporter = _Collect()
    meta: dict = {}
    with reporting_to(reporter):
        vocal_separation.run_vocal_separation(
            input_path,
            output_path=tmp_path / "out.flac",
            block_seconds=0,
            pad_seconds=0,
            gpu_tier="standard",
            device="xpu",
            metadata_sink=meta,
        )

    assert meta["device"] == "cpu"
    assert meta["amp"] is False
    assert [code for code, _ in reporter.messages] == ["cpu-fallback"]
    assert reporter.messages[0][1] == (
        "the Intel GPU was requested for vocal separation but this machine has "
        "no Intel GPU; falling back to CPU."
    )


def test_acquire_pins_autocast_on_the_pooled_clone(monkeypatch) -> None:
    """The clone inherits the master's flag, so acquisition must overwrite it."""

    master = _fake_separator()
    master.use_autocast = False
    pool = vocal_separation._SharedSeparatorPool()

    _fake_gpu(monkeypatch)
    monkeypatch.setattr(vocal_separation.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(vocal_separation, "_SHARED_SEPARATOR_POOL", pool)
    monkeypatch.setattr(
        vocal_separation,
        "_build_separator",
        lambda output_dir, output_format, batch_size, *, device: master,
    )
    monkeypatch.setattr(
        vocal_separation,
        "_warm_up_shared_roformer",
        lambda model_instance, *, use_amp: None,
    )

    lease = vocal_separation._acquire_separator(
        "worker", "flac", 1, use_amp=True, device="cuda"
    )

    assert lease.separator is not master
    assert lease.separator.use_autocast is True
    lease.release()


def test_a_failed_intel_pin_is_recorded_as_the_cpu_run_it_is(monkeypatch) -> None:
    """The plan said xpu; the wrapper never left the CPU. The records follow it.

    A version bump that renames what `_build_separator` steers leaves the
    library on the CPU. Recording the plan would then make the artefact, the
    support-level notice and the worker count describe a run that never touched
    the card -- the one thing a user cannot check afterwards.
    """

    from finesub.reporting import NullReporter, reporting_to

    class _Collect(NullReporter):
        def __init__(self) -> None:
            self.messages: list[tuple[str, str]] = []

        def warning(self, code, message, **kwargs) -> None:  # noqa: D102
            self.messages.append((code, message))

    def announce(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("a CPU landing must not be announced as accelerated")

    monkeypatch.setattr(vocal_separation, "note_accelerated_stage", announce)
    lease = SimpleNamespace(placement="cpu")
    meta: dict = {"device": "xpu", "amp": True}

    reporter = _Collect()
    with reporting_to(reporter):
        actual = vocal_separation._record_separation_placement(
            "xpu", lease, metadata_sink=meta, use_amp=True
        )

    assert actual == "cpu"
    assert meta["device"] == "cpu"
    assert meta["amp"] is False
    assert [code for code, _ in reporter.messages] == ["cpu-fallback"]
    assert reporter.messages[0][1] == (
        "vocal separation could not be steered to the Intel GPU; "
        "running on the CPU."
    )


def _install_counting_separator(monkeypatch, state: dict, *, barrier_parties: int):
    """Fake demix recording concurrency and completion order.

    Stands in for `demix.separate_waveform`, which is where a block is turned
    into audio now that no block ever reaches disk: the stage hands the runner a
    waveform and gets one back.
    """

    counter_lock = threading.Lock()
    started = threading.Barrier(barrier_parties)

    class FakeLease:
        accel = vocal_separation._EAGER_ACCEL

        def __init__(self, output_dir: str, *, device: str = "cpu") -> None:
            # The stage records where the wrapper really landed by reading this
            # back, so the fake has to say the same thing it was asked for.
            self.placement = device
            self.separator = SimpleNamespace(
                output_dir=Path(output_dir),
                model_instance=object(),
                use_autocast=True,
            )

        def release(self) -> None:
            self.separator = None

    def fake_separate_waveform(model_instance, waveform, source_rate, *, use_autocast):
        with counter_lock:
            call_index = state["calls"]
            state["calls"] += 1
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            if call_index < barrier_parties:
                started.wait(timeout=2)
            # Blocks are submitted in order, so the first call is block 0.
            # Making it finish last is what would scramble the output if the
            # stage appended on completion order instead of block order.
            time.sleep(0.06 if call_index == 0 else 0.01)
            audio = np.array(waveform.detach().cpu().numpy(), dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[None, :]
            return audio, source_rate
        finally:
            with counter_lock:
                state["active"] -= 1

    _fake_gpu(monkeypatch)
    monkeypatch.setattr(vocal_separation.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        vocal_separation.demix, "separate_waveform", fake_separate_waveform
    )

    def fake_acquire(
        output_dir,
        output_format,
        batch_size,
        *,
        use_amp,
        accel_backend="eager",
        device: str | None = "cuda",
        sample_rate=vocal_separation.DEFAULT_SEPARATOR_SAMPLE_RATE,
    ):
        state.setdefault("formats", []).append(output_format)
        return FakeLease(output_dir, device=device or "cpu")

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
        gpu_tier="high",
        metadata_sink=meta,
    )

    # 0.3s of audio: the 300s ladder allows exactly one worker whatever the
    # `high` tier permits, so the block pool is never created.
    assert meta["profile_limit"] == 3
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
    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 3)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=3)
    meta: dict = {}

    vocal_separation.run_vocal_separation(
        input_path,
        output_path=output_path,
        block_seconds=0.1,
        pad_seconds=0,
        gpu_tier="high",
        metadata_sink=meta,
    )

    # Blocks are now sized to the workers, so all three run at once.
    assert meta["effective"] == 3
    assert state["peak"] == 3
    assert state["calls"] % 3 == 0     # a whole number of rounds

    actual, actual_sr = sf.read(output_path, dtype="float32")
    assert actual_sr == sample_rate
    np.testing.assert_allclose(actual, source, atol=1e-4)
    assert np.allclose(actual, source, atol=1 / 32768)


def test_blocks_never_reach_disk_and_the_asr_merge_is_uncompressed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """No block file, and the ASR merge does not pay for compression.

    A block used to be written by the separator and read straight back into the
    merged track; it is handed over in memory now. What is left is the merged
    container, and in ASR mode that is itself a temporary the delivery is
    re-encoded from -- so it is uncompressed, while the lossless delivery, which
    *is* the merge, stays FLAC.
    """

    sample_rate = 8000
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.ogg"
    _write_striped_source(input_path, sample_rate)

    monkeypatch.setattr(vocal_separation, "separator_worker_limit", lambda seconds: 2)
    state = {"active": 0, "peak": 0, "calls": 0}
    _install_counting_separator(monkeypatch, state, barrier_parties=2)

    merged_formats: list[str] = []
    real_append = vocal_separation._append_separated_block

    def spy(*, merge_format, **kwargs):
        merged_formats.append(merge_format)
        return real_append(merge_format=merge_format, **kwargs)

    monkeypatch.setattr(vocal_separation, "_append_separated_block", spy)

    vocal_separation.run_vocal_separation(
        input_path,
        output_path=output_path,
        block_seconds=0.1,
        pad_seconds=0,
        gpu_tier="high",
    )

    assert merged_formats and set(merged_formats) == {vocal_separation.ASR_MERGE_FORMAT}
    assert vocal_separation.merge_format_for(vocal_separation.LOSSLESS_MODE) == "flac"
    # Nothing per-block was left behind, because nothing per-block was written.
    assert not list(tmp_path.glob("*block*"))
    delivered = sf.info(output_path)
    assert delivered.format == "OGG"
    # ...and the delivery is the shape every reader of it resamples to anyway.
    assert (delivered.samplerate, delivered.channels) == (
        vocal_separation.ASR_TARGET_SR,
        1,
    )


_MP3 = pytest.mark.skipif("MP3" not in sf.available_formats(), reason="this libsndfile cannot write mp3")


@pytest.mark.parametrize(
    ("src_sr", "suffix"),
    [(44100, "flac"), (48000, "flac"), pytest.param(48000, "mp3", marks=_MP3)],
)
def test_windowed_resampling_matches_a_whole_file_resample(
    tmp_path: Path,
    monkeypatch,
    src_sr: int,
    suffix: str,
) -> None:
    """The seam test: windows must not be visible in the delivered signal.

    Windows are shrunk so a short fixture spans a couple of dozen of them, with
    the production context width kept -- that is the ratio the argument for
    seamlessness rests on. The reference is the same signal resampled in one
    piece, which is what a naive implementation would fail to reproduce at
    every window boundary. 48 kHz and mp3 are what `--no-separate` hands over
    when the source is one (the reads are sequential: see the next test).
    """

    rng = np.random.default_rng(20260818)
    # Broadband and stereo: a seam shows up as a click, which is broadband, and
    # the delivery downmixes -- so both halves of the transform are exercised.
    source = rng.standard_normal((src_sr * 2, 2)).astype(np.float32) * 0.25
    merged = tmp_path / f"merged.{suffix}"
    if suffix == "mp3":
        sf.write(merged, source, src_sr, format="MP3")
    else:
        sf.write(merged, source, src_sr, subtype="PCM_16")

    monkeypatch.setattr(vocal_separation, "_RESAMPLE_WINDOW_SECONDS", 0.2)

    streamed = np.concatenate(list(vocal_separation.stream_asr_frames(merged)))

    stored, _ = sf.read(merged, dtype="float32", always_2d=True)
    whole = vocal_separation.resample_if_needed(
        torch.from_numpy(stored.mean(axis=1)).unsqueeze(0),
        src_sr,
        vocal_separation.ASR_TARGET_SR,
    )[0].squeeze(0).numpy()

    assert streamed.shape == whole.shape
    assert float(np.max(np.abs(streamed - whole))) < 1e-6


def test_the_delivery_reads_its_input_once_front_to_back(tmp_path: Path, monkeypatch) -> None:
    """No seek, one open: libsndfile seeks an mp3 by decoding from the start, so
    a seek per window made the `--no-separate` encode quadratic in the length
    (Nonoka X's engine patch 0012). Windows are sized in seconds, so 48 kHz gets
    20s windows rather than the 0.125s a fixed count of ratio steps gave it."""

    src_sr = 48000
    source = np.zeros((src_sr * 3, 1), dtype=np.float32)
    merged = tmp_path / "merged.flac"
    sf.write(merged, source, src_sr, subtype="PCM_16")
    opened: list[str] = []
    real_init, real_seek = sf.SoundFile.__init__, sf.SoundFile.seek

    def counting_init(self, file, *args, **kwargs):
        opened.append(str(file))
        real_init(self, file, *args, **kwargs)

    def no_seek(self, frames, whence=sf.SEEK_SET):
        # soundfile's own `read()` asks where it is (`seek(0, SEEK_CUR)`) and then seeks to where it already
        # is; neither moves. Any other seek does.
        here = real_seek(self, 0, sf.SEEK_CUR)
        if (whence == sf.SEEK_CUR and frames == 0) or (whence == sf.SEEK_SET and frames == here):
            return real_seek(self, frames, whence)
        raise AssertionError("the ASR delivery seeked its input")

    monkeypatch.setattr(sf.SoundFile, "__init__", counting_init)
    monkeypatch.setattr(sf.SoundFile, "seek", no_seek)
    monkeypatch.setattr(vocal_separation, "_RESAMPLE_WINDOW_SECONDS", 0.5)

    streamed = np.concatenate(list(vocal_separation.stream_asr_frames(merged)))

    assert opened == [str(merged)]
    assert streamed.shape == (vocal_separation.ASR_TARGET_SR * 3,)
    step = vocal_separation._rational_ratio(src_sr, vocal_separation.ASR_TARGET_SR)[0]
    assert vocal_separation._whole_steps(20, src_sr, step) == src_sr * 20
    assert vocal_separation._whole_steps(20, 44100, 441) == 2000 * 441  # 44.1 kHz windows as before


def test_the_vorbis_delivery_is_written_in_bounded_chunks() -> None:
    """libsndfile's Vorbis writer dies on one oversized write.

    Not an exception -- the process goes away and leaves a header-only file, so
    nothing downstream gets a chance to report it. When a resample window was a
    fixed count of source steps, the frame count handed to `write()` moved with
    the source rate: 44.1 kHz produced 320000 per window and survived, 22.05 kHz
    produced 640000 and did not. Both the window size and the source rate are
    tunable, so the chunk cap is what keeps this from coming back.
    """

    # Measured on this stack: a single write survives to ~502k frames.
    assert vocal_separation._OGG_WRITE_FRAMES <= 320_000


def test_separator_sample_rate_accepts_only_the_offered_rates() -> None:
    """The switch is a fixed ladder, not a free number.

    16000 is deliberately absent: it drops whole passages of vocals, which is
    a different failure from the graded cost of the other two.
    """

    assert vocal_separation.DEFAULT_SEPARATOR_SAMPLE_RATE == 44100
    assert vocal_separation.SEPARATOR_SAMPLE_RATES[0] == 44100
    assert 16000 not in vocal_separation.SEPARATOR_SAMPLE_RATES

    resolve = vocal_separation.resolve_separator_sample_rate
    assert resolve(None) == 44100
    for rate in vocal_separation.SEPARATOR_SAMPLE_RATES:
        assert resolve(rate) == rate
    with pytest.raises(SystemExit):
        resolve(16000)


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

    def failing_separate_waveform(model_instance, waveform, source_rate, *, use_autocast):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        vocal_separation.demix, "separate_waveform", failing_separate_waveform
    )

    class FakeLease:
        accel = vocal_separation._EAGER_ACCEL
        placement = "cuda"
        separator = SimpleNamespace(model_instance=object(), use_autocast=True)

        def release(self) -> None:
            nonlocal released
            released = True
            self.separator = None

    _fake_gpu(monkeypatch)
    monkeypatch.setattr(
        vocal_separation,
        "_acquire_separator",
        lambda output_dir, output_format, batch_size, *, use_amp, device="cuda", accel_backend="eager", sample_rate=44100: (
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


def test_asr_merge_container_has_no_four_gigabyte_ceiling() -> None:
    """RF64, not WAV: WAV answers a long source by silently truncating.

    The ASR merge holds the separation at the model's own rate and channel
    count, so a two-hour source is already past a gigabyte and a long one clears
    4 GiB.
    """

    assert vocal_separation.ASR_MERGE_FORMAT.upper() in sf.available_formats()
    assert vocal_separation.ASR_MERGE_FORMAT != "wav"
    assert vocal_separation.MERGE_SUBTYPE in sf.available_subtypes(
        vocal_separation.ASR_MERGE_FORMAT.upper()
    )


# --- JIT first-forward verification in the shared pool ----------------------


def _jit_pool(monkeypatch, *, warmups: list[str], fail_at: set[int] = frozenset()):
    """A pool whose acceleration reports jit and whose warm-ups can be scripted.

    ``warmups`` collects one label per warm-up call; the n-th call (1-based)
    raises when ``n`` is in ``fail_at``.
    """

    from finesub.speech.preprocessing.separator import accel

    pool = vocal_separation._SharedSeparatorPool()
    _fake_gpu(monkeypatch)
    monkeypatch.setattr(vocal_separation.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        vocal_separation, "_build_separator", lambda *a, **k: _fake_separator()
    )
    monkeypatch.setattr(vocal_separation, "_accel_paths", lambda: None)
    rollback = SimpleNamespace(reverted=0)

    def fake_apply(model_instance, backend, paths):
        return accel.AccelerationResult(
            requested=backend, effective=backend, rollback=rollback
        )

    reverted: list[tuple] = []

    def fake_revert(result, exc, paths):
        reverted.append((result, exc, paths))
        return accel.AccelerationResult(
            requested=result.requested,
            effective="eager",
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )

    monkeypatch.setattr(vocal_separation.accel, "apply_acceleration", fake_apply)
    monkeypatch.setattr(vocal_separation.accel, "revert_jit", fake_revert)

    def fake_warmup(model_instance, *, use_amp):
        warmups.append("warm")
        if len(warmups) in fail_at:
            raise RuntimeError(f"forward {len(warmups)} failed")

    monkeypatch.setattr(vocal_separation, "_warm_up_shared_roformer", fake_warmup)
    return pool, reverted


def test_jit_is_exercised_once_before_any_block_runs(monkeypatch) -> None:
    warmups: list[str] = []
    pool, reverted = _jit_pool(monkeypatch, warmups=warmups)

    lease = pool.acquire("w", "flac", 1, use_amp=True, accel_backend="jit", device="cuda")

    # Eager warm-up (rotary cache), then the compiled one that triggers the
    # lazy torch.compile. Nothing to revert.
    assert warmups == ["warm", "warm"]
    assert reverted == []
    assert lease.accel.effective == "jit"
    meta: dict = {}
    vocal_separation._record_applied_accel(meta, lease)
    assert meta == {"accel": "jit", "accel_requested": "jit"}


def test_jit_first_forward_failure_reverts_and_the_run_continues_on_eager(
    monkeypatch,
) -> None:
    warmups: list[str] = []
    pool, reverted = _jit_pool(monkeypatch, warmups=warmups, fail_at={2})

    lease = pool.acquire("w", "flac", 1, use_amp=True, accel_backend="jit", device="cuda")

    # eager warm-up, compiled warm-up (fails), eager warm-up again on the
    # reverted model -- and the lease carries the reason.
    assert warmups == ["warm", "warm", "warm"]
    assert len(reverted) == 1 and str(reverted[0][1]) == "forward 2 failed"
    assert lease.accel.effective == "eager"
    assert pool._master is not None
    meta: dict = {}
    vocal_separation._record_applied_accel(meta, lease)
    assert meta == {
        "accel": "eager",
        "accel_requested": "jit",
        "accel_fallback_reason": "RuntimeError: forward 2 failed",
    }


def test_eager_failure_after_revert_is_the_models_problem_and_propagates(
    monkeypatch,
) -> None:
    warmups: list[str] = []
    pool, reverted = _jit_pool(monkeypatch, warmups=warmups, fail_at={2, 3})

    with pytest.raises(RuntimeError, match="forward 3 failed"):
        pool.acquire("w", "flac", 1, use_amp=True, accel_backend="jit", device="cuda")

    assert len(reverted) == 1
    assert pool._master is None
    assert pool._active_leases == 0


def test_aoti_and_eager_do_not_pay_a_second_warm_up(monkeypatch) -> None:
    for backend in ("aoti", "eager"):
        warmups: list[str] = []
        pool, _ = _jit_pool(monkeypatch, warmups=warmups)
        lease = pool.acquire("w", "flac", 1, use_amp=True, accel_backend=backend, device="cuda")
        assert warmups == ["warm"]
        assert lease.accel.effective == backend


def test_the_cli_leaves_an_absent_flag_to_the_backend(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    monkeypatch.setattr(sys, "argv", ["separate", "clip.wav", "--pad-seconds", "5"])
    monkeypatch.setattr(
        vocal_separation, "run_vocal_separation", lambda path, **options: calls.append(options) or tmp_path
    )
    assert vocal_separation.main() == 0
    assert calls == [{"pad_seconds": 5.0}]
