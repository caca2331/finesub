"""CLI wrapper around audio-separator for vocal extraction."""

from __future__ import annotations

import argparse
import copy
import concurrent.futures as cf
import gc
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from audio_separator.separator import Separator

import numpy as np
import soundfile as sf
import torch

from finesub_bootstrap.model_caches import SEPARATOR_CHECKPOINT

from .... import config as app_config
from ....paths import resolve_separator_model_dir
from ....run_metadata import record_scratch_file
from ....reporting import (
    bind_reporter,
    current_reporter,
    format_clock,
    libraries_quieted,
    quieted_libraries,
    reporting_to,
    terminal_reporter,
)
from ...runtime.device import cuda_unusable_reason, cuda_usable
from ...runtime.gpu_stage_gate import GPU_STAGE_GATE, GpuStageLease
from ...runtime import stall_watchdog
from ...runtime.resources import (
    DEFAULT_GPU_TIER,
    get_resource_profile,
    AUTO_GPU_TIER,
    gpu_tier_cli_choices,
    warn_if_vram_is_short,
    gpu_tier_help,
)
from . import accel, demix
from ..audio import (
    # The rate every consumer of the ASR delivery resamples to, taken from the
    # shared constant rather than restated here.
    TARGET_SR as ASR_TARGET_SR,
    as_numpy_float32,
    ensure_decodable_input,
    get_audio_info,
    load_audio_slice,
    resample_if_needed,
    to_mono,
)
from ...runtime.resource_usage import (
    print_peak_resource_usage,
    reset_peak_gpu_memory_stats_for_run,
    start_stage_memory_sampling,
)

# Single source of truth lives in the bootstrap layer, which path lookups
# can import without pulling in torch.
MODEL_NAME = SEPARATOR_CHECKPOINT

#: The rates the separator may be told to decode and emit at. 44100 is the
#: model's own and the only one its weights were trained for; the other two buy
#: fewer chunks per second of audio -- the chunk is a fixed 352800 samples, so
#: halving the rate halves the chunk count -- and pay for it in separation
#: quality. Neither is recommended; see docs/separator-optimization.md "E12"
#: for what each costs and why 16000 is not on this list at all.
SEPARATOR_SAMPLE_RATES = (44100, 32000, 22050)
DEFAULT_SEPARATOR_SAMPLE_RATE = 44100
BATCH_SIZE = get_resource_profile(DEFAULT_GPU_TIER).vocal_separation_batch_size


class _BlockProgress:
    """One progress line for the whole stage, counted as blocks finish.

    Counted on completion rather than on merge: blocks are merged in order, so
    a run with four workers would have shown nothing until the first block
    landed and then jumped, which reads as a stall exactly when the most work
    is in flight.
    """

    def __init__(self, total: int, *, workers: int, clock=time.monotonic) -> None:
        self.total = total
        self.workers = workers
        self._clock = clock
        self._started = clock()
        self._done = 0
        self._lock = threading.Lock()
        self._reporter = current_reporter()

    def report(self) -> None:
        with self._lock:
            done = self._done
            elapsed = self._clock() - self._started
        parts = [f"{self.workers} 并发"] if self.workers > 1 else []
        parts.append(format_clock(elapsed))
        self._reporter.progress(
            "vocal",
            completed=done,
            total=self.total,
            unit="blocks",
            detail=" · ".join(parts),
        )

    def block_finished(self, future=None) -> None:
        # A cancelled or failed future is "done" too, and counting it would
        # claim work that never landed while the run is being torn down.
        if future is not None and (future.cancelled() or future.exception()):
            return
        with self._lock:
            self._done += 1
        self.report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate vocals from audio.")
    parser.add_argument("input", help="Path to input audio.")
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Path to output vocals file (default: <input>-vocal.ogg). The "
            "suffix picks the delivery: .ogg is the 16 kHz mono ASR track, "
            ".flac the lossless separation."
        ),
    )
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=600.0,
        help="Core block size in seconds (default: 600). Use 0 to disable.",
    )
    parser.add_argument(
        "--pad-seconds",
        type=float,
        default=10.0,
        help="Padding seconds on each side of a block (default: 10).",
    )
    parser.add_argument(
        "--gpu-tier",
        choices=gpu_tier_cli_choices(),
        default=AUTO_GPU_TIER,
        help=gpu_tier_help(),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override separator batch size (default: selected GPU budget profile).",
    )
    parser.add_argument(
        "--separator-rate",
        type=int,
        default=DEFAULT_SEPARATOR_SAMPLE_RATE,
        choices=SEPARATOR_SAMPLE_RATES,
        help=(
            "Rate the separator works at (default: 44100, the model's own). "
            "Lower rates cut the chunk count roughly in proportion and cost "
            "separation quality; neither is recommended. An existing vocal "
            "file is reused as-is, so switching rates means deleting it first."
        ),
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}-vocal.ogg")


#: The two shapes a finished vocal track is delivered in, chosen by the output
#: suffix. They answer different questions and nothing in between is offered:
#:
#: * ``.flac`` -- the separation itself, at the model's own rate and channel
#:   count, no lossy step anywhere. For listening, measuring, and experiments.
#: * ``.ogg``  -- what the ASR chain consumes: 16 kHz mono Vorbis. Every reader
#:   of this file (energy VAD, whisper, the Qwen referee) downmixes and
#:   resamples to exactly that before doing anything, so a 44.1 kHz stereo
#:   delivery spent roughly three quarters of its bitrate on samples the next
#:   stage discards. Encoding at the rate that survives is both smaller and
#:   -- in the band that survives -- cleaner.
LOSSLESS_MODE = "lossless"
ASR_MODE = "asr"
_MODE_BY_SUFFIX = {"flac": LOSSLESS_MODE, "ogg": ASR_MODE}

#: Separation runs unless the caller says the input is already a vocal track.
#: On by default because the pipeline's normal input is a mixed source, and
#: separating one that needed it is a cost while *not* separating one that
#: needed it is a quality collapse that reports no error.
DEFAULT_SEPARATE = True


def resolve_separate(explicit: bool | None = None) -> bool:
    """Three layers, in order: the flag, `[separator] enabled`, the default.

    The same shape as `resolve_vad_silero_assist` and `resolve_split_params`,
    and for the same reason: one function owns the whole chain, so every front
    end lands on the same answer and there is one place to read to find out
    what "unset" means.
    """

    if explicit is not None:
        return bool(explicit)
    configured = app_config.config_bool("separator", "enabled")
    if configured is not None:
        return configured
    return DEFAULT_SEPARATE


#: libsndfile's scale, 0.0 keeps the most and 1.0 the least. The default (0.6)
#: measured 24.0 dB against the lossless separation in the 16 kHz band; 0.2
#: buys 4.6 dB and still halves the file, because the bits now go where the
#: signal survives (docs/separator-optimization.md).
ASR_VORBIS_COMPRESSION = 0.2


def resolve_separator_sample_rate(value: int | None) -> int:
    """Validate a requested working rate, defaulting to the model's own."""

    if value is None:
        return DEFAULT_SEPARATOR_SAMPLE_RATE
    rate = int(value)
    if rate not in SEPARATOR_SAMPLE_RATES:
        allowed = ", ".join(str(item) for item in SEPARATOR_SAMPLE_RATES)
        raise SystemExit(
            f"Unsupported separator sample rate {rate}: choose one of {allowed}."
        )
    return rate


def output_mode_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    try:
        return _MODE_BY_SUFFIX[suffix]
    except KeyError:
        raise SystemExit(
            f"Unsupported vocal output format {suffix!r}: use .flac for the "
            f"lossless separation or .ogg for the 16 kHz mono ASR delivery."
        ) from None


#: What the blocks are merged into. Blocks themselves are no longer written at
#: all -- `demix.separate_waveform` hands each one back in memory -- so the only
#: container left is this one, and it does not follow the delivered format:
#: doing so meant Vorbis-encoding the separated vocals for files that never
#: reach the user.
#:
#: In lossless mode the merge *is* the delivery, so it is FLAC. In ASR mode it
#: is a temporary that gets re-encoded to 16 kHz mono moments later, and FLAC
#: costs about 8x what an uncompressed container does to write (1.82s against
#: 0.23s per 600s block) for a file nobody keeps. RF64 rather than plain WAV
#: because a long source passes WAV's 4 GiB ceiling, which WAV answers by
#: silently truncating.
MERGE_FORMAT = "flac"
ASR_MERGE_FORMAT = "rf64"
#: PCM_16 both ways: the delivery this feeds is 16 kHz Vorbis, and matching the
#: previous FLAC subtype keeps the merged track's precision where it was.
MERGE_SUBTYPE = "PCM_16"


def merge_format_for(output_mode: str) -> str:
    return MERGE_FORMAT if output_mode == LOSSLESS_MODE else ASR_MERGE_FORMAT


def _accel_paths() -> Any:
    try:
        return accel.resolve_accel_paths(MODEL_NAME)
    except Exception:
        return None


_EAGER_ACCEL = accel.AccelerationResult(requested="eager", effective="eager")


def _record_applied_accel(metadata_sink: Any, lease: "_SharedSeparatorLease") -> None:
    """Report the tier that survived setup, not the one that was requested.

    A tier can degrade while being installed -- no compiler, an unloadable
    package -- and metadata naming the intent instead of the outcome hides
    exactly the case worth seeing. Read off the lease rather than the shared
    pool, which the CPU path never goes through.
    """

    if metadata_sink is None:
        return
    applied = lease.accel
    metadata_sink["accel"] = applied.effective
    metadata_sink["accel_requested"] = applied.requested
    if applied.fallback_reason:
        metadata_sink["accel_fallback_reason"] = applied.fallback_reason


def _select_accel_backend(duration_sec: float) -> str:
    """Choose the compiled tier, or eager when anything is unavailable."""

    try:
        return accel.select_backend(_accel_paths(), duration_sec)
    except Exception:
        return "eager"


def place_separator_files() -> None:
    """Put the checkpoint, its config and the model index where load expects.

    Here, in the stage, because the CLI has no prefetch at all: left to
    audio-separator, these three files come straight from
    GitHub -- outside `FINESUB_GITHUB_FILE_PROXY`, outside the region fallback,
    and outside any digest check. `load_model` then finds them already present
    and reaches the network for nothing.

    Best-effort by design: unlisted in the manifest, or a fetch that fails, and
    the library downloads them itself exactly as before.
    """

    try:
        from finesub_bootstrap import model_fetch
        from finesub_bootstrap.download_routes import resolve_region
        from finesub_bootstrap.model_manifest import entry_for
        from finesub_bootstrap.paths import default_data_root

        entry = entry_for("separator")
        if entry is None or not entry.files:
            return
        data_root = default_data_root()
        model_fetch.fetch_fixed_files(
            entry,
            Path(resolve_separator_model_dir()),
            data_root=data_root,
            region=resolve_region(data_root).region,
        )
    except Exception as error:
        current_reporter().debug(
            "separator files not pre-placed; the library will fetch them",
            {"error": f"{type(error).__name__}: {error}"},
        )


def _build_separator(
    output_dir: str, output_format: str, batch_size: int, *, use_cuda: bool
) -> Separator:
    place_separator_files()
    try:
        from audio_separator.separator import Separator
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "audio-separator is required for vocal separation. Install with `pip install -e .`."
        ) from exc

    import logging

    separator = Separator(
        output_dir=output_dir,
        output_format=output_format,
        output_single_stem="Vocals",
        model_file_dir=str(resolve_separator_model_dir()),
        mdxc_params={"batch_size": batch_size},
        mdx_params={"batch_size": batch_size},
        # Told directly, because it configures its logger from here and would
        # otherwise undo a level set before it was built. WARNING rather than
        # silence: its warnings are worth keeping, its version banner and host
        # inventory are not.
        log_level=logging.WARNING if libraries_quieted() else logging.INFO,
    )
    if not use_cuda:
        # audio-separator picks its device inside __init__ off a bare
        # torch.cuda.is_available(), which is True for a card whose kernels this
        # torch build does not ship -- it would then load the weights onto a GPU
        # that cannot run them. It has no device parameter, so steer the choice
        # it already made, before load_model puts weights anywhere. These are
        # its internals: say so loudly if a version bump renames them, because
        # the failure this prevents is a bare CUDA error deep in the stage.
        #
        # `use_cuda` rather than `cuda_usable()` since 2026-09-02: the caller
        # folds the tier's POLICY (`ResourceProfile.gpu`) into the capability,
        # so `--gpu-tier cpu` actually reaches the weights. Asking the
        # capability here would make that tier a label with no effect.
        if hasattr(separator, "torch_device_cpu"):
            separator.torch_device = separator.torch_device_cpu
            separator.onnx_execution_provider = ["CPUExecutionProvider"]
        else:
            current_reporter().warning(
                "separator-device",
                "cannot pin audio-separator to CPU (it no longer exposes "
                "torch_device_cpu); vocal separation may try a GPU this "
                "PyTorch build cannot use.",
            )
    separator.load_model(model_filename=MODEL_NAME)
    if hasattr(separator, "mdx_batch_size"):
        separator.mdx_batch_size = batch_size
    elif hasattr(separator, "mdxc_batch_size"):
        separator.mdxc_batch_size = batch_size
    elif hasattr(separator, "vr_batch_size"):
        separator.vr_batch_size = batch_size
    return separator


def _warm_up_shared_roformer(model_instance: Any, *, use_amp: bool) -> None:
    """Initialize lazy model caches using the requested inference precision.

    Takes the model rather than its Separator wrapper: the AOTI builder warms
    the same live model to capture module inputs, and it only ever holds the
    former.
    """

    if not cuda_usable():
        return
    model = model_instance.model_run
    device = next(model.parameters()).device
    if device.type != "cuda":
        return

    config = model_instance.model_data_cfgdict
    stft_hop_length = getattr(config.model, "stft_hop_length", None)
    if stft_hop_length is None:
        stft_hop_length = config.audio.hop_length
    chunk_size = int(stft_hop_length) * (int(config.inference.dim_t) - 1)
    audio_channels = int(getattr(model, "audio_channels", 2))

    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        enabled=use_amp,
    ):
        output = model(
            torch.zeros(
                1,
                audio_channels,
                chunk_size,
                dtype=torch.float32,
                device=device,
            )
        )
    del output
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()


def _clone_separator_with_shared_model(
    master: Any,
    *,
    output_dir: str,
    output_format: str,
    batch_size: int,
) -> Any:
    """Copy per-call state while retaining the master's model_run object."""

    worker = copy.copy(master)
    worker.model_instance = copy.copy(master.model_instance)
    model_instance = worker.model_instance

    worker.output_dir = output_dir
    worker.output_format = output_format
    worker.output_single_stem = "Vocals"
    model_instance.output_dir = output_dir
    model_instance.output_format = output_format
    model_instance.output_single_stem = "Vocals"
    model_instance.batch_size = batch_size
    model_instance.cached_sources_map = {}
    model_instance.clear_file_specific_paths()
    return worker


class _SharedSeparatorLease:
    def __init__(
        self,
        pool: "_SharedSeparatorPool | None",
        separator: Any,
        *,
        accel: accel.AccelerationResult | None = None,
    ) -> None:
        self._pool = pool
        self.separator: Any | None = separator
        self.accel = accel or _EAGER_ACCEL

    def release(self) -> None:
        if self.separator is None:
            return
        self.separator = None
        if self._pool is not None:
            self._pool.release()


def _evict_separator_weights(master: Any | None) -> None:
    """Drop a retired separator's weights, wherever they live. Never raises.

    **Not `.to("cpu")`.** That was the first attempt and it was wrong: the
    hidden reference that keeps this module alive keeps it alive on the host
    too, so moving the weights turned a 0.6 GiB VRAM leak per call into a
    0.6 GiB RAM leak per call -- measured 2026-09-01 at 0.595 GiB of live CPU
    tensors per run, four runs in a row, RSS climbing with it. Caught in review;
    the CUDA-only regression test could not see it.

    So release the storages instead of relocating them. Whoever still holds the
    module is left with one whose parameters are empty, which is harmless
    because nothing reuses it: the pool has already dropped `_master`, and the
    next `acquire` builds a new one from the checkpoint.

    Best-effort by design: this runs on the teardown path of a stage that has
    already produced its output, so a failure here must not turn a finished
    separation into a failed one.
    """

    if master is None:
        return
    model_instance = getattr(master, "model_instance", None)
    for attribute in ("model_run", "model"):
        module = getattr(model_instance, attribute, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        try:
            with torch.no_grad():
                empty = torch.empty(0)
                for parameter in module.parameters(recurse=True):
                    parameter.data = empty
                    parameter.grad = None
                for name, buffer in list(module.named_buffers(recurse=True)):
                    owner = module
                    *path, leaf = name.split(".")
                    for step in path:
                        owner = getattr(owner, step)
                    if buffer is not None:
                        setattr(owner, leaf, empty)
        except Exception:  # noqa: BLE001 - teardown never fails the stage
            pass


class _SharedSeparatorPool:
    """Share immutable Roformer weights while isolating per-call wrapper state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._master: Any | None = None
        self._active_leases = 0
        self._accel = _EAGER_ACCEL

    def acquire(
        self,
        output_dir: str,
        output_format: str,
        batch_size: int,
        *,
        use_amp: bool,
        accel_backend: str = "eager",
    ) -> _SharedSeparatorLease:
        with self._lock:
            built_master = False
            if self._master is None:
                # The pool is only ever reached on the CUDA path
                # (`_acquire_separator` builds an unshared worker otherwise),
                # so the policy has already been applied by the caller.
                self._master = _build_separator(
                    output_dir,
                    output_format,
                    batch_size,
                    use_cuda=True,
                )
                built_master = True
                try:
                    _warm_up_shared_roformer(
                        self._master.model_instance,
                        use_amp=use_amp,
                    )
                    # Workers are shallow copies sharing model_run, so the
                    # compiled modules go on once, here, and every clone gets
                    # them. Warm-up first: it initialises the rotary cache the
                    # packages expect to be populated.
                    paths = _accel_paths()
                    applied = accel.apply_acceleration(
                        self._master.model_instance,
                        accel_backend,
                        paths,
                    )
                    if applied.effective == "jit":
                        # torch.compile is lazy: the compile itself, and so
                        # the failure seen in the field (a Triton kernel whose
                        # cache entry is missing its .json), happens at the
                        # first forward. Run it here, before any block is in
                        # flight, so a failure degrades this master to eager
                        # in place instead of aborting the separation.
                        try:
                            _warm_up_shared_roformer(
                                self._master.model_instance,
                                use_amp=use_amp,
                            )
                        except Exception as exc:
                            applied = accel.revert_jit(applied, exc, paths)
                            # A failure here is the model's, not the
                            # compiler's: let it propagate like any other.
                            _warm_up_shared_roformer(
                                self._master.model_instance,
                                use_amp=use_amp,
                            )
                    self._accel = applied
                except BaseException:
                    self._master = None
                    gc.collect()
                    if cuda_usable():
                        torch.cuda.empty_cache()
                    raise
            try:
                worker = _clone_separator_with_shared_model(
                    self._master,
                    output_dir=output_dir,
                    output_format=output_format,
                    batch_size=batch_size,
                )
            except BaseException:
                if built_master:
                    self._master = None
                    gc.collect()
                    if cuda_usable():
                        torch.cuda.empty_cache()
                raise
            self._active_leases += 1
            return _SharedSeparatorLease(self, worker, accel=self._accel)

    def release(self) -> None:
        with self._lock:
            if self._active_leases <= 0:
                raise RuntimeError("Shared separator lease released more than once.")
            self._active_leases -= 1
            if self._active_leases > 0:
                return
            master, self._master = self._master, None
            # Dropping the reference is not enough. Measured 2026-09-01: after
            # the last lease released, 676 fp32 weight tensors (0.60 GiB) were
            # still resident -- `_master = None`, zero leases, eager accel,
            # `gc.collect()` and `empty_cache()` all done, and
            # `torch.compiler.reset()` changed nothing. Something downstream of
            # `audio-separator` keeps the module alive, and one whole copy of
            # the Roformer weights stayed on the card for the rest of the
            # process. In a batch that is per *file*: ten files, six GiB, and
            # the ASR stage running with that much less than it thinks it has.
            #
            # So move the weights off the device rather than hoping the
            # reference dies. Whoever still holds the module gets a CPU-resident
            # one; nothing here reuses it, because the next acquire builds a new
            # master from scratch.
            _evict_separator_weights(master)
            # The tier belonged to that master's model_run; the next one is
            # selected again from scratch.
            self._accel = _EAGER_ACCEL
            gc.collect()
            if cuda_usable():
                torch.cuda.empty_cache()


_SHARED_SEPARATOR_POOL = _SharedSeparatorPool()


class _SeparatorBlockSlot:
    def __init__(self, limiter: "_SeparatorBlockLimiter", weight: int) -> None:
        self._limiter = limiter
        self._weight = weight

    def release(self) -> None:
        if self._weight <= 0:
            return
        weight = self._weight
        self._weight = 0
        self._limiter.release(weight)


class _SeparatorBlockLimiter:
    """Globally cap nested batch/file concurrency to the selected profile."""

    _CAPACITY = 12

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._used = 0

    def acquire(self, instances: int) -> _SeparatorBlockSlot:
        count = max(1, min(4, int(instances)))
        weight = self._CAPACITY // count
        with self._condition:
            while self._used + weight > self._CAPACITY:
                self._condition.wait()
            self._used += weight
        return _SeparatorBlockSlot(self, weight)

    def release(self, weight: int) -> None:
        with self._condition:
            if weight <= 0 or weight > self._used:
                raise RuntimeError("Separator block slot released more than once.")
            self._used -= weight
            self._condition.notify_all()


_SEPARATOR_BLOCK_LIMITER = _SeparatorBlockLimiter()


def _acquire_separator(
    output_dir: str,
    output_format: str,
    batch_size: int,
    *,
    use_amp: bool,
    use_cuda: bool,
    accel_backend: str = "eager",
    sample_rate: int = DEFAULT_SEPARATOR_SAMPLE_RATE,
) -> _SharedSeparatorLease:
    # CUDA workers share a model only after the Roformer rotary-position cache
    # has been warmed. Preserve independent instances on other backends instead
    # of sharing that lazily mutated cache without a CUDA synchronization point.
    if not use_cuda:
        lease = _SharedSeparatorLease(
            None,
            _build_separator(
                output_dir, output_format, batch_size, use_cuda=False
            ),
        )
    else:
        lease = _SHARED_SEPARATOR_POOL.acquire(
            output_dir,
            output_format,
            batch_size,
            use_amp=use_amp,
            accel_backend=accel_backend,
        )
    # `separate()` reads this flag per call, and a pooled worker is a shallow copy
    # carrying the master's value. Pin it on the way out so every caller runs at
    # the precision it asked for without repeating this at each acquisition site.
    lease.separator.use_autocast = use_amp
    # Read at call time in all three places that matter -- `librosa.load` on the
    # way in, `sf.write`/`AudioSegment` on the way out -- so pinning it per
    # worker here is enough, and the shared master needs no separate identity.
    # `_clone_separator_with_shared_model` gives each worker its own shallow
    # copy of `model_instance`, so this does not leak across leases.
    lease.separator.sample_rate = sample_rate
    lease.separator.model_instance.sample_rate = sample_rate
    return lease


@dataclass(frozen=True)
class _SeparationBlock:
    index: int
    block_start: int
    read_start: int
    read_end: int


# One extra separator worker allowed per this much audio. Separation runs before
# VAD, so unlike the WT ladder it can only see wall-clock duration, never
# effective speech. Load-bearing since blocks became a multiple of the worker
# count: that removed the implicit gate (a short file used to yield one block,
# which capped workers at one all by itself).
WORKER_DURATION_THRESHOLD_SEC = 300.0


def separator_worker_limit(
    duration_sec: float,
    *,
    threshold_sec: float = WORKER_DURATION_THRESHOLD_SEC,
) -> int:
    if threshold_sec <= 0:
        return 1
    return int(max(0.0, duration_sec) // threshold_sec) + 1


def plan_separation_blocks(
    total_frames: int,
    sample_rate: int,
    *,
    workers: int,
    max_core_seconds: float,
    pad_samples: int,
) -> list[_SeparationBlock]:
    """Cut the timeline into equal blocks, a whole multiple of ``workers``.

    A fixed core length left every worker a different amount of work and a short
    final block; sizing the count to the workers instead gives each of them the
    same number of equal blocks. The cost is that separated audio now depends on
    the worker count -- block edges move, and Roformer is chunk-sensitive even
    with the pad. That is accepted (docs/gpu-profiles.md), so existing
    ``-vocal.ogg`` files are not reproducible and must be deleted to rerun.

    The ladder in ``separator_worker_limit`` doubles as the guard against
    absurdly short cores, so no separate floor is needed: at one round the core
    is ``duration / workers``, which the 300s ladder bounds below by
    ``300k / (k + 1)`` for ``k`` whole thresholds -- smallest at k=1, i.e. 150s.
    Against a 10s pad per side that is 13% redundant compute at worst.
    """

    workers = max(1, int(workers))
    core_limit = max(1, int(round(max_core_seconds * sample_rate)))
    # Smallest whole number of rounds that keeps every core within the limit.
    rounds = max(1, -(-total_frames // (core_limit * workers)))
    block_count = rounds * workers

    edges = [round(index * total_frames / block_count) for index in range(block_count)]
    edges.append(total_frames)

    blocks: list[_SeparationBlock] = []
    for index in range(block_count):
        core_start, core_end = edges[index], edges[index + 1]
        if core_end <= core_start:
            continue
        blocks.append(
            _SeparationBlock(
                index=len(blocks),
                block_start=core_start,
                read_start=max(0, core_start - pad_samples),
                read_end=min(total_frames, core_end + pad_samples),
            )
        )
    return blocks


def _separate_block(
    *,
    input_path: Path,
    tmpdir: str,
    batch_size: int,
    use_amp: bool,
    use_cuda: bool,
    accel_backend: str,
    instances: int,
    sample_rate: int,
    block: _SeparationBlock,
) -> tuple[_SeparationBlock, np.ndarray, int]:
    """Decode one block, separate it, and hand the stem back in memory.

    Nothing touches disk between the source and the merged track: the block used
    to be written as a WAV, re-read by librosa, and written back out through
    ffmpeg only for the next line to decode it again.
    """

    read_frames = max(0, block.read_end - block.read_start)
    waveform, read_sr = load_audio_slice(
        str(input_path),
        block.read_start,
        read_frames,
    )
    if read_sr <= 0:
        raise SystemExit(f"Invalid sample rate while loading: {input_path}")

    slot = _SEPARATOR_BLOCK_LIMITER.acquire(instances)
    lease: _SharedSeparatorLease | None = None
    try:
        lease = _acquire_separator(
            tmpdir,
            MERGE_FORMAT,
            batch_size,
            use_amp=use_amp,
            use_cuda=use_cuda,
            accel_backend=accel_backend,
            sample_rate=sample_rate,
        )
        stem, stem_rate = demix.separate_waveform(
            lease.separator.model_instance,
            waveform,
            read_sr,
            use_autocast=lease.separator.use_autocast,
        )
        return block, stem, stem_rate
    finally:
        if lease is not None:
            lease.release()
        slot.release()


def _append_separated_block(
    *,
    out_file: Optional[sf.SoundFile],
    stem: np.ndarray,
    stem_rate: int,
    output_path: Path,
    merge_format: str,
    block: _SeparationBlock,
    total_frames: int,
    pad_seconds: float,
    chunk_frames: int,
) -> sf.SoundFile:
    trim_left = 0.0 if block.block_start == 0 else pad_seconds
    trim_right = 0.0 if block.read_end >= total_frames else pad_seconds

    channels, block_frames = stem.shape
    if out_file is None:
        out_file = sf.SoundFile(
            str(output_path),
            mode="w",
            samplerate=stem_rate,
            channels=channels,
            format=merge_format.upper(),
            subtype=MERGE_SUBTYPE,
        )
    elif stem_rate != out_file.samplerate or channels != out_file.channels:
        raise SystemExit("Block output format mismatch.")

    start_frame = int(round(trim_left * stem_rate))
    end_frame = max(block_frames - int(round(trim_right * stem_rate)), start_frame)
    # Written in slices rather than as one `stem.T`: transposing the whole block
    # would materialise a second copy of several hundred megabytes.
    for offset in range(start_frame, end_frame, chunk_frames):
        stop = min(offset + chunk_frames, end_frame)
        out_file.write(stem[:, offset:stop].T)
    return out_file


#: One resample window, counted in whole ratio steps (see `_rational_ratio`):
#: 2000 steps is 20s at 44.1 kHz, ~7 MiB of float32 stereo.
_RESAMPLE_WINDOW_STEPS = 2000
#: Real audio carried on each side of a window and then discarded. 20 steps is
#: 0.2s -- orders of magnitude past the resampling filter's support, which is
#: what makes the kept samples indistinguishable from a whole-file resample.
_RESAMPLE_CONTEXT_STEPS = 20


def _rational_ratio(src_sr: int, dst_sr: int) -> tuple[int, int]:
    """Smallest whole (source, target) frame counts that map onto each other.

    44100 -> 16000 gives (441, 160): every 441 source frames are exactly 160
    target frames, so a window sized in multiples of 441 can never leave a
    fractional frame behind for the next window to inherit.
    """

    divisor = gcd(src_sr, dst_sr)
    return src_sr // divisor, dst_sr // divisor


def stream_asr_frames(merged_path: Path) -> Iterator[np.ndarray]:
    """Yield the merged track downmixed and resampled to `ASR_TARGET_SR`.

    Windowed, because a feature-length track does not fit in memory -- and
    windowed resampling is exactly where seams come from: a filter whose
    support runs off the end of its input invents the edge, once per window.
    Two properties remove that here.

    Windows are whole multiples of the ratio's source step, so each maps to a
    whole number of output frames and no rounding drift accumulates over hours.
    And every window is resampled with real audio on both sides, which is then
    dropped -- so each sample kept was computed from a fully populated filter,
    which is what resampling the whole file at once would have produced. A test
    pins the two against each other rather than trusting that argument.
    """

    info = sf.info(str(merged_path))
    src_sr, total_frames = int(info.samplerate), int(info.frames)
    step_src, step_dst = _rational_ratio(src_sr, ASR_TARGET_SR)
    window = _RESAMPLE_WINDOW_STEPS * step_src
    context = _RESAMPLE_CONTEXT_STEPS * step_src
    position = 0
    while position < total_frames:
        core = min(window, total_frames - position)
        left = min(context, position)
        right = min(context, total_frames - position - core)
        waveform, read_sr = load_audio_slice(
            str(merged_path), position - left, left + core + right
        )
        if int(read_sr) != src_sr:
            raise SystemExit(f"Sample rate changed mid-file: {merged_path}")
        resampled, _ = resample_if_needed(
            to_mono(waveform).unsqueeze(0), src_sr, ASR_TARGET_SR
        )
        frames = resampled.squeeze(0)
        # `left` is a whole number of steps, so its share of the output is
        # exact. The tail window is the only one whose core is not, and rounding
        # it up is what a whole-file resample does with the same remainder.
        start = left // step_src * step_dst
        keep = min(-(-core * step_dst // step_src), int(frames.numel()) - start)
        yield as_numpy_float32(frames[start : start + keep])
        position += core


#: Frames per `write()` into the Vorbis delivery. libsndfile's Vorbis writer
#: dies on a single write past roughly half a million frames -- **process-level,
#: no exception**, leaving a header-only file behind. That is not hypothetical:
#: one resample window is `_RESAMPLE_WINDOW_STEPS` source steps, so 44.1 kHz
#: yields 320000 output frames per window and survives while 22.05 kHz yields
#: 640000 and does not. The window size is a tunable and the source rate is now
#: a switch, so neither may be relied on to stay under the limit.
_OGG_WRITE_FRAMES = 262_144


def _encode_asr_delivery(merged_path: Path, output_path: Path) -> None:
    """Write the 16 kHz mono Vorbis delivery from the merged lossless track."""

    with sf.SoundFile(
        str(output_path),
        mode="w",
        samplerate=ASR_TARGET_SR,
        channels=1,
        format="OGG",
        compression_level=ASR_VORBIS_COMPRESSION,
    ) as out_file:
        for frames in stream_asr_frames(merged_path):
            for start in range(0, len(frames), _OGG_WRITE_FRAMES):
                out_file.write(frames[start : start + _OGG_WRITE_FRAMES])


def encode_asr_delivery(
    source_path: str | Path,
    output_path: str | Path,
    *,
    run_metadata_path: str | Path | None = None,
) -> Path:
    """Produce the ASR delivery from a source that needs no separation.

    `--no-separate` says the input already *is* a clean vocal track. What it
    skips is the separation, not the stage's contract: the same 16 kHz mono ogg
    lands at the same path, so the existence-based skip, resume, and every
    reader of `-vocal.ogg` need no case for "there is no vocal file". Handing
    the source file over directly would have made all of them care.

    A compressed or video source is decoded to a temporary flac first, and it is
    removed **on success only** -- `ensure_decodable_input`'s contract, and what
    `run_vocal_separation` does with its own. A failed run keeps it so a rerun
    skips the decode, which is why `run_metadata_path` matters: that record is
    the only thing that can name the file cleanup has to remove afterwards.
    """

    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    readable, temporary_input = ensure_decodable_input(source_path, output_path.parent)
    if temporary_input is not None and run_metadata_path is not None:
        # Written before the encode, not after: the copy this record exists to
        # clean up after is the one a *failed* run leaves behind.
        record_scratch_file(run_metadata_path, temporary_input)
    current_reporter().debug(
        "encoding ASR vocal delivery without separation",
        {"target_sr": ASR_TARGET_SR, "compression_level": ASR_VORBIS_COMPRESSION},
    )
    _encode_asr_delivery(readable, output_path)
    if temporary_input is not None:
        # Deliberately not in a `finally`: an hour of video decodes once, and a
        # run that died after paying for it should not make the retry pay again.
        temporary_input.unlink(missing_ok=True)
    # Returned like `run_vocal_separation`'s: `_use_or_create` publishes what
    # its `create` hands back, so the two producers of this artifact answer in
    # the same shape.
    return output_path


def _finish_delivery(merged_path: Path, output_path: Path, output_mode: str) -> None:
    """Turn the merged lossless track into whatever the caller asked for.

    In lossless mode the merge already wrote the delivery in place, so there is
    nothing left to do.
    """

    if output_mode == LOSSLESS_MODE:
        return
    current_reporter().debug(
        "encoding ASR vocal delivery",
        {"target_sr": ASR_TARGET_SR, "compression_level": ASR_VORBIS_COMPRESSION},
    )
    _encode_asr_delivery(merged_path, output_path)
    merged_path.unlink(missing_ok=True)


def run_vocal_separation(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    block_seconds: float = 600.0,
    pad_seconds: float = 10.0,
    # `auto` = ask the card. A tier names what CLASS of card this is, not a
    # cap on what the pipeline may use; `resolve_gpu_tier` owns the answer.
    gpu_tier: str = AUTO_GPU_TIER,
    # The user's request (`--device`). `None` is "not chosen" and means the
    # code default, cuda; an explicit `cpu` keeps this stage off the card
    # whatever the tier or the hardware say -- the pipeline promises that
    # `--device cpu` leaves the GPU alone for the WHOLE run, and separation
    # is its first and heaviest stage (review 2026-09-02).
    device: str | None = None,
    batch_size: Optional[int] = None,
    use_amp: bool = True,
    separator_sample_rate: int | None = None,
    metadata_sink: dict[str, Any] | None = None,
    run_metadata_path: str | Path | None = None,
) -> Path:
    sample_rate = resolve_separator_sample_rate(separator_sample_rate)
    resource_profile = get_resource_profile(gpu_tier)
    selected_batch_size = (
        resource_profile.vocal_separation_batch_size
        if batch_size is None
        else int(batch_size)
    )
    if selected_batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    # Intent, policy AND capability, folded once here so every helper below is
    # handed a decision rather than re-deriving part of it. The user may ask
    # for the CPU (`--device cpu`); the tier may forbid the GPU
    # (`--gpu-tier cpu`); `cuda_usable()` answers whether torch could have
    # used it anyway. Any one saying no lands the whole stage on the CPU --
    # and unlike the ASR stage, that is the right granularity here: one model,
    # one device, and the worker count follows from it.
    wants_cuda = str(device or "cuda").strip().lower().startswith("cuda")
    use_cuda = wants_cuda and bool(resource_profile.gpu) and cuda_usable()
    device_for_usage: Optional[str] = "cuda" if use_cuda else None
    # After the device is settled, so a CPU run stays silent about VRAM. Same
    # rule and same place whether the tier came from `auto` or by name.
    warn_if_vram_is_short(
        resource_profile, stage="vocal separation", device=device_for_usage or "cpu"
    )
    # Autocast only exists on the CUDA path; the CPU fallback always runs FP32.
    amp_enabled = bool(use_amp and device_for_usage is not None)
    out_file: Optional[sf.SoundFile] = None
    merge_dir: tempfile.TemporaryDirectory | None = None
    separator = None
    separator_lease: _SharedSeparatorLease | None = None
    gpu_stage_lease: GpuStageLease | None = None
    temporary_input: Optional[Path] = None
    separation_completed = False
    watchdog = stall_watchdog.arm("vocal-separation")
    reset_peak_gpu_memory_stats_for_run(device_for_usage)
    memory_sampler = start_stage_memory_sampling()
    # Third-party bars and logging are quieted for the whole run by the front
    # end (`reporting.quieted_libraries`), not per stage: the separator was
    # never the only library that narrates itself -- transformers draws a
    # weight-loading bar during verification, and the separator's own logger
    # produced more lines than the pipeline did.
    try:
        if device_for_usage is None and wants_cuda and resource_profile.gpu:
            # A FALLBACK only: an explicit `--device cpu` and a `cpu` tier are
            # choices, and warning "CUDA is unavailable" about a choice would
            # send the user to check a driver that is fine.
            # Name the actual reason. Separation is the first and by far the
            # slowest stage on CPU, so "CUDA is unavailable" against a card that
            # is merely too old sends the user off checking their driver for the
            # ten minutes before VAD-ASR prints the truth. Asked for only on this
            # branch so that cuda_usable() stays the module's single seam for
            # *whether* the GPU is in play -- a second branching predicate here
            # is what let a monkeypatched test pass on a GPU box and fail on CI.
            reason = cuda_unusable_reason() or "it is unavailable"
            current_reporter().warning(
                "cpu-fallback",
                f"CUDA is the default vocal separation device but {reason}; "
                f"falling back to CPU.",
                impact="速度会显著下降",
            )
        input_path = Path(input_path).expanduser().resolve()
        if not input_path.exists():
            raise SystemExit(f"Input not found: {input_path}")

        output_path = (
            Path(output_path).expanduser().resolve()
            if output_path
            else default_output_path(input_path)
        )
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".ogg")
        output_mode = output_mode_for(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_path, temporary_input = ensure_decodable_input(
            input_path,
            output_path.parent,
        )
        if temporary_input is not None and run_metadata_path is not None:
            # Written now, not at the end: the copy this cleans up after is the
            # one a *failed* run leaves behind, and a stage that dies never
            # reaches its own tidying.
            record_scratch_file(run_metadata_path, temporary_input)

        # In ASR mode the merged lossless track is an intermediate: blocks are
        # merged into it, then it is downmixed, resampled and encoded once. In
        # lossless mode the merge *is* the delivery and is written in place.
        merge_format = merge_format_for(output_mode)
        if output_mode == ASR_MODE:
            merge_dir = tempfile.TemporaryDirectory(prefix="vocal_merge_")
            merge_path = Path(merge_dir.name) / f"{output_path.stem.lstrip('.')}.wav"
        else:
            merge_path = output_path
        src_sr, total_frames = get_audio_info(str(input_path))
        if src_sr <= 0 or total_frames <= 0:
            raise SystemExit(f"Unable to read audio info for {input_path}")
        duration_sec = total_frames / float(src_sr)
        accel_backend = _select_accel_backend(duration_sec)
        gpu_stage_lease = GPU_STAGE_GATE.acquire(
            "separator",
            enabled=device_for_usage is not None,
        )
        separator_instances = (
            resource_profile.vocal_separator_instances
            if device_for_usage is not None
            else 1
        )
        if metadata_sink is not None:
            metadata_sink.update(
                {
                    "profile_limit": resource_profile.vocal_separator_instances,
                    "effective": 1,
                    "device": "cuda" if device_for_usage is not None else "cpu",
                    "amp": amp_enabled,
                    "accel": "pending",
                    "sample_rate": sample_rate,
                }
            )
        if block_seconds <= 0:
            slot = (
                _SEPARATOR_BLOCK_LIMITER.acquire(separator_instances)
                if device_for_usage is not None
                else None
            )
            try:
                separator_lease = _acquire_separator(
                    str(merge_path.parent),
                    MERGE_FORMAT,
                    selected_batch_size,
                    use_amp=amp_enabled,
                    use_cuda=use_cuda,
                    accel_backend=accel_backend,
                    sample_rate=sample_rate,
                )
                separator = separator_lease.separator
                _record_applied_accel(metadata_sink, separator_lease)

                waveform, read_sr = load_audio_slice(str(input_path), 0, total_frames)
                if read_sr <= 0:
                    raise SystemExit(
                        f"Invalid sample rate while loading: {input_path}"
                    )
                stem, stem_rate = demix.separate_waveform(
                    separator.model_instance,
                    waveform,
                    read_sr,
                    use_autocast=separator.use_autocast,
                )
                del waveform
                sf.write(
                    str(merge_path),
                    stem.T,
                    stem_rate,
                    format=merge_format.upper(),
                    subtype=MERGE_SUBTYPE,
                )
                del stem
            finally:
                if slot is not None:
                    slot.release()
            _finish_delivery(merge_path, output_path, output_mode)
            separation_completed = True
            return output_path

        pad_samples = int(round(pad_seconds * src_sr))

        # Plan workers before blocks: the block count is a multiple of the
        # worker count, so the duration ladder is the only thing keeping a short
        # file off the profile's full width.
        duration_limit = separator_worker_limit(duration_sec)
        separator_instances = min(separator_instances, duration_limit)
        if metadata_sink is not None:
            metadata_sink["duration_limit"] = duration_limit

        chunk_frames = 262144
        blocks = plan_separation_blocks(
            total_frames,
            src_sr,
            workers=separator_instances,
            max_core_seconds=block_seconds,
            pad_samples=pad_samples,
        )

        # Empty for the whole run: blocks are handed over in memory now. It
        # exists because `Separator.__init__` insists on an output directory and
        # creates it, and pointing that at a real directory would leave the
        # separator holding a path it might yet write to. Keep it a temp.
        with tempfile.TemporaryDirectory(prefix="vocal_separator_") as tmpdir:
            separator_lease = _acquire_separator(
                tmpdir,
                MERGE_FORMAT,
                selected_batch_size,
                use_amp=amp_enabled,
                use_cuda=use_cuda,
                accel_backend=accel_backend,
                sample_rate=sample_rate,
            )
            separator = separator_lease.separator
            _record_applied_accel(metadata_sink, separator_lease)

            if separator_instances > 1 and len(blocks) > 1:
                max_workers = min(separator_instances, len(blocks))
                if metadata_sink is not None:
                    metadata_sink["effective"] = max_workers
                progress = _BlockProgress(len(blocks), workers=max_workers)
                progress.report()
                # Equal blocks removed the straggler that the 2x look-ahead
                # existed to hide, so one spare block is enough to keep every
                # worker fed while holding fewer decoded blocks in RAM.
                max_pending = max_workers + 1
                pending: dict[
                    int,
                    cf.Future[tuple[_SeparationBlock, np.ndarray, int]],
                ] = {}
                next_submit = 0
                reporter = current_reporter()
                executor = cf.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="vocal-block",
                    # The reporter is thread-local and worker threads start
                    # without one; bind it here so a block can report from
                    # where it actually runs.
                    initializer=bind_reporter,
                    initargs=(reporter,),
                )
                try:
                    for expected in blocks:
                        while (
                            next_submit < len(blocks)
                            and len(pending) < max_pending
                        ):
                            block = blocks[next_submit]
                            future = executor.submit(
                                _separate_block,
                                input_path=input_path,
                                tmpdir=tmpdir,
                                batch_size=selected_batch_size,
                                use_amp=amp_enabled,
                                use_cuda=use_cuda,
                                accel_backend=accel_backend,
                                instances=separator_instances,
                                sample_rate=sample_rate,
                                block=block,
                            )
                            future.add_done_callback(progress.block_finished)
                            pending[block.index] = future
                            next_submit += 1
                        actual, stem, stem_rate = pending.pop(expected.index).result()
                        if actual.index != expected.index:
                            raise RuntimeError(
                                "Separator block scheduler returned out-of-order metadata."
                            )
                        out_file = _append_separated_block(
                            out_file=out_file,
                            stem=stem,
                            stem_rate=stem_rate,
                            output_path=merge_path,
                            merge_format=merge_format,
                            block=actual,
                            total_frames=total_frames,
                            pad_seconds=pad_seconds,
                            chunk_frames=chunk_frames,
                        )
                        del stem
                finally:
                    for future in pending.values():
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
            else:
                progress = _BlockProgress(len(blocks), workers=1)
                progress.report()
                for block in blocks:
                    read_frames = max(0, block.read_end - block.read_start)
                    waveform, read_sr = load_audio_slice(
                        str(input_path),
                        block.read_start,
                        read_frames,
                    )
                    if read_sr <= 0:
                        raise SystemExit(
                            f"Invalid sample rate while loading: {input_path}"
                        )
                    slot = (
                        _SEPARATOR_BLOCK_LIMITER.acquire(separator_instances)
                        if device_for_usage is not None
                        else None
                    )
                    try:
                        stem, stem_rate = demix.separate_waveform(
                            separator.model_instance,
                            waveform,
                            read_sr,
                            use_autocast=separator.use_autocast,
                        )
                    finally:
                        if slot is not None:
                            slot.release()
                    del waveform
                    out_file = _append_separated_block(
                        out_file=out_file,
                        stem=stem,
                        stem_rate=stem_rate,
                        output_path=merge_path,
                        merge_format=merge_format,
                        block=block,
                        total_frames=total_frames,
                        pad_seconds=pad_seconds,
                        chunk_frames=chunk_frames,
                    )
                    del stem
                    gc.collect()
                    progress.block_finished()

        if out_file is None:
            raise SystemExit("Vocal separation produced no audio to merge.")
        out_file.close()
        out_file = None
        _finish_delivery(merge_path, output_path, output_mode)
        separation_completed = True
        return output_path
    finally:
        # Only on success: a failed run keeps it so a rerun skips the decode.
        if separation_completed and temporary_input is not None:
            try:
                temporary_input.unlink(missing_ok=True)
            except Exception:
                pass
        if out_file is not None:
            try:
                out_file.close()
            except Exception:
                pass
        if merge_dir is not None:
            try:
                merge_dir.cleanup()
            except Exception:
                pass
        # The final active lease owns the shared model lifetime. Once it exits,
        # release the weights before the same pipeline worker loads Whisper.
        separator = None
        if separator_lease is not None:
            separator_lease.release()
            separator_lease = None
        gc.collect()
        if device_for_usage is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if gpu_stage_lease is not None:
            gpu_stage_lease.release()
            gpu_stage_lease = None
        print_peak_resource_usage(
            device_for_usage, resource_profile, sampler=memory_sampler
        )
        watchdog.disarm()


def main() -> int:
    args = parse_args()
    reporter = terminal_reporter()
    try:
        # Without a bound renderer this command's own CPU-fallback warning
        # reaches a reporter that shows nothing -- and a run silently on CPU
        # is the thing that warning exists to prevent.
        with reporting_to(reporter), quieted_libraries(reporter.level):
            output = run_vocal_separation(
                args.input,
                output_path=args.output,
                block_seconds=args.block_seconds,
                pad_seconds=args.pad_seconds,
                gpu_tier=args.gpu_tier,
                batch_size=args.batch_size,
                separator_sample_rate=args.separator_rate,
            )
        print(output)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
