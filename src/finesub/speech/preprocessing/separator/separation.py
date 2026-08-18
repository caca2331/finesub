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
    DEFAULT_GPU_BUDGET_GB,
    get_resource_profile,
    gpu_budget_choices,
)
from . import accel
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
BATCH_SIZE = get_resource_profile(DEFAULT_GPU_BUDGET_GB).vocal_separation_batch_size


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
        "--gpu-budget-gb",
        type=int,
        choices=gpu_budget_choices(),
        default=DEFAULT_GPU_BUDGET_GB,
        help="GPU memory budget profile in GiB (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override separator batch size (default: selected GPU budget profile).",
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

#: libsndfile's scale, 0.0 keeps the most and 1.0 the least. The default (0.6)
#: measured 24.0 dB against the lossless separation in the 16 kHz band; 0.2
#: buys 4.6 dB and still halves the file, because the bits now go where the
#: signal survives (docs/separator-optimization.md).
ASR_VORBIS_COMPRESSION = 0.2


def output_mode_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    try:
        return _MODE_BY_SUFFIX[suffix]
    except KeyError:
        raise SystemExit(
            f"Unsupported vocal output format {suffix!r}: use .flac for the "
            f"lossless separation or .ogg for the 16 kHz mono ASR delivery."
        ) from None


#: What a block is written as, and what the blocks are merged into. A block is
#: a temporary file read straight back into the merged track and deleted
#: moments later, and in ASR mode the merged track is itself temporary. Neither
#: follows the delivered format: doing so meant Vorbis-encoding the separated
#: vocals twice over -- once per block, then again on append -- for files that
#: never reach the user. FLAC lands at PCM_16, the precision the PCM_16 block
#: input already carries, so the round trip adds nothing.
MERGE_FORMAT = "flac"


def _accel_paths() -> Any:
    try:
        return accel.resolve_accel_paths(MODEL_NAME)
    except Exception:
        return None


def _record_applied_accel(metadata_sink: Any, lease: "_SharedSeparatorLease") -> None:
    """Report the tier that survived setup, not the one that was requested.

    A tier can degrade while being installed -- no compiler, an unloadable
    package -- and metadata naming the intent instead of the outcome hides
    exactly the case worth seeing. Read off the lease rather than the shared
    pool, which the CPU path never goes through.
    """

    if metadata_sink is None:
        return
    metadata_sink["accel"] = lease.accel_backend


def _select_accel_backend(duration_sec: float) -> str:
    """Choose the compiled tier, or eager when anything is unavailable."""

    try:
        return accel.select_backend(_accel_paths(), duration_sec)
    except Exception:
        return "eager"


def place_separator_files() -> None:
    """Put the checkpoint, its config and the model index where load expects.

    Here rather than in the desktop's prefetch, because the CLI has no prefetch
    at all: left to audio-separator, these three files come straight from
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


def _build_separator(output_dir: str, output_format: str, batch_size: int) -> Separator:
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
    if not cuda_usable():
        # audio-separator picks its device inside __init__ off a bare
        # torch.cuda.is_available(), which is True for a card whose kernels this
        # torch build does not ship -- it would then load the weights onto a GPU
        # that cannot run them. It has no device parameter, so steer the choice
        # it already made, before load_model puts weights anywhere. These are
        # its internals: say so loudly if a version bump renames them, because
        # the failure this prevents is a bare CUDA error deep in the stage.
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
        accel_backend: str = "eager",
    ) -> None:
        self._pool = pool
        self.separator: Any | None = separator
        self.accel_backend = accel_backend

    def release(self) -> None:
        if self.separator is None:
            return
        self.separator = None
        if self._pool is not None:
            self._pool.release()


class _SharedSeparatorPool:
    """Share immutable Roformer weights while isolating per-call wrapper state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._master: Any | None = None
        self._active_leases = 0
        self._accel_backend = "eager"

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
                self._master = _build_separator(
                    output_dir,
                    output_format,
                    batch_size,
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
                    self._accel_backend = accel.apply_acceleration(
                        self._master.model_instance,
                        accel_backend,
                        _accel_paths(),
                    )
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
            return _SharedSeparatorLease(
                self,
                worker,
                accel_backend=self._accel_backend,
            )

    def release(self) -> None:
        with self._lock:
            if self._active_leases <= 0:
                raise RuntimeError("Shared separator lease released more than once.")
            self._active_leases -= 1
            if self._active_leases > 0:
                return
            self._master = None
            # The tier belonged to that master's model_run; the next one is
            # selected again from scratch.
            self._accel_backend = "eager"
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
    accel_backend: str = "eager",
) -> _SharedSeparatorLease:
    # CUDA workers share a model only after the Roformer rotary-position cache
    # has been warmed. Preserve independent instances on other backends instead
    # of sharing that lazily mutated cache without a CUDA synchronization point.
    if not cuda_usable():
        lease = _SharedSeparatorLease(
            None,
            _build_separator(output_dir, output_format, batch_size),
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
    return lease


def _collect_output_paths(output_files) -> list[Path]:
    paths: list[Path] = []

    def add(item) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple, set)):
            for sub in item:
                add(sub)
            return
        if isinstance(item, dict):
            for sub in item.values():
                add(sub)
            return
        paths.append(Path(str(item)))

    add(output_files)
    return paths


def _block_output_stem(output_path: Path, block_index: int) -> str:
    """Build a per-block stem that audio-separator will keep on disk.

    Pipeline atomic temps look like ``.name.part.ogg`` (leading dot). Some
    writers strip that dot from the emitted filename, so keep the stem free of
    a leading ``.`` and match files against the sanitized form.
    """

    base = output_path.stem.lstrip(".") or output_path.stem
    return f"{base}-block{block_index:05d}"


def _resolve_separator_path(item: Path, output_dir: Path) -> Path:
    if item.is_absolute():
        return item
    return output_dir / item


def _find_output_file(output_files, stem: str, output_dir: Path) -> Path:
    stem = stem.lstrip(".")
    candidates = [
        _resolve_separator_path(item, output_dir)
        for item in _collect_output_paths(output_files)
    ]
    for item in candidates:
        if stem in item.name and item.exists():
            return item
    for item in candidates:
        if item.exists():
            return item
    for item in output_dir.glob(f"*{stem}*"):
        if item.is_file():
            return item
    raise RuntimeError("No output files were produced by audio-separator.")


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


def _process_parallel_block(
    *,
    input_path: Path,
    tmpdir: str,
    output_path: Path,
    batch_size: int,
    use_amp: bool,
    accel_backend: str,
    instances: int,
    block: _SeparationBlock,
) -> tuple[_SeparationBlock, Path]:
    read_frames = max(0, block.read_end - block.read_start)
    waveform, read_sr = load_audio_slice(
        str(input_path),
        block.read_start,
        read_frames,
    )
    if read_sr <= 0:
        raise SystemExit(f"Invalid sample rate while loading: {input_path}")
    block_input = Path(tmpdir) / f"block_{block.index:05d}.wav"
    sf.write(
        str(block_input),
        waveform.detach().cpu().numpy().T,
        read_sr,
        subtype="PCM_16",
    )
    del waveform

    output_stem = _block_output_stem(output_path, block.index)
    output_names = {"Vocals": output_stem}
    slot = _SEPARATOR_BLOCK_LIMITER.acquire(instances)
    lease: _SharedSeparatorLease | None = None
    try:
        lease = _acquire_separator(
            tmpdir,
            MERGE_FORMAT,
            batch_size,
            use_amp=use_amp,
            accel_backend=accel_backend,
        )
        separator = lease.separator
        output_files = separator.separate(str(block_input), output_names)
        if not output_files:
            separator = None
            lease.release()
            lease = _acquire_separator(
                tmpdir,
                MERGE_FORMAT,
                batch_size,
                use_amp=use_amp,
                accel_backend=accel_backend,
            )
            separator = lease.separator
            output_files = separator.separate(str(block_input), output_names)
        block_output = _find_output_file(output_files, output_stem, Path(tmpdir))
        if not block_output.exists():
            raise SystemExit("No output files were produced by audio-separator.")
        return block, block_output
    finally:
        if lease is not None:
            lease.release()
        slot.release()
        try:
            block_input.unlink(missing_ok=True)
        except Exception:
            pass


def _append_separated_block(
    *,
    out_file: Optional[sf.SoundFile],
    block_output: Path,
    output_path: Path,
    block: _SeparationBlock,
    total_frames: int,
    pad_seconds: float,
    chunk_frames: int,
) -> sf.SoundFile:
    trim_left = 0.0 if block.block_start == 0 else pad_seconds
    trim_right = 0.0 if block.read_end >= total_frames else pad_seconds

    with sf.SoundFile(block_output, mode="r") as in_f:
        if out_file is None:
            out_file = sf.SoundFile(
                str(output_path),
                mode="w",
                samplerate=in_f.samplerate,
                channels=in_f.channels,
                format=MERGE_FORMAT.upper(),
            )
        elif (
            in_f.samplerate != out_file.samplerate
            or in_f.channels != out_file.channels
        ):
            raise SystemExit("Block output format mismatch.")

        total_out_frames = len(in_f)
        start_frame = int(round(trim_left * in_f.samplerate))
        end_frame = total_out_frames - int(round(trim_right * in_f.samplerate))
        end_frame = max(end_frame, start_frame)
        in_f.seek(start_frame)
        remaining = end_frame - start_frame
        while remaining > 0:
            frames = in_f.read(
                min(remaining, chunk_frames),
                dtype="float32",
                always_2d=True,
            )
            if frames.size == 0:
                break
            out_file.write(frames)
            remaining -= frames.shape[0]
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
            out_file.write(frames)


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
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB,
    batch_size: Optional[int] = None,
    use_amp: bool = True,
    metadata_sink: dict[str, Any] | None = None,
    run_metadata_path: str | Path | None = None,
) -> Path:
    resource_profile = get_resource_profile(gpu_budget_gb)
    selected_batch_size = (
        resource_profile.vocal_separation_batch_size
        if batch_size is None
        else int(batch_size)
    )
    if selected_batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    device_for_usage: Optional[str] = "cuda" if cuda_usable() else None
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
        if device_for_usage is None:
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
        if output_mode == ASR_MODE:
            merge_dir = tempfile.TemporaryDirectory(prefix="vocal_merge_")
            merge_path = Path(merge_dir.name) / f"{output_path.stem.lstrip('.')}.flac"
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
                    accel_backend=accel_backend,
                )
                separator = separator_lease.separator
                _record_applied_accel(metadata_sink, separator_lease)

                output_names = {"Vocals": merge_path.stem}
                output_files = separator.separate(str(input_path), output_names)
                if not output_files:
                    raise SystemExit(
                        "No output files were produced by audio-separator."
                    )
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

        with tempfile.TemporaryDirectory(prefix="vocal_blocks_") as tmpdir:
            separator_lease = _acquire_separator(
                tmpdir,
                MERGE_FORMAT,
                selected_batch_size,
                use_amp=amp_enabled,
                accel_backend=accel_backend,
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
                    cf.Future[tuple[_SeparationBlock, Path]],
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
                                _process_parallel_block,
                                input_path=input_path,
                                tmpdir=tmpdir,
                                output_path=merge_path,
                                batch_size=selected_batch_size,
                                use_amp=amp_enabled,
                                accel_backend=accel_backend,
                                instances=separator_instances,
                                block=block,
                            )
                            future.add_done_callback(progress.block_finished)
                            pending[block.index] = future
                            next_submit += 1
                        actual, block_output = pending.pop(expected.index).result()
                        if actual.index != expected.index:
                            raise RuntimeError(
                                "Separator block scheduler returned out-of-order metadata."
                            )
                        out_file = _append_separated_block(
                            out_file=out_file,
                            block_output=block_output,
                            output_path=merge_path,
                            block=actual,
                            total_frames=total_frames,
                            pad_seconds=pad_seconds,
                            chunk_frames=chunk_frames,
                        )
                        block_output.unlink(missing_ok=True)
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
                    block_input = Path(tmpdir) / f"block_{block.index:05d}.wav"
                    sf.write(
                        str(block_input),
                        waveform.detach().cpu().numpy().T,
                        read_sr,
                        subtype="PCM_16",
                    )

                    output_stem = _block_output_stem(merge_path, block.index)
                    output_names = {"Vocals": output_stem}
                    slot = (
                        _SEPARATOR_BLOCK_LIMITER.acquire(separator_instances)
                        if device_for_usage is not None
                        else None
                    )
                    try:
                        output_files = separator.separate(
                            str(block_input),
                            output_names,
                        )
                        if not output_files:
                            separator = None
                            separator_lease.release()
                            separator_lease = None
                            gc.collect()
                            if device_for_usage is not None:
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            separator_lease = _acquire_separator(
                                tmpdir,
                                MERGE_FORMAT,
                                selected_batch_size,
                                use_amp=amp_enabled,
                                accel_backend=accel_backend,
                            )
                            separator = separator_lease.separator
                            output_files = separator.separate(
                                str(block_input),
                                output_names,
                            )
                    finally:
                        if slot is not None:
                            slot.release()
                    block_output = _find_output_file(
                        output_files,
                        output_stem,
                        Path(tmpdir),
                    )
                    if not block_output.exists():
                        raise SystemExit(
                            "No output files were produced by audio-separator."
                        )
                    out_file = _append_separated_block(
                        out_file=out_file,
                        block_output=block_output,
                        output_path=merge_path,
                        block=block,
                        total_frames=total_frames,
                        pad_seconds=pad_seconds,
                        chunk_frames=chunk_frames,
                    )
                    block_input.unlink(missing_ok=True)
                    block_output.unlink(missing_ok=True)
                    del waveform
                    gc.collect()
                    progress.block_finished()

        if out_file is None:
            raise SystemExit("No output files were produced by audio-separator.")
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
                gpu_budget_gb=args.gpu_budget_gb,
                batch_size=args.batch_size,
            )
        print(output)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
