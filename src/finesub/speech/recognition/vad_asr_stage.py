"""VAD-energy + Whisper alignment stage for vocal audio."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import gc
import json
import os
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import align_sentinel
from . import asr_backend as asr_backends
from . import transcribe as asr_align
from . import checkpoint as checkpoint_store
from . import segments as segment_ops
from . import word_starts
from ..postprocessing import segmentation as segment_split
from ..runtime.resources import (
    get_resource_profile,
    AUTO_GPU_TIER,
    gpu_tier_cli_choices,
    warn_if_vram_is_short,
    gpu_tier_help,
    resolve_asr_decode_batch,
)
from ..runtime.gpu_stage_gate import GPU_STAGE_GATE, GpuStageLease
from ..runtime.device import resolve_asr_device
from ..runtime import hf_weights, phase_timing
from ..runtime import stall_watchdog
from ..runtime.thread_budget import bounded_intra_op_threads
from ..preprocessing import energy as vad_energy
from ..preprocessing import vad as vad_detection
from ..preprocessing.audio import ensure_decodable_input
from ...run_metadata import record_scratch_file, update_run_metadata
from ... import config as app_config
from ...reporting import (
    bind_reporter,
    current_reporter,
    reporting_to,
    terminal_reporter,
)
from ...subtitles import time_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAD-energy + Whisper alignment.")
    parser.add_argument("input", help="Path to vocal audio.")
    parser.add_argument("--output", help="Path to output JSON.")
    parser.add_argument("--model", default=asr_align.DEFAULT_MODEL, help="Whisper model name.")
    parser.add_argument(
        "--asr-backend",
        choices=asr_backends.BACKEND_CHOICES,
        default=asr_backends.AUTO,
        help="ASR engine (auto uses MLX on Apple Silicon and patched CT2 elsewhere).",
    )
    parser.add_argument("--device", default="cuda", help="Device override (cpu/cuda/mps).")
    parser.add_argument(
        "--gpu-tier",
        choices=gpu_tier_cli_choices(),
        default=AUTO_GPU_TIER,
        help=gpu_tier_help(),
    )
    parser.add_argument(
        "--asr-decode-batch",
        default=None,
        help=(
            "Windows per decoder generate call ('auto' = the GPU profile's "
            "entry, currently 1 everywhere, i.e. off). Above 1 the aligner "
            "prefetches the next single-window groups in one batched decode; "
            "it is an opt-in because on the default model the end-to-end gain "
            "missed its floor (docs/bench-baselines.md 二十二)."
        ),
    )
    parser.add_argument("--language", default=None, help="Language override.")
    parser.add_argument(
        "--gap",
        type=float,
        default=asr_align.DEFAULT_GAP_SEC,
        help=(
            "Synthetic silence inserted before each next interval when "
            "combining segments (after up to 0.7s of kept real gap audio)."
        ),
    )
    parser.add_argument(
        "--vad-silero-assist",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Two-signal post-pass over the energy VAD: un-suppress creep-"
            "suppressed loud speech under silero voicing, drop unvoiced ghost "
            "intervals, carve unvoiced noise prefixes/bridges, restore "
            "swallowed seams. On by default; --no-vad-silero-assist opts out "
            "for clean, unseparated source audio."
        ),
    )
    parser.add_argument(
        "--qwen-verify",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Second-model verification evidence at the vad-asr tail "
            "(Qwen3-ASR referee, docs/vad-asr.md): auto = when "
            "transformers 5.x is installed, on = require it, off = skip."
        ),
    )
    parser.add_argument(
        "--vad-prefix",
        help=(
            "Path to the VAD stage artifact (default: <output>-vad.json). "
            "Reused when it matches this audio and switches, written when it "
            "does not -- so a rerun of recognition alone skips the VAD pass."
        ),
    )
    parser.add_argument(
        "--lang-redecode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Inline language-vote-collapse redecode "
            "(docs/asr-align.md): "
            "when a group's detected language contradicts the recent "
            "majority, ask the Qwen referee and redecode the group with the "
            "majority language forced; adopt only when the evidence agrees. "
            "auto = when transformers 5.x is installed, on = require it, "
            "off = skip. Default: auto. Auto-language runs only."
        ),
    )
    parser.add_argument(
        "--split-length-scale",
        type=float,
        default=None,
        help=(
            "How long a subtitle may get before the splitter buys a cut "
            f"({segment_split.LENGTH_SCALE_MIN}-{segment_split.LENGTH_SCALE_MAX}, "
            "default 1.0; below 1 = shorter subtitles). Overrides "
            "[segmentation] length_scale in config.toml."
        ),
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    base = input_path.with_suffix("")
    return base.with_name(f"{base.name}-aligned.json")


def default_vad_prefix_path(output_path: Path) -> Path:
    """The VAD artifact beside an aligned JSON, named from the same stem."""

    base = output_path.name.removesuffix("-aligned.json").removesuffix(".json")
    return output_path.with_name(f"{base}-vad.json")


def vad_prefix_energy_path(prefix_path: Path) -> Path:
    """The frame-level track lives beside the JSON, not inside it.

    Hundreds of thousands of frames is a sidecar, not a document -- the same
    line `build_vad_timeline` draws for what belongs in the aligned JSON. The
    JSON is the artifact whose existence means "this stage ran", so it is
    written last and this is what it points at.
    """

    return prefix_path.with_name(f"{prefix_path.stem}-energy.npz")


#: The backend default for the silero assist. It lives here, once: the CLI
#: passes `None` for "the user did not say" and this stage resolves it, so a
#: front end that wants something else overrides deliberately rather than by
#: carrying a second copy of the answer (`README_DEV.md` -> 开发原则).
#:
#: On by default because the pipeline separates first, and the two-signal
#: post-pass exists for exactly that kind of noisy vocal. Clean, unseparated
#: source audio should pass `--no-vad-silero-assist`: there the packing
#: perturbation is paid for nothing.
DEFAULT_VAD_SILERO_ASSIST = True


def resolve_vad_silero_assist(explicit: bool | None = None) -> bool:
    """Three layers, in order: the flag, `[vad] silero_assist`, the default.

    The same shape as `resolve_split_params`, and for the same reason: one
    function owns the whole chain, so every front end lands on the same answer
    and there is exactly one place to read to find out what "unset" means.
    """

    if explicit is not None:
        return bool(explicit)
    configured = app_config.config_bool("vad", "silero_assist")
    if configured is not None:
        return configured
    return DEFAULT_VAD_SILERO_ASSIST


def torch_auxiliary_device(asr_device: str, *, mps_available: bool | None = None) -> str:
    """Map an ASR runtime device to a device understood by PyTorch helpers.

    MLX is an ASR backend, not a valid ``torch.device`` name.  Silero and the
    optional verification helpers therefore need their own placement on an
    MLX run: MPS on Apple Silicon when available, otherwise CPU.
    """

    normalized = str(asr_device or "cpu").strip().lower()
    if normalized != "mlx":
        return normalized
    if mps_available is None:
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        try:
            mps_available = bool(
                backend is not None
                and backend.is_built()
                and backend.is_available()
            )
        except Exception:  # pragma: no cover - platform driver query
            mps_available = False
    return "mps" if mps_available else "cpu"


#: Bumped when a payload written here stops being readable by the loader below.
#: Stored in the artifact, so a stale file is rejected and recomputed rather
#: than misread.
VAD_PREFIX_SCHEMA = 1


@dataclass(frozen=True)
class VadPrefix:
    """Everything the VAD half of this stage produces, before Whisper loads.

    One value with one producer. The alternative -- a second function that
    "does the same work" for the resumable path -- is how the two copies drift:
    the reporting, the device, a later fix to one of them.
    """

    raw_segments: list[dict[str, object]]
    segments: list[dict[str, object]]
    vad_meta: dict[str, object]
    audio_duration: float
    timing: dict[str, float]
    energy_track: vad_energy.VadEnergyTrack


def _prefix_identity(source_path: Path) -> dict[str, object]:
    """What has to match for a stored prefix to describe this run.

    Keyed on the input the caller named rather than the decoded copy the
    readers take -- the same rule the ASR checkpoint follows. A scratch decode
    is deleted when a run succeeds, so keying on it would leave every prefix
    stale by the next run.

    Not a digest of the audio either: the vocal track is hundreds of MB, and
    hashing it would cost more than the VAD pass this saves. Size and mtime
    answer "is this the same file" for an artifact that lives in the same
    output directory, and a rerun of the separator rewrites both.

    **This is the compatibility key, and only that.** It holds what makes a
    stored prefix unreadable or wrong for this run -- nothing else. A run
    parameter like `--vad-silero-assist` is *provenance*: it says how the
    prefix was produced, and a mismatch is worth a warning, not a recompute.
    Resume continues the task's own artifacts; the current parameters do not
    retroactively redefine a stage that already finished (`README_DEV.md` ->
    「复用的依据是任务身份」). It used to live here, which meant flipping that
    switch mid-task silently threw the prefix away.
    """

    stat = source_path.stat()
    return {
        "name": source_path.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def write_vad_prefix(
    prefix_path: Path,
    prefix: VadPrefix,
    *,
    source_path: Path,
    vad_silero_assist: bool,
) -> None:
    """Persist a prefix so the next run can skip straight to Whisper."""

    track = prefix.energy_track
    energy_path = vad_prefix_energy_path(prefix_path)
    arrays: dict[str, np.ndarray] = {
        "energy_db": track.energy_db.detach().cpu().numpy()
    }
    if track.frame_dbfs is not None:
        arrays["frame_dbfs"] = track.frame_dbfs.detach().cpu().numpy()
    payload = {
        "schema": VAD_PREFIX_SCHEMA,
        "source": _prefix_identity(source_path),
        # How it was produced, not what it has to match. Read back for the
        # mismatch warning; never compared for reuse.
        "provenance": {"vad_silero_assist": bool(vad_silero_assist)},
        "audio_duration": float(prefix.audio_duration),
        "timing": {str(key): float(value) for key, value in prefix.timing.items()},
        "vad_meta": prefix.vad_meta,
        "raw_segments": prefix.raw_segments,
        "segments": prefix.segments,
        "energy_track": {
            "hop_sec": float(track.hop_sec),
            "frame_sec": float(track.frame_sec),
            "energy_mode": str(track.energy_mode),
            "arrays": energy_path.name,
        },
    }
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    # Arrays first, document last: the JSON is what existence-skip reads, so it
    # must not be able to name a sidecar that is not there yet.
    energy_temporary = energy_path.with_name(f".{energy_path.name}.part")
    with energy_temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(energy_temporary, energy_path)
    json_temporary = prefix_path.with_name(f".{prefix_path.name}.part")
    json_temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(json_temporary, prefix_path)


def _warn_on_prefix_provenance_mismatch(
    payload: dict[str, object],
    prefix_path: Path,
    *,
    vad_silero_assist: bool,
) -> None:
    """Say so when a reused prefix was produced under a different switch.

    Reads the new `provenance` block and falls back to the legacy position
    inside `source`, so a prefix written before the split still reports rather
    than going quiet.
    """

    provenance = payload.get("provenance")
    stored: object = None
    if isinstance(provenance, dict) and "vad_silero_assist" in provenance:
        stored = provenance.get("vad_silero_assist")
    else:
        source = payload.get("source")
        if isinstance(source, dict) and "vad_silero_assist" in source:
            stored = source.get("vad_silero_assist")
    if stored is None or bool(stored) == bool(vad_silero_assist):
        return
    current_reporter().warning(
        "vad-prefix-provenance",
        f"{prefix_path.name} was produced with --vad-silero-assist="
        f"{bool(stored)}, this run asked for {bool(vad_silero_assist)}",
        impact="the reused intervals are the ones the earlier setting produced",
        action="delete the prefix to recompute, or start a new task for a clean run",
    )


def read_vad_prefix(
    prefix_path: Path,
    *,
    source_path: Path,
    vad_silero_assist: bool,
) -> VadPrefix | None:
    """A stored prefix for exactly this audio, or None.

    None rather than an exception for every "not usable" case: a stale artifact
    is an ordinary state in a tree whose stages skip on existence, and the
    answer to it is to recompute, which is what the caller does with None.

    **`vad_silero_assist` is not part of the match.** A prefix produced with the
    other setting is still a prefix of this audio: it is complete, this stage
    can consume it, and resuming does not error or lose data. So it is reused
    and the mismatch is reported. Wanting the whole thing regenerated under new
    parameters is what a new task is for.

    The comparison is **subset-wise against the compatibility key** rather than
    whole-dict equality, and that is load-bearing for the migration: prefixes
    written before this change carry `vad_silero_assist` inside `source`, so an
    equality test would find "four stored keys vs three expected" and recompute
    **every prefix on disk at the first upgrade** -- the exact behaviour this
    change exists to stop, just happening once.
    """

    if not prefix_path.is_file():
        return None
    try:
        payload = json.loads(prefix_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != VAD_PREFIX_SCHEMA:
            return None
        stored_source = payload.get("source")
        if not isinstance(stored_source, dict):
            return None
        identity = _prefix_identity(source_path)
        if any(stored_source.get(key) != value for key, value in identity.items()):
            return None
        _warn_on_prefix_provenance_mismatch(
            payload, prefix_path, vad_silero_assist=vad_silero_assist
        )
        track_meta = payload["energy_track"]
        energy_path = prefix_path.with_name(str(track_meta["arrays"]))
        with np.load(energy_path) as arrays:
            energy_db = torch.from_numpy(arrays["energy_db"])
            frame_dbfs = (
                torch.from_numpy(arrays["frame_dbfs"])
                if "frame_dbfs" in arrays.files
                else None
            )
        return VadPrefix(
            raw_segments=list(payload["raw_segments"]),
            segments=list(payload["segments"]),
            vad_meta=dict(payload["vad_meta"]),
            audio_duration=float(payload["audio_duration"]),
            timing={
                str(key): float(value)
                for key, value in dict(payload["timing"]).items()
            },
            energy_track=vad_energy.VadEnergyTrack(
                energy_db=energy_db,
                hop_sec=float(track_meta["hop_sec"]),
                frame_sec=float(track_meta["frame_sec"]),
                energy_mode=str(track_meta["energy_mode"]),
                frame_dbfs=frame_dbfs,
            ),
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        # `np.load` on an npz whose zip structure is damaged -- a direct
        # `Exception` subclass, so the tuple above does not cover it.
        zipfile.BadZipFile,
    ) as error:
        current_reporter().debug(
            "vad prefix unusable", {"path": prefix_path.name, "why": str(error)}
        )
        return None


def _report_verify_failed(exc: BaseException) -> None:
    """Downgrade a failed referee pass to a missing-evidence warning.

    The referee produces *evidence, never decisions* (`qwen_referee`), so under
    `auto` its absence costs one layer of cross-checking and nothing else.
    Letting it raise costs the whole run instead -- including the alignment
    pass that just finished and was already paid for, which is what a
    `ConnectTimeout` inside the model load did on a machine that could not
    reach the hub. `on` still raises: that flag is a caller asking for the
    evidence, and returning quietly without it would be the wrong answer.

    Deliberately NOT recorded in `align_meta["qwen_verify"]`: an absent key is
    what makes a later run willing to try again, and a pass lost to a
    transient network fault should get that second chance. A key saying
    "failed" would freeze one bad afternoon into the artifact.

    Same three fields as the `transformers 5.x` branch below, on purpose --
    both are "the evidence did not happen", and a reader comparing two logs
    should not have to notice which one it was.
    """

    current_reporter().warning(
        "qwen-verify-failed",
        f"second-model verification could not run ({type(exc).__name__}: {exc}); "
        "skipping it for this run.",
        impact="少一层校验证据",
    )


def tail_verify_device(device: str) -> str:
    """Where the tail referee runs, once the Whisper pool has been closed.

    Derived from the *resolved* ASR device, and ⚠ **deliberately back to that**
    after a 2026-09-04 reroute was undone the same day.

    The reroute sent this through `lang_redecode.referee_device` so intent,
    tier policy and the referee's own backend capability would be answered
    here as everywhere else -- the argument being that a Whisper which fell
    back for CTranslate2 reasons says nothing about torch, so this referee was
    being kept off a card it could have used. That argument was **entirely
    about speed**, and it rested on "placement changes how long it takes, not
    what it says".

    That premise is false. Production dtype is `bfloat16` on CUDA and
    `float32` on CPU (`qwen_referee._load_model`); the measurement that reads
    like it licenses the swap compares **CPU float32 against CPU bf16/fp16**
    (`asr-align.md`, referee device), a different pair, and no measurement in
    this repository covers the pair that matters. And this referee's output is
    not inert: `stabilization` reads it to decide whether a noise-leg drop
    stands down, i.e. whether a line stays in the subtitle.

    So the reroute would have moved a *numeric* path on the default output
    route, on every machine whose CT2 falls back while torch is fine, in
    exchange for speed alone. With its only justification gone there is
    nothing left to weigh, and this returns to what it did before.

    ⚠ Reinstating it needs the comparison listed under `asr-align.md`
    「待标定」: CPU-float32 against GPU-bf16 on the same material, scoring
    decision agreement and the resulting subtitle diff -- not another appeal
    to the CPU-only measurement.
    """

    return device if str(device).strip().lower().startswith("cuda") else "cpu"


def tail_verification_referee(
    inline,
    *,
    device: str,
    resource_profile,
    asr_context,
    build,
):
    """The referee the tail pass will use, and the device it runs on.

    The two are one decision -- `resolve_verification_referee` reuses the
    inline referee only when the device matches, so asking where first and
    building second is the same step twice over. Its own function so that
    `run_vad_asr` states the intent rather than the assembly.
    """

    return resolve_verification_referee(
        inline,
        device=tail_verify_device(device),
        asr_context=asr_context,
        build=build,
        # The pool is gone, so the whole tier budget.
        vram_budget_gib=resource_profile.usable_gpu_gb,
    )


def contained_verification(run, *, qwen_verify: str):
    """`run()`'s result, or `None` when a failed pass was contained.

    Its own function for the reason `resolve_verification_referee` is: the
    policy here -- which failures end the run and which become a missing
    layer of evidence -- is worth a test that actually exercises it, and a
    source-string guard over `run_vad_asr` cannot tell a correct `except` from
    one that swallows `on` too.

    `None` rather than a sentinel result because the caller has three things to
    skip on failure (the timing, the warm flag and the meta key), and every one
    of them would be a lie about a pass that did not happen.

    Nothing has to be rolled back on the contained path: `apply_verification`
    either returns a new segment list or raises, so the caller's own list is
    still the pre-verification one it always was.
    """

    try:
        return run()
    except Exception as exc:  # noqa: BLE001 - the policy is the point
        if qwen_verify == "on":
            raise
        _report_verify_failed(exc)
        return None


def _recovery_summary(
    stats: Mapping[str, int],
    intervals: list[dict[str, object]],
) -> dict[str, object]:
    """The stage's one line: what it processed, and what it had to recover from.

    Labelled rather than pre-formatted, and zero counters are dropped by the
    renderer -- a clean run says only how much it recognised.
    """

    return {
        "区间": len(intervals),
        "临时召回": stats.get("temporary_recalls", 0),
        "异常隔离": stats.get("isolated_intervals", 0),
        "beam 救援": (
            f"{stats.get('beam_rescue_accepted', 0)}/{stats.get('beam_rescue_attempted', 0)}"
            if stats.get("beam_rescue_attempted")
            else 0
        ),
        "对齐重试": stats.get("alignment_retries", 0),
        "丢弃 group": stats.get("dropped_groups", 0),
    }


def build_vad_timeline(
    intervals: list[dict[str, object]],
    vad_meta: dict[str, object],
) -> dict[str, object]:
    """What the VAD *saw*, as opposed to how it was configured.

    Kept out of ``metadata`` on purpose: that half is the invocation (small,
    diffable, compared across the streaming and in-memory paths), while this is
    observational data a downstream pass consumes -- the splitter's yardstick,
    and what a re-split off an existing aligned JSON would need. Frame-level
    tracks do *not* belong here: thousands of entries are fine in JSON,
    hundreds of thousands are a sidecar.
    """

    return {
        "intervals": [
            {"start": item.get("start"), "end": item.get("end")}
            for item in intervals
        ],
        "pause_hints": vad_meta.get("pause_hints") or {"scorer": [], "padding": []},
    }


def audio_coverage(
    intervals: list[dict[str, object]],
    audio_duration: float,
) -> dict[str, object]:
    """How much of the source was actually handed to the ASR model.

    Recorded every run, not because this chain is suspect -- the
    `explore/speaker-clustering` branch cross-checked our VAD against
    Sortformer frame activity and found **98.2% recall**, with the missing 52 s
    confirmed by Whisper and RMS (median -50 dB) not to be speech. It is
    recorded so that **a future degradation is visible**: a VAD that silently
    starts dropping half the speech produces subtitles that look fine and are
    missing half the content, and nothing else in the artifact says so.

    Deliberately a plain ratio with no threshold attached. There used to be a
    predicate that *judged* the ratio and warned (`preprocessing/vad_failover.py`,
    removed 2026-08-30): it could only fire when the whole file was under 1%
    speech, which is a failure the user already sees as a nearly empty subtitle
    file -- and the pipeline exposes no VAD knob to act on the warning with.
    The reasoning and the measurements are in `docs/bench-baselines.md` 17.12.
    """

    total = max(0.0, float(audio_duration))
    speech = 0.0
    for item in intervals:
        try:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end > start:
            speech += end - start
    return {
        "audio_sec": round(total, 3),
        "speech_sec": round(speech, 3),
        "ratio": round(speech / total, 4) if total > 0 else None,
        "intervals": len(intervals),
    }


def write_aligned_json(
    output_path: Path,
    segments: list[dict[str, object]],
    *,
    vad_meta: dict[str, object],
    align_meta: dict[str, object],
    vad_timeline: dict[str, object],
    audio_duration: float | None = None,
) -> None:
    payload = {
        "segments": segments,
        "vad_timeline": vad_timeline,
        "metadata": {
            "vad": vad_meta.get("vad", {}),
            "asr_align": align_meta,
        },
    }
    # Both quantities: this artifact feeds the span writer *and* the word
    # writer, and they can disagree -- a list whose spans are monotone can
    # still have words running backwards, which is precisely the shape that
    # ships a subtitle jumping back in time.
    for quantity in ("spans", "words"):
        time_order.report_backward(
            segments, using=quantity, where=f"aligned JSON ({output_path.name})"
        )
    # Text right, timestamps garbage is the one failure nothing downstream can
    # see. Report only -- see the module docstring for why it does not repair.
    align_sentinel.report(
        segments,
        audio_duration=audio_duration,
        where=f"aligned JSON ({output_path.name})",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def annotate_segments_with_vad_energy(
    segments: list[dict[str, object]],
    energy_track: vad_energy.VadEnergyTrack,
) -> list[dict[str, object]]:
    """Attach VAD weighted energy, and the absolute-level tier when there is one.

    Two different quantities on purpose. `vad_weighted_energy_db` is the
    adaptive weighted scale the stabilize legs are calibrated on; the tier is
    read off true dBFS, and only the SUSPECT tier can appear here -- anything
    in the drop tier was folded back into non-speech before the decoder ever
    saw it (`preprocessing/energy.py`, the dBFS tiers).

    Tagging is all this does. Whether the tag costs anything follows
    `--qwen-verify`: `qwen_referee.collect_suspect_indices` reads it, so a run
    with the referee off carries the field and buys no inference.
    """

    annotated: list[dict[str, object]] = []
    for segment in segments:
        item = dict(segment)
        value = vad_energy.aggregate_segment_weighted_energy_db(
            energy_track,
            item.get("start"),
            item.get("end"),
        )
        if value is not None:
            item[vad_energy.SEGMENT_ENERGY_FIELD] = value
        tier = _segment_level_tier(item, energy_track)
        if tier is not None:
            item[vad_energy.SEGMENT_LEVEL_TIER_FIELD] = tier
        annotated.append(item)
    return annotated


def _segment_level_tier(
    segment: Mapping[str, object],
    energy_track: vad_energy.VadEnergyTrack,
) -> str | None:
    """The absolute-level tier of a segment's span, or None."""

    if energy_track.frame_dbfs is None:
        return None
    try:
        start = float(segment.get("start"))  # type: ignore[arg-type]
        end = float(segment.get("end"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (end > start):
        return None
    level = vad_energy.span_level_dbfs(energy_track.frame_dbfs, start, end)
    if level is None:
        return None
    return vad_energy.level_tier(*level)


def resolve_split_params(
    explicit: float | None = None,
) -> segment_split.SplitParams:
    """Subtitle-length knob, resolved once for the whole stage.

    Three layers, in order: an explicit value (CLI flag or caller) beats
    ``[segmentation] length_scale`` in ``config.toml``, which beats the
    calibrated code default. Whatever wins is written into
    ``metadata.asr_align.segment_split``, so a produced artifact says which
    length target it was cut for without anyone having to know where the value
    came from.
    """

    scale = explicit
    source = "--split-length-scale"
    if scale is None:
        scale = app_config.config_float("segmentation", "length_scale")
        source = "config.toml [segmentation] length_scale"
    if scale is None:
        return segment_split.DEFAULT_SPLIT_PARAMS
    try:
        return segment_split.split_params_for_length_scale(scale)
    except ValueError as exc:
        raise ValueError(f"{exc} (from {source})") from exc


def ensure_asr_weights(
    model_name: str, *, backend: str = asr_backends.FW_REFINE
) -> hf_weights.HfLoad:
    """Put the ASR weights on disk before the stage tries to load them.

    A `stat` when they are already there, which is every run after the first.
    When they are not, this is what gives the CLI the mirror routing and the
    per-model fallback the desktop's prefetch used to have -- previously the owning
    library fetched them mid-stage, and a mirror having a bad afternoon failed
    the run instead of costing it a retry.

    Only for the models the manifest describes -- the managed default and the
    listed alternatives. A run with `--model something-else` must not pay a
    download of weights it will never load, so an unlisted model keeps the old
    lazy path; so does a model that is not a repository id at all, which is why
    this maps names rather than handing one through.

    The rest is `hf_weights.prepare`, shared with the referee: the pinned
    revision and whether the load may stay offline.
    """

    try:
        from finesub_bootstrap.model_caches import (
            WHISPER_JA_REPO_ID,
            WHISPER_MLX_REPO_ID,
            WHISPER_REPO_ID,
        )

        # Both spellings of the default resolve to the same manifest entry:
        # `large-v3-turbo` is the faster-whisper alias for that repository.
        manifest_ids = {
            asr_align.DEFAULT_MODEL: "whisper",
            WHISPER_REPO_ID: "whisper",
            WHISPER_JA_REPO_ID: "whisper-ja",
            WHISPER_MLX_REPO_ID: "whisper-mlx",
        }
        if backend == asr_backends.MLX_REFINE and model_name == asr_align.DEFAULT_MODEL:
            manifest_id = "whisper-mlx"
        else:
            manifest_id = manifest_ids.get(model_name)
    except Exception:  # noqa: BLE001 - the loader reports for real
        return hf_weights.UNMANAGED
    if manifest_id is None:
        return hf_weights.UNMANAGED
    return hf_weights.prepare(manifest_id)


def _resolve_asr_runtime(model_name, requested_backend, requested_device, profile):
    backend = asr_backends.resolve_backend(requested_backend)
    model_name = asr_backends.resolve_model_name(
        model_name, backend, default_model=asr_align.DEFAULT_MODEL
    )
    load = ensure_asr_weights(model_name)
    device_request = str(requested_device or "cuda")
    device = (
        "mlx"
        if backend == asr_backends.MLX_REFINE
        else resolve_asr_device(device_request, gpu_allowed=bool(profile.gpu))
    )
    return backend, model_name, load, device_request, device


def _backend_metadata(backend: str, load: hf_weights.HfLoad) -> dict[str, object]:
    dependency_contract = (
        {"version": 1, "mlx_whisper": "0.4.3", "mlx": "0.32.2"}
        if backend == asr_backends.MLX_REFINE
        else {
            "version": 1,
            "faster_whisper": "1.2.1",
            "ctranslate2": "4.8.1+finesub0.4.0",
        }
    )
    return {
        "backend": backend,
        "alignment_mode": "one-pass-wt+teacher-force-fallback",
        "model_revision": load.revision or "unmanaged",
        "dependency_contract": dependency_contract,
        backend.replace("-", "_"): {
            "detect_disfluencies": asr_align.FW_REFINE_DETECT_DISFLUENCIES,
            "collect_path_signals": asr_align.FW_REFINE_COLLECT_PATH_SIGNALS,
            "collect_boundary_signals": asr_align.FW_REFINE_COLLECT_BOUNDARY_SIGNALS,
            "event_field": "alignment_events",
        },
    }


def _checkpoint_backend_identity(backend: str, load: hf_weights.HfLoad) -> dict:
    return {
        "backend": backend,
        "model_revision": load.revision or "unmanaged",
        "alignment_mode": "one-pass-wt+teacher-force-fallback",
        "trace_contract_version": 1,
    }


def _record_backend_stats(metadata: dict, backend: str, model) -> None:
    if backend != asr_backends.MLX_REFINE:
        return
    one_pass = int(getattr(model, "one_pass_count", 0))
    fallback = int(getattr(model, "teacher_force_count", 0))
    total = one_pass + fallback
    metadata["mlx_refine"].update(
        {
            "runtime_versions": dict(getattr(model, "runtime_versions", {})),
            "one_pass_windows": one_pass,
            "teacher_force_windows": fallback,
            "one_pass_hit_rate": round(one_pass / total, 4) if total else 0.0,
            "fallback_reasons": dict(getattr(model, "fallback_reasons", {})),
            "peak_memory_bytes": int(getattr(model, "peak_memory_bytes", 0)),
        }
    )


def run_vad_prefix(
    audio_source: Path,
    *,
    device: str,
    vad_silero_assist: bool,
) -> VadPrefix:
    """The VAD half of this stage: energy detection, the optional assist, and
    normalization -- everything `run_vad_asr` does before Whisper is loaded.

    A function rather than a block inside the stage so the resumable path and
    the one-pass path cannot be two implementations of the same thing.
    """

    collector = None
    if vad_silero_assist:
        from ..preprocessing import silero_ghost

        # Rides along on the VAD's normalized blocks: the probabilities are
        # ready by the time detect_segments returns.
        collector = silero_ghost.SileroProbCollector(device)

    try:
        (
            raw_segments,
            vad_meta,
            audio_duration,
            timing,
            energy_track,
        ) = vad_detection.detect_segments(audio_source, observer=collector)
    except Exception as exc:
        raise RuntimeError(f"Failed to load/prepare audio: {exc}") from exc

    if collector is not None:
        # The probabilities were scored inside the VAD pass, so their cost
        # sits in vad_sec; report it rather than let it hide there.
        timing["silero_probs_sec"] = collector.seconds
        t_ghost = time.perf_counter()
        raw_segments, assist_stats = silero_ghost.assist_segments(
            audio_source, raw_segments, energy_track, audio_duration,
            device=device, probs=collector.probs(),
        )
        timing["silero_assist_sec"] = time.perf_counter() - t_ghost
        vad_meta = dict(vad_meta)
        inner_vad = dict(vad_meta.get("vad") or {})
        inner_vad["silero_assist"] = assist_stats
        # Recorded rather than consumed here. The failover predicate runs in
        # the stage, where the *reused* prefix path also passes; persisting the
        # second opinion means a prefix written by an earlier run still carries
        # it, instead of reuse silently downgrading the check to one signal.
        probabilities = collector.probs()
        if probabilities is not None and len(probabilities) > 0:
            inner_vad["silero_voiced_fraction"] = float(
                (probabilities >= 0.5).mean()
            )
        vad_meta["vad"] = inner_vad
        current_reporter().debug(
            "silero assist",
            {
                "intervals": f"{assist_stats['base_intervals']} -> "
                f"{assist_stats['intervals']}",
                "speech": f"{assist_stats['base_speech_sec']:.0f}s -> "
                f"{assist_stats['speech_sec']:.0f}s",
                "ghost_dropped": assist_stats["ghost_dropped"],
                "seams_restored": assist_stats["seams_restored"],
            },
        )

    segments = asr_align.normalize_vad_segments(raw_segments, audio_duration)
    return VadPrefix(
        raw_segments=raw_segments,
        segments=segments,
        vad_meta=vad_meta,
        audio_duration=audio_duration,
        timing=timing,
        energy_track=energy_track,
    )


def _short_vram_line(free: float, needed: float) -> str:
    """The half of a question-5 message that is the same everywhere: the gap.

    What differs is what the gap *costs*, and that is why the two placement
    wrappers below word their own -- see `lang_redecode.referee_device`.
    """

    return (
        f"显卡上只剩 {free:.2f} GiB 空闲，第二模型要与识别模型共驻需要 "
        f"{needed:.2f} GiB"
    )


def redecode_referee_device(
    device: str,
    resource_profile,
    model_name: str,
    decode_batch: int = 1,
    *,
    requested_device: str | None = None,
) -> str:
    """Placement for the inline language-vote referee.

    ⚠ **Deliberately does not ask question 5** (live free VRAM), and that is a
    correctness decision rather than a performance one.

    This referee runs wherever it is placed, and what it says decides whether
    a group's decode is *replaced* (`lang_redecode.maybe_redecode`, adoption).
    Placement therefore switches a numeric path: the production dtype is
    `bfloat16` on CUDA and `float32` on CPU. Nothing in this repository shows
    those two agree -- the measurement that reads like it does
    (`asr-align.md`, referee device) compares **CPU float32 against CPU
    bf16/fp16**, which is a different pair. Letting the driver's *live* free
    figure choose between them would make the same audio produce different
    subtitles depending on what else happened to be open, which is the one
    thing tier selection is deliberately shaped to avoid.

    The tier's own budget still decides, exactly as before: stable per machine
    and per run. ⚠ Note that the tier already places this referee on the CPU
    at `entry` and on the card from `standard` up, so CPU-versus-GPU divergence
    -- if there is any -- already differs *between machines* today. That is a
    pre-existing open question (`asr-align.md`, 待标定); what must not be added
    on top of it is divergence *within* one machine between two runs.

    `referee_warm_device` does ask question 5, and may: a veto there only skips
    the preload, and the tail referee is placed afterwards by
    `tail_verify_device`, so no device that produces output changes.
    """

    from . import lang_redecode

    return lang_redecode.referee_device(
        device,
        resource_profile,
        model_name,
        decode_batch,
        requested_device=requested_device,
    )


def referee_warm_device(
    *,
    qwen_verify: str,
    device: str,
    resource_profile,
    model_name: str,
    decode_batch: int = 1,
    requested_device: str | None = None,
) -> Optional[str]:
    """Where to warm the verification referee while Whisper is still decoding.

    `None` means "do not": the run will not verify, is not on CUDA, or the
    profile's spare VRAM beside the resident pool does not hold the referee
    (`lang_redecode.referee_device` -- the entry tier lands here, and there the
    load keeps waiting for the pool to be released). On the tiers that fit,
    the ~3 s load -- 71% of the referee's cost on a typical run, A4 -- hides
    under the decode instead of extending the stage.

    ⚠ Called *after* `FwRefineModelPool.warm`, and `pool_resident=True` says
    so: `referee_device`'s live free-VRAM veto has to know the pool is already
    paid for out of the figure it reads, or it subtracts the same 2-4 GiB
    twice and sends the referee to the CPU on exactly the tiers this exists
    for. The call order is what makes that true -- keep the warm above it.

    ⚠ A question-5 veto here costs **only the warm**, so it is a `debug` and
    not a warning. Returning `None` means "do not preload during the decode";
    the tail referee is placed afterwards by `tail_verify_device`, which does
    not ask question 5, so it still goes on the card. Saying "the check will
    run on the CPU" here -- or quoting the referee's 2.2-2.5x -- would be
    false: what is actually lost is the ~3 s load that would have hidden under
    the decode. It is also not a thing the user can act on mid-run, which is
    the line `docs/reporting.md` draws for warnings.
    """

    if qwen_verify == "off":
        return None
    from . import lang_redecode

    def veto(free: float, needed: float) -> None:
        current_reporter().debug(
            f"{_short_vram_line(free, needed)}; skipping the warm under decode",
            {"free_gib": f"{free:.2f}", "needed_gib": f"{needed:.2f}"},
        )

    # No early return on the ASR device any more: since the ASR stage asks
    # CTranslate2 and the referee asks torch, "Whisper is on the CPU" no longer
    # implies "the card is unavailable" -- it can now mean the card is *free*.
    placed = lang_redecode.referee_device(
        device,
        resource_profile,
        model_name,
        decode_batch,
        requested_device=requested_device,
        pool_resident=True,
        live_vram_veto=veto,
    )
    return placed if placed.strip().lower().startswith("cuda") else None


class RefereeWarm:
    """The helper thread that loads the referee during the Whisper decode.

    Its phase table is its own (the collector is thread-local) and is merged
    into the stage's at `join`, so `qwen.load` lands in the same ledger it
    always did. A failed load is not an error here: the verification pass
    asks for the model again and reports the real reason then. Only a
    one-line summary of the failure is kept, never the exception: its
    traceback would pin the loader's frames -- and with them a half-built
    model -- until the tail pass has already loaded a second copy.
    """

    def __init__(self, referee) -> None:
        self.referee = referee
        self.phases: dict = {}
        self.elapsed_sec = 0.0
        self.error: Optional[str] = None
        reporter = current_reporter()

        def run() -> None:
            bind_reporter(reporter)
            started = time.perf_counter()
            try:
                with phase_timing.collect(into=self.phases):
                    referee.warm()
            except BaseException as exc:  # noqa: BLE001 - reported at use
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.elapsed_sec = time.perf_counter() - started

        self._thread = threading.Thread(
            target=run, name="qwen-referee-warm", daemon=True
        )
        self._thread.start()

    def join(self, into: dict) -> None:
        self._thread.join()
        phase_timing.merge(into, self.phases)


def resolve_verification_referee(
    inline, *, device, asr_context, build, vram_budget_gib=None
):
    """The referee the verification pass should use, and what it costs.

    Reusing the inline language referee on the same device avoids a second
    model load after an adopted or checked group; a 4GB run deliberately drops
    its CPU inline referee for the normal post-ASR CUDA one once Whisper is
    gone, and that means closing the one being replaced.

    ⚠ The inline referee is built WITHOUT the ASR context, deliberately:
    `lang_redecode` and `lang_audit` read its language field, and a list of
    Japanese names in the system prompt would bias exactly the quantity the
    audit exists to cross-check with an independent model. Both have finished
    by the time this runs, so the context goes on HERE -- miss it and a run
    with `--asr-context terms` under the default `--lang-redecode auto`
    silently verifies with nothing injected, which is the common path.

    Its own function so that reuse-versus-build is testable without a model:
    a source-string guard cannot tell that the context actually lands.
    """

    if inline is not None and inline.requested_device == device:
        inline.set_context(asr_context)
        # Built beside the pool with the spare-beside-Whisper figure; the
        # pool is gone now, so the whole tier budget applies.
        inline.set_vram_budget(vram_budget_gib)
        return inline
    if inline is not None:
        inline.close()
    return build(
        device=device, context=asr_context, vram_budget_gib=vram_budget_gib
    )


def run_vad_asr(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name: str = asr_align.DEFAULT_MODEL,
    asr_backend: str = asr_backends.AUTO,
    device: str = "cuda",
    language: Optional[str] = None,
    gap_sec: float = asr_align.DEFAULT_GAP_SEC,
    # `auto` = ask the card. A tier names what CLASS of card this is, not a
    # cap on what the pipeline may use; `resolve_gpu_tier` owns the answer.
    gpu_tier: str = AUTO_GPU_TIER,
    # None = "the caller did not say"; resolved once, here, by
    # `resolve_vad_silero_assist`. A literal default would be a second copy of
    # the answer that lives in that resolver.
    vad_silero_assist: bool | None = None,
    qwen_verify: str = "auto",
    lang_redecode: str = "auto",
    split_length_scale: float | None = None,
    asr_decode_batch: int | str | None = None,
    # Plain text: names the recording is likely to contain, assembled by
    # whoever knows where names live. This stage never learns -- `speech` must
    # not import `llm`, so the knowledge base reaches it as a string.
    asr_context: str = "",
    run_metadata_path: str | Path | None = None,
    vad_prefix_path: str | Path | None = None,
) -> Path:
    # Before anything else: an out-of-range knob must not surface after the
    # GPU work is already done.
    split_params = resolve_split_params(split_length_scale)
    decode_batch = resolve_asr_decode_batch(
        asr_decode_batch, gpu_tier=gpu_tier
    )
    # Resolved once, here, so every front end lands on the same answer and the
    # rest of this function sees a plain bool.
    vad_silero_assist = resolve_vad_silero_assist(vad_silero_assist)
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    # Normalize "auto" to None (whisper auto-detection).
    if language and language.strip().lower() == "auto":
        language = None
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_output_path(input_path)
    )
    # A stage of its own, on the same existence-skip terms as everything else:
    # deleting `-aligned.json` to re-run recognition, re-split, or re-verify no
    # longer re-runs the VAD pass that produced the same intervals.
    vad_prefix_path = (
        Path(vad_prefix_path).expanduser().resolve()
        if vad_prefix_path is not None
        else None
    )
    # Only the readers take the decoded path; checkpoint keys and output naming
    # stay on the input the caller named, so a rerun still resumes.
    audio_source, temporary_audio = ensure_decodable_input(input_path, output.parent)
    if temporary_audio is not None and run_metadata_path is not None:
        # Written now, not at the end: the copy this cleans up after is the one
        # a *failed* run leaves behind, and a stage that dies never reaches its
        # own tidying.
        record_scratch_file(run_metadata_path, temporary_audio)
    stage_completed = False
    resource_profile = get_resource_profile(gpu_tier)
    backend, model_name, asr_load, requested_device, device = _resolve_asr_runtime(
        model_name, asr_backend, device, resource_profile
    )
    auxiliary_device = torch_auxiliary_device(device)
    device_for_usage = None
    memory_sampler = None
    model_pool = None
    redecode_referee = None
    tail_referee = None
    referee_warm = None
    gpu_stage_lease: GpuStageLease | None = None
    watchdog = stall_watchdog.arm("vad-asr")
    try:
        t_start = time.perf_counter()
        # The ASR stage decodes with CTranslate2, so it asks CTranslate2 --
        # not `resolve_device`, which answers for torch. The request is kept:
        # "the user asked for the CPU" and "the CPU is all this machine can do"
        # are different facts, and the referee below needs the first one.
        # `None` is "not chosen", i.e. the code default -- never the resolved
        # device below, which is "cpu" after a CT2-only fallback too.
        device_for_usage = device
        # After `resolve_device`, so an explicit `--device cpu` and a CPU
        # fallback both stay silent. Same rule and same place whether the tier
        # came from `auto` or by name.
        warn_if_vram_is_short(resource_profile, stage="VAD-ASR", device=device)
        asr_align.reset_peak_gpu_memory_stats_for_run(device_for_usage)
        memory_sampler = asr_align.start_stage_memory_sampling()
        align_meta = asr_align.asr_align_metadata(
            model=model_name,
            device=device,
            language=language,
            gap_sec=gap_sec,
        )
        align_meta.update(_backend_metadata(backend, asr_load))
        # Provenance, NOT a compatibility key: changing the batch size must
        # not expire a partial. Resuming with a different one is a legitimate
        # thing to do and loses no data (bench-baselines 10.4).
        align_meta["asr_decode_batch"] = decode_batch
        align_meta["gpu_tier"] = resource_profile.gpu_tier
        align_meta["gpu_limit_gb"] = resource_profile.usable_gpu_gb
        align_meta["ram_budget_gb"] = resource_profile.ram_budget_gb
        align_meta["segment_split"] = segment_split.split_params_metadata(split_params)
        if split_params is not segment_split.DEFAULT_SPLIT_PARAMS:
            current_reporter().debug(
                "subtitle length scale",
                {
                    "scale": align_meta["segment_split"]["length_scale"],
                    "target": f"{split_params.dur_ideal_hi:.2g}s / "
                    f"{split_params.chars_ideal_hi:.3g} chars",
                },
            )

        prefix = (
            read_vad_prefix(
                vad_prefix_path,
                # The named input, not `audio_source`: a decoded scratch copy
                # is deleted on success, so keying on it would never match.
                source_path=input_path,
                vad_silero_assist=vad_silero_assist,
            )
            if vad_prefix_path is not None
            else None
        )
        prefix_reused = prefix is not None
        if prefix is None:
            prefix = run_vad_prefix(
                audio_source,
                device=auxiliary_device,
                vad_silero_assist=vad_silero_assist,
            )
            if vad_prefix_path is not None:
                write_vad_prefix(
                    vad_prefix_path,
                    prefix,
                    source_path=input_path,
                    vad_silero_assist=vad_silero_assist,
                )
        raw_segments = prefix.raw_segments
        segments = prefix.segments
        vad_meta = prefix.vad_meta
        audio_duration = prefix.audio_duration
        # Copied: the later stages add their own keys to it, and a reused
        # prefix's own dict must not grow a second run's measurements.
        timing = dict(prefix.timing)
        energy_track = prefix.energy_track
        if vad_prefix_path is not None:
            align_meta["vad_prefix"] = {
                "path": vad_prefix_path.name,
                "reused": prefix_reused,
            }

        # Recorded here -- where the fresh and reused paths meet -- rather than
        # inside `run_vad_prefix`: a reused prefix needs recording just as much
        # as a fresh one, arguably more, since nothing else ever re-examines it.
        #
        # Above the empty early-return on purpose: a run that found no speech
        # is precisely the case `audio_coverage` exists to record, and it is
        # the one the return used to skip.
        coverage = audio_coverage(segments, audio_duration)
        align_meta["audio_coverage"] = coverage
        if run_metadata_path is not None:
            update_run_metadata(
                run_metadata_path, {"audio_coverage": dict(coverage)}
            )

        if not raw_segments or not segments:
            timing["total_sec"] = time.perf_counter() - t_start
            align_meta["timing"] = {
                key: round(value, 3) for key, value in timing.items()
            }
            write_aligned_json(
                output,
                [],
                vad_meta=vad_meta,
                align_meta=align_meta,
                vad_timeline=build_vad_timeline([], vad_meta),
            )
            current_reporter().warning(
                "no-speech",
                "VAD found no speech in this audio; the subtitles will be empty.",
            )
            stage_completed = True
            return output

        gpu_stage_lease = GPU_STAGE_GATE.acquire(
            "wt",
            enabled=str(device).strip().lower().startswith("cuda"),
        )

        # Stream the alignment audio from disk in blocks instead of holding the
        # whole recording in RAM. Matches the standalone asr_align.main config
        # (600s core + 10s pad, no bandpass) so ASR input stays consistent.
        def _make_audio_loader() -> asr_align.AudioBlockLoader:
            return asr_align.AudioBlockLoader(
                str(audio_source),
                target_sr=vad_energy.TARGET_SR,
                block_seconds=600.0,
                pad_seconds=10.0,
                preprocess=False,
            )

        # Inline language-vote-collapse redecode
        # (docs/asr-align.md).
        # Resolved before the checkpoint key: enabled-and-available is what
        # changes the partials, so it is what the fingerprint must carry.
        # `on` buys one more thing than `auto` does: the run-level language
        # audit (`lang_audit`), the only check that survives when the whole run
        # is mislabelled the same way -- including under `--language`, where the
        # trigger is inert because there are no votes to contradict and where
        # "the user forced the wrong language" is a real, reachable failure.
        #
        # Not on by default, deliberately. It costs a referee load plus a
        # handful of clips (16.9s measured, docs/bench-baselines.md 15.7) on a
        # run with nothing wrong with it, and the rate of the failure it
        # catches is exactly what the blindness prevented us from measuring.
        # Same shape as P1's deferred auto-fallback: ship the check, leave the
        # default alone until there is a number.
        from . import lang_audit

        lang_redecoder = None
        redecode_mode = lang_audit.resolve_mode(lang_redecode, language)
        audit = redecode_mode in ("redecode+audit", "audit-only")
        audit_only = redecode_mode == "audit-only"
        if redecode_mode is not None:
            try:
                # Same availability probe as --qwen-verify below.
                from transformers import AutoModelForMultimodalLM  # noqa: F401

                from ..verification import qwen_referee as redecode_qwen
                from . import lang_redecode as lang_redecode_mod
            except Exception as exc:
                if lang_redecode == "on":
                    raise RuntimeError(
                        "Missing dependency for --lang-redecode on: the [asr] "
                        "extra ships transformers 5.x (see docs/vad-asr.md)."
                    ) from exc
                current_reporter().warning(
                    "lang-redecode-unavailable",
                    "transformers 5.x not available; skipping inline "
                    "language-vote-collapse redecode.",
                    impact="语言票翻转窗口不会被重解",
                )
            else:
                redecode_device = redecode_referee_device(
                    device,
                    resource_profile,
                    model_name,
                    decode_batch,
                    requested_device=requested_device,
                )
                redecode_referee = redecode_qwen.QwenReferee(
                    device=redecode_device,
                    vram_budget_gib=lang_redecode_mod.referee_vram_budget(
                        resource_profile,
                        model_name,
                        # A fact about the pool, not about the tier: when the
                        # ASR went to the CPU there is nothing to sit beside.
                        beside_pool=device.strip().lower().startswith("cuda"),
                        decode_batch=decode_batch,
                    ),
                )
                lang_redecoder = lang_redecode_mod.LangRedecoder(
                    redecode_referee, str(audio_source), contained=lang_redecode != "on"
                )
                align_meta["lang_redecode"] = {
                    "device": redecode_device,
                    "mode": redecode_mode,
                }

        checkpoint_key = checkpoint_store.build_key(
            model_name=model_name,
            language=language,
            gap_sec=gap_sec,
            audio_path=input_path,
            detect_disfluencies=asr_align.FW_REFINE_DETECT_DISFLUENCIES,
            # Audit-only is deliberately absent from the key: it reads the
            # decode, it never changes it, so it cannot invalidate a partial.
            lang_redecode=lang_redecoder is not None and not audit_only,
            **_checkpoint_backend_identity(backend, asr_load),
        )
        model_pool = asr_backends.build_pool(
            backend,
            model_name,
            device=device,
            size=1,
            refine_sec=asr_align.REFINE_SEC,
            load=asr_load,
        )
        t0 = time.perf_counter()
        model_pool.warm()
        timing["whisper_load_sec"] = time.perf_counter() - t0

        # Load the verification referee under the decode where the profile
        # has room for both models. The inline language referee is that same
        # object when it sits on the same device (context-free, as it must
        # be; the context goes on at reuse); otherwise a tail referee is built
        # now and handed to the verification pass below.
        warm_device = referee_warm_device(
            qwen_verify=qwen_verify,
            device=device,
            resource_profile=resource_profile,
            model_name=model_name,
            decode_batch=decode_batch,
            requested_device=requested_device,
        )
        if warm_device is not None:
            try:
                from transformers import AutoModelForMultimodalLM  # noqa: F401

                from ..verification import qwen_referee as warm_qwen
            except Exception:
                warm_device = None  # the verification pass reports or raises
        if warm_device is not None:
            if (
                redecode_referee is not None
                and redecode_referee.requested_device == warm_device
            ):
                referee_warm = RefereeWarm(redecode_referee)
            else:
                from . import lang_redecode as budget_mod

                tail_referee = warm_qwen.QwenReferee(
                    device=warm_device,
                    vram_budget_gib=budget_mod.referee_vram_budget(
                        resource_profile,
                        model_name,
                        beside_pool=device.strip().lower().startswith("cuda"),
                        decode_batch=decode_batch,
                    ),
                )
                referee_warm = RefereeWarm(tail_referee)

        t0 = time.perf_counter()
        # `asr_align_sec` used to be the whole ASR budget in one opaque number,
        # which made every estimate of batching, stage overlap and referee cost
        # a guess (docs/plans/crispasr-followups.md -> A3.2). The collector breaks it
        # into encode / decode / refine / rescue without changing what runs.
        with phase_timing.collect() as asr_phases:
            with asr_align.collecting_stats() as recovery_stats:
                with bounded_intra_op_threads(1), model_pool.lease() as model:
                    aligned_segments = asr_align.align_segments(
                        segments,
                        None,
                        vad_energy.TARGET_SR,
                        model=model,
                        gap_sec=gap_sec,
                        language=language,
                        audio_loader=_make_audio_loader(),
                        checkpoint_path=checkpoint_store.path_for_output(output),
                        checkpoint_key=checkpoint_key,
                        lang_redecode=lang_redecoder,
                        decode_batch=decode_batch,
                        batch_decode_fn=asr_backends.batch_decoder(backend),
                    )
                if lang_redecoder is not None and audit:
                    # Inside the collector on purpose: the audit buys referee
                    # inference, and an untimed cost is how the referee's own
                    # share stayed unknown for so long (A4). Outside the
                    # Whisper lease, so the two models do not need the card at
                    # the same moment on a 4GB profile.
                    lang_redecoder.run_audit()
        timing["asr_align_sec"] = time.perf_counter() - t0
        align_meta["recovery"] = dict(recovery_stats)
        _record_backend_stats(align_meta, backend, model)
        if referee_warm is not None:
            referee_warm.join(asr_phases)
            timing["qwen_warm_sec"] = referee_warm.elapsed_sec
        if lang_redecoder is not None:
            align_meta["lang_redecode"].update(lang_redecoder.stats())
            if qwen_verify == "off":
                # No tail consumer can reuse it. Release early, especially on
                # the 4GB profile where it may hold multi-GiB CPU weights.
                redecode_referee.close()
                redecode_referee = None

        # Word-start correction (docs/asr-align.md): resolve [*] disfluency
        # blocks and leading candidates against the energy track, then apply
        # the VAD interval / pause-hint anchor clamps. Runs before the ghost
        # and overlap passes so they see the final spans.
        # Both sources clamp the same way; only the artifact keeps them apart.
        hint_sources = vad_meta.get("pause_hints") or {}
        pause_hints = sorted(
            {
                hint
                for source in hint_sources.values()
                for hint in source
            }
        )
        t_post = time.perf_counter()
        aligned_segments, disfluency_stats = word_starts.apply_disfluency_rules(
            aligned_segments,
            energy_track=energy_track,
        )
        aligned_segments, clamp_stats = word_starts.clamp_word_starts(
            aligned_segments,
            vad_intervals=segments,
            pause_hints=pause_hints,
        )
        timing["word_starts_sec"] = time.perf_counter() - t_post
        align_meta["word_start_correction"] = {
            **disfluency_stats,
            **clamp_stats,
        }

        t_post = time.perf_counter()
        nonempty_segments = segment_ops.drop_empty_segments(aligned_segments)
        # ⚠ BEFORE `split_segments`, and that is load-bearing, not incidental.
        # The stabilize ladder's tags are per-segment and rate-based, so a
        # ghost the DP has merged into a real piece inherits that piece's
        # length and rate and stops tripping any of them. Measured by turning
        # this off and diffing the shipped SRT: on one clip two ghosts came
        # back as `…のスッスッ` appended to a real line (bench-baselines 20.11).
        nonempty_segments, ghost_drops = segment_ops.drop_ghost_duplicate_segments(
            nonempty_segments
        )
        align_meta["ghost_duplicate_segments_dropped"] = ghost_drops
        for record in ghost_drops:
            # Spelled out rather than interpolating the record: the field
            # became a dict in 2026-08-31 so an audit could reconstruct the
            # span, and a raw dict repr in a user-facing warning is that
            # change leaking into a line it has nothing to do with.
            current_reporter().warning(
                "ghost-duplicate-dropped",
                "dropped ghost duplicate segment "
                f"({record['start']:.3f}-{record['end']:.3f} "
                f"text='{record['text']}')",
            )
        monotonic_segments = segment_ops.clamp_segment_overlaps(nonempty_segments)
        monotonic_segments = segment_ops.extend_zero_length_segments(
            monotonic_segments
        )
        timing["segment_cleanup_sec"] = time.perf_counter() - t_post
        # DP split of over-long whisper segments (docs/segmentation-split.md);
        # runs before energy annotation so pieces get their own energy.
        t_post = time.perf_counter()
        split_result_segments = segment_split.split_segments(
            monotonic_segments,
            segments,
            params=split_params,
        )
        timing["dp_split_sec"] = time.perf_counter() - t_post
        synthetic_word_segments = sum(
            1
            for segment in split_result_segments
            if any(
                word.get(segment_split.SYNTHETIC_WORD_KEY)
                for word in segment.get("words") or []
            )
        )
        align_meta["segment_split"]["synthetic_word_segments"] = (
            synthetic_word_segments
        )
        if synthetic_word_segments:
            current_reporter().warning(
                "synthetic-word-span",
                "synthesized one segment-span word for "
                f"{synthetic_word_segments} text-only ASR segment(s).",
            )
        t_post = time.perf_counter()
        energy_segments = annotate_segments_with_vad_energy(
            split_result_segments,
            energy_track,
        )
        timing["energy_annotate_sec"] = time.perf_counter() - t_post

        # Second-model verification evidence (docs/asr-align.md): suspects
        # and coverage gaps get a Qwen3-ASR re-recognition, recorded as
        # fields for downstream deciders. Runs after the Whisper pool is
        # released so the referee (~2.3 GB with its batched decode) fits every
        # GPU budget.
        if qwen_verify != "off":
            try:
                # Only the transformers 5.x line has the multimodal class the
                # referee needs; probing it here keeps the referee lazy.
                from transformers import AutoModelForMultimodalLM  # noqa: F401

                from ..verification import qwen_referee
            except Exception as exc:
                if qwen_verify == "on":
                    raise RuntimeError(
                        "Missing dependency for --qwen-verify on: the [asr] "
                        "extra ships transformers 5.x (see docs/vad-asr.md)."
                    ) from exc
                current_reporter().warning(
                    "qwen-verify-unavailable",
                    "transformers 5.x not available; skipping second-model "
                    "verification evidence.",
                    impact="少一层校验证据",
                )
            else:
                model_pool.close()
                t0 = time.perf_counter()
                # Reuse, replace-and-close, or build -- and attach the ASR
                # context, which the inline referee must not have had.
                referee = tail_verification_referee(
                    redecode_referee if redecode_referee is not None else tail_referee,
                    device=device,
                    resource_profile=resource_profile,
                    asr_context=asr_context,
                    build=qwen_referee.QwenReferee,
                )
                redecode_referee = None
                tail_referee = None
                # Continues the same table rather than starting a second one:
                # the referee runs after the align scope closed, but it may
                # already have run *inside* it for a language-vote redecode.
                # `into=` adds the two up; a separate table merged with
                # `dict.update` would have thrown the inline run's numbers away
                # (docs/bench-baselines.md -> P8/A4).
                with phase_timing.collect(into=asr_phases):
                    try:
                        verified = contained_verification(
                            lambda: qwen_referee.apply_verification(
                                energy_segments,
                                vad_intervals=segments,
                                audio_path=str(audio_source),
                                referee=referee,
                            ),
                            qwen_verify=qwen_verify,
                        )
                    finally:
                        referee.close()
                if verified is not None:
                    energy_segments, verify_stats = verified
                    timing["qwen_verify_sec"] = time.perf_counter() - t0
                    if referee_warm is not None:
                        verify_stats["warmed_under_decode"] = referee_warm.error is None
                    align_meta["qwen_verify"] = verify_stats

        # Serialised once, after every scope that can contribute has closed --
        # the referee may add to it from two different places.
        align_meta["asr_phases"] = phase_timing.as_dict(asr_phases)
        output_segments = [asr_align.round_floats(seg) for seg in energy_segments]
        total = time.perf_counter() - t_start
        timing["total_sec"] = total
        align_meta["timing"] = {
            key: round(value, 3) for key, value in timing.items()
        }
        write_aligned_json(
            output,
            output_segments,
            vad_meta=vad_meta,
            align_meta=align_meta,
            # The normalized intervals, i.e. exactly the yardstick the splitter
            # was scored against (post silero assist, post normalization).
            vad_timeline=build_vad_timeline(segments, vad_meta),
            audio_duration=audio_duration,
        )
        reporter = current_reporter()
        reporter.summary("aligned", _recovery_summary(recovery_stats, segments))
        # Timing is written to the metadata sidecar either way; on screen it is
        # profiling, not progress.
        reporter.debug(
            "timing",
            {
                key: f"{timing[key]:.3f}"
                for key in (
                    "loading_sec",
                    "energy_sec",
                    "noise_sec",
                    "vad_sec",
                    "whisper_load_sec",
                    "asr_align_sec",
                    "word_starts_sec",
                    "segment_cleanup_sec",
                    "dp_split_sec",
                    "energy_annotate_sec",
                    "qwen_verify_sec",
                )
                if key in timing
            }
            | {"total_sec": f"{total:.3f}"},
        )
        # The split of `asr_align_sec`. Separate line because it answers a
        # different question than the stage totals above: not "which stage is
        # slow" but "what is the ASR stage made of".
        if align_meta.get("asr_phases"):
            current_reporter().debug(
                "asr phases",
                {
                    name: f"{stat['exclusive_s']:.3f}s self / "
                    f"{stat['inclusive_s']:.3f}s total x{stat['calls']}"
                    for name, stat in align_meta["asr_phases"].items()
                },
            )
        stage_completed = True
        return output
    finally:
        # Only on success: a failed run keeps it so a rerun skips the decode.
        if stage_completed and temporary_audio is not None:
            try:
                temporary_audio.unlink(missing_ok=True)
            except Exception:
                pass
        for leftover in (redecode_referee, tail_referee):
            if leftover is not None:
                try:
                    leftover.close()
                except Exception:
                    pass
        # Release the Whisper models so downstream stages (LLM) start with
        # a clean GPU. Mirrors preprocessing.separation's cleanup pattern.
        if model_pool is not None:
            try:
                model_pool.close()
            except Exception:
                pass
        gc.collect()
        if device_for_usage is not None and device_for_usage.strip().lower() == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        if gpu_stage_lease is not None:
            gpu_stage_lease.release()
            gpu_stage_lease = None
        asr_align.print_peak_resource_usage(
            device_for_usage, resource_profile, sampler=memory_sampler
        )
        watchdog.disarm()


def main() -> int:
    args = parse_args()
    try:
        with reporting_to(terminal_reporter()):
            run_vad_asr(
                args.input,
                output_path=args.output,
                model_name=args.model,
                asr_backend=args.asr_backend,
                device=args.device,
                language=args.language,
                gap_sec=args.gap,
                gpu_tier=args.gpu_tier,
                vad_silero_assist=args.vad_silero_assist,
                qwen_verify=args.qwen_verify,
                lang_redecode=args.lang_redecode,
                split_length_scale=args.split_length_scale,
                asr_decode_batch=args.asr_decode_batch,
                vad_prefix_path=(
                    args.vad_prefix
                    if args.vad_prefix
                    else default_vad_prefix_path(
                        Path(args.output).expanduser()
                        if args.output
                        else default_output_path(
                            Path(args.input).expanduser().resolve()
                        )
                    )
                ),
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
