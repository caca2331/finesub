"""VAD-energy + Whisper alignment stage for vocal audio."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import json
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from . import transcribe as asr_align
from . import checkpoint as checkpoint_store
from . import segments as segment_ops
from . import word_starts
from ..postprocessing import segmentation as segment_split
from ..runtime.resources import (
    DEFAULT_GPU_BUDGET_GB,
    get_resource_profile,
    gpu_budget_choices,
)
from ..runtime.gpu_stage_gate import GPU_STAGE_GATE, GpuStageLease
from ..runtime.device import resolve_device
from ..runtime import stall_watchdog
from ..runtime.thread_budget import bounded_intra_op_threads
from ..preprocessing import energy as vad_energy
from ..preprocessing import vad as vad_detection
from ..preprocessing.audio import ensure_decodable_input
from ...run_metadata import record_scratch_file
from ... import config as app_config
from ...reporting import current_reporter, reporting_to, terminal_reporter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VAD-energy + Whisper alignment.")
    parser.add_argument("input", help="Path to vocal audio.")
    parser.add_argument("--output", help="Path to output JSON.")
    parser.add_argument("--model", default=asr_align.DEFAULT_MODEL, help="Whisper model name.")
    parser.add_argument("--device", default="cuda", help="Device override (cpu/cuda).")
    parser.add_argument(
        "--gpu-budget-gb",
        type=int,
        choices=gpu_budget_choices(),
        default=DEFAULT_GPU_BUDGET_GB,
        help="GPU memory budget profile in GiB (default: 4).",
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
        action="store_true",
        help=(
            "Two-signal post-pass over the energy VAD: un-suppress creep-"
            "suppressed loud speech under silero voicing, drop unvoiced ghost "
            "intervals, carve unvoiced noise prefixes/bridges, restore "
            "swallowed seams. Opt-in; intended for noisy separated vocals."
        ),
    )
    parser.add_argument(
        "--qwen-verify",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Second-model verification evidence (Qwen3-ASR referee, "
            "docs/asr-align.md): auto = run when the qwen-asr package is "
            "installed, on = require it, off = skip."
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


def write_aligned_json(
    output_path: Path,
    segments: list[dict[str, object]],
    *,
    vad_meta: dict[str, object],
    align_meta: dict[str, object],
    vad_timeline: dict[str, object],
) -> None:
    payload = {
        "segments": segments,
        "vad_timeline": vad_timeline,
        "metadata": {
            "vad": vad_meta.get("vad", {}),
            "asr_align": align_meta,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def annotate_segments_with_vad_energy(
    segments: list[dict[str, object]],
    energy_track: vad_energy.VadEnergyTrack,
) -> list[dict[str, object]]:
    """Attach VAD weighted energy using each final segment's time span."""

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
        annotated.append(item)
    return annotated


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


def run_vad_asr(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name: str = asr_align.DEFAULT_MODEL,
    device: str = "cuda",
    language: Optional[str] = None,
    gap_sec: float = asr_align.DEFAULT_GAP_SEC,
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB,
    vad_silero_assist: bool = False,
    qwen_verify: str = "auto",
    split_length_scale: float | None = None,
    run_metadata_path: str | Path | None = None,
) -> Path:
    # Before anything else: an out-of-range knob must not surface after the
    # GPU work is already done.
    split_params = resolve_split_params(split_length_scale)
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
    # Only the readers take the decoded path; checkpoint keys and output naming
    # stay on the input the caller named, so a rerun still resumes.
    audio_source, temporary_audio = ensure_decodable_input(input_path, output.parent)
    if temporary_audio is not None and run_metadata_path is not None:
        # Written now, not at the end: the copy this cleans up after is the one
        # a *failed* run leaves behind, and a stage that dies never reaches its
        # own tidying.
        record_scratch_file(run_metadata_path, temporary_audio)
    stage_completed = False
    resource_profile = get_resource_profile(gpu_budget_gb)
    device_for_usage = None
    memory_sampler = None
    model_pool = None
    gpu_stage_lease: GpuStageLease | None = None
    watchdog = stall_watchdog.arm("vad-asr")
    try:
        t_start = time.perf_counter()
        device = resolve_device(device, context="VAD-ASR")
        device_for_usage = device
        asr_align.reset_peak_gpu_memory_stats_for_run(device_for_usage)
        memory_sampler = asr_align.start_stage_memory_sampling()
        align_meta = asr_align.asr_align_metadata(
            model=model_name,
            device=device,
            language=language,
            gap_sec=gap_sec,
        )
        align_meta["backend"] = "fw-refine"
        align_meta["fw_refine"] = {
            "detect_disfluencies": asr_align.FW_REFINE_DETECT_DISFLUENCIES,
            "collect_path_signals": asr_align.FW_REFINE_COLLECT_PATH_SIGNALS,
            "collect_boundary_signals": asr_align.FW_REFINE_COLLECT_BOUNDARY_SIGNALS,
            "event_field": "alignment_events",
        }
        align_meta["gpu_budget_gb"] = resource_profile.gpu_budget_gb
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

        checkpoint_key = checkpoint_store.build_key(
            model_name=model_name,
            language=language,
            gap_sec=gap_sec,
            audio_path=input_path,
            detect_disfluencies=asr_align.FW_REFINE_DETECT_DISFLUENCIES,
        )
        try:
            from .fw_refine_backend import FwRefineModelPool
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: faster-whisper plus the patched CTranslate2 "
                'runtime. Install with `pip install -e ".[asr]"` and see '
                "tools/wt_refine_port/ct2-patches/README.md for the runtime."
            ) from exc
        model_pool = FwRefineModelPool(
            model_name, device=device, size=1, refine_sec=asr_align.REFINE_SEC
        )
        t0 = time.perf_counter()
        model_pool.warm()
        timing["whisper_load_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
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
                )
        timing["asr_align_sec"] = time.perf_counter() - t0
        align_meta["recovery"] = dict(recovery_stats)

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
        aligned_segments, disfluency_stats = word_starts.apply_disfluency_rules(
            aligned_segments,
            energy_track=energy_track,
        )
        aligned_segments, clamp_stats = word_starts.clamp_word_starts(
            aligned_segments,
            vad_intervals=segments,
            pause_hints=pause_hints,
        )
        align_meta["word_start_correction"] = {
            **disfluency_stats,
            **clamp_stats,
        }

        nonempty_segments = segment_ops.drop_empty_segments(aligned_segments)
        nonempty_segments, ghost_drops = segment_ops.drop_ghost_duplicate_segments(
            nonempty_segments
        )
        align_meta["ghost_duplicate_segments_dropped"] = ghost_drops
        for description in ghost_drops:
            current_reporter().warning(
                "ghost-duplicate-dropped",
                f"dropped ghost duplicate segment ({description})",
            )
        monotonic_segments = segment_ops.clamp_segment_overlaps(nonempty_segments)
        monotonic_segments = segment_ops.extend_zero_length_segments(
            monotonic_segments
        )
        # DP split of over-long whisper segments (docs/segmentation-split.md);
        # runs before energy annotation so pieces get their own energy.
        split_result_segments = segment_split.split_segments(
            monotonic_segments,
            segments,
            params=split_params,
        )
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
        energy_segments = annotate_segments_with_vad_energy(
            split_result_segments,
            energy_track,
        )

        # Second-model verification evidence (docs/asr-align.md): suspects
        # and coverage gaps get a Qwen3-ASR re-recognition, recorded as
        # fields for downstream deciders. Runs after the Whisper pool is
        # released so the referee's ~1.5 GB fits every GPU budget.
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
                referee = qwen_referee.QwenReferee(
                    device=device
                    if str(device).strip().lower().startswith("cuda")
                    else "cpu"
                )
                try:
                    energy_segments, verify_stats = (
                        qwen_referee.apply_verification(
                            energy_segments,
                            vad_intervals=segments,
                            audio_path=str(audio_source),
                            referee=referee,
                        )
                    )
                finally:
                    referee.close()
                timing["qwen_verify_sec"] = time.perf_counter() - t0
                align_meta["qwen_verify"] = verify_stats

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
                    "qwen_verify_sec",
                )
                if key in timing
            }
            | {"total_sec": f"{total:.3f}"},
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
                device=args.device,
                language=args.language,
                gap_sec=args.gap,
                gpu_budget_gb=args.gpu_budget_gb,
                vad_silero_assist=args.vad_silero_assist,
                qwen_verify=args.qwen_verify,
                split_length_scale=args.split_length_scale,
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
