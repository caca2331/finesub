"""VAD-energy + Whisper alignment stage for vocal audio."""

from __future__ import annotations

import argparse
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
from ..postprocessing import segmentation as segment_split
from . import sharding as wt_shard
from ..runtime.resources import (
    DEFAULT_GPU_BUDGET_GB,
    get_resource_profile,
    gpu_budget_choices,
)
from ..runtime.gpu_stage_gate import GPU_STAGE_GATE, GpuStageLease
from ..runtime.model_pool import WtModelPool
from ..runtime import stall_watchdog
from ..runtime.thread_budget import bounded_intra_op_threads
from ..preprocessing import energy as vad_energy
from ..preprocessing import vad as vad_detection


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
    parser.add_argument(
        "--wt-workers",
        type=int,
        default=None,
        help=(
            "[DEV/UNSAFE] Override WT shard workers for benchmarking (default: "
            "use the GPU profile). Production callers should not pass this; it "
            "may exceed the profile and changes aligned output. See "
            "docs/wt-parallelism.md."
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
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    base = input_path.with_suffix("")
    return base.with_name(f"{base.name}-aligned.json")


def resolve_device(requested_device: str, *, context: str = "VAD-ASR") -> str:
    device = str(requested_device or "cuda")
    if device.strip().lower() == "cuda" and not torch.cuda.is_available():
        print(
            f"Warning: CUDA requested for {context} but unavailable; falling back to CPU.",
            file=sys.stderr,
        )
        return "cpu"
    return device


def write_aligned_json(
    output_path: Path,
    segments: list[dict[str, object]],
    *,
    vad_meta: dict[str, object],
    align_meta: dict[str, object],
) -> None:
    payload = {
        "segments": segments,
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


def run_vad_asr(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name: str = asr_align.DEFAULT_MODEL,
    device: str = "cuda",
    language: Optional[str] = None,
    gap_sec: float = asr_align.DEFAULT_GAP_SEC,
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB,
    wt_workers: Optional[int] = None,
) -> Path:
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
    resource_profile = get_resource_profile(gpu_budget_gb)
    if wt_workers is not None:
        if int(wt_workers) < 1:
            raise ValueError("wt_workers must be at least 1")
        if int(wt_workers) != 1:
            print(
                "Warning: wt_workers is an unsafe development override "
                f"(requested={int(wt_workers)}, "
                f"profile_default={resource_profile.wt_instances}); "
                "it may exceed the GPU budget and change aligned output.",
                file=sys.stderr,
            )
    device_for_usage = None
    memory_sampler = None
    model_pool: WtModelPool | None = None
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
        align_meta["gpu_budget_gb"] = resource_profile.gpu_budget_gb
        align_meta["gpu_limit_gb"] = resource_profile.usable_gpu_gb
        align_meta["ram_budget_gb"] = resource_profile.ram_budget_gb
        align_meta["segment_split"] = segment_split.split_params_metadata()

        try:
            import whisper_timestamped as whisper
        except Exception as exc:
            raise RuntimeError(
                "Missing dependency: whisper-timestamped. Install with `pip install -e .`."
            ) from exc

        try:
            (
                raw_segments,
                vad_meta,
                audio_duration,
                timing,
                energy_track,
            ) = vad_detection.detect_segments(input_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load/prepare audio: {exc}") from exc

        segments = asr_align.normalize_vad_segments(raw_segments, audio_duration)
        if not raw_segments or not segments:
            timing["total_sec"] = time.perf_counter() - t_start
            align_meta["timing"] = {
                key: round(value, 3) for key, value in timing.items()
            }
            align_meta["wt"] = wt_shard.WtShardPlan(
                workers=0,
                total_vad_seconds=0.0,
                shards=[],
            ).metadata()
            write_aligned_json(output, [], vad_meta=vad_meta, align_meta=align_meta)
            print(f"Wrote {output}")
            return output

        gpu_stage_lease = GPU_STAGE_GATE.acquire(
            "wt",
            enabled=str(device).strip().lower().startswith("cuda"),
        )

        # Plan the shards before loading anything: the plan decides how many
        # models to build, and a short file must not pay for idle instances.
        plan = wt_shard.plan_wt_shards(
            asr_align.build_alignment_groups(segments, gap_sec=gap_sec),
            max_workers=max(1, int(wt_workers or resource_profile.wt_instances)),
        )
        align_meta["wt"] = plan.metadata()
        if plan.workers > 1:
            print(
                f"Info: WT sharding (workers={plan.workers}, "
                f"speech={plan.total_vad_seconds:.1f}s, "
                f"shards={[s.interval_count for s in plan.shards]})",
                file=sys.stderr,
            )

        # Stream the alignment audio from disk in blocks instead of holding the
        # whole recording in RAM. Matches the standalone asr_align.main config
        # (600s core + 10s pad, no bandpass) so ASR input stays consistent.
        def _make_audio_loader() -> asr_align.AudioBlockLoader:
            return asr_align.AudioBlockLoader(
                str(input_path),
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
        )
        model_pool = WtModelPool(
            whisper, model_name, device=device, size=plan.workers or 1
        )
        t0 = time.perf_counter()
        model_pool.warm()
        timing["whisper_load_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with bounded_intra_op_threads(plan.workers):
            aligned_segments = asr_align.align_segments_sharded(
                segments,
                None,
                vad_energy.TARGET_SR,
                plan=plan,
                model_pool=model_pool,
                gap_sec=gap_sec,
                language=language,
                audio_loader_factory=_make_audio_loader,
                aligned_output=output,
                checkpoint_key=checkpoint_key,
            )
        timing["asr_align_sec"] = time.perf_counter() - t0

        nonempty_segments = segment_ops.drop_empty_segments(aligned_segments)
        monotonic_segments = segment_ops.clamp_segment_overlaps(nonempty_segments)
        monotonic_segments = segment_ops.extend_zero_length_segments(
            monotonic_segments
        )
        # DP split of over-long whisper segments (docs/segment_split.md);
        # runs before energy annotation so pieces get their own energy.
        split_result_segments = segment_split.split_segments(
            monotonic_segments,
            segments,
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
            print(
                "Warning: synthesized one segment-span word for "
                f"{synthetic_word_segments} text-only ASR segment(s).",
                file=sys.stderr,
            )
        energy_segments = annotate_segments_with_vad_energy(
            split_result_segments,
            energy_track,
        )
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
        )
        print(f"Wrote {output}")

        print("Timing:")
        for key in (
            "loading_sec",
            "energy_sec",
            "noise_sec",
            "vad_sec",
            "whisper_load_sec",
            "asr_align_sec",
        ):
            if key in timing:
                print(f"  {key}: {timing[key]:.3f}")
        print(f"  total_sec: {total:.3f}")
        return output
    finally:
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
        run_vad_asr(
            args.input,
            output_path=args.output,
            model_name=args.model,
            device=args.device,
            language=args.language,
            gap_sec=args.gap,
            gpu_budget_gb=args.gpu_budget_gb,
            wt_workers=args.wt_workers,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
