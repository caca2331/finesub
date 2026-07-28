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

import asr_align
import segment_split
from resource_profiles import (
    DEFAULT_GPU_BUDGET_GB,
    get_resource_profile,
    gpu_budget_choices,
)
import vad_energy


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
        help="GPU memory budget profile in GiB (default: 8).",
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


def load_and_detect_segments(
    input_path: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    float,
    dict[str, float],
    vad_energy.VadEnergyTrack,
]:
    timing: dict[str, float] = {}
    vad_params = vad_energy.vad_params()

    # Streamed VAD: loading, normalization and framing run block by block
    # (bit-identical to the old full-load path, RAM bounded by one block);
    # alignment later streams the audio back from disk too (AudioBlockLoader),
    # so no stage holds the whole recording resident.
    t0 = time.perf_counter()
    with torch.inference_mode():
        (
            raw_segments,
            vad_meta,
            audio_duration,
            energy_track,
        ) = vad_energy.run_vad_file(
            str(input_path),
            params=vad_params,
            timing=timing,
        )
    timing["vad_sec"] = time.perf_counter() - t0

    if energy_track.energy_mode == "weighted":
        vad_meta.setdefault("vad", {})["segment_energy"] = (
            vad_energy.segment_energy_metadata()
        )

    segments = asr_align.normalize_vad_segments(raw_segments, audio_duration)
    return raw_segments, segments, vad_meta, audio_duration, timing, energy_track


def run_vad_asr(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name: str = asr_align.DEFAULT_MODEL,
    device: str = "cuda",
    language: Optional[str] = None,
    gap_sec: float = asr_align.DEFAULT_GAP_SEC,
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB,
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
    device_for_usage = None
    model = None
    try:
        t_start = time.perf_counter()
        device = resolve_device(device, context="VAD-ASR")
        device_for_usage = device
        asr_align.reset_peak_gpu_memory_stats_for_run(device_for_usage)
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
                segments,
                vad_meta,
                audio_duration,
                timing,
                energy_track,
            ) = load_and_detect_segments(input_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load/prepare audio: {exc}") from exc

        if not raw_segments or not segments:
            write_aligned_json(output, [], vad_meta=vad_meta, align_meta=align_meta)
            print(f"Wrote {output}")
            return output

        t0 = time.perf_counter()
        model = whisper.load_model(model_name, device=device)
        timing["whisper_load_sec"] = time.perf_counter() - t0

        # Stream the alignment audio from disk in blocks instead of holding the
        # whole recording in RAM. Matches the standalone asr_align.main config
        # (600s core + 10s pad, no bandpass) so ASR input stays consistent.
        audio_loader = asr_align.AudioBlockLoader(
            str(input_path),
            target_sr=vad_energy.TARGET_SR,
            block_seconds=600.0,
            pad_seconds=10.0,
            preprocess=False,
        )

        t0 = time.perf_counter()
        aligned_segments = asr_align.align_segments(
            segments,
            None,
            vad_energy.TARGET_SR,
            model=model,
            gap_sec=gap_sec,
            language=language,
            audio_loader=audio_loader,
            checkpoint_path=asr_align.asr_checkpoint_path(output),
            checkpoint_key=asr_align.asr_checkpoint_key(
                model_name=model_name,
                language=language,
                gap_sec=gap_sec,
                audio_path=input_path,
            ),
        )
        timing["asr_align_sec"] = time.perf_counter() - t0

        nonempty_segments = asr_align.drop_empty_segments(aligned_segments)
        monotonic_segments = asr_align.clamp_segment_overlaps(nonempty_segments)
        monotonic_segments = asr_align.extend_zero_length_segments(monotonic_segments)
        # DP split of over-long whisper segments (docs/segment_split.md);
        # runs before energy annotation so pieces get their own energy.
        split_result_segments = segment_split.split_segments(
            monotonic_segments,
            segments,
        )
        energy_segments = annotate_segments_with_vad_energy(
            split_result_segments,
            energy_track,
        )
        output_segments = [asr_align.round_floats(seg) for seg in energy_segments]
        write_aligned_json(
            output,
            output_segments,
            vad_meta=vad_meta,
            align_meta=align_meta,
        )
        print(f"Wrote {output}")

        total = time.perf_counter() - t_start
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
        # Release the Whisper model so downstream stages (LLM) start with
        # a clean GPU.  Mirrors vocal_separation.py's cleanup pattern.
        if model is not None:
            try:
                del model
            except Exception:
                pass
        gc.collect()
        if device_for_usage is not None and device_for_usage.strip().lower() == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        asr_align.print_peak_resource_usage(device_for_usage, resource_profile)


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
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
