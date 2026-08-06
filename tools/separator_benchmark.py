"""Isolated BS-Roformer performance and waveform-similarity benchmark.

Run one variant per process so model loading, allocator state, and peak-memory
statistics do not leak between variants. This is a development tool, not a
production entrypoint.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from asr_playground.speech.preprocessing import accel
from asr_playground.speech.preprocessing import separation as vocal_separation
from asr_playground.speech.preprocessing import separator_aoti

# The variants below differ from production only in what they turn on, so the
# shared pieces -- package rewriting, constant injection, the per-axis SDPA
# choice -- are imported rather than re-implemented. A private copy here would
# be free to drift from what production actually runs, which is the one thing a
# benchmark must not do.


class _TimedModel(torch.nn.Module):
    """Measure synchronized model forward calls without changing model outputs."""

    def __init__(self, inner: torch.nn.Module, state: dict[str, Any]) -> None:
        super().__init__()
        self.inner = inner
        self._timing_state = state

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("inner"), name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.inner(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        tensor_inputs = [value for value in args if isinstance(value, torch.Tensor)]
        call = {
            "phase": self._timing_state["phase"],
            "elapsed_sec": elapsed,
            "thread": threading.current_thread().name,
            "inputs": [
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
                for value in tensor_inputs
            ],
        }
        with self._timing_state["lock"]:
            self._timing_state["calls"].append(call)
        return result


def _configure_experimental_features(args: argparse.Namespace) -> dict[str, Any]:
    original_build = vocal_separation._build_separator
    original_warmup = vocal_separation._warm_up_shared_roformer
    timing_state: dict[str, Any] = {
        "phase": "setup",
        "calls": [],
        "lock": threading.Lock(),
        "compile_wrapper_sec": 0.0,
        "artifact_load_and_inject_sec": 0.0,
        "compiled_module_count": 0,
        "warmup_first_wall_sec": None,
        "warmup_reuse_wall_sec": None,
        "probe_overhead_sec": 0.0,
    }

    def build(output_dir: str, output_format: str, batch_size: int):
        separator = original_build(output_dir, output_format, batch_size)
        model_instance = separator.model_instance
        if args.defer_per_file_cache_clear:
            model_instance.clear_gpu_cache = lambda: None
        if args.axis_sdpa:
            accel._install_axis_attention(model_instance.model_run)
        if args.aoti_transformer_dir is not None:
            manifest_path = args.aoti_transformer_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("weights_serialized") is not False:
                raise RuntimeError("AOTI manifest does not describe weightless packages")
            if manifest.get("torch") != torch.__version__:
                raise RuntimeError(
                    f"AOTI Torch mismatch: {manifest.get('torch')} != {torch.__version__}"
                )

            load_started = time.perf_counter()
            packages = manifest["packages"]
            package_bytes = {
                name: (args.aoti_transformer_dir / package["file"]).read_bytes()
                for name, package in packages.items()
            }
            aoti_temp_dir = tempfile.TemporaryDirectory(prefix="separator_aoti_")
            timing_state["_aoti_temp_dir"] = aoti_temp_dir
            model_run = model_instance.model_run

            def install(name: str, target: torch.nn.Module, suffix: str) -> None:
                # The 2.9 loader extracts to a fixed directory keyed on the
                # archive's top-level prefix, so each runner needs its own.
                runner_package = Path(aoti_temp_dir.name) / f"{name}_{suffix}.pt2"
                separator_aoti._write_package_with_prefix(
                    package_bytes[name],
                    runner_package,
                )
                compiled = torch._inductor.aoti_load_package(runner_package)
                compiled.load_constants(
                    separator_aoti._constant_map(compiled, target),
                    check_full_update=False,
                    user_managed=True,
                )
                target.forward = compiled
                timing_state["compiled_module_count"] += 1

            for name, package in packages.items():
                if package.get("kind", "transformer") == "transformer":
                    axis = package.get("axis", name)
                    index = 0 if axis == "time" else 1
                    for block_index, block in enumerate(model_run.layers):
                        transformer = block[index]
                        separator_aoti._populate_rotary_cache(
                            transformer,
                            int(package["input_shape"][1]),
                        )
                        install(name, transformer, f"{block_index:02d}")
                else:
                    install(name, model_run.get_submodule(package["module_path"]), "0")
            torch.cuda.synchronize()
            timing_state["artifact_load_and_inject_sec"] = (
                time.perf_counter() - load_started
            )
            model_instance.model_run = _TimedModel(
                model_instance.model_run,
                timing_state,
            )
        if args.torch_compile:
            # Windows hosts without MSVC can still compile the CUDA graph. Keep
            # symbolic shape guards in Python instead of JIT-building a tiny
            # C++ guard DLL with cl.exe.
            torch._dynamo.config.enable_cpp_symbolic_shape_guards = False
            compile_started = time.perf_counter()
            if args.compile_scope == "full":
                model_instance.model_run = torch.compile(
                    model_instance.model_run,
                    mode=args.compile_mode,
                )
                timing_state["compiled_module_count"] = 1
            else:
                model_run = model_instance.model_run
                for transformer_block in model_run.layers:
                    for index, transformer in enumerate(transformer_block):
                        transformer_block[index] = torch.compile(
                            transformer,
                            mode=args.compile_mode,
                        )
                        timing_state["compiled_module_count"] += 1
                if args.compile_scope == "all":
                    # Same target set as the AOTI default build, so the two
                    # compiled paths are compared at equal scope.
                    model_run.band_split = torch.compile(
                        model_run.band_split,
                        mode=args.compile_mode,
                    )
                    model_run.mask_estimators[0] = torch.compile(
                        model_run.mask_estimators[0],
                        mode=args.compile_mode,
                    )
                    timing_state["compiled_module_count"] += 2
            timing_state["compile_wrapper_sec"] += (
                time.perf_counter() - compile_started
            )
            model_instance.model_run = _TimedModel(
                model_instance.model_run,
                timing_state,
            )
        if args.inference_mode:
            separator_class = separator.__class__
            if not hasattr(separator_class, "_benchmark_original_separate"):
                separator_class._benchmark_original_separate = separator_class.separate

                def separate(self: Any, input_path: str, output_names: dict):
                    with torch.inference_mode():
                        return self._benchmark_original_separate(
                            input_path,
                            output_names,
                        )

                separator_class.separate = separate
        return separator

    def warmup(model_instance: Any, *, use_amp: bool) -> None:
        timing_state["phase"] = "warmup_first"
        started = time.perf_counter()
        original_warmup(
            model_instance,
            use_amp=bool(use_amp and args.amp_warmup),
        )
        timing_state["warmup_first_wall_sec"] = time.perf_counter() - started
        if (
            args.torch_compile or args.aoti_transformer_dir is not None
        ) and args.probe_compile_timing:
            timing_state["phase"] = "warmup_reuse"
            started = time.perf_counter()
            original_warmup(
                model_instance,
                use_amp=bool(use_amp and args.amp_warmup),
            )
            timing_state["warmup_reuse_wall_sec"] = time.perf_counter() - started
            timing_state["probe_overhead_sec"] = timing_state[
                "warmup_reuse_wall_sec"
            ]
        timing_state["phase"] = "separation"

    vocal_separation._build_separator = build
    vocal_separation._warm_up_shared_roformer = warmup
    return timing_state


def _summarize_compile_timing(state: dict[str, Any]) -> dict[str, Any]:
    calls = list(state["calls"])

    def phase_calls(phase: str) -> list[dict[str, Any]]:
        return [call for call in calls if call["phase"] == phase]

    def duration_summary(items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not items:
            return None
        durations = [float(item["elapsed_sec"]) for item in items]
        return {
            "count": len(durations),
            "total_sec": sum(durations),
            "first_sec": durations[0],
            "median_sec": statistics.median(durations),
            "min_sec": min(durations),
            "max_sec": max(durations),
        }

    warmup_first = phase_calls("warmup_first")
    warmup_reuse = phase_calls("warmup_reuse")
    separation = phase_calls("separation")
    separation_reuse = separation[1:]
    warmup_first_sec = warmup_first[0]["elapsed_sec"] if warmup_first else None
    warmup_reuse_sec = warmup_reuse[0]["elapsed_sec"] if warmup_reuse else None
    separation_first_sec = separation[0]["elapsed_sec"] if separation else None
    separation_reuse_sec = (
        statistics.median(call["elapsed_sec"] for call in separation_reuse)
        if separation_reuse
        else None
    )
    input_signatures: list[list[dict[str, Any]]] = []
    for call in calls:
        if call["inputs"] not in input_signatures:
            input_signatures.append(call["inputs"])

    return {
        "compile_wrapper_sec": state["compile_wrapper_sec"],
        "artifact_load_and_inject_sec": state["artifact_load_and_inject_sec"],
        "compiled_module_count": state["compiled_module_count"],
        "warmup_first_wall_sec": state["warmup_first_wall_sec"],
        "warmup_reuse_wall_sec": state["warmup_reuse_wall_sec"],
        "probe_overhead_sec": state["probe_overhead_sec"],
        "warmup_first_forward_sec": warmup_first_sec,
        "warmup_reused_forward_sec": warmup_reuse_sec,
        "warmup_compile_or_restore_estimate_sec": (
            max(0.0, warmup_first_sec - warmup_reuse_sec)
            if warmup_first_sec is not None and warmup_reuse_sec is not None
            else None
        ),
        "separation_first_forward_sec": separation_first_sec,
        "separation_reused_forward_median_sec": separation_reuse_sec,
        "separation_first_compile_estimate_sec": (
            max(0.0, separation_first_sec - separation_reuse_sec)
            if separation_first_sec is not None and separation_reuse_sec is not None
            else None
        ),
        "phase_forward_summaries": {
            "warmup_first": duration_summary(warmup_first),
            "warmup_reuse": duration_summary(warmup_reuse),
            "separation": duration_summary(separation),
            "separation_after_first": duration_summary(separation_reuse),
        },
        "forward_input_signatures": input_signatures,
    }


def _waveform_similarity(reference: Path, candidate: Path) -> dict[str, Any]:
    chunk_frames = 262_144
    ref_power = 0.0
    candidate_power = 0.0
    error_power = 0.0
    dot = 0.0
    absolute_error = 0.0
    max_absolute_error = 0.0
    sample_count = 0

    with sf.SoundFile(reference) as ref_file, sf.SoundFile(candidate) as candidate_file:
        if ref_file.samplerate != candidate_file.samplerate:
            raise ValueError("Sample-rate mismatch between reference and candidate")
        if ref_file.channels != candidate_file.channels:
            raise ValueError("Channel-count mismatch between reference and candidate")
        if len(ref_file) != len(candidate_file):
            raise ValueError("Frame-count mismatch between reference and candidate")

        while True:
            ref = ref_file.read(chunk_frames, dtype="float64", always_2d=True)
            candidate_chunk = candidate_file.read(
                chunk_frames, dtype="float64", always_2d=True
            )
            if ref.size == 0:
                break
            error = candidate_chunk - ref
            ref_power += float(np.sum(ref * ref))
            candidate_power += float(np.sum(candidate_chunk * candidate_chunk))
            error_power += float(np.sum(error * error))
            dot += float(np.sum(ref * candidate_chunk))
            absolute_error += float(np.sum(np.abs(error)))
            max_absolute_error = max(
                max_absolute_error,
                float(np.max(np.abs(error))),
            )
            sample_count += int(ref.size)

        eps = np.finfo(np.float64).tiny
        cosine = dot / math.sqrt(max(ref_power * candidate_power, eps))
        projected_power = (dot * dot) / max(ref_power, eps)
        residual_power = max(candidate_power - projected_power, eps)
        return {
            "sample_rate": ref_file.samplerate,
            "channels": ref_file.channels,
            "frames": len(ref_file),
            "cosine_similarity": cosine,
            "mae": absolute_error / max(sample_count, 1),
            "rmse": math.sqrt(error_power / max(sample_count, 1)),
            "max_abs_error": max_absolute_error,
            "snr_db": 10.0 * math.log10(max(ref_power, eps) / max(error_power, eps)),
            "si_sdr_db": 10.0 * math.log10(projected_power / residual_power),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("fp32", "amp"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpu-budget-gb", type=int, default=4)
    parser.add_argument("--block-seconds", type=float, default=600.0)
    parser.add_argument("--pad-seconds", type=float, default=10.0)
    parser.add_argument(
        "--defer-per-file-cache-clear",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--inference-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--axis-sdpa",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp-warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--aoti-transformer-dir",
        type=Path,
        help="Load weightless regional AOTI packages generated by separator_aoti.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument(
        "--compile-scope",
        choices=("full", "transformers", "all"),
        default="full",
        help=(
            "all matches the AOTI default target set: the 24 Transformers plus "
            "the band-wise band_split and mask_estimator."
        ),
    )
    parser.add_argument(
        "--probe-compile-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run one extra warmup forward to measure same-process graph reuse; "
            "the extra probe is excluded from elapsed_sec."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.torch_compile and args.aoti_transformer_dir is not None:
        raise ValueError("--torch-compile and --aoti-transformer-dir are exclusive")
    if args.aoti_transformer_dir is not None and args.mode != "amp":
        raise ValueError("The regional AOTI packages require --mode amp")
    if args.aoti_transformer_dir is not None and args.axis_sdpa:
        raise ValueError("AOTI packages already contain their attention implementation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    timing_state = _configure_experimental_features(args)

    started = time.perf_counter()
    metadata: dict[str, Any] = {}
    vocal_separation.run_vocal_separation(
        args.input,
        output_path=args.output,
        block_seconds=args.block_seconds,
        pad_seconds=args.pad_seconds,
        gpu_budget_gb=args.gpu_budget_gb,
        use_amp=args.mode == "amp",
        metadata_sink=metadata,
    )
    elapsed_raw = time.perf_counter() - started
    compile_timing = _summarize_compile_timing(timing_state)
    elapsed = elapsed_raw - float(timing_state["probe_overhead_sec"])
    compile_or_restore_parts = [
        compile_timing["compile_wrapper_sec"],
        compile_timing["artifact_load_and_inject_sec"],
        compile_timing["warmup_compile_or_restore_estimate_sec"],
        compile_timing["separation_first_compile_estimate_sec"],
    ]
    if all(part is not None for part in compile_or_restore_parts):
        compile_or_restore_total = sum(compile_or_restore_parts)
        compile_timing["compile_or_restore_total_estimate_sec"] = (
            compile_or_restore_total
        )
        compile_timing["runtime_excluding_compile_or_restore_estimate_sec"] = max(
            0.0,
            elapsed - compile_or_restore_total,
        )

    payload: dict[str, Any] = {
        "mode": args.mode,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "elapsed_sec": elapsed,
        "elapsed_sec_raw": elapsed_raw,
        "separator": metadata,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "features": {
            "defer_per_file_cache_clear": args.defer_per_file_cache_clear,
            "inference_mode": args.inference_mode,
            "axis_sdpa": args.axis_sdpa,
            "amp_warmup": args.amp_warmup,
            "torch_compile": args.torch_compile,
            "aoti_transformer_dir": (
                str(args.aoti_transformer_dir.resolve())
                if args.aoti_transformer_dir is not None
                else None
            ),
            "compile_mode": args.compile_mode if args.torch_compile else None,
            "compile_scope": args.compile_scope if args.torch_compile else None,
            "probe_compile_timing": args.probe_compile_timing,
        },
    }
    if args.torch_compile or args.aoti_transformer_dir is not None:
        payload["compile_timing"] = compile_timing
    if torch.cuda.is_available():
        payload["gpu"] = torch.cuda.get_device_name()
        payload["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        payload["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if args.reference is not None:
        payload["similarity"] = _waveform_similarity(args.reference, args.output)

    args.result.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
