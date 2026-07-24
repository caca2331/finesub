"""CLI wrapper around audio-separator for vocal extraction."""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from resource_profiles import (
    DEFAULT_GPU_BUDGET_GB,
    get_resource_profile,
    gpu_budget_choices,
)
from utils import (
    get_audio_info,
    load_audio_slice,
    print_peak_resource_usage,
    reset_peak_gpu_memory_stats_for_run,
)

MODEL_NAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
BATCH_SIZE = get_resource_profile(DEFAULT_GPU_BUDGET_GB).vocal_separation_batch_size
CACHE_DIR = str(Path.home() / ".cache" / "audio-separator")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate vocals from audio.")
    parser.add_argument("input", help="Path to input audio.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output vocals file (default: <input>-vocal.ogg).",
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
        help="GPU memory budget profile in GiB (default: 8).",
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


def output_format_for(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "ogg"


def _build_separator(output_dir: str, output_format: str, batch_size: int) -> Separator:
    try:
        from audio_separator.separator import Separator
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "audio-separator is required for vocal separation. Install with `pip install -e .`."
        ) from exc

    separator = Separator(
        output_dir=output_dir,
        output_format=output_format,
        output_single_stem="Vocals",
        model_file_dir=CACHE_DIR,
        mdxc_params={"batch_size": batch_size},
        mdx_params={"batch_size": batch_size},
    )
    separator.load_model(model_filename=MODEL_NAME)
    if hasattr(separator, "mdx_batch_size"):
        separator.mdx_batch_size = batch_size
    elif hasattr(separator, "mdxc_batch_size"):
        separator.mdxc_batch_size = batch_size
    elif hasattr(separator, "vr_batch_size"):
        separator.vr_batch_size = batch_size
    return separator


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


def _find_output_file(output_files, stem: str, output_dir: Path) -> Path:
    candidates = _collect_output_paths(output_files)
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


def run_vocal_separation(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    block_seconds: float = 600.0,
    pad_seconds: float = 10.0,
    gpu_budget_gb: int = DEFAULT_GPU_BUDGET_GB,
    batch_size: Optional[int] = None,
) -> Path:
    resource_profile = get_resource_profile(gpu_budget_gb)
    selected_batch_size = (
        resource_profile.vocal_separation_batch_size
        if batch_size is None
        else int(batch_size)
    )
    if selected_batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    device_for_usage: Optional[str] = "cuda" if torch.cuda.is_available() else None
    out_file: Optional[sf.SoundFile] = None
    separator = None
    reset_peak_gpu_memory_stats_for_run(device_for_usage)
    try:
        if device_for_usage is None:
            print(
                "Warning: CUDA is the default vocal separation device but is unavailable; falling back to CPU.",
                file=sys.stderr,
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
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_format = output_format_for(output_path)
        if block_seconds <= 0:
            separator = _build_separator(
                str(output_path.parent),
                output_format,
                selected_batch_size,
            )

            output_names = {"Vocals": output_path.stem}
            output_files = separator.separate(str(input_path), output_names)
            if not output_files:
                raise SystemExit("No output files were produced by audio-separator.")
            print(f"Wrote {output_path}")
            return output_path

        src_sr, total_frames = get_audio_info(str(input_path))
        if src_sr <= 0 or total_frames <= 0:
            raise SystemExit(f"Unable to read audio info for {input_path}")
        block_samples = int(round(block_seconds * src_sr))
        pad_samples = int(round(pad_seconds * src_sr))
        block_samples = max(1, block_samples)

        chunk_frames = 262144

        with tempfile.TemporaryDirectory(prefix="vocal_blocks_") as tmpdir:
            separator = _build_separator(tmpdir, output_format, selected_batch_size)

            block_start = 0
            block_index = 0
            while block_start < total_frames:
                read_start = max(0, block_start - pad_samples)
                read_end = min(total_frames, block_start + block_samples + pad_samples)
                read_frames = max(0, read_end - read_start)
                if read_frames <= 0:
                    break

                waveform, read_sr = load_audio_slice(str(input_path), read_start, read_frames)
                if read_sr <= 0:
                    raise SystemExit(f"Invalid sample rate while loading: {input_path}")
                block_input = Path(tmpdir) / f"block_{block_index:05d}.wav"
                sf.write(
                    str(block_input),
                    waveform.detach().cpu().numpy().T,
                    read_sr,
                    subtype="PCM_16",
                )

                output_stem = f"{output_path.stem}-block{block_index:05d}"
                output_names = {"Vocals": output_stem}
                output_files = separator.separate(str(block_input), output_names)
                if not output_files:
                    # Drop the old model before loading a fresh one so the two
                    # never co-reside on the GPU (setting to None keeps the
                    # finally-clause guard valid if the rebuild raises).
                    separator = None
                    gc.collect()
                    if device_for_usage is not None:
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    separator = _build_separator(tmpdir, output_format, selected_batch_size)
                    output_files = separator.separate(str(block_input), output_names)
                block_output = _find_output_file(output_files, output_stem, Path(tmpdir))
                if not block_output.exists():
                    raise SystemExit("No output files were produced by audio-separator.")

                trim_left = 0.0 if block_start == 0 else pad_seconds
                trim_right = 0.0 if read_end >= total_frames else pad_seconds

                with sf.SoundFile(block_output, mode="r") as in_f:
                    if out_file is None:
                        out_file = sf.SoundFile(
                            str(output_path),
                            mode="w",
                            samplerate=in_f.samplerate,
                            channels=in_f.channels,
                            format=output_format.upper(),
                        )
                    elif in_f.samplerate != out_file.samplerate or in_f.channels != out_file.channels:
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

                try:
                    block_input.unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    block_output.unlink(missing_ok=True)
                except Exception:
                    pass

                del waveform
                gc.collect()

                block_index += 1
                block_start += block_samples

        if out_file is None:
            raise SystemExit("No output files were produced by audio-separator.")
        out_file.close()
        out_file = None
        print(f"Wrote {output_path}")
        return output_path
    finally:
        if out_file is not None:
            try:
                out_file.close()
            except Exception:
                pass
        # Release the separator model before the (same-process) pipeline loads
        # Whisper, so the two GPU models never need to co-reside within the
        # 8GB budget. Output is already fully written; this is memory-only.
        if separator is not None:
            try:
                del separator
            except Exception:
                pass
        gc.collect()
        if device_for_usage is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print_peak_resource_usage(device_for_usage, resource_profile)


def main() -> int:
    args = parse_args()
    try:
        run_vocal_separation(
            args.input,
            output_path=args.output,
            block_seconds=args.block_seconds,
            pad_seconds=args.pad_seconds,
            gpu_budget_gb=args.gpu_budget_gb,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

