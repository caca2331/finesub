"""Resident VRAM of one loaded Whisper pool, per model.

This is the number `lang_redecode.WHISPER_RESIDENT_GIB_BY_MODEL` holds and the
one `referee_device` subtracts from a tier's usable figure. It cannot come from
torch: CT2 allocates outside torch's CUDA counters, which is exactly why a
model missing from that table has to stay on the CPU rather than be estimated.

So the measurement is **device-wide used memory from the driver**, taken around a load
plus one real decode -- "what is actually occupied while the pool is leased",
not "how big is the checkpoint". One decode matters: loading alone leaves the
encoder/decoder scratch unallocated, and that scratch is part of what the
referee has to fit beside.

    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.probe_whisper_resident \\
        --audio out/ts-baseline-20260831-hq/BV1UBjq6fEgb/BV1UBjq6fEgb-vocal.ogg \\
        large-v3-turbo large-v3 TransWithAI/whisper-ja-1.5B-ct2

**Run it on an idle card** (bench-baselines P4): the reading is device-wide, so
anything else holding VRAM lands in the baseline. The script prints the
baseline and the other compute processes it can see, and one model per
subprocess is deliberate -- a second model in the same process would measure
the allocator's high-water mark rather than that model's residency.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BYTES_PER_GIB = 1 << 30
#: Seconds of audio fed to the one warm-up decode. One 30 s Whisper window is
#: the unit the encoder works in; a shorter clip would leave part of the
#: scratch unallocated and read low.
CLIP_SECONDS = 30.0


def _device_used_gib() -> float:
    """Whole-card used VRAM, from the driver.

    `torch.cuda.mem_get_info` rather than NVML: it is the same driver number
    production reads (`speech/runtime/device.free_vram_gib`), and it needs no
    dependency the ASR environment does not already have. Device-wide is the
    point -- CT2's allocations are invisible to torch's own counters.
    """

    import torch

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return float(total_bytes - free_bytes) / BYTES_PER_GIB


def _other_processes() -> list[dict[str, object]]:
    """Anything else holding VRAM, so a polluted reading reports itself."""

    import os

    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001 - reporting only
        return []
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        if int(parts[0]) == os.getpid():
            continue
        # Windows/WDDM reports per-process memory as `[N/A]`: the driver owns
        # the allocations, not the process. That is exactly why the reading
        # above is device-wide -- record the pid anyway, so a busy card still
        # shows up as busy.
        used = parts[1] if parts[1].isdigit() else None
        rows.append({"pid": int(parts[0]), "used_mb": int(used) if used else None})
    return rows


def measure(model_name: str, audio_path: Path) -> dict[str, object]:
    """Load `model_name`, decode one window, report the device-wide delta."""

    from finesub.speech.preprocessing import audio as audio_io
    from finesub.speech.recognition.fw_refine_backend import RefinedWhisperModel

    import torch

    torch.cuda.init()
    # Settle: a just-exited process can still be releasing.
    time.sleep(2.0)
    baseline = _device_used_gib()
    others = _other_processes()

    source_rate, _frames = audio_io.get_audio_info(str(audio_path))
    waveform, sample_rate = audio_io.load_audio_slice(
        str(audio_path), 0, int(CLIP_SECONDS * source_rate)
    )
    waveform = audio_io.to_mono(waveform)
    waveform, _sample_rate = audio_io.resample_if_needed(waveform, sample_rate, 16000)
    clip = audio_io.as_numpy_float32(waveform)

    started = time.perf_counter()
    model = RefinedWhisperModel(
        model_name, device="cuda", compute_type="float16", refine_sec=1.0
    )
    loaded = _device_used_gib()
    load_seconds = time.perf_counter() - started

    model.transcribe_wt(clip, beam_size=1, language="ja")
    after_decode = _device_used_gib()

    return {
        "model": model_name,
        "baseline_gib": round(baseline, 3),
        "after_load_gib": round(loaded, 3),
        "after_decode_gib": round(after_decode, 3),
        "resident_gib": round(after_decode - baseline, 3),
        "load_only_gib": round(loaded - baseline, 3),
        "load_seconds": round(load_seconds, 1),
        "other_compute_processes": others,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="Whisper model names or HF repo ids")
    parser.add_argument("--audio", required=True, help="any speech file, >= 30 s")
    parser.add_argument("--out", default="", help="write the rows as JSON here")
    parser.add_argument(
        "--one",
        default="",
        help="internal: measure exactly this model and print one JSON row",
    )
    args = parser.parse_args(argv)

    if args.one:
        print(json.dumps(measure(args.one, Path(args.audio))))
        return 0

    rows = []
    for model_name in args.models:
        # One subprocess per model: see the module docstring.
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.bench.probe_whisper_resident",
                "--audio",
                args.audio,
                "--one",
                model_name,
                model_name,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if process.returncode != 0:
            print(f"{model_name}: FAILED\n{process.stderr[-2000:]}", file=sys.stderr)
            continue
        row = json.loads(process.stdout.strip().splitlines()[-1])
        rows.append(row)
        print(
            f"{row['model']:44} resident {row['resident_gib']:6.2f} GiB "
            f"(load only {row['load_only_gib']:5.2f}, baseline {row['baseline_gib']:5.2f}, "
            f"load {row['load_seconds']}s)"
        )
        others = row["other_compute_processes"]
        if others:
            # On Windows the per-process figure is `[N/A]`, so the count and
            # the baseline are what there is to report.
            print(f"     {len(others)} other compute processes on the card")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
