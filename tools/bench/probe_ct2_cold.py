"""A3.3 -- how much does a cold CUDA JIT cache cost the first CT2 model?

`docs/plans/crispasr-followups.md` -> A3.3: measuring this once permanently closes
the question "should we compile CTranslate2 natively for sm_120?". If the
wheel ships no sm_120 cubin, every kernel is JIT-compiled from PTX on first
use, and the cost lands on the first model construction plus the first encode
of every run whose driver cache was cleared.

The measurement has to isolate JIT from everything else that is also slow the
first time (file reads, allocator growth, cuBLAS handle creation). It does that
by running the *same* child process three times:

  1. cold  -- `CUDA_CACHE_PATH` points at an empty directory
  2. warm  -- the same directory, now populated by run 1
  3. native-- the machine's own cache, i.e. what production actually sees

`cold - warm` is the JIT cost, because everything else is identical between
them. If that difference is small, the wheel already carries native code for
this architecture and the question is closed.

    python -m tools.bench.probe_ct2_cold --audio ../asr-playground/assets/harvard.wav
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _child(audio: Path, model: str, encodes: int) -> int:
    """One measured process. Prints a single JSON line on stdout."""

    import numpy as np
    import soundfile as sf

    started_import = time.perf_counter()
    from finesub.speech.recognition.fw_refine_backend import (
        RefinedWhisperModel,
        _encoder_window,
    )

    import_s = time.perf_counter() - started_import

    data, sample_rate = sf.read(str(audio), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=16000, res_type="soxr_hq")
    # One encoder window is 30 s; anything longer would be trimmed anyway.
    mono = mono[: 16000 * 30]

    started = time.perf_counter()
    model_object = RefinedWhisperModel(
        model, device="cuda", compute_type="float16", refine_sec=1.0
    )
    construct_s = time.perf_counter() - started

    features = _encoder_window(model_object, mono)
    batch = np.stack([features])

    encode_times: list[float] = []
    for _ in range(encodes):
        started = time.perf_counter()
        model_object.encode(batch)
        encode_times.append(time.perf_counter() - started)

    print(
        json.dumps(
            {
                "import_s": import_s,
                "construct_s": construct_s,
                "encode_s": encode_times,
                "audio_seconds": len(mono) / 16000.0,
                "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH", "<default>"),
            }
        )
    )
    return 0


def _run_child(cache_dir: Path | None, args: argparse.Namespace) -> dict[str, object]:
    env = dict(os.environ)
    if cache_dir is not None:
        env["CUDA_CACHE_PATH"] = str(cache_dir)
    else:
        env.pop("CUDA_CACHE_PATH", None)
    # A cold JIT cache only stays cold if the cache is also allowed to be
    # large enough to hold the result; the driver default is small.
    env.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.bench.probe_ct2_cold",
            "--child",
            "--audio",
            str(args.audio),
            "--model",
            args.model,
            "--encodes",
            str(args.encodes),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    if completed.returncode != 0:
        raise SystemExit(
            "FAIL: child exited "
            f"{completed.returncode} -- a failed run is not a timing\n"
            f"{completed.stderr[-4000:]}"
        )
    payload = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not payload:
        raise SystemExit(f"FAIL: child produced no measurement\n{completed.stdout[-2000:]}")
    return json.loads(payload[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--encodes", type=int, default=5)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.child:
        return _child(args.audio, args.model, args.encodes)

    if not args.audio.exists():
        raise SystemExit(f"FAIL: audio not found: {args.audio}")

    results: dict[str, object] = {"model": args.model, "audio": str(args.audio)}
    with tempfile.TemporaryDirectory(prefix="ct2-jit-") as scratch:
        cache = Path(scratch)
        print("run 1/3  cold   (empty CUDA_CACHE_PATH) ...")
        results["cold"] = _run_child(cache, args)
        print("run 2/3  warm   (same cache, now populated) ...")
        results["warm"] = _run_child(cache, args)
    print("run 3/3  native (the machine's own cache) ...")
    results["native"] = _run_child(None, args)

    cold, warm, native = results["cold"], results["warm"], results["native"]
    jit_s = cold["construct_s"] - warm["construct_s"]
    first_encode_delta = cold["encode_s"][0] - warm["encode_s"][0]
    # The question A3.3 exists to close is not "is the first run slow" but "do
    # we pay for JIT on every encode". Those have different answers and only
    # the second one would justify building a wheel, so they are reported
    # separately rather than folded into one number.
    def steady(block: dict[str, object]) -> float:
        tail = block["encode_s"][1:]
        return min(tail) if tail else float("nan")

    steady_ratio = steady(cold) / steady(warm) if steady(warm) else float("nan")

    print("\n" + "=" * 78)
    print("A3.3 VERDICT -- CT2 cold-start JIT")
    print("=" * 78)
    for name in ("cold", "warm", "native"):
        block = results[name]
        encodes = ", ".join(f"{value * 1000:.1f}" for value in block["encode_s"])
        print(
            f"  {name:<7} import {block['import_s']:6.2f} s | "
            f"construct {block['construct_s']:6.2f} s | encodes(ms) {encodes}"
        )
    print(f"\n  one-time JIT on model construction : {jit_s:+.2f} s")
    print(f"  one-time JIT on the first encode   : {first_encode_delta * 1000:+.1f} ms")
    print(f"  steady-state encode cold/warm      : {steady_ratio:.3f}x")

    recurring = abs(steady_ratio - 1.0) > 0.05
    one_time = first_encode_delta > 1.0 or jit_s > 1.0
    if recurring:
        verdict = (
            "JIT'd kernels run SLOWER in steady state -- native compilation for "
            "this arch would buy throughput, not just startup"
        )
    elif one_time:
        verdict = (
            "the wheel ships no cubin for this arch, but JIT'd PTX runs at full "
            "speed: native compilation would buy a ONE-TIME startup saving only. "
            "Price it against driver-cache sizing, which buys the same thing for free."
        )
    else:
        verdict = "no measurable JIT cost -- native code is already present"
    print(f"  => {verdict}")
    results["verdict"] = {
        "jit_construct_s": jit_s,
        "jit_first_encode_s": first_encode_delta,
        "steady_state_ratio": steady_ratio,
        "cost_is_recurring": recurring,
        "summary": verdict,
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
