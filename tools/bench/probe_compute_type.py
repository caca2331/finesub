"""A8 -- is `int8_float16` worth anything on this GPU? Measure once, close it.

`docs/plans/crispasr-followups.md` -> A8: production hardcodes `float16` in two
places and the GPU side has never been measured. The CPU side was: `int8` beat
`float32` by only ~15%, nowhere near the 2-4x that gets quoted, and it changes
the output.

Two opposing priors make this un-guessable, which is why it gets measured
rather than argued:

* the decoder might be weight-bandwidth-bound, where int8 halves the traffic;
* but the P7 occupancy measurement says the decoder is *launch*-bound (42%
  median utilisation), where narrower weights buy nothing -- and the encoder is
  compute-bound with fp16 already sitting on the tensor cores, where int8 may
  well be **slower**.

Both models are loaded at once and the arms alternate within each round, with
the within-round order swapped every round and the two orders aggregated
*separately*, then combined geometrically (`discipline.paired_ratio`). Loading
all of A then all of B would compare a cold card against a warm one; always
calling A first would hand B a permanent second-caller advantage; and pooling
the two orders into one median would leave that advantage in the answer.

⚠ `int8_float16` is **not** output-preserving -- it quantises weights. This
probe measures speed only. If speed turned out to favour it, a quality
acceptance would still be owed; if it does not, the question closes with no
quality work needed at all.

    python -m tools.bench.probe_compute_type --audio out/<stem>/<stem>-vocal.ogg
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from tools.bench import discipline, gpu


def _load_audio(path: Path, seconds: float) -> np.ndarray:
    import soundfile as sf

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        mono = librosa.resample(
            mono, orig_sr=sample_rate, target_sr=16000, res_type="soxr_hq"
        )
    return mono[: int(16000 * seconds)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--rounds", type=int, default=6, help="must be even")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    # See `probe_wddm` for why this is not `torch.cuda.is_available()`.
    from finesub.speech.runtime.device import cuda_usable

    if not cuda_usable():
        print("FAIL: A8 is a GPU question and this machine has no usable GPU")
        return 2

    from finesub.speech.recognition.fw_refine_backend import (
        RefinedWhisperModel,
        _encoder_window,
    )

    audio = _load_audio(args.audio, args.seconds)
    print(f"idle baseline: {json.dumps(gpu.idle_baseline(1.5), default=str)}\n")

    models = {}
    for compute_type in ("float16", "int8_float16"):
        print(f"loading {compute_type} ...")
        models[compute_type] = RefinedWhisperModel(
            args.model, device="cuda", compute_type=compute_type, refine_sec=1.0
        )

    def encode_once(model):
        def run() -> object:
            # The cache would make the second call free and measure nothing.
            model._encoder_cache.clear()
            return model.encode(np.stack([_encoder_window(model, audio)]))

        return run

    def transcribe_once(model):
        def run() -> object:
            model._encoder_cache.clear()
            result = model.transcribe_wt(
                audio, language="ja", beam_size=1, temperature=0.0
            )
            discipline.refuse_empty(result.get("segments"), what="transcribe")
            return result

        return run

    report: dict[str, object] = {"model": args.model, "seconds": args.seconds}

    for name, factory in (("encode", encode_once), ("transcribe", transcribe_once)):
        # Warm both arms before the paired rounds; the first call of either
        # pays allocator and cuBLAS setup that is not what we are comparing.
        factory(models["float16"])()
        factory(models["int8_float16"])()
        paired = discipline.paired_ratio(
            factory(models["float16"]),
            factory(models["int8_float16"]),
            rounds=args.rounds,
            labels=("float16", "int8_float16"),
        )
        report[name] = paired
        f16 = statistics.median(paired["float16_ms"])
        i8 = statistics.median(paired["int8_float16_ms"])
        ratio = paired["ratio"]
        print()
        print(f"=== {name} ===")
        print(f"  float16       median {f16:8.1f} ms")
        print(f"  int8_float16  median {i8:8.1f} ms")
        print(
            f"  per-order medians (f16/int8): AB {paired['ratio_ab_median']:.3f}"
            f"  BA {paired['ratio_ba_median']:.3f}"
        )
        print(f"  order effect removed (k)     : {paired['order_effect']:.3f}")
        print(f"  ratio, order effect cancelled: {ratio:.3f}")
        verdict = (
            "int8_float16 FASTER" if ratio > 1.02
            else "int8_float16 SLOWER" if ratio < 0.98
            else "no difference"
        )
        print(f"  => {verdict}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
