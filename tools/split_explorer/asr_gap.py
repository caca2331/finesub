"""Run real ASR with an experimental inter-interval gap policy.

This tool deliberately keeps the experiment outside ``src/``.  It temporarily
overrides the two gap-policy hooks used by ``asr_align``, runs the normal
alignment implementation with a real Whisper model, and restores production
state before returning.  It never replays or remaps results from another run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import sys
import time
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import asr_align  # noqa: E402
from utils import TARGET_SR  # noqa: E402


AdaptiveGap = tuple[float, float, float]


def adaptive_silence_seconds(
    original_gap_sec: float,
    *,
    base_sec: float,
    factor: float,
    cap_sec: float,
) -> float:
    """Return ``min(base + factor * original_gap, cap)`` in seconds."""

    gap = max(0.0, float(original_gap_sec))
    return min(float(base_sec) + float(factor) * gap, float(cap_sec))


def _validate_policy(real_gap_max_sec: float, adaptive_gap: AdaptiveGap | None) -> None:
    if real_gap_max_sec < 0.0:
        raise ValueError("real-gap-max must be non-negative")
    if adaptive_gap is None:
        return
    base_sec, factor, cap_sec = adaptive_gap
    if min(base_sec, factor, cap_sec) < 0.0:
        raise ValueError("adaptive-gap values must be non-negative")
    if base_sec > cap_sec:
        raise ValueError("adaptive-gap BASE must not exceed CAP")


@contextmanager
def temporary_gap_policy(
    *,
    real_gap_max_sec: float,
    adaptive_gap: AdaptiveGap | None,
) -> Iterator[None]:
    """Install an experiment policy for one in-process alignment run."""

    _validate_policy(real_gap_max_sec, adaptive_gap)
    original_real_gap_max = asr_align.GAP_KEEP_REAL_MAX_SEC
    original_inserted_gap_parts = asr_align.inserted_gap_parts
    asr_align.GAP_KEEP_REAL_MAX_SEC = float(real_gap_max_sec)

    if adaptive_gap is not None:
        base_sec, factor, cap_sec = adaptive_gap

        def inserted_gap_parts(left, right, *, silence_sec: float):
            original_gap = max(
                0.0,
                float(right["start"]) - float(left["end"]),
            )
            real_sec = min(original_gap, asr_align.GAP_KEEP_REAL_MAX_SEC)
            synthetic_sec = adaptive_silence_seconds(
                original_gap,
                base_sec=base_sec,
                factor=factor,
                cap_sec=cap_sec,
            )
            return real_sec, synthetic_sec

        asr_align.inserted_gap_parts = inserted_gap_parts

    try:
        yield
    finally:
        asr_align.GAP_KEEP_REAL_MAX_SEC = original_real_gap_max
        asr_align.inserted_gap_parts = original_inserted_gap_parts


def _policy_metadata(
    *,
    real_gap_max_sec: float,
    fixed_gap_sec: float | None,
    adaptive_gap: AdaptiveGap | None,
) -> dict[str, object]:
    if adaptive_gap is None:
        synthetic: dict[str, object] = {
            "kind": "fixed",
            "seconds": fixed_gap_sec,
        }
    else:
        synthetic = {
            "kind": "adaptive",
            "base_sec": adaptive_gap[0],
            "factor": adaptive_gap[1],
            "cap_sec": adaptive_gap[2],
            "formula": "min(base_sec + factor * original_gap_sec, cap_sec)",
        }
    return {
        "real_gap_max_sec": real_gap_max_sec,
        "synthetic_silence": synthetic,
        "real_asr": True,
    }


def run(
    input_path: Path,
    audio_path: Path,
    output_path: Path,
    *,
    model_name: str,
    device: str,
    language: str | None,
    fixed_gap_sec: float | None,
    adaptive_gap: AdaptiveGap | None,
    real_gap_max_sec: float,
) -> None:
    """Run one real-ASR experiment and write an aligned JSON artifact."""

    if not input_path.exists():
        raise FileNotFoundError(f"VAD input not found: {input_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")
    _validate_policy(real_gap_max_sec, adaptive_gap)
    if fixed_gap_sec is not None and fixed_gap_sec < 0.0:
        raise ValueError("gap must be non-negative")
    if (fixed_gap_sec is None) == (adaptive_gap is None):
        raise ValueError("choose exactly one of fixed_gap_sec or adaptive_gap")

    import torch
    if device.strip().lower() == "cuda" and not torch.cuda.is_available():
        print(
            "Warning: CUDA requested for ASR alignment but unavailable; falling back to CPU.",
            file=sys.stderr,
        )
        device = "cpu"

    data = json.loads(input_path.read_text(encoding="utf-8"))
    audio_loader = asr_align.AudioBlockLoader(
        str(audio_path),
        target_sr=TARGET_SR,
        block_seconds=600.0,
        pad_seconds=10.0,
        preprocess=False,
    )
    intervals = asr_align.normalize_vad_segments(
        data.get("segments") or [],
        audio_loader.duration,
    )
    tail_silence_sec = (
        float(fixed_gap_sec)
        if adaptive_gap is None
        else adaptive_silence_seconds(
            0.0,
            base_sec=adaptive_gap[0],
            factor=adaptive_gap[1],
            cap_sec=adaptive_gap[2],
        )
    )

    t0 = time.perf_counter()
    with temporary_gap_policy(
        real_gap_max_sec=real_gap_max_sec,
        adaptive_gap=adaptive_gap,
    ):
        align_meta = asr_align.asr_align_metadata(
            model=model_name,
            device=device,
            language=language,
            gap_sec=tail_silence_sec,
        )
        if intervals:
            import whisper_timestamped as whisper

            model = whisper.load_model(model_name, device=device)
            aligned = asr_align.align_segments(
                intervals,
                None,
                TARGET_SR,
                model=model,
                gap_sec=tail_silence_sec,
                language=language,
                audio_loader=audio_loader,
            )
        else:
            aligned = []

    cleaned = asr_align.drop_empty_segments(aligned)
    output_segments = [asr_align.round_floats(segment) for segment in cleaned]
    align_meta["split_explorer_gap_experiment"] = _policy_metadata(
        real_gap_max_sec=real_gap_max_sec,
        fixed_gap_sec=fixed_gap_sec,
        adaptive_gap=adaptive_gap,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "segments": output_segments,
                "metadata": asr_align.merge_metadata(
                    data.get("metadata", {}),
                    align_meta,
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Wrote {output_path} (segments={len(output_segments)}, "
        f"elapsed={time.perf_counter() - t0:.3f}s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="VAD JSON/cache")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=asr_align.DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--real-gap-max",
        type=float,
        default=asr_align.GAP_KEEP_REAL_MAX_SEC,
        help="maximum seconds of original gap audio retained",
    )
    gap_group = parser.add_mutually_exclusive_group(required=True)
    gap_group.add_argument("--gap", type=float, help="fixed synthetic silence")
    gap_group.add_argument(
        "--adaptive-gap",
        type=float,
        nargs=3,
        metavar=("BASE", "FACTOR", "CAP"),
        help="synthetic silence: min(BASE + FACTOR * original_gap, CAP)",
    )
    args = parser.parse_args()
    run(
        args.input.expanduser().resolve(),
        args.audio.expanduser().resolve(),
        args.output.expanduser().resolve(),
        model_name=args.model,
        device=args.device,
        language=args.language,
        fixed_gap_sec=args.gap,
        adaptive_gap=tuple(args.adaptive_gap) if args.adaptive_gap else None,
        real_gap_max_sec=args.real_gap_max,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
