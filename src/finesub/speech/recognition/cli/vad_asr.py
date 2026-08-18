"""Console entry point for the VAD + ASR stage."""

from ..vad_asr_stage import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
