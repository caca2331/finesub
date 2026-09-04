"""What does one referee clip actually cost, and what drives it?

The language audit's first measurement came out at 52s for eight clips, which
is two orders of magnitude off the "0.61s per suspect" figure recorded for
`--qwen-verify` (`bench-baselines.md` 4.4). Two candidate drivers -- clip
length (encoder work, prefill) and generated token count (decode work) -- and
no way to tell them apart from the aggregate, so: sweep both.

    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.probe_referee_cost \\
        tmp/bench/ja-only/bilingual.wav

Discipline (docs/bench-baselines.md, P4): the first measurement of each cell is
discarded as warm-up, absolute ms sit beside any ratio, and a non-zero exit is
a FAIL rather than a number.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.speech.verification import qwen_referee  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"FAIL: no audio at {args.audio}")
        return 2

    reader = qwen_referee._SpanReader(str(args.audio))
    referee = qwen_referee.QwenReferee(device=args.device)
    referee.warm()

    rows = []
    for clip_sec in (2.0, 5.0, 10.0, 20.0):
        clip = reader.read(30.0, 30.0 + clip_sec)
        if len(clip) < int(clip_sec * qwen_referee.TARGET_SR * 0.9):
            print(f"FAIL: could not read a {clip_sec}s clip")
            return 2
        for tokens in (16, 48, 128, 256):
            samples = []
            for index in range(args.repeat + 1):
                begin = time.perf_counter()
                out = referee.transcribe_batch([clip], max_new_tokens=tokens)
                elapsed = time.perf_counter() - begin
                if index:  # discard the first: warm-up
                    samples.append(elapsed)
            text, language = out[0]
            rows.append(
                {
                    "clip_sec": clip_sec,
                    "max_new_tokens": tokens,
                    "median_ms": round(1000 * statistics.median(samples), 1),
                    "language": language,
                    "chars": len(text),
                }
            )
            print(
                f"clip {clip_sec:>4.1f}s  tokens {tokens:>3}  "
                f"{rows[-1]['median_ms']:>8.1f} ms  lang={language}  "
                f"chars={len(text)}"
            )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"audio": str(args.audio), "rows": rows}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
