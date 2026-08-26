"""Line up two arms' raw SRTs on the timeline and print where they disagree."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def parse_srt(path: Path):
    blocks = []
    for chunk in re.split(r"\n\s*\n", path.read_text(encoding="utf-8").strip()):
        lines = [line for line in chunk.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        times = _TIME_RE.findall(lines[1])
        if len(times) != 2:
            continue

        def to_sec(item):
            h, m, s, ms = (int(part) for part in item)
            return h * 3600 + m * 60 + s + ms / 1000.0

        blocks.append((to_sec(times[0]), to_sec(times[1]), " ".join(lines[2:])))
    return blocks


def bucket(blocks, width=10.0, span=540.0):
    slots = {}
    for start, end, text in blocks:
        slots.setdefault(int(start // width), []).append(text)
    return slots


def main() -> int:
    a = parse_srt(Path(sys.argv[1]))
    b = parse_srt(Path(sys.argv[2]))
    sa, sb = bucket(a), bucket(b)
    keys = sorted(set(sa) | set(sb))
    silent_in_b = 0
    silent_in_a = 0
    for key in keys:
        left = "".join(sa.get(key, []))
        right = "".join(sb.get(key, []))
        if left == right:
            continue
        if left and not right:
            silent_in_b += 1
        if right and not left:
            silent_in_a += 1
        print(f"--- {key * 10:>4}s")
        print(f"  A: {left}")
        print(f"  B: {right}")
    print(f"\n10s buckets with speech in A but nothing in B: {silent_in_b}")
    print(f"10s buckets with speech in B but nothing in A: {silent_in_a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
