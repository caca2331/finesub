"""Two-way content diff between a long-window Qwen transcript and the baseline transcript.

Answers the recall question directly — what does each side have that the other does not —
instead of collapsing everything into one CER number, which hides direction.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from .common import load_aligned, normalize_ja


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-aligned", required=True)
    ap.add_argument("--arm", required=True, help="run_asr.py output (any window mode)")
    ap.add_argument("--min-run", type=int, default=6, help="report insert/delete runs at least this long")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = normalize_ja("".join(s.get("text", "") for s in load_aligned(args.baseline_aligned)["segments"]))
    arm_json = json.loads(Path(args.arm).read_text(encoding="utf-8"))
    arm = normalize_ja("".join(s["text"] for s in arm_json["segments"]))

    matcher = difflib.SequenceMatcher(None, base, arm, autojunk=False)
    only_base: list[str] = []
    only_arm: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("delete", "replace") and i2 - i1 >= args.min_run:
            only_base.append(base[i1:i2])
        if tag in ("insert", "replace") and j2 - j1 >= args.min_run:
            only_arm.append(arm[j1:j2])

    lines = [
        "# Content recall diff\n",
        f"- baseline chars: {len(base)} · arm chars: {len(arm)} · similarity {matcher.ratio():.3f}",
        f"- runs >= {args.min_run} chars only\n",
        f"## Present in baseline, missing/replaced in arm ({len(only_base)} runs)\n",
    ]
    lines += [f"- {run}" for run in only_base]
    lines += [f"\n## Present in arm, missing/replaced in baseline ({len(only_arm)} runs)\n"]
    lines += [f"- {run}" for run in only_arm]

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}: base-only {len(only_base)} runs, arm-only {len(only_arm)} runs")


if __name__ == "__main__":
    main()
