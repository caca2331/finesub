"""Does the run-level language audit fire where it should, on real audio?

The unit tests pin the predicate against a fake referee. What they cannot show
is whether the *real* referee, on *real* clips sampled the way production
samples them, reaches the conclusion the predicate needs. This probe closes
that gap on existing artifacts -- no ASR rerun.

Two arms, because a guard is only worth what its two error rates are worth:

* ``--as-is``      the whisper side is what the run actually decided.
                   On material the run got right this must stay quiet.
* ``--force-lang`` the whisper side is overwritten with one language for every
                   group. That is exactly the shape of the failure the audit
                   exists for -- a run uniformly labelled with the wrong
                   language -- and it is reachable from a correct artifact,
                   which a real end-to-end failure is not.

Usage:

    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.probe_lang_audit \\
        tmp/bench/ja-only --force-lang en

⚠ The `;` form of PYTHONPATH does NOT work under Git Bash on this machine --
it silently resolves `finesub` to the MAIN checkout's editable install, so a
worktree's changes vanish without any error. Use `:`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.speech.recognition import lang_audit, lang_redecode  # noqa: E402


def _observations(aligned: dict, *, force_lang: Optional[str]) -> List[lang_audit.Observation]:
    """One observation per decoded segment carrying a language.

    Production observes one span per *group*; an artifact does not record group
    boundaries, so this uses segments instead. The unit is different, the
    predicate is not -- and the sample is capped the same way, so the audit
    still buys `MAX_CLIPS` clips either way.
    """

    out: List[lang_audit.Observation] = []
    for segment in aligned.get("segments") or []:
        language = force_lang or str(segment.get("lang") or "").strip()
        if not language or language == "None":
            continue
        try:
            start = float(segment.get("start"))
            end = float(segment.get("end"))
        except (TypeError, ValueError):
            continue
        if end - start < lang_audit.MIN_CLIP_SEC:
            continue
        out.append(
            lang_audit.Observation(
                start, min(end, start + lang_audit.MAX_CLIP_SEC), language
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="directory holding aligned.json + audio")
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--aligned", type=Path, default=None)
    parser.add_argument(
        "--force-lang",
        default=None,
        help="pretend every group was labelled this language",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    audio = args.audio or (args.run / "bilingual.wav")
    if not audio.exists():
        print(f"REFUSED: no audio at {audio}", flush=True)
        return 2
    aligned_path = args.aligned or (args.run / "aligned.json")
    aligned = json.loads(aligned_path.read_text(encoding="utf-8"))

    observations = _observations(aligned, force_lang=args.force_lang)
    if not observations:
        print("REFUSED: no segment carried a language", flush=True)
        return 2

    from finesub.speech.verification import qwen_referee

    redecoder = lang_redecode.LangRedecoder(
        qwen_referee.QwenReferee(device=args.device), str(audio)
    )
    redecoder._observations = observations
    verdict = redecoder.run_audit()

    print(f"run           {args.run}")
    print(f"whisper side  {args.force_lang or 'as decoded'}")
    for key, value in verdict.items():
        print(f"  {key:18} {value}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "run": str(args.run),
                    "force_lang": args.force_lang,
                    "observations": len(observations),
                    "verdict": verdict,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
