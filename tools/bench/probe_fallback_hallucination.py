"""What does the deferred VAD auto-fallback actually produce on residue?

P1 shipped the failover predicate as warn-only. Turning it into an automatic
fallback (chunk the file at a fixed length and decode everything) needs the
cost of getting it *wrong*: our input is a separated vocal track, so on a
recording that really is quiet the fallback hands Whisper separator residue --
the material it hallucinates on hardest (`docs/plans/crispasr-followups.md` P1).

The fallback is simulated by **substituting the stage's VAD prefix**: the stage
reads its intervals from `<stem>-vad.json`, so writing fixed-length chunks over
the whole file there makes production decode exactly what the fallback would,
through the production path.

⚠ Two details are load-bearing, and getting either wrong silently turns this
into a tautology (it did once, see `bench-baselines.md` 17.3):

* `raw_segments` **and** `segments` both have to be replaced -- the failover
  predicate reads the first, the decoder reads the second;
* the energy `.npz` must keep **the filename the payload's `energy_track.arrays`
  names**. Rename it and `read_vad_prefix` fails, returns None, and the stage
  quietly recomputes the real VAD and overwrites your prefix.

    python -m tools.bench.probe_fallback_hallucination prepare RUNDIR CHUNK_SEC
    # then run the stage on the vocal track with --vad-prefix <printed path>
    python -m tools.bench.probe_fallback_hallucination score RUNDIR
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

#: A transcript this short is punctuation or a stray particle, not a claim.
MIN_HALLUCINATION_CHARS = 2


def _stem(run_dir: Path) -> str:
    hits = sorted(run_dir.glob("*-vad.json"))
    if not hits:
        raise SystemExit(f"FAIL: no <stem>-vad.json in {run_dir}")
    return hits[0].name[: -len("-vad.json")]


def prepare(run_dir: Path, chunk_sec: float) -> int:
    stem = _stem(run_dir)
    payload = json.loads(
        (run_dir / f"{stem}-vad.json").read_text(encoding="utf-8")
    )
    duration = float(payload["audio_duration"])
    chunks = []
    start = 0.0
    while start < duration:
        chunks.append({"start": start, "end": min(duration, start + chunk_sec)})
        start += chunk_sec
    payload["segments"] = chunks
    payload["raw_segments"] = list(chunks)

    # Its own directory, not beside the original: if the prefix turns out to be
    # unusable the stage recomputes *and writes the result back*, which in a
    # shared directory would silently overwrite the run's real VAD artifacts.
    target_dir = run_dir / "fallback"
    target_dir.mkdir(exist_ok=True)
    target = target_dir / "vad.json"
    target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # The payload names its own sidecar; it has to keep exactly that name.
    arrays = str(payload["energy_track"]["arrays"])
    source_npz = run_dir / arrays
    if not source_npz.exists():
        print(f"FAIL: {source_npz} is missing; the prefix would be unusable")
        return 2
    shutil.copy2(source_npz, target_dir / arrays)

    vocal = next(iter(sorted(run_dir.glob(f"{stem}-vocal.*"))), None)
    if vocal is None:
        print(f"FAIL: no {stem}-vocal.* in {run_dir}")
        return 2
    print(f"{stem}: {len(chunks)} chunks of {chunk_sec:.0f}s over {duration:.0f}s")
    print(f"  --vad-prefix {target}")
    print(f"  input        {vocal}")
    return 0


_CJK_OR_LETTER = re.compile(r"[\w぀-ヿ一-鿿]")


def score(run_dir: Path) -> int:
    aligned = run_dir / "fallback" / "aligned.json"
    if not aligned.exists():
        print(f"FAIL: {aligned} missing; run the stage first")
        return 2
    payload = json.loads(aligned.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    texts = [
        str(s.get("text") or "").strip()
        for s in segments
        if len(_CJK_OR_LETTER.findall(str(s.get("text") or "")))
        >= MIN_HALLUCINATION_CHARS
    ]
    duration = float(
        (payload.get("metadata", {}).get("asr_align", {}).get("audio_coverage") or {})
        .get("audio_sec")
        or 0.0
    )
    print(f"run             {run_dir.name}")
    print(f"audio           {duration / 60:.1f} min")
    print(f"segments        {len(segments)}")
    print(f"with text       {len(texts)}")
    if duration > 0:
        print(f"rate            {len(texts) / (duration / 60):.2f} per minute of residue")
    for text in texts[:25]:
        print(f"  {text[:110]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("run", type=Path)
    p.add_argument("chunk_sec", type=float, nargs="?", default=30.0)
    s = sub.add_parser("score")
    s.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        return prepare(args.run, args.chunk_sec)
    return score(args.run)


if __name__ == "__main__":
    raise SystemExit(main())
