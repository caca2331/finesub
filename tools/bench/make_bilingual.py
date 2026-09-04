"""P16/P8 -- build bilingual audio with a *known* language per span.

Both P8 and P16 are blocked on the same missing thing: **negatives**. Our corpus
is Japanese streams, so the language-vote path has almost nothing to arbitrate
(58 of 27122 segments disagree with their run's majority) and its thresholds
have never seen a case where the answer is "yes, that really is another
language". Reading the ASR's own output cannot supply that -- it is the thing
under test.

Concatenating real speech from two assets whose language is already known
*does* supply it: every span carries a ground-truth label that came from the
source asset, not from a detector.

⚠ **Synthetic evidence characterises behaviour and finds bugs. It must not set
a default or a threshold.** That rule is the project's own
(`crispasr-followups.md` -> 批次 F 评测纪律): an external project set an
estimator default from 5 synthetic samples and the real benchmark reversed the
conclusion (DER 11.4% vs 5.3%). The same caution applies here, and doubly so --
concatenation removes the conversational context a real code-switch would carry.

    python -m tools.bench.make_bilingual --seconds 12 --out tmp/bench/bilingual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAIN_OUT = Path(r"C:\Users\Carl\Documents\Carl\projects\asr-playground\out")
TARGET_SR = 16000


def _segments(stem: str) -> tuple[Path, list[dict]]:
    aligned = MAIN_OUT / stem / f"{stem}-aligned.json"
    vocal = MAIN_OUT / stem / f"{stem}-vocal.ogg"
    payload = json.loads(aligned.read_text(encoding="utf-8"))
    return vocal, payload["segments"]


def _read(path: Path, start: float, end: float) -> np.ndarray:
    import soundfile as sf

    info = sf.info(str(path))
    begin = max(0, int(start * info.samplerate))
    stop = min(int(info.frames), int(end * info.samplerate))
    if stop <= begin:
        return np.zeros(0, dtype=np.float32)
    data, _ = sf.read(str(path), start=begin, stop=stop, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if info.samplerate != TARGET_SR:
        import librosa

        mono = librosa.resample(
            mono, orig_sr=info.samplerate, target_sr=TARGET_SR, res_type="soxr_hq"
        )
    return mono


#: One slice of source audio, with where it came from. The source range is
#: carried so the generator can prove it never re-uses audio -- see `_take`.
Chunk = tuple[np.ndarray, str, float, float]


def _take_raw(path: Path, want_seconds: float, offset: float) -> tuple[list[Chunk], float]:
    """Fixed-length chunks from a bare audio file.

    For a language we have no aligned artifacts for there is no segmentation to
    follow, so the cut is by clock. The text is unknown, which is fine: the
    label under test is the *language*, and that comes from the source file.
    """

    chunks: list[Chunk] = []
    taken = 0.0
    step = 4.0
    while taken < want_seconds:
        begin, stop = offset + taken, offset + taken + step
        audio = _read(path, begin, stop)
        if audio.size == 0:
            break
        chunks.append((audio, "", begin, stop))
        taken += step
    return chunks, offset + taken


def _take(stem: str, want_seconds: float, offset: float) -> tuple[list[Chunk], float]:
    """Consecutive real segments from one asset, so prosody stays natural.

    Returns the chunks **and the source time to resume from**. Advancing by the
    summed *speech* duration would be wrong: segments have pauses between them,
    so the next call would start before the previous one ended and re-use audio
    that is already in the mix. That silently duplicates content -- an earlier
    version of this file did exactly that (a ja run had 27 spans but only 15
    distinct texts), which inflates any per-span rate computed from the result.
    """

    candidate = Path(stem)
    if candidate.suffix and candidate.exists():
        return _take_raw(candidate, want_seconds, offset)
    vocal, segments = _segments(stem)
    chunks: list[Chunk] = []
    total = 0.0
    reached = offset
    for segment in segments:
        start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
        if start < offset or end - start < 0.6:
            continue
        audio = _read(vocal, start, end)
        if audio.size == 0:
            continue
        chunks.append((audio, str(segment.get("text") or ""), start, end))
        total += end - start
        reached = end
        if total >= want_seconds:
            break
    return chunks, reached


def _overlapping_reuse(truth: list[dict]) -> int:
    """Count spans whose source range overlaps an earlier span of the same source.

    The mix must be made of distinct audio: a duplicated span is counted twice
    in every per-span rate derived from the result.
    """

    used: dict[str, list[tuple[float, float]]] = {}
    clashes = 0
    for span in truth:
        source = str(span["source"])
        begin, end = float(span["source_start"]), float(span["source_end"])
        for previous_begin, previous_end in used.get(source, []):
            if min(end, previous_end) - max(begin, previous_begin) > 1e-3:
                clashes += 1
                break
        used.setdefault(source, []).append((begin, end))
    return clashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ja-stem", default="BV1aP8E67EbF")
    parser.add_argument(
        "--en-stem",
        default="mt8g-cIgoqAy",
        help="asset stem, or a path to a bare audio file for a language we have no artifacts for",
    )
    parser.add_argument("--b-lang", default="en", help="ground-truth language label for side B")
    parser.add_argument(
        "--seconds",
        type=float,
        default=12.0,
        help="approximate length of each single-language run before switching",
    )
    parser.add_argument("--switches", type=int, default=12)
    parser.add_argument(
        "--offset",
        type=float,
        default=60.0,
        help="skip this far into each source before cutting (avoids intros)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import soundfile as sf

    args.out.mkdir(parents=True, exist_ok=True)
    pieces: list[np.ndarray] = []
    truth: list[dict] = []
    cursor = 0.0
    ja_offset = en_offset = args.offset

    for index in range(args.switches):
        language = "ja" if index % 2 == 0 else args.b_lang
        is_a = index % 2 == 0
        stem = args.ja_stem if is_a else args.en_stem
        offset = ja_offset if is_a else en_offset
        chunks, reached = _take(stem, args.seconds, offset)
        if not chunks:
            print(f"FAIL: no usable audio from {stem} at offset {offset:.1f}s")
            return 2
        for audio, text, src_start, src_end in chunks:
            pieces.append(audio)
            duration = len(audio) / TARGET_SR
            truth.append(
                {
                    "start": round(cursor, 3),
                    "end": round(cursor + duration, 3),
                    "language": language,
                    "source": stem,
                    "source_start": round(src_start, 3),
                    "source_end": round(src_end, 3),
                    "text": text,
                }
            )
            cursor += duration
        # A short pause so the VAD has a boundary to find, as it would in real
        # speech. Not silence-padding a language *into* its neighbour's window.
        gap = np.zeros(int(0.25 * TARGET_SR), dtype=np.float32)
        pieces.append(gap)
        cursor += 0.25
        # Resume past the last segment's *source end*, never by summed speech.
        if is_a:
            ja_offset = reached + 1.0
        else:
            en_offset = reached + 1.0

    audio = np.concatenate(pieces)
    wav = args.out / "bilingual.wav"
    sf.write(str(wav), audio, TARGET_SR)
    (args.out / "bilingual-truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # The manifest exists so a reported number can be re-derived. The label bug
    # this guards against was real: a ja+ja control was generated without
    # `--b-lang ja`, so half its truth said "en" and the committed scorer could
    # not reproduce the number the report quoted.
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "argv": vars(args) | {"out": str(args.out)},
                "spans": len(truth),
                "languages": sorted({t["language"] for t in truth}),
                "sources": sorted({t["source"] for t in truth}),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    overlaps = _overlapping_reuse(truth)
    if overlaps:
        print(f"FAIL: {overlaps} span(s) re-use source audio already in the mix")
        return 2

    ja = sum(t["end"] - t["start"] for t in truth if t["language"] == "ja")
    other = sum(t["end"] - t["start"] for t in truth if t["language"] != "ja")
    print(f"wrote {wav}  ({len(audio) / TARGET_SR:.1f}s)")
    print(
        f"  spans: {len(truth)}   ja {ja:.1f}s   {args.b_lang} {other:.1f}s"
        f"   switches {args.switches}"
    )
    print(f"  truth: {args.out / 'bilingual-truth.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
