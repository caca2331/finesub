"""Turn a hand-corrected word-level SRT into a labelled disfluency dataset.

Provenance: WT was run twice on the same clip, once plain and once with
``detect_disfluencies=True``. The word-level SRT of the disfluency run was then
corrected by hand:

* disfluency blocks that were genuinely filled pauses were **deleted**;
* blocks that had wrongly carved an onset off a neighbouring word were **kept**,
  which calibrates that word's true start;
* a few segment ends that had stopped early were extended, and one collapsed
  segment was spread back out.

The annotation is not a delete/keep flag. Every block spans a window in which
the following word's true onset lies somewhere, and the correction places it:

* at the block start -- the whole block is the word's onset (``word_onset``);
* inside the block -- part pause, part onset (``partial``), produced by deleting
  the block and moving the next word's start into it, or by trimming a kept block;
* at the block end -- the whole block is a filled pause (``filled_pause``).

So the gold value per block is ``onset`` (absolute) / ``onset_fraction`` (0..1),
which supports both "is this a real disfluency" and "where exactly does the word
begin".

A block counts as ``segment-boundary`` when it is the first or last word of a
production segment, or when the word after it opens one; the annotator's positions
of interest are those and ``after-gap``. Note that the annotator also observed that
mid-phrase blocks are often the PRECEDING word's tail rather than the following
word's onset; the convention still attaches them forward, and
``annotator_note`` records that this is unresolved.

Usage (paths are local, untracked reference artifacts):

    python -m tools.wt_refine_validation.build_disfluency_gold \
        out/reference/BV1cqLR6hEp3/BV1cqLR6hEp3-vocal_Subtitle_Export \
        --clip BV1cqLR6hEp3 -o tools/wt_refine_validation/disfluency_gold.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

DISFLUENCY = "[*]"
GAP_SEC = 0.05  # a pause this long before a block counts as "after-gap"


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        stamp = re.match(
            r"(\d+):(\d+):(\d+),(\d+) --> (\d+):(\d+):(\d+),(\d+)", lines[1])
        if not stamp:
            continue
        g = [int(x) for x in stamp.groups()]
        cues.append({
            "start": g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
            "end": g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
            "word": "\n".join(lines[2:]),
        })
    return cues


def flatten(stable: dict) -> list[dict]:
    return [
        {**word, "segment": si, "index_in_segment": wi,
         "segment_length": len(segment["words"])}
        for si, segment in enumerate(stable["segments"])
        for wi, word in enumerate(segment["words"])
    ]


def align(left: list[dict], right: list[dict]) -> dict[int, int]:
    """Map indices of `left` to `right` by word text."""
    matcher = difflib.SequenceMatcher(
        a=[w["word"] for w in left], b=[w["word"] for w in right], autojunk=False)
    pairs = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                pairs[i1 + offset] = j1 + offset
    return pairs


def classify(words: list[dict], index: int) -> str:
    """Where does the block sit relative to the production segmentation?"""
    block = words[index]
    following = words[index + 1]
    if (block["index_in_segment"] == 0
            or block["index_in_segment"] == block["segment_length"] - 1
            or following["index_in_segment"] == 0):
        return "segment-boundary"
    if index > 0 and (block["start"] - words[index - 1]["end"]) >= GAP_SEC:
        return "after-gap"
    return "mid-phrase"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--clip", required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.export_dir
    disfl_dir = base / "detect_disfluencies"
    stable = json.loads(
        (disfl_dir / f"{args.clip}-detect-disfluencies-stable.json").read_text("utf-8"))
    words = flatten(stable)
    disfl_srt = parse_srt(disfl_dir / f"{args.clip}-detect-disfluencies-raw-word.srt")
    fixed_srt = parse_srt(base / f"{args.clip}-fixed.srt")
    plain_srt = parse_srt(base / "original" / f"{args.clip}-original-raw-word.srt")

    if [w["word"] for w in words] != [w["word"] for w in disfl_srt]:
        raise SystemExit("stable JSON and the disfluency word SRT disagree on words")

    to_fixed = align(disfl_srt, fixed_srt)
    to_plain = align(disfl_srt, plain_srt)

    candidates = []
    for i, word in enumerate(words):
        if word["word"] != DISFLUENCY or i + 1 >= len(words):
            continue
        following = words[i + 1]
        block_start, block_end = word["start"], word["end"]

        if i in to_fixed:
            # block survived; the annotator may still have trimmed its front
            onset = fixed_srt[to_fixed[i]]["start"]
        else:
            # block deleted; the next word's start may have moved into its span
            j = to_fixed.get(i + 1)
            onset = fixed_srt[j]["start"] if j is not None else block_end
        onset = min(max(onset, block_start), block_end)
        span = block_end - block_start
        fraction = (onset - block_start) / span if span > 0 else 0.0
        label = ("word_onset" if fraction <= 0.01
                 else "filled_pause" if fraction >= 0.99 else "partial")

        plain_index = to_plain.get(i + 1)
        candidates.append({
            "start": round(block_start, 3),
            "end": round(block_end, 3),
            "duration": round(span, 3),
            "label": label,
            "position": classify(words, i),
            # the gold: where the following word actually begins
            "onset": round(onset, 3),
            "onset_fraction": round(fraction, 3),
            "preceding_gap": round(block_start - words[i - 1]["end"], 3) if i else None,
            "preceding_word": words[i - 1]["word"] if i else None,
            "next_word": following["word"],
            "next_word_reported_start": round(following["start"], 3),
            "next_word_start_without_disfluencies": (
                round(plain_srt[plain_index]["start"], 3)
                if plain_index is not None else None),
            "segment": following["segment"],
            "annotator_note": (
                "mid-phrase blocks may in fact be the preceding word's tail; the "
                "convention still attaches them to the following word"
                if label != "filled_pause" and classify(words, i) == "mid-phrase"
                else None),
        })

    corrections, collapse = [], []
    for i, j in sorted(to_fixed.items()):
        before, after = disfl_srt[i], fixed_srt[j]
        d_start = round(after["start"] - before["start"], 3)
        d_end = round(after["end"] - before["end"], 3)
        if d_start or d_end:
            corrections.append({
                "word": before["word"],
                "reported": [round(before["start"], 3), round(before["end"], 3)],
                "corrected": [round(after["start"], 3), round(after["end"], 3)],
                "delta_start": d_start,
                "delta_end": d_end,
                "kind": "late_start" if d_start < 0 else (
                    "early_end" if d_end > 0 else "other"),
            })
    matcher = difflib.SequenceMatcher(
        a=[w["word"] for w in disfl_srt], b=[w["word"] for w in fixed_srt],
        autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            collapse.append({
                "reported": [w["word"] for w in disfl_srt[i1:i2]],
                "corrected": [
                    {"word": w["word"], "start": round(w["start"], 3),
                     "end": round(w["end"], 3)} for w in fixed_srt[j1:j2]],
            })

    payload = {
        "clip": args.clip,
        "source": {
            "backend": "whisper-timestamped",
            "disfluency_run": f"{args.clip}-detect-disfluencies-*",
            "plain_run": f"{args.clip}-original-*",
            "hand_corrected": f"{args.clip}-fixed.srt",
            "segments": len(stable["segments"]),
            "words": len(words),
        },
        "conventions": {
            "gap_sec": GAP_SEC,
            "kept_block_belongs_to": "following word",
            "priority": ["segment-boundary", "after-gap", "mid-phrase"],
            "labels": {
                "word_onset": "whole block is the following word's onset",
                "partial": "onset lies inside the block",
                "filled_pause": "whole block is a pause"},
        },
        "disfluency_candidates": candidates,
        "timestamp_corrections": corrections,
        "collapse_repairs": collapse,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from collections import Counter
    tally = Counter(c["label"] for c in candidates)
    print(f"{len(candidates)} candidates -> {dict(tally)}; "
          f"{len(corrections)} timestamp fixes; {len(collapse)} collapse repairs")
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
