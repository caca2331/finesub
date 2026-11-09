"""Subtitle lines for models that do not emit them: the production global DP over the model's word stream.

The DP (`finesub.speech.postprocessing.segmentation.split_segments`) runs unchanged, with production's VAD
intervals for the same audio. What a model contributes:

* words (standalone punctuation tokens folded into the word before them, as production's words carry it);
* seams: a speaker change starts a new input segment, i.e. it gets `whisper_segment_bonus` -- the role an
  ASR segment seam plays in production. Without speakers, a pause > `SEAM_PAUSE` does;
* optionally the text-side sentence-boundary model (`tools/bench/probe_sbd_split.py`, the P18 model):
  its `p_sbd` after each word is subtracted from that boundary's cost, `b -= lam * p_sbd`. That is owner
  2026-08-31's direction (the probability enters the DP as one more boundary signal). ⚠ `lam` is not
  calibrated; 1.0 makes a certain sentence end worth exactly what sentence punctuation is worth in `t_score`.

Arms (`--arm`):
    punct       model punctuation, no SBD
    sbd         model punctuation + SBD
    strip-sbd   punctuation removed, SBD only -- what SBD does for a model that emits none

Production signals a model arm can borrow (the "model as text source" experiment):

    --ends vad      a word's end runs on to the next word's start or its VAD interval's end, whichever is first
                    (mai2 gives every character 79 ms, so its phrase ends come early)
    --seams prod    input segments start where production's Whisper segments start (`whisper_segment_start`
                    in `-aligned.json`, mapped to the model word starting nearest within SEAM_SNAP), instead of
                    at the model's own speaker changes

MOSS has no word timing; its segments are cut into characters with evenly spread times (flagged in the
output as `approx_word_times`) -- enough for the DP's length terms, not for timing claims.

    python -m tools.ts_diar_explore.seg_arm <arm.json> <vad.json> <out.json> --arm punct|sbd|strip-sbd [--lam 1.0]
        [--ends model|vad] [--seams speaker|prod --aligned <-aligned.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

from .common import WT, save

sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT))
from finesub.speech.postprocessing import segmentation as seg  # noqa: E402

SEAM_PAUSE = 1.0
SEAM_SNAP = 0.3


def _is_punct(text: str) -> bool:
    return bool(text) and all(unicodedata.category(ch)[0] in "PZ" for ch in text)


def model_words(arm: dict) -> tuple[list[dict], bool]:
    """(words in DP shape, approximate?)."""
    if arm.get("words"):
        raw, approx = arm["words"], False
    else:
        raw, approx = [], True
        for s in arm["segments"]:
            chars = [ch for ch in s["text"] if not ch.isspace()]
            step = (s["end"] - s["start"]) / max(1, len(chars))
            raw += [{"start": s["start"] + i * step, "end": s["start"] + (i + 1) * step,
                     "speaker": s.get("speaker"), "text": ch} for i, ch in enumerate(chars)]
    words: list[dict] = []
    for w in sorted(raw, key=lambda w: w["start"]):
        text = w["text"].strip()
        if not text:
            continue
        if _is_punct(text) and words:
            words[-1]["word"] += text
            continue
        words.append({"start": w["start"], "end": max(w["end"], w["start"] + 0.01), "word": text,
                      "space_before": False, "speaker": w.get("speaker")})
    return words, approx


def strip(words: list[dict]) -> list[dict]:
    out = []
    for w in words:
        text = "".join(ch for ch in w["word"] if unicodedata.category(ch)[0] != "P")
        if text:
            out.append({**w, "word": text})
    return out


def vad_ends(words: list[dict], vad: list[dict]) -> list[dict]:
    """Extend (never shorten) each word's end to the next word's start or its VAD interval's end."""
    out = []
    for i, w in enumerate(words):
        inside = next((v for v in vad if v["start"] <= w["start"] < v["end"]), None)
        limit = min(words[i + 1]["start"] if i + 1 < len(words) else float("inf"), inside["end"] if inside else w["end"])
        out.append({**w, "end": max(w["end"], limit)})
    return out


def prod_seams(words: list[dict], aligned: dict) -> list[dict]:
    """Mark the model word starting nearest each production Whisper segment start (within SEAM_SNAP)."""
    starts = [x["start"] for s in aligned["segments"] for x in s.get("words", []) if x.get(seg.WHISPER_SEGMENT_WORD_TAG)]
    marked = set()
    for t in starts:
        k = min(range(len(words)), key=lambda i: abs(words[i]["start"] - t), default=None)
        if k is not None and abs(words[k]["start"] - t) <= SEAM_SNAP:
            marked.add(k)
    return [{**w, "seam": i in marked} for i, w in enumerate(words)]


def seams(words: list[dict], pause: float = SEAM_PAUSE) -> list[dict]:
    """Input segments: a new one at every explicit seam (`prod_seams`), else at every speaker change (or long
    pause when there are no speakers)."""
    explicit = any("seam" in w for w in words)
    has_speakers = any(w.get("speaker") for w in words)
    segs: list[dict] = []
    for w in words:
        new = (not segs or (w.get("seam") if explicit else
                            (has_speakers and w.get("speaker") != segs[-1]["speaker"])
                            or (not has_speakers and w["start"] - segs[-1]["words"][-1]["end"] > pause)))
        if new:
            segs.append({"speaker": w.get("speaker"), "words": []})
            w = {**w, seg.WHISPER_SEGMENT_WORD_TAG: True}
        segs[-1]["words"].append(w)
    for s in segs:
        s["start"], s["end"] = s["words"][0]["start"], s["words"][-1]["end"]
        s["text"] = "".join(w["word"] for w in s["words"])
    return segs


def attach_sbd(words: list[dict]) -> int:
    """`p_sbd_after` on every word; returns how many boundaries fell inside a subword (left at 0)."""
    from tools.bench.probe_sbd_split import SbdModel, map_boundaries
    probs = map_boundaries(words, list(range(len(words) - 1)), SbdModel())
    for k, w in enumerate(words):
        w["p_sbd_after"] = probs.get(k) or 0.0
    return sum(1 for v in probs.values() if v is None)


def split(words: list[dict], vad: list[dict], *, lam: float, pause: float = SEAM_PAUSE) -> list[dict]:
    original = seg.boundary_score

    def with_sbd(left, right, intervals, params):
        b = original(left, right, intervals, params)
        b.b -= lam * float(left.source.get("p_sbd_after", 0.0))
        return b

    seg.boundary_score = with_sbd
    try:
        return seg.split_segments(seams(words, pause), vad)
    finally:
        seg.boundary_score = original


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_json", type=Path)
    ap.add_argument("vad_json", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--arm", required=True, choices=["punct", "sbd", "strip-sbd"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--ends", default="model", choices=["model", "vad"])
    ap.add_argument("--seams", default="speaker", choices=["speaker", "prod"])
    ap.add_argument("--aligned", type=Path, help="production -aligned.json (for --seams prod)")
    a = ap.parse_args()
    arm = json.loads(a.arm_json.read_text(encoding="utf-8"))
    vad = json.loads(a.vad_json.read_text(encoding="utf-8"))["segments"]
    words, approx = model_words(arm)
    if a.arm == "strip-sbd":
        words = strip(words)
    if a.ends == "vad":
        words = vad_ends(words, vad)
    if a.seams == "prod":
        words = prod_seams(words, json.loads(a.aligned.read_text(encoding="utf-8")))
    inside = attach_sbd(words) if a.arm != "punct" else 0
    pieces = split(words, vad, lam=a.lam if a.arm != "punct" else 0.0)
    lines = [{"start": p["start"], "end": p["end"], "text": p.get("text", ""),
              "speaker": (p.get("words") or [{}])[0].get("speaker")} for p in pieces]
    save({"model": arm["model"], "seg_arm": a.arm, "lam": a.lam, "ends": a.ends, "seams": a.seams, "approx_word_times": approx,
          "sbd_inside_subword": inside, "segments": lines}, a.out)
    print(f"{a.arm_json.stem} {a.arm}: {len(words)} words -> {len(lines)} lines"
          f"{' (approx word times)' if approx else ''} -> {a.out}")


if __name__ == "__main__":
    main()
