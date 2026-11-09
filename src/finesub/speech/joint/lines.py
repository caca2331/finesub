"""A cloud transcript's units -> subtitle lines through production's DP.

The rules are the 2026-10 experiment's (arms M1 for MAI2, G1s for 35t, S5b / S6b for MOSS Pro; readouts and
reasons in `docs/joint-asr.md`), which the readers preferred:

* words in DP shape: punctuation folded into the word before it, `space_before` between Latin words, cut at
  the audio's end;
* a derived VAD: the words' own spans, gaps shorter than VAD_GAP closed (no VAD runs on a cloud task);
* input segments ("seams", the role an ASR segment plays in production's DP): at every speaker change and
  every pause over SEAM_PAUSE (35t, MAI2), or every MOSS Pro line after merging its short segments;
* 35t only: abutting words pulled GRID_GAP apart (its 100 ms grid makes them abut, and the DP reads a zero
  pause as connected speech) and the sentence-boundary model's evidence taken off each cut (`sbd.py`).

Turns: 35t and MOSS Pro each run the DP once per turn -- a 35t turn is one
speaker's run in output order, a MOSS Pro turn is a line after the merge -- with a VAD derived from that turn's
own words, so no line takes words from two turns and two people talking at once get lines that overlap rather
than interleave. MAI2's words never overlap: one DP over all of them by start, a speaker change only a seam
(arm M1), so a MAI2 line may hold two people's words. Each line gets `line_speaker`, the speaker holding most of
its words' time; lines are in order by start, same start by `line_speaker`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

from ..postprocessing import segmentation as seg
from ..recognition import segments as segment_ops
from ...subtitles import line_groups
from ...subtitles.metrics import weighted_char_count
from .model import UNKNOWN_SPEAKER

#: A pause longer than this starts a new input segment (35t, MAI2; arm M1's, owner 2026-10-30).
SEAM_PAUSE = 1.0
#: Word gaps shorter than this are one speech interval of the derived VAD (all three; the experiment's c).
VAD_GAP = 0.8
#: 35t's abutting words are pulled this far apart (owner 2026-10-30).
GRID_GAP = 0.01
#: MOSS Pro: a short segment joins the same speaker's neighbour across a gap shorter than this (S0's merge).
MERGE_GAP = 0.8
#: MAI2's lines start this much earlier and end this much later (`widened`; owner 2026-11-03): its lines start late
#: (in zh, en and ja alike) and end early -- every CJK character it times lasts the same 79 ms.
MAI2_EDGE_PAD = 0.1

#: The field this module adds to every line.
LINE_SPEAKER = line_groups.LINE_SPEAKER
_WORD_SCRATCH = ("speaker", "seam", seg.SENTENCE_END_BONUS_KEY)

_LATIN = re.compile(r"[0-9A-Za-zÀ-ɏ]")
_TOKEN = re.compile(r"\S+")
_PIECE = re.compile(r"[A-Za-z0-9]+(?:['.\-][A-Za-z0-9]+)*|.", re.S)


def spaced(prev: str, word: str) -> bool:
    """`space_before` for a model word: it starts with a Latin letter or digit, and so does the end of the word
    before it once trailing punctuation is set aside ("what?" + "I" -> "what? I")."""
    core = prev.rstrip("".join(ch for ch in prev if unicodedata.category(ch)[0] in "PS"))
    return bool(core) and bool(word) and bool(_LATIN.match(core[-1])) and bool(_LATIN.match(word[0]))


def dp_words(units: Iterable[dict], *, duration: float, keep_order: bool = False) -> list[dict]:
    """Units (`start`, `end`, `text`, `speaker`) as DP words, cut at `duration`: punctuation folded into the
    word before it, `space_before` between Latin words. Sorted by start unless `keep_order`."""
    raw = list(units) if keep_order else sorted(units, key=lambda w: w["start"])
    words: list[dict] = []
    for w in raw:
        text = (w.get("text") or "").strip()
        if not text or w["start"] >= duration:
            continue
        if all(unicodedata.category(ch)[0] in "PZ" for ch in text) and words:
            words[-1]["word"] += text
            continue
        prev = words[-1]["word"] if words else ""
        end = min(max(w["end"], w["start"] + 0.01), duration)
        words.append({"start": round(w["start"], 3), "end": round(end, 3), "word": text,
                      "space_before": spaced(prev, text), "speaker": w.get("speaker")})
    return words


def ungrid(words: Sequence[dict]) -> list[dict]:
    """Abutting words pulled GRID_GAP apart (the earlier word's end moves). A step back in time is left alone."""
    out = [dict(w) for w in words]
    for a, b in zip(out, out[1:]):
        if b["start"] >= a["start"] and b["start"] - a["end"] <= 0 and a["end"] - a["start"] > 2 * GRID_GAP:
            a["end"] = round(min(a["end"], b["start"]) - GRID_GAP, 3)
    return out


def derived_vad(spans: Iterable[dict], gap: float = VAD_GAP) -> list[dict]:
    """Speech intervals from spans sorted by start: gaps shorter than `gap` closed."""
    out: list[dict] = []
    for w in spans:
        if out and w["start"] - out[-1]["end"] < gap:
            out[-1]["end"] = max(out[-1]["end"], w["end"])
        else:
            out.append({"start": w["start"], "end": w["end"]})
    return out


def mark_seams(words: list[dict], *, pause: float = SEAM_PAUSE) -> list[dict]:
    """`seam` on the first word, every speaker change and every pause over `pause`."""
    return [{**w, "seam": i == 0 or w.get("speaker") != words[i - 1].get("speaker")
             or w["start"] - words[i - 1]["end"] > pause} for i, w in enumerate(words)]


def _input_segments(words: Sequence[dict]) -> list[dict]:
    segs: list[dict] = []
    for w in words:
        if not segs or w.get("seam"):
            segs.append({"speaker": w.get("speaker"), "words": []})
            w = {**w, seg.WHISPER_SEGMENT_WORD_TAG: True}
        segs[-1]["words"].append(w)
    for s in segs:
        s["start"], s["end"] = s["words"][0]["start"], s["words"][-1]["end"]
        s["text"] = "".join(w["word"] for w in s["words"])
    return segs


def line_speaker(words: Sequence[dict]) -> str:
    """The speaker holding most of the words' time (the first to reach it on a tie)."""
    held: dict[str, float] = {}
    for w in words:
        key = w.get("speaker") or UNKNOWN_SPEAKER
        held[key] = held.get(key, 0.0) + max(0.0, float(w["end"]) - float(w["start"]))
    return max(held, key=held.get) if held else UNKNOWN_SPEAKER


def speaker_runs(words: Sequence[dict]) -> list[list[dict]]:
    """Consecutive words of one speaker, in the order given (a word without a speaker is `UNKNOWN_SPEAKER`'s)."""
    runs: list[list[dict]] = []
    for w in words:
        if not runs or (w.get("speaker") or UNKNOWN_SPEAKER) != (runs[-1][0].get("speaker") or UNKNOWN_SPEAKER):
            runs.append([])
        runs[-1].append(w)
    return runs


def split(turns: Sequence[list[dict]], *, params: seg.SplitParams, duration: float,
          language: str | None) -> list[dict]:
    """Production's DP once per turn of seam-marked words, with the VAD derived from that turn's words; lines
    with `line_speaker` in order, the timeline contract per speaker (one speaker's overlapping lines pulled
    back, as on the Whisper path; two speakers' kept), scratch fields gone."""
    lines: list[dict] = []
    for words in turns:
        vad = derived_vad(sorted(words, key=lambda w: w["start"]))
        lines.extend(seg.split_segments(_input_segments(words), vad, params=params))
    for line in lines:
        line.pop("speaker", None)
        line[LINE_SPEAKER] = line_speaker(line.get("words") or [])
    lines = line_groups.grouped(line_groups.in_order(lines), segment_ops.clamp_segment_overlaps)
    lines, _ = segment_ops.normalize_segment_timeline(lines, audio_end=duration)
    for line in lines:
        for w in line.get("words") or []:
            for key in _WORD_SCRATCH:
                w.pop(key, None)
        if language:
            line["lang"] = language
    return lines


# --- MOSS Pro: segments -> units ---------------------------------------------------------------------------


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch)[0] in "PSZ"


def units_of(text: str) -> list[str]:
    """CJK by character, Latin by word (a run of ASCII letters and digits inside CJK text is one word too, or
    lines would spell `bug` as `b u g`); punctuation joins the unit before it."""
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        if tok.isascii():
            out.append(tok)
            continue
        for piece in _PIECE.findall(tok):
            if len(piece) == 1 and _is_punct(piece) and out:
                out[-1] += piece
            else:
                out.append(piece)
    return out


def unit_weight(unit: str) -> float:
    return max(weighted_char_count(unit), 0.5)


def interpolated(segment: dict) -> list[dict]:
    """A segment's units with its time shared out by weighted character count; contiguous."""
    us = units_of(segment["text"])
    weights = [unit_weight(u) for u in us]
    total = sum(weights) or 1.0
    t, out = segment["start"], []
    for u, w in zip(us, weights):
        d = (segment["end"] - segment["start"]) * w / total
        out.append({"start": t, "end": t + d, "text": u, "speaker": segment.get("speaker")})
        t += d
    return out


def merge_short(segments: Sequence[dict], *, params: seg.SplitParams, duration: float,
                gap: float = MERGE_GAP) -> list[dict]:
    """MOSS Pro segments as lines: one under the splitter's ideal lower bound joins the same speaker's
    neighbour across a gap shorter than `gap` when the result stays within the ok upper bounds. Never splits.
    Each segment's timed `units` are carried along and concatenated by a merge."""
    lines = [{"start": s["start"], "end": s["end"], "text": s["text"].strip(), "speaker": s.get("speaker"),
              "units": list(s["units"])}
             for s in sorted(segments, key=lambda s: s["start"]) if s["text"].strip() and s["start"] < duration]

    def short(line: dict) -> bool:
        return (weighted_char_count(line["text"]) < params.chars_ideal_lo
                or line["end"] - line["start"] < params.dur_ideal_lo)

    def fits(a: dict, b: dict) -> bool:
        return (a["speaker"] == b["speaker"] and b["start"] - a["end"] < gap
                and weighted_char_count(a["text"] + b["text"]) <= params.chars_ok_hi
                and b["end"] - a["start"] <= params.dur_ok_hi)

    changed = True
    while changed:
        changed = False
        for i, line in enumerate(lines):
            if not short(line):
                continue
            cands = [j for j in (i - 1, i + 1) if 0 <= j < len(lines)
                     and fits(*sorted((lines[i], lines[j]), key=lambda x: x["start"]))]
            if not cands:
                continue
            j = min(cands, key=lambda j: abs(lines[j]["start"] - line["start"]))
            a, b = sorted((i, j))
            sep = " " if lines[a]["text"][-1:].isascii() and lines[b]["text"][:1].isascii() else ""
            lines[a] = {"start": lines[a]["start"], "end": lines[b]["end"], "speaker": line["speaker"],
                        "text": lines[a]["text"] + sep + lines[b]["text"],
                        "units": lines[a]["units"] + lines[b]["units"]}
            del lines[b]
            changed = True
            break
    return lines


# --- per backend -----------------------------------------------------------------------------------------


def word_lines(units: Iterable[dict], *, duration: float, params: seg.SplitParams, language: str | None,
               per_turn: bool = False, grid: bool = False, sentence_ends=None) -> tuple[list[dict], list[dict]]:
    """35t (`per_turn`, `grid`, `sentence_ends`) and MAI2: (lines, derived VAD). `units` in output order;
    `per_turn` keeps that order and runs the DP per speaker run, else one DP over the words by start.
    `sentence_ends(words)` sets each word's `SENTENCE_END_BONUS_KEY` (over every word, in output order)."""
    if per_turn:
        turns = [dp_words(run, duration=duration, keep_order=True) for run in speaker_runs(list(units))]
    else:
        turns = [dp_words(units, duration=duration)]
    turns = [ungrid(words) if grid else words for words in turns if words]
    if sentence_ends is not None:
        sentence_ends([w for words in turns for w in words])
    vad = derived_vad(sorted((w for words in turns for w in words), key=lambda w: w["start"]))
    lines = split([mark_seams(words) for words in turns], params=params, duration=duration, language=language)
    return lines, vad


def widened(lines: Sequence[dict], *, pad: float, duration: float) -> list[dict]:
    """MAI2: `lines` (by start, none overlapping) with every start `pad` s earlier and every end `pad` s later,
    never onto a neighbour or past the audio: an end takes the gap up to the next line's start first, a start only
    what the previous line's new end left. The first word's start and the last word's end move with the line."""
    out: list[dict] = []
    for k, line in enumerate(lines):
        before = out[-1]["end"] if out else 0.0
        after = lines[k + 1]["start"] if k + 1 < len(lines) else duration
        start = round(max(line["start"] - pad, min(before, line["start"])), 3)
        end = round(min(line["end"] + pad, max(after, line["end"])), 3)
        words = [dict(w) for w in line.get("words") or []]
        if words:
            words[0]["start"] = min(words[0]["start"], start)
            words[-1]["end"] = max(words[-1]["end"], end)
        out.append({**line, "start": start, "end": end, **({"words": words} if words else {})})
    return out


def segment_lines(timed: Sequence[dict], *, duration: float, params: seg.SplitParams,
                  language: str | None) -> tuple[list[dict], list[dict]]:
    """MOSS Pro: segments with their timed `units` (`moss_timing.timed`) -> (lines, derived VAD). Short
    segments merge on MOSS Pro's own edges; each merged line is a turn, one input segment of its own DP."""
    vad = derived_vad(sorted((u for s in timed for u in s["units"]), key=lambda u: u["start"]))
    turns = [[{**w, "seam": k == 0} for k, w in enumerate(ungrid(dp_words(m["units"], duration=duration,
                                                                          keep_order=True)))]
             for m in merge_short(timed, params=params, duration=duration)]
    return split([words for words in turns if words], params=params, duration=duration, language=language), vad


def summary(lines: Sequence[dict]) -> dict[str, Any]:
    return {"lines": len(lines), "speakers": len({line[LINE_SPEAKER] for line in lines})}
