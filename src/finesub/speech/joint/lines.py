"""A cloud transcript's units -> subtitle lines through production's DP.

The rules are the 2026-10 experiment's (arms M1 for MAI2, G1s for 35t, S5b / S6b for MOSS Pro) and the 2026-11
Scribe round's (E1st, MAI2's route); readouts and reasons in `docs/joint-asr.md`. The readers preferred them:

* words in DP shape: punctuation folded into the word before it, `space_before` between Latin words, cut at
  the audio's end;
* a derived VAD: the words' own spans, gaps shorter than VAD_GAP closed (no VAD runs on a cloud task);
* input segments ("seams", the role an ASR segment plays in production's DP): at every speaker change and
  every pause over SEAM_PAUSE (35t, MAI2), or every MOSS Pro line after merging its short segments;
* 35t only: abutting words pulled GRID_GAP apart (its 100 ms grid makes them abut, and the DP reads a zero
  pause as connected speech); 35t and Scribe: the sentence-boundary model's evidence taken off each cut (`sbd.py`);
* in Chinese (every backend): sentence-end evidence after every clause mark (`clause_bonus`).

Turns: 35t and MOSS Pro each run the DP once per turn -- a 35t turn is one
speaker's run in output order, a MOSS Pro turn is a line after the merge -- with a VAD derived from that turn's
own words, so no line takes words from two turns and two people talking at once get lines that overlap rather
than interleave. MAI2's and Scribe's words never overlap: one DP over all of them by start, a speaker change only a seam
(arm M1), so a MAI2 line may hold two people's words. Each line gets `line_speaker`, the speaker holding most of
its words' time; lines are in order by start, same start by `line_speaker`.
"""

from __future__ import annotations

import functools
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
#: In Chinese, a word ending in a clause mark carries CLAUSE_BONUS of sentence-end evidence (`clause_bonus`). Human
#: Chinese subtitles break at nearly every comma and production's Whisper segments do too; without that evidence the
#: word DP fills its ideal box and Chinese lines come out about twice as long (`docs/joint-asr.md`「中文的子句切点」).
#: Evidence, not a seam: a seam's bonus would also cut a lone 「呃，」 off as its own line.
CLAUSE_MARKS = frozenset("，。？！；：,.?!;:")
CLAUSE_BONUS = 2.0

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
    kept: dict | None = None  # the unit before this one, if it was kept
    for w in raw:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        if w["start"] >= duration:
            # past the cut, only a mark or symbol with no time of its own, at the end of the kept unit before it,
            # stays (a unit the aligner gives no time takes that unit's end, which may be the audio's very end):
            # folded into it
            if (kept is not None and w["start"] == w["end"] == kept["end"]
                    and all(unicodedata.category(ch)[0] in "PSZ" for ch in text)):
                words[-1]["word"] += text
            else:
                kept = None
            continue
        kept = w
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


def clause_bonus(language: str | None) -> float | None:
    """The clause evidence `word_lines` takes for a task in `language`: CLAUSE_BONUS in Chinese, else none."""
    return CLAUSE_BONUS if (language or "").lower().startswith("zh") else None


def clause_ends(words: Iterable[dict], bonus: float = CLAUSE_BONUS) -> None:
    """At least `bonus` of sentence-end evidence on every word ending in a CLAUSE_MARKS mark."""
    for w in words:
        if w["word"][-1:] in CLAUSE_MARKS:
            w[seg.SENTENCE_END_BONUS_KEY] = max(float(w.get(seg.SENTENCE_END_BONUS_KEY) or 0.0), bonus)


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


#: Hiragana and katakana, without the middle dot and the long-vowel mark (Chinese text uses them in names).
_KANA = re.compile(r"[\u3041-\u3096\u30a1-\u30fa]")
#: nagisa's parts of speech that belong with the word before them in a line (`_glued`).
_GLUE_POS = frozenset({"助詞", "助動詞", "接尾辞"})
#: What a te-form's て / で follows (読ん+で, 寒く+て, させ+て); after a noun で is the case particle.
_TE_FORM = frozenset({"動詞", "形容詞", "助動詞"})


def japanese(text: str) -> bool:
    """A segment with any kana is Japanese (judged per segment, so a task without `--language` is judged too)."""
    return bool(_KANA.search(text))


def _glued(word: str, pos: str, prev: tuple[str, str], before: str) -> bool:
    """Whether a Japanese word stays in one line with the word before it (`prev`: its surface and part of speech;
    `before`: the part of speech of the word before that): a particle, an auxiliary or a suffix; a word after a
    prefix; a noun after a noun (a compound: ファースト+コンタクト, 血統+代理+人); a verb after a verb (a compound
    verb: 走り+出す; also a repeat: わかる+わかる) or after a verb's te-form (待っ+て+ください -- not a case particle's で: ゲームで / 遊ぶ); a
    suru-verb's する after its noun (作用+する)."""
    prev_word, prev_pos = prev
    if pos in _GLUE_POS or prev_pos == "接頭辞" or (pos == "名詞" and prev_pos == "名詞"):
        return True
    if pos == "動詞":
        return prev_pos == "動詞" or (prev_pos == "助詞" and prev_word in ("て", "で") and before in _TE_FORM) or (
            prev_pos == "名詞" and word.startswith(("し", "す", "さ", "せ", "でき")))
    return False


def _japanese_words(token: str) -> list[tuple[str, str]]:
    """(surface, part of speech) for a space-free token, by nagisa (the dictionary segmenter transformers' Qwen3
    processors use for Japanese; light, with wheels on every platform). nagisa normalises some characters (？ -> ?) but not their count, so the
    surfaces are cut from `token` by its words' lengths; a token it would change in length is cut by character."""
    import nagisa  # noqa: PLC0415 -- dictionary model, loaded on the first Japanese segment

    tagged = nagisa.tagging(token)
    if sum(map(len, tagged.words)) != len(token):
        return [(piece, "") for piece in _PIECE.findall(token)]
    out: list[tuple[str, str]] = []
    at = 0
    for word, pos in zip(tagged.words, tagged.postags):
        surface = token[at:at + len(word)]
        at += len(word)
        if out and _LATIN.match(out[-1][0][-1]) and _LATIN.match(surface[0]):
            out[-1] = (out[-1][0] + surface, out[-1][1])  # 4|5 -> 45: no `space_before` inside one run
        else:
            out.append((surface, pos))
    return out


@functools.lru_cache(maxsize=8192)
def _words(text: str) -> tuple[tuple[str, bool], ...]:
    """(word, glued to the word before it) for a Japanese segment; punctuation joins the word before it."""
    out: list[list] = []  # [word, glued, part of speech]
    for tok in _TOKEN.findall(text):
        for piece, pos in [(tok, "")] if tok.isascii() else _japanese_words(tok):
            if all(_is_punct(ch) for ch in piece) and out:
                out[-1][0] += piece
                out[-1][2] = "punct"
                continue
            glued = bool(out) and out[-1][2] != "punct" and _glued(
                piece, pos, (out[-1][0], out[-1][2]), out[-2][2] if len(out) > 1 else "")
            out.append([piece, glued, pos])
    return tuple((w, g) for w, g, _ in out)


def units_of(text: str) -> list[str]:
    """A MOSS segment's units, which Silero times (`moss_timing.warp`): CJK by character, Latin by word (a run of
    ASCII letters and digits inside CJK text is one word too, or lines would spell `bug` as `b u g`); punctuation
    joins the unit before it."""
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


def japanese_segment(text: str, language: str | None) -> bool:
    """Whether MOSS text is handled as Japanese: the task's `language` is Japanese or not given, and the text has
    kana (a Chinese task's stray の is not)."""
    return not (language and not language.lower().startswith("ja")) and japanese(text)


def groups(units: Sequence[dict], text: str, *, language: str | None) -> list[dict]:
    """A segment's timed `units` (spelling `text` without its spaces, as `units_of` does) merged into groups the
    line DP does not cut inside: in Japanese a dictionary word (nagisa) with what stays with it (`_glued`: 出て,
    ギミクが, 作用する) for a `japanese_segment`; elsewhere every unit is its own group. A unit is merged when its
    first character does not start a group."""
    if not japanese_segment(text, language):
        return [dict(u) for u in units]
    starts, at = set(), 0
    for word, glued in _words(text):
        if not glued:
            starts.add(at)
        at += len(word)
    out: list[dict] = []
    at = 0
    for u in units:
        if out and at not in starts:
            out[-1].update(text=out[-1]["text"] + u["text"], end=max(out[-1]["end"], u["end"]))
        else:
            out.append(dict(u))
        at += len(u["text"])
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
               per_turn: bool = False, grid: bool = False, sentence_ends=None,
               clauses: float | None = None) -> tuple[list[dict], list[dict]]:
    """35t (`per_turn`, `grid`, `sentence_ends`), MAI2 and Scribe: (lines, derived VAD). `units` in output order;
    `per_turn` keeps that order and runs the DP per speaker run, else one DP over the words by start.
    `sentence_ends(words)` sets each word's `SENTENCE_END_BONUS_KEY` (over every word, in output order);
    `clauses`: that much sentence-end evidence after every clause mark (`clause_ends`)."""
    if per_turn:
        turns = [dp_words(run, duration=duration, keep_order=True) for run in speaker_runs(list(units))]
    else:
        turns = [dp_words(units, duration=duration)]
    turns = [ungrid(words) if grid else words for words in turns if words]
    if sentence_ends is not None:
        sentence_ends([w for words in turns for w in words])
    if clauses:
        clause_ends((w for words in turns for w in words), clauses)
    vad = derived_vad(sorted((w for words in turns for w in words), key=lambda w: w["start"]))
    lines = split([mark_seams(words) for words in turns], params=params, duration=duration, language=language)
    return lines, vad


def widened(lines: Sequence[dict], *, pad: float, duration: float, end_pad: float | None = None) -> list[dict]:
    """MAI2 and Scribe: `lines` (by start, none overlapping) with every start `pad` s earlier and every end `end_pad` (default
    `pad`) s later, never onto a neighbour or past the audio: an end takes the gap up to the next line's start first,
    a start only what the previous line's new end left. The first word's start and the last word's end move with
    the line."""
    end_pad = pad if end_pad is None else end_pad
    out: list[dict] = []
    for k, line in enumerate(lines):
        before = out[-1]["end"] if out else 0.0
        after = lines[k + 1]["start"] if k + 1 < len(lines) else duration
        start = round(max(line["start"] - pad, min(before, line["start"])), 3)
        end = round(min(line["end"] + end_pad, max(after, line["end"])), 3)
        words = [dict(w) for w in line.get("words") or []]
        if words:
            words[0]["start"] = min(words[0]["start"], start)
            words[-1]["end"] = max(words[-1]["end"], end)
        out.append({**line, "start": start, "end": end, **({"words": words} if words else {})})
    return out


def segment_lines(timed: Sequence[dict], *, duration: float, params: seg.SplitParams,
                  language: str | None, sentence_ends=None,
                  clauses: float | None = None) -> tuple[list[dict], list[dict]]:
    """MOSS Pro and local MOSS: segments with their timed `units` (`moss_timing.timed`) -> (lines, derived VAD).
    Japanese units merge into `groups` first; short segments merge on MOSS's own edges; each merged line is a
    turn, one input segment of its own DP. `sentence_ends(words)` and `clauses` as in `word_lines`, over every turn's
    words."""
    vad = derived_vad(sorted((u for s in timed for u in s["units"]), key=lambda u: u["start"]))
    grouped = [dict(s, units=groups(s["units"], s["text"], language=language)) for s in timed]
    turns = [[{**w, "seam": k == 0} for k, w in enumerate(ungrid(dp_words(m["units"], duration=duration,
                                                                          keep_order=True)))]
             for m in merge_short(grouped, params=params, duration=duration)]
    turns = [words for words in turns if words]
    if sentence_ends is not None and turns:
        sentence_ends([w for words in turns for w in words])
    if clauses:
        clause_ends((w for words in turns for w in words), clauses)
    return split(turns, params=params, duration=duration, language=language), vad


def summary(lines: Sequence[dict]) -> dict[str, Any]:
    return {"lines": len(lines), "speakers": len({line[LINE_SPEAKER] for line in lines})}


# --- moss-diarization-aligner: the lines' new speakers, and overlap cues -----------------------------------------

#: The line's speaker as its ASR gave it, kept beside the new `line_speaker`.
ASR_SPEAKER = "asr_speaker"
SPEAKER_MARGIN = line_groups.SPEAKER_MARGIN
#: Overlap cues (`overlap_cues`; the exploration's readers, `docs/joint-asr.md`): one person's pieces join across
#: other people's lines between them when every gap along the chain is at most CUE_GAP s, those lines last CUE_INTER s
#: at most in all, and the next piece starts at most CUE_HOLE s after the cue so far ends.
CUE_GAP = 0.1
CUE_INTER = 0.6
CUE_HOLE = 1.5
#: A cue whose text so far ends a sentence takes no further piece (the speaker had finished). A single `.` counts;
#: `...` does not: an unfinished line cut off by someone is what an overlap cue is for.
SENTENCE_STOPS = "。？！?!"


def relabelled(lines: Sequence[dict], speakers: Sequence[str], margins: Sequence[float | None], *,
               duration: float) -> list[dict]:
    """`lines` with `line_speaker` replaced (the old one in `asr_speaker`, the margin in `speaker_margin`), then the
    timeline contract redone per speaker: two lines the ASR gave two people may now be one person's and overlap."""
    out = []
    for line, who, margin in zip(lines, speakers, margins, strict=True):
        new = {**line, ASR_SPEAKER: line[LINE_SPEAKER], LINE_SPEAKER: who}
        if margin is not None:
            new[SPEAKER_MARGIN] = margin
        out.append(new)
    out = line_groups.grouped(line_groups.in_order(out), segment_ops.clamp_segment_overlaps)
    out, _ = segment_ops.normalize_segment_timeline(out, audio_end=duration)
    return out


def sentence_closed(text: str) -> bool:
    """Whether `text` ends a sentence (`SENTENCE_STOPS`, or one `.` that is not the end of `...`)."""
    text = text.rstrip()
    return bool(text) and (text[-1] in SENTENCE_STOPS or (text[-1] == "." and not text.endswith("..")))


def overlap_cues(lines: Sequence[dict], *, params: seg.SplitParams) -> tuple[list[dict], list[list[int]]]:
    """(lines, the merges as lists of positions in `lines`): a line joins the open cue of its speaker when other
    people's lines lie between them (CUE_GAP, CUE_INTER, CUE_HOLE), the cue does not end a sentence, and the joined cue
    stays within the split's limits (`dur_ok_hi`, weighted `chars_ok_hi`: `--split-length-scale` scales both). The
    lines between are kept as they are, overlapping the cue. A joined cue: the pieces' words in order, the seam's
    `space_before` recomputed, its text from its words; `asr_speaker` the first piece's, the margin the smallest.
    `UNKNOWN_SPEAKER`'s lines join nothing. `lines` by start."""
    from ...text import words_to_text

    cues: list[dict] = []
    open_: dict[str, dict] = {}
    for k, line in enumerate(lines):
        who = line[LINE_SPEAKER]
        cue = open_.get(who) if who != UNKNOWN_SPEAKER else None
        if cue is not None and _joins(cue, line, lines, k, params):
            cue["ids"].append(k)
            cue["end"] = line["end"]
            cue["text"] += line["text"]
            continue
        cue = {"ids": [k], "end": line["end"], "text": line["text"]}
        cues.append(cue)
        open_[who] = cue
    out = []
    for cue in cues:
        pieces = [lines[k] for k in cue["ids"]]
        if len(pieces) == 1:
            out.append(pieces[0])
            continue
        words = [dict(w) for w in pieces[0].get("words") or []]
        for piece in pieces[1:]:
            more = [dict(w) for w in piece.get("words") or []]
            if words and more:
                more[0]["space_before"] = spaced(str(words[-1].get("word", "")), str(more[0].get("word", "")))
            words += more
        joined = {**pieces[0], "end": pieces[-1]["end"], "words": words, "text": words_to_text(words)}
        sure = [p[SPEAKER_MARGIN] for p in pieces if p.get(SPEAKER_MARGIN) is not None]
        joined.pop(SPEAKER_MARGIN, None)
        if sure:
            joined[SPEAKER_MARGIN] = min(sure)
        out.append(joined)
    return line_groups.in_order(out), [c["ids"] for c in cues if len(c["ids"]) > 1]


def _joins(cue: dict, line: dict, lines: Sequence[dict], k: int, params: seg.SplitParams) -> bool:
    who = line[LINE_SPEAKER]
    between = [x for j, x in enumerate(lines[:k])
               if j not in cue["ids"] and cue["end"] <= x["start"] < line["start"]]
    if not between or any(x[LINE_SPEAKER] == who for x in between) or sentence_closed(cue["text"]):
        return False
    chain = [cue["end"]] + [t for x in between for t in (x["start"], x["end"])] + [line["start"]]
    first = lines[cue["ids"][0]]
    return (all(b - a <= CUE_GAP for a, b in zip(chain[::2], chain[1::2]))
            and sum(x["end"] - x["start"] for x in between) <= CUE_INTER
            and line["start"] - cue["end"] <= CUE_HOLE
            and line["end"] - first["start"] <= params.dur_ok_hi
            and weighted_char_count(cue["text"] + line["text"]) <= params.chars_ok_hi)
