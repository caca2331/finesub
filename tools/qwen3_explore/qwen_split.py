"""A segmenter for the Qwen word stream, built around the merge stage's cost asymmetry.

One DP over the whole clip's word stream — no decode-window partitioning, no separate hard cut.
Everything is a boundary the DP weighs, over the production shape F(j) = min F(i) + B(i) + P(i+1..j).

Four things differ from production `segment_split`:

1. `base` 1.0 -> 0.05. Production encodes "don't cut unless forced" because Whisper already
   supplies sentence segments; Qwen supplies none and the merge contract's asymmetry runs the
   other way — a lost boundary is unrecoverable, an extra one usually is not.
2. Word-level pauses feed the gap term at a trust discount. Production deliberately ignores them
   because Whisper-DTW smears word times; a separate non-autoregressive aligner does not.
   **The second half of that premise is false** (FINDINGS §4.3): the Qwen aligner does not smear
   but it *invents* — 95%+ of its intra-segment pauses have no VAD silence under them, 7.5-12.6%
   of all junctions. Discounting them symmetrically was tried and bought nothing, because at
   `word_pause_trust=0.25` the term is already too small to drive cuts; the premise is wrong but
   the parameter happens not to matter.
3. The gap discount is steep above the knee and clipped (see `g_score`), which subsumes what an
   explicit hard-cut partitioning step used to do without letting it swamp the length penalty.
4. `fragment_penalty` charges for cutting *before* a token that never begins a cue — the
   mid-word cut, which manual adjudication found under most of the real errors.

The piece score P and the punctuation table are production's, unchanged. Three attempts to improve
on them — degrading punctuation trust, a Japanese particle term, a turn-marker bonus — were all
reverted after ablation showed they buy nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from subtitle_metrics import weighted_char_count  # noqa: E402

from .lexicon import PUNCT_STRIP, lexicon_for  # noqa: E402

MIN_WORD_SEC = 0.12   # floor for the aligner's 80 ms-grid zero-duration words

SENTENCE = set("。．｡.!！?？…‥‼⁇⁈⁉")
CLAUSE = set("、，,､；;：:")


@dataclass(frozen=True)
class Params:
    # Selected by leave-one-clip-out CV on the 8 tuning clips, reproducible with
    # `python -m tools.qwen3_explore.bench --cv ...`. The three interacting terms were re-selected
    # jointly after the CANNOT_START lexicon was fixed (see lexicon.py): 6 of 8 folds agree on
    # 0.25/0.5/1.5, and the held-out sources improve as well.
    base: float = 0.05            # per-cut cost; production uses 1.0 (repair duty)
    a: float = 1.0
    b: float = 1.0
    g_knee: float = 0.25
    g_power: float = 3.0          # production's square is 2.0; higher gets no better
    g_floor: float = -3.0         # clip, so the piece score (max 2.87) can still overrule
    no_gap_penalty: float = 0.2   # production 1.0; softened because word pauses now certify too
    word_pause_trust: float = 0.25  # discount on aligner pauses vs VAD gaps
    # Trust in a VAD gap the aligner does *not* corroborate (it reports the two words contiguous).
    # 0 = such a gap is no certificate at all. Not CV-selectable: the 8 tuning clips are flat
    # across the whole range (they lack the defect it addresses), so this value comes from the
    # mechanism, not from the metric. See FINDINGS §3.2 / §7（`yingtao`：分离残留抬高底噪 → 能量 VAD 漏语音）.
    vad_gap_trust_no_pause: float = 0.0
    # Reverted to the production trust after ablation: degrading it was justified by punctuation
    # precision measured against refined cue times, and those cues are re-timed — the wrong
    # yardstick. Ablation on the merge-contract criterion prefers the production values.
    punct_sentence: float = 0.0
    punct_clause: float = 0.2
    # Cost of cutting *before* a token that never begins a cue (`lexicon.CANNOT_START`). This is
    # the term the earlier STICKY_TAIL experiment was reaching for and missing — it penalised the
    # left side of the junction, and the data says the signal is on the right. Cross-validated
    # separately from the four above.
    fragment_penalty: float = 1.5
    # Discount for cutting where the *source ASR* already ended a segment. Default 0 = the
    # behaviour every earlier number was measured under.
    #
    # Step 1 of `segment()` concatenates all source segments into one stream, which for Qwen is
    # right (its segments are decode windows) but for Whisper throws away a sentence hypothesis
    # produced by a language model. The gold set measured the cost: of the 23 `must` positions the
    # whisper arm misses, 22 are places Whisper itself had cut, and 11 of those carry no pause, no
    # VAD and no punctuation — no acoustic or lexical term can reach them at any parameter value
    # (two sweeps, `a` over 100x, all flat at 54-64% recall). This term is the only input that can.
    #
    # Must stay 0 for the Qwen arm: there a "segment boundary" is an artefact of where the decode
    # window fell, and rewarding it reinstates the forced window cuts step 1 exists to remove.
    #
    # What it does NOT do: converge to production as it grows. It only *subtracts* cost at segment
    # boundaries and never *adds* cost elsewhere, so raising it can only add cuts. At bonus=50 the
    # arm misses just 1 of production's 364 in-window cuts — but also makes 27 production never
    # makes. The interpolation runs between the DP and `DP union production`, never to production
    # itself; that, not any empirical trade-off, is why violations rise monotonically with it.
    # Testing "high bonus = conservative" properly needs a matching penalty on non-boundary cuts.
    asr_boundary_bonus: float = 0.0
    dur_ideal = (1.2, 4.5)
    dur_ok = (0.6, 8.0)
    char_ideal = (5.0, 20.0)
    char_ok = (3.0, 36.0)


def g_score(g: float, p: Params) -> float:
    """Gap discount: linear below the knee, steep power above it, clipped at a floor.

    Production's square at knee 0.5 is too gentle to make a 0.5-0.8 s silence a cut on its own —
    that band is exactly why a separate hard cut existed. Moving the knee to 0.25 with a cube
    covers it while leaving gaps under 0.25 s untouched. Unclipped it would swamp the piece score
    (the worst possible piece is only 2.87), so every medium gap would become an unconditional cut
    and tiny cues would return; the floor is what keeps the length penalty able to say no.
    Steeper powers were tried up to 6 and buy nothing (identical metrics, harsher curve).
    """
    raw = -g if g <= p.g_knee else -(g ** p.g_power) / (p.g_knee ** (p.g_power - 1))
    return max(raw, p.g_floor)


def piece_score(c: float, d: float, p: Params) -> float:
    over_d = (max(0.0, d - p.dur_ideal[1]) + max(0.0, d - p.dur_ok[1])) / 3.5
    over_c = (max(0.0, c - p.char_ideal[1]) + max(0.0, c - p.char_ok[1])) / 16
    under_d = (max(0.0, p.dur_ideal[0] - d) + max(0.0, p.dur_ok[0] - d)) / 0.6 / 2
    under_c = (max(0.0, p.char_ideal[0] - c) + max(0.0, p.char_ok[0] - c)) / 2 / 2
    return p.b * (over_d + over_c + under_d + under_c)


def vad_gap_at(non_speech, t: float, tol: float = 0.12) -> float:
    best = 0.0
    for s, e in non_speech:
        if s - tol <= t <= e + tol:
            best = max(best, e - s)
    return best


def split_block(words, non_speech, p: Params, cannot_start: frozenset[str]):
    """DP over a word stream; returns the list of piece (start, end) word-index ranges.

    `cannot_start` must be the table derived from the tokenizer that produced `words` — see
    lexicon.py; feeding the other arm's table steers the DP away from good cuts.
    """
    n = len(words)
    if n < 2:
        return [(0, n)]

    boundary = []
    for k in range(n - 1):
        left, right = words[k], words[k + 1]
        pause = max(0.0, right["start"] - left["end"])
        vad = vad_gap_at(non_speech, (left["end"] + right["start"]) / 2)
        if vad > 0 and pause <= 1e-6:
            # VAD says silence, the aligner says the two words are contiguous. On material where
            # vocal separation leaves a loud residual floor (yingtao: non-speech at -48.7 dB vs
            # -115/-180 dB elsewhere) the energy VAD drops quiet speech, and both aligners then
            # place words inside those "silences". Treating such a gap as a cut certificate is how
            # `見えて|き|た` got sliced. Discount rather than ignore — the aligner is not the more
            # trustworthy of the two either.
            vad *= p.vad_gap_trust_no_pause
        g_eff = vad if vad > 0 else pause * p.word_pause_trust

        text = left.get("trailing_punct", "") or left["word"][-1:]
        if any(ch in SENTENCE for ch in text):
            t = p.punct_sentence
        elif any(ch in CLAUSE for ch in text):
            t = p.punct_clause
        else:
            t = 1.0
        if right["word"].strip(PUNCT_STRIP) in cannot_start:
            t += p.fragment_penalty
        if right.get("asr_boundary"):
            t -= p.asr_boundary_bonus

        cost = p.a * (t + g_score(g_eff, p)) + p.base
        if g_eff <= 1e-6:
            cost += p.no_gap_penalty
        boundary.append(cost)

    char = [weighted_char_count(w["word"] + w.get("trailing_punct", "")) for w in words]
    prefix = [0.0]
    for c in char:
        prefix.append(prefix[-1] + c)

    # Backward window bound. Beyond it the DP is provably not just unpromising but wrong to
    # explore: the piece score is monotone increasing past the ideal band (>=0.125/char, >=0.57/s),
    # while one extra boundary can save at most `-a*g_floor - base` = 2.95, so a piece this long
    # is always beaten by splitting it. Keeps the DP near-linear instead of O(n^2) — the full
    # sweep over 11 clips goes from minutes to seconds, with byte-identical output.
    MAX_PIECE_CHARS, MAX_PIECE_SEC = 100.0, 25.0

    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for j in range(1, n + 1):
        for i in range(j - 1, -1, -1):
            d = words[j - 1]["end"] - words[i]["start"]
            if i < j - 1 and (prefix[j] - prefix[i] > MAX_PIECE_CHARS or d > MAX_PIECE_SEC):
                break
            if best[i] == INF:
                continue
            cand = best[i] + piece_score(prefix[j] - prefix[i], d, p)
            if i > 0:
                cand += boundary[i - 1]
            if cand < best[j]:
                best[j] = cand
                back[j] = i

    pieces, j = [], n
    while j > 0:
        i = back[j]
        pieces.append((i, j))
        j = i
    return list(reversed(pieces))


def clamp_words_to_speech(words: list[dict], non_speech, min_sil: float = 0.30) -> list[dict]:
    """Trim word spans that swallow a silence — the aligner's interpolation artefact.

    Same shape as the DTW smearing `segment_split.adjust_words` clamps for Whisper (docs
    情况 2/3): a word cannot really span audio the model never heard speech in, so the
    swallowed silence is fabricated duration. Without this the gap term has no word boundary to
    score at and the silence stays buried inside a cue.
    """
    trimmed = []
    for w in words:
        start, end = w["start"], w["end"]
        for s, e in non_speech:
            if e - s < min_sil:
                continue
            lo, hi = max(s, start), min(e, end)
            if hi - lo < min_sil:
                continue
            # keep the side with more speech contact, drop the fabricated overhang
            if (lo - start) >= (end - hi):
                end = lo
            else:
                start = hi
        trimmed.append((w, start, end, end <= start))

    # A word left with no speech has a meaningless timestamp but its text still belongs
    # somewhere. Fold it into the nearest surviving neighbour — previous if there is one,
    # otherwise the next. Leaving 0.08 s stubs let the DP strand them as one-character cues.
    out: list[dict] = []
    pending = ""
    for w, start, end, collapsed in trimmed:
        if collapsed:
            if out:
                out[-1]["word"] += w["word"]
            else:
                pending += w["word"]
            continue
        entry = dict(w, start=round(start, 3), end=round(end, 3), raw_start=w["start"])
        if pending:
            entry["word"] = pending + entry["word"]
            pending = ""
        out.append(entry)
    if pending and not out:  # every word collapsed: keep one carrier so the text survives
        w, start, _end, _c = trimmed[0]
        out.append(dict(w, word=pending, start=round(start, 3), end=round(start + 0.08, 3),
                        raw_start=w["start"]))

    # Floor the aligner's zero-duration words (~12% of them, the 80 ms grid artefact) by
    # borrowing from the following gap. Without this a zero-duration word can become a
    # zero-length cue, which every downstream length metric and `to_srt` treats as garbage.
    # A run of words sharing one timestamp is `_fix_timestamps` collapsing a block (FINDINGS §4.3)
    # at small scale — spread them across whatever room exists before the next distinct stamp.
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1]["start"] - out[i]["start"] < 1e-6:
            j += 1
        count = j - i + 1
        if count > 1:
            limit = out[j + 1]["start"] if j + 1 < len(out) else out[j]["end"] + count * MIN_WORD_SEC
            room = limit - out[i]["start"]
            if room > 0:
                step = room / count
                for k in range(count):
                    out[i + k]["start"] = round(out[i]["start"] + k * step, 3)
                    out[i + k]["end"] = round(out[i]["start"] + (k + 1) * step, 3)
        i = j + 1

    for i, w in enumerate(out):
        if w["end"] - w["start"] >= MIN_WORD_SEC:
            continue
        limit = out[i + 1]["start"] if i + 1 < len(out) else w["start"] + MIN_WORD_SEC
        w["end"] = round(max(w["end"], min(w["start"] + MIN_WORD_SEC, limit)), 3)
    return out


def snap_cues_to_speech(cues: list[dict], non_speech, duration: float, max_stretch: float = 1.0) -> None:
    """Let a cue occupy the speech it sits in, instead of its words' literal span.

    An isolated短 word between two silences is *meant* to become its own cue
    (docs/segment_split.md), but the aligner's zero-duration words give it an 0.08 s span, which
    every length metric then reads as a defect. The speech region around it is the honest extent.
    Only ever extends, never shrinks, never crosses a neighbouring cue or a silence.
    """
    speech = []
    prev = 0.0
    for a, b in sorted(non_speech):
        if a > prev:
            speech.append((prev, a))
        prev = max(prev, b)
    if prev < duration:
        speech.append((prev, duration))

    for i, cue in enumerate(cues):
        mid = (cue["start"] + cue["end"]) / 2
        region = next((r for r in speech if r[0] <= mid <= r[1]), None)
        if region is None:
            continue
        lo = cues[i - 1]["end"] if i else 0.0
        hi = cues[i + 1]["start"] if i + 1 < len(cues) else duration
        cue["start"] = round(max(region[0], lo, cue["start"] - max_stretch, min(cue["start"], region[1])), 3)
        cue["end"] = round(min(region[1], hi, cue["end"] + max_stretch), 3)
        if cue["end"] <= cue["start"]:
            cue["end"] = round(cue["start"] + MIN_WORD_SEC, 3)


def attach_punctuation(text: str, words: list[dict]) -> list[dict]:
    """Re-attach the punctuation nagisa stripped, so the DP can see sentence boundaries."""
    out = [dict(w, trailing_punct="") for w in words]
    idx = 0
    consumed = 0
    for ch in text:
        if idx < len(out) and consumed >= len(out[idx]["word"]):
            idx += 1
            consumed = 0
        if ch in SENTENCE or ch in CLAUSE:
            target = min(idx if consumed else idx - 1, len(out) - 1)
            if target >= 0:
                out[target]["trailing_punct"] += ch
        elif idx < len(out) and consumed < len(out[idx]["word"]) and ch == out[idx]["word"][consumed]:
            consumed += 1
    return out


def segment(raw: dict, non_speech, duration: float, p: Params, words_carry_punct: bool,
            clamp: bool = True, snap: bool = True) -> dict:
    """Re-segment `raw` in place and return it. Callable in-process by the bench / CV drivers.

    `clamp` and `snap` exist because both re-time cues, and re-timing is not free: measured against
    the gold set, a cue set with *identical* word-level boundaries scores 95.8% -> 75.0% must
    recall purely from the shift (median 0.012 s but 0.385 s at the 90th percentile, 0.713 s worst
    — `snap` moves a boundary from the middle of a silence to its edge). Both were written for the
    Qwen aligner's artefacts (zero-duration words on an 80 ms grid, spans swallowing silence); on
    a Whisper word stream they may be unnecessary. Defaults keep the measured behaviour.
    """
    # 1. Concatenate every decode window into one word stream first. Splitting per window made
    #    the ASR's ~30 s window boundaries into forced cue boundaries, which the DP never got to
    #    weigh: on the same data, concatenating first drops unrecoverable over-splits 23 -> 15
    #    (tuning) and 53 -> 49 (unseen) and improves shape 0.216 -> 0.195 / 0.330 -> 0.309.
    all_words: list[dict] = []
    lang = None
    for seg in raw["segments"]:
        words = seg.get("words")
        if not words:
            continue
        lang = lang or seg.get("lang")
        if clamp:
            words = clamp_words_to_speech(words, non_speech)
        if words_carry_punct:
            words = [dict(w, trailing_punct="") for w in words]
        else:
            words = attach_punctuation(seg["text"], words)
        if words:
            # Tag rather than record an index: the sort below reorders, and clamping can drop or
            # merge words, so any index captured here would not survive.
            words[0] = dict(words[0], asr_boundary=True)
        all_words.extend(words)
    all_words.sort(key=lambda w: (w["start"], w["end"]))

    segments = []
    # 2. One DP over the whole stream. The clipped steep gap discount now covers the 0.5-0.8 s
    #    band that used to need a separate hard cut, so there is no partitioning step left and no
    #    minimum-block patch: every boundary is a decision the DP weighs.
    lex = lexicon_for(words_carry_punct)
    for i, j in split_block(all_words, non_speech, p, lex) if all_words else []:
        chunk = all_words[i:j]
        if not chunk:
            continue
        text = "".join(w["word"] + w.get("trailing_punct", "") for w in chunk)
        segments.append(
            {
                "start": round(chunk[0]["start"], 3),
                "end": round(chunk[-1]["end"], 3),
                "text": text,
                "lang": lang,
                "words": [{k: v for k, v in w.items() if k not in ("trailing_punct", "asr_boundary")}
                          for w in chunk],
            }
        )

    segments.sort(key=lambda s: s["start"])
    if snap:
        snap_cues_to_speech(segments, non_speech, duration)
    raw["segments"] = segments
    raw.setdefault("metadata", {})["qwen_split"] = {
        "base": p.base,
        "word_pause_trust": p.word_pause_trust,
        "no_gap_penalty": p.no_gap_penalty,
        "fragment_penalty": p.fragment_penalty,
        "punct_sentence": p.punct_sentence,
        "punct_clause": p.punct_clause,
    }
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="rescued_asr.py output")
    ap.add_argument("--vad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=float, default=Params.base)
    ap.add_argument("--word-pause-trust", type=float, default=Params.word_pause_trust)
    ap.add_argument("--no-gap-penalty", type=float, default=Params.no_gap_penalty)
    ap.add_argument("--fragment", type=float, default=Params.fragment_penalty)
    ap.add_argument(
        "--words-carry-punct",
        action="store_true",
        help="input words already contain punctuation (Whisper aligned.json); skip re-attachment",
    )
    args = ap.parse_args()

    p = Params(
        base=args.base,
        word_pause_trust=args.word_pause_trust,
        no_gap_penalty=args.no_gap_penalty,
        fragment_penalty=args.fragment,
    )

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))
    non_speech = [(float(a), float(b)) for a, b in vad["non_speech"]]

    out = segment(raw, non_speech, vad["duration"], p, args.words_carry_punct)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.out}: {len(out['segments'])} cues")


if __name__ == "__main__":
    main()
