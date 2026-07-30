"""Score a segmentation by what the downstream merge stage can and cannot repair.

Exact-match F1 against the refined subtitles is the wrong yardstick twice over: the refined
cues are re-timed (docs/merge-calibration.md says timing judgements go by raw ASR gaps), and
many boundaries are genuinely optional. What is *not* optional is the cost asymmetry the merge
contract already encodes:

- 错并代价高于漏并 — a merged pair cannot be split back downstream. Inverted for splitting:
  **a missing boundary is unrecoverable, an extra boundary is usually recoverable.**
- but recovery has limits: merge takes at most two adjacent sources (three only for a filler
  sandwich), and refuses results over 4 s / 20 weighted chars. An over-split that needs three
  pieces glued back, or whose pieces exceed those caps, is unrecoverable too.

Gaps are measured from the shared VAD non-speech track rather than either arm's word times, so
Whisper's DTW smearing and Qwen's 80 ms quantisation do not bias the comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from subtitle_metrics import weighted_char_count  # noqa: E402

# docs/merge-calibration.md 机械信号
GAP_MERGE_CONSIDER = 0.35      # <0.35s 才考虑合并
GAP_MERGE_REFUSE = 0.48        # >=0.48s 默认不合并 -> 段内含此 gap = merge 绝不会产出的形态
GAP_FORCED = 1.00              # segment_split: g=1 起近乎强制切分
MERGE_SPAN_CAP = 4.0           # capableB/C 合并后硬门槛
MERGE_CHAR_CAP = 20.0
MERGE_SPAN_ABS = 7.0           # 绝对门槛
MERGE_CHAR_ABS = 36.0

# docs/merge-calibration.md 语义闸门「宜分开」：话题/情绪转折、问答轮替、换人。
# Matched against whole words, never as substrings: 「いつまでも」 contains でも,
# 「思い出せないや」 contains いや, 「集まってる」 contains まって — substring matching flagged
# all of those as missed turns and made this metric useless.
TURN_MARKERS = frozenset(
    ("でも", "いや", "だけど", "けども", "てか", "つか", "ってか", "じゃあ", "しかも", "ところで")
)

# A cue starting with one of these was cut mid-word or mid-phrase — the error class the manual
# adjudication found underneath most of what the other two metrics were flagging. See
# `lexicon.py` for how it is derived and why it spans two tokenizations.
from .lexicon import PUNCT_STRIP, lexicon_for  # noqa: E402


SENTENCE_MARKS = frozenset("。．｡.!！?？…‥‼⁇⁈⁉")


# docs/segment_split.md 的片段分：三档铰链（理想区 0；可接受区单铰链；超出后双铰链），
# 过短罚在可接受边缘 0.5、过长罚在边缘 1.0 —— 字幕偏短比偏长便宜，这个不对称是有意的。
# 直接沿用生产常数，不另立一套，否则调参就是在对着自己发明的尺子优化。
TIER_OK_MAX = 1.0  # 片段分 <=1 大致对应「勉强可接受」的边缘


def piece_penalty(c: float, d: float) -> float:
    """The production piece score P(c, d), reused verbatim as a shape quality metric."""
    over_d = (max(0.0, d - 4.5) + max(0.0, d - 8.0)) / 3.5
    over_c = (max(0.0, c - 20.0) + max(0.0, c - 36.0)) / 16
    under_d = (max(0.0, 1.2 - d) + max(0.0, 0.6 - d)) / 0.6 / 2
    under_c = (max(0.0, 5.0 - c) + max(0.0, 3.0 - c)) / 2 / 2
    return over_d + over_c + under_d + under_c


EDGE_MARGIN = 0.10


def vad_gaps_within(non_speech, a: float, b: float) -> list[float]:
    """Overlap lengths of shared non-speech stretches with (a, b)."""
    out = []
    for s, e in non_speech:
        lo, hi = max(s, a), min(e, b)
        if hi - lo > 0:
            out.append(hi - lo)
    return out


def buried_silences(non_speech, a: float, b: float) -> list[float]:
    """Silences genuinely *inside* a cue, with speech on both sides.

    A silence merely touching a cue edge is not an under-split — the cue simply starts or ends
    next to a pause. Only a silence with speech either side of it is a boundary the cue swallowed.
    """
    return [
        e - s
        for s, e in non_speech
        if s > a + EDGE_MARGIN and e < b - EDGE_MARGIN
    ]


def first_token(seg: dict) -> str:
    for w in seg.get("words") or []:
        t = str(w.get("word", "")).strip(PUNCT_STRIP)
        if t:
            return t
    return str(seg.get("text") or "").strip()[:2]


def analyse(data: dict, non_speech, label: str, whisper_words: bool = False) -> dict:
    """`whisper_words`: the stream is whisper-timestamped subwords, not nagisa morphemes.

    It only selects the `CANNOT_START` table (lexicon.py). Manual adjudication of 12 flags per arm
    put mid-word precision at 7/12 (nagisa) and 3/12 (whisper) when both used the union, so this
    column is reported per arm and never compared between them.
    """
    cannot_start = lexicon_for(whisper_words)
    segs = [s for s in data["segments"] if str(s.get("text") or "").strip()]

    under_hard = under_forced = midword = 0
    over_recoverable = over_unrecoverable = 0
    turn_missed = 0
    free = 0

    for seg in segs:
        # A cue whose own text is already a fragment is not an under-split: the damage was a cut
        # elsewhere, and "cut inside it too" is meaningless. 31% of the old under-split flags were
        # this shape (manual adjudication, out/qwen-explore/boundary-adjudication.md).
        fragment = first_token(seg) in cannot_start
        if fragment:
            midword += 1
        else:
            internal = buried_silences(non_speech, seg["start"], seg["end"])
            under_hard += sum(1 for g in internal if g >= GAP_MERGE_REFUSE)
            under_forced += sum(1 for g in internal if g >= GAP_FORCED)
        # Word-level so a marker only counts when it really is a discourse token, and only
        # mid-cue — a cue *starting* with でも is correctly segmented, not a miss.
        words = [str(w.get("word", "")).strip("、。 　") for w in seg.get("words") or []]
        if any(w in TURN_MARKERS for w in words[1:]):
            turn_missed += 1

    # Over-splitting is only an error when the pieces are short enough that they plausibly
    # belong together. Two long cues separated by a small gap is a *necessary* cut — merge
    # would refuse to join them anyway (docs: 前句已完整再起信息，即便 gap≈0 也分开).
    # Merge takes at most two adjacent sources, so a run of 3+ tight short pieces is the one
    # over-split shape it cannot undo.
    run = [segs[0]] if segs else []
    runs = []
    for prev, cur in zip(segs, segs[1:]):
        gap = max(0.0, cur["start"] - prev["end"])
        vad_gap = max(vad_gaps_within(non_speech, prev["end"] - 0.05, cur["start"] + 0.05) or [0.0])
        gap = max(gap, vad_gap)
        if gap >= GAP_MERGE_CONSIDER:
            free += 1
            runs.append(run)
            run = [cur]
        else:
            run.append(cur)
    runs.append(run)

    for group in runs:
        if len(group) < 2:
            continue
        span = group[-1]["end"] - group[0]["start"]
        chars = sum(weighted_char_count(s["text"]) for s in group)
        # Three cues that each end in sentence punctuation are three sentences, not an
        # over-split — merge refuses to join them too (前句已完整再起信息). 60% of the old
        # unrecoverable flags were exactly this.
        complete = all(str(s["text"]).strip()[-1:] in SENTENCE_MARKS for s in group[:-1])
        joinable = span <= MERGE_SPAN_ABS and chars <= MERGE_CHAR_ABS and not complete
        if not joinable:
            free += len(group) - 1         # merge would refuse anyway -> these cuts are necessary
        elif len(group) == 2:
            over_recoverable += 1
        else:
            over_unrecoverable += len(group) - 1

    durations = [s["end"] - s["start"] for s in segs]
    chars = [weighted_char_count(s["text"]) for s in segs]
    n = len(segs)
    shape = [piece_penalty(c, d) for c, d in zip(chars, durations)]
    return {
        "label": label,
        "cues": n,
        "under_hard": under_hard,
        "midword": midword,
        "under_forced": under_forced,
        "turn_missed": turn_missed,
        "over_recoverable": over_recoverable,
        "over_unrecoverable": over_unrecoverable,
        "free_cuts": free,
        # Independent of the DP objective on purpose: `shape_mean` reuses the production piece
        # score, which is also what the DP minimises, so optimising it alone is circular. The
        # error counts and this ratio are outside that loop and act as the guard rails.
        "cut_certified": 100 * free / max(1, n - 1),
        "shape_mean": sum(shape) / n,
        "shape_p90": sorted(shape)[min(n - 1, int(0.9 * n))],
        "tier_ideal": 100 * sum(s == 0 for s in shape) / n,
        "tier_ok": 100 * sum(0 < s <= TIER_OK_MAX for s in shape) / n,
        "tier_bad": 100 * sum(s > TIER_OK_MAX for s in shape) / n,
        "dur_ok": 100 * sum(0.6 <= d <= 8.0 for d in durations) / n,
        "char_ok": 100 * sum(3.0 <= c <= 36.0 for c in chars) / n,
        "char_ideal": 100 * sum(5.0 <= c <= 20.0 for c in chars) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vad", required=True)
    ap.add_argument(
        "--arm",
        action="append",
        required=True,
        help="LABEL=path/to/aligned.json — a LABEL starting with 'whisper' picks the whisper 词表",
    )
    args = ap.parse_args()

    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))
    non_speech = [(float(a), float(b)) for a, b in vad["non_speech"]]

    print(
        f"{'arm':14s} {'cues':>5s} | {'欠切':>4s} {'≥1s':>4s} {'转折':>4s} {'不可救过切':>10s} "
        f"{'词中切':>6s} | {'形态分':>6s} {'p90':>5s} {'理想%':>6s} {'尚可%':>6s} {'不行%':>6s} {'刀在静音%':>10s}"
    )
    for spec in args.arm:
        label, path = spec.split("=", 1)
        r = analyse(
            json.loads(Path(path).read_text(encoding="utf-8")),
            non_speech,
            label,
            whisper_words=label.startswith("whisper"),
        )
        print(
            f"{r['label']:14s} {r['cues']:5d} | {r['under_hard']:4d} {r['under_forced']:4d} "
            f"{r['turn_missed']:4d} {r['over_unrecoverable']:10d} {r['midword']:6d} | {r['shape_mean']:6.3f} "
            f"{r['shape_p90']:5.2f} {r['tier_ideal']:6.0f} {r['tier_ok']:6.0f} {r['tier_bad']:6.0f} "
            f"{r['cut_certified']:10.0f}"
        )


if __name__ == "__main__":
    main()
