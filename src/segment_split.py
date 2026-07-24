"""DP-based splitting of over-long ASR segments (docs/segment_split.md).

Runs inside the vad_asr aligned-JSON stage, after overlap/zero-length
hygiene and before energy annotation. Whole-whisper-segment mapping keeps
decoder segments intact; this pass re-splits the over-long ones at the
lowest-total-score partition: boundary scores prefer sentence punctuation,
CJK-context spaces and (above all) VAD-certified silences; piece scores
penalize over/under-long subtitles. Segments whose optimal partition is
"no split" are returned unchanged (bit-identical).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from asr_align import GAP_KEEP_REAL_MAX_SEC
from subtitle_metrics import weighted_char_count
from utils import coerce_optional_float, words_to_text
from utils.text import punct_class

# Lead-in kept before an interval start when folding fabricated word
# coordinates (mirrors the VAD negative padding's ~140ms speech lead-in).
SPLIT_LEAD_IN_SEC = 0.14


@dataclass(frozen=True)
class SplitParams:
    """Scoring constants (docs/segment_split.md). CLI-tunable via
    tools/split_explorer; production uses these defaults."""

    a: float = 1.0                    # boundary shape weight
    b: float = 1.0                    # piece urgency weight
    base: float = 1.0                 # per-cut base cost
    g_knee: float = 0.5               # gap discount acceleration knee (sec)
    no_gap_penalty: float = 1.0       # extra cost for non-VAD-gap cuts
    dur_ideal_lo: float = 1.2
    dur_ideal_hi: float = 4.5
    dur_ok_lo: float = 0.6
    dur_ok_hi: float = 8.0
    chars_ideal_lo: float = 5.0
    chars_ideal_hi: float = 20.0
    chars_ok_lo: float = 3.0
    chars_ok_hi: float = 36.0


DEFAULT_SPLIT_PARAMS = SplitParams()


def split_params_metadata(params: SplitParams = DEFAULT_SPLIT_PARAMS) -> Dict[str, float]:
    return {
        "a": params.a,
        "b": params.b,
        "base": params.base,
        "g_knee": params.g_knee,
        "no_gap_penalty": params.no_gap_penalty,
        "dur_ideal": [params.dur_ideal_lo, params.dur_ideal_hi],
        "dur_ok": [params.dur_ok_lo, params.dur_ok_hi],
        "chars_ideal": [params.chars_ideal_lo, params.chars_ideal_hi],
        "chars_ok": [params.chars_ok_lo, params.chars_ok_hi],
        "lead_in_sec": SPLIT_LEAD_IN_SEC,
    }


# ---------------------------------------------------------------- scoring

def g_score(g: float, params: SplitParams) -> float:
    if g <= 0:
        return 0.0
    if g <= params.g_knee:
        return -g
    return -(g * g) / params.g_knee


def is_cjk(ch: str) -> bool:
    """Han / kana / hangul: scripts without intra-clause spaces."""

    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF      # hiragana + katakana
        or 0x31F0 <= o <= 0x31FF   # katakana ext
        or 0x3400 <= o <= 0x4DBF   # CJK ext A
        or 0x4E00 <= o <= 0x9FFF   # CJK unified
        or 0xF900 <= o <= 0xFAFF   # CJK compat
        or 0xAC00 <= o <= 0xD7AF   # hangul
        or 0xFF66 <= o <= 0xFF9D   # half-width katakana
    )


def t_score(left_text: str, right_text: str, right_space_before: bool) -> float:
    left = left_text.rstrip()
    right = right_text.lstrip()
    lc = punct_class(left[-1]) if left else "none"
    rc = punct_class(right[0]) if right else "none"
    # anti-tier: stranding an opener at a piece end / a closer at a piece
    # start is worse than a bare boundary
    if lc == "opening" or rc == "closing":
        return 1.5
    if lc == "sentence":
        return 0.0
    if lc in ("clause", "closing"):
        return 0.2
    if rc == "opening":
        return 0.3
    if lc == "other" or rc in ("sentence", "clause", "other"):
        return 0.5
    if right_space_before:
        left_cjk = bool(left) and is_cjk(left[-1])
        right_cjk = bool(right) and is_cjk(right[0])
        if left_cjk and right_cjk:
            return 0.2
        if left_cjk or right_cjk:
            return 0.5
    return 1.0


def piece_score(c: float, d: float, params: SplitParams) -> float:
    dur_long = (
        max(0.0, d - params.dur_ideal_hi) + max(0.0, d - params.dur_ok_hi)
    ) / (params.dur_ok_hi - params.dur_ideal_hi)
    chars_long = (
        max(0.0, c - params.chars_ideal_hi) + max(0.0, c - params.chars_ok_hi)
    ) / (params.chars_ok_hi - params.chars_ideal_hi)
    dur_short = (
        max(0.0, params.dur_ideal_lo - d) + max(0.0, params.dur_ok_lo - d)
    ) / (params.dur_ideal_lo - params.dur_ok_lo) / 2.0
    chars_short = (
        max(0.0, params.chars_ideal_lo - c) + max(0.0, params.chars_ok_lo - c)
    ) / (params.chars_ideal_lo - params.chars_ok_lo) / 2.0
    return params.b * (dur_long + chars_long + dur_short + chars_short)


# ------------------------------------------------- zones & word adjustment

@dataclass
class AdjWord:
    source: Dict[str, object]  # original word dict (never mutated)
    text: str
    start: float               # adjusted
    end: float                 # adjusted
    space_before: bool
    case: str                  # "real" | "case1" | "case2L" | "case2R" | "case3"
    anchor: int                # interval index the word belongs to
    zone: int = -1             # artificial-zone id (left interval index), -1 = real
    raw_start: float = 0.0
    raw_end: float = 0.0
    anchor_source: str = "default"  # "default" | "glue" (pass-2 override)


def build_zones(
    intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float, int]]:
    """Per gap i: the artificial zone (az_start, az_end, left_idx) with
    az_start = interval_i.end + min(gap, GAP_KEEP_REAL_MAX_SEC); exists only
    when the gap exceeds the kept-real-audio allowance."""

    zones: List[Tuple[float, float, int]] = []
    for i in range(len(intervals) - 1):
        e_i = intervals[i][1]
        s_j = intervals[i + 1][0]
        gap = s_j - e_i
        az_start = e_i + min(max(gap, 0.0), GAP_KEEP_REAL_MAX_SEC)
        if s_j - az_start > 1e-6:
            zones.append((az_start, s_j, i))
    return zones


def locate_interval(intervals: Sequence[Tuple[float, float]], t: float) -> int:
    """Index of the interval containing t, or the nearest one left of t."""

    lo, hi = 0, len(intervals) - 1
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if intervals[mid][0] <= t:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def boundary_is_separator(
    left_text: str,
    right_text: str,
    right_space_before: bool,
) -> bool:
    """Text-glue separator between two adjacent words.

    Opening punctuation on the left / closing on the right forces glue;
    other punctuation on either side separates; a space separates only when
    at least one neighbor is CJK (between two non-CJK words a space is the
    default word boundary, not an attachment signal)."""

    left = left_text.rstrip()
    right = right_text.lstrip()
    lc = punct_class(left[-1]) if left else "none"
    rc = punct_class(right[0]) if right else "none"
    if lc == "opening" or rc == "closing":
        return False
    if lc in ("sentence", "clause", "closing", "other"):
        return True
    if rc in ("sentence", "clause", "opening", "other"):
        return True
    if right_space_before:
        both_non_cjk = (
            bool(left) and bool(right)
            and not is_cjk(left[-1]) and not is_cjk(right[0])
        )
        return not both_non_cjk
    return False


def adjust_words(
    words: Sequence[Dict[str, object]],
    intervals: Sequence[Tuple[float, float]],
    zones: Sequence[Tuple[float, float, int]],
) -> List[AdjWord]:
    """Classify words against artificial zones and realize adjusted
    coordinates (docs/segment_split.md: three overlap cases + text-glue
    anchoring overrides; multi-word runs move as glued blocks)."""

    zone_by_left = {li: (az_s, az_e) for az_s, az_e, li in zones}
    out: List[AdjWord] = []
    # Pass 1: classify each word and record its case-based default anchor.
    for w in words:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", 0.0))
        text = str(w.get("word", ""))
        sb = bool(w.get("space_before", False))
        case = "real"
        anchor = -1
        zone = -1
        for az_s, az_e, li in zones:
            if we <= az_s or ws >= az_e:
                continue
            ri = li + 1
            in_left = ws < az_s
            in_right = we > az_e
            if in_left and in_right:
                # case 3: bridges the whole artificial zone; default anchor =
                # RIGHT. The old "longer real-audio contact" heuristic
                # systematically mis-anchored wt's stretch pathology (a word
                # whose raw span pads leftward across the pause always
                # touches more left audio, yet is almost always the right
                # group's first word: 真|ん中, カ|ウントダウン, 隣|から —
                # 16-sample study in docs/segment_split.md). Genuinely
                # left-belonging tails (…です。) are glued-left/separated-
                # right and pass 2 anchors them left before the default
                # matters; the residual both-glued trailing-particle shape
                # (…だよ|ね) is accepted and auditable via the
                # split_anchor_uncertain tag.
                case, anchor = "case3", ri
            elif in_left:
                case, anchor = "case2L", li
            elif in_right:
                case, anchor = "case2R", ri
            else:
                # case 1: fully artificial; default = prefix of the right
                # interval's first word.
                case, anchor = "case1", ri
            zone = li
            break
        if case == "real":
            # words in an interval or its kept-real gap audio: adopt as-is.
            # Anchor by the larger interval overlap (a word straddling gap
            # audio into the next interval is that interval's lead-in word);
            # words fully inside gap audio anchor left (they are A's tail).
            idx = locate_interval(intervals, ws)
            anchor = idx
            if idx + 1 < len(intervals):
                left_ov = max(
                    0.0, min(we, intervals[idx][1]) - max(ws, intervals[idx][0])
                )
                right_ov = max(
                    0.0,
                    min(we, intervals[idx + 1][1]) - max(ws, intervals[idx + 1][0]),
                )
                if right_ov > left_ov:
                    anchor = idx + 1
        out.append(AdjWord(w, text, ws, we, sb, case, anchor, zone, ws, we))

    # Pass 2: text-glue override per artificial-zone run. Words glued (no
    # separator chain) to the left neighbor anchor left; glued to the right
    # neighbor anchor right; the rest keep their case default. A run glued
    # through on both sides (no separator anywhere) stays on defaults.
    idx = 0
    n = len(out)
    while idx < n:
        if out[idx].zone < 0:
            idx += 1
            continue
        run_start = idx
        li = out[idx].zone
        while idx < n and out[idx].zone == li:
            idx += 1
        run_end = idx  # exclusive
        run = out[run_start:run_end]
        # separators at the len(run)+1 boundaries of: prev | g1 | ... | gn | next
        seps: List[bool] = []
        for b in range(len(run) + 1):
            if b == 0 and run_start == 0:
                seps.append(True)  # segment edge acts as a break
                continue
            if b == len(run) and run_end == n:
                seps.append(True)
                continue
            left_word = out[run_start + b - 1]
            right_word = out[run_start + b]
            seps.append(
                boundary_is_separator(
                    left_word.text, right_word.text, right_word.space_before
                )
            )
        if not any(seps):
            continue  # glued straight through: keep case defaults
        # left-glued prefix
        k = 0
        while k < len(run) and not seps[k]:
            run[k].anchor = li
            run[k].anchor_source = "glue"
            k += 1
        # right-glued suffix
        k = len(run) - 1
        while k >= 0 and not seps[k + 1]:
            run[k].anchor = li + 1
            run[k].anchor_source = "glue"
            k -= 1

    # Pass 3: realize adjusted coordinates from the final anchor.
    for w in out:
        if w.zone < 0:
            continue
        az_s, _az_e = zone_by_left[w.zone]
        ri = w.zone + 1
        s_ri = intervals[ri][0]
        if w.anchor == w.zone:
            # attached to A's tail: end at the kept-real-audio edge; a real
            # start is adopted, a fabricated one folds into the real tail.
            w.end = az_s
            w.start = w.raw_start if w.raw_start < az_s else az_s - SPLIT_LEAD_IN_SEC
        else:
            # attached to B: real end is adopted, a fabricated one folds to
            # B's start; start folds into the lead-in.
            w.start = min(max(w.raw_start, s_ri - SPLIT_LEAD_IN_SEC), s_ri)
            w.end = w.raw_end if w.raw_end > s_ri else s_ri
    return out


def interval_gap_between(
    intervals: Sequence[Tuple[float, float]], i: int, j: int
) -> float:
    """Total silence between interval i and j (raw VAD-gap yardstick)."""

    if j <= i:
        return 0.0
    total = intervals[j][0] - intervals[i][1]
    for k in range(i + 1, j):
        total -= intervals[k][1] - intervals[k][0]
    return max(0.0, total)


# ---------------------------------------------------------------- DP core

@dataclass
class Boundary:
    banned: bool
    g: float
    t: float
    b: float
    no_gap: bool = False


@dataclass
class SplitResult:
    pieces: List[Tuple[int, int]]        # word index ranges [a, b)
    boundaries: List[Boundary]           # per word gap k = 0..n-2
    total: float
    no_split: float


def score_boundaries(
    adj: Sequence[AdjWord],
    intervals: Sequence[Tuple[float, float]],
    params: SplitParams,
) -> List[Boundary]:
    out: List[Boundary] = []
    for k in range(len(adj) - 1):
        left, right = adj[k], adj[k + 1]
        # never cut between a gap word and the side it is anchored to
        left_bound_right = left.zone >= 0 and left.anchor == left.zone + 1
        right_bound_left = right.zone >= 0 and right.anchor == right.zone
        if left_bound_right or right_bound_left:
            out.append(Boundary(True, 0.0, 0.0, math.inf))
            continue
        g = interval_gap_between(intervals, left.anchor, right.anchor)
        t = t_score(left.text, right.text, right.space_before)
        b = params.a * (t + g_score(g, params)) + params.base
        no_gap = g <= 0.0
        if no_gap:
            b += params.no_gap_penalty
        out.append(Boundary(False, g, t, b, no_gap))
    return out


def piece_text(adj: Sequence[AdjWord], a: int, b: int) -> str:
    parts: List[str] = []
    for i in range(a, b):
        if parts and adj[i].space_before:
            parts.append(" ")
        parts.append(adj[i].text)
    return "".join(parts).strip()


def dp_split(
    adj: Sequence[AdjWord],
    boundaries: Sequence[Boundary],
    params: SplitParams,
) -> SplitResult:
    n = len(adj)
    char_w = [
        weighted_char_count(w.text) + (0.5 if w.space_before else 0.0) for w in adj
    ]
    prefix = [0.0]
    for cw in char_w:
        prefix.append(prefix[-1] + cw)

    def p_cost(a: int, b: int) -> float:
        d = adj[b - 1].end - adj[a].start
        c = prefix[b] - prefix[a]
        return piece_score(c, d, params)

    f = [math.inf] * (n + 1)
    back = [-1] * (n + 1)
    f[0] = 0.0
    for j in range(1, n + 1):
        for i in range(0, j):
            cut = 0.0 if i == 0 else boundaries[i - 1].b
            if cut is math.inf:
                continue
            cand = f[i] + cut + p_cost(i, j)
            if cand < f[j]:
                f[j] = cand
                back[j] = i
    pieces: List[Tuple[int, int]] = []
    j = n
    while j > 0:
        i = back[j]
        pieces.append((i, j))
        j = i
    pieces.reverse()
    return SplitResult(pieces, list(boundaries), f[n], p_cost(0, n))


# ---------------------------------------------------------------- driver

# Segment-level provenance tag on every piece after the first: "this
# segment's start was created by a split cut". It describes the segment
# itself, not its current neighbor, so it stays valid even if the left
# partner is later dropped by stabilization; premerge (stabilize profile 3)
# uses it to structurally refuse re-joining boundaries split created on
# purpose (a real word can never span a dropped middle segment, so keeping
# the tag after neighbor deletion is still the correct exclusion).
SPLIT_TAG = "splitted_before"

# Segment tag for pieces whose first word is a zone-bridging (case3) word
# anchored right purely by the default (no glue corroboration): the residual
# shape where the anchor could be wrong (e.g. a both-glued trailing particle).
# Audit trail now; candidate LLM hint later.
ANCHOR_UNCERTAIN_TAG = "split_anchor_uncertain"


def _build_piece_segment(
    source_segment: Dict[str, object],
    adj: Sequence[AdjWord],
    a: int,
    b: int,
    seg_start: float,
    seg_end: float,
) -> Optional[Dict[str, object]]:
    words: List[Dict[str, object]] = []
    for i in range(a, b):
        adj_word = adj[i]
        word = dict(adj_word.source)
        word["start"] = adj_word.start
        word["end"] = adj_word.end
        if adj_word.case != "real":
            # Gap-word adjustment provenance: persist the zone case, the
            # final anchor side and its source, and — when the coordinates
            # were actually moved — the raw word times, so artifact forensics
            # (e.g. "was this word misplaced by wt or by the zone
            # anchoring?") never has to reconstruct VAD geometry.
            word["split_adjust_case"] = adj_word.case
            word["split_anchor"] = (
                "left" if adj_word.anchor == adj_word.zone else "right"
            )
            word["split_anchor_source"] = adj_word.anchor_source
            if (
                adj_word.raw_start != adj_word.start
                or adj_word.raw_end != adj_word.end
            ):
                word["raw_start"] = adj_word.raw_start
                word["raw_end"] = adj_word.raw_end
        words.append(word)
    text = words_to_text(words)
    if not text:
        return None
    # clamp piece times inside the source segment bounds so folded gap-word
    # coordinates can never resurrect cross-segment overlaps
    start = min(max(adj[a].start, seg_start), seg_end)
    end = min(max(adj[b - 1].end, start), seg_end)
    piece: Dict[str, object] = {
        "start": start,
        "end": end,
        "words": words,
        "text": text,
    }
    # pieces inherit the source segment's whole-segment metrics (dilution
    # documented in docs/segment_split.md)
    for key in ("lang", "confidence", "no_speech_prob"):
        if key in source_segment:
            piece[key] = source_segment[key]
    return piece


def split_segments(
    segments: Sequence[Dict[str, object]],
    intervals: Sequence[Dict[str, object]],
    *,
    params: SplitParams = DEFAULT_SPLIT_PARAMS,
) -> List[Dict[str, object]]:
    """Split over-long segments at the DP-optimal partition.

    ``intervals`` are the VAD speech intervals on the same timeline.
    Segments whose optimal partition is a single piece are passed through
    unchanged (bit-identical output for them)."""

    interval_spans: List[Tuple[float, float]] = []
    for item in intervals:
        s = coerce_optional_float(item.get("start"))
        e = coerce_optional_float(item.get("end"))
        if s is None or e is None or e <= s:
            continue
        interval_spans.append((s, e))
    interval_spans.sort()
    if not interval_spans:
        return list(segments)
    zones = build_zones(interval_spans)

    out: List[Dict[str, object]] = []
    for seg in segments:
        words = seg.get("words") or []
        if len(words) < 2:
            out.append(seg)
            continue
        adj = adjust_words(words, interval_spans, zones)
        if len(adj) < 2:
            out.append(seg)
            continue
        boundaries = score_boundaries(adj, interval_spans, params)
        result = dp_split(adj, boundaries, params)
        if len(result.pieces) <= 1:
            out.append(seg)
            continue
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        built: List[Dict[str, object]] = []
        for a, b in result.pieces:
            piece = _build_piece_segment(seg, adj, a, b, seg_start, seg_end)
            if piece is None:
                continue
            if built and float(piece["end"]) <= float(piece["start"]):
                # degenerate piece: fold its words into the previous piece
                prev = built[-1]
                prev_words = list(prev.get("words") or []) + list(piece["words"])
                prev["words"] = prev_words
                prev["text"] = words_to_text(prev_words)
                prev["end"] = max(float(prev["end"]), float(piece["end"]))
                continue
            built.append(piece)
        if not built:
            out.append(seg)
            continue
        for piece in built[1:]:
            tags = [str(tag) for tag in piece.get("tags") or []]
            if SPLIT_TAG not in tags:
                piece["tags"] = tags + [SPLIT_TAG]
        for piece in built:
            first = (piece.get("words") or [{}])[0]
            if (
                first.get("split_adjust_case") == "case3"
                and first.get("split_anchor") == "right"
                and first.get("split_anchor_source") == "default"
            ):
                tags = [str(tag) for tag in piece.get("tags") or []]
                if ANCHOR_UNCERTAIN_TAG not in tags:
                    piece["tags"] = tags + [ANCHOR_UNCERTAIN_TAG]
        out.extend(built)
    return out
