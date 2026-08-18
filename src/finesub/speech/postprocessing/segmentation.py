"""Subtitle segmentation by global DP over the word stream (docs/segmentation-split.md).

Runs inside the vad_asr aligned-JSON stage, after overlap/zero-length
hygiene and before energy annotation. The ASR's own segmentation is not
authoritative here: the DP spans the whole clip and an ASR segment seam is
an ordinary candidate boundary, merely discounted by ``whisper_segment_bonus``.
So this pass both splits over-long segments and merges adjacent ones, and its
output segments stand in no containment relation to the ASR's.

Boundary scores prefer sentence punctuation, CJK-context spaces and (above
all) VAD-certified silences; piece scores penalize over/under-long subtitles.
Word classification and intra-segment scoring stay strictly per segment --
that invariant is what makes ``whisper_segment_bonus -> inf`` reproduce
per-segment splitting exactly. A piece the DP leaves whole is emitted from
its source segment untouched.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, replace
from typing import AbstractSet, Dict, List, Optional, Sequence, Tuple

from ..recognition.transcribe import GAP_KEEP_REAL_MAX_SEC
from ... import config as app_config
from ...subtitles.metrics import weighted_char_count
from ...text import coerce_optional_float, punct_class, words_to_text

# Lead-in kept before an interval start when folding fabricated word
# coordinates (mirrors the VAD negative padding's ~140ms speech lead-in).
SPLIT_LEAD_IN_SEC = 0.14


@dataclass(frozen=True)
class SplitParams:
    """Scoring constants (docs/segmentation-split.md). CLI-tunable via
    tools/split_explorer; production uses these defaults."""

    a: float = 1.0                    # boundary shape weight
    b: float = 1.0                    # piece urgency weight
    base: float = 1.0                 # per-cut base cost
    g_knee: float = 0.5               # gap discount acceleration knee (sec)
    non_vad_gap_penalty: float = 1.0  # cost of cutting where VAD saw no gap
    whisper_segment_bonus: float = 4.5  # discount on an ASR segment seam
    max_piece_sec: float = 20.0       # provably lossless DP predecessor bound
    dur_ideal_lo: float = 1.2
    dur_ideal_hi: float = 4.5
    dur_ok_lo: float = 0.6
    dur_ok_hi: float = 8.0
    chars_ideal_lo: float = 5.0
    chars_ideal_hi: float = 20.0
    chars_ok_lo: float = 3.0
    chars_ok_hi: float = 36.0


DEFAULT_SPLIT_PARAMS = SplitParams()
SYNTHETIC_WORD_KEY = "synthetic_from_segment"

# The one tunable meant for humans: how long is "too long". Everything else in
# SplitParams is a calibration constant whose right value comes from the gold
# set, not from taste (tune those through tools/split_explorer).
#
# The accepted range is re-exported from ``finesub.config``, which owns
# the shared file's contract: the settings panel validates a value before
# writing it and cannot import this module (torch comes with it).
LENGTH_SCALE_MIN = app_config.SPLIT_LENGTH_SCALE_MIN
LENGTH_SCALE_MAX = app_config.SPLIT_LENGTH_SCALE_MAX


def split_params_for_length_scale(
    scale: float,
    base: SplitParams = DEFAULT_SPLIT_PARAMS,
) -> SplitParams:
    """Scale the *upper* length walls (docs/segmentation-split.md), nothing else.

    Below 1.0 the DP tolerates less over-length and buys more cuts; above 1.0
    it tolerates more. Two properties make this the safe way to express the
    preference, and both are why the knob does not instead reprice a cut
    (``base``/``a`` against ``b``):

    * The ideal band still scores 0, so a subtitle already inside the target
      box is never cut for its own sake -- the pressure only ever comes from
      being outside it, and it is spent on the cheapest boundary available
      (VAD gap, then ASR seam, then punctuation, and only then a bare word).
    * ``max_piece_sec``'s losslessness proof is scale-invariant: the halving
      saving ``(ideal_hi + ok_hi) / (ok_hi - ideal_hi)`` does not depend on the
      scale, so it stays above the ``b_max = 3.5`` worst-case cut and the bound
      just travels with the walls. Repricing cuts instead has a hard ceiling at
      ratio 1.02, beyond which *no* finite bound exists (a long enough piece
      stops being worth cutting at all).

    The lower walls stay put on purpose: a subtitle too short to read is
    physics, not preference.
    """

    value = app_config.validate_split_length_scale(scale)
    if value == 1.0:
        # Identity, not an optimization: the default path must stay bit-for-bit
        # what it was before the knob existed.
        return base
    return replace(
        base,
        dur_ideal_hi=base.dur_ideal_hi * value,
        dur_ok_hi=base.dur_ok_hi * value,
        chars_ideal_hi=base.chars_ideal_hi * value,
        chars_ok_hi=base.chars_ok_hi * value,
        max_piece_sec=base.max_piece_sec * value,
    )


def split_params_metadata(params: SplitParams = DEFAULT_SPLIT_PARAMS) -> Dict[str, float]:
    return {
        # Derived rather than stored: the effective walls below are the truth,
        # and this is the one number an operator greps for.
        "length_scale": round(params.dur_ok_hi / DEFAULT_SPLIT_PARAMS.dur_ok_hi, 6),
        "a": params.a,
        "b": params.b,
        "base": params.base,
        "g_knee": params.g_knee,
        "non_vad_gap_penalty": params.non_vad_gap_penalty,
        "whisper_segment_bonus": params.whisper_segment_bonus,
        "max_piece_sec": params.max_piece_sec,
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
    coordinates (docs/segmentation-split.md: three overlap cases + text-glue
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
                # 16-sample study in docs/segmentation-split.md). Genuinely
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
    non_vad_gap: bool = False   # VAD certified no boundary here at all


@dataclass
class SplitResult:
    pieces: List[Tuple[int, int]]        # word index ranges [a, b)
    boundaries: List[Boundary]           # per word gap k = 0..n-2
    total: float
    no_split: float


def boundary_score(
    left: AdjWord,
    right: AdjWord,
    intervals: Sequence[Tuple[float, float]],
    params: SplitParams,
) -> Boundary:
    """Score a cuttable boundary. The ban check lives in ``score_boundaries``
    (which is why the ASR-seam path can call this directly: a seam must never
    come out banned, or no bonus could ever open it).

    Which yardstick applies is decided by *whether VAD certified a boundary
    here at all*, not by the gap's magnitude. ``interval_gap_between`` has two
    unrelated ways of returning 0 -- both words anchored to one interval (VAD
    gave no verdict) and a genuine interval crossing that measures 0 (abutting
    intervals). Only the former should fall back to the aligner's word-level
    pause; testing ``g > 0`` would route the latter there too, mixing the two
    yardsticks in one formula."""

    t = t_score(left.text, right.text, right.space_before)
    if right.anchor > left.anchor:
        g = interval_gap_between(intervals, left.anchor, right.anchor)
        shape = g_score(g, params)
        non_vad_gap = False
    else:
        g = 0.0
        # The pause must come from the raw timestamps: pass 3 rewrote the
        # adjusted ones against interval geometry, which would make this a
        # function of the anchoring rather than of what the decoder heard.
        pause = max(0.0, right.raw_start - left.raw_end)
        shape = max(
            -1.0,
            (params.non_vad_gap_penalty if pause <= 0.0 else 0.0) - pause * pause,
        )
        non_vad_gap = True
    return Boundary(False, g, t, params.a * (t + shape) + params.base, non_vad_gap)


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
        out.append(boundary_score(left, right, intervals, params))
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

    # Running max of word starts: non-decreasing, so the duration estimate
    # below is monotone in i and the scan can break instead of skipping. It
    # only ever under-estimates a piece's duration (start_max[i] >=
    # adj[i].start), so a predecessor is dropped only when the real piece is
    # also over the bound -- the bound stays lossless either way.
    start_max: List[float] = []
    running = -math.inf
    for w in adj:
        running = max(running, w.start)
        start_max.append(running)

    f = [math.inf] * (n + 1)
    back = [-1] * (n + 1)
    f[0] = 0.0
    for j in range(1, n + 1):
        # Second pass only if the bound left no path at all (every candidate
        # inside it banned): rare enough to pay a full scan for rather than
        # leave the DP stuck.
        for limit in (params.max_piece_sec, math.inf):
            for i in range(j - 1, -1, -1):
                # i == j-1 is always considered, so a single word longer than
                # the bound can still form its own piece.
                if i < j - 1 and adj[j - 1].end - start_max[i] > limit:
                    break
                cut = 0.0 if i == 0 else boundaries[i - 1].b
                if cut is math.inf:
                    continue
                cand = f[i] + cut + p_cost(i, j)
                if cand < f[j]:
                    f[j] = cand
                    back[j] = i
            if back[j] >= 0:
                break
    pieces: List[Tuple[int, int]] = []
    j = n
    while j > 0:
        i = back[j]
        pieces.append((i, j))
        j = i
    pieces.reverse()
    return SplitResult(pieces, list(boundaries), f[n], p_cost(0, n))


# ---------------------------------------------------------------- driver

# Word-level marker on the first word of every ASR (Whisper) source segment.
# This is the one fact about the input that the output cannot otherwise
# recover: the DP is global, so a piece may swallow a segment seam entirely
# and no segment-level field could express a boundary that now sits in a
# piece's interior. Everything else about the segmentation is derived from
# it -- the segment tag below, and the regrouping that makes a re-run a
# no-op (``regroup_by_whisper_segments``).
WHISPER_SEGMENT_WORD_TAG = "whisper_segment_start"

# Segment tag for the rare complement: a piece whose start is *not* an ASR
# segment boundary, i.e. the DP cut inside a Whisper segment. Emitting the
# positive fact instead would tag ~98% of segments (2308/2348 on the 9-clip
# bed) and carry almost no signal; the interior cuts are the 31 an auditor
# actually greps for. The positive fact stays readable off the word marker.
MID_SEGMENT_TAG = "mid_segment_start"

# Segment tag for pieces whose first word is a zone-bridging (case3) word
# anchored right purely by the default (no glue corroboration): the residual
# shape where the anchor could be wrong (e.g. a both-glued trailing particle).
# Audit trail now; candidate LLM hint later.
ANCHOR_UNCERTAIN_TAG = "split_anchor_uncertain"
ALIGNMENT_EVENTS_FIELD = "alignment_events"

# Word fields written by the gap-word adjustment; stripped when regrouping so
# a second pass classifies against zone geometry from scratch.
_ADJUST_PROVENANCE_KEYS = (
    "split_adjust_case",
    "split_anchor",
    "split_anchor_source",
    "raw_start",
    "raw_end",
)


def _inherit_segment_fields(
    sources: Sequence[Tuple[Dict[str, object], int]],
) -> Dict[str, object]:
    """Whole-segment scalars for a piece, weighted by each source segment's
    word count in it (docs/segmentation-split.md). One source -> field-for-field
    identical to plain inheritance, which is what keeps the "bonus -> inf
    reproduces per-segment splitting" equivalence exact.

    Numeric fields take the weighted mean, categorical ones the weighted
    mode (ties to the earliest source). Picking the majority source instead
    would need a tiebreak rule for a decision that is measurably inert: on
    the bed, the rule choice moves the only thing downstream reads (the
    ``low_conf`` gate in asr_stabilize) for at most one segment in 2348."""

    out: Dict[str, object] = {}
    for key in ("confidence", "no_speech_prob"):
        values = [
            (value, count)
            for segment, count in sources
            if (value := coerce_optional_float(segment.get(key))) is not None
        ]
        if not values:
            continue
        if all(value == values[0][0] for value, _ in values):
            # Short-circuit, not an optimization: regrouping hands every
            # rebuilt source the same inherited value, and re-averaging it
            # would drift in the last bits and break idempotency.
            out[key] = values[0][0]
            continue
        weight = float(sum(count for _, count in values))
        out[key] = sum(value * count for value, count in values) / weight
    votes: Dict[object, float] = {}
    order: Dict[object, int] = {}
    for index, (segment, count) in enumerate(sources):
        lang = segment.get("lang")
        if lang is None:
            continue
        votes[lang] = votes.get(lang, 0.0) + count
        order.setdefault(lang, index)
    if votes:
        out["lang"] = max(votes, key=lambda k: (votes[k], -order[k]))
    return out


def _mark_seam_word(segment: Dict[str, object]) -> Dict[str, object]:
    """Copy of ``segment`` whose first word carries the seam marker. Used for
    pieces the DP left whole, which are emitted from the source dict so their
    coordinates stay untouched."""

    piece = dict(segment)
    words = [dict(word) for word in segment.get("words") or []]
    if words:
        words[0][WHISPER_SEGMENT_WORD_TAG] = True
        piece["words"] = words
    return piece


def _build_piece_segment(
    sources: Sequence[Tuple[Dict[str, object], int]],
    adj: Sequence[AdjWord],
    a: int,
    b: int,
    seam_words: AbstractSet[int],
) -> Optional[Dict[str, object]]:
    words: List[Dict[str, object]] = []
    for i in range(a, b):
        adj_word = adj[i]
        word = dict(adj_word.source)
        word["start"] = adj_word.start
        word["end"] = adj_word.end
        if i in seam_words:
            word[WHISPER_SEGMENT_WORD_TAG] = True
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
    # Clamp piece times inside the union of the source segments it covers, so
    # folded gap-word coordinates can never resurrect cross-segment overlaps.
    # min/max rather than first.start/last.end: costs nothing and does not
    # assume the sources are time-ordered (clamp_segment_overlaps guarantees
    # that upstream in production, but archived artifacts do contain overlaps).
    seg_start = min(float(segment.get("start", 0.0)) for segment, _ in sources)
    seg_end = max(float(segment.get("end", 0.0)) for segment, _ in sources)
    start = min(max(adj[a].start, seg_start), seg_end)
    end = min(max(adj[b - 1].end, start), seg_end)
    piece: Dict[str, object] = {
        "start": start,
        "end": end,
        "words": words,
        "text": text,
    }
    piece.update(_inherit_segment_fields(sources))
    return piece


def _event_anchor(
    event: Dict[str, object],
    source: Dict[str, object],
) -> float:
    """Choose one absolute-timeline owner point for a refine event.

    Events describe evidence rather than subtitle text, so a global split may
    cut through their span.  Keeping each event exactly once is more useful to
    downstream routing than copying it onto every derived subtitle piece.
    """

    for key in ("refined_start", "peak_time"):
        value = coerce_optional_float(event.get(key))
        if value is not None:
            return value
    start = coerce_optional_float(event.get("start"))
    end = coerce_optional_float(event.get("end"))
    if start is not None and end is not None:
        return (start + end) / 2.0
    for value in (
        start,
        coerce_optional_float(event.get("original_start")),
        end,
    ):
        if value is not None:
            return value
    source_start = coerce_optional_float(source.get("start")) or 0.0
    source_end = coerce_optional_float(source.get("end"))
    return (
        (source_start + source_end) / 2.0
        if source_end is not None
        else source_start
    )


def _redistribute_alignment_events(
    sources: Sequence[Dict[str, object]],
    pieces: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Carry refine events through global re-segmentation without duplicates."""

    output = [dict(piece) for piece in pieces]
    for piece in output:
        piece.pop(ALIGNMENT_EVENTS_FIELD, None)
    if not output:
        return output

    piece_spans = [
        (
            coerce_optional_float(piece.get("start")) or 0.0,
            coerce_optional_float(piece.get("end")) or 0.0,
        )
        for piece in output
    ]
    for source in sources:
        events = source.get(ALIGNMENT_EVENTS_FIELD)
        if not isinstance(events, list):
            continue
        source_start = coerce_optional_float(source.get("start"))
        source_end = coerce_optional_float(source.get("end"))
        eligible = [
            index
            for index, (start, end) in enumerate(piece_spans)
            if source_start is None
            or source_end is None
            or (end >= source_start and start <= source_end)
        ]
        if not eligible:
            eligible = list(range(len(output)))
        for raw_event in events:
            if not isinstance(raw_event, dict):
                continue
            event = dict(raw_event)
            anchor = _event_anchor(event, source)

            def distance(index: int) -> tuple[float, float]:
                start, end = piece_spans[index]
                outside = max(start - anchor, anchor - end, 0.0)
                return outside, abs((start + end) / 2.0 - anchor)

            owner = min(eligible, key=distance)
            output[owner].setdefault(ALIGNMENT_EVENTS_FIELD, []).append(event)
    return output


def regroup_by_whisper_segments(
    segments: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Undo a previous split back to the ASR's own segmentation.

    ``split_segments`` is only well-defined on ASR-native segments:
    ``adjust_words`` runs per segment and its glue runs treat the word list's
    edges as separators, so re-splitting its own output would score interior
    boundaries differently (measured: 17 of 25297 flip, several between
    ``inf`` and a large negative). Regrouping first makes a re-run a no-op.

    Both halves of the restore need the word level. The grouping comes from
    ``WHISPER_SEGMENT_WORD_TAG`` -- a piece that swallowed a seam holds it in
    its interior, where no segment-level tag could reach. The coordinates come
    from ``raw_start``/``raw_end``, which pass 3 persisted precisely for the
    words it moved; without them pass 1 would re-classify already-folded
    coordinates against the zones."""

    groups: List[List[Dict[str, object]]] = []
    donors: List[Dict[str, object]] = []
    for segment in segments:
        for word in segment.get("words") or []:
            restored = {
                k: v for k, v in word.items() if k not in _ADJUST_PROVENANCE_KEYS
            }
            if "raw_start" in word:
                restored["start"] = word["raw_start"]
            if "raw_end" in word:
                restored["end"] = word["raw_end"]
            if restored.pop(WHISPER_SEGMENT_WORD_TAG, None) or not groups:
                groups.append([])
                # Whole-segment scalars are aggregates; a rebuilt group takes
                # them from the piece its first word came from.
                donors.append(segment)
            groups[-1].append(restored)
    if not groups:
        return list(segments)

    out: List[Dict[str, object]] = []
    for group, donor in zip(groups, donors):
        rebuilt: Dict[str, object] = {
            "start": float(group[0]["start"]),
            "end": float(group[-1]["end"]),
            "words": group,
            "text": words_to_text(group),
        }
        for key in ("lang", "confidence", "no_speech_prob"):
            if key in donor:
                rebuilt[key] = donor[key]
        out.append(rebuilt)
    return out


def split_segments(
    segments: Sequence[Dict[str, object]],
    intervals: Sequence[Dict[str, object]],
    *,
    params: SplitParams = DEFAULT_SPLIT_PARAMS,
) -> List[Dict[str, object]]:
    """Re-partition the ASR's segments at the globally lowest-score cut set.

    ``intervals`` are the VAD speech intervals on the same timeline. The DP
    spans the whole clip: every ASR segment seam is a candidate boundary like
    any other, discounted by ``whisper_segment_bonus`` so it is kept unless
    the surrounding evidence is clearly against it. Word classification and
    intra-segment scoring stay strictly per segment -- that is what makes
    ``whisper_segment_bonus -> inf`` reproduce per-segment splitting exactly.

    Already-split input is regrouped first, so re-running is a no-op."""

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

    normalized_segments = [_ensure_minimum_word(segment) for segment in segments]
    event_sources = list(normalized_segments)

    if any(
        word.get(WHISPER_SEGMENT_WORD_TAG)
        for segment in normalized_segments
        for word in segment.get("words") or []
    ):
        normalized_segments = regroup_by_whisper_segments(normalized_segments)

    # Per-segment classification and scoring, concatenated into one stream.
    adj: List[AdjWord] = []
    bounds: List[Boundary] = []
    sources: List[Dict[str, object]] = []
    starts: List[int] = []                      # first word index per source
    for seg in normalized_segments:
        seg_adj = adjust_words(seg.get("words") or [], interval_spans, zones)
        if not seg_adj:
            # An invalid segment cannot participate in a word-level global DP,
            # and joining the surrounding segments across it would reorder or
            # hide content. Prefer leaving the whole input untouched.
            return normalized_segments
        if adj:
            # The seam is scored fresh rather than reused: score_boundaries is
            # the only thing that produces `banned`, and a banned seam would be
            # one no bonus could ever open.
            seam = boundary_score(adj[-1], seg_adj[0], interval_spans, params)
            bounds.append(
                Boundary(
                    False,
                    seam.g,
                    seam.t,
                    seam.b - params.whisper_segment_bonus,
                    seam.non_vad_gap,
                )
            )
        if len(seg_adj) >= 2:
            bounds.extend(score_boundaries(seg_adj, interval_spans, params))
        starts.append(len(adj))
        adj.extend(seg_adj)
        sources.append(seg)
    if len(adj) < 2:
        return normalized_segments
    ends = starts[1:] + [len(adj)]
    seam_words = set(starts)

    result = dp_split(adj, bounds, params)

    out: List[Dict[str, object]] = []
    prev_source = -1
    for a, b in result.pieces:
        first = bisect.bisect_right(starts, a) - 1
        last = bisect.bisect_right(starts, b - 1) - 1
        whole = first == last and a == starts[first] and b == ends[first]
        if whole:
            # The piece is exactly one source segment: emit it untouched, so a
            # segment the DP chose not to cut keeps its original coordinates
            # (the gap-word adjustment stays virtual, as it always has).
            piece = _mark_seam_word(sources[first])
        else:
            covered = [
                (sources[i], min(b, ends[i]) - max(a, starts[i]))
                for i in range(first, last + 1)
                if min(b, ends[i]) > max(a, starts[i])
            ]
            piece = _build_piece_segment(covered, adj, a, b, seam_words)
            if piece is None:
                continue
        if (
            out
            and not whole
            and first == prev_source
            and float(piece["end"]) <= float(piece["start"])
        ):
            # Degenerate piece: fold its words into the previous one. Only
            # within a source segment -- a whole segment that is itself
            # zero-length is the ASR's output, not a cut this pass made, and
            # passing it through is what the per-segment scope always did.
            prev = out[-1]
            prev_words = list(prev.get("words") or []) + list(piece["words"])
            prev["words"] = prev_words
            prev["text"] = words_to_text(prev_words)
            prev["end"] = max(float(prev["end"]), float(piece["end"]))
            continue
        prev_source = last
        tags = [str(tag) for tag in piece.get("tags") or []]
        if a not in seam_words and MID_SEGMENT_TAG not in tags:
            tags.append(MID_SEGMENT_TAG)
        head = (piece.get("words") or [{}])[0]
        if (
            head.get("split_adjust_case") == "case3"
            and head.get("split_anchor") == "right"
            and head.get("split_anchor_source") == "default"
            and ANCHOR_UNCERTAIN_TAG not in tags
        ):
            tags.append(ANCHOR_UNCERTAIN_TAG)
        if tags:
            piece["tags"] = tags
        out.append(piece)
    return _redistribute_alignment_events(event_sources, out or normalized_segments)


def _ensure_minimum_word(segment: Dict[str, object]) -> Dict[str, object]:
    """Return a safe word-level representation for text-only ASR segments.

    A segment without word timestamps cannot be split meaningfully. Represent
    its whole text as one explicitly synthetic word spanning the segment so
    downstream word-based code preserves it without pretending to have a real
    alignment.
    """

    if segment.get("words"):
        return segment
    text = str(segment.get("text") or "").strip()
    start = coerce_optional_float(segment.get("start"))
    end = coerce_optional_float(segment.get("end"))
    if not text or start is None or end is None or end <= start:
        return segment
    normalized = dict(segment)
    normalized["words"] = [
        {
            "word": text,
            "start": start,
            "end": end,
            SYNTHETIC_WORD_KEY: True,
        }
    ]
    return normalized
