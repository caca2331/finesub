"""Deterministic pre-merge of ASR segments (stabilize profile 3).

v2 "strong evidence" rules. The v1 premise — no punctuation + no space +
small gap means a mid-word cut — was falsified by offline evaluation:
unpunctuated Japanese ASR makes ordinary
sentence boundaries look identical (42% precision, gap=0 mostly meant
"acoustically continuous"), while true word cuts often sat at 0.4–1.2s gaps.
v2 therefore optimizes precision only: a wrong merge is irreversible (a
merged source line can never be split back downstream) while a missed merge
is recoverable by the model, so this pass merges only junctions with
positive word-form evidence and leaves judgment merging to the prompt.

The decision pipeline shares the splitter's scoring vocabulary
(``segment_split`` / docs/segment_split.md) — premerge is its dual, only
reversing boundaries split itself would score as terrible cuts:

1. Candidate gate: ``t_score(left, right, space_before) >= 1.0`` (the bare
   no-punct-no-space junction, split's worst normal cut) and gap ≤
   ``PREMERGE_REJOIN_MAX_GAP_SEC`` (1.0s — strong evidence buys a wide gap,
   mirroring g_score's "a bigger gap needs a better reason").
2. Positive evidence (any):
   - E1: right starts with a small kana / sokuon / long-vowel mark / ん —
     shapes a Japanese utterance essentially never starts with — excluding
     quotative ``って…`` starts (semantically joinable, not a word cut; left
     to the model) and ん-backchannels (ん/んー/んっ).
   - E2: right normalizes to a single kana, excluding standalone
     interjections (あ/え/お/ん/わ…).
   Kanji-junction cuts (戦いた/かったし) have no surface signature and are
   deliberately not merged (accepted recall loss).
3. Negative vetoes (double guard): left ends in a terminal form
   (です/ます/ました…), right starts with a new-sentence/response marker.
4. Result-shape guard, aligned with split's acceptable piece shape and the
   subtitle contract: merged span ≤ 7s, weighted chars ≤ 36, sources ≤ 3.

Filler attachment is direction-typed: leading fillers (あの/えっと/なんか…)
attach only forward onto the next content entry, trailing particles (ね/ねー)
only backward; response words (はい/うん/ううん/そう) are never fillers, and
a filler junction is classified before the rejoin rule so it can't be
claimed as a rejoin. Gap ≤ ``PREMERGE_FILLER_MAX_GAP_SEC`` (0.2s).

Calibration caveats:
- **Overfitting risk**: every signature, exclusion list and threshold here
  was tuned on one corpus (8 BV + yui, Japanese); treat the current rules
  as provisional until validated on held-out material.
- **Japanese-specific**: E1/E2 and the filler sets are Japanese
  optimizations. On other languages the signatures should almost never
  fire (kana shapes), so the expected failure mode is "no
  merge" rather than wrong merges — but side effects are unverified;
  audit offline (tmp/premerge_eval.py) before relying on this pass for
  non-Japanese sources.

Applied as **stabilize profile 3** (``asr_stabilize.py``), after the
hallucination-phrase cleanup but BEFORE noise tagging/dropping: word-cut
fragments are naturally low-confidence and would otherwise be misclassified
as hallucinations and dropped, destroying the word (corpus comparison in
docs/asr-stabilize.md). Split (aligned stage) marks every piece after a cut
with the segment-level ``splitted_before`` tag; premerge structurally
refuses those junctions instead of reasoning that split would not have cut
there. The junction position survives the merge as a ``premerge_before``
word-level tag on the first absorbed word (the segment boundary itself is
destroyed, so only a word can carry it); ``premerge_sources`` keeps the
input positions for audits. Rule changes bump ``PREMERGE_RULES_VERSION``
(recorded in the stable metadata) — regenerating stale stable artifacts
follows the usual delete-stage-and-downstream convention.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from segment_split import SPLIT_TAG, is_cjk, t_score
from subtitle_metrics import weighted_char_count
from utils.text import punct_class

PREMERGE_RULES_VERSION = "v2.4-no-e3"

# Word-level junction marker written onto the first absorbed word of every
# merge (see module docstring).
PREMERGE_WORD_TAG = "premerge_before"

PREMERGE_REJOIN_MAX_GAP_SEC = 1.0
PREMERGE_FILLER_MAX_GAP_SEC = 0.2
PREMERGE_MAX_MERGED_DURATION_SEC = 7.0
PREMERGE_MAX_MERGED_WEIGHTED_CHARS = 36.0
PREMERGE_MAX_MERGED_SOURCES = 3

# E1: shapes a Japanese utterance essentially never starts with.
_SMALL_KANA_START = frozenset(
    "っゃゅょぁぃぅぇぉーゎんッャュョァィゥェォヮン"
)
# ん-backchannels that only look like continuations.
_BACKCHANNEL_RIGHTS = frozenset({"ん", "んー", "んっ", "んん"})
# E2 exclusions: kana that stand alone as interjections/responses/discourse
# markers (で = "so/then" turn-opener; じゃ/ま likewise).
_INTERJECTION_SINGLES = frozenset("あえおんわはーそねでじゃま")

# Negative vetoes.
_LEFT_TERMINAL_ENDINGS = (
    "です",
    "ます",
    "ました",
    "でした",
    "ですよ",
    "ですね",
    "ますね",
    "ですか",
    "ますか",
    "だよ",
    "だね",
)
_RIGHT_NEW_SENTENCE_STARTS = (
    "でも",
    "だから",
    "なぜなら",
    "さあ",
    "いや",
    "はい",
    "うん",
    "じゃあ",
    "あと",
    "それで",
)

# Direction-typed fillers. Response words (はい/うん/ううん/そう) are never
# fillers — offline evaluation showed them to be independent turns.
PREMERGE_LEADING_FILLERS = frozenset(
    {
        "あの",
        "あのー",
        "えっと",
        "えーっと",
        "えー",
        "なんか",
        "その",
        "そのー",
        "ちょっと",
        "まあ",
        "まぁ",
    }
)
PREMERGE_TRAILING_FILLERS = frozenset({"ね", "ねー"})


def premerge_metadata() -> Dict[str, Any]:
    """Rule/threshold snapshot, recorded in the stable JSON metadata."""

    return {
        "rules_version": PREMERGE_RULES_VERSION,
        "rejoin_max_gap_sec": PREMERGE_REJOIN_MAX_GAP_SEC,
        "filler_max_gap_sec": PREMERGE_FILLER_MAX_GAP_SEC,
        "max_merged_duration_sec": PREMERGE_MAX_MERGED_DURATION_SEC,
        "max_merged_weighted_chars": PREMERGE_MAX_MERGED_WEIGHTED_CHARS,
        "max_merged_sources": PREMERGE_MAX_MERGED_SOURCES,
        "leading_fillers": sorted(PREMERGE_LEADING_FILLERS),
        "trailing_fillers": sorted(PREMERGE_TRAILING_FILLERS),
    }


def _word_token(word: Any) -> str:
    if not isinstance(word, dict):
        return ""
    return str(word.get("word") or word.get("text") or "")


def _boundary_has_space(prev: Dict[str, Any], nxt: Dict[str, Any]) -> bool:
    words = nxt.get("words")
    if isinstance(words, list):
        for word in words:
            if _word_token(word):
                return bool(word.get("space_before", False))
    # No word info: between two non-CJK junction chars a space is the default
    # word boundary — assume it exists and leave the entries alone.
    left = str(prev.get("text") or "").rstrip()
    right = str(nxt.get("text") or "").lstrip()
    return bool(
        left and right and not is_cjk(left[-1]) and not is_cjk(right[0])
    )


def _normalized(text: str) -> str:
    """Strip whitespace and leading/trailing punctuation for classification."""

    text = text.strip()
    while text and punct_class(text[0]) != "none":
        text = text[1:].lstrip()
    while text and punct_class(text[-1]) != "none":
        text = text[:-1].rstrip()
    return text


def _is_kana(ch: str) -> bool:
    o = ord(ch)
    return 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF


def _rejoin_evidence(left: str, right: str) -> Optional[str]:
    """Name of the positive word-form evidence for a rejoin, or None."""

    right_norm = _normalized(right)
    if not right_norm:
        return None
    if right_norm in _BACKCHANNEL_RIGHTS:
        return None
    if right_norm.startswith("んっ"):
        # んっ… is a vocalization/laugh shape (んっくっくー), not a
        # continuation; plain ん + content (ごめんごめ|ん、…) stays evidence.
        return None
    if right_norm.startswith("って"):
        # Quotative continuation: often semantically joinable, but that is a
        # judgment merge, not a word cut — leave it to the model.
        return None
    if right_norm[0] in _SMALL_KANA_START:
        return "E1-small-kana-start"
    if (
        len(right_norm) == 1
        and _is_kana(right_norm)
        and right_norm not in _INTERJECTION_SINGLES
    ):
        return "E2-single-kana"
    return None


def _rejoin_veto(left: str, right: str) -> Optional[str]:
    left_norm = _normalized(left)
    right_norm = _normalized(right)
    if left_norm.endswith(_LEFT_TERMINAL_ENDINGS):
        return "left-terminal-form"
    if right_norm.startswith(_RIGHT_NEW_SENTENCE_STARTS):
        return "right-new-sentence-marker"
    return None


def _filler_class(segment: Dict[str, Any]) -> Optional[str]:
    text = _normalized(str(segment.get("text") or ""))
    if text in PREMERGE_LEADING_FILLERS:
        return "leading"
    if text in PREMERGE_TRAILING_FILLERS:
        return "trailing"
    return None


def _merged_shape_block(prev: Dict[str, Any], nxt: Dict[str, Any]) -> Optional[str]:
    """Shape guard on the would-be merged entry (split-aligned bounds)."""

    span = max(float(prev["end"]), float(nxt["end"])) - float(prev["start"])
    if span > PREMERGE_MAX_MERGED_DURATION_SEC:
        return "over-max-duration"
    chars = weighted_char_count(f"{prev['text']}{nxt['text']}")
    if chars > PREMERGE_MAX_MERGED_WEIGHTED_CHARS:
        return "over-max-chars"
    sources = len(prev.get("premerge_sources") or [prev["id"]]) + 1
    if sources > PREMERGE_MAX_MERGED_SOURCES:
        return "over-max-sources"
    return None


def _merge_pair(prev: Dict[str, Any], nxt: Dict[str, Any]) -> Dict[str, Any]:
    """Merge semantics: text per space_before, span union, words appended,
    confidence min, first source id kept, source ids recorded for audit."""

    merged = dict(prev)
    merged["end"] = max(float(prev["end"]), float(nxt["end"]))
    joiner = " " if _boundary_has_space(prev, nxt) else ""
    merged["text"] = f"{prev['text']}{joiner}{nxt['text']}"
    nxt_words = [dict(word) for word in (nxt.get("words") or [])]
    if nxt_words:
        # The merged segment no longer has a boundary at the junction; the
        # first absorbed word carries its position instead.
        nxt_words[0][PREMERGE_WORD_TAG] = True
    words = list(prev.get("words") or []) + nxt_words
    if words:
        merged["words"] = words
    # Union both sides' segment tags, minus the absorbed side's positional
    # splitted_before (it described nxt's own start, now interior).
    prev_tags = [str(tag) for tag in prev.get("tags") or []]
    nxt_tags = [
        str(tag)
        for tag in nxt.get("tags") or []
        if str(tag) != SPLIT_TAG
    ]
    tags = prev_tags + [tag for tag in nxt_tags if tag not in prev_tags]
    if tags:
        merged["tags"] = tags
    confidences = [
        value
        for value in (prev.get("confidence"), nxt.get("confidence"))
        if isinstance(value, (int, float))
    ]
    if confidences:
        merged["confidence"] = min(confidences)
    sources = list(prev.get("premerge_sources") or [prev["id"]])
    merged["premerge_sources"] = sources + [nxt["id"]]
    return merged


def _junction_decision(
    prev: Dict[str, Any], segment: Dict[str, Any], gap: float
) -> Tuple[str, str]:
    """(action, detail): action is "merge"/"skip"; detail names the evidence
    for merges and the closest rejection reason for auditable skips ("" when
    the junction was never a candidate)."""

    # Structural exclusion: split deliberately created this boundary (the
    # tag is self-provenance on the right segment, so it stays correct even
    # when the original left partner was dropped by stabilization).
    right_tags = segment.get("tags")
    if isinstance(right_tags, list) and SPLIT_TAG in map(str, right_tags):
        return "skip", ""

    prev_filler = _filler_class(prev)
    cur_filler = _filler_class(segment)
    if prev_filler or cur_filler:
        # Filler junctions are classified before the rejoin rule so a filler
        # can never be claimed as a rejoin.
        if gap >= PREMERGE_FILLER_MAX_GAP_SEC:
            return "skip", ""
        if prev_filler == "leading" and cur_filler is None:
            block = _merged_shape_block(prev, segment)
            return ("skip", f"filler-blocked:{block}") if block else (
                "merge",
                "filler-forward",
            )
        if cur_filler == "trailing" and prev_filler is None:
            block = _merged_shape_block(prev, segment)
            return ("skip", f"filler-blocked:{block}") if block else (
                "merge",
                "filler-backward",
            )
        return "skip", ""

    if gap > PREMERGE_REJOIN_MAX_GAP_SEC:
        return "skip", ""
    left = str(prev.get("text") or "").rstrip()
    right = str(segment.get("text") or "").lstrip()
    if not left or not right:
        return "skip", ""
    if t_score(left, right, _boundary_has_space(prev, segment)) < 1.0:
        return "skip", ""
    evidence = _rejoin_evidence(left, right)
    if evidence is None:
        return "skip", "no-evidence"
    veto = _rejoin_veto(left, right)
    if veto is not None:
        return "skip", f"vetoed:{veto}"
    block = _merged_shape_block(prev, segment)
    if block is not None:
        return "skip", f"blocked:{block}"
    return "merge", evidence


def premerge_segments(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Single left-to-right pass over loader-normalized segment dicts.

    Each input dict must carry ``id``/``start``/``end``/``text`` (the loader
    guarantees this). Returns the merged list plus a report with per-rule
    counts, merge events (evidence-tagged), and rejected candidates (gate
    passed but evidence vetoed/blocked) for offline audits.
    """

    merged: List[Dict[str, Any]] = []
    rejoined = 0
    filler_attached = 0
    skipped_no_evidence = 0
    events: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    def _entry(prev: Dict[str, Any], seg: Dict[str, Any], gap: float, detail: str):
        return {
            "gap": gap,
            "left_id": prev["id"],
            "right_id": seg["id"],
            "left_text": prev["text"],
            "right_text": seg["text"],
            "detail": detail,
        }

    for segment in segments:
        if merged:
            prev = merged[-1]
            # Round to ms so float error can't slip an at-threshold gap
            # under the cutoff.
            gap = round(float(segment["start"]) - float(prev["end"]), 3)
            action, detail = _junction_decision(prev, segment, gap)
            if action == "merge":
                span = max(float(prev["end"]), float(segment["end"])) - float(
                    prev["start"]
                )
                event = _entry(prev, segment, gap, detail)
                event["merged_span_sec"] = round(span, 3)
                event["merged_weighted_chars"] = weighted_char_count(
                    f"{prev['text']}{segment['text']}"
                )
                merged[-1] = _merge_pair(prev, segment)
                event["merged_sources"] = list(merged[-1]["premerge_sources"])
                events.append(event)
                if detail.startswith("filler"):
                    filler_attached += 1
                else:
                    rejoined += 1
                continue
            if detail == "no-evidence":
                skipped_no_evidence += 1
            elif detail:
                rejected.append(_entry(prev, segment, gap, detail))
        merged.append(dict(segment))

    report = {
        "rules_version": PREMERGE_RULES_VERSION,
        "rejoined": rejoined,
        "filler_attached": filler_attached,
        "skipped_no_evidence": skipped_no_evidence,
        "groups": [
            list(entry["premerge_sources"])
            for entry in merged
            if entry.get("premerge_sources")
        ],
        "events": events,
        "rejected": rejected,
    }
    return merged, report
