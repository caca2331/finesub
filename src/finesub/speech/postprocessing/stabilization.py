"""Stabilize aligned ASR JSON before SRT and LLM consumers."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Iterable
import unicodedata

from ...reporting import current_reporter, reporting_to, terminal_reporter
from ...subtitles.metrics import weighted_char_count
from ...text import COMMON_HALLUCINATION_TEXT, normalized_compact


DEFAULT_ASR_STABILIZE_PROFILE = 0
SUPPORTED_ASR_STABILIZE_PROFILES = (-1, 0, 1, 2)

MAX_HALLUCINATION_WORDS = 5

TAG_HIGHLY_SUSPECTED_HALLUCINATION = "高度疑似幻觉"
TAG_HIGHLY_SUSPECTED_FILLER = "高度疑似语气填充词"
TAG_PHRASE_GHOST = "套话幽灵"
TAG_LANG_SWITCH_HALLUCINATION = "语言切换幻觉"
TAG_TIME_DRIFT = "时间漂移"
TAG_ORDER = (
    TAG_HIGHLY_SUSPECTED_HALLUCINATION,
    TAG_HIGHLY_SUSPECTED_FILLER,
    TAG_PHRASE_GHOST,
    TAG_LANG_SWITCH_HALLUCINATION,
    TAG_TIME_DRIFT,
)

# Closing-phrase ghosts: a segment that IS one of Whisper's stock closing
# phrases, squeezed into a physically impossible duration. Whole corpus
# audit (74 artifacts + 400-window sweep + references, 2026-08-05): every
# squeezed occurrence was a hallucination; the one confirmed real occurrence
# (an end-of-stream thanks, verified by Qwen re-recognition) ran at normal
# speed with ~2x margin. Confidence does NOT separate real from hallucinated
# (overlapping 0.16-0.999), so the offline gate is rate-only; normal-rate
# occurrences are only droppable with second-model evidence (see
# _is_verified_closing_phrase_ghost). Longer real sentences merely containing
# a phrase are excluded by the whole-segment length bound.
CLOSING_GHOST_PHRASES = ("おわり", "それではまた", "ありがとうございました")
CLOSING_GHOST_MAX_EXTRA_CHARS = 2
CLOSING_GHOST_MIN_CHARS_PER_SEC = 20.0

# Language-switch suspicion: mostly-Latin low-confidence segments inside a
# CJK-dominant run. Observation-only (see the discard-set note below): the
# wide-corpus review found the matches mix true hallucinations with real
# English lyrics/dubs and translation-mode renderings of real speech, so this
# marks segments for downstream consumers instead of dropping them.
LANG_SWITCH_RUN_MAX_LATIN_RATIO = 0.3
LANG_SWITCH_SEGMENT_MIN_LATIN_RATIO = 0.7
LANG_SWITCH_SEGMENT_MIN_LETTERS = 8
LANG_SWITCH_MAX_CONFIDENCE = 0.6

# The very-low-energy hallucination legs must not fire when the decoder was
# highly confident on every word: with a squeezed or drifted timeline the
# energy is sampled at the wrong audio, and the human-audited failure mode of
# those legs is deleting real speech whose timing collapsed (words quantized
# to 20ms points land in silence). Confident hallucinations remain covered by
# the phrase cleanup and the word-level repetition rules upstream.
VERY_LOW_ENERGY_DROP_WORD_CONFIDENCE_EXEMPT = 0.9
# ...except at the measurement floor: audited drift victims measured -24 to
# -68 dB (real speech nearby), while confident hallucinations over absolute
# digital silence sit at the -100 dB floor and stay droppable.
VERY_LOW_ENERGY_EXEMPT_FLOOR_DB = -80.0


@dataclass
class AsrStabilizeReport:
    profile: int
    applied_profiles: tuple[int, ...]
    input_segments: int
    output_segments: int = 0
    phrase_occurrences_removed: int = 0
    phrase_segments_changed: int = 0
    emptied_segments: int = 0
    tag_counts: dict[str, int] = field(
        default_factory=lambda: {tag: 0 for tag in TAG_ORDER}
    )
    suspicious_segments_dropped: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "applied_profiles": list(self.applied_profiles),
            "input_segments": self.input_segments,
            "output_segments": self.output_segments,
            "phrase_occurrences_removed": self.phrase_occurrences_removed,
            "phrase_segments_changed": self.phrase_segments_changed,
            "emptied_segments": self.emptied_segments,
            "tag_counts": dict(self.tag_counts),
            "suspicious_segments_dropped": self.suspicious_segments_dropped,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stabilize aligned ASR JSON into stable JSON."
    )
    parser.add_argument("input", help="Path to *-aligned.json.")
    parser.add_argument("-o", "--output", help="Path to *-stable.json.")
    parser.add_argument(
        "--profile",
        type=int,
        choices=SUPPORTED_ASR_STABILIZE_PROFILES,
        default=DEFAULT_ASR_STABILIZE_PROFILE,
        help=(
            "ASR stabilize profile: -1 no-op; 0 default (1->2->drop); "
            "1 hallucination phrase cleanup; 2 noise tags."
        ),
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith("-aligned.json"):
        return input_path.with_name(
            input_path.name[: -len("-aligned.json")] + "-stable.json"
        )
    base = input_path.with_suffix("")
    return base.with_name(f"{base.name}-stable.json")


def _coerce_finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _word_text(word: dict[str, object]) -> str:
    value = word.get("word")
    if value is None:
        value = word.get("text")
    return str(value or "")


def _words_to_text(words: Iterable[dict[str, object]]) -> str:
    parts: list[str] = []
    for word in words:
        token = _word_text(word)
        if not token:
            continue
        if parts and bool(word.get("space_before", False)):
            parts.append(" ")
        parts.append(token)
    return "".join(parts).strip()


def _render_words_with_owners(
    words: list[dict[str, object]],
) -> tuple[str, list[int | None], list[tuple[int, int]]]:
    parts: list[str] = []
    owners: list[int | None] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    has_text = False
    for index, word in enumerate(words):
        token = _word_text(word)
        if token and has_text and bool(word.get("space_before", False)):
            parts.append(" ")
            owners.append(None)
            cursor += 1
        start = cursor
        parts.append(token)
        owners.extend([index] * len(token))
        cursor += len(token)
        spans.append((start, cursor))
        has_text = has_text or bool(token)
    return "".join(parts), owners, spans


def _eligible_phrase_deletions(
    text: str,
    owners: list[int | None],
) -> list[tuple[int, int]]:
    deletions: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(COMMON_HALLUCINATION_TEXT, cursor)
        if start < 0:
            break
        phrase_end = start + len(COMMON_HALLUCINATION_TEXT)
        word_indices = {
            owner for owner in owners[start:phrase_end] if owner is not None
        }
        if word_indices and len(word_indices) <= MAX_HALLUCINATION_WORDS:
            end = phrase_end
            while end < len(text) and unicodedata.category(text[end]).startswith("P"):
                end += 1
            deletions.append((start, end))
        cursor = phrase_end
    return deletions


def _cleanup_common_hallucination(
    segment: dict[str, object],
) -> tuple[dict[str, object] | None, int, bool]:
    raw_words = segment.get("words")
    if not isinstance(raw_words, list) or not raw_words:
        return segment, 0, False
    words = [dict(word) for word in raw_words if isinstance(word, dict)]
    if not words:
        return segment, 0, False

    text, _owners, spans = _render_words_with_owners(words)
    deletions = _eligible_phrase_deletions(text, _owners)
    if not deletions:
        return segment, 0, False

    deleted = [False] * len(text)
    for start, end in deletions:
        for index in range(start, end):
            deleted[index] = True

    kept_words: list[tuple[int, dict[str, object]]] = []
    original_nonempty = [index for index, word in enumerate(words) if _word_text(word)]
    for index, (word, (start, end)) in enumerate(zip(words, spans)):
        token = _word_text(word)
        kept = "".join(
            char for offset, char in enumerate(token, start) if not deleted[offset]
        )
        if not kept:
            continue
        updated_word = dict(word)
        updated_word["word"] = kept
        updated_word.pop("text", None)
        kept_words.append((index, updated_word))

    if not kept_words:
        return None, len(deletions), True

    kept_words[0][1]["space_before"] = False
    updated = dict(segment)
    updated_word_values = [word for _index, word in kept_words]
    updated["words"] = updated_word_values
    updated["text"] = _words_to_text(updated_word_values)

    if original_nonempty:
        if kept_words[0][0] > original_nonempty[0]:
            new_start = _coerce_finite_float(kept_words[0][1].get("start"))
            if new_start is not None:
                updated["start"] = new_start
        if kept_words[-1][0] < original_nonempty[-1]:
            new_end = _coerce_finite_float(kept_words[-1][1].get("end"))
            if new_end is not None:
                updated["end"] = new_end
    return updated, len(deletions), True


def _apply_profile_1(
    segments: list[dict[str, object]], report: AsrStabilizeReport
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for segment in segments:
        updated, removed, changed = _cleanup_common_hallucination(segment)
        report.phrase_occurrences_removed += removed
        if changed:
            report.phrase_segments_changed += 1
        if updated is None:
            report.emptied_segments += 1
            continue
        output.append(updated)
    return output


def weighted_word_confidence(segment: dict[str, object]) -> float | None:
    words = segment.get("words")
    if not isinstance(words, list):
        return None
    weighted_sum = 0.0
    total_weight = 0.0
    for word in words:
        if not isinstance(word, dict):
            continue
        confidence = _coerce_finite_float(word.get("confidence"))
        if confidence is None:
            continue
        weight = weighted_char_count(_word_text(word))
        if weight <= 0:
            continue
        weighted_sum += confidence * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None


def _without_unicode_punctuation(text: str) -> str:
    return "".join(
        char for char in text if not unicodedata.category(char).startswith("P")
    )


def _latin_letter_stats(text: str) -> tuple[int, float]:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0, 0.0
    latin = sum(1 for char in letters if "LATIN" in unicodedata.name(char, ""))
    return len(letters), latin / len(letters)


def _run_is_cjk_dominant(segments: list[dict[str, object]]) -> bool:
    letters, latin_ratio = _latin_letter_stats(
        "".join(str(segment.get("text") or "") for segment in segments)
    )
    return letters > 0 and latin_ratio < LANG_SWITCH_RUN_MAX_LATIN_RATIO


def _closing_phrase_of(text: str) -> str | None:
    """The stock phrase a whole-segment text amounts to, or None."""

    compact = normalized_compact(text)
    for phrase in CLOSING_GHOST_PHRASES:
        if (
            phrase in compact
            and len(compact) <= len(phrase) + CLOSING_GHOST_MAX_EXTRA_CHARS
        ):
            return phrase
    return None


def _is_closing_phrase_ghost(text: str, duration: float | None) -> bool:
    if duration is None or duration <= 0:
        return False
    phrase = _closing_phrase_of(text)
    return (
        phrase is not None
        and duration < len(phrase) / CLOSING_GHOST_MIN_CHARS_PER_SEC
    )


def _qwen_verify_text(segment: dict[str, object]) -> str | None:
    """Normalized second-model evidence text, or None when absent.

    Produced by ``speech.verification.qwen_referee`` at the vad-asr tail;
    empty string means Qwen heard no speech in the segment's span.
    """

    verify = segment.get("qwen_verify")
    if not isinstance(verify, dict):
        return None
    return normalized_compact(str(verify.get("text") or ""))


def _is_verified_closing_phrase_ghost(segment: dict[str, object]) -> bool:
    """Normal-rate stock phrase whose audio, per the second model, does not
    contain the phrase. The 67-clip audit measured 11/11 on this criterion;
    shout blindness does not apply to the polysyllabic phrase family."""

    phrase = _closing_phrase_of(str(segment.get("text") or ""))
    if phrase is None:
        return False
    evidence = _qwen_verify_text(segment)
    return evidence is not None and phrase not in evidence


def _is_lang_switch_hallucination(segment: dict[str, object]) -> bool:
    letters, latin_ratio = _latin_letter_stats(str(segment.get("text") or ""))
    if letters < LANG_SWITCH_SEGMENT_MIN_LETTERS:
        return False
    if latin_ratio < LANG_SWITCH_SEGMENT_MIN_LATIN_RATIO:
        return False
    confidence = _coerce_finite_float(segment.get("confidence"))
    return confidence is not None and confidence < LANG_SWITCH_MAX_CONFIDENCE


def _profile_2_tags(
    segment: dict[str, object],
    *,
    run_cjk_dominant: bool = False,
) -> list[str]:
    text = str(segment.get("text") or "")
    start = _coerce_finite_float(segment.get("start"))
    end = _coerce_finite_float(segment.get("end"))
    duration = end - start if start is not None and end is not None else None
    rate = (
        (weighted_char_count(text) - 2.0) / duration
        if duration is not None and duration > 0
        else None
    )
    high_speed = rate is not None and rate > 20.0

    segment_confidence = _coerce_finite_float(segment.get("confidence"))
    word_confidence = weighted_word_confidence(segment)
    low_conf = (
        segment_confidence is not None
        and word_confidence is not None
        and segment_confidence < 0.3
        and word_confidence < 0.3
    )

    energy = _coerce_finite_float(segment.get("vad_weighted_energy_db"))
    low_energy = energy is not None and energy < 0.0
    very_low_energy = energy is not None and energy < -20.0
    stripped_length = weighted_char_count(_without_unicode_punctuation(text))

    energy_exempt = (
        word_confidence is not None
        and word_confidence > VERY_LOW_ENERGY_DROP_WORD_CONFIDENCE_EXEMPT
        and energy is not None
        and energy > VERY_LOW_ENERGY_EXEMPT_FLOOR_DB
    )
    highly_suspected_hallucination = (
        not energy_exempt
        and (
            (duration is not None and duration > 0.1 and very_low_energy)
            or (stripped_length <= 2.0 and very_low_energy)
        )
    ) or (low_conf and low_energy)
    highly_suspected_filler = (
        low_conf
        and energy is not None
        and not low_energy
        and stripped_length <= 2.0
    )
    time_drift = high_speed or low_conf or low_energy

    # Second-model veto: when Qwen heard speech in the span, the noise-leg
    # drops stand down (the drop audit caught real shouts deleted at
    # positive energy). The rate-based phrase-ghost leg is not vetoable —
    # its criterion is physical impossibility of the timing, not silence.
    verify_text = _qwen_verify_text(segment)
    if verify_text:
        highly_suspected_hallucination = False
        highly_suspected_filler = False

    tags: list[str] = []
    if highly_suspected_hallucination:
        tags.append(TAG_HIGHLY_SUSPECTED_HALLUCINATION)
    if highly_suspected_filler:
        tags.append(TAG_HIGHLY_SUSPECTED_FILLER)
    if _is_closing_phrase_ghost(text, duration) or _is_verified_closing_phrase_ghost(
        segment
    ):
        tags.append(TAG_PHRASE_GHOST)
    if run_cjk_dominant and _is_lang_switch_hallucination(segment):
        tags.append(TAG_LANG_SWITCH_HALLUCINATION)
    if time_drift:
        tags.append(TAG_TIME_DRIFT)
    return tags


def _apply_profile_2(
    segments: list[dict[str, object]], report: AsrStabilizeReport
) -> list[dict[str, object]]:
    run_cjk_dominant = _run_is_cjk_dominant(segments)
    output: list[dict[str, object]] = []
    for segment in segments:
        updated = dict(segment)
        existing = updated.get("tags")
        existing_tags = (
            [str(tag) for tag in existing] if isinstance(existing, list) else []
        )
        detected = _profile_2_tags(updated, run_cjk_dominant=run_cjk_dominant)
        for tag in detected:
            report.tag_counts[tag] += 1
        combined = existing_tags + detected
        tags = [tag for tag in TAG_ORDER if tag in combined]
        tags.extend(
            tag for tag in existing_tags if tag not in TAG_ORDER and tag not in tags
        )
        if tags:
            updated["tags"] = tags
        else:
            updated.pop("tags", None)
        output.append(updated)
    return output


def _drop_suspicious_segments(
    segments: list[dict[str, object]], report: AsrStabilizeReport
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    # 语言切换幻觉 is deliberately NOT in this set: wide-corpus review found
    # real English content (sung lyrics, English game-PV dubs kept by the
    # human-refined reference) and translation-mode hallucinations of real
    # Japanese speech among the matches — deletion would lose real content or
    # the only trace of it. The tag stays observational; the proper fix for
    # translation-mode output is a language-forced re-decode, not deletion.
    discard_tags = {
        TAG_HIGHLY_SUSPECTED_HALLUCINATION,
        TAG_HIGHLY_SUSPECTED_FILLER,
        TAG_PHRASE_GHOST,
    }
    for segment in segments:
        tags = segment.get("tags")
        if isinstance(tags, list) and discard_tags.intersection(map(str, tags)):
            report.suspicious_segments_dropped += 1
            continue
        output.append(segment)
    return output


def stabilize_payload(
    payload: dict[str, object],
    *,
    profile: int = DEFAULT_ASR_STABILIZE_PROFILE,
) -> tuple[dict[str, object], AsrStabilizeReport]:
    if profile not in SUPPORTED_ASR_STABILIZE_PROFILES:
        expected = ", ".join(str(item) for item in SUPPORTED_ASR_STABILIZE_PROFILES)
        raise ValueError(
            f"Unsupported ASR stabilize profile: {profile}; expected one of {expected}"
        )
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Aligned JSON must contain a 'segments' list.")
    if any(not isinstance(segment, dict) for segment in raw_segments):
        raise ValueError("Every aligned JSON segment must be an object.")

    applied_profiles = (
        () if profile == -1 else ((1, 2) if profile == 0 else (profile,))
    )
    report = AsrStabilizeReport(
        profile=profile,
        applied_profiles=applied_profiles,
        input_segments=len(raw_segments),
    )
    result = copy.deepcopy(payload)
    segments = [dict(segment) for segment in result["segments"]]  # type: ignore[index]
    if profile in (0, 1):
        segments = _apply_profile_1(segments, report)
    if profile in (0, 2):
        segments = _apply_profile_2(segments, report)
    if profile == 0:
        segments = _drop_suspicious_segments(segments, report)
    result["segments"] = segments
    report.output_segments = len(segments)
    return result, report


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(data)
        handle.flush()
    try:
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def stabilize_json_file(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    profile: int = DEFAULT_ASR_STABILIZE_PROFILE,
) -> tuple[Path, AsrStabilizeReport]:
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input not found: {source}")
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else default_output_path(source)
    )
    raw = source.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid aligned JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Aligned JSON root must be an object.")

    result, report = stabilize_payload(payload, profile=profile)
    if profile == -1:
        rendered = raw
    else:
        rendered = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
    _atomic_write_bytes(output, rendered)
    return output, report


def run_asr_stabilize(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    profile: int = DEFAULT_ASR_STABILIZE_PROFILE,
) -> Path:
    output, report = stabilize_json_file(
        input_path,
        output_path=output_path,
        profile=profile,
    )
    current_reporter().summary(
        "stable",
        {
            "段": f"{report.input_segments} -> {report.output_segments}",
            "移除短语": report.phrase_occurrences_removed,
            "丢弃可疑段": report.suspicious_segments_dropped,
            "标记": sum(report.tag_counts.values()) if report.tag_counts else 0,
        },
    )
    current_reporter().debug(
        "asr stabilize",
        {"profile": profile, "tags": report.tag_counts},
    )
    return output


def main() -> int:
    args = parse_args()
    try:
        with reporting_to(terminal_reporter()):
            run_asr_stabilize(
                args.input, output_path=args.output, profile=args.profile
            )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
