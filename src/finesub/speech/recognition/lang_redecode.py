"""Inline language-vote-flip redecode (docs/asr-align.md).

A window whose prompt context is polluted can collapse into a foreign-language
hallucination while Whisper's language detection stays on the true language
(asr-align.md, root cause) — so the fix is not detection-side. When a group's collapsed
language vote contradicts the recent cross-group majority, this module asks
the Qwen referee what the group's VAD intervals actually contain, redecodes
the group with the majority language forced (a fresh ``align_group`` call, so
the polluted prompt context is gone), and adopts the redecode only when the
referee evidence sides with it. Every failure direction — no evidence, no
progress, low similarity — keeps the original decode.

Wired in by the vad-asr stage behind ``--lang-redecode`` (default auto);
``align_segments`` calls :meth:`LangRedecoder.maybe_redecode` once per group
and stays inert when no redecoder is passed.
"""

from __future__ import annotations

import difflib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...text import normalized_compact
from ..verification import qwen_referee

# Adoption margin is a dead zone, not a classification boundary:
# when in doubt keep the original decode. The true-positive side measured
# >= +0.400 (asr-align.md, measurements); recalibrate against real
# foreign-language negatives before changing either threshold.
SIM_MARGIN = 0.15
# Duration-weighted share of referee interval votes that must back the
# majority language for the vote branch (asr-align.md, adoption).
VOTE_SHARE_MIN = 2.0 / 3.0
# Same floor as apply_verification: shorter clips read as "no speech heard"
# and must not reach the model.
MIN_CLIP_SEC = 0.05

# Referee-device decision (asr-align.md): the Whisper pool must stay resident
# during the inline check, so the referee goes to the GPU only when the
# profile's spare VRAM fits it. Both numbers are measured: production-model
# (large-v3-turbo, B=1) residency and the 0.6B referee's bf16 peak.
QWEN_REFEREE_GIB = 1.5
# Measured B=1 production residency (docs/asr-align.md, referee device).
# Unknown/custom models stay on CPU: CT2 allocations are not visible to
# torch's CUDA counters, so guessing low can OOM while the pool is leased.
WHISPER_RESIDENT_GIB_BY_MODEL = {
    "large-v3-turbo": 2.07,
    "large-v3": 6.15,
}

# Qwen reports language names ("Japanese"), Whisper records codes ("ja").
# Charset classification was rejected for this job — shared CJK ideographs
# misclassify real Japanese as Chinese — so the referee's own
# language field is the vote, mapped here.
_QWEN_LANG_CODES = {
    "arabic": "ar",
    "cantonese": "yue",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "hindi": "hi",
    "indonesian": "id",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "malay": "ms",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "thai": "th",
    "turkish": "tr",
    "ukrainian": "uk",
    "urdu": "ur",
    "vietnamese": "vi",
}

AlignFn = Callable[..., Tuple[List[Dict[str, object]], List[Dict[str, object]]]]


def qwen_language_code(name: object) -> Optional[str]:
    """Whisper-style code for a Qwen language reply; None when unknown."""

    text = str(name or "").strip().lower().replace("-", "_")
    if not text:
        return None
    if len(text) <= 3:
        return text
    return _QWEN_LANG_CODES.get(text)


def referee_device(
    asr_device: str,
    resource_profile,
    model_name: str,
) -> str:
    """Referee placement while the Whisper pool stays loaded."""

    device = str(asr_device or "").strip().lower()
    if not device.startswith("cuda"):
        return "cpu"
    resident = WHISPER_RESIDENT_GIB_BY_MODEL.get(str(model_name).strip().lower())
    if resident is None:
        return "cpu"
    spare = float(resource_profile.usable_gpu_gb) - resident
    return str(asr_device) if spare >= QWEN_REFEREE_GIB else "cpu"


def _similarity(a: str, b: str) -> float:
    """Pinned metric: SequenceMatcher ratio over compacted text."""

    return difflib.SequenceMatcher(
        None, normalized_compact(a), normalized_compact(b)
    ).ratio()


def _segments_text(segments: List[Dict[str, object]]) -> str:
    return "".join(str(segment.get("text") or "") for segment in segments)


def _interval_span(interval: Dict[str, object]) -> Optional[Tuple[float, float]]:
    try:
        start = float(interval.get("start", 0.0))
        end = float(interval.get("end", 0.0))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


class LangRedecoder:
    """Per-run state for the inline check: referee, thresholds, event log."""

    def __init__(
        self,
        referee,
        audio_path: str,
        *,
        margin: float = SIM_MARGIN,
        vote_share_min: float = VOTE_SHARE_MIN,
    ) -> None:
        self._referee = referee
        self._audio_path = str(audio_path)
        self._reader = None
        self._margin = float(margin)
        self._vote_share_min = float(vote_share_min)
        self.events: List[Dict[str, object]] = []

    def stats(self) -> Dict[str, object]:
        return {
            "triggers": len(self.events),
            "adopted": sum(1 for event in self.events if event.get("adopted")),
            "events": list(self.events),
        }

    def _probe_intervals(
        self, group: List[Dict[str, object]]
    ) -> List[Dict[str, object]]:
        """Referee evidence per VAD interval — interval spans, not segment
        spans: hallucinated segment geometry is wrong in both directions
        (asr-align.md, tradeoffs)."""

        if self._reader is None:
            self._reader = qwen_referee._SpanReader(self._audio_path)
        spans = [
            span for span in (_interval_span(i) for i in group) if span is not None
        ]
        clips = [self._reader.read(start, end) for start, end in spans]
        min_samples = int(MIN_CLIP_SEC * qwen_referee.TARGET_SR)
        usable = [i for i, clip in enumerate(clips) if len(clip) >= min_samples]
        replies = self._referee.transcribe_batch([clips[i] for i in usable])
        results: List[Tuple[str, Optional[str]]] = [("", None)] * len(clips)
        for position, reply in zip(usable, replies):
            results[position] = reply
        return [
            {
                "start": start,
                "end": end,
                "duration": end - start,
                "text": text,
                "language": qwen_language_code(language),
            }
            for (start, end), (text, language) in zip(spans, results)
        ]

    def _vote_share(
        self, evidences: List[Dict[str, object]], language: str
    ) -> Tuple[float, float]:
        """(share of ``language`` among votes, voted seconds). Evidence-empty
        intervals cast no vote."""

        voted = [e for e in evidences if e["text"] and e["language"]]
        total = sum(float(e["duration"]) for e in voted)
        if total <= 0.0:
            return 0.0, 0.0
        backing = sum(
            float(e["duration"]) for e in voted if e["language"] == language
        )
        return backing / total, total

    def maybe_redecode(
        self,
        *,
        align_fn: AlignFn,
        model,
        group: List[Dict[str, object]],
        segments: List[Dict[str, object]],
        audio: Optional[np.ndarray],
        sr: int,
        gap_sec: float,
        auto_language_history: List[str],
        history_before: List[str],
        recent_language: Optional[str],
        audio_loader=None,
        tail_real_limit_sec: float,
    ) -> List[Dict[str, object]]:
        """One inline check for a decoded group; returns the segments to keep.

        ``history_before`` is the history snapshot taken before the group's
        decode; the group's collapsed vote (if any) is the entry appended
        since. Adoption rolls the history back to the snapshot and casts one
        majority-language vote instead — the one-vote-per-group ledger must
        not keep the hallucinated language.
        """

        # Trigger: the truthiness guard is part of the criterion —
        # an empty history means "no majority to contradict", not a mismatch.
        if recent_language is None:
            return segments
        if auto_language_history == history_before:
            return segments  # group cast no vote (no usable detection)
        group_language = auto_language_history[-1]
        if group_language == recent_language:
            return segments

        from .transcribe import _note, _trim_language_history

        group_start = float(group[0].get("start", 0.0)) if group else 0.0
        _note(
            "language mismatch vs recent majority; asking referee "
            f"(start={group_start:.3f}s, detected={group_language}, "
            f"majority={recent_language})",
            count="lang_redecode_triggers",
        )
        evidences = self._probe_intervals(group)
        q_text = "\n".join(e["text"] for e in evidences if e["text"])
        event: Dict[str, object] = {
            "start": round(group_start, 3),
            "detected": group_language,
            "majority": recent_language,
            "intervals": len(evidences),
            "adopted": False,
        }
        self.events.append(event)

        # Hard gate: with no referee evidence at all, no redecode
        # can be adjudicated — pure laughter/sighs land here and must stay.
        if not normalized_compact(q_text):
            event["rejected"] = "no-evidence"
            _note(
                f"referee heard nothing usable; keeping original decode "
                f"(start={group_start:.3f}s)",
                count="lang_redecode_no_evidence",
            )
            return segments

        # Forced-language redecode via the production entry point, prompt
        # context fresh by construction. The rescue ladder inside
        # may hand intervals back; consume them here — the outer loop's queue
        # advancement was already fixed by the original decode.
        redecoded: List[Dict[str, object]] = []
        pending = list(group)
        while pending:
            new_segments, unconsumed = align_fn(
                pending,
                audio,
                sr,
                model=model,
                gap_sec=gap_sec,
                language=recent_language,
                auto_language_history=None,
                audio_loader=audio_loader,
                tail_real_limit_sec=tail_real_limit_sec,
            )
            if len(unconsumed) >= len(pending):
                # Same stall the outer loop guards against; here the safe exit
                # is keeping the original decode, not killing the run.
                event["rejected"] = "redecode-stalled"
                _note(
                    "forced-language redecode made no progress; keeping "
                    f"original decode (start={group_start:.3f}s)",
                    count="lang_redecode_stalled",
                )
                return segments
            redecoded.extend(new_segments)
            pending = unconsumed

        # Second hard gate: adopting an empty redecode would delete the
        # group's content, and deletion is outside this feature's boundary
        # (replacement only, never deletion). The similarity branch fails naturally
        # on empty text, but the vote branch never looks at the redecode's
        # output and would adopt it blindly.
        if not normalized_compact(_segments_text(redecoded)):
            event["rejected"] = "redecode-empty"
            _note(
                "forced-language redecode came back empty; keeping original "
                f"decode (start={group_start:.3f}s)",
                count="lang_redecode_empty",
            )
            return segments

        sim_new = _similarity(q_text, _segments_text(redecoded))
        sim_old = _similarity(q_text, _segments_text(segments))
        share, voted_sec = self._vote_share(evidences, recent_language)
        event.update(
            {
                "sim_new": round(sim_new, 3),
                "sim_old": round(sim_old, 3),
                "vote_share": round(share, 3),
                "voted_sec": round(voted_sec, 3),
            }
        )

        adopt = sim_new > sim_old + self._margin or (
            voted_sec > 0.0 and share > self._vote_share_min
        )
        if not adopt:
            event["rejected"] = "evidence-disagrees"
            _note(
                "referee evidence does not favor the redecode; keeping "
                f"original (start={group_start:.3f}s, sim_new={sim_new:.3f}, "
                f"sim_old={sim_old:.3f}, vote_share={share:.2f})",
                count="lang_redecode_rejected",
            )
            return segments

        # Roll the ledger back to the pre-group snapshot and cast one
        # majority-language vote for this group: the forced
        # decode itself neither reads nor writes the history.
        auto_language_history[:] = history_before
        auto_language_history.append(recent_language)
        _trim_language_history(auto_language_history)
        event["adopted"] = True
        _note(
            "adopted forced-language redecode "
            f"(start={group_start:.3f}s, language={recent_language}, "
            f"sim_new={sim_new:.3f}, sim_old={sim_old:.3f}, "
            f"vote_share={share:.2f})",
            count="lang_redecode_adopted",
        )
        return redecoded
