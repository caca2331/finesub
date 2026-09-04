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
from . import lang_audit

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
# (large-v3-turbo, B=1) residency and the 0.6B referee's peak with its
# batched decode (2.3 GiB eager at the 120 s padded cap; the compiled path's
# 2.9 GiB is gated separately by `qwen_referee.COMPILE_MIN_VRAM_GIB`). The
# same number gates warming it beside the pool (vad_asr_stage).
QWEN_REFEREE_GIB = 2.5
# Measured B=1 production residency, whole-card, load + one 30 s decode
# (`tools/bench/probe_whisper_resident.py`; docs/asr-align.md, referee device).
# Unknown/custom models stay on CPU: CT2 allocations are not visible to
# torch's CUDA counters, so guessing low can OOM while the pool is leased.
#
# `large-v3` read 6.15 until 2026-09-02, which was **the B=8 figure** off the
# wt-refine-port tier table -- that table's own B=1 baseline for it is 4.34 GB,
# and a re-measure puts B=1 at 3.89 GiB. turbo was not affected (its 2.07
# reproduced at 2.08), because asr-align had already drawn the B=1/B=8
# distinction for turbo and nobody drew it for large-v3. The correction lets
# the referee co-reside on `standard` for the big models; the margin there is
# 0.11 GiB over QWEN_REFEREE_GIB, which the eager path's measured 2.3 GiB peak
# fits and the compiled path is separately kept out of by COMPILE_MIN_VRAM_GIB.
#
# Keys are lower-cased: the lookup lower-cases the model name so an alias and
# a repository id spelled with capitals both land here.
WHISPER_RESIDENT_GIB_BY_MODEL = {
    "large-v3-turbo": 2.07,
    "large-v3": 3.89,
    # The Japanese finetune is architecturally large-v3, and measures like it
    # (3.69 GiB loaded, identical to large-v3's; 3.82 after one decode).
    "transwithai/whisper-ja-1.5b-ct2": 3.82,
}

# Extra residency per additional window in a batched decode
# (`--asr-decode-batch`), from the per-item slopes in
# docs/wt-refine-port.md's tier table. **The table above is B=1**, and the two
# are one number to a caller: at B=8 large-v3 occupies ~5.85 GiB, not 3.89.
#
# This exists because the B=1 correction removed an accidental guard. While
# `large-v3` carried the B=8 figure as if it were B=1, an opt-in batched run
# could not put the referee on the GPU -- the inflated number kept it on the
# CPU. With the honest B=1 number it would fit "on paper" and then OOM on the
# 8 GB card `standard` is written for, and CT2 answers a CUDA OOM with a
# process-level abort. Slopes are read as GiB although the source table says
# GB: that rounds residency *up*, and the safe error here is over-estimating.
WHISPER_RESIDENT_SLOPE_GIB_BY_MODEL = {
    "large-v3-turbo": 0.13,
    "large-v3": 0.28,
    "transwithai/whisper-ja-1.5b-ct2": 0.28,
}


def whisper_resident_gib(model_name: str, decode_batch: int = 1) -> Optional[float]:
    """Resident VRAM of the pool for this model at this decode batch.

    `None` for a model nobody measured -- the callers turn that into "keep the
    referee on the CPU" rather than into an estimate.
    """

    key = str(model_name).strip().lower()
    base = WHISPER_RESIDENT_GIB_BY_MODEL.get(key)
    if base is None:
        return None
    slope = WHISPER_RESIDENT_SLOPE_GIB_BY_MODEL.get(key, 0.0)
    return base + slope * max(0, int(decode_batch) - 1)

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
    decode_batch: int = 1,
    *,
    requested_device: str | None = None,
) -> str:
    """Referee placement, asked as four separate questions.

    They used to be one -- "is the ASR on CUDA" -- and that conflated things
    that stopped agreeing once the ASR stage got its own oracle:

    1. **Intent.** `--device cpu` means "leave the card alone", not "leave it
       alone for Whisper only". Pass `requested_device` so an explicit CPU run
       keeps the referee off the card even when it is idle. `None` is the
       absence of a choice and reads as the default (cuda) -- never as the
       resolved `asr_device`, which no longer says anything about intent.
    2. **Policy.** `--gpu-tier cpu` says the same thing at tier level.
    3. **Capability, of the referee's OWN backend.** The referee is a
       transformers model, so `cuda_usable()` (torch) decides -- CTranslate2's
       verdict about the ASR stage says nothing about it. This is what keeps a
       torch-unusable card from being handed a torch model just because CT2 is
       happily decoding on it.
    4. **Room.** Only a Whisper pool that is actually resident costs anything.
       When the ASR went to the CPU, the whole tier budget is free.

    `decode_batch` is the *resolved* one, not the profile's: an explicit
    `--asr-decode-batch` beats the tier table, and the pool grows with the
    value actually in force.
    """

    # `None` is "nobody chose", which is the code default: a request for the
    # card. It must NOT fall back to the *resolved* ASR device -- that reads
    # "cpu" after a CTranslate2-only fallback as well, and taking it for the
    # request pinned the referee to the CPU beside an idle card that torch
    # could use perfectly well (desktop "automatic", review 2026-09-02).
    requested = str(requested_device or "cuda").strip().lower()
    if not requested.startswith("cuda"):
        return "cpu"
    if not bool(getattr(resource_profile, "gpu", True)):
        return "cpu"
    from ..runtime.device import cuda_usable

    if not cuda_usable():
        return "cpu"
    if not str(asr_device or "").strip().lower().startswith("cuda"):
        # Whisper is not on the card, so nothing has to fit beside it.
        return "cuda"
    resident = whisper_resident_gib(model_name, decode_batch)
    if resident is None:
        return "cpu"
    spare = float(resource_profile.usable_gpu_gb) - resident
    return "cuda" if spare >= QWEN_REFEREE_GIB else "cpu"


def referee_vram_budget(
    resource_profile,
    model_name: str,
    *,
    beside_pool: bool,
    decode_batch: int = 1,
) -> float:
    """Free VRAM the referee may count on: the tier's usable figure, minus the
    resident Whisper pool while it is still loaded. What the referee's
    compiled path is gated on (`qwen_referee.COMPILE_MIN_VRAM_GIB`)."""

    usable = float(resource_profile.usable_gpu_gb)
    if not beside_pool:
        return usable
    resident = whisper_resident_gib(model_name, decode_batch)
    return usable if resident is None else usable - resident


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
        # Every group's vote, for the run-level audit — including (especially)
        # the ones that never trigger. Spans only; no audio is held.
        self._observations: List[lang_audit.Observation] = []
        self._audit: Optional[Dict[str, object]] = None

    def stats(self) -> Dict[str, object]:
        stats: Dict[str, object] = {
            "triggers": len(self.events),
            "adopted": sum(1 for event in self.events if event.get("adopted")),
            "events": list(self.events),
        }
        if self._audit is not None:
            stats["audit"] = dict(self._audit)
        return stats

    def observe(self, group: List[Dict[str, object]], language: object) -> None:
        """Remember what this group was labelled, and the span that would prove it.

        Two callers, and the second is the whole reason this is public: under
        ``--language`` there is no vote, no history and no trigger, so
        `maybe_redecode` never runs -- yet "the user forced the wrong language"
        is the one uniformly-wrong run that actually happens on purpose. The
        stage feeds those groups straight in.
        """

        language = str(language or "").strip()
        if not language or language == "None":
            return  # nothing to hold the referee's answer against
        spans = [
            span for span in (_interval_span(i) for i in group) if span is not None
        ]
        if not spans:
            return
        # The longest interval is the best single piece of evidence about which
        # language the group is in, and the cheapest to have judged.
        start, end = max(spans, key=lambda span: span[1] - span[0])
        if end - start < lang_audit.MIN_CLIP_SEC:
            return
        self._observations.append(
            lang_audit.Observation(
                start, min(end, start + lang_audit.MAX_CLIP_SEC), language
            )
        )

    def amend_last_observation(self, language: object) -> None:
        """Correct the language on the group just observed.

        `observe` runs before the redecode decision -- it has to, because most
        of `maybe_redecode`'s early returns are groups the audit specifically
        needs. But when a redecode IS adopted, the text that ships is in the
        forced language, and the ledger votes that one. Auditing the raw
        detection would then warn that the run "may be in the wrong language"
        about a language no longer present in the product.
        """

        language = str(language or "").strip()
        if not language or language == "None" or not self._observations:
            return
        last = self._observations[-1]
        self._observations[-1] = lang_audit.Observation(
            last.start, last.end, language
        )

    def observations(self) -> List[Tuple[float, float, str]]:
        """The ledger, for the ASR checkpoint to carry across a resume.

        Without this the audit after a resume samples only the groups the
        second half decoded -- a different sample, a different verdict, and
        under `MIN_ANSWERED` no audit at all. Where the interruption fell would
        decide what the run reports.
        """

        return [(o.start, o.end, o.language) for o in self._observations]

    def restore_observations(self, rows: object) -> None:
        """Counterpart of `observations`; tolerant of an older payload."""

        restored: List[lang_audit.Observation] = []
        for row in rows or ():
            try:
                start, end, language = row
                restored.append(
                    lang_audit.Observation(float(start), float(end), str(language))
                )
            except (TypeError, ValueError):
                continue
        self._observations = restored

    def run_audit(self) -> Dict[str, object]:
        """Ask the referee about an even sample of the run; warn on conflict.

        Call this while the referee is still loaded: it costs one batched
        inference over at most ``lang_audit.MAX_CLIPS`` clips, and nothing at
        all on a run too short to have a majority.
        """

        observations = self._observations
        # Too few to ever reach the answer floor, so buy no inference at all.
        if len(observations) < lang_audit.MIN_ANSWERED:
            self._audit = lang_audit.inspect(
                (), groups=len(observations), sampled=0
            ).as_dict()
            return self._audit

        chosen = [observations[i] for i in lang_audit.pick(len(observations))]
        if self._reader is None:
            self._reader = qwen_referee._SpanReader(self._audio_path)
        clips = [self._reader.read(o.start, o.end) for o in chosen]
        min_samples = int(lang_audit.MIN_CLIP_SEC * qwen_referee.TARGET_SR)
        usable = [i for i, clip in enumerate(clips) if len(clip) >= min_samples]
        # Its own span, not just `qwen.infer`: this is the one referee cost the
        # user pays on a run with nothing wrong with it, so it has to stay
        # separable from the suspect batch it would otherwise be pooled with.
        from ..runtime import phase_timing

        with phase_timing.phase("qwen.audit"):
            replies = self._referee.transcribe_batch(
                [clips[i] for i in usable],
                max_new_tokens=lang_audit.AUDIT_NEW_TOKENS,
            )

        votes: List[Tuple[float, str, Optional[str]]] = []
        for position, (text, language) in zip(usable, replies):
            observation = chosen[position]
            # Same discipline as `_vote_share`: a clip the referee heard
            # nothing usable in casts no vote, in either direction.
            code = qwen_language_code(language) if normalized_compact(text) else None
            votes.append(
                (observation.end - observation.start, observation.language, code)
            )

        self._audit = lang_audit.report(
            votes, groups=len(observations), sampled=len(chosen)
        ).as_dict()
        return self._audit

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

        # Feed the run-level audit first: it must see every group, and most of
        # this method's early returns are groups it specifically needs. An
        # unchanged history means the group cast no vote.
        if auto_language_history != history_before:
            self.observe(group, auto_language_history[-1])

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
        # The ledger now votes the forced language, so the audit's record of
        # this group has to as well -- it judges what shipped, not what the
        # detector first guessed.
        self.amend_last_observation(recent_language)
        event["adopted"] = True
        _note(
            "adopted forced-language redecode "
            f"(start={group_start:.3f}s, language={recent_language}, "
            f"sim_new={sim_new:.3f}, sim_old={sim_old:.3f}, "
            f"vote_share={share:.2f})",
            count="lang_redecode_adopted",
        )
        return redecoded
