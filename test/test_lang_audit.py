"""Run-level language audit: the external anchor `lang_redecode` lacks.

The bug being guarded against is not a wrong threshold, it is a wrong *shape*:
`lang_redecode`'s trigger compares each group's language against the rolling
majority of the same quantity, so a systematically wrong majority produces no
contradiction and the detector never fires. Measured on a synthetic ja+en mix:
89.7% of the Japanese labelled `en`, **zero** triggers
(`docs/bench-baselines.md` 15.3).

So the load-bearing test here is not "does it warn when told to" but
`test_the_case_lang_redecode_is_blind_to`: a run where every group agrees with
every other group, which is exactly what makes the relative predicate silent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.speech.recognition import lang_audit, lang_redecode


# --- what the flag buys -------------------------------------------------------
#
# All four cells, because "wired but never reached" is the failure this whole
# batch of guards exists for, and a check that only runs in a mode nobody uses
# is indistinguishable from no check at all.


@pytest.mark.parametrize(
    "flag,language,expected",
    [
        ("auto", None, "redecode"),          # today's default, unchanged
        ("on", None, "redecode+audit"),      # explicit opt-in adds the anchor
        ("on", "ja", "audit-only"),          # the trigger is inert, the audit is not
        ("auto", "ja", None),                # nothing to do: buy no referee
        ("off", None, None),
        ("off", "ja", None),
        ("auto", "", "redecode"),            # empty string is not a forced language
    ],
)
def test_the_flag_resolves_to_one_mode(flag, language, expected) -> None:
    assert lang_audit.resolve_mode(flag, language) == expected


def test_the_default_run_still_buys_nothing_new() -> None:
    """The audit costs a referee load plus clips; `auto` must not pay it."""

    assert "audit" not in (lang_audit.resolve_mode("auto", None) or "")


# --- the sample ---------------------------------------------------------------


def test_short_runs_are_sampled_whole() -> None:
    assert lang_audit.pick(5, max_clips=8) == [0, 1, 2, 3, 4]


def test_the_sample_is_capped() -> None:
    assert len(lang_audit.pick(400, max_clips=8)) == 8


def test_the_sample_spans_the_whole_run() -> None:
    """Stratified midpoints: no clustering at either end."""

    chosen = lang_audit.pick(80, max_clips=8)

    assert chosen == sorted(set(chosen))
    assert chosen[0] < 80 // 8  # something from the first stratum
    assert chosen[-1] >= 80 - 80 // 8  # ...and from the last


def test_the_sample_is_deterministic() -> None:
    """A randomised sample would make the same file audit differently twice."""

    assert lang_audit.pick(137, max_clips=8) == lang_audit.pick(137, max_clips=8)


def test_no_observations_no_sample() -> None:
    assert lang_audit.pick(0) == []


# --- the verdict --------------------------------------------------------------


def _votes(pairs, duration=10.0):
    return [(duration, whisper, referee) for whisper, referee in pairs]


def test_the_case_lang_redecode_is_blind_to() -> None:
    """Every group says `en`; the referee hears Japanese in all of it.

    The relative predicate sees a perfectly consistent run. This one does not.
    """

    verdict = lang_audit.inspect(
        _votes([("en", "ja")] * 8), groups=60, sampled=8
    )

    assert verdict.suspect
    assert verdict.whisper_majority == "en"
    assert verdict.referee_majority == "ja"
    assert "en" in verdict.reason and "ja" in verdict.reason


def test_agreement_is_quiet() -> None:
    verdict = lang_audit.inspect(_votes([("ja", "ja")] * 8), groups=60, sampled=8)

    assert not verdict.suspect
    assert verdict.as_dict()["agreement"] == 1.0


def test_a_minority_disagreement_does_not_fire() -> None:
    """Genuine code-switching is not a broken run, and this must not claim it is.

    Three of eight clips disagreeing is a bilingual recording. The majority is
    still right, and there is no calibrated rate that would let us say more.
    """

    verdict = lang_audit.inspect(
        _votes([("ja", "ja")] * 5 + [("ja", "en")] * 3), groups=60, sampled=8
    )

    assert not verdict.suspect
    assert verdict.as_dict()["agreement"] == pytest.approx(5 / 8)


def test_the_majority_is_duration_weighted_not_clip_counted() -> None:
    """One long clip outweighs several short ones, on both sides."""

    votes = [(60.0, "en", "ja")] + [(2.0, "en", "en")] * 5

    verdict = lang_audit.inspect(votes, groups=60, sampled=6)

    assert verdict.suspect
    assert verdict.referee_majority == "ja"


def test_abstentions_are_not_disagreements() -> None:
    """The referee has no code for Welsh; that is silence, not a vote for `en`."""

    votes = [(10.0, "cy", None)] * 6 + [(10.0, "cy", "cy")] * 3

    verdict = lang_audit.inspect(votes, groups=60, sampled=9)

    assert not verdict.suspect
    assert verdict.answered == 3
    assert verdict.as_dict()["agreement"] == 1.0


def test_too_few_answers_states_nothing(monkeypatch) -> None:
    votes = [(10.0, "en", "ja")] * 2 + [(10.0, "en", None)] * 6

    verdict = lang_audit.inspect(votes, groups=60, sampled=8)

    assert not verdict.suspect
    assert "answered 2/8" in verdict.reason


def test_a_short_run_with_enough_answers_still_audits() -> None:
    """Five groups is a 163s file of continuous speech, not a non-sample.

    An earlier draft required six groups and skipped exactly that file.
    """

    verdict = lang_audit.inspect(_votes([("en", "ja")] * 5), groups=5, sampled=5)

    assert verdict.suspect


def test_red_the_verdict_needs_both_majorities() -> None:
    """With nothing answered there is no referee majority to conflict with."""

    verdict = lang_audit.inspect([(10.0, "en", None)] * 8, groups=60, sampled=8)

    assert not verdict.suspect
    assert verdict.referee_majority is None


# --- the warning --------------------------------------------------------------


class RecordingReporter:
    def __init__(self) -> None:
        self.warnings = []

    def warning(self, code, message, *, impact="", action="") -> None:
        self.warnings.append((code, message, impact, action))


@pytest.fixture()
def reporter(monkeypatch):
    recorder = RecordingReporter()
    monkeypatch.setattr("finesub.reporting.current_reporter", lambda: recorder)
    return recorder


def test_a_conflict_warns_and_names_the_escape_hatch(reporter) -> None:
    lang_audit.report(_votes([("en", "ja")] * 8), groups=60, sampled=8)

    assert len(reporter.warnings) == 1
    code, _, _, action = reporter.warnings[0]
    assert code == "language-audit-conflict"
    assert "--language ja" in action


def test_agreement_warns_about_nothing(reporter) -> None:
    lang_audit.report(_votes([("ja", "ja")] * 8), groups=60, sampled=8)

    assert reporter.warnings == []


# --- wired into the redecoder -------------------------------------------------


class FakeReader:
    def __init__(self, audio_path) -> None:
        pass

    def read(self, start, end):
        return np.zeros(int(max(0.0, end - start) * 16000), dtype=np.float32)


class FakeReferee:
    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.clip_counts = []

    def transcribe_batch(self, clips, **kwargs):
        self.clip_counts.append(len(clips))
        return [self.replies[i % len(self.replies)] for i in range(len(clips))]


def _group(start, end):
    return [{"start": start, "end": end}]


def _observe_run(redecoder, count, language, *, span=10.0):
    """Drive `maybe_redecode` over `count` groups that all agree — the blind case."""

    history: list[str] = []
    for index in range(count):
        before = list(history)
        history.append(language)
        redecoder.maybe_redecode(
            align_fn=lambda *a, **k: ([], []),
            model=None,
            group=_group(index * 30.0, index * 30.0 + span),
            segments=[{"start": index * 30.0, "end": index * 30.0 + span}],
            audio=None,
            sr=16000,
            gap_sec=0.5,
            auto_language_history=history,
            history_before=before,
            recent_language=(language if before else None),
            tail_real_limit_sec=1.0,
        )
    return history


def test_the_redecoder_observes_groups_it_never_triggers_on(monkeypatch, reporter):
    monkeypatch.setattr(lang_redecode.qwen_referee, "_SpanReader", FakeReader)
    referee = FakeReferee([("ใใใซใกใฏ", "Japanese")])
    redecoder = lang_redecode.LangRedecoder(referee, "audio.wav")

    _observe_run(redecoder, 30, "en")

    # The point of the whole feature: a perfectly consistent run.
    assert redecoder.stats()["triggers"] == 0
    audit = redecoder.run_audit()
    assert referee.clip_counts == [lang_audit.MAX_CLIPS]
    assert audit["suspect"] is True
    assert (audit["whisper_majority"], audit["referee_majority"]) == ("en", "ja")
    assert redecoder.stats()["audit"] == audit
    assert reporter.warnings[0][0] == "language-audit-conflict"


def test_the_audit_is_quiet_when_the_referee_agrees(monkeypatch, reporter):
    monkeypatch.setattr(lang_redecode.qwen_referee, "_SpanReader", FakeReader)
    referee = FakeReferee([("hello there", "English")])
    redecoder = lang_redecode.LangRedecoder(referee, "audio.wav")

    _observe_run(redecoder, 30, "en")
    audit = redecoder.run_audit()

    assert audit["suspect"] is False
    assert audit["agreement"] == 1.0
    assert reporter.warnings == []


def test_a_run_too_short_to_conclude_buys_no_inference(monkeypatch, reporter):
    """Fewer observations than the answer floor: pay for nothing."""

    monkeypatch.setattr(lang_redecode.qwen_referee, "_SpanReader", FakeReader)
    referee = FakeReferee([("ใใใซใกใฏ", "Japanese")])
    redecoder = lang_redecode.LangRedecoder(referee, "audio.wav")

    _observe_run(redecoder, lang_audit.MIN_ANSWERED - 1, "en")
    audit = redecoder.run_audit()

    assert referee.clip_counts == []
    assert audit["suspect"] is False
    assert reporter.warnings == []


def test_groups_that_cast_no_vote_are_not_observed(monkeypatch):
    """No detection means nothing to hold the referee's answer against."""

    monkeypatch.setattr(lang_redecode.qwen_referee, "_SpanReader", FakeReader)
    redecoder = lang_redecode.LangRedecoder(FakeReferee([("x", "English")]), "a.wav")
    history = ["en"]

    redecoder.maybe_redecode(
        align_fn=lambda *a, **k: ([], []),
        model=None,
        group=_group(0.0, 10.0),
        segments=[],
        audio=None,
        sr=16000,
        gap_sec=0.5,
        auto_language_history=history,
        history_before=list(history),  # unchanged: the group cast no vote
        recent_language="en",
        tail_real_limit_sec=1.0,
    )

    assert redecoder.run_audit()["groups"] == 0


def test_clips_are_capped_in_length(monkeypatch):
    """A 10-minute interval must not become a 10-minute referee clip."""

    monkeypatch.setattr(lang_redecode.qwen_referee, "_SpanReader", FakeReader)
    redecoder = lang_redecode.LangRedecoder(FakeReferee([("x", "English")]), "a.wav")

    _observe_run(redecoder, 10, "en", span=600.0)

    spans = [o.end - o.start for o in redecoder._observations]
    assert spans and max(spans) == lang_audit.MAX_CLIP_SEC


def test_a_sub_second_interval_is_not_evidence(monkeypatch):
    monkeypatch.setattr(lang_redecode.qwen_referee, "_SpanReader", FakeReader)
    redecoder = lang_redecode.LangRedecoder(FakeReferee([("x", "English")]), "a.wav")

    _observe_run(redecoder, 10, "en", span=0.3)

    assert redecoder._observations == []
