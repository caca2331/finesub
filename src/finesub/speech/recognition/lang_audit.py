"""An external anchor for the run's language decision.

`lang_redecode` fires when a group's detected language contradicts the rolling
majority of the recent groups. That predicate is *relative*: it compares the
quantity against itself, so it can only ever see a **local** deviation. When
the error is systematic the majority is wrong too, the contradiction never
materialises, and the detector stays silent through exactly the failure that
costs the most -- a file decoded in the wrong language from end to end.

That is measured, not hypothetical: on a synthetic ja+en mix **89.7% of the
Japanese was labelled `en` and the detector fired zero times**
(`docs/bench-baselines.md` 15.3).

> A detector defined as "disagrees with the majority" is blind to
> "the majority is wrong".

The missing piece is an anchor *outside* the quantity being audited. This
module supplies one: a deterministic, evenly spaced sample of the run's own
audio, re-read by the Qwen referee -- a second model that never sees Whisper's
decision -- and compared with what Whisper decided for that same audio.

**This reports; it does not act.** Acting would
mean forcing a language over a whole run, and the cost of getting *that* wrong
is total (an English file decoded as Japanese scores 0.000 against truth,
`bench-baselines.md` 15.5). Authorising it needs a false-positive rate measured
on real code-switching material, which we do not have -- synthetic splices
prove a defect exists, never how often it occurs (15.6). Until then, making the
silent failure loud is the whole job. See `docs/plans/crispasr-followups.md`.

The verdict is deliberately **threshold-free**: it fires when two independent
models disagree about the single most robust quantity in the run -- which
language most of the sampled audio is in. There is no tuned number to
recalibrate, which is the point; every number below is a *cost cap* or a
"do we have a sample at all" floor, and each says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: How many clips the audit is allowed to buy. A **cost cap**, so that the
#: audit's price is a constant rather than a fraction of the file: a four-hour
#: recording is audited for the same money as a four-minute one.
MAX_CLIPS = 8

#: A clip shorter than this cannot carry a language.
MIN_CLIP_SEC = 1.0

#: ...and a longer one costs a lot more without answering better. **Clip length
#: is the cost driver here, and it was measured the hard way**: at 20s this
#: audit took 52.3s for eight clips, two orders of magnitude off the "0.61s per
#: suspect" recorded for `--qwen-verify`, because that figure was for the short
#: segment clips that check probes. A sweep (`tools/bench/probe_referee_cost`)
#: put a 2s clip near 1s and a 20s clip at 6-9s.
#:
#: The choice of 6 is not itself a measurement -- every clip length in the
#: sweep, down to 2s, identified the language correctly, so the evidence bounds
#: this from below rather than pinpointing it. 6s is a deliberate margin over
#: the 2s that worked, because a 2s clip can land entirely on a filler.
MAX_CLIP_SEC = 6.0

#: The audit reads the referee's *language* field, not its transcript, so it
#: stops generation early. ⚠ Measured, this is **not** the lever it looks like:
#: dropping 256 to 48 on 20s clips moved the audit from 52.3s to 54.9s, i.e.
#: nothing outside the noise. It is kept because it is free and it bounds the
#: worst case, not because it bought the speedup. It cannot go to zero either:
#: the language field is only trusted when the model transcribed something.
AUDIT_NEW_TOKENS = 48

#: The smallest sample that can carry a majority at all rather than a tie or a
#: single voice. Below it the audit says what it saw and draws no conclusion.
#:
#: This is the only floor. An earlier draft also required a minimum number of
#: *groups*, which turned out to be both redundant and wrong: a 163s file with
#: continuous speech is five groups, and the audit skipped it entirely -- while
#: five clips is a perfectly serviceable sample. What matters is how many
#: answers came back, not how the decoder happened to carve the timeline.
#:
#: Note the referee abstains on any language outside its own map
#: (`lang_redecode`'s `_QWEN_LANG_CODES`), so "answered" can be far below
#: "sampled" on legitimate material -- that is an abstention, not a
#: disagreement, and it lands here rather than in the conflict branch.
MIN_ANSWERED = 3


@dataclass(frozen=True)
class Observation:
    """One group's language vote, plus the span to re-read if it is sampled."""

    start: float
    end: float
    language: str


@dataclass(frozen=True)
class Verdict:
    """What the two models each concluded, and whether they conflict."""

    suspect: bool
    reason: str
    groups: int
    sampled: int
    answered: int
    agreed_sec: float
    voted_sec: float
    whisper_majority: Optional[str]
    referee_majority: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "suspect": bool(self.suspect),
            "reason": self.reason,
            "groups": int(self.groups),
            "sampled": int(self.sampled),
            "answered": int(self.answered),
            "agreed_sec": round(float(self.agreed_sec), 3),
            "voted_sec": round(float(self.voted_sec), 3),
            "agreement": (
                round(self.agreed_sec / self.voted_sec, 3)
                if self.voted_sec > 0.0
                else None
            ),
            "whisper_majority": self.whisper_majority,
            "referee_majority": self.referee_majority,
        }


def resolve_mode(lang_redecode: object, language: object) -> Optional[str]:
    """What `--lang-redecode` buys for this run; ``None`` = build no referee.

    A separate function because this is the exact shape that goes wrong
    silently: a check that is wired, tested, and never actually reached. The
    four cells, and why:

    ``auto`` + auto language  -> ``redecode``        today's default, unchanged
    ``on``   + auto language  -> ``redecode+audit``  adds the run-level anchor
    ``on``   + ``--language`` -> ``audit-only``      no votes to contradict, so
                                                     the trigger is inert -- but
                                                     the *user* can still have
                                                     forced the wrong language
    ``auto`` + ``--language`` -> ``None``            nothing to do, buy nothing
    """

    flag = str(lang_redecode or "").strip().lower()
    if flag == "off":
        return None
    forced = bool(str(language or "").strip())
    if flag == "on":
        return "audit-only" if forced else "redecode+audit"
    return None if forced else "redecode"


def pick(count: int, *, max_clips: int = MAX_CLIPS) -> List[int]:
    """Evenly spaced sample indices over ``count`` observations.

    Deterministic on purpose. A reservoir sample would be the textbook answer
    for a stream of unknown length, but randomised sampling here would make the
    audit's verdict differ between two runs of the same file -- and this project
    has already been bitten once by exactly that (a term list read out of a
    `set` moved a measured hit rate by 8 percentage points). Spans are cheap to
    remember, so the whole list is kept and thinned at the end instead.

    Stratified midpoints rather than endpoints: the first and last group of a
    recording are the least representative parts of it.
    """

    if count <= 0 or max_clips <= 0:
        return []
    if count <= max_clips:
        return list(range(count))
    return [min(count - 1, int((i + 0.5) * count / max_clips)) for i in range(max_clips)]


def inspect(
    votes: Sequence[Tuple[float, str, Optional[str]]],
    *,
    groups: int,
    sampled: int,
    min_answered: int = MIN_ANSWERED,
) -> Verdict:
    """Judge ``(duration, whisper language, referee language)`` triples.

    Duration-weighted throughout: a 30-second clip is more evidence about what
    language the run is in than a 2-second one, and the referee's own vote
    share (`lang_redecode.VOTE_SHARE_MIN`) is weighted the same way.
    """

    answered = [
        (float(duration), str(whisper), str(referee))
        for duration, whisper, referee in votes
        if referee and duration > 0.0
    ]
    voted_sec = sum(duration for duration, _, _ in answered)
    agreed_sec = sum(
        duration for duration, whisper, referee in answered if whisper == referee
    )

    def _majority(index: int) -> Optional[str]:
        totals: Dict[str, float] = {}
        for vote in answered:
            totals[vote[index]] = totals.get(vote[index], 0.0) + vote[0]
        if not totals:
            return None
        best = max(totals.values())
        # Ties resolve to the alphabetically first code so the verdict is a
        # function of the votes and nothing else.
        return sorted(code for code, total in totals.items() if total == best)[0]

    whisper_majority = _majority(1)
    referee_majority = _majority(2)

    base = dict(
        groups=groups,
        sampled=sampled,
        answered=len(answered),
        agreed_sec=agreed_sec,
        voted_sec=voted_sec,
        whisper_majority=whisper_majority,
        referee_majority=referee_majority,
    )

    if len(answered) < max(1, int(min_answered)):
        return Verdict(
            False,
            f"referee answered {len(answered)}/{sampled} sampled clips",
            **base,
        )
    if whisper_majority is None or referee_majority is None:
        return Verdict(False, "no majority on one side", **base)
    if whisper_majority == referee_majority:
        return Verdict(False, "", **base)

    share = agreed_sec / voted_sec if voted_sec > 0.0 else 0.0
    return Verdict(
        True,
        f"whisper decoded this run as {whisper_majority}, the referee hears "
        f"mostly {referee_majority} "
        f"({len(answered)} clips, {voted_sec:.1f}s, they agree on "
        f"{share * 100:.0f}% of it)",
        **base,
    )


def report(
    votes: Sequence[Tuple[float, str, Optional[str]]],
    *,
    groups: int,
    sampled: int,
) -> Verdict:
    """`inspect`, and a warning when the two models conflict."""

    verdict = inspect(votes, groups=groups, sampled=sampled)
    if not verdict.suspect:
        return verdict

    from ...reporting import current_reporter

    current_reporter().warning(
        "language-audit-conflict",
        verdict.reason,
        impact=(
            "the whole run may be transcribed in the wrong language; the "
            "inline redecode check cannot see this, because it only compares "
            "each group against the same majority that is wrong"
        ),
        action=(
            f"re-run with --language {verdict.referee_majority} if the "
            "recording is in one language; on a genuinely bilingual recording "
            "expect this warning and ignore it -- the two models are then just "
            "picking different halves (measured: a 50/50 ja+en splice fires "
            "this, bench-baselines.md 15.7)"
        ),
    )
    return verdict
