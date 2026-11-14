"""The contract between the alignment loop and an ASR backend, declared once.

`transcribe.py` drives decoding through a small option surface
(``model.transcribe_wt(audio, **options)`` and the batch decoder). Every option
name that may cross that seam is declared in `DECODE_OPTIONS` and nowhere else:
the loop builds its keyword arguments only through `decode_options`, and a
backend rejects any name outside the set it accepts (`reject_unknown`).

Why both halves. The failure this exists to stop is silent: a backend that
reads the options it knows with ``options.get`` and ignores the rest keeps
running when the loop grows a new option, and quietly decodes differently on
that backend -- with nothing turning red, because the backends other than
CTranslate2 run on hardware no CI has. A declared set is what a test can check.

Every option is one of two kinds:

- **required** -- every backend implements it;
- **capability-flagged** -- a backend may lack it, says so by leaving the
  capability out of its ``capabilities``, and the loop then takes the declared
  ``degrade`` path instead of sending the option.

There is no "declared but refused" kind: an option the loop sends on every
call and a backend refuses would fail that backend on every call.

Adding a backend: give the class ``capabilities`` and ``accepted_decode_options``
-- the latter **written out**, never computed from `DECODE_OPTIONS`. Deriving it
is what the check is meant to catch: a backend that derives its own answer
agrees with the contract by construction, so a newly declared option would be
claimed by every backend and implemented by none, silently. `accepted_for` is
for the *checker*, which compares the two. Register it in `BACKENDS`, and
``test_asr_backend_contract`` holds it to the set. The public description of
this contract, including what a contributed backend must bring, is
``docs/asr-backends.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional


#: The Whisper model a run decodes with unless told otherwise. Declared here,
#: torch-free, so `run_pipeline`'s signature can carry it without importing
#: `transcribe`.
DEFAULT_MODEL = "large-v3-turbo"


@dataclass(frozen=True)
class DecodeOption:
    name: str
    #: None for a required option; otherwise the capability a backend declares.
    capability: Optional[str] = None
    #: What the loop does instead when the backend lacks the capability.
    degrade: str = ""


BEAM = "beam"
#: Not an option: whether the backend provides a batched first-decode driver
#: (`DecodePrefetch`, used only when `--asr-decode-batch` > 1). Without it the
#: loop decodes window by window whatever the batch setting says.
BATCH_DECODE = "batch_decode"

DECODE_OPTIONS: tuple[DecodeOption, ...] = (
    DecodeOption("language"),
    DecodeOption("temperature"),
    DecodeOption(
        "beam_size",
        capability=BEAM,
        degrade="the low-coverage rescue skips its beam rung and goes straight to the split rung",
    ),
    DecodeOption(
        "best_of",
        capability=BEAM,
        degrade="sent only together with beam_size",
    ),
    DecodeOption("detect_disfluencies"),
    DecodeOption("collect_refine_signals"),
    DecodeOption("collect_attention_signals"),
    DecodeOption("force_teacher_force"),
)

OPTION_NAMES = frozenset(option.name for option in DECODE_OPTIONS)
CAPABILITIES = frozenset(
    option.capability for option in DECODE_OPTIONS if option.capability is not None
) | {BATCH_DECODE}

#: Every backend the pipeline can build, as ``module:attribute``. Imported by
#: the contract test only, so a backend's heavy dependencies stay lazy.
BACKENDS: dict[str, str] = {
    "fw-refine": "finesub.speech.recognition.fw_refine_backend:RefinedWhisperModel",
}


def accepted_for(capabilities: frozenset[str]) -> frozenset[str]:
    """The option names a backend with these capabilities must accept -- exactly."""

    unknown = set(capabilities) - CAPABILITIES
    if unknown:
        raise ValueError(f"unknown ASR backend capabilities: {sorted(unknown)}")
    return frozenset(
        option.name
        for option in DECODE_OPTIONS
        if option.capability is None or option.capability in capabilities
    )


def contract_violations(backend: Any) -> list[str]:
    """What a backend class gets wrong about the contract; empty when it holds.

    A backend must declare ``capabilities`` (known names only) and accept
    exactly `accepted_for` those capabilities: a missing option is one the loop
    would send and the backend would refuse, an extra one is an option the
    backend claims without a declared capability behind it.

    ⚠ This only means anything while backends **write their set out**. One that
    computes it from `DECODE_OPTIONS` passes whatever the contract says, which
    is how a new required option would arrive claimed by everyone and handled
    by no one -- the silent divergence the whole module exists to stop.
    """

    problems: list[str] = []
    capabilities = getattr(backend, "capabilities", None)
    accepted = getattr(backend, "accepted_decode_options", None)
    if capabilities is None:
        return ["declares no capabilities"]
    unknown = set(capabilities) - CAPABILITIES
    if unknown:
        return [f"declares unknown capabilities {sorted(unknown)}"]
    if accepted is None:
        return ["declares no accepted_decode_options"]
    expected = accepted_for(frozenset(capabilities))
    missing = expected - set(accepted)
    extra = set(accepted) - expected
    if missing:
        problems.append(f"does not accept {sorted(missing)}")
    if extra:
        problems.append(f"accepts {sorted(extra)} without the capability for them")
    return problems


def decode_options(**values: Any) -> dict[str, Any]:
    """Build a decode-options mapping, refusing any name not declared above."""

    unknown = set(values) - OPTION_NAMES
    if unknown:
        raise TypeError(
            f"undeclared ASR decode option(s) {sorted(unknown)}; declare them in "
            f"finesub.speech.recognition.asr_backend.DECODE_OPTIONS"
        )
    return dict(values)


def reject_unknown(options: Mapping[str, Any], accepted: frozenset[str], *, backend: str) -> None:
    """A backend's guard: fail on any option it does not implement."""

    unknown = set(options) - accepted
    if unknown:
        raise TypeError(
            f"ASR backend {backend!r} does not implement decode option(s) "
            f"{sorted(unknown)}"
        )


def supports(model: Any, capability: str) -> bool:
    """Whether a model declares a capability.

    A model without a ``capabilities`` attribute is not a registered backend
    (test doubles are the only such objects) and is treated as having every
    capability, which is what the loop assumed before capabilities existed.
    The contract test is what stops a registered backend from relying on that.
    """

    if capability not in CAPABILITIES:
        raise ValueError(f"unknown ASR backend capability: {capability!r}")
    declared = getattr(model, "capabilities", None)
    if declared is None:
        return True
    return capability in declared
