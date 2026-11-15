"""Joint ASR backends: one model hands back the whole transcript with its times and speakers, in place of
separation, VAD, Whisper and acoustic diarization. In the cloud: MAI2, Gemini 3.5 Transcribe ("35t") and MOSS Pro;
on this machine's GPU: open MOSS-Transcribe-Diarize with run-time rescues (`moss_local`).

Each provider module turns the source audio into a `model.Transcript` -- the
provider's words (or, for the two MOSS, its segments) grouped into speaker turns in
the order the provider emitted them. The cloud ones cache every call's raw answer in the
task directory (`cache.CallCache`), so a rerun spends no quota; local MOSS caches nothing (its
`-aligned.json` is the result). What happens to the turns after that (line building, timing,
overlap rules) is not here.

What each provider does and why it is wired this way: `docs/joint-asr.md`.
"""

from __future__ import annotations

import threading

LOCAL = "local"
MAI2 = "mai2"
GEMINI_35T = "gemini-35t"
MOSS_PRO = "moss-pro"
MOSS_LOCAL = "moss-local"

BACKENDS = (LOCAL, MAI2, GEMINI_35T, MOSS_PRO, MOSS_LOCAL)
#: Backends whose transcript comes in segments (segment times, one speaker each, overlapping across speakers):
#: their units are timed by Silero (S5b) or the forced aligner (S6b), and each turn is split on its own.
SEGMENT_BACKENDS = (MOSS_PRO, MOSS_LOCAL)
#: Owner 2026-10-30: the short names follow the provider's current model, so a
#: newer model moves the alias instead of renaming the option. `moss` is MOSS Pro.
ALIASES = {"gemini": GEMINI_35T, "moss": MOSS_PRO}
#: One joint model computing on the GPU at a time: the forced aligner (`--asr-aligner`, cloud items on the batch's
#: network lane) and local MOSS (the GPU lane). ⚠ Take it before Silero's lock (`silero_exclusive`), never inside
#: it: local MOSS runs Silero while holding this one.
GPU_LOCK = threading.Lock()
CHOICES = (*BACKENDS, *ALIASES)


def resolve_backend(name: str) -> str:
    """The full backend name for an option value (aliases resolved); metadata records this."""

    value = name.strip().lower()
    value = ALIASES.get(value, value)
    if value not in BACKENDS:
        raise ValueError(f"unknown ASR backend {name!r}; choose one of {', '.join(CHOICES)}")
    return value
