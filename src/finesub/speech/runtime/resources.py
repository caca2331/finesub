"""Resource tiers for the production ASR pipeline.

A tier answers **"what class of GPU is this, and therefore how much work may
run on it at once"** -- it is a statement about the machine, not a cap the
pipeline promises to stay under. That distinction used to be invisible: the
option was called a *budget* and its values were the numbers 4/8/12/16, so "8"
read equally well as "my card has 8GB" and as "only use 8GB". The two readings
pick opposite tiers, and both consumers of a tier break under the wrong one --
the separator scales its instance count by it, and the language referee asks
whether there is room beside the resident Whisper pool. Picking too small
silently drops the referee to CPU; too large opens separator instances the card
cannot hold. Hence names rather than numbers, and `auto` as the default: the
tiers now say what they are, and the usual answer is "ask the driver" anyway.

A tier carries exactly one VRAM figure, `usable_gib`: what has to be **free**,
not what the card holds. The hints quote it, the violation check compares
against it, and the language referee computes its headroom from it. Quoting the
card's own size instead is how "my card is 8GB so it fits" turns into an OOM --
the desktop and the driver were already holding some of it.

The card's nominal size appears in exactly one place, and only for a moment:
`tier_for_vram` subtracts `reserve_for_capacity` from what the driver reports to
estimate how much is actually free, then matches that. Nothing downstream ever
sees it.

There is deliberately **no soft cap**. CT2 answers a CUDA OOM with a
process-level abort, so a limit the pipeline pretended to enforce would be a
promise it cannot keep (`ASR_DECODE_BATCH_BY_TIER` records the same reasoning
one level down). `usable_gib` is a *requirement* and a violation threshold --
never a quota anything is held under.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from ...reporting import current_reporter

BYTES_PER_GIB = 1024**3

#: The detection reserve, as `capacity / GPU_RESERVE_DIVISOR + GPU_RESERVE_FLOOR_GB`.
#:
#: It scales because what is already on a card scales with it: a bigger card
#: usually means a bigger desktop, more displays, and a driver willing to cache
#: more. A flat figure is either too generous on a 4GB card or far too mean on a
#: 24GB one.
#:
#: The two constants are not fitted to anything -- they are chosen so the curve
#: passes exactly through the tier table: 4GB -> 1.0, 8GB -> 1.5, 12GB -> 2.0,
#: which is precisely `capacity - usable_gib` for `entry` / `standard` / `high`.
#: That is the point of writing it as a rule rather than a fourth column: the
#: same three numbers, but now they extend to cards the table never named.
#:
#: Still a *detection margin*, not an accounting of real residency. Nothing can
#: know that from inside the process, and `reserve_for_capacity` never pretends
#: to -- see `docs/gpu-profiles.md` for what was measured against it.
GPU_RESERVE_DIVISOR = 8.0
GPU_RESERVE_FLOOR_GB = 0.5

#: The largest tier `auto` will pick on its own, and the reported capacity that
#: lifts that ceiling.
#:
#: Having room is not a reason to use it. `high` runs three separator workers,
#: and the worker sweep (`docs/separator-optimization.md` E7) puts the
#: throughput peak at *two* -- three is measurably slower. So a 16GB card
#: getting `high` from `auto` would be handed a slower run for having a bigger
#: card, which is the opposite of what a tier is for.
#:
#: `high` stays selectable by hand, and `auto` reaches for it only on a card
#: large enough (>=24GB) that the question is about headroom rather than speed.
#:
#: ⚠ **The 24GB threshold is an owner decision, not a measurement.** Raised in
#: review 2026-09-01: more VRAM proves three workers *fit*, never that they are
#: faster, and E7 measured them slower on the one card we have. The owner chose
#: it anyway, knowing that. Nobody has run the sweep on a 24GB card -- if that
#: ever happens and three workers are still slower there, the honest move is to
#: drop this lift and leave `high` hand-only.
#:
#: The ceiling is `standard_large_vram` rather than `standard` (2026-09-02):
#: capping at `standard` meant a 12-16GB card was denied the *VRAM budget* it
#: really has purely to avoid the third separator worker, and the referee paid
#: for that by never compiling beside a large model. The two are now separate
#: tiers, so the ceiling can give the headroom without the slower worker.
AUTO_TIER_CEILING = "standard_large_vram"
AUTO_CEILING_LIFT_CAPACITY_GB = 24

#: System RAM the pipeline is checked against, in GiB. One figure for every
#: tier -- but **not** because RAM is tier-independent. It is not: separator
#: workers each hold their audio block in host memory, and the 2026-09-01
#: measurement on this machine reads 2.43 / 4.00 / 4.81 GiB across the three
#: tiers (`docs/gpu-profiles.md`). The flat 8 stands because it is a violation
#: *warning* and nothing reserves against it, and because at `high` -- the
#: worst tier -- the peak is still only 60% of it. This is the number to
#: revisit first if the budgets are ever tightened: RAM is far closer to its
#: limit than VRAM is to any of the tier limits. See `resource_limit_violations`.
RAM_BUDGET_GB = 8

#: Ask the driver instead of naming a tier. The default everywhere.
AUTO_GPU_TIER = "auto"

#: The smallest GPU tier -- where a card lands when it exists but this build
#: cannot size it. **No longer where "no GPU" lands**: that is `CPU_TIER`
#: (2026-09-02). `entry` used to do both jobs, which is how a CPU-only run
#: ended up with artifacts claiming a 3 GiB VRAM budget.
DEFAULT_GPU_TIER = "entry"


@dataclass(frozen=True)
class TierSpec:
    """One tier: who it is for, and what the machine needs to run it."""

    name: str
    #: VRAM that has to be **free** for this tier, in GiB -- not what the card
    #: holds. The hints quote it, the violation check compares against it, and
    #: the language referee computes its headroom from it. The only VRAM number
    #: a tier has.
    usable_gib: float
    vocal_separator_instances: int
    #: One line for `--help`. It has to answer "which one am I" without the
    #: reader opening a doc, which is why the requirement is in the text.
    #: Localized copy would belong to a GUI front end's own translations; this
    #: is the English CLI face.
    summary: str
    #: Whether this tier may put work on the GPU at all. **A policy, not a
    #: capability**: `False` means "do not ask", which is a different statement
    #: from "asked and the answer was no". Every capability question stays
    #: per-backend and per-stage (`device.cuda_usable`, and the CT2 probe when
    #: it lands) -- this only decides whether that question gets asked.
    #: `usable_gib` is meaningless when it is `False`, and nothing reads it:
    #: the VRAM check returns early and the referee never places beside a pool
    #: that is not on the card.
    gpu: bool = True


#: Ordered smallest to largest. The instance count is the only thing a tier
#: actually changes (ASR is always one worker, whatever the tier).
#:
#: **There is no 4-separator tier.** The worker sweep on this hardware
#: (`docs/separator-optimization.md` E7) puts the throughput peak at *two*
#: workers -- 3 and 4 are slower, not merely less good -- and 4 workers are
#: reachable only on audio past 15 minutes, where the loss is largest. A tier
#: that costs VRAM to run slower is not a tier. The 3-worker rung stays because
#: the measurement is one card's and the worker count is bound to block
#: planning, so narrowing further would move artifact boundaries for a gain
#: nobody has measured off this machine.
#: The tier that means "this machine is not using the GPU". Split out of
#: `entry` on 2026-09-02: before that, a machine with no CUDA device was handed
#: `entry` and its artifacts claimed a 3 GiB VRAM budget it never had, while
#: the VRAM warning needed a hardcoded "unless there is no CUDA" branch.
CPU_TIER = "cpu"

GPU_TIERS: tuple[TierSpec, ...] = (
    TierSpec(
        CPU_TIER,
        0.0,
        1,
        "CPU only: no GPU work at all. Chosen automatically when the machine "
        "has no CUDA device, and selectable by hand to leave the card to "
        "something else",
        gpu=False,
    ),
    TierSpec(
        "entry",
        3.0,
        1,
        "baseline GPU acceleration: needs 3GB free VRAM + 8GB RAM, 1 separator. "
        "A machine whose GPU cannot be used at all lands here too and runs on CPU",
    ),
    TierSpec("standard", 6.5, 2, "mainstream card, needs 6.5GB free VRAM + 8GB RAM, 2 separators"),
    # Same separator count as `standard`, `high`'s VRAM figure. It exists
    # because those two numbers answer different questions and the old table
    # conflated them: the separator count is a *throughput* choice (two workers
    # is the measured peak, E7), while the VRAM figure is a *budget* the
    # referee spends -- and the referee's compiled decode step needs 3.5 GiB
    # free BESIDE the resident Whisper pool (`COMPILE_MIN_VRAM_GIB`). On
    # `standard` a large-v3-class model leaves 6.5 - 3.82 = 2.7, so the referee
    # co-resides but can only run eager; here it leaves 6.2 and compiles, and
    # still compiles at `--asr-decode-batch 8` (10 - 5.85 = 4.2).
    #
    # So this is the tier a 12-16GB card should actually get: the headroom of
    # `high` without `high`'s slower third separator worker.
    TierSpec(
        "standard_large_vram",
        10.0,
        2,
        "mainstream card with headroom, needs 10GB free VRAM + 8GB RAM, "
        "2 separators (same as standard) -- the spare VRAM goes to the "
        "second-model referee, which can then take its compiled path",
    ),
    TierSpec("high", 10.0, 3, "high-end card, needs 10GB free VRAM + 8GB RAM, 3 separators"),
)

#: ASR decode batch per tier. **Derived statically, never probed**: CT2 answers
#: a CUDA OOM with a process-level abort, so an adaptive "try a bigger batch and
#: back off" would take the run down rather than degrade
#: (`docs/bench-baselines.md` 10.4).
#:
#: Every entry is 1 today, i.e. **the feature is off at every tier**. The knob,
#: the resolver and the provenance record exist so that turning it on is one
#: table edit plus the assembly step -- see `resolve_asr_decode_batch` for what
#: is deliberately still missing.
# 1 everywhere: the assembly exists (`transcribe.DecodePrefetch`) and was swept
# on 12 products at B=4/8/16 (docs/bench-baselines.md 二十二) -- text and
# timing pass the pre-registered gate at 4 and 8, but the end-to-end speedup is
# 1.06x against the 1.15x floor (single-window groups only; multi-window groups
# cascade), so the default stays off and the knob is an explicit opt-in.
#
# The table stays 1 even though the same sweep on `large-v3` clears the floor
# by a wide margin (1.40x at B=4, 1.57x at B=8; 二十三). A tier entry is
# model-independent, and it is the *default* model that decides what the
# default costs: batching pays on large-v3 because its per-window decode is
# ~3.9x dearer, so the batchable share of alignment time is much larger. That
# is a reason to reach for `--asr-decode-batch 8` when running a big model, not
# a reason to turn it on for everyone.
ASR_DECODE_BATCH_BY_TIER = {spec.name: 1 for spec in GPU_TIERS}


@dataclass(frozen=True)
class ResourceProfile:
    gpu_tier: str
    #: Free VRAM the tier requires, carried so the referee and the limit check
    #: do not have to look the tier up again.
    usable_gib: float
    vocal_separator_instances: int
    vocal_separation_batch_size: int
    ram_budget_gb: int = RAM_BUDGET_GB
    #: See `TierSpec.gpu`. Carried so a stage can ask the profile rather than
    #: re-deriving the policy from the tier name.
    gpu: bool = True

    @property
    def asr_decode_batch(self) -> int:
        return ASR_DECODE_BATCH_BY_TIER.get(self.gpu_tier, 1)

    @property
    def usable_gpu_gb(self) -> float:
        return float(self.usable_gib)

    @property
    def gpu_limit_bytes(self) -> int:
        return int(self.usable_gpu_gb * BYTES_PER_GIB)

    @property
    def ram_limit_bytes(self) -> int:
        return int(float(self.ram_budget_gb) * BYTES_PER_GIB)


RESOURCE_PROFILES = {
    spec.name: ResourceProfile(
        gpu_tier=spec.name,
        usable_gib=spec.usable_gib,
        vocal_separator_instances=spec.vocal_separator_instances,
        vocal_separation_batch_size=1,
        gpu=spec.gpu,
    )
    for spec in GPU_TIERS
}


def gpu_tier_names() -> tuple[str, ...]:
    """Every tier, smallest first -- `cpu` included."""

    return tuple(spec.name for spec in GPU_TIERS)


def gpu_backed_tiers() -> tuple[TierSpec, ...]:
    """Only the tiers that actually use the GPU.

    `auto` matches a card against these: `cpu` is not a rung on the same ladder
    (its `usable_gib` is 0, so it would swallow every card that fits nothing
    else, when the documented answer there is the smallest GPU tier).
    """

    return tuple(spec for spec in GPU_TIERS if spec.gpu)


def gpu_tier_cli_choices() -> tuple[str, ...]:
    """What every front end offers: `auto`, then the tiers."""

    return (AUTO_GPU_TIER, *gpu_tier_names())


def gpu_tier_help() -> str:
    """One `--help` string that also carries each tier's requirement."""

    rows = "; ".join(f"{spec.name}={spec.summary}" for spec in GPU_TIERS)
    return (
        "GPU tier -- how much parallel work this machine is asked to do. It "
        "scales separator concurrency, and each GPU tier lists the FREE VRAM "
        "that needs (not the card's own size: the desktop and the driver hold "
        f"some); `{CPU_TIER}` has no VRAM requirement because it uses none. "
        f"Default: {AUTO_GPU_TIER}, ask the driver. {rows}"
    )


def reserve_for_capacity(capacity_gib: float) -> float:
    """How much of a card of this size `auto` assumes is already taken."""

    return float(capacity_gib) / GPU_RESERVE_DIVISOR + GPU_RESERVE_FLOOR_GB


def tier_for_vram(total_gib: float) -> str:
    """The largest tier whose free-VRAM requirement this card can plausibly meet.

    Two steps, and both exist because of a way `auto` could otherwise be quietly
    wrong.

    **Round to the nominal size first.** The driver never reports the number on
    the box: a 16GB RTX 5070 Ti answers 15.92 GiB, and a 12GB card answers just
    under 12. Carrying that shortfall into the comparison drops cards a rung --
    silently running a separator instance short, which is exactly the
    misconfiguration `auto` exists to remove. The gap is well under half a GiB
    on every card measured, so rounding to the nearest whole GiB recovers the
    nominal size without ever inventing one.

    **Then subtract the reserve**, because a tier asks for free VRAM and a card
    is never entirely free. Without it a 10GB card would match `high` -- three
    separator instances against 10GiB of headroom that was never there. The
    reserve scales with the card (`reserve_for_capacity`), because so does what
    is already on it.

    **Then stop at `AUTO_TIER_CEILING`**, unless the card is big enough to lift
    it. Fitting is not a reason to choose: `high`'s third separator worker is
    measurably slower than `standard`'s second, so promoting a 16GB card would
    hand it a slower run for owning a bigger card.

    Below the smallest tier there is nothing to fall back to (4GB is already the
    documented minimum), so a smaller card gets `entry` and finds out from the
    OOM, exactly as it did when the tier was picked by hand. Above the largest
    there is nothing either: a 32GB card gets `high`, deliberately -- see
    `GPU_TIERS` on why no wider tier exists.
    """

    nominal = math.floor(float(total_gib) + 0.5)
    free = nominal - reserve_for_capacity(nominal)
    # `<=` and not `<`: the curve is built to land exactly on the tier figures
    # at 4/8/12GB, so a strict comparison would drop every nominal card a rung.
    tiers = gpu_backed_tiers()
    fits = [index for index, spec in enumerate(tiers) if spec.usable_gib <= free]
    chosen = fits[-1] if fits else 0
    if nominal < AUTO_CEILING_LIFT_CAPACITY_GB:
        chosen = min(chosen, [spec.name for spec in tiers].index(AUTO_TIER_CEILING))
    return tiers[chosen].name


def check_tier_device_agreement(gpu_tier: str, device: str | None) -> None:
    """Refuse `--gpu-tier cpu --device cuda`, which asks for both at once.

    Silently honouring one of them is how the next silent misconfiguration
    happens: the tier is a policy ("do not use the GPU") and `--device cuda` is
    a request to use it, so there is no reading under which both hold. The
    other direction needs no check -- a GPU tier with `--device cpu` is the
    ordinary "leave the card alone this run", and every stage already reports
    the CPU it settled on.
    """

    if not gpu_tier or not device or gpu_tier == AUTO_GPU_TIER:
        return
    wants_cuda = str(device).strip().lower().startswith("cuda")
    profile = RESOURCE_PROFILES.get(gpu_tier)
    if wants_cuda and profile is not None and not profile.gpu:
        raise ValueError(
            f"--gpu-tier {gpu_tier} means this run does not use the GPU, but "
            f"--device {device} asks for it. Pick one: drop --device to let the "
            f"tier decide, or name a GPU tier "
            f"({'/'.join(spec.name for spec in gpu_backed_tiers())})."
        )


def detect_gpu_tier() -> str:
    """The tier this machine's card sits in, or the default if none answers.

    Deliberately silent about a missing GPU: `resolve_device` already warns once
    per stage about the CPU fallback, and a second warning from here would only
    say the same thing in different words.
    """

    from .device import cuda_device_present, total_vram_gib

    if not cuda_device_present():
        # No card at all -- a GPU tier would be a claim about hardware that is
        # not there. This is the policy branch, and it is the ONLY one.
        return CPU_TIER
    total = total_vram_gib()
    if total is None:
        # A card exists but this PyTorch build cannot size it (too old for its
        # kernels). Still a GPU tier on purpose: another backend may run on it,
        # and folding torch's verdict into the tier would decide that for every
        # stage at once. The smallest rung is the honest guess -- the torch
        # stages will degrade themselves per stage anyway.
        return DEFAULT_GPU_TIER
    return tier_for_vram(total)


def resolve_gpu_tier(gpu_tier: str | None = None) -> str:
    """Turn what a front end was given into a tier name.

    `None` and `"auto"` are the same request -- "ask the card" -- so an unset
    option and an explicit `auto` cannot drift apart.
    """

    text = str(gpu_tier).strip().lower() if gpu_tier is not None else ""
    if not text or text == AUTO_GPU_TIER:
        return detect_gpu_tier()
    if text not in RESOURCE_PROFILES:
        choices = ", ".join(gpu_tier_cli_choices())
        raise ValueError(f"Unsupported GPU tier: {gpu_tier}. Use one of: {choices}.")
    return text


def get_resource_profile(gpu_tier: str | None = None) -> ResourceProfile:
    return RESOURCE_PROFILES[resolve_gpu_tier(gpu_tier)]


def resolve_asr_decode_batch(
    explicit: int | str | None = None,
    *,
    gpu_tier: str | None = None,
) -> int:
    """How many windows the ASR decoder should decode per generate call.

    Resolution order is the house one: an explicit value beats the tier's
    static entry. ``"auto"`` and ``None`` both mean "ask the tier".

    A value above 1 makes `align_segments` prefetch the next windows in one
    batched decode (`transcribe.DecodePrefetch`, wavefront in order); the
    tier table is the default and is set from the measured break-even per
    tier (`docs/bench-baselines.md` 第十 for the gate, 二十二 for the numbers).
    The value is recorded in the artifact as provenance, never in the
    checkpoint key -- resuming with another batch size is legitimate.

    Returning 1 rather than raising on a bad value: a batch size is a
    performance knob, and taking a transcription down over one is the wrong
    trade.
    """

    if explicit is not None and str(explicit).strip().lower() not in ("", "auto"):
        try:
            value = int(explicit)
        except (TypeError, ValueError):
            return 1
        return max(1, value)
    return get_resource_profile(gpu_tier).asr_decode_batch


def _gib(value: int) -> str:
    """Bytes as GiB, for a line a person reads.

    The warning used to quote the raw counts -- `(3639984783 > 2791728742)` --
    which says "over budget" and nothing a reader can act on. The debug line
    two calls earlier was already formatted; only the warning was not.
    """

    return f"{float(value) / BYTES_PER_GIB:.2f} GiB"


def warn_if_vram_is_short(
    profile: ResourceProfile, *, stage: str, device: str | None = None
) -> None:
    """Say so when the card has less free VRAM than this tier asks for.

    **Warns; never downgrades and never refuses.** Two reasons, and the second
    is the one that decides it:

    * A tier picks the separator's worker count, which picks the block plan, so
      changing it mid-flight would change where the artifact boundaries fall --
      the same file would produce different output depending on what else the
      user happened to have open. `auto` reads *capacity* precisely so that its
      answer is stable; a live figure must not sneak back in as a decision.
    * Being short is not the same as failing. The measured `entry` peak is
      2.26GiB against a 3GiB requirement, so a card a little under still
      finishes plenty of clips. Refusing would take away runs that work.

    Applied identically however the tier was chosen. A 6GB card that was handed
    `standard` by name gets the same sentence in the same place as one that
    reached it through `auto` -- the user asked for a tier, and what they are
    owed is the fact, not a second-guess.

    Silent when the run is not going to the GPU at all -- either because no
    usable CUDA device answers (that machine is on the CPU path and
    `resolve_device` already said so) or because the user asked for `--device
    cpu` on a machine that *has* one. The second is the easy one to get wrong:
    the card is perfectly visible, so a check that only asks the driver would
    warn about VRAM for a run that will not touch it.
    """

    from .device import free_vram_gib

    if not profile.gpu:
        # The tier says this run does not use the GPU. Warning about free VRAM
        # would be answering a question nobody asked -- and this replaces the
        # hardcoded "unless there is no CUDA" branch that used to live here.
        return
    if device is not None and not str(device).strip().lower().startswith("cuda"):
        return

    free = free_vram_gib()
    if free is None or free >= profile.usable_gib:
        return
    # The smallest *GPU* tier: `--gpu-tier cpu` is not a smaller rung, it is a
    # different answer, and it is already offered below as `--device cpu`.
    smallest = gpu_backed_tiers()[0].name
    # On the smallest tier there is no smaller one to suggest, and suggesting
    # the tier the user is already on is the least useful thing to say in
    # exactly the case where they most need a way out.
    downgrade = "" if profile.gpu_tier == smallest else f"改用 --gpu-tier {smallest}，"
    current_reporter().warning(
        "gpu-vram-short",
        f"{stage}: the {profile.gpu_tier} tier expects "
        f"{profile.usable_gib:.2f} GiB of free VRAM, but only "
        f"{free:.2f} GiB is free right now.",
        impact="可能显存不足而中断；CT2 遇 CUDA OOM 是进程级中止，接不住",
        action=f"关掉占用显存的程序，{downgrade}实在不行用 --device cpu（慢但能跑完）",
    )


def resource_limit_violations(
    *,
    peak_gpu_bytes: Optional[int],
    peak_ram_bytes: Optional[int],
    profile: ResourceProfile,
) -> list[str]:
    violations: list[str] = []
    if peak_gpu_bytes is not None and peak_gpu_bytes > profile.gpu_limit_bytes:
        violations.append(
            f"peak_gmem exceeds the {profile.gpu_tier} tier limit "
            f"({_gib(peak_gpu_bytes)} > {_gib(profile.gpu_limit_bytes)})"
        )
    if peak_ram_bytes is not None and peak_ram_bytes > profile.ram_limit_bytes:
        violations.append(
            f"peak_mem exceeds the {profile.gpu_tier} tier limit "
            f"({_gib(peak_ram_bytes)} > {_gib(profile.ram_limit_bytes)})"
        )
    return violations
