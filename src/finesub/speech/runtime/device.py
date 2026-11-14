"""Whether this machine's GPU can run the stack, decided in one place.

Two different situations end in the same CPU fallback: there is no usable CUDA
at all (no driver, no card, ``CUDA_VISIBLE_DEVICES`` emptied), and there is a
card whose architecture the installed PyTorch ships no kernels for. The second
one is why this module exists -- torch reports ``is_available() == True`` for a
GTX 1060, and the mismatch only surfaces at the first real op as ``no kernel
image is available for execution on the device``, which names neither the card
nor the fix.

The check is deliberately derived from ``torch.cuda.get_arch_list()`` rather
than a hard-coded floor: the supported set is a property of the installed wheel,
so it moves on its own when the torch pin moves.

The ASR stage asks a second copy of that question against a different binary
(``ct2_cuda_unusable_reason``), because CTranslate2 ships its own CUDA build and
its own architecture list. That one *is* a hard-coded floor -- the library
exposes no arch list to derive it from -- and it is checked here rather than
left to the first encode for exactly the reason above: below the floor the
failure is a driver-level "no kernel image", which names neither the card nor
the fix.

**Which vendor** is answered here too, and nowhere else. PyTorch's ROCm build
reuses the ``cuda`` device name and the whole ``torch.cuda`` API for AMD cards,
so a stage that says ``"cuda"`` runs on either vendor without knowing which --
and must not find out by probing ``torch.cuda`` itself. What does differ is
answered below: the kernel check (AMD kernels do not run across ``gfx``
targets, so it is membership, not a floor), the runtime/architecture strings a
compile cache is keyed on, and which of several cards to use. NVIDIA answers
are unchanged by all of it; a source guard (`test_device_probes_live_in_one_module`)
keeps the probes in this module.

Intel is the third vendor, and the one that changes what a placement *means*.
PyTorch's Intel GPU build is a separate binary like the ROCm one -- it carries
no CUDA at all and, unlike ROCm, it does not reuse the ``cuda`` name either, so
the family has to be named explicitly (that is what `gpu_device` and the
``"xpu"`` device string are for). The bigger difference is one stage down:
**CTranslate2 ships no Intel GPU backend**, so the ASR stage stays on the CPU
even when everything else runs on the card. That is why the ASR device question
has its own Intel answer (`resolve_asr_device`) rather than inheriting torch's
verdict: on this hardware the two stages genuinely cannot share one.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Optional
import weakref

import torch

from ...doc_links import REPO_INSTALL_DOC
from ...reporting import current_reporter, reporter_delivers

# Spelled as product lines because that is what a user can actually check on
# their own machine. sm_75 is the oldest consumer architecture in the cu128
# wheel's kernel list; sm_70 covers the datacenter TITAN V / V100. Keep in sync
# with README.md's hardware table.
SUPPORTED_GPU_HINT = "GTX 1650/1660 or RTX 20/30/40/50 series and newer"

#: The oldest architecture the patched CTranslate2 wheel ships kernels for,
#: from its build flags (see `_ct2_architecture_unusable_reason`).
#:
#: ⚠ **Do not read this as "the same floor as torch".** The torch side is
#: derived from the installed wheel's arch list and moves with the pin, this one
#: is fixed until a new CT2 wheel is built, and 7.0 (Volta) is precisely the gap
#: where the two can disagree -- which is the whole reason the ASR stage asks a
#: second question rather than reusing `cuda_usable`. One `SUPPORTED_GPU_HINT`
#: still serves both messages because no *consumer* card sits at 7.0, so the
#: same sentence stays true for either floor.
CT2_MIN_COMPUTE_CAPABILITY = (7, 0)

#: The AMD architectures the patched CTranslate2 HIP wheel is built for.
#:
#: The AMD counterpart of `CT2_MIN_COMPUTE_CAPABILITY`, and a list rather than a
#: floor because HIP kernels are compiled per ``gfx`` target and do not run on
#: another one. It must equal ``CT2_HIP_GFX_TARGETS`` in
#: ``tools/wt_refine_port/ct2-build/common.sh`` (a test holds the two together),
#: which is upstream's own ROCm wheel list: RDNA2 through RDNA4 cards plus the
#: Strix APUs.
CT2_HIP_GFX_TARGETS = (
    "gfx1030",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1150",
    "gfx1151",
    "gfx1200",
    "gfx1201",
)

#: Environment variables through which a user already chose a card. When any is
#: set, the choice is theirs and `larger_amd_card_index` stays out of it.
_VISIBLE_DEVICE_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)


def is_rocm() -> bool:
    """Whether this PyTorch is the ROCm build (AMD), which still says ``cuda``."""

    return bool(getattr(torch.version, "hip", None))


def _xpu_backend():
    """``torch.xpu`` when this build has one, else ``None``.

    Read through ``getattr`` on every call rather than bound once at import: a
    build without the Intel GPU backend has no such attribute at all, and an
    ``AttributeError`` raised while this module loads would take down every
    importer of it -- a failure mode far worse than the query it serves.
    """

    return getattr(torch, "xpu", None)


def is_xpu() -> bool:
    """Whether this PyTorch build carries the Intel GPU (XPU) backend.

    ``torch.version.xpu`` is set by exactly the builds that carry it (the same
    way ``torch.version.hip`` marks ROCm), so this is a property of the
    installed wheel, not of the machine: a PyPI ``torch`` on an Intel laptop
    answers False and every XPU path stays inert.
    """

    return getattr(torch.version, "xpu", None) is not None


def _xpu_is_the_backend() -> bool:
    """Whether this torch build's GPU stack is Intel's rather than CUDA's.

    One predicate for the five call sites that need the same answer (runtime
    tag, arch tag, device name, `accelerator`, bf16). The XPU build carries
    neither CUDA nor HIP, so it is distinguishable without asking the driver --
    which matters, because some of those call sites run on a machine whose
    driver is exactly what is broken.
    """

    return is_xpu() and not (
        torch.version.cuda or getattr(torch.version, "hip", None)
    )


def _xpu_device_count() -> int:
    """How many Intel GPUs ``torch.xpu`` reports, or 0 when it cannot answer."""

    xpu = _xpu_backend()
    if xpu is None:
        return 0
    try:
        return int(xpu.device_count())
    except Exception:  # pragma: no cover - driver-level failure
        return 0


def xpu_device_present() -> bool:
    """Whether the driver reports an Intel GPU at all -- **not** whether we can use it.

    The Intel side of `cuda_device_present`, and weaker than `xpu_usable` on
    purpose: this answers the policy question ("is there a card here, so is a
    GPU tier meaningful"), while `xpu_usable` answers the placement question
    ("may work go on it now"). A stock PyPI ``torch`` on an Intel machine is
    exactly the gap between the two -- the driver has the card, this build has
    no XPU backend compiled in.

    Never use it for placement, and do not report it as an accelerator token:
    that is `gpu_device`, the usable answer.
    """

    return _xpu_device_count() > 0


def xpu_unusable_reason() -> Optional[str]:
    """A reason XPU cannot be used here, phrased for a user, or None if it can.

    The Intel counterpart of `cuda_unusable_reason`, kept in the same shape on
    purpose: first "was this even compiled in", then "does the driver give it a
    card". The first is the state a stock PyPI ``torch`` has on an Intel
    machine -- it imports, it runs, it just never touches the GPU -- which is
    why the hint points at the XPU index rather than at the card.

    ⚠ **There is no third question here**, unlike on the CUDA side. Intel's
    stack compiles its kernels from SPIR-V at load time instead of shipping a
    per-architecture cubin list, so there is nothing to compare a card against:
    any GPU the driver enumerates is one this build can run on.
    """

    if not is_xpu() or _xpu_backend() is None:
        return (
            f"this PyTorch has no Intel GPU support compiled in "
            f"({torch.__version__}); install it from the XPU index, see "
            f"docs/manual/repo-install.md"
        )
    if _xpu_device_count() > 0:
        return None
    return (
        "the Intel GPU driver gives torch.xpu no device (no Intel GPU present, "
        "or its driver is too old for this PyTorch build); see "
        "docs/manual/repo-install.md"
    )


def xpu_usable() -> bool:
    """True when this torch build has XPU *and* the driver hands it a card."""

    return xpu_unusable_reason() is None


@dataclass(frozen=True)
class Accelerator:
    """What GPU this machine offers, whether or not anything can run on it.

    ``vendor`` is ``"nvidia"``, ``"amd"``, ``"intel"``, ``"apple"`` or ``""``
    (none). A description, never a placement decision: `cuda_usable`,
    `xpu_usable` and `gpu_device` decide placement. Apple's GPU is reported so
    a run can say what it saw, but no stage uses MPS yet.

    ⚠ ``gpu_family`` used to be called ``cuda_family``. The rename is the
    point: Intel is a GPU family that nothing reaches through ``torch.cuda``,
    so a predicate named after that API would answer False about a card it
    exists to describe. Callers ask "is this a GPU", never "is this CUDA".
    """

    vendor: str
    name: str = ""
    arch: str = ""

    @property
    def gpu_family(self) -> bool:
        """A GPU from a vendor this pipeline can place work on (CUDA or XPU)."""

        return self.vendor in {"nvidia", "amd", "intel"}


def accelerator() -> Accelerator:
    """The GPU this machine offers torch, described for reports and support levels."""

    try:
        if int(torch.cuda.device_count()) > 0:
            return Accelerator(
                "amd" if is_rocm() else "nvidia", gpu_name(), gpu_arch_tag()
            )
    except Exception:  # pragma: no cover - driver-level failure
        pass
    if _xpu_is_the_backend() and _xpu_device_count() > 0:
        try:
            return Accelerator("intel", gpu_name(), gpu_arch_tag())
        except Exception:  # pragma: no cover - driver-level failure
            # The same guard the CUDA branch above has: a driver that counts a
            # device but fails the per-device queries still means "there is an
            # Intel card". Letting that escape would take down every stage that
            # asks where work goes, to withhold a name.
            return Accelerator("intel")
    mps = getattr(torch.backends, "mps", None)
    try:
        if mps is not None and mps.is_available():
            return Accelerator("apple", "Apple GPU (MPS)")
    except Exception:  # pragma: no cover - driver-level failure
        pass
    return Accelerator("")


def _gfx_from_override(value: str) -> Optional[str]:
    """``HSA_OVERRIDE_GFX_VERSION=10.3.0`` -> ``gfx1030``, the target it forces."""

    parts = [part for part in value.strip().split(".") if part]
    if not parts or not all(part.isdigit() for part in parts):
        return None
    return "gfx" + "".join(parts)


def _rocm_arch() -> Optional[str]:
    """The ``gfx`` target the ROCm runtime will load kernels for on this card.

    ``HSA_OVERRIDE_GFX_VERSION`` wins when set: it is exactly how RX 6000-era
    users run cards absent from a build's list, by telling the runtime to treat
    the card as a listed target -- so that target, not the card's own name, is
    what the kernel check must look for.
    """

    override = os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    if override:
        forced = _gfx_from_override(override)
        if forced:
            return forced
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # pragma: no cover - driver-level failure
        return None
    name = str(getattr(properties, "gcnArchName", "") or "")
    return name.split(":", maxsplit=1)[0].strip() or None


def _gfx_matches(entry: str, arch: str) -> bool:
    """Does a build's arch-list entry cover this card's ``gfx`` target?

    Exact names, plus the two generic spellings ROCm builds use for a family:
    ``gfx11-generic`` and ``gfx110X``.
    """

    entry = entry.strip().lower()
    arch = arch.strip().lower()
    if entry == arch:
        return True
    if entry.endswith("-generic"):
        return arch.startswith(entry.removesuffix("-generic"))
    if entry.endswith("x"):
        return arch.startswith(entry[:-1])
    return False


def _rocm_build_has_kernels_for(arch: str, entries: list[str]) -> bool:
    """Membership, not a floor: AMD kernels do not run on another ``gfx`` target.

    An empty list means the build reports nothing to compare against; as on the
    NVIDIA side, let it try.
    """

    if not entries:
        return True
    return any(_gfx_matches(entry, arch) for entry in entries)


def _arch_number(entry: str) -> Optional[int]:
    """``sm_86`` / ``compute_90a`` -> 86 / 90, the way torch itself reads them.

    Mirrors ``torch.cuda._extract_arch_version``: the number is
    ``major * 10 + minor``, and a trailing ``a``/``f`` on entries like
    ``sm_90a`` marks a variant rather than part of the number.
    """

    parts = entry.split("_", maxsplit=2)
    if len(parts) < 2:
        return None
    base = parts[1].removesuffix("a").removesuffix("f")
    return int(base) if base.isdigit() else None


def _capability_number(capability: tuple[int, int]) -> int:
    """``(8, 6)`` -> 86, the same major*10+minor `_arch_number` reads out of
    ``sm_86`` -- so a card and an arch-list entry compare as plain integers."""

    return capability[0] * 10 + capability[1]


def _build_has_kernels_for(capability: tuple[int, int]) -> bool:
    """Can the installed torch actually run on a card of this capability?

    Only the *lower* bound of the build's arch list is treated as disqualifying,
    and membership is never required. Two traps drive that:

    - Requiring an exact ``sm_XY`` match would be plainly wrong. The cu128 list
      has no ``sm_89``, yet every RTX 40-series card runs on it, because a cubin
      is binary-compatible with later minor revisions of the same major version.
      Pushing those to CPU would be a far worse bug than the one this guard is
      here for.
    - Above the top of the list, a card may still work -- via a ``compute_XY``
      PTX entry the driver JITs forward, or that same in-major compatibility. So
      let torch try. A wrong fallback costs a silent 10x slowdown; a wrong
      attempt costs a loud error, which is the better way to be wrong.

    Below the minimum there is no such escape hatch (cubins are not backward
    compatible), which is exactly the GTX 10-series case. The bound therefore
    matches the condition torch warns about in ``torch.cuda._check_capability``.
    """

    numbers = [
        number
        for number in (_arch_number(entry) for entry in torch.cuda.get_arch_list())
        if number is not None
    ]
    if not numbers:
        # Nothing to compare against (ROCm, or a build that reports no list);
        # torch skips its own check here too, so assume the card is fine.
        return True
    return _capability_number(capability) >= min(numbers)


def cuda_unusable_reason() -> Optional[str]:
    """A reason CUDA cannot be used here, phrased for a user, or None if it can."""

    if not torch.cuda.is_available():
        if not (torch.version.cuda or getattr(torch.version, "hip", None)):
            if is_xpu():
                if not xpu_usable():
                    # The build is right for this machine. What is missing is a
                    # driver or an environment variable, not a package, and the
                    # XPU reason says which -- pointing at a reinstall here sent
                    # users with a masked card (ZE_AFFINITY_MASK) to replace a
                    # torch that was already correct.
                    return xpu_unusable_reason()
                # A real GPU build, just not one that can see CUDA. Saying "no
                # GPU support compiled in" would be false on the machine this
                # branch exists for: an Intel card whose torch is the XPU build.
                return (
                    "this PyTorch is the Intel GPU (XPU) build and carries no "
                    "CUDA support; see docs/manual/repo-install.md"
                )
            # The one failure that looks exactly like "no card" but is not: a
            # torch installed from PyPI instead of the vendor index. It has no
            # GPU support compiled in, so every stage silently runs on the CPU
            # on a machine that has a perfectly good card. Both vendors reach
            # it -- `pip install -e ".[asr]"` over a ROCm torch replaces it
            # with this, which is why AMD on Windows has its own extra.
            return (
                f"this PyTorch has no GPU support compiled in "
                f"({torch.__version__}); install it from the index for your "
                f"card, see docs/manual/repo-install.md"
            )
        return "it is unavailable"
    if is_rocm():
        warn_if_another_amd_card_is_larger()
        try:
            arch = _rocm_arch()
            entries = list(torch.cuda.get_arch_list())
            if arch is None or _rocm_build_has_kernels_for(arch, entries):
                return None
            name = torch.cuda.get_device_name()
        except Exception as exc:  # pragma: no cover - driver-level failure
            return f"the AMD device could not be queried ({exc})"
        return (
            f"{name} ({arch}) is not among the AMD architectures this ROCm build of "
            f"PyTorch has kernels for ({', '.join(entries)}); see docs/manual/resources.md"
        )
    try:
        capability = torch.cuda.get_device_capability()
    except Exception as exc:  # pragma: no cover - driver-level failure
        return f"the CUDA device could not be queried ({exc})"
    if _build_has_kernels_for(capability):
        return None
    # Only now is the device name worth the call: this runs on every stage.
    major, minor = capability
    return (
        f"{torch.cuda.get_device_name()} (compute capability {major}.{minor}) is "
        f"older than anything this PyTorch build has kernels for; it needs "
        f"{SUPPORTED_GPU_HINT}"
    )


def cuda_usable() -> bool:
    """True when CUDA is present *and* this build has kernels for the card.

    The replacement for a bare ``torch.cuda.is_available()`` anywhere the answer
    decides whether real work goes to the GPU.
    """

    return cuda_unusable_reason() is None


def _device_index_refusal(device: str) -> str:
    """The one sentence both resolvers say about an indexed device string.

    Shared because the two resolvers must refuse the same spellings for the
    same reason, and a message that drifted between them would make one stage
    look more permissive than the other.
    """

    return (
        f"--device accepts 'cpu', 'cuda' or 'xpu', not {device!r}; select a "
        f"specific GPU with the CUDA_VISIBLE_DEVICES / ZE_AFFINITY_MASK "
        f"environment variable."
    )


#: The three spellings a front end may hand any stage: the CPU, and the two
#: accelerator families this project can drive. An index is not one of them --
#: see `_device_index_refusal`.
_DEVICE_TOKENS = frozenset({"cpu", "cuda", "xpu"})


def validate_device_token(device: str | None) -> None:
    """Refuse a ``--device`` spelling no stage can honour, before any work runs.

    The two resolvers below already refuse an indexed string, but they answer
    per stage -- and vocal separation asks before the ASR stage does. On a
    machine with a card, that meant ``--device cuda:0`` paid for the whole
    separation pass and then died with the ValueError it could have raised in
    the first second (found in the Intel acceptance run, 2026-10-22).

    ``None`` is "not chosen" and is accepted: every front end passes it through
    so `run_pipeline`'s own default stays the only place that names a card.
    """

    if device is None:
        return
    if str(device).strip().lower() in _DEVICE_TOKENS:
        return
    raise ValueError(_device_index_refusal(str(device)))


_CT2_INTEL_WARNED: "weakref.WeakKeyDictionary[object, bool]" = weakref.WeakKeyDictionary()


def _warn_ct2_has_no_intel_backend() -> None:
    """Say why the ASR stage stays on the CPU on an Intel GPU machine.

    CTranslate2 -- the ASR binary, see `resolve_asr_device` -- ships CUDA and
    oneDNN builds only, with no Intel GPU backend at all. The other accelerated
    stages do run on the card, which is why the sentence names them: a user who
    sees "the GPU is in use" for separation is owed the reason transcription is
    not, or they will spend an evening trying to fix a design limit. The tail
    qwen-verify follows the ASR device (`tail_verify_device`), so it is named
    with ASR rather than with the referee that re-decodes languages.

    Once per reporter, the rule `support_levels.note_accelerated_stage`
    follows: a run asks `resolve_asr_device` more than once (the recall repair
    asks again), and a batch binds a fresh reporter per task.
    """

    reporter = current_reporter()
    try:
        if _CT2_INTEL_WARNED.get(reporter):
            return
    except TypeError:  # a reporter that cannot be weakly referenced
        pass
    reporter.warning(
        "cpu-fallback",
        "CTranslate2 has no Intel GPU backend, so ASR and the qwen-verify check "
        "after it run on the CPU; vocal separation, the Silero VAD assist and "
        "the Qwen referee that re-decodes languages use the Intel GPU.",
        impact="转写与转写后的 qwen-verify 用不上显卡，其余阶段照常",
    )
    if reporter_delivers(reporter):
        try:
            _CT2_INTEL_WARNED[reporter] = True
        except TypeError:
            pass


#: Local MOSS (`--asr moss-local`) runs on NVIDIA cards from this compute capability on: below it bf16 has no
#: tensor cores and the prefill's SDPA falls back to a kernel whose score matrix does not fit (owner 2026-11-03).
MOSS_LOCAL_MIN_COMPUTE_CAPABILITY = (8, 0)
MOSS_LOCAL_GPU_HINT = ("an NVIDIA card of compute capability 8.0 or newer (RTX 30 series and newer) with the free VRAM "
                       "of the standard tier (an 8 GB card)")
#: The free VRAM it needs (`standard`'s figure): below it the weights, the prefill's peak and the margin leave no
#: room for a minute of audio, so the run would fail after the 1.7 GB download and the load.
MOSS_LOCAL_MIN_USABLE_GIB = 6.5


def moss_local_unusable_reason(gpu_tier: Optional[str], device: Optional[str]) -> Optional[str]:
    """Why this run cannot decode with local MOSS, as the sentence to stop on, or None if it can.

    Unlike every other stage there is no CPU fallback: someone who picked MOSS gets MOSS or this error, not
    another model behind their back. `gpu_tier` and `device` are the run's `--gpu-tier` and `--device`.
    """

    from .resources import AUTO_GPU_TIER, RESOURCE_PROFILES, get_resource_profile

    tier = str(gpu_tier or "").strip().lower()
    reason = None
    if str(device or "").strip().lower() == "cpu":
        reason = "--device cpu keeps this run off the GPU"
    elif tier not in ("", AUTO_GPU_TIER) and tier in RESOURCE_PROFILES and not RESOURCE_PROFILES[tier].gpu:
        reason = f"--gpu-tier {tier} keeps this run off the GPU"
    elif (unusable := cuda_unusable_reason()) is not None:
        reason = f"the GPU cannot be used ({unusable})"
    elif is_rocm():
        reason = f"{gpu_name()} is an AMD card"
    else:
        try:
            capability = tuple(torch.cuda.get_device_capability())
        except Exception as exc:  # pragma: no cover - driver-level failure
            capability, reason = None, f"the CUDA device could not be queried ({exc})"
        if capability is not None and capability < MOSS_LOCAL_MIN_COMPUTE_CAPABILITY:
            reason = f"{gpu_name()} has compute capability {capability[0]}.{capability[1]}"
        elif reason is None:
            profile = get_resource_profile(gpu_tier, device=device)
            if profile.usable_gib < MOSS_LOCAL_MIN_USABLE_GIB:
                reason = f"the run's GPU tier ({profile.gpu_tier}) gives it {profile.usable_gib:g} GiB of VRAM"
    if reason is None:
        return None
    return (f"--asr moss-local needs {MOSS_LOCAL_GPU_HINT}, but {reason}; use --asr local (Whisper) or a cloud "
            "backend (docs/manual/joint-asr.md)")


def multiprocessor_count() -> int:
    """Streaming multiprocessors on the current card (a kernel's grid is sized by it)."""

    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


def resolve_asr_device(requested_device: str, *, gpu_allowed: bool = True) -> str:
    """Where the ASR stage will decode -- answered by CTranslate2, not torch.

    Deliberately not `resolve_device`. That one answers for torch, and this
    stage does not decode with torch: it decodes with the patched CTranslate2,
    a separate binary with its own CUDA build and its own architecture list.
    Routing the question through torch got it wrong in both directions -- a
    CPU-only CT2 wheel on a working card failed at the first encode, and a card
    too old for this torch build sent CT2 to the CPU for no reason.

    Four questions, in order, because they are not the same question:

    1. **Intent** -- `--device cpu` is a request, and it is honoured whatever
       the hardware can do.
    2. **Policy** -- `--gpu-tier cpu` says this run does not use the GPU.
    3. **Intel** -- and before asking CTranslate2 anything: no build of it has
       an Intel GPU backend, so on a machine whose accelerator is Intel the
       answer is the CPU no matter what the wheel says. It gets its own
       warning, naming the stages that *do* use the card, because nothing here
       is broken and the user would otherwise go looking for a fix.
    4. **Capability** -- and only then, can CTranslate2 use the card.

    ⚠ Passing here does not promise the model will build: the patched wheel
    loads cuBLAS by name at runtime, which this cannot see. The warm-up encode
    in `FwRefineModelPool.warm` is what turns that into an early, legible
    failure.
    """

    device = str(requested_device or "cuda")
    normalized = device.strip().lower()
    if normalized.startswith("xpu"):
        if normalized != "xpu":
            raise ValueError(_device_index_refusal(device))
        if not gpu_allowed:
            return "cpu"
        reason = xpu_unusable_reason()
        if reason is not None:
            current_reporter().warning(
                "cpu-fallback",
                f"XPU requested for ASR but {reason}; falling back to CPU.",
                impact="速度会显著下降",
            )
            return "cpu"
        # The card works; the ASR binary just has no backend for it.
        _warn_ct2_has_no_intel_backend()
        return "cpu"
    if not normalized.startswith("cuda"):
        return device
    if normalized != "cuda":
        raise ValueError(_device_index_refusal(device))
    if not gpu_allowed:
        return "cpu"
    if xpu_usable() and not cuda_usable():
        # `"cuda"` is the historical spelling of "the accelerator", so this is
        # the Intel machine reaching ASR through a request it did not spell
        # `"xpu"`. The card it means cannot take ASR either way.
        _warn_ct2_has_no_intel_backend()
        return "cpu"
    reason = ct2_cuda_unusable_reason()
    if reason is None:
        return "cuda"
    current_reporter().warning(
        "cpu-fallback",
        f"CUDA requested for ASR but {reason}; falling back to CPU.",
        impact="速度会显著下降",
    )
    return "cpu"


def ct2_cuda_unusable_reason() -> Optional[str]:
    """Why CTranslate2 cannot use CUDA here, or None if it can.

    The ASR stage's real requirement, and **not** the same question
    `cuda_usable` answers: CTranslate2 is a separate binary with its own CUDA
    build and its own architecture list. Asking torch about it is how a machine
    with a working card and a CPU-only CT2 wheel ended up handing `"cuda"` to
    CTranslate2 and failing at the first encode.

    Two questions, because the library answers only the first: does the driver
    show it a device, and is that device's architecture inside the wheel's
    kernel list. `get_cuda_device_count` is `cudaGetDeviceCount` -- it counts
    what the driver enumerates and knows nothing about which cubins were built,
    so on a GTX 10-series card it answers 1 and the decode then dies inside the
    first encode with `no kernel image is available for execution on the
    device`. That is a hard failure where a CPU fallback was the promise, so the
    floor below is checked here rather than discovered there.

    ⚠ **Necessary, never sufficient.** The patched build loads cuBLAS by name at
    runtime (`cuda_libs`), and this query answers the same whether or not that
    directory is on the search path -- so a pass here does not promise the model
    will build. What guarantees an early, legible failure is the warm-up encode
    in `FwRefineModelPool.warm`, not this.

    Imported lazily on purpose: `import ctranslate2` costs ~9 s cold on the
    reference machine while the query itself is ~25 ms, so a run that never
    reaches the ASR stage must not pay for it.
    """

    try:
        import ctranslate2
    except Exception as exc:  # pragma: no cover - the stage reports for real
        return f"CTranslate2 could not be imported ({exc})"
    try:
        if int(ctranslate2.get_cuda_device_count()) > 0:
            return _ct2_architecture_unusable_reason()
    except Exception as exc:  # pragma: no cover - driver-level failure
        return f"CTranslate2 could not be queried ({exc})"
    if is_rocm():
        return (
            "this CTranslate2 build reports no GPU device (an AMD card needs the "
            "ROCm/HIP build of the patched wheel); see docs/manual/ct2-wheel.md"
        )
    return (
        "this CTranslate2 build reports no CUDA device (a CPU-only wheel, or a "
        "driver it cannot see); see docs/manual/ct2-wheel.md"
    )


def _ct2_architecture_unusable_reason() -> Optional[str]:
    """Whether the card is too old for the patched wheel's kernels.

    A hard-coded floor, unlike `_build_has_kernels_for`'s derived one, because
    CTranslate2 exposes no equivalent of `torch.cuda.get_arch_list()`: the
    number comes from the wheel's own build flags
    (`CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"`, recorded in
    `tools/wt_refine_port/ct2-patches/README.md`) and moves only when a new
    wheel is built. Bump it there and here together.

    The same two asymmetries as the torch side, for the same reasons:

    - Only the floor disqualifies. Above the top of the list a card runs on the
      `9.0+PTX` entry the driver JITs forward -- that is how sm_120 decodes
      today -- so a capability this build never heard of must be let through.
    - Membership is not required either: a cubin is binary-compatible with
      later minor revisions of the same major version.

    The capability comes from torch only as a *hardware* reading; whether torch
    itself could run there is a different question, already answered by
    `cuda_unusable_reason`. When torch cannot see the device at all there is
    nothing to compare, and the answer stays what it was before this check
    existed: let CTranslate2 try, and let the warm-up encode report.

    On AMD the question is membership in `CT2_HIP_GFX_TARGETS` instead, for the
    reason `_rocm_build_has_kernels_for` gives; the compute-capability numbers
    torch reports there mean nothing to a floor built for NVIDIA.
    """

    try:
        if not torch.cuda.is_available():
            return None
        if is_rocm():
            arch = _rocm_arch()
            if arch is None or _rocm_build_has_kernels_for(
                arch, list(CT2_HIP_GFX_TARGETS)
            ):
                return None
            return (
                f"this CTranslate2 build has no kernels for {arch}; it is built for "
                f"{', '.join(CT2_HIP_GFX_TARGETS)} (set HSA_OVERRIDE_GFX_VERSION to "
                f"one of them if your card is close to it)"
            )
        capability = torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - driver-level failure
        return None
    if _capability_number(capability) >= _capability_number(
        CT2_MIN_COMPUTE_CAPABILITY
    ):
        return None
    major, minor = capability
    return (
        f"this CTranslate2 build has no kernels for compute capability "
        f"{major}.{minor}; it needs {SUPPORTED_GPU_HINT}"
    )


#: Developer switch for flash attention in the ASR encoder: ``auto`` (the
#: default), ``on`` or ``off``. Read only by `ct2_flash_attention`.
CT2_FLASH_ATTENTION_ENV = "FINESUB_CT2_FLASH_ATTENTION"

#: The oldest card CTranslate2's flash-attention kernels run on (Ampere);
#: below it the model load itself raises.
CT2_FLASH_ATTENTION_MIN_CAPABILITY = (8, 0)

#: The compute types whose activations are float16: the patched build has the
#: fp16 flash-attention kernels only (patch 0013), and CTranslate2 refuses to
#: load a flash model under float32 at all.
_CT2_FLASH_ATTENTION_COMPUTE_TYPES = frozenset({"float16", "int8_float16"})


def ct2_flash_attention(device: str, compute_type: str) -> bool:
    """Whether to load the ASR model with flash attention in its encoder.

    ``FINESUB_CT2_FLASH_ATTENTION`` picks the mode and the default is ``auto``
    (owner 2026-10-10): on wherever it can run, silently off elsewhere.
    ``on`` does the same and says why when it cannot; ``off`` keeps the
    standard attention, for comparing or bisecting. Flash attention is an
    execution-form change (README_DEV.md development principles): the encoder
    output differs at fp16-noise level, and through the discrete alignment
    choices a few cue boundaries can move (up to ~0.8 s on the parity
    fixtures; the text was unchanged). Neither mode fails a run over anything
    checked here; the model's head dimension is not (only the converted
    model knows it), so a custom conversion that is not 64 fails its first
    encode with a message naming the ``off`` switch (`_flash_build_limit`).

    Each condition is one CTranslate2 would otherwise enforce later and worse:
    a build without ``WITH_FLASH_ATTN`` accepts the option and raises only at
    the first encode, which the alignment loop reads as "drop this group";
    a pre-Ampere card, a ROCm build, the CPU or a float32 compute type make the
    load itself raise. In the patched build the option reaches the encoder
    only (patch 0012); the decoder keeps the standard attention and cache
    layout, but reads the encoder's output, so its numbers move with it.
    """

    raw = os.environ.get(CT2_FLASH_ATTENTION_ENV, "").strip().lower()
    mode = raw or "auto"
    if mode not in {"auto", "on", "off"}:
        current_reporter().warning(
            "ct2-flash-attention-unavailable",
            f"{CT2_FLASH_ATTENTION_ENV}={raw!r} is not auto, on or off; flash "
            f"attention stays off.",
        )
        return False
    if mode == "off":
        return False
    reason = _ct2_flash_attention_unavailable_reason(device, compute_type)
    if reason is None:
        return True
    if mode == "on":
        current_reporter().warning(
            "ct2-flash-attention-unavailable",
            f"{CT2_FLASH_ATTENTION_ENV}=on but {reason}; the ASR model loads "
            f"without flash attention.",
        )
    return False


def _ct2_flash_attention_unavailable_reason(
    device: str, compute_type: str
) -> Optional[str]:
    """Why flash attention cannot run for this load, or None if it can."""

    if not str(device).strip().lower().startswith("cuda"):
        return "the ASR model runs on the CPU"
    if is_rocm():
        return "CTranslate2 has no flash attention on ROCm"
    if str(compute_type).strip().lower() not in _CT2_FLASH_ATTENTION_COMPUTE_TYPES:
        return f"it needs a float16 compute type, not {compute_type}"
    try:
        import ctranslate2

        compiled = bool(ctranslate2.flash_attention_available())
    except AttributeError:
        compiled = False
    except Exception as exc:  # pragma: no cover - the stage reports for real
        return f"CTranslate2 could not be queried ({exc})"
    if not compiled:
        return "this CTranslate2 build has no flash attention (WITH_FLASH_ATTN)"
    try:
        if not torch.cuda.is_available():
            return "torch sees no CUDA device to check the card against"
        # Device 0, explicitly: the ASR model loads on CTranslate2's default
        # device_index 0 (no caller passes another), and both libraries
        # number the same visible set (CUDA_VISIBLE_DEVICES), while torch's
        # *current* device is per thread and could have been moved.
        capability = torch.cuda.get_device_capability(0)
    except Exception as exc:  # pragma: no cover - driver-level failure
        return f"the card could not be queried ({exc})"
    if _capability_number(capability) < _capability_number(
        CT2_FLASH_ATTENTION_MIN_CAPABILITY
    ):
        major, minor = capability
        return (
            f"the card is compute capability {major}.{minor} and flash attention "
            f"needs 8.0 (RTX 30 series) or newer"
        )
    return None


def cuda_device_present() -> bool:
    """Whether the driver reports a CUDA device at all -- **not** whether we can use it.

    Deliberately weaker than :func:`cuda_usable`, and the two must not be
    confused. This one answers a *policy* question ("does this machine have a
    GPU, so is a GPU tier even meaningful"); `cuda_usable` answers a
    *placement* question ("may torch put work on it"). A card too old for this
    PyTorch build is present but unusable: the tier should still be a GPU tier,
    because another backend -- the patched CTranslate2, which has its own arch
    list -- may still run on it, and folding torch's verdict into the tier would
    decide that for every stage at once.

    Never use it for placement. That is what `cuda_usable` is for, and the
    reason a bare ``torch.cuda.is_available()`` is banned everywhere else.

    ⚠ **`device_count`, not `is_available`.** They disagree, and the case where
    they do is one this project has to get right. Measured 2026-09-02:

    ======================= ================ ================ ==============
    ``CUDA_VISIBLE_DEVICES``  ``is_available``  ``device_count``  CT2 count
    ======================= ================ ================ ==============
    ``-1``                    False            0                0
    ``""`` (empty)            **True**         **0**            **1**
    ======================= ================ ================ ==============

    The empty spelling is the one the module docstring above names as a way to
    have no usable CUDA, and `is_available()` answers True to it -- which put a
    machine with zero visible devices on a GPU tier. `device_count()` is the
    question actually being asked.

    (CTranslate2 still reports a device under the empty spelling: its runtime
    reads that as "unset". The two libraries genuinely disagree there, which is
    exactly why the ASR stage asks CT2 and the tier asks torch -- neither is
    speaking for the other.)
    """

    try:
        return int(torch.cuda.device_count()) > 0
    except Exception:  # pragma: no cover - driver-level failure
        return False


def gpu_device_present() -> bool:
    """Whether the driver reports **any** accelerator, NVIDIA/AMD or Intel.

    The presence counterpart of `gpu_device`, and the one `detect_gpu_tier`
    asks: present means "the driver enumerated a card", usable means "this
    build can actually compute on it". They differ in both directions, and both
    directions decide whether the machine gets a GPU tier at all -- but only one
    of them is a CPU machine:

    * a CUDA card too old for this torch build is **present and unusable**. It
      still gets a GPU tier, because CTranslate2 has its own arch list and may
      run it; `total_vram_gib()` cannot size it, so `detect_gpu_tier` lands on
      ``DEFAULT_GPU_TIER``.
    * a stock PyPI ``torch`` on an Intel machine is **absent and unusable**: this
      build has no ``torch.xpu`` to ask, so there is nothing to report presence
      from. That machine falls to the CPU tier and gets the unused-card warning,
      which is correct -- the driver does have the card, and the warning is how
      the user finds out.

    ⚠ **Use it for policy, never for placement.** A machine can be present and
    unusable and still want a GPU tier, because CTranslate2 is a separate
    binary with its own arch list: it may run a card torch cannot. Placement
    asks `gpu_device`.
    """

    return cuda_device_present() or xpu_device_present()


def gpu_device() -> str:
    """The accelerator this run may put work on: ``"cuda"``, ``"xpu"`` or ``""``.

    The token the stages place work with, and the reason they no longer write
    ``"cuda"`` by hand: with Intel in the picture that is a fact about the
    machine, and this module is the one place allowed to learn it. It answers
    **usable**, not present -- the distinction `cuda_device_present` draws in
    the other direction: a card this torch build has no kernels for is not
    somewhere work may go, while it *is* still a card the tier should count.

    ``"cuda"`` wins when both could answer, which is what keeps NVIDIA and AMD
    runs on the same token they had before Intel existed.
    """

    if cuda_usable():
        return "cuda"
    if xpu_usable():
        return "xpu"
    return ""


def device_type(device: str | None) -> str:
    """The family of a device string: ``"cuda"``, ``"xpu"`` or ``"cpu"``.

    For handing to ``torch.device(...)``, ``autocast`` and the cache helpers --
    not for deciding whether work may go there. ``None`` is ``"cpu"``: every
    caller that holds a None means "this stage placed nothing on a card", and a
    caller that wants the default accelerator says ``"cuda"`` (which
    `resolve_device` maps to the Intel card on an Intel-only machine). Reading
    None as "the accelerator" made a CPU fallback look like a CPU choice and
    silenced its warning.
    """

    text = str(device or "cpu").strip().lower()
    if text == "cuda" or text.startswith("cuda:"):
        return "cuda"
    if text == "xpu" or text.startswith("xpu:"):
        return "xpu"
    return "cpu"


def is_gpu_device(device: str | None) -> bool:
    """Whether a device string names a GPU family: cuda, cuda:N, xpu or xpu:N.

    The replacement for ``startswith("cuda")`` in every gate that means "the
    user asked for the accelerator": after Intel that test silently routes an
    ``xpu`` request to the CPU-only path, which is the failure this exists to
    prevent.
    """

    return device_type(device) != "cpu"


def empty_device_cache(device: str | None = None) -> None:
    """Drop the caching allocator's free blocks on the device's family.

    A placement courtesy, never a decision: it gives the *next* allocation back
    memory the current one no longer needs. So it must never raise -- a backend
    that cannot answer is no reason to fail a run over housekeeping.
    """

    kind = device_type(device)
    try:
        if kind == "xpu":
            xpu = _xpu_backend()
            if xpu is not None:
                xpu.empty_cache()
        elif kind == "cuda":
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - driver-level failure
        pass


def synchronize(device: str | None = None) -> None:
    """Wait for queued work on the device's family; ``cpu`` is a no-op.

    Unlike `empty_device_cache` it raises: an error here is a kernel queued
    earlier failing asynchronously, and the separator warm-up syncs precisely
    so that such a failure surfaces there rather than mid-stage.
    """

    kind = device_type(device)
    if kind == "xpu":
        xpu = _xpu_backend()
        if xpu is not None:
            xpu.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()


def _gpu_family_and_index(
    device: Optional[str],
) -> Optional[tuple[str, Optional[int]]]:
    """The family and in-string index a GPU memory helper should act on.

    ``device`` is a placement string (``"cuda"``, ``"xpu"``, ``"cuda:1"``, ...)
    or None for a stage that placed nothing on a card. The index is Optional because an unindexed
    request means the backend's *current* device, which only the backend can
    name. Returns None for the CPU family, where there is nothing to ask.
    """

    kind = device_type(device)
    if kind == "cpu":
        return None
    text = "" if device is None else str(device).strip().lower()
    prefix = kind + ":"
    if not text.startswith(prefix):
        return kind, None
    try:
        return kind, int(text[len(prefix) :])
    except (TypeError, ValueError):
        return kind, None


def reset_peak_gpu_memory_stats(device: Optional[str] = None) -> None:
    """Zero the GPU allocator's peak counters so the next stage starts clean.

    The peak figures below are read by the resource-usage report, and torch's
    counters are peak-since-reset, not per stage: without this, stage two
    inherits stage one's spike and the budget warning becomes permanent noise.
    So it sits beside `peak_gpu_memory_bytes` in this module -- that pair is one
    question ("how big did this stage get on the card") split by vendor, and
    the source guard keeps every ``torch.cuda``/``torch.xpu`` probe here.

    ``device`` is the same placement string the stage ran on, so a run on
    ``"xpu"`` resets the Intel allocator rather than silently doing nothing.

    Never raises. This runs at stage boundaries, i.e. in `finally` paths; a
    driver that cannot answer must not turn bookkeeping into a crash.
    """

    resolved = _gpu_family_and_index(device)
    if resolved is None:
        return
    kind, index = resolved
    try:
        if kind == "xpu":
            xpu = _xpu_backend()
            if xpu is None:  # pragma: no cover - family implies the backend
                return
            if index is None:
                xpu.reset_peak_memory_stats()
            else:
                xpu.reset_peak_memory_stats(index)
        elif cuda_usable():
            if index is None:
                torch.cuda.reset_peak_memory_stats()
            else:
                torch.cuda.reset_peak_memory_stats(index)
    except Exception:  # pragma: no cover - driver-level failure
        pass


def peak_gpu_memory_bytes(device: Optional[str] = None) -> Optional[int]:
    """Largest GPU allocation this stage reached, or 0 when there is no GPU.

    ``max`` of the allocator's own figures, which are the ones that move when a
    stage loads a model; driver-level totals are a different question (`free_vram_gib`).
    The three return values are a contract with the usage report:

    * ``0`` -- no usable accelerator, so there is nothing to report (a CPU run
      must not print ``N/A`` for a number that does not apply);
    * ``None`` -- there is a card but the runtime could not be read;
    * a positive int -- the peak, in bytes.

    ``device`` is the placement string of the stage being measured; None is a
    stage that ran on the CPU, which reads 0.
    """

    resolved = _gpu_family_and_index(device)
    if resolved is None:
        return 0
    kind, index = resolved
    try:
        if kind == "xpu":
            if not xpu_usable():
                return 0
            xpu = _xpu_backend()
            if xpu is None:  # pragma: no cover - usability implies the backend
                return 0
            get_allocated = xpu.max_memory_allocated
            get_reserved = xpu.max_memory_reserved
        else:
            if not cuda_usable():
                return 0
            get_allocated = torch.cuda.max_memory_allocated
            get_reserved = torch.cuda.max_memory_reserved
        if index is None:
            return max(int(get_allocated()), int(get_reserved()))
        return max(int(get_allocated(index)), int(get_reserved(index)))
    except Exception:  # pragma: no cover - driver-level failure
        return None


def resolve_device(requested_device: str, *, context: str = "VAD-ASR") -> str:
    """Honour an accelerator request only when this machine can actually run it.

    Three answers, one per request shape (see `docs/manual/resources.md`):

    * ``"cuda"`` is the historical token for "use the accelerator". On a
      machine whose accelerator is Intel it resolves to ``"xpu"`` silently --
      the user asked for the card, not for a vendor, and an AMD machine already
      reaches the same code under the ``cuda`` name for the same reason. No
      warning: nothing went wrong, nothing was downgraded.
    * ``"xpu"`` is the explicit Intel request: honoured when it works, and
      answered with a warning plus the CPU when it does not.
    * ``"cpu"`` (and any other string) is honoured as given.

    ⚠ An empty request here is a *request* not yet made, so it means
    ``"cuda"`` (as in `resolve_asr_device`) -- the opposite of the placement
    helpers (`device_type` and the ones after it), where None is a stage that
    put nothing on a card. Callers resolve first and hand the helpers the
    answer, so the two readings never meet on one value.

    An index (``cuda:1`` / ``xpu:1``) is refused rather than passed through.
    Nothing here plumbs a device index: the capability checks read the
    *current* device, the ASR backend hands the string to CTranslate2 which
    rejects anything but ``"cuda"`` (it takes the index as a separate
    device_index), and every CLI advertises one name per family. Accepting
    ``cuda:1`` would mean falling back to CPU on the wrong card's verdict, then
    failing with an opaque ValueError from inside faster-whisper.

    Warns through the bound reporter and returns ``"cpu"`` otherwise. Called
    per stage, so a pipeline run says once per stage which device it settled
    on.
    """

    device = str(requested_device or "cuda")
    normalized = device.strip().lower()
    if normalized.startswith("xpu"):
        if normalized != "xpu":
            raise ValueError(_device_index_refusal(device))
        reason = xpu_unusable_reason()
        if reason is None:
            return "xpu"
        current_reporter().warning(
            "cpu-fallback",
            f"XPU requested for {context} but {reason}; falling back to CPU.",
            impact="速度会显著下降",
        )
        return "cpu"
    if not normalized.startswith("cuda"):
        return device
    if normalized != "cuda":
        raise ValueError(_device_index_refusal(device))
    if cuda_usable():
        return device
    if xpu_usable():
        # The accelerator the user asked for is there, under another token.
        return "xpu"
    reason = cuda_unusable_reason()
    current_reporter().warning(
        "cpu-fallback",
        f"CUDA requested for {context} but {reason}; falling back to CPU.",
        impact="速度会显著下降",
    )
    return "cpu"


def free_vram_gib() -> Optional[float]:
    """VRAM free on the card *right now*, in GiB, or None when none answers.

    The live counterpart to :func:`total_vram_gib`. Deliberately not used to
    pick a tier: what is free swings with whatever else the machine has open,
    and a tier decides the separator's worker count, which decides the block
    plan -- so choosing from it would make the same file produce different
    artifact boundaries on different days.

    Two callers, and the line between them is what makes that rule hold:

    * *Telling the user*, where being current is the whole value.
    * *Placing the second-model referee* (`lang_redecode.referee_device`,
      question 5). Admissible for the same reason the tier is not: the referee
      only ever produces evidence, so CPU-versus-GPU changes how long it takes
      and nothing about what it says or about any artifact's shape. It is a
      veto only -- it can move the referee off the card, never onto one the
      earlier questions ruled out.

    Anything that would change an *output* from this figure belongs in neither
    list. `referee_vram_budget` is the near miss to watch: it decides whether
    the referee compiles its decode step, and those two paths are not
    bit-exact.

    ⚠ On an Intel GPU this figure is a slice of **system memory**, not
    dedicated VRAM (`torch.xpu.mem_get_info`), and it moves with whatever else
    the machine is doing. That is still the right question here -- "how much
    will the driver hand out right now" -- but it is why `tier_for_vram` does
    not treat the reported size as a budget on that hardware.
    """

    token = gpu_device()
    if token == "xpu":
        xpu = _xpu_backend()
        try:
            free_bytes, _total = xpu.mem_get_info()
        except Exception:  # pragma: no cover - driver-level failure
            return None
        return float(free_bytes) / float(1024**3)
    if token != "cuda":
        return None
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover - driver-level failure
        return None
    return float(free_bytes) / float(1024**3)


def total_vram_gib() -> Optional[float]:
    """Whole-card VRAM in GiB, or None when no usable accelerator answers.

    Dispatches on `gpu_device` rather than asking CUDA directly: the Intel
    counterpart is `torch.xpu.get_device_properties(...).total_memory`, same
    question, different runtime.

    Lives here for the same reason ``cuda_usable`` does: it is a question about
    *this machine's* GPU, and answering it anywhere else means answering it from
    a bare ``is_available()`` again. Gated on usability rather than presence so a
    card this build has no kernels for reports nothing -- its VRAM is not a
    budget anyone can spend.

    ⚠ On Intel the number is shared system memory, not dedicated VRAM; the
    tier layer knows that and stops `auto` at `standard` there.
    """

    token = gpu_device()
    if token == "xpu":
        xpu = _xpu_backend()
        try:
            properties = xpu.get_device_properties(xpu.current_device())
        except Exception:  # pragma: no cover - driver-level failure
            return None
        return float(properties.total_memory) / float(1024**3)
    if token != "cuda":
        return None
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # pragma: no cover - driver-level failure
        return None
    return float(properties.total_memory) / float(1024**3)


#: AMD architectures with native bfloat16, as a floor within each family.
#: RDNA3 (``gfx1100``) brought it to the consumer line -- RDNA2 cards like
#: ``gfx1030`` emulate it, slowly. The CDNA line has had it since MI100
#: (``gfx908``), and those names are three characters, which is what keeps the
#: two families apart here. ⚠ The CDNA half is reasoning, not measurement: no
#: such card has run this.
_ROCM_BF16_CONSUMER_FLOOR = 1100


def _rocm_has_native_bf16(arch: str) -> bool:
    """Whether a ``gfx`` target computes bfloat16 natively."""

    target = arch.strip().lower().removeprefix("gfx")
    if not target:
        return False
    if len(target) <= 3:
        # CDNA, where the last character can be a letter (gfx90a sits between
        # gfx908 and gfx942), so the name reads as hex -- which is how AMD
        # orders it.
        try:
            return int(target, 16) >= 0x908
        except ValueError:
            return False
    if not target.isdigit():
        return False
    return int(target) >= _ROCM_BF16_CONSUMER_FLOOR


def bf16_supported() -> bool:
    """Whether the current card computes bfloat16 natively.

    Every card this project supports on NVIDIA does, so the answer there is
    True without asking -- keeping the referee's dtype exactly what it was.

    ⚠ **Not `torch.cuda.is_bf16_supported()`**: on ROCm that returns True for
    every card before it looks at one (`torch/cuda/__init__.py` in the pinned
    2.11: ``if torch.version.hip: return True``), so asking it would have made
    this branch unconditional and the float32 fallback dead code. The card's
    own ``gfx`` target is what says, and `_rocm_arch` already reads it --
    including the ``HSA_OVERRIDE_GFX_VERSION`` override, which is the right
    answer here too: a card forced to present itself as another target gets
    that target's kernels.

    On Intel the runtime is asked when it has an answer, because it is the one
    that knows (`torch.xpu.is_bf16_supported` exists for this) -- with
    ``including_emulation=False``, since its default answers True for any
    available card, emulated or not. When it does
    not -- an older torch, a fake in a test -- the answer is True: every Intel
    GPU this pipeline targets (Arc Xe2 and the Xe line) computes bf16 in
    hardware, and bf16 is the faster path for the referee's evidence, whose
    content the dtype only moves at fp16-noise level. Guessing False would cost
    real time on every Intel run to avoid a difference no checked criterion
    sees.
    """

    if is_rocm():
        arch = _rocm_arch()
        if arch is None:
            # Nothing to judge by. bf16 is the faster path and the dtype only
            # moves the referee's evidence, so take the same answer as before
            # this check.
            return True
        return _rocm_has_native_bf16(arch)
    if _xpu_is_the_backend():
        xpu = _xpu_backend()
        checker = getattr(xpu, "is_bf16_supported", None)
        if checker is not None:
            try:
                return bool(checker(including_emulation=False))
            except Exception:  # pragma: no cover - driver-level failure
                pass
        return True
    return True


# --- Identity strings for compile caches ------------------------------------
#
# A compiled artefact is bound to the runtime and the architecture it was built
# for. These spell both per vendor, so a cache keyed on them cannot hand an
# NVIDIA build to an AMD card, nor a CUDA build to an Intel one. The NVIDIA
# spellings are the ones the caches used before AMD existed here -- changing
# them would only force a rebuild, but there is no reason to.

def _arch_token(value: str) -> str:
    """Free text as one cache-key component: lowercase, runs of junk -> ``_``."""

    return re.sub(r"[^0-9a-z]+", "_", str(value).strip().lower()).strip("_")


def _xpu_properties():
    """``torch.xpu.get_device_properties(current_device())``, or None."""

    xpu = _xpu_backend()
    if xpu is None:
        return None
    try:
        return xpu.get_device_properties(xpu.current_device())
    except Exception:  # pragma: no cover - driver-level failure
        return None


def _xpu_arch_tag() -> str:
    """A stable architecture label for the Intel GPU, for the compile-cache key.

    ⚠ **It must not move between runs on the same card and driver.** This
    string is part of the separator's compiled-artifact cache key
    (`separator/accel.py`), so a new spelling rebuilds the cache once --
    acceptable -- but a figure that varied *per run* (index, free memory, a
    timestamp) would rebuild it every time, which is why nothing unstable is in
    here.

    The label is the PCI device id in hex (``xpu7d51`` for an Arc 130T): it is
    what actually differs between Intel GPU models, it is stable across runs,
    and it is short enough to read in a cache directory name. The
    ``architecture`` field is deliberately **not** used -- it is an opaque large
    integer (13195281408 on this card), so it would put an unreadable,
    torch-version-coupled number in the key. When the id is unavailable, fall
    back to the normalized device name so each card still gets its own bucket;
    when even that name cannot be read, ``xpuunknown`` is the one cache bucket
    this build cannot tell cards apart in.
    """

    properties = _xpu_properties()
    device_id = (
        getattr(properties, "device_id", None) if properties is not None else None
    )
    if device_id:
        try:
            return f"xpu{int(device_id):04x}"
        except (TypeError, ValueError):  # pragma: no cover - odd fake or runtime
            pass
    name = getattr(properties, "name", None) if properties is not None else None
    try:
        token = _arch_token(name or gpu_name())
    except Exception:  # pragma: no cover - driver-level failure
        token = ""
    # An all-punctuation name tokenizes to "" -- that is "no usable name", not
    # an architecture called "" (the old code returned a bare "xpu_" for it).
    return f"xpu_{token}" if token else "xpuunknown"


def gpu_runtime_version() -> Optional[str]:
    """``torch.version.cuda`` on NVIDIA, ``hip-<ver>`` on AMD, ``xpu-<ver>`` on Intel."""

    if is_rocm():
        return f"hip-{torch.version.hip}"
    if _xpu_is_the_backend():
        return f"xpu-{torch.version.xpu}"
    return torch.version.cuda


def gpu_runtime_tag() -> str:
    """``cuda12.8``/``cudanone`` on NVIDIA, ``hip6.4.x`` on AMD, ``xpu2025.0.2`` on Intel."""

    if is_rocm():
        return f"hip{torch.version.hip}"
    if _xpu_is_the_backend():
        return f"xpu{torch.version.xpu}"
    return f"cuda{torch.version.cuda or 'none'}"


def gpu_arch_tag() -> str:
    """``sm120`` for an NVIDIA card, its ``gfx`` target for AMD, ``xpu_<...>`` for Intel."""

    if is_rocm():
        return _rocm_arch() or "gfxunknown"
    if _xpu_is_the_backend():
        return _xpu_arch_tag()
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def gpu_name(index: Optional[int] = None) -> str:
    """The card's marketing name, of the current device unless an index is given."""

    if _xpu_is_the_backend():
        xpu = _xpu_backend()
        if xpu is not None:
            if index is None:
                return str(xpu.get_device_name())
            return str(xpu.get_device_name(index))
    if index is None:
        return str(torch.cuda.get_device_name())
    return str(torch.cuda.get_device_name(index))


# --- A card the OS can see and torch cannot ---------------------------------

_WARNED_UNUSED_CARD = False

#: PCI vendor ids worth reporting. Intel is reported only on an XPU build
#: (`warn_if_an_installed_card_is_unused`): elsewhere naming it would be an
#: offer most Intel iGPUs cannot take up.
_GPU_PCI_VENDORS = {"10de": "NVIDIA", "1002": "AMD", "8086": "Intel"}

#: Fields of Windows' DISPLAY_DEVICEW, for `EnumDisplayDevicesW`.
class _DisplayDeviceW(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("DeviceName", ctypes.c_wchar * 32),
        ("DeviceString", ctypes.c_wchar * 128),
        ("StateFlags", ctypes.c_uint32),
        ("DeviceID", ctypes.c_wchar * 128),
        ("DeviceKey", ctypes.c_wchar * 128),
    ]


def installed_gpu_vendors() -> frozenset[str]:
    """Vendors of the graphics cards the OS reports, ignoring what torch can do.

    Asked only when torch says there is nothing, so it cannot go through torch.
    Stdlib and read-only on both platforms: ``EnumDisplayDevicesW`` on Windows
    and PCI class 0x03 devices under ``/sys`` on Linux. macOS returns nothing --
    no stage runs on an Apple GPU, so there is nothing to report.

    ⚠ **Windows asks the display API, not the driver registry.** The registry's
    display class keeps a subkey per adapter ever installed: this machine still
    lists the RTX 5060 Ti that was physically removed in 2026-08, so a machine
    whose card left (or changed vendor) would be told forever that it has one.
    A compute card with no outputs may not enumerate here, which errs the safe
    way -- silence, the behaviour before this probe existed.
    """

    vendors: set[str] = set()
    try:
        if os.name == "nt":
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            index = 0
            while True:
                device = _DisplayDeviceW()
                device.cb = ctypes.sizeof(_DisplayDeviceW)
                if not user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
                    break
                index += 1
                identifier = str(device.DeviceID).lower()
                if not identifier.startswith("pci\\ven_"):
                    continue
                vendors.update(
                    label
                    for vendor, label in _GPU_PCI_VENDORS.items()
                    if identifier.startswith(f"pci\\ven_{vendor}")
                )
        elif sys.platform.startswith("linux"):
            for device in Path("/sys/bus/pci/devices").glob("*"):
                try:
                    if not (device / "class").read_text().strip().startswith("0x03"):
                        continue
                    vendor = (device / "vendor").read_text().strip().lower()
                except OSError:
                    continue
                label = _GPU_PCI_VENDORS.get(vendor.removeprefix("0x"))
                if label:
                    vendors.add(label)
    except Exception:  # pragma: no cover - a probe must never break a run
        return frozenset()
    return frozenset(vendors)


#: The Intel GPUs PyTorch validates its XPU build on (its "Getting Started on
#: Intel GPU" page for 2.11): the Xe iGPUs of older Core chips (Iris Xe, UHD)
#: are not among them, which is why an XPU build can still see no card.
_INTEL_XPU_RANGE = "Arc A / B 系独显与带 Arc 核显的 Core Ultra"


def warn_if_an_installed_card_is_unused() -> None:
    """Say once that a card is sitting there unused, when `auto` falls to the CPU.

    Falling to the CPU tier is right when the machine has no card, and silence
    is right with it. It is the same silence, though, when a card *is* there and
    the environment cannot reach it -- a torch without GPU support (one `[asr]`
    over a ROCm install does that), a missing or too-old driver, a
    ``CUDA_VISIBLE_DEVICES`` that hides everything. That run is many times
    slower for a reason nothing on screen mentions, which is what this says
    (owner 2026-09-17).

    An Intel card counts only on an XPU build of torch, i.e. for a user who
    set out to run on it and whose card is still unreachable (a driver, a
    ``ZE_AFFINITY_MASK``, or a card outside the range PyTorch supports). Any
    other build stays silent about it, as it did before Intel support: almost
    every machine has an Intel iGPU, most of them (Iris Xe, UHD) no XPU build
    can use, and on an NVIDIA laptop the advice to install the XPU torch would
    replace the CUDA build driving the NVIDIA card.
    """

    global _WARNED_UNUSED_CARD
    if _WARNED_UNUSED_CARD:
        return
    vendors = installed_gpu_vendors()
    if not is_xpu():
        vendors = vendors - {"Intel"}
    if not vendors:
        return
    built_for = (
        torch.version.cuda
        or getattr(torch.version, "hip", None)
        or getattr(torch.version, "xpu", None)
    )
    # Which vendors this *build* can drive, as opposed to which the machine has.
    driven = {
        "NVIDIA": bool(torch.version.cuda),
        "AMD": is_rocm(),
        "Intel": is_xpu(),
    }
    found = " / ".join(sorted(vendors))
    if any(driven.get(vendor) for vendor in vendors):
        # Build and card agree, so the card being invisible is the driver or an
        # environment variable, not the wheel. Reinstalling torch would be the
        # wrong next step for CUDA users and for XPU users alike.
        action = (
            "确认显卡驱动已安装且不过旧；若是有意屏蔽，检查 CUDA_VISIBLE_DEVICES / "
            "HIP_VISIBLE_DEVICES / ZE_AFFINITY_MASK 之类的环境变量"
        )
        if "Intel" in vendors:
            action += f"；XPU 版 PyTorch 只支持 {_INTEL_XPU_RANGE}，别的 Intel 核显只能走 CPU"
    elif not built_for:
        action = (
            f"这个 PyTorch（{torch.__version__}）不带 GPU 支持，按 {REPO_INSTALL_DOC} "
            f"从对应的索引重装 torch 三件套"
        )
    elif "AMD" in vendors and not is_rocm():
        # ⚠ An XPU build on an NVIDIA- or AMD-only machine also lands on this
        # branch or the last one, and neither names the actual mistake (the
        # wrong vendor's wheel). Left as is: nobody installs Intel's torch
        # without an Intel card, and a wrong wheel is not this warning's case.
        #
        # Naming the range matters more here than the instructions: AMD's
        # Windows ROCm covers only RX 7000/9000 and the Strix APUs, so for an
        # older card or an iGPU there is nothing to do, and a hint that does not
        # say so is one the user will chase.
        action = (
            f"A 卡要装 ROCm 版 PyTorch，而托管 CLI 装的是 NVIDIA 版；Windows 上这条路"
            f"只覆盖 RX 7000 / 9000 系与 Strix APU，且要改用源码装法（{REPO_INSTALL_DOC}"
            f"「Windows 上的 A 卡」）。不在这个范围内的卡目前只能走 CPU"
        )
    else:
        action = (
            "确认显卡驱动已安装且不过旧；若是有意屏蔽，检查 CUDA_VISIBLE_DEVICES / "
            "HIP_VISIBLE_DEVICES / ZE_AFFINITY_MASK 之类的环境变量"
        )
    reporter = current_reporter()
    reporter.warning(
        "gpu-unused",
        f"a graphics card is installed ({found}) but nothing here can use it; "
        f"this run is on the CPU",
        impact="速度会比用得上显卡时慢很多（人声分离尤其）",
        action=action,
    )
    # Only a reporter that actually shows it counts as having said it, the same
    # rule `support_levels.note_accelerated_stage` follows: a first call on a
    # thread with no bound reporter would otherwise eat the one warning.
    if reporter_delivers(reporter):
        _WARNED_UNUSED_CARD = True


# --- Which card, when there are several -------------------------------------

_WARNED_LARGER_CARD = False


def larger_amd_card_index() -> Optional[int]:
    """Another visible AMD card with more memory than device 0, or None.

    On AMD an integrated GPU is often device 0 (Ryzen 7000 desktops, most
    laptops with a discrete card), and every placement, the tier and the kernel
    check then read the iGPU. This does **not** switch cards: torch's current
    device is per thread, so a process-wide ``set_device`` would leave the
    separator's worker threads on device 0 while the model sits elsewhere, and
    CTranslate2 takes its own index. Choosing is the user's, through a
    visible-devices variable, which applies to every thread and library at once;
    this only finds the card worth pointing them at. None when a user already
    chose (any visible-devices variable set), on NVIDIA, or with one card.
    """

    if not is_rocm():
        return None
    if any(os.environ.get(name) for name in _VISIBLE_DEVICE_VARIABLES):
        return None
    try:
        count = int(torch.cuda.device_count())
        if count <= 1:
            return None
        totals = [
            int(torch.cuda.get_device_properties(index).total_memory)
            for index in range(count)
        ]
    except Exception:  # pragma: no cover - driver-level failure
        return None
    best = max(range(count), key=lambda index: (totals[index], -index))
    return best or None


def warn_if_another_amd_card_is_larger() -> None:
    """Say once per process which card to select, when device 0 is not the largest."""

    global _WARNED_LARGER_CARD
    if _WARNED_LARGER_CARD:
        return
    index = larger_amd_card_index()
    if index is None:
        return
    _WARNED_LARGER_CARD = True
    try:
        names = (torch.cuda.get_device_name(0), torch.cuda.get_device_name(index))
    except Exception:  # pragma: no cover - driver-level failure
        names = ("device 0", f"device {index}")
    current_reporter().warning(
        "amd-device-choice",
        f"using {names[0]} (device 0), but {names[1]} (device {index}) has more memory",
        impact="集成显卡可能被当成主显卡，速度与档位都按它来",
        action=f"设置 HIP_VISIBLE_DEVICES={index} 后重跑，只让程序看见那张卡",
    )
