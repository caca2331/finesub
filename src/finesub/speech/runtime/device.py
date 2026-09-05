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
"""

from __future__ import annotations

from typing import Optional

import torch

from ...reporting import current_reporter

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
        return "it is unavailable"
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


def resolve_asr_device(requested_device: str, *, gpu_allowed: bool = True) -> str:
    """Where the ASR stage will decode -- answered by CTranslate2, not torch.

    Deliberately not `resolve_device`. That one answers for torch, and this
    stage does not decode with torch: it decodes with the patched CTranslate2,
    a separate binary with its own CUDA build and its own architecture list.
    Routing the question through torch got it wrong in both directions -- a
    CPU-only CT2 wheel on a working card failed at the first encode, and a card
    too old for this torch build sent CT2 to the CPU for no reason.

    Three questions, in order, because they are not the same question:

    1. **Intent** -- `--device cpu` is a request, and it is honoured whatever
       the hardware can do.
    2. **Policy** -- `--gpu-tier cpu` says this run does not use the GPU.
    3. **Capability** -- and only then, can CTranslate2 use the card.

    ⚠ Passing here does not promise the model will build: the patched wheel
    loads cuBLAS by name at runtime, which this cannot see. The warm-up encode
    in `FwRefineModelPool.warm` is what turns that into an early, legible
    failure.
    """

    device = str(requested_device or "cuda")
    normalized = device.strip().lower()
    if not normalized.startswith("cuda"):
        return device
    if normalized != "cuda":
        raise ValueError(
            f"--device accepts 'cpu' or 'cuda', not {device!r}; select a "
            f"specific GPU with the CUDA_VISIBLE_DEVICES environment variable."
        )
    if not gpu_allowed:
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
    """

    try:
        if not torch.cuda.is_available():
            return None
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


def resolve_device(requested_device: str, *, context: str = "VAD-ASR") -> str:
    """Honour a CUDA request only when this machine can actually run it.

    Warns through the bound reporter and returns ``"cpu"`` otherwise. Called per stage, so a
    pipeline run says once per stage which device it settled on.
    """

    device = str(requested_device or "cuda")
    normalized = device.strip().lower()
    if not normalized.startswith("cuda"):
        return device
    if normalized != "cuda":
        # Refused rather than passed through. Nothing here plumbs a device
        # index: the capability check below reads the *current* device, the
        # backend hands the string to CTranslate2 which rejects anything but
        # "cuda" (it takes the index as a separate device_index), and every CLI
        # advertises cpu/cuda. Accepting "cuda:1" would mean falling back to CPU
        # on the wrong card's verdict, then failing with an opaque ValueError
        # from inside faster-whisper.
        raise ValueError(
            f"--device accepts 'cpu' or 'cuda', not {device!r}; select a "
            f"specific GPU with the CUDA_VISIBLE_DEVICES environment variable."
        )
    reason = cuda_unusable_reason()
    if reason is None:
        return device
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
    """

    if not cuda_usable():
        return None
    try:
        free_bytes, _total = torch.cuda.mem_get_info()
    except Exception:  # pragma: no cover - driver-level failure
        return None
    return float(free_bytes) / float(1024**3)


def total_vram_gib() -> Optional[float]:
    """Whole-card VRAM in GiB, or None when no usable CUDA device answers.

    Lives here for the same reason ``cuda_usable`` does: it is a question about
    *this machine's* GPU, and answering it anywhere else means answering it from
    a bare ``is_available()`` again. Gated on ``cuda_usable()`` rather than
    ``is_available()`` so a card this build has no kernels for reports nothing --
    its VRAM is not a budget anyone can spend.
    """

    if not cuda_usable():
        return None
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    except Exception:  # pragma: no cover - driver-level failure
        return None
    return float(properties.total_memory) / float(1024**3)
