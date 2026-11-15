"""Which platform + GPU combinations have been accepted, and the notice when not.

Three levels (user-facing table: ``docs/manual/resources.md``):

- **official** -- verified on the maintainer's machine;
- **community** -- a contributor sent an acceptance record the maintainer read;
- **experimental** -- installs and runs, nobody has verified it.

⚠ A level is a label and **changes no behaviour** (owner 2026-09-17): which
device a stage uses, and what is on by default, never read it. Its one effect
is a notice -- a run that put a stage on an accelerator below "official" says
so once per stage, with the way back (``--device cpu``) and where to report.
That is the only warning users on AMD or Intel get that a result may differ,
because the difference itself (e.g. the referee's decisions) is invisible in
the output.

Only GPU placements are announced. CPU paths off Windows are also unverified,
but "which stage ran on an accelerator nobody checked" is the question the
notice answers; the CPU label lives in the documentation.

The gate is `device.is_gpu_device`, not a ``cuda`` prefix: after Intel, an
``xpu`` placement is just as much an accelerator placement -- and on that
hardware it is the unverified one that most needs saying -- while a ``cpu`` one
still needs no notice.
"""

from __future__ import annotations

import platform
import sys
import weakref
from typing import Optional

from ...reporting import current_reporter, reporter_delivers

OFFICIAL = "official"
COMMUNITY = "community"
EXPERIMENTAL = "experimental"

#: (platform family, GPU vendor) pairs verified on the maintainer's machine.
#: Linux + NVIDIA is official only under WSL2, where it was verified; native
#: Linux loads the CUDA libraries through a different path (owner 2026-09-11).
_OFFICIAL: frozenset[tuple[str, str]] = frozenset({("windows", "nvidia"), ("wsl", "nvidia")})

#: Pairs with a community acceptance record. Empty until one is received; the
#: record (with the versions it was taken on) goes in docs/bench-baselines.md.
_COMMUNITY: frozenset[tuple[str, str]] = frozenset()

FEEDBACK_URL = "https://github.com/caca2331/finesub/issues"

#: Stages already announced, per reporter. Weak keys: a batch gives each item its
#: own reporter, and an id() of a collected one can be reused by the next item,
#: which would then swallow its own notice.
_ANNOUNCED: "weakref.WeakKeyDictionary[object, set[str]]" = weakref.WeakKeyDictionary()


def platform_family() -> str:
    """``windows``, ``wsl``, ``linux``, ``macos`` or ``other``."""

    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        release = platform.uname().release.lower()
        return "wsl" if "microsoft" in release else "linux"
    return "other"


def support_level(vendor: str, family: Optional[str] = None) -> str:
    key = (family or platform_family(), vendor)
    if key in _OFFICIAL:
        return OFFICIAL
    if key in _COMMUNITY:
        return COMMUNITY
    return EXPERIMENTAL


def note_accelerated_stage(stage: str, device: Optional[str]) -> Optional[str]:
    """Announce, once per stage per run, a GPU placement below official support.

    Returns ``device`` unchanged, so a placement can be announced where it is
    assigned.
    """

    from .device import accelerator, is_gpu_device

    if not is_gpu_device(device):
        return device
    gpu = accelerator()
    if not gpu.gpu_family:
        return device
    family = platform_family()
    level = support_level(gpu.vendor, family)
    if level == OFFICIAL:
        return device
    reporter = current_reporter()
    try:
        announced = _ANNOUNCED.setdefault(reporter, set())
    except TypeError:  # a reporter that cannot be weakly referenced
        announced = set()
    if stage in announced:
        return device
    label = "community-verified" if level == COMMUNITY else "not verified"
    reporter.warning(
        "unverified-accelerator",
        f"{stage} runs on {gpu.name or gpu.vendor} ({gpu.vendor} on {family}), a "
        f"combination that is {label} for this pipeline",
        impact="结果可能与已验收的平台不同，也可能失败",
        action=f"若有异常，加 --device cpu 退回 CPU；欢迎反馈：{FEEDBACK_URL}",
    )
    if reporter_delivers(reporter):
        announced.add(stage)
    return device
