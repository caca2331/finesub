from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch

from finesub.reporting import reporting_to, terminal_reporter
from finesub.speech.runtime import device as runtime_device

#: The real probe, before `conftest`'s autouse fixture stubs the attribute out.
_REAL_INSTALLED_GPU_VENDORS = runtime_device.installed_gpu_vendors


@pytest.fixture(autouse=True)
def _not_an_xpu_build(monkeypatch: pytest.MonkeyPatch):
    """Every test here starts on a torch with **no** Intel GPU backend.

    That is not the default on the reference machine for this work: it has an
    Arc 130T and `torch 2.11.0+xpu`, so `torch.version.xpu` is set and
    `torch.xpu.is_available()` is True. Left alone, that sends every CUDA and
    CPU case below down the Intel branch -- `resolve_device("cuda")` would
    answer "xpu" on a machine a test is trying to describe as having no
    accelerator, and the suite would pass or fail depending on whose desk it
    runs on.

    Tests that *do* want Intel ask for the `xpu` fixture, which runs after
    this one and installs a fake backend on top.
    """

    monkeypatch.setattr(torch.version, "xpu", None, raising=False)
    monkeypatch.setattr(torch, "xpu", None, raising=False)


def _resolve_with_report(device: str, *, context: str) -> tuple[str, str]:
    """resolve_device under a bound terminal reporter; returns (device, report)."""

    stream = io.StringIO()
    with reporting_to(terminal_reporter(stream)):
        resolved = runtime_device.resolve_device(device, context=context)
    return resolved, stream.getvalue()


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch):
    """Present an arbitrary card and kernel list, GPU or not on this machine."""

    def install(*, name: str, capability: tuple[int, int], arch_list: list[str]):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch_list)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *a, **k: capability
        )
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: name)

    return install


class _FakeXpu:
    """A stand-in for ``torch.xpu`` asking only what `device.py` asks.

    Real fields come from a real ``torch 2.11.0+xpu`` probe on the reference
    machine (an Arc 130T): ``device_id=0x7D51``, ``driver_version='1.13.35227'``,
    ``total_memory≈16 GiB``, and an opaque integer ``architecture`` that the
    tag deliberately does not use. Cards are ``(name, total_bytes, device_id)``;
    ``device_id=None`` models a property set that does not carry one.
    """

    def __init__(self, *, cards, bf16: bool = True) -> None:
        self.cards = list(cards)
        self._bf16 = bf16
        self._current = 0
        self.emptied = 0
        self.synchronized = 0
        self.peak_resets = 0
        self.allocated = 0
        self.reserved = 0

    def device_count(self) -> int:
        return len(self.cards)

    def is_available(self) -> bool:
        return bool(self.cards)

    def current_device(self) -> int:
        return self._current

    def set_device(self, index: int) -> None:
        self._current = index

    def get_device_name(self, index=None) -> str:
        return self.cards[self._current if index is None else index][0]

    def get_device_properties(self, index: int):
        name, total_memory, device_id = self.cards[index]
        return SimpleNamespace(
            name=name,
            total_memory=total_memory,
            device_id=device_id,
            driver_version="1.13.35227",
            # Real torch reports this as an opaque integer; nothing may key on it.
            architecture=13195281408,
        )

    def mem_get_info(self):
        total = self.cards[self._current][1]
        return total - GIB, total

    def empty_cache(self) -> None:
        self.emptied += 1

    def synchronize(self) -> None:
        self.synchronized += 1

    def reset_peak_memory_stats(self) -> None:
        self.peak_resets += 1

    def max_memory_allocated(self) -> int:
        return self.allocated

    def max_memory_reserved(self) -> int:
        return self.reserved

    def is_bf16_supported(self, including_emulation: bool = True) -> bool:
        # torch's own shape: the default answers True for any available card.
        return including_emulation or self._bf16

    def get_arch_list(self):
        return ["mtl", "mtl-h", "bmg", "dg2", "arl-h", "lnl-m"]

    def get_device_capability(self, index=None):
        return {"architecture": 13195281408}


@pytest.fixture
def xpu(monkeypatch: pytest.MonkeyPatch):
    """Present a torch built for Intel GPUs, with the given cards attached."""

    def install(*, cards=None, bf16: bool = True, version: str = "20250200"):
        module = _FakeXpu(
            cards=(
                [("Intel(R) Arc(TM) 130T GPU (16GB)", 16 * GIB, 0x7D51)]
                if cards is None
                else cards
            ),
            bf16=bf16,
        )
        monkeypatch.setattr(torch.version, "xpu", version, raising=False)
        monkeypatch.setattr(torch, "xpu", module, raising=False)
        # An XPU build carries no CUDA and no HIP at all; pretending otherwise
        # would make the two branches fight over the same machine.
        monkeypatch.setattr(torch.version, "cuda", None, raising=False)
        monkeypatch.setattr(torch.version, "hip", None, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        return module

    return install


CU128_ARCHES = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


def test_a_supported_card_keeps_cuda(gpu) -> None:
    gpu(name="NVIDIA GeForce RTX 3060", capability=(8, 6), arch_list=CU128_ARCHES)

    assert runtime_device.cuda_unusable_reason() is None
    assert runtime_device.cuda_usable() is True
    assert runtime_device.resolve_device("cuda", context="VAD-ASR") == "cuda"


def test_a_pascal_card_falls_back_with_its_model_name(gpu) -> None:
    """The 1060's compute capability is absent from every cu128 kernel list."""

    gpu(name="NVIDIA GeForce GTX 1060 6GB", capability=(6, 1), arch_list=CU128_ARCHES)

    assert runtime_device.cuda_usable() is False
    resolved, err = _resolve_with_report("cuda", context="VAD-ASR")
    assert resolved == "cpu"
    assert "GTX 1060 6GB" in err
    assert "compute capability 6.1" in err
    # The message has to name cards a user can recognise, not architectures.
    assert runtime_device.SUPPORTED_GPU_HINT in err
    assert "falling back to CPU" in err


def test_the_oldest_supported_consumer_card_is_the_gtx_16_series(gpu) -> None:
    """sm_75 is in the list, so a GTX 1650/1660 must not be pushed to CPU."""

    gpu(name="NVIDIA GeForce GTX 1660 SUPER", capability=(7, 5), arch_list=CU128_ARCHES)

    assert runtime_device.cuda_usable() is True


def test_the_rtx_40_series_is_not_in_the_arch_list_but_still_supported(gpu) -> None:
    """The regression that matters: cu128 ships no sm_89, yet 40-series works.

    A cubin is binary-compatible with later minor revisions of the same major
    version, so requiring list membership would strand every RTX 4060/4090 on
    CPU.
    """

    gpu(name="NVIDIA GeForce RTX 4090", capability=(8, 9), arch_list=CU128_ARCHES)

    assert "sm_89" not in CU128_ARCHES
    assert runtime_device.cuda_usable() is True


def test_a_card_newer_than_the_whole_list_is_left_to_torch(gpu) -> None:
    """Above the top, PTX or in-major compatibility may carry it; let it try.

    Being wrong here costs a loud error. Being wrong the other way costs a
    silent 10x slowdown.
    """

    gpu(name="Future card", capability=(13, 0), arch_list=["sm_80", "compute_90"])

    assert runtime_device.cuda_usable() is True


def test_a_card_below_the_whole_list_falls_back(gpu) -> None:
    """Below the minimum there is no escape hatch -- cubins are not backward
    compatible, and PTX only JITs forward."""

    gpu(name="NVIDIA GeForce GTX 1080", capability=(6, 1), arch_list=["compute_90"])

    assert runtime_device.cuda_usable() is False


def test_variant_suffixes_are_parsed_like_torch_does(gpu) -> None:
    """sm_90a is a variant of 9.0, not an unparseable entry."""

    gpu(name="NVIDIA H100", capability=(9, 0), arch_list=["sm_90a"])

    assert runtime_device.cuda_usable() is True


def test_an_empty_arch_list_assumes_the_card_is_fine(gpu) -> None:
    """torch skips its own capability check in that case, so neither do we."""

    gpu(name="Some ROCm device", capability=(6, 1), arch_list=[])

    assert runtime_device.cuda_usable() is True


def test_missing_cuda_reports_that_instead(monkeypatch) -> None:
    """No CUDA at all reads differently from a card that is merely too old.

    A CUDA build of torch on a machine without a card -- stated, because the
    installed build decides which message is right, and a CPU-only build (the
    CI lanes install one) is the next test's case.
    """

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    resolved, err = _resolve_with_report("cuda", context="ASR alignment")
    assert resolved == "cpu"
    assert "CUDA requested for ASR alignment but it is unavailable" in err
    assert "compute capability" not in err


def test_a_cpu_only_torch_build_says_so_instead_of_blaming_the_card(
    monkeypatch,
) -> None:
    """`pip install` from PyPI over a vendor torch: a card that is simply unused.

    It looks identical to "no GPU here" from inside torch, and the accident is
    one command away on both vendors -- `.[asr]` over a ROCm torch replaces it
    with the CUDA-less PyPI build.
    """

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    resolved, err = _resolve_with_report("cuda", context="ASR alignment")
    assert resolved == "cpu"
    assert "no GPU support compiled in" in err
    assert "repo-install.md" in err


def _unused_card_report() -> str:
    """The warning twice over, so the report also shows it is said once."""

    stream = io.StringIO()
    with reporting_to(terminal_reporter(stream)):
        runtime_device.warn_if_an_installed_card_is_unused()
        runtime_device.warn_if_an_installed_card_is_unused()
    return stream.getvalue()


def test_an_installed_card_nothing_can_use_is_reported_once(monkeypatch) -> None:
    """The CPU tier is right here; the silence that used to come with it was not."""

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(
        runtime_device, "installed_gpu_vendors", lambda: frozenset({"NVIDIA"})
    )
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    report = _unused_card_report()

    assert report.count("nothing here can use it") == 1
    assert "NVIDIA" in report
    # No GPU support in this build, so the way out is reinstalling torch.
    assert "repo-install.md" in report


def test_a_machine_with_no_card_stays_silent(monkeypatch) -> None:
    """Nothing to report, and `resolve_device` already covers the CPU fallback."""

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(runtime_device, "installed_gpu_vendors", frozenset)

    assert _unused_card_report() == ""


def test_an_amd_card_under_a_cuda_torch_is_told_which_torch(monkeypatch) -> None:
    """The `[asr]`-over-ROCm accident: torch works, just not for this card."""

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(
        runtime_device, "installed_gpu_vendors", lambda: frozenset({"AMD"})
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    assert "ROCm" in _unused_card_report()


def test_the_vendor_probe_swallows_a_broken_display_api(monkeypatch) -> None:
    """It runs on machines nobody here can test; a probe must not end a run.

    Calls the real function, captured at import: `conftest`'s autouse fixture
    replaces the module attribute with an empty-set stub, so asserting against
    the name would only be testing the stub.
    """

    monkeypatch.setattr(runtime_device.os, "name", "nt")

    def no_user32(*args, **kwargs):
        raise OSError("user32 unavailable")

    # raising=False: off Windows ctypes has no WinDLL to replace.
    monkeypatch.setattr(runtime_device.ctypes, "WinDLL", no_user32, raising=False)

    assert _REAL_INSTALLED_GPU_VENDORS() == frozenset()


def test_an_amd_hint_names_the_range_it_cannot_cover(monkeypatch) -> None:
    """The managed-CLI case: cu128 torch, an AMD card, nothing to be done.

    AMD's Windows ROCm covers RX 7000/9000 and the Strix APUs only, so on
    anything older the hint has to say that rather than send the user off to
    install something that does not exist for their card.
    """

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(
        runtime_device, "installed_gpu_vendors", lambda: frozenset({"AMD"})
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    report = _unused_card_report()

    assert "RX 7000 / 9000" in report
    # A URL, not a repo path: a CLI user installed from PyPI has no checkout.
    assert runtime_device.REPO_INSTALL_DOC in report
    assert "docs/manual/repo-install.md）" not in report


def test_the_install_doc_url_is_the_same_everywhere() -> None:
    """Two copies, one address -- the bootstrap cannot import the main package.

    Inside `finesub` there is exactly one (`finesub.doc_links`); `cuda_libs`
    reads it too, and may not reach for `device` because that imports torch.
    """

    from finesub import doc_links
    from finesub.speech.runtime import cuda_libs
    from finesub_bootstrap import shell

    assert runtime_device.REPO_INSTALL_DOC is doc_links.REPO_INSTALL_DOC
    assert cuda_libs.REPO_INSTALL_DOC is doc_links.REPO_INSTALL_DOC
    assert doc_links.REPO_INSTALL_DOC == shell.REPO_INSTALL_DOC


def test_an_explicit_cpu_request_is_left_alone(gpu) -> None:
    gpu(name="NVIDIA GeForce RTX 3060", capability=(8, 6), arch_list=CU128_ARCHES)

    resolved, err = _resolve_with_report("cpu", context="VAD-ASR")
    assert resolved == "cpu"
    assert err == ""


def test_a_cuda_index_is_refused(gpu) -> None:
    """Honouring "cuda:1" would be a false guarantee at three separate layers.

    The capability check reads the current device, not device 1; CTranslate2
    rejects the string outright (`unsupported device cuda:1`, it wants
    device_index); and nothing in this repo plumbs an index. So say no here,
    where the message can name the alternative.
    """

    gpu(name="NVIDIA GeForce RTX 3060", capability=(8, 6), arch_list=CU128_ARCHES)

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        runtime_device.resolve_device("cuda:1", context="VAD-ASR")


# --- AMD (ROCm) -------------------------------------------------------------


@pytest.fixture
def amd(monkeypatch: pytest.MonkeyPatch):
    """Present a ROCm build of torch with one or more AMD cards."""

    def install(
        *,
        cards: list[tuple[str, str, int]],
        arch_list: list[str],
        bf16: bool = True,
    ):
        # (name, gcnArchName, total bytes) per device index.
        current = {"index": 0}
        monkeypatch.setattr(torch.version, "hip", "6.4.43482", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(cards))
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch_list)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])
        monkeypatch.setattr(
            torch.cuda, "set_device", lambda index: current.__setitem__("index", index)
        )
        monkeypatch.setattr(
            torch.cuda,
            "get_device_name",
            lambda index=None: cards[current["index"] if index is None else index][0],
        )
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda index: type(
                "P", (), {"gcnArchName": cards[index][1], "total_memory": cards[index][2]}
            )(),
        )
        monkeypatch.setattr(runtime_device, "_WARNED_LARGER_CARD", False)
        for name in runtime_device._VISIBLE_DEVICE_VARIABLES + ("HSA_OVERRIDE_GFX_VERSION",):
            monkeypatch.delenv(name, raising=False)
        return current

    return install


GIB = 1024**3
ROCM_ARCHES = ["gfx900", "gfx906", "gfx1030", "gfx1100", "gfx1101", "gfx1200", "gfx1201"]


def test_an_amd_card_in_the_build_list_keeps_cuda(amd) -> None:
    amd(cards=[("AMD Radeon RX 7900 XTX", "gfx1100", 24 * GIB)], arch_list=ROCM_ARCHES)

    assert runtime_device.is_rocm() is True
    assert runtime_device.cuda_unusable_reason() is None
    assert runtime_device.accelerator().vendor == "amd"
    assert runtime_device.gpu_arch_tag() == "gfx1100"
    assert runtime_device.gpu_runtime_tag() == "hip6.4.43482"


def test_an_amd_card_outside_the_list_is_refused_by_membership_not_a_floor(amd) -> None:
    # gfx1102 is *newer* than gfx1101 and still has no kernels: a floor would pass it.
    amd(cards=[("AMD Radeon RX 7600", "gfx1102", 8 * GIB)], arch_list=ROCM_ARCHES)

    reason = runtime_device.cuda_unusable_reason()
    assert reason is not None
    assert "RX 7600" in reason and "gfx1102" in reason


def test_generic_rocm_targets_cover_their_family(amd) -> None:
    amd(cards=[("AMD Radeon RX 7600", "gfx1102:sriov", 8 * GIB)], arch_list=["gfx110X"])
    assert runtime_device.cuda_unusable_reason() is None

    amd(cards=[("AMD Radeon RX 7600", "gfx1102", 8 * GIB)], arch_list=["gfx11-generic"])
    assert runtime_device.cuda_unusable_reason() is None


def test_hsa_override_is_the_target_the_kernel_check_looks_for(amd, monkeypatch) -> None:
    # An RX 6700 XT (gfx1031) run the way its owners do: as gfx1030.
    amd(cards=[("AMD Radeon RX 6700 XT", "gfx1031", 12 * GIB)], arch_list=ROCM_ARCHES)
    assert runtime_device.cuda_unusable_reason() is not None

    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    assert runtime_device.cuda_unusable_reason() is None
    assert runtime_device._ct2_architecture_unusable_reason() is None


def test_ct2_on_amd_checks_the_hip_wheel_targets(amd) -> None:
    amd(cards=[("AMD Radeon RX 9070 XT", "gfx1201", 16 * GIB)], arch_list=ROCM_ARCHES)
    assert runtime_device._ct2_architecture_unusable_reason() is None

    amd(cards=[("AMD Radeon Pro W5700", "gfx1010", 8 * GIB)], arch_list=ROCM_ARCHES)
    reason = runtime_device._ct2_architecture_unusable_reason()
    assert reason is not None and "gfx1010" in reason


def test_a_larger_amd_card_is_recommended_not_switched_to(amd, reported) -> None:
    # torch's current device is per thread, so switching it process-wide would
    # leave worker threads on device 0; the user selects through a variable.
    current = amd(
        cards=[
            ("AMD Radeon Graphics", "gfx1036", 2 * GIB),
            ("AMD Radeon RX 7800 XT", "gfx1101", 16 * GIB),
        ],
        arch_list=ROCM_ARCHES + ["gfx1036"],
    )

    assert runtime_device.larger_amd_card_index() == 1
    assert runtime_device.cuda_unusable_reason() is None
    runtime_device.cuda_unusable_reason()
    assert current["index"] == 0
    assert reported.codes() == ["amd-device-choice"]
    assert "HIP_VISIBLE_DEVICES=1" in reported.warnings[0].action


def test_a_user_choice_of_card_is_left_alone(amd, monkeypatch) -> None:
    amd(
        cards=[
            ("AMD Radeon Graphics", "gfx1036", 2 * GIB),
            ("AMD Radeon RX 7800 XT", "gfx1101", 16 * GIB),
        ],
        arch_list=ROCM_ARCHES,
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")

    assert runtime_device.larger_amd_card_index() is None


def test_nvidia_answers_are_what_they_were(gpu, monkeypatch) -> None:
    gpu(name="NVIDIA GeForce RTX 5070 Ti", capability=(12, 0), arch_list=CU128_ARCHES)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert runtime_device.larger_amd_card_index() is None
    assert runtime_device.gpu_runtime_tag() == "cuda12.8"
    assert runtime_device.gpu_runtime_version() == "12.8"
    assert runtime_device.gpu_arch_tag() == "sm120"
    assert runtime_device.bf16_supported() is True
    assert runtime_device.accelerator().vendor == "nvidia"


def test_ct2_build_targets_match_the_device_layer() -> None:
    """The wheel's build flags and the kernel check must name the same cards."""

    import re
    from pathlib import Path

    common = (
        Path(__file__).resolve().parents[1]
        / "tools/wt_refine_port/ct2-build/common.sh"
    ).read_text(encoding="utf-8")
    hip = re.search(r'^CT2_HIP_GFX_TARGETS="([^"]+)"', common, re.MULTILINE).group(1)
    cuda = re.search(r'^CT2_CUDA_ARCH_LIST="([^"]+)"', common, re.MULTILINE).group(1)

    assert tuple(hip.split(";")) == runtime_device.CT2_HIP_GFX_TARGETS
    floor = min(tuple(int(part) for part in entry.split("+")[0].split(".")) for entry in cuda.split(";"))
    assert floor == runtime_device.CT2_MIN_COMPUTE_CAPABILITY


def test_bf16_is_decided_by_the_gfx_target_not_by_torch(amd, monkeypatch) -> None:
    """`torch.cuda.is_bf16_supported()` answers True for every ROCm card.

    It returns before looking at one (`if torch.version.hip: return True` in
    the pinned 2.11), so asking it made the float32 branch dead code -- and the
    test that pinned the branch faked the API into saying False, which no card
    ever will. The card's own target is the thing that knows.
    """

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)

    amd(cards=[("AMD Radeon RX 6800", "gfx1030", 16 * GIB)], arch_list=ROCM_ARCHES)
    assert runtime_device.bf16_supported() is False  # RDNA2 emulates it

    amd(cards=[("AMD Radeon RX 7900 XTX", "gfx1100", 24 * GIB)], arch_list=ROCM_ARCHES)
    assert runtime_device.bf16_supported() is True  # RDNA3 on


def test_an_overridden_gfx_target_decides_bf16_too(amd, monkeypatch) -> None:
    """`HSA_OVERRIDE_GFX_VERSION` is what the runtime loads kernels for."""

    amd(cards=[("AMD Radeon RX 6700 XT", "gfx1031", 12 * GIB)], arch_list=ROCM_ARCHES)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

    assert runtime_device.bf16_supported() is True


def test_the_cdna_line_reads_as_native(amd) -> None:
    """Three-character targets are CDNA, where bf16 has been native since MI100."""

    amd(cards=[("AMD Instinct MI210", "gfx90a", 64 * GIB)], arch_list=["gfx90a"])
    assert runtime_device.bf16_supported() is True

    amd(cards=[("AMD Radeon VII", "gfx906", 16 * GIB)], arch_list=["gfx906"])
    assert runtime_device.bf16_supported() is False


@pytest.fixture
def ct2_flash(monkeypatch: pytest.MonkeyPatch, gpu):
    """An Ampere card on CUDA torch and a CTranslate2 that may or may not have flash."""

    import sys
    import types

    def install(*, compiled: bool | None, capability: tuple[int, int] = (8, 6), mode: str | None = "auto"):
        gpu(name="NVIDIA GeForce RTX 3060", capability=capability, arch_list=CU128_ARCHES)
        monkeypatch.setattr(torch.version, "hip", None, raising=False)
        fake = types.ModuleType("ctranslate2")
        if compiled is not None:  # None: a wheel from before patch 0012
            fake.flash_attention_available = lambda: compiled
        monkeypatch.setitem(sys.modules, "ctranslate2", fake)
        if mode is None:
            monkeypatch.delenv(runtime_device.CT2_FLASH_ATTENTION_ENV, raising=False)
        else:
            monkeypatch.setenv(runtime_device.CT2_FLASH_ATTENTION_ENV, mode)

    return install


def test_ct2_flash_attention_is_on_by_default_and_off_on_request(ct2_flash, reported) -> None:
    ct2_flash(compiled=True, mode=None)
    assert runtime_device.ct2_flash_attention("cuda", "float16") is True
    ct2_flash(compiled=True, mode="")
    assert runtime_device.ct2_flash_attention("cuda", "float16") is True
    ct2_flash(compiled=True, mode="off")
    assert runtime_device.ct2_flash_attention("cuda", "float16") is False
    assert reported.codes() == []


def test_the_default_falls_back_silently(ct2_flash, reported) -> None:
    # Every pre-Ampere card, CPU run and older wheel takes this path on every
    # run; a warning there would be noise nobody asked for.
    ct2_flash(compiled=True, capability=(7, 5), mode=None)
    assert runtime_device.ct2_flash_attention("cuda", "float16") is False
    ct2_flash(compiled=None, mode=None)
    assert runtime_device.ct2_flash_attention("cuda", "float16") is False
    assert runtime_device.ct2_flash_attention("cpu", "float32") is False
    assert reported.codes() == []


def test_ct2_flash_attention_runs_where_every_condition_holds(ct2_flash, reported) -> None:
    for mode in ("auto", "on", " ON "):
        ct2_flash(compiled=True, mode=mode)
        assert runtime_device.ct2_flash_attention("cuda", "float16") is True
    assert reported.codes() == []


@pytest.mark.parametrize(
    ("setup", "device", "compute_type", "says"),
    [
        ({"compiled": True}, "cpu", "float32", "CPU"),
        ({"compiled": True}, "cuda", "float32", "float16 compute type"),
        ({"compiled": True}, "cuda", "bfloat16", "float16 compute type"),
        ({"compiled": False}, "cuda", "float16", "WITH_FLASH_ATTN"),
        ({"compiled": None}, "cuda", "float16", "WITH_FLASH_ATTN"),
        ({"compiled": True, "capability": (7, 5)}, "cuda", "float16", "compute capability 7.5"),
    ],
)
def test_ct2_flash_attention_falls_back_and_on_says_why(
    ct2_flash, reported, setup, device, compute_type, says
) -> None:
    # A build without flash would accept the option and fail at the first
    # encode, which the alignment loop reads as "drop this group"; the others
    # make the model load raise. Each is caught before the load instead.
    ct2_flash(**setup, mode="auto")
    assert runtime_device.ct2_flash_attention(device, compute_type) is False
    assert reported.codes() == []

    ct2_flash(**setup, mode="on")
    assert runtime_device.ct2_flash_attention(device, compute_type) is False
    assert reported.codes() == ["ct2-flash-attention-unavailable"]
    assert says in reported.warnings[0].message


def test_ct2_flash_attention_is_never_tried_on_rocm(ct2_flash, amd, reported) -> None:
    ct2_flash(compiled=True, mode="on")
    amd(cards=[("AMD Radeon RX 7900 XTX", "gfx1100", 24 * GIB)], arch_list=ROCM_ARCHES)

    assert runtime_device.ct2_flash_attention("cuda", "float16") is False
    assert "ROCm" in reported.warnings[0].message


def test_an_unknown_ct2_flash_attention_mode_is_off_and_said(ct2_flash, reported) -> None:
    ct2_flash(compiled=True, mode="yes")

    assert runtime_device.ct2_flash_attention("cuda", "float16") is False
    assert reported.codes() == ["ct2-flash-attention-unavailable"]


# --- Intel GPU (XPU) --------------------------------------------------------


@pytest.mark.parametrize(
    ("vendor", "family"),
    [
        ("nvidia", True),
        ("amd", True),
        ("intel", True),
        # Reported so a run can say what it saw, but no stage uses MPS.
        ("apple", False),
        ("", False),
    ],
)
def test_gpu_family_means_this_pipeline_can_place_work(vendor, family) -> None:
    """The renamed predicate still answers about NVIDIA/AMD and now Intel."""

    accelerator = runtime_device.Accelerator(vendor)
    assert accelerator.gpu_family is family
    # The rename has no alias: a caller still reaching for the old name is
    # asking the wrong question and should fail loudly, not silently get False.
    assert not hasattr(accelerator, "cuda_family")


def test_a_torch_without_the_xpu_build_answers_safely(monkeypatch) -> None:
    """A stock PyPI wheel: every Intel entry point is inert, none is missing."""

    # The host's own CUDA card would otherwise answer `gpu_device()`.
    monkeypatch.setattr(runtime_device, "cuda_usable", lambda: False)
    assert runtime_device.is_xpu() is False
    assert runtime_device.xpu_usable() is False
    reason = runtime_device.xpu_unusable_reason()
    assert reason is not None
    assert "no Intel GPU support compiled in" in reason
    assert "XPU index" in reason
    assert runtime_device.gpu_device() == ""
    assert runtime_device.device_type(None) == "cpu"
    assert runtime_device.is_gpu_device(None) is False
    assert runtime_device.is_gpu_device("xpu") is True
    # Placement helpers are safe calls on a machine with no card at all.
    runtime_device.empty_device_cache()
    runtime_device.synchronize()
    runtime_device.reset_peak_gpu_memory_stats()
    assert runtime_device.peak_gpu_memory_bytes() == 0


def test_an_intel_card_is_reported_as_its_own_vendor(xpu) -> None:
    xpu()

    accelerator = runtime_device.accelerator()

    assert accelerator.vendor == "intel"
    assert accelerator.name == "Intel(R) Arc(TM) 130T GPU (16GB)"
    # The PCI device id, not the opaque `architecture` integer.
    assert accelerator.arch == "xpu7d51"
    assert accelerator.gpu_family is True


def test_a_cuda_request_lands_on_the_intel_card_without_a_word(xpu) -> None:
    """`cuda` is the historical token for "the accelerator", not for NVIDIA."""

    xpu()

    resolved, report = _resolve_with_report("cuda", context="vocal separation")

    assert resolved == "xpu"
    # Nothing went wrong and nothing was downgraded, so there is nothing to say.
    assert report == ""


def test_an_explicit_xpu_request_is_honoured(xpu) -> None:
    xpu()

    resolved, report = _resolve_with_report("xpu", context="VAD-ASR")

    assert resolved == "xpu"
    assert report == ""


def test_an_xpu_request_on_a_cuda_machine_falls_back_and_says_why(gpu) -> None:
    gpu(name="NVIDIA GeForce RTX 3060", capability=(8, 6), arch_list=CU128_ARCHES)

    resolved, err = _resolve_with_report("xpu", context="VAD-ASR")

    assert resolved == "cpu"
    assert "XPU requested for VAD-ASR" in err
    assert "no Intel GPU support compiled in" in err


@pytest.mark.parametrize("requested", ["xpu:0", "cuda:1"])
def test_both_resolvers_refuse_an_indexed_device_string(requested) -> None:
    """One message for both stages: the family list and where the index goes."""

    resolvers = (
        lambda device: runtime_device.resolve_device(device, context="VAD-ASR"),
        runtime_device.resolve_asr_device,
    )
    for resolver in resolvers:
        with pytest.raises(ValueError) as caught:
            resolver(requested)
        message = str(caught.value)
        assert "'cpu', 'cuda' or 'xpu'" in message
        assert "CUDA_VISIBLE_DEVICES" in message
        assert "ZE_AFFINITY_MASK" in message


def test_asr_on_an_intel_machine_stays_on_the_cpu_and_says_why(xpu, reported) -> None:
    """The card works; CTranslate2 just has no backend for it."""

    xpu()

    assert runtime_device.resolve_asr_device("xpu") == "cpu"

    assert reported.codes() == ["cpu-fallback"]
    warning = reported.warnings[0]
    assert "CTranslate2 has no Intel GPU backend" in warning.message
    # The tail check follows the ASR device (`tail_verify_device`), so it is
    # named with ASR -- and the stages that *do* use the card are named, or a
    # user goes hunting for a bug that does not exist.
    assert "ASR and the qwen-verify check after it run on the CPU" in warning.message
    assert (
        "vocal separation, the Silero VAD assist and the Qwen referee that "
        "re-decodes languages use the Intel GPU"
    ) in warning.message
    assert "其余阶段照常" in warning.impact


def test_the_intel_asr_warning_is_said_once_per_reporter(xpu) -> None:
    """The recall repair asks `resolve_asr_device` again; a batch task is a new reporter."""

    from test.conftest import _Recorder

    xpu()
    first, second = _Recorder(), _Recorder()
    with reporting_to(first):
        runtime_device.resolve_asr_device("cuda")
        runtime_device.resolve_asr_device("cuda")
    with reporting_to(second):
        runtime_device.resolve_asr_device("cuda")

    assert first.codes() == ["cpu-fallback"]
    assert second.codes() == ["cpu-fallback"]


def test_the_silero_assist_follows_an_intel_card(xpu, reported) -> None:
    """A torch stage: the default request lands on the only accelerator there is."""

    from finesub.speech.preprocessing.silero_ghost import resolve_silero_device

    xpu()

    assert resolve_silero_device("cuda") == "xpu"
    assert "cpu-fallback" not in reported.codes()


def test_the_forced_aligner_stays_off_an_intel_card(xpu, reported, monkeypatch) -> None:
    """Only ever run on CUDA (bf16) and the CPU: an Intel card is not a third option."""

    import sys

    from finesub.speech.joint import aligner

    xpu()
    placed: dict = {}

    class _Model:
        def to(self, device):
            placed["device"] = device
            return self

        def eval(self):
            return self

    def load_model(*args, dtype, **kwargs):
        placed["dtype"] = dtype
        return _Model()

    # A stand-in module: the real one would import its torch side against the fake `torch.xpu`.
    fake_transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=lambda *a, **k: object()),
        AutoModelForTokenClassification=SimpleNamespace(from_pretrained=load_model),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(aligner, "_LOADED", None)

    aligner._model()

    assert placed == {"device": "cpu", "dtype": torch.float32}
    assert reported.codes() == ["cpu-fallback"]


def test_a_cuda_asr_request_on_an_intel_machine_gets_the_intel_warning(
    xpu, reported
) -> None:
    """`--device cuda` on Intel means "the accelerator", so it reaches this too."""

    xpu()

    assert runtime_device.resolve_asr_device("cuda") == "cpu"

    assert reported.codes() == ["cpu-fallback"]
    assert "CTranslate2 has no Intel GPU backend" in reported.warnings[0].message


def test_asr_on_a_requested_cpu_says_nothing_at_all(xpu, reported) -> None:
    xpu()

    assert runtime_device.resolve_asr_device("cpu") == "cpu"

    assert reported.codes() == []


def test_vram_figures_come_from_the_xpu_runtime(xpu) -> None:
    xpu(cards=[("Intel(R) Arc(TM) 130T GPU (16GB)", 16 * GIB, 0x7D51)])

    assert runtime_device.total_vram_gib() == pytest.approx(16.0)
    assert runtime_device.free_vram_gib() == pytest.approx(15.0)


def test_xpu_identity_strings_are_spelled_for_intel(xpu) -> None:
    xpu()

    assert runtime_device.gpu_runtime_tag() == "xpu20250200"
    assert runtime_device.gpu_runtime_version() == "xpu-20250200"
    assert runtime_device.gpu_name() == "Intel(R) Arc(TM) 130T GPU (16GB)"
    # The label keys a compiled cache, so it must be stable across calls.
    assert runtime_device.gpu_arch_tag() == "xpu7d51"
    assert runtime_device.gpu_arch_tag() == "xpu7d51"


def test_the_arch_tag_falls_back_to_the_card_name_without_a_device_id(xpu) -> None:
    xpu(cards=[("Intel(R) Arc(TM) B580", 12 * GIB, None)])

    tag = runtime_device.gpu_arch_tag()

    assert tag.startswith("xpu_")
    assert "arc" in tag
    assert "b580" in tag


def test_bf16_on_intel_asks_the_runtime(xpu) -> None:
    """Native bf16 only: an emulated one answers True by torch's default."""

    xpu(bf16=True)
    assert runtime_device.bf16_supported() is True

    module = xpu(bf16=False)
    assert module.is_bf16_supported() is True
    assert runtime_device.bf16_supported() is False


def test_bf16_on_intel_without_an_answer_assumes_yes(xpu) -> None:
    """Arc Xe2 computes bf16 natively; guessing False would cost every run time."""

    module = xpu()
    module.is_bf16_supported = None

    assert runtime_device.bf16_supported() is True


def test_device_family_helpers_split_cuda_xpu_and_cpu(xpu) -> None:
    module = xpu()

    assert runtime_device.gpu_device() == "xpu"
    # None is "placed nothing on a card", even where a card is usable.
    assert runtime_device.device_type(None) == "cpu"
    assert runtime_device.is_gpu_device(None) is False
    assert runtime_device.device_type("xpu") == "xpu"
    assert runtime_device.device_type("xpu:0") == "xpu"
    assert runtime_device.device_type("cuda") == "cuda"
    assert runtime_device.device_type("cpu") == "cpu"
    assert runtime_device.is_gpu_device("cuda:0") is True
    assert runtime_device.is_gpu_device("xpu") is True
    assert runtime_device.is_gpu_device("cpu") is False

    runtime_device.empty_device_cache("xpu")
    runtime_device.synchronize("xpu")
    runtime_device.empty_device_cache("cpu")  # no-op, and no crash

    assert module.emptied == 1
    assert module.synchronized == 1


def test_peak_memory_comes_from_the_xpu_allocator(xpu) -> None:
    module = xpu()
    module.allocated = 3 * GIB
    module.reserved = 5 * GIB

    runtime_device.reset_peak_gpu_memory_stats("xpu")
    assert module.peak_resets == 1
    assert runtime_device.peak_gpu_memory_bytes("xpu") == 5 * GIB
    # A stage that ran on the CPU reads nothing off the card.
    runtime_device.reset_peak_gpu_memory_stats(None)
    assert module.peak_resets == 1
    assert runtime_device.peak_gpu_memory_bytes(None) == 0


def test_cuda_keeps_the_token_when_a_cuda_card_is_usable(xpu, gpu, monkeypatch) -> None:
    xpu()
    gpu(name="NVIDIA GeForce RTX 3060", capability=(8, 6), arch_list=CU128_ARCHES)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert runtime_device.gpu_device() == "cuda"
    assert runtime_device.resolve_device("cuda", context="VAD-ASR") == "cuda"
    assert runtime_device.accelerator().vendor == "nvidia"


def test_presence_is_weaker_than_usability(gpu, monkeypatch) -> None:
    """A card this build has no kernels for is present, and not usable.

    The distinction `gpu_device_present` exists for, and the one the tier gate
    depends on: a GTX 1060 with a cu128-only torch is enumerated by the driver
    (so the machine wants a GPU tier and CTranslate2 may still run it) while
    torch refuses to place work on it.
    """

    gpu(name="NVIDIA GeForce GTX 1060 6GB", capability=(6, 1), arch_list=CU128_ARCHES)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert runtime_device.cuda_device_present() is True
    assert runtime_device.gpu_device_present() is True
    assert runtime_device.cuda_usable() is False
    assert runtime_device.gpu_device() == ""


def test_the_intel_presence_answer_follows_the_driver(xpu) -> None:
    """No card in the driver -- present is False; one card -- present is True."""

    xpu(cards=[])
    assert runtime_device.xpu_device_present() is False
    assert runtime_device.gpu_device_present() is False

    xpu()
    assert runtime_device.xpu_device_present() is True
    assert runtime_device.gpu_device_present() is True
    assert runtime_device.gpu_device() == "xpu"


def test_the_windows_vendor_probe_recognizes_an_intel_card(monkeypatch) -> None:
    monkeypatch.setattr(runtime_device.os, "name", "nt")
    monkeypatch.setattr(runtime_device.sys, "platform", "win32")

    def enum_display_devices(_display, index, device, _flags):
        if index > 0:
            return 0
        device._obj.DeviceID = "PCI\\VEN_8086&DEV_7D51&SUBSYS_00000000&REV_00"
        return 1

    monkeypatch.setattr(
        runtime_device.ctypes,
        "WinDLL",
        lambda *a, **k: SimpleNamespace(EnumDisplayDevicesW=enum_display_devices),
        raising=False,
    )

    assert _REAL_INSTALLED_GPU_VENDORS() == frozenset({"Intel"})


def test_the_linux_vendor_probe_recognizes_an_intel_card(monkeypatch) -> None:
    class _Leaf:
        def __init__(self, body: str) -> None:
            self._body = body

        def read_text(self) -> str:
            return self._body

    class _Device:
        def __init__(self, class_name: str, vendor: str) -> None:
            self._values = {"class": class_name, "vendor": vendor}

        def __truediv__(self, name: str) -> _Leaf:
            return _Leaf(self._values[name])

    class _Bus:
        def glob(self, _pattern: str):
            # A 3D controller (class 0x030000) that belongs to Intel (0x8086).
            return [_Device("0x030000", "0x8086")]

    monkeypatch.setattr(runtime_device.os, "name", "posix")
    monkeypatch.setattr(runtime_device.sys, "platform", "linux")
    monkeypatch.setattr(runtime_device, "Path", lambda _path: _Bus())

    assert _REAL_INSTALLED_GPU_VENDORS() == frozenset({"Intel"})


def test_a_mixed_laptop_is_not_told_to_swap_torch_for_an_igpu(monkeypatch) -> None:
    """Intel + NVIDIA, CUDA build, NVIDIA card hidden: the driver, not the wheel.

    Almost every laptop with a discrete NVIDIA card also enumerates an Intel
    iGPU, so "Intel is among the installed vendors" is the normal state of a
    working CUDA machine -- not a reason to suggest the XPU wheel. Following
    that hint replaces the CUDA torch serving the NVIDIA card, and CTranslate2's
    CUDA ASR goes with it.
    """

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(
        runtime_device,
        "installed_gpu_vendors",
        lambda: frozenset({"NVIDIA", "Intel"}),
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    report = _unused_card_report()

    assert "(NVIDIA)" in report
    assert "CUDA_VISIBLE_DEVICES" in report


def test_a_masked_intel_card_is_a_driver_problem_not_a_reinstall(
    xpu, monkeypatch
) -> None:
    """XPU build installed, ZE_AFFINITY_MASK hides the card: same next step.

    The build is already the right one, so the only useful hint names the
    driver and the mask variables -- not another install of the same wheel.
    """

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(
        runtime_device, "installed_gpu_vendors", lambda: frozenset({"Intel"})
    )
    xpu(cards=[], version="20250302")

    report = _unused_card_report()

    assert "Intel" in report
    assert "XPU 版的 PyTorch" not in report
    assert "ZE_AFFINITY_MASK" in report
    assert "驱动" in report
    # The other reason an XPU build sees no card: one outside PyTorch's range.
    assert runtime_device._INTEL_XPU_RANGE in report


def test_a_mixed_laptop_with_no_gpu_build_gets_the_generic_wheel_hint(
    monkeypatch,
) -> None:
    """Intel + NVIDIA, stock PyPI torch: recommending XPU would lose NVIDIA too."""

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(
        runtime_device,
        "installed_gpu_vendors",
        lambda: frozenset({"NVIDIA", "Intel"}),
    )
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    report = _unused_card_report()

    assert "Intel" not in report
    assert "NVIDIA" in report
    assert "不带 GPU 支持" in report
    assert runtime_device.REPO_INSTALL_DOC in report


@pytest.mark.parametrize("cuda_build", ["12.8", None])
def test_an_intel_card_is_not_mentioned_without_an_xpu_build(
    monkeypatch, cuda_build
) -> None:
    """Almost every machine has an Intel iGPU, and most no XPU build can use.

    So outside an XPU build an Intel-only machine stays as silent as it was
    before Intel support, rather than sending every Iris Xe owner to install a
    torch that will not see their card.
    """

    monkeypatch.setattr(runtime_device, "_WARNED_UNUSED_CARD", False)
    monkeypatch.setattr(
        runtime_device, "installed_gpu_vendors", lambda: frozenset({"Intel"})
    )
    monkeypatch.setattr(torch.version, "cuda", cuda_build, raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)

    assert _unused_card_report() == ""


def test_is_xpu_answers_about_the_build_not_the_machine(xpu, monkeypatch) -> None:
    """Three builds: no Intel backend, Intel only, and both stacks.

    The last one is the one to keep straight: XPU is *compiled in*, but a build
    that also carries CUDA does not run on Intel, so `_xpu_is_the_backend` is
    what the identity strings key on -- not `is_xpu`.
    """

    assert runtime_device.is_xpu() is False
    assert runtime_device._xpu_is_the_backend() is False

    xpu()
    assert runtime_device.is_xpu() is True
    assert runtime_device._xpu_is_the_backend() is True

    monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
    assert runtime_device.is_xpu() is True
    assert runtime_device._xpu_is_the_backend() is False


def test_xpu_unusable_reason_names_the_step_that_is_missing(xpu) -> None:
    """Install torch vs. check the driver: two failures, two different next steps."""

    no_build = runtime_device.xpu_unusable_reason()
    assert "no Intel GPU support compiled in" in no_build
    assert "docs/manual/repo-install.md" in no_build

    xpu(cards=[])
    no_card = runtime_device.xpu_unusable_reason()
    assert "gives torch.xpu no device" in no_card
    assert "docs/manual/repo-install.md" in no_card

    xpu()
    assert runtime_device.xpu_unusable_reason() is None


def test_a_masked_intel_card_gets_the_driver_reason_not_the_reinstall_one(
    xpu,
) -> None:
    """The right wheel is installed; the missing piece is the visible card.

    The CUDA question on an XPU machine still has to answer with the actionable
    cause: pointing at a reinstall here would send the user to replace a torch
    that was already correct.
    """

    xpu(cards=[])
    reason = runtime_device.cuda_unusable_reason()

    assert reason == runtime_device.xpu_unusable_reason()
    assert "gives torch.xpu no device" in reason
    assert "no Intel GPU support compiled in" not in reason


def test_a_working_intel_card_under_an_xpu_build_is_named_as_such(xpu) -> None:
    """The original Intel answer survives for the machine it describes."""

    xpu()
    reason = runtime_device.cuda_unusable_reason()

    assert "Intel GPU (XPU) build" in reason
    assert "no CUDA support" in reason


def test_accelerator_answers_intel_even_when_the_name_query_fails(
    xpu, monkeypatch
) -> None:
    """The CUDA branch is wrapped for this; the Intel branch must not be a hole.

    A driver that counts a card but fails the per-device query would otherwise
    raise into every stage that asks where work goes, to withhold a name.
    """

    xpu()

    def boom(*args, **kwargs):
        raise RuntimeError("driver went away")

    monkeypatch.setattr(torch.xpu, "get_device_name", boom)

    accelerator = runtime_device.accelerator()

    assert accelerator.vendor == "intel"
    assert accelerator.gpu_family is True
    assert accelerator.name == ""


def test_the_arch_tag_does_not_invent_a_name_token(xpu) -> None:
    """A name with nothing tokenizable in it is "no name", not an empty token."""

    xpu(cards=[("???", 8 * GIB, None)])

    assert runtime_device.gpu_arch_tag() == "xpuunknown"


@pytest.mark.parametrize("accepted", ["cpu", "cuda", "xpu", " CUDA ", "Xpu", None])
def test_validate_device_token_accepts_what_the_stages_can_honour(accepted) -> None:
    runtime_device.validate_device_token(accepted)


@pytest.mark.parametrize("refused", ["cuda:0", "xpu:1", "cuda:0:cuda", "gpu", ""])
def test_validate_device_token_refuses_a_spelling_no_stage_can_honour(refused) -> None:
    """Indexed and invented device strings fail in the first second, not the last.

    An index is not a device name here: CTranslate2 takes its index separately
    and every other stage reads the *current* device, so accepting `cuda:1`
    would silently run on device 0 (see `resolve_device`). The pipeline entry
    calls this so the refusal does not wait for the stage that would have raised
    it.
    """

    with pytest.raises(ValueError, match="--device accepts 'cpu', 'cuda' or 'xpu'"):
        runtime_device.validate_device_token(refused)
