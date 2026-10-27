from __future__ import annotations

import importlib.util
import io
import os
import sys

import pytest
import torch

from finesub.reporting import reporting_to, terminal_reporter
from finesub.speech.runtime import device as runtime_device

#: The real probe, before `conftest`'s autouse fixture stubs the attribute out.
_REAL_INSTALLED_GPU_VENDORS = runtime_device.installed_gpu_vendors


@pytest.mark.parametrize(
    ("platform", "hip", "cuda", "existing", "expected"),
    [
        ("win32", "7.15", None, None, "TRUE"),
        ("win32", "7.15", None, "FALSE", "FALSE"),
        ("win32", "7.15", None, "TRUE", "TRUE"),
        ("linux", "7.15", None, None, None),
        ("win32", None, "12.8", None, None),
        ("win32", None, None, None, None),
    ],
)
def test_openmp_workaround_is_windows_rocm_only_and_respects_the_environment(
    monkeypatch, platform, hip, cuda, existing, expected
) -> None:
    """Import-time setup must precede CT2 without replacing the user's choice."""

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.version, "cuda", cuda, raising=False)
    if existing is None:
        # Record an undo even if the key was absent before module import.
        monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "")
        monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    else:
        monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", existing)

    # Execute the complete module under an alias, so its dataclasses initialise
    # normally while other tests keep their original module and probe fixtures.
    name = "finesub.speech.runtime._device_openmp_test"
    spec = importlib.util.spec_from_file_location(name, runtime_device.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)

    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == expected
    assert module.is_rocm() is bool(hip)
    assert module.Accelerator("amd").cuda_family is True


def _resolve_with_report(device: str, *, context: str) -> tuple[str, str]:
    """resolve_device under a bound terminal reporter; returns (device, report)."""

    stream = io.StringIO()
    with reporting_to(terminal_reporter(stream)):
        resolved = runtime_device.resolve_device(device, context=context)
    return resolved, stream.getvalue()


@pytest.fixture
def gpu(monkeypatch: pytest.MonkeyPatch):
    """Present a CUDA build with a card and kernel list, independent of the host."""

    def install(*, name: str, capability: tuple[int, int], arch_list: list[str]):
        monkeypatch.setattr(torch.version, "hip", None, raising=False)
        monkeypatch.setattr(torch.version, "cuda", "12.8", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch_list)
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *a, **k: capability
        )
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: name)

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
        monkeypatch.setattr(torch.version, "cuda", None, raising=False)
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
