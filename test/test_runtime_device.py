from __future__ import annotations

import io

import pytest
import torch

from finesub.reporting import reporting_to, terminal_reporter
from finesub.speech.runtime import device as runtime_device


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
    """No CUDA at all reads differently from a card that is merely too old."""

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    resolved, err = _resolve_with_report("cuda", context="ASR alignment")
    assert resolved == "cpu"
    assert "CUDA requested for ASR alignment but it is unavailable" in err
    assert "compute capability" not in err


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
