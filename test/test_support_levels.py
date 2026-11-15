"""Support levels are labels; their only effect is the unverified-accelerator notice."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from finesub.speech.runtime import device as device_module
from finesub.speech.runtime import support_levels


@pytest.fixture
def card(monkeypatch):
    def install(vendor: str, family: str, name: str = "Some GPU"):
        monkeypatch.setattr(
            device_module, "accelerator", lambda: device_module.Accelerator(vendor, name)
        )
        monkeypatch.setattr(support_levels, "platform_family", lambda: family)
        monkeypatch.setattr(support_levels, "_ANNOUNCED", support_levels.weakref.WeakKeyDictionary())

    return install


def test_only_windows_and_wsl_nvidia_are_official() -> None:
    assert support_levels.support_level("nvidia", "windows") == support_levels.OFFICIAL
    assert support_levels.support_level("nvidia", "wsl") == support_levels.OFFICIAL
    assert support_levels.support_level("nvidia", "linux") == support_levels.EXPERIMENTAL
    assert support_levels.support_level("amd", "linux") == support_levels.EXPERIMENTAL
    assert support_levels.support_level("amd", "windows") == support_levels.EXPERIMENTAL


def test_wsl_is_told_apart_from_native_linux(monkeypatch) -> None:
    monkeypatch.setattr(support_levels.sys, "platform", "linux")
    monkeypatch.setattr(
        support_levels.platform,
        "uname",
        lambda: SimpleNamespace(release="5.15.167.4-microsoft-standard-WSL2"),
    )
    assert support_levels.platform_family() == "wsl"
    monkeypatch.setattr(
        support_levels.platform, "uname", lambda: SimpleNamespace(release="6.8.0-45-generic")
    )
    assert support_levels.platform_family() == "linux"


def test_an_unverified_gpu_placement_is_announced_once_per_stage(card, reported) -> None:
    card("amd", "linux", "AMD Radeon RX 7900 XTX")

    support_levels.note_accelerated_stage("ASR", "cuda")
    support_levels.note_accelerated_stage("ASR", "cuda")
    support_levels.note_accelerated_stage("vocal separation", "cuda")

    assert reported.codes() == ["unverified-accelerator", "unverified-accelerator"]
    first = reported.warnings[0]
    assert "RX 7900 XTX" in first.text and "ASR" in first.text
    assert "--device cpu" in first.action
    assert support_levels.FEEDBACK_URL in first.action


def test_official_and_cpu_placements_stay_silent(card, reported) -> None:
    card("nvidia", "windows")
    support_levels.note_accelerated_stage("ASR", "cuda")
    card("amd", "linux")
    support_levels.note_accelerated_stage("ASR", "cpu")

    assert reported.codes() == []


def test_intel_is_experimental_on_every_platform() -> None:
    """Nothing has been accepted on Intel hardware yet, on any OS."""

    for family in ("windows", "wsl", "linux", "macos"):
        assert support_levels.support_level("intel", family) == support_levels.EXPERIMENTAL


def test_an_xpu_placement_is_announced_as_an_accelerator(card, reported) -> None:
    """`xpu` must not slip past the notice the way a `cuda` prefix test let it.

    The gate is `is_gpu_device`, and Intel is the reason: on that hardware the
    CUDA token is not even in play, so a prefix test would keep the one notice
    users on the unverified platform most need from ever firing.
    """

    card("intel", "windows", "Intel(R) Arc(TM) 130T GPU (16GB)")

    support_levels.note_accelerated_stage("vocal separation", "xpu")

    [warning] = reported.warnings
    assert "Intel" in warning.text
    assert "vocal separation" in warning.text
    assert "--device cpu" in warning.action
    assert support_levels.FEEDBACK_URL in warning.action


def test_a_cpu_placement_on_an_intel_machine_is_not_announced(card, reported) -> None:
    card("intel", "windows")
    support_levels.note_accelerated_stage("ASR", "cpu")
    assert reported.codes() == []
