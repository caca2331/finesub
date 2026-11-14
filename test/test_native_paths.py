"""Native loaders must not depend on the ANSI code page to read a path.

The maintainer's machine runs with code page 65001, where narrow-character
opens already decode UTF-8, so these tests cannot reproduce the GBK failure
itself. They pin the two measures instead: the CRT locale switch happens
before the pipeline opens anything, and TorchScript archives reach libtorch as
file objects rather than path strings.
"""

from __future__ import annotations

import locale
import sys
from pathlib import Path

import pytest

from finesub.speech.runtime import native_paths


def _torch():
    """Only the tests that load TorchScript need torch; the locale ones run
    everywhere, including CI's torch-less environments."""

    return pytest.importorskip("torch")


def _save_tiny_module(path: Path) -> None:
    torch = _torch()

    class Doubler(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value * 2

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(Doubler()), str(path))


def test_torchscript_loads_from_a_non_ascii_directory(tmp_path: Path) -> None:
    torch = _torch()
    archive = tmp_path / "模型目录" / "doubler.pt"
    _save_tiny_module(archive)

    module = native_paths.load_torchscript(archive, map_location="cpu")

    assert module(torch.tensor([1.5])).item() == 3.0


def test_libtorch_is_handed_a_file_object_not_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = _torch()
    archive = tmp_path / "doubler.pt"
    _save_tiny_module(archive)
    received: list[object] = []
    real_load = torch.jit.load

    def spy(source, map_location=None):
        received.append(source)
        return real_load(source, map_location=map_location)

    monkeypatch.setattr(torch.jit, "load", spy)
    native_paths.load_torchscript(archive, map_location="cpu")

    assert len(received) == 1
    assert not isinstance(received[0], (str, bytes, Path))
    assert hasattr(received[0], "read")


@pytest.mark.skipif(sys.platform != "win32", reason="the UCRT locale is Windows-only")
def test_the_crt_locale_switches_to_utf8() -> None:
    before = locale.setlocale(locale.LC_CTYPE)
    try:
        native_paths.use_utf8_crt_locale()
        assert locale.setlocale(locale.LC_CTYPE).lower().endswith((".utf8", ".utf-8"))
    finally:
        locale.setlocale(locale.LC_CTYPE, before)


def test_off_windows_the_locale_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(locale, "setlocale", lambda *args: calls.append(args))

    native_paths.use_utf8_crt_locale()

    assert calls == []


def test_a_crt_without_utf8_support_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args):
        raise locale.Error("unsupported locale setting")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(locale, "setlocale", refuse)

    problem = native_paths.use_utf8_crt_locale()

    assert problem is not None and "UTF-8" in problem


def test_the_pipeline_switches_the_locale_before_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _torch()  # importing the pipeline imports the speech stages
    from finesub import pipeline

    order: list[str] = []

    def stop_after_parse():
        order.append("parse_args")
        raise SystemExit(0)

    monkeypatch.setattr(pipeline, "use_utf8_crt_locale", lambda: order.append("locale"))
    monkeypatch.setattr(pipeline, "parse_args", stop_after_parse)

    with pytest.raises(SystemExit):
        pipeline.main()

    assert order == ["locale", "parse_args"]


def test_silero_is_loaded_through_the_file_object_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _torch()
    pytest.importorskip("silero_vad")
    from finesub.speech.preprocessing import silero_ghost

    loaded: list[Path] = []
    real = silero_ghost.load_torchscript

    def spy(path, *, map_location):
        loaded.append(Path(path))
        return real(path, map_location=map_location)

    monkeypatch.setattr(silero_ghost, "load_torchscript", spy)
    monkeypatch.setattr(silero_ghost, "_MODEL", None)
    threads = torch.get_num_threads()

    silero_ghost.silero_model()

    assert [path.name for path in loaded] == ["silero_vad.jit"]
    # Importing silero_vad sets one thread process-wide; the loader restores it.
    assert torch.get_num_threads() == threads
