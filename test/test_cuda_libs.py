"""Finding cuBLAS on purpose instead of by accident of import order.

CTranslate2 loads `cublas64_12.dll` by name from C++. Nothing in our wheel
ships it; it was found only because torch had already been imported and had
put its own bundled copy's directory on the search path. These tests are about
the deliberate search that replaced that accident -- none of them touch CUDA.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from finesub.speech.runtime import cuda_libs


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A blank search, on a machine that has DLL directories at all."""

    monkeypatch.setattr(cuda_libs, "_RESOLVED", None)
    monkeypatch.setattr(cuda_libs, "_HANDLES", [])
    monkeypatch.setattr(cuda_libs.os, "name", "nt")
    added: list[str] = []
    monkeypatch.setattr(
        os, "add_dll_directory", lambda path: added.append(path), raising=False
    )
    return added


def _install(monkeypatch: pytest.MonkeyPatch, roots: dict[str, Path]) -> None:
    monkeypatch.setattr(cuda_libs, "_package_directory", roots.get)


def _with_cublas(root: Path, subdirectory: str) -> Path:
    directory = root / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / cuda_libs.CUBLAS_DLL).write_bytes(b"")
    return directory


def test_torchs_bundled_copy_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str]
) -> None:
    directory = _with_cublas(tmp_path / "torch", "lib")
    _install(monkeypatch, {"torch": tmp_path / "torch"})

    assert cuda_libs.ensure_cublas_available() == str(directory)
    assert windows == [str(directory)]


def test_a_deliberate_install_wins_over_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str]
) -> None:
    """`nvidia-cublas-cu12` is somebody's decision; torch's copy is a by-product."""

    nvidia = _with_cublas(tmp_path / "nvidia" / "cublas", "bin")
    _with_cublas(tmp_path / "torch", "lib")
    _install(
        monkeypatch,
        {
            "nvidia.cublas": tmp_path / "nvidia" / "cublas",
            "torch": tmp_path / "torch",
        },
    )

    assert cuda_libs.ensure_cublas_available() == str(nvidia)


def test_a_package_without_the_dll_is_skipped_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str]
) -> None:
    """A cu13x torch has a `lib` directory and the wrong SONAME inside it."""

    (tmp_path / "torch" / "lib").mkdir(parents=True)
    (tmp_path / "torch" / "lib" / "cublas64_13.dll").write_bytes(b"")
    nvidia = _with_cublas(tmp_path / "nvidia" / "cublas", "bin")
    _install(
        monkeypatch,
        {
            "nvidia.cublas": tmp_path / "nvidia" / "cublas",
            "torch": tmp_path / "torch",
        },
    )

    assert cuda_libs.ensure_cublas_available() == str(nvidia)


def test_finding_nothing_warns_and_lets_the_model_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """A system CUDA toolkit on PATH still resolves it, so this is not fatal."""

    _install(monkeypatch, {})

    assert cuda_libs.ensure_cublas_available() == ""
    assert reported.codes() == ["cublas-not-found"]
    assert "cu12x" in reported.warnings[0].action


def test_the_wrong_cuda_generation_is_named_rather_than_called_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """A cu13x torch is the likely way here, and the file is not missing.

    Pinning CUDA 12 is the intent -- CUDA 13 buys far less compatibility --
    so `cublas64_13.dll` is the wrong environment rather than a newer one.
    Saying "not found" would send the reader hunting for a file sitting right
    there under a name this build cannot use.
    """

    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    (lib / "cublas64_13.dll").write_bytes(b"")
    _install(monkeypatch, {"torch": tmp_path / "torch"})

    assert cuda_libs.ensure_cublas_available() == ""
    assert "cublas64_13.dll" in reported.warnings[0].text
    assert "CUDA 12" in reported.warnings[0].action


def test_the_search_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """Which is also what keeps one warning from becoming one per pool model."""

    _install(monkeypatch, {})

    for _ in range(3):
        assert cuda_libs.ensure_cublas_available() == ""

    assert len(reported.warnings) == 1


def test_elsewhere_there_is_nothing_to_add(
    monkeypatch: pytest.MonkeyPatch, reported
) -> None:
    """Other loaders resolve by RPATH, which this process cannot change."""

    monkeypatch.setattr(cuda_libs, "_RESOLVED", None)
    monkeypatch.setattr(cuda_libs.os, "name", "posix")

    assert cuda_libs.ensure_cublas_available() == ""
    assert reported.warnings == []


def test_a_broken_namespace_package_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch, windows: list[str]
) -> None:
    """`find_spec` raises on a half-installed one; the run must not die there."""

    def explode(name: str):
        raise ValueError(name)

    monkeypatch.setattr(cuda_libs.importlib.util, "find_spec", explode)

    assert cuda_libs.ensure_cublas_available() == ""


def test_the_model_constructor_still_asks_before_it_builds() -> None:
    """A source guard, because the failure it prevents is invisible here.

    `faster_whisper` is not installed in the light suite, so the constructor
    cannot be exercised. It is also the wrong thing to exercise: a refactor
    that drops this call breaks nothing on a cu12x machine, and shows up only
    on the hardware nobody runs the tests on. Read the source instead.
    """

    import ast

    source = (
        Path(__file__).resolve().parents[1]
        / "src/finesub/speech/recognition/fw_refine_backend.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    model = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "RefinedWhisperModel"
    )
    init = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    called = [
        node.func.id
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "ensure_cublas_available" in called

    def line_of(predicate) -> int:
        return next(
            node.lineno
            for node in ast.walk(init)
            if isinstance(node, ast.Call) and predicate(node)
        )

    ours = line_of(
        lambda node: isinstance(node.func, ast.Name)
        and node.func.id == "ensure_cublas_available"
    )
    base = line_of(
        lambda node: isinstance(node.func, ast.Attribute)
        and node.func.attr == "__init__"
    )
    # After the model exists, CUDA may already have gone looking.
    assert ours < base
