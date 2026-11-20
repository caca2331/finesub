"""Finding cuBLAS on purpose instead of by accident of import order.

CTranslate2 loads `cublas64_12.dll` by name from C++. Nothing in our wheel
ships it; it was found only because torch had already been imported and had
put its own bundled copy's directory on the search path. These tests are about
the deliberate search that replaced that accident -- none of them touch CUDA.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.speech.runtime import cuda_libs


@pytest.mark.parametrize(
    "platform,hip,cuda,xpu,ct2_hip,existing,expected",
    [
        ("win32", "7.15", None, None, True, None, "TRUE"),
        ("win32", "7.15", None, None, False, None, "TRUE"),
        ("win32", "7.15", None, None, True, "FALSE", "FALSE"),
        ("win32", "7.15", None, None, True, "TRUE", "TRUE"),
        ("win32", "7.15", None, None, True, "", ""),
        ("linux", "7.15", None, None, True, None, None),
        ("darwin", "7.15", None, None, True, None, None),
        ("win32", None, "12.8", None, False, None, None),
        ("win32", None, None, None, False, None, None),
        ("win32", None, None, "2026.0", False, None, None),
    ],
)
def test_native_preparation_sets_openmp_before_loading_and_logs_only_its_default(
    monkeypatch, caplog, platform, hip, cuda, xpu, ct2_hip, existing, expected
) -> None:
    """The torch build decides the workaround; the CT2 build decides DLL setup."""

    import torch

    environment = {} if existing is None else {"KMP_DUPLICATE_LIB_OK": existing}
    # Replace only this module's OS/platform boundary: no global environment
    # writes, SDK loads or host-dependent WindowsPath construction are needed.
    monkeypatch.setattr(
        cuda_libs, "os", SimpleNamespace(name="nt" if platform == "win32" else "posix", environ=environment)
    )
    monkeypatch.setattr(
        cuda_libs, "sys", SimpleNamespace(platform=platform, modules={"torch": torch})
    )
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    monkeypatch.setattr(torch.version, "cuda", cuda, raising=False)
    monkeypatch.setattr(torch.version, "xpu", xpu, raising=False)
    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: ct2_hip)
    seen_before_loading: list[str | None] = []
    monkeypatch.setattr(
        cuda_libs,
        "ensure_cublas_available",
        lambda: seen_before_loading.append(environment.get("KMP_DUPLICATE_LIB_OK")),
    )

    with caplog.at_level("DEBUG", logger=cuda_libs.__name__):
        cuda_libs.prepare_native_runtime_for_import()
        cuda_libs.prepare_native_runtime_for_import()

    assert environment.get("KMP_DUPLICATE_LIB_OK") == expected
    assert seen_before_loading == ([expected, expected] if platform == "win32" and ct2_hip else [])
    supplied_default = platform == "win32" and bool(hip) and existing is None
    records = [record for record in caplog.records if "KMP_DUPLICATE_LIB_OK" in record.getMessage()]
    assert len(records) == int(supplied_default)
    if supplied_default:
        assert "KMP_DUPLICATE_LIB_OK=TRUE" in records[0].getMessage()
        assert "Windows ROCm" in records[0].getMessage()


def test_native_preparation_does_not_import_torch_to_ask_its_build(monkeypatch) -> None:
    """Before torch is imported there is no LLVM OpenMP to clash with, and
    importing it here would bring back the import-order dependency this module removes."""

    from finesub.speech.runtime import device

    environment: dict[str, str] = {}
    monkeypatch.setattr(cuda_libs, "os", SimpleNamespace(name="nt", environ=environment))
    monkeypatch.setattr(cuda_libs, "sys", SimpleNamespace(platform="win32", modules={}))
    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: False)

    def asked() -> bool:
        raise AssertionError("asked torch's build before torch was imported")

    monkeypatch.setattr(device, "is_rocm", asked)

    cuda_libs.prepare_native_runtime_for_import()

    assert "KMP_DUPLICATE_LIB_OK" not in environment


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A blank search, on a machine that has DLL directories at all."""

    monkeypatch.setattr(cuda_libs, "_RESOLVED", None)
    monkeypatch.setattr(cuda_libs, "_HANDLES", [])
    monkeypatch.setattr(cuda_libs.os, "name", "nt")
    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: False)
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


def test_macos_has_nothing_to_add(monkeypatch: pytest.MonkeyPatch, reported) -> None:
    monkeypatch.setattr(cuda_libs, "_RESOLVED", None)
    monkeypatch.setattr(cuda_libs.os, "name", "posix")
    monkeypatch.setattr(cuda_libs.sys, "platform", "darwin")

    assert cuda_libs.ensure_cublas_available() == ""
    assert reported.warnings == []


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """A blank search on Linux, recording what gets preloaded and how."""

    monkeypatch.setattr(cuda_libs, "_RESOLVED", None)
    monkeypatch.setattr(cuda_libs, "_HANDLES", [])
    monkeypatch.setattr(cuda_libs.os, "name", "posix")
    monkeypatch.setattr(cuda_libs.sys, "platform", "linux")
    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: False)
    loaded: list[tuple[str, int]] = []
    monkeypatch.setattr(
        cuda_libs.ctypes, "RTLD_GLOBAL", getattr(cuda_libs.ctypes, "RTLD_GLOBAL", 256), raising=False
    )
    monkeypatch.setattr(
        cuda_libs.ctypes, "CDLL", lambda path, mode=0: loaded.append((path, mode)) or object()
    )
    return loaded


def test_linux_preloads_the_wheel_copy_globally_lt_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linux, reported
) -> None:
    directory = tmp_path / "nvidia" / "cublas" / "lib"
    directory.mkdir(parents=True)
    for soname in cuda_libs.CUBLAS_SONAMES:
        (directory / soname).write_bytes(b"")
    _install(monkeypatch, {"nvidia.cublas": tmp_path / "nvidia" / "cublas"})

    assert cuda_libs.ensure_cublas_available() == str(directory)
    # Split by hand: with os.name patched, pathlib on a Windows host misreads "\\".
    names = [path.replace("\\", "/").rsplit("/", 1)[-1] for path, _mode in linux]
    assert names == list(cuda_libs.CUBLAS_SONAMES)
    assert all(mode == cuda_libs.ctypes.RTLD_GLOBAL for _path, mode in linux)
    assert reported.warnings == []


def test_linux_without_cublas_warns_with_the_soname(
    monkeypatch: pytest.MonkeyPatch, linux, reported
) -> None:
    _install(monkeypatch, {})

    assert cuda_libs.ensure_cublas_available() == ""
    assert "libcublas.so.12" in reported.joined()


def test_a_half_loaded_directory_ends_the_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, linux, reported
) -> None:
    """Lt in, libcublas refused: the next directory would bind to the wrong Lt.

    RTLD_GLOBAL cannot be taken back, so there is no "try the next one" left
    once half a pair is in the process.
    """

    first = tmp_path / "torch" / "lib"
    second = tmp_path / "nvidia" / "cublas" / "lib"
    for directory in (first, second):
        directory.mkdir(parents=True)
        for soname in cuda_libs.CUBLAS_SONAMES:
            (directory / soname).write_bytes(b"")

    def refuse_the_second(path: str, mode: int = 0):
        linux.append((path, mode))
        if path.endswith(cuda_libs.CUBLAS_SONAMES[-1]):
            raise OSError("wrong ABI")
        return object()

    monkeypatch.setattr(cuda_libs.ctypes, "CDLL", refuse_the_second)
    _install(
        monkeypatch,
        {"torch": tmp_path / "torch", "nvidia.cublas": tmp_path / "nvidia" / "cublas"},
    )

    assert cuda_libs.ensure_cublas_available() == ""
    # One directory attempted, not two: Lt then the refusal, and nothing after.
    assert len(linux) == len(cuda_libs.CUBLAS_SONAMES)
    assert reported.warnings  # the user is told cuBLAS is not usable


def test_the_amd_build_needs_no_cublas(monkeypatch: pytest.MonkeyPatch, linux, reported) -> None:
    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: True)

    assert cuda_libs.ensure_cublas_available() == ""
    assert linux == [] and reported.warnings == []


class _FakeRocmSdk:
    """Stands in for AMD's `rocm_sdk`, whose real API returns absolute paths."""

    def __init__(self, libraries: dict[str, list[Path]], *, preloads: bool = True) -> None:
        self.libraries = libraries
        self.asked: list[str] = []
        self.preloaded: list[str] | None = None
        if not preloads:  # an older SDK without the entry point
            self.initialize_process = None

    def find_libraries(self, shortname: str):
        self.asked.append(shortname)
        if shortname not in self.libraries:
            raise ModuleNotFoundError(shortname)
        return self.libraries[shortname]

    def initialize_process(self, *, preload_shortnames=None, **kwargs):
        self.preloaded = list(preload_shortnames or [])


def _install_rocm_sdk(monkeypatch, libraries) -> _FakeRocmSdk:
    fake = _FakeRocmSdk(libraries)
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake)
    return fake


def test_the_windows_amd_build_adds_the_sdk_dll_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """Windows has no system ROCm: the DLLs are in site-packages, off PATH.

    Both wheels are asked for, because the HIP runtime and BLAS ship in
    different ones and each has its own `bin`.
    """

    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: True)
    core = tmp_path / "_rocm_sdk_core" / "bin"
    libraries = tmp_path / "_rocm_sdk_libraries_gfx120X" / "bin"
    for directory in (core, libraries):
        directory.mkdir(parents=True)
    fake = _install_rocm_sdk(
        monkeypatch,
        {
            "amdhip64": [core / "amdhip64_7.dll"],
            "hipblas": [libraries / "hipblas.dll"],
            "hipblaslt": [libraries / "hipblaslt.dll"],
        },
    )

    resolved = cuda_libs.ensure_cublas_available()

    assert resolved == str(core)
    # Each directory once, in the order the libraries were asked for.
    assert windows == [str(core), str(libraries)]
    assert fake.asked == list(cuda_libs.ROCM_SDK_LIBRARIES)
    assert reported.warnings == []


def test_the_windows_amd_build_lets_the_sdk_preload_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """`initialize_process` is AMD's entry point for exactly this moment.

    The directories still go on the search path: it preloads the libraries we
    name, and CTranslate2 links more than those.
    """

    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: True)
    core = tmp_path / "_rocm_sdk_core" / "bin"
    core.mkdir(parents=True)
    fake = _install_rocm_sdk(monkeypatch, {"amdhip64": [core / "amdhip64_7.dll"]})

    assert cuda_libs.ensure_cublas_available() == str(core)
    assert fake.preloaded == list(cuda_libs.ROCM_SDK_LIBRARIES)
    assert windows == [str(core)]
    assert reported.warnings == []


def test_an_sdk_without_the_entry_point_still_gets_its_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """Older SDKs have no `initialize_process`; the directories are the fallback."""

    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: True)
    core = tmp_path / "_rocm_sdk_core" / "bin"
    core.mkdir(parents=True)
    fake = _FakeRocmSdk({"amdhip64": [core / "amdhip64_7.dll"]}, preloads=False)
    monkeypatch.setitem(sys.modules, "rocm_sdk", fake)

    assert cuda_libs.ensure_cublas_available() == str(core)
    assert windows == [str(core)]
    assert reported.warnings == []


def test_the_windows_amd_build_says_so_when_the_sdk_is_absent(
    monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """`pip install` of the wheels was skipped: the run must name them."""

    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: True)
    monkeypatch.setitem(sys.modules, "rocm_sdk", None)  # import -> ImportError

    assert cuda_libs.ensure_cublas_available() == ""
    assert windows == []
    said = reported.joined()
    assert "rocm_sdk" in said and "repo-install.md" in said


def test_a_failed_ct2_import_names_the_rocm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`DLL load failed while importing _ext` is not something a user can act on."""

    monkeypatch.setattr(cuda_libs.os, "name", "nt")
    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: True)

    said = cuda_libs.describe_ct2_import_failure(
        ImportError("DLL load failed while importing _ext")
    )

    assert "ROCm" in said and "rocm_sdk" in said and "ct2-wheel.md" in said
    assert "DLL load failed" in said  # the original stays in the sentence


def test_a_failed_ct2_import_elsewhere_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the HIP wheel on Windows gets the ROCm explanation."""

    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: False)

    assert (
        cuda_libs.describe_ct2_import_failure(ImportError("No module named x"))
        == "No module named x"
    )


def test_the_windows_amd_build_never_goes_looking_for_cublas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, windows: list[str], reported
) -> None:
    """It wants ROCm, not cuBLAS -- even on a machine that has both.

    (This test used to assert the Windows HIP build needed nothing at all,
    which was the Linux answer applied to a platform with no system ROCm.)
    """

    monkeypatch.setattr(cuda_libs, "_ct2_is_hip_build", lambda: True)
    _with_cublas(tmp_path / "torch", "lib")
    _install(monkeypatch, {"torch": tmp_path / "torch"})
    core = tmp_path / "_rocm_sdk_core" / "bin"
    core.mkdir(parents=True)
    _install_rocm_sdk(monkeypatch, {"amdhip64": [core / "amdhip64_7.dll"]})

    assert cuda_libs.ensure_cublas_available() == str(core)
    assert windows == [str(core)]
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
