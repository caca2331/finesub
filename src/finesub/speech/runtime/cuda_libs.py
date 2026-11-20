"""Make cuBLAS findable before CTranslate2 goes looking for it.

The patched CTranslate2 links the CUDA runtime statically but loads cuBLAS
**by name at runtime**: `cublas64_12.dll` appears as a string in the binary and
not in its import table (2026-08-05 teardown, recorded in
`docs/ct2-distribution.md`). Our wheel does not ship that DLL, so something
else has to have put its directory on the process search path first.

Until now that something was torch, by accident of ordering. `import torch`
calls `add_dll_directory` on `torch/lib`, where the cu128 build keeps its own
bundled `cublas64_12.dll`, and the VAD stage happens to import torch before the
ASR stage builds a model. The accident holds today and is invisible when it
stops holding -- reordering stages, a torch build that stops bundling CUDA, or
anything that reaches CTranslate2 without touching torch first all end in a
`LoadLibrary` failure from inside a C++ DLL, which names neither cuBLAS nor a
fix.

This asks for the directory on purpose instead. `find_spec` locates a package
without executing it, so nothing here imports torch: importing it would
recreate the very ordering dependency being removed, and would build a CUDA
context earlier than the caller asked for. The one question about torch's
build (`device.is_rocm()`, for the OpenMP workaround in
`prepare_native_runtime_for_import`) is asked only of a torch that is already
imported.

**Why `add_dll_directory` reaches a C++ `LoadLibrary` at all**: CPython calls
`SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)` at startup, which
changes the search order for the whole process -- user directories included --
rather than only for Python's own loads. That is the same mechanism torch's
accidental fix rides on, which is why the experiment above behaved as it did.

**CUDA 12 is the target, not a debt.** The SONAME is compiled into the binary
and stays there on purpose: CUDA 13 is a new major with a far narrower
compatibility surface. So a `cublas64_13.dll` next door is not a newer version
of what we want -- it is the wrong environment, and this says so by name.

**Linux** has the same by-name load (`libcublas.so.12`, via ``dlopen``) and a
harder version of the problem: the loader's search path is fixed when the
process starts, so ``LD_LIBRARY_PATH`` cannot be amended from here. What does
work is loading the library first, by full path, with ``RTLD_GLOBAL``: a later
``dlopen`` of the same SONAME finds it already loaded. The copy comes from the
``nvidia-cublas-cu12`` wheel that a CUDA build of torch depends on there.
``libcublasLt`` goes first because ``libcublas`` needs it. Under WSL2 the
driver-side libraries live in ``/usr/lib/wsl/lib`` instead of the usual place,
but cuBLAS is a user-space library from the wheel on both, so this path is the
same -- it is the driver underneath that differs.

Nothing to do on macOS (no CUDA), and nothing to do for the AMD HIP build **on
Linux**, where the wheel links against the system ROCm in ``/opt/rocm`` and the
loader finds it through ``ldconfig``.

**Windows has no system ROCm.** There the HIP wheel is built against the ROCm
SDK's own pip wheels (``ct2-build/windows-rocm.sh``), whose DLLs sit in
``<rocm_sdk_*>/bin`` inside site-packages -- not on ``PATH``. The ROCm build of
torch adds those directories when it is imported and CTranslate2 does not, so
whether ``import ctranslate2`` succeeds would again depend on whether torch was
imported first: the same accident this module exists to remove, one vendor over.
`_preload_windows_rocm` asks the SDK where its libraries are
(``rocm_sdk.find_libraries``, the API AMD gives frameworks for exactly this) and
adds those directories. ⚠ Written without an AMD machine to try it on; the tests
pin the shape, the first real card will decide the rest.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import logging
import os
import sys
from pathlib import Path

from ...doc_links import CT2_WHEEL_DOC, REPO_INSTALL_DOC
from ...reporting import current_reporter

#: Compiled into the patched CTranslate2, deliberately. CUDA 12.x keeps this
#: SONAME across its minor versions, which is the whole compatibility budget
#: this project wants; CUDA 13 does not.
CUBLAS_DLL = "cublas64_12.dll"

#: Where a wheel keeps it, best candidate first. `nvidia-cublas-cu12` is not a
#: dependency of this project -- on Windows torch vendors its own copy instead
#: of depending on the `nvidia-*` packages -- but an environment that has it
#: (a Linux-style install, or a user who added it) is the more deliberate of
#: the two answers, so it is asked first.
_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("nvidia.cublas", "bin"),
    ("torch", "lib"),
)

#: Short names for `rocm_sdk.find_libraries`, covering both packages the DLLs
#: are split across: the HIP runtime lives in the core wheel, BLAS in the
#: target-specific libraries wheel, and each has its own ``bin`` directory.
#: Asked for by name rather than by path because the layout is AMD's to change
#: and `find_libraries` is the interface they offer frameworks.
ROCM_SDK_LIBRARIES = ("amdhip64", "hipblas", "hipblaslt")

#: The Linux counterpart: the SONAMEs, in load order, and where wheels keep them.
CUBLAS_SONAMES = ("libcublasLt.so.12", "libcublas.so.12")
_LINUX_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("nvidia.cublas", "lib"),
    ("torch", "lib"),
)

#: `add_dll_directory` hands back a handle that removes the directory again
#: when closed. Nothing must close these for the life of the process.
_HANDLES: list[object] = []

#: The resolved directory, "" for "looked and found nothing". None means the
#: search has not run: the whole body runs once, which is also what keeps the
#: warning below from repeating once per model in the pool.
_RESOLVED: str | None = None


def _package_directory(name: str) -> Path | None:
    """Where an installed package lives, without importing it."""

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, AttributeError):
        # A partially installed namespace package raises rather than returning
        # None. Not finding cuBLAS is a warning; crashing the run over the
        # search for it would be worse than the problem.
        return None
    if spec is None:
        return None
    for location in spec.submodule_search_locations or []:
        return Path(location)
    return None


def ensure_cublas_available() -> str:
    """Put cuBLAS on the DLL search path; return the directory, or "".

    Idempotent and cheap after the first call. A "" result is not fatal -- a
    machine with a system CUDA toolkit on PATH resolves the load anyway -- so
    this warns and lets the model build.
    """

    global _RESOLVED
    if _RESOLVED is not None:
        return _RESOLVED
    if _ct2_is_hip_build():
        if os.name == "nt":
            # No system ROCm on Windows: the libraries came as pip wheels and
            # nothing has put them on the search path.
            _RESOLVED = _preload_windows_rocm()
            if not _RESOLVED:
                _warn_missing_rocm()
            return _RESOLVED
        # On Linux the wheel links the system ROCm; ldconfig finds it.
        _RESOLVED = ""
        return _RESOLVED
    if os.name != "nt":
        if not sys.platform.startswith("linux"):
            _RESOLVED = ""
            return _RESOLVED
        _RESOLVED = _preload_linux_cublas()
        if not _RESOLVED:
            _warn_missing()
        return _RESOLVED
    for package, subdirectory in _CANDIDATES:
        root = _package_directory(package)
        if root is None:
            continue
        directory = root / subdirectory
        if not (directory / CUBLAS_DLL).is_file():
            continue
        try:
            _HANDLES.append(os.add_dll_directory(str(directory)))
        except OSError:
            continue
        _RESOLVED = str(directory)
        return _RESOLVED
    _RESOLVED = ""
    _warn_missing()
    return _RESOLVED


def _ct2_is_hip_build() -> bool:
    """Whether the installed CTranslate2 is the AMD (ROCm/HIP) build of our wheel."""

    try:
        return "rocm" in importlib.metadata.version("ctranslate2")
    except importlib.metadata.PackageNotFoundError:
        return False


def _preload_linux_cublas() -> str:
    """Load cuBLAS by full path with RTLD_GLOBAL; return its directory, or ""."""

    for package, subdirectory in _LINUX_CANDIDATES:
        root = _package_directory(package)
        if root is None:
            continue
        directory = root / subdirectory
        if not (directory / CUBLAS_SONAMES[-1]).is_file():
            continue
        loaded = 0
        try:
            for soname in CUBLAS_SONAMES:
                if (directory / soname).is_file():
                    _HANDLES.append(
                        ctypes.CDLL(str(directory / soname), mode=ctypes.RTLD_GLOBAL)
                    )
                    loaded += 1
        except OSError:
            if loaded:
                # RTLD_GLOBAL cannot be taken back, so a half-loaded directory
                # ends the search: another directory's libcublas would bind to
                # the libcublasLt already in the process.
                return ""
            continue
        return str(directory)
    return ""


def _preload_windows_rocm() -> str:
    """Add the ROCm SDK's DLL directories; return the first, or "".

    Two steps, and the second is not redundant. `initialize_process` is what
    AMD tells a consuming framework to call (it is how the ROCm build of torch
    comes up on Windows): it preloads the named libraries, after which a load
    by name finds them already in the process. But CTranslate2 links more than
    the three named here, and a preloaded DLL's *own* imports still resolve
    through the normal search order -- so the directories go on the search path
    as well. Both sides ask `rocm_sdk` where things are rather than guessing a
    layout that is AMD's to change.
    """

    try:
        import rocm_sdk
    except ImportError:
        return ""
    # AMD's own entry point for a framework that is about to load native code
    # linked against ROCm: it preloads the libraries so later loads resolve them
    # by name. `preload_libraries` caches handles per short name, so calling it
    # after torch already did costs nothing (read from `rocm_sdk/__init__.py`:
    # `if shortname in _ALL_CDLLS: continue`).
    initialize = getattr(rocm_sdk, "initialize_process", None)
    if initialize is not None:
        try:
            initialize(preload_shortnames=list(ROCM_SDK_LIBRARIES))
        except Exception:
            # An SDK that renamed a library, or one whose preload fails for a
            # reason the directories below can still fix.
            pass
    # Still add the directories: preloading covers the libraries named above,
    # and CTranslate2 pulls in more than those. A preloaded DLL's own imports
    # go through the ordinary Windows search order, which is what these entries
    # are for.
    directories: list[str] = []
    for shortname in ROCM_SDK_LIBRARIES:
        try:
            found = rocm_sdk.find_libraries(shortname)
        except Exception:
            # Missing package, missing file, an unknown short name after an SDK
            # update: each is one library's problem, not the run's.
            continue
        for path in found:
            # os.path, not pathlib: the tests drive this Windows-only branch on
            # Linux by patching os.name, and pathlib then tries to build a
            # WindowsPath, which it cannot do off Windows.
            directory = os.path.dirname(os.fspath(path))
            if directory not in directories:
                directories.append(directory)
    for directory in directories:
        try:
            _HANDLES.append(os.add_dll_directory(directory))
        except OSError:
            continue
    return directories[0] if directories else ""


def _warn_missing_rocm() -> None:
    current_reporter().warning(
        "rocm-runtime-not-found",
        (
            "这是 ROCm 版 CTranslate2，但没能在本环境里找到 ROCm 的运行库"
            "（rocm_sdk 系列包）；识别可能在加载模型时失败。"
        ),
        impact="识别阶段可能报 DLL load failed",
        action=f"确认 rocm_sdk 几个包与本项目装在同一个环境里，见 {REPO_INSTALL_DOC}",
    )


def prepare_native_runtime_for_import() -> None:
    """Whatever must be on the search path *before* CTranslate2 is imported.

    Two things, both Windows-only. First, a Windows ROCm torch carries LLVM
    OpenMP and CTranslate2 carries Intel OpenMP; the second one to initialise
    in a process that already has the first exits unless
    `KMP_DUPLICATE_LIB_OK` is set. So when that torch is already imported, an
    unset variable is filled with `TRUE` (a user's own value, `FALSE`
    included, is kept) and a debug line says this module set it. A torch that
    is not imported yet is not asked: importing it here is exactly what this
    module must not do, and the pipeline always imports torch before the ASR
    stage. ⚠ So a caller that imports CTranslate2 first and a ROCm torch
    afterwards is not covered; set the variable yourself there.

    Second, only the HIP build needs the DLL search path: there the ROCm DLLs are
    in its import table, so the process must be able to find them at import
    time -- unlike cuBLAS, which the CUDA build dlopens by name at the first
    GEMM and which therefore survives being prepared later, in the model
    constructor.

    Preparing it in the constructor was the bug: the constructor lives in a
    module whose import is what fails. Cheap and idempotent (`_RESOLVED` is
    computed once), so the constructor keeps asking too -- that call is what
    the source guard checks, and it still covers a caller that reaches
    CTranslate2 some other way.
    """

    if (
        sys.platform == "win32"
        and "KMP_DUPLICATE_LIB_OK" not in os.environ
        and "torch" in sys.modules
    ):
        from .device import is_rocm

        if is_rocm():
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            logging.getLogger(__name__).debug(
                "Set KMP_DUPLICATE_LIB_OK=TRUE for Windows ROCm native runtime coexistence"
            )
    if os.name == "nt" and _ct2_is_hip_build():
        ensure_cublas_available()


def missing_asr_backend_message(error: BaseException) -> str:
    """What to tell a user whose `fw_refine_backend` import just failed.

    Two shapes reach here and they need different advice. A
    `ModuleNotFoundError` means the packages are not installed, so naming the
    extra helps -- except that `[asr]` is the wrong extra on Windows + AMD
    (it replaces a ROCm torch with the CUDA-less PyPI build), so the install
    page decides which one rather than this sentence. Any other `ImportError`
    means the packages are there and something native did not load, and the
    only useful thing to say is what `describe_ct2_import_failure` worked out.
    """

    if isinstance(error, ModuleNotFoundError):
        # Not the native-load story: the package is simply absent, and dressing
        # that up as a ROCm runtime problem would send them after the wrong one.
        return (
            f"缺少 ASR 依赖：faster-whisper 与打过补丁的 CTranslate2（{error}）。"
            f"按你的平台装，见 {REPO_INSTALL_DOC}"
        )
    return f"ASR 后端装着，但没能加载起来：{describe_ct2_import_failure(error)}"


def describe_ct2_import_failure(error: BaseException) -> str:
    """What to say when importing CTranslate2 (or faster-whisper) blows up.

    A failed `import ctranslate2` on Windows reads `DLL load failed while
    importing _ext`, which names nothing a user can act on. For the HIP wheel
    the cause is nearly always that the ROCm SDK wheels are absent from this
    environment, so say that instead of leaving them the raw message.
    """

    if os.name == "nt" and _ct2_is_hip_build():
        return (
            f"ROCm 版 CTranslate2 未能加载它的运行库（{error}）；"
            f"确认 rocm_sdk 系列包与本项目装在同一个环境里，见 {CT2_WHEEL_DOC}"
        )
    return str(error)


def _warn_missing() -> None:
    wanted = CUBLAS_DLL if os.name == "nt" else CUBLAS_SONAMES[-1]
    wrong = _other_sonames()
    current_reporter().warning(
        "cublas-not-found",
        (
            f"已安装的包里只有 {'、'.join(wrong)}，没有 {wanted}；"
            "本项目的 CTranslate2 按名字加载后者。"
            if wrong
            else f"未能在已安装的包里找到 {wanted}"
            "（CTranslate2 在运行时按名字加载它）。"
        ),
        impact="若系统里也没有它，GPU 解码会在第一次矩阵运算时失败",
        action=(
            "本项目钉在 CUDA 12：请把 torch 换回 cu12x 构建"
            if wrong
            else "确认 torch 是 cu12x 构建，或安装 nvidia-cublas-cu12"
        ),
    )


def _other_sonames() -> list[str]:
    """Which cuBLAS *is* installed, when the one we want is not.

    A cu13x torch is the likely way to get here, and "cuBLAS not found" would
    send the reader looking for a missing file while the file sits right there
    under a name this build cannot use. Naming it turns the message into the
    actual diagnosis.
    """

    found: set[str] = set()
    if os.name == "nt":
        candidates, pattern = _CANDIDATES, "cublas64_*.dll"
    else:
        candidates, pattern = _LINUX_CANDIDATES, "libcublas.so.*"
    for package, subdirectory in candidates:
        root = _package_directory(package)
        if root is None:
            continue
        try:
            found.update(path.name for path in (root / subdirectory).glob(pattern))
        except OSError:
            continue
    return sorted(found)
