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
context earlier than the caller asked for.

**Why `add_dll_directory` reaches a C++ `LoadLibrary` at all**: CPython calls
`SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)` at startup, which
changes the search order for the whole process -- user directories included --
rather than only for Python's own loads. That is the same mechanism torch's
accidental fix rides on, which is why the experiment above behaved as it did.

**CUDA 12 is the target, not a debt.** The SONAME is compiled into the binary
and stays there on purpose: CUDA 13 is a new major with a far narrower
compatibility surface. So a `cublas64_13.dll` next door is not a newer version
of what we want -- it is the wrong environment, and this says so by name.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

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
    if os.name != "nt":
        # Elsewhere the loader resolves by RPATH/`LD_LIBRARY_PATH`, which this
        # process cannot change after start; there is nothing to add.
        _RESOLVED = ""
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
    wrong = _other_sonames()
    current_reporter().warning(
        "cublas-not-found",
        (
            f"已安装的包里只有 {'、'.join(wrong)}，没有 {CUBLAS_DLL}；"
            "本项目的 CTranslate2 按名字加载后者。"
            if wrong
            else f"未能在已安装的包里找到 {CUBLAS_DLL}"
            "（CTranslate2 在运行时按名字加载它）。"
        ),
        impact="若系统里也没有它，GPU 解码会在第一次矩阵运算时失败",
        action=(
            "本项目钉在 CUDA 12：请把 torch 换回 cu12x 构建"
            if wrong
            else "确认 torch 是 cu12x 构建，或安装 nvidia-cublas-cu12"
        ),
    )
    return _RESOLVED


def _other_sonames() -> list[str]:
    """Which cuBLAS *is* installed, when the one we want is not.

    A cu13x torch is the likely way to get here, and "cuBLAS not found" would
    send the reader looking for a missing file while the file sits right there
    under a name this build cannot use. Naming it turns the message into the
    actual diagnosis.
    """

    found: set[str] = set()
    for package, subdirectory in _CANDIDATES:
        root = _package_directory(package)
        if root is None:
            continue
        try:
            found.update(
                path.name for path in (root / subdirectory).glob("cublas64_*.dll")
            )
        except OSError:
            continue
    return sorted(found)
