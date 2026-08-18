"""The wheel build's two file lists must name files that exist.

`build-wheel.ps1` copies a few source trees into `_vendor`, then asserts that a
hand-written list of paths turned up -- once against the staging directory and
once inside the built wheel. Both lists are plain strings, so a rename moves
the files and leaves the lists behind, and the failure is a release-time
`Wheel staging is incomplete` from a script only CI runs.

That is exactly what the 2026-08 rename did: `llm` moved under `finesub` and
four entries in the staging list kept pointing at `_vendor\\src\\llm\\...`.
Nothing here ran PowerShell, so nothing noticed.
"""

from __future__ import annotations

from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPOSITORY_ROOT / "cli" / "scripts" / "build-wheel.ps1"

#: Both spellings the script uses: the staging check writes Windows paths, the
#: in-wheel check writes the zip's posix ones.
VENDORED_PATH = re.compile(r"_vendor[\\/]src[\\/]([\w\\/.-]+)")
COPIED_PACKAGES = re.compile(r"foreach \(\$Package in @\(([^)]*)\)\)")


def _script() -> str:
    return BUILD_SCRIPT.read_text(encoding="utf-8")


def test_every_vendored_path_the_build_checks_for_exists() -> None:
    referenced = {
        match.group(1).replace("\\", "/").rstrip("/")
        for match in VENDORED_PATH.finditer(_script())
    }

    assert referenced, "no vendored paths found; did the script change shape?"
    missing = sorted(
        path for path in referenced if not (REPOSITORY_ROOT / "src" / path).exists()
    )
    assert missing == [], (
        "build-wheel.ps1 checks for files that are not in src/: "
        + ", ".join(missing)
    )


def test_the_packages_the_build_copies_exist() -> None:
    match = COPIED_PACKAGES.search(_script())
    assert match, "could not find the package copy loop"
    packages = re.findall(r'"([^"]+)"', match.group(1))

    assert packages
    missing = sorted(
        name for name in packages if not (REPOSITORY_ROOT / "src" / name).is_dir()
    )
    assert missing == [], "build-wheel.ps1 copies packages that do not exist: " + ", ".join(
        missing
    )


def test_the_copied_packages_cover_every_checked_path() -> None:
    """A path under a package nobody copies can never turn up in staging."""

    script = _script()
    packages = re.findall(r'"([^"]+)"', COPIED_PACKAGES.search(script).group(1))
    referenced = {
        match.group(1).replace("\\", "/") for match in VENDORED_PATH.finditer(script)
    }

    orphans = sorted(
        path
        for path in referenced
        if not any(path == name or path.startswith(f"{name}/") for name in packages)
    )
    assert orphans == [], (
        "checked but never copied (the check would always fail): "
        + ", ".join(orphans)
    )
