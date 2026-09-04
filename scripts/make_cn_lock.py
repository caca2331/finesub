"""Derive a regional lock from the canonical one by rewriting artifact URLs.

The canonical lock is the only place a version is ever resolved. This rewrites
where each already-decided file is fetched from and nothing else, so the two
locks describe the same environment served by different hosts -- which is what
makes falling back between them safe and what `assert_equivalent` checks.

Why this is a URL rewrite and not a second resolve: resolving twice can
legitimately produce different versions (a mirror lagging an index, a yanked
release), and then "fall back to the other lock" would silently change the
environment instead of just its source.

Usage:

    python -m scripts.make_cn_lock \\
        src/finesub_bootstrap/pylock.win-py312.toml \\
        --mirror https://mirror.example/pypi/web/ \\
        [--torch-mirror https://mirror.example/pytorch-wheels/] \\
        --output src/finesub_bootstrap/pylock.win-py312.cn.toml
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tomllib
from typing import Any

PYPI_PREFIX = "https://files.pythonhosted.org/packages/"
TORCH_PREFIXES = (
    "https://download-r2.pytorch.org/whl/",
    "https://download.pytorch.org/whl/",
)

_URL = re.compile(r'url = "([^"]+)"')


class LockMismatch(RuntimeError):
    """The two locks do not describe the same environment."""


def _pypi_files_base(index_url: str) -> str:
    """Turn a PyPI *index* URL into the base its package files sit under.

    A mirror publishes two things at different paths on the same host: the
    simple index it hands to pip, and the mirrored `files.pythonhosted.org`
    tree the lock's URLs point into. Every bandersnatch-style mirror
    (TUNA, BFSU, SJTU) serves the second at `.../packages`, so the index's
    trailing `/simple` is what has to be replaced.
    """

    trimmed = index_url.strip().rstrip("/")
    if not trimmed:
        return ""
    if trimmed.endswith("/simple"):
        trimmed = trimmed[: -len("/simple")]
    return f"{trimmed}/packages"


def rewrite(text: str, *, mirror: str, torch_mirror: str = "") -> str:
    """Point every rewritable artifact URL at a mirror, leaving the rest alone.

    Partial acceleration is a deliberate outcome: a file with no verified
    mirror keeps its official URL rather than being dropped or guessed at.
    """

    def replace(match: re.Match[str]) -> str:
        url = match.group(1)
        if mirror and url.startswith(PYPI_PREFIX):
            return f'url = "{mirror.rstrip("/")}/{url[len(PYPI_PREFIX):]}"'
        if torch_mirror:
            for prefix in TORCH_PREFIXES:
                if url.startswith(prefix):
                    tail = url[len(prefix):]
                    return f'url = "{torch_mirror.rstrip("/")}/{tail}"'
        return match.group(0)

    return _URL.sub(replace, text)


def _packages(body: dict[str, Any]) -> dict[str, Any]:
    return {
        str(package.get("name")): package
        for package in body.get("packages", [])
        if isinstance(package, dict)
    }


def _fingerprint(package: dict[str, Any]) -> dict[str, Any]:
    """Everything about a package except where its files are fetched from."""

    def files(entries: Any) -> list[tuple[str, str, Any]]:
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            return []
        described = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            url = str(entry.get("url", ""))
            hashes = entry.get("hashes")
            described.append(
                (
                    # The filename is part of the identity; the host is not.
                    url.rsplit("/", 1)[-1],
                    str((hashes or {}).get("sha256", "")),
                    entry.get("size"),
                )
            )
        return sorted(described)

    return {
        "version": package.get("version"),
        "marker": package.get("marker"),
        "requires-python": package.get("requires-python"),
        "wheels": files(package.get("wheels")),
        "sdist": files(package.get("sdist")),
        "archive": files(package.get("archive")),
    }


def assert_equivalent(canonical: Path, regional: Path) -> None:
    """Fail unless the two locks differ only in hosts.

    This is the gate that makes a regional lock safe to ship: same packages,
    same versions, same markers, same filenames, same digests. Any drift means
    the fallback would install a different environment than the one that was
    tested.
    """

    left = tomllib.loads(canonical.read_text(encoding="utf-8"))
    right = tomllib.loads(regional.read_text(encoding="utf-8"))
    left_packages, right_packages = _packages(left), _packages(right)

    missing = sorted(set(left_packages) - set(right_packages))
    extra = sorted(set(right_packages) - set(left_packages))
    if missing or extra:
        raise LockMismatch(
            f"package sets differ (missing: {missing}, unexpected: {extra})"
        )
    for name, package in left_packages.items():
        expected = _fingerprint(package)
        actual = _fingerprint(right_packages[name])
        if expected != actual:
            raise LockMismatch(
                f"{name} differs beyond its host: {expected} != {actual}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("canonical", type=Path)
    parser.add_argument(
        "--mirror",
        default=None,
        help="PyPI mirror base URL (default: download-sources.json).",
    )
    parser.add_argument(
        "--torch-mirror",
        default=None,
        help=(
            "Mirror for the torch wheels, which are most of the download "
            "(default: download-sources.json)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    # Defaulted from the shipped table so the hosts a release uses are written
    # down in one reviewable place rather than in whoever's shell history.
    if arguments.mirror is None or arguments.torch_mirror is None:
        from finesub_bootstrap.download_sources import load_sources

        sources = load_sources()
        if arguments.mirror is None:
            arguments.mirror = _pypi_files_base(str(sources.get("pypiIndex", "")))
        if arguments.torch_mirror is None:
            arguments.torch_mirror = str(sources.get("torchMirror", ""))

    if not arguments.mirror and not arguments.torch_mirror:
        print("nothing to rewrite: pass --mirror and/or --torch-mirror", file=sys.stderr)
        return 2
    text = arguments.canonical.read_text(encoding="utf-8")
    rewritten = rewrite(
        text, mirror=arguments.mirror, torch_mirror=arguments.torch_mirror
    )
    arguments.output.write_text(rewritten, encoding="utf-8", newline="\n")
    try:
        assert_equivalent(arguments.canonical, arguments.output)
    except LockMismatch as error:
        arguments.output.unlink(missing_ok=True)
        print(f"refusing to write a lock that is not equivalent: {error}", file=sys.stderr)
        return 1
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
