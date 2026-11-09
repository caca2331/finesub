"""`scripts/pre-push` refuses dev's history on the public remote, and only there.

The public remote is recognised by the name `public` or by its GitHub URL;
`origin` is a private remote that legitimately carries `dev`, so a hook that
still matched it by name would block every ordinary push.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "pre-push"


def _find_sh() -> str | None:
    """A POSIX sh: on PATH, or the one Git for Windows ships next to git.

    From PowerShell, Git's `usr/bin` is not on PATH, and Windows is where the
    hook actually runs -- skipping there would leave it to Linux CI alone.
    """
    found = shutil.which("sh")
    if found:
        return found
    exec_path = subprocess.run(["git", "--exec-path"], capture_output=True, text=True).stdout.strip()
    if not exec_path:
        return None
    git_root = Path(exec_path).parents[2]
    for candidate in (git_root / "usr/bin/sh.exe", git_root / "bin/sh.exe"):
        if candidate.exists():
            return str(candidate)
    return None


SH = _find_sh()
ZERO = "0" * 40

pytestmark = pytest.mark.skipif(SH is None, reason="needs a POSIX sh")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "dev")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "dev")
    return tmp_path


def _orphan(repo: Path) -> str:
    tree = _git(repo, "write-tree")
    return _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit-tree", tree, "-m", "snapshot")


def _push(repo: Path, remote: str, url: str, sha: str, ref: str = "refs/heads/x") -> int:
    stdin = f"{ref} {sha} {ref} {ZERO}\n"
    return subprocess.run(
        [SH, str(HOOK), remote, url], cwd=repo, input=stdin, capture_output=True, text=True
    ).returncode


PUBLIC_URLS = (
    "https://github.com/caca2331/finesub.git",
    "https://github.com/caca2331/finesub",
    "git@github.com:caca2331/finesub.git",
    # GitHub paths are case-insensitive: a hand-typed URL must not slip past.
    "https://github.com/Caca2331/FineSub",
    "https://www.github.com/caca2331/finesub.git/",
)


def test_dev_history_is_refused_by_the_public_name(repo: Path) -> None:
    assert _push(repo, "public", "https://example.invalid/any.git", _git(repo, "rev-parse", "dev")) == 1


@pytest.mark.parametrize("name", ("mirror", "origin"))
@pytest.mark.parametrize("url", PUBLIC_URLS)
def test_dev_history_is_refused_by_the_public_url_under_any_name(repo: Path, url: str, name: str) -> None:
    assert _push(repo, name, url, _git(repo, "rev-parse", "dev")) == 1


@pytest.mark.parametrize(
    "url",
    (
        "https://github.com/someone/elsewhere.git",
        "https://github.com/caca2331/finesub-fork.git",
        "git@github.com:caca2331/finesub-fork.git",
    ),
)
def test_dev_history_passes_to_origin_on_another_url(repo: Path, url: str) -> None:
    assert _push(repo, "origin", url, _git(repo, "rev-parse", "dev")) == 0


def test_orphan_snapshot_passes_to_public(repo: Path) -> None:
    assert _push(repo, "public", PUBLIC_URLS[0], _orphan(repo), "refs/heads/ci-gate") == 0


def test_branch_deletion_passes_to_public(repo: Path) -> None:
    assert _push(repo, "public", PUBLIC_URLS[0], ZERO, "refs/heads/ci-gate") == 0
