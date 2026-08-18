"""Deriving a regional lock, and refusing one that is not equivalent."""

from __future__ import annotations

from pathlib import Path

import pytest

from desktop.scripts.make_cn_lock import (
    LockMismatch,
    assert_equivalent,
    main,
    rewrite,
)

CANONICAL = """\
[[packages]]
name = "numpy"
version = "2.4.6"
marker = "sys_platform == 'win32'"
wheels = [{ url = "https://files.pythonhosted.org/packages/ab/numpy-2.4.6-cp312-win_amd64.whl", size = 11, hashes = { sha256 = "aaa" } }]

[[packages]]
name = "torch"
version = "2.11.0+cu128"
wheels = [{ url = "https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl", hashes = { sha256 = "bbb" } }]

[[packages]]
name = "ctranslate2"
version = "4.8.1+wtrefine1"
archive = { url = "https://github.com/example/releases/download/x/ctranslate2.whl", hashes = { sha256 = "ccc" } }
"""


def test_pypi_urls_move_to_the_mirror(tmp_path: Path) -> None:
    rewritten = rewrite(CANONICAL, mirror="https://mirror.example/pypi/web/")

    assert "https://mirror.example/pypi/web/ab/numpy-2.4.6-cp312-win_amd64.whl" in rewritten
    # Untouched without a mirror of its own: partial acceleration beats a guess.
    assert "https://download-r2.pytorch.org/whl/cu128/" in rewritten


def test_torch_is_rewritten_separately(tmp_path: Path) -> None:
    """Torch is most of the download; it earns its own, separately verified host."""

    rewritten = rewrite(
        CANONICAL, mirror="", torch_mirror="https://mirror.example/pytorch-wheels/"
    )

    assert (
        "https://mirror.example/pytorch-wheels/cu128/torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl"
        in rewritten
    )
    assert "https://files.pythonhosted.org/packages/ab/" in rewritten


def test_a_url_with_no_mirror_keeps_its_official_host() -> None:
    rewritten = rewrite(
        CANONICAL,
        mirror="https://mirror.example/pypi/web/",
        torch_mirror="https://mirror.example/pytorch-wheels/",
    )

    assert "https://github.com/example/releases/download/x/ctranslate2.whl" in rewritten


def test_a_rewritten_lock_is_equivalent_to_its_source(tmp_path: Path) -> None:
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    regional = tmp_path / "pylock.cn.toml"
    regional.write_text(
        rewrite(
            CANONICAL,
            mirror="https://mirror.example/pypi/web/",
            torch_mirror="https://mirror.example/pytorch-wheels/",
        ),
        encoding="utf-8",
    )

    assert_equivalent(canonical, regional)


def test_a_changed_version_is_refused(tmp_path: Path) -> None:
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    regional = tmp_path / "pylock.cn.toml"
    regional.write_text(CANONICAL.replace('version = "2.4.6"', 'version = "2.4.7"'), encoding="utf-8")

    with pytest.raises(LockMismatch, match="numpy"):
        assert_equivalent(canonical, regional)


def test_a_changed_digest_is_refused(tmp_path: Path) -> None:
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    regional = tmp_path / "pylock.cn.toml"
    regional.write_text(CANONICAL.replace('"aaa"', '"zzz"'), encoding="utf-8")

    with pytest.raises(LockMismatch):
        assert_equivalent(canonical, regional)


def test_a_changed_marker_is_refused(tmp_path: Path) -> None:
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    regional = tmp_path / "pylock.cn.toml"
    regional.write_text(
        CANONICAL.replace("sys_platform == 'win32'", "sys_platform == 'linux'"),
        encoding="utf-8",
    )

    with pytest.raises(LockMismatch):
        assert_equivalent(canonical, regional)


def test_a_dropped_package_is_refused(tmp_path: Path) -> None:
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    regional = tmp_path / "pylock.cn.toml"
    regional.write_text(CANONICAL.split("[[packages]]\nname = \"torch\"")[0], encoding="utf-8")

    with pytest.raises(LockMismatch, match="package sets differ"):
        assert_equivalent(canonical, regional)


def test_a_renamed_file_is_refused(tmp_path: Path) -> None:
    """Only the host may change; the filename is part of what was resolved."""

    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    regional = tmp_path / "pylock.cn.toml"
    regional.write_text(
        CANONICAL.replace("numpy-2.4.6-cp312-win_amd64.whl", "numpy-2.4.6-py3-none-any.whl"),
        encoding="utf-8",
    )

    with pytest.raises(LockMismatch):
        assert_equivalent(canonical, regional)


def test_the_generator_writes_nothing_when_the_result_would_drift(
    tmp_path: Path, monkeypatch
) -> None:
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    output = tmp_path / "pylock.cn.toml"

    def drifting(text, *, mirror, torch_mirror=""):
        return text.replace('version = "2.4.6"', 'version = "9.9.9"')

    monkeypatch.setattr("desktop.scripts.make_cn_lock.rewrite", drifting)

    status = main(
        [str(canonical), "--mirror", "https://mirror.example/", "--output", str(output)]
    )

    assert status == 1
    assert not output.exists(), "a lock that fails the gate must not be left behind"


def test_the_generator_needs_somewhere_to_point(
    tmp_path: Path, monkeypatch
) -> None:
    """With an empty table and no flags there is nothing to rewrite.

    The shipped table is populated, so this is reached by emptying it -- the
    state a fork with its own sources would be in.
    """

    empty = tmp_path / "sources.json"
    empty.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(empty))
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")

    assert main([str(canonical), "--output", str(tmp_path / "out.toml")]) == 2


def test_an_index_url_becomes_the_base_its_files_sit_under() -> None:
    """A mirror serves the simple index and the mirrored file tree at two
    different paths; the lock's URLs point into the second."""

    from desktop.scripts.make_cn_lock import _pypi_files_base

    assert (
        _pypi_files_base("https://pypi.tuna.tsinghua.edu.cn/simple")
        == "https://pypi.tuna.tsinghua.edu.cn/packages"
    )
    assert (
        _pypi_files_base("https://mirror.sjtu.edu.cn/pypi/web/simple/")
        == "https://mirror.sjtu.edu.cn/pypi/web/packages"
    )
    assert _pypi_files_base("") == ""


def test_the_generator_defaults_to_the_shipped_source_table(tmp_path) -> None:
    """The hosts a release uses belong in one reviewable file, not in whoever's
    shell history."""

    from finesub_bootstrap.download_sources import load_sources

    sources = load_sources()
    canonical = tmp_path / "pylock.toml"
    canonical.write_text(CANONICAL, encoding="utf-8")
    output = tmp_path / "pylock.cn.toml"

    assert main([str(canonical), "--output", str(output)]) == 0

    rewritten = output.read_text(encoding="utf-8")
    assert sources["torchMirror"].rstrip("/") in rewritten
    assert "pypi.tuna.tsinghua.edu.cn/packages" in rewritten


RUNTIME = Path(__file__).resolve().parents[2] / "runtime"
CANONICAL_LOCK = RUNTIME / "pylock.win-py312.toml"
REGIONAL_LOCK = RUNTIME / "pylock.win-py312.cn.toml"


def test_the_shipped_locks_describe_the_same_environment() -> None:
    """The gate the whole regional path rests on.

    Falling back between the two is only safe while they differ in nothing but
    the host, so this runs against the files that actually ship -- a
    regenerated lock that drifted would otherwise reach a user's machine.
    """

    assert REGIONAL_LOCK.is_file(), "the regional lock is expected to ship"
    assert_equivalent(CANONICAL_LOCK, REGIONAL_LOCK)


def test_the_shipped_regional_lock_points_somewhere_else() -> None:
    """Equivalence would also hold for a copy that rewrote nothing."""

    canonical = CANONICAL_LOCK.read_text(encoding="utf-8")
    regional = REGIONAL_LOCK.read_text(encoding="utf-8")

    assert "files.pythonhosted.org" not in regional
    assert "download-r2.pytorch.org" not in regional
    # The patched CT2 wheel has no verified mirror, so it keeps its own host --
    # partial acceleration rather than a guess.
    assert "github.com/caca2331/finesub" in regional
    assert canonical.count("url = ") == regional.count("url = ")
