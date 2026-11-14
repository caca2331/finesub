"""The quick and full checks compare a runtime with its lock, file by file."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from finesub_bootstrap import runtime_health
from finesub_bootstrap.runtime_health import check_runtime


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LOCKS = sorted((REPOSITORY_ROOT / "src" / "finesub_bootstrap").glob("pylock.*.toml"))

#: Every environment marker the locks may carry: each is true on the platform
#: the locks target (64-bit CPython on Windows). The quick check does not
#: evaluate markers -- it is stdlib-only -- so a marker false on that platform
#: would read as a package gone missing, and the rebuild could never install
#: it: every run would stop at the runtime check.
MARKERS_TRUE_ON_THE_TARGET = {
    "sys_platform == 'win32'",
    "implementation_name != 'PyPy'",
    "platform_machine == 'AMD64' or platform_machine == 'aarch64' or "
    "platform_machine == 'amd64' or platform_machine == 'arm64' or "
    "platform_machine == 'x86_64'",
}


def _digest(content: bytes) -> str:
    raw = hashlib.sha256(content).digest()
    return "sha256=" + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _install(site_packages: Path, name: str, version: str, files: dict[str, bytes]) -> Path:
    """A distribution as uv leaves it: its files plus a RECORD listing them."""

    dist_info = site_packages / f"{name.replace('-', '_')}-{version}.dist-info"
    dist_info.mkdir(parents=True)
    rows = []
    for relative, content in files.items():
        target = site_packages / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        rows.append(f'"{relative}",{_digest(content)},{len(content)}')
    # The RECORD's own row carries no digest; a console-script launcher lives
    # outside site-packages. Neither is something the check can hold.
    rows.append(f"{dist_info.name}/RECORD,,")
    rows.append(f"../../Scripts/{name}.exe,{_digest(b'launcher')},8")
    (dist_info / "RECORD").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return dist_info


def _lock(path: Path, packages: dict[str, str]) -> Path:
    entries = "".join(
        f'\n[[packages]]\nname = "{name}"\nversion = "{version}"\n'
        for name, version in packages.items()
    )
    path.write_text('lock-version = "1.0"\n' + entries, encoding="utf-8")
    return path


@pytest.fixture
def runtime(tmp_path: Path) -> tuple[Path, Path]:
    site_packages = tmp_path / "Lib" / "site-packages"
    _install(site_packages, "silero-vad", "6.2.1", {
        "silero_vad/__init__.py": b"",
        "silero_vad/data/silero_vad.jit": b"torchscript archive",
    })
    _install(site_packages, "tiny", "1.0", {"tiny, with comma.py": b"x = 1\n"})
    lock = _lock(tmp_path / "pylock.toml", {"silero-vad": "6.2.1", "tiny": "1.0"})
    return site_packages, lock


def test_an_untouched_runtime_matches_its_lock(runtime: tuple[Path, Path]) -> None:
    site_packages, lock = runtime

    report = check_runtime(site_packages, lock, verify_hashes=True)

    assert report is not None and not report.mismatched
    assert report.extras == []


def test_a_quarantined_file_is_found_by_the_quick_check(runtime: tuple[Path, Path]) -> None:
    site_packages, lock = runtime
    (site_packages / "silero_vad" / "data" / "silero_vad.jit").unlink()

    report = check_runtime(site_packages, lock)

    assert report.missing_files == {"silero-vad": ["silero_vad/data/silero_vad.jit"]}
    assert report.packages_to_reinstall() == ["silero-vad"]
    assert "silero_vad.jit" in report.problems()[0]


def test_shipped_bytecode_that_python_rewrote_is_not_damage(tmp_path: Path) -> None:
    """The patched CTranslate2 wheel lists a `.pyc` with a digest; the first
    import rewrites it because the installed source has a new mtime."""

    site_packages = tmp_path / "Lib" / "site-packages"
    _install(site_packages, "ctranslate2", "4.8.1", {
        "ctranslate2/__init__.py": b"",
        "ctranslate2/__pycache__/__init__.cpython-312.pyc": b"compiled at build time",
    })
    lock = _lock(tmp_path / "pylock.toml", {"ctranslate2": "4.8.1"})
    (site_packages / "ctranslate2" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"recompiled")

    assert not check_runtime(site_packages, lock, verify_hashes=True).mismatched


def test_changed_contents_need_the_full_check(runtime: tuple[Path, Path]) -> None:
    site_packages, lock = runtime
    (site_packages / "tiny, with comma.py").write_bytes(b"x = 2\n")

    assert not check_runtime(site_packages, lock).mismatched
    report = check_runtime(site_packages, lock, verify_hashes=True)
    assert report.hash_mismatches == {"tiny": ["tiny, with comma.py"]}


def test_a_removed_or_swapped_package_is_a_mismatch(tmp_path: Path, runtime: tuple[Path, Path]) -> None:
    site_packages, _lock_path = runtime
    lock = _lock(tmp_path / "other.toml", {"silero-vad": "6.3.0", "tiny": "1.0", "gone": "2.0"})

    report = check_runtime(site_packages, lock)

    assert report.version_mismatches == [("silero-vad", "6.2.1", "6.3.0")]
    assert report.missing_packages == ["gone"]


def test_two_installs_of_one_package_are_a_mismatch(runtime: tuple[Path, Path]) -> None:
    # Audit O-11: keyed by name, the check read whichever `.dist-info` the glob
    # listed last -- healthy if that was the locked one, a needless rebuild if not.
    site_packages, lock = runtime
    _install(site_packages, "silero-vad", "6.3.0", {"silero_vad/new.py": b""})

    report = check_runtime(site_packages, lock)

    assert report.mismatched
    assert report.duplicate_distributions == {"silero-vad": ["6.2.1", "6.3.0"]}
    assert report.version_mismatches == []
    assert report.packages_to_reinstall() == ["silero-vad"]
    assert any("同一个包装了多份" in line for line in report.problems())


def test_a_package_the_lock_does_not_know_is_only_listed(tmp_path: Path, runtime: tuple[Path, Path]) -> None:
    site_packages, lock = runtime
    _install(site_packages, "added-by-hand", "0.1", {"added_by_hand.py": b""})

    report = check_runtime(site_packages, lock)

    assert not report.mismatched
    assert report.extras == [("added-by-hand", "0.1")]
    assert report.losses(include_versions=True) == ["added-by-hand 0.1"]


def test_an_upgrade_does_not_list_upgraded_dependencies_as_losses(tmp_path: Path, runtime: tuple[Path, Path]) -> None:
    site_packages, _lock_path = runtime
    newer = _lock(tmp_path / "newer.toml", {"silero-vad": "6.3.0", "tiny": "1.0"})

    report = check_runtime(site_packages, newer)

    assert report.losses(include_versions=False) == []
    assert report.losses(include_versions=True) == ["silero-vad 6.2.1"]


def test_every_lock_marker_is_true_on_the_target_platform() -> None:
    parser = runtime_health.toml_parser()
    assert parser is not None
    assert LOCKS, "no lock found to check"
    for lock in LOCKS:
        with lock.open("rb") as handle:
            markers = {entry["marker"] for entry in parser(handle)["packages"] if "marker" in entry}
        assert markers <= MARKERS_TRUE_ON_THE_TARGET, (lock.name, markers - MARKERS_TRUE_ON_THE_TARGET)


def test_the_shipped_lock_reads_as_package_versions() -> None:
    lock = REPOSITORY_ROOT / "src" / "finesub_bootstrap" / "pylock.win-py312.toml"
    locked = runtime_health.locked_packages(lock)

    assert locked is not None
    with lock.open("rb") as handle:
        entries = runtime_health.toml_parser()(handle)["packages"]
    # An entry without a version would be skipped silently -- neither checked
    # nor kept, and listed as a loss on every rebuild.
    assert len(locked) == len(entries)
    assert runtime_health.REQUIRED_CTRANSLATE2_LOCAL_LABEL in locked["ctranslate2"]
    assert "silero-vad" in locked


def test_without_a_toml_parser_the_check_does_not_run(runtime: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    site_packages, lock = runtime
    monkeypatch.setattr(runtime_health, "toml_parser", lambda: None)

    assert check_runtime(site_packages, lock) is None


# ---- settings and when to check ---------------------------------------------


def test_settings_default_to_checking_and_repairing(tmp_path: Path) -> None:
    settings = runtime_health.read_runtime_settings(tmp_path)

    assert (settings.integrity_check, settings.auto_repair, settings.problems) == ("auto", True, ())


def test_settings_are_read_from_the_runtime_table(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[runtime]\nintegrity_check = "off"\nauto_repair = false\n', encoding="utf-8"
    )

    settings = runtime_health.read_runtime_settings(tmp_path)

    assert (settings.integrity_check, settings.auto_repair) == ("off", False)


def test_unknown_values_fall_back_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        '[runtime]\nintegrity_check = "sometimes"\nauto_repair = "yes"\n', encoding="utf-8"
    )

    settings = runtime_health.read_runtime_settings(tmp_path)

    assert (settings.integrity_check, settings.auto_repair) == ("auto", True)
    assert len(settings.problems) == 2


def test_an_unreadable_config_does_not_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a parser the user may have written `auto_repair = false`."""

    from finesub_bootstrap import user_config

    (tmp_path / "config.toml").write_text("[runtime]\nauto_repair = false\n", encoding="utf-8")
    monkeypatch.setattr(user_config, "toml_parser", lambda: None)

    assert runtime_health.read_runtime_settings(tmp_path).auto_repair is False


def test_auto_checks_until_this_version_has_run_once(tmp_path: Path) -> None:
    root = tmp_path / "python"
    root.mkdir()

    assert runtime_health.should_check_before_run("auto", root, "0.5.2")
    runtime_health.record_success(root, "0.5.2")
    assert not runtime_health.should_check_before_run("auto", root, "0.5.2")
    assert runtime_health.should_check_before_run("auto", root, "0.5.3")
    assert runtime_health.should_check_before_run("on", root, "0.5.2")
    assert not runtime_health.should_check_before_run("off", root, "0.5.3")


def test_without_a_version_every_run_is_checked(tmp_path: Path) -> None:
    root = tmp_path / "python"
    root.mkdir()

    runtime_health.record_success(root, "")

    assert not (root / runtime_health.STATE_FILE).exists()
    assert runtime_health.should_check_before_run("auto", root, "")


# ---- non-ASCII paths ---------------------------------------------------------


def test_a_non_ascii_path_on_a_legacy_code_page_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_health, "ansi_code_page", lambda: 936)

    problem = runtime_health.non_ascii_path_problem([Path("C:/Users/张三/AppData/Local/FineSub")])

    assert problem is not None and "张三" in problem and "FINESUB_HOME" in problem
    assert runtime_health.non_ascii_path_problem([Path("C:/FineSub")]) is None


def test_a_utf8_code_page_needs_no_explanation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime_health, "ansi_code_page", lambda: 65001)

    assert runtime_health.non_ascii_path_problem([Path("C:/Users/张三/FineSub")]) is None


def test_the_repair_command_carries_a_custom_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINESUB_HOME", raising=False)
    assert runtime_health.repair_command() == "finesub setup --repair"

    monkeypatch.setenv("FINESUB_HOME", r"C:\Users\x\AppData\Roaming\Nonoka X\finesub")
    assert runtime_health.repair_command() == (
        r"$env:FINESUB_HOME='C:\Users\x\AppData\Roaming\Nonoka X\finesub'; finesub setup --repair"
    )


@pytest.mark.parametrize(
    ("installed", "locked", "same"),
    [
        ("1.0", "1.0.0", True),
        ("2024.01", "2024.1", True),
        ("2.11.0+cu128", "2.11+cu128", True),
        ("4.8.1+finesub0.4.0.cu128", "4.8.1+finesub0.4.0.cu128", True),
        ("2.4.0", "2.5.0", False),
        ("1.0rc1", "1.0", False),
        ("2.11.0+cu128", "2.11.0", False),
    ],
)
def test_versions_compare_the_way_pep_440_does(installed: str, locked: str, same: bool) -> None:
    assert runtime_health._same_version(installed, locked) is same


def test_a_package_whose_record_is_gone_is_damaged(runtime: tuple[Path, Path]) -> None:
    """Without its RECORD nothing about a package can be vouched for."""

    site_packages, lock = runtime
    (site_packages / "silero_vad-6.2.1.dist-info" / "RECORD").unlink()

    report = check_runtime(site_packages, lock, verify_hashes=True)

    assert report.missing_files == {"silero-vad": ["silero_vad-6.2.1.dist-info/RECORD"]}
    assert report.packages_to_reinstall() == ["silero-vad"]
