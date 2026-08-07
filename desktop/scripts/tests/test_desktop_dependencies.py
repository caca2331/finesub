from __future__ import annotations

from pathlib import Path as _Path


def test_both_packagers_ship_every_source_package() -> None:
    """A new package under src/ must be a decision, not an omission.

    The wheel vendors a hand-written list while the desktop package copies the
    whole `src/` tree, so adding a package makes them disagree silently: the
    desktop would ship it and the CLI would not, and the failure would surface
    as an ImportError on a user's machine.
    """

    repo_root = _Path(__file__).resolve().parents[3]
    packages = {
        entry.name
        for entry in (repo_root / "src").iterdir()
        if entry.is_dir() and (entry / "__init__.py").is_file()
    }
    wheel_script = (
        repo_root / "cli" / "scripts" / "build-wheel.ps1"
    ).read_text(encoding="utf-8")
    vendored_line = next(
        line for line in wheel_script.splitlines() if "foreach ($Package in @(" in line
    )

    missing = [name for name in packages if f'"{name}"' not in vendored_line]

    assert missing == [], (
        f"{missing} live under src/ but the CLI wheel does not vendor them; "
        "add them to build-wheel.ps1 or exclude them deliberately"
    )

import json
from pathlib import Path
import re
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_desktop_extra_declares_every_direct_python_dependency() -> None:
    document = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = document["project"]["optional-dependencies"]["desktop"]
    names = {
        re.split(r"[<>=!~ ;\[]", dependency, maxsplit=1)[0].lower()
        for dependency in dependencies
    }

    assert names == {
        "cryptography",
        "httpx",
        "packaging",
        "pillow",
        "pydantic",
        "pystray",
        "pywebview",
    }

    development = document["project"]["optional-dependencies"]["dev"]
    development_names = {
        re.split(r"[<>=!~ ;\[]", dependency, maxsplit=1)[0].lower()
        for dependency in development
    }
    assert {"pillow", "pyinstaller", "pyinstaller-hooks-contrib"} <= (
        development_names
    )


def test_windows_ai_runtime_lock_matches_the_pipeline_extras() -> None:
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lock_path = (
        REPOSITORY_ROOT / "desktop" / "runtime" / "pylock.win-py312.toml"
    )
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    packages = {package["name"]: package["version"] for package in lock["packages"]}

    # Transitive deps of audio-separator, so the extras below never name them --
    # but the worker imports them directly and has shipped without them before.
    assert "beartype" in packages
    assert "ml-collections" in packages

    # The lock is generated from [asr]+[harness]+[desktop-worker], so every
    # exact pin on those extras has to survive into it. This drifted once and
    # went unnoticed for a month: the lock was compiled while [asr] still used
    # whisper-timestamped, and after the fw-refine migration it contained no
    # decoder at all -- installable, and unable to transcribe a thing.
    requirements = {
        Requirement(raw).name.lower(): (extra, Requirement(raw))
        for extra in ("asr", "harness", "desktop-worker")
        for raw in project["project"]["optional-dependencies"][extra]
    }
    for name, (extra, requirement) in requirements.items():
        assert name in packages, (
            f"{name} is required by [{extra}] but is missing from "
            f"desktop/runtime/pylock.win-py312.toml. Regenerate the lock with "
            f"the command in its header."
        )
        if not requirement.specifier:
            continue
        # Locked versions carry local labels ("2.11.0+cu128") that the extras'
        # specifiers do not spell out; PEP 440 matches those, so compare whole.
        assert Version(packages[name]) in requirement.specifier, (
            f"{name}: [{extra}] asks for {requirement.specifier} but the "
            f"desktop lock pins {packages[name]}. Regenerate the lock."
        )

    # Stock CTranslate2 satisfies ctranslate2==4.8.1 -- PEP 440 local labels are
    # not an exclusion mechanism -- so the pin above cannot catch this on its
    # own. [desktop-worker] carries a direct reference for that reason, and the
    # patched build is what fw-refine needs at runtime.
    assert "wtrefine" in packages["ctranslate2"]


def test_the_cli_shell_and_the_desktop_manifest_pin_the_same_uv() -> None:
    # Both products build the multi-gigabyte runtime from the same lock; a
    # different resolver version on either side would make "same lock, same
    # environment" a hope instead of a guarantee.
    manifest = json.loads(
        (
            REPOSITORY_ROOT / "desktop" / "resources" / "runtime-manifest.json"
        ).read_text(encoding="utf-8")
    )
    manifest_uv = next(
        resource["version"]
        for resource in manifest["resources"]
        if resource["id"] == "uv"
    )
    shell = tomllib.loads(
        (REPOSITORY_ROOT / "cli" / "pyproject.toml").read_text(encoding="utf-8")
    )
    uv_requirements = [
        Requirement(dependency)
        for dependency in shell["project"]["dependencies"]
        if Requirement(dependency).name == "uv"
    ]
    assert len(uv_requirements) == 1
    assert str(uv_requirements[0].specifier) == f"=={manifest_uv}"


def test_the_cli_shell_exposes_only_the_launcher_entry_point() -> None:
    # The shell venv has no torch: any pipeline entry point on PATH would be a
    # command that always crashes with ImportError. Everything goes through
    # the `finesub` launcher, which re-executes inside the managed runtime.
    shell = tomllib.loads(
        (REPOSITORY_ROOT / "cli" / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert set(shell["project"]["scripts"]) == {"finesub"}


def test_the_cli_and_the_desktop_app_ship_one_version_number() -> None:
    # One version, one tag, one GitHub Release. The updater resolves a release
    # by `v{manifest.version}`, so a desktop version that drifts from the CLI's
    # would either collide with a tag that carries no desktop assets or point at
    # one that does not exist. v0.3.0 was published as a CLI-only release before
    # this contract existed, which is why the joint line restarts at 0.3.1.
    project = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    desktop_version = (
        (REPOSITORY_ROOT / "desktop" / "VERSION").read_text(encoding="utf-8").strip()
    )
    frontend = json.loads(
        (REPOSITORY_ROOT / "desktop" / "frontend" / "package.json").read_text(
            encoding="utf-8"
        )
    )
    launcher = json.loads(
        (REPOSITORY_ROOT / "desktop" / "resources" / "launcher.json").read_text(
            encoding="utf-8"
        )
    )
    installer = (
        REPOSITORY_ROOT / "desktop" / "installer" / "FineSubDesktop.iss"
    ).read_text(encoding="utf-8")

    expected = Version(project["project"]["version"])

    assert Version(desktop_version) == expected
    assert Version(frontend["version"]) == expected
    assert Version(launcher["appVersion"]) == expected
    assert Version(launcher["launcherVersion"]) == expected
    assert f'#define AppVersion "{expected}"' in installer


def test_release_defaults_do_not_promise_deltas_from_unreleased_versions() -> None:
    # The old defaults offered an app-only delta to installs of 0.2.3, a version
    # that never shipped a signed manifest -- so the delta could never apply and
    # the value was pure staleness. An empty -SupportedFrom means "full package
    # for everyone", which is the correct direction to fail.
    script = (
        REPOSITORY_ROOT / "desktop" / "scripts" / "build-release.ps1"
    ).read_text(encoding="utf-8")
    version = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]

    assert "[string[]]$SupportedFrom = @()" in script
    for name in ("MinimumLauncherVersion", "MinimumSupportedVersion"):
        match = re.search(rf'\[string\]\${name} = "([^"]+)"', script)
        assert match is not None, f"{name} default not found in build-release.ps1"
        assert Version(match.group(1)) <= Version(version), (
            f"{name} default {match.group(1)} is newer than the version being "
            f"built ({version}); no install could ever satisfy it."
        )
