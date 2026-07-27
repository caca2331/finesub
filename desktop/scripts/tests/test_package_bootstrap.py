from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "desktop" / "scripts" / "package-bootstrap.ps1"


def test_package_bootstrap_excludes_tests_and_keeps_runtime_sources() -> None:
    work = REPO_ROOT / "dist" / f"package-bootstrap-test-{os.getpid()}"
    fixture_repo = work / "repo"
    output = work / "output"
    launcher_dist = output / "FineSub Desktop.dist"
    updater_dist = output / "FineSub Desktop Updater.dist"
    try:
        (fixture_repo / "src").mkdir(parents=True)
        (fixture_repo / "src" / "pipeline.py").write_text("PIPELINE = True\n", "utf-8")
        (fixture_repo / "desktop" / "backend" / "launcher").mkdir(parents=True)
        (fixture_repo / "desktop" / "backend" / "launcher" / "main.py").write_text(
            "MAIN = True\n", "utf-8"
        )
        (fixture_repo / "desktop" / "backend" / "tests").mkdir(parents=True)
        (fixture_repo / "desktop" / "backend" / "tests" / "must_not_ship.py").write_text(
            "raise RuntimeError\n", "utf-8"
        )
        (fixture_repo / "desktop" / "backend" / "__pycache__").mkdir(parents=True)
        (fixture_repo / "desktop" / "backend" / "__pycache__" / "cache.pyc").write_bytes(
            b"cache"
        )
        (fixture_repo / "desktop" / "resources").mkdir(parents=True)
        (fixture_repo / "desktop" / "resources" / "manifest.json").write_text(
            "{}\n", "utf-8"
        )
        (fixture_repo / "desktop" / "frontend" / "out").mkdir(parents=True)
        (fixture_repo / "desktop" / "frontend" / "out" / "index.html").write_text(
            "<main>FineSub</main>\n", "utf-8"
        )
        (fixture_repo / "desktop" / "__init__.py").write_text("", "utf-8")
        (fixture_repo / "pyproject.toml").write_text("[project]\nname='fixture'\n", "utf-8")

        launcher_config = fixture_repo / "launcher.json"
        launcher_config.write_text(
            json.dumps(
                {
                    "appVersion": "0.0.0",
                    "launcherVersion": "0.0.0",
                    "channel": "stable",
                }
            ),
            "utf-8",
        )
        trusted_keys = fixture_repo / "trusted-update-keys.json"
        trusted_keys.write_text('{"keys":[]}\n', "utf-8")

        launcher_dist.mkdir(parents=True)
        (launcher_dist / "FineSub Desktop.exe").write_bytes(b"launcher")
        updater_dist.mkdir(parents=True)
        (updater_dist / "FineSub Desktop Updater.exe").write_bytes(b"updater")

        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-RepoRoot",
                str(fixture_repo),
                "-OutputDirectory",
                str(output),
                "-Version",
                "2.3.4",
                "-LauncherConfigPath",
                str(launcher_config),
                "-TrustedKeysPath",
                str(trusted_keys),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stderr

        version_root = launcher_dist / "app" / "versions" / "2.3.4"
        assert (version_root / "src" / "pipeline.py").is_file()
        assert (version_root / "desktop" / "backend" / "launcher" / "main.py").is_file()
        assert not (version_root / "desktop" / "backend" / "tests").exists()
        assert not (version_root / "desktop" / "backend" / "__pycache__").exists()
        assert (
            launcher_dist / "updater" / "FineSub Desktop Updater.exe"
        ).is_file()
        assert (launcher_dist / "FineSub.exe").read_bytes() == b"launcher"
        assert (
            launcher_dist / "updater" / "FineSubUpdater.exe"
        ).read_bytes() == b"updater"
        pointer = json.loads((launcher_dist / "app" / "current.json").read_text("utf-8-sig"))
        assert pointer["current"] == "2.3.4"
        assert not (launcher_dist / "app" / "current.json").read_bytes().startswith(
            b"\xef\xbb\xbf"
        )
        config = json.loads((launcher_dist / "launcher.json").read_text("utf-8-sig"))
        assert config["appVersion"] == "2.3.4"
        assert config["launcherVersion"] == "2.3.4"
        assert not (launcher_dist / "launcher.json").read_bytes().startswith(
            b"\xef\xbb\xbf"
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
