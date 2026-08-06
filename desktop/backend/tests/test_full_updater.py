from __future__ import annotations

from pathlib import Path
import json

import pytest

from desktop.backend.updater_main import (
    FullUpdateRequest,
    apply_full_update,
    main,
)


def test_default_preserved_list_keeps_the_installed_marker(
    tmp_path: Path,
) -> None:
    # The update service relies on this default: the marker decides where an
    # installed copy keeps personal data, and no update payload ships one, so
    # losing it during a full update would silently flip the install to
    # portable mode.
    request = FullUpdateRequest(
        source=str(tmp_path / ".update" / "source"),
        target=str(tmp_path),
        backup=str(tmp_path / ".update" / "backup"),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
    )

    assert "installed.marker" in request.preserved
    assert {"app", "user-data", "models", "runtime", "cache"} <= set(
        request.preserved
    )


def test_full_update_replaces_program_and_preserves_mutable_data(
    tmp_path: Path,
) -> None:
    target = tmp_path / "FineSub"
    source = target / ".update" / "source"
    backup = target / ".update" / "backup"
    source.mkdir(parents=True)
    (source / "FineSub Desktop.exe").write_bytes(b"new")
    (source / "desktop").mkdir()
    (source / "desktop" / "marker.txt").write_text("new", encoding="utf-8")
    new_app = source / "app" / "versions" / "2.0.0"
    (new_app / "src" / "asr_playground").mkdir(parents=True)
    (new_app / "src" / "asr_playground" / "pipeline.py").write_text("new", encoding="utf-8")
    (new_app / "desktop" / "backend" / "worker").mkdir(parents=True)
    (new_app / "desktop" / "backend" / "worker" / "main.py").write_text(
        "new",
        encoding="utf-8",
    )
    (new_app / "desktop" / "frontend" / "out").mkdir(parents=True)
    (new_app / "desktop" / "frontend" / "out" / "index.html").write_text(
        "new",
        encoding="utf-8",
    )
    (new_app / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (new_app / "app-manifest.json").write_text(
        '{"version":"2.0.0","platform":"windows-x64"}',
        encoding="utf-8",
    )
    (source / "app" / "current.json").write_text(
        '{"current":"2.0.0","previous":null,"pendingHealth":false}',
        encoding="utf-8",
    )
    (source / "user-data").mkdir()
    (source / "user-data" / "marker.txt").write_text(
        "must-not-overwrite",
        encoding="utf-8",
    )
    target.mkdir(exist_ok=True)
    (target / "FineSub Desktop.exe").write_bytes(b"old")
    for directory in ("user-data", "models", "runtime", "cache"):
        path = target / directory
        path.mkdir()
        (path / "marker.txt").write_text(directory, encoding="utf-8")
    old_app = target / "app" / "versions" / "1.0.0"
    old_app.mkdir(parents=True)
    (old_app / "marker.txt").write_text("old-app", encoding="utf-8")
    (target / "app" / "current.json").write_text(
        '{"current":"1.0.0","previous":null,"pendingHealth":false}',
        encoding="utf-8",
    )

    request = FullUpdateRequest(
        source=str(source),
        target=str(target),
        backup=str(backup),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
        preserved=["app", "user-data", "models", "runtime", "cache"],
    )

    apply_full_update(request, relaunch=False)

    assert (target / "FineSub Desktop.exe").read_bytes() == b"new"
    assert (target / "desktop" / "marker.txt").read_text("utf-8") == "new"
    assert (backup / "FineSub Desktop.exe").read_bytes() == b"old"
    app_pointer = json.loads(
        (target / "app" / "current.json").read_text(encoding="utf-8")
    )
    assert (target / "app" / "versions" / "1.0.0" / "marker.txt").is_file()
    assert (
        target
        / "app"
        / "versions"
        / "2.0.0"
        / "src"
        / "asr_playground"
        / "pipeline.py"
    ).is_file()
    assert app_pointer["current"] == "2.0.0"
    assert app_pointer["previous"] == "1.0.0"
    assert app_pointer["pendingHealth"] is True
    for directory in ("user-data", "models", "runtime", "cache"):
        assert (target / directory / "marker.txt").read_text("utf-8") == directory


def test_full_update_rejects_source_outside_application_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "FineSub"
    source = tmp_path / "outside"
    backup = target / ".update" / "backup"
    target.mkdir()
    source.mkdir()

    request = FullUpdateRequest(
        source=str(source),
        target=str(target),
        backup=str(backup),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
    )

    with pytest.raises(ValueError, match="source"):
        apply_full_update(request, relaunch=False)


def test_a_failed_update_records_the_reason_instead_of_blocking(
    tmp_path: Path,
) -> None:
    # The updater ships as a windowed build with no console, so an unhandled
    # exception becomes PyInstaller's modal traceback dialog -- which waits for
    # a click that never comes, because FineSub has already exited to let this
    # process replace it. Verified against the built exe: before this, a bad
    # request left the process alive; now it exits 1 and leaves the trace.
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "source": str(tmp_path / "missing-source"),
                "target": str(tmp_path / "missing-target"),
                "backup": str(tmp_path / "backup"),
                "parent_pid": 0,
                "relaunch_path": "FineSub Desktop.exe",
            }
        ),
        encoding="utf-8",
    )

    assert main(["--request", str(request_path)]) == 1

    recorded = request_path.with_suffix(".error.txt").read_text(encoding="utf-8")
    assert "missing-target" in recorded


def test_a_malformed_request_is_recorded_too(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{ not json", encoding="utf-8")

    assert main(["--request", str(request_path)]) == 1
    assert request_path.with_suffix(".error.txt").is_file()
