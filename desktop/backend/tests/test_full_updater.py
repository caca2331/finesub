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
    (new_app / "src" / "finesub").mkdir(parents=True)
    (new_app / "src" / "finesub" / "pipeline.py").write_text("new", encoding="utf-8")
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
        / "finesub"
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


def _minimal_full_update(tmp_path: Path) -> tuple[Path, Path, Path]:
    """An install root plus a `.update` payload, enough to exercise the swap."""
    target = tmp_path / "FineSub"
    source = target / ".update" / "source"
    backup = target / ".update" / "backup"
    source.mkdir(parents=True)
    (source / "FineSub Desktop.exe").write_bytes(b"new")
    (source / "extra").mkdir()
    (source / "extra" / "payload.txt").write_text("new", encoding="utf-8")
    target.mkdir(exist_ok=True)
    (target / "FineSub Desktop.exe").write_bytes(b"old")
    (target / "finesub.cmd").write_text("old shim", encoding="utf-8")
    return target, source, backup


def test_a_shutdown_mid_swap_still_puts_the_program_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows delivers KeyboardInterrupt on shutdown -- Exception missed it.

    The handler caught `Exception`, so a shutdown during the copy skipped the
    unwind entirely: the install root kept no executable and the only copy of
    the program stayed in `.update/backup-*`.
    """
    import shutil as shutil_module

    target, source, backup = _minimal_full_update(tmp_path)
    request = FullUpdateRequest(
        source=str(source),
        target=str(target),
        backup=str(backup),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
        preserved=[],
    )

    original = shutil_module.copy2

    def interrupt(source_path, destination, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(shutil_module, "copy2", interrupt)
    with pytest.raises(KeyboardInterrupt):
        apply_full_update(request, relaunch=False)
    monkeypatch.setattr(shutil_module, "copy2", original)

    assert (target / "FineSub Desktop.exe").read_bytes() == b"old"
    assert (target / "finesub.cmd").read_text("utf-8") == "old shim"


def test_an_old_services_shorter_preserved_list_cannot_eat_user_data(
    tmp_path: Path,
) -> None:
    """The request is serialized by whichever service asked for the update.
    Shipped 0.3.2 requests predate `tasks`/`locations.json` in the preserved
    list, and honoring them verbatim moved a user's finished subtitles into a
    backup that later gets discarded (0.4.0 release rehearsal). The updater's
    own list is a floor: requests extend it, never narrow it."""
    target, source, backup = _minimal_full_update(tmp_path)
    (target / "tasks").mkdir()
    (target / "tasks" / "finished.srt").write_text("subtitle", encoding="utf-8")
    (target / "locations.json").write_text("{}", encoding="utf-8")
    request = FullUpdateRequest(
        source=str(source),
        target=str(target),
        backup=str(backup),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
        # Exactly what a shipped 0.3.2 service serializes.
        preserved=[
            "app",
            "user-data",
            "models",
            "runtime",
            "cache",
            "installed.marker",
        ],
    )

    apply_full_update(request, relaunch=False)

    assert (target / "tasks" / "finished.srt").read_text("utf-8") == "subtitle"
    assert (target / "locations.json").read_text("utf-8") == "{}"
    assert (target / "FineSub Desktop.exe").read_bytes() == b"new"


def test_a_transient_lock_on_program_files_is_waited_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 0.4.0 rehearsal hit this live: the just-exited parent's DLLs were
    still unmapping when the updater started moving `_internal`, and a single
    failed rename aborted the whole update. The holders are transient --
    antivirus scans, image sections of the dead process -- so the move retries
    the way the runtime swap always has."""
    import desktop.backend.updater_main as updater_main

    target, source, backup = _minimal_full_update(tmp_path)
    request = FullUpdateRequest(
        source=str(source),
        target=str(target),
        backup=str(backup),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
        preserved=[],
    )

    real_rename = updater_main.os.rename
    denials = {"remaining": 2}

    def flaky_rename(source_path, destination_path):
        if denials["remaining"]:
            denials["remaining"] -= 1
            raise PermissionError(5, "Access is denied")
        return real_rename(source_path, destination_path)

    monkeypatch.setattr(updater_main.os, "rename", flaky_rename)
    monkeypatch.setattr(updater_main.time, "sleep", lambda _seconds: None)

    apply_full_update(request, relaunch=False)

    assert (target / "FineSub Desktop.exe").read_bytes() == b"new"
    assert denials["remaining"] == 0


def test_a_persistent_lock_leaves_every_tree_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shutil.move` used to fall back to copy-and-delete when the rename was
    refused, and the delete of a still-held tree stopped halfway: the rehearsal
    found a live install missing certifi's cacert.pem afterwards, which the
    restore path cannot see because the directory itself still exists. Rename
    either happens or it does not, so a lock that never clears must fail the
    update with the installation byte-for-byte intact."""
    import desktop.backend.updater_main as updater_main

    target, source, backup = _minimal_full_update(tmp_path)
    held = target / "_internal"
    held.mkdir()
    (held / "early.dll").write_text("early", encoding="utf-8")
    (held / "locked.dll").write_text("locked", encoding="utf-8")
    request = FullUpdateRequest(
        source=str(source),
        target=str(target),
        backup=str(backup),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
        preserved=[],
    )

    real_rename = updater_main.os.rename

    def refuse_held(source_path, destination_path):
        if Path(source_path).name == "_internal":
            raise PermissionError(5, "Access is denied")
        return real_rename(source_path, destination_path)

    monkeypatch.setattr(updater_main.os, "rename", refuse_held)
    monkeypatch.setattr(updater_main.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError):
        apply_full_update(request, relaunch=False)

    # The held tree was never half-deleted, and whatever had already been
    # moved to the backup came home.
    assert (held / "early.dll").read_text("utf-8") == "early"
    assert (held / "locked.dll").read_text("utf-8") == "locked"
    assert (target / "FineSub Desktop.exe").read_bytes() == b"old"
    assert (target / "finesub.cmd").read_text("utf-8") == "old shim"


def test_recovery_restores_an_install_left_without_an_executable(
    tmp_path: Path,
) -> None:
    from desktop.backend.updates.recovery import recover_interrupted_update

    target = tmp_path / "FineSub"
    backup = target / ".update" / "backup-2.0.0"
    backup.mkdir(parents=True)
    (backup / "FineSub Desktop.exe").write_bytes(b"old")
    (backup / "updater").mkdir()
    (backup / "updater" / "u.exe").write_bytes(b"old updater")

    message = recover_interrupted_update(target)

    assert message is not None and "上次更新未完成" in message
    assert (target / "FineSub Desktop.exe").read_bytes() == b"old"
    assert (target / "updater" / "u.exe").is_file()


def test_recovery_leaves_a_healthy_install_alone(tmp_path: Path) -> None:
    from desktop.backend.updates.recovery import recover_interrupted_update

    target = tmp_path / "FineSub"
    target.mkdir()
    (target / "FineSub Desktop.exe").write_bytes(b"current")
    backup = target / ".update" / "backup-2.0.0"
    backup.mkdir(parents=True)
    (backup / "FineSub Desktop.exe").write_bytes(b"old")

    assert recover_interrupted_update(target) is None
    assert (target / "FineSub Desktop.exe").read_bytes() == b"current"


def test_backups_are_never_discarded_while_the_install_is_broken(
    tmp_path: Path,
) -> None:
    """The old code cleared backups at the start of the next attempt."""
    from desktop.backend.updates.recovery import discard_backups

    target = tmp_path / "FineSub"
    backup = target / ".update" / "backup-2.0.0"
    backup.mkdir(parents=True)
    (backup / "FineSub Desktop.exe").write_bytes(b"the only copy")

    discard_backups(target)
    assert backup.is_dir(), "an unbootable root must keep its backup"

    (target / "FineSub Desktop.exe").write_bytes(b"restored")
    discard_backups(target)
    assert not backup.exists()


def test_an_incomplete_app_version_is_replaced_rather_than_adopted(
    tmp_path: Path,
) -> None:
    """The wreckage of an earlier failed copy used to be pointed at silently."""
    target, source, backup = _minimal_full_update(tmp_path)
    new_app = source / "app" / "versions" / "2.0.0"
    (new_app / "src" / "finesub").mkdir(parents=True)
    (new_app / "src" / "finesub" / "pipeline.py").write_text("new", encoding="utf-8")
    (new_app / "desktop" / "backend" / "worker").mkdir(parents=True)
    (new_app / "desktop" / "backend" / "worker" / "main.py").write_text("new", encoding="utf-8")
    (new_app / "desktop" / "frontend" / "out").mkdir(parents=True)
    (new_app / "desktop" / "frontend" / "out" / "index.html").write_text("new", encoding="utf-8")
    (new_app / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (new_app / "app-manifest.json").write_text(
        '{"version":"2.0.0","platform":"windows-x64"}', encoding="utf-8"
    )
    (source / "app" / "current.json").write_text(
        '{"current":"2.0.0","previous":null,"pendingHealth":false}', encoding="utf-8"
    )
    # Half a tree from an attempt that died inside copytree.
    stump = target / "app" / "versions" / "2.0.0" / "src" / "finesub"
    stump.mkdir(parents=True)
    (stump / "pipeline.py").write_text("half", encoding="utf-8")
    (target / "app" / "current.json").write_text(
        '{"current":"1.0.0","previous":null,"pendingHealth":false}', encoding="utf-8"
    )

    request = FullUpdateRequest(
        source=str(source),
        target=str(target),
        backup=str(backup),
        parent_pid=0,
        relaunch_path="FineSub Desktop.exe",
        preserved=["app"],
    )
    apply_full_update(request, relaunch=False)

    adopted = target / "app" / "versions" / "2.0.0"
    assert (adopted / "desktop" / "frontend" / "out" / "index.html").is_file()
    assert (adopted / "app-manifest.json").is_file()


def test_the_updater_waits_long_enough_for_a_running_task_to_finish() -> None:
    """The UI promises completion on exit with no deadline attached."""
    from desktop.backend.updater_main import PARENT_EXIT_TIMEOUT_SECONDS

    assert PARENT_EXIT_TIMEOUT_SECONDS >= 1800, (
        "two minutes killed the updater before a transcription could finish, "
        "and the install manager had already latched its terminal state"
    )


def test_an_updater_failure_report_is_read_once_and_archived(
    tmp_path: Path,
) -> None:
    from desktop.backend.updates.recovery import take_update_error_reports

    target = tmp_path / "FineSub"
    update_root = target / ".update"
    update_root.mkdir(parents=True)
    (update_root / "request-2.0.0.error.txt").write_text(
        "TimeoutError: Parent process 1234 did not exit", encoding="utf-8"
    )

    first = take_update_error_reports(target)
    second = take_update_error_reports(target)

    assert len(first) == 1 and "TimeoutError" in first[0]
    assert second == [], "a report must not be surfaced on every start"
    assert list(update_root.glob("*.seen"))
