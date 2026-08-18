from __future__ import annotations

from datetime import datetime
from pathlib import Path

from finesub_bootstrap.logs import log_directory, prune
from desktop.backend.launcher.session_log import SessionLog


def _only_session_log(user_data: Path) -> Path:
    logs = sorted(log_directory(user_data).glob("session-*.log"))
    assert len(logs) == 1
    return logs[0]


def test_a_run_records_how_it_started_and_how_it_ended(tmp_path: Path) -> None:
    session = SessionLog.open(tmp_path)
    session.write("version=1.2.3")
    session.finish("exited normally")

    body = _only_session_log(tmp_path).read_text(encoding="utf-8")

    assert "version=1.2.3" in body
    assert "--- exited normally ---" in body


def test_an_unhandled_error_leaves_its_traceback_behind(tmp_path: Path) -> None:
    # The whole point: an app that vanishes has to say why somewhere.
    session = SessionLog.open(tmp_path)
    try:
        raise RuntimeError("edgechromium is missing")
    except RuntimeError as error:
        session.exception("startup", error)
    session.finish("exited with an error")

    body = _only_session_log(tmp_path).read_text(encoding="utf-8")

    assert "startup raised RuntimeError: edgechromium is missing" in body
    assert "Traceback (most recent call last)" in body


def test_a_session_that_never_finishes_still_has_its_start(tmp_path: Path) -> None:
    # A killed process writes no ending. What it did write must survive, which
    # is why every line is flushed rather than buffered.
    session = SessionLog.open(tmp_path)
    session.write("starting")

    body = _only_session_log(tmp_path).read_text(encoding="utf-8")

    assert "starting" in body
    assert "---" not in body


def test_sessions_and_installs_share_one_budget_of_a_hundred(
    tmp_path: Path,
) -> None:
    # "Keep the newest hundred" is a promise about the folder. Two independent
    # hundreds would be two folders' worth of files under one name.
    directory = log_directory(tmp_path)
    directory.mkdir(parents=True)
    for index in range(60):
        path = directory / f"install-ffmpeg-2026010{index // 10}-0000{index % 10}.log"
        path.write_text("old", encoding="utf-8")
    for index in range(60):
        path = directory / f"session-2026020{index // 10}-0000{index % 10}.log"
        path.write_text("new", encoding="utf-8")

    prune(directory, keep=100)

    remaining = list(directory.glob("*.log"))
    assert len(remaining) == 100
    # Both kinds are eligible; nothing is protected by its prefix.
    assert any(path.name.startswith("session-") for path in remaining)


def test_an_unwritable_directory_does_not_stop_the_app(tmp_path: Path) -> None:
    blocked = tmp_path / "user-data"
    blocked.write_text("this is a file, not a directory", encoding="utf-8")

    session = SessionLog.open(blocked)
    session.write("starting")
    session.finish("exited normally")

    assert session.path is None


def test_the_file_is_named_for_when_the_run_began(tmp_path: Path) -> None:
    session = SessionLog.open(tmp_path, now=datetime(2026, 8, 8, 21, 30, 5))
    session.close()

    assert session.path is not None
    assert session.path.name == "session-20260808-213005.log"
