from __future__ import annotations

import concurrent.futures as cf
import json
import os
from pathlib import Path

from finesub import state as state_store


def test_a_section_write_preserves_the_other_subsystems(tmp_path) -> None:
    path = tmp_path / ".state"
    path.write_text(json.dumps({"other": {"keep": 1}}), encoding="utf-8")

    with state_store.state_section("mine", path) as section:
        section["value"] = 2

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["other"] == {"keep": 1}
    assert document["mine"] == {"value": 2}


def test_concurrent_writers_do_not_lose_each_others_sections(tmp_path) -> None:
    path = tmp_path / ".state"

    def write(index: int) -> None:
        # Every writer rewrites the whole document, so without the lock the last
        # one out would drop whatever landed after it read.
        for _ in range(20):
            with state_store.state_section(f"ns{index}", path) as section:
                section["count"] = section.get("count", 0) + 1

    with cf.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(4)))

    document = json.loads(path.read_text(encoding="utf-8"))
    assert {key: value["count"] for key, value in document.items()} == {
        f"ns{index}": 20 for index in range(4)
    }


def test_unreadable_state_is_reported_not_swallowed(tmp_path) -> None:
    import io

    from finesub.reporting import TerminalReporter, reporting_to

    path = tmp_path / ".state"
    path.write_text("{ truncated", encoding="utf-8")
    stream = io.StringIO()

    with reporting_to(TerminalReporter(stream, isatty=False)):
        assert state_store.read_section("mine", path) == {}

    assert "discarding unreadable state" in stream.getvalue()


def test_a_corrupt_document_does_not_block_new_writes(tmp_path) -> None:
    path = tmp_path / ".state"
    path.write_text("{ truncated", encoding="utf-8")

    with state_store.state_section("mine", path) as section:
        section["value"] = 1

    assert json.loads(path.read_text(encoding="utf-8"))["mine"] == {"value": 1}


def test_no_partial_file_is_left_where_readers_look(tmp_path) -> None:
    path = tmp_path / ".state"
    with state_store.state_section("mine", path) as section:
        section["value"] = "x" * 100_000

    # The document is swapped in by rename, so the temporary never survives and
    # a reader never observes a half-written state file.
    assert not (tmp_path / ".state.tmp").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["mine"]["value"]


def test_lock_file_is_separate_from_the_state_file(tmp_path) -> None:
    path = tmp_path / ".state"
    with state_store.state_section("mine", path) as section:
        section["value"] = 1

    # Holding the lock on the state file itself would not survive the atomic
    # replace, which swaps the inode out from under it.
    assert (tmp_path / ".state.lock").exists()


def test_read_section_on_a_missing_file_is_empty(tmp_path) -> None:
    assert state_store.read_section("mine", tmp_path / "absent") == {}


def test_sections_of_the_wrong_shape_read_as_empty(tmp_path) -> None:
    path = tmp_path / ".state"
    path.write_text(json.dumps({"mine": ["not", "a", "mapping"]}), encoding="utf-8")

    assert state_store.read_section("mine", path) == {}
    with state_store.state_section("mine", path) as section:
        assert section == {}
        section["value"] = 1
    assert json.loads(path.read_text(encoding="utf-8"))["mine"] == {"value": 1}


def test_state_file_path_defaults_to_the_checkout(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FINESUB_STATE_FILE", str(tmp_path / "elsewhere"))
    with state_store.state_section("mine") as section:
        section["value"] = 1
    assert (tmp_path / "elsewhere").is_file()
    assert os.path.exists(str(tmp_path / "elsewhere") + ".lock")


def _replace_failing_once(monkeypatch, target: Path) -> list[str]:
    """`os.replace` onto `target` that fails once, the way Windows does while a reader holds it.

    `fsops.os` is the `os` module itself, so the patch is process-wide: a thread another test left running in
    this worker may rename its own files meanwhile. Only renames onto `target` are counted or failed.
    """

    from finesub_bootstrap import fsops

    real = os.replace
    calls: list[str] = []

    def replace(source, destination):
        if Path(destination) != target:
            return real(source, destination)
        calls.append(str(destination))
        if len(calls) == 1:
            raise PermissionError(5, "Access is denied")
        real(source, destination)

    monkeypatch.setattr(fsops.os, "replace", replace)
    return calls


def test_a_write_waits_out_a_reader_holding_the_state(tmp_path, monkeypatch) -> None:
    """A lock-free `read_section` elsewhere used to cost the write: one
    `state-write-failed` line and that quota or cooldown update gone."""

    path = tmp_path / ".state"
    calls = _replace_failing_once(monkeypatch, path)

    with state_store.state_section("mine", path) as section:
        section["value"] = 1

    assert len(calls) == 2
    assert json.loads(path.read_text(encoding="utf-8")) == {"mine": {"value": 1}}



def test_a_writer_waits_for_the_lock_however_long_it_is_held(tmp_path, monkeypatch) -> None:
    # Audit O-14: `msvcrt.locking(LK_LOCK)` gave up after ten one-second tries
    # and raised out of the limiter that called it. The state lock is now the
    # same polling lock every other subsystem takes, which waits on both
    # platforms: with a fast poll, thirty tries in it is still waiting.
    import threading

    from finesub_bootstrap import locks

    monkeypatch.setattr(locks, "POLL_SECONDS", 0.01)
    path = tmp_path / ".state"
    done = threading.Event()

    def write() -> None:
        with state_store.state_section("mine", path) as section:
            section["value"] = 1
        done.set()

    with locks.holding_lock(path.with_name(path.name + ".lock")):
        writer = threading.Thread(target=write)
        writer.start()
        assert not done.wait(0.3)
    writer.join(5)
    assert done.is_set()
    assert state_store.read_section("mine", path) == {"value": 1}
