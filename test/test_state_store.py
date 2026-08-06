from __future__ import annotations

import concurrent.futures as cf
import json
import os
from pathlib import Path

from asr_playground import state as state_store


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


def test_unreadable_state_is_reported_not_swallowed(tmp_path, capsys) -> None:
    path = tmp_path / ".state"
    path.write_text("{ truncated", encoding="utf-8")

    assert state_store.read_section("mine", path) == {}
    assert "discarding unreadable state" in capsys.readouterr().err


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
    monkeypatch.setenv("FINESUB_STATE_DIR", str(tmp_path / "elsewhere"))
    with state_store.state_section("mine") as section:
        section["value"] = 1
    assert (tmp_path / "elsewhere").is_file()
    assert os.path.exists(str(tmp_path / "elsewhere") + ".lock")
