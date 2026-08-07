from __future__ import annotations

import json
from pathlib import Path

from finesub_bootstrap import migrations
from finesub_bootstrap.migrations import Migration, apply_pending, applied_ids
from finesub_bootstrap.migrations.knowledge_location import MIGRATION_ID
from finesub_bootstrap.paths import AppPaths


def _knowledge(directory: Path, marker: str) -> Path:
    (directory / "streamer").mkdir(parents=True)
    (directory / "streamer" / "index.md").write_text(marker, encoding="utf-8")
    (directory / ".git").mkdir()
    return directory


def test_a_knowledge_base_left_in_the_app_directory_is_rescued(
    tmp_path: Path,
) -> None:
    # app/ is not on the updater's preserved list, so a knowledge base sitting
    # there is one update away from being deleted without a word.
    paths = AppPaths.for_root(tmp_path / "root")
    stray = _knowledge(paths.app_versions / "0.3.2" / "knowledge", "rescue me")
    messages: list[str] = []

    assert MIGRATION_ID in apply_pending(paths, log=messages.append)

    destination = paths.user_data / "knowledge"
    assert (destination / "streamer" / "index.md").read_text("utf-8") == "rescue me"
    assert (destination / ".git").is_dir()
    assert not stray.exists()
    assert any("迁移" in message for message in messages)

    # Recorded, so the next start does not look again.
    assert MIGRATION_ID in applied_ids(paths)
    assert apply_pending(paths, log=messages.append) == []


def test_two_knowledge_bases_are_never_merged_and_keep_warning(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    stray = _knowledge(paths.app_versions / "0.3.2" / "knowledge", "stray")
    destination = _knowledge(paths.user_data / "knowledge", "in use")
    messages: list[str] = []

    assert MIGRATION_ID not in apply_pending(paths, log=messages.append)

    assert (stray / "streamer" / "index.md").read_text("utf-8") == "stray"
    assert (destination / "streamer" / "index.md").read_text("utf-8") == "in use"
    assert any("未自动合并" in message for message in messages)
    # Deliberately not recorded: the warning has to keep coming back until a
    # human resolves it, because the stray copy is the one that gets deleted.
    assert MIGRATION_ID not in applied_ids(paths)


def test_an_install_without_an_app_directory_is_recorded_as_done(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root", data_root=tmp_path / "data")

    assert MIGRATION_ID in apply_pending(paths)
    assert not (paths.user_data / "knowledge").exists()


def test_a_failing_migration_is_retried_rather_than_recorded(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")

    def explode(_paths, _log):
        raise RuntimeError("disk on fire")

    broken = (Migration(id="0002-broken", run=explode),)
    messages: list[str] = []

    assert apply_pending(paths, log=messages.append, migrations=broken) == []
    assert applied_ids(paths) == set()
    assert any("disk on fire" in message for message in messages)


def test_migrations_stand_aside_while_another_process_holds_the_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Both front ends run migrations at start-up, so a desktop app that is
    # already open and a CLI command started beside it would otherwise move the
    # same trees at once.
    paths = AppPaths.for_root(tmp_path / "root", data_root=tmp_path / "data")
    _knowledge(paths.app_versions / "0.3.2" / "knowledge", "rescue me")
    messages: list[str] = []

    from finesub_bootstrap.locks import holding_lock

    monkeypatch.setattr(migrations, "LOCK_TIMEOUT_SECONDS", 0.1)
    # Anchored on the data root, which is the only path a desktop install and a
    # CLI agree on -- their install roots are different directories, so a lock
    # there would let exactly the case this guards against through.
    other_front_end = AppPaths.for_root(
        tmp_path / "another-install", data_root=paths.data_root
    )
    with holding_lock(other_front_end.data_root / migrations.LOCK_NAME):
        assert apply_pending(paths, log=messages.append) == []

    assert applied_ids(paths) == set()
    # Nothing moved, and the next start will try again.
    assert (paths.app_versions / "0.3.2" / "knowledge").is_dir()
    assert MIGRATION_ID in apply_pending(paths, log=messages.append)


def test_a_crash_during_the_rescue_leaves_the_source_intact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Cross-volume moves are copy-then-delete; interrupted, they used to leave
    # a half-copy that sent the next start into "two knowledge bases" forever.
    paths = AppPaths.for_root(tmp_path / "root", data_root=tmp_path / "data")
    stray = _knowledge(paths.app_versions / "0.3.2" / "knowledge", "rescue me")

    from finesub_bootstrap import fsops

    monkeypatch.setattr(
        fsops.shutil,
        "copytree",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk on fire")),
    )
    messages: list[str] = []

    assert MIGRATION_ID not in apply_pending(paths, log=messages.append)

    assert (stray / "streamer" / "index.md").read_text("utf-8") == "rescue me"
    assert not (paths.user_data / "knowledge").exists()
    assert MIGRATION_ID not in applied_ids(paths)


def test_personal_data_left_in_a_portable_folder_is_brought_along(
    tmp_path: Path,
) -> None:
    # Portable copies used to keep settings and the knowledge base beside the
    # executable, which is how one user ended up with three knowledge bases.
    paths = AppPaths.for_root(tmp_path / "portable", data_root=tmp_path / "data")
    stray = paths.root / "user-data"
    (stray / "logs").mkdir(parents=True)
    (stray / ".env").write_text("GEMINI_FREE=key", encoding="utf-8")

    assert "0002-user-data-to-managed-location" in apply_pending(paths)

    # 0004 may have encrypted the value in the same pass; the plaintext is
    # what must survive the move, not the on-disk representation.
    from finesub_bootstrap import secrets

    assert secrets.read_env_file(paths.user_data / ".env")["GEMINI_FREE"] == "key"
    assert not stray.exists()


def test_two_sets_of_personal_data_are_never_merged(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "portable", data_root=tmp_path / "data")
    stray = paths.root / "user-data"
    stray.mkdir(parents=True)
    (stray / ".env").write_text("portable", encoding="utf-8")
    paths.user_data.mkdir(parents=True)
    (paths.user_data / ".env").write_text("managed", encoding="utf-8")
    messages: list[str] = []

    apply_pending(paths, log=messages.append)

    assert (stray / ".env").read_text("utf-8") == "portable"
    assert (paths.user_data / ".env").read_text("utf-8") == "managed"
    assert any("未自动合并" in message for message in messages)


def test_task_outputs_move_to_the_big_data_root_and_history_follows(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "install", data_root=tmp_path / "data")
    old_tasks = paths.user_data / "tasks"
    (old_tasks / "clip-2026-abc123").mkdir(parents=True)
    (old_tasks / "clip-2026-abc123" / "clip.srt").write_text("1\n", encoding="utf-8")
    history = paths.user_data / "tasks.json"
    history.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "tasks": [
                    {
                        "task_id": "clip-2026-abc123",
                        "request": {
                            "input": "clip.wav",
                            "output": str(
                                old_tasks / "clip-2026-abc123" / "clip.srt"
                            ),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert "0003-tasks-out-of-user-data" in apply_pending(paths)

    assert (paths.tasks / "clip-2026-abc123" / "clip.srt").is_file()
    assert not old_tasks.exists()
    # Recorded relative to the tasks root, so this is the last time these paths
    # need touching: moving the folder again resolves against wherever it is.
    recorded = json.loads(history.read_text("utf-8"))["tasks"][0]["request"]["output"]
    assert recorded == "clip-2026-abc123/clip.srt"


def test_task_outputs_wait_while_the_app_is_running(tmp_path: Path) -> None:
    # A running JobManager holds the history in memory and writes the whole
    # file back, so anything rewritten underneath it would be lost.
    from finesub_bootstrap.locks import holding_lock

    paths = AppPaths.for_root(tmp_path / "install", data_root=tmp_path / "data")
    old_tasks = paths.user_data / "tasks"
    (old_tasks / "clip").mkdir(parents=True)
    paths.tasks.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    with holding_lock(paths.tasks.parent / ".active.lock"):
        applied = apply_pending(paths, log=messages.append)

    assert "0003-tasks-out-of-user-data" not in applied
    assert (old_tasks / "clip").is_dir()
    assert any("正在运行" in message for message in messages)


class _FakeBackend:
    def wrap(self, data: bytes) -> bytes:
        return b"kek|" + data

    def unwrap(self, blob: bytes) -> bytes:
        if not blob.startswith(b"kek|"):
            raise Exception("not ours")
        return blob[len(b"kek|"):]


def test_env_keys_are_protected_and_siblings_warned(tmp_path: Path) -> None:
    from finesub_bootstrap import secrets

    secrets.set_backend(_FakeBackend())
    try:
        paths = AppPaths.for_root(tmp_path / "root", data_root=tmp_path / "data")
        env_path = paths.user_data / ".env"
        env_path.parent.mkdir(parents=True)
        env_path.write_bytes(
            b'# my note\nGEMINI_FREE={"main":"AIzaSecretKey12345"}\n'
        )
        (paths.user_data / ".env.bak").write_text("old copy", encoding="utf-8")
        messages: list[str] = []

        assert "0004-protect-env-keys" in apply_pending(paths, log=messages.append)

        text = env_path.read_bytes().decode("utf-8")
        assert "AIzaSecretKey12345" not in text
        assert "# my note\n" in text  # comments survive byte for byte
        assert secrets.read_env_file(env_path) == {
            "GEMINI_FREE": '{"main":"AIzaSecretKey12345"}'
        }
        assert any(".env.bak" in message for message in messages)
        # The sibling file itself is not ours to touch.
        assert (paths.user_data / ".env.bak").read_text("utf-8") == "old copy"

        # Recorded: the next start does not run it again.
        assert apply_pending(paths, log=messages.append) == []
    finally:
        secrets.set_backend(None)


def test_env_protection_without_dpapi_is_recorded_as_done(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub_bootstrap import secrets
    from finesub_bootstrap.migrations import env_protection

    monkeypatch.setattr(secrets, "available", lambda: False)
    paths = AppPaths.for_root(tmp_path / "root", data_root=tmp_path / "data")
    env_path = paths.user_data / ".env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("GEMINI_FREE=plain-key\n", encoding="utf-8")
    messages: list[str] = []

    assert env_protection.MIGRATION_ID in apply_pending(
        paths, log=messages.append
    )
    # Retrying on a machine without DPAPI would never change the answer.
    assert env_path.read_text("utf-8") == "GEMINI_FREE=plain-key\n"
    assert any("明文" in message for message in messages)


def test_the_ledger_survives_being_written_twice(tmp_path: Path) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    first = Migration(id="0002-first", run=lambda _paths, _log: True)
    second = Migration(id="0003-second", run=lambda _paths, _log: True)

    apply_pending(paths, migrations=(first,))
    apply_pending(paths, migrations=(second,))

    body = json.loads(
        (paths.user_data / migrations.LEDGER_NAME).read_text("utf-8")
    )
    assert body["applied"] == ["0002-first", "0003-second"]
