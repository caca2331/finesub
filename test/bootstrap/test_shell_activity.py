"""What `doctor` says about who is using this store.

`relocate` and `uninstall` refuse while anything is running. Before this they
refused without saying what was running, and the diagnostic the user was told
to run next had nothing to say about it either -- so the only way forward was
closing things until the refusal stopped.
"""

from __future__ import annotations

from pathlib import Path

from finesub_bootstrap import locks
from finesub_bootstrap.environment import RuntimeEnvironment
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.resources import ResourceManager
from finesub_bootstrap.shell import Shell


def _shell(tmp_path: Path) -> Shell:
    paths = AppPaths.for_root(tmp_path / "root")
    return Shell(
        paths=paths,
        resources=ResourceManager(paths, []),
        runtime=RuntimeEnvironment(
            paths=paths,
            app_source=tmp_path / "source",
            runtime_lock=tmp_path / "source" / "pylock.win-py312.toml",
            uv_executable=lambda: tmp_path / "uv.exe",
        ),
    )


def test_a_quiet_store_says_so_in_one_line(tmp_path: Path) -> None:
    shell = _shell(tmp_path)

    assert shell._activity_report() == ["activity     idle"]


def test_a_held_task_is_named_with_its_front_end(tmp_path: Path) -> None:
    shell = _shell(tmp_path)
    lock = locks.task_lock_path(shell.paths.tasks, "task-77")

    with locks.holding_lock(lock, lease=locks.lease_record("task-77", "desktop")):
        report = shell._activity_report()

    assert "1 个任务被占用" in report[0]
    assert "task-77" in report[1]
    assert "桌面端" in report[1]


def test_a_run_without_a_task_is_still_counted(tmp_path: Path) -> None:
    """This is what refuses a relocation, so a diagnostic has to show it.

    A run that has published its lease but not yet claimed a task -- planning,
    downloading, an older version -- has no name to print. Saying "one instance
    is active" is the whole truth available, and it beats "idle".
    """

    shell = _shell(tmp_path)

    with locks.holding_activity(shell.paths.user_data):
        report = shell._activity_report()

    assert report == ["activity     1 个运行实例，0 个任务被占用"]


def test_the_refusal_names_the_holder_when_there_is_one(tmp_path: Path) -> None:
    shell = _shell(tmp_path)
    lock = locks.task_lock_path(shell.paths.tasks, "task-9")

    assert shell._busy_note() == ""
    with locks.holding_lock(lock, lease=locks.lease_record("task-9", "cli")):
        assert "命令行" in shell._busy_note()


def test_an_unnamed_holder_leaves_the_refusal_as_it_was(tmp_path: Path) -> None:
    """No lease is ordinary; the sentence just loses its parenthetical."""

    shell = _shell(tmp_path)

    with locks.holding_lock(locks.task_lock_path(shell.paths.tasks, "old")):
        assert shell._busy_note() == ""


def test_the_refusal_does_not_nest_its_brackets(tmp_path: Path) -> None:
    """`describe_lease` already ends in a parenthesised pid and start time."""

    shell = _shell(tmp_path)
    lock = locks.task_lock_path(shell.paths.tasks, "task-1")

    with locks.holding_lock(lock, lease=locks.lease_record("task-1", "cli")):
        note = shell._busy_note()

    assert note.startswith("当前占用：")
    assert "（（" not in note


def test_an_install_in_progress_is_not_idle(tmp_path: Path) -> None:
    """`relocate` refuses on this lock too, so `idle` here would be a lie."""

    shell = _shell(tmp_path)

    with locks.holding_lock(shell.paths.runtime / ".install.lock"):
        report = shell._activity_report()

    assert report[-1].strip().startswith("安装或更新正在进行")


def test_a_legacy_worker_is_named_when_it_is_the_only_evidence(
    tmp_path: Path,
) -> None:
    """A version predating leases holds `.active.lock` and nothing else."""

    shell = _shell(tmp_path)

    with locks.holding_lock(locks.active_lock_path(shell.paths.tasks)):
        report = shell._activity_report()

    assert len(report) == 2
    assert "旧版本" in report[1]


def test_a_current_run_does_not_double_report_the_legacy_lock(
    tmp_path: Path,
) -> None:
    """It holds both; two lines for one holder would read as two holders."""

    shell = _shell(tmp_path)

    with locks.holding_activity(shell.paths.user_data):
        with locks.holding_lock(locks.active_lock_path(shell.paths.tasks)):
            report = shell._activity_report()

    assert report == ["activity     1 个运行实例，0 个任务被占用"]
