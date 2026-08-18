from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from finesub_bootstrap import shell as shell_module
from finesub_bootstrap.environment import RuntimeEnvironment
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.resources import ResourceManager
from finesub_bootstrap.locks import (
    holding_activity,
    holding_lock,
    task_lock_path,
    task_workspace_lock_path,
)
from finesub_bootstrap.shell import Shell, package_shell
from finesub_bootstrap.system_tools import SystemTool


def _shell(tmp_path: Path, *, can_provision: bool = True) -> Shell:
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
        can_provision=can_provision,
    )


def _manifest_shell(tmp_path: Path, monkeypatch, *, can_provision: bool = True):
    """A shell whose resources are the real manifest and no system tools."""

    monkeypatch.setattr(shell_module, "system_tool", lambda _resource_id: None)
    shell = _shell(tmp_path, can_provision=can_provision)
    shell.resources = ResourceManager(
        shell.paths,
        shell_module.resource_specs(
            json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "resources"
                    / "runtime-manifest.json"
                ).read_text(encoding="utf-8")
            ),
            exclude=("uv",),
        ),
    )
    return shell


def test_only_the_output_flag_is_the_shells_business(tmp_path: Path) -> None:
    """Every other pipeline flag reaches the pipeline untouched.

    `-o` is the single exception, and only ever an addition: a run that names
    no output gets one under `tasks`, so it is filed where both front ends
    look. Anything else the shell decided to understand would be a table of the
    pipeline's options going stale.
    """

    shell = _shell(tmp_path)
    calls: list[tuple[str, list[str]]] = []
    shell.run_in_runtime = lambda module, arguments: (
        calls.append((module, list(arguments))) or 0
    )

    assert shell.dispatch(["input.wav", "--language", "en", "--word"]) == 0
    module, forwarded = calls[0]
    assert module == "finesub.pipeline"
    assert forwarded[:4] == ["input.wav", "--language", "en", "--word"]
    assert forwarded[4] == "-o"
    assert Path(forwarded[5]).parent.parent == shell.paths.tasks
    assert Path(forwarded[5]).name == "input.srt"


def test_agent_clean_uses_the_shell_python_without_provisioning(
    tmp_path: Path, monkeypatch
) -> None:
    shell = _shell(tmp_path)
    calls: list[tuple[list[str], dict]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(shell_module.subprocess, "run", run)
    monkeypatch.setattr(
        shell,
        "ensure_ready",
        lambda: pytest.fail("agent-clean must not provision the runtime"),
    )

    assert shell.dispatch(["agent-clean", "--all-domains", "--force"]) == 7
    command, kwargs = calls[0]
    assert command == [
        sys.executable,
        "-m",
        "finesub.llm.agent.agent_cleanup",
        "--all-domains",
        "--force",
    ]
    assert kwargs["cwd"] == shell.runtime.app_source
    assert str(shell.runtime.app_source) in kwargs["env"]["PYTHONPATH"]
    assert kwargs["env"]["FINESUB_AGENT_CAPSULE_ROOT"] == str(
        shell.paths.agent_capsules
    )


def test_uninstall_refuses_while_an_agent_activity_lease_is_live(
    tmp_path: Path,
) -> None:
    shell = _shell(tmp_path)
    shell.paths.runtime.mkdir(parents=True)

    with holding_activity(shell.paths.user_data, lease_id="agent-call"):
        assert shell.uninstall([]) == 1

    assert shell.paths.runtime.is_dir()


def test_an_explicit_output_is_where_the_run_happens(tmp_path: Path) -> None:
    """Named a destination, got the run there -- artifacts and all.

    Running elsewhere and copying the result over at the end leaves the folder
    the user is watching empty for the whole job, and empty for good if the run
    dies at the translation stage: the finished transcript would sit in a task
    directory they were never told about.
    """

    shell = _shell(tmp_path)
    calls: list[list[str]] = []
    shell.run_in_runtime = lambda module, arguments: (
        calls.append(list(arguments)) or 1
    )

    shell.dispatch(["input.wav", "-o", str(tmp_path / "mine.srt")])

    assert calls == [["input.wav", "-o", str(tmp_path / "mine.srt")]]


def _task_dirs(shell) -> list[Path]:
    """The task directories, ignoring the advisory lock a run leaves beside them."""

    return sorted(path for path in shell.paths.tasks.iterdir() if path.is_dir())


def _recording_shell(tmp_path: Path, produced: str = "-raw.srt"):
    """A shell whose "pipeline" just writes the file that stage would."""

    shell = _shell(tmp_path)

    def run(_module, arguments):
        target = Path(arguments[arguments.index("-o") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        stem = target.with_suffix("")
        stem.with_name(f"{stem.name}{produced}").write_text("subs", encoding="utf-8")
        stem.with_name(f"{stem.name}-stable.json").write_text("{}", encoding="utf-8")
        stem.with_name(f"{stem.name}-vocal.ogg").write_bytes(b"audio")
        return 0

    shell.run_in_runtime = run
    return shell


def test_a_rerun_of_the_same_source_continues_the_same_task(
    tmp_path: Path, capsys
) -> None:
    """The whole point of matching: the second run reads the first one's work.

    A fresh directory each time would make every rerun redo separation and
    recognition, which is the expensive half and exactly what the artifacts
    left in the task directory exist to avoid.
    """

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)

    shell.dispatch([str(source)])
    first = {path.name for path in _task_dirs(shell)}
    capsys.readouterr()

    shell.dispatch([str(source)])
    second = {path.name for path in _task_dirs(shell)}

    assert first == second
    assert "Continuing task" in capsys.readouterr().err


def test_same_name_in_two_directories_is_two_tasks(
    tmp_path: Path, monkeypatch
) -> None:
    """Matching on the typed string subtitled one file with another's audio.

    `clip.wav` in two directories is the same eleven characters and nothing
    else. Reusing the first task would skip separation and ASR -- its
    `-stable.json` is right there -- and deliver a transcript of the wrong
    recording, with no error anywhere.
    """

    shell = _recording_shell(tmp_path)
    for directory in ("a", "b"):
        source = tmp_path / directory / "clip.wav"
        source.parent.mkdir()
        source.write_bytes(b"audio")
        monkeypatch.chdir(source.parent)
        shell.dispatch(["clip.wav"])

    assert len(_task_dirs(shell)) == 2


def test_continuing_a_task_records_the_cli_defaults_that_actually_ran(
    tmp_path: Path,
) -> None:
    """Desktop defaults must not survive a CLI run that used different ones."""

    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)
    shell.dispatch([str(source)])

    stored = task_index.read(shell._index_path(), shell.paths.tasks)
    entry = stored[-1]
    entry["request"].update({"llm_media": "video", "knowledge": "update"})
    entry["created_at"] = 1.0
    entry["updated_at"] += 0.000001
    task_index.merge_write(shell._index_path(), [entry], shell.paths.tasks)

    shell.dispatch([str(source), "--language", "ja"])

    request = task_index.read(shell._index_path(), shell.paths.tasks)[-1]
    assert request["request"]["llm_media"] == "video"
    assert request["request"]["knowledge"] == "none"
    assert request["request"]["device"] == "cuda"
    assert request["request"]["language"] == "ja"
    assert request["created_at"] == 1.0, "a continued task was created once"


def test_cli_history_preserves_every_setting_the_desktop_can_retry(
    tmp_path: Path,
) -> None:
    from desktop.backend.common.models import TaskRequest
    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    info = tmp_path / "context.txt"
    info.write_text("from file", encoding="utf-8")
    shell = _recording_shell(tmp_path)

    shell.dispatch(
        [
            str(source),
            "--stage",
            "raw-srt",
            "--name",
            "chosen-name",
            "--model",
            "medium",
            "--device",
            "cpu",
            "--language",
            "ja",
            "--gpu-budget-gb",
            "12",
            "--word",
            "--asr-stabilize-profile",
            "2",
            "--split-length-scale",
            "0.8",
            "--llm-media",
            "audio",
            "--llm-retrieval",
            "native",
            "--llm-difficulty",
            "med",
            "--llm-fast",
            "off",
            "--llm-output-scale",
            "0.05",
            "--extra-info",
            "inline",
            "--extra-info-file",
            str(info),
            "--extra-style",
            "concise",
            "--knowledge",
            "collect",
            "--postprocess-profile",
            "4",
        ]
    )

    body = task_index.read(shell._index_path(), shell.paths.tasks)[-1]["request"]
    request = TaskRequest.model_validate(body)
    assert request.model_dump() == {
        "input": str(source.resolve()),
        "output": str(
            shell.paths.tasks
            / next(path.name for path in _task_dirs(shell))
            / "chosen-name.srt"
        ),
        "name": "chosen-name",
        "cleanup_intermediate": False,
        "stage": "raw-srt",
        "model_name": "medium",
        "device": "cpu",
        "gpu_index": None,
        "gpu_name": "",
        "language": "ja",
        "gpu_budget_gb": 12,
        "word": True,
        "asr_stabilize_profile": 2,
        "split_length_scale": 0.8,
        "llm_media": "audio",
        "llm_retrieval": "native",
        "llm_difficulty": "intermediate",
        "llm_fast": "off",
        "llm_output_scale": 0.05,
        "extra_info": "inline\nfrom file",
        "extra_style": "concise",
        "knowledge": "collect",
        "postprocess_profile": 4,
    }


def test_shell_rejects_a_name_the_pipeline_cannot_record(tmp_path: Path) -> None:
    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)

    with pytest.raises(ValueError, match="--name must be a bare name"):
        shell.dispatch([str(source), "--name", "folder/name"])

    assert task_index.read(shell._index_path(), shell.paths.tasks) == []


def test_explicit_output_ignores_an_invalid_name_and_records_no_name(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    output = tmp_path / "chosen.srt"
    shell = _recording_shell(tmp_path)

    assert shell.dispatch(
        [str(source), "-o", str(output), "--name", "ignored/path"]
    ) == 0

    entry = task_index.read(shell._index_path(), shell.paths.tasks)[-1]
    assert entry["request"]["output"] == str(output.resolve())
    assert entry["request"]["name"] == ""


def _mark_running(shell) -> None:
    from finesub_bootstrap import task_index

    entry = task_index.read(shell._index_path(), shell.paths.tasks)[-1]
    entry["state"] = "running"
    task_index.merge_write(shell._index_path(), [entry], shell.paths.tasks)


def test_a_run_holds_the_lock_that_says_a_task_is_running(tmp_path: Path) -> None:
    """The signal everything else consults, which the CLI used to leave unset.

    `relocate` refuses to move the tasks tree while a task runs, and a task
    marked `running` is only skipped while something really is -- both read
    this lock. A CLI run that did not take it was claiming to be idle.
    """

    shell = _shell(tmp_path)
    seen: list[bool] = []
    shell.run_in_runtime = lambda module, arguments: (
        seen.append(shell._nothing_is_running()) or 0
    )

    shell.dispatch(["input.wav"])

    assert seen == [False], "the run was invisible to everything that asks"
    assert shell._nothing_is_running(), "and the lock is released afterwards"


def test_a_running_mark_left_by_a_crash_does_not_strand_the_task(
    tmp_path: Path,
) -> None:
    """The mark outlives the process that set it.

    Only the desktop's next launch clears it, so a user who crashed it once and
    then works from the terminal would start a fresh task for that source
    forever -- redoing separation and recognition every time. When the lock
    says nothing is running anywhere, the mark describes work that stopped.
    """

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)
    shell.dispatch([str(source)])
    _mark_running(shell)

    shell.dispatch([str(source)])

    assert len(_task_dirs(shell)) == 1


def test_a_stage_with_no_subtitle_keeps_the_output_it_was_given(
    tmp_path: Path,
) -> None:
    # There is nothing to copy out afterwards, so filing the run under tasks
    # would answer an explicit `-o` with silence.
    shell = _shell(tmp_path)
    calls: list[list[str]] = []
    shell.run_in_runtime = lambda module, arguments: (
        calls.append(list(arguments)) or 0
    )

    shell.dispatch(["a.wav", "--stage", "stable", "-o", str(tmp_path / "x.srt")])

    assert calls == [
        ["a.wav", "--stage", "stable", "-o", str(tmp_path / "x.srt")]
    ]


def test_a_different_source_starts_a_new_task(tmp_path: Path) -> None:
    shell = _recording_shell(tmp_path)
    for name in ("one.wav", "two.wav"):
        source = tmp_path / name
        source.write_bytes(b"audio")
        shell.dispatch([str(source)])

    assert len(_task_dirs(shell)) == 2


def test_an_explicit_output_keeps_everything_and_records_two_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    destination = tmp_path / "out" / "mine.srt"
    shell = _recording_shell(tmp_path)

    shell.dispatch([str(source), "-o", str(destination)])

    # The user's folder is theirs: nothing is taken out of it.
    produced = {path.name for path in destination.parent.iterdir()}
    assert produced == {"mine-raw.srt", "mine-stable.json", "mine-vocal.ogg"}
    # The task directory keeps only what a later run cannot reproduce: the ASR
    # result, and the correction records when there are any. Not the audio.
    task = _task_dirs(shell)[0]
    assert {path.name for path in task.iterdir()} == {"mine-stable.json"}


def test_an_interrupted_run_still_leaves_a_findable_task(tmp_path: Path) -> None:
    """Recorded before it starts, so being killed does not strand the work.

    A run that dies during the LLM stage has already produced the ASR result;
    without an entry pointing at its directory the next run would mint a new id
    and redo the expensive half, leaving that result somewhere nothing looks.
    """

    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)

    def killed(module, arguments):
        raise KeyboardInterrupt

    shell.run_in_runtime = killed
    with pytest.raises(KeyboardInterrupt):
        shell.dispatch([str(source)])

    stored = task_index.read(shell._index_path(), shell.paths.tasks)
    assert [entry["task_id"] for entry in stored] != []
    # The word the desktop offers a Continue button for, and what happened.
    assert stored[-1]["state"] == "interrupted"


def test_the_history_names_the_subtitle_the_way_the_desktop_reads_it(
    tmp_path: Path,
) -> None:
    # The desktop's history only looks for finalSrt/translatedSrt/rawSrt; any
    # other key is a path it can see and still not offer to open.
    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)

    shell.dispatch([str(source)])

    stored = task_index.read(shell._index_path(), shell.paths.tasks)[-1]
    assert set(stored["outputs"]) == {"rawSrt"}


def test_a_failed_llm_run_still_records_the_transcript_it_made(
    tmp_path: Path,
) -> None:
    """Recording only the requested stage left the useful file unreachable.

    A `final-srt` run that dies in translation has already written `-raw.srt`.
    Filing nothing at all is exactly backwards: that is the case where being
    able to open what did get made matters most.
    """

    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)
    original = shell.run_in_runtime
    shell.run_in_runtime = lambda module, arguments: (
        original(module, arguments) and None
    ) or 1

    shell.dispatch([str(source), "--llm-correct-translate"])

    stored = task_index.read(shell._index_path(), shell.paths.tasks)[-1]
    assert stored["state"] == "failed"
    assert set(stored["outputs"]) == {"rawSrt"}


def test_a_record_is_never_left_half_copied(tmp_path: Path, monkeypatch) -> None:
    # The pipeline skips a stage on the *existence* of its output, so a
    # truncated `-stable.json` would be taken for a finished ASR result: the
    # next run would skip recognition and fail parsing it instead.
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)

    def die_mid_copy(src, dst):
        Path(dst).write_text("half", encoding="utf-8")
        raise OSError("no space left on device")

    monkeypatch.setattr(shell_module.shutil, "copy2", die_mid_copy)
    shell.dispatch([str(source), "-o", str(tmp_path / "out" / "mine.srt")])

    task = _task_dirs(shell)[0]
    assert [path.name for path in task.iterdir()] == []


def test_global_idle_does_not_make_an_owned_running_task_stale(
    tmp_path: Path,
) -> None:
    """A different task releasing the global lock proves nothing about ours.

    This is the three-process race that the tree-wide lock cannot close: A
    holds it, B starts without it, A ends, then C takes it and used to call B's
    fresh ``running`` mark stale.  B's task-id lock remains held, so C must use
    a different directory even though the global lock is now idle.
    """

    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)
    shell.dispatch([str(source)])
    _mark_running(shell)
    entry = task_index.read(shell._index_path(), shell.paths.tasks)[-1]

    with holding_lock(task_lock_path(shell.paths.tasks, entry["task_id"])):
        shell.dispatch([str(source)])

    assert len(_task_dirs(shell)) == 2


def test_cli_does_not_reuse_a_workspace_owned_by_another_task(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap import task_index

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)
    shell.dispatch([str(source)])
    first = task_index.read(shell._index_path(), shell.paths.tasks)[-1]
    first_output = first["request"]["output"]

    with holding_lock(
        task_workspace_lock_path(shell.paths.tasks, first_output)
    ):
        shell.dispatch([str(source)])

    entries = task_index.read(shell._index_path(), shell.paths.tasks)
    assert len(entries) == 2
    assert entries[-1]["task_id"] != first["task_id"]
    assert Path(entries[-1]["request"]["output"]).parent != Path(
        first_output
    ).parent


def test_an_error_from_the_run_is_not_mistaken_for_a_lock_failure(
    tmp_path: Path,
) -> None:
    # The body's exception used to be thrown back into the generator, caught as
    # if the lock had failed, and answered with a second yield -- replacing the
    # real error with `generator didn't stop after throw()`.
    shell = _shell(tmp_path)

    with pytest.raises(OSError, match="the disk went away"):
        with shell._announced_as_running():
            raise OSError("the disk went away")


def test_the_task_is_chosen_and_claimed_under_the_lock(tmp_path: Path) -> None:
    """Two terminals started at the same moment used to pick the same task.

    Choosing one and marking it `running` were both outside the lock, so both
    could read the same completed entry and settle on the same id and output
    path before either announced itself. Planning happens inside the lock now;
    the second terminal cannot take it, so it does not see an idle machine and
    will not treat the first one's fresh mark as stale.

    The idle answer itself has to be taken *before* that, which the crash test
    above depends on: asked from inside, our own hold would say "busy" and no
    crashed mark would ever be recognised as stale again.
    """

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)
    holding: list[bool] = []
    original = shell._plan_run

    def watched(arguments, **kwargs):
        holding.append(not shell._nothing_is_running())
        return original(arguments, **kwargs)

    shell._plan_run = watched
    shell.dispatch([str(source)])

    assert holding == [True], "planned before announcing this run"


def test_a_system_tool_is_probed_once_per_command(tmp_path: Path, monkeypatch) -> None:
    # Finding one means running it, and the token counter loads a vocabulary
    # before it answers -- the reason its probe is allowed 30 seconds.
    shell = _manifest_shell(tmp_path, monkeypatch)
    probes: list[str] = []
    monkeypatch.setattr(
        shell_module,
        "system_tool",
        lambda resource_id: probes.append(resource_id) or None,
    )

    shell._prefer_capabilities(["a.wav", "--stage", "final-srt"])
    shell.tool_file("tokcount", "tokcount.exe")
    shell.tool_directory("tokcount", "tokcount.exe")

    assert probes == ["tokcount"]


def test_records_are_kept_even_when_the_run_fails(tmp_path: Path) -> None:
    # A failure after the ASR stage is exactly when having the result recorded
    # matters: the next attempt reads it instead of redoing recognition.
    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    shell = _recording_shell(tmp_path)
    original = shell.run_in_runtime
    shell.run_in_runtime = lambda module, arguments: (
        original(module, arguments) and None
    ) or 1

    shell.dispatch([str(source), "-o", str(tmp_path / "out" / "mine.srt")])

    task = _task_dirs(shell)[0]
    assert {path.name for path in task.iterdir()} == {"mine-stable.json"}


def test_a_command_that_does_not_start_with_the_source_is_left_alone(
    tmp_path: Path, capsys
) -> None:
    # Which token is the input would take a table of every flag that carries a
    # value; being wrong there files the run under a flag's argument.
    shell = _shell(tmp_path)
    calls: list[list[str]] = []
    shell.run_in_runtime = lambda module, arguments: (
        calls.append(list(arguments)) or 0
    )

    shell.dispatch(["--language", "en", "input.wav"])

    assert calls == [["--language", "en", "input.wav"]]
    assert "has to come first" in capsys.readouterr().err


def test_batch_dispatches_to_the_batch_runner(tmp_path: Path) -> None:
    shell = _shell(tmp_path)
    calls: list[tuple[str, list[str]]] = []
    shell.run_in_runtime = lambda module, arguments: (
        calls.append((module, list(arguments))) or 0
    )

    assert shell.dispatch(["batch", "--manifest", "tasks.jsonl"]) == 0
    assert calls == [("finesub.batch", ["--manifest", "tasks.jsonl"])]


def test_keys_is_a_shell_command_not_a_pipeline_argument(
    tmp_path: Path, capsys
) -> None:
    # Unregistered commands fall through to the pipeline; a typo'd "keys"
    # would otherwise be sent there as an input file.
    shell = _shell(tmp_path)
    shell.run_in_runtime = lambda *arguments: pytest.fail(
        "keys must not reach the pipeline"
    )

    assert shell.dispatch(["keys"]) == 0
    assert "尚未配置任何 API key" in capsys.readouterr().out


def test_keys_masks_by_default_and_reveals_on_request(
    tmp_path: Path, capsys
) -> None:
    shell = _shell(tmp_path)
    env_path = shell.paths.user_data / ".env"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(
        'GEMINI_FREE={"main":"AIzaVeryLongSecret123"}\n', encoding="utf-8"
    )

    assert shell.keys([]) == 0
    masked_output = capsys.readouterr().out
    assert "AIzaVeryLongSecret123" not in masked_output
    assert "main=AIza…t123" in masked_output

    assert shell.keys(["--reveal"]) == 0
    revealed = capsys.readouterr().out
    assert 'GEMINI_FREE={"main":"AIzaVeryLongSecret123"}' in revealed

    assert shell.keys(["--bogus"]) == 2


def test_a_system_tool_is_reported_as_ready_without_downloading(
    tmp_path: Path, monkeypatch
) -> None:
    found = SystemTool(path=Path("C:/tools/ffmpeg.exe"), version="ffmpeg 7.1")
    monkeypatch.setattr(
        shell_module,
        "system_tool",
        lambda resource_id: found if resource_id == "ffmpeg" else None,
    )
    shell = _shell(tmp_path)
    installed: list[str] = []
    monkeypatch.setattr(
        shell.resources, "install", lambda *a, **k: installed.append(a[0])
    )

    shell._ensure_resource("ffmpeg", "reason")

    assert installed == []
    assert shell._tool_state("ffmpeg") == "ready"
    assert "system" in shell._tool_report("ffmpeg", "")


def test_on_demand_tools_install_only_when_the_command_needs_them(
    tmp_path: Path, monkeypatch
) -> None:
    shell = _manifest_shell(tmp_path, monkeypatch)
    installed: list[str] = []
    monkeypatch.setattr(
        shell.resources, "install", lambda *a, **k: installed.append(a[0])
    )

    shell._ensure_capabilities(["a.wav"])
    assert installed == []

    shell._ensure_capabilities(["https://example.test/v"])
    assert installed == ["yt-dlp"]


def test_the_token_counter_is_fetched_for_llm_runs_only(
    tmp_path: Path, monkeypatch
) -> None:
    shell = _manifest_shell(tmp_path, monkeypatch)
    installed: list[str] = []
    monkeypatch.setattr(
        shell.resources, "install", lambda *a, **k: installed.append(a[0])
    )

    shell._prefer_capabilities(["a.wav"])
    assert installed == []

    shell._prefer_capabilities(["a.wav", "--stage", "final-srt"])
    assert installed == ["tokcount"]


def test_a_token_counter_that_cannot_be_fetched_does_not_stop_the_run(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The whole point of the preferred/required split.

    Without the binary the pipeline counts tokens through the free countTokens
    endpoint, so a dead mirror here costs a network round trip per count. It
    must never cost the run.
    """

    shell = _manifest_shell(tmp_path, monkeypatch)

    def explode(*_args, **_kwargs):
        raise RuntimeError("mirror is down")

    monkeypatch.setattr(shell.resources, "install", explode)

    shell._prefer_capabilities(["a.wav", "--stage", "final-srt"])

    assert "continuing without it" in capsys.readouterr().err


def test_a_packaged_shell_does_not_fetch_the_token_counter(
    tmp_path: Path, monkeypatch
) -> None:
    # It runs on the managed interpreter and provisions nothing; an optional
    # tool is the last thing that should make it try.
    shell = _manifest_shell(tmp_path, monkeypatch, can_provision=False)
    monkeypatch.setattr(
        shell.resources,
        "install",
        lambda *a, **k: pytest.fail("a packaged shell must not provision"),
    )

    shell._prefer_capabilities(["a.wav", "--stage", "final-srt"])


def test_a_shell_that_cannot_provision_sends_the_user_to_the_app(
    tmp_path: Path, monkeypatch
) -> None:
    # The packaged command line runs *on* the managed interpreter, so it cannot
    # be what installs or replaces it -- and a dead-end "run setup" would be
    # worse than saying where setup actually lives.
    found = SystemTool(path=Path("C:/tools/ffmpeg.exe"), version="ffmpeg 7.1")
    monkeypatch.setattr(shell_module, "system_tool", lambda _resource_id: found)
    monkeypatch.setattr(shell_module.os, "name", "nt")
    shell = _shell(tmp_path, can_provision=False)
    monkeypatch.setattr(
        shell.resources,
        "install",
        lambda *a, **k: pytest.fail("a packaged shell must not provision"),
    )
    monkeypatch.setattr(
        shell.runtime,
        "install",
        lambda *a, **k: pytest.fail("a packaged shell must not provision"),
    )

    with pytest.raises(SystemExit, match="FineSub Desktop"):
        shell.ensure_ready()


def test_relocate_moves_all_big_data_and_leaves_the_runtime(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.paths import ensure_store, recorded_big_data

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    shell.paths.models.mkdir(parents=True, exist_ok=True)
    (shell.paths.models / "weights.bin").write_bytes(b"data")
    shell.paths.agent_capsules.mkdir(parents=True)
    (shell.paths.agent_capsules / "failed-call").mkdir()
    shell.paths.runtime.mkdir(parents=True, exist_ok=True)
    (shell.paths.runtime / "python").mkdir()

    assert shell.relocate([str(tmp_path / "elsewhere")]) == 0

    moved = (tmp_path / "elsewhere").resolve()
    assert (moved / "models" / "weights.bin").is_file()
    assert (moved / "agent-capsules" / "failed-call").is_dir()
    assert not (tmp_path / "root" / "models").exists()
    assert not (tmp_path / "root" / "agent-capsules").exists()
    # The runtime is version-bound and hardlinks out of the download cache, so
    # it stays where the application is.
    assert (tmp_path / "root" / "runtime" / "python").is_dir()
    assert recorded_big_data(shell.paths.data_root) == moved
    assert (moved / ".finesub-store.json").is_file()
    assert (moved / "register-location.cmd").is_file()


def test_relocate_reset_brings_the_data_back_to_the_installation(
    tmp_path: Path,
) -> None:
    # The install root is the one destination that legitimately contains the
    # runtime, so the "do not swallow the runtime" guard has to exempt it --
    # otherwise --reset can never succeed.
    from finesub_bootstrap.paths import ensure_store, recorded_big_data

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    shell.paths.models.mkdir(parents=True, exist_ok=True)
    (shell.paths.models / "weights.bin").write_bytes(b"data")
    shell.paths.runtime.mkdir(parents=True, exist_ok=True)
    assert shell.relocate([str(tmp_path / "elsewhere")]) == 0

    assert shell.relocate(["--reset"]) == 0

    home = (tmp_path / "root").resolve()
    assert (home / "models" / "weights.bin").is_file()
    assert not (tmp_path / "elsewhere" / "models").exists()
    assert recorded_big_data(shell.paths.data_root) == home


def test_relocate_refuses_a_destination_that_is_not_ours(tmp_path: Path) -> None:
    shell = _shell(tmp_path)
    occupied = tmp_path / "someone-elses"
    occupied.mkdir()
    (occupied / "important.txt").write_text("do not touch", encoding="utf-8")

    assert shell.relocate([str(occupied)]) == 2
    assert (occupied / "important.txt").is_file()


def test_relocate_adopts_data_carried_over_by_hand(tmp_path: Path) -> None:
    # Upgrading by installing fresh elsewhere: the user keeps the several GB
    # they already downloaded by moving models/cache out of the old
    # installation. That folder has no marker -- nothing wrote one for it --
    # so adoption has to recognise the contents.
    from finesub_bootstrap.paths import recorded_big_data

    shell = _shell(tmp_path)
    carried = tmp_path / "carried-over"
    (carried / "models").mkdir(parents=True)
    (carried / "models" / "weights.bin").write_bytes(b"expensive")
    (carried / "cache").mkdir()

    assert shell.relocate([str(carried)]) == 0

    assert (carried / "models" / "weights.bin").read_bytes() == b"expensive"
    assert (carried / ".finesub-store.json").is_file()
    assert recorded_big_data(shell.paths.data_root) == carried.resolve()


def test_relocate_still_refuses_a_whole_old_installation(tmp_path: Path) -> None:
    # An installation directory also holds models and cache, but pointing the
    # store at it would put the app and its runtime inside the data root.
    shell = _shell(tmp_path)
    old_install = tmp_path / "finesub-full-0.3.2-win-x64"
    (old_install / "models").mkdir(parents=True)
    (old_install / "runtime").mkdir()
    (old_install / "FineSub Desktop.exe").write_bytes(b"MZ")

    assert shell.relocate([str(old_install)]) == 2


def test_relocate_waits_for_a_running_task(tmp_path: Path) -> None:
    from finesub_bootstrap.paths import ensure_store

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    shell.paths.tasks.mkdir(parents=True, exist_ok=True)

    with holding_lock(shell.paths.tasks / ".active.lock"):
        assert shell.relocate([str(tmp_path / "elsewhere")]) == 1

    assert not (tmp_path / "elsewhere").exists()


def test_relocate_stays_blocked_until_every_parallel_task_ends(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.paths import ensure_store

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    with holding_activity(shell.paths.user_data, lease_id="task-b"):
        with holding_activity(shell.paths.user_data, lease_id="task-a"):
            pass
        assert shell.relocate([str(tmp_path / "elsewhere")]) == 1

    assert not (tmp_path / "elsewhere").exists()


def test_a_run_refreshes_paths_after_another_process_relocates(
    tmp_path: Path,
) -> None:
    from finesub_bootstrap.paths import ensure_store

    stale_shell = _recording_shell(tmp_path)
    relocating_shell = _shell(tmp_path)
    ensure_store(relocating_shell.paths)
    destination = tmp_path / "elsewhere"
    assert relocating_shell.relocate([str(destination)]) == 0

    source = tmp_path / "clip.wav"
    source.write_bytes(b"audio")
    stale_shell.dispatch([str(source)])

    assert stale_shell.paths.tasks.parent == destination.resolve()
    assert len(_task_dirs(stale_shell)) == 1
    assert not (stale_shell.paths.root / "tasks").exists()


def test_relocate_adopts_an_existing_store_without_copying(tmp_path: Path) -> None:
    from finesub_bootstrap.paths import AppPaths, ensure_store, recorded_big_data

    shell = _shell(tmp_path)
    existing = AppPaths.for_root(
        tmp_path / "shared", data_root=shell.paths.data_root
    )
    ensure_store(existing)
    existing.models.mkdir(parents=True, exist_ok=True)
    (existing.models / "weights.bin").write_bytes(b"already here")

    assert shell.relocate([str(existing.big_data)]) == 0

    assert (existing.models / "weights.bin").read_bytes() == b"already here"
    assert recorded_big_data(shell.paths.data_root) == existing.big_data


def _packaged_install(root: Path, version: str = "2.3.4") -> Path:
    source = root / "app" / "versions" / version
    (source / "src" / "finesub").mkdir(parents=True)
    (source / "src" / "finesub" / "pipeline.py").write_text(
        "PIPELINE = True\n", "utf-8"
    )
    (source / "desktop" / "resources").mkdir(parents=True)
    (source / "desktop" / "resources" / "runtime-manifest.json").write_text(
        json.dumps({"resources": []}), "utf-8"
    )
    (source / "desktop" / "runtime").mkdir(parents=True)
    (source / "desktop" / "runtime" / "pylock.win-py312.toml").write_text(
        'lock-version = "1.0"\n', "utf-8"
    )
    (root / "app").mkdir(parents=True, exist_ok=True)
    (root / "app" / "current.json").write_text(
        json.dumps({"current": version}), "utf-8"
    )
    return source


def test_a_packaged_shell_drives_the_install_it_sits_in(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "FineSub-portable"
    source = _packaged_install(root)
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    shell = package_shell(root)

    assert shell.paths.root == root.resolve()
    assert shell.runtime.app_source == source.resolve()
    assert shell.paths.runtime == (root / "runtime").resolve()
    # Personal data is shared with every other front end; the big, rebuildable
    # half stays with this installation.
    assert shell.paths.user_data == (
        local_app_data / "FineSub" / "user-data"
    ).resolve()
    assert shell.paths.models == (root / "models").resolve()
    assert not shell.can_provision


def test_a_packaged_shell_adopts_a_registered_store(
    tmp_path: Path, monkeypatch
) -> None:
    from finesub_bootstrap.paths import AppPaths, ensure_store

    root = tmp_path / "FineSub-portable"
    _packaged_install(root)
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    elsewhere = AppPaths.for_root(
        tmp_path / "shared", data_root=(local_app_data / "FineSub")
    )
    ensure_store(elsewhere)

    shell = package_shell(root)

    assert shell.paths.models == elsewhere.models
    assert shell.paths.runtime == (root / "runtime").resolve()


def test_relocate_records_the_destination_before_it_moves_anything(
    tmp_path: Path,
) -> None:
    """The record must never lag behind the data.

    Within one volume `move_store` is a rename, so the source is released the
    instant it succeeds; recording afterwards meant a crash in between left
    `tasks/` -- which the docs call irreplaceable -- at a location nothing
    pointed at, and the next start looked like a fresh install.
    """
    from finesub_bootstrap.paths import (
        ensure_store,
        recorded_big_data,
        recorded_migration_source,
    )
    from finesub_bootstrap import shell as shell_module

    shell = _shell(tmp_path)
    ensure_store(shell.paths)
    shell.paths.tasks.mkdir(parents=True, exist_ok=True)
    (shell.paths.tasks / "job-1").mkdir()
    data_root = shell.paths.data_root
    source_root = shell.paths.big_data
    destination = (tmp_path / "elsewhere").resolve()

    seen: dict[str, object] = {}
    original = shell_module.move_store

    def watch(source, target, names):
        # What the record says at the exact moment the data starts moving.
        seen["recorded"] = recorded_big_data(data_root)
        seen["migrating_from"] = recorded_migration_source(data_root)
        return original(source, target, names)

    shell_module.move_store = watch
    try:
        assert shell.relocate([str(destination)]) == 0
    finally:
        shell_module.move_store = original

    assert seen["recorded"] == destination, "destination recorded before the move"
    assert seen["migrating_from"] == source_root, "old root kept searchable"
    # ... and the marker is gone once it completed, so resolving is root-level.
    assert recorded_migration_source(data_root) is None
    assert (destination / "tasks" / "job-1").is_dir()
