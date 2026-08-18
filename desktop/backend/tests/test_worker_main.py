from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.reporting import current_reporter
from finesub_bootstrap.locks import (
    LockUnavailable,
    TASK_ACTIVITY_ROOT_VARIABLE,
    holding_lock,
    task_lock_path,
    task_workspace_lock_path,
    try_lock,
)

from desktop.backend.common.models import TaskRequest
from desktop.backend.worker.main import (
    _announcing_this_task,
    run_request,
)


def _fake_paths(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        vocal_audio=root / "a-vocal.ogg",
        aligned_json=root / "a-aligned.json",
        stable_json=root / "a-stable.json",
        raw_srt=root / "a-raw.srt",
        translated_srt=root / "a-translated.srt",
        final_srt=root / "a.srt",
        task_artifact_dir=root / "a.llm-artifacts",
        # Every name here has to be the one `default_pipeline_paths` derives:
        # cleanup now resolves them from the delivered SRT rather than reading
        # this object, so a fake that invents a name tests nothing real.
        metadata_json=root / "a-metadata.json",
    )


def test_worker_task_lock_is_exclusive_but_different_tasks_can_run(
    tmp_path: Path, monkeypatch
) -> None:
    from desktop.backend.worker import main as worker_main

    monkeypatch.setenv("FINESUB_TASKS_ROOT", str(tmp_path))
    monkeypatch.setenv(TASK_ACTIVITY_ROOT_VARIABLE, str(tmp_path / "user-data"))
    # Production waits briefly for the advisory global lock. Tests make both
    # lock attempts immediate so this exercises the same fallback without a
    # five-second delay.
    monkeypatch.setattr(
        worker_main,
        "holding_lock",
        lambda path, **_kwargs: holding_lock(path, timeout=0),
    )

    first = task_lock_path(tmp_path, "task-1")
    second = task_lock_path(tmp_path, "task-2")
    with _announcing_this_task("task-1"):
        assert not try_lock(first)
        with pytest.raises(LockUnavailable):
            with _announcing_this_task("task-1"):
                pass
        with _announcing_this_task("task-2"):
            assert not try_lock(second)

    assert try_lock(first)
    assert try_lock(second)


def test_worker_exclusively_owns_the_output_workspace_it_reuses(
    tmp_path: Path, monkeypatch
) -> None:
    from desktop.backend.worker import main as worker_main

    monkeypatch.setenv("FINESUB_TASKS_ROOT", str(tmp_path))
    monkeypatch.setenv(TASK_ACTIVITY_ROOT_VARIABLE, str(tmp_path / "user-data"))
    monkeypatch.setattr(
        worker_main,
        "holding_lock",
        lambda path, **_kwargs: holding_lock(path, timeout=0),
    )
    reused = tmp_path / "task-a" / "clip.srt"
    separate = tmp_path / "task-c" / "clip.srt"
    workspace = task_workspace_lock_path(tmp_path, reused)

    with _announcing_this_task("task-b", str(reused)):
        assert not try_lock(workspace)
        with pytest.raises(LockUnavailable):
            with _announcing_this_task("task-a", str(reused)):
                pass
        with _announcing_this_task("task-c", str(separate)):
            pass

    assert try_lock(workspace)


def test_worker_maps_request_to_pipeline_keywords(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    events = []
    request = TaskRequest(
        input=str(tmp_path / "a.wav"),
        language="ja",
        gpu_budget_gb=12,
        llm_media="video",
        llm_retrieval="local",
        llm_difficulty="quality",
        cleanup_intermediate=True,
    )

    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    request = request.model_copy(update={"input": str(source_file)})
    private_output = tmp_path / "private" / "task-1"
    private_output.mkdir(parents=True)
    paths = _fake_paths(private_output)
    paths.raw_srt.write_text("raw subtitle", encoding="utf-8")
    paths.metadata_json.write_text("{}", encoding="utf-8")
    paths.vocal_audio.write_bytes(b"vocal")
    paths.aligned_json.write_text("{}", encoding="utf-8")

    result = run_request(
        request,
        task_id="task-1",
        pipeline=lambda source, **kwargs: (
            calls.append((source, kwargs)) or paths
        ),
        emit=events.append,
    )

    source, kwargs = calls[0]
    assert source == request.input
    assert kwargs["stage"] == "raw-srt"
    assert kwargs["language"] == "ja"
    assert kwargs["gpu_budget_gb"] == 12
    assert kwargs["llm_media"] == "video"
    assert kwargs["llm_retrieval"] == "local"
    assert kwargs["llm_difficulty"] == "quality"
    assert kwargs["task_id"] == "task-1"
    assert result == {"rawSrt": str(tmp_path / "a-raw.srt")}
    assert (tmp_path / "a-raw.srt").read_text(encoding="utf-8") == "raw subtitle"
    assert not paths.metadata_json.exists()
    assert not paths.vocal_audio.exists()
    assert not paths.aligned_json.exists()
    assert "translatedSrt" not in result
    assert "finalSrt" not in result
    assert events[0].type == "started"
    assert events[-1].type == "completed"


def test_the_only_stage_reported_is_the_one_the_pipeline_entered(
    tmp_path: Path,
) -> None:
    # The worker used to announce the *requested* stage the moment the task
    # started, which the progress list read as "everything before it is done":
    # asking for raw-srt ticked separation, ASR and stabilization instantly and
    # then never moved again. Nothing may be reported until the pipeline says
    # where it is.
    events = []
    paths = _fake_paths(tmp_path / "private")
    paths.raw_srt.parent.mkdir(parents=True)
    paths.raw_srt.write_text("raw subtitle", encoding="utf-8")
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")

    def pipeline(source: str, **kwargs: object) -> object:
        # The pipeline reads the reporter the worker bound for it rather than
        # taking a callback parameter.
        reporter = current_reporter()
        reporter.planned(["vocal", "aligned", "stable", "raw-srt"])
        reporter.stage_started("vocal", reused=True)
        reporter.stage_started("aligned")
        reporter.stage_started("stable")
        reporter.stage_started("raw-srt")
        return paths

    run_request(
        TaskRequest(input=str(source_file)),
        task_id="task-stages",
        pipeline=pipeline,
        emit=events.append,
    )

    stages = [
        (event.payload["stage"], event.payload["reused"])
        for event in events
        if event.type == "stage"
    ]
    assert stages == [
        ("vocal", True),
        ("aligned", False),
        ("stable", False),
        ("raw-srt", False),
    ]
    assert events[0].type == "started"
    assert events[1].payload["stage"] == "vocal"


def test_worker_keeps_private_artifacts_when_pipeline_fails(tmp_path: Path) -> None:
    events = []
    paths = _fake_paths(tmp_path / "private")
    paths.raw_srt.parent.mkdir(parents=True)
    paths.raw_srt.write_text("partial", encoding="utf-8")

    def fail_after_partial(source: str, **kwargs: object) -> object:
        raise RuntimeError("translation failed")

    try:
        run_request(
            TaskRequest(input=str(tmp_path / "a.wav")),
            task_id="task-partial",
            pipeline=fail_after_partial,
            emit=events.append,
        )
    except RuntimeError:
        pass

    assert paths.raw_srt.exists()


def test_worker_emits_failed_event_before_reraising(tmp_path: Path) -> None:
    events = []

    try:
        run_request(
            TaskRequest(input=str(tmp_path / "a.wav")),
            task_id="task-2",
            pipeline=lambda source, **kwargs: (_ for _ in ()).throw(
                RuntimeError("GPU unavailable")
            ),
            emit=events.append,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("run_request should re-raise pipeline failures")

    assert events[-1].type == "failed"
    assert events[-1].payload["message"] == "GPU unavailable"


def _run_with(request: TaskRequest, tmp_path: Path) -> SimpleNamespace:
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    private_output = tmp_path / "private" / "task-1"
    private_output.mkdir(parents=True)
    paths = _fake_paths(private_output)
    paths.raw_srt.write_text("raw subtitle", encoding="utf-8")
    paths.metadata_json.write_text("{}", encoding="utf-8")
    paths.vocal_audio.write_bytes(b"vocal")
    paths.aligned_json.write_text("{}", encoding="utf-8")
    paths.stable_json.write_text("{}", encoding="utf-8")
    paths.task_artifact_dir.mkdir()
    (paths.task_artifact_dir / "session.json").write_text("{}", encoding="utf-8")

    run_request(
        request.model_copy(update={"input": str(source_file)}),
        task_id="task-1",
        pipeline=lambda source, **kwargs: paths,
        emit=lambda event: None,
    )
    return paths


def test_intermediate_artifacts_survive_by_default(tmp_path: Path) -> None:
    # Deleting them was the old default. It made every rerun redo separation
    # and recognition, and left a standalone correction pass with no input.
    paths = _run_with(TaskRequest(input="placeholder"), tmp_path)

    assert paths.stable_json.exists()
    assert paths.aligned_json.exists()
    assert paths.vocal_audio.exists()
    assert paths.metadata_json.exists()
    assert paths.task_artifact_dir.is_dir()


def test_cleanup_keeps_what_a_rerun_needs(tmp_path: Path) -> None:
    # stable.json survives even here: it is what every later stage reads, and it
    # is tiny next to the vocal audio, which is what actually fills a disk.
    paths = _run_with(
        TaskRequest(input="placeholder", cleanup_intermediate=True), tmp_path
    )

    assert paths.stable_json.exists()
    assert not paths.vocal_audio.exists()
    assert not paths.aligned_json.exists()
    assert not paths.metadata_json.exists()


def test_cleanup_removes_the_llm_checkpoints(tmp_path: Path) -> None:
    # They exist to survive a failure, and cleanup only runs when the task
    # finished. Keeping them left the biggest remaining directory on disk for a
    # resume that can no longer happen.
    paths = _run_with(
        TaskRequest(input="placeholder", cleanup_intermediate=True), tmp_path
    )

    assert not paths.task_artifact_dir.exists()


def test_a_task_that_cannot_be_tidied_is_still_a_finished_task(
    tmp_path: Path, monkeypatch
) -> None:
    # Cleanup runs after the subtitle is published, inside the block that marks
    # a task failed. One open handle -- an antivirus scan, an editor holding an
    # artifact -- must not turn a finished task into a failed one.
    from finesub_bootstrap import artifacts

    def _refuse(path: Path) -> None:
        raise PermissionError(f"{path} is in use")

    monkeypatch.setattr(artifacts, "remove_tree", _refuse)

    events = []
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    private_output = tmp_path / "private" / "task-1"
    private_output.mkdir(parents=True)
    paths = _fake_paths(private_output)
    paths.raw_srt.write_text("raw subtitle", encoding="utf-8")
    paths.task_artifact_dir.mkdir()

    outputs = run_request(
        TaskRequest(
            input=str(source_file), cleanup_intermediate=True
        ),
        task_id="task-1",
        pipeline=lambda source, **kwargs: paths,
        emit=events.append,
    )

    assert events[-1].type == "completed"
    assert outputs
    assert paths.task_artifact_dir.is_dir()


def test_a_failed_task_keeps_everything_it_can_resume_from(tmp_path: Path) -> None:
    # The point of the switch being "clean up when finished": a run that dies
    # part way must still find its checkpoints and its separated audio.
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    paths = _fake_paths(tmp_path / "run")
    paths.final_srt.parent.mkdir(parents=True)
    paths.vocal_audio.write_bytes(b"vocal")
    paths.stable_json.write_text("{}", encoding="utf-8")
    paths.task_artifact_dir.mkdir()
    (paths.task_artifact_dir / "session-checkpoints.jsonl").write_text(
        "{}", encoding="utf-8"
    )

    def failing_pipeline(source: str, **kwargs: object):
        raise RuntimeError("the LLM stage died")

    import pytest

    with pytest.raises(RuntimeError):
        run_request(
            TaskRequest(
                input=str(source_file),
                stage="final-srt",
                cleanup_intermediate=True,
            ),
            task_id="task-1",
            pipeline=failing_pipeline,
            emit=lambda event: None,
        )

    assert paths.vocal_audio.exists()
    assert paths.stable_json.exists()
    assert (paths.task_artifact_dir / "session-checkpoints.jsonl").exists()


def test_name_becomes_the_output_stem_without_moving_the_run_directory(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    paths = _fake_paths(tmp_path / "run")
    paths.final_srt.parent.mkdir(parents=True)
    paths.raw_srt.write_text("raw", encoding="utf-8")

    run_request(
        TaskRequest(input=str(source_file), name="my-clip"),
        task_id="task-1",
        pipeline=lambda source, **kwargs: (calls.append(kwargs) or paths),
        emit=lambda event: None,
    )

    # The CLI's --name contract: out/<name>/<name>.srt.
    assert calls[0]["output_path"] == str(Path("out") / "my-clip" / "my-clip.srt")


def test_an_explicit_output_path_still_wins_over_name(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    paths = _fake_paths(tmp_path / "run")
    paths.final_srt.parent.mkdir(parents=True)
    paths.raw_srt.write_text("raw", encoding="utf-8")

    run_request(
        TaskRequest(
            input=str(source_file),
            name="my-clip",
            output=str(tmp_path / "explicit.srt"),
        ),
        task_id="task-1",
        pipeline=lambda source, **kwargs: (calls.append(kwargs) or paths),
        emit=lambda event: None,
    )

    assert calls[0]["output_path"] == str(tmp_path / "explicit.srt")


def test_a_name_with_a_separator_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        TaskRequest(input="a.wav", name="../escape")
    with pytest.raises(ValueError):
        TaskRequest(input="a.wav", name="nested/name")


def _publish_beside(source_file: Path, *, task_id: str, run_dir: Path) -> dict[str, str]:
    paths = _fake_paths(run_dir)
    paths.final_srt.parent.mkdir(parents=True, exist_ok=True)
    paths.raw_srt.write_text(f"subtitle from {task_id}", encoding="utf-8")
    return run_request(
        TaskRequest(input=str(source_file)),
        task_id=task_id,
        pipeline=lambda source, **kwargs: paths,
        emit=lambda event: None,
    )


def _record_outputs(outputs: dict[str, str], *, task_id: str) -> None:
    """File `outputs` in the shared index, the way a finished task does."""

    from finesub_bootstrap import task_index
    from finesub_bootstrap.paths import default_data_root, load_app_paths

    data_root = default_data_root()
    app_paths = load_app_paths(data_root, data_root=data_root)
    task_index.merge_write(
        app_paths.user_data / "tasks.json",
        [{"task_id": task_id, "outputs": outputs}],
        app_paths.tasks,
    )


def test_a_subtitle_we_never_wrote_is_not_overwritten(tmp_path: Path) -> None:
    # The user's own `a-raw.srt` beside their video: hand-made, or another
    # tool's. Publishing over it destroys work the app never produced.
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    stranger = tmp_path / "a-raw.srt"
    stranger.write_text("hand made", encoding="utf-8")

    outputs = _publish_beside(source_file, task_id="task-1", run_dir=tmp_path / "run")

    assert stranger.read_text(encoding="utf-8") == "hand made"
    assert outputs == {"rawSrt": str(tmp_path / "a-raw.finesub.srt")}
    assert (tmp_path / "a-raw.finesub.srt").is_file()


def test_a_rerun_replaces_the_subtitle_the_last_run_published(tmp_path: Path) -> None:
    # Same file, new task id. Treating our own output as a stranger's would
    # leave one video with a folder of near-identical subtitles.
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")

    first = _publish_beside(source_file, task_id="task-1", run_dir=tmp_path / "run-1")
    _record_outputs(first, task_id="task-1")
    second = _publish_beside(source_file, task_id="task-2", run_dir=tmp_path / "run-2")

    assert second == first
    assert (
        Path(second["rawSrt"]).read_text(encoding="utf-8")
        == "subtitle from task-2"
    )
    assert not (tmp_path / "a-raw.finesub.srt").exists()


def test_a_second_stranger_pushes_the_serial_along(tmp_path: Path) -> None:
    source_file = tmp_path / "a.wav"
    source_file.write_bytes(b"audio")
    (tmp_path / "a-raw.srt").write_text("hand made", encoding="utf-8")
    (tmp_path / "a-raw.finesub.srt").write_text("also theirs", encoding="utf-8")

    outputs = _publish_beside(source_file, task_id="task-1", run_dir=tmp_path / "run")

    assert outputs == {"rawSrt": str(tmp_path / "a-raw.finesub.2.srt")}
    assert (tmp_path / "a-raw.finesub.srt").read_text(encoding="utf-8") == "also theirs"


def test_a_url_task_publishes_into_the_run_directory(tmp_path: Path) -> None:
    # There is no source file next to which a subtitle could be published, so
    # the generated path is the result. Path("https://...").is_file() is what
    # decides this, and it must not be mistaken for a relative path.
    paths = _fake_paths(tmp_path / "run")
    paths.final_srt.parent.mkdir(parents=True)
    paths.raw_srt.write_text("raw", encoding="utf-8")

    outputs = run_request(
        TaskRequest(input="https://example.test/watch?v=1"),
        task_id="task-url",
        pipeline=lambda source, **kwargs: paths,
        emit=lambda event: None,
    )

    assert outputs == {"rawSrt": str(paths.raw_srt)}
    assert paths.raw_srt.exists()


def test_a_url_reaches_the_pipeline_unchanged(tmp_path: Path) -> None:
    calls: list[str] = []
    paths = _fake_paths(tmp_path / "run")
    paths.final_srt.parent.mkdir(parents=True)
    paths.raw_srt.write_text("raw", encoding="utf-8")

    run_request(
        TaskRequest(input="https://example.test/watch?v=1"),
        task_id="task-url",
        pipeline=lambda source, **kwargs: (calls.append(source) or paths),
        emit=lambda event: None,
    )

    assert calls == ["https://example.test/watch?v=1"]
