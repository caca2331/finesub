from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from desktop.backend.common.models import TaskRequest
from desktop.backend.worker.main import run_request


def _fake_paths(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        vocal_audio=root / "a-vocal.ogg",
        aligned_json=root / "a-aligned.json",
        stable_json=root / "a-stable.json",
        raw_srt=root / "a-raw.srt",
        translated_srt=root / "a-translated.srt",
        final_srt=root / "a.srt",
        task_artifact_dir=root / "a.llm-artifacts",
        metadata_json=root / "a-run.json",
    )


def test_worker_maps_request_to_pipeline_keywords(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    events = []
    request = TaskRequest(
        input=str(tmp_path / "a.wav"),
        language="ja",
        gpu_budget_gb=12,
        llm_level="high",
        enable_web_search=False,
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
    assert kwargs["llm_level"] == "high"
    assert kwargs["enable_web_search"] is False
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
    # Even when the user asks for cleanup: stable.json is what every later stage
    # reads, and the artifact directory holds the LLM checkpoints. Both are tiny
    # next to the vocal audio, which is what actually fills a disk.
    paths = _run_with(
        TaskRequest(input="placeholder", cleanup_intermediate=True), tmp_path
    )

    assert paths.stable_json.exists()
    assert paths.task_artifact_dir.is_dir()
    assert not paths.vocal_audio.exists()
    assert not paths.aligned_json.exists()
    assert not paths.metadata_json.exists()


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
