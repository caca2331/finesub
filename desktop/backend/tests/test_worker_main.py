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
    assert not private_output.exists()
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
