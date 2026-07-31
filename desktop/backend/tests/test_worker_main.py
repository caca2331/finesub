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

    paths = _fake_paths(tmp_path)
    paths.raw_srt.write_text("raw subtitle", encoding="utf-8")
    paths.metadata_json.write_text("{}", encoding="utf-8")

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
    assert result["rawSrt"].endswith("a-raw.srt")
    assert result["metadataJson"].endswith("a-run.json")
    assert "translatedSrt" not in result
    assert "finalSrt" not in result
    assert events[0].type == "started"
    assert events[-1].type == "completed"


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
