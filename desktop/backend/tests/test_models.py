import pytest
from pydantic import ValidationError

from desktop.backend.common.models import TaskRequest


def test_task_request_defaults_to_local_raw_srt() -> None:
    request = TaskRequest.model_validate({"input": "D:/media/a.mp4"})

    assert request.stage == "raw-srt"
    assert request.device == "cuda"
    assert request.model_name == "large-v3-turbo"
    assert request.gpu_budget_gb == 4
    assert request.language is None


def test_task_request_rejects_fields_that_could_become_commands() -> None:
    with pytest.raises(ValidationError):
        TaskRequest.model_validate(
            {"input": "D:/media/a.mp4", "command": "calc.exe"}
        )


def test_task_request_normalizes_blank_language_to_auto_detection() -> None:
    request = TaskRequest.model_validate(
        {"input": "D:/media/a.mp4", "language": "  "}
    )

    assert request.language is None


def test_task_request_rejects_unsupported_gpu_budget() -> None:
    with pytest.raises(ValidationError):
        TaskRequest.model_validate(
            {"input": "D:/media/a.mp4", "gpu_budget_gb": 10}
        )


def test_task_request_accepts_4gb_gpu_budget() -> None:
    request = TaskRequest.model_validate(
        {"input": "D:/media/a.mp4", "gpu_budget_gb": 4}
    )

    assert request.gpu_budget_gb == 4
