import pytest
from pydantic import ValidationError

from desktop.backend.common.models import TaskRequest


def test_task_request_defaults_to_local_raw_srt() -> None:
    request = TaskRequest.model_validate({"input": "D:/media/a.mp4"})

    assert request.stage == "raw-srt"
    # None, not "cuda": "not chosen" has to stay distinguishable from a
    # choice, or `--gpu-tier cpu` plus an explicit cuda cannot be refused.
    assert request.device is None
    assert request.model_name == "large-v3-turbo"
    assert request.gpu_tier == "auto"
    assert request.language is None
    assert request.llm_media == "audio"
    assert request.llm_retrieval == "local"
    assert request.llm_difficulty == "quality"


def test_the_difficulty_is_the_word_the_llm_layer_actually_accepts() -> None:
    """The worker passes this straight through to `resolve_profile`.

    It used to send the retired `high`/`med`/`minimum`, which that function
    rejects -- so any desktop run reaching the LLM stage died there, *after*
    ASR had already paid for itself. Nothing caught it because a plain
    transcription never reads the value.
    """

    from finesub.llm.routing.profiles import DIFFICULTY

    request = TaskRequest.model_validate({"input": "D:/media/a.mp4"})

    assert request.llm_difficulty in DIFFICULTY


def test_history_written_before_the_rename_still_loads() -> None:
    """Old task records must not vanish from the user's history.

    `jobs/history.py` skips entries it cannot validate rather than failing --
    one unreadable record must not cost the rest of the list. That makes a
    narrowed vocabulary silently destructive: every record predating the
    2026-08-12 rename carries `high`, so rejecting it would empty the task
    list with no error anywhere.
    """

    request = TaskRequest.model_validate(
        {"input": "D:/media/a.mp4", "llm_difficulty": "high"}
    )

    assert request.llm_difficulty == "quality"


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


def test_task_request_rejects_an_unknown_gpu_tier() -> None:
    with pytest.raises(ValidationError):
        TaskRequest.model_validate(
            {"input": "D:/media/a.mp4", "gpu_tier": "gigantic"}
        )


def test_task_request_accepts_a_named_tier_and_auto() -> None:
    """`auto` is a legal value, not just the default: the backend resolves it."""

    request = TaskRequest.model_validate(
        {"input": "D:/media/a.mp4", "gpu_tier": "standard"}
    )
    assert request.gpu_tier == "standard"

    detected = TaskRequest.model_validate(
        {"input": "D:/media/a.mp4", "gpu_tier": "auto"}
    )
    assert detected.gpu_tier == "auto"


def test_task_request_accepts_all_pipeline_postprocess_profiles() -> None:
    for profile in (-1, 0, 1, 2, 3, 4):
        request = TaskRequest.model_validate(
            {
                "input": "D:/media/a.mp4",
                "postprocess_profile": profile,
            }
        )

        assert request.postprocess_profile == profile
