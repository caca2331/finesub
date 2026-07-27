from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
import json
import sys
from typing import Any, Protocol

from desktop.backend.common.models import TaskRequest
from desktop.backend.worker.protocol import EventLogWriter, WorkerEvent, encode_event


class PipelineCallable(Protocol):
    def __call__(self, source: str, **kwargs: Any) -> Any: ...


Emit = Callable[[WorkerEvent], None]


def _result_paths(paths: Any) -> dict[str, str]:
    return {
        "vocalAudio": str(paths.vocal_audio),
        "alignedJson": str(paths.aligned_json),
        "stableJson": str(paths.stable_json),
        "rawSrt": str(paths.raw_srt),
        "translatedSrt": str(paths.translated_srt),
        "finalSrt": str(paths.final_srt),
        "taskArtifactDir": str(paths.task_artifact_dir),
    }


def run_request(
    request: TaskRequest,
    *,
    task_id: str,
    pipeline: PipelineCallable,
    emit: Emit,
) -> dict[str, str]:
    emit(WorkerEvent.started(task_id))
    emit(
        WorkerEvent.progress(
            task_id,
            stage=request.stage,
            message="FineSub 任务已开始",
        )
    )
    try:
        paths = pipeline(
            request.input,
            output_path=request.output,
            stage=request.stage,
            model_name=request.model_name,
            device=request.device,
            language=request.language,
            gpu_budget_gb=request.gpu_budget_gb,
            word=request.word,
            asr_stabilize_profile=request.asr_stabilize_profile,
            llm_route=request.llm_route,
            llm_level=request.llm_level,
            llm_fast=request.llm_fast,
            llm_output_scale=request.llm_output_scale,
            extra_info=request.extra_info,
            extra_style=request.extra_style,
            enable_web_search=request.enable_web_search,
            knowledge=request.knowledge,
            task_id=task_id,
            postprocess_profile=request.postprocess_profile,
        )
    except Exception as error:
        emit(WorkerEvent.failed(task_id, str(error)))
        raise
    outputs = _result_paths(paths)
    emit(WorkerEvent.completed(task_id, outputs))
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FineSub isolated desktop worker")
    parser.add_argument("--task-id", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    request_line = sys.stdin.readline()
    if not request_line:
        raise ValueError("worker request is missing")
    request = TaskRequest.model_validate(json.loads(request_line))

    protocol_output = sys.stdout

    def emit(event: WorkerEvent) -> None:
        protocol_output.write(encode_event(event))
        protocol_output.flush()

    log_writer = EventLogWriter(args.task_id, emit)
    try:
        with redirect_stdout(log_writer), redirect_stderr(log_writer):
            from pipeline import run_pipeline

            run_request(
                request,
                task_id=args.task_id,
                pipeline=run_pipeline,
                emit=emit,
            )
        return 0
    except Exception:
        return 1
    finally:
        log_writer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
