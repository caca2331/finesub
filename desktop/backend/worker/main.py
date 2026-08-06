from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Protocol

from desktop.backend.common.models import TaskRequest
from desktop.backend.worker.protocol import EventLogWriter, WorkerEvent, encode_event


class PipelineCallable(Protocol):
    def __call__(self, source: str, **kwargs: Any) -> Any: ...


Emit = Callable[[WorkerEvent], None]


_SUBTITLE_BY_STAGE = {
    "raw-srt": ("rawSrt", "raw_srt", "-raw.srt"),
    "translated-srt": (
        "translatedSrt",
        "translated_srt",
        "-translated.srt",
    ),
    "final-srt": ("finalSrt", "final_srt", ".srt"),
}


def _publish_subtitle(
    paths: Any,
    request: TaskRequest,
    *,
    task_id: str,
) -> dict[str, str]:
    """Publish only the requested subtitle beside a local input file."""

    subtitle = _SUBTITLE_BY_STAGE.get(request.stage)
    if subtitle is None:
        return {}
    key, attribute, suffix = subtitle
    generated = Path(getattr(paths, attribute)).expanduser().resolve()
    if not generated.is_file():
        raise FileNotFoundError(
            f"FineSub completed without producing {request.stage}: {generated}"
        )

    imported = Path(request.input).expanduser()
    if not imported.is_file():
        return {key: str(generated)}

    destination = imported.resolve().with_name(f"{imported.stem}{suffix}")
    if destination != generated:
        temporary = destination.with_name(
            f".{destination.name}.{task_id}.part"
        )
        temporary.unlink(missing_ok=True)
        try:
            shutil.copy2(generated, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {key: str(destination)}


def _resolve_output_path(request: TaskRequest) -> str | None:
    """Map the request's naming choice onto the pipeline's output path.

    ``output`` is an explicit path and wins. ``name`` is the CLI's ``--name``:
    a bare stem that produces out/<name>/<name>.srt, leaving the location alone.
    """

    if request.output:
        return request.output
    if not request.name:
        return None
    from asr_playground.paths import resolve_name_output_path

    return str(resolve_name_output_path(request.name))


def _cleanup_intermediate_outputs(paths: Any, *, preserve: set[Path]) -> None:
    """Remove the bulky private artifacts of one successful task.

    Two things are never removed, whatever the user asked for:

    * ``*-stable.json`` -- the ASR result every later stage reads. Delete it and
      a rerun redoes separation and recognition from the audio, and a standalone
      correction/translation pass has no input at all.
    * the task artifact directory -- LLM session and window checkpoints live
      there, so removing it turns a resumable task into a restart.

    They are small next to the vocal audio and the source copy, which are what
    actually fill a disk.
    """

    final_srt = Path(paths.final_srt).expanduser().resolve()
    vocal_audio = Path(paths.vocal_audio).expanduser().resolve()
    files = {
        vocal_audio,
        vocal_audio.with_suffix(".flac"),
        Path(paths.aligned_json).expanduser().resolve(),
        Path(paths.raw_srt).expanduser().resolve(),
        Path(paths.translated_srt).expanduser().resolve(),
        final_srt,
        Path(paths.metadata_json).expanduser().resolve(),
        final_srt.with_name(f"{final_srt.stem}-source.ogg"),
    }
    for path in files - preserve:
        path.unlink(missing_ok=True)

    # rmdir only succeeds on an empty directory, so the kept artifacts also keep
    # the run directory -- exactly the intent.
    try:
        final_srt.parent.rmdir()
    except OSError:
        pass


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
            output_path=_resolve_output_path(request),
            stage=request.stage,
            model_name=request.model_name,
            device=request.device,
            language=request.language,
            gpu_budget_gb=request.gpu_budget_gb,
            # Opt-in on the CLI, always on here: every desktop task runs the
            # separator first, and the two-signal post-pass exists for exactly
            # that kind of noisy separated vocal. Since the streaming rework it
            # rides along on blocks the energy VAD already normalized, so it
            # costs ~1s on a full-length track -- not worth a setting.
            vad_silero_assist=True,
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
        outputs = _publish_subtitle(paths, request, task_id=task_id)
        if request.cleanup_intermediate:
            _cleanup_intermediate_outputs(
                paths,
                preserve={Path(path).resolve() for path in outputs.values()},
            )
    except Exception as error:
        emit(WorkerEvent.failed(task_id, str(error)))
        raise
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
            from asr_playground.pipeline import run_pipeline

            run_request(
                request,
                task_id=args.task_id,
                pipeline=run_pipeline,
                emit=emit,
            )
        return 0
    except Exception as error:
        traceback.print_exc(file=log_writer)
        log_writer.flush()
        emit(
            WorkerEvent.failed(
                args.task_id,
                f"{type(error).__name__}: {error}",
            )
        )
        return 1
    finally:
        log_writer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
