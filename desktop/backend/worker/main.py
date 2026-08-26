from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Protocol

from finesub.reporting import quieted_libraries, reporting_to
from finesub_bootstrap.artifacts import (
    DELIVERABLE_KEY_BY_STAGE,
    DELIVERABLE_SUFFIX_BY_STAGE,
    cleanup_intermediate,
)
from finesub_bootstrap.locks import (
    LockUnavailable,
    TASK_ACTIVITY_ROOT_VARIABLE,
    active_lock_path,
    holding_activity,
    holding_lock,
    lease_record,
    task_lock_path,
    task_workspace_lock_path,
)

from desktop.backend.common.models import TaskRequest
from desktop.backend.worker.protocol import EventLogWriter, WorkerEvent, encode_event


class PipelineCallable(Protocol):
    def __call__(self, source: str, **kwargs: Any) -> Any: ...


Emit = Callable[[WorkerEvent], None]


class WorkerReporter:
    """Turn pipeline events into the protocol the launcher already speaks.

    Stage transitions become `stage` events, which is what the progress list
    renders. Everything else -- summaries, warnings, the final path -- becomes
    a log line, because that is the channel the drawer and `task-log.txt`
    share. `debug` becomes a `debug` event, which reaches the file only:
    diagnosing a desktop run should not mean reproducing it on the CLI, but
    per-group recovery detail would spend the whole drawer budget.
    """

    def __init__(self, task_id: str, emit: Emit) -> None:
        self.task_id = task_id
        self.emit = emit

    def planned(self, stages) -> None:
        return

    def stage_started(self, stage: str, *, reused: bool = False, detail: str = "") -> None:
        # The pipeline names the stage it is entering. Announcing anything
        # before that -- the requested stage, in particular -- is a claim about
        # where the run is that nothing has established: it read as every
        # earlier stage being finished the moment the task started.
        self.emit(
            WorkerEvent.progress(
                self.task_id,
                stage=stage,
                message="已有结果，跳过" if reused else (detail or "正在处理"),
                reused=reused,
            )
        )

    def progress(self, stage, *, completed, total=None, unit="", detail="") -> None:
        # Deliberately not a log line per update: the UI shows the stage, and
        # the counts would be hundreds of rows in the task log.
        return

    def summary(self, stage: str, metrics) -> None:
        body = "，".join(
            f"{label} {value}"
            for label, value in metrics.items()
            if value not in (None, 0, "")
        )
        if body:
            self._log(f"{stage}: {body}")

    def warning(self, code: str, message: str, *, impact: str = "", action: str = "") -> None:
        tail = "；".join(part for part in (impact, action) if part)
        self._log(f"Warning: {message}" + (f"（{tail}）" if tail else ""))

    def debug(self, message: str, fields=None) -> None:
        body = " ".join(f"{key}={value}" for key, value in (fields or {}).items())
        self.emit(
            WorkerEvent.debug(self.task_id, f"{message} {body}".strip())
        )

    def completed(self, output, elapsed_sec: float) -> None:
        self._log(f"完成：{output}")

    def failed(self, stage: str, message: str) -> None:
        self._log(f"失败（{stage}）：{message}")

    def _log(self, message: str) -> None:
        self.emit(WorkerEvent.log(self.task_id, message))


# Which `PipelinePaths` attribute holds each stage's subtitle. The name it is
# recorded under, and the file it resolves to, are `finesub_bootstrap.artifacts`
# -- shared with the CLI so both front ends file a subtitle the same way.
_PATHS_ATTRIBUTE_BY_STAGE = {
    "raw-srt": "raw_srt",
    "translated-srt": "translated_srt",
    "final-srt": "final_srt",
}


def _claimed_outputs() -> set[Path]:
    """Every subtitle path this machine's task index already claims as ours.

    Publishing beside the user's own media means the name we want may already
    be taken. A path we wrote before is ours to replace -- rerunning a file has
    to land on the same name, or one video accumulates a folder of copies. A
    path we have never written belongs to whoever put it there.

    The index is read directly rather than taken from the request: `TaskRequest`
    is the front end's payload and is persisted verbatim, so a field saying
    "this file is ours" would let a front end claim any path on the disk.

    A machine whose index cannot be read yields nothing, which errs towards
    writing a new name rather than overwriting a stranger's subtitle.
    """

    from finesub.paths import resolve_managed_app_paths
    from finesub_bootstrap import task_index

    # The worker is handed a task id and a request, never a layout, so it
    # resolves the installation the same way any other process without an
    # environment does: the package shipping this code, else the managed
    # location every front end shares.
    app_paths = resolve_managed_app_paths()
    if app_paths is None:
        return set()
    claimed: set[Path] = set()
    for entry in task_index.read(
        app_paths.user_data / "tasks.json", app_paths.tasks
    ):
        outputs = entry.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for value in outputs.values():
            if isinstance(value, str) and value:
                claimed.add(Path(value).expanduser().resolve())
    return claimed


def _unclaimed_destination(destination: Path) -> Path:
    """`destination`, or a free name beside it when a stranger holds that one."""

    if not destination.exists():
        return destination
    claimed = _claimed_outputs()
    if destination in claimed:
        return destination
    candidate = destination.with_name(
        f"{destination.stem}.finesub{destination.suffix}"
    )
    serial = 2
    while candidate.exists() and candidate not in claimed:
        candidate = destination.with_name(
            f"{destination.stem}.finesub.{serial}{destination.suffix}"
        )
        serial += 1
    return candidate


def _publish_subtitle(
    paths: Any,
    request: TaskRequest,
    *,
    task_id: str,
) -> dict[str, str]:
    """Publish only the requested subtitle beside a local input file."""

    attribute = _PATHS_ATTRIBUTE_BY_STAGE.get(request.stage)
    key = DELIVERABLE_KEY_BY_STAGE.get(request.stage)
    suffix = DELIVERABLE_SUFFIX_BY_STAGE.get(request.stage)
    # All three, because they are separate tables: a stage added to one and not
    # the others should publish nothing, the way an unknown stage always has,
    # rather than raise inside the worker and fail an otherwise finished task.
    if attribute is None or key is None or suffix is None:
        return {}
    generated = Path(getattr(paths, attribute)).expanduser().resolve()
    if not generated.is_file():
        raise FileNotFoundError(
            f"FineSub completed without producing {request.stage}: {generated}"
        )

    imported = Path(request.input).expanduser()
    if not imported.is_file():
        return {key: str(generated)}

    destination = _unclaimed_destination(
        imported.resolve().with_name(f"{imported.stem}{suffix}")
    )
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
    from finesub.paths import resolve_name_output_path

    return str(resolve_name_output_path(request.name))


def run_request(
    request: TaskRequest,
    *,
    task_id: str,
    pipeline: PipelineCallable,
    emit: Emit,
) -> dict[str, str]:
    emit(WorkerEvent.started(task_id))

    try:
        # "normal": the drawer and the task log get the pipeline's own report,
        # not a library's version banner or a progress bar flattened into a
        # file. Verbose detail still reaches the file, as debug events.
        with reporting_to(WorkerReporter(task_id, emit)), quieted_libraries(
            "normal"
        ):
            paths = pipeline(
                request.input,
                output_path=_resolve_output_path(request),
                stage=request.stage,
                model_name=request.model_name,
                device=request.device,
                language=request.language,
                gpu_budget_gb=request.gpu_budget_gb,
                # Opt-in on the CLI, always on here: every desktop task runs
                # the separator first, and the two-signal post-pass exists for
                # exactly that kind of noisy separated vocal. Since the
                # streaming rework it rides along on blocks the energy VAD
                # already normalized, so it costs ~1s on a full-length track --
                # not worth a setting.
                vad_silero_assist=True,
                word=request.word,
                asr_stabilize_profile=request.asr_stabilize_profile,
                split_length_scale=request.split_length_scale,
                llm_media=request.llm_media,
                llm_retrieval=request.llm_retrieval,
                llm_difficulty=request.llm_difficulty,
                llm_fast=request.llm_fast,
                llm_output_scale=request.llm_output_scale,
                extra_info=request.extra_info,
                extra_style=request.extra_style,
                knowledge=request.knowledge,
                task_id=task_id,
                postprocess_profile=request.postprocess_profile,
            )
        outputs = _publish_subtitle(paths, request, task_id=task_id)
        if request.cleanup_intermediate and outputs:
            # `outputs` is empty for the stages that produce no subtitle
            # (`vocal`/`aligned`/`stable`), and with nothing to preserve the
            # cleanup deleted the very artifact the task was asked for: a green
            # task, an empty `outputs`, and nothing to open. The desktop UI only
            # offers the subtitle stages today, so this was a bridge-contract
            # hole rather than a visible one -- but `TaskRequest` accepts all of
            # them and `validate_stage` only guards the two LLM stages.
            cleanup_intermediate(
                paths.final_srt,
                preserve=outputs.values(),
            )
    except Exception as error:
        emit(WorkerEvent.failed(task_id, str(error)))
        raise
    emit(WorkerEvent.completed(task_id, outputs))
    return outputs


@contextmanager
def _announcing_this_task(task_id: str, output: str | None = None):
    """Own this task id and announce writes for as long as we run.

    The launcher used to hold this, which said nothing in the one case it
    exists for: closing the window does not stop us -- we are our own process
    group -- so the lock was released while this process kept writing, and
    `finesub relocate` and the task-outputs migration were free to move the
    tree out from under it. Held here it stays true even when we are orphaned,
    and the OS drops it when we actually end.

    The per-task lock is mandatory: without it, retrying from two front ends
    can corrupt the same artifacts. The tree-wide announcement remains best
    effort because unrelated tasks are allowed to run concurrently. Only
    taking that advisory lock is guarded; errors from the run itself must
    propagate unchanged.
    """

    tasks_root = os.environ.get("FINESUB_TASKS_ROOT", "")
    activity_root = os.environ.get(TASK_ACTIVITY_ROOT_VARIABLE, "")
    if not tasks_root:
        yield None
        return
    if not activity_root:
        raise RuntimeError(
            f"{TASK_ACTIVITY_ROOT_VARIABLE} is required for a managed task"
        )
    with ExitStack() as stack:
        stack.enter_context(holding_activity(Path(activity_root)))
        root = Path(tasks_root)
        root.mkdir(parents=True, exist_ok=True)
        stack.enter_context(
            holding_lock(
                task_lock_path(root, task_id),
                timeout=5,
                lease=lease_record(task_id, "desktop"),
            )
        )
        if output:
            # Fixed order across CLI and workers: identity before workspace.
            # Two task ids may intentionally reuse one output directory.
            stack.enter_context(
                holding_lock(
                    task_workspace_lock_path(root, output), timeout=5
                )
            )
        try:
            stack.enter_context(holding_lock(active_lock_path(root), timeout=5))
        except (OSError, LockUnavailable):
            pass
        yield None


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
        with _announcing_this_task(args.task_id, request.output), redirect_stdout(
            log_writer
        ), redirect_stderr(log_writer):
            from finesub.pipeline import run_pipeline

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
