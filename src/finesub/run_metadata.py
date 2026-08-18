"""Pipeline-wide run metadata stored beside the primary artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


RUN_METADATA_SCHEMA_VERSION = 1


def metadata_path_for_output(output_path: str | Path) -> Path:
    output = Path(output_path).expanduser()
    base = output.with_suffix("")
    return base.with_name(f"{base.name}-metadata.json")


def load_run_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    if not source.exists():
        return {}
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def update_run_metadata(
    path: str | Path,
    patch: Mapping[str, Any],
) -> dict[str, Any]:
    """Recursively merge ``patch`` into the sidecar and write it atomically."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = load_run_metadata(target)
    data.setdefault("schema_version", RUN_METADATA_SCHEMA_VERSION)
    _merge_dict(data, patch)
    data["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    rendered = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
        handle.flush()
    try:
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return data


#: Sidecar key listing files a run created that nothing else can name.
#:
#: A decoded copy of the input (`<source stem>-decoded.flac`) is deleted the
#: moment the stage that made it succeeds, so this exists for the other case: a
#: run that died leaves one behind, under the *source* media's stem and in
#: whichever directory that stage was writing to -- neither derivable from the
#: delivered subtitle, which is all a later cleanup has to go on. So the run
#: writes the path down as soon as the file exists.
SCRATCH_FILES_KEY = "scratch_files"


def record_scratch_file(path: str | Path, scratch: str | Path) -> None:
    """Note a scratch file in the sidecar, so a failed run's copy can be found."""

    target = Path(path).expanduser()
    recorded = load_run_metadata(target).get(SCRATCH_FILES_KEY)
    known = [item for item in recorded if isinstance(item, str)] if isinstance(recorded, list) else []
    value = str(Path(scratch).expanduser().resolve())
    if value in known:
        return
    update_run_metadata(target, {SCRATCH_FILES_KEY: [*known, value]})


def scratch_files(path: str | Path) -> list[str]:
    recorded = load_run_metadata(path).get(SCRATCH_FILES_KEY)
    if not isinstance(recorded, list):
        return []
    return [item for item in recorded if isinstance(item, str)]


def stage_record(
    *,
    status: str,
    elapsed_sec: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": status}
    if elapsed_sec is not None:
        result["elapsed_sec"] = round(max(0.0, float(elapsed_sec)), 3)
    return result


def summarize_llm_rounds(artifact_dir: str | Path) -> list[dict[str, Any]]:
    """Aggregate retained API attempts by stable logical round.

    The span runs from the first provider attempt until the last retained
    artifact for the round, so validation/format retries and the work between
    attempts are included. Detailed provider rows remain in exchanges/.
    """

    path = Path(artifact_dir).expanduser() / "task-artifacts.jsonl"
    if not path.exists():
        return []
    groups: dict[str, dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            continue
        round_name = _logical_round_name(str(record.get("kind") or ""), payload)
        if not round_name:
            continue
        group = groups.setdefault(
            round_name,
            {
                "round": round_name,
                "attempts": [],
                "first_started_at": "",
                "last_finished_at": "",
                "last_artifact_at": "",
                "status": "failed",
            },
        )
        created_at = str(record.get("created_at") or "")
        if created_at and created_at > group["last_artifact_at"]:
            group["last_artifact_at"] = created_at
        logical_started = str(payload.get("logical_started_at") or "")
        if logical_started and (
            not group["first_started_at"]
            or logical_started < group["first_started_at"]
        ):
            group["first_started_at"] = logical_started
        attempts = (
            payload.get("execution_attempts") or payload.get("api_attempts") or []
        )
        if isinstance(attempts, list):
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    continue
                item = dict(attempt)
                group["attempts"].append(item)
                started = str(item.get("started_at") or "")
                finished = str(item.get("returned_at") or "")
                if started and (
                    not group["first_started_at"] or started < group["first_started_at"]
                ):
                    group["first_started_at"] = started
                if finished and finished > group["last_finished_at"]:
                    group["last_finished_at"] = finished
        if _record_succeeded(str(record.get("kind") or ""), payload):
            group["status"] = "completed"

    rows: list[dict[str, Any]] = []
    for group in groups.values():
        attempts = group.pop("attempts")
        first = _parse_iso(group.pop("first_started_at"))
        last_api = _parse_iso(group.pop("last_finished_at"))
        last_artifact = _parse_iso(group.pop("last_artifact_at"))
        end = max((item for item in (last_api, last_artifact) if item), default=None)
        api_sec = sum(
            (
                float(item.get("elapsed_sec") or 0.0)
                if item.get("elapsed_sec") is not None
                else float(item.get("duration_ms") or 0.0) / 1000.0
            )
            for item in attempts
            if isinstance(item.get("elapsed_sec"), (int, float))
            or isinstance(item.get("duration_ms"), (int, float))
        )
        attempt_count = len(attempts)
        row = {
            "round": group["round"],
            "elapsed_sec": (
                round(max(0.0, (end - first).total_seconds()), 3)
                if first is not None and end is not None
                else 0.0
            ),
            "api_sec": round(api_sec, 3),
            "api_attempts": attempt_count,
            "retries": max(0, attempt_count - 1),
            "status": (
                group["status"] if attempt_count else "reused"
            ),
        }
        rows.append(row)
    return sorted(rows, key=lambda item: str(item["round"]))


def _logical_round_name(kind: str, payload: Mapping[str, Any]) -> str:
    if kind == "research_round1_response":
        return "research-r1"
    if kind == "fast_round1_response":
        return "fast-r1"
    if kind == "research_round2_response":
        return "research-r2"
    if kind == "search_loop_round":
        session = str(payload.get("session") or "")
        prefix = "fast-search" if session.startswith("fast-") else "research-search"
        return f"{prefix}-round{payload.get('round', 0)}"
    if kind.startswith("correction_query_"):
        return f"correction-{payload.get('chunk_id', '?')}-query"
    if kind.startswith("correction_window_"):
        return f"correction-{payload.get('chunk_id', '?')}-answer"
    if kind == "knowledge_update_response":
        return f"knowledge-update-chunk{int(payload.get('chunk') or 0):02d}"
    return ""


def _record_succeeded(kind: str, payload: Mapping[str, Any]) -> bool:
    # One definition of "this record is an LLM response, not a search ledger":
    # `search_loop_round` carries both, and the two readers must not drift.
    from finesub.llm.exchange_metadata import is_session_response_record

    if not kind.endswith("_response") and kind != "search_loop_round":
        return False
    if payload.get("call_error") or payload.get("error_type"):
        return False
    if kind == "search_loop_round" and not is_session_response_record(kind, payload):
        return False
    if kind == "correction_window_response":
        return bool(payload.get("validation_ok")) and not bool(payload.get("output_limited"))
    return not bool(payload.get("parse_error"))


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _merge_dict(
    target: dict[str, Any],
    patch: Mapping[str, Any],
    path: tuple[str, ...] = (),
) -> None:
    for key, value in patch.items():
        child_path = (*path, key)
        if (
            len(child_path) == 3
            and child_path[:2] == ("timing", "stages")
            and isinstance(value, Mapping)
        ):
            # A stage record is one coherent snapshot. Recursive merging would
            # turn {"status": "reused"} into the contradictory
            # {"status": "reused", "elapsed_sec": <old execution>}.
            target[key] = dict(value)
        elif isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value, child_path)
        else:
            target[key] = value
