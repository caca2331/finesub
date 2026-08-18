"""Readable per-call exchange logs for LLM API interactions.

Each API interaction is written as one markdown file under
``<task-artifact-dir>/exchanges/``: a small ``key: value`` metadata header,
then the prompt text per message role, then the model response — full text, no
JSON payloads. Downstream tasks (knowledge updates, prompt iteration, manual
review) read these instead of digging through ``task-artifacts.jsonl``.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Iterable, List, Mapping, Sequence

EXCHANGE_DIR_NAME = "exchanges"


def messages_to_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render chat messages as plain readable text (prompt dumps, dry runs)."""

    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


def _message_text_parts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _message_text_parts(item)
        return
    if isinstance(value, Mapping):
        if value.get("type") == "file" or "fileData" in value or "file_data" in value:
            file_info = value.get("file") if isinstance(value.get("file"), Mapping) else value
            filename = ""
            if isinstance(file_info, Mapping):
                filename = str(
                    file_info.get("filename") or file_info.get("file_id") or ""
                )
            yield f"[附件文件: {filename or '（未知）'}]"
            return
        if value.get("type") == "text":
            yield str(value.get("text", ""))
            return
        if "text" in value and len(value) == 1:
            yield str(value.get("text", ""))
            return
        for item in value.values():
            yield from _message_text_parts(item)
        return
    yield str(value)


def render_message_text(content: Any) -> str:
    return "\n".join(part for part in _message_text_parts(content) if part)


def _escape_table_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _render_api_attempts(attempts: Any) -> List[str]:
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        return []
    rows = [item for item in attempts if isinstance(item, Mapping)]
    if not rows:
        return []
    lines = [
        "## API Calls",
        "",
        "| api provider&tier | model | api key name | call # for this api key and model | return code | time when call made | time when response returned | time elapsed |",
        "|---|---|---:|---:|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                _escape_table_cell(row.get(key, ""))
                for key in (
                    "provider_tier",
                    "model",
                    "api_key_name",
                    "call_number_for_api_key_and_model",
                    "return_code",
                    "started_at",
                    "returned_at",
                    "elapsed_sec",
                )
            )
            + " |"
        )
    lines.append("")
    return lines


# Rendered as their own section instead of a `- key: value` line.
_SECTION_RENDERED_KEYS = frozenset(
    {
        "api_attempts",
        "execution_attempts",
        "validation_errors",
        "validation_warnings",
        # Derived from validation_errors, so it only ever restated a subset of
        # the section below it.
        "validation_locations",
    }
)

# Enough to see the pattern without burying the exchange. The full lists are
# always in the task artifacts (`correction_window_response`).
_VALIDATION_RENDER_LIMIT = 25


def _render_validation_group(label: str, items: Any) -> List[str]:
    entries = [str(item).strip() for item in (items or ()) if str(item).strip()]
    if not entries:
        return []
    lines = [f"**{label} ({len(entries)})**", ""]
    for entry in entries[:_VALIDATION_RENDER_LIMIT]:
        lines.append(f"- {entry}")
    remaining = len(entries) - _VALIDATION_RENDER_LIMIT
    if remaining > 0:
        lines.append(f"- （另有 {remaining} 条，见 task artifacts）")
    lines.append("")
    return lines


def _render_validation(metadata: Mapping[str, Any] | None) -> List[str]:
    """The reason a window failed, next to the response that failed it.

    It was only ever written to `correction-windows.jsonl`, so reading an
    exchange told you `validation_ok: False` and nothing about why -- the one
    question the file exists to answer.
    """

    if not metadata:
        return []
    body = _render_validation_group("errors", metadata.get("validation_errors"))
    body += _render_validation_group("warnings", metadata.get("validation_warnings"))
    if not body:
        return []
    return ["## Validation", ""] + body


def _render_execution_attempts(attempts: Any) -> List[str]:
    if not isinstance(attempts, Sequence) or isinstance(attempts, (str, bytes)):
        return []
    rows = [item for item in attempts if isinstance(item, Mapping)]
    if not rows:
        return []
    # `driver` and `session id` only ever carry a value on local-agent rows.
    # They earn a column because together they are what lets a person open the
    # vendor's *own* record of this exact call -- Claude Code keeps one file per
    # session id, agy keys its conversation database by it -- and neither is
    # recoverable from anything else here once a successful call's capsule has
    # been deleted.
    lines = [
        "## Execution Attempts",
        "",
        "| target | backend | model | driver | session id | return code | started | returned | duration ms | capsule |",
        "|---|---|---|---|---|---|---|---|---:|---|",
    ]
    for row in rows:
        driver = " ".join(
            part
            for part in (str(row.get("driver") or ""), str(row.get("driver_version") or ""))
            if part
        )
        lines.append(
            "| "
            + " | ".join(
                _escape_table_cell(value)
                for value in (
                    row.get("target_id", ""),
                    row.get("backend", ""),
                    row.get("reported_model", ""),
                    driver,
                    row.get("session_id", ""),
                    row.get("return_code", ""),
                    row.get("started_at", ""),
                    row.get("returned_at", ""),
                    row.get("duration_ms", ""),
                    row.get("capsule_id", ""),
                )
            )
            + " |"
        )
    lines.append("")
    return lines


class ExchangeLogger:
    """Writes one readable markdown file per LLM API interaction."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Index allocation is a lock + monotonic counter seeded from the
        # directory once. The old per-call ``len(glob)+1`` handed two
        # concurrent writers the same number, and the later write silently
        # replaced the earlier exchange file (docs/llm_harness_behavior.md).
        self._index_lock = threading.Lock()
        self._next: int | None = None

    @classmethod
    def for_task_artifact_dir(cls, task_artifact_dir: str | Path | None) -> "ExchangeLogger | None":
        if not task_artifact_dir:
            return None
        return cls(Path(task_artifact_dir).expanduser().resolve() / EXCHANGE_DIR_NAME)

    def _next_index(self) -> int:
        with self._index_lock:
            if self._next is None:
                self._next = len(list(self.root.glob("*.md")))
            self._next += 1
            return self._next

    def reserve_block(self) -> "ExchangeBlock":
        """Claim the next major index at *scheduling* time.

        Under parallel dispatch the completion order is nondeterministic, so
        numbering at ``log()`` time made two identical runs produce different
        filenames. The scheduler claims one block per window in window order
        before dispatch (plan A.6 "调度时按窗口 id 定序分配"); calls within a
        block sub-number in call order, which is sequential per window.
        """

        return ExchangeBlock(self, self._next_index())

    def log(
        self,
        name: str,
        *,
        messages: List[Mapping[str, Any]] | None = None,
        response_text: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        return self._write(
            f"{self._next_index():03d}-{name}",
            name,
            messages=messages,
            response_text=response_text,
            metadata=metadata,
        )

    def _write(
        self,
        stem: str,
        name: str,
        *,
        messages: List[Mapping[str, Any]] | None = None,
        response_text: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        lines: List[str] = [f"# {name}", ""]
        api_attempts = (metadata or {}).get("api_attempts") if metadata else None
        execution_attempts = (
            (metadata or {}).get("execution_attempts") if metadata else None
        )
        lines.extend(_render_api_attempts(api_attempts))
        lines.extend(_render_execution_attempts(execution_attempts))
        for key, value in (metadata or {}).items():
            if key in _SECTION_RENDERED_KEYS:
                continue
            lines.append(f"- {key}: {value}")
        if metadata:
            lines.append("")
        lines.extend(_render_validation(metadata))
        if messages is None:
            lines.append("## 请求")
            lines.append("")
            lines.append("（本次运行未留存请求文本）")
            lines.append("")
        else:
            for message in messages:
                role = str(message.get("role", "user"))
                lines.append(f"## 请求（{role}）")
                lines.append("")
                lines.append(render_message_text(message.get("content", "")).strip())
                lines.append("")
        lines.append("## 模型响应")
        lines.append("")
        lines.append((response_text or "").strip())
        lines.append("")
        path = self.root / f"{stem}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


class ExchangeBlock:
    """A deterministically numbered slice of an :class:`ExchangeLogger`.

    Files land as ``{major:03d}-{minor:02d}-{name}.md``: the major index was
    claimed at scheduling time (window order), the minor index counts this
    block's calls (query round, correction attempts) in their per-window
    sequential order. Duck-typed to ``ExchangeLogger.log`` so round runners
    accept either.
    """

    def __init__(self, logger: ExchangeLogger, index: int) -> None:
        self._logger = logger
        self.index = index
        self._sub = 0
        self._sub_lock = threading.Lock()

    def log(
        self,
        name: str,
        *,
        messages: List[Mapping[str, Any]] | None = None,
        response_text: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        with self._sub_lock:
            self._sub += 1
            sub = self._sub
        return self._logger._write(
            f"{self.index:03d}-{sub:02d}-{name}",
            name,
            messages=messages,
            response_text=response_text,
            metadata=metadata,
        )
