"""Readable per-call exchange logs for LLM API interactions.

Each API interaction is written as one markdown file under
``<task-artifact-dir>/exchanges/``: a small ``key: value`` metadata header,
then the prompt text per message role, then the model response — full text, no
JSON payloads. Downstream tasks (knowledge updates, prompt iteration, manual
review) read these instead of digging through ``task-artifacts.jsonl``.
"""

from __future__ import annotations

from pathlib import Path
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


class ExchangeLogger:
    """Writes one readable markdown file per LLM API interaction."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def for_task_artifact_dir(cls, task_artifact_dir: str | Path | None) -> "ExchangeLogger | None":
        if not task_artifact_dir:
            return None
        return cls(Path(task_artifact_dir).expanduser().resolve() / EXCHANGE_DIR_NAME)

    def _next_index(self) -> int:
        return len(list(self.root.glob("*.md"))) + 1

    def log(
        self,
        name: str,
        *,
        messages: List[Mapping[str, Any]] | None = None,
        response_text: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        lines: List[str] = [f"# {name}", ""]
        api_attempts = (metadata or {}).get("api_attempts") if metadata else None
        lines.extend(_render_api_attempts(api_attempts))
        for key, value in (metadata or {}).items():
            if key == "api_attempts":
                continue
            lines.append(f"- {key}: {value}")
        if metadata:
            lines.append("")
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
        path = self.root / f"{self._next_index():03d}-{name}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
