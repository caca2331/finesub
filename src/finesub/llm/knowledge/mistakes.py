"""Common translation mistakes ledger under knowledge/translation/.

Unlike streamer/common entries (one file per entry with section ops), this is a
single append-only ledger plus a curated "精选" set of about 10 entries that is
injected into every correction system prompt. Updates flow only through the
post-task knowledge update via a dedicated ``<mistake_proposals>`` JSONL block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, List, Mapping, Tuple

from .base import (
    CODE_FENCE_RE,
    DEFAULT_KNOWLEDGE_ROOT,
    KnowledgeApplyRecord,
    KnowledgeApplyReport,
    commit_knowledge,
    current_month,
    ensure_knowledge_git,
    knowledge_root_path,
)


COMMON_MISTAKES_RELATIVE_PATH = Path("translation") / "common-mistake.md"
GOOD_EXAMPLES_RELATIVE_PATH = Path("translation") / "good-example.md"
MAX_FEATURED_MISTAKES = 30
MISTAKE_OPS = ("add_mistake", "set_featured", "add_example")

MISTAKE_PROPOSALS_RE = re.compile(
    r"<mistake_proposals\b[^>]*>(?P<body>.*?)</mistake_proposals>",
    re.IGNORECASE | re.DOTALL,
)
MISTAKE_ID_RE = re.compile(r"^M(?P<num>\d{4,})$")
EXAMPLE_ID_RE = re.compile(r"^G(?P<num>\d{4,})$")
ENTRY_HEADING_RE = re.compile(r"^###\s+(?P<id>M\d{4,})\s*$")
EXAMPLE_HEADING_RE = re.compile(r"^###\s+(?P<id>G\d{4,})\s*$")
FIELD_LINE_RE = re.compile(r"^-\s*(?P<name>[^:：]+)[:：]\s?(?P<value>.*)$")

_FIELD_TO_ATTR = {
    "原文片段": "source",
    "错误译文": "wrong",
    "正确译文": "correct",
    "说明": "note",
    "prompt_version": "prompt_version",
    "记录时间": "month",
}

_EXAMPLE_FIELD_TO_ATTR = {
    "原文片段": "source",
    "精修译文": "translation",
    "说明": "note",
    "记录时间": "month",
}


@dataclass(frozen=True)
class MistakeEntry:
    id: str
    source: str
    wrong: str
    correct: str
    note: str = ""
    prompt_version: str = ""
    month: str = ""


@dataclass(frozen=True)
class ExampleEntry:
    """One exemplary refined translation in translation/good-example.md."""

    id: str
    source: str
    translation: str
    note: str = ""
    month: str = ""


@dataclass(frozen=True)
class MistakeProposal:
    op: str
    source: str = ""
    wrong: str = ""
    correct: str = ""
    translation: str = ""
    note: str = ""
    prompt_version: str = ""
    ids: tuple[str, ...] = ()
    reason: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MistakeProposal":
        ids_raw = data.get("ids") or ()
        if isinstance(ids_raw, str):
            ids = tuple(part.strip() for part in re.split(r"[、,]", ids_raw) if part.strip())
        else:
            ids = tuple(str(item).strip() for item in ids_raw if str(item).strip())
        return cls(
            op=str(data.get("op", "")).strip(),
            source=str(data.get("source", "")).strip(),
            wrong=str(data.get("wrong", "")).strip(),
            correct=str(data.get("correct", "")).strip(),
            translation=str(data.get("translation", "")).strip(),
            note=str(data.get("note", "")).strip(),
            prompt_version=str(data.get("prompt_version", "")).strip(),
            ids=ids,
            reason=str(data.get("reason", "")).strip(),
            raw=dict(data),
        )

    def validation_error(self, known_ids: set[str]) -> str:
        if self.op not in MISTAKE_OPS:
            return f"unknown op {self.op!r}"
        if self.op == "add_mistake":
            for name in ("source", "wrong", "correct", "note"):
                if not getattr(self, name):
                    return f"{name} is empty"
            return ""
        if self.op == "add_example":
            for name in ("source", "translation", "note"):
                if not getattr(self, name):
                    return f"{name} is empty"
            return ""
        if not self.ids:
            return "ids is empty"
        if len(self.ids) > MAX_FEATURED_MISTAKES:
            return f"featured ids exceed the limit of {MAX_FEATURED_MISTAKES}"
        for mistake_id in self.ids:
            if not MISTAKE_ID_RE.match(mistake_id):
                return f"invalid mistake id {mistake_id!r}"
            if mistake_id not in known_ids:
                return f"unknown mistake id {mistake_id!r}"
        return ""


def common_mistakes_path(knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT) -> Path:
    return knowledge_root_path(knowledge_root) / COMMON_MISTAKES_RELATIVE_PATH


def parse_common_mistakes(text: str) -> Tuple[List[MistakeEntry], List[str]]:
    """Parse the ledger file into (entries, featured ids). Tolerant of an empty
    or missing 精选/条目 section."""
    entries: List[MistakeEntry] = []
    featured: List[str] = []
    section = ""
    current: dict[str, str] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.get("id"):
            entries.append(
                MistakeEntry(
                    id=current.get("id", ""),
                    source=current.get("source", ""),
                    wrong=current.get("wrong", ""),
                    correct=current.get("correct", ""),
                    note=current.get("note", ""),
                    prompt_version=current.get("prompt_version", ""),
                    month=current.get("month", ""),
                )
            )
        current = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            continue
        if section == "精选":
            if line.startswith("- "):
                candidate = line[2:].strip()
                if MISTAKE_ID_RE.match(candidate):
                    featured.append(candidate)
            continue
        if section == "条目":
            heading = ENTRY_HEADING_RE.match(line)
            if heading:
                flush()
                current = {"id": heading.group("id")}
                continue
            if current is None:
                continue
            match = FIELD_LINE_RE.match(line)
            if not match:
                continue
            attr = _FIELD_TO_ATTR.get(match.group("name").strip())
            if attr:
                current[attr] = match.group("value").strip().strip("`")
    flush()
    return entries, featured


def raw_featured_lines(text: str) -> List[str]:
    """The `## 精选` section exactly as written, annotations and all.

    The docs call this section human-maintained, and the auto-update path
    passes ``allow_featured=False`` so the model never touches it -- yet the
    whole file was re-rendered from parsed values, which reduced every pick to
    a bare id. `- M0001  <- 这条最重要` came back as `- M0001`, and a pick whose
    line did not match the id pattern exactly vanished from the prompt block
    entirely, with no warning.
    """

    lines: List[str] = []
    section = ""
    for raw_line in (text or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("## "):
            section = stripped[3:].strip()
            continue
        if section == "精选" and stripped:
            lines.append(raw_line.rstrip())
    return lines


def render_common_mistakes(
    entries: List[MistakeEntry],
    featured: List[str],
    *,
    featured_lines: List[str] | None = None,
) -> str:
    lines: List[str] = [
        "# 常见翻译错误",
        "",
        "条目 id 由 apply 层按 M0001 递增分配，永不复用；`## 精选` 最多 "
        f"{MAX_FEATURED_MISTAKES} 条，由人工维护并注入纠错 prompt。",
        "",
        "## 精选",
        "",
    ]
    if featured_lines is None:
        lines.extend(f"- {mistake_id}" for mistake_id in featured)
    else:
        lines.extend(featured_lines)
    lines.extend(["", "## 条目", ""])
    for entry in entries:
        lines.extend(
            [
                f"### {entry.id}",
                f"- 原文片段: `{entry.source}`",
                f"- 错误译文: `{entry.wrong}`",
                f"- 正确译文: `{entry.correct}`",
                f"- 说明: {entry.note}",
                f"- prompt_version: {entry.prompt_version}",
                f"- 记录时间: {entry.month}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def load_common_mistakes(
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
) -> Tuple[List[MistakeEntry], List[str]]:
    path = common_mistakes_path(knowledge_root)
    if not path.exists():
        return [], []
    return parse_common_mistakes(path.read_text(encoding="utf-8"))


def load_common_mistakes_text(
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    *,
    max_tokens: int = 20_000,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Raw ledger text (token-capped), for prompt/harness iteration inputs."""
    from ..token_truncate import cap_tokens

    path = common_mistakes_path(knowledge_root)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return cap_tokens(text, max_tokens, count_tokens, marker="\n...[truncated]")


def good_examples_path(knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT) -> Path:
    return knowledge_root_path(knowledge_root) / GOOD_EXAMPLES_RELATIVE_PATH


def parse_good_examples(text: str) -> List[ExampleEntry]:
    entries: List[ExampleEntry] = []
    current: dict[str, str] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.get("id"):
            entries.append(
                ExampleEntry(
                    id=current.get("id", ""),
                    source=current.get("source", ""),
                    translation=current.get("translation", ""),
                    note=current.get("note", ""),
                    month=current.get("month", ""),
                )
            )
        current = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        heading = EXAMPLE_HEADING_RE.match(line)
        if heading:
            flush()
            current = {"id": heading.group("id")}
            continue
        if current is None:
            continue
        match = FIELD_LINE_RE.match(line)
        if not match:
            continue
        attr = _EXAMPLE_FIELD_TO_ATTR.get(match.group("name").strip())
        if attr:
            current[attr] = match.group("value").strip().strip("`")
    flush()
    return entries


def render_good_examples(entries: List[ExampleEntry]) -> str:
    lines: List[str] = [
        "# 翻译范例",
        "",
        "精修对照中收录的出彩翻译（信达雅/梗与双关的示范性处理）。条目 id 由 apply 层按 "
        "G0001 递增分配，永不复用。",
        "",
        "## 条目",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"### {entry.id}",
                f"- 原文片段: `{entry.source}`",
                f"- 精修译文: `{entry.translation}`",
                f"- 说明: {entry.note}",
                f"- 记录时间: {entry.month}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def load_good_examples(
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
) -> List[ExampleEntry]:
    path = good_examples_path(knowledge_root)
    if not path.exists():
        return []
    return parse_good_examples(path.read_text(encoding="utf-8"))


def load_good_examples_text(
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    *,
    max_tokens: int = 20_000,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    """Raw example-ledger text (token-capped) for knowledge-update dedupe input."""
    from ..token_truncate import cap_tokens

    path = good_examples_path(knowledge_root)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    return cap_tokens(text, max_tokens, count_tokens, marker="\n...[truncated]")


def _next_example_id(entries: List[ExampleEntry]) -> str:
    highest = 0
    for entry in entries:
        match = EXAMPLE_ID_RE.match(entry.id)
        if match:
            highest = max(highest, int(match.group("num")))
    return f"G{highest + 1:04d}"


def render_featured_mistakes_block(
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
) -> str:
    """Fixed system-prompt block listing the curated mistakes; "" when empty."""
    entries, featured = load_common_mistakes(knowledge_root)
    by_id = {entry.id: entry for entry in entries}
    picked = [by_id[mistake_id] for mistake_id in featured if mistake_id in by_id]
    if not picked:
        return ""
    lines = ["\n常见翻译错误对照（来自历史精修反馈，遇到同类表达时避免重蹈覆辙）："]
    for idx, entry in enumerate(picked, start=1):
        note = f"（{entry.note}）" if entry.note else ""
        lines.append(
            f"{idx}. 原文「{entry.source}」曾被误译为「{entry.wrong}」，"
            f"应译为「{entry.correct}」{note}"
        )
    return "\n".join(lines) + "\n"


def parse_mistake_proposals(text: str) -> list[MistakeProposal]:
    match = MISTAKE_PROPOSALS_RE.search(text or "")
    if match:
        payload = match.group("body").strip()
    else:
        fence = CODE_FENCE_RE.search(text or "")
        payload = fence.group(1).strip() if fence else ""
    proposals: list[MistakeProposal] = []
    for line_no, raw_line in enumerate(payload.splitlines(), start=1):
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid mistake proposal JSONL at line {line_no}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Invalid mistake proposal at line {line_no}: expected object")
        proposals.append(MistakeProposal.from_mapping(data))
    return proposals


def _next_mistake_id(entries: List[MistakeEntry]) -> str:
    highest = 0
    for entry in entries:
        match = MISTAKE_ID_RE.match(entry.id)
        if match:
            highest = max(highest, int(match.group("num")))
    return f"M{highest + 1:04d}"


def _normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def apply_mistake_proposals(
    proposal_text: str,
    *,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    task_id: str = "",
    source: str = "",
    commit: bool = True,
    allow_featured: bool = True,
    evidence_text: str = "",
) -> KnowledgeApplyReport:
    """Apply ``<mistake_proposals>`` JSONL to the translation ledgers.

    ``add_mistake`` assigns the next M-id in common-mistake.md and dedupes on
    exact (source, wrong); ``add_example`` assigns the next G-id in
    good-example.md and dedupes on (source, translation); ``set_featured``
    replaces the whole 精选 list (last one wins). Commits via the same
    embedded knowledge git repo.

    The post-task knowledge update passes ``allow_featured=False``: 精选 is
    curated manually (or by a future dedicated maintenance task), so a stray
    model-emitted ``set_featured`` is skipped at the harness, not trusted to
    the prompt.

    ``evidence_text`` (the task's window material text) enables the
    anti-fabrication check: an ``add_mistake`` whose ``wrong`` cannot be found
    in it is skipped — every audited run so far contained fabricated ``wrong``
    values the model invented as plausible mistranslations. Whitespace is
    ignored when matching; an empty ``evidence_text`` disables the check.
    """
    root = knowledge_root_path(knowledge_root)
    if commit and not ensure_knowledge_git(
        root,
        snapshot_dirty=True,
        task_id=task_id,
    ):
        # ensure_knowledge_git already printed the reason. Applying without a
        # usable repository would overwrite the user's manual edits with no way
        # back, so refuse -- but as an empty report, not an exception: the
        # caller's subtitle is finished and a by-product must not undo it.
        print(
            "Warning: 知识库仓库不可用，跳过错误清单更新（未写入任何内容）。",
            file=sys.stderr,
        )
        return KnowledgeApplyReport(applied=[], skipped=[])
    path = common_mistakes_path(root)
    examples_file = good_examples_path(root)
    entries, featured = load_common_mistakes(root)
    # Kept verbatim so a human's annotations survive the re-render; dropped
    # only if a set_featured proposal actually rewrites the picks.
    original_featured_lines = raw_featured_lines(
        path.read_text(encoding="utf-8") if path.is_file() else ""
    )
    examples = load_good_examples(root)
    normalized_evidence = _normalize_for_search(evidence_text)
    applied: list[KnowledgeApplyRecord] = []
    skipped: list[KnowledgeApplyRecord] = []
    mistakes_changed = False
    examples_changed = False

    def record(proposal: MistakeProposal, status: str, reason: str) -> KnowledgeApplyRecord:
        if proposal.op == "add_mistake":
            detail = f"{proposal.source} -> {proposal.correct}"
        elif proposal.op == "add_example":
            detail = f"{proposal.source} -> {proposal.translation}"
        else:
            detail = ",".join(proposal.ids)
        is_example = proposal.op == "add_example"
        return KnowledgeApplyRecord(
            category="translation",
            entry="good-example" if is_example else "common-mistake",
            op=proposal.op,
            section=detail,
            status=status,
            reason=reason,
            target_path=str(examples_file if is_example else path),
        )

    for proposal in parse_mistake_proposals(proposal_text):
        if proposal.op == "set_featured" and not allow_featured:
            skipped.append(
                record(proposal, "skipped", "set_featured is not allowed here")
            )
            continue
        known_ids = {entry.id for entry in entries}
        error = proposal.validation_error(known_ids)
        if error:
            skipped.append(record(proposal, "skipped", error))
            continue
        if proposal.op == "add_mistake":
            if normalized_evidence and (
                _normalize_for_search(proposal.wrong) not in normalized_evidence
            ):
                skipped.append(
                    record(
                        proposal,
                        "skipped",
                        "wrong not found in task outputs (fabricated?)",
                    )
                )
                continue
            if any(
                entry.source == proposal.source and entry.wrong == proposal.wrong
                for entry in entries
            ):
                skipped.append(record(proposal, "skipped", "duplicate (source, wrong)"))
                continue
            entries.append(
                MistakeEntry(
                    id=_next_mistake_id(entries),
                    source=proposal.source,
                    wrong=proposal.wrong,
                    correct=proposal.correct,
                    note=proposal.note,
                    prompt_version=proposal.prompt_version,
                    month=current_month(),
                )
            )
            mistakes_changed = True
        elif proposal.op == "add_example":
            if any(
                entry.source == proposal.source and entry.translation == proposal.translation
                for entry in examples
            ):
                skipped.append(
                    record(proposal, "skipped", "duplicate (source, translation)")
                )
                continue
            examples.append(
                ExampleEntry(
                    id=_next_example_id(examples),
                    source=proposal.source,
                    translation=proposal.translation,
                    note=proposal.note,
                    month=current_month(),
                )
            )
            examples_changed = True
        else:
            featured = list(proposal.ids)
            original_featured_lines = None
            mistakes_changed = True
        applied.append(record(proposal, "applied", "applied"))

    committed = False
    message = ""
    changed = mistakes_changed or examples_changed
    if changed:
        if mistakes_changed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                render_common_mistakes(
                    entries, featured, featured_lines=original_featured_lines
                ),
                encoding="utf-8",
            )
        if examples_changed:
            examples_file.parent.mkdir(parents=True, exist_ok=True)
            examples_file.write_text(render_good_examples(examples), encoding="utf-8")
        if commit:
            summary = "; ".join(f"{r.op}:{r.section}" for r in applied[:5])
            if len(applied) > 5:
                summary += f"; +{len(applied) - 5} more"
            message = f"[{task_id or 'manual'}] translation-ledger {summary}"
            if source:
                message += f"\n\nsource: {source}"
            committed = commit_knowledge(root, message)

    return KnowledgeApplyReport(
        applied=applied,
        skipped=skipped,
        committed=committed,
        commit_message=message,
    )
