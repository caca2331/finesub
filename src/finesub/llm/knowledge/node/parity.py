"""Shadow-phase parity (plan §7.2): legacy projection must reproduce the
source files byte-for-byte; human projection may differ only by
whitelisted normalizations, each listed for approval."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..base import parse_index_text
from .importer import EMPTY_SLOT_RE
from .render import render_index_line, render_subject, subject_aliases
from .store import KnowledgeStore

WHITELIST = (
    "trailing-whitespace", "blank-after-heading", "term-line-form", "entry-bullet",
    "full-preview-scaffold",
)

_ENTRY_BULLET_RE = re.compile(r"^- ")


@dataclass
class FileParity:
    source_path: str
    subject_id: str
    legacy_equal: bool
    human_equal: bool
    human_diff_classes: list[str] = field(default_factory=list)
    legacy_diff: str = ""
    human_diff: str = ""


@dataclass
class ParityReport:
    rev: int
    files: list[FileParity] = field(default_factory=list)
    index_mismatches: list[dict[str, Any]] = field(default_factory=list)

    @property
    def legacy_ok(self) -> bool:
        """Entry files must round-trip byte-for-byte. The index is a derived
        file today as well, so a mismatch there means the *source* index is
        stale relative to its entries -- reported, not a parity failure."""

        return all(item.legacy_equal for item in self.files)

    @property
    def human_ok(self) -> bool:
        return all(
            item.human_equal or all(cls in WHITELIST for cls in item.human_diff_classes)
            for item in self.files
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rev": self.rev,
            "legacy_ok": self.legacy_ok,
            "human_ok": self.human_ok,
            "files": [item.__dict__ for item in self.files],
            "index_mismatches": self.index_mismatches,
        }


def _project_term_line(line: str) -> str:
    """The deterministic 5-seg → four-column projection (plan §11.2 + A1):
    drop the reading column, keep the alias column third (rendered from the
    alias items the import derived from that very column), desc last."""

    parts = line.split("|")
    if len(parts) < 5:
        return line
    from .importer import split_names

    surface, zh, alias_text, _reading = parts[:4]
    desc = "|".join(parts[4:])
    aliases = "、".join(dict.fromkeys(split_names(alias_text)))
    return f"{surface}|{zh}|{aliases}|{desc}"


def _classify_human_diff(original: str, rendered: str) -> list[str]:
    """Name the kinds of difference; anything unnamed is 'other' (not whitelisted)."""

    classes: set[str] = set()
    orig_lines = original.split("\n")
    new_lines = rendered.split("\n")
    if [line.rstrip() for line in orig_lines] == [line.rstrip() for line in new_lines]:
        classes.add("trailing-whitespace")
        return sorted(classes)
    orig_nonblank = [line.rstrip() for line in orig_lines if line.strip()]
    # the full preview adds guidance comments and one empty slot per core
    # label; neither is content, so both drop out before comparing
    from .importer import strip_comments

    uncommented = strip_comments(Path("parity"), new_lines)
    new_nonblank = [line.rstrip() for line in uncommented if line.strip()]
    if len(uncommented) != len(new_lines):
        classes.add("full-preview-scaffold")
    # human mode prefixes entry lines with a markdown bullet (presentation
    # only; the edit round-trip strips it) — compare the bare grammar
    debulleted = [_ENTRY_BULLET_RE.sub("", line) for line in new_nonblank]
    if debulleted != new_nonblank:
        classes.add("entry-bullet")
    unslotted = [line for line in debulleted if not EMPTY_SLOT_RE.match(line)]
    if unslotted != debulleted:
        classes.add("full-preview-scaffold")
        debulleted = unslotted
    if orig_nonblank == debulleted:
        classes.add("blank-after-heading")
        return sorted(classes)
    if [_project_term_line(line) for line in orig_nonblank] == debulleted:
        classes.add("term-line-form")
        return sorted(classes)
    classes.add("other")
    return sorted(classes)


def check_parity(store: KnowledgeStore, source_root: str | Path, *, rev: int | None = None) -> ParityReport:
    root = Path(source_root)
    at = store.current_rev() if rev is None else rev
    report = ParityReport(rev=at)
    seen_by_category: dict[str, list[str]] = {}
    for subject in store.subjects(at):
        aux = store.migration_aux(subject.local_id)
        if aux is None or aux.layout.get("index_only"):
            continue
        original = (root / aux.source_path).read_text(encoding="utf-8")
        legacy = render_subject(store, subject.local_id, rev=at, mode="legacy")
        # Parity compares against the source markdown, which carries every
        # section — so it needs the FULL preview (plan §11). The scaffolding
        # that preview adds is normalized away by the classifier below.
        human = render_subject(store, subject.local_id, rev=at, mode="human", preview="full")
        item = FileParity(
            source_path=aux.source_path,
            subject_id=subject.local_id,
            legacy_equal=legacy == original,
            human_equal=human == original,
        )
        if not item.legacy_equal:
            item.legacy_diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True), legacy.splitlines(keepends=True), "source", "legacy"
                )
            )
        if not item.human_equal:
            item.human_diff_classes = _classify_human_diff(original, human)
            item.human_diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True), human.splitlines(keepends=True), "source", "human"
                )
            )
        report.files.append(item)
        seen_by_category.setdefault(subject.payload.get("category", ""), []).append(
            render_index_line(subject, subject_aliases(store, subject.local_id, at))
        )

    for category, rendered_lines in seen_by_category.items():
        index_path = root / category / "index.md"
        if not index_path.exists():
            continue
        original_entries = {
            entry.key: entry for entry in parse_index_text(index_path.read_text(encoding="utf-8"))
        }
        rendered_entries = {entry.key: entry for entry in parse_index_text("\n".join(rendered_lines))}
        for key, entry in original_entries.items():
            rendered = rendered_entries.get(key)
            if rendered is None or rendered != entry:
                report.index_mismatches.append(
                    {
                        "category": category,
                        "key": key,
                        "original": entry.to_line(),
                        "rendered": rendered.to_line() if rendered else None,
                    }
                )
    return report


def write_parity_report(report: ParityReport, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "parity.json").write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# Parity @rev {report.rev}",
        "",
        f"legacy byte-equal: {'OK' if report.legacy_ok else 'FAIL'}",
        f"human within whitelist: {'OK' if report.human_ok else 'FAIL'}",
        "",
    ]
    for item in report.files:
        status = "=" if item.legacy_equal else "X"
        classes = ",".join(item.human_diff_classes) or ("=" if item.human_equal else "")
        lines.append(f"- [{status}] {item.source_path}  human: {classes}")
    for mismatch in report.index_mismatches:
        lines.append(f"- source index stale: {mismatch['category']}/{mismatch['key']}")
    lines.append("")
    for item in report.files:
        if item.legacy_diff:
            lines.extend(["", f"## legacy diff: {item.source_path}", "", "```diff", item.legacy_diff.rstrip(), "```"])
        if item.human_diff and "other" in item.human_diff_classes:
            lines.extend(["", f"## human diff: {item.source_path}", "", "```diff", item.human_diff.rstrip(), "```"])
    (out / "parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
