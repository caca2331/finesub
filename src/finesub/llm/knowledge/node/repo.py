"""Knowledge repository facade: the store plus the read API production uses.

``KnowledgeRepo.open(root)`` is the single entry point. The SQLite file lives
at ``<root>/knowledge.sqlite``; a root that only holds the legacy markdown
tree is imported once (lossless, plan §7.1) on first open. ``rendered/``
under the root is a derived, read-only markdown cache refreshed after every
revision so humans (and the desktop file views) still see files.

Connections are per thread: the correction drivers read entries from worker
threads, and a sqlite3 connection must not be shared across them.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

from finesub.reporting import current_reporter

from ..base import IndexEntry, _match_normalize, parse_index_text
from .human_marks import line_marks
from .importer import import_knowledge_root
from .model import NodeVersion, entry_name_problem
from .presets import (
    CATEGORIES,
    MATCHABLE_CATEGORIES,
    STANDALONE_CATEGORIES,
    preset_for_category,
)
from .render import HandleMap, render_index, render_subject, subject_aliases
from .store import KnowledgeStore

STORE_FILENAME = "knowledge.sqlite"
RENDERED_DIRNAME = "rendered"
MANIFEST_FILENAME = ".manifest.json"


def _sha256(text: str) -> str:
    # normalize line endings so an editor that saves CRLF does not read as an
    # edit by itself (content-identical text is content-identical)
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


#: The characters a file system reads as structure or rejects in a name, mapped
#: to their full-width look-alikes: ``Re:ゼロ`` renders as ``Re：ゼロ.md``. Two
#: names differing only there are one name to the store (NFKC folds them), so
#: the mapping cannot make two live entries collide.
_PATH_LOOKALIKES = str.maketrans({
    "\\": "＼", "/": "／", ":": "：", "*": "＊", "?": "？", '"': "＂", "<": "＜", ">": "＞",
})
#: Windows opens these as devices whatever the extension (``CON.md`` is the console).
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{n}" for n in (*range(1, 10), "¹", "²", "³")}
    | {f"LPT{n}" for n in (*range(1, 10), "¹", "²", "³")}
)


def rendered_file_name(subject: NodeVersion) -> str:
    """``<name>.md`` with path characters swapped for look-alikes, or
    ``<local id>.md`` when even that is no file name: a path component, a
    device name, too many bytes, or a row imported or pulled before entry
    names were checked. The H1 inside always shows the real name, and the
    harvest maps files back through the manifest, never through the name."""

    surface = str(subject.payload.get("surface") or "")
    stem = surface.translate(_PATH_LOOKALIKES)
    # The name rule counts characters; ext4 and APFS cap a path component at
    # 255 BYTES, and 100 CJK characters are 300 in UTF-8. A name that long
    # is legal as a name and would still fail every refresh as a file name.
    if (
        entry_name_problem(surface) is None
        and stem not in {".", ".."}
        and stem.split(".", 1)[0].rstrip(" ").upper() not in _WINDOWS_RESERVED_NAMES
        and len(stem.encode("utf-8")) <= _MAX_NAME_BYTES
    ):
        return f"{stem}.md"
    return f"{subject.local_id}.md"


#: Leaves room for ".md" (and an editor's swap suffix) under the 255-byte limit.
_MAX_NAME_BYTES = 240


def _is_inside(directory: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


@dataclass
class RenderedRefreshReport:
    written: list[str] = field(default_factory=list)
    dirty: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


# `CATEGORIES` (every category that exists) and `MATCHABLE_CATEGORIES` (the ones
# a name can be matched into from free text) both come from `presets.py`. Which
# of the two a call site wants is the question this module keeps answering, so
# neither is re-exported under a shorter name here.
#
# The index headers used to sit here as a dict keyed on category -- the same
# shape as the blurb dict in `render.py`, and the same failure. They are
# `preset.index_header` now (2026-09-07).

_registry: dict[tuple[str, int], "KnowledgeRepo"] = {}
_registry_lock = threading.Lock()


@dataclass(frozen=True)
class ResolvedEntry:
    category: str
    key: str
    subject_id: str


#: Separates a category from a name in the human-facing lookup
#: (``style/某字幕组``). Entry names may contain ``/`` too, so only a prefix
#: that is a known category counts (`resolve_qualified`).
CATEGORY_SEPARATOR = "/"


class AmbiguousName(LookupError):
    """A bare name that several categories answer to. The caller must ask the
    user which one rather than choose (see `KnowledgeRepo.resolve_qualified`)."""

    def __init__(self, name: str, matches: list[ResolvedEntry]) -> None:
        self.name = name
        self.matches = matches
        options = "、".join(f"{m.category}{CATEGORY_SEPARATOR}{m.key}" for m in matches)
        super().__init__(f"{name!r} 在多个类别里都有：{options}——请写成 <类别>/<名字>")


class ImportParityError(RuntimeError):
    """The lossless import did not reproduce its source byte-for-byte."""


class KnowledgeRepo:
    def __init__(self, root: Path, store: KnowledgeStore) -> None:
        self.root = root
        self.store = store

    # ---- lifecycle ------------------------------------------------------------------

    @classmethod
    def open(cls, knowledge_root: str | Path, *, auto_import: bool = True) -> "KnowledgeRepo":
        root = Path(knowledge_root).expanduser().resolve()
        key = (str(root), threading.get_ident())
        with _registry_lock:
            repo = _registry.get(key)
        if repo is not None:
            return repo
        root.mkdir(parents=True, exist_ok=True)
        store = KnowledgeStore(root / STORE_FILENAME)
        repo = cls(root, store)
        if auto_import and store.current_rev() == 0 and _has_legacy_tree(root):
            report = import_knowledge_root(root, store, task_id="auto-import")
            # Parity is the ONLY mechanical gate the migration has (plan §7).
            # It used to run in the shadow CLI only, so the import that
            # actually happened never checked itself; a failure now aborts
            # instead of leaving a half-built store behind.
            from .parity import check_parity

            parity = check_parity(store, root, rev=report.rev)
            if not parity.legacy_ok:
                store.close()
                (root / STORE_FILENAME).unlink(missing_ok=True)
                raise ImportParityError(
                    "知识库导入未通过 legacy parity，已中止并删除半成品库；"
                    f"不一致的文件：{[f.source_path for f in parity.files if not f.legacy_equal]}"
                )
            current_reporter().warning(
                "knowledge-migrated",
                f"知识库已从 Markdown 一次性导入 {root / STORE_FILENAME}"
                f"（{report.subjects} 条目，rev {report.rev}）；原 Markdown 目录保留为历史档案，不再被读取",
                action="如需核对或修改，rendered/ 下是派生的可编辑投影（改动会在下次运行时收割入库）",
            )
            repo.refresh_rendered()
        # After the import, so the styles a legacy tree brings are on the list.
        from .base_style import seed_pending

        seed_pending(store)
        with _registry_lock:
            _registry[key] = repo
        return repo

    @classmethod
    def forget(cls, knowledge_root: str | Path | None = None) -> None:
        """Drop cached connections (tests / after external rewrites)."""

        with _registry_lock:
            if knowledge_root is None:
                for repo in _registry.values():
                    repo.store.close()
                _registry.clear()
                return
            root = str(Path(knowledge_root).expanduser().resolve())
            for key in [k for k in _registry if k[0] == root]:
                _registry.pop(key).store.close()

    # ---- identity --------------------------------------------------------------------

    @property
    def rev(self) -> int:
        return self.store.current_rev()

    def version(self, rev: int | None = None) -> str:
        """Resume/checkpoint identity of the knowledge base (plan §2.5)."""

        at = self.rev if rev is None else rev
        return f"rev:{at}"

    # ---- index --------------------------------------------------------------------------

    def subjects(self, category: str | None = None, rev: int | None = None) -> list[NodeVersion]:
        return [
            subject
            for subject in self.store.subjects(rev)
            if category is None or subject.payload.get("category") == category
        ]

    def index_entries(self, category: str, rev: int | None = None) -> list[IndexEntry]:
        return parse_index_text(self.index_text(category, rev))

    def index_text(self, category: str, rev: int | None = None) -> str:
        if category not in MATCHABLE_CATEGORIES:
            # An index exists to be matched against text. A standalone
            # category is addressed by name, so an index for it would be a
            # matching surface nobody asked for.
            raise ValueError(f"no index for knowledge category {category!r}")
        if not self.subjects(category, rev):
            return ""
        header = preset_for_category(category).index_header
        return render_index(self.store, category, rev=rev, header=header)

    def resolve(
        self,
        name: str,
        rev: int | None = None,
        *,
        category: str | None = None,
        exclude: str | None = None,
    ) -> ResolvedEntry | None:
        """``exclude`` skips one subject: a rename asks "does ANOTHER entry
        answer to this name?", and the entry being renamed answers to its own
        aliases, native names and script variants of its H1."""

        needle = _match_normalize(name)
        if not needle:
            return None
        for candidate in MATCHABLE_CATEGORIES if category is None else (category,):
            for subject in self.subjects(candidate, rev):
                if exclude is not None and subject.local_id == exclude:
                    continue
                payload = subject.payload
                terms = [payload.get("surface", ""), *payload.get("native_names", []), *subject_aliases(self.store, subject.local_id, rev)]
                if needle in {_match_normalize(term) for term in terms if term}:
                    return ResolvedEntry(candidate, payload.get("surface", ""), subject.local_id)
        return None

    def resolve_in(
        self, name: str, category: str, rev: int | None = None, *, exclude: str | None = None
    ) -> ResolvedEntry | None:
        """Uniqueness lookup: does `name` already exist in the namespace this
        category belongs to?

        Two namespaces, because two rules:

        * a matchable category shares ONE namespace with the others — keys are
          globally unique there, which is what lets `create_entry` reject a
          new key and `append_lines` find it afterwards (proposals.py's
          category-hint fallback depends on it);
        * a standalone category is its own namespace — a style is usually
          named after the group or streamer it serves, so its name will often
          equal a proper-noun entry's, and neither should block the other.
          (Two same-named styles still collide: that is one namespace's own
          business, and the collision is real.)

        Sweeping every category instead would refuse the first case, which is
        exactly what `style` exists for."""

        if category in STANDALONE_CATEGORIES:
            return self.resolve(name, rev, category=category, exclude=exclude)
        return self.resolve(name, rev, exclude=exclude)

    def resolve_qualified(self, name: str, rev: int | None = None) -> ResolvedEntry | None:
        """Human-facing lookup (`knowledge show/edit/retire/ingest/repair`,
        `share mark/push`). Text-matching paths must keep calling `resolve()` —
        a style entry reached by matching is exactly what this design refuses.

        Two namespaces means a bare name can be genuinely ambiguous, and the
        design EXPECTS that: a style is usually named after the group or
        streamer it serves. So:

        * ``style/某字幕组`` addresses one category explicitly. Only a prefix
          that is a known category counts: entry names may contain ``/``
          (``Fate/stay night``), and an unknown prefix is part of the name.
        * a bare name that exists in exactly one category resolves to it;
        * a bare name that exists in several raises `AmbiguousName`. Picking a
          side silently is the one thing this must not do — it would retire,
          edit or push the entry the user did not mean.
        """

        category, separator, bare = name.partition(CATEGORY_SEPARATOR)
        if separator and category in CATEGORIES and bare.strip():
            return self.resolve(bare.strip(), rev, category=category)
        matches = [
            found
            for candidate in CATEGORIES
            for found in (self.resolve(name, rev, category=candidate),)
            if found is not None
        ]
        if len(matches) > 1:
            raise AmbiguousName(name, matches)
        return matches[0] if matches else None

    # ---- entry text ---------------------------------------------------------------------

    def entry_text(self, subject_id: str, rev: int | None = None) -> str:
        """HUMAN projection (markdown bullets on entry lines): the rendered/
        cache, ``show`` and the editor round-trip. Never inject this into a
        model prompt — that is ``entry_injection_text`` (review round 12)."""

        # FULL preview: rendered/ is what a human edits, so it carries every
        # section, the preset guidance and an empty slot per core label (plan
        # §11). The only exit that carries human-version marks: wherever the
        # model moved a line away from what a human last endorsed, the human
        # sees both (owner 2026-10-01, `human_marks.py`).
        return render_subject(
            self.store, subject_id, rev=rev, mode="human", preview="full",
            marks=line_marks(self.store, subject_id, rev),
        )

    def entry_injection_text(self, subject_id: str, rev: int | None = None) -> str:
        """PROMPT projection without handles: bare grammar lines, exactly what
        model injection paths have always carried (no bullets, no @k noise)."""

        return render_subject(self.store, subject_id, rev=rev, mode="prompt")

    def entry_prompt_text(
        self, subject_id: str, handles: HandleMap, rev: int | None = None, *, alias_row_handle: bool = True
    ) -> str:
        # the knowledge-update session decides where to write: full preview
        return render_subject(
            self.store, subject_id, rev=rev, mode="prompt", handles=handles, preview="full",
            alias_row_handle=alias_row_handle,
        )

    def load_entry_texts(self, names: Iterable[str], rev: int | None = None) -> tuple[dict[str, str], list[str]]:
        """Model-facing bulk read (research / search / correction injection):
        the PROMPT projection. Human faces read ``entry_text`` instead."""

        found: dict[str, str] = {}
        missing: list[str] = []
        for name in names:
            resolved = self.resolve(name, rev)
            if resolved is None:
                missing.append(name)
                continue
            if resolved.key not in found:
                found[resolved.key] = self.entry_injection_text(resolved.subject_id, rev)
        return found, missing

    # ---- derived markdown cache ------------------------------------------------------------

    def rendered_manifest(self) -> dict:
        path = self.root / RENDERED_DIRNAME / MANIFEST_FILENAME
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"store_rev": 0, "files": {}}

    def rendered_dirty_files(self) -> list[tuple[str, Path, str]]:
        """Rendered entry files whose on-disk content no longer matches the
        manifest hash — i.e. pending user edits (plan §11.3).

        The v3 grammar is not versioned: brackets disambiguate labels and the
        body's shape decides its kind, so an edit written against an older
        rendering parses the same way. The per-file grammar stamp the upgrade
        harvest needed retired with it."""

        out = self.root / RENDERED_DIRNAME
        dirty: list[tuple[str, Path, str]] = []
        for rel, entry in self.rendered_manifest().get("files", {}).items():
            path = out / rel
            if not _is_inside(out, path) or not path.is_file():
                # A manifest written before entry names were checked may list
                # a file outside rendered/; it is nobody's edit to harvest.
                continue
            if _sha256(path.read_text(encoding="utf-8")) != entry.get("sha256"):
                dirty.append((rel, path, str(entry.get("subject", ""))))
        return dirty

    def refresh_rendered(self, *, force: Iterable[str] = ()) -> "RenderedRefreshReport":
        """Diff-faced cache refresh (plan §11.3 / O5): only files whose target
        text changed are rewritten, and a file the USER edited since the last
        refresh (disk hash != manifest hash) is left untouched so the edit
        can be harvested — pass its relpath in ``force`` once absorbed.
        ``index.md`` is derived-only and never dirty-protected."""

        out = self.root / RENDERED_DIRNAME
        manifest = self.rendered_manifest()
        old_files: dict = manifest.get("files", {})
        forced = set(force)
        report = RenderedRefreshReport()
        new_files: dict = {}
        for category in CATEGORIES:
            directory = out / category
            directory.mkdir(parents=True, exist_ok=True)
            wanted: set[str] = set()
            if category in MATCHABLE_CATEGORIES:
                wanted.add("index.md")
                (directory / "index.md").write_text(
                    self.index_text(category)
                    or f"{preset_for_category(category).index_header}\n",
                    encoding="utf-8",
                )
            taken = {name.casefold() for name in wanted}
            for subject in self.subjects(category):
                name = rendered_file_name(subject)
                if name.casefold() in taken:
                    # The category's `index.md`, or two names a file system
                    # cannot tell apart that slipped past the name dedupe (a
                    # legacy row): the later one keeps its id for a file name.
                    name = f"{subject.local_id}.md"
                taken.add(name.casefold())
                wanted.add(name)
                rel = f"{category}/{name}"
                path = directory / name
                if not _is_inside(directory, path):
                    raise RuntimeError(f"rendered path escapes {directory}: {name!r}")
                target = self.entry_text(subject.local_id)
                entry = old_files.get(rel)
                disk = path.read_text(encoding="utf-8") if path.is_file() else None
                if (
                    disk is not None
                    and entry is not None
                    and rel not in forced
                    and _sha256(disk) != entry.get("sha256")
                ):
                    report.dirty.append(rel)
                    new_files[rel] = {**entry, "subject": subject.local_id}
                    continue
                if disk != target:
                    path.write_text(target, encoding="utf-8")
                    report.written.append(rel)
                new_files[rel] = {"sha256": _sha256(target), "subject": subject.local_id, "rev": self.rev}
            for stale in directory.glob("*.md"):
                if stale.name in wanted:
                    continue
                rel = f"{category}/{stale.name}"
                entry = old_files.get(rel)
                disk = stale.read_text(encoding="utf-8")
                if entry is not None and rel not in forced and _sha256(disk) != entry.get("sha256"):
                    report.dirty.append(rel)  # edited file of a gone subject: keep for the human
                    new_files[rel] = dict(entry)
                    continue
                stale.unlink()
                report.removed.append(rel)
        manifest_path = out / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {"store_rev": self.rev, "files": new_files},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
        return report

    def today(self) -> str:
        return date.today().isoformat()


def _has_legacy_tree(root: Path) -> bool:
    return any(
        (root / category).is_dir() and any((root / category).glob("*.md"))
        for category in MATCHABLE_CATEGORIES  # the frozen archive predates `style`
    )
