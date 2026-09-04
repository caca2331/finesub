"""Selecting a translation style and rendering it for injection.

`docs/plans/translation-style-plan.md` §2.5: selection is **task-static**. The task
names the style (`--style 某字幕组`, else `[llm] style` in the config, else
`default_style` when the store has one); nothing matches a style from the
material, and the model never picks one. Saying so explicitly — `--style ""` —
is a decision to run without any, and does NOT fall through to the default. Three reasons, all in the plan — the load-bearing one here is that a style
has no surface form to match on, so any automatic selection would have to
invent a signal that does not exist.

The rendered block is the entry's PROMPT projection (bare grammar lines), the
same face every other knowledge injection uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from finesub.config import config_str

from .node.repo import STORE_FILENAME, AmbiguousName, KnowledgeRepo

#: The category holding style entries (`node/model.py` keeps the tuples).
STYLE_CATEGORY = "style"


class StyleSelectionError(ValueError):
    """A named style that cannot be resolved to exactly one entry.

    Loud rather than silent: a run asked for a style and would otherwise
    translate the whole file without it, which looks like the style simply
    having no effect.
    """


def parse_style_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """`"甲组,乙组"` or a sequence into a tuple, order preserved, blanks dropped."""

    if value is None:
        return ()
    parts: Iterable[str]
    parts = value.split(",") if isinstance(value, str) else value
    seen: list[str] = []
    for part in parts:
        name = str(part).strip()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


def style_was_named(
    explicit: str | Sequence[str] | None = None,
    *,
    config_path: str | Path | None = None,
) -> bool:
    """Did anyone state a choice — including the choice of "none"?

    `--style ""` and `style = ""` in the config are decisions, and a decision
    to run without a style must not be answered with the implicit default.
    Comparing the resolved names against empty cannot tell the two apart
    (review 2026-09-02), so the question is asked of the INPUTS.
    """

    if explicit is not None:
        return True
    return config_str("llm", "style", path=config_path) is not None


def resolve_style_names(
    explicit: str | Sequence[str] | None = None,
    *,
    config_path: str | Path | None = None,
) -> tuple[str, ...]:
    """CLI argument → project config → nothing.

    The implicit default is NOT applied here: it depends on the store, and
    mixing it in would erase the difference between "nobody said" and "said
    none" (`resolve_style_selection` owns that decision).
    """

    if explicit is not None:
        return parse_style_names(explicit)
    return parse_style_names(config_str("llm", "style", path=config_path))


def default_style_candidate(
    knowledge_root: str | Path | None,
    *,
    rev: int | None = None,
) -> tuple[str, ...]:
    """`(DEFAULT_STYLE_NAME,)` when the store has that entry, else empty.

    ⚠ Only ever called when NOBODY named a style, and that is the whole point:
    an entry the user asked for by name must fail loudly when it is missing
    (silently running without the style you asked for looks exactly like the
    style having no effect), while the implicit default must be silent — a
    fresh install has no such entry. The first cut decided by comparing the
    resolved names against `DEFAULT_STYLE_NAME`, which cannot tell the two
    apart: `--style default_style` on a store without it was dropped without a
    word (review 2026-09-02).
    """

    # No root at all (a packaged run outside a checkout with no user data:
    # `resolve_knowledge_root(required=False)` returns None) means no store to
    # hold the default, so it is absent — not a TypeError three frames later.
    if knowledge_root is None:
        return ()
    names = (DEFAULT_STYLE_NAME,)
    # ⚠ Probing must not CREATE anything. `KnowledgeRepo.open` mkdirs the root,
    # writes an empty `knowledge.sqlite`, and may kick off the markdown
    # auto-import with its warning — all of that for a question about a store
    # that does not exist yet. A run that names no style used to touch none of
    # it, and must keep not touching it (review 2026-09-02).
    if not (Path(knowledge_root).expanduser() / STORE_FILENAME).is_file():
        return ()
    repo = KnowledgeRepo.open(knowledge_root)
    at = repo.rev if rev is None else rev
    return names if repo.resolve(DEFAULT_STYLE_NAME, at, category=STYLE_CATEGORY) else ()


#: What a run may do with the style it named, mirroring `--knowledge`'s
#: tri-state. `read` is the default because injecting is what a style is FOR,
#: while writing back changes stored data and should be asked for.
STYLE_MODES: tuple[str, ...] = ("none", "read", "update")
DEFAULT_STYLE_MODE = "read"

#: The style a run uses when nobody named one (owner 2026-09-02). Style is ON
#: by default and READ-only by default, so a base of house conventions applies
#: without being asked for — but only if it exists: a fresh install has no such
#: entry, and a missing default is silence, not an error. A name given
#: explicitly is never silently skipped (`load_style_entries` raises).
DEFAULT_STYLE_NAME = "default_style"


def resolve_style_mode(
    explicit: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> str:
    """CLI argument → `[llm] style_mode` → `read` (`CLAUDE.md`'s precedence)."""

    value = explicit if explicit is not None else config_str("llm", "style_mode", path=config_path)
    if value is None or not str(value).strip():
        return DEFAULT_STYLE_MODE
    mode = str(value).strip().lower()
    if mode not in STYLE_MODES:
        raise StyleSelectionError(
            f"--style-mode {value!r}：只能是 {'/'.join(STYLE_MODES)}"
        )
    return mode


def style_injects(mode: str) -> bool:
    return mode in ("read", "update")


@dataclass(frozen=True)
class StyleSelection:
    """What one run does with the style it named.

    Two questions come out of the same two inputs — which entries go into the
    correction prompt, and which the post-task update may write — so they are
    answered once, here, rather than at each front end. A second front end
    re-deriving `writable` from `mode` is exactly how the two would drift.
    """

    mode: str
    names: tuple[str, ...]

    @property
    def writable(self) -> tuple[str, ...]:
        return self.names if self.mode == "update" else ()


def resolve_style_selection(
    style: str | Sequence[str] | None = None,
    style_mode: str | None = None,
    *,
    knowledge_root: str | Path | None = None,
    difficulty: str = "",
    config_path: str | Path | None = None,
) -> StyleSelection:
    """`--style` + `--style-mode` → what this run injects and may write.

    `difficulty="efficiency"` turns the *unset* mode into `none`, the same rule
    `resolve_knowledge_switch` applies to the knowledge switch: the cheapest
    shape reads nothing, and a default that injects ~600 characters into every
    window would quietly contradict that. Naming a style explicitly still works
    — this only decides what "unset" means (review 2026-09-02).
    """

    if style_mode is None and difficulty == "efficiency":
        style_mode = "none"
    mode = resolve_style_mode(style_mode, config_path=config_path)
    if not style_injects(mode):
        return StyleSelection(mode=mode, names=())
    names = resolve_style_names(style, config_path=config_path)
    # Nobody stated a choice at all: fall back to the default, and only if it
    # exists. Provenance decides, not the resulting names — an explicit empty
    # value is "no style this run" (`style_was_named`).
    if not names and not style_was_named(style, config_path=config_path):
        names = default_style_candidate(knowledge_root)
    return StyleSelection(mode=mode, names=names)


def load_style_entries(
    knowledge_root: str | Path,
    names: Sequence[str],
    *,
    rev: int | None = None,
) -> list[tuple[str, str]]:
    """`[(display name, prompt projection)]` for the named styles.

    Names come from a human (an argument or the config file), so they take the
    human lookup with its qualifier: `style/某字幕组` when a proper-noun entry
    shares the name. A miss or an ambiguity raises — see `StyleSelectionError`.
    """

    if not names:
        return []
    repo = KnowledgeRepo.open(knowledge_root)
    at = repo.rev if rev is None else rev
    out: list[tuple[str, str]] = []
    for name in names:
        try:
            resolved = repo.resolve_qualified(name, at)
        except AmbiguousName as exc:
            raise StyleSelectionError(str(exc)) from exc
        if resolved is None:
            raise StyleSelectionError(
                f"--style {name!r}：知识库里没有这个条目"
                f"（用 `python -m finesub.llm.knowledge new style {name} --intro …` 建）"
            )
        if resolved.category != STYLE_CATEGORY:
            raise StyleSelectionError(
                f"--style {name!r} 指到的是 {resolved.category} 条目，不是风格"
            )
        out.append((resolved.key, repo.entry_injection_text(resolved.subject_id, at)))
    return out


def resolve_style_keys(
    knowledge_root: str | Path,
    names: Sequence[str],
    *,
    rev: int | None = None,
) -> list[str]:
    """The store keys of the named styles — what the update task pins into its
    entry selection so the model can propose into them."""

    return [key for key, _text in load_style_entries(knowledge_root, names, rev=rev)]


def render_style_block(
    knowledge_root: str | Path,
    names: Sequence[str],
    *,
    rev: int | None = None,
) -> str:
    """The system-prompt block, or `""` when no style is selected."""

    entries = load_style_entries(knowledge_root, names, rev=rev)
    if not entries:
        return ""
    parts = [
        "\n本任务的翻译风格约定（由 --style 指定；「约定」是要遵守的规则，"
        "正例/反例按同名 [标记] 对应到各自那条约定）：",
    ]
    for name, text in entries:
        parts.append(f"<style name=\"{name}\">\n{text.strip()}\n</style>")
    return "\n".join(parts) + "\n"
