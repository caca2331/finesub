"""Selecting a translation style and rendering it for injection.

`docs/knowledge.md`, 「style 类别」: selection is **task-level**. The task
names the style (`--style 某字幕组`, else `[llm] style` in the config); nobody
naming one means `auto` (owner 2026-10-01: the default flipped once every
cell could pick) -- the model picks one per task from a description catalog,
never by matching the material's surface, since a style has no surface form to
match on. Saying so explicitly — `--style ""` — is a decision to run without
any of the user's styles. `difficulty=efficiency` uses only the factory default
style: it continues a style a resumed task recorded, and a style named on the
command line is refused (`resolve_style_selection`).

Whenever the settled selection names no user style, the factory default style
is injected instead (owner 2026-10-31, `FACTORY_STYLE_NAME`): the tendencies
the general correction and proofread rules used to carry live there now, so
"no style" never meant "none of them".

The rendered block is the entry's PROMPT projection (bare grammar lines), the
same face every other knowledge injection uses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from finesub import settings

from .node.base_style import warn_if_pending
from .node.repo import (
    CATEGORY_SEPARATOR,
    STORE_FILENAME,
    AmbiguousName,
    KnowledgeRepo,
)

#: The category holding style entries (`node/model.py` keeps the tuples).
STYLE_CATEGORY = "style"

#: `--style auto` (docs/knowledge.md, 「style 类别」): the preparation task picks
#: ONE candidate by its 适用范围 against the material; the harness then
#: injects that entry's full projection into every window. A real entry
#: named "auto" is addressed as `style/auto`.
AUTO_STYLE_NAME = "auto"


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
    return settings.STYLE.configured(path=config_path) is not None


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

    # The explicit value stays with the domain: `--style` may be a list, which
    # the declaration (a string in the file) does not describe.
    if explicit is not None:
        return parse_style_names(explicit)
    return parse_style_names(settings.STYLE.configured(path=config_path))


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
DEFAULT_STYLE_MODE = settings.STYLE_MODE.default

#: The style a run uses when nobody named one (owner 2026-09-02). Style is ON
#: by default and READ-only by default, so a base of house conventions applies
#: without being asked for — but only if it exists: a fresh install has no such
#: entry, and a missing default is silence, not an error. A name given
#: explicitly is never silently skipped (`load_style_entries` raises).
DEFAULT_STYLE_NAME = "default_style"

#: The factory default style (owner 2026-10-31): the recommended base taste,
#: shipped as a prompt template in the same grammar as a style entry, NOT a
#: store entry -- `--style` cannot name it, and `--style ""` is how a run asks
#: for it. It is injected whenever the settled selection names no user style,
#: and `knowledge new style` copies its lines into a new entry (copy, not
#: layering: a user's style never inherits later changes to it, so it stays
#: small and stable).
FACTORY_STYLE_NAME = "出厂默认"
FACTORY_STYLE_TEMPLATE = "style_factory_default_v1.md"

#: The sections whose lines a new style is seeded with and the migration
#: overlays -- all the factory default has: no title and no 档案, since a
#: session tells it by `<style name="出厂默认">` and who it is for is the
#: maintainer's to know (copied into a user's style it would mislead `auto`).
FACTORY_SEED_SECTIONS: tuple[str, ...] = ("约定", "正例", "反例")


def factory_style_text() -> str:
    """The factory default style's prompt projection."""

    from ..prompt_compose import load_prompt_template

    return load_prompt_template(FACTORY_STYLE_TEMPLATE).strip()


def factory_style_lines() -> dict[str, list[str]]:
    """`{section: [grammar line, ...]}` of the factory default, seed sections only."""

    out: dict[str, list[str]] = {name: [] for name in FACTORY_SEED_SECTIONS}
    section = ""
    for raw in factory_style_text().splitlines():
        line = raw.strip()
        if line.startswith("## "):
            section = line[3:].strip()
        elif line.startswith("[") and section in out:
            out[section].append(line)
    return out


def resolve_style_mode(
    explicit: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> str:
    """CLI argument → `[llm] style_mode` → `read` (`README_DEV.md` 开发原则's precedence)."""

    value = (
        explicit if explicit is not None else settings.STYLE_MODE.configured(path=config_path)
    )
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
    #: `--style auto`: `names` is empty until the preparation task decides
    #: (`StyleChoice`), then `resolved()` pins the pick for the rest of the run.
    auto: bool = False
    #: The store revision the pick was made against, carried so a resume
    #: renders the rules the earlier windows actually saw (auto-style plan §5:
    #: "续跑复用已提交的选择与实际风格输入；……改写……不导致重新抽签"). `None`
    #: = read the current revision, which is what an explicit `--style <名>`
    #: does -- there the user named the entry, not a past decision.
    rev: int | None = None
    #: Under `auto`: whether a model may be asked to pick. False is
    #: `difficulty=efficiency`, which uses only the factory default style
    #: (owner 2026-10-30): it never starts a pick and ignores the user's
    #: `default_style`, but it still continues the task's recorded choice, and
    #: with none it records "no user style" (the factory default) as one -- so a
    #: resume at another difficulty sees the same decision.
    ask: bool = True
    #: Styles read for their `[字幕长度]` alone, not injected: a correct-only task's untagged styles the command
    #: line names (`CORRECT_ONLY_TASK`).
    tier_names: tuple[str, ...] = ()

    @property
    def writable(self) -> tuple[str, ...]:
        return self.names if self.mode == "update" else ()

    @property
    def wants_write(self) -> bool:
        """The run asked to write a style back, whether or not it is named yet.

        `writable` is empty under `auto` until the preparation task decides, so
        a prerequisite check (`--knowledge update`, `--refined-srt`) has to read
        the *intent* -- checking `writable` silently skipped every `auto` run.
        """

        return self.mode == "update" and bool(self.names or self.auto)

    def resolved(self, names: Sequence[str], rev: int | None = None) -> "StyleSelection":
        return StyleSelection(
            mode=self.mode, names=tuple(names), auto=self.auto, rev=rev, ask=self.ask, tier_names=self.tier_names
        )


def resolve_style_selection(
    style: str | Sequence[str] | None = None,
    style_mode: str | None = None,
    *,
    knowledge_root: str | Path | None = None,
    difficulty: str = "",
    config_path: str | Path | None = None,
) -> StyleSelection:
    """`--style` + `--style-mode` → what this run injects and may write.

    Nobody naming a style -- no argument, nothing in the config -- is `auto`
    (auto-style-gaps plan C, owner 2026-10-01). `auto` asks nobody and costs
    nothing when the store has no choice to offer (only `default_style`, or no
    style): the orchestrator then records today's default silently.

    `difficulty="efficiency"` takes `_efficiency_selection`.
    """

    mode = resolve_style_mode(style_mode, config_path=config_path)
    if difficulty == "efficiency":
        return _efficiency_selection(style, style_mode, mode)
    if not style_injects(mode):
        return StyleSelection(mode=mode, names=())
    names = resolve_style_names(style, config_path=config_path)
    if any(name.lower() == AUTO_STYLE_NAME for name in names):
        # `auto` is an intent, not a name: it cannot be stacked with names
        # (the plan's "不允许 auto,某组 这种隐含叠加"), and an entry that really
        # is called auto is spelled `style/auto`.
        if len(names) != 1:
            raise StyleSelectionError(
                "--style auto 不能与具体名字组合；要点名一个叫 auto 的条目，写 style/auto"
            )
        return StyleSelection(mode=mode, names=(), auto=True)
    # Nobody stated a choice at all. Provenance decides, not the resulting
    # names — an explicit empty value is "no style this run" (`style_was_named`).
    if not names and not style_was_named(style, config_path=config_path):
        return StyleSelection(mode=mode, names=(), auto=True)
    return StyleSelection(mode=mode, names=names)


def _efficiency_selection(
    style: str | Sequence[str] | None, style_mode: str | None, mode: str
) -> StyleSelection:
    """`difficulty=efficiency` uses only the factory default style (owner
    2026-10-30): it injects that, asks no model and writes nothing back. The user's own choices do not reach it -- a configured
    `style` (and the store's `default_style`) is the user's default, silently
    not applied here, while a style named on the command line, `auto`
    included, is refused. A resumed task still continues the style it
    recorded: `auto` that never asks (``ask=False``) reads the task's record
    first (`correction_translation._apply_auto_style`). `--style ""` and
    `--style-mode none` say what this tier does anyway and are accepted; an
    explicit `--style-mode update` is refused, and `update` from the config
    file is a standing default for other runs, so it reads here.
    """

    if style_mode is not None and mode == "update":
        raise StyleSelectionError(
            "difficulty=efficiency（命令行或预设给的）不写回风格："
            "改用 --style-mode read，或把 difficulty 换成 intermediate"
        )
    if style is not None and parse_style_names(style):
        raise StyleSelectionError(
            "difficulty=efficiency（命令行或预设给的）只用出厂默认风格，不接受 --style 点名"
            "或 auto：去掉 --style，或把 difficulty 换成 intermediate。续跑的任务去掉 --style 照旧用它记下的风格"
        )
    if not style_injects(mode):
        return StyleSelection(mode=mode, names=())
    if style is not None:
        return StyleSelection(mode="read", names=())
    return StyleSelection(mode="read", names=(), auto=True, ask=False)


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
    A style still waiting for the factory default warns (`warn_if_pending`).
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
        warn_if_pending(repo, resolved.subject_id, resolved.key)
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
    correct_only: bool = False,
) -> str:
    """The system-prompt block of the settled selection.

    ``names`` empty is a settled "no user style", which renders the factory
    default (`FACTORY_STYLE_NAME`) -- or nothing for a ``correct_only`` task, whose
    factory default would be translation habits (`CORRECT_ONLY_TASK`). A caller whose pick is still pending (a
    fast window that picks itself, a plan previewing that pick) passes no
    block at all rather than calling this.

    A line repeated verbatim across stacked styles (`--style A,B`, both seeded
    from the factory default) is rendered once, under the first style that has
    it; a line the user rewrote is a different line and stays.

    Each style is cut by lines to the style preset's `max_entry_tokens`
    (plan kb-entry-size; per style, not for the block): the write rule that
    used to keep a style under it is gone, so the injection holds the line.
    Counted with the dependency-free upper bound -- this runs while a prompt
    is assembled, with no token counter in reach.
    """

    from ..injection_budget import fit_entry_text
    from ..prompt_compose import load_prompt_template, task_mode_texts
    from ..token_budget import HeuristicTokenCounter
    from .node.presets import preset_for_category

    if correct_only and not names:
        return ""
    entries = load_style_entries(knowledge_root, names, rev=rev) or [
        (FACTORY_STYLE_NAME, factory_style_text())
    ]
    entries = _dedupe_stacked_lines(entries)
    cap = preset_for_category(STYLE_CATEGORY).max_entry_tokens
    count = HeuristicTokenCounter().count_text
    # The examples' pairing rule only where some style has examples to pair.
    examples = any(f"## {section}" in text for _, text in entries for section in ("正例", "反例"))
    parts = [
        "\n" + task_mode_texts(correct_only)["style_header"]
        + (load_prompt_template("fragment_style_examples_pairing_v1.md").strip() if examples else "") + "）：",
    ]
    for name, text in entries:
        body, notice = (text.strip(), "") if cap is None else fit_entry_text(name, text, cap, count)
        parts.append(f"<style name=\"{name}\">\n{body}\n</style>")
        if notice:
            parts.append(notice)
    return "\n".join(parts) + "\n"


def _dedupe_stacked_lines(entries: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for name, text in entries:
        kept: list[str] = []
        section = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                section = stripped[3:].strip()
            elif section in FACTORY_SEED_SECTIONS and stripped.startswith("["):
                if stripped in seen:
                    continue
                seen.add(stripped)
            kept.append(line)
        out.append((name, "\n".join(kept)))
    return out


# --- `--style auto`: candidates, catalog and the recorded choice ---------------
# docs/knowledge.md, 「style 类别」. The model sees a DESCRIPTION catalog
# (name, intro, 作者, 适用范围 -- the profile lines the preset already has),
# picks exactly one, and the harness turns that into the injected block. The
# choice is a task result: persisted with the research context, reused on
# resume, fixed before any window starts.
#
# ⚠ The pick is forced (owner 2026-10-01, auto-style-gaps plan 「已裁定」 8):
# with something to choose from, the model must name a candidate -- and
# `default_style` is one, so "nothing specific fits" is the model choosing it.
# A missing block, a name outside the catalog or (with `default_style` in the
# store) an empty pick is a contract violation of the stage that asked, which
# retries / repairs it and fails the task when that runs out. There is no
# silent fallback: a forced pick and a fallback cannot both hold.


class StyleContractError(ValueError):
    """A `<style_selection>` that breaks the forced-pick contract.

    A ``ValueError`` so each stage's own parse / shape / validation loop treats
    it like any other malformed reply; the message is written for the model,
    which reads it back on the repair round.
    """


@dataclass(frozen=True)
class StyleCandidate:
    key: str
    intro: str = ""
    author: str = ""
    scope: str = ""


class ProfileFields(NamedTuple):
    """The 档案 section's lines a run reads, as written (empty when absent)."""

    author: str = ""
    scope: str = ""
    #: `[字幕长度]`: the style's length tier (`length_tier.TIER_RATIOS`), unchecked here.
    length: str = ""
    #: `[任务]`: the tasks the style is written for (`CORRECT_ONLY_TASK`).
    task: str = ""


#: The 档案 labels `profile_fields` reads, by field.
_PROFILE_LABELS = {"作者": "author", "适用范围": "scope", "字幕长度": "length", "任务": "task"}

#: A style whose `[任务]` names this is written for a correct-only task (a Chinese source,
#: docs/llm_harness_behavior.md「只纠错」), which injects it whole (owner 2026-11-05). Such a task injects no other
#: style: it reads one the command line names for its `[字幕长度]` alone, and none at all otherwise.
CORRECT_ONLY_TASK = "中文纠错"


def for_correct_only(prompt_text: str) -> bool:
    """Whether a rendered style entry's `[任务]` names `CORRECT_ONLY_TASK` (among others: 「翻译、中文纠错」)."""

    task = profile_fields(prompt_text).task
    return CORRECT_ONLY_TASK in re.split(r"[、，,/\s]+", task)


def profile_fields(prompt_text: str) -> ProfileFields:
    """`[作者]` / `[适用范围]` / `[字幕长度]` of the 档案 section of a rendered entry."""

    found: dict[str, str] = {}
    in_profile = False
    for raw in prompt_text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            in_profile = line[3:].strip() == "档案"
            continue
        if not in_profile or not line.startswith("["):
            continue
        label, _sep, body = line[1:].partition("]")
        field_name = _PROFILE_LABELS.get(label.strip())
        if field_name:
            found[field_name] = body.strip()
    return ProfileFields(**found)


@dataclass(frozen=True)
class StyleChoice:
    """What `--style auto` decided for one task, in the shape it is recorded.

    ``selected`` names a style; ``none`` runs without one -- the model's explicit
    "none of these" (legal only when the store has no `default_style`), the
    no-choice shortcut on a store without styles, or a record that can no longer
    be rendered.
    """

    outcome: str  # selected | none
    key: str = ""
    reason: str = ""
    rev: int | None = None
    intent: str = AUTO_STYLE_NAME

    @property
    def names(self) -> tuple[str, ...]:
        """The pick as a style *reader* must address it: category-qualified.

        ``key`` is the catalog key, and a style is usually named after the
        group or streamer it serves -- ``resolve_qualified`` would send the
        bare name into the matchable namespace and either land on that
        proper-noun entry or raise ``AmbiguousName``, failing a run over a
        pick that was never ambiguous. The category is not the model's to
        choose: the catalog only ever lists style entries.
        """

        return (f"{STYLE_CATEGORY}{CATEGORY_SEPARATOR}{self.key}",) if self.key else ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "outcome": self.outcome,
            "key": self.key,
            "reason": self.reason,
            "rev": self.rev,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StyleChoice":
        return cls(
            outcome=str(data.get("outcome") or "none"),
            key=str(data.get("key") or ""),
            reason=str(data.get("reason") or ""),
            rev=int(data["rev"]) if isinstance(data.get("rev"), int) else None,
            intent=str(data.get("intent") or AUTO_STYLE_NAME),
        )


@dataclass(frozen=True)
class StyleCatalog:
    """Every style entry of one store revision, as the model may choose from."""

    candidates: tuple[StyleCandidate, ...] = ()
    rev: int | None = None

    def __bool__(self) -> bool:
        return bool(self.candidates)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.candidates)

    @property
    def has_default(self) -> bool:
        return DEFAULT_STYLE_NAME in self.keys

    @property
    def has_choice(self) -> bool:
        """Whether there is anything to choose: a candidate besides `default_style`.

        Without one -- no styles at all, or only `default_style` -- picking
        changes nothing, so nobody is asked (auto-style-gaps plan, C 「先读记录」
        step 2): the run takes today's default without an extra call.
        """

        return any(key != DEFAULT_STYLE_NAME for key in self.keys)

    @classmethod
    def from_keys(cls, keys: Sequence[str], *, rev: int | None = None) -> "StyleCatalog":
        """The names alone -- all a validator in another process needs to judge
        a pick (`agent.agent_validators`)."""

        return cls(candidates=tuple(StyleCandidate(key=str(key)) for key in keys), rev=rev)

    def render(self) -> str:
        return "\n".join(
            f"- {c.key}｜{c.intro or '（无简介）'}｜作者：{c.author or '未记'}｜适用范围：{c.scope or '未记'}"
            for c in self.candidates
        )

    def decide(self, *, selected: str, reason: str) -> StyleChoice:
        """The model's pick → a recorded choice, or `StyleContractError`.

        A name outside the catalog is never recorded as anything: it is the
        reply that is wrong. An empty pick is the model's explicit "none of
        these" only where that is a legal answer -- with `default_style` in
        the store, the default *is* the answer to "nothing specific fits", so
        an empty pick is a missed one.
        """

        wanted = selected.strip()
        if wanted.startswith(f"{STYLE_CATEGORY}/"):
            wanted = wanted[len(STYLE_CATEGORY) + 1 :].strip()
        keys = self.keys
        if wanted and wanted in keys:
            return StyleChoice(outcome="selected", key=wanted, reason=reason.strip(), rev=self.rev)
        if wanted:
            raise StyleContractError(
                f"<style_selection> 选了不在候选里的 {wanted!r}；只能从候选名里选一个："
                + "、".join(keys)
                # without `default_style` an empty pick is the legal "none fits"
                + ("" if self.has_default else "，或填空字符串表示都对不上")
            )
        if self.has_default:
            raise StyleContractError(
                f"<style_selection> 的 selected 不能留空：对不上专门的那几套时请明确选 {DEFAULT_STYLE_NAME}"
            )
        return StyleChoice(outcome="none", key="", reason=reason.strip() or "候选都对不上", rev=self.rev)

    def choose(self, blocks: Sequence[str]) -> StyleChoice:
        """The `<style_selection>` blocks a reply carried → the choice.

        Exactly one block is the contract; none is a missed pick, never "no
        match" (that is an explicit empty `selected`, where legal).
        """

        if not blocks:
            raise StyleContractError("缺少 <style_selection> 块：本任务必须明确选定一套风格")
        if len(blocks) > 1:
            raise StyleContractError("<style_selection> 块只能有一个")
        selected, reason = parse_style_selection_block(blocks[0])
        return self.decide(selected=selected, reason=reason)


def catalog_offers_default(rendered: str) -> bool:
    """Whether a catalog rendered by `StyleCatalog.render` lists `default_style`.

    The prompt layer only ever sees the rendered text (it is also what the
    dumped preparation fixture keeps), and the forced-pick clause depends on
    whether "nothing fits" is answered by the default or by an empty pick.
    Read off the same line format `render` writes.
    """

    return any(
        line.startswith(f"- {DEFAULT_STYLE_NAME}｜") for line in (rendered or "").splitlines()
    )


# --- chunked preparation: one vote per chunk ----------------------------------
# auto-style-gaps plan B (owner 2026-10-01, 「已裁定」 1 and 6): an over-long
# material is prepared in chunks, each chunk's task picks under the forced-pick
# contract, and the pick is a vote. Most votes win; a tie is broken
# deterministically and the numbers that broke it go into the reason, because
# usage moves with time and the same votes counted later could break the tie
# differently -- which is why the result is persisted and a resume reads it.


@dataclass(frozen=True)
class StyleVoteFacts:
    """What the tie-break reads, frozen at aggregation time.

    ``usage``: style key → number of distinct tasks that injected it
    (`node.signals.style_usage_counts`); None when it cannot be read (the
    signal gate is closed, as in a worktree) and the level is skipped.
    ``last_rev``: style key → the latest revision that touched the entry, as
    of the catalog's own revision (not the store head: a style edited after
    the pick must not change a replay's answer).
    """

    usage: Mapping[str, int] | None = None
    last_rev: Mapping[str, int] = field(default_factory=dict)


def tally_style_votes(
    votes: Sequence[tuple[str, StyleChoice]],
    catalog: StyleCatalog,
    facts: StyleVoteFacts,
) -> StyleChoice:
    """``[(chunk_id, that chunk's pick)]`` → the task's one choice.

    A ``none`` vote (only possible with no `default_style` in the store) is a
    vote for "no style", counted like any other. Ties, in order: usage (more
    wins; skipped when unreadable), the entry's latest revision at the
    catalog's revision (newer wins), then the name (lexicographically first).
    "No style" tying with a style loses to it: a chunk judged that style to
    fit, while "none" only says nothing specific was found.
    """

    if not votes:
        raise ValueError("no chunk votes to tally")
    ballots: dict[str, list[str]] = {}
    for chunk_id, choice in votes:
        ballots.setdefault(choice.key, []).append(chunk_id)
    top = max(len(chunks) for chunks in ballots.values())
    tied = sorted(key for key, chunks in ballots.items() if len(chunks) == top)
    tally = "、".join(
        f"{key or '不用风格'} {len(chunks)} 票（{'、'.join(chunks)}）"
        for key, chunks in sorted(ballots.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    )
    notes: list[str] = []
    if len(tied) > 1 and "" in tied:
        tied.remove("")  # a style beats "none" on a tie
        notes.append("与「不用风格」打平时选风格")
    last_rev = dict(facts.last_rev)
    usage = None if facts.usage is None else dict(facts.usage)
    if len(tied) > 1:
        if usage is None:
            notes.append("使用率读不到，跳过")
        else:
            best = max(usage.get(key, 0) for key in tied)
            notes.append(
                "使用率：" + "、".join(f"{key} {usage.get(key, 0)}" for key in tied)
            )
            tied = [key for key in tied if usage.get(key, 0) == best]
    if len(tied) > 1:
        newest = max(last_rev.get(key, -1) for key in tied)
        notes.append(
            "最近修订（rev）："
            + "、".join(f"{key} {last_rev.get(key, '未知')}" for key in tied)
        )
        tied = [key for key in tied if last_rev.get(key, -1) == newest]
    if len(tied) > 1:
        notes.append("按名字字典序")
        tied = sorted(tied)[:1]
    winner = tied[0]
    reason = f"分块投票：{tally}" + (f"；平局：{'；'.join(notes)}" if notes else "")
    return StyleChoice(
        outcome="selected" if winner else "none", key=winner, reason=reason, rev=catalog.rev
    )


def style_vote_facts(knowledge_root: str | Path | None, catalog: StyleCatalog) -> StyleVoteFacts:
    """Read the tie-break facts for the catalog's candidates.

    ``last_rev`` is a pure read of the store at ``catalog.rev``; ``usage`` is
    telemetry behind the signal gate and degrades to None (skip the level)
    rather than failing the aggregation.
    """

    if (
        knowledge_root is None
        or not catalog
        or not (Path(knowledge_root).expanduser() / STORE_FILENAME).is_file()
    ):
        return StyleVoteFacts(usage=None, last_rev={})
    from .node.signals import style_usage_counts, subject_pack_node_ids

    repo = KnowledgeRepo.open(knowledge_root)
    at = repo.rev if catalog.rev is None else catalog.rev
    store = repo.store
    subject_of: dict[str, str] = {}
    last_rev: dict[str, int] = {}
    for key in catalog.keys:
        resolved = repo.resolve_qualified(f"{STYLE_CATEGORY}{CATEGORY_SEPARATOR}{key}", at)
        if resolved is None:
            continue
        subject_of[key] = resolved.subject_id
        revs: list[int] = []
        for node_id in subject_pack_node_ids(store, resolved.subject_id, at):
            node = store.node(node_id, at)
            if node is not None:
                revs.append(int(node.valid_from_rev))
            revs.extend(int(m.valid_from_rev) for m in store.children(node_id, at))
            revs.extend(int(item.valid_from_rev) for item in store.items_of(node_id, at))
        last_rev[key] = max(revs) if revs else -1
    usage: dict[str, int] | None
    try:
        from .base import knowledge_signals_enabled

        if knowledge_signals_enabled(knowledge_root):
            counts = style_usage_counts(store)
            usage = {key: counts.get(subject, 0) for key, subject in subject_of.items()}
        else:
            usage = None
    except Exception:  # telemetry: unreadable means "skip this level"
        usage = None
    return StyleVoteFacts(usage=usage, last_rev=last_rev)


def record_style_injected(
    knowledge_root: str | Path | None,
    names: Sequence[str],
    *,
    task_id: str,
    rev: int | None = None,
) -> None:
    """Fail-soft front of `node.signals.book_style_injected`.

    Telemetry with the shadow scan's discipline: behind the knowledge signal
    gate (closed in a worktree), and never worth failing a run over. Probing
    must not create a store either -- no names, nothing to book.
    """

    if not names or knowledge_root is None:
        return
    try:
        from .base import knowledge_root_path, knowledge_signals_enabled
        from .node.signals import book_style_injected

        if not knowledge_signals_enabled(knowledge_root):
            return
        if not (Path(knowledge_root).expanduser() / STORE_FILENAME).is_file():
            return
        book_style_injected(
            KnowledgeRepo.open(knowledge_root_path(knowledge_root)),
            names,
            task_id=task_id,
            rev=rev,
        )
    except Exception as exc:  # telemetry must never sink the run
        from finesub.reporting import current_reporter

        current_reporter().debug(f"style usage booking skipped: {exc}")


def load_style_catalog(
    knowledge_root: str | Path | None, *, rev: int | None = None, correct_only: bool = False
) -> StyleCatalog:
    """The candidates of `--style auto`; empty when there is no store at all. A ``correct_only`` task's are the
    styles written for it (`CORRECT_ONLY_TASK`).

    Like `default_style_candidate`, probing must not create a store.
    """

    if knowledge_root is None:
        return StyleCatalog()
    if not (Path(knowledge_root).expanduser() / STORE_FILENAME).is_file():
        return StyleCatalog()
    repo = KnowledgeRepo.open(knowledge_root)
    at = repo.rev if rev is None else rev
    candidates: list[StyleCandidate] = []
    for subject in repo.subjects(STYLE_CATEGORY, at):
        key = str(subject.payload.get("surface") or subject.payload.get("key") or "").strip()
        if not key:
            continue
        text = repo.entry_injection_text(subject.local_id, at)
        if correct_only and not for_correct_only(text):
            continue
        author, scope = profile_fields(text)[:2]
        candidates.append(
            StyleCandidate(
                key=key,
                intro=str(subject.payload.get("intro") or "").strip(),
                author=author,
                scope=scope,
            )
        )
    candidates.sort(key=lambda c: c.key)
    return StyleCatalog(candidates=tuple(candidates), rev=at)


def parse_style_selection_block(body: str) -> tuple[str, str]:
    """`{"selected": ..., "reason": ...}` → `(selected, reason)`.

    Garbage raises `StyleContractError`: an unreadable block is a missed pick,
    not an empty one -- reading it as `("", ...)` would turn a malformed reply
    into the legal "none of these".
    """

    try:
        data = json.loads(body.strip() or "{}")
    except ValueError:
        raise StyleContractError("<style_selection> 块不是合法 JSON") from None
    if not isinstance(data, dict):
        raise StyleContractError("<style_selection> 块不是 JSON 对象")
    if not isinstance(data.get("selected"), str):
        raise StyleContractError('<style_selection> 块缺少字符串字段 "selected"')
    return str(data["selected"]).strip(), str(data.get("reason") or "").strip()
