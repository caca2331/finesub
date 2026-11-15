"""Preset loader (plan §3): presets are versioned TOML data shipped with the
package, not prompt text.

Two generations coexist on purpose (kb-line-grammar plan §8):

* ``version = 1`` — the legacy shape (``tier`` / ``kinds`` / ``line_form``),
  kept only under ``presets/legacy/`` so Phase A can import the markdown
  archive against the grammar it was written in.
* ``version = 2`` — the current shape (``body_kinds`` / ``labels`` /
  ``purpose`` / ``exclude`` / ``share`` / ``verify``). Since these fields now
  gate SHARING and EXTERNAL VERIFICATION, the loader validates strictly:
  every deviation raises instead of being coerced (``core = "false"`` through
  ``bool()`` would silently read as True — review 2026-08-29 P2-4).
"""

from __future__ import annotations

import tomllib
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

from .model import METADATA_SECTION

BODY_KINDS: tuple[str, ...] = ("note", "term")
SHARE_VALUES: tuple[str, ...] = ("inherit", "local")
VERIFY_VALUES: tuple[str, ...] = ("external", "none")
LABEL_ROLES: tuple[str, ...] = ("identity", "aliases")

#: Which categories exist, in the order the sweeps visit them. This tuple is the
#: only ORDERED declaration; everything else about a category -- what it is, what
#: it collects, its template, and whether it joins the matching surface -- lives
#: in its own ``presets/<name>.toml``. `test_llm_knowledge_presets.py` ties the
#: two together: the directory and this tuple must name the same set.
#:
#: Ordered here rather than globbed from the directory because a glob is
#: alphabetical, while this order is the one `repo.resolve()` sweeps in and
#: `repo` writes the index files in. Deriving it would have silently reordered
#: both -- the kind of change that produces no error and no test failure.
CATEGORIES: tuple[str, ...] = ("streamer", "common", "style")

_PRESET_KEYS = frozenset({
    "name", "version", "strict_sections", "sections", "default_section", "share_inherit",
    "allow_custom_labels", "unknown_label_share", "unknown_label_verify",
    "max_entry_tokens", "soft_entry_tokens", "splittable", "blurb", "matchable", "index_header",
})
_SECTION_KEYS = frozenset({
    "name", "body_kinds", "purpose", "exclude", "staging", "verify", "labels",
    "max_lines", "max_body_chars", "labels_from",
})
_LABEL_KEYS = frozenset({"name", "core", "role", "share", "verify", "note", "values", "model_hidden"})
_LEGACY_PRESET_KEYS = frozenset({
    "name", "version", "strict_sections", "sections", "default_section", "share_inherit",
})
_LEGACY_SECTION_KEYS = frozenset({"name", "tier", "kinds", "line_form"})


class PresetError(ValueError):
    """A preset file that cannot be trusted. Never coerced, never defaulted."""


@dataclass(frozen=True)
class LabelSpec:
    """One registered label. Registration and ``core`` are separate things:
    ``core`` decides whether the full preview renders an empty slot, while
    registration exists to carry ``role`` / ``share`` / ``verify``."""

    name: str
    core: bool = False
    role: str | None = None
    share: str | None = None
    verify: str | None = None
    note: str = ""
    #: The bodies a line under this label may have; empty = free text. A line
    #: with another body is refused where lines are classified
    #: (`importer.classify_line`), so a model's proposal is skipped with the
    #: reason and a human edit is refused or parked.
    values: tuple[str, ...] = ()
    #: A label only people write: its note stays out of the model's full preview (`render.section_comment`), so
    #: no model is told it exists. The knowledge update also holds every change to it (`proposals.hidden_label_gate`).
    model_hidden: bool = False


@dataclass(frozen=True)
class SectionSpec:
    #: Write-side caps (`docs/knowledge.md`, 「style 类别」). They live on
    #: the preset because that is where every other section rule lives: the
    #: prompt renders its guidance from here and the apply engine validates
    #: against the same object, so a cap cannot be stated twice and drift.
    #: `None` means uncapped, which is what every category but `style` is.
    #:
    #: A cap belongs on the WRITE side on purpose: sorting or truncating at
    #: read time only hides the excess, while a cap forces the choice — to add
    #: one convention you must retire, merge or shorten another.
    name: str
    body_kinds: tuple[str, ...] = ("note",)
    purpose: str = ""
    exclude: str = ""
    staging: bool = False
    verify: str = "none"
    max_lines: int | None = None
    max_body_chars: int | None = None
    #: This section's labels are REFERENCES into another section: a non-empty
    #: label here must name a line there. That is how a style entry pairs a
    #: convention with its example without a container the model has no way to
    #: express (`docs/knowledge.md`, 「style 类别」) — and it is what makes
    #: an orphan example (a label naming no convention) mechanically
    #: detectable rather than a rule only the prompt states.
    labels_from: str | None = None
    labels: tuple[LabelSpec, ...] = ()
    # legacy (version 1) fields — only populated by presets/legacy/*.toml
    tier: str = "optional"
    kinds: tuple[str, ...] = ()
    line_form: str = "note"

    def label(self, name: str) -> LabelSpec | None:
        wanted = _normalize(name)
        for spec in self.labels:
            if _normalize(spec.name) == wanted:
                return spec
        return None

    def core_labels(self) -> tuple[LabelSpec, ...]:
        return tuple(spec for spec in self.labels if spec.core)


@dataclass(frozen=True)
class Preset:
    name: str
    version: int
    strict_sections: bool
    sections: tuple[SectionSpec, ...]
    default_section: SectionSpec | None
    share_inherit: dict[str, str] = field(default_factory=dict)
    allow_custom_labels: bool = True
    unknown_label_share: str = "local"
    unknown_label_verify: str = "none"
    #: Injection cap on the WHOLE entry's prompt projection, in tokens: an
    #: injected entry longer than this is cut line by line to fit
    #: (`injection_budget.render_knowledge_entries_block`). It is NOT a write
    #: rule -- nothing refuses a write for the entry's length (owner
    #: 2026-10-02: a refusal drops new evidence until someone compresses; the
    #: pressure to compress is a notice instead). `None` means uncapped.
    max_entry_tokens: int | None = None
    #: Past this many tokens a WRITABLE surface asks the model to compress the
    #: entry (`entry_size.size_notice`). `None`: never asked.
    soft_entry_tokens: int | None = None
    #: Whether an entry of this category may be split into sub-entries by
    #: section. False keeps the split wording out of every prompt for it: a
    #: streamer entry is injected whole by the streamer's name, so a split-off
    #: section would leave the default injection (plan kb-entry-size).
    splittable: bool = False
    #: One line saying what this category is, for the structure table the
    #: prompts carry. It lived in `render.py` as a hardcoded dict of three
    #: entries until 2026-09-07 -- and that dict's own comment records the
    #: 2026-09-02 incident where a hardcoded pair of categories made the table
    #: silently stop covering the store. A category is defined by its preset,
    #: so its one-line description belongs there too.
    blurb: str = ""
    #: Does this category join the MATCHING surface -- swept by
    #: `repo.resolve()`, given an index, and pre-injected by keyword -- or is it
    #: addressed only by explicit name?
    #:
    #: `style` is the latter, and that is a decision rather than an omission:
    #: matching would buy it nothing, and the global key check would then reject
    #: two subtitle groups owning a same-named style -- a style is usually named
    #: after the group or streamer it serves. The cost is that name uniqueness
    #: inside such a category is nobody else's job; `repo.resolve_in()` is where
    #: it happens.
    #:
    #: The default is only ever taken by the frozen version-1 archive presets
    #: (both matchable, and predating `style` entirely). Every version-2 preset
    #: must state it -- `_parse_v2` requires the key.
    matchable: bool = True
    #: Header written at the top of this category's `rendered/<cat>/index.md`:
    #: a markdown heading plus the line telling a human what the columns mean.
    #: It lived in `repo.py` as a two-entry dict keyed on category, which is
    #: the same shape -- and the same failure -- as the blurb dict did.
    #:
    #: Required exactly when `matchable` is true and meaningless otherwise: an
    #: index exists to be matched against, so a standalone category has none
    #: (`repo.index_text` raises for those on purpose).
    index_header: str = ""

    def section(self, name: str) -> SectionSpec | None:
        wanted = _normalize(name)
        for spec in self.sections:
            if _normalize(spec.name) == wanted:
                return spec
        if self.strict_sections:
            return None
        return self.default_section

    def section_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.sections)

    def staging_section(self) -> SectionSpec | None:
        """Where a line nobody could classify waits for a judgement call.

        Every preset that can reject a line needs one, or the only remaining
        answer is to reject the whole edit -- which throws the user's text
        away. Declared per preset rather than by name so a preset can opt out
        by simply not having one (the caller then falls back to refusing)."""

        return next((spec for spec in self.sections if spec.staging), None)

    def tier_of(self, section: str) -> str:
        spec = self.section(section)
        return spec.tier if spec else "optional"

    # ---- policy resolution -------------------------------------------------
    #
    # Both policies resolve label -> section -> preset default, and both are
    # FAIL-CLOSED for labels nobody registered: an unregistered label may well
    # be `[中之人]`, so it never inherits sharing and never leaves the machine
    # (review 2026-08-29 P1-2 / P1-3).

    def _section_share(self, section: str) -> str:
        spec = self.section(section)
        key = spec.name if spec is not None and spec.name else "*"
        value = self.share_inherit.get(section) or self.share_inherit.get(key)
        return value if value in SHARE_VALUES else "local"

    def share_for(self, section: str, label: str | None = None) -> str:
        if label:
            spec = self.section(section)
            registered = spec.label(label) if spec is not None else None
            if registered is None:
                return self.unknown_label_share
            if registered.share is not None:
                return registered.share
        return self._section_share(section)

    def verify_for(self, section: str, label: str | None = None) -> str:
        spec = self.section(section)
        if label:
            registered = spec.label(label) if spec is not None else None
            if registered is None:
                return self.unknown_label_verify
            if registered.verify is not None:
                return registered.verify
        return spec.verify if spec is not None else "none"

    def label_by_role(self, role: str) -> tuple[str, LabelSpec] | None:
        """``(section name, label)`` for a structural role — this is how the
        code asks "which label carries the native names" instead of matching
        the literal 本名 (plan §3 single source of truth)."""

        for spec in self.sections:
            for label in spec.labels:
                if label.role == role:
                    return spec.name, label
        return None


def _normalize(name: str) -> str:
    return "".join(unicodedata.normalize("NFKC", name or "").split())


def _require_bool(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise PresetError(f"{where}: expected a boolean, got {value!r}")
    return value


def _require_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise PresetError(f"{where}: expected a string, got {value!r}")
    return value


def _require_enum(value: Any, allowed: tuple[str, ...], where: str) -> str:
    text = _require_str(value, where)
    if text not in allowed:
        raise PresetError(f"{where}: {text!r} not in {allowed}")
    return text


def _require_positive_int(value: Any, where: str) -> int | None:
    if value is None:
        return None
    # bool is an int subclass and `max_lines = true` is a typo, not a 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PresetError(f"{where}: expected a positive integer, got {value!r}")
    return value


def _reject_unknown(entry: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(entry) - allowed)
    if unknown:
        raise PresetError(f"{where}: unknown key(s) {unknown}")


def _parse_label(entry: Any, where: str) -> LabelSpec:
    if not isinstance(entry, dict):
        raise PresetError(f"{where}: label must be a table")
    _reject_unknown(entry, _LABEL_KEYS, where)
    if "name" not in entry:
        raise PresetError(f"{where}: label needs a name")
    name = _require_str(entry["name"], f"{where}.name")
    core = _require_bool(entry.get("core", False), f"{where}.core")
    note = _require_str(entry.get("note", ""), f"{where}.note")
    if core and not note.strip():
        # an empty slot without an explanation is scaffolding with no guidance
        raise PresetError(f"{where}: core label {name!r} needs a note")
    role = entry.get("role")
    if role is not None:
        role = _require_enum(role, LABEL_ROLES, f"{where}.role")
    share = entry.get("share")
    if share is not None:
        share = _require_enum(share, SHARE_VALUES, f"{where}.share")
    verify = entry.get("verify")
    if verify is not None:
        verify = _require_enum(verify, VERIFY_VALUES, f"{where}.verify")
    raw_values = entry.get("values", [])
    if not isinstance(raw_values, list) or not all(isinstance(value, str) and value.strip() for value in raw_values):
        raise PresetError(f"{where}.values: expected a list of non-empty strings")
    values = tuple(value.strip() for value in raw_values)
    if len(set(values)) != len(values):
        raise PresetError(f"{where}.values: duplicate entries")
    model_hidden = _require_bool(entry.get("model_hidden", False), f"{where}.model_hidden")
    if core and model_hidden:
        raise PresetError(f"{where}: core label {name!r} leaves an empty slot in every preview; it cannot be model_hidden")
    return LabelSpec(
        name=name, core=core, role=role, share=share, verify=verify, note=note, values=values, model_hidden=model_hidden
    )


def _parse_section(entry: Any, where: str, *, named: bool) -> SectionSpec:
    if not isinstance(entry, dict):
        raise PresetError(f"{where}: section must be a table")
    _reject_unknown(entry, _SECTION_KEYS, where)
    name = _require_str(entry["name"], f"{where}.name") if named else ""
    if named and not name.strip():
        raise PresetError(f"{where}: section needs a name")
    raw_kinds = entry.get("body_kinds", ["note"])
    if not isinstance(raw_kinds, list) or not raw_kinds:
        raise PresetError(f"{where}.body_kinds: expected a non-empty list")
    body_kinds = tuple(_require_enum(k, BODY_KINDS, f"{where}.body_kinds") for k in raw_kinds)
    if len(set(body_kinds)) != len(body_kinds):
        raise PresetError(f"{where}.body_kinds: duplicate entries")
    labels = tuple(
        _parse_label(item, f"{where}.labels[{index}]")
        for index, item in enumerate(entry.get("labels", []))
    )
    seen: set[str] = set()
    for label in labels:
        key = _normalize(label.name)
        if key in seen:
            raise PresetError(f"{where}: duplicate label {label.name!r} (after normalization)")
        seen.add(key)
    return SectionSpec(
        name=name,
        body_kinds=body_kinds,
        purpose=_require_str(entry.get("purpose", ""), f"{where}.purpose"),
        exclude=_require_str(entry.get("exclude", ""), f"{where}.exclude"),
        staging=_require_bool(entry.get("staging", False), f"{where}.staging"),
        verify=_require_enum(entry.get("verify", "none"), VERIFY_VALUES, f"{where}.verify"),
        max_lines=_require_positive_int(entry.get("max_lines"), f"{where}.max_lines"),
        max_body_chars=_require_positive_int(entry.get("max_body_chars"), f"{where}.max_body_chars"),
        labels_from=(
            _require_str(entry["labels_from"], f"{where}.labels_from")
            if entry.get("labels_from") is not None
            else None
        ),
        labels=labels,
    )


def _parse_v2(data: Mapping[str, Any]) -> Preset:
    where = f"preset {data.get('name', '?')!r}"
    _reject_unknown(data, _PRESET_KEYS, where)
    sections = tuple(
        _parse_section(entry, f"{where}.sections[{index}]", named=True)
        for index, entry in enumerate(data.get("sections", []))
    )
    seen_sections: set[str] = set()
    for spec in sections:
        key = _normalize(spec.name)
        if key == _normalize(METADATA_SECTION):
            raise PresetError(f"{where}: presets must not declare the metadata section")
        if key in seen_sections:
            raise PresetError(f"{where}: duplicate section {spec.name!r}")
        seen_sections.add(key)
    default = (
        _parse_section(data["default_section"], f"{where}.default_section", named=False)
        if "default_section" in data
        else None
    )
    roles: dict[str, str] = {}
    for spec in (*sections, *((default,) if default else ())):
        for label in spec.labels:
            if label.role is None:
                continue
            if label.role in roles:
                raise PresetError(
                    f"{where}: role {label.role!r} claimed twice"
                    f" ({roles[label.role]!r} and {label.name!r})"
                )
            roles[label.role] = label.name
    # `labels_from` names another section, so a typo is a preset that loads
    # fine and then rejects every labelled line in that section as an orphan —
    # the failure would surface at write time, far from its cause. Same
    # treatment as `share_inherit`: the reference is resolved at load.
    by_key = {_normalize(spec.name): spec for spec in sections}
    for spec in sections:
        if spec.labels_from is None:
            continue
        target = _normalize(spec.labels_from)
        if target not in by_key:
            raise PresetError(
                f"{where}.sections[{spec.name!r}].labels_from:"
                f" {spec.labels_from!r} is not a declared section"
            )
        if target == _normalize(spec.name):
            raise PresetError(
                f"{where}.sections[{spec.name!r}].labels_from: a section cannot"
                " source its labels from itself"
            )
    # A chain is fine (a → b → c); a cycle is not — nothing in it could ever
    # be satisfied, since every label would have to exist somewhere upstream.
    for spec in sections:
        seen: set[str] = {_normalize(spec.name)}
        cursor = spec
        while cursor.labels_from is not None:
            nxt = _normalize(cursor.labels_from)
            if nxt in seen:
                raise PresetError(
                    f"{where}: labels_from forms a cycle through {spec.name!r}"
                )
            seen.add(nxt)
            cursor = by_key[nxt]
    share_inherit = data.get("share_inherit", {})
    if not isinstance(share_inherit, dict):
        raise PresetError(f"{where}.share_inherit: expected a table")
    # `index_header` is required exactly when the category joins the matching
    # surface, and rejected otherwise -- a standalone category has no index, so
    # a header for one would be a value nothing reads, sitting there looking
    # authoritative.
    matchable = _require_bool(data.get("matchable"), f"{where}.matchable")
    if matchable:
        index_header = _require_str(data.get("index_header"), f"{where}.index_header")
        if not index_header.strip():
            raise PresetError(f"{where}.index_header: must not be blank for a matchable category")
    else:
        if "index_header" in data:
            raise PresetError(
                f"{where}.index_header: only a matchable category has an index"
            )
        index_header = ""
    known = seen_sections | ({"*"} if default is not None else set())
    for key, value in share_inherit.items():
        _require_enum(value, SHARE_VALUES, f"{where}.share_inherit[{key!r}]")
        if _normalize(key) not in known and key != "*":
            raise PresetError(f"{where}.share_inherit: {key!r} is not a declared section")
    return Preset(
        name=_require_str(data["name"], f"{where}.name"),
        version=2,
        strict_sections=_require_bool(data.get("strict_sections", False), f"{where}.strict_sections"),
        sections=sections,
        default_section=default,
        share_inherit=dict(share_inherit),
        allow_custom_labels=_require_bool(
            data.get("allow_custom_labels", True), f"{where}.allow_custom_labels"
        ),
        unknown_label_share=_require_enum(
            data.get("unknown_label_share", "local"), SHARE_VALUES, f"{where}.unknown_label_share"
        ),
        unknown_label_verify=_require_enum(
            data.get("unknown_label_verify", "none"), VERIFY_VALUES, f"{where}.unknown_label_verify"
        ),
        max_entry_tokens=_require_positive_int(
            data.get("max_entry_tokens"), f"{where}.max_entry_tokens"
        ),
        soft_entry_tokens=_soft_entry_tokens(data, where),
        splittable=_require_bool(data.get("splittable", False), f"{where}.splittable"),
        # Both required, both without a fallback: a missing `blurb` would leave
        # the structure table naming a category it cannot describe, and a
        # missing `matchable` would silently default a new category ONTO the
        # matching surface -- the expensive direction of that mistake.
        blurb=_require_str(data.get("blurb"), f"{where}.blurb"),
        matchable=matchable,
        index_header=index_header,
    )


def _soft_entry_tokens(data: Mapping[str, Any], where: str) -> int | None:
    soft = _require_positive_int(data.get("soft_entry_tokens"), f"{where}.soft_entry_tokens")
    cap = data.get("max_entry_tokens")
    if soft is not None and isinstance(cap, int) and soft >= cap:
        raise PresetError(f"{where}.soft_entry_tokens must be below max_entry_tokens")
    return soft


def _parse_v1(data: Mapping[str, Any]) -> Preset:
    """Frozen legacy shape — only the markdown archive import reads these."""

    where = f"legacy preset {data.get('name', '?')!r}"
    _reject_unknown(data, _LEGACY_PRESET_KEYS, where)

    def _section(entry: Mapping[str, Any], name: str) -> SectionSpec:
        _reject_unknown(entry, _LEGACY_SECTION_KEYS, where)
        return SectionSpec(
            name=name,
            tier=str(entry.get("tier", "optional")),
            kinds=tuple(entry.get("kinds", ("note",))),
            line_form=str(entry.get("line_form", "note")),
        )

    sections = tuple(_section(entry, str(entry["name"])) for entry in data.get("sections", []))
    if any(_normalize(spec.name) == _normalize(METADATA_SECTION) for spec in sections):
        raise PresetError(f"{where}: presets must not declare the metadata section")
    default = _section(data["default_section"], "") if "default_section" in data else None
    return Preset(
        name=str(data["name"]),
        version=1,
        strict_sections=bool(data.get("strict_sections", False)),
        sections=sections,
        default_section=default,
        share_inherit=dict(data.get("share_inherit", {})),
    )


def parse_preset(data: Mapping[str, Any]) -> Preset:
    version = data.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise PresetError(f"preset version must be an integer, got {version!r}")
    if version == 2:
        return _parse_v2(data)
    if version == 1:
        return _parse_v1(data)
    raise PresetError(f"unknown preset version {version!r}")


def _preset_folder():  # type: ignore[no-untyped-def]
    """Where the preset files live.

    A function rather than an inlined expression so a test can point the loader
    at a temporary directory and exercise a category that does not ship -- the
    acceptance case in `docs/knowledge.md` (「加一个类别就是两步」) is exactly
    that, and it must not be forced to become a real category to be testable.
    A test that redirects this must also `load_preset.cache_clear()` and
    re-derive `MATCHABLE_CATEGORIES` via `_split_by_matching()`; both are
    computed once and would otherwise answer from before the redirect.
    """

    return resources.files(__package__) / "presets"


@lru_cache(maxsize=None)
def load_preset(name: str, *, legacy: bool = False) -> Preset:
    folder = _preset_folder()
    package = (folder / "legacy" / f"{name}.toml") if legacy else (folder / f"{name}.toml")
    with package.open("rb") as handle:
        preset = parse_preset(tomllib.load(handle))
    expected = 1 if legacy else 2
    if preset.version != expected:
        raise PresetError(f"preset {name!r}: expected version {expected}, got {preset.version}")
    return preset


def head_section_names() -> frozenset[str]:
    """Sections that belong to an entry's head: those whose preset carries the
    identity or aliases label (`档案`). A cut entry always keeps them -- they
    are what make it still that entry (plan kb-entry-size)."""

    names = set()
    for category in CATEGORIES:
        for spec in preset_for_category(category).sections:
            if any(label.role in ("identity", "aliases") for label in spec.labels):
                names.add(spec.name)
    return frozenset(names)


def preset_for_category(category: str, *, legacy: bool = False) -> Preset:
    """Categories map 1:1 onto presets (`legacy=True` only for the frozen
    version-1 archive import, which predates `style` entirely)."""

    known = ("streamer", "common") if legacy else CATEGORIES
    if category not in known:
        raise ValueError(f"no preset for category {category!r}")
    return load_preset(category, legacy=legacy)


def preset_digest(preset: Preset) -> str:
    """Fingerprint of what a preset MEANS, for the migration ledger.

    Hashed from the parsed object rather than the file bytes, so rewording a
    comment or re-indenting a table is not a "preset change" -- the ledger
    would cry wolf and get regenerated without thought, which is how a ratchet
    stops being read.

    ⚠ The trigger for a migration is a CONTENT change, not a `version` bump:
    dropping a section, renaming a label or tightening `max_lines` all leave
    `version = 2` while existing entries stop conforming (review 2026-09-07).
    So `version` is only one of the fields hashed here, not the thing watched.
    """

    import hashlib
    import json

    def _section(spec: SectionSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "body_kinds": list(spec.body_kinds),
            "purpose": spec.purpose,
            "exclude": spec.exclude,
            "staging": spec.staging,
            "verify": spec.verify,
            "max_lines": spec.max_lines,
            "max_body_chars": spec.max_body_chars,
            "labels_from": spec.labels_from,
            "labels": [
                {
                    "name": label.name,
                    "core": label.core,
                    "role": label.role,
                    "share": label.share,
                    "verify": label.verify,
                    # only when set, so a label without a value list hashes as before
                    **({"values": list(label.values)} if label.values else {}),
                }
                for label in spec.labels
            ],
        }

    shape = {
        "name": preset.name,
        "version": preset.version,
        "strict_sections": preset.strict_sections,
        "allow_custom_labels": preset.allow_custom_labels,
        "unknown_label_share": preset.unknown_label_share,
        "unknown_label_verify": preset.unknown_label_verify,
        "max_entry_tokens": preset.max_entry_tokens,
        "soft_entry_tokens": preset.soft_entry_tokens,
        "splittable": preset.splittable,
        "matchable": preset.matchable,
        "index_header": preset.index_header,
        "share_inherit": dict(sorted(preset.share_inherit.items())),
        "sections": [_section(spec) for spec in preset.sections],
        "default_section": (
            _section(preset.default_section) if preset.default_section else None
        ),
    }
    blob = json.dumps(shape, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _split_by_matching() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition `CATEGORIES` by each preset's own `matchable` declaration.

    Loading three small TOMLs at import is the price of having `matchable`
    stated once, in the file that defines the category, instead of twice. The
    alternative -- a second tuple here, kept in step by a test -- is what this
    replaced.

    A declared category with no file behind it dies here, at import, with a
    sentence naming the problem. It used to surface far downstream as a bare
    `FileNotFoundError` out of `load_preset`, which reads like a packaging
    fault rather than like the missing declaration it is.
    """

    for name in CATEGORIES:
        try:
            load_preset(name)
        except FileNotFoundError as exc:
            raise PresetError(
                f"category {name!r} is declared in CATEGORIES but there is no "
                f"presets/{name}.toml behind it"
            ) from exc
    matchable = tuple(name for name in CATEGORIES if load_preset(name).matchable)
    return matchable, tuple(name for name in CATEGORIES if name not in matchable)


#: Categories reached by MATCHING a name or alias found in free text:
#: `repo.resolve()` sweeps them, the index and the keyword pre-injection are
#: built from them, and `create_entry` checks new keys against all of them at
#: once -- keys are globally unique across the whole matchable set, which is
#: what lets `append_lines` find an entry afterwards and what
#: `proposals.py`'s category-hint fallback depends on.
#:
#: `STANDALONE_CATEGORIES` is the rest: addressed only by explicit name, each
#: its own namespace. See `Preset.matchable` for why `style` is one.
MATCHABLE_CATEGORIES, STANDALONE_CATEGORIES = _split_by_matching()
