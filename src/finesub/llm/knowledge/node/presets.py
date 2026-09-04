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

from .model import CATEGORIES, METADATA_SECTION

BODY_KINDS: tuple[str, ...] = ("note", "term")
SHARE_VALUES: tuple[str, ...] = ("inherit", "local")
VERIFY_VALUES: tuple[str, ...] = ("external", "none")
LABEL_ROLES: tuple[str, ...] = ("identity", "aliases")

_PRESET_KEYS = frozenset({
    "name", "version", "strict_sections", "sections", "default_section", "share_inherit",
    "allow_custom_labels", "unknown_label_share", "unknown_label_verify",
    "max_entry_tokens",
})
_SECTION_KEYS = frozenset({
    "name", "body_kinds", "purpose", "exclude", "staging", "verify", "labels",
    "max_lines", "max_body_chars", "labels_from",
})
_LABEL_KEYS = frozenset({"name", "core", "role", "share", "verify", "note"})
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


@dataclass(frozen=True)
class SectionSpec:
    #: Write-side caps (`docs/plans/translation-style-plan.md` §2.3). They live on
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
    #: express (`docs/plans/translation-style-plan.md` §2.2) — and it is what makes
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
    #: Backstop on the WHOLE entry's injected projection, in tokens. The
    #: per-section caps bound each line and each section; this bounds what the
    #: entry costs a prompt when every section sits near its own limit. `None`
    #: means uncapped, which is every category but `style`.
    #:
    #: It must NOT simply refuse writes when exceeded: the fix for an oversized
    #: entry is itself a write, and refusing every write would lock the entry
    #: at its worst. The rule is monotone instead (`_check_entry_budget`).
    max_entry_tokens: int | None = None

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
    return LabelSpec(name=name, core=core, role=role, share=share, verify=verify, note=note)


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
    )


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


@lru_cache(maxsize=None)
def load_preset(name: str, *, legacy: bool = False) -> Preset:
    folder = resources.files(__package__) / "presets"
    package = (folder / "legacy" / f"{name}.toml") if legacy else (folder / f"{name}.toml")
    with package.open("rb") as handle:
        preset = parse_preset(tomllib.load(handle))
    expected = 1 if legacy else 2
    if preset.version != expected:
        raise PresetError(f"preset {name!r}: expected version {expected}, got {preset.version}")
    return preset


def preset_for_category(category: str, *, legacy: bool = False) -> Preset:
    """Categories map 1:1 onto presets (`legacy=True` only for the frozen
    version-1 archive import, which predates `style` entirely)."""

    known = ("streamer", "common") if legacy else CATEGORIES
    if category not in known:
        raise ValueError(f"no preset for category {category!r}")
    return load_preset(category, legacy=legacy)
