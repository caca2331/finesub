"""Installed game glossaries: one per game, kept beside the knowledge base and outside its sqlite store.

A glossary is a maintainer-built JSON file (``tools/game_glossary/``) that reaches a user through
``knowledge glossary fetch`` (``fetch``), which downloads what ``glossary-manifest.json`` (shipped with the
package) lists for the game, verifies it and installs it. It lives in ``<knowledge root>/glossary/``:

- ``<sha256>.json``: the file as published, named by its own hash, never edited;
- ``installed.json``: per game, the hash in use and the header the preparation task's index shows.

Kept out of ``CATEGORIES`` on purpose: every knowledge path (the update's material and proposals, the repair
sessions, ``rendered/``, ``knowledge edit``, sharing, verification, candidate scans, size notices, the structure
table and the proposal grammar's category enum) walks that tuple, so a glossary category would need a skip in each,
and a missed one rewrites or shares a list the user never wrote. Stored apart, none of them sees it.

Re-importing a game replaces its entry; the replaced file stays for a run that started on it (``KEEP_FILES`` per
game), so a resumed run reads what it started with (``load``).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

FORMAT = 1
DIRECTORY = "glossary"
INSTALLED = "installed.json"
KEEP_FILES = 3
QUALIFIER = "glossary/"  # how the preparation task names one: glossary/原神
MANIFEST = Path(__file__).with_name("glossary-manifest.json")
STALE_DAYS = 183  # the core list is "the last two years"; past this the build is worth refetching


class GlossaryError(ValueError):
    """A glossary file that cannot be used."""


@dataclass(frozen=True)
class GlossaryTerm:
    ja: str
    zh: str
    en: str
    kana: str       # reading, or the katakana spelling of a kana name; "" when unknown (verbatim matching only)
    reading: bool   # whether ``kana`` is a reading rather than a spelling
    label: str      # 角色 / 地区 / 组织 ...
    note: str
    group: str      # the entity: several spellings of one name share it
    core: bool      # on the resident list
    famous: bool    # a closed-book model knew it

    def name(self, language: str) -> str:
        return {"ja": self.ja, "zh": self.zh, "en": self.en}.get(language, "")


@dataclass(frozen=True)
class Glossary:
    game: str
    aliases: tuple[str, ...]
    intro: str
    built: str
    sha256: str
    terms: tuple[GlossaryTerm, ...]
    noise: Mapping[tuple[str, str], float]  # (ja, ASR spelling) -> background per-row rate

    def stale(self, today: date | None = None) -> bool:
        return ((today or date.today()) - date.fromisoformat(self.built)).days > STALE_DAYS


@dataclass(frozen=True)
class Installed:
    game: str
    aliases: tuple[str, ...]
    intro: str
    sha256: str

    @property
    def key(self) -> str:
        return QUALIFIER + self.game


def _text(raw: Mapping[str, Any], key: str, where: str, required: bool = True) -> str:
    value = raw.get(key, "")
    if not isinstance(value, str) or (required and not value.strip()):
        raise GlossaryError(f"{where}: {key!r} must be a non-empty string")
    return value.strip()


def parse(data: bytes) -> Glossary:
    """The glossary in ``data``, validated; raises ``GlossaryError``."""

    try:
        raw = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlossaryError(f"not a JSON file: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise GlossaryError(f"unsupported glossary format {raw.get('format') if isinstance(raw, dict) else None!r}"
                            f" (this FineSub reads {FORMAT})")
    head = raw.get("game")
    if not isinstance(head, dict):
        raise GlossaryError("missing 'game' header")
    aliases = head.get("aliases", [])
    if not isinstance(aliases, list) or not all(isinstance(a, str) and a.strip() for a in aliases):
        raise GlossaryError("game.aliases must be a list of strings")
    built = _text(raw, "built", "glossary")
    try:
        date.fromisoformat(built)
    except ValueError as exc:
        raise GlossaryError(f"built: {built!r} is not a date") from exc
    terms = []
    for i, t in enumerate(raw.get("terms") or ()):
        where = f"terms[{i}]"
        if not isinstance(t, dict):
            raise GlossaryError(f"{where}: not an object")
        if not isinstance(t.get("reading", False), bool) or not all(
                isinstance(t.get(k, False), bool) for k in ("core", "famous")):
            raise GlossaryError(f"{where}: reading / core / famous must be booleans")
        terms.append(GlossaryTerm(
            ja=_text(t, "ja", where), zh=_text(t, "zh", where), en=_text(t, "en", where, False),
            kana=_text(t, "kana", where, False), reading=t.get("reading", False), label=_text(t, "label", where, False),
            note=_text(t, "note", where, False), group=_text(t, "group", where, False) or t["ja"].strip(),
            core=t.get("core", False), famous=t.get("famous", False),
        ))
    if not terms:
        raise GlossaryError("no terms")
    noise: dict[tuple[str, str], float] = {}
    for key, rate in (raw.get("noise") or {}).items():
        ja, sep, surface = key.partition("|")
        if not sep or not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 <= rate <= 1:
            raise GlossaryError(f"noise[{key!r}]: expected 'ja|surface': rate in [0, 1]")
        noise[(ja, surface)] = float(rate)
    return Glossary(
        game=_text(head, "name", "game"), aliases=tuple(a.strip() for a in aliases), intro=_text(head, "intro", "game"),
        built=built, sha256=hashlib.sha256(data).hexdigest(), terms=tuple(terms), noise=noise,
    )


@dataclass(frozen=True)
class Published:
    """One downloadable glossary, as the shipped manifest lists it."""

    game: str
    aliases: tuple[str, ...]
    url: str
    size: int
    sha256: str


def published(manifest: Path | None = None) -> list[Published]:
    raw = json.loads((manifest or MANIFEST).read_text(encoding="utf-8"))
    return [Published(e["game"], tuple(e.get("aliases", ())), e["url"], int(e["size"]), e["sha256"])
            for e in raw["glossaries"]]


def fetch(knowledge_root: str | Path | None, name: str, *, manifest: Path | None = None) -> Glossary:
    """Download the published glossary for ``name`` (game or alias), verify it against the manifest and install it.

    The download goes through the same route resolution as model files (region, GitHub mirror, fallback to the
    official URL), so a proxy or mirror the user set applies here too."""

    from finesub_bootstrap import model_fetch
    from finesub_bootstrap.download_routes import resolve_region
    from finesub_bootstrap.model_manifest import ManifestFile, ModelEntry
    from finesub_bootstrap.paths import default_data_root

    entries = published(manifest)
    wanted = _fold(name)
    entry = next((e for e in entries if wanted in {_fold(e.game), *map(_fold, e.aliases)}), None)
    if entry is None:
        offered = "、".join(e.game for e in entries) or "none yet"
        raise GlossaryError(f"no published glossary for {name!r} (published: {offered})")
    folder = directory(knowledge_root)
    folder.mkdir(parents=True, exist_ok=True)
    file = ManifestFile(name=f"{entry.sha256}.json", url=entry.url, size=entry.size, sha256=entry.sha256)
    data_root = default_data_root()
    with tempfile.TemporaryDirectory(prefix="fetch-", dir=folder) as scratch:
        model_fetch.fetch_fixed_files(ModelEntry(model_id="glossary", files=(file,)), Path(scratch),
                                      data_root=data_root, region=resolve_region(data_root).region)
        data = (Path(scratch) / file.name).read_bytes()
    return install(knowledge_root, data)


def directory(knowledge_root: str | Path | None) -> Path:
    from .knowledge.base import knowledge_root_path

    return knowledge_root_path(knowledge_root) / DIRECTORY


def installed(knowledge_root: str | Path | None) -> dict[str, Installed]:
    """Game -> its installed glossary, in install order; empty when none (or no knowledge root)."""

    from .knowledge.base import resolve_knowledge_root

    root = resolve_knowledge_root(knowledge_root, required=False)
    path = Path(root) / DIRECTORY / INSTALLED if root else None
    if path is None or not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {game: Installed(game, tuple(e["aliases"]), e["intro"], e["sha256"]) for game, e in raw["games"].items()}


def install(knowledge_root: str | Path | None, data: bytes) -> Glossary:
    """Validate ``data``, store it and make it the game's glossary (replacing an earlier one)."""

    glossary = parse(data)
    folder = directory(knowledge_root)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{glossary.sha256}.json"
    if not target.is_file():
        _write_atomic(target, data)
    games = {game: {"aliases": list(e.aliases), "intro": e.intro, "sha256": e.sha256,
                    "history": e_history}
             for game, e, e_history in _with_history(folder)}
    history = [h for h in games.get(glossary.game, {}).get("history", []) if h != glossary.sha256]
    history = [glossary.sha256, *history][:KEEP_FILES]
    games[glossary.game] = {"aliases": list(glossary.aliases), "intro": glossary.intro, "sha256": glossary.sha256,
                            "history": history}
    _write_atomic(folder / INSTALLED, json.dumps({"games": games}, ensure_ascii=False, indent=1).encode("utf-8"))
    keep = {h for e in games.values() for h in e["history"]}
    for old in folder.glob("*.json"):
        if old.name != INSTALLED and old.stem not in keep:
            old.unlink(missing_ok=True)
    return glossary


def _with_history(folder: Path):
    path = folder / INSTALLED
    if not path.is_file():
        return
    for game, e in json.loads(path.read_text(encoding="utf-8"))["games"].items():
        yield game, Installed(game, tuple(e["aliases"]), e["intro"], e["sha256"]), e.get("history", [e["sha256"]])


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def load(knowledge_root: str | Path | None, sha256: str) -> Glossary | None:
    """The glossary stored under ``sha256``; None once it has been pruned."""

    path = directory(knowledge_root) / f"{sha256}.json"
    return _load(str(path)) if path.is_file() else None


@lru_cache(maxsize=4)
def _load(path: str) -> Glossary:
    return parse(Path(path).read_bytes())


def resolve(knowledge_root: str | Path | None, name: str) -> Installed | None:
    """The installed glossary a requested name means: ``glossary/<game or alias>``."""

    if not name.strip().startswith(QUALIFIER):
        return None
    wanted = _fold(name.strip()[len(QUALIFIER):])
    for entry in installed(knowledge_root).values():
        if wanted in {_fold(entry.game), *map(_fold, entry.aliases)}:
            return entry
    return None


def named_in(knowledge_root: str | Path | None, text: str) -> list[Installed]:
    """Installed glossaries whose game name or an alias (2+ characters) appears in ``text`` (the user's notes)."""

    haystack = _fold(text)
    if not haystack:
        return []
    return [entry for entry in installed(knowledge_root).values()
            if any(len(t) >= 2 and _fold(t) in haystack for t in (entry.game, *entry.aliases))]


def _fold(text: str) -> str:
    return "".join(text.split()).casefold()


def index_lines(knowledge_root: str | Path | None) -> str:
    """One line per installed glossary for the preparation task's index; "" when none."""

    return "\n".join(f"- {e.key} | {'、'.join(e.aliases) or '-'} | {e.intro}"
                     for e in installed(knowledge_root).values())


def with_index(indexes: Mapping[str, str], knowledge_root: str | Path | None) -> dict[str, str]:
    """The preparation task's indexes: ``indexes`` plus the installed glossaries' lines under ``DIRECTORY``, for the
    task to pick one (``glossary/<game>``) outside its entry slots."""

    lines = index_lines(knowledge_root)
    return {**indexes, DIRECTORY: lines} if lines else dict(indexes)


def split_requests(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(glossary names, the rest)`` of the entry names a preparation task asked for."""

    ours = [n for n in names if n.strip().startswith(QUALIFIER)]
    return ours, [n for n in names if n not in ours]


def selection(knowledge_root: str | Path | None, notes: str, requested: Sequence[str]) -> list[dict[str, str]]:
    """The glossaries a task uses: the ones its notes name and the ones it requested, pinned by hash."""

    chosen = {e.game: e for e in named_in(knowledge_root, notes)}
    for name in requested:
        if (entry := resolve(knowledge_root, name)) is not None:
            chosen.setdefault(entry.game, entry)
    return [{"game": e.game, "sha256": e.sha256} for e in chosen.values()]


def load_selection(knowledge_root: str | Path | None, picks: Sequence[Mapping[str, str]]) -> list[Glossary]:
    """The pinned glossaries; one pruned since is replaced by the game's current one, with a warning."""

    out = []
    current = installed(knowledge_root)
    for pick in picks:
        glossary = load(knowledge_root, pick["sha256"])
        if glossary is None and pick["game"] in current:
            from finesub.reporting import current_reporter

            current_reporter().warning(
                "game-glossary-replaced",
                f"the game glossary this task started with ({pick['game']}) has been replaced since",
                impact="the remaining windows use the one installed now",
            )
            glossary = load(knowledge_root, current[pick["game"]].sha256)
        if glossary is not None:
            out.append(glossary)
    return out
