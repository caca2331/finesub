"""Slim a correction window's knowledge entries, and hint the terms its rows mishear.

Developer switch, off by default: ``FINESUB_KB_SLIM=on``. Off, nothing here runs and the correction
prompt is byte-identical to what it was. On, the serial window loop (`stages/correction/serial.py`)
hands it each window's injected entries -- only when the window has no request channel, whose fill
turn promises the entries in full -- and a size gate decides:

- the entries rendered whole fit in ``GATE_TOKENS``: they go in unchanged, with no hints;
- above it, every entry keeps its head (H1, intro, 档案); of the term lines elsewhere, a term the
  window says verbatim keeps its whole line, a hinted term is left to the hint block (which carries
  its line and rows), and every other term stays as its names without a description. Prose sections
  and 元数据 go.

Hints: every row of the window is matched against the terms of the window's entries -- a stored
misheard spelling found verbatim, or a loose romaji reading within ``THRESHOLD`` of a term's surface or
alias (no calibration; a reading needs ``MIN_MORA`` morae and ``MIN_ROMAJI`` letters). A term written
verbatim in the row is skipped: that is not a mishearing. Each hinted term is one row listing every
window row it was found on; ``MAX_HITS`` counts terms, in order of first appearance. The fragment that
frames the rows is ``fragment_kb_phonetic_hints_v1.md``; the hints enter the window's entry signature.

``--continuity parallel`` shares one fixed entry block per session and keeps it whole.

Readings need the ``kb-phonetic`` extra (pyopenjtalk-plus, pykakasi, rapidfuzz). Without it the switch
warns once and the entries go in whole: slimming without hints loses names the hints would have found.
"""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from finesub.reporting import current_reporter, reporter_delivers

from .chunking import SubtitleWindow

ENV = "FINESUB_KB_SLIM"
GATE_TOKENS = 2000
THRESHOLD = 95
MIN_MORA = 3
MIN_ROMAJI = 6
MAX_HITS = 24

_SMALL = str.maketrans("ァィゥェォャュョヮヵヶ", "アイウエオヤユヨワカケ")
_READINGS: dict[str, str] = {}


def enabled() -> bool:
    return os.environ.get(ENV, "").strip().lower() == "on"


# --- readings ---------------------------------------------------------------


# Which reader gives the kana: "pyopenjtalk" (production) or "pykakasi" (much smaller; compared
# offline by ``tools/kb_phonetic_eval/eval.py --reader`` and found to recall less). Set before the
# first reading: readings are cached per process.
READER = "pyopenjtalk"


_warned_missing = False


@lru_cache(maxsize=1)
def _missing_reader() -> str:
    """The first reading library that does not import, or ""."""

    for name in ("pyopenjtalk", "pykakasi", "rapidfuzz"):
        try:
            __import__(name)
        except ImportError:
            return name
    return ""


def readers_available() -> bool:
    """Whether the ``kb-phonetic`` extra is installed; says so once when it is not."""

    global _warned_missing
    missing = _missing_reader()
    if missing and not _warned_missing:
        reporter = current_reporter()
        reporter.warning(
            "kb-slim-readers",
            f"{ENV}=on but the reading library {missing} is missing",
            impact="knowledge entries go in whole, with no phonetic hints",
            action='pip install -e ".[kb-phonetic]"',
        )
        # Only a reporter that shows it counts as having said it (`speech/runtime/device.py`).
        _warned_missing = reporter_delivers(reporter)
    return not missing


def reading(text: str) -> str:
    """Katakana reading, punctuation and spaces dropped (cached per process)."""

    if text not in _READINGS:
        normalized = unicodedata.normalize("NFKC", text)
        if READER == "pykakasi":
            kana = "".join(part["kana"] for part in _kakasi().convert(normalized))
        else:
            import pyopenjtalk

            kana = pyopenjtalk.g2p(normalized, kana=True)
        _READINGS[text] = "".join(ch for ch in kana if "ァ" <= ch <= "ヺ" or ch == "ー")
    return _READINGS[text]


@lru_cache(maxsize=None)
def loose(kana: str) -> str:
    """Coarse kana: no long marks or sokuon, no voicing, small kana made big."""

    out = unicodedata.normalize("NFD", kana.translate(_SMALL))
    out = "".join(ch for ch in out if ch not in "゙゚ーッ")
    return unicodedata.normalize("NFC", out).replace("ヲ", "オ")


@lru_cache(maxsize=1)
def _kakasi():
    import pykakasi

    return pykakasi.kakasi()


@lru_cache(maxsize=None)
def romaji(kana: str) -> str:
    return "".join(part["hepburn"] for part in _kakasi().convert(kana)).replace("'", "")


def mora_count(kana: str) -> int:
    return sum(1 for ch in kana if ch not in "ァィゥェォャュョヮ")


def reading_complete(text: str, kana: str) -> bool:
    """False when the reader dropped characters -- typically a simplified-Chinese alias
    (绝区零 reads as just レー), whose reading is noise."""

    ideographs = sum(1 for ch in text if "一" <= ch <= "鿿")
    kana_chars = sum(1 for ch in text if "぀" <= ch <= "ヿ")
    return mora_count(kana) >= ideographs + int(kana_chars * 0.8)


def fuzzy_hit(key_romaji: str, text_romaji: str, threshold: int) -> bool:
    """``partial_ratio`` slides the SHORTER string, so a text shorter than the key
    would be "found" inside the key: compare whole strings then."""

    from rapidfuzz import fuzz

    if len(text_romaji) < len(key_romaji):
        return fuzz.ratio(key_romaji, text_romaji) >= threshold
    return fuzz.partial_ratio(key_romaji, text_romaji) >= threshold


@lru_cache(maxsize=None)
def fuzzy_forms(names: tuple[str, ...]) -> tuple[str, ...]:
    """Loose romaji of the names whose reading is complete and long enough to fuzzy-match."""

    forms = []
    for name in names:
        kana = reading(name)
        if not kana or not reading_complete(name, kana) or mora_count(kana) < MIN_MORA:
            continue
        form = romaji(loose(kana))
        if len(form) >= MIN_ROMAJI:
            forms.append(form)
    return tuple(dict.fromkeys(forms))


# --- the library's term table ------------------------------------------------


@dataclass(frozen=True)
class Term:
    address: str  # "<entry surface> / <section>"
    line: str  # the term line as the entry renders it, without the list dash
    exact: tuple[str, ...]  # surface + aliases, normalized
    misheard: tuple[str, ...]  # stored misheard spellings, normalized
    names: tuple[str, ...]  # surface + aliases as written: what readings are taken of

    @property
    def surface(self) -> str:
        return self.line.split("|", 1)[0].strip()


@lru_cache(maxsize=4)
def library_terms(root: str, rev: int) -> Mapping[str, tuple[Term, ...]]:
    """Every entry's terms at ``rev``, by subject id (no readings yet: `fuzzy_forms` takes them)."""

    from .knowledge.node.matching import ExactIndex
    from .knowledge.node.render import format_line, node_aliases
    from .knowledge.node.repo import KnowledgeRepo

    store = KnowledgeRepo.open(Path(root)).store
    index = ExactIndex.build(store, rev, include_tentative=False)
    keys: dict[str, dict[str, set[str]]] = {}
    subject_of: dict[str, str] = {}
    for text, owners in index._keys.items():  # noqa: SLF001 -- same engine, keyed by node
        for key in owners:
            bucket = keys.setdefault(key.node_id, {"exact": set(), "misheard": set(), "raw": set()})
            if key.kind == "misheard":
                bucket["misheard"].add(text)
            else:
                bucket["exact"].add(text)
                bucket["raw"].add(key.raw or text)
            subject_of[key.node_id] = key.subject_id
    by_subject: dict[str, list[Term]] = {}
    for node_id, bucket in keys.items():
        subject_id = subject_of[node_id]
        if node_id == subject_id:
            continue  # entry names are the preparation task's business
        node = store.node(node_id, rev)
        subject = store.node(subject_id, rev)
        if node is None or subject is None:
            continue
        parents = store.parents(node_id, rev)
        section = str(parents[0].section) if parents else ""
        by_subject.setdefault(subject_id, []).append(
            Term(
                address=" / ".join(p for p in (str(subject.payload.get("surface") or ""), section) if p),
                line=format_line(node, node_aliases(store, node, rev)).lstrip("- ").strip(),
                exact=tuple(sorted(bucket["exact"])),
                misheard=tuple(sorted(bucket["misheard"])),
                names=tuple(sorted(bucket["raw"])),
            )
        )
    return {
        subject_id: tuple(sorted(terms, key=lambda term: (term.address, term.line)))
        for subject_id, terms in by_subject.items()
    }


def window_terms(root: str, rev: int, entry_keys) -> dict[str, tuple[Term, ...]]:
    """The terms of each of ``entry_keys`` that resolves to an entry, keyed like ``entry_keys``."""

    from .knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(Path(root))
    table = library_terms(root, rev)
    out = {}
    for key in entry_keys:
        resolved = repo.resolve(key, rev)
        if resolved is not None:
            out[key] = table.get(resolved.subject_id, ())
    return out


# --- one window ---------------------------------------------------------------


def window_hits(window: SubtitleWindow, terms) -> dict[Term, list[int]]:
    """Terms the window's rows mishear (stored spelling or soundalike), with the rows, in first-row order."""

    from .knowledge.node.matching import scan_normalize

    found: dict[Term, list[int]] = {}
    for local_id, segment in enumerate(window.segments, start=1):
        normalized = scan_normalize(segment.text)
        spoken = None
        for term in terms:
            if any(key in normalized for key in term.exact):
                continue
            hit = any(key in normalized for key in term.misheard)
            forms = () if hit else fuzzy_forms(term.names)
            if forms:
                if spoken is None:
                    spoken = romaji(loose(reading(segment.text)))
                hit = any(fuzzy_hit(form, spoken, THRESHOLD) for form in forms)
            if hit:
                found.setdefault(term, []).append(local_id)
    return dict(list(found.items())[:MAX_HITS])


@dataclass(frozen=True)
class SlimmedEntries:
    bodies: dict[str, str]
    hints: str  # `rows|entry|<the term's line>` rows, "" when nothing sounds alike
    hinted: tuple[str, ...]  # the hinted terms' surfaces, for the run's report


def slim_window_entries(
    window: SubtitleWindow, bodies: Mapping[str, str], *, root: str, rev: int
) -> SlimmedEntries | None:
    """The window's entries slimmed, with its hints; None when the readers are missing."""

    if not readers_available():
        return None
    from .knowledge.node.matching import scan_normalize

    terms_by_key = window_terms(root, rev, bodies)
    terms = tuple(dict.fromkeys(term for key in bodies for term in terms_by_key.get(key, ())))
    rows = [scan_normalize(segment.text) for segment in window.segments]
    said = {term.line for term in terms if any(key in row for row in rows for key in term.exact)}
    hits = window_hits(window, terms)
    # A term said verbatim on one row and misheard on another is hinted AND keeps its whole line.
    hinted = {term.line for term in hits} - said
    return SlimmedEntries(
        # An entry that does not resolve has no terms to say or hint: it stays whole.
        bodies={key: _slim_body(body, said, hinted) if key in terms_by_key else body for key, body in bodies.items()},
        hints="\n".join(f"{','.join(map(str, lines))}|{term.address}|{term.line}" for term, lines in hits.items()),
        hinted=tuple(term.surface for term in hits),
    )


def _slim_body(body: str, said: set[str], hinted: set[str]) -> str:
    from .knowledge.base import TERM_LINE_COLUMNS, strip_term_description

    out: list[str] = []
    heading = ""  # a section heading waiting for its first kept line
    section = "#"
    for line in body.splitlines():
        text = line.strip()
        if text.startswith("# "):
            section, heading = "#", ""
            out.append(line)
            continue
        if text.startswith("## "):
            section, heading = text[3:].strip(), line
            if section == "档案":
                out.append(line)
                heading = ""
            continue
        if section in ("#", "档案"):
            out.append(line)
            continue
        if section == "元数据" or not text or text.count("|") < TERM_LINE_COLUMNS - 1:
            continue  # prose and metadata go
        bare = text.lstrip("- ").strip()
        if bare in hinted:
            continue
        kept = line if bare in said else strip_term_description(line)
        if heading:
            out.extend(([""] if out and out[-1].strip() else []) + [heading])
            heading = ""
        out.append(kept)
    return "\n".join(out).strip() + "\n"
