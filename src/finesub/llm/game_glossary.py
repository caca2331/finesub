"""The game-glossary block of a correction window (`fragment_game_glossary_v2.md`).

A task gets a glossary when the preparation task selects an installed one (`glossary_store`): the user's notes
name the game, or the task names ``glossary/<game>`` in ``<requested_entries>``. Unselected, nothing here runs and
the correction prompt is unchanged.

``GlossaryView`` is built once per run from the whole transcript and renders each window's block, three parts:

- **resident list**: the glossary's core terms that a closed-book model did not know (``core and not famous``),
  the same for every window;
- **frequency list**: names written verbatim on ``FREQUENT_ROWS`` or more rows of this recording;
- **window candidates**: names written verbatim on a row of the window (a famous name written verbatim is left out:
  a model writes it right without help) and, for a Japanese source with the ``kb-phonetic`` extra, names a row
  sounds like (`glossary_match`, frequent names at a looser line), each with its rows and basis (原样 / 音近).

The user's own entries win (owner O3): a name the window's injected entries carry is listed nowhere. Columns follow
the source language: Japanese or English sources show ``source=中文`` pairs (a pair whose two sides are the same
is dropped), a Chinese source shows the Chinese names. The block has its own token cap
(``GAME_GLOSSARY_BLOCK_TOKENS``): candidates go first, sound-alikes with the highest background rate first.

``--continuity parallel`` renders one fixed block for the whole session (``fixed_block``): the two lists and the
candidates that recur on ``PARALLEL_MIN_ROWS`` rows of the transcript at the base line.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from string import Template

from . import glossary_match as gm
from .prompt_tables import prompt_phrases
from .glossary_store import Glossary, GlossaryTerm

FREQUENT_ROWS = 2
PARALLEL_MIN_ROWS = 2
NOTE_CHARS = 40
COLUMN = {"ja": "日文", "en": "英文", "zh": "中文"}


@dataclass(frozen=True)
class Candidate:
    term: GlossaryTerm
    rows: tuple[int, ...]   # window-local, from 1
    basis: str              # 原样 / 音近
    rank: tuple             # lower goes first when the block is cut


class GlossaryView:
    def __init__(self, glossaries: Sequence[Glossary], texts: Sequence[str], source: str, *, sound: bool):
        """``texts``: every ASR row of the recording; ``sound``: run the sound matcher (Japanese source, extra
        installed)."""

        self.games = tuple(g.game for g in glossaries)
        self.source = source if source in COLUMN else "ja"
        seen: set[tuple[str, str]] = set()
        self.terms: list[GlossaryTerm] = []
        for g in glossaries:
            for t in g.terms:
                if (t.ja, t.zh) not in seen and self._usable(t):
                    seen.add((t.ja, t.zh))
                    self.terms.append(t)
        self.by_form: dict[str, list[GlossaryTerm]] = defaultdict(list)
        for t in self.terms:
            self.by_form[self._form(t)].append(t)
        self.forms = gm.by_length(self.by_form)
        self.verbatim = {text: gm.verbatim(text, self.forms) for text in set(texts)}
        rows_per_group: dict[str, int] = defaultdict(int)
        for text in texts:
            for group in {t.group for f in self.verbatim[text] for t in self.by_form[f]}:
                rows_per_group[group] += 1
        frequent_groups = {g for g, n in rows_per_group.items() if n >= FREQUENT_ROWS}
        self.core = [t for t in self.terms if t.core and not t.famous]
        core_keys = {(t.ja, t.zh) for t in self.core}
        self.frequent = [t for t in self.terms if t.group in frequent_groups and (t.ja, t.zh) not in core_keys]
        self.listed = core_keys | {(t.ja, t.zh) for t in self.frequent}
        self.sound: dict[str, list[gm.SoundHit]] = {}
        self.by_ja: dict[str, list[GlossaryTerm]] = defaultdict(list)  # one ja may carry two names (ナッツ: 松果, 小松果)
        for t in self.terms:
            self.by_ja[t.ja].append(t)
        if sound:
            noise: dict[tuple[str, str], float] = {}
            for g in glossaries:
                noise.update(g.noise)
            matcher = gm.SoundMatcher((gm.Term(t.ja, t.kana, t.reading) for t in self.terms if t.kana), noise)
            self.sound = matcher.match(texts, frozenset(t.ja for t in self.terms if t.group in frequent_groups))

    def _form(self, t: GlossaryTerm) -> str:
        return t.name(self.source)

    def _usable(self, t: GlossaryTerm) -> bool:
        """Whether ``t`` can be matched in this source: it has a form there, and in English that form is a name, not
        an ordinary word the upstream lists in lower case (contract, element)."""

        form = self._form(t)
        return bool(form) and not (self.source == "en" and form[:1].islower())

    def _pair(self, t: GlossaryTerm) -> str:
        if self.source == "zh":
            return t.zh
        src = self._form(t)
        return "" if src == t.zh else f"{src}={t.zh}"

    def _list(self, terms: Sequence[GlossaryTerm], skip: str) -> str:
        groups: dict[str, list[str]] = {}
        for t in terms:
            if (pair := self._pair(t)) and not _mentioned(t, skip):
                groups.setdefault(t.label or "其他", []).append(pair)
        return "\n".join(f"{label}：{'；'.join(dict.fromkeys(pairs))}" for label, pairs in groups.items())

    def candidates(self, texts: Sequence[str], skip: str = "") -> list[Candidate]:
        found: dict[tuple[str, str], dict] = {}
        for i, text in enumerate(texts, start=1):
            for f in self.verbatim.get(text, ()):
                for t in self.by_form[f]:
                    if not t.famous:
                        found.setdefault((t.ja, t.zh), {"t": t, "rows": [], "basis": "原样", "rank": (0, 0.0, 0.0)})
                        found[(t.ja, t.zh)]["rows"].append(i)
            for hit in self.sound.get(text, ()):
                for t in self.by_ja[hit.ja]:
                    c = found.setdefault((t.ja, t.zh), {"t": t, "rows": [], "basis": "音近",
                                                        "rank": (1, hit.noise, hit.cost)})
                    if c["basis"] == "音近":
                        c["rank"] = min(c["rank"], (1, hit.noise, hit.cost))
                    c["rows"].append(i)
        return sorted((Candidate(c["t"], tuple(dict.fromkeys(c["rows"])), c["basis"], c["rank"])
                       for key, c in found.items()
                       if key not in self.listed and self._pair(c["t"]) and not _mentioned(c["t"], skip)),
                      # a total order: names found on one row come as a set, whose order changes with the process
                      key=lambda c: (c.rows[0], c.rank, c.term.ja, c.term.zh))

    def block(self, texts: Sequence[str], render: Callable[..., str], count_tokens: Callable[[str], int],
              limit: int, skip: str = "") -> str:
        """The window's block for its rows ``texts``; ``skip``: the text of the user's entries injected there."""

        return self._render(self.candidates(texts, skip), render, count_tokens, limit, skip, rows=True)

    def fixed_block(self, texts: Sequence[str], render: Callable[..., str], count_tokens: Callable[[str], int],
                    limit: int, skip: str = "") -> str:
        """One block for every window of a ``--continuity parallel`` session: recurring candidates, no row ids."""

        recurring = [c for c in self.candidates(texts, skip) if len(c.rows) >= PARALLEL_MIN_ROWS
                     and (c.basis == "原样" or c.rank[2] <= gm.LIMITS["kata"])]  # the base line, not the loosened one
        return self._render(recurring, render, count_tokens, limit, skip, rows=False)

    def _render(self, cands: list[Candidate], render, count_tokens, limit: int, skip: str, *, rows: bool) -> str:
        core, frequent = self._list(self.core, skip), self._list(self.frequent, skip)
        head = COLUMN[self.source]
        phrases = prompt_phrases("game_glossary")

        def table(chosen: list[Candidate]) -> str:
            lines = [f"{'rows|' if rows else ''}{head}|中文|依据|类别|注" if self.source != "zh" else
                     f"{'rows|' if rows else ''}中文|依据|类别|注"]
            for c in chosen:
                name = c.term.zh if self.source == "zh" else f"{self._form(c.term)}|{c.term.zh}"
                note = c.term.note.replace("|", "/").replace("\n", " ")[:NOTE_CHARS]
                lines.append(f"{','.join(map(str, c.rows)) + '|' if rows else ''}{name}|{c.basis}|{c.term.label}|{note}")
            return "\n".join(lines)

        def whole(chosen: list[Candidate], dropped: int) -> str:
            return render(
                "fragment_game_glossary_v2.md",
                games="、".join(self.games),
                core=core or "（无）",
                frequent=frequent or "（无）",
                candidates_intro=phrases["window_intro" if rows else "fixed_intro"],
                candidates=table(chosen) + ("\n" + Template(phrases["dropped"]).substitute(count=dropped)
                                            if dropped else ""),
            ).strip() + "\n\n"

        order = sorted(cands, key=lambda c: c.rank)
        keep = len(order)
        text = whole(cands, 0)
        while keep and count_tokens(text) > limit:
            keep -= 1
            chosen = set(id(c) for c in order[:keep])
            text = whole([c for c in cands if id(c) in chosen], len(cands) - keep)
        return text

    def core_text(self) -> str:
        """The resident list alone (an agent's ``kb_read`` of the glossary)."""

        return self._list(self.core, "")


def _mentioned(t: GlossaryTerm, skip: str) -> bool:
    return bool(skip) and t.ja in skip


@dataclass(frozen=True)
class GlossaryPick:
    """The glossaries a task selected (`glossary_store.selection`) and its source language: what its correction run
    builds the windows' blocks from."""

    glossaries: tuple[Glossary, ...]
    source_language: str

    def run(self, texts: Sequence[str], count_tokens: Callable[[str], int], *, fixed: bool = False,
            warn: bool = True) -> "RunGlossary":
        """The run's glossary over the whole transcript ``texts``; ``fixed`` (``--continuity parallel``): every
        window gets the one fixed block. ``warn``: a stale build and missing readers are reported (a dry run
        previews quietly)."""

        if warn:
            _warn_stale(self.glossaries)
        sound = self.source_language == "ja" and (_sound_available() if warn else gm.available())
        view = GlossaryView(self.glossaries, texts, self.source_language, sound=sound)
        return RunGlossary(view, count_tokens, tuple(texts) if fixed else None)


@dataclass
class RunGlossary:
    """The selected glossaries of one correction run (`GlossaryPick.run`). ``fixed_texts`` makes every window get
    the one fixed block."""

    view: GlossaryView
    count_tokens: Callable[[str], int]
    fixed_texts: tuple[str, ...] | None = None

    def block(self, rows: Sequence[str], entry_details: str = "") -> str:
        from .prompts import _load_prompt_template
        from .routing.config import GAME_GLOSSARY_BLOCK_TOKENS

        if self.fixed_texts is not None:
            return self.view.fixed_block(self.fixed_texts, _load_prompt_template, self.count_tokens,
                                         GAME_GLOSSARY_BLOCK_TOKENS, entry_details)
        return self.view.block(rows, _load_prompt_template, self.count_tokens, GAME_GLOSSARY_BLOCK_TOKENS,
                               entry_details)


def _warn_stale(glossaries: Sequence[Glossary]) -> None:
    from finesub.reporting import current_reporter

    for g in glossaries:
        if g.stale():
            current_reporter().warning(
                "game-glossary-stale",
                f"the {g.game} glossary was built on {g.built}; its resident list (core names of the last two years)"
                " no longer covers the newest names",
                impact="names added since the build get no help from the glossary",
                action=f"update FineSub, then run `finesub knowledge glossary fetch {g.game}`: a newer glossary"
                       " ships with a newer release's list",
            )


def _sound_available() -> bool:
    if gm.available():
        return True
    from finesub.reporting import current_reporter

    current_reporter().warning(
        "game-glossary-readers",
        "the game glossary's sound matching needs the kb-phonetic extra",
        impact="correction windows get the glossary's lists and verbatim candidates, no sound-alike ones",
        action='from a source checkout: pip install -e ".[kb-phonetic]" (the installed app cannot add it yet)',
    )
    return False
