"""Relative links between tracked docs must resolve, and `§N` must land.

Moving a doc silently breaks every relative link inside it -- the file still
renders, the link just goes nowhere, and nothing in a text repo notices. It
happened the same afternoon this test was written: `docs/tools/prompt-iterate.md`
moved up one level and took six `../` links with it.

The guards live in one file because they answer the same question -- "which
document owns the number in front of me". They are *near* copies, not one
rule: the markdown side only accepts a document named in a link or backticks
(`DOC_HINT`), the source side accepts any bare `.md` string in the window
(`DOC_NAME`), because comments quote paths without markup. Keep that asymmetry
deliberate -- two rules drifting apart unnoticed is the failure this family
exists to catch.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

#: Markdown inline links, minus the `#fragment`. Bare `[x](y)` only -- reference
#: definitions and autolinks do not carry repository paths here.
LINK = re.compile(r"\]\(([^)#\s]+?)(?:#[^)]*)?\)")

#: Fenced blocks and inline spans, stripped before scanning. These docs quote
#: shell one-liners, and a `rg "…\(llm|asr_align\)"` in one of them looks
#: exactly like a link. A checker that cries wolf is one people learn to ignore.
CODE = re.compile(r"```.*?```|~~~.*?~~~|`[^`\n]*`", re.DOTALL)

#: A trailing `:228` on a link target: the line to open, not part of the path.
LINE_ANCHOR = re.compile(r":\d+$")

#: Directories a link may point into without the target having to exist in the
#: tree being checked: either untracked local material, or -- for the two
#: `docs/` ones -- notes tracked on `dev` but stripped from the public snapshot
#: by `scripts/publish-main.ps1`, where this same test runs on the gate branch.
LOCAL_ONLY = ("assets/", "docs/archive/", "docs/report/", "knowledge/", "out/", "data/")


def _tracked_markdown() -> list[Path]:
    # `--others --exclude-standard` as well as the index: a doc written but not
    # yet committed is exactly when its links are most likely wrong, and asking
    # only the index made this test pass on a file with a deliberately broken
    # link.
    #
    # `-z`, because the default output quotes and octal-escapes any path with a
    # non-ASCII character -- and `examples/knowledge/` is full of them.
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    )
    names = [name for name in listing.stdout.decode("utf-8").split("\0") if name]
    # LOCAL_ONLY again, now as "not ours to scan" -- the two readings coincide,
    # and a third copy of the list would be a third thing to keep in step.
    # `docs/archive/` and `docs/report/` are tracked on `dev` so worktrees and
    # fresh clones carry them, and stripped from the public snapshot: they
    # exist here and not on `main`, while one test run has to reach the same
    # verdict on both. Scanning them would let a dangling link inside a
    # historical note fail CI on the gate branch, where the files are gone.
    return [REPOSITORY_ROOT / name for name in names if not name.startswith(LOCAL_ONLY)]


#: `§12.5` and friends. These docs cite each other by section constantly, and
#: a section number is an anchor with no link behind it -- nothing renders
#: differently when it points at a heading that no longer exists.
SECTION = re.compile(r"§\s?(\d+(?:\.\d+)*)")
#: `## 3.2 …` / `### 4.1、…`
HEADING = re.compile(r"^#{1,6}\s+(\d+(?:\.\d+)*)[.、 ]")
#: A `.md` path named as a link or in backticks; both are used to say "that doc".
DOC_HINT = re.compile(r"\[[^\]]*\]\(([^)]+\.md)\)|`([A-Za-z0-9_./-]+\.md)`")
#: How far back a document name can sit and still own the number after it.
#: The window also always reaches the start of the current line: an index table
#: names the document in its first cell and cites sections at the far end of the
#: row, and a fixed character budget missed one such citation by 13 characters.
HINT_REACH = 200


def _headings(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        m.group(1)
        for line in path.read_text(encoding="utf-8").split("\n")
        if (m := HEADING.match(line))
    }


def test_every_section_reference_points_at_a_real_heading() -> None:
    """`§N` is an anchor with nothing to break when it goes stale.

    Splitting `llm_local_agent.md` into four proved it: forty intra-document
    references became cross-document overnight, every one of them still
    rendering perfectly and pointing nowhere. The convention this enforces is
    that a cross-document reference names its document -- ``` `file.md` §3 ```
    -- so both a reader and this test can follow it.

    Same document wins when the number exists locally; a named document we do
    not have (an external report) is skipped rather than guessed at.
    """

    cache: dict[Path, set[str]] = {}
    broken: list[str] = []
    for document in _tracked_markdown():
        text = document.read_text(encoding="utf-8")
        own = cache.setdefault(document, _headings(document))
        for match in SECTION.finditer(text):
            number = match.group(1)
            if number in own:
                continue
            line_start = text.rfind("\n", 0, match.start()) + 1
            # max(0, …): a citation inside the first HINT_REACH characters
            # would otherwise produce a negative slice start, which Python
            # resolves from the end of the text -- an empty window, and the
            # citation silently treated as unqualified.
            window_start = max(0, min(line_start, match.start() - HINT_REACH))
            window = text[window_start : match.start()]
            hints = [a or b for a, b in DOC_HINT.findall(window)]
            target = document
            for name in reversed(hints):
                candidates = [document.parent / name, REPOSITORY_ROOT / name]
                found = next((c for c in candidates if c.exists()), None)
                if found is None:
                    target = None  # names a doc we do not have
                break
            else:
                found = None
            if hints:
                if target is None:
                    continue
                target = found.resolve()
            if not cache.setdefault(target, _headings(target)):
                continue  # a document that numbers nothing
            if number not in cache[target]:
                line = text[: match.start()].count("\n") + 1
                source = document.relative_to(REPOSITORY_ROOT).as_posix()
                broken.append(
                    f"{source}:{line} §{number} -> "
                    f"{target.relative_to(REPOSITORY_ROOT).as_posix()}"
                )

    assert broken == []


#: `` `bench-baselines.md` 二十二 `` / `第十八节` -- the same citation with a
#: Chinese numeral, which `SECTION` cannot see because these headings carry no
#: `§`. Backticks optional, like `DOC_NAME`: source comments write the path bare
#: (`docs/bench-baselines.md 二十二`), and the first version of this guard --
#: markdown-only, backticks required -- was blind to all eleven of them.
CJK_SECTION = re.compile(
    r"`?([A-Za-z0-9_./-]+\.md)`?\s*(?:的\s*)?(?:第)?([一二三四五六七八九十]+)"
)
#: `## 二十二、A1 组批本体…`
CJK_HEADING = re.compile(r"^#{1,6}\s+([一二三四五六七八九十]+)、")


def _cjk_headings(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        m.group(1)
        for line in path.read_text(encoding="utf-8").split("\n")
        if (m := CJK_HEADING.match(line))
    }


def test_every_chinese_numeral_section_reference_lands() -> None:
    """The half of the section convention `§N` never covered.

    One document numbers its sections in Chinese (`bench-baselines.md`, 24 of
    them) and is cited that way twenty-four times -- from `CLAUDE.md`, from
    other docs, and from a dozen source comments. None of those citations was
    checked by anything: `docs-reorg-plan.md` therefore had to write "never
    renumber that file" as a rule people must remember. A guard is the better
    half of that rule.

    **Both surfaces, because the first version had only one.** Markdown-only
    and backticks-required, it passed while eleven bare-path citations sat in
    `src/`, `test/` and `tools/bench/` -- the same shape as
    `refactor-followups.md`'s fifth lesson, that a guard's scanning surface is
    part of the guard. Sources are read with the same file set as the `§N`
    source guard.

    Numerals are compared as written rather than parsed -- both sides use the
    same spelling, and `十九` cannot be mistaken for `十` because the character
    class is greedy. Documents that number nothing this way are skipped, so
    adding the style elsewhere opts that document in automatically.

    ⚠ Same resolution rule as the `§N` guard, same limitation: a citation whose
    document name does not resolve next to the citing file or at the repository
    root is skipped rather than guessed at.
    """

    broken: list[str] = []
    for document in [*_tracked_markdown(), *_tracked_sources()]:
        text = document.read_text(encoding="utf-8", errors="replace")
        for match in CJK_SECTION.finditer(text):
            name, number = match.group(1), match.group(2)
            target = next(
                (
                    candidate
                    for candidate in (document.parent / name, REPOSITORY_ROOT / name)
                    if candidate.exists()
                ),
                None,
            )
            if target is None:
                continue
            headings = _cjk_headings(target.resolve())
            if not headings or number in headings:
                continue
            line = text[: match.start()].count("\n") + 1
            source = document.relative_to(REPOSITORY_ROOT).as_posix()
            broken.append(f"{source}:{line} {name} 第{number}节 -> no such heading")

    assert broken == []


def test_every_tracked_doc_appears_in_both_indexes() -> None:
    """Two indexes, two readers, and a name may go missing from neither.

    `docs/README.md` carries the map -- every tracked doc with its topic and
    status -- for whoever maintains the docs. `CLAUDE.md` carries the routing
    list: the same names grouped by domain, plus the few judgement hints, and
    that is what an agent has in context when it decides what to open.
    Descriptions live on one side by design; *names* live on both, because a
    name missing from either side makes the file invisible to that side's
    reader. Checking only one side was the earlier shape of this test, and it
    would have passed happily while the map went stale. Archive and report
    never reach this test -- `_tracked_markdown` drops them -- because they
    are local notes.
    """

    tracked = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in _tracked_markdown()
        if path.is_relative_to(REPOSITORY_ROOT / "docs")
    }

    index = (REPOSITORY_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    missing = sorted(name for name in tracked if f"`{name}`" not in index)
    assert missing == [], (
        "add these to the docs index in CLAUDE.md: " + ", ".join(missing)
    )

    # The map spells its paths relative to `docs/` -- `manual/env.md`, not
    # `docs/manual/env.md` -- and does not list itself: it *is* the map.
    document_map = (REPOSITORY_ROOT / "docs/README.md").read_text(encoding="utf-8")
    mapped = {name[len("docs/") :] for name in tracked} - {"README.md"}
    unmapped = sorted(name for name in mapped if f"`{name}`" not in document_map)
    assert unmapped == [], (
        "add these to the 文档地图 in docs/README.md: " + ", ".join(unmapped)
    )


def test_relative_links_between_tracked_docs_resolve() -> None:
    broken: list[str] = []
    for document in _tracked_markdown():
        text = CODE.sub("", document.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "<")):
                continue
            # `file.ps1:228` -- a line anchor, and a link the editor follows.
            # The path in front of it still has to exist, which is the whole
            # point of checking; the number is not ours to verify.
            target = LINE_ANCHOR.sub("", target)
            resolved = (document.parent / target).resolve()
            try:
                relative = resolved.relative_to(REPOSITORY_ROOT).as_posix()
            except ValueError:
                # Points outside the repository; not ours to check.
                continue
            if relative.startswith(LOCAL_ONLY):
                continue
            if not resolved.exists():
                source = document.relative_to(REPOSITORY_ROOT).as_posix()
                broken.append(f"{source} -> {target}")

    assert broken == []


#: Source trees whose comments cite docs by section. `tools/` is in despite
#: being maintained on demand only -- a citation that no longer resolves is not
#: tool maintenance, it is the same rot this file exists to catch, and the fix
#: is one line.
#:
#: `test/` used to be excluded on the grounds that it "cites the code under
#: test, not prose". That was simply false: 90 section citations live there and
#: fifty test files name a document. Adding it (2026-09-03) turned up three
#: stale ones on the first run -- two pointing into subsections the agent tool
#: protocol document lost in a restructure, one written as a range whose lower
#: end never existed. (Deliberately spelled without the section sign and
#: without naming those documents: this file is now inside its own scanning
#: surface, and an example citation here would be read as a real one.)
SOURCE_ROOTS = ("src/", "cli/", "tools/", "scripts/", "test/")

#: Data files ship citations too. `model_routes.toml` outlived `§16.5` by a
#: full document split while every `.py` around it was being repaired, because
#: the first version of this guard only globbed code.
SOURCE_GLOBS = ("*.py", "*.ps1", "*.ts", "*.tsx", "*.toml", "*.psv", "*.json")

#: A document named in a comment -- bare path, backticks or reST double
#: backticks all reduce to the same thing here.
DOC_NAME = re.compile(r"([A-Za-z0-9_./-]+\.md)")


def _tracked_sources() -> list[Path]:
    listing = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            *SOURCE_GLOBS,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    )
    names = [name for name in listing.stdout.decode("utf-8").split("\0") if name]
    return [REPOSITORY_ROOT / name for name in names if name.startswith(SOURCE_ROOTS)]


def test_every_section_reference_in_source_points_at_a_real_heading() -> None:
    """Code cites docs by section too, and nothing was watching those.

    Splitting `llm_local_agent.md` into four renumbered its sections; six
    citations in `src/` kept pointing at sections that had stopped existing --
    12.5.3, 15.5, 16.5 and friends -- and a seventh in `tools/` outlived the
    section it named by a different route, its target having been compressed
    away. One of the six is the `NotImplementedError` message a user sees when
    they set `pseudo-conversational`: it handed them a section number they
    could not find. Imports still resolved, tests stayed green.

    Only doc-qualified citations are checkable. A bare section number in a
    comment has no local numbering to fall back on, so this test cannot resolve
    it and does not guess -- naming the document is what makes one verifiable.

    So read a pass narrowly. Measured over the trees below: 141 section
    numbers, 21 of them qualified and checked here -- 15%. Most of the rest sit
    in `tools/segmentation_gold/labels/*.json`, 71 real citations of
    `docs/segmentation-gold.md` written without the document name; naming it
    there would take the checked share past 65% in one edit. Green means "no
    *qualified* citation is dangling", not "the source cites no dead sections"
    -- `tools/qwen3_explore` has four bare ones pointing at a section its
    `FINDINGS.md` lost. The fix is to write the document name, not to teach
    this test to guess.
    """

    cache: dict[Path, set[str]] = {}
    broken: list[str] = []
    for source in _tracked_sources():
        text = source.read_text(encoding="utf-8", errors="replace")
        for match in SECTION.finditer(text):
            number = match.group(1)
            line_start = text.rfind("\n", 0, match.start()) + 1
            # Same max(0, …) clamp as the markdown guard above.
            window_start = max(0, min(line_start, match.start() - HINT_REACH))
            window = text[window_start : match.start()]
            names = DOC_NAME.findall(window)
            if not names:
                continue  # unqualified; see the docstring
            # Sibling first, mirroring the markdown guard: `tools/qwen3_explore`
            # cites its own `FINDINGS.md`, and resolving only from the root
            # would file that under "a document we do not have" and skip it.
            candidates = [
                source.parent / names[-1],
                REPOSITORY_ROOT / names[-1],
                REPOSITORY_ROOT / "docs" / names[-1],
            ]
            target = next((c for c in candidates if c.exists()), None)
            if target is None:
                # A document this tree does not carry: an archived plan, or one
                # of the local-only trees `publish-main.ps1` strips. Same
                # reading as the markdown guard above -- not ours to check.
                continue
            if not cache.setdefault(target, _headings(target)):
                continue  # a document that numbers nothing
            if number not in cache[target]:
                line = text[: match.start()].count("\n") + 1
                where = source.relative_to(REPOSITORY_ROOT).as_posix()
                broken.append(
                    f"{where}:{line} section {number} -> "
                    f"{target.relative_to(REPOSITORY_ROOT).as_posix()}"
                )

    assert broken == []


def test_links_inside_the_local_only_notes_resolve_where_those_notes_exist() -> None:
    """Archiving a doc moves it one level down and breaks every `../` in it.

    `_tracked_markdown` skips these directories on purpose -- the same run has
    to reach the same verdict on `dev` and on the filtered gate branch, and
    there the files are gone. That exemption also means nothing checks the
    links *inside* a note, which is why archiving one has twice ended with a
    sweep afterwards (`4dd60e2` fixed eighteen at once).

    So: check them only where they exist. On the gate branch the directories
    are absent and this test has nothing to do, which is exactly the property
    the exemption was protecting.

    **Only `.md` targets.** That is the class a move breaks: sibling documents
    that were one `../` away and are now two. The code paths these notes cite
    went stale years earlier for a different reason -- the 2026-08 rename --
    and rewriting them inside a historical record would falsify what the plan
    actually said. Artefact directories (`out/`, `data/`) are generated and
    legitimately absent. Neither is what archiving broke, and folding them in
    would make this test noisy enough to be silenced.
    """

    broken: list[str] = []
    for directory in ("docs/archive", "docs/report"):
        root = REPOSITORY_ROOT / directory
        if not root.is_dir():
            continue
        for document in sorted(root.rglob("*.md")):
            text = CODE.sub("", document.read_text(encoding="utf-8"))
            for match in LINK.finditer(text):
                target = match.group(1).strip()
                if target.startswith(("http://", "https://", "mailto:", "<")):
                    continue
                path = LINE_ANCHOR.sub("", target)
                if not path.endswith(".md"):
                    continue
                resolved = (document.parent / path).resolve()
                try:
                    resolved.relative_to(REPOSITORY_ROOT)
                except ValueError:
                    continue  # Outside the repository; not ours.
                if not resolved.exists():
                    source = document.relative_to(REPOSITORY_ROOT).as_posix()
                    broken.append(f"{source} -> {target}")

    assert broken == [], (
        "a relative link in a local-only note points nowhere -- moving a doc "
        "into docs/archive/ puts it one directory deeper, so every ../ in it "
        "needs one more level: " + ", ".join(broken)
    )
