"""Relative links between tracked docs must resolve.

Moving a doc silently breaks every relative link inside it -- the file still
renders, the link just goes nowhere, and nothing in a text repo notices. It
happened the same afternoon this test was written: `docs/tools/prompt-iterate.md`
moved up one level and took six `../` links with it.
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

#: Directories that exist on a developer's machine but are not tracked, so a
#: link into them is correct and unresolvable in a fresh clone alike.
LOCAL_ONLY = ("assets/", "docs/archive/", "docs/report/", "knowledge/", "out/", "data/")


def _tracked_markdown() -> list[Path]:
    # `--others --exclude-standard` as well as the index: a doc written but not
    # yet committed is exactly when its links are most likely wrong, and asking
    # only the index made this test pass on a file with a deliberately broken
    # link. `--exclude-standard` still leaves `docs/archive/` and the rest of
    # the gitignored local notes out.
    #
    # `-z`, because the default output quotes and octal-escapes any path with a
    # non-ASCII character -- and `examples/knowledge/` is full of them.
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    )
    names = listing.stdout.decode("utf-8").split("\0")
    return [REPOSITORY_ROOT / name for name in names if name]


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
            window = text[min(line_start, match.start() - HINT_REACH) : match.start()]
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


def test_every_tracked_doc_appears_in_the_index() -> None:
    """`docs/README.md` states this as a rule, so something has to hold it.

    A doc nobody indexed is a doc nobody finds: `CLAUDE.md`'s index is what an
    agent reads to decide what to open, and a missing row means the file is
    invisible exactly when it is needed. Archive and report are local notes and
    deliberately out.
    """

    index = (REPOSITORY_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    tracked = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in _tracked_markdown()
        if path.is_relative_to(REPOSITORY_ROOT / "docs")
    }
    tracked = {
        name
        for name in tracked
        if "/archive/" not in name and "/report/" not in name
    }

    missing = sorted(name for name in tracked if f"`{name}`" not in index)
    assert missing == [], (
        "add these to the docs index in CLAUDE.md: " + ", ".join(missing)
    )


def test_relative_links_between_tracked_docs_resolve() -> None:
    broken: list[str] = []
    for document in _tracked_markdown():
        text = CODE.sub("", document.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "<")):
                continue
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
