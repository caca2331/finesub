"""Whatever `.gitignore` un-ignores must be stripped from the public snapshot.

`main` is a snapshot of `dev`'s tree, so every tracked file reaches the public
repository unless `scripts/publish-main.ps1` removes it -- gitignore cannot
say "ignored on one branch", it only governs untracked files. That makes
un-ignoring a path and forgetting the publisher a silent leak, onto a branch
this project deliberately never force-pushes.
"""

from __future__ import annotations

from pathlib import Path
import posixpath
import re
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = REPOSITORY_ROOT / "scripts" / "publish-main.ps1"

#: `[string[]]$Name = @(`, the start of a parameter default the publisher runs
#: with. Read from the script rather than restated here: a copy would keep
#: agreeing with itself long after the script stopped agreeing with it.
PARAMETER = r'\[string\[\]\]\${name}\s*=\s*@\('
MEMBER = re.compile(r'"([^"]+)"')

#: Paths un-ignored on purpose *and* meant for the public tree. Adding a row
#: here is the deliberate act this test exists to make deliberate.
PUBLIC_BY_DESIGN: tuple[str, ...] = ()


def _array(name: str) -> list[str]:
    """The members of `$Name = @(...)`, read line by line with comments dropped.

    The first version matched `@\\(([^)]*)\\)` and so stopped at the first `)`
    anywhere inside the array -- including one in a comment. Every member after
    a parenthesised comment silently vanished from these checks
    (`agent-tasks/run-audit/evals` did, for weeks), which is a guard quietly
    narrowing its own scanning surface.
    """

    text = PUBLISHER.read_text(encoding="utf-8")
    match = re.search(PARAMETER.format(name=name), text)
    assert match, f"{PUBLISHER.name} no longer declares ${name}"
    members: list[str] = []
    rest = text[match.end() :]
    for line in rest.splitlines():
        code = line.split("#", 1)[0]
        if ")" in code:
            members += MEMBER.findall(code.split(")", 1)[0])
            return members
        members += MEMBER.findall(code)
    raise AssertionError(f"{PUBLISHER.name}: ${name} = @( is never closed")


def _unignored() -> list[str]:
    lines = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line[1:].strip("/") for line in lines if line.startswith("!")]


def _covered_by(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


def _stripped(path: str) -> bool:
    """Whether the publisher removes `path` from the public snapshot."""

    return _covered_by(path, _array("PrivatePaths")) and not _covered_by(
        path, _array("PublicExceptions")
    )


def _shipping_markdown() -> list[str]:
    """Every markdown file on `dev` (tracked or about to be) that reaches `main`."""

    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    )
    names = [name for name in listing.stdout.decode("utf-8").split("\0") if name]
    return [name for name in names if not _stripped(name)]


def test_every_unignored_path_is_stripped_from_the_public_snapshot() -> None:
    private = _array("PrivatePaths")
    leaking = [
        path
        for path in _unignored()
        if not _covered_by(path, private) and path not in PUBLIC_BY_DESIGN
    ]
    assert leaking == [], (
        "these are tracked on dev and would land on public main; add them to "
        "$PrivatePaths in scripts/publish-main.ps1, or to PUBLIC_BY_DESIGN "
        "here if that is the intent: " + ", ".join(leaking)
    )


def test_every_public_exception_sits_strictly_inside_a_private_path() -> None:
    """An exception to nothing, or to everything, reads like a decision.

    `$PublicExceptions` carves a subtree back out of `$PrivatePaths`. One that
    matches no private path does nothing at all, and the script would still
    publish exactly what it published before. One that repeats a private path
    whole puts every file back -- and the publisher's own leak check cannot
    object, because it reads `$PublicExceptions` as the statement of intent.
    A path that public belongs out of `$PrivatePaths`, not cancelled inside it.
    """

    private = _array("PrivatePaths")
    stray = [path for path in _array("PublicExceptions") if not _covered_by(path, private)]
    assert stray == [], (
        "these carve nothing out of $PrivatePaths: " + ", ".join(stray)
    )
    whole = [path for path in _array("PublicExceptions") if path in private]
    assert whole == [], (
        "these cancel a private path entirely instead of carving part of it "
        "out; drop them from $PrivatePaths instead: " + ", ".join(whole)
    )


def test_every_stripped_docs_subtree_is_local_only_to_the_link_guards() -> None:
    """`$PrivatePaths` and `LOCAL_ONLY` are one decision written down twice.

    The publisher is told what to strip; `test_doc_links` is told which
    directories a link may point into without the target having to exist
    *here*. A `docs/` subtree named in only one of them is green on `dev` --
    the files are there and every link resolves -- and red on the gate branch,
    where they are gone and the links dangle for real. The two spellings differ
    (a bare path here, a trailing slash there), which is exactly why this is
    checked rather than remembered.
    """

    from test.test_doc_links import LOCAL_ONLY

    stripped = {
        path.strip("/") for path in _array("PrivatePaths") if path.startswith("docs/")
    }
    local_only = {path.strip("/") for path in LOCAL_ONLY}
    missing = sorted(stripped - local_only)
    assert missing == [], (
        "these $PrivatePaths subtrees are stripped from the public snapshot but "
        "are missing from test_doc_links.LOCAL_ONLY, so a link into one of them "
        "stays green here and turns the gate branch red: " + ", ".join(missing)
    )


def test_no_public_document_links_into_a_stripped_path() -> None:
    """A markdown link out of the public tree and into `$PrivatePaths` dangles.

    `docs/archive/` and `docs/reports/` are tracked on `dev` -- so every
    worktree has them and every link in them resolves *here* -- and stripped
    from each public snapshot. A document that ships must not **link** to one:
    the link renders for a reader who cannot follow it, and neither guard next
    door objects. (Naming one without a link is ruled out too, by the test
    after this one.) `test_doc_links` cannot:
    it whitelists these directories precisely so the same run reaches the same
    verdict on `dev` and on the filtered gate branch, where the targets are
    gone. So the rule lives here, with the rest of what `$PrivatePaths` means.

    The asymmetry is deliberate and only runs one way. A private note linking
    into the public tree is fine -- it is only ever read from a checkout that
    has both.
    """

    from test.test_doc_links import CODE, LINE_ANCHOR, LINK

    private = _array("PrivatePaths")
    public_again = _array("PublicExceptions")

    dangling: list[str] = []
    for name in _shipping_markdown():
        document = REPOSITORY_ROOT / name
        text = CODE.sub("", document.read_text(encoding="utf-8"))
        for match in LINK.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "<")):
                continue
            resolved = (document.parent / LINE_ANCHOR.sub("", target)).resolve()
            try:
                relative = resolved.relative_to(REPOSITORY_ROOT).as_posix()
            except ValueError:
                continue  # Outside the repository; not ours.
            if _covered_by(relative, private) and not _covered_by(relative, public_again):
                dangling.append(f"{name} -> {target}")

    assert dangling == [], (
        "these links ship to readers who will not have the target, because "
        "scripts/publish-main.ps1 strips it. Move what the reader needs into a "
        "tracked public doc: " + ", ".join(dangling)
    )


#: Paths under a private directory that name a harness's own configuration
#: rather than a note of ours: Claude Code reads `.claude/settings.json` in any
#: project, so saying so is not pointing a reader at a file we withhold.
HARNESS_PATHS = (".claude/settings.json", ".claude/settings.local.json", ".claude/worktrees/")

#: Stripped directories whose name is also a git branch prefix: the docs say
#: "branch `explore/speaker-clustering`" throughout. Under these, a mention
#: counts only when the file is really there, as relative spellings do.
BRANCH_PREFIXES = ("explore",)


def test_no_public_document_names_a_file_in_a_stripped_path() -> None:
    """A public reader cannot open `docs/plans/x-plan.md`, linked or not.

    The link rule above leaves the plain mention, and 137 of them had piled up
    (owner ruling 2026-09-24: clear them all and hold the line). Naming the
    directory itself (`docs/plans/`) is fine -- it tells the reader the notes
    exist; naming a file in it sends them after something they cannot have.
    What they need goes into a tracked public doc instead.

    It reads at least what the project conventions' `check_docs.py --public`
    reads, so the two can be compared: a path spelled from the repository root,
    anywhere in the file (code blocks included), with a `<placeholder>` or a
    harness's own config path excused. It also reads the spelling relative to
    the document's own directory (`plans/x-plan.md` inside `docs/`,
    `../docs/archive/x.md` inside `tools/`), which that script does not: the
    first cleanup found such mentions it had passed over. A relative spelling
    counts only when the file really exists, because short prefixes collide
    with other things -- `archive/seg-start-onset-2026-07` in
    `docs/wt-parallelism.md` is a git tag, not `docs/archive/`.
    """

    private = [path.strip("/") for path in _array("PrivatePaths")]
    # Full-width punctuation ends a path too: these docs are Chinese, and
    # `docs/reports/（…` must not be read as a file named `（…`.
    tail = r")/[^\s`'\")\]>|,;（）、，。；：「」]+"

    named: list[str] = []
    for name in _shipping_markdown():
        directory = posixpath.dirname(name) or "."
        spellings = {prefix: prefix for prefix in private}
        spellings.update({posixpath.relpath(prefix, directory): prefix for prefix in private})
        mention = re.compile(
            r"(?<![\w./-])(?:\./)?(" + "|".join(re.escape(s) for s in spellings) + tail
        )
        text = (REPOSITORY_ROOT / name).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            for match in mention.finditer(line):
                written = match.group(0).removeprefix("./").rstrip(".")
                spelled = match.group(1)
                path = spellings[spelled] + written[len(spelled) :]
                if "<" in path or path.startswith(HARNESS_PATHS) or not _stripped(path):
                    continue
                loose = spelled != spellings[spelled] or spellings[spelled] in BRANCH_PREFIXES
                if loose and not (REPOSITORY_ROOT / path).exists():
                    continue
                named.append(f"{name}:{number} {written}")

    assert named == [], (
        "these public documents name a file the publisher strips; distil what "
        "the reader needs into a tracked public doc and drop the path: "
        + ", ".join(named)
    )


#: A maintainer's home directory -- the three spellings a Windows machine
#: produces, plus macOS `/Users/<name>` and Linux `/home/<name>` -- with any
#: `<placeholder>` segment excused (`C:\Users\<你>\...` is how the manual tells
#: a user where their own files are). No user name is written here: this file
#: ships too, and the check has to hold on CI, whose home is someone else's.
HOME_PATH = re.compile(
    r"(?:(?:\b[A-Za-z]:[/\\]|/[a-z]/|/mnt/[a-z]/)Users[/\\]|(?<![\w./-])/Users/|(?<![\w./-])/home/)"
    r"(?!<)[^\s/\\`'\")\]（）、，。]+",
    re.IGNORECASE,
)
PRIVATE_ROOTS = ("~/.agents",)


def test_no_public_document_carries_a_local_home_path() -> None:
    """A path on the maintainer's machine is broken for everyone else.

    The publisher's gate would not catch it, and a reader copying the command
    gets a path that does not exist on their machine. Tracked READMEs use a
    placeholder (`<repo>`, `<models>`) instead; only markdown that ships is
    scanned -- private notes may record where evidence sat on this machine.
    """

    found: list[str] = []
    for name in _shipping_markdown():
        text = (REPOSITORY_ROOT / name).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            hits = [m.group(0) for m in HOME_PATH.finditer(line)]
            hits += [root for root in PRIVATE_ROOTS if root in line]
            found += [f"{name}:{number} {hit}" for hit in hits]

    assert found == [], (
        "these public documents carry a path from one machine; use a "
        "placeholder such as <repo> or <models>: " + ", ".join(found)
    )
