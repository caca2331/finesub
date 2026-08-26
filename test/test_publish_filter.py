"""Whatever `.gitignore` un-ignores must be stripped from the public snapshot.

`main` is a snapshot of `dev`'s tree, so every tracked file reaches the public
repository unless `scripts/publish-main.ps1` removes it -- gitignore cannot
say "ignored on one branch", it only governs untracked files. That makes
un-ignoring a path and forgetting the publisher a silent leak, onto a branch
this project deliberately never force-pushes.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = REPOSITORY_ROOT / "scripts" / "publish-main.ps1"

#: `[string[]]$Name = @("a", "b")`, the parameter defaults the publisher runs
#: with. Read from the script rather than restated here: a copy would keep
#: agreeing with itself long after the script stopped agreeing with it.
PARAMETER = r'\[string\[\]\]\${name}\s*=\s*@\(([^)]*)\)'
MEMBER = re.compile(r'"([^"]+)"')

#: Paths un-ignored on purpose *and* meant for the public tree. Adding a row
#: here is the deliberate act this test exists to make deliberate.
PUBLIC_BY_DESIGN: tuple[str, ...] = ()


def _array(name: str) -> list[str]:
    text = PUBLISHER.read_text(encoding="utf-8")
    match = re.search(PARAMETER.format(name=name), text)
    assert match, f"{PUBLISHER.name} no longer declares ${name}"
    return MEMBER.findall(match.group(1))


def _unignored() -> list[str]:
    lines = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line[1:].strip("/") for line in lines if line.startswith("!")]


def _covered_by(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


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


def test_no_public_document_links_into_a_stripped_path() -> None:
    """A markdown link out of the public tree and into `$PrivatePaths` dangles.

    `docs/archive/` and `docs/report/` are tracked on `dev` -- so every
    worktree has them and every link in them resolves *here* -- and stripped
    from each public snapshot. A document that ships may therefore *mention* a
    private note (a backticked path reads as "a local file you may not have"),
    but must not **link** to one: the link renders for a reader who cannot
    follow it, and neither guard next door objects. `test_doc_links` cannot:
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

    def ships(path: str) -> bool:
        return not _covered_by(path, private) or _covered_by(path, public_again)

    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=True,
    )
    names = [name for name in listing.stdout.decode("utf-8").split("\0") if name]

    dangling: list[str] = []
    for name in names:
        if not ships(name):
            continue
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
        "scripts/publish-main.ps1 strips it. Cite the note as a plain "
        "backticked path instead of a markdown link, or move the content into "
        "a tracked doc: " + ", ".join(dangling)
    )
