"""Step 1 of the README: when each term was added, for "the last two years" (``build.py --dates``).

    # Genshin: genshin-langdata's full git history (run inside a full clone of it)
    git log --reverse --format='@@C %cI %s' -p -U15 -- dataset/dictionary > <history.txt>
    python <this> genshin <history.txt> <out dir>
    # Star Rail: StarRailRes' history for what it indexes, text-table snapshots for the rest
    python <this> starrailres <StarRailRes clone> <terms.jsonl> <out.json>
    python <this> snapshots <terms.jsonl> --versions <versions.json> --by zh --prefix starrail: \
        --url 'https://gitlab.com/Dimbreath/turnbasedgamedata/-/raw/{sha}/TextMap/TextMapCHS.json' --out <out.json>
    # ZZZ: ZenlessData snapshots by key; Enka's avatar list for when an agent became playable
    python <this> snapshots <terms.jsonl> --versions <versions.json> --by key --prefix zzz: \
        --url 'https://git.mero.moe/dimbreath/ZenlessData/raw/commit/{sha}/TextMap/TextMap_JATemplateTb.json' --out <out.json>
    python <this> enka <EnkaNetwork/API-docs clone> --path store/zzz/avatars.json --prefix zzz:Partner_Name_ --out <out.json>

``build.py`` takes ``--dates`` more than once and reads a term's date by its ``id``, then by its ``en`` name (the
Genshin file); the later of the files' dates wins, so a playable date overrides an earlier appearance.
``versions.json`` is ``{version: [commit, date]}``: the first commit of each game version in the data repository
(its commit messages carry the version, ``OSPRODWin2.6.0_...``).

Genshin (``genshin``) writes ``ld-dates.json``, per ``en`` key of the current dataset:

  added     the earliest commit adding a line with the word's current ``en``, ``ja`` or ``zhCN`` (a renamed English
            key keeps the old date); 2022-06 is the bulk import of older words
  playable  the earliest commit adding a ``tags`` line with ``character-main`` inside the word's object (the nearest
            ``en:`` line above it in the hunk); only for words that carry the tag now
  snapshot  the first "Version X.Y" snapshot (the last commit per version message, readable from 5.6) holding the
            word: the fallback when the history gives no ``added``
  date      the later of ``added`` (or ``snapshot``) and ``playable``

Also ``ld-versions.json`` with the version snapshots.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import _clean, parse_ts_literal  # noqa: E402


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


STR = r'"((?:[^"\\]|\\.)*)"'


def history_dates(history: Path) -> tuple[dict, dict, dict, dict]:
    """First dates of added ``en`` / ``ja`` / ``zhCN`` strings, and of a ``character-main`` tags line per ``en``."""

    first = {"en": {}, "ja": {}, "zhCN": {}}
    playable: dict[str, str] = {}
    cur, en = "", None
    for line in history.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("@@C "):
            cur = line[4:14]
            continue
        if line.startswith("@@"):
            en = None
            continue
        if not line or line[0] not in " +":
            continue
        m = re.match(r"[ +]\s+(en|ja|zhCN):\s*" + STR, line)
        if m:
            key, value = m.group(1), m.group(2).replace('\\"', '"')
            if key == "en":
                en = value
            if line[0] == "+":
                first[key].setdefault(value, cur)
        elif line[0] == "+" and "tags:" in line and '"character-main"' in line and en:
            playable.setdefault(en, cur)
    return first["en"], first["ja"], first["zhCN"], playable


def version_snapshots() -> dict:
    last: dict[str, tuple[str, str]] = {}
    for line in git("log", "--reverse", "--format=%H %cI %s").splitlines():
        h, d, s = line.split(" ", 2)
        m = re.search(r"Version (\d+\.\d+)", s)
        if m:
            last[m.group(1)] = (h, d[:10])  # the sequel commit wins
    first: dict[str, tuple[str, str]] = {}
    playable: dict[str, tuple[str, str]] = {}
    for version, (h, d) in sorted(last.items(), key=lambda kv: kv[1][1]):
        for path in git("ls-tree", "--name-only", h, "dataset/dictionary/").split():
            if not path.endswith(".ts"):
                continue
            try:
                words = parse_ts_literal(git("show", f"{h}:{path}"))
            except ValueError:
                continue  # older snapshots use another layout
            for w in words:
                en = w.get("en")
                if not en:
                    continue
                first.setdefault(en, (version, d))
                if "character-main" in (w.get("tags") or []):
                    playable.setdefault(en, (version, d))
    return {"first": first, "playable": playable, "versions": sorted(last.items(), key=lambda kv: kv[1][1])}


def genshin(history: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    first_en, first_ja, first_zh, playable = history_dates(history)
    snaps = version_snapshots()
    current = {}
    for path in Path("dataset/dictionary").glob("*.ts"):
        for w in parse_ts_literal(path.read_text(encoding="utf-8")):
            if w.get("en"):
                current[w["en"]] = w
    dates = {}
    for en, w in current.items():
        found = [d for d in (first_en.get(en), first_ja.get(w.get("ja", "")), first_zh.get(w.get("zhCN", ""))) if d]
        added = min(found) if found else None
        snapshot = snaps["first"].get(en, (None, None))[1]
        base = added or snapshot or ""
        play = playable.get(en) if "character-main" in (w.get("tags") or []) else None  # a tag removed later
        dates[en] = {"added": added, "playable": play, "snapshot": snapshot, "date": max(base, play or "")}
    (out / "ld-dates.json").write_text(json.dumps(dates, ensure_ascii=False), encoding="utf-8")
    (out / "ld-versions.json").write_text(json.dumps(snaps, ensure_ascii=False), encoding="utf-8")
    print(len(dates), "words;", sum(1 for d in dates.values() if not d["added"]), "without an added date;",
          sum(1 for d in dates.values() if not d["date"]), "undated;", len(playable), "playable tag commits")


# --- the other games: first appearance in dated snapshots ----------------------------------------------------------


def _terms(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save(out: Path, dates: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dates, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    print(f"{out}: {len(dates)} dated")


def starrailres(repo: Path, terms_file: Path, out: Path) -> None:
    """``{term id: {"date"}}`` for the StarRailRes terms: the first commit whose ``index_new/jp/<file>.json`` holds
    the id. StarRailRes indexes what a released version can show, so for characters this is (within a day or two)
    the version they became playable in; run in a clone with full history (``--filter=blob:none`` is enough)."""

    wanted: dict[str, set[str]] = {}
    for t in _terms(terms_file):
        _, file, key = t["id"].split(":", 2)
        wanted.setdefault(file, set()).add(key)
    dates = {}
    for file, keys in wanted.items():
        path = f"index_new/jp/{file}.json"
        commits = git("-C", str(repo), "log", "--reverse", "--format=%H %cI", "--", path).split("\n")
        for line in filter(None, commits):
            sha, when = line.split(" ", 1)
            try:
                data = json.loads(git("-C", str(repo), "show", f"{sha}:{path}"))
            except json.JSONDecodeError:
                continue
            for key in keys & set(data if isinstance(data, dict) else ()):
                dates.setdefault(f"starrail:{file}:{key}", {"date": when[:10], "basis": f"StarRailRes {sha[:10]}"})
    _save(out, dates)


def snapshots(terms_file: Path, versions_file: Path, url: str, by: str, out: Path, prefix: str) -> None:
    """``{term id: {"date", "basis"}}``: the first game-version snapshot of a text table (``--versions``:
    ``{version: [commit, date]}``, ``--url`` with ``{sha}``) that holds the term, matched by the table key its id
    names (``--by key``, ZenlessData's ``zzz:<key>``) or by its Chinese name among the values (``--by zh``, the
    hashed tables of turnbasedgamedata). Only terms whose id starts with ``--prefix`` (comma-separated) are dated. A term the first
    snapshot already holds is dated to it, which is all "older than two years" needs. Tables appear in the data
    a version or more before a character is playable: ``build.py``'s ``Game.released`` corrects those."""

    import tempfile
    import urllib.request

    terms = [t for t in _terms(terms_file) if t["id"].startswith(tuple(prefix.split(",")))]
    versions = sorted(json.loads(versions_file.read_text(encoding="utf-8")).items(), key=lambda kv: kv[1][1])
    dates: dict[str, dict] = {}
    for version, (sha, when) in versions:
        request = urllib.request.Request(url.format(sha=sha), headers={"User-Agent": "finesub-glossary-dates"})
        with tempfile.TemporaryFile() as tmp:
            with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310 (a maintainer-given URL)
                while chunk := response.read(1 << 20):
                    tmp.write(chunk)
            tmp.seek(0)
            table = json.loads(tmp.read().decode("utf-8"))
        values = {_clean(v) for v in table.values() if isinstance(v, str)}  # as extract.py wrote the names
        for t in terms:
            if t["id"] in dates:
                continue
            key = t["id"].split(":", 1)[1]
            # a concept (found by its Chinese, no key of its own) is looked up by either name among the values
            if key in table if by == "key" and ":concept:" not in t["id"] else {t["zh"], t["ja"]} & values:
                dates[t["id"]] = {"date": when[:10], "basis": f"{version} snapshot {sha[:10]}"}
        print(version, when[:10], len(dates), "dated", file=sys.stderr, flush=True)
    _save(out, dates)


def enka(repo: Path, path: str, prefix: str, out: Path) -> None:
    """``{prefix + avatar id: {"date"}}``: the first commit of EnkaNetwork/API-docs' ``store/<game>/avatars.json``
    (``path``) listing the avatar. Enka shows played characters, so a later commit than its first is a playable date;
    the avatars of the first commit (2025-01-29 for ZZZ) only get an upper bound and are left out."""

    commits = [c.split(" ", 1) for c in git("-C", str(repo), "log", "--reverse", "--format=%H %cI", "--", path)
               .split("\n") if c]
    first = set(json.loads(git("-C", str(repo), "show", f"{commits[0][0]}:{path}")))
    dates = {}
    for sha, when in commits[1:]:
        for key in set(json.loads(git("-C", str(repo), "show", f"{sha}:{path}"))) - first:
            dates.setdefault(f"{prefix}{key}", {"date": when[:10], "basis": f"Enka {sha[:10]}"})
    _save(out, dates)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("genshin", help="genshin-langdata's history (run inside the clone)")
    g.add_argument("history", type=Path)
    g.add_argument("out", type=Path, help="directory for ld-dates.json and ld-versions.json")
    r = sub.add_parser("starrailres")
    r.add_argument("repo", type=Path)
    r.add_argument("terms", type=Path)
    r.add_argument("out", type=Path)
    n = sub.add_parser("snapshots")
    n.add_argument("terms", type=Path)
    n.add_argument("--versions", type=Path, required=True)
    n.add_argument("--url", required=True)
    n.add_argument("--by", choices=("key", "zh"), required=True)
    n.add_argument("--prefix", default="")
    n.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("enka")
    e.add_argument("repo", type=Path)
    e.add_argument("--path", required=True)
    e.add_argument("--prefix", required=True)
    e.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "genshin":
        genshin(a.history, a.out)
    elif a.cmd == "starrailres":
        starrailres(a.repo, a.terms, a.out)
    elif a.cmd == "snapshots":
        snapshots(a.terms, a.versions, a.url, a.by, a.out, a.prefix)
    else:
        enka(a.repo, a.path, a.prefix, a.out)


if __name__ == "__main__":
    main()
