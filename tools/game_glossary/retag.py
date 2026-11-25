"""Rerun the build's last step on a built glossary: the rule tags and the marks (maintainers only).

    PYTHONPATH=<checkout>/src python tools/game_glossary/retag.py <game> <glossary.json> --out <glossary.json>

Every term's fact tags (``kind:``, ``important:``, ``short-reading``, ``everyday:``, ``lines:``, ``mentions:``,
``noise:``, ``famous:``, ``skip:``...) are kept; its rule tags (``risk:``, ``evidence:``, ``override:``) and its
marks (``skip``, ``rare``, ``exact``, ``fold``, ``core``, ``famous``, ``core_skip``) are written again by ``build.apply_rules``
from the current ``Game.tiers``, ``Game.risk_rule`` and rule constants, so a changed rule or override is applied
without the upstream, the text tables or a new probe; the noise table is kept whole. What it cannot do: a name
moved from rare to common has no background rate until the next full build, and a name newly among the ones the
probe must cover (``build.probe_pool``: the resident-list pool, important names the risk rule leaves without
evidence) and not probed needs the probe (it is listed, and the file is not written).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build  # noqa: E402  (the sibling script)


def retag(game: build.Game, data: dict) -> tuple[dict, collections.Counter]:
    """``data`` with its terms re-tagged, and how many terms moved (``old->new`` tier, ``fold+`` / ``fold-``).

    ``fold`` asks this machine's dictionary again whether a name's hiragana is a word (``_hiragana_word``): with
    other reader versions than the ones that last judged it (``provenance.fold_readers``, written here; else the
    build's ``readers``) it warns, and the fold moves show what changed."""

    def tier(t: dict) -> str:
        return "skip" if t.get("skip") else "rare" if t.get("rare") else "exact" if t.get("exact") else "common"

    terms = data["terms"]
    before = [(tier(t), bool(t.get("fold"))) for t in terms]
    pool = {(t["ja"], t["zh"]) for t in build.probe_pool(terms)}
    provenance = data.setdefault("provenance", {})
    recorded, here = provenance.get("fold_readers") or provenance.get("readers"), build._reader_versions()
    if recorded and recorded != here:
        print(f"warning: this machine's readers {here} are not the ones that judged fold {recorded}: fold is"
              " judged again with them (the fold+ / fold- moves)", file=sys.stderr)
    build.apply_rules(game, terms)
    provenance["fold_readers"] = here
    moved = collections.Counter(f"{a}->{tier(t)}" for (a, _), t in zip(before, terms) if a != tier(t))
    moved.update(f"fold{'+' if t['fold'] else '-'}" for (_, f), t in zip(before, terms) if f != t["fold"])
    unprobed = [f"{t['ja']}={t['zh']}" for t in build.probe_pool(terms)
                if (t["ja"], t["zh"]) not in pool and not any(x.startswith("famous:") for x in t["tags"])]
    if unprobed:
        raise SystemExit("error: names newly to probe need the famous probe and a full build:\n"
                         + "\n".join(unprobed))
    return data, moved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=sorted(build.GAMES))
    ap.add_argument("glossary", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data, moved = retag(build.GAMES[args.game], json.loads(args.glossary.read_text(encoding="utf-8")))
    body = build.encode(data)
    glossary = build.gs.parse(body)
    args.out.write_bytes(body)
    print(f"moved {dict(sorted(moved.items()))}; {args.out}: {len(body)} bytes, sha256 {glossary.sha256}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
