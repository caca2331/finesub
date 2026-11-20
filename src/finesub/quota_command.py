"""`finesub quota`: the keys set aside for their quota, and putting them back early.

    finesub quota                          list them
    finesub quota thaw <name>... | --all   put them back

Lists the keys frozen in `key_freeze` -- ElevenLabs keys until their credits refresh, Gemini keys spent for the
Pacific day on one model (`<provider tier>/<model>/<key>`), agent subscriptions (`agent/<pool>`, 2 h), search keys
(`search/<provider>/<key>`, 24 h) -- and today's account of the 35t and MOSS Pro keys (`speech/joint/quota.py`:
which are spent, and the audio each has taken). `thaw` clears a freeze or a spent flag; a name is a row as listed or
a prefix of rows (`elevenlabs`, `GEMINI_FREE`, `agent`, `search/exa`, `moss-pro/<model>`). The hours MOSS Pro's
account has counted stay: it still holds a key to its free hours until Beijing midnight. A key that really is
spent is set aside again at its next balance check or refusal.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

from . import key_freeze
from .speech.joint import quota


def _rows(now: datetime | None = None) -> tuple[list[key_freeze.Frozen], list[tuple[str, dict]]]:
    return key_freeze.listing(now=now), quota.today_rows(now=now)


def render(frozen: list[key_freeze.Frozen], account: list[tuple[str, dict]]) -> str:
    lines = []
    if frozen:
        lines.append("Frozen (`finesub quota thaw <row>` puts one back early):")
        lines += [f"  {f.row}  until {f.until:%Y-%m-%d %H:%M} UTC  ({f.reason})" for f in frozen]
    if account:
        lines.append("Today's account (35t: Pacific day, MOSS Pro: Beijing day):")
        lines += [f"  {row}  {u['calls']} calls, {u['seconds'] / 3600:.2f} h of audio" + (", spent" if u["exhausted"] else "")
                  for row, u in account]
    return "\n".join(lines) if lines else "No key is frozen or has an account today."


def _matches(name: str, row: str) -> bool:
    return row == name or row.startswith(name.rstrip("/") + "/")


def thaw(names: list[str], *, every: bool) -> tuple[list[str], list[str]]:
    """(rows put back, names that matched nothing)."""
    frozen, account = _rows()
    candidates = [f.row for f in frozen] + [row for row, u in account if u["exhausted"]]
    chosen = candidates if every else [r for r in candidates if any(_matches(n, r) for n in names)]
    unknown = [] if every else [n for n in names if not any(_matches(n, r) for r in candidates)]
    back = key_freeze.thaw(chosen) + quota.clear_spent(chosen)
    return back, unknown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finesub quota", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command")
    put_back = sub.add_parser("thaw", help="put frozen or spent keys back")
    put_back.add_argument("names", nargs="*", help="rows as `finesub quota` lists them, or a prefix (elevenlabs, GEMINI_FREE, agent, search, moss-pro)")
    put_back.add_argument("--all", action="store_true", help="every frozen or spent key")
    args = parser.parse_args(argv)
    if args.command is None:
        print(render(*_rows()))
        return 0
    if not args.names and not args.all:
        parser.error("thaw needs a name or --all")
    back, unknown = thaw(args.names, every=args.all)
    for row in back:
        print(f"Thawed {row}")
    if unknown:
        print(f"Nothing frozen or spent matches: {', '.join(unknown)} (see `finesub quota`)", file=sys.stderr)
        return 1
    if not back:
        print("Nothing was frozen or spent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
