"""Replay the production slimming on executed windows (maintainers only).

    python tools/kb_phonetic_eval/slim_size.py <root> [--arm A1] [--clips id,id]

A dry-run renders no ``entry_details`` -- the entries reach a window only in the executed loop -- so
this replays it offline: from each clip's executed correction exchanges in <root>/<arm>/ it takes the
window's rows and its ``<entry_details>``, splits the block back into entries, and hands them to the
serial loop's own ``_carried_view`` with ``FINESUB_KB_SLIM=on``, against <root>/kb at its current
revision. Prints, per window, whether the gate fired, the whole block's tokens and the slimmed block +
hint tokens, counted by the run's default token counter. Needs the ``kb-phonetic`` extra.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.llm import kb_phonetic_hints as kh  # noqa: E402
from finesub.llm.knowledge.node.repo import KnowledgeRepo  # noqa: E402
from finesub.llm.stages.correction.serial import _carried_view  # noqa: E402
from finesub.llm.token_budget import default_token_counter  # noqa: E402

ROW = re.compile(r"^(\d+)\|[\d.]+\|[\d.]+\|[^|\n]*\|(.*)$", re.M)


def split_entries(details: str) -> dict[str, str]:
    """The block's entries, keyed by H1; each body starts at its ``# `` line."""

    bodies: dict[str, list[str]] = {}
    current = None
    for line in details.splitlines():
        if line.startswith("# "):
            current = line[2:].strip()
            bodies[current] = []
        if current is not None:
            bodies[current].append(line)
    return {key: "\n".join(lines).strip() + "\n" for key, lines in bodies.items()}


def window_of(exchange: str) -> tuple[SimpleNamespace, str]:
    user = exchange.split("## 请求（user）", 1)[1].split("## 模型响应", 1)[0]
    details = re.search(r"<entry_details>\n(.*?)\n</entry_details>", user, re.S).group(1)
    rows = re.search(r"<asr_result>\nlocal_id\|.*?\n(.*?)</asr_result>", user.split("<entry_details>")[1], re.S)
    texts = [m.group(2) for m in ROW.finditer(rows.group(1))]
    return SimpleNamespace(segments=[SimpleNamespace(text=t) for t in texts]), details


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--arm", default="A1")
    ap.add_argument("--clips", default="")
    args = ap.parse_args()
    root = args.root.resolve()
    os.environ[kh.ENV] = "on"
    if not kh.readers_available():
        raise SystemExit("needs the kb-phonetic extra")
    kb = root / "kb"
    run = SimpleNamespace(
        token_counter=default_token_counter(), request_channel=False, knowledge_root=kb,
        knowledge_version=f"rev:{KnowledgeRepo.open(kb).store.current_rev()}",
    )
    clips = [c for c in args.clips.split(",") if c] or sorted(p.name for p in (root / args.arm).iterdir() if p.is_dir())
    print("clip|window|gate|whole|entries+hints (less)|hinted")
    for clip in clips:
        for exchange in sorted((root / args.arm / clip / "artifacts" / "exchanges").glob("*correction-*-attempt0.md")):
            if "-query-" in exchange.name or "-request-" in exchange.name:  # request turns carry only the transfer
                continue
            window, details = window_of(exchange.read_text(encoding="utf-8"))
            slim = _carried_view(run, window, split_entries(details)).slim
            whole = slim["whole_tokens"]
            after = slim.get("entry_tokens", whole) + slim.get("hint_tokens", 0)
            print(f"{clip}|{exchange.name[:6]}|{'on' if slim['gate'] else 'off'}|{whole}|{after} ({1 - after / whole:.0%})"
                  f"|{','.join(slim.get('hinted', ())) or slim.get('skipped', '')}")


if __name__ == "__main__":
    main()
