"""Step 1 of the README: the closed-book "famous" probe of the resident-list pool (``build.py --famous``).

    python -I tools/game_glossary/famous.py <game> --pool <pool.txt> --famous <famous.json> --cache <dir> \
        [--model gemini-3.5-flash-lite] [--passes 3] [--batch 120] [--max-calls 40]

``<pool.txt>`` is what ``build.py`` prints when names of the pool are missing from ``--famous`` (one ``日文=中文`` per
line; run the build with an empty ``{}`` file to list the whole pool). Each pass shuffles the names (a fixed seed per
pass), numbers them and asks a model with no tools and no files for the official simplified Chinese of each one,
``?`` when unsure, no transliteration of its own; a name answered with its official ``zh`` (spaces, interpuncts and
「」『』“” removed, then character for character) scores one. The scores are merged into ``<famous.json>``
(``{中文名: right answers}``); ``build.py`` counts ``FAMOUS_PASSES`` (3) as famous.

The model is called through the Gemini API with the key in ``GEMINI_API_KEY`` (never printed). Every request and
answer is cached under ``--cache`` by its text, so a rerun or a follow-up probe of a few added names spends calls on
the new requests only; ``<cache>/calls.json`` counts the calls made per day, and the run stops before going over
``--max-calls`` for the day (the free tier allows a few hundred Flash-Lite calls a day).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

GAME_NAMES = {"genshin": "原神", "starrail": "崩坏：星穹铁道", "zzz": "绝区零"}
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TASK = """下面是《{game}》里的 {n} 个日文专名（角色、组织、阵营、地名、属性、机制等），每行「编号|日文名」。
请写出每个名字在《{game}》简体中文版里的官方译名。不要搜索，不要查任何文件，只凭记忆作答。
不认识或不确定官方译名时，写 ?（不要猜，不要自己音译）。

只输出 {n} 行，每行「编号|日文名|官方简体中文名或 ?」，不要别的文字。

{body}
"""


def _norm(text: str) -> str:
    return re.sub(r"[·・･•\s「」『』“”‘’]", "", text or "")


class Budget:
    def __init__(self, cache: Path, limit: int):
        self.path, self.limit, self.today = cache / "calls.json", limit, date.today().isoformat()
        self.counts = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}

    @property
    def used(self) -> int:
        return int(self.counts.get(self.today, 0))

    def spend(self) -> None:
        if self.used >= self.limit:
            raise SystemExit(f"stopping: {self.used} calls today reach --max-calls {self.limit}; rerun tomorrow"
                             " (the answers so far are cached)")
        self.counts[self.today] = self.used + 1
        self.path.write_text(json.dumps(self.counts, indent=1), encoding="utf-8")


def ask(prompt: str, model: str, cache: Path, budget: Budget, n: int) -> str:
    """The answer of pass ``n`` to ``prompt``: a pass is its own request even when the shuffle gives the same text."""

    key = hashlib.sha256(f"{model}\n{n}\n{prompt}".encode("utf-8")).hexdigest()[:24]
    hit = cache / f"{key}.json"
    if hit.is_file():
        return json.loads(hit.read_text(encoding="utf-8"))["answer"]
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set")
    body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 1.0}}).encode("utf-8")
    for attempt in range(4):
        budget.spend()
        request = urllib.request.Request(API.format(model=model), data=body, method="POST",
                                         headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (the fixed Gemini API)
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code in (429, 500, 503) and attempt < 3:
                print(f"HTTP {exc.code}, retrying: {detail}", file=sys.stderr)
                time.sleep(20 * (attempt + 1))
                continue
            raise SystemExit(f"Gemini API error {exc.code}: {detail}") from None
    answer = "".join(p.get("text", "") for c in data.get("candidates", [])[:1]
                     for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
    hit.write_text(json.dumps({"model": model, "pass": n, "prompt": prompt, "answer": answer}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return answer


def parse(answer: str, n: int) -> dict[int, str]:
    got = {}
    for line in answer.splitlines():
        parts = [p.strip() for p in line.strip().strip("`").split("|")]
        if len(parts) >= 2 and parts[0].isdigit() and 0 < int(parts[0]) <= n:
            got[int(parts[0])] = parts[-1]
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=sorted(GAME_NAMES))
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--famous", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--batch", type=int, default=120)
    ap.add_argument("--max-calls", type=int, default=40, help="calls allowed today across runs")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.cache, args.max_calls)
    pool = []
    for line in args.pool.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("error:"):
            ja, zh = line.strip().split("=", 1)
            pool.append((ja.strip(), zh.strip()))
    pool = list(dict.fromkeys(pool))
    right = {zh: 0 for _, zh in pool}
    for n in range(args.passes):
        order = list(range(len(pool)))
        random.Random(1000 + n).shuffle(order)
        for start in range(0, len(order), args.batch):
            chunk = order[start:start + args.batch]
            body = "\n".join(f"{i + 1}|{pool[k][0]}" for i, k in enumerate(chunk))
            got = parse(ask(TASK.format(game=GAME_NAMES[args.game], n=len(chunk), body=body), args.model,
                            args.cache, budget, n), len(chunk))
            for i, k in enumerate(chunk):
                if _norm(got.get(i + 1, "")) == _norm(pool[k][1]):
                    right[pool[k][1]] += 1
    famous = json.loads(args.famous.read_text(encoding="utf-8")) if args.famous.is_file() else {}
    famous.update(right)
    args.famous.parent.mkdir(parents=True, exist_ok=True)
    args.famous.write_text(json.dumps(famous, ensure_ascii=False, indent=0, sort_keys=True), encoding="utf-8")
    passed = sum(v >= args.passes for v in right.values())
    print(f"{len(pool)} names probed, {passed} right in all {args.passes} passes -> {args.famous}; "
          f"{budget.used} Gemini calls today", file=sys.stderr)


if __name__ == "__main__":
    main()
