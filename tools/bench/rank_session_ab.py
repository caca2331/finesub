"""Blind per-window ranking of every run of two session-mode arms (docs/llm_local_agent_experiments.md §3.5).

Pairwise judging needs a pair per comparison; with three runs per arm that is
fifteen pairs per material. Ranking all six translations of one window in one
call answers both questions at once: the arms' mean rank (does one arm do
better?) and the spread of rank inside an arm (the run-to-run noise it has to
beat). Labels A-F are shuffled per window and the key goes to a sibling
``<out>-keys`` directory, away from the pack the judge reads.

    python -m tools.bench.rank_session_ab pack out/session-ab/x --arms agy38f-per-window-capsule agy38f-resume \\
        --stable <clip-stable.json> --refined <refined.ass|srt> --clip-end S --out <dir>
    python -m tools.bench.rank_session_ab collect <dir>   # reads <dir>/verdict-wNN.json

Cross-window consistency is not judged here (a judge sees one window); that is
the pre-registered probe list and the entity count.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import string
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from finesub.subtitles.model import parse_srt

from tools.bench.judge_session_ab import _lines, _mmss, _translated
from tools.bench.score_session_ab import _windows, refined_srt

INSTRUCTIONS = """\
# 字幕译文盲评：同一窗口的 {n} 份译文排序

下面是同一段日语节目（{speakers}）同一个时间窗口的 {n} 份机器中文译文，标为 {labels}。它们来自同一个
系统的不同运行，你不知道哪份是怎么产生的，也不该去猜。另外给出：

- **原文**：日语语音识别结果（可能有识别错误，不是金标；方括号里是时间）；
- **精修参照**：人工字幕组精修过的中文。它**不是金标**：断句、填充词取舍、意译与口语化都是译者自由，
  差异不等于译错；把它当作「懂行的人会怎么理解这段话」的参考。

## 判断标准（按重要性）

1. **意思对不对**：误译、漏译（原文有实质内容而译文没有）、凭空添加；
2. **专名与称呼**：人名、作品名、角色名、对人的称呼译法是否正确，本窗内是否前后一致；
3. **通顺**：中文是否自然、可读。

断句、行数、标点、语气词取舍上的差别**不算**优劣，除非它让意思错了。

## 输出

把 {n} 份从好到差排成若干组；没有实质差别的放进同一组（并列），不要硬分先后。只输出一个 JSON 对象，
不要任何其它文字：

```json
{{
  "ranking": [["最好的标签", "与它并列的标签"], ["次一档"], ["..."]],
  "errors": {{"A": [{{"time": "mm:ss", "kind": "误译|漏译|添加|专名|不通顺", "note": "一句话"}}], "B": []}},
  "reason": "两三句话说明排序的主要依据"
}}
```

`ranking` 必须恰好包含每个标签一次。`errors` 为每个标签列出你确认的问题（带时间，便于复核），没有就给空数组。
"""


def _runs(material: Path, arms: List[str]) -> List[Path]:
    runs = []
    for arm in arms:
        for run in sorted((material / arm).glob("run*")):
            meta = run / "run.json"
            if meta.is_file() and json.loads(meta.read_text(encoding="utf-8")).get("exit_code") == 0:
                runs.append(run)
    return runs


def _source(stable: Path) -> List[Any]:
    rows = []
    for s in json.loads(stable.read_text(encoding="utf-8"))["segments"]:
        # The diarization record ({labels, track, coverage}) names no person, so it is left out.
        rows.append(type("Seg", (), {"start": float(s["start"]), "text": s.get("text", "")})())
    return rows


def pack(material: Path, arms: List[str], stable: Path, refined: Path, clip_end: float | None, out: Path, seed: int) -> None:
    runs = _runs(material, arms)
    if len(runs) > len(string.ascii_uppercase):
        raise SystemExit("too many runs to label")
    plans = {run: _windows(run, stable) for run in runs}
    windows = plans[runs[0]]
    if not windows or any(plan != windows for plan in plans.values()):
        raise SystemExit("every run must share one window plan")
    source = _source(stable)
    speakers = "多人对谈" if any("speaker" in s for s in json.loads(stable.read_text(encoding="utf-8"))["segments"][:50]) else "单人"
    translated = {run: parse_srt(_translated(run).read_text(encoding="utf-8-sig")) for run in runs}
    with tempfile.TemporaryDirectory() as tmp:
        ref = parse_srt(refined_srt(refined, Path(tmp), clip_end).read_text(encoding="utf-8"))
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    keys = _keys_dir(out)
    labels = list(string.ascii_uppercase[: len(runs)])
    for index, (start, _end) in enumerate(windows, 1):
        end = windows[index][0] if index < len(windows) else float("inf")
        order = runs[:]
        rng.shuffle(order)
        parts = [INSTRUCTIONS.format(n=len(runs), labels="、".join(labels), speakers=speakers)]
        parts.append(f"\n## 窗口 {index}（{_mmss(start)} 起）\n")
        parts.append("### 原文\n\n" + ("\n".join(_lines(source, start, end)) or "（空）") + "\n")
        for label, run in zip(labels, order):
            parts.append(f"### {label}\n\n" + ("\n".join(_lines(translated[run], start, end)) or "（空）") + "\n")
        parts.append("### 精修参照\n\n" + ("\n".join(_lines(ref, start, end)) or "（空）") + "\n")
        (out / f"pack-w{index:02d}.md").write_text("\n".join(parts), encoding="utf-8")
        key = {label: f"{run.parent.name}/{run.name}" for label, run in zip(labels, order)}
        keys.mkdir(parents=True, exist_ok=True)
        (keys / f"key-w{index:02d}.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(windows)} windows, {len(runs)} runs -> {out}")


def _keys_dir(out: Path) -> Path:
    return out.with_name(out.name + "-keys")


def _verdict(path: Path) -> Dict[str, Any]:
    """The judge's JSON. Judges write unescaped ASCII quotes inside notes and
    trail extra text now and then; the fields scoring needs are recovered by
    pattern when the object as a whole will not parse."""

    text = path.read_text(encoding="utf-8")
    start = text.find("{")
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except json.JSONDecodeError:
        pass
    recovered: Dict[str, Any] = {}
    ranking = re.search(r'"ranking"\s*:\s*(\[\s*\[.*?\]\s*\])', text, re.S)
    if ranking:
        recovered["ranking"] = json.loads(ranking.group(1))
        recovered["errors"] = {
            label: [{}] * len(re.findall(r'"time"', body))
            for label, body in re.findall(r'"([A-Z])"\s*:\s*\[(.*?)\](?=\s*,\s*"[A-Z]"\s*:|\s*\})', text, re.S)
        }
    probes = re.findall(r'"id"\s*:\s*"(\w+)"\s*,\s*"verdicts"\s*:\s*(\{[^{}]*\})', text)
    if probes:
        recovered["probes"] = [{"id": pid, "verdicts": json.loads(body)} for pid, body in probes]
    if not recovered:
        raise SystemExit(f"{path}: no parsable verdict")
    return recovered


def _ranks(groups: List[List[str]]) -> Dict[str, float]:
    """Mid-rank for ties: a group occupying places 2-3 gives each member 2.5."""

    ranks, place = {}, 1
    for group in groups:
        mid = place + (len(group) - 1) / 2
        for label in group:
            if label in ranks:
                # A label ranked twice would be merged here and shift every
                # place after it, while the set check in `collect` still passes.
                raise SystemExit(f"ranking names {label!r} more than once")
            ranks[label] = mid
        place += len(group)
    return ranks


def collect(out: Path) -> Dict[str, Any]:
    per_run: Dict[str, List[float]] = {}
    errors: Dict[str, int] = {}
    windows = []
    for key_path in sorted(_keys_dir(out).glob("key-w*.json")):
        tag = key_path.stem.split("-")[-1]
        verdict_path = out / f"verdict-{tag}.json"
        if not verdict_path.is_file():
            continue
        key = json.loads(key_path.read_text(encoding="utf-8"))
        verdict = _verdict(verdict_path)
        ranks = _ranks(verdict["ranking"])
        if sorted(ranks) != sorted(key):
            raise SystemExit(f"{verdict_path}: ranking must name every label once")
        for label, run in key.items():
            per_run.setdefault(run, []).append(ranks[label])
            errors[run] = errors.get(run, 0) + len((verdict.get("errors") or {}).get(label) or [])
        windows.append(tag)
    arms: Dict[str, Dict[str, Any]] = {}
    for run, values in per_run.items():
        arm = run.split("/")[0]
        arms.setdefault(arm, {"runs": {}})["runs"][run] = {
            "mean_rank": round(statistics.mean(values), 2), "errors": errors[run], "ranks": values,
        }
    for arm, data in arms.items():
        means = [r["mean_rank"] for r in data["runs"].values()]
        data["mean_rank"] = round(statistics.mean(means), 2)
        data["run_spread"] = round(max(means) - min(means), 2)
    result = {"windows": windows, "arms": arms}
    if len(arms) == 2:
        a, b = (sorted(r["mean_rank"] for r in data["runs"].values()) for data in arms.values())
        # Mann-Whitney U on run-level mean ranks: how often an A run beats a B run.
        wins = sum(1.0 if x < y else 0.5 if x == y else 0.0 for x in a for y in b)
        result["u_first_arm_better"] = {"wins": wins, "of": len(a) * len(b), "first_arm": list(arms)[0]}
    print(json.dumps(result, ensure_ascii=False, indent=2))  # product output
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("material")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--stable", required=True)
    p.add_argument("--refined", required=True)
    p.add_argument("--clip-end", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261002)
    c = sub.add_parser("collect")
    c.add_argument("out")
    args = parser.parse_args(argv)
    if args.cmd == "pack":
        pack(Path(args.material), args.arms, Path(args.stable), Path(args.refined), args.clip_end, Path(args.out), args.seed)
    else:
        collect(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
