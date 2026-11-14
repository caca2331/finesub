"""How much each model call thought, read back from finished tasks' `task-artifacts.jsonl` (no model call).

Every response record that carries usage, per route (call kind x model x backend): how many calls, how many
reported no usage at all (input and output both 0: a usage gap, not a call that skipped thinking), how many
thought 0 with usage present, and the thinking of the rest (5th, 25th, 50th percentile). `--zero` lists the
zero-thinking calls with what corroborates them (duration, input, visible output against the route's median).
The reading and the rule it supports: `docs/llm_followups.md`「模型回答的退化」.

    python tools/thinking_scan.py tmp out data [--zero] [--model gemini-3.8-flash]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def backend_of(payload):
    for attempt in payload.get("execution_attempts") or []:
        if isinstance(attempt, dict):
            return attempt.get("driver") or attempt.get("backend") or ""
    for link in (payload.get("route_decision") or {}).get("effective_chain") or []:
        return link.get("backend", "")
    return ""


def calls(roots):
    """One row per call; a call logged again (a resumed task, a copied task directory) counts once."""

    seen = {}
    for root in roots:
        for path in Path(root).rglob("task-artifacts.jsonl"):
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = record.get("payload") or {}
                usage = payload.get("usage")
                if not isinstance(usage, dict) or "thinking_tokens" not in usage:
                    continue
                row = {
                    "task": str(path.parent), "kind": record["kind"].removesuffix("_response"),
                    "call": payload.get("call") or payload.get("session") or "",
                    "model": payload.get("model") or "", "backend": backend_of(payload),
                    "created": record.get("created_at") or "",
                    "thinking": int(usage.get("thinking_tokens") or 0),
                    "output": int(usage.get("output_tokens") or 0),
                    "input": int(usage.get("total_input_tokens") or usage.get("prompt_tokens") or 0),
                    "seconds": sum(int(a.get("duration_ms") or 0) for a in payload.get("execution_attempts") or []
                                   if isinstance(a, dict)) // 1000 or None,
                    "valid": payload.get("validation_ok"),
                }
                seen[(row["created"][:19], row["kind"], row["thinking"], row["output"], row["input"])] = row
    return list(seen.values())


def percentile(values, share):
    return values[int(share * (len(values) - 1))] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--zero", action="store_true", help="list the zero-thinking calls")
    parser.add_argument("--model", help="only this model")
    parser.add_argument("--min-calls", type=int, default=10)
    args = parser.parse_args()
    rows = [row for row in calls(args.roots) if not args.model or row["model"] == args.model]
    routes = defaultdict(list)
    for row in rows:
        routes[(row["kind"], row["model"], row["backend"])].append(row)
    print("route | calls | no usage | thinking 0 | thinking p5 / p25 / p50")
    for route, group in sorted(routes.items(), key=lambda item: -len(item[1])):
        if len(group) < args.min_calls:
            continue
        reported = [row for row in group if row["input"] or row["output"]]
        thought = sorted(row["thinking"] for row in reported if row["thinking"])
        zero = len(reported) - len(thought)
        spread = " / ".join(str(percentile(thought, share)) for share in (0.05, 0.25, 0.5)) if thought else "-"
        print(f"{' '.join(part for part in route if part)} | {len(group)} | {len(group) - len(reported)} | {zero} | {spread}")
        if args.zero and thought and zero:
            median = {key: statistics.median(row[key] for row in reported if row["thinking"] and row[key])
                      for key in ("seconds", "input", "output")
                      if any(row["thinking"] and row[key] for row in reported)}
            for row in sorted((row for row in reported if not row["thinking"]), key=lambda row: row["created"]):
                ratios = ", ".join(f"{key} {row[key]} ({row[key] / median[key]:.2f} of median)"
                                   for key in median if row[key])
                print(f"    {row['created'][:16]} {row['call']} valid={row['valid']} {ratios}  {row['task']}")


if __name__ == "__main__":
    main()
