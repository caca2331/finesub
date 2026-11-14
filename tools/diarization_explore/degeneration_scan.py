"""Degeneration signals of finished three-pass tasks, read from their artifacts (no model call).

For each task directory (the one holding `llm/` and `input/`) under the given roots: per pass 2 window the
signals `finesub.llm.degeneration.pass2_signals` records (rebuilt from the window plan, the judge's sidecar and
the ledger, so tasks run before the record existed count too), per pass 1 window the over-long joins the script
refused and its act counts, and for every call its thinking and visible output tokens. A pass 2 window split at
run time is listed, not read: its halves are not in the plan (a thinking of 0 is a signal
only where the route thinks and the usage is not under-reported: read the visible tokens next to it).

    python tools/diarization_explore/degeneration_scan.py tmp/spk-merge/accept [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from finesub.llm.chunking import load_segments_from_stable_json, rebuild_windows_from_plan
from finesub.llm.correction_translation import _on_product_lines
from finesub.llm.degeneration import pass2_signals
from finesub.llm.output_protocol import validate_correction_window_output
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.stages import three_pass

LONG = ("seconds", "rows")


def tasks(roots):
    for root in roots:
        for mode in Path(root).rglob("llm/correction-mode.json"):
            if json.loads(mode.read_text(encoding="utf-8")).get("mode") == three_pass.THREE_PASS:
                yield mode.parent.parent


def artifacts(task):
    path = task / "llm" / "task-artifacts.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scan(task):
    stable = next((task / "input").glob("*-stable.json"))
    records = artifacts(task)
    shape = next(r["payload"]["shape"] for r in records if r["kind"] == "correction_mode")
    plan = json.loads((task / "llm" / "correction-window-plan.json").read_text(encoding="utf-8"))
    windows = rebuild_windows_from_plan(plan["plan"], load_segments_from_stable_json(stable))
    windows = _product(windows, stable)
    pass2, grouping, _record = three_pass.pass2_plan(windows, stable, shape, warn=lambda _m: None)
    by_chunk = {window.chunk_id: window for window in pass2}
    out = {"task": str(task), "pass2": [], "split": [], "pass1": [], "calls": []}
    for line in (task / "llm" / "correction-windows.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "split_into" in record:
            out["split"].append({"chunk": record["chunk_id"], "into": record["split_into"]})
        window = by_chunk.get(record["chunk_id"])
        if window is None or not window.grouped or "content" not in record:
            continue
        variant = resolve_variant(record["variant"])
        rows = validate_correction_window_output(record["content"], window, variant=variant).segments
        out["pass2"].append({"chunk": record["chunk_id"],
                             **pass2_signals(window, rows, record["content"]).to_metadata()})
    out["pass1"] = _pass1(stable, grouping)
    calls = {}  # by call: a resumed task logs a replayed window's response again
    for r in records:
        payload = r.get("payload") or {}
        usage = payload.get("usage")
        if r["kind"].endswith("_response") and isinstance(usage, dict):
            call = payload.get("call") or f"correction-{payload.get('chunk_id')}-attempt{payload.get('attempt')}"
            calls[call] = {"call": call, "thinking": usage.get("thinking_tokens", 0),
                           "output": usage.get("output_tokens", 0)}
    out["calls"] = list(calls.values())
    return out


def _product(windows, stable):
    from dataclasses import dataclass

    @dataclass
    class Plan:
        windows: list

    return _on_product_lines(Plan(windows), stable).windows


def _pass1(stable, grouping):
    sidecar = json.loads(next(stable.parent.glob("*-speaker-judge.json")).read_text(encoding="utf-8"))
    order = [str(row.get("id", index + 1)) for index, row in
             enumerate(json.loads(stable.read_text(encoding="utf-8"))["segments"])]
    place = {row: index for index, row in enumerate(order)}
    long = [key for key, why in grouping.refused if why in LONG]
    out = []
    for window in sidecar["windows"]:
        first, last = place.get(window["first"]), place.get(window["last"])
        acts = Counter(item["act"] for item in (window.get("acts") or {}).values())
        refused = sum(1 for key in long if first is not None and last is not None
                      and first <= place.get(key.rstrip("abcdefghijklmnopqrstuvwxyz"), -1) <= last)
        out.append({"window": window["id"], "rows": (last - first + 1) if None not in (first, last) else None,
                    "join": acts["join"], "may_join": acts["may_join"], "drop": acts["drop"],
                    "long_refused": refused})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--json")
    args = parser.parse_args()
    results = []
    for task in sorted(set(tasks(args.roots))):
        try:
            results.append(scan(task))
        except Exception as error:  # noqa: BLE001 - one unreadable task does not stop the scan
            print(f"{task}: skipped ({type(error).__name__}: {error})")
            continue
        result = results[-1]
        print(result["task"])
        for row in result["pass2"]:
            print(f"  pass2 {row['chunk']}: 可并 {row['may_merged']}/{row['may_seams']}, notes {row['notes']}"
                  f" {row['signals'] or ''}")
        for row in result["split"]:
            print(f"  pass2 {row['chunk']}: split into {', '.join(row['into'])}, not read")
        for row in result["pass1"]:
            print(f"  pass1 {row['window']}: rows {row['rows']}, join {row['join']} (long refused "
                  f"{row['long_refused']}), may_join {row['may_join']}, drop {row['drop']}")
        for row in result["calls"]:
            print(f"  call {row['call']}: thinking {row['thinking']}, output {row['output']}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
