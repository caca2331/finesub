"""Drive one agy session through the blind ASR listening batches.

Headless agy auto-denies any tool that would need a permission prompt, and the
two ways past that are not equivalent. `--dangerously-skip-permissions`
approves *every* tool, shell included, for a job that only ever needs to read
files this script produced. So this reuses what the production driver already
does instead: a controlled project whose `.agents/hooks.json` runs a
deny-by-default `PreToolUse` guard, entitling exactly one tool -- `view_file`
on absolute paths inside the project root (`AGY_GUARD_SCRIPT`).

Nothing here writes the user's global agy settings; the project lives in the
audit directory and agy registers it itself (docs/llm_local_agent_agy.md §4,
and `llm_agent_tool_protocol.md`'s standing rule about global settings).

    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.run_asr_ab_audit \\
        --dir tmp/bench/ja-ab/audit --model gemini-3.8-flash --effort medium
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finesub.llm.agent.local_agent import (  # noqa: E402
    AGY_AGENT_DOCUMENT,
    AGY_AGENT_NAME,
    AGY_GUARD_SCRIPT,
    AGY_HOOKS_DOCUMENT,
    _AGY_PROJECT_ID_RE,
)

#: One conversation for the whole audit, so the judge keeps its calibration
#: across batches. Safe only because 甲/乙 is redrawn per item -- see the
#: exporter; a per-batch arm identity would let drift become bias.
CONVERSATION_STATE = "audit-session.json"


def write_project(root: Path) -> None:
    agents = root / ".agents"
    documents = {
        agents / "hooks.json": json.dumps(AGY_HOOKS_DOCUMENT, ensure_ascii=False, indent=2) + "\n",
        agents / "scripts" / "guard_view_file.py": AGY_GUARD_SCRIPT,
        agents / "agents" / AGY_AGENT_NAME / "agent.md": AGY_AGENT_DOCUMENT,
    }
    for path, content in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")


def create_project(root: Path, agy: str) -> str:
    """Register the project with agy and return its id.

    `/hooks` is a zero-token query; running it with `--new-project` is how the
    production driver both creates the project and confirms the hook actually
    loaded, which is the point -- a project whose hook did not load has no
    boundary at all.
    """

    result = subprocess.run(
        [agy, "--print", "/hooks", "--output-format", "text", "--new-project"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    blob = f"{result.stdout}\n{result.stderr}"
    if "finesub-view-boundary" not in blob:
        # Not a warning: without a project the hook list comes back EMPTY
        # (verified on agy 1.1.24), which is no boundary at all.
        raise SystemExit(
            "the deny-by-default hook did not load; refusing to run without a "
            f"boundary:\n{blob[-2000:]}"
        )
    match = _AGY_PROJECT_ID_RE.search(blob)
    if match:
        return match.group(1)
    return newest_project_for(root)


def newest_project_for(root: Path) -> str:
    """Read the id back from the record agy just wrote.

    agy 1.1.24 does not announce a new project id on stdout. Matching on the
    resource folder rather than on mtime alone keeps a concurrent, unrelated
    `--new-project` from being adopted as ours.
    """

    from finesub_bootstrap.agy_records import agy_project_records_dir

    wanted = root.resolve().as_posix().lower().rstrip("/")
    best: tuple[float, str] | None = None
    for record_path in agy_project_records_dir().glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        resources = (record.get("projectResources") or {}).get("resources") or []
        for resource in resources:
            uri = str(resource.get("folderUri") or "")
            if uri.removeprefix("file://").lower().rstrip("/") == wanted:
                stamp = record_path.stat().st_mtime
                if best is None or stamp > best[0]:
                    best = (stamp, str(record.get("id") or record_path.stem))
    if best is None:
        raise SystemExit(f"agy registered no project for {root}")
    return best[1]


def run_batch(
    *,
    agy: str,
    root: Path,
    project_id: str,
    model: str,
    effort: str,
    prompt: str,
    conversation: str | None,
    timeout: int,
) -> tuple[str, str]:
    argv = [agy, "--model", model, "--project", project_id, "--agent", AGY_AGENT_NAME]
    if model.lower().startswith("gemini-"):
        argv.extend(("--effort", effort))
    if conversation:
        argv.extend(("--conversation", conversation))
    argv.extend(("--output-format", "stream-json", f"--print={prompt}"))
    result = subprocess.run(
        argv,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    text_parts: list[str] = []
    session = conversation or ""
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # agy 1.1.24 shape: `{"event": "init"|"step_update"|"result", ...}`,
        # with the assistant text at `result.response` and the id on both the
        # init event and the terminal one.
        payload = event.get("result") if isinstance(event.get("result"), dict) else event
        for holder in (event, payload, event.get("init") or {}):
            if not isinstance(holder, dict):
                continue
            value = holder.get("conversation_id") or holder.get("conversationId")
            if isinstance(value, str) and value:
                session = value
        if event.get("event") == "result" and isinstance(payload, dict):
            status = str(payload.get("status") or "")
            if status and status != "SUCCESS":
                text_parts.append(f"[agy status {status}]")
            response = payload.get("response")
            if isinstance(response, str) and response.strip():
                text_parts.append(response)
    if not text_parts:
        # Keep the raw stream when the shape is not what we expect: the error
        # path matters more than the happy one here.
        text_parts.append(result.stdout)
    if not parse_rows("\n".join(text_parts)):
        # A failed turn that keeps only "[agy status ERROR]" is unactionable,
        # and this failure IS transient (the same batch answered on a retry),
        # so the raw stream has to survive or the next one is undiagnosable too.
        text_parts.append("\n--- raw ---\n" + result.stdout[-4000:])
        if result.stderr.strip():
            text_parts.append("\n--- stderr ---\n" + result.stderr[-2000:])
    return "\n".join(text_parts), session


ROW = re.compile(r"^\s*(S\d{3})\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.*?)\s*$")


def parse_rows(reply: str) -> list[dict[str, str]]:
    rows = []
    for line in reply.splitlines():
        match = ROW.match(line)
        if match:
            rows.append(
                {
                    "id": match.group(1),
                    "verdict": match.group(2),
                    "category": match.group(3),
                    "reason": match.group(4),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="tmp/bench/ja-ab/audit")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--agy", default="agy")
    parser.add_argument("--only", default="", help="run one batch, e.g. 01")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pause", type=int, default=20)
    args = parser.parse_args(argv)

    root = Path(args.dir).resolve()
    batches = sorted(root.glob("batch-*.md"))
    if args.only:
        batches = [p for p in batches if p.stem.endswith(args.only)]
    if not batches:
        parser.error(f"no batch-*.md under {root}")

    write_project(root)
    state_path = root / CONVERSATION_STATE
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {}
    )
    project_id = state.get("project_id") or create_project(root, args.agy)
    state["project_id"] = project_id
    conversation = state.get("conversation") or ""
    answers = state.setdefault("answers", {})
    print(f"project {project_id}  ({len(batches)} batches)")

    for path in batches:
        if answers.get(path.name, {}).get("rows"):
            print(f"{path.name}: already answered, skipping")
            continue
        prompt = (
            f"读取并执行 {path.name} 里的全部指示。"
            f"音频路径相对本目录（{root}）。"
        )
        began = time.perf_counter()
        rows: list[dict[str, str]] = []
        reply = ""
        for attempt in range(1, args.retries + 1):
            reply, conversation = run_batch(
                agy=args.agy,
                root=root,
                project_id=project_id,
                model=args.model,
                effort=args.effort,
                prompt=prompt,
                conversation=conversation or None,
                timeout=args.timeout,
            )
            rows = parse_rows(reply)
            if rows:
                break
            # Empirically transient: the same batch answered on the next try.
            # Back off rather than hammering one conversation.
            wait = args.pause * attempt
            print(f"{path.name}: attempt {attempt} produced no rows; retrying in {wait}s")
            time.sleep(wait)
        answers[path.name] = {"reply": reply, "rows": rows}
        state["conversation"] = conversation
        state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(
            f"{path.name}: {len(rows)} rows in {time.perf_counter() - began:.0f}s"
            + ("" if rows else "  ⚠ still no rows; see the raw stream in the state file")
        )
        # A pause between batches for the same reason as the backoff.
        time.sleep(args.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
