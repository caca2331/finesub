"""Per agy run: the in-session submissions (payload with <translated>) and what the validator said back."""
import json, re, sys
from pathlib import Path
M = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground/tmp/spk-merge")
BRAIN = Path.home() / ".gemini/antigravity-cli/brain"
def session(run):
    # the correction's last session (a judge run's exchanges come first; an agy session ended without a submit is retried)
    ex = sorted((run / "llm" / "exchanges").glob("*-correction-*.md"))[-1].read_text(encoding="utf-8")
    return re.search(r"\| agy [\d.]+ \| ([0-9a-f-]{36}) \|", ex).group(1)
def submissions(sid):
    steps = [json.loads(l) for l in (BRAIN / sid / ".system_generated/logs/transcript_full.jsonl").open(encoding="utf-8", errors="replace")]
    out = []
    for n, s in enumerate(steps):
        for tc in s.get("tool_calls") or []:
            payload = json.dumps(tc, ensure_ascii=False)
            if "<translated>" in payload:
                args = tc.get("args", {}).get("Arguments", {})
                text = args.get("payload") if isinstance(args, dict) else None
                reply = next((x.get("content") for x in steps[n + 1:] if x.get("type") == "GENERIC"), "")
                out.append((text or "", reply or ""))
    return out
if __name__ == "__main__":
    for arm in sys.argv[1:]:
        for key in ["hina-17-c0002", "hina-24-c0002", "hina-31-c0005"]:
            run = M / arm / key / ("pass2" if arm.startswith("two-pass") else "")
            subs = submissions(session(run))
            print(arm, key, len(subs), [(r.count("换人"), r[:160].replace("\n", " ")) for _, r in subs])
