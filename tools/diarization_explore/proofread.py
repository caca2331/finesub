"""Third pass, proofreading (docs/reports/2026-10-17-three-pass.md 「三个 pass 的分工」): reads the text pass's finished
lines and changes only clear errors, never the line split. An observation object for now, not part of the comparison.

    python tools/diarization_explore/proofread.py prompt <key>...   # render the prompt next to the run (no call)
    python tools/diarization_explore/proofread.py run <key>...      # call it; proofread.json holds the changes and usage

Input: two_pass's run for TRIAL_LABEL (tmp/spk-merge/two-pass-<TRIAL_LABEL>/<key>/): pass2/<key>-annotated.csv and
grouped/<key>-stable.json (the ASR text each pass-2 position came from).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import merge_trial as mt  # noqa: E402
import two_pass as tp  # noqa: E402

#: the trial windows run with no user style; production passes the task's style block here
STYLE = "（未指定，按一般的中文字幕）"


def bare(text):
    """The characters a viewer reads: punctuation, symbols and spaces left out."""
    return "".join(c for c in text if not unicodedata.category(c).startswith(("P", "Z", "S")))


def shown(srt):
    """Seconds each cue of an SRT stays on screen, in order: after the timeline's padding, which is what a viewer reads by."""
    stamp = lambda s: (lambda h, m, rest: int(h) * 3600 + int(m) * 60 + float(rest.replace(",", ".")))(*s.split(":"))
    return [stamp(b) - stamp(a) for a, b in re.findall(r"^(\S+) --> (\S+)$", srt.read_text(encoding="utf-8"), re.M)]


def lines(key):
    """[(position, seconds shown, chars per second, ASR text or `=`, corrected, translation, doubt, note)] of the text
    pass's sub lines: what proofreading needs, and nothing of the timeline or the speakers. Reading speed is production's
    weighted char count (the annotated `char_count`) over the cue's display time in pass 2's SRT."""
    folder = tp.ROOT / key
    grouped = json.loads((folder / "grouped" / f"{key}-stable.json").read_text(encoding="utf-8"))["segments"]
    display = shown(folder / "pass2" / f"{key}.srt")
    out = []
    for line in (folder / "pass2" / f"{key}-annotated.csv").read_text(encoding="utf-8").splitlines():
        cells = line.split("|")
        if not line or line.startswith("#") or cells[0] != "sub":
            continue
        positions = [int(p) for p in cells[1].split(",")]
        asr = "".join(grouped[p - 1]["text"].strip() for p in positions)
        seconds = display[len(out)]
        speed = f"{float(cells[7]) / seconds:.1f}" if seconds > 0 else ""
        doubt = cells[6] if cells[6] in ("low", "median") else ""
        out.append((len(out) + 1, f"{seconds:.1f}", speed, "=" if bare(asr) == bare(cells[4]) else asr, cells[4], cells[5], doubt,
                    cells[9] if len(cells) > 9 else ""))
    if len(out) != len(display):
        raise ValueError(f"{key}: {len(out)} sub lines but {len(display)} cues")
    return out


def prompt(key):
    ident = key.rsplit("-c", 1)[0]
    context = json.loads((mt.MAIN / "tmp/diarization-retrack/accept" / ident / "llm" / f"{ident}-research-context.json")
                         .read_text(encoding="utf-8"))["context_pack"]["general_context"]
    rows = lines(key)
    listed = lambda items, render: "\n".join(f"- {render(i)}" for i in items) or "（无）"
    speakers = listed(context.get("speakers", []), lambda s: f"{s['name']}（{s.get('role', '')}）：{s.get('note', '')}")
    terms = "\n".join(filter(None, [
        listed(context.get("entities", []), lambda e: f"{e['name']} → {e.get('zh', '')}（{e.get('type', '')}）：{e.get('note', '')}"),
        listed(context.get("terminology", []), lambda t: f"{t['source']} → {t['zh']}：{t.get('rule', '')}")]))
    risks = listed(context.get("asr_risks", []), lambda r: f"{r['heard_as']} → {r['should_be']}：{r.get('reason', '')}")
    system = (HERE.parent / "proofread_system.md").read_text(encoding="utf-8")
    user = Template((HERE.parent / "proofread_user.md").read_text(encoding="utf-8")).substitute(
        title=ident, summary=context.get("global_summary", ""), speakers=speakers, terms=terms, risks=risks,
        notes=listed(context.get("streamer_notes", []), str), uncertain=listed(context.get("uncertainties", []), str),
        style=STYLE, rows="\n".join("|".join(str(c) for c in row) for row in rows), count=len(rows))
    return system, user


def parse(text, rows):
    """[{position, kind, corrected, translation, why}] from the <proofread> block; refuses a reply that is not one."""
    match = re.search(r"<proofread>(.*?)</proofread>", text or "", re.S)
    if match is None:
        raise ValueError("no closed <proofread> block")
    out, seen = [], set()
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        cells = line.split("|")
        if len(cells) != 5 or not cells[0].strip().isdigit() or not re.fullmatch(r"[1-6]", cells[1].strip()):
            raise ValueError(f"bad line {line!r}")
        position = int(cells[0])
        if not 1 <= position <= len(rows) or position in seen:
            raise ValueError(f"bad position {line!r}")
        seen.add(position)
        out.append({"position": position, "kind": int(cells[1]), "corrected": cells[2].strip(),
                    "translation": cells[3].strip(), "why": cells[4].strip()})
    return out


def run(keys):
    from finesub.llm.client import RoleClient, extract_token_distribution
    from finesub.llm.routing.config import LLMRole
    from finesub.llm.routing.model_routes import install_runtime_preferred, parse_llm_model_args

    install_runtime_preferred(parse_llm_model_args([mt.MODEL]))
    client = RoleClient()
    for key in keys:
        target = tp.ROOT / key / "proofread.json"
        if target.exists():
            print(f"{key}: proofread before")
            continue
        system, user = prompt(key)
        rows = lines(key)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        calls = []
        for _ in range(2):
            result = client.complete(LLMRole.AUDIO_MULTIMODAL, messages, task_group="correction-text")
            calls.append({"model": result.model, "text": result.content,
                          "usage": extract_token_distribution(result.raw_response)})
            try:
                changes = parse(result.content, rows)
            except ValueError as error:
                calls[-1]["error"] = str(error)
                continue
            target.write_text(json.dumps({"changes": changes, "calls": calls}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            print(f"{key}: {len(changes)} lines changed", flush=True)
            break
        else:
            print(f"{key}: no answer: {calls[-1].get('error')}")


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "prompt":
        for key in rest:
            system, user = prompt(key)
            target = tp.ROOT / key / "proofread-prompt.md"
            target.write_text(f"{system}\n\n=====\n\n{user}", encoding="utf-8")
            print(target)
    else:
        {"run": run}[mode](rest)
