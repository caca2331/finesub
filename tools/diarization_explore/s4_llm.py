"""Plan 5h.4: can a multimodal model fill in the drowned speaker?

Same 3 AliMeeting meetings and the same 357 sampled regions as 5h.3, grouped
into <=120 s chunks. Each chunk also carries as many decoy spans (one speaker
only, well clear of any overlap), worded identically, so confabulation shows up
as additions where nothing was missing.

  --ask MODEL RUN   MODEL = lite (3.5 flash-lite, normal routing) | agy (agy's
                    Gemini 3.7 Flash, pinned via the agy preset's test target)
  --score MODEL     all runs of that model, against the pre-registered criteria
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np

import s4_headroom as ref
import s4_projection as proj

HERE = Path(__file__).resolve().parent
CHUNK_MAX, EDGE, DECOY_MIN, DECOY_CLEAR, DECOY_LEN = 120.0, 5.0, 1.0, 1.0, 3.0
BLOCK = re.compile(r"<overlap_fix>(.*?)</overlap_fix>", re.DOTALL)
ADD_SPURIOUS_MAX, DECOY_MAX, GAIN_MIN = 0.443, 0.10, 0.05


def source(session):
    if ref.AUDIO_CORPUS is None:  # our own stream (5o.9 E4): the production vocal track is the source
        return proj.ALI / session / f"{session}-vocal.ogg"
    return ref.DATA / "public" / ref.AUDIO_CORPUS / "audio" / f"{session}.wav"


def decoy_runs(active):
    single = active.sum(axis=1) == 1
    busy = active.sum(axis=1) >= 2
    clear = int(DECOY_CLEAR / proj.FRAME)
    near = np.convolve(busy.astype(int), np.ones(2 * clear + 1, dtype=int), mode="same") > 0
    usable = single & ~near
    runs, start = [], None
    for index, flag in enumerate(np.append(usable, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * proj.FRAME >= DECOY_MIN:
                runs.append((start * proj.FRAME, index * proj.FRAME))
            start = None
    return runs


def chunks(session):
    import soundfile

    regions = [(r["start"], r["end"]) for r in json.loads((proj.ALI / session / "projection.json").read_text(encoding="utf-8"))]
    tracks = proj.timeline(session)
    duration = soundfile.info(str(source(session))).duration
    runs = decoy_runs(proj.activity(tracks, int(duration / proj.FRAME) + 2))
    result, current = [], []
    for region in sorted(regions):
        if current and region[1] + EDGE - (current[0][0] - EDGE) > CHUNK_MAX:
            result.append(current)
            current = []
        current.append(region)
    if current:
        result.append(current)
    out = []
    for group in result:
        begin, end = max(0.0, group[0][0] - EDGE), min(duration, group[-1][1] + EDGE)
        inside = [run for run in runs if run[0] >= begin and run[1] <= end]
        picks = [inside[round(i * (len(inside) - 1) / max(1, len(group) - 1))] for i in range(len(group))] if inside else []
        decoys = []
        for run in dict.fromkeys(picks):
            length = min(DECOY_LEN, run[1] - run[0])
            middle = (run[0] + run[1]) / 2
            decoys.append((middle - length / 2, middle + length / 2))
        spans = sorted([(s, e, "real") for s, e in group] + [(s, e, "decoy") for s, e in decoys])
        out.append({"begin": begin, "end": end, "spans": [{"id": i + 1, "start": s, "end": e, "kind": k} for i, (s, e, k) in enumerate(spans)]})
    return out


def stable_segments(session):
    payload = json.loads((proj.ALI / session / f"{session}-stable.json").read_text(encoding="utf-8"))
    return [(i + 1, s["start"], s["end"], s["text"]) for i, s in enumerate(payload["segments"])]


def build_messages(chunk, segments):
    lines = "\n".join(
        f"{line}|{max(0.0, start - chunk['begin']):.1f}|{end - chunk['begin']:.1f}|{text.strip()}"
        for line, start, end, text in segments
        if end > chunk["begin"] and start < chunk["end"]
    )
    spans = "\n".join(f"{s['id']}|{s['start'] - chunk['begin']:.1f}|{s['end'] - chunk['begin']:.1f}" for s in chunk["spans"])
    system = (HERE / "overlap_fix_system.md").read_text(encoding="utf-8")
    user = string.Template((HERE / "overlap_fix_user.md").read_text(encoding="utf-8")).substitute(
        duration=f"{chunk['end'] - chunk['begin']:.1f}", lines=lines or "（本段没有现有字幕）", spans=spans
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


#: Without a task group, `complete` resolves the role against the *global*
#: default routes and ignores the router handed to the client -- the first agy
#: smoke test quietly landed on free 3.5 flash-lite that way. A cell lookup is
#: what reads `client.router.routes`, and under `test_profile` it pins the
#: preset's test target (agy: `local-agy-gemini-3_8-flash`).
CALL_ROUTING = {"lite": {}, "agy": {"task_group": "planning-mm", "difficulty": "quality"}}


def client_for(model):
    from finesub.llm.client import RoleClient

    if model == "lite":
        return RoleClient()
    from finesub.llm.routing.execution_policy import load_execution_settings
    from finesub.llm.routing.model_router import ModelRouter
    from finesub.llm.routing.model_routes import load_model_routes

    settings = load_execution_settings()
    routes = load_model_routes(user_config={"preset": "agy"})
    return RoleClient(router=ModelRouter(policy_id=settings.policy_id, routes=routes), execution_settings=settings, test_profile=True)


def parse(text):
    match = BLOCK.search(text or "")
    if not match:
        return None
    body = match.group(1)
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("kind") in ("add", "fix"):
            rows.append(row)
    return rows


def ask(model, run):
    import soundfile

    from finesub.llm.media_upload import window_media_ref
    from finesub.llm.routing.config import LLMRole

    client = client_for(model)
    for session in ref.SESSIONS:
        out = proj.ALI / session / f"llm-{model}-run{run}.json"
        if out.exists():
            continue
        audio, rate = soundfile.read(str(source(session)), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        segments = stable_segments(session)
        clip_dir = proj.ALI / session / "llm-clips"
        clip_dir.mkdir(exist_ok=True)
        records = []
        for index, chunk in enumerate(chunks(session)):
            clip = clip_dir / f"c{index:02d}.wav"
            if not clip.exists():
                soundfile.write(str(clip), audio[int(chunk["begin"] * rate) : int(chunk["end"] * rate)], rate)
            record = dict(chunk)
            try:
                media = window_media_ref(clip, execution_settings=client.execution_settings, routes=client.router.routes)
                reply = client.complete(
                    LLMRole.LIGHTWEIGHT_MULTIMODAL, build_messages(chunk, segments), file_ref=media, **CALL_ROUTING[model]
                )
                record["raw"] = reply.content
                record["model"] = str(getattr(reply, "model", "") or getattr(reply, "target_id", ""))
                record["rows"] = parse(record["raw"])
            except Exception as error:  # record and move on: a failed chunk is "nothing added"
                record["raw"], record["rows"], record["error"] = "", None, repr(error)
            records.append(record)
            added = len([r for r in record["rows"] or () if r.get("kind") == "add"])
            print(f"{session} c{index:02d} spans={len(chunk['spans'])} adds={added} {'PARSE-FAIL' if record['rows'] is None else ''} {record.get('error', '')[:120]}", flush=True)
        out.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")


def unmatched_share(text, reference_bigrams):
    grams = ref.bigrams(text)
    total = sum(grams.values())
    return None if not total else sum(n for g, n in grams.items() if g not in reference_bigrams) / total


def nearby_bigrams(references, start, end):
    grams = set()
    for row in references:
        if row["start"] < end + 1 and row["end"] > start - 1:
            grams |= set(ref.bigrams(row["text"]))
    return grams


def score_run(model, run, prefix="llm"):
    gain_rows, base_rows = [], []
    add_total = add_unmatched = 0
    decoys = decoys_confab = 0
    fixes = Counter()
    failures = chunks_total = 0
    for session in ref.SESSIONS:
        records = json.loads((proj.ALI / session / f"{prefix}-{model}-run{run}.json").read_text(encoding="utf-8"))
        references = ref.reference(session)
        segments = {line: (start, end, text) for line, start, end, text in stable_segments(session)}
        adds = []
        for record in records:
            chunks_total += 1
            if record["rows"] is None:
                failures += 1
                continue
            by_span = {}
            for row in record["rows"]:
                if row["kind"] == "add":
                    try:
                        start, end = record["begin"] + float(row["start"]), record["begin"] + float(row["end"])
                    except (TypeError, ValueError, KeyError):
                        continue
                    text = ref.normalize(str(row.get("text", "")))
                    if not text:
                        continue
                    adds.append((start, end, text))
                    share = unmatched_share(text, nearby_bigrams(references, start, end))
                    if share is not None:
                        grams = sum(ref.bigrams(text).values())
                        add_total += grams
                        add_unmatched += round(share * grams)
                    by_span.setdefault(row.get("span"), []).append(share)
                elif row["kind"] == "fix" and row.get("line") in segments:
                    start, end, original = segments[row["line"]]
                    near = nearby_bigrams(references, start, end)
                    before, after = unmatched_share(ref.normalize(original), near), unmatched_share(ref.normalize(str(row.get("text", ""))), near)
                    if before is not None and after is not None:
                        fixes["better" if after < before else "worse" if after > before else "same"] += 1
            for span in record["spans"]:
                if span["kind"] == "decoy":
                    decoys += 1
                    if any(share is not None and share > 0.5 for share in by_span.get(span["id"], ())):
                        decoys_confab += 1
        regions = json.loads((proj.ALI / session / "projection.json").read_text(encoding="utf-8"))
        words = ref.hypothesis(session)
        rows = ref.score(session)
        for a, b in ref.pairs(rows):
            middle = (max(a["start"], b["start"]) + min(a["end"], b["end"])) / 2
            if not any(r["start"] <= middle <= r["end"] for r in regions):
                continue
            worse = min((a, b), key=lambda m: m["recall"])
            extra = "".join(text for s, e, text in adds if worse["start"] - ref.PAD <= (s + e) / 2 <= worse["end"] + ref.PAD)
            base_rows.append(worse["recall"])
            gain_rows.append(ref.recall(worse["text"], ref.window_text(words, worse["start"], worse["end"]) + extra))
    return {
        "pairs": len(base_rows),
        "base": mean(base_rows),
        "with_adds": mean(gain_rows),
        "gain": mean(gain_rows) - mean(base_rows),
        "add_bigrams": add_total,
        "add_spurious": add_unmatched / add_total if add_total else 0.0,
        "decoys": decoys,
        "decoy_confab": decoys_confab / decoys if decoys else 0.0,
        "fixes": dict(fixes),
        "parse_failures": f"{failures}/{chunks_total}",
    }


def score(model, prefix="llm"):
    results = []
    for run in (1, 2):
        if not (proj.ALI / ref.SESSIONS[0] / f"{prefix}-{model}-run{run}.json").exists():
            continue
        result = score_run(model, run, prefix)
        results.append(result)
        print(f"{prefix}-{model} run{run}: {json.dumps(result, ensure_ascii=False)}")
    if not results:
        return
    gain = mean(r["gain"] for r in results)
    spurious = mean(r["add_spurious"] for r in results)
    confab = mean(r["decoy_confab"] for r in results)
    verdict = [
        ("召回收益 >= 5 pp", gain >= GAIN_MIN, f"{100 * gain:+.1f} pp"),
        ("新增虚增率 <= 0.443", spurious <= ADD_SPURIOUS_MAX, f"{spurious:.3f}"),
        ("诱饵编造率 <= 10%", confab <= DECOY_MAX, f"{confab:.1%}"),
    ]
    for name, ok, value in verdict:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {value}")
    print("  => " + ("可行" if all(ok for _, ok, _ in verdict) else "不做"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", nargs=2, metavar=("MODEL", "RUN"))
    parser.add_argument("--score", metavar="MODEL")
    parser.add_argument("--show-chunks", action="store_true")
    arguments = parser.parse_args()
    if arguments.show_chunks:
        for session in ref.SESSIONS:
            items = chunks(session)
            spans = [s for c in items for s in c["spans"]]
            print(session, "chunks", len(items), "real", sum(s["kind"] == "real" for s in spans), "decoy", sum(s["kind"] == "decoy" for s in spans),
                  "chunk seconds", round(mean(c["end"] - c["begin"] for c in items), 1))
    if arguments.ask:
        ask(arguments.ask[0], int(arguments.ask[1]))
    if arguments.score:
        score(arguments.score)


if __name__ == "__main__":
    main()
