"""R2 and its gates (report `docs/reports/2026-09-26-ts-diar-models.md`, 补充（2026-10-13）; closed plan `git show bff4b478:docs/plans/asr-recall-timing-plan.md`, 召回): an audio judge (agy Gemini 3.8 Flash) decides,
clip by clip, whether each route wrote what was said.

A batch mixes three kinds of item, the judge told nothing about which is which:
  real    R1 candidates (`recall.candidates`), sampled
  plant   a stretch every route wrote (`consensus`), deleted from some routes in a copy (G1 / G2): the deleted
          routes must be judged a miss, the rest not; strata `one` (a single route), `moss_only` (every word-timed
          route, so only MOSS Pro has it), `word_only` (MOSS Pro and production, so only MAI2 / 35t have it)
  decoy   known not-a-miss (G3): `respell` (a route's text there rewritten in kana), `misplace` (its text moved to
          just after the clip), `halluc` (a sentence from elsewhere put into a silent stretch: the route wrote too much)

The judge sees every route of the episode as A, B, C… (shuffled per clip; the key is kept), each as its units near
the clip (phrases for word-timed routes, segments for MOSS Pro) with times relative to the clip, and the fansub's
lines when they are Japanese. One call takes PER_CALL clips, one audio file each.

    python -m tools.ts_diar_explore.recall_judge build --items hina-72 hina-25 --tag tune1 [--real 20 --plants 8 --decoys 5]
    python -m tools.ts_diar_explore.recall_judge run --tag tune1
    python -m tools.ts_diar_explore.recall_judge gates --tag tune1
    python -m tools.ts_diar_explore.recall_judge g1 --items hina-72 hina-25 [--per-round 40]   # mechanical, no calls
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import random
import re
import string
import sys
from pathlib import Path

import numpy as np

from . import recall as R
from .common import MAIN, vocal_track

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "diarization_explore"))
import row_truth as rt  # noqa: E402

SYSTEM = HERE / "recall_judge_system.md"
VERDICTS = ("全", "错写", "漏一截", "整段漏", "多写")
MISS = ("漏一截", "整段漏")
SPEECH = ("正片", "背景", "无")
KINDS = ("内容", "短反应", "填充", "-")
PER_CALL = 16
PAD = 0.5          # audio either side of a candidate
CTX = 2.0          # route text shown either side of the clip
MAX_CLIP = 15.0
MIN_CLIP = 3.0
EFFORT = "high"
NOTE = ("这一批片段的录音是工作目录里的 {files}，各自的长度见下面每个片段。在同一次回复里依次调用 view_file 读完全部录音，"
        "最后读提示 {after}（录音会完整返回，不会截断），都读完再开始判断；之后不要再读文件，也不要运行任何命令。")
AFTER = "录音已经全部读完。现在按要求逐个片段输出 <clip> 块，按片段编号的顺序，不要输出别的。\n"


# --- what the judge is shown -------------------------------------------------------------------------------

def shown_units(src: dict, route: str) -> list[dict]:
    units = src[route]["units"]
    return R.phrases(units) if route in R.WORD_TIMED else [u for u in units if R.S.norm(u.get("text", ""))]


def render_clip(cid: str, item: dict, src: dict) -> tuple[str, dict]:
    """The clip's text block and its letter key {letter: route}."""
    a, b = item["clip"]
    routes = [r for r in R.ROUTES if r in src]
    rng = random.Random(f"{item['id']}{item.get('salt', '')}")
    letters = dict(zip(string.ascii_uppercase, rng.sample(routes, len(routes))))
    lines = [f"### 片段 {cid}（录音 {cid}.mp4，长 {b - a:.1f} 秒）"]
    for letter, route in letters.items():
        near = [u for u in shown_units(src, route) if u["end"] > a - CTX and u["start"] < b + CTX]
        lines.append(f"{letter}:" + ("".join(f"\n  [{u['start'] - a:+.1f} → {u['end'] - a:+.1f}] {u['text'].strip()}" for u in near)
                                      or " （这段时间附近没有写字）"))
    ref = [x for x in src.get("ref", {}).get("lines", []) if x.get("text") and x["end"] > a - CTX and x["start"] < b + CTX]
    if ref:
        lines.append("参考字幕（字幕组整理，只作参考）:" + "".join(
            f"\n  [{x['start'] - a:+.1f} → {x['end'] - a:+.1f}] {x['text']}" for x in ref))
    return "\n".join(lines), letters


def parse(text: str, clips: dict[str, dict]) -> dict | None:
    """{cid: {heard, speech, kind, verdicts: {letter: (verdict, note)}}} when every clip and letter is there."""
    out = {}
    for cid, body in re.findall(r"<clip\s+(c\d+)>(.*?)</clip>", text or "", re.S):
        got = {"verdicts": {}}
        for line in body.strip().splitlines():
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key in ("heard", "speech", "kind"):
                got[key] = value
            elif len(key) == 1 and key in string.ascii_uppercase:
                verdict, _, note = value.partition("|")
                got["verdicts"][key] = (verdict.strip(), note.strip())
        out[cid] = got
    for cid, letters in clips.items():
        got = out.get(cid)
        if (not got or got.get("speech") not in SPEECH or got.get("kind") not in KINDS
                or set(got["verdicts"]) != set(letters)
                or any(v not in VERDICTS for v, _ in got["verdicts"].values())):
            return None
    return out


# --- plants and decoys -------------------------------------------------------------------------------------

def _hira_map(text: str) -> tuple[str, list[int]]:
    """`text` as hiragana and, for each hiragana character, the index of the source character it came from
    (in the normalised text)."""
    norm = R.S.norm(text)
    out, idx, at = [], [], 0
    for tok in R._kakasi().convert(norm):
        n = len(tok["orig"])
        h = R.S.norm(tok["hira"])
        for k, ch in enumerate(h):
            out.append(ch)
            idx.append(at + min(n - 1, k * n // max(1, len(h))))
        at += n
    return "".join(out), idx


def match_span(text: str, target: str, least: float = 0.8) -> tuple[int, int] | None:
    """The span of `text` (normalised source characters) that reads as `target` (hiragana), when >= `least` of it
    lines up."""
    h, idx = _hira_map(text)
    blocks = [m for m in difflib.SequenceMatcher(None, target, h, autojunk=False).get_matching_blocks() if m.size]
    if not blocks or sum(m.size for m in blocks) < least * len(target):
        return None
    lo, hi = blocks[0].b, blocks[-1].b + blocks[-1].size - 1
    return idx[lo], idx[hi] + 1


def consensus(src: dict, rng: random.Random, n: int, min_chars: int = 5, max_chars: int = 20,
              spacing: float = 10.0) -> list[dict]:
    """Production phrases of min_chars..max_chars that every other route also wrote (>= 80% of it within ±0.5 s),
    spaced >= `spacing` s apart."""
    out, used = [], []
    phrases = [p for p in R.phrases(src["prod"]["units"]) if min_chars <= len(R.hira(p["text"])) <= max_chars]
    rng.shuffle(phrases)
    for p in phrases:
        if len(out) >= n:
            break
        if any(abs(p["start"] - u) < spacing for u in used):
            continue
        h = R.hira(p["text"])
        ok = True
        for route in R.ROUTES:
            if route == "prod" or route not in src:
                continue
            near = [u.get("text", "") for u in src[route]["units"] if u["end"] > p["start"] - 0.5 and u["start"] < p["end"] + 0.5]
            texts = ["".join(near)] if route in R.WORD_TIMED else near  # a segment route must hold it in one segment
            if not any(match_span(x, h) for x in texts):
                ok = False
                break
        if ok:
            out.append({"start": p["start"], "end": p["end"], "hira": h, "text": p["text"]})
            used.append(p["start"])
    return out


def delete(src: dict, route: str, stretch: dict) -> bool:
    """Remove the stretch from one route of the copy `src`: a word-timed route loses its words inside, MOSS Pro the
    characters of its segment that read as it (the whole segment when nothing else is left)."""
    a, b = stretch["start"], stretch["end"]
    units = src[route]["units"]
    if route in R.WORD_TIMED:
        keep = [u for u in units if not (a - 0.05 <= (u["start"] + u["end"]) / 2 <= b + 0.05)]
        if len(keep) == len(units):
            return False
        src[route]["units"] = src[route]["spans"] = keep
        return True
    for k, u in enumerate(units):
        if u["end"] > a - 0.5 and u["start"] < b + 0.5:
            span = match_span(u.get("text", ""), stretch["hira"])
            if span:
                text = R.S.norm(u["text"])
                rest = text[:span[0]] + text[span[1]:]
                units[k] = {**u, "text": rest}
                src[route]["units"] = src[route]["spans"] = [x for x in units if R.S.norm(x.get("text", ""))]
                return True
    return False


def respell(src: dict, route: str, stretch: dict) -> bool:
    a, b = stretch["start"], stretch["end"]
    units = src[route]["units"]
    if route in R.WORD_TIMED:
        hit = False
        for k, u in enumerate(units):
            if a - 0.05 <= (u["start"] + u["end"]) / 2 <= b + 0.05:
                units[k] = {**u, "text": R.hira(u.get("text", "")) or u.get("text", "")}
                hit = True
        return hit
    for k, u in enumerate(units):
        if u["end"] > a - 0.5 and u["start"] < b + 0.5:
            span = match_span(u.get("text", ""), stretch["hira"])
            if span:
                text = R.S.norm(u["text"])
                units[k] = {**u, "text": text[:span[0]] + R.hira(text[span[0]:span[1]]) + text[span[1]:]}
                return True
    return False


def misplace(src: dict, route: str, stretch: dict, clip_end: float) -> bool:
    """Move the route's text for the stretch to a unit starting 0.6 s after the clip."""
    if not delete(src, route, stretch):
        return False
    t0 = clip_end + 0.6
    src[route]["units"] = src[route]["spans"] = sorted(
        src[route]["units"] + [{"start": t0, "end": t0 + stretch["end"] - stretch["start"], "text": stretch["text"]}],
        key=lambda u: u["start"])
    return True


def silences(src: dict, rng: random.Random, n: int, length: float = 3.0) -> list[tuple[float, float]]:
    """Stretches of `length` s where no source writes anything within 1 s."""
    duration = max(u["end"] for s in src.values() for u in s["units"])
    m = np.zeros(int(duration / R.STEP) + 1, bool)
    for s in src.values():
        m |= R._mask(s["units"], len(m), 1.0)
    runs = [(x, y) for x, y in R._runs(~m, length + 1.0)]
    rng.shuffle(runs)
    return [(x + 0.5, x + 0.5 + length) for x, y in runs[:n]]


# --- building a batch ------------------------------------------------------------------------------------

def clip_of(a: float, b: float) -> tuple[float, float]:
    a, b = a - PAD, b + PAD
    if b - a < MIN_CLIP:  # character times inside a long unit are estimates: a short clip may miss the words
        mid = (a + b) / 2
        a, b = mid - MIN_CLIP / 2, mid + MIN_CLIP / 2
    if b - a > MAX_CLIP:
        mid = (a + b) / 2
        a, b = mid - MAX_CLIP / 2, mid + MAX_CLIP / 2
    return max(0.0, a), b


#: plants per episode cycle through these strata: `moss` and `word_only` give segment-route cells, `moss_only`,
#: `word_only` and `one` word-route cells, so both gate cells fill at a similar rate
PLANT_CYCLE = ("moss", "word_only", "moss_only", "moss", "one", "word_only", "moss", "moss_only", "word_only", "one")
LONG_CLIP = 4.0


def overlaps(a, b, spans, gap=1.0):
    return any(a < y + gap and b > x - gap for x, y in spans)


def home(tag):
    """(judge folder, candidates folder) of a tag: the frozen copy for the tags frozen there (tune1, formal,
    formal-r2), else WORK, where new candidates and runs go."""
    root = R.FROZEN if (R.FROZEN / "judge" / tag).is_dir() else R.WORK
    return root / "judge" / tag, root


def writable(tag):
    folder, root = home(tag)
    if root == R.FROZEN:
        raise SystemExit(f"{tag} is frozen ({folder}): pick another tag")
    return folder


def found_of(root, ident):
    return json.loads((root / f"{ident}-candidates.json").read_text(encoding="utf-8"))


def build(items, tag, real, plants, decoys, long_points, seed):
    rng = random.Random(seed)
    folder = writable(tag)
    folder.mkdir(parents=True, exist_ok=True)
    batch = []
    for ident in items:
        base = R.sources(ident)
        found = found_of(R.WORK, ident)
        word_routes = [r for r in R.WORD_TIMED if r in base]
        taken = []
        stretches = consensus(base, rng, plants + 2 * decoys)
        for k, st in enumerate(stretches[:plants]):
            stratum = PLANT_CYCLE[k % len(PLANT_CYCLE)]
            routes = {"moss": ["mosspro"], "word_only": ["mosspro", "prod"], "moss_only": word_routes,
                      "one": [rng.choice(word_routes)]}[stratum]
            batch.append({"id": f"{ident}-p{len(batch):04d}", "ident": ident, "kind": "plant", "stratum": stratum,
                          "clip": clip_of(st["start"], st["end"]), "stretch": st, "edits": [["delete", r] for r in routes]})
            taken.append(batch[-1]["clip"])
        for k, st in enumerate(stretches[plants:plants + 2 * decoys]):
            how = "respell" if k % 2 == 0 else "misplace"
            route = "mosspro" if (k // 2) % 2 == 0 else rng.choice(word_routes)
            batch.append({"id": f"{ident}-d{len(batch):04d}", "ident": ident, "kind": "decoy", "stratum": how,
                          "clip": clip_of(st["start"], st["end"]), "stretch": st, "edits": [[how, route]]})
            taken.append(batch[-1]["clip"])
        donors = [p for p in R.phrases(base["prod"]["units"]) if 6 <= len(R.hira(p["text"])) <= 20]
        for k, (a, b) in enumerate(silences(base, rng, decoys)):
            route = "mosspro" if k % 2 == 0 else rng.choice(word_routes)
            batch.append({"id": f"{ident}-d{len(batch):04d}", "ident": ident, "kind": "decoy", "stratum": "halluc",
                          "clip": (a, b), "stretch": {"start": a + 0.5, "end": b - 0.5, "text": rng.choice(donors)["text"]},
                          "edits": [["insert", route]]})
            taken.append((a, b))
        # long gaps: every route's, merged across routes; a few evenly spaced points each
        gaps = sorted((g["start"], g["end"], r) for r, gs in found["long"].items() for g in gs)
        merged = []
        for a, b, r in gaps:
            if merged and a < merged[-1]["end"]:
                merged[-1]["end"] = max(merged[-1]["end"], b)
                merged[-1]["routes"].add(r)
            else:
                merged.append({"start": a, "end": b, "routes": {r}})
        for g in merged:
            for k in range(long_points):
                mid = g["start"] + (g["end"] - g["start"]) * (k + 0.5) / long_points
                clip = (mid - LONG_CLIP / 2, mid + LONG_CLIP / 2)
                batch.append({"id": f"{ident}-l{len(batch):04d}", "ident": ident, "kind": "long", "clip": clip,
                              "gap": [g["start"], g["end"]], "gap_routes": sorted(g["routes"]), "edits": []})
                taken.append(clip)
        # real candidates, 70% with content (another source wrote >= 5 characters there), clear of the items above
        cands = [c for c in found["items"] if not overlaps(c["start"], c["end"], taken)]
        content = [c for c in cands if max((len(R.hira(t)) for k, t in c["text"].items() if k != "ref"), default=0) >= 5]
        picks = rng.sample(content, min(len(content), int(real * 0.7)))
        rest = [c for c in cands if c not in picks]
        picks += rng.sample(rest, min(len(rest), real - len(picks)))
        for c in picks:
            batch.append({"id": f"{ident}-r{len(batch):04d}", "ident": ident, "kind": "real", "clip": clip_of(c["start"], c["end"]),
                          "flags": c["flags"], "content": c in content, "weight": len(content if c in content else rest),
                          "edits": []})
    rng.shuffle(batch)
    (folder / "batch.json").write_text(json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8")
    counts = {}
    for it in batch:
        key = (it["kind"], it.get("stratum", ""))
        counts[key] = counts.get(key, 0) + 1
    for k, v in sorted(counts.items()):
        print(k, v)
    print("items", len(batch), "audio s", round(sum(it["clip"][1] - it["clip"][0] for it in batch)))


def apply(src: dict, it: dict) -> dict:
    """The sources as this item's clip shows them: a copy with the item's edits made."""
    if not it["edits"]:
        return src
    src = copy.deepcopy(src)
    for how, route in it["edits"]:
        st = it["stretch"]
        if how == "delete":
            done = delete(src, route, st)
        elif how == "respell":
            done = respell(src, route, st)
        elif how == "misplace":
            done = misplace(src, route, st, it["clip"][1])
        else:
            unit = {"start": st["start"], "end": st["end"], "text": st["text"]}
            src[route]["units"] = src[route]["spans"] = sorted(src[route]["units"] + [unit], key=lambda u: u["start"])
            done = True
        if not done:
            raise RuntimeError(f"{it['id']}: {how} on {route} found nothing to change")
    return src


# --- G1: does R1 list what was deleted -------------------------------------------------------------------

G1_LENGTHS = {"<=4": (1, 4), "5-15": (5, 15), ">15": (16, 40)}
G1_SPACING = 25.0  # R1 aligns in 20 s chunks: plants this far apart do not see each other


def g1(items, seed, per_round):
    """Plants by stratum and length, many per episode copy (G1_SPACING apart), then R1 on the copy: a plant is found
    when every route it was deleted from has a candidate (or a long gap) within 0.5 s of it."""
    rng = random.Random(seed)
    strata = {"one": None, "moss_only": "word", "word_only": ["mosspro", "prod"]}
    tally = {}
    for ident in items:
        base = R.sources(ident)
        for stratum, routes in strata.items():
            for length, (lo, hi) in G1_LENGTHS.items():
                src = copy.deepcopy(base)
                planted = []
                for st in consensus(base, rng, per_round, lo, hi, G1_SPACING):
                    rs = (routes if isinstance(routes, list) else [r for r in R.WORD_TIMED if r in base] if routes
                          else [rng.choice([r for r in R.ROUTES if r in base])])
                    if all(delete(src, r, st) for r in rs):
                        planted.append((st, rs))
                found = R.candidates(ident, src)
                for st, rs in planted:
                    a, b = st["start"] - 0.5, st["end"] + 0.5
                    hit = all(any(it["start"] < b and it["end"] > a and r in it["flags"] for it in found["items"])
                              or any(g["start"] < b and g["end"] > a for g in found["long"].get(r, [])) for r in rs)
                    for r in rs:
                        c = tally.setdefault((route_type(r), stratum, length), [0, 0])
                        c[0] += hit
                        c[1] += 1
                print(f"{ident} {stratum} {length}: {len(planted)} planted", flush=True)
    print("generator sensitivity (planted misses R1 lists), by route type, stratum and length:")
    for k, (hit, n) in sorted(tally.items()):
        print(f"  {k[0]:8s} {k[1]:10s} {k[2]:5s} {hit}/{n} = {hit / max(n, 1):.1%}")


# --- running ---------------------------------------------------------------------------------------------

def render_call(chunk, folder, sources):
    """Clips into `folder` and the call's content -> (content, {cid: letters}, {cid: item id})."""
    folder.mkdir(parents=True, exist_ok=True)
    blocks, keys, ids, files = [], {}, {}, []
    for n, it in enumerate(chunk, 1):
        cid = f"c{n:02d}"
        a, b = it["clip"]
        wav = rt.cut_audio(vocal_track(it["ident"]), a, b, folder / f"{cid}.wav")
        mp4 = folder / f"{cid}.mp4"
        if not mp4.exists():
            from finesub.media.ffmpeg import containerize_audio_for_agy
            containerize_audio_for_agy(wav, mp4)
        text, letters = render_clip(cid, it, apply(sources[it["ident"]], it))
        blocks.append(text)
        keys[cid] = letters
        ids[cid] = it["id"]
        files.append(mp4.name)
    after = folder / "after.txt"
    after.write_text(AFTER, encoding="utf-8")
    content = (SYSTEM.read_text(encoding="utf-8") + "\n\n=====\n\n" + NOTE.format(files="、".join(files), after=after.name)
               + "\n\n" + "\n\n".join(blocks))
    return content, keys, ids


def run(tag, parallel=1):
    from concurrent.futures import ThreadPoolExecutor
    folder = writable(tag)
    batch = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
    sources = {ident: R.sources(ident) for ident in {it["ident"] for it in batch}}
    calls = [(f"call{k // PER_CALL:03d}", batch[k:k + PER_CALL]) for k in range(0, len(batch), PER_CALL)]
    calls = [(name, chunk) for name, chunk in calls if not (folder / f"{name}.json").exists()]
    rendered = [(name, *render_call(chunk, R.WORK / "judge" / tag / name, sources)) for name, chunk in calls]

    def one(job):
        name, content, keys, ids = job
        work = R.WORK / "judge" / tag / name
        attempts, answer, text = [], None, ""
        for attempt in range(3):
            text, usage, conversation, status, error = rt.agy(content, work, EFFORT)
            attempts.append({"usage": usage, "conversation": conversation, "status": status, "error": error})
            answer = parse(text, keys)
            if answer is None:
                kept = rt.last_reply(conversation)
                if (answer := parse(kept, keys)) is not None:
                    text = kept
            if answer is not None:
                break
            print(f"{name}: {status} {error[:100]}; retry", flush=True)
        payload = {"answer": answer, "keys": keys, "ids": ids, "raw": text, "model": f"{rt.MODEL}-{EFFORT}",
                   "attempts": attempts, "content_chars": len(content)}
        path = folder / f"{name}.json"
        (path if answer else path.with_suffix(".failed.json")).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        used = sum(a["usage"].get("input_tokens", 0) for a in attempts)
        print(f"{name}: {'ok' if answer else 'GAVE UP'} after {len(attempts)}, input {used}", flush=True)

    with ThreadPoolExecutor(parallel) as pool:
        list(pool.map(one, rendered))


# --- reading ---------------------------------------------------------------------------------------------

def verdicts(tag):
    """[(item, {route: (verdict, note)}, clip answer)] over the answered calls."""
    folder = home(tag)[0]
    batch = {it["id"]: it for it in json.loads((folder / "batch.json").read_text(encoding="utf-8"))}
    out = []
    for path in sorted(folder.glob("call*.json")):
        if path.name.endswith(".failed.json"):
            continue
        got = json.loads(path.read_text(encoding="utf-8"))
        for cid, answer in got["answer"].items():
            letters = got["keys"][cid]
            out.append((batch[got["ids"][cid]], {letters[l]: v for l, v in answer["verdicts"].items()}, answer))
    return out


def quotes(note: str) -> list[str]:
    """The note's quoted words as hiragana pieces (a quote split at commas: one may run several things together)."""
    return [h for part in re.findall(r"「([^」]*)」", note) for piece in re.split(r"[、，,]", part) if (h := R.hira(piece))]


def _hits(piece: str, target: str) -> int:
    return sum(m.size for m in difflib.SequenceMatcher(None, piece, target, autojunk=False).get_matching_blocks())


def about(verdict, note: str, stretch: dict) -> bool:
    """A miss verdict about the planted / decoy stretch, not about other speech in the clip (a clip of >= MIN_CLIP
    often holds someone else's interjection that the route did miss): 整段漏, or a quoted piece >= 60% the stretch's
    words and sharing >= 3 characters with it (two-character quotes such as えー match by chance)."""
    if verdict == "整段漏":
        return True
    if verdict != "漏一截":
        return False
    target = R.hira(stretch["text"])
    return bool(target) and any(_hits(q, target) >= 0.6 * len(q) and _hits(q, target) >= min(3, len(target))
                                for q in quotes(note))


#: interjections and answers (短反应), and fillers (填充): a miss whose every quoted piece is one of these is not content
REACTIONS = {R.hira(w) for w in (
    "うん うんうん うんうんうん はい はいはい ええ へえ へー ほう ほー そう そうそう そうですね そうね そうだね ああ あー "
    "あ おお おー お ね ねえ ねー なるほど 確かに たしかに ふーん ふうん わあ わー うわ うわあ うわー はあ よし おっ "
    "いや いやいや いやあ やば すごい すげー マジ まじ 何 なに").split()}
FILLERS = {R.hira(w) for w in (
    "え えー えっ えっと えーと あの あのー その まあ ま うーん んー ん なんか あれ ふふ ふふふ ふふん あはは ははは").split()}
FILLER_NOTE = re.compile(r"笑|语气|重复|说错|口头禅|吸气|叹气")
KANJI = re.compile(r"[\u4e00-\u9fff]")


def route_kind(verdict, note: str, clip_kind: str) -> str:
    """The kind of what this route missed, from its note: 填充 when the note says laughter, a repeat or a restart, or
    every quoted piece is a filler; 短反应 when every piece is an interjection or answer (or <= 2 kana with no
    kanji); else 内容. Without a quote (整段漏, or none given): the clip's kind, which the judge gives for the clip as
    a whole (the most important of what any route missed)."""
    pieces = [(raw, h) for part in re.findall(r"「([^」]*)」", note) for raw in re.split(r"[、，,]", part)
              if (h := R.hira(raw))]
    if verdict == "整段漏" or not pieces:
        return clip_kind if clip_kind in KINDS[:3] else "内容"
    if FILLER_NOTE.search(note) or all(h in FILLERS for _, h in pieces):
        return "填充"
    if all(h in REACTIONS or h in FILLERS or (len(h) <= 2 and not KANJI.search(raw)) for raw, h in pieces):
        return "短反应"
    return "内容"


def route_type(route):
    return "word" if route in R.WORD_TIMED else "segment"


def rerun(tag, n, seed):
    """G4: a second batch `<tag>-r2` of n items from `tag` (gate items and real ones, those with a miss first), in a
    new order and with the letters drawn anew (`salt`), so each call holds other company."""
    rng = random.Random(seed)
    rows = verdicts(tag)
    missed = [it for it, v, _ in rows if any(x[0] in MISS for x in v.values())]
    rest = [it for it, v, _ in rows if it not in missed]
    pick = rng.sample(missed, min(n, len(missed)))
    pick += rng.sample(rest, min(len(rest), n - len(pick)))
    pick = [{**it, "salt": "r2"} for it in pick]
    rng.shuffle(pick)
    folder = writable(f"{tag}-r2")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "batch.json").write_text(json.dumps(pick, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(pick)} items ({min(n, len(missed))} with a miss) -> {folder}")


def g4(tag):
    first = {it["id"]: v for it, v, _ in verdicts(tag)}
    pairs = [(first[it["id"]], v) for it, v, _ in verdicts(f"{tag}-r2") if it["id"] in first]
    cells = [(a[r][0] in MISS, b[r][0] in MISS) for a, b in pairs for r in a if r in b]
    either = [c for c in cells if c[0] or c[1]]
    same = sum(c[0] == c[1] for c in either)
    print(f"G4: {len(pairs)} clips, {len(cells)} cells; judged a miss at least once {len(either)}, both times {same} "
          f"= {same / max(len(either), 1):.1%} (plan: >= 80%)")


def cp(hit, n):
    """95% Clopper–Pearson interval of hit / n."""
    from scipy.stats import beta
    lo = beta.ppf(0.025, hit, n - hit + 1) if hit else 0.0
    hi = beta.ppf(0.975, hit + 1, n - hit) if hit < n else 1.0
    return lo, hi


def gate_cells(rows):
    """{type: {"sens": [hit, n], "sens_clips": [hit, n], "fp": {decoy kind: [miss, n]}, "excluded": [notes]}}.
    Sensitivity: the plants' deleted routes (and per clip: every deleted route caught). False misses: respell and
    misplace decoys, the edited route judged to miss the stretch; halluc decoys, per clip and per type of the other
    routes (the ones the judge would wrong by taking the inserted sentence for said): a route of that type judged to
    miss anything, or the silent clip heard as speech. `excluded`: miss verdicts about() set aside."""
    out = {t: {"sens": [0, 0], "sens_clips": [0, 0], "fp": {}, "excluded": [], "inserted": [0, 0]}
           for t in ("word", "segment")}
    for it, v, ans in rows:
        if it["kind"] == "plant":
            deleted = sorted({r for _, r in it["edits"]})
            caught = [about(*v[r], it["stretch"]) for r in deleted]
            for route, hit in zip(deleted, caught):
                c = out[route_type(route)]["sens"]
                c[0] += hit
                c[1] += 1
                if not hit and v[route][0] in MISS:
                    out[route_type(route)]["excluded"].append(("plant", v[route]))
            for kind in {route_type(r) for r in deleted}:
                c = out[kind]["sens_clips"]
                c[0] += all(h for r, h in zip(deleted, caught) if route_type(r) == kind)
                c[1] += 1
        elif it["kind"] == "decoy":
            route = it["edits"][0][1]
            cell = out[route_type(route)]
            if it["stratum"] == "halluc":  # the wronged routes are the others: count the clip under each of their types
                cell["inserted"][0] += v[route][0] == "多写"
                cell["inserted"][1] += 1
                heard = ans["speech"] == "正片"
                for kind in {route_type(r) for r in v if r != route}:
                    c = out[kind]["fp"].setdefault("halluc", [0, 0])
                    c[0] += heard or any(v[r][0] in MISS for r in v if r != route and route_type(r) == kind)
                    c[1] += 1
                continue
            c = cell["fp"].setdefault(it["stratum"], [0, 0])
            miss = about(*v[route], it["stretch"])
            c[0] += miss
            c[1] += 1
            if not miss and v[route][0] in MISS:
                cell["excluded"].append((it["stratum"], v[route]))
    return out


def formal_gates(rows, show_excluded=8) -> dict:
    """Prints G2 / G3 per route type against the plan's bars; -> {type: (sensitivity, lowest fp, highest fp,
    sensitivity's lower bound, fp's upper bound)} (rates by decoy kind; bounds Clopper–Pearson)."""
    out = {}
    for kind, cell in gate_cells(rows).items():
        hit, n = cell["sens"]
        lo, _ = cp(hit, n)
        ok2 = n and hit / n >= 0.90 and lo >= 0.80
        miss = sum(c[0] for c in cell["fp"].values())
        m = sum(c[1] for c in cell["fp"].values())
        _, hi = cp(miss, m)
        ok3 = m and miss / m <= 0.05 and hi <= 0.12
        rates = {k: c[0] / c[1] for k, c in cell["fp"].items() if c[1]}
        ch, cn = cell["sens_clips"]
        print(f"  {kind:8s} G2 sensitivity {hit}/{n} = {hit / max(n, 1):.1%} [lower {lo:.1%}] {'PASS' if ok2 else 'FAIL'}"
              f" (per clip {ch}/{cn}, lower {cp(ch, cn)[0]:.1%}); G3 false misses {miss}/{m} = {miss / max(m, 1):.1%}"
              f" [upper {hi:.1%}] {'PASS' if ok3 else 'FAIL'} ("
              + ", ".join(f"{k} {c[0]}/{c[1]}" for k, c in sorted(cell["fp"].items()))
              + f"); inserted sentence judged 多写 {cell['inserted'][0]}/{cell['inserted'][1]}")
        for what, (verdict, note) in cell["excluded"][:show_excluded]:
            print(f"      set aside ({what}): {verdict} | {note}")
        if len(cell["excluded"]) > show_excluded:
            print(f"      … {len(cell['excluded']) - show_excluded} more set aside")
        out[kind] = (hit / max(n, 1), min(rates.values(), default=0.0), max(rates.values(), default=0.0), lo, hi)
    return out


def corrected(p, sens, fp):
    """Rogan–Gladen: the true share behind an observed share p, given sensitivity and false-positive rate."""
    return min(1.0, max(0.0, (p - fp) / (sens - fp))) if sens > fp else float("nan")


def candidate_of(it, found):
    """The R1 candidate a real item was cut from (its clip is `clip_of` the candidate)."""
    for c in found:
        a, b = clip_of(c["start"], c["end"])
        if abs(a - it["clip"][0]) < 1e-6 and abs(b - it["clip"][1]) < 1e-6:
            return c
    raise LookupError(it["id"])


def inclusion(batch, root, real=None):
    """{item id: probability it was drawn}, redoing `build`'s draw: n_c of the |C| content candidates, then n_r of
    the rest (the other candidates and the content ones not drawn) -- clear of the gate items' clips."""
    out = {}
    for ident in {it["ident"] for it in batch}:
        mine = [it for it in batch if it["ident"] == ident]
        taken = [it["clip"] for it in mine if it["kind"] != "real"]
        found = found_of(root, ident)["items"]
        cands = [c for c in found if not overlaps(c["start"], c["end"], taken)]
        content = [c for c in cands if max((len(R.hira(t)) for k, t in c["text"].items() if k != "ref"), default=0) >= 5]
        n = real or sum(it["kind"] == "real" for it in mine)
        n_c = min(len(content), int(n * 0.7))
        n_r = min(len(cands) - n_c, n - n_c)
        p_rest = n_r / max(1, len(cands) - n_c)
        for it in mine:
            if it["kind"] == "real":
                c = candidate_of(it, cands)
                is_content = any(c is x for x in content)
                out[it["id"]] = (n_c / len(content) + (1 - n_c / len(content)) * p_rest) if is_content else p_rest
    return out


def attributed(route, verdict, note, it, cand, src) -> bool:
    """A real candidate's miss that is about the candidate: 整段漏, or a quoted piece the other sources wrote within
    the candidate's interval (±0.3 s). Misses of other speech in the clip are set aside (`incidental`)."""
    if verdict == "整段漏":
        return True
    a, b = cand["start"] - 0.3, cand["end"] + 0.3
    target = R.hira("".join(u.get("text", "") for name, s in src.items() if name != route
                            for u in s["units"] if u["end"] > a and u["start"] < b))
    return bool(target) and any(_hits(q, target) >= 0.6 * len(q) and _hits(q, target) >= min(3, len(q))
                                for q in quotes(note))


def readout(tag, real=50, reps=2000, seed=11):
    rows = verdicts(tag)
    folder, root = home(tag)
    batch = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
    print("formal gates (plan: G2 >= 90% with lower bound >= 80%; G3 <= 5% with upper bound <= 12%):")
    acc = formal_gates(rows)
    prob = inclusion(batch, root, real)
    idents = sorted({it["ident"] for it in batch})
    src = {ident: R.sources(ident) for ident in idents}
    found = {ident: found_of(root, ident)["items"] for ident in idents}
    minutes = {ident: max(u["end"] for s in src[ident].values() for u in s["units"]) / 60 for ident in idents}
    # per real clip and route: (weight, listed?, attributed miss kind or None, incidental?)
    cells = []
    agree = [0, 0]
    for it, v, ans in rows:
        if it["kind"] != "real":
            continue
        cand = candidate_of(it, found[it["ident"]])
        misses = [r for r in v if v[r][0] in MISS]
        if len(misses) == 1 and ans["kind"] in KINDS[:3] and v[misses[0]][0] == "漏一截":  # check route_kind
            agree[0] += route_kind(*v[misses[0]], ans["kind"]) == ans["kind"]
            agree[1] += 1
        for route, (verdict, note) in v.items():
            miss = verdict in MISS
            att = miss and attributed(route, verdict, note, it, cand, src[it["ident"]])
            cells.append({"ident": it["ident"], "route": route, "w": 1 / prob[it["id"]], "listed": route in it["flags"],
                          "kind": route_kind(verdict, note, ans["kind"]) if att else None, "incidental": miss and not att,
                          "clip": it["id"]})
    print(f"\nper-route kind from the note vs the judge's clip kind, on clips where one route missed: "
          f"{agree[0]}/{agree[1]} agree")
    rng = np.random.default_rng(seed)

    def rates(sub, eps):
        """{route: {kind: misses per minute}} over episodes `eps`, weighted back to every candidate."""
        mins = sum(minutes[e] for e in eps)
        out = {}
        for c in sub:
            if c["ident"] in eps and c["kind"]:
                out.setdefault(c["route"], {}).setdefault(c["kind"], 0.0)
                out[c["route"]][c["kind"]] += c["w"] / mins
        return out

    def boot(eps, a, b, kind="内容"):  # kind None: every attributed miss
        """95% interval of route a minus route b (misses per minute of `kind`), clips resampled within episodes."""
        by = {}
        for c in cells:
            if c["ident"] in eps:
                by.setdefault(c["ident"], {}).setdefault(c["clip"], []).append(c)
        mins = sum(minutes[e] for e in eps)
        sims = []
        for _ in range(reps):
            d = 0.0
            for e, clips in by.items():
                keys = list(clips)
                for k in rng.integers(0, len(keys), len(keys)):
                    for c in clips[keys[k]]:
                        if c["kind"] and (kind is None or c["kind"] == kind):
                            d += c["w"] * ((c["route"] == a) - (c["route"] == b))
            sims.append(d / mins)
        return np.percentile(sims, 2.5), np.percentile(sims, 97.5)

    sets = [("10 episodes, production / MOSS Pro / MAI2", idents, ("prod", "mosspro", "mai2")),
            ("episodes with 35t", [e for e in idents if "g35t" in src[e]], R.ROUTES)]
    for name, eps, routes in sets:
        r = rates(cells, eps)
        mins = sum(minutes[e] for e in eps)
        print(f"\n{name}: {len(eps)} episodes, {mins:.0f} min; misses per minute, weighted to every R1 candidate")
        for route in routes:
            k = r.get(route, {})
            mine = [c for c in cells if c["ident"] in eps and c["route"] == route]
            listed = [c for c in mine if c["listed"]]
            n_listed = sum(c["w"] for c in listed)
            p = sum(c["w"] for c in listed if c["kind"] == "内容") / max(n_listed, 1e-9)
            sens, fp_lo, fp_hi, sens_lower, fp_upper = acc[route_type(route)]
            lo, hi = (corrected(p, sens, fp) * n_listed / mins for fp in (fp_hi, fp_lo))
            # every miss on every judged cell, with the judge's error at its interval edges: fp from 0 to its upper
            # bound, sensitivity from its lower bound to the estimate
            n_mine = sum(c["w"] for c in mine)
            p_all = sum(c["w"] for c in mine if c["kind"]) / max(n_mine, 1e-9)
            edges = [corrected(p_all, s, fp) * n_mine / mins for s in (sens_lower, sens) for fp in (0.0, fp_upper)]
            unlisted = sum(c["w"] for c in mine if not c["listed"] and c["kind"] == "内容") / mins
            pi = corrected(p, sens, fp_hi)  # the true share on listed cells, conservative
            ppv = sens * pi / (sens * pi + fp_hi * (1 - pi)) if pi == pi and pi > 0 else float("nan")
            incidental = sum(c["incidental"] for c in mine)
            print(f"  {route:8s} all {sum(k.values()):.2f} (judge error at its bounds {min(edges):.2f}-{max(edges):.2f}), content {k.get('内容', 0):.2f} (corrected on listed cells {lo:.2f}-{hi:.2f}; of it not listed by R1"
                  f" {unlisted:.2f}), short reactions {k.get('短反应', 0):.2f}, fillers {k.get('填充', 0):.2f};"
                  f" incidental misses set aside {incidental}; judged misses that are real (highest fp) {ppv:.0%}")
        for a in routes:
            if a != "prod":
                lo, hi = boot(eps, a, "prod")
                d = r.get(a, {}).get("内容", 0) - r.get("prod", {}).get("内容", 0)
                lo2, hi2 = boot(eps, a, "prod", None)
                d2 = sum(r.get(a, {}).values()) - sum(r.get("prod", {}).values())
                print(f"  {a} − prod: content {d:+.2f}/min [{lo:+.2f}, {hi:+.2f}], all {d2:+.2f}/min [{lo2:+.2f}, {hi2:+.2f}]")
    # G6: on the fansub episodes, pairs of routes whose difference is clear in both readings must agree in sign
    fan = [e for e in idents if e in R.FANSUB_JA]
    if fan:
        c_runs = {e: R.fansub_recall(e)["routes"] for e in fan}
        print(f"\nG6 on {len(fan)} fansub episodes: route − prod, judge content misses/min and fansub runs/min:")
        mins = sum(minutes[e] for e in fan)
        r = rates(cells, fan)
        for a in ("mosspro", "mai2"):
            j_lo, j_hi = boot(fan, a, "prod")
            jd = r.get(a, {}).get("内容", 0) - r.get("prod", {}).get("内容", 0)
            d = np.concatenate([c_runs[e][a]["blocks"] - c_runs[e]["prod"]["blocks"] for e in fan])
            sims = [d[rng.integers(0, len(d), len(d))].sum() / mins for _ in range(reps)]
            c_lo, c_hi = np.percentile(sims, 2.5), np.percentile(sims, 97.5)
            clear_j, clear_c = j_lo > 0 or j_hi < 0, c_lo > 0 or c_hi < 0
            verdict = ("agree" if np.sign(jd) == np.sign(d.sum()) else "DISAGREE") if clear_j and clear_c else "not separable"
            print(f"  {a}: judge {jd:+.2f} [{j_lo:+.2f}, {j_hi:+.2f}], fansub {d.sum() / mins:+.2f} [{c_lo:+.2f}, {c_hi:+.2f}] -> {verdict}")
    print("\nby how many routes wrote it (not judged a miss), content misses per route (clips, unweighted):")
    real_rows = [(it, v, a) for it, v, a in rows if it["kind"] == "real"]
    for k in range(0, len(R.ROUTES) + 1):
        group = [(it, v, a) for it, v, a in real_rows if sum(x[0] not in MISS for x in v.values()) == k]
        if group:
            print(f"  {k} wrote it: {len(group)} clips; " + ", ".join(
                f"{r} {sum(v[r][0] in MISS and route_kind(*v[r], a['kind']) == '内容' for _, v, a in group if r in v)}"
                for r in R.ROUTES))
    longs = [(it, v, a) for it, v, a in rows if it["kind"] == "long"]
    if longs:
        print("\nlong gaps: seconds per minute where the route wrote nothing and the judge heard 正片 there (gap length x "
              "share of its points; by episode set as above)")
        gaps = {}
        for it, v, a in longs:
            g = gaps.setdefault((it["ident"], tuple(it["gap"])), {"routes": it["gap_routes"], "points": []})
            g["points"].append((v, a))
        for name, eps, routes in sets:
            mins = sum(minutes[e] for e in eps)
            line = []
            for route in routes:
                sec = 0.0
                for (ident, (a, b)), g in gaps.items():
                    if ident in eps and route in g["routes"]:
                        share = np.mean([a_["speech"] == "正片" and v_.get(route, ("",))[0] in MISS for v_, a_ in g["points"]])
                        sec += (b - a) * share
                line.append(f"{route} {sec / mins:.2f} s/min")
            print(f"  {name}: " + ", ".join(line))


DENSE = ("hina-14", "hina-25", "hina-27", "onetonton-01", "onetonton-05")  # talk shows thick with interjections


def cross(tag, real=50, reps=2000, seed=3):
    """A text filter on the judge's misses, against the judge following one detailed text: a judged miss counts only
    when a quoted piece of >= 3 kana is in both other routes' text within CTX of the clip (one contiguous block of
    >= 80% of it) and not in the route's own. It sees only what the judge called a miss. MOSS
    Pro against production, the other two being MAI2 and the third; misses per minute, weighted to every candidate,
    intervals from resampled clips."""
    rows = verdicts(tag)
    folder, root = home(tag)
    batch = json.loads((folder / "batch.json").read_text(encoding="utf-8"))
    prob = inclusion(batch, root, real)
    idents = sorted({it["ident"] for it in batch})
    found = {i: found_of(root, i)["items"] for i in idents}
    src = {i: R.sources(i) for i in idents}
    minutes = {i: max(u["end"] for s in src[i].values() for u in s["units"]) / 60 for i in idents}

    def has(route, it, q):
        a, b = it["clip"][0] - CTX, it["clip"][1] + CTX
        text = R.hira("".join(u.get("text", "") for u in src[it["ident"]][route]["units"] if u["end"] > a and u["start"] < b))
        longest = difflib.SequenceMatcher(None, q, text, autojunk=False).find_longest_match(0, len(q), 0, len(text)).size
        return longest >= 0.8 * len(q)  # one contiguous block: scattered kana would favour long (MOSS) segment text

    rng = np.random.default_rng(seed)
    for name, eps in (("10 episodes", idents), ("talk-dense 5", [e for e in idents if e in DENSE]),
                      ("other 5", [e for e in idents if e not in DENSE])):
        mins = sum(minutes[e] for e in eps)
        per = {}
        for route, others in (("prod", ("mai2", "mosspro")), ("mosspro", ("mai2", "prod"))):
            vals = []
            for it, v, _ in rows:
                if it["kind"] != "real" or it["ident"] not in eps:
                    continue
                verdict, note = v[route]
                strong = False
                if verdict in MISS and attributed(route, verdict, note, it, candidate_of(it, found[it["ident"]]), src[it["ident"]]):
                    strong = any(all(has(o, it, q) for o in others) and not has(route, it, q)
                                 for q in quotes(note) if len(q) >= 3)
                vals.append(strong / prob[it["id"]])
            per[route] = np.array(vals)
        d = per["mosspro"] - per["prod"]
        sims = [d[rng.integers(0, len(d), len(d))].sum() / mins for _ in range(reps)]
        print(f"{name}: misses both other routes back, per minute: prod {per['prod'].sum() / mins:.2f}, MOSS Pro "
              f"{per['mosspro'].sum() / mins:.2f}; MOSS − prod {d.sum() / mins:+.2f} [{np.percentile(sims, 2.5):+.2f}, "
              f"{np.percentile(sims, 97.5):+.2f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "gates", "g1", "readout", "rerun", "g4", "cross"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--items", nargs="*", default=["hina-72", "hina-25"])
    ap.add_argument("--tag")
    ap.add_argument("--per-round", type=int, default=40)
    ap.add_argument("--real", type=int, default=20)
    ap.add_argument("--plants", type=int, default=8)
    ap.add_argument("--decoys", type=int, default=5)
    ap.add_argument("--long-points", type=int, default=2)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    R.ready()
    if args.cmd == "build":
        build(args.items, args.tag, args.real, args.plants, args.decoys, args.long_points, args.seed)
    elif args.cmd == "g1":
        g1(args.items, args.seed, args.per_round)
    elif args.cmd == "rerun":
        rerun(args.tag, args.n, args.seed)
    elif args.cmd == "g4":
        g4(args.tag)
    elif args.cmd == "cross":
        cross(args.tag)
    elif args.cmd == "readout":
        readout(args.tag)
    elif args.cmd == "run":
        run(args.tag, args.parallel)
    else:
        gates(args.tag)


if __name__ == "__main__":
    main()
