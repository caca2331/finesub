"""LLM with limited authority over speaker tracks (owner 2026-10-04): P1 fill, P3 mixed-track check.

The past designs that let a model decide speakers asked it to redo the acoustic layer's job, or to
write text and assign speakers in one breath (`docs/reports/2026-10-04-llm-speaker-prompts.md`).
These two give it a narrow question on top of production's own answer:

  fill   (P1) rows production left with NO label -- Nemotron-3 missed them. Choose one of the
               file's existing tracks, `other`, or `abstain`. Baseline to beat: the track of the
               nearest labelled row, which production could do without a model.
  mixed  (P3) is each large track ONE person? The acoustic signals for 串轨 both failed
               (findings, 中途换人); the evidence for it is in the words.

Arms are pinned to one target each and every reply's `target_id` is checked: `agy` =
`local-agy-gemini-3_8-flash`, `lite` = `gemini-free-3_5-flash-lite` (the reference arm).

    python tools/diarization_explore/llm_retrack.py build              # prompts + exact token count, no model
    python tools/diarization_explore/llm_retrack.py run --arm agy --task fill
    python tools/diarization_explore/llm_retrack.py score --task fill

Material: fill = the W2 files with a production Nemotron-3 timeline (`abstain_profile.py`'s cache:
姐姐咚咚 15, hina 16); mixed = the 22 files of `oversplit_sweep.py` (星穹铁道 7 with the survey's
reference, 姐姐咚咚 15), truth = row-level track purity (`crosstalk_signal.row_purity`).
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nemotron3_probe as n3  # noqa: E402

sys.path.insert(0, str(n3.WT / "src"))
sys.path.insert(0, str(n3.WT))

OUT = n3.MAIN / "tmp" / "diarization-retrack"
ABSTAIN = n3.MAIN / "tmp" / "diarization-abstain"
ARMS = {"agy": "local-agy-gemini-3_8-flash", "lite": "gemini-free-3_5-flash-lite"}
#: Neutral on purpose (roles_run.py): a real title can carry a guest's name.
TITLE = "一档多人对谈节目（标题略）"
CONTEXT = 5          # rows either side of a row to fill
PER_CALL = 20        # rows to fill per call
PROFILE_LINES = 6
MIXED_SHARE = 0.05   # tracks the mixed check asks about
BLOCKS, BLOCK_ROWS = 8, 40
MIXED_PURITY = 0.80  # a large track below this row-level purity counts as several people


def template(name):
    return string.Template((HERE / name).read_text(encoding="utf-8"))


def cell(speaker):
    tracks = speaker.to_dict().get("tracks") or []
    return "+".join(f"s{t}" for t in tracks) or "??"


def line(row, label):
    return f"{float(row['start']):.1f}|{label}|{str(row.get('text', '')).strip()}"


def owners(tracks, grid, people):
    """`{track id: majority person}` keyed by the timeline's OWN track ids.

    ⚠ `backend_cost.timeline_probs` renumbers tracks to columns 0..K-1 in sorted order, and
    `cost_metrics.owner_of_tracks` answers per column. Production ids often start at 1 (16 of the 40
    files here), so looking a track id up in the column-keyed dict read the NEIGHBOURING track's owner.
    Every P1 score before 2026-10-05 had this; this mapping is the fix.
    """
    import backend_cost as bc
    import cost_metrics as cm
    ids = sorted({int(t["track"]) for t in tracks})
    by_column = cm.owner_of_tracks(bc.timeline_probs(tracks, len(grid))[: len(grid)], grid, people)
    return {ids[column]: person for column, person in by_column.items()}


# --- material --------------------------------------------------------------------------------

def _w2(corpus, key):
    import backend_cost as bc
    import relabel_eval as ev
    if corpus == "hina":
        loaded = bc.hina_episode(key)
        if loaded is None:
            return None
        (stable, _, grid, people, _), _, starts = loaded
        return key, stable, grid, people, starts
    data = ev.episode(key)
    if data is None:
        return None
    stable, _, grid, people, _ = data
    return f"onetonton-{key:02d}", stable, grid, people, bc.refined_starts(key)


def _assign(stable, tracks):
    from finesub.speech.diarization.assign import assign_all
    from finesub.speech.diarization.segments import TrackSegment
    segments = [TrackSegment(track=int(t["track"]), start=float(t["start"]), end=float(t["end"])) for t in tracks]
    return assign_all(stable, segments)


def profiles(stable, speakers):
    seconds, rows = {}, {}
    for index, (row, speaker) in enumerate(zip(stable, speakers)):
        tracks = speaker.to_dict().get("tracks") or []
        if not tracks:
            continue
        seconds[tracks[0]] = seconds.get(tracks[0], 0.0) + float(row["end"]) - float(row["start"])
        if len(tracks) == 1 and len(re.sub(r"[\W_]", "", str(row.get("text", "")))) >= 8:
            rows.setdefault(tracks[0], []).append(index)
    total = sum(seconds.values()) or 1.0
    out = []
    for track in sorted(seconds, key=lambda k: -seconds[k]):
        picks = rows.get(track, [])
        picks = [picks[round(i * (len(picks) - 1) / max(1, PROFILE_LINES - 1))] for i in range(min(PROFILE_LINES, len(picks)))]
        out.append(f"- s{track}：占语音 {seconds[track] / total * 100:.0f}%。例句：" +
                   " / ".join(f"「{str(stable[i].get('text', '')).strip()}」" for i in dict.fromkeys(picks)))
    return "\n".join(out), sorted(seconds)


def fill_material():
    """[(ident, corpus, stable, speakers, tracks, grid, people, starts, items)] -- hina first."""
    import nemotron3_cost as nc
    out = []
    for corpus in ("hina", "onetonton"):  # ⚠ onetonton's corpus_keys rebinds refined_multi.sources
        for key in nc.corpus_keys(corpus):
            loaded = _w2(corpus, key)
            if loaded is None:
                continue
            ident, stable, grid, people, starts = loaded
            path = ABSTAIN / ident / "nemotron3-diar.json"
            if not path.exists():
                continue
            tracks = json.loads(path.read_text(encoding="utf-8"))["tracks"]
            speakers = _assign(stable, tracks)
            items = [i for i, s in enumerate(speakers) if not (s.to_dict().get("tracks") or [])]
            out.append((ident, corpus, stable, speakers, tracks, grid, people, starts, items))
    return out


def fill_prompts(ident, stable, speakers, items):
    profile_text, known = profiles(stable, speakers)
    cells = [cell(s) for s in speakers]
    calls = []
    for start in range(0, len(items), PER_CALL):
        chunk = items[start:start + PER_CALL]
        blocks = []
        for index in chunk:
            lo, hi = max(0, index - CONTEXT), min(len(stable), index + CONTEXT + 1)
            body = [("▶ " if j == index else "") + line(stable[j], cells[j]) for j in range(lo, hi)]
            blocks.append(f"### q{index}\n" + "\n".join(body))
        user = template("retrack_fill_user.md").substitute(
            title=TITLE, profiles=profile_text, items="\n\n".join(blocks), count=len(chunk))
        calls.append((f"{ident}-{start // PER_CALL}", [f"q{i}" for i in chunk], [f"s{k}" for k in known], user))
    return template("retrack_fill_system.md").template, calls


def mixed_material():
    import crosstalk_signal as cs
    import oversplit_sweep as osw
    out = []
    for ident in osw.DEVELOPMENT + osw.CONFIRMATION:
        if not (osw.OUT / f"{ident}.npz").exists():
            continue
        tracks = osw.timeline(ident, 0.6, 0.8)
        if ident.startswith("hina-"):
            stable = json.loads((n3.MAIN / "out" / "refined-2609" / ident / f"{ident}-stable.json")
                                .read_text(encoding="utf-8"))["segments"]
        else:
            import nemotron3_cost as nc
            import relabel_eval as ev
            nc.corpus_keys("onetonton")
            stable = ev.episode(int(ident[-2:]))[0]
        purity = {f"s{k[1:]}": v for k, v in cs.row_purity(ident, tracks).items()}
        out.append((ident, stable, _assign(stable, tracks), purity))
    return out


def mixed_prompt(ident, stable, speakers):
    cells = [cell(s) for s in speakers]
    seconds, first, last = {}, {}, {}
    for row, speaker in zip(stable, speakers):
        for k in speaker.to_dict().get("tracks") or []:
            seconds[k] = seconds.get(k, 0.0) + float(row["end"]) - float(row["start"])
            first.setdefault(k, float(row["start"]))
            last[k] = float(row["end"])
    total = sum(seconds.values()) or 1.0
    summary = "\n".join(f"- s{k}：占语音 {seconds[k] / total * 100:.0f}%，{first[k]:.0f}–{last[k]:.0f} 秒之间出现"
                        for k in sorted(seconds, key=lambda k: -seconds[k]))
    asked = [f"s{k}" for k in sorted(seconds) if seconds[k] >= MIXED_SHARE * total]
    starts = [round(b * (len(stable) - BLOCK_ROWS) / max(1, BLOCKS - 1)) for b in range(BLOCKS)]
    blocks = []
    for b, s in enumerate(dict.fromkeys(max(0, x) for x in starts)):
        blocks.append(f"### 第 {b + 1} 段\n" + "\n".join(line(stable[j], cells[j]) for j in range(s, min(len(stable), s + BLOCK_ROWS))))
    user = template("retrack_mixed_user.md").substitute(
        title=TITLE, summary=summary, asked="、".join(asked), blocks="\n\n".join(blocks))
    return template("retrack_mixed_system.md").template, user, asked


# --- calls -----------------------------------------------------------------------------------

def client_for(target):
    from dataclasses import replace

    from finesub.llm.client import RoleClient
    from finesub.llm.routing.config import LLMRole, role_config_for

    if target.startswith("local-agy"):
        base = role_config_for("research", preset_id="agy", role=LLMRole.GENERAL_CAPABLE)
    else:
        base = role_config_for("correction-text", "intermediate", preset_id="default", role=LLMRole.GENERAL_CAPABLE)
    chain = tuple(e for e in base.endpoint_chain if e.target_id == target)
    if not chain:
        raise SystemExit(f"{target} is not in the base chain")
    # ⚠ model_group_id="" is the pin: a group id makes the router ignore endpoint_chain (turn_llm_bed.py).
    client = RoleClient(role_configs={LLMRole.GENERAL_CAPABLE: replace(
        base, endpoint_chain=chain, test_endpoint=chain[0], model_group_id="")})

    def complete(system, user):
        reply = client.complete(LLMRole.GENERAL_CAPABLE,
                                [{"role": "system", "content": system}, {"role": "user", "content": user}])
        landed = str(getattr(reply, "target_id", ""))
        if landed != target:
            raise SystemExit(f"pinned to {target} but landed on {landed!r}; stopped after one call")
        return reply.content
    return complete


def parse(text, block, key, ids):
    match = re.search(rf"<{block}>(.*?)</{block}>", text or "", re.S)
    if not match:
        return None
    rows = {}
    for raw in match.group(1).splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        rows[str(payload.get(key, ""))] = payload
    return rows if set(rows) == set(ids) else None


def run(arm, task, repeat, shard=0, shards=1):
    complete = client_for(ARMS[arm])
    jobs = []
    if task == "fill":
        for ident, _, stable, speakers, *_, items in fill_material():
            system, calls = fill_prompts(ident, stable, speakers, items)
            jobs += [(key, system, user, ids, "fill", "id") for key, ids, _, user in calls]
    else:
        for ident, stable, speakers, _ in mixed_material():
            system, user, asked = mixed_prompt(ident, stable, speakers)
            jobs.append((ident, system, user, asked, "tracks", "track"))
    folder = OUT / f"{task}-{arm}-r{repeat}"
    folder.mkdir(parents=True, exist_ok=True)
    for key, system, user, ids, block, field in jobs[shard::shards]:
        path = folder / f"{key}.json"
        if path.exists():
            continue
        answer, text = None, ""
        for attempt in range(4):
            try:
                text = complete(system, user)
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{key}: {type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            # ⚠ agy's answer channel is malformed ~23% of the time (thought leakage, empty): retried, never scored.
            answer = parse(text, block, field, ids)
            if answer is not None:
                break
            print(f"{key}: malformed reply; retry", flush=True)
        path.write_text(json.dumps({"answer": answer, "raw": text}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: {'ok' if answer else 'GAVE UP'}", flush=True)


# --- build / score ---------------------------------------------------------------------------

def build():
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    total = {"fill": 0, "mixed": 0}
    folder = OUT / "prompts"
    folder.mkdir(parents=True, exist_ok=True)
    items = 0
    for ident, _, stable, speakers, *_, rows in fill_material():
        system, calls = fill_prompts(ident, stable, speakers, rows)
        items += len(rows)
        for key, _, _, user in calls:
            (folder / f"fill-{key}.md").write_text(system + "\n\n---\n\n" + user, encoding="utf-8")
            total["fill"] += count(system) + count(user)
    files = 0
    for ident, stable, speakers, _ in mixed_material():
        system, user, _ = mixed_prompt(ident, stable, speakers)
        (folder / f"mixed-{ident}.md").write_text(system + "\n\n---\n\n" + user, encoding="utf-8")
        total["mixed"] += count(system) + count(user)
        files += 1
    print(f"fill: {items} rows to fill, {total['fill']} input tokens; mixed: {files} files, {total['mixed']} input tokens; "
          f"text total {sum(total.values())}")


def score_fill(arm, repeat):
    import cost_metrics as cm
    import refined_multi as rm
    import relabel_eval as ev
    import backend_cost as bc
    from eval_contract import DEFAULT_POLICY

    folder = OUT / f"fill-{arm}-r{repeat}"
    tally = {k: 0 for k in ("items", "covered", "llm_filled", "llm_right", "llm_high", "llm_high_right",
                            "nb_right", "other", "single", "s_filled", "s_right", "s_nb_right",
                            "onset", "o_filled", "o_right", "o_high", "o_high_right", "o_nb_right")}
    costs = []
    for ident, corpus, stable, speakers, tracks, grid, people, starts, items in fill_material():
        answers = {}
        for path in sorted(folder.glob(f"{ident}-*.json")):
            answers.update(json.loads(path.read_text(encoding="utf-8"))["answer"] or {})
        owner = owners(tracks, grid, people)
        labelled = [{**row, "speaker": s.to_dict()} for row, s in zip(stable, speakers)]
        centres = np.array([(float(r["start"]) + float(r["end"])) / 2 for r in stable])
        with_tracks = [i for i, s in enumerate(speakers) if s.to_dict().get("tracks")]
        filled_all, filled_high = [dict(r) for r in labelled], [dict(r) for r in labelled]
        for index in items:
            tally["items"] += 1
            cover, _ = ev.span_cover(stable[index], grid)
            answer = answers.get(f"q{index}", {})
            track = str(answer.get("track", "abstain"))
            if track.startswith("s") and track[1:].isdigit():
                filled_all[index]["speaker"] = {"tracks": [int(track[1:])]}
                if answer.get("conf") == "high":
                    filled_high[index]["speaker"] = {"tracks": [int(track[1:])]}
            if not cover.any():
                continue
            tally["covered"] += 1
            person = people[int(cover.argmax())]
            # ⚠ With two people under the row, "the person covering most of it" need not be the one whose
            # words these are; rows with one reference person are the clean subset.
            single = int((cover > 0).sum()) == 1
            tally["single"] += single
            nearest = min(with_tracks, key=lambda j: abs(centres[j] - centres[index]))
            nb_right = owner.get(speakers[nearest].to_dict()["tracks"][0]) == person
            tally["nb_right"] += nb_right
            tally["s_nb_right"] += single and nb_right
            # ⚠ Fansub end times run late (findings 11.3), so a short interjection is often "covered" by the
            # previous speaker's stretched line. The person whose refined line STARTS inside the row is the
            # cleaner truth -- the same evidence W2 calls strong; used only when exactly one person starts there.
            starters = {w for a, w in starts if float(stable[index]["start"]) - 0.3 <= a < float(stable[index]["end"])}
            onset = next(iter(starters)) if len(starters) == 1 else None
            if onset is not None:
                tally["onset"] += 1
                tally["o_nb_right"] += owner.get(speakers[nearest].to_dict()["tracks"][0]) == onset
            tally["other"] += track == "other"
            if track.startswith("s") and track[1:].isdigit():
                right = owner.get(int(track[1:])) == person
                if onset is not None:
                    o_right = owner.get(int(track[1:])) == onset
                    tally["o_filled"] += 1
                    tally["o_right"] += o_right
                    if answer.get("conf") == "high":
                        tally["o_high"] += 1
                        tally["o_high_right"] += o_right
                tally["s_filled"] += single
                tally["s_right"] += single and right
                tally["llm_filled"] += 1
                tally["llm_right"] += right
                if answer.get("conf") == "high":
                    tally["llm_high"] += 1
                    tally["llm_high_right"] += right
        price = lambda rows: rm.strong_cost(rows, grid, people, owner, starts, DEFAULT_POLICY.add,  # noqa: E731
                                            merge=DEFAULT_POLICY.merge, price_unlabelled=True)[0]
        neighbour = [dict(r) for r in labelled]
        for index in items:
            nearest = min(with_tracks, key=lambda j: abs(centres[j] - centres[index]))
            neighbour[index]["speaker"] = {"tracks": [speakers[nearest].to_dict()["tracks"][0]]}
        # Descriptive (not the gate): the model where it answered, the neighbour where it abstained.
        hybrid = [dict(r) for r in neighbour]
        for index in items:
            if (filled_all[index]["speaker"].get("tracks") or []):
                hybrid[index]["speaker"] = filled_all[index]["speaker"]
        costs.append((price(labelled), price(filled_all), price(filled_high), price(neighbour), price(hybrid)))
    t = tally
    print(f"fill {arm} r{repeat}: {t['items']} rows, {t['covered']} with a reference person")
    print(f"  model filled {t['llm_filled']}/{t['covered']}, right {t['llm_right']} ({t['llm_right'] / max(1, t['llm_filled']):.0%}); "
          f"high {t['llm_high']}, right {t['llm_high_right']} ({t['llm_high_right'] / max(1, t['llm_high']):.0%}); other {t['other']}")
    print(f"  nearest-labelled-row baseline (fills every row): right {t['nb_right']}/{t['covered']} ({t['nb_right'] / max(1, t['covered']):.0%})")
    print(f"  rows where exactly one person's line STARTS ({t['onset']}): model right {t['o_right']}/{t['o_filled']} "
          f"({t['o_right'] / max(1, t['o_filled']):.0%}), high {t['o_high_right']}/{t['o_high']} "
          f"({t['o_high_right'] / max(1, t['o_high']):.0%}); baseline right {t['o_nb_right']}/{t['onset']} "
          f"({t['o_nb_right'] / max(1, t['onset']):.0%})")
    print(f"  rows with ONE reference person ({t['single']}): model right {t['s_right']}/{t['s_filled']} "
          f"({t['s_right'] / max(1, t['s_filled']):.0%}), baseline right {t['s_nb_right']}/{t['single']} "
          f"({t['s_nb_right'] / max(1, t['single']):.0%})")
    from accuracy_report import bootstrap_mean_ci
    base, every, high, nb, hyb = (np.array(c) for c in zip(*costs))
    hlo, hhi = bootstrap_mean_ci(list(hyb - nb))
    print(f"  descriptive: model where it answered + neighbour elsewhere, minus neighbour {np.mean(hyb - nb):+.2f} [{hlo:+.2f}, {hhi:+.2f}]")
    lo, hi = bootstrap_mean_ci(list(every - nb))
    print(f"  edit cost (abstentions priced), mean per file: production {base.mean():.1f}, + all fills {every.mean():.1f} "
          f"({(every - base).mean():+.1f}), + high fills {high.mean():.1f} ({(high - base).mean():+.1f}), "
          f"neighbour baseline {nb.mean():.1f} ({(nb - base).mean():+.1f})")
    print(f"  GATE: model (all fills) minus neighbour baseline {np.mean(every - nb):+.2f} [{lo:+.2f}, {hi:+.2f}] over {len(costs)} files")


def score_mixed(arm, repeat):
    folder = OUT / f"mixed-{arm}-r{repeat}"
    rows = []
    for ident, _, _, purity in mixed_material():
        path = folder / f"{ident}.json"
        answer = json.loads(path.read_text(encoding="utf-8"))["answer"] if path.exists() else None
        for track, value in sorted(purity.items()):
            said = (answer or {}).get(track, {})
            rows.append((ident, track, value, said.get("people", "-"), said.get("conf", "-")))
            print(f"  {ident:14}{track:5} purity {value:.2f}  -> {said.get('people', '-'):8} {said.get('conf', '-'):7} "
                  f"{str(said.get('why', ''))[:60]}")
    mixed = [r for r in rows if r[2] < MIXED_PURITY]
    clean = [r for r in rows if r[2] >= MIXED_PURITY]
    print(f"mixed {arm} r{repeat}: mixed tracks called several {sum(r[3] == 'several' for r in mixed)}/{len(mixed)}; "
          f"clean tracks called several {sum(r[3] == 'several' for r in clean)}/{len(clean)} "
          f"(high: {sum(r[3] == 'several' and r[4] == 'high' for r in clean)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("build", "run", "score"))
    parser.add_argument("--arm", choices=sorted(ARMS), default="agy")
    parser.add_argument("--task", choices=("fill", "mixed"), default="fill")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0, help="run jobs[shard::shards] (agy takes four at once)")
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "run":
        run(args.arm, args.task, args.repeat, args.shard, args.shards)
    elif args.task == "fill":
        score_fill(args.arm, args.repeat)
    else:
        score_mixed(args.arm, args.repeat)


if __name__ == "__main__":
    main()
