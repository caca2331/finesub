"""Row truth scoring (the closed plan's 「计分」, `git show 423458df:docs/plans/asr-row-truth-plan.md`): candidates against the frozen row truth.

Candidates are production's outputs as production writes them (`data/diarization/row-truth/products/`): the acoustic
layer's `spk` cells, and the speaker judge's (Sol, Luna), where a moved row leads its new track alone and every other
row keeps its acoustic cell. Their tracks are anonymous, so each window maps labels to people:

1. rule: every label goes to the person with the most of its rows in the window (many-to-one; ties to the name that
   sorts last, as `window_partition_score.match`); a person's main label is the one carrying most of their rows, every
   other label mapped to them is an extra one and pays the merge price;
2. review (`review` command): an LLM reads the truth, the labels and the rule's mapping and may remap a label (or map
   it to nobody); `--mapping review` scores with those changes (pass 1, scoring windows only; the 其他 category is
   listed, not applied). Two passes on the same input agreed on 3 of 6 changes on the dev set, under the plan's 80%,
   so the primary readout is the rule's and the review's is descriptive.

Readouts, per candidate, on the rows the ruler marks main (`one` / `multi` with a person), side rows apart:
row accuracy (the label's person is the row's person; for `multi` the part with the most characters) with its split
into rows on main and on extra labels, the main-labels-only reading, the wide reading (any person present), and edit
cost per 100 rows at the owner's prices plus the merge price. A candidate minus the acoustic layer is paired row by
row; intervals resample clusters, the same for every candidate: production's (window, lead acoustic track).

    python tools/diarization_explore/row_truth_score.py readout [--scope score|all] [--mapping rule|review] [--windows]
    python tools/diarization_explore/row_truth_score.py bridge        # old ruler + one-to-one / + new rule / row truth
    python tools/diarization_explore/row_truth_score.py review [--only hina-24] [--candidates sol] [--rep 1]   # agy calls
    python tools/diarization_explore/row_truth_score.py show --only hina-24 --candidates sol                  # the prompt
    python tools/diarization_explore/row_truth_score.py consistency   # review pass 1 vs 2 on the windows run twice
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "examples"))
import row_truth as rt  # noqa: E402
import spk_cell  # noqa: E402

r9 = rt.r9
DATA = r9.MAIN / "data" / "diarization" / "row-truth"
PRODUCTS = DATA / "products"
RULERS = ("dev-v2-high", "heldout-v3-high")
DEV = ("manosaba-b", "hina-71", "hina-24", "hina-17", "hina-57")
CANDIDATES = ("acoustic", "sol", "luna")
REVIEW = rt.OUT / "review"
SEED = 7
REPS = 5000
#: the owner's prices (findings「编辑代价」) and the merge price (plan 已裁定)
MOVE, DELETE, ADD, MERGE, ASR_MERGE = 2, 1, 2, 2, 3
NOBODY = "无人"
EVERYONE = "全员"


# --- truth --------------------------------------------------------------------------------------------------

@dataclass
class Row:
    """One ASR row of the truth: `present` the people speaking in it (`全员` dropped), `person` the main-口径 one."""
    i: int
    cls: str
    prio: str
    src: str
    present: frozenset
    person: str | None
    text: str


def person_of(row, text):
    """`one`: its person; `multi`: the person of the part with the most characters, a part `A+全员` counting as A;
    None for one part, a tie, or a largest part naming several people."""
    if row["cls"] == "one":
        return row["who"][0]
    if row["cls"] != "multi" or len(row["parts"] or []) < 2:
        return None
    starts, cursor = [], 0
    for part in row["parts"]:
        cursor = text.find(part["from"], cursor)
        starts.append(cursor)
        cursor += 1
    sizes = [end - start for start, end in zip(starts, [*starts[1:], len(text)])]
    best = max(sizes)
    if sizes.count(best) > 1:
        return None
    who = [n for n in row["parts"][sizes.index(best)]["who"] if n != EVERYONE]
    return who[0] if len(who) == 1 else None


_rulers = {}


def rulers():
    """{correction window key: {row id: row}} from the frozen rulers (stored rows, never re-parsed)."""
    if _rulers:
        return _rulers
    out = _rulers
    for folder in RULERS:
        for path in sorted((DATA / folder).glob("*.json")):
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return out


def stable(ident, candidate="acoustic"):
    return json.loads((PRODUCTS / candidate / ident / f"{ident}-stable.json").read_text(encoding="utf-8"))["segments"]


def truth_rows(ident):
    """{row: Row} over every core row the rulers judged in this file."""
    found = rulers()
    segments = stable(ident)
    out = {}
    for key, kid, lo, hi, source in rt.windows():
        if kid != ident or source != "accept":
            continue
        for i in range(lo, hi):
            row = found[key][f"q{i}"]
            text = str(segments[i].get("text", "")).strip()
            present = frozenset(n for n in row["who"] if n != EVERYONE) if row["cls"] in ("one", "multi") else frozenset()
            person = person_of(row, text) if row["cls"] in ("one", "multi") else None
            out[i] = Row(i, row["cls"], row["prio"], row["src"], present, person, text)
    return out


# --- candidates -----------------------------------------------------------------------------------------------

def cell_labels(cell):
    """The labels of a `spk` cell in order, after the judge when it has a `<`: `s5+s4` -> [s5, s4];
    `s3?(s5)` -> [s3] (the alternative is no label); `s4*` -> [s4]; `s1~+s0` -> [s1, s0]; "" -> []."""
    return spk_cell.parse(cell or "")["final"]["tracks"]


def candidate_cells(ident, candidate):
    """{row: [labels]} as production writes them for this candidate."""
    from finesub.speech.diarization.judgments import segment_ids, sidecar_path
    from finesub.speech.diarization.labels import speaker_cell, speaker_of
    acoustic = {i: cell_labels(speaker_cell([speaker_of(s)])) for i, s in enumerate(stable(ident))}
    if candidate == "acoustic":
        return acoustic
    path = PRODUCTS / candidate / ident / f"{ident}-stable.json"
    segments = stable(ident, candidate)
    index = {key: i for i, key in enumerate(segment_ids(segments))}
    out = dict(acoustic)
    payload = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
    for window in payload["windows"]:
        for unit in window.get("parts") or [window]:
            for result in unit.get("results") or []:
                i = index[str(result["row"])]
                if "track" in result:
                    out[i] = [f"s{result['track']}"]
                elif "label" in result:
                    out[i] = [f"@{unit.get('id')}/{result['label']}"]
                elif "carried" in result:
                    out[i] = [f"@{result['carried']['window']}/{result['carried']['label']}"]
    return out


# --- windows --------------------------------------------------------------------------------------------------

@dataclass
class Window:
    key: str            # `hina-24` (scoring window) or `hina-24-c0002` (correction window)
    ident: str
    rows: list
    group: str          # round 9's group
    dev: bool
    truth: dict = field(default_factory=dict)


def windows(scope):
    """The scoring windows (`score`, round 9's 1800 rows) or the correction windows the rulers judged (`all`)."""
    plan = {w["ident"]: w for w in r9.plan()}
    out = []
    for ident, w in plan.items():
        truth = truth_rows(ident)
        if scope == "score":
            out.append(Window(ident, ident, list(range(w["a"], w["b"])), w["group"], ident in DEV, truth))
            continue
        for key, kid, lo, hi, source in rt.windows():
            if kid == ident and source == "accept":
                out.append(Window(key, ident, list(range(lo, hi)), w["group"], ident in DEV, truth))
    return out


# --- mapping and scoring ----------------------------------------------------------------------------------------

def fit(window, cells):
    """{label: person} by the rule: each lead label to the person with the most of its determinable rows."""
    weight = defaultdict(Counter)
    for i in window.rows:
        row = window.truth[i]
        if row.person and cells.get(i):
            weight[cells[i][0]][row.person] += 1
    return {label: max(c.items(), key=lambda kv: (kv[1], kv[0]))[0] for label, c in weight.items()}


def main_labels(window, cells, mapping):
    """{person: their main label}: of the labels mapped to them, the one leading most of that person's rows."""
    count = Counter((cells[i][0], window.truth[i].person) for i in window.rows if cells.get(i) and window.truth[i].person)
    out = {}
    for label, person in mapping.items():
        if person == NOBODY:
            continue
        if person not in out or (count[label, person], label) > (count[out[person], person], out[person]):
            out[person] = label
    return out


def row_cost(row, labels, mapping):
    """Edit cost of one scored row: lead on someone absent = move (or delete when a second label already covers the
    rest: `delete lead`), a second label that does not add a present person = `delete second`, every present person
    left uncovered = add.
    -> Counter of the parts."""
    cost = Counter()
    if not labels:
        cost["add"] = ADD * len(row.present)
        return cost
    people = [mapping.get(label) for label in labels]
    covered = set()
    for person in people[1:]:
        if person in row.present and person != people[0] and person not in covered:
            covered.add(person)
        else:
            cost["delete second"] += DELETE
    if people[0] in row.present:
        covered.add(people[0])
    uncovered = len(row.present - covered)
    if people[0] not in row.present:
        if uncovered:
            cost["move"] += MOVE
            uncovered -= 1
        else:
            cost["delete lead"] += DELETE
    cost["add"] += ADD * uncovered
    return cost


def score_window(window, cells, mapping):
    """{row: verdict} for the window's scored rows and the merge price per extra label."""
    mains = main_labels(window, cells, mapping)
    verdicts = {}
    for i in window.rows:
        row = window.truth[i]
        if row.cls not in ("one", "multi") or not row.present:
            continue
        labels = cells.get(i) or []
        lead = labels[0] if labels else None
        person = mapping.get(lead) if lead else None
        verdicts[i] = dict(
            right=row.person is not None and person == row.person,
            wide=person in row.present,
            on_main=lead is not None and mains.get(person) == lead,
            cost=row_cost(row, labels, mapping),
            determinable=row.person is not None)
    extra = {label: MERGE for label, person in mapping.items() if person != NOBODY and mains.get(person) != label}
    return verdicts, extra


def review_changes(candidate, window_key, rep=1):
    path = REVIEW / f"r{rep}" / candidate / f"{window_key}.json"
    if not path.exists():
        return {}
    return {c["label"]: c["person"] for c in json.loads(path.read_text(encoding="utf-8")).get("changes", [])
            if c.get("counted", True)}


def readout(scope, mapping_kind, per_window):
    rng = np.random.default_rng(SEED)
    found = windows(scope)
    records = []   # one per scored row: window, row, cluster, prio, src, {candidate: verdict}
    merges = defaultdict(lambda: defaultdict(float))   # candidate -> cluster -> merge price
    unscored = defaultdict(Counter)
    skipped = []   # the window of every row not scored (background, filler, hallucination, unsure, `全员` only)
    for w in found:
        acoustic = candidate_cells(w.ident, "acoustic")
        verdicts = {}
        for candidate in CANDIDATES:
            cells = acoustic if candidate == "acoustic" else candidate_cells(w.ident, candidate)
            mapping = fit(w, cells)
            if mapping_kind == "review":
                mapping.update(review_changes(candidate, w.key))
            verdicts[candidate], extra = score_window(w, cells, mapping)
            # the merge price goes to the cluster where the extra label leads most rows
            for label, price in extra.items():
                where = Counter((w.key, acoustic[i][0] if acoustic.get(i) else None)
                                for i in w.rows if cells.get(i) and cells[i][0] == label)
                if where:
                    merges[candidate][where.most_common(1)[0][0]] += price
        for i in w.rows:
            row = w.truth[i]
            if i in verdicts["acoustic"]:
                records.append(dict(window=w, i=i, cluster=(w.key, acoustic[i][0] if acoustic.get(i) else None),
                                    prio=row.prio, src=row.src, v={c: verdicts[c][i] for c in CANDIDATES},
                                    short=len(re.sub(r"[\W_]", "", row.text)) <= r9.BACKCHANNEL))
            else:
                kind = "multi 全员" if row.cls == "multi" else row.cls
                unscored[kind]["acoustic labelled" if acoustic.get(i) else "acoustic none"] += 1
                skipped.append(w)

    def table(name, keep):
        chosen = [r for r in records if keep(r)]
        main = [r for r in chosen if r["prio"] == "main" and r["v"]["acoustic"]["determinable"]]
        if not main:
            return
        clusters = sorted({r["cluster"] for r in chosen}, key=str)
        keys = {r["window"].key for r in chosen}
        undetermined = sum(r["prio"] == "main" and not r["v"]["acoustic"]["determinable"] for r in chosen)
        out_of = len(main) + undetermined + sum(w.key in keys for w in skipped)
        print(f"\n{name}: main rows {len(main)}, side rows {sum(r['prio'] == 'side' for r in chosen)}, "
              f"rows not scored {out_of - len(main) - undetermined}, main multi rows without a main person {undetermined}")
        for candidate in CANDIDATES:
            right = np.mean([r["v"][candidate]["right"] for r in main])
            on_main = np.mean([r["v"][candidate]["right"] and r["v"][candidate]["on_main"] for r in main])
            wide = np.mean([r["v"][candidate]["wide"] for r in chosen if r["prio"] == "main"])
            cost_rows = [r for r in chosen if r["prio"] == "main"]
            merge = sum(merges[candidate].get(c, 0.0) for c in clusters)
            parts = sum((r["v"][candidate]["cost"] for r in cost_rows), Counter())
            cost = 100 * (sum(parts.values()) + merge) / len(cost_rows)
            side = [r for r in chosen if r["prio"] == "side" and r["v"][candidate]["determinable"]]
            line = (f"  {candidate:8s} right {right:.1%} (main labels only {on_main:.1%}, on extra labels {right - on_main:.1%}), "
                    f"wide {wide:.1%}, edit cost {cost:.1f}/100 rows ("
                    + ", ".join(f"{k} {100 * parts[k] / len(cost_rows):.1f}" for k in ("move", "delete lead", "delete second", "add"))
                    + f", merge {100 * merge / len(cost_rows):.1f})"
                    + (f", side right {np.mean([r['v'][candidate]['right'] for r in side]):.1%}" if side else "")
                    + f", lower bound {sum(r['v'][candidate]['right'] for r in main) / out_of:.1%}")
            if candidate != "acoustic":
                d_right = paired(main, candidate, "right", rng)
                d_cost = paired_cost(cost_rows, candidate, merges, rng, clusters)
                # the truth names who speaks in a row, never who only made a sound in it, so a second label on such a
                # person is priced as a delete it may not deserve: the difference without the delete part as well
                d_kept = paired_cost(cost_rows, candidate, merges, rng, clusters, skip=("delete second",))
                fixed = sum(r["v"][candidate]["right"] and not r["v"]["acoustic"]["right"] for r in main)
                broke = sum(r["v"]["acoustic"]["right"] and not r["v"][candidate]["right"] for r in main)
                line += (f"\n           vs acoustic: right {d_right[0]:+.1f} [{d_right[1]:+.1f}, {d_right[2]:+.1f}] "
                         f"(fixed {fixed}, broke {broke}); edit cost {d_cost[0]:+.1f} [{d_cost[1]:+.1f}, {d_cost[2]:+.1f}], "
                         f"without second-label deletes {d_kept[0]:+.1f} [{d_kept[1]:+.1f}, {d_kept[2]:+.1f}]")
            print(line)
        # the bridge to earlier readouts: rows over production's back-channel length, whatever their main / side
        long = [r for r in chosen if not r["short"] and r["v"]["acoustic"]["determinable"]]
        print(f"  not back-channel (> {r9.BACKCHANNEL} chars, {len(long)} rows): " + ", ".join(
            f"{c} {np.mean([r['v'][c]['right'] for r in long]):.1%}" for c in CANDIDATES))
        for src in ("ref", "infer"):
            sub = [r for r in main if r["src"] == src]
            if sub:
                print(f"  src {src} ({len(sub)} rows): " + ", ".join(
                    f"{c} {np.mean([r['v'][c]['right'] for r in sub]):.1%}" for c in CANDIDATES))

    def paired(rows, candidate, key, rng):
        by = defaultdict(list)
        for r in rows:
            by[r["cluster"]].append(float(r["v"][candidate][key]) - float(r["v"]["acoustic"][key]))
        return bootstrap(by, rng)

    def paired_cost(rows, candidate, merges, rng, clusters, skip=()):
        """Edit cost difference per 100 rows; the merge price of every cluster of the group counts, a cluster with
        side rows only included (no rows, its price), as in the point estimate. `skip`: parts left out."""
        by = {c: [] for c in clusters}
        total = lambda cost: sum(v for k, v in cost.items() if k not in skip)  # noqa: E731
        for r in rows:
            by[r["cluster"]].append(total(r["v"][candidate]["cost"]) - total(r["v"]["acoustic"]["cost"]))
        extra = {c: merges[candidate].get(c, 0.0) - merges["acoustic"].get(c, 0.0) for c in by}
        return bootstrap(by, rng, extra)

    def bootstrap(by, rng, extra=None):
        keys = list(by)
        sums = np.array([sum(by[k]) + (extra or {}).get(k, 0.0) for k in keys])
        counts = np.array([len(by[k]) for k in keys])
        point = 100 * sums.sum() / counts.sum()
        draws = rng.integers(0, len(keys), (REPS, len(keys)))
        drawn = counts[draws].sum(1)
        sims = 100 * sums[draws].sum(1)[drawn > 0] / drawn[drawn > 0]   # a draw of row-less clusters only says nothing
        return point, np.percentile(sims, 2.5), np.percentile(sims, 97.5)

    merged = lambda r: r["window"].group not in r9.FRESH  # noqa: E731
    print(f"scope {scope} ({'round 9 scoring windows' if scope == 'score' else 'every judged correction window'}), "
          f"mapping {mapping_kind}")
    table("9 new windows (readouts 1 and 2)", lambda r: not merged(r))
    table("  dev, 3 new windows", lambda r: not merged(r) and r["window"].dev)
    table("  held-out, 6 windows", lambda r: not r["window"].dev)
    table("  hina normal, 8 windows", lambda r: r["window"].group == "hina 普通")
    table("merged, 2 windows (describe only)", merged)
    if per_window:
        for w in found:
            table(f"window {w.key} ({w.group})", lambda r, key=w.key: r["window"].key == key)
    print("\nnot scored (rows; what the acoustic layer gave):")
    for kind, c in sorted(unscored.items()):
        print(f"  {kind}: {sum(c.values())} ({dict(c)})")


def bridge():
    """Old ruler + one-to-one (round 9), old ruler + the new rule, row truth + the new rule, on the same rows: main
    rows of the scoring windows that both rulers give a person."""
    import window_partition_score as ps
    old = {w["ident"]: r9.load(w["ident"]) for w in r9.plan()}
    for w in windows("score"):
        m = old[w.ident]
        acoustic = candidate_cells(w.ident, "acoustic")
        rows = [i for i in w.rows if m.truth.get(i) and w.truth[i].person and w.truth[i].prio == "main"]
        out = []
        for candidate in CANDIDATES:
            cells = acoustic if candidate == "acoustic" else candidate_cells(w.ident, candidate)
            lead = {i: cells[i][0] if cells.get(i) else None for i in w.rows}
            old_rows = [i for i in w.rows if m.truth.get(i)]
            one = ps.match(old_rows, lead, m.truth, one_to_one=True)
            many = ps.match(old_rows, lead, m.truth, one_to_one=False)
            new = fit(w, cells)
            scores = [np.mean([one.get(lead[i]) in m.truth[i] for i in rows]),
                      np.mean([many.get(lead[i]) in m.truth[i] for i in rows]),
                      np.mean([new.get(lead[i]) == w.truth[i].person for i in rows])]
            out.append(f"{candidate} " + " / ".join(f"{s:.1%}" for s in scores))
        print(f"{w.key:11s} {len(rows):3d} rows: " + "; ".join(out))
    print("(each: old ruler + one-to-one / old ruler + many-to-one / row truth + many-to-one)")


# --- LLM review of the mapping -----------------------------------------------------------------------------------

CATEGORIES = ("平票", "时长", "无人", "其他")
COUNTED = ("平票", "时长", "无人")   # plan: the primary readout takes these; 其他 is listed, not applied


def truth_cell(row):
    side = "·次" if row.prio == "side" else ""
    if row.cls == "one":
        return f"one {next(iter(row.present))}{side}"
    if row.cls == "multi":
        if not row.present:
            return f"multi {EVERYONE}{side}"
        return f"multi {'+'.join(sorted(row.present))}（主 {row.person or '?'}）{side}"
    return row.cls


def review_call(window, candidate):
    """(system, user, labels, people) for one candidate on one window: the truth, the labels, the rule's mapping."""
    # a judge's new track is labelled with the name it gave (`@0002/井上麻里奈`): shown as a number, so the review sees
    # anonymous labels only (plan); `back` turns them into the real labels again
    real = candidate_cells(window.ident, candidate)
    named = list(dict.fromkeys(label for i in window.rows for label in (real.get(i) or []) if label.startswith("@")))
    alias = {label: f"@n{k}" for k, label in enumerate(named, 1)}
    back = {v: k for k, v in alias.items()}
    cells = {i: [alias.get(label, label) for label in labels] for i, labels in real.items()}
    mapping = fit(window, cells)
    verdicts, _ = score_window(window, cells, mapping)
    segments = stable(window.ident)
    origin = float(segments[window.rows[0]]["start"])
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    tally = defaultdict(Counter)
    idle = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))   # label -> truth class -> [rows, seconds] not scored
    for i in window.rows:
        row, labels = window.truth[i], cells.get(i) or []
        if labels and row.person:
            stat = stats[labels[0]][row.person]
            stat[0] += 1
            stat[1] += float(segments[i]["end"]) - float(segments[i]["start"])
        if labels and i in verdicts and verdicts[i]["determinable"]:
            tally[labels[0]]["对" if verdicts[i]["right"] else "错"] += 1
        elif labels:
            stat = idle[labels[0]][row.cls if i not in verdicts else "multi 定不出主"]
            stat[0] += 1
            stat[1] += float(segments[i]["end"]) - float(segments[i]["start"])
    seen = list(dict.fromkeys(label for i in window.rows for label in (cells.get(i) or [])))
    lines = []
    for label in seen:
        people = sorted(stats[label].items(), key=lambda kv: (-kv[1][0], kv[0]))
        spread = "；".join(f"{name} {n} 行 {s:.1f} 秒" for name, (n, s) in people) or "没有计分行"
        lines.append(f"{label}|{mapping.get(label, '（没对上：它没当过计分行的第一个标签）')}|{spread}|"
                     f"对 {tally[label]['对']} / 错 {tally[label]['错']}|"
                     + ("；".join(f"{kind} {n} 行 {s:.1f} 秒" for kind, (n, s) in sorted(idle[label].items())) or "无"))
    rows = []
    for i in window.rows:
        row = window.truth[i]
        verdict = "不计" if i not in verdicts or not verdicts[i]["determinable"] else "对" if verdicts[i]["right"] else "错"
        rows.append(f"q{i}|{float(segments[i]['start']) - origin:.1f}–{float(segments[i]['end']) - origin:.1f}|"
                    f"{'+'.join(cells.get(i) or [])}|{row.text}|{truth_cell(row)}|{verdict}")
    people = sorted({p for i in window.rows for p in window.truth[i].present})
    system = rt.lr.template("row_truth_review_system.md").substitute()
    user = rt.lr.template("row_truth_review_user.md").substitute(
        title=rt.MATERIALS[rt.kind(window.ident)]["title"], mapping="\n".join(lines), rows="\n".join(rows),
        count=len(rows), labels=len(seen))
    return system, user, list(mapping), people, back


def parse_review(text, labels, people):
    """{"changes": [{label, person, category, reason, counted}], "doubts": [{id, reason}]} or None when malformed."""
    if not text or text.lstrip().lower().startswith("thought") or "<changes>" not in text or "<doubts>" not in text:
        return None
    changes, done = [], set()
    for line in rt.block(text, "changes"):
        cells = [c.strip() for c in line.split("|", 3)]
        if len(cells) < 3 or cells[0] in done or cells[2] not in CATEGORIES \
                or (cells[1] != NOBODY and cells[1] not in people):
            return None
        if cells[0] not in labels:   # a label the rule never mapped (shown, but it leads no scored row): kept, not applied
            cells[2] = "其他"
        done.add(cells[0])
        changes.append({"label": cells[0], "person": cells[1], "category": cells[2],
                        "reason": cells[3] if len(cells) > 3 else "", "counted": cells[2] in COUNTED})
    doubts = [{"id": c[0].strip(), "reason": c[1].strip() if len(c) > 1 else ""}
              for c in (line.split("|", 1) for line in rt.block(text, "doubts"))]
    return {"changes": changes, "doubts": doubts}


def review(candidates, only, rep, effort):
    for window in windows("score"):
        if only and window.key not in only:
            continue
        for candidate in candidates:
            path = REVIEW / f"r{rep}" / candidate / f"{window.key}.json"
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            system, user, labels, people, back = review_call(window, candidate)
            workdir = REVIEW / f"r{rep}" / "work" / f"{candidate}-{window.key}"
            workdir.mkdir(parents=True, exist_ok=True)
            answer, text, attempts = None, "", []
            for attempt in range(4):
                text, usage, conversation, status, error = rt.agy(system + "\n\n=====\n\n" + user, workdir, effort)
                attempts.append({"usage": usage, "conversation": conversation, "status": status, "error": error})
                answer = parse_review(text, labels, people) if status == "SUCCESS" else None
                if answer is None:
                    attempts[-1]["text"] = text[:3000]   # what came back, for auditing the malformed ones
                if answer is None:
                    kept = rt.last_reply(conversation)
                    answer = parse_review(kept, labels, people)
                    if answer is not None:
                        text, attempts[-1]["from_transcript"] = kept, True
                if answer is not None:
                    break
                print(f"{candidate} {window.key}: {'agy ' + status + ' ' + error[:120] if status != 'SUCCESS' else 'malformed reply'}; retry",
                      flush=True)
            for change in (answer or {}).get("changes", []):
                change["shown"], change["label"] = change["label"], back.get(change["label"], change["label"])
            payload = {**(answer or {}), "aliases": back, "raw": text, "model": f"{rt.MODEL}-{effort}", "attempts": attempts}
            target = path if answer else path.with_suffix(".failed.json")
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{candidate} {window.key}: {'ok' if answer else 'GAVE UP'}, "
                  f"{len((answer or {}).get('changes', []))} changes after {len(attempts)} replies", flush=True)


def consistency(first=1, second=2):
    """Mapping changes of two review passes on the same input: agreement over the union of (candidate, window, label)
    changed in either pass, counting the changes the readout applies (平票, 时长, 无人; 其他 is listed, not applied);
    a pass that names the rule's own person still counts as a change (plan: below 80% the primary readout falls back
    to the rule's mapping)."""
    agree = total = 0
    for path in sorted((REVIEW / f"r{second}").glob("*/*.json")):
        if path.name.endswith(".failed.json"):
            continue
        other = REVIEW / f"r{first}" / path.parent.name / path.name
        if not other.exists():
            continue
        a, b = (review_changes(path.parent.name, path.stem, rep) for rep in (first, second))
        union = set(a) | set(b)
        same = sum(a.get(label) == b.get(label) for label in union)
        agree, total = agree + same, total + len(union)
        print(f"{path.parent.name:8s} {path.stem:11s} r{first} {a} r{second} {b}: {same}/{len(union)}")
    print(f"agreement {agree}/{total}" + (f" = {agree / total:.0%}" if total else " (no changes in either pass)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("readout", "bridge", "review", "show", "consistency"))
    parser.add_argument("--only", nargs="*", help="scoring windows (idents) for review / show")
    parser.add_argument("--candidates", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--effort", default="high", choices=("low", "medium", "high"))
    parser.add_argument("--scope", choices=("score", "all"), default="score")
    parser.add_argument("--mapping", choices=("rule", "review"), default="rule")
    parser.add_argument("--windows", action="store_true", help="also one table per window")
    args = parser.parse_args()
    rt.ready()
    if args.command == "readout":
        if args.scope == "all" and args.mapping == "review":
            parser.error("the review ran on the scoring windows only: --mapping review needs --scope score")
        readout(args.scope, args.mapping, args.windows)
    elif args.command == "bridge":
        bridge()
    elif args.command == "consistency":
        consistency()
    elif args.command == "review":
        review(args.candidates, args.only, args.rep, args.effort)
    else:  # the review prompt of one window, no call
        window = next(w for w in windows("score") if w.key in args.only)
        system, user, *_ = review_call(window, args.candidates[0])
        print(system + "\n\n=====\n\n" + user)


if __name__ == "__main__":
    main()
