"""D1 rerun readout (plan 「D1 复跑（owner 2026-10-31 定）」), from the argon-judge worktree:

    python tools/diarization_explore/d1_readout.py

Per window group (held-out: hina-22 + hina-46 summed; merged: hina-71), each arm's three readings per run and their
medians: net (fix - wrong over the rows the judge moved, the pre-registered one), whole (rows right after the judge minus
rows right before, labels refit many-to-one over the window) and main (the same, a person counted only on her main
label); fusion (p9_gates textbase-incell) over every audio x text pair of the group. Then splits on the held-out
windows (row-split ruler 1, cut within +-1 char) and usage per run and window."""
import glob
import re
import statistics
import sys
from collections import Counter

sys.path.insert(0, "tools/diarization_explore")
import judge_moves_readout as M  # noqa: E402
import p9_gates as G  # noqa: E402
import row_split as R  # noqa: E402
import row_truth_score as S  # noqa: E402

H71 = ("hina-71-c0001",)
GROUPS = (("held-out", M.HELDOUT), ("hina-71", H71))
P7 = ["38F split r1", "38F split r2", "38F split r3", "38F bare r2", "38F bare r3", "38F bare r4"]
ARMS = {
    "held-out": dict(text=["38F text tv7a", "38F text d1a", "38F text d1b"], sol=["Sol text v7 r1", "Sol text v7 r2"],
                     audio=P7, p13=["38F p13gv8a r1", "38F p13gv8b r1"]),
    "hina-71": dict(text=["38F text d1a", "38F text d1b"], sol=["Sol text d1a", "Sol text d1b"],
                    audio=["38F split d1a", "38F split d1b"], p13=["38F p13gv8a r1", "38F p13gv8b r1"]),
}
ACCEPT = "C:/Users/Carl/Documents/Carl/projects/asr-playground/tmp/spk-merge/accept"
wins, results = M.load(None)
levels = {}


def net(c):
    return c["fix"] - c["wrong"]


def tally(win, payload, gate=None):
    """G.tally plus `whole`: rows right after the judge (labels refit over the window) minus rows right before --
    also counts rows that turn right or wrong because a moved neighbour changed the label's majority person."""
    rows, correct, _ = M.score(win, payload, gate=gate)
    c = Counter(r["kind"] for r in rows.values())
    cells = M.judged(payload, win.segs, win.acoustic, gate=gate)[0]
    return Counter(wrong=c["wrong"], cross=sum(r["obvious"] for r in rows.values()), fix=c["fix"],
                   whole=sum(correct.values()) - win.a_right, main=main_right(win, cells) - main_right(win, win.acoustic))


def main_right(win, cells):
    """Scored rows on their person's main label (the row-truth ruler's main-labels-only reading): a person split over
    two labels has the smaller one's rows counted wrong."""
    main = S.main_labels(win.window, cells, S.fit(win.window, cells))
    return sum(bool(cells.get(i)) and cells[i][0] == main.get(win.truth[i].person) for i in win.scored)


def group_tally(keys, payload_of, gate=None):
    tot = Counter()
    for key in keys:
        tot.update(tally(wins[key], payload_of(key), gate))
    return tot


def fusion(keys, audio, text):
    def fused(key):
        win = wins[key]
        lv = levels.setdefault(key, G.Levels(win))
        return G.textbase(results[audio, key][2], results[text, key][2], win, lv, True)
    return group_tally(keys, fused)


def show(label, tallies):
    nets, whole, main = [net(c) for c in tallies], [c["whole"] for c in tallies], [c["main"] for c in tallies]
    cells = " ".join(f"{n:+d}/{c['whole']:+d}/{c['main']:+d}(w{c['wrong']}/x{c['cross']}/f{c['fix']})" for n, c in zip(nets, tallies))
    print(f"  {label:<26} median {statistics.median(nets):+6.1f} whole {statistics.median(whole):+6.1f} main {statistics.median(main):+6.1f}  [{cells}]")
    return statistics.median(nets)


medians = {}
for group, keys in GROUPS:
    arms = ARMS[group]
    print(f"== {group} {keys}")
    for name in sum(arms.values(), []) + [p.replace(" r1", " draft") for p in arms["p13"]]:
        missing = [k for k in keys if (name, k) not in results]
        if missing:
            print(f"  MISSING {name} {missing}, left out")
    arms = {arm: [n for n in names if all((n, k) in results for k in keys)] for arm, names in arms.items()}
    one = lambda name, gate=None: group_tally(keys, lambda k: results[name, k][2], gate)  # noqa: E731
    m = medians[group] = {}
    m["agy text v7"] = show("agy text v7", [one(n) for n in arms["text"]])
    m["Sol text v7"] = show("Sol text v7", [one(n) for n in arms["sol"]])
    m["audio v7"] = show("audio v7", [one(n) for n in arms["audio"]])
    m["audio v7 + F1b"] = show("audio v7 + F1b", [one(n, "short") for n in arms["audio"]])
    m["fusion agy"] = show("fusion agy v7", [fusion(keys, a, t) for a in arms["audio"] for t in arms["text"]])
    m["fusion Sol"] = show("fusion Sol v7", [fusion(keys, a, t) for a in arms["audio"] for t in arms["sol"]])
    m["P13g v8"] = show("P13g v8 (patched)", [one(n) for n in arms["p13"]])
    m["P13g v8 draft"] = show("P13g v8 draft", [one(n.replace(" r1", " draft")) for n in arms["p13"]])
    m["fusion v8 draft"] = show("fusion (v8 draft + audio)", [fusion(keys, a, n.replace(" r1", " draft"))
                                                              for a in arms["audio"] for n in arms["p13"]])
    for key in keys:
        per = {n: net(tally(wins[key], results[n, key][2])) for n in arms["text"] + arms["sol"] + arms["p13"]}
        print(f"  {key}: " + "  ".join(f"{n} {v:+d}" for n, v in per.items()))
for lv in levels.values():
    lv.save()

print("\n== criteria")
for group in medians:
    m = medians[group]
    print(f"  {group}: fusion agy {m['fusion agy']:+.1f} vs text {m['agy text v7']:+.1f}, audio+F1b {m['audio v7 + F1b']:+.1f};"
          f" P13g v8 {m['P13g v8']:+.1f} (>= fusion - 5: {m['P13g v8'] >= m['fusion agy'] - 5});"
          f" Sol fusion - agy fusion {m['fusion Sol'] - m['fusion agy']:+.1f}")

# splits on the held-out windows (row-split truth, ruler 1)
items, _ = R.load("1")
SPLIT = {(it["ident"], it["i"]): it["truth"] for it in items if it["truth"] and it["truth"].get("kind") == "split"}


def units(payload):
    return [u for w in payload["windows"] for u in (w.get("parts") or [w])]


def splits(name, keys=M.HELDOUT):
    c = Counter()
    for key in keys:
        win = wins[key]
        ident = key.rsplit("-c", 1)[0]
        at = {(round(s["start"], 3), round(s["end"], 3)): k for k, s in enumerate(S.stable(ident))}
        k_of = {i: at.get((round(s["start"], 3), round(s["end"], 3))) for i, s in enumerate(win.segs)}
        index = {r: i for i, r in enumerate(win.ids)}
        truth_rows = {i for i, k in k_of.items() if (ident, k) in SPLIT}
        c["truth"] += len(truth_rows)
        for u in units(results[name, key][2]):
            for s in u.get("splits") or []:
                c["made"] += 1
                i = index[str(s["row"])]
                if i in truth_rows:
                    cut = sum(len(x["word"]) for x in win.segs[i]["words"][:s["at"]])
                    c["right"] += any(lo - 1 <= cut <= hi + 1 for lo, hi in SPLIT[ident, k_of[i]]["cuts"])
            c["fills"] += len(u.get("fills") or [])
            c["inserts"] += len(u.get("inserts") or [])
    return c


print("\n== splits on the held-out windows (made / right cut / truth)")
for name in ["38F p13gv8a r1", "38F p13gv8b r1", "38F p13g1 r1", "38F p13g2 r1", "38F p13gb1 r1", "38F p13gb2 r1"] + P7:
    c = splits(name)
    print(f"  {name:<18} made {c['made']:>2} right {c['right']:>2} / {c['truth']}  fills {c['fills']} inserts {c['inserts']}")
print("== splits made on hina-71")
for name in ["38F p13gv8a r1", "38F p13gv8b r1", "38F split d1a", "38F split d1b"]:
    print(f"  {name:<18} made {splits(name, H71)['made']}")


def harness(name, keys):
    c = Counter()
    for key in keys:
        for u in units(results[name, key][2]):
            pt = u.get("patch") or {}
            c["units"] += 1
            c["patched"] += pt.get("status") == "success"
            for b in pt.get("blocked") or []:
                c["blocked " + b["kind"]] += 1
    return dict(c)


print("\n== P13g v8 patch status")
for name in ["38F p13gv8a r1", "38F p13gv8b r1"]:
    for group, keys in GROUPS:
        print(f"  {name} {group}: {harness(name, keys)}")

UUID = r"\| ([0-9a-f]{8}-[0-9a-f-]{27}) \|"


def usage(folder, key):
    """Per agy session (or per Codex exchange chain), the largest cumulative numbers its exchange records show."""
    per = {}
    for f in sorted(glob.glob(f"{ACCEPT}/{folder}/{key}/llm/exchanges/*.md")):
        t = open(f, encoding="utf-8").read(6000)
        m = re.search(UUID, t)
        i = re.search(r"input_tokens: (\d+) / (\d+) /", t)
        o = re.search(r"output_tokens_breakdown: (\d+) / (\d+) /", t)
        if i and o:
            now = tuple(map(int, (i[1], i[2], o[1], o[2])))
            sid = m[1] if m else f
            per[sid] = max(per.get(sid, (0, 0, 0, 0)), now)
    return len(per), [sum(v[k] for v in per.values()) for k in range(4)]


print("\n== usage per run and window (sessions, uncached / cached / visible out / thinking, in 万)")
for folder in ["main-rtv7a", "main-rd1a", "main-rd1b", "main-text-rsol7a", "main-text-rd1sola", "main-text-rd1solb",
               "audio-rsplit1", "audio-rd1a", "audio-rd1b", "audio-rp13gv8a", "audio-rp13gv8b"]:
    for key in M.HELDOUT + H71:
        n, (unc, cac, out, think) = usage(folder, key)
        if n:
            print(f"  {folder:<20} {key}: {n} x  {unc / 1e4:6.1f} / {cac / 1e4:6.1f} / {out / 1e4:5.1f} / {think / 1e4:5.1f}"
                  f"  weighted {(unc / 5 + out + think) / 1e4:5.1f}")
