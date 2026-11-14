"""At adjacent-line seams with clean ruler-1 truth: does the acoustic label (main track) tell a speaker change?"""
import json, sys
from collections import Counter
from pathlib import Path
MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
sys.path.insert(0, str(MAIN / "tools/diarization_explore"))
import row_split as R
items, _ = R.load("1")
by = {}
for it in items:
    by.setdefault(it["ident"], {})[it["i"]] = it["truth"]
def ends(t):
    if t is None: return None
    if t["kind"] == "single" and len(t["people"]) == 1: return (t["people"][0], t["people"][0])
    if t["kind"] == "split": return (t["people"][0], t["people"][-1])
    return None
c = Counter()
for ident, rows in by.items():
    segs = json.loads((MAIN / "data/diarization/row-truth/products/acoustic" / ident / f"{ident}-stable.json").read_text(encoding="utf-8"))["segments"]
    for i in sorted(rows):
        if i + 1 not in rows: continue
        a, b = ends(rows[i]), ends(rows[i + 1])
        if a is None or b is None: continue
        sa, sb = segs[i].get("speaker") or {}, segs[i + 1].get("speaker") or {}
        ta, tb = sa.get("track"), sb.get("track")
        truth = "cross" if a[1] != b[0] else "same"
        if ta is None or tb is None:
            c[truth, "unlabelled"] += 1; continue
        low = any((s.get("mark") or {}).get("tier") == "low" for s in (sa, sb))
        c[truth, ("differ" if ta != tb else "agree") + (" (a ? side)" if low else " (both sure)")] += 1
for truth in ("cross", "same"):
    n = sum(v for k, v in c.items() if k[0] == truth)
    print(truth, n, {k[1]: v for k, v in sorted(c.items()) if k[0] == truth})
