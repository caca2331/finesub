"""Word-gap scoring for non-VAD boundaries: `max(-1, (pause==0)*penalty - pause**2)`.

**SUPERSEDED (2026-07-29): this shipped.** Production `segment_split` now implements this
term, the ASR-seam bonus and the global DP; see docs/segment_split.md. What follows describes
the *pre-migration* production it was written against, and the arms below rebuild that old
formula locally as the control. Kept as the record of how the constants were chosen -- to
sweep the shipped parameters instead, vary `whisper_segment_bonus` / `non_vad_gap_penalty`
on `DEFAULT_SPLIT_PARAMS` and call `split_segments` directly.

Production scores a boundary as `a*(t + g_score(g)) + base`, plus `no_gap_penalty` when
`g <= 0` — and `interval_gap_between` returns 0 for **any** two words anchored to the same VAD
interval, however far apart they are. 24 588 of the 25 114 intra-segment boundaries in the bed
(97.9%) sit in that one bucket, ordered only by `t_score`'s six discrete values.

This replaces the `g <= 0` branch with a single continuous term over the aligner's *word-level*
pause. It degenerates exactly to production at `pause == 0` (`t + penalty + base` = today's
`t + base + no_gap_penalty` at `penalty = 1`), so it is a generalisation, not a substitute.
The `-1` floor keeps a long word pause from outvoting everything else.

Measured (14 gold windows, baseline = `split_segments` re-run, holdout 163 `must`):

    bonus 4.5-7 @ penalty=1   strictly dominant — +1 `must`, no window worse on either axis
    bonus 4.5                 156/163, 403 cuts, 122 violations (-9 / -10, -7.6%)
    bonus <= 4                one window loses a `must`
    penalty 0.75-1.0          plateau; >= 1.5 over-merges (fit set 58 -> 57 -> 48 at 3.0)

The quadratic term itself changes nothing here (`b=4.5` quad 403/122 vs pure waiver 401/122):
the `g == 0` bucket only ever holds short pauses, because a long one would have made the VAD
split the interval. It earns its place on the edge case that bucket cannot show — material where
separation residue lifts the noise floor and the energy VAD drops quiet speech (`yingtao`,
non-speech at -48.7 dB), so a 0.5-0.9 s word pause coexists with `g = 0`. No `must` sits on such
a position in this gold set.

    python -m tools.qwen3_explore.wordgap_ladder
"""

import json, sys, collections
sys.path.insert(0,"."); sys.path.insert(0,"src")
from segment_split import (DEFAULT_SPLIT_PARAMS as SP, Boundary, adjust_words, build_zones,
                           dp_split, score_boundaries, split_segments, t_score,
                           g_score as pg, interval_gap_between)
from tools.segmentation_gold.gold import cuts_to_indices, evaluate_by_index, load_golds, load_words
from tools.qwen3_explore.common import baseline_aligned, vad_json
from tools.qwen3_explore.scope_control import segments_in_range, PAD_SEC
from tools.qwen3_explore.gold_sweep import GOLD_DIR, _cues
FIT={"BV1nxje63ERi-115-235","BV1UBjq6fEgb-46-166","yui-660-780"}

def quad(x, L, R, penalty):
    """g<=0 时用 max(-1, (pause==0)*penalty - pause^2) 取代 g_score + no_gap_penalty。"""
    if x.banned or x.g > 0: return x
    p = max(0.0, float(R.source["start"]) - float(L.source["end"]))
    s = max(-1.0, (penalty if p <= 0 else 0.0) - p*p)
    return Boundary(False, x.g, x.t, SP.a*(x.t + s) + SP.base, x.non_vad_gap)
def waive(x, L, R):
    if x.banned or x.g > 0: return x
    p = max(0.0, float(R.source["start"]) - float(L.source["end"]))
    return x if p <= 0 else Boundary(False, x.g, x.t, SP.a*x.t + SP.base, x.non_vad_gap)

def build(segs, spans, zones):
    adj=[]; slots=[]; junc=[]
    for s in segs:
        a = adjust_words(s.get("words") or [], spans, zones)
        if not a: continue
        if adj:
            L,R = adj[-1], a[0]
            gg = interval_gap_between(spans, L.anchor, R.anchor); tt = t_score(L.text,R.text,R.space_before)
            b = Boundary(False,gg,tt,SP.a*(tt+pg(gg,SP))+SP.base+(SP.non_vad_gap_penalty if gg<=0 else 0),gg<=0)
            junc.append(len(slots)); slots.append((b,L,R))
        if len(a)>=2:
            for i,x in enumerate(score_boundaries(a,spans,SP)): slots.append((x,a[i],a[i+1]))
        adj.extend(a)
    return adj, slots, set(junc)

def run(adj, slots, junc, bonus, mode, penalty=1.0):
    bd=[]
    for i,(x,L,R) in enumerate(slots):
        y = x if mode=="none" else (waive(x,L,R) if mode=="waive" else quad(x,L,R,penalty))
        bd.append(Boundary(False,y.g,y.t,y.b-bonus,y.non_vad_gap) if i in junc and not y.banned else y)
    return [{"start":adj[p].start,"end":adj[q-1].end,"text":"".join(adj[k].text for k in range(p,q))}
            for p,q in dp_split(adj,bd,SP).pieces]

ARMS=[("基线=重跑",None)]
ARMS+=[(f"b={b:g} 二次式",("quad",b,1.0)) for b in (3.5,4,4.5,5,5.5,6,7)]
ARMS+=[(f"b={b:g} 纯免罚",("waive",b,0)) for b in (4.5,5.5)]
ARMS+=[(f"b=4.5 penalty={p:g}",("quad",4.5,p)) for p in (0.25,0.5,0.75,1.5,2.0,3.0)]
T=collections.defaultdict(lambda: collections.defaultdict(collections.Counter)); per=collections.defaultdict(list)
for g in load_golds(GOLD_DIR):
    name=f"{g['clip']}-{int(g['window'][0])}-{int(g['window'][1])}"; grp="拟合" if name in FIT else "留出"
    raw=json.loads(baseline_aligned(g["clip"]).read_text(encoding="utf-8"))
    vad=json.loads(vad_json(g["clip"]).read_text(encoding="utf-8"))
    spans=sorted((float(a),float(b)) for a,b in vad["speech"] if b>a); zones=build_zones(spans)
    sd=[{"start":a,"end":b} for a,b in spans]
    words=load_words(baseline_aligned(g["clip"]))
    segs=segments_in_range(raw,float(g["window"][0])-PAD_SEC,float(g["window"][1])+PAD_SEC)
    if len(segs)<2: continue
    sc=lambda cues: evaluate_by_index([g],cuts_to_indices(_cues({"segments":cues}),words))
    base=sc(split_segments(segs,sd,params=SP))
    adj,slots,junc=build(segs,spans,zones)
    for n,cfg in ARMS:
        e = base if cfg is None else sc(run(adj,slots,junc,cfg[1],cfg[0],cfg[2]))
        for gg in (grp,"全部"):
            for k in ("must","must_hit","declared","presumed","in_window","skipped"): T[n][gg][k]+=e[k]
        per[n].append((e["must_hit"]-base["must_hit"],(e["declared"]+e["presumed"])-(base["declared"]+base["presumed"])))
def row(n):
    t=T[n]["留出"]; f=T[n]["拟合"]; r=per[n]
    fv=f["declared"]+f["presumed"]; tv=t["declared"]+t["presumed"]
    m=f"{sum(1 for x in r if x[0]>0)}/{sum(1 for x in r if x[0]==0)}/{sum(1 for x in r if x[0]<0)}"
    v=f"{sum(1 for x in r if x[1]<0)}/{sum(1 for x in r if x[1]==0)}/{sum(1 for x in r if x[1]>0)}"
    st=" ★" if sum(1 for x in r if x[0]<0)==0 and sum(1 for x in r if x[1]>0)==0 and (sum(1 for x in r if x[0]>0) or sum(1 for x in r if x[1]<0)) else ""
    print(f"{n:20} {f['must_hit']:>2}/{f['must']} {f['in_window']:>4}刀{fv:>4}违 | "
          f"{t['must_hit']:>3}/{t['must']} {t['in_window']:>4}刀{tv:>4}违 | {m:>10} {v:>10}{st}")
print(f"{'臂':20} {'拟合(61)':>16} | {'留出(163)':>20} | {'must 好/平/坏':>10} {'违反 好/平/坏':>10}")
for n,_ in ARMS:
    if n.startswith("b=4.5 penalty=0.25"): print()
    row(n)
