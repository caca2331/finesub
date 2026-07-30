"""目标 3 重测：换 aligner 对**分割结果**有没有影响（干净口径）。

早期对比把两件事混在一起：Qwen ForcedAligner 自带 nagisa 分词（787 -> 589 词，且**丢标点**），
所以直接比等于同时换了分词和时间戳。这里把 Qwen 的词边界按**字符偏移**映回 whisper 的分词，
得到「同文本、同分词、同标点，只有词时间戳不同」的两条臂，再各自过两种分句器。

重定时的坑：`end` 必须用该 token 的**最后一个**字符去定位所属 Qwen 词，否则会落进下一个词、
把词间停顿整个抹平（第一版就把信号做成了 100% 零）。

结论（3 个有 gold 窗口的 clip，48 个 `must`）：wt 与 qwen 在生产分句器下**逐位相同**
（41/48、123 刀、41 违反），刀集差异 <=1 处且那 1 处是 `never`。不需要按 Qwen 调参——
它的特征提示的那个旋钮（最小停顿阈值，防其伪造停顿）在 0.05 无作用、0.1 反而掉 1 个 `must`。
样本只有 3 窗，结论是「测不出差异」而非「已证相等」。

    python -m tools.qwen3_explore.aligner_gold
"""

"""目标 3 重测：同文本、同分词、同标点，只有词时间戳来自不同 aligner。"""
import json, sys, collections, bisect
sys.path.insert(0,"."); sys.path.insert(0,"src")
from segment_split import (DEFAULT_SPLIT_PARAMS as SP, Boundary, adjust_words, build_zones,
                           dp_split, score_boundaries, split_segments, t_score,
                           g_score as pg, interval_gap_between)
from tools.segmentation_gold.gold import cuts_to_indices, evaluate_by_index, load_golds, load_words
from tools.qwen3_explore.common import baseline_aligned, vad_json
from tools.qwen3_explore.gold_sweep import GOLD_DIR, _cues
PAIR="out/qwen-explore/{}-align-seg-baselinetext.json"
STRIP="。．｡.!！?？…‥、，,､；;：:「」『』（）()　 \t\n・-–—"
def strip(s): return "".join(c for c in s if c not in STRIP)

def retime(segs):
    """把 Qwen 的词边界按字符偏移映到 baseline 分词上，返回 baseline 分词 + Qwen 时间戳。"""
    qo=[0.0]; qt=[]; off=0
    for s in segs:
        for w in (s.get("words") or []):
            n=len(strip(w["word"]))
            if n==0: continue
            qt.append((off, off+n, float(w["start"]), float(w["end"]))); off+=n
    if not qt: return None
    starts=[x[0] for x in qt]
    def at(o, edge):
        """edge='s' 取含该字符的 Qwen 词内插起点；'e' 取含该字符的词内插终点。
        关键：终点必须用 token 的**最后一个**字符定位，否则会落进下一个 Qwen 词、
        把词间停顿抹平（第一版就是这样把信号做成了 100% 零）。"""
        i=min(bisect.bisect_right(starts,o)-1, len(qt)-1); i=max(i,0)
        a,b,st,en=qt[i]
        if b<=a: return st if edge=="s" else en
        f=min(max((o-a)/(b-a),0.0),1.0); g=min(max((o+1-a)/(b-a),0.0),1.0)
        return st+(en-st)*(f if edge=="s" else g)
    out=[]; off=0
    for s in segs:
        ws=[]
        for w in (s.get("baseline_words") or []):
            n=len(strip(w["word"]))
            st=at(off,"s"); en=at(off+max(n,1)-1,"e")
            if en<=st: en=st+0.01
            ws.append(dict(w, start=round(st,3), end=round(en,3))); off+=n
        if ws: out.append({"start":ws[0]["start"],"end":ws[-1]["end"],
                           "text":s.get("baseline_text") or s.get("text"),"words":ws})
    return out

def wt_arm(segs):
    out=[]
    for s in segs:
        ws=[dict(w) for w in (s.get("baseline_words") or [])]
        if ws: out.append({"start":ws[0]["start"],"end":ws[-1]["end"],
                           "text":s.get("baseline_text") or s.get("text"),"words":ws})
    return out

def quad(x,L,R,pen=1.0):
    if x.banned or x.g>0: return x
    p=max(0.0,float(R.source["start"])-float(L.source["end"]))
    return Boundary(False,x.g,x.t,SP.a*(x.t+max(-1.0,(pen if p<=0 else 0.0)-p*p))+SP.base,x.non_vad_gap)
def new_split(segs,spans,zones,bonus=4.5,pen=1.0):
    adj=[];bd=[]
    for s in segs:
        a=adjust_words(s.get("words") or [],spans,zones)
        if not a: continue
        if adj:
            L,R=adj[-1],a[0]
            gg=interval_gap_between(spans,L.anchor,R.anchor);tt=t_score(L.text,R.text,R.space_before)
            b=quad(Boundary(False,gg,tt,SP.a*(tt+pg(gg,SP))+SP.base+(SP.non_vad_gap_penalty if gg<=0 else 0),gg<=0),L,R,pen)
            bd.append(Boundary(False,b.g,b.t,b.b-bonus,b.non_vad_gap))
        if len(a)>=2:
            bs=score_boundaries(a,spans,SP); bd.extend(quad(x,a[i],a[i+1],pen) for i,x in enumerate(bs))
        adj.extend(a)
    if len(adj)<2: return list(segs)
    return [{"start":adj[x].start,"end":adj[y-1].end,"text":"".join(adj[k].text for k in range(x,y))}
            for x,y in dp_split(adj,bd,SP).pieces]

T=collections.defaultdict(collections.Counter)
for g in load_golds(GOLD_DIR):
    clip=g["clip"]
    import os
    if not os.path.exists(PAIR.format(clip)): continue
    pair=json.loads(open(PAIR.format(clip),encoding="utf-8").read())["segments"]
    vad=json.loads(vad_json(clip).read_text(encoding="utf-8"))
    spans=sorted((float(a),float(b)) for a,b in vad["speech"] if b>a); zones=build_zones(spans)
    sd=[{"start":a,"end":b} for a,b in spans]
    words=load_words(baseline_aligned(clip))
    sc=lambda cues: evaluate_by_index([g],cuts_to_indices(_cues({"segments":cues}),words))
    arms={"wt":wt_arm(pair),"qwen":retime(pair)}
    add=lambda n,e:[T[n].__setitem__(k,T[n][k]+e[k]) for k in ("must","must_hit","declared","presumed","in_window","skipped")]
    add("生产原样(存档)",sc([s for s in json.loads(baseline_aligned(clip).read_text(encoding='utf-8'))["segments"] if s.get("words")]))
    for an,segs in arms.items():
        if not segs: continue
        add(f"{an} + 生产分句", sc(split_segments(segs,sd,params=SP)))
        add(f"{an} + 新设计",   sc(new_split(segs,spans,zones)))
print(f"gold 窗口 3 个（BV1kYLR6AEXv / BV1UBjq6fEgb / BV1ySjz6FEzD），48 个 must\n")
print(f"{'臂':22} {'必切召回':>16} {'刀':>5} {'违反':>5} {'/百条':>7}")
for n,t in T.items():
    j=t["in_window"]-t["skipped"]; v=t["declared"]+t["presumed"]
    print(f"{n:22} {100*t['must_hit']/t['must']:5.1f}% ({t['must_hit']:2d}/{t['must']}) {t['in_window']:>5} {v:>5} {100*v/j if j else 0:>7.1f}")
