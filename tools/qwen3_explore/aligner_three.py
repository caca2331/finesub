"""三臂对照：生产 whisper_timestamped / Qwen ForcedAligner / openai-whisper 自带 word_timestamps。

同文本、同分词、同标点，只有词时间戳不同（后两者按字符偏移映回 baseline 分词）。

`ow` 是**重新解码**，文本与 baseline 只有 0.888-0.947 相似度，所以它的映射必须走 difflib
字符级对齐（裸偏移会整体漂移；未过滤时出现过 31 s 的 Δt 极值——那是对齐失配，不是时间戳
差异）。统计 Δt 时只取落在长匹配块内部、离两端 >=5 字的位置。`qwen` 是 forced-align 到
baseline 文本的，相似度 1.000，没有这个问题。

用途：`ow` 与 `wt` 是同一思路的两个实现（都是 cross-attention DTW，同一个 large-v3-turbo），
它们的差就是**同方法噪声地板**——切点上中位 0.015 s、超 0.5 s 为 0%。据此判断 Qwen 的
分歧是否真实：它是中位 0.060 s、超 0.5 s 5.7%，远在地板之上。

先产出 ow 词时间戳（约 13.5 min 音频、GPU 约 1 min）：
    python tools/qwen3_explore/ow_words.py
再跑对照：
    python -m tools.qwen3_explore.aligner_three
"""

"""三臂对照：wt（生产 whisper_timestamped）/ qwen ForcedAligner / ow（openai-whisper 自带）。
同文本、同分词、同标点，只有词时间戳不同。"""
import json, sys, os, bisect, collections
sys.path.insert(0,"."); sys.path.insert(0,"src")
from segment_split import (DEFAULT_SPLIT_PARAMS as SP, Boundary, adjust_words, build_zones,
                           dp_split, score_boundaries, split_segments, t_score,
                           g_score as pg, interval_gap_between)
from tools.segmentation_gold.gold import cuts_to_indices, evaluate_by_index, load_golds, load_words
from tools.qwen3_explore.common import baseline_aligned, vad_json
from tools.qwen3_explore.gold_sweep import GOLD_DIR, _cues
ns={}; exec(open("tools/qwen3_explore/aligner_gold.py",encoding="utf-8").read().split("T=collections.defaultdict")[0], ns)
wt_arm, PAIR, strip, quad, new_split = ns["wt_arm"], ns["PAIR"], ns["strip"], ns["quad"], ns["new_split"]

def remap(donor_words, pair):
    """把 donor 的词边界映到 baseline 分词上。donor_words: [(word,start,end)]

    donor 文本若与 baseline 不同（ow 是重新解码），裸偏移会整体漂移，必须走字符级对齐。
    """
    import difflib
    qt=[]; off=0
    for w,s,e in donor_words:
        n=len(strip(w))
        if n: qt.append((off,off+n,float(s),float(e))); off+=n
    if not qt: return None
    starts=[x[0] for x in qt]
    def at(o,edge):
        i=max(min(bisect.bisect_right(starts,o)-1,len(qt)-1),0); a,b,st,en=qt[i]
        if b<=a: return st if edge=="s" else en
        f=min(max((o-a)/(b-a),0,),1); g=min(max((o+1-a)/(b-a),0),1)
        return st+(en-st)*(f if edge=="s" else g)
    base="".join(strip(w["word"]) for s in pair for w in (s.get("baseline_words") or []))
    don="".join(w for w,_,_ in donor_words); don=strip(don)
    blocks=difflib.SequenceMatcher(None,base,don,autojunk=False).get_matching_blocks()
    if don==base:
        m=lambda o:o
    else:
        def m(o):
            for a,b,size in blocks:
                if a<=o<=a+size: return b+(o-a)
            return min((b+(0 if o<a else size) for a,b,size in blocks if size),
                       key=lambda x:abs(x-o), default=0)
    def clean(o, margin=5):
        """该 baseline 字符是否落在长匹配块内部（离两端 >=margin）——映射可信才计入 dt。"""
        return don == base or any(size >= 2 * margin and a + margin <= o <= a + size - margin
                                  for a, b, size in blocks)

    out=[]; off=0; ok=[]
    for s in pair:
        ws=[]
        for w in (s.get("baseline_words") or []):
            n=len(strip(w["word"])); st=at(m(off),"s"); en=at(m(off+max(n,1)-1),"e")
            if en<=st: en=st+0.01
            ws.append(dict(w,start=round(st,3),end=round(en,3)))
            ok.append(clean(off) and clean(off+max(n,1)-1)); off+=n
        if ws: out.append({"start":ws[0]["start"],"end":ws[-1]["end"],"text":s.get("baseline_text"),"words":ws})
    return out, ok

T=collections.defaultdict(collections.Counter); DT=collections.defaultdict(list); TXT={}
for g in load_golds(GOLD_DIR):
    clip=g["clip"]
    ow_p=f"out/qwen-explore/{clip}-ow-words.json"
    if not os.path.exists(PAIR.format(clip)) or not os.path.exists(ow_p): continue
    pair=json.loads(open(PAIR.format(clip),encoding="utf-8").read())["segments"]
    ow=json.loads(open(ow_p,encoding="utf-8").read())["segments"]
    vad=json.loads(vad_json(clip).read_text(encoding="utf-8"))
    spans=sorted((float(a),float(b)) for a,b in vad["speech"] if b>a); zones=build_zones(spans)
    sd=[{"start":a,"end":b} for a,b in spans]; words=load_words(baseline_aligned(clip))
    base_txt=strip("".join(w["word"] for s in pair for w in (s.get("baseline_words") or [])))
    for nm,don in (("qwen",[(w["word"],w["start"],w["end"]) for s in pair for w in (s.get("words") or [])]),
                   ("ow",  [(w["word"],w["start"],w["end"]) for s in ow  for w in (s.get("words") or [])])):
        TXT.setdefault(nm,[]).append((clip, strip("".join(w for w,_,_ in don)), base_txt))
    A={"wt":wt_arm(pair)}
    OK={"wt":None}
    A["qwen"],OK["qwen"]=remap([(w["word"],w["start"],w["end"]) for s in pair for w in (s.get("words") or [])],pair)
    A["ow"],OK["ow"]  =remap([(w["word"],w["start"],w["end"]) for s in ow  for w in (s.get("words") or [])],pair)
    lab={it["k"] for it in g["items"] if isinstance(it.get("k"),int)}; lo,hi=min(lab),max(lab)
    sc=lambda cues: evaluate_by_index([g],cuts_to_indices(_cues({"segments":cues}),words))
    add=lambda n,e:[T[n].__setitem__(k,T[n][k]+e[k]) for k in ("must","must_hit","declared","presumed","in_window","skipped")]
    for an,segs in A.items():
        if not segs: continue
        add(f"{an} + 生产分句", sc(split_segments(segs,sd,params=SP)))
        add(f"{an} + 新设计",   sc(new_split(segs,spans,zones)))
    W={n:[w for s in A[n] for w in s["words"]] for n in A if A[n]}
    cuts={k for k in cuts_to_indices(_cues({"segments":split_segments(A["wt"],sd,params=SP)}),words) if lo<=k<=hi}
    # 三个配对必须同底：ow 的可信位置最少，全部配对都限制到它，否则 wt~qwen 用 123 刀、
    # qwen~ow 用 63 刀，两个数字不可比（曾据此得出错误的「3 倍」结论）。
    same = lambda i: OK["ow"][i] and OK["ow"][i+1]
    for a,b in (("wt","qwen"),("wt","ow"),("qwen","ow")):
        for i in range(len(W[a])-1):
            if not same(i): continue
            d=abs((W[a][i]["end"]+W[a][i+1]["start"])/2-(W[b][i]["end"]+W[b][i+1]["start"])/2)
            DT[f"{a}~{b} 全部"].append(d)
            if i in cuts: DT[f"{a}~{b} 切点"].append(d)
import difflib
print("文本一致性（去标点后）：")
for nm,rows in TXT.items():
    for clip,t,b in rows:
        r=difflib.SequenceMatcher(None,b,t,autojunk=False).ratio()
        print(f"  {nm:5} {clip:14} donor {len(t):>5} 字 vs baseline {len(b):>5}  相同={t==b}  相似度 {r:.3f}")
print(f"\n{'臂':22} {'必切召回':>15} {'刀':>5} {'违反':>5}")
for n,t in T.items():
    print(f"{n:22} {100*t['must_hit']/t['must']:5.1f}% ({t['must_hit']:2d}/{t['must']}) {t['in_window']:>5} {t['declared']+t['presumed']:>5}")
print(f"\n{'配对':16} {'n':>6} {'中位':>8} {'p90':>8} {'>0.3s':>8} {'>0.5s':>8} {'最大':>7}")
for k,v in DT.items():
    v=sorted(v); p=lambda f: v[min(int(len(v)*f),len(v)-1)]
    print(f"{k:16} {len(v):>6} {p(.5):>8.3f} {p(.9):>8.3f} {sum(x>0.3 for x in v)/len(v):>7.1%} {sum(x>0.5 for x in v)/len(v):>7.1%} {v[-1]:>7.2f}")
