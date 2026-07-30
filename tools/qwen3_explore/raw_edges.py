"""原生对照：两个 whisper 实现的词时间戳，按「段首 / 段尾 / 段内」分开比。

回答：`aligner_three` 里 wt~ow 只差 0.015 s，是方法本来就一样，还是工程抹平了？
——都不是。那个数字是重定时之后量的，段边界的分歧被洗掉了（donor 的段结构与 baseline
不重合，段边界处取到的其实是 donor 的段内时间）。

三个必须守住的口径（各错过一次）：

1. **参数与生产一致**。`refine_whisper_precision` 生产用 1.0，是 wt 自带的边界精修。
   最初把它关成 0 再去量段边界 = 关掉被测机制本身，段首分歧虚高一倍（0.360 vs 0.180）。
   它只动边界：同一次解码下段内移动中位恰好 0.000 s，解码结果逐词不变。
2. **「与生产一致」指有效行为，不是名义参数值**。`asr_align.py` 没有设
   `condition_on_previous_text`，名义上取 whisper 的默认 `True`（`asr_wt.py` 里那个
   `False` 属于独立工具，无人 import）。但生产**不做整片解码**——`build_combined_audio`
   按 VAD 分组拼音频、`group_target_sec=30 s`，而 whisper 的内部窗口也是 30 s，每次
   `transcribe` 基本只有一个窗口，「上一段文本」不存在，**有效等价于 `False`**。
   本对照因此用 `False` 整片跑；照抄名义值会累积几分钟的条件漂移（两边相似度从
   0.88-0.98 掉到 0.52-0.60），那是生产从不会有的状态。
3. **两侧必须同时判为该类型**。生产的 segment 是 VAD 区间分组，原生的是解码器分段，
   不是同一种东西；只按一侧判定会混进「归属不一致」的位置（中位 0.205 s），
   曾把段首的工程移动量虚报成 0.330 s（实际 0.025 s）。

实测（214/225/2081 个位置，文本相似度 0.882-0.984）：

    位置       wt(refine=0) vs ow   wt(refine=1.0) vs ow   生产工程移动量
    段首词·头   0.360 s              0.180 s                0.025 s
    段尾词·尾   0.080 s              0.020 s                0.115 s   <- 能量补齐
    段内词      0.020 s              0.020 s                0.010 s

结论：两个实现只在段内与段尾一致；「segment 从哪里开始」始终分歧，即便带 refine 也是段内
的 9 倍。生产在 refine 之上的额外工程只动段尾。0.015 s 是**段内**噪声地板，不能外推到切点。

    python tools/qwen3_explore/raw_edges.py        # 三次解码：ow / wt refine=0 / wt refine=1
    python -m tools.qwen3_explore.raw_edges_cmp
"""

"""只改 refine_whisper_precision 这一个变量，其余保持 raw2 的高一致性设置。"""
import json, sys, time, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0,".")
import whisper, whisper_timestamped as wts
m=whisper.load_model("large-v3-turbo", device="cuda")
BASE=dict(language="ja", temperature=0.0, beam_size=None, best_of=None,
          condition_on_previous_text=False, verbose=None, vad=False,
          detect_disfluencies=False, trust_whisper_timestamps=True,
          naive_approach=False, remove_empty_words=False,
          compute_word_confidence=False, min_word_duration=0.0)
for clip in ("BV1kYLR6AEXv","BV1UBjq6fEgb","BV1ySjz6FEzD"):
    audio=json.loads(open(f"out/qwen-explore/{clip}-align-seg-baselinetext.json",encoding="utf-8").read())["metadata"]["audio"]
    a=whisper.load_audio(audio); out={}
    for tag,rf in (("wt_r0",0.0),("wt_r1",1.0)):
        t=time.perf_counter(); r=wts.transcribe(m,a,refine_whisper_precision=rf,**BASE)
        out[tag]=[{"start":s["start"],"end":s["end"],"text":s["text"],
                   "words":[{"word":w["text"],"start":w["start"],"end":w["end"]} for w in (s.get("words") or [])]}
                  for s in r["segments"]]
        print(f"  {clip} {tag} {len(out[tag]):>3}段/{sum(len(s['words']) for s in out[tag]):>4}词 ({time.perf_counter()-t:.0f}s)",flush=True)
    open(f"out/qwen-explore/{clip}-raw4.json","w",encoding="utf-8").write(json.dumps({"metadata":{"audio":audio},**out},ensure_ascii=False))
