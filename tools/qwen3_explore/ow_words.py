"""openai-whisper 自带 word_timestamps 的第三对照臂：只产出词时间戳，落盘备用。"""
import json, sys, time
sys.path.insert(0,".")
import whisper
from tools.qwen3_explore.common import baseline_aligned
CLIPS={"BV1kYLR6AEXv":None,"BV1UBjq6fEgb":None,"BV1ySjz6FEzD":None}
model=whisper.load_model("large-v3-turbo", device="cuda")
for clip in CLIPS:
    audio=json.loads(open(f"out/qwen-explore/{clip}-align-seg-baselinetext.json",encoding="utf-8").read())["metadata"]["audio"]
    t=time.perf_counter()
    r=model.transcribe(audio, language="ja", word_timestamps=True, verbose=False,
                       condition_on_previous_text=False)
    dt=time.perf_counter()-t
    out={"metadata":{"impl":"openai-whisper word_timestamps","model":"large-v3-turbo","audio":audio,
                     "rtf":round(dt/max(r["segments"][-1]["end"],1),4)},
         "segments":[{"start":s["start"],"end":s["end"],"text":s["text"],
                      "words":[{"word":w["word"],"start":w["start"],"end":w["end"]} for w in s.get("words") or []]}
                     for s in r["segments"]]}
    p=f"out/qwen-explore/{clip}-ow-words.json"
    open(p,"w",encoding="utf-8").write(json.dumps(out,ensure_ascii=False))
    nw=sum(len(s["words"]) for s in out["segments"])
    print(f"{clip:14} {len(out['segments']):>4} 段 {nw:>5} 词  {dt:.0f}s → {p}", flush=True)
