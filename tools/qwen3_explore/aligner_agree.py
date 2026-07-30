"""Where do whisper-timestamped and the Qwen ForcedAligner disagree, on identical text?

The two stacks tokenise differently (whisper sub-words vs nagisa morphemes), so words cannot be
paired one-to-one. The transcript is the same string though, so boundaries are compared at
**character offsets**: each side's token boundaries become a set of offsets into the segment text,
and only offsets *both* sides propose are compared. That measures the same physical question —
"when does the audio move from this character to the next" — without either tokenisation biasing
the sample.

Deliberately reports agreement only. Neither side is ground truth: FINDINGS §4 already puts them
at a tie by re-transcription CER, so a disagreement here is a question, not a verdict. `--dump`
cuts each disputed boundary out of the vocal track with context so a human can arbitrate.

    python -m tools.qwen3_explore.aligner_agree                       # distribution, all clips
    python -m tools.qwen3_explore.aligner_agree --clip BV1kYLR6AEXv --top 20
    python -m tools.qwen3_explore.aligner_agree --clip BV1kYLR6AEXv --dump out/qwen-explore/agree
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import EXPLORE_OUT
from .qwen_split import CLAUSE, SENTENCE

PAIRED = "-align-seg-baselinetext.json"
# Everything either stack may attach to a word but the other may not; also spaces, which the
# Latin-script clips tokenise differently.
PUNCT_ALL = SENTENCE | CLAUSE | set("　 「」『』（）()、。…‥·・-–—")


def offsets(words: list[dict]) -> tuple[str, dict[int, float]]:
    """Concatenated text plus {char offset -> time at that boundary}.

    **Punctuation is stripped before counting offsets.** The Qwen side (nagisa) drops punctuation
    from its word text while whisper keeps it, so any segment containing punctuation had different
    concatenations and was discarded by the equality check below. That filter turned out to be
    perfectly correlated with the thing it excluded: of the four paired clips, *every* comparable
    segment was unpunctuated and *every* discarded one was punctuated (44/28, 68/14, 67/49,
    61/89). Since a splitter puts ~85% of its non-silence cuts right after punctuation, the
    excluded population was exactly the one that matters. Offsets therefore count only
    non-punctuation characters, which both stacks agree on.

    Interior boundaries only: offset 0 and len(text) are segment edges, where the two stacks
    agree by construction (both are handed the same segment span) and would dilute the sample.
    """
    text, at = "", {}
    for w in words:
        s = "".join(ch for ch in str(w.get("word", "")) if ch not in PUNCT_ALL)
        if not s:
            continue
        text += s
        at[len(text)] = float(w["end"])
    at.pop(len(text), None)
    return text, at


def compare(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for seg in data["segments"]:
        bw = seg.get("baseline_words") or []
        qt, qa = offsets(seg.get("words") or [])
        bt, ba = offsets(bw)
        # Which stripped offsets had punctuation on them in the original text — the junctions a
        # splitter actually cuts (~85% of its non-silence cuts follow punctuation).
        punct_at, off = set(), 0
        for w in bw:
            raw = str(w.get("word", ""))
            off += sum(1 for c in raw if c not in PUNCT_ALL)
            if any(c in SENTENCE or c in CLAUSE for c in raw):
                punct_at.add(off)
        if not qt or qt != bt:
            # Punctuation handling differs between the stacks; a segment whose concatenations do
            # not match is not comparable and is dropped rather than force-aligned.
            continue
        for off in sorted(set(qa) & set(ba)):
            out.append({
                "seg": seg.get("index"),
                "off": off,
                "q": qa[off],
                "b": ba[off],
                "d": qa[off] - ba[off],
                "left": qt[max(0, off - 8):off],
                "right": qt[off:off + 8],
                "punct": off in punct_at,
            })
    return out


def pct(xs: list[float], p: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip")
    ap.add_argument("--top", type=int, default=0, help="列出分歧最大的 N 个边界")
    ap.add_argument("--tail", action="store_true",
                    help="段尾专项：内部边界 vs 段尾，并给 Qwen 补上 whisper 侧的能量扩展")
    ap.add_argument("--thresh", type=float, default=0.30, help="算作分歧的阈值 (s)")
    ap.add_argument("--dump", help="把分歧边界切成 wav 供人工裁决（需 qwen-asr 环境）")
    ap.add_argument("--ab", action="store_true",
                    help="A/B 形式：每条两段音频，各自停在一个候选时刻，听哪段停在该停的地方")
    ap.add_argument("--limit", type=int, default=0, help="只导出分歧最大的 N 条")
    ap.add_argument("--punct-only", action="store_true",
                    help="只取紧跟句读的边界——分句器真正下刀的那一类")
    ap.add_argument("--skip-repeats", action="store_true",
                    help="跳过重复音/叠字边界（あーあー、ビルビル）——那里没有正确答案")
    args = ap.parse_args()

    if args.tail:
        cmd_tail()
        return

    paths = sorted(EXPLORE_OUT.glob(f"{args.clip or '*'}{PAIRED}"))
    if not paths:
        raise SystemExit(f"没有配对产物 (*{PAIRED})")

    allrows: list[tuple[str, dict]] = []
    print(f"{'clip':<16}{'可比边界':>8}{'|Δ| p50':>9}{'p90':>8}{'p99':>8}{'≥0.3s':>10}{'有向中位':>9}")
    for p in paths:
        clip = p.name[: -len(PAIRED)]
        rows = compare(p)
        if not rows:
            print(f"{clip:<16}{'—':>8}  (文本拼接不一致，无可比边界)")
            continue
        allrows += [(clip, r) for r in rows]
        d = [abs(r["d"]) for r in rows]
        big = sum(1 for x in d if x >= args.thresh)
        print(f"{clip:<16}{len(rows):>8}{pct(d,.5):>9.3f}{pct(d,.9):>8.3f}{pct(d,.99):>8.3f}"
              f"{f'{big} ({100*big/len(rows):.1f}%)':>10}{pct([r['d'] for r in rows],.5):>9.3f}")

    if args.top:
        print(f"\n分歧最大的 {args.top} 个边界（Δ = Qwen − wt，正=Qwen 更晚）")
        for clip, r in sorted(allrows, key=lambda x: -abs(x[1]["d"]))[: args.top]:
            print(f"  {clip:<14} seg{r['seg']:<4} Δ={r['d']:+6.2f}s  wt={r['b']:8.2f} qwen={r['q']:8.2f}"
                  f"   …{r['left']} ▮ {r['right']}…")

    if args.dump:
        rows = [(c, r) for c, r in allrows if abs(r["d"]) >= args.thresh]
        if args.punct_only:
            rows = [(c, r) for c, r in rows if r["punct"]]
            print(f"\n只取紧跟句读的边界: {len(rows)} 条")
        if args.skip_repeats:
            n0 = len(rows)
            rows = [(c, r) for c, r in rows if not _repetitive(r)]
            print(f"\n跳过 {n0 - len(rows)} 个重复音/叠字边界")
        n0 = len(rows)
        rows = _cluster(rows)
        print(f"合并 {n0} -> {len(rows)} 条：相邻边界属于同一处错位，不是独立样本")
        if args.limit:
            rows = _balanced(rows, args.limit)
            neg = sum(1 for _, r in rows if r["d"] < 0)
            print(f"取 {len(rows)} 条，qwen 更早 {neg} / qwen 更晚 {len(rows) - neg}"
                  "（正负都要，否则「谁更准」与「更晚的更准」分不开）")
        (dump_ab if args.ab else dump)(rows, args.dump, args.thresh)


def _cluster(rows, window: float = 1.5):
    """One misalignment spans several character boundaries; keep the largest of each run.

    Without this a single bad stretch (`さまざまな` mis-recognised as `まさまの`) contributes four
    or five "disagreements" that are neither independent evidence nor separate listening tasks.
    """
    out: list = []
    for clip, r in sorted(rows, key=lambda x: (x[0], min(x[1]["b"], x[1]["q"]))):
        if out and out[-1][0] == clip and min(r["b"], r["q"]) - max(out[-1][1]["b"], out[-1][1]["q"]) < window:
            if abs(r["d"]) > abs(out[-1][1]["d"]):
                out[-1] = (clip, r)
            continue
        out.append((clip, r))
    return sorted(out, key=lambda x: -abs(x[1]["d"]))


def _balanced(rows, limit: int):
    """Half from each sign of Δ, so "which aligner" is not confounded with "which is later"."""
    neg = [x for x in rows if x[1]["d"] < 0][: limit // 2]
    pos = [x for x in rows if x[1]["d"] > 0][: limit - len(neg)]
    neg = neg[: limit - len(pos)]
    return sorted(neg + pos, key=lambda x: -abs(x[1]["d"]))


def cmd_tail() -> None:
    """Is the '段尾早' difference the aligner, or the extension only the whisper side receives?

    `asr_align.extend_last_word_end_with_energy` runs once per ASR segment inside a VAD speech
    interval and moves **only that segment's last word end**, scanning forward while weighted
    energy stays within 20 dB of baseline (cap 1.0 s, upper-bounded by the next segment's first
    word). The paired artefact's `baseline_words` come from the production `*-aligned.json`, so
    they already carry it; the Qwen side is raw ForcedAligner output that never has. This applies
    the same function, with the same per-interval scope and the same bound, to the Qwen side.

    Interior boundaries are the control: production does not touch them on either side.
    """
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from asr_align import extend_last_word_end_with_energy  # noqa: PLC0415

    from .common import TARGET_SR, load_audio_16k, vad_json  # noqa: PLC0415

    def med(xs):
        return sorted(xs)[len(xs) // 2] if xs else float("nan")

    print(f"{'clip':<16}{'内部 n':>7}{'内部Δ中位':>10} | {'段尾 n':>7}{'段尾Δ中位':>10}"
          f"{'补扩展后':>10}{'覆盖':>7}")
    for p in sorted(EXPLORE_OUT.glob(f"*{PAIRED}")):
        clip = p.name[: -len(PAIRED)]
        vp = vad_json(clip)
        if not vp.exists():
            print(f"{clip:<16}  (无 VAD，跳过)")
            continue
        speech = [(float(a), float(b)) for a, b in
                  json.loads(vp.read_text(encoding="utf-8"))["speech"]]
        data = json.loads(p.read_text(encoding="utf-8"))
        segs = data["segments"]
        wav = load_audio_16k(data["metadata"]["audio"])

        inner, tail_raw, tail_ext = [], [], []
        for i, seg in enumerate(segs):
            qw, bw = seg.get("words") or [], seg.get("baseline_words") or []
            if not qw or not bw:
                continue
            qt, qa = offsets(qw)
            bt, ba = offsets(bw)
            if not qt or qt != bt:
                continue
            inner += [qa[o] - ba[o] for o in set(qa) & set(ba)]

            q_end, b_end = float(qw[-1]["end"]), float(bw[-1]["end"])
            iv = next((x for x in speech if x[0] <= q_end <= x[1]), None)
            if iv is None:
                continue
            nxt = next((float(segs[j]["words"][0]["start"]) for j in range(i + 1, len(segs))
                        if segs[j].get("words")), None)
            a, b = max(0, int(iv[0] * TARGET_SR)), min(len(wav), int(iv[1] * TARGET_SR))
            ext = extend_last_word_end_with_energy(
                [dict(w) for w in qw], interval_start=iv[0], interval_end=iv[1],
                interval_audio=wav[a:b], sr=TARGET_SR, next_word_start=nxt)
            tail_raw.append(q_end - b_end)
            tail_ext.append(float(ext[-1]["end"]) - b_end)
        print(f"{clip:<16}{len(inner):>7}{med(inner):>10.3f} | {len(tail_raw):>7}"
              f"{med(tail_raw):>10.3f}{med(tail_ext):>10.3f}{f'{100*len(tail_raw)/len(segs):.0f}%':>7}")
    print("\nΔ = Qwen − wt，负 = Qwen 更早。内部边界两侧都无后处理，是对照组。"
          "\n覆盖率不到 100% 是因为两侧文本拼接不一致的 segment 无法比较（标点口径不同），"
          "\n以及少数 segment 末词不落在任何 VAD 语音区间内。")


def _repetitive(r: dict) -> bool:
    """Repeated sounds have no true boundary — both aligners are guessing, so neither can be wrong.

    `あーあー` / `ビルビルビ` / `うんうん`: nothing for a listener to arbitrate either, so these
    only cost listening time. Three tests, because one is not enough: a narrow character set
    across the junction, a tail that collapses to a single character, or the same character on
    both sides of it.
    """
    left = r["left"].replace("ー", "")
    right = r["right"].replace("ー", "")
    if len(set(left[-4:] + right[:4])) <= 2:
        return True
    if len(set(left[-3:])) <= 1 or len(set(right[:3])) <= 1:
        return True
    return bool(left) and bool(right) and left[-1] == right[0]


def _rms_db(wav, t: float, half: float = 0.06) -> float:
    """Level in a short window centred on `t`. A token boundary should sit at a local minimum."""
    import numpy as np

    from .common import TARGET_SR

    a, b = max(0, int((t - half) * TARGET_SR)), min(len(wav), int((t + half) * TARGET_SR))
    if b <= a:
        return -120.0
    return float(20 * np.log10(max(1e-9, np.sqrt(np.mean(wav[a:b].astype("float64") ** 2)))))


def dump(allrows, outdir: str, thresh: float) -> None:
    import soundfile as sf

    from .common import TARGET_SR, load_audio_16k, vad_json

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    picked = [(c, r) for c, r in allrows if abs(r["d"]) >= thresh]
    picked.sort(key=lambda x: -abs(x[1]["d"]))
    index = []
    cache: dict[str, object] = {}
    sil: dict[str, list] = {}
    for n, (clip, r) in enumerate(picked, 1):
        meta = json.loads((EXPLORE_OUT / f"{clip}{PAIRED}").read_text(encoding="utf-8"))["metadata"]
        if clip not in cache:
            cache[clip] = load_audio_16k(meta["audio"])
            vp = vad_json(clip)
            sil[clip] = ([(float(a), float(b)) for a, b in
                          json.loads(vp.read_text(encoding="utf-8"))["non_speech"]] if vp.exists() else [])
        wav = cache[clip]
        lo, hi = min(r["b"], r["q"]) - 1.0, max(r["b"], r["q"]) + 1.0
        a, b = max(0, int(lo * TARGET_SR)), min(len(wav), int(hi * TARGET_SR))
        name = f"{n:03d}-{clip}-seg{r['seg']}-d{abs(r['d']):.2f}.wav"
        sf.write(str(root / name), wav[a:b], TARGET_SR)
        inq = any(s <= r["q"] <= e for s, e in sil[clip])
        inb = any(s <= r["b"] <= e for s, e in sil[clip])
        index.append({
            "wav": name, "clip": clip, "seg": r["seg"], "clip_start": round(lo, 3),
            "wt": round(r["b"], 3), "qwen": round(r["q"], 3), "delta": round(r["d"], 3),
            "wt_in_clip": round(r["b"] - lo, 3), "qwen_in_clip": round(r["q"] - lo, 3),
            # Independent evidence: neither aligner sees the energy VAD or these levels.
            "wt_db": round(_rms_db(wav, r["b"]), 1), "qwen_db": round(_rms_db(wav, r["q"]), 1),
            "wt_in_silence": inb, "qwen_in_silence": inq,
            "left": r["left"], "right": r["right"],
        })
    (root / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(index)} 个分歧边界写入 {root}（index.json 含两个候选时刻、局部电平、是否落在 VAD 静音里）")


def dump_ab(allrows, outdir: str, thresh: float) -> None:
    """Two clips per disagreement, each *ending* at one candidate boundary.

    Easier to judge than one clip with two marked instants: the listener only has to say which
    recording stops where the left-hand text stops, which is a judgement ears make well. Which of
    A/B is whose is recorded only in the answer key, so the listening is blind.
    """
    import random

    import soundfile as sf

    from .common import TARGET_SR, load_audio_16k

    # Both clips start at the same instant and differ only in where they stop. Giving each clip its
    # own lead (start = its own end - LEAD) made the earlier candidate's clip start earlier too, so
    # the pair differed at the *beginning* as well — listeners reported the left context being
    # truncated rather than hearing a clean difference in the ending.
    LEAD = 2.0
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    picked = sorted([(c, r) for c, r in allrows if abs(r["d"]) >= thresh],
                    key=lambda x: -abs(x[1]["d"]))
    cache: dict[str, object] = {}
    sheet = ["# 对齐器 A/B 听辨表", "",
             "每条两段音频。**它们都应该在「左侧文字」说完的瞬间结束。**",
             "听 A 和 B，判断哪一段停在正确的位置：",
             "",
             "- 停早了 = 左侧最后一个音被切掉、听起来没说完；",
             "- 停晚了 = 已经带进了右侧的下一个音。",
             "",
             "在「你的答案」列填 A / B / 都不对 / 分不出。A、B 哪个是哪家已随机打乱，答案在 key.json。",
             "",
             "| # | 左侧（应完整听到） | 右侧（不该听到） | A | B | Δ | 你的答案 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    key = []
    rnd = random.Random(0)
    for n, (clip, r) in enumerate(picked, 1):
        meta = json.loads((EXPLORE_OUT / f"{clip}{PAIRED}").read_text(encoding="utf-8"))["metadata"]
        if clip not in cache:
            cache[clip] = load_audio_16k(meta["audio"])
        wav = cache[clip]
        pair = [("wt", r["b"]), ("qwen", r["q"])]
        rnd.shuffle(pair)
        start = min(r["b"], r["q"]) - LEAD
        for tag, (who, t) in zip("AB", pair):
            a, b = max(0, int(start * TARGET_SR)), min(len(wav), int(t * TARGET_SR))
            sf.write(str(root / f"{n:03d}{tag}.wav"), wav[a:b], TARGET_SR)
            key.append({"n": n, "slot": tag, "aligner": who, "t": round(t, 3), "clip": clip})
        sheet.append(f"| {n} | …{r['left']} | {r['right']}… | {n:03d}A.wav | {n:03d}B.wav | "
                     f"{abs(r['d']):.2f}s |  |")
    (root / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "听辨表.md").write_text("\n".join(sheet), encoding="utf-8")
    print(f"\n{len(picked)} 条 × 2 段音频写入 {root}；听辨表.md 是工作表，key.json 是答案对照（先别看）")


if __name__ == "__main__":
    main()
