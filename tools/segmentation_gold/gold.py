"""Human-labelled segmentation gold set: worksheet generation, validation, scoring.

Why this exists: every segmentation number in this repo so far is a *mechanical* verdict, and
two rounds of adjudication put those verdicts at 58-67% precision with a ~8% blind spot in the
largest stratum (tools/qwen3_explore/FINDINGS.md §4.6). A mechanical metric cannot see semantics,
so it has to be calibrated against something that can. This is that something.

Labels carry both a **time** and a **substrate word index**, and `score` reads either:

- time (default, τ swept) scores anything with cue times, including a delivered `.srt`, but it
  also measures alignment precision — the arm that supplied the substrate matches at |Δt| ≈ 0 by
  construction, and merely re-timing cues moves the score by 20 points;
- `--by-index` scores the *decision* alone. Exact, no tolerance. Clean when the segmenter runs on
  the substrate's own transcript; for a different transcript it maps through a character alignment
  that only covers 75-92%, which trades the timing confound for a text one rather than removing it.

    prepare   freeze a labelling worksheet for one window (this is what a labeller works from)
    validate  schema + consistency check on saved label files
    score     score any segmentation against the gold set, and audit the mechanical metric

Methodology, rubric and rationale: docs/segmentation-gold.md. Do not change the label vocabulary
or the candidate rule without reading it — both are load-bearing for the denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

LABELS = ("must", "ok", "never", "unknown")

# `never` is the **default**, not a label you have to write. The labeller's contract is to declare
# every position that is `must` or `ok` (plus `unknown` where they cannot tell); everything else in
# the window is a no-cut position by definition. A default that turns out wrong is a labelling
# omission, not a metric flaw — which is what makes the contract well-formed rather than an
# approximation to tolerate. Declaring `never` explicitly is still allowed and useful as recorded
# evidence, and scoring keeps the two apart, but it is never required.
#
# Consequence: `unknown` must be expressible as a **span**, or unreadable material (yingtao-class
# ASR garbage) would default to "no cut allowed anywhere" and penalise a segmenter for cutting
# inside noise.

# A position is a *candidate* if there is any acoustic, lexical or reference signal at it.
# Deliberately arm-neutral: it must not depend on where the segmenters under test happen to cut,
# or the gold set would only ever be able to judge today's arms.
#
# The lexical rule (left word ends in sentence punctuation) is what makes **pause-free turn
# boundaries** reachable at all. Without it the rule was acoustic-only plus the reference
# subtitle, so `…って言ってた|でも今日はさ…` — two turns with no breath between them, the very
# example the rubric gives for `must` — could only enter if the reference happened to cut there,
# which caps `must` recall at the quality of a machine-produced subtitle.
#
# It is still a floor, not a ceiling: a labeller may add positions the rule missed (schema `k`).
# Scoring reports coverage, so a segmenter cutting outside the candidate set shows up as "未覆盖"
# rather than silently scoring clean.
CAND_PAUSE = 0.10
CAND_PUNCT = "。．｡.!！?？…‥、，,"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_words(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "word": str(w.get("word", "")),
            "punct": str(w.get("trailing_punct", "") or "") or str(w.get("word", ""))[-1:],
            "start": float(w["start"]),
            "end": float(w["end"]),
        }
        for seg in data["segments"]
        for w in (seg.get("words") or [])
        if str(w.get("word", "")).strip()
    ]


_SRT_TIME = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d+)")


def _t(m: re.Match) -> float:
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")[:3]) / 1000


def load_cues(path: Path) -> list[dict]:
    """Cue list from either an SRT or any `{"segments": [{start, end, text}]}` JSON."""
    if path.suffix.lower() == ".srt":
        out = []
        for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig")):
            times = _SRT_TIME.findall(block)
            lines = [ln for ln in block.strip().splitlines() if ln.strip()]
            if len(times) >= 2 and len(lines) >= 3:
                ms = list(_SRT_TIME.finditer(block))
                out.append({"start": _t(ms[0]), "end": _t(ms[1]), "text": " ".join(lines[2:])})
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text") or "")}
        for s in data["segments"]
        if str(s.get("text") or "").strip()
    ]


def candidates(words, non_speech, window, ref_cuts, alt_pauses=()) -> list[dict]:
    a, b = window
    pos = []
    for k in range(len(words) - 1):
        left, right = words[k], words[k + 1]
        t = (left["end"] + right["start"]) / 2
        if not (a <= t <= b):
            continue
        pause = max(0.0, right["start"] - left["end"])
        vad = max(
            (min(e, right["start"] + 0.05) - max(s, left["end"] - 0.05)
             for s, e in non_speech
             if min(e, right["start"] + 0.05) - max(s, left["end"] - 0.05) > 0),
            default=0.0,
        )
        punct = any(ch in CAND_PUNCT for ch in left["punct"])
        pos.append({"k": k, "t": t, "pause": pause, "vad": vad, "punct": punct})

    keep = {p["k"] for p in pos if p["pause"] >= CAND_PAUSE or p["vad"] > 0 or p["punct"]}
    # The reference subtitle is re-timed, so its cue times do not land on word boundaries. Snap
    # each one to its single nearest position instead of taking every position within a window —
    # the latter drags in mid-word positions no segmenter would ever cut (`キュ|ンポイント`).
    for c in ref_cuts:
        near = [p for p in pos if abs(p["t"] - c) <= 0.30]
        if near:
            keep.add(min(near, key=lambda p: abs(p["t"] - c))["k"])

    # A second aligner's pause observations, snapped the same way. Symmetric and therefore still
    # arm-neutral: this uses the aligners' *acoustic observations*, never their cut *decisions*.
    # It matters because the primary substrate is DTW-smeared — a real 0.15 s pause can read as
    # 0.05 s there and drop out, which measured as ~1/3 of the Qwen arm's cuts landing outside the
    # candidate set. Those are exactly the positions where the default-`never` backstop below is
    # most likely to be wrong, so converting them into human judgements is well spent.
    for c in alt_pauses:
        near = [p for p in pos if abs(p["t"] - c) <= 0.15]
        if near:
            keep.add(min(near, key=lambda p: abs(p["t"] - c))["k"])

    out = [
        {"k": p["k"], "t": round(p["t"], 2), "pause": round(p["pause"], 2),
         "vad": round(p["vad"], 2), "punct": "标" if p["punct"] else ""}
        for p in pos
        if p["k"] in keep
    ]
    for i, c in enumerate(out, start=1):
        c["i"] = i
    return out


def cmd_prepare(args) -> None:
    words = load_words(Path(args.words))
    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))
    non_speech = [(float(s), float(e)) for s, e in vad["non_speech"]]
    a, b = (float(x) for x in args.window.split(","))

    ref = load_cues(Path(args.reference)) if args.reference else []
    ref_cuts = [c["start"] for c in ref] + [c["end"] for c in ref]
    alt = []
    if args.words2:
        w2 = load_words(Path(args.words2))
        alt = [(w2[k]["end"] + w2[k + 1]["start"]) / 2 for k in range(len(w2) - 1)
               if w2[k + 1]["start"] - w2[k]["end"] >= CAND_PAUSE]
    cands = candidates(words, non_speech, (a, b), ref_cuts, alt)

    ctx = lambda k, n: "".join(w["word"] for w in words[max(0, k - n + 1) : k + 1])  # noqa: E731
    rgt = lambda k, n: "".join(w["word"] for w in words[k + 1 : k + 1 + n])  # noqa: E731

    L = [
        f"# 分割点标注工作表 — {args.clip} 窗口 [{a:.0f}, {b:.0f}] s",
        "",
        f"- 词流底本 substrate_path: `{args.words}`  (sha {_sha(Path(args.words))})",
        f"- 参考文本: `{args.reference or '(无)'}`",
        f"- 候选位: {len(cands)} 个（停顿 ≥{CAND_PAUSE}s / VAD 有静音 / 左词带句读 / 参考字幕换条"
        + ("/ 第二底本有停顿）" if args.words2 else "）"),
        "- **未列入本表的词间位置，打分时一律按「推定禁切」处理**——两个底本都说没有停顿、",
        "  没有 VAD、没有标点的地方，切下去几乎一定是错的。所以你不需要标满全部词间位置。",
        "  但推定归推定，打分时与你人标的 never 分列，不合成一个数。",
        "",
        "标注规范见 `docs/segmentation-gold.md`。**窗口内每个候选位都要给标签**，",
        "否则禁切违反率没有分母。工作表冻结后不要改动，标签按 `i` 引用。",
        "",
        "## 参考文本（已抹掉分条边界，只用于确认「说了什么」，不要照抄它的断句）",
        "",
    ]
    flow = " ".join(c["text"].strip() for c in ref if a - 5 <= c["start"] <= b + 5)
    for j in range(0, len(flow), 200):
        L.append(f"> {flow[j : j + 200]}")
    L += ["", "## 候选位（每个都必须给标签）", "",
          "| i | k | t (s) | 停顿 | VAD | 标点 | 左侧上文 | ⋯切在这里⋯ | 右侧下文 |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in cands:
        L.append(
            f"| {c['i']} | {c['k']} | {c['t']:.2f} | {c['pause']:.2f} | {c['vad']:.2f} | "
            f"{c['punct']} | …{ctx(c['k'], 6)} | ✂ | {rgt(c['k'], 6)}… |"
        )
    L += [
        "",
        "## 全窗口连续词流（判断语义时看这个）",
        "",
        "每行行首 `[k=N]` 是该行**第一个词**的序号，往后逐词递增。若你发现上表漏掉了一个真正",
        "该标 `must` 或 `never` 的位置，数出它左侧那个词的 k，用 `{\"k\": N, ...}` 补进 items ——",
        "候选位规则是**下限不是上限**，补位不限标签。",
        "",
        "**最该补的是 `never`**：候选位靠「有边界信号」筛选，而词中/短语中位置按定义没有信号",
        "（`ア|ラン`、`死|ねば` 停顿 0、无 VAD、无标点），结构性地进不了上表。`⟨x s⟩` 是停顿。",
        "",
    ]
    line, first_k, last_end = [], None, None
    for k, w in enumerate(words):
        if not (a - 2 <= w["start"] <= b + 2):
            continue
        if first_k is None:
            first_k = k
        if last_end is not None and w["start"] - last_end > 0.6:
            line.append(f" ⟨{w['start'] - last_end:.1f}s⟩ ")
        line.append(w["word"])
        last_end = w["end"]
        if k - first_k >= 14:
            L.append(f"`[k={first_k}]` {''.join(line)}")
            line, first_k = [], None
    if line:
        L.append(f"`[k={first_k}]` {''.join(line)}")
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"{args.out}: {len(cands)} 个候选位, 词流 {sum(1 for w in words if a <= w['start'] <= b)} 词")


def cmd_validate(args) -> None:
    ok = True
    for p in sorted(Path(args.dir).glob("*.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        errs = []
        for f in ("clip", "window", "substrate_path", "substrate_sha", "labeler", "date", "regime", "items"):
            if f not in g:
                errs.append(f"缺字段 {f}")
        seen = set()
        a, b = g.get("window", (0, 0))

        # Items the labeller added (schema `k`, no `i`) carry no time yet — the worksheet only
        # prints `t` for rule-generated candidates. Resolve them from the substrate and write back,
        # echoing context so the addition can actually be reviewed rather than taken on trust.
        added = [it for it in g.get("items", []) if "t" not in it and "span" not in it]
        if added:
            sub = Path(g.get("substrate_path", ""))
            if not sub.exists():
                errs.append(f"{len(added)} 条补标项需要 substrate_path 指向词流底本才能解析 k")
            else:
                words = load_words(sub)
                for it in added:
                    k = it.get("k")
                    if not isinstance(k, int) or not 0 <= k < len(words) - 1:
                        errs.append(f"补标项 k={k} 越界")
                        continue
                    it["t"] = round((words[k]["end"] + words[k + 1]["start"]) / 2, 2)
                    lf = "".join(w["word"] for w in words[max(0, k - 5) : k + 1])
                    rt = "".join(w["word"] for w in words[k + 1 : k + 6])
                    print(f"  + 补标 k={k} t={it['t']} [{it.get('label')}] …{lf} ✂ {rt}…")

        # The reverse: rule-generated candidates carry `t` but not `k`, and word-index scoring
        # (`score --by-index`) needs `k` on every item. Backfilled here so the label files stay the
        # single source and nothing has to re-derive it from the worksheet later.
        need_k = [it for it in g.get("items", []) if "k" not in it and "span" not in it]
        need_k += [it for it in g.get("items", []) if "span" in it and "span_k" not in it]
        if need_k:
            sub = Path(g.get("substrate_path", ""))
            if not sub.exists():
                # Not an error: `k` is backfilled by whoever holds the substrate, and a labeller
                # working from a frozen worksheet legitimately has neither the substrate nor any
                # need for `k`. Only `score --by-index` requires it.
                print(f"  · {len(need_k)} 条缺 k（底本不在本机，稍后在有底本的环境补）")
            else:
                words = load_words(sub)
                mids = [(words[k]["end"] + words[k + 1]["start"]) / 2 for k in range(len(words) - 1)]
                near = lambda t: min(range(len(mids)), key=lambda k: abs(mids[k] - t))  # noqa: E731
                for it in need_k:
                    if "span" in it:
                        x, y = it["span"]
                        it["span_k"] = [near(float(x)), near(float(y))]
                    else:
                        it["k"] = near(float(it.get("t", -1)))
        if added or need_k:
            p.write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")

        for it in g.get("items", []):
            if it.get("label") not in LABELS:
                errs.append(f"i={it.get('i')} 标签非法 {it.get('label')!r}")
            if it.get("i") is not None:
                if it["i"] in seen:
                    errs.append(f"i={it['i']} 重复")
                seen.add(it["i"])
            if "span" in it:
                x, y = it["span"]
                if it["label"] != "unknown":
                    errs.append(f"span 形式只允许 unknown，收到 {it['label']!r}")
                elif not (a <= x < y <= b):
                    errs.append(f"span {it['span']} 不在窗口内或首尾颠倒")
            elif not (a <= float(it.get("t", -1)) <= b):
                errs.append(f"i={it.get('i')} t={it.get('t')} 不在窗口内")
            if it.get("label") in ("must", "never", "unknown") and not str(it.get("why", "")).strip():
                errs.append(f"i={it.get('i')} {it['label']} 必须写 why")
        n = sum(1 for it in g.get("items", []) if "i" in it)
        cov = g.get("candidates_total")
        if cov and n < cov:
            errs.append(f"候选表没走完: 标了 {n}/{cov}。候选表是清单，走完它才能保证没漏掉带信号的位置")
        print(f"{p.name}: {n} 条" + ("  OK" if not errs else "\n  - " + "\n  - ".join(errs)))
        ok &= not errs
    raise SystemExit(0 if ok else 1)


def _cuts(cues: list[dict]) -> list[tuple[float, float, float]]:
    """Each inter-cue boundary as (gap_start, gap_end, midpoint)."""
    return [(p["end"], c["start"], (p["end"] + c["start"]) / 2) for p, c in zip(cues, cues[1:])]


# --------------------------------------------------------------- word-index anchoring
# Time anchoring measures alignment precision as much as segmentation: the arm that supplied the
# substrate matches at |Δt| ≈ 0 by construction, others pay a tolerance tax (0.07-0.33 s for the
# Qwen arm, same order as τ), and merely re-timing cues — `snap_cues_to_speech` moving a boundary
# from the middle of a silence to its edge — drops an otherwise *word-identical* segmentation from
# 95.8% to 75.0% recall. None of that is a segmentation decision.
#
# Anchoring on the substrate's word index removes all three at once: a cut is identified by *which
# junction* it sits at, so timestamps stop mattering entirely. Arms with different text (Qwen ASR)
# are mapped through a character-level alignment of their transcript to the substrate's.
_STRIP = "。．｡.!！?？…‥、，,､；;：:「」『』（）()　 \t\n・-–—"


def _stripped(text: str) -> str:
    return "".join(ch for ch in text if ch not in _STRIP)


def word_junctions(words: list[dict]) -> tuple[str, list[int]]:
    """Substrate as a punctuation-free char stream + the char offset of each word junction.

    `ends[k]` is the offset just past word `k`, i.e. the position of the junction between word `k`
    and word `k+1` — the same `k` the gold labels use.
    """
    text, ends = "", []
    for w in words:
        text += _stripped(w["word"])
        ends.append(len(text))
    return text, ends


def cuts_to_indices(cues: list[dict], words: list[dict]) -> list[int]:
    """Substrate word indices this segmentation cuts after. Text-anchored, time-free.

    Returns `k` for each inter-cue boundary, meaning "cut between substrate word k and k+1".
    """
    import difflib

    sub_text, ends = word_junctions(words)
    arm_text, at = "", []
    for c in cues:
        arm_text += _stripped(str(c.get("text") or ""))
        at.append(len(arm_text))
    at = at[:-1]  # the final cue end is not a cut

    # Map arm char offsets onto substrate char offsets. Identical transcripts give the identity
    # map; different ones (Qwen ASR vs Whisper ASR) go through the longest-matching-block chain,
    # which is what makes this usable across word streams at all.
    blocks = difflib.SequenceMatcher(None, arm_text, sub_text, autojunk=False).get_matching_blocks()
    out = []
    for pa in at:
        ps = None
        for a, b, size in blocks:
            if a <= pa <= a + size:
                ps = b + (pa - a)
                break
        if ps is None:  # inside a replaced stretch: take the nearest block edge
            ps = min((b + (0 if pa < a else size) for a, b, size in blocks if size),
                     key=lambda x: abs(x - pa), default=0)
        # nearest junction at or before ps; bisect over the monotone `ends`
        lo, hi = 0, len(ends) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ends[mid] <= ps:
                lo = mid
            else:
                hi = mid - 1
        out.append(lo)
    return out


def evaluate_by_index(golds: list[dict], cut_idx: list[int]) -> dict:
    """Same contract as `evaluate`, keyed on word index instead of time. No tolerance."""
    cuts = set(cut_idx)
    r = dict(must=0, must_hit=0, declared=0, presumed=0, in_window=0, skipped=0)
    for g in golds:
        items = [it for it in g["items"] if isinstance(it.get("k"), int)]
        lo = min((it["k"] for it in items), default=0)
        hi = max((it["k"] for it in items), default=0)
        fine = {it["k"] for it in items if it["label"] in ("must", "ok")}
        never = {it["k"] for it in items if it["label"] == "never"}
        unk = {it["k"] for it in items if it["label"] == "unknown"}
        spans = [tuple(it["span_k"]) for it in g["items"] if "span_k" in it]
        for it in g["items"]:
            if it["label"] == "must" and isinstance(it.get("k"), int):
                r["must"] += 1
                r["must_hit"] += it["k"] in cuts
        for k in cuts:
            if not (lo <= k <= hi):
                continue
            r["in_window"] += 1
            if k in unk or any(x <= k <= y for x, y in spans):
                r["skipped"] += 1
            elif k in fine:
                pass
            elif k in never:
                r["declared"] += 1
            else:
                r["presumed"] += 1
    judged = r["in_window"] - r["skipped"]
    r["recall"] = r["must_hit"] / r["must"] if r["must"] else float("nan")
    r["viol_per100"] = 100 * (r["declared"] + r["presumed"]) / judged if judged else 0.0
    return r


def _hit(t: float, cuts, tau: float) -> bool:
    return any(s - tau <= t <= e + tau or abs(t - m) <= tau for s, e, m in cuts)


def load_golds(gold_dir: str | Path, clip: str | None = None) -> list[dict]:
    golds = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(gold_dir).glob("*.json"))]
    return [g for g in golds if clip is None or g["clip"] == clip]


def evaluate(golds: list[dict], cuts, tau: float) -> dict:
    """The scoring contract, in one place: production `score`, sweeps and any future driver.

    `cuts` is `_cuts(cues)`. Counts pool over all `golds`; per-window numbers come from calling
    this once per window.
    """
    r = dict(must=0, must_hit=0, declared=0, presumed=0, in_window=0, skipped=0)
    for g in golds:
        a, b = g["window"]
        items = g["items"]
        spans = [tuple(it["span"]) for it in items if it["label"] == "unknown" and "span" in it]
        unk = [it["t"] for it in items if it["label"] == "unknown" and "span" not in it]
        # Positions where cutting is asserted not to be an error.
        fine = [it["t"] for it in items if it["label"] in ("must", "ok")]
        said_never = [it["t"] for it in items if it["label"] == "never"]

        for it in items:
            if it["label"] == "must":
                r["must"] += 1
                r["must_hit"] += _hit(it["t"], cuts, tau)

        for _cs, _ce, m in cuts:
            if not (a <= m <= b):
                continue
            r["in_window"] += 1
            if any(x <= m <= y for x, y in spans) or any(abs(m - t) <= tau for t in unk):
                r["skipped"] += 1
            elif any(abs(m - t) <= tau for t in fine):
                pass
            elif any(abs(m - t) <= tau for t in said_never):
                r["declared"] += 1
            else:
                r["presumed"] += 1
    judged = r["in_window"] - r["skipped"]
    r["recall"] = r["must_hit"] / r["must"] if r["must"] else float("nan")
    r["viol_per100"] = 100 * (r["declared"] + r["presumed"]) / judged if judged else 0.0
    return r


def _score_by_index(args, cues) -> None:
    golds = load_golds(args.gold, args.clip)
    if not golds:
        raise SystemExit("没有匹配的 gold 窗口")
    subs = {g["substrate_path"] for g in golds}
    if len(subs) > 1:
        raise SystemExit("--by-index 一次只能评一个 clip（底本不同）")
    words = load_words(Path(subs.pop()))
    e = evaluate_by_index(golds, cuts_to_indices(cues, words))
    recall = f"{100 * e['recall']:5.1f}% ({e['must_hit']}/{e['must']})"
    print(f"{'必切召回':>18} | {'人标禁切':>8} {'推定禁切':>8} {'合计/百条':>9} | {'窗内刀':>6} {'存疑内':>6}")
    print(f"{recall:>18} | {e['declared']:8d} {e['presumed']:8d} {e['viol_per100']:9.2f} | "
          f"{e['in_window']:6d} {e['skipped']:6d}")
    print(
        "\n按底本**词序号**匹配，无容差：时间戳完全不参与，重定时（snap）、对齐精度差异、"
        "\n底本优势三者都不影响读数。"
        "\n换了 ASR 文本的臂（Qwen）经字符级对齐映射，只有 75–92% 的字符能映上，"
        "\n那部分读数把时间混淆换成了文本对齐混淆——不是干净的，见 docs/segmentation-gold.md §2.1。"
    )


def cmd_score(args) -> None:
    cues = load_cues(Path(args.seg))
    if args.by_index:
        return _score_by_index(args, cues)
    cuts = _cuts(cues)
    tau_list = [float(x) for x in args.tau.split(",")]

    golds = load_golds(args.gold, args.clip)
    if not golds:
        raise SystemExit("没有匹配的 gold 窗口")

    print(f"{'τ':>5s} | {'必切召回':>18s} | {'人标禁切':>8s} {'推定禁切':>8s} {'合计/百条':>9s} | "
          f"{'窗内刀':>6s} {'存疑内':>6s}")
    for tau in tau_list:
        e = evaluate(golds, cuts, tau)
        must, must_hit = e["must"], e["must_hit"]
        declared, presumed = e["declared"], e["presumed"]
        in_window, skipped = e["in_window"], e["skipped"]
        f = lambda n, d: f"{100 * n / d:5.1f}% ({n}/{d})" if d else "     —      "  # noqa: E731
        print(
            f"{tau:5.2f} | {f(must_hit, must):>18s} | {declared:8d} {presumed:8d} "
            f"{e['viol_per100']:9.2f} | {in_window:6d} {skipped:6d}"
        )

    # Timing-vs-decision separator. Gold `t` values are computed from the *substrate's* word
    # timings, so the arm that supplied the substrate matches at |Δt| ≈ 0 by construction while
    # every other arm pays a tolerance tax for its own alignment noise. Without this line a low
    # 必切召回 reads as "missed the boundary" when it may be "cut it, 0.25 s late".
    off, n_must = [], 0
    for g in golds:
        for it in g["items"]:
            if it["label"] != "must":
                continue
            n_must += 1
            near = [m - it["t"] for _, _, m in cuts if abs(m - it["t"]) <= 1.0]
            if near:
                off.append(min(near, key=abs))
    if off:
        med = sorted(abs(x) for x in off)[len(off) // 2]
        sgn = sorted(off)[len(off) // 2]
        print(
            f"\n对齐偏移诊断：{len(off)}/{n_must} 个必切位在 ±1.0 s 内有刀，最近刀 |Δt| 中位 "
            f"{med:.3f} s（有向中位 {sgn:+.3f} s）。"
            "\n  提供底本的那一臂此值 ≈0，是构造出来的优势。若 |Δt| 中位与 τ 同量级，"
            "上表的必切召回测的是对齐精度而非分割决策——跨臂比较必须连这一行一起引。"
        )
    print(
        "\n必切召回 = 该切的切了多少（漏 = 欠切，下游 merge 不可救）"
        "\n禁切违反 = 切在了未被声明为 must/ok 的位置上。**never 是默认**：标注者的义务是声明"
        "\n           全部 must 与 ok，其余按定义即禁切，所以默认判错是漏标而不是指标缺陷。"
        "\n  人标 = 标注者显式写了 never（有 why，是证据）"
        "\n  推定 = 由默认得出（是推断）。两者永远分列；推定项占比高说明该回头补标或补候选规则。"
        "\n存疑内 = 落在 unknown 区间/位置里的刀，整条排除，不计入任何分子分母。"
    )
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="冻结一个窗口的标注工作表")
    p.add_argument("--clip", required=True)
    p.add_argument("--words", required=True, help="词流底本 aligned.json")
    p.add_argument("--vad", required=True)
    p.add_argument("--window", required=True, metavar="A,B")
    p.add_argument("--words2", help="第二底本 aligned.json，只取它的停顿观测并进候选")
    p.add_argument("--reference", help="参考文本 srt/json，仅用于确认内容")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("validate", help="校验已保存的标签文件")
    p.add_argument("--dir", default=str(Path(__file__).parent / "labels"))
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("score", help="用 gold 给任意分割结果打分")
    p.add_argument("--gold", default=str(Path(__file__).parent / "labels"))
    p.add_argument("--seg", required=True, help="待评分的 srt 或 {segments:[...]} json")
    p.add_argument("--clip", help="只用该 clip 的窗口")
    p.add_argument("--tau", default="0.15,0.30,0.50", help="容差扫描，逗号分隔")
    p.add_argument("--by-index", action="store_true",
                   help="按底本词序号打分而非时间：无容差，不受重定时与对齐精度影响。"
                        "同文本的臂精确映射；换了 ASR 文本的臂经字符对齐，覆盖率见输出")
    p.set_defaults(fn=cmd_score)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
