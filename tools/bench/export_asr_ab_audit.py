"""Export the ASR A/B disagreements as a blind listening task for an agent.

The third-model metric in `probe_asr_model_ab` answers "who is closer to what
Qwen heard". That is symmetric but it is not the question: it cannot tell an
orthography preference from a misheard word, and it inherits whatever Qwen got
wrong. This exports the same disagreements as clips plus a table, so a
multimodal agent can listen and say which transcript matches the audio.

Three decisions that make the result readable:

* **Blind, per item.** Which arm is 甲 and which is 乙 is drawn per row from a
  seeded RNG, so no arm has a stable identity even within one session -- a
  judge that drifts toward "甲 has been winning" cannot drift toward a model.
  The key is written OUTSIDE the directory the agent is given.
* **No Qwen transcript in the prompt.** Showing it would anchor the judge onto
  the very metric this exists to cross-check; it stays in the key file for the
  post-hoc comparison.
* **A category per verdict.** "Which is better" alone cannot separate 写法
  (both correct, different convention) from 听错 (one is wrong), and that
  separation is the whole reason to ask a listener rather than an edit
  distance.

Sampling is proportional by product and independent of Qwen's verdict: a
sample stratified on the metric under test would bake that metric's opinion
into the answer.

    PYTHONPATH="$(pwd)/src:$(pwd)" python -m tools.bench.export_asr_ab_audit \\
        --report tmp/bench/ja-ab/report.json --n 200 --batch 20
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from finesub.media.ffmpeg import containerize_audio_for_agy, _run_ffmpeg, resolve_ffmpeg  # noqa: E402

#: Seconds kept either side of the VAD interval. Enough that a word onset is
#: not clipped, small enough that the clip does not acquire speech neither arm
#: was asked to transcribe. The prompt tells the judge to ignore fragments at
#: the very edges.
PAD_SEC = 0.2

INSTRUCTIONS = """\
# 日语听写对照评审（第 {index} 批 / 共 {total} 批）

你要做的事：**逐条听音频，判断哪一份转写更符合实际说的话。**

## 规则

1. **每一条都必须先用 `view_file` 读那条音频**再下判断。只看文本不听音频的判断无效。
2. 音频是从更长的素材里按语音区间切出来的，**首尾各留了 0.2 秒**。如果开头或结尾有
   半个词的碎片，**忽略它**——两份转写都没被要求转它。
3. 甲和乙来自两个不同的语音识别模型，**每一条的甲乙是独立随机排的**，不要假设
   「甲一直是某个模型」。
4. `（无输出）` 表示该模型对这段音频什么都没转出来。如果这段确实没有可转写的语音
   （只有笑声、气声、噪声、静音），那么「无输出」是**正确**的；如果确实有话而它没转，
   那是漏识别。

## 判决

每条给一个 `裁决` 和一个 `类别`：

| 裁决 | 什么时候用 |
| --- | --- |
| `甲` / `乙` | 那一份明显更符合音频 |
| `平` | 两份都对，或差异不影响理解 |
| `都错` | 两份都没听对这段话 |

| 类别 | 含义 |
| --- | --- |
| `听错` | 有一份把词听成了别的词 |
| `漏字` | 有一份漏了实际说出来的内容（含该出声却「无输出」） |
| `多字` | 有一份多出了音频里没有的内容 |
| `幻觉` | 有一份在没有语音（或只有笑声/气声）的地方编出了话 |
| `写法` | 两份内容相同，只是假名/汉字/片假名/促音/标点等写法不同 |
| `其他` | 以上都不是，在理由里说明 |

**`写法` 与 `平` 的区别**：`写法` 说的是差异的性质，`平` 说的是谁更好。写法不同但都
正确 → 裁决 `平`、类别 `写法`。写法不同且其中一种明显不是这句话该有的写法 → 裁决
`甲`/`乙`、类别 `写法`。

## 回答格式

只回一个代码块，每行一条，用 `|` 分隔，**不要表头、不要多余文字**：

```
编号|裁决|类别|理由（不超过 25 字）
```

全部 {count} 条都要有一行，顺序不限。

## 本批条目

{items}
"""

ITEM = """\
### {id}
- 音频：`clips/{clip}`
- 甲：{a}
- 乙：{b}
"""


def cut(ffmpeg_bin: str, source: Path, start: float, end: float, out: Path) -> None:
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-nostdin",
            "-ss",
            f"{max(0.0, start - PAD_SEC):.3f}",
            "-t",
            f"{(end - start) + 2 * PAD_SEC:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="tmp/bench/ja-ab/report.json")
    parser.add_argument("--out", default="tmp/bench/ja-ab/audit")
    parser.add_argument("--key", default="tmp/bench/ja-ab/audit-key.json")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    arm_a, arm_b = report["arms"]
    rows = [
        {**v, "product": name, "product_dir": entry["product"]}
        for name, entry in report["files"].items()
        for v in entry.get("verdicts", [])
    ]
    if not rows:
        parser.error("report has no verdicts; run the adjudication phase first")

    rng = random.Random(args.seed)
    # Proportional by product, so one long stream cannot become the audit.
    by_product: dict[str, list[dict]] = {}
    for row in rows:
        by_product.setdefault(row["product"], []).append(row)
    sample: list[dict] = []
    for name, group in sorted(by_product.items()):
        take = max(1, round(args.n * len(group) / len(rows)))
        sample.extend(rng.sample(group, min(take, len(group))))
    rng.shuffle(sample)
    sample = sample[: args.n]

    out_dir = Path(args.out)
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = resolve_ffmpeg()

    key = []
    items_by_batch: list[list[str]] = []
    current: list[str] = []
    for index, row in enumerate(sample, start=1):
        item_id = f"S{index:03d}"
        product = Path(row["product_dir"])
        source = product / f"{product.name}-vocal.ogg"
        raw = clips_dir / f"{item_id}.wav"
        clip = clips_dir / f"{item_id}.mp4"
        cut(ffmpeg_bin, source, row["start"], row["end"], raw)
        containerize_audio_for_agy(raw, clip, ffmpeg=ffmpeg_bin)
        raw.unlink(missing_ok=True)

        first_is_a = rng.random() < 0.5
        left = row["a_text"] if first_is_a else row["b_text"]
        right = row["b_text"] if first_is_a else row["a_text"]
        blank = "（无输出）"
        current.append(
            ITEM.format(
                id=item_id,
                clip=clip.name,
                a=left.strip() or blank,
                b=right.strip() or blank,
            )
        )
        key.append(
            {
                "id": item_id,
                "product": row["product"],
                "start": row["start"],
                "end": row["end"],
                # Which arm sits in the 甲 slot for this row.
                "first": arm_a if first_is_a else arm_b,
                "second": arm_b if first_is_a else arm_a,
                "turbo_slot": "甲" if first_is_a else "乙",
                "text_turbo": row["a_text"],
                "text_ja": row["b_text"],
                "qwen_text": row["referee_text"],
                "qwen_winner": row["winner"],
                "qwen_sim_turbo": row["sim_a_referee"],
                "qwen_sim_ja": row["sim_b_referee"],
            }
        )
        if len(current) == args.batch:
            items_by_batch.append(current)
            current = []
    if current:
        items_by_batch.append(current)

    for index, batch in enumerate(items_by_batch, start=1):
        (out_dir / f"batch-{index:02d}.md").write_text(
            INSTRUCTIONS.format(
                index=index,
                total=len(items_by_batch),
                count=len(batch),
                items="\n".join(batch),
            ),
            encoding="utf-8",
        )

    Path(args.key).write_text(
        json.dumps(
            {"arms": [arm_a, arm_b], "seed": args.seed, "items": key},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    audio_sec = sum(row["end"] - row["start"] + 2 * PAD_SEC for row in sample)
    print(f"{len(sample)} items, {len(items_by_batch)} batches -> {out_dir}")
    print(f"key (not visible to the judge) -> {args.key}")
    print(
        f"clip audio {audio_sec / 60:.1f} min; rough token envelope "
        f"{int(audio_sec * 32 + len(sample) * 269):,} "
        "(32/s audio + 269 per single frame)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
