# ASR 稳定化

`asr-stabilize` 是 VAD-ASR 对齐之后、raw SRT 和 LLM 消费之前的独立 stage：读取
`*-aligned.json`，按 profile 清理或标记 ASR segment，输出 `*-stable.json`。源码和 CLI
入口为 `src/asr_stabilize.py`。

```text
*-aligned.json -> ASR stabilization -> *-stable.json
```

## CLI 与 pipeline

```powershell
python src/asr_stabilize.py out/input/input-aligned.json `
  -o out/input/input-stable.json --profile 0

python src/pipeline.py data/input.wav --stage stable --asr-stabilize-profile 0
```

独立 CLI 参数为 `--profile {-1,0,1,2,3}`；pipeline 和 batch 对应
`--asr-stabilize-profile`，默认均为 `0`。这与最终中文字幕 SRT 的
`--postprocess-profile` 是两套互不相关的 profile。

pipeline 顺序为：

```text
vocal -> aligned -> stable -> raw-srt -> translated-srt -> final-srt
```

## Profiles

### Profile -1：原样复制

输入仍会执行 JSON/schema 校验，但不会解析后重写输出；校验成功时，`stable.json` 与输入 aligned 保持字节一致。

### Profile 1：常见套话幻觉清理

在 segment 的 `words` 拼接文本中查找精确的
`ご視聴ありがとうございました`。一次匹配只有在组成它的 word 不超过 5 个时才处理：

- 删除匹配字符及其后直接相连的 Unicode 标点；
- 完全变空的 word 删除，部分命中的 word 保留剩余文本，例如
  `ありがとうございました!ではまた` 保留为 `ではまた`；
- 根据剩余 words 重建 `text`；只有首/尾 word 被删空时才把 segment `start`/`end`
  收缩到新的首/尾 word，中间删除不改变外边界；
- 部分 word 保留原时间，segment/word confidence、`no_speech_prob`、energy 等诊断值不重算；
- 没有剩余 word 时删除整个 segment；没有 word-level 数据的 segment 不处理。

同一 segment 内所有符合条件且不重叠的精确匹配都会清理。

### Profile 2：高噪音标记

输出 tag 放在 segment 的 `tags: string[]`；按下列固定顺序追加且不重复，无 tag 时省略字段：

1. `高度疑似幻觉`
2. `高度疑似语气填充词`
3. `时间漂移`

指标定义：

```text
duration = end - start
rate = (weighted_char_count(text) - 2) / duration
high_speed = rate > 20

weighted_word_confidence =
  sum(weighted_char_count(word) * word.confidence)
  / sum(weighted_char_count(word))

low_conf = segment.confidence < 0.3
           and weighted_word_confidence < 0.3
low_energy = vad_weighted_energy_db < 0
very_low_energy = vad_weighted_energy_db < -20
```

weighted word confidence 只纳入具有有限数值 confidence 且权重大于 0 的 word。
`weighted_char_count` 的共享口径为：拉丁字母、Unicode 数字、标点和空格计 `0.5`，
其他可见字符计 `1`，组合符和不可见控制/格式字符计 `0`。“去标点后字数”先删除 Unicode
category `P*`，再使用同一加权公式。

判定按三个独立 `if` 执行，因此一个 segment 可以获得多个 tag：

```text
if (duration > 0.1 and very_low_energy)
   or (去标点后字数 <= 2 and very_low_energy)
   or (low_conf and low_energy):
    tag += 高度疑似幻觉

if low_conf and energy存在 and not low_energy and 去标点后字数 <= 2:
    tag += 高度疑似语气填充词

if high_speed or low_conf or low_energy:
    tag += 时间漂移
```

阈值全部是上述严格比较。缺失、非数值或非有限 confidence/energy 不命中依赖该指标的
条件；但 `low_conf` 本身仍会命中“时间漂移”。

### Profile 3：确定性预合并

调用 `src/premerge.py`（词形/词典强证据的词中接回 + 方向化语气词附着，规则、阈值、
过拟合与日语特化警告见该模块 docstring 与 `docs/llm_harness_behavior.md` 预合并节）。
要点：

- 合并只在强证据下发生（E1/E2 表面签名 ≤1.0s gap：E1 右侧以小假名/促音/长音/ん
  开头，E2 右侧归一化为单假名且非独立感叹词）；
  形状护栏 ≤7s / ≤36 加权字 / ≤3 源。
- split（aligned 阶段）在切点新段打 segment 级 `splitted_before` tag；预合并对右侧带
  该 tag 的交界**结构性拒绝**（tag 是段自身的起源描述，左邻被删后仍然有效且拒绝仍正确
  ——真实的词不可能跨越一个被删除的中间段）。
- 合并语义：text 拼接、span 并集、words 顺接且**被并入侧首 word 打 `premerge_before`
  word 级 tag**（交界位置随之保留在产物内）、confidence 取 min、两侧 segment tags 并集
  （被并入侧的 `splitted_before` 除外，它描述的位置已成段内部）、`premerge_sources`
  记录输入位置供审计。
- 输出 `metadata.premerge`（含 `rules_version` 与阈值/词表快照）；规则变更即产物语义
  变更，按惯例删 stable 及下游重跑。report 增加 `premerge_rejoined` /
  `premerge_filler_attached`。

### Profile 0：默认稳定化

依次执行 profile `1 -> 3 -> 2`，随后删除带 `高度疑似幻觉` 或
`高度疑似语气填充词` 的 segment。只带 `时间漂移` 的 segment 保留。

**为什么 3 在 2 之前**（2026-07-19 语料对比，8BV+yui 的 aligned 与旧 stable 双语料）：
词中切断的碎片天然低置信（如 `次はキッ|と` 的 `と` conf 0.089、能量为负），先跑 profile 2
会把它标为幻觉丢弃，词永久残缺；先预合并则拼回 `次はキッと` 后不再命中丢弃条件。两套语料
上「3 先」均多救回 1 处合并、少丢 1 段，且没有出现「并入后整段被丢」的反向损失。代价是
profile 2 在合并段上用的能量/静音诊断值继承自左半段（未重算），目前未观察到误判。

## Schema 与复用

- aligned 的 schema 与此前未稳定化的 stable schema 相同，包含 `segments` 和原
  `metadata.vad` / `metadata.asr_align`；aligned 侧 split 产生的段可带
  `tags: ["splitted_before"]`，word 可带 `premerge_before: true`（profile 3 写入）。
- 稳定化保留未知顶层字段、metadata 和未修改的 segment 字段；除 profile 3 写入的
  `metadata.premerge`（规则版本/阈值/词表快照）外不写额外 profile metadata。
- pipeline 只按输出是否存在复用：stable 已存在时不回补 aligned；aligned 已存在且 stable
  缺失时只跑稳定化；显式 `--stage aligned` 必须生成或复用 aligned。
- profile 改变不会自动使现有 stable 失效。要重跑需删除 stable 及全部下游 artifact。

## 验证

```powershell
python -m pytest -q test/test_asr_stabilize.py test/test_pipeline_refactor.py
```
