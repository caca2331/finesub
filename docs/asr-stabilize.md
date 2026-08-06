# ASR 稳定化

`asr-stabilize` 是 VAD-ASR 对齐之后、raw SRT 和 LLM 消费之前的独立 stage：读取
`*-aligned.json`，按 profile 清理或标记 ASR segment，输出 `*-stable.json`。源码和 CLI
实现位于 `src/asr_playground/speech/postprocessing/stabilization.py`，安装后入口为
`asr-stabilize`。

```text
*-aligned.json -> ASR stabilization -> *-stable.json
```

## CLI 与 pipeline

```powershell
asr-stabilize out/input/input-aligned.json `
  -o out/input/input-stable.json --profile 0

asr-pipeline data/input.wav --stage stable --asr-stabilize-profile 0
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

认识论口径（2026-08-05，用户确认）：这条精确短语删除并没有"原理上正确"的判别子——
它成立靠的是**先验赔率**：该短语的真实语音出现极罕见而幻觉极高频（全语料未见确认的
真实出现；曾疑似的 H6 PV 片尾两行后查明保留自 LLM 产物、相邻 `おわり` 经用户重听为
幻觉，此两行大概率同为幻觉）。`套话幽灵` 的语速判据同理，只是把赔率换成了物理
不可能性。幻觉的工程化判断到此为止；进一步压误删要靠**低幻觉第二模型校验**
（Qwen3 ASR 对嫌疑段重认，冒烟已验证），见 wt-refine-handoff P1。

### Profile 2：高噪音标记

输出 tag 放在 segment 的 `tags: string[]`；按下列固定顺序追加且不重复，无 tag 时省略字段：

1. `高度疑似幻觉`
2. `高度疑似语气填充词`
3. `套话幽灵`
4. `语言切换幻觉`
5. `时间漂移`

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
energy_exempt = weighted_word_confidence > 0.9
                and vad_weighted_energy_db > -80    # 不在测量地板上

if (not energy_exempt
    and ((duration > 0.1 and very_low_energy)
         or (去标点后字数 <= 2 and very_low_energy)))
   or (low_conf and low_energy):
    tag += 高度疑似幻觉

if low_conf and energy存在 and not low_energy and 去标点后字数 <= 2:
    tag += 高度疑似语气填充词

if high_speed or low_conf or low_energy:
    tag += 时间漂移
```

阈值全部是上述严格比较。缺失、非数值或非有限 confidence/energy 不命中依赖该指标的
条件；但 `low_conf` 本身仍会命中“时间漂移”。

`energy_exempt`（2026-08-05 新增）：对 H6dTZf9QFTY 全程人工字幕的删除审计发现，
very_low_energy 两条腿的实际误删形态是**时间轴坍缩/漂移的真实语音**——词被量化到 20ms
点或整段位移后，能量采样落在静音处，而 decoder 对每个词都高度自信（实测受害段词加权
置信 0.92–0.99、能量 −24～−68dB）。词置信严格高于 0.9 且能量不在 −100dB 地板附近时，
能量证据不再触发丢弃，段降级为 `时间漂移` 保留；地板条件挡住纯静音上的自信幻觉
（kaguya60 的 `音楽`×5，e=−100）。79 份产物回归：仅 H6 变化（挽回 5 段人工确认的真
内容，另保留 4 段同族歌词回声/软语气词）。高置信的复读与已知短语幻觉不受影响——
它们由词级规则与 profile 1 短语清理负责。

`套话幽灵`（2026-08-05 新增，**参与 profile 0 丢弃**）：整段就是 Whisper 惯用收尾套话
（`おわり` / `それではまた` / `ありがとうございました`，归一化后 ≤ 短语+2 字），且被压进
物理不可能的时长（>20 字/秒，即时长 < 短语字数×50ms）。全语料审计（74 份产物 +
400 窗 sweep + 参照对照）：该形态的每一次出现都是幻觉（含此前唯一漏网的
`yui:37` 残留），而确认的真实出现（H6dTZf9QFTY 直播收尾致谢，Qwen 双模型重认证实）
语速正常、距阈值 ~2 倍裕度。**confidence 不能分离真话与幻觉**（重叠区间 0.16-0.999），
故判据只用语速；更长的真句子只是包含套话时被整段长度上限排除。
出处更正（2026-08-05）：H6 的两处正常语速 `おわり` 最初被当作"人工保留的真话"反例，
后查明其保留来自 **LLM 纠错层产物**（H6 无人工字幕），用户重听确认**均为幻觉**——
即正常语速的套话幻觉真实存在，语速判据刻意不碰它们（无声学判别子）。
**第二模型证据补上这块**（同日落地）：带 `qwen_verify` 证据（vad-asr 尾部产出，
docs/vad-asr.md）的整段套话，若证据文本不含该短语（67 clip 标定 11/11；喊叫盲区
不影响多音节套话）也打 `套话幽灵` 丢弃；无证据时维持只删语速幽灵。
反向地，**证据 veto**：`高度疑似幻觉`/`高度疑似语气填充词` 两条噪声腿在 Qwen 听到
语音（证据文本非空）时不触发——丢弃审计实测它们在正能量上删过真实喊叫
（kaguya `あ!`×2）。语速幽灵腿不可 veto（判据是时间物理不可能，不是无声）。正常语速的 BGM 上套话（如 kaguya60
片头）由既有能量腿负责，精确短语 `ご視聴ありがとうございました` 仍由 profile 1
清理——三者分工互补。

`语言切换幻觉`（2026-08-04 新增，**仅打标，不参与 profile 0 丢弃**）标记 CJK 主导素材里
突然出现的大段 Latin 低置信文本：

```text
run_gate = 全文件字母中 Latin 占比 < 0.3   # 真英文/双语素材整体关闭
segment 命中 = 字母数 >= 8
             and 段内 Latin 字母占比 >= 0.7
             and segment.confidence < 0.6
```

**为什么不丢弃**：405 窗口验证集上命中的确实全是幻觉（4/4、0 误报），但更大范围复核
（170 份产物 + 人工修正字幕对照）发现命中集合里混着两类**不能删**的东西——
(a) 真实英文内容：歌回/英配 PV 素材（H6dTZf9QFTY）里 `Yes, my lord!`、
`Making good, being you, that's alright` 等 15+ 行真实英文歌词/台词（最初依据 LLM 层
保留判定；2026-08-05 经 Qwen 双模型重认抽检 5/5 确认音频确为英文），
低置信只是因为唱歌难识别；(b) **翻译型幻觉**：BV1cqLR6hEp3 224–251s 底下是真实日语台词
（人工字幕：女皇陛下の偉大なる計画…），Whisper 输出了它的英文翻译——删除会连真实对话的
唯一痕迹一起丢掉，正确修复是强制语言重解（未实现，见 wt-refine-handoff P1）。
标签保留给下游（LLM 纠错层、未来的强制语言重解触发器）作观测证据。
confidence 阈值按一遍式解码值标定；teacher-force fallback 路径存在约 −0.14 的系统偏差。

### Profile 3：确定性预合并（**已删除**，2026-07-29）

原 profile 3 调用 `src/premerge.py` 做词形强证据的词中接回。`segment_split` 迁到全局 DP
之后它失去了对象：DP 自己就在决定每一个 ASR 段接缝是否保留，词中切断的碎片在切分阶段
就不再产生。实测 9 clip 测试床（8BV + yui）——同一套 premerge 规则在**原始 ASR 分段**与
**旧逐段 split 输出**上各合并 1 处，在**新全局 split 输出**上合并 **0 处**。模块、
stabilize profile 3、`metadata.premerge`、`premerge_rejoined` /
`premerge_filler_attached` report 字段与相关测试一并删除。

历史结论仍然有效、迁移时请勿重犯：预合并当年必须排在 profile 2 之前，因为词中切断的碎片
天然低置信（`次はキッ|と` 的 `と` conf 0.089、能量为负），先跑 profile 2 会把它当幻觉丢弃、
词永久残缺。现在这条约束由「不产生这种碎片」满足，而不是由「事后修补」满足。

### Profile 0：默认稳定化

依次执行 profile `1 -> 2`，随后删除带 `高度疑似幻觉`、`高度疑似语气填充词` 或
`套话幽灵` 的 segment。只带 `时间漂移` 或 `语言切换幻觉` 的 segment 保留
（后者仅观测，理由见上）。

## Schema 与复用

- aligned 的 schema 与此前未稳定化的 stable schema 相同，包含 `segments` 和原
  `metadata.vad` / `metadata.asr_align`；aligned 侧 split 的产物见
  `docs/segment_split.md`：word 可带 `whisper_segment_start: true`（ASR 原生分段首词），
  段可带 `tags: ["mid_segment_start"]`（起点是 DP 在 ASR 段内部切出的）。
- 稳定化保留未知顶层字段、metadata 和未修改的 segment 字段；不写额外 profile metadata。
- pipeline 只按输出是否存在复用：stable 已存在时不回补 aligned；aligned 已存在且 stable
  缺失时只跑稳定化；显式 `--stage aligned` 必须生成或复用 aligned。
- profile 改变不会自动使现有 stable 失效。要重跑需删除 stable 及全部下游 artifact。

## 验证

```powershell
python -m pytest -q test/test_asr_stabilize.py test/test_pipeline_refactor.py
```
