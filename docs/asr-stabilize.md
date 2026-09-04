# ASR 稳定化

`python -m finesub.speech.postprocessing.stabilization` 是 VAD-ASR 对齐之后、raw SRT 和 LLM 消费之前的独立 stage：读取
`*-aligned.json`，按 profile 清理或标记 ASR segment，输出 `*-stable.json`。源码和 CLI
实现位于 `src/finesub/speech/postprocessing/stabilization.py`，安装后入口为
`python -m finesub.speech.postprocessing.stabilization`。

```text
*-aligned.json -> ASR stabilization -> *-stable.json
```

## CLI 与 pipeline

```powershell
python -m finesub.speech.postprocessing.stabilization out/input/input-aligned.json `
  -o out/input/input-stable.json --profile 0

python -m finesub.pipeline data/input.wav --stage stable --asr-stabilize-profile 0
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
4. `第二模型否决`
5. `语言切换幻觉`
6. `时间漂移`

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

**英文套话（2026-08-30 补）**：`CLOSING_GHOST_PHRASES` 此前**只有那三条日文**，
所以英文侧「自信的幻觉由短语清理兜住」这句注释是**空头支票**——P1 环境音语料的定长
回退跑里，壁炉那份 106 段幻觉有 61 段活到最后，其中 15 段 `Thank you.` 靠
`VERY_LOW_ENERGY_DROP_WORD_CONFIDENCE_EXEMPT` 豁免逃掉（词置信 0.952 vs 送检那批的
0.632，能量几乎一样：−63.0 vs −63.7 dB），而它们该撞的短语表根本不存在。现在拆成
两族：

| | 日文族 `CLOSING_GHOST_PHRASES_JA` | 英文族 `CLOSING_GHOST_PHRASES_EN` |
| --- | --- | --- |
| 离线语速腿 | **参与**（>20 字/秒） | **不参与** |
| 第二模型证据腿 | 参与 | 参与（**唯一入口**） |

⚠ 语速腿刻意不覆盖英文，两条理由都是实测的：**(a) 英文的失效形态是「拉长」不是
「压缩」**——漏网的 `Thank you.` 是 10 个字符摊 11.6 秒（0.86 字/秒），语速腿永远不会
响；**(b) 20 字/秒对拉丁文不是物理不可能，而是正常快语速**（≈200 wpm），把它套上去
只有单向伤害（无证据删掉真实的快嘴 "Thank you."）。所以英文族一律证据门控，
`CLOSING_GHOST_RATE_GATED_PHRASES` 就是这个边界，红验证钉着。

短语表来自那份语料的实际产出（369 行、8 种文本）而非民间清单：`Thank you.` 247、
被重分句切开的 `Thank` / `you.` 各 56、`I'm sorry.` 6。`thank` 单列是因为切分点
25/25 完全相接；**`you.` 故意不列**——三个字符会命中真实代词。它是已知缺口，
补它需要一条「相接半句合并后再判」的规则，不是加个字符串。

`第二模型否决`（2026-09-03 新增，**仅打标，不参与 profile 0 丢弃**）标记「噪声腿本来要丢它，
但第二模型在这一段听到了东西，所以撤回了丢弃」。**它不改变任何取舍**，记的是**为什么留下**。

加它是因为那个否决是一根很大的无声杠杆，而且有实测错误率：存档里它救回 49 段，**41 段救对、
8 段救错**——那 8 段的证据根本不是那一段音频的（2026-09-03 定案，见
`plans/crispasr-followups.md`「第二模型否决只问『证据非空』」）。⚠ **两个分母答的是两个问题，别混**：
**8/49 = 16.3%** 是「这根杠杆一旦拉动、有多大比例拉错」；**8/129 = 6.2%** 是「全部裁判判例里
有多大比例因此被错误保留」——前者衡量判据本身，后者衡量它对成品的代价。⚠ **在打标之前这 6.2% 在产物里
完全不可见**：`_profile_2_tags` 清掉两个标志之后，被救回的段和从未被怀疑的段长得一模一样，
所以那个错误率是靠反事实重放（把证据置空重跑）才反推出来的。现在它自己写在产物里。

判据是**「否决改变了结局」而不是「有证据」**：绝大多数带 `qwen_verify` 的段本来就不在危险中，
按后者打标会给几千段健康 segment 盖章、标记随之无用。存档实测这条判据在 **8752 段里命中 49 段**，
与反事实重放认定的集合**逐段一致（零漏零多）**。

⚠ **不要顺手把它变成丢弃依据**：没有任何本地信号能把那 41 和 8 分开——加权能量、`confidence`、
`no_speech_prob`、证据与原文的音/形对应四个都试过并否掉了（数字在 `plans/crispasr-followups.md`）。
唯一还活着的候选是**绝对 `frame_dbfs`**（注意与加权能量是两个量），但它的工作点是看完标注
才挑的、n=8，按 `README_DEV.md` 的开发原则必须先预注册再到新素材上验。

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
`套话幽灵` 的 segment。只带 `时间漂移`、`语言切换幻觉` 或 `第二模型否决` 的 segment 保留
（后两个仅观测，理由见上）。

## Schema 与复用

- aligned 的 schema 与此前未稳定化的 stable schema 相同，包含 `segments` 和原
  `metadata.vad` / `metadata.asr_align`；aligned 侧 split 的产物见
  `docs/segmentation-split.md`：word 可带 `whisper_segment_start: true`（ASR 原生分段首词），
  段可带 `tags: ["mid_segment_start"]`（起点是 DP 在 ASR 段内部切出的）。
- 稳定化保留未知顶层字段、metadata 和未修改的 segment 字段；不写额外 profile metadata。
- pipeline 只按输出是否存在复用：stable 已存在时不回补 aligned；aligned 已存在且 stable
  缺失时只跑稳定化；显式 `--stage aligned` 必须生成或复用 aligned。
- profile 改变不会自动使现有 stable 失效。要重跑需删除 stable 及全部下游 artifact。

## 验证

```powershell
python -m pytest -q test/test_asr_stabilize.py test/test_pipeline_refactor.py
```
