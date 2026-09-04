# asr-align

`python -m finesub.speech.recognition.cli.align` 使用 `fw-refine` backend（打过补丁的 CTranslate2 一遍式 WT refine）对已有 VAD
interval 做 ASR、词级时间映射和结果清理。**`whisper-timestamped` backend 已于 2026-08-02 移除**，
回溯点见 [`wt-refine-handoff.md`](wt-refine-handoff.md)。
实现位于 `src/finesub/speech/recognition/transcribe.py`，薄 CLI 入口位于
`src/finesub/speech/recognition/cli/align.py`。
识别输出的 overlap clamp、零时长修复和空段过滤位于
`src/finesub/speech/recognition/segments.py`，不属于 profile 驱动的字幕稳定化。
ASR partial 的 identity、schema 和原子读写位于
`src/finesub/speech/recognition/checkpoint.py`。**ASR 固定单 worker**——
单文件分片已于 2026-08-02 移除，见 [`wt-parallelism.md`](wt-parallelism.md)。

## 输入与输出

输入 JSON 至少包含：

```json
{
  "segments": [
    {"start": 1.2, "end": 4.8}
  ]
}
```

`--audio` 必须指向生成这些 interval 的同一时间轴音频。默认输出名为
`<vad-json-stem>-asr.json`。

```powershell
python -m finesub.speech.recognition.cli.align out/input/vad.json \
  --audio out/input/input-vocal.ogg \
  -o out/input/input-asr.json \
  --model large-v3-turbo \
  --language ja
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--audio` | 必填 | 与 VAD JSON 同时间轴的音频 |
| `--model` | `large-v3-turbo` | Whisper 模型。备选与取舍见 [`manual/models.md`](manual/models.md)：`large-v3` 无实测优势（只作对照）、`TransWithAI/whisper-ja-1.5B-ct2` 是未实测的日语微调 |
| `--device` | CUDA 优先 | 无 CUDA 时告警并回退 CPU |
| `--language` | 自动检测 | 显式指定可避免语言误判 |
| `--gap` | `0.3` 秒 | 组尾合成静音时长（inter-interval 静音为自适应，不受此参数控制）；其前保留至多 `0.7` 秒原始 gap 音频 |
| `--block-seconds` | `600` | 流式音频 block；`0` 表示关闭分块 |
| `--pad-seconds` | `10` | block 左右上下文 |

## 当前对齐逻辑

1. 校验并裁剪 VAD interval，按时间排序。
2. 按目标时长和自然间隔动态分组。相邻 interval 之间插入
   `min(original_gap, 0.7)` 秒的**原始 gap 音频**（保留 VAD 截掉的低能量尾音）
   加自适应合成静音 `min(0.1 + 0.2 × original_gap, 0.8)` 秒
   （GAP_SILENCE_BASE_SEC=0.1, GAP_SILENCE_GROWTH=0.2, GAP_SILENCE_MAX_SEC=0.8；
   给解码器统一的断句提示）；组尾再垫
   `min(0.7, 到下一 interval 的间隙)` 秒原始音频 + `--gap` 秒静音（默认 0.3）。
   recall 补录
   批次的组尾额外受**下一个已覆盖 segment span** 约束（补录链常终结在已有
   segment 起点处，此时垫 0 秒，不重复转写已覆盖语音；终结在 interval 边缘时
   同样能拿到至多 0.7 秒真实尾音）。

   **分组时按合成后的音频长度判断，组尾垫料计入在内**
   （`combined_group_audio_seconds()` = `combined_group_duration()` + `group_tail_seconds()`）。
   在此之前只算「语音 + 区间之间的插入」，组尾最多 1.0 秒不计，于是按 30 秒规划出的组
   实际可达 31 秒、溢出编码窗口；2026-08-02 起改为按实际长度规划。11 个真实 clip 上
   **超窗分组 102 → 56**，总组数 388 → 405（+4.4%）。
   与之相对，`combined_group_duration()` 保持「说了多少话」的语义不变——auto language
   的短组启发式用它，垫料不该计入那个判断。

   剩余 56 个超窗分组不是 gap 计算问题：分组器在**找不到足够大的自然间隙时会继续累积**
   （语义边界优先），因此密集语音段会产生任意长的组，最长实测 73.9 秒。是否为了适配编码
   窗口而强制切分，是另一个待决策的取舍。
3. 调用 `fw-refine` backend。常规路径使用一遍式对齐：greedy decoding、单一
   `temperature=0`、不使用 beam search 或 temperature fallback（只有覆盖率救援中的 beam
   重解是例外，见第 6 步）；`refine_sec=1.0` 保持不变。beam 与 temperature fallback 都会
   离开一遍式轨迹，因而改用后端自己的 teacher-force 对齐。
   `--language` 未指定时，合成后时长不超过 10 秒的 group 优先沿用最近 10 个真正经过
   自动检测的输出 ASR segment 中的语言众数；没有历史时仍自动检测，频率并列时取最近值。
   这里的 group 指每一次实际 Whisper 调用，包括正常 group、regroup 后的 subgroup、recall
   group，以及异常后降级的逐 interval/segment ASR。未被最终采用的异常候选不会写入语言历史。
   `fw-refine` 提供修复后的 `detect_disfluencies`，默认关闭。启用后复用 1-pass compact
   attention；首个实词前的空-gap 候选只调整该词起点，不渲染会改变 segment 边界的 `[*]`，从而
   避免下游 overlap clamp 错缩上一段尾词终点；其他候选保持 WT 的 `[*]` 行为。所有候选同时以
   `alignment_events` 透传。`remove_empty_words` 已删除；零时长 chunk 尾保留原词并上报
   `zero_duration_chunk_tail`。完整取舍与信号路线见 `docs/wt-refine-port.md`。
4. 将 Whisper 输出映射回原时间轴：**每个 ASR 段整体保留**，不再在
   VAD interval 边界切开；词坐标在 interval 内 1:1 映射，落在 gap 保留音频内的词同样
   1:1（合成静音区按剩余原始 gap 比例映射）。段挂到词数占多的 interval 供 finalize
   使用。相邻段因尾词能量延长产生的重叠在输出前收回（后段起点优先）；
   收回时被完全甩到新末尾之后的词不留零时长残骸，而是就近并入本段最后一个
   存活词（文本拼接、confidence 取最小值）；整段塌缩时其文本并入后段首词
   作前缀、空段删除。`fw-refine` 的 path/disfluency events 同步映射回原时间轴；全局 DP 重分句后
   每个事件只保留一份，归给覆盖其锚点的输出 segment。当前 FineSub 不消费这些事件。
5. 检测异常重复/超长结果后**直接进入异常 interval 隔离**（无 regroup 重试、
   无整组 beam 重解，两者已于 2026-07 移除，依据见下方「救援策略的取舍」）。
   两类豁免不进隔离：(a) `[*]` disfluency 标记词不参与词级异常判定——它是注意力
   候选跨度不是解码内容，长 `[*]` 会被误判为拉伸词（实测 BV1dwjP6LECU 一例
   5.35s）而其消解本来就归 `word_starts`；(b) **纯套话堆叠早退**
   （`_is_known_phrase_stack_only`）：全部异常都是 collapse stack、堆叠文本只是
   `ご視聴ありがとうございました`（重复或截断片段）、且被标 interval 的剩余词
   自身干净时，跳过救援——堆叠下没有可回收语音（stabilize 短语清理会整体删除），
   重解只是把挤压形态换成拉伸形态、还对健康的剩余词重掷骰子。2026-08-05 放宽为
   允许「真话 + 零宽套话尾」形态（400 窗审计中纯套话异常窗全部如此，
   覆盖率不足时覆盖率救援照常另行触发）。
   隔离过程：定位第一个异常 interval `k`，把它之前的干净 interval 合成**一窗**
   重解（保住上下文），异常 interval **单独**一窗，`k` 之后的 interval 作为
   未消费尾部**交还主循环**——与其后的 interval 一起重新分组，短残段因而并入
   下一个正常尺寸窗口，而不是以最少的上下文单独解码。前窗重解退化时，只有当
   候选切片**整体**不含异常且覆盖达标才沿用切片，否则对前窗递归隔离。
   注意「整体」二字：`_first_abnormal_interval_index` 是逐 interval 判定，而
   `repeating_group_cycle` 需 32 units 才触发，一条横跨两个 interval 的循环
   各占 20 units 时每个 interval 单看都干净、切片整体却正是那条坍缩。
   `long_word_token` 使用独立 word-unit 指标：空格分词语言每个词计 3，CJK 等
   无空格文字每字计 1，纯数字串计 1，混合文字按各部分累加；单个 ASR word
   达到 15 units 时判为异常。该指标不是 Whisper tokenizer token 数，也不是字幕
   `weighted_char_count`。此外会检查整个 group 的局部精确循环；局部循环次数
   不少于 4 且循环跨度不少于 32 units 时判为异常。该规则只触发
   隔离/回退，不直接压缩局部循环文本；旧的单 token 和相同 word run 检测继续保留。
6. **覆盖率救援**（正常 group 与 recall 批次都适用）：greedy 解码可能在 30 秒
   窗口内提前 EOT，整段跳过而所有输出质量指标（`no_speech_prob`、
   `avg_logprob`、异常词检测）全部正常——唯一可靠信号是覆盖率。当输出
   segment 与本批 interval 的重叠时长低于 `0.6 × interval 语音时长 − 2 秒`
   时（不足 ~3.3 秒的小批天然豁免），启动救援阶梯：先整批 `beam_size=5`
   重解（干净且覆盖达标即采纳）；仍不足则回到 greedy，把首个 interval 剥离
   为独立窗口、其余合为后窗重解，后窗覆盖率仍低就继续剥离（收敛于逐
   interval）。救援结果仅在覆盖时长严格增加时替换原结果。
7. 对正常结果未覆盖且累计不少于 5 秒的 complement 做临时 recall ASR；短于
   `0.25` 秒的 complement 碎屑不参与（孤立碎屑单独解码必然幻觉，也不计入
   5 秒阈值）。
8. 使用词尾附近的加权能量将最后一个词最多延长 1 秒，然后清理、排序并输出。
   fallback 清理不再逐 word 压缩，也不再单独合并相同 word run；它在完整 segment
   原始文本上查找连续精确重复，标点和空格同样参与比较。重复次数超过 7 时保留 5 次，
   并将覆盖重复区间的所有 words 合成为一个 word（时间取首尾、confidence 取最小值）。
   清理会反复执行到稳定，因此同一 segment 内多个不同的重复区间也会分别处理。

## 语言票翻转重解（`--lang-redecode`，默认 auto）

2026-08-19 新增（`speech/recognition/lang_redecode.py`）。方案原稿与逐轮实验编年在本地
`docs/archive/speech-quality-plan.md`；**现行行为、取舍与待办以本节为准**。

### 根因：前文污染下的窗内崩塌，不是语言检测漂移

最初的假设是「Whisper 把这段的语言检测错了」。实验推翻了它——同一段音频按生产口径拼窗，
只改 `initial_prompt`：

| 前文 | 语言检测 | 输出 |
| --- | --- | --- |
| 无 | ja p=0.59 | 正确日语，与人工参考相似度 0.862 |
| 前一窗输出（含「ご視聴ありがとうございました」套话） | ja p=0.59 | 正确日语，0.812 |
| **纯英文** `"I'm not going to die." ×2` | **ja p=0.59** | **`"I'm not going to die."`** |

**语言检测一直是 ja，输出却崩成英文。** 三条直接含义：`lang='en'` 是伴随现象不是原因；
崩塌会经前文自我传播（现场表现为连续 9 段同一句）；该处语言判定本就边缘（p=0.59 近乎
翻硬币），窗的 interval 组成稍变就会翻到 en。因此**重解必须清前文**，且采纳判据不能建立在
「重解后语言变了」上。

首版仅以 group 语言票翻转作为可观测代理，因此只覆盖该崩塌家族的一个子集；
「语言票仍正确、文本却崩成外语」不在本开关的承诺范围内。

### 流程

修复放在解码主循环内、该 group 的解码+覆盖率救援之后、recall complement 之前：

1. **触发**：仅 auto 语言 run。group 折叠后的语言票与近期 10-group 众数不符
   （众数为 None 时不触发——真值守卫是判据的一部分）。
2. **证据**：Qwen referee 按该 group 的 **VAD interval 跨度**逐一重认（幻觉段自己的
   几何两个方向都不可信，见下「取舍」）。
3. **重解**：以众数语言强制重跑 `align_group`（新调用天然清前文），交还的 interval
   在重解侧循环消费完；救援阶梯照常工作。阶梯负责把新结果做到最好，判据负责决定要不要它。
4. **采纳**：硬门 = 证据非空**且重解产物非空**（投票支路不看重解产物，无此门会把
   空重解当替换采纳——那是删除，越出「重解+条件替换」的边界）；然后
   `sim(证据, 新) > sim(证据, 旧) + margin`（字符级 `SequenceMatcher.ratio`，输入先过
   `normalized_compact`）**或**逐 interval 时长加权语言投票 > 2/3 且 == 众数。
   任何失败方向都保留原结果。
5. **账本**：采纳时回滚该 group 已入 `auto_language_history` 的幻觉票、补一张众数票
   （与正常路径「一 group 一票」同口径，裁剪走共用的 `_trim_language_history`）；
   拒绝时原票不动。

开关进 `checkpoint.build_key()`（两种设定的 partial 不得互相 resume）。事件与计数写入
aligned metadata `asr_align.lang_redecode`。默认 `auto`（依赖可用即运行）；`on` 要求依赖必须
可用，`off` 完全跳过该路径、产物与改动前逐字相同。

替换后段的 `lang` 会从幻觉语言变成众数语言。**LLM 层今天不读这个字段**——全仓 `"lang"` 的读取
都在 speech 层（`segmentation.py` 的继承投票与字段透传、`transcribe.py` 的语言历史），所以
「让纠错层知道这一段的源语言」是新建契约而不是同步契约，不在第一版里。

### referee 设备（四个问题，不是一个）

`referee_device` 依次问四件事。它们曾经是同一个问题——「ASR 在不在 CUDA 上」——而 2026-09-02
ASR 阶段改由 CTranslate2 决定自己的设备之后，那个合并不再成立
（[stage-device-plan.md](plans/stage-device-plan.md)）：

| # | 问题 | 谁回答 | 答否时 |
| --- | --- | --- | --- |
| 1 | **意图**：用户显式 `--device cpu` 了吗 | `requested_device` | cpu。「把卡让出去」不是「只让 Whisper 让」 |
| 2 | **策略**：档位允许用 GPU 吗 | `ResourceProfile.gpu` | cpu |
| 3 | **能力**：**裁判自己的后端**能用这张卡吗 | `device.cuda_usable()`（torch） | cpu |
| 4 | **余量**：pool 真的在卡上吗？占多少 | 下面这张表 | cpu |

第 3 问是新的，也是必须的：裁判是 transformers 模型，而 ASR 的设备现在由 CT2 决定——
**CT2 能在一张卡上解码，不代表 torch 有它的 kernel**。跟着 CT2 上卡会把 torch 模型塞到
一张它跑不动的卡上。

第 4 问也不再恒真：**ASR 落 CPU 时 pool 根本不在卡上**，于是整档预算都归裁判
（`referee_vram_budget(beside_pool=False)`）。只有 pool 真的常驻时，下面的减法才登场。

判定要在解码循环内做，Whisper pool 此时**必须留在显存里**，所以第 4 问的依据是「ASR 之外
的余量够不够放下 referee」（Qwen 0.6B bf16 峰值约 1.5 GiB）：

| ASR 模型 | 实测常驻（B=1） | `entry`（3 GiB） | `standard`（6.5） | `high`（10） |
| --- | --- | --- | --- | --- |
| `large-v3-turbo`（生产默认） | 2.07 GiB | cpu | cuda 共驻 | cuda 共驻 |
| `large-v3` | 3.89 GiB | cpu | cuda 共驻 | cuda 共驻 |
| `TransWithAI/whisper-ja-1.5B-ct2` | 3.82 GiB | cpu | cuda 共驻 | cuda 共驻 |
| 其它 / 未知 | — | cpu | cpu | cpu |

余量 = **档位要求的空闲显存** − 常驻，够 `QWEN_REFEREE_GIB`（2.5）就共驻。上表只列了
三个 GPU 档；`standard_large_vram` 与 `high` 同为 10 GiB，落点相同，档位全表见
[gpu-profiles.md](gpu-profiles.md)。**未知模型一律 CPU**：CT2 的分配
torch 的 CUDA 计数器看不见，猜低会在池被租用时 OOM。

**2026-09-02 复测（RTX 5070 Ti，`tools/bench/probe_whisper_resident.py`，整卡口径、装载 +
一次 30 s 真解码）**：turbo 复现为 2.08 GiB（记录 2.07，方法在新机上仍成立），而
`large-v3` 量到 **3.89 GiB**——原来表里的 **6.15 是 [wt-refine-port.md](wt-refine-port.md)
档位表里 B=8 那一行的进程显存**，被当成 B=1 抄了进来；那张表自己的 B=1 基线写的是
4.34 GB（= 4.04 GiB），与本次实测一致。turbo 没抄错，因为下一段恰好特意区分过它的
B=1 与 B=8，而 `large-v3` 没人做这个区分。日语微调与 `large-v3` 逐位同级
（装载后两者都是 3.69 GiB），符合「同架构」的预期。

改数之后 `standard` 档上两个大模型从 CPU 翻成共驻，余量 2.61 / 2.68 GiB —— 比
`QWEN_REFEREE_GIB` 只多约 0.1。**这个薄余量是知情选择**（owner，2026-09-02）：裁判 eager
路径实测峰值 2.3 GiB（[bench-baselines.md](bench-baselines.md) 22.3）放得下，而更吃显存的
编译路径由 `COMPILE_MIN_VRAM_GIB`（3.5）单独挡在外面——`referee_vram_budget` 在这里算出
2.6，够不到那个门槛。要再往这个预算里加东西之前，先重量一次常驻。turbo 的 2.07 GiB 低于
[wt-refine-port.md](wt-refine-port.md) 档位表里 B=8 的 2.92 GB——生产路径没有 multi-audio
batch 的调用者，实跑就是 B=1，两个数用途不同（那张表答「能承受多大 batch」，这里要「现在实际
占了多少」）。

CPU 上用 **float32**：bf16/fp16 省约 1.4 GiB 但慢 2.2–2.5 倍，输出逐字相同——是速度问题不是
正确性问题。

**不做 swap**（把 Whisper 换出显存、Qwen 换入、判完换回，往返 1.25–1.32s）：真混合多语素材上
触发频繁，每次都要付；且快速 swap 要求两个模型同时常驻主机内存，否则退回磁盘加载
（Whisper 2.5s、Qwen 首次约 6s）。CPU referee 把这条整个绕开。

inline referee 与尾部 `--qwen-verify` 的 referee 在**设备相同时复用同一个实例**；设备不同
（4GB 档 inline 在 CPU、Whisper 释放后尾部可用 CUDA）则关掉重建。两者是两个独立的取证入口，
inline 不改尾部校验的嫌疑面与跨度口径（见 [vad-asr.md](vad-asr.md)）。

### 取舍与已否决的做法

**为什么 inline 而不是事后修**：① 事后修整个 run 里 `auto_language_history` 一直是脏的，会
影响后续短 group 的语言复用（`_language_for_group` 对 <10s 的 group 查历史）；② 漂移窗若同时
触发异常判定，事后方案面对的是一条已经「用错误语言做过隔离」的时间轴；③ `unconsumed` 是顺序
状态，事后重建 interval 集合复现不出 inline 会得到的分组边界。代价是改动落在高危区，但形态上
是顺着既有的「异常判据 → 救援路由」加一条同形状的，不是塞进去一个异物。

**为什么取音频用 VAD interval、不用 segment 跨度**：幻觉段的几何两个方向都能错——9 个候选
对其覆盖的 VAD 并集，起点差 −0.06~+1.54s、终点差 −2.78~+1.99s；有一段段尾超出任何 VAD 语音
1.99s，另一段只有 0.05s 宽。实例：某段跨度 236.96–238.32、所在 VAD interval 是 236.80–239.74，
按段跨度读证据只到「財政はパン。」，按 interval 读才拿到完整的「財政はパンタローネに一人」。

**为什么整窗重解、不切分**：整窗与人工参考相似度 0.862；在最大间隔（4.75s）处切开取右半能到
0.941，但切点选择需要一个要标定的间隔阈值，而切在次大间隔（1.33s）处反而降到 0.673/0.452
——那 0.08 不值得。整窗也比逐块更完整（逐块漏掉的「バカだれ」整窗一次拿到）；按生产口径塞
0.3s 静音拼接（packed）优于连续音频（0.862 vs 0.794）。

**为什么不用「重解后语言变了」**：见上，语言可能一直没变。

**为什么不用 Qwen 证据的字符集分类**：中日共用汉字，实测有一段证据「我看到。」按字符集会被判成
中文而真值是日语。相似度不碰这个问题；而且相似度这一支顺带防住「重解本身也幻觉」（音译垃圾
同样不像证据），字符集判据做不到——垃圾片假名照样是「日语字符集」。

**两支判据是 OR，是有意的保守取向**：相似度在证据充分时判别力极强（+0.732 对 0.000），但证据
稀疏时会失灵（某素材只有 31% 的 run 时长有证据）；时长加权语言投票在那一例给出干脆的 86%。
硬门独立于两支之外——纯笑声/叹息素材上证据全空，判据自动归零，不依赖任何阈值。
**不设投票覆盖率下限**：31% 覆盖那一例本来就判对了，加下限只是凭空多一个要标定的数。

### 实测速览

2026-08-19，`out/reference/` 的 8 份生产产物 + 本机 GPU（large-v3-turbo，greedy，
Qwen3-ASR-0.6B）。**样本量都很小。**

**触发面**：`lang ≠ 主语言` 命中 25 段，拉丁串判据（letters≥8 且 latin≥0.7）命中 12 段且全部
与前者重合；多出来的 13 段主要是笑声与感叹（`'Ahahaha!'`、`'Eeeh!'`、`'Oh?'`），由采纳判据
挡掉。宽是可接受的——误触发的代价只是白判一次。（这 25 段用**整 run 静态主语言**作代理量得，
生产判据是滚动 10-group 众数，混语素材上两者会有出入；该数字只用于论证「宽而无害」。）

**采纳判据**（n=3 run，3 素材）：

| 素材 | sim(证据,新) | sim(证据,旧) | 差 | 语言投票 | 采纳 |
| --- | --- | --- | --- | --- | --- |
| BV1cqLR6hEp3 | 0.732 | 0.000 | **+0.732** | ja 93%（覆盖 80%） | 是 |
| BV1ojjc6MEAs | 0.400 | 0.000 | **+0.400** | ja 86%（覆盖 31%） | 是 |
| BV1ySjz6FEzD | 0.000 | 0.000 | +0.000 | 无票（证据全空） | **否**（硬门） |

第二例拿回了真实台词（「ちゃんと凝ってんなクッションがあってここに一回取りに行って投げて」），
老结果全是英文感叹词；第三例是纯笑声，两个结果其实是同一内容的不同书写，不该动。
上表绝对值由未存档的实验脚本以同类字符级度量测得，无法确认与实现内置的
`SequenceMatcher.ratio` 逐位同源：**标定时一律用实现内置的度量重测**，不直接沿用这些数
（三例的采纳/拒绝方向在任何合理字符级度量下都不变）。

**开销**：

| 项 | 实测 |
| --- | --- |
| 单窗强制重解（22.6s 音频） | 0.43–0.66s |
| Qwen CUDA 推理 / 显存 | ~2.0s per clip / ~1.5 GiB（2026-09-01 起成批解码：每步 60 ms 与 batch 无关，8 条 6 s clip 一批 ≈1.5 s；显存见 `vad-asr.md`） |
| Qwen CPU float32 推理 / 主机 RSS | **3.05s per clip / +3.26 GiB** |
| Qwen CPU bf16 / fp16 推理 | 7.69s / 6.78s per clip（输出与 fp32 逐字相同） |
| Whisper CT2 加载 / 卸载到 CPU / 载回 | 2.50–2.85s / 0.77–0.84s / 0.47–0.48s |

**只有单 clip 口径，没有端到端增量。** 4GB 档（默认档）走 CPU referee，每次触发要对该 group
的每个 VAD interval 各跑一次，且模型一旦加载就常驻到 ASR 结束——资源含义记在
[vad-asr.md](vad-asr.md)「资源与失败行为」。

**后处理链幂等**（n=1 素材、158 段）：`word_starts.apply_disfluency_rules` →
`clamp_word_starts` → 四个 `segment_ops` → `segment_split.split_segments` 在自己的输出上再跑
一遍逐字相同（clamp 第一遍 5 次、第二遍 0 次，收敛），所以被替换的段照常流过下游。

### 待标定（P2，未做）

`margin=0.15` 与投票门槛 `2/3` 两个数**等的是同一批素材**——`out/reference/` 的 8 份产物里
没有任何真外语段落，**一个负例都没有**。

| 待定 | 现值 | 需要什么 |
| --- | --- | --- |
| `margin` | 0.15 | 真外语窗口的 `sim(证据,新) − sim(证据,旧)` 分布。已知真阳性侧最小 +0.400，所以 margin 不应超过 ~0.2 |
| 时长加权占比门槛 | 2/3 | 同上 |
| 语言检测置信度预筛（未实施，记一笔） | — | 幻觉窗重解时 p=0.59，真日语块 p=0.72–0.98。若「真语言切换的检测置信度高、幻觉性漂移低」成立，可在问 Qwen 之前用概率筛掉大部分真切换，混合多语素材上的第二模型调用会降到接近零。inline 时该概率白拿。数据太少，现在不做 |

素材要求很具体：**主语言之外的真实外语段落**——英配 PV、外语歌、外语嘉宾。已知候选是
`H6dTZf9QFTY`（[wt-refine-validation.md](wt-refine-validation.md) 记的歌回/英配 PV，15+ 行真
英文命中拉丁串判据），但它不在 `out/reference/` 里，需要重跑一次 vad-asr，或另找两三个双语
素材。这条直接对着 wt-refine-validation.md 的方法论教训：**405 窗口的 0 FP 是类型盲区给出的
假保证**——语料全是谈话向直播，歌回/英配一个都没有。

`margin` 的定位是**死区而不是分类边界**，存疑即保守。硬门与两支判据的 OR 结构都不依赖它，
所以即使定得不准，失败方向也是「不修」而不是「改坏」。

## 全局语言审计（`lang_audit.py`，随 `--lang-redecode on`）

2026-08-30 新增。上面那个触发判据有一处**结构性**缺陷，不是阈值问题：

> 它比的是「本 group 的语言票」与「最近 10 个 group 的滚动众数」。
> 判据是**相对的**——**众数整体错时，矛盾永远不会出现**。

实测：合成 ja+en 快速交替那一跑，**89.7% 的日语被判成 en，而它一次都没触发**
（`bench-baselines.md` 15.3）。一个定义为「与多数不一致」的探测器，
对「多数整体错了」这件事是盲的。

缺的是一个**在被审计量之外**的锚。本模块提供一个：把本次 run 自己的音频
**确定式均匀抽样**若干段，交给 Qwen referee（它看不到 Whisper 的判定）重认，
两边的**时长加权众数**对比。

**只报警，不动手。** 动手意味着对整条 run 强制语言，
而这件事做错的代价是全损（英文按日语解码，与真值相似度 0.000，`bench-baselines.md` 15.5）。
授权它需要在**真**语码转换素材上量到误报率，我们没有——合成拼接能证明缺陷存在，
证不出发生率（15.6）。

### 判据是无阈值的

只有一条：**两个模型对「这条 run 主要是什么语言」给出不同答案**。没有要标定的数。
其余常量都是成本上限或「有没有样本」的下限，源码里逐条注明。

`resolve_mode()` 是这四格的唯一真相（`--lang-redecode` × 有没有 `--language`）：

| | auto 语言 | `--language X` |
| --- | --- | --- |
| `auto`（默认） | `redecode`——与改动前逐字相同 | **不建 referee**，什么都不买 |
| `on` | `redecode+audit` | **`audit-only`**——触发器天然失效，但**用户强制错了语言**是真实可达的失效 |

`audit-only` **不进 checkpoint 复用键**：它只读解码、从不改解码，因此不可能让 partial 失效。

### 两条 resume 契约（2026-08-31 补，两条都由测试固化）

- **抽样账本随 partial 走。** 审计抽的是「被告知过的 group」，所以那份账本是 run 状态，
  和语言历史一样要进 checkpoint（`lang_observations`）。少了它，续跑只会审计后半段——
  换一个抽样、换一个结论，而且剩余不足 `MIN_ANSWERED` 时**干脆不审计**：
  中断落在哪里就决定了这一跑报什么。
- **账本记的是「发出去的那个语言」。** `observe()` 必须在决策**之前**跑（`maybe_redecode`
  的多数提前返回正是审计要的 group），但**重解一旦被采纳**，账本会投强制语言、正文也
  是那个语言，所以这条记录要被 `amend_last_observation()` 改写。否则审计会拿一个
  「已经不在成品里」的语言去报警。

### 为什么默认不开

一次 referee 加载 + 8 个 clip，**实测 16.9 s**（见下；2026-09-01 起 8 个 clip 成批解码约
2 s、加载可在 standard 档下与 Whisper 重叠——账已经变了，但默认仍等一次重测再翻）——而它要抓的
那类失效，发生率恰恰就是「结构性失明」让我们量不到的那个数。与 P1 推迟自动回退同形：
先把检查交付，默认留到有数字再说。两条把它变成可默认开的候选路线：
① 更便宜的取样；② 直接复用 `--qwen-verify` 已经拿到的 referee 语言字段（零额外推理，
但样本偏向「看起来可疑」的段）。

### 实测（2026-08-30，`bench-baselines.md` 15.7）

| run | 素材 | Whisper | referee | 一致率 | 报警 |
| --- | --- | --- | --- | --- | --- |
| `out/wata1` | **真实生产**，日语 | ja | ja | 1.00 | 否 ✓ |
| `out/yingtao` | **真实生产**，日语 | ja | ja | 1.00 | 否 ✓ |
| ja-only | 合成 ja+ja | ja | ja | 0.85 | 否 ✓ |
| **ja-only 强制 `--language en`** | 均匀误标 | en | **ja** | 0.13 | **是 ✓** 端到端 |
| ja-de | 合成混语 | de | de | 0.78 | 否 |
| bi-fast | 合成混语 | en | en | 0.89 | 否 |
| **bi-slow** | 合成混语（约 50/50） | en | **ja** | 0.57 | **是 ⚠** |

两个方向都在**真实音频 + 真实 referee** 上验过。⚠ 两条要一起读的限制：

1. **它抓不到 15.3 那一跑。** bi-fast 里日语确实被大量误标，但那条 run 里**一半音频
   本来就是英语**，两个模型对「run 的主语言」并无分歧。**run 级众数判据只能抓「整条错」，
   抓不到「一半错」**——后者是语码转换问题，需要逐段判据（P16 已评估：候选池 0.21%，先不做）。
2. **真双语素材上它会响**（bi-slow）。50/50 的文件没有「主语言」可言，两个模型各挑一半，
   谁都不算错。所以 warning 的 action 里明说了这一条。

**开销**：clip 长度是成本主因，不是 token 预算。20 s clip 时整轮 **52.3 s**；
把 `max_new_tokens` 从 256 降到 48 →**54.9 s（没有变化）**；把 clip 上限降到 6 s
→ **16.9 s**，判定不变。测时 GPU 有约 81% 的外部占用，绝对值偏高，
比值（52.3→16.9）比绝对值可信。

## 救援策略的取舍

2026-07 在一条 2h12m 日语直播素材上实测过救援阶梯的各条路径（同一批 VAD
interval，指标为硬件无关的 transcribe 调用数与喂入 whisper 的音频秒数）。结论
与依据如下，改动这块前先读：

**为什么删掉 regroup 重试**：26 个 rescue 样本上解决率仅 23%，而 15 个 group
中有 7 个（47%）输出与失败的 greedy 逐字节相同——纯空操作；且它产出的复读行
占比 42%，比它要救的 greedy（38%）还高，是高置信长复读的主要来源。

**为什么删掉无条件整组 beam 重解**：beam 在 greedy 已经正常的 clean 桶上有 10% 单向
坍缩率（greedy 干净而 beam 崩，0 例反向）；在 hard 桶上一次救回率仅 20%，低于
它原先在阶梯末尾的 25.5%；且既坍缩又丢内容（总输出 139 行 vs isolation 164）。
注意 beam 的置信度看似更低是两遍对齐路径的**系统性偏差**（clean 桶实测
-0.141），不是质量差 —— 不要据此比较两条路径。这一结论只反对“检测异常就无条件整组 beam”，
不否认 beam search 可能小幅改善语义文本；coverage rescue 仍保留 beam=5，后续也可在有文本不确定
证据时把它作为第二质量候选。

**为什么异常 interval 单独隔离、不并入邻窗**：25 个隔离点上比较四种窗口
（成功 = 解码干净 **且** 坏 interval 覆盖 ≥50%）：

| 窗口 | 成功率 | 音频秒 |
| --- | --- | --- |
| 单独隔离（现状） | **68%** | **5.7** |
| 并入前窗 | 40% | 15.9 |
| 并入后窗 | 56% | 17.0 |
| 整窗重解 | 0% | 27.2 |

坏音频会污染同窗邻居：单独隔离时解码干净率 88%，并入前窗即跌至 44%。单独隔离
失败的 8 例中邻居窗口仅能救回 1 例，其中 5 例是「干净但空」（该段本无可识别
语音），非策略可解。

**全片效果**：调用 992 → 795（-19.9%），转录音频 5.27h → 3.94h（-25.2%），
音频重转倍率 2.39x → 1.79x。输出侧 763 个 10 秒窗口中 63% 完全相同、平均文本
相似度 0.872，定式幻觉 101 → 96 次；但高置信复读（c>0.8）17 → 20，删掉 regroup
并未消灭高置信复读、只是换了位置。对最低相似度的 10 个窗口做语义审计为
新版更好 4 / 更差 3 / 平手 3，且败绩集中在 BGM/非人声段（根因是缺音乐门控，
非阶梯策略）。

## 输出字段语义

每个输出 segment 包含 `start`、`end`、`text`、`lang` 和 `words[]`。存在上游数据时还会包含：

> **`confidence` 不是质量指标。** 下游基本不采信它，唯一用途是辅助判断
> 有限的几类经验确定性幻觉、决定是否丢弃。它尤其**不能**用来比较两次解码的
> 好坏：复读坍缩时模型往往极其自信（实测有 conf 0.9+ 的 60 行复读），
> 置信度反而被拉高；一遍式与 teacher-force 两条对齐路径之间还有约 0.14 的
> 系统性偏差。

- segment `confidence`：来自对应 Whisper 来源 segment。
- segment `no_speech_prob`：来自对应 Whisper 来源 segment。
- word `confidence`：来自一遍式解码轨迹的 chosen-token logprob；合成/合并词取来源最小值。
- segment `alignment_events[]`：仅在有命中时出现；为原时间轴上的 refine 观测，当前不改变
  controller/FineSub 路由。

输出 segment 与 Whisper 来源段一一对应，两个 segment 指标即来源段指标。注意它们是
Whisper 在合批拼接音频（interval + 保留 gap 音频 + 0.3 秒合成静音）上算出的：
`no_speech_prob` 对应的 30 秒窗口是拼接产物而非原始音频，其分布与常规整轨 Whisper
用法系统性不同，按常规语义设阈值过滤会失准。

独立 `python -m finesub.speech.recognition.cli.align` 不持有 VAD 的逐帧能量轨，因此不会新增
`vad_weighted_energy_db`；该字段由组合工具 `python -m finesub.speech.recognition.cli.vad_asr` 在最终边界上计算。

## 词首修正（`word_starts.py`，2026-08-05）

fw-refine 的 `detect_disfluencies` 默认开启（实测解码零成本）：attention 在词首前
探测到的迟疑/语气块以 `[*]` 词（span = 原起点→收紧起点）进入解码结果，段首块则以
`is_leading_word` 的 `disfluency_candidate` 事件保留证据。`word_starts.py` 在对齐后、
幽灵清理前把它们全部消解，**任何出口的产物都不含 `[*]`**，且只动时间戳、不动文本：

1. 短于 `0.12s` 的块直接融合回后词（能量帧数不足以判定）；
2. 每个块的 span 与所采动作以词级字段 `disfluency_span` / `disfluency_action`
   记在后词上，可穿过 DP 重分句；
3. 通过能量门控（块内低于「块±2s 能量中位 −12dB」的帧占比 ≥0.4）的块执行删除：
   后词起点取块内最后一段安静帧的结束点（统一处理 onset 在块中间的 partial 块）。
   **常规后移不设位置门**（2026-08-05 与用户确认放宽）：位置限制在 gold 上从未提供
   词头误删保护——能量门独测所有位置 0/25——只损失召回（BV1cq 7 个高 quiet_frac
   mid-phrase 块）。放宽实测：BV1cq +6 删（5 处 gold 人工确认可删、0 词头误删）、
   kaguya60 +26 删（抽查全为谈话段词前长停顿；两例 >1s 长移经实听确认均为前词残余，
   删除正确——曾短暂要求长移提供位置证据，实听后取消）。唯一兜底是**后移上限 3s**
   （`DELETE_MOVE_CAP_SEC`，实测块长最大 2.4s，预期永不触发，in-case 保险）；
4. 其余块融合回后词（即 plain 解码起点）。段首候选同样过这套门控——后端的无条件
   收紧被改为门控采纳，门控不过则回退原起点（回退下界为上一段 end，避免重叠钳位
   吃掉上一段尾词）。

随后两级**锚点 clamp**（同一套守卫，只后移不前移，与四规则天然按 max 组合）：

- **VAD interval 起点**：interval 首词 `start < S+0.1` 且 `start ≥ S−0.5`、
  `end > S+0.15`、前词 end 距 S ≥0.3s 时，钳到 `min(S+0.1, end−0.05)`。
  gold 实测真实 onset 相对 interval 起点中位 +0.10s（VAD 迟到仅 1/26）；
- **pause_hint**（VAD 打分器里死掉的停顿候选，位置≈下一语音起点前 40ms）：
  段内非 interval 首词同守卫钳到 hint（无额外 lead）。hint 对 gold 重偏早家族
  覆盖 7/27——它是子集信号，主修复力量是上面的能量门控删除。

标定与误伤审计（gold n=61：quiet_frac 分离 filled 0.70 / 词头 0.00、删除 0/25 词头
误删、调整后误差中位 ~10ms vs 融合基线 223ms）见 docs/wt-refine-validation.md。
独立 `python -m finesub.speech.recognition.cli.align` 无能量轨：门控必不通过，全部块融合回退（=plain 行为 + span 标注）。
`detect_disfluencies` 进 checkpoint key，翻转开关不会复用旧 partial。

## 验证

```powershell
python -m pytest -q test/test_asr_and_text_utils.py test/test_intervals.py \
  test/test_lang_redecode.py
```

## 组批预取（`DecodePrefetch`，2026-09-02，默认关）

`align_segments(decode_batch=B)`（管线侧 `--asr-decode-batch`，档位表默认 1）在循环前按顺序取
接下来 B 个 group、用与顺序路径同一个 `build_combined_audio` 构出各自的 combined 音频，交
`fw_refine_backend.transcribe_batch` 一次批解码，结果按「音频字节 + 解码参数」缓存；
`_transcribe_group_candidate` 先查缓存，命中跳过顺序解码，未命中原样走。循环的状态机
（动态分组、语言历史、救援阶梯、recall、checkpoint）一个字不动——预取层只回答循环本来就要问的
那次解码。三条边界：只批**单窗**（combined ≤30 s）的 group，长组批第一窗会让 seek 级联改写整组；
救援重解、强制语言重解、语言历史变动后的短组都是未命中，顺序解决；重规划时保留未变的 group，
掉出计划的计入 `prefetch_wasted`。计数：`prefetch_hits/misses/decoded/wasted/too_long`，
phase `asr.prefetch`（含编码与回放）与 `asr.decode_batch`（只含批 generate）。

12 份产物实测（`bench-baselines.md` 二十二）：B=4/8 文本与词时间在第十节门槛内，端到端 **1.06×**
不到 1.15× 的下限，B=16 文本一致率不过——所以默认 1，opt-in 时取 4。
