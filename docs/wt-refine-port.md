# WT refine → faster-whisper / CTranslate2 移植

本文是完整移植 `whisper-timestamped==1.15.9`（WT）refine 行为的实现契约。目标不是
继续改进段首启发式，而是让 faster-whisper（FW）在相同 decoded tokens 上复现 WT 的
word alignment、边界修复与 confidence 语义，并保留 CT2 的单位计算效率。

从零接手本研究时先读 `docs/wt-refine-handoff.md`；本文继续作为算法与 checkpoint 的详细契约。

## 2026-08 checkpoint

ASR 侧已合入 `dev`（2026-08-02）；patched CT2 在独立仓库的 `codex/wt-refine-ct2-research`
（tip `dcc02ac`，基线 `v4.8.1`），交付形态是
[`ct2-patches/`](../tools/wt_refine_port/ct2-patches/) 的 11 个补丁。`fw-refine` 仍是显式
**唯一 ASR backend**（`whisper-timestamped` 已于 2026-08-02 移除），默认启用以下已验证行为：

- greedy 与 beam=5 都使用 winner lineage 的 1-pass compact refine；
- early-EOT、unfinished span、confidence、WT DTW/分词/segment 边界行为在同次 decode 闭合；
- 收集 `alignment_stack`、`long_token_span`、`decoder_repetition`、`unfinished` 和
  `zero_duration_chunk_tail`；
- 事件以 segment `alignment_events[]` 映射回原时间轴，经过全局 DP 分句时只归属一个输出段，
  `asr-stabilize` 原样保留。FineSub 当前不解析、也不据此触发重解。

修复后的 `detect_disfluencies` 保持显式开关、默认关闭：启用时首个实词前不物化边界扰动 `[*]`，
所有实词 end 不变，并额外透传 `disfluency_candidate`。在句首候选完成声学门控前不纳入默认行为。

`boundary_uncertainty` 的 entropy/次峰比尚无人工边界 gold，默认不收集，避免每个 span 固定增加两条
未标定事件；研究 runner 仍可显式开启。

### refine 何时会“重解码”

这里的 trace 是第一次生成文本时顺手保存的旁路记录：最终 tokens、每个 winner token 的 logprob、
token→frame compact path（显式开启 disfluency 时还包括相应 weights），以及 EOT/decoding-limit 等
终止标记。“使用第一次 decode 的 trace”是指文本搜索结束后直接读取这些记录来生成词时间和
confidence，不再把音频送进 decoder 搜索另一遍。early-EOT 用已保存的 terminal-EOT query，unfinished
用第一次生成已经选中的尾 token；DTW 和 disfluency 只做 alignment 后处理。

因此，生产契约内的 WT refine 行为**不会重新搜索文本**。beam=5 也在 beam 完成后按 parent lineage
重建 winner，不做第二次语义 decode。

下列情况会退回 faster-whisper 的 teacher-force word alignment：非单一 `temperature=0`、不请求
word timestamps、without-timestamps、多个返回 hypothesis 等非主契约，或 decoded segments 与
compact spans/tokens 无法一一核对。该路径有第二次 decoder forward，但 tokens 已固定，只重新求
alignment，不是重新做 beam/greedy 文本搜索。当前 `transcribe_wt` 生产参数满足 1-pass 契约，正常
greedy/beam 都不触发此 fallback。

真正重新转录音频的是外层 ASR controller：word-level abnormal interval isolation、coverage
shortfall 的 beam/peel rescue，以及 uncovered complement recall。它们不是 WT refine 状态机的一部分。

外层的崩溃兜底：compact trace 与 span 对账失败时，同一次调用内固定 tokens 退回 teacher-force
alignment（不重新搜索文本）；那一步也失败才丢弃该 group。历史上 whisper-timestamped 走的是另一条
路——它的 efficient hook 断言 segment 数一致，不一致就抛裸 `AssertionError`，外层只能以
`naive_approach=True` 从头重新转录一次。那不是值得模仿的质量行为，随 wt backend 一并移除。

## 成功口径

分开验收三层，避免把文本分叉误算成 refine 误差：

1. **alignment core**：相同 token、相同 attention matrix 时，token-frame DTW path 与 WT
   oracle 一致。
2. **refine state machine**：相同 decoded token stream、logits、attention 与终止事件时，
   segment flush、缺失 token/end timestamp 修复、分词和 word timestamps 一致。
3. **生产 backend**：FW 自己解码得到的文本经过移植 refine 后，覆盖率、异常救援、confidence、
   segment schema 和下游 DP 分句满足现有生产契约。

生产切换前不以「FW 最终 JSON 与 WT 逐字节相同」为目标：CT2 与 PyTorch 的解码数值路径可以
产生文本分叉；同文本的 refine 等价才是本项目可控制的边界。

## WT 1.15.9 行为冻结

### 解码期 efficient 路径

WT 在一次 greedy decode 中通过 hook 收集：

- encoder MFCC；
- decoder 输入 tokens；
- 选定 alignment heads 的逐步 cross-attention；
- decoder LN 输出 logits（word confidence、末 token 修复）；
- 达到 decoding limit、EOT、timestamp 连续性等终止事件。

每次 segment flush 时：

1. decoding limit 时从下一窗 prompt 或最后 logits 补末 token；
2. 没有 end timestamp 时补 EOT；
3. end timestamp 不晚于 start 时做受约束重估；
4. 调用 `perform_word_alignment`；
5. Whisper 返回最终 segments 后，再按内部 timestamp token 拆回对应 segment。

### `perform_word_alignment` 默认契约

| 项 | WT 1.15.9 默认/生产值 |
| --- | --- |
| refine window | 生产 `1.0s`；左右扩展后 clamp 到 30s 窗 |
| minimum frames | `end = max(end, start + len(tokens))` |
| tokens > frames | 截断正文 token，保留最后 timestamp，递归对齐 |
| median filter | WT 默认 9 |
| alignment heads | 模型内置 alignment heads |
| attention | frame 维 softmax → heads mean → token 维 L2 normalize |
| early cell | 代价矩阵 `weights[0,0] = weights.min()` |
| DTW | `dtw-python stepPattern.symmetric1` |
| empty subword | 默认允许（多个 token 可共用同一 frame） |
| 分词 | 空格语言与 CJK/日文走不同 splitter；标点有独立归属规则 |
| segment 边界 | refine 后跟随首尾 word |

`subwords_can_be_empty=False` 是 WT 的可选严格 step pattern，不得与 `encourage_early`
绑定。现有 CT2 prototype 曾把二者耦合；完整移植从这里开始纠正。

## 实现路线

### M0：oracle 与离线 alignment parity

- `tools/wt_refine_port/oracle.py` 固化 WT 使用的两个 DTW step pattern。
- CT2 单测使用 oracle 生成的固定路径，不在测试时依赖 Python WT。
- patched `Whisper.align` 暴露 frame range、timestamp prefix、early cell、empty-subword
  policy，但各选项保持正交。
- parity 探针还要求 CT2 选取 WT 的两侧 boundary query rows，并使用 WT 的 attention
  后处理次序；这两项与 prefix/DTW step pattern 分别开关，避免消融实验再次耦合。

### M1：post-decode teacher-force 等价路径

先复用 FW encoder output，批量对所有 segment 做 teacher-force alignment。它不是最终最快路径，
但可以最早验证完整 `perform_word_alignment`、token 修复、分词和 confidence 契约。

第一阶段 `teacher_force_probe.py` 固定 FW greedy tokens 与独立扩展窗口，顺序加载 CT2/OW，先验收
boundary-query layout、attention 后处理和 word boundary。多 segment 的前后状态传递、token 修复和
confidence 随后的 state-machine probe 再加入，避免首轮结果混入窗口递归差异。

首个 `hello.flac` 单 segment probe 暴露并拆开了三个原实现差异：CT2 缺少 boundary queries；
CT2 median 使用 whole-sample mirror/fp16，而 WT 使用 Scipy half-sample reflect/float32；CT2 的
`num_frames` 同时承担 DTW 截止与真实音频截止，而 WT 允许 refine window 进入 padding、再只把
非末 boundary rows 的 padding 权重压低。patched CT2 已将三项分别实现，并用独立
`real_audio_frames` 拆开 DTW window 与真实音频边界。该样本在同 token/window 下，CT2 与 WT 的
首词 start/end 均相差 1 frame（20ms）；完整 path 仍有 1-frame 分叉，后续需在多样本上判断是
float kernel/转换权重的正常数值差，还是尚有未冻结细节。

第二阶段 `full_window_probe.py` 不再从后处理后的 `segment.start/end` 反推边界，而是保留 FW
原始 decoded timestamp tokens，并在同一个 encoder output 上对多个 segment 依次 teacher-force
align。为此 patched CT2 的 `align` 新增逐 batch item 的 `prefix_tokens`，使每段从自己的真实
起始 timestamp query 开始。`harvard.flac` 的前 4 段共 28 个词均与 WT 分词一致：4 段中 3 段
全部 start/end 边界落在 1 frame（20ms）内；剩余一段仅末词 end 相差 3 frames（60ms），其他
边界仍在 1 frame 内。各段完整 DTW path 均存在少量分叉，结合边界高度一致，当前将其作为
PyTorch 原模型与转换后 CT2 模型的 attention 数值差继续量化，而不误判为 decoded-span 语义
错误。缺失起止 timestamp 的非正常 segment 会显式进入 `rejected`，由下一阶段 repair state
machine 处理，探针不会用最终 segment 时间静默替代原始 token 状态。

`tools/wt_refine_port/state_machine.py` 进一步把 WT hook 中与框架无关的策略冻结为纯函数：
decoder-limit 判断、连续 timestamp/SOT flush、unfinished fallback、early-EOT 修复、倒退 end
timestamp 的受约束重估、alignment query 取舍，以及 confidence 对 segment 边界 token 的切片。
实现刻意同时保留 alignment 临时 tokens 与 WT 记录的 segment tokens：WT 的 end timestamp
重估只修改前者，这个细节会影响随后 confidence/token 对账，不能被“顺手修正”。

### M2：1-pass greedy efficient 路径

在 CT2 generator 中只导出被选 token 对应的 alignment heads、必要 logits 和终止事件；segment
flush 时直接 refine，不重复 encoder 或 decoder forward。此阶段的计算效率验收按「每单位实际
解码/救援音频的成本」比较，不要求发生额外救援后的总 wall time 等于无救援 FW。

首版 generator trace 已验证这条路线可行：patched `generate(return_attention=True)` 在 greedy、
单 hypothesis 下返回模型配置的 6 个 timing heads，并额外保留 terminal EOT 所对应的最后 query；
对外 decoded tokens 仍去掉 EOT，length-penalty score 也还原到原生 FW 口径。`hello.flac` 的
1-pass DTW path 与同 encoder output 的 teacher-force `align` 逐点相同。`harvard.flac` 前 4 段中
第 1 段逐点相同，后 3 段因 iterative-cache 与 full-sequence kernel 的微小数值差选择了不同的
等价 DTW 步；可见 word 边界中位差为 0--20ms、最大差 60ms。

RTX 5060 Ti/CUDA fp16 热态、只计 decoder 的 5 次中位数（large-v3-turbo）：Harvard 66-token 窗口原生
greedy 为 228.20ms；导出 6-head attention 为 228.89ms（约 +0.3%，2.41MB）；attention 加当前
每步全词表 logits 为 242.64ms（约 +6.3%）。因此 attention 采集本身已经接近原生计算效率；
随后加入的 compact refine trace 已把 logits 压缩为每步 chosen-token logprob 加末两步完整
log-probs；滚动窗口始终留在 GPU，解码结束才传回两行。对 Harvard 采用交替次序的 14 对热态测量，原生/compact 中位数分别为
231.84/240.86ms，即约 +3.9%；这已经符合“单位救援解码效率接近原生 FW”的可行性预期，后续
的 production path 已不再收集或返回末两步全词表行。旧 `return_logits_vocab` 和带 tail rows 的
trace 只保留作 parity/诊断对照，不进入生产路径。

CT2 现已直接在已收集 attention 上运行相同 median/softmax/L2/DTW，只返回逐 span compact path、
临时 repair tokens 与 confidence logits，不再向 Python 展开 2.41MB attention。hello 的 1 段和
Harvard 的 6 段均与旧 Python trace path 逐点相同，path 模式返回的 attention 长度为 0。
同一 Harvard 18 秒窗的 production backend 热态交替 10 对测量（包含 feature/encoder、decoder、
compact path、分词）：原生 FW greedy 且不请求 word timestamps 为 671.63ms，compact refine 为
691.06ms，增量 19.42ms / +2.89%。这已经优于早期 decoder-only trace 的 +3.9% 噪声中位数，说明
完整 greedy refine 的单位计算效率可以视为接近原生 FW。移除 production tail-vocab 复制后的同口径
空闲态最终复测（热态交替 10 对）为 354.49/366.80ms，增量 12.31ms / +3.47%；此前偏高的
绝对时间来自同机其他 CPU/GPU 负载，去除竞争后仍维持低个位数百分比增量。

首个隔离生产 backend 已接入 `vad-asr`，并自 2026-08-02 起是唯一 backend。
`RefinedWhisperModel` 拦截单温度、单返回 hypothesis 的 greedy/beam decode，直接把 compact trace
转为现有 `segments[].words/confidence/no_speech_prob` schema；多温度和无法与 decoded timestamp
span 一一对账的防御性异常仍退回 faster-whisper 自带的 teacher-force alignment。`hello.flac` 端到端
经过现有 `_transcribe_group_candidate` 与 timeline mapper 后无异常，显式计数确认正常路径没有
调用 `find_alignment`，即没有隐藏第二遍 decoder forward。当前 runtime 仍要求本研究分支构建的
patched CTranslate2；未带扩展的标准 wheel 会在模型构造时立即报错，不会静默降级。

compact path 还覆盖 WT greedy state machine 的两个末端状态：

- “有 start timestamp、正文后直接 EOT”的 early-EOT span 使用 terminal EOT query 对齐；即使窗口
  前面已有完整 spans，最后一个 early-EOT span 也会单独 flush。公开 decoded tokens 仍不含 EOT，
  confidence 则按 WT 使用其 chosen logprob。
- 达到 decoding limit 的 unfinished span 直接使用 CT2 已选入结果的最后一个 token（WT hook 之所以
  要从下一 prompt/logits 补 token，是因为 hook 少看了这一步），标记 `unfinished=True`，保留最后正文
  token 的 confidence，并一次消费当前窗口，避免 faster-whisper 把同一尾部重复解码。GPU 强制
  `max_length=12/20/30` 已覆盖纯 unfinished 和“完整 span + unfinished span”，均未调用
  `find_alignment`。

CT2 的 `ApplyTimestampRules` 在生成 end timestamp 时严格禁用“不晚于最近 timestamp”的所有候选；
相等的连续 timestamps 只可能是上一段 end 与下一段 start。因此 WT 的倒退-end 修复分支在 CT2
greedy token stream 中不可达，production path 不再为它逐步复制全词表 logits。至此已知 WT greedy
flush/repair 状态均在同一次 decode 中闭合；下一节在同一 compact contract 上扩展 beam lineage。

### M3：beam search

1-pass beam 已实现，不需要把 WT 式 2-pass 作为生产主路径。CT2 原有 BeamSearch 已经会随
`gather_indices` 维护最终 hypothesis 的 attention lineage，但旧实现把每个 beam 的完整 attention
history 每步 concat/gather，Whisper 的多 timing heads 会把这条路径放大成 O(beam·T²) 的 CPU 内存
搬运。首版直接复用该实现虽能得到正确 winner，Harvard beam=5 却从 native 669.72ms 增至
1305.71ms（+94.96%），不可接受。

最终实现把 beam history 改成 persistent parent-node lineage：每步只保存当前 parent beams 的
timing-head row，每个候选只记录 `parent_node` 和 row index，hypothesis 完成后再反向重建 winner。
同时以 fp16 先传至 CPU、再转 float32，避免为传输提前扩成两倍字节。winner token log-prob 也沿
同一 beam origin 保存；Harvard 102 个含 EOT log-probs 全部有限，`sum(logprobs)` 与
`final_score × public_token_count` 的误差为 `7.6e-7`，证明 score lineage 未串 beam。

RTX 5060 Ti 空闲负载、Harvard 18 秒窗、beam=5、完整 production backend 热态交替 10 对：native
FW text-only 中位数 519.42ms，1-pass compact refine 535.76ms，增量 16.34ms / +3.15%。独立的原生
teacher-force 2-pass 对照为 539.63/534.65ms（-0.92%，落在短窗调度噪声内），因此 wall-time 不足以
给 1-pass/2-pass 排序；确定的结构差异是 1-pass 不再执行第二次 decoder forward，并已达到接近原生
FW 的单位解码效率，故最终选择 1-pass。两条 1-pass 输出与 native beam 的文本、6 段 decoded tokens
完全一致，共 43 词，且 `find_alignment=0`。与 2-pass full-sequence
attention 对照，词边界中位差 20ms；最大 460ms 来自首词 start，属于 iterative-cache 与
full-sequence attention 的既有数值/布局差异。非单温度、多个返回 hypotheses 等非精确契约仍保留
faster-whisper teacher-force fallback。

### M4：可选 disfluency

`remove_empty_words` 曾作为 WT 对照接入，但源码审计确认它只在整次转录物化后删除连续零时长尾词，
不能改善解码或 previous-text prompt，反而会在 FineSub 判断前抹掉 collapse stack 证据并改变 coverage /
dominant interval 分流，因此已经删除。零时长 chunk 尾应原样保留并上报
`zero_duration_chunk_tail`，由救援层结合能量、重复和重解码结果决策。

disfluency 路径只在显式开启 `detect_disfluencies` 时让 CT2 返回 compact DTW 已经算好的
token-frame 后处理矩阵；Python
继续使用 WT 的 `scipy.signal.find_peaks(width=3, prominence=0.02)`，不重复 encoder/decoder forward。
与 WT 1.15.9 有一项有意差异：首个实词前的空-gap 候选仍用于更新该词起点，但不渲染成普通
`[*]` word。WT 多峰检测本身只改词起点；实测的上一段尾词终点收缩来自前置 `[*]` 提前 segment
起点后，后续 overlap clamp 回拉上一段终点。此策略保留多峰定位收益，同时保证 detect 开关不改变
任何实词的原始 end。Harvard beam=5 实测开启后产生 2 个 `[*]` 候选，文本和 43 个原词不变，
并以禁止 `find_alignment` 的探针确认仍是 1-pass。交替 10 对热态中位数为关闭 601.45ms、开启
597.84ms（-0.60%，调度噪声），可视为没有可测量的计算开销。

## 质量优先的缺口取舍

目标顺序改为“时间轴精度、文本质量、召回，其次效率”后，不应把 WT 的所有公开选项逐项照搬：

| WT/现有缺口 | 取舍 | 原因与目标实现 |
| --- | --- | --- |
| `detect_disfluencies` | **已补，显式开关、默认关闭** | 多峰区间约 80% 有定位价值，但句首仍有过度收缩风险；首词前 gap 不渲染为边界词，其他候选保持 WT 的 `[*]` 表达，同时用事件传给上层。所有实词 end 保持不变。 |
| `remove_empty_words` | **删除，替换为非破坏性信号** | 提前删除会抹掉 FineSub 的 collapse stack 证据并改变 coverage/interval 归属；初始候选保留原词，只上报 `zero_duration_chunk_tail`，在隔离救援失败且能量/重复证据支持幻觉时才由上层采用。 |
| 全局 monotonic / `min_word_duration` | **补 validator，不照搬全局摊平** | 1-pass 已人工确认更准；WT 的递归改边会移动相邻正确边界。只修非法重叠/负时长，并输出修复事件。 |
| `tokens > frames` 尾文本截断 | **不照搬删除，改为强异常信号 + 隔离重解** | 直接截正文与召回目标冲突；应输出 density/vertical-run，触发 interval isolation，最后才保留可审计降级。 |
| `remove_punctuation_from_words` / punctuation timing | 低优先级 | 影响展示和 confidence 口径，不解决主要时轴/召回问题；现有生产默认已匹配。 |
| multi-temperature / 多 hypothesis 1-pass | 低优先级 | 生产以单温度、单 winner beam 为主；现有 teacher-force fallback 足够覆盖非主路径。 |
| `trust_whisper_timestamps=False`、WT VAD/backend/plot | 不移植 | FineSub 已拥有 VAD、原时间轴、诊断和 backend 选择，重复实现会模糊所有权。 |

### 应提前到 ASR controller 的逻辑

- coverage shortfall、early-EOT、逐 interval isolation：需要 VAD interval 和重解码权限，继续归 ASR
  controller；不能放在纯 refine 中。
- repetition motif、same-word run、collapse word stack：检测可在 decoder/refine 更早产生事件，但是否
  截断/重解必须由 ASR controller 结合能量、覆盖和上下文决定。
- 低能量/音乐幻觉、固定日语幻觉、跨 segment 精确复读：分别依赖声学上下文或完整 segment 文本，
  留在 ASR/stabilization；不要塞进 CT2 token loop。
- previous-text prompt 污染：当 decoder 报告强 collapse 时，由 ASR controller 清 prompt 并隔离坏
  interval；refine 只上报，不自行更改跨窗状态。

### refine 应新增的质量逻辑与信号

refine 层适合做“观测和局部边界约束”，不适合静默删文本。按优先级建议：

1. `alignment_stack`：DTW 同 frame 的 token 数、最长 vertical run、tokens/active-frames；直接替代
   上层仅靠 25ms word stack 的滞后检测。
2. `attention_stall`：winner attention cursor 连续多少 token 未推进，配合 decoder token motif；两者
   同时强才允许 force legal timestamp/EOT 作为熔断。
3. `disfluency_candidate`：原区间、收紧后起点、peak count/prominence、是否首/尾词；默认只传信号。
4. `boundary_uncertainty`：首尾 query 的峰间距、峰值比、entropy、是否碰 refine window/padding；供
   FineSub 决定保守边界或局部重对齐。
5. `zero_duration_tail` / `dense_alignment` / `unfinished`：保留原 tokens、候选删除范围和 repair 原因，
   让上层能做可逆比较。
6. `beam_consensus`：winner/runner-up 的文本前缀分歧、时间 cursor 分歧和 margin；比单一 confidence
   更适合判断“文本确定但时间不稳”与“文本本身不稳”。

建议统一输出为结构化 `decoder_events` / `alignment_events`，而不是注入可见特殊 token。只有模型原生
timestamp/EOT 可以作为受约束动作；任何强制动作都必须保留 raw tokens、标记
`forced_collapse_boundary`、清除坏尾 prompt，并交给 interval isolation 复核。

第一轮 13-group 异常倾向验证、path/attention 信号开销、greedy isolation 与 1-pass beam=5 对照见
`docs/wt-refine-validation.md`。生产 ASR 现已收集并透传 checkpoint 信号，但尚不依据这些新事件路由；
FineSub 也未解析。研究 runner 继续负责策略验证。

## 性能记账

必须分别记录：

- 原生 FW decode；
- attention/logits 采集增量；
- refine CPU/GPU 后处理；
- teacher-force fallback；
- production coverage/abnormal/recall 重解码的音频秒数。

核心比值是 `各路径计算秒 / 该路径实际处理的音频秒`。例如救援重解 20% 内容时，这 20% 的
单位计算效率应接近对应的原生 FW greedy/beam，而不是要求整任务 wall time 不增加。

## multi-audio batch

测量环境：RTX 5060 Ti 16 GB · 96 个真实生产窗口（kaguya60 的 74 个 + yui 的 22 个
`build_alignment_groups` 分组，全部 ≤30s，combined audio 中位 27.9s）· `min/小时` 按实测的
74 组/小时素材折算 · 显存均为**进程占用**（设备采样值减去本机桌面常驻 1.35 GB）。
Windows 上 conda 与 py-3.12 均无 triton，openai-whisper/WT 的 DTW 走 fallback 实现——
生产同样如此，故下表与生产口径一致。

### 为什么值得做，以及它排在哪

| 配置 | min/小时素材 | 相对今天 |
| --- | --- | --- |
| `whisper-timestamped` greedy（**今天的生产**） | 3.54 | 1.00× |
| `fw-refine` turbo greedy B=1（**不开批**） | 0.56 | 6.3× |
| `fw-refine` turbo greedy B=16 | 0.31 | 11.4× |

**迁移本身值 6.3×，batch 只再叠 1.8×。** 所以 batch 不得阻塞 P0；它是迁移完成后的增量优化。
batch 的另一重价值是把「换 large-v3」的代价从 3.9× 压到 1.25×（见下表）。

### CT2 侧改动（已完成）

`WhisperOptions::real_audio_frames` 由 `size_t` 改为 `std::vector<size_t>`：空向量表示用满
encoder 输出，单值广播到全批，否则长度必须等于 batch。改动三处——`whisper.h` 字段、
`whisper.cc` 的批大小校验与按 item 取值、`python/cpp/whisper.cc` 的
`variant<size_t, vector<size_t>>` 绑定（形状照抄同文件里 `align()` 已有的写法）。

decoder 侧**无需改动**：generate 的结果循环本来就按 item 遍历，greedy 的
`token_logprobs[step.batch_id]` 按 batch 索引，beam 的 `alive_attention_nodes` 在 beam 重排与
**batch compaction** 两处都会 gather 且 node id 是全局绝对索引。

验证契约：同样窗口与 prompt 下，batch=N 的每个 item 与 N 次 batch=1 对比 tokens、
`token_logprobs`、以及 `refine_alignments` 的每个 span。覆盖各 item 不同 `real_audio_frames`、
提前 EOT（触发 compaction）、beam=5 的 winner lineage、错长度向量被拒、标量广播等价。

### 两种批模式：必须用 split-encode

encoder **完全不吃批**——每窗 145–148 ms，B=1 到 B=16 恒定（两种计时方法互校、每个形状单独
预热、特征上传移出计时区）。收益全部来自 decoder。

因此逐窗口 encode（batch=1）、只把 decoder 批起来，**吞吐一分不亏**，却消除了 fp16 分叉的
主要来源。实测（32 窗口 × 2 片源，greedy）：

| 模式 | kaguya60 文本分叉 | yui | 词起点变化 |
| --- | --- | --- | --- |
| full batch（encode 也批） | 7/32 | 2/32 | 2.3–3.7%，尾 103 帧 |
| **split-encode** | **0/32** | **1/32** | **0.1–0.5%**，尾 1–22 帧 |

实现要点：一次性把 B 个窗口的特征上传显存，在设备上切片后逐片 encode，再把各自的 encoder
输出零拷贝拼成批张量（`torch.as_tensor` ↔ `StorageView.from_array` 双向都支持）。

### 确定性契约

批处理**不是**「同样结果跑得更快」，它会改变输出：

- 分叉是「批 vs 不批」的**二元属性，不随 B 增长**：B=4 与 B=16 的分叉统计逐项相同。
- **无系统偏置**：词起点带符号偏差均值 <0.05 帧（1 ms），正负号大致对半。
- 单发重跑自检为 0 差异；同一形状重复运行逐位一致。

由此产生两条硬约束：**ASR partial checkpoint 的 fingerprint 必须带上 backend 与批配置**
（现有 key 只有 model/language/gap），否则一次 resume 换了批大小就会在同一份产物里混进两种
数值状态；**批与非批的产物对照中，20 ms 级差异不是回归**。

### 设计规则

- **波次批处理**：只有单窗口（≤30s）item 进批；跨窗口分组的第 2 窗起 prompt 非空，退顺序路径。
  实测生产分组 window/组 为 1.00–1.22，>30s 占比多数片源 0–22%。
- **prompt 必须同形状**。CT2 的 `check_prompts` 要求批内 SOT 位置与任务 token 数一致（内容可不同，
  故混语言合法）。生产每组是独立 `transcribe()` 调用、不传 prev text，天然满足——**写成断言，
  不要写成 pad**：pad 会真实改变条件化内容。上游 `BatchedInferencePipeline` 的做法同样是
  「一个 prompt 复制 N 份 + 就地替换语言 token」。
- **投机执行**（仅在保留动态分组时需要）：`align_segments` 当前每轮重新分组，isolation 会吐回
  未消费区间使后续计划失效，此时批内首个触发 isolation 之后的结果必须**作废重算**、不得将就使用。
  **若按计划改为静态分组**（isolation 只影响它所在的组、不重排后续分组），投机与回滚都不需要，
  批大小也不再受 isolation 率约束——下表的档位正是按这一前提取值。
- **不要批 beam**：turbo 上 beam 峰值仅 1.11×（B=8 起负收益），且 beam 的分叉来自 decoder 的
  假设排序，split-encode 修不掉。已排除长度离散度这一混杂——把窗口挑到 max/med=1.03 后曲线不变。
- **不做长度分桶**：TIGHT 与 SPREAD 在 greedy B=8 只差 5%（1.85× vs 1.76×），不值得打乱投机回滚顺序。
- **批大小必须由预算静态推导**：CT2 遇 CUDA OOM 是**进程级硬中止**，接不住，不能靠试探自适应。

### profile 档位

`usable = gpu_budget_gb − GPU_SYSTEM_RESERVE_GB(1.0)`。BS 为生产配置值；耗时与显存为本机实测。

| 模型 | beam | 4 GB | 8 GB | 12 GB | 16 GB |
| --- | --- | --- | --- | --- | --- |
| turbo | 1 | **8** | **16** | **24** | **32** |
| turbo | 5 | **2** | **2** | **3** | **4** |
| large-v3 | 1 | ✗ | **8** | **24** | **32** |
| large-v3 | 5 | ✗ | **8** | **8** | **8** |

| 模型 | beam | B | min/小时 | 进程显存 | 档位余量 |
| --- | --- | --- | --- | --- | --- |
| turbo | 1 | 8 | 0.33 | 2.92 | 4 GB 档余 0.08（**2.7%，最薄的一格**） |
| turbo | 1 | 16 | 0.31 | 3.95 | 8 GB 档余 3.05 |
| turbo | 1 | 24 | 0.31 | 5.02 | 12 GB 档余 5.98 |
| turbo | 1 | 32 | 0.32 | 6.08 | 16 GB 档余 8.92 |
| turbo | 5 | 2 | 0.75 | 2.08 | — |
| turbo | 5 | 3 | 0.74 | 2.21 | — |
| turbo | 5 | 4 | 0.76 | 2.40 | — |
| large-v3 | 1 | 8 | 0.71 | 6.15 | 8 GB 档余 0.85 |
| large-v3 | 1 | 24 | 0.54 | 10.22 | 12 GB 档余 0.78（7%） |
| large-v3 | 1 | 32 | 0.52 | 14.24 | 16 GB 档余 0.76（5%） |
| large-v3 | 5 | 8 | 1.42 | 6.24 | — |

`large-v3` 基线 4.34 GB 已超 4 GB 档可用量，该档只能用 turbo。

两条本机实测事实，与上表**不矛盾但需知情**：turbo 在本机 B>16 不再变快（0.31→0.31→0.32），
`large-v3` B=24 已达 B=32 的 96%；档位仍取较激进值，是因为更强的卡上 decoder 饱和点更高，
而显存余量充足。**本机上 B=64 是真实的墙**：turbo 退化到 0.34，`large-v3` 塌回 B=1 水平
（进程 14.6 GB 却仍在颠簸），更早一次同配置直接进程硬中止。

按 item 的显存斜率（用于外推别的档位）：turbo 约 **0.13 GB/item**、`large-v3` 约
**0.28 GB/item**（B>16 后超线性，B=32 比线性模型多约 1.2 GB）。beam=5 的显存与 greedy 几乎相同，
说明 CT2 的 cross-attention K/V 按 batch item 而非 batch×beam 存储。

### 顺带发现：可复用的 encoder 输出

`transcribe.py` 的覆盖率 beam 救援用**同一个 group、同一份 audio** 重跑，`build_combined_audio`
产出逐位相同的采样，encoder 输出必然相同，但当前实现走完整 `model.transcribe()` 会重算一遍——
按每窗 452 ms 中 146 ms 是 encode，这条路径可省约三分之一，零风险。

反之，**异常隔离与 coverage recall 改了音频，不可复用**：Whisper encoder 是 1500 位置全连接的
双向自注意力，改动任一采样点会改变整个输出，不存在 causal LM 那种前缀 KV 复用。对隔离而言
这还是**设计意图**——隔离的目的正是把污染性邻近音频移出模型视野。

### batched driver（`transcribe_batch`，2026-08 实现）

`fw_refine_backend.transcribe_batch(model, audios, language=..., beam_size=...)`
接收一组 ≤30s 窗口，返回与 `transcribe_wt` 同 schema 的结果列表。三步：

1. **逐窗口 encode（batch=1）** —— split-encode，因为 encoder 批处理增益为零；
2. **一次 `generate`** —— 拼接 encoder 输出（GPU 上零拷贝），N 个相同 prompt，逐样本
   `real_audio_frames`。这是唯一的批处理点；
3. **逐条回放** —— 把第 i 条结果装进 `_Playback`，再走一遍普通 `transcribe()`。
   `encode()` 与 `generate_with_fallback()` 在回放模式下直接返回缓存值。

**回放而不是重写 seek 循环，是为了让 segment/词/事件的组装只有一份实现**，与顺序路径的
一致性因此是构造性的，deque 状态也不必改成逐样本。

实测（24 个真实生产窗口、turbo greedy、B=8）：**文本一致 22/24，其中 18 条连词级时间都逐位
相同；加速 1.73×**。剩余分歧属于[确定性契约](#确定性契约)——批与非批不可逐位复现。

三个必须知道的边界：

- **只有窗口的第一次解码走批。** faster-whisper 的 seek 循环在解码提前结束时会对同一窗口再解
  一次；批里没有那一份，`_Playback` 用尽后自动回落到顺序路径。多数窗口用不到第二次。
- **`combined_group_duration ≤ 30s` 不保证拼接音频 ≤ 30s。** gap padding 会把一部分分组顶过
  编码窗口（实测 29 个分组里 5 个），driver 的校验会拒绝它们——调用方需要自己处理。
- **特征必须在特征域补齐**（`pad_or_trim(features)`），不能把音频零填充到 30s 再算 mel：
  数字静音的 log-mel 是个很大的负常数而非 0，encoder 看到的填充区完全不同。实现时踩过这个坑，
  文本一致率从 5/24 跳到 22/24。

### 上层暂不组批（当前状态）

**调用方一律以 B=1 调用**：`align_segments` 逐 group 顺序解码，`transcribe_batch` 虽然可用，
生产路径上没有调用者。profile 档位表里的 B 上限因此**目前不生效**，它记录的是显存能承受多少，
不是当前会用多少。

暂缓的理由是收益已经被摊薄：换到 fw-refine 后**人声分离占语音段 72%**，ASR 只剩 28%。
driver 自身的 1.73× 折算到整条语音链约 9%，而组批要引入波前调度、isolation 掉队处理、
批配置进 checkpoint 等一批状态。**下一个值得优化的阶段是人声分离，不是 ASR。**

### 可选待办：真要做上层组批时

按依赖顺序，每条都可独立验收：

1. **静态组批 + 波前推进**。一次取 B 个 ≤30s 的 group 组成一批；解码提前结束需要第二次
   seek 的 item 掉出本批，与下一波一起再组。B 由 GPU profile **静态推导**——CT2 的 CUDA OOM
   是进程级硬中止，不能试探自适应。
2. **prompt 非空退顺序**。依赖 `condition_on_previous_text` 的连续 group（>30s 分组的第 2 窗起）
   prompt 非空，CT2 要求整批同一 prompt 形状，直接退顺序路径即可，不必 pad。
3. **>30s 分组的处理**。`combined_group_audio_seconds ≤ 30s` 已是分组规划的判据，但仍有约
   14% 的组超窗（密集语音找不到切点，最长 73.9s）。这些组要么走顺序路径，要么先决定
   「是否为适配编码窗口而强制切分」——那是独立的质量取舍，见 [`asr-align.md`](asr-align.md)。
4. **isolation 与批的交互**。异常隔离会改音频、必须重解，实测异常率 ≈14.5%。静态分组下这不影响
   批的组成（掉队者下一波再来）；若将来恢复动态调整分组，需要先重估投机浪费。
5. **checkpoint fingerprint 补批配置**。批大小改变解码的 fp16 归约顺序，因而改变输出。
6. **端到端计时**。现有 1.73× 是 driver 自身的，不含组批开销。

不建议做的：**encoder 批处理**。实测 B=1→16 吞吐完全不变（145–148 ms/窗口，两种计时法一致），
split-encode 已经是最优形态。

### 未验证项

- 上表档位以**静态分组**为前提。若仍保留动态分组，需先按实测 isolation 率（≈14.5%）重估：
  批内可用比例按 p 估算，B=16 在 p=1%/3%/5% 下为 93%/80%/70%，B=32 为 86%/65%/50%。
- driver 只在 turbo greedy B=8 上验证过；beam=5、large-v3 与更大 B 未测。
- 端到端吞吐（含调用方组批开销）未测——1.73× 是 driver 自身的。

## beam 不抑制幻觉与复读（2026-08 实测）

310 个真实生产窗口（10 个 clip，`build_alignment_groups` 的 ≤30s 分组），greedy 与 beam=5 各解一遍，
用生产自身的判据打分：`text.detect_abnormal_asr_words`（触发隔离的那套）、
`text.COMMON_HALLUCINATION_TEXT`、faster-whisper 自己的 `get_compression_ratio > 2.4`，
以及 fw-refine 的 path signals。

| 指标 | greedy | beam=5 |
| --- | --- | --- |
| **含任一生产异常的窗口数** | **45/310** | **48/310** |
| `hallucination_phrase`（ご視聴…） | 11 | **22** |
| `collapse_word_stack` | 47 | 52 |
| `signal:alignment_stack` | 246 | 264 |
| `compression_ratio_high` | 25 | 25 |
| `repeating_group_cycle` | 20 | 20 |
| `repeating_word_run` | 3032 | **2528** |
| `signal:long_token_span` | 24 | 21 |

**beam 没有抑制幻觉，窗口级异常率反而略升（+7%），已知幻觉短语翻倍。** 唯一稳定的改善是逐词复读
（−17%）与 `long_token_span`（−12%）；复读那个计数会被长串放大（超阈值后每个词各记一次），
方向可信、幅度不可直读为「复读发生率降低 17%」。

结合成本（beam=5 最优 0.75 vs greedy 最优 0.31 min/小时素材，2.4×），**不应为抑制幻觉或提升质量
而切到 beam**。这与 [`wt-refine-validation.md`](wt-refine-validation.md) 记录的「5 个 hard group 上
4/5 与 greedy 文本等价、无 beam 独有胜例」一致；beam 作为 coverage rescue 的第二候选仍保留，但那是
「换一条搜索路径再试」的价值，不是「beam 更少幻觉」。

**口径边界**：生产分组已剥除静音，而幻觉主要发生在静音/低能量段。本结论成立于「生产实际喂给模型的
输入」，不可推广到未剥静音的长音频。

## large-v3 不比 turbo 更干净（2026-08 实测）

同一批 310 个生产窗口、同一套判据，两个模型都跑 greedy：

| 指标 | turbo | large-v3 |
| --- | --- | --- |
| **含任一生产异常的窗口数** | **45/310** | **41/310** |
| `long_word_duration` | 22 | 17 |
| `signal:long_token_span` | 24 | 17 |
| `collapse_word_stack` | 47 | 43 |
| `hallucination_phrase` | 11 | 10 |
| `repeating_word_run` | 3032 | **3463** |
| `signal:zero_duration_chunk_tail` | 30 | 34 |
| 解码总耗时 | 177s | 597s（3.4×） |

**窗口级异常率只差 4 个窗口（−9% 相对），方向还互相矛盾**：长词/长 token span 降了两成，
逐词复读反而多了 14%。付出 3.4× 的解码成本换不到这个代价下值得的干净度。
本次只记了聚合数，没记 per-window 的 discordant pair，因此不能对 45 vs 41 做配对显著性检验——
但 4/310 的差距无论如何都不足以支撑换模型。

**这条测量只能证伪，不能证实**：异常率高必然更差，异常率持平不等于语义相当。

### 文本语义抽查（无音频，仅文本对照）

在 BV1cqLR6hEp3 上把两个 arm 的逐段输出并排（159 个时间桶），按「recall + 不乱说」和
「用词准确（含同音/近音词）」两条标准人工核对，明显异常（复读/坍缩/已知幻觉短语）因生产会
自动修复而不计分：

- **用词准确：互有胜负。** turbo 对的：`タルタリア`（vs `タレタリア`）、`こもり歌`（vs 造出的
  人名`小森ゆうた`）、`聖者の彼岸`（vs `悲願`）、`後半は泣いてた`（vs `後輩が`）。
  large-v3 对的：`フォンテーヌ`（vs `ボンテーヌ`）、`パジャマ姿`（vs 不成词的`パジャマスガル`）、
  `老いぼれ`（vs `追いぼれ`）、`厳しい冬来ちゃった`（vs `冬着ちゃった`）。
- **recall：打平。** 单边空桶 turbo 21 / large-v3 19；两边都有内容的 117 桶字符数差中位为 0。
  large-v3 覆盖语音 319.8s vs turbo 285.7s 但只多 28 个词——是时间轴更松，不是内容更多。
- **不乱说：turbo 很弱地领先**，只有 2 处 large-v3 独有的语义崩坏，turbo 无同级失误。

⚠️ **逐段对照最大的陷阱是分桶伪影**：一句话落在相邻桶就会看起来像「一方漏识」。本次初判的
10 处 recall 差异里有 4 处经复核是伪影。判「漏识」前必须看上下相邻行。

## 词起点边界准确度（2026-08 实测）

用 [`tools/wt_refine_validation/disfluency_gold.json`](../tools/wt_refine_validation/README.md)
的 61 个人工标注块打分：标注给出后词的真实起点，误差 = 运行报告的词起点 − 标注起点
（负号 = 起早了）。匹配率 60/61。fw-refine 时间戳量化到 20ms。

两个 fw-refine arm 的配置**只差 model 名**，全部走生产默认（`metadata.asr_align` 实录）：
`detect_disfluencies=false`、`beam_size=None`（贪心）、`temperature=0.0`、`seed=0`、fp16、
`refine_sec=1.0`、`language=ja` 显式、`gap=0.3`、16GB profile（wt_workers=3）；beam=5 只在
异常组的 rescue 梯子上出现。

中位 |误差|（ms），以及与 turbo 逐条配对的胜/负/平（阈值 20ms）：

| 类别 | n | WT 关闭 disfluency（今日生产） | fw-refine turbo | fw-refine large-v3 |
| --- | --- | --- | --- | --- |
| **segment-boundary** | 11 | 329 | **225**（4/3/4） | 210（4/6/1） |
| after-gap | 9 | 317 | **64**（6/3/0） | 285（4/2/3） |
| mid-phrase | 39 | **0** | 37（4/10/25） | 149（31/3/5） |
| 全部 | 59 | 217 | **68**（14/16/29） | 160（39/11/9） |

**结论一：fw-refine 没有修好 segment 首的起点问题。** 全局中位数看似从 217ms 降到 68ms，
但逐条配对是 14 胜 16 负 29 平——中位数的改善来自分布形状，不是逐例更准。在最关键的
segment-boundary 上是 4/3/4，纯平局；两者都系统性地把词起点提早约 0.26–0.33s（把 filled
pause 吞进后词）。真正稳定的改善只在 after-gap（6/3/0）。

**结论二：turbo vs large-v3 的全局差距是标注锚点造成的，不能当作模型差距。**
标注是在一次 **turbo 系运行**（WT backend + `large-v3-turbo`）的词级输出上修改而成，
块首块尾都是那次运行的词边界，因此**同模型的 arm 天然占便宜**。按标注 label 拆开就看得很清楚：

| 标注 label | n | 中位 turbo / large-v3 | turbo 胜/负/平 | 金标准起点来自 |
| --- | --- | --- | --- | --- |
| `word_onset` | 25 | 9 / 83 ms | **19/2/4** | 块首 = turbo 系 run 的词边界 |
| `partial` | 3 | 162 / 48 ms | 1/1/1 | 块内人工点 |
| `filled_pause` | 31 | 232 / 320 ms | 19/8/4 | 块尾 = disfluency run 的切点 |

全局 39:11 基本由 `word_onset` 那 25 行（占 42%）贡献，而那些行的「真实起点」就等于 turbo 系
run 的块首。**在标注者指定的重点子集上，结论反而相反**：

| 子集 | n | 中位 turbo / large-v3 | turbo 胜/负/平 |
| --- | --- | --- | --- |
| segment-boundary（全部） | 11 | 225 / 210 | 4/6/1 |
| segment-boundary（两边文本都精确匹配） | 9 | 225 / 210 | **3/6/0** |
| segment-boundary ∩ `filled_pause`（偏向最小） | 7 | 271 / 419 | 3/4/0 |

即 **large-v3 在 segment 首上略优**，控制掉分词差异后更明显。「large-v3 词时间戳更差」只在
mid-phrase 上成立（28/1，控制分词后），而那正是偏向最重、标注者又明确说「由后词吸收问题不大」
的子集，**不能拿它当反对换模型的理由**。

结合本文档另两节（异常率 41 vs 45 属噪声；文本语义抽查互有胜负），对 large-v3 的结论应表述为
**「没有证据支持它更好」，而不是「它更差」**；在 3.4× 解码成本下，前者已足够支撑不换。

**口径边界**

- **标注锚定在一次 turbo 系运行上**，任何跨模型比较都带此偏向；`filled_pause` 的块尾虽然不是
  turbo 会产生的边界，但块首块尾同出一次运行，large-v3 若在别处听到词起点仍会被扣分。
  要做无偏的跨模型时间戳比较，需要一份不依赖任一 ASR 运行的独立标注。

- **两个 arm 都关着 disfluency，结构上就不可能把 filled pause 切出来**，该块必然被后词吞掉。
  这正是 segment-boundary 上系统性提早的来源。因此本表测的是「不切 disfluency 时 DTW 会不会
  自然把起点放得靠后」，答案是基本不会；**它不能读作「fw-refine 的 disfluency 能力不比 WT 强」**
  ——那条路径（CT2 patch 0009–0010 提供的 disfluency weights）根本没启用。要回答那个问题需要
  开 `detect_disfluencies=true` 重跑，并换一个不依赖 WT disfluency run 的评分口径。
- `segment-boundary` 只有 11 例（最干净的那个切片只有 7 例）、`after-gap` 9 例，够做定性判断，
  不足以支撑小差异的显著性结论。**看重点子集时优先看配对计数，全局中位数会被 mid-phrase 主导。**
- 61 例中有 1 例（461.5s）三个 arm 全部解错文本，误差 ~1.0s，会主导均值——故表中用中位数与
  配对计数，不用均值。
- 「WT 开启 disfluency」那一列没有列出：它在 `filled_pause` 行上与标注同源（标注保留了该运行的
  边界），误差恒为 0，不是独立测量。

## 迁移验收（2026-08-02）

5 个素材 / 50.6 分钟音频，两个 backend 各跑完整语音链（vad-asr → stabilize → to-srt + 时间轴
后处理），语言 auto，16GB profile。产物在 `out/acceptance/<clip>/`。

| clip | 时长 | 耗时 wt→fw | 词数 wt/fw | 覆盖秒 wt/fw | 生产异常 | 文本相似 |
| --- | --- | --- | --- | --- | --- | --- |
| BV1UBjq6fEgb | 3.1m | 62→16s | 759/770 | 159/162 | 0/0 | 85.9% |
| BV1kYLR6AEXv | 4.5m | 58→14s | 778/781 | 158/157 | 0/0 | 99.7% |
| BV1cqLR6hEp3 | 8.9m | 163→44s | 1168/1175 | 286/285 | 0/0 | 91.8% |
| BV1dwjP6LECU | 10.6m | 84→31s | 1714/1707 | 348/347 | 0/0 | 97.2% |
| yingtao | 23.5m | 188→69s | 3350/3343 | 766/758 | 0/1 | 95.2% |
| **合计** | **50.6m** | **555→174s（3.19×）** | | | | |

**内容量无损失**：词数与覆盖秒每个素材都在 ±1% 内。稳定化后的生产异常判据 5 个素材共 1 条。

**救援活动一致下降，无一反例**——比速度更有意义，说明 fw-refine 的一遍解码本身更少产生需要
抢救的结果：

| 事件 | wt | fw-refine |
| --- | --- | --- |
| 异常 interval 隔离 | 65 | 40 |
| 异常触发 | 31 | 20 |
| 覆盖率救援 | 10 | 8 |
| beam 救援 | 20 | 16 |
| 合并清理兜底 | 5 | 3 |

**未完成**：人工对听。相似度最低的 BV1UBjq6fEgb（85.9%，41 处差异）多为正字法差异
（`かわいい`/`可愛い`、标点），但有实质分歧，且恰好是 wt 做了 6 次隔离而 fw 一次都没有的素材。

**口径**：单次运行未取噪声基线；3 倍级差距稳，救援次数那种十几到几十的计数需复跑确认。
aligned metadata 只记配置语言（auto=None）不记实际检测语言，**两个 backend 的语言检测是否一致
未验证**。

## 硬件要求（2026-08-02 实测）

| | wt | fw-refine |
| --- | --- | --- |
| 峰值内存 | 3.77 GB | **2.23 GB** |
| 峰值显存（w=1） | 3.57 GB | 3.47 GB |
| GPU 架构 | sm_70–sm_120（torch 自带） | **取决于构建** |
| CPU 回退 | 可用（慢） | **取决于构建** |

内存差距来自模型加载：torch 要先在主机内存物化 checkpoint（+1.68 GB），CT2 直接加载到显存
（+0.15 GB）。显存在同 worker 数下基本持平——省显存来自 worker=1，不是 backend 本身。

后两行标"取决于构建"是因为 2026-08 的首次构建两项都踩了坑：只编了 sm_86、且关掉了全部 CPU
GEMM 后端。**这两项由构建标志决定，不在补丁里**，要求见
[`ct2-patches/README.md`](../tools/wt_refine_port/ct2-patches/README.md)。

## 变更边界

M0/M1 不改变生产 VAD、group、coverage 或异常隔离参数。生产 backend 接入时才修改
`speech/recognition`，且必须同步 `docs/asr-align.md`、测试和 run metadata。
