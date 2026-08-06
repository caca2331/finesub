# CT2 WT refine 研究交接

本文是 `whisper-timestamped`（WT）refine 全链路移植研究的交接入口，记录为什么做、如何验证、
已经得到什么、当前代码能做什么，以及下一位开发者应从哪里继续。算法契约和实验细节分别以
[`wt-refine-port.md`](wt-refine-port.md) 和
[`wt-refine-validation.md`](wt-refine-validation.md) 为准；本文不重复充当完整规格。

## 一页结论

- 启发式段首修补已经不能继续解决坍缩、unfinished、early-EOT 和多人声等 decoder/alignment
  内部问题，所以研究目标改为：把 WT refine 所需的 decoder trace 与状态机完整迁入
  faster-whisper（FW）/CTranslate2（CT2）。
- greedy 和 beam=5 的 **1-pass compact refine 已经跑通**。文本生成时保存 winner lineage、chosen-token
  logprob、terminal EOT/decoding-limit 和 timing-head path，生成结束后直接形成词时间与 confidence；
  正常生产契约不再做第二次 decoder forward。
- 人工审核 `assets/harvard.flac` 的 beam=5 对照后，**1-pass 时间轴准确，2-pass 不精确**。因此 checkpoint
  选择 1-pass；2-pass 只保留为非主契约的 teacher-force fallback 和研究 oracle。
- 单位计算效率已经接近原生 FW：Harvard 18 秒窗的完整生产 backend，greedy 增量约
  `+2.89%`～`+3.47%`，beam=5 增量约 `+3.15%`。这满足“若只救援 20% 音频，则这 20% 的单位解码效率
  接近原生 FW”的目标；不要求发生救援后的整任务 wall time 不增加。
- 修复后的 `detect_disfluencies` 已实现，但默认关闭。它保留多峰起点定位，不再让前置 `[*]` 通过
  overlap clamp 错误缩短上一段尾词；所有实词原始 end 保持不变。
- `remove_empty_words` 已删除。源码审计表明它只在整次转录物化后删除连续零时长尾词，不改善解码
  或 prompt，反而会抹掉 FineSub 判断 collapse 的证据。现在保留原词并上报
  `zero_duration_chunk_tail`。
- 低成本 path signals 已默认收集并通过 segment `alignment_events[]` 传到 aligned/stable JSON；
  ASR controller 与 FineSub **尚未消费**这些事件。
- 端到端 backend 目前仍以 B=1 调用。`transcribe_batch` 已实现并验证（22/24 文本一致、1.73×），
  但**没有生产调用者**——换到 fw-refine 后人声分离占语音段 72%，组批收益被摊薄，故暂缓。
- 改动必须落在 **CTranslate2**，而不是 faster-whisper。1-pass 必需的 attention、beam parent lineage、
  终止事件和 logprob 在 CT2 内部产生；只改 FW 无法在不做第二遍的前提下恢复它们。
  **交付方式后来定为「钉版 + patch series + 预编译」而非 fork**，见
  [`../tools/wt_refine_port/ct2-patches/README.md`](../tools/wt_refine_port/ct2-patches/README.md)。

## 研究目标与成功口径

优先级已经明确为：

1. 时间轴精度；
2. 文本质量；
3. 召回；
4. 单位计算效率；
5. 总 wall time。

研究不要求 CT2/PyTorch 的最终文本逐字节相同。验收分三层：

1. 相同 token 与 attention 时，CT2 DTW path 对齐 WT oracle；
2. 相同 decoded trace 时，segment flush、early-EOT、unfinished、分词、confidence 和词边界遵守
   WT 状态机；
3. FW 自己生成的文本进入现有 VAD/group/coverage/DP 分句链路后，输出 schema、时间轴和异常救援满足
   FineSub 的生产要求。

`refine` 的职责是消费第一次 decode 的 trace，完成局部 alignment 和可审计观测；真正重新搜索文本的
隔离、coverage rescue、peel 和 recall 仍属于外层 ASR controller。

## 代码与 checkpoint

### ASR / FineSub 侧

**已于 2026-08-02 合入 `dev`**（快进 31 个提交，`57c7400` → `f768a9e`），研究 worktree 与
`codex/wt-refine-port` 分支已删除。本节此后只作为实现入口索引。

核心实现入口：

- [`fw_refine_backend.py`](../src/asr_playground/speech/recognition/fw_refine_backend.py)：
  `RefinedWhisperModel`、compact trace 接管、teacher-force fallback、模型池和 segment events；
- [`fw_refine.py`](../src/asr_playground/speech/recognition/fw_refine.py)：WT 分词、confidence、path signals、
  disfluency 与边界事件；
- [`transcribe.py`](../src/asr_playground/speech/recognition/transcribe.py)：backend 参数、异常候选和原时间轴映射；
- [`stage.py`](../src/asr_playground/speech/recognition/stage.py)：`vad-asr` 阶段编排与 metadata；
- [`segmentation.py`](../src/asr_playground/speech/postprocessing/segmentation.py)：全局 DP 后将一个 event 只归属
  到一个输出 segment。

### patched CT2 侧

- worktree：`C:\Users\Carl\Documents\Carl\projects\CTranslate2`
- branch：`codex/wt-refine-ct2-research`
- upstream base：`v4.8.1` / `0d8bcd3`
- tip：`dcc02ac`（`feat(whisper): accept per-sample real audio frames`）——multi-audio batch 的唯一
  CT2 前置，已编译并通过等价性验证，见 [`wt-refine-port.md`](wt-refine-port.md)
- 源码提交完整；`build*/`、`install*/` 是未跟踪的本机构建产物。
- **交付形态是 patch series，不是 fork**：11 个补丁已导出到
  [`tools/wt_refine_port/ct2-patches/`](../tools/wt_refine_port/ct2-patches/)，基线 `v4.8.1`
  （`0d8bcd3`），本仓库据此可独立复现 CT2 侧改动。

CT2 侧关键改动集中在：

- `include/ctranslate2/models/whisper.h`：refine options/result schema；
- `src/models/whisper.cc`：compact alignment、padding/真实音频边界、WT attention 后处理；
- `src/decoding.cc`：chosen logprob、EOT 和 beam persistent parent-node lineage；
- `python/cpp/whisper.cc`：Python API；
- `src/dtw.*`、`tests/dtw_test.cc`：WT step pattern 与 oracle 测试；
- `src/ops/median_filter*`：WT 所需的 reflect/float 语义。

patched runtime 当前从本机 `python\build\wt-refine-runtime-wide` 与 `install-cu-wide\bin` 加载
（宽架构构建，含 sm_70–90 原生 SASS + sm_90 PTX + Ruy CPU 后端）。
**仍未形成可发布、可复现的 wheel**——这不阻塞本机开发，但阻塞任何形式的分发，也是
`ctranslate2` 的 local-version pin 尚不能启用的原因。

## 研究过程

### 1. 冻结 WT 1.15.9 行为

先读 WT 源码并将 DTW step pattern、attention 后处理、boundary query、padding、分词、confidence、
segment flush 和末端修复拆开。关键发现是，WT 的 `subwords_can_be_empty`、early-cell、padding 截止和
boundary query 是正交开关，不能用一个近似选项捆绑实现。

### 2. 用 teacher-force 建立 alignment oracle

先让 FW greedy 固定 tokens，再分别用 patched CT2 与 OpenAI Whisper 对相同 tokens、相同窗口做
alignment。该阶段定位并修正了三类差异：缺失 boundary queries、median filter 语义不同，以及
`num_frames` 把 DTW 窗口与真实音频截止混为一谈。

`hello.flac` 达到约 20ms 边界差；`harvard.flac` 前四段共 28 词分词一致，三段所有边界在 20ms 内，
另一段只有末词 end 相差 60ms。剩余少量 path 分叉被判定为 PyTorch/CT2 attention 数值路径差异，
而不是状态机语义差异。

### 3. 冻结与框架无关的 refine 状态机

把 decoder-limit、连续 timestamp/SOT flush、unfinished fallback、early-EOT、倒退 end 修复、query
选取和 confidence 切片写成纯函数并加单测。这样 CT2 只导出 trace，不在 C++ 内复制一套容易漂移的
上层策略。

### 4. 从 2-pass 过渡到 1-pass greedy

CT2 generator 开始在文本生成时收集 timing heads、chosen-token logprob 和终止事件。先验证 raw
attention 与 teacher-force path，再把 DTW/后处理移入 CT2，只向 Python 返回 compact path，避免
每窗传输完整 attention 和全词表 logits。

production backend 明确探测 patched API；标准 CT2 wheel 会立即失败，不会静默退回另一套算法。
正常 greedy、early-EOT 和 forced max-length unfinished 均验证未调用 `find_alignment`。

### 5. 实现真正的 1-pass beam

首版复用完整 beam attention history，Harvard beam=5 从约 670ms 增至 1306ms，证明 O(beam·T²)
搬运不可用。最终改为 persistent parent-node lineage：每步只保存当前 attention row 和 parent，winner
完成后反向重建。winner logprob 与 score 对账误差仅 `7.6e-7`；完整 backend 相对 native beam 的
中位增量约 `+3.15%`。

独立 2-pass wall-time 落在短窗调度噪声内，不能仅凭总时间排序；但 1-pass 结构上少一次 decoder
forward，且人工审核确认 1-pass 时间轴更准，因此选择 1-pass。

### 6. 审计 WT 可选行为

- `remove_empty_words`：接入后结合源码重新评估，最终删除，换成非破坏性
  `zero_duration_chunk_tail` event；
- `detect_disfluencies`：确认所谓“尾词 end 收缩”并非 WT 多峰算法直接改 end，而是前置 `[*]` 改变
  segment start 后触发 overlap clamp。修复为首词前 gap 不物化 `[*]`、仍可更新首词 start，且所有
  实词 end 不变；
- `boundary_uncertainty`：可以收集 entropy/次峰比，但没有人工边界 gold，默认关闭；
- 可见特殊 token hack：不采用。结构化 events 比把内部判断伪装成字幕 token 更可逆、更易审计。

### 7. 小验证集与路由研究

建立 13-group corpus：3 个普通 control、10 个历史异常倾向 group。当前版本中只有 5 个历史异常组仍
出现 hard word-level abnormality；它们都能定位到单 interval。研究用 greedy isolation 全部清除结构
异常，重解音频占原 group 的 `3.59%`～`6.08%`，中位 `4.60%`。可靠人工参考样本
`stack-severe-1555` 的相似度从 `0.506` 升到 `1.000`。

5 个 hard group 的 beam=5 isolation 中，4/5 与 greedy 文本等价，另一例多出尾部 `pre`；本小样本
没有 beam 独有胜例。单位音频计算中位数显示 beam=5 比 greedy 慢 `34.2%`，所以局部异常的第一重试
仍建议 greedy，beam 留给 coverage rescue 或有独立文本不确定证据的第二候选。

2026-08 用 310 个生产窗口复核了这条结论并把它推得更硬：**beam=5 不抑制幻觉**——窗口级异常率
45→48（更差），已知幻觉短语 11→22，只有逐词复读 −17%。因此 beam 的价值是「换一条搜索路径再试」，
不是「beam 更少幻觉」，不应为质量切主路径。数据见
[`wt-refine-port.md`](wt-refine-port.md) 的「beam 不抑制幻觉与复读」。

## 当前实际行为

`fw-refine` 是唯一 ASR backend（`whisper-timestamped` 已于 2026-08-02 移除）：

```powershell
vad-asr <vocal-audio> ...
```

默认 checkpoint 配置：

| 选项 | 默认值 | 行为 |
| --- | --- | --- |
| `detect_disfluencies` | `False` | 不改词 start，不产生 `[*]`；显式开启后透传候选 |
| `collect_refine_signals` | `True` | 收集低成本 path/repetition/unfinished/zero-tail events |
| `collect_attention_signals` | `False` | 不为未标定 boundary metrics 返回额外 weights |

已默认透传的信号包括 `alignment_stack`、`long_token_span`、`decoder_repetition`、`unfinished` 和
`zero_duration_chunk_tail`。它们当前只作观测，不自动删除文本、不自动重解，也不被 FineSub 解析。

生产主契约是单一 `temperature=0`、单 winner、word timestamps。非单温度、multiple hypotheses、
without-timestamps、文本 span 与 compact trace 无法核对等路径会固定现有 tokens，退回 FW
teacher-force alignment；那一步也失败才丢弃该 group。

## 研究脚本与产物 pointer

所有脚本都是开发工具，不是生产入口。详细参数见两个目录内的 README。

| 入口 | 用途 | 何时复用 |
| --- | --- | --- |
| [`tools/wt_refine_port/oracle.py`](../tools/wt_refine_port/oracle.py) | 生成 WT 两种 DTW step pattern 的固定路径 | 改 DTW、early-cell 或 empty-subword policy 时 |
| [`teacher_force_probe.py`](../tools/wt_refine_port/teacher_force_probe.py) | 单 segment，固定 FW tokens，对比 CT2 与 OpenAI Whisper alignment | 排查 query、median、padding、confidence 差异 |
| [`full_window_probe.py`](../tools/wt_refine_port/full_window_probe.py) | 保留原始 timestamp tokens，在同一 encoder window 逐段 teacher-force 对齐 | 排查多 segment 状态传递和 decoded span |
| [`state_machine.py`](../tools/wt_refine_port/state_machine.py) | 无 Torch/CT2 依赖的 WT flush/repair 纯函数 oracle | 改 early-EOT、unfinished、confidence 切片时 |
| [`one_pass_probe.py`](../tools/wt_refine_port/one_pass_probe.py) | 对比 1-pass trace/path 与同 encoder output 的 2-pass align | 改 CT2 trace 或 beam lineage 时；其中 full-vocab/tail trace 是旧诊断接口，不是生产路径 |
| [`tools/wt_refine_validation/run.py`](../tools/wt_refine_validation/run.py) | 重跑 13-group、收集信号、执行研究用 isolation、汇总覆盖和单位计算成本 | 改信号、路由或重试策略时 |
| [`artifact_survey.py`](../tools/wt_refine_validation/artifact_survey.py) | 离线只读：已完成 run 的 `alignment_events` 与词级判定/幻觉短语/stable 处置的段级对照 | 评估信号在生产产物上的表现时；无 GPU 依赖 |
| [`window_sweep.py`](../tools/wt_refine_validation/window_sweep.py) + [`window_score.py`](../tools/wt_refine_validation/window_score.py) | 生产窗口无救援解码 dump + 判定器覆盖率打分（2026-08-04 405 窗口 + 入库裁决标注） | 改判定规则/信号阈值时重算覆盖率 |
| [`manifest.json`](../tools/wt_refine_validation/manifest.json) | 冻结 3 normal + 10 historical-hard group selector | 扩样时保持旧 case 不漂移 |
| [`test_fw_refine.py`](../test/test_fw_refine.py) | production adapter/refine 单测 | 任何 backend 行为变更 |
| [`test_wt_refine_validation.py`](../test/test_wt_refine_validation.py) | runner policy/schema 单测 | 改验证路由或结果 schema |

常用轻量回归：

```powershell
python -m pytest tools/wt_refine_port test/test_fw_refine.py test/test_wt_refine_validation.py -n 0
```

完整验证 runner 命令见
[`tools/wt_refine_validation/README.md`](../tools/wt_refine_validation/README.md)。它依赖 patched CT2、GPU、
本地模型和未纳入 git 的大音频，属于 heavy-resource 测试；不要在普通文档或单测改动中顺手执行。

本机可复核产物：

- `out/wt-refine/`：hello/Harvard teacher-force、full-window 和 one-pass JSON；
- `out/wt-refine-validation/full-v3.json`：当前 13-group path/full 信号主结果；
- `out/wt-refine-validation/beam5-hard.json`：5 个 hard group 的 beam=5 isolation；
- `out/wt-refine-review/harvard-beam5-1pass.srt` 与 `harvard-beam5-2pass.srt`：人工时轴对照，源为
  `assets/harvard.flac`；审核结论是 1-pass 准、2-pass 不准。

这些 `out/` 文件不入版本控制；长期可复用的是脚本、manifest、测试和两份 tracked 研究文档。

## 待办项

### P0：把默认 backend 切到 fw-refine 之前

已完成（2026-08-02 随合并落地）：

- ~~建立正式 CT2 fork~~ —— **改为钉版 + patch series + 预编译**，不维护上游分叉。
- ~~把 backend 穿透主 pipeline~~ —— 2026-08-02 起 `fw-refine` 是唯一 backend，开关已移除。
- ~~修复 resume/reuse 身份~~ —— checkpoint fingerprint 已含 `asr_backend`，且无默认值。

仍未完成：

1. **发布可复现 wheel（一个就够）。** 宽架构 + `CUDA_DYNAMIC_LOADING=ON` + Ruy 的单一构建
   即可覆盖 GPU 与无卡机器——与官方 PyPI wheel 同构，导入表零 CUDA 依赖。完整构建标志、
   踩过的坑与证据链见
   [`ct2-patches/README.md`](../tools/wt_refine_port/ct2-patches/README.md)。当前本机 build
   目录不能作为交付方式；wheel 到位后把 `pyproject.toml` 的 `ctranslate2` pin 换成指向 release 资产的 direct reference。
   打包时还需决定 `cublas64_12.dll` 的来源（随包 / `nvidia-cublas-cu12` / 要求用户装 toolkit）。
2. **补稳定 capability/version API。** 当前 adapter 通过 `Whisper.generate.__doc__` 检查
   `return_refine_paths/weights`，过于脆弱；改成显式 extension API/version，并在 metadata 中记录。
3. ~~**验证 CPU 路径**~~ —— 2026-08-02 已解决。旧构建把 MKL/DNNL/OpenBLAS/Ruy 全关了，
   CPU 上直接 `RuntimeError: No SGEMM backend on CPU`；带 `WITH_RUY=ON` 重建后**可用且正确**：
   同一段音频 CPU 与 GPU 输出逐字相同。代价是 8 秒音频解码 41.3s（GPU 1.4s，约 30×）——
   注意 Whisper encoder 无论音频多短都要过完整 30 秒窗口，所以短片段的固定成本占比极高，
   真实 30 秒分组上的倍率会好得多。该构建的 CPU 还支持 `int8`/`int8_float32`，是需要时的提速旋钮。
   **仍未验证**：带 CUDA 编译的二进制能否在无 NVIDIA 驱动的机器上加载（本机有卡，测不出来）；
   `CUDA_DYNAMIC_LOADING=ON` 正是为此打开的。
4. ~~**验证宽架构构建**~~ —— 2026-08-02 已构建并验证：二进制含 sm_70/75/80/86/89/90 原生 SASS
   + sm_90 PTX。生产链上与旧构建输出**逐字相同**（165 段 / 1175 词 / 0 异常 / 救援活动逐项相同）。
   **仍未验证**：在一张非 sm_86 的真实显卡上运行——本机是 sm_120，两个构建都走 PTX JIT。
5. ~~**建立迁移验收**~~ —— 已完成（2026-08-02，5 个素材 / 50.6 分钟）。结论见
   [`wt-refine-port.md`](wt-refine-port.md) 的「迁移验收」一节：**fw-refine 全面通过**，
   耗时 3.19×，词数与覆盖秒差 ≤1%，救援活动在每个素材上都更少。人工对听尚未做，
   相似度最低的 BV1UBjq6fEgb（85.9%）产物留在 `out/acceptance/` 供审阅。
6. **评估下游漂移。** 分句变化会传导到 LLM 纠错窗口划分与知识库条目。已决定不为旧 wt 产物
   做特殊保全——差异已证明不大，真需要可从移除前的 commit 重新生成。
7. **清理 metadata 谎言。** `asr_transcribe_seed` 仍写进 aligned metadata，但 fw-refine 不读
   `seed`（`fp16`、`refine_whisper_precision` 同样忽略）。

不建议 fork faster-whisper：Python adapter 留在 FineSub 更容易迭代，也避免同时维护两份上游分叉。
只有多个项目开始复用 adapter，或 FW 内部接口频繁破坏本实现时，再拆成独立包/fork。

### P1：质量与路由

2026-08-04 做了两轮信号-异常对照（工具与全部数字见
[`wt-refine-validation.md`](wt-refine-validation.md) 的「生产产物离线复核」与「生产窗口
覆盖率对比」）：先离线复核 5 个验收素材的最终产物，再用 405 个生产窗口的无救援 greedy
解码测各判定器覆盖率。核心结论：现有词规则联合覆盖 46/49 结构异常（94%）、加 coverage_low
达 83/86（96.5%），**信号联合只有 84%、不能替代**（拉伸型坍缩的 BPE 合并让 token motif
检测失效）；调阈值信号在现有之上净增 +2——`lang_switch_lowconf`（CJK 素材 Latin 段 +
低 conf，4/4、0 FP）补英文/BGM 幻觉盲区，`alignment_stack∧zero_tail`（12 TP/1 FP）补
已知短语尾幽灵；`decode_limit_signature` 27 命中 0 误报但与坍缩全重合，价值在救援路由
（跳过注定失败的重解）；原始事件不调阈值直接用会有 11 FP。据此调整以下条目的优先级。
**两个盲区已于同日纳入生产**（语言切换幻觉 → stabilize 丢弃；幽灵重复段 → `vad-asr`
清理步骤；`decode_limit_signature` 暂不接线，理由见 validation 文档「已纳入生产」）：

1. 扩大异常验证集，特别补 `zero_duration_chunk_tail`、多 hard interval、early-EOT、unfinished、多人声
   重叠和首词 disfluency；现有 zero-tail 只有一个异常 case，不能估精确率。离线复核后补充：
   英文/BGM 幻觉家族应作为独立类目进验证集——它是目前唯一有证据的「信号可新增召回」的类别。
2. ~~`[*]` 词首修正（含句首候选声学门控与 VAD 锚点 clamp）~~——**已全部实施**
   （2026-08-05，`recognition/word_starts.py`）：四规则（短块融合 / 词级
   `disfluency_span`+`disfluency_action` 标注 / 能量门控删除（无位置门，仅 3s cap）/
   其余融合）+ 段首候选门控化 + interval(+0.1s 五守卫)/pause_hint 两级锚点 clamp；
   `detect_disfluencies` 默认开启（解码零成本）并进 checkpoint key，`[*]` 豁免于
   词级异常判定。规则与常量见 docs/asr-align.md「词首修正」；gold 标定
   （quiet_frac 分离、0/25 词头误删）、位置门放宽依据与端到端数字（词首误差
   41→18ms）见 docs/wt-refine-validation.md「VAD 改版后的复测」。
   注意 pause_hints 只是当初设想 merged-gap 的子集（重偏早家族覆盖 7/27），
   作次级锚点；主修复力量是能量门控删除。

   仍未做的两个方向：
   - **强制语言重解**（翻译型幻觉的正解）：语切嫌疑段用锁定语言重解，替换仅当
     重解干净且覆盖不降。重试类，比删除安全（删除已被大范围复核否决）。
   - **低幻觉第二模型校验**（2026-08-05 用户方向，**已落地为生产证据层**：
     `speech/verification/qwen_referee.py` 在 vad-asr 尾部产证据（`--qwen-verify`，
     默认 auto），stabilize 消费（套话删除授权 + 噪声腿 veto），gap 补认暂只记
     metadata 证据；BV1cq 实测嫌疑 5/5 与离线审计一致、峰值显存 1.70GB、
     全程 11.8s 含加载。安装与依赖取舍见 docs/vad-asr.md）。标定过程记录：
     对嫌疑段用 Qwen3 ASR 0.6B 重认对照，把"先验赔率赌注"换成"双模型一致性"。
     实测（31 clip，`tmp/qwen-smoke*`，venv `tmp/qwen-venv`，0.6B 与 1.7B 无差别）：
     (a) **套话嫌疑**：幽灵 5/5 短语不在、真收尾致谢精确回认、BGM 位点 3/3 不复现
     Whisper 的套话幻觉；H6 两处正常语速 `おわり` Qwen 判不存在 + 用户重听确认
     **均为幻觉**（其"人工保留"实为 LLM 产物保留——H6 无人工字幕，出处已订正）——
     即 Qwen 短语包含判据在全部已裁决位点 11/11 正确，正常语速套话的删除授权可行；
     (b) **翻译型幻觉**：5/5 强制 ja 认出底下的真日语，语义与人工中文字幕对齐——
     强制语言重解的实现载体直接成立；真英文 5/5 auto 输出匹配英文（判别子 =
     Qwen 是否输出日语；语言强制是软约束，真英文下仍出英文）；
     (c) **低覆盖 gap 补认**：yui 3 段认出 Whisper 整段 EOT 漏掉的完整台词、
     2 段与人工字幕一致地回空、零编造——37/86 异常里最大家族的召回来源；
     (d) **坍缩窗重认** 2/2 给出真实内容替换候选。
     第三轮扩样（36 clip，累计 67）补充：语切判别子定稿为 **auto 遍输出语言**
     （真英文 17/17 auto=EN、翻译型 5/5 auto=ja；强制 ja 对真英文歌词出音译垃圾，
     仅用于提取替换文本）；gap 补认累计 6 真话 / 7 正确回空 / 2-3 漏听（喊叫段）、
     零编造；存疑复读终审 2/2（喜んでる×4 与 わー!×10 均为真实，幽灵规则三条件
     收紧再次验证）；**丢弃审计发现 2 处真实喊叫误删**（kaguya `あ!` c=0.27 e=+9.4，
     Qwen 听到 `啊！`）。**已知弱点：喊叫/尖叫段 Qwen 会漏听或跨语言渲染**
     （啊/哦/哈哈）——Qwen 回空不能给喊叫形态授权删除；多音节套话不受影响。
     **放置建议（待确认）**：证据生产单点放 vad-asr 尾部（gap 段需 energy track，
     它只活在 stage 进程内；共享一次模型加载；证据字段落 aligned JSON 可离线重放），
     删除决策留在 stabilize（消费证据，含删除前 veto）；不进 rescue ladder 内部
     （省得少、transcribe.py 红线区）。
     集成注意：Qwen 文本无词时间戳，gap/坍缩需 VAD span 定 cue 边界或交 LLM 层作证据；
     校验器只有三种权力：确认既有嫌疑 / 否决我方删除 / 附证据，绝不静默改文本。
     Qwen3-ForcedAligner 精度不高且同样坍缩（用户实测），只配粗交叉验证，
     不能当时间戳裁判，也不产生断句点（Whisper 原生分段无替代）。
3. 实现 ASR controller 的保守路由：单一 hard interval 可先做 greedy isolation，保留原候选，只有 retry
   clean、coverage 不回归且文本证据不变差时采纳。`decoder_repetition` 或 zero-tail 单独出现继续 deferred。
4. FineSub 开始解析 `alignment_events[]`，实现需要邻段上下文的 regroup/redecode；与局部 isolation 做
   同一验证集的质量、重解音频秒数和 wall-time 对照。
5. 为 `boundary_uncertainty` 建人工边界 gold 后再定 entropy/peak-ratio 阈值。未标定前不要默认开启。
6. 探索 `beam_consensus`、attention stall 与 decoder motif 联合信号；强制 timestamp/EOT 只能作为有 raw
   tokens、forced-event 和清 prompt 记录的可审计候选，不能静默删文本。

当前建议仍是：局部且确定、且在 ASR 处理能提高质量或以相同质量减少重解音频，就优先 ASR；局部但
不确定则保留原样传 FineSub；涉及分组/邻段的判断交给 FineSub regroup 后重解。

### P2：multi-audio batch（吞吐优化，不阻塞迁移）

**设计、实测与 profile 档位见 [`wt-refine-port.md`](wt-refine-port.md) 的 multi-audio batch 一节**；
本条只记状态与剩余工作。

优先级说明：`whisper-timestamped` → `fw-refine` 迁移本身已值 6.3×（3.54 → 0.56 min/小时素材），
batch 再叠 1.8× 到 11.4×。**P0 的价值远大于本项**，batch 不应插队。

已完成：

1. ~~CT2 generate 接受逐样本 `real_audio_frames`~~ —— 已实现（`WhisperOptions::real_audio_frames`
   改为 `std::vector<size_t>`，空=全帧、单值广播、否则长度须等于 batch）。decoder 侧原本就是
   batch-aware，无需改动。
2. ~~batch=2 greedy / beam=5 parity~~ —— 已用更强形式验证：B=4 与 B=16、各 item 不同
   `real_audio_frames`、提前 EOT 触发 compaction、beam winner lineage、错长度拒绝与标量广播。
   结论：tokens/logprobs/refine paths 在 fp16 舍入之外一致；**批与非批不可逐位复现**，详见 port 文档
   的「确定性契约」。

剩余（按序）：

3. Python adapter 改为逐样本状态：`encode()` 的 padding 边界、`generate_with_fallback` 的
   `[prompt]`/`[0]`，以及 `_pending_refine_trace` 与两个 FIFO deque——**deque 串号是这里唯一的
   静默失败模式**，应按显式 id 存取并断言，而不是依赖顺序。
4. ~~绕开 FW seek 循环的批量 driver~~ —— `fw_refine_backend.transcribe_batch` 已实现
   （split-encode：逐窗 encode、只批 decoder；否则 greedy 会产生 2–7/32 的文本分叉），
   并有测试，但**尚无生产调用方**，接线前必须先处理下面三条（2026-08-03 代码审查）：
   - **`transcribe_batch` 不设置自己的信号开关。** 它读 `model._detect_disfluencies` /
     `_collect_attention_signals` 决定 `return_refine_weights`，却从不从自己的 `options`
     里设置它们——只有单窗路径 `transcribe_wt` 会设。所以传 `detect_disfluencies=True`
     进批路径会被静默忽略，拿回没有 weights 的对齐。这是接线时第一个会踩的。
   - **批与单窗的解码参数不一致。** 批路径硬编码 `max_initial_timestamp_index`（1.0s）、
     `length_penalty=1.0`、`suppress_blank=True`，单窗路径走 `options.*`。docstring 只声明了
     窗口长度、单一显式语言、无 prompt 三条约束，这几个没写进去。同一段音频走两条路可能
     解出不同结果——而这正是批 driver 最需要保证的等价性。
   - **`FwRefineModelPool.close()` 不 `notify_all`**，且清零 `_loaded` 的同时，仍租出去的
     model 之后会被 `_release` 放回 `_idle`。等在 `_acquire` 里的线程会永久等待。当前只在
     收尾调用所以没暴露，批 driver 会显著增加并发租借。
5. `align_segments` 的投机批规划与 isolation 回滚；批大小由 GPU profile **静态推导**
   （CT2 的 CUDA OOM 是进程级硬中止，不能试探自适应）。
6. checkpoint fingerprint 补批配置（backend 已在合并时补上）。

~~前置未知：真实 isolation 率未测~~ —— 已测：310 个生产窗口里 45 个含生产异常，**p ≈ 14.5%**，
远高于建模投机浪费时假设的 1–5%。但**决定放弃动态调整分组**后投机约束整体消失，
[`wt-refine-port.md`](wt-refine-port.md) 的档位表以静态分组为前提，因此大 B 档位仍然成立。

同一音频中依赖 `condition_on_previous_text` 的连续 group（>30s 分组的第 2 窗起）不是首批 batch 化
对象——其 prompt 非空，退顺序路径即可，无需 pad。batch 预期主要提高吞吐，不保证降低单 group 延迟。

### P3：低优先级与暂不移植

- multi-temperature / multiple hypotheses 的 1-pass trace；当前 teacher-force fallback 足够；
- punctuation timing、`remove_punctuation_from_words`；不解决主要时轴/召回问题；
- WT 自带 VAD、plot、`trust_whisper_timestamps=False`；FineSub 已拥有对应上下文和所有权；
- 原样照搬 `tokens > frames` 的正文截断；应先作为 dense/stall 强信号隔离重解；
- 仅靠 word alignment 恢复多人说话的“合理字幕重叠”。单条 Whisper token stream 的 alignment 只能给
  已生成词定位，不能凭空恢复未生成的第二说话人；该问题需要声源/说话人分支或重叠区独立解码。

## 接手建议顺序

1. 读本文，再读 [`wt-refine-port.md`](wt-refine-port.md) 的 checkpoint、M2～M4 和缺口取舍；
2. 运行轻量测试（`python -m pytest -q`）；
3. 按 [`ct2-patches/README.md`](../tools/wt_refine_port/ct2-patches/README.md) 从 `v4.8.1`
   重建 patched CT2，不要把本机 build 目录当源码；
4. 先完成 wheel 与 capability API（P0 剩余项），再谈切默认；
5. 用 Harvard 1-pass/2-pass SRT 和 13-group runner，加上多素材 WT/fw-refine 对照建迁移基线；
6. 主线切换稳定后，再做 controller/FineSub 对 `alignment_events` 的消费。

这样可以把“完整 WT refine 移植”“质量策略研究”“FineSub 分组决策”和“GPU 吞吐优化”四个问题分开
验收，避免一次改动同时改变文本搜索、词时间、重试路由和调度方式。
