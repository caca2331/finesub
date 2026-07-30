# LLM 纠错与翻译：架构意图与设计决策

本文取代原《LLM 纠错与翻译架构 RFC》（`llm_correction_translation.md`），只保留**意图、取舍与决策记录**——即"为什么长这样"。现行行为以 [`llm_harness_behavior.md`](llm_harness_behavior.md) 为准；知识库见 [`knowledge.md`](knowledge.md)；prompt 组装见 [`llm_prompts.md`](llm_prompts.md)。已实现的原始设计稿与实验日志在本地 `docs/archive/`（gitignore，不入库）。

## 背景与目标

最初的单次 prompt 试图一步完成：理解整段直播、搜索背景、修正 ASR 误听、译成轻小说风格简中、排版并输出 SRT。真实需求全覆盖，但一步到位有结构性问题：

- 职责过多：资料收集、听音纠错、翻译、排版、格式化混在一起，错误难定位。
- 预算不稳定：长音频下输入、检索资料、思考、输出都可能同时逼近上限。
- 输出脆弱：直接产出完整 SRT 易出现编号缺失、时间轴错乱、中途截断；截断后"继续写"极易重复/漏段。
- 无法学习：任务中发现的术语、误听模式、风格问题没有结构化沉淀。

因此拆成可测试、可预算、可回滚的多阶段架构：

```text
stable.json 渲染的 raw CSV + 原始音频
  -> 窗口规划（边界同时供背景调查分窗标记）
  -> 两轮背景调查（+可选多轮搜索 loop；本地搜索代理执行检索）
  -> 每窗查询轮（轻量多模态提 query）+ 本地搜索
  -> 听音纠错翻译（注入检索结果与建议链）
  -> 本地校验 + SRT 渲染
  -> 可选：任务反馈采集 -> 统一知识更新
```

核心分工原则：**模型只负责"提出 query / 提出 proposal"，harness 负责"执行检索/写入并注入结果"**——因为 gemini-3.x 免费层不开放 `google_search` grounding（实测立即 429），也因为本地执行可预算、可重试、可审计。

其他贯穿性取舍：

- **CSV 而非 SRT 作为模型输入/输出载体**：`global_id|local_start|duration|gap|text` 一行一条，比 SRT 省去序号+时间戳的 token 开销，且时间轴由本地按源序号回填，模型永远不产出时间戳（insert 行除外）。
- **音频用原始音频而非 `*-vocal.flac`**：保留游戏声、角色语音与分离算法可能损伤的语音细节；时间轴仍来自 VAD-ASR。
- **截断只拆窗，不"续写"**：输出超限把窗口对半拆成重叠半窗重试；多轮补写在长输出下不可靠。
- **格式错误同窗重试、供应商错误不缩窗**：503/timeout/限流是 provider 层问题；缩窗只响应输出上限。
- **prompt 缓存友好**：system 只放稳定要求；user 先背景资料后动态 payload，易忘要求放尾部 recap。

## 五角色抽象（为什么这样分）

模型能力拆为五类接口而非把模型名写散在业务代码里（现行角色/模型映射见 harness 文档；事实配置在 `src/llm/model_catalog.psv` + `config.py` 的 `endpoint_chain`）：

- `audio_multimodal` —— 纠错窗 / fast 纠错步（纠错 r2）：3.6 优先链；text-high 除外走 `internet_capable`。
- `general_capable` —— 内容理解与整理（调查 r1/r2、fast r1、统一知识更新等）：3.5 → 3.6 → 3.5-lite。
- `lightweight_multimodal` —— 纠错查询轮（纠错 r1）；与 `lightweight` 共用 3.5-lite 优先链。
- `lightweight` —— search-loop 查询 judge（纯文本）；同 3.5-lite 链。
- `internet_capable` —— text-high 专用：唯一允许模型自带搜索工具的角色，单独隔离是因为免费层不可用、必须由用户显式配置。

## 输出预算公式的推导（routes/levels 设计）

六档 preset 的预期输出估算 `k × c × csv_tokens` 中，系数 c 是**加法结构落成的常量**（实现只暴露 6 档，不暴露自由组合）：

```text
text:  c = 2.0(基础) + 1.5(上调思考, med/high) + 1.0(内置联网, high)
mm:    c = 4.5(基础，含外部注入与上调思考纯文本的等价开销) + 0.5(音频) + 1.0(视频)
```

一致性检查：mm-low 的 4.5 = text 的 2.0+1.5+1.0，与「mm-low ≈ 外部注入检索版的 text-high」的定位吻合。

窗口约束 `k×c×csv_tokens ≤ 0.9×output_limit − 5000`（=53,982 @65,536）反解出 k=1 时每窗 CSV token 上限：

| preset | c | 常规窗上限 | 快速模式全量上限 |
| --- | ---: | ---: | ---: |
| text-low | 2.0 | 26,991 | 21,214 |
| text-med | 3.5 | 15,423 | 12,122 |
| text-high / mm-low | 4.5 | 11,996 | 9,428 |
| mm-med | 5.0 | 10,796 | 8,485 |
| mm-high | 6.0 | 8,997 | 7,071 |

历史注记：该公式取代了更早的 `csv×5 + 10k` 启发式；mm-med 的有效窗口约放大 80%（6000→10,796 csv tokens），担心质量回退时可把 `--output-scale` 调到 1.2–1.3 获得接近旧行为的窗口大小。

## 统一知识更新的决策记录

（知识更新 redesign 的过程草稿在本地 `docs/archive/`；现行为见 `knowledge.md`。保留已确认决策供后续演进对照：）

| # | 决策 | 理由 |
| --- | --- | --- |
| A.1 | 每个 100k 大块均带**完整** research hints | research 视野全局，其线索对任一块都可能相关 |
| A' | final_csv 以 `position` 的 id 集合为键（n:1），insert 行无 global_id 按时间落窗 | final 行可合并多行 raw，「同 id 同行」不成立 |
| B/C | 无精修省略 `refined_csv`；两种证据模式各一段 prompt | 混一段 prompt 会让无精修模式看到大量不适用规则 |
| E | final_csv = annotated.csv 按**序号 1:1** overlay 后处理 final.srt 的时间与 translation；corrected 不 overlay | annotated 与 translated.srt 由同一 `rendered_segments` 渲染、postprocess 不增删条目，1:1 已核实成立；corrected.srt 不走 postprocess，保留模型原纠错 |
| F/G | 无精修模式禁写 mistake 库，且**禁用点在 harness**（prompt 不含该块、解析忽略、不调 apply） | 不依赖模型自律输出空块 |
| H | 三态 `--knowledge none|collect|update`（update 隐含 collect） | 采集与执行解耦但避免双 flag 组合出无意义状态 |
| I | feedback v3：任务反馈 confidence 仍为 1-9；字幕行 conf 独立改为 high/median/low；v1 字段直接删除不迁移 | 两者表达不同对象，避免混淆知识线索可信度与字幕复核优先级 |
| J | refined_csv 用 `start|end|text`（end 而非 duration）+ harness 按 start 重排；**不做**对齐健康检查，噪音由 prompt 告知 | 精修文件可能 index 错乱、含注释性重叠字幕 |
| K | 分块预算：CSV 三块 ≤100k、词条块 ≤40k、context ≤10k、feedback ≤10k；组装后 >194k 按窗口再切（错误/范例台账全文不再注入） | 各块独立上限 + 总量硬校验，杜绝静默截断 |
| L | 多块顺序 apply + `knowledge-update-chunks.jsonl` ledger 幂等 | 块 2 失败重跑不能把块 1 的 append_history 追加两次 |
| M | 窗口材料按 **stitch 后实际归属**分组（id 属最后包含它的窗） | 物理重叠窗直接分组会重复计料、注入互相矛盾的行 |
| N | CLI 位置参数 = 标准 final SRT，其余路径按 stem 派生 | 四条路径全是同 stem 派生，逐个传参易错 |
| O | 知识更新不注入已有 common-mistake / good-example 台账；跨任务查重留给独立维护模块 | post-task 只产出提案；台账对照与清理另路维护 |

## capability tier 与确定性预合并的决策记录

（设计过程草稿在本地 `docs/archive/`，不入库。**预合并（premerge / stabilize profile 3）已于
2026-07-29 随 `segment_split` 全局 DP 迁移删除**——分句器自己决定 ASR 段接缝去留，词中切断的
碎片不再产生，实测 9 clip 上预合并 0 次命中。M.1–M.9 保留为决策史：它们记录的是**为什么这条
路走不通/走通了**，重开同类设计前先读。现行为见 `llm_harness_behavior.md`、`llm_prompts.md`、
`asr-stabilize.md` 与 `segment_split.md`。）

| # | 决策 | 理由 |
| --- | --- | --- |
| T.1 | 纠错 prompt 按 capability tier 分层，tier 在 `client.complete` 的 endpoint 循环内**随实际应答模型**解析（消息传工厂、按 tier 惰性记忆化组装） | 回退链 `3.5→lite` 上同一窗口可能被任一模型应答；prompt 与模型的一致性必须由同一 `endpoint` 结构性保证，而非链序纪律 |
| T.2 | tier **不进**会话签名，失效由 `PROMPT_VERSION` 承担；catalog 查不到默认 CAPABLE | tier 是限流态不是任务身份，进签名会让配额波动频繁废 resume |
| T.3 | basic 档 = 保守 1:1（仅词中接回），tier 无关产出纪律抽为 common 片段 | 弱模型判断型合并的错并代价远大于少并；共用纪律防两版漂移 |
| M.1 | 「需合并」判据：仅当**不合并会严重影响阅读体验或语义准确**（典型：词中切断）；可并可不并不计，人工精修粒度不作 ground truth | 该判据同时约束 prompt（v46 起写入合并片段）与预合并的评估口径 |
| M.2 | 预合并只做强证据合并（E1/E2 词形签名、~~E3 sudachi 词典证据~~ **已移除** v2.4-no-e3 2026-07，当前仅 E1+E2 + 否决词表 + 7s/36字/3源护栏），弱交界一律不并 | v1「无标点无空格+小 gap」被实测证伪（42% 精确率）：日语 ASR 句界与词中切断同形；**错并在源序号层不可逆、漏并可由模型恢复**，只许优化精确率 |
| M.3 | 真词中切断允许宽 gap（表面签名 ≤1.0s、词典证据 ≤1.5s） | 实测切断 gap 常在 0.4~1.2s，v1 的 0.15s 方向反了；gap 越大要求证据越硬（呼应 split 的 g_score） |
| M.4 | ~~预合并落位 stabilize profile 3；split 只打 `splitted_before` 段级 tag，预合并结构性拒绝这些交界~~ **已随 premerge 删除**（tag 现反转为词级 `whisper_segment_start`，见 segment_split.md） | 当年成立的理由：premerge 纯重组适合 stabilize；split 含 gap-word 时间调整属对齐职权不拆；tag 把互斥从推理保证升级为结构保证。全局 DP 之后互斥无对象——每个边界都是 DP 决策 |
| M.5 | ~~profile 0 顺序 `1 → 3 → 2 → 丢弃`~~ 现为 `1 → 2 → 丢弃` | 原理由仍然有效且**必须记住**：词中碎片天然低置信（`次はキッ|と` conf 0.089），先过滤会被误标幻觉丢弃、词永久残缺。现在该约束由「分句器不产生这种碎片」满足，而非事后修补 |
| M.6 | 合并交界以 **word 级** tag 留存（原 `premerge_before`） | merge 消灭段边界，位置只有 word 能承载。**同一论证在全局 DP 下复用**：piece 可吞掉整条 ASR 段接缝，所以分段起源也只能由词级 `whisper_segment_start` 承载 |
| M.7 | ~~规则/词表/阈值标注**过拟合风险**（单语料调参）与**日语特化**~~ 随 premerge 一并删除 | 该风险最终未被 held-out 验证消化，而是由删除模块消解；教训：单语料调出的表面签名规则，先问「上游能不能不产生这个问题」 |
| M.8 | ~~E3a 追加双约束：gap ≤0.3s **且** 交界一侧词素碎片性（OOV/黏着词类）；「词典内复合词即强证据」前提废弃~~ **已废弃**（E3 整体移除，v2.4-no-e3 2026-07） | held-out（kaguya60/yingtao）证伪：生僻词条 中胸 在普通句界假匹配（gap 0.545s、两侧自由名词）；全部真例 gap ≤0.275s 且碎片侧 OOV。已知损失：白騎|士 形（两半皆自由词） |
| M.9 | 「split 不会切进真词」前提**被证伪**（真|ん中、カ|ウントダウン），split-tag 结构性拒绝恰好挡住修复 | 根因取证（raw 词时持久化后定案）：wt 把右词组首词拉伸横跨停顿 + case3「接触较长侧」启发式必然锚左；修复走 M.10，词典否决/E4 不再需要 |
| M.10 | case3 锚定默认改**右**，无词典依赖、无例外表；胶连（pass 2）优先于默认；纯默认归右的片段首词打 `split_anchor_uncertain` 段 tag | 16 例研究：错误 100% 集中于「case3 锚左 + 落段尾贴停顿」（4 例错 3）、落右 12 例零错；属左收尾词由「左胶连+右分隔」的 pass 2 覆盖，无需词尾标点例外（词尾标点描述的是右交界不是归属）；残余风险=两侧全胶终助词拖尾，零实例、可审计,不加零样本词表 |

## wt 对齐坍缩：检测与救援梯的决策记录

（实验与接入过程草稿在本地 `docs/archive/`；逐例产物在 `out/collapse-exp/`、
对照评估在 `out/collapse-eval*/`；现行为在 `src/asr_playground/speech/recognition/transcribe.py` +
`asr_playground/text.py`。2026-07-19。）

| # | 决策 | 理由 |
| --- | --- | --- |
| C.1 | 坍缩签名用**词级形态**：≥3 个连续 ≤25ms 词（`COLLAPSE_STACK_*`，进 `detect_abnormal_asr_words`），不用段级 CPS 阈值 | wt efficient 路径把词挤进近零跨度时词时长帧量化在 ~20ms；确证真台词坍缩（8/8）全部含 ≥3 连零时长词，而真实快语速只有**孤立**零时长词——CPS≥12 单独用会误报快语速（实测 2 例全变体时间轴一致），变体间共识即非坍缩证据 |
| C.2 | 修复不新增机制，接现有 abnormal 梯：regroup ×2（scale 2/3、1/2）→ 组级 beam → 异常 interval 隔离剥离 | 实验（V0 逐词复现坍缩）证明 regroup∪beam 修复全部确证真台词坍缩且互补（单变体只 5-6/8）；refine 0.5 也有效但属全局参数不动（且更小 refine 可能损时间轴精度） |
| C.3 | regroup 第 3 轮（scale 2/5）移除 | 11 源 eval 实测 1/32 解决率，成本一整轮全子组重解码；失败组更早进 beam/隔离 |
| C.4 | 末级从「全组逐 interval 碎片化」改为**异常位置导向的剥离**（照 coverage rescue 末级建模）：异常前干净 run 一窗重解、异常 interval 单独成窗、剩余重解再检；上轮干净 subgroup 直接保留结果 | 健康邻居不再被碎片化/重复解码（短段 9/11 源下降、耗时反降）；各窗音频不相交防重复转写 |
| C.5 | 干净前窗重解码若返回异常，回退候选干净切片（裁越界溢出词）——但切片须先过 `_coverage_shortfall` **覆盖率闸门**，不足则降级为逐 interval 重解 | 两轮 held-out 各抓到一半：前窗单独解码可退化成复读循环（へ×134）吞真实台词→需要切片回退；但「interval-clean」也可能是 interval-**empty**（词按 whisper 段整段归属主导 interval，候选可把前窗语音全部吸附到异常 interval），空切片=丢内容→需要覆盖率闸门+逐 interval 兜底 |
| C.6 | 纯已知短语堆叠（`COMMON_HALLUCINATION_TEXT`，常量在 `src/asr_playground/text.py`，与 stabilization 共用）**早退**跳过整个救援梯；判定从严（混任何真实文本/其他异常不早退） | 假 ご視聴 下面没有可恢复语音，重试纯耗 GPU 且只把挤压转成拉伸；stabilize profile 1 按词跨度（≤5 词）整段清除，真说的短语是十几个词不受影响——形态判别不依赖能量，天然覆盖非低能量区 |
| C.7 | 幻觉/坍缩不用 confidence 区分 | 实测方向与直觉相反：真台词坍缩 conf 中位 0.85（最高 1.0），ご視聴 幻觉 0.77（0.46-0.90），重叠严重；仅 <0.5 / >0.95 两端有弱判别力。段能量才是强区分器（幻觉中位 −43dB vs 真台词 +9.7dB，profile 2 已在用） |
| C.8 | 顽固幻觉（音乐/静音区 ご視聴、笑声）不指望重试消除，维持 stabilize 链（短语清理+能量 tag+drop）兜底 | 实测重试只消一半；stable 级残留=0（基线与新版同），说明下游链已完备。真正缺口是**笑声拉伸长段**（くっふっふ×35s 类，能量不低、字数>2 全规则穿透）→ 挪词 pass / 笑声模式 tag 的输入（呼应 M.9/M.10 线） |
| C.9 | 阈值沿用单语料标定标注（同 M.7）；8 BV held-out 零调参验证通过（词级 stack 37→10） | held-out 只验证不调参；C.5 两次修正均由 held-out 对照跑抓出（collapse-eval2/3），最终生产产物 stable 级 ご視聴 残留 0、词级 stack 0 |

## 自适应 gap 静音的决策记录

（标定实验见 `out/split-explorer-8bv-20260718/adaptive-gap-analysis.md`（split_explorer
`asr_gap.py` 的 monkeypatch 版）；转正对照评估 `out/collapse-gap-report/report.md`，
基线 `out/collapse-eval3/`（同代码定 gap 版），覆盖前原品备份 `out/collapse-eval3/orig/`。
2026-07-19。）

| # | 决策 | 理由 |
| --- | --- | --- |
| G.1 | 组内 interval 间合成静音从固定 0.3s 改为 `min(0.1 + 0.2*原gap, 0.8)`（`GAP_SILENCE_*`；保留原声 ≤0.7s 不变；组尾静音仍用 gap_sec） | 紧邻边界给短静音减少解码器切碎、宽停顿给长静音强化分段线索；11 源对照：段数/短段普遍下降（BV1cqLR 212→174 段、<0.4s 短段 30→20；BV1kYLR 短段 8→0）、时长中位上升，碎片化明确改善 |
| G.2 | 文本质量按「区域混合、净方向中性」接受；困难区解码路径两向漂移（有恢复真台词处也有变糊处），坍缩指标真台词无回归（命中回升均为早退保留的低能量幻觉堆叠+已知快语速误报） | 逐区人工过目：BV1UBjq 180-210s 基线幻觉短语被 gap 版恢复为真实台词；BV1nxje 300-330s gap 版一段变糊——同量级、非系统性；stable 级最终残留全零 |
| G.3 | 两个更激进变体（v1「还原原 gap」`min(max(0,gap-0.7),0.8)`、v2「紧邻零静音」`gap≤0.2→0 否则 min(0.6*(gap-0.2),0.8)`）实测后**均拒绝**，公式定格现行 | 见下表：v1 合并偏向 >0.3s 真停顿（增量合并 57% 跨 0.3s vs 现行自身 43%），超限段 +40%、split tag +26%——过合并被 DP splitter 重切，正是 M.9/M.10 力图减少的低质切点；v2 与现行几乎持平且无一处胜出，还有局部回归（yingtao 命中 2→6）——说明现行 0.1~0.16s 弱静音已不阻碍应并之并，继续减静音只剩风险 |

四方案聚合对比（11 源 ≈3200-3350 段，逐例产物 `out/collapse-eval3/`（naive）、
生产 aligned（现行）、`out/collapse-gap-v1|v2/`；naive 列与其余三列间还差一个
C.5 覆盖率闸门修复，主要影响个别隔离窗、不影响量级）：

| 方案 | 段数 | <0.4s 短段 | wcc>36 | >7s | split tag | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| naive 固定 0.3s（原） | 3345 | 198 | 14 | 6 | 135 | 被替换：紧邻边界切碎最多 |
| **现行 `min(0.1+0.2g, 0.8)`** | 3226 | 178 | 20 | 5 | 126 | **采用**：碎片明显减少，超限仅小幅上升且 stable 后无残留 |
| v1 `min(max(0,g-0.7), 0.8)` | 3079 | 157 | 28 | 7 | 159 | 拒绝：过合并→超限+40%、split tag+26% |
| v2 `g≤0.2→0; min(0.6(g-0.2),0.8)` | 3144 | 173 | 20 | 5 | 118 | 拒绝：与现行持平、无胜出、局部回归 |

### 可维护的 Shared Context

把调查 context pack 从"一次产出、只读注入"升级为全 session 可见、模型可增删改行的共享上下文。**评估结论：暂不做**——与现有 advice ledger（逐窗累积、全会话可见、8k 上限）功能高度重叠，且行编辑协议给每个 session 增加新的解析失败/重试面，显著复杂化 resume 与 `-a`/`-b` 半窗重试的确定性重建。**触发条件**：8k advice 台账在长任务中证明不够用（后期关键上下文被挤出、模型频繁重申旧建议）时再实现。

协议草案要点（届时参考）：harness 注入带行号的 `<shared_context>`；模型输出 `<shared_context_update>`（`change/insert/add/remove <行号>`）；一个 update 块内行号冻结、执行完统一重排；`_commit_window` 时应用并把快照写入 window cache 以保证 resume 确定性等价。

### Prompt/harness 自我迭代

prompt 迭代由 `tools/session_replay` 的受控重放驱动（见 `docs/tools/prompt-iterate.md`）：冻结上游注入物、只换 prompt、用 benchmark 评分与逐行抽查看模型真实反应，据此改模板。知识更新（`llm.knowledge.update`）不再生成 `<harness_notes>`——该职责已完整移交给 session replay。人工审阅 replay 产物后手动改模板，绝不自动应用。长期方向可借鉴：

- [DSPy GEPA optimization](https://dspy.ai/getting-started/gepa-optimization/)：样例+metric 搜索 prompt 变体（离线 optimizer）。
- [OpenAI Working with evals](https://developers.openai.com/api/docs/guides/evals) / [evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)：先定义 eval 再迭代；LLM-as-judge 需人工校准防漂移。
- [TextGrad](https://arxiv.org/abs/2406.07496)：让模型对系统组件输出反馈，再转成修改建议。

## 口语颗粒（语气词/填充/结巴）的取舍指导

（prompt 侧落点：`fragment_goals_translation_v1.md` 第 4 条为压缩版；本节为完整判定
框架与依据。相关机制：singles/translated 分层、中央合并规则、stabilize 的
「高度疑似语气填充词」确定性 drop。2026-07-20。）

### 为什么需要一套次序

直播切片字幕的两个目标天然拉扯：**可读性**（每条字幕的阅读时间预算有限，垫词挤占
正句）与**人设保真**（VTuber 切片的核心价值是人，口癖、犹豫、破音是身份的一部分——
删掉口癖等于给主播消音）。任何「一律删」或「一律留」的规则都会在其中一端翻车，
所以取舍必须是**判定次序**而不是词表。

### 判定次序：先问是不是内容，再问怎么处理

**第一问：它是内容吗？——是则保留（允许整形），判据任一即算：**

| 判据 | 例 | 说明 |
| --- | --- | --- |
| 情绪载体 | 「うわぁ、ほんとだ、翼みたい!」的うわぁ | 惊叹/哽咽/兴奋的发声是情绪本体，不是包装 |
| 性格口癖 | 主播标志性口头禅 | 知识库为口癖立词条；口癖是身份，不因「无信息」删除 |
| 笑点/包袱 | 试名失败连环「ユミ、ユメ、や、ミ……あ、だめだ」 | 整段结巴就是内容本身，一个都不能少 |
| 对话功能 | 应答 うん→嗯、惊叹 えっ?→诶？、呼名 レミ! | 这些是台词；短≠可删 |
| 叙事悬停 | 「ママが新しい髪型に…」被打断 | 欲言又止的省略号承载信息；被插话打断的续句各自独立保留 |

**第二问：不是内容，是机械噪声吗？——是则压缩（保意图、去机械性）：**

- **口吃 repair**：保留一次重复——「ち、違うよ」→「不、不是的」。一次口吃有表演
  价值；机械多次（「や、や、や、やばい」）压成一次「呀……好险」。判据：重复次数
  超出表演需要的部分是发音机制产物，不是表达。
- **false start + 改口**：保留改口结果；false start 只有带情绪/喜剧价值才保留
  （试名段保留，普通说错重来则删）。
- **句中垫词**：なんか/こう/あの 位于句中时多数删。依据：中文对应垫词（「就是说」
  「那个」）的语用频率远低于日语，逐词对应会造出原文没有的啰嗦感——这是语言差异，
  不是删减。
- **压缩的形态优先级**：能**并入邻句**优于删除——孤立垫词行（0.3s 的「えっと」）
  并进后句（「呃……好像是咩咩的」）既消除闪现短行、又保住犹豫感。oneshot 的
  10,11 即此示范。次序：能并则并 → 并不了再看第三问。
  （合并受中央规则约束：≤2 连续源；未来若加 filler 合并特例，应带严格判据回归，
  如合计 ≤4s 且被并侧为 ≤2 字纯语气词。）

**第三问：压缩/并入后还有残余信息吗？——没有则丢弃：**

- 纯 floor-holding 的孤立垫词行（前后紧接正句、自身无情绪色彩）；
- 无语义感叹转写（喘息、干笑填充）；复读幻觉串（ルビルビルビルビ）；
- 套话幻觉（ご視聴 家族）走幻觉处置，不属本节。

### 三条结构性原则

1. **分层承担，丢弃≠抹除**：singles 的 corrected_text 忠实转写全部口语颗粒并在
   note 记录取舍（「宜丢弃」），取舍只发生在 translation/终稿层——「丢弃」的真实
   语义是「不进 SRT」。人工核对、后续窗口 advice、知识更新都还能看到原文。
2. **「不译某词」与「丢弃某行」是两个层级**：尾助词（ね/よ/さ/かな）通常不译、
   语气助词并进语气（「〜じゃん」→「……啊」）都是翻译常规，不需要任何丢弃授权；
   反方向同理：中文有自然对应的颗粒（えっと→那个/呃、まあ→嘛）照译，「轻小说感
   +可爱感」的风格要求本身依赖这些颗粒，不要为了「干净」除净。
3. **拿不准往保留偏**：删错人设颗粒不可逆且伤害核心价值，多留一个语气词只损失
   一点阅读预算——不对称性与 M.2（错并不可逆、漏并可恢复）同构。conf 与 note
   是表达不确定的正道，不是删除的理由。

### 与管线其他层的关系

- stabilize 的「高度疑似语气填充词」drop（低 conf + 短 + 能量正常）在**上游**先删
  一批明显噪声——本节次序作用于**存活到 LLM 层**的颗粒，两层判据独立、不重复。
- 本节第一问的「对话功能」判据同时约束 stabilize 侧未来的规则收紧：应答词
  （うん/はい）即使极短也不应被确定性规则误删。

### 三源「filler 三明治」合并特例（已实现：capable 档 v49；basic 保持严格）

v48 把合并硬上限收到 ≤2 连续源（依据：gap 自适应上游在解码时带音频证据做同句拼接，
位置优于 LLM 的文本级判断；3.5 生产 run 三源使用率仅 5/499，其中还混有不当的两句
合一）。唯一无法被 ≤2 上限等价表达的真实场景是 **filler 三明治**：
`正句碎片 + ≤2字纯语气词/口吃碎片 + 正句补全`——「丢中间再并两边」不可行（合并要求
连续源序号，丢 23 并 22,24 等于在源序号层撒谎），只剩「拆两行」或「丢 filler」，
前者伤并读流畅、后者伤犹豫感。

设计约束（缺一不可，均已落地）：

1. 判据必须模型可本地核查：被吸收源为 **≤3 字纯语气词/口吃碎片**（覆盖 えっと/なんか；ちょっと=4 字被拒）、合计 **≤4s**、
   合并后 **≤16 字**（不适用 36 字放宽）——数字+类别门，不用「同一句」类判断词
   （v47/48 实证：判断词是弱模型漂移的裂缝）。
2. **validation 机器闸门**：validation 手握源文本，三源行合法 ⇔ 中间源源文本加权
   长度 ≤3（`TRANSLATED_MAX_MERGED_SOURCES` 改为条件放行，不是放宽到 3）。
   规则不靠模型自律。
3. **仅 capable 档**；basic 保持 1:1+词中接回，不给例外。
4. oneshot 同步一个正例 + 一个反例（中间源为 3 字实词 → 拒绝），防例外泛化。

（原定按审计证据阈值触发；按用户决定于 v49 提前落地 capable 档，validation 同时补上合并源连续性硬校验。）

## 实测沉淀的 harness 原则

这些经验已固化为当前实现，改动时不要回退：

- 单靠"允许合并短片段"的 prompt 不够，模型会保守逐段输出；须在 prompt 中给清晰正反例，而不是 harness 提供本地合并候选。
- `finishReason=STOP` 不代表内容可靠：本地 CSV 解析、超长字幕检查、源序号覆盖检查缺一不可。
- SRT/CSV token 计数必须含结构开销；只算文本会严重低估窗口预算。
- 免费层 prompt 输入实测约 195k 报错——以 194k 为安全基线，而非官方 1M 上限。
- 采样默认显式 `temperature=1.0`；校验重试逐次 −0.01 并换 seed；`top_p`/`top_k` 不设。
- **Sticky retry 宜短、退避宜长（2026-07-29）**：观察发现 Gemini 即使返回 5xx，也会占用日额度；
  同 key sticky budget 从 7 降到 3，退避基数从 0.5s 提到 4s（`4×2^attempt`）。PerDay strike
  仍要连续 3 次才日封，但去掉「首末跨度 ≥5 分钟」门槛——在更短 sticky 预算下，反复 PerDay
  的 key 应更快轮换，而不是为 flicker 等待跨度。失败/重试的 HTTP 尝试同样计入本地 RPM
  （`note_request`），与 provider 侧按次计数对齐。
- **组合临时冷却（tier+model+key）**：某组合在一次 `chat_complete` 内耗尽 sticky retry（仅
  可重试错误）后，写入 `.state` 的 `combo_cooldowns`。**0–20 分钟**直接 skip（立刻换链上下一
  组合，不 sleep）；**20–120 分钟** probe 阶段 sticky retry=0（只打 1 次 HTTP），成功则清除，
  失败则重置冷却起点；**≥120 分钟**自动清除。与 daily-exhausted 独立。
- **test_profile（gemini 3.1 flash-lite）是优化基准，不是打折的近似**：harness/prompt 以
  「gemini 3.1 flash-lite 的 capability 能完成主要任务」为目标打磨（基准钉在这个具体版本，
  不随 flash-lite 档位的代际更替漂移）。它跑不好的地方默认按 prompt/harness 待改进处理，
  不要归咎模型；产线模型只是额外裕度。审计/评测报告注明所用模型是为了对照，不是为了给失效开脱。
