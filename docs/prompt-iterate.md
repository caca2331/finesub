# 纠错 prompt 迭代方法论

如何用 `tools/session_replay` 在固定测试床上迭代纠错（correction R2）各处的
prompt。本文是**长期沉淀**：定位、原则、测试协议、失效模式与产物约定。某一轮的现场
交接笔记放 `tmp/`、不进 `docs/`，接手后仍成立的发现提回本文。prompt 组装事实见
[`../llm_prompts.md`](llm_prompts.md)；精修标定的合并软门槛见
[`../merge-calibration.md`](merge-calibration.md)；决策记录见
[`../llm_design_notes.md`](llm_design_notes.md)。

## 0. 定位

prompt-iterate 的目的是**通过针对性观察来迭代各处的 prompt**：在冻结上游注入物的
固定测试床上重放同一个窗口，只换 prompt，用元信息 + 逐行抽查看模型的真实反应，据此
改 prompt。它是 prompt/harness 迭代的**唯一机制**——知识更新
（`finesub.llm.knowledge.update`）不再生成 `<harness_notes>` 或任何 prompt 改进建议。

**边界**：这**不影响知识库更新**（`finesub.llm.knowledge.update` 那条链路照旧——主播设定、
术语等仍走原有反馈采集与自动 apply）。prompt-iterate 只管 prompt
文本本身怎么写得更好，知识库管往 prompt 里注入什么事实。两者正交。

**与 run-audit 的分工**：对**已完成**生产/reference run 做离线诊断（schema、知识提案、
成品 vs 精修、重试/并发时间线）走
[`agent-tasks/run-audit/SKILL.md`](../agent-tasks/run-audit/SKILL.md)；
那里会引用本文的验收/抽样口径，但**不替代** session_replay 迭代。发现「该改 fragment」
后，回到本文 [§2 测试协议](#2-测试协议) 开受控重放。

## 1. 变体（variant）与 tier

prompt 集合由**命名变体**决定（`prompt_variants.py`，Design A）：一个
`CorrectionVariant` 打包了所有曾经是 `tier is BASIC` 分支的选择（merge 片段、
用哪份契约与 user 模板、注入它们的短句；`basicA` 的 `<decisions>` 决定清单就写在它自己的那份里）。**选变体与端点能力
解耦**：tier 仍分类应答模型，但只挑一个**默认变体**（`DEFAULT_VARIANT_FOR_TIER`）；
可用 `--variant NAME` 覆盖，服务任意注册变体。加变体 = 注册表加一条（+ 需要的新
fragment），无分支改动；每个变体**全量指定、无稀疏继承**，故耦合子句一起搬、不会漂移。

现注册两个，另加三个 pass 的 pass 2 用的分组版：

变体名是 `<tier><字母>`——tier 已内嵌进名字（capableC 面向 capable 档/3.8 或
3.6 Flash，basicA 面向 basic 档/3.5 Flash Lite），因此一个变体不再按 tier 二次细分。

- **basicA**（basic 档生产默认，如 3.5-flash-lite）：一行对一个源，只接回词中切断，可丢行，交付终稿时先写不超过 8 行的 `<decisions>` 决定清单（v109）；v101 起不再先写逐源的 `<singles>`，带 header、含 `start`；v104 起七列（不写 `duration|gap|char_count`，v101–v103 是十列）。原来的 basic 档默认 basicB（判断型合并、20 字/4 秒硬门槛）在乙的对比里输给它，2026-11-02 删除。
- **capableC**（v99 起，融合单 pass，owner 2026-10-30）：一次调用里先按认人与定形状的判据（`shape_criteria_*`）认人、判每处接缝
  （`join` / `may_join` / 其余，整行不要的 `drop`），再按 pass 2 的合并规则纠错、翻译、合并：`join` 的接缝要并，
  `may_join` 的看停顿、长度与翻译风格，其余不并；一条最多 4 行原始 ASR；六列输出、不写 `#` 注释，合并行的 note 写为什么是
  同一个人接着说；并后越过 36 加权字或 7 秒的合并行校验不过、进修复轮。拼法见 [`llm_prompts.md`「单 pass 的拼法」](llm_prompts.md#单-pass-的拼法)。
  v62–v98 的 capableC 是九列加决策点前置 `#` 注释的那份（2026-10-03 owner 定的双向门控随 capable 一族去掉注释作废，
  见 [`llm_design_notes.md`](llm_design_notes.md)「capable prompt 去掉辅助列与 `#` 注释的决策记录」）。
  `--variant capableC` 可用于显式重放同一配置。
- **capableC-grouped**（三个 pass 的 pass 2）：注释与辅助列一起去掉（owner 2026-10-23）：
  pass 2 上四个模型的对照里，中档能力以上的模型用不着辅助列与注释，`note` 留着主要为可解释；basic 一族当时保留辅助列，
  `basicA` 在 csv-v104 也去掉了。输出是六列 `type|position|corrected_text|translation|conf|note`，超一般长度的
  理由用一句短结论写进 note，校验数格子（[`llm_harness_behavior.md`「Prompt 信息」](llm_harness_behavior.md#prompt-信息)）。
  合并规则、契约与 user 模板与单 pass 的 capableC 是同几份，参数洞填「只并标着可并的行」。
  名字里的 C 原本表示「带注释」，名字不改。

所有单窗口 prompt 把目标行重编号为 `1..N`；只读前文按时间顺序编号为 `1-M..0`，离目标最近
的一行是 `0`。回复只能引用目标正序号，validator 通过后由 harness 映射回稳定源序号。research
transcript 仍是多窗口输入，保留全局源序号，但它生成的 context pack 不应把裸序号带进后续窗口。

同一份改动要在所有变体下都自洽——尤其别把只对一个变体成立的语义漏进另一个
（跨档语义实测是合并纪律的漂移源）。

**结构性变体的代价（capableB 的教训；capableB 是最早去掉 singles 的变体，2026-10 已删，它的去-singles 结构由 capableC 与 basicA 沿用）**：去掉 singles 不止改契约——它波及了**约 6 个 fragment**
（output_contract、user 模板、merge_rules、examples oneshot、insert 示例、insert_rules
与 goals_translation 的单句），因为整套 prompt 是围绕"先 singles 再 translated"两阶段写的。
whole-file 换 + 小句 param-gate 能做，但这正是"结构性变体"比"换 param"贵得多的地方，也是
将来若要频繁做结构变体时，值得把契约/examples 进一步分解（Design B）的信号。生产默认变体
发生变化时必须 bump `PROMPT_VERSION`。bump 不作废已提交的窗口：渲染出的输入变了的窗、当前校验器不再放行的窗才重做，见
[`llm_prompts.md`](llm_prompts.md) 里 `PROMPT_VERSION` 那一条。

## 2. 测试协议

```powershell
# 冻结 R1 注入物，只换 prompt 重打 R2；默认调 API 花配额
python -m tools.session_replay correction `
  --model 3.5-flash-lite -n 3 --max-attempts 10 `
  --label v51-<主要改动>-<模型> --note "改动说明"
```

- fixture：`session-fixtures/correction-0001.json`（BV1ojjc6MEAs 首个纠错窗口，286 源行、
  mm-high、新版低碎片 raw）；窗口 segments 在重放时从**当前** stable 重建，自动跟随最新 raw。
- `--model <id>`：把端点链钉在单一 FREE 模型上；推荐传精确短 ID，短 ID 精确匹配优先，
  模糊值命中多个模型会报错。成功计数按模型统计；**配额耗尽先落 summary 再报错，绝不静默
  回退/溢到付费**。未显式传 `-n`/`--max-attempts` 时，工具会按下述验收标准自动取值。
- `--variant <name>`：纠错 replay 直接钉住一个命名变体；Capable 变体在 lite 上的跨档对照也用
  同一个 `--variant capableX`（reply meta 仍记端点真实 tier）。`--force-tier` 只用于选择该 tier
  的默认变体，不能代替 capableC 这类具名变体。
- 温度阶梯 1.00 起每次调用 −0.01；lite 实测 thinking=0。

**验收标准（当前）**：

- **Basic 组**：`--model 3.5-flash-lite -n 3 --max-attempts 10`，最多 10 次尝试内取得
  3 次 validation-ok 回复。
- **Capable 组**：同一 prompt/变体必须同时取得两组结果：
  1. 默认 `--model local-agy-gemini-3_8-flash -n 2 --max-attempts 5`（agy 的 Gemini Flash，owner 2026-10-06
     定为常设，代替 API 的 `3.7-flash`，那个 API 模型已于 2026-10-11 弃用；agy 上的 3.7 于 2026-10-12 下线，改用 3.8），最多 5 次尝试内取得 2 次 validation-ok 回复；agy 不可用时依次改用
     `--model 3.8-flash`、`--model 3.6-flash`（同样为 2/5）。⚠ agy 走 agent 通道，prompt
     外面多一层外壳与胶囊，读数与 API 上的不完全可比，报告里写明用的是哪条；
  2. `--model 3.5-flash-lite -n 3 --max-attempts 10`，纠错变体须同时传同一个
     `--variant capableX`，最多 10 次尝试内取得 3 次 validation-ok 回复。
- **其他 session replay 沿用相同分档**：`prepare-task`（`GENERAL_CAPABLE`）按 Capable 组的 `2/5 + lite 3/10`，
  没有命名变体，直接用 `--model` 钉模型，不传 `--variant`/`--force-tier`；`correction-request` 跑在纠错 cell 上，
  按 correction 的分档并传同一个 `--variant`。
- **调度约束**：同一模型的 replay 必须串行（跨变体、跨 session 也不能重叠）；不同模型可以并行。
  长任务启动后约每 10 分钟检查一次进度，不做高频轮询，也不要放任运行无人查看。
- **replay 按 variant 检查当前契约**：basicA 是七列 start CSV；capableC 与分组窗的 `capableC-grouped` 是六列 CSV。
  `start` 只是引导字段，最终时间轴不信任它；benchmark 报偏差仅作
  能力观测。
- **transport 保护（replay 与生产一致）**：单请求 timeout 随允许的输出量推算（软/硬两段，见 `llm_harness_behavior.md`「API 调用的两段超时」），网络层 sticky retry budget 为 3 次；但连续两次 timeout 时立即抛出当次原始 timeout failure，效果等同耗尽 retry 后的同类 failure，只是不再执行剩余 retry。transport failure 不伪装成 validation sample，也不计入 3/10 或 2/5 的模型回复数。
- **失败回复审计有最低抽样量**：一轮存在失败时，至少抽查 **5 个失败 sample**；若失败总数
  少于 5，则审计全部。对每个被抽中的失败 sample，至少逐条核查 **10 条 error**；若该 sample
  的 error 总数少于 10，则核查全部。这里的 error 指 validator/parser 报出的具体错误实例或
  对应的实际错误行，不能只看 `summary.md` 的聚合描述。完成这一级抽样后，才可归纳“主要失败
  原因”或真实 error pattern；不得从单个失败、单条报错或仅看错误计数直接外推。
- 质量判定 = 元信息（tokens / merges / note 纪律）+ 逐行抽查；固定窗
  `BV1ojjc6MEAs-0001` 另须用 `tools/session_replay/benchmarks/BV1ojjc6MEAs-0001-merge-drop-v1.json`
  跑离线 merge/drop gold（它的人工标注依据在维护者本地的报告里，不随仓库发布）。中性边界/行双向零罚；漏合并 1、错误合并边界 1；错误合并行只越
  软门槛另加 1，越硬门槛先累计软门槛 1、再另加 2；错 drop 5、漏 drop 2。不能再用压缩率或
  validation-ok 代替质量判断。两臂成品的读速、行长、时长与间隙用合规表比（[`segmentation-split.md`](segmentation-split.md)
  「全文件的合规表」），它量的是形态，同样代替不了逐行抽查。`may_merge` 应容纳人工精修也支持、且在本 prompt 形状约束下
  两种切法都合理的局部边界；不要把“中性”误收紧成“默认禁止”，也不要机械照搬精修中的
  4–8 源超长注释块。
- **词条管理质量**（准备任务的请求批、纠错请求回合与交付轮均适用）：
  `keep_entries` 只应包含 carried/preinjected 中实际出现且对后续窗口仍有价值的 key——
  审计时对照 `<carried_entries>` 或 `<preinjected_entries>` 检查：有无引用未注入的词条
  （幻觉 key）、有无遗漏明显仍相关的词条、空块是否合理（确实无后续价值）。
  `requested_entries` 不应重复已透传词条，且请求的 key 必须在 index 中可查；审计时对照
  prompt 里的 `<*_index>` 块验证（每个可匹配类别一块，2026-09-07 起由类别集合生成，
  不再是写死的 `<streamer_index>` / `<common_index>` 两块）。

产物：`out/prompt-iterate/BV1ojjc6MEAs-0001/<label>/`（`summary.md` 为入口；**强制变体时只落
`prompt.system.<variant>.txt`/`prompt.user.<variant>.txt`**——就是实际下发的那份；不强制时才按
tier 落 `prompt.system.txt`(capable)+`.basic.txt`(basic) 两份参考；`reply-NN.translated.csv` 为解析后终稿；
`failed-attemptNN.md` 为失败样本）。只有正常跑满 `-n` 才写 `summary.md`；中途抛异常
（配额/断连）时 reply/failed/prompt 产物仍在，缺汇总。

### 产物目录命名

`out/prompt-iterate/<stem>-<chunk>/v<NN>-<主要改动>-<模型>/`，例如
`v51-speaker-nomerge-35flash`、`v51-relaxmerge-lite`。`v<NN>` 跟 PROMPT_VERSION；
主要改动用短横线连接的关键词；模型用 `36flash` / `35flash` / `35lite` / `30preview` 等简写。

## 3. prompt 原则（迭代实证，持续累积）

1. **不提供会使模型混乱的信息**（总原则）：跨档语义、死条款、冗余复述都是干扰。
   改 prompt 时站在**执行模型的视角**审视是否够清晰；除非为强调，一律简洁不冗余。
2. **死条款就是噪声**：模型做不到/不肯做的要求（如 lite 写 gap 数值）留在契约里只会
   稀释其他条款的权威——诚实的契约比严格的契约表现更好。
3. **过约束的阈值会被无视、连带削弱其他条款**：软门槛设得远低于模型实际行为（16 字曾被
   3.5 的 70% 合并行突破），就沦为耳边风。软门槛要**贴合真实边界**，并把措辞强化成
   “默认边界，不是尽量”——数值现实 + 语气坚定，两者缺一不可。
4. **正面说“做什么”，别负面禁止一个模型根本没有的概念**：对只输出 `<translated>` 的
   变体（当时的 capableB，已删）说“不要写 `<singles>`”，等于凭空引入 singles 概念、制造困惑；直接说“输出一个
   `<translated>` 终稿”即可（见 [§5 失效模式知识](#5-失效模式知识) singles 残留案例）。
5. **罢工是规模下的能力投降**：具名禁令/行数注入救不了错配的模型×任务组合；对症的是
   给对 tier 的 prompt（及必要时结构性缩窗）。
6. **例外必须带机器可核查的门**（数字+类别）并配正反例；纯判断词例外会被泛化。
7. **例子的长度会锚定输出长度**（模型倾向停在示例条数）；basic 用小例从根上避开。
   例子必须与规则**逐字自洽**：改了阈值（16→20）就要把示例里因旧阈值成立的判断
   （8,9 曾按 >16 否决）一并改对，否则示例与规则打架就是最坏的混淆源。
8. **分工明确**：单条的取舍/合并判断写在该行 note（local）里；没有会话要求写出推理（纠错窗 v107 起，其它会话 2026-11-08 起），`basicA` 的
   `<decisions>` 只列全窗的结论（专名、不可靠区间、要丢的行），不逐行预演；有原生 thinking 的模型在 thinking 里同理。
   给模型留可见的块时写成「结论清单」而不是「推理」：明着要求写推理会被 Claude 以 `reasoning_extraction` 拒答。
9. **结构性改动会波及多处，改前先 grep 概念全貌**：去掉一个“阶段/块”（如 singles）往往
   横跨契约、示例、merge 规则、reminder、insert 例等多个 fragment；换 param 只碰一处，
   删块要顺藤摸瓜（见 [§5 失效模式知识](#5-失效模式知识)）。
10. **重构要有逐字护栏**：抽取/参数化 prompt 时，对未变的变体做 golden byte-diff，
    确保只动了想动的。
11. **迭代不能只做加法**：修一个失效模式，先看能不能靠删掉、合并或改写已有的句子解决，
    实在不行才加；加了新句子，同时找出因此变得多余的旧句子删掉。只增不减的 prompt 越修越长，
    条款互相稀释（第 2、3 条），新规则还会和旧规则打架。目标是在清晰的前提下尽量简短。

## 4. 当前合并口径（prompt 层）

- **basicA**：一行对一个源，只允许词中接回（两个连续源、边界无标点无空格、拼回同一个词、后一行的 gap ≤0.3s、跨度 ≤7s）。
- **capableC（融合单 pass）**：先认人、判接缝（pass 1 的判据），`join` 的接缝要并、`may_join` 的看停顿、长度与翻译风格，
  其余不并；一条最多 **4 个连续源**，夹在句中的垫词走三明治、不单独 discard；长度一般 20 字/4 秒左右，同一句话确实连着
  可以稍长、理由写进 note，>7s 或 >36 字为绝对门槛（越过的合并行校验拒、进修复轮，owner Q13）。
- **capableC-grouped（pass 2）**：只并 pass 1 标着「可并」的接缝；最多两个连续源（分组后的行），三源仅限三明治；长度同 capableC，
  绝对门槛只写在 prompt 里（越限的 join 由 `grouping.py` 在 pass 1 那边改成可并）。
- **换人说话绝不合并**（依音视频/语境判断说话人；存疑按不同人处理）；明显停顿、转折、
  情绪变化、话题切换、问答轮替也分开。
  任务有说话人标签时（v95），`<asr_result>` 多一列 `turn`，标了「换人」的行不得与上一行合并；由 validator 检出、只告警，不发修复请求。

**validation 只硬查**：标签/列结构、每个源都被覆盖或显式丢弃、相邻性（合并源必须连续）；capableC 另查合并行的绝对门槛（Q13）。
跨换人标记的合并只告警、不发修复请求。源数上限与其余变体的合并长度**不硬拒**（2026-07-20 放松），只记 warning——口径靠 prompt 自觉，
让模型的自然合并行为可观测。

## 5. 失效模式知识

- **3.0-flash-preview（`gemini-3-flash-preview`）**：常出格式/结构性错误——典型是用**冒号
  替代小数点或列分隔符**，导致 CSV 解析失败。**已从验收协议弃用**，不再作为 A 系对照。
  （旧 catalog id `gemini-3.0-flash` 已 404 失效，正确 id 是 `gemini-3-flash-preview`。）
- **flash-lite 长窗罢工**：286 行整窗下 lite 会模仿示例只输出 N 行、或直接写「(此处省略…)」
  占位——能力投降，靠对 tier 的 prompt + 反占位契约缓解，不是靠加禁令；纯「反偷懒」禁令
  曾被模型在 reasoning 里复述后照样占位，无效。
- **弱模型会把输入的列布局迁移到输出**（v55–v58，flash-lite）：输入带 `start` 列时，lite 会在九列输出里
  插回 `start`，v55 的失败绝大多数是这种。对策是让输出列与输入对齐（basicA 的十列，`start` 只作引导，
  时间轴按源序号回填），而不是禁止它；header 对结构稳定仍有正收益（v57 3/4 对 v58 3/6）。
- **给 lite 换成 JSONL 对象输出不比带 header 的 CSV 可靠**（v60 / v61 各 0/10）：键名漂移与 JSON 转义是
  两类新的自由度。该支线已随 v65 删除，别重做。
- **说话人裁判的逐行 JSON 在 quality 档的模型上不是失败来源**（2026-10-19 离线计数，主 checkout 的 `out/` 与 `data/`
  里留下的全部裁判与 pass 3 调用，speaker-judge-v2 到 v6，素材是几份 hina 验收窗）：裁判 GPT-6.1 Sol 36 次、GPT-6 Luna 18 次全部一次读通；
  agy Gemini 3.8 Flash 80 次里 9 次重问，7 次是 agy 答案通道的畸形（思考泄漏、只回 `test` 一类，见
  [答案通道本身不可靠](llm_local_agent_agy.md#7-答案通道本身不可靠约四分之一的回答是畸形的2026-09-02)），只有 2 次跟格式有关：一行把键名写成 `id0`、一行把形状字段的
  `drop` 填进了 `track`，都不是 JSON 语法错，重问后都过了。pass 3 校对（五列 `|` 分隔，全是 agy Gemini 3.8 Flash）15 次全部一次读通。
  上一条 lite 的结论是纠错窗的多列长输出，不外推到这里；这组读数也只说解析，不说裁判或校对判得好不好。
- **v77 / v78c 的成品质量审计：查不出可归因的差别，run 间噪声压倒一切（2026-08-25）**：
  每臂 5 份成品，按「这份答案把哪些相邻源接在一起」算 Jaccard——
  **臂内一致度 0.360 / 0.363，跨臂 0.325**：两臂彼此的差别，和各自复跑的差别一样大。
  所以合并数（27.0 vs 30.8）、discard 数（4.2 vs 2.4）这类差异都**不能归因于 prompt**。
  抽查分歧最大的几处边界也是两边各有优劣（v78c 不再把相隔 1.4 秒的 `そのー`/`あれですか?`
  接起来，更合理；却把 `ちょっと待ってごめん` 并进了一条长句，更可疑）。
  两处方向性差别值得留意，但 n=5 下都算不上定论：**越软门槛的合并行（>2 源）v77 有 3 处、
  v78c 为 0**；char_count 越容差行 11 → 2。覆盖率两臂都是 303 源里 299/301。
  **金标准打分（对齐层修好后补跑）证实了同一结论**：加权代价 v77 均值 17.6（中位 18，9–28）、
  v78c 均值 13.0（中位 13，5–25）——差 4.6 而臂内标准差约 7.5，两组区间大幅重叠，不显著。
  分项里 overmerge（4.8 vs 5.0）与 undermerge（3.4 vs 3.6）几乎相同，唯一有方向的是
  **false_drop 1.8 → 0.8**（权重 5，几乎解释了全部差额），与上面按原始 discard 计数看到的一致。
  **教训**：合并/取舍这类决定在单窗 n=5 上信噪比太低，别拿它给 prompt 定性——要把 n 抬上去。
- **金标准的源行对齐层（2026-08-25 修复）**：金标准按 **286 源**标注，窗口现已重切成 **303 源**，
  于是每个 id 都指向别处、`benchmark.py` 直接拒绝执行。修法不是重新编号，而是**按时间对齐**——
  金标准里那条「边界」是音频上的一个时刻，新切分若仍在那里断开，判断就适用于此刻相邻的那两行。
  实现在 `tools/session_replay/alignment.py`，金标准 JSON 里新增 `sources`（它当年审的那 286 行，
  从同期 replay 产物里捞回来）作为对齐依据；`benchmark.py` 先试精确匹配，失败再对齐。
  **关键是不许硬凑**：新切分多出来的边界（正是那些把短语切碎的细分）金标准从没审过，按它的
  `must_not_merge` 默认去罚就是在惩罚正确行为——所以它们进**第三类「未审」**，两个方向都不计分，
  并在报告首行打出覆盖率。本窗实测：**223/302 边界、298/303 源带判断**，金标准里 3 条 must_merge、
  21 条 may_merge、1 条 may_drop 因为边界已不存在而丢失。
  副作用要知道：软/硬长度附加费只对「并了非中性边界」的行触发，所以完全由未审边界拼出来的
  超长行不会被罚——保守方向（对边界的沉默不能变成惩罚），但也意味着对齐后的分数是**下界**。
  同日的三处收口：
  - **对齐只答"重切"这一种失配**。金标准指纹连 ASR 文本一起哈希，所以解码变了、边界一条没动
    也会失配；那种情况下对齐是个空重映射，却顺手把指纹守卫删掉——等于抹掉"金标准已过期"的
    唯一信号。现在 `prepare_benchmark` 先比切分（`alignment.same_cut`）：切分没变就拒绝，
    让人去重标。金标准本身坏了（默认值写错、两个集合不互斥）也永远不走对齐，它是
    `BenchmarkMalformedError`，与失配类 `BenchmarkWindowMismatchError` 分开。
  - **重切把"该丢"和"该留"并进同一新行时，那行也归未审**。原来它按任意时间重叠继承 `must_drop`，
    于是打分会去奖励删掉金标准要求保留的内容。覆盖率行单列这类行的条数。行的归属还要求**实质
    重叠**（0.05 秒，或短行的一半），否则相邻两行漂移 1 毫秒就算彼此覆盖。
  - **`--json` 带上覆盖率**。顶层从裸 score 数组改成 `{"alignment": …|null, "scores": [...]}`；
    自动化比对正是最容易把下界当定论的读者。
- **降低精度要求（v78c）：可见推理里确实不提了、列也更准，但 thinking 总量仍然升（2026-08-25）**：
  按 owner 意见换成"不必逐字符推演、差不多就行、只作分句参考"——不禁止思考，而是让它**不值得**
  精算。agy 3.7-flash api 档 n=5，与 v77 的 n=7 对照：

  | 臂 | thinking 均值 | 可见 `<reasoning>` 长度 | 提到 char_count 的回复 | 越容差的 char_count 行 |
  | --- | ---: | ---: | --- | --- |
  | v77 | 16,025 | 919 字 | 3/5 | 11/1829，**全部高估**，均值 +6.5 |
  | v78c | 24,863 | 743 字 | **0/5** | 2/1633，**全部低估**，均值 −10.2 |

  **可见层面完全奏效**：模型不再在 reasoning 里谈 char_count，推理块短了 19%，而且越容差的行从
  11 降到 2——**估得比算得还准**（原来是系统性高估，说明它"算"的规则本来就错）。
  **但内部 thinking 仍然涨了 55%**，而涨的部分不在 char_count 上。涨到哪去了**答不出来**：
  Gemini 只回 thinking 的 token 数，不回文本，产物里没有可抽查的思考正文；要查得专门开
  `includeThoughts` 的思考摘要跑一轮。
  **三臂放在一起看**（v77 16.0k < v78c 24.9k < v78b 32.4k）：单调关系仍然成立——**在契约里提到
  一列该花多少力气，就会让总思考变多**，哪怕提的是"少花点"。目前没有一种措辞能把它降下来。
- **禁止模型在思考里推演某个量，会让它在那个量上想得更多（2026-08-25，v78b 试作，未采纳）**：
  上一条在 lite 上被"模型太弱、动 prompt 就崩"解释掉了，于是按 owner 意见改到带思考的模型
  重做，措辞也从含糊的"是估算"收紧成明确的**禁止行为**：「不要在思考里推演这三个数……禁止在
  reasoning 或内部思考中逐字计数、逐项相加、反复核对小数位或自行验算」。
  **agy 3.7-flash（api 档，逐调用新会话）对照，每臂 n=7**（一次 2 样本 + 一次 5 样本）：

  | 臂 | thinking 均值 | 中位 | 范围 | 可见输出均值 | validation-ok |
  | --- | ---: | ---: | --- | ---: | --- |
  | v77 原文 | 16,025 | 13,619 | 2.5k–29.4k | ~15.2k | 全部一次过 |
  | v78b 加禁令 | **32,412** | 31,175 | 15.8k–40.4k | ~14.5k | 全部一次过 |

  **thinking 翻了一倍（2.02×）**，方向与意图相反；可见输出与质量闸门没有区别。中位数相差
  2.3 倍、区间几乎不重叠，远大于臂内波动。可信的解释是**「不要想 X」本身就把 X 抬成了显著
  话题**——模型先要判断"我是不是在推演"，就得先推演一遍。
  **推论（写给下一次）**：想减少某个量上的思考，不要去禁止思考它，而是**把它变得不值得想**
  ——例如降低它的精度要求、或干脆不让模型填。owner 已定「继续由模型填」是为了逼它形成长度
  判断（取舍记录在维护者的本地计划里），
  所以这条路暂时封闭；真要动，得先重新审那个取舍。
  产物：`out/prompt-iterate/BV1ojjc6MEAs-0001/{v77-n5-agy37,v78b-n5-agy37,v77-baseline-agy37,v78b-no-derivation-agy37}/`。
- **「这几列是估算，不必核算」写进契约会降低结构服从（2026-08-24，v78 试作，未采纳）**：
  conversational 首次真机实测里那个 agent 为求 `char_count` 精确而逐字数数、输出剧增
  （读数记录在维护者的本地计划里），于是在三个
  output contract fragment 的 `char_count` 条目后加了一句「`char_count`/`duration`/`gap` 是
  估算不是核算，写下判断即可，不要逐字数、反复核对小数位或另写脚本验算」，同时把加权字数
  规则按实现写实（`P*`/`N*`/空格/拉丁字母 0.5，`S*` 计 1，`+ = → ♪` 不算标点）。
  **固定窗 `BV1ojjc6MEAs-0001` + `3.5-flash-lite` 对照（各 10 次尝试、同参数同窗）**：

  | 臂 | validation-ok | 失败主因 | 输出 token（合计/单次成功） |
  | --- | --- | --- | --- |
  | v77 原文 | **3/10**（达标） | — | 128,165 / ~12.9k |
  | v78 加两句 | **1/10**（不达标） | `Row N has too few fields` 106 例 + 覆盖缺口 | 129,627 / 13.2k |

  两臂 token 几乎相同，**省不出输出**；而结构合规率掉到验收线以下。可信的机制是：告诉模型
  这几列「不必仔细」，它连**把这几列写出来**都变松了——占压倒多数的失败正是少列。这与上面
  「死条款噪声」是一体两面：**放宽一条约束的措辞会外溢到相邻约束的服从度**，不能只按字面
  预期它的作用范围。n 小（3 vs 1），没有再跑噪声基线，因为按预注册验收标准 v78 已经不达标；
  要重启这条改动，先分开试「只写实加权规则」与「只加反过度思考句」两臂。
  另外两臂都观察到：lite 写的 `char_count` **系统性偏大**（成功回复里 19 行与实测值不符，
  几乎全部偏高），说明 0.5 加权规则在这个模型上本来就没落地——把规则写得更准并不能改变它。
  产物：`out/prompt-iterate/BV1ojjc6MEAs-0001/{v77-baseline-lite,v78-estimate-not-audit-lite}/`。
- **死条款噪声**：把模型长期无视的数值/格式死要求写进契约，会降低对其它条款的服从；
  放宽无效条款后合并纪律可反而更好（以抽查为准，勿用压缩率代替）。
- **tier 错配**（工具 bug 教训）：replay 若不按应答端点逐 tier 组装，会把 capable prompt
  喂给 lite，basic 档改动永不到达模型——务必确认落盘 prompt 与实际下发一致（强制变体时看
  `prompt.system.<variant>.txt`）。

### 案例：变体 capableB 去 singles 的两轮 `<singles>` 残留

capableB 已于 2026-10 删除，它的去-singles 结构由 capableC 与 basicA 沿用，下面的教训照旧成立。

做变体 capableB（去掉整窗 singles、直接输出 `<translated>`）时，`<singles>` 概念的清除**分两轮
才干净**，正好各印证一条原则：

1. **结构性改动波及多处（原则 9）**：先只改了契约和 user 模板，`--dry-run` 组装后 grep
   仍有 **29 处 `singles`**——merge 规则里的“先完成 singles 再写 translated”、42 行
   oneshot 的整个 `<singles>` 块、insert 示例的 `<singles>` 块、以及 insert_rules /
   goals_translation 的单句都还在。（注：insert/插轴自 v63 起已完全废弃，相关模板已移入
   `legacy/`；此处提及的 insert 相关 fragment 仅作历史上下文参考。）去一个”阶段”实际横跨 ~6 个 fragment；顺藤摸瓜（whole-file
   换 + 小句 param-gate）后才降到只剩“禁令”。**教训：删块前先 grep 概念全貌，别只改最显眼处。**
2. **别负面禁止一个模型没有的概念（原则 4）**：即便降到只剩 3 处，那 3 处全是
   “**不要输出 `<singles>`**”“不写 singles 对照稿”这类**禁令**。对一个从来没有 singles
   阶段的模型，这是在凭空引入 singles、制造困惑。最终全删，只**正面**描述“输出一个
   `<translated>` 终稿”——C 的 singles 计数归零。**教训：说做什么，不说别做那个它压根不知道的东西。**

### 案例：合并越限后自己拆开，却不撤回原行（2026-08-15，BV1ojjc6MEAs 窗口 0001）

605 条素材的第一个窗（源 1–303）连续 5 次验证失败、第 6 次才通过。**六次里没有一次是内容
质量问题**，全部是输出契约的机械违规，而且每一次的正确内容都已经在响应里。

| attempt | 行数 | 覆盖 | 错误 |
| --- | ---: | --- | --- |
| 0 | 243 | 303/303 | 4 对源 id 重复 |
| 1 | 289 | 303/303 | 9 对源 id 重复 |
| 2 | 262 | 302/303 | 漏 id 47 |
| 3 | 272 | 302/303 | 漏 id 6 |
| 4 | 252 | 303/303 | 33,34 与 32 顺序颠倒 |
| 5 | 253 | 303/303 | 通过 |

**主失效模式（attempt 0/1）**：模型合并了两条源，发现越门槛，正确地拆成两行输出，
**却没有撤回那条合并行**。形状每次都一样，连续三行：

```text
sub|120,121|4.2|0.3|…|high|36|# 12     ← 越限的合并行（4.2s > 4s；36 字触顶）
sub|120  |2.5|…                          ← 拆出的前半
sub|121  |1.7|…                          ← 拆出的后半
```

它**知道自己在干什么**——attempt 1 的 note 直接写着「合并后 23 字越硬门槛，拆开」
「越硬门槛拆开」「合并后 5.3s 越硬门槛，拆开」。残留的四行时长分别 4.2/5.5/5.2/4.9 秒，
全部越 4 秒软门槛，其中一行 37 字越了 36 字绝对门槛：**留下的恰恰是违规的那一版**。
prompt 为这种情况准备了 `<void>`（「写完某行才发现不对时，行尾 `<void>` 废弃再重写」），
模型没有用。

**残留是纯局部的，没有波及别处**——这一点对判断严重度很关键：

- 逆序点数**恰好等于**重复对数（4/4、9/9），减掉重复自身贡献的回退后剩 0，说明 id 序列
  其余部分严格单调
- 重复副本最大间隔 2 行，全部紧邻
- 覆盖率 303/303 完整

即「一处出错、后面全跟着串位」并**没有**发生；残留行占 1.6%（4/243）与 3.1%（9/289）。

**attempt 2/3 漏的是不同的 id**（47、6），邻居正常，说明不是素材难点而是随机遗漏。
**attempt 4 的两条报错其实是同一次错位**：先输出 `33,34` 再回头补 `32`，validator 拒掉逆序行
之后 32 就成了「缺失」；内容本身正确，按时间轴重排即可。

**教训**：这类「模型知道规则、也执行了判断、只是漏了一个收尾动作」的失效，靠加强 prompt
禁令收益递减（与本节前面几条同理）。真正的解法是让它看见 validation 错误再改一遍。
盲重掷去撞它的代价：6 次调用、47 分钟、整个免费档日额度。

**已实施**（2026-08-15）：同窗重试现在带上一轮输出与逐条校验错误，契约见
[`../llm_harness_behavior.md`](llm_harness_behavior.md)「重试与拼接」。**这条对 prompt 迭代
的影响**：这类机械违规不再是 prompt 强化的靶子——再收紧措辞前，先确认修复轮也治不好它。

## 6. 关键文件

- prompt：`src/finesub/llm/prompt_templates/`（契约=`fragment_output_contract`、capable 合并规则/例=
  `fragment_merge_rules` / `fragment_examples_merge`、basic=`*_basic`、翻译取舍=
  `fragment_goals_translation`）+ `prompt_compose.py`（tier 参数化集中处、PROMPT_VERSION）。
- validation：`src/finesub/llm/output_protocol.py`（结构/覆盖/相邻性硬查；源数与长度仅 warning）。
- 工具：`tools/session_replay/`（`--model` 钉定、`--force-tier`、逐 tier 组装、配额中断报告）。
- 生产对照物：`out/reference/BV1ojjc6MEAs/llm-artifacts-prod-mm-high-20260719/`。
