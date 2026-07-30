# 纠错 prompt 迭代方法论

如何用 `tools/session_replay` 在固定测试床上迭代纠错（correction R2）各处的
prompt。本文是**长期沉淀**：定位、原则、测试协议、失效模式与产物约定。某一轮的现场
交接笔记可写在本地 `docs/report/`（gitignore，不入库）。prompt 组装事实见
[`../llm_prompts.md`](../llm_prompts.md)；精修标定的合并软门槛见
[`../merge-calibration.md`](../merge-calibration.md)；决策记录见
[`../llm_design_notes.md`](../llm_design_notes.md)。

## 0. 定位

prompt-iterate 的目的是**通过针对性观察来迭代各处的 prompt**：在冻结上游注入物的
固定测试床上重放同一个窗口，只换 prompt，用元信息 + 逐行抽查看模型的真实反应，据此
改 prompt。它是 prompt/harness 迭代的**唯一机制**——知识更新
（`llm.knowledge.update`）不再生成 `<harness_notes>` 或任何 prompt 改进建议。

**边界**：这**不影响知识库更新**（`llm.knowledge.update` 那条链路照旧——主播设定、
术语、mistake ledger 等仍走原有反馈采集与自动 apply）。prompt-iterate 只管 prompt
文本本身怎么写得更好，知识库管往 prompt 里注入什么事实。两者正交。

**与 run-audit 的分工**：对**已完成**生产/reference run 做离线诊断（schema、知识提案、
成品 vs 精修、重试/并发时间线）走
[`.claude/skills/run-audit/SKILL.md`](../../.claude/skills/run-audit/SKILL.md)；
那里会引用本文的验收/抽样口径，但**不替代** session_replay 迭代。发现「该改 fragment」
后，回到本文 §2 开受控重放。

## 1. 变体（variant）与 tier

prompt 集合由**命名变体**决定（`prompt_variants.py`，Design A）：一个
`CorrectionVariant` 打包了所有曾经是 `tier is BASIC` 分支的选择（merge 片段、
reasoning 是否限界、是否要整窗 singles、注入契约/user 的短句）。**选变体与端点能力
解耦**：tier 仍分类应答模型，但只挑一个**默认变体**（`DEFAULT_VARIANT_FOR_TIER`）；
可用 `--variant NAME` 覆盖，服务任意注册变体。加变体 = 注册表加一条（+ 需要的新
fragment），无分支改动；每个变体**全量指定、无稀疏继承**，故耦合子句一起搬、不会漂移。

现注册四个：

变体名是 `<tier><字母>`——tier 已内嵌进名字（capableB/capableC 面向 capable 档/3.6 或
3.5 Flash，basicA/basicB 面向 basic 档/3.5 Flash Lite），因此一个变体不再按 tier 二次细分。

- **basicA**（basic 档对照，如 3.5-flash-lite）：保守 1:1，仅词中接回，限界 reasoning；v57 使用带 header 的十列输出，并在 `position` 后抄入首源 `start` 与输入对齐。
- **basicB**（basic 档生产默认）：继承 capableB 的去-singles 合并规则，输出带 `start` 的十列 CSV。
- **capableB**：判断型合并、开放 reasoning、无整窗 singles；使用带 header 九列 CSV、无 start。
  仅经 `--variant capableB` 手动选用，作为无局部 reasoning 的对照。
- **capableC**：capableB + **决策点前置 `# <局部推理>` 行**（合并 ≥2/discard/conf=low/越硬门槛
  的行必须在正上方写、单源在界内禁止写）——把 singles 唯一有用的"先想后写"以轻量形式带回、不要 1:1 全窗块；
  note 退化成纯短结论、逐行推理进 reasoning 行。门控只在 prompt 层要求；csv_utils 跳过 reasoning 行、
  仅**核对其 ids 与紧随 sub/discard 行的锚定**并记 `reasoning_rows`/`reasoning_unanchored`（告警级、不使整窗作废）。
  v62 起为 capable 档生产默认；`--variant capableC` 可用于显式重放同一配置。

所有单窗口 prompt 把目标行重编号为 `1..N`；只读前文按时间顺序编号为 `1-M..0`，离目标最近
的一行是 `0`。回复只能引用目标正序号，validator 通过后由 harness 映射回稳定源序号。research
transcript 仍是多窗口输入，保留全局源序号，但它生成的 context pack 不应把裸序号带进后续窗口。

同一份改动要在所有变体下都自洽——尤其别把只对一个变体成立的语义漏进另一个
（跨档语义实测是合并纪律的漂移源）。

**结构性变体的代价（capableB 的教训）**：去掉 singles 不止改契约——它波及了**约 6 个 fragment**
（output_contract、user 模板、merge_rules、examples oneshot、insert 示例、insert_rules
与 goals_translation 的单句），因为整套 prompt 是围绕"先 singles 再 translated"两阶段写的。
whole-file 换 + 小句 param-gate 能做，但这正是"结构性变体"比"换 param"贵得多的地方，也是
将来若要频繁做结构变体时，值得把契约/examples 进一步分解（Design B）的信号。生产默认变体
发生变化时必须 bump `PROMPT_VERSION`，由现有 task fingerprint 统一失效旧窗口缓存。

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
  的默认变体，不能代替 capableB/capableC 这类具名变体。
- 温度阶梯 1.00 起每次调用 −0.01；lite 实测 thinking=0。

**验收标准（当前）**：

- **Basic 组**：`--model 3.5-flash-lite -n 3 --max-attempts 10`，最多 10 次尝试内取得
  3 次 validation-ok 回复。
- **Capable 组**：同一 prompt/变体必须同时取得两组结果：
  1. 默认 `--model 3.6-flash -n 2 --max-attempts 5`，最多 5 次尝试内取得 2 次
     validation-ok 回复；只有 3.6 Flash 额度耗尽，或用户主动要求时，才改用
     `--model 3.5-flash`（同样为 2/5）；
  2. `--model 3.5-flash-lite -n 3 --max-attempts 10`，纠错变体须同时传同一个
     `--variant capableX`，最多 10 次尝试内取得 3 次 validation-ok 回复。
- **其他 session replay 沿用相同分档**：`GENERAL_CAPABLE` 轮（`research-r1`、`research-r2`、
  `fast-round1`）按 Capable 组的 `2/5 + lite 3/10`；纠错 r1（`query`，`LIGHTWEIGHT_MULTIMODAL`）
  与 search-loop judge（`search-judge`，`LIGHTWEIGHT`）按 Basic 组的 `3.5-flash-lite 3/10`。非 correction
  session 没有命名变体，直接用 `--model` 钉模型，不传 `--variant`/`--force-tier`。
- **调度约束**：同一模型的 replay 必须串行（跨变体、跨 session 也不能重叠）；不同模型可以并行。
  长任务启动后约每 10 分钟检查一次进度，不做高频轮询，也不要放任运行无人查看。
- **replay 按 variant 检查当前契约**：capableB/C 是九列 CSV；BasicA/B 是十列 start CSV。
  `start` 只是引导字段，最终时间轴不信任它；benchmark 报偏差仅作
  能力观测。
- **transport 保护（replay 与生产一致）**：单请求 timeout 为 15 分钟，网络层 sticky retry budget 为 3 次；但连续两次 timeout 时立即抛出当次原始 timeout failure，效果等同耗尽 retry 后的同类 failure，只是不再执行剩余 retry。transport failure 不伪装成 validation sample，也不计入 3/10 或 2/5 的模型回复数。
- **失败回复审计有最低抽样量**：一轮存在失败时，至少抽查 **5 个失败 sample**；若失败总数
  少于 5，则审计全部。对每个被抽中的失败 sample，至少逐条核查 **10 条 error**；若该 sample
  的 error 总数少于 10，则核查全部。这里的 error 指 validator/parser 报出的具体错误实例或
  对应的实际错误行，不能只看 `summary.md` 的聚合描述。完成这一级抽样后，才可归纳“主要失败
  原因”或真实 error pattern；不得从单个失败、单条报错或仅看错误计数直接外推。
- 质量判定 = 元信息（tokens / merges / note 纪律）+ 逐行抽查；固定窗
  `BV1ojjc6MEAs-0001` 另须跑
  [`../report/BV1ojjc6MEAs-0001-merge-drop-gold-v1.md`](../report/BV1ojjc6MEAs-0001-merge-drop-gold-v1.md)
  的离线 merge/drop gold。中性边界/行双向零罚；漏合并 1、错误合并边界 1；错误合并行只越
  软门槛另加 1，越硬门槛先累计软门槛 1、再另加 2；错 drop 5、漏 drop 2。不能再用压缩率或
  validation-ok 代替质量判断。`may_merge` 应容纳人工精修也支持、且在本 prompt 形状约束下
  两种切法都合理的局部边界；不要把“中性”误收紧成“默认禁止”，也不要机械照搬精修中的
  4–8 源超长注释块。
- **词条管理质量**（query / research-r1 / fast-round1 / correction R2 均适用）：
  `keep_entries` 只应包含 carried/preinjected 中实际出现且对后续窗口仍有价值的 key——
  审计时对照 `<carried_entries>` 或 `<preinjected_entries>` 检查：有无引用未注入的词条
  （幻觉 key）、有无遗漏明显仍相关的词条、空块是否合理（确实无后续价值）。
  `requested_entries` 不应重复已透传词条，且请求的 key 必须在 index 中可查；审计时对照
  `<streamer_index>` / `<common_index>` 验证。

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
   capableB 说“不要写 `<singles>`”，等于凭空引入 singles 概念、制造困惑；直接说“输出一个
   `<translated>` 终稿”即可（见 §5 singles 残留案例）。
5. **罢工是规模下的能力投降**：具名禁令/行数注入救不了错配的模型×任务组合；对症的是
   给对 tier 的 prompt（及必要时结构性缩窗）。
6. **例外必须带机器可核查的门**（数字+类别）并配正反例；纯判断词例外会被泛化。
7. **例子的长度会锚定输出长度**（模型倾向停在示例条数）；basic 用小例从根上避开。
   例子必须与规则**逐字自洽**：改了阈值（16→20）就要把示例里因旧阈值成立的判断
   （8,9 曾按 >16 否决）一并改对，否则示例与规则打架就是最坏的混淆源。
8. **分工明确**：全局/跨行判断（专名统一、话题、高风险区间、验证思路）写在 `<reasoning>`；
   单条的取舍/合并判断写在该行 note（local）里，不要在 reasoning 里逐行重复预演；有原生
   thinking 的模型在 thinking 里同理。
9. **结构性改动会波及多处，改前先 grep 概念全貌**：去掉一个“阶段/块”（如 singles）往往
   横跨契约、示例、merge 规则、reminder、insert 例等多个 fragment；换 param 只碰一处，
   删块要顺藤摸瓜（见 §5）。
10. **重构要有逐字护栏**：抽取/参数化 prompt 时，对未变的变体做 golden byte-diff，
    确保只动了想动的。

## 4. 当前合并口径（prompt 层）

- 单源为主（通常 ≥2/3 源不合并）；同一句被切开时最多合并**两个连续源**。
- 三源仅限 **filler 三明治**（两段正句碎片夹一个 ≤3 字纯语气词/口吃，合计 ≤4s、≤16 字），
  且仅 CAPABLE 教；BASIC 只允许词中接回。
- **换人说话绝不合并**（依音视频/语境判断说话人；存疑按不同人处理）；明显停顿、转折、
  情绪变化、话题切换、问答轮替也分开。
- **capableB/C**：合并后 >4s 或 >20 加权字即越硬门槛，原则上
  必须拒绝；仅同一个词或不可拆固定短语被源切断可特批。>7s 或 >36 字为绝对门槛，任何情况
  都不得越过。filler 三明治三源更紧：≤4s、≤16 字，不适用特批。basicA 仍只允许词中接回。

**validation 只硬查**：标签/列结构、basicA 的 singles 逐源覆盖、相邻性（合并源必须连续）。
源数上限与合并长度**不再硬拒**（2026-07-20 放松），只记 warning——口径靠 prompt 自觉，
让模型的自然合并行为可观测。char_count 不一致仅在 |误差| > 2 + 20%×实际值 时报 warning。

## 5. 失效模式知识

- **3.0-flash-preview（`gemini-3-flash-preview`）**：常出格式/结构性错误——典型是用**冒号
  替代小数点或列分隔符**，导致 CSV 解析失败。**已从验收协议弃用**，不再作为 A 系对照。
  （旧 catalog id `gemini-3.0-flash` 已 404 失效，正确 id 是 `gemini-3-flash-preview`。）
- **flash-lite 长窗罢工**：286 行整窗下 lite 会模仿示例只输出 N 行、或直接写「(此处省略…)」
  占位——能力投降，靠对 tier 的 prompt + 反占位契约缓解，不是靠加禁令；纯「反偷懒」禁令
  曾被模型在 reasoning 里复述后照样占位，无效。
- **死条款噪声**：把模型长期无视的数值/格式死要求写进契约，会降低对其它条款的服从；
  放宽无效条款后合并纪律可反而更好（以抽查为准，勿用压缩率代替）。
- **tier 错配**（工具 bug 教训）：replay 若不按应答端点逐 tier 组装，会把 capable prompt
  喂给 lite，basic 档改动永不到达模型——务必确认落盘 prompt 与实际下发一致（强制变体时看
  `prompt.system.<variant>.txt`）。

### 案例：变体 capableB 去 singles 的两轮 `<singles>` 残留

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

## 6. 关键文件

- prompt：`src/llm/prompt_templates/`（契约=`fragment_output_contract`、capable 合并规则/例=
  `fragment_merge_rules` / `fragment_examples_merge`、basic=`*_basic`、翻译取舍=
  `fragment_goals_translation`）+ `prompt_compose.py`（tier 参数化集中处、PROMPT_VERSION）。
- validation：`src/llm/csv_utils.py`（结构/覆盖/相邻性硬查；源数与长度仅 warning）。
- 工具：`tools/session_replay/`（`--model` 钉定、`--force-tier`、逐 tier 组装、配额中断报告）。
- 生产对照物：`out/reference/BV1ojjc6MEAs/llm-artifacts-prod-mm-high-20260719/`。
