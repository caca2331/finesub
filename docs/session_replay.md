# Session Replay：冻结上游产物，重打下游 session

> **前向提示（2026-08-19）**：本文描述的是**今天**的形态。知识库若按
> [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§7 未做与闸门](llm_agent_tool_protocol.md#7-未做与闸门)（设计在维护者的本地归档里）改成「索引必读 + 词条自主
> query」，`prepare-task` 与 `correction-request` 两个 replay session 的
> validation 契约含 `<keep_entries>`，届时要一并改；那一节还写了更大的一件——拉取式下 fixture
> 冻的不能再只是「发出去的 messages」，而要记录并回放 **pull 序列**。**不要据此实施新功能**，
> 先确认那个改动落地没有。

`python -m tools.session_replay` 用于快速迭代 prompt：把上游 stage 的注入物冻成 fixture，用**当前**模板重装目标 session 并调用 API。

> 位于 `tools/session_replay/`，**按需维护**：harness 接口或 fixture schema 变化时不要求
> 同步本工具，其测试不进默认套件（`python -m pytest tools/session_replay -n 0` 按需运行）。

当前已注册 5 个 session：`correction`（纠错窗口的交付轮，默认）、`prepare-task`（准备任务的首条消息）与 `correction-request`（纠错窗口带请求通道的首条消息），后两个见 [任务式 session](#任务式-session)；`speaker-judge`（说话人裁判的一个单元，也是三个 pass 的 pass 1）与 `proofread`（三个 pass 的 pass 3），见 [三个 pass](#三个-pass)。旧的 `query` / `research-r1` / `research-r2` / `search-judge` / `fast-round1` 随阶段 D 一起删了（2026-09-09）。`--list-sessions` 列出全部。合并软门槛与模型边界见 [`merge-calibration.md`](merge-calibration.md)；变体与验收协议见 [`prompt-iterate.md`](prompt-iterate.md)。

固定纠错窗 `BV1ojjc6MEAs-0001` 的 replay 还可用
[`tools/session_replay/benchmark.py`](../tools/session_replay/benchmark.py) 离线评估 merge/drop；完整 gold、
中性项及非对称代价标注在 `docs/reports/`（在 `dev` 上被跟踪，但不随仓库发布）。
评分不调用模型或网络，结构无效的回复直接标 invalid，不用 validation-ok 或压缩率冒充质量分。

> **变体（`--variant`/`--force-tier`）只有 `correction` 与 `correction-request` 支持**（后者叠在同一份纠错 prompt 上）。命名变体系统（`finesub.llm.prompt_variants`：`basicA`/`capableC`）绑定纠错专属的合并 fragment、decision reasoning 与 start 列要求；`prepare-task` 只有一套固定 prompt，对它传变体会**直接报错**（`NotImplementedError`），不静默回退到 baseline——避免 A/B 跑看似变了实则没变。要给它加变体，需注册 per-session 变体集并接进它的 builder。

生产与 `--force-tier` 的默认映射为 `capable → capableC`、`basic → basicA`；`--variant` 可显式选另一个已注册的变体。

## 纠错交付轮（默认）

复用同一窗的：

- **完整** `<search_results>` 正文（本地 search **与** extract 合并后的渲染结果；有 fixture 后**绝不**再调 `web_search`）
- `<pre_round_notes>` / `<entry_details>` / `<previous_advice>` / context_pack / 窗 CSV+clip
- system prompt 里那两个**非模板**来源：`style_block`（`--style` 选中的风格条目的投影，
  2026-09-02 前是渲染自 `## 精选` 的 `common_mistakes_block`）与 `extra_style`

⚠ **凡是进 system prompt 又不来自模板的输入，都必须进 fixture schema。** 上面最后一条正是
为此存在：知识库是活的，同一条命令隔天跑出的 `style_block` 可以不同，冻进 fixture
才让两臂比的是模板差异而不是库的差异。**给 harness 加一个新的 prompt 输入时，同一改动里要
往 `tools/session_replay/fixture.py` 加一个字段**——漏了不会报错，只会让 replay 悄悄漂移，
而漂移看起来和 prompt 改动的效果一模一样。

（注意这只保证 **replay** 的可比性。生产 run 之间没有这层冻结：知识库词条正文与常见错误正文
**不在 resume 失效键里**（[`llm_harness_behavior.md`](llm_harness_behavior.md)），所以跨生产
run 的 A/B 要靠实验时自己钉住库的状态。）

每轮变更的是现行 `prompt_templates` + `build_correction_csv_messages`；媒体按 profile 的
`correction_media` 重切上传。

旧 run 若仍把 `window_contexts` 存成 `chunk_id -> text`，生产 resume 会继续严格拒绝（窗口几何变化后
无法证明区间归属）。replay 抽取单个已冻结窗口时已经同时知道目标 chunk 与 source ids，因此只把
该目标旧笔记绑定到 fixture 的 source-id 区间；其他无法定位的旧笔记保留为 unbound 审计数据且不注入。

### 默认样例

| 项 | 默认 |
| --- | --- |
| Run | `out/reference/BV1ojjc6MEAs` |
| Chunk | `0001` |
| Fixture 落盘 | `<产物目录>/session-fixtures/correction-0001.json` |

抽取顺序：已有 fixture → 否则从交付轮 exchange 解析 → 写出 fixture：任务式交付轮的 exchange 是整条对话（system / 首条 user / assistant 请求回复 / 回填 user），窗口输入与建议从**首条** user 段取，搜索结果与词条全文从回填段取（直接交付没有回填，检索面冻成空；首条消息里的占位符不会被当成结果）。`task-artifacts` 里的 `correction_search_results` 没有渲染正文，不能单独当源。

### 命令

```powershell
# 只重装 prompt（不花生成配额）
python -m tools.session_replay correction --dry-run --label dry1 --note "检查组装"

# 默认调用 API，采满 2 条 validation 通过的回复
python -m tools.session_replay correction --label round-1 --note "本轮改动说明" --test-profile -n 2

python -m tools.session_replay --list-sessions
```

| 参数 | 含义 |
| --- | --- |
| `--run` | run 目录（含产物目录与媒体）。产物目录识别 `llm-artifacts`、`llm-artifacts-<label>`、`<stem>.llm-artifacts` 和 `<stem>-artifacts`；一次 run 同时留下多个时，**按实际含有的 replay 输入排序**取胜者（`-o` 派生的那个瘦目录只有 research-context，会被跳过），也可以直接把 `--run` 指到某个产物目录 |
| `--chunk` | 窗口 id |
| `-n` | 成功回复条数；未传时按 `--model` 自动取值：3.8/3.6 Flash 为 2，3.5 Flash Lite 为 3，其他为 3 |
| `--max-attempts` | 最多尝试次数；未传时按 `--model` 自动取值：3.8/3.6 Flash 为 5，3.5 Flash Lite 为 10，其他为 9 |
| `--label` | 输出子目录名 |
| `--note` | 写入 `summary.md` 的改动重点 |
| `--dry-run` | 不调生成 API |
| `--test-profile` | 走 flash-lite 测试链 |
| `--thinking-level` | 显式覆盖本次调用的 `thinkingLevel`（`minimal`/`low`/`medium`/`high`）；留空时使用当前 task-group × difficulty 格子的映射，不再读取旧 fixture/profile 的 `thinking_override` |
| `--profile` | 覆盖 fixture 的开关向量（如 `media=audio,retrieval=local,difficulty=quality`；退役的 preset 旧名 `mm-med` 等仍接受并换算；`media=text` 时不上传媒体） |
| `--model` | 把端点链钉在**单一**模型上：FREE 模型用精确短 ID（如 `3.8-flash`、`3.6-flash`、`3.5-flash-lite`），本地 agent 用它的 fact id（如 Capable 组默认的 `local-agy-gemini-3_8-flash`，这时显式给 `-n` 与 `--max-attempts`，默认采样计划不认它）；短 ID 精确匹配优先，模糊值命中多个模型时报错。成功计数按模型统计，配额耗尽时先落 summary 再报错，不静默回退。与 `--test-profile` 互斥 |
| `--temperature` | 采样温度（默认 1.00），**每次调用都用同一个值**；重试靠逐次递增的 `seed` 换采样，所以 n 条成功回复是同一分布的抽样，可直接当对照样本 |
| `--force-extract` | 忽略已有 fixture，从 exchange 重抽 |

### 输出

```text
out/prompt-iterate/<stem>-<chunk>/<label>/
  fixture.json
  prompt.system.txt
  prompt.user.txt
  reply-01.md / reply-02.md
  reply-01.translated.csv
  failed-*.md
  summary.md          # 对照主入口：token 汇总 + 改动重点 + 成功 translated
  clips/              # 本轮重切的媒体
```

`reply-*.md` / `failed-*.md` 顶部 JSON 保留关键 call meta：`model`、`api_key_label`、`thinking_level` / `thinking_budget`、`temperature`、`finish_reason`、`fallback_used`，以及规范化后的 `usage`（含 `thinking_tokens` / `output_tokens` / input 拆分）。usage 从 `LLMCallResult.raw_response` 提取，不再读不存在的 `.usage`。⚠ agent 路径的 `thinking_budget` 恒为 0，是占位值；实际思考档看 `thinking_level`（即传给 CLI 的 `--effort`）。

`summary.md`：元数据 → Token 汇总（非 dry-run）→ 本轮改动重点 → 成功回复（含 per-reply tokens + 全文 translated）→ 失败简表。

2026-08-13 用历史 286-row correction fixture 做过一次 3.5 Flash-Lite 真机兼容验收：1/1 成功，
产出 13,600 visible output tokens，13 个旧 fixture `char_count` 差异按兼容规则降为 warning；证明旧
fixture 迁移、当前 model-routing 格子注入、live adapter、usage 提取与 summary 落盘可共同工作。
长请求可能数分钟无新 stdout，监控时须等待实际进程/结果文件终态，不能把短轮询超时当成进程退出。

correction replay 按 variant 检查当前契约：basicA 是带 header 的七列 CSV；capableC（v99 起）是带 header 的六列 CSV，
格数与 header 不同的行判失败，并后越过 36 加权字或 7 秒的合并行也判失败；三个 pass 的 pass 2 的窗（fixture 重建出分组窗）按
`capableC-grouped` 的六列 CSV 校验，格数与 header 不同的行判失败。replay 每次按现在的 prompt 重新调模型、
只校验这次的回答，不复验录下的旧回答。模型看到的目标窗口序号每窗重置为 `1..N`，只读前文
按时间顺序编号为 `1-M..0`；validator 只接受目标正序号，并在通过后映射回 fixture 中的稳定源序号。
未强制 variant 时，replay 与生产一样使用 fixture profile 对应的 task-group × difficulty 格子
variant；模型端点只负责回答，不再用端点 tier 反推 prompt。`--model` 只把该格的 endpoint chain
收窄到指定模型，保留格子的 variant 与 thinking；跨变体对照仍须显式传 `--variant`。
离线 benchmark 对 basicA 使用 `--start-column`；它只读九列与十列的回答，六列的分组窗回答评不了；
start 偏差只作能力观测，不计入 validation 或 merge/drop 代价。replay 与生产统一使用同一套单请求
timeout（随允许的输出量推算的软/硬两段，见 `llm_harness_behavior.md`）和 3 次 transport sticky retry budget；连续两次 timeout 时提前抛出原始 timeout failure，
与耗尽 retry 后的 failure 类型和文本相同。

### Prompt iterate 采样协议

- Capable prompt/session：默认在 `local-agy-gemini-3_8-flash`（agy 的 Gemini 3.8 Flash；owner 2026-10-06 定为 agy 的 Flash，3.7 于 2026-10-12 下线）运行
  `-n 2 --max-attempts 5`；agy 不可用时依次改用 `3.8-flash`、`3.6-flash`（同样为 2/5），报告写明用的是哪条
  （agy 外面多一层 agent 外壳，读数与 API 的不完全可比）。另在
  `3.5-flash-lite` 运行 `-n 3 --max-attempts 10`。
  correction 的跨档 lite 对照须传与真机相同的 `--variant capableX`。
- Basic prompt/session：在 `3.5-flash-lite` 运行 `-n 3 --max-attempts 10`。
- `prepare-task` 属于 Capable，不传变体参数；`correction-request` 与 correction 同档、同变体。
- 同一模型的所有 replay 串行；不同模型可并行。运行期间约每 10 分钟查看一次进度。
- 审计失败回复时至少抽查 5 个失败 sample（不足 5 个则全部）；每个入样失败至少核查 10 条
  validator/parser error 或对应错误行（不足 10 条则全部）。必须基于这一级抽样确认真实错误
  pattern，不能只依据 `summary.md` 的失败简表、单个样本或单条报错下结论。

完整方法论与命令示例见 [`prompt-iterate.md`](prompt-iterate.md)。

## 任务式 session

除 correction 外，另有 2 个 session 各重放任务式流程的**首条消息**（冻结该消息的输入、只重打这一条；不重跑回填——回填要 harness 真的执行请求批）。它们共享 `sessions/base.py` 的 `run_text_replay`（纯文本、不上传媒体）与 `validate_session_contract`——直接查 `finesub.llm.session_contract.SESSION_CONTRACTS`（生产与 replay 的**同一份**输出契约），按 `nonempty`（顶层存在且非空）/`present`（顶层存在、可为空块）逐块校验；顶层提取用 `find_top_level_tag_blocks`（stack 解析，块内名字提及不算兄弟块）。一条回复先按「有没有 `<context_pack>` / `<translated>`」判定是交付还是请求批，再套对应契约。correction 因有 CSV 硬校验（含 `<translated>` 内 `<void>` 等二级标签处理）与媒体上传，走自己的骨架、不进本契约。

| session | 冻结输入（fixture 源） | 重放的 builder | 契约（nonempty / present 可空） | 调用角色 |
| --- | --- | --- | --- | --- |
| `prepare-task` | `research-prepare-input.json`（harness 在首条消息前 dump 的完整 builder 入参；超长素材分块 run 另有 `research-prepare-<块>-input.json`，adapter 只读主文件） | `build_prepare_task_messages`（`--agentic` 覆盖 dump 里记录的策略，重打工具会话形态） | 二选一：交付 `reasoning`,`context_pack`；请求批 `reasoning`,`analysis_notes` / `requested_entries`,`keep_entries`,`search_queries` | GENERAL_CAPABLE |
| `correction-request` | 复用 correction fixture（知识索引从该窗的 `*-correction-<chunk>-request-*.md` exchange 取，交付轮的 exchange 里没有） | `build_window_task_messages`（`retrieval=local` 或 fixture 带知识索引时有请求通道——none/native 下只请求词条；两者都没有时就是纯纠错首条消息；`--agentic` 换成工具会话形态：无请求通道、一次交付；`--variant`/`--force-tier` 同 correction） | 二选一：交付按 correction 的 CSV 硬校验；请求批 `reasoning` / `window_notes` + local 的 `search_queries` 或 none/native 的 `requested_entries` | 纠错 cell 的角色 |

validation 失败在同窗重试，与 correction 的格式重试同机制：温度不变，换 seed。

### 补中间态落盘（前置依赖）

`prepare-task` 的输入不是某条 exchange 的正文，而是 builder 的完整入参，因此依赖生产运行时落下的
`research-prepare-input.json`（`stages/prepare_task.py` 在首条消息前 dump，仅当传了 `task_artifact_dir`）。
即：要**先跑一次真实生产**把它 dump 下来，replay 才有源可读；缺文件时 adapter 报 `FileNotFoundError` 并提示跑一次生产。

命令与 correction 同形，把 `correction` 换成对应 session 名即可：

```powershell
python -m tools.session_replay prepare-task --dry-run --label p1 --note "准备任务 prompt 调整"
python -m tools.session_replay correction-request --label r1 -n 2
python -m tools.session_replay correction-request --agentic --dry-run --label r1-agentic
```

## 三个 pass

三个 pass 各能单独回放（pass 是什么见 [`llm_harness_behavior.md`](llm_harness_behavior.md) 的「三个 pass」）：

| pass | session | 输入 | 一次采样是什么 |
| --- | --- | --- | --- |
| pass 1 | `speaker-judge`，`--chunk` 是裁判单元的 id（`0001`，被对半拆过的是 `0001-a`） | 任务目录的 `speaker-judge-input.json`（裁判开跑时写：标题、概要、候选人物、模态、形状）+ stable 与它的裁判 sidecar（单元的行、前面单元带过来的新轨） | 生产的 `_judge_unit`：一个单元的调用连同它的重试与解析；`reply-NN.json` 是生产会记进 sidecar 的那条（`acts`、`results`…） |
| pass 2 | `correction`（照旧） | correction fixture；三个 pass 的任务多记 `three_pass.shape`，每次重建窗口时按 pass 1 的 sidecar 重新分组（`three_pass.pass2_plan`），窗口的原始行取窗口计划（`correction-window-plan.json`） | 同 [纠错交付轮](#纠错交付轮默认) |
| pass 3 | `proofread`，`--chunk` 是纠错窗口的 id | 任务目录的 `proofread-input.json`（pass 3 开跑时写：每窗的行与只读前后行、标题、背景、风格、知识库的名词） | 生产的 `_ask`：一窗的调用连同解析不了时的重问（次数同纠错窗口的 `max_retries_per_window`）；agent 路径的校验器照带 |

- 模板总是现在的，输入是那次运行的：pass 1、pass 3 改 prompt 后在同一份输入上重打。
- 走的是生产这一 pass 自己的调用，所以 `--variant` / `--force-tier` / `--thinking-level` / `--profile` / `--temperature` / `--agentic` 都不接受（报错），`--model` 照常只钉端点；调用的记录照生产写在输出目录的 `exchanges/`。
- dump 是这一版才开始写的：更早的运行没有，跑一次生产才有。
- pass 2 被对半拆开的窗（`0001-a`）不在窗口计划里，不能当三个 pass 的 pass 2 回放；只读前文按交付轮里看到的行数取原始行，可能比生产少几行（分组之后行数变少）。

```powershell
python -m tools.session_replay speaker-judge --chunk 0002 --dry-run --label p1 --run out/<stem>
python -m tools.session_replay proofread --chunk 0002 --label p3 --model local-agy-gemini-3_8-flash -n 2 --max-attempts 5 --run out/<stem>
```

## 与生产 resume 的区别

| | `session_replay` | 生产 resume |
| --- | --- | --- |
| 输入/状态 | 真实 run 冻结的单轮 builder 入参 | 重建本地状态；按精确输入 hash 复用已验证 session/纠错窗输出 |
| 目的 | 单轮 prompt 对照质量 | 生产崩溃续跑 |
| 默认 | 调 API | 完整管线；搜索/上传等廉价步骤可重做 |
