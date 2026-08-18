# Session Replay：冻结上游产物，重打下游 session

`python -m tools.session_replay` 用于快速迭代 prompt：把上游 stage 的注入物冻成 fixture，用**当前**模板重装目标 session 并调用 API。

> 位于 `tools/session_replay/`，**按需维护**：harness 接口或 fixture schema 变化时不要求
> 同步本工具，其测试不进默认套件（`python -m pytest tools/session_replay -n 0` 按需运行）。

当前已注册 6 个 session：`correction`（纠错窗口 R2，默认）、`query`、`research-r1`、`research-r2`、`search-judge`、`fast-round1`（见 [多轮 session](#多轮-session)）。`--list-sessions` 列出全部。合并软门槛与模型边界见 [`merge-calibration.md`](merge-calibration.md)；变体与验收协议见 [`prompt-iterate.md`](prompt-iterate.md)。

固定纠错窗 `BV1ojjc6MEAs-0001` 的 replay 还可用
[`tools/session_replay/benchmark.py`](../tools/session_replay/benchmark.py) 离线评估 merge/drop；完整 gold、
中性项及非对称代价标注在本地 `docs/report/`（gitignore）。
评分不调用模型或网络，结构无效的回复直接标 invalid，不用 validation-ok 或压缩率冒充质量分。

> **变体（`--variant`/`--force-tier`）仅 `correction` 支持。** 命名变体系统（`finesub.llm.prompt_variants`：`basicA`/`basicB`/`capableB`/`capableC`）绑定纠错专属的合并 fragment、decision reasoning 与 start 列要求，其余轮各只有一套固定 prompt。对非 correction 轮传变体会**直接报错**（`NotImplementedError`），不静默回退到 baseline——避免 A/B 跑看似变了实则没变。要给某轮加变体，需为该轮注册 per-round 变体集并接进它的 builder。

生产与 `--force-tier` 的默认映射为 `capable → capableC`、`basic → basicB`；`--variant` 可显式选择其余已注册对照组。

## 纠错 R2（默认）

复用同一窗的：

- **完整** `<search_results>` 正文（本地 search **与** extract 合并后的渲染结果；有 fixture 后**绝不**再调 `web_search`）
- `<pre_round_notes>` / `<entry_details>` / `<previous_advice>` / context_pack / 窗 CSV+clip

每轮变更的是现行 `prompt_templates` + `build_correction_csv_messages`；媒体按 profile 的
`correction_media` 重切上传，`planning_media` 只属于 query replay，不会误开 correction clip。

旧 run 若仍把 `window_contexts` 存成 `chunk_id -> text`，生产 resume 会继续严格拒绝（窗口几何变化后
无法证明区间归属）。replay 抽取单个已冻结窗口时已经同时知道目标 chunk 与 source ids，因此只把
该目标旧笔记绑定到 fixture 的 source-id 区间；其他无法定位的旧笔记保留为 unbound 审计数据且不注入。

### 默认样例

| 项 | 默认 |
| --- | --- |
| Run | `out/reference/BV1ojjc6MEAs` |
| Chunk | `0001` |
| Fixture 落盘 | `<产物目录>/session-fixtures/correction-0001.json` |

抽取顺序：已有 fixture → 否则从 R2 exchange 的 **user** 段解析 → 写出 fixture。`task-artifacts` 里的 `correction_search_results` 没有渲染正文，不能单独当源。

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
| `-n` | 成功回复条数；未传时按 `--model` 自动取值：3.7/3.6/3.5 Flash 为 2，3.5 Flash Lite 为 3，其他为 3 |
| `--max-attempts` | 最多尝试次数；未传时按 `--model` 自动取值：3.7/3.6/3.5 Flash 为 5，3.5 Flash Lite 为 10，其他为 9 |
| `--label` | 输出子目录名 |
| `--note` | 写入 `summary.md` 的改动重点 |
| `--dry-run` | 不调生成 API |
| `--test-profile` | 走 flash-lite 测试链 |
| `--thinking-level` | 显式覆盖本次调用的 `thinkingLevel`（`minimal`/`low`/`medium`/`high`）；留空时使用当前 task-group × difficulty 格子的映射，不再读取旧 fixture/profile 的 `thinking_override` |
| `--profile` | 覆盖 fixture 的开关向量（如 `media=audio,retrieval=local,difficulty=quality`；退役的 preset 旧名 `mm-med` 等仍接受并换算；`media=text` 时不上传媒体） |
| `--model` | 把端点链钉在**单一** FREE 模型上（推荐精确短 ID，如 `3.7-flash`、`3.6-flash`、`3.5-flash`、`3.5-flash-lite`）；短 ID 精确匹配优先，模糊值命中多个模型时报错。成功计数按模型统计，配额耗尽时先落 summary 再报错，不静默回退。与 `--test-profile` 互斥 |
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

`reply-*.md` / `failed-*.md` 顶部 JSON 保留关键 call meta：`model`、`api_key_label`、`thinking_level` / `thinking_budget`、`temperature`、`finish_reason`、`fallback_used`，以及规范化后的 `usage`（含 `thinking_tokens` / `output_tokens` / input 拆分）。usage 从 `LLMCallResult.raw_response` 提取，不再读不存在的 `.usage`。

`summary.md`：元数据 → Token 汇总（非 dry-run）→ 本轮改动重点 → 成功回复（含 per-reply tokens + 全文 translated）→ 失败简表。

2026-08-13 用历史 286-row correction fixture 做过一次 3.5 Flash-Lite 真机兼容验收：1/1 成功，
产出 13,600 visible output tokens，13 个旧 fixture `char_count` 差异按兼容规则降为 warning；证明旧
fixture 迁移、当前 model-routing 格子注入、live adapter、usage 提取与 summary 落盘可共同工作。
长请求可能数分钟无新 stdout，监控时须等待实际进程/结果文件终态，不能把短轮询超时当成进程退出。

correction replay 按 variant 检查当前契约：capableB/C 是带 header 的九列 CSV，多出的 `start`
判失败；BasicA/B 是带 header 的十列 CSV。模型看到的目标窗口序号每窗重置为 `1..N`，只读前文
按时间顺序编号为 `1-M..0`；validator 只接受目标正序号，并在通过后映射回 fixture 中的稳定源序号。
未强制 variant 时，replay 与生产一样使用 fixture profile 对应的 task-group × difficulty 格子
variant；模型端点只负责回答，不再用端点 tier 反推 prompt。`--model` 只把该格的 endpoint chain
收窄到指定 FREE 模型，保留格子的 variant 与 thinking；跨变体对照仍须显式传 `--variant`。
离线 benchmark 对 BasicA/B 使用 `--start-column`；
start 偏差只作能力观测，不计入 validation 或 merge/drop 代价。replay 与生产统一使用 15 分钟单请求
timeout 和 3 次 transport sticky retry budget；连续两次 timeout 时提前抛出原始 timeout failure，
与耗尽 retry 后的 failure 类型和文本相同。

### Prompt iterate 采样协议

- Capable prompt/session：默认在 `3.7-flash` 运行 `-n 2 --max-attempts 5`；额度耗尽时依次改用
  `3.6-flash`、`3.5-flash`（同样为 2/5）。另在
  `3.5-flash-lite` 运行 `-n 3 --max-attempts 10`。
  correction 的跨档 lite 对照须传与真机相同的 `--variant capableX`。
- Basic prompt/session：在 `3.5-flash-lite` 运行 `-n 3 --max-attempts 10`。
- `research-r1`、`research-r2`、`fast-round1` 属于 Capable；`query`、`search-judge`
  属于 Basic（search loop judge 因而只跑后一套）。非 correction session 不传变体参数。
- 同一模型的所有 replay 串行；不同模型可并行。运行期间约每 10 分钟查看一次进度。
- 审计失败回复时至少抽查 5 个失败 sample（不足 5 个则全部）；每个入样失败至少核查 10 条
  validator/parser error 或对应错误行（不足 10 条则全部）。必须基于这一级抽样确认真实错误
  pattern，不能只依据 `summary.md` 的失败简表、单个样本或单条报错下结论。

完整方法论与命令示例见 [`prompt-iterate.md`](prompt-iterate.md)。

## 多轮 session

除 correction 外，另有 5 个 session 各重放 harness 的一轮（冻结该轮输入、只重打这一轮，不重跑循环）。它们共享 `sessions/base.py` 的 `run_text_replay`（纯文本、不上传媒体）与 `validate_session_contract`——直接查 `finesub.llm.session_contract.SESSION_CONTRACTS`（生产与 replay 的**同一份**输出契约），按 `nonempty`（顶层存在且非空）/`present`（顶层存在、可为空块）逐块校验；顶层提取用 `find_top_level_tag_blocks`（stack 解析，块内名字提及不算兄弟块）。correction 因有 CSV 硬校验（含 `<translated>` 内 `<void>` 等二级标签处理）与媒体上传，走自己的骨架、不进本契约。

| session | 冻结输入（fixture 源） | 重放的 builder | 契约（nonempty / present 可空） | 调用角色 |
| --- | --- | --- | --- | --- |
| `query` | 复用 correction fixture 的窗口 + context | `build_correction_query_messages` | `reasoning` / `window_notes`,`keep_entries`,`search_queries` | LIGHTWEIGHT_MULTIMODAL |
| `research-r1` | `research-round1-input.json`（旧 run 回退到 artifact 或 run 根目录的 `*-research-context.json`） | `build_research_round1_messages` | `reasoning`,`analysis_notes` / `requested_entries`,`keep_entries`,`search_queries` | GENERAL_CAPABLE |
| `research-r2` | `research-round2-input.json`（旧 run 同上回退） | `build_research_round2_messages` | `reasoning`,`context_pack` / `keep_entries` | GENERAL_CAPABLE |
| `search-judge` | `search-loop-round-<N>.json` | `build_search_loop_messages`（`--loop-version v2` 时用 `build_search_loop_v2_messages`） | `reasoning`,`evidence_pack` / —（检索块可选，缺失即终止信号） | LIGHTWEIGHT |
| `fast-round1` | `fast-round-input.json` | `build_fast_round1_messages` | `reasoning`,`analysis_notes` / `requested_entries`,`keep_entries`,`search_queries` | GENERAL_CAPABLE |

`search-judge` 用 `--round-index` 选轮（缺省取目录里最新一轮）；`--loop-version {v1,v2}` 选 prompt 版本（v2 = 全量 pack 每轮、无 progress_update、query 驱动终止）。validation 失败在同窗重试，与 correction 的格式重试同机制：温度不变，换 seed。

research 两轮现在会把各自 builder 的完整语义入参落到 `research-round{1,2}-input.json`。旧 run 仍可回退读取 artifact / run 根目录的 `*-research-context.json`；该降级源通常不含完整 R1/R2 中间态，重放保真度低于专用 input dump。

### 补中间态落盘（前置依赖）

这些 session 的输入不是某条 exchange 的正文，而是循环内的完整入参状态，因此 fixture 依赖生产运行时落下的中间态：

- `research.py`：R1/R2 分别落 `research-round1-input.json` / `research-round2-input.json`。
- `search_loop.py`：每轮 judge 把 `build_search_loop_messages` 的全部入参序列化到 `search-loop-round-<N>.json`（仅当传了 `task_artifact_dir`）。
- `fast_session.py`：fast round-1 把合一轮入参落到 `fast-round-input.json`。

即：这两轮需要**先跑一次真实生产**把中间态 dump 下来，replay 才有源可读；缺文件时 adapter 报 `FileNotFoundError` 并提示跑一次生产。

命令与 correction 同形，把 `correction` 换成对应 session 名即可：

```powershell
python -m tools.session_replay query --dry-run --label q1 --note "查询轮 prompt 调整"
python -m tools.session_replay search-judge --round-index 2 --label j2 -n 2
```

## 与生产 resume 的区别

| | `session_replay` | 生产 resume |
| --- | --- | --- |
| 输入/状态 | 真实 run 冻结的单轮 builder 入参 | 重建本地状态；按精确输入 hash 复用已验证 session/纠错窗输出 |
| 目的 | 单轮 prompt 对照质量 | 生产崩溃续跑 |
| 默认 | 调 API | 完整管线；搜索/上传等廉价步骤可重做 |
