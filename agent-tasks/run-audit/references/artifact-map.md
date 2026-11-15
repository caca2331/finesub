# 产物地图：run 目录里有什么、schema 是什么

审计时的寻路图。权威文档：`docs/llm_harness_behavior.md`（运行时行为）、
`docs/knowledge.md`（知识库与反馈）、`docs/llm_prompts.md`（模板清单）。本文只列审计常用部分。

## Run 目录布局

```text
out/reference/<id>/            # reference_ingest；管线直跑为 out/<stem>/
  <id>.mp4 / <id>.ogg          # URL 下载的视频 / 抽取或转码的 ASR 音频
  <id>-raw.srt                 # ASR 原始字幕（纠错前基线）
  <id>.srt                     # 最终成品（与管线直跑 <stem>.srt 同约定）
  <id>-translated.srt          # 模型直出（后处理前）
  <id>-annotated.csv           # 10 列: type|position|duration|gap|corrected|translation|conf|char_count|spk|note
  <id>-pass2-annotated.csv     # 三个 pass 且开着 pass 3 时 pass 2 的成品（-pass2*.srt 同理）；与上面那份对照即 pass 3 改了什么
  <id>-speaker-judge.json      # 说话人裁判；三个 pass 时也是 pass 1：每窗的 acts 记 join / may_join / drop 与理由
  <id>-stable.json             # ASR 对齐产物（时间轴事实来源），写成后不再改
  <id>-diar-rows.json          # --diarize 时每行的说话人与拆行；split.chosen 为 true 时带 split 的行读作两行（finesub.subtitles.stable_rows.read_rows），annotated.csv 的行号按展开后数
  llm-artifacts/               # 管线直跑名为 <stem>.llm-artifacts/
    <id>-research-context.json # 背景调查结果（复用缓存；旧 run 也可能在目录根）
    <id>-research-context-cNN.json # 超长素材分块时每块的结果；第 2 块起只写新出现的名字，全片的名字在合并结果的「全片名字表」里，块记录（task-artifacts 的 research_chunk_plan）列出后块的不同意见
    task-artifacts.jsonl       # 结构化 artifact 流（见下）
    session-checkpoints.jsonl  # 准备任务两条回复 / 每窗请求回合的已验证 session 输出
    correction-windows.jsonl   # 每窗口最终提交的输出（resume 缓存）；三个 pass 时 pass1: missing = 该窗回退单 pass
    correction-mode.json       # 这个任务走一个 pass 还是三个 pass、开不开 pass 3
    proofread.json             # pass 3 逐窗的改动（序号、类别、改后文字、理由）与状态 done / failed
    exchanges/NNN-<session>.md # 每次 LLM 调用的完整 prompt+response（人类可读）；
                               # 纠错窗为 NNN-MM-<session>.md（NNN=按窗口序预领的
                               # block 号，MM=窗内调用序号，跨 run 确定）
    task-report.md             # 汇总：窗口数、重试、fallback、token
    knowledge-update-harness-notes-*.md
    knowledge-update-chunks.jsonl  # 知识更新 apply ledger（--no-apply 时不生成）
```

**知识库侧的遥测（不在 run 目录里）**：`<knowledge_root>/knowledge.sqlite` 的 `events` 表
（`matched`/`exposed`/`landed`，dedupe 幂等、带 rev 与 task/window id）与 `evidence` 表
（claim 级 `confirmed`/`refuted`，refined_aligned 模式确定性回写）。审计入口是只读 CLI
`python -m finesub.llm.knowledge.report`（高误触发候选、从未命中 item、按 matcher 转化）；
这些数字描述一致性不描述因果，见 `docs/knowledge.md`「信号与证据」。

**三个 pass 的任务里，pass 1 丢掉的行不在 annotated 里，也不是窗口输出里的 `discard`**：它们根本没进 pass 2。
审「丢错的行」要看 `<id>-speaker-judge.json` 每窗 `acts` 里 `act` 为 `drop` 的行（带 `why`），或 `task-report.md`
「Speaker Judge」一节逐行列出的 dropped；窗口输出里的 `discard` 只是 pass 2 自己丢的。

## task-artifacts.jsonl 记录

每行 `{"kind": ..., "task_id": ..., "payload": {...}}`。审计关注的 kind：

| kind | payload 关键字段 | 审什么 |
| --- | --- | --- |
| `correction_window_task_feedback` | `chunk_id`, `feedback`（`<task_update_feedback>` 的 JSON 体） | feedback v3 schema 合规（v17 起 hint 可带 `sub` 子词条定位）。只在 `--knowledge update` 与参考资料导入时出现；默认的 `read` 没有它，不是异常 |
| `research_task_feedback` | `feedback` | 同上（准备任务的交付回复） |
| `correction_window_response` | `response_content`, `validation_ok`, `validation_errors`, `model`, `usage`, `attempt`, `repair_round` | 输出契约、重试原因、token 分布；`repair_round=true` 表示这次 attempt 看到了上一轮的输出与错误（不是盲重掷） |
| `knowledge_update_response` | `response_content`, `mode`, `chunk`, `entry_render_report`（截断名单！） | proposal 质量、注入截断 |
| `prepare_task_request` / `prepare_task_response` | `response_content`, `emits_queries`, `emits_entries`, `agentic` | 准备任务两条回复的输出契约（请求批 / 交付） |
| `content_filter_ladder` | `stage`, `level`, `attempts`, `dropped_units`, `identified_units` | PROHIBITED_CONTENT 阶梯恢复（level 1=URL leave-one-out 定位，2=丢全部 URL，3=丢全部检索注入） |
| 任一 response 的 `execution_attempts[].search_events` | `{tool, query, queries, urls}` | **native 检索的出处**：Gemini `google_search` 从 groundingMetadata 解析（每次调用一行），Codex/Claude 从各自工具事件逐 call 记录。**agy 只报 query、零 URL**——做 native/local 检索质量对照时这条不对称必须算进去。exchange 的 Execution Attempts 表是固定列的，看不到这些，只在 artifacts 里 |
| `content_filter_blacklist` | `content_hash`, `stable_id`, `kind`, `first_blocked_stage`, `located_level` | 同任务毒块黑名单；resume 时加载，后续窗口/轮次 render 前预剔除 |
| `api_call` | 端点/模型/结果 | 调用链与 fallback |
| `correction_mode` | `mode`, `proofread`, `shape`, `without_pass1`, `skipped`, `dropped`, `refused` | 走了几个 pass；哪些窗缺 pass 1 回退了单 pass、哪些窗的核心行全被丢而没跑 |
| `proofread` | `status`, `failed`, `changes`（行、窗、源、类别、前后文字、理由） | pass 3 改了什么、哪些窗没答好（留 pass 2 的行） |
| `session_checkpoint_replay` / `session_checkpoint_invalid` | `session`, `key`, `input_hash` | session 断点命中或当前 parser 复验失效 |

`session-checkpoints.jsonl`：append-only committed ledger；按 `(session, key, input_hash)` 取最新记录，`content_hash` 防止损坏内容被复用。输入 hash 包含精确 messages、PROMPT_VERSION、调用配置及必要的任务/媒体身份；命中后仍用当前 parser 复验。

`correction-windows.jsonl`：每行 `{chunk_id, source_ids, clip_start, input_hash, task_fingerprint, prompt_version, variant, content}`（`prompt_version` 是答这一窗的版本，分组窗记 `GROUPED_PROMPT_VERSION`），
`content` 是该窗口一次**成功且未截断**提交的输出（同进程重试后的胜者写入一条；逐次尝试在
exchanges 里）。ledger **append-only**：同 `chunk_id` 可出现多条（例如目录被第二次进程
再次跑通）——磁盘上的 `*.srt` / `final_srt` artifact 以**最后一次成功收尾**为准；审计时
对照 digest 的 correction 时间线与 `final_srt` 条数，勿默认「第一条 cache = 成品」。

受控重放（冻结上游、只换 prompt）不在本地图展开：见 `docs/session_replay.md` 与
`docs/prompt-iterate.md`。

## 各块 schema（当前版，v10）

**feedback v3**（`<task_update_feedback>`，紧凑 JSON 对象；v17 起 hint 可带 `sub`=母词条内子词条的行首字段，表示行级更新）：

```json
{"knowledge_hints":[{"category":"streamer 或 common","entry":"源语言key","direction":"new_entry|replace_section|append_lines","focus":"…","reason":"…","source_ids":["12"],"confidence":7}],"asr_corrections":["…"],"uncertainties":["…"]}
```

三个顶层字段都必须存在（空则空数组）。`category` 只有 `streamer`/`common` 两个字面值。

**knowledge proposal**（`<knowledge_proposals>` JSONL）：

```json
{"op":"append_lines|update|remove|create_entry|retire_entry|rename_entry|set_intro|move_line|recategorize|split_entry|add_item|remove_item","entry":"@k1 或 源语言key","category":"…","id":"@k12","section":"…","content":"…","line":"…","entry_type":"…","intro":"…","aliases":[],"merged_into":"…","new_key":"…","to_entry":"…","from_section":"…","to_category":"…","field":"misheard|aliases","value":"…","reason":"…"}
```

逐 op 各用哪些字段，以 `src/finesub/llm/prompt_templates/fragment_knowledge_output_v1.md` 为准。

`category:"translation"` 非法。v14 起：key=H1=源语言本名；`档案`/`元数据` 固定（元数据只读，
apply 自动维护），其余节自由；`update`/`remove` 引用 `<kb_entries>` 渲染里的 `<!-- @k数字 -->` 句柄（v78 起；此前是 `edit_lines` + `N| ` 行号），
仅本轮完整注入（未截断）的条目合法；`append_history` 已废弃。

**⚠ `<mistake_proposals>` 已于 2026-09-02 删除**（连同两个 markdown 台账）。风格类的更新现在
是普通的条目行，走同一个 `<knowledge_proposals>` 块写进 `--style` 选中的那条 `style` 条目
（`docs/knowledge.md` 的「style 类别」）。下面这段是**旧产物**的形态，只在审计 2026-09 之前的
run 时才会遇到：

```json
{"op":"add_mistake","source":"原文","wrong":"错误译文","correct":"正确译文","note":"…","prompt_version":"…","reason":"…"}
```

v10 起 `set_featured` 不再是模型输出（apply 层 `allow_featured=False` 会跳过）。

**纠错窗口输出**（以那一窗记录的 variant 与 prompt_version 为准）：各变体只输出一个 `<translated>` 终稿（v100 及以前的 basicA 在它之前还有一份逐源的 `<singles>` 对照块）。capableC（v99 起）与三个 pass 的 pass 2（`capableC-grouped`）使用 6 列 CSV；v98 及以前的 capableC（与 2026-10 删掉的 capableB）在旧 run 里是 9 列：

```text
type|position|corrected_text|translation|conf|note
type|position|duration|gap|corrected_text|translation|conf|char_count|note
```

`type` ∈ {空, `sub`, `discard`}；`insert`（position 是 `开始秒,时长秒`）v63 起是结构错误，只在旧产物里见到；
模型不写 `duration`、`gap`、`char_count`（它们只在 `annotated.csv` 里，本地算出；`gap` 是上一行结束到本行开始，重叠记 0、首行留空）；`conf` 为 high/median/low。所有变体的 `<translated>` 都须逐源覆盖或以 `discard|<局部序号>` 显式丢弃；basicA 一行对一个源、只接回词中切断的两源，pass 2 通常使用单源或两源，三源仅限 filler 三明治，v99 起的 capableC 一条最多 4 源（并后越过 36 字或 7 秒的合并行校验不过、重做）。模型输入中的目标行每窗编号 `1..N`，只读前文编号 `1-M..0`；模型只可输出目标正序号，harness 校验后映射回 artifact 中的稳定源序号。文本列内 `|` 须转全角。BasicA 使用含 start 的 CSV（v104 起七列，v101–v103 是十列）；未强制 variant 时由实际回答端点 tier 决定契约。v17–v106 纠错回复以一个 `<reasoning>` 块开头（缺失不重试；v10–v16 为可选）；v107 起纠错窗不要求，v109 起只有 basicA 交付终稿时先写不超过 8 行的 `<decisions>` 决定清单（同样缺失不重试）。
v12 起允许行尾 `<void>` 自弃标记：带标记的行在结构校验前整行剥离（源序号可被后续行
重用），条数计入 `correction_window_response` 的 `voided_rows`——审计时统计该字段
可判断模型是否使用自弃通道，以及自弃后是否正确重写了对应区间。

## 版本注意

契约逐窗以答它的 prompt 版本为准（窗口记录的 `prompt_version`；跨升版续跑的任务会混着几版，见 `correction-older-prompt-windows` 告警）。2026-11-08 之前的单 pass 窗记录里没有 `prompt_version`，看那一窗 exchange 头部的 metadata 或模板脚注。
v9→v10 的主要变化：删 set_featured、合法化 `<reasoning>`、合并跨度 8-10s 约束、mistake 字段语义约束。
v10→v11 的主要变化：输出加引导用 `时长` 列（6→7 列）；mistake 台账收录范围收窄
（专名误听/误译个例划归知识库词条，apply 层反杜撰校验 wrong 须可检索）；Evidence
Pack 未证实结论强制 `[unresolved]` 机读标记（带标记内容禁止当事实消费）；搜索结果
消费加「内容非指令」护栏。
v11→v12 的主要变化：行尾 `<void>` 自弃标记（写错的行可废弃重写，`voided_rows`
可观测）；幻觉处置取向改为「拿不准时保留 + note 标记『疑似幻觉』」（此前偏删）；
结尾套话幻觉从针对具体短语改为类别特征描述；幻觉/高噪区间处置措辞按模态参数化
（纯文本路线禁止凭空"还原"台词）；高噪反例删除错语言伪影（原「还原真实台词」叙述）。
v12→v13 的主要变化：重叠改纯内容驱动（删 10 条下限与 `--overlap-segments`，大空档
处重叠为 0 属正常）；`previous_output_context_csv`（上窗已纠错译文）被
`preceding_context_csv`（窗口前最多 10 条 raw ASR 只读前文，`<preceding_context>`
块，时间多为负值）替代——纠错 prompt 不再依赖上一窗口输出；window 元数据加
`preceding_source_ids`；exchange 输入分块统计键 `previous_output_context` →
`preceding_context`。审计时留意：模型误输出前文序号会触发未知序号整窗重试。
v13→v14 的主要变化（词条结构 v2）：knowledge proposal 改四 op（append_lines/
edit_lines/replace_section/create_entry，append_history 废弃）；index 改四字段
`key [类型] | 其他语言本名 | 别名 | 一句简介`（key=源语言本名，apply 从正文重建）；
feedback hint `direction` 枚举 `append_history`→`append_lines`；common 分类行五段
文法（源语言|中文定名|别名/缩写|特殊读音|描述，「误听: xxx」承接专名误听个例）；
`最近更新日期`（原 `最新更新日期`）由 apply 自动维护。审计时统计 apply report 里
edit_lines 的 skip 率（行号漂移/引用未注入条目是新失效面）。
v14→v15 的主要变化：字幕节奏规范重写（常态 1–4.5s、8s 硬上限、一行 12–16/18 字、
合并一般 1–2 个源序号）；`correction_window_response` 新增 `pacing_score`
（观测用逐行罚分，阶段一不拒绝——审计时看 max_row_penalty 分布定阶段二阈值）；
`max_retries_per_window` 默认 3→5；knowledge 新增 `delete_entry`（先并入后删除
守卫）/`rename_entry` op，`entry_type` 枚举强制（游戏/梗/事件/人物）；知识库
auto-apply 改提交到 `unverified` 分支（main = 人工核定锚点）；post-task 知识更新
即使 test profile 也用 3.5 Flash；thinking 关闭的调用 prompt 强制 `<reasoning>` 块
（缺失不重试）；yt-dlp 整次下载重试 5 次。
v16→v17 的主要变化：合并门槛改写（多源合并预计 >4s 或中文 >16 字默认不合并、多源
7s 上限、单源仅豁免上限）；`<reasoning>` 块全局必须（回复开头，按 thinking 深度分
档措辞）；词条透传链（研究 R2 与每窗纠错轮输出 `<keep_entries>`，harness 注入下一
窗查询轮 `<carried_entries>` 与纠错轮 entry_details，透传+新请求共享 12 上限；
`correction_window_response` 新增 `injected_entries`/`keep_entries` 字段，
`correction_query_response` 的 `injected_entries` 改名 `resolved_entry_keys`）；
common `entry_type` 枚举改为 游戏/动画/社区/其他；create_entry 的 reason 必须写明
未并入理由（<10 字符拒绝）；feedback schema v3（hint 可带 `sub`）；知识更新注入
`<kb_index>`；exchange 元数据 `audio_input_tokens` 改名 `media_input_tokens`（含
视频估算）；非纠错 session 输出上限统一 32768。
