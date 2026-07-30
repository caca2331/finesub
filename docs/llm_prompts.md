# Prompt 体系：模板、fragment 与组装

本文是 prompt 层的现行参考：模板文件清单、`prompt_compose.py` 的组装规则与槽位选择表。运行时行为（注入内容、上限、重试）见 [`llm_harness_behavior.md`](llm_harness_behavior.md)；设计取舍见 [`llm_design_notes.md`](llm_design_notes.md)。

原则：

- **Prompt/harness 文本不硬编码在 Python 里**——全部模板在 `src/llm/prompt_templates/*.md`（主仓库跟踪），运行时经 `Template.safe_substitute` 加载填充；prompt 迭代不需要改代码。
- **reasoning 全局必须（v17）**：所有 system 模板经 `$reasoning_clause` 要求回复以 `<reasoning>` 块开头，措辞按 thinking 深度三档（`prompt_compose.REASONING_DEPTH_CLAUSES` + `reasoning_clause(depth)`；纠错/查询轮按 profile 经 `correction_reasoning_depth` 取档，research/loop/知识更新默认 medium）；缺块不校验不重试。
- **变体优先做整文件 fragment**：结构性差异（有无音频、有无检索）用整个 fragment 文件切换；短语级差异用参数（`$judgment_basis`、`$csv_time_note` 等，见 `prompt_compose._AUDIO_MODAL_PARAMS` / `_TEXT_MODAL_PARAMS`）。
- `PROMPT_VERSION`（当前 `zh-subtitle-correction-csv-v65`，定义于 `src/llm/prompt_compose.py`）：prompt 语义变化时递增。它进入纠错 resume 的 task fingerprint、research context 复用校验、mistake 台账的 `prompt_version` 字段，以及 **exchange 元数据头**——bump 会使旧 resume 缓存与已保存 research context 失效重算。v65 删除不再使用的 capableA/BasicC 与 JSONL 输出支线；所有单窗口模型调用把目标行重编号为 `1..N`，只读前文按时间顺序编号为 `1-M..0`，validator 后由 harness 映射回稳定源序号；oneshot 同步使用该局部编号。
- Prompt 中只用“拉丁字母、数字和标点计 0.5”简述字数规则，不展开控制字符等 Unicode 实现细节；运行时完整口径以 `asr_playground.subtitles.metrics.weighted_char_count` 及 `docs/llm_harness_behavior.md` 为准。
- **词条 key**：index 行首主 key = 条目 Markdown 文件一级标题（`# 源语言本名`）；`<requested_entries>` / `<keep_entries>` 每行写主 key 或别名即可（详见 `docs/knowledge.md`）。

## 模板清单（按用途分组）

```text
src/llm/prompt_templates/
├─ 纠错（骨架 + fragment 组装，见下节）
│  correction_main_v1.md                  # 纠错 system 骨架（全 preset + fast 共用）
│  correction_user_v2.md                  # 纠错 user 模板
│  fragment_corr_role_{audio,text,video}_v1.md   # 角色/剪辑说明（video 为 mm-high 增补段）
│  fragment_goals_correction_{audio,text}_v1.md  # 纠错目标（听音版/纯文本版）
│  fragment_goals_translation_v1.md       # 翻译与内容取舍（共享；口语颗粒三步判定：内容→保留/机械噪声→压缩/无残值→丢弃，见 llm_design_notes）
│  fragment_csv_input_v1.md               # CSV 输入格式（时间列措辞参数化）
│  fragment_output_contract_v1.md         # 输出契约：BasicA 为带 header、含 start 十列；动态条数只计算字幕行
│  fragment_output_contract_nosingles{,_reasoning}_v1.md # capableB/C：去 singles、带 header 九列；BasicB 复用并动态加 start
│  fragment_weighted_char_count_v1.md     # 字幕加权字数的简短共享说明（运行时公式见 src/asr_playground/subtitles/metrics.py）
│  fragment_hallucination_v1.md           # 幻觉与丢弃（套话特征、保守保留、丢弃取舍子句仅 audio；$hallucination_handling 分模态）
│  fragment_translated_common_v1.md       # translated 产出纪律（tier 无关：gap 方向、char_count 列纪律、列核对），恒定拼在合并策略片段之前
│  fragment_examples_merge_nosingles{,_reasoning}_v1.md # capableB/C 无 singles 的完整 43 行 oneshot；C 用 # 前置局部推理
│  fragment_merge_rules_basic_v1.md / fragment_examples_merge_basic_v1.md   # basicA 保守 1:1 策略（仅词中接回）与完整 43 行 oneshot
│  fragment_alignment_v1.md / fragment_advice_v1.md / fragment_keep_entries_v1.md   # keep_entries 透传规则（v18，全 preset；词条 key = H1）
│  fragment_window_overlap_v1.md           # 窗口策略：重叠（可为空）+ 只读前文块规则与人造示例（$preceding_audibility_note 分模态）
│  fragment_retrieval_injected_v1.md      # 注入检索消费引言（内嵌 $search_results_usage）
│  fragment_native_search_v1.md           # 内置搜索指引（仅 text-high）
│  fragment_effort_{low,deep}_v1.md       # 思考力度 prose（仅 text 路线）
│  fragment_user_reminders_{audio,text}_v1.md    # user 侧易忘要求
├─ 查询轮 / 快速模式
│  correction_query_v2.md / correction_query_user_v1.md
│  fast_round1_v1.md / fast_round1_user_v1.md
├─ 背景调查与搜索 loop
│  research_round1_v1.md / research_round1_user_v1.md
│  research_round2_v1.md / research_round2_user_v1.md
│  search_loop_v1.md / search_loop_user_v1.md
│  fragment_search_loop_continue_notice_v1.md  # 非末轮轮次提示（续搜 nudge，带 $remaining_rounds）
│  fragment_search_loop_final_notice_v1.md     # 末轮轮次提示（强制收尾）
│  fragment_search_queries_output_v1.md   # 单轮 query 输出规则（内嵌 $query_style）
│  fragment_search_contract_output_v1.md  # 多轮变体（Research Contract + 第 0 轮；内嵌 $query_style）
│  fragment_query_style_v1.md             # query 写法通则（自包含/语言选择/引导语，两个输出规则共用）
│  fragment_search_results_usage_v1.md    # 原始搜索结果消费规则
│  fragment_evidence_pack_usage_v1.md     # 多轮变体（Evidence Pack 消费）
├─ 任务反馈采集（--knowledge collect/update）
│  fragment_task_feedback_schema_v3.md    # feedback v3 JSON schema（共享；source_ids 规则按单窗口局部/多窗口稳定序号参数化）
│  correction_task_update_feedback_v2.md  # 纠错窗口采集要求
│  research_task_feedback_v1.md           # research 末轮 / fast round 1 采集要求
├─ 统一知识更新（docs/knowledge.md）
│  knowledge_update_artifacts_only_v1.md  # 无精修模式 system（不含 mistake 块）
│  knowledge_update_refined_v1.md         # 精修对照模式 system（+mistake；精选不归模型管）
│  knowledge_update_user_v1.md            # 共用 user 模板（窗口包 + 全局块）
│  fragment_knowledge_update_inputs_v1.md # 两模式共用的输入说明（refined_csv bullet 与库描述参数化）
│  fragment_knowledge_structure_v1.md     # 知识库定位/结构 v2/更新原则（共享；含行文法、吸收判据、迁移指令）
│  fragment_knowledge_output_v1.md        # <knowledge_proposals> 六 op schema（append_lines/edit_lines/replace_section/create_entry/delete_entry/rename_entry）
└─ entry_preset_{streamer,common}_v1.md   # 词条骨架预设（create_entry/隐式建档共用，非 prompt 注入）
```

## 组装器（`src/llm/prompt_compose.py`）

四个入口，按 `TranslationProfile` 选 fragment 并填参：

```python
compose_correction_system(profile, *, tier=CapabilityTier.CAPABLE, evidence_pack_mode=False,
                          extra_style="", common_mistakes_block="")
compose_correction_user(profile, *, general_context_json, window_context, entry_details,
                        previous_advice, pre_round_notes, search_results,
                        preceding_context_csv, current_asr_csv,
                        current_asr_row_count, tier=CapabilityTier.CAPABLE)
compose_correction_query_system(profile, *, search_queries_rules, max_entries=8, total_entries=12)
compose_fast_round1_system(profile, *, search_queries_rules, task_update_feedback_block="",
                           max_requested_entries=8, max_keep_entries=8, max_total_entries=12)
```

更高层的 `build_*_messages` 在 `src/llm/prompts.py`，负责把 harness 注入内容填进 user 槽位；其中 `current_asr_row_count` 由实际窗口片段数计算：basicA 用它锁定 singles 行数，B/C 变体用它重申 translated 必须完整覆盖本窗。纠错 query、纠错终稿和 fast round 1 的 `<asr_result>` 使用 `local_id|start|duration|gap|text`：目标行每个执行窗口重置为 `1..N`；终稿轮并列的 `<preceding_context>` 使用非正数，最近前文为 0。模型输出在 validator 后立即映射回稳定源序号，窗口拼接、时间轴、annotated CSV 与知识材料始终使用稳定源序号。

### 纠错 system 槽位选择表（`—` = 空槽塌缩）

| 骨架槽位 | text-low | text-med | text-high | mm-low | mm-med | mm-high |
| --- | --- | --- | --- | --- | --- | --- |
| `$role_block` | text | text | text | text | audio | audio+video 增补 |
| `$goals_correction_block` | text 版 | text 版 | text 版 | text 版 | audio 版 | audio 版 |
| `$retrieval_block` | — | — | native | injected | injected | injected |
| 插轴（rules/example/output 子句） | — | — | — | — | — | — |
| `$hallucination_block` 丢弃取舍子句 | — | — | — | — | 有 | 有 |
| 模态参数（`$judgment_basis`/`$csv_time_note`/`$paren_rule` 等） | 文本版 | 文本版 | 文本版 | 文本版 | 音频版 | 音频版 |
| `$effort_block` | low | deep | deep | — | — | — |
| `$window_block` / `$advice_block` / `$keep_block` / `$hallucination_block` 主体 / 翻译目标 | 全 preset 共有 | | | | | |

`$merge_block` / `$examples_block` 不随 preset 变，而由具名 variant 选择（tier 只选择默认 variant）：capableC 是 capable 档生产默认，去 singles、使用九列 CSV，并在决策点前置 reasoning 注释；basicB 是 basic 档生产默认（继承 capableB 合并 + 带 start 十列 CSV）。capableB 是无 reasoning 的去-singles 对照；basicA 是保守 1:1/singles 对照组。B/C 行为使用 20 字/4 秒硬门槛与 36 字/7 秒绝对门槛。

fast 模式的纠错轮（round 2）复用同一骨架，`evidence_pack_mode=True` 时检索消费 fragment 换成 Evidence Pack 变体。查询轮与 fast round 1 的音频/纯文本变体由 `compose_correction_query_system` / `compose_fast_round1_system` 内的参数字典切换（mm-low 查询轮无音频附件，但仍用 `lightweight_multimodal` 角色；search-loop judge 才是纯文本 `lightweight`）。

普通 research R1 与 fast R1 均按 `<analysis_notes>` → `<requested_entries>` → `<keep_entries>` → 搜索 contract/query 输出；request 只负责新加载并按重要性排序，keep 只负责保留本轮可见的预注入词条。两类 canonicalize 后各自最多 8 条、合计最多 12 条，harness keep-first 合并，超限时从 request 尾部丢弃。Search-loop user prompt 把上一调用的 request/keep 名单放在 `<knowledge_entries>` 前，并把实际执行的 contract/query/extract 快照紧邻放在 `<search_results>` 前。Raw query/URL section 分隔符分别是 `--- query: ... ---` 与 `--- 深度提取 url: ... ---`。

输出完整性约束集中在纠错契约：模型不得省略必需标签、header 或记录。capableB/C 要求精确九列 CSV header；BasicA/B 要求精确十列 header 与 `start`。validator 只接受当前窗口的正局部序号，0、负数和越界值都会整窗重试；通过后统一映射回稳定源序号。

## 迭代惯例

- 改模板 wording 不改语义：不 bump 版本；改输出契约/输入结构/职责边界：bump `PROMPT_VERSION` 并同步更新 `test_llm_prompt_compose.py` 的版本断言与相关 snapshot 断言。
- 用 `llm.correction_translation --prompt-dir` 输出真实任务的完整 prompt（常规窗口为 capable 档，另落 `correction-0001-basic-tier.txt` 首窗 basic 档变体）；冻结注入重打纠错 R2 用 [`session_replay.md`](session_replay.md)。
- prompt/harness 迭代由 `tools/session_replay` 受控重放驱动（见 [`tools/prompt-iterate.md`](tools/prompt-iterate.md)），知识更新不再输出 `<harness_notes>`。人工审阅 replay 产物后手动改模板，绝不自动应用。
- 合并策略：精修标定的软门槛与模型边界（thinking=0、错并代价）见 [`merge-calibration.md`](merge-calibration.md)；
  现行变体契约见 [`tools/prompt-iterate.md`](tools/prompt-iterate.md) §4。
- 设计过程草稿与过期实验日志在本地 `docs/archive/`（gitignore），不入库。
