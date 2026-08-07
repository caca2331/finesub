# 知识库

存放公开网络中很少存在、难以进入 LLM 语料的知识（主播设定/经历、社区常用梗与人物、常见翻译错误台账），供背景调查、纠错窗口和知识更新流程注入。能直接在网络上简单搜到的知识，或 LLM 已知的大众知识，无需收集。

本文是知识库相关行为的唯一权威文档。相关文档：

- 纠错/调查/资料注入等运行时 harness 行为：[`llm_harness_behavior.md`](llm_harness_behavior.md)
- 架构意图与设计决策：[`llm_design_notes.md`](llm_design_notes.md)

涉及模块速查：

- `src/llm/knowledge/base.py` — index 加载、`match_index_keywords` 本地别名匹配。
- `src/llm/knowledge/update.py` — 统一知识更新入口 `run_knowledge_update`，CLI 为 `python -m llm.knowledge.update`。
- `src/llm/knowledge/materials.py` — 按窗口/全局分组构造统一知识更新的输入材料。
- `src/llm/knowledge/feedback.py` — `<task_update_feedback>` 的聚合与 hint 归并。
- `src/llm/knowledge/entries.py` — 知识条目预取（`<kb_entries>` 块）。
- `src/llm/knowledge/mistakes.py` — `translation/common-mistake.md` 的 apply 层。
- `src/asr_playground/workflows/reference_ingest.py` — 参考素材导入 workflow 入口。

## 知识库定位与结构

`knowledge/` 不再被主 git 追踪（`.gitignore` 中排除），目录内建立**独立 git 仓库**；仓库不存在时自动 `git init`。v15 起每次 auto-apply 提交到 **`unverified` 分支**（工作区常驻该分支）。auto-apply 开始时若 `unverified` 有 staged/unstaged/untracked/deletion，先将全部既存改动单独提交为 `[user-adjustment] snapshot before auto-apply`（带 `change-kind: user-adjustment` trailer），再生成正常的 harness commit，二者绝不混合；快照失败即中止。dirty tree 若位于 main/其他分支仍拒绝自动处理，避免污染可靠锚点。用户人工核定后手动 merge 回 main——main 是「可靠知识」的显式锚点，运行时读取的始终是工作区（unverified）状态。**git 缺失与失败一律降级为 warning，绝不失败整个任务**（2026-08-06）：字幕是主产物，知识库是副产物，副产物的问题不该抹掉主产物的成功。三条路径都**不推进 chunk ledger**，因此装好 git / 理清仓库后重跑会完整重做：①`git` 不在 PATH 上 —— `run_knowledge_update` 在 `execute and apply` 时**前置拦截**，返回 `skipped: "git_unavailable"`，一次 API 调用都不发；②仓库脏或在别的分支 —— 打 Warning 后停止应用（提案仍保留在产物里）；③文件已改动但 commit 失败 —— 打 Warning 并阻止 ledger 推进（否则 resume 会跳过这批永远未被记录的改动）。`_run_git` 捕获 `FileNotFoundError`/`OSError` 并返回 `returncode=127`，所以调用方现有的「非零即失败」处理原样生效。本项目不安装 git，所以「没有 git」是常态而非异常。

内嵌 git 是过渡方案，未来 pending to be replaced by 在线托管方案，以实现多用户间的知识库交换/同步。不再使用 `history/*.jsonl` 与 before/after hash；历史完全由 git 承担。主仓另有只读样板 [`examples/knowledge/`](../examples/knowledge/)（迷你 index + 示例条目 + translation 空骨架），不参与运行时默认路径。

**知识库根的解析顺序**（`asr_playground.paths.resolve_knowledge_root`）：显式
`--knowledge-root` → `FINESUB_KNOWLEDGE_ROOT` → **源码 checkout 的 `<repo>/knowledge`** →
发行包自身的 user-data → 本机受管安装的 user-data（`%LOCALAPPDATA%\FineSub\user-data`）。
绝不静默落到 CWD。

- **checkout 优先是默认值**：几乎所有从仓库发起的运行都是开发，而真实 API key 就在
  `<repo>/.env`；把它们导向共享知识库会让开发噪声与真实条目在提交流里交织，而分裂只需事后
  合并一次。`FINESUB_CHECKOUT_DATA=0` 显式退出。**git worktree 解析到主仓**（`.git` 文件里的
  `gitdir:` 上溯三段），且 worktree 内的 auto-apply 默认**跳过并告警**，除非
  `FINESUB_KNOWLEDGE_WRITE=1`。
- 桌面端与 CLI 壳正常都会注入 `FINESUB_KNOWLEDGE_ROOT`；后两档是给「绕过启动器、直接用包内
  解释器跑 pipeline」兜底：按模块所在的 `app/versions/<版本>` 布局反推安装根
  （`finesub_bootstrap.paths.packaged_app_root`）。发行包同样带 `pyproject.toml` +
  `src/asr_playground`，所以 checkout 探测**显式排除**这种布局——否则知识库会写进
  `app/versions/<版本>/knowledge`，而 `app/` 会被下次更新整体替换，数据静默消失。
- `.env`、`config.toml`、限流状态文件按同样顺序解析。三种终端用户形态（安装版/便携版/CLI）的
  user-data 现在是**同一个目录**，所以同一个用户只有一份知识库。

**跨进程写锁**：知识库 auto-apply 在 `<knowledge_root>.lock` 上排他（知识库目录的兄弟文件——
放在目录里会被 auto-commit 收编，锚在任何安装根下则三个前端锁的是三个不同文件）。等待超时的
行为与"仓库脏"一致：跳过应用、保留提案、不推进 ledger、只打 warning。

**迁移**（`finesub_bootstrap/migrations/`，按 id 记账于 `user-data/.migrations.json`，启动器与
命令行都会在读用户数据前跑一次，跨进程加锁，失败只告警并在下次重试）：
`0001` 把 `app/versions/*/knowledge` 搬回 `user-data/knowledge`，`0002` 把便携包内的整个
`user-data` 搬到 `%LOCALAPPDATA%`。两处都有时**不自动合并**，持续告警直到人工处理。

目录结构：

```text
knowledge/            # 主 git 不追踪；目录内独立 git
  streamer/
    index.md
    <key>.md        # 文件名 = 中文key
  common/
    index.md
    <key>.md        # 游戏/梗/事件/社区人物等
  translation/
    common-mistake.md # 常见翻译错误台账 + 精选（见下）
    good-example.md   # 翻译范例台账（出彩的精修翻译，见下）
```

index 每条目一行、四字段（v14）：`key [类型] | 其他语言本名 | 别名 | 一句简介`（`[类型]` 仅 common 使用，如 `[游戏]`、`[梗]`）。key = 条目 H1 = 文件名 = **源语言本名**；「其他语言本名」是正式名的中/英等写法；「别名」收窄为昵称、简称、全称、俗称（误听变体不进 index）。三列 + key 都参与备注关键词匹配与按名索取。整个 index 行由 apply 层从条目正文（H1、首行描述、档案节的 `本名:`/`别名:` 行）自动重建，proposal 不携带 index 字段；旧 3 字段行仍可解析（首次触及时重建为 4 字段）。

条目结构（v14，两类共通）：H1 → 一句话描述 → `## 档案`（固定格式：`本名: 源语言（读音）/ 中文 / 英文`、`别名: …` 等）→ 若干自由二级节 → `## 元数据`（永远最后，apply 层自动维护 `最近更新日期`，模型不可修改）。新条目由 `create_entry` 按预设骨架建档（`src/llm/prompt_templates/entry_preset_{streamer,common}_v1.md`，空节保留——节名即收集清单）：

- streamer 预设节：`直播内容`、`说话风格`（自称/口癖/语体）、`喜好 / 特点`、`重要经历`（每行 `日期: 事件`，绝对日期升序，无月份子标题）、`人际关系`（每行 `对象名 | 自然语言关系描述，含互称`）。
- common 分类节由模型自由分组命名（持续更新的游戏建议按大版本开二级节，节内可用 `###` 三级目录细分人名/地名/其他专有名词；预设骨架仅含 档案+元数据）；分类行五段文法 `源语言|中文定名|别名/缩写（全称）|特殊读音|一句话描述`，描述内可带「误听: xxx」承接专名误听个例。碎知识（只有一两行可说、附属既有主题的专名/梗）作为行进母条目分类节，不建独立文件。
- 尺寸纪律：单条目软上限 ~3.5k token（注入 4k 截断留余量）；分类养肥后拆子条目（如 `原神·角色`），拆分由人工确认。

翻译风格不是知识类别：通用风格常驻纠错模板；特殊风格通过 `--extra-style` 在固定位置注入（管理机制未来实现，与翻译范例注入合并设计，见遗留项）。prompt/harness 模板在 `src/llm/prompt_templates/`，由主 git 追踪。

## 常见翻译错误库（translation/common-mistake）

`knowledge/translation/common-mistake.md` 记录实践中碰到的典型翻译错误（尤其来自用户指出/用户提供精修中文 SRT），不走 streamer/common 的条目文件模型，而是单文件台账：

- `## 条目`：每条错误一个 `### M0001` 区块（id 由 apply 层递增分配，永不复用），字段为原文片段、错误译文、正确译文、说明（上下文与注意点）、`prompt_version`（产生错误时的 prompt 版本）、记录时间。
- `## 精选`：至多 30 条 id 的列表，由**人工维护**（或未来的专门维护任务）；每次纠错运行启动时渲染为固定段落注入纠错 system prompt（`$common_mistakes_block`），让模型避开同类错误。全文（≤20k token）另作为 prompt/harness 迭代输入；**不再**注入任务后知识更新 prompt（跨任务台账查重/维护留给独立维护模块）。
- 条目新增只发生在统一知识更新的**精修对照模式**（见下）：模型输出独立的 `<mistake_proposals>` JSONL 块（与 `<knowledge_proposals>` 并列），schema：

```json
{"op":"add_mistake","source":"原文片段","wrong":"错误译文","correct":"正确译文","note":"简要上下文与注意点","prompt_version":"产生错误时的 prompt 版本","reason":"证据来源"}
```

  收录对象是**成类的翻译/表达问题**（句式处理、字幕长度失控、风格抹平、翻译退化等换个素材还会再犯的模式）与 ASR 非误听类错误且未被纠错修正的；**具体专有名词（人名/地名/术语）的误听/误译个例不收**——那是知识库词条的职责（未来的误读/误判子词条，见遗留项），纯误听走 feedback 的 `asr_corrections`。在此范围内门槛从宽（注入侧有精选 30 条 + 全文 20k token 双上限兜底）。apply 层做反杜撰校验：`add_mistake` 的 `wrong` 必须能在该任务的窗口材料文本中检索到（忽略空白），检索不到跳过记 report——已审计的 run 全部出现过模型杜撰 wrong。
- apply 层（`llm/knowledge/mistakes.py`）：`add_mistake` 按 `(source, wrong)` 精确去重、分配下一个 id；`set_featured` op 保留给人工/维护任务使用（整体替换精选，id 必须存在、≤30 条），统一知识更新以 `allow_featured=False` 调用——模型误输出的 `set_featured` 在 harness 层被跳过。变更走同一内嵌 git 自动 commit。`category:"translation"` 对 `<knowledge_proposals>` 仍然非法。
- **翻译范例库** `translation/good-example.md`（同一 `<mistake_proposals>` 块的 `add_example` op 维护）：收录精修中特别出彩的翻译（信达雅/梗与双关的示范性处理），字段为原文片段、精修译文、说明、记录时间，id 按 `G0001` 递增、按 `(source, translation)` 去重。判定原则写在精修模板里：精修有意译/润色的自由，差异≠错误——歪曲原意才进错误台账，出彩处进范例库。范例库**不**注入知识更新 prompt（跨任务查重留给独立维护模块）；当前只收集，如何消费（如注入纠错 prompt 作风格 few-shot）pending 未来实现。
- 只有精修对照模式（`refined_aligned`，见下）才维护此台账；无精修的 `artifacts_only` 模式 prompt 不含 `<mistake_proposals>` 说明，harness 解析时忽略该块且不调用 mistake apply（禁用点在 harness，不依赖模型自律）。

## 任务反馈采集（--knowledge）

**三态开关**（`--knowledge none|collect|update`，`llm.correction_translation` 与 `pipeline.py` 同名透传，默认 `none`）：

- `none`：不采集、不更新（默认）。
- `collect`：纠错各窗口与 research 末轮（round 2；fast mode 为 fused round 1）额外输出 `<task_update_feedback>` v3 JSON 块（`knowledge_hints`（category/entry/可选 sub/direction/focus/reason/source_ids/confidence 1-9；`entry` 永远填主词条，`sub` 为母词条内子词条的行首字段、表示行级更新）+ `asr_corrections` + `uncertainties`），harness 分别留存为 `correction_window_task_feedback` / `research_task_feedback` artifact。纠错窗口与 fast round 1 的模型输出使用本窗口 `1..N` 局部 `source_ids`，harness 在落盘前映射回稳定源序号；普通 research round 2 的多窗口反馈原本就使用稳定源序号。解析失败只告警、不重试；`category` 的枚举拼接误写（如 `streamer|game_lore`，纠错阶段刚输出完 `|` 分隔 CSV 时高发）取 `|` 前段救回而非整条丢弃。**该档位进入纠错 resume 的 task fingerprint**——切换 `--knowledge` 会让已缓存窗口失效重算。
- `update`：`collect` 的全部行为 + 任务结束后执行统一知识更新；配合 `--refined-srt` 走精修对照模式。

采集点与落盘 artifact：

| 采集点 | 触发条件 | artifact kind |
| --- | --- | --- |
| 每个纠错窗口 | `--knowledge collect/update` | `correction_window_task_feedback` |
| research 末轮（round 2；fast mode 为 fused round 1） | `--knowledge collect/update` | `research_task_feedback` |

两者都是 `<task_update_feedback>` 标签包裹的 v3 JSON：`knowledge_hints`（每条含 `category`/`entry`/可选 `sub`/`direction`/`focus`/`reason`/`source_ids`/`confidence` 1-9）+ `asr_corrections` + `uncertainties`。落盘 artifact 内的 `source_ids` 一律是稳定源序号，不暴露窗口局部编号；这些 artifact 是统一知识更新阶段读取的原始素材，聚合逻辑见下文「输入材料」。

## 统一知识更新（knowledge update）

更新走单一入口 `llm.knowledge.update`（旧 `task_auto` + `post_task` 双路径已删除），实现在 `llm/knowledge/update.py`（`run_knowledge_update`）。

**证据模式**（各一段独立 system prompt）：

- `artifacts_only`（无精修）：证据 = 按窗口分组的 raw/final CSV + context + feedback；写入标准从严（宁缺毋滥）。prompt **不含** `<mistake_proposals>` 说明。
- `refined_aligned`（`--refined-srt`）：精修行按窗口时间切成 `<refined_csv>`（`start|end|text`，harness 先按 start 重排；index 错乱/注释性重叠字幕因此可容忍），是最高优先级证据；可维护 mistake 台账。精修噪音（非音频注释、拆合行、时间偏移、与 final 不一一对应）由 prompt 明示，不做 harness 侧对齐健康检查，也不再生成/注入 `alignment-report.md`（`src/asr_playground/subtitles/alignment.py` 保留给人工使用）。

**输入材料**（`src/llm/knowledge/materials.py`）按 stitch 后实际归属分组，分两层：

- 每个纠错窗口一个窗口包：
  - `<context_slice>`：该窗口的背景调查 context（`general_context` + `window_contexts` 对应条目）。
  - `<feedback_slice>`：该窗口纠错调用产出的 `<task_update_feedback>`。
  - `<raw_csv>`：`源序号|开始|时长|gap|文本`（全局秒）——ASR 原始输入。
  - `<final_csv>`：10 列 `type|position|start|end|gap|corrected|translation|conf|char_count|note`；由 `*-annotated.csv` 按序号 1:1 overlay 后处理 final SRT 的时间与 translation，`corrected` 不 overlay。
  - 可选 `<refined_csv>`：仅 `refined_aligned` 模式，见上文。
- 全局块（跨所有窗口共享一份）：
  - `<general_context>`：背景调查的全局摘要。
  - `<research_feedback>`：research 末轮（或 fast round 1）的 `<task_update_feedback>`。
  - `<aggregated_feedback>`：所有窗口 feedback 的聚合摘要，聚合逻辑在 `llm/knowledge/feedback.py`。
  - `<kb_entries>`：由 feedback hints 频率排序 top 20 预取（research hints ×2 加权，别名归并；≤4k token/条、整块 ≤40k；前序块已更新的条目标注提示），预取逻辑在 `llm/knowledge/entries.py`。v14 起正文按物理行渲染 `N| ` 行号供 `edit_lines` 引用；被截断条目禁用 `edit_lines`/`replace_section`。
  - 不再注入 `<common_mistakes>` / `<good_examples>`（台账维护与跨任务查重留给独立维护模块）。

**分块与幂等**：三块 CSV 合计超 100k token 时按窗口边界顺序切块；组装后整块超 194k 输入硬限再按窗口对半拆（单窗口仍超限则报错）。每块调用 `general_capable` → 校验 `<knowledge_proposals>` / `<mistake_proposals>` JSONL 语法（失败则采样重试，默认共 2 次）→ 先向 `<artifact_dir>/knowledge-update-chunks.jsonl` 写 intent（材料/proposal hash + apply 前 git HEAD）→ knowledge 与 mistake 两类 apply 不单独提交 → 一次统一 git commit（commit message 带 proposal hash）→ 写 applied ledger → 下一块重新加载词条块。崩溃在 commit 后、applied ledger 前时，只有 HEAD 已变化且最新 commit 的 proposal hash 与 intent 相符才恢复为已提交；崩溃在写文件后、commit 前会留下 dirty tree，下一次自动更新拒绝继续，避免按旧行号重复 `edit_lines`。重跑时 fingerprint+hash 命中的块直接跳过（`--no-resume` 强制全量重跑）。

proposal schema（v14 起四种 op，v15 增至六种，每行一条）：

```json
{"category":"streamer|common","entry":"源语言key","op":"append_lines","section":"目标小节名","content":"一行或多行","reason":"…"}
{"category":"…","entry":"…","op":"edit_lines","edits":[{"action":"change|insert_after|remove","line":12,"content":"新行（remove 免填）"}],"reason":"…"}
{"category":"…","entry":"…","op":"replace_section","section":"…","content":"小节完整新全文","reason":"…"}
{"category":"…","entry":"…","op":"create_entry","entry_type":"游戏|动画|社区|其他（仅 common）","intro":"一句简介","aliases":["初始别名，可选"],"reason":"…"}
{"category":"…","entry":"…","op":"delete_entry","reason":"内容已并入哪个词条的哪个分类"}
{"category":"…","entry":"旧key","op":"rename_entry","new_key":"新源语言key","reason":"…"}
```

apply 层职责（`llm/knowledge/base.py`，分阶段）：先 `create_entry`（预设骨架建档），再按条目合并全部 `edit_lines` 对**冻结快照**降序执行（行号 = 该轮 `<kb_entries>` 渲染的 `N| ` 行号；仅本轮完整注入未截断的条目合法——`line_editable` 守卫由 update 层传入；第 1 行 H1 与元数据节只读），再按 proposal 顺序执行 `append_lines`（行首字段去重、缺节自动建在元数据之前）/`replace_section`，最后对每个触及条目刷新 `最近更新日期` 并从正文重建 index 行。「新」条目 key 经简繁归一（NFKC+casefold+opencc t2s）命中既有 key/本名/别名时**重定向**到既有条目而不建重复文件；对缺失条目的 `append_lines`/`replace_section` 会隐式按骨架建档；非法 proposal 跳过并记录 apply report。统一更新的 knowledge/mistake 变更每 chunk 合成一个 git commit；仓库不存在时自动 `git init`。`append_history` 已废弃（重要经历改行式日期）；其余 op 与 v17 的类型/reason/index 约束保持不变。post-task 更新调用固定走真实 `GENERAL_CAPABLE` 链（5 端点 fallback 链），test_profile 不降级。

局部检索匹配：背景调查 Round 1 与快速第 1 轮对用户备注（`extra_info`/note）做 key+alias 的 casefold 子串匹配（`llm/knowledge/base.py` 的 `match_index_keywords`，别名去重到条目、按出现频次排序，最多 8 条，1 字符词跳过），命中条目全文按预算渲染预注入；text 路线没有背景调查，v17 起预注入条目作为**首窗口的透传 seed**（不再恒注入每窗）。

词条透传链（v17；v18 起 prompt 明示词条 key = 条目 H1 / index 主 key）：research round 2 输出 `<keep_entries>`（≤8，选自其注入词条；持久化于 research-context.json 顶层 `keep_entries`）种子首窗口；每个纠错窗口的纠错轮再输出 `<keep_entries>`（选自本窗实际注入集合，canonical 化、截 8）决定链条续/断。透传词条全文注入下一窗查询轮的 `<carried_entries>`（勿重复请求）并自动进入其纠错轮 entry_details；查询轮新请求上限 8、与透传合计 ≤12（超出裁新请求，预算渲染透传优先完整）。透传 keys+正文 hash 进入逐窗 resume input_hash，cache 记录 `keep_entries`/`injected_entries` 供回放续链。search loop 各轮只读可见 research r1 请求的词条（persistent base，与 loop 自请求去重），无中继权。

使用示例：

```powershell
# 主任务内一条龙（采集 + 更新）
python -m llm.correction_translation out/input/input-stable.json `
  --audio data/input.wav -o out/input/input.srt --execute `
  --knowledge update

# 只采集反馈，事后独立更新（位置参数 = 标准 final SRT，其余路径按 stem 派生）
python -m llm.correction_translation ... --execute --knowledge collect
python -m llm.knowledge.update out/input/input.srt --execute

# 有精修 SRT（精修对照模式，额外维护 mistake 台账）
python -m llm.knowledge.update out/input/input.srt --execute --refined-srt data/manually-refined-subs/精修.srt

# 只看 prompt（不调模型）/ 只生成不写库
python -m llm.knowledge.update out/input/input.srt --prompt-dir out/ku-prompts
python -m llm.knowledge.update out/input/input.srt --execute --no-apply
```

**降级行为**：独立 CLI 找不到 feedback artifact 时警告并降级继续（词条块为空、证据只剩 CSV/context）；找不到窗口元数据时全部行落入单一 fallback 窗口。

**路径派生与覆盖**：`--stable-json`/`--annotated-csv`/`--research-context`/`--artifact-dir` 可覆盖派生路径（`reference_ingest` 因 stable 与 final SRT stem 不同而使用）。默认 `*-research-context.json` 在 artifact 目录下。

## 参考素材导入（reference_ingest）

`llm-reference-ingest` 接受一组任务，每个任务是竖线分隔的一行
`srt | media | note | preset | args`，端到端执行。任务来源二选一或并用：`--index <目录>`
读取 `<目录>/index.csv`（每行一个任务，`#` 注释与空行跳过），`--task "<行>"`（可重复）单条传入。

**字段**：

- `srt`（必填）：精修 SRT。批量模式下不含 `.srt` 后缀的裸名解析为 `<index目录>/<名>.srt`，否则按路径；单条模式必须是路径。
- `media`：视频/音频 URL（走 yt-dlp）或本地文件。批量模式裸名（无后缀）在 index 目录里 glob 同名文件，否则按路径/URL；本地媒体跳过下载。
- `note`：注入 research 的 `extra_info`（不能含 `|`）。
- `preset`：一组设置的命名捆绑（`PRESETS`），留空 = `mm-med`。内置：`mm-med`（mm/med + `test_profile`，全角色 gemini-3.5-flash-lite，便宜，适合知识/prompt 迭代）、`prod`（mm/med，真实模型）、`text`（text/med，真实模型）、`text-high`（text/high，真实模型）、`mm-low`（mm/low，真实模型）、`mm-high`（mm/high，真实模型）。
- `args`：像 CLI flag 一样解析并**覆盖 preset**（行内优先）：`--route/--level/--fast/--output-scale/--video/--model/--language/--gpu-budget-gb/--test-profile/--no-test-profile/--no-web-search`。mm/high 需要视频：本地视频作 media 直接用；URL media 会下载一份视频到 artifact 目录（默认 `out/reference/<id>/<id>.mp4`，优先 720p、并选最低 fps，因 LLM 只按 detail=low/0.25fps 采样），并从该视频抽取 `<id>.ogg` 给 pipeline；或用 `--video` 显式指定；三者皆无（且无 media）则报错。

处理步骤：

1. URL：URL→id 映射缓存于 `data/reference/url-map.json`；下载媒体放在本次 artifact 目录。普通 URL 下载/转换为 `<id>.ogg`（16 kHz mono Vorbis，已存在则跳过）；mm/high URL 下载 `<id>.mp4`，并从该视频抽取 `<id>.ogg` 给 pipeline。本地 media：直接使用，`<id>` 取文件名 stem。媒体守卫：视频的音频流远短于视频流（断流 resume 损坏后 merger 静默截断）会直接报错；已存在的 `<id>.ogg` 若明显短于视频会自动重抽——这是「存在即跳过」的唯一例外。
2. 完整 pipeline（人声分离 → VAD+ASR → raw SRT，`pipeline.run_pipeline` 函数直调，按阶段存在跳过）。
3. LLM 纠错翻译（`run_full_correction(knowledge="collect")`：research 或复用已有 `research-context.json` → 纠错窗口（各窗口输出 `<task_update_feedback>`）→ SRT 后处理；`out/reference/<id>/<id>.srt` 已存在则整步跳过，`*-translated.srt` 作为模型直出保留）。
4. **统一知识更新（refined_aligned 模式）**：`run_knowledge_update(refined_srt=精修SRT, stable_json=..., artifact_dir=...)`——精修行按窗口切成 `<refined_csv>` 注入，apply `<knowledge_proposals>` + `<mistake_proposals>`；块级 apply ledger 使重跑免重复写库。不再生成时间对齐报告。

注意：该工具**默认全执行**（下载、GPU pipeline、Gemini 配额、知识库写入），偏离 repo 的 `--execute` 惯例——用户主动发起即视为授权；`--dry-run` 只打印每个任务的解析后计划；`--no-apply` 照常跑完全流程但知识更新只生成不写库（proposals 留在 exchanges 供人工审阅，不写 chunk ledger）。`--model`/`--language`/`--gpu-budget-gb`/`--no-web-search` 是全局默认，可被行内 `args` 覆盖。

**执行模型（三 bin 流水线）**：任务跑在 `src/asr_playground/batch.py` 的通用三 bin 引擎上——下载（×2 并行）→
ASR（4/8/12/16GB 档分别 ×1/×2/×3/×4）→ LLM（×1，纠错 + 知识更新为一个不可拆单元）。后面任务的下载/ASR 与前面任务
的 LLM 重叠执行；**LLM bin 严格按任务顺序消费**（上游乱序完成也不打乱），保证批内知识累积顺序与
逐条串行完全一致（后一个任务的纠错能用上前一个任务刚提交的词条）。LLM 并发固定为 1：限流器为
进程内无锁状态、知识库 auto-apply 走内嵌 git，均不支持并发。**单任务失败被隔离**（记录 stage +
error，跳过其下游阶段，继续其余任务；结束时汇总、exit code 非 0），不再首败全停；失败任务可用
`--task` 单条重跑，靠各级存在跳过廉价续跑。多任务批（或显式 `--batch-id`）在
`out/batch/<batch-id>/batch-status.jsonl` 记录事件流，batch-id 默认 `reference-<时间戳>`。
执行前仍先整批校验精修 SRT 存在与行格式（投喂前失败要趁早，投喂后失败要隔离）。

```powershell
# 批量：目录内 index.csv，行如 `clipA|clipA|备注|prod|--level high`
llm-reference-ingest --index data/reference/batch1 --gpu-budget-gb 8
# 单条：
llm-reference-ingest --task "out/refined-ep12.srt|https://www.bilibili.com/video/BVxxxx|备注|mm-med|--language en"
```

## 遗留开放项（下一轮）

- **kb_entries 超限条目的 prompt 压缩**（pending feature）：`<kb_entries>` 预取每条 ≤4k token，超限条目被截断注入；对被截断条目做 `replace_section` 会静默覆盖模型没见过的小节尾部。方向是用某种 prompt 压缩让条目不再超限；压缩落地前该数据丢失风险存在（harness 知道截断名单——`RenderedBlock.truncated`——可作临时守卫）。
- mistake 台账 `## 精选` 的专门维护任务（当前为人工维护；`set_featured` op 保留在 apply 层供其使用）。
- **翻译风格注入的统一机制**（原「--extra-style 管理」与「范例库注入」两项合并——风味化翻译本质也是一种翻译风格）：`--extra-style` 特殊风格文本与 good-example 范例 few-shot 走同一套注入设计（固定位置、按主播/题材选取、预算上限、范例库的精选/清理策略）。范例库当前只收集与查重，**暂不注入 harness context**，细节想清楚后再实现。
- 语义级输出校验（harness 侧）：元话语关键词进字幕文本列、单条字幕时间跨度超阈值、长时段无 sub/insert 覆盖的空档统计告警——现有校验只查格式形状，三个 test run 证明「格式合规 ≠ 内容合规」。
- 用更长、多窗口素材复测 feedback schema 偏差与 `<reasoning>` 块合法化后的行为（2026-07 的三个 test run 每个只有 1 个窗口，样本太小）。
- **子词条拆分的自动化**：专名误听/误译个例已由 v14 分类行的「误听: xxx」标记承接；剩余问题是分类节超限时的子词条拆分（如 `原神·角色`）当前靠模型在 reason 里建议 + 人工执行，是否要 apply 层自动检测超限并提示，观察后再定。
- **PROHIBITED_CONTENT 误杀**：已落地 content filter 阶梯（`src/llm/content_filter.py`；调查/fast/loop/查询/纠错/知识更新均接入，任务级黑名单 resume 可见）。仍可能被源文本本身误杀——阶梯耗尽后的报错即该情形；积累样本后再看是否要做源文本侧降级。
