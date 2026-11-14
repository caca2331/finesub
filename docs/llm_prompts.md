# Prompt 体系：模板、fragment 与组装

> **前向提示（2026-08-19）**：本文描述的是**今天**的形态。知识库若按
> [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§7 未做与闸门](llm_agent_tool_protocol.md#7-未做与闸门)（设计在维护者的本地归档里）改成「索引必读 + 词条自主
> query」，这里的预注入名额、`<keep_entries>` 透传链与那组注入上限会一并消失。**不要据此实施
> 新功能**，先确认那个改动落地没有。

本文是 prompt 层的现行参考：模板文件清单、`prompt_compose.py` 的组装规则与槽位选择表。运行时行为（注入内容、上限、重试）见 [`llm_harness_behavior.md`](llm_harness_behavior.md)；设计取舍见 [`llm_design_notes.md`](llm_design_notes.md)。

原则：

- **Prompt/harness 文本不硬编码在 Python 里**——全部模板在 `src/finesub/llm/prompt_templates/*.md`（主仓库跟踪），运行时经 `Template.safe_substitute` 加载填充；按模式变的字句在 `mode_texts_v1.toml`（见[只纠错](#只纠错)），Python 按条件挑选、拼接的短语与各纠错变体的字句（`[variant.<变体名>]`）在 `phrases_v1.toml`；两张表由 `prompt_tables.py` 读（`task_mode_texts`、`prompt_phrases`）；prompt 迭代不需要改代码。例外是**参数层**：按键（媒体模式、行的种类）取值的模块级 dict，每个值只填模板里的一个 `$槽位`，由 Python 按条件挑、由模板给上下文——`prompt_compose` 的 `_MEDIA_PARAMS`、`_TEXT_MEDIA_PARAMS`、`_VIDEO_PARAM_OVERRIDES`、`_INPUT_INVENTORY_MEDIA` 与说话人裁判的 `PHRASES`；值可以是一整句，但不能自带要别处配合的结构（多个槽位的拼接、跨值的引用），那样的写进模板（owner 2026-11-07）。`test/test_llm_prompt_literals.py` 钉着这条：渲染 prompt 的模块（名字带 `prompt` 的与 `shape_criteria.py`）里出现够一个分句长的中日文字面量就红，参数层之外一句也不留；参数层的 dict 在同一文件的 `_PARAMETER_LAYER` 里逐个点名，新加的 dict 不点名就照样受查。
- **没有会话要求写推理**（纠错窗 v107 起，其它会话 2026-11-08 起）：只交规定的块；`basicA` 交付终稿时先写 `<decisions>` 决定清单（v109：不超过 8 行结论——专名与称呼、识别不可靠的区间、要丢的行；重交被退回的输出时先列这一轮要改的问题和它违反的规则；只写结论、不写推导过程），缺块不校验不重试。`test_llm_prompt_compose.py` 扫全部模板钉着这条。理由与读数在 [`llm_design_notes.md`](llm_design_notes.md)「efficiency 档与 `basicA` 的决策记录」。
- **变体优先做整文件 fragment**：结构性差异（有无音频、有无检索）用整个 fragment 文件切换；短语级差异用参数（`$csv_time_note`、`$verify_basis` 等，见 `prompt_compose._MEDIA_PARAMS` / `_TEXT_MEDIA_PARAMS` / `_VIDEO_PARAM_OVERRIDES`）。
- **示例 builder**（v69 起，`src/finesub/llm/example_builder.py` + `prompt_templates/examples/*.md`）：素材只写场景/译文/教学点，输出行写 `type|position|corrected|translation|conf|note` 六格，行用稳定 `label` 寻址；输入的 gap 与 `{calc:span|dur|gap|cc:...}` 全部由 builder 计算，阈值经 `{thr:}` 指向 `finesub/llm/prompt_constants.py`（**阈值在仓库里只有这一处定义**），按窗口的长度比例由 `Thresholds.at` 算出，示例输入的行时长跟着比例缩放（见 [`llm_harness_behavior.md`「合并门槛与任务的长度比例」](llm_harness_behavior.md#合并门槛与任务的长度比例)）。列布局取自 `output_protocol.output_csv_header`（按变体带不带 `start`）。未解析的 `{ref:}`/`{thr:}`/`{calc:}` 是硬错误。**全部示例均由 builder 生成**：43 行主场景是单一素材，输出按 overlay 分组（`nosingles` / `basic` / `grouped` / `single`；变体读哪几组、按什么先后在 `example_builder.OVERLAY_GROUPS_FOR_VARIANT`：单 pass 的 capableC 先找 `single`、没有就用 `nosingles`），素材里不再有行间 `#` 注释（没有变体写它了）；块与块之间的文字只能写在 `<input>` 与它的普通 `<output>` 之间（渲染在输出块前面，反例的「错误输出：」就这样写），写在别处是硬错误；素材可用 `requires: media>=audio` 之类的开关谓词限定适用向量。三个 pass 的 pass 2 也用这批素材（`build_examples(grouped=True)`）：输入的 `turn` 列在 frontmatter `may:` 列出的行上写「可并」（示例并到上一行的每一行都要列进去，作废的合并也算，否则是硬错误；只因长度没并的也列，模型才要判长度），有 `grouped` 这一组 overlay 或 notes 的示例换用它，`grouped: no` 的示例在 pass 2 不出现（`grouped: only` 反过来）。`single_pass_skip:` 列出的变体在单 pass 时不渲染这个示例（pass 2 照常）：filler 三明治那个示例（30）写 `capableC`，它教「唯一允许的三源形态」，而单 pass 的 capableC 一条最多并 4 行，三明治只是其中一种。三份手写 examples fragment、`_add_start_to_example_outputs` 与「9 列→10 列」`replace` 链均已删除。配套 golden 快照在 `test/data/prompt_goldens/`（8 组合 × 2 变体：capableC、basicA；另有带 `turn` 列的一组、pass 2 的两份，以及云端任务带分组键的单 pass 与分组版各一份），`manifest.json` 记 `PROMPT_VERSION → {文件: sha256}`——同一版本下内容改了即测试失败，必须 bump。重生成：`pytest test/test_llm_prompt_goldens.py --regenerate-goldens`。
- **槽位注册表**（v68）：开关驱动的 fragment 选择集中在 `prompt_compose.CORRECTION_SLOTS`，一条记录 = `(谓词, fragment, requires)`，首个命中的规则生效。`requires` 是该 fragment 假定存在的注入块；`PromptContext.injected_blocks` 由开关向量推出，二者不符即抛 `PromptAssemblyError`。加档位只改表，不加 if。
- **三个 pass 的 pass 2 有自己的版本号** `GROUPED_PROMPT_VERSION`（当前 `zh-subtitle-correction-3pass-v20`，同在 `prompt_compose.py`；v3 加了 owner 的 D9：用户写得非常明确的风格要求优先、说得不那么肯定的与通用规则折中，pass 3 的 prompt 同一条；v4 去掉示例里一处越过 7 秒的「可并」，分组从不这样标；v5 写明 D9 不越过合并规则的绝对门槛（按任务放宽门槛后来由 v13 的字幕长度档位做）；v6 随单 pass 的 v96 一起改，另加一句输出不带输入的 `start`、`turn` 列；v7 随单 pass 的 v97 一起改；v8 随单 pass 的 v98 一起改；v9 去掉输出里的 `duration|gap|char_count` 三列与 `#` 注释（owner 2026-10-23）：六列 header，超一般长度的理由写进该行 note，加权字数的算法挪到长度规则旁边、模型自己估，校验数格子，见 [`llm_harness_behavior.md`「Prompt 信息」](llm_harness_behavior.md#prompt-信息)；v10 随单 pass 的 v100 一起改；v11 随单 pass 的 v103：带分组键的任务里 `turn` 列多 `同时`、`换回` 两个值的说明；v12 随单 pass 的 v104，另外风格冲突那条写明出厂默认让给长度与合并规则；v13 随单 pass 的 v106）：改它不作废单 pass 任务已交付的窗口，升 `PROMPT_VERSION` 也不作废三个 pass 的。它的 prompt 是 capableC 的「分组版」`capableC-grouped`（`prompt_variants.grouped_variant`，只有 capableC 有；quality、intermediate 档的格都是 capableC；合并规则、契约与 user 模板和单 pass 的 capableC 是同几份，参数洞填分组版的话，见下面[单 pass 的拼法](#单-pass-的拼法)；示例照 capableC 的出，按它自己的六列渲染，窗口记录的 `variant` 写 `capableC-grouped`，回放时认回它。名字里的 C 原本表示「带注释」，capable 一族去掉注释时名字没改（owner 2026-10-23），没有变体再写辅助列。不背向后兼容：记录与会话里写着 `capableC` 或 `capableC-grouped` 的旧九列回答按六列去认、通不过校验；生产的分组窗不受影响（输入哈希带 `GROUPED_PROMPT_VERSION`，升版就重做，不复验旧回答），`tools/session_replay` 每次都按现在的 prompt 重新调模型、只校验这次的回答，不复验录下的旧回答），由窗口自己说（`SubtitleWindow.grouped`，`grouping.grouped_windows` 设上），`compose_correction_system/user(grouped=True)` 组装：换成下面「三个 pass 的 pass 2」那组整份模板，`turn` 列三态，示例按上一条标「可并」。golden 另有 `*-grouped__capableC.txt`（audio 与 text 各一份，另有带分组键的 `audio-local-high-cloud-grouped__capableC.txt`），记在 manifest 的这个版本下。三个 pass 的 pass 1（说话人裁判：只判轨号、加形状一节、带音频三份 `speaker-judge-*.txt`，另有云端任务那份 `speaker-judge-cloud-shape-heard.txt`）与 pass 3（`proofread.txt`）也在同一个 golden 测试里，各记在自己的版本号（`speaker_judge/run.py`、`proofread/run.py` 的 `PROMPT_VERSION`）下：改了内容不升版本号，测试就红。
- **按传输追加的 fragment（不进 `PROMPT_VERSION`）**：`fragment_conversational_correction_effort_v1.md`
  只在 conversational 路的**纠错** task 上被追加到 protocol 文档末尾（`client._run_conversational_call`）。
  判据与 agent bootstrap 模板相同——它不改输出契约（列、格式、覆盖要求一概不动），只说这条路上
  某一列值得花多少力气，因此既不 bump 版本也不进 `input_hash`。想让某句话对**所有** backend 生效，
  那就是输出契约的改动，要进 `fragment_output_contract_*` 并 bump（2026-08-25 试过一次，见
  [`prompt-iterate.md`](prompt-iterate.md) [§5 失效模式知识](prompt-iterate.md#5-失效模式知识) 的 v78/v78b/v78c 三条）。
- `PROMPT_VERSION`（当前 `zh-subtitle-correction-csv-v114`，定义于 `src/finesub/llm/prompt_compose.py`）：prompt 语义变化时递增。它写入 planning/exchange/session 元数据与每条纠错窗口记录的 `prompt_version` 字段（分组窗记 `GROUPED_PROMPT_VERSION`）。**bump 不作废已提交的东西**：三层恢复身份里只有 L1 单次调用 checkpoint 按它精确匹配（未提交的那次调用重发）；L2 的 `WINDOW_INVALIDATION_INPUTS` 与 L3 的 `research_reuse_key` 都不含它，只记作审计（见 [`llm_harness_behavior.md`「纠错窗口中途 resume」](llm_harness_behavior.md#纠错窗口中途-resume默认开启)）。未完成的任务续跑时，旧版本答的窗照用、告警一次，只重做两种：渲染出的输入变了的窗（窗口身份哈希覆盖输入 CSV，所以改了列、`gap` 的含义或局部编号的 bump 照样重做全部窗口，只改指令措辞的才沿用），和当前校验器不再放行的窗。由此一条规矩：**输出列的含义一变就换一个 header 名**——回放逐字校验 header，旧窗因此重做；同名换了含义的列会被当成旧义照用。prompt 渲染的内容在同一版本下变了，golden 测试照样失败、照样要 bump：bump 管的是可追溯，不再是作废。只影响知识更新路径的 fragment（如 `fragment_knowledge_structure_v1.md`）不进纠错契约，改它不 bump。**准备任务的模板（`prepare_task_*` 与 `fragment_prepare_*`）同理不 bump**：它们只改准备阶段那条对话，纠错契约与 `<context_pack>` 的语义都不随它变；改动本身会因为 messages 变了而自然让 L1 checkpoint 落空，不需要连坐作废已提交的窗口。v65 删除不再使用的 capableA/BasicC 与 JSONL 输出支线；所有单窗口模型调用把目标行重编号为 `1..N`，只读前文按时间顺序编号为 `1-M..0`，validator 后由 harness 映射回稳定源序号；oneshot 同步使用该局部编号。v66 转到开关轴（`--media/--retrieval/--difficulty`）：撤回 text-med 的检索说明特判（它描述了一个永不到达的 `<search_results>`）、reasoning depth 规则化、`$verify_basis` 改为按轴拼接而非整句写死、difficulty 作为 prompt 上限封顶变体。v67 落地 §1.4 的轮结构拆分：研究轮两轮各自按轴组装（`research_round{1,2}_v2.md` 及其 user 模板取代 v1），depth 谓词改为「有无 context_pack」——`retrieval=native` 因此由 high 降 medium，只剩 `retrieval=none` 用 high；无查询轮的窗不再被要求输出 `<keep_entries>`（harness 强制全量透传）。v68 把纠错组装改为**声明式槽位注册表**（`prompt_compose.CORRECTION_SLOTS`）：每个开关驱动的槽位是一张「谓词 → fragment + `requires`」表，`requires` 声明该 fragment 假定会被注入的块，组装时对着 `PromptContext.injected_blocks` 校验——「prompt 提到不存在的输入」从此是组装期错误而非评审事项。同时 `media=video` 有了自己的参数层（不再是「audio 措辞 + 视频补充段」），role/goals 的输入清单与「严防背景污染」段改为按轴拼装（无检索就不写「背景资料」、无知识库就不写「知识库词条」、无音频或无可冲突材料就整段不出现）。v69 起 capableB/basicB 的小例改由 **builder 生成**（见下「示例 builder」）：派生数值与局部序号由机器算，注释里的阈值走 `{thr:}`——迁移过程中查出并修正了三处手算错误（一处 gap、一处 char_count、一处合并跨度）。v73：查询轮 user prompt 的知识输入段（双索引、`<carried_entries>`、剩余额度行）与词条请求规则同谓词——`--knowledge none` 或空库时整段撤除，不再对模型展示「（空）」索引（`fragment_query_{index,carried}_input_v1.md`）；同时 `fragment_merge_rules_nosingles_v1.md` 与 `fragment_output_contract_nosingles_reasoning_v1.md` 中的字面阈值改为 `${thr_*}` 占位符（渲染逐字节不变，纠错 prompt 本体与 v72 相同）。v74：fast round 1 同享该谓词——`--knowledge none` 或空库时，输入清单、背景条目、职责条目、`<requested_entries>`/`<keep_entries>` 输出块与 user 侧的双索引/预注入段整体撤除（背景/职责列表改由组装层按轴编号，`fragment_fast_{entry_blocks,knowledge_inputs}_v1.md`）；两个搜索规则 fragment 里「已注入的知识库条目…」句尾同谓词撤除（查询轮与 research r1 在 local+knowledge=none 下同受益）；knowledge 开启时全部渲染逐字节不变。v75：第四条开关轴 `continuity`（P7c）——`parallel` 下 advice 台账的输入段（`fragment_advice_input_v1.md` 化的 `<previous_advice>`）、`<next_advice>` 输出要求（system 的 advice 槽位、user reminders 第 5 条、feedback 附加块的锚定措辞）与 `<keep_entries>` 全体同谓词撤除；`serial`（含全部 golden 组合）渲染逐字节不变。 v76 将可见 `<reasoning>` 改为不随 thinking 档缩放的中性措辞；v77 增加“内部已推演时只写梳理后结论”的软要求。v78（2026-08-22，知识库真相源切换）：`<kb_entries>` 改为 prompt 投影并带 `@k` 句柄、`N| ` 行号渲染删除，知识提案 op 换 handle 形态（`docs/knowledge.md`）。v79（2026-08-26）：`<kb_entries>` 行内停渲染 membership 句柄 `@m`——现行 op 集合没有消费它的操作，只是逐行噪音 token。v80（2026-08-28，条目结构 §11.2）：term 行改三/四段（`表面形|中文|描述[|别名]`，读音列删除、别名列由 items 渲染），「常被误听」停止注入（仅匹配器消费），streamer preset 增「频道用语」节，知识结构 fragment 同步。v81（2026-08-28，kb-followups A1/A5）：term 行改**固定四列** `源语言|中文定名|别名|描述`（别名第三列恒在、desc 居末可含竖线、恰三列报错），ops 契约补行格式规范与 fewshot worked example（示例由守卫测试喂真实解析器钉住），repair 会话新增 `<candidate_verdicts>` 块。另：知识类 prompt 组装点全部走 `load_prompt_template(strict=True)`——占位符覆盖在**模板层**校验（`Template.get_identifiers()` 减传入键，非空即抛），不扫渲染产物（素材/条目内容里的 `$WORD` 合法），修复了 repair 曾把字面 `$reasoning_clause` 发给模型的静默失效。v82–v88（2026-09-07 起）是知识库类别可扩展与条目结构那一族的 bump（逐条见 `git log -G'csv-v8' -- src/finesub/llm/prompt_compose.py`）。v89（2026-09-30）：`retrieval=none|native` 有可读知识库时窗口也能请求词条，`<keep_entries>` 的槽与 user 侧提醒随之出现在这些格（判据换成 `task_graph.window_requests`）；local 与无知识库的格逐字节不变（纠错 golden 与 local 窗口的请求通道都是）。v90（2026-10-01，`--style auto` 补齐 A）：风格选择改成强制选（`fragment_prepare_style_selection_v1.md` 与新的 `fragment_style_*`），fast 窗口的请求通道在要选风格时改「不需要外部输入也不能直接交付」（`$no_request_clause`），只为选风格开的请求回合有自己的开场（`fragment_window_style_channel_v1.md`），agentic 窗口的交付要附选择；**没有风格目录的格逐字节不变**，纠错 golden 全部同 v89；bump 照常作废已提交的窗口与 checkpoint（设计如此），让带目录的旧请求回合不会按旧契约回放。v91（2026-10-01，知识库 followup 第二批）：只有搜索、没有词条通道的 batched 窗口，回填轮不再要求 `<keep_entries>`；分块准备时每块都带风格目录；知识更新与两个 repair 会话的 op 词表加 `set_intro`（`fragment_knowledge_output_v1.md`；共享冲突修复会话也列，只放行本轮有简介冲突的条目；同一模板还写明 H1 冲突用 `rename_entry` 答，只放行本轮有 H1 冲突的条目）；`move_line` 多一个 `to_entry`（跨条目挪行，合并条目时先挪再 `retire_entry`）；带精修字幕且钉了 style 的知识更新多一个 `<style_marks>` 交付块（`fragment_knowledge_style_marks_v1.md`，其余格逐字节不变）。纠错 golden 与 v90 逐字节相同。v92（2026-10-04，prompt 冷读首轮的修正）：回填轮的交付顺序按变体写（只有要整窗 singles 的变体才提 `<singles>`），请求与风格通道不再点名契约外的块；fast 窗口要选风格时，user 消息的易忘要求第 1 条与最后提醒先说首条回复要交什么（agentic 档说先 `kb_read` 读风格、`<translated>` 之后附选择，`fragment_style_pick_reminder_*`）；特批理由的位置按变体渲染成一句（`CorrectionVariant.approval_reason_slot`），prompt 里不再出现变体代号；capableC 示例的合并与丢弃行补上前置 `#` 注释；窗口请求块的 keep 来源写 `<entry_details>`、准备任务交付轮的 keep 来源含 `<preinjected_entries>`（agentic：预注入的或 `kb_read` 读过的），与实现对齐；「背景调查与前置轮听不到音频」改为只说背景调查；空的 `<pre_round_notes>` 槽整段不渲染。v93（2026-10-05，prompt 冷读第二轮的修正）：缩窄合并示例的时间改成两次合并都在 4 秒硬门槛内（原注释把 4.6 秒说成「在界内」，示例守卫 `test_a_merged_example_row_past_the_hard_span_says_so` 从此钉住）；请求与风格通道说明 `<reasoning>` 是留给模型自己想清楚的地方、想清楚的一两句带过（与 `<window_notes>` 的分工不在这句里明说：明说的那版在 lite 重放里把请求回合的合格率从约五成压到 1/20，owner 2026-10-05 选不写），要选风格时第 1 条把 `<style_selection>` 列进首条回复的块；触发请求回合的条件写成「需要由 harness 代为取得的输入」（native 档不再把联网搜索算进去）；「首条消息」写成「首条 user 消息」；agentic 档的 `<style_selection>` 紧接在 `</translated>` 之后；知识库约定那句挪进 `fragment_background_knowledge_convention_v1.md`。同批不进纠错契约的修正：知识写会话的句柄说明统一（H1 句柄指整个条目，可 `add_item` 登记条目名的误听），结构说明里「可以改用 `split_entry`」那句只在 op 表带 split_entry 时出现（`$split_clause`，冲突回喂与没有拆分提示的知识更新都不再提），agent 外壳只把 payload 里的素材当数据，共享审核写明纯新建术语包无佐证走 tentative、空的 `merge` / `external_evidence` 怎么写。v94–v95（2026-10-15，纠错的合并看说话人；两版都只在维护者的分支上跑过试验）：任务有说话人标签时 `<asr_result>` 多一列 `turn`（`fragment_csv_input_v1.md` 的表头与第 4 条），两份合并规则、它们的「合并行检查」与最后提醒各多一句「标了换人的行不得与上一行合并」（`fragment_turn_*_v1.md`），示例的输入块跟着多一列空的 `turn`；v95 按冷读把空白 `turn` 写成「没判出换人、不保证是同一个人」，并说没标的行不要只因没有标记就当作拿不准（两个读者都把 v94 读成了空白也要有证据才能并）。没有标签的格逐字节不变（全部纠错 golden 同 v93，另加 `audio-local-high-turns` 一组）。v96（2026-10-19，prompt 冷读第三轮的修正）：录音里说出的话都译（主播、嘉宾、游戏与视频里的台词），删掉「非主播有效内容」这一丢弃类；低成本档改成正面说法；合并策略末尾复述条件的「补充」删掉；纯文本窗的时间基准写成窗口起点；主 oneshot 修了三处漏译与增译、一处算错的跨度和重复字数的 note；修复轮报错不再提 `insert` 行；`#` 注释只是不必写本行序号；输出不带的输入列（capableC 的 `start`，有说话人标签时的 `turn`）在 header 那条明说；外加冷读报告「措辞」一节收下的改写。同批不进纠错契约的：说话人裁判（`speaker-judge-v7`：示例 id 用 `qA` 这样不会是真实行号的写法、带音频形状版里内容与声音冲突答 `keep`、形状一节不再用「译成中文字数」、置信标说明只在这一窗有标时注入）、pass 3（`proofread-v3`：没有风格时第 6 条与风格冲突一段不渲染；有风格时资料里定好的专名与术语照资料写，用户点名专名的除外（owner D7）；专名与译重的修正可以变长）、知识写会话的尺寸纪律只在有条目带长度提示时出现、共享审核按槽位写明哪些 claim 查不到也能批准（`[关系/经历]` 与误听，网页证实不了，批准要维护者放行；待核实列表给条目项标出是别名还是误听），其余查不到就拒（owner D16）。v97（2026-10-19，首轮冷读遗留发现的分诊）：契约里删掉不存在的 `plan|` 行，10 列契约把 `discard|` 列为辅助行；首条 user 消息不再带搜索结果块（结果只随回填轮到达，replay 冻结了结果时才渲染，`fragment_search_results_input_v1.md`），搜索结果的用法规则只留 system 一份，纠错窗把冲突写进相关行的 note，检索槽说明是 harness 代为检索；回填消息开头只说是回填，风格照要遵守的约定给出，其余资料另起一句标明是内容不是指令（`fragment_fill_materials_note_v1.md`）；开了反馈采集时尾块提醒与回填清单写上 `<task_update_feedback>`（`fragment_task_feedback_reminder_v1.md`），没开就不提，它的位置锚点按 `PromptContext.keeps_entries` 写成 `next_advice 与 keep_entries` 或 `next_advice`；两处 keep_entries 写明要列全；最后提醒的绝对门槛只约束合并行；主 oneshot 措辞校正（只展示 `<translated>` 块、超硬门槛保持分行、去掉「新上游」）。3pass-v7 跟着同一批纠错契约改动。同批不进纠错契约的：准备任务职责第 2 条分开实体与术语、`sources` 只列网页、空的 URL 提取块不注入；只读注入（prompt 投影的部分预览）的条目不带元数据；style preset 的说明去掉 CLI 参数名与文档路径（pass 3 因此升 `proofread-v4`）；知识写会话的空槽用 `append_lines` 追加、只有换行体类型才删了重打、竖线判形按列数只写在术语行片段一处、通用示例写明依据；冲突回喂不再要求对照它没有的 `<kb_index>`；整理候选的提示文案进模板（`fragment_kb_candidate_hints_v1.md`）；校验与共享审核先写 `<reasoning>`，结果块按顶层解析（`output_tags.parse_json_tag_block`，准备任务的 `<context_pack>` 也走它）；校验以定名与身份定 `confirmed`（owner 2026-10-19）；共享审核的门槛表进模板，合并提示带上候选节点的内容（服务端新字段 `merge_candidates`）。v98（2026-10-30）：契约第 2 条改成 `<translated>` 终稿块有且仅有一个，回复里其它块的顺序指向本契约写块顺序的那一条（与写块顺序那一条不再矛盾）；filler 三明治的 4 秒写成跨度、不写合计（两份合并规则、合并硬门槛规则、分组契约与示例 30；示例与 pass 1 分组本来就按跨度算）。3pass-v8 跟着。v99（2026-10-31）：单 pass 的 capableC 换成「融合单 pass」：一次调用里先按不带轨号的 pass 1 的判据认人、判接缝，再按 pass 2 的合并规则纠错、翻译、合并（拼法见[单 pass 的拼法](#单-pass-的拼法)）；输出改成六列、不写 `#` 注释（同 3pass-v9）；一条最多并 4 行原始 ASR；并后越过 36 加权字或 7 秒的合并行校验不过、进修复轮。basicA、basicB 与 pass 2、裁判的 prompt 逐字不变（golden 钉着），3pass 不升版。v100（2026-10-31）：高噪声反例的「错误输出：」挪回错误输出块前面（以前 builder 把块间的文字并进了示例末尾，它渲染在错误输出之后、下面空一行）；带音频的纠错 golden 都只差这一处，3pass-v10 跟着。v101（2026-10-31）：`basicA` 改成一行对一行，不再先写一份逐源的 `<singles>`：只交一个 `<translated>`，契约与 user 模板与 basicB 同一份（十列带 `start`，user 模板第 3 条的合并提醒按变体写）；只接回词中切断，示例里教判断型合并、长度门槛、三明治的几条不再对它渲染，译文不留日语助词那条对它渲染；`<singles>` 的模板、校验与示例支持随之删掉。其余变体的 golden 逐字不变。v102（2026-10-31，`basicA` 的冷读）：词中接回的间隔写成数字（后一行的 gap ≤0.3 秒，csv-v111 起；`SPLICE_MAX_GAP_SECONDS`），合并规则、契约提醒与 user 侧复述用同一组四项条件；主 oneshot 的 basic 终稿去掉七条只在复述规则的 note。其余变体的 golden 逐字不变。v103（2026-11-02，云端任务里重叠的行）：行带分组键 `line_speaker`（云端转写写）的任务，`turn` 列多 `同时`、`换回` 两个值（`fragment_turn_{together,back}_v1.md` 接在列说明后面，格式行的那一格写出三个值），单 pass 的合并规则、合并行检查、最后提醒与融合单 pass 的认人线索写成「标了「换人」「同时」或「换回」」（`$turn_marks`），线索那句说标签是转写模型给的（`$turn_source`）；这些在带键任务的每一窗都渲染，一个任务一份系统 prompt（owner 2026-10-31）。没有键的格逐字节不变（全部纠错 golden 同 v102，另加 `audio-local-high-cloud__capableC.txt`）。这批改动在 `dev` 上先记成 v99 与 3pass-v9，与本表的 v99 不是同一版：旧记录里的 v99 要看日期与分支。同批不进纠错契约的：说话人裁判 `speaker-judge-v8`，讲轨号来源的措辞按行选（`speaker_judge_source_{acoustic,cloud}_*_v1.md`，本地任务逐字节不变）。v104（2026-11-03，实验分支 `exp/basica-noaux`）：`basicA` 的输出不写 `duration|gap|char_count`，七列带 `start`（契约、user 模板第 2 条、词中接回的行例与产出纪律跟着改，产出纪律换成 capable 一族不带辅助列的那份）；validator 在 basicA 的行多写这三列时点名报错。其余变体的 golden 逐字不变。v105（2026-11-04，出厂默认风格，与 v104 并行开发，合并时升版）：翻译目标里的口味要求（语感、不加注、语气词的取舍、意译尺度、语序、口语译法、圈内借词，原第 1、2、4 条的一部分与第 7–9 条）挪进出厂默认风格 `style_factory_default_v1.md`，通用规则只留机制，「哪些算内容、哪些算噪声」改成按翻译风格判断；选择定下来、没有用户的风格时注入它（纠错 golden 因此每份都带它，改它就要升版）。分组版的风格冲突那条写明用户的风格（`<style>` 里不叫「出厂默认」的那些与特殊翻译风格要求）写得明确的按风格来、出厂默认与长度、合并规则冲突时按规则来（owner O2）；`--extra-style` 排在风格前面、冲突时以它为准（`fragment_extra_style_v1.md`，owner O3）。3pass-v12 跟着。同批的 pass 3（`proofread-v6`）：第 6 条与冲突一段恒在（风格永不为空），出厂默认让给长度与读速，第 5 条留下风格要的口癖与语气词；没有风格时那句与另写的风格要求两份 proofread 专用模板删掉，另写的要求与纠错共用 `fragment_extra_style_v1.md`。v106（2026-11-04，字幕长度档位）：合并门槛（一般长度、绝对门槛、filler 三明治的字数与秒数）按任务的长度比例算（`prompt_constants.Thresholds`），窗口带着比例（`SubtitleWindow.length_scale`），prompt 写的数与校验拦的数是同一组；比例 1.0 时数字与 v105 相同。风格冲突那条（翻译目标第 8 条）收成只管合并的倾向，写明长短由风格的档位决定、`<style>` 里的长短要求不再另外放宽门槛；示例输入的每行时长跟着比例缩放（行间停顿不变），s19 的 gap 改成算出来的，两条合并行的译文缩短到偏短档的一般长度以内（s08,s09 与 filler 三明治那条，tail 的时长也短了 0.1 秒留余量）。3pass-v13 跟着。pass 1 的 prompt 三档通用、不变，只有脚本在 pass 2 之前的两道挡跟着比例。v107–v108（2026-11-05，实验分支 `exp/no-reasoning`）：纠错窗的各回合不再要求 `<reasoning>`：契约写块顺序那一条改成只管交付终稿的回复，最后提醒写「交付终稿时以 `<translated>` 开头」，请求与风格通道、回填轮、补丁修复轮去掉它；v108 是定向冷读之后的四处措辞（3pass-v14、v15 跟着）。v109（同日）：`basicA` 交付终稿时先写 `<decisions>` 决定清单（只在它自己的契约与 user 模板里，其余变体逐字不变）。v110（2026-11-06，prompt 冷读第四轮的修正）：最后提醒的丢弃说「高度疑似幻觉」（「疑似幻觉」是要保留的行的 note 标记）；翻译目标第 8 条按实际注入的风格渲染（只有出厂默认时只剩一句）；agentic 窗口的 keep 来源写成本任务里读过的词条，条目块说明改成预注入或前面窗口透传的，没有知识库时不注入条目块；出厂默认风格只留约定、正例、反例，`[圈内借词]` 写成保留中文写法、不留假名；没有知识库时检索规则不讲与知识库冲突；`basicA` 的回填轮交付先 `<decisions>`、词中接回的行例换了序号、补丁说明不再诱导它合并；短示例的终稿都带 header（错误输出反例除外）；外加冷读报告收下的措辞。3pass-v16 跟着。同批不进纠错契约的：说话人裁判 `speaker-judge-v9`（开头任务句带上定形状、要并的两行答成同一条轨）、pass 3 `proofread-v7`（不许改空、第 6 条的长度按风格冲突那段定）、准备任务、知识写会话与 agent 开场的修正（见冷读报告）。v111（2026-11-07，分支 `gap-direction`）：输入的 `gap` 改成上一行结束到本行开始的停顿（与上一行重叠记 0，首行留空），与同一行的 `turn` 说同一道接缝；规则与 note 不再教「与前一句的间隔看前一行的 gap」，判断本行是否并进上一行看本行的 gap，单 pass 与 pass 1 的形状判据合成一份；示例的输入与 note 跟着换方向。3pass-v17 跟着；说话人裁判 `speaker-judge-v10` 的 `gap` 钳到 0（不再给负数与 `-0.0`）；知识更新的 `<final_csv>` 不再带 `gap`。三臂 A/B（v110、v111、去掉 `gap`）里 v111 没有触发退化门槛，owner 2026-11-08 采纳。v112（2026-11-08，A/B 期间押后的冷读后续，A/B 没测）：分组版终稿删掉 s20 那条把不并写成停顿缘故的 note（它的 `turn` 本来就没标可并），四份短示例（30、40、50、60）的输入补上 header 行，现在每份示例的输入都以 header 开头，分组版对照要点不再说 s20 写了那条 note；3pass-v18 跟着；`speaker-judge-v11` 的输入说明补上「与上一行重叠时记 0」。同批知识更新的输入说明把 `<final_csv>` 的列名写成表头上的 `corrected`，决定与理由在 `docs/llm_design_notes.md`「纠错输入 `gap` 方向的决策记录」。v113（2026-11-08，分支 `prepare-context`）：窗口背景里有准备任务转述的用户要求时（通用背景的 `user_requests`，或专属背景写明「出自用户」），user 模板多一段怎么照做（`fragment_user_requests_v1.md`）：总体上按用户要求统一格式，写法与格式要求用在 `translation`（只纠错是 `polished`），`corrected_text` 照常只纠错；用户给的原文（例如歌词）只作依据，与录音里实际的不一致时内容照实际、格式照用户；上下文明显需要时可以变通；两种情况都在 note 写一句；有 `--extra-style` 的翻译任务再加一句风格冲突时听它。没有转述的要求时 user 模板与 v112 相同。分块调查的通用背景按窗口所在的块取，合并说明与点名本窗所在分段那句一起渲染。3pass-v19 跟着。同批不进纠错契约的：准备任务职责第 5 条转述用户要求、专属背景可以多窗共享；pass 3 `proofread-v8` 在任务有转述的 `user_requests` 时多一段用户要求（写法与格式管译文、不符合的按第 6 条改，纠错按上下文变通过、备注写明的不改回去）；知识更新的输入说明列名改成表头上的英文、补上 `conf` 与 `note` 的含义。v114（2026-11-09，同一分支的定向冷读）：用户要求那段不再说「统一纠错与翻译的格式」（与 `corrected_text` 只纠错打架），改成总体上照做、写法与格式要求只用在译文列；note 只在内容照实际写了或做了变通时写，照要求改写法本身不写。3pass-v20 跟着。同批不进纠错契约的：准备任务职责第 5 条写明只介绍背景的不算要求，分段接续说名字表会连同交给后续调用、`name_disputes` 也管类型；知识校验示例的 note 不再预设「此人」；共享审核的 reason 只在查了却没核实到时写查过的方向；知识写会话按 apply 的方式说被拒的后果（维护会话与共享冲突修复是整批回滚）。同批 pass 3 升 `proofread-v10`（`proofread-v9` 是同一天冷读前的中间版，只差用户要求那段的措辞）：每窗多一块准备任务给它的专属背景（`proofread_window_context_v1.md`），用户要求那段在专属背景写明「出自用户」时也出现、分「全片适用的」与「只关这一段的」两部分，并写明照要求改写的译文可以比原来长、读速超线也不按第 2 条改短。**任务式流程（2026-09-09，阶段 A–E 与删旧流程的阶段 D）没有 bump**：纠错 prompt 本体逐字节不变，窗口的请求通道由 `request_prompts.py` 叠在它之上，准备任务另有自己的模板；v67/v73/v74 提到的研究两轮、查询轮与 fast 融合轮模板已随阶段 D 删除，这里只作历史。
- Prompt 中只用“拉丁字母、数字和标点计 0.5”简述字数规则，不展开控制字符等 Unicode 实现细节；运行时完整口径以 `finesub.subtitles.metrics.weighted_char_count` 及 `docs/llm_harness_behavior.md` 为准。
- **词条 key**：index 行首主 key = 条目 Markdown 文件一级标题（`# 源语言本名`）；`<requested_entries>` / `<keep_entries>` 每行写主 key 或别名即可（详见 `docs/knowledge.md`）。

## 模板清单（按用途分组）

```text
src/finesub/llm/prompt_templates/
├─ 纠错（骨架 + fragment 组装，见下节）
│  correction_main_v1.md                  # 纠错 system 骨架（全部开关向量 + fast 共用）
│  fragment_corr_role_{audio,text,video}_v1.md   # 角色/剪辑说明（video 为 media=video 增补段）
│  fragment_goals_correction_{audio,text}_v1.md  # 纠错目标（听音版/纯文本版）
│  fragment_goals_translation_v1.md       # 翻译与内容取舍的机制（共享；口语颗粒的保留与压缩按翻译风格判，出厂默认的 `[语气词]` 是压缩版，完整判定见 llm_design_notes）
│  fragment_csv_input_v1.md               # CSV 输入格式（时间列措辞参数化）
│  style_factory_default_v1.md            # 出厂默认风格：只有约定、正例、反例三节（没有标题与档案），没有用户的风格时由 `knowledge/style.render_style_block` 注入（纠错与 pass 3），`knowledge new style` 抄这三节
│  fragment_style_no_match_factory_v1.md  # agentic 窗口选风格、目录里没有 default_style 时，选择提示末尾的出厂默认全文（只在 `selected` 填空字符串时适用）
│  fragment_style_conflict{,_factory}_v1.md # 翻译目标第 8 条（风格与合并规则冲突时）：按实际注入的风格渲染（`prompt_compose._style_conflict_rule`）；只有出厂默认、没有 `--extra-style` 时用 _factory 那一句
│  fragment_extra_style_v1.md             # `--extra-style` 的那段：排在风格前面、冲突时以它为准（纠错与 pass 3 共用）
│  correction_user_nosingles_v1.md        # basicA 的 user 模板（第 3 条合并提醒：`merge_habit_reminder`）
│  fragment_output_contract_nosingles_v1.md # basicA 的输出契约：一个 `<translated>`、带 header、含 start 七列（不写辅助列）
│  fragment_output_start_v1.md            # 输出契约里 `start` 列那条（只有输出带 `start` 的变体，即 basicA）
│  fragment_output_dropped_columns_v1.md  # header 那条后面的「输入里的 `start`、`turn` 列不抄进输出」（按变体、有无 `turn` 列拼；不写辅助列的变体还列 `duration`、`gap`）
│  fragment_weighted_char_count_v1.md     # 字幕加权字数的简短共享说明（运行时公式见 src/finesub/subtitles/metrics.py）
│  fragment_hallucination_v1.md           # 幻觉与丢弃（套话特征、保守保留、丢弃取舍子句仅 audio；$hallucination_handling 分模态）
│  fragment_translated_common_noaux_v1.md # translated 产出纪律（变体无关：输入 gap 的含义、列核对），拼在合并策略片段之前
│  fragment_merge_rules_basic_v1.md         # basicA 一行对一个源（仅词中接回）
│  examples/*.md                           # 所有 oneshot 与小例的素材（builder 生成，见下「示例 builder」）
│  fragment_alignment_v1.md / fragment_advice_v1.md / fragment_keep_entries_v1.md   # advice/keep 槽位（v75 起仅 continuity=serial 渲染；词条 key = H1）
│  fragment_keep_source_{injected,agentic}_v1.md   # keep 第 1 条「能透传哪些词条」：本窗口的 <entry_details>，或 agentic 窗口本任务里读过的（含 kb_read）
│  fragment_entry_details_input_v1.md / fragment_entry_details_label_agentic_v1.md   # user 消息里的「知识库条目详情」块（没有知识库时整块不注入）与 agentic 窗口的块说明
│  fragment_window_knowledge_index_v1.md   # 窗口请求通道里知识库索引的标题与 keep 的来源
│  fragment_advice_input_v1.md            # user 侧 <previous_advice> 输入段（同谓词）
│  fragment_next_advice_reminder_v1.md    # user 提醒第 5 条：<next_advice> 产出要求（仅 serial）
│  fragment_trailing_blocks_{serial,parallel}_v1.md  # 尾部块提醒两形态（serial 带 $keep_entries_reminder 与 $feedback_reminder 连接子；parallel 只在开了反馈采集时渲染）
│  fragment_window_overlap_v1.md           # 窗口策略：重叠（可为空）+ 只读前文块规则与人造示例（$preceding_audibility_note 分模态）
│  fragment_retrieval_injected_v1.md      # 注入检索消费引言（内嵌 $search_results_usage）
│  fragment_retrieval_self_v1.md          # agentic 窗口自己联网检索时的检索槽（内嵌 fragment_self_search_usage_v1 的规则）
│  fragment_native_search_v1.md           # 内置搜索指引（retrieval=native）
│  fragment_effort_{low,deep}_v1.md       # 思考力度 prose（retrieval≠local；minimum 取 low）
│  fragment_user_reminders_{audio,text}_v1.md    # user 侧易忘要求
│  fragment_turn_{cell,column,rule,check,reminder}_v1.md  # 有说话人标签时的 `turn` 列说明与「标了换人不并」各处的那一句（带分组键的任务里 `$turn_values` / `$turn_marks` 写出同时、换回）
│  fragment_turn_{together,back}_v1.md          # 带分组键的任务：列说明后面解释 `同时`、`换回` 的两句
│  fragment_turn_merge_note{,_grouped}_v1.md     # 跨换人（pass 2：跨没标可并）的合并，修复轮里顺带发回的那句说明（output_protocol）
├─ pass 2 的那组模板：三个 pass 的 pass 2（`grouped=True`）与单 pass 的 capableC 共用，几处参数洞按变体填（见「单 pass 的拼法」）
│  correction_user_grouped_v1.md          # user 模板：合并提醒是一般长度、绝对门槛；第 3 条开头（$mid_reminder_merge_rule）与最后提醒的合并范围（$merge_reminder）按变体填
│  fragment_output_contract_grouped_v1.md # 输出契约：六列、没有 `#` 注释；长度写成一般长度 + 绝对门槛，加权字数的算法写在这里、模型自己估，超一般长度的理由用一句短结论写进 note；三明治的更严长度（$sandwich_limit_clause，单 pass 没有）按变体填
│  fragment_translated_common_noaux_v1.md # translated 产出纪律的六列版：gap 读输入那一列，不提 char_count
│  fragment_merge_rules_grouped_v1.md     # 合并规则：五处参数洞（开头一段、第 1 条的源数、第 2 条、第 3 条的开头、第 4 条的接缝检查与 note）说合并能落在哪些接缝上；分组版填「行已由上一步分好、只并标着可并的行」，单 pass 填「你判成 join 的要并、may_join 的看情况」；看停顿与长度那两件共用
│  fragment_merge_{intro,scope,seams,check_seams}_{grouped,single}_v1.md  # 上面四处成段的参数洞，分组版与单 pass 各一份（变体的 `merge_*_fragment` 字段点名）；第 3 条开头那几个字是变体字段 `may_merge_lead`
│  fragment_turn_{cell,column}_grouped_v1.md     # pass 2 的 `turn` 列三态（换人、可并、空）
│  fragment_alignment_grouped_v1.md / fragment_goals_translation_grouped_v1.md   # 去掉「多数源保持独立」；翻译目标多一条风格与通用规则冲突时的优先与折中（D9；出厂默认让给长度与合并规则，O2）
│  fragment_hallucination_grouped_v1.md   # pass 2 的丢弃规则加上 pass 1 的保留清单（实义词、应答附和、重复行、时长或时间看着异常的行不丢）；单 pass 的 capableC 没有这一节，丢弃按 pass 1 的判据
├─ 单 pass 的 capableC 自己的（拼在上面那组之上）
│  fragment_shape_single_v1.md            # 「认人与定形状」一节：胶水（先认人、判接缝，再按合并规则做）+ pass 1 的判据（shape_criteria_*）；$voice_clause、$drop_listen 只在有音频时有内容；判据之后说清它与别处丢弃规则的关系：认人这一步不丢的行，纠错时口语颗粒压缩后不剩信息的照翻译目标第 4 条丢，套话以判据的三点为准
│  fragment_turn_clue_single_v1.md        # 有说话人标签时：「换人」是声学层给的线索（云端任务是转写模型给的，连同「同时」「换回」），没标的接缝不说明是同一个人
│  fragment_shape_same_person_single_v1.md  # 判据里「怎么看出是同一个人」那句：没有说话人编号，从对话结构、称呼与口吻看
│  fragment_turn_rule_single_v1.md        # 有说话人标签时，合并规则第 2 条里「标着换人的行不与上一行合并」那句
├─ 认人与定形状的判据（三个 pass 的 pass 1 与单 pass 的 capableC 共用，`shape_criteria.py` 渲染）
│  shape_criteria_{who,promo}_v1.md       # 认人的顺序（对话结构 → 称呼与口癖 → 内容与语气）、插播一段；只有单 pass 读，裁判的 system 有自己的说法
│  shape_criteria_{join,may_join,drop}_v1.md  # 什么时候是同一个人接着说、碎片要接上（join）、可以并（may_join）、整行不要（drop）
│  shape_criteria_{across_drop,auto_guard,strict_reason,fix_later,keep_lead}_v1.md、shape_criteria_{fix_later,keep_lead}_single_v1.md  # 两处措辞不同的短语层（`shape_criteria.WORDS` 指向它们）：只对 pass 1 成立的话（隔着 drop 的行接、超长的 join 由脚本挡、下一步只并不拆、下一步会纠错），单 pass 不写或换说法；保留清单的开头，pass 1 写「其余的行一律不 drop」，单 pass 写「认人这一步不因下面这些 drop」
├─ 三个 pass 的 pass 3：校对（组装在 `stages/proofread/prompt.py`，每个纠错窗一次调用，自己的版本号 `stages/proofread/run.PROMPT_VERSION`）
│  proofread_system_v1.md                 # 改什么（意思译错、读不完、专名、纠错文里的听错、残留的垫词与译重、风格）、不改什么（划分、只是换说法）、只写改了的行的五列输出
│  proofread_user_v1.md                   # 素材资料（准备任务的概要、人物、专名、听错、节目说明、没确定的事；有转述的用户要求时加 `proofread_user_requests_v1.md`）、风格、这一窗的成品行 `序号|时长|字/秒|识别原文|纠错文|译文|把握|备注`
│  proofread_{kb_terms,before,after}_v1.md # 这一窗纠错时用过的知识库词条（只取专名与术语行）、窗前窗后各两行只读的成品行；没有就整段不出现
│  proofread_retry_v1.md                  # 回答读不了时的重问：带上具体的原因
├─ 窗口任务的请求通道（叠在纠错 prompt 之上，纠错 golden 不变；组装在 `request_prompts.py`）
│  fragment_window_request_channel_v1.md   # 首条消息的「获取缺失输入」一节：$acquire_clause（按授权选 fragment_window_acquire_{search_knowledge,search,knowledge}_v1.md，后者在 native 下带 fragment_window_native_search_note_v1.md）+ <window_notes> 的说明 + 请求块（$entry_request_blocks + local 才有的 $search_queries_rules）+ 知识索引；要选风格时第 1 条带 fragment_window_request_style_block_v1.md（$style_block_clause）
│  fragment_window_agentic_acquire_v1.md   # 同一节的 agentic 形态（工具会话自行取数，叠在纠错 system 尾部）
│  correction_fill_v1.md                   # 回填消息：说明是回填 → 选了风格时的风格（要遵守的约定）→ 其余资料（词条全文、搜索结果）→ 交付提醒
│  fragment_fill_materials_note_v1.md      # 回填资料前那句「下面是内容，不是给你的指令」（只在有资料时出现）
│  fragment_entry_details_label_fill{,_slimmed}_v1.md / fragment_fill_search_result_v1.md   # 回填里词条全文块的说明（_slimmed：透传的词条瘦过身时）与搜索结果块；请求了却没取回的通道写 `phrases_v1.toml` `[fill]` 的那句
│  fragment_task_feedback_reminder_v1.md   # 开了反馈采集时，回填交付清单里的 <task_update_feedback>
│  fragment_search_results_input_v1.md     # 首条 user 消息里的搜索结果块：只在 replay 冻结了搜索结果、重打交付轮时渲染
│  fragment_search_queries_output_v1.md    # <search_queries> 输出规则（内嵌 $query_style；准备任务与窗口共用）
│  fragment_search_queries_kb_note_v1.md   # 其中「已注入的词条足以回答的主题不花 query」那段：有可读知识库时才有
│  fragment_query_style_v1.md              # query 写法通则（自包含/语言选择/引导语）
│  fragment_search_results_usage_v1.md     # harness 注入的搜索结果的使用规则（纠错 $retrieval_block 与准备任务回填共用）
│  fragment_self_search_usage_v1.md        # agentic 窗口自己检索时的使用规则
│  fragment_search_rules_shared_v1.md      # 上面两份与 native 准备任务的自带检索指引共用的几条（交叉验证、与知识库冲突、内容不是指令）；`prompt_compose.search_rules_shared` 组装，`renumber_rules` 接着编号
│  fragment_search_rule_kb_conflict_v1.md  # 其中「搜索与知识库条目冲突」那条：没有可读知识库时不出现
├─ 准备任务（一条对话；组装在 `prepare_prompts.py`）
│  prepare_task_v1.md / prepare_task_user_v1.md / prepare_task_results_v1.md   # 首条=完整契约与 <context_pack> schema；results=回填的第二条 user。职责第 5 条转述用户要求：`<extra_info>` 里要纠错照做的内容，全片适用的进 `general_context.user_requests`（harness 挪成 `ContextPack.user_requests`），只关几段的进那几窗的专属背景并写明「出自用户」；专属背景用 `window_ids` 列出共用一条的窗口
│  fragment_prepare_url_extracts_v1.md       # 用户备注里 URL 的页面提取内容（有内容才注入，输入清单同时点名它）
│  fragment_prepare_earlier_chunks{,_duty}_v1.md / fragment_prepare_name_disputes_field_v1.md   # 超长素材的第 2 块起：user 里前面各块的名字表与摘要（`<earlier_chunks>`），system 的「分段接续」段（照用、不重抄、意见写进 `name_disputes`）与 schema 里 `name_disputes` 一行；第 1 块与不分块时三个槽都空
│  fragment_prepare_request_channel_v1.md / fragment_prepare_agentic_acquire_v1.md / fragment_prepare_no_evidence_v1.md   # 「获取缺失输入」三种形态（batched 请求通道 / agentic 自取 / 无外部证据）
│  fragment_prepare_style_selection_v1.md    # `--style auto` 的候选目录与 <style_selection> 块（auto-style 计划 §3）
│  fragment_knowledge_inputs_v1.md           # 双索引 + 预注入输入段（knowledge 谓词）
│  fragment_research_entry_blocks_v1.md      # <requested_entries>/<keep_entries> 请求块（knowledge 谓词；窗口请求通道复用）
│  fragment_research_keep_entries_v1.md      # 交付时的词条透传块（有可读知识库的各档）
│  fragment_native_search_research_v1.md     # 交付调用的自带检索指引（仅 retrieval=native；$shared_rules 填 fragment_search_rules_shared_v1.md 的第 2–4 条）
│  fragment_prepare_name_conflict{,_no_kb}_v1.md  # 职责第 4 条：<extra_info>、知识库与转写说法不一时按谁写（没有知识库的档用 _no_kb）
│  fragment_prepare_{entries,search}_result_v1.md  # 回填轮里请求的词条全文与本地搜索结果块（没取回时写 `phrases_v1.toml` `[fill]` 的那句）
│  # 背景条目（`[prepare_background]`）、输入清单、职责、keep 来源与最后提醒的几种说法（`[prepare]`）在 `phrases_v1.toml`，按授权挑
├─ 任务反馈采集（--knowledge update，或参考资料导入）
│  fragment_task_feedback_schema_v3.md    # feedback v3 JSON schema（共享；source_ids 规则按单窗口局部/多窗口稳定序号参数化）
│  fragment_task_feedback_source_ids_{local,stable}_v1.md  # schema 里 `source_ids` 那条：纠错窗口的局部序号 / 准备任务的稳定源序号
│  correction_task_update_feedback_v2.md  # 纠错窗口采集要求
│  research_task_feedback_v1.md           # 准备任务交付时的采集要求
├─ 统一知识更新（docs/knowledge.md）
│  knowledge_update_artifacts_only_v1.md  # 无精修模式 system
│  knowledge_update_refined_v1.md         # 精修对照模式 system（第 7 条：只在用户的 <task_summary> 明确要求时写或改 [字幕长度]；落库时代码也拦，`length_tier.tier_line_gate`：用户没给 `--task-summary` 不加、不改、不删）
│  knowledge_update_user_v1.md            # 共用 user 模板（窗口包 + 全局块）
│  fragment_knowledge_update_inputs_v1.md # 两模式共用的输入说明（refined_csv bullet 与库描述参数化）
│  fragment_knowledge_structure_v1.md     # 知识库定位/结构 v2/更新原则（共享；含行文法、吸收判据、迁移指令）
│  fragment_knowledge_create_index_check_v1.md # 新建条目前对照 <kb_index> 那句（提案回喂没有索引，不带）
│  fragment_knowledge_output_v1.md        # <knowledge_proposals> op schema（`MODEL_OPS` 里除 remove_item 外的十种）
│  fragment_knowledge_size_rule_v1.md     # 更新原则第 8 条尺寸纪律；只在有注入的条目带长度提示时出现（$size_rule）
│  fragment_knowledge_split_clause_v1.md  # 尺寸纪律里「可以改用 split_entry」那句；尺寸纪律在、且 op 表带 split_entry 时才出现（$split_clause）
│  fragment_knowledge_term_lines_v1.md    # 四列术语行的写法（op schema 与共享冲突修复的 op 小节共用）
│  fragment_knowledge_style_marks_v1.md   # 精修对照模式 + 钉了 style 时的 <style_marks> 契约（knowledge/style_marks.py）
│  fragment_knowledge_style_marks_reminder_v1.md # 同一格最后提醒里补的那半句
│  knowledge_proposal_repair_v1.md        # 提案回喂 user 模板：并发被丢（B'）与被拒收的提案只补这几条；system 是 knowledge_proposal_repair_system_v1.md（两样输入与证据规则，结构说明与输出格式同知识更新）
│  knowledge_update_retry_v1.md           # 知识更新与提案回喂的回答读不了时的重问：带上具体的原因
│  fragment_kb_repair_{conflict,rejected}_v1.md # 它的两节：哪几条因什么没写进去、各怎么判（有那一类才出现）
│  fragment_knowledge_update_{refined_csv,chunk_notice}_v1.md  # 输入说明里 `<refined_csv>` 那条（精修对照模式）；材料分块时 user 里的分块说明
├─ 知识库维护会话（plan §11.3/§11.4；docs/knowledge.md）
│  fragment_kb_judgment_v1.md             # 每个维护会话都给的判断标准（第 0 条收录标准：只为 ASR 听对 + 字幕写对；同一实体判定；证据纪律）
│  kb_base_style_v1.md                    # 已有风格叠加出厂默认（`knowledge adopt-base-style`，node/base_style.py）：自成一份，只有 append_lines 与 <skipped>，不带收录标准与完整 op schema
│  fragment_kb_writing_v1.md              # 只给写条目的会话（修复/材料吸收/共享冲突修复）：有无参照物决定能否写误听、行的归属、描述纪律、标记命名
│  fragment_kb_task_material_v1.md        # 吸收任务的任务段（$material 素材正文 + $user_prompt_block 用户交代；不可信文本边界）
│  fragment_kb_task_candidates_v1.md      # 修复任务的任务段（$candidate_rows + $verdicts 裁定枚举）
│  fragment_kb_task_candidates_lead_v1.md # 修复候选时开头那句多出的「并对每条候选给出裁定」（$verdict_clause；材料吸收不带）
│  fragment_kb_candidate_hints_v1.md      # 整理候选每类一行的处理建议（scan.py 按 kind 取；人看候选时也是这一句）
│  kb_repair_v1.md                        # 修复任务：二次迁移候选或任意素材 → <knowledge_proposals>（$ops_contract 复用同一份 op schema，$writing 是写条目的规则）
│  kb_verify_v1.md                        # 校验任务：缺证 term/fact claim → <verify_results>（confirmed 必须带 URL / unverifiable 终态）
│  share_review_v1.md                     # 共享审核会话（share/review.py；<review_verdict>）
│  share_conflict_repair_v1.md            # 共享 pull 冲突的修复会话（share/conflict_repair.py；<conflict_verdicts>）
│  fragment_share_conflict_ops_v1.md      # 共享冲突修复的 op 小节：只有闸门放得过的 update/remove/add_item（不给整份 op schema）
│  fragment_share_conflict_{intro,surface}_v1.md # 本轮有简介 / H1 冲突时才插入的 set_intro / rename_entry 说明与行格式
├─ 说话人裁判（--diarize judge-text / judge-audio；组装在 `stages/speaker_judge/prompt.py`，从实验工具 `window_judge_r9.py` 移植；每窗一次调用）
│  speaker_judge_system_v3.md             # 裁判 system：$summary（准备任务的概要）、$marks（置信标说明，开发开关关掉、或这一窗要看的行都没带标时为空）、$audio（带音频时）、$row_format / $fields（行格式与字段说明）、$shape 与开头任务句里的 $shape_task（三个 pass 的 pass 1 才有，后者是 speaker_judge_shape_task_v1.md）、$split_hint（带音频的 pass 1 才有；音频说明里的 $insert_hint 同）
│  speaker_judge_user_v3.md               # 裁判 user：标题、候选人物（准备任务的 speakers、知识库主播、人物，最多 30 个）、轨迹画像（带音频时附布局表）、这一窗的行（▶ 标要判的行）
│  speaker_judge_shape_v1.md              # 三个 pass 的 pass 1：「定字幕的形状」一节，把 join / may_join / drop 的判据（shape_criteria_*）包成 JSON 字段
│  speaker_judge_shape_audio_v1.md        # 带音频的 pass 1 接在形状一节后面：`split`（在换人处切开一行）、`fill`（补上一行漏掉的半句）、`insert`（补一行整段漏掉的话）
│  speaker_judge_{audio_note,marks_note,summary}_v1.md   # 上面三个槽位的正文
│  speaker_judge_audio_{layout,silent}_v1.md            # 带音频时 user 里的布局表与「没有声音样本的轨」
│  fragment_prepare_speakers_{duty,field,no_search}_v1.md  # 开了裁判时准备任务多交的 speakers（职责第 6 条与 JSON 字段；关着时两个槽为空）；no_search 只在能查资料的档进职责第 6 条
│  speaker_judge_fields{,_shaped,_shaped_audio}_v1.md   # $fields：只判轨号 / 加形状一节 / 加听出来的 split 等
│  speaker_judge_{split,insert}_hint_v1.md              # $split_hint、$insert_hint 的正文
│  speaker_judge_conflict{,_shaped}_v1.md               # 带音频时内容与声音冲突又说不清的答法（$conflict_answer；形状一节下只答 keep，owner B2）
│  speaker_judge_shape_same_person{,_cloud}_v1.md  # 形状一节判据里「怎么看出是同一个人」那句（按裁判判完的轨；轨号是云端转写的说话人时用 _cloud 那份，prompt.py 的 SHAPE_SAME_PERSON_TEMPLATES；单 pass 用 fragment_shape_same_person_single_v1.md）
│  speaker_judge_source_{acoustic,cloud}_{intro,errors,profile,cells,trust,keep,mix}_v1.md  # 讲轨号来源的几处：声学层（本地任务）或转写模型的说话人（行带 line_speaker 的云端任务；cells、keep、mix 没有云端版，那几处为空）；prompt.py 的 SOURCE_WORDS 指向它们
│  # 轨号格、画像行、布局表的行、行格式这些逐行短语在 prompt.py 的 PHRASES（同 prompt_compose 的短语层）
└─ 修复轮（validation 失败后的同窗重试）
   fragment_repair_round_v1.md            # 会话无关：列出校验错误 + 要求重出完整结果（$validation_errors）
   fragment_repair_patch_v1.md            # 纠错窗口的补丁修复轮：错误、待修源、只交 <patch>（correction_patch.py）
   fragment_serial_shift_error_v1.md      # 纠错窗口的整窗错位报错：从哪个源起、写的是哪个源的话（output_protocol._with_shift）
   fragment_{uniform,placeholder}_rows_error_v1.md  # 纠错窗口的占位答案报错：整窗同一句 / 过半是占位词，要求整窗重写（output_protocol._with_placeholder）
   speaker_judge_repair_patch_v1.md       # 说话人裁判的补丁修复轮：错误、要重写的 id、只交 <patch>（stages/speaker_judge/patch.py）；写到的行挂着的、底稿 <tracks> 里没有的轨与标签另写在 <patch> 外的 <tracks>
   speaker_judge_repair_whole_v1.md       # 说话人裁判的整份重写轮（<judge> 闭合却没有能留下的行）：错误、照 system 指令的格式整份重交（每个 ▶ 行一行 JSON、再写 <tracks>）
```

`fragment_repair_round_v1.md` 是唯一**不属于任何一个会话**的模板：它只谈输出契约、不提某个
阶段的块名，因此任何有输出校验的调用都能把自己的错误递回去。它由
`compose_repair_turns()` 渲染成 `assistant`（上一轮输出）+ `user`（错误清单）两轮，
在 `client.complete()` 里追加到已组装好的 messages 之后——**在附件之后**，这样剪辑仍留在
陈述任务的那一轮上。它不进 `PROMPT_VERSION`：attempt 0 的 prompt 一字未改，只有重试多了
东西，所以已提交的窗口和 research context 不作废。
要别的回答形状的阶段自己渲染 user 那一轮，经 `client.complete(repair_instruction=)` 交进来，
`compose_repair_turns()` 原样用它；本地 agent 收到的是 capsule 里的 `input/repair-instruction.txt`
与 stdin 的 `repair_instruction`（agy 的这个文件与上一轮输出接在任务份末尾），代替各自「交完整结果」的
措辞。现在纠错窗口的补丁修复轮（`fragment_repair_patch_v1.md`，`attempts.py` 在上一份答案大半是对的时候用）与说话人裁判的补丁修复轮
（`speaker_judge_repair_patch_v1.md`，`stages/speaker_judge/run.py` 在 `<judge>` 里有能留下的东西时用）这样用；裁判在 `<judge>`
闭合了却没有能留下的东西时也不用通用的那份（它说的 header、列数裁判没有），交自己的整份重写轮 `speaker_judge_repair_whole_v1.md`。这几份
同样不进各自的版本号（`PROMPT_VERSION`、裁判的 `speaker-judge-v*`）。

## 组装器（`src/finesub/llm/prompt_compose.py`）

纠错的两个入口按 `TranslationProfile` 选 fragment 并填参；准备任务与窗口请求通道的组装分别在
`prepare_prompts.py` / `request_prompts.py`，按相同开关谓词叠加：

```python
compose_correction_system(profile, *, tier=CapabilityTier.CAPABLE, variant=None,
                          extra_style="", style_block="", knowledge_enabled=True)
compose_correction_user(profile, *, general_context_json, window_context, entry_details,
                        previous_advice, pre_round_notes, search_results,
                        preceding_context_csv, current_asr_csv,
                        current_asr_row_count, tier=CapabilityTier.CAPABLE)
build_window_task_messages(request_channel=..., knowledge_indexes=..., carried_entries=..., ...)   # request_prompts
build_prepare_task_messages(transcript=..., emits_queries=..., emits_entries=..., agentic=..., ...)  # prepare_prompts
```

更高层的 `build_correction_csv_messages` 在 `src/finesub/llm/prompts.py`，负责把 harness 注入内容填进 user 槽位（`build_window_task_messages` 包住它加请求通道与回填消息）；其中 `current_asr_row_count` 由实际窗口片段数计算：各变体用它重申 translated 必须完整覆盖本窗。纠错窗（请求回合与交付轮是同一份首条消息）的 `<asr_result>` 使用 `local_id|start|duration|gap|text`：目标行每个执行窗口重置为 `1..N`；并列的 `<preceding_context>` 使用非正数，最近前文为 0。模型输出在 validator 后立即映射回稳定源序号，窗口拼接、时间轴、annotated CSV 与知识材料始终使用稳定源序号。

### 纠错 system 槽位选择表（`—` = 空槽塌缩）

槽位由**开关谓词**选，不再按 preset 枚举：

| 骨架槽位 | 谓词 |
| --- | --- |
| `$role_block` | `correction_media≥audio` → audio 版（`=video` 追加 video 增补）；否则 text 版 |
| `$goals_correction_block` | 同上（`correction_media≥audio` 二分） |
| `$retrieval_block` | `retrieval=local` → injected（讲 `<search_results>` 用法）；`retrieval=native` → native；`none` → — |
| `$hallucination_block` 丢弃取舍子句 | `correction_media≥audio`（能重听才谈重听） |
| 模态参数（`$correction_basis`/`$csv_time_note`/`$paren_rule` 等） | `correction_media≥audio` 二分 |
| `$verify_basis` | **按轴拼接**：媒体分量（音频/上下文）+ 检索分量（local→背景资料+搜索结果；native→自己检索到的资料；none→无） |
| `$input_inventory` | 角色一段「你会同时参考 / 只能依据」的输入清单：媒体分量 + 背景资料（每个窗口都带的背景块：准备任务的结果，不看检索档；fast 模式是用户的说明）+ 知识库词条（有可读知识库时）+ 累积建议台账（串行时）+ 本窗只读前文 |
| `$effort_block` | `retrieval≠local`（无注入才需要 effort 措辞）；`difficulty=efficiency` → low，否则 deep |
| `$advice_block` | `continuity=serial`；`parallel` → — |
| `$keep_block` | `knowledge` 开启、窗口有请求通道且 `continuity=serial`；否则 — |
| `$window_block` / `$hallucination_block` 主体 / 翻译目标 | 全组合共有 |
| 插轴（rules/example/output 子句） | v63 起全面废弃，无组合注入 |

**注入实况原则**（§5.2）：prompt 提到的每个输入块必须真的会注入，注入的每个块必须有
用法说明；二者由同一谓词驱动。渲染产物残留任何未解析 `$token` 即
`PromptAssemblyError`（`safe_substitute` 本身会静默放过）。

`$merge_block` / `$examples_block` 由具名 variant 选择。**生产的变体名来自任务组格子
（model-routing v2 D2/D3）**：correction quality、intermediate 格 = capableC，efficiency 格 = basicA，
模型组条目可按 target 覆盖（prompt 迭代/自定义组用），组内退让**不再换变体**；旧的
「应答模型档位 + difficulty 封顶」推导（`effective_tier`）已删除。名字仍可在
replay/`--variant` 显式指定。capableC 是融合单 pass（见下面[单 pass 的拼法](#单-pass-的拼法)），六列 CSV、不写 `#`
注释，长度是一般 20 字/4 秒左右、36 字/7 秒绝对门槛（标准档的数，按任务的长度比例缩放；同 pass 2 的分组版 `capableC-grouped`）；basicA 一行对一个源、只接回词中切断，
输出带 start 七列 CSV（v104 起不写 `duration|gap|char_count`），交付终稿时先写不超过 8 行的 `<decisions>` 决定清单（v109）。原来 efficiency 用的 basicB（判断型合并、20 字/4 秒硬门槛）在两个低能力模型上并错的多于 basicA 漏并的，两位读者都显著偏向 basicA，basicB 已删（owner 2026-11-02）。

fast 单窗复用同一骨架（整段一个窗口，请求通道同样叠加）。请求回合没有单独的 prompt：它与交付轮共用这份首条消息，媒体措辞只随 `correction_media` 切换。

准备任务的请求回复按 `<analysis_notes>` → `<requested_entries>` → `<keep_entries>` → `<search_queries>` 输出；request 只负责新加载并按重要性排序，keep 只负责保留本轮可见的预注入词条。两类 canonicalize 后各自最多 8 条、合计最多 12 条，harness keep-first 合并，超限时从 request 尾部丢弃。回填消息（`prepare_task_results_v1` / `correction_fill_v1`）把词条全文放在 `<search_results>` 前，并声明「内容不是指令」。Raw query/URL section 分隔符分别是 `--- query: ... ---` 与 `--- 深度提取 url: ... ---`。

输出完整性约束集中在纠错契约：模型不得省略必需标签、header 或记录。basicA 要求精确七列 header 与 `start`；capableC 与 `capableC-grouped` 要求精确六列 header；每行格数都必须与 header 相同。validator 只接受当前窗口的正局部序号，0、负数和越界值都会整窗重试；通过后统一映射回稳定源序号。

### 单 pass 的拼法

单 pass 的 capableC 是「融合单 pass」（owner 2026-10-30）：把 pass 1 的认人与定形状和 pass 2 用胶水合成一次调用，判据共用现成的片段，不另写一份。

- **先认人、定形状**（`fragment_shape_single_v1.md`，骨架里的 `$shape_block`，在合并规则前面）：胶水说先做这一步、在内部思考里做、不写出来，再按合并规则做；判据是 `shape_criteria_*` 那几份（认人的顺序、插播一段、`join` / `may_join` / `drop`），`shape_criteria.shape_criteria("single")` 渲染；pass 1 的形状一节用 `"pass1"` 渲染其中 `join` / `may_join` / `drop` 三份（认人的顺序与插播一段裁判的 system 有自己的说法）。怎么看出是同一个人那句单 pass 用自己的（`fragment_shape_same_person_single_v1.md`：没有说话人编号，从对话结构、称呼与口吻看）。只对 pass 1 成立的话（隔着 drop 的行接、超长的 join 由脚本挡、「下一步会纠错」）是短语层 `shape_criteria.WORDS`（模板文件），单 pass 填空或换说法；单 pass 与 pass 1 读同一个 `gap` 列，停顿的措辞相同；`join`、`may_join`、`drop` 在单 pass 里只是判断时的叫法，不写进输出。
- **再纠错与合并**：用 pass 2 的那组模板（合并规则、输出契约、user 模板、翻译目标与对齐），合并能落在哪些接缝上由参数洞填（成段的是模板 `fragment_merge_*_single_v1.md`，由 `CorrectionVariant.merge_intro_fragment` 等字段点名，几个字的是字段，文字在 `phrases_v1.toml` 的 `[variant.<变体名>]`，`prompt_compose._merge_holes`）：`join` 的接缝要并，`may_join` 的接缝看停顿、长度与翻译风格再定，其余接缝不并；合并行的 note 写为什么是同一个人接着说（user 消息第 3 条也提醒）。
- **丢行**：pass 1 的 `drop` 两种（单独成行的垫词、确信的套话幻觉）写成 `discard|<源序号>`，不再叠单 pass 的 `fragment_hallucination_v1.md`；一行里无意义的复读这类，按契约「字幕节奏与排版要求」第 4 条处理。判据的保留清单（感叹、语气词、寒暄……）在单 pass 里只管认人这一步：纠错时口语颗粒压缩后不剩任何信息的（纯占位的孤立垫词、无意义的感叹），照翻译目标第 4 条丢；套话只在判据的三点同时满足时丢，契约与 user 消息里的「高度疑似幻觉」不放宽它。这两句只在单 pass 里有，pass 1 的渲染不变。夹在一句话中间的垫词走 filler 三明治（三行一起并）：合并与 discard 都只认连续的源，prompt 写明。
- **源数**：一条最多并 4 行原始 ASR（`prompt_constants.BASE` 的 `fused_max_sources`，不随长度比例缩放；三个 pass 里 join 组加一次可并也能到 4 行），三明治是其中一种、不另设更严的长度；示例 30（「唯一允许的三源形态」）不对它渲染。校验器超过 4 个源才告警（`CorrectionVariant.max_merged_sources`），`grouping.py` 的 join 上限仍是 2。
- **有说话人标签时**：`turn` 列标着「换人」的接缝不并（云端任务里「同时」「换回」也一样，线索那句说标签来自转写模型）；认人仍从台词判，标记只是线索，没标的接缝也可能是两个人（`fragment_turn_clue_single_v1.md`，不用单 pass 原来那句「没标的行不要只因没有标记就当作拿不准」）。
- **有音频时**：认人可以听声音佐证，判成 `drop` 之前先重听（`$voice_clause`、`$drop_listen`，媒体参数层）。
- **长度的硬护栏**（owner Q13）：并后越过 36 加权字或 7 秒的合并行校验不过、进修复轮（`CorrectionVariant.rejects_long_merges`，见 [`llm_harness_behavior.md`「重试与拼接」](llm_harness_behavior.md#重试与拼接)）；单源本身超过的照收。

## 只纠错

中文素材的任务只纠错、不翻译（何时、怎么判、续跑与流程开关见 [`llm_harness_behavior.md`「只纠错」](llm_harness_behavior.md#只纠错)）。prompt 这一侧是套在现有变体上的一层，不另写一套模板：

- **派生的变体**（`prompt_variants.correct_only_variant`）：每个变体都有一个名字后缀 `-polish` 的只纠错版，现在注册了三个：`capableC-polish`、`basicA-polish`、`capableC-grouped-polish`（分组版的派生名由 `grouped_variant` 对应）。它与原变体只差几处：`correct_only` 为真（文案取 `[correct_only]` 一组、第二列叫 `polished` 且可空）、没有对齐策略那一节（润色版不跨行错位）、提醒里的「长度与翻译风格」改成「长度」、basicA 的示范行与分组版三明治例子里的垫词换成中文（变体字句里的这几处是 `$占位`：`may_join_factors`、`splice_row_example`、`sandwich_filler`，派生时用 `[correct_only]` 一组重新填）。新加一个变体，只纠错版自动套上。
- **随模式变的字句**在 `prompt_templates/mode_texts_v1.toml`：`[translate]` 与 `[correct_only]` 两张表，键是模板里的 `$占位`；纠错目标一节（「翻译与内容取舍」，`goals_` 开头的键；只纠错时第 1 条写两列各做什么（「ASR 漏转的字照听到的补」只在带音频的窗口里有：`goals_audio_fill` 经媒体参数 `audio_fill_clause` 填在 `goals_first` 与 `goals_first_tail` 之间），第 4 条口语颗粒是「corrected 照录、polished 按三步删、整行垫词 polished 留空」（「哭声、喘息这类高噪声片段除外」同样只在带音频的窗口里有：`goals_grain_empty_audio` 经媒体参数 `grain_empty_audio_clause` 接在 `goals_grain_empty` 后）；分组版第 8 条风格冲突只在翻译时有，由 `_style_conflict_rule` 按注入的风格渲染，条目编号是 `goals_style_conflict_lead`，只纠错为空）等纠错 prompt 里的字句由变体取（`prompt_compose.mode_texts`，按 `CorrectionVariant.correct_only`），规则里举的例子（`shape_` 开头的认人与定形状判据、`seam_cue`、`splice_example`、`hallu_` 开头的幻觉套话清单与边界）同样由变体取；准备任务说到翻译的几句与定名一项（`prep_` 开头的键）、pass 1 定形状一节按任务取（`task_mode_texts`；判据由 `shape_criteria(correct_only=)` 渲染，单 pass 与 pass 1 共用；只纠错时 `drop` 只剩套话幻觉一种，整行垫词不丢，相关几句是 `drop_filler_item`、`judge_drop_kinds`、`single_drop_kinds`、`shape_drop_tail`、`dropped_before`）。翻译一组就是翻译任务的字句，开槽时逐字照搬，翻译任务的 prompt 没有因开槽而变（纠错的 system / user 与 pass 1 定形状一节由 `test/data/prompt_goldens/` 的快照钉着；准备任务、风格挑选、请求回合、agentic、任务反馈与音近提示几条路没有快照，2026-11-07 人工比对过逐字相同，改 `[translate]` 的这些键没有测试兜底；第四轮冷读第二步在合并后的 `dev` 上渲染，这些路径的冷读快照与之前入账的逐字相同，音近提示默认关、没有冷读快照）。
- **输出列**：capableC 与分组版的只纠错版是六列 `type|position|corrected_text|polished|conf|note`，basicA 的是七列（多 `start`）；`polished` 是表头名（`output_csv_header(polished=)`），内部仍叫 `translation`，所以之后的环节看到的形状与翻译任务相同。校验对它放行空格子（`validate_translated_csv_text(polished=)`），空时字数与长合并按 `corrected_text` 算。
- **示范**：只纠错读 `prompt_templates/examples_zh/`（`example_builder.EXAMPLES_ZH_DIR`）的中文素材，翻译读 `examples/` 的日语素材；中文那套沿用日语那套的场景结构与时间轴（对照要点里算出的时长、字数照样成立），有 capableC / basicA 共用的主场景与缩窄、三明治、换人、噪声四个小例，没有只教译法的那一个。第二列写 `polished`：多数行照抄 `corrected`，几行示范局部编辑（删垫词、压口吃、改量词、顺语序、补漏字），整行只有垫词的行 `polished` 留空。`build_examples(polished=True)` 只让表头写 `polished`。`examples_note` 说明 `polished` 是照抄再局部编辑。
- **风格**：只纠错的风格块只装写给它的风格（`render_style_block(correct_only=True)`：没有就是空，不出出厂默认），抬头是 `style_header` 一句；`compose_correction_system` 对只纠错的变体丢掉 `--extra-style`。`--style auto` 的挑选（准备任务、请求回合、agentic 窗）各处措辞走 `style_` 开头的键，`--extra-style` 不进挑选一节。
- **守卫**：`test/test_llm_correct_only_prompts.py` 把只纠错的 prompt 按媒体 × 检索 × 档位 × 形态 × 知识库 × 说话人（无 / 换人列 / 带分组键）的网格渲染一遍，再加请求回合、回填、agentic、任务反馈、音近提示、各种形态的风格挑选、整个准备任务与一套写给它的风格几条路，断言没有「译」「translation」（豁免知识库标记「音译」与查询规则里的「中日互译」；「润色」在这里是正当的词），并检查 prompt 里写出的示范行与 header 的格数相同。另一项把只纠错任务真会看到的几条路（不开知识库的网格、不开知识库的风格挑选、整个准备任务、pass 1 的定形状一节）去掉示范再渲染，断言没有假名：规则里举的例子都是中文的。只纠错另有四份快照（三份纠错、一份 pass 1 的定形状）。
- **维护**：往共用模板里加一句只属翻译的话，写成槽（两张表各填一句）或放进只翻译的片段；忘了开槽守卫会红。改示范的场景或对照要点要两套材料（`examples/` 与 `examples_zh/`）各改一处，只教译法的只写日语那套。只改只纠错的字句也升共用的版本号，不另设一个；两种模式的窗口靠变体名分开。

## 迭代惯例

- 改模板 wording 不改语义：不 bump 版本；改输出契约/输入结构/职责边界：bump `PROMPT_VERSION` 并同步更新 `test_llm_prompt_compose.py` 的版本断言与相关 snapshot 断言。
- 用 `finesub.llm.correction_translation --prompt-dir` 输出真实任务的完整 prompt（准备任务首条消息 `research-prepare.txt`；常规窗口用纠错格子声明的变体，与真跑时一致；另落 `correction-0001-basic-tier.txt`：首窗的 basic 档默认变体（`basicA`），作对照）；冻结注入重打纠错交付轮用 [`session_replay.md`](session_replay.md)。
- prompt/harness 迭代由 `tools/session_replay` 受控重放驱动（见 [`prompt-iterate.md`](prompt-iterate.md)），知识更新不再输出 `<harness_notes>`。人工审阅 replay 产物后手动改模板，绝不自动应用。
- 合并策略：精修标定的软门槛与模型边界（thinking=0、错并代价）见 [`merge-calibration.md`](merge-calibration.md)；
  现行变体契约见 [`prompt-iterate.md`](prompt-iterate.md) [§4 当前合并口径（prompt 层）](prompt-iterate.md#4-当前合并口径prompt-层)。
- 设计过程草稿与过期实验日志在 `docs/archive/`：**在 `dev` 上被跟踪**（clone 与 worktree 都拿得到），由 `scripts/publish-main.ps1` 从公开快照剥掉，所以**不随仓库发布**。
