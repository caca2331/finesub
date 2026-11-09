# LLM Harness 后续实验与未实施工作

本文只记录**仍需验证或尚未实施**的工作，按 harness / Agent / A/B / 知识库 / 前端分节，每条一行：
状态 + 一句话现状 + 去处。要展开的放在后面的同名小节。现行 runtime 行为以
[`llm_harness_behavior.md`](llm_harness_behavior.md) 为准，prompt 结构以
[`llm_prompts.md`](llm_prompts.md) 为准，Agent 接线以 [`llm_local_agent.md`](llm_local_agent.md)
[§12.1 会话形态](llm_local_agent.md#121-会话形态四档已决定什么还差什么接在哪) 为准。

**已完成的项目不在这里留正文**：现行行为归 owner 文档，实施过程与当时的论证在维护者的本地归档里
（2026-09-02 整理前的整份原文，不随公开快照发布）。
本文以后新增一条要写清**触发条件**（什么时候该做）或**重启前提**（什么条件下再看），
没有的就写「记一笔」。

## harness：窗口、重试、预算、路由

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| **「模型有没有真的收到窗口正文」缺一个独立信号** | 待触发（2026-09-03 记） | 现在的判据是保留数下限 `KEPT_FLOOR`（owner 2026-09-30，整窗与半窗都适用，见 [`llm_harness_behavior.md`](llm_harness_behavior.md)「保留数下限」）：没被丢弃的源 ≤ `min(10, 总数×0.5−10)` 才拒。它拦得住「一行 sub + 其余全丢」，拦不住**没读正文、却留下超过下限条数**的回复，20 条及以下的小窗也只剩「全部丢弃」一道。补它要的是**另一种信号**——回执里能证明模型读到了正文的东西（例如要求回引窗口首末源的片段，或让 agent 前端把「正文已投递」做成可校验的事实），不是另一个数字。⚠ **今天没有触发证据**。真出现一例空读，再按这条做 |
| **视频多模态对质量的帮助** | 待实验 | 音频优于纯文本已知；输出系数不再含媒体项、不再按开关标定（owner 2026-09-30）。剩下只量 video 相对 audio 的质量收益，见下方同名小节 |
| **准备任务一次交付全量 `window_contexts`** | 有触发条件，未做（owner 2026-10-03） | 几十个窗口的长素材下单次输出可能触顶或被省略；超长素材本来就按块各开一次准备任务（2026-11-08 起一块约 50k 转写、十几个窗口），schema 也允许省略无内容的窗口，几个窗口共用的说明可以写一条、用 `window_ids` 列出。触发条件：真实产物里出现 `window_contexts` 截断或省略，再定分块交付还是只写非空窗口 |
| **知识更新与纠错输入里相同文字只写一次；`<raw_csv>` 与 `<final_csv>` 合成一块** | 以后一起改（owner 2026-11-08） | 知识更新一个窗口包里 raw 的文本与 final 的 `corrected` 很多行逐字相同，纠错输入也有同类重复；合并两块能再省一截。两件一起改，改时同步知识更新的说明与材料哈希 |
| **分块准备任务的并行** | 可选，以后（owner 2026-11-09：「关于并行，先暂时不做，以后作为一项可选的 follow up」） | 超长素材的准备任务一块接一块跑（后块要前块的名字），`fast`、`agy-fast` 这类并行预设上块越多越慢。要做时的形状：第 1 块串行跑完，第 2 到 N 块并行、都带第 1 块的名字表与摘要，墙钟约两块；第 2 到 N 块之间的分歧照样先出现的为准。不取「各块并行、事后对账」（定名时互不知情，原文键不同的同一个人对不上，还要多一个会话类型）与「先通读全片抽名字」（全片超分块上限）。现行见 [`llm_harness_research.md`「超长素材分块调查」](llm_harness_research.md#超长素材分块调查2026-08-30-落地plan-w7--p8) |
| **分块准备任务的后块改正前块的写法** | 以后（owner 2026-11-09：「把后续的纠正的更好的设计也作为一个 follow up」） | 今天名字表先出现的为准，后块的不同意见只在告警里列给用户。要让后块能改：改中文写法时，harness 在前块摘要与专属背景里出现旧写法的地方旁边加一行「本段背景里的 X 已改正为 Y」（不改模型写的原文）；改原文写法时用显式字段指明替换哪条，并作废前块误听表里指向旧写法的条目；依据要写来源或本段的原话，改写列给用户。限制：字面匹配挡不住一字的名字、嵌在别的词里的旧名、只改身份不改名字的情况；改错时全片一起错。触发条件：分歧告警里常出现后块对、前块错 |
| **准备任务引用 `<extra_info>` 的原文而不是抄一遍** | 以后（owner 2026-11-08） | 转述用户要求（`user_requests`、专属背景里出自用户的）今天逐字照抄，长歌词会花输出 token。做法：给 `<extra_info>` 加行号，准备任务写 `{{E12-E20}}`、harness 原样展开，引用错了报错 |
| **模型回答的退化：怎么判断、判出来重试一次、几份里取更好的** | 没思考时再问一次、思考泄露取末尾的答案已实现，待真实运行验收；pass 2 几乎不并只记录与提示（2026-10-17） | 最明显的质量滑坡是模型完全不思考（owner 2026-10-17），pass 2 整窗几乎不并较轻、只提示。判法、读数与还要定的事见下方同名小节 |
| **补丁修复轮在 agent 上的实际表现** | 待读数（2026-10-24 记） | 重放 A/B 只跑了 API 与 capsule（Luna 42 次交补丁 41 次，DeepSeek 20 次全交）；工具会话路径（agent 默认档）只有单元测试，接线之后的 agy 没跑过，冒烟里 agy 两次都无视补丁说明、整份重写（照收，只是不省）。那次冒烟之后，agy capsule 的补丁轮改了接法（合并 driver 输入输出上限时）：修复说明与上一份答案不再是 argv 点名的单独文件，而是接在任务份末尾、第一步与正文一起读完，新接法没在真机上跑过（owner 2026-10-24 接受不经真机验收）。等真实运行积累了修复轮，按驱动读 `correction_window_response` 的 `repair_mode` / `patch_reply` 分布：agy 若多半回整份，再看说明对它是否要换写法 |
| **校验器对空译文的行同时报「空译文」和「缺源」** | 记一笔（2026-10-24，修复轮冷读） | 一行只是译文为空，校验器作废它之后覆盖检查又报这个源缺失；冷读的三个读者都说两条并列会让模型以为上一轮漏了整行。改法是覆盖检查不再为已被逐行报错的源重复报。改动前就如此；触发：修复轮里出现因此改错（比如把该行改成 `discard`）的回复 |
| **PROHIBITED_CONTENT 被源文本本身误杀** | 长期搁置（owner 2026-10-02） | Gemini 拦下一次调用时，[`llm_harness_behavior.md`「重试与拼接」](llm_harness_behavior.md#重试与拼接)里的 content filter 阶梯按注入单元逐级丢弃重建 prompt，全部丢完仍被拦就报错——那时触发的只能是源文本本身。源文本侧的降级没做，也没有样本。**重启条件**：真碰到这种报错，且阶梯解决不了 |
| **Claude 的 `reasoning_extraction` 拒答，finesub 当成空回答** | 已观测，未做（2026-10-30 记） | Anthropic 对明着要求写推理的 prompt（`<reasoning>` 块、行前 `#` 推理注释）回 `stop_reason: "refusal"`、输出 0 token，`stop_details.category` 是 `reasoning_extraction`（直连 API 复现，开不开思考都一样；agy 上转述成「blocked by content safety filters」再重试 4 次）。纠错窗 csv-v109 起、其它会话 2026-11-08 起都不再要求写推理：Sonnet 5.5 直连 API 上准备任务（长素材一块与两块）和纠错窗都正常作答（2026-11-08）；知识更新也在 Sonnet 上跑过一次、正常作答（其余会话不专门在 Claude 上测，owner 2026-11-09）。要做：认出这类拒答，把类别与说明报给用户，不按空回答重试。 |
| **中文只纠错：准备任务的背景写成导读腔、术语规则写译法** | 记一笔（2026-11-06） | 一份中文讲座里，准备任务写的全局摘要与窗口提要用「学术开拓」「家国情怀」这类导读措辞，术语规则写「字幕保留英文或对译为抒情传统」，后者和纠错窗「外语照原样写」冲突。冷读与 Codex Sol 都指出了；Sol 估计导读腔可能助推 `polished` 写成书面语，但正常的那一窗收到的背景同样是导读腔，证据不足，所以没改。要改的话：只纠错时让准备任务的摘要只列帮助辨认话题、指代、专名的事实，用平实短句，术语规则不写对译。现行的只纠错见 [`llm_harness_behavior.md`「只纠错」](llm_harness_behavior.md#只纠错) |
| **纠错窗口字幕上限的别的大小与别的素材** | 等 owner 要求（2026-11-08 记） | 10,000 对 5,000 在两段普通对谈上看不出质量差别，默认留在 10,000（[`llm_design_notes.md`「纠错窗口字幕上限的决策记录」](llm_design_notes.md#纠错窗口字幕上限的决策记录2026-11-0708)）。owner 提过以后可能补 7,000、3,000 等大小。补的时候固定窗口大小：切片按这个大小装得下的整窗数截取，装不下一整窗的尾巴不跑（生产是把整份源均分，3,000 会把 1 万切成 4 个 2,500）；在那次的同一个提交上跑，拿那次的 10,000 当对照，不重跑（owner 2026-11-07）。长独白、专名多的素材上 5,000 有没有收益没测过；小窗要单列专名跨窗一致。切片、知识库快照与跑批脚本在维护者本地 |
| **整窗错位校验漏掉短的局部错位** | 记一笔（2026-11-06） | 中文只纠错的验收里，一份讲座的 quality 输出在约 10 行里整体错一行（这一段每行写的是下一行的话），校验没拦下。原因还没查：quality 的行多是两三个源并成的，错开一行之后和自己的源仍有一半重叠，可能达不到「比自己的源高出 0.15」，也可能只拿单个源比时相似度不到 0.5。要改门槛得先在已有回答上量误报。现行做法见 [`llm_harness_behavior.md`「重试与拼接」](llm_harness_behavior.md#重试与拼接)的整窗错位一条 |
| **旧格式 catalog 的转换命令与改名提示** | 到 1.0.0 删（2026-11-09） | `finesub.llm.routing.migrate_catalog`、加载器对数据根 `model_catalog.psv` 的拒绝与转换提示、`model_routes._renamed_hint`（「target 不存在」时报出改了名的新名）一起删；之后改接口或产物格式前先问 owner |
| **自己接的 API 端点的联网搜索没真机跑过** | 等第一次实用（2026-11-09 记） | Anthropic 服务端 `web_search` 与 manual「自己的端点联网搜索」里各家的 `search_request` 只按官方文档与手写的事件流夹具验收，没真调 API（owner 定）。智谱的搜索引擎参数、OpenAI 的搜索模型名、火山方舟流式时能不能搜，文档也没说全。第一次实用时核：请求被接受、回答带出处、调用记录里有搜索行（Anthropic 还有 `web_search_requests`）、`pause_turn` 多不多见；哪家的写法不对就改 manual 那张表 |

## Agent：会话、长驻 worker、工具协议

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| **会话模式四档的对照 + 复用经济账重测** | 正式两段已完成（2026-10-02）；两家都不接入，`resume` 留在实验档 | 读数在 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) [3.9 `resume` 对 `per-window`](llm_local_agent_experiments.md#39-resume-对-per-window跨窗信息传递与单窗质量2026-10-02agy-每臂-n5codex-luna-每臂-n3)（试跑在其前一节 3.8）：`resume` 把前面窗口的信息带到了后面（预先冻结的跨窗评测点上，五格里四格 `resume` 每次运行都高于 `per-window`），专名写法也更统一；单窗整体质量只有方向相反的弱信号（agy 一份素材上后段窗口有变差的迹象，Luna 一份素材上 `resume` 稍好，p 都在 0.1 上下）。缓存读取多 4–20 倍，未缓存相当。**不接入的理由不是质量，是还没探索完**（owner 2026-10-02）：进生产前的两项前提（每窗只发增量、历史上限或轮换，见下方「未做：`resume` 档在工具会话下仍未定义」）没做，agy 后段的迹象若属实，也可能正是历史过长所致。**重启条件**：两项前提做完后，按 3.9 的设计（共用准备任务、`--knowledge none`、逐窗排名 + 冻结评测点 + 专名一致性）重测；dev 上知识库默认已是 `read`，那时要先定 `--knowledge`——开着时 `resume`（恒走 capsule）是 harness 注入、`per-window` 工具会话是模型自己取，比的就不只是会话档位 |
| **`resume` 档在工具会话下未定义** | 共存已验证（2026-09-30）；接线等 `resume` 的两项前提做完、按 3.9 重测 | 今天恒走 capsule 窄路。三家的续会话 flag 与 MCP 配置能同时用；真要接线要改五处（含每个任务列同一份工具、CodeBuddy 的搜索逐次把关）。会话档位实验已做（3.9），`resume` 留在实验档——不是因为质量，是还没探索完（owner 2026-10-02）；进生产前还缺两件。见下方同名小节 |
| **agy 模型下线被当成额度用尽，冻结整个额度池** | 记一笔（2026-10-24） | 2026-10-23 冒烟误钉 agy 3.6 Flash（已从 agy 的模型表下线）时，失败被归成额度，`AGY_GEMINI` 冻了两小时，同池的 3.8 Flash 也跟着停。agy 的「模型不存在」应归配置错误、不冻池；原始报错没留下，先在下次碰到时存下 stderr 原文再改分类 |
| **agy 整段收下长附件之后撤掉切段** | 等 agy（owner 2026-11-02） | agy 现在超过 5 分钟的附件只给前一部分，所以 driver 把音频与视频都切成不超过 295 秒的几段（`split_audio_for_agy`、`split_video_for_agy`）。agy 升级之后按 [`llm_local_agent_agy.md`](llm_local_agent_agy.md)「[2 音频：MIME 被工具层拦住，必须容器化](llm_local_agent_agy.md#2-音频mime-被工具层拦住必须容器化)」里的 11 分钟验收复测（不切段、一个附件）；整段都收下了，音频与视频的切段一起撤掉，回到一个附件 |
| **工具会话的轮次精简** | 进行中（owner 2026-10-01 定方向；2026-10-09 加了去重与检索少轮两件；2026-10-16 做掉 Codex 截断与 pseudo 重发两处重复，单任务会话开场带上任务、默认开） | 工具会话该优于 capsule：同进程修复、按需调工具、持久化状态是它的长处，不为模仿 capsule 而放弃；但要有意砍掉没必要的轮次。今天 agy 上 `per-window` 工具会话一窗比 capsule 未缓存约 1.5 倍、缓存读取约 3.5 倍、思考约 2 倍：输入差额一部分来自多出的轮次、一部分来自开场前缀，思考差额出在作答那一次请求里。候选、怎么量、怎么验见下方同名小节 |
| **带音视频的调用走工具会话** | 未做（owner 2026-10-01 记） | 今天工具协议只管文本，带媒体部件的调用一律走 capsule（[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§7 未做与闸门](llm_agent_tool_protocol.md#7-未做与闸门)）。出厂 `--llm-media` 是 `audio`，而能吃媒体的只有 agy，所以绑到 agy 的纠错窗在默认的 `audio` 下**从不**走工具会话——同进程修复、按需工具、持久化状态这些长处都用不上，答案通道畸形（思考泄漏）也只能靠续会话的修复轮。要补的：任务清单里给媒体一种块（受控 project 内的本地路径，模型用 `view_file` 读）、媒体预处理（音频切成不超过 5 分钟、各带 1 秒黑帧的 mp4）挪进工具会话的输入准备、工具会话那对 project 的读取边界覆盖媒体目录。与「工具会话的轮次精简」同一原则：媒体路径随首个任务一起给，不为它多开一轮；与「数据面迁移」相关：媒体是又一种按需读取的块，两边的块种类要一起设计。**做之前要量**：直接把音频随任务交过去，和让模型自己用工具读，行为有没有差别。owner 记得测到过：agy 用工具读进来的媒体，在工具调用结束、进入下一个用户轮时就被释放；后果是续会话时媒体已不在上下文里，媒体之后的文字往前拼接、缓存失效。文档里能对上的只有一半：agy 会话在首次回复后插入 `CHECKPOINT`，把媒体挪进 `.tempmediaStorage` 改用 artifact 引用（[`llm_local_agent_agy.md`](llm_local_agent_agy.md) [3 视频：分辨率不可调，现有 token 公式失真](llm_local_agent_agy.md#3-视频分辨率不可调现有-token-公式失真)），同一次调用的工具循环里媒体不卸载、不打断缓存；跨用户轮、续会话之后的那一半没有记录。这决定修复轮（续会话）能不能再看到媒体 |
| **准备任务解析失败改走修复轮** | 低优先级，有空再做（owner 2026-10-01：触发面低） | 准备任务的输出解析失败时，今天是同一份 prompt 原样重问（最多 5 次，agent 不接受降温），不带上一版答案与错误说明；修复轮只有纠错窗有。本机 188 次准备 / 研究调用里解析失败 3 次（约 1.6%：2 次漏 `<requested_entries>` 块、1 次 JSON 少逗号），没有一例把重试用光。这类调用输入中位数约 6.7 万 token，重问一次整份重发；但修复轮在 API 路上还要多带上一版答案，省不下钱，只是成功率更高。**触发条件**：同一任务连续重问都失败（某模型固定漏某块），或失败率明显上升 |
| **capsule 每窗从同一条基会话分叉，让固定前缀吃上缓存** | 大半已查清（2026-10-16/17）；剩 Claude Code、CodeBuddy 的 capsule 路没量 | **目的是泛用**：capsule 每窗开新会话，固定部分（CLI 系统提示、driver 框架句、纠错 system 消息约 1.7 万字符、user 消息开头的全局 context pack 约 4.8k 字符，一条 run 内逐字相同）能不能被后面的窗吃上。各家跨会话能读到哪，见 [`llm_local_agent.md`](llm_local_agent.md) [14.1 Agent 细节由 owner 拍板](llm_local_agent.md#141-agent-细节由-owner-拍板不鼓励用户自行折腾2026-08-14) 末尾「跨会话的前缀缓存各家能读到哪」。已查清的：**Codex** 从父会话 `codex exec --ephemeral fork` 沿用父会话的 cache key，10/10 读到整段前缀（持久 fork 换了新 id，3 次只命中 1 次），工具会话已按此接线（[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [各 driver 接线](llm_agent_tool_protocol.md#6-各-driver-接线)表下 `fork` 形态）；capsule 路的修复链靠 `exec resume`，ephemeral 子会话续不了，所以 capsule 不 fork。**agy** headless 不能分叉（`--print` 拒绝 `/fork`）；手工复制会话库能续上历史，但缓存读 0，缓存跟着会话走。**CodeBuddy**（DeepSeek-V4.1-Flash）每次新 cwd 只读到 2,944（它自己系统提示的前段）；固定 cwd（默认开，capsule 也用）之后读到约 1.6 万，但 capsule 的固定部分是不是也落进缓存没量。**Claude Code** 的 capsule 路可以照工具会话的做法，把固定部分放进 `--append-system-prompt-file` 并加 `--exclude-dynamic-system-prompt-sections`，没量。生产默认里走 capsule 的主要是 agy 的带媒体窗，而 agy 跨会话一点都不读，所以眼下收益很小。**触发条件**：要压 Claude Code / CodeBuddy 的 capsule 成本，或接入一家断点式缓存的 driver |
| **数据面迁移（索引必读 + 自主 query）** | 明确暂缓，独立项目（owner 2026-10-01 再确认） | 今天喂给模型的资料（上下文包、知识词条、修复上下文）由 harness 事先拼进 prompt；迁移后只给索引，正文由模型按需拉取。知识库那一半已落地。⚠ 那个「独立项目」的设想在维护者的本地计划里（不随公开快照发布）。量级与 B 步相当，禁止夹进控制面补丁；真做时要遵从「工具会话的轮次精简」的思路——拉取能批量、每一轮都有理由，验收加「每窗请求数不高于迁移前」。顺序与闸门见 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§7 未做与闸门](llm_agent_tool_protocol.md#7-未做与闸门) |
| **动态 driver（进程外版本化协议）** | 剩余架构项 | 长驻 worker runtime 其余部分已完整；见 [`llm_local_agent.md`](llm_local_agent.md) [§12 模块边界与分阶段实施](llm_local_agent.md#12-模块边界与分阶段实施) |
| **agy 原生搜索的 URL provenance** | 继续暂缓，低优先级（owner 2026-10-01；2026-11-01 前不再看） | `--retrieval native` 已通（[`llm_local_agent_agy.md`](llm_local_agent_agy.md) [6 原生搜索：第二个 project](llm_local_agent_agy.md#6-原生搜索第二个-project2026-08-15-打通)）。2026-10-01 在 agy 1.2.14 上复查：`search_web` 那一步仍只有查询词、不报结果 URL；模型用 `read_url_content` 打开的页面会带 URL （一次真机探针：20 次搜索 `urls` 全空，2 次打开页面都有 URL）。所以 agy 能证明「查了什么、打开了哪几页」，不能证明「搜到了什么」；Codex/Claude 每次搜索都带结果 URL。做 native/local 检索质量对照时要把这条不对称算进去 |
| **Agent cache ping** | 搁置（owner 2026-10-01） | 主动发请求给 provider 会话保活缓存。现有数据里没有需要它的情形：同类连续 task 下 agy 与 Codex 的 `resume` 历史大多命中缓存，试跑里窗口一个接一个、TTL 从没起作用；agy 的缓存不跨会话，ping 救不了新会话；CodeBuddy 的 `resume` 历史在 2.156.0 及之前因它自己的 bug 不命中，2.164.0 上 hy3 已整段命中（DeepSeek / GLM 未重测）。**触发条件**：同一会话的两次调用之间出现超过该家 TTL 的空档（等 API、等人），或量到 compact / 重连之后缓存明显丢失。task heartbeat 不承担 cache 保活。[`llm_local_agent.md`](llm_local_agent.md) [§13 长期基线与明确暂缓的能力](llm_local_agent.md#13-长期基线与明确暂缓的能力) |
| **CodeBuddy 思考撞上限后续写从头重想** | 待复测（触发条件 2026-10-10 已满足） | 2026-09-30/10-01 在 WorkBuddy 附带 CLI 上测得：一次请求写满约 48000 token 时 CLI 自己续写，但被截断的思考对模型等于丢了；思考约 15 分钟还会被网关的流超时打断。期间按现有逻辑重试与报错，缓解靠降思考强度、调小窗口字幕上限或换模型（[`llm_local_agent.md`](llm_local_agent.md)「12.1.5 codebuddy」）。**触发条件**：已满足，2026-10-10 换成了独立 CLI 2.164.0。复测用同一个 180 条窗口、强制 `high` 跑一次，模型要用 DeepSeek-V4.1-Flash 或 GLM-5.3-Flash（hy3 在 `high` 下不撞，测不出），花积分，跑之前问 owner |
| **CodeBuddy 的模型在工具会话里不调 `submit`** | 已观测，未做（2026-11-02 记） | hy3 与 DeepSeek V4.1 Flash 在单任务工具会话里有时把整份答案写成正文、不调 `submit`；harness 重开一次会话，还不交就报 `AgentRuntimeCallError`，这一窗没有结果。2026-10-31 的两组 A/B 里见了三格：hy3 与 DeepSeek 各一格，两次会话都没交；另一格 DeepSeek 前三次会话没交、第四次交了。hy3 更早的同类见 [3.8 会话档位与传输试跑](llm_local_agent_experiments.md#38-会话档位与传输试跑q1q32026-09-30每臂-n1)。受影响最大的是把 `codebuddy-basic` 组绑到 efficiency 的预设。先查它为什么不调（会话记录里调没调过别的工具、工具列表在不在）；可能的方向：退回时明说答案只认 `submit`，或正文里是一份能过校验的完整答案时当作交付。 |
| **Claude Code / CodeBuddy 设置文件里的路由键** | 未做，今天够不着（2026-10-01 记） | 两家隔离时（`--setting-sources ""`）丢掉 `settings.json` 的 `env` 块里的网关变量（`ANTHROPIC_BASE_URL`、`ANTHROPIC_AUTH_TOKEN`、模型映射）与 `apiKeyHelper`，与 Codex 曾经丢 provider 同形（[`llm_local_agent.md`](llm_local_agent.md) [§14 当前实现与迁移注意事项](llm_local_agent.md#14-当前实现与迁移注意事项) 讲 Codex 隔离的那段）。**触发条件**：有用户给这两家配了自定义网关。做法照 Codex：只读路由键、token 走子进程环境变量 |
| **目录没法声明「关掉思考」** | 已观测，未做（2026-10-24 记） | 目录的 `thinking=false` 只是「不发思考参数」（给见到未知字段就 400 的自建端点用），可 DeepSeek `deepseek-flash` 与 Anthropic `claude-haiku-5-5` 不发参数时默认思考：小探针上不发参数分别输出 547 / 528 token，发 `thinking: {"type":"disabled"}` 是 1 / 5，DeepSeek 发 `reasoning_effort: low` 仍有 497。缓存验收时这两家把 32k 输出上限全用在思考上，请求轮连续截断。要补的是目录里一个「关掉」的写法，由各方言发自己的关闭参数（两家都认 `thinking: {"type":"disabled"}`）。**触发条件**：要在这两家上跑不思考的配置，或补跑 Anthropic 缓存验收 |
| **Codex 的全局 `AGENTS.md` 结构性挡住** | 凭据走环境变量的连接已解决（harness 自己的 `CODEX_HOME`，默认开，见 `llm_local_agent.md`）；ChatGPT 登录的连接未做，做法已定（owner 2026-10-17：为了干净，不把个人规则带进工作者） | ChatGPT 登录今天留在用户自己的 home，靠开场一句兜底；协议父会话也建在那里。做法：一个固定位置的 FineSub 专用 Codex home（不在系统临时目录、不随协调域变），用户设好 `CODEX_HOME` 指向它、在里面**单独 `codex login` 一次**：独立授权有自己的 refresh token，与用户 `~/.codex` 互不轮换。**不复制 `auth.json`**：ChatGPT 的 refresh token 一次性，复制后先刷新的一份会让另一份失效，用户被登出；硬链接能避开轮换，但退出再登录会拆开链接、凭据在钥匙串时不通，不如单独登录。要改的：harness home 的检查放行 `auth.json`；调用前查那个 home 有没有登录（`codex login status`），没有就照今天用用户的 home；登录过期或被撤销时报清楚、告诉用户怎么重登；`llm_local_agent.md` 写登录步骤。收益：工作者读不到全局 `AGENTS.md`、FineSub 的父会话不进用户的 Codex 历史；省 token 很少（fork 后 `AGENTS.md` 已在父会话前缀里按缓存读）。真机验收要用户先登录一次 |
| **CodeBuddy 的用户级 `CODEBUDDY.md` 挡不住** | 没有干净的办法（2026-10-17 实测） | `~/.codebuddy/CODEBUDDY.md` 每次都进上下文，现靠开场一句用户规则提示兜底（`docs/manual/agent.md` 写明了）。试过的：`--setting-sources ""`、`--system-prompt` 都挡不住（2026-09-30）；`CODEBUDDY_CONFIG_DIR` 指向空目录也挡不住，CLI 照样从系统 home 读这个文件（DeepSeek-V4.1-Flash，2/2，登录照常）；`USERPROFILE` / `HOME` 指向空目录挡得住，但登录也没了：认证文件按 `USERPROFILE` 下的 `AppData/Local/CodeBuddyExtension` 找，CLI 退出码 0、答案为空、用量 0（2/2）。要走这条只能把凭据挪进去，和 Codex 复制 `auth.json` 同类风险，不做。**触发条件**：CLI 加了不读用户级记忆的开关，或认证路径不再跟着 `USERPROFILE` 走 |

## A/B：prompt 与给模型的信息

要按 [`prompt-iterate.md`](prompt-iterate.md) 的协议做对照实验才能回答的问题：prompt 怎么写，以及给模型哪些信息。

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| **研究轮「来源无法核对」的说明写着「agent 后端」** | 下次冷读改（2026-11-09 记） | `research._mark_unverified_sources` 给无法核对的 `sources` 加的 `sources_note` 写「本次应答的 agent 后端不回报检索结果 URL」。自定义 `openai_compat` 端点用 `search_request` 搜索时也会走到这里（它的检索记录没有 URL），而它不是 agent；改成「后端」即可。这句可能进下游 prompt，所以随下一轮冷读改，不单独改 |
| **占位答案的两条修复报错没冷读过** | 下次冷读读（2026-11-09 记） | 纠错窗的占位答案检查（移植自 Nonoka X，见 [`llm_harness_behavior.md`「重试与拼接」](llm_harness_behavior.md#重试与拼接)的「占位答案」一条）新加了两份报错模板 `fragment_uniform_rows_error_v1.md` 与 `fragment_placeholder_rows_error_v1.md`，只在修复轮出现，写好后没走冷读，随下一轮一起读 |
| **可选请求通道被用得太少** | **低优先级**（owner 2026-09-30）；待测（2026-09-09 验收带出，owner 已裁定先合） | 那次验收的读数：任务式 21 次 run 里只有 1 次真的请求，而 p1 的相似度贴线 0.002、方向偏低。2026-09-30 none/native 也有了只请求词条的通道后，flash-lite 在 `retrieval=none` + 可读知识库上跑了 3 次（1 次 fast、1 次 5 窗、1 次 fast 且备注点名了主播与游戏，索引里有对应条目），2.5 Flash 在 `retrieval=native` 上跑了 1 次 fast，8 个窗口都直接交付、一次请求也没有——同一现象，样本小。三条素材都答不了「可选通道被用得太少，会不会在依赖外部信息的素材上劣化」。⚠ **代码里没有任何缓解**，也没有对照臂可跑了（旧流程随阶段 D 删了）。要答就得按那份验收的下一步（维护者的本地计划里（不随公开快照发布））：找一条确实依赖时效／冷门信息的素材，或按 prompt-iterate 协议 A/B 两种改法（`local` 下要求首条回复先请求；prompt 里把「不认识的专名必须请求」写硬） |
| **被插话隔开的同一个人的两行合并** | 后做（owner 2026-10-30：等按话轮排行的读数出来，先派 subagent 研究难度与做法） | 云端任务里 A 说着一句话、B 插一句附和、A 接着说时，A 的两行中间隔着 B 的行，纠错只能合并连续的源，所以不并（`换回` 标的就是这一处）。要并就得放开「合并源必须连续」，合并行的时间、`gap` 的含义（见上一行）与三个 pass 的分组都要跟着改；纠错带说话人列的译文连贯性收益（输入只读、给裁判起的名字）到时一起 A/B |
| **pass 2 跨过没标可并的接缝合并** | 记一笔（2026-11-05，字幕长度档位验收） | 三个 pass 的 pass 2 跨过 pass 1 没标可并的接缝合并时只记 `turn_violations`、不拒收。档位验收 12 臂里只有 hina-17 偏长出现，4 处：3 处是 pass 1 漏标的同一个人的半句（并了反而顺），1 处是 pass 1 标了可并、但并完 10.3 秒越过偏长的绝对门槛、被脚本撤回的接缝，pass 2 照样并了，交付里多出一条越绝对门槛的行。要不要拒收撤回的那一类（漏标的那类拒收反而变差），看更多运行里出现的频率再定；读数在维护者的本地报告里 |
| **stabilize 的段级 `tags` 进纠错判据** | 搁置（owner 2026-09-30） | 纠错窗输入今天不带 stabilize 给段打的 `tags`（`高度疑似幻觉`、`套话幽灵`、`语言切换幻觉`、`时间漂移` 等，见 [`asr-stabilize.md`](asr-stabilize.md)「Profile 2：高噪音标记」）。值得试的是把它们交给模型当纠错判据；要动 `<asr_result>` 契约与 `PROMPT_VERSION`。「第二模型救回、纠错又丢弃」不算问题（owner 2026-09-30：被丢的是低价值的声音），不为它补第二模型证据 |
| **intermediate 走单 pass 的变体：吸收说话人裁判与校对** | 记一笔（owner 2026-11-08 定的方向，没开始） | 现在 `intermediate`（默认档位）等于三个 pass 关掉裁判与校对、前两步合成融合单 pass（capableC）。方向：研究单 pass 能不能把校对吸收进来、在单 pass 里做说话人裁判；做成后 `intermediate` 是单 pass 的变体，与三个 pass 真正分开，两边仍共用 prompt 模块。要过的坎：校对靠独立会话重读成品才找得到前两步漏的错，先按类别拆 pass 3 历来改了什么，专名与读速超线这类可考虑确定性检查加定点修复；裁判并进单 pass 要看跨人合并与输出量，前身是[路线二](#说话人裁判的另两种接法)（认人并进纠错窗，2026-10-08 纠错质量明显下滑，重启前提在那一节）。门槛事先定：行真值量跨人合并与换人标记，加上成本上限（不超过现在 `intermediate` 的某个倍数，两边都连修复轮一起算：单 pass 首次交付不过的比 pass 2 多）。结构上档位仍只管定位，pass 的形状改 `correction_mode` 里按档位的默认。成本参照（2026-11-08，从已有运行算；按价格加权：未缓存输入 ×1、缓存输入 ×0.1、输出与思考 ×6）：agy Gemini 3.8 Flash 上 `quality` 的三个 pass 约是单 pass 的 1.4–2.9 倍（三个窗；单 pass 取 gap 方向 A/B 的三臂各一遍，三个 pass 取 11-04 至 11-08 的出厂默认风格、长度档位、gap 方向三组运行，每窗 pass 1 三遍、pass 2 八遍、pass 3 两遍；逐窗 0.8–3.6 倍：单 pass 进修复轮时差距小），三个 pass 各占约三分之一、pass 3 几乎全是思考；GPT-6.1 Sol 上推算约 1.6 倍（裁判 + 纠错对单 pass 1.36 倍来自 10-31 的四个窗，pass 3 占一成多来自 10-24 的别的素材，Codex 父会话的开销没算） |
| **说话人分离的 `auto`** | 预留（owner 2026-10-07、10-09）；2026-10-17 先不做、以后要再看 | `--diarize` 之外的另一个开关：开了时由 harness 在不超过 `--diarize` 的档位里自己选（要不要说话人、要不要裁判、要不要听音频）；部分判断可以靠单独的 LLM 任务，或在准备任务里顺带分析。没有选项，也没有代码。2026-10-17 默认开（跑到 LLM 阶段的 `quality` 档默认 `judge-audio`）时讨论过「只分出一个人就不跑判人的裁判」：按声学分轨定不行，单人的游戏实况（kaguya60，60 分钟）被分成用量相近的三条轨；在准备任务里判人数可行，好处是后面的任务更简单、模型注意力更集中，token 约省一成（验收里不带轨号的 pass 1 对带轨号的约 1.33 对 1.50 倍）。限制：快速模式没有准备任务；游戏实况、背景人声会判错（判错的面小，以前本来就没有轨号）；三条声学轨仍在，降到 `acoustics` 还是 `off` 要再看 |
| **说话人裁判的另两种接法（合并进纠错窗 / 多个 pass）** | 路线二（合并进纠错窗）待探索（owner 2026-10-08）；路线三已做成三个 pass（2026-10-17，`quality` 档默认走），还开着几件 | 路线二的读数与重启前提、三个 pass 之后还开着的几件见[说话人裁判的另两种接法](#说话人裁判的另两种接法) |
| **中文只纠错的余项** | 记一笔（2026-11-07 采纳两列） | 现行行为在 [`llm_harness_behavior.md`「只纠错」](llm_harness_behavior.md#只纠错)，取舍理由在 [`llm_design_notes.md`「中文只纠错的决策记录」](llm_design_notes.md#中文只纠错的决策记录2026-11-0507)。还开着：（1）**润色力度不稳**：quality 档上 `polished` 改了的行各窗从 0% 到 68%，同一份素材同一配置两次运行一次改 51 行、一次改 1 行，改得少时字幕里留着「那个」「呢」（新字一直在 0.4% 以内，不是书面腔的问题）；要收，先在两三份素材上各跑几次量清波动，再看 prompt 收不收得住，不加机器校验（owner 2026-11-06）。（2）**中文任务的知识库**（押后，owner 2026-11-06）：读哪些条目、存什么（中文专名的正确写法、主播）、知识更新的 prompt 怎么写；现在一律不读写，设计时从条目结构里的「译名」说起。（3）显式 `--llm-proofread on` 照跑翻译的校对 prompt（写的是日语原文对译文），效果没量过。（4）corrected 里台湾说法偶尔换成大陆说法（「萤幕」→「屏幕」「伯克莱」→「伯克利」，每次运行 1–3 处），单列时也有 |
| **英文源的翻译风格与校对 prompt 里的日语专属内容** | 记一笔（owner 2026-10-24：英文源出什么字幕不是本次重点） | 出厂默认风格里日语专属的三条已改成方向性写法、日语只作例子（owner 2026-11-04：不同语言的译法放在同一份文件里，只要是方向性的）。还剩：校对 prompt 写死的「日语原文」改中性写法；出厂默认 `[意译适度]` 的「句式能跟就跟、宁可中文略不顺」对英文源是否会成翻译腔，等有英文素材再看。其余日语专属内容（假名规则、日语示例、说话人裁判的日语举例词、音近提示、裁判补行的语速 `CHARS_PER_SECOND`）在中英上不会让输出出错，不改 |
| **垫词长度 `filler_max_chars` 按源文 `len()` 数** | 记一笔，没量（2026-10-24） | 「三行夹一个垫词」的合并要求垫词源文不超过 3 个字符（`llm/prompt_constants.py` 的 `BASE`，不随长度比例缩放），英文的 "uh-huh" 是 6，过不了；按加权字数或按语言给值都行。英译中的长度读数没靠它，所以没数它在英文源上挡掉了多少 |
| **efficiency 档的读速超线比出厂默认之前多** | 记一笔（owner 2026-11-04：主要面向 quality 优化，efficiency 只做兼顾，以后适时再想办法） | 纠错与校对 prompt 里的风格要求挪进出厂默认风格之后，3.5 Flash Lite、`basicA` 上 hina-17、hina-24、manosaba 三窗合计的读速超线（`字/秒` 超过 9 且译文 10 字以上）基线三遍 36 / 38 / 38、新版两遍 50 / 52（串行的格作废），重复（量的是带辅助列的十列 `basicA`，v104 去掉辅助列之后没重量）；多出来的多是译文加修饰、说得更满，DeepSeek 在命中处偏向基线。quality 档（agy Gemini 3.8 Flash）两臂差不多。试过把「不过度改写」拆成单独一条 `[不过度改写]`、`[语序]` 补回软化词：读速仍是 43 / 53，迁移时 DeepSeek 反而把整条 `[不过度改写]` 加进要求重写的用户风格，已撤回。以后的方向：efficiency 自己的通用规则里补一句「别加料」（只在 efficiency 生效，不动风格），或给 `basicA` 的译文按源的时长给字数上限。逐窗读数与样例在维护者的本地验收报告里 |
| **`basicA` 的短行一闪而过** | 记一笔（owner 2026-11-02：删 `basicB` 时不移植它的合并） | efficiency 的 `basicA` 一行对一个源，只接回词中切断；ASR 切得太碎时，0.1–1 秒的短行（「普通に」「はいでは」）或中文挂在半空的半句照原样出。2026-11-02 的对比里两位读者共同判只 `basicA` 错的有 5 处（两臂断法不同的 74 处里），`basicB` 在那里都并对了，但它别处并错的多得多（只 `basicB` 错 32–33 处，被并接缝 33% 跨人）。可能的方向：不靠模型判断、在代码里按时长、字数、停顿与声学主轨把极短的行接上；或给 `basicA` 加一条带硬上限的短行规则，要再比一轮、看跨人合并。**触发条件**：有人反馈 efficiency 的字幕一闪而过，或 efficiency 用得多起来。 |

## 知识库

其余知识库遗留项在 [`knowledge.md`](knowledge.md) 末节，这里只放牵涉 harness 或要 owner 定方向的几条。

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| **小节上限说明「超了整批拒绝」在知识更新里说重了** | 下次冷读改（owner 2026-11-09） | `render.py` 渲染给模型的小节上限说明写着「写入时会校验，超了整批拒绝」。维护会话里仍是整批回滚，这句对；知识更新现在只拒收出问题的行、其余照常提交（现行行为在 [`knowledge.md`](knowledge.md) 的 style 类别一节，搜「整批校验不过时」），这句说重了，模型会略保守，没有害处。改成两边都成立的说法，随下一轮冷读一起读 |
| **候选修复的裁定与提案只靠 `@c` 句柄对上** | 记一笔，先看频率（2026-10-19，prompt 冷读第三轮） | 知识库修复会话里，`<candidate_verdicts>` 判 `propose` 的候选，靠模型在对应提案的 `reason` 里写上候选句柄（如 `@c1`）供人工追溯，没有校验：裁定说提了、却找不到带句柄的提案，或反过来，都不报错。**触发条件**：生产的修复会话里看到这种脱节，再定是加校验还是改块结构 |
| **Agent 直接编辑知识库** | 低优先级，有空再做（owner 2026-10-01） | 今天 agent 只能提交 proposal / update，由 harness 校验后写入。要做的形态：agent 直接改知识库（隔离的 worktree，见 [`llm_local_agent.md`](llm_local_agent.md) [§13 长期基线与明确暂缓的能力](llm_local_agent.md#13-长期基线与明确暂缓的能力)），**收取标准与人直接编辑一致**——按收割人改 `knowledge/rendered/` 投影的同一套规则判收。验收：与 proposal / update 路对比成功率与 token 效率 |
| **auto 选风格的效果** | 有触发条件，未做（owner 2026-10-01：效果验收不挡翻默认，改成有数据之后再看） | 触发条件：攒到一定量的 auto 运行——库里有多套候选、`style_selection` 产物里各套都被选中过。到时按首版设计里「实施顺序与验收」的形态评估（选取是否符合预先标注的意图、规则在译文里是否落实、同片是否一致），再决定要不要优化。数据来源：各任务的 `style_selection` 产物与 `style_injected` 使用率 |
| **dry-run 与 `tools/session_replay` 不预演 `--style auto` 的选择** | 未做，记一笔（验收审查 2026-10-02） | 选风格发生在准备任务（普通模式）或整段任务的请求回合（fast）里，dry-run 不调模型，所以两条路都看不到：普通模式写出的窗口 prompt 里没有风格块，fast 的「先选后译」开场不进 dry-run 的产物；`session_replay` 重打的也是选定之后的调用。`--style auto` 翻成默认之后，这影响到**所有不给 `--style` 的 dry-run**——它们展示的 prompt 比实际运行少一块。**触发条件**：有人拿 dry-run 的 prompt 当实际 prompt 审（风格相关的 prompt 迭代），或要用 replay 复现一次风格选择 |
| **auto 没选中专门 style 时建议新建 style** | 可选，低优先级（owner 2026-10-01：暂不做自动新建） | 今天新建 style 只有人手动 `knowledge new style`，更新任务只能改本次选中的那套。可做的形态：auto 下模型选了 `default_style`（或库里没有它时选了「无」）、且 `--style-mode update` + 有精修字幕时，把这次观察到的成类做法作为一条「建议新建 style」放进人工裁定队列，不自动建——一套 style 对应一个字幕组或译者，名字与适用范围只有人知道。它也缓解另一件事：auto 下对不上专门那几套的素材会选 `default_style`，选中即可写，学到的东西都进 `default_style`。**触发条件**：`--style auto` 成为默认之后，`default_style` 里常出现只适用于某个字幕组或题材的约定 |
| **索引里标出长条目（≈2x、3x）** | 有触发条件，未做（owner 2026-10-02） | 给模型看的索引行在条目名后标相对长度（单位 2000 token，即注入块里平均分给一个条目的预算；≥4000 才标），让一次性请求词条时知道哪条贵。只出现在给模型看的索引面上：索引文本还会被解析回条目名、用于关键词预注入与说话人名册，混进标记会把长条目的名字解析错。**触发条件**：出现因整批长度太长导致的截断（注入报告里条目被削，原因是块满而不是单条超过上限） |
| **拉取时「两边都挪过同一行」交给 AI 判** | 有触发条件，未做（owner 2026-10-02） | 一行在本地被挪到别的条目下（人手、知识更新合并或拆分），共享库之后又把它挪到第三个地方时，拉取保留本地位置，摘要里记一笔「local kept: N line(s) currently moved both here and in the library」。可做的形态：记进 `share-conflicts.jsonl`，`share conflicts --repair` 的 AI 只给二选一裁定——跟随库里由 harness 以拉取写入执行（让 AI 自己 `move_line` 会被当成本地挪动，库里下次再挪又会被问一遍），保留本地记 `dismissed`；冲突身份只用父条目 id，判定必须三方比较（库里自上次拉取后挪过且本地也挪过），否则推送只加不减会让每次本地合并都冒出冲突。**触发条件**：拉取摘要里出现这句提示。它数的是现存的分歧，同几行在之后每次拉取都会再出现，所以看的是有没有，不是次数 |
| **按节请求词条（`key#节名`）** | 有触发条件，推迟（owner 2026-10-02） | 让 API 的请求块写 `条目#节名` 只注入那几节。要动请求解析、载入、keep 透传、resume 哈希、并行台账六七处；拆分之后 `common` 的长节本身就是能按名请求的子条目，大部分用途被覆盖。**触发条件**：注入报告里不能拆的条目（主播、style）经常被削 |
| **共享推送只加不减** | 未做，记一笔（2026-10-02 修拉取侧时发现） | 服务端合并推上来的包时，membership 只新增、不移除旧父条目下的那一条，推送也表达不了删除。所以本地把一行挪到别的条目（跨条目 `move_line`）再推送，共享库里两个条目下都有这一行，别人拉下来也是两边都有，直到维护者清理。反方向（库里挪了、本地拉取）已修。另一个可能的同源问题（读代码推断，未证实）：推送一行后、首次拉取之前维护者就把它挪到别的条目，拉取找不到本地那一行（推送时它还没有 canonical membership id），会在新条目下另建一份。修它要让推送包能表达挪动与删除，并接进投稿审核。**触发条件**：共享库里出现这种重复（条目拆分上线后会更常见；拆分不等这一项） |
| **类别多了以后，维护任务改为先看模版描述再选** | 有触发条件，未做（owner 2026-09-06：模版数量证明有必要之前不做） | 维护任务今天把每个类别的完整模版都给 agent（[`knowledge.md`](knowledge.md)「加一个类别就是两步」）。类别多到全量模版明显占预算时，改成两阶段：先给各类别的一句话说明（preset 里已有的 `blurb`），agent 选定后再取那一份模版。数量没到之前多写这一层是纯负债 |
| **新作品的主体条目进共享库没有冷启动路径** | 记一笔（owner 2026-10-03） | `approve_tentative` 只收纯新建 term（[`knowledge.md`](knowledge.md)「tentative 分发与 digest」），新作品的主体条目不是 term 节点，没有外部佐证时只能 reject，没有先以 tentative 入库、在影子匹配里攒佐证的路。触发条件：真出现新作品条目因此进不了共享库 |
| **预注入条目默认透传到窗口** | 有触发条件，未做（owner 2026-10-03） | 今天预注入条目要模型在交付的 `<keep_entries>` 里再写一遍才会透传；这次只把 prompt 里 keep 的 key 来源改到与实现一致。触发条件：产物里出现该透传却没透传的条目 |
| **词条瘦身翻成默认开**（`FINESUB_KB_SLIM`） | owner 择时（owner 2026-10-16：默认先关） | 现行行为与读数在 [`knowledge.md`](knowledge.md)「精修字幕怎么读」的词条瘦身一段。翻默认要做：把 `kb-phonetic` extra 并进 `harness` 与运行时锁（已有安装整个重建运行环境，约 330 MB 增量，大头是 sudachi 词典）、写 `CHANGELOG.md`、决定要不要给用户一个开关。没测过、可能改变结论的：只压描述而保留散文行的形态（省得更少、信息损失也更小） |
| **精修稿长短的提示（只提示、不写档位）** | 有触发条件，未做（owner 2026-11-05） | 知识更新不再按实测写风格的 `[字幕长度]`，只按用户的 `--task-summary` 改（[`knowledge.md`](knowledge.md)「style 类别」的字幕长度档位一条）。可做的形态：带精修字幕的知识更新跑完，在报告里打一行「精修稿每行比这次交付长约 N%，要长一些可以给风格写 `[字幕长度] 偏长`」，不改库。要量的是两份字幕同一素材上每行加权字数与秒数的 p90 之比（删掉的那条实测路的做法，原文在分支 `feat/length-tier` 的 `dc9eb6f4`）。触发条件：有用户反映不知道有档位、或精修稿长短与风格档位长期不符 |
| **知识库遥测卷积（`events` 表）** | 有触发条件，未做 | 见下方同名小节 |
| **知识库里仍只能人工的环节**（待放开） | owner 原则（2026-10-02），不做 | 见下方同名小节 |
| **纯拉丁词条按词边界匹配** | 记一笔（owner 2026-10-24：多语言知识库本次范围外） | 纯拉丁字母的词条现在按子串匹配，英文源上会命中词中片段。要的是两侧不能紧贴拉丁字母或数字（不能用 `\b`：「Apexやろう」会漏）；会改日语素材的注入集合，改前要离线比一遍 |

## 前端与分发

| 项目 | 状态 | 现状与去处 |
| --- | --- | --- |
| **`finesub` 任务记录的其余字段要不要也只记显式值** | 记一笔（2026-09-30） | LLM 的五个开关已改成只记命令行写了的（预设会填没写的，打包 CLI 读不到预设）；device、model、gpu_tier、`llm_fast` 等其余字段仍按代码默认写全，那是为已退役的 0.4.x 桌面端从记录重试留下的。owner 2026-09-30 说过不用再照顾旧版桌面端，所以这层理由已经没了；改的话 `_recorded_request` 整段镜像可删，但下游读者会少拿到这些值。**触发条件**：有人要改这份记录，或镜像又和管线默认值对不上 |
| **下载侧被墙时的其余缺口** | 记一笔（2026-11-09） | 地区探测失败时判成 `global` 且不写缓存（下次再探）；uv、git、ffmpeg 从 GitHub 直连。都打印进度或报错、不是静默的，所以没进连不上判定那一轮（[`download-routes.md`「1. 路线解析」](download-routes.md#1-路线解析)）。**触发条件**：用户报告其中一处等得久又看不出原因 |
| **tokcount 的 `workflow_dispatch` 发布还没跑过** | 记一笔（2026-11-09） | `0.7.0-gemma3-0` 是用测试分支那次运行的产物手工建的 Release（workflow 不在 `main` 上时只能这样，做法在 `tools/tokcount/README.md`）。`publish=true` 那一段要等 workflow 随下一次发布进了 `main`、下一次重编 tokcount 时才第一次跑 |

## 明确不做（owner 裁定，别再当待办）

每条一行：不做什么 → 它的决策记录。理由与重开条件只写在记录里，重开之前先读。

- 纠错窗口尾部带只读后文、重叠区不再重答；或让后一个窗口按需接管切点前几行 → [`llm_design_notes.md`「纠错窗口拼接改成 core 归属的决策记录」](llm_design_notes.md#纠错窗口拼接改成-core-归属的决策记录2026-11-08)
- 纠错输入去掉 `gap`、让模型自己算停顿 → [`llm_design_notes.md`「纠错输入 `gap` 方向的决策记录」](llm_design_notes.md#纠错输入-gap-方向的决策记录2026-11-0708)
- 只来自网页的知识提案先写成 tentative → [`llm_design_notes.md`「只来自网页的知识提案不先写成 tentative 的决策记录」](llm_design_notes.md#只来自网页的知识提案不先写成-tentative-的决策记录2026-11-07)
- 把行内补漏没收下的 Qwen 片段交给纠错去判 → [`llm_design_notes.md`「行内补漏没收下的 Qwen 片段不交给纠错的决策记录」](llm_design_notes.md#行内补漏没收下的-qwen-片段不交给纠错的决策记录2026-11-02)
- 再改 prompt、换模型或加一轮去修带音频说话人裁判的明显的错 → [`llm_design_notes.md`「带音频说话人裁判那类明显的错不再加修法的决策记录」](llm_design_notes.md#带音频说话人裁判那类明显的错不再加修法的决策记录2026-10-3011-02)
- 三个 pass 合成一个由单个 agent 依次做完的任务 → [`llm_design_notes.md`「三个 pass 不合成一个 agent 任务的决策记录」](llm_design_notes.md#三个-pass-不合成一个-agent-任务的决策记录2026-10-2530)
- 在 prompt 里告诉模型保留数下限 `KEPT_FLOOR` → [`llm_design_notes.md`「prompt 里不告诉模型保留数下限 `KEPT_FLOOR`」](llm_design_notes.md#prompt-里不告诉模型保留数下限-kept_floorowner-2026-10-03)
- 去掉 fast 单窗为选风格多开的请求回合 → [`llm_design_notes.md`「fast 单窗为选风格多开的请求回合保留」](llm_design_notes.md#fast-单窗为选风格多开的请求回合保留owner-2026-10-03)
- 把开了反馈采集的准备任务的 `context_pack` 与 `task_update_feedback` 拆成两次输出 → [`llm_design_notes.md`「纠错与准备任务几项不改的决策记录」](llm_design_notes.md#开了反馈采集的准备任务在同一次输出里交-context_pack-与-task_update_feedback不拆owner-2026-10-03)
- 改知识校验一次调用的规模 → [`llm_design_notes.md`「知识校验一次调用的规模不改」](llm_design_notes.md#知识校验一次调用的规模不改owner-2026-10-03)
- 修复轮改成「只给错误、不给上一轮输出」，或以 prompt 侧强化作主解 → [`llm_design_notes.md`「纠错与准备任务几项不改的决策记录」](llm_design_notes.md#修复轮不改成只给错误不给上一轮输出prompt-侧强化不作主解owner-2026-08-15)
- 工具协议 D 步：删 capsule 输入路径 → [`llm_design_notes.md`「工具协议 D 步：不删 capsule 输入路径」](llm_design_notes.md#工具协议-d-步不删-capsule-输入路径owner-2026-10-01-撤回)
- 再投入 `pseudo-conversational` 与 conversational（已冻结）→ [`llm_design_notes.md`「`pseudo-conversational` 与 conversational 冻结」](llm_design_notes.md#pseudo-conversational-与-conversational-冻结owner-2026-10-01)
- P7d：parallel 并发上限的标定实验 → [`llm_design_notes.md`「P7d：parallel 并发上限不做标定实验」](llm_design_notes.md#p7dparallel-并发上限不做标定实验owner-2026-08-30)
- conversational 的 worker 能力申报与逐 task 对账 → [`llm_design_notes.md`「conversational 的 worker 不申报能力、不逐 task 对账」](llm_design_notes.md#conversational-的-worker-不申报能力不逐-task-对账owner-2026-08-22)
- 给 free Gemini 的 `max_input_tokens` 正名 → [`llm_design_notes.md`「模型路由 v2 的持久决策」](llm_design_notes.md#模型路由-v2-的持久决策)里「catalog 里的 194000 不一律是厂商值」那一行
- 接 Gemini 的 agentic video → [`llm_design_notes.md`「Gemini agentic video 不接的决策记录」](llm_design_notes.md#gemini-agentic-video-不接的决策记录2026-09-02)
- 可维护的 Shared Context（跨窗口、模型可改的共享上下文）→ [`llm_design_notes.md`「可维护的 Shared Context」](llm_design_notes.md#可维护的-shared-context)

---

## 视频多模态对质量的帮助

已知的（owner 2026-09-30）：
- 音频多模态在复杂场景、低质量 ASR 上的修复优于纯文本，`--media audio` 作默认不再待证。
- 输出长度主要随**模型本身与推理强度**变化，与开关关系不大，所以输出系数只剩
  `c = 3.5 + 1.0(retrieval≠none)`，媒体不进系数（剪辑只占输入）；不再按开关组合逐个标定系数。

剩下要量的只有一件：`--media video` 相对 audio 对**质量**有没有帮助、值不值它多出的输入成本
（低分辨率每帧 71 token × 0.25 fps，见 `profiles.py` 的 `VIDEO_*` 常量）。

实验纪律：

1. 每个对照臂至少 `n≥5`，并做同配置复跑作为噪声基线。
2. 固定素材、知识库快照、prompt version、endpoint/tier、模型与推理强度。
3. 同时报告产出质量、输入/输出/thinking token、重试和实际墙钟。
4. 素材要选画面确实携带信息的（字幕外的画面文字、看图才懂的指代），否则测不出差别。

## 工具会话的轮次精简

**方向**（owner 2026-10-01）：工具会话应当优于 capsule——同进程修复、按需调知识库与检索工具、持久化的任务状态都是
它的长处，不为了压成 capsule 的样子而放弃；但每一轮都要有理由，没必要的轮次要砍掉。

**现状**（2026-10-01，第二份素材，agy 3.7 Flash，`per-window`，accept 后立即结束 CLI 已生效；同一窗按 agy 自己的
会话记录对比）：

| | 工具会话 | capsule |
| --- | --- | --- |
| 请求 | 4 次：调 `next_task` 领任务 → `view_file` 读 `protocol.md`（37 KB）→ `view_file` 读 `payload.md`（14 KB）→ 写答案调 `submit` | 3 次：读任务文件第一页 → 第二页（`view_file` 单次最多给 46,080 字节，文件 50.5 KB）→ 作答 |
| 第一次请求的未缓存 | 约 1.65 万 | 约 0.5 万 |
| 每窗输入 | 未缓存约 4.6 万、缓存读取约 5.7 万 | 未缓存约 3 万、缓存读取约 1.6 万 |
| 每窗输出（含思考） | 约 1.7–2.5 万，其中前三次请求合计不到 1k，其余都在作答那一次 | 约 0.85–1.5 万，作答那一次约 1.3 万 |

六窗之间这组数几乎不变。整条 run 的读数在 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md)
[3.8 会话档位与传输试跑](llm_local_agent_experiments.md#38-会话档位与传输试跑q1q32026-09-30每臂-n1) 的 Q3 那条。
⚠ 以上是 agy 在开场带上任务之前的读数。Codex（GPT-6 Luna，同一份 5 窗素材，2026-10-16；每窗均值，未缓存 / 缓存读 / 输出含思考）：capsule 1 次请求、3.4 万 / 0 / 3.0k、约 23 s；开场带上任务之后的工具会话 3 次请求、3.2 万 / 7.9 万 / 3.2k、约 29 s（改前 6 次、3.8 万 / 13.8 万 / 3.8k、约 38 s）。未缓存与输出已和 capsule 持平，多出的只有缓存读；按常见定价（缓存输入是未缓存的 1/10–1/30、输出是它的 4–5 倍）折算，工具会话每窗约比 capsule 贵 2–13%（改前 24–44%）；capsule 仍快约 6 秒。这组是 `--retrieval none --knowledge none`，capsule 每窗只一次调用。开 `--retrieval local --knowledge read`（2026-10-16，各 1 窗）：三个纠错窗都选了直接交付、仍是一次调用；准备任务先提请求、回填后交付，回填那次（t2）是**新开的 Codex 会话**、把前面整段重放进一条消息，未缓存 3.2 万、缓存读 0。改成 `resume` 同一会话，会话内缓存大概率能吃上，这条没做。agy 与 CodeBuddy 改后（2026-10-17，本地 agent 缓存复用那批改动都默认开，uchida 10 分钟 5 窗、同一份准备结果，两种传输各 1 次，`--retrieval none --knowledge none`；未缓存 / 缓存读 / 输出含思考）：CodeBuddy（DeepSeek-V4.1-Flash）整条 run 工具会话 4.72 万 / 12.7 万 / 6.13 万、capsule 3.30 万 / 8.92 万 / 5.23 万，第 2–5 窗每窗未缓存 0.46–1.07 万对 0.28–0.37 万，首次接受都是 5/5，折算 capsule 约便宜 17%（输出占大头、n=1，这个数只是方向）；agy（3.8 Flash）两边都记到思考的前 3 窗，工具会话每窗 2.4–2.7 万 / 1.9–4.2 万 / 2.5–3.6 万、capsule 2.8 万 / 2.0 万 / 1.9–3.1 万（另有一窗缓存没命中：4.8 万 / 0），每窗 142–264 秒对 141–222 秒，分不出；capsule 后两窗的调用记下的思考是 0、其中一窗第一次没过校验，原因没查，不拿来比。两家都没有让工具会话比 capsule 明显更贵的差距，不改「工具会话优先」的方向。

**Codex 上看到的**（2026-10-09 GPT-6.1 Sol 说话人裁判验收时，只剩失败或中断的会话可看，请求数是推的；2026-09-30 两条
GPT-6 Luna pseudo 会话打开了 `local_agent_keep_episodes`，用 rollout 逐请求核过）：
一次调用报的未缓存 / 缓存读是会话内所有请求的累计，每轮重发整段上下文。准备任务一次约 9–14 万未缓存、100–190 万缓存读，纠错一窗约 11–12 万、180–230 万；
裁判一窗（约 300 行字幕，与纠错窗对齐后）约 1.2–4.6 万未缓存、7–29 万缓存读；对齐前每窗 150 行、另有一次挑人调用时，字幕只有约 1.3k token 的一次调用也要 1–4 万未缓存，大头是 Codex 自带指令与工具定义的固定前缀。多出来的轮次与重复内容：

- **重读已内联的块：是截断，不是模型不守规矩，已改**（2026-10-16 先分页，2026-10-23 改成直调）。Codex 的 code mode 把工具输出截在 1 万 token、砍掉中间
  （[`llm_local_agent_agy.md`](llm_local_agent_agy.md) [§5 MCP 工具结果的内联上限](llm_local_agent_agy.md#5-mcp-工具结果的内联上限约-4k-字节超过即外置成文件无预览2026-08-22) 末段），
  纠错窗的 `next_task` 约 1.3–1.4 万 token，每窗都缺 3–4k。两条 Luna 会话里，一条每窗用 `read_context` 补读，7 个任务 24 次请求中
  7 次是补读，这 7 次占整条会话输入的 29%（82 万 / 282 万）；另一条改调 `pull_status`，台账说什么都不欠，就带着缺口交了
  （它在前一窗补读过协议，那次没丢东西；`per-window` 每窗新会话，没有这份旧副本）。现在 FineSub 的工具在 Codex 上直调、不过 `exec`，回复整份送到（截在 `tool_output_token_limit` 的 80 万字节，超过 70 万外置成份），
  互不依赖的只读调用经 `batch` 放进一轮；直调设置没生效时送达核对会告警（[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [通道上限](llm_agent_tool_protocol.md#通道上限超限的回复与送达核对)）。
- **pseudo 会话每窗重推同一份 protocol，已改**（2026-10-16）：17,030 字符、同一 digest，7 个任务的会话里 6 份副本留在同一上下文，
  agy 那边是同一份文件每窗重列一次。现在同一条 CLI 交付过、仍在台账里的 protocol 只标 `read: seen`（载荷是任务本身，照发）。
  改后 Luna 与 agy 各跑一条 3 窗 pseudo 会话（2026-10-16，各 n=1）：拿到 `seen` 的任务都没有重读协议（agy 没有重开那个文件），
  全部首次就被接受。这也去掉了一份隐性保障：原先每窗重推，等于 compact 之后自动补回一份协议；现在要靠模型自己发现丢了，再去重读。
- **检索**：准备任务与纠错窗各做 4–9 次 `web_search` / `web_fetch`（每次结果 1–3.8 万字符），之后每轮都带着。在 code mode 下，
  打印出来过 4 万字节的 `web_*` 结果与 `kb_*` 回复也被截过；直调之后不再经过那道截断。

**要做的两件**（owner 2026-10-09，之后做）：
- **上下文里不出现重复内容**：同一块只进上下文一次（内联了就不让再读、或只给索引按需读，二选一）。上面两处已做；单任务会话开场就带上块（候选 1）已做、默认开。
- **联网检索少几轮**：对较强的模型，用 prompt 要求一次把要查的想清、合并检索；或让它派便宜的子代理专做调查，只把结论带回主会话。
  检索客户端已有 `search_many` / `extract_many`，MCP 的 `web_search` / `web_fetch` 却一次只收一个 query / URL——工具收数组，一轮就能查几条。

**怎么权衡**（owner 2026-10-16）：
- 一轮的代价看它发生时上下文有多长。开场附近上下文还短，首轮的缓存读很便宜，多一轮问题不大；贵的是大块已进上下文之后
  每多一轮都整段重发的那几轮。所以请求数只说明轮次砍没砍掉，值不值看未缓存输入与墙钟。
- 跨会话、跨 harness task 的缓存复用分两部分：规则 prompt（CLI 自带指令、协议、外壳）各窗相同；长输入（载荷、知识库索引、乃至音频）
  在同一条 run 的几个任务之间相同。driver 的复用机制也有两套：一是按 token 的固定位置自动做前缀缓存（有的要开开关，有的不用）；
  二是 Claude Code 那样只在交界处（断点）缓存。拼 prompt 时把不变的放前面、变的放后面，两套机制都有机会吃上；断点式的还要看
  不变的部分是不是正好落在一个交界上。这两条不一定立刻改变哪个方案，设计开场与块的位置时记着。

**候选**（每一处都先回答它会不会碰到工具协议已定的事）：

1. **单任务会话不先领任务：已做，默认开**（2026-10-16）：harness 代领 task，开场带上块，省掉只为调 `next_task` 的那一次请求；
   agy 上把协议写进 agent 文档、省掉读协议的 `view_file`，`kb_index` 在 Codex、Claude Code、CodeBuddy 上随开场交付。
   现行接法在 [`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [§1 档位与形态](llm_agent_tool_protocol.md#1-档位与形态)「开场带上任务」；
   `FINESUB_TOOL_SESSION_OPENING=0` 关掉（A/B 用）。
   Codex（GPT-6 Luna）已验过（2026-10-16）：每窗请求 6 → 3、墙钟约 −23%、缓存读约 −43%，未缓存的差在噪声内，质量不差于基线；agy、Claude Code、CodeBuddy 各一窗核过机制（B 下都不再 `next_task`、全部首次接受）。
   还开着：模型仍会自己多调 `pull_status`；Claude Code 在 `--knowledge read` 下又取了一次随开场到的 `kb_index`（n=1）；Codex 每会话开头一次的找工具请求（code mode 下先用一个 `exec` 列出 finesub 有哪些工具）随直调消失：工具定义直接在工具表里，探针里第一步就调 `next_task`（2026-10-22），生产里的读数等这次改动的验收。读数见 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md) [3.11 单任务工具会话开场带上任务：验收与对 capsule](llm_local_agent_experiments.md#311-单任务工具会话开场带上任务验收与对-capsule2026-10-16)。
2. **必读块怎么给**：在 agy 上，合成一个文件省不下——`protocol.md` 加 `payload.md` 共 51 KB，超过 `view_file` 单次 46 KB，
   照样要读两次；除非两块合起来压到 46 KB 以内。**不改** `read_context` 的语义：静态块只保证模型「有机会看过」，
   compact 之后要靠它重读（工具协议「必读块台账与 submit 门」）。Codex 上的重读已查清是截断所致（见上）。

已经做掉的：accept 之后不再发收尾请求（`[llm] local_agent_exit_on_accept = auto`；Codex 只在最后报一次用量，默认照旧）；
Codex 改成直调（不再分页，2026-10-23）、同一条 CLI 不重发已交付的 protocol（2026-10-16，见上）。

**怎么量、怎么验**：

- 逐请求拆（agy）：用它自己的会话记录（`conversations/<id>.db`；`gen_metadata` 里 (1,4,2) 是未缓存、(1,4,5) 是缓存读取、
  (1,4,3) 是输出合计），配 `audit/mcp-frames.jsonl` 与 steps 认出每次请求做了什么。accept 后立即结束 CLI 时最后一次请求
  来不及写进记录，它的用量 = 事件流补出的整窗总量减去记录里的各行。CodeBuddy 与 Codex 各找自己的逐请求来源
  （CodeBuddy 的 `~/.codebuddy/traces`，Codex 的 rollout）。
- 改一处量一处：同一素材、同一 driver，打开 `local_agent_keep_episodes`，对比改前改后的请求数、逐项 token 与墙钟。
- 验收：请求数与未缓存输入下降；质量不变差（同一份准备任务产物，盲评对噪声基线）；知识库与检索工具在需要时仍可用
  （开 `--knowledge read` 与 `--retrieval local` 各跑一次）；compact 后重读必读块的路径不受影响。

## 未做：`resume` 档在工具会话下仍未定义

`resume` 今天恒走 capsule 窄路——三家都报 MCP 能力，若按「有 MCP 就工具会话」它会永远退化成
`per-window`，A/B 路径就没了。要把它挪到工具会话，原先有两问：

- **(a) 三家的续会话 flag 与 per-invocation MCP 配置能否共存——能（2026-09-30 真机探针）。**
  CodeBuddy 2.137.1 + hy3、agy 1.2.13 + 3.7 Flash、Codex 0.157.1 + Luna 各跑一次两轮：第 2 轮带着新的
  MCP server 续第 1 轮的会话，三家都 `next_task → submit` 被 accept、交回的是只在第 1 轮给过的暗号、
  handle 不变；第 2 轮输入约是第 1 轮的两倍且大半命中缓存。n=1、两轮紧接、同一 `conversation_key`。
- **(b) 续上的会话里模型已有上窗的必读块，而台账每次 lease 清零**——没测到（探针第 2 轮的块全是
  新内容）。倾向的处理：先接受重读。窗口之间重复的只有静态块（协议、context pack），按台账再拉一遍
  就是 `per-window` 今天付的那份，浪费但无害；跨窗台账等量出重读确实贵再说。

**真要接线，改五处**（前四处探针为此在代码外绕过，没改源码）：

1. agy 的 `_argv` 一见 `mcp_server` 就走 `_tool_argv`，后者不收 handle——给了也被静默丢掉、等于新会话
   （`local_agent.py` 的 `AgyLocalAgentDriver._argv`）。`_tool_argv` 要能发 `--conversation`；
2. client 的工具会话入口写死 `session_scope="task"`（`RoleClient._run_agent_tool_session`），要按档位给
   assignment scope 与 `conversation_key`（key 决定 cwd `.conversations/<digest>`，两轮必须同一个；
   换 key 能否续上没测）；
3. 工具会话上不查 handle 漂移：accept 之后的提前返回（`_runtime_completed_result`）在漂移检查之前。
   续会话的工具会话要把这项检查挪到它前面；
4. `agent_transport_for` 里「`resume` 恒 capsule」那条规则改成探到 MCP 就工具会话。
5. **续上的会话每个任务列同一份工具，搜索授权逐次把关**（2026-10-30 一次续会话实验里看到）：工具表在请求前缀里（CodeBuddy 的 `--tools`、
   agy system prompt 里的 MCP 段），按任务的知识库与搜索授权列工具，换任务时工具一变，整段历史不命中缓存（那次每换一个 pass 按未缓存重读
   14–22 万）。固定下来之后，没有搜索授权的任务也看得到搜索工具：agy 有项目守卫逐次拒绝，CodeBuddy 只有 `--tools` 这一道，要另加逐次守卫，
   可行的是 `--settings` 带一个 `PreToolUse` hook 读本任务的授权；`--setting-sources ""` 下 hook 是否加载、`--print` 下是否触发都没验，先真机探。
   `--resume` 时新给的 `--tools` 会生效（同一实验里 WebSearch 只在有搜索授权的任务上列出，请求的工具表随之变）。今天的生产不受影响：
   没有续会话的工具会话，`--tools` 每次发车按本任务授权重建。

连带的一条：`resume` 的 handle 缓存仍挂在 `RoleClient` 上，没并入 run 作用域的会话注册表
（owner 决定 4 要求共用）。和上面五处一起做才划算。

**什么时候接**：会话档位实验已做（[`llm_local_agent_experiments.md`](llm_local_agent_experiments.md)
[3.9 `resume` 对 `per-window`](llm_local_agent_experiments.md#39-resume-对-per-window跨窗信息传递与单窗质量2026-10-02agy-每臂-n5codex-luna-每臂-n3)），owner 2026-10-02 定 `resume` 留在实验档、两家都不接入——理由是还没探索完，
不是质量。先做完下面两项前提，再按 3.9 的设计重测；决定接入后再改这五处，接完按工具会话形态补跑一次 `resume` 臂——
协议明说换传输就要重跑，capsule 路上的数不能沿用。探针脚本与证据在维护者本地的数据目录里。

**`resume` 暂时只作实验档（owner 2026-10-01）**。不论走哪种传输，进生产前还缺两件：

1. **每窗只发增量**：capsule 窄路上每窗把整份 messages（含约 1.7 万字符的 system、context pack、累积建议）
   再发进同一条会话，六窗就是六份副本；
2. **历史上限或轮换**：一条 run 一条会话（`resume-run:<lane>`），历史没有上限。CodeBuddy DeepSeek 每窗涨约
   2–2.5 万，按 catalog 的 300k 输入上限约 13 窗溢出（第二份素材的密度下约 23 分钟音频）；溢出或 CLI 自动 compact
   之后的质量没测过。

另：CodeBuddy 2.156.0 及之前的 `resume` 吃不到历史缓存，DeepSeek 与 GLM 都是，模型看到的历史里每窗还有 2–6 个乱码字符；
根因是它 `--resume` 分块解码会话记录的 bug。2026-10-11 在独立 CLI 2.164.0 上用 hy3 复测，乱码不再出现、历史整段命中缓存
（证据与没测的部分见 [`llm_local_agent_experiments.md`](llm_local_agent_experiments.md)
[3.8 会话档位与传输试跑](llm_local_agent_experiments.md#38-会话档位与传输试跑q1q32026-09-30每臂-n1) 末尾）。DeepSeek / GLM 的缓存命中
没有重测，按 3.9 的设计重测 `resume` 时一并量。

## 知识库里仍只能人工的环节

owner 原则（2026-10-02）：「原则上知识库的所有问题都应该能被 AI 自动解决。」共享冲突里的简介与 H1 已按这条交给
修复会话；下面是按这条原则仍待放开的环节，**本批不做**，各条写着今天为什么只能人工。放开其中任何一条都要先有
一个会话能兑现的 op 与对应的写入面限制，否则只会得到「模型说改了、op 被跳过、事情永远挂着」。

- **其余条目层字段的共享冲突**（`entry_type`、`native_names` 等）：模型词表里没有改这些字段的 op。
- **已退役条目上的冲突**（本地节点或它所属的条目已退役）：没有条目可给会话看；要放开得先定「该不该把条目
  `restore` 回来」由谁判，而 `restore` 今天只在人工 CLI 里。
- **`knowledge candidates` 的 `pending_human` 队列**：repair 会话判 `needs_human` 并写明缺什么证据后，只能人用
  `candidates --resolve` 结案；没有接到 `verify` 或搜索上去补证据。
- **`rendered/` 的文件级损坏**（首行不是 H1、未闭合的 HTML 注释、多于一行的简介、意外的元数据行、改名撞车）：
  收割逐文件告警、原样保留，不交给 AI 修——结构坏了就不知道哪一行本该是什么，改名撞车要人拍板。
- **`rendered/` 里对应条目已不存在的编辑**：收割留着文件、告警请人处理（`edit.py` 的 orphan 分支）。
- **phase-b 标为重复、但胜出条目已不存在的条目**：保持原样交人工（`phase_b.py`）。
- **共享服务器上的投稿审核**：prose claim 的裁定就是维护者的人工判断（`share/review.py`），批准是信任边界上的
  人工闸门；放不放开不只是能力问题。

## 知识库遥测卷积（`events` 表）

**版本历史不是增长源，事件表才是。** 真库实测（2026-09-02，rev 53，5 天 30 个生产任务，文件
2.5 MB）：`events` 4581 行 / 848 KB（59%）、`evidence` 1274 行 / 342 KB（24%），四张
`*_versions` 合计 139 KB（10%），其中已关闭的旧版本行约 40 KB（3%）。所以
知识库节点设计时「初版不实现 GC」的决定维持——压版本行
最多省 3%，且它列的前置条件（`snapshot_pins`、revert 可达规则、dry-run 与显式确认）一个都没有。

`events` 每个生产任务 150–500 行、每行 185 字节，`exposed` 占 85%——量是「注入节点数 × 窗口数」，
库越大、任务越长它越多。三个读者都是全表扫：`report.py`、`share/exchange.py` 打包 bundle 时按
节点聚合、以及每次 `INSERT OR IGNORE` 撞的 `dedupe_key` 唯一索引。

**形态**（到触发条件再做，不是现在）：

1. 新表 `event_rollups`，主键 `(node_id, item_id, kind, opportunity, matcher)`，字段是任务数、
   窗口数、首末 task_id、首末 rev。§5.2 的误触发判据、report 的 per-matcher 转化率、bundle 的
   按节点计数，三个消费者要的都只是这些聚合，没有一个读单行 span。
2. 保留窗口：最近 K 个任务（建议 20）留原始行（逐窗视图与排查用），更早的折进 rollup 后删除。
3. **幂等性不能丢**：折掉的任务写进 `rolled_tasks(task_id)`，`signals.py` 的插入对已折任务直接
   跳过，否则重跑一次旧任务就双计。这是唯一要碰写路径的一处。
4. `evidence` 暂不动：它是 claim 级记录，bundle 逐行导出、共享 server 按 `(node, field,
   value_hash)` 查存在性、`verify.py` 也读它。若将来要收，`transcript` 类占 83%，可按同一键收成
   计数加末次日期，键要原样保留。
5. 顺手清 `migration_aux`（plan 写了「切换后可删」，20 KB）。
6. 落点：`maintain.py` 加子命令 `compact`，默认 dry-run 打印将折多少行、聚合前后是否一致，
   `--execute` 才写；写完 `VACUUM`。`report.py` 与 `exchange.py` 改为读 rollup 加原始行之和。
7. 验收：折叠前后 `report` 的聚合数字逐字节相等（声称语义不变的优化，按 README_DEV 必须
   bit-exact）；折两次等于折一次；已折任务重跑不改变任何计数。

- **触发条件**：`report` 或 `push` 单次超过 5 秒，或文件超过 50 MB——按当前速率约 1500 个任务
  之后。
- **不要做的两件事**：版本行 squash（前置条件未备、收益 3%）；为省空间改 `exposed` 的记账粒度
  （那是 §5.1 一致性度量的分母，改了报告就没法与旧数据比）。

## 说话人裁判的另两种接法

说话人裁判（由 LLM 按纠错窗口复核每行的说话人）已按**路线一**接入（`--diarize judge-text` / `judge-audio`，跑到 LLM 阶段的 `quality` 档默认开，见 [`docs/llm_harness_behavior.md`](llm_harness_behavior.md)「说话人裁判」）：单独一步、读原始 ASR 的行与声学标签，判决和纠错互不依赖（窗口嵌在纠错窗口里），
排在纠错之前，结果按源段存 sidecar（owner 2026-10-08）。另两条路线的实验读数（2026-10-07–08，11 个窗口、每种接法各一个 pass）：
三种接法在认人的主读数上点估计差距都在 3 个点以内（单次运行，区间宽到约 ±5）；裁判读纠错后的文本时，差在纠错把不同的人并进了一行（一行只有一个判决）。

**路线二：把认人并进纠错窗口**（纠错窗口的输出多一列说话人、一列改轨理由）。
- 读数：认人与单独裁判相当（新窗主读数 −0.3 / −0.5 个点，区间宽）；但**纠错质量明显下滑、不是抖动**（owner：不能接受）：
  抽样审计里纠错文或译文实质不同的 24 处，单独纠错更好 14 处、合并的更好 5 处；分组不同的 58 处里 46 处是单独纠错合并得对——带上说话人列以后，
  同一人被切开的句子也不合并了。好的一面是跨人合并少了。正文 `note` 的检查项少了约一半。
- 混在里面的因素：那次合并调用开的是 high 思考，单独纠错是 medium；350 行一窗时输出贴着 65,536 的上限，agy 在撞上限后让模型接着写，输入涨到估计的约 4 倍。
- 重启前提：裁判与纠错同一思考档（owner 2026-10-08 起的规则）下重测；prompt 先写明「说话人列不是少合并的理由」；看纠错质量要用同样的逐源段对照审计，不只看认人读数。

**路线三：拆成几个 pass，按工作量、要的输入与逻辑分工**（owner 2026-10-08）：**已做成三个 pass**（2026-10-17）。pass 1 是说话人裁判兼定字幕形状（`join` / `may_join` / `drop`，带音频时还能拆行、补半句、补漏句），pass 2 是只在「可并」处合并的纠错翻译，pass 3 是独立会话的校对；`quality` 与 `intermediate` 档开了说话人裁判时走（只有声学标签、不开裁判的任务走单 pass；`efficiency` 一律单 pass，开裁判报错），跑到 LLM 阶段的 `quality` 档默认开裁判。现行行为见 [`llm_harness_behavior.md`](llm_harness_behavior.md)「三个 pass」，决定与理由见 [`llm_design_notes.md`](llm_design_notes.md)「三个 pass 的决策记录」。还开着的：

- **pass 之间接着同一个会话**（owner 2026-10-08 的设想）：量过一轮（2026-10-16，agy 上的 Gemini 3.8 Flash、3 窗、pass 1 之后接 pass 2）：pass 2 那一步的花费约少 35%，几乎全省在思考上；并得更多、行更长；样本只一轮，第一期不接（owner D5）。agy 的用量是整段会话的累计，比较时要逐轮相减。生产的 pass 2 走工具会话，要接得新做「从别的会话续接」（capsule 路已有同一会话键内的续接，只发新消息，见 [`llm_local_agent.md`](llm_local_agent.md) [§12.1 会话形态（四档）](llm_local_agent.md#121-会话形态四档已决定什么还差什么接在哪) 下「只发增量的续接」那条，可作起点：要的是裁判那条会话的键与 messages 记录交给 pass 2，而 pass 2 的消息并不是裁判消息的延长，前缀比对那一步得换）；两个 pass 切在同样的窗口上（裁判窗就是纠错窗），例外是回答放不下而对半拆开的窗，以及纠错之后重新规划过的任务。
- **pass 3 修跨人合并**：要让它拆行。pass 2 只并整源，按源的边界拆回时间轴是准的，难在两半各自的译文；可行的做法是 pass 3 输出「拆回」与两半的纠错文、译文，脚本按源边界落时间。等跨人合并的量值得时再议。
- **补出的行与原有的行时间重叠**：云端任务的成品已经能交付重叠的行（不同说话人各自成行，见[云端转写「两个人同时说话」](manual/joint-asr.md#两个人同时说话)），但排不进空当的补行（`deferred`）在两种任务上都仍不显示、只留听到的起止；以后按听到的起止显示，云端任务上它自成一个分组键、可以与别人的行重叠。
- **pass 2 对「上一行没说完的可并行」写明一般就并**：owner 2026-10-16 暂缓，先当作退化处理（见[模型回答的退化](#模型回答的退化)）；若重试后仍常见两份都退化，再与别的改动一起测。
- **声学标签分了多个 block 的任务**：第一期整份回退单 pass。pass 1（裁判）靠声学轨号认人、判可并，跨 block 的轨号不可比，要接先定跨 block 时这两件怎么判。
- **拆行、补半句、补漏句归在裁判里是不是最好的分法**（owner 2026-10-14）：现在和认人放在同一次调用里。已有的读数（2026-10-07 的上限实验，agy 上的 Gemini 3.7 Flash 带音频、样本小，结论在 [`diarization-findings.md`](diarization-findings.md)）：裁判拆行拆了的多半对、几乎不误拆，插行整句做对只有 22–28%，改字每行净改对 +0.20 个读音字。窗口裁判里顺带做（2026-10-16，Gemini 3.8 Flash 带音频、3 窗一轮）：拆的 10 行全对、单人行零误拆，两人行召回约四分之一（其中一半多是声学拆行没拆对的），复跑一轮 11 行全对；补漏句两轮、补半句一轮都一处没有。留出窗验收（2026-10-17，约 38 分钟）：拆 4 行，落在两人行上的 3 行与真值一词不差，1 行拆了单人行；两人行召回 3 / 28；补半句、补漏句 0 处。带音频的裁判把同一个人误标成换人的次数是纯文本裁判的 1.1–2 倍（原因与处理见上面「明确不做」里「再改 prompt、换模型或加一轮去修带音频说话人裁判的明显的错」一条）。
- **字幕空当里漏掉的话**（2026-10-16 量）：窗口裁判开着补半句、补漏句也一处不补，但空当里确有漏掉的话。同样 3 窗，100 处 1.2 秒以上的空当逐段截出给 agy 上的 Gemini 3.8 Flash 听，它说 19 处有字幕里没有的话；whisper large-v3-turbo 不开 VAD 逐段转写核对，13 处一致，6 处 whisper 只听到笑声、游戏声或一个音节。一致的 13 处里有 4 处是「ありがとうございます / ました」，正是 whisper 自己常写出的幻觉，这几处 whisper 不算独立核对（owner 2026-10-16：像幻觉，之后优化幻觉处理时看能不能在 ASR 阶段消掉）；其余 9 处是有内容的话（每窗约 3 处）。第一期不做（owner 2026-10-16）。可能的做法：逐个空当去听的一步（读者单独的准确率约七成，要第二个来源核对），或在 speech 侧对空当不开 VAD 重转写、再由会听的模型确认；后者动的是 VAD 召回，要按 speech 侧的规矩写明质量影响。

## 模型回答的退化

**状态**（2026-10-17）：纠错窗、说话人裁判与校对的回答报了没思考时，开新会话再问一次、两份留一份；回答是漏出来的思考时，取它末尾的答案照常校验。两件都已实现，现行行为在 [`llm_harness_behavior.md`「重试与拼接」](llm_harness_behavior.md#重试与拼接)；不专门验收，以后跑别的任务时顺便看（理由在[定下的做法与理由](#定下的做法与理由owner-2026-10-17)）。pass 2 整窗几乎不并（本窗标了可并的接缝不少于 30 处、并掉的不超过 5%）仍只在纠错阶段的摘要里提一句「可并几乎没并 N 窗」（不是 Warning），分组窗的窗口记录带 `degeneration`（[`llm_harness_behavior.md`「三个 pass」](llm_harness_behavior.md#三个-pass)），攒数据。

**为什么**（owner 2026-10-16、17）：同一份输入，模型有时答得明显差。最明显的质量滑坡是模型完全不思考；pass 2 整窗几乎不并也是一种，但较轻。

### 「没有思考」怎么判

对所有后端（agent 与 API）用同一条：**写出交付回答的那一次请求没有思考**，且同时满足——

1. 用量报上来了。输入、输出、思考全为 0 是用量缺失，不是没思考。
2. 这条路由会报思考。不报的一律算「不知道」，不当信号：codebuddy 的模型、agy 上的 Claude 从来只报 0。
3. 送出的思考档不允许不想：服务方的值（模型目录映射之后）是 off / none / minimal / low，或者根本不送思考参数时，0 不算信号。只看映射之后的值：映射之前是 low、映射之后更高（例如 Luna 的 low 映射到 medium）时，我们明确要了它想，0 照样算（owner 2026-10-17）。依据：agy 上 `gemini-3.8-flash-low` 的 6 个会话每一步都没想，low 档不想是常态。
4. 看写答案的那一步，不是会话里所有请求的合计：工具步不想很常见（codex 尤其多）。不过在现有记录里两种看法没有分歧——没见过工具步想了、写答案那步真的没想的会话，所以先按合计判也可以，等后端能逐步报了再改。

不另设「思考偏少」的门槛：思考量随任务种类差很多（研究、查询这类小任务正常也只想 2000–4000，纠错窗 5 万上下），与窗口行数只弱相关（3.8 上 0.47，其余路由约 0），定不出统一的「预期」。落在 0 与正常之间的调用都是回答通道出的毛病，不是思考退化，已有机制处理：

- 思考漏进回答（`thought_leak`，回答以 `thought` 开头）：思考被记成可见输出，所以思考只剩一两千、可见输出暴涨到 3.6–4.5 万，但总输出（可见 + 思考）与同一窗正常重答的约 4 万相当——模型想了，只是放错了通道。它与零思考是两回事：零思考的调用没有一次是泄露，泄露也很少（agy 会话库里 354 个写了回答的步骤中 3 个，日志里能对上的 2 次都是裁判）。原样交付时校验不过（思考里提到了好几次答案标签，解析读错了块），同一次运行里重答正常，没进成品。但**思考之后的最后一个答案块是完整的**：两次都能通过裁判的校验；对行真值（尺子 1），认人对的行 145 / 153 对重答 147 / 153、269 / 285 对 262 / 284，两份答得不同的行里各有对错；形状上合并略多（join 15 对 11、24 对 18），跨人合并处数相当（4 对 2、2 对 2），在同一窗两次正常运行的抖动之内。只有两例。
- 可见输出出奇地短（几百 token）：空回答、截断，全部被校验打回、重答正常。

真的不想与这两种分得开：真零思考时可见输出与正常相当、总输出只有正常的约五分之一，回答也不以 `thought` 开头。

后端怎么拿到「写答案那一步」的思考：Gemini REST 一次调用就是一步（`thoughtsTokenCount`）；agy 的 `step_update` 逐步带用量，现在只取了合计；codex 的 `token_count` 事件逐次请求带 `reasoning_output_tokens`；Claude 的用量不单列思考，要看回答里有没有 thinking 块。各后端报不报现在是 driver 的一项能力（`LocalAgentDriver.reports_thinking`，打在调用记录的 `thinking_reported` 上；API 后端在 `degeneration.reports_thinking` 里认）。

### 读数（2026-10-17，本机全部旧记录）

- 约 2000 次调用（`tools/thinking_scan.py`）：真的思考为 0 几乎只出现在 agy 上的 Gemini 3.8 Flash（medium 档），纠错窗 17 / 116、裁判 8 / 66、校对 2 / 14；同一模型 high 档的研究与知识更新 47 次一次也没有。3.6、3.7 Flash 与 Luna 都在 1% 以下。用量缺失的：agy 3.7 Flash 301 次里 23 次、codex Luna 280 次里 21 次。
- 不是漏报：零思考的纠错调用耗时只有同路由中位数的 7%–45%；agy 自己的会话库（`~/.gemini/antigravity-cli/conversations/<会话>.db`，模型步骤的元数据里逐次请求记着总输出、可见输出与思考）里，2026-10 还留着的 11 次零思考调用每一步都没有思考，总输出正好等于可见输出（9 月的已清掉）。两种会话送出的模型标识都是 `gemini-3.8-flash-medium`，不是 agy 换了档位。
- 真零思考时思考没有挪进预留的 `<reasoning>` 块：这一块中位数约 1000 字、最长 1815 字，正常时约 760 字、最长 1438 字，可见输出总量相当。挪进回答本身的是另一回事（上面的 `thought_leak`）。
- 一个会话里是否一致（agy 482 个会话、codex 183 个，逐步扫）：agy 3.8 medium 91 个会话里 15 个全程不想，5 个前面的工具步想了、最后一步没想——其中 2 个是思考漏进回答（被打回重答），2 个最后一步只是十来个 token 的收尾话，1 个是一步空回答之后正常作答；high 档 3 个只在零星的小工具步没想。codex 上写答案那步几乎都想（Sol high 有 1 例没想），全程 0 的只是 5–6 个 token 的探活。
- 跨用户轮：agy 上多用户轮的会话 17 个，16 个每轮都想、1 个每轮都不想，没见到中途切换；但修复轮里两种都见过（一次裁判第一次回答没想、修复轮想了 8.6 万，交付的回答仍带着第一次的 27 个越限 join；一次纠错窗两轮都没想）。所以按轮判，不假设一致，且修复轮救不回来。
- 前后两次调用基本独立（上一次为 0 时下一次为 0 是 2 / 26，基础率约 11%），偶有成串（2026-09-03 连着 4 次）。
- 能自动量出的坏表现目前只在裁判上：思考为 0 的裁判窗 5 个里 4 个写了 14–29 个越限 join，思考正常的 10 个窗只有 0–5 个；纠错窗上粗的代理指标（保留行比例、注释密度、节奏分）分不出，要量质量差距得盲读。

### 定下的做法与理由（owner 2026-10-17）

- 两份都没想时接受：同一输入连着两次独立的零思考按独立估算约 1%，真出现多半是这窗简单或服务那一阵在抽风；先比校验错误数、分不出留原交付。
- `resume` 档里一轮没想时，那一窗就是续会话链的轮换点：agy 上见过整个续会话每轮都不想，原会话可能卡在不想的状态，同会话再问又带着坏回答；历史上限 / 轮换也是 `resume` 本来就要的机制（见[未做：`resume` 档在工具会话下仍未定义](#未做resume-档在工具会话下仍未定义)）。代价是这一窗丢掉前面窗口的历史；`resume` 用得多时这个代价落在生产上，不只是实验档。
- 思考泄露不整份打回：两例的质量与重答相当，省一次约 4 万输出的重答。
- agy 送出的 `--effort` 写进调用记录（`isolation.reasoning_effort`），好分开档位与任务类型。
- 不专门跑验收（owner 2026-10-18）：判错的两个方向后果都有界。把想了的误判成没想（用量漏报、续接时减错），只是这一窗多一次调用，每窗最多一次，重问那份想了又合格才留，原来那份本来也合格；把没想的漏判成想了（续接时用整条会话的累计，它只会多算），就和没有这条规则时一样。重问停下、报错时留原来那份，有测试。剩下的只有在真实后端上多问了几次，这在别的任务的运行里顺便就能看到。

### 还开着的

- 跑别的任务时顺便看：纠错阶段摘要里「没思考、重答 N 窗」的次数，窗口记录与 sidecar 里 `thinking.chosen` 留了哪份、有没有 `retry_error`。重问的比例明显高过上面读数里 3.8 Flash 的一成多（12%–15%），多半是用量漏报或续接时减错，再回头查。
- 现在按整次调用的合计判，不单看写答案的那一步；agy 的 `step_update` 已逐步带用量，要改成看最后一步时从这里接。
- pass 2 几乎不并攒够数据后再定要不要重试。现有的相关机制：校验失败的修复轮；agy 回答通道的畸形重试；v15 的节奏评分（`output_protocol.py` 的 pacing scorer）设计过「返工上限 2、取几次里最好的」但没接线。
