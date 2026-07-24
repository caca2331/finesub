# 已知失效模式：FLAG → 如何核实 → 归属

按类组织。每条给：症状（digest 里长什么样）、核实方法、归属（修在哪）。
来源是历次真实 run 的审计沉淀（首批：2026-07 四月一日三连 run）——发现新模式请追加到本文。

**不要在这里堆 prompt 迭代方法论。** 固定测试床上的失效模式、验收抽样、合并口径与
「validation-ok ≠ 质量」见仓库文档：

- [`docs/tools/prompt-iterate.md`](../../../../docs/tools/prompt-iterate.md) §2（协议/抽样）、§4（合并口径）、§5（迭代侧失效）
- 归属到具体 fragment 后若要改 prompt：同一文档 §0–§3 + `tools/session_replay`

本文只服务**已完成 run** 的 FLAGS 核实；D 类质量结论必须配合精修抽查或 prompt-iterate
的 gold/元信息流程，禁止单靠压缩率或 `validation_ok` 下判。

## A. Schema / 输出契约类

| 失效 | 症状 / FLAG | 核实 | 归属 |
| --- | --- | --- | --- |
| category 枚举拼接 | `hint category 非法 'streamer|game_lore'` | 看 feedback 原文；纠错阶段高发（刚输出完 `|` 分隔 CSV 被带偏），research 阶段少见 | `fragment_task_feedback_schema_v3.md`（已有枚举澄清）；解析层已做取前段救回 |
| proposal 枚举/缺字段 | `proposal op 非法` / `缺字段 X` | 读该条 proposal 全文，判断是笔误还是理解错 schema | `fragment_knowledge_output_v1.md` |
| set_featured 越权 | `模型输出了 set_featured` | v9 及以前合法（当时契约要求维护精选）；v10 起违规但 harness 会跳过 | `knowledge_update_refined_v1.md`；apply 层 `allow_featured=False` |
| `<reasoning>` 泄漏 | 窗口/响应 `+reasoning` | v17 起**必须**（回复开头有且仅有一个，按 thinking 深度分档；v10–v16 为可选）；核实是否超长挤占输出预算 | `fragment_output_contract_v1.md` A#9 |
| Markdown 代码块包裹 / 声称已写库 | 脚本不测；读 proposal 块首尾 | 直接看 exchange | `fragment_knowledge_output_v1.md` 规则 2 |

## B. mistake 台账字段语义类（仅精修对照模式）

| 失效 | 症状 / FLAG | 核实 | 归属 |
| --- | --- | --- | --- |
| ASR 问题混进翻译台账 | `wrong==source` | 看 source 是否为幻觉/误听文本——听写问题应走 feedback 的 asr_corrections | `knowledge_update_refined_v1.md` 规则 3 |
| 占位符字段 | `wrong/correct 是占位符 '(删除幻觉)'` | 字段必须是真实文本 | 同上规则 4 |
| correct 非中文 | `correct 含假名` | correct 必须是简体中文译文 | 同上规则 4 |
| 杜撰 wrong | 脚本测不了 | 拿 `wrong` 全文在 annotated csv / raw srt 里搜——搜不到即杜撰 | 同上规则 4 |

## C. 知识内容质量类

| 失效 | 症状 / FLAG | 核实 | 归属 |
| --- | --- | --- | --- |
| 单次证据常态化 | `content 含频率词「常在」` | 看 reason 引用的证据是否只有本次一场直播 | `fragment_knowledge_structure_v1.md` 原则 8 |
| 同月重复记录 | 脚本测不了 | 同场直播切多段分别 ingest 时，对比各段 proposal 与条目现有全文 | 同上原则 9 |
| 非长尾知识 | 脚本测不了 | 该条目内容是否网上一搜即得/LLM 已知 | 同上定位段 |
| replace_section 丢尾 | `entry_render_report` 里 truncated 非空 + 同条目有 replace_section | 对比 apply 前后条目全文（知识库内嵌 git：`git -C knowledge log/show`） | pending 压缩 feature（docs/knowledge.md 遗留项） |

## D. 纠错/翻译输出质量类（有精修对照时才可靠）

| 失效 | 症状 / FLAG | 核实 | 归属 |
| --- | --- | --- | --- |
| 高情绪连坐删除 | `空档 X–Y 精修版同区间有 N 条台词` | 听感核实做不了，就对照 raw srt：空档区间 raw 是复读/乱码 + 精修有实词 = 模型没重听音频 | `fragment_goals_correction_audio_v1.md` 第 4 条、`fragment_insert_rules_v1.md` 第 6 条 |
| 可疑过度合并 | `压缩率偏高` / `单条字幕跨度 >10s` | **嫌疑**：对照精修同区间切了几条；一句连续长话可豁免。现行 validation 对源数/合并长度多为 warning（prompt-iterate §4）——不能用压缩率或 validation-ok 代替质量分；固定窗深挖用 merge/drop gold | `fragment_merge_rules_v1.md`；迭代协议见 `docs/tools/prompt-iterate.md` §2/§4 |
| 元话语泄漏 | `字幕文本列含元话语「（注」` | 看该行是否在向观众解释翻译决策（vs 正当的非语音事件括注如「（会员加入提示）」） | `fragment_goals_translation_v1.md` 规则 2 |
| insert 零使用 | （v63 起生产变体已废弃插轴；此 FLAG 不再作为审计项） | — | — |
| 成类翻译错误 | 脚本测不了 | 三方对照抽查（原文/机器/精修），只收成类问题；语义翻转（肯定↔否定）优先于个别用词 | 视类型：翻译目标 fragment 或 mistake 台账素材 |
| 繁体字形泄漏 | `成品字幕疑似繁体` | 打开成品 SRT 直接看；知识 proposal 也会连带混用字形 | `fragment_goals_translation_v1.md` 规则 1、`fragment_user_reminders_*` 简体条 |
| 圈内借词被抹平 | 脚本测不了 | 三方对照：精修保留「萌」等借词而机器意译成通用词 | `fragment_goals_translation_v1.md` 规则 8 |

## E. 运行时 / harness 类

| 失效 | 症状 / FLAG | 核实 | 归属 |
| --- | --- | --- | --- |
| 同窗 validation 重试 | digest 时间线 `ok=False` + `correction_window_retry` reason=`validation_same_window` | 读该次 `validation_errors` 与 exchange；同进程 attempt 递增 | `stages/correction_loop.py`；具体 error 再归 A/D 模板 |
| 并发双跑 / 后写覆盖 | 时间线出现**两次** `attempt=0` 且 API 时间重叠；或 `final_srt`×2；`correction-windows.jsonl` 同 chunk 多条 | 比 exchange 头里的 call 起止时间与 `final_srt` artifact 时间戳；磁盘 SRT 通常是**最后一次**成功提交 | 操作/调度（同目录并行 ingest）；非 prompt。llm 并发设计为 1（`batch.py`） |
| 重试/降级 | task-report 关键行有 retries>0 / fallback | 读对应 exchange 的 validation_errors 与逐次 attempt | `stages/correction_loop.py`；重试原因归相应模板 |
| IP-risk 误判 | 日志把无关报错当 IP 风险 | 看原始错误文案是否真为地区/代理拦截 | `client.py::is_likely_ip_risk_error` |
| token 分布异常 | `token_distribution_report` | thinking=0 且可见输出含大段推理 → 模型没用隐藏思维链（flash-lite 常见） | 记录并评估输出预算；flash-lite 是优化基准，不以换模型作为处置结论 |
| 意译被误判为错误 | mistake proposal 内容像“换个说法” | 对照精修：机器译文是否真的歪曲原意/丢信息？精修的信达雅改写不是错误，出彩者应走 add_example 范例库 | `knowledge_update_refined_v1.md` 台账规则 2/6 |
| 注入截断/丢弃 | `entry_render_report`/搜索注入 report 的 truncated/dropped | 看被截内容是否关键 | `injection_budget.py` 预算参数；C 类丢尾风险 |
| 媒体音频截断 | `覆盖范围可疑`（成品远短于精修） | ffprobe 分流时长：视频流 vs 音频流（实例：2014s 视频只带 50s 音频——断流 resume 后损坏，merger 静默拷贝可读前缀） | `media_source.py::validate_video_audio_coverage`（已有守卫）；旧产物需删除重下 |
| ASR 语言设置错 | digest 的 `ASR language 设置` 行与素材语言不符；日语素材 raw 全汉字无假名 | 看 stable.json `metadata.asr_align.language` 与 index.csv 行 args | 调用参数；reference_ingest 默认 auto，测试素材行内 `--language ja` |

## 知识库健康（--kb 模式）

index↔文件一致性、streamer 固定小节、精选 id 存在性与上限、条目超 4k token
（注入截断风险）。修复归属：`llm/knowledge/base.py` apply 层或手工编辑 + 内嵌 git commit。
