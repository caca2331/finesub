---
name: run-audit
description: >-
  审查/诊断 finesub 已完成的纠错翻译 run、reference_ingest run 与知识库健康：schema
  违规、知识提案/反馈合规、成品字幕与精修对照的成类问题、以及 harness 运行时异常
  （重试链、双提交覆盖等）。用户提到「审查 run」「诊断产物」「审计 out/reference」
  「检查知识库」「这次跑得怎么样」「validation 为何失败」「为什么还重试」等时使用。
  若目标是在固定测试床上改 prompt 并验收，不要用本 skill 代替迭代——改读
  docs/tools/prompt-iterate.md 与 session_replay。
---

# Run Audit：已完成 run 的离线诊断

对**已经跑完**的产物目录做只读诊断，写出可落地的报告（证据 → 归属 → 建议）。
契约与失效分类在本 skill 的 `references/`；**纠错 prompt 怎么迭代、怎么验收质量**的成熟
流程在仓库文档里——本 skill 不复制那套协议，只在对口处引用。

## 与 prompt-iterate 的分工（先读这段）

| | **run-audit（本 skill）** | **prompt-iterate** |
| --- | --- | --- |
| 对象 | 生产/reference 一次完整 run 的落盘产物；知识库健康 | 冻结 fixture 的固定窗，只换 prompt 重放 |
| 目的 | 解释「这次跑出了什么/坏在哪/归谁」 | 迭代 prompt/harness 文本并按协议验收 |
| 入口 | 本 `SKILL.md` + `extract_digest.py` | [`docs/tools/prompt-iterate.md`](../../../docs/tools/prompt-iterate.md) |
| 工具 | 离线读 `out/...`；**禁止** `--execute` | `tools/session_replay`（默认会调 API） |
| 质量判定 | 精修三方抽查找**成类**问题；压缩率/validation-ok **不能**当质量分 | §2：元信息 + 逐行抽查 +（固定窗）merge/drop gold；失败样本最低抽样量 |

审计若结论是「该改某 fragment / 合并纪律 / singles 残留类问题」：报告里写清归属模板，
并**指向** prompt-iterate 做下一轮（含 `--variant`、分档 `-n`/`--max-attempts`、失败抽样
规则）。不要在审计回合里直接开 replay 烧配额，除非用户明确要求。

工具契约：[`docs/session_replay.md`](../../../docs/session_replay.md)

## 先分流：用户问的是哪一类

1. **运行时 / harness**（「为什么 010 成功还跑了 011」「哪次 validation 挂了」「成品是谁写的」）  
   → 走下方 **轨道 B**（先看 digest 的 correction 时间线）。不必强行写完整「质量审计报告」。

2. **成品 / 知识 / schema**（「审查这个 run」「知识提案合不合规」「和精修比质量」）  
   → 走 **轨道 A**（digest FLAGS → 核实 → 语义抽查 → 结构化报告）。

3. **要改 prompt 并对比前后**（「下一轮怎么测」「capableC 合并怎样」）  
   → **离开本 skill 流程**，按 prompt-iterate §2 开 session_replay；可用本 skill 的归因作改动起点。

## 轨道 A：成品与知识审计

1. **定位输入**。`out/<stem>/` 或 `out/reference/<id>/`；精修常在
   `data/manually-refined-subs/<系列>/`（对照 `index.csv`）。有精修时质量结论才硬。

2. **跑抽取脚本**（每 run 一次）：

   ```powershell
   python .claude/skills/run-audit/scripts/extract_digest.py out/reference/<id> --refined <精修.srt>
   python .claude/skills/run-audit/scripts/extract_digest.py --kb knowledge
   ```

   digest = 布局 + 统计 + **correction 时间线** + FLAGS。FLAGS 是嫌疑不是结论；脚本测不了
   语义与「成类」问题。

3. **核实 FLAGS**。分类与归属见 [`references/failure-modes.md`](references/failure-modes.md)；
   路径/schema 见 [`references/artifact-map.md`](references/artifact-map.md)。
   - 用 **run 当时**的 prompt_version / variant 判契约（exchange 头或 fingerprint）；用**当前**
     契约提改进建议。
   - `test_profile` / flash-lite 暴露的失效默认归「prompt/harness 待改进」，不要用「模型太弱」打发。
   - 合并长度/源数：现行 validation **多半只 warning**（见 prompt-iterate §4）——不要把
     「压缩率高」或「validation_ok」单独当成质量结论。

4. **语义抽查**（脚本测不了）：
   - 有精修：抽 2–3 个时间段做 原文 / 机器 / 精修三方对照，只收**成类**问题。
   - knowledge/mistake proposals：长尾定位、reason 证据、`wrong` 是否真在 annotated csv 出现过。
   - 需要更严的失败样本纪律（例如多条 validation 失败要归纳主因）时，对齐
     prompt-iterate §2「失败回复审计最低抽样量」，不要从单条 error 外推。
   - **清单外巡检**：忘掉 FLAGS，扫成品 SRT 头/中/尾 + 一份 exchange——字形、覆盖时长、
     语言设置等重大问题常靠这一步发现。

5. **写报告**（轨道 A 默认结构）：

   ```markdown
   ## <问题一句话>
   - 证据：<路径 + 行号/时间戳 + 引文>
   - 归属：<prompt 模板 / src 模块 / harness 行为>
   - 建议：<可落地表述或代码改法；若属 prompt 迭代则链到 docs/tools/prompt-iterate.md>
   ```

   文末：优先级 +「正面结果」。若建议进入 prompt 迭代，写明建议的 session
   （通常 `correction`）、是否需要 `--variant`，**不要**在未获许可时调用 API。

## 轨道 B：运行时时间线（轻量）

digest 已打印 correction 尝试表时优先用它；否则从 `task-artifacts.jsonl` 抽：

- `correction_window_response`：`created_at` / `attempt` / `validation_ok` / `validation_errors`
- `correction_window_retry`：`reason`（`validation_same_window` 等）
- `final_srt` 出现次数与时间；`correction-windows.jsonl` 是否同 `chunk_id` 多条提交
- exchange 文件名里的 `attemptN` 与 API 调用起止时间（两趟 `attempt0` 时间重叠 → 并发写同目录）

回答格式：时间线 → 因果（同进程重试 vs 双进程覆盖）→ 磁盘上最终生效的是哪次。
只有用户接着要「那译文质量呢」再转入轨道 A。

## 边界

- 只读产物、写报告；**不修** prompt 模板、**不写**知识库、**不重跑**任务（那些是用户决定后的事；
  prompt 侧后续用 prompt-iterate）。
- 不要用 `--execute` 或任何会花生成配额的 LLM CLI。
- 多 run 可并行派子代理（把本 skill 流程与 references 路径写入子代理 prompt）。
