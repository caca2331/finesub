# agent-tasks：写给 agent 的任务说明

这里每个子目录是**一件事怎么做**的完整说明，给要替用户干这件事的 agent 读。

从 2026-09-01 起它们不再依赖任何 harness 的「skill」注入机制——**没有谁会把它们自动塞进你的
上下文**。你自己按下面这张表判断手上的任务对不对得上，对得上就把那份 `SKILL.md` **整份读完**
再动手；它们写的是判据、边界和「不要做什么」，跳着读会漏掉后者。

## 有哪些

| 目录 | 什么时候读它 |
| --- | --- |
| [`run-audit/`](run-audit/SKILL.md) | 审查/诊断已经跑完的纠错翻译 run、reference_ingest run 或知识库健康：schema 违规、提案与反馈合规、成品字幕的成类问题、harness 运行时异常（重试链、双提交覆盖）。⚠ 目标若是「在固定测试床上改 prompt 并验收」，别用它代替迭代——读 `docs/prompt-iterate.md` 与 `tools/session_replay` |
| [`feedback-pack/`](feedback-pack/SKILL.md) | 把跑完的任务打包成一个 zip 放到用户桌面，供他们**自己**发给开发者：报 bug（带人声轨的完整现场）或贡献素材（只收有精修字幕的任务，不带音频）。⚠ 它只产出文件、给路径——**绝不替用户发出去**；诊断问题本身是 `run-audit` |
| [`knowledge-ingest/`](knowledge-ingest/SKILL.md) | 把一份材料蒸馏进知识库：用户写的主播习惯笔记、存下来的网页、别处的术语表，或迁移时机械导入读不懂而泊进「待归类」的旧内容 |
| `desktop-portable/` ⚠ | 在本机构建可直接运行的 FineSub Desktop portable 包（开发/试跑用，不是正式发版） |
| `release/` ⚠ | 发布新版本的完整流程：版本 lockstep bump、orphan `main` 快照与 CI 把关、桌面端签名构建、GitHub Release、PyPI |

⚠ 带 ⚠ 的两份是**维护者专用**，由 `scripts/publish-main.ps1` 从每次公开快照里剥掉，所以
公开仓库里没有它们（`main` 的全部历史里也从未有过，2026-09-01 核实）。这里只列名字不给
链接——公开树里点不开的链接比没有更糟。在本仓库的工作副本里它们就在同名目录下。

## 约定（要加新的就照这个来）

- 一个目录一件事，主文件叫 `SKILL.md`，开头是 YAML frontmatter：`name` 与 `description`。
  **`description` 要写「用户说什么话时该读它」**，而不只是功能摘要——它是唯一的路由依据。
- 需要脚本、参考资料、评测就放同目录的 `scripts/`、`references/`、`evals/`；
  `SKILL.md` 里用**仓库根起算**的相对路径引用它们（例：`agent-tasks/run-audit/scripts/...`）。
- 只写这件事本身。契约细节归 `docs/` 的 owner 文档，这里引用、不复制——复制出来的那份
  迟早和真相分家。
- 每份都要有「不要做什么」/ 边界那一段。这类文档最贵的价值在那里。

## 与 `docs/` 的分工

`docs/` 回答「这个东西是什么、契约是什么」；这里回答「要做这件事，按什么顺序、用什么判据」。
用户向的能力清单在 [`docs/manual/agent-tasks.md`](../docs/manual/agent-tasks.md)。
