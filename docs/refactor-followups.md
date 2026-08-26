# 重构：未做完的、明确不做的，以及下次动手前先读的

2026-08 那一轮结构重构（批次 0–4 + 文档批次）**已全部完成并合入 `dev`**。计划正文连同
逐批实施记录与偏离表迁到了本地 `docs/archive/refactor-plan-2026-08.md`（gitignore，不随
仓库发布）——那份是过程台账，读它只为考古。本文只留**仍然要做的事**、**判定不做的事**，
以及**做同类改动前值得先读的几条**。

## 一、挂在发版流程上（唯一有明确顺序的一项）——**已完成（2026-08-17，0.4.0 发版时）**

三步按序做完：`tokcount-1.62.0-0` Release 已发布，资产 size/sha256 与
`runtime-manifest.json` 逐字核对一致（zip 时间戳固定，可重建复核），随后
`git rm --cached bin/windows-amd64/tokcount.exe` 并 gitignore `/bin/`。
新 clone 没有该 exe，按设计退到免费 `countTokens` 端点；要本地二进制就在
`tools/tokcount` `go build` 或解开 Release 资产（见其 README）。
公开 `main` 历史里已有的 blob 不改写，这是已知并接受的。

**同一片区域已完成**（不必重做）：Go 模块已从 `src/tools/gemini-token-counter/` 移到
`tools/tokcount/`、名字统一为 `tokcount`、三处硬编码名字的解析器折进
`finesub_bootstrap/token_counter.py`。

## 二、发版验收要额外算进去的一件事

**2026-08 的大改名没有等版本边界。** 计划要求「一次发版刚结束后立刻做」以求最长浸泡期，
实际是在 0.3.2 之后、下一次发版之前的周期中段做的。那批未发布的改动本来就要一起发，
所以损失的是「改名先单独发一版」这个选项，不是浸泡本身。下次发版的验收面里要包含：
`asr_playground` → `finesub`、顶层 `llm` → `finesub.llm`、仓库根包不再提供 console
script（改用 `python -m finesub.<模块>`）。破坏性说明已在 `CHANGELOG.md`。

## 三、判定押后（理由已定，重开前先看理由）

| 项 | 为什么押后 |
| --- | --- |
| **`globals.css` 的暗色覆盖挪回各组件旁** | **挪不动**：`[data-theme="dark"] .custom-select-trigger` 与 `.appearance-item .custom-select-trigger` 优先级相同（都是 0,2,0），暗色现在靠**排在后面**取胜；提到组件旁边就翻盘。真要做得先把这类组件规则改写成走变量、或抬高暗色侧优先级——那是行为改动而不是搬运，而这一层没有任何视觉断言。`dark.css` 已单独成文件、位置不动，`app/globals.css` 与 `desktop/README_DEV.md` 都写了为什么不能挪 |
| **长函数第二梯队** | `RoleClient.complete`（`llm/client.py`）与 `run_research`（`llm/research.py`）。审计判为「与 `execute_correction_windows` 同理但低一档」，等那次拆分的效果沉淀后再决定 |
| **`stages/` 归属自相矛盾** | `research.py`/`correction_translation.py` 平铺在 `llm/` 根、`stages/` 只装几个。随 correction 子包落地后重估，不单独立项 |
| **`text.py` 改名归位** | 740 行、通用名、实为异常 ASR 判据；连同包根几个横切模块的 `run/` 分组，低优先 |
| **`desktop/backend/resources/` 杂物袋** | `gpus.py`/`install_log.py`/`model_prefetch.py` 错位，`desktop/resources/` → `desktop/config/`；连同 `desktop/FineSub*.py` 的 CamelCase 缺注释 |
| **`tools/separator_aoti.py` 改名消歧**；`tools/` 根两个散文件补 README | `tools/` 按需维护，不随其他改动顺手做 |

## 三之二、押后项里已经做完的一件

**LLM 层纳入 `test_pipeline_reporting_boundary`** —— **已完成（2026-08-19）**。押后的理由是
「14 个模块的 `print` 改成 `current_reporter()` 是行为改动，不该由改名夹带」；2026-08-17 落盘
run 日志上线后它涨了价——不再只是守卫覆盖不到，而是**用户拿到的日志里 LLM 段是空的**。

最后分两笔做掉，顺序不能反（先加点后转换的话守卫全程是关的）：42 处裸 `print` 转 reporter 并
清空 `EXEMPT_PREFIXES`，再补新增上报点。**当时最值钱的一条判断**：日志里缺的那部分**不在这
42 处里**——`attempts.py`、`rate_limit.py`、`llm_runtime.py` 今天一个字都不输出，重试、配额、
校验失败从来没被打印过，所以主体是**新增上报点**而不是转换。按「把裸 print 接进 reporter」
去估工，会把这件事的规模估错一个量级。

现行契约见 [`reporting.md`](reporting.md)。

## 四、明确不做 / 接受现状

- **同名不同义不做批量改名**（`capabilities`/`paths`/`resources` 各有多义、
  `model_routes` vs `model_router`、`postprocess` vs `postprocessing`）：churn 大于收益。
  各文件被实质触碰时顺手消歧即可。
- `agent-sessions.jsonl` 纯追加无上限（~150 B/次调用，风险极小）。
- `tasks.json.invalid.N` 备份序号无上限（只在损坏时发生）。
- `agent_cleanup.py` 的 `failures` 绑定依赖当前控制流（改动该函数时顺手收紧，不立项）。
- **dev 侧 CI 闸门不做**：维持发版时 `ci-gate` 为唯一 CI 闸门。`dev` 不推远端，
  `on: push branches:[dev]` 永不触发；快照 gate 会把未发布的 dev 整棵树提前推上公开
  `origin`，而 orphan `main` 布局的存在理由正是中间内容不公开。残余缺口只剩 Linux 侧
  行为，接受由发版时的 `ci-gate` 兜底。

## 五、做同类改动前先读：四条踩实的教训

那一轮里同一类失败反复出现——**改动全绿、渲染正常、引用指向空气**。四条都是实证：

1. **筛选面要比改写面宽。** 用 `git grep -lE 'llm[./]'` 选文件会漏掉 `from llm import x`
   （后面跟的是空格），规则本身能处理，文件却从未进入清单。先用最宽的 `\b名字\b` 选文件，
   再靠规则决定改不改。
2. **worktree + editable 安装 = 假绿。** venv 的 editable 指向主 checkout，worktree 里
   `import llm` 会回落到旧树；标了 `requires_main_checkout` 的测试在 worktree 里直接 skip。
   两者都让整条分支跑不到真相，合并后才炸。**合并后必须在主 checkout 上再跑一次全量。**
3. **绿不等于对。** 守卫只能证明锚点存在。`§17` 重编成 `§15` 之后，一句「原 §15」照样
   解析得到——标题在，测试绿，读者点过去是另一节。
4. **改写脚本与校验脚本共用启发式常量时会一起瞎。** 两边都用「往前 200 字符找文件名」，
   于是长表格行里改不到的，校验也查不到——盲区完全重合，看起来一切正常。

配套的自动闸门已经在仓库里，改文档或改名前先确认它们会不会红：
`test/test_doc_links.py`（跨文档相对链接、被跟踪 docs 必须进 `CLAUDE.md` 索引、
`§`引用必须指到真标题）与 `test/test_import_boundaries.py`（禁 import 已删除的旧包名、
`finesub_bootstrap` 不 import 主包、`src/` 里只有 `paths.py` 可以数 `parents[N]`）。
跨文档引用的书写约定是 `` `文件名` §N ``。

## 六、指向归档

过程细节——每一批的验收口径、逐轮实施记录、「计划怎么说 vs 实际怎么做」的偏离表——都在
`docs/archive/refactor-plan-2026-08.md`。**要重开一个已被判定的决策时先去读它**：那张
偏离表记的正是「照抄计划会踩的坑」，包括依赖归属（pydantic 只能进 `[dev]`）、fixture
搬迁、`parents[N]` 路径算术、以及若干自述的微语义偏离。
