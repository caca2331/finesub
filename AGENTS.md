# AGENTS.md

Agent 进这个仓库先读这份。`README.md`、`README_DEV.md` 与 `docs/` 是中文写的规范文档；这里只放每次都用得上的
事实、护栏与按任务找入口的短表，细节去所属文档读。

若仓库根有 `AGENTS.local.md`（维护者的私有补充，公开仓库里没有）且它还不在你的上下文里，开工前先读。

## 项目是什么

把长音频转成字幕的本地管线：

```text
source audio -> vocal separation -> VAD + ASR alignment -> ASR stabilization -> SRT
```

生产入口是 `python -m finesub.pipeline`（打包的 `finesub` 命令跑同一个模块）。`src/finesub/llm/` 的 LLM
纠错 / 翻译层是实验性的，**不在默认流程里**：默认 `--stage` 停在 `raw-srt`，`translated-srt` / `final-srt`
才进入。模块地图见 [`docs/architecture.md`](docs/architecture.md)。

## 常用命令

```powershell
pip install -e ".[asr]"                    # GPU ASR 栈，含 qwen-verify 裁判（patched CTranslate2，见 README_DEV）
pip install -e ".[harness]"                # 只装 LLM 层（约 4GB 内存，ffmpeg）
pip install -e ".[asr,cloud,harness,dev]"  # 全部运行时 + pytest + ruff
# 只有 CPU 的 Linux 开发环境（同 CI 的 ubuntu 主作业）只装 [harness,dev] 加 CPU torch，不装 [asr]

# 一个源；产物默认在 out/<stem>/
python -m finesub.pipeline data/input.wav --model large-v3-turbo --language en --gpu-tier standard
# 多个源走同一个入口（队列、控制面、--resume-batch 见 docs/manual/batch.md）
python -m finesub.pipeline --manifest tasks.jsonl

# LLM 纠错 / 翻译：默认 dry-run，只产计划与 prompt
python -m finesub.llm.correction_translation out/input/input-stable.json --audio data/input.wav --prompt-dir out/input-llm-prompts

# 测试（轻量，不加载模型、不用 GPU）
python -m pytest -q
# 一批改动做完
python -m compileall -q src test && ruff check && python -m pytest -q && git status --short
```

跑测试最常见的三种情况：

- 改一个文件时：只跑它对应的测试文件（通常同名，对不上就 `git grep -l <模块名> -- test`），`-n 0`，秒级。
- 一批改动做完（合并或交给 owner 之前）：跑上面「一批改动做完」那一行，一批只跑一次；同一批里中间的提交只跑改到的文件对应的测试。
- 只改了文档（改动全是 `src/`、`test/` 之外的 `.md`），提交前：只跑文档守卫
  `python -m pytest -q test/test_doc_links.py test/test_doc_facts.py test/test_doc_style.py test/test_publish_filter.py test/test_settings.py test/test_packaging.py test/test_llm_coldread_scripts.py`，
  加上 `git grep -l <被改文件名> -- test` 找到的测试，再 `git status --short`。

其余情形（动顶层模块、设备解析、换 CT2、worktree、发版）与按组跑的命令见 [`docs/test-runs.md`](docs/test-runs.md)。

重资源测试（`--run-heavy-resource`）与完整生产运行：只在用户明确要求时跑。linter 只有 ruff 的 F 规则
（未定义的名字、没用的 import 与变量），范围是 `src/` `test/` `cli/`，配置在 `pyproject.toml` 的 `[tool.ruff]`、版本钉在
`[dev]`，CI 的 ubuntu 主作业跑 `ruff check`；没有 formatter。

## 护栏

- **LLM 默认 dry-run**；只有 `--execute` 花 Gemini 生成额度，用户没要求就不加。例外：
  `finesub.workflows.reference_ingest` 默认全执行（用户发起即 opt-in）。Gemini `countTokens` 免费，规划可以随便调。
- **不改 VAD/ASR 参数**（高危核心 `speech/recognition/transcribe.py`、`speech/preprocessing/energy.py`）；
  要改就写明**质量**影响并补测试或实验记录。
- **逐位不一致本身不是否决**：执行形态的改动（组批、编译、硬件精度、分段、移植到新后端）是 A 类，噪音级抖动
  不算损失，刻画即可；改模型、输入或算法是 B 类，先写好指标与门槛，闸门在「接受 / 翻默认值」。判不准就按更严的
  一类记一笔往下走，别花时间争归类（owner 2026-09-17、2026-09-19）。全文：[README_DEV 开发原则](README_DEV.md#开发原则)。
- **选项默认值只在后端**：argparse 传 `None`，由签名或共享 resolver 回答；`config.toml` 的标量在
  `finesub.settings` 声明一次、用 `settings.X.resolve(explicit)` 读；开关用 `BooleanOptionalAction` 或
  `auto|on|off`。优先级链的中间两层（逐键的项目 / 全局配置、前端默认值）还不存在。全文同上。
- **resume 延续任务自己的产物与当时的选择**：参数不同不是让产物失效的理由；只有损坏、身份不符、契约读不了才失效，
  只在「直接续跑会报错或丢数据」的形状变化时作废。provenance 指纹只告警，compatibility key 才失效。全文：
  [复用的依据是任务身份](README_DEV.md#复用的依据是任务身份不是质量判断)。
- **GPU 是预期配置**（≥4GB 显存，`--gpu-tier` 默认 `auto`）；CPU 回退必须往 stderr 打 `Warning:`；没检查过别假设
  没有 GPU。「能不能用 GPU」只问 `speech/runtime/device.py`，不用裸 `torch.cuda.is_available()`。换 CT2 wheel 前读
  `tools/wt_refine_port/ct2-patches/README.md`。
- **阶段按存在性跳过**，不校验内容：要重跑就删该阶段产物及其下游。
- **依赖只写 `pyproject.toml`**，不建 `requirements.txt`；生产入口直接调函数，不起子进程。
- **prompt 文本不硬编码在 Python 里**，模板在 `src/finesub/llm/prompt_templates/`。
- **知识库 `knowledge/` 不被主仓跟踪**；真相源是 `knowledge/knowledge.sqlite`，`knowledge/rendered/` 是可编辑的投影
  （下次纠错运行时收割），见 `docs/manual/knowledge.md`。
- **从 checkout 运行用 checkout 自己的数据**（`knowledge/`、`.env`、`.state`），不碰 `%LOCALAPPDATA%`；
  `FINESUB_CHECKOUT_DATA=0` 关掉。git worktree 解析回主 checkout，worktree 里的知识库自动写入默认跳过
  （`FINESUB_KNOWLEDGE_WRITE=1` 才写）——要从 worktree 更新主仓知识库，先问开发者。
- **不背向后兼容**：接口直接改，旧产物重跑即可；但 prompt 升版不作废未完成任务已提交的窗口与阶段产物，续跑照用并告警，只重做渲染出的输入变了、或当前校验器不再放行的窗。
- **改管线或 LLM 行为时，测试与 owner 文档同一次改**（owner 文档见 [`docs/architecture.md`](docs/architecture.md)）。
  改了用户看得见的行为，在 `CHANGELOG.md` 写一条；加了用户能给的选项，在 `docs/manual/tuning.md` 落一行；
  只给开发用的开关（环境变量）不进 manual，写进它所属的开发者文档。

## 项目约定

- **目录**：产物只进四个被忽略的目录，不散在仓库根——`assets/`（原始媒体，清单在本地 `assets/index.md`）、`data/`
  （人工标注、精修字幕、缓存，**重跑不清理**；规则在 `docs/data-index.md`）、`out/`（生成物）、`tmp/`（草稿）。
  从 checkout 运行时仓库根还会有运行时状态（`.state`、`.env`、`agent-sessions.jsonl`、`.task-activity/`、`cache/`），
  那是 checkout 代替了 `%LOCALAPPDATA%\FineSub`，也被忽略。`tools/` 只按需维护，不作为别的改动的副作用去更新；
  它的测试不进默认套件。一次性实验放 `explore/<名字>/`：跟踪，发布时剥离（脚本常写死本机路径、点名素材），公开文档只点名 `explore/` 这个目录、不点名里面的条目；清单在它自己的 README。
- **路径**只经 `src/finesub/paths.py` 解析；`src/` 里别处不用 `parents[N]` 找仓库根。
- **文档**：给使用者的在 `docs/manual/`，给开发者的在 `docs/` 根；全量地图与状态在 `docs/README.md`，写文档的规矩
  （先写给人、只写现在成立的事、按标题名引用、plan 开头是状态块）也在那里。
- **私有笔记**：`docs/plans/`、`docs/archive/`、`docs/reports/` 在 `dev` 上跟踪，发布时剥离，公开文档不点名、不链接
  里面的文件。往里移文档前，把仍然成立的事实提回跟踪的公开文档（owner 文档或新建一份），同一次修好断链。
- **分支**：日常在 `dev` 上开发；公开的 `main` 只由发布脚本写入，不手工提交，也绝不把 `dev` 推上公开远端。
  worktree 放 `.worktrees/<槽位>/`；清掉槽位（`git worktree remove`）之前先 `git -C <槽位> clean -xdn` 看被忽略的文件：
  git 会挡住未提交、未合并的改动，却不声不响删掉被忽略的文件。其中以后还会被读的（文档引用它，
  或复核结论、开着的线要用它）挪进主 checkout 的 `data/` 并在 `data/index.md` 登记一行，还会重跑的脚本提交进 `explore/<名字>/`
  （分支已合并就在主 checkout 提交）；其余随 `git worktree remove` 一起删，花过额度也一样。每次提交后报告当前分支与提交哈希。

## 按任务找入口

| 要做的事 | 先读 | 判断提示 |
| --- | --- | --- |
| 改某个模块 | [`docs/architecture.md`](docs/architecture.md) 对应一行 | 那一行写着 owner 文档与改之前必须知道的；✱ 的约束有测试钉着 |
| 改用户看得见的行为或选项 | `docs/manual/index.md` → 对应 manual | 用户向描述以 manual 为准；新增或改名一份 manual 要同步 `index.md`（覆盖面有测试，那句话说得对不对没有）；`troubleshooting.md` 只做路由，别把内容搬进去 |
| 让 agent 替用户办事 | `agent-tasks/`：`run-audit/`（审一次 run 的产物与知识库，离线只读）、`knowledge-ingest/`（把资料收进知识库）、`feedback-pack/`（打反馈包）；约定在 `agent-tasks/README.md` | 一个子目录一件事，主文件 `SKILL.md`；**没有 harness 会把它们注入上下文**，任务对得上就自己整份读完再动手 |
| 找数据、跑实验 | `docs/data-index.md`，逐条清单在本地 `data/index.md` | 动性能前先读 `docs/bench-discipline.md`（短）；`docs/bench-baselines.md` 的节号一律不重编号 |
| 语音链路（VAD / ASR / 分句 / 分离器） | `docs/vad-energy.md` `docs/vad-asr.md` `docs/asr-align.md` `docs/asr-stabilize.md` `docs/segmentation-split.md` | 动分离器性能 / 并发前读 `docs/separator-optimization.md`；审分割质量前读 `docs/segmentation-gold.md`；单文件 WT 分片**已移除**（回溯点 `dev` 的 `1fcc4e1`），别按 `docs/wt-parallelism.md` 实现 |
| speech 侧还没做的事 | `docs/speech-followups.md` | 状态只写在这里、只改这里 |
| 已经定了的事（不做的、取舍） | `docs/decisions.md`（speech、评测与跨层）；`docs/llm_design_notes.md`（LLM 层） | 重开或改之前先读那条的理由与重开条件；follow-up 里只留指过去的一行 |
| 说话人分离 | `docs/diarization-findings.md` | 能力边界只认 findings；过程记录已归档，别从那里推断现状 |
| LLM 运行时 | `docs/llm_harness_behavior.md` | 总入口；纠错窗口的 CSV 输出契约在它的「Prompt 信息」一节 + `src/finesub/llm/output_protocol.py`；任务级并行的现行为在它的「任务级并行与 agent 槽位预算」与 `docs/llm_harness_research.md`「超长素材分块调查」；conversational 已冻结（`docs/llm_design_notes.md`「`pseudo-conversational` 与 conversational 冻结」） |
| Agent 执行后端 | `docs/llm_local_agent.md` [§12.1 会话形态](docs/llm_local_agent.md#121-会话形态四档已决定什么还差什么接在哪) | 「什么已接线」只看那一节；动 agent 传输前读 `docs/llm_agent_tool_protocol.md`（不碰用户全局设置、不为过度保守的安全策略加机制） |
| LLM 侧还没做的事 | `docs/llm_followups.md` | 唯一入口；别从归档历史推断现状 |
| 知识库 | `docs/knowledge.md` | 行文法 `[标记] 行体` 的 owner 锚点在 `docs/llm_design_notes.md`「知识库行文法 v3 的决策记录」，动结论前先看 |
| 迭代纠错 prompt | `docs/prompt-iterate.md`、`docs/session_replay.md` | replay 默认调 API；`python -m tools.session_replay correction --dry-run` 只看计划 |
| 分发、报告、下载 | `docs/reporting.md` `docs/download-routes.md` `docs/ct2-distribution.md` | 这几份是 owner 文档，契约细节以它们为准；动多前端并发前读 `docs/cross-frontend-lease.md` |
| 产物哪些可删 | `README_DEV.md`「任务目录的清理与保留」 | 有两类故意不删；清单在 `src/finesub_bootstrap/artifacts.py` |
| 其余文档 | `docs/README.md` 的文档地图 | 全量清单与状态只在那里 |

## Claude Code 专用

- 暂不用 EnterWorktree / ExitWorktree（它的隔离守卫会拒掉大部分复合命令；守卫放宽后改回 `path` 形式，
  `.claude/settings.json` 的 `worktree.baseRef: "head"` 让它从本地 HEAD 分叉；这个文件只在 `dev` 上，公开 clone 里没有）：会话留在主 checkout，用
  `git worktree add .worktrees/<槽> -b <分支>` 开槽，合并就在主 checkout 里 `git merge <分支>`。
  由于你实际位于主 checkout，因此请务必注意环境的区别：相对路径、Bash 的 cwd、Grep / Glob 的默认搜索范围和自动加载的入口文件，都是主 checkout 的。
  - 读改槽里的文件用槽内绝对路径；搜索给 `path=<槽>`。不给时，Grep 只搜主 checkout（`.worktrees/` 被 `.gitignore` 挡着，槽里的文件搜不到）；Glob 则把主 checkout 和槽里的同名文件一起列出来。两种情况下都别把主 checkout 的文件当成槽里的。
  - git 用 `git -C <槽>`。
  - 要在槽里跑的命令写成 `(cd <槽> && …)`。括号让 `cd` 只在这一条命令里生效；不带括号时，harness 会把会话的主工作目录切进槽，之后删槽可能报 Permission denied。
