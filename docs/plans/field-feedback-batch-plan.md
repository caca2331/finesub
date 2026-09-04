# 一轮用户反馈带出的五项（2026-09-03）实施计划

状态：**五项全部实施完毕**（2026-09-03 起草、owner 逐条拍板、两轮复审、同日落地）。
六条 owner 决定在 §7，两轮复审的六处改动在 §8，落地过程中偏离计划的四处在 §9。
**无未决项。** 现行行为的真相源是代码与各 owner 文档；读这份是为取舍依据。

同一轮反馈带来五件互不相干的事。三件围着**开发者反馈闭环**转——用户装不上模型（§1）、
出了问题拿不到现场（§4）、拿到现场也看不出 API 那一层发生过什么（§3）；另两件是独立旋钮
（§2 的配置护栏、§5 的跳过分离）。它们共用一个分支只是因为同一次对话带出来，彼此没有依赖，
**可以任意顺序单独落地与验收**。

本文是取舍依据与执行顺序。现行行为的真相源始终是代码与 owner 文档，不是这份计划。

---

## 0. 五项一览

| § | 事 | 面 | 风险 | 依赖 |
| --- | --- | --- | --- | --- |
| §1 | HF 镜像下 Xet 401 导致 `cn` 装不上模型 | `finesub_bootstrap/model_fetch.py` 两个函数 | 低，且现状是全坏 | 无 |
| §2 | 模型组窗口下限的 warning / 退出 | `llm/routing/` 一处校验 + 一处调用 | 低，出厂 preset 一条都不响 | 无 |
| §3 | 关键 API 交互（LLM + 搜索）进 run 日志 | `llm_runtime.py` / `web_search.py` / `client.py` 各一处 | 低（只加上报点） | 无 |
| §4 | 反馈打包 agent-task | 新目录 `agent-tasks/feedback-pack/` | 中（隐私面） | §3 落地后包更有用，但不阻塞 |
| §5 | `--no-separate` 跳过人声分离 | `stages.py` + separator 一处提升为公开函数 | 中（碰产物**与上报**契约，见 §5.4） | 无 |

建议顺序 **§1 → §5 → §2 → §3 → §4**：前两项面小、能独立验收；§4 最大且隐私面最贵，放最后。

---

## 1. HF 镜像下的 Xet 401

### 1.1 反馈与现状核查

反馈原文：`huggingface_hub` 1.x 默认走 Xet，而 HF 镜像不代理 Xet——镜像返回的元数据仍指向官方
`cas-server.xethub.hf.co`，随附的短期令牌该 CAS 不认，于是元数据请求成功、下载在第一个
reconstruction 请求上 `401 Unauthorized`，`cn` 路由下模型完全装不上。

核查结果，两条都成立：

- `src/finesub_bootstrap/pylock.win-py312.toml` 锁的是 `huggingface-hub 1.26.0` + `hf-xet 1.6.0`
  （两份 lock 都有 `hf-xet`），全仓 grep 不到任何 `HF_HUB_DISABLE_XET`。**Xet 默认开着。**
  （查证时它还在 `desktop/runtime/`；desktop-split 阶段 A 把它搬到了 `finesub_bootstrap`，
  这里写的是合入 `dev` 之后的位置。）
- ⚠ **还有被忽略的第二半**：`model_fetch.py:144` 的 `is_mirror_failure` 里，
  `NETWORK_FAILURE_MARKERS` 有 `404 / 429 / 502 / 503 / 504`，**没有 401 / 403**；而该函数明写
  「无法识别的失败不算镜像的锅」。所以今天的行为是 Xet 401 → 判为不可重试 →
  **连回退官方源都不会发生**，整次失败。这正是「完全装不上」而不是「慢一点」的原因。

  ⚠ **为什么 httpx 那条兜底没救下它**（复审补，2026-09-03）：`is_mirror_failure` 在标记表
  之前还有一条 `isinstance(error, httpx.HTTPError) → True`（`model_fetch.py:178`），而锁定的
  huggingface-hub 1.x 正是走 httpx——**进程内**它本该被判为可重试。真正让 401 掉进标记表的是
  **两个前端都在子进程里下载**（`model_ensure.py:107` 的 `_download` 起 `python -m` 跑自己，
  桌面 prefetch 同理；「A subprocess is not fastidiousness」那段理由在 `model_ensure.py:18`
  的模块 docstring，`model_fetch.py:112` 的注释也记着「异常过不了进程边界、只剩它的文本」）：
  异常类型回到父进程只剩一段文本。**所以加 401/403 标记补的是「跨进程」那一半**——将来谁把下载
  改回进程内，兜底靠的是 httpx 那条分支而不是这张表，两者缺一都不完整。

### 1.2 两处修改

| | 改哪 | 怎么改 |
| --- | --- | --- |
| 主修 | `model_fetch.py:53` `apply_hf_endpoint`、`:73` `fetch_with_fallback` | 凡是往 env dict 里写非官方 `HF_ENDPOINT` 的地方，同时写 `HF_HUB_DISABLE_XET=1`；回退到官方源那一支把它一起 `pop` 掉（那一支已经在 `pop(HF_ENDPOINT)`） |
| 兜底 | `model_fetch.py:144` `is_mirror_failure` 的标记表 | 认 401 / 403：镜像返回的认证错误恰恰是官方源能救的那类。**它补的是跨进程那一半**（见 §1.1 的 ⚠），进程内那一半由既有的 `httpx.HTTPError` 分支承担 |

两个函数是**两个前端唯一的公共通路**（`RuntimeEnvironment.worker_context` 经
`environment.py:177` 调 `apply_hf_endpoint`），所以改这里就覆盖 CLI 的惰性下载与桌面的 prefetch。

兜底那条不是锦上添花：没有它，下次换个镜像出同类问题仍然是一步到位地死，而不是降级。
它也不违反 `is_mirror_failure` 的既定原则——401/403 是**被识别的**状态码，不是「不认识就赖镜像」。

### 1.3 取舍：用户自己设的 `HF_ENDPOINT` 也关 Xet

现有原则是「用户已显式设置 `HF_ENDPOINT` 时不覆盖——他们有意指向那里，一个地区猜测不足以
推翻它」（`download-routes.md` §4）。**这条原则对 endpoint 保留，对 Xet 不适用。**

理由不是地区猜测，是事实判断：**官方 endpoint 是唯一能正确代理 Xet 的那个**。所以判据改成
「`HF_ENDPOINT` 非空 → 关 Xet」，与这个值是我们设的还是用户设的无关。留一个出口：用户显式
设了 `HF_HUB_DISABLE_XET` 时（含设成 `0`）完全不动——企业内网真有 Xet 能力的网关照样能开。

**代价核过**：关掉 Xet 就是走普通 HTTP range 下载，而镜像本来只能提供这个，`cn` 路线没有
吞吐损失；`global` 不设 endpoint，因此完全不受影响、继续吃 Xet 的去重加速。

### 1.4 验收

- env 形状：镜像 → `HF_ENDPOINT` 与 `HF_HUB_DISABLE_XET` 都在；官方回退那一支 → 两个都不在；
  用户预设 `HF_HUB_DISABLE_XET=0` → 我们不改它。
- `is_mirror_failure` 对 `401 Unauthorized` / `403 Forbidden` 返回真，且 `LOCAL_FAILURE_MARKERS`
  的既有用例不受影响。
- 文档：[`download-routes.md`](../download-routes.md) §4 补一句「镜像路线一并关闭 Xet 及其理由」，
  [`manual/resources.md`](../manual/resources.md) 的环境变量表加 `HF_HUB_DISABLE_XET` 一行。

---

## 2. 模型组窗口下限的 warning 与退出

### 2.1 要的是什么

模型组里存在成员满足下面任一条时报 warning；更低的一档直接退出：

| | 最大输出 | 最大输入 |
| --- | --- | --- |
| warning | < 64,000 | < 194,000 |
| 退出 | < 32,000 | < 96,000 |

⚠ **阈值用十进制而非 2 的幂**（owner 决定，§7）。这一个字有实际后果：`local-claude-haiku-4_5`
的 `max_output_tokens` 是 **64000**，写 65536 它的**输出**就常态告警，写 64000 则不响。

### 2.1.1 闸门比的是 catalog 两列，不是规划包络

这一节是两轮复审 + owner 裁定改出来的，**是 §2 里最容易做错的地方**。

第一轮复审问：`group_planning_envelope` 的输入上限是 `min(max_input, context_window −
min_output)`，haiku 是 ctx 200000 / out 64000，包络算出来 **136000**，立刻撞 194,000——
那 haiku 到底该不该响？

第二轮复审纠正了配套的机制描述：**catalog 里全部 `gemini-free-*` 行的 `context_window` 列是
空的**，`model_catalog.py:393` 对空值填 `max_input + max_output`。所以对这些行，
`ctx − min_output` 按构造就 ≥ `max_input`，包络等于 194,000 **不是巧合、是填充规则的结果**。
换句话说：**没写 ctx 的行，包络比的其实就是 `max_input_tokens`；只有显式写了 ctx 的行
（haiku 的 200000）才真的走那个减法。**

**owner 裁定（2026-09-03）：haiku 放行——「in / out 两边的量都足够，虽然总量受限，
但缩放的形状健康」。** 所以：

> ✱ **闸门比 catalog 的 `max_input_tokens` 与 `max_output_tokens` 两列本身**（取组内最小），
> **不比 `group_planning_envelope` 的包络**。

理由就是那句裁定：`context_window` 表达的是「输入与输出不能同时吃满」这个**总量**约束，而
闸门问的是**形状**——这个成员单看输入够不够长、单看输出够不够写。总量受限是模型的正常形态，
不是配置错误。包络是**规划**用的，它必须保守（规划时还不知道谁来答），闸门不承担那个职责。

⚠ 别把这两个量混起来用：`planning_limits_for` 继续用包络，一个字不动。

按这条规则重新核过全部 catalog 行：**haiku 恰好压在两条线上**（in 200000 ≥ 194,000、
out 64000 ≥ 64,000），其余可达成员全部有余量，**一条都不响**——阈值就是照这个形状定的。

### 2.2 落点：`ModelRouteCatalog` 上一个新的只读方法

按 §2.1.1，闸门要的是**组内 `min(max_input_tokens)` 与 `min(max_output_tokens)`**，
现成的方法都不是这个：`group_planning_envelope`（`model_routes.py:535`）返回的是规划包络
（input 那一半已扣掉 output），语义上不能借用。

所以加一个并列的只读方法，比如 `group_declared_minima(group_id) -> tuple[int, int]`：
两行 `min(...)`，取的是 `target_fact(...)` 的两列原值。它与 `group_planning_envelope` 同层、
同数据源，只是不做减法——**两个方法各自回答一个问题，不要合并**（合并的那一刻就会有人
在闸门里用包络，或者在规划里用原值）。

### 2.3 三条边界（否则会误伤）

1. ⚠ **扫 `model_groups`，绝不扫 catalog。** `gemini-free-gemma-4-31b` 的
   `max_input_tokens=16000`，扫 catalog 会直接把它判成退出档——但它**不在任何 model group 里**，
   只是本地检索代理 grounded search 用的 target，本来就不参与纠错。
2. **只校验当前 preset 可达的组**（cell 绑定到的那些）。`model_routes.toml` 里还躺着
   `agy-*` / `dsh-*` 等本次运行根本用不到的组，整份扫会为别人的配置退出。
3. **一视同仁，`lightweight-default` / `search_judge` 也照这张表**。它们今天的数值同样是
   194000 / 65536，过得去；先不做例外，出现真实反例再拆（§6 记着这是有意的暂缓）。

按出厂三个 preset 枚举全部可达组：declared minima 一律 **in=194000 / out=65536**，
**一条都不会响**。这是给自定义配置与未来新模型准备的护栏，不改变现状。

### 2.4 在哪儿退出

⚠ **不能等 LLM 阶段开始才退**——那时 ASR 已经跑了几十分钟，退出等于把整次运行扔掉。
落点在 `run_pipeline` 入口：`--stage` 达到 `translated-srt` 时先校验一次，不达到就不校验
（纯 ASR 运行不该被 LLM 配置挡住）。warning 走 `reporter.warning`（有 code / impact / action），
退出走既有的失败路径，消息里写清是哪个组、哪个成员、哪个数字、门槛是多少。

### 2.5 验收

- 构造一个含小窗口成员的组：warning 档只报不停、退出档在 ASR 之前就停；
- 出厂 preset 全部可达组静默通过（回归护栏，防止以后调 catalog 时把自己锁死）；
- `gemini-free-gemma-4-31b` 不被卷进来（就是第 2.3 条第 1 点的回归）；
- ✱ **haiku 放行的回归**：一个只含 `local-claude-haiku-4_5`（in 200000 / out 64000）的组
  静默通过。它同时钉住两件事——阈值是十进制，以及闸门比的是 catalog 两列而不是包络
  （用包络的话输入算出来 136000，这条测试立刻红）。
- 文档：[`manual/model-routing.md`](../manual/model-routing.md) 写用户看到这条 warning 该怎么办。

---

## 3. 关键 API 交互进 run 日志

### 3.1 基础设施已经齐了，缺的是上报点

落盘 run 日志 `<user-data>/logs/run-<时间戳>-<输入名>.log` 是**恒定 verbose** 的
（[`reporting.md`](../reporting.md) §6），与 `--log-level` 无关。所以 `reporter.debug(...)`
会无条件进文件、终端保持干净——这条需求不需要任何新机制。

盘过现有覆盖：

| 位置 | 现状 |
| --- | --- |
| `rate_limit.py` / `search_loop.py` / `research.py` / `content_filter.py` / 纠错窗口与重试 | ✅ 已有 debug / warning |
| **`llm_runtime.py`（真正发 REST 的地方）** | ❌ **零上报**。它有 `_record_api_attempt`（`:219`）把每次尝试记进结构化台账带进产物，但一个字都不进 run 日志 |
| **`web_search.py` 的逐次 provider 调用** | ❌ 只有一条 warning（`:461`），没有「发了什么查询、哪个 provider、几条结果」 |
| **本地 agent 那条路**（`client.py:1200` 一带的 `LocalAgentDriver` 调用） | 部分：`agent/` 自己有 warning/debug，但没有与 REST 同口径的「一次调用一行」 |

### 3.2 三个落点

- **`_record_api_attempt`（`llm_runtime.py:219`）里镜像一条 `debug`。** 它已经握着
  `provider_tier / model / api_key_name（是 label 不是 key）/ return_code / elapsed_sec /
  这把 key 与这个模型的第几次调用`。**一个函数、一处改动，覆盖全部 REST 后端**——
  `chat_complete` 是唯一的 REST 收口（`client.py:2774` 是它唯一的调用者），Gemini REST 与
  OpenAI-compat / Anthropic 三条 transport 都在它内部分支。两个调用点（`:708` 成功、`:737` 失败）
  自动都被覆盖。
- **`web_search.py` 的 `_search_pending_with_provider` / `_extract_pending_with_provider`**：
  每次请求一条 `debug`，带 provider、query、guided query、结果条数或错误。
- **本地 agent 调用点**：同口径补一条，让「这一步是谁答的、花了多久、结果如何」在两条路上一样读。

### 3.3 一行有多长：只写状态与它的一句话描述

⚠ **不写 prompt / 响应正文。** 全文已经在 `<stem>.llm-artifacts/exchanges/*.md` 里逐次留着，
run 日志再来一份会让一次长任务涨到几十 MB，反而不利于用户直接把文件发过来（而那正是这份
日志存在的理由）。每条就是**状态 + 一句话描述**：谁、什么状态、多久、这一步在干什么。
搜索那条带上 query 本身——它本来就短，且没有它这行等于没说。

判据：**一条日志行不应超过一屏宽度的量级**。要看正文就去 `exchanges/`，所以行里带上对应
exchange 文件名，让两者能对上。

### 3.4 验收

- 一次 `--stage translated-srt` 的运行，日志里能逐条读出「哪个模型答的、返回码、耗时、第几次」，
  以及每次检索的 provider 与查询；
- 日志里**不出现** API key（label 不是 key）、不出现 Files API 的 resumable session URL
  （[`reporting.md`](../reporting.md) §5 已有这条禁令，本次扩大了上报面，回归要覆盖到）；
- 现有的 `test_pipeline_log_shape.py` 仍绿（尤其「日志不拖第三方 logging 与 tqdm」那条）。

> 顺带（已在本计划的提交里改掉，不留作待办）：[`reporting.md`](../reporting.md) §2 的
> renderer 表把落盘日志写成「见 §5」，而那一节的标题是 §6。守卫查不出来——§5 确实存在，
> 只是指错了地方。

> **落地时的一处收缩（2026-09-03）：行里不带 exchange 文件名。** §3.3 原本要求带上，
> 但那个文件名由上面一层（`ExchangeLogger`）掌握，为一行日志把它穿到 transport 里会给
> `llm_runtime` 增加一个它不需要的依赖。改用内容对齐：exchange 正文渲染的就是
> `api_attempts` 这份清单（`exchange_log._render_api_attempts`），所以两边本来就对得上。
> 本地 agent 那条的落点是 `local_agent._report_attempt`，由 `_run_episode` 的
> `finish_attempt` 调用——每个 CLI attempt 的唯一收口，成功失败同一处，不必按 transport 分。

---

## 4. 反馈打包 agent-task

### 4.1 形态：`SKILL.md` + 一个脚本，不是纯文字

新目录 `agent-tasks/feedback-pack/`，主文件 `SKILL.md`，确定性搬运放
`agent-tasks/feedback-pack/scripts/`。

**为什么要脚本**：产物布局的真相在 `finesub_bootstrap/artifacts.py` 与 `stages.PipelinePaths`
里，让 agent 每次手抄一遍路径清单，迟早和代码分家——这正是 `agent-tasks/README.md`
「契约细节归 `docs/`，这里引用、不复制」的同一条道理。分工是：**脚本做搬运，`SKILL.md` 做判断**
（收哪些 task、跟用户确认什么、包里有什么要念给用户听）。

这份是**用户向**的（不像 `release/` 与 `desktop-portable/` 那两份维护者专用），所以它进公开
快照，并且要在 [`manual/agent-tasks.md`](../manual/agent-tasks.md) 的能力清单里加一行。

### 4.2 两个模式

| | 模式 A `debug`（用户碰到问题） | 模式 B `corpus`（用户愿意贡献数据） |
| --- | --- | --- |
| 收谁 | 用户指名的 task（或最近若干次运行） | 尽量多的、**有精修字幕**且台账里没记过的 task |
| 音频 | 有 | **无** |
| 文本 | SRT 各档、`-annotated.csv`、`-stable.json`、`run-metadata.json`、`correction-windows.jsonl`、`exchanges/`、对应的 run 日志 | 同左，去掉体积最大的 `-stable.json` |

多个 task 整体打成**一个 zip**，落在用户桌面。

### 4.3 五个已定的取舍

1. **音频收 `<stem>-vocal.ogg`，不是 flac、不是源媒体。** 16 kHz 单声道，一小时几 MB；
   `.flac` 是无损交付、源媒体常是几百 MB 的视频，扔到桌面上是在给用户添麻烦。**例外**：
   问题本身就出在分离阶段时才另收源媒体，且必须先问过用户——判据写进 `SKILL.md`。
2. **「有精修字幕」由执行的 agent 判断**（owner 决定，§7）。不新增文件名约定、不改
   `run-metadata.json`：精修字幕是用户手上的东西，路径由 agent 问出来或认出来，作为参数交给脚本。
   脚本不猜。
3. **去重台账**放数据根下 `feedback/packed.json`。⚠ **键不能只是路径**——否则用户改进了精修
   字幕以后永远再也传不上来。键取 `(task 路径, 最终 SRT 的 mtime+size, 精修字幕的 mtime+size)`，
   并且 A / B 两种用途**各记一份**：同一个 task 可以先作为 bug 现场发过一次，再作为语料发一次。
4. **落点**：`Path.home()/"Desktop"`，不存在就退回 `Path.home()`（Windows 上桌面常被 OneDrive
   重定向，不能写死）。文件名 `finesub-feedback-<mode>-<时间戳>.zip`。最后把**绝对路径**打给用户。
5. **隐私——这是本节最贵的一段**：
   - 硬排除 `.env*` 与任何 key 形态的文件；`config.toml` 只按**白名单键**收，不整份塞进去。
   - 包里生成 `MANIFEST.txt`：有哪些文件、每一类是什么、里面会包含什么（绝对路径含用户名、
     字幕全文、知识库条目全文）。agent 把它念给用户，由**用户**决定发不发。
   - ⚠ **agent 不替用户把包发出去**——不上传、不贴 issue、不发邮件，只给路径。这既是产品决定，
     也是 harness 的安全边界。这条要写进 `SKILL.md` 的「不要做什么」。

### 4.4 验收

- 模式 A 在一个跑完的 task 上产出 zip，解开后 run 日志、`exchanges/`、`-vocal.ogg` 都在；
- 模式 B 连跑两次，第二次因台账而跳过全部已收 task；改动其中一个 task 的精修字幕后，
  第三次只收那一个；
- 包里不含 `.env`、不含 key；`MANIFEST.txt` 覆盖了包里每一个文件。

---

## 5. `--no-separate`：跳过人声分离

### 5.1 现状：已经有事实上的绕法

`stages.py:525` 按 `resolve_vocal_audio()` 的**存在性**跳过分离，所以预先放一个
`<stem>-vocal.ogg` 就能达到目的。这一项要的是把它变成显式选项。

### 5.2 选项形态

`--separate / --no-separate`（`BooleanOptionalAction`，argparse 传 `None`，默认落在
`run_pipeline` 签名 + 后端 resolver），与 `--word`、`--vad-silero-assist` 一致，符合
`CLAUDE.md` 的选项默认链。**不叫 `--skip-separation`**：依据是 `CLAUDE.md` 那条
「新开关用 `BooleanOptionalAction` / `auto|on|off`」，不是全仓惯例——⚠ **全仓惯例这个说法
不成立**（复审纠，2026-09-03）：`pipeline.py:352` 的 `--no-download-video` 与 `:423` 的
`--no-resume` 都是只有反面的 `store_false`。它们是既有形态，本计划不动它们。

### 5.3 实现走 (b)：跳过分离，但仍产出同契约的 `-vocal.ogg`

两条路线，选后者（owner 决定，§7）：

- **(a) 把源文件直接当 vocal 轨喂给下游。** 改动面大：VAD / ASR / qwen 复核 / 词首修正每一个
  消费点都假设「那个文件在」，(a) 要在每处解释「有时不在」；视频输入还得先抽音轨。
- **(b) 跳过分离本身，用既有的编码路径把源转成同契约的 `<stem>-vocal.ogg`（16 kHz 单声道）。**
  复用现有的存在性跳过、resume 语义与产物树，改动最小；代价是几秒转码与一份几 MB 的文件。

(b) 的复用点是现成的：`separator/separation.py` 的 `_encode_asr_delivery(merged_path,
output_path)`（`:959`，「Write the 16 kHz mono Vorbis delivery」）——把它提升为公开函数即可；
压缩源与视频容器先走 `speech/preprocessing/audio.py:141` 一带**已有的**
`transcode_to_lossless_audio` 准备路径，那也正是分离自己在用的那条。

⚠ 严格说下游读者本来就会重采样（`audio.py` 的 `TARGET_SR = 16000`），所以 (b) 不是正确性
要求；它的价值是**把「跳过分离」变成一次可复用的产物**，让 resume 与存在性跳过的语义一个字
都不用改。

### 5.4 三件配套

1. **不联动改 `--vad-silero-assist` 的默认值。** 它默认 on 的理由写在 help 里——「the pipeline
   separates first and this exists for that kind of noisy vocal」——跳过分离后这个前提确实不
   成立，但自动联动违反选项默认链，而且纯人声也可能有底噪。做法是在 `--no-separate` 的 help
   与文档里明写「纯净人声通常配 `--no-vad-silero-assist`」，把决定权留给用户。
2. **`run-metadata.json` 里给一个新的 status `skipped`**，别复用 `reused`——否则事后查
   「这次到底分没分离」查不出来，而这恰恰是要拿去 debug 的那个字段。

   ⚠ **改动面不止 `run-metadata.json`**（复审补，2026-09-03）。今天的词表只有
   `executed` / `reused`（`stages.py:277`），而且有两个消费点按二值处理：`stages.py:301`
   用 `== "reused"` 判断本次运行要不要保留上一趟的 `executed` 记录，`stages.py:498` 把它折成
   `reporter.stage_started(stage_key, reused=status == "reused")`。**加第三个值而不动这两处，
   终端上跳过的分离会显示成「running」。**

   > **落地时的裁决（2026-09-03）：上报契约不加第三态。** 因为走的是路线 (b)，跳过时**确实
   > 有东西在跑**（同一份产物，转码而来），所以「在跑」这件事没有变，`reused`（= 这一趟什么
   > 都没跑）仍然是准确的 `False`；换了哪条路由既有的 `detail` 参数说。给 `reused` 加取值会
   > 让四个 renderer 各长一个它们并不需要区分的分支。⚠ 反过来把 `skipped` 折进 `reused` 是
   > 错的——那会显示「已有结果，跳过」，而实际上产物是这一趟新转出来的。
   > `stages.py:301` 的那处仍然要改（`skipped` 与 `executed` 同样算「跑过」）。
   > 记在 [`reporting.md`](../reporting.md) §1 的 `stage_started` 条。
3. **GPU 档位不动。** 档位决定分离器实例数与裁判放哪；跳过分离时分离器不加载，裁判那半仍然
   需要档位（[`gpu-profiles.md`](../gpu-profiles.md)）。

### 5.5 验收与文档

- 纯人声输入 `--no-separate` 跑通到 `raw-srt`，`-vocal.ogg` 是 16 kHz 单声道，
  `run-metadata.json` 里 `vocal_separation` 为 `skipped`；
- 同一 task 再跑一次走存在性跳过，不重转码；
- ⚠ 文档：[`manual/tuning.md`](../manual/tuning.md) **必须加一行**（`CLAUDE.md` 的硬要求：加了
  用户能给的选项就要在旋钮总表落一行），外加 [`manual/outputs.md`](../manual/outputs.md)、
  `README_DEV.md` 的产物树、[`separator-optimization.md`](../separator-optimization.md)。

---

## 6. 明确不做

- **不把 §2 的门槛做成 per-task-group 的表。** 今天所有可达组数值相同，先一视同仁；
  出现真实反例（例如轻量组确实只需要 96k 输入）再拆，那时才知道该拆成几档。
- **不给 §3 加「日志里带 prompt 全文」的开关。** 正文在 `exchanges/`，两处全文是重复而不是冗余；
  真需要时该做的是让 §4 的包更好收，不是把日志撑大。
- **§4 的 agent 不发送任何东西。** 不上传、不贴 issue、不发邮件——只产出文件并给路径。
- **§5 不自动推断输入是否已是纯人声。** 判断由用户给，猜错的代价（该分离而没分离）是整条
  链路质量塌掉且不报错。
- **§1 不为「镜像未来可能支持 Xet」留探测逻辑。** 一次探测换不到任何东西：关掉 Xet 的代价是零
  （见 §1.3），而探测本身是新的失败面。

---

## 7. owner 决定台账（2026-09-03）

| # | 决定 |
| --- | --- |
| 1 | §1 全部按方案：主修 + 401/403 兜底，且用户自设 `HF_ENDPOINT` 时同样关 Xet |
| 2 | §2 全部按方案：阈值取十进制 64,000 / 32,000 / 194,000 / 96,000；在 `run_pipeline` 入口、`--stage` 达 `translated-srt` 时校验；只校验当前 preset 可达的组；轻量组一视同仁 |
| 3 | §3 的「文本描述」**指的是状态附带的一句话描述，不是 prompt / 响应正文**——这条澄清定死了 §3.3 的口径 |
| 4 | §4 的「哪些 task 有精修字幕」**由执行的 agent 判断**，不新增文件名约定、不改 `run-metadata.json`；其余按方案 |
| 5 | §5 按方案，实现走 (b) |

| 6 | **Haiku 放行**——「in / out 两边的量都足够，虽然总量受限，但缩放的形状健康」。这一句定死了 §2.1.1：闸门比 catalog 的两列，不比规划包络；`context_window` 表达的总量约束不进闸门 |

**本计划已无留给 owner 的未决。**（第一轮复审提的那条 194,000 输入门槛的语义，由决定 6 结案。）

## 8. 复审记录（2026-09-03，两轮）

### 第一轮

核实了绝大多数带行号的断言（`group_planning_envelope` 的两遍算法、`llm_runtime.py`
零上报、`_encode_asr_delivery`、`stages.py:525` 的存在性跳过、两份 pylock 的 hf-xet 等），
并揪出四处要改的，全部已并入正文：

| # | 问题 | 落在哪 |
| --- | --- | --- |
| 1 | §5.2 的命名依据「全仓惯例」不成立——`--no-download-video` / `--no-resume` 都是只有反面的 `store_false` | §5.2 改成引 `CLAUDE.md` 的 `BooleanOptionalAction` 条 |
| 2 | §1.1 少了一半解释：`httpx.HTTPError` 分支本该救下 401，真正的原因是子进程边界让异常只剩文本 | §1.1 的第二个 ⚠、§1.2 兜底那一行 |
| 3 | §2.1 拿 haiku 论证输出阈值，但它其实先撞输入线；且 194,000 正好压在今天的地板上 | §2.1 新增两个 ⚠ + §7 的未决 |
| 4 | §5.4 的 `skipped` 改动面不止产物字段：`stages.py:301` / `:498` 按二值消费，UI 会显示成 running | §5.4 第 2 条的 ⚠，`reporting.md` 进必改清单 |

另有一条顺带观察（`reporting.md` §2 指错节号）已在本计划的提交里直接改掉，见 §3.4 的引用块。

### 第二轮

只查第一轮改出来的新文字，揪出两处（都已改）：

| # | 问题 | 落在哪 |
| --- | --- | --- |
| 5 | §1.1 引错文件：「A subprocess is not fastidiousness」在 `model_ensure.py:18` 的模块 docstring，不在 `model_fetch.py`（后者是 `:112` 的一条注释） | §1.1 改成同时引这两处 |
| 6 | §2.1 的「Gemini free 各行 ctx 恰好 259,536、以零余量通过」是错觉：那些行的 `context_window` 列**是空的**，`model_catalog.py:393` 对空值填 `max_input + max_output`，所以包络等于 194,000 是填充规则的结果。真正的分界是「没写 ctx 的行比的就是 `max_input`，只有显式写了 ctx 的行才走减法」 | §2.1.1 整节重写 |

第 6 条把问题问清楚之后，owner 当场裁定 haiku 放行（§7 决定 6），闸门的判据因此从
「规划包络」改成「catalog 两列」，§2.2 的落点也随之从「复用现成方法」改成
「加一个并列的只读方法」。

---

## 9. 实施记录（2026-09-03）

五项按 §0 的建议顺序落地，各自一个提交。全套 `python -m pytest -q` 每一步都跑过，
`desktop/backend/tests/test_model_fetch.py` 因不在根 `testpaths` 里而单独跑。

### 9.1 四处偏离计划的地方

| # | 计划说 | 实际做 | 为什么 |
| --- | --- | --- | --- |
| 1 | §5.4：`reporting.md` 进必改清单，`stage_started` 要么三态、要么显式裁定「跳过算 reused」 | **两个都不是**：`reused` 保持二值，`skipped` 只进产物侧，UI 用既有的 `detail` 说明换了哪条路 | 走的是路线 (b)，跳过时**确实有东西在跑**，所以 `reused`（=这一趟什么都没跑）本来就该是 `False`。加第三态会让四个 renderer 各长一个它们不需要区分的分支 |
| 2 | §3.3：日志行带对应 exchange 文件名 | **不带**，改为内容对齐 | 文件名在 `ExchangeLogger` 那一层，为一行日志把它穿到 transport 会给 `llm_runtime` 加一个不需要的依赖。exchange 正文渲染的就是 `api_attempts` 这份清单，两边本来就对得上 |
| 3 | §3.2：本地 agent「同口径补一条」，位置未定 | 落在 `local_agent._report_attempt`，由 `_run_episode` 的 `finish_attempt` 调用 | 那是每个 CLI attempt 的唯一收口，成功失败同一处，不必按 transport 分三次 |
| 4 | §2.2：闸门「在既有计算上加四行」 | 新增 `group_declared_minima`，与 `group_planning_envelope` 并列 | §2.1.1 的裁定：闸门比的是 catalog 两列，不是包络。两个方法各答一个问题 |

### 9.2 函数大小棘轮拦了两次，两次都改成了拆分

`test_function_size.py` 的棘轮在 §5 与 §2 各红过一次——两次都是「加功能必然让
`run_pipeline` 变长」。处理方式都是**拆而不是抬**：

- vocal 阶段整块 → `_run_vocal_stage`（`run_pipeline` 486 → 441，helper 142）；
- `--style` 与新的窗口闸门这两处 ASR 前校验 → `_validate_llm_configuration`（→ 439）。

只抬了一处：`pipeline.py:build_parser` 414 → 429。它是一串 `add_argument`，
新增一个用户可见选项必然多一块，按主题拆只会把同样的行藏到三个名字后面。

⚠ 这条经验值得单独记：**棘轮红了，第一反应应该是「这个函数里有没有一整块能出去」，
而不是改那个数字**。两次里有两次答案都是有。

### 9.2.1 两轮自查（2026-09-03）

落地后一轮自查 + 一轮外部复审，改出的问题按类记在这里——**同一处 owner 语义被弄错过两次**，
值得单独看：

| # | 问题 | 教训 |
| --- | --- | --- |
| 1 | `encode_asr_delivery` 对解码临时文件的归属**连错两次**：先是 `finally` 里无条件删（丢掉「失败了让重跑省一次解码」），自查时改成完全不删（在任务目录里永久留下几百 MB 的无损副本）。正确形状是 `ensure_decodable_input` 的契约、也是 `run_vocal_separation` 一直在做的：**成功才删，失败保留，scratch 记录负责善后** | 抄一个既有函数的行为时，去读那个函数，不要凭记忆复述它。两次注释都写着「和 `run_vocal_separation` 一样」，两次都不一样 |
| 2 | `feedback-pack` 按目录取文件，显式 `-o` 的布局下会收进邻居任务的产物；改成按 stem 过滤之后又漏了分隔符，`a` 仍会认领 `abc-raw.srt` | 「同一个目录」不等于「同一个任务」，而「同一个前缀」也不等于 |
| 3 | `config-excerpt.toml` 用 `repr` 写值，`True` / `'text'` 都不是合法 TOML | 文件名是一种承诺：叫 `.toml` 就得能被 `tomllib` 读 |
| 4 | `capabilities.py` 里函数内 import `model_routes` 并注「Inverted layer」，但该模块顶部本来就 import 了它 | 从别处抄来的注释会连同它的前提一起抄错 |
| 5 | `manual/resources.md` 的「有哪些节」表漏了 `[separator]`——那是唯一列全节名的地方 | 加一个配置键要问的不只是「旋钮表写了没」 |

⚠ **测试为什么没拦住第 1 条**：管线层的用例把 `encode_asr_delivery` 整个替换掉了，所以两次
都绿。补的那条用例调真函数、只替换它内部的两步，专测临时文件的归属。

第三轮（两位复审各一份）又揪出三处，都在**新写的那些代码**里：

| # | 问题 | 教训 |
| --- | --- | --- |
| 6 | §3 新加的 `why`（端点原话）会把 httpx 引用的请求 URL 一起带进日志，而 `[llm] proxy` 与自定义 `base_url` 允许写成 `https://user:token@host`。仓库自己的 `shell._safe_host` 早就为 `doctor` 立过同一条规矩 | **加一个「把对方的话原样记下来」的字段，就是加了一个外部文本入口。** 现成的先例要去找，不要等复审提醒 |
| 7 | `pack.py` 的 `run_logs_for` 用 `run-*<label>*.log`——任务 `a` 匹配所有带 `a` 的日志，`input` 会收走 `my-input` 的 | 产物侧的同类问题已经改过**两次**（目录≠任务、前缀≠归属），日志侧却是另写的一段 glob，守卫没跟过去。⚠ **改一处归属判据时，去找同一个概念的其它实现** |
| 8 | zip 里的文件夹名只用 `task.label`，两个都叫 `final.srt` 的任务会写进同一个文件夹并互相覆盖（zip 允许重名成员，先写的那份直接没了） | 台账那边早就用了绝对路径当键——同一个事实没有传到归档那一层 |

第 7 条最值得记：**同一个「这东西属于谁」的判断，在这个脚本里被写了两遍**，一遍改对了两次，
另一遍从没被审视过。归属判据应当只有一处实现。

第四轮（放行，但指出两处残留）：`bundle_folders` 给第二个同名任务发 `label-2`，
而包里**可能真有**一个任务就叫 `label-2`——同一种碰撞往后挪了一格，而且同样是静默的；
以及 MANIFEST 的分段标题仍用 `task.label`，于是两个 `final` 显示成两段一模一样的标题，
和 zip 里的 `final` / `final-2` 对不上。两处都已修（生成名要避开所有**已被独占**的 label，
标题改用文件夹名）。

⚠ **第 8 与这两处是同一个错误犯了三次**：修一处重名时，只想着「让新名字与已发出去的不同」，
没想「新名字会不会撞上一个本来就存在的名字」。去重要针对**全集**，不是针对已处理过的那部分。

### 9.2.2 corpus 模式排除的是 `-vad*`，不是 `-stable.json`

§4.2 的表写的是「去掉体积最大的 `-stable.json`」，实现去掉的是 `-vad.json` 与
`-vad-energy.npz`。**实现是对的，计划那一行想错了**：`-stable.json` 是纠错的**输入**，
正是做 prompt 工作的人要看的东西；而 VAD 那两件既大又能从人声轨重建。SKILL.md 写的是实现，
这里补记这处偏离。

### 9.2.3 §5.5 列了 `separator-optimization.md`，这次没动

不是漏掉。那份文档记的是**分离器推理本身**的优化实验（AMP、编译、分块、并发档位），
而「不跑分离」不是一种分离优化，它是管线开关；写进去只会让那份实验记录多一段与它无关的话。
`--no-separate` 的落点是 `manual/tuning.md`（用户旋钮）、`manual/outputs.md`、
`README_DEV.md` 的产物树与复用规则，以及 `reporting.md` 的状态词表。

### 9.3 落地时新写下的三条

1. `speech/preprocessing/separator/separation.py` 的 `_encode_asr_delivery` 提升为公开的
   `encode_asr_delivery`，并**返回输出路径**——`_use_or_create` 发布的是 `create` 交回的东西，
   两个产出同一份产物的生产者得回答同一种形状。
2. 反馈打包脚本用 `default_model_catalog()` 之外的一个教训：**Gemini 系的 catalog 行不会
   自动长出 target**（只有 custom provider kind 会），所以测试里造小模型要**改写已有行**
   而不是加新行。这一条写进了 `test_llm_capabilities.py` 的 fixture docstring。
3. 打包脚本的排除规则是**白名单式的失败方向**：`config.toml` 只按节白名单摘录，
   配置长出新键时它漏在包外而不是被发出去。
