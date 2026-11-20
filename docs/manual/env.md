# 环境变量与 API Key

API key 写在 **`%LOCALAPPDATA%\FineSub\user-data\.env`**：用文本编辑器打开它（没有就新建；第一次运行任务之前连 `user-data` 目录也还没有，一起建上），按下文的格式写 key，
下次运行时自动加密；`finesub keys` 查看已经配好的 key。⚠ 用记事本新建时「保存类型」选「所有文件」，否则会存成 `.env.txt`、读不到；存好后跑一次 `finesub keys`，看得到你的 key 才算配上。key 池怎么选、哪些服务开着，写在同一目录的 `config.toml`，
它的创建方法与内容见 [`resources.md`](resources.md)「设置文件 `config.toml`」。系统环境变量里同名的 key 会盖过 `.env`。

从源码运行时（`python -m finesub.pipeline`），`.env` 与 `config.toml` 默认在仓库根目录（`FINESUB_ROOT` 可显式指定仓库根），也可以用 `FINESUB_ENV_FILE` / `FINESUB_CONFIG_FILE` 分别指定别的文件；**别把**这两个文件提交到 git。
源码运行没有 `finesub keys`（PATH 上若有 `finesub`，那是安装版，读的是另一份 `.env`）：跳过上面那步，直接跑一次任务看有没有缺 key 的报错；要导出明文见下一节最后一条。

## 密钥保护（绑定 Windows 账户）

`.env` 里的 key 会自动加密、绑定这个 Windows 账户：防文件误传与同机别的账户读取，不防以你的身份运行的程序。

- 别动文件顶上的 `FINESUB_KEYRING` 那一行：删了，所有 key 都解不开。
- **换电脑、重装 Windows 或换账户之前**，先 `finesub keys --out <文件>` 导出明文（`NAME=值` 格式），到新机器粘回 `.env`。
- 拷到别的机器上的 `.env` 会显示「未配置」并告警，文件本身不改；把解不开的值**全部**重填，就按新机器重新加密。
- 随时可以往 `.env` 写明文 key，下次运行会加密。
- `finesub keys` 默认打码显示。
- Linux、macOS 上只做混淆、不加密，拷到别的机器也能直接读。这种 `.env` 拷到 Windows 上照样能用，但保持原样、不会改成绑定账户。
- Windows 上保护不了时（系统的加密接口用不了、文件被锁住），key 继续以明文存放并打一行 `Warning:`。`finesub doctor` 的 `env-keys` 一行
  列出加了保护（protected）、仍是明文（plaintext）、本机解不开（unreadable）的各有几个。
- 自己写的脚本直接读 `.env`（比如 `load_dotenv()`）拿到的是密文：用 `finesub keys --out <文件>` 导出明文给它用；源码环境里的脚本
  也可以调 `finesub_bootstrap.secrets.read_env_file` 读。

## 关掉某家服务、挑用哪几把 key（`[providers]` 与 `[pools]`）

`.env` 保存带名字的 key（服务，即 provider；一家服务的一组 key 叫一个 pool）；`config.toml` 只按名字引用，不保存 key 本身：

```toml
[providers]
gemini_free = true
gemini_paid = true
exa = true
gemma4_grounded = true
tavily = true

[pools]
gemini_free = ["main", "spare"]
gemini_paid = []
exa = []
tavily = []
```

- `[providers]` 中缺失的项默认启用；`false` 表示运行时跳过这家服务。`gemma4_grounded`（用 Gemma 4 做 `local` 检索）可
  单独关闭，它用的是 `gemini_free` 那组 key。
- pool 缺失或为空时按 `.env` 声明顺序选择：Gemini Free 前 2 把、Exa/Tavily、MOSS Pro 与 ElevenLabs 前 3 把；Gemini
  Paid 默认全部启用且没有推荐上限。
- 显式 pool 可筛选、去重和重排 key。引用不存在的名字是配置错误。
- 显式 Gemini Free 超过 2 把或 Exa/Tavily/MOSS Pro/ElevenLabs 超过 3 把时不会截断，但会输出一次 `Warning:`,
  提示 key 多了可能触发服务方风控。
- 空 pool 表示使用上述默认选择；若要停用一家服务，必须在 `[providers]` 中设为 `false`。

## Gemini(Google AI Studio)

纠错翻译、准备任务、知识更新等生成调用都走 Gemini。免费档与付费档可同时配置：

| 变量 | 用途 |
| --- | --- |
| `GEMINI_FREE` | 免费档 key 池，`{显示名:密钥,...}` |
| `GEMINI_PAID` | 付费档 key 池，格式同上 |

**申请步骤(AI Studio):**

1. 打开 [Google AI Studio](https://aistudio.google.com/apikey) 并登录 Google 账号。
2. 创建 API key。
3. 将 key 写入 `.env`，例如：

```text
GEMINI_FREE={"main":"AIza...","spare":"AIza..."}
GEMINI_PAID={}
```

说明：

- 第一次跑 LLM 阶段会自动下载约 9 MB 的本地计数器 `tokcount`，之后数 token 不用联网；装不上不影响任务，改用免费的 `countTokens` 接口（不消耗生成额度），只是每次多一个网络往返。
  已经有自己那份的话设 `GEMINI_TOKEN_COUNTER_EXE`（或让 `tokcount` 在 PATH 上），就不会再下一份。两样都用不上时（例如 Google 连不上）改用本地估算并打一条 Warning；连不上时还会看到什么，见
  [`model-routing.md`「走代理：`[api] proxy`」](model-routing.md#走代理api-proxy)。
- `GEMINI_BASE_URL` 可改为直接连接其他 REST 端点（镜像/代理/本地 mock），例如
  `GEMINI_BASE_URL=https://your-mirror.example/v1beta`;不设置时使用 Google 官方地址。
  **该变量仅覆盖生成调用**：文件上传、`countTokens`、grounded 搜索仍直连 Google，因此它解决的是
  「生成请求不通」的问题，而非「完全不访问 Google 域名」。设了它而上传仍然失败时，告警末尾会提醒上传没走它。
- 主命令跑到 `--stage translated-srt` / `final-srt` 就真的调用模型、花额度。`finesub knowledge-update` 与 `finesub knowledge …`、`finesub knowledge-share …` 里要动用 AI 的子命令默认只预演（dry-run），加 `--execute` 才调用。
  从源码单独跑 LLM 层（`python -m finesub.llm.correction_translation`，安装版没有对应命令）也是默认只预演、加 `--execute` 才调用。
- 免费额度**按 Google Cloud 项目算，不按 key 算**（[Gemini 官方说明](https://ai.google.dev/gemini-api/docs/rate-limits)）：同一个项目里多建几把 key 不加额度，
  要多一份额度就在另一个项目里建 key。每个模型各有自己的每日次数，每天的额度从太平洋时间零点起算（北京时间夏令时 15:00、冬令时 16:00）。
- 一个项目的免费额度一天大约够纠 **3–4 小时**素材（默认 `--llm-difficulty intermediate`），`quality` 约 2 小时。这是估算：出厂纠错用免费的
  Gemini 4 Hydrogen、3.8 Flash 与 3.6 Flash，各 20 次请求一天；说话多的直播每小时素材约 4 个纠错窗口，每窗约 2 次调用，再加准备任务与重试；
  `quality` 每窗还多出说话人裁判与校对。估算**不含知识更新**：带 `--knowledge update` 时，任务后的知识更新也用这几个免费模型、占同样的次数，一天能纠的素材相应变少。
  `efficiency` 的纠错改用 Gemini 3.5 Flash Lite（每天 500 次），只有准备任务与知识更新还占前面几个模型的次数。
  没配付费 key 时用完就停下，不会偷偷换成更弱的模型，额度恢复后续跑即可（见[模型路由配置](model-routing.md#difficulty-是定位)）。
- 免费档有每分钟与每天的次数上限；一把 key 用到额度满才换下一把（所以池里的几把 key 要来自不同项目才有用）。付费 key 池不设上限，但可在 `config.toml` 中筛选和改变顺序。

## Exa（网页搜索）

搜索代理优先走 Exa deep search（未配置则静默跳过该 provider）:

| 变量 | 用途 |
| --- | --- |
| `EXA_KEYS` | `{显示名:密钥,...}` |

**申请步骤：**

1. 打开 [Exa Dashboard](https://dashboard.exa.ai/api-keys) 注册/登录。
2. 创建 API key。
3. 写入 `.env`，例如：

```text
EXA_KEYS={"exa1":"exa-..."}
```

说明：

- 注册即送约 3000 次搜索额度。
- 绑定支付方式后，每月额外赠送约 1000 次搜索额度。不主动充值不会扣费。

## Tavily（搜索回退）

Exa / Gemini grounded 之后的搜索回退；未配置则跳过：

```text
TAVILY_KEYS={"tvly1":"tvly-..."}
```

申请入口：[Tavily](https://tavily.com/) → API Keys。

说明：

- 无需设置付款方式，每月约 1000 次免费搜索。质量逊于 Exa。

## 额度与解冻

额度用完或被拒的 key 本机先搁到一边，到时间之前不再用：

- Gemini（纠错、翻译等）：一把 key 在某个模型上连续三次撞上每天的请求上限，记成这个模型当天用完，到太平洋时间零点；别的模型照用。
- 本机 agent（Codex、Claude Code、agy 等）：同一份订阅额度连续失败两次、再探测一次也失败，停用 2 小时；任何一次成功立即解除（见[本机 agent](agent.md)）。
- 联网搜索（Exa、Tavily、Gemma 4 检索）：key 被拒（鉴权失败、额度用完或限流，重试之后仍然如此）时停用 24 小时。
- ElevenLabs（Scribe）：查到余额不到 5 分钟时冻结到它的刷新时间（ElevenLabs 给的下次重置时间，查不到时 24 小时）。
- 35t：撞上当天的请求上限，记成当天用完，到太平洋时间零点。
- MOSS Pro：前面的 key 被拒（HTTP 402）时记成当天用完，到北京时间零点。本机还记着每把 key 当天转了多少音频，用来估免费额度还剩多少。

充了值、换了档、额度已经恢复，或者确认停用是误判时，可以提前放回：

```powershell
finesub quota                                # 列出被搁置的 key 与今天的用量
finesub quota thaw elevenlabs/scribe_v3/main  # 放回一把：名字照列出的写
finesub quota thaw agent                      # 也可以只写前缀：elevenlabs、moss-pro、GEMINI_FREE、agent、search
finesub quota thaw --all                     # 全部放回
```

放回在别的正在运行的 FineSub 里也立即生效。放回只清「用完」的记号；MOSS Pro 当天已经记下的用量不清，本机仍按它估免费额度还剩多少。
真没有额度的 key 下次被拒或查余额时会再被搁置。

## 云端转写（`--asr`）

转写交给云端时（[`asr-backend-choice.md`](asr-backend-choice.md)）用的 key，和上面几项一样写在 `.env`（本机显卡上的 `moss-local` 不用 key）：

| 后端 | 变量 | 说明 |
| --- | --- | --- |
| `gemini-35t` | `GEMINI_FREE` | 与 LLM 层用同一组 key，但 35t 是另一个模型，次数与纠错用的模型各算各的、互不挤占；每个项目每天的请求数有限，每天的额度从太平洋时间零点起算。一把用到撞上日上限才换下一把 |
| `moss-pro` | `MOSS_PRO` | `{显示名:密钥,...}`，MOSI 开放平台（控制台 `platform.moss.cn`，API 在 `api.mosi.cn`）的 key。免费额度、超出后的扣费与 HTTP 402 怎么办，见[转写后端「额度与费用」](asr-backend-choice.md#额度与费用) |
| `mai2` | `FINESUB_KEY_VERCEL_STT`，或 `FINESUB_KEY_OPENROUTER_STT`，或 `FINESUB_AZURE_SPEECH_ENDPOINT` + `FINESUB_KEY_AZURE_SPEECH` | 三条路任选：Vercel AI Gateway 的 key、OpenRouter 的 key，或你自己的 Azure Speech 资源的地址（如 `https://<名字>.cognitiveservices.azure.com`）与 key；配了几条时按这个顺序用第一条。按音频时长计费（价格见[转写后端](asr-backend-choice.md)），开工前在 stderr 打一行估计费用 |
| `scribe` | `ELEVENLABS` | ElevenLabs 的 key，权限开 Speech to Text 与 User: Read（查余额）。可以写成 `{显示名:密钥,...}` 配最多 3 把，从前往后用，最后一把当付费的那把；怎么挑 key、什么时候换，见[转写后端「额度与费用」](asr-backend-choice.md#额度与费用) |

每家写进 `.env` 的那一行示例在[转写后端「拿到 key」](asr-backend-choice.md#拿到-key)。MOSS Pro 配几把时写成
`MOSS_PRO={"main":"sk-...","spare":"sk-..."}`；`config.toml` 的 `[pools] moss_pro = [...]`（Scribe 是 `[pools] elevenlabs = [...]`）可以筛选、重排这几把，规则同上面的 pool。

## 最小可用组合

| 目标 | 最少需要 |
| --- | --- |
| 只跑 ASR → raw SRT | 无需 API key |
| 云端转写 → raw SRT | 所选后端的 key（见[云端转写（`--asr`）](#云端转写--asr)） |
| 从源码单独跑 LLM 层的 dry-run（本地 tokenizer 在位） | 可不配 key |
| LLM 纠错翻译（`--stage translated-srt` / `final-srt`） | 至少一把 `GEMINI_FREE` 或 `GEMINI_PAID`，或配好本机 agent（见[本机 Agent 后端](agent.md)） |
| 联网查资料（`--llm-retrieval local`，默认） | 有 `GEMINI_FREE` 就能搜：按 Exa → Gemma 4（用 Gemini 免费 key）→ Tavily 的顺序，配了哪家用哪家。推荐另配 `EXA_KEYS`，检索质量最好。只配了付费 key 时没有 Gemma 4 这一家，`execution_policy = "agent-only"` 时也不用它；三家都用不了时不检索 |

额度用完、限流时任务会怎样，见[模型路由配置](model-routing.md)「difficulty 是定位」。
