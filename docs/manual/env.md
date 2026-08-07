# 环境变量与 API Key

LLM harness 与网页搜索默认从源码 checkout 根目录的 `.env` 读取密钥，从
`config.toml` 读取 provider 开关和 key pool。也可以分别用
`FINESUB_ENV_FILE` / `FINESUB_CONFIG_FILE` 指向其他文件；process environment
中的 key 变量覆盖 `.env`。**不要**把 `.env` 或本机 `config.toml` 提交进 git。
checkout 根目录可由 `FINESUB_ROOT` 显式指定。配置模板见
[`config.example.toml`](../../config.example.toml)。

非 checkout 用户不用管上面这些路径：**桌面端在设置页填的 API Key 存进
`%LOCALAPPDATA%\FineSub\user-data\.env`，`finesub` CLI 会自动读同一个文件**——
安装版、便携版、CLI 三种形式共用这一份，任一端配置一次，处处可用。CLI 用户也可
直接手动编辑该 `.env`（格式见下文）。个人数据以外的东西（模型、缓存、任务产物）
默认跟着安装目录走，可以搬，见 [`resources.md`](resources.md)。

## 密钥保护（绑定 Windows 账户）

Windows 上 `.env` 里的密钥不以明文存放：首次运行（桌面端/CLI 的启动迁移，或源码
checkout 的首次读取）会把每个密钥值原地替换为 `fs$…` 密文，并在文件顶部写入一行
`FINESUB_KEYRING`——主密钥，经 DPAPI 绑定当前 Windows 账户。变量名、命名 key 的
显示名、注释与格式逐字节保留，`cat .env` 仍能看清有哪些 key、与 `config.toml` 的
`[pools]` 对照。

- 这是**绑定当前 Windows 账户的保护 + 防泛扫混淆**，防的是误传文件、通用扫盘、
  同机其他账户；**不能防御以你的身份运行的恶意程序**（程序必须能无口令自解密，
  密钥材料必然在其可达范围内）。
- `FINESUB_KEYRING` 行不要手改或删除——删了它，所有密文将永久无法恢复。
- **换机、重装 Windows、换 Windows 账户之前，先导出明文**：`finesub keys --reveal`
  （桌面端：设置 → 显示已保存的密钥）。输出是 `NAME=值` 形式，可直接粘回新机器的
  `.env`。
- 把这份 `.env` 拿到别的机器上时，密钥表现为「未配置」并有警告，**文件不会被改
  动**，拿回原机器一切照旧。确要在新机器上用：重填（或删除）**全部**不可解密的
  值之后，保护会自动以新机器的账户重建。
- 手动往 `.env` 里写明文 key 仍然可以：下次运行会被自动加密。唯一限制是明文值里
  不能出现 `fs$` 开头的 token 形态子串（会被当作损坏的密文拒绝）。
- 非 Windows 或 DPAPI 不可用时自动退回明文并打 `Warning:`。
- **过渡开关**：设 `FINESUB_ENV_PROTECT=0` 可暂停自动加密（静默；已有密文仍正常
  解密，迁移保持未完成、变量移除后的下一次启动自动补上转换）。用途是还有旧代码
  直接读 `.env` 的过渡期（如未收敛的 worktree——旧解析器会把密文当垃圾 key）；
  收敛后请移除该变量，不要长期明文。
- `finesub keys` 默认掩码显示；`finesub doctor` 的 `env-keys` 一行给出
  protected / plaintext / unreadable 计数。
- 自写脚本若直接 `load_dotenv()` 读这份文件，拿到的是密文；请改走
  `finesub keys --reveal` 导出，或读取时用 `finesub_bootstrap.secrets.read_env_file`。

## Provider 与 pool 配置

`.env` 只保存命名密钥；`config.toml` 只引用显示名，不保存 secret：

```toml
[providers]
gemini_free = true
gemini_paid = true
exa = true
gemma4_grounded = true
tavily = true
duckduckgo = true

[pools]
gemini_free = ["main", "spare"]
gemini_paid = []
exa = []
tavily = []
```

- `[providers]` 中缺失的项默认启用；`false` 表示运行时跳过该 provider。
  `gemma4_grounded` 可单独关闭，但仍复用 `gemini_free` pool 的 key。
- pool 缺失或为空时按 `.env` 声明顺序选择：Gemini Free 前 2 把、Exa/Tavily
  前 3 把；Gemini Paid 默认全部启用且没有推荐上限。
- 显式 pool 可筛选、去重和重排 key。引用不存在的名字是配置错误。
- 显式 Gemini Free 超过 2 把或 Exa/Tavily 超过 3 把时不会截断，但会输出一次
  `Warning:`，提示较大的 pool 可能触发 provider 风控。
- 空 pool 表示使用上述默认选择；若要禁用 provider，必须在 `[providers]` 中设为
  `false`。

## Gemini（Google AI Studio）

纠错翻译、背景调查、知识更新等生成调用都走 Gemini。免费档与付费档可同时配置：

| 变量 | 用途 |
| --- | --- |
| `GEMINI_FREE` | 免费档 key 池，`{显示名:密钥,...}` |
| `GEMINI_PAID` | 付费档 key 池，格式同上 |

**申请步骤（AI Studio）：**

1. 打开 [Google AI Studio](https://aistudio.google.com/apikey) 并登录 Google 账号。
2. 创建 API key。
3. 将 key 写入 `.env`，例如：

```text
GEMINI_FREE={"main":"AIza...","spare":"AIza..."}
GEMINI_PAID={}
```

说明：

- `countTokens` 只用于鉴权，不消耗生成配额；本地有 `bin/.../tokcount.exe` 时可完全离线数 token。
- 真实生成需 CLI 显式加 `--execute`（默认 dry-run）。
- 单账号每天的免费配额约能做1-2小时的高质量翻译。超过后质量会下降。
- 免费档有 RPM/日限额；生成调用按选定 pool 顺序执行 sticky retry 和 quota
  failover。Gemini Paid pool 不设上限，但可在 `config.toml` 中筛选和改变顺序。

## Exa（网页搜索）

搜索代理优先走 Exa deep search（未配置则静默跳过该 provider）：

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
- 注册即送约3000次搜索额度。
- 绑定支付方式后，每月额外赠送约1000次搜索额度。不主动充值不会扣费。

## Tavily（搜索回退）

Exa / Gemini grounded 之后的搜索回退；未配置则跳过：

```text
TAVILY_KEYS={"tvly1":"tvly-..."}
```

申请入口：[Tavily](https://tavily.com/) → API Keys。

说明：
- 无需设置付款方式，每月约1000次免费搜索。质量逊于exa。

## 最小可用组合

| 目标 | 最少需要 |
| --- | --- |
| 只跑 ASR → raw SRT | 无需 API key |
| LLM dry-run（本地 tokenizer 在位） | 可不配 key |
| LLM `--execute` | 至少一把 `GEMINI_FREE` 或 `GEMINI_PAID` |
| 启用网页搜索增强 | 另配 `EXA_KEYS` 和/或 `TAVILY_KEYS`（DuckDuckGo 免 key 兜底仍可用） |

字段语义与限流行为见 [`docs/llm_harness_behavior.md`](../llm_harness_behavior.md)。
