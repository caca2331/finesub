# 环境变量与 API Key

LLM harness 与网页搜索默认从源码 checkout 根目录下的 `.env` 读取密钥，从 `config.toml` 读取
provider 开关与 key pool；也可通过 `FINESUB_ENV_FILE` / `FINESUB_CONFIG_FILE` 分别指定其他文件。
process environment 中的 key 变量会覆盖 `.env`。**请勿**将 `.env` 或本机 `config.toml` 提交到 git。
checkout 根目录可通过 `FINESUB_ROOT` 显式指定。`config.toml` 的创建方法与内容见
[`resources.md`](resources.md)「设置文件 `config.toml`」。

非 checkout 用户无需关注上述路径：**`finesub` CLI 读的是
`%LOCALAPPDATA%\FineSub\user-data\.env`**，用 `finesub keys` 写入，或直接手动编辑该
`.env`（格式见下文）。个人数据以外的内容（模型、缓存、任务产物）默认存放在安装目录下，可以
迁移，见 [`resources.md`](resources.md)。

## 密钥保护（绑定 Windows 账户）

Windows 上 `.env` 里的密钥不以明文存放：首次运行（CLI 的启动迁移，或源码 checkout 的
首次读取）会把每个密钥值原地替换为 `fs$…` 密文，并在文件顶部写入一行 `FINESUB_KEYRING`,
即经 DPAPI 绑定当前 Windows 账户的主密钥。变量名、命名 key 的显示名、注释与格式逐字节保留，
`cat .env` 仍能看清有哪些 key、与 `config.toml` 的 `[pools]` 对照。

- 该机制**将密钥绑定到当前 Windows 账户，并对明文进行混淆**，用于防止文件误传、通用扫盘及
  同机其他账户的读取；**无法防御以你的身份运行的恶意程序**（程序必须以无口令方式解密，密钥材料
  必然位于其可访问范围内）。
- 请勿手动修改或删除 `FINESUB_KEYRING` 行；一旦删除，所有密文将永久无法恢复。
- **在更换设备、重装 Windows 或更换 Windows 账户之前，请先导出明文**：执行 `finesub keys --reveal`。
  输出为 `NAME=值` 格式，可直接粘贴到新机器的 `.env`。
- 将 `.env` 复制到其他机器后，密钥会显示为「未配置」并出现警告，**文件本身不会被修改**；拿回原
  机器后一切照常。如需在新机器上使用：重新填写（或删除）**全部**无法解密的值后，加密保护会自动
  以新机器的账户重新建立。
- 手动向 `.env` 写入明文 key 仍然可行，下次运行时会自动加密。唯一限制：明文值中不能出现以
  `fs$` 开头的类 token 子串（该子串会被当作损坏的密文拒绝）。
- 在非 Windows 平台或 DPAPI 不可用时，自动回退为明文存储并输出 `Warning:`。
- **过渡开关**：设置 `FINESUB_ENV_PROTECT=0` 可暂停自动加密（静默生效；既有密文仍可正常解密，
  迁移保持未完成状态，变量移除后的下一次启动会自动补做转换）。适用于仍有旧代码直接读取 `.env`
  的过渡期（例如尚未收敛的 worktree，旧解析器会把密文当作无效 key）；收敛完成后请移除该变量，
  避免长期明文存储。
- `finesub keys` 默认掩码显示；`finesub doctor` 的 `env-keys` 一行给出 protected / plaintext /
  unreadable 计数。
- 若自写脚本直接通过 `load_dotenv()` 读取该文件，获取到的是密文；请改用 `finesub keys --reveal`
  导出，或在读取时使用 `finesub_bootstrap.secrets.read_env_file`。

## Provider 与 pool 配置

`.env` 只保存命名密钥；`config.toml` 只引用显示名，不保存 secret:

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

- `[providers]` 中缺失的项默认启用；`false` 表示运行时跳过该 provider。`gemma4_grounded` 可
  单独关闭，但仍复用 `gemini_free` pool 的 key。
- pool 缺失或为空时按 `.env` 声明顺序选择：Gemini Free 前 2 把、Exa/Tavily 前 3 把；Gemini
  Paid 默认全部启用且没有推荐上限。
- 显式 pool 可筛选、去重和重排 key。引用不存在的名字是配置错误。
- 显式 Gemini Free 超过 2 把或 Exa/Tavily 超过 3 把时不会截断，但会输出一次 `Warning:`,
  提示较大的 pool 可能触发 provider 风控。
- 空 pool 表示使用上述默认选择；若要禁用 provider，必须在 `[providers]` 中设为 `false`。

## Gemini(Google AI Studio)

纠错翻译、背景调查、知识更新等生成调用都走 Gemini。免费档与付费档可同时配置：

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

- `countTokens` 只用于鉴权，不消耗生成配额；本地有 `bin/.../tokcount.exe` 时可完全离线数 token。
- `GEMINI_BASE_URL` 可改为直接连接其他 REST 端点（镜像/代理/本地 mock），例如
  `GEMINI_BASE_URL=https://your-mirror.example/v1beta`;不设置时使用 Google 官方地址。
  **该变量仅覆盖生成调用**：文件上传、`countTokens`、grounded 搜索仍直连 Google，因此它解决的是
  「生成请求不通」的问题，而非「完全不访问 Google 域名」。
- 真实生成需 CLI 显式加 `--execute`（默认 dry-run）。
- 单账号每天的免费配额约能做 1-2 小时的高质量翻译，超过后质量会下降。
- 免费档有 RPM/日限额；生成调用按选定 pool 顺序执行 sticky retry 和 quota failover。Gemini
  Paid pool 不设上限，但可在 `config.toml` 中筛选和改变顺序。

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

## 最小可用组合

| 目标 | 最少需要 |
| --- | --- |
| 只跑 ASR → raw SRT | 无需 API key |
| LLM dry-run（本地 tokenizer 在位） | 可不配 key |
| LLM `--execute` | 至少一把 `GEMINI_FREE` 或 `GEMINI_PAID` |
| 启用网页搜索增强 | 另配 `EXA_KEYS` 和/或 `TAVILY_KEYS`（自 0.5.0 起不再提供免 key 后备方案；一个 key 都没有则不执行检索） |

字段语义与限流行为见 [`docs/llm_harness_behavior.md`](../llm_harness_behavior.md)。
