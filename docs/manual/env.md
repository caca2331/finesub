# 环境变量与 API Key

LLM harness 与网页搜索默认从源码 checkout 根目录的 `.env` 读取密钥，从
`config.toml` 读取 provider 开关和 key pool。也可以分别用
`FINESUB_ENV_FILE` / `FINESUB_CONFIG_FILE` 指向其他文件；process environment
中的 key 变量覆盖 `.env`。**不要**把 `.env` 或本机 `config.toml` 提交进 git。
checkout 根目录可由 `FINESUB_ROOT` 显式指定。配置模板见
[`config.example.toml`](../../config.example.toml)。

非 checkout 用户不用管上面这些路径：**FineSub Desktop（安装器版）在设置页填的
API Key 存进 `%LOCALAPPDATA%\FineSub\user-data\.env`，`finesub` CLI 会自动读同
一个文件**——任一端配置一次，两端都能用。portable 版桌面端的 `.env` 在其安装目
录的 `user-data\` 下。CLI 用户也可直接手动编辑该 `.env`（格式见下文）。

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
