# 环境变量与 API Key

LLM harness 与网页搜索从项目根目录的 `.env` 读取密钥。复制 `.env-sample` 为 `.env` 后按需填写；**不要**把 `.env` 提交进 git。

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
- 免费档有 RPM/日限额；多把 key 可写成 dict，运行时按 harness 的 pool 规则轮换。但不建议同时超过两个账号，以免被风控。

## Exa（网页搜索）

搜索代理优先走 Exa deep search（未配置则静默跳过该 provider）：

| 变量 | 用途 |
| --- | --- |
| `EXA_KEYS` | `{显示名:密钥,...}` |
| `EXA_POOL` | 可选；从 `EXA_KEYS` 里挑参与 pool 的名字，默认取前 3 个 |

**申请步骤：**

1. 打开 [Exa Dashboard](https://dashboard.exa.ai/api-keys) 注册/登录。
2. 创建 API key。
3. 写入 `.env`，例如：

```text
EXA_KEYS={"exa1":"exa-..."}
# 可选：EXA_POOL={exa1}
```

说明：
- 注册即送约3000次搜索额度。
- 绑定支付方式后，每月额外赠送约1000次搜索额度。不主动充值不会扣费。

## Tavily（搜索回退）

Exa / Gemini grounded 之后的搜索回退；未配置则跳过：

```text
TAVILY_KEYS={"tvly1":"tvly-..."}
TAVILY_POOL={tvly1}
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
