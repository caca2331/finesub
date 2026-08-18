# 自定义 provider 与文本 adapter

> **要接一个自己的 endpoint，先去** [`manual/model-routing.md` §5「接自己的模型」](manual/model-routing.md#5-接自己的模型)：
> 该写哪几列、key 放哪、怎么绑格子、第一版的三条硬限制，都在那儿。**本文不讲怎么配**——
> 它是实现契约与供应商行为差异的调研台账，写给要改 adapter、或要判断「这家为什么这么回」
> 的人。两份是同一件事的两半，分界是读者不是主题。

状态：已实现第一刀（纯文本）。实现在 `src/finesub/llm/provider_transports.py`（两个薄传输）+
`llm_runtime.chat_complete` 的按 provider 分派；用户声明入口是数据根的 `model_catalog.psv`
（事实）+ `config.toml` 的 `[llm.model_groups]` / `[llm.presets]`（编排）——样例见
`config.example.toml`，有一致性测试防漂移。
不用 litellm（D19：会丢 `thinkingLevel`、结构化 `quotaId`、cached-token usage 三处精度）。

## 供应商行为调研（2026-08-11，Sonnet 调研代理产出，已按实现取舍裁剪）

关键事实（★ = 已在实现中消化）：

| 主题 | OpenAI 官方 | DeepSeek | Anthropic |
| --- | --- | --- | --- |
| ★ 模型名 | — | **`deepseek-chat`/`deepseek-reasoner` 已于 2026-07-24 停用**，现为 `deepseek-v4-flash`/`-pro` + thinking 开关 | — |
| ★ reasoning 模型 × 采样参数 | 传 temperature 等 → **400** | 静默忽略 | 新模型（Opus 5/Sonnet 5 等）整体移除采样参数，传了 400 |
| ★ 思考深度 | `reasoning_effort`（none…max，按模型） | `thinking:{type}` 或 `reasoning:{effort}` 两种方言 | `thinking:{type:"enabled",budget_tokens}`（另有 `output_config.effort`） |
| ★ 截断 | `finish_reason="length"` | 同左；另有特有值 `insufficient_system_resource`（资源不足中断，宜重试） | `stop_reason="max_tokens"` |
| ★ 拒答 | `content_filter` | `content_filter` | **`refusal`**（HTTP 200 + stop_reason，可能已部分输出） |
| ★ 限流/余额 | 429；余额尽也是 **429 `insufficient_quota`** | 429；余额尽是 **402** | 429 `rate_limit_error`；过载 **529 `overloaded_error`** |
| ★ cached usage | `prompt_tokens_details.cached_tokens` | 扁平 `prompt_cache_hit_tokens`/`prompt_cache_miss_tokens` | 顶层 `cache_read_input_tokens`/`cache_creation_input_tokens` |
| ★ 思考输出 | 不返回内容，只计数（`completion_tokens_details.reasoning_tokens`） | `message.reasoning_content`（vLLM 已改名 `reasoning`） | `thinking` content block；token 并入 output 不单列 |
| ★ 原生 seed | Chat Completions 有（Responses API 无；确定性弱） | 仅第三方提及，未一手确认 | 无 |
| ★ Messages API | — | — | `x-api-key` + `anthropic-version`；`system` 是顶层参数；`max_tokens` 必填；同角色连续消息服务端自动合并，首条须 user |
| 方言差异 | — | 与 OpenAI 一致（Bearer / `/chat/completions` / `max_tokens`） | — |
| vLLM/Ollama | 未知参数**可能 400 而非忽略**（vLLM 有社区实例）；Ollama 本地默认无认证但要求占位 key | | |

未确认项（用前实测）：DeepSeek 384K max output 的准确性、DeepSeek 错误体字段名、
`reasoning_tokens` 计费口径、OpenAI `xhigh`/`max` 按型号可用性。

## 实现契约（对应 §10.1 七项）

- **归一化目标**：两个传输都输出 OpenAI-ish 形状（`choices[0].message.content` +
  `finish_reason` + 嵌套 `usage` details）——harness 下游（`extract_token_distribution` /
  `_response_finish_reason` / `is_prompt_blocked`）本就双兼容 Gemini 与该形状。
  Anthropic `end_turn→stop`、`max_tokens→length`，`refusal` 原样透传并被
  `is_prompt_blocked` 识别；DeepSeek 扁平 cache/reasoning 字段折进嵌套 details；
  `reasoning_content`/`reasoning`（原始思维链）一律剥除，绝不当可见输出。
- **D18**：非 Gemini 一律**不传采样参数**；重掷扰动是 prompt 末尾 `(seed=N)` 文本
  （provider 无关，`chat_complete` 在分派前统一追加）。
- **thinking**：抽象档位来自预设级思考旋钮（`[presets.<id>.thinking]`，难度回落、缺省
  medium）；模型在 catalog/`[llm.models]` 的 `thinking` 列声明自己的映射。**缺省是恒等**
  （`true`；2026-08-12 起也是 `[llm.models]` 不写该字段时的默认）——三家的档位词本来就同名：
  Gemini `thinkingLevel`、OpenAI 系 `reasoning_effort`、Anthropic `output_config.effort`
  都取 low/medium/high（前两者另有 xhigh/minimal 等扩展值）。端点用别的词才写
  `high映射,med映射,low映射` 三个取值覆盖；`false` = 不发思考参数（对未知字段会 400 的
  自建端点的声明式出口）。client 逐候选换算，产物记录映射后实际发出的值。
- **失败分类**：非 2xx 抛 `RuntimeError("HTTP <code> <body>")` 走现有关键词分类；
  新增两条映射——**402 → quota**（DeepSeek 余额尽）与 **429 `insufficient_quota` → quota**
  （OpenAI 账户额度尽），组内直接推进而不是原地重试。**日封禁不适用**：strike 依赖
  Gemini 结构化 `quotaId`，其他供应商的 429 归 rate_limit，撞满就退避重试。
- **usage**：报告与校准以 API 返回值为准（缺失记 0 交由现有口径处理）。
- **媒体**：第一刀不做（`supports_audio/video=false` 固定）；纯文本强模型用在音视频素材
  上的路径是 `--correction-media text`。
- **key**：`FINESUB_KEY_<PROVIDER_TIER>`（可用 catalog 的 `key_env` 列覆盖），走 `secrets.py`
  同一条加密 `.env` 路径，不另开明文存放点。
- **声明位置（2026-08-12 起）**：provider 方言与 base_url 是**事实**，写在数据根的
  `model_catalog.psv` 行上（`provider_kind` / `base_url` / `key_env` 三列），一行即声明
  provider + fact + target；`config.toml` 只留模型组与预设。旧的 `[llm.providers]` /
  `[llm.models]` 会报错并指向新位置。
