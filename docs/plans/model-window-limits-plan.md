# 窗口限额三档化：catalog 记厂商值，planner 只做算术

**状态：已实施（2026-09-03）**。方案经 owner 与 reviewer 审计后按 §8 七步落地；下面保留的是
取舍依据与验收口径，**现行行为以代码与 `manual/model-routing.md` 的列说明为准**。
两处 owner 裁定与实施中发现的一处方案错误记在 §9。

> **本文的两个代称**：`HARNESS_INPUT_CAP` 与 `HARNESS_OUTPUT_CAP` **不是代码里的标识符**，
> grep 不到。它们指的是 `routing/config.py:455-457` 里把 `DEFAULT_LIMITS.prompt_input_limit`
> （194000）和 `DEFAULT_LIMITS.output_limit`（65536）**当作上限用**的那两处 `min(...)`——同一个
> 常量在别处是「取不到路由声明时的兜底值」，本方案只删它作为上限的那一层。

要解决的是一件事：`model_catalog.psv` 只有 `max_input_tokens` / `max_output_tokens` 两列，
而代码需要三个量（上下文池、输入上限、输出上限），于是第一列被迫身兼两职，且真正决定窗口
大小的是三个写死在 harness 里的常量而不是 catalog。本方案把三个量分开记在 catalog，删掉那三个
常量，让 planner 只剩算术。

---

## 1. 现状：谁读哪个数

`derive`（`routing/config.py:443-463`）：

```python
min_in, min_out = 组内 min(max_input_tokens), min(max_output_tokens)
scale           = 组内 max(token_scale, 1.0)

output_limit = min(65536, min_out)
if min_in >= 194000 and output_limit >= 65536 and scale == 1.0 and not video_hi:
    return DEFAULT_LIMITS                                    # ← 提前返回
prompt_input_limit = min(194000, (min_in - output_limit - 1000) / scale)
context_limit      = min(256000, min_in / scale)
```

下游只读 `ModelLimits`，公式本身本方案不动：

| 位置 | 公式 | 读 |
| --- | --- | --- |
| `chunking.py:577` | `input_capacity = prompt_input_limit − 研究上下文 − prompt − 重叠文本/媒体 − 媒体填充` | input |
| `chunking.py:568` | `coefficient_cap = output_limit / (output_scale × output_coefficient)` | output |
| `profiles.py:335` | `window_output_budget = 0.9 × output_limit − 5000` | output |
| `plan.py:290` | `input_budget = prompt_input_limit − 56000`（fast 第二轮预留） | input |
| `token_budget.py:673/683` | `input > prompt_input_limit` ／ `input+输出+margin > context_limit` | 三档 |
| `correction/run.py:634-638` | resume 重拟合，同上三条，触发**对半切** | 三档 |

绕过 `ModelLimits` 直接读 catalog 的三处：

| 位置 | 公式 | 把 `max_input_tokens` 当 |
| --- | --- | --- |
| `client.py:2550` | `max_tokens > max_output_tokens` → skip `output_limit` | 输出上限 ✓ |
| `client.py:2560/2576` | `estimated_input > max_input_tokens` → 先丢 repair context，再 skip `input_limit` | **输入上限** |
| `capabilities.py:171` · `model_routes.py:544/628` | 组内取 min，报「谁是限制成员」 | 输入上限 |

### 1.1 四个具体毛病

1. **一列两职**：`derive` 当它是上下文池（减掉输出才是输入包络），`client.py` 当它是输入上限。
   两个含义要求的数值不同。
2. **提前返回把 catalog 架空**：`min_in ≥ 194000 且 min_out ≥ 65536` 即返回默认档。所以 codex
   填 194000 还是 272000 结果完全一样——catalog 的值没有进过算式。
3. **`safety_margin` 出现在两侧**：`derive` 从上限减 1000，`token_budget` 又把 1000 加到被测总量
   上。进的是不同比较，不构成重复扣减，但读的人得先证明这一点。
4. **三个默认值在角上自相矛盾**：`194000 + 65536 + 1000 = 260536 > 256000`。今天摸不到，只因预期
   输出从不接近 65536。

---

## 2. 新语义

catalog 记三个**厂商值**，一条不变式：`context_window ≥ max(max_input_tokens, max_output_tokens)`。

| 列 | 含义 | 留空时 |
| --- | --- | --- |
| `context_window`（新） | input 与 output 共用的池子 | `= max_input_tokens + max_output_tokens`，即**联合约束不存在** |
| `max_input_tokens` | API 接受的最大输入 | 必填（不变） |
| `max_output_tokens` | API 接受的最大 `max_tokens` | 不变 |

留空的默认值就是「这一行不声明联合约束」；单池供应商（Codex、Anthropic）必须显式填。
⚠ Gemini 分别公布输入与输出两个上限，所以留空是它的诚实读法——但**只有免费档真的留空**：
付费档与 agy 按 owner 2026-09-03 的决定仍然填了 `ctx = max_input = 1048576`（§4.1），
宁可按单池规划，也不赌两个公布值能同时花掉。加载时校验不变式，违反了报行号——那是声明错误，不该有
默认值兜底。

**`context_window` 的唯一作用是动态压低输入上限**（owner 2026-09-03）。它不参与别的判断。

---

## 3. 新公式

```python
output_limit = min(f.max_output_tokens for f in facts)
prompt_input_limit = min(
    min(f.max_input_tokens, f.context_window - output_limit) for f in facts
) / scale
```

⚠ **两遍，不是逐列取 min**。先定 `output_limit`，再用它算每个成员的输入天花板然后取 min。
逐列取 min 会在异质组里高估：成员 A `ctx=200000/out=64000`（真实天花板 136000）、成员 B
`ctx=210000/out=128000`（真实 82000），逐列取 min 得 `200000−64000=136000`，比 B 的真实值高
54000——那正是规划包络必须避免的方向。

### 3.1 三个删除

| 删什么 | 依据 |
| --- | --- |
| `min(DEFAULT_LIMITS.output_limit, …)`（代称 `HARNESS_OUTPUT_CAP`） | `output_limit` 直接取 `min_out` |
| `min(DEFAULT_LIMITS.prompt_input_limit, …)`（代称 `HARNESS_INPUT_CAP`） | 上式已经给出包络 |
| `context_limit` 字段及其两处检查 | **按构造冗余**：`input ≤ ctx − output_limit` 且 `expected_output ≤ output_limit` ⇒ `input + expected_output ≤ ctx`，那道检查永远不会响 |
| `safety_margin` 字段 | 删掉 `context_limit` 后它没有消费者。它想防的估算偏差已由 `window_output_budget = 0.9 × output_limit − 5000` 兜着（10% + 5000 的双重松弛），比一个 1000 的常量厚 |

⚠ **上表最后两行是一个整体，不能只做一半。** 「`context_limit` 按构造冗余」成立的前提是
`safety_margin` 同时被删：今天被比较的量是 `total_with_margin = input + expected_output +
1000`，输入输出都顶满时它正好**超出 ctx 一个 margin**，检查会响。只删 `context_limit` 会丢掉
一道真的能响的检查；只删 `safety_margin` 则让那道检查退化成恒不响的死代码。

连带：`CorrectionBudget.total_with_margin` 失去唯一消费者。它**写在产物里**
（`correction/metadata.py:70`、`agent_validators.py:47/77`），按「不做向后兼容」直接删字段，
两处读者同步改，旧产物 resume 失效重跑。

`DEFAULT_LIMITS` 里的三个数不消失，但**降格为「取不到路由声明时的兜底值」**（stub config、
测试、`prompt_artifacts` 的元数据），需要改名以免再被当成 cap 读。

### 3.2 `client.py` 派发闸门

```python
if max_tokens > entry.max_output_tokens:                 skip "output_limit"    # 不变
if estimated_input > entry.max_input_tokens:             skip "input_limit"     # 不变
if estimated_input + max_tokens > entry.context_window:  skip "context_limit"   # 新增
```

新增那条今天不会响（最坏 `194000 + 65536 = 259536`，codex 272000 ✓，免费 Gemini 那档留空、
池子恰好也是 259536 ✓），它补的是语义完整性：单池模型的联合约束此前只在规划期间接管着，派发侧没有。

---

## 4. catalog 逐行改动

`context_window` 一列，只有单池供应商填：

**今天每一行的有效输入上限都是 194000**（`HARNESS_INPUT_CAP`），无一例外——所以下表最后一列比的
是「删掉那道上限之后」。

| 行 | ctx | input | output | 有效输入上限 | 与今天（194000）比 |
| --- | ---: | ---: | ---: | ---: | --- |
| codex luna / terra / sol | 272000 | 272000 | 65536 | 206464 | **+12464** |
| claude opus-5 / sonnet-5 | 1000000 | 1000000 | 128000 | 872000 | **+678000** |
| **claude haiku-4.5** | 200000 | 200000 | 64000 | **136000** | **−58000** |
| dsh v4-flash / v4-pro | 1000000 | 1000000 | 256000 | 744000 | **+550000** |
| gemini-paid 3.8 / 3.7 | 1048576 | 1048576 | 65536 | 983040 | **+789040** |
| agy gemini-3.8-flash | 1048576 | 1048576 | 65536 | 983040 | **+789040** |
| gemini-free ×7 | 留空 | 194000 | 65536 | 194000 | 不变 |
| gemini-paid-3_5-flash-lite | 1048576 | 1048576 | 65536 | 983040 | **+789040** |
| agy opus-4.6 · conversational | 留空 | 194000 | 65536 | 194000 | 不变 |
| gemini-free-gemma-4-31b | 留空 | 16000 | 32768 | 16000 | 见 §4.2 |

haiku 变小是 owner 2026-09-03 的决定：200K 是 Anthropic 真实的共享池，今天的 194000 只是恰好等于
harness 上限，并不代表它能同时装下 194000 输入和 64000 输出。

### 4.1 ⚠ free 与满血 Gemini 在这次改动后会真正分家

它们今天**看起来一样**（都是 194000），但那是 `HARNESS_INPUT_CAP` 把 paid 的值压下来的假象。
删掉上限后 paid 与 agy 跳到 983040，free 留在 194000——**5 倍差距第一次真实生效**。

**owner 决定（2026-09-03）**：

- **付费 Gemini（API 三行与 agy）填 `ctx = input = 1048576`**（flash-lite 那行 2026-09-03 补裁，
  理由同上：付费档 `tpm=4000000`，194000 在那里没有 TPM 依据），与 claude / dsh 同样按单池处理，
  于是有效输入是 `1048576 − 65536 = 983040`。
- **free 留空**，`194000` 不动。owner 确认这个数**当初就是按 TPM 拍的**，并且裁定
  **不为它额外改代码**——「这是个对真实行为的高效模拟」：free 档 `tpm=250000`，一次 194000 的
  请求已占 78%，规划到更大的窗只会在供应商侧换来 429。

所以这一列在 free 那行装的确实不是上下文窗口，而是一个速率导出值；**这是有意保留的近似，不是
待办**。相关事实（限流器 `rate_limit.py:565` 的 `if projected > tpm and active_events` ——窗口为空
时超额单请求被直接放行，我们不拦、拦的是供应商）记在这里备查，不构成改动理由。

### 4.2 新公式顺带拆掉一个哑弹

`gemini-free-gemma-4-31b` 是 `16000` 输入 / `32768` 输出。今天的公式：

```
output_limit = min(65536, 32768) = 32768
prompt_input_limit = max(1, 16000 - 32768 - 1000) = 1        ← 一个 token
```

任何含它的模型组，纠错包络会塌成 1。今天摸不到，只因为它在 `model_routes.toml` 里**只有 target
定义、不属于任何 model_group**（检索路径直接用 target，不走 `derive`）。新公式给出
`min(16000, 48768 − 32768) = 16000`，合理。**这不是本方案的目标，是顺带结果**，但它说明「输出比
输入大」的行在旧公式下是雷。

---

## 5. 行为差异与验收

窗口数是 `k = max(k_out, k_in)`（`chunking.py:585`）：

```
k_out = total_text / (min(coefficient_cap, max_window_subtitle_tokens) − overlap_text)
k_in  = total_mass / (prompt_input_limit − 各种占用)
```

**`k_out` 由质量护栏钉着，不由输出上限钉着**。默认 audio+retrieval 档实算：

```
planning_output_limit = 0.9 × 65536 − 5000 = 53982
coefficient_cap       = 53982 / (2.0+1.5+1.0+0.5) = 10796
max_subtitle_tokens   = min(10796, 10000) = 10000      ← 护栏赢，余量 8%
```

换成 opus 的 128000 输出，`coefficient_cap` 升到 22040，`min(…, 10000)` 仍是 10000。所以删掉输出
上限**不改变 `k_out`**；改变的是我们实际请求的 `max_tokens`（`requested_output_limit` 就是
`output_limit`）与 `is_likely_output_limited` 的判定阈值。

`k_in` 的交叉条件：

```
k_in > k_out  ⟺  媒体 token / 文本 token  ≳ 18
```

| 场景 | 媒体率 | 比值 | 主导 |
| --- | --- | --- | --- |
| audio | 32 tok/s | ≈11 | `k_out` |
| video 低清 | ≈50 tok/s | ≈17 | `k_out`（接近临界） |
| **video 高清**（agy 那条） | ≈99 tok/s | ≈33 | **`k_in`** |

**验收**：

⚠ **验收的前提是先分清哪些组的包络真的没变**。见 §4 的表：free Gemini、agy opus、conversational
的包络确实不变，而 **paid Gemini、agy gemini、claude、dsh、codex 的包络都变了**——对它们「逐窗
相等」是不成立的目标。所以：

1. **包络未变的组**（free Gemini 那一族）：audio 与低清 video 的 `plan_correction_windows` 输出
   **逐窗相等**（可 bit-exact 证明，按 `README_DEV.md` 开发原则，这是「声称语义不变的优化」那一
   类，必须精确相等）。这是本方案唯一一条硬等值验收。
2. **包络变了的组**：逐条记录窗口数变化并说明方向是预期的。haiku 变多（包络 −58000）；paid
   Gemini / agy gemini / claude / dsh / codex 在**高清 video** 上变少（`k_in` 松开，总媒体量不
   变、重叠段少发几遍，是省钱方向），在 audio 与低清 video 上**仍然不变**——那里 `k_out` 主导，
   与包络无关。⚠ 这一条不能用等值证明，必须实跑一份素材记数字。
3. 新增一条「所有现有 catalog 行的 `ModelLimits` 逐字段等于手算值」的表驱动测试，把第 4 节那张
   表钉住。

---

## 6. 已知风险与前置条件

**`docs/manual/model-routing.md:494` 明确写着**：「194,000 是 harness 的硬上限，要动它得先重标定
输出系数与窗口预算（`llm_followups.md` 的 P6/P8）」。本方案就是在动它，所以必须交代为什么风险
是有界的，以及在哪里不成立：

- **有界**：标定的对象是输出系数 `c`（输出 token 与 CSV token 的比），而 `k_out` 的分母被
  `max_window_subtitle_tokens = 10000` 钉着，删上限**不会让一窗装进更多字幕**。放开的只是「同样
  10000 字幕能带多少媒体」。
- **不成立的地方**：配置里把质量护栏关掉（`[chunking] max_window_subtitle_tokens = 0`）之后，
  `coefficient_cap` 变成唯一分母，而它随 `output_limit` 走——dsh 的 256000 输出会让它从 10796
  涨到 45080，窗口大 4 倍，正是 P6 警告的未标定区。⚠ **删掉输出上限之前必须确认这条路**：要么
  接受（护栏是用户自己关的），要么给 `coefficient_cap` 单独留一个上限。**owner 2026-09-03：
  接受**——把 `max_window_subtitle_tokens` 设成 0 是用户的显式行为，`coefficient_cap` 随输出
  上限走正是它的本意。`manual/model-routing.md` 的告警表已写明这一层去掉之后会进入未标定区。
- `capabilities.RECALIBRATION_PENDING` 已经登记了两个未重标定的开关组合
  （`text/none/efficiency`、`video/local/quality`），本方案不改变它们的状态。

---

## 7. catalog 审计：需要 owner 回答的可疑值

与本方案相邻但**不在本方案范围内**，逐条列出供裁决。

### 7.1 `194000` 出现 13 次，等于 harness 上限

这个数是 `DEFAULT_LIMITS.prompt_input_limit`（terra 那行是本分支新加的，所以是 13 不是 12）。凡是带它的行都有「填了 harness 默认值而非厂商值」
的嫌疑：

| 行 | 情况 |
| --- | --- |
| codex ×3 | 厂商 272000，**已确认是默认值误填**，本方案修 |
| gemini-free ×7 | ~~疑问~~ **已裁定（owner 2026-09-03）**：194000 当初就是按 `tpm=250000` 拍的，且**不为它改代码**——「对真实行为的高效模拟」。留空 ctx、值不动。详见 §4.1 |
| `gemini-paid-3_5-flash-lite` | **已裁定（owner 2026-09-03）**：与同档 3.8 / 3.7 一样填 `ctx = input = 1048576`。此前的 194000 在付费档没有 TPM 理由（`tpm=4000000`），是默认值误填 |
| `local-agy-opus-4_6` | **已裁定：留空不动（owner 2026-09-03）**。Opus 是单池，但没人核过 agy 是否把完整上下文透传过来，所以这一行保持它一直以来的包络 194000，而不是照厂商的 1M 认账——把 `max_input` 与池子一起填成 1M 会让包络跳到 934464，建立在一个没验证的假设上。⚠ 它的 `max_input=194000` 同样不是厂商数（恰好等于旧的 harness 上限）。真要收紧，方向是把池子填成 Anthropic 标准的 200000（包络降到 134464），前提是先核实 agy 的实际上下文 |
| `local-conversational-agent` | 本地 agent 无厂商数，填 harness 默认可理解，但应显式注明 |

### 7.2 其余

- **`gemini-free-gemma-4-31b`：16000 输入 / 32768 输出** —— 输出是输入的两倍，形状异常；且
  `tpm=-1`（无限）而其他 free 行是 250000。两处都未核。
- **quality_score 同分**：free 的 3.8 / 3.7 / 3.6 都是 75，paid 的 3.8 / 3.7 都是 75。psv 里已有
  注释说明 3.8 的限额沿用 3.7 未核，分数看来同源。
- **`token_scale` 全 1.0**：该列的意义是校正本地估算与供应商口径的系统偏差，至今没有任何模型被
  标定过——不是错，但意味着 `scale` 那条代码路径从未被真实数据走过。
- psv 里以 `#` 开头的注释行是现成的「未核值」标注约定（3.8 的限额就是这么标的），新增未核值沿用。

---

## 8. 实施顺序

1. catalog 加 `context_window` 列 + 加载校验 + 第 4 节的逐行填值；补一条表驱动测试。
2. 重写 `derive`：删提前返回，改成第 3 节两行；删 `context_limit` / `safety_margin` 字段。
3. 拆掉 `token_budget` 的第三支检查与 `correction/run.py` 的 `context_envelope`；删
   `total_with_margin` 及其两处产物读者。
4. `client.py` 加联合闸门。
5. `capabilities.py` / `model_routes.py` 的「限制成员」报告考虑 ctx（一个成员可能因 ctx 而非
   input 成为限制者）；`ENVELOPE_BASELINE_INPUT = 194_000` 重新锚定或删除。
6. 同步 `manual/model-routing.md` **两处**：第 226 行 `max_input_tokens` 的列说明（「harness 自己
   有一道 194,000 的上限，填得更大不会切出更大的窗」）与第 494 行告警表里同义的那句；再加
   `llm_harness_routing.md` 的列清单、`CHANGELOG.md`。
7. `tools/session_replay` 同步（按 CLAUDE.md，删字段属于改名/移动类改动必须同步的例外）：
   `fixture.py` 两处读写 `total_with_margin`，以及 `fast_round.py` 那处构造。
   ⚠ **本方案在这一步写错过**：它说 `fast_round.py:74` 是 `ModelLimits(safety_margin=1_000)`。
   实际是 `CorrectionBudget(max_input_tokens=…, max_output_tokens=…, safety_margin=…)`，而这
   三个参数**没有一个是 `CorrectionBudget` 的字段**——那行在本方案动它之前就已经 `TypeError`。
   写方案时只看了 grep 出来的那一行，没有去看构造的是什么类。tools/ 的测试不在默认套件里，
   所以它一直没被发现。

## 9. 实施记录（2026-09-03）

七步全部落地。审计到落地之间的三处变化，按锚点记：

| 事项 | 裁定 / 结果 |
| --- | --- |
| §6 的拍板点：护栏关掉后 `coefficient_cap` 随输出上限走 | **owner：接受**。设 `max_window_subtitle_tokens = 0` 是用户显式行为，那正是它的本意。`manual/model-routing.md` 的告警表写明了去掉这层会进入未标定区 |
| §7.1 的 `gemini-paid-3_5-flash-lite` | **owner：与同档一致，`ctx = input = 1048576`**。付费档 `tpm=4000000`，194000 在那里没有 TPM 依据 |
| 为 `provider_kind="anthropic"` 的行强制要求 `context_window`（reviewer2 建议） | **owner：不做**。它只堵住一种用户自建方言，代价是现有 override 文件启动即报错，而真正的漏洞——打包的 10 行 `local_agent`——它一行也管不着。⚠ **没有任何方言能推出池型**：`openai_compat` 底下两种都有（OpenAI 分开公布、DeepSeek 单池），每个本地 agent 又都声明 `local_agent`。所以「单池必须填」只能是文档约定，不是可执行的校验 |
| `local-agy-opus-4_6` 的 `context_window` | **owner：留空不动**，理由与实算见 §7.1 |
| §8 第 7 步对 `fast_round.py` 的描述 | **方案写错了**，见该步的 ⚠。顺带修好了那处早已存在的 `TypeError` |

**与方案预期一致、无需改口的两点**：`derive` 删掉提前返回后，免费 Gemini 那一族的
`ModelLimits` 逐字段不变（`CATALOG_WINDOWS` 表钉住）；`ENVELOPE_BASELINE_INPUT = 194_000`
不需要重新锚定——免费 Gemini 不声明 `context_window`，池子是 `194000 + 65536`，包络仍然
**恰好**落在这个基线上。

**一处方案外的扩展，复审后已实施**（owner 2026-09-03，commit `32890987`）：`client.py` 的
repair-context 丢弃逻辑原本只看 `estimated_input > max_input_tokens`，联合闸门只会跳过候选。
单池供应商上因此有个缺口——一次调用可以低于输入上限却装不下答案，于是本来还能成的那次
「盲修复」重试被丢掉了。repair context 是辅助不是任务，这一点对两道天花板同样成立，所以两者
走同一条出路，产物里的原因分开记（`dropped_input_limit` / `dropped_context_limit`）。
落地时**没有**写进原方案 §3.2 的三道闸门清单里，是复审提出后 owner 追加的。

---

## 10. 不做什么

- **不动 `max_window_subtitle_tokens = 10000` 这条质量护栏**。它是本方案风险有界的唯一依据。
- **不动输出系数 `c` 与 `window_output_budget`**。P6 标定是独立的事。
- **不动 `FAST_ROUND2_INPUT_RESERVE_TOKENS`**、`chunking` 的容量算术、resume 的重拟合逻辑——它们
  读 `ModelLimits`，跟着变即可，公式本身不碰。
- **不顺手修第 7 节的可疑值**。那是另一批决定，混进来会让「行为零变化」的验收失效。
