# 模型选择

这条管线包含四类模型：**人声分离 → 语音识别（ASR）→ 第二模型校验（Qwen3-ASR）→ LLM 纠错/翻译**，另有可选的说话人分离。
可以替换的有：ASR 使用的 Whisper 模型、说话人分离的后端与 LLM 供应商；`--asr` 还能把人声分离到说话人分离这一串整个换成一个一体转写模型（见[转写后端](asr-backend-choice.md)）。人声分离与第二模型校验各只有一个模型，本文
仅作简介，说明下载的模型分别是什么。

**结论先行**：默认值即为推荐值。下表仅在你有明确需求时阅读。

| 你想做什么 | 怎么做 |
| --- | --- |
| 正常出字幕 | 保持默认设置即可 |
| 日语素材想尝试其他模型 | 日语微调模型与默认互有胜负、净差在噪声以内，但速度**慢 2.4 倍**，详见下文 |
| 不想花 API 额度，有 Antigravity（下称 agy）订阅 | `[llm] preset = "agy"`，见 [`agent.md`](agent.md) |
| Gemini 免费额度不够用，想用完后接着用 agy 订阅 | `[llm] preset = "agy-hybrid"` |

## 1. 语音识别（ASR）

通过 `--model` 指定，批量处理时写入 manifest 行的 `"model"` 键。取值可以是 faster-whisper 的
别名（`large-v3-turbo`、`large-v3`、`tiny` 等），也可以是 Hugging Face 仓库 id 或本地目录。

| 模型 | 下载 | 相对速度 | 什么时候用 |
| --- | ---: | --- | --- |
| `large-v3-turbo`（**默认**） | 约 1.6 GB | 1× | 全部正常场景 |
| `large-v3` | 约 3.1 GB | 约 0.26×（慢 3.9 倍，只算识别解码这一步） | 只用于开发对照 |
| `TransWithAI/whisper-ja-1.5B-ct2` | 约 3.1 GB | **0.42×（慢 2.4 倍）** | 日语素材；与默认互有胜负 |

> 用后两个（large-v3 级）模型时加上 **`--asr-decode-batch 8`**，能快 1.4–1.6 倍（见[调参](tuning.md)）；
> `standard` 档（8 GB 卡，档位见[资源与大文件](resources.md#显卡档位与-cpu-回退)）上第二模型校验会因此挪到 CPU。

### `large-v3-turbo`：默认模型，整条管线按它标定

整条管线的参数都按它调过。换别的模型能跑，但那些参数对它就不再是调好的值。

### `large-v3`：**没有实测优势，只用于开发测试**

它更大、更慢、更占显存，却没有更准，只用于开发对照。

### `TransWithAI/whisper-ja-1.5B-ct2`：日语微调，**互有胜负，但慢 2.4 倍**

[上游](https://huggingface.co/TransWithAI/whisper-ja-1.5B-ct2) 是
`efwkjn/whisper-ja-1.5B`（Whisper large-v3 的日语微调）的 CTranslate2 转换版，
作者称在其测试集上字错率（CER）有竞争力。

在日语直播素材上，它与默认模型互有胜负，净差在噪声以内；它稳定好的一处是很少陷进整段复读，整行漏识别也少一些。
（实测见 [`bench-baselines.md`](../bench-baselines.md)「二十四、日语微调 vs turbo」，开发者参考，可跳过。）

### 代价：主要是慢，而且是成倍的

| | 默认 turbo | 日语微调 |
| --- | ---: | ---: |
| **识别耗时**（同 207 分钟素材，对齐阶段合计） | 290.6 s | **689.5 s（2.4 倍）** |
| 下载 | 约 1.6 GB | 约 3.1 GB |
| 常驻显存 | 2.07 GiB | 3.82 GiB |
| `entry` 档（显存预算 3 GiB，见[资源与大文件](resources.md#显卡档位与-cpu-回退)）能否使用 | 能 | **不能**（仅该模型已超出限制） |
| 第二模型校验能否与它同时放在显卡上 | `standard` 档起可以，全速 | `standard` 档上可以，但**用不了编译加速、跑得慢**；`standard_large_vram` 档（要 10 GB 空闲显存，`--gpu-tier auto` 在 12–23 GB 的卡上选它）起全速；`entry` 档退到 CPU |

**这意味着 3 小时素材的识别耗时（只算识别这一步，不含人声分离等）将从约 5 分钟增加到约 12 分钟。** 对单条视频影响有限，
对批量处理则是实际的成本。因此默认不更换：**不存在足以抵消该代价的质量收益**。

唯一值得尝试的场景是：**素材的主要问题为整段复读或崩坏**——这正是该模型稳定领先的方面。

- **请显式指定 `--language ja`。** 该模型面向日语，若交由模型自行检测语言，会浪费一次不必要的检测。
- **下载为按需触发**：该模型已列入模型清单，因此与默认权重一样支持镜像路由与校验；但它**不在
  默认下载清单中**，仅当通过 `--model` 显式指定时才会下载（约 3.1 GB）。

## 2. 人声分离与第二模型校验（不可更换）

| 模型 | 下载 | 干什么 |
| --- | ---: | --- |
| BS-RoFormer `model_bs_roformer_ep_317_sdr_12.9755` | 约 0.64 GB | 从原始音轨里分出人声，交给 VAD/ASR |
| Qwen3-ASR-0.6B | 约 1.5 GB | `--qwen-verify` 的第二模型证据：对可疑区间进行二次识别，供稳定化阶段判断；稳定化阶段的几项补救（见[调参「识别这一侧」](tuning.md#识别这一侧)）也用它听一闪而过的行、没有字幕的空档与可能漏字的行 |

两个模型均无备选，也没有用于更换模型的开关。分离器可配置的是**交付格式**（ASR 使用的 16 kHz
单声道 `.ogg` 或无损 `.flac`），这属于输出格式而非模型本身。第二模型校验可整体关闭
（`--qwen-verify off`）；默认值 `auto` 表示依赖齐全时自动启用。

## 2b. 说话人分离用的模型（`--diarize`）

只在跑说话人分离时下载：写了 `--diarize` 的某一档，或 `quality` 档默认开说话人分离时（见[说话人分离「怎么开」](diarization.md#怎么开)）。
不写 `--diarize` 时，默认只出 `-raw.srt` 的运行**一个也不会下载**。用的时候默认是 `nemotron3`，第一次用时自动下载（约 450 MB）。

| 后端 | 模型 | 获取 | 许可 |
| --- | --- | --- | --- |
| `nemotron3`（**默认**） | NVIDIA Nemotron-3-Diarization + WeSpeaker ResNet34 声纹与 PLDA（把声纹归成几个人的打分参数） | 自动下载 | 可商用（OpenMDW-1.1；声纹与 PLDA 为 CC BY 4.0，要署名） |
| `diarizen`（研究用） | DiariZen `wavlm-large-s80-md-v2` + 同一套声纹，转成本机直接能跑的格式（TorchScript） | 自动下载 | ⚠ **CC BY-NC 4.0：仅限非商业使用** |
| `sortformer`（研究用） | NVIDIA Sortformer v2.1 | 用 `tools/sortformer_export/` 自行导出 | NVIDIA Open Model License |

各组件的来源、许可与署名、GPU 与 CPU 的速度差别都在[说话人分离](diarization.md)。

## 2c. 本机的一体转写模型（`--asr moss-local`）

只在用 `--asr moss-local` 时下载，默认一个也不下。

| 模型 | 下载 | 许可 |
| --- | ---: | --- |
| [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)（固定版本，下完校验） | 约 1.7 GB | Apache-2.0 |

它代替人声分离、Whisper 与说话人分离那一串，只在 RTX 30 系列或更新的 NVIDIA 卡（计算能力 8.0 以上）上跑，见[转写后端「本地 MOSS」](asr-backend-choice.md#本地-moss--asr-moss-local)。

## 3. LLM（纠错与翻译）

管线（`finesub`）跑到 `--stage translated-srt` / `final-srt` 时直接调模型、消耗配额。
哪个任务落到哪个模型由「预设 × 任务组 × 难度」决定，完整机制见
[`model-routing.md`](model-routing.md)，本节只回答「各家好不好用」。

出厂提供五个预设：`default`（仅 Gemini API）、`agy-hybrid`（免费 API 优先，订阅作为后备）、
`agy`（仅使用订阅），以及长输入快速粗纠用的 `agy-fast`、`fast`。「快速粗纠」指这两个预设（便宜、并行、纯文本地过一遍）。

### 各后端的使用体验

以下内容为**实际使用感受，而非基准测试**，可作为选型参考，但不构成结论。

| 后端 | 多模态 | 印象 |
| --- | --- | --- |
| **Gemini（API 或 agy）的 full Flash** | ✅ 音频/视频 | 新的 Flash 档（3.8）**质量足以胜任**，是默认预设的主力 |
| **Gemini 4 Hydrogen**（API 或 agy） | ✅ 音频（视频未实测） | 对标 Flash，纠错与知识更新和 3.8 比过：**质量分不出高低**；在 agy 上想得少得多（思考约是 3.8 的四分之一到一半）、快得多，在免费 API 上反而想得更多。是 agy 预设的主力，免费档排在纠错组首位 |
| **agy 的 Gemini 4 Argon** | ✅ 音频/视频 | 在日语直播素材上与 3.8 Flash 比过一次（带音频、说话人分离开着）：**质量略好**，用的 token 少一些，字幕切得稍碎；单价更贵 |
| **3.5 Flash Lite** | ✅ | **能力偏弱，不适合纠错**；仅绑定于 `efficiency` 这一已声明的降档位 |
| **Claude Code**（Opus / Sonnet / Haiku 的最新版）与 **agy 的 Claude**（Sonnet 5.5 / Opus 5.5） | ❌ 纯文本 | 质量尚可，但 **token 效率偏低**——相同的工作需要更多 token。这是在 Claude Code 上的观察，agy 上尚未实测。Haiku 5.5 在 API 上试跑过：纠错与准备任务只**摸到能力档「中」的下界**（准备任务交给纠错的背景里漏的专名多），**不推荐用于知识更新**（出过写坏的条目）；Claude Code 这条通道没测 |
| **Codex**（GPT-6 Luna / GPT-6.1 Terra / GPT-6.2 Sol） | ❌ 纯文本 | GPT-6 Luna 试跑过：纠错与准备任务只**摸到能力档「中」的下界**（合并得少，靠译文撑着），**不推荐用于知识更新**（提的条目比 3.6 Flash 还少）。Terra 与 Sol 的纠错质量未经这样验证，可手动绑定 |
| **DeepSeek（经 dsh）** v4.1-flash / v4-pro | ❌ 纯文本 | dsh 是 DeepSeek 的本机命令行工具（DeepSeek Harness，见[本机 Agent 后端 4.1](agent.md#41-dshdeepseek-harness只能按窗口跑)），不是直连 DeepSeek API；直连 API 见[接自己的模型](custom-models.md)。**只做过单窗耗时实测**（上一代 V4 Flash 一窗 270 条约 13 分钟，见 [agent.md 4.1](agent.md#41-dshdeepseek-harness只能按窗口跑)），**没做质量评估**；它只能绑按窗口跑的格子（`agent_session` 写 `api` 或 `resume` 会报错） |

两条会直接影响选择的结构性事实：

- **只有出厂的 Gemini API 与 agy 的 Gemini 能处理音视频。** 自己经 OpenAI 兼容接口（如 OpenRouter）接的 Gemini 也只能纯文本（见[接自己的模型](custom-models.md#自己接的端点现在的三条限制)）。纯文本后端要加 `--llm-media text` 才接得到纠错，见[模型路由「媒体开关」](model-routing.md#媒体开关---llm-media)。
- **agy 的两个出厂预设里，文字任务先交给 agy 的 Sonnet 5.5，带音视频的落到 agy 的 Gemini**；组里完整的先后见 [`model-routing.md`](model-routing.md)「agent 的两个出厂预设」。

### 免费档的两个代价

- **知识更新为尽力而为（best-effort）。** 知识更新（`knowledge`）与准备任务（`research`）均以免费的强模型（Gemini 3.6 / 3.8 Flash 与 Gemini 4 Hydrogen）优先，
  而免费档每个模型每天只有 20 次请求，且**无 lite 后备**（有意不静默降质）。带准备任务的运行先花掉准备任务的次数，
  再分块做知识更新，所以免费档很容易在知识更新时用完当天的次数。
- **`--llm-retrieval native` 需要具备联网能力的成员。** 出厂由付费 Flash 提供联网；免费 3.x 完全
  无法联网，纯免费档运行 native 会在启动校验时报错。这时可配上付费 key，或显式绑定出厂的
  `gemini-native-search` 组（含免费 2.5 Flash，但它的能力档「低」够不着纠错与知识更新要的「中」）。

## 4. 模型的存放位置与删除

模型权重按需下载，存放于 `models/`（默认位于安装目录，可整体迁移到其他磁盘）。若机器上已存在
同一批权重的常规缓存（`~/.cache/huggingface`），会优先复用，避免重复下载。
路径、迁移与卸载见 [`resources.md`](resources.md)。
