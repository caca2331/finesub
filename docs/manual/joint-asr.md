# 一体转写（`--asr`）：云端三家与本地 MOSS

把转写交给一个「一体」的模型：它直接转写原始混音，一次交回带时间与说话人的整份转写，不跑人声分离、Whisper、第二模型与声学说话人分离。
这样的模型有云端三家，也有一个在本机显卡上跑的开源模型（[本地 MOSS](#本地-moss--asr-moss-local)）。云端三家时本机只做轻量的事（解码音频、
Silero 语音检测、生产的分句、时间轴整理）。出的还是 `-aligned.json` / `-stable.json` / `-raw.srt`，之后的
说话人、LLM 纠错与导出照常。**默认关**：不写 `--asr` 时照旧用本机的 Whisper 那一串转写。
`--stage vocal` 是明确要人声轨，一体转写的任务上也照样在本机跑人声分离，只停在这一步。

```powershell
finesub input.mp4 --asr moss
```

⚠ **云端三家的音频会离开本机**：第一次上传之前打一行 `Warning:`，说明发给了哪家。本地 MOSS 不上传。

## 怎么选、怎么接

`--asr` 有五个选择：默认的本机 Whisper、本机显卡上的开源 MOSS，以及云端三家。

| | `local`（默认） | `moss-local` | `moss-pro`（`moss`） | `mai2` | `gemini-35t`（`gemini`） |
| --- | --- | --- | --- | --- | --- |
| 是什么 | 本机 Whisper 那一串 | MOSS Pro 的开源本地版 | MOSI 的 MOSS Pro，云端 | Microsoft MAI-Transcribe-2，云端 | Gemini 3.5 Transcribe，云端 |
| 字准 | 中等 | 好，略逊 MOSS Pro | 好 | 最好 | 好 |
| 词级时间戳 | 有 | 无 | 无 | 有 | 有 |
| 段级时间戳 | 有 | 有 | 有 | 有 | 有 |
| 时间准确度 | 很好，偶有退化 | 不错 | 不错 | 很好 | 不错 |
| 说话人识别 | 不错（要开 `--diarize`） | 好，人多时标签较碎 | 很好 | 较差 | 一般 |
| 允许多个字幕重叠 | 否（修好的坍缩行例外） | 是 | 是 | 否 | 是 |
| 分句质量 | 不错 | 很好 | 很好 | 很好 | 不错 |
| 音频去向 | 本机 | 本机 | 上传 MOSI | 上传微软 | 上传 Google |
| 费用 | 免费 | 免费 | 每把 key 每天 1.25 音频小时免费；超出 2.00 元人民币 / 小时 | 2026 年约 0.10 美元 / 小时（限时优惠），估计 2027 年起约 0.36 美元 / 小时 | 免费 key 每天约 24 次请求（约 3.6 音频小时）；付费性价比低，不推荐 |
| 需要 | GPU，4 GB 显存以上（没有时退回 CPU，很慢） | RTX 30 系列或更新，8 GB 显存以上；Windows 上直接运行，不需要 WSL | MOSS key | OpenRouter key 或自己的 Azure Speech 资源 | Gemini 免费 key |

- `local` 是唯一提供逐行说话人置信标、行内换人拆行与坍缩行修复的路线。
- `moss-local` 的性质与 MOSS Pro 相同，模型更小，听错、漏话比 MOSS Pro 多，见[本地 MOSS](#本地-moss--asr-moss-local)。
- `gemini-35t` 不用额外配置：一把 Gemini 免费 key 就能跑完整条管线，只受每天的额度限制。它的字准按 Google 官方公布的字错率与另几家对比定，本项目没量过它的字错率。
- MOSS Pro 的价格以 MOSI 的[价目页](https://platform.moss.cn/app/pricing)为准（2026-11，按它的积分账扣），控制台计的时长比文件长约 5–6%；超出免费额度的部分由池里最后一把 key 付费，见[额度与费用](#额度与费用)。
  MAI2 的优惠价到 2026-12-31，以 [OpenRouter 的模型页](https://openrouter.ai/microsoft/mai-transcribe-2)与微软的[价目页](https://azure.microsoft.com/pricing/details/cognitive-services/speech-services/)为准。
- 云端几家都由本机在模型给的词或段上重新分句，表里的分句质量是本机修正之后的。MAI2 给的时间偏紧，本机把每一行两头各放宽 0.1 秒，表里的时间准确度是放宽之后的。
- 评价来自 2026-10 与 2026-11 的几轮评测（纠错之前的字幕对人工字幕，再加读者模型盲判），只能当方向看。
- 短名 `gemini`、`moss` 跟着提供方的当前模型走，产物里记的是全名。

### 拿到 key

key 写在 `.env`，变量名与写法见[环境变量与 API Key](env.md) 的「[云端转写（`--asr`）](env.md#云端转写--asr)」一节。各家去这里申请：

- **MOSS Pro**：MOSI 开放平台的控制台 [platform.moss.cn](https://platform.moss.cn)，在它的「[API Keys](https://platform.moss.cn/app/api-keys)」页建 key（API 在 `api.mosi.cn` 上）。
- **35t**：在 [Google AI Studio](https://aistudio.google.com/apikey) 建的 Gemini key，与 LLM 层共用一池，见[环境变量与 API Key](env.md)「[Gemini(Google AI Studio)](env.md#geminigoogle-ai-studio)」。
  35t 也可以经 OpenRouter 接、在那边付费（对中国大陆用户付款可能更方便），本项目不接这条路：免费额度用完不如改用另两家。
- **MAI2 经 OpenRouter**：在 [openrouter.ai/keys](https://openrouter.ai/keys) 建 key；当前价格在它的[模型页](https://openrouter.ai/microsoft/mai-transcribe-2)。
- **MAI2 经 Azure**：在 [Azure 门户](https://portal.azure.com) 建一个 Speech 资源，区域选快速转写（fast transcription）可用的（[区域表](https://learn.microsoft.com/azure/ai-services/speech-service/regions?tabs=stt)）；
  地址与 key 在资源的「Keys and Endpoint」页。微软的[价目页](https://azure.microsoft.com/pricing/details/cognitive-services/speech-services/)注明 MAI-Transcribe-2 的价格是到 2026-12-31 的限时优惠。

### 设成默认、走代理

可以在 `config.toml` 里把一家设成默认：

```toml
[asr]
backend = "moss-pro"
```

请求走 `[api] proxy` 设的代理（见[模型路由配置](model-routing.md)「走代理：`[api] proxy`」）。

## 额度与费用

- **一把 key 用到当天额度用完才换下一把**，不轮流用：出了问题能知道是哪把 key。池里所有 key 当天都用完时任务报错停下，额度日翻过去再跑。
- **多配几把免费 key 能多转，但别太多**：在同一台设备上先后用两把免费 key（一把用完换下一把），没出过问题；更多把没试过，有被服务方风控的风险。
  `GEMINI_FREE` 建议最多 2 把、`MOSS_PRO` 最多 3 把：`config.toml` 的 `[pools]` 没点名时只用 `.env` 里排在前面的这几把，点名的超过建议数时照用、打一行 `Warning:`
  （见[环境变量与 API Key](env.md)「[Provider 与 pool 配置](env.md#provider-与-pool-配置)」）。
- **MOSS Pro**：本地记着每把 key 当天用了多少音频，一次调用整个放不进某把 key 当天的免费额度就换下一把。
  **池里最后一把 key 例外**：一次调用放不进前面几把剩下的免费额度时，照常用它，超出它免费额度的部分从它的账户余额扣（MOSS Pro 是预付费，扣到余额为零为止）。
  不想付费就在 `config.toml` 写 `[asr] moss_pro_paid = false`，这时所有 key 的免费额度用完任务就停下。「最后一把」是实际用到的那几把里的最后一把：
  `[pools] moss_pro` 点名时按点名的顺序，没点名时是 `.env` 里前 3 把中的最后一把。
  服务端拒绝（HTTP 402）时：
  - 付费的那把：余额用完了，或当天的免费额度还没刷新，任务停下。登录一次 MOSI 控制台（这同时刷新当天的额度），余额为零时充值，再重跑，当天就能接着用。
  - 前面的 key：多半是当天的额度还没刷新，本机把它记成当天用完、换下一把，到北京时间零点为止不再用它；登录一次 MOSI 控制台，等第二天再跑。
  额度日从北京时间零点起。
- **Gemini 3.5 Transcribe**：撞上某把 key 的日上限就换下一把，额度日从太平洋时间零点起。
- **MAI2**：即使回答都已缓存（例如删了 `-aligned.json` 重新分句），也要 `.env` 里有它的 key，开工前那行费用估计照样打。OpenRouter 与 Azure 两条路切窗不同（超过 30 分钟的源），换了路线就对不上缓存，会重新计费。
- 每次调用的原始回答存在任务目录旁的 `<名字>-cloud/` 里：**重跑不再上传、不再花额度**。这个目录要留着（清理中间产物时也不删），
  删了就只能再花一次额度重转，而且同一段音频再问一次会得到不同的回答。源文件换成了另一段录音（同名任务目录）时，旧回答对不上这段音频，
  会被删掉、重新请求并花额度，同时打一行 `Warning:`。

## 本地 MOSS（`--asr moss-local`）

开源的 MOSS-Transcribe-Diarize（MOSS Pro 的开源版，0.9B 参数，Apache-2.0 许可）在本机显卡上转写原始混音，本机另加了一套救援，补它已知的失败
（复读、时间跳走、提前收尾、整段换成别的语言、漏掉的片段）。说话人、两个人同时说的行、`--asr-aligner` 都与 MOSS Pro 一样。

```powershell
finesub input.mp4 --asr moss-local --language ja
```

- **什么卡能用**：NVIDIA 显卡、计算能力 8.0 以上（RTX 30 系列及更新、A100 之类），8 GB 显存以上（`standard` 档）。别的卡（RTX 20 / GTX 16 系列、AMD、6 GB 及以下的卡）、没有显卡、`--gpu-tier entry` 或 `cpu`、`--device cpu` 时，开工前报错停下，
  请改用本机的 Whisper（`--asr local`）或云端；不会悄悄换成别的模型。只在 RTX 5070 Ti 上实测过。
- **要多少显存**：跟着 `--gpu-tier` 走（`auto` 按显卡定）。8 GB 卡（`standard`，6.5 GiB）一趟转 33 分钟，更长的源自动切成几段再拼回（段间重叠 1 分钟）；
  10 GiB（`high`）一趟 68 分钟。30 分钟的素材约 2–4 分钟（RTX 5070 Ti）。
- **第一次用**：下载约 1.7 GB 的模型（只下一次，按所在地区走镜像或官方源，下完校验）；第一次解码还要编译几个 GPU 内核，多等一会儿，之后存在缓存里。
  批处理里下载发生在本机转写那条道上，排在后面的本机 Whisper 任务会一起等。
- **语言**：模型自己支持 50 多种语言，`--language` 什么都收；英文用英文的提示词，其余用上游默认的中文提示词（中文提示词也能转日语等）。
  **请给 `--language`**：不给时打一行 `Warning:`，英文素材在默认的中文提示词下可能整份转成别的语言（见过一次转成德语），给了之后本机在转写中途检查转出的文字，像是转成了别的语言就重解那一段、尽量拉回来，拉不回就照原样留下：英文看常用词（拉丁字母的别种语言查得出，整段汉字、假名查不出），日语、中文看假名与字母；其余常见语言只查文字体系（例如拉丁字母变成了汉字），再少见的语言不查。只验过中文、英文、日语。
- **和 MOSS Pro 比**：2026-11 在 15 份中英日剪段上，字错率与 MOSS Pro 打平、好于本机 Whisper；但读者盲判里 MOSS Pro 更常被选中，差在听错与漏话（MOSS Pro 是一个更强的模型）。
  说话人与 MOSS Pro 相近，在人多、不是同一档节目的素材上标签更碎（一个人拆成几个标签）。
- **不缓存**：删了 `-aligned.json` 重跑就重新解码（只花时间）；`--asr-repair` 对它不起作用。
- 依赖都在 `[asr]` 里（transformers、Windows 上的 Triton），不用另装。Windows 上直接运行，不需要 WSL，也不用在 BIOS 里开虚拟化。

## 一个任务记着自己用的是谁

任务的产物（`-aligned.json`、`-stable.json` 的 metadata）记着是哪家转写的，**续跑时照产物走，不看这次的 `--asr`**：
云端转写过的任务换成 `--asr local` 续跑，照样用原来的云端结果；本机转写过的任务加上 `--asr moss` 续跑，也照样用本机的结果。两种情况都只多一行提示。
要换一家（或换成本机 / 云端），删掉 `-aligned.json` 与 `-stable.json` 以及它们下游的产物再跑。本地 MOSS 也一样。

## 说话人

一体转写任务的说话人标签**来自转写模型自己的识别结果**，不跑本机的声学说话人分离：

- 不管 `--diarize` 是哪一档，每行都标上转写模型给的说话人（写进 `-diar-rows.json`，不花额度、不跑模型），所以 `off` 与 `acoustics` 在一体转写任务上是一样的；
  `judge-text` / `judge-audio` 再加 LLM 说话人裁判，裁判听的是原始混音。详见[说话人分离](diarization.md)。
- 没有逐行置信标，也不做行内换人拆行；`--diarize-backend` 不起作用，写了只打一行说明。
- `gemini-35t` 偶尔有一整个窗（约 9 分钟）的回答不带说话人：这一段的行都算同一个说话人，彼此不同时显示（按[同一个人](#两个人同时说话)的规则截短），两个人的字也可能进同一行。
- MOSS Pro 的回答撞了输出上限要续问、或中间漏了一段要补问时，补回来的那次调用会重新给说话人编号。本机用声纹（与默认的说话人分离同一个 26 MB 的模型，在 CPU 上跑，30 分钟素材约半分钟，第一次用时下载）把这些标签对回已经出现过的人；对不上的（说得太少，或声音不像任何已出现的人）当新人，标签带 `c1:` 这样的前缀。声纹模型拿不到时打一行 `Warning:`，补回来的人全当新人。

## 两个人同时说话

一体转写给出的几个说话人同时说话时（MOSS Pro、本地 MOSS 与 `gemini-35t` 常见，例如一个人说着长句、另一个人插一句附和），各自的行**在时间上重叠**，交付时也保留：

- **分句**：`gemini-35t`、`moss-pro` 与 `moss-local` 按话轮分句（一个人连着说的一段），两个人的字不会进同一行；行按开始时间排。
- **SRT**：两个人的行写成各自的 cue，播放器同时显示（VLC、mpv 等基于 libass 的播放器把后开始的叠在上面）；**同一个人**的行之间照旧不重叠，重叠时截短较早的那条。字幕后处理的延长也不会让一条 cue 盖到它原本不重叠的别人的行上。
- **分轨导出**（`--speaker-tracks ass`）：每个说话人一个样式，同时说的两句是两条同时的事件。
- **LLM 纠错**：两个人同时说的行不并成一条字幕；被插话隔开的同一个人的两行也不并。纠错窗口不切在附和这类「别人还没说完时开始的行」之前。
- **说话人裁判**：裁判被告知轨号是转写模型标的说话人（常把一个人拆到几条轨），而不是声学层的轨。

本机转写的任务不受影响：行照旧按「不重叠」交付。

## 补漏转与复读：`--asr-repair`

云端转写偶尔会漏掉一段话、把一个词连写几十次。默认（`on`）本机找出这两样并补上：

- **漏转**：两个词之间约 2.6 秒以上的空档（每个词前后先各放 0.3 秒，再看剩下的空档够不够 2 秒）里，Silero 听到 ≥1.5 秒语音，就补问这一段。
  - `mai2`：把这一段（前后各多 10 秒）再交给 MAI2 转一次，只取落在空档里的词。补问按音频计费，约多 6%，开工前那行费用估计里写着。
    说话人按前后多出来的 10 秒里谁和谁同时在说对回已有的人，对不上的再比声纹，都对不上记作 `unknown`。
  - `gemini-35t`：交给 Gemini Live（不占每天的请求数），连同空档两边各约 5 秒已经转出来的话一起送去，只取 Live 写在这两头之间的字；
    和两边已有的话近似（同一句再写一遍的变体）、不到 3 个字、或 Live 没写出两边的话（定不了空档在哪）时不补。Live 写出了两边的话时，夹在中间的外语也照补（中文讲座里引的英文原文）。
    这一次连不上、或什么都没写而两边明明有话时，改送这一段本身（前后各 1 秒）。
    Live 只回文字：时间由 Silero 按字数分（开了 `--asr-aligner` 时用强制对齐模型），说话人按声纹配到这一份里最像的人，配不上记作 `unknown`。
    补的文字来自另一个模型，有时只补回一部分，补回的一句也会摊在整个空档上。
- **复读**（`gemini-35t`）：同一个词或短语连写 ≥10 次。快过人能说的（每次不到 0.12 秒）本机直接裁成每秒 4 次；正常语速的照上面的做法连同前后的话交给 Live
  再听一遍（Live 写不出前后的话、或什么都没写时改送这一串本身），Live 连写的次数不到一半才换成 Live 的文字，所以真的口令会留着。

`gemini-35t` 偶尔整个窗（约 9 分钟）不给说话人：`on` 不补，这一窗的话说话人是 `unknown`（按声纹配人的补法还没验收）。

`quota` 在这之上让 `gemini-35t` 花每天的请求数，把漏转、复读的那一段与不带说话人的窗再问它自己一次（每个源最多 4 次请求；一把 key 一天约 24 次），
问不了的照 `on` 走。`off` 什么都不补。`moss-pro` 不看这个选项：它自己的续问与补问照旧。也可以写进 `config.toml` 的 `[asr] repair`。

- 补问的回答也缓存在 `<名字>-cloud/` 里（`mai2-repair/`、`gemini-35t-live-ctx/`、`gemini-35t-live/`、`gemini-35t-repair/`），重跑不再花。
- 补问失败（连不上、服务端拒绝、额度用完）只放弃那一处，打一行 `Warning:`，任务照常完成；续跑也不再补，要补就删掉 `-aligned.json` 与它下游的产物再跑。
- `-aligned.json` 的 `metadata.joint_asr` 里，`repair` 记着这次的取值，`log` 逐处记着补了什么、走的哪条路。

## 重的本地计算：`--asr-aligner`

MOSS Pro 与本地 MOSS 只给到段（一段常是一个人的一整段话），段内每个字的时间由本机的 Silero 语音检测分配。`--asr-aligner`
（或 `config.toml` 的 `[asr] aligner = true`）让超出字幕长度上限的长段改用本机的强制对齐模型（Qwen3-ForcedAligner）对时间。

- 对 `moss-pro`、`moss-local` 的长段与 `gemini-35t` 修补时 Live 补回的文字起作用；命令行上给别的后端写了只打一行说明，`config.toml` 里的设置在别的后端上不提示。
- 要 `[asr]` 依赖里的 transformers，用 GPU（没有 GPU 时退回 CPU，打 `Warning:`）；第一次用时下载模型。

## 安装

Windows 上用 `uv tool install finesub` 装的命令行不用另装：它首次运行时建的运行环境已经带着云端转写要的依赖。Linux / macOS 走[仓库安装](repo-install.md)，按下面的源码安装加上 `[cloud]`。

源码安装：只用云端转写时 `pip install -e ".[cloud]"`（不含本机转写那一套，torch 只在 CPU 上跑 Silero）；已经装了 `[asr]` 的，
再加上它：`pip install -e ".[asr,cloud]"`（`gemini-35t` 的分句模型要的 onnx、sentencepiece 不在 `[asr]` 里，缺了会在转写回来之后报错，回答已缓存，不丢额度）。
⚠ Windows 上从 PyPI 装到的 torch 就是 CPU 版；Linux 上要 CPU 版，先从 `https://download.pytorch.org/whl/cpu` 装 torch 与 torchaudio，否则会拉下 CUDA 版。
`gemini-35t` 第一次用时下载一个约 1.1 GB 的分句模型（只下一次）。

## 现在的限制

- MOSS Pro 偶尔会把英文素材转成日语译文：本机检查到文字不是素材语言的文字时重转一次，仍不对就报错停下。这项检查要知道素材的语言（`--language`）。
- 一批任务里，云端任务不排在本机转写后面：它们另走一条可以同时跑两个的道（见[一次跑多个输入](batch.md)）。本地 MOSS 排在本机转写那条道上。
- [补漏转与复读](#补漏转与复读--asr-repair)的补法验收时只让读者对着人工字幕读文字，没人听音频；Live 补的字偶尔不对或只补回一部分，背景里另一个人同时说的话它多半也听不出。
  `quota` 自己重问 `gemini-35t` 还没验收。
