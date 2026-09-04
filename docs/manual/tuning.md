# 调参：字幕口味、质量与速度

`finesub --help` 列出的选项远多于本页，但**绝大多数任务无需修改任何选项**：默认值已按默认模型
标定，改动通常只是把既有平衡替换为另一种取舍。本页仅介绍**实际会用到**的选项，说明它们解决的
问题、带来的代价，以及修改后是否需要删除产物并重跑。

三类问题各去一处：换哪个模型见 [`models.md`](models.md)；哪个任务走哪个 LLM、思考多深见
[`model-routing.md`](model-routing.md)；一次跑多个输入见 [`batch.md`](batch.md)。

⚠ 修改参数**不会重跑已完成的阶段**——阶段仅根据产物是否存在进行判断。要使新参数生效，须先
删除对应阶段的产物（见 [`outputs.md`](outputs.md)「重跑某一阶段：需删除的产物」）；本页各项
下方均注明需删除的内容。

## 字幕太长 / 太短

```powershell
finesub input.mp4 --split-length-scale 0.8
```

`--split-length-scale`（取值范围 0.6–1.6，默认 1.0）是**唯一**控制字幕长短的选项：取值小于 1
会更早切分，字幕更短更碎；大于 1 则更倾向于保留长句。1.0 为标定值（理想上限约 4.5 秒 / 20 字，
可接受上限 8 秒 / 36 字），该选项按比例缩放这一对上限。

⚠ **下限不随该选项变化**：字幕过短无法阅读属于质量问题而非风格偏好，因此任何取值都不会产生
不可读的过碎字幕。
⚠ 它在识别对齐阶段生效——**要重跑得删 `-vad.json` + `-vad-energy.npz` + `-aligned.json`
及其下游**。
⚠ 想每次都用同一个值，写进 `config.toml`:

```toml
[segmentation]
length_scale = 0.8
```

命令行指定值优先于配置文件。分句的具体判定规则（在何处切分）见
[`../segmentation-split.md`](../segmentation-split.md)。

## 识别这一侧

| 选项 | 默认 | 什么时候动它 |
| --- | --- | --- |
| `--model` | `large-v3-turbo` | 换 Whisper。三个可选模型的实测对比见 [`models.md`](models.md) |
| `--asr-decode-batch` | `auto`(= 1，关) | 运行 `large-v3` 级大模型时填 `8`，实测提速 **1.4–1.6 倍**；在默认模型上收益仅约 6%，不值得开启 |
| `--qwen-verify` | `auto` | 第二模型(Qwen)对可疑区间进行二次识别，供稳定化阶段参考。`auto` = 依赖齐全时自动启用；`off` 可节省少量时间 |
| `--asr-context` | `off` | 将知识库中的**专有名词**提供给上述第二模型：`terms` 仅提供词条名与别名，`full` 提供纠错层所见内容。素材中人名/游戏专名较多时建议开启 |
| `--lang-redecode` | `auto` | 适用于素材混用两种语言、部分段落被误判为其他语言的场景。`auto` 仅在自动检测语言（未传 `--language`）的运行中生效 |
| `--vad-silero-assist` | 开 | 默认启用，用于分离后仍较嘈杂的人声轨。源音频**非常干净**时，可使用 `--no-vad-silero-assist` 节省时间 |
| `--asr-stabilize-profile` | `0` | `0` 默认清理；`1` 只清常见幻觉；`2` 只标嘈杂段；`-1` 什么都不做（拿原始轴自己处理） |
| `--word` | 关 | 额外输出一份词级 SRT（卡拉 OK 式逐词时间轴） |
| `--gap` | `0.3` 秒 | ⚠ **不建议修改**：它是送入解码器的合成静音，属于已按默认值标定的识别核心参数 |
| `--separator-rate` | `44100` | ⚠ **不建议动**：降到 32000/22050 按比例减少分块数，代价是分离质量 |
| `--separate` | 开 | 输入**已为纯人声**（如自行分离的结果或录音棚干声）时，可用 `--no-separate` 跳过分离阶段，避开最耗资源的一步。⚠ 仅在可以确认时使用：应当分离而未分离**不会报错**，但识别质量会整体下降。通常与 `--no-vad-silero-assist` 配合使用 |

重跑时需删除的产物：`--asr-stabilize-profile` 只需删除 `-stable.json` 及其下游；其余选项均需
删除至 `-aligned.json`(`--model` / `--separator-rate` / `--separate` 还需删除 `-vocal.ogg`)。

⚠ `--vad-silero-assist` 例外：它**不参与产物复用判定**，使用旧 `-vad.json` 续跑时仅提醒一次，
不会因该值变化而重新计算；如需使其生效，同样需删除相关产物。写入 `config.toml` 的形式：

```toml
[vad]
silero_assist = false
```

## LLM 这一侧

仅在 `--stage translated-srt` / `final-srt` 时生效。**修改这些选项不会导致已提交的窗口重跑**——
删除 `-translated.srt` 等产物后，只会从**未完成的窗口**继续，详见 [`outputs.md`](outputs.md)。

| 选项 | 默认 | 什么时候动它 |
| --- | --- | --- |
| `--llm-difficulty` | `quality` | 三档 prompt/思考深度。`efficiency` 可节省配额，并同时关闭知识库；取舍见 [`model-routing.md`](model-routing.md) |
| `--llm-media` | `audio` | 模型看到什么：`text` 只给文本（最省）、`audio` 给音频、`video` 给画面（还需 `--llm-video` 指定视频） |
| `--llm-retrieval` | `local` | 联网检索：`local` 走本机检索代理，`none` 完全不查，`native` 让模型自己搜（需要该模型支持） |
| `--llm-fast` | `auto` | 将短素材合并至一个窗口处理。`off` 强制按常规窗口切分 |
| `--llm-continuity` | `serial` | `parallel` 同时发多个纠错窗口，快，但放弃窗口之间的上下文衔接。并发数由 `--llm-parallel-windows` 定（默认 1） |
| `--llm-output-scale` | `1.0` | 调整对「本窗将输出多少内容」的估计：取值大于 1 会规划更小的窗口。当模型频繁因输出长度上限被截断时才需要调整 |
| `--max-retries-per-window` | `5` | 一个窗口内格式修复的重试次数 |
| `--max-replacements-per-window` | `1` | 上述重试用尽后，为整窗更换全新会话重试的次数 |
| `--extra-info-file` | — | 与 `--extra-info` 相同，提供背景信息，但内容从文件读取——适合较长的设定或术语表 |
| `--postprocess-profile` | `0` | 成品 SRT 的后处理：`0` 繁转简 + 重叠修复 + 时长下限 + 标点；`1` 只做时长；`2` 只做标点；`3` 只做繁转简；`4` 只修重叠；`-1` 只重新渲染一遍、语义不变 |
| `--knowledge-root` | 共用那一份 | 让这次运行读写另一个知识库目录（试验、或给不同客户分库） |

⚠ 上述两档均针对**输出格式不符合要求**的情况进行重试，与「模型无法完成回答」是两回事：当单句
过长无法完成时，程序会将窗口对半拆分后重试，**最多拆一次**；拆分后仍失败则停止该任务并指明
对应窗口——不再为同一失败继续消耗配额。该上限不可配置。

风格与知识库怎么用(`--style` / `--style-mode` / `--knowledge` / `--refined-srt`)是另一件
事，见 [`knowledge.md`](knowledge.md)。

## 输入与下载

- `--no-download-video`:URL 输入时**仅下载音频**。默认会一并下载视频，因为生成的字幕通常需要
  压制回视频，重新解析 URL 的成本更高；仅需字幕或磁盘空间有限时使用该选项。
- ⚠ 该选项与 `--llm-media` 无关：`--llm-media` 决定的是**模型接收何种媒体**，与是否下载视频
  无关。

## 跑多久

以下为参考量级：**23.5 分钟**的素材，在一台 RTX 5070 Ti 上运行至 `raw-srt`（不含 LLM）约
**101 秒**，即约 14 倍实时；其中人声分离约占 41%，语音检测与识别约占 59%，其余步骤合计不足
0.5 秒。（出处：[`../bench-baselines.md`](../bench-baselines.md) 第四节 4.1，默认模型、
`--gpu-tier high`,n=1。）

该数据仅为单台机器上的一次测量，用于判断数量级，并非性能承诺。以下因素会显著改变耗时：

- **是否使用显卡**：全程 CPU 运行大约慢一个数量级（识别约为十倍差距），长音频不推荐。
- **更换更大的模型**：`large-v3` 级模型慢 2 倍以上，此时建议添加 `--asr-decode-batch 8`。
- **LLM 阶段**：耗时主要取决于「窗口数 × 每窗一次 API 往返」，与素材长度成正比；免费配额还会
  限制速率，因此该阶段通常比整条语音链路更耗时。实际调用次数与重试情况，可查看运行结束后的
  `<名字>.llm-artifacts/task-report.md`。

如需定位本次运行的耗时环节，可查看 `-metadata.json` 中的各阶段耗时，或在运行时添加
`--log-level verbose`。
