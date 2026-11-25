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
# 在你原来的命令后面加上这个选项，例如：
finesub input.mp4 --language ja --stage final-srt --split-length-scale 0.8
```

`--split-length-scale`（取值范围 0.6–1.6，默认 1.0）是控制字幕长短的比例：取值小于 1 字幕更短更碎，
大于 1 更倾向于保留长句。它管两处，都按这个比例缩放上限：

- **识别对齐时的分句**：1.0 为标定值（理想上限约 4.5 秒 / 20 字，可接受上限 8 秒 / 36 字）。
- **LLM 纠错时的合并**（跑到 `translated-srt` 以上才有）：几行并成一条字幕的上限，1.0 时一般约 20 字 / 4 秒、
  绝对不超过 36 字 / 7 秒；0.8 时是 16 字 / 3.2 秒、29 字 / 5.6 秒，1.25 时是 25 字 / 5 秒、45 字 / 8.8 秒。
  本来就长的单句照原样输出，不受它约束：纠错只会合并行、不会把一行拆开，这样的行要在成品的副本里手工拆（见[运行产物「自己改字幕」](outputs.md#自己改字幕)）。

⚠ **下限不随该选项变化**：字幕过短无法阅读属于质量问题而非风格偏好，所以把它调小也不会让字幕碎到读不了。
读速（每秒几个字算读不完）也不随它变。这一条只说这个选项管得到的范围；另有一个与它无关的已知问题：极少数素材末尾会有看不到的约 1 毫秒字幕，
见[运行产物「`--stage`：指定停止的阶段」](outputs.md#--stage指定停止的阶段)。
⚠ 想每次都用同一个值，写进 `config.toml`:

```toml
[segmentation]
length_scale = 0.8
```

**LLM 合并用哪个比例**，依次看：命令行的 `--split-length-scale`、这次所用风格的字幕长度档位（风格档案节的
`[字幕长度] 偏短 / 标准 / 偏长`，分别是 0.8 / 1.0 / 1.25，见[知识库](knowledge.md#风格条目style记住一套翻译习惯)）、
`config.toml`、1.0。识别对齐看不到风格，只看命令行与 `config.toml`。所以风格写了档位时，两处的比例可以不同：
风格是偏长，LLM 会把识别切开的短句并得更长；风格是偏短，LLM 只能少并，**切不得比识别更碎**——要更碎，得用更小的
`--split-length-scale` 重跑识别对齐。

⚠ **识别对齐**：比例在这一阶段生效，**要重跑得删 `-vad.json` + `-vad-energy.npz` + `-aligned.json`
及其下游**。分句的具体判定规则（在何处切分）见 [`../segmentation-split.md`](../segmentation-split.md)（开发者参考，可跳过）。
⚠ **LLM 纠错**：一个任务的比例在开始纠错时定下，记在 `<名字>.llm-artifacts/correction-mode.json`；续跑沿用记下的
比例，命令行给了不同的值只提示一行、不换，`config.toml` 或风格的档位后来改了也不换。要换比例，删掉 LLM 产物（含
`<名字>.llm-artifacts/` 目录，见[重跑某一阶段](outputs.md#重跑某一阶段需删除的产物)）再跑；还没交付成品的任务也可以加 `--no-resume`（已有成品时它照样复用成品）。

## 字幕停留时间

生成字幕时，每条字幕的结尾会先延长一小段（默认 0.3 秒），让字幕在说完后多停留片刻；延长后与
下一条之间仍不足一定时长（默认 0.3 秒）的空隙会直接合上，避免字幕闪一下就消失。两个值都只能在
`config.toml` 里改：

```toml
[postprocess]
end_extend_seconds = 0.5
flash_gap_seconds = 0.3
```

两者都须 ≥ 0，没有上限：延长永远不会越过下一条字幕的开始，也不会越过音频末尾（音频长度已知时；没记音频长度的旧产物
见[运行产物](outputs.md)），写得再大也只是一直延长到下一条开始。
⚠ 它在生成 SRT 时生效——原文导出删 `-raw.srt` 重新生成即可；翻译成品只删成品 `.srt`，会从
`-translated.srt` 重新生成，**译文保留**。

## 识别这一侧

| 选项 | 默认 | 什么时候动它 |
| --- | --- | --- |
| `--asr` | `local` | 谁来转写：本机的 Whisper 那一串，云端的 `mai2` / `scribe` / `moss-pro`（`moss`）/ `gemini-35t`（`gemini`），或本机显卡上的开源 MOSS `moss-local`（8 GB 以上的 RTX 30 系列以上 N 卡），以及两者的组合 `moss-diarization-aligner`（文字用云端一家、说话人由本机的 MOSS 定，配下面的 `--asr-rows`）。除了 `local`，都不跑人声分离，也不用 Whisper 转写（MOSS 两家有 GPU 时只借 Whisper 重定一部分切点的时间，见下一行）；云端的音频会发给提供方。费用、额度、显卡要求与续跑规则见[一体转写](asr-backend-choice.md)。也可写进 `config.toml` 的 `[asr] backend` |
| `--asr-rows` | 无 | 只给 `--asr moss-diarization-aligner`（文字用云端一家、说话人由本机的开源 MOSS 定）：文字从哪家来，`mai2` / `scribe` / `gemini-35t`；这个后端新转写时必须有（读 `--asr-transcript` 时也要：按它认文件的格式），续跑已有的结果不用。别的后端在命令行或 manifest 里给了它会报错，`config.toml` 里的不算。见[一体转写](asr-backend-choice.md#别家的文字本地-moss-的说话人--asr-moss-diarization-aligner)。也可写进 `config.toml` 的 `[asr] rows` |
| `--asr-transcript` | 无 | 只给 `--asr moss-diarization-aligner`：已经转好的文字，不再调 API——之前任务的 `<名字>-cloud/` 目录，或一份 `--asr-rows` 那家形状的 JSON（自己转写的按 Scribe 的形状写）。每个源一份，批处理写在 manifest 那一行的 `asr_transcript`；别的后端给了它会报错。见[一体转写](asr-backend-choice.md#别家的文字本地-moss-的说话人--asr-moss-diarization-aligner)「自己转好的文字怎么接进来」 |
| `--asr-overlap-cue` / `--no-asr-overlap-cue` | 开 | 只给 `--asr moss-diarization-aligner`：同一个人被别人插话切开的几行合成一条字幕、插话与它同时显示；关掉就不合，被切开的几行各成一条（不同的人同时说话的行照样重叠显示）。见[一体转写](asr-backend-choice.md#别家的文字本地-moss-的说话人--asr-moss-diarization-aligner)。也可写进 `config.toml` 的 `[asr] overlap_cue` |
| `--asr-aligner` | `auto` | 超长段在段内切行的时间由本机的强制对齐模型对：`auto` 给 `moss-local`、`moss-pro` 的超长段与 `gemini-35t` 修补补回的文字，有 GPU 时在卡上并由 Whisper 再修一遍 MOSS 两家没把握的切点（英语不修），没有时在 CPU 上慢跑；`on` 不看语言、英语也修；`off` 都用 Silero。见[一体转写](asr-backend-choice.md)「重的本地计算」。也可写进 `config.toml` 的 `[asr] aligner` |
| `--asr-repair` | `on` | `mai2` 与 `gemini-35t`：补云端漏转的话、裁掉失控的复读；`scribe`：时间被挤坏的那一段再转一次。`on` 不花 `gemini-35t` 每天的请求数（`mai2` 的补问约多 6% 费用），`quota` 让 `gemini-35t` 也花请求数再问它自己（也给不带说话人的一窗配人），`off` 不补。见[一体转写](asr-backend-choice.md)「补漏转与复读」。也可写进 `config.toml` 的 `[asr] repair` |
| `finesub quota`（命令） | — | 某把 key 因额度用完或被拒被搁置（Gemini 某个模型当天用完、本机 agent 停用 2 小时、搜索 key 停用 24 小时、ElevenLabs 冻结到刷新时间、35t 与 MOSS Pro 当天记成用完），而你知道它已经能用了（充了值、换了档、额度恢复了）：`finesub quota` 看是哪几把，`finesub quota thaw <名字>` 或 `--all` 提前放回。见[密钥](env.md)「额度与解冻」 |
| `--model` | `large-v3-turbo` | 换 Whisper。三个可选模型的实测对比见 [`models.md`](models.md) |
| `--asr-decode-batch` | `auto`(= 1，关) | 运行 `large-v3` 级大模型时填 `8`，实测提速 **1.4–1.6 倍**；在默认模型上收益仅约 6%，不值得开启 |
| `--qwen-verify` | `auto` | 第二模型(Qwen)对可疑区间进行二次识别，供稳定化阶段参考。`auto` = 依赖齐全时自动启用；`off` 可节省少量时间，同时 `--squeeze-repair`、`--gap-fill` 的 `auto` 也不跑（两件补救不再提示） |
| `--asr-context` | `terms`（`--knowledge none` 时为 `off`） | 将知识库中与本视频相关的**专有名词**提供给上述第二模型：`terms` 仅提供词条名与别名，`full` 提供纠错层所见内容，`off` 不提供。名字从视频标题与 `--extra-info` 选出；想让第二模型只凭音频判断时用 `off` |
| `--lang-redecode` | `auto` | 适用于素材混用两种语言、部分段落被误判为其他语言的场景。`auto` 仅在自动检测语言（未传 `--language`）的运行中生效 |
| `--vad-silero-assist` | 开 | 默认启用，用于分离后仍较嘈杂的人声轨。源音频**非常干净**时，可使用 `--no-vad-silero-assist` 节省时间 |
| `--asr-stabilize-profile` | `0` | `0` 默认清理；`1` 只清常见幻觉；`2` 只标嘈杂段；`-1` 什么都不做（拿原始轴自己处理） |
| `--squeeze-repair` | `auto` | 一整句被挤进不到一秒、一闪而过的行（坍缩行）：挪到真正说这句话的地方，和下一行重叠时两行同时显示。`auto` 在缺东西时（人声轨被删了、没有 transformers 5.x、读音库缺失、模型下不了）或 Qwen 推理时运行出错（缺 kernel、显存不足）时跳过并提示，`on` 则报错；`--qwen-verify off` 时 `auto` 也跟着不跑。只在默认的 `--asr-stabilize-profile 0` 下生效。已有的 `-stable.json` 不会重做，要重做删掉它与下游产物（人声轨要还在） |
| `--gap-fill` | `auto` | 有人声、却没有任何一行字幕的空档（至少 1 秒）：Qwen 与 Whisper 各听一遍，两家听到同样的话才补一行（或把附近错位的行挪过去）。补的行没有逐词时间。Whisper 用产出这些字幕的那个模型（默认 `large-v3-turbo`，只在它上面评测过）。`auto`、`on` 与 `--qwen-verify` 的关系同上。两件一起，长音频的总耗时约多 15–20%，每个源另加约 10 秒加载；没有 GPU 时在 CPU 上跑并提示 |
| `--row-fill` | `off` | `auto`、`on`、`off`。行的时间对、行里却漏了几个字或一个短语时补回去：Qwen 与 Whisper 各把行再听一遍，两家都听到、行里没有、也不是隔壁行的话，才插进行里它该在的位置（行的时间不变）。**多花时间换少量质量**：`on` 长音频每小时约多 1.4 分钟（整条管线约 +35%）；`auto` 只重听最可能漏字的那些行，时间约一半（约 +18%），补回的约是 `on` 的四分之三。所以默认关。所有语言都跑，评测只在日语、普通话、英语上做过。缺东西时（人声轨、transformers 5.x、读音库、模型）或 Qwen 推理时运行出错（缺 kernel、显存不足）时 `on` 报错、`auto` 跳过并警告；`--qwen-verify off` 时 `auto` 不跑，`on` 照跑。只在 `--asr-stabilize-profile 0` 下生效，已有的 `-stable.json` 不会重做 |
| `--diarize` | 跑到 LLM 阶段的 `quality` 档 `judge-text`，其余 `off` | 说话人分离，四档：`off` / `acoustics`（声学分人）/ `judge-text` / `judge-audio`（再让 LLM 读字幕或边读边听，逐行复核说话人，要跑到 `--stage translated-srt` 以上）。**默认**：跑到 LLM 阶段、`--llm-difficulty` 是 `quality` 时 `judge-text`，其余 `off`（带音频的 `judge-audio` 不一定更好，有时更差）；续跑不写这个选项时照任务已定下的档走。开了以后字幕的分行与 LLM 纠错的走法会变，见[说话人分离](diarization.md)。第一次用会自动下载约 450 MB 的模型；没有 GPU 也能跑，但慢得多；跑不起来时打一行 `Warning:`，字幕照常产出。⚠ 结果**当产物留着**，续跑复用；改档后删 `-speaker-judge.json` 才会按新档重判。效果、代价、许可与产物见[说话人分离](diarization.md) |
| `--diarize-backend` | `nemotron3` | 用哪个模型产出时间线。`diarizen` 与 `sortformer` 留作研究用，一般不要选（见[说话人分离](diarization.md)）。⚠ 指定了就不回落：满足不了会提示并跳过。⚠ **已有的 `-diar.json` 不会因为换了这个选项而失效**：它照样被复用，只多一行提示说缓存来自哪个后端；要换后端的结果，按[运行产物「重跑某一阶段」](outputs.md#重跑某一阶段需删除的产物)的「只重做说话人分离」一行删（不止 `-diar.json`：行跟着变时纠错也要重跑，会花额度） |
| `--diarize-max-speakers` | 不限 | 最多分出几个说话人（`sortformer` 会不加提示地忽略它）。⚠ **建议别开**：实测把上限设成**真实人数 + 1** 仍让 24 个文件里 23 个变差——多出来的轨常常不是噪声，而是嘉宾、游戏角色配音、背景人声，上限一紧它们只会被并进某个真人的轨。只在你确知人数、而且默认结果明显拆碎了时才用。**改完要按[「只重做说话人分离」](outputs.md#重跑某一阶段需删除的产物)删产物才会重跑** |
| `--diarize-split` / `--no-diarize-split` | 开 | 开了 `--diarize` 时，日语里两个人先后说进同一行、声学上听得清换人处的，拆成两行（不用 LLM）。默认 `-raw.srt` 与 LLM 阶段用拆过的版本，关掉则用原来的。效果与限制见[拆开两个人的行](diarization.md#拆开两个人的行)。⚠ 续跑跟着任务已定下的那一版、不看这个选项；要换见同一节 |
| `--speaker-tracks` | `off` | 按说话人导出 **ASR 原文**（不是纠错、翻译后的字幕）：`srt`、`ass` 或 `both`。需要说话人标签（`--diarize` 不是 `off`）。导出什么、怎么拆成品字幕、已经跑过的任务怎么补导出，见[转成分轨](diarization.md#转成分轨) |
| `--word` | 关 | 把原文的 `-raw.srt` 写成一个词一条（卡拉 OK 式逐词时间轴），不另出文件。纠错、翻译与成品字幕不受影响（LLM 读的是 `-stable.json`），所以出不了逐词的中文。已经跑过的任务删 `-raw.srt` 再跑就会重写成逐词的 |
| `--empty-cues` / `--no-empty-cues` | 关 | 没有文字的行：默认不写进 `-raw.srt` 与分轨 SRT；打开后写成内容是 `""` 的一条，字幕软件不会自动删掉它，也看得出它占了多长。`-stable.json` 与 LLM 阶段不受影响（纠错、翻译的成品本来就不写没有文字的行）。也可写进 `config.toml` 的 `[postprocess] empty_cues`。在生成 SRT 时生效：删 `-raw.srt` 重新生成 |
| `--gap` | `0.3` 秒 | ⚠ **不建议修改**：它是送入解码器的合成静音，属于已按默认值标定的识别核心参数 |
| `--separator-rate` | `44100` | ⚠ **不建议动**：降到 32000/22050 按比例减少分块数，代价是分离质量 |
| `--separate` | 开 | 输入**已为纯人声**（如自行分离的结果或录音棚干声）时，可用 `--no-separate` 跳过分离阶段，避开最耗资源的一步。⚠ 仅在可以确认时使用：应当分离而未分离**不会报错**，但识别质量会整体下降。通常与 `--no-vad-silero-assist` 配合使用 |

重跑时需删除的产物：格里写了的照格里删；没写的识别侧选项作用在识别对齐，删 `-vad.json` + `-vad-energy.npz` + `-aligned.json` 及下游，`--separator-rate` / `--separate` 再删 `-vocal.ogg`；`--asr-stabilize-profile` 只删 `-stable.json` 及下游。按阶段的对照见[运行产物](outputs.md)「[重跑某一阶段：需删除的产物](outputs.md#重跑某一阶段需删除的产物)」。

⚠ `--vad-silero-assist` **不参与产物复用判定**，使用旧 `-vad.json` 续跑时仅提醒一次，
不会因该值变化而重新计算；如需使其生效，同样需删除相关产物。写入 `config.toml` 的形式：

```toml
[vad]
silero_assist = false
```

输入一律是纯人声时，`--no-separate` 也可以写进配置，免得每次都带：

```toml
[separator]
enabled = false
```

## LLM 这一侧

仅在 `--stage translated-srt` / `final-srt` 时生效。LLM 把字幕分成一段一段送给模型，一段叫一个**窗口**；一个窗口的结果存下了，
就叫这个窗口**已提交**。**修改这些选项不会导致已提交的窗口重跑**——删除 `-translated.srt` 等产物后，只会从**未完成的窗口**继续，
详见 [`outputs.md`](outputs.md)。

| 选项 | 默认 | 什么时候动它 |
| --- | --- | --- |
| `--llm-difficulty` | `intermediate` | 默认的 `intermediate` 适合多数素材：强模型一步纠错。多人对谈、要分清谁在说或要校对时用 `quality`（更贵）；额度用完或要大量便宜跑时用 `efficiency`（便宜的 lite 模型，有几项限制）。各档做什么、怎么选见 [`model-routing.md`](model-routing.md#difficulty-是定位) |
| `--llm-media` | `audio` | 模型看到什么：`text` 只给文本（最省）、`audio` 给音频、`video` 再给画面；选 `video` 时画面从哪来见[模型路由「媒体开关」](model-routing.md#媒体开关---llm-media)。能听音频、看画面的只有出厂的 Gemini API 与 agy 的 Gemini，见[模型选择](models.md#各后端的使用体验) |
| `--llm-retrieval` | `local` | 联网检索：`local` 由 FineSub 自己调用在线搜索服务（Exa 等，见[环境变量与 API Key「最小可用组合」](env.md#最小可用组合)）、把结果交给模型，`none` 完全不查，`native` 让模型自己搜（需要该模型支持） |
| `--llm-fast` | `auto` | 快速模式：短素材整段放进一个窗口处理。`off` 强制按常规窗口切分。和 `agy-fast` / `fast` 预设的「快速粗纠」是两回事 |
| `--llm-proofread` | `auto` | 成品出来后另开一次校对，只改文字、不动分行和时间。`auto` 看档位：`quality` 校对，不论纠错走没走[三个 pass](diarization.md#三个-pass)，其余档不校对；中文素材只纠错的任务 `auto` 不校对（见[中文素材「LLM 阶段做什么」](chinese.md#llm-阶段做什么)）。`on` 哪一档都校对，`off` 不校对 |
| `--llm-proofread-inline` | `auto` | 每个纠错窗口交卷前，让同一个模型在同一段对话里把这一窗再校对一遍，只改文字；改不成就保留原来那份，不会让窗口失败。`auto` 只在 `intermediate` 做（它的 `auto` 不跑成品后的校对），`quality` 不做；`on` 只对一步纠错的 `quality` / `intermediate` 生效（含[一步纠错兼认人](diarization.md#一步纠错兼认人)），三个 pass、中文只纠错的任务与 `efficiency` 不做（`on` 时提示一行，`auto` 不提示）。和 `--llm-proofread` 互不影响，可以同开；每窗多一到两次调用 |
| `--llm-continuity` | `serial` | `parallel` 同时发多个纠错窗口，快，但放弃窗口之间的上下文衔接。并发数由 `--llm-parallel-windows` 定（默认 1） |
| `--llm-output-scale` | `1.0` | 调整对「本窗将输出多少内容」的估计：取值大于 1 会规划更小的窗口。当模型频繁因输出长度上限被截断时才需要调整 |
| `--max-retries-per-window` | `3` | 一个窗口内格式修复的重试次数（校对每窗同样；任务后的知识更新回答读不了时，重问次数也用它） |
| `--max-replacements-per-window` | `2` | 上述重试用尽后，为整窗更换全新会话重试的次数 |
| `--extra-info-file` | — | 与 `--extra-info` 相同，提供背景信息，但内容从文件读取——适合较长的设定或术语表。写进去的要求会被照做；只对这一次的翻译风格要求写进 `--extra-style`，见[知识库](knowledge.md#风格条目style记住一套翻译习惯) |
| `--postprocess-profile` | `0` | 成品 SRT 的后处理：`0` 繁转简 + 重叠修复 + 时长下限 + 标点；`1` 只做时长；`2` 只做标点；`3` 只做繁转简；`4` 只修重叠；`-1` 只重新渲染一遍、语义不变。改了它只删成品 `.srt`、保留 `-translated.srt`，不重做纠错，见[运行产物「重跑某一阶段」](outputs.md#重跑某一阶段需删除的产物)的「只重做最后的后处理」一行 |
| `--task-summary` | — | 给任务后的知识更新的一段说明（`--knowledge update` 时用）。要这套风格的字幕更长或更短，在这里写明，它才会改风格的[字幕长度档位](knowledge.md#字幕长度档位)；不写就不改 |
| `--knowledge-root` | 共用那一份 | 让这次运行读写另一个知识库目录（试验、或给不同客户分库） |
| `[llm] api_timeout_overhead_seconds` / `api_timeout_floor_tokens_per_second` / `api_soft_timeout_fraction`（`config.toml`） | `180` / `25` / `0.5` | API 调用等多久：硬超时 = 第一个 + 这次允许的输出上限 ÷ 第二个（秒），到了断开重试；软超时 = 硬超时 × 第三个，按**静默**算——这么久一个字节都没收到，就打一条 `llm-api-soft-timeout` 警告并另发一个备份请求，原请求不断，谁先答用谁。慢模型上的长回答在硬超时前还没写完、被断开重试时，调小第二个；想早点发备份就调小第三个，备份发得太勤（模型其实只是在长时间思考）就调大它 |
| `[llm] local_agent_keep_episodes` / `local_agent_exit_on_accept`（`config.toml`） | `false` / `auto` | 只在用本机 agent 时有用：排查时留下每次调用的现场、交稿被接受后提前结束 agent。什么时候改、代价是什么，见[本机 Agent 后端「磁盘上留下的内容」](agent.md#9-磁盘上留下的内容)与[「进阶：执行策略」](agent.md#5-进阶执行策略-execution_policy) |

⚠ `--max-retries-per-window` 与 `--max-replacements-per-window` 都只针对**输出格式不符合要求**的情况进行重试，与「模型无法完成回答」是两回事：当单句
过长无法完成时，程序会将窗口对半拆分后重试，**最多拆一次**；拆分后仍失败则停止该任务并指明
对应窗口——不再为同一失败继续消耗配额。该上限不可配置。
⚠ 两档重试想每次都用同一个值，写进 `config.toml`（命令行指定值优先）：

```toml
[llm]
max_retries_per_window = 2
max_replacements_per_window = 3
```

⚠ `--llm-media` / `--llm-retrieval` / `--llm-difficulty` / `--llm-continuity` / `--knowledge` 想每次都用同一套值，写在预设的
`[llm.presets.<名>.defaults]` 里（命令行指定值优先），写法见 [`model-routing.md`](model-routing.md)「预设带开关默认值」。
`--llm-difficulty efficiency` 不改媒体与检索开关，见 [`model-routing.md`](model-routing.md#difficulty-是定位)。
很长的输入只想粗纠一遍、找亮点：`--llm-preset agy-fast`（有 agy）或 `--llm-preset fast`（只用 API），见 [`model-routing.md`](model-routing.md)「快速粗纠」。

风格与知识库怎么用(`--style` / `--style-mode` / `--knowledge` / `--refined-srt`)是另一件
事，见 [`knowledge.md`](knowledge.md)。

## 输入与下载

- `--no-download-video`:URL 输入时**仅下载音频**。默认会一并下载视频，因为生成的字幕通常需要
  压制回视频，重新解析 URL 的成本更高；仅需字幕或磁盘空间有限时使用该选项。
- ⚠ 和 `--llm-media video`（让模型看画面）一起用、又没用 `--llm-video <视频文件>` 另给画面时，下载完音频就报错停下。
  想让模型看画面，要么别加 `--no-download-video`，要么用 `--llm-video` 给一个本地视频（给了它就不另下视频）。

## 运行环境

这两项只能写在 `config.toml` 的 `[runtime]` 里，说明见[资源与大文件](resources.md)「运行环境的核对与重建」。

| 设置 | 默认 | 什么时候动它 |
| --- | --- | --- |
| `integrity_check` | `auto` | 什么时候核对运行环境：`auto` 新装或升级后、成功跑完一次之前每次都查，失败后也查；`on` 每次转写前都查；`off` 不自动查 |
| `auto_repair` | `true` | 核对不上时是否自动按锁重建。自己往运行环境里装了东西、不想被去掉时设 `false`（含升级带来的重建；第一次安装不受影响）。需要额外的包更好的办法是 `FINESUB_EXTRA_PYTHONPATH` |

## 跑多久

以下为参考量级：**23.5 分钟**的素材，在一台 RTX 5070 Ti 上运行至 `raw-srt`（不含 LLM）约
**101 秒**，即约 14 倍实时；其中人声分离约占 41%，语音检测与识别约占 59%，其余步骤合计不足
0.5 秒。（出处：[`../bench-baselines.md`](../bench-baselines.md) 第四节 4.1，默认模型、
`--gpu-tier high`,n=1。）

该数据仅为单台机器上的一次测量，用于判断数量级，并非性能承诺。以下因素会显著改变耗时：

- **是否使用显卡**：全程 CPU 运行大约慢一个数量级（识别约为十倍差距），长音频不推荐。用哪张卡由 `--device`
  决定（`cpu` / `cuda` / `xpu`，小写），见 [`resources.md`](resources.md)「哪些显卡能用」。
- **更换更大的模型**：`large-v3` 级模型慢 2 倍以上，此时建议添加 `--asr-decode-batch 8`。
- **LLM 阶段**：耗时主要取决于「窗口数 × 每窗的调用次数」（默认档每窗约 2 次：模型可能先要一批资料、回填后再交稿，另加重试；见[环境变量与 API Key「Gemini」](env.md#geminigoogle-ai-studio)），与素材长度成正比；免费配额还会
  限制速率，因此该阶段通常比整条语音链路更耗时。实际调用次数与重试情况，可查看运行结束后的
  `<名字>.llm-artifacts/task-report.md`。

如需定位本次运行的耗时环节，可查看 `-metadata.json` 中的各阶段耗时，或在运行时添加
`--log-level verbose`。
