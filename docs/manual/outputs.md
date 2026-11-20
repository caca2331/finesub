# 运行产物：文件说明、阶段与标注

一次运行结束后，输出目录中并不只有一个 SRT 文件。本页回答三个问题：**哪个文件是最终产物**、
**`--stage` 各取值停在哪个阶段**、**低置信度的行在哪里查看**。

## 一次运行留下什么

产物都在一个任务目录里，位置看怎么跑：

- 安装版跑单个输入：`%LOCALAPPDATA%\FineSub\tasks\<名字>-<时间>-<后缀>\`（跑完会打印），规则见
  [`../../cli/README.md`](../../cli/README.md)「成品放在哪」；
- 批量运行（几个输入或 `--manifest`）与源码运行：当前目录下的 `out/<名字>/`；
- 传了 `-o`：你指定的位置。

以名字 `input` 为例：

| 文件 | 是什么 | 一般怎么处理 |
| --- | --- | --- |
| `input.srt` | **最终成品**：已完成纠错、翻译与后处理 | 直接使用（FineSub 只出字幕文件，不出带字幕的视频，见表后） |
| `input-raw.srt` | 未经纠错的原文字幕（ASR 结果）；开了 `--diarize` 时默认是[拆开两个人的行](diarization.md#拆开两个人的行)之后的版本 | 用于对照；如需查看 LLM 的改动，请保留该文件 |
| `input-corrected.srt` | 纠错后的**原文**，未翻译 | 需要确认「听错的是哪个词」时查看 |
| `input-translated.srt` | 已完成纠错翻译、**尚未后处理**的中文 | 通常无需关注；后处理的具体内容见下文 |
| `input-annotated.csv` | 逐行标注：置信度、改动说明 | **人工核对时以此为准**，见下一节 |
| `input-pass2-translated.srt` · `input-pass2-corrected.srt` · `input-pass2-annotated.csv` · `input-pass2.srt` | 纠错走[三个 pass](diarization.md#三个-pass)、开着校对时，校对之前的版本 | 想看校对改了什么时对照；可以删 |
| `input-vocal.ogg` | 分离出的人声轨（`--no-separate` 时为源音频直接转换的同规格音轨） | 需要试听分离效果时保留；重跑最耗资源的阶段时使用 |
| `input-vad.json` · `input-vad-energy.npz` · `input-aligned.json` · `input-stable.json` | 中间数据 | 重跑时用来跳过已完成的阶段；重做 stable 这一步时也要用到 `-vad.json` 与 `-vad-energy.npz`，别单删 |
| `input-diar.json` | 说话人时间线（仅跑过说话人分离）。元数据 `hints` 里的 `samples` 给每条轨一段「这个人单独说话、而且最像他自己」的时间区间——**想知道 s1 是谁，照着听十秒最快**（没有连续单独说满 2 秒的轨、裁判新建的轨都没有样本）；`hints` 里其余几条只是参考数字，不改任何标签。同一份也写在 `-diar-rows.json` 顶层的 `speaker` 里 | ⚠ **别当中间数据删**：同一解码输入、同一执行路径的复跑可以相同，但重新编码、位深转换或硬件数值差异可能改变分轨，重跑也未必更准。它只读人声音轨，标签在下一行的文件里 |
| `input-diar-rows.json` | 时间线对 `-stable.json` 每一行说了什么（仅跑过说话人分离）：每行挂哪条轨、[逐行置信标](diarization.md#逐行置信标)、[拆开的行](diarization.md#拆开两个人的行)切在哪、用不用拆过的版本；格式见[产出了什么](diarization.md#产出了什么) | ⚠ **别删**：`-raw.srt` 与 LLM 产物数的是它定下的行；删了会按当时的证据与开关重写。`-stable.json` 写成后不再改，说话人分离的结果只在这里 |
| `input-diar-evidence.npz` | 说话人分离算出来的声学证据（`nemotron3` 与 `diarizen`；`sortformer` 不产生），每小时几 MB | **可以删**，没有它照样跑完：[逐行置信标](diarization.md#逐行置信标)与切点已经记在 `-diar-rows.json` 里，只有那份要重写时才再读它，那时没有它就没有置信标、也不拆。⚠ **只删它不会让它回来**：下次跑看到 `-diar.json` 还在就跳过这一步，要重算得**两份一起删**（而删 `-diar.json` 会换一个答案） |
| `input-speaker-judge.json` | [LLM 说话人裁判](diarization.md#llm-说话人裁判)的结果：每个窗口（LLM 分批处理的一段字幕）判了哪些行、挪到哪条轨、哪些判断不了、理由（仅 `--diarize judge-text` / `judge-audio`，且跑了 LLM 阶段；[一步纠错兼认人](diarization.md#一步纠错兼认人)的任务由纠错窗口写，没有理由） | ⚠ 同上，**别删**：同样的输入再问一次，答案可能不同。`-annotated.csv` 的 `spk` 列与分轨导出读它；删掉它就回到声学标签 |
| `input-speakers/` | 按说话人拆开的 **ASR 原文**（仅 `--speaker-tracks`），不是纠错、翻译后的字幕；里面有哪些文件、哪行进哪份见[转成分轨](diarization.md#转成分轨) | 拿去打轴时用；自己的校对副本别放在这个目录里（重新导出会清掉，见同一链接） |
| `input-metadata.json` | 各阶段耗时、并行数 | 排查「慢在哪一步」时查看 |
| `input.llm-artifacts/` | LLM 过程记录 | 其中 `task-report.md` 为供人阅读的汇总（API 调用次数、模型的回答是否退化：格式检查通过、实际却做坏了，比如该合并的行几乎都没合并）；`resolved-switches.json` 记这次媒体、检索、难度、连续性、知识库五个开关最后取了什么、各自来自命令行、哪个预设还是内置默认——预设像是没生效时先看它（安装版的任务记录里这五个开关只记命令行上写了的，没写的以这份为准） |

拿到成品之后：把 `input.srt` 改成和视频同名、放在同一个目录，VLC、mpv、PotPlayer 这类播放器会自动加载它，用来预览；
要把字幕压进视频，用 ffmpeg 或剪辑软件，FineSub 不做这一步。改字幕前先看下面的[自己改字幕](#自己改字幕)。

⚠ 这些文件同时也作为**续跑依据**：阶段按**文件是否存在**跳过，不校验内容。因此删除中间产物后，
下次运行将从对应阶段重新开始（见本页最后一节）。

## `--stage`：指定停止的阶段

共六个取值，按流水线顺序执行。默认值为 `raw-srt`——`--asr` 没选云端三家时，跑到这里**不调用任何 API、不消耗配额**。

| `--stage` | 跑完得到 | 需要什么 |
| --- | --- | --- |
| `vocal` | `-vocal.ogg`，仅执行人声分离 | 显卡（或 CPU，速度较慢） |
| `aligned` | `-aligned.json`，语音检测 + 识别对齐 | 同上 |
| `stable` | `-stable.json`，稳定化后的数据 | 同上 |
| `raw-srt` | `-raw.srt`，**默认**，未纠错的原文字幕 | 同上 |
| `translated-srt` | `-translated.srt` + `-corrected.srt` + `-annotated.csv` | 需配置 Gemini API 或本机 agent |
| `final-srt` | `input.srt`，**成品** | 同上 |

`translated-srt` 与 `final-srt` 之间仅差**后处理**：繁转简、重叠时间轴修复、时长下限与标点整理
(`--postprocess-profile`，见 [`tuning.md`](tuning.md))。不再调用模型改写，繁转简与标点整理只改个别字符，需要自行处理
后处理的用户可停在 `translated-srt`；其余用户直接使用 `final-srt`。

后处理不让字幕结束在音频结尾之后。极少数情况下，识别给的最后几行开始时间就晚于音频结尾：成品里它们各被压成约 1 毫秒、排在结尾之前，
播放时看不到，文字仍在 SRT 里，**目前不打任何提示**。在字幕软件里看最后几条的时长就能发现；要留住这些字，把它们并进前一条或手动挪时间。

直接从零运行 `final-srt` 时，上述文件同样会全部生成——它们是同一次纠错的副产物，并非
额外运行所得。

## `-annotated.csv`：核对低置信度的行

LLM 每输出一行字幕，都会附带一个自评置信度，以及（该行有改动时）一句说明。SRT 格式无法
承载这些信息，因此另存为一份以竖线分隔的 CSV:

```text
# type|position|duration|gap|corrected|translation|conf|char_count|spk|note
sub|1|2.9|1.2|このところよく出入りしてたから目立たないだろ|最近经常出入这里，所以不会引人注意吧|high|17.5||
sub|8|1.6|0.6|二人可愛いよねー|两个人好可爱啊|high|7|s0|「トゥガリン」按听感修正为「二人」
sub|21|1.2|0.6|あのお目付役が|那个负责监视的人？|median|8.5|s0+s1|「お目つき役」修正为「お目付役」
```

| 列 | 含义 |
| --- | --- |
| `type` | 行类型，目前总是 `sub` |
| `position` | 这一行来自原文的哪几句（`3,4` = 两句并成了一行） |
| `duration` · `gap` | 本行时长、与上一行之间的停顿（秒；与上一行重叠时为 0，首行留空） |
| `corrected` | **纠错后的原文**——和 `-raw.srt` 比就知道 ASR 听错了什么 |
| `translation` | 译文；成品 SRT 里那一行是它经过后处理（繁转简、标点整理）之后的样子 |
| `conf` | 模型对这一行的自评：`high` / `median` / `low` |
| `char_count` | 译文的加权字数（本地按译文算：拉丁字母、数字、标点和空格各算半个字；阅读时可忽略） |
| `spk` | 说话人轨号，如 `s2`、`s2?+s3`、`s3<s2`；每个记号的意思、怎么取主轨、怎么按它把字幕拆成分轨、什么时候整列为空，见[读 `spk` 列](diarization.md#读-spk-列) |
| `note` | 改了什么、为什么。**没改动的行这里是空的** |

怎么用：

1. 用表格软件打开，分隔符选 `|`（第一行以 `#` 开头，是列名注释）。
2. **按 `conf` 筛选 `median` 与 `low`**——这些是模型本身不确定的行，可作为人工抽查的对象。
   举一次运行作例子：413 行里 405 行为 `high`、8 行为 `median`、0 行为 `low`；别的素材上比例会不同。
   `high` 只是模型自认有把握，不保证内容对，要紧的素材仍要通读。
3. 查看 `note` 非空的行，即 LLM 修改过的位置；将 `corrected` 与 `-raw.srt` 对照即可确认。

⚠ **`conf` 是 LLM 对自己这一行的自评，不是语音识别的置信度**，两者没有关系。
⚠ CSV 的行与 `-corrected.srt` 的字幕**一一对应、顺序一致**，可直接按行号对应；翻译任务上与成品 SRT 也一一对应。中文只纠错的任务里，整行只有垫词的行留在 CSV（`translation` 格空着）但不进成品，所以 CSV 的第 N 行先对到 `-corrected.srt` 的第 N 条，再按那一条的时间到成品里找。
## 自己改字幕

- **改成品 `<名字>.srt` 的副本，别改原件**。重跑同一个源会接着这个任务继续，成品已经在就不重做，不会覆盖你的改动；
  但按[下一节](#重跑某一阶段需删除的产物)重做后处理或字幕时间轴时，要删掉成品重新生成，原件上的改动就没了。
  副本也别放进 `<名字>-speakers/`：重新导出分轨会清掉那里的文件。`-annotated.csv` 与 `-corrected.srt` 不会跟着你的副本变。
- **改 `-raw.srt` 影响不了纠错**：纠错读的是 `-stable.json`，不读 `-raw.srt`。想让纠错用对某个名字，写进 `--extra-info` 或知识库。
- **改好的副本可以回喂知识库**，让它学到这一集机器听错了什么、名字该怎么写：
  `finesub knowledge-update <任务目录>\<名字>.srt --refined-srt <改好的副本> --execute`，见
  [知识库「把精修好的字幕回喂」](knowledge.md#4-把精修好的字幕回喂)。

## 重跑某一阶段：需删除的产物

阶段**只看文件在不在**。要重跑某一步，就把它的产物和**它下游的全部产物**删掉，然后原样再跑
一次命令。产物从上游到下游的顺序（表里「及之后的全部」按它数）：

`-vocal.ogg` → `-vad.json`、`-vad-energy.npz` → `-aligned.json` → `-stable.json` → `-diar-rows.json`（开了说话人分离时）→
`-raw.srt` → `-speaker-judge.json`（开了说话人裁判时）→ `-translated.srt`、`-corrected.srt`、`-annotated.csv`、四件 `-pass2-*`、`<名字>.llm-artifacts/` → 成品 `<名字>.srt`

`-speaker-judge.json` 单独复用：重做纠错时留着它，就沿用原来的裁判结果、不再为它花额度；要重判才删它（见下表「只重做说话人裁判」）。
不在这条线上的：`-diar.json` 与 `-diar-evidence.npz` 只读人声轨，只有重做人声分离或说话人分离时才删（见表）；`-metadata.json` 不用管；云端转写留下的 `<名字>-cloud/` 别删（删了要再花一次额度重转）。

| 想重做 | 删掉 |
| --- | --- |
| 人声分离（换 `--separator-rate`、开关 `--separate` 之类） | `-vocal.ogg`（或 `-vocal.flac`）及之后的全部 |
| 语音检测 / 识别 / 分句（换 `--model`、`--split-length-scale`…） | `-vad.json` + `-vad-energy.npz` + `-aligned.json` 及之后的全部 |
| 只重做稳定化（换 `--asr-stabilize-profile`） | `-stable.json` + `-raw.srt` 及之后的（`-diar-rows.json` 对不上新的行，开着 `--diarize` 时自动重写） |
| 只重做说话人分离 | `-diar.json`（`-stable.json` 不用删；`-diar-rows.json` 跟着重写，标签与[拆开的行](diarization.md#拆开两个人的行)重新决定）。⚠ 拆开的行可能跟着变，所以 `-raw.srt` 与 LLM 产物（`-translated.srt`、`-corrected.srt`、`-annotated.csv`、成品 `.srt`、`-speaker-judge.json` 与 `<名字>.llm-artifacts/` 目录）也要一起删、重新纠错，会花额度（行变了，目录里已完成的窗口本来也接不上）；只留着 LLM 产物的话，它们数的还是旧的行。`--no-diarize-split` 的任务不拆，可以只补裁判：开着裁判档再跑一次就只改写 `spk` 列。同一输入与执行路径可复现；输入或数值路径变化可能改标签，重跑不保证更准 |
| 只重做 LLM 纠错翻译 | `-translated.srt` + `-corrected.srt` + `-annotated.csv` + 成品 `.srt` + 有的话四件 `-pass2-*` + `<名字>.llm-artifacts/`（不删它会从没做完的窗口接着跑，已做完的窗口照用，见表后）；`-speaker-judge.json` 留着，裁判结果照用 |
| 只重做说话人裁判 | `-speaker-judge.json`，再带着 `--diarize judge-text`（或 `judge-audio`）原样跑（纠错不重跑，只改写 `-annotated.csv` 的 `spk` 列与分轨导出）。⚠ 同样是**另一份**答案。[一步纠错兼认人](diarization.md#一步纠错兼认人)的任务这样重做不了：说话人是纠错窗口顺带判的，要连纠错一起重做（`--no-resume`） |
| 只重做最后的后处理（换 `--postprocess-profile`） | 仅删除成品 `.srt`，保留 `-translated.srt` |
| 只重做字幕时间轴（改 `[postprocess]` 的两个值） | 原文导出：删 `-raw.srt`（从 `-stable.json` 重新生成）；翻译成品：仅删除成品 `.srt`（从 `-translated.srt` 重新后处理，译文保留） |

⚠ **不删除产物而直接以新参数重跑，已完成的阶段不会应用新参数**——这是有意设计：重跑任务表示
「从上次进度继续」，而非「按新参数重新执行」。想留着原结果、另试一组参数：源码运行与批量换一个 `--name`（产物进新的 `out/<名字>/`）；安装版跑单个输入时同一个源总归到同一个任务（换 `--name` 不另开），加 `-o <新目录>` 就在新位置从头跑，原任务目录里的成品不受影响。
⚠ LLM 阶段支持更细粒度的续跑：已完成窗口的记录保存在 `<名字>.llm-artifacts/`，删了 SRT 而留着它，
会从**未完成的窗口**继续，不会重复消耗已使用的配额。

## 跑完想腾空间

按之后还要做什么决定留哪些；每个文件是什么、说话人与云端转写那几样为什么别删，见[一次运行留下什么](#一次运行留下什么)的表。

| 之后还要做什么 | 留着 | 可以删 |
| --- | --- | --- |
| 只要字幕 | 成品 `<名字>.srt`（想对照原文再留 `-raw.srt`） | 其余的 |
| 回喂精修字幕（[`knowledge-update`](knowledge.md#4-把精修好的字幕回喂)） | 成品 `.srt`、`-stable.json`、`-annotated.csv` 与 `<名字>.llm-artifacts/`，留在原位置：它按成品的位置找另外几样 | `-vocal.ogg`、`-vad.json`、`-vad-energy.npz`、`-aligned.json`、四件 `-pass2-*` |
| 续跑，或调参后重跑某一阶段 | 全部：阶段按文件在不在跳过，删掉的那一步及下游会重做（见[重跑某一阶段](#重跑某一阶段需删除的产物)） | 四件 `-pass2-*`、`-diar-evidence.npz`（见表里那一行） |

卸载 FineSub 时删什么、留什么，见[资源与大文件「清理」](resources.md#清理)。
