# 运行产物：文件说明、阶段与标注

一次运行结束后，输出目录中并不只有一个 SRT 文件。本页回答三个问题：**哪个文件是最终产物**、
**`--stage` 各取值停在哪个阶段**、**低置信度的行在哪里查看**。

产物的完整字段契约（面向开发者）参见 [`../../README_DEV.md`](../../README_DEV.md);
本页仅说明实际会用到的部分。

## 一次运行留下什么

不传 `-o` 时全部落在 `out/<名字>/`；传 `-o` 时就在你指定的位置，产物一个个出现在那里。
托管 CLI 另有一套任务目录(`tasks/<任务名>-<时间>-<后缀>/`)，规则见
[`../../cli/README.md`](../../cli/README.md)「成品放在哪」。以名字 `input` 为例：

| 文件 | 是什么 | 一般怎么处理 |
| --- | --- | --- |
| `input.srt` | **最终成品**：已完成纠错、翻译与后处理 | 直接使用 |
| `input-raw.srt` | 未经纠错的原文字幕（ASR 结果）；开了 `--diarize` 时默认是[拆开两个人的行](diarization.md#拆开两个人的行)之后的版本 | 用于对照；如需查看 LLM 的改动，请保留该文件 |
| `input-corrected.srt` | 纠错后的**原文**，未翻译 | 需要确认「听错的是哪个词」时查看 |
| `input-translated.srt` | 已完成纠错翻译、**尚未后处理**的中文 | 通常无需关注；后处理的具体内容见下文 |
| `input-annotated.csv` | 逐行标注：置信度、改动说明 | **人工核对时以此为准**，见下一节 |
| `input-pass2-translated.srt` · `input-pass2-corrected.srt` · `input-pass2-annotated.csv` · `input-pass2.srt` | 纠错走[三个 pass](diarization.md#三个-pass)、开着校对时，校对之前的版本 | 想看校对改了什么时对照；可以删 |
| `input-vocal.ogg` | 分离出的人声轨（`--no-separate` 时为源音频直接转换的同规格音轨） | 需要试听分离效果时保留；重跑最耗资源的阶段时使用 |
| `input-vad.json` · `input-vad-energy.npz` · `input-aligned.json` · `input-stable.json` | 中间数据 | 重跑时用来跳过已完成的阶段；重做 stable 这一步时，坍缩行修法也要读 `-vad.json` 与 `-vad-energy.npz`（缺了退回只拉长，并提示） |
| `input-diar.json` | 说话人时间线（仅跑过说话人分离）；元数据里的 `hints` 是几条声学提示，**都只是提示、不改任何标签**：`co_activity`，每对轨同时发声的比例（按较短那条轨算；同时响得多常常是两个人，但主播回放自己的录音、回声，以及模型把同一个人切成两条轨时，这两条轨也会一起响，**别据此认定是两个人**）；以及 `fourth_track_share`，第四忙的那条轨占全部说话时间的比例（素材里超过 4 个人时，多出来的人的话会记在别人的编号下；这个数高说明四条轨都有人在用，但它分不开「正好 4 人」与「5 人以上」，只作参考）。用默认的 `nemotron3` 或 `diarizen` 时还多两条（`sortformer` 没有）：`single_speaker_log_bayes`（聚类模型给的证据比：大 = 模型更信不止一个人，接近 0 = 「只有一个人」没被排除；⚠ **它还没标定**：它该抓的是单人素材被**过拆**，而真正过拆了的单人素材只量过两份——按嵌入数除过的 `single_speaker_log_bayes_per_embedding` 读数是 0.326 与 1.385（在 `diarizen` 上量的；默认的 `nemotron3` 只量过一份过拆的单人素材，1.22），低于 39 份多人素材里的 38 份，方向对，但两个点不是标定（[findings「4.4 过拆与单人证据」](../diarization-findings.md#44-过拆与单人证据)）；⚠ 它是**总量**、随音频变长而变大，**两个文件之间别直接比**，要横向比看旁边除好的 `single_speaker_log_bayes_per_embedding`）与 `samples`（每条轨一段「这个人单独说话、而且最像他自己」的时间区间——**想知道 s1 是谁，照着听十秒最快**；没有连续单独说满 2 秒的轨、裁判新建的轨都没有样本）。同一份也写在 `-diar-rows.json` 顶层的 `speaker` 里 | ⚠ **别当中间数据删**：同一解码输入、同一执行路径的复跑可以相同，但重新编码、位深转换或硬件数值差异可能改变分轨，重跑也未必更准。它只读人声音轨，标签在下一行的文件里 |
| `input-diar-rows.json` | 时间线对 `-stable.json` 每一行说了什么（仅跑过说话人分离）：每行挂哪条轨、[逐行置信标](diarization.md#逐行置信标)、[拆开的行](diarization.md#拆开两个人的行)切在哪、用不用拆过的版本；格式见[产出了什么](diarization.md#产出了什么) | ⚠ **别删**：`-raw.srt` 与 LLM 产物数的是它定下的行；删了会按当时的证据与开关重写。`-stable.json` 写成后不再改，说话人分离的结果只在这里 |
| `input-diar-evidence.npz` | 聚类自己算出来的声学证据（`nemotron3` 与 `diarizen`；`sortformer` 不产生这些）：每段到每个说话人中心的距离、中心本身、20 ms 的精细活动 | **可以删**，没有它照样跑完：[逐行置信标](diarization.md#逐行置信标)与切点已经记在 `-diar-rows.json` 里，只有那份要重写时才再读它，那时没有它就没有置信标、也不拆。默认的 `nemotron3` 还在里面存了置信标要用的槽位活动与逐帧概率（每小时约 5–6 MB）。⚠ **只删它不会让它回来**：下次跑看到 `-diar.json` 还在就跳过这一步，要重算得**两份一起删**（而删 `-diar.json` 会换一个答案） |
| `input-speaker-judge.json` | [LLM 说话人裁判](diarization.md#llm-说话人裁判)的结果：每个窗口判了哪些行、挪到哪条轨、哪些判断不了、理由（仅 `--diarize judge-text` / `judge-audio`，且跑了 LLM 阶段） | ⚠ 同上，**别删**：同样的输入再问一次，答案可能不同。`-annotated.csv` 的 `spk` 列与分轨导出读它；删掉它就回到声学标签 |
| `input-speakers/` | 按说话人拆开的 **ASR 原文**（仅 `--speaker-tracks`），不是纠错、翻译后的字幕；里面有哪些文件、哪行进哪份见[转成分轨](diarization.md#转成分轨) | 拿去打轴时用；自己的校对副本别放在这个目录里（重新导出会清掉，见同一链接） |
| `input-metadata.json` | 各阶段耗时、并行数 | 排查「慢在哪一步」时查看 |
| `input.llm-artifacts/` | LLM 过程记录 | 其中 `task-report.md` 为供人阅读的汇总（API 调用次数、是否出现退化） |

⚠ 这些文件同时也作为**续跑依据**：阶段按**文件是否存在**跳过，不校验内容。因此删除中间产物后，
下次运行将从对应阶段重新开始（见本页最后一节）。

需要比较说话人候选配置时，开发工具提供高上下文 Sortformer 与独立的 community-1 入口，生成字幕副本供核对。
使用方式见[探索工具说明](../../tools/diarization_explore/README.md)，
效果与限制见[说话人实验记录](../diarization-findings.md)。主流水线默认模型不因此改变。

## `--stage`：指定停止的阶段

共六个取值，按流水线顺序执行。默认值为 `raw-srt`——该阶段**不调用任何 API、不消耗配额**。

| `--stage` | 跑完得到 | 需要什么 |
| --- | --- | --- |
| `vocal` | `-vocal.ogg`，仅执行人声分离 | 显卡（或 CPU，速度较慢） |
| `aligned` | `-aligned.json`，语音检测 + 识别对齐 | 同上 |
| `stable` | `-stable.json`，稳定化后的数据 | 同上 |
| `raw-srt` | `-raw.srt`，**默认**，未纠错的原文字幕 | 同上 |
| `translated-srt` | `-translated.srt` + `-corrected.srt` + `-annotated.csv` | 需配置 Gemini API 或本机 agent |
| `final-srt` | `input.srt`，**成品** | 同上 |

`translated-srt` 与 `final-srt` 之间仅差**后处理**：繁转简、重叠时间轴修复、时长下限与标点整理
(`--postprocess-profile`，见 [`tuning.md`](tuning.md))。文字内容不再变化，需要自行处理
后处理的用户可停在 `translated-srt`；其余用户直接使用 `final-srt`。

时长后处理在已知音频长度时会限制字幕终点，包括输入已有的越界和闪轴补齐。
正常末条仍延长 0.3 秒；连续越界的尾条各被压成约 1 毫秒、依次向前排在音频末尾之内，必要时收短前邻条，文本都保留（这几条实际上看不到，
文本仍在 SRT 里）。
这属于时间格式修复，不能当作真实发声位置；没有音频长度的旧产物无法自动检查这一项。

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
| `translation` | 译文，和最终 SRT 里那一行一致 |
| `conf` | 模型对这一行的自评：`high` / `median` / `low` |
| `char_count` | 译文的加权字数（本地按译文算：拉丁字母、数字、标点和空格各算半个字；阅读时可忽略） |
| `spk` | 说话人轨号，如 `s2`、`s2?+s3`、`s3<s2`；每个记号的意思、怎么取主轨、怎么按它把字幕拆成分轨，见[读 `spk` 列](diarization.md#读-spk-列)。**没开 `--diarize` 时整列为空** |
| `note` | 改了什么、为什么。**没改动的行这里是空的** |

怎么用：

1. 用表格软件打开，分隔符选 `|`（第一行以 `#` 开头，是列名注释）。
2. **按 `conf` 筛选 `median` 与 `low`**——这些是模型本身不确定的行，可作为人工抽查的对象。
   实际占比很低：一次 413 行的真实运行中，405 行为 `high`、8 行为 `median`、0 行为 `low`。
3. 查看 `note` 非空的行，即 LLM 修改过的位置；将 `corrected` 与 `-raw.srt` 对照即可确认。

⚠ **`conf` 是 LLM 对自己这一行的自评，不是语音识别的置信度**，两者没有关系。
⚠ CSV 的行与最终 SRT 的字幕**一一对应、顺序一致**，可直接按行号对应。
⚠ 如需将人工精修结果回馈知识库（使其学习你的修改方式），请参见 [`knowledge.md`](knowledge.md)
的 `--refined-srt` 说明。

## 重跑某一阶段：需删除的产物

阶段**只看文件在不在**。要重跑某一步，就把它的产物和**它下游的全部产物**删掉，然后原样再跑
一次命令：

| 想重做 | 删掉 |
| --- | --- |
| 人声分离（换 `--separator-rate`、开关 `--separate` 之类） | `-vocal.ogg`（或 `-vocal.flac`）及之后的全部 |
| 语音检测 / 识别 / 分句（换 `--model`、`--split-length-scale`…） | `-vad.json` + `-vad-energy.npz` + `-aligned.json` 及之后的全部 |
| 只重做稳定化（换 `--asr-stabilize-profile`） | `-stable.json` + `-raw.srt` 及之后的（`-diar-rows.json` 对不上新的行，开着 `--diarize` 时自动重写） |
| 只重做说话人分离 | `-diar.json`（`-stable.json` 不用删；`-diar-rows.json` 跟着重写，标签与[拆开的行](diarization.md#拆开两个人的行)重新决定）。⚠ 拆开的行可能跟着变，所以 `-raw.srt` 与 LLM 产物（`-translated.srt`、`-corrected.srt`、`-annotated.csv`、成品 `.srt`、`-speaker-judge.json` 与 `<名字>.llm-artifacts/` 目录）也要一起删、重新纠错，会花额度（行变了，目录里已完成的窗口本来也接不上）；只留着 LLM 产物的话，它们数的还是旧的行。`--no-diarize-split` 的任务不拆，可以只补裁判：开着裁判档再跑一次就只改写 `spk` 列。同一输入与执行路径可复现；输入或数值路径变化可能改标签，重跑不保证更准 |
| 只重做 LLM 纠错翻译 | `-translated.srt` + `-corrected.srt` + `-annotated.csv` + 成品 `.srt` |
| 只重做说话人裁判 | `-speaker-judge.json`，再带着 `--diarize judge-text`（或 `judge-audio`）原样跑（纠错不重跑，只改写 `-annotated.csv` 的 `spk` 列与分轨导出）。⚠ 同样是**另一份**答案 |
| 只重做最后的后处理（换 `--postprocess-profile`） | 仅删除成品 `.srt`，保留 `-translated.srt` |
| 只重做字幕时间轴（改 `[postprocess]` 的两个值） | 原文导出：删 `-raw.srt`（从 `-stable.json` 重新生成）；翻译成品：仅删除成品 `.srt`（从 `-translated.srt` 重新后处理，译文保留） |

⚠ **不删除产物而直接以新参数重跑，已完成的阶段不会应用新参数**——这是有意设计：重跑任务表示
「从上次进度继续」，而非「按新参数重新执行」。如需全新的参数组合，请新建任务（更换 `--name` 或 `-o`）。
⚠ LLM 阶段支持更细粒度的续跑：已完成窗口的记录保存在 `input.llm-artifacts/`，仅删除上述 SRT
文件会从**未完成的窗口**继续，不会重复消耗已使用的配额。如需整体重做，请连同
`input.llm-artifacts/` 一并删除。

跑完之后哪些是记录、哪些能删，见
[`resources.md`](resources.md) 的「清理」一节。
