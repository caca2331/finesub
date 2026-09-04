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
| `input-raw.srt` | 未经纠错的原文字幕（ASR 结果） | 用于对照；如需查看 LLM 的改动，请保留该文件 |
| `input-corrected.srt` | 纠错后的**原文**，未翻译 | 需要确认「听错的是哪个词」时查看 |
| `input-translated.srt` | 已完成纠错翻译、**尚未后处理**的中文 | 通常无需关注；后处理的具体内容见下文 |
| `input-annotated.csv` | 逐行标注：置信度、改动说明 | **人工核对时以此为准**，见下一节 |
| `input-vocal.ogg` | 分离出的人声轨（`--no-separate` 时为源音频直接转换的同规格音轨） | 需要试听分离效果时保留；重跑最耗资源的阶段时使用 |
| `input-vad.json` · `input-vad-energy.npz` · `input-aligned.json` · `input-stable.json` | 中间数据 | 重跑时用来跳过已完成的阶段 |
| `input-metadata.json` | 各阶段耗时、并行数 | 排查「慢在哪一步」时查看 |
| `input.llm-artifacts/` | LLM 过程记录 | 其中 `task-report.md` 为供人阅读的汇总（API 调用次数、是否出现退化） |

⚠ 这些文件同时也作为**续跑依据**：阶段按**文件是否存在**跳过，不校验内容。因此删除中间产物后，
下次运行将从对应阶段重新开始（见本页最后一节）。

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

直接从零运行 `final-srt` 时，上述文件同样会全部生成——它们是同一次纠错的副产物，并非
额外运行所得。

## `-annotated.csv`：核对低置信度的行

LLM 每输出一行字幕，都会附带一个自评置信度，以及（该行有改动时）一句说明。SRT 格式无法
承载这些信息，因此另存为一份以竖线分隔的 CSV:

```text
# type|position|duration|gap|corrected|translation|conf|char_count|note
sub|1|2.9|1.2|このところよく出入りしてたから目立たないだろ|最近经常出入这里，所以不会引人注意吧|high|17.5|
sub|8|1.6|0.6|二人可愛いよねー|两个人好可爱啊|high|7|「トゥガリン」按听感修正为「二人」
sub|21|1.2|0.6|あのお目付役が|那个负责监视的人？|median|8.5|「お目つき役」修正为「お目付役」
```

| 列 | 含义 |
| --- | --- |
| `type` | 行类型，正常字幕是 `sub` |
| `position` | 这一行来自原文的哪几句（`3,4` = 两句并成了一行） |
| `duration` · `gap` | 本行时长、与下一行之间的间隔（秒） |
| `corrected` | **纠错后的原文**——和 `-raw.srt` 比就知道 ASR 听错了什么 |
| `translation` | 译文，和最终 SRT 里那一行一致 |
| `conf` | 模型对这一行的自评：`high` / `median` / `low` |
| `char_count` | 译文的加权字数（供模型自检长度，阅读时可忽略） |
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
| 只重做稳定化（换 `--asr-stabilize-profile`） | `-stable.json` + `-raw.srt` 及之后的 |
| 只重做 LLM 纠错翻译 | `-translated.srt` + `-corrected.srt` + `-annotated.csv` + 成品 `.srt` |
| 只重做最后的后处理（换 `--postprocess-profile`） | 仅删除成品 `.srt`，保留 `-translated.srt` |

⚠ **不删除产物而直接以新参数重跑，已完成的阶段不会应用新参数**——这是有意设计：重跑任务表示
「从上次进度继续」，而非「按新参数重新执行」。如需全新的参数组合，请新建任务（更换 `--name` 或 `-o`）。
⚠ LLM 阶段支持更细粒度的续跑：已完成窗口的记录保存在 `input.llm-artifacts/`，仅删除上述 SRT
文件会从**未完成的窗口**继续，不会重复消耗已使用的配额。如需整体重做，请连同
`input.llm-artifacts/` 一并删除。

跑完之后哪些是记录、哪些能删，见
[`resources.md`](resources.md) 的「清理」一节。
