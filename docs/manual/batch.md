# 一次跑多个输入

`finesub` 只有一个命令。给它单个输入就是普通运行，给它多个输入（或一份 manifest）就是批量
处理——没有独立的批量入口。

```powershell
# 多个输入
finesub a.wav b.mp4 --stage final-srt --language ja

# 一份 manifest，每行一项
finesub --manifest tasks.jsonl --knowledge update
```

（源码安装把 `finesub` 换成 `python -m finesub.pipeline`，参数一样。）

多输入时，流水线按 **下载 ×2 → ASR ×1 → LLM ×2** 的并发度推进。ASR 恒为 1，因为每个文件需要
独占整块显卡；LLM 并发由 `--max-parallel-tasks` 控制（默认 2，取值说明见
[`agent.md`](agent.md)「同时运行的 agent 数量」）。后一个任务的下载与 ASR 会与前一个任务的 LLM
阶段重叠执行，因此整批的墙钟时间远小于逐个串行处理。

## manifest 长什么样

JSONL，一行一项，`source` 必填：

```json
{"source": "https://www.bilibili.com/video/BVxxxxxxxxxx"}
{"source": "D:/media/talk.mp4", "language": "ja", "stage": "final-srt"}
{"source": "D:/media/short.wav", "priority": 5}
```

**命令行上的任何一个管线选项都可以写进行里**，把 `--` 去掉、连字符换下划线即可：
`--llm-difficulty quality` → `"llm_difficulty": "quality"`,`--qwen-verify on` → `"qwen_verify": "on"`。
命令行上给的值是所有行的默认值，行里写的覆盖它。

行里另有三个键不是管线选项：

| 键 | 作用 |
| --- | --- |
| `source` | 媒体 URL 或本地路径（必填） |
| `group` | 同 `group` 的任务**严格按提交顺序、一次一个**跑 LLM 阶段。用于同一视频的分段、或必须让后一条读到前一条刚写进知识库的词条的同语料批 |
| `priority` | 数字，大的先跑（默认 0）。同时作用于 ASR 队列与 LLM 调度 |

## 运行期间：新增任务、调整优先级、撤销任务

多输入运行会在 `out/batch/<id>/` 下生成两个供你操作的文件，启动时会打印其路径（同目录另有若干
以 `.` 开头的运行器内部文件，可忽略）:

| 文件 | 谁写 | 干什么 |
| --- | --- | --- |
| `queue.jsonl` | **运行器** | 这一批的现状：每项一行，带 `_state`(queued / running / done / failed / skipped / dropped)，在跑的项还带 `_stage`(download/asr/llm) |
| `control.jsonl` | **你**（只追加） | 加任务、改优先级、撤掉还没开始的 |

两个文件各自只有一个写入方：你编辑或追加时不会与运行器冲突，运行器也不会覆盖你写入的内容。

（单个输入配合 `--batch-id` 时只会生成 `queue.jsonl`，作为本次运行的记录。该运行在前台独占
终端且仅含一项，没有可增删的任务，因此不提供控制面。）

`queue.jsonl` 的每一行带的是这一项**当初解析出的完整选项**（几十个键，包括你没写、取了默认值
的那些），看状态只看 `_state` 就行——续跑时这些记下来的值怎么用，见下面「接着跑没跑完的批」。

往 `control.jsonl` 追加一行就是一条指令，几秒内生效：

```powershell
# 加一个任务：和 manifest 行写法完全一样
Add-Content out/batch/20260831-101500/control.jsonl '{"source":"D:/media/new.mp4"}'

# 让还没开始的某项插队（名字取 queue.jsonl 里的 source/label）
Add-Content out/batch/20260831-101500/control.jsonl '{"item":"new.mp4","priority":9}'

# 撤掉一个还没开始的任务
Add-Content out/batch/20260831-101500/control.jsonl '{"item":"talk.mp4","drop":true}'
```

带 `item` 的行属于指令，不带 `item` 的则是任务——无需显式标记，因为任务行必须包含 `source`,
指令行必须指向队列中已有的任务名。无法解析的行会被跳过并告警，不影响其余行。

**已开始的任务不会被中断**：修改优先级只影响尚未轮到的任务（该操作同时重排 ASR 队列，已在排队
的任务也会随之调整）;`drop` 仅对**当前无任何阶段在运行**的任务生效（`_state` 为 `queued`）——
正在下载、转写或纠错的任务会提示「已在运行，无法撤销」。处于两个阶段之间的任务仍可撤销，此时
撤销可节省后续阶段的资源。

撤销为**永久**操作：被撤销的任务在 `queue.jsonl` 中记为 `_state: dropped`，续跑时不会被重新拾起
（只有因中断或失败而未完成的任务才会）。当整批仅剩被撤销的任务时，该批视为完成，退出码为 0。

### 续跑未完成的批次（无需记录批次 ID）

`queue.jsonl` 同时是这次运行的**记录**，跑完/中断都留着，而且它的行就是合法的 manifest 行
(`_` 开头的字段是运行器自己的，回读时会被忽略):

```powershell
# 昨天那批跑了一半，今天接着跑——不用记 id
finesub --resume-batch

# 或者指名道姓
finesub --resume-batch 20260831-101500
# 直接喂那份文件也行，但那是**另起一批**（下面的检查也不做——路径是你自己写的）
finesub --manifest out/batch/20260831-101500/queue.jsonl
```

不带 id 的 `--resume-batch` 会选择**最近一个未完成的批次**（包括被强制终止、未及正常收尾的）。
该命令查询 `<用户数据>/batches.json`——该文件仅记录批次 id、queue 文件位置与更新时间等指针
信息，不保存任何任务。

**续跑针对原批次本身，而非创建新批次**：使用相同的 id、相同的 `out/batch/<id>/` 目录、同一份
`queue.jsonl` 与 `control.jsonl`，对应同一条记录。因此续跑完成即代表该批次完成，下次不带 id 的
续跑不会再选中它。基于同样原因，`--batch-id` 不能与 `--resume-batch` 同时使用（否则会被视为
两个不同的批次）。

续跑使用各项当初解析出的完整选项——因此即使本次未再指定 `--language ja`，配置也不会改变。
**但本次命令行显式给出的选项会覆盖已记录的值**，未给出的选项按记录执行：

```powershell
# 那批当初是 --device cuda 跑的，现在这台机器没显卡了
finesub --resume-batch --device cpu
# [batch] demo: device taken from this command line instead of what the batch recorded
```

判断依据是命令行上**是否出现了该选项**，而非其取值是否与默认值相同——即使显式给出的值与默认值
一致，同样生效。覆盖是**持久**的：本轮结束后会按新值重新生成 `queue.jsonl`，后续续跑沿用新值。
批次未运行时也可直接编辑 `queue.jsonl`，此时它等同于普通 manifest，不存在并发写入。

该规则**仅适用于续跑时读入的行**。你自己 manifest 中的行、以及追加到 `control.jsonl` 的行，
仍遵循「行覆盖命令行」——这些行由你书写，而续跑的行由运行器记录。并发参数
(`--max-parallel-tasks`、`--download-workers`、`--asr-queue-size`)始终以本次命令行指定值为准，
因为它们属于运行器配置，不归属于任何单项。

（注意：将 `--stage` 向后续阶段推进会使**已完成的任务**也一并纳入——它们的后续阶段现在需要执行，
而 raw-srt 之后的阶段会消耗 LLM 配额。启动时会提示有多少项因此重新进入运行队列。）

续跑期间仍通过 `control.jsonl` 追加新任务（queue.jsonl 此时由运行器负责写入，请勿再手动追加）。
**控制面的进度是持久的，且仅在指令真正生效后推进**：若你追加的行尚未被轮询即遇强制终止，该行
**在下次续跑时仍会执行**——写入文件并不等于已执行；反之，已生效的行不会重复执行。

**超过 7 天的批次不会被静默续跑**：程序会提示该批已多久未运行，并输出含 id 的确切命令供你自行
决定。不使用交互确认是为了避免阻塞脚本；不静默续跑则是防止一周前的批次在无人注意的情况下突然
恢复运行。**显式提供 id 时不受该时限限制**——此时属于明确的指定。

另有两种情况即使提供 id 也会被拒绝，因为它们与指令是否明确无关：

- **批次仍在运行**（该批持有 `out/batch/<id>/.batch.lock`）。此时续跑会导致两套 worker 写入同一批
  产物；如需新增任务，请向 `control.jsonl` 追加。
- **批次在其他目录下运行**。`out/` 及 manifest 中的相对路径均相对于当前目录，更换目录续跑会
  找不到已有产物，并将后续产物写入其他位置。程序会在提示中说明应切换到的目录。

不想接着跑就什么都不用做——**没有常驻队列，也就没有要撤销的东西**。

### manifest 同样支持追加

manifest 批本身也是**增长型**的：直接往你自己那份 manifest 末尾追加行就行，不用停下重来。

```powershell
# 让它跑着
finesub --manifest tasks.jsonl --stage final-srt

# 另开一个终端，追加一项
Add-Content tasks.jsonl '{"source":"https://example.com/urgent","priority":10}'
```

几秒内它就会加入当前批。规则：

- **只追加，不修改**。已被消费的行即使修改也不会重新执行。
- **只读写完整行**。尚未写完的行（尚无换行符）会留待下一轮处理，不会被当作半行读取。
- **坏行仅跳过自身**并输出告警，不影响正在运行的批次。
- 如需新任务优先执行，可为其设置 `priority`；否则它将与其他任务一样按「数值大者优先」排队。
- 批次在「所有已知任务执行完毕 **且** 最后一次轮询未发现新行」时结束。此后追加的行属于下一次
  运行——由于重跑同一份 manifest 时，已完成阶段会依据产物存在性被跳过，直接再次运行即可。

## 每一项必须有自己的产物位置

若两个任务解析到同一个 SRT 路径，会产生**静默错误**：阶段依据产物存在性跳过，后一个任务会把
前一个的产物当作自己的结果，进而「成功」交付本属于其他任务的字幕。因此这类冲突会被直接拒绝：

- 同一个源列两次；
- 不同目录下的同名文件（`dir1/a.wav` 与 `dir2/a.wav` 都算出 `out/a/a.srt`）;
- 两行写同一个 `output` 或同一个 `task_artifact_dir`;
- 不同写法指向同一视频的 URL——此类冲突需在下载解析出视频 id 后才能发现，发现时**仅使该任务
  失败**，不影响批次中的其他任务。

同理，以下与单次运行绑定的选项在多输入时**不能出现在命令行**（否则会成为每一项的默认值）:
`-o/--output`、`--name`、`--task-id`、`--task-artifact-dir`，以及逐素材提供的只读输入
`--llm-video`、`--refined-srt`。如需使用，请写入对应的 manifest 行。整批共用的说明
`--task-summary` 不受此限制。

## 出错、中断、重跑

- **单项失败不影响其他任务**：记录失败的阶段与错误后，跳过该项的剩余阶段，其他任务照常执行；
  命令以退出码 1 结束，并在末尾列出失败项。
- **失败的阶段会自动重试一次**（通过 `--retry-failed N` 配置，设为 0 关闭）。重试对象是**阶段**
  而非整项任务：该阶段之前的产物已保存在磁盘上，本就会被跳过。
- **按一次 Ctrl-C**：不再启动新任务，等待当前任务完成后退出；**再按一次**：立即退出。已完成的
  阶段均保存在磁盘上，重新运行即可续跑。
- **重新运行即续跑**：对同一份 manifest 再次运行，已完成部分会立即跳过。如需强制重做某一步，
  删除该步产物**及其全部下游产物**。

## 事件流与日志

- `out/batch/<批次 id>/batch-status.jsonl`：每个阶段的 started/done/failed/retry、实际派发顺序
  (`llm-scheduled`)、运行中追加的任务(`added`)、控制面指令的结果(`dropped` / `reprioritised`)、
  知识库并发冲突(`knowledge-conflict`)。
  批次 id 默认是时间戳，`--batch-id` 可以自己指定。
- `out/batch/<批次 id>/queue.jsonl`：这一批的现状与记录（见上面「接着跑没跑完的批」），跑完/中断
  都留着；同目录的 `control.jsonl` 是你写的控制面。
- `<用户数据>/logs/run-<时间戳>-<输入名>.log`：**每一项一份**的详细日志（恒定 verbose，与
  `--log-level` 无关）。报障时直接发这个文件。

单个输入的前台运行默认不生成 batch-status（没有批次可审计），显式指定 `--batch-id` 时才写入。
