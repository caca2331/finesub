# WT 单文件分片

`whisper-timestamped`（WT）在单个长音频内使用多个独立模型实例，把对齐阶段并行化。
**已实现并完成首轮标定。**

- 规划与执行：`src/asr_playground/speech/recognition/sharding.py`（规划函数保持纯函数；
  执行器通过参数接收单 shard recognition callable，不反向导入 `transcribe.py`）
- 模型生命周期：`src/asr_playground/speech/runtime/model_pool.py`
- checkpoint identity/读写：`src/asr_playground/speech/recognition/checkpoint.py`
- 入口：生产缺省由 profile 决定；`asr-pipeline … --wt-workers N` /
  `vad-asr … --wt-workers N` 只作为 **DEV/UNSAFE benchmark 覆盖**，
  可超过 profile 并改变产物，不应由生产调用方直接传入。
- 测试：`test/test_wt_shard.py`（规划）、`test/test_wt_sharding.py`（执行）
- 设计期完整记录（实施顺序、逐条测试计划、验收标准、设计期收益预期）在本地
  `docs/archive/wt_parallelism_plan.md`，不随仓库发布。

## 实测标定（2026-07-29，RTX 5060 Ti，`large-v3-turbo`）

| 素材 | 语音 | `--wt-workers 1` | 多 worker | 实测加速 |
| --- | ---: | ---: | ---: | ---: |
| `BV1nxje63ERi` | 412.9s | 139.9s | 117.9s（w=2） | **1.19×** |
| `yingtao` | 923.4s | 233.1s | 200.2s（w=4） | **1.16×** |

（`asr_align_sec`；`BV1nxje63ERi` 的 w=2 复跑两次 126.3 / 117.9s，噪声约 7%。）

**这两个数是端到端口径，含模型加载错峰与 shard 配平失准，不是 WT 的并发效率。**
把这些开销排除、只看 WT 调用内的时间，单文件分片实测到 **1.42×**（推导见下方
「损失分解」）。引用本文数字时务必带上基线——1.1–1.2× 说的是「今天这条路径实际拿到多少」，
1.42× 说的是「WT 本身能给多少」，二者的差额是**可修复的工程损失**，不是扩展性上限。

**正确性：实测全部逐字节一致。** `workers=1` 的 `segments` 与分片实现之前完全相同；
`w=2`（显式语言、auto 各一次）与 `w=4` 的 segments/words/text 也与 `w=1` 完全相同——
好于契约承诺（契约只保证 `workers==1` 一致）。显存 w=4 峰值 7.69GiB，在 16GB 档预算内。

### 耗时构成：97.9% 在 WT 调用内

对 `BV1nxje63ERi` 包裹 `_transcribe_with_naive_fallback`（`whisper.transcribe` 的唯一入口）
逐调用计时，排除模型加载、库导入与 VAD：

| | w=1 | w=2 |
| --- | ---: | ---: |
| 对齐墙钟 | 132.9s | 124.4s |
| **WT 调用内合计** | **130.1s（36 次）** | 183.2s（同样 36 次） |
| **WT 占对齐比例** | **97.9%** | 两个 shard 各约 98% |
| 管线自身 Python | **2.1%** | 同左 |

**所以「减少 Python 侧工作」不是杠杆——本仓在 `whisper.transcribe` 之外只占 2%。**
瓶颈完全在 WT 内部（GPU 推理 + whisper-timestamped 自己的 DTW/refine）。

### 损失分解：丢的不是并发效率，是配平

同样 36 次调用，串行合计 130.1s，两路并发合计 183.2s → **每次调用在 2 路并发下慢 1.41×**。
若完美配平且无加载错峰，墙钟应是 183.2/2 = **91.6s**。对这个理想值有两个常被混用的基线，
引用时必须写明是哪个：

| 基线 | 数值 | 相对 91.6s | 含义 |
| --- | ---: | ---: | --- |
| WT 调用内合计（w=1） | 130.1s | **1.42×** | **WT 本身的 2 路并发效率**，排除加载/VAD/Python |
| 对齐墙钟（w=1） | 132.9s | 1.45× | 端到端可达上限，仍含 w=1 侧的固有开销 |

（`gpu-profiles.md` 的 1.49× 是另一套独立基准，不要与上面两个互相印证。）

也就是说 **WT 的并发扩展性本身没有问题**；实际只拿到 1.07–1.19× 是被两项可修复的损失
吃掉的：

    完美配平、无加载等待   91.6s   → 1.45×
    + 模型加载错峰         +9.9s
    + shard 配平失准      +20.4s   （两 shard 的 WT 时间 71.1s vs 112.0s，差 1.58×）
    = 实际               124.4s   → 1.07×

**配平失准是第一大损失**，且用语音秒数预测不了：两 shard 语音量差 <8%，WT 耗时却差 58%
（`yingtao` w=4 同样：语音差 <8%、耗时差 36%）。异常隔离、coverage rescue、naive 回退是否
触发都是数据相关的。

**结论与修复优先级**：

1. **`shard 数 > worker 数 + 动态派发`**——直接消掉上面两项损失（慢 shard 不再独占一个
   worker 到最后，空闲 worker 立即领下一块；模型也在首块结束时就已就位）。这是唯一值得做
   的优化，目标是把端到端的 1.1–1.2× 拉回接近 WT 自身已经能给的 1.42×。
2. 降低管线自身 Python 开销：**无效**（只有 2%）。
3. **换更强的 GPU 会按 WT 的并发曲线改善**——既然 98% 在 WT 内，显存带宽/SM 更多的卡
   饱和点更靠后，收益上限会提高。（此处修正了首轮标定的初步归因：当时误判为管线 Python 侧
   的 GIL 争用，逐调用计时证否。）

## 与 batch 模式的分工

**batch 与单文件现在走同一条路径**：`batch.py` 的 asr bin 固定为 1，每个文件独占整个
profile 的 shard/separator 宽度。

| 场景 | WT 并发来源 | 每任务 worker |
| --- | --- | --- |
| `pipeline.py` 单文件 | shard 级（本文） | 生产缺省 `task_workers`（≤ profile 上限） |
| `batch.py` 批处理 | 同上 | 同上（asr bin = 1） |

### 为什么反转了原来的设计（2026-07-30）

原设计是「asr bin = `wt_instances`，N 个文件各占一个 WT 实例、每个文件 `worker=1`」，理由是
显存包络相同、且每个 batch 产物都走 `worker=1` 的逐字节一致路径。反转的原因是**内存**：
那种做法把 N 个文件的非模型状态同时压在一个进程里，而 8GB 档**单个文件**就已实测越预算
（分离阶段 7.79GB / 上限 8GB）。当时「内存」只是没量化的担忧，现在是实测事实。

**代价要如实记账**：

- **长文件几乎无损**。别拿 1.1–1.2× 去对比文件级并行——那是**端到端**数字，含可修复的加载
  错峰与配平失准。WT 自身的 2 路并发效率是 1.42×（见「损失分解」），与文件级并行的差额
  远小于端到端数字的暗示。
- **代价集中在短文件**：防碎片阶梯会把它们挡在单 worker，文件级并行则不受此限。阈值同期
  由 300s 降到 150s 正是为了压住这一项（见「Worker 数量」）。
- **batch 产物不再保证与 `worker=1` 逐字节一致**。契约本就只保证 `workers==1` 一致；实测
  `w=2`/`w=4` 与 `w=1` 完全相同，但那是实测而非承诺。

原「搁置项」里的跨任务共享池因此也失去了动机：队列尾部只剩一个长文件时，它现在本来就用
满整个 profile。

## Worker 数量

只统计 interval 内的有效语音，不看墙钟时长，也不看静音 gap：

```python
task_workers = min(
    dev_override or profile.wt_instances,
    floor(total_vad_seconds / 150.0) + 1,      # 防碎片阶梯
    len(initial_groups),
)
```

| 总 VAD 语音时长 | 单任务最多 worker |
| ---: | ---: |
| `<2.5min` | 1 |
| `[2.5, 5)min` | 2 |
| `[5, 7.5)min` | 3 |
| `>=7.5min` | 4，再受生产 profile 上限约束；开发覆盖可更高 |

**这条阶梯的职责是防碎片，不是标定吞吐**：短音频若按 profile 上限直接分片，会被切成很多段
短 shard，每段都要付模型预热、边界 tail、shard 首组语言重检测和 recall 重叠的固定成本。
门槛保证「只有总语音量确实够多，才允许再加一个 worker」。**它的存在与方向是设计意图，
不要因为"没标定"就删掉**。没有 interval 时不加载 WT；只要有 interval 就至少一个 worker。

阈值 2026-07-30 由 300s 降到 150s，理由是**这些固定成本里最大的一项已经被消掉**：
`WtModelPool.warm()` 现在在 shard 工作开始前就用同一份 checkpoint 建好全部实例，shard 不再
以「等一次模型加载」开场（实测最差 `wait=40.3s` → 现在 `0.0s`）。同时这也是 batch 改为
单文件并行的前置条件——阈值留在 300s 的话，多数短文件会退化成单 worker。

## 语义优先的分片

shard **只能在 initial group 之间切**，不得切开 initial group——这样 shard 边界直接继承
`build_alignment_groups()` 的语义停顿策略，而不是发明第二套分段。连续讲话形成的超长 group
会让 shard 明显不均衡，这是有意的：语义完整性优先于负载相等。

> **这条不变量是整个设计的支点**，且已实证：
> `test_shard_slices_regroup_exactly_like_the_full_file` 用 12 组随机 interval 布局断言
> 「各 shard 切片重算出的 group 序列 == 全文件 initial group」。成立的原因是
> `build_alignment_groups` 是从 `remaining[0]` 起的左到右贪心、回看不越出当前 group
> （末组的 `MIN_GROUP_LENGTH` 豁免也不产生差异，因为该组本已达标）。

负载权重只用 group 内 VAD interval 的纯语音时长，不计真实 gap、保留尾音、合成静音或文件
跨度。边界从左到右选，每次按 `target = 剩余语音 / 剩余 worker` 取累计最接近的位置，约束与
排序：

1. 当前 shard 至少含一个完整 initial group；
2. 每个剩余 worker 至少保留一个完整 initial group；
3. 与 `target` 的绝对偏差最小；
4. 偏差相同时优先边界两侧真实 gap 更大处；
5. 再相同时取更早的边界（保证确定性）。

比较前按 9 位小数取整，使并列由上述规则而非浮点噪声决定。

## 执行架构

```text
source
  -> 全局 VAD（CPU/流式）
  -> initial groups
  -> shard planner（`asr_playground.speech.recognition.sharding.plan_wt_shards`）
  -> WT model pool（任务内，1..task_workers）
  -> 各 shard 并行 align_segments
  -> 按 interval 所有权合并
  -> 全局 overlap clamp / 零时长延长 / 全局 DP 分句 / energy annotation
  -> aligned JSON
```

`segment_split` 现在是**全局 DP**（`docs/segment_split.md`），在合并之后对整条 clip 重新
分句，shard 边界处的分段本就会被重算——这对边界人工痕迹有利，**但它不修复漏识别或重复
文本**，边界质量仍须由下面的 tail / 所有权机制保证。

### WT 模型池

`asr_playground.speech.runtime.model_pool.WtModelPool`：每 worker 独占一个完整模型，
一个实例只由所属线程串行调用。
**不共享模型对象**——WT 在调用期间动态注册 forward hooks，且有进程级配置与随机种子状态。
模型**惰性构建**（实际用几个就建几个）并在 shard 间复用，加载仍由
`_WHISPER_MODEL_LOAD_LOCK` 逐个串行，加载完成后推理并行；加载失败会释放槽位而不是永久占用。
`warm()` 预先加载第一个模型，使模型加载计入 `whisper_load_sec` 而非对齐耗时，也让坏模型名
在任何 shard 启动前就报错。现有 GPU model-family gate 继续保证 WT 池与 separator 不共驻。

进程级前提：`_disable_whisper_sdpa_processwide()` 把 WT 必需的非 SDPA 模式固定为进程级
常量——WT 的 `disable_sdpa()` 会保存/恢复一个 class-global flag，多线程下会竞态并丢失
attention weights。

## Shard 内部状态

每个 shard 跑完整的 `align_segments()` 顺序逻辑（动态重组、efficient/naive 解码、异常隔离、
coverage rescue、shard 内 recall 与语言历史、checkpoint），并独占 WT 模型、
`AudioBlockLoader`、`remaining`/`out`、`prev_tail_segments`、`auto_language_history` 与
checkpoint writer。loader 在 shard 结束时 `close()`（每个约 38MB 常驻块）。

动态异常隔离返回的 `unconsumed` 只能与当前 shard 内剩余 interval 重组，不跨越 shard 边界。
这是多 worker 与单 worker 可能不同的来源之一。

## Shard 边界

### 右侧 tail 上限

前一个 shard 不能把自己当作文件结尾。`align_segments()` 接受可选 `successor_start`，
让**列表末组**沿用非末组本就在用的公式，而不是无条件退回 `GAP_KEEP_REAL_MAX_SEC`：

```python
tail_limit = min(max(0.0, successor.start - current_group_last.end), GAP_KEEP_REAL_MAX_SEC)
```

⚠️ **函数内有两处依赖「下一 interval」，`_next_interval_start()` 统一了它们**：
`group_tail_limit`（padding 上限）与 `upcoming_interval_starts`（recall 临时组右边界）。
只改前者会让 shard 末组的 recall 仍按「文件到此为止」计算。

### Interval 所有权

`asr_playground.speech.recognition.sharding.tag_interval_ids()` 给每个 interval
挂上全局索引作为 `_interval_id`
（shard 是这份列表上的连续区间，索引即身份）。`_finalize_group_candidate()` 逐 interval
生成 segment，归属是**构造性**的，不需要按时间中点猜。合并时：

1. 每个 `_interval_id` 只被其所属 shard 的输出接受（shard 的 padding/recall 可以伸进邻居的
   首个 interval，那些结果在此被丢弃）；
2. 按 `(interval id, start, end)` 排序，**worker 完成顺序不影响结果**；
3. 在完整结果上再统一跑现有的 `drop_empty_segments`、overlap clamp、零时长延长、
   全局 DP 分句与 VAD energy annotation；
4. 写 JSON 前 `strip_interval_ids()` 删除该字段。

## Auto language

显式指定语言时所有 shard 一致；`language=auto` 时每个 shard 从空的
`auto_language_history` 开始，首组自行检测、后续短 group 用 shard 内最近历史。不把首 shard
的语言广播到全文件，以免破坏多语言内容。这是已接受的分歧源（首轮标定的 auto 素材全程单一
语言，该分歧未被实际激活）。

## Checkpoint 与恢复

**没有单独的 planner checkpoint**——设计阶段计划过一个，实现时判定为冗余：分片方案是
`(VAD intervals, max_workers)` 的确定性函数，重启后重新规划必得同一套 shard，各 shard 再
各自从自己的 partial 续跑。少一个要与 partial 保持同步的状态文件。

只有 shard partial，沿用现有 ASR checkpoint 机制与 `checkpoint.SCHEMA_VERSION`：

- **单 shard 计划沿用无后缀名** `<stem>-aligned.partial.json`，且不往 fingerprint 里
  加 shard 字段；checkpoint schema 当前为 v2，v1/缺版本的旧 partial 明确失效并从头重跑，
  不做历史 payload migration；
- 多 shard 时为 `<stem>-aligned.partial.shard-NNN.json`，fingerprint 追加 `shard`（id + 总数）
  与 `shard_intervals`，所以 worker 数或边界一变，旧 partial 自动失效而非被错误复用；
- 每个 shard 跑完即清除自己的 partial；计划变窄后残留的高编号 partial 由
  `_sweep_stale_shard_partials()` 在下次成功运行时清扫。

## 输出一致性契约

- `task_workers == 1`：与分片实现之前逐字节一致（**已实测**）。
- 相同输入、依赖、profile、worker 数：可重复。
- 多 worker **不保证**与单 worker 逐字节一致（首轮实测一致，但那是观察不是保证）。
- worker 完成顺序不影响合并结果。
- 多 worker 不得降低 VAD coverage、引入未解决重叠或显著增加异常重复。

允许的差异来源：shard 边界阻止动态 regroup 跨 worker；auto-language history 每 shard 重置；
跨 worker recall 搁置；并发 CUDA 的数值边界差异。

### ⚠️ profile 影响产物内容

`--gpu-budget-gb` 或开发参数 `--wt-workers` 通过 `task_workers` 改变 shard 划分，进而可能改变
aligned JSON——此前 profile 只影响速度与显存。后果与纪律：

- 同一音频在不同显存档/不同机器上可能跑出不同字幕；
- pipeline **只按产物是否存在跳过阶段**，换 profile 重跑不会自动作废已有 aligned JSON
  （fingerprint 保护的是 checkpoint，不是最终产物）；
- 因此 `metadata.asr_align.wt` 记录实际 worker 数、总语音秒数与各 shard 边界，产物能自证
  是在什么并发下生成的。审计或横向对比前先看这个字段。

batch 模式因每任务 worker=1 而不受影响。

## 日志与指标

- `Info: WT sharding (workers=…, speech=…, shards=[…])`——规划结果；
- `Info: shard N done (intervals=…, speech=…, wait=…s, align=…s)`——多 shard 时每个 shard
  的排队等待与对齐耗时。**`wait` 与 `align` 的分离正是上面归因的依据**：`wait` 大说明模型
  加载错峰，`align` 离散说明静态配平失准。
- `metadata.asr_align.wt`——产物内的并发快照。

## CPU 线程预算（`bounded_intra_op_threads`）

**每个 shard 线程进入 torch CPU 算子时，都会拉起自己的一支 OpenMP 团队**，规模为
`torch.get_num_threads()`。本机该值为 6（物理核数），于是 2 个 shard 实际要 12 个 OMP
worker 压在 6 个物理核上——超订。torch 自带 libiomp，分离阶段又在同进程里留下另一套原生
线程池，这正是 `tmp/mt8g-8gb-multithread-handoff.md` 候选 (a) 的机制。

`speech/runtime/thread_budget.py` 在分片对齐期间把预算按 shard 数均分（2 shard → 每个 3
线程），退出时恢复。**实测这不是代价，是收益**（`clip700`，w=2，空闲 GPU，两对独立复跑）：

| | shard0 align | shard1 align | 墙钟 |
| --- | ---: | ---: | ---: |
| 2×6 线程（均分前） | 67.3s / 66.7s | 72.0s / 71.8s | 97.5s / 96.8s |
| 2×3 线程（均分后） | 63.3s / 61.3s | 65.5s / 66.1s | **89.5s / 88.3s** |

均分后墙钟为均分前的 0.92× 与 0.91×，即**约 8–9%**，两对高度可复现。

> ⚠️ **务必在空闲机器上测这个数。** 首轮标定时后台在跑游戏，同样两对得到 0.84× 与 0.59×，
> 被丢弃：外部 GPU 负载会把减少 CPU 超订的收益显著放大，而且方差巨大（0.84 vs 0.59 本身
> 就是污染的信号，当时却被误当成「方向一致」的佐证）。绝对值也差得离谱——同一条 unbounded
> 配置带负载时 170.5s，空闲时 97.5s。

**输出一致性**：以 `wt_workers=1` 分别在 6 线程与 3 线程下对齐 `clip700`，**251 段产物完全
相同**——线程数是纯调度旋钮，不改浮点归约顺序。验证脚本
`out/mt8g-stress/verify_thread_count.py`（本地产物）：`--mode consistency` 比产物，
`--mode perf` 比墙钟。

**与 Problem B（候选 a）的关系**:本改动做出时,Problem B（2026-07-29 晚的双 shard 卡死）
尚未定位,候选 (a) 的 OpenMP 超订机制是它的预防目标。**事后（2026-07-30）Problem B 已定位
为 stdio 背压冻结（见下节）,与线程数无关**——本改动与卡死案不再有因果关联,它作为可测量
的吞吐收益独立成立,不必回退。

## 卡死案（Problem B）根因：stdio 背压冻结（2026-07-30 定位）

2026-07-29 晚的 8GB 双 shard 卡死（进程活着、CPU≈0、GPU util 个位数、显存不释放、只有
shard-000 的 partial 且 mtime 长时间不动）**不是 speech 栈内部死锁**：进程由一个用管道
捕获 stdout/stderr 的工具包装层启动，读端中途停止排水。ASR 阶段的 stderr 是持续流
（whisper 的 tqdm 帧进度 + 每 group 的 `Info:` 行），管道写满后下一次 write 在内核态
永久阻塞并**持有 Python stderr 流的内部锁**，于是所有会打印的线程在各自下一次 print 处
冻住。

证据与吻合点：

- `run.err.log` 从创建到 kill 始终 0 字节，而同进程的分离阶段完整跑完（分离单独跑就有
  ~16KB stderr）。Python 3.12 的 stderr 重定向到文件时**行缓冲**（实测，PS/cmd 皆然），
  「全缓冲被 kill 丢掉」的旧解释不成立——0 字节意味着输出根本没送达文件，被扣在包装层的
  管道/内存缓冲里（对照：健康重跑的 `run-aligned.err.log` 59KB 在跑完那一秒整体落盘）。
- 冻结点与产物精确吻合：checkpoint 在 group 末尾写、下一 group 开头先 print。shard-0 停在
  「checkpoint 5 已写、group 6 的 print 处」；shard-1 当时第二个模型还是懒加载
  （4520c8d 之后才全量预载），冻结时刻未完成首个 group → shard-001 partial 从未出现。
- 最小复现（管道无人读 + 两个打印线程共享 stderr）：管道收下 ~4KB 后两个线程永久冻结、
  `cpu_percent=0.0`、不打印的线程照常运行——症状签名全中。
- 从未复现的原因：所有探针与重跑都由活着的读端驱动（交互 pwsh、pwsh 自己执行的 `*>`
  文件重定向）。触发条件在启动环境，不在 workload。

推论：**长跑必须由 shell 直连文件句柄重定向（`>`/`2>` 到文件），不要经过可能被弃读的
输出捕获管道**（agent 工具的后台输出捕获是典型反例）。候选 (a)（OpenMP 超订）与候选 (b)
（RAM 越界换页）均非本案根因；(a) 作为独立的性能问题已由线程均分解决（见上节）。

## 卡死诊断（stall watchdog，仅开发期启用）

若再出现「进程活着但不再产出」，可信的第一手证据是全线程栈——若是 stdio 背压，栈会直接
显示线程停在 `sys.stderr.write`（watchdog 输出走独立文件，不受 stdio 背压影响）。

`src/asr_playground/speech/runtime/stall_watchdog.py` 提供缺省关闭的诊断钩子，
`run_vad_asr` 与 `run_vocal_separation` 各自 arm 一次（嵌套时只有最外层生效，探针可以在
进程级 arm 一次，跨越 sep→WT 边界保持单一时间线）。

> **策略：仅在开发/诊断时开启，生产跑一律不设这两个环境变量。** 未设置时 `arm()` 直接返回
> 惰性句柄——不起线程、不装计时器、不写任何输出，生产路径的成本只有一次环境变量读取。

```powershell
$env:PYTHONUNBUFFERED = "1"                       # stdout 重定向后是块缓冲，别丢尾部
$env:ASR_STALL_WATCHDOG_SEC = "180"               # 每 180s dump 一次全线程栈
$env:ASR_STALL_WATCHDOG_LOG = "out/<task>/stall-dumps.log"
asr-pipeline ...
```

用 `faulthandler.dump_traceback_later` 而不是 `sys._current_frames`：前者的计时器在独立的
原生线程上，**GIL 被某个阻塞在 C 调用里的线程持有时依然能 dump**，而后者此时根本没有
Python 字节码能执行。缺省（未设环境变量）零开销、零输出。

**但这个「不持 GIL」正是它的危险所在**，必须理解才能改这块代码：计时器线程会在不持 GIL 的
情况下遍历其它线程的帧，而 3.12 的帧是惰性物化的。若目标线程正高频进出栈帧（实测是
separator 的 `bs_roformer.forward`），就可能读到已释放的内存——2026-07-30 的压力跑里，
`repeat=True` 的周期 dump 把一个**健康**的分离阶段崩成了 `0xc0000005`（故障模块
`python312.dll`，dump 文件写到一半截断）。

因此现行实现是**一次性计时器 + Python 侧喂狗**：

- 健康时喂狗线程不断把一次性计时器推后 → **计时器永不触发，零帧遍历、零风险**；
- 真正楔死时（值得诊断的那种：没有任何 Python 能跑）喂狗线程也停了 → 计时器**恰好触发一次**。

若调用方用别的手段（如产物 mtime）判定停滞、而解释器仍然活着，应改调 `dump_now()`：
**从 Python 同步 dump 会持有 GIL，把其它线程挡住，帧是稳定的，安全**。
一句话：**同步 dump 安全，异步计时器只用于「解释器已经不跑了」这一种情况。**

判读要点：**不要只凭一次 CPU 采样为 0 就断定死锁**——分离阶段在健康状态下也会出现整秒
`cpu_percent=0.0`。可信的停滞证据是 checkpoint mtime 长时间不动；再用
`psutil` 的 `num_page_faults` 增量区分换页抖动（增量高）与原生线程死锁（增量≈0）。

## 搁置项

### 跨 worker recall

同 worker 的下一 group 会用 `normal_segments + prev_tail_segments` 计算 recall complement，
而并行 shard 启动时右侧 worker 拿不到左侧的最终 tail。现行为：shard 内 recall 完整保留，
非首 shard 的首组以空 `prev_tail_segments` 开始，右侧 tail 上限仍由真实 successor 限制，
最后做全局 overlap clamp。风险偏向「右侧首组多做一次 recall / 产生可清理重叠」而非漏识别；
首轮标定未见边界质量问题。

若日后语料审计显示边界重复明显，再评估两阶段实现（shard 先并行完成正常解码与救援 → 非首
shard 首组暂缓 recall → 左 shard 完成后导出最后 5 秒 → 再算右 shard 首组 complement →
达阈值才追加 WT job → checkpoint 增加 `normal_done`/`boundary_recall_done`）。它需要拆开
`align_segments()` 中紧耦合的正常解码、recall 与 checkpoint，在出现实测必要性前不做。

### 跨任务共享 WT 池

能消化「batch 队列尾部只剩一个长文件」的空闲，但要求改写 `batch.py` 的 asr bin 与
`vad_asr` 的模型生命周期（今天的共享是隐式的：一个 file-worker 线程持一个模型），成本很可能
高于 planner 本身。在 ~1.2× 的实测天花板下不做。

### 动态派发（shard 数 > worker 数）——**下一步该做的优化**

见「损失分解」：配平失准与模型加载错峰合计吃掉约 26% 的对齐墙钟，而它们正是静态 1:1
分片的固有缺陷。把 shard 切得比 worker 多、由空闲 worker 动态领取，可望把 1.1–1.2× 拉回
接近 1.45×（= 本机 WT 两路并发的实际上限）。代价是边界变多（每个边界都带 tail 限制、
语言重检测与 recall 空缺），需要重测边界质量。目前未做。

## 已覆盖 / 未覆盖

**单元测试**（`test/test_wt_shard.py`、`test/test_wt_sharding.py`）：worker 阶梯与各上限、
语义边界约束、平衡只用语音时长、tie-break 与确定性；successor 的两处传递、interval 归属
合并与乱序完成下的稳定性、模型不被并发复用、每 shard 独立 loader 并释放、单 shard 沿用无
后缀 partial、多 shard fingerprint 隔离与残留清扫、模型池惰性/阻塞/加载失败释放槽位；
以及 shard 切片重算 group 的随机回归。

**真实语料**（首轮）：`workers=1` 逐字节一致、w=2/w=4 产物一致、显存包络、吞吐 1.16–1.19×。

**未覆盖**：多语言混合素材（per-shard 语言重置这条分歧源实际未被激活）；shard 边界恰好
落在异常隔离 / coverage rescue 触发点的情形；>1 小时素材与更多 worker 的组合。

## 备选路线：换掉 WT

本设计的全部复杂度都来自「WT 只能单线程、必须多实例」。若改用支持 batch 推理的
`faster-whisper`，并发问题在推理引擎内部解决，本文大部分内容都不需要。

阻碍是词级时间戳质量：`tools/qwen3_explore/FINDINGS.md` §5 实测**原生 whisper 词时戳的段首
漂移中位 0.180 s、26% 超 0.3 s，且全链路无人纠正**（段尾只有 0.020 s，有能量补齐兜底）。
切点的时间不确定性就等于段首不确定性，这是目前必须用 WT（`refine` 能把段首分歧减半）的
原因。同节亦记录：`faster-whisper` 的词时戳行为与提速幅度**均未实测**（未安装）。

所以这条路线的前置问题是**能否修掉原生 whisper 的段首漂移**。若能，应优先走它，本文设计
可整体作废——考虑到分片实测只有 ~1.2×，这条路线的相对吸引力比设计时更高。
