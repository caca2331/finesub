# GPU profile 标定

本文记录 `src/asr_playground/speech/runtime/resources.py` 的 4/8/12/16GB 档位依据。档位表示整卡显存；
每档先扣除 1GiB 系统预留，再用余量约束 pipeline。标定日期为 2026-07-28，
机器为 RTX 5060 Ti 16GB，PyTorch 2.9.0，指标为
`torch.cuda.max_memory_reserved`，不含桌面等其他进程的整卡占用。

## 最终映射

| 档位 | 可供 pipeline 使用 | WT 实例数 | Separator 实例数 | Separator BS |
| ---: | ---: | ---: | ---: | ---: |
| 4GB | 3GiB | 1 | 1 | 1 |
| 8GB | 7GiB | 2 | 2 | 1 |
| 12GB | 11GiB | 3 | 3 | 1 |
| 16GB | 15GiB | 4 | 4 | 1 |

默认 profile 为 4GB。实例数采用固定硬件档位策略：4GB 为 1，每增加 4GB
各增加 1 个 WT 和 separator 实例。显存容量在常见产品线上也近似代表 GPU
算力等级，因此映射不再取本次 RTX 5060 Ti 跑分的局部吞吐最优点。混合 profile
批处理的 WT 数取其中最小值；Whisper 模型加载仍逐个执行，加载结束后推理并行。
Separator 的 `batch_size` 在当前依赖中无效，固定为 1；“实例数”指并行处理的
独立音频块数。

单个文件实际创建的 separator worker 数为
`min(profile separator instances, 文件块数)`；每个 worker 至少取得一个块。
短音频只有一个块时直接走当前线程，不创建多余线程池。

## WT 并发

模型为 `large-v3-turbo`。生产加载路径在 CPU 上**直接以 FP16 构建**模型，
LayerNorm 参数保留 FP32，再迁到 CUDA；单实例显存峰值 2.17GiB（活跃 tensor
2.00GiB）。发布的 checkpoint 本就是 FP16，而 `Whisper(dims)` 按默认 dtype 构建，
所以原来的 `whisper.load_model(...).half()` 路径会让 FP32 模型（3.2GB）与
checkpoint（1.6GB）同时驻留、再额外分配一份 FP16 副本；直接建 FP16 把**加载瞬时
RAM 从 5.10GB 降到 3.57GB（−30%）**——两条路径各在独立进程测两次，分别为
5.10/5.11 与 3.57/3.57GB（同进程连测会读高：释放模型并不会把页还给 OS，
第二条路径从被抬高的工作集起算）。改默认 dtype 是进程级操作，但它只发生在
`_WHISPER_MODEL_LOAD_LOCK` 内、且仅 CUDA 路径启用，而 CUDA 下 GPU stage gate
已保证 separator 模型族不与 WT 同时驻留。非官方模型名（本地 `.pt`、HF 标识符）
仍回落 whisper-timestamped 自己的解析与原加载路径。

`WtModelPool.warm()` 一次性读入 checkpoint 并**构建全部实例**，构建完立即丢弃
checkpoint（因此它计入预热期的峰值，而非常驻内存）。池的大小来自 shard 计划，
所有实例本来就都会被用到，惰性加载只是把其中一次构建挪到了第二个 shard 的关键
路径上——`wt-parallelism.md`「损失分解」把这项「模型加载错峰」计为实测损失。Whisper 的内部上下文满窗为 30 秒；并发 sweep 给每个实例输入
60 秒真实人声音频，使每个实例连续处理两个满窗，避免短任务启动噪声左右结论。

| 实例数 | 总 wall time | 相对单实例吞吐 | 合计峰值 reserved |
| ---: | ---: | ---: | ---: |
| 1 | 9.56s | 1.00× | 2.17GiB |
| 2 | 12.86s | 1.49× | 4.29GiB |
| 3 | 17.89s | **1.60×** | 6.01GiB |
| 4 | 24.25s | 1.58× | 8.25GiB |
| 5 | 30.77s | 1.55× | 9.98GiB |
| 6 | 37.01s | 1.55× | 12.04GiB |

> ⚠️ **这条曲线本身是对的，但单文件分片实际只拿到 1.1–1.2×。** 逐调用计时显示对齐阶段
> **97.9% 的时间就在 `whisper.transcribe` 内**（管线自身 Python 仅 2.1%），两路并发下
> 完美配平应得 1.45×，与本表的 1.49× 吻合；实际差距全部来自 shard 配平失准与模型加载
> 错峰，不是本表高估。详见 [`wt-parallelism.md`](wt-parallelism.md)「损失分解」。

另用每实例 30 秒单满窗扩展测到 7/8 实例，吞吐分别为 1.61×/1.28×，
仍未超过同轮 3 实例的 1.73×。这说明 3 是本机的局部吞吐最优点，但不再
用于截断通用 profile。显存上，1/2/3/4 实例的实测 2.17/4.29/6.01/8.25GiB
分别低于对应 profile 扣除 1GiB 后的 3/7/11/15GiB。

所有并发输出 hash 一致。并发实验同时暴露了 `whisper-timestamped` 的
`disable_sdpa()` 会保存/恢复一个 class-global flag，线程间可能竞态并丢失
attention weights；生产路径现在把 WT 必需的非 SDPA 模式固定为进程级状态，
避免并发改变识别行为。

单文件 WT 分片**已实现并完成首轮标定**：按完整 VAD interval 的有效语音时长确定 worker
数量，再只在 `build_alignment_groups()` 认可的语义边界切 shard；并发只在单文件
路径做，batch 内每任务恒 1 worker。长素材实跑得到 1.16–1.19×，而
**上表的吞吐曲线是该特性的硬件收益上限**；两者都是
本机特性——换 GPU 须重跑本节实验并更新上面的 profile 映射。跨 worker recall 与
跨任务共享池明确搁置。完整架构、扩容门槛、checkpoint、一致性契约和测试计划见
[`wt-parallelism.md`](wt-parallelism.md)。

## 人声分离

生产模型为 BS-Roformer。测试输入使用流式分离的最大读窗：600 秒 core 加左右
各 10 秒 pad，共 620 秒真实 WAV。

| batch size | wall time | 峰值 reserved |
| ---: | ---: | ---: |
| 1 | 115.843s | 2.86GiB |
| 2 | 115.669s | 2.86GiB |
| 4 | 115.503s | 2.86GiB |

当前 `audio-separator` 0.44.3 的 Roformer 路径明确不使用 `batch_size`，
bs=1/2/4 实际走的是同一条逐块单样本推理路径，因此时间和显存没有可区分的变化。
这组数据不能用于推断真正 batched Roformer 的收益；它只说明当前依赖里调大参数无效。
四档统一取语义明确的 bs=1。其 2.86GiB 峰值低于 4GB 档扣除 1GiB 后的
3GiB 上限。

为利用同卡并发，生产路径现在让多个 CUDA 分离块共享同一个 Roformer
`model_run`（也就是同一份权重和模型级缓存），但为每项浅拷贝独立的
`Separator` / `model_instance` wrapper，并清空输入路径、输出路径和
`cached_sources_map` 等任务状态。共享前先用一个完整 8 秒模型块 warm-up，
把 rotary embedding 的 lazy cache 初始化完，避免首轮并发写同一缓存。模型权重
本身为 699 个 FP32 tensor，共 639,035,184 bytes（0.595GiB）；最后一个活跃
lease 退出后才释放。

同一份 620 秒最大读窗并发实测如下；“任务显存”是相对测试启动基线的整进程
NVIDIA 显存增量，“PyTorch reserved”是 allocator 峰值：

| 并发实例 | 总 wall time | 相对单实例吞吐 | 任务显存 | PyTorch reserved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 115.843s | 1.000× | 2.86GiB | 2.86GiB |
| 2 | 206.670s | **1.121×** | 4.06GiB | 3.92GiB |
| 3 | 313.473s | 1.109× | 5.22GiB | 5.07GiB |

各实例输出 hash 均与单实例一致。2 实例是本机分离阶段的局部吞吐最优点，
但 profile 仍按 1/2/3/4 的硬件档位策略扩展。3 实例测试的进程 RAM 峰值为
8.11GB，接近“至少 8GB 空余系统内存”的边界。

### 分块：等长、且块数是 worker 数的整数倍（2026-07-30）

原先固定 600 秒 core，块数 = `ceil(时长/600)`，于是每个 worker 分到的块数不等、
末块还特别短。现在改为：先定 worker 数，再取**最小的整数轮数**使每块 core ≤ 600 秒，
块数 = `轮数 × worker 数`，各块等长。44.5 分钟素材 2 worker 时由 5 块（4×600s + 269s）
变为 **6 块 × 444.8s**——每个 worker 恰好 3 块，且单块缓冲小了约 26%。

- **worker 数由时长阶梯决定**（`separator_worker_limit`，每 300 秒音频多一个 worker）。
  分离跑在 VAD 之前，拿不到有效语音，所以只能看**墙钟时长**——这与 WT 那条按**有效语音**
  的阶梯口径不同，两个 300 不是同一个东西。
- 这条阶梯是**必需项**：块数变成 worker 数的整数倍之后，原来「短文件只有一块 → 自然只有
  一个 worker」的隐式门槛就消失了。
- **不需要另设 core 最小值**：一轮时 core = 时长/worker 数，阶梯把它下界在
  `300k/(k+1)`，k=1 时最小，即 **150 秒**；对 10 秒 pad 而言最坏是 13% 的冗余计算。
- 待拼接结果上限由 `2 × instances` 降为 **`instances + 1`**：等长块消除了那个
  2 倍前瞻本来要掩盖的落后块，同时少驻留几块已解码音频。

**产物影响**：块边界移动，分离结果因此**与 worker 数相关**，且与旧版本不同。

但**既有 `-vocal.ogg` 可以放心复用**——它就是一段有效的人声音轨，下游 VAD/ASR 不关心它
当初是怎么分块的，块边界移动带来的接缝差异与换个 profile 重跑属于同一量级。这正是
「按存在性跳过、不校验内容」的既有 resume 设计，**不要因为改了分块就强制要求重跑，也不要
求 resume 时 GPU profile 一致**。

只有一种情况需要先删除重跑：**你要的是可复现的对照**（A/B 实验、prompt 迭代测试床、
分割金标准标定等需要冻结上游的场景）。日常生产跑不在此列。

## RAM 口径：`peak_mem` 是**按 stage 采样**的

`peak_gmem` 可以用 `torch.cuda.reset_peak_memory_stats` 每个 stage 归零，但
Windows 的 `PeakWorkingSetSize`（以及 POSIX 的 `ru_maxrss`）是**进程生命周期峰值，
没有任何 API 能重置**。同进程跑 sep→WT 时，WT 报的数必然 ≥ 分离阶段的峰值；
batch 模式下第一个任务冲高后，后面每个任务都继承那个数，于是预算告警变成永久噪声、
真正的回归反而看不见。

因此 `StageMemorySampler`（`speech/runtime/resource_usage.py`）用一个 0.25s 间隔的
采样线程记录**当前**工作集，给每个 stage 独立的峰值，**预算判定用的就是这个 stage 值**。
两个代价必须记住：

- 采样会**漏掉短于采样间隔的尖峰**，所以它是下界。模型加载那种数秒级瞬时占用能稳定抓到。
- 进程生命周期峰值并未丢弃：当它高于 stage 值时，额外打印一行 `peak_mem_process`，
  这样既不会把 stage 数误读成整轮，也能看出前面的 stage 冲得更高。

单进程 CLI（`vad_asr` / `energy` 的 `main()`）不传 sampler，退化为原来的进程峰值——
那里一个进程只有一个 stage，进程峰值本来就是对的。

## 一致性与边界

- FP16 主体 / FP32 LayerNorm 的 185 秒真实音频对照输出与旧 FP32 staging
  路径逐字节一致，并把单实例峰值从 5.28GiB 降至 2.17GiB。
- FP16 直建（2026-07-30）与旧 `load_model(...).half()` 路径构建出的模型，
  **589 个 parameter/buffer 全部逐比特相同，dtype 亦相同**——权重相同则输出
  不可能不同，故未另跑对齐比对。验证脚本 `out/mt8g-stress/verify_fp16_build.py`
  （本地产物，不随仓库发布）：`--mode compare` 比权重，`--mode stock|new`
  各自独占一个进程测峰值。
- 并发 sweep 覆盖正常 efficient 路径；beam rescue / naive fallback 不会增加
  模型权重份数，但极端输入的临时 activation 仍由 `heavy_resource` 守卫测试持续观察。
- GPU model-family gate 允许多个 separator 或多个 WT 同类任务并行，但不允许
  两个模型族跨任务同时驻留；切换前由最后一个活跃任务释放上一模型族。
- 数值是这套硬件与依赖版本上的实测标定，不应外推为所有 GPU 的同等吞吐比例。
