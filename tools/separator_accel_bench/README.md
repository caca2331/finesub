# separator_accel_bench —— eager / torch.compile / AOTI 三档重测（2026-08-25）

在 `torch 2.11.0+cu128` 上重测三档效率，**只取热缓存**，固定开销单独拆出来。

结论在 [`docs/separator-optimization.md`](../../docs/separator-optimization.md) 的
「三档效率重测」一节。**本文是协议、全部数据与复现方式。**

## 先说一处测量缺陷（它影响 E5–E11 的旧数字）

`_TimedModel` 此前**在编译臂无条件安装、eager 臂从不安装**。它每次 forward 前后各做一次
`torch.cuda.synchronize()`，而那次同步会把下一块的 H2D 拷贝与上一块的 CPU overlap-add
**串行化**——不是免费的：同一条 AOTI 生产路径，带它 32.64s、不带 29.35s，34 块上差 3.3s
（~97ms/块）。三档从来没在同一把尺子上量过，且**编译臂被系统性量慢**。

已把两个安装点统一收到 `--time-forwards` 开关下，默认不装。**带仪器的运行只做归因，
报吞吐一律用不带的。** 读 E5–E11 的绝对值时要把这一条计进去。

## 环境

`tmp/venv-torch211/`：torch 2.11.0+cu128 / torchaudio 2.11.0 / triton-windows 3.6.0.post26 /
audio-separator 0.44.3 / **librosa 0.11.0**。

两个装机坑：**librosa 1.0 与 audio-separator 0.44.3 不兼容**，症状是静默产出零个输出文件
（`RuntimeError: No output files were produced`），必须钉 0.11.0；**`audioread` 在 librosa 1.0
里不再是依赖但仍被需要**，要单独装。

wheel 缓存在 `cache/wheels/torch-2.11.0-cu128/`（2.6 GB），重建 venv 是离线的：

```bash
bash tools/separator_accel_bench/run/fetch_torch211.sh   # 只在缓存为空时需要
bash tools/separator_accel_bench/run/build_venv211.sh
```

三档统一 `FINESUB_SEPARATOR_ACCEL=0` 走基准工具自己的路径，避免生产 accel 再叠一层；
`TORCHINDUCTOR_CACHE_DIR` 钉到 accel 根下的 `inductor/`，热缓存可复现。

## 吞吐（无 forward 仪器）

**270 秒素材 / 4GB profile / 1 worker：**

| 档 | wall | 相对 eager | 实时倍率 | 模型加载 | 编译/恢复 | 去掉固定开销 | 相对 eager | forward 中位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 34.35s | 1.000× | 7.9× | 3.72s | — | 30.62s | 1.000× | 747.3ms |
| `torch.compile` 热 | 46.45s | **0.739×** | 5.8× | 3.91s | **21.81s** | 20.73s | 1.477× | 397.1ms |
| AOTI | **28.86s** | **1.190×** | 9.4× | 3.74s | 9.13s | 16.00s | **1.914×** | 353.4ms |

**2015 秒素材 / 8GB profile / 2 worker：**

| 档 | wall | 相对 eager | 实时倍率 | 模型加载 | 编译/恢复 | 去掉固定开销 | 相对 eager | forward 中位 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 210.37s | 1.000× | 9.6× | 3.68s | — | 206.69s | 1.000× | 1474.4ms |
| `torch.compile` 热 | 149.32s | 1.409× | 13.5× | 4.08s | **32.40s** | 112.84s | 1.832× | 809.0ms |
| AOTI | **115.59s** | **1.820×** | 17.4× | 3.70s | 9.84s | 102.05s | **2.025×** | 698.5ms |

冷编译作参照：270 秒素材上 `torch.compile` 冷缓存 193.68s，其中 167.57s 是编译本身。
长素材那次「冷」只有 159.95s、与热的 159.46s 无异——**inductor 缓存按形状键**，短素材那轮
已经喂热了同一批形状，跨素材共享。

## 恢复阶段的构成（`breakdown.py`，带仪器）

| 档 | 包装/产物加载 | 预热首次 fwd 多付 | 分离首块又多付 |
| --- | ---: | ---: | ---: |
| 270s JIT 热 | 1.54s | 12.80s | **7.47s** |
| 270s AOTI | 3.87s | 5.25s | 0.01s |
| 2015s JIT 热 | 1.63s | 13.65s | **17.11s** |
| 2015s AOTI | 3.52s | 5.59s | 0.74s |

**JIT 要付两次**——预热一次，真正第一块又一次（长素材上 17.11s）。AOTI 预热完就干净了。

## 恢复阶段几乎不用 GPU 算力（`gpu_trace.sh` + `read_trace.py`）

`nvidia-smi -lms 200` 采样，以 `load_model` 完成的日志时间戳为锚，按工具报的各段时长对齐
（各段严格顺序执行）。本机桌面应用占着基线，所以基线不是 0：

| 阶段 | GPU 均值 | 中位 | >50% 的采样 | 显存增量 |
| --- | ---: | ---: | ---: | ---: |
| 基线（进程未启动） | 17% | 16% | 0% | — |
| JIT 包装 + 预热首次 forward（1.5 + 13.0s） | 19% | 17% | **0%** | +1047 MiB |
| AOTI 产物加载 + 预热首次 forward（3.4 + 5.4s） | 22% | 20% | **0%** | +954 MiB |
| 分离阶段 | 49–62% | 19–62% | **43–72%** | — |

恢复期整段**没有一个采样超过 50%**；分离阶段有 43–72%。与它的内容吻合：dynamo 追踪与 guard
重建是纯 Python，inductor 缓存命中后是磁盘读，AOTI 是产物解包与常量注入
（`user_managed=True`，绑指针而非搬运）——都要 CUDA 上下文，都不占 SM。

## 能不能先载进内存、就绪了再占显存

不能，对编译产物是结构性的：inductor 缓存键**包含设备**，模型在 CPU 上 trace 出来的是
C++/OpenMP kernel 不是 Triton CUDA kernel；AOTI 包为 sm120 编译死，`load_constants
(user_managed=True)` 绑的是**已在显存里**的权重指针。不存在设备中立的已编译形态。

能挪的只有真正设备中立的那部分（`load_split.py` / `load_device.py`）：

| 组成 | 耗时 | 能否先在内存里做 |
| --- | ---: | --- |
| audio-separator 自身 setup | ~1.8s | ✅ |
| `load_model` 的模块构造 | 1.65s | ✅ CPU 1.65s vs CUDA 2.00s，**82% 中立** |
| checkpoint 读盘（610 MiB / 699 张量） | 0.23–0.33s | ✅ |
| 权重搬显存 | 0.22s（pinned 0.09s） | ❌ |
| AOTI 产物加载 + 首次 forward 恢复 | ~9s | ❌ |
| JIT dynamo/guard/inductor 恢复 | ~22s | ❌ |

可挪的约 3.4s，占 AOTI 那 13s 空转窗口的 26%、JIT 26s 窗口的 13%。**注意反直觉的一点：
`load_model` 那几秒的大头不是读盘也不是搬显存（合计 0.5s），是 Python 建模块。**

**但今天挪了不值钱。** 闸门全仓只有两个获取者（`separation.py`、`vad_asr_stage.py`），
一次 pipeline 里严格顺序，batch 的 asr bin 恒为 1 worker——**闸门在现有并发模型下从不争用**，
缩短租约不改变任何墙钟时间。而且闸门管的是**显存**不是 SM（"never mixed families"），
恢复期确实已占约 1 GiB，把它挪到闸门外等于废掉闸门的目的。要让这条有意义，得先有
asr 并发 > 1——而 `docs/gpu-profiles.md` 的结论恰恰是 **ASR 永远单 worker**。

## 与 E11 的对照

长素材复现良好：AOTI 1.820× vs E11 的 1.895×（−4%），JIT 热 1.409× vs 1.381×（+2%）。
**270 秒素材的 AOTI 没有复现**：本轮 1.190×，E11 记 1.481×；eager 侧反而本轮略快
（34.35s vs 36.51s）。去掉仪器只会让编译臂更快，所以差距不由上面那处缺陷解释。本机当时
**不满足固定协议的「空闲桌面环境」**（Chrome / Cursor / 桌面应用都占着 GPU），这是已知的
唯一差异，但它解释不了「eager 更快而 AOTI 更慢」的方向。**这一格存疑，别拿它当基线。**

## 复现

```bash
export FINESUB_ACCEL_WORK=out/separator-accel-bench
bash tools/separator_accel_bench/run/bench211.sh        # 带仪器，用于归因
bash tools/separator_accel_bench/run/bench211_clean.sh  # 无仪器，用于报吞吐
bash tools/separator_accel_bench/run/gpu_trace.sh trace-jit --torch-compile --compile-scope all

python tools/separator_accel_bench/report.py            # 吞吐表
python tools/separator_accel_bench/breakdown.py         # 归因表
python tools/separator_accel_bench/read_trace.py trace-jit "01:52:32,808"
python tools/separator_accel_bench/load_split.py        # checkpoint 读盘 vs 搬显存
python tools/separator_accel_bench/load_device.py       # load_model 的设备中立占比
```
