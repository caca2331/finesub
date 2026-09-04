# BS-Roformer 推理效率探索

本文记录人声分离推理优化的实验协议、逐项结果与取舍。生产配置只吸收经过同素材性能复测、
波形相似度检查和资源上限检查的改动；探索过程不以 GPU utilization 作为吞吐效率的替代指标。

## 固定协议

- 硬件：**E0–E13 为 RTX 5060 Ti 16GB（sm_120）**；2026-08-27 起本机 CPU/内存/显卡整体升级，
  现为 **RTX 5070 Ti 16GB（70 SM，仍 sm_120）**，E14 起用新机。同架构，所以按架构编死的
  SASS 仍然可加载；但**卡型现在也在缓存键里**（见下），换卡会重建包——`max_autotune` 是在
  当前卡上实测选 kernel 的，tile 选择随 SM 数变，只按架构分目录会把旧卡调优的包喂给新卡。
  **绝对时间不可跨机比较**——E0–E13 的每个数字都偏慢，做新旧对照必须在新机上重取对照臂。
  单卡、空闲桌面环境。
- 软件：当前分支锁定的项目代码；运行结果同时记录 Torch/CUDA 版本。**E0–E10 取自
  `torch 2.9.0+cu128`；生产钉版已改为 `2.11.0+cu128`，关键行由 E11 在该版本上重取。**
- **判据**：主判据是**下游 VAD 分段与 eager 的一致性**；SI-SDR 只作量级粗筛。理由见 E11
  末尾——它衡量的是与 eager 的一致性而非质量，且是全局能量比，会稀释局部失效。
- 质量素材：`BV1kYLR6AEXv-source.wav`，270.016 秒真实音频。
- 性能长素材：`clip700.ogg`，700.032 秒，8GB profile、2 worker。worker 阶梯（E7）另用
  它自拼的 1400.072 秒 `tmp/clip1400.flac`——700 秒会被时长阶梯封顶在 3 个 worker。
  E10 改用真实长素材 `assets/bilibili/BV1ojjc6MEAs.ogg`（2014.753 秒），并在同一素材上
  量了 ASR，用来定分离与 ASR 的相对占比。
- 默认先用 4GB profile（单 separator worker），隔离模型/算子收益；之后才重新 sweep worker 数。
- 固定 600 秒 core、10 秒 pad；输出 FLAC，避免有损编码污染差异。
- 每个 variant 在独立 Python 进程运行。比较 wall time、peak allocated/reserved、cosine、MAE、
  RMSE、SNR 和 SI-SDR。影响数值的优化还抽查下游 VAD 边界。

两轮探索的脚本、协议与逐项数据各自成目录，不在本文里展开：
[`tools/separator_rate/`](../tools/separator_rate/README.md)（工作采样率）与
[`tools/separator_accel_bench/`](../tools/separator_accel_bench/README.md)（三档效率重测）。

基准工具：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m tools.separator_benchmark INPUT OUTPUT --mode fp32 --result RESULT.json
python -m tools.separator_benchmark INPUT OUTPUT --mode amp --reference FP32.flac --result RESULT.json
```

实验开关（`--axis-sdpa`、`--inference-mode`、`--defer-per-file-cache-clear`、
`--no-amp-warmup`、`--torch-compile`、`--aoti-transformer-dir`、`--model-sample-rate`、
`--time-forwards`）只存在于独立基准工具；
被否决的 hook 不进入生产模块。生产的 AOTI package 由第一次运行自建到
`cache/separator-accel/<key>/aoti/`；`python -m tools.separator_aoti OUTPUT_DIR` 只用于把
**变体**建到指定目录做对照，默认即最终配置（运行期常量折叠 + `--attention-backend axis`
+ `--targets all`；`emulate_precision_casts` 在 2.11 上必须关闭，见 E11）。
工具默认复现最终生产配置；重放 E0–E3 的 FP32 预热条件时需加 `--no-amp-warmup`。
JIT 侧的 `--compile-scope all` 对齐 AOTI 的默认 target 集合，用于同 scope 比较（E10）。
编译实验可加 `--probe-compile-timing`，额外做一次同进程 warmup forward，并从可比 wall time
中扣除这次 probe。**逐 forward 的 CUDA 同步现在由 `--time-forwards` 控制，默认关**——它会把
H2D 与 CPU overlap-add 串行化（短素材上约 3.3s），只适合做时间归因，不能用来报吞吐；
E5–E11 的编译臂数字是在它无条件开启的情况下取的，见「三档效率重测与一处测量缺陷」。

## 档位探测：`cl.exe` 不等于工具链

`cxx_toolchain_available()` 在**选档之前**回答「这台机器能不能编译」，因为答错的代价是用户被
告知要跑 90 秒编译、然后掉回 eager——正好是这个探测存在的理由。

它曾经只查 `shutil.which("cl.exe")`，于是一个半配好的开发者 shell（继承了编译器目录、没有
vcvars 的环境）会被判成可用，构建随后死在 `#include <array>` 上。现在 `cl.exe` 那一支还要求
`INCLUDE` 里真能找到标准头；`_find_vcvars()` 那一支不变，因为启用 vcvars 本来就会配好
`INCLUDE`。`_activate_msvc` 用同一个判据——两者的 docstring 一直声明「不能互相不同意」。

**不做**的一件相关事：Finoka 的下游 patch 还给 Triton 的 Windows driver 打了运行时猴补丁
（字符串替换 `CUlaunchAttribute clusterAttr = {}` 这类 C11 非法空初始化，外加 `/wd5105`）。
我们不收：靠匹配第三方 codegen 的措辞、失败时 `except: pass`，Triton 一改措辞就静默失效，
而静默失效意味着回到它本来要修的构建失败。将来若真要收，底线是按 Triton 版本 gate、
补丁没打上时 `current_reporter().warning`——不允许静默。

## 实验日志

### E0：AMP 质量基线

状态：通过，设为后续质量与性能基线。

- 生产固定启用 `audio-separator` 的 CUDA autocast；FP32 只由开发基准工具生成回归基线。
- FP32 与 AMP 使用同一模型、分块、输入量化和输出编码路径。
- 先以单 worker 测量，避免把精度收益与多 stream 调度混在一起。

| variant | wall time | 相对 FP32 | peak reserved | 与 FP32 cosine | SNR / SI-SDR |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 54.476s | 1.000× | 2.86GiB | 1.0 | ∞ |
| AMP | 34.157s | **1.595×** | 2.86GiB | 0.999999964 | 71.40 / 71.41dB |
| AMP repeat | 34.048s | 1.600× | 2.86GiB | 与首轮逐样本一致 | ∞ |

结论：AMP 把整阶段 wall time 降低 37.3%，两次 AMP 时间相差 0.3%，且输出逐样本一致。
相对 FP32 的 MAE 为 6.54e-6、RMSE 1.73e-5、最大误差 4.27e-4（约 14 个 PCM16 LSB）。
峰值未下降是因为这组基线的共享模型 warm-up 仍以 FP32 运行；E4 单独处理该问题。

下游 VAD 对照仍为 81 段，81 个 start 全部相同；仅 2 个 end 各移动一个 20ms 帧、
方向相反，累计语音时长只差 0.0038ms。能量轨迹 MAE 为 0.0395dB，最大单帧差
3.27dB。结合波形 71.4dB SI-SDR，接受为质量基线。

### E1：延后 allocator/cache 清理

状态：否决，无稳定端到端收益。

`audio-separator` 在每个外层文件块结束时执行 `gc.collect()` 和
`torch.cuda.empty_cache()`。实验路径改为只在整个 separation stage 的最后一个 lease
退出后清理，检查一个并发块结束时是否会干扰仍在推理的 sibling block。

- 单 worker wall：34.05s → 33.55s（1.5%）；输出与 AMP 基线逐样本一致。
- 干净环境 700 秒、2 worker：75.368s → 75.502s（+0.18%）；输出逐样本一致。
- 并发结果落在噪声内且未改善，因此不覆盖依赖的资源释放语义。

### E2：`torch.inference_mode()`

状态：否决，依赖内部 `no_grad()` 已覆盖主要收益。

在 `audio-separator` 的完整单文件调用外层使用 `torch.inference_mode()`，覆盖其内部张量创建；
依赖自身已有 `no_grad()`；实验检查关闭 version counter 与 view tracking 是否还有可测收益。

- 早期组合实验为 33.55s → 33.16s，但混有 E1，不能单独归因。
- 干净隔离重测：35.564s → 36.765s（−3.4%）；输出逐样本一致。
- 没有端到端收益，不进入生产。

### E3：按轴选择 FP16 SDPA backend

状态：**后被 E8 翻案并采纳**——本节否决的是**运行期**选 backend，编译期固定下来之后
那个不稳定性就不存在了。别停在这一节。

RTX 5060 Ti / Torch 2.9 的真实 shape 微基准：

| 轴 | shape | efficient | cuDNN | 选择 |
| --- | --- | ---: | ---: | --- |
| 时间 | `[62, 8, 801, 64]` | 7.786ms | 4.708ms | cuDNN |
| 频率 | `[801, 8, 62, 64]` | 0.365ms | 0.796ms | efficient |

仅对 CUDA FP16/BF16 生效；阈值 256 只区分本模型固定的 801/62 两种序列长度。

- 270 秒单 worker：35.017s，比 E0 AMP 34.157s 慢 2.5%。
- 700 秒、2 worker：第一次 72.245s，相对同轮纯 AMP 75.368s 快 4.15%；第二次
  77.111s，反而慢 2.31%，两次自身相差 6.7%。
- 相对 AMP 的 cosine 0.999999990、SI-SDR 77.10dB；两次 SDPA 输出之间 SI-SDR
  98.11dB。质量可接受，但 backend 选择/双 stream 性能不稳定，不能作为生产优化。

### E4：AMP 精度共享模型预热

状态：通过，进入生产。

原共享模型先用一个完整 8 秒 FP32 零输入初始化 rotary cache，再进入 AMP 推理；这使一次
FP32 activation 峰值计入整个 stage。现在生产预热固定使用 AMP；开发基准的 FP32 mode
仍会完整走 FP32 预热与推理。

| variant | wall time | peak allocated | peak reserved | 输出差异 |
| --- | ---: | ---: | ---: | --- |
| FP32 warm-up + AMP | 35.564s | 1.58GiB | 2.86GiB | 基线 |
| AMP warm-up + AMP | 34.834s | 1.55GiB | **2.26GiB** | 逐样本一致 |

峰值 reserved 下降 21.0%（约 0.60GiB），wall 同轮改善 2.1%。相对普通 AMP 的 81 个
VAD segment 和 27,001 帧能量轨迹也全部精确一致。

## 最终生产改动

AMP（E0/E4，2026-08-02）：

1. CUDA 人声分离生产路径固定 `use_autocast=True`；内部 `use_amp=False` 仅供开发基准生成
   FP32 回归结果，不暴露为生产 CLI。
2. 共享 Roformer warm-up 跟随本次运行精度，避免 AMP 任务先制造 FP32 activation 峰值。
3. 元数据记录实际 `amp` 状态。未新增运行时依赖。

编译路径（E5/E6/E9/E10，2026-08-03，随 torch 钉版到 2.11 落地）：

4. 三档自动选择 `aoti` / `jit` / `eager`，实际生效的那档记进元数据的 `accel` 字段。
   两档都要 `triton`（Windows 由 `triton-windows` 提供，自带 TinyCC）；AOTI 另需 MSVC，
   由 `vswhere` 在**选档之前**探测。选择规则与失效机制见 README_DEV「分离器的编译加速」。
5. 首次运行在**已加载的那个模型上**建包，产物落 `cache/separator-accel/<key>/`；
   `<key>` 含 `BUILD_FORMAT` / torch / CUDA / GPU 架构 / **卡型** / checkpoint，换任意一项
   即换目录。卡型是 `max_autotune` 落地后加的：同架构不同卡（sm_120 从 5060 Ti 到 5090）
   调优出的 tile 不一样，而症状只是「加速没有实测的那么好」——最不容易被发现的那类。
6. 任何一步失败都降级并在 stderr 说明，失败结论写进 `probe.json` 以免每次重付构建。
   新增运行时依赖只有 `triton-windows`（Windows）。

### E5：TorchInductor / CUDA Triton

状态：**已进入生产**（2026-08-03，随 torch 钉版到 2.11 一并落地），作为 `jit` 档——
只在输入 ≥600 秒时启用，因为每进程准备成本约 35 秒。档位选择见 README_DEV
「分离器的编译加速」。本节写于它还在等 Torch 升级的时候，数字取自 2.9.0。

- TensorRT 10.13 Windows runtime wheel 实测约 1,448MiB，另需 Torch-TensorRT；不满足本轮
  轻量依赖边界，按决策规则跳过。NVIDIA 的 pip 安装路径虽不要求手工安装 C++ SDK，但仍会
  拉取完整 runtime；Torch-TensorRT 还要求 TensorRT 与 Torch 版本配套。
- Torch 2.9 对应 `triton-windows==3.5.1.post24`，CPython 3.12 wheel 为 46.5MB，并自带
  最小 CUDA 12.8 toolchain。
- CUDA FP16 GEMM + SiLU 的 `torch.compile(backend="inductor")` 自检通过。
- Windows 未安装 MSVC 时，Dynamo 默认会用 `cl.exe` 编译 symbolic-shape guard；基准工具把
  `enable_cpp_symbolic_shape_guards` 关闭，改用 Python guard，CUDA graph 仍由 Triton 编译。

全模型编译的 steady 很快，但顶层 `einops.rearrange` 因 rank 2/4 复用而达到 8 次重编译上限；
最终方案只编译每层 time/frequency Transformer，STFT、band split、mask estimator 和顶层
rearrange 保持 eager。

| variant | 素材 | wall | 相对 eager | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| eager AMP | 270s / 1 worker | 34.834s | 1.000× | 2.26GiB |
| full compile，首次冷编译 | 270s / 1 worker | 247.328s | 0.141× | 2.11GiB |
| full compile，磁盘缓存命中 | 270s / 1 worker | 58.571s | 0.595× | 1.88GiB |
| Transformer 局部编译，全新缓存 | 270s / 1 worker | 54.965s | 0.634× | 2.07GiB |
| Transformer 局部编译，磁盘缓存命中 | 270s / 1 worker | 35.917s | 0.970× | 1.88GiB |
| eager AMP | 700s / 2 worker | 75.368s | 1.000× | 3.09GiB |
| Transformer 局部编译，热缓存 | 700s / 2 worker | 62.235 / 62.487s | **1.208×** | 2.42/2.36GiB |
| Transformer 局部编译，全新缓存 | 700s / 2 worker | 82.450s | 0.914× | 2.44GiB |

#### 270 秒素材统一分项计时

`torch.compile(...)` 是惰性的：API 调用只包装模块，kernel/graph 编译发生在首次 forward。
因此这里分别记录包装、首次 forward（准备 + 执行）以及同进程第二次同形状 forward（复用图
执行）。全新 `TORCHINDUCTOR_CACHE_DIR` 测真正冷编译，再用同一目录启动新进程测磁盘缓存复用。
“运行”是端到端 wall 扣除 `torch.compile` 独有的编译/恢复增量后的 pipeline 运行估算，仍包含
所有方案都有的模型加载（本机约 4.1 秒）、warmup 实际执行、音频 I/O、overlap-add 和编码，
不是只统计 GPU kernel。eager 行的 0 只表示没有编译增量，不表示没有初始化或准备工作。

| 方案 | 冷编译增量 | 冷运行（含模型加载等） | 冷启动总计 | 热恢复增量 | 热运行（含模型加载等） | 热启动总计 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全模型编译 | 222.212s | 25.115s | **247.328s** | 34.224s | 24.347s | **58.571s** |
| 仅编译 Transformer 主干（24 个模块） | 31.682s | 23.284s | **54.965s** | 13.056s | 22.861s | **35.917s** |
| 不编译（eager AMP） | 0 | 34.834s | **34.834s** | 0 | 34.834s | **34.834s** |

| 进程状态 | compile 包装 | warmup 首次 forward | warmup 复用 forward | separation 首次 forward | separation 后续中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全模型，全新缓存 | 1.988s | 150.987s | **0.418s** | 70.028s | **0.372s** |
| 全模型，同缓存新进程 | 2.016s | 21.081s | **0.422s** | 11.926s | **0.377s** |
| Transformer，全新缓存 | 2.045s | 26.404s | **0.359s** | 3.943s | **0.351s** |
| Transformer，同缓存新进程 | 2.021s | 8.278s | **0.362s** | 3.468s | **0.348s** |

用“首次 forward − 同进程复用 forward”估算惰性准备时间；它能清楚区分实际 kernel 执行，
但磁盘命中进程里的 Dynamo guard、Python graph 装载和缓存反序列化不能再细分成单独项目：

| 进程状态 | warmup 编译/恢复估算 | separation 编译/恢复估算 | 含包装的准备总计 | 扣除准备后的实际 pipeline 运行估算 |
| --- | ---: | ---: | ---: | ---: |
| 全模型，全新缓存 | 150.569s | 69.656s | **222.212s** | **25.115s** |
| 全模型，同缓存新进程 | 20.659s | 11.549s | **34.224s** | **24.347s** |
| Transformer，全新缓存 | 26.045s | 3.592s | **31.682s** | **23.284s** |
| Transformer，同缓存新进程 | 7.915s | 3.119s | **13.056s** | **22.861s** |

两个 phase 的 tensor signature 都是 `[1, 2, 352800] / float32 / cuda:0`，但 warmup 与正式
separation 的调用上下文仍触发各自的首次准备；单看 shape 不能推断图已完全复用。冷编译结果
相对 eager AMP 的 cosine 为 0.999999983、SI-SDR 74.65dB；磁盘缓存复用进程与冷编译输出
逐样本一致。Transformer 冷编译相对 eager 的 cosine 为 0.999999986、SI-SDR 75.66dB，
其热缓存输出也与冷编译逐样本一致。结论是两种编译范围的纯运行都更快，但全模型一次性冷编译
约 222 秒，即使已有磁盘缓存，新进程仍有约 34 秒准备成本；Transformer 主干把准备成本压到
冷 31.7 秒/热 13.1 秒，但 270 秒任务的热启动端到端仍比 eager 慢 3.1%。

局部编译稳态从单 worker 约 1.48 提升到 2.80 it/s，双 worker 从每路约 0.76 提升到
1.43 it/s。两次热缓存 700 秒测试只差 0.4%，输出逐样本一致。相对 eager 的 700 秒输出
cosine 为 0.999999990、SI-SDR 76.99dB；270 秒下游 VAD 仍为 81 段，所有边界完全一致，
能量轨迹 MAE 0.0337dB。

决策：热缓存长任务有约 17% 端到端收益，但全新缓存的 700 秒任务仍慢 9%。（E10 用同 AOTI
的编译范围复测了这条路径，并给出了回本长度阈值；本节的收益数字只在足够长的任务上成立。）按实测斜率，
最坏情况下约 15–16 分钟才能回本，生产若接入应使用至少 20 分钟的保守阈值。当前仓库正式
依赖仍是 Torch 2.8，对应 Triton 3.4；本机验证环境是 Torch 2.9 + Triton 3.5。项目明确禁止
直接升级到 Torch 2.9（torchaudio 解码路径回归），因此本分支不修改正式依赖和生产路径，
只保留可复现基准。等 Torch 栈正式升级后，可直接复用局部编译实现与阈值数据。

### E6：Regional AOTInductor（无权重 artifact）

状态：2026-08-03 修复常量烘焙缺陷后质量与性能双双通过（270 秒 1.42×、700 秒 2 worker
1.62×，峰值 reserved 降 18.8%/33%，VAD 边界与 eager 逐段相同），**同日进入生产**作为
`aoti` 档：有包或本机能建包就用它，不设时长门槛。生产的包由第一次运行自建到
`cache/separator-accel/<key>/aoti/`，`tools/separator_aoti` 只用于建变体。

24 个 Transformer 参数结构一致，但有 time `[62, 801, 512]` 与 frequency
`[801, 62, 512]` 两种 FP16 输入签名。离线只编译各一份 `.pt2`，设置
`aot_inductor.package_constants_in_so=False`（2.9 还要另设 `package_constants_on_disk=False`；
2.11 把它换成了 `package_constants_on_disk_format`，默认值已经是「不落盘」）；运行时从正常
checkpoint 模型的参数和 rotary cache 构造 constant map，通过 `load_constants(...,
user_managed=True)` 注入 24 个 runner。zip 检查确认没有 `data/weights/`：

| package | 大小 | export | AOT compile | 同块相对误差 | 换 block 1 权重后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| time | 634,150B | 1.06s | 15.9s | 4.34e-4 | 5.12e-4 |
| frequency | 613,352B | 0.27s | 8.0s | 4.85e-4 | 5.20e-4 |

（上表为运行期折叠 + `emulate_precision_casts` 的 Transformer package；单模块 cosine 在
这个量级已无分辨力，改用相对 L2 误差并成对给出交叉 block 结果。E8/E9 之后默认构建还会
固定按轴 SDPA backend 并多出两个 band 级 package。）

Windows 下离线编译使用本机已有 MSVC 14.44。PyTorch 2.9 的 AOTI link list 漏掉生成 wrapper
实际使用的 CUDA Runtime，实验配置通过 `aot_inductor.custom_op_libs=["cudart"]` 补齐；没有
修改 site-packages。该版本 loader 还会按 zip 顶层前缀使用固定临时解压目录，同一 package
不能直接加载 12 次；基准运行时只重写 24 份约 0.6MB 代码归档的顶层前缀，进程退出即清理，
持久 artifact 仍只有上面两份且不含权重。

修复前（编译期折叠）测得 270 秒端到端 22.808s / 23.067s。那是用错 gate bias 的运行，
数字作废；本节后面的所有计时都是修复后 2026-08-03 在空闲机器同一时段重测的。

#### 常量烘焙缺陷与修复（2026-08-03）

初版无权重 package 的质量不达标（相对 eager SI-SDR 39.04dB、VAD 81 段掉到 76 段），
当时归因为“24 层低精度累积”，`emulate_precision_casts=True` 复测也没救回来。实际原因
不是精度：

- 单模块随机输入下 AOTI 相对 FP32 的误差是 2.87e-4，eager AMP 自己是 3.69e-4、JIT 是
  3.72e-4——AOTI 并不更差。
- 但在整模型里逐层测同一输入的隔离误差：block 0 为 5.1e-4，**block 1 起跳到 2.2e-2**
  并在其后各层维持该量级。误差与层深无关，只与“是不是编译时那一块”有关。
- 根因：Inductor 在编译期把小常量折叠进生成代码。`to_gates.bias` 只有 8 个元素，被
  内联成 kernel 里的字面量，因此 `get_constant_fqns()` 只有 10 项，`to_gates.bias`
  根本无法注入；一份 package 复用到 12 个 block 时，24 个 runner 全部用 block 0 的
  gate bias（各 block 差值最大 0.427）。gate 是 attention 输出的 sigmoid 门，偏一点
  就是每层 2e-2 的相对误差。

修复：编译时设 `aot_inductor.use_runtime_constant_folding=True`，把折叠推迟到加载后
第一次运行，折叠结果以 `_FOLDED_CONST_*` 出现在常量表里，由注入的 checkpoint 张量派生
（注入端需跳过这些名字并用 `check_full_update=False`；重新注入会把折叠状态重置，
下次运行重算）。构建工具另加交叉校验：同一 package 注入 block 1 的权重后与 block 1 的
eager 输出比对，相对误差超过同块误差 3 倍即构建失败——缺陷版在该检查下 block 1/block 2
分别是 12.5×/25×。

修复后逐层隔离误差全部回到 3.6e-4~5.2e-4（与 eager 自身的 fp16 噪声同量级），
270 秒素材端到端：

| 变体 | 相对 eager cosine / SI-SDR | VAD segment | energy MAE / P99 / max |
| --- | ---: | ---: | ---: |
| 缺陷版（编译期折叠） | 0.999937689 / 39.04dB | 76（eager 81） | 2.27 / 32.30 / 61.27dB |
| 运行期折叠 | 0.999999985 / 75.21dB | 80 | 0.0341 / 0.4174 / 2.3524dB |
| 运行期折叠 + `emulate_precision_casts` | 0.999999986 / **75.41dB** | **81，起止全等** | 0.0334 / 0.4038 / 3.1983dB |
| 对照：E5 JIT Transformer | 0.999999986 / 75.66dB | 81，起止全等 | 0.0337 / 0.4064 / 2.8804dB |

只开运行期折叠时唯一的差异是 38.815–39.220s 这段 405ms、峰值约 −53dBFS 的极弱语音被
判成非语音；该窗口四个变体（FP32 / eager / JIT / AOTI）的能量轨迹逐帧只差 0.1~0.4dB，
属于阈值临界翻转，不是系统性劣化。`emulate_precision_casts=True` 让融合 kernel 按 eager
的方式舍入中间值，边界随之完全一致，因此构建工具把它设为默认。

质量已达到与 E5 JIT 相同的验收标准：波形优于 E0 的 AMP-vs-FP32 基线 71.4dB，VAD 边界
逐段相同。

#### 修复后的性能重测（2026-08-03，空闲机器，同一时段）

`emulate_precision_casts` 的开销单独取样：270 秒下开/关各 4 次，中位数 24.25s vs
24.14s（+0.5%），落在同配置自身的波动内（同配置极差约 2%，另有一次 26.06s 的孤立
离群值）。因此默认开启不付性能代价，下表的 AOTI 行均为默认构建（运行期折叠 +
`emulate_precision_casts`）。

| 270s / 4GB profile / 1 worker | 端到端 | 相对 eager | peak reserved |
| --- | ---: | ---: | ---: |
| eager AMP | 34.479 / 34.284s | 1.000× | 2.26GiB |
| JIT Transformer，磁盘缓存命中 | 38.213s | 0.900× | 1.88GiB |
| AOTI 无权重 package | 23.810 / 24.256 / 24.249 / 24.296s | **1.418×** | **1.84GiB** |

| 700s / 8GB profile / 2 worker | 端到端 | 相对 eager | peak reserved |
| --- | ---: | ---: | ---: |
| eager AMP | 77.239 / 77.351s | 1.000× | 2.94–3.13GiB |
| JIT Transformer | 65.713s | 1.176× | 2.40GiB |
| AOTI 无权重 package | 47.617 / 47.728s | **1.621×** | **1.96GiB** |

启动成本分项（270 秒进程，`--probe-compile-timing`）：

| 方案 | 包装 / 加载注入 | warmup 首次 / 复用 | separation 首次 / 稳态中位数 | 准备总计 |
| --- | ---: | ---: | ---: | ---: |
| JIT Transformer，缓存命中 | 2.136s 包装 | 8.817 / 0.373s | 3.654 / 0.357s | 13.878s |
| AOTI | 0.596s 加载注入 | 0.734 / 0.343s | 0.346 / **0.333s** | **1.000s** |

关键差异不只是稳态更快，而是 AOTI 没有 E5 那条“热缓存新进程仍要 13 秒准备、最坏 15–16
分钟才回本”的门槛：准备成本恒定约 1 秒，270 秒短任务也直接受益。两档的 AOTI 重复运行
输出逐样本一致（max abs error 0），700 秒 2 worker 下 24 个 runner 并发共享也没有出现
容器争用问题；700 秒输出相对 eager 的 cosine 为 0.999999989、SI-SDR 76.66dB。

结论：修复后 AOTI 在质量上与 JIT 同档、在性能和显存上明显更好（270 秒 1.42×、700 秒
1.62×，峰值 reserved 分别降 18.8% 与 33%）；E8/E9 把它进一步推到 270 秒 1.53×、
1400 秒 2 worker 1.88×。仍不进生产的唯一理由与 E5 相同——本机验证栈是 Torch 2.9 +
Triton 3.5，仓库正式依赖还是 Torch 2.8，且项目明令禁止直接升级（torchaudio 解码路径
回归）。生产代码和正式依赖不变；等 Torch 栈升级后可直接复用本节的构建工具与阈值数据。

### E7：编译路径上的 worker 阶梯（2026-08-03）

状态：结论明确——**1 个 worker 已经把这张卡吃满**，2 个是局部最优且只值约 9%，3/4 反而更慢。

`separator_worker_limit` 按每 300 秒音频加一个 worker，700 秒素材封顶 3 个，测不到 4 档，
因此本实验把 `clip700.ogg` 拼成 1400.072 秒（`tmp/clip1400.flac`）。块规划本身跟 worker 数
联动（块数 = 轮数 × worker 数），所以各档的块边界和 pad 冗余略有差别，这是设计固有的，
不做修正。每档两次，包为运行期折叠 + `emulate_precision_casts`：

| worker | AOTI wall | 相对 1 worker | AOTI peak reserved | eager wall | 相对 1 worker | eager peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 81.09 / 81.35s | 1.000× | 1.84GiB | 137.00s | 1.000× | 2.26GiB |
| 2 | 74.18 / 73.57s | **1.098×** | 1.99–2.12GiB | 130.71s | **1.048×** | 3.17GiB |
| 3 | 76.42 / 76.53s | 1.061× | 2.19–2.23GiB | 130.99s | 1.046× | 3.64GiB |
| 4 | 78.12 / 76.55s | 1.050× | 2.45–2.50GiB | 133.07s | 1.029× | 4.21GiB |

两条曲线形状一致，与 `gpu-profiles.md` 里 FP32 时代按“N 个并发任务”测出的 1.121×（2 实例）
是同一个结论：分离阶段的并发收益很小，2 实例之后转负。AOTI 把单 worker 的绝对时间压掉
41%，比任何加 worker 的收益都大一个量级；而且 AOTI 的显存几乎不随 worker 数增长
（1.84→2.50GiB），eager 是 2.26→4.21GiB。

对 profile 映射的含义：`budget // 4` 这条阶梯给 12/16GB 档分配 3/4 个分离实例，在这张卡上
只是白占显存。本轮**不改生产映射**——同一张卡上的实测不足以推广，且 worker 数与分块规划
绑定（块数 = 轮数 × worker 数），改了会改变产物边界。要扩容分离并发、或反过来收窄档位，
先在目标卡上复测本表。同一结论已记入
[`gpu-profiles.md`](gpu-profiles.md) 的「人声分离」节，那里是映射的权威位置。

### 权重不还给显卡（2026-09-01 修）

`run_vocal_separation` 返回后，**一整份 fp32 Roformer 权重（676 个张量，约 0.60 GiB）
仍留在显存里**。当时 pool 的 `_master` 已置 None、租约归零、accel 是 eager，
`gc.collect()`、`torch.cuda.empty_cache()` 乃至 `torch.compiler.reset()` 全做过——
说明 `audio-separator` 下游还有一条我们看不见的引用吊着那个 module。

**是按调用泄漏，不是按 worker。** 最初的读数（0.615 / 1.239 / 1.854 GiB）看着像随 worker
缩放，其实是三个档位在同一进程里顺序跑、逐次累加；强制 worker 数复测后，1 worker 与
3 worker 各自只增加约 0.62 GiB。批量是同进程线程，所以这是**每个文件漏一次**：十个文件
六 GiB，而 `entry` 档声称只要 3 GiB。

它还有第二个后果：VAD-ASR 阶段的 `peak_gmem` 其实主要是这份残留（实测 0.87 / 1.55 /
2.22 GiB = 残留 + 恒定 0.15），所以那个阶段的显存预算检查一直没在量 ASR。**顺带发现**
CT2 的分配对 torch 计数器不可见（`lang_redecode.py`），Whisper 解码器本来也没被它覆盖。

**修法走过一次弯路，值得记下来。** 第一版是 `module.to("cpu")`——把权重搬下 GPU。
显存读数确实好看了（0.615 → 0.009 GiB），但那条隐藏引用在主机上一样吊着它：实测四次连跑，
**活着的 CPU 张量每次精确 +0.595 GiB**，RSS 跟着涨。**显存泄漏被搬成了内存泄漏**，
而当时的回归测试只看 `torch.cuda.memory_allocated()`，看不见。评审抓到了这一点。

现行修法是**释放存储而不是搬家**：把每个 parameter/buffer 的 data 换成空张量
（`_evict_separator_weights`）。谁还握着这个 module，拿到的是一个参数为空的壳——无害，
因为没人复用它：pool 已经丢掉 `_master`，下一次 acquire 从 checkpoint 重建。
best-effort、绝不抛：它跑在产物已经落盘的拆解路径上。

实测四次连跑：CUDA 残留 0.009/0.009/0.010/0.010 GiB，**CPU 张量全 0.000**，
RSS 从第二次起持平（第一次那 1.2 GiB 是库与模型机制的一次性加载）。
回归测试 `test_the_separator_gives_its_weights_back` 连跑三次，**两侧**都设 0.15 GiB 上限
且不随次数放宽——只看显存会漏掉上面那种修法。

### E8：在编译路径上复测此前被否的 E1–E3（2026-08-03）

E1/E2 用 1400 秒素材、2 worker、交替取样（每臂 2 次，对照 4 次）；E3 用 270 秒单 worker。

| 实验 | 编译路径上的结果 | 结论 |
| --- | --- | --- |
| E1 延后 allocator/cache 清理 | 68.98 / 68.52s，对照 69.19 / 69.70 / 69.90 / 70.06s（−1.8%） | 维持否决 |
| E2 `torch.inference_mode()` | 69.93 / 69.95s，对照同上（−0.1%） | 维持否决 |
| E3 按轴选择 SDPA backend | **端到端 24.25s → 22.70s（+6.4%）** | **翻案，采纳** |

- E1 在 eager 上是噪声，在 AOTI 上变成一个可分辨但很小的收益（两次都低于全部四次对照）。
  它的否决理由本来就不只是没收益，而是要覆盖依赖的资源释放语义；1.8% 不足以换这个，
  何况 AOTI 的峰值只有 1.84–2.03GiB，省显存的动机也更弱。维持原判。
- E2 依然是纯噪声，`audio-separator` 内部的 `no_grad()` 已经覆盖了主要收益。
- E3 原来的否决理由是**运行期**选 backend 跨进程不稳定（两次 700 秒相差 6.7%）。AOTI 把
  选择固定在编译期，这个不稳定性就不存在了：`--attention-backend axis` 让 time 轴走
  cuDNN、frequency 轴走 memory-efficient，产物里确实变成
  `aoti_torch_cuda__scaled_dot_product_cudnn_attention`。四次 270 秒为
  22.79 / 22.62 / 22.70 / 22.70s，极差 0.8%，比 auto 的 24.25s 稳定地快 6.4%；
  单次 forward 中位数 332.9ms → 294.9ms（−11.4%）。质量不变：SI-SDR 75.38dB，
  VAD 81 段起止全等。

实现上有一个坑：导出后的图里留的是复合 `scaled_dot_product_attention`，**backend 是
Inductor lowering 时才选的**，所以 `sdpa_kernel(...)` 必须罩住 export 和 compile 两步；
只罩 export 时产物仍然是 efficient。同时 `Attend.flash_attn` 自己会开一个
`sdp_kernel` 覆盖全局 flag，追踪期间必须把该方法换掉。

### E9：把 band 级模块也编译进来（2026-08-03）

状态：采纳。`band_split` 与 `mask_estimator` 各是 62 个 band 的小 `Linear`/`RMSNorm`
列表，eager 下的时间几乎全花在 kernel launch 上，正好是编译能回收的部分。

单模块实测（8 秒 chunk 的真实输入形状，20 次取平均）：

| 模块 | 输入 | 参数数 | eager | AOTI | 加速 | 相对误差 | package |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `band_split` | `[1, 801, 4100]` fp32 | 186 | 19.17ms | **4.15ms** | 4.61× | 1.15e-5 | 2.5MiB |
| `mask_estimators.0` | `[1, 801, 62, 512]` fp32 | 248 | 16.91ms | 11.53ms | 1.47× | 3.96e-4 | 2.8MiB |

这两个模块每个模型只有一份、只调用一次，没有“换 block 校验”可做，改用**参数覆盖检查**：
包里注入的常量必须覆盖模块的全部 named_parameters，否则构建失败。这与 Transformer 的
交叉 block 校验拦的是同一类缺陷。

单次 forward 的分项（8 秒 chunk，逐模块 CUDA 同步计时）：

| 配置 | 一次 forward | 24 个 Transformer | 其余 |
| --- | ---: | ---: | ---: |
| eager AMP | 633.2ms | 580.6ms (91.7%) | 52.6ms |
| AOTI（auto backend） | 332.9ms | 280.0ms | 52.4ms |
| + 按轴 SDPA | 294.9ms | 260.2ms | 51.2ms |
| + band 级模块 | **278.0ms** | 260.2ms | 约 18ms |

“其余”里 `band_split` 占 19.3ms、`final_norm` 1.8ms，剩下约 32ms 是 STFT/iSTFT、
mask 拼接和层间 rearrange。

### 编译路径的最终配置（2026-08-03，torch 2.9.0）

⚠️ 本节是 **2.9.0** 上的配置与数字。生产已钉 2.11.0，那里
`emulate_precision_casts` **必须关闭**，方向与本节相反；现行配置与数字见 E11。

默认构建 = 运行期常量折叠 + `emulate_precision_casts` + `--attention-backend axis` +
`--targets all`（4 个 package：time / frequency / band_split / mask_estimator，共 26 个
runner，全部无权重）。

| 素材 / 并发 | eager AMP | AOTI 最终 | 相对 eager | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| 270s / 1 worker | 34.48 / 34.28s | 22.41 / 22.27 / 22.52 / 22.51s | **1.53×** | 2.26 → 1.84GiB |
| 1400s / 1 worker | 137.00s | 75.38 / 75.49s | **1.82×** | 2.26 → 1.84GiB |
| 1400s / 2 worker | 130.71s | 69.70 / 69.19s | **1.88×** | 3.17 → 1.84–2.03GiB |

质量：270 秒相对 eager cosine 0.999999983、SI-SDR 74.99dB，VAD 81 段起止逐段全等，
energy MAE 0.0337dB。加编译范围后 SI-SDR 从 75.41dB 略降到 74.99dB，仍远优于 E0 的
71.4dB 验收基线，且 VAD 判定完全一致。

### JIT 首次 forward 的验证与事务式安装（2026-08-21）

起因是一次现场事故：JIT 装好后第一个 block 在 `torch.compile` 的惰性编译里撞上托管
inductor 缓存缺 kernel `.json`（`FileNotFoundError`），异常从 forward 抛出，不在
`apply_acceleration` 的 try 块里，整个分离中止；同一份坏缓存让每次重跑都死在同一处，
用户只能手动改 eager。改动：

- `_apply_jit` 变成**事务**：每次替换前记录原模块（`OptimizedModule._orig_mod` 就是它），
  中途任何一次 `torch.compile` 抛错按记录回滚再重抛——此前第 N 次失败会留下 N-1 个已编译
  模块，模型半编译而 metadata 写 `eager`。
- `apply_acceleration` 返回 `AccelerationResult(requested, effective, rollback,
  fallback_reason)`，共享池在 `effective == "jit"` 时**多做一次 warm-up** 触发编译；失败交给
  `revert_jit`：按记录原地还原 + `torch._dynamo.reset()`、`rmtree` 托管 `inductor/`
  （用户自设 `TORCHINDUCTOR_CACHE_DIR` 时不动）、写 probe `jit=unavailable`、发
  `separator-jit-failed`，然后再 eager warm-up 一次继续。还原而不是重建 master：4GB 档放不下
  第二份权重，而 setattr 还原没有显存峰值。
- probe 改读改写，键按后端独立（`aoti`/`aoti_reason`/`jit`/`jit_reason` 各带 `_checked_at`）。
- **有意的取舍**：除 `torch.cuda.OutOfMemoryError` 外任何首次 forward 异常都持久化为
  `jit=unavailable`。Inductor/Triton 的失败形态太杂，按类型白名单很难写对；代价只是这个
  torch+GPU 组合分离慢一些，warning 的 action 给出删目录恢复的方法。

实测（2026-08-21，RTX 5060 Ti sm120，torch 2.11.0+cu128，`harvard.flac` 18s，三个独立进程）：
冷启动 `acquire(backend=jit)` 178.8s，编译确实发生在 warm-up 内而不是第一个 block；
随后分离 14.6s，峰值 reserved 2.26 GiB。删掉托管缓存里 kernel 的 `.json`（事故形态）**在这台
机器上没有复现失败**——这版 Triton 直接重编了缺的 kernel（acquire 17.3s），所以现场的
`FileNotFoundError` 不只是「文件没了」，还叠着别的条件（杀软实时删除或并发写）。改用注入：
第二次 warm-up 抛 `FileNotFoundError` → `effective='eager'`、metadata 三字段齐全、probe 写入
`jit=unavailable` 且 `aoti=ok` 保留、`inductor/` 被删、还原后的 eager 模型分离 3.9s 正常，
峰值 reserved 仍是 2.26 GiB（没有第二份权重）。

AOTI 的 `load_packages` 同样逐个替换 `target.forward`，中途失败此前也留半安装态；同日补成
事务式（记录已替换的模块，失败时 `del target.forward` 逐个还原、清掉 scratch 再重抛）。它是
即时加载、build 又在活模型上做，首次 forward 不另做验证。

### E10：33.6 分钟真实素材，以及 JIT 在同 scope 下的复测（2026-08-03）

素材 `assets/bilibili/BV1ojjc6MEAs.ogg`，2014.753 秒，8GB profile（时长阶梯算出 7 个
worker，被 profile 封到 2）。ASR 为 `python -m finesub.speech.recognition.cli.vad_asr`（large-v3-turbo + fw-refine），跑在 AOTI
的分离产物上。

| 阶段 | wall | 相对 eager | peak reserved |
| --- | ---: | ---: | ---: |
| 分离 eager AMP | 194.1s | 1.00× | 3.01GiB |
| 分离 JIT 冷缓存 | 256.7s | **0.76×** | 2.06GiB |
| 分离 JIT 热缓存 | 140.1s | 1.39× | 2.26GiB |
| 分离 AOTI | **101.2s** | **1.92×** | 2.03GiB |
| ASR（VAD 8.8s + 对齐 98.2s + 加载 5.4s） | 112.6s | — | — |

**分离不是配角**：eager 下它占 sep+ASR 的 63%，比 ASR 还贵；AOTI 后降到 47%，整个 GPU
段 306.7s → 213.8s（−30.3%）。调优先级应据此排——这是本文此前缺的那个数字。

JIT 这次用与 AOTI 默认构建**相同的 target 集合**（`--compile-scope all`：24 个 Transformer
加 band_split、mask_estimator，共 26 个模块）外加 `--axis-sdpa`，以排除「E5 只是编译范围不够」
这个解释。它不成立，差距的归因是：

```
gap total       38.93s
  来自准备      32.88s   （每进程 35.7s vs 2.8s）
  来自 forward   9.07s   （262 次 × 34.6ms）
```

**同 scope 下 JIT 的 kernel 只比 AOTI 慢 5.7%**，差距几乎全部是每进程重建计算图的固定
成本。磁盘缓存省掉的是编译，省不掉 Dynamo guard、graph 装载和缓存反序列化。因此这是
JIT 的结构性上限，不是调参能改善的。两者质量都过线（JIT SI-SDR 71.88dB、AOTI 72.53dB，
验收基线 71.4dB）。

**这同时给 E5 的结论划了适用边界**：E5 记的「热缓存长任务约 17% 收益」只在足够长的任务上
成立。按本次实测线性外推，JIT 的 35.7s 固定成本要约 **13.4 分钟**音频才回本，更短的任务
开了是负收益；而冷缓存（256.7s）比 eager 还慢 32%，即用户机器上的第一次运行必然倒退。

### 分发相关的两个产物事实（2026-08-03）

拆 `time.pt2` 得到的，决定了编译路径能以什么形式交付：

- **运行期不需要 triton，也不需要 MSVC。** 包内是 16 个已编译 `.cubin` 加一个 193KB 的
  `.wrapper.pyd`（已编译 host wrapper），另有 `.cpp` 仅作记录。加载只用 torch 和 CUDA
  driver，没有任何东西现场编译。给用户的新增运行期依赖为 0。
- **架构锁死是绝对的。** cubin 里只有 `sm_120`，**连 `compute_XX` PTX 都没有**——不是老卡
  更慢，是根本加载不了。根因见 `torch/_inductor/codecache.py` 里 `emit_multi_arch_kernel`
  分支的注释：Triton 只能为当前架构生成 PTX。开 `aot_inductor.emit_multi_arch_kernel=True`
  可加一份 `compute_120` PTX 换取向 sm_121+ 的前向兼容，但**永远无法向下**覆盖 sm_86/sm_89。

推论：预编译分发对用户零负担且可行，但构建侧**每个架构世代必须一台真机**（sm_75/86/89/120），
且 torch 每次升级都要全矩阵重编——loader 硬校验 `manifest["torch"]`。JIT 不能用来绕开这件事，
理由见上。

### E11：迁到 torch 2.11.0，以及 `emulate_precision_casts` 的反转（2026-08-03）

生产钉版从 `2.9.0+cu128` 改为 `2.11.0+cu128`（理由见 README_DEV「torch 版本范围」：
2.11 是最后一个有配套 torchaudio、且仍在 cu128 的版本）。关键行在新版本上重取。

**先修两处构建工具的不兼容**：2.11 把 `aot_inductor.package_constants_on_disk`(bool)
换成了 `package_constants_on_disk_format`(Optional[str])，旧写法直接 AttributeError；改为
依赖新默认值（None 即不落盘），无权重检查与交叉 block 校验仍然通过。

#### `emulate_precision_casts` 的正确取值是版本相关的

沿用 2.9 的默认（开）后质量明显退步。逐旋钮二分（270 秒素材，对同一 eager 参考）：

| 变体 | SI-SDR | VAD（eager 81 段） |
| --- | ---: | --- |
| 默认：emulate + axis + all | 72.65dB | 81 段，一处起点差 0.010s |
| `--attention-backend auto` | 72.64dB | **82 段** |
| **`--no-emulate-precision-casts`** | **75.49dB** | **81 段，起止全等** |
| `--targets transformers` | 75.70dB | 81 段，但边界差 **9.04 / 10.94s** |

- **`emulate_precision_casts` 在 2.11 上是唯一的退化来源**：关掉同时赢下两个口径。在 2.9
  上它的作用相反——开着才能保住那个边缘 VAD 段（E6）。**升级 torch 必须重测这个开关，
  不能沿用。** 已把默认改为关闭，并把这段结论写进该 flag 的 help。
- 模块级编译质量在 2.11 反而更好（time 轴同块误差 2.762e-4 → 6.341e-5），所以退化与
  codegen 无关。eager 自身在 2.11 上逐位可复现，指标可信。
- **`--targets transformers` 是「单一分数会骗人」的实例**：SI-SDR 最高，下游边界却差了
  九到十一秒。band 级模块必须保留。
- `--attention-backend axis` 在 2.11 不再只是性能项：`auto` 会多出一段。

#### 2.11 上的最终数字

配置 = 运行期常量折叠 + `--attention-backend axis` + `--targets all`，**emulate 关闭**。

| 素材 / 并发 | eager AMP | AOTI | 相对 eager | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| 270s / 1 worker | 36.51s | 24.65s | **1.481×** | 2.26 → 1.84GiB |
| 2015s / 2 worker | 198.89s | 104.96s | **1.895×** | 3.01 → 2.08GiB |
| 2015s / 2 worker，JIT 热缓存 | 198.89s | 144.01s | 1.381× | 3.01 → 2.06GiB |
| 2015s / 2 worker，JIT 冷缓存 | 198.89s | 268.06s | 0.742× | 3.01 → 2.06GiB |

比值与 2.9 基本一致（1.53→1.481、1.92→1.895），绝对速度慢 2–4%。

一致性：**270 秒 VAD 81 段起止全等**；2015 秒 675 vs 673，多出的是
`297.07–297.56s`（0.48 秒，峰值 −43.3dBFS、RMS −53.8dBFS）——极弱片段跨过 VAD 判定线，
与 E6 记录的那次（405ms、约 −53dBFS）同一指纹。

#### 三档效率重测与一处测量缺陷（2026-08-25，torch 2.11.0+cu128）

全部协议、逐项数据、环境与复现见
[`tools/separator_accel_bench/README.md`](../tools/separator_accel_bench/README.md)。这里只留结论。

**一处会改变旧数字读法的缺陷：** `_TimedModel` 此前在编译臂无条件安装、eager 臂从不安装。
它每次 forward 前后的 `cuda.synchronize()` 会把 H2D 与上一块的 CPU overlap-add 串行化——
同一条 AOTI 路径带它 32.64s、不带 29.35s（34 块差 3.3s）。**三档从来没在同一把尺子上量过，
且编译臂被系统性量慢。** 已统一收到 `--time-forwards` 下、默认关；带仪器只做归因，报吞吐用
不带的。**E5–E11 的编译臂绝对值都带着这份开销。**

吞吐（无仪器，热缓存）：

| 素材 / 并发 | eager | `torch.compile` 热 | AOTI |
| --- | ---: | ---: | ---: |
| 270s / 1 worker | 34.35s | 46.45s（**0.739×**） | **28.86s（1.190×）** |
| 2015s / 2 worker | 210.37s | 149.32s（1.409×） | **115.59s（1.820×）** |

- **模型加载恒定 3.7–4.1s**，与档位无关。
- **kernel 层加速只有 1.9–2.1×**（forward 中位 747→397→353ms、1474→809→699ms），而 AOTI 相对
  `torch.compile` 在 kernel 上只快约 10%。**AOTI 的优势主要不在 kernel，而在每进程恢复成本**：
  9.1–9.8s 对 21.8–32.4s。
- **热缓存的 `torch.compile` 每进程仍要付 22–32 秒**重建 dynamo/guard，且要付两次（预热一次、
  真正第一块又一次）。于是它在 270 秒素材上**净亏**。冷编译 193.68s 作参照。
- **盈亏平衡点**：单 worker 下每块省 350ms（JIT）/ 394ms（AOTI），一块 8 秒音频。JIT 要 ~63 块
  即**约 8.4 分钟音频**才回本，AOTI 只要 ~23 块即**约 3 分钟**。这解释了 `select_backend`
  为什么给 JIT 设 `JIT_MIN_DURATION_SEC` 门槛而 AOTI 无条件优先。
- **恢复阶段几乎不用 GPU 算力**（利用率停在桌面基线，整段无一采样超过 50%），但占约 1 GiB
  显存与 CUDA 上下文。推论：**`torch.compile` 那 22–32 秒不会因为换更快的显卡而变短。**
- **「先载进内存、就绪了再占显存」对编译产物结构性不可行**（inductor 缓存键含设备；AOTI 包为
  sm120 编译死且绑已在显存的权重指针）。能挪的只有约 3.4s 的设备中立部分，而且**今天不值钱**：
  闸门只有分离器与 vad-asr 两个获取者、严格顺序、asr bin 恒为 1 worker，**从不争用**；
  闸门管的又是显存不是 SM，挪出去正好废掉它的目的。

**270 秒那一格与 E11 对不上**（本轮 AOTI 1.190×，E11 记 1.481×，而 eager 侧本轮更快），
本机不满足协议的「空闲桌面环境」但那解释不了方向。**存疑，别拿它当基线。** 长素材复现良好
（AOTI 1.820× vs 1.895×，JIT 1.409× vs 1.381×）。

#### 关于怎么读这些数字

- **SI-SDR 衡量的是与 eager 的一致性，不是质量。** eager 自己也只是 FP32 的近似，分数高
  只说明忠实复现了 eager（包括它的误差）。E0 那条 71.4dB 是 AMP-vs-FP32 的一致性，作为
  门槛只意味着「这个量级的偏差以前被接受过一次」，是先例而非质量阈值，且与
  AOTI-vs-eager 是两对不同的东西。
- **一致性是一张免费的证明，不是及格线。** 拿到了就可以确定没有严重质量下滑、无需进一步
  评估；没拿到不代表变差，只说明这张证明不可用，要么接受要么另行刻画。
  这是**项目级原则**而不是本文的局部约定，见 [`README_DEV.md`](../README_DEV.md#开发原则)
  「一致性是一张证明，不是及格线」。所以本文里凡是写「换 X 就拿不到 VAD 逐段相等这张证明」
  的地方，一律是**在定价，不是在否决**——它抬高的是验收成本，不是判这条路走不通。
  ⚠ 但那条原则**有作用域**：它适用于**有意改变数值路径**的改动（换 checkpoint、换采样率、
  编译/量化路径）。本文里两类东西**不在其内、仍按精确一致验收**：
  声称「与 eager 语义等价」的实现改动（E13/E14 那种「只是不再落盘一次」的重构），
  以及确定性/幂等类契约（如「eager 自身在 2.11 上逐位可复现」这条使指标可信的前提）。
  另外，**放弃一致性证明时必须预先写明替代指标与门槛**——本文 E12 的
  「人工时间轴 + 词准确率」就是这样一套，不是事后补的说法。
- **全局 SI-SDR 会稀释局部失效**，必须配合定位。2015 秒这份的逐秒 SNR：中位 75.09dB、
  最低 9.54dB——但按响度分层后，**有信号的 1768 秒里最差一秒仍有 35.8dB**（p05 63.2dB），
  那些 0dB 的秒参考 RMS 是 −300dBFS 的数字静音，是分母为零的产物。没有出现「某个 chunk
  算坏」的特征。**逐秒诊断必须按响度分层看，否则静音区会主导「最差」榜单。**

参考：[PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)、
[PyTorch regional AOT compilation](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)、
[NVIDIA TensorRT pip 安装](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)、
[Torch-TensorRT 安装](https://docs.pytorch.org/TensorRT/getting_started/installation.html)、
[`triton-windows` 版本对应](https://github.com/triton-lang/triton-windows)。

### 块产物固定为 FLAC（2026-08-18）

不是性能实验，是一处白丢的质量。分块路径把每块交给 audio-separator 写进 tmpdir，再由
`_append_separated_block` 读回来追加进最终文件——而块的格式此前**跟随交付格式**
（当时的 `output_format_for(output_path)`，默认 ogg）。于是人声轨被 Vorbis 编了两次：一次在块上，
一次在追加时，而块文件几秒后就删了。

改法是一个常量：块固定 flac，最终产物格式仍由输出后缀决定（当时叫 `BLOCK_OUTPUT_FORMAT`；
下一节把合并也收进同一个常量后改名为 `MERGE_FORMAT`，格式判定改为 `output_mode_for`）。
`block_seconds <= 0` 的单发路径当时不经过 tmpdir、直接写交付文件，因此不受本节改动影响；
下一节把合并收进同一条链之后它也一并走了（ASR 模式下先写 `vocal_merge_*` 再编码）。

**输出一致性影响**：交付的 `-vocal.ogg` 内容会变——少一代有损，比旧版更接近模型输出。
时间轴不变：分块规划、pad、trim、追加顺序都没动，帧数与采样率逐块相同。既有 `-vocal.ogg`
不必重跑（它本来就因块边界随 worker 数移动而不可复现，见本文与 `gpu-profiles.md`）。

**精度**：块写出走的是 audio-separator 的 pydub/ffmpeg 路径（`use_soundfile` 只对 >1h
素材启用），该路径**无条件先转 int16** 再按扩展名 export，`flac` 还会带上 `-sample_fmt s16`。
块输入 wav 本来就是 PCM_16，所以块往返在 16-bit 上完全无损，不引入新的量化台阶。代价是
tmpdir 峰值变大——FLAC 约为 Vorbis 的 3–5 倍；块输入 wav 不变。

**库支持不是推断**：424a1c8（2026-07-23）把交付格式从 FLAC 改成 OGG 之前，本项目就是以
`flac` 作为 `_build_separator` 的 `output_format` 跑生产的，代码形状与今天相同。

**为什么不能直接裁剪拼接块文件、连解码都省掉**：块之间有 `pad_seconds` 的重叠，追加前要
按**样本**裁掉（`int(round(pad * sr))` 帧）。Vorbis 是重叠变换，最小可切单位是包而不是样本，
按包切会把时间轴挪掉几十毫秒——而块边界严丝合缝正是 pad+trim 的全部目的。Ogg 的 chained
stream 拼接虽然合法，但下游 `soundfile` 对链式流的处理不可靠（可能只读到第一段）。所以
「解码 → 裁剪 → 重编」这一趟省不掉；能省的只有块**这一代**的有损，也就是本节的改动。

#### 实测（2026-08-18）

素材 `assets/bilibili/BV1kYLR6AEXv.mp4` 的 60–120 秒，44.1 kHz 立体声 PCM_16；
`block_seconds=20`、pad 默认 10s、8GB profile、时长阶梯给 1 worker → 3 块；accel 走 aoti、
AMP 开。四个 run 只差块格式与交付格式，其余一切固定，因此差异只来自块往返。每 run 约 12 秒。

| run | 块格式 | 交付 | 帧数 | lag | SNR vs A | 体积 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | flac | flac | 2645995 | 0 | —（无损参考） | 2569 KiB |
| B | ogg | flac | 2645995 | 0 | 22.10 dB | 2654 KiB |
| C | flac | ogg | 2645995 | 0 | **23.38 dB** | 719 KiB |
| D | ogg | ogg | 2645995 | 0 | 19.94 dB | 690 KiB |

- **时间轴零影响**：四者帧数逐一相同，互相关最优整数 lag 全为 0。这是最重要的一条——
  `-aligned.json` 的时间语义不因它移动。
- **生产口径（D→C）提升 3.44 dB**。分频带看（对 A 的 SNR）：0–1k `29.2→33.6`、
  1–4k `19.6→23.2`、4–8k `12.8→15.9`、8–16k `6.5→9.4`。ASR 只用 16 kHz 以下，
  提升正落在这几档。
- **C 反而好过 B**：交付那一代由 libsndfile 写，比 pydub 调 ffmpeg 的 ogg 默认档更保守，
  所以「块上那一代」比「交付那一代」更贵——这也解释了为什么只改块格式就能拿到 3.4 dB。
- **B.flac 比 A.flac 大**（2654 vs 2569 KiB）：Vorbis 解码后的信号更难压。同理 C.ogg 比
  D.ogg 大 4%——干净输入在同 VBR 档下要更多比特。这是本次改动唯一的成本。
- 想要 0 代有损就是 run A（交付 FLAC），代价 3.5× 体积——那条后来成了正式的无损交付模式，见下一节。

**验证**：`test_vocal_separation_pool.py::test_blocks_are_separated_as_flac_whatever_the_delivered_format`
断言块一律按 flac 申请、交付文件仍是 OGG；把常量改回 `"ogg"` 该测试会红（已实测）。
**未做**：没有把 C/D 继续跑 VAD-ASR 比对转写差异——频带数据说明差异落在 ASR 带内，但
「转写会不会因此改变」没有测。口径已定：现成 pipeline 跑同素材 A/B，比 stable JSON 的
词级 diff + 异常计数。**挂在下次要动分离器交付格式时做，不单独排期。**


### 交付分两模式，ASR 那一路改成 16 kHz 单声道（2026-08-18）

上一节把块的那一代有损去掉之后，剩下的一代（交付编码）暴露出一个更大的问题：**码率花错了
地方**。读 `-vocal.ogg` 的每一个下游——energy VAD（`TARGET_SR = 16000`）、whisper、Qwen
裁判——第一件事都是下混单声道 + 重采样到 16 kHz。而我们交付的是 44.1 kHz 立体声，第二声道
和 8 kHz 以上的比特全部被下一阶段丢掉。

同一段素材（上一节那 60 秒），在**下混重采样之后**、也就是 ASR 真正看到的信号上测：

| 交付形态 | kbps | 体积 | SNR vs 无损分离 |
| --- | ---: | ---: | ---: |
| 44.1k 立体声，libsndfile 默认（实测等于 `compression_level=0.6`） | 98.2 | 719 KiB | 23.97 dB |
| 44.1k 立体声，`compression_level=0.4` | 143.2 | — | 28.64 dB |
| 44.1k 立体声，`compression_level=0.2` | 189.0 | — | 33.17 dB |
| **16k 单声道，`compression_level=0.2`（现行）** | **48.5** | **355 KiB** | **28.58 dB** |
| 16k 单声道，默认档 | 26.5 | — | 21.75 dB |

现行档位是端到端实跑复核过的：**355 KiB / 28.58 dB**，即体积减半、带内 +4.6 dB。同样的
28.6 dB 若坚持 44.1k 立体声要花 143 kbps，是三倍的代价。

**两个模式**（`output_mode_for`，由输出后缀决定，其它后缀报错）：

- `.ogg` —— ASR 交付：16 kHz 单声道 Vorbis，`ASR_VORBIS_COMPRESSION = 0.2`。
- `.flac` —— 无损交付：模型自己的采样率与声道数，全链零有损。试听、测量、实验用这个。

两者共用同一条无损合并链；ASR 交付只是在合并完成后多一次流式扫描。合并产物在无损模式下
**就是**交付文件，在 ASR 模式下落在 `vocal_merge_*` 临时目录、编码完即删。

#### 接缝：为什么窗口式重采样在这里是安全的

整轨一次性重采样放不下（两小时 44.1k 立体声 float32 > 1GB），而窗口式重采样正是接缝的来源：
滤波器支撑跑出窗口边界时，边界值是被"编"出来的。`stream_asr_frames` 用两件事消掉它：

1. **窗口是比率步长的整数倍**。44100→16000 约分为 441:160，每 441 个源帧恰好 160 个目标帧，
   因此每个窗口都产出整数个输出帧，长轨上不会累积舍入漂移。末窗是唯一非整数倍的，向上取整——
   这正是整轨重采样对同样余数的做法。实测 2645995 帧 → 959999 帧，与整轨公式逐帧一致。
2. **每个窗口两侧都带真实音频作为上下文，算完丢弃**。上下文宽 20 步（0.2 秒），远超滤波器
   支撑，因此**保留下来的每个样本都是在滤波器被填满的情况下算出的**，与整轨重采样的结果相同。

这不是论证，是被测试钉住的：`test_windowed_resampling_matches_a_whole_file_resample` 把窗口
缩到 20 步（让 2 秒素材跨 20 多个窗口、保持生产的上下文宽度），断言与整轨重采样的最大逐样本
差 < 1e-6。把 `_RESAMPLE_CONTEXT_STEPS` 改成 0，该差立刻变成 **0.155**——那就是每个窗口边界
上的咔哒声。

**顺带**：ASR 交付本身就是 16 kHz，`energy.py` 与 `transcribe.py` 里的 `resample_if_needed`
在这条路上退化为 no-op，那两处**按块重采样后 concat**（无重叠）的既有做法也就不再作用于
生产音轨。


### E12：让分离器在更低采样率上工作（32 / 22.05 / 16 kHz，2026-08-24/25）

全部协议、五素材逐项数据、人工时间轴裁判与复现见
[`tools/separator_rate/README.md`](../tools/separator_rate/README.md)。这里只留决策与结构性事实。

状态：**三档全部不采纳，理由分三种，别混着记：**

- **16 kHz** 省 1.82×，但会**整段抹掉人声**且随素材开盲盒——生产条件下最差一例丢掉 476 个
  有声秒里的 98 个、最长空洞 13 秒，漏掉的时间里 70.5% 是真语音。**硬否决。**
- **22.05 kHz** 省 1.52×（渐近 2.0×），**没有**那个失败模式：不丢内容、不漏真语音、
  边界对齐是三臂里最准的。在生产真正的输入上复验后词准确率代价也**不再可判定**
  （+0.014，符号 3/5，与空白对照区分不开）。唯一还越界的是 VAD 多 admit 约 2.7 倍时间。
  **不做是因为收益不成比例（只折合完整 run 的约 4%），不是因为它坏。**
- **32 kHz** 省 1.26×、代价 +0.014——**同一条代价曲线上的半剂量，被 22.05 kHz 支配**。

#### 两件常被混为一谈的事

- **「喂低采样率的输入」一秒都省不下来。** `audio-separator` 用
  `librosa.load(mix, sr=self.sample_rate)` 读，**任何输入都会被重采样到它自己的率**。
  实测同素材，16 kHz 单声道 ogg 与原生 44.1 kHz 立体声 wav，43.2s vs 44.0s，在噪声内。
- **只有「让模型在低采样率上工作」能省。** `chunk_size = stft_hop_length × (dim_t − 1)
  = 441 × 800 = 352800`，单位是**采样点**，与 `self.sample_rate` 无关；`overlap=8` 让
  `desired_step` 等于 `chunk_size`，块间无重叠。一块覆盖 `352800 / sample_rate` 秒，
  于是每秒音频的块数按比例下降——加速只能从这里来，没有任何算子级好处
  （稳态每块成本各档相同，都是 0.33 s/块）。

#### 为什么低采样率会整段抹掉人声

band split 的 `freqs_per_bands` 按 **STFT bin 序号**切，最细的 24 个 band 各占 2 个 bin：
44.1k 下前 48 个 bin 覆盖 0–1033 Hz，16k 下同样 48 个 bin 只覆盖 0–375 Hz。权重学到的
「人声落在哪几个 band」整体错位约 1.46 个八度，模型看到的等价于一段被慢放的音频；
错得越多，把人声整段误判成伴奏的概率越大。

**放大 16k 损伤的是声道而非带宽**：同素材同模型率 16000，原生立体声丢 98s、原生单声道
（带宽相同）丢 181s、旧 16k ogg 丢 205s——78% 的差距来自声道。BS-Roformer 是 `stereo: true`
训练的，声道差是它判定人声的一条线索。**但 44.1k 下单声道丢 0 秒**：这条线索只在 band
已错位时才承重，所以那批 16 kHz 单声道 ogg 缓存**没有在损害生产分离质量**
（44.1k 臂上与原生输入的 CER 均值差 +0.002）。

#### 22.05 kHz 之外没有「刚好对齐」的率——可以证明

band-group 边界落在 bin `{48, 96, 192, 384, 768}`，是 48 的**严格 2 的幂**倍。率 r 下边界
bin `k` 对应 `k·r/2048` Hz，所以要让 r 的边界频率集合与 44.1 kHz 重合，必须
`r = 44100 × 2^n`——解只有 **22050 / 11025 / 5512.5**，**(22050, 44100) 区间内一个都没有**。
穷举该区间内所有 `r = 44100·k_j/k_i` 只有三个候选（33042.7 / 37800 / 38549.9），各只对上
**1/7** 条边界，且靠的是顶部 `128+129` 那个**本来就破坏八度规律**的收尾切分；22050 对上 4/7，
是四条八度边界整体下移一组。

**还有一层：两个约束方向相反。** 模型的细分辨率区（bin 0–192）覆盖到 `0.09375·r` Hz，
要让语音 F3（~3.5 kHz）留在细区里需要 **r ≥ 37333 Hz**，那只买到 1.18×。于是「保住共振峰
细分辨率」的区间是 r ≥ 37.3 kHz，「块数削减值得一提」的区间是 r ≤ 22.05 kHz，
**两者不重叠，中间是空的**。22.05 kHz 像 magic number 只因为它是唯一同时「对齐」且
「收益够大」的点，代价是把 F3 挤出细分辨率区。实测里 32k 与 29.4k 的差别小于空白对照自己的
散布，本来也分不出先后。

#### 顺带订正一条容易搞反的直觉

**ASR 交付是 16 kHz，分离器不是。** 下游只用 8 kHz 以下并不意味着分离可以在 16 kHz 上做——
分离恰恰要靠全带宽才能把人声和伴奏拆开，而交付阶段的降采样发生在拆分**之后**
（见上一节「交付分两模式」）。

#### 重开这条的条件

如果分离 wall time 变成真瓶颈（batch 吞吐、或长素材占比大幅上升），22.05 kHz 是唯一还在
桌面上的档——生产开关 `--separator-rate` 见 [README_DEV](../README_DEV.md)「分离器的工作采样率」。届时该补的是：更多素材把 +0.014
的置信区间收紧；用人工时间轴之外的**人工日文转写**复核词准确率（本轮只有中文译文，判不了词）。


### E13：把「非 forward 的那 13 秒」拆开（2026-08-27，CPU only）

状态：**推翻队列第 4/5 条的前提**。专用 demix runner 仍值得做，但理由从「非 GPU 部分是大头」
换成了一条小得多、也具体得多的账。

E9 之后队列里写着「270 秒任务里 forward 只占 9.5s / 22.4s，非 GPU 部分现在是大头，这条的
优先级比之前高」。那句话把**「不是 forward」等同于「每秒音频都要付、换个 runner 就能省」**。
本节测的就是这个等号。协议、脚本与复现见
[`tools/separator_accel_bench/cpu_chain.py`](../tools/separator_accel_bench/cpu_chain.py)：
forward 之外的每一步都是**形状决定**的，所以不加载 checkpoint、不用 GPU，在生产形状的合成
音频上原样重放即可（torch 2.9.0+cu128、librosa 0.11.0、8 线程；此处只量 CPU 与磁盘，
torch 版本不影响结论）。

| 分组 | 270s | 600s（一个块 core） | 随时长 |
| --- | ---: | ---: | --- |
| input：`sf.write` 块 wav + `prepare_mix` | 1.274s | 1.628s | **大部分不随** |
| current：现行的 forward 后链路 | 1.608s | 3.675s | 线性 |
| proposed：单 stem + 直写 + 内存交接 | 0.954s | 2.128s | 线性 |

- **`librosa.load` 的 1.0s 是每进程一次性的**（懒加载 + numba），稳态 0.044s/270s，与
  `sf.read` 的 0.035s 同量级。它不是每块都要付的成本，量它一次就够。
- **整条 CPU/IO 链只有音频时长的约 0.6%**：270 秒付 1.6s，600 秒付 3.7s。那「13 秒」里
  真正随时长增长的部分不到 2 秒，其余是 torch import、模型加载（恒定 3.7–4.1s，见上一节）、
  AOTI 加载注入（约 1.0s）、warm-up 与 librosa 首调用——**全是固定启动成本，换 runner 一秒
  都省不掉**。
- 链路里**最贵的两项都不是计算，是编解码**：块经 pydub/ffmpeg 编成 FLAC（600s 付 0.887s），
  再被 `_append_separated_block` 解回来（含在 2.391s 里）。双 stem 缓冲、window/counter、
  全尺寸除法三项加起来 600s 只有 0.24s——**队列第 4 条点名的那件事本身几乎不值钱**，
  值钱的是它旁边那趟往返。

#### 合并容器：ASR 模式下 FLAC 是白付的

`MERGE_FORMAT = "flac"` 对两种交付模式一视同仁，但**无损模式下合并产物就是交付文件，
ASR 模式下它只是 `vocal_merge_*` 里的临时文件**，编完即被重扫成 16k 单声道 ogg。
同一份 600 秒立体声：

| 容器 | 写 | 读回 | 体积 |
| --- | ---: | ---: | ---: |
| FLAC / PCM_16 | 1.820s | 0.338s | 81.0 MiB |
| WAV / PCM_16 | **0.228s** | 0.085s | 100.9 MiB |
| WAV / FLOAT | 0.276s | 0.055s | 201.9 MiB |

ASR 模式把临时合并容器换成 WAV，每 600 秒省 1.6s，代价是临时目录峰值 +25%。

#### 这条路线的总账

三项合起来（去掉块的编解码往返、单 stem 缓冲、ASR 模式临时合并改 WAV）约为**音频时长的
0.52%**。按 E11/2026-08-25 的绝对值折算：2015 秒 / 2 worker 从 115.6s 降到约 105s
（**1.10×**），270 秒 / 1 worker 从约 24.6s 降到约 23.2s（**1.06×**）。

**顺带是一处质量改善而不是退化**：现行链路把 float32 量化成 int16 **两次**（块 FLAC 一次、
合并 FLAC 一次），去掉往返就只剩一次。这与 2026-08-18「块产物固定为 FLAC」是同一类改动，
交付内容会变、时间轴不变。

**结论**：值得做，但它是 1.1× 而不是队列暗示的量级；**分离阶段的钱仍然压倒性地在 forward 里**
（2015 秒 / 2 worker：forward 约 91s / 115.6s）。要越过 1.1×，只能动模型本身——见下一节。

### E14：forward 的 roofline 归因，以及它给换模型方案定的价（2026-08-27，RTX 5070 Ti）

状态：**结论有决策力**。这是本文第一次回答「forward 这 167ms 是由什么构成的」，而不是
只报它有多长。工具见
[`tools/separator_accel_bench/roofline.py`](../tools/separator_accel_bench/roofline.py)：
FLOPs 由 `FlopCounterMode` 在 **eager** 上精确统计（AOTI runner 不走 ATen dispatch，装上之后
counter 看到的是空图——**必须先数后装**），时间由 `torch.profiler` 的 kernel self time 统计，
两者都在生产形状的单块 `[1, 2, 352800]`（8 秒）上取。

**一块的 FLOPs：8.696 TFLOP，其中注意力（SDPA）只有 1.053 TFLOP = 12.1%。**

| kernel 类别 | eager | 占比 | AOTI | 占比 |
| --- | ---: | ---: | ---: | ---: |
| GEMM（cuBLAS/cutlass f16 tensorop） | 98.15ms | 30.6% | **97.53ms** | **60.4%** |
| elementwise / 布局搬运 | 191.08ms | 59.5% | 4.31ms | 2.7% |
| triton 融合（AOTI 的融合 epilogue） | — | — | 39.58ms | 24.5% |
| 注意力 | 28.93ms | 9.0% | 20.01ms | 12.4% |
| forward wall | **353.9ms** | | **167.4ms** | |
| wall − kernel 之和（launch/空转） | +32.8ms | 9% | **+5.9ms** | **4%** |

四条读出来的结论：

- **AOTI 的全部收益是把 191ms 的 elementwise 融成 40ms，外加换 attention backend
  （28.9→20.0ms）。GEMM 一秒没动：98.15 → 97.53ms。** 编译做的从来不是让矩阵乘更快。
- **GEMM 现在是编译路径的 60%**，且已经跑在 cuBLAS 的 cutlass tensorop 上。非注意力的
  7.643 TFLOP 落在约 97.5ms 里 ≈ **78 TFLOP/s**，整个 forward ≈ 52 TFLOP/s。这条已经贴着
  这张卡的 FP16 tensor 天花板，**没有编译器能再拿走它**。
- **launch/空转只有 4%**：CUDA graph、异步化、减少 kernel 数这一类工作在这里价值为零。
- **注意力占 12.1% 的 FLOPs 与 12.4% 的时间——两个口径互相印证。**

#### 这条给「换注意力」的方案定了价

Windowed Sink Attention 的 **44.5× 是注意力 FLOPs**。在本模型本分块下把注意力压到零，
上限也只有 `1 / (1 − 0.124)` = **1.14×**。原因是结构性的：**WSA 针对的是整首歌不分块推理
（时间轴上万帧、注意力二次项压倒一切），而我们按 801 帧一块切**，注意力在这个长度上本来
就不贵。**这条路对本项目基本关闭**，不必再为它付人工验收成本。

#### `max-autotune` 在新卡上不再被拒绝

E5 记着「`max-autotune` 在这张卡上被 Inductor 以 SM 不足拒绝」。新卡 70 SM，**它通过了**，
且赢了 cuBLAS。单个 transformer block（`torch.compile`，生产形状）：

| block | eager | Inductor 默认 | `max-autotune-no-cudagraphs` | 相对默认 |
| --- | ---: | ---: | ---: | ---: |
| time `[62, 801, 512]` | 14.419ms | 8.411ms | **7.892ms** | **+6.6%** |
| freq `[801, 62, 512]` | 12.520ms | 6.601ms | **6.088ms** | **+8.4%** |

autotune 日志里可以看到为什么：FFN 下投影 `addmm(49662×512, 49662×2048, 2048×512)` 上
**`triton_mm` 1.1026ms 赢过 `bias_addmm`（cuBLAS）1.1964ms**，快 8%，而且 Triton 模板还允许
把 epilogue 融进 GEMM——正好吃掉上表那 24.5%。

#### 落地：`max_autotune` 进 AOTI 构建（2026-08-27，已采纳）

`build_packages(max_autotune=...)` 取 `off` / `transformers`（默认）/ `all`，写进 manifest；
`BUILD_FORMAT` 从 `1` 升到 `2`，所以既有 package 目录会被整体忽略、自动重建——这正是那个常量
存在的理由。

**必须按 target 分组开关，不能一个 bool 开到底。** 三个变体的构建期校验误差：

| package | off | transformers | all |
| --- | ---: | ---: | ---: |
| time（fp16） | 4.060e-04 | 4.062e-04 | 4.060e-04 |
| frequency（fp16） | 4.439e-04 | 4.431e-04 | 4.431e-04 |
| `band_split`（fp32） | 9.828e-06 | 9.004e-06 | **5.138e-05** |
| `mask_estimator`（fp32） | 3.184e-06 | 4.091e-06 | **5.194e-04** |

**两个 Transformer 轴一动不动，退化全部来自两个 fp32 band 模块**（5× 与 163×），端到端表现为
SI-SDR 74.73 → 71.34dB。因此默认只对 Transformer 开。

270 秒素材（1 worker，`--reference` 取同轮 eager），每臂 3 次交替取样：

| 臂 | 端到端 | forward 稳态中位 | warmup 首次 forward | SI-SDR vs eager | VAD |
| --- | ---: | ---: | ---: | ---: | --- |
| off | 17.31/17.40/17.42s | 166.9ms | 3.52s | 74.73dB | 72 段起止全等 |
| **transformers** | 17.52/17.59/17.60s | **155.5ms（+6.8%）** | **4.07s（+0.55s）** | **74.71dB** | **72 段起止全等** |
| all | 17.72/17.54/17.51s | 153.1ms（+8.3%） | 4.11s | 71.34dB | 72 段起止全等 |

**这里有一条不能只看 forward 的教训**：forward 确实快了 6.8%，但**首次 forward 多付 0.55 秒**
（Triton 模板的 cubin 更大，package 0.61→0.93 MiB），而 270 秒任务的 forward 总共才 34 ×
0.167 = 5.7 秒。两者相抵，端到端**反而慢 0.19 秒**——与实测完全吻合。
**回本点 = 550ms / 11.4ms ≈ 48 块 ≈ 6.4 分钟音频。**

2015 秒真实素材（8GB profile / 2 worker），两次：

| 臂 | 端到端 |
| --- | ---: |
| off | 60.36 / 60.12s |
| **transformers** | **58.18 / 57.83s（1.039×）** |

**结论**：默认开在 Transformer 上。长素材 +3.7%，短素材（<6.4 分钟）付最多 0.55 秒。
质量两个口径都过：SI-SDR 与 off 相同（74.71 vs 74.73），VAD 72 段起止逐段全等。
`all` 不采纳——多出的 1.5% forward 换不来那 3.4dB，而 SI-SDR 是本文所有采纳项都保住的
那张免费证明。构建时间从 72s 涨到约 110s（`all` 是 151s），首次运行自建包的成本相应变长。

### E15：块在内存里交接，不再往返磁盘（2026-08-27，已采纳）

状态：采纳。这是 E13 那条账的落地，外加它没算到的两项。实现见
[`separator/demix.py`](../src/finesub/speech/preprocessing/separator/demix.py)。

`audio-separator` 的入口是文件到文件：`librosa.load` 读块、分离、pydub 调 ffmpeg 写 stem，
finesub 下一行再把它解回来追加进合并轨。现在块由 `demix.separate_waveform` 直接交接：

- 块不再落盘（少一次 WAV 写、一次 librosa 解码、一次 int16 转换、一次 FLAC 编码、一次解码）；
- CPU 缓冲按**一个** stem 分配，不再按 `len(training.instruments)`；
- ASR 模式的合并容器从 FLAC 换成 **RF64**（不是 WAV——长素材过 4 GiB，WAV 的回答是静默截断）。

**分块数学是原样照抄的**：window、counter 与最后那次除法都保留。`overlap=8` 让步长等于块长，
于是除最后一块外 window 自己抵消——但 `(x*w)/w` 在浮点下**不等于** `x`，而最后一块确实是与前
一块的交叉淡入。E13 已量过这三项 600 秒合计只要 0.24s，简化它们是拿时间轴换零收益。

| 素材 | 旧 | 新 | |
| --- | ---: | ---: | ---: |
| 270s / 1 worker | 17.52/17.59/17.60s | **16.05/15.68/15.62s** | **1.122×** |
| 2015s / 2 worker | 58.18/57.83s | **54.49/54.25s** | **1.067×** |

累计（E14 + E15，相对本轮开始时）：270 秒 **1.11×**，2015 秒 **1.11×**。

#### 两个只有真素材才会暴露的缺陷

**都不是新代码算错，而是被替掉的那层在悄悄替我们做事。** 记下来是因为两者的形状会重演：
凡是接管第三方入口，就要问「这个入口除了我看到的，还做了什么」。

1. **`autocast` 住在 `Separator.separate` 里**，也就是被替掉的那层外壳，直接调 `model_run`
   不会继承它——整个模型**静默跑成了 FP32**。它不只是慢：AOTI 包是在 autocast 下编译的，
   错配表现为响音上每秒 SNR 只有 12 dB，而输出**听起来仍然是音乐**，下游没有任何东西会报警。
   `use_autocast` 现在是必填参数，由调用方传 `separator.use_autocast`，并有测试钉住。
2. **mono→stereo 住在 `prepare_mix` 的 `ndim == 1` 分支**里，而它靠的是 `librosa.load` 把
   单声道文件交成一维数组。`load_audio_slice` **恒返回二维**，所以单声道源以 `[1, N]` 到达、
   分支永不触发，直接撞上 checkpoint 的 `stereo: true` 断言。按形状判断并复制，测试钉住。

#### 交付内容会变，三个原因，第三个是修了一处缺陷

时间轴不变（互相关最优整数 lag 为 0，帧数逐一相同）。

1. 块输入不再被量化成 PCM_16；
2. 块输出不再经 FLAC 往返，少一次 int16 量化；
3. **超过 ±1.0 的样本不再被削顶。** 这是真缺陷：旧路径把块写成 PCM_16，2015 秒那份素材的
   block 2 有 **48 个样本峰值到 1.268**，被削到 1.0——而 `normalize(max_peak=0.9)` 是按峰值
   定增益的，于是整个 524 秒的块被按 `0.9/1.0` 而不是 `0.9/1.268` 送进模型。实测两版在该块上
   最佳拟合增益 **1.2596**，与 1.268 对得上；其余块增益 0.9996–1.0。

**这条要连着读**：新行为是 `audio-separator` 本来的设计（它对浮点输入就会这么算），旧行为是
finesub 写 PCM_16 块的副产物。但代价是**逐块归一化的增益差被放大了**——旧版四个块是
0.90/0.97/0.96/0.91，新版 block 2 变成 0.71，交付轨在块边界会有一级电平台阶。
**这不是本次引入的机制**（逐块归一化一直如此），只是以前被削顶掩盖着。想真正解决要把归一化
提到整文件一次，那是另一件事，记在待探索队列。

#### 验收

工具：[`compare_outputs.py`](../tools/separator_accel_bench/compare_outputs.py)。
**它抓到了上面两个缺陷，而全局 cosine/SI-SDR 一个都没拦住**——用法与四个视图见该文件。

- **加速本身没变**：AOTI 相对同版 eager 的 SI-SDR，旧实现 74.71dB、新实现 **74.71dB**，一致。
- 270 秒 VAD：新 AOTI 对新 eager **68/72 段起止全等，其余 4 段全部落在一个 20ms 帧以内**
  （总语音差 5ms）。旧实现那次是 72/72——差别不是加速变了，是边界临界点上的运气；
  两版的波形一致性数字完全相同。
- 2015 秒 VAD：671 vs 670 段，七处不一致，五处峰值在 −35～−45dBFS 的极弱片段
  （含 E11 点过名的 `297.2–297.6s`），另两处落在 block 2 那段增益改变的区间里。
- 新实现**逐位可复现**：同配置两次运行输出完全相同。

### E16：新机基线，以及「还剩多少」的三个上限（2026-08-27）

E14/E15 落地后重取，用来给后续决策定价。**结论是分离这条线基本收口了**，三个上限各自封住
一个方向。

#### 改动后的 forward 归因

`max_autotune` 的效果比预期彻底：**cuBLAS GEMM 几乎从 profile 里消失**（97.53ms → 2.46ms），
被带融合 epilogue 的 Triton 模板取代。

| kernel 类别 | 改动前 | 改动后 |
| --- | ---: | ---: |
| GEMM（cuBLAS/cutlass） | 97.53ms (60.4%) | 2.46ms (1.6%) |
| GEMM（Triton 模板，epilogue 已融入） | — | **86.69ms (57.7%)** |
| 逐点 / 归约 | 39.58ms (24.5%) | 36.40ms (24.2%) |
| 注意力 | 20.01ms (12.4%) | 20.01ms (13.3%) |
| 布局搬运 | 4.31ms (2.7%) | 4.39ms (2.9%) |
| **forward wall** | **167.4ms** | **156.5ms** |

⚠ **读 profile 时 `triton_tem_` 与 `triton_poi_` 必须分开算**：前者是 GEMM 模板，后者才是逐点。
混在一起会把矩阵乘的占比报成约 0，任何关于精度的推算都会因此作废。`roofline.py` 的分类表已按
这条排序。

一处**没吃到的**：`triton_poi_fused_gelu_view_17` **12.28ms（7.8%）仍是独立 kernel**——两条
FFN 里只有一条的 GELU 融进了 GEMM。查清另一条为什么没融，是这里最便宜的一笔。

> ✅ **已查清（2026-08-29），结论：生产路径上不存在，不用做。**
> 同一条 AOTI 路径只换解释器重测：**torch 2.9.0（生产跑的那个）上两条 FFN 的 GELU
> 都融进了 GEMM 模板**（34.24 ms + 33.67 ms 两个 `triton_tem_…gelu…`），
> **没有任何独立的 GELU kernel**——全部 77 个 `triton_poi_` 加起来才 10.51 ms。
> 而在 **torch 2.11.0 上 `triton_poi_fused_gelu_view_17` 逐字复现**（31.38 ms）。
> 所以这是一条 **torch 2.11 的回归**，属于「升级 torch 时复查」，不属于待办。
> 模型侧也不存在「两种 FFN」：24 个 `FeedForward` 结构完全相同。
> 全文见 [`bench-baselines.md`](bench-baselines.md) 第十二节，那里还记了
> 两条不可省的说明（2.11 那次的绝对时间不可比；`jit` 臂与 `aoti` 臂结论不同）。

#### 上限一：定精度下 GEMM 已经没有空间

在**生产实际形状**上直接量（`M = 62×801 = 49662`，两轴通用）：

| GEMM | 形状 | FP16 | FP8（`_scaled_mm`，张量级标量） | 提升 |
| --- | --- | ---: | ---: | ---: |
| qkv proj | 49662×512×1536 | 83.0 TFLOP/s | 152.5 | 1.84× |
| attn out | 49662×512×512 | 81.7 | 169.9 | 2.08× |
| ffn up | 49662×512×2048 | 82.5 | 163.0 | 1.97× |
| ffn down | 49662×2048×512 | 85.4 | 165.5 | 1.94× |
| **理想 8192³** | — | **86.6** | 176.1 | 2.03× |

- **生产形状跑在理想形状的 95–98%。** 没有 shape 红利、没有 tile 调优空间；86.6 TFLOP/s 正是这张
  卡 FP16（FP32 累加）的物理上限（= 其 44 TFLOPS FP32 着色器的 2×）。**手写 kernel 这条路封死。**
- **FP8 是唯一的算子层大额，实测约 1.95×**（不是 spec 推的，是同形状实测）。但折算到端到端要
  打折：注意力那 13.3% 不跟着走，逐点那 24.2% 完全不动，而 epilogue 现在**融在 GEMM kernel 里**、
  不随精度加速。**端到端上限约 1.25×**——`max_autotune` 已经先吃掉了一部分。
  另有一项事先无法判断的税：激活量化的 amax+cast，24 层合计约 12.8 GB 额外读写（≈21ms），
  **融得进前一个 epilogue 就接近免费，融不进去就吃掉一半收益**。
- 这张卡**不支持 row-wise scaling**（`_scaled_mm` 直接报错），只能张量级标量缩放——对 12 层
  残差 + 门控注意力的模型，这会显著抬高质量风险。

#### 上限二：并行只值 9%，且第 3、4 个 worker 是负的

E7 的阶梯在新卡上重取（2015 秒素材，AOTI，每档两次）：

| 档位 | workers | 两次 | 中位 | 相对 1w | peak reserved |
| ---: | ---: | --- | ---: | ---: | --- |
| 4GB | 1 | 59.92 / 59.85 | 59.88s | 1.000× | 1.84 GiB |
| 8GB | 2 | 54.98 / 54.91 | **54.94s** | **1.090×** | 1.84–2.03 GiB |
| 12GB | 3 | 55.89 / 55.47 | 55.68s | 1.075× | 2.22–2.27 GiB |
| 16GB | 4 | 59.67 / 59.76 | 59.72s | **1.003×** | 2.27–2.37 GiB |

**与旧卡的形状几乎完全一致**（E7：1.098× / 1.061× / 1.050×），换卡没有改变它。原因在归因里：
**launch 空转只有 4%**，单块 forward 已经把 70 个 SM 吃满，加 worker 只加争用。

**两条可执行的**，但都改动生产映射（块边界随 worker 数移动，产物不可复现），**本节只记不改**：

1. **`budget // 4` 这条阶梯现在是错的方向**：16GB 用户拿到 4 个实例 = 59.72s，比 8GB 的
   54.94s **慢 9%**。两张卡都是这个形状，可以收窄为「上限 2」了。
2. **默认档（4GB）只给 1 个实例，是四档里最慢的**，而 **AOTI 下 2 个 worker 峰值只有 2.03 GiB**
   ——4GB 预算完全放得下。`budget // 4` 是按 **eager** 的显存曲线标的（2.26→4.21 GiB），
   AOTI 的曲线平得多（1.84→2.37 GiB），映射没跟着改。

#### 上限三：真 batch（队列第 3 条）可以直接结掉，不必再跑

队列第 3 条是「真实 batch（B=2）与当前多 stream 并发的吞吐比较」。上限一的数据已经回答了它：
**B=2 的作用就是把 GEMM 的 M 从 49662 翻倍，而 M=49662 已经拿到理想形状 96% 的效率**——
翻倍最多再拿 4%。注意力那一侧本来就已经按 62 个 band 批处理。**结论：真 batch 的上限是几个
百分点，不值得实现专用 demix runner 去验证。** 标记为已回答。

#### 分离已经不是 GPU 段的大头了

同素材（2015.75 秒）逐阶段实测，8GB 档：

| 阶段 | 旧卡（E10） | 新卡 + 本轮 | 占比变化 |
| --- | ---: | ---: | --- |
| 分离（含加载与 ASR 交付编码） | 101.2s | **62s** | 47% → **39%** |
| VAD + ASR（large-v3-turbo + fw-refine） | 112.6s | **98s** | 53% → **61%** |
| GPU 段合计 | 213.8s | **160s** | −25% |

（ASR 那一栏跑在 conda base 的 torch 2.9 + patched CT2 上——bench venv 没有 ASR 栈；
whisper 解码走 CT2，torch 版本影响有限，但这不是同一把尺子，作量级用。）

**这一格是后续优先级的依据**：分离再快 1.25×（FP8 的全部上限）只折合 GPU 段的 1.08×，
再快 1.5×（22.05 kHz）也只折合 1.13×。**继续在分离上投入的边际收益已经很薄了。**

## 外部推理引擎与替代模型调研（2026-08-27）

本轮只做文献/仓库调研，**没有任何一项在本机实测**。记在这里是为了让下一次「要不要换引擎」
不必重查。

| 方案 | 是什么 | 结论 |
| --- | --- | --- |
| **TensorRT** | 把 STFT/iSTFT 拆出 BSRoformer 后可导出 ONNX 并转 TRT engine | **社区实测反向**：RTX 4090 上一个 slice torch 0.13s、TRT 0.27s，初步归因到 `Tile` 算子。与本文 E5 跳过它的判断一致，且现在有了「不只是依赖大」的第二个理由 |
| **BSRoformer.cpp** | GGML 的纯 C++ 推理，支持 CPU/CUDA/Vulkan，可 Q8_0/Q5_1/Q4_0 量化 | 唯一提供**量化**这条我们拿不到的轴。但仓库只有几十星、**未公布任何速度或质量对比**，需自建 GGUF 转换，且会整体取代本文全部 AOTI 工作。**期望值低，除非先有人给出 Q8_0 的 SDR 数字** |
| **openmirlab/bs-roformer-infer** | inference-only 工具包，torch / MLX 双后端 | 加速全在 **MLX（Apple Silicon）**：M2 上 2.5×。CUDA 路径就是普通 torch，**对本项目为零** |
| **mlx-audio-separator** | MLX 原生的分离栈 | 同上，Apple 专用，**不适用** |
| **Windowed Sink Attention**（`smulelabs/windowed-roformer`，MIT） | 把时间轴全注意力换成小窗口 + attention sink，从原 checkpoint 微调 | **FLOPs 降 44.5×、保住 92% SDR**，放出 `mbr-win10-sink8.ckpt`。但那 44.5× 是**注意力** FLOPs，而 **E14 实测本模型注意力只占 12.1% 的 FLOPs / 12.4% 的时间——上限 1.14×**。它针对的是整首歌不分块（时间轴上万帧）的场景，我们按 801 帧切块，注意力本来就不贵。**这条对本项目关闭** |
| **ZFTurbo `mel_band_roformer` vocals_v1** | 更小、`hop_length` 512 的人声 checkpoint | 自述 **RTF 低约 27%**、分离强度约为 Kim Vocal 2 的 95%。换 checkpoint 是纯配置改动，但**换模型就换输出**：VAD 逐段相等这张免费证明不再可用，必须按 E12 的口径做词级验收 |

**读这张表的一条纪律**：本文此前所有采纳项都保住了「与 eager 的下游 VAD 边界逐段相同」这
张免费证明（见「关于怎么读这些数字」）。**换模型的三条路一条都拿不到它**——它们要走的是
E12 那套人工时间轴 + 词准确率的验收，成本高一个量级。这不是反对，是定价。

## 待探索队列

本轮的中间产物（`out/separator-opt/`，含 4 份 AOTI package 与全部对照 FLAC）**已删除**：
上面所有数字都已落到本文，剩下这几条的预期收益又不足以让 2GB 产物长期占盘。要重跑先按
「固定协议」重建素材与 package——`tools/separator_aoti.py` 建包约 30 秒，1400 秒素材的拼接
配方见本地 `data/index.md` 第四节（素材与重建步骤不进 git）。

1. **E14 已归因，这条改写为三个具体候选**，按实测份额排序：
   - ~~`max_autotune` 进 AOTI 构建~~ —— **已做，见 E14「落地」**：长素材 1.039×，
     短素材付 0.55 秒。
   - **那条没被融合的 GELU**：`triton_poi_fused_gelu_view_17` 独占 12.28ms = forward 的
     7.8%，而另一条 FFN 的 GELU 已经融进 GEMM 模板。查清差别即可，**这是最便宜的一笔**。
   - **FP8**（吃 GEMM 那 57.7%）：sm_120 原生支持，**同形状实测 1.95×**，但端到端上限只有
     **约 1.25×**（E16「上限一」——注意力与逐点不跟着走，epilogue 已融在 GEMM 里）。
     工程量与质量风险都大，且这张卡不支持 row-wise scaling，要重做一次 E11 那样的质量二分。
   - **减少 token 数**：即 E12 的 22.05 kHz，**唯一已实现、已刻画质量的大额收益**（旧机 1.52×，
     渐近 2.0×）。开关 `--separator-rate` 已在生产里。
   已经**关掉**的：注意力（只占 13%）、CUDA graph / 异步化 / 减少 kernel 数（launch 空转只有
   4%）、指望编译器把 GEMM 变快（AOTI 相对 eager 一秒没动过它）、**手写/调优 GEMM kernel**
   （生产形状已达理想形状的 95–98%，见 E16「上限一」）。
2. forward 之外还剩约 32ms/chunk 未归因（STFT/iSTFT、mask 拼接、层间 rearrange）。
   层间 rearrange 每块要搬两次 51MiB，是其中可预估的一项。**优先级低**：E16 的归因里
   `copy/layout` 只有 4.39ms (2.9%)，编译路径已经把它融掉了大半。
7. **收窄分离器的 worker 映射**（E16「上限二」）。`budget // 4` 是按 eager 的显存曲线标的，
   AOTI 下 2 个 worker 只要 2.03 GiB。现状是 16GB 档比 8GB 档**慢 9%**、默认的 4GB 档是四档
   里最慢的。两张卡形状一致，可以收窄为「上限 2」了——但改映射会移动块边界、产物不可复现，
   要一起重测下游。
3. ~~真实 batch（B=2）~~ —— **已回答，见 E16「上限三」**：B=2 只是把 GEMM 的 M 翻倍，
   而 M=49662 已经拿到理想形状 96% 的效率，上限是几个百分点。不必实现去验证。
4. ~~专用 demix runner~~ —— **已做，见 E15**：270 秒 1.122×、2015 秒 1.067×。
5. pinned buffer、异步 H2D/D2H、GPU overlap-add：**E13 之后基本可以关掉**。整条 CPU/IO 链
   只有音频时长的 0.6%，其中拼接与拷贝合计 600 秒不到 0.3s，不存在值得异步化的瓶颈。
6. **归一化提到整文件一次**（E15 挖出来的）。`normalize(max_peak=0.9)` 现在按**块**的峰值
   定增益，于是一个块里的一次瞬时峰值会决定整块送进模型的电平，交付轨在块边界留下电平台阶
   （实测同一文件四个块 0.71–0.97）。这既影响送进模型的信号，也影响交付电平。修法是先扫全文件
   峰值、所有块共用一个增益；代价是多一遍读，且要重测下游 VAD。**没有实测收益数字，先记着。**

TensorRT 需要先解决 BS-Roformer 的复数 STFT、动态分块和 attention 导出/插件支持，且
runtime wheel 远大于 Triton；本轮跳过。ONNX Runtime 当前也只有 CPU provider 可用。
自定义 runner 应先在 PyTorch 内验证真实 batch 的上限。
