# BS-Roformer 推理效率探索

本文记录人声分离推理优化的实验协议、逐项结果与取舍。生产配置只吸收经过同素材性能复测、
波形相似度检查和资源上限检查的改动；探索过程不以 GPU utilization 作为吞吐效率的替代指标。

## 固定协议

- 硬件：RTX 5060 Ti 16GB（sm_120）；单卡、空闲桌面环境。
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

基准工具：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m tools.separator_benchmark INPUT OUTPUT --mode fp32 --result RESULT.json
python -m tools.separator_benchmark INPUT OUTPUT --mode amp --reference FP32.flac --result RESULT.json
```

实验开关（`--axis-sdpa`、`--inference-mode`、`--defer-per-file-cache-clear`、
`--no-amp-warmup`、`--torch-compile`、`--aoti-transformer-dir`）只存在于独立基准工具；
被否决的 hook 不进入生产模块。生产的 AOTI package 由第一次运行自建到
`cache/separator-accel/<key>/aoti/`；`python -m tools.separator_aoti OUTPUT_DIR` 只用于把
**变体**建到指定目录做对照，默认即最终配置（运行期常量折叠 + `--attention-backend axis`
+ `--targets all`；`emulate_precision_casts` 在 2.11 上必须关闭，见 E11）。
工具默认复现最终生产配置；重放 E0–E3 的 FP32 预热条件时需加 `--no-amp-warmup`。
JIT 侧的 `--compile-scope all` 对齐 AOTI 的默认 target 集合，用于同 scope 比较（E10）。
编译实验可加 `--probe-compile-timing`，额外做一次同进程 warmup forward；工具会对每次
model forward 做 CUDA 同步，并从可比 wall time 中扣除这次 probe。

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
   `<key>` 含 torch / CUDA / GPU 架构 / checkpoint，换任意一项即换目录。
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

### E10：33.6 分钟真实素材，以及 JIT 在同 scope 下的复测（2026-08-03）

素材 `assets/bilibili/BV1ojjc6MEAs.ogg`，2014.753 秒，8GB profile（时长阶梯算出 7 个
worker，被 profile 封到 2）。ASR 为 `vad-asr`（large-v3-turbo + fw-refine），跑在 AOTI
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

#### 关于怎么读这些数字

- **SI-SDR 衡量的是与 eager 的一致性，不是质量。** eager 自己也只是 FP32 的近似，分数高
  只说明忠实复现了 eager（包括它的误差）。E0 那条 71.4dB 是 AMP-vs-FP32 的一致性，作为
  门槛只意味着「这个量级的偏差以前被接受过一次」，是先例而非质量阈值，且与
  AOTI-vs-eager 是两对不同的东西。
- **一致性是一张免费的证明，不是及格线。** 拿到了就可以确定没有严重质量下滑、无需进一步
  评估；没拿到不代表变差，只说明这张证明不可用，要么接受要么另行刻画。
- **全局 SI-SDR 会稀释局部失效**，必须配合定位。2015 秒这份的逐秒 SNR：中位 75.09dB、
  最低 9.54dB——但按响度分层后，**有信号的 1768 秒里最差一秒仍有 35.8dB**（p05 63.2dB），
  那些 0dB 的秒参考 RMS 是 −300dBFS 的数字静音，是分母为零的产物。没有出现「某个 chunk
  算坏」的特征。**逐秒诊断必须按响度分层看，否则静音区会主导「最差」榜单。**

参考：[PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)、
[PyTorch regional AOT compilation](https://docs.pytorch.org/tutorials/recipes/regional_aot.html)、
[NVIDIA TensorRT pip 安装](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html)、
[Torch-TensorRT 安装](https://docs.pytorch.org/TensorRT/getting_started/installation.html)、
[`triton-windows` 版本对应](https://github.com/triton-lang/triton-windows)。

## 待探索队列

本轮的中间产物（`out/separator-opt/`，含 4 份 AOTI package 与全部对照 FLAC）**已删除**：
上面所有数字都已落到本文，剩下这几条的预期收益又不足以让 2GB 产物长期占盘。要重跑先按
「固定协议」重建素材与 package——`tools/separator_aoti.py` 建包约 30 秒，1400 秒素材的拼接
配方见 [`data-index.md`](data-index.md)。

1. 编译路径下 forward 已压到 278ms，其中 24 个 Transformer 仍占 260ms。要再进一步只能
   动 attention/GEMM 本身（`max-autotune` 在这张卡上被 Inductor 以 SM 不足拒绝），
   或者换 Torch 版本后复测 lowering。
2. forward 之外还剩约 32ms/chunk 未归因（STFT/iSTFT、mask 拼接、层间 rearrange）。
   层间 rearrange 每块要搬两次 51MiB，是其中可预估的一项。
3. 真实 batch（B=2）与当前多 stream block 并发的吞吐/显存比较；0.44.3 当前明确忽略
   Roformer `batch_size`，需要专用 demix runner 才能验证。E7 显示加 worker 已经没有
   收益，真正的并发余量只可能来自 batch。
4. 依赖当前按训练 instrument 数分配双倍 CPU result/counter，即使只输出一个 stem；可在
   专用 runner 中缩成 `num_stems=1`，并测 CPU overlap-add 占比。270 秒任务里 forward
   只占 9.5s / 22.4s，非 GPU 部分现在是大头，这条的优先级比之前高。
5. pinned buffer、异步 H2D/D2H、GPU overlap-add；需先证明 copy/CPU 拼接是显著瓶颈。

TensorRT 需要先解决 BS-Roformer 的复数 STFT、动态分块和 attention 导出/插件支持，且
runtime wheel 远大于 Triton；本轮跳过。ONNX Runtime 当前也只有 CPU provider 可用。
自定义 runner 应先在 PyTorch 内验证真实 batch 的上限。
