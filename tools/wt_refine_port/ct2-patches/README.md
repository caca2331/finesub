# CTranslate2 patch series

`fw-refine` 后端要求打过补丁的 CTranslate2：WT refine 所需的 attention、beam winner lineage、
终止事件与 chosen logprob 都在 CT2 内部产生，只改 faster-whisper 无法在不做第二遍解码的前提下
恢复它们。

本目录是这些改动的 `git format-patch` 导出，**基线 `v4.8.1`（upstream `0d8bcd3`）**，顺序即依赖顺序。
它让本仓库能独立描述 CT2 侧需要什么，不必依赖某台机器上的本地 clone。

```bash
git clone https://github.com/OpenNMT/CTranslate2 && cd CTranslate2
git checkout 0d8bcd3
git am /path/to/tools/wt_refine_port/ct2-patches/*.patch
```

| 补丁 | 内容 |
| --- | --- |
| 0001–0004 | WT 风格的 alignment 控制、attention 后处理、padding 与真实音频边界分离 |
| 0005–0008 | decoded alignment 前缀、compact one-pass trace 与 refine path、unfinished span |
| 0009–0010 | beam winner lineage（compaction-safe）、disfluency 所需的 refine weights |
| 0011 | 逐样本 `real_audio_frames`——multi-audio batch 的前置 |

**本目录是长期方案，不是权宜之计。** 已决定不 fork CTranslate2，而是钉住上游某个版本
（当前 `v4.8.1` / `0d8bcd3`）、以本 patch series 打补丁、并预编译分发——避免长期维护一份上游分叉。
因此升级上游版本时，本目录需要重新 rebase 并重测，这是该方案的已知成本。

版本选择：`v4.8.1` 是 CTranslate2 当前最新 release，且满足 faster-whisper 1.2.1（当前最新）
声明的 `ctranslate2>=4.0,<5`。两者已在 `pyproject.toml` 精确钉版。**升级顺序是先 faster-whisper
后 CT2**——CT2 的可选范围由 fw 决定，反过来不成立。

## 构建（这一节是硬性要求，不是建议）

补丁只是源码，**必须编译**。构建标志不在补丁里，所以下面这些必须显式给出——
2026-08 的第一次构建就因为漏了其中两项而产出了一个只能在 Ampere 上跑、且完全无法用 CPU 的
二进制，两个问题都要到运行时才暴露。

以下命令在本机（Windows 11 / MSVC 2022 / CUDA 12.8 / CMake 4.2）**实测配置通过**：

```bash
# GPU wheel: 4.8.1+wtrefine1.cu128
cmake -S . -B build-cu -G "Visual Studio 17 2022" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DOPENMP_RUNTIME=COMP \
      -DWITH_CUDA=ON -DCUDA_DYNAMIC_LOADING=ON \
      -DWITH_MKL=OFF -DWITH_RUY=ON -DBUILD_CLI=OFF \
      -DCUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
cmake --build build-cu --config Release --parallel

# CPU wheel: 4.8.1+wtrefine1.cpu（仅当 CUDA_DYNAMIC_LOADING 不足以覆盖无驱动的机器）
cmake -S . -B build-cpu -G "Visual Studio 17 2022" \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DOPENMP_RUNTIME=COMP \
      -DWITH_CUDA=OFF -DWITH_MKL=ON -DWITH_DNNL=ON
cmake --build build-cpu --config Release --parallel
```

前两个标志与补丁无关，是**上游 + 环境组合的产物，缺了配置阶段就失败**：

- `CMAKE_POLICY_VERSION_MINIMUM=3.5` —— CMake 4.x 移除了对 `cmake_minimum_required < 3.5` 的
  兼容，而 `third_party/cpu_features` 仍是旧声明（`ENABLE_CPU_DISPATCH` 默认 ON 会引入它）。
- `OPENMP_RUNTIME=COMP` —— 默认值是 `INTEL`，会去找 Intel oneAPI 的 `libiomp5`，未安装则
  `FATAL_ERROR: Intel OpenMP runtime libiomp5 not found`。`COMP` 用 MSVC 自带的 OpenMP。

验证架构真的生成了（不要只看 cache 里的 `CUDA_ARCH_LIST`，那只是输入）：

```bash
grep -rohE "compute_[0-9]+,code=[a-z_0-9]+" build-cu | sort -u
# 期望：compute_70..compute_89,code=sm_XX 各一条 + compute_90,code=compute_90（PTX）
```

构建完成后再验一次二进制本身：

```bash
cuobjdump --list-elf build-cu/.../ctranslate2.dll | grep -oE "sm_[0-9]+" | sort -u
```

**没有 `WITH_PYTHON` 这个选项**——Python 扩展由 `python/setup.py` 单独构建，不在 cmake 链内。

**一个 wheel 就够，不需要单独的 CPU wheel。** 上面第二条命令保留只是为了追求 CPU 性能
（Ruy 慢于 MKL）；覆盖"没有 NVIDIA 卡/驱动"的机器不需要它。三层机制叠加使得同一个二进制
在无驱动机器上也能 `import` 并走 CPU 路径：

1. **`CUDA_DYNAMIC_LOADING=ON`** 把 cuBLAS/NCCL/MPI 改成 stub（`src/cuda/cublas_stub.cc`），
   惰性 `LoadLibrary`，找不到才抛可捕获的 `runtime_error`。**它不管 cudart**。
2. **cudart 由 nvcc 默认静态链接**，因此根本不产生 `cudart64_*.dll` 依赖。
3. **驱动库 `nvcuda.dll` / `libcuda.so` 从来不是链接期依赖**，一直是运行时惰性加载。

`src/cuda/utils.cc` 的 `get_gpu_count()` 拿不到设备时吞掉错误码返回 0，不抛不崩。

本机实测导入表（`dumpbin /dependents`）可交叉验证：

| | 加载期 CUDA 依赖 | 运行时查找 |
| --- | --- | --- |
| 旧构建（`DYNAMIC_LOADING=OFF`） | **`cublas64_12.dll`（在导入表里）** | — |
| 新构建（`ON`） | 无 | `cublas64_12.dll` |
| 官方 PyPI wheel 4.8.1 | 无 | `cublas64_12.dll` |

**新构建与官方 wheel 同构**——上游正是靠这套机制用一个包同时支持 CPU 与 GPU。

⚠️ **仍未在真正没有 NVIDIA 驱动的机器上实测**（本机有卡）。证据链完整且与上游一致，
但这是推断加旁证，不是实测。

**CUDA 版本要求**：`cublas64_12.dll` 里的 `12` 是主版本，任何 CUDA 12.x 都可以
（minor version compatibility）。不需要 cuDNN（`WITH_CUDNN=OFF`）。但 **Blackwell 例外**：
它没有原生 SASS、必须 JIT sm_90 的 PTX，而 PTX ISA 8.7 需要 CUDA 12.8 级别的驱动
（Windows 约 571.x+）；sm_70–90 的卡有原生 SASS，较老的 12.x 驱动即可。

**分发待办**：真正用 GPU 时 `cublas64_12.dll` 必须在 DLL 搜索路径上。目前靠调用方
`os.add_dll_directory(CUDA toolkit bin)`。官方 wheel 同样不打包它——即它也要求机器上有
CUDA 12 的 cuBLAS。打包时需在三者中选一：随包分发、声明 `nvidia-cublas-cu12` 依赖、
或要求用户装 CUDA toolkit。

**本机没有 MKL / oneDNN / OpenBLAS**（`WITH_MKL` 默认 ON，build4 显式关掉正是因为这个）。
仓库内自带的 CPU 后端只有 **Ruy**（`third_party/ruy` submodule，已初始化），所以它是唯一
零外部依赖的选择。要追 CPU 性能就得先装 oneAPI MKL 或 oneDNN。

Ruy 构建的 CPU 实测（2026-08-02，8 秒音频）：**可用且正确**——与 GPU 输出逐字相同；
解码 41.3s vs GPU 1.4s，约 30×。注意 Whisper encoder 无论音频多短都要过完整 30 秒窗口，
短片段的固定成本占比极高，真实 30 秒分组上的倍率会好得多。该构建的
`get_supported_compute_types("cpu")` 返回 `['float32', 'int8', 'int8_float32']`
（无 CPU 后端时只有 `['float32']`），`int8` 是需要时的提速旋钮。

**`CUDA_ARCH_LIST` 不能省。** 默认值是 `"Auto"`，只编本机架构。更糟的是 CMake 的
`FindCUDA/select_compute_arch.cmake` 架构表**停在 Ampere**（最后一个分支是 CUDA≥11.1 加 8.6），
所以在更新的卡上 `Auto` 探测不到、静默回退到 8.6。

⚠️ **该模块无法表达 Blackwell。** 它的数值匹配正则是
`^([0-9]\.[0-9](\([0-9]\.[0-9]\))?)$`——**major 只允许一位数**，`12.0` 匹配不上，会走到名字比对
然后 `SEND_ERROR: Unknown CUDA Architecture`。列表里写 `9.0+PTX` 即可：RTX 50 系靠 PTX JIT 运行
（首次加载多花点时间，之后有驱动缓存）。要原生 sm_120 必须绕开该宏，直接往 `CUDA_NVCC_FLAGS`
追加 `-gencode arch=compute_120,code=sm_120`，或把 CMakeLists 迁到现代的
`CMAKE_CUDA_ARCHITECTURES`（那需要再加一个补丁）。

**CPU GEMM 后端不能全关。** `WITH_MKL` 默认 ON，但一旦显式关掉且没开 DNNL/OpenBLAS/Ruy 中的
任何一个，CPU 推理会在第一次 encode 抛 `No SGEMM backend on CPU`——而
`get_supported_compute_types("cpu")` 仍然报 `['float32']`，查不出来。
GPU wheel 带上 Ruy（体积开销很小）是为了让 `resolve_device()` 的 CUDA→CPU 降级路径真的可用；
CPU wheel 用 MKL/oneDNN 追性能。

`WITH_CUDNN` 保持 OFF——本补丁集不需要 cuDNN，少一个分发依赖。

可选性能旋钮（未采用）：CPU 上 `compute_type="int8"` 比 `float32` 快 2–4×，但会改变输出，
属于要单独验收的取舍，不是构建标志。当前 CPU 路径固定 `float32`。

wheel 必须带明确的 PEP 440 local version（如 `4.8.1+wtrefine1.cu128`），并记录编译器、CUDA
版本与完整构建命令。带 local label 的 `==` 约束要求 local 部分完全一致，因此钉到该版本后
stock CT2 会被解析器直接拒绝——这正是想要的。
