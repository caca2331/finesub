# patched CTranslate2 的打包与分发

面向维护者。用户侧的安装步骤见
[`manual/ct2-wheel.md`](manual/ct2-wheel.md)；补丁内容与 CMake 构建标志见
[`../tools/wt_refine_port/ct2-patches/README.md`](../tools/wt_refine_port/ct2-patches/README.md)。
本文只讲**编译产物怎么变成用户能装的东西**。

## 决策：发 Release，不进仓库

wheel 约 17 MB（内含未压缩 79 MB 的 DLL），且每次 CT2/CUDA/补丁变动都要重编。git 按内容去重，同一份提交多次只占
一份，但每个**不同**版本都会在公开仓库里永久留一个 blob，clone 的代价落到所有人头上——
包括只用 LLM 层、根本不装 ASR 的人。Release 资产不进 clone，且给出稳定 URL。

仓里已有 `bin/windows-amd64/tokcount.exe`（17.9 MB）的先例，但那是一次性产物，与这里的
重编频率不是一回事。

**用独立 tag**，形如 `ct2-<上游版本>+finesub<产品版本>`，例如 `ct2-4.8.1+finesub0.4.0`。
两段各有分工：`4.8.1` 是上游 CT2 版本，`finesub0.4.0` 记的是**引入这个 wheel 的那次 finesub
发布**——出问题时能直接对上是哪一版换的二进制。

⚠️ 这不等于「每发一次产品版本就重发 wheel」：wheel 的生命周期仍由上游 CT2 和补丁集决定，
后续 finesub 版本若没换二进制就继续用旧 label，不重发。反过来，升级 CT2 或重编补丁时才起新
label，取当次发布的产品版本号。

（本条 2026-08-10 修订：原先写的是「不跟产品版本走」，用 `wtrefine1` 这样的自增序号；改成
带产品版本是为了让「哪一版引入的」可追溯。`src/finesub_bootstrap/environment.py` 的
`REQUIRED_CTRANSLATE2_LOCAL_LABEL` 因此只匹配 `finesub` 而不含版本号——它判的是补丁版还是
原版，不该每次重编都跟着改。）

## 打包

前提：CT2 已按 `ct2-patches/README.md` 编译完成，得到

- `build-cu-dnnl/Release/ctranslate2.dll`（79 MB），以及 `cmake --install` 出来的
  `install-cu-dnnl/`（头文件 + `ctranslate2.lib`，供 `setup.py` 用）
- Intel OpenMP 的 `libiomp5md.dll`（1.6 MB，从 pip 的 `intel-openmp` 取，见 ct2-patches README）

`python/setup.py` 在 Windows 上已经声明了 `package_data["ctranslate2"] = ["*.dll"]`，
所以**把 DLL 拷进包源码目录**就会被打进 wheel：

```bash
cp build-cu-dnnl/Release/ctranslate2.dll python/ctranslate2/ctranslate2.dll
cp <IOMP>/bin/libiomp5md.dll            python/ctranslate2/libiomp5md.dll
```

这一步是自包含的关键。Python 3.8+ 从扩展模块**自身目录**解析依赖 DLL，因此装完不需要
`os.add_dll_directory()`，也不需要把任何东西加进 PATH——这正是 2026-08-03 之前那套
「把构建目录注入 `sys.path` 再手工 `add_dll_directory`」的替代品。

版本号改 `python/ctranslate2/version.py`：

```python
__version__ = "4.8.1+finesub0.4.0.cu128"
```

然后构建（需要 `pybind11`、`wheel`，以及编译 pybind 绑定用的 MSVC）：

```bash
cd python && CTRANSLATE2_ROOT=../install-cu-dnnl python setup.py bdist_wheel
```

产物落在 `python/dist/`，例如
`ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl`（17.3 MB，含未压缩
79 MB 的 `ctranslate2.dll` 和 1.6 MB 的 `libiomp5md.dll`）。构建完把 `version.py` 还原
（`git checkout`）并删掉拷进去的 DLL——wheel 里已经带了各自的副本，源码树不该留。

## 发布

```bash
gh release create "ct2-4.8.1+finesub0.4.0" \
  python/dist/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl \
  --title "patched CTranslate2 4.8.1+finesub0.4.0 (cu128)" \
  --notes "WT refine trace extension. Built from ct2-patches on upstream 0d8bcd3."
```

tag 名里的 `+` 在 URL 中要写成 `%2B`。

⚠️ **这一步漏过一次，代价是整个 0.4.0 装不上**（2026-08-18 发现，当天补发）：`efdeb84`
把重编后的引用改进了 `pyproject.toml` 和两份 `src/finesub_bootstrap/pylock.win-py312*.toml`，
sha256 也算对了，就是没跑上面那条 `gh release create`。产品 release 照常发出去，
`ct2-4.8.1+finesub0.4.0` 这个 tag 却不存在，于是所有前端的首次环境安装都 404 在 ASR 那步。

**没有任何测试能拦住它**：`ci.yml` 故意只装 `[harness,dev]`、跳过 `[asr]`，因此永远不会去
解析那条 direct reference。拦它的是发布流程里的一步——`scripts/check-pinned-urls.ps1`
（release skill 第 3 步），会把这里发的 tag 与 lock 里的 sha256 对照校验。
**改完 wheel 引用就顺手把 release 发掉**，别攒到发版时。

## 安装约束：只有 direct reference 能排除 stock

PEP 440 的一个反直觉之处：**不带 local label 的约束会匹配带 local label 的版本**。也就是
说 `ctranslate2==4.8.1` 同时接受 stock 的 `4.8.1` 和补丁版的 `4.8.1+finesub0.4.0.cu128`，
解析器装到哪个都合法。local version 本身**不构成**排除机制。

真正排除 stock 的是 direct reference。本项目不发 PyPI（只 `pip install -e .`），所以
`pyproject.toml` 可以直接写 URL：

```toml
asr = [
  "faster-whisper==1.2.1",
  "ctranslate2 @ https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.4.0/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl",
]
```

direct reference 优先级高于版本约束，解析器不会再去 PyPI 找。

**`[asr]` 不用它，`[runtime]` 用。** 取舍的分界是平台自由度：direct reference 把
URL 里的三元组（win_amd64 / cp312 / cu128）变成硬约束，任何其它平台连解析都过不去。

- `[asr]` 面向命令行用户，将来要支持别的平台，所以保留 `ctranslate2==4.8.1` 加用户手动
  覆盖一步（见 `manual/ct2-wheel.md`）。
- `[runtime]` 只喂给 `src/finesub_bootstrap/pylock.win-py312.toml`，而托管运行环境本来就**只有**
  Windows / CPython 3.12 / cu128 这一个组合，钉死是零成本的。于是 lock 里直接锁到带
  sha256 的 wheel：

  ```toml
  [[packages]]
  name = "ctranslate2"
  version = "4.8.1+finesub0.4.0.cu128"
  archive = { url = "https://github.com/.../ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl", hashes = { sha256 = "636d69f..." } }
  ```

  端用户安装（`RuntimeEnvironment.install`）是 `uv pip install --requirement <lock>`，所以
  自动拿到补丁版，不需要补一步 force-reinstall。`src/finesub_bootstrap/environment.py` 的运行时探针再查一次
  `__version__` 里的 `finesub`（`REQUIRED_CTRANSLATE2_LOCAL_LABEL`，不含版本号），兜住环境被
  手工改坏的情况。

换 wheel（升级 CT2 或重编补丁）时要一起动的：`[runtime]` 里的 URL、重跑
`uv pip compile` 更新 lock 里的 sha256（下节）。`test_windows_ai_runtime_lock_pins_torch_stack`
会在两者不一致时报错。

## 约束与已定项

- ~~**cuBLAS 把 wheel 锁在 CUDA 12**~~ **已定（2026-08-20）：钉在 CUDA 12 是目标，不是债。**
  CUDA 13 是新 major，兼容面窄得多；本项目要的正是 CUDA 12.x 那份「SONAME 不变、内部前后
  兼容」的预算。所以 `cublas64_13.dll` 不是「更新的版本」，是**不受支持的环境**。
  下面的机制记录保留，因为它解释了这条钉子长什么样、以及为什么曾经是隐式的。

  拆 `ctranslate2.dll` 看到：CUDA runtime 是**静态链接**的（导入表里没有 cudart，二进制里也
  没有该字符串），但 cuBLAS 是**运行时 `LoadLibrary`** 的——二进制里有 `cublas64_12.dll`
  字符串却不在导入表。这个 SONAME 属于 CUDA **12**，于是：

  | torch 构建 | `torch/lib/` 里的 cuBLAS | cu128 的 CT2 wheel |
  | --- | --- | --- |
  | cu126 / cu128 | `cublas64_12.dll` | ✅ 可用 |
  | cu130 | `cublas64_13.dll` | ❌ 找不到它要的 `_12` |

  CUDA 12.x 内部前后兼容（SONAME 不变），**跨到 CUDA 13 则不兼容**。torch 停在
  2.11/cu128 与此一致——那是选定的组合，不是被卡住。

  **2026-08-20 修的是另一半：隐式的导入顺序依赖。**

  此前靠的是巧合（2026-08-05 剥 PATH 实测）：torch import 时会对 `torch/lib/` 做
  `add_dll_directory`，那里就有 `cublas64_12.dll`；把 PATH 剥到只剩 `system32` 后，裸环境
  `LoadLibrary("cublas64_12.dll")` 失败，`import torch` 之后成功。链条成立只因为 torch 是
  `[asr]` 硬依赖、且 VAD 阶段必然先于 ASR 跑——**顺序变了就断，而断了没有任何征兆**。

  现在由 `finesub/speech/runtime/cuda_libs.py` 主动找：用 `find_spec` 定位包目录（**不导入
  torch**——导入它就等于把刚去掉的顺序依赖换个地方写回来，还会提前建 CUDA context），依次看
  `nvidia/cublas/bin` 与 `torch/lib`，第一个真的有这个 DLL 的目录进 `add_dll_directory`。
  调用点在 `RefinedWhisperModel.__init__` 里、`super().__init__` **之前**，由
  `test_cuda_libs.py` 的源码守卫钉住顺序。找不到不是致命错——装了系统级 CUDA Toolkit 的机器
  照样能解析——所以只报一条 `cublas-not-found` 告警，并说清它到底找到了什么。

  ⚠️ 本机 cu130 下 CT2 也跑通过，但那是因为这台机器装了系统级 CUDA Toolkit 12.8/12.6 且
  `bin` 在 PATH 上——**别把它读成跨代可用**。干净机器上没有这个巧合。

  搜到的若是别的世代，告警会**点名**（`已安装的包里只有 cublas64_13.dll…`）而不是说「找不到」：
  文件就在那儿，只是这个 build 用不了它，而「找不到」会把人支去找一个并不缺的文件。

  **为什么不声明 `nvidia-cublas-cu12`**：Windows 上 torch 是把 CUDA 库**打进 `torch/lib`**
  而不是依赖那些 `nvidia-*` 包，所以加这个依赖等于在盘上放第二份约 400 MB 的同一个东西。
  搜索路径里仍然把它排在 torch 前面：装了它是某人的明确决定，torch 那份只是副产品。
- ~~**CPU GEMM 后端**~~ **已修（2026-08-10），保留记录以免重犯。** 上一版
  `4.8.1+wtrefine1.cu128` 的 CPU GEMM 后端只有 Ruy，而 **Ruy 会在模型析构时死锁**——CPU 上
  解码过一次之后 `del model` 永不返回，产物落盘但 pipeline 停在 ASR 阶段末尾，端到端 CPU
  回退不可用（0.3.2 的现场故障）。现版换成**裁剪过的静态 oneDNN**：解码 14.9s（Ruy 22.1s）、
  峰值 RSS 3.78 GB（Ruy 3.95）、析构 0.37s。
  **中途试过的 MKL 不要再走**：不死锁但多吃 3.4 GB 内存，且默认只在 Intel CPU 上启用
  （AMD 上会抛 `No SGEMM backend on CPU`）。完整配方与三后端实测矩阵见
  `tools/wt_refine_port/ct2-patches/README.md`。
  顺带收掉了「自包含」这条线的一半：wheel 现在把 `libiomp5md.dll` 打进包目录（照 stock 的做法），
  不再依赖「调用方先 import torch 才找得到 OpenMP DLL」这条隐式链。**剩下的一半仍是上面那条
  cuBLAS**。
  体积：DLL 61.2 → 79.3 MB，wheel 12.1 → 17.3 MB。oneDNN 已按 `DNNL_ENABLE_WORKLOAD=INFERENCE`
  + 四个原语 + 关掉 graph 组件裁过（未裁剪时是 94.6 / 21.6 MB，裁剪后反而快了约 8%）。**ISA 保持
  `ALL` 是有意的**——那正是 CPU 性能来源，为体积裁它会在部分 CPU 上变慢。
- ~~**内嵌 GPU 架构没有核实清楚**~~ **已收口（2026-08-20，owner 确认）：新架构可用。**
  `cuobjdump --list-elf` 报告 SASS 为 `sm_70/75/80/86/89/90`，而这块 sm_120
  （Blackwell）的卡上 ASR 确实在 GPU 上跑通，因此不再当作风险项。
  ⚠ **更正（2026-08-29）**：此处原写「**且无 PTX**」——那是**工具用法造成的假象**，
  `--list-elf` 按定义**只列 ELF（SASS），不列 PTX**。构建缓存里
  `CUDA_ARCH_LIST=7.0;7.5;8.0;8.6;8.9;9.0+PTX`，每个 `.cu` 的 nvcc 行都有
  `compute_90,code=compute_90`——**PTX 是在的**，CT2 自己的 kernel 在 sm_120 上走
  compute_90 PTX 的驱动 JIT。与 [`wt-refine-handoff.md`](wt-refine-handoff.md)
  「含 sm_70–90 原生 SASS + sm_90 PTX」一致，**以那处为准**。
  换**目标架构**（改编译参数）时仍要在真机验证，`ct2-patches/README.md` 里那条 `cuobjdump`
  检查就是为此存在的。
- **只有 Windows / CPython 3.12 / CUDA 12.8 一个组合。** wheel 是 CPython ABI 专属的
  （`cp312` 只能装 Python 3.12），换任意一维都要重编重发。不需要单独的 CPU-only wheel：
  `CUDA_DYNAMIC_LOADING=ON` 让同一个二进制在无驱动机器上也能 import 并走 CPU 路径
  （理由见 `ct2-patches/README.md`）。
- **升级上游 CT2 时**，`ct2-patches/` 需要重新 rebase 并重测，这是该方案的已知成本。
  升级顺序是先 faster-whisper 后 CT2——CT2 的可选范围由 fw 决定。

## 锁的重建

终端用户的 Windows / Python 3.12 / CUDA 12.8 运行环境锁在 `src/finesub_bootstrap/pylock.win-py312.toml`
（2026-09-03 起住在包里，命令行与曾经的桌面端都读它）。更新 AI 依赖后，在仓库根目录重新生成：

```powershell
uv pip compile pyproject.toml `
  --extra asr `
  --extra harness `
  --extra runtime `
  --python-platform x86_64-pc-windows-msvc `
  --python-version 3.12 `
  --torch-backend cu128 `
  --format pylock.toml `
  --output-file src/finesub_bootstrap/pylock.win-py312.toml
```

运行时 marker 记的是锁的**内容**摘要（`lock_content_digest`：去掉 `#` 注释行、统一行尾），所以
改头部注释或换一个 autocrlf 不同的 checkout 都不会被判成「依赖变了」。0.5.0 之前记的是整文件
的 sha256，`environment.py` 的 `_LEGACY_LOCK_FILE_DIGESTS` 列着那一份锁的两种行尾形态，让旧安装
升级时不重建——**下一次真的重新生成锁时把那个常量删掉**（重新生成本来就要重建，之后写入的就是
内容摘要）。

改完 canonical lock **必须重新生成地区 lock**，否则两者会漂移：

```powershell
python -m scripts.make_cn_lock src/finesub_bootstrap/pylock.win-py312.toml `
  --output src/finesub_bootstrap/pylock.win-py312.cn.toml
```

它只改 artifact URL（镜像地址取自 `download-sources.json`），生成后立刻自检包名/版本/
marker/文件名/摘要是否与 canonical 逐项一致，不一致就删产物报错；`test/bootstrap/test_cn_lock.py`
对入库的两份文件再跑一次同样的比对。**运行时 marker 只哈希 canonical lock**——两份 lock 描述的是
同一批文件、只差谁发货，按实际安装用的那份算会让跨地区被判成「依赖变了」而重建整个环境。

外部工具（uv / ffmpeg / MinGit / yt-dlp / tokcount）**不进 lock**：它们在同目录的
`runtime-manifest.json` 里（url + size + sha256 + required_files），由 `ResourceManager` 通用地
下载/校验/版本化/原子切换。改 manifest 不碰运行时，改 lock 会触发整个 Python 环境重建——对
yt-dlp 这种要跟版本的工具，差别是 3MB 对上数 GB。
