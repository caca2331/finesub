# patched CTranslate2 的打包与分发

面向维护者。用户侧的安装步骤见
[`manual/ct2-wheel.md`](manual/ct2-wheel.md)；补丁内容与 CMake 构建标志见
[`../tools/wt_refine_port/ct2-patches/README.md`](../tools/wt_refine_port/ct2-patches/README.md)。
本文只讲**编译产物怎么变成用户能装的东西**。

## 决策：发 Release，不进仓库

wheel 约 60 MB，且每次 CT2/CUDA/补丁变动都要重编。git 按内容去重，同一份提交多次只占
一份，但每个**不同**版本都会在公开仓库里永久留一个 blob，clone 的代价落到所有人头上——
包括只用 LLM 层、根本不装 ASR 的人。Release 资产不进 clone，且给出稳定 URL。

仓里已有 `bin/windows-amd64/tokcount.exe`（17.9 MB）的先例，但那是一次性产物，与这里的
重编频率不是一回事。

**用独立 tag，不跟产品版本走**，例如 `ct2-4.8.1+wtrefine1`。wheel 的生命周期由上游 CT2
版本和补丁决定，与 finesub 的 `vX.Y.Z` 无关；分开之后升级 CT2 不必发产品版本，反之亦然。

## 打包

前提：CT2 已按 `ct2-patches/README.md` 编译完成，得到

- `install-cu-wide/bin/ctranslate2.dll`（约 61 MB）
- `python/build/wt-refine-runtime-wide/`（含 `_ext.<abi>.pyd`）

`python/setup.py` 在 Windows 上已经声明了 `package_data["ctranslate2"] = ["*.dll"]`，
所以**把 DLL 拷进包源码目录**就会被打进 wheel：

```bash
cp install-cu-wide/bin/ctranslate2.dll python/ctranslate2/ctranslate2.dll
```

这一步是自包含的关键。Python 3.8+ 从扩展模块**自身目录**解析依赖 DLL，因此装完不需要
`os.add_dll_directory()`，也不需要把任何东西加进 PATH——这正是 2026-08-03 之前那套
「把构建目录注入 `sys.path` 再手工 `add_dll_directory`」的替代品。

版本号改 `python/ctranslate2/version.py`：

```python
__version__ = "4.8.1+wtrefine1.cu128"
```

然后构建（需要 `pybind11`、`wheel`，以及编译 pybind 绑定用的 MSVC）：

```bash
cd python && CTRANSLATE2_ROOT=../install-cu-wide python setup.py bdist_wheel
```

产物落在 `python/dist/`，例如
`ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl`（约 12 MB，含未压缩
61 MB 的 DLL）。构建完把 `version.py` 还原（`git checkout`）并删掉拷进去的 DLL——wheel
里已经带了各自的副本，源码树不该留。

## 发布

```bash
gh release create "ct2-4.8.1+wtrefine1" \
  python/dist/ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl \
  --title "patched CTranslate2 4.8.1+wtrefine1 (cu128)" \
  --notes "WT refine trace extension. Built from ct2-patches on upstream 0d8bcd3."
```

tag 名里的 `+` 在 URL 中要写成 `%2B`。

## 安装约束：只有 direct reference 能排除 stock

PEP 440 的一个反直觉之处：**不带 local label 的约束会匹配带 local label 的版本**。也就是
说 `ctranslate2==4.8.1` 同时接受 stock 的 `4.8.1` 和补丁版的 `4.8.1+wtrefine1.cu128`，
解析器装到哪个都合法。local version 本身**不构成**排除机制。

真正排除 stock 的是 direct reference。本项目不发 PyPI（只 `pip install -e .`），所以
`pyproject.toml` 可以直接写 URL：

```toml
asr = [
  "faster-whisper==1.2.1",
  "ctranslate2 @ https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bwtrefine1/ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl",
]
```

direct reference 优先级高于版本约束，解析器不会再去 PyPI 找。

**`[asr]` 不用它，`[desktop-worker]` 用。** 取舍的分界是平台自由度：direct reference 把
URL 里的三元组（win_amd64 / cp312 / cu128）变成硬约束，任何其它平台连解析都过不去。

- `[asr]` 面向命令行用户，将来要支持别的平台，所以保留 `ctranslate2==4.8.1` 加用户手动
  覆盖一步（见 `manual/ct2-wheel.md`）。
- `[desktop-worker]` 只喂给 `desktop/runtime/pylock.win-py312.toml`，而桌面版本来就**只有**
  Windows / CPython 3.12 / cu128 这一个组合，钉死是零成本的。于是 lock 里直接锁到带
  sha256 的 wheel：

  ```toml
  [[packages]]
  name = "ctranslate2"
  version = "4.8.1+wtrefine1.cu128"
  archive = { url = "https://github.com/.../ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl", hashes = { sha256 = "66a2780..." } }
  ```

  开发机（`desktop/scripts/setup-dev.ps1`）和端用户安装（`RuntimeEnvironment.install`）
  都是 `uv pip install --requirement <lock>`，所以两边自动拿到补丁版，不需要各自补一步
  force-reinstall。`desktop/backend/runtime/environment.py` 的运行时探针再查一次
  `__version__` 里的 `wtrefine`，兜住环境被手工改坏的情况。

换 wheel（升级 CT2 或重编补丁）时要一起动的：`[desktop-worker]` 里的 URL、重跑
`uv pip compile` 更新 lock 里的 sha256。`test_windows_ai_runtime_lock_pins_torch_stack`
会在两者不一致时报错。

## 未决项

- **`cublas64_12.dll` 由 torch 提供（2026-08-05 实测确认），但它把 wheel 锁在 CUDA 12。**
  拆 `ctranslate2.dll` 看到：CUDA runtime 是**静态链接**的（导入表里没有 cudart，二进制里也
  没有该字符串），但 cuBLAS 是**运行时 `LoadLibrary`** 的——二进制里有 `cublas64_12.dll`
  字符串却不在导入表。这个 SONAME 属于 CUDA **12**，于是：

  | torch 构建 | `torch/lib/` 里的 cuBLAS | cu128 的 CT2 wheel |
  | --- | --- | --- |
  | cu126 / cu128 | `cublas64_12.dll` | ✅ 可用 |
  | cu130 | `cublas64_13.dll` | ❌ 找不到它要的 `_12` |

  CUDA 12.x 内部前后兼容（SONAME 不变），**跨到 CUDA 13 则不兼容**。这是 torch 停在
  2.11/cu128 的直接原因之一。

  **干净机器上由 torch 兜住，不需要额外动作**（2026-08-05 剥 PATH 实测）：torch import 时会对
  `torch/lib/` 做 `add_dll_directory`，那里就有 `cublas64_12.dll`。把 PATH 剥到只剩
  `system32` 后，裸环境 `LoadLibrary("cublas64_12.dll")` 失败，`import torch` 之后成功。
  torch 是 `[asr]` 硬依赖且生产路径必然先于 CT2 导入（VAD 阶段就用 torch），所以这条链成立。

  ⚠️ 但**不要**据此认为跨代可用：本机 cu130 下 CT2 也跑通过，那是因为这台机器装了系统级
  CUDA Toolkit 12.8/12.6 且 `bin` 在 PATH 上，`LoadLibrary` 从那里拿到了 `_12`。干净机器上
  没有这个巧合，而 torch 一旦换到 cu130 就只带 `cublas64_13.dll`，兜底立刻失效。

  仍未采用的备选：随 wheel 分发 cuBLAS、或声明 `nvidia-cublas-cu12` 并在
  `ctranslate2/__init__.py` 里自行 `add_dll_directory`（可摆脱对 torch 导入顺序的隐式依赖）。
- **内嵌 GPU 架构没有核实清楚。** `cuobjdump --list-elf` 报告 SASS 为
  `sm_70/75/80/86/89/90` 且**无 PTX**，但这块 sm_120（Blackwell）的卡上 ASR 确实在 GPU 上
  跑通了。可能是 cuobjdump 对该 DLL 列举不全，也可能重活都走了 cuBLAS。未查实——换目标
  架构前必须在真机验证，`ct2-patches/README.md` 里那条 `cuobjdump` 检查就是为此存在的。
- **只有 Windows / CPython 3.12 / CUDA 12.8 一个组合。** wheel 是 CPython ABI 专属的
  （`cp312` 只能装 Python 3.12），换任意一维都要重编重发。CPU-only 构建见
  `ct2-patches/README.md` 的 `build-cpu` 配置。
- **升级上游 CT2 时**，`ct2-patches/` 需要重新 rebase 并重测，这是该方案的已知成本。
  升级顺序是先 faster-whisper 后 CT2——CT2 的可选范围由 fw 决定。
