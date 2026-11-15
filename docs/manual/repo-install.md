# 仓库（源码）安装

本页面向两类读者：需要运行仓库开发版的开发者，以及希望将 pipeline 安装到自有 Python 环境（而非使用
`finesub` CLI 的托管运行环境）的用户。安装后的入口为 `python -m finesub.pipeline`（与 README 中
`finesub` 命令的参数一致）；传入多个输入或使用 `--manifest` 即为批量处理，没有独立的批量入口。

需要自行安装 ffmpeg 并加入 PATH，且须包含 `libx264` 编码器（LLM 纠错阶段的每份剪辑均使用该编码器；
执行 `ffmpeg -encoders | findstr libx264` 有输出即表示已包含，常见发行版中仅标记 lgpl 的构建缺少
该编码器）；另需一张可用的 NVIDIA 显卡（AMD 卡见下面「其他平台」）——**没有显卡也能运行**，但会自动回退到 CPU,
速度明显变慢（见 README「环境要求」）。默认使用 uv；坚持使用 pip 请跳到
[第二节](#用-pip-安装)，该路径的坑更多。

## 用 uv 安装（默认）

没有 uv 的话先 `winget install astral-sh.uv`;Python 3.12 由 uv 自动准备，无需预装。在源码目录下：

```powershell
# 创建并启用虚拟环境
uv venv --python 3.12
.venv\Scripts\activate

# ASR 全栈 + LLM 层(含 Qwen3-ASR 第二模型校验,首次运行时自动下载模型 ~1.5GB)
# --torch-backend cu128 确保拿到 CUDA 版 torch(PyPI 上的 Windows 构建不带 CUDA)
uv pip install --torch-backend cu128 -e ".[asr,harness]"

# ASR 必需的 patched CTranslate2(原版装上也跑不了,详见 ct2-wheel.md)
uv pip install --reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.6.0/ctranslate2-4.8.1+finesub0.6.0.cu128-cp312-cp312-win_amd64.whl"
```

## 用 pip 安装

pip 没有 `--torch-backend`，需要自行绕开两个由 uv 自动处理的问题：**torch 必须从
download.pytorch.org 获取 CUDA 构建**（PyPI 上的 Windows torch 不包含 CUDA，装错后 GPU 路径会静默
失效、速度极慢），且 **patched CTranslate2 需单独安装**。还需自行准备 **Python 3.12**(pip 不会
自动下载解释器)。

在源码目录下按以下顺序执行：

```powershell
# 1. 虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 2. 先从 PyTorch 官方索引装 CUDA 版 torch 三件套(版本必须与 pyproject 钉的一致)
pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3. 再装项目(torch 已满足约束,pip 不会用 PyPI 的 CPU 版覆盖它)
pip install -e ".[asr,harness]"

# 4. patched CTranslate2(原版装上也跑不了 ASR,见 ct2-wheel.md)
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.6.0/ctranslate2-4.8.1+finesub0.6.0.cu128-cp312-cp312-win_amd64.whl"
```

为何将第 2、3 步分开：`torch==2.11.0` 这一约束同时被 PyPI 的 CPU 构建与 `+cu128` 构建满足，在单条
命令中使用 `--extra-index-url` 时，pip 无法保证会选择哪一个；先将 CUDA 版装入环境后，后续解析只会
沿用该版本。

## 其他平台：Linux、AMD、macOS

非 Windows 平台只支持源码装法，各平台的支持等级见 [`resources.md`](resources.md)「平台与支持等级」。
步骤与上面相同，区别只有两处：**torch 从哪个索引装**，以及**装哪个 patched CTranslate2**。
以下用 pip 写。Linux 用 uv 时把 `--index-url` 换成对应的 `--torch-backend`；macOS 不需要这个参数；
Windows + AMD 只能按下面的 URL 装，没有 uv 捷径。虚拟环境在非 Windows 上用 `source .venv/bin/activate` 启用。

| 平台 + 硬件 | 第 2 步：torch 三件套 | 第 4 步：patched CTranslate2 |
| --- | --- | --- |
| Linux（含 WSL2）+ NVIDIA | `--index-url https://download.pytorch.org/whl/cu128` | Release 里文件名带 `cu128` 与 `manylinux` 的 wheel |
| Linux + AMD | `--index-url https://download.pytorch.org/whl/rocm7.1` | Release 里文件名带 `rocm` 与 `manylinux` 的 wheel；**系统需装 ROCm 7.2**（wheel 按 7.2 编，torch 自带的是 7.1 运行库） |
| Windows + AMD | 先装 ROCm SDK 四件，再装 `2.9.1+rocm7.2.1` 三件套；第 3 步改用 `[asr-win-rocm]`（见下面「Windows 上的 A 卡」） | Release 里文件名带 `rocm` 与 `win_amd64` 的 wheel |
| Windows + Intel | `--index-url https://download.pytorch.org/whl/xpu` 装 2.11.0 / 2.11.0 / 0.26.0 三件套；第 3 步改用 `[asr-win-xpu]`（见下面「Intel 显卡」） | Release 里文件名带 `cu128` 与 `win_amd64` 的 wheel（识别仍在 CPU 上跑） |
| macOS（Apple Silicon） | 直接 `pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0`（PyPI 的 macOS 构建） | Release 里文件名带 `cpu` 与 `macosx` 的 wheel |

Release 地址与覆盖命令同 [`ct2-wheel.md`](ct2-wheel.md)：把命令里的文件名换成上表那一份。

- **Linux + NVIDIA**：CTranslate2 按名字加载 cuBLAS，程序会在建模型前从 `nvidia-cublas-cu12`（CUDA 版 torch
  的依赖）预载它，不需要设置 `LD_LIBRARY_PATH`。
- **AMD**：PyTorch 的 ROCm 版自带一套 ROCm 库，CTranslate2 的 ROCm 版使用系统安装的 ROCm，同一进程里会有
  两套运行库——这是未经验收的主要风险之一。遇到异常先加 `--device cpu` 确认，再反馈。
- **Windows 上的 A 卡**：AMD 只在 `repo.radeon.com/rocm/windows/` 发 Windows 的 ROCm 版 PyTorch，最新一版
  （`rocm-rel-7.2.1`，2026-09-17 核）停在 **2.9.1**，而 `[asr]` 钉 2.11.0——用 `[asr]` 装，pip 会把 ROCm 版
  换成 PyPI 的 2.11.0（不带 ROCm），显卡就用不上了，而且**不会报错**。所以这个组合用另一个 extra
  `[asr-win-rocm]`，它就是 `[asr]` 换成 AMD 那条 torch 线。命令：

  前提（AMD 的要求）：**Python 3.12**、显卡驱动 **26.2.2**，卡是 RX 7000 / 9000 系或 Strix APU，
  系统 Windows 11——这条线是 AMD 的预览版。按 AMD 的顺序，先 ROCm SDK 四件，再 torch 三件套：

  ```powershell
  # 1. ROCm SDK（torch 的 wheel 依赖它们，缺了解析就会失败）
  pip install --no-cache-dir `
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl `
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl `
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
  # 2. torch 三件套
  pip install --no-cache-dir `
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl `
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl `
      https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl
  # 3. 本项目
  pip install -e ".[asr-win-rocm,harness]"
  ```

  ⚠ **接着补上「用 pip 安装」的第 4 步**：覆盖安装 Release 里带 `rocm` 与 `win_amd64` 的补丁 wheel。它用的 ROCm 运行库
  就是第 1 步装的 ROCm SDK，程序在建模型前向 SDK 要库的位置，不用改 `PATH`。

  ⚠ **这个 wheel 没有在任何一张 A 卡上跑过**。识别报错或结果异常时，先加 `--device cpu` 对照，再改装上面 Windows
  那个 `cu128` wheel 退回旧状态：识别回到 CPU（每趟多一条 `an AMD card needs the ROCm/HIP build` 的告警），
  人声分离、silero 辅助与第二模型校验照样上卡——分离本来就是整条管线最重的一段（CPU 路径上约占八成耗时）。
  两种结果都请反馈。Linux + AMD 用 `[asr]` 即可：那条线有 `2.11.0+rocm7.1`。
- **macOS**：目前只走 CPU。人声分离在 CPU 上最慢，长音频可能耗时很久。
- **自检**：`torch.__version__` 应分别带 `+cu128` / `+rocm7.1` / `+rocm7.2.1` / `+xpu`（macOS 不带后缀）；
  `ctranslate2.__version__` 应以 `+finesub…` 结尾，末段是 `cu128` / `rocm7.2` / `cpu`。

### Intel 显卡

Intel 卡用 XPU 版 PyTorch，设备名是它自己的：`--device xpu`。步骤与上面相同，区别同样只有两处：
**torch 从哪个索引装**，以及**装哪个补丁版 CTranslate2**。

前提：**Python 3.12**（XPU 的 torch 轮子本身也有 cp313/cp314，但这个组合必须用 3.12：补丁版 CTranslate2
只有 cp312 的 Windows 轮子）、Windows、Intel Arc A / B 系独显或带 Arc 核显的 Core Ultra（PyTorch 的 Intel GPU
支持列表；Iris Xe、UHD 等老核显装上也用不了）。命令：

```powershell
# 2. torch 三件套。XPU 索引只含 PyTorch 官方发布的包（torch 三件套、triton-xpu，以及它代理的 PyPI 依赖）
pip install --index-url https://download.pytorch.org/whl/xpu `
    torch==2.11.0+xpu torchaudio==2.11.0+xpu torchvision==0.26.0+xpu
# 3. 本项目
pip install -e ".[asr-win-xpu,harness]"
```

⚠ 这条命令只从 XPU 源解析，所以第 2 步只装这三件；第 3 步不带索引，其余依赖仍从 PyPI 走。
`[asr-win-xpu]` 就是 `[asr]` 换成 XPU 那条 torch 线，并没有 Intel 版的 CTranslate2。

⚠ **接着补上「用 pip 安装」的第 4 步**：Windows 没有 CPU 轮子，用 Release 里 `cu128 + win_amd64` 那个
补丁 wheel。CUDA 运行时是延迟加载的，不装 N 卡驱动也能跑 CPU 识别（CPU 路径走它自带的 oneDNN GEMM）。

⚠ **识别始终在 CPU 上**：CTranslate2 没有 Intel GPU 后端，上卡的只有人声分离、Silero VAD 辅助与语言重解用的
Qwen 裁判；`--qwen-verify` 的尾段二次校验跟随识别设备留在 CPU（float32）。运行时会有一条
`cpu-fallback` 告警说明这一点，属正常，不是配置错了。装完自检：

```powershell
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

⚠ 这个组合还没有任何验收记录，支持等级是实验；结果异常时先加 `--device cpu` 对照，再反馈。

## 自检

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望:2.11.0+cu128 True —— 版本号不带 +cu128 就是装到 CPU 版了,重装 torch 三件套

python -c "import ctranslate2; print(ctranslate2.__version__)"
# 期望:4.8.1+finesub0.6.0.cu128 —— 只有 4.8.1 就是原版,重跑 CT2 覆盖命令

python -m finesub.pipeline --help
# 期望:参数表。本项目不再安装 asr-pipeline / vad-asr 之类的命令,仓库开发版
# 一律 python -m;PATH 上如果有 finesub,那是 CLI 发行版装的,另一套运行环境
```

## 注意事项

- **重装或升级项目后 CT2 会恢复为原版**：重装会按 `==4.8.1` 装回 stock 版，重新执行 CT2 覆盖命令
  即可（同 [ct2-wheel.md](ct2-wheel.md)）。
- URL 输入还需安装 yt-dlp:`uv pip install yt-dlp`（或 `pip install yt-dlp`）;`finesub` CLI 的
  托管运行环境已内置该依赖，无需此步骤。
- 跑测试加装 `dev` extra:`... -e ".[asr,harness,dev]"`。
- 若完全不想管理环境，可直接使用 CLI 发行版（README「命令行 CLI」,
  [cli/README.md](../../cli/README.md)），其运行环境完全托管，卸载彻底。
