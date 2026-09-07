# 仓库（源码）安装

本页面向两类读者：需要运行仓库开发版的开发者，以及希望将 pipeline 安装到自有 Python 环境（而非使用
`finesub` CLI 的托管运行环境）的用户。安装后的入口为 `python -m finesub.pipeline`（与 README 中
`finesub` 命令的参数一致）；传入多个输入或使用 `--manifest` 即为批量处理，没有独立的批量入口。

需要自行安装 ffmpeg 并加入 PATH，且须包含 `libx264` 编码器（LLM 纠错阶段的每份剪辑均使用该编码器）。
Windows 的生产路径使用 NVIDIA CUDA；Apple Silicon macOS 使用 MLX + MPS/CoreML，不需要
CTranslate2。Linux 尚未完成 patched CTranslate2 分发与端到端验收，本页不把它列为受支持路径。
没有可用加速器时仍可运行兼容性 CPU 路径，但速度和 refine 契约会降级。默认使用 uv；坚持使用 pip 请跳到
[第二节](#用-pip-安装)，该路径的坑更多。

## 用 uv 安装（默认）

Python 3.12 由 uv 自动准备，无需预装。在源码目录下先按平台准备工具：

```console
# Apple Silicon macOS
brew install uv ffmpeg

# Windows（安装后新开终端）
winget install astral-sh.uv
```

### Apple Silicon macOS

```console
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[asr,harness]"
```

Darwin/arm64 的依赖 marker 会精确安装 `mlx-whisper==0.4.3` 与 `mlx==0.32.2`，跳过
faster-whisper/CTranslate2。`--asr-backend auto` 映射到固定 revision 的
`mlx-community/whisper-large-v3-turbo`。人声分离保留 audio-separator 的 MPS/CoreML
选择，Silero VAD 辅助也使用 MPS；显式 `--device cpu` 或 `--gpu-tier cpu` 仍会强制 CPU。

### Windows + NVIDIA CUDA

```powershell
# 创建并启用虚拟环境
uv venv --python 3.12
.venv\Scripts\activate

# ASR 全栈 + LLM 层(含 Qwen3-ASR 第二模型校验,首次运行时自动下载模型 ~1.5GB)
# --torch-backend cu128 确保拿到 CUDA 版 torch(PyPI 上的 Windows 构建不带 CUDA)
uv pip install --torch-backend cu128 -e ".[asr,harness]"

# ASR 必需的 patched CTranslate2(原版装上也跑不了,详见 ct2-wheel.md)
uv pip install --reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.4.0/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl"
```

## 用 pip 安装

Apple Silicon macOS：

```console
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[asr,harness]"
```

以下 CUDA 索引与 patched CTranslate2 步骤仅适用于 Windows/NVIDIA。pip 没有
`--torch-backend`，需要自行绕开两个由 uv 自动处理的问题：**torch 必须从
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
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.4.0/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl"
```

为何将第 2、3 步分开：`torch==2.11.0` 这一约束同时被 PyPI 的 CPU 构建与 `+cu128` 构建满足，在单条
命令中使用 `--extra-index-url` 时，pip 无法保证会选择哪一个；先将 CUDA 版装入环境后，后续解析只会
沿用该版本。

## 自检

Apple Silicon macOS：

```console
python -c "import mlx, mlx_whisper; print(mlx.__version__, mlx_whisper.__version__)"
# 期望：0.32.2 0.4.3

python -m finesub.pipeline --help
# --asr-backend 应列出 auto、fw-refine、mlx-refine
```

Windows/NVIDIA：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望:2.11.0+cu128 True —— 版本号不带 +cu128 就是装到 CPU 版了,重装 torch 三件套

python -c "import ctranslate2; print(ctranslate2.__version__)"
# 期望:4.8.1+finesub0.4.0.cu128 —— 只有 4.8.1 就是原版,重跑 CT2 覆盖命令

python -m finesub.pipeline --help
# 期望:参数表。本项目不再安装 asr-pipeline / vad-asr 之类的命令,仓库开发版
# 一律 python -m;PATH 上如果有 finesub,那是 CLI 发行版装的,另一套运行环境
```

## 注意事项

- **Windows 重装或升级项目后需复核 CT2**：正确的 direct wheel 版本为
  `4.8.1+finesub0.4.0.cu128`（同 [ct2-wheel.md](ct2-wheel.md)）。Apple Silicon 不执行该步骤。
- macOS 紧急 CPU 兼容路径需额外安装精确版本 `faster-whisper==1.2.1` 和
  `ctranslate2==4.8.1`，再显式传 `--asr-backend fw-refine --device cpu --gpu-tier cpu`。
  该路径使用 stock teacher-force timestamps，不提供与 `mlx-refine`/patched CT2 等价的
  `alignment_events` 和 segment confidence，仅用于故障降级。
- URL 输入还需安装 yt-dlp:`uv pip install yt-dlp`（或 `pip install yt-dlp`）;`finesub` CLI 的
  托管运行环境已内置该依赖，无需此步骤。
- 跑测试加装 `dev` extra:`... -e ".[asr,harness,dev]"`。
- 若完全不想管理环境，可直接使用 CLI 发行版（README「命令行 CLI」,
  [cli/README.md](../../cli/README.md)），其运行环境完全托管，卸载彻底。
