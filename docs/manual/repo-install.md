# 仓库（源码）安装

面向两类人：要跑仓库开发版的开发者，和想把 pipeline 装进自己管理的 Python
环境（而不是用 `finesub` CLI 的托管运行环境）的用户。装完后的入口是
`asr-pipeline`（README 里的 `finesub` 同参数等价）和
`python -m asr_playground.batch`。

需要 NVIDIA GPU 与 ffmpeg（自备并加入 PATH）。默认用 uv；坚持 pip 的话跳到
[第二节](#用-pip-安装)，那条路的坑更多。

## 用 uv 安装（默认）

没有 uv 的话先 `winget install astral-sh.uv`；Python 3.12 由 uv 自动准备，
无需预装。在源码目录下：

```powershell
# 创建并启用虚拟环境
uv venv --python 3.12
.venv\Scripts\activate

# ASR 全栈 + LLM 层（含 Qwen3-ASR 第二模型校验，首次运行时自动下载模型 ~1.5GB）
# --torch-backend cu128 确保拿到 CUDA 版 torch（PyPI 上的 Windows 构建不带 CUDA）
uv pip install --torch-backend cu128 -e ".[asr,harness]"

# ASR 必需的 patched CTranslate2（原版装上也跑不了，详见 ct2-wheel.md）
uv pip install --reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bwtrefine1/ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl"
```

## 用 pip 安装

pip 没有 `--torch-backend`，两个 uv 帮你挡掉的坑要自己绕：**torch 必须从
download.pytorch.org 拿 CUDA 构建**（PyPI 上的 Windows torch 不带 CUDA，装错后
GPU 路径静默失效、慢到怀疑人生），以及 **patched CTranslate2 要单独装**。
还需自备 **Python 3.12**（pip 不会帮你下载解释器）。

在源码目录下，顺序不能乱：

```powershell
# 1. 虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 2. 先从 PyTorch 官方索引装 CUDA 版 torch 三件套（版本必须与 pyproject 钉的一致）
pip install torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 3. 再装项目（torch 已满足约束，pip 不会用 PyPI 的 CPU 版覆盖它）
pip install -e ".[asr,harness]"

# 4. patched CTranslate2（原版装上也跑不了 ASR，见 ct2-wheel.md）
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bwtrefine1/ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl"
```

为什么 2、3 要分开：`torch==2.11.0` 这个约束同时被 PyPI 的 CPU 构建和
`+cu128` 构建满足，一条命令里给 `--extra-index-url` 时 pip 选哪个没有保证；
先把 CUDA 版装进环境，后续解析就只会沿用它。

## 自检

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望：2.11.0+cu128 True —— 版本号不带 +cu128 就是装到 CPU 版了，重装 torch 三件套

python -c "import ctranslate2; print(ctranslate2.__version__)"
# 期望：4.8.1+wtrefine1.cu128 —— 只有 4.8.1 就是原版，重跑 CT2 覆盖命令
```

## 注意事项

- **重装/升级项目后 CT2 会被打回原版**：重装会按 `==4.8.1` 把 stock 版装回来，
  重跑 CT2 覆盖命令即可（同 [ct2-wheel.md](ct2-wheel.md)）。
- URL 输入另需 `uv pip install yt-dlp`（或 `pip install yt-dlp`）；Desktop 和
  `finesub` CLI 的托管运行环境已内置，无需此步。
- 跑测试加装 `dev` extra：`... -e ".[asr,harness,dev]"`。
- 完全不想管环境的话，出口是 CLI 发行版（README「命令行 CLI」，
  [cli/README.md](../../cli/README.md)），运行环境全托管、卸载干净。
