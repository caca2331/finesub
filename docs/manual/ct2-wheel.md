# 安装 patched CTranslate2（仅 Windows/CUDA）

Windows/CUDA 的 `fw-refine` ASR 阶段需要一份打过补丁的 CTranslate2。PyPI 上的原版无法提供
FineSub 所需的 decoder trace。

适用于 **Windows + Python 3.12 + NVIDIA 显卡**。其它组合见
[`../ct2-distribution.md`](../ct2-distribution.md)。

Apple Silicon macOS 的 `auto` 后端是 `mlx-refine`，不安装本页 wheel；见
[`../mlx-refine.md`](../mlx-refine.md)。

## 装

先按[仓库安装](repo-install.md)（uv 或 pip 流程）装好项目本体。那一步会装入 **stock** 版 CTranslate2，随后**必须**用补丁 wheel 覆盖它（仓库安装手册已包含该步骤；本页供故障排查时参考）:

```powershell
# uv:
uv pip install --reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.4.0/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl"

# pip:
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.4.0/ctranslate2-4.8.1+finesub0.4.0.cu128-cp312-cp312-win_amd64.whl"
```

## 验证安装

```powershell
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

- 打印 `4.8.1+finesub0.4.0.cu128` —— 正确。
- 输出为 `4.8.1` 则说明仍是原版，请重新执行上面的覆盖安装命令。

## 故障排查

**运行 ASR 时报 `CTranslate2 was not built with the WT refine trace extension`**
当前安装的是原版，请重新执行覆盖安装命令。

**重装或升级项目后再次出现该错误**
重装项目会按 `==4.8.1` 装回原版，请再次执行覆盖安装命令。
