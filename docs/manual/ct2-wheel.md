# 安装 patched CTranslate2(ASR 必需)

ASR 阶段需要一份打过补丁的 CTranslate2。PyPI 上的原版无法运行本项目的 ASR，必须额外安装补丁版本。

⚠ **用 `uv tool install finesub` 装的安装版**自带补丁版，不用照本页手动装：遇到下面这些 CTranslate2 报错，先跑
`finesub setup --repair` 按锁重建运行环境。本页的「装」「验证安装」与故障排查里的覆盖安装只适用于源码安装；
最后一条（显卡上的报错）两种装法都适用。

下面的命令适用于 **Windows + Python 3.12 + NVIDIA 显卡**。其它平台用同一个 Release 里对应的 wheel，
覆盖方式相同，逐平台说明见 [仓库安装](repo-install.md)「其他平台」。文件名末段与平台的对应是
**按平台 × 显卡**，不是只看显卡：

| 平台 + 显卡 | 末段 |
| --- | --- |
| Windows / Linux + NVIDIA | `cu128` |
| Windows / Linux + AMD | `rocm7.2` |
| Windows + Intel | `cu128` |
| macOS | `cpu` |

Windows + Intel 也用 `cu128` 那份：识别在 CPU 上跑，CUDA 库是延迟加载的，不需要 N 卡驱动。

五个都由 CI 构建。⚠ 两个 `rocm7.2` 没有在任何一张
A 卡上跑过；识别报错或结果异常时，改装同一 Release 里同平台的 `cu128` 那份（没有 N 卡也能在 CPU 上解码）：
识别回到 CPU，其余阶段照常用卡。两种结果都请反馈。

## 装

先按[仓库安装](repo-install.md)（uv 或 pip 流程）装好项目本体。那一步会装入 **stock** 版 CTranslate2，随后**必须**用补丁 wheel 覆盖它（仓库安装手册已包含该步骤；本页供故障排查时参考）:

```powershell
# uv:
uv pip install --reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.6.0/ctranslate2-4.8.1+finesub0.6.0.cu128-cp312-cp312-win_amd64.whl"

# pip:
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bfinesub0.6.0/ctranslate2-4.8.1+finesub0.6.0.cu128-cp312-cp312-win_amd64.whl"
```

## 验证安装

```powershell
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

- 打印 `4.8.1+finesub0.6.0.cu128` —— 正确（其它平台末段换成上表对应的那个）。
- 输出为 `4.8.1` 则说明仍是原版，请重新执行上面的覆盖安装命令。

## 故障排查

**运行 ASR 时报 `CTranslate2 was not built with the WT refine trace extension`**
当前安装的是原版，请重新执行覆盖安装命令。

**报 `CTranslate2 was not built with WT disfluency weight support`**
装的是早期的补丁版，缺语气词检测需要的那部分，请用本页的 wheel 重新覆盖安装。这项检查在加载模型时进行，
不会跑完一趟再交出空字幕。

**AMD 卡上识别回退到 CPU，警告说 `an AMD card needs the ROCm/HIP build`**
装的是 CUDA 版或 CPU 版，换成文件名带 `rocm` 的 wheel（Windows 上是带 `rocm` 与 `win_amd64` 的那份，
见 [`repo-install.md`](repo-install.md)「Windows 上的 A 卡」）。

**重装或升级项目后再次出现该错误**
重装项目会按 `==4.8.1` 装回原版，请再次执行覆盖安装命令。

**报 `FlashAttention is built for ... only`，或在 RTX 30 / 40 系上识别结果异常**
设环境变量 `FINESUB_CT2_FLASH_ATTENTION=off` 再跑（PowerShell：`$env:FINESUB_CT2_FLASH_ATTENTION = "off"`），安装版与源码安装都一样。
再跑之前先删掉识别阶段的产物及其下游，否则已完成的阶段照用旧结果、不会重做；删哪几个见
[运行产物「重跑某一阶段」](outputs.md#重跑某一阶段需删除的产物)的「语音检测 / 识别 / 分句」一行。成品会跟着删掉，手改过的副本先另存。
原因：RTX 30 系及更新的 N 卡上，补丁版默认开一项识别加速（flash attention），自己转换的部分模型用不了它，RTX 30 / 40 系上这项加速也还没在实卡上验证过。
