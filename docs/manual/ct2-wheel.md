# 安装 patched CTranslate2（ASR 必需）

ASR 阶段需要一份打过补丁的 CTranslate2。PyPI 上的版本装了跑不起来，要额外装一次。

适用于 **Windows + Python 3.12 + NVIDIA 显卡**。其它组合见
[`../ct2-distribution.md`](../ct2-distribution.md)。

## 装

两条命令，**顺序不能反**：

```bash
pip install -e ".[asr,harness,dev]"
```

```bash
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bwtrefine1/ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl"
```

## 检查装对了没

```bash
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

- 打印 `4.8.1+wtrefine1.cu128` —— 正确。
- 打印 `4.8.1` —— 还是原版，重跑上面第二条命令。

## 出问题时

**跑 ASR 报 `CTranslate2 was not built with the WT refine trace extension`**
装的是原版。重跑第二条命令。

**重装或升级项目之后又报这个错**
`pip install -e .` 会把原版装回来。重跑第二条命令。
