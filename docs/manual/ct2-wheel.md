# 安装 patched CTranslate2（ASR 必需）

ASR 阶段需要一份打过补丁的 CTranslate2。PyPI 上的版本装了跑不起来，要额外装一次。

适用于 **Windows + Python 3.12 + NVIDIA 显卡**。其它组合见
[`../ct2-distribution.md`](../ct2-distribution.md)。

## 装

先按[仓库安装](repo-install.md)（uv 或 pip 流程）装好项目本体——那一步会装进
**stock** CTranslate2，然后**必须**用补丁 wheel 覆盖它（仓库安装手册里
已含此步；本页供出问题时排查）：

```bash
# uv：
uv pip install --reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bwtrefine1/ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl"

# pip：
pip install --force-reinstall --no-deps "https://github.com/caca2331/finesub/releases/download/ct2-4.8.1%2Bwtrefine1/ctranslate2-4.8.1+wtrefine1.cu128-cp312-cp312-win_amd64.whl"
```

## 检查装对了没

```bash
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

- 打印 `4.8.1+wtrefine1.cu128` —— 正确。
- 打印 `4.8.1` —— 还是原版，重跑上面的覆盖安装命令。

## 出问题时

**跑 ASR 报 `CTranslate2 was not built with the WT refine trace extension`**
装的是原版。重跑覆盖安装命令。

**重装或升级项目之后又报这个错**
重装项目会按 `==4.8.1` 把原版装回来。重跑覆盖安装命令。
