# FineSub Desktop

FineSub Desktop 是现有 FineSub CLI/pipeline 的可选 Windows 客户端。它使用
pywebview 承载静态导出的 Next.js 界面，通过受限 bridge 调用 Python 服务，并在
隔离 worker 进程中运行 `src/asr_playground/pipeline.py` 提供的生产管线。桌面端不会替换 CLI。

Desktop 使用独立于 Python 包版本的发布版本；唯一版本源是 `desktop/VERSION`。
构建脚本、CI、launcher、前端 package 和 Windows 版本资源必须与该文件保持一致。

## 架构

```text
Next.js static UI
        │ pywebview API
DesktopBridge
        ├── JobManager ── isolated worker ── existing FineSub pipeline
        ├── ResourceManager ── uv / Python 3.12 / FFmpeg
        ├── SettingsStore ── local API keys
        └── signed GitHub Release check
```

任务恢复会复用原 task ID、请求、输出路径和历史记录。现有 pipeline 会跳过已经
完成的中间产物，并从同一个 LLM artifact 目录读取 session/window checkpoint。

当前版本只检查签名的 GitHub Release；应用内更新安装和独立 updater 暂不打包。
发现新版本后，用户需要在浏览器中打开 Release 页面并手动下载安装器。

## 依赖

Python extras 位于根目录 `pyproject.toml`：

- `desktop`：桌面运行依赖，包括 pywebview、Pydantic、HTTP、版本及签名校验。
- `dev`：测试和桌面构建依赖，包括 Pillow、PyInstaller 和 hooks。
- `asr` / `harness`：原 FineSub pipeline；不属于桌面壳本身。

前端使用 Node.js 22，依赖由 `frontend/package-lock.json` 锁定。发布包使用
Windows 自带的 `Microsoft YaHei UI`、`Segoe UI`、`Cascadia Mono` 和
`Consolas`，不携带 Web Font。

最终用户的 Windows/Python 3.12/CUDA 12.8 AI 环境锁在：

```text
runtime/pylock.win-py312.toml
```

更新 AI 依赖后，在仓库根目录重新生成：

```powershell
uv pip compile pyproject.toml `
  --extra asr `
  --extra harness `
  --python-platform x86_64-pc-windows-msvc `
  --python-version 3.12 `
  --torch-backend cu128 `
  --format pylock.toml `
  --output-file desktop/runtime/pylock.win-py312.toml
```

`pylock.toml` 为每个分发包记录来源、平台 wheel 和 SHA-256，避免 PyPI 与
PyTorch 索引混用时降低 uv 的依赖混淆防护。运行环境只按该 lock 安装第三方依赖；
worker 通过 `PYTHONPATH` 直接运行当前版本随包发布的 FineSub 源码。lock 变化会使
runtime marker 失效并触发环境重建，普通应用更新不会无故重装数 GB AI 环境。

## 开发

要求：

- Windows x64
- Python 3.12
- Node.js 22
- Edge WebView2

```powershell
# 桌面 UI、后端、测试和构建依赖
.\desktop\scripts\setup-dev.ps1

# 如需在开发窗口中实际运行 ASR/LLM pipeline
.\desktop\scripts\setup-dev.ps1 -IncludePipeline

.\desktop\scripts\run-dev.ps1
```

API Key 保存在 FineSub Desktop 用户数据目录的 `.env` 中，不会返回给前端，但
当前仍是本机明文文件。Desktop 的 Gemini、Exa、Tavily 字段分别注入 CLI 的
`GEMINI_FREE`、`EXA_KEYS`、`TAVILY_KEYS`；Gemini 用于翻译，Exa/Tavily
仅在启用网页搜索时使用。旧版 Desktop 保存的三个单 Key 变量会在首次读取时
一次性迁移并改写。raw SRT 全程本地处理；纠错翻译会按所选 LLM profile 向
Gemini 上传必要的音频或视频片段。

## 测试

桌面测试不属于根项目的默认 pytest `testpaths`，必须显式运行：

```powershell
python -m compileall -q desktop
python -m pytest -q -n 0 desktop/backend/tests desktop/scripts/tests

Push-Location desktop/frontend
npm test
npm run typecheck
npm run build
Pop-Location

python desktop/scripts/verify_static_export.py desktop/frontend/out/index.html
```

`.github/workflows/desktop-ci.yml` 在 Windows 上执行同样的后端、前端和
PyInstaller bootstrap smoke build。根项目原有 CI 不承担桌面验证。

## 构建

正式 bootstrap 需要非示例 launcher 配置和可信 Ed25519 公钥：

```powershell
Copy-Item desktop/resources/launcher.example.json `
  desktop/resources/launcher.json
# 创建被 desktop/.gitignore 忽略的：
# desktop/resources/trusted-update-keys.json

.\desktop\scripts\build-bootstrap.ps1
```

未显式传入 `-Version` 时，构建脚本会读取 `desktop/VERSION`；发布自动化如需
显式传值，也应先从该文件读取，避免生成版本不一致的资源。

bootstrap 只包含 `FineSub Desktop.exe`，不会生成 `FineSub.exe` 兼容副本，也
不会打包独立 updater。

使用 Inno Setup 6 生成手动安装器：

```powershell
.\desktop\scripts\build-installer.ps1 `
  -ApplicationDirectory ".\dist\bootstrap\FineSub Desktop.dist"
```

发布私钥必须位于仓库之外。`scripts/build-release.ps1` 中的签名更新包工具暂时
保留供未来恢复应用内安装时使用，但当前桌面公共 API 不会下载或应用这些包。
