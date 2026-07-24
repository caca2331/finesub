# ASR Confidence Explorer

`index.html` 是离线分析快照（**顶层单文件，无 iframe 包装**），当前内嵌 `exp-whisper-segment-map`
分支下 8 个 BV 的 `stable-segmap` 数据。两个 tab：

## Tab 1 · 置信度片段筛选

- 四个条件：加权字数速率、segment confidence、加权 word confidence、VAD weighted energy；
- 每个条件都是**命中区间 `[最小, 最大]`**（任一端留空即不限），不再是单一大于 / 小于；
  两个 confidence 的默认区间上界为 `0.2`；
- OR / AND 组合只作用于已启用的条件，四个条件全部关闭时显示全部片段；
- 每个片段多一行用 `|` 分隔的 word 级 token；
- 每行两个播放按钮：「片段」只放该片段，「±5s 原声」额外播放首尾各 5 秒原声；
- 可按速率、两种 confidence 或 VAD weighted energy 排序。

## Tab 2 · SRT × 音频

- 页面在**构建时**读取同名 `index.txt`（每行 `SRT 路径 | 音频路径`，`#` 开头为注释），把每个
  pair 及其 SRT cue **预烘焙**进 `index.html`——因此**无需任何文件权限**即可预载展示；
- 每个 pair 默认折叠，展开后逐条字幕两个播放按钮（片段 / ±5s 原声），可删除（仅内存，不改动 `.txt`）；
- 也可手动 **拖拽 / 选择 / 输入** 添加 pair（拖拽、文件选择走 blob，无需任何权限）；
- 若浏览器以允许本地文件访问的方式启动，点「重新读取同名 .txt」可实时载入编辑后的 `index.txt`。

### 为什么要预烘焙

浏览器允许 `<audio src="file:">` 播放，但**禁止脚本读取 `file://` 文本**（XHR / fetch），除非以
`--allow-file-access-from-files` 启动。所以普通打开方式下同名 `.txt` 和各 SRT 的文本都读不到；
构建时把 pair 的 cue 烘焙进页面、音频仍用绝对 `file:` URI 播放，从而开箱即用。

## 维护策略

**此工具保留，但无需自动维护。**

- 它不是生产 pipeline 的组成部分。
- 修改 stable schema、VAD/ASR 算法、reference 数据或通用文档时，不要顺带更新此工具。
- 只有用户明确要求维护、刷新数据或适配新环境时，才更新 `index.html` 和本 README。
- 刷新数据需重新嵌入目标 stable 数据与 SRT 快照，并逐项验证：区间筛选、排序、OR/AND、两个播放
  按钮、word 行；SRT tab 的预载 / 折叠 / 删除 / 手动添加 / 本地音频播放。

## 打开与限制

- 直接用支持本地文件访问的 Chromium/Edge 打开 `index.html`。音频 URI 指向生成环境中的本地 vocal
  FLAC，工作区移动后播放需重新适配。
- 顶层单文件页面：内联 CSP 已含 `media-src file:`（音频）与 `connect-src file:`（实时读取 .txt /
  SRT，需 `--allow-file-access-from-files`）。旧的 iframe 包装与 `enable_local_audio.py` 流程已弃用。
- 同名 `index.txt` 是**本地侧车**：被 `.gitignore` 的 `*.txt` 规则忽略、不入库。仓库只提交
  `index.html`（已内嵌构建快照）；需要实时编辑侧车时，自行在同目录放置 `index.txt`。
- 页面不读取网络数据、不调用业务 API，也不会修改任何 stable JSON 或 `.txt`。
