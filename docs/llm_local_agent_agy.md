# agy（Antigravity CLI）：唯一具备多模态能力的 agent

> 本文从 [`llm_local_agent.md`](llm_local_agent.md) 拆出（2026-08-16，原文 2,198 行），
> **章节号已按本文重编、从 1 起**——每份文档独立编号，才能在中间插入新节。跨文档引用
> 一律写成 `` `文件名` §N `` 的形式，由 `test_doc_links.py` 守着：指不到的章节号会红。

来源：`../common/tmp/audio_video_multimodal_report.md`（初始媒体报告）与
`tmp/agy-3_7-flash-probe.md`（3.7 Flash headless/A1 复测）。Antigravity CLI 底层是 Gemini
多模态，通过 `view_file` 工具读取媒体文件。

## 1 定位与 catalog 行

- 新 provider tier `LOCAL_AGY`，backend 仍是 `local_agent`，由 tier 选 driver（§2 已有的机制）。
- Catalog 已加入 fact `local-agy-gemini-3_8-flash` 并路由 target `local-agy-media-gemini-3_8-flash`：CLI base id
  `gemini-3.8-flash`，`max_input_tokens=1048576`、`context_window=1048576`（单池，包络 983040）、
  `max_output_tokens=65536`、`quality_score=75`、
  thinking 映射 `medium,medium,low`、`token_scale=1.0`。`--effort low|medium|high` 与 base id 可
  直接组合；CLI 的 model list 也同时列出 `gemini-3.8-flash-low|medium|high`。3.7 有一份除
  base id、显示名与恒等 thinking 之外逐字段相同的 fact `local-agy-gemini-3_7-flash`，以及同样
  的 media / native 一对 target——出厂模型组用的是这一对。
- **2026-09-02：接线从 3.7 Flash 整体换成 3.8 Flash**（`agy models` 里 3.8 已上，3.7 仍在）。
  owner 同日决定**不做复测就换**：同尺寸的新一代 Flash 默认更好，同素材 A/B 也不指望测出显著
  差别。下面各节的实测数字**全部取自 3.7 Flash**，没有在 3.8 上复测过——形态结论（MIME 拦截、
  容器化、帧采样、project 授权）跟的是 CLI 与工具层，换代不动它们；带具体秒数、字节数和
  命中率的那些数字要当作 3.7 的记录读，不是 3.8 的承诺。
- **2026-09-03：又换回 3.7 Flash**（owner 决定）。一次五对配对实测显示 3.8 在同一抽象档位上
  多想约 1.5 倍，而没有谁能拿出对应的质量收益；3.8 的两个 target 仍然声明着、`--llm-model` 指
  得到，只是不再进任何出厂模型组。于是上一条的注意事项**反过来不再成立**：各节的 3.7 数字重新
  就是主力路径的记录。3.8 那一行另外把 thinking 映射封在 `medium,medium,low`（见
  `docs/manual/model-routing.md` 的 `thinking` 列）。
- **它是唯一 `supports_audio=true` / `supports_video=true` 的 agent 行**，且要额外声明"只有高
  分辨率媒体档"（见 16.3）。Codex 与 Claude Code 都是纯文本。这直接改变一条现有行为：`agent-only` + 媒体开关此前必然"无可用 target"
  （`test_agent_only_media_call_never_falls_through_to_api` 正是钉这个），加入 agy 后该前提
  不再成立，测试要按新语义改写。
- 配额列填 `-1`（与 Codex/Claude 行一致）：本地 agent 由订阅计量，不由我们做 RPM/TPM 限流。
- 2026-08-14 生产复测 CLI 为 `agy.exe 1.1.13`；支持 `--print`、`--conversation`、`--project`、
  `--agent`、`--sandbox`、`--effort` 与 `text|json|stream-json`。`stream-json` 有 `init`、
  `step_update`、终态 `result`，会话 id 可恢复；同一 conversation 连续两 turn 能保留历史，也能
  在第二 turn 重新读取已变化的 protocol 文件。

## 2 音频：MIME 被工具层拦住，必须容器化

3.7 Flash 实测中，`view_file` 在 payload 到达模型**之前**就按 MIME 拒绝裸音频：
`.flac` → `unsupported mime type application/octet-stream`，`.wav` → `audio/wave`。

绕法（报告已验证，转写 6 句全对）：把音频封进带黑画面视频轨的 MP4。

```bash
ffmpeg -y -f lavfi -i color=c=black:s=16x16:r=0.05 -i input.flac \
       -c:v libx264 -tune stillimage -c:a aac -shortest output.mp4
```

成本：音频 **≈32 tok/s**，外加**视觉帧数 × 每帧 token**。
**估算继续按 32（owner 定，2026-08-14）**：它是上限侧的保守值，实际会因抖动与压缩低一些。
一次实测供参考——986.589s 的生产剪辑，Gemini REST 返回 `prompt_audio_tokens=25,476`，
即 25.82 tok/s（见 `llm_local_agent_experiments.md` §3.3）。**不要据此把估算改成 25.8**：上限用于包络才安全。

owner 定的做法：**伪造视频就压成恰好一帧**，开销即一帧常量，可忽略不计。注意报告里的
`r=0.05` 只是在 18.36s 素材上凑巧得到 1 帧，直接照搬会随片长膨胀（300s × 0.05 = 15 帧），
所以要按"一帧"这个目标来编码，而不是照抄那个 fps 数字。按一帧计，300s 窗口的容器开销是
269 / 9,600 ≈ **+2.8%**。

实现由 `containerize_audio_for_agy` 先探测素材时长，再生成覆盖完整音频时长的单帧 H.264 + AAC
容器。真 ffmpeg 回归覆盖 2s 音频：输出时长保留且视频流恰好 1 帧；这也修掉了早期 `-shortest`
把 10s 音频截成约 1s 的错误。

## 3 视频：分辨率不可调，现有 token 公式失真

报告里的 258 tok/帧**不是新常量**：owner 判定仓库预设的 `VIDEO_TOKENS_PER_FRAME_HIGH = 269`
才是对的，258 是非整数视频时长造成的抖动。所以 agy 不需要新的 token 档，它就是**现有的
HIGH 档**——差别只在于走 Gemini REST 时我们主动买了便宜档，走 agy 时买不到。

| 量 | Gemini REST 路径（现状） | agy |
| --- | --- | --- |
| 每帧 token | **71**：`client.py` 显式发 `detail="low"` → `mediaResolution.level=low` | **269**：`view_file` 不暴露 `media_resolution`，只有标准档 |
| 采样 FPS | 0.25，经 `videoMetadata.fps` 下发 | 工具参数不可设，但**剪辑是我们自己压的**（见下） |
| tokens/秒 | `71 × 0.25 = 17.75` | `269 × 0.25 = 67.25`（FPS 对齐后） |

**FPS 可以 hack**：把视频本身压成 0.25 fps 即可，代价是要重压一遍。两个坑：

- 抽帧位置**吸附到整秒**，不是均匀分布——0.4 fps 取到的是 `t=0,3,5,8` 而不是
  `t=0,2.5,5,7.5`。视频档本来就是给时间轴/画面线索用的，取样点偏移要在质量侧评估。
- 重压是额外的 ffmpeg 成本，且与音频容器化那步可以合并做，别分两趟。

**分辨率不可 hack**：`MEDIA_RESOLUTION_LOW` 在 CLI 里没有开关，把画面降到 16×16 也**零收益**
（ViT 最小 patch 网格恒定）。所以 FPS 对齐之后仍有 `269 / 71 ≈ **3.8×**` 的残差，这是硬的。

已落地的改造（比原先设想的小，因为不必新造 token 档）：

1. **fact 声明自己只有高分辨率档**（一个布尔位即可），估算改调
   `video_tokens_per_second(high_resolution=True)`。默认仍是 low，现有行为不变。
2. **规划期用"最贵媒体档"作包络**。窗口几何在路由选定 target **之前**就算好了
   （`chunking.py:53/490`、`exchange_metadata.py:220`、`rate_limit.py:826` 都调全局
   `video_tokens_per_second()`）。这与输入上限是同一类问题，沿用同一原则：**按绑定模型组里
   最贵的那档规划**，否则 agy 一旦被选中，窗口就超包络。
3. `client.py` 的 `detail="low"` + `videoMetadata.fps` 是 Gemini REST 专有路径；agy 走 driver，
   对应动作要挪到**剪辑阶段**——切片时就把 FPS 定死成 0.25。

`ModelLimits.video_high_resolution`、catalog 的 `video_high_resolution_only` 与 planning envelope 已把
这三处口径收成同一事实。agent policy 下的本地媒体引用先不上传；只有路由真实回退到 Gemini REST
候选时才按候选 tier 惰性上传，从而让 `agent-only` 完全不触达 Files API。

**判据是 policy ∧ 绑定**（`window_media_ref` + `ModelRouteCatalog.binds_local_agent()`）：policy 只
做减法，`agent-text-preferred` 配一个全是 Gemini 的预设根本够不着 agent，此时推迟上传只是把同一
次上传挪到后面。所以只有当激活预设（或它回落到的 default）里真有 `local_agent` 成员时才走本地
引用——默认预设下行为与从前的 `api-only` 完全一致。

判据读的是**调用方自己的** route catalog：`window_media_ref` 是自由函数、够不到即将发车的
client，所以两个 stage 显式传 `routes=client.router.routes`。落回全局表会让"剪辑怎么携带"和
"谁来应答"由两张可能不一致的表决定——注入过自定义 routes 的调用点就会白传一次，或者把本地引用
交给一条只有 API 候选的链。

2026-08-13 用 3.7 Flash 复测，所有样本均不超过 20s：18s 裸 WAV/FLAC 仍按上述 MIME 拒绝；
18s 单帧 MP4 能正确转写；20s、0.25fps、恰好 5 帧的五色测试片按顺序识别出
`red, green, blue, yellow, magenta`。16×16 与 640×360 的单帧样本都成功，但 CLI 汇总的 usage
包含两次 agent invocation/system prompt，且不同 run 不单调，不能拿总 `input_tokens` 反推每帧
成本；269 tok/帧仍沿用独立报告的受控测量，不从这批 headless usage 重估。

**2026-08-14 容量复测：长片能整段密集读完；但它的 token 在我们这种调用里不入账。**
`tmp/agy_video_capacity_probe.py` 生成 300 秒、0.25fps、80 帧的纯色片（按规划口径约 21,500
token），一次 `view_file` 读入。

**读取能力：确认整段且密集。** 判据设计过两轮——第一版用 red/green/blue/… 这个**经典测试图案
顺序**，模型答全了，但那种序列没看画面也能猜出来，**该结果作废**。改成 10 段乱序后逐字答对，
再加严到**20 段乱序且颜色重复**（每段仅约 3.75 帧），仍然逐字答对。少量采样帧无法重建这种序列，
所以整条时间轴都到了模型手里。**文本 `view_file` 的返回上限不适用于媒体**——那条上限此前记作
「约 12k token」（`llm_local_agent_experiments.md` §3.2 的顺带观察），2026-08-22 按 transcript
核实为 **≈46k 字节/次且可续读**：回复自带「Content truncated: showing bytes 0-46080 of N … call
this tool again with ContentOffset=46080」，见 §5。

**计费口径：矛盾，且矛盾本身才是结论。** 同一个 `1.4.2` 字段（已用外部报告的逐行数据校验，
6 行中 5 行逐字吻合）：

| 场景 | 该次 `view_file` 让 `1.4.2` 增加 |
| --- | --- |
| 我们的 headless 调用读 80 帧视频 | **约 360** |
| 我们的 headless 调用读大文本 | 16,800 |
| 外部报告的交互式会话读视频（`harvard_lowfps.mp4` / `harvard_025fps.mp4`） | 27,361 / 41,758 |

同样是"view_file 读视频"，一边入账一边不入账。**所以 agy 的账本对媒体不是稳定记账的**，
这比原先"不同 run 不单调"的说法更硬：**任何一侧的数字都不能用来验证每帧成本**。
269 tok/帧既未被证实也未被推翻，规划包络继续按最贵档保守取值；真要核成本只能去账单侧。

**顺带一个固有现象**：我们创建的每一个 agy 会话，都会在首次回复之后立刻插入一条
`CHECKPOINT`（"earlier parts … truncated"），**与内容大小无关**——纯文本、无 view_file 的会话
同样有。它不阻止缓存（`llm_local_agent_experiments.md` §3.2 里命中的两次也有 checkpoint），但它会把媒体挪进
`.tempmediaStorage` 并以 artifact 引用替代，值得在排查上下文问题时先想到。

**单次调用内的工具循环里，媒体既不中途卸载、也不打断缓存**
（`tmp/agy_mid_media_probe2.py`，一次调用 8 步：大文本 → 3 个小文件 → **视频** → 小文件 →
问题文件 → 作答）：

| gen | 该轮之前发生的事 | `uncached` | `cache_read` |
| --- | --- | --- | --- |
| #0–#4 | 大文本 + 三个小文件 | 5,353 → 15,169 | 0（**都没到写入门槛**） |
| #5 | **读完视频** | 15,478 | 0 |
| #6 | 读完小文件 | 1,047 | **14,930** |
| #7 | 读完问题文件 | 1,280 | **15,205** |

- **不卸载**（问法已修正）：第一版把"要问第几段"写在开场 prompt 里，模型完全可以在看视频时
  就把答案记进回复——那版结果作废。改成**问题由第 7 步的文件才揭晓**，视频在第 5 步读入，
  中间还隔着一次工具调用；模型答对 `block 7=magenta / block 16=cyan`，而 transcript 里
  **中间七次 planner 回复全为空**（无预先抄录），最终那次的 thinking 显示它是在作答时才去数
  视频块。所以媒体在同一用户轮次内一直在上下文里，剥离发生在**进入下一个用户轮次**时，
  与外部报告（`audio_video_multimodal_report.md`，随 `common` 项目，不在本仓）§8 一致；
- **不打断缓存**：视频插入（#5）之后的第一次可读（#6）就读回 **14,930**——正是视频之前那段
  前缀。若媒体让前缀失效，这个数不可能出现。（这里没能做出"插入前已有 cache read"的基线，
  原因见下：前几轮压根没到写入门槛。）
- **口径限制不变**：这段视频的 token 在账本里几乎不体现，所以"不打断缓存"严格说是"对一份
  计量器看不见的载荷不打断"。外部报告里媒体**被计入**的那些轮次命中率确实掉到 15%–27%，
  两者不冲突但不能互相替代——要下通用结论，得先解决媒体记账时有时无的问题。

**第二个视频既不挤掉第一个，也不打断缓存**（`tmp/agy_two_videos_probe.py`，一次调用：
大文本 → 视频 A → 视频 B → 问题文件 → 作答；问题在两个视频都读完后才揭晓）：

| gen | 之前发生的事 | `uncached` | `cache_read` |
| --- | --- | --- | --- |
| #0–#1 | 起手 + 大文本 | 5,037 / 13,688 | 0 / 0 |
| #2 | 读完视频 A | 14,000 | 0 |
| #3 | **读完视频 B** | 6,245 | **8,067** |
| #4 | 读完问题文件 | 1,066 | **13,770** |

模型答对 `A=red`（第一个视频的第 4 段）与 `B=yellow`（第二个的第 7 段）——**A 在 B 读入之后
依然可见**；而 B 插入之后 cache_read 不但非零，还继续增长到 13,770，**没有大面积失效**。
口径限制同上：媒体不进这个计数器，所以严格说是"对计量器看不见的载荷不打断"。

**跨 session 不继承缓存**（`tmp/agy_pseudo_turns_probe.py` 同参数跑第二遍）：同一份输入、
全新会话，上一轮已经写下过一模一样的块，但新会话 gen#1（17,020）**没有读到**，仍自己重写、
到 gen#3 才读回 16,328；而上一轮是 gen#2 就命中。**缓存只在会话内继承**，这也是
`session_scope=assignment` 唯一能拿到它的原因。
（「让**首轮**请求本身就越过门槛」没能做到：argv 受 Windows 32k 字符上限约束（约 8k token，
加基线仍不足 16k），而在受控 project 根写 158KB 的 `AGENTS.md` **不进首轮前缀**——gen#0 仍是
4,583。所以只能用"两个全新会话、输入完全相同"这个等价问法，结论同样成立。）

**顺带修正 §3.2 的"隔一次请求才可读"**：真正的规律是**写入有门槛**。三次实验对照——前缀
16,800 与 16,218 的那两次，缓存立刻被写入并在随后 1–2 次请求内可读；而本次 gen#1–#4 停在
13.6k–15.2k，连续五次请求一次都没写入，直到 15,478 之后才出现命中。所以"没命中"要先看
**这一次请求的前缀有没有过门槛**，再谈滞后。

## 4 准入硬门：`view_file` 的读取边界

agy 方案依赖 `view_file` 读媒体，但 16.2/16.3 只讨论了 MIME、封装与 token 成本，**没有规定这个
工具能读到哪里**。如果它接受任意本地路径，那么把它加进授权就等于给 agent 开放用户磁盘的读取，
而 prompt 里的措辞不是安全边界。

先把现状说清楚，免得把这条误读成"agy 独有的新风险"：**读本来就没有隔离**。Codex 的 read-only
sandbox 挡写不挡读，driver 自己记着 `"read_isolation": False`，followups 里也早写明。所以"agent
能读用户磁盘"今天就成立。agy 的不同在于**读取是一个由模型驱动、接受路径参数的显式工具**：
Codex 是"如果它想读，它能读"，agy 是"我们必须把一个读文件的工具授权给它，否则它连本次任务的
媒体都看不到"。前者是既有缺口，后者是我们要主动签字的授权。

因此 A1 增加四条硬门（做不到就 agy 不进生产 target，只留实验）：

1. 本次任务的媒体**复制或链接进 capsule 内的固定路径**，agent 只被告知这一个路径；
2. 优先靠 OS sandbox 或工具代理限定可读范围；退一步至少要有**工具层的路径校验**；
3. 非授权路径必须在**调用发生之前**被拒，而不是事后在事件流里发现；
4. 若该 CLI 既不能限制路径、也不能代理该工具，则 agy 只能作为实验后端，不进 `agent-*` 生产
   policy 的模型组。

2026-08-13 3.7 Flash/CLI 1.1.12 的实测结论：

- 只传 `--sandbox`（即使保留 `request-review`）时，`view_file` 仍能读取 cwd 外的绝对路径；
  `init.tools` 还宣告约 50 个读写、shell、browser、MCP 与 subagent 工具。`--sandbox` 只约束终端，
  不是文件读取边界。
- Declarative agent 的 `tools: [view_file]` 在主 agent 的 `init.tools` 中**没有**缩小工具集，不能单独
  当安全边界。
- Workspace `.agents/hooks.json` 的 `PreToolUse` 可以在调用前 hard-deny。实测的 deny-by-default
  hook 能拒绝 `run_command`、cwd 外 `view_file`，也能在先 `realpath` 后拒绝藏在 workspace 内的
  junction 逃逸，同时允许真实位于 workspace 内的媒体。
- 这套 hook 只有在该 invocation 显式带正确 `--project <id>` 时才加载；漏传会退回
  `default-cli-project`，`/hooks` 为空，边界随即消失。因此 driver 必须创建并持久化 project id，
  每次启动/恢复都显式传入，并在发车前用无配额 `/hooks` 校验 hook 路径与 digest。只检查 cwd 或
  `--agent` 不够。

所以 A1 不再是“CLI 完全做不到”，而是**有条件可过**：生产 driver 必须生成受控 project、固定
allowlist hook、固定 custom agent，并把 project id/hook digest/tool policy 纳入 execution identity；
任一 readiness 检查不符就 fail closed，不把该 fact 加入生产 target。

生产实现会在 episode 同域的 `.agents` 下生成 project、hook、guard script 与 custom agent；每次
调用先以零 token 的 `/hooks` 查询核对唯一 hook 的来源与动作，再显式带 project id、agent 名与
sandbox 发车。媒体先复制/转换进本次 capsule，guard 对请求路径做 `realpath` + `commonpath` 校验；
未知工具、越界路径、junction 逃逸和 hook 漂移均在模型工具执行前失败。2026-08-14 真机用
3.7 Flash/low 跑 18s 音频成功，预处理为 1 帧 MP4（295,808 → 76,878 bytes），返回正确的语音
概述；usage 为 input 16,629 / output 706 / thinking 358。每次 `/hooks` preflight 增加一段零 token
启动延迟，这是 fail-closed 边界的明确成本。

这与"知识库都是公开信息"无关——风险对象是用户磁盘上的其他文件，不是知识库内容。

## 5 MCP 工具结果的内联上限：约 4k 字节，超过即外置成文件、无预览（2026-08-22）

agy 对**大的 MCP 工具回复**不截断也不预览，而是整个写到
`~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/steps/<n>/output.txt`，只给模型一句
「The output was large and was saved to: file:///…」。实测（1.1.18，Gemini 3.7 Flash，
`tmp/agy_result_size_probe.py` 等三个哨兵探针 + brain transcript 对账）：回复 ≤3,639 **字节**内联
可见，≥4,248 字节一律只剩文件路径——单位是回复的 UTF-8 大小，与字符数、行数无关（中文 3000 字
一页 9k 字节被外置，ASCII 单行 6000 字符反而可见）。模型还会用 `read_context(ref="invalid")` 套出
合法 ref、再用 `offset` 只取小页，所以哨兵实验必须对照 transcript 而不是只看答案。配合 §4 的 hook（只放行 `call_mcp_tool`），模型既读不了那个
文件，也就**完全看不见** protocol 与 payload——2026-08-22 第一次 4 窗 canary 正是这样失败的：模型
`submit` 了 `test`、用猜的 ref 反复 `read_context`，两条会话烧掉 19 万 uncached + 81 万 cached
input 才耗尽预算。

对策有两层。第一层是通用的 MCP 分页：`AgentDriverConfig.mcp_page_chars`（按 UTF-8 字节；agy
2800，其余 0），server 只推送放得下的块，其余由 `read_context(ref, offset)` 分页读（换行处断页、
最后一页才记台账）。但中文每页 ≈900 字，一个生产窗要 ~50 页、每页一轮模型调用——所以 agy 实际走
**第二层：块作为文件交给 agy 自己读**（owner 定 2026-08-22，`mcp_block_files = True`）：

- 块本来就以文件存在于 assignment root（`control/protocols/<type>/<digest>.md`、
  `contexts/<digest>/payload-<task>.md`）；server 在 `FINESUB_MCP_BLOCK_FILES=1` 下把每个必读块标
  `read: "file"` 并给出绝对 `path`，按 push 记台账（文件在手 = 有机会看过），`read_context` 仍可用；
- driver 每次 invocation 把本次调用的 assignment root 写进槽位 project 的
  `.agents/view_roots.json`，tool guard 放行 **realpath 落在这些根之下的现有文件**的 `view_file`，
  其余 `view_file` 与一切别的原生工具照旧 deny；agent 文档的 `tools` 加 `view_file`；project 记录
  **不需要**额外 grant（媒体 project 同款，hook 的 allow 即足够）；
- agy 的 `view_file` 文本读取 ≈46k 字节/次，截断时明说「showing bytes 0-46080 of N … call this tool
  again with ContentOffset=46080」，模型可以续读，**不预截断 payload**（owner）；一个生产窗
  ≈ 协议 3 次 + 正文 1 次读取。
- bootstrap 两份模板都写明 `read: file` 的读法（用文件工具读 `path`，截断就按提示续读）。

Claude Code 与 Codex 的等价旋钮（owner 调查 2026-08-22）：Claude 是环境变量 `MAX_MCP_OUTPUT_TOKENS`
（默认 25,000；文件读取另有 `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS`，工具会话不读文件、不设），
driver 只在工具会话的**本次 invocation** 进程环境里设（`ClaudeCodeDriverConfig.mcp_output_tokens` /
`file_read_output_tokens`）；Codex 是 `tool_output_token_limit`（没有独立的文件读取字段），driver 以
`-c` 覆盖随 MCP 声明一起传（`CodexDriverConfig.tool_output_token_limit`）。**三个都是 owner 统一定的
200k token（2026-08-22）**——模型上下文才是真天花板，放宽没有代价；这些值没有实测过各家的真实上限。两家都不写项目级或用户级
配置文件（`.claude/settings.json`、`.codex/config.toml`），与协议总原则一致。

另一条顺带观察：agy 在会话**第一步**就把 bootstrap 压成 CHECKPOINT 摘要（与长度无关，§3 已记），
所以 bootstrap 里的规则必须短、祈使、可被摘要保留。

## 6 原生搜索：第二个 project（2026-08-15 打通）

工具真名是 **`search_web`**（取数）与 **`read_url_content`**（取页），从真机 `system.init` 的工具
表里读出来的，不是猜的。`--retrieval native` 现在能落到 `local-agy-native-gemini-3_7-flash`。

**为什么必须是第二个 project。** 授权本身就是 project 的 `.agents/` 树（hooks.json + guard +
agent document）。两档共用一个 project 就意味着每次调用都要按本次是否联网重写 guard——而
`max_parallel=4`，四个并发调用会在"决定模型能碰什么"的那个文件上打架。两个 project 则各写一次、
永不在运行中被改。

native project **嵌在运行域里面**（`<domain>/.finesub-native`）而不是并排：guard 的读取边界是
`dirname(guard)/../../..`（比 media 档多走一级），因此仍然是运行域本身——capsule 还在那里，
`view_file` 照常能读到任务文件和媒体。

**授权差异**：native guard 放行 `search_web` / `read_url_content`（不检查参数——查询词不是路径，
URL 一旦允许联网就由模型决定），`view_file` 仍按 realpath+commonpath 边界，其余一律 deny。
四份 guard/agent document（capsule 两档 + tool 两档）的 sha256 都进 execution identity，
改任何一份都会移动身份。

### 6.1 tool-session 也要一对 project（2026-08-30 补齐）

上面那一对只服务 **capsule** 路。**tool-session** 路（§4 的第三种 project
`.finesub-tool-<slot>`）在 2026-08-30 之前只有一个不带检索授权的变体，于是：

- 研究轮没有媒体，按传输规则走 tool-session；
- 会话 protocol 里的 `fragment_native_search_v1.md` 让模型联网查专名，模型照办；
- guard 落到 deny 默认，拒了 `search_web`；
- **被拒的调用不产生任何结果步**（agy transcript 里 `search_web` 那步没有对应的偶数号结果
  步，而失败的 `kb_validate` 尚有一条错误结果），模型于是安静地改用 `kb_search` 继续。

统计侧同时失效：`_normalize_agy_events` 只在 `state == "DONE"` 时才把行标成
`item_type=web_search` 并写入 `tool` 名，被拒的调用是无名的 `ACTIVE -> ERROR`，两个条件都不
满足，所以既不进 `search_events` 也不进越权审计；`native_search_not_used` 这条 note 直接由
`search_events` 为空推出，措辞却是「模型没搜」。2026-08-30 的批量跑批据此得出过
「agy 17 条零检索」的错误结论——实际是发起了 11 次、9 次被拒。

修法与 capsule 路同构，且**同样是两个 project 而不是就地改写**：
`.finesub-tool-native-<slot>`，guard 与 agent document 各有一份，由
`AGY_TOOL_GUARD_TEMPLATE` 派生两个变体（一份模板保证两者不漂移）。

**prompt 也必须一起改**。worker bootstrap 的第 4 条原本写死「不要使用 `finesub` 以外的任何
工具」，与 protocol 的联网指令直接冲突——hook 开了授权而 prompt 还在禁止，就等于把矛盾从
「被拒」换成「不敢用」。所以 `agent_tool_worker_v1.md` / `agent_tool_worker_session_v1.md`
第 4 条带 `$retrieval_exception` 占位，授权时嵌入
`fragment_agent_tool_retrieval_v1.md`。**嵌在同一句里而不是另起一段**：agy 会在会话第一步
把 bootstrap 压成 CHECKPOINT 摘要（§3），另起的段落可能被摘掉。

### 6.2 `read_url_content` 还要一条 permission grant（2026-08-30 实测补上）

hook 不是唯一的闸门。agy 在 `read_url_content` 之前还要一条 **permission**，而 headless
模式无法弹窗询问——没有 grant 就自动拒绝，并且**整个 turn 就此结束、不产出 assistant
message**（driver 侧表现为 `LocalAgentTransientError: ... did not contain a final
assistant message`）。也就是说一次被拒的抓取会连带丢掉它前面已经完成的搜索。
`search_web` 不需要这条 grant，所以在模型「只搜不抓」之前，授权看起来是完整的。

规则写进 **project 自己的记录**（`~/.gemini/config/projects/<id>.json` 的 `permissionGrants`），
与 `mcp(finesub/<tool>)` 同一处，**绝不写用户全局 settings**。取值是通配 `read_url(*)`，
理由与 native guard 不检查参数一致：一旦允许联网，开哪一页就是模型的事；域名白名单只会在
不在名单上的站点静默失败。capsule 与 tool-session 两条 native 路都要写——capsule 那条没有
MCP server，此前完全没有 grant 步骤。

实测（agy 1.1.22，`local-agy-native-gemini-3_7-flash`，tool-session）：补 grant 前
2 次 `search_web` 成功、随后 `read_url_content` 报 ERROR 并中止整次调用；补上之后同一
探针拿到 2 次 `search_web` + 1 次 `read_url_content`，`urls` 有值，答案带模型真实打开的页面。

**教训**：判「模型有没有联网」不能只看本仓库产物。`search_events` 为空既可能是没搜，也可能
是搜了被拒。真相源是 agy 自己的 transcript：
`~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/logs/transcript_full.jsonl`，
conversation handle 即产物里的 `execution_attempts[].session_id`；每个工具调用都跟一个偶数号
结果步，缺结果步即该调用没有完成。

**证实过的事**（真机，2026-08-15）：
- agent document 不是安全边界——`--agent` 指向一个只声明 `view_file` 的文档时，`system.init`
  依然列出全部 56 个工具。这正是 §4 说"第 2 道才是真正的守卫"的原因，也是为什么这里必须动
  hook 而不是只改文档。
- hook payload 带 `toolCall{name,args}`、`modelName`、`conversationId`、`workspacePaths`，
  **不带 agent 名**；guard 的 cwd 恒为 `<project>/.agents`。所以"按 agent 名分流"和"按 capsule
  放标记文件"两条路都不通，只剩两个 project。
- 端到端：driver 以 `native_search=True` 跑通，真实检索并返回正确答案，
  `search_events` 记到一行 `item.completed`/`web_search`。

**已知缺口：agy 不报来源 URL。** 整条事件流里没有任何 http 链接（连终止 `result` 事件也没有），
只有 `tool_info.parameters.query`。所以 `urls` 一律为空，不伪造。后果是 **agy 的 native 证据比
Codex/Claude 薄一档**：那两家有逐 call 的 URL provenance，agy 只能证明"查了什么"，不能证明
"看了哪些页"。拿 native 和 local 做检索质量对照时这条是硬约束——local 有完整 URL ledger
（且 fetch 被硬门限制在本 task 已搜过的 URL 上），两臂在证据可审计性上并不对称。

**另一个真机教训**：agy 的 Gemini 系列**必须**带 `--effort`（`low|medium|high`），不带直接
"invalid model selection" 硬失败；Claude 系列则相反，带了就失败（§2 的 `thinking = false`）。
两条都由 `_agy_model_takes_effort` 单点处理。
