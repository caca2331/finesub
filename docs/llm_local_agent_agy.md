# agy（Antigravity CLI）：唯一具备多模态能力的 agent

> 本文从 [`llm_local_agent.md`](llm_local_agent.md) 拆出（2026-08-16，原文 2,198 行），
> **章节号已按本文重编、从 1 起**——每份文档独立编号，才能在中间插入新节。跨文档引用
> 一律写成 `` `文件名` §N `` 的形式，由 `test_doc_links.py` 守着：指不到的章节号会红。

来源：`../common/tmp/audio_video_multimodal_report.md`（初始媒体报告）与
`tmp/agy-3_7-flash-probe.md`（3.7 Flash headless/A1 复测）。Antigravity CLI 底层是 Gemini
多模态，通过 `view_file` 工具读取媒体文件。

## 1 定位与 catalog 行

- 新 provider tier `LOCAL_AGY`，backend 仍是 `local_agent`，由 tier 选 driver（[§2 音频：MIME 被工具层拦住，必须容器化](#2-音频mime-被工具层拦住必须容器化) 已有的机制）。
- Catalog 已加入 fact `local-agy-gemini-3_8-flash` 并路由 target `local-agy-gemini-3_8-flash`：CLI base id
  `gemini-3.8-flash`，`max_input_tokens=1048576`、`context_window=1048576`（单池，包络 983040）、
  `max_output_tokens=65536`、能力档「中」、
  thinking 映射 `medium,medium,low`、`token_scale=1.0`。`--effort low|medium|high` 与 base id 可
  直接组合；CLI 的 model list 也同时列出 `gemini-3.8-flash-low|medium|high`。出厂模型组用的是
  与它同名的 target（每个后面接 Argon，见下）；2026-11-09 前它有 media / native 一对 target。3.7 原有一份逐字段相同的 fact 与一对 target，2026-10-12 3.7 Flash
  从 agy 的模型表里消失（服务端，CLI 没升级）后删掉，所有 agy 组改回 3.8（owner 同日决定，thinking 映射照旧）。
  3.6 Flash（只给验收跑当「弱一点的 capable」，不在出厂组里）到 2026-10-23 也从模型表里消失（agy 1.4.7），fact 与一对 target 随之删掉
  （owner 2026-10-24：agy 的 Flash 只留 3.8）。
- **Gemini 4 Argon**（2026-10-18 接线）：fact `local-agy-gemini-4-argon`，target 同名，窗口、输出上限与媒体能力照 3.8 Flash 填（没单独核实），能力档「高」（估计）。
  2026-10-19 在三个日语直播窗上与 3.8 Flash 各跑一遍带音频的三个 pass（同为 medium 思考）：owner 判质量略优、token 效率高出一些（未缓存输入约一半、输出加思考约三分之一，单遍），字幕合并得比 Flash 保守；单价比 Flash 贵。
  thinking 映射与 3.8 Flash 相同，`medium,medium,low`（owner 2026-10-19：两者同档比较）。Pro 订阅上它有单独一份只按周计的额度（`/usage` 里的「Gemini 4 Argon」组），所以额度池是自己的
  `AGY_ARGON`。每个出厂 agy 组里，3.8 Flash 的每个 target 后面紧跟 Argon 对应的那个，当它的替补（owner 2026-10-18）。
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
- **起 agy 一律带 `AGY_CLI_DISABLE_AUTO_UPDATE=true`**（`local_agent.py` 的 `AGY_QUIET_ENV`，2026-09-29）：agy 至多
  每 15 分钟查一次更新，由一个脱离控制台的助手进程跑 `agy --version`，它自己开一个**看得见的**控制台窗口，
  `no_window()` 管不到（父进程有没有控制台都一样）。只有 `true` 管用，`1` 照样检查（实测 agy 1.2.13）。
  owner 手动用 agy 时照常更新。

## 2 音频：MIME 被工具层拦住，必须容器化

3.7 Flash 实测中，`view_file` 在 payload 到达模型**之前**就按 MIME 拒绝裸音频：
`.flac` → `unsupported mime type application/octet-stream`，`.wav` → `audio/wave`。

绕法（报告已验证，转写 6 句全对）：把音频封进带黑画面视频轨的 MP4。

```bash
ffmpeg -y -f lavfi -i color=c=black:s=16x16:r=0.05 -i input.flac \
       -c:v libx264 -tune stillimage -c:a aac -shortest output.mp4
```

成本：音频 **≈32 tok/s**；视频轨按它的长度计（不按帧数），所以 driver 只给每段 1 秒的视频轨（见下）。
**估算继续按 32（owner 定，2026-08-14）**：它是上限侧的保守值，实际会因抖动与压缩低一些。
一次实测供参考——986.589s 的生产剪辑，Gemini REST 返回 `prompt_audio_tokens=25,476`，
即 25.82 tok/s（见 `llm_local_agent_experiments.md` [§3.3 生产尺寸纠错窗实测](llm_local_agent_experiments.md#33-生产尺寸纠错窗实测门槛确实被越过2026-08-14)）。**不要据此把估算改成 25.8**：上限用于包络才安全。

以前（owner 2026-08-14）的做法是把伪造视频压成恰好一帧、铺满整段，按「一帧常量」估开销；2026-10 实测这个估法不对，
截断也出在这里，owner 2026-10-30 定了改成下面的切段。

**超过 5 分钟的附件只听到前一部分**（2026-10-24 查出、10-29 实测）：一帧画面铺满整段的 mp4，模型只听到小于全长的
最后一个 5 分钟整数倍（11 分钟的只听到前 10 分钟），帧数不影响，视频轨的长度才影响；花费上，铺满的那一帧让前 5 分钟按视频计。
所以 driver 现在把音频**切成不超过 295 秒的几段**（`split_audio_for_agy`），每段封成视频轨只有 1 秒黑帧的 mp4，按音频计价
（约每秒 25 token），一个音频变成几个附件，后面跟一句说明（模板 `agy_audio_pieces_v1.md`）：每个附件是整段的哪一截，
所有时间按整段算。11 分钟的片段上验过（试验版与生产代码各一次，生产代码切在 208、426 秒，两刀都在行间）：开头、结尾、中间 12 处都听到，问时刻说了什么，位置都对；20 分钟的片段问 38 个时刻，36 个位置对。
不选裸音频（agy 1.1.18 起也收）：它报的时间偏晚、随时长变大（11 分钟的晚 20–60 秒），而纠错与说话人裁判的 prompt 都按秒指位置。

切点：每个等分点前后 30 秒内，挑不在任何一行字幕里、又最安静的 0.2 秒；那一段全在行里（长段读信、歌）才退回最安静的 0.2 秒。
行的位置由调用方随附件给（`UploadedFileRef.speech_spans`，剪辑内的秒数）：纠错窗按窗里与前后的行，说话人裁判按声音样本与录音里的行。
片段时长按解码出的采样数算，不信 ffprobe 对裸 ADTS 报的时长。真 ffmpeg 回归覆盖切成三段：首尾相接、每段视频轨 1 秒。

`containerize_audio_for_agy`（覆盖完整音频时长的单帧 H.264 + AAC，帧长同样按解码出的采样数算）只留给 5 分钟以内的探针与工具脚本。

## 3 视频：分辨率不可调，现有 token 公式失真

**超过 5 分钟的视频也切段**（owner 2026-11-02，与音频一致）：截断按视频轨的长度算，所以长视频同样只给前一部分。driver 用同样的切点（`agy_media_cuts`：在音轨上找，优先落在两行字幕之间）把视频切成不超过 295 秒的几段，每段照常转成 0.25 fps（`split_video_for_agy`），后面跟一句说明（模板 `agy_video_pieces_v1.md`）：每个附件是整段的哪一截，时间按整段算。2026-11-03 在 agy 3.8 Flash 上验过：音频验收那段 11 分钟的录音配上印着时钟的画面，预采样成 0.25 fps 交给 driver，切在 208、426 秒（与音频相同），12 个时间点的话都听到、位置对（与音频验收相当），画面上离这些时间点最近一帧的时钟读数差 0–1 秒；一次调用未缓存输入约 6.9 万。不到 5 分钟、或没有音轨的视频照旧一个附件（没有音轨的超过 5 分钟时，后面那部分仍可能收不到：切点要从声音里找）。agy 哪天整段收下长附件了，音频与视频的切段一起撤掉（owner 2026-11-02），见 [`llm_followups.md`](llm_followups.md) 的「agy 整段收下长附件之后撤掉切段」。

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
「约 12k token」（`llm_local_agent_experiments.md` [§3.2 缓存为什么没接上](llm_local_agent_experiments.md#32-缓存为什么没接上前缀太小不是路不通2026-08-14) 的顺带观察），2026-08-22 按 transcript
核实为 **≈46k 字节/次且可续读**：回复自带「Content truncated: showing bytes 0-46080 of N … call
this tool again with ContentOffset=46080」，见 [§5 MCP 工具结果的内联上限](#5-mcp-工具结果的内联上限约-4k-字节超过即外置成文件无预览2026-08-22)。

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
同样有。它不阻止缓存（`llm_local_agent_experiments.md` [§3.2 缓存为什么没接上](llm_local_agent_experiments.md#32-缓存为什么没接上前缀太小不是路不通2026-08-14) 里命中的两次也有 checkpoint），但它会把媒体挪进
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

**音频读入后的下一次请求整段未命中；跨轮次续接（`--conversation`）不卸载音频**（2026-10-16，3.8 Flash，capsule 路径、`assignment` 作用域，合成的 56 秒语音，两个会话）：同一次调用里，模型用 `view_file` 读完音频之后的那次请求约 3 万全部未缓存、缓存读 0——和上面视频那次不同，连音频之前的前缀都没读到（音频由 driver 封成单帧 mp4）。续接后没重读文件却答对了第一轮没问过的内容，媒体 token 仍在上下文里，文件被挪进 `brain/<id>/.tempmediaStorage`；续接那轮一个会话第一次请求就读回 28,241（含音频），另一个前两次只读到 4,789、下一轮才读回 28,887。不带媒体的续接读回整段（21,934）。续接本身不慢：每轮 16–19 秒，生产记录里连续续接每次 40–48 秒（含写完整答案）。

**打断缓存的是会话里第一份媒体，不是媒体被挪到请求开头**（2026-10-17，3.8 Flash，同上路径，探针在被忽略的 `data/session-ab-2026-10/probes-live/cache-reuse-2026-10-16/agy_media_position_probe.py`）：

- 同一会话先后读两份音频（各约 5.5k token，读第一份之前的那次请求约 1.86 万、已过写入门槛），两次复跑一致：读完第一份那次整段未命中（24,771 / 0、24,284 / 0，未缓存 / 缓存读）；读完第二份那次几乎读回前一次的全部（6,073 / 24,300、5,959 / 23,813）。若媒体被挪到请求最前，不论按读入顺序排还是最新的在前，第二份进来时前缀都会变，读不回这么多。
- 三个新会话依次读字节相同的同一份音频：读完音频那次都读 0，与读另一份音频的对照会话一样。agy 的系统提示在第 462 字节（约 120 token）就出现每会话不同的路径，音频若在最开头，后面的会话应读到约 5k。这条没有正对照（造不出跨会话共有长前缀的请求，跨会话能不能读到缓存本身没证实），只算旁证。
- agy 会话库里存的最后一次请求结构（`gen_metadata`）中，媒体是 `view_file` 工具结果里的内联 `video/mp4` 部件，按读入顺序排；上面续接时那次 4,789 约等于系统提示加工具定义的长度，说明那次请求仍以系统提示开头。
- **开局先读一个占位媒体就躲开了**（探针 `ph` 臂，两次一致）：提示里让模型先打开一段 1 秒静音的单帧 mp4、再读任务文件、再读真音频。读完占位那次 6,099 / 0，读任务文件那次 19,225 / 0（写下缓存），读完真音频那次 6,022 / **18,641**（另一次 18,673）；不读占位时同一位置是 0。
- **更省的是一步同时打开**（各两次一致；同样约 1.9 万的任务文件、一段 56 秒音频，每次调用的未缓存合计）：占位和任务文件同一步打开（两个并行的 `view_file`），读完真音频那次照样读回 18,595 / 18,597，请求少一次，合计约 3.06 万；更好的是提示里直接给出真音频的路径、和任务文件同一步打开，不要占位：开头那次之后就是作答那次（23,912 / 0），合计约 2.93 万、只有两次请求。对照：先读任务文件、再读音频约 4.88 万（三次请求，缓存读 0），先读占位再依次读约 3.67 万（四次请求）。模型照提示在同一步里并行打开了两个文件（会话步骤核过）。
- **开场消息里附不上媒体**：`--input-format stream-json` 的内容块只收 `text`（其它类型报 `only "text"`），print 模式下 prompt 里的 `@文件` 不展开成附件（模型只看到这段文字）。所以媒体最早在第一次工具调用时进来；第一次请求约 5.4k、没到写入门槛，本来就没有缓存可丢，第一步就读媒体在缓存上等于开局就带媒体。
- **直接调 Gemini API 没有这个现象**（gemini-3.5-flash-lite，同样的 mp4 内联、对话每步全量重发，两次一致）：带进第一份音频的请求照常读到前面的文本缓存，第二份也一样；第二次跑时带音频那步读到的比上一步整段还多（13,645），读的是第一次跑写下的含音频前缀，说明 API 侧媒体就在原位置、也能进缓存。所以这是 agy（或它背后那条服务路径）的行为，不是 Gemini 缓存本身的规则。
- 推断，机制没验证：agy 里会话第一次带媒体的请求换了一份缓存（例如带媒体的请求走另一组后端，或缓存键里含模态），之后带媒体的请求之间照常命中。对生产的含义：带音频的窗口读音频之前写下的缓存（任务文件那段，约 1.9 万）作废一次，之后照常；让模型第一步就把任务文件和音频一起打开能省掉这一次、还少一次请求：已接进 capsule 路径（2026-10-17 起默认）：新会话、带媒体的调用在 argv 提示里那句第一步（模板 `agy_first_step_v1.md`）点名改写后的媒体 `local_path`，要求第一步用并行的 `view_file` 同时打开全部任务份与这些媒体；继承会话的调用（续接、继承会话的修复轮）不点名媒体，只打开任务份。execution attempt 的 `input_preparation.first_step_media` 记下点名的媒体（capsule 内的相对路径）。

**跨 session 不继承缓存**（2026-10-16 复测：3.8 Flash，`per-window` 工具会话，开场带上任务、
协议写在 agent 文档里）：同一个 2 分钟纠错窗经 `run_session_ab` 背靠背跑 4 次，FineSub 交给 agy
的输入只差开场里的载荷路径。4 次调用的首请求都是约 19.3k 未缓存、**缓存读 0**；同一次调用里的
第二次请求读回约 19.3k，所以写入发生过、前缀也远过下面说的门槛，后三次调用在前一次结束几秒后
发出，一次都没读到。**缓存只在会话内继承**：同一调用里的后续请求与 `session_scope=assignment`
的续会话拿得到，跨窗、跨任务的新会话拿不到。FineSub 看不到 agy 拼出的完整 prompt，分不清是后端
不跨会话共享、还是 agy 在前缀早段放了逐会话的内容，两种情形 FineSub 都改变不了。后来逐项排除了变量（2026-10-16）：agy 存在会话库 `gen_metadata` 里的请求显示，agent 文档排在最前、各次逐字节相同，紧跟其后的 `<user_information>` 里是每次不同的 capsule 路径与每会话不同的 Conversation ID；把 agent 文档加长到约 2.1 万 token（远过写入门槛）、两次逐字节相同，第二次新会话的首请求仍读 0。所以不是前缀太短，也不是 cwd：新会话就是读不到别的会话写下的缓存。
（更早一次探针 `tmp/agy_pseudo_turns_probe.py` 用两个全新会话跑同一份输入，结论相同：新会话
gen#1（17,020）没读到上一轮写下的块，到 gen#3 才读回 16,328。）

**顺带修正 `llm_local_agent_experiments.md` [§3.2 缓存为什么没接上](llm_local_agent_experiments.md#32-缓存为什么没接上前缀太小不是路不通2026-08-14) 的"隔一次请求才可读"**：真正的规律是**写入有门槛**。三次实验对照——前缀
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
- Declarative agent 的 `tools: [view_file]` 不改变 `init.tools` 的宣告（文档生效与否都报全集），但文档生效时模型请求里
  的工具确实变少：文档列出的那些，加上 agy 自带的 `manage_task`（capsule 2 个、native 4 个，2026-10-06 按会话库核）。它仍不能单独当安全边界：文档写错会被 agy
  静默整份忽略、退回默认 agent（工具 project 的 `finesub-tool` 08-21 到 10-06 就是这样；现在 project 核对会报出来，见
  [`llm_agent_tool_protocol.md`「6. 各 driver 接线」](llm_agent_tool_protocol.md#6-各-driver-接线)的 agy 一行）。
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

guard 按字节读 hook 载荷（`json.loads(sys.stdin.buffer.read())`），不按文本读：agy 写的是 UTF-8，而文本 stdin
按系统代码页解码（agy 的环境走白名单，没有 `PYTHONUTF8`）。中文 Windows 上是 cp936，答案里中文紧挨着转义引号
`\"` 时，末字节会和反斜杠拼成一个 GBK 字符，JSON 解析失败；崩掉的 hook 被 agy 当成拒绝，`submit` 整份被拒，模型只能
整份重写。这是 Nonoka X 在中文系统上查出来的（它的补丁 0008）；本机代码页是 UTF-8，复现不了，测试用 `PYTHONIOENCODING=gbk` 模拟。

这与"知识库都是公开信息"无关——风险对象是用户磁盘上的其他文件，不是知识库内容。

**agy 1.2.13 起 `view_file` 还要一条 `read_file` grant**（2026-09-29，在 agent-dispatch 上实测，gemini-3.7-flash
与 claude opus；2026-09-30 finesub 真机复核：运行根放在临时目录外，capsule 调用照常通过）：hook 放行之后，agy 自己的权限检查还会拒掉 workspace（project 与 cwd）
以外的读取，系统临时目录除外，拒法与 [§6.2](#62-read_url_content-还要一条-permission-grant2026-08-30-实测补上)
一样：整个 turn 提前结束，stderr 同时含 `permission` 与 `headless`。从 checkout 运行时 capsule 域在临时目录下，
所以碰不到；managed 域（用户数据目录）和显式指定的根不在，tool-session 读的 assignment 根又在 project 与 cwd
之外。所以每个 agy project 都授予 `read_file(*)`（`AGY_READ_PERMISSION_RULES`）；读取边界仍是上面的 guard。

⚠ **custom agent 文档不加 `excludeDefaultComponents`**（owner 2026-09-30）：这个键没公开，agy 遇到不认的 agent 文档内容会
静默整份失效；近亲键 `inheritCustomizations` 会关掉 hook，而 hook 就是上面这条读边界，`/hooks` 核对看不出它被关。收益只是
少些用户全局规则的干扰。官方公开这个键、并承诺不影响 hooks 时可以重新评估。

## 5 MCP 工具结果的内联上限：约 4k 字节，超过即外置成文件、无预览（2026-08-22）

agy 对**大的 MCP 工具回复**不截断也不预览，而是整个写到
`~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/steps/<n>/output.txt`，只给模型一句
「The output was large and was saved to: file:///…」。实测（1.1.18，Gemini 3.7 Flash，
`tmp/agy_result_size_probe.py` 等三个哨兵探针 + brain transcript 对账）：回复 ≤3,639 **字节**内联
可见，≥4,248 字节一律只剩文件路径——单位是回复的 UTF-8 大小，与字符数、行数无关（中文 3000 字
一页 9k 字节被外置，ASCII 单行 6000 字符反而可见）。模型还会用 `read_context(ref="invalid")` 套出
合法 ref、再用 `offset` 只取小页，所以哨兵实验必须对照 transcript 而不是只看答案。配合 [§4 准入硬门：view_file 的读取边界](#4-准入硬门view_file-的读取边界) 的 hook（只放行 `call_mcp_tool`），模型既读不了那个
文件，也就**完全看不见** protocol 与 payload——2026-08-22 第一次 4 窗 canary 正是这样失败的：模型
`submit` 了 `test`、用猜的 ref 反复 `read_context`，两条会话烧掉 19 万 uncached + 81 万 cached
input 才耗尽预算。

1.4.6（Gemini 3.8 Flash，桩 server，2026-10-22）重测：4,023 字节内联、4,546 字节起外置，日文、英文、数字密集三种内容在同一个字节区间翻转，
与 1.1.18 的读数合起来阈值在 4,023 与 4,248 之间，像是 4 KiB；规则在 agy 的工具层，与模型的分词无关。2026-10-15 至 10-20 的生产里，
15 次 `kb_read` 有 6 次回复 7–12 KB 被换成文件路径，模型一次也没读到。

对策是**外置成份**（[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [通道上限](llm_agent_tool_protocol.md#通道上限超限的回复与送达核对)）：
agy 的通道是成功与错误回复各 3,600 字节（实测内联过的最大值往下取整，离外置的最小值还有约 650 字节）、超过时 `overflow = files`。
server 把放不下的回复写成 `<assignment>/control/spill/<请求 id>/part-NN.md`，模型拿到的桩点名全部份、叫它一步里用并行的 `view_file` 整份读完；
任务正文（工具会话开场的载荷与 `kb_index`、capsule 调用的消息）走同一个写份的函数，开场或 argv 点名全部份。

- 每份 ≤ 40,000 字节 / 700 行，留在 `view_file` 单次的 46,081 字节 / 800 行之下（1.4.x 实测），一次必然整份读到，
  不用 `ContentOffset`、不切坏字；内容按行写成文本，不是一行 JSON。一步里并行三个 45 KB 的 `view_file`，三份都整份读到（2026-10-22）。
- driver 每次 invocation 把本次调用的 assignment root 写进槽位 project 的 `.agents/view_roots.json`，tool guard 放行 **realpath 落在这些根之下的现有文件**的
  `view_file`，其余 `view_file` 与一切别的原生工具照旧 deny；project 记录不需要额外 grant（媒体 project 同款，hook 的 allow 即足够）。
- 份的清单放不进 argv（整条命令行按 UTF-16 超过 30,000）或桩时，改写成「目录 + `part-01.md` to `part-NN.md`」，模型照样一步读完。
- 不选的路：继续按 2,800 字节分页（中文一页约 800 字，一个大条目十几轮）；让 hook 放行 agy 自己外置的那个文件（在 agy 的私有目录里，hook 要知道会话 id，
  超过 46 KB 还要续读，放宽读边界会露出别的会话的文件）；正文走 stdin（`--input-format stream-json`，实测 150 KB 整条送到，但多维护一条路，带媒体的调用又用不上，owner 2026-10-23 定只留一条路）。

Claude Code 与 Codex 的等价旋钮（owner 调查 2026-08-22）：Claude 是环境变量 `MAX_MCP_OUTPUT_TOKENS`
（默认 25,000；文件读取另有 `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS`，工具会话不读文件、不设），
driver 只在工具会话的**本次 invocation** 进程环境里设（`ClaudeCodeDriverConfig.mcp_output_tokens` /
`file_read_output_tokens`）；Codex 是 `tool_output_token_limit`（没有独立的文件读取字段），driver 以
`-c` 覆盖随 MCP 声明一起传（`CodexDriverConfig.tool_output_token_limit`）。**三个都是 owner 统一定的
200k token（2026-08-22）**——模型上下文才是真天花板，放宽没有代价。两家都不写项目级或用户级
配置文件（`.claude/settings.json`、`.codex/config.toml`），与协议总原则一致。

⚠ **Codex 上 `tool_output_token_limit` 只管直调的工具**（2026-09-30 的 rollout 与 2026-10-16 探针，0.157.1 / 0.160.0、GPT-6 Luna）：
Codex 默认经 code mode 的 JS `exec` 调 MCP 工具，`exec` 的输出截在 **10,000 token**，砍中间、两头留下，并写一行「Warning: truncated output
(original token count: N)」。这个 token 数是脚本打印出来的字节数除以 4、向上取整，所以真正的上限是打印出来的 4 万字节；纠错窗的 `next_task`
打印出来约 5.2 万字节，每窗都被截掉一段。所以 FineSub 的工具设成直调（`features.code_mode.direct_only_tool_namespaces`，见
[`llm_agent_tool_protocol.md`](llm_agent_tool_protocol.md) [通道上限](llm_agent_tool_protocol.md#通道上限超限的回复与送达核对)）：直调时 76 KB 的回复整份送到（探针 2026-10-22），
截断回到 `tool_output_token_limit` × 4 字节。`--disable code_mode_host` 不是出路：关掉之后模型一个工具都没有。

另一条顺带观察：agy 在会话**第一步**就把 bootstrap 压成 CHECKPOINT 摘要（与长度无关，[§3 视频](#3-视频分辨率不可调现有-token-公式失真) 已记），
所以 bootstrap 里的规则必须短、祈使、可被摘要保留。

## 6 原生搜索：第二个 project（2026-08-15 打通）

工具真名是 **`search_web`**（取数）与 **`read_url_content`**（取页），从真机 `system.init` 的工具
表里读出来的，不是猜的。`--retrieval native` 现在能落到 `local-agy-gemini-3_8-flash`。

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

上面那一对只服务 **capsule** 路。**tool-session** 路（[§4 准入硬门：view_file 的读取边界](#4-准入硬门view_file-的读取边界) 的第三种 project
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
把 bootstrap 压成 CHECKPOINT 摘要（[§3 视频](#3-视频分辨率不可调现有-token-公式失真)），另起的段落可能被摘掉。

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

**这一次只修了 grant，没修分类（2026-09-04 补上）**：上面那个「整个 turn 结束、不产 assistant
message」落到 driver 手里，命中的是通用的空回答分支，报出来的是一句
`did not contain a final assistant message`——它既不读 stderr，也不带 capsule 指针，而且被判为
**transient**。transient 意味着 router 丢掉这个 target 接着往下走，于是用户最终看到的是**链尾**
那一环的话（典型是「某个 provider 没有 API key」），与真因毫无关系；真相只在 agy 自己的
transcript 里。

现在 `AgyLocalAgentDriver._empty_answer_error` 先认这个形状：stderr 同时含 `permission` 与
`headless` 就判 `LocalAgentUnavailableError`（与「未登录」同一族——权限缺失不会在下一个 target
上自愈，它该可归因地让这个后端出局），并把 stderr 里引号内的工具名带进消息。判据只看那两个
子串，**措辞变了只丢工具名、不丢结论**；其余空回答仍是 transient，但补上了 capsule 指针。
同一行 stderr（`required the "mcp" permission`）也会跟在 agy 拒掉模型的畸形 MCP 调用之后出现——ToolName
是 server 根本没有的名字，自然没有 grant。那不是缺权限：工具会话里事件流认出这种被拒的调用时，空 turn 判
`LocalAgentRefusedCallError`、按 premature stop 再问一次，不走上面的分支（判据在
[`llm_agent_tool_protocol.md`「各 driver 接线」](llm_agent_tool_protocol.md#6-各-driver-接线)的 agy 一行）。
⚠ 这个签名在 184 份生产 agy exchange 里**零命中**，所以「跑一遍生产语料没变化」证明不了任何
事——回归靠的是构造 capsule 的单测。

**教训**：判「模型有没有联网」不能只看本仓库产物。`search_events` 为空既可能是没搜，也可能
是搜了被拒。真相源是 agy 自己的 transcript：
`~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/logs/transcript_full.jsonl`，
conversation handle 即产物里的 `execution_attempts[].session_id`；每个工具调用都跟一个偶数号
结果步，缺结果步即该调用没有完成。

**证实过的事**（真机，2026-08-15）：
- agent document 不是安全边界——`--agent` 指向一个只声明 `view_file` 的文档时，`system.init`
  依然列出全部 56 个工具（init 的计数不是模型实际看到的工具集，文档生效时请求里的工具会变少）。这正是 [§4 准入硬门：view_file 的读取边界](#4-准入硬门view_file-的读取边界) 说"第 2 道才是真正的守卫"的原因，也是为什么这里必须动
  hook 而不是只改文档。
- hook payload 带 `toolCall{name,args}`、`modelName`、`conversationId`、`workspacePaths`，
  **不带 agent 名**；guard 的 cwd 恒为 `<project>/.agents`。所以"按 agent 名分流"和"按 capsule
  放标记文件"两条路都不通，只剩两个 project。
- 端到端：driver 以 `native_search=True` 跑通，真实检索并返回正确答案，
  `search_events` 记到一行 `item.completed`/`web_search`。

**已知缺口：agy 的搜索不报来源 URL**（2026-10-01 在 agy 1.2.14 上复查，仍然如此）。`search_web` 那一步的事件只有
`tool_info.parameters.query`，搜到了哪些结果不在流里，所以这类行的 `urls` 为空，不伪造；模型用 `read_url_content`
真正打开的页面，URL 就在调用参数里，会记进 `urls`。后果是 **agy 的 native 证据比 Codex/Claude 薄一档**：那两家
每次搜索都带结果 URL，agy 只能证明「查了什么、打开了哪几页」，不能证明「搜到了什么」。拿 native 和 local 做检索质量对照时这条是硬约束——local 有完整 URL ledger
（且 fetch 被硬门限制在本 task 已搜过的 URL 上），两臂在证据可审计性上并不对称。

**另一个真机教训**：agy 的每个模型都**必须**带 `--effort`（`low|medium|high`），不带直接
"invalid model selection" 硬失败——Gemini 一直如此，Claude 从 5.5 起也是（2026-10-03，agy 1.2.16；1.1.x 时的
`claude-opus-4-6-thinking` 相反，带了就失败）。agy 没有 `xhigh`，`[llm].local_agent_reasoning_effort` 填它时按 `high` 发。
都由 `_agy_effort` 单点处理。

## 7 答案通道本身不可靠：约四分之一的回答是畸形的（2026-09-02）

与媒体、`--effort`、工作目录都无关：26 次调用里 6 次回答畸形——4 次把思考过程（thought）当成回答
漏出来、1 次空回答、1 次退化解码。判一次回答好坏，先看它开头是不是思考过程。

- **空回答**有防线：它按传输失败处理，换到链上的下一个目标（不在 agy 上原地重试）。
- **thought 泄漏**（回答以 `thought\n` 开头，后接英文推理）：HTTP 成功、`status=SUCCESS`、回答非空。泄露的推理之后通常跟着
  完整的答案，所以纠错窗、说话人裁判与校对都先**剥掉推理、取最后一个完整的答案块**再照常校验（`output_protocol.strip_leaked_thought`，
  owner 2026-10-17）：从最后一处、之后每个必需块都恰好是一个顶层块的答案标签起算——先只看独占一行的标签，都不成才看夹在行里的
  （推理里点名的标签、答案自己行里写到的标签都在行中间，块的开头独占一行）——紧挨在它前面的兄弟块（中间只有空白，如
  `<reasoning>`）一并带上。交换记录留原样的回答，窗口记录记剥掉的字数（`thought_stripped`）。没有完整答案的（只有推理、只有
  不完整的草稿）照原样校验、拒收、进修复轮，校验错误第一条是「上一次交回的是推理过程，只交答案」（`output_protocol.LEAKED_THOUGHT_ERROR`）；
  `api` 档的 agy 没有续得上的会话，不接修复上下文，那里只能盲重试，这句提示到不了模型。
  driver 只在调用记录上打 `answer_shape: thought_leak`，**不**改失败分类：判成 transient 会丢掉这条会话、换到链上下一个
  目标且记进额度账，比续会话的修复轮更贵。工具会话不看最终文本（答案走 `submit`；运行时的校验器按剥过的答案判，交回的是原样，由 harness 再剥、记账）。

另见 [§8 轮内重试之后的答案](#8-轮内重试之后的答案撞输出上限断流空回复2026-10-13)：agy 在一轮里自己重试（空回复、撞输出上限、断流）之后模型作答的那些轮，答案从 agy 的 transcript 取回，
不再当作失败。

## 8 轮内重试之后的答案：撞输出上限、断流、空回复（2026-10-13）

**现象**：一段回复写到单次输出上限（65,536 token，**含思考**）、流断了、或回复为空 / 函数调用畸形时，agy 在同一次
`agy --print` 里自己插一步 `ERROR_MESSAGE`（如「Your previous response was cut off because it exceeded the output
token limit. Please continue from where you left off, keeping your response shorter. Retries remaining: 3」）让模型
接着来，不需要 `--conversation`。模型通常会把答案写完，但 `result` 事件报的是最后那次错误（`status=ERROR`，`error`
就是插进去的那句话）；只有空回复重试的轮次以 `SUCCESS` 结束，`response` 却只是重试之后那一段。完整答案在 agy 自己的
transcript 里：`~/.gemini/antigravity-cli/brain/<conversation_id>/.system_generated/logs/transcript_full.jsonl`，
每步一行，带 `step_index`（与 stream-json 的 `step_update` 同一套编号，续会话时也沿用整个会话的编号）与 `type`
（`USER_INPUT`、`PLANNER_RESPONSE` 带或不带 `tool_calls`、`GENERIC` 工具结果、`ERROR_MESSAGE` 带 `error` 字段）。

**续写长什么样**（2026-10-08–13，agy 1.2–1.3，Gemini 3.7 / 3.8 Flash high，row-truth 实验与生产纠错窗共 41 个这样的轮次）：
撞上限时思考已吃掉六万上下，第一段可见答案往往只写出几千字就断了。之后要么**原地接着写**（从断处的半个字段接下去，
逐字接上；见过一次在行尾少了分隔符、两行粘成一行，被输出校验拒收）；要么第一段根本不是答案（草稿、漏出的推理），
模型**从答案的第一个块起整份重写**。接缝前后的判断质量没有看出差别（5 个窗与未截断的运行对照）。续写那一段要把第一段的
全部输出连同思考当输入再发一遍，约 0.3–1.5 分钟；整轮重来是 5–7 分钟。

**生产怎么收**（`AgyLocalAgentDriver._recover_answer`、`finesub/llm/agent/agy_transcript.py`）：只在 capsule 路径
（最终文本即答案；带媒体的调用都走它）、退出码 0 时看。本轮（`step_index` 不小于本次 stream 里最小的那个）有
`ERROR_MESSAGE`、最后一个之后模型作答了，且 `result` 是 `SUCCESS` 或它的 `error` 就是那一步的 `error`（排除「答完
之后又终态失败」），就用 transcript 里的答案：取最后一次工具调用之后的回复，丢掉以 `thought\n` 开头的漏出推理；
锚定第一段以开标签开头的回复的那个标签，从最后一段以它开头的回复起顺序拼到末尾（整份重写只留最后一份，原地续写全拼上，
从块边界接续的也拼上）。`result` 行的 `error` 移到 `agy_error`，于是不再判成 provider 故障；答案确有变化时（`SUCCESS` 的空回复重试里
`response` 本就是全文，不记），调用记录上记 `transcript_recovery`（段数、`dropped` 即是否丢掉了前面的段、agy 的错误），exchange 的
执行尝试表下多一行说明。**接缝不另做检测**
（owner 2026-10-13）：拼坏了由输出契约拒收、走修复轮。取不回（没有 transcript、对不上、超过 `max_result_bytes`）时
照旧是 transient，换下一个目标。工具会话不经过这里：答案走 `submit`。

**为什么读 transcript、不拼 stream-json 的 `text_delta`**：后者更简单，天然只含本轮，也不依赖 agy 的私有目录；但被截断的
那一段在流里是否完整、原地续写接缝处结尾的换行怎么还原，都没有真实撞限的流可核（owner 不做真跑验证），而 transcript 的
形态有几十个真实轮次为证。存下一条真实撞限的 stream-json 之后可以重新考虑。

撞上限那类 `ERROR` 的退出码是 0：2026-09-03 一次生产运行的 4 次撞限尝试都是 0（agy 1.1.25）；轮内重试类的 safety
filter 也是 0，终态错误如 429 是 3。哪天 agy 改成非 0，这条路径静默不生效，行为同改动之前。
