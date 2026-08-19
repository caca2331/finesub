# Changelog

## [Unreleased]

## [0.4.1] - 2026-08-18

### ⚠️ 从 0.4.0 升级请用 Setup 覆盖安装，不要用应用内更新

**0.4.0 的应用内更新走不通**，两条路都不行，而且这两处都在 0.4.0 自己那半，
0.4.1 无论怎么改都够不着——应用内更新用的是**你机器上已装的**那套更新程序：

- 增量更新装完后应用起不来（0.4.0 冻结的组件读不懂本版新增的 ffmpeg 下载配置
  形状）。会连续两次启动失败后**自动回滚**回 0.4.0，数据不受影响，但更新不生效。
- 完整更新根本没应用上（见下一条的根因），安装保持 0.4.0 可用。

**正确做法**：下载 `FineSub-Desktop-0.4.1-Setup.exe` 直接覆盖安装。已实测：
设置、API Key、知识库、成品字幕、模型与缓存全部原样保留，装完即认作 0.4.1。

0.4.1 之后的版本恢复正常，应用内更新与增量都可用。

### 修复：应用内更新从来没有等过应用退出

更新程序被拉起后**立刻**开始替换程序文件，而此时应用还在运行，于是它跟自己抢，
十秒后放弃并报「拒绝访问」——用户往往还没来得及关闭应用。它本该等最多一小时，
实际等了 0 秒：判断「应用是否还活着」用的方法在 Windows 上对**父进程**永远回答
不了，而那个回答被当成了「已经退出」。

现在改为等待应用真正退出后再动手。这也解释了为什么此前的完整更新从未成功过。

### 修复：全新安装拿不到 ffmpeg（以及 0.4.0 的 ASR 环境装不上）

两个装机链上的下载地址失效，症状都是首次运行环境安装中途报错：

- **patched CTranslate2 的 wheel 从未发布**。0.4.0 引用的
  `ct2-4.8.1+finesub0.4.0` 这个 tag 不存在，所有前端首次装 ASR 环境都会在这一步 404。
  已补发；**已经装好的 0.4.0 不需要升级或重装**，那个地址是装机时才解析的，重试即可。
- **ffmpeg 钉的构建被上游删了**。BtbN/FFmpeg-Builds 只保留几天的滚动构建，manifest 里钉的
  那个已经不在。

第二个换了做法而不是换个新地址：ffmpeg 现在跟随上游的 `latest`
（FFmpeg 9.0 发布分支），**校验没有取消**——安装时先向 GitHub 询问该文件当前的 sha256，
再据此校验下载内容。代价是不同时间安装的机器会拿到不同的 ffmpeg 构建。

顺带修好一个断点续传的隐患：暂停后隔天再续传，如果上游文件在这期间变了，以前会把新文件的
后半段接到旧文件的前半段上，下载完才校验失败；现在会认出目标变了并重新开始。

### 人声分离的交付分两个模式，管线用的那个改成 16 kHz 单声道

读人声轨的每一个下游（VAD、识别、Qwen 裁判）第一件事都是下混单声道 + 重采样到 16 kHz，
而我们此前交付 44.1 kHz 立体声——四分之三的码率花在了下一步就被丢掉的样本上。现在按输出
后缀分两个模式：

- `.ogg`（管线默认）：**16 kHz 单声道**，Vorbis 质量档从默认调到更保守的一档。同素材实测
  **355 KiB / 28.6 dB**，对比此前 **719 KiB / 23.97 dB**——体积减半，识别真正读到的那段
  频带反而干净了 4.6 dB。
- `.flac`：无损交付，保持模型自己的采样率与声道数，全链零有损。试听、测量、做实验用它。
  只能通过直接调用分离模块得到（`-o x.flac`），管线不会产出。

其它后缀会直接报错，不再默默按后缀猜格式。

既有的 44.1 kHz 立体声 `-vocal.ogg` 照读不误，不必删；重跑才会拿到新形态。

同时修好一处：目录里只有 `-vocal.flac` 时，管线此前会当作「没有人声轨」再跑一遍分离，
尽管后面的阶段本来就能直接读它。现在两种交付任一存在都算这一步已完成。

### 人声分离的产物不再被有损编码两次

分离按块进行，每块先由分离器写成临时文件，再合并成最终的 `-vocal.ogg`。此前临时块用的是
和最终产物一样的格式，于是这段人声被 Vorbis 编了两次。现在块固定用 FLAC，最终产物格式不变，
少掉的那一代损失直接落在 VAD/ASR 读到的音轨上。

实测（60 秒素材）：相对完全无损的参考，人声轨信噪比从 19.9 dB 提升到 23.4 dB，改善集中在
ASR 实际使用的 8 kHz 以下；时间轴逐帧不变，文件大约 4%。

已有的 `-vocal.ogg` 不必删——它仍是一段有效人声轨；重跑才会拿到新的。

### 本地视频输入不再在人声分离前被有损压一道

以前输入本地视频（`.mp4`/`.mkv`/…）时，管线会先用 ffmpeg 抽一份
`<stem>-source.ogg`（Vorbis q5 + 下混单声道 + 重采样 16 kHz），再拿它去做人声分离。
这是全流程里唯一一次由我们自己引入的有损压缩，而人声分离用的是 44.1 kHz 立体声模型，
等于先把信息丢掉再让模型去还原。

现在本地视频和 URL 视频走同一条路：源媒体原样交给各阶段，只有当 soundfile 打不开容器时
才由分离阶段解一份**无损** FLAC（保持采样率与声道数，跑完即删）。`-source.ogg` 这个产物随之
消失，清理逻辑也不再找它——旧任务目录里遗留的同名文件需要自己删。

知识库素材导入（`finesub.workflows.reference_ingest`）的 `media=video` 任务同理：下载的
`<id>.mp4` 现在直接作为 pipeline 输入，不再先抽一份 `<id>.ogg`。

## [0.4.0] - 2026-08-17

### 每次运行都会存一份详细日志

出问题时不用重跑一遍加 `--log-level verbose` 了：每次运行都往
`%LOCALAPPDATA%\FineSub\user-data\logs\run-<时间戳>-<输入名>.log` 写一份完整记录，终端显示的
内容不变。

日志只记 FineSub 自己的判断，不含第三方库的刷屏和进度条，所以一次运行通常只有几 KB。和桌面端
的安装/会话日志共用「保留最新 100 个」这一条规则，不会一直堆积。

**这一版覆盖的是识别侧**：选了哪块显卡、各阶段耗时与显存、跳过了哪个阶段、异常片段在救援阶梯的
哪一级被隔离以及判据。**纠错翻译那一步目前只记开始和结束**——它的细节仍在任务产物目录里
（`task-report.md`、`correction-windows.jsonl`、`exchanges/`），报障时请连那个目录一起提供。

### 破坏性：Python 包改名 `asr_playground` → `finesub`，`llm` 并入其中，命令行只留 `finesub`

**只影响直接用 Python 包或仓库开发版的人**。桌面端与 `finesub` CLI 的用法、参数、
产物、任务记录一律不变。

- **import 路径**：`asr_playground.*` → `finesub.*`，顶层 `llm.*` → `finesub.llm.*`。
  后者顺带消掉一个通用名——装上本项目后 `import llm` 不再是我们的包。
- **仓库根包不再提供 console script**。`asr-pipeline`、`vad-asr`、`vocal-separation`、
  `asr-align`、`asr-stabilize`、`vad-energy`、`to-srt`、`llm-correct-translate`、
  `llm-knowledge-update`、`llm-reference-ingest`、`llm-token-compare`、
  `finesub-agent-clean/ping/task` 全部删除，改用 `python -m finesub.<模块>`
  （写法见 `docs/manual/repo-install.md` 与 `README_DEV.md`）。对外命令只剩一个
  `finesub`，来自 CLI wheel——它会准备托管运行环境，而根包的那些不会，两者同时出现在
  PATH 上只会让人踩坑。
- `python -m finesub.speech.recognition.cli.align` 与 `...cli.vad_asr` **此前没有
  `__main__` 守卫**，`python -m` 会一声不响地成功退出。已补上，并加了守护测试。
- 仓库目录名（`asr-playground`）不变；CHANGELOG 里的历史条目保留旧名。

> 维护者注（0.4.0 发版演练发现的四件事）：
> ① 0.3.x 冻结启动器按 `src/asr_playground/pipeline.py` 校验载荷并定位应用源，
> 0.4.0 起载荷携带 asr_playground 占位文件（`package-bootstrap.ps1` 生成、
> `build_release.py` 把关）。② bridge 冻结在 exe 里、app 增量不换 exe，0.4.0 新增
> `get/save_preferences` 后 0.3.2 增量混血会静默丢设置持久化——增量只留给不动
> 冻结层的版本。③ full updater 搬程序文件改为带退避的纯 rename：旧实现一次
> PermissionError 即中止,copy+rmtree 回退还会把 `_internal` 删残。④ 发货的 0.3.2
> 序列化的 preserved 名单没有 `tasks`/`locations.json`，其应用内 full 更新会把成品
> 字幕挪进日后被清理的 backup——**因此 v0.4.0 不发 update-manifest**：旧版在应用内
> 看不到本次更新，迁移一律走 Setup 覆盖安装；updater 同时把自身名单设为地板与
> request 取并集。下一版本恢复 manifest。

### 没有显卡也能跑完，以及说清支持哪些显卡

**CPU 回退不再跑完就卡死**：没有可用显卡时，语音识别此前会正确算完、把中间结果写到硬盘，
然后**永远停在那里**——进程既不继续也不退出，字幕永远等不到。有位用户的 GTX 1060 因此白花了
九分多钟。原因在我们自己编的 CTranslate2 里：它选的 CPU 计算后端在释放模型时会死锁。换成
另一个后端后，CPU 路径不但能正常结束，还比原来**快了一倍**（同一段音频，识别 28.2s → 13.1s），
内存占用也略低。字幕内容逐字不变。

**显卡支持范围写进 README 了**，用型号说，不用「计算能力」这种查不到的说法：RTX 50/40/30/20
系、GTX 1660、GTX 1650 及更新（显存 ≥4GB）可用；GTX 10 系及更早不支持。

**不支持的显卡会自动回退 CPU，而不是报一个看不懂的错**。老显卡此前的表现是：驱动装得好好的，
torch 也说显卡可用，然后在真正开始计算时抛出 `no kernel image is available for execution on
the device`——人声分离阶段就崩了，根本到不了识别。现在会明确告诉你是哪张卡、它的计算能力多少、
需要什么型号，然后转 CPU 继续跑完。

**`--device` 只接受 `cpu` 和 `cuda`**：`cuda:1` 这样的写法此前会被接受但并不真的选到第二张卡，
现在直接拒绝并提示用 `CUDA_VISIBLE_DEVICES` 指定显卡。多显卡机器请改用该环境变量。

> 维护者注：patched CTranslate2 wheel 的 CPU GEMM 后端由 Ruy 换成裁剪过的静态 oneDNN，
> wheel 自带 Intel OpenMP 运行时，label 改为 `4.8.1+finesub<产品版本>.cu128`
> （运行时校验的判据相应改为只匹配 `finesub`，不含版本号）。中途试过的 MKL 已否决：
> 多吃 3.4 GB 内存且默认只在 Intel CPU 上启用。三后端实测与换 wheel 的最小验收见
> `tools/wt_refine_port/ct2-patches/README.md`。

### 字幕长度旋钮与设置持久化

**断句长短可调**：新增 `--split-length-scale`（0.6–1.6，默认 1.0，越小字幕越短），也可以
写进 `config.toml` 的 `[segmentation] length_scale`，桌面端「设置 → 字幕断句」有三档。它
缩放的是「多长算太长」的上限，不动下限——太短来不及读不是口味问题；已经合适的字幕不会
因此被切开。在 14 窗人工金标准上标定过：默认值不变，低于 0.8 只会多切不会切得更准。

**桌面端会记住你的选择了**：主题、语言、关窗行为、「别再问我」和处理设备此前存在网页缓存
里——安装版的网页缓存标识每次启动都不同，这些设置实际上**连一次重启都活不过**（0.4.0
发版演练实测）。现在统一存进 `user-data\settings.json`，随 `finesub relocate` 搬迁、和
卸载策略一致；任务表单也会记住上次用的模型、语言、显存档等选项。共享给命令行的设置仍写
`config.toml`，且**保留你手写的注释与排版**。

### 一轮全项目审查带来的修复

**字幕不再被静默改坏**：模型返回空答复或只有表头时，那一窗字幕此前会被当成「合法的空
结果」整段抹掉；答复少一列或多一列时，解析器会挪动列去凑，把置信度当成译文、把字数当成
时长写进成品。两处现在都按位置严格解析，对不上就判失败重试——宁可重来一次，也不要一份
看起来正常、内容已经错位的字幕。

**任务**：退出程序会一并终止正在跑的任务（缩到托盘不会），任务留在「已中断」，继续时
流水线跳过已完成的阶段；重试不再另起目录，产物仍在原处；历史页的点击不会再影响正在跑
的任务；某些可选组件缺失不再挡住所有任务。

**更新**：安装到一半被中断（关机、强杀）后不再留下一个打不开的版本——下次启动会自动
把上一版恢复回来，并把失败原因显示出来，而不是静默退出。

**桌面端选「最终字幕」不再崩**：桌面端一直在往纠错翻译传一个它已经不认识的档位值，
于是只要输出选到「最终字幕」，识别跑完之后就会在最贵的位置报错退出。只做识别的任务不受
影响，所以这个问题一直没被发现。

**任务历史里的「打开文件夹」不会再失效**：0.3.2 把任务产物从个人数据目录搬到了大文件目录，
搬迁有三处缺陷——中途被打断时历史里的路径不会被补写、便携版的产物在合并后仍留在原处、
以及从便携版带过来的历史记的是搬迁前的旧路径。任一情况下这些任务的「打开文件夹」都会指向
一个不存在的位置（文件本身一直都在）。已修，升级后自动补上。

**搬迁与卸载**：`finesub relocate` 跨盘搬迁时，你自己用 `mklink /J` 做的目录联接会
原样保留、重新指向同一个位置，不会把目标数据复制一份到新盘；卸载同样不会跟着联接删到
FineSub 目录之外。搬迁的新位置在开始搬之前就已登记，中途断电也能接着搬完。

**配额与网络**：请求过大之类的错误不再被误判为「配额用完」而白白冷却一把 key；每日配额
的计数跨日会正确重置；联网搜索拿不到某条结果时如实报告失败，不再把没拿到的内容当成
成功结果往下传。

### 历史记录里可以单独清理一个任务的中间产物

历史页每条任务多了「清理中间产物」：删掉人声分离音频、对齐结果这些占地方的文件，字幕和
识别结果（`*-stable.json`）保留，下次重跑仍然能跳过已完成的阶段。**跑挂或中断的任务也能
清**——它们才是最容易留下一堆大文件的（任务没跑完，自动清理根本没轮到），而且断点会保留，
「继续」照常可用。

### 不会再覆盖你自己的字幕

桌面版把字幕写在素材旁边，用的是固定名字（`视频名-raw.srt`）。如果那个名字已经被你手改过
的、或别的软件生成的字幕占着，它此前会被直接覆盖。现在只有**本机 FineSub 自己写过的**文件
才会被覆盖（重跑同一个视频仍然落回同一个名字，不会攒出一堆副本）；不是我们写的就改落
`视频名-raw.finesub.srt`，再冲突就往后编号。

### 临时文件不再落在「程序当时所在的目录」

纠错时每个窗口切出的音视频片段此前写在 `tmp/llm-audio-clips/`，URL 任务的「链接→视频 id」
对照表写在 `data/reference/`——**都是相对于启动程序时所在的目录**。装好的桌面版从自己的
程序目录启动，于是这两样东西落进了下次更新就会被整个替换掉的位置：片段没人清理，对照表
更新一次就没了。

现在片段放进该任务自己的产物目录（任务清理时一并删除），对照表放进用户数据目录（跟着
`finesub relocate` 走、和卸载策略一致）。仓库运行方式不变，仍是 `data/reference/`。

### 仓库运行方式：三条 `python -m` 命令换了模块路径

只影响**从源码 checkout 直接跑模块**的人；`finesub agent-clean` / `agent-ping` /
`agent-task` 这三个命令本身没变，桌面端与 CLI 壳也没变。

| 旧 | 新 |
|---|---|
| `python -m llm.agent_cleanup` | `python -m llm.agent.agent_cleanup` |
| `python -m llm.agent_ping` | `python -m llm.agent.agent_ping` |
| `python -m llm.agent_task_control` | `python -m llm.agent.agent_task_control` |

> 维护者注：`llm` 顶层的模块按职责收进了两个子包——`llm.routing`（模型事实/路由/执行身份/
> 开关轴/`config`/`api_keys`，层次方向由 `test_import_boundaries` 固化）与 `llm.agent`
> （本机 agent 后端）。`model_catalog.psv` 与 `model_routes.toml` 跟着进 `llm/routing/`，
> `pyproject` 的 package-data 键相应改成 `"llm.routing"`。纠错窗口循环
> `llm/stages/correction_loop.py` 拆成 `llm/stages/correction/` 八个模块。所有 import 路径
> 都变了；不做兼容别名。

### 模型事实表：两列改了名（自定义覆盖表需要改一行）

`model_catalog.psv` 的 `litellm_model` 改名 `api_model_id`、`model` 改名 `display_name`。
前者早就不经过 litellm，后者与「模型」这个词在别处的含义打架；新名字直说这一列是什么——
一个是请求里发给端点的模型名，一个是给人看的显示名。

**只影响自己写过覆盖表的人**：把表头那一行的两个旧列名改掉即可，底下的格子不用动。仍用
旧名启动会直接报错并给出新旧对照，不会静默忽略——一列被忽略的后果是「这个模型不在表里」，
离病灶太远。同时，改名会让 routing digest 变化，已有的纠错断点缓存作废、需重跑。

## [0.3.2] - 2026-08-06

### 桌面端：先识别、后纠错翻译，不再从头重跑

此前每个任务都写进独立目录，先跑一遍「原始字幕」、再对同一文件跑「最终字幕」时，
人声分离和识别会整个重来。现在复用只需把新任务指向旧任务的输出目录，流水线按产物
存在性自动跳过已完成的阶段（这套机制本来就在，桌面端只是从未给过入口）：

- 历史页里已完成的原始字幕任务多了「继续纠错翻译」——一键回到新任务页，识别结果
  已锁定复用，补充信息沿用当时填的（表单里已有的优先），确认后直接进入 LLM 阶段。
- 新任务页选了「最终字幕」而该文件恰有完成过的识别任务时，会主动提示可复用，一键
  接受，也可以无视它从头跑。
- 复用状态有明确的横幅说明（识别相关设置不再生效）并可随时取消。只有「原始字幕」
  任务可以被复用——已出最终字幕的目录若被复用，LLM 阶段也会因产物已存在被跳过，
  等于原样重发旧字幕。
- 顺带修了一个隐患：取消任务后换一个文件再开始，新任务会写进被取消任务的目录。

### 桌面端：新任务设置分成「语音识别」与「纠错翻译」两个标签页

处理选项按流水线的两半重新分组：识别语言、Whisper 模型、显存预算归「语音识别」；
补充信息、知识库归「纠错翻译」。输出结果（原始/最终字幕）仍在标签页之上——它是决定
LLM 阶段是否参与的总开关。选择原始字幕时纠错翻译页置灰但可见，页内一键切回最终字幕；
缺少 API key 时该标签上有小圆点提示，说明文字在页内。顺带删除了高级设置里残留的
per-task 处理设备下拉——设备选择全局化之后它已不再生效，留着只会误导。

### 桌面端：设备选择、可复制的界面、安装日志、真实进度

**处理设备**：机器上有多张显卡时，设置页可以指定用哪张（也可以选 CPU）。选择通过
`CUDA_VISIBLE_DEVICES` 生效——下游只看得见那一张、编号为 0，因此分离器/ASR/校验三个
消费者一行不用改，显存档位的假设也仍然成立；同时固定 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，
否则 CUDA 按「最快优先」排序，选单里的 1 号和实际跑的可能不是同一张卡。显卡清单用
`nvidia-smi` 在后台线程探测，界面只读结果、从不等待；选过的卡不在了会自动回退并在任务
日志里说明（重跑/续跑同样覆盖）。

**界面文字可选中复制**：此前 pywebview 注入的全局 `user-select: none` 让整个应用无法
选中——报错信息、路径、日志只能手抄。现在除标题栏与按钮外都可以正常选中复制。

**安装日志自动留存**：安装运行环境时的完整输出写入 `user-data\logs\`，逐行落盘，成功、
失败、暂停三种结局都会记一行结果；保留最近 100 份，资源页可一键打开该目录。此前只有
内存里的最后 100 行，关掉窗口就没了。

**任务进度不再一开始就跳到最后**：worker 过去在任务刚开始时就上报「目标阶段」，界面把
它当成当前阶段，于是前几步立刻打勾、最后一步一直转圈，全程再不更新。现在由 pipeline
在进入每个阶段时上报，复用已有产物的阶段显示为「已有结果」而不是刚跑完的勾。

### 桌面窗口：原生缩放、标题栏行为与主题动效

无边框窗口重新可以用鼠标拖边缩放：加回 `WS_THICKFRAME` 并用 `WM_NCHITTEST`
子类化自己判定边框与标题栏区域，安装在 WinForms 的 UI 线程上。pywebview 的启动
回调早于 WinForms 窗口存在，所以这些工作等 `shown` 之后再做；等待超时只打
`Warning:` 并继续绑定文件拖放，不再让整个回调抛异常。最小尺寸 900×620 → 720×520。
标题栏双击最大化/还原，最大化按钮改为切换。

原生边框与标题栏配色跟随应用主题（新增 bridge 方法 `set_window_chrome`，前端在
每次外观变更后回传当前 `--app-bg`/`--text`）；窗口刚创建、前端还没画出来的那几帧
按 Windows 的浅色/深色设置取色，默认的「跟随系统」主题因此不会闪烁另一半配色。

动效：侧栏切换的移动指示条、主题色切换的 View Transition、任务高级选项的展开。
全部是 CSS/浏览器原生能力，没有引入动画库；设置页可整体关闭，也尊重系统的
`prefers-reduced-motion`。设置页的「查看文档」改为应用内对话框。

安装器：简体中文不再取决于编译机上有没有装社区译本（Inno Setup 官方不含简中，
此前的探测在 CI 上一直落空，发出去的其实是英文界面）——译本随仓库分发。英文保留，
按系统 UI 语言自动选择，不再弹语言选择框。

### `.env` 密钥保护：绑定 Windows 账户

`.env` 里的 API key 不再以明文落盘：随机主密钥经 DPAPI（绑定当前 Windows 账户，带
应用 entropy）包裹后存在文件首部的 `FINESUB_KEYRING` 行，每个密钥值原地替换为
`fs$…` 密文（信封加密，纯 stdlib）。变量名、命名 key 的显示名、注释与换行逐字节
保留，`cat .env` 仍能与 `config.toml` 的 `[pools]` 对照。转换由启动迁移
`0004-protect-env-keys` 完成（源码 checkout 则在首次读取时兜底），失败一律退回明文
加 `Warning:`，绝不阻塞运行。

对外措辞统一为**绑定当前 Windows 账户的保护 + 防泛扫混淆**：它防误传文件、通用
扫盘、DPAPI 批量收割、同机其他账户，不防以当前用户身份执行的代码（程序必须能无
口令自解密）。密文随 `.env` 拿到别的机器上表现为「未配置」且**一个字节都不动**；
只有在用户显式重填/删除全部不可解密的值之后，保护才会以新机器的账户自动重建。

导出通道：`finesub keys`（默认掩码，`--reveal` 输出可直接粘回 `.env` 的明文，
`--out FILE` 落盘并警告）、桌面端设置页「显示已保存的密钥」（新增 bridge 方法
`reveal_api_keys`）、`finesub doctor` 新增 `env-keys` 状态行。换机、重装 Windows
之前先导出。

过渡开关 `FINESUB_ENV_PROTECT=0` 暂停自动加密（解密照常、迁移保持未完成），给
仍直接读 `.env` 的旧代码（未收敛的 worktree）留缓冲；收敛后移除变量即自动转换。

配套：`finesub_bootstrap/secrets.py` 成为全项目唯一的 `.env` 解析/写入层（
`llm_runtime` 与桌面 SettingsStore 的两个自建解析器删除；桌面写侧从「白名单整文件
重写」改为逐行保留式更新）；`.env-sample` 删去死配置 `EXA_POOL`/`TAVILY_POOL`；
`.gitignore` 补 `.env.*`。

CLI 与桌面从这一版起**共用一个版本号、一个 tag、一个 GitHub Release**（由
`test_the_cli_and_the_desktop_app_ship_one_version_number` 强制）。`v0.3.0` 是这条
契约成立之前发的 CLI-only release，所以联合发布线从 0.3.1 起。

### 数据布局：一个用户一份数据

三种安装形式（桌面安装版、便携版、pip 安装的 CLI）的**个人数据统一到
`%LOCALAPPDATA%\FineSub\user-data`**——此前便携版把它放在包内，于是同一个用户可能有三个知识库，
还不知道应用在读哪一个。大文件（`models`/`cache`/`tasks`）默认仍在安装目录下，可以用
`finesub relocate <目录>` 整体搬到别的盘，也可以让两个安装指向同一处、不重复下载几个 GB；
位置记在数据根的 `locations.json`，找不到就自动回落，不报错。用资源管理器整个搬走安装目录能被
自动认出来；只搬大文件目录的话，到新位置双击里面的 `register-location.cmd` 即可。

`runtime` 永远留在安装目录下：它与版本绑定，而且 uv 在同盘时用**硬链接**把 wheel 从缓存链进
环境（实测 `torch_cpu.dll` 一份数据两个路径），把两者分到不同磁盘会让硬链接退化成复制、
总占用**反而多出约 5 GB**。装环境前会比较两者所在磁盘并在不同磁盘时告警。同理，单独清
`cache\uv` 几乎不释放空间——先删运行环境再清缓存才有效。

`tasks` 从 `user-data` 里挪了出去（每任务几十 MB、无上限增长），并进了更新器的保留名单；
任务历史改存相对路径，所以手工搬动文件夹之后"打开文件夹"不再全部失效。
卸载改成三档：默认删可再生的运行环境/模型/缓存，成品字幕与个人数据分别要
`--purge-tasks`、`--purge-user-data`；大文件目录一旦搬走或共用，默认也不删（另一个安装多半还在用）。

**从仓库源码运行仍然用仓库自己的数据**（`knowledge/`、`.env`、`.state`），
`FINESUB_CHECKOUT_DATA=0` 可退出；git worktree 解析到主仓，且 worktree 内的知识库自动更新默认
跳过，需要 `FINESUB_KNOWLEDGE_WRITE=1` 才写。

### 修复：并发写坏共享数据的三处

个人数据共享之后，几个此前不可能发生的竞争变成了可能，一并补上跨进程锁：数据迁移
（锚在 user-data **外面**，因为迁移要搬的正是它）、知识库 auto-apply（锚在
`<knowledge_root>.lock`，即知识库目录的兄弟文件——放里面会被 auto-commit 收编，锚在安装根则三个
前端锁的是三个不同文件）、以及共享下载缓存里同一个压缩包的并发下载（此前会互相写坏
`.part`，报出看起来像被投毒的 SHA-256 不匹配）。任务历史改成"锁 + 重读 + 按 id 合并"回写，
不再是内存快照整体覆盖。知识库拿不到锁的行为与"仓库脏"一致：跳过、保留提案、不推进 ledger、
只打 warning。

### 修复：装完 2.8 GB 依赖后倒在最后一步改名

有用户在装 Python 运行环境时拿到 `[WinError 5] 拒绝访问`：依赖全部装完、校验也过了，
只差把 `runtime/python.staging` 改名就位。Windows 上目录改名在树内还有句柄时会被拒
（刚写完的数 GB 文件正被杀软或网盘同步扫描），目标名被占用时也是同一个错误——
`MOVEFILE_REPLACE_EXISTING` 对目录无效。改名现在带退避重试；目标名的判定改用
`os.path.lexists`（`Path.exists()` 会跟随链接，把指向别处的 junction 当成不存在）；
清理旧目录时链接只删链接本身，不再递归进它指向的目录。仍然失败时报错说明是占用并给出
处置建议，而不是抛原始 `WinError 5`；且**保留已装好的 staging**——它已通过校验，重试
只需再做一次改名，不必重装数 GB。

### 修复：知识库被写进应用目录，会被下次更新删掉

桌面端装不上时，用户会直接用包内解释器跑 pipeline。这条路绕开启动器注入的
`FINESUB_KNOWLEDGE_ROOT`，而发行包的 `app/versions/<版本>` 同样带 `pyproject.toml` +
`src/asr_playground`，于是被当成源码 checkout，知识库落在了 `app/versions/<版本>/knowledge`。
更新器的保留名单只有 `user-data`/`models`/`runtime`/`cache`，`app/` 整体替换——数据会静默消失。
checkout 探测现在显式排除这种布局，并按安装布局反推数据根（安装版走
`%LOCALAPPDATA%\FineSub\user-data`、便携版走包内 `user-data`），与启动器注入的位置一致；
`.env`、`config.toml`、限流状态文件同理。已经写错位置的那份由新的数据迁移搬回来。

### 桌面包自带命令行

包根新增 `finesub.cmd` + `finesub.py`，与 pip 安装的 `finesub` **同源子命令**
（实现下沉到 `finesub_bootstrap/shell.py`，两个前端共用），直接驱动它所在的那份安装。
用户和 agent 不必再自己拼 `runtime\python -m asr_playground.pipeline`——那正是上一条的
成因。它不负责装资源：自己就跑在托管运行时上，缺资源时指回应用内的资源面板。

### 用户数据迁移机制

新增 `finesub_bootstrap/migrations/`：按 id 记账（`user-data/.migrations.json`）而非版本区间，
因为桌面与 CLI 共享同一棵 `user-data` 且各自跳版本。启动器与命令行都会在读用户数据之前跑一次；
失败只记日志、下次重试，绝不影响启动。首个迁移把 `app/versions/*/knowledge` 搬进
`user-data/knowledge`；两边都有知识库时不自动合并，持续告警直到人工处理。

### 首次任务提示

首次任务要按需下载模型权重（合计约 3.4 GB）并预热分离器的编译路径，这些都发生在任何
进度出现之前，不说明就像卡住了。新建任务页（选好输入后）与处理页各提示一次。判据是
「没有已完成的任务」而非「历史为空」——只失败过的机器同样还没缓存任何权重。

### 修复：安装 Python 环境时界面假死

`status()` 里的运行时体检会**同步**起一个 Python 去 import torch + 整条解码链——
实测 **14.7 秒**（热缓存，冷启动更久）。而它跑在 pywebview 绘制窗口的那个线程上，
`get_bootstrap_state` 和每次资源轮询都会调。表现就是"点安装卡住、点暂停后又显示成功"
（那次点击触发了新一轮，此时探针刚好返回）。是上一轮把 `REQUIRED_RUNTIME_IMPORTS`
从 4 个扩到 8 个引入的。

`install()` 本来就在写 marker 之前验证过，而 marker 绑定 lock 哈希——所以 `status()`
再跑一遍是在 UI 线程上重复证明已证之事。改为纯文件系统检查（site-packages 下的包目录
是否还在，CT2 的补丁版从 dist-info 目录名读），进程零开销；导入探针保留在安装时。

### 模型优先复用本机已有缓存

`FINESUB_MODEL_DIR` 一旦设置就完全接管，从不看常见缓存目录，于是已经下过的权重被再下
一遍。改为"先找再下"（`finesub_bootstrap/model_caches.py`），两种粒度：

- **分离器**：精确到文件——检查 `~/.cache/audio-separator` 里有没有那个 ckpt。
- **Hugging Face**：只有一个内容寻址的缓存根、无法搜索多个，所以是**按缓存整体**判断：
  常规根里已有本管线用到的任一仓库就整体复用（包括之后新下的）。里面若只有别人的模型
  则不动它——往别人的缓存里下载既意外、卸载也清不掉。显式设了 `HF_HOME` 一律不猜。

编译加速产物（accel）**不跟着走**：它绑定单一 torch 构建与 GPU，写进共享缓存会留下
无人能归属、也无人能清理的文件。为此新增 `managed_separator_model_dir()`。

### 修复：拖拽时的卡顿

`dragenter`/`dragover` 的 Python 回调是空的，只为 `preventDefault` 而存在——而
pywebview 生成的监听器同步执行 `preventDefault` 后，仍会把整个 DragEvent（含
dataTransfer）序列化过桥调用它，`dragover` 每秒几十次。加 `debounce=500` 压掉这些无用
往返；`preventDefault` 不受影响，drop 仍然即时。

### 修复：LLM 阶段因缺 tzdata 直接崩溃

`llm/rate_limit.py` 在**模块导入时**构造 `ZoneInfo("America/Los_Angeles")`（对齐 Gemini
的日配额窗口），而 Windows 的 Python 不自带 tz 数据库。`tzdata` 从来没有出现在
`pyproject.toml` 里——只在 `.github/workflows/ci.yml` 有一行临时 `pip install`，那正是
同一个 bug 的补丁。任何按 extras 安装的环境（包括桌面托管运行时）一进 LLM 阶段就
`ZoneInfoNotFoundError`。已加进 `[harness]`（限 Windows）并重编 lock；CI 的临时行删除。

### 任务目录与日志

- 任务 id 由裸 uuid 改为 `<stem>-YYMMDD-HHMM-<6位hex>`，`user-data/tasks` 下终于能看出
  哪个目录对应哪次任务。stem 取自输出名称，没设则取源文件名。
- 任务结束后（成功/失败/取消）自动把日志写到该任务目录的 `task-log.txt`——日志抽屉
  是有上限的环形缓冲且随应用关闭消失，而值得上报的失败往往过后才被注意到。
- 「复制日志」改为「导出日志」（下载为文件），历史页右上角新增「打开任务目录」。

### 桌面任务表单调整

- 输出结果收敛为两项：原始字幕（无 LLM 处理）/ 最终字幕（须 LLM 处理）。
- 补充信息移到基础设置并给出实例文案；处理设备移入高级设置。
- 输出名称去掉常驻说明，改为**填错才提示**（与 `TaskRequest.validate_name` 同规则）。
- 知识库默认改为「自动更新」。配套修正一处会误伤的门禁：
  `required_capabilities` 现在要求 stage 真的进到 LLM 才需要 git——否则默认设置下
  每个纯转写任务都会被要求下载一个根本不会运行的 git。
- 运行日志显式可选中，并新增「复制日志」按钮。
- 资源磁盘估算修正：`uv` 那一栏此前标 24.5MB（uv 二进制），但安装它会拉取整个
  `pylock.win-py312.toml`——**实测 torch 一个就 2.56 GiB，合计约 2.83 GiB**，低估了两个
  数量级。另在提示里补上模型权重的按需下载估算（约 3.4 GB），并把「模型如何管理」
  的说明改为逐个列出权重与体积。

### 桌面支持 URL 输入

- DropZone 增加链接输入（文件选择 / 拖放 / 粘贴链接三选一）。管线一直支持 URL，
  桌面此前单方面砍掉了这个入口。链接与文件走同一条状态路径，yt-dlp 按需拉取。
- 前端的 URL 判定 `isUrlSource` 与后端 `finesub_bootstrap.capabilities.is_url`
  规则一致——不一致就会出现"UI 收下了、后端拒绝"的输入。

### 外部工具改为托管资源（git / yt-dlp），并复用系统已有的

- **manifest 新增 git（MinGit）与 yt-dlp（PyPI wheel）**，`ResourceManager` 一行未改。
  不进 lock：运行时 marker 含 lock 哈希，改 lock 会触发数 GB 的环境重建，而改 manifest
  不碰运行时。注入方式按性质分——git 走 PATH，yt-dlp 走 **PYTHONPATH**（管线是
  `import yt_dlp`，不是调可执行文件）。
- **懒装**：git 只在 `--knowledge update` 时装，yt-dlp 只在 URL 输入时装。规则收在
  `finesub_bootstrap/capabilities.py`，桌面（读 TaskRequest）与 CLI（读命令行）共用，
  避免两个入口对"这次运行需要什么"产生分歧。
- **复用系统已有的 ffmpeg / git**：照 `_find_system_python` 的模式——`which` 找到后
  实际执行校验（ffmpeg 还要查必需编解码器，缺了就退回托管副本，否则会在管线中段才炸）。
  一台已有 ffmpeg 的机器因此省掉 146MB。探测可注入，否则测试结果会取决于跑测试的机器。
  yt-dlp 无法这样复用：托管解释器看不见用户的 site-packages。
- `task_ready` 改为按请求校验，错误信息说明缺的是哪个工具（此前无论缺什么都说
  "请先安装 Python 运行环境和 FFmpeg"）。
- CLI 从 manifest 取全部资源（此前硬编码只取 ffmpeg），`doctor` 统一报告三者状态与来源。
- run metadata 新增 `tools.ffmpeg`：复用系统版本让行为依赖用户机器，路径与版本要可追溯。
- `ResourceStatus` 新增 `optional`：按需工具在资源面板里列出（否则"缺 git"的报错会把
  用户指向一个找不到 git 的面板），但不计入就绪数与所需空间，缺失时也不显示告警图标。

### 知识库：git 缺失不再让任务失败

- **`_run_git` 不再抛裸异常。** 本项目不安装 git，所以「没有 git」是常态；此前
  `subprocess.run(["git", ...])` 会抛 `FileNotFoundError`，在字幕已经落盘之后把整个
  任务带崩。改为返回 `returncode=127` 的合成结果，调用方现有的「非零即失败」处理原样生效。
- **前置拦截，不浪费配额。** `run_knowledge_update` 在 `execute and apply` 时先查 git，
  不可用就返回 `skipped: "git_unavailable"`，**一次 API 调用都不发**（此前要先花钱生成提案、
  走到 apply 才发现装不了）。
- **三条失败路径统一降级为 warning，且都不推进 chunk ledger**，所以修好后重跑会完整重做：
  git 缺失 / 仓库脏或在别的分支 / 文件已改但 commit 失败。最后一条原先在改成 warning 后
  会继续走到 `_append_chunk_ledger`——等于把「没提交」记成「已完成」，那批改动将永远不被记录；
  现已阻止。

### 桌面任务控件

- 新增 `name`（对齐 CLI 的 `--name`，产出 `out/<name>/<name>.srt`，带路径分隔符校验）、
  `extra_info`、`knowledge`（暂只暴露 none/update）、`cleanup_intermediate` 四个控件；
  移除联网检索开关（保持默认开启）。
- **中间产物默认不再清理。** 此前无条件删除，连 `stable.json` 和整个 LLM artifact 目录
  一起删——后果是重跑要从头做分离与识别，纠错翻译也没了输入和 checkpoint。现在改为可选，
  且**即便勾选也始终保留 `stable.json` 与 artifact 目录**：它们相对人声音频很小，却决定了
  重跑是否廉价。

### 桌面自动更新（此前从未跑通）

- **更新检查不再打 `/releases/latest`。** 那是仓库级的，而本仓库还发 CLI 快照与
  patched CT2 wheel——发完 wheel 之后 "latest" 就指向它，更新检查必然抛
  "missing the signed update manifest"。改为列举 releases、取最新一个**真正带签名
  manifest** 的（`is_desktop_release()`），签名与 tag 校验仍在其后兜底。顺带覆盖
  资产分批上传期间的半成品 release。
- **签名发布流程从未被执行过**：v0.2.7 只发了 portable zip，`build_release.py`
  产出的 `update-manifest.json` / `.sig` 一次都没上传。`desktop/README.md` 新增
  发布 runbook，说明四个资产缺一不可。
- `build-release.ps1` 的陈旧默认值（`-SupportedFrom 0.2.3`，一个从未发过签名
  manifest 的版本）改为空 = 所有旧版本拿 full 包，增量必须显式声明。
- **应用内一键更新接线完成。** 新增 `install_update` / `get_update_install` 两个 bridge
  方法与 `UpdateInstallManager`：安装跑在后台线程，前端轮询进度快照（与运行时资源下载
  同一套形状）。bridge 调用必须立即返回——pywebview 在绘制窗口的线程上派发它们，而 full
  包是几百 MB。设置页显示下载进度，完成后按 app/full 分别提示"重启"或"退出以完成更新"。
- **独立 updater 恢复构建。** `desktop/FineSubUpdater.py` 一直在仓库里，但
  `build-bootstrap.ps1` 不构建它——而 `_install_full` 要求
  `<root>/updater/FineSub Desktop Updater.exe` 存在，否则抛 "Installed updater runtime
  is missing"。也就是说 full 更新在任何真实安装上都不可能成功，而按上面的策略 0.2.7 →
  0.3.1 恰恰只能走 full。已补回第二个 PyInstaller 目标并实跑构建验证。
- **updater 失败不再挂住进程。** 它是 windowed 构建（无控制台），未捕获异常会变成
  PyInstaller 的模态 traceback 弹窗——而此时 FineSub 已退出，没人会去点。实测确认：修复前
  一个坏请求会让进程一直活着。现在兜住异常、写 `<request>.error.txt`、以 1 退出。

### 托管运行时

- **lock 重建。** `desktop/runtime/pylock.win-py312.toml` 上次对齐是
  2026-07-31，`[asr]` 还是 whisper-timestamped 时代；fw-refine 迁移之后它**一个解码器
  都不含**（faster-whisper / ctranslate2 / transformers / silero-vad / triton-windows
  全缺），装得上、跑不了。现按今天的 `[asr]+[harness]+[desktop-worker]` 重新生成：
  torch 2.8→2.11.0+cu128，包数 75→88。
- **补丁版 CTranslate2 进 lock。** `[desktop-worker]` 用 direct reference 锁到 release
  wheel（带 sha256）。桌面运行时只有 win_amd64/cp312/cu128 一个组合，所以 `[asr]` 那边
  规避的平台钉死在这里是零成本的。开发机与端用户安装都走同一份 lock，两边自动拿到
  补丁版。
- **运行时校验补全。** `REQUIRED_RUNTIME_IMPORTS` 原本只查分离器一侧，加入
  `faster_whisper` / `ctranslate2` / `silero_vad` / `transformers`，并校验 CT2 的
  `__version__` 带 `wtrefine`——原版能 import、能满足 `==4.8.1`，只是跑不了 fw-refine。
  探针脚本抽成 `runtime_probe_source()` 以便直接测试。
- **lock 漂移进默认测试。** `desktop/scripts/tests` 加入根 `testpaths`：契约由仓库根的
  `pyproject.toml` 打破，而 desktop CI 只在 `main` 上跑。断言从"torch 版本对不对"改为
  "三个 extra 的每个直接依赖都在 lock 里且版本相容"，拿旧 lock 验过会报 10 条。
- 桌面任务恒开 `vad_silero_assist`（CLI 仍是 opt-in）：桌面任务必经分离器，而流式化后
  它的边际成本约 1s。
- 后台控制台隐藏、真实系统托盘、角色主题与 Yanami 主题、深色完成页、纯字幕产物发布、
  本地视频先转码再 ASR、分离器/RoFormer 运行时依赖校验、实时日志跟随（PR #7）。
- 清理：`setup-dev.ps1` 的 `-IncludePipeline` 已成空开关，删除；
  `media/source.py` 的 `ensure_aac_audio` 死别名删除；测试里 8 处
  `whisper_timestamped` 桩模块 monkeypatch 与 `HEAVY_IMPORTS` 条目删除（生产代码早已
  零引用）。

## [0.3.0] - 2026-08-05

### 移除

- **`whisper-timestamped` backend 整个移除**，`fw-refine` 成为唯一 ASR backend。
  同时删除：`--asr-backend` 开关（pipeline 与 vad-asr）、`asr-wt` 命令、`WtModelPool`、
  `naive_approach` 退避梯子、`asr_transcribe_seed` / `whisper_timestamped_mode` metadata
  （fw-refine 从不读 seed，那条记录一直是假的），以及 `whisper-timestamped`、`dtw-python`
  两个依赖。checkpoint fingerprint 的 `asr_backend` 字段随之删除（只剩一个 backend）。
  理由：为优化要改 refine 内部逻辑，维持两套行为对齐的成本不划算；迁移验收（5 素材 /
  50.6 分钟）显示 fw-refine 快 3.19×、内容量差 ≤1%、救援活动在每个素材上都更少。
  回溯点：`dev` 的 `1fcc4e1`。
  **`asr-refine` extra 因此从可选变为必需**。

### 新增

- **`--qwen-verify {auto,on,off}`（默认 auto）：第二模型校验证据。** Whisper 池释放后加载
  Qwen3-ASR-0.6B，对三类嫌疑段（整段收尾套话、CJK 主导 run 里的 Latin 段、噪声腿将丢弃的段）
  重认，证据写进段级 `qwen_verify`，决策留给 stabilize。`[asr]` 因此新增
  `transformers>=5.13,<6`（pip 增量约 100MB，模型首次运行下载约 1.5GB 至 HF 缓存）。
  并入 `[asr]` 而非可选 extra，是为了让同一条命令在任何安装上产出相同的 stable。
- **词首修正**（`speech/recognition/word_starts.py`）：`[*]` disfluency 块按能量门决定
  融合/删除，再对首词做 VAD interval 与 pause_hint 锚点 clamp。gold 上词首 |err| 中位
  41→18ms。
- **LLM 窗口质量护栏 `max_window_subtitle_tokens`**（`ModelLimits` 默认 10,000，config.toml
  `[chunking]` 可覆盖，`0` 关闭）：单窗 `<asr_result>` 的 token 上限，独立于输出系数——
  窗口过长时翻译质量会掉，哪怕输出装得下。窗口数估算与真实 countTokens 校验两处生效，
  超限走既有的 k+1 重排；快速模式的 auto 判定也以它为第三道门（快速窗口就是全片，
  最容易撞上）。
- `energy.WaveformObserver` 钩子：让第二个信号搭车读取 VAD 已经算好的归一化 block，
  不必再解码/重采样/归一化一轮。不传时 energy 模块的每个输出值逐字节不变。
- `fw-refine` 的 multi-audio batch 设计与本机实测落入 `docs/wt-refine-port.md`：CT2 的
  `real_audio_frames` 改为逐样本、split-encode 批模式、确定性契约，以及模型 × beam × GPU profile
  的 batch size 档位表。迁移本身值 6.3×、batch 再叠 1.8×，故 batch 不阻塞 P0。
- 新增显式 opt-in 的 patched CT2 `fw-refine` checkpoint：greedy/beam=5 以 1-pass
  winner trace 对齐 WT refine，默认收集低成本 path 信号；修复后的 disfluency 保持显式
  开关，启用后同样以 `alignment_events` 透传到 aligned/stable 产物。FineSub 暂不消费。
- `fw_refine_backend.transcribe_batch()`：一次批量解码若干 ≤30s 窗口。split-encode（逐窗口
  encode、只批 decoder），再把每条结果回放进普通 `transcribe()`，因此 segment/词/事件的组装
  只有一份实现。24 个真实生产窗口上文本一致 22/24（18 条连词级时间逐位相同），加速 1.73×。
  调用方的组批策略尚未实现。

### 变更

- **VAD 归一化的峰值限幅由全局缩放改为逐样本 clamp——会改变 VAD 输出。** 旧实现是
  `x * (0.98 / 全局峰值)`，让**单个最响样本**决定施加到整轨的位移：kaguya60 的 0.00029%
  样本把 60 分钟压低 2.18 dB，mia 的 0.00004%（约 25 个样本）把 108 分钟压低 4.12 dB；
  7 条真实分离人声里 4 条触发。而 VAD 判据里有一批绝对 dBFS 阈值，它们能跨文件通用正是
  靠上一步把局部 RMS 对到 −24 dBFS，全局缩放随后又按文件拆掉一部分标定。改为削掉越界样本
  后，weighted 轨上受影响帧从「全部有信号帧」降到 179/4292，越阈帧从约 2000/20000 降到
  0/1。**限幅曾触发的文件重跑会得到不同 segment**（kaguya60 752→753、语音 +4.8s；
  mia 2561→2522、+46.1s），未触发的文件逐字节不变；旧产物重跑即可。同时也删掉了流式
  路径中专为施加全局标量而存在的第二遍 pass。详见 `docs/vad-energy.md`。
- **ASR 固定单 worker，移除单文件分片设计**。`sharding.py`、`ResourceProfile.wt_instances`、
  `--wt-workers`（pipeline 与 vad-asr 两处）、interval ownership 标记与合并、shard partial
  一并删除；GPU profile 现在只决定人声分离实例数，metadata 不再记录 asr workers。
  实测 worker=3 相对 worker=1 在 wt 上仅 1.40×、fw-refine 上仅 1.20×，代价是显存 2.4→6.5 GB；
  换到 fw-refine 后 ASR 已非瓶颈（人声分离占语音段 72%）。回溯点：`dev` 的 `1fcc4e1`。
- **ASR 分组改为按合成后的音频长度规划**：组尾垫料（至多 0.7 秒原始音频 + `--gap` 秒静音）
  此前不计入分组长度，于是按 30 秒规划的组实际可达 31 秒、溢出编码窗口。11 个真实 clip 上
  超窗分组 102 → 56，总组数 388 → 405。**这会改变分组边界，因而改变 ASR 输出**（11 个 clip
  中 9 个分组不同）；旧产物不会自动失效，需要重跑才能得到新分组。`combined_group_duration()`
  语义不变（auto language 短组启发式仍按「说了多少话」判断）。
- `asr-pipeline` 新增开发用 `--asr-backend {wt,fw-refine}`（默认 `wt`，与 `vad-asr` 一致）。
  面向一般用户的文档不介绍该开关。
- ASR checkpoint fingerprint 新增 `asr_backend` 且**无默认值**：两个 backend 对同一段音频给出
  不同词级时间，中断后换 backend 续跑此前会静默复用另一侧的 partial，把两种输出缝进同一份产物。
- `faster-whisper` 与 `ctranslate2` 移出 `asr`，独立为 **`asr-refine`** extra 并精确钉版
  （1.2.1 / 4.8.1，均为当前最新且互相兼容）。fw-refine 不在 `asr-pipeline` 的可达路径上，
  普通用户不必装一份链了 CUDA 的 CTranslate2；同时也消除了 `asr` 与 desktop runtime lock
  之间「lock 缺这两个包」的静默不一致。fw-refine 继承 faster-whisper 内部实现并读取 CT2
  解码轨迹，小版本变动可能悄然改变输出；升级顺序固定为先 faster-whisper 后 CT2——CT2 的
  可选范围由 fw 声明的 `>=4.0,<5` 决定。

### 性能

- **`--vad-silero-assist` 从「比 VAD 本身还贵 4 倍」降到几乎免费**：60 分钟分离人声上
  该 opt-in 此前在 17 秒的 VAD 阶段之上再花 72 秒，现在 CPU 约 3 秒、CUDA 约 1 秒。
  silero 的逐帧 JIT 调用（每 32ms 一次，一小时 112,500 次派发）改为：帧间独立的
  STFT+encoder 整批计算，LSTMCell 的权重驱动全序列 `nn.LSTM`；概率再搭车在 VAD 自己的
  流式 block 上算，省掉第二遍解码与整段波形常驻（0.23 GB/小时）。与逐帧实现的最大概率
  差 1.4e-05，判据阈值零翻转。CUDA 路径显式关闭 TF32（开启会让 112,500 帧中 16 帧越过
  `CAP_SIL_THR`）。

### 修复

- `fw-refine` 在 CTranslate2 缺少该设备的矩阵后端时，给出指向构建要求的可读错误，而不是从库
  深处抛出的 `No SGEMM backend on CPU`。该情况无法从 `get_supported_compute_types()` 查出
  （只有 CUDA 后端的构建对 CPU 仍报 `float32`），只能在首次 encode 处补上下文。
- `fw-refine` 的词切分不再丢弃永远无法解析的 `U+FFFD` token。真实幻觉撞上解码上限时会在半个
  字符处截断，旧实现一直等待补全、循环结束时静默丢弃该 token，使词分组与 one-pass 解码轨迹错位并
  抛 `ValueError`；310 个真实生产窗口里有 6 次触发。同时补上 fw-refine 的退避链——任何失败改退到
  该后端自己的 teacher-force 对齐（`naive_approach` 是 whisper-timestamped 的选项，对 fw-refine 会
  原样重放同一次调用），两次都失败才丢弃该 group，不再终止整个 run。
- recall 临时组的 complement 切片现在继承源 interval 的 shard 归属。此前它新建裸 dict，
  分片合并时以「missing interval ownership」中止**整个 run**；触发条件是 workers ≥ 2 且
  某个 block 命中 recall（≥5s 未覆盖），与 ASR backend 无关。
- Desktop 的 API Key 设置改用现行 CLI provider pool 变量，Windows AI runtime
  lock 重新对齐 Torch/Torchaudio 2.8；新增跨 Desktop/CLI 契约测试，避免两侧再次漂移。
- Desktop 完成页只展示实际存在的产物并暴露 run metadata；WebView 重载会恢复仍在
  运行的任务，翻译模式明确提示 Gemini 媒体片段上传。
- Desktop 默认 LLM level 和后处理 profile 范围与生产 pipeline 对齐；新增独立
  `desktop/VERSION` 作为桌面发布版本的单一来源。

## [0.2.0] - 2026-07-30

### 新增

- ASR 新增单文件 Whisper Timestamped 分片、分组 checkpoint、运行时 metadata、
  GPU stage gate、stall watchdog 与资源用量记录；4/8/12/16GB profile 会在文件内部
  分配 separator 和 WT 并发。
- 字幕分句改为全局 DP，并新增分割点金标准、标注工具和系统化评测资料。
- LLM harness 新增可配置 API key pool、sticky retry 后的组合冷却，以及更完整的
  token budget、任务报告和搜索证据处理。
- 新增 `config.example.toml`、桌面 launcher 资源配置和 Windows token counter 更新。

### 变更

- 生产代码重组到 `asr_playground` namespace，明确 media、speech、subtitles 和
  workflows 边界；命令行入口和打包清单同步迁移。
- Batch 从文件级 ASR 并发改为单文件独占 profile、文件内分片并发，避免两层并发相乘。
- 桌面应用同步外观控制与 UI 刷新，并在构建 bootstrap 时优先使用 conda env-root Python。
- OpenCC 转简加载器与本地 token counter 改为跨调用复用；ASR 模型从共享 checkpoint
  直接构建 FP16 实例，降低重复加载开销。

### 修复

- 修复 wheel 漏装 `batch`、`gpu_stage_gate`、`run_metadata`、`segment_split` 和
  `wt_shard` 顶层模块，以及 license metadata 无法在声明的 setuptools 下限构建的问题；
  增加顶层源码与 packaging 清单一致性测试。
- ASR checkpoint schema 升至 v2；旧 partial 明确失效并从头重跑，sharded merge
  遇到缺失 interval ownership 的结果会显式报错，不再静默丢字幕。
- `segment_split` 对有文本但没有 word timestamps 的 segment 合成一条带来源标记的
  segment-span word；无法安全归一化时保留原输入，不再从全局 DP 输出中消失。
- Reference ingest 在迁移到统一 batch workflow 前，先与普通 batch 一样固定每任务
  `wt_workers=1`，避免文件级并发与 shard 并发相乘。
- Pipeline 的 LLM round 汇总和 task report 现在遵循显式 `task_artifact_dir`；batch
  同一 logical run 的后续 pass 会继承已执行 stage，stage metadata 不再混出
  `reused` 加旧 `elapsed_sec` 的矛盾记录。
- 字幕渲染在时间轴后处理前修复 cue 重叠，保证不丢文字。
- 修复 separator dotted 临时文件识别、Gemini key 全部跳过时的错误类型，以及
  flash gap/end pad、RPM 失败计数等重试与边界问题。

### 变更

- `--wt-workers` 明确为开发/不安全 benchmark 覆盖参数；生产调度继续由 GPU profile
  和 batch runner 决定。
- 补齐 run metadata、WT sharding、segment split 和 packaging 测试的 pytest 域 marker。

## [0.1.1] - 2026-07-27

Prompt version: `zh-subtitle-correction-csv-v65`。

### 新增

- 新增 Windows Desktop 应用，提供任务管理、资源管理、运行时设置与日志查看等桌面工作流。
- Pipeline 新增 `--name`；视频任务默认使用高质量多模态模型，并在不可用时自动降级。

### 变更

- Prompt 升级到 `zh-subtitle-correction-csv-v65`：删除 CapableA、BasicC 及 BasicC 的 JSONL 输出支线，现行变体为 capableB/C + basicA/B。
- 单窗口 query、correction 与 fast round 1 的目标字幕序号每窗重置为 `1..N`，只读前文按时间顺序编号为 `1-M..0`；harness 校验后映射回稳定源序号，oneshot、replay、benchmark 与任务反馈同步采用该契约。
- 改进搜索证据包和研究阶段，减少冗余上下文并提高可用证据密度。
- 完善 ASR 语言历史、分组 checkpoint 与 Whisper fallback，扩展救援阶梯和尾段回交策略。
- ASR 依赖更新至 PyTorch 2.8 系列。

### 修复

- Pipeline 与最终 SRT 输出改为原子写入，避免中断时留下不完整文件。
- 修复 clip 预取线程安全问题，并补充 Pipeline 失败诊断信息。

## [0.1.0] - 2026-07-23

首个 beta 版本。Prompt version: `zh-subtitle-correction-csv-v63`。

### 功能

- 本地长音频转字幕完整流水线：人声分离 → VAD + Whisper ASR 对齐 → ASR 稳定化 → SRT 输出
- 实验性 LLM 纠错与翻译后处理（Gemini），支持 6 档 preset（route × level）、fast 模式、多轮搜索调查
- 批量运行（三阶段流水线并行，单项失败隔离，断点续跑）
- 本地知识库（自动采集/统一更新/精修对照）
- URL 输入支持（yt-dlp 下载 + 自动 ID 映射）
- 词级字幕输出
- SRT 后处理（繁简转换、短轴延长、标点清理）
- GPU 显存档位（8/12/16GB）自适应
- 流式 VAD 与流式 ASR 对齐（内存上界恒定，支持任意时长音频）
- LLM session resume + 纠错窗口中途 resume
- Prompt 变体系统（capableA/B/C + basicA/B/C）

### 依赖

- Python >= 3.12
- 核心流水线：torch~=2.9.0, torchaudio~=2.9.0, numpy, soundfile, whisper-timestamped, onnxruntime, audio-separator, numba
- LLM harness：httpx, yt-dlp, opencc-python-reimplemented
- 开发：pytest, pytest-xdist, sudachipy, sudachidict_core

### 已知限制

- LLM 纠错翻译层为实验性功能，默认不包含在生产 stage 中（需 `--stage translated-srt` 或 `final-srt` 显式启用）
- 仅提供 Windows 预编译 token counter 二进制；Linux/macOS 回退到免费 countTokens API 或启发式
- 无 GPU 时回退 CPU（速度显著下降）
