# speech 侧未完成工作（索引）

语音链路（VAD / ASR / 分句 / 分离器 / 第二模型 / 说话人分离 / 一体转写）还没做的事，一处看全。与
[`llm_followups.md`](llm_followups.md) 对称：那份是 LLM 侧的唯一入口，这份是 speech 侧的。

> ⚠ **状态只写在这里。** 第一节就是现行状态，改状态**只改这里**。
> 逐项证据与判据在维护者的本地计划里（**不随公开快照发布**）；
> 那边的状态格记的是**注明日期的判定，不再随进度更新**，所以两边不需要同步。

说话人分离的能力边界、失败模式与实测结论在 [`diarization-findings.md`](diarization-findings.md)，用户向说明在
[`manual/diarization.md`](manual/diarization.md)。

## 开着的工作

每条一行：状态、一句话现状、触发条件或下一步、去处。定了不做或暂缓的在[定了的](#定了的不做暂缓)，次要、没有触发证据的在[搁置的小项](#搁置的小项)。

### 说话人分离

| 项 | 状态 | 现状、触发条件与去处 |
| --- | --- | --- |
| **多人串轨** | 下一轮的主目标 | 一个人的行落进另一个人的轨。换声纹已试过、不采纳（[`decisions.md`「vbx 聚类的声纹不换模型、也不改喂法」](decisions.md#vbx-聚类的声纹不换模型也不改喂法2026-11-08)）；候选修法：串轨文件上用 MOSS（Pro 或开源版整集单趟）自己的说话人。读数都是 Pro 的：≤5 人、≤46 分钟的 4 集上 B³ 0.96–1.00，两个声学参照 0.73–0.94（当时的评估说开源版整集单趟读数相同、Pro 多的是稳定）；但样本是同一档节目、每格 1 次，另一档节目（生产本来不串轨）上开源版没复现；Pro 用它自己的行最省，挂到生产的行上贵近一倍（[`diarization-findings.md`](diarization-findings.md)「10.3」的表）。前提：先能按文件认出串轨，再在别的节目上冻结规则验（按参照估人数、≤4 人才用、说话时长覆盖不足就回退；覆盖率抓不到「覆盖完整但并人」）；要用就用它自己的行，字幕行另切 |
| **按时段给身份** | 没开工 | 轨会中途换主人 |
| **其他语言上的分轨与置信标** | 没做 | 置信标的合成分只在日语上拟合过，最重的一项是一行的字数；先做英语。做法：先照 [`diarization-findings.md`](diarization-findings.md)「4.6 逐行置信标」「6.5 声学换人拆行」的口径写下门槛，再在 AMI（英语）、AliMeeting（中文）上用行真值读数，维护者本地已有这两套的 MAI2 转写。中英的字幕长度 2026-10-24 已量过、不用改 |
| **只出 `-raw.srt` 的运行也默认开声学分离** | 暂缓（[`decisions.md`「只出 `-raw.srt` 的运行不默认开声学说话人分离」](decisions.md#只出--rawsrt-的运行不默认开声学说话人分离暂缓2026-10-15)） | 翻默认还缺：hina 以外的多人日语素材；其他语言上的分轨与置信标（上一行） |
| **裁判的拆行、补字、插行** | 带音频的说话人裁判在生产里开着收数据 | 余项（补出的行怎么显示、与纠错的分工）在 [`llm_followups.md`](llm_followups.md) |

### 识别、稳定化与时间轴

| 项 | 状态 | 现状、触发条件与去处 |
| --- | --- | --- |
| **没人说话的长段上的误听** | 没开工（2026-10-18） | 有歌、BGM、环境音的素材上每小时 12–21 行，多是「you」「Bye.」这类短英文；stabilize 已删掉约三分之二，剩下的近一半是证据否决救回的。要别的信号认，例如 Silero 的发声概率（产物里没存，要另跑一遍） |
| **两处坍缩检出合成一条阶梯** | 没开工（2026-11-03） | ASR 阶段的词堆叠检出（触发 `transcribe.py` 的异常隔离重解）与 stable 的挤短检出（`is_squeezed`，触发阶梯）各管各的：维护者第二级与留出集的 375 行挤短行里，前者只抓到 1 行。合成之前先用同一把尺子量 ASR 阶段自己的触发点（只换时间对换文字）；改成只换时间就是改 ASR 行为，要写明质量影响（[`decisions.md`「坍缩行的阶梯」](decisions.md#坍缩行的阶梯重解只取时间不收重试与两端有支持2026-10-2911-03)） |
| **尾条被压成约 1 ms 时不提示** | 记一笔（2026-11-09，手册冷读查出） | `subtitles/postprocess.py` 的 `_bound_cues_to_audio_end` 把开始时间越过音频结尾的行压成 1 ms、往前排，不打 `Warning:`，用户只能自己翻最后几条的时长（[`manual/outputs.md`「`--stage`：指定停止的阶段」](manual/outputs.md#--stage指定停止的阶段)写了怎么查）。改法：压了几行就打一行告警，写明哪几行 |
| **silero 辅助的「无声段切除」切掉安静真话** | 等触发（owner 2026-10-06） | 触发条件：再收到一例安静单人素材整句被切的反馈，或手头攒到 3 份以上这类素材。病例、触发面与改的方向在 [`decisions.md`「silero 辅助的「无声段切除」暂不改」](decisions.md#silero-辅助的无声段切除暂不改等触发2026-10-06)；脚本、逐份表与听审清单在本地 `data/silero-carve-surface-2026-10/` |
| **语言票翻转重解的两个阈值没标定** | 等素材 | `margin=0.15`、投票门槛 `2/3` 一个负例都没有：要英配 PV、外语歌、外语嘉宾这类素材。见 [`asr-align.md`「待标定（P2，未做）」](asr-align.md#待标定p2未做) |
| **CPU 全链路跑完之后收尾挂死** | 等复现（2026-09-08 一次，未定位） | 现场：`--device cpu --gpu-tier cpu` 跑完 11.1 分钟素材，aligned 已完整序列化进 `.part`（157 段、`total_sec` 725.8），之后进程 0% CPU、95 线程、5.2 GB 常驻挂了 57 分钟，`.part` 始终没被 rename；杀掉之后 `.part` 内容完好可用。下次复现直接开 `ASR_STALL_WATCHDOG_SEC=60`（日志落点 `ASR_STALL_WATCHDOG_LOG`），它周期性 dump 全线程栈。证据只缩到「aligned 写完之后的收尾」：最后一条产出是 `peak_mem exceeds the cpu tier limit` 告警，之后只剩 `watchdog.disarm()`（join 有 2 s 上限）与 `os.replace`，都解释不了一小时。别照 `ct2-patches/README.md` 的「Ruy 析构死锁」直接归因：同一趟里 `model_pool.close()` 已经成功过一次 |
| **被墙时仍会逐次等超时的语音侧加载** | 记一笔（2026-11-09） | 不在模型清单里的 Hugging Face 加载（`speech/joint/sbd.py` 每次 `sentence_ends` 都重试、`--asr-aligner`）没接连不上的判定；现行判定只管 API 调用与下载（[`llm_harness_behavior.md`「连不上的 host」](llm_harness_behavior.md#连不上的-host)）。35t Live 的 websocket 与 urllib 调用拆不开 connect 时限，定了不动（[`llm_design_notes.md`「连不上的 host 与离线 tokcount 的决策记录」](llm_design_notes.md#连不上的-host-与离线-tokcount-的决策记录2026-11-09)） |

### 一体转写

现行行为在 [`joint-asr.md`](joint-asr.md)。

| 项 | 状态 | 现状、触发条件与去处 |
| --- | --- | --- |
| **云端任务可选的声学说话人分离，裁判多一列声学编号** | 要做，排在后面（owner 2026-11-14：做进生产，先记 follow-up） | 现在云端任务的说话人只来自它自己的识别，不跑声学说话人分离（D6，[`joint-asr.md`](joint-asr.md) 设计取舍表）。2026-11-14 在四窗上量过：生产的云端裁判 prompt 每行多一列声学层的编号（加一段「声学层会把声音相近的人并起来、以内容为准」的说明与编号对照表），agy Gemini 4 Hydrogen 文本裁判之后的每分钟编辑代价 MAI2 5.2 → 3.6、35t 3.9 → 3.5、Scribe 1.7 → 1.5，补跑第二遍方向一致；收益来自后端把两个人并成一条轨的窗。本地 MOSS 与 MOSS Pro 没跑这一臂。做法（owner 2026-11-14 定的思路，与云端 ASR 的本地工程一样）：重的工作 auto 时关、允许用户手动打开，所以 auto 不跑声学分离，用户打开时才跑、裁判才多这一列。要改：云端任务跑声学说话人分离与 `-diar.json`（今天不读不写）、裁判行格式与说明（过冷读）、测试与 `docs/manual/` 同改；声学列的取法（覆盖一行一半以上的声学轨）、说明文字与读数在维护者的本地实验记录 |
| **35t 整窗不带说话人的声纹配人** | 默认关，待判 | `repair.SPEAKERLESS_BY_VOICE`（不是用户选项）：读者包看不见说话人，要在有逐行说话人真值的整集上、碰上整窗不带说话人时判；判完删掉这个开关（[`joint-asr.md`「退化的修补」](joint-asr.md#退化的修补)） |
| **Scribe 的三个没试的旋钮** | 候选（2026-11-11） | keyterms：把知识库里的人名、专名送给 Scribe（另加费，价目表 $0.05 / h 与参数说明 +20% 不一致）；两条触发（专名漏写率比 MAI2 高 1.5 倍、输窗里专名听错占 1/3）都不成立，只在 hina 对 MAI2 的输窗里专名占 6 / 16。要试先重转 hina 3 集 + 中文 1 份（约 3 h）。`diarization_threshold`（默认约 0.22）调低可能分开 CALLHOME 两通只给一个标签的、hina-70 并掉的人，要重转，文字也会变。服务端切段（>8 分钟的文件切最多 4 段）的接缝没定位，hina-73 那次坍缩是不是在接缝上不知道 |
| **ElevenLabs 免费档没实测的两件** | 不专门测，遇到再看（owner 2026-11-11） | 免费档的订阅接口给刷新时间已实测（2026-11-11）；每小时积分按 2222 起算，owner 认可这个量级、由调用后的修正兜着。没测的：调用刚返回时已用积分是否已更新（没更新时只是不修正换算率）；额度用完时转写接口回什么——429 现在按限流重试三轮后报错停下、不换 key，若它表示额度用完，没开 User: Read 的 key 会让任务停下而不是换下一把。哪次真碰到，看那次的报错与 `.state` 的 `cloud_asr` 里 `elevenlabs/scribe_v3/<key>` 一行（[`joint-asr.md`](joint-asr.md) 的设计取舍表里 Scribe 的 key 池一行） |
| **两个顺手看到的旧问题** | 没修 | `lines._PIECE` 只认 ASCII：「café」拆成「caf é」、邮箱在 `@` 处断开（宽窗不查字符集之后更容易碰上英文引语）；35t 一窗退化拆两半时，不到 60 s 的源后一半的起点是负数（`mid - OVERLAP / 2`） |
| **MOSS 两家 S5b 的段尾：Silero 一直听到语音到段尾时不再减 0.1 s** | 候选，没改（owner 2026-11-09 定先记在这里） | 现在段尾取 Silero 最后一段语音的终点再减 `OFF_LAG` 0.1 s（`moss_timing.py`），Silero 一直听到段尾时也照减。约 55% 的段属于这种，日语人工逐词尺子上原始段尾几乎正好（中位 −3 ms），减了之后绝对误差中位 76 → 106 ms；这一半不减时，全部段尾的绝对误差中位 72 → 59 ms、p90 208 → 190 ms（另约 45% 的段 Silero 在段尾之前判出语音结束，这一半按 Silero 取段尾在各把尺子上都明显改善，日语人工尺子上 136 → 55 ms；不减的候选只针对前面那约 55%）。证据只有一份日语素材加 9 个英语段尾，改的是 VAD 一类的时间常数（B 类）：要改先在一两份不出自 Whisper 的人工逐词时间上验（最好是自然对话、英文或中文），并量到交付层（行尾另有 0.3 s 的显示补边）。读数在维护者的本地评估报告里 |

### 硬件与分发

| 项 | 状态 | 现状、触发条件与去处 |
| --- | --- | --- |
| **encoder flash attention 在 RTX 30 / 40 上的实卡验证** | 等拿到这两代卡的验收记录 | 开着时的正确性只在 RTX 5070 Ti 上验过（走 compute_90 PTX 的 JIT）；RTX 30 / 40 跑的是 flash kernel 的 sm_86 / sm_89 SASS，从没执行过，而它们正是默认开的主要用户（owner 2026-10-10：照现状发、在 CHANGELOG 注明）。做法：同一段特征分别开 / 关 flash 调 `encode()`，比 encoder 输出的余弦与最大差，应在 fp16 噪声量级（RTX 5070 Ti 上 cos 0.99958）；再跑两份夹具看文本 |
| **分离器 AOTI 构建期间，同进程的 SDPA 只剩一个后端** | 记一笔（2026-11-13，修 issue #19 时查出） | 构建包时 `sdpa_kernel` 罩住 export 与 compile，这段时间里同进程别处的 SDPA 调用（例如批处理里与之重叠的 Qwen 裁判）只看得到那一个后端，可能报 `No available kernel`。只在还没有包、要首次构建的那一趟才有这一窗；运行期已改成直接调算子（[`separator-optimization.md`「E8：在编译路径上复测此前被否的 E1–E3（2026-08-03）」](separator-optimization.md#e8在编译路径上复测此前被否的-e1e32026-08-03)）。**触发**：有人在首次构建的那一趟里报这个错 |
| **多张 Intel 卡时的「更大的卡」提示** | 等真机（2026-11-13） | 程序总用 0 号 Intel 设备，核显排在 Arc 独显前面时分离会落在核显上；AMD 已有一次性的 `amd-device-choice` 提示（[`gpu-profiles.md`「厂商：AMD 与 Apple」](gpu-profiles.md#厂商amd-与-apple2026-09-17)），Intel 还只在手册里写了 `ZE_AFFINITY_MASK`。做法照 AMD 那条，但两张卡的枚举顺序与 `total_memory` 读数要在真机上看过才写得对。**触发**：拿到有两张 Intel 卡的机器的验收记录或反馈 |
| **AMD 的 CT2 变体** | 等反馈 | 收到验收记录或问题报告怎么办、下次升级上游 CTranslate2 时没人反馈就删哪些，都在 [`decisions.md`「AMD 的 CT2 变体」](decisions.md#amd-的-ct2-变体盲编也发布没人反馈就在下次升级-ct2-时删2026-09-17) |

## 定了的（不做、暂缓）

每条都是量过或核实过之后定的。每条一行：定了什么 → 它的决策记录；理由与重开条件只写在记录里，动手前先读。

- 分离器分块权重改三角权重（已撤回），以及 FP8、手写 GEMM、真 batch、换 checkpoint、TensorRT、16 kHz 分离 → [`decisions.md`「分离器分块权重不改三角权重」](decisions.md#分离器分块权重不改三角权重2026-08撤回)
- 把 Silero 提升为基础检测器 → [`decisions.md`「不把 Silero 提升为基础检测器」](decisions.md#不把-silero-提升为基础检测器2026-08)
- 向 Whisper 主解码器注入术语（`initial_prompt` / `hotwords`）→ [`decisions.md`「不向 Whisper 主解码器注入术语」](decisions.md#不向-whisper-主解码器注入术语2026-08-31)
- 拿外部 DER 数字当目标 → [`decisions.md`「不拿外部发表的 DER 数字当说话人分离的目标」](decisions.md#不拿外部发表的-der-数字当说话人分离的目标2026-08)
- 凭机器参考的 CER 给 ASR 底座定级 → [`decisions.md`「不凭机器参考的 CER 给 ASR 底座定级」](decisions.md#不凭机器参考的-cer-给-asr-底座定级2026-09-07)
- 靠改 Whisper 权重修幻觉、复读、提前收尾与时间戳 → [`decisions.md`「不靠改 Whisper 权重修幻觉、复读、提前收尾与时间戳」](decisions.md#不靠改-whisper-权重修幻觉复读提前收尾与时间戳2026-11-07)
- 往 CT2 加 CUDA graph capture → [`decisions.md`「不往 CT2 加 CUDA graph capture」](decisions.md#不往-ct2-加-cuda-graph-capture2026-08)
- 靠第二模型比字删疑似幻觉、证据一两个字不算否决、删极短复制段 → [`decisions.md`「疑似幻觉不靠第二模型比字删除，也不删极短复制段」](decisions.md#疑似幻觉不靠第二模型比字删除也不删极短复制段2026-10-1718)
- 坍缩行的重解换文字、扰动与采样重试、「两端有支持」 → [`decisions.md`「坍缩行的阶梯：重解只取时间，不收重试与「两端有支持」」](decisions.md#坍缩行的阶梯重解只取时间不收重试与两端有支持2026-10-2911-03)
- 坍缩行挪过去之后，被压住的行叠着留着 → [`decisions.md`「坍缩行挪过去之后，被压住的行叠着留着」](decisions.md#坍缩行挪过去之后被压住的行叠着留着2026-11-03)
- 让 Qwen3-ASR-1.7B 接管产出文字的调用 → [`decisions.md`「不让 Qwen3-ASR-1.7B 接管产出文字的调用」](decisions.md#不让-qwen3-asr-17b-接管产出文字的调用2026-11-08)
- 用 MOSS 解码器的 attention 给字定时间 → [`decisions.md`「不用 MOSS 解码器的 attention 给字定时间」](decisions.md#不用-moss-解码器的-attention-给字定时间2026-11-09)
- 强制对齐的长段，段首段尾换回 Silero → [`decisions.md`「强制对齐的长段，段首段尾不换回 Silero」](decisions.md#强制对齐的长段段首段尾不换回-silero2026-11-09)
- 本地 MOSS 为 WSL2 与 Linux 专属的库做优化 → [`decisions.md`「本地 MOSS 不为 WSL2 与 Linux 专属的库做优化」](decisions.md#本地-moss-不为-wsl2-与-linux-专属的库做优化2026-11-10)
- 本地 MOSS 中文行首比 Pro 晚约 0.08 s，不改 → [`joint-asr.md`「本地 MOSS」](joint-asr.md#本地-moss)
- 文本侧 LID 作第三票（P16）→ [`decisions.md`「不做文本侧 LID 第三票」](decisions.md#不做文本侧-lid-第三票p16)
- 组批（A1，`--asr-decode-batch`）默认开 → [`decisions.md`「组批默认关，推荐值按模型分」](decisions.md#组批默认关推荐值按模型分a12026-09)
- `collect_gaps` 扩探测面覆盖碎片化缺口 → [`decisions.md`「`collect_gaps` 不扩探测面」](decisions.md#collect_gaps-不扩探测面2026-10-0611-09)
- 字幕空当里漏掉的话由 LLM 层去补 → [`decisions.md`「字幕空当补漏：LLM 层不做，归 speech 侧」](decisions.md#字幕空当补漏llm-层不做归-speech-侧2026-10-16)
- vbx 聚类的声纹换模型或改喂法 → [`decisions.md`「vbx 聚类的声纹不换模型、也不改喂法」](decisions.md#vbx-聚类的声纹不换模型也不改喂法2026-11-08)
- 只出 `-raw.srt` 的运行默认开声学说话人分离（暂缓）→ [`decisions.md`「只出 `-raw.srt` 的运行不默认开声学说话人分离」](decisions.md#只出--rawsrt-的运行不默认开声学说话人分离暂缓2026-10-15)
- Whisper 链路里修耳语（分离器的救援、换原始轨解码）→ [`decisions.md`「Whisper 链路不修耳语，交给一体转写」](decisions.md#whisper-链路不修耳语交给一体转写2026-11-10)
- MOSS 在听不清的耳语上提前收尾、Silero 当闸门的救援接不住 → [`joint-asr.md`「本地 MOSS 的救援」](joint-asr.md#本地-moss-的救援)
- 一体转写模型整体替代生产链路 → [`decisions.md`「一体转写模型不整体替代生产链路」](decisions.md#一体转写模型不整体替代生产链路2026-09-2610-13)

## 搁置的小项

还开着，但次要、没有触发证据，第一节不再列。每条一行：是什么、为什么搁置、什么情况下再看。攒得多了再拆成单独的文件（owner 2026-11-10）。

- **分轨导出的 ASS 里给 cue 打上置信标**：标记已经在 `-annotated.csv` 与 `-diar-rows.json` 里，分轨 SRT 正文不加标。再看：有人要在字幕编辑器里按标复核。
- **第二模型否决的镜像**：否决判对了、留下的那行文本却是错的（2 段确证）。重解机制有效，但本语料上每 42 clip 只值 2 行，且现有 `lang_redecode` 强制的是多数派语言、方向正好相反。否决判据本身已结案（[`asr-stabilize.md`](asr-stabilize.md) 的 `第二模型否决` tag）。
- **Whisper 把紧贴上一行开始的行的起点放早**：第二级整期 28% 比参照早 0.3 s 以上（停顿之后开始的行 12%），被压住的行多半由此而来，没查。再看：要给被压住的行找位置来源时。
- **解码参数与切块的自动搜索**：VAD、切块、重解的参数凭经验调，可以在维护者的评测开发集上自动扫温度回退、`no_speech` 门限、`condition_on_previous_text`、切块长度与静音，按同一套验收。没有证据说现值差；默认值改不改由 owner 定。
- **非日语素材上的套话与字幕样式幻觉**（2026-10-24，各只在一份素材上见过）：stabilize 的整句清理只认日语那一句结束语，中文的「字幕志愿者 某某」漏网过一次；BBC 风格的英文素材上 Whisper 输出 `LAUGHTER` / `MUSIC` 音效标注（20 处，2 处粘在台词上）；台湾综艺上有 25 行带 `- ` 前缀、没有对应语音的行。再看：同类再出现。
- **两个阈值的候选触发面**（`transcribe.py`）：`MIN_DROP_TIME_RECALL = 5.0` 与覆盖率救援的 2 s 绝对容差（`ASR_COVERAGE_TOLERANCE_SEC`，它把有效门槛做成了组长的函数），只有事后对剩余缺口的分析（[`bench-baselines.md`「二十七、两个阈值的事后候选分析」](bench-baselines.md#二十七两个阈值的事后候选分析短缺口值得进一步复核)），不能证明非劣效。要动先做实际 recall 调用前的快照、同轨配对、人工内容核验与耗时测量，动阈值由 owner 定。
- **Qwen 裁判的生命周期收进一个对象**（owner 2026-09-25：不单独立项，下次实质改动裁判的放置或设备选择时顺手做，例如 A 卡、I 卡的裁判分支）：`run_vad_asr` 今天用三个可空局部变量（`redecode_referee`、`tail_referee`、`referee_warm`）手工交接同一个裁判。收成每次 run 一个的对象，这套交接第一次能用假裁判做行为测试，替掉 `test/test_asr_context.py` 里用 `inspect.getsource` 查接线的守卫。做时：[`asr-align.md`「referee 设备（五个问题，不是一个）」](asr-align.md#referee-设备五个问题不是一个)给三个调用点各自的策略，要原样搬进去；预热仍与 Whisper 解码并行；`test/test_function_size.py` 的 docstring 反对横切 `run_vad_asr`，按概念切要回应它。A 类改动，但在高危语音区。
- **本地 MOSS 的默认 prompt 译成各语言**（owner 2026-11-05 的想法）：现在只有 `en` 用英文 prompt，其余用上游的中文 prompt；中文 prompt 也能转日语，只验过中英日。
- **本地 MOSS 剪掉 Silero 判为长段无人声的部分**（片头 PV、休息段）：送进模型前剪掉、时间映射回去，K/V 与 prefill 都变短。B 类，要量质量。
- **本地链路与云端深度融合**（owner 2026-10-30：不是简单地把本地的文字换成云端的，而是更深入的融合；暂且不做，只是一个想法）。最浅的一种是生产分句 + mai2 文字替换：切点与时间不动，6 集里 5 集 CER 比生产低约 2 个点（相对约 −20%，少的是汉字错字），对谈密集的节目里还写下了生产漏掉的附和、抢话；但会在静音里幻觉（一集的 28 秒静音里写了 26 秒印尼语），替换前要按生产的 VAD 与能量滤掉静音里的字；代价是云端依赖、约 $0.10/h。
- **本地链路里的重叠处按人分别转写**：云端任务已经按人各自成行、交付时重叠，本地链路仍只有一路 Whisper，重叠处另一人的话只能丢。MOSS 重叠段的文字召回只在中文远场会议上测过，融合之前要先有日语带文字的重叠真值。

## 相关

- 维护者的本地计划（**不随公开快照发布**）：各项的逐项判据、验收与证据出处（状态格是注明日期的历史，现行状态看本文）。
- [`decisions.md`](decisions.md)：speech、评测与跨层已经定了的事。
- [`bench-discipline.md`](bench-discipline.md)：量之前先读；[`bench-baselines.md`](bench-baselines.md)：量到了什么。
- [`llm_followups.md`](llm_followups.md)：LLM 侧的同类入口。⚠ 两边的待办**不要混在一起排期**：这边的瓶颈是素材与显卡时间，那边是额度与 prompt 迭代。
