# vad-asr

`python -m finesub.speech.recognition.cli.vad_asr` 是生成 `*-aligned.json` 的组合阶段：先运行 `python -m finesub.speech.preprocessing.energy`，再运行 `python -m finesub.speech.recognition.cli.align`，
并把 VAD 能量聚合到最终 ASR segment。实现位于
`src/finesub/speech/recognition/vad_asr_stage.py`，薄 CLI 入口位于
`src/finesub/speech/recognition/cli/vad_asr.py`。

流式 VAD 检测独立在 `src/finesub/speech/preprocessing/vad.py`；模型的加载、
生命周期与 patched CT2 适配层在
`src/finesub/speech/recognition/fw_refine_backend.py`。recognition stage
只负责把两者与识别、分段及 aligned JSON 产物编排起来。

## CLI

```powershell
python -m finesub.speech.recognition.cli.vad_asr out/input/input-vocal.ogg \
  --output out/input/input-aligned.json \
  --model large-v3-turbo \
  --language ja \
  --gpu-tier standard
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input` | 必填 | vocal audio |
| `--output` | `<input>-aligned.json` | aligned JSON 路径 |
| `--model` | `large-v3-turbo` | Whisper 模型 |
| `--device` | `cuda` | 无 CUDA 时告警并回退 CPU |
| `--gpu-tier` | `auto` | 选 `entry`/`standard`/`high` 资源档（`auto` 读驱动定档）并写入 metadata |
| `--language` | 自动检测 | Whisper 语言覆盖 |
| `--gap` | `0.3` 秒 | ASR 合批组尾静音时长（inter-interval 静音为自适应，不受此参数控制） |
| `--split-length-scale` | `1.0`（或 `config.toml` 的 `[segmentation] length_scale`） | 分句器的长度目标缩放 γ ∈ [0.6, 1.6]，只缩放「多长算长」的上墙与 DP 剪枝界，下墙不动；<1 = 字幕更短、刀更多。三层优先级（代码默认 < config.toml < 本参数）在 stage 入口解析，越界立刻报错并指出值的来源；生效值写入 aligned metadata `asr_align.segment_split.length_scale`。设计与标定要求见 [segmentation-split.md](segmentation-split.md#长度缩放旋钮-length_scale唯一面向用户的分句参数)；`python -m finesub.pipeline` 同名 flag 透传 |
| `--asr-context` | `off` | 告诉 Qwen 裁判多少知识库内容（P9）：`off` 不注入 / `terms` **只给名字**——带 label 的条目与术语行的前三列（源语言·中文定名·别名），**丢掉一句话描述** / `full` 与纠错层看到的同投影。选词复用纠错层已有的检索（条目级 key+alias 匹配 + 小词条匹配），输入是 **`extra-info` 整块**——里面已经含自动抓取的视频标题，外加用户自己写的备注（用户点名的 subject 至少和标题一样有用）。⚠ **默认 off**：它改变第二个模型听到的东西。测量已经做了（`bench-baselines.md` 13.5–13.7，生产口径 n=59：`off` 3.4% → `terms` 30.5%，误插 0/59；`full` 28.8% 与 `terms` 分不出高下、只是贵 3.6 倍），但那些数**量的是与纠错器一致、不是准确率**，而且它改动了 49/59 条输出（更近 29 / 更远 8），翻默认还差**逐段听审**（`crispasr-followups.md` P9）。⚠ **层次**：`speech` 不得 import `llm`，所以选词发生在 `pipeline.py`，跨边界的是一个**字符串**——识别侧永远不知道名字从哪来。命中明细进 run metadata 的 `asr_context` |
| `--lang-redecode` | `auto` | 解码循环内的语言票翻转重解（auto=装了 transformers 5.x 就跑 / on=必须 / off=跳过；仅 auto 语言 run 生效）：group 语言票与近期众数不符时用 Qwen referee 按 VAD interval 逐一重认取证，以众数语言强制重跑 `align_group`，证据同意才替换。首版不覆盖「语言票仍正确、但文本崩成外语」的形态；阈值的真外语负例标定仍未完成。行为、取舍、实测与待标定见 [asr-align.md](asr-align.md)「语言票翻转重解」。开关进 checkpoint key；事件写入 aligned metadata `asr_align.lang_redecode`；`python -m finesub.pipeline` 同名 flag 透传。**`on` 比 `auto` 多买一件事**：全局语言审计（抽 8 段交 referee 重认、比时长加权众数，不一致只报警），因为一轮约 16.9 s 所以不进默认；`on` 在 `--language X` 下也不再空转，转为 `audit-only`（专抓「用户强制错了语言」，且**不进 checkpoint key**——它只读不改解码）。四格模式的唯一真相是 `lang_audit.resolve_mode()`，见 [asr-align.md](asr-align.md)「全局语言审计」 |
| `--vad-prefix` | `<output 去掉 -aligned.json>-vad.json` | VAD 阶段产物的路径（见下「VAD 阶段产物」）。匹配就复用、不匹配就重算并覆写 |
| `--vad-silero-assist` | **开**（`--no-vad-silero-assist` 关闭；`config.toml` 的 `[vad] silero_assist` 可覆盖） | 两信号后置组合（energy AND silero）：(1) voicing 门控 cap——floor 压至滚动最小锚+10dB,仅在 silero voicing（右向膨胀 0.3s,不向前）处生效,解禁被 creep 压掉的响句,silero 失灵=回落原行为,只增不减;(2) ghost-drop——silero peak<0.3 且峰值≤0dB 且 ≤12s 的区间整段丢弃;(3) 无声 span carve——区间内无证据（无 voicing、<0dB）的前缀/尾部/桥接修剪切分;(4) 接缝恢复——被合并吞掉的基础检测 gap 按原边界还原,除非缝内有 ≥-5dB 的捞回内容。概率搭车在 energy 的流式 block 上算（见下「资源与失败行为」）,60min 素材约 +3s(CPU)/+1s(CUDA)。**默认开**：管线一律先过分离器，这个后置组合正是为那种残留噪声的人声存在的;**干净的、未经分离的源音频建议 `--no-vad-silero-assist`**（打包扰动白付）。解析链在 `vad_asr_stage.resolve_vad_silero_assist`：flag > `[vad] silero_assist` > 代码默认。标定与验收：FINDINGS 附录 V2/W/Z 及后续。统计写入 aligned metadata `vad.silero_assist`;`python -m finesub.pipeline` 同名 flag 透传 |

生产中通常由 `python -m finesub.pipeline --stage aligned`（或更下游 stage）调用 `run_vad_asr()`，随后由
[`python -m finesub.speech.postprocessing.stabilization`](asr-stabilize.md) 从 aligned 生成 stable；stage 间直接调用函数，不使用 subprocess。
**ASR 固定单 worker**：2026-08-02 移除了单文件分片与 `--wt-workers` 开关，GPU profile 现在只
决定人声分离的实例数。理由与回溯点见 [`wt-parallelism.md`](wt-parallelism.md)。

`fw-refine` 是唯一 backend：正常 greedy 和 beam=5 timestamp span 都在同一次 CT2 decode 中完成
WT-compatible word refine。非单温度/多 hypothesis 等非主契约，或 compact trace 无法与最终 segments
核对时，才退回 faster-whisper teacher-force alignment；该 fallback 固定已有 tokens，不重新搜索文本。
构造模型时会校验 patched CT2 API，缺少扩展即失败，不静默切换算法。

## VAD 阶段产物

VAD 前缀（能量检测 → 可选 silero assist → `normalize_vad_segments`）是**自己的阶段**，产物
两个文件：

| 文件 | 内容 |
| --- | --- |
| `<stem>-vad.json` | schema、身份、`audio_duration`、`timing`、`vad_meta`、`raw_segments`、`segments`，以及帧级轨的 `hop_sec`/`frame_sec`/`energy_mode` 与 npz 文件名 |
| `<stem>-vad-energy.npz` | `energy_db`（必有）与 `frame_dbfs`（有则存），原 dtype |

**为什么拆**：`-aligned.json` 原本是一个大颗粒产物，把 VAD 前缀和 Whisper 绑死。想改
`--split-length-scale`、换 stabilize profile、或重跑 qwen 复核，都得删掉 aligned，于是那段
产出完全相同 interval 的 CPU 计算又跑一遍。拆开后「存在即跳过」的粒度对上了实际的重跑边界。

**帧级轨为什么在 npz 而不在 JSON**：几十万帧是 sidecar 不是文档——和 `build_vad_timeline`
把帧级数据挡在 aligned JSON 之外是同一条线。写入顺序是**先 npz 后 JSON**：JSON 才是存在性
判据，它不能指向一个还不存在的 sidecar。

**什么时候作废**（`read_vad_prefix` 返回 `None`，调用方重算并覆写，不抛异常——在按存在性
跳过的树里，过期产物是常态不是错误）：

- schema 不是当前版本；
- 音频身份不符：文件名 + 字节数 + mtime。取的是**调用方点名的那个输入**，不是 readers 实际
  读的解码副本——与 ASR checkpoint 同一条规则；解码出来的临时文件成功后就删了，拿它当键会让
  下一次必然失配。也**不是音频摘要**：人声轨几百 MB，为省一次 VAD 去哈希它并不划算，而重跑
  分离器会同时改掉大小与 mtime；
- JSON 读不动，或 npz 不在了。

⚠ **`--vad-silero-assist` 不在这张表里，这是刻意的。** 它改变的确实是 interval 本身，
但那不是作废的判据：不匹配既不会让续跑报错、也不会让数据拿不到，所以它记在产物的
`provenance` 块里，**不匹配时只打 warning，前缀照常复用**（`README_DEV.md` →
「复用的依据是任务身份」）。要全量按新参数重算，新建任务而不是翻开关。

复用与否记在 aligned metadata 的 `asr_align.vad_prefix`（`path` + `reused`）。**时间**沿用写入
时那次的读数（`vad_sec` 等如实反映 VAD 跑的那一次），因此复用时 `total_sec` 会小于各分项之和。

生产者只有一个：`run_vad_prefix()`。resumable 路径和一次跑完的路径共用它，不存在「另写一份
做同样事情的函数」——那正是这类拆分最常见的失败方式（两份实现随后各自漂移）。
两个文件都在 `finesub_bootstrap/artifacts.py` 的 `REMOVABLE_SUFFIXES` 里：一次 CPU 过带就能
重建，跑完随其它中间产物一起删。

## 数据流

```text
normalized vocal audio
  -> streamed vad-energy（语音 interval + VadEnergyTrack + pause_hints）
  -> asr-align（regroup / fallback / 覆盖率救援 / 语言票翻转重解（--lang-redecode，默认 auto）/
     recall / 尾词能量延长；detect_disfluencies 开）
  -> 词首修正（`src/finesub/speech/recognition/word_starts.py`：`[*]` 块四规则
     + VAD interval / pause_hint 锚点 clamp，docs/asr-align.md「词首修正」）
  -> 幽灵重复段清理 + 重叠收回 + 零时长段延长（`src/finesub/speech/recognition/segments.py`，见下）
  -> 全局 DP 分句（segment_split，docs/segmentation-split.md；可切可并）
  -> 按最终 segment 时间范围聚合 VAD weighted energy
  -> 第二模型校验证据（`speech/verification/qwen_referee.py`，见下；--qwen-verify）
  -> *-aligned.json
```

**第二模型校验证据**（2026-08-05，`--qwen-verify {auto,on,off}`，默认 auto=装了
`transformers` 5.x 就跑）：Whisper 池释放后用 Qwen3-ASR-0.6B（bf16 eager 峰值 ~1.8GB，
批内 padded 音频封顶 120 s 时 ~2.3GB，所有 GPU 档位可容纳；inline 已在同设备加载时复用，
设备不同时重新加载）。**2026-09-01 起裁判的三项效率改动**（实测与验收见
`bench-baselines.md` 二十一）：① 片段按长度排序后**成批解码**（≤16 条/批、批内
`条数×最长` ≤120 s，`plan_batches`），每步解码成本与 batch 无关，16 条生产片段 12.7 s→1.5 s；
② `standard` 及以上档位在 Whisper 解码期间用后台线程**预热**裁判（`referee_warm_device`，
放置判据与 inline 裁判同源；entry 档仍等池释放），一跑省 ~3 s 加载，产物里记
`timing.qwen_warm_sec` 与 `qwen_verify.warmed_under_decode`；③ 单次调用的 step-pacing 音频
≥150 s（该批大小本机已编译过；否则 ≥600 s）且 VRAM 预算 ≥3.5 GiB 时走**定形解码步**
（`qwen_decode.FixedShapeDecoder`：prefill 照旧 eager，解码步的 token/mask/位置/静态 cache 全部
定形、4-D mask 在步内由 cache_position 生成，一张 CUDA graph 覆盖所有长度；每步 65→9 ms，
全片复核 23.5 min 素材 31.6→16.3 s）。首编每进程每个 `(批大小, cache 长度)` 形状约 25 s（本机冷
90 s），`qwen.compile` 单独计时，磁盘上的已编译记录也按这对键记；解码步的位置号逐行从各自首个
真 token 数起（与 `generate` 的 `attention_mask.cumsum()` 同源），cache 槽位则全批共用；音频编码器逐条跑、静态 cache 按需取 512/1024，16 条一批的峰值 2.6 GiB、全片复核 3.3 GiB，
VRAM 门槛 3.5 GiB；`close()` 重置 Dynamo 代码缓存与 graph 池，实测回收到 0.01 GiB。transformers 自带的 static-cache 自动编译**不能用**：2-D mask 每
token 长一列、每个长度录一张 graph（21.6）。`FINESUB_REFEREE_ACCEL=0` 关闭。①③都不是 bit-exact（bf16 近平局翻转，文本 23/25 同），验收口径是 stabilize 读到的
**决策**一致率（否决/套话幽灵/缺口回收 31/31）。批大小 16、批内 padded 音频封顶 120 s。
继续只对三类嫌疑段重认：整段收尾套话、
CJK 主导 run 里的 Latin 段、stabilize 噪声腿将标记丢弃的段。三类都保持原有
段跨度 ±0.1s。`--lang-redecode` 的 inline referee 是独立取证路径，按 group 内 VAD interval
逐一读取，不改 `qwen_verify` 的嫌疑面或字段契约。尾部证据写进
段级 `qwen_verify: {text, language}`；≥3s 的未覆盖 VAD 区间同批重认，听到语音的记入
`metadata.asr_align.qwen_verify.qwen_gap_recoveries`（仅证据，不插入字幕流）。
`qwen_verify` 的 run 级字段是 `model` / `device` / `suspects` / `gaps_probed`——**`device`
记的是裁判实际跑在哪**（2026-09-02 补）。它不是用户给的：`lang_redecode.referee_device`
拿档位的空闲显存减去 Whisper 常驻算出来，所以少了这个字段，一次「悄悄退到 CPU」的运行
与一次拿到显卡的运行在产物上分不出来。逐阶段设备记录的全貌见
[stage-device-plan.md](plans/stage-device-plan.md) §2.6。
决策全部留给下游（stabilize 消费，见 docs/asr-stabilize.md）。67 clip 标定与
已知弱点（喊叫盲区）见 docs/wt-refine-handoff.md P1。
**失败不再带走整趟运行**（2026-09-04）：裁判只产证据、从不做决策，所以 `auto` 下它抛出的
任何异常都降级成一条 `qwen-verify-failed` warning（impact「少一层校验证据」，与
`qwen-verify-unavailable` 同一字段面）并继续；`on` 照旧原样抛出——那个档位是调用方在要这份
证据，静默返回没有它的结果才是错的答案。策略集中在
`vad_asr_stage.contained_verification`，独立成函数是为了能被真正测到（对
`run_vad_asr` 做源码字符串守卫分不出「正确的 except」与「把 `on` 也吞掉的 except」）。
⚠ **失败时不写 `align_meta.asr_align.qwen_verify`**：键缺失正是后续重跑愿意再试的条件，
写一条「失败了」会把一次瞬时网络故障固化成产物里的永久结论。
起因是一台连不上 hub 的机器：裁判加载时抛 `ConnectTimeout`，把**已经跑完并付过账**的整趟
对齐一起丢掉，traceback 里写的却是 httpx。

**安装**：包含在 `[asr]` 内（`transformers>=5.13,<6`，pip 增量 ~100MB；模型
Qwen3-ASR-0.6B-hf 首次运行时下载至 HF 缓存 ~1.5GB，与分离器模型同模式）。并入
`[asr]` 而非可选 extra 的理由：`--qwen-verify` 默认启用，若依赖可选则同一条命令
在不同安装上产出不同 stable，破坏再现性。模型用 `-hf` 权重经原生 transformers
推理；曾先实现过 `qwen-asr` 包 + 非-hf 权重的路线，因其 pin 死 transformers、
拖 gradio/flask/nagisa 无用负载、且需要 pyproject 无法表达的 `--no-deps` 安装而
弃用——同权重，输出逐字 parity 已验证（见 pyproject 注释）。

幽灵重复段清理（2026-08-04 新增，`drop_ghost_duplicate_segments`）在零时长延长**之前**
执行，三个条件缺一不可才删除：

1. **跨度**：整段 ≤0.1s 且归一化文本 ≥2 字（≥20 字/秒，不可能的语速）；
2. **解码证据**：段上带 `zero_duration_chunk_tail` 或 `alignment_stack` 事件——decoder
   自己报告了 chunk 尾挤压。没有这一条时，时间被量化压扁的**真实急促复读**（连喊两声
   `おい!`、歌词复唱）也会满足跨度+重复条件；全产物扫描（170 份 JSON / 5 万段）实测
   wt 时代产物里正是这类形态构成了多数命中，事件门控把它们全部挡下；
3. **重复来源**：归一化文本是 ±3s 内某个**非幽灵**段文本的子串（幽灵之间不能互证）。

不满足任一条件的微跨度段照常保留、走后续零时长延长与既有异常阶梯。删除明细
（`index`/`start`/`end`/`text` 逐条一条记录，2026-08-31 起带 `end`，为的是能回到音频复核那一段）
写入 `metadata.asr_align.ghost_duplicate_segments_dropped` 并逐条输出
Warning，产物内可审计。验证记录见 docs/wt-refine-validation.md「已纳入生产」。

零时长 segment（映射单调钳位塌缩或 whisper 自身的零时长词）会被延长
`0.01` 秒——下游消费者（`finesub.subtitles.rendering`、LLM 层入口）都会静默过滤 `end <= start`
的条目，不延长其文本会在所有路径中丢失。延长允许挤占后一段：被挤占段的
起点（及受影响词的起点）后延，连锁情形按时间顺序依次解决。

若 ASR/rescue 产出“segment 有文本但 `words` 为空”，分句入口会用整段文本和
segment 的 start/end 合成一条 word，并标记 `synthetic_from_segment: true`。它只用于
保证依赖 word 时间戳的后处理不丢文本，不表示真实 forced alignment；该段因此不会被内部切分。
发生数量写入 `metadata.asr_align.segment_split.synthetic_word_segments` 并输出 warning，
便于确认生产数据中是否真实出现。

### 已知缺陷：`*-aligned.json` 里的 segment 会时间重叠

`*-aligned.json` **不保证 segment 互不重叠**。11 个 clip 的测试床上共 49 处，
全部是词级重叠（段字段忠实跟随词，`段end − 末词end = +0.00`），两种形态：

- **零宽感叹词嵌在长 segment 里**（`[405.3, 405.3] ん`）——`finesub.subtitles.rendering` 的
  `end <= start` 过滤本来就会丢掉它们；
- **幻觉长段吞掉真台词**（`[47.5, 76.6] おぉぉぉぉぉ` 里裹着 `[54.3, 56.1]` 的真台词）——
  这类**能活到成品**：`asr_stabilize` profile 0 之后仍剩 43 处，最大重叠 27 s。

根因未定位（`extend_last_word_end_with_energy` 的 `next_word_start` 只取**同一 VAD interval
内**的下一个 ASR 段，是嫌疑之一，但只解释得了 ≤1.0 s 的那 29 处；另外 20 处最大到 27.8 s，
量级远超该函数的上限，来自别处）。`src/finesub/speech/recognition/transcribe.py` 是高风险核心，
未在此改动。

**不变式改由 `finesub.subtitles.rendering.resolve_overlaps` 兜底**：
SRT 要求 cue 有序不重叠，
渲染时截断**较早**那条 cue 的 end（而不是后移较晚那条的 start——后者会把真台词推过自己的
终点直接删掉，而元凶恰恰是左边那条幻觉长段）；两条同起点时改为后移较晚那条。
文字永不改动，只缩短显示时长，且缩短量就是重叠量。发生时向 stderr 打 `Warning:`
——不变式在这里恢复，但成因在上游，不要当成已修。

能量在所有 ASR 边界处理完成后计算，不会把一个 VAD interval 的单值复制给其下多个 aligned
segment。聚合公式为：

```text
10 * log10(
  sum(overlap_seconds * 10^(frame_db / 10))
  / sum(overlap_seconds)
)
```

## Aligned JSON

顶层三块：`segments`、`vad_timeline`、`metadata`。

**`vad_timeline`（2026-08-08 新增）= VAD「看到了什么」，与 `metadata`「怎么跑的」分开**：

```json
{
  "intervals": [{"start": 1.23, "end": 4.56}],
  "pause_hints": {"scorer": [7.81], "padding": []}
}
```

- `intervals`：归一化后的语音区间，**正是分句器打分时用的那把尺**（silero assist 之后）。
  落盘的动机是让「不重跑 ASR 就换 `length_scale` 重切」成为可能，也让审计不必反推 VAD 几何。
- `pause_hints` 按来源分开，两者语义不同：`scorer` 是攒够静音证据却没跨过 interval 阈值的
  候选（区间表达不了的停顿结构）；`padding` 是被负 padding 吃掉的 raw gap。
  **`padding` 恒空是当前的预期状态**——加权计数下最短可认证静音约 200ms，而 shrink 只杀
  <190ms（40+140+min-keep）的 gap，两者不相交；它变成非空正是改动 `NEGATIVE_PAD_RIGHT_MS`
  的人需要看见的信号。两者对词首 clamp 一视同仁（`clamp_hint`），只有产物里分家。
- **帧级轨不放这里**：量级 10³ 进 JSON，10⁵（能量轨）以上走边车。
- stable JSON 由 `copy.deepcopy` 原样继承该字段（未知顶层键一律保留）。

每个正常 segment 包含：

- `start` / `end` / `text` / `lang`
- `words[]` 及可选 word `confidence`；被 disfluency 块修正过起点的词带
  `disfluency_span: [块首, 块尾]` 与 `disfluency_action`
  （`merge`/`merge_short`/`delete`/`leading_*`，词级字段可穿过 DP 分句，
  见 docs/asr-align.md「词首修正」）；块长足以测量时另带 `disfluency_quiet_frac`
  （能量门的实测静音帧占比，<0.4 即被吸收）。**只有它能区分被吸收块的两类来源**——
  gold 上填充停顿中位 0.70、词首中位 0.00，而 `action` 对两者都是 `merge`；没有能量轨
  就再也算不回来。`[*]` 本身不进产物
- 可选 segment `confidence`、`no_speech_prob`
- 可选 `alignment_events[]`：`fw-refine` 默认收集的 path 观测与 disfluency 候选
  （`detect_disfluencies` 已默认开启）；时间已映射回原音轨，DP 分句只归属一条输出
  segment，stable 阶段原样保留，FineSub 当前不解析
- `vad_weighted_energy_db`：最终 segment 在 normalized vocal VAD 能量轨上的功率均值 dB

词首修正的动作计数写入 `metadata.asr_align.word_start_correction`
（`merge`/`delete`/`clamp_interval`/`clamp_hint` 等 → 次数）。

所有浮点输出按当前 `finesub.speech.recognition.transcribe.ROUND_DIGITS`
保留 3 位；例外是 `no_speech_prob`
（`ROUND_DIGITS_BY_KEY`）保留 6 位——它按 log 尺度消费，常见取值 1e-4~1e-2，3 位
会把小概率坍缩成 0.0。aligned schema 不含 VAD 置信度字段。

字段语义边界：

- segment `confidence` / `no_speech_prob` 是 Whisper 在**合批拼接时间轴**（interval +
  至多 0.7 秒保留 gap 音频 + 0.3 秒合成静音，见 asr-align 文档）上算出的，其 30 秒
  窗口是拼接产物；`no_speech_prob` 的分布与常规整轨 Whisper 用法系统性不同，
  按常规语义调阈值会失准。
- whisper 的分段会被全局 DP 重新划分（`docs/segmentation-split.md`）：一个 whisper 段可被切成
  多条，相邻 whisper 段也可被合成一条，**输出段与 whisper 段没有包含关系**。片段的
  `confidence` / `no_speech_prob` / `lang` 按各来源段贡献的**词数加权**继承（单来源时即
  原样继承；语义被稀释，逐片阈值判断需留意）；`vad_weighted_energy_db` 在分句后按片段
  自身边界计算，无此稀释。参数快照在 `metadata.asr_align.segment_split`。
- 每条 whisper 源段的首词带 `whisper_segment_start: true`；起点不是 whisper 边界的片段
  带段级 tag `mid_segment_start`。前者是还原原始 ASR 分段的唯一依据。
- `metadata.vad.segment_energy` 中 `"audio": "normalized_vocal"` 是生产管线的假设声明
  （pipeline 先做人声分离）；直接对未分离音频运行 `vad_asr` 时该声明不代表实际输入，
  此时能量值不适合与正常产物横向比较。

`metadata.vad.segment_energy` 记录：

```json
{
  "field": "vad_weighted_energy_db",
  "source": "adaptive_weighted_spectral_energy",
  "aggregation": "overlap_weighted_power_mean_db",
  "frame_ms": 25.0,
  "hop_ms": 10.0,
  "audio": "normalized_vocal"
}
```

若 energy mode 不是 `weighted`，或 segment 与能量轨没有有效重叠，则省略该字段，不写默认值。

## 哪些异常交给 Qwen 裁判，哪些不交

裁判（`--qwen-verify`）**不是全覆盖的**——它按段取证，每段一次推理，所以「交给谁」
是一条明确的取舍线，不是遗漏。这一节把线画出来，免得每次都要重读
`collect_suspect_indices`。

**会交给它的**——四类嫌疑段（`qwen_referee.collect_suspect_indices`）加 VAD 缺口
（`collect_gaps`，未被任何 segment 覆盖且 ≥ `GAP_MIN_SEC`）：

1. 整段是**套话形状**（幻觉套话族；英文那一支**只能**靠这个探针够到）；
2. CJK 主导的输出里的**拉丁串**（真英文 vs 翻译模式）；
3. **绝对电平可疑档**（`vad_level_tier == suspect`，`preprocessing/energy.py` 只在
   上游打标，这里才把标记变成推理——裁判关着就一分钱不花）；
4. 稳定化 profile 2 会打上「**高度疑似幻觉 / 高度疑似语气填充词**」的段。

另外两处也在用它，容易被忽略：`lang_redecode`（重解要裁判证据背书才采纳）与
`lang_audit`（确定式抽样交裁判重认语言）。

**不交给它的，以及为什么**：

| 异常 | 现状 | 判断 |
| --- | --- | --- |
| **绝对电平丢弃档**（−60/−70 dBFS） | 区间在**进解码器之前**就被丢掉，裁判永远看不到 | 设计如此。标定是单向压下去的（耳语语料一条带文本的区间都不丢再留 5 dB），42 份生产跑里丢掉 194 个区间、其中 8 个有文本，**这 8 个事后全被裁判验过、全是幻觉** |
| **`ghost-duplicate-dropped`** | 直接丢、只报一行 | ⚠ 曾被我记成「无证据删除」，**那是错的**：三条判据缺一不可——帧量化的不可能语速、**解码器自己报的挤压事件**（`zero_duration_chunk_tail`/`alignment_stack`）、文本被 ±3 s 内的**非 ghost** 段包含。实测见下 |
| `时间漂移` `TAG_TIME_DRIFT` | 只打标不探测 | 合理：纯时间异常，第二模型给不出证据 |
| `语言切换幻觉` `TAG_LANG_SWITCH_HALLUCINATION` | 不单独触发探测 | 合理：它按**拉丁字符比例**判，与上面第 2 类同族，基本已被覆盖 |
| **对齐坍塌** `alignment-collapsed` | 只报警不修复 | 合理（有可能的修法但故意不做，见 `align_sentinel` 的 docstring）。⚠ **从没量过生产里多久触发一次** |
| `timeline-out-of-order` / `srt-invalid` | 只报警 | 合理：结构问题不需要听音频 |
| 复读折叠 | 确定式文本折叠 | 合理 |
| `confidence` / `no_speech_prob` | 基本不进决策 | 合理：naive/efficient 两条路径间有 0.14 的系统偏差，不可比 |
| **分离器吃掉耳语的救援** | 判据已量清、**未接线** | 这一条**本来就该给裁判**——三条判据的第三条正是「第二模型在**原始**音频上认出词」。42 份生产跑 1113 个窗一次都不触发，所以不急（`bench-baselines.md` 17.13–17.18） |

### `ghost-duplicate` 的实测：4 次 / 56 跑，零内容损失

2026-08-31 扫了主 checkout 的 131 份 aligned 产物（56 份带
`ghost_duplicate_segments_dropped` 字段，其余早于该字段）：**4 份各丢 1 段，共 4 段。**

| 丢掉的 | 邻近幸存段是否仍含该文本 |
| --- | --- |
| `はぁ…` | ✅ 55.14–56.38 那条就是它 |
| `しー` | ✅ 427.87–428.40 |
| `ご視聴ありがとうございました` | ✅ 921.20–921.38（⚠ 邻居**自己也是幻觉**，但那条走套话族、**是交给裁判的**——分层正确） |
| `わー!` | ✅ 三条相邻段都含它 |

**4/4 的文本都在 ±3 s 内的幸存段里活着**，即规则自己的承诺（「删掉它不会丢掉语音」）
在生产上成立。触发率 4/56 跑，**不需要为它加裁判探测**。

⚠ 复现（离线只读，不需要 GPU）：扫 `out/**/*-aligned*.json` 的
`metadata.asr_align.ghost_duplicate_segments_dropped`，再对每条丢弃在同一产物里查
±`GHOST_SEGMENT_CONTEXT_SEC` 内是否有段包含其 `normalized_compact` 文本。

## 资源与失败行为

- VAD 在 CPU 上流式执行，仅保留小型整段能量轨。
- `--vad-silero-assist` 开启时，silero 概率**搭车**在 energy 的流式 block 上算
  （`SileroProbCollector` 实现 `energy.WaveformObserver`），不再二次解码/归一化，
  也不再常驻整段波形。详见 [`vad-energy.md`](vad-energy.md#waveformobserver-钩子)。
- Whisper 默认使用 CUDA；CPU 仅为回退路径。回退由 `speech/runtime/device.py` 的
  `resolve_device()` 统一判定，**两种情况都回退**：没有可用 CUDA，以及有卡但装好的 torch
  没有它的 kernel（老卡，`is_available()` 为 True 却在第一次真算时炸）。判定拿设备的计算能力
  比 `torch.cuda.get_arch_list()`，所以支持范围跟着 torch pin 走；用户可读的型号表在
  `README.md`。两种回退都往 stderr 打一条 `Warning:`，`--device cpu` 显式指定则不告警。
- CPU 回退按整机线程数并行，无需任何旋钮。这依赖 patched CT2 wheel 用 **oneDNN** 作 CPU GEMM
  后端：`4.8.1+wtrefine1.cu128` 用的 Ruy 会在模型析构时死锁（产物落盘后进程再也不返回，
  0.3.2 的现场故障）。**换 wheel 前先读**
  [`ct2-patches/README.md`](../tools/wt_refine_port/ct2-patches/README.md) 的后端对照表与最小验收
  ——`get_supported_compute_types("cpu")` 查不出这类问题。
- ASR 音频由 `AudioBlockLoader` 以 600 秒 core + 10 秒 pad 流式读取。
- **`--lang-redecode` 的 inline referee 与 Whisper 池同时存在**（判定必须在解码循环内做），
  设备按 ASR 模型的实测常驻显存选，档位余量放不下就落 CPU——**默认的 4GB 档就是 CPU**。
  模型 lazy 加载，首次触发才付；一旦加载就常驻到 ASR 结束（尾部 `--qwen-verify` 同设备时
  直接复用它，省一次加载）。CPU 上是 float32，实测 **+3.26 GiB 主机 RSS、3.05s per clip**，
  且每次触发要对该 group 的每个 VAD interval 各跑一次——`RAM_BUDGET_GB` 是 8 GiB，
  长素材上留意 `resource-budget` 告警。完整设备表与实测见
  [`asr-align.md`](asr-align.md)「语言票翻转重解」。`--lang-redecode off` 时这条完全不存在。
- 空 VAD 输出仍生成合法的 `{"segments": [], "metadata": ...}`。
- `metadata.asr_align.timing` 保留 loading/energy/noise/VAD、Whisper load、
  alignment 和 VAD-ASR total 秒数；task report 默认只展示 ASR total。
  开 `--vad-silero-assist` 时另有 `silero_probs_sec`（概率，已含在 `vad_sec` 内，
  单列出来避免被 VAD 总时长吞掉）与 `silero_assist_sec`（VAD 之后的判据部分）。
- 每次运行结束仍打印 VAD、Whisper 和资源峰值统计。

## 验证

```powershell
python -m pytest -q \
  test/test_vad_segment_energy.py \
  test/test_vad_streaming.py \
  test/test_vad_silero_ghost.py \
  test/test_vad_carve_hints.py \
  test/test_asr_and_text_utils.py \
  test/test_lang_redecode.py \
  test/test_qwen_verify.py \
  test/test_pipeline_refactor.py
# silero 概率与 WaveformObserver 搭车（需加载模型）
python -m pytest -q test/test_vad_silero_probs.py --run-heavy-resource
```

