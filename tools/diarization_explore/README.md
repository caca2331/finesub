# `diarization_explore/` —— Sortformer 探索期的复现脚本

2026-09-16 新一轮准确率实验：`accuracy_lab.py`（同输入 cache / 几何 / split 对照）、
`accuracy_trace.py`（选帧反馈因果干预）、`accuracy_models.py`（本地 community-1 与 WavLM）、
`accuracy_report.py`（配对汇总与融合）、`accuracy_identity.py`（惩罚过拆的 B-cubed）、
`accuracy_hybrid.py`（局部时间／全局身份协同）、`accuracy_export.py`（候选字幕副本导出）。
`accuracy_merge.py` 检验独立 WavLM 声纹合轨与共同发声禁并约束。
`accuracy_queue.py` 串行续跑留出，`accuracy_cross.py` 检查跨节目泛化；
`candidate_run.py` 将冻结的高上下文 Sortformer 或 community-1 配置用于任意音频和已有 stable 字幕，不读取参考标签。
`accuracy_finalize.py` 重算统一参考快照的指标，并归档紧凑结果与源文件指纹。
实验协议与结果在维护者的本地归档里（不随公开快照发布）；契约测试命令：
`python -m pytest -n 0 tools/diarization_explore/test_accuracy_lab.py`。
GPU 进程必须先安装 `vram_guard`，一个文件一个进程，模型推理串行；所有下载在工作区 `tmp/diarization-models/`。

显式运行替代模型（当前探索环境需要 `pyannote.audio==4.0.3`、可用 CUDA 和已有访问权的本地模型；
不自动下载、不调用 API、不重新识别文字，输出目录必须不存在）：

```powershell
python tools/diarization_explore/candidate_run.py out/refined-2609/hina-07/hina-07-vocal.ogg `
  --stable out/refined-2609/hina-07/hina-07-stable.json `
  --backend community --model-dir tmp/diarization-models/community-1 --out out/hina-07-community-review
```

保留 Sortformer 模型时，改用 `--backend sortformer-high --model-dir <三个 .ts 的目录>`，
并选择另一个新输出目录；参数固定为 chunk=340、right=40、FIFO=40、update=300。
该路径不需要 pyannote，仍有四个槽位上限。这两个候选都不是“每份文件必然更准”。

输出是保留原文与原时间的 stable / diar JSON、按轨着色 ASS 和分轨 SRT；轨号不代表已识别角色姓名。
这是显式实验入口；主流水线的默认后端是 `nemotron3`。为绕过本机未使用的可选依赖冲突，独立进程禁用
torchcodec / SpeechBrain / NeMo 适配器并用 soundfile 解码；这不修改安装环境，不应复制进长期服务进程。

第二轮（上下文、重叠、DiariZen）入口：

| 脚本 | 用途 |
| --- | --- |
| `accuracy_v2.py` | 七种长块／右上下文／cache 组合，`batch` 或 `summary` |
| `accuracy_diarizen.py` | 官方 DiariZen 独立进程适配，保存局部分割供组件组合复用 |
| `accuracy_recluster.py` | DiariZen 分割 + community-1 embedding / PLDA / VBx / 重建 |
| `accuracy_consensus.py`、`accuracy_vote.py` | 跨模型身份核实合轨、Sortformer 局部证据仲裁 |
| `accuracy_v2_fusion.py`、`accuracy_native.py` | 同家族三视图融合、原生边界对照 |
| `accuracy_v2_queue.py` | 每文件／模型新进程，串行跑开发或 Onetonton 扩展集 |
| `accuracy_overlap.py` | AliMeeting RTTM / CALLHOME 日语话轮的零 collar、包含重叠 DER 与全员召回 |
| `accuracy_history.py` | 重访 nepolabo 原来的 20 个人工标注窗口 |
| `accuracy_v2_report.py` | 汇总配对差值／bootstrap 区间，保存紧凑结果 |
| `accuracy_v2_export.py`、`candidate_diarizen.py` | 从缓存或任意音频生成新的 DiariZen 字幕预览 |

DiariZen 接入使用官方源码 commit `844f5555b0a98acd0931511fc641a8c5b8ba92c7`，
放在 `tmp/diarizen-source/`；仅将 `semver==3.1.0` 安装到 `tmp/diarizen-deps/`，
不要照旧 requirements 覆盖现有 torch / CUDA 环境。使用本轮 asr 环境即可；适配器在独立进程
选择官方 vendored pyannote，补充版本元数据，跳过未用训练模块，并以同一 soundfile 浮点波形代替旧解码接口。
local batch 固定 2，窗口 16 s，步长 1.6 s。权重与配置来自
`BUT-FIT/diarizen-wavlm-large-s80-md-v2`（放 `tmp/diarization-models/diarizen-v2/`），
embedding 来自 `pyannote/wespeaker-voxceleb-resnet34-LM`（放 `tmp/diarization-models/wespeaker/`）。
本轮固定的模型 revision 分别为 `027b3e221b9d81dce2be794084f1dbd3ba2da403` 与
`837717ddb9ff5507820346191109dc79c958d614`；更换源码或权重后属于新实验，不能沿用旧成绩。
源代码和模型文件哈希随结果保存；不自动联网。**DiariZen 权重是 CC-BY-NC 4.0，仅作非商业研究候选**，不随本项目打包。

```powershell
python tools/diarization_explore/candidate_diarizen.py out/refined-2609/hina-07/hina-07-vocal.ogg `
  --stable out/refined-2609/hina-07/hina-07-stable.json --out out/hina-07-diarizen-review
python tools/diarization_explore/accuracy_v2_queue.py --ids hina-01 --models diarizen dz_community
python tools/diarization_explore/accuracy_overlap.py R8001_M8004 community
python tools/diarization_explore/accuracy_vote.py --anchor diarizen
python tools/diarization_explore/accuracy_v2_report.py
python tools/diarization_explore/accuracy_v2_plot.py
python -m pytest -n 0 tools/diarization_explore/test_accuracy_lab.py tools/diarization_explore/test_accuracy_overlap.py
```

`candidate_diarizen.py --native-boundaries` 可保留模型原生时间边界，跳过 80 ms 栅格和平滑。
默认仍复现冻结主检验的路径；原生边界在第二轮被单独评分，是离线研究选项。

第二轮原始输出在 `tmp/diarization-accuracy-v2/`，公开的能力结论仍以 findings 为准；
完整协议在维护者的本地归档里。第一轮脚本与缓存不改写。
`accuracy_overlap_queue.py` 顺序跑公开语料，`--history` 改为 nepolabo；只能在上一 GPU 队列结束后启动。
公开语料评分是 full-audio UEM、零 collar、含重叠 DER；先合并同一说话人的重复区间，Hungarian
一对一标签映射仅用于评分。重叠全员召回统计真值重叠期间是否找全所有人，额外误报另看 DER/FA；
与 `pyannote.metrics` 的独立随机区间交叉核验已经纳入契约测试。

第三轮（DiariZen 4+ 与局部窗口，2026-09-17）使用 `tmp/diarization-dz3/` 独立缓存：

| 脚本 | 用途 |
| --- | --- |
| `accuracy_dz3.py` | 8/16 s、AHC/VBx、活动槽与重叠约束；保存局部预测和声纹；`--cpu-replay` 先逐位验证默认重建 |
| `accuracy_dz3_boundaries.py` | 冻结身份，只改变重建中值滤波；11 帧对照必须复现默认 |
| `accuracy_dz3_multiscale.py` | 8/16 s 声纹联合聚类与证据密度补偿；两个时间尺度分别重建 |
| `accuracy_dz3_meetings.py` | 两场新四人会议及六份 5–13 人 VoxConverse，连续时间 DER 和多人全员召回 |
| `accuracy_dz3_validate.py` | 读取冻结清单后串行执行扩展验证；失败即停，可续跑；不负责调参 |
| `accuracy_dz3_report.py`、`accuracy_dz3_plot.py` | 紧凑结果和独立报告图 |
| `accuracy_dz3_diagnose.py` | 一对一身份映射下逐人召回；另标记允许复用多数人轨的时域覆盖，不能当作身份恢复 |

任意音频预览可显式加 `candidate_diarizen.py --window-seconds 8`；默认16 s。
8 s 的单位时间窗口数约翻倍，并改变声纹片段长度；在六份 VoxConverse 上改善、两场四人会议上退步，
19份字幕冻结验证已完成：13份5–8人F1均值提升1.26 pp，但95%区间跨零；六份四人F1下降0.75 pp，
不能作为通用推荐。可与 `--native-boundaries` 组合，预览记录窗长与边界模式，真实60秒音频导出已验证。
完整结果与指纹见 `results/diarizen-v3-20260917.json`；协议与报告在维护者的本地归档里。

缓存复放必须执行官方 `fix_reproducibility`：完整分段前向会关闭 TF32，直接跳过分段却不恢复该设置，
会改变后续声纹数值。七份默认复放已逐位核验。原始大张量不入库，拆 worktree 前须备份。
完整协议与结果同在那份本地归档；生产入口和默认参数没有改变。

| 脚本 | 用途 | 对应结论 |
| --- | --- | --- |
| `dz_probe.py` | **DiariZen 有没有 Sortformer 那两个失败模式**（`owners` 轨主人在片中换不换人 / `micro` 输入 ×(1+1e-7) 与 PCM16 网格会不会重组时间线 / `stress` #14 / `replay` 逐位复放自检）。⚠ `owners` 把生产的 Sortformer 臂**走同一个函数**打印出来，所以两栏由构造可比——它那一栏必须复现 `relabel_eval.py --owners`。⚠ 这条规则对**轨多的模型偏向「没换人」**（更纯、更容易过 60%，也更容易因不足 30 s 而整窗退出），所以合格窗数要和结论一起读 | 维护者本地归档的 DiariZen 探针记录 |

2026-08-22 那轮 NeMo Sortformer 实测的原始脚本，2026-09-05 从
`explore/speaker-clustering` 分支的 `tmp/sortformer/`（**从未入库**）搬出来。
结论在 [`docs/diarization-findings.md`](../../docs/diarization-findings.md)（精炼版）；
2026-08 的完整评估与 0.6.0 的实施计划在维护者的本地笔记里（不随公开快照发布），重跑要知道的坑已写进下面「素材在哪」一节。
下表「对应结论」列里的 §5a–§5r 是那些本地实验记录与归档计划的节号，节号都不变。

⚠ **这些是一次性实验，不是维护中的工具**（`tools/README.md` 的第二类）。搬过来的理由只有
一个：那两份 findings 里的每个数字都由这里的某个脚本产出，脚本没了就只剩结论没有账。
**不要在它们上面建东西**——0.6.0 的生产实现是从零写的 `speech/diarization/`。

**唯一的例外是 `calibrate.py`**，它是**要被反复跑的**：阈值（P11 的滞回与最短时长、
assign 的三个占比、P14 的共活率）默认值一动就要重跑它。它吃 `audio.wav=reference.rttm`
成对的参数——RTTM 是 VoxConverse / AliMeeting / AMI / DIHARD 共同的标注格式——回答三个问题：
我们的平滑有没有**凭空造出重叠**（真值 vs 我们报的段级重叠率）、`second_share` 该落在哪
（对真值扫精确率/召回率，而不是拍脑袋）、以及共活率能不能跨多份素材分开单人与多人。

```powershell
conda run -n asr python tools/diarization_explore/calibrate.py `
    data/diarization/public/<x>.wav=data/diarization/public/<x>.rttm `
    --artifacts <三个 .ts 所在目录>
```

## 哪个脚本产出了哪个数字

| 脚本 | 产出 | 落在旧版 findings（2026-09-21 改写前的版本，在维护者的本地归档里）的哪一节 |
| --- | --- | --- |
| `run_diarize.py` | 跑一次 diarization，出段与帧级张量 | 全部的输入 |
| `analyze_output.py` | 轨迹占比/时长统计 | §3.1（nepolabo 4 轨迹、kaguya60 被强分 2 轨迹） |
| `spotcheck_compare.py` | 与人工抽查表对照 | §3.1 的 **70%** |
| `vad_experiment.py` · `vad_metrics.py` | 帧级活动 → 区间，与项目 stable 比 | §3.2 的召回 98.2% / 精度 87.5% / 重叠率 20.1% |
| `vad_diff_asr.py` | 差集的 ASR + RMS 交叉验证 | §3.2 的「漏掉的 52s 是静音边缘、补回的 413s 是人声碎片」 |
| `overlap_asr_experiment.py` | 按说话人掩码投送 whisper | §3.3 的重叠分离概念实证 |
| `speaker_id_experiment.py` | 段 → ECAPA → 组内/组间距离 | §3.4 的 gap 0.036（同管道）/ 0.304（跨管道） |
| `anchor_classify_test.py` · `anchor_classify_n.py` | 锚点档案分类（1 个 / N 个锚点） | §3.5 的 9/9 保真 + 0/8 误收；§4 场景 3 的 8/8 |
| `window_propagation.py` | 固定/动态滑窗与跨窗身份传递 | §3.6 的 44–65% |

**2026-09-10 那一轮**（**不需要 nemo**：走生产的 `speech/diarization/` 与当时的角色层 `llm/speaker_roles`，数字落在
计划而不是 findings）。路径由环境变量给：`DIAR_EXPLORE_DATA`（默认 `<仓库>/data/diarization`，
在主 checkout 上）与 `DIAR_EXPLORE_WORK`（默认 `<仓库>/tmp/diarization_explore`；三个 `.ts` 放
`export_v2/`，三个 16 kHz wav 放 `vocal-16k/`）。`PYTHONPATH` 要带 `src` 与本目录。

角色层（`--speaker-roles`）2026-10-09 被 LLM 说话人裁判取代、删掉，它的实验脚本（`roles_*.py` 与它们的 prompt、`voice_check_system.md`）一并删掉，原件用 `git show c4b2d1c0:tools/diarization_explore/<文件名>` 取回；结论在 findings「角色命名：到顶」。

| 脚本 | 产出 | 落在计划的哪一节 |
| --- | --- | --- |
| `names_vs_spotcheck.py` | 模型逐轨给的名字对人工抽查表 | §5g.2 的「轨迹只有一半纯」：9/18 天花板、8/18 |
| `extract_textgrids.py` | 从 `Eval_Ali.tar.gz` 流式抽出三场的 TextGrid（带文字） | §5h 的真值 |
| `s4_headroom.py` | 重叠句 vs 干净句的字二元组召回 | §5h.1 的 H = +49.1 pp |
| `s4_projection.py` | 基线 / 假投送 / 投送三臂（`--decode` 一次 GPU，`--score` 可反复） | §5h.3 的 +7.5 pp 与虚增 0.343 → 0.475 |
| `s4_incremental.py` | 投送与假投送**新增**内容的精度（事后读法） | §5h.3 末尾的 17.4% / 34.0% |
| `s4_repair.py` + `overlap_repair_{system,user}.md` | 投送出候选、多模态模型核听（`--decode-decoys` 一次 GPU，`--ask`，`--score`；打分复用 `s4_llm`） | §5h.6 |
| `over4_confidence.py` | onepass 归错帧的 `p_max` / margin 能否挑出来（AUC 与阈值表） | §5i.2 |
| `over4_anchor.py` | 多遍算法：锚定滑窗 6 种与递归拆分 4 种（`--dev` 在开发集迭代，`--test VARIANT` 只在冻结后跑一次留出集，`--list-test` 看测试集） | §5i.4 开发集表、§5i.5 的 +8.6 pp |
| `over4_v2.py` | 拆分 v2：生产路径行级指标、三扰动噪声带、bootstrap（`--dev` / `--final CONFIG` / `--list-final`） | §5k.1–5k.2 |
| `over4_input.py` | 三种输入（原始 / 分离后 ogg / 分离后 flac）上重跑 onepass、拆分与高质量几何（`--only` 出部分结果） | §5m.1 |
| `over4_verify.py` | §5n 逐对核验：`--measure` 核验器本身的准确率、`--effect` 开发集上拆分 vs 拆分 + 否决 | §5n.1 |
| `over4_final.py` | §5n.2 终判：VoxConverse test（≤12 min 的 57 + 56 份）、raw 与生产 ogg、三臂 | §5n.3 |
| `s4_breakdown.py` | §5h.12 新口径：逐字「对 / 听错 / 漏」、归错人、假设侧拆分（`--spot-a` / `--spot-b` 出抽查单，`--score` 全量，`--posthoc` 未预注册的口水词与按句视角，`--clean-miss` 干净句漏在哪）；只读已有解码。原始解码与抽查单归档在 `data/diarization/results/s4-alimeeting-20260910/` | §5h.13 |
| `s4_ami.py` | §5h.20 把 AMI 六场会议（英语，CC BY 4.0）放到 S4 脚本认的位置：下载、逐词 XML → TextGrid、收集管线产物与重叠区；之后各 S4 脚本加 `DIAR_S4_CORPUS=ami` 即在 AMI 上跑（英语按词计） | §5h.20 |
| `s4_timecut.py` | 按时间切：各轨在「区间 ±1 s」里响的帧单独重转（硬切 T1、硬切 + 远处锚 T2、独占切 + 重叠多标 T6，Whisper 与 Qwen 各一份）、整窗不切 T0，以及按词时间戳重分 T4（`--whisper` / `--qwen` / `--score`，对照是同一窗口的生产现状）。⚠ 它的 `--score` 是 §5h.17 那个会量进分行方式的打分器，只留作记录 | §5h.17 |
| `s4_score2.py` + `test_s4_score2.py` | 复审四轮的打分器 v2：每窗全部行拼成一串对齐一次（内容与诱饵与分行无关）、重复另算、归属另问同标签的行；报正确召回、保留率、两种口径的归错人、编辑量与按 120 s 片段的 bootstrap 区间（`--score`，`DIAR_S4_CORPUS=ami` 切 AMI）。契约测试不在默认套件里：`python -m pytest tools/diarization_explore/test_s4_score2.py` | §5o.4 / 5o.7 |
| `s4_holdout.py` | 留出集：从 `Eval_Ali.tar.gz` 抽出没用过的 5 场 AliMeeting（8 路阵列用 `ffmpeg -ac 1 -ar 16000` 下混，与原三场逐位同法）、收集管线产物到 `ali-holdout/`（`--extract` / `--collect`；之后各 S4 脚本加 `DIAR_S4_CORPUS=ali-holdout`） | §5o.9 E3 |
| `s4_callhome.py` | E5 的日语素材：把 TalkBank CallHome 日语的 `.cha` 解析成 TextGrid、按转写范围截音频、双声道电话音混单声道升 16 kHz（`--prepare`；之后 `DIAR_S4_CORPUS=callhome-ja`）。⚠ 音频要 TalkBank 账号，由 owner 自己下载 | §5o.9 E5 |
| `refined_ass.py` · `refined_compare.py` | 带说话人标注的多人精修轴（`data/refined-multi/姐姐咚咚/`）：前者解析 ASS（**样式名 = 说话人**，译注单独一类）并测量轴本身（`describe`），后者按 §5p.2 的五条判据把我们的 diarization 与它对照（M1 按 §5p.5 改用起点差、原互相关只照报；M3 以完整行跨度为准、1 s 口径只附带；M4 另报去掉同一人两条轨之后的数；M6 数每个人工轮次里我们换了几次人，同一人两轨间的切换另计——切分位置的第一个自动量法，未预注册，§5q）。⚠ 文字是中文翻译 ⇒ **任何字级内容判据在这份素材上都没有意义**；结束时间被顶到下一行 ⇒ 只用起点与「起止完全相同」的那批重叠 | §5p |
| `s4_diar_probs.py` | Sortformer 每帧四个 sigmoid 的取出与缓存（`-diar-probs.npz`），用来给**已有**的段落标注说话人置信度：`segment_confidence` 标注 `-diar.json` 的帧级段，`stable_confidence` 标注管线自己的分句。⚠ **只标注、不替换**——`-diar.json` 是不可复现产物 | §5o.11 探针 3 |
| `s4_freeform.py` + `freeform_system.md` | 探针：只给 ~2 分钟音频，不给候选 / 轨 / 人数，让 agy 自己按说话人转写并给时间（`--ask SESSION INDEX...` / `--score SESSION`）。有真值的语料对真值报内容召回与按说话人算对，没真值的（nepolabo）与生产互比信息量。四个变体：自主 / `--count N` 告知人数 / `--timeline 1\|2` 给空轴（帧级或分句表）/ `--timeline 3` 两趟综合（`freeform_{timeline,timeline2,repair}_system.md`）；`--audit` 只看重叠窗口、`--retention [--windows] [--slack S]` 算「保留生产写对的字」（slack 默认 0、两侧同样放宽；§5o.12 原来只给 agy 加了 2 s，§5q）。⚠ 粗尺子（Ratcliff 共同字数），只报数不判定 | §5o.11 / 5o.12 |
| `s4_spot2.py` | 抽查单 C / D：打分器 v2 自己对不对。固定种子取 30 个真区间窗口（每窗随机一个臂）与 20 个诱饵窗口，把参考句、该臂的输出行与打分器的逐字判定并排打印，供人逐字核（`--sheet C\|D`） | §5o.9 抽查 C / D |
| `s4_e4_page.py` + `e4_page.html` | E4 的 owner 盲评页：从 nepolabo 取 40 个重叠单元，切窗口音频与每条轨的单人参考音，四个臂（A0-W / T4 / T2-Q / T2-W）每单元独立打乱成甲乙丙丁（`--build` 出 `units.json` 与音频，`--page` 注入模板）。⚠ 盲标签到臂的对应只留在 `units.json`，不进页面；nepolabo 没有真值，所以走人判而不是打分器 | §5o.9 E4 |
| `resegment_{v2,v2t}_text_system.md` + `resegment_user_text.md` | 重分段的纯文本参照：只改「听」相关的句子，不传音频（`s4_resegment.py --variant v2-text / v2t-text`） | §5o.9 E1 |
| `track_mixing.py` | 每条轨在真值里有人说话的帧上：本人没在说而别人在说（**混入**）、同一人第二条及以后的轨（**拆轨**），按轨时长计。S4 语料加 `DIAR_S4_CORPUS`，另有 `vox`（onepass 缓存）与 `refined`（姐姐咚咚，完整行跨度）。给真值定价的三个开关（只对 S4 语料）：`--snap` 把结束时间顶到下一句（模拟精修轴，混入只变 ±1 pp）、`--chop 2.1 --claim 1` 切成 2.1 s 一行且只记起点后 1 s（读出 18–26%，§5p.5 再更正的依据）。每次都另报三样：只算我们仅一条轨在响的帧的混入（排掉与主句重叠的漏写插话；被顶满的空隙里没写的话排不掉，所以不是严格下界）、我们仅一条轨在响而真值没人说话的帧占比（空隙没被顶满处漏写的旁证），以及多出来的轨按纯度 80% 分成「一人的复制」与「几人共用的短槽」；`vox` 另按 2–4 / ≥ 5 人分组（轨多于人时「有人被拆」是按份额定主人的必然，不是观察） | §5q |
| `speaker_reseg.py` | 按词时间戳把多人行拆成「一人一行、插话单独一行」（T4 的字幕形态；抖动与短插话规则、`hold_to_a_boundary`）。纠错前更碎、纠错后与不拆差不多，**判不采用** | §5r |
| `speaker_relabel.py` | 只改 spk 列：多标签行按词数 / 首选 / 覆盖占比重排并删掉次要标签（`--mode words\|lead\|coverage`、`--csv` 经 `_speaker_cells` 重写 `-annotated.csv`）；只重排、不加标签 | §5r A |
| `redecode_candidates.py` | 分轨重转再纠错：重叠窗口按轨时间切 + 远处锚、Qwen 重解（`--plan` / `--decode`，qwen-asr 环境），写 `<stem>-overlap-candidates.json`；`--insert` 把候选作为 `candidate: true` 段插进 stable（B1）。B2 用生产里的 `FINESUB_OVERLAP_CANDIDATES` 开关读同一份文件 | §5r B |
| `preview_ass.py` | 给 owner 预览的 ASS：`-annotated.csv` 的译文 + 纠错后日文，行首 `s{轨}·名字`；`--stable-only` 只渲染 stable | §5r |
| `relabel_eval.py` | 改标签对精修轴的四种读数（都从 `out/onetonton-NN/` 读生产产物）：`--sweep` #1–#6 上三种规则 × θ 的首选一致 / 真人在集合里 / 每段标签数（描述性，θ 就是在这里选的）；`--criteria` #7–#15 的预注册三条判据（另报去掉 #14）；`--cost` 已**作废**（按时间覆盖算「在场」且重复收费，审计 2026-09-14 §3.1–3.2，保留备案）；**`--strong`** owner 现行定价下每个 θ 的代价（删 1、换轨 2、强证据漏标 2、合并本身 3），与 `refined_multi.py relabel` 同一把尺子所以两份素材可比；`--owners` 每 5 分钟窗一条轨的多数人，列出中途换主人。⚠ 轨的主人按整集多数定，`--owners` 列出的集这个前提不成立 | §5r.2 |
| `mocai_diar.py` | 魔裁的 >4 人仪器：**直接拿时间线对脚本真值打分**，不读 stable、不需要 ASR（所以对比两条臂时只差一件事）。逐条真值 cue 问「最活跃的轨是不是对的角色」，并报**槽位天花板**：4 槽一人一个能拿到多少、以及主播占掉一槽后的 3 槽上限。⚠ 主播不在真值里；cue 跨度是上界，绝对水平只能臂间比 | §5r.2 |
| `vram_guard.py` | 探索用的 **8 GiB 显存硬上限**（owner 2026-09-16）：分配器封顶（超过就 `OutOfMemoryError`、不溢出到共享显存）+ Windows `GPU Process Memory` 计数器看门狗（超过就 `os._exit`）。⚠ 看门狗 0.5 秒一采，仍会**漏掉瞬时尖峰**，真正挡住的是分配器封顶。⚠ 它**不能改用 `torch.cuda.memory_reserved()`**——那正是分配器封顶已经封住的量，且看不见共享显存（模块 docstring 有实测）。计数器由一个常驻 PowerShell 采（每采 ~4 ms；早先每采一个新进程要 0.83 core-s，因为枚举 GPU 计数器集就占 0.64 s）。每个 GPU 探索脚本第一件事就是 `vram_guard.install()` | §5r.2 |
| `mocai_run.py` | 魔裁整部不分离地跑一遍 diarizer（`--no-separate` 转码成与分离同规格的 ogg → 8 GiB 守卫下跑），**一个文件一个进程**（缓存分配器整个进程不还高水位）。五部 6.16–11.04 h 峰值一律 1.69 GiB | §5r.2 |
| `mocai_long.py` | 一条轨在 6–11 小时里保不保得住身份：按窗比 **global**（整片多数主人）与 **local**（该窗多数），差值是映射漂移的代价；`--transitions` 把每次换主人按**旧主人的语音去了哪里**分成 复用 / 从未拥有 / 共用 / 走掉。⚠ 测衰减只看 local（global 的前后半对比是被多数投票决定的）；`transitions` 的 SWAP 计数只是上界，错误数看 `classify` 的 MOVED | §5r.2、findings 第 6 条 |
| `mocai_split.py` | 把魔裁的大混合轨递归拆分（over4 那套：拼出这条轨自己的音频再跑一遍），看子轨能不能把主播与角色分开、纯度有没有升 | §5r.2 |
| `mocai_split_null.py` | 拆分的纯度增益是真的还是算术：同尺寸对照**随机打乱**与**连续时间块**两个零假设（读 `mocai_split.py` 的输出、不跑 GPU）。⚠ 任何划分的加权多数占比都 ≥ 父集，所以「全部变纯」不是证据；要看赢没赢过时间块 | findings 第 7 条 |
| `mocai_compose.py` | 拆分对**最终归属**有没有用：把 `mocai_split.py` 存下的帧分配拼回整条时间线，与原始 4 轨、以及「同帧同大小、只按时间分组」的对照一起打分。⚠ 拆分 > 原始本身不证明任何事（轨多了就会升），要赢过对照 | findings 第 7 条 |
| `mocai_truth.py` | 魔裁（yuka）galgame 实况的**带时间说话人真值**：游戏 Naninovel 脚本逐行带说话人标签，flowocr 早已把 OCR 轴对到脚本文本，本模块把两边对回去（五部 98.9–99.8%，共 22,539 段）。它是目前**唯一的长录音（5.7–10.7 h）与多人（每小时 11–15）素材**。⚠ 跨度是上界不是真值（OCR 字幕包含语音、可能带空白）；跨录音链，回答不了具名；`Narrative` 是标签不是人 | §5r.2 |
| `sortformer_traits.py` | 「首选轨 = 真人」的下降是人数带的还是时长带的（读 `tmp/refined-2609/agree.json` + 各自的 `-diar.json` 取时长，无推理、无 GPU）：两个子命令。`length` 偏相关与人数×时长两维中位表——结论是**人数**（−0.486 对 −0.037）；`slots` 每文件四个槽的时长占比——**76/76 全都吐满 4 条轨**，且「槽不够」之前就已经在掉（2/3/4 人 93/85/80%）。⚠ 本批时长只到 1.41 h，厂商说的「长录音退化」没有素材可测 | §5r.2 |
| `refined_multi.py` | 姐姐咚咚以外的多人精修轴（主 checkout 的 `data/refined-multi/希娜小姐应援会/` 75 期、`data/manually-refined-subs/2026-09反馈/nagi-bd`）对 `out/refined-2609/<id>/` 的生产产物（`--stage stable --diarize acoustics`，id 表在 worktree `tmp/refined-2609/ids.json`）：`agree` 首选一致 / 真人在集合里 / 每段标签数；`owners` 5 分钟窗中途换主人；`overlap` 精修轴里异人重叠时间有多少被我们判成 ≥ 2 条轨；`relabel` §5r A 三条判据在 θ = 0.5 上的读数与「强证据才计漏标」的最少编辑代价。⚠ 希娜的样式名是**声优所配的角色名**、非对白样式按名字启发式剔除，说话人集合是近似；双语期只算中文行 | §5r.2 |
| `clip_audit.py` | 两个 7 分钟预览片段（姐姐咚咚 #2 27:30、nepolabo 30:30）各臂 `out/<clip>-<arm>/` 纠错后的行：`--cost` 对精修轴的改动代价与时间重叠 cue 对（轨主人按片段人工指定），末尾另按 owner 现行定价（漏标 2 + 合并 3）重算一遍——**名次会变**，因为旧尺子看不见「这一行本就该拆开」；`--b12[=SUFFIX] [--detail]` 分轨重转候选在纠错里的去向（窗口内外对生产的相似度、丢掉的原段、用上的候选） | §5r.1 |
| `calibrate_hints.py` | 声学提示两个值的标定：同人 / 异人轨对的共活率（短轨下限 × 比例阈值）、第四轨占比规则在 VoxConverse / AMI / 直播上的触发率（`--collect ali\|ami\|vox\|streams`、`--decide`） | §5o.5 / 5o.8 |
| `s4_resegment.py` + `resegment_{system,user}.md` | 按说话人重分段：LLM 听着片段音频、拿着声学事实与 Qwen 两类候选，把重叠区里每个人的话各写一行（`--show` / `--ask lite\|agy RUN` / `--score`：A0 生产、A1 按轨直接切、A2 LLM） | §5h.14 |
| `multilabel_confidence.py` | 多标签行里哪个标签是文字的主人：四种情形、覆盖占比等信号的 AUC、留一场的工作点、生产现成的 `review` 旗子（读同一批 AliMeeting 缓存） | §5l.7 |
| `s4_live.py` · `s4_local.py` | S4 的专用转录模型臂：35t Live（API）与 Qwen3-ASR / Whisper（GPU；Qwen 要 transformers ≥5，在 `qwen-asr` 环境跑） | §5h.9–5h.10 |
| `over4_rescore.py` | 复审三轮第 2–4 条：身份错换固定分母（§5k.2 与 §5n.3 重算）、test 上按人数分档、§5n.1 核验器在 test 上挡掉的 unit 里有几个真孤儿（全读缓存） | §5n.5 |
| `over4_breakdown.py` | §5k.2 的数字由什么构成：5 s / 20 s 触发数、孤儿 unit 与合并后身份数、行级错拆成身份错与边界 / 集合错（读缓存） | §5k.4 |
| `over4_vorbis.py` | 单独隔离 Vorbis：原始 wav 直接过生产 `encode_asr_delivery`，onepass 对照 | §5m.3 |
| `geometry_spread.py` | 新几何下导出件对 NeMo 的逐字幕一致率（§5a.4b 判据） | §5j.1 A |
| `over4.py` | >4 人：onepass vs 滑窗 + 跨窗关联，VoxConverse dev 确定性抽样（`--list` 看选了哪些） | §5i.1 的 0.861 / 12.8% 与 −0.1 / −17.6 pp |
| `s4_llm.py` + `overlap_fix_{system,user}.md` | 多模态补漏：片段 + 时间轴 + 混排诱饵（`--ask lite\|agy RUN`，`--score`）。⚠ agy 必须带 task group 才会走传入的 routes，见 `CALL_ROUTING` | §5h.5 的 +0.0 / +0.9 pp |

## 素材在哪

⚠ **脚本里的路径是当年 `tmp/sortformer/` 的相对路径，跑之前要改。** 素材分三处：

| 素材 | 位置 | 可再生？ |
| --- | --- | --- |
| 合成测试集（`anchor_pollution.wav`、`5spk_*.wav`、`2spk_alt.wav` + 三个 `*_timeline.json`） | `data/diarization/synthetic/` | ❌ **不可再生**——没有构造脚本，2026-09-05 抢救出来的 |
| nepolabo 人工抽查表 | `data/diarization/nepolabo-spotcheck.md` | ❌ 人工产出 |
| 16 kHz 单声道 wav（nepolabo / yui / kaguya60，共约 300 MB） | 原在 `.worktrees/speaker-clustering/tmp/sortformer/`，2026-11-04 清理时删掉了（同目录的结果 JSON 还在） | ✅ 由 `assets/` 的源音 + 分离产物转码即可；`accuracy_history.py` 要它们，跑之前先转回原路径 |
| Sortformer 权重 `.nemo`（471 MB） | 同上的 `models/` | ✅ 从 HF 重新下载。⚠ Windows 上 `snapshot_download` 之后 snapshot 目录里的 `.nemo` 可能是 0 字节占位、真实数据在 `blobs/`：从 blobs 复制成 `.nemo` 再 `restore_from` 本地路径 |

| VoxConverse dev / test（CC BY 4.0，Oxford VGG） | 主 checkout 的 `data/diarization/public/voxconverse_{dev,test}_wav.zip`（1.99 / 4.27 GB，216 / 232 份）；RTTM 在同目录的 `voxconverse/voxconverse-master/{dev,test}/` | ✅ 重新下载。⚠ test 包内路径前缀是 `voxconverse_test_wav/`（dev 是 `audio/`），另有 232 个 `__MACOSX` 条目不是音频。**test 集留作 §5n 终判**，之前不许拿来调参 |
| AliMeeting 评测集（CC BY-SA 4.0） | `data/diarization/public/Eval_Ali.tar.gz` 与抽出的三场 `alimeeting/{audio,rttm,textgrid}/` | ✅ 重新下载 |
| 各轮逐文件结果 | `data/diarization/results/`（带 git HEAD 与起止时刻） | ❌ 实验产物，删了要重跑 GPU |

主 checkout 的 `data/diarization/README.md` 与 `data/index.md` 已登记上面几行（2026-09-21 合回 `dev` 时补）。

⚠ **合成素材只能用来理解机制，不能定任何阈值或默认值**——拼接系统性低估模型，
findings 第 7 节写着这一条。

## Nemotron-3-Diarization（2026-09-24）

NVIDIA `nvidia/Nemotron-3-Diarization`（revision `f667ed7`，OpenMDW-1.1，可商用；Streaming Sortformer 的后继，
8 槽、10 ms 输出）。**PyPI 的 NeMo 3.0.0 加载不了它**（RoPE）：把 NVIDIA-NeMo/Speech@`cf724ac` 的 `nemo/`
解到 `tmp/nemo-main`、`lhotse==2.0.0a6` 以 `--no-deps --target tmp/n3-deps` 装好，两者放 PYTHONPATH 最前；
asr 环境本身不动。权重放 `tmp/diarization-models/nemotron3/`。

| 脚本 | 用途 |
| --- | --- |
| `nemotron3_probe.py` | 公开真值集（CALLHOME 日语 / AliMeeting / VoxConverse）DER，与前几轮缓存臂同一把尺子；`--micro` 输入扰动探针 |
| `nemotron3_cost.py` | W2 / W2b 编辑代价对 DiariZen（**docstring 是预注册原文**，先于推理提交） |
| `nemotron3_diagnose.py` | `fixed` 掉在哪：逐人 / 逐轨 / 每 10 分钟主人 |
| `nemotron3_speed.py` | 三后端同波形单遍墙钟（不含加载） |
| `over8.py` | 超过 8 人的候选方法台：`vbx`（Nemotron 当 DiariZen 链的局部分割 + WeSpeaker/PLDA/VBx）、`recur`、`enroll` 及组合，外加两个生产后端参照臂 |
| `nemotron3_port.py` | 生产移植（`nemotron3_model` + `nemotron3_backend.vbx`）对 NeMo eager 的 A 类对照，以 σ=1e-7 扰动为噪声带（**docstring 是预注册原文**） |

⚠ PYTHONPATH 在 Git Bash 里要写成 `/c/...` 形式、冒号分隔：写成 `C:/...:C:/...` 时盘符里的冒号把列表切碎，
静默 import 到 site-packages 里的 PyPI NeMo（加载 Nemotron-3 时报 `rope is not supported`）。

⚠ 训练集含 VoxConverse dev+test（DiariZen 也训过）与 NIST SRE2000 CALLHOME Part 1：VoxConverse 读数只能比
**同一模型族的不同方法**；CALLHOME 日语可能重叠。AliMeeting Eval 两边都干净。
⚠ `over8.py` 的 `vbx` 默认用快配置（声纹 batch 16、TF32 开）；`--param prod=1` 换成生产配置，魔裁 4 小时上读数逐位相同。
逐文件原始输出已丢弃，指标都在 `results/nemotron3-20260924.json`；要复算先重跑推理（`nemotron3_probe.py`、`nemotron3_cost.py infer`、`over8.py prep/infer/run`）。

## 运行环境

需要 `nemo_toolkit`，而**生产不吃它**（计划 §8 台账的 D1：走导出）。这些脚本只在探索环境里跑。

`degeneration_scan.py`：从已跑完的三个 pass 任务的产物里读回答退化的信号（pass 2 可并接缝并掉的比例与注释行、pass 1 被改记为可并的越限 join、
每次调用的思考与可见输出 token），不调模型；判据与读数见 `docs/llm_followups.md`「模型回答的退化」。
