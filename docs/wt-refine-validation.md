# WT refine 信号与局部重试验证

## 目的与边界

本验证回答三个问题：哪些问题能在 word 后处理之前被看到；哪些信号足以让 ASR controller 立即
隔离重解；局部重试是否比把问题无条件推给后续重分组更有质量或效率价值。当前只在研究 runner 中
解析事件，FineSub 不收集、不解析，也没有生产默认路由变更。

验证集定义在 `tools/wt_refine_validation/manifest.json`，共 13 个 group：3 个普通 control、10 个从
现有生产日志挑出的异常倾向 group。后者覆盖轻/重 token stack、短/长 repeat motif、数字坍缩、
同组多异常和 group tail；同源人工纠正 SRT 只作文本代理。SRT cue 可能跨 VAD 边界，素材也包含
英文原声与日语同传，因此 runner 会记录 boundary spill，并且只在 spill 不超过 0.5 秒且原候选
相似度不低于 0.25 时自动把参考分数用于判胜。

历史异常标签表示该 group 在旧 WT 生产运行中曾异常，不表示当前 CT2 1-pass 必须再次坍缩。验证时
10 个异常倾向 group 中只有 5 个仍产生现行 word-level abnormality；另外 5 个应视为 hard-distribution
clean control，不能按“信号漏报”计算。

## 信号分层

低成本 path 信号不请求额外 attention matrix：

- `alignment_stack`：至少 3 个连续正文 token 各推进不超过 1 个 20ms frame；
- `long_token_span`：单正文 token 占据至少 5 秒；
- `decoder_repetition`：连续精确 token motif 至少重复 4 次且总计至少 8 token；
- `unfinished`：CT2 decoding-limit/无正常 end timestamp 的 span；
- `zero_duration_chunk_tail`：保留原词，只报告 chunk 尾零时长词。

研究用 attention 信号单独开关：

- `disfluency_candidate`：WT multi-peak 的原始/收紧起点、peak count/prominence；
- `boundary_uncertainty`：首尾 query 的 normalized entropy、次峰比、峰位和是否碰 weight window。

加入 token motif 后重新做 Harvard greedy plain/path/full 三档交替 10 轮热态测量：398.05ms、
402.16ms、408.38ms；path 增量 1.03%，全量 attention 增量 2.60%，且所有 word text/start/end
逐项一致。path 档仍接近调度噪声，生产优先开该档；attention 档留给研究或显式 disfluency 模式。

## 13-group 结果

普通组没有 `alignment_stack`、`long_token_span`、`decoder_repetition`、`unfinished` 或
`zero_duration_chunk_tail`。两个长普通组出现 disfluency 候选，Harvard 没有；这再次确认 disfluency
只能作为边界观测，不能触发重解。`boundary_uncertainty` 每个 span 都产生原始指标，尚未标定阈值。

10 个异常倾向组中：

- 5 个当前仍被现有 word-level detector 判为 hard；它们全部定位到单一 interval 并进入研究用
  immediate isolation；
- 其中 severe phrase collapse 被 `alignment_stack` 提前看到；numeric collapse 同时出现两个
  `alignment_stack`、`long_token_span`、`decoder_repetition` 和 `unfinished`；
- `zero_duration_chunk_tail` 只在 `credit-repeat-1393` 出现（2 个），普通组为 0。该 group 当前正文
  干净，同时另一个 interval 有 decoder repetition，因此正确路由仍是 deferred，而非删除或立即重解；
- 其余 4 个历史异常 group 在当前 1-pass 下干净，只携带 disfluency 观测，不执行无谓重试。

5 个 policy isolation 全部把结构异常清为 0，重解音频只占原 group 的 3.59%–6.08%，中位数 4.60%。
其中 `stack-severe-1555` 有可靠人工参考，文本相似度 0.506 → 1.000、覆盖 +0.12 秒；3 个结果结构
恢复但局部参考因声道/边界歧义不足以自动判胜；`short-motif-1222` 虽清除 repeat issue，覆盖少
0.12 秒，标为 inconclusive。结论是强局部信号值得立即生成隔离候选，但在缺少可靠比较证据时不能
静默替换原候选。

对当前没有 hard signal 的历史异常点仍跑了 oracle isolation probe。部分文本近似，但没有理由为
每个 soft signal 支付重解成本；这支持“disfluency 只记录、zero tail 待决”的保守路由。

## greedy 与 1-pass beam=5 隔离重试

在 5 个当前 hard group 上补跑 beam=5：4/5 隔离文本与 greedy 等价，剩余一例多出尾部 `pre`；小样本
没有观测到 beam 独有胜例，但 beam search 理论上仍可能小幅提高语义文本质量。单位音频计算中位数为
greedy 0.06714、beam=5 0.090084，beam 慢 34.2%。因此异常 interval 的首个局部重试继续选 greedy；
beam 保留为 coverage rescue 或有独立文本不确定证据时的第二质量候选，不能恢复无条件整组 beam 救援。

## 当前研究路由

1. 单一 interval 出现 `alignment_stack` / `long_token_span`，或现有 word-level hard issue：ASR
   controller 可立即做一次 greedy isolation，但保留原候选；只有 retry clean、覆盖不显著下降且文本
   证据不回归时才采纳。
2. `zero_duration_chunk_tail` 或 `decoder_repetition` 单独出现：传 deferred event，不删词、不重解；
   与 alignment stall/coverage failure 联合后才升级。
3. disfluency 和未标定 boundary metrics：`keep_with_signals`，不产生重试。
4. 多个 hard interval：需要决定先隔离、交还尾部还是整体重分组；在验证替代分组前保持 deferred。

这一路由落实两条额外标准：能证实质量提升时优先在 refine/ASR 做；质量相近但局部处理显著少解
音频时也优先。仅“更早看到信号”本身不构成下沉理由。

## 生产产物离线复核（2026-08-04）

用 `tools/wt_refine_validation/artifact_survey.py`（离线只读）对 5 个迁移验收素材
（`out/acceptance/`，去重后 897 个输出 segment）把最终 aligned/stable 产物里的
`alignment_events[]` 与三类既有证据做段级对照：现有词级判定、已知幻觉短语、stable 阶段
丢弃/打标。注意口径：**产物只含救援后的存活解码**，被隔离/覆盖救援拒绝的候选的事件不落盘，
所以这里测不出 decode-time recall——那需要 instrumented rerun。

带事件的 unique segment 共 22 个，逐例人工分类后：

| 信号 | 段数 | 病理 | 良性 | 分界特征 |
| --- | --- | --- | --- | --- |
| `alignment_stack` | 10 | 9（ご視聴×2、英文幻觉×3、decode-limit 坍缩残留×4） | 1 | n≥15 或 tpaf≥7 全为坍缩；n=3–4 混合 |
| `zero_duration_chunk_tail` | 11 | 3（全部与 stack 同现） | ~8 | 单独出现基本良性 |
| `decoder_repetition` | 6 | 3（n≥217） | 3（バカ×7 等真实复读，n≤16） | token_count 量级 |
| `long_token_span` | 3 | 3（全部 ~26s，坍缩伴生） | 0 | — |
| `unfinished` | 3 | 3（全部尖叫坍缩） | 0 | token_count 均 ≥218 |

四条主要发现：

1. **decode-limit 坍缩签名**：`unfinished`(≥218 tokens) 与相邻段的
   `decoder_repetition`(n≥217) + `long_token_span`(~26s) 成对出现，本 corpus 4/4 全为真实
   尖叫/坍缩窗口（样本小，但机制上自洽：解码打满 token 上限本身就是失控定义）。这些窗口现有
   词级判定**都能捕获**并触发隔离，但运行日志显示其中 2 例隔离重解仍坍缩、落到合并兜底，白付
   一次重解；签名可用来仿照 `_is_known_phrase_stack_only` 提前判「无可救内容」，或给兜底输出
   收紧边界。
2. **英文/BGM 幻觉家族是现有判定的真实盲区**：`The great plan…`、`Well, it's a kind of…`、
   `If there is something in the art,` 均带小 `alignment_stack`（n=3–4）+ 低段 confidence
   （0.19–0.40）+ 语言切换，词级判定全部漏过，3/4 存活到最终输出（至多被打时间漂移标）。
   单靠 stack n=3–4 精确率不够（普通句也会命中），需与语言切换/能量门控联合；这是信号真正
   可能新增召回的地方。
3. **`zero_duration_chunk_tail` 与小型 `decoder_repetition` 单独出现时以良性为主**，维持
   deferred 路由不变。token 级重复阈值（≥4 次且 ≥8 token）比词级（>7 连词）更敏感，バカ×7、
   おー×4 这类真实复读只有 token 级命中——「任一事件⇒异常」的接法会把真实语音判掉，不可行。
4. **残留异常的主类信号完全看不见**：118 个被 stable 丢弃/打标的段里只有 6 个带事件；主导类
   是时间漂移与语义错词，path 信号与它们不相关。path 信号不能当作通用段级质量指标，价值集中在
   decode-time 路由与上面两个特定家族。

## 生产窗口覆盖率对比（2026-08-04，405 窗口）

回答「现有各异常判定方法与信号候选各自能覆盖多少真实异常」。语料：`out/qwen-explore` 的
11 条 VAD 轨按当前生产分组规则重新规划为 **405 个 ≤30s 窗口**，每窗执行一次生产配置的
greedy 解码（turbo、temperature 0、path 信号开），**不做任何救援**——这是 310 窗口 sweep 的
再现与扩样，且这次保留了 per-window 明细。工具：`window_sweep.py`（GPU 解码 dump）+
`window_score.py`（离线判定器打分）。

被任一判定器命中的 98 窗全部人工裁决（标注入库
`window_sweep_labels_20260804.json`，id 绑定当次分组，分组参数漂移后需重扫）；对未命中的
307 窗用文本启发式（已知短语/英文块/超长词/短语级复读）扫描，**零漏网**。真值：86 个异常
窗 = 49 个文本可验证结构异常（坍缩复读 30、已知短语尾幽灵 6、幽灵词堆 6、语言切换幻觉 4、
拉伸词 3）+ 37 个低覆盖截断窗；12 个良性误报窗。

**49 个结构异常上的覆盖率**（TP/类内总数；FP 为良性窗误报数）：

| 判定器 | 坍缩30 | 短语尾6 | 幽灵6 | 语切4 | 拉伸3 | 合计 | FP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E:collapse_word_stack | **30** | 5 | 5 | 1 | 0 | 41 (84%) | 2 |
| E:long_word_duration | 23 | 0 | 0 | 2 | **3** | 28 (57%) | 0 |
| E:repeating_word_run | 27 | 0 | 0 | 0 | 0 | 27 (55%) | 1 |
| E:repeating_group_cycle | **30** | 0 | 0 | 0 | 0 | 30 (61%) | 0 |
| E:coverage_low | 0 | 0 | 1 | 1 | 0 | 2 (4%) | 0 |
| **E 词规则联合** | **30** | 5 | 5 | 3 | **3** | **46 (94%)** | 3 |
| S:unfinished | 27 | 0 | 0 | 0 | 0 | 27 (55%) | 0 |
| S:decoder_repetition(n≥64) | 27 | 0 | 0 | 0 | 0 | 27 (55%) | 0 |
| S:alignment_stack(n≥8∨tpaf≥4) | 27 | **6** | 4 | 0 | 0 | 37 (76%) | 2 |
| S:long_token_span | 22 | 0 | 0 | 2 | 0 | 24 (49%) | 0 |
| S:zero_duration_chunk_tail | 2 | **6** | 5 | 1 | 0 | 14 (29%) | 4 |
| S:decode_limit_signature | 27 | 0 | 0 | 0 | 0 | 27 (55%) | **0** |
| S:lang_switch_lowconf（新） | 0 | 0 | 0 | **4** | 0 | 4 (8%) | **0** |
| **S 联合（调阈值后）** | 27 | **6** | 4 | **4** | 0 | **41 (84%)** | 2 |
| S 联合（任一事件，不调阈值） | 27 | 6 | 5 | 4 | 0 | 42 (86%) | **11** |
| **E ∪ S(调后)** | **30** | **6** | 5 | **4** | **3** | **48 (98%)** | 4 |

**86 个全量异常（含 37 截断窗）上的综合覆盖率**：现有词规则联合 46/86（53.5%）；
现有词规则 ∪ coverage_low（即现行生产判定全集）**83/86（96.5%）**；再并入调阈值信号
**85/86（98.8%）**。唯一残留是 `BV1UBjq6fEgb:0` 的单个零时长幽灵词（仅
`zero_duration_chunk_tail` 可见，而它单独用有 4 FP，不值得为此升级）。

结论（按证据强度排序）：

1. **信号不能替代现有词级判定。** 核心坍缩类 30/30 被 `collapse_word_stack` 与
   `repeating_group_cycle` 各自独立全覆盖；信号联合只有 27/30——3 个**拉伸型**坍缩
   （え、/ぁ 长串被 BPE 合并成变长 token）让 token 精确 motif 与 stack 检测双双失效。
   token 级重复检测对 BPE 合并敏感是结构性缺陷，词级文本比对没有这个问题。
2. **调阈值后的信号在现有之上净增 +2/49**：`lang_switch_lowconf`（CJK 素材中 Latin 占比
   >0.7 且段 conf<0.6，4/4、0 FP）补上英文/BGM 幻觉家族的唯一漏网；
   `alignment_stack(big)∧zero_tail`（12 TP / 1 FP）补上词规则漏掉的已知短语尾幽灵
   （短语只有 2-3 词、不够 `collapse_word_stack` 的 count≥3 时）。
3. **`decode_limit_signature`（unfinished + 大块 repetition/stack）27 命中 0 误报**，
   与坍缩窗完全重合。检测上是冗余的，但它是「这窗打满 decode limit 的失控解码」的免费
   确证——适合做救援路由（隔离仍失败直接走合并兜底，省一次注定失败的重解），而不是检测。
4. **不调阈值直接用原始事件不可行**：any-event 联合 11 FP（tuned 2 FP），
   `decoder_repetition` 小 motif（真实的 バカバカ/おー おー/そうだよな 复读）与散发
   `zero_duration_chunk_tail` 是主要误报源，维持 deferred 定位。
5. **覆盖率维度与结构规则近乎正交**：coverage_low 在结构异常上只命中 2/49，但独占全部
   37 个截断窗——现行生产综合覆盖 96.5% 的主要贡献者。`long_word_token` 与
   `repeating_token` 本语料 0 命中 0 误报（历史 stress 语料中有命中形态，保留无害）。

口径边界：单次 greedy、单模型（turbo）；真值由文本+事件人工裁决，语义错词不在「结构异常」
口径内；低覆盖窗的真值即 coverage_low 判据本身（循环），故结构表将其单列；良性/异常裁决
含少量边界主观判断（幽灵词堆类）。

### 已纳入生产（2026-08-04 初版，08-05 按大范围误伤复核收紧）

两类盲区的落点都不在 ASR 隔离——它们的失败模式是「垃圾存活到成品」，故接在输出清理侧，
零新增 GPU 成本。初版依据 405 窗口的 0 误报落了「语切丢弃 + 幽灵文本删除」；随后的
大范围复核（170 份 aligned/stable 产物约 5 万段 + 5 条未进过 sweep 的素材约 5.7h 重新
解码 + 人工修正字幕对照）暴露了两处外推失败，据此收紧：

- **幽灵重复段** → `vad-asr` 链 `drop_ghost_duplicate_segments`（docs/vad-asr.md）。
  初版只有「跨度 + 邻段重复」两条件；全产物扫描发现 wt 时代产物里多数命中是
  **时间被量化压扁的真实急促复读**（连喊两声 `おい!`、歌词 `Ten` 复唱、笑声）——删了
  就是真内容。收紧为三条件：跨度 + **段上必须带 `zero_duration_chunk_tail`/
  `alignment_stack` 解码证据** + 非幽灵邻段重复。收紧后 wt 形态全部被挡，fw 时代的
  真幽灵（乙女 / ロザリンまで×2 / もう待って）保留命中。删除明细入 metadata 可审计。
- **语言切换幻觉** → 降级为 **stabilize profile 2 仅打标**（`语言切换幻觉`），
  **不参与 profile 0 丢弃**。405 窗口上 4/4、0 误报没有外推住：H6dTZf9QFTY（歌回/
  英配 PV，未进 sweep 语料）有 15+ 行真实英文歌词/台词命中该规则（保留判定最初来自
  LLM 层产物——H6 无人工字幕；2026-08-05 Qwen 双模型重认抽检 5/5 确认音频确为英文）；
  BV1cqLR6hEp3 的英文块经人工字幕对照实为**真实日语台词的翻译型幻觉**——删除会丢掉
  真实对话的唯一痕迹，正确修复是强制语言重解（记入 handoff P1，属重试类手段而非删除）。
- **`decode_limit_signature` 决定不接**：本可用于「隔离前预判无可救内容、直接走合并兜底」，
  但现有证据里 decode-limit 窗口的隔离成败参半（yingtao 2 例失败落兜底、BV1cq 1 例成功），
  跳过重解会放弃成功案例的修复机会；等救援链中间候选的信号数据补齐后再议。

同一轮审计还修了**存量删除机器**的一处误删（2026-08-05）：stabilize `高度疑似幻觉` 的
very_low_energy 两条腿会删掉时间轴坍缩/漂移的真实语音（能量采样落在静音处、词置信却
0.92+）。加入「词加权置信 >0.9 且能量高于 −80dB 地板则能量证据不触发丢弃」的豁免，
规则与审计数据见 docs/asr-stabilize.md。注意历史 stable 产物是旧版代码生成的——审计
删除行为必须用现行代码重跑 stabilize，直接 diff 旧产物会把已修复/已变更的行为算进来。

方法论教训（下次改判定规则前先读）：**验证集的类型覆盖比样本量更重要**——405 窗口全部
来自谈话向直播，歌回/英配/双语类型一个都没有，0 FP 是类型盲区给出的假保证；删除类规则
的验收必须包含「规则命中处与人工参考对照」，只看「干净样本不误伤」不够。

一致性验收（3 clip 全链 A/B：dev 基线代码 vs 本分支，同音频跑 `vad-asr`+stabilize）：
BV1UBjq6fEgb 唯一差异即幽灵清除——且基线里幽灵把邻段挤压到 0.1s
（`[15.25-15.35] 満載って感じですけど乙女`），清理后邻段恢复完整跨度
（`[15.25-16.39] 満載って感じですけど`）；BV1cqLR6hEp3 与 kaguya60 的 stable 输出
**零差异**（kaguya 8 个语切标签与既有能量类丢弃完全重叠——分离后 BGM 段能量极低时
`高度疑似幻觉` 已能接住，语切标签的净增量在能量不低的场景）。对旧验收产物直接重跑新
stabilize：BV1cqLR6hEp3 / BV1dwjP6LECU 中此前存活到成品的英文幻觉
（`The great plan…`、`If there is something…`、`Well, it's a kind…`）全部被
`语言切换幻觉` 标签丢弃。

## VAD 改版后的复测（2026-08-05，400 窗口）

dev 合入 VAD 重做（-45 峰值底线、carve、记账式扣分、`pause_hints`）后，用新检测器
重生成同 11 条轨（`out/qwen-explore-vadv2/`，语音秒数普遍收缩 1-6%）、重扫 400 窗口
（旧 405），98 条旧裁决按时间重叠 1:1 映射（零碰撞）。结论：

- **24/86 旧异常在新窗口下未复现**——坍缩/幽灵/短语尾对窗口边界敏感，8 个低覆盖窗
  因 VAD 修剪反而过线。这不是判定器退化，是「窗口即样本」的语料特性；
- **13 个新真异常出现**（9 低覆盖 + ご視聴 短语尾 + 6.13s 拉伸词 + 2 坍缩），
  全部被现行阶梯命中；
- 刷新真值 **74 个仍存在的异常：生产判定+清理覆盖 74/74**；良性误报 4/12
  （全为重试类，无删除风险）。乙女幽灵三条件在新解码上逐条复验仍命中；
  语切命中 3/3 仍全为幻觉（kaguya60 的 BGM 幻觉窗在新 VAD 下根本不再解出内容）；
- 未命中窗启发式无漏网扫描：6 个真实急促复读（正确不删）、1 个量化真实感叹词
  （正确保留）、**1 个残留**：`yui:37` 的 `ありがとうございました`（0.04s、无解码
  事件、无重复源——保守幽灵规则设计内的放行，同旧语料 1/405 残留同族）。

gold 侧重标定（新能量轨 + 新 VAD）：quiet_frac 分离与旧结论一致（filled 0.70 /
词头 0.00；gate 0.4 召回 25/32、0/25 词头误删；onset 精修 filled 中位 0ms worst 15ms）；
VAD lead 中位 +102ms（旧 +120ms，+0.1s clamp 常数成立，VAD 迟到 1/26）。
**`pause_hints` ≠ 当初设想的 merged-gap 全集**：它按 noise-floor 判据记录，对 gold
重偏早 filled 家族仅覆盖 7/27（帧级 ≥14dB 凹陷是语音级形态，多数够不着 floor+6dB），
故 hint 锚点是子集信号；该家族的主修复力量是 `[*]` 能量门控删除（见 asr-align
「词首修正」）。位置门实测代价：BV1cq 上 7 个高 quiet_frac 的 mid-phrase 块因无
gap/hint 证据落入融合，保留 0.2-0.7s 的保守偏早。**该位置门随后经用户确认放宽**
（同日）：它在 gold 上从未提供词头误删保护（能量门独测所有位置 0/25），只损失召回。
放宽实测：BV1cq +6 删（5 处 gold 确认可删、1 处无对照、0 词头误删）、kaguya60 +26 删
（前 8 大后移抽查全为谈话段词前长停顿，音乐段照旧被能量门挡下）；删除仍只动时间戳，
风险上界为块长。长后移的暴露面核查：kaguya60 仅 2 例 >1s 无位置证据（1.05s/1.59s），
逐例实听确认均为前词残余 + 真停顿、删除正确（其中 1.59s 例为 1.4s 数字静音）。
据此不设位置类兜底，仅保留后移 3s 上限（预期永不触发的 in-case 保险）。

词首修正端到端（新生产链 BV1cqLR6hEp3 全 clip，位置门放宽后最终形态）：52 块 →
17 删 / 32 融 / 3 短融 + 5 段首门控（全部回退），interval clamp 45 处、hint clamp
11 处；全体 gold 行词首 |err| 中位 41→**18ms**（无修正 vs 修正后）、p90 264→233ms，
delete 行中位 25ms、0 词头误删（1 例 267ms 为新旧解码块边界不一致的歧义行，
能量证据支持新边界）。

### 本节数据的 VAD 基线（合入 dev 时补记）

以上全部标定跑在**峰值 clamp 变更之前**的 VAD 上（dev `8ced6de`，见
[vad-energy.md](vad-energy.md)「峰值 clamp 为什么不是全局缩放」）。该变更只影响归一化后
峰值曾超 0.98 的文件，本节 11 条源里 **kaguya60** 属于此列（752→753 interval；起点
94.8% 逐字不变，7 个位移 >0.1s）。逐层看：

- **quiet gate 免疫**——判据是「局部中位数 − 12dB」的相对量，旧实现那个均匀 dB 位移正好抵消；
- **`CLAMP_LEAD_SEC = 0.1`** 标的是 VAD lead 中位（+102ms），锚点 ~95% 未动，结论预期不变；
- **`window_sweep_labels_20260805_vadv2.json`** 的窗口 id 绑定 `out/qwen-explore-vadv2/*-vad.json`
  这批**缓存**产物，与标签仍自洽；若删缓存重跑 VAD 再打分，受影响的源需按该文件
  `_meta.derived_from` 记录过的同一套 max-time-overlap 重映射走一遍。

## 尚未完成

- validation set 仍小，特别是 `zero_duration_chunk_tail` 只有一个异常 case；扩大前不能估计精确率；
- ~~被隔离/覆盖救援拒绝的候选解码不落盘事件，decode-time 的信号 precision/recall 无法离线
  估计~~——2026-08-04 已由 `window_sweep.py` 的 405 窗口首遍解码语料补上（见上节）；仍未
  覆盖的是救援链中间候选（隔离重解、beam 重解）的信号表现；
- boundary entropy/peak ratio 只有原始分布，没有人工边界 gold，不设阈值；
- runner 尚未模拟 FineSub 的相邻 regroup，因此目前只能量化局部 isolation，不能宣称全面优于 regroup；
- checkpoint 已让生产 `fw-refine` 默认收集 path events，并随 `alignment_events` 透传到
  aligned/stable segment；`detect_disfluencies` 自 2026-08-05 起默认开启，其候选由
  `recognition/word_starts.py` 在 stage 内消解（docs/asr-align.md「词首修正」）。
  ASR controller 尚不依据 path 事件触发路由动作，FineSub 也未解析。

可重复运行命令与结果 schema 见 `tools/wt_refine_validation/README.md`。本轮研究产物写在
`out/wt-refine-validation/full-v3.json` 与 `beam5-hard.json`，不纳入版本控制。
