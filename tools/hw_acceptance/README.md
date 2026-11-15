# 社区验收套件（hw_acceptance）

给有 A 卡 / I 卡 / Mac 等硬件的贡献者用：在**同一台机器**上跑两臂管线——候选（要验收的加速路径）
与参照（默认是 CPU 路径，`--device cpu`）——再按开发原则里的复核门槛打分，交回一份带版本的 JSON。
这些平台的等级（官方 / 社区验收 / 实验）见 `docs/manual/resources.md`；**等级不改变任何行为**，
交回记录只是把「实验」升为「社区验收」。

## 跑

需要装了 `[asr]` 的源码 checkout（`docs/manual/repo-install.md`）和 ffmpeg。

```bash
python -m tools.hw_acceptance.run --out tmp/hw-acceptance a.mp4 b.mp4 long-talk.mp4
python -m tools.hw_acceptance.score tmp/hw-acceptance
```

- **至少三段素材，其中一段 ≥30 分钟**：连锁变化出在长 group 上，短素材看不出差异是否随时长增长。
  少于这个数也能跑，但只算一次刻画，不算验收记录。
- 两臂的管线选项可改：`--reference-args` / `--candidate-args`（例如 Mac GPU 以同机 CT2-CPU 为参照）。
- **默认两臂共用候选臂分出的人声轨**：先跑候选臂，参照臂直接用它的 `-vocal.ogg`（分离阶段按产物存在跳过）。
  这样比的是识别与第二模型校验，门槛也是在同一条人声轨上定的；CPU 上的分离又极慢（本机 3 分钟音频
  CPU 分离约 11 分钟），30 分钟素材要几个小时。要连分离一起比，加 `--own-vocals`。
- 要比较 Qwen 裁判，两臂都得开着 `--qwen-verify`（默认 `auto`，依赖齐全时就开）。它的差异只在
  stabilize 的取舍上看得见，所以是靠下面的 `stabilize` 一致率量的，不进对齐产物那几项指标。

## 交回什么

只交 `tmp/hw-acceptance/acceptance.json`（issue 附件即可）。里面是分数与环境（Python、torch、
CTranslate2、加速器与平台），不含音频和字幕。

## 怎么读

**先看 `valid_record`**。它问的是「候选臂真的跑在别的设备上了吗」：每份产物记着各阶段实际用的设备
（`files[].devices`），两臂逐阶段比，**每个文件都得至少有一个阶段换了设备**才算 `true`——只要有一份
悄悄回退了，它就会稀释掉其余文件的数字，而最可能回退的恰好是那份长素材。
两边都得报得出设备才算数：读不到的阶段进 `stages_without_device_record`，缺数据不当成换了设备。
最常见的原因是装的 CTranslate2 不是本平台的那个构建，识别已经悄悄回退到 CPU。
被比较的阶段是**识别、第二模型裁判**，以及（不共用人声轨时）**分离**——`accelerated_stages`
列出真正换了设备的那些。裁判没跑（`--qwen-verify` 没开、或依赖不全）时它记不下设备，同样进
`stages_without_device_record`。
⚠ **识别本来就留在 CPU 的平台**（Windows + A 卡，以及 Windows + Intel 卡）要特别当心：识别由 CTranslate2
决定，`--qwen-verify` 的尾段裁判又跟随识别设备（记录里 `referee` 一栏就是它），两处都不换设备——只有加
`--own-vocals` 把分离纳入比较，这份记录才证明得了什么；两臂都开 `--qwen-verify` 不够。Intel 卡上 silero 辅助与
语言重解的 Qwen 裁判也会换到卡上，而记录不记它们的设备，所以差异不全来自分离。

**`files[].stabilize` 是裁判那一半**：两臂 `-stable.json` 的逐条比对——留下的字幕是不是同一批
（`kept_only_by`）、标记是否相同（`tag_differences`，含「第二模型否决」），以及合起来的 `agreement`。
它**没有及格线**，但**低于 1.0 就会让 `review_needed` 为真**：裁判在有原生 bf16 的卡上用 bf16、
没有的用 float32，而它的结论要经 stabilize 删或留一行字幕才到得了输出——对齐产物那几项指标一个都
动不了。所以它不是不通过，而是和越过门槛的项一样，进同一份书面复核；不把它接进 `review_needed`，
就会出现「候选臂把字幕全删了、一致率 0，记录却说无需复核」。两臂有一边没跑到 stabilize 时是 `available: false`。
配对只按时间重叠（重叠达到自身时长的 30% 即算同一条），所以两条可能配到同一条、零时长的段配不上——
作为只报数的指标够用，别拿它当门槛。

`review_triggers` 列出越过门槛的项（`README_DEV.md` 开发原则 A 类，默认沿用
`docs/bench-baselines.md` 第十节判据 1–3，另加 ASR 覆盖率差 ≤1%，即识别出的段覆盖的语音占 VAD 语音的比例）。**越过门槛不等于不通过**，它要求
在记录里逐条回答三问：

1. 带方向的量（词起点偏差、覆盖率差）是否在所有文件上同号，且每份都超过门槛
   （`question_1_same_signed_bias_beyond_trigger` 给出词起点那一项的机械判断）；
2. 差异的**比例**是否随时长增长（看 `by_duration`）；
3. 能否指出一个局部差异传播开来的机制（要人看，数字答不了）。

三问都答「否」仍是噪音；任一为「是」，那条路径在该平台上按 B 类处理——移植算法的后端（例如 MLX）
则是修实现。救援次数只比两边都有的级：不支持 beam 的后端，beam 那一级被剔除
（`rescue.beam_rescue_attempted.excluded`）；`refine_fallback` 是退回 teacher-force 的窗口数。

记录由维护者整理进 `docs/bench-baselines.md`，并在 `src/finesub/speech/runtime/support_levels.py`
的 `_COMMUNITY` 里登记那一对（平台，厂商）。
