# asr-align

`asr-align` 使用 `whisper-timestamped` 对已有 VAD interval 做 ASR、词级时间映射和结果清理。
源码与 CLI 入口位于 `src/asr_align.py`。

## 输入与输出

输入 JSON 至少包含：

```json
{
  "segments": [
    {"start": 1.2, "end": 4.8}
  ]
}
```

`--audio` 必须指向生成这些 interval 的同一时间轴音频。默认输出名为
`<vad-json-stem>-asr.json`。

```powershell
python src/asr_align.py out/input/vad.json \
  --audio out/input/input-vocal.ogg \
  -o out/input/input-asr.json \
  --model large-v3-turbo \
  --language ja
```

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--audio` | 必填 | 与 VAD JSON 同时间轴的音频 |
| `--model` | `large-v3-turbo` | Whisper 模型 |
| `--device` | CUDA 优先 | 无 CUDA 时告警并回退 CPU |
| `--language` | 自动检测 | 显式指定可避免语言误判 |
| `--gap` | `0.3` 秒 | 组尾合成静音时长（inter-interval 静音为自适应，不受此参数控制）；其前保留至多 `0.7` 秒原始 gap 音频 |
| `--block-seconds` | `600` | 流式音频 block；`0` 表示关闭分块 |
| `--pad-seconds` | `10` | block 左右上下文 |

## 当前对齐逻辑

1. 校验并裁剪 VAD interval，按时间排序。
2. 按目标时长和自然间隔动态分组。相邻 interval 之间插入
   `min(original_gap, 0.7)` 秒的**原始 gap 音频**（保留 VAD 截掉的低能量尾音）
   加自适应合成静音 `min(0.1 + 0.2 × original_gap, 0.8)` 秒
   （GAP_SILENCE_BASE_SEC=0.1, GAP_SILENCE_GROWTH=0.2, GAP_SILENCE_MAX_SEC=0.8；
   给解码器统一的断句提示）；组尾再垫
   `min(0.7, 到下一 interval 的间隙)` 秒原始音频 + `--gap` 秒静音（默认 0.3）。
   recall 补录
   批次的组尾额外受**下一个已覆盖 segment span** 约束（补录链常终结在已有
   segment 起点处，此时垫 0 秒，不重复转写已覆盖语音；终结在 interval 边缘时
   同样能拿到至多 0.7 秒真实尾音）。分组长度计算与实际合成音频使用同一 gap 长度。
3. 调用 `whisper-timestamped`，显式启用 word confidence。常规路径使用 efficient 单遍对齐：
   greedy decoding、单一 `temperature=0`、不使用 beam search 或 temperature fallback
   （异常/覆盖率救援中的 beam 重解是例外，见第 5/6 步）；
   `refine_whisper_precision=1.0s` 保持不变。该配置避免 naive 两遍模式把上一段对齐后的
   结束时间作为下一段的对齐起点，但识别文本可能与旧的 beam/fallback 配置不同。
   `--language` 未指定时，合成后时长不超过 10 秒的 group 优先沿用最近 10 个真正经过
   自动检测的输出 ASR segment 中的语言众数；没有历史时仍自动检测，频率并列时取最近值。
   这里的 group 指每一次实际 Whisper 调用，包括正常 group、regroup 后的 subgroup、recall
   group，以及异常后降级的逐 interval/segment ASR。未被最终采用的异常候选不会写入语言历史。
4. 将 Whisper 输出映射回原时间轴：**每个 whisper-timestamped 段整体保留**，不再在
   VAD interval 边界切开；词坐标在 interval 内 1:1 映射，落在 gap 保留音频内的词同样
   1:1（合成静音区按剩余原始 gap 比例映射）。段挂到词数占多的 interval 供 finalize
   使用。相邻段因尾词能量延长产生的重叠在输出前收回（后段起点优先）；
   收回时被完全甩到新末尾之后的词不留零时长残骸，而是就近并入本段最后一个
   存活词（文本拼接、confidence 取最小值）；整段塌缩时其文本并入后段首词
   作前缀、空段删除。
5. 检测异常重复/超长结果，最多 regroup 重试 2 次；重试用尽后先做一次
   `beam_size=5`（temperature 0）的整组解码，干净且覆盖率不低时直接采纳，
   否则才逐 interval fallback。beam 解码会让 whisper-timestamped 切到 naive
   两遍对齐，词级时间精度低于 efficient 单遍路径。
   `long_word_token` 使用独立 word-unit 指标：空格分词语言每个词计 3，CJK 等
   无空格文字每字计 1，纯数字串计 1，混合文字按各部分累加；单个 ASR word
   达到 15 units 时判为异常。该指标不是 Whisper tokenizer token 数，也不是字幕
   `weighted_char_count`。此外会检查整个 group 的局部精确循环；局部循环次数
   不少于 4 且循环跨度不少于 32 units 时触发 regroup。该规则只触发
   重试/回退，不直接压缩局部循环文本；旧的单 token 和相同 word run 检测继续保留。
6. **覆盖率救援**（正常 group 与 recall 批次都适用）：greedy 解码可能在 30 秒
   窗口内提前 EOT，整段跳过而所有输出质量指标（`no_speech_prob`、
   `avg_logprob`、异常词检测）全部正常——唯一可靠信号是覆盖率。当输出
   segment 与本批 interval 的重叠时长低于 `0.6 × interval 语音时长 − 2 秒`
   时（不足 ~3.3 秒的小批天然豁免），启动救援阶梯：先整批 `beam_size=5`
   重解（干净且覆盖达标即采纳）；仍不足则回到 greedy，把首个 interval 剥离
   为独立窗口、其余合为后窗重解，后窗覆盖率仍低就继续剥离（收敛于逐
   interval）。救援结果仅在覆盖时长严格增加时替换原结果。
7. 对正常结果未覆盖且累计不少于 5 秒的 complement 做临时 recall ASR；短于
   `0.25` 秒的 complement 碎屑不参与（孤立碎屑单独解码必然幻觉，也不计入
   5 秒阈值）。
8. 使用词尾附近的加权能量将最后一个词最多延长 1 秒，然后清理、排序并输出。
   fallback 清理不再逐 word 压缩，也不再单独合并相同 word run；它在完整 segment
   原始文本上查找连续精确重复，标点和空格同样参与比较。重复次数超过 7 时保留 5 次，
   并将覆盖重复区间的所有 words 合成为一个 word（时间取首尾、confidence 取最小值）。
   清理会反复执行到稳定，因此同一 segment 内多个不同的重复区间也会分别处理。

## 输出字段语义

每个输出 segment 包含 `start`、`end`、`text`、`lang` 和 `words[]`。存在上游数据时还会包含：

- segment `confidence`：来自对应 Whisper 来源 segment。
- segment `no_speech_prob`：来自对应 Whisper 来源 segment。
- word `confidence`：来自 `whisper-timestamped`；合成/合并词取来源最小值。

输出 segment 与 Whisper 来源段一一对应，两个 segment 指标即来源段指标。注意它们是
Whisper 在合批拼接音频（interval + 保留 gap 音频 + 0.3 秒合成静音）上算出的：
`no_speech_prob` 对应的 30 秒窗口是拼接产物而非原始音频，其分布与常规整轨 Whisper
用法系统性不同，按常规语义设阈值过滤会失准。

独立 `asr-align` 不持有 VAD 的逐帧能量轨，因此不会新增
`vad_weighted_energy_db`；该字段由组合工具 `vad-asr` 在最终边界上计算。

## 验证

```powershell
python -m pytest -q test/test_asr_and_text_utils.py test/test_intervals.py
```
