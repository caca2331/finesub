# separator_rate —— 分离器工作采样率的探索（2026-08-24/25）

问的是：**让 BS-Roformer 在低于 44.1 kHz 的采样率上工作，能省多少、赔多少。**

结论与生产取舍在 [`docs/separator-optimization.md`](../../docs/separator-optimization.md)
的「E12」一节，那里只留决策与结构性事实。**本文是过程与全部逐素材数据**，以及怎么重跑。

## 为什么这条路子本身是对的

`audio-separator` 的 `Separator(sample_rate=...)` 在 `common_separator.py` 里被三处读取，
**都在调用期**：`librosa.load(mix, mono=False, sr=self.sample_rate)` 读输入、
`AudioSegment(..., frame_rate=...)` 与 `sf.write(..., self.sample_rate)` 写输出。所以基准工具
在 build 之后写属性与走构造参数等价——`check_api_equivalence.py` 实测同一 60 秒 clip 两条路
输出 FLAC 的 **sha256 完全相同**（eager、无 accel）。

上游 README 把这个参数写成 "Modify the sample rate of the output audio"，**只说输出**。
实现上它同时是解码率，也就是模型的工作采样率。任何人以为自己只在「把输出存成 16k」，
实际上是在让 BS-Roformer 在 16k 上跑。**这是上游的坑。**

模型侧根本没有采样率概念：lucidrains 的 BS-RoFormer 不接受 `sample_rate`，STFT 的
`n_fft`/`hop_length`/`win_length` 全以采样点为单位，`freqs_per_bands` 只是一串 bin 数
（构造时断言其和等于 STFT 频点数）。所以改解码率是唯一杠杆。

## 协议

- 5 个素材：`BV1UBjq6fEgb` 186s、`BV1kYLR6AEXv` 270s、`BV1ySjz6FEzD` 354s、
  `BV1cqLR6hEp3` 536s、`BV1dwjP6LECU` 636s。
- 4GB profile（单 worker），AMP + AOTI，`torch 2.9.0+cu128`，RTX 5060 Ti，600s core / 10s pad，
  输出 `.flac`（避免 Vorbis 混进差异）。下游各臂只换 `-vocal.flac`，其余走
  `finesub.pipeline --stage raw-srt --language ja`。
- **下游逐字确定**：同一份 `-vocal.flac` 重跑，`-raw.srt` 逐字节相同。噪声底是 0。
- **空白对照臂（不可省）**：同样 44.1 kHz，只把 `--block-seconds` 从 600 改成 120 让分块边界
  挪位。分块边界随 worker 数变动、Roformer 对分块敏感，这两件事项目已接受为质量中性，
  所以这条臂扰动波形却不带质量主张——它画出的就是所有指标的噪声底。**没有它，任何单素材
  读数都不可解释。**
- **与所有臂无关的时间轴裁判**：`data/manually-refined-subs/四月一日/` 的人工中文字幕，
  按时长与素材一一对应。译文无用，**时间轴是人标的**。局限：cue 是显示时长不是发声区间，
  会过覆盖并桥接短停顿，且译者**不给笑声与感叹配字幕**——这一条恰好卡在关键处。

## 两轮的输入条件（重要）

第一轮全部取自 `assets/bilibili/*.ogg`，那是 **16 kHz 单声道**——旧下载路径的遗留缓存。
`download_audio` 现在取 `bestaudio/best` 且不转码（`media/source.py`），生产喂给分离器的是
**原生采样率的全带宽立体声**。第二轮在原生输入上重验，**结论有实质变化**。

## 数据

### 性能（与输入格式无关，块数只取决于时长与模型率）

| 素材时长 | 44100 | 32000 | 29400 | 22050 | 16000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 186s | 24.7s | 21.9s (1.13×) | — | 19.6s (1.26×) | 18.4s (1.34×) |
| 270s | 28.6s | 26.1s (1.10×) | 24.9s | 21.7s (1.32×) | 20.8s (1.37×) |
| 354s | 33.0s | 29.4s (1.12×) | 29.4s | 25.2s (1.31×) | 22.1s (1.49×) |
| 536s | 43.2s | 37.1s (1.16×) | — | 28.9s (1.49×) | 25.1s (1.72×) |
| 636s | 49.4s | 39.2s (1.26×) | — | 32.6s (1.52×) | 27.2s (1.82×) |

块数比恒定（→32k 1.378、→29.4k 1.5、→22.05k 2.0、→16k 2.756）；wall 上的加速小于它，
且随时长增长——差额是模型加载/解码/写盘这些不随采样率缩放的固定开销。稳态下**每块成本各档
相同**（扣掉首块约 5s 预热，44.1k 22.4s/67 块、16k 8.0s/24 块，都是 0.33 s/块）。

### 掉段（`dropout.py`）—— 16k 的死因

数「基线判为有声（>−45 dB）而该臂落到底噪（<−80 dB）」的秒数。同参数复跑基线得 0。

| 素材 | 有声秒 | 32000 | 22050 | 16000 | 16k 最长空洞 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **旧 ogg 输入** | | | | | |
| BV1UBjq6fEgb | 184 | 0 | 0 | 0 | 0 |
| BV1kYLR6AEXv | 204 | 0 | 0 | 2 | 2s |
| BV1ySjz6FEzD | 324 | 0 | 2 | 16 | 8s |
| BV1cqLR6hEp3 | 438 | 0 | 0 | 26 | 7s |
| BV1dwjP6LECU | 483 | 0 | 0 | **207** | 14s |
| 合计 | 1633 | 0 | 2 | **251** | |
| **原生输入** | | | | | |
| BV1UBjq6fEgb | 184 | 0 | 0 | 0 | 0 |
| BV1kYLR6AEXv | 204 | 0 | 0 | 0 | 0 |
| BV1ySjz6FEzD | 315 | 0 | 0 | 13 | 8s |
| BV1cqLR6hEp3 | 439 | 0 | 0 | 1 | 1s |
| BV1dwjP6LECU | 476 | 2 | 0 | **98** | 13s |
| 合计 | 1618 | 2 | **0** | **112** | |

**放大 16k 损伤的是声道，不是带宽。** 同素材 `BV1dwjP6LECU`、模型率固定 16000，只换输入：
原生立体声丢 98s、原生单声道（带宽相同）丢 **181s**、旧 ogg 丢 205s。98→181 全部由声道造成，
占总差 107s 中的 83s（**78%**）。BS-Roformer 是 `stereo: true` 训练的，人声居中、伴奏更宽，
声道差是它判定人声的一条线索。**但 44.1k 下单声道丢 0 秒**——声道线索只在 band 已错位时才承重。

### VAD 区间裁定（`vad_verdict.py` / `native_analysis.py`）

以人工时间轴为裁判逐 10ms 判定。**基线 VAD 自身的真语音命中率是 82%**（各素材 71–92%），
这是读下表的标尺——不是 100%，因为 VAD 带呼吸与静音余量，而人工字幕不标笑声与感叹。

| 输入 | 臂 | 漏掉 | 其中真语音 | 多出 | 其中真语音 |
| --- | --- | ---: | ---: | ---: | ---: |
| 旧 ogg | 空白对照 | 32.8s | 18.7% | 28.5s | 30.9% |
| 旧 ogg | 32000 | 47.5s | 17.4% | 62.1s | 28.1% |
| 旧 ogg | 29400 | 44.3s | 18.8% | 56.1s | 33.1% |
| 旧 ogg | 22050 | 48.0s | 18.7% | 72.6s | 33.6% |
| 旧 ogg | 16000 | **274.5s** | **70.5%** | 62.2s | 34.3% |
| 原生 | 空白对照 | 23.0s | 17.6% | 22.2s | 31.0% |
| 原生 | 22050 | 30.9s | 30.4% | 61.1s | 19.6% |

**漏掉一侧低采样率没有退化**（真语音占比与空白对照同量级）；**16k 完全不同**——漏的 274.5s 里
70.5% 是真语音，`BV1dwjP6LECU` 一个素材就漏 200.5s、其中 76.7% 是真语音、**71 段超过 1 秒**。
**多出来一侧是唯一越界的轴**：22.05k 是空白对照的 2.5–2.7 倍。

但**「不在人工时间轴内」不等于「不是人声」**。逐句核对 `BV1cqLR6hEp3` 的 96–126 秒
（`diff_cues.py`），人工字幕在 98.6–108.4s 是空的，22.05k 在那里转出三串 `あははは`——
那是**真的笑声**，译者没配字幕而已。基线把笑声压掉了。正确说法是：低采样率保留了更多
**真实但不想要的**人声，而不是凭空幻觉。代价出现在同一处，它把 cue 花在笑声上时把词认错了：

| 时间 | 人工中文 | 44.1k | 22.05k |
| --- | --- | --- | --- |
| 95s | 阿兰吉约丹…… | `アランギヨタンの` ✓ | `荒木よたんの` ✗ |
| 108s | 简直是摇篮曲啊 | `おこもり唄じゃんこれ` ✓ | `痛じゃん!` ✗ |
| 113s | 在门前唱歌 | `永遠の前で` ✗ | `トアの前で` 更近 ✓ |

**时间轴本身没有退化**（`boundaries.py`，对每个人工 cue 边界取最近边界的距离）：

| 臂 | 旧 ogg 中位 / <0.3s | 原生 中位 / <0.3s |
| --- | --- | --- |
| 44100 | 0.153s / 68.9% | 0.158s / 67.9% |
| 空白对照 | 0.167s / 66.7% | 0.156s / 67.7% |
| 32000 | 0.151s / 69.6% | — |
| 22050 | 0.159s / 68.3% | **0.149s / 69.0%** |
| 16000 | **0.232s / 55.4%** | — |

「完全落在人工时间轴外的字幕条」也没爆发（`ghost_cues.py`）：基线 39/588、空白对照 44/585、
32k 41/602、22.05k 48/584、16k 41/503。VAD 多 admit 的几十秒**绝大部分在 ASR 与稳定化阶段
被吸收了**。

**稳定化阶段的计数器（异常隔离、移除短语）不是可靠的臂间判据**：五素材上它并不一致偏向基线
（`BV1ySjz6FEzD` 是 44.1k 6 次、22.05k 2 次）。只有掉段秒数、区间裁定、边界距离与配对 CER
经受住了空白对照。

### 词准确率（`cer.py`）

参照系 `out/reference/<id>/<id>-corrected.srt` 是一次 44.1 kHz 生产 run 的 LLM 纠错稿，
**对基线系统性有利**，只能配对读且必须对着空白对照读。Δ CER 相对同条件的 44.1k 基线臂：

| 素材 | 空白对照 | 32000 | 29400 | 22050 | 16000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BV1cqLR6hEp3 | −0.043 | +0.010 | −0.033 | +0.033 | +0.085 |
| BV1kYLR6AEXv | +0.016 | +0.021 | +0.013 | +0.014 | +0.013 |
| BV1UBjq6fEgb | +0.033 | +0.025 | +0.028 | +0.041 | −0.010 |
| BV1ySjz6FEzD | +0.009 | +0.017 | +0.025 | +0.022 | +0.087 |
| BV1dwjP6LECU | −0.006 | −0.000 | +0.003 | +0.013 | **+0.419** |
| **均值** | **+0.002** | +0.014 | +0.007 | **+0.025** | +0.119 |
| 同号为正 | 3/5 | 4/5 | 4/5 | **5/5** | 4/5 |

**原生输入下重验（唯一在生产条件下成立的一组）：**

| 素材 | 空白对照 | 22050 |
| --- | ---: | ---: |
| BV1kYLR6AEXv | +0.000 | −0.007 |
| BV1UBjq6fEgb | +0.026 | +0.029 |
| BV1ySjz6FEzD | −0.009 | −0.010 |
| BV1cqLR6hEp3 | +0.007 | +0.049 |
| BV1dwjP6LECU | −0.002 | +0.007 |
| **均值** | **+0.004** | **+0.014** |
| 同号为正 | 2/5 | **3/5** |

旧输入下判 22.05k「系统性退化」的唯一依据是 5/5 的符号一致性。换到正确输入，
**符号一致性消失（3/5 就是掷硬币）**，均值降到 +0.014。

### 旧 ogg 缓存在 44.1 kHz 下没有代价

同素材同参照，ogg 输入与原生输入的 44.1k 臂：+0.018 / +0.021 / +0.002 / −0.040 / +0.007，
**均值差 +0.002**，方向还略偏 ogg。与「单声道在 44.1k 下丢 0 秒」互相印证：模型在自己的
采样率上对单声道与带限输入是稳健的。那批缓存**没有在损害生产分离质量**。

## 怎么重跑

`FINESUB_DATA_ROOT` 指向持有 `assets/ data/ out/` 的 checkout（worktree 没有自己的），
`FINESUB_RATE_WORK` 指向产物目录（默认 `out/separator-rate`）。

```bash
export FINESUB_DATA_ROOT=../asr-playground        # 从 worktree 跑时才需要
export FINESUB_RATE_WORK=out/separator-rate

bash tools/separator_rate/run/screen.sh           # 旧 ogg 输入，三档分离
bash tools/separator_rate/run/downstream.sh       # 对应下游
bash tools/separator_rate/run/null_arm.sh         # 空白对照（必须）
bash tools/separator_rate/run/extract_native.sh   # 从 .mp4 抽原生音轨
bash tools/separator_rate/run/native_screen.sh    # 原生输入，四档分离
bash tools/separator_rate/run/native_down.sh      # 原生下游 44100/22050/null

python tools/separator_rate/dropout.py  <基线.flac> <臂.flac>...
python tools/separator_rate/vad_verdict.py
python tools/separator_rate/cer.py
python tools/separator_rate/boundaries.py
python tools/separator_rate/native_analysis.py
```

单臂手动跑：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m tools.separator_benchmark assets/bilibili/BV1cqLR6hEp3.ogg out/x.flac `
  --mode amp --gpu-budget-gb 4 --model-sample-rate 22050 --result out/x.json
```

再把 `x.flac` 复制成某个 `<stem>-vocal.flac`、用 `finesub.pipeline --stage raw-srt` 走下游
（`resolve_vocal_audio` 在没有 `.ogg` 时会回落到 `.flac`）。空白对照把 `--model-sample-rate`
换成 `--block-seconds 120`。

本轮的中间产物（约 2GB）**未保留**。
