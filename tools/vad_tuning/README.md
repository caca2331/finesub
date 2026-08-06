# vad_tuning — VAD 调参与 silero 结合探索

**维护策略：按需**。探索现场，不进默认测试，不改生产代码。结论读 [`FINDINGS.md`](FINDINGS.md)。

## 回答什么

现有 energy VAD 是手调的，能不能更优？能不能和 silero 结合拿到更好的 recall / 精度？

**最终答案（2026-08-05,已进生产,FINDINGS 附录 X/Z/AB）**：recall 天花板是真的,
但精度侧有实打实的收益——`-45` 绝对峰值底线 + exit-run 累加器 + pause hints 默认开;
voicing 门控 cap / ghost-drop / 无声 carve / 接缝恢复并入 opt-in `--vad-silero-assist`。
早期"padR 140→100 免费"的结论**已被端到端推翻**（附录 H）,勿引用。
中途多轮被否决的方向（换 floor 家族、无门控 cap、纯区间级指标决策）都记录在
FINDINGS 里,动 VAD 前先读附录 H/I/U/Y3——那是三次"区间级指标指错方向"的现场。

## 文件

| 文件 | 作用 |
| --- | --- |
| `backends.py` | 统一接口：生产 energy VAD / silero（frame 概率缓存 + 自实现迟滞后处理）/ 区间运算 |
| `energy_sweep.py` | 只重跑生产检测器的打分与负 padding，前半段（framing/噪声底）缓存；`verify()` 对照流式生产入口 |
| `hybrid.py` | `AdaptiveHead`（silero 确信无声时才裁头）与 `GuardedAggressive`（silero 兜底 union） |
| `refs.py` | 三套参考的加载，含 `load_valid_words`（去幻觉/漂移/复读/超长/标点） |
| `score.py` | lost / clipH / clipT / recall / pause_excl；两类错误从不合并 |
| `adaptive.py` | 分尺度（整段/内部/边界）重判 energy 保留但 silero 否掉的区域 |
| `floor_variants.py` | 底噪估计的替代实现：窗口目标值与追踪器各自可换，供消融 |
| `precision.py` | 两把替代 `lost` 的尺子：边界收紧度（需标注）、送入的非语音秒数（不需标注） |
| `onset_snap.py` | 逐区间把语音起点吸附到真实 onset（候选，未上线） |
| `vad_db_srt.py` | 把 VAD 内部量导成 SRT 供人眼看：加权能量 / 底噪两条 dB 轨（0.1s 一条、整数），`--intervals` 再出一份语音区间轨 |
| `v1`–`v33` | 按顺序的实验，文件头写明各自回答哪一问。要点：`v26` step0 切片 ASR 探针＋`v26b` 标注页、`v27` cap 家族、`v28` empty-real 链路诊断、`v29` ghost 验收、`v30/v30b` churn 噪声带标定、`v31` exit-run + 门控 cap、`v31b` 任意区间集注入真实生产链、`v32` 最终形态组合、`v33` partial carve + 接缝恢复 |
| `step0_labels/` | **人工标注（进 git）**：144 个争议片段的听审结论＋全特征 join，判据标定的依据 |

`vad_db_srt.py` 用法（两条轨分开出，叠在同一时间轴上看，纵向差就是检测器在卡的 SNR）：

```bash
python vad_db_srt.py --audio <vocal.flac> --outdir <放 SRT 的目录> --stem <名字> --intervals
# --floor legacy（默认，改动前的估计器）/ production（当前 energy.py 里的）
# --bucket 0.1  --agg median|mean|max|min
# --pad-right-ms 40   只影响区间轨，用来对照不同 padR
```

区间轨出的是**语音**区间（生产写的 SRT 是它的补集，非语音、标签一律 `""`）：
和词级 SRT 对齐的是语音区间，且每条标了 `#序号 时长`，编辑器里能分辨。

`v10`–`v13` 是底噪那一轮：`v10_quiet` 轻语队列 + 合成"完美分离"压力测试、
`v10_lost_dump` 看新丢的到底是什么词、`v11_floor_ab` 规则消融、`v11b_floor_snr`
噪声抑制诊断量、`v12_prod_check` 走流式入口的终检、`v13_ghost_snr` 重标 adaptive 阈值。

## 数据依赖

`disfluency_gold.json` 进 git；音频、`stable.json`、人工修正 SRT 都是本机的，
全部走命令行参数。见 `docs/data-index.md`。

## 注意

- silero 的 `read_audio` 在本机因 torchcodec/FFmpeg 加载失败，`backends.py` 自己用 librosa 读。
- `energy_sweep` 是非流式捷径，但**必须与生产完全一致**：它走 `_load_asr_audio_streamed`
  并从模块读取可扫参数（两者都曾出过 bug，见 FINDINGS D5）。扫描前务必看 `verify()` 输出，
  非 0 差异就别信扫描结论。
- `cached_tracks(path, cache_dir)` 缓存 framing 结果（几小时素材的重跑靠它）；
  **底噪不进缓存**——它正是被研究的对象，缓存会让旧估计器悄悄参与打分。
