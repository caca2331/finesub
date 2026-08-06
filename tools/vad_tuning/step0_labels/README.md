# step0 人工标注（2026-08-04）

144 个 VAD 争议片段的人工听审结论（真语音/听不清/语气词/幻觉/噪声抖动），
以及带全特征的 join 结果。采集方法与切片定义见 `../v26_step0.py` 与
FINDINGS 附录 T；这批标注是 −45 底线、ghost-drop、voicing 门控 cap 等
判据的标定依据，丢了要重标。

- `step0-labels.csv`：`clip,idx,label` 人工标注原始导出。
- `step0-joined.csv`：join 到 `regions.csv` 特征（能量/时长/silero/谱平坦度/
  从未解码占比/切片 ASR 文本/自动判定）后的完整表。

音频切片本身在本地 `tmp/vad-step0/<clip>/snips/`（可由 v26 重新生成，
只要 vocal 音频还在）。
