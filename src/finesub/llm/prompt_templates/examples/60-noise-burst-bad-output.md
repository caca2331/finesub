---
id: noise-burst-bad-output
kind: bad-output
requires: media>=audio
applies: [capableB, capableC, basicA, basicB]
teach: 四源以上禁止；弱信息堆叠不构成同一句，场面描述不得进字幕
---
高噪声/情绪爆发反例（四源以上禁止；弱信息堆叠不构成可合并的同一句）：
输入块：

<input>
a1 | 410.4 | 1.1 | 啊
a2 | 411.8 | 1.0 | 啊
a3 | 413.0 | 1.0 | 啊
a4 | 414.2 | 1.0 | 啊
a5 | 415.4 | 1.0 | 啊
a6 | 416.6 | 1.0 | 啊
a7 | 417.8 | 1.0 | 啊
a8 | 419.0 | 1.0 | 啊
a9 | 420.2 | 1.0 | 啊
a10 | 421.4 | 1.0 | 啊
a11 | 422.6 | 1.0 | 啊
a12 | 426.5 | 1.0 | 啊
</input>

错误输出：
<output>
sub | a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 | AUTO | AUTO | 啊 啊 啊 啊 啊 啊 啊 啊 啊 啊 啊 啊 | 啊（惊讶声） | median | AUTO | 情绪爆发打包
</output>

错在：四源以上、跨度 {calc:span:a1,a12}s、弱信息重复不构成同一句、场面描述进入字幕。正确处置：$noisy_span_handling。
