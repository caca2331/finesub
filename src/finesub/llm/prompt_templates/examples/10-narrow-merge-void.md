---
id: narrow-merge-void
kind: mini
applies: [capableB, capableC, basicA, basicB]
headers: true
teach: 合并行先检查末源是否更宜留给后句；缩窄用 void 而非重排
---
缩窄合并反例（translated 内）：

<input>
head | 120.0 | 1.5 | 世界に溶け込もうと
mid  | 121.7 | 2.7 | その子はいっぱい時間をかけて
tail | 124.5 | 1.8 | 他の人の真似をしたけど
</input>

<output>
sub | head,mid | AUTO | AUTO | 世界に溶け込もうとその子はいっぱい時間をかけて | 为了融入世界，那孩子花了大量时间 | median | AUTO | 局部序号{ref:mid}更宜与{ref:tail}合并；缩窄范围 <void>
sub | head | AUTO | AUTO | 世界に溶け込もうと | 为了融入世界 | high | AUTO |
sub | mid,tail | AUTO | AUTO | その子はいっぱい時間をかけて他の人の真似をしたけど | 那孩子花了大量时间去模仿其他人 | high | AUTO | 缩窄后同一句
</output>
