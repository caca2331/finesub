---
id: filler-sandwich-boundary
kind: mini
applies: [capableB, capableC, basicA, basicB]
teach: 三源仅限 filler 三明治；4 字修饰词不是纯语气词，同形不同判
---
filler 三明治三源特例（唯一允许的三源形态：两段正句碎片夹一个 ≤{thr:filler_max_chars} 字纯语气词/口吃碎片；合计 ≤{thr:sandwich_max_seconds} 秒、合并后 ≤{thr:sandwich_max_chars} 字，不适用 {thr:absolute_chars} 字放宽）：

<input>
head    | 300.0 | 1.6 | うちらが知ってるのは
filler  | 301.8 | 0.6 | えっと
tail    | 302.5 | 1.4 | 少年期だけなんだよね
lead2   | 305.0 | 1.1 | それは
modifier| 306.3 | 0.9 | ちょっと
tail2   | 307.3 | 1.5 | 違う気がするんだよね
</input>

<output>
sub | head,filler,tail | AUTO | AUTO | うちらが知ってるのは、えっと、少年期だけなんだよね | 我们知道的，那个，就只有少年期 | median | AUTO | filler三明治：中间源为{thr:filler_max_chars}字纯语气词，合计{calc:span:head,tail}秒
sub | lead2 | AUTO | AUTO | それは | 那是…… | median | AUTO | 与后两源全并需吸收「ちょっと」——4字修饰词不是纯语气词，拒绝三源
sub | modifier,tail2 | AUTO | AUTO | ちょっと違う気がするんだよね | 总觉得有点不对呢 | high | AUTO | 两源合并，同一句
</output>

对照要点：局部序号 {ref:filler} 是纯垫词（≤{thr:filler_max_chars} 字）→ 三源成立；{ref:modifier}「ちょっと」是修饰实词（4 字）→ 三源不成立，改为 {ref:lead2} 独立 + {ref:modifier},{ref:tail2} 两源。
