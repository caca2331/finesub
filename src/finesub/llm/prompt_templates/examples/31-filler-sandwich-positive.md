---
id: filler-sandwich-positive
kind: mini
applies: [capableB, capableC, basicA, basicB]
teach: 三明治正例；拆开会切断同一语义单元
---
filler 三明治三源正例（仅此一种三源合法：两段正句碎片夹一个 ≤{thr:filler_max_chars} 字纯语气词）：

<input>
head   | 250.0 | 1.2 | 世界に溶けこもうとその子は
filler | 251.3 | 0.4 | えっと
tail   | 251.8 | 1.8 | いっぱい時間をかけて
</input>

<output>
sub | head,filler,tail | AUTO | AUTO | 世界に溶けこもうとその子はいっぱい時間をかけて | 为了融入世界，那孩子花了大量时间 | median | AUTO | filler三明治，合计{calc:span:head,tail}s，未越 ≤{thr:sandwich_max_seconds}s/{thr:sandwich_max_chars}字
</output>

对照：局部序号 {ref:filler} 为无独立残值的「えっと」；拆开会切断同一语义单元。合并后仍须严守 ≤{thr:sandwich_max_seconds}s 且 ≤{thr:sandwich_max_chars} 字，不适用 {thr:absolute_chars} 字放宽。
