---
id: filler-sandwich-boundary
kind: mini
headers: true
output_headers: true
applies: [capableC]
may: [filler, tail, modifier, tail2]
single_pass_skip: [capableC]
teach: 三源仅限 filler 三明治；4 字修饰词不是纯语气词，同形不同判
---
filler 三明治三源特例（唯一允许的三源形态：两段正句碎片夹一个 ≤{thr:filler_max_chars} 字纯语气词/口吃碎片；跨度 ≤{thr:sandwich_max_seconds} 秒、合并后 ≤{thr:sandwich_max_chars} 字，不适用 {thr:absolute_chars} 字放宽）：

<input>
head    | 300.0 | 1.6 | 我们知道的
filler  | 301.8 | 0.6 | 呃
tail    | 302.5 | 1.3 | 只有小时候的事
lead2   | 305.0 | 1.1 | 这个嘛
modifier| 306.3 | 0.9 | 总觉得有点
tail2   | 307.3 | 1.5 | 不太对劲
</input>

<output>
sub | head,filler,tail | 我们知道的，呃，只有小时候的事 | 我们知道的，只有小时候的事 | median | filler三明治：中间源为{thr:filler_max_chars}字以内的纯语气词，跨度{calc:span:head,tail}秒；垫词只在 polished 删
sub | lead2 | 这个嘛 | 这个嘛 | median | 与后两源全并需吸收「总觉得有点」——5字修饰语不是纯语气词，拒绝三源
sub | modifier,tail2 | 总觉得有点不太对劲 | 总觉得有点不太对劲 | high | 两源合并，同一句
</output>

对照要点：局部序号 {ref:filler} 是纯垫词（≤{thr:filler_max_chars} 字）→ 三源成立，polished 删掉它；{ref:modifier}「总觉得有点」是修饰语（5 字）→ 三源不成立，改为 {ref:lead2} 独立 + {ref:modifier},{ref:tail2} 两源。
