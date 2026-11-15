---
id: narrow-merge-void
kind: mini
output_headers: true
applies: [capableC]
headers: true
may: [mid, tail]
teach: 合并行先检查末源是否更宜留给后句；缩窄用 void 而非重排
---
缩窄合并反例（translated 内）：

<input>
head | 120.0 | 1.5 | 为了融进这个世界
mid  | 121.7 | 2.2 | 那个孩子花了好多时间
tail | 124.0 | 1.4 | 去学别人的样子
</input>

<output>
sub | head,mid | 为了融进这个世界，那个孩子花了好多时间 | 为了融进这个世界，那个孩子花了好多时间 | median | 局部序号{ref:mid}更宜与{ref:tail}合并；缩窄范围 <void>
sub | head | 为了融进这个世界 | 为了融进这个世界 | high |
sub | mid,tail | 那个孩子花了好多时间去学别人的样子 | 那个孩子花了好多时间去学别人的样子 | high | 缩窄后同一句
</output>
