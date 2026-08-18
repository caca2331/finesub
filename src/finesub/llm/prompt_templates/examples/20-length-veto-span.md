---
id: length-veto-span
kind: mini
applies: [capableB, capableC, basicA, basicB]
teach: 跨度硬门槛否决合并；派生的合并跨度由 builder 算，不手写
---
长度否决合并正例 1（合并后跨度 >{thr:hard_seconds}s）：

<input>
lead | 200.0 | 2.2 | なんかあのイベント報酬がさ
tail | 202.4 | 2.8 | 思ったより地味でびっくりしたんだけど
</input>

<output>
sub | lead | AUTO | AUTO | なんかあのイベント報酬がさ | 那个活动奖励啊 | high | AUTO | 合并后约{calc:span:lead,tail}s，拒绝合并
sub | tail | AUTO | AUTO | 思ったより地味でびっくりしたんだけど | 比想象中朴素，有点吃惊 | high | AUTO |
</output>
