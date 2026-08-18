---
id: length-veto-chars
kind: mini
applies: [capableB, capableC, basicA, basicB]
teach: 字数硬门槛否决合并
---
长度否决合并正例 2（合并后 char_count >{thr:hard_chars}）：

<input>
lead | 210.0 | 1.4 | だから結局その仕様変更は
tail | 211.5 | 1.6 | プレイヤー側からするとかなり厳しい選択になった
</input>

<output>
sub | lead | AUTO | AUTO | だから結局その仕様変更は | 所以那次规格改动 | high | AUTO | 合并后越过{thr:hard_chars}字硬门槛，拒绝合并
sub | tail | AUTO | AUTO | プレイヤー側からするとかなり厳しい選択になった | 对玩家来说变成了相当残酷的选择 | high | AUTO |
</output>
