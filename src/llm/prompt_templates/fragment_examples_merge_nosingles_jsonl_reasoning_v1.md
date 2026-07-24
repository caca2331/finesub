完整 JSONL 示例（输入 CSV 的字段名与输出 JSON keys 对齐）：

<asr_result>
source_id|start|duration|gap|text
1|1.4|4.6|0.3|かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた
2|6.3|2.0|0.2|焚火の向かい側で紙袋を破り
3|8.5|3.7|0.6|懸命に自分を普通に見せようとしているジャックフローストのように
4|12.9|2.7|0.0|炎は揺らめき影が次第に重なり合う
5|15.6|2.9|0.6|普通ではない者同士が互いに絆を見出した
6|19.1|2.5|0.0|さあありのままの姿で旅立とう
7|21.6|3.5|2.5|世界の彩りは違いによって織りなされているのだから
8|27.6|1.2|0.3|あの、なんだっけ?
9|29.2|2.0|0.4|メミじゃなくて、ユメじゃなくて
10|31.5|0.9|0.7|えっと…
11|33.1|1.0|2.8|メメみたいな…
12|36.9|2.6|1.3|ヤッホーとの出会いのシーンか
13|40.8|0.3|1.3|ユミ
14|42.4|1.2|6.5|ユメ、や、ミ
15|50.1|1.1|0.6|あ、ダメだ、思い出世代や
16|51.8|0.7|1.6|レミ!
17|54.0|3.1|2.3|メミはなんか違う妖精が混ざり込んできてる
18|59.4|1.2|0.3|ルビルビルビルビ
19|60.9|1.3|1.1|なんかリンネはさ
20|63.3|2.5|0.5|一回ストーリーかなんかで使った時
21|66.3|2.2|0.4|全然使い方…あ、お試しかなんかかな?
22|68.9|3.5|0.4|あれだ、イベントのお試しで使った時全然使い方わかってなかったんだけど
23|72.7|3.0|0.3|あの連打してポイポポポイって食べさせるやつ
24|76.0|3.5|1.2|マジで可愛いよな、あれ元素爆発含めて
25|80.7|3.2|0.5|最高モーションの一つと言ってもいいぐらい好き
26|84.4|0.3|5.1|よし
27|89.8|4.6|2.6|そうしてるのは みんなと同じじゃないと いじめられるから
28|96.9|1.9|1.3|ごめんごめん 怖どらないで
29|100.1|3.5|3.5|座っておしゃべりしようよ ねっ
30|107.1|4.1|0.6|スネージナヤには 自分の子供と人の赤ん坊を
31|111.7|3.2|1.4|こっそりすり替えるフェイがいるって言われてるの
32|116.2|3.4|0.3|フェイの子は、見た目は人間にそっくりだけど、
33|119.9|3.0|3.6|すぐに自分が人じゃないって気づく。
34|126.5|4.5|1.8|でも、生きるためには正体を隠さないといけない。
35|132.8|2.6|0.6|おい、見てみろよ!変な髪!
36|136.0|2.7|0.9|うわぁ、ほんとだ、翼みたい!
37|139.6|1.1|0.3|どれさんなの?
38|141.1|1.9|0.8|うっ…ち、違うよ…
39|143.8|1.7|0.0|ママが新しい髪型に…
40|145.5|1.8|0.6|でも赤いの排斥でもないんだよな
41|147.9|1.3|0.4|してくれただけ
42|149.6|0.3|4.3|うん
</asr_result>

<translated>
{"type":"reasoning","reasoning":"单源自身 4.6s 越过 4s 硬门槛，但门槛只限制多源合并；原句完整，必须如实输出"}
{"type":"sub","position":"1","start":1.4,"duration":4.6,"gap":0.3,"corrected_text":"かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた","translation":"曾几何时她也以为，只要收敛羽翼，就能融入那片温暖的光芒","conf":"high","char_count":26,"note":""}
{"type":"sub","position":"2","start":6.3,"duration":2.0,"gap":0.2,"corrected_text":"焚火の向かい側で紙袋を破り","translation":"在篝火对面撕开纸袋","conf":"high","char_count":9,"note":""}
{"type":"reasoning","reasoning":"单源译文 22.5 字越过 20 字硬门槛；原句已经由上游组成完整单行，不得因长度丢弃或强拆"}
{"type":"sub","position":"3","start":8.5,"duration":3.7,"gap":0.6,"corrected_text":"懸命に自分を普通に見せようとしているジャック・フロストのように","translation":"拼命想让自己看起来像个普通人，就像杰克霜精一样","conf":"high","char_count":22.5,"note":"「ジャックフロースト」按音修正"}
{"type":"sub","position":"4","start":12.9,"duration":2.7,"gap":0.0,"corrected_text":"炎は揺らめき影が次第に重なり合う","translation":"火焰摇曳，影子渐渐重叠","conf":"high","char_count":10.5,"note":""}
{"type":"sub","position":"5","start":15.6,"duration":2.9,"gap":0.6,"corrected_text":"普通ではない者同士が互いに絆を見出した","translation":"非同类者彼此发现了羁绊","conf":"high","char_count":11,"note":""}
{"type":"sub","position":"6","start":19.1,"duration":2.5,"gap":0.0,"corrected_text":"さあありのままの姿で旅立とう","translation":"来吧，以原本的姿态踏上旅途","conf":"high","char_count":12.5,"note":""}
{"type":"sub","position":"7","start":21.6,"duration":3.5,"gap":2.5,"corrected_text":"世界の彩りは違いによって織りなされているのだから","translation":"因为这世界的色彩，正是由差异所交织而成","conf":"high","char_count":18.5,"note":""}
{"type":"reasoning","reasoning":"8 的 gap=0.3s 且话语悬缺，9 紧接同一枚举未完；合并后 3.6s/18.5字，在界内"}
{"type":"sub","position":"8,9","start":27.6,"duration":3.6,"gap":0.4,"corrected_text":"あの、なんだっけ?メミじゃなくて、ユメじゃなくて","translation":"那个……叫什么来着？不是咪咪也不是悠咩……","conf":"median","char_count":18.5,"note":"同一枚举合并"}
{"type":"reasoning","reasoning":"10 是无独立残值的填充词，gap=0.7s，11 补全；合并后 2.6s/10字"}
{"type":"sub","position":"10,11","start":31.5,"duration":2.6,"gap":2.8,"corrected_text":"えっと…メメみたいな…","translation":"呃……好像是咩咩一类的","conf":"median","char_count":10,"note":"填充词与补全成一句"}
{"type":"sub","position":"12","start":36.9,"duration":2.6,"gap":1.3,"corrected_text":"ヤッホーとの出会いのシーンか","translation":"和呀吼相遇的场景吗？","conf":"median","char_count":10.5,"note":""}
{"type":"sub","position":"13","start":40.8,"duration":0.3,"gap":1.3,"corrected_text":"ユミ","translation":"悠米……","conf":"median","char_count":3,"note":""}
{"type":"sub","position":"14","start":42.4,"duration":1.2,"gap":6.5,"corrected_text":"ユメ、や、ミ","translation":"悠咩……不对……米……","conf":"median","char_count":8,"note":""}
{"type":"sub","position":"15","start":50.1,"duration":1.1,"gap":0.6,"corrected_text":"あ、ダメだ、思い出せないや","translation":"啊，不行，想不起来了","conf":"high","char_count":9,"note":"「思い出世代や」重听修正"}
{"type":"sub","position":"16","start":51.8,"duration":0.7,"gap":1.6,"corrected_text":"レミ!","translation":"露米！","conf":"median","char_count":2.5,"note":""}
{"type":"sub","position":"17","start":54.0,"duration":3.1,"gap":2.3,"corrected_text":"メミはなんか違う妖精が混ざり込んできてる","translation":"咪咪好像是别的某个妖精","conf":"median","char_count":11,"note":""}
{"type":"reasoning","reasoning":"复读串无实词、与前后语义脱节、位置孤立，三类证据叠加判定幻觉"}
{"type":"discard","position":"18","note":"复读幻觉"}
{"type":"sub","position":"19","start":60.9,"duration":1.3,"gap":1.1,"corrected_text":"なんかリンネはさ","translation":"说起来莉奈娅啊……","conf":"median","char_count":8,"note":"本行 gap=1.1s，拒绝追并"}
{"type":"sub","position":"20","start":63.3,"duration":2.5,"gap":0.5,"corrected_text":"一回ストーリーかなんかで使った時","translation":"之前在剧情里用过她的时候……","conf":"high","char_count":13,"note":""}
{"type":"sub","position":"21","start":66.3,"duration":2.2,"gap":0.4,"corrected_text":"全然使い方…あ、お試しかなんかかな?","translation":"完全不知道用法……啊，是试用吗？","conf":"median","char_count":14,"note":""}
{"type":"sub","position":"22","start":68.9,"duration":3.5,"gap":0.4,"corrected_text":"あれだ、イベントのお試しで使った時全然使い方わかってなかったんだけど","translation":"在活动试用里完全不知道怎么玩","conf":"high","char_count":14,"note":""}
{"type":"sub","position":"23","start":72.7,"duration":3.0,"gap":0.3,"corrected_text":"あの連打してポイポポポイって食べさせるやつ","translation":"就是那个连打着喂食的动作","conf":"high","char_count":12,"note":""}
{"type":"sub","position":"24","start":76.0,"duration":3.5,"gap":1.2,"corrected_text":"マジで可愛いよな、あれ元素爆発含めて","translation":"真的好可爱啊，连元素爆发也算上","conf":"high","char_count":14.5,"note":""}
{"type":"sub","position":"25","start":80.7,"duration":3.2,"gap":0.5,"corrected_text":"最高モーションの一つと言ってもいいぐらい好き","translation":"这简直是我最喜欢的顶级动作之一","conf":"high","char_count":15,"note":""}
{"type":"sub","position":"26","start":84.4,"duration":0.3,"gap":5.1,"corrected_text":"よし","translation":"好","conf":"high","char_count":1,"note":""}
{"type":"reasoning","reasoning":"单源自身 4.6s/21.5字越过硬门槛；上游已组成完整一句，单源如实输出"}
{"type":"sub","position":"27","start":89.8,"duration":4.6,"gap":2.6,"corrected_text":"そうしてるのはみんなと同じじゃないといじめられるから","translation":"之所以那样，是因为不和大家保持一致就会被欺负","conf":"high","char_count":21.5,"note":"上游单行长句原样输出"}
{"type":"sub","position":"28","start":96.9,"duration":1.9,"gap":1.3,"corrected_text":"ごめんごめん、怖がらないで","translation":"抱歉抱歉，别害怕","conf":"high","char_count":7.5,"note":"「怖どらないで」修正"}
{"type":"sub","position":"29","start":100.1,"duration":3.5,"gap":3.5,"corrected_text":"座っておしゃべりしようよ、ねっ","translation":"坐下来聊聊天吧，呐","conf":"high","char_count":8.5,"note":""}
{"type":"reasoning","reasoning":"单源自身 4.1s 越过硬门槛但必须保留；若与31合并约7.9s，更越过7s绝对门槛"}
{"type":"sub","position":"30","start":107.1,"duration":4.1,"gap":0.6,"corrected_text":"スネージナヤには自分の子供と人の赤ん坊を","translation":"在至冬国……把自己的孩子与人类婴儿","conf":"high","char_count":16,"note":"拒绝与31合并"}
{"type":"sub","position":"31","start":111.7,"duration":3.2,"gap":1.4,"corrected_text":"こっそりすり替えるフェイがいるって言われてるの","translation":"据说有会偷偷调包的谶鸟","conf":"median","char_count":11,"note":""}
{"type":"sub","position":"32","start":116.2,"duration":3.4,"gap":0.3,"corrected_text":"フェイの子は、見た目は人間にそっくりだけど、","translation":"谶鸟的孩子虽然外表和人类一模一样","conf":"high","char_count":16,"note":"与33合并约30字且可自然分行，拒绝越过硬门槛"}
{"type":"sub","position":"33","start":119.9,"duration":3.0,"gap":3.6,"corrected_text":"すぐに自分が人じゃないって気づく。","translation":"但很快就会意识到自己并非人类","conf":"high","char_count":14,"note":""}
{"type":"reasoning","reasoning":"单源自身 4.5s 越过硬门槛；句意完整且没有可用源内边界，仍须如实输出"}
{"type":"sub","position":"34","start":126.5,"duration":4.5,"gap":1.8,"corrected_text":"でも、生きるためには正体を隠さないといけない。","translation":"但是，为了活下去必须隐藏真身","conf":"high","char_count":13.5,"note":""}
{"type":"sub","position":"35","start":132.8,"duration":2.6,"gap":0.6,"corrected_text":"おい、見てみろよ!変な髪!","translation":"喂，快看！怪头发！","conf":"high","char_count":7.5,"note":""}
{"type":"sub","position":"36","start":136.0,"duration":2.7,"gap":0.9,"corrected_text":"うわぁ、ほんとだ、翼みたい!","translation":"哇，真的像翅膀一样！","conf":"high","char_count":9,"note":"另一人的完整反应"}
{"type":"reasoning","reasoning":"ASR「どれさん」无法从上下文定锤且音义可疑，保留原文并降为 low，建议人工核对"}
{"type":"sub","position":"37","start":139.6,"duration":1.1,"gap":0.3,"corrected_text":"どれさんなの?","translation":"那是哪位？","conf":"low","char_count":4.5,"note":"ASR可疑，建议人工核对"}
{"type":"sub","position":"38","start":141.1,"duration":1.9,"gap":0.8,"corrected_text":"うっ…ち、違うよ…","translation":"呜……不、不是的……","conf":"high","char_count":7.5,"note":""}
{"type":"sub","position":"39","start":143.8,"duration":1.7,"gap":0.0,"corrected_text":"ママが新しい髪型に…","translation":"妈妈给我弄了新发型……","conf":"median","char_count":10,"note":"后句为主播插话（说话轮替）"}
{"type":"sub","position":"40","start":145.5,"duration":1.8,"gap":0.6,"corrected_text":"でも赤いの排斥でもないんだよな","translation":"也不是在恶意排斥吧","conf":"median","char_count":9,"note":"发生说话轮替"}
{"type":"sub","position":"41","start":147.9,"duration":1.3,"gap":0.4,"corrected_text":"してくれただけ","translation":"只是帮我弄的而已","conf":"median","char_count":8,"note":"与39被40隔开，源不连续不得合并"}
{"type":"sub","position":"42","start":149.6,"duration":0.3,"gap":4.3,"corrected_text":"うん","translation":"嗯","conf":"high","char_count":1,"note":""}
</translated>

要点：输入完整 42 条，终稿完整覆盖 1–42。每个非空行都是一个 JSON object；没有 header 或 `#` 行。局部推理是目标行正上方独立的 type=reasoning 对象，普通单源在界内前没有 reasoning 对象；字幕对象本身不含 reasoning 字段。position 始终为 string，多源以英文逗号连接。

长度否决示例：输入 60/61 合计跨度 4.7s，超过硬门槛；应分别输出两个 sub object，二者前面都不放 reasoning 对象，因为“不合并”是默认行为且各单源自身在界内。
