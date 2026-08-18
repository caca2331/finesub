---
id: main-oneshot
kind: main
headers: true
teach: 43 行完整场景；三个变体共享同一输入，输出按 overlay 分组，场景改一次即可
---
完整 oneshot。`<preceding_context>` 是只读前文（不属于本窗输出范围）。
场景：主播四月一日看莉奈娅 PV——旁白念白 → 认妖精名 → 试用手感 → 「被欺负」与至冬谶鸟节选。

<preceding>
p0 | -8.2 | 1.5 | さっきのボス硬すぎだろ
p1 | -6.4 | 1.2 | 回復買ってから行くか
</preceding>

<input>
s01 | 1.4 | 4.6 | かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた
s02 | 6.3 | 2.0 | 焚火の向かい側で紙袋を破り
s03 | 8.5 | 3.7 | 懸命に自分を普通に見せようとしているジャックフローストのように
s04 | 12.9 | 2.7 | 炎は揺らめき影が次第に重なり合う
s05 | 15.6 | 2.9 | 普通ではない者同士が互いに絆を見出した
s06 | 19.1 | 2.5 | さあありのままの姿で旅立とう
s07 | 21.6 | 3.5 | 世界の彩りは違いによって織りなされているのだから
s08 | 27.6 | 1.2 | あの、なんだっけ?
s09 | 29.2 | 2.0 | メミじゃなくて、ユメじゃなくて
s10 | 31.5 | 0.9 | えっと…
s11 | 33.1 | 1.0 | メメみたいな…
s12 | 36.9 | 2.6 | ヤッホーとの出会いのシーンか
s13 | 40.8 | 0.3 | ユミ
s14 | 42.4 | 1.2 | ユメ、や、ミ
s15 | 50.1 | 1.1 | あ、ダメだ、思い出世代や
s16 | 51.8 | 0.7 | レミ!
s17 | 54.0 | 3.1 | メミはなんか違う妖精が混ざり込んできてる
s18 | 59.4 | 1.2 | ルビルビルビルビ
s19 | 60.9 | 1.3 | なんかリンネはさ
s20 | 63.3 | 2.5 | 一回ストーリーかなんかで使った時
s21 | 66.3 | 2.2 | 全然使い方…あ、お試しかなんかかな?
s22 | 68.9 | 3.5 | あれだ、イベントのお試しで使った時全然使い方わかってなかったんだけど
s23 | 72.7 | 3.0 | あの連打してポイポポポイって食べさせるやつ
s24 | 76.0 | 3.5 | マジで可愛いよな、あれ元素爆発含めて
s25 | 80.7 | 3.2 | 最高モーションの一つと言ってもいいぐらい好き
s26 | 84.1 | 0.3 | んー
s27 | 84.4 | 0.3 | よし
s28 | 89.8 | 4.6 | そうしてるのは みんなと同じじゃないと いじめられるから
s29 | 96.9 | 1.9 | ごめんごめん 怖どらないで
s30 | 100.1 | 3.5 | 座っておしゃべりしようよ ねっ
s31 | 107.1 | 4.1 | スネージナヤには 自分の子供と人の赤ん坊を
s32 | 111.7 | 3.2 | こっそりすり替えるフェイがいるって言われてるの
s33 | 116.2 | 3.4 | フェイの子は、見た目は人間にそっくりだけど、
s34 | 119.9 | 3.0 | すぐに自分が人じゃないって気づく。
s35 | 126.5 | 4.5 | でも、生きるためには正体を隠さないといけない。
s36 | 132.8 | 2.6 | おい、見てみろよ!変な髪!
s37 | 136.0 | 2.7 | うわぁ、ほんとだ、翼みたい!
s38 | 139.6 | 1.1 | どれさんなの?
s39 | 141.1 | 1.9 | うっ…ち、違うよ…
s40 | 143.8 | 1.7 | ママが新しい髪型に…
s41 | 145.5 | 1.8 | でも赤いの排斥でもないんだよな
s42 | 147.9 | 1.3 | してくれただけ
s43 | 149.6 | 0.3 | うん
</input>

<output variant=nosingles lead="oneshot（直接输出 `<translated>` 终稿）——终稿：">
# 单源自身 {calc:dur:s01}s 越过 {thr:hard_seconds}s 硬门槛，但门槛只限制多源合并；原句完整，必须如实输出
sub | s01 | AUTO | AUTO | かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた | 曾几何时她也以为，只要收敛羽翼，就能融入那片温暖的光芒 | high | AUTO |
sub | s02 | AUTO | AUTO | 焚火の向かい側で紙袋を破り | 在篝火对面撕开纸袋 | high | AUTO |
# 单源译文 {calc:cc:s03} 字越过 {thr:hard_chars} 字硬门槛；原句已经由上游组成完整单行，不得因长度丢弃或强拆
sub | s03 | AUTO | AUTO | 懸命に自分を普通に見せようとしているジャック・フロストのように | 拼命想让自己看起来像个普通人，就像杰克霜精一样 | high | AUTO | 「ジャックフロースト」按音修正
sub | s04 | AUTO | AUTO | 炎は揺らめき影が次第に重なり合う | 火焰摇曳，影子渐渐重叠 | high | AUTO |
sub | s05 | AUTO | AUTO | 普通ではない者同士が互いに絆を見出した | 非同类者彼此发现了羁绊 | high | AUTO |
sub | s06 | AUTO | AUTO | さあありのままの姿で旅立とう | 来吧，以原本的姿态踏上旅途 | high | AUTO |
sub | s07 | AUTO | AUTO | 世界の彩りは違いによって織りなされているのだから | 因为这世界的色彩，正是由差异所交织而成 | high | AUTO |
# {ref:s08} 的 gap={calc:gap:s08}s 且话语悬缺，{ref:s09} 紧接同一枚举未完；合并后 {calc:span:s08,s09}s/{calc:cc:s08}字，在界内
sub | s08,s09 | AUTO | AUTO | あの、なんだっけ?メミじゃなくて、ユメじゃなくて | 那个……叫什么来着？不是咪咪也不是悠咩…… | median | AUTO | 18.5字<20硬门槛，同一枚举合并
# {ref:s10} 是无独立残值的填充词，gap={calc:gap:s10}s，{ref:s11} 补全；合并后 {calc:span:s10,s11}s/{calc:cc:s10}字
sub | s10,s11 | AUTO | AUTO | えっと…メメみたいな… | 呃……好像是咩咩一类的 | median | AUTO | 填充词与补全成一句
sub | s12 | AUTO | AUTO | ヤッホーとの出会いのシーンか | 和“呀吼”相遇的场景吗？ | median | AUTO |
sub | s13 | AUTO | AUTO | ユミ | 悠米…… | median | AUTO |
sub | s14 | AUTO | AUTO | ユメ、や、ミ | 悠咩……不对……米…… | median | AUTO |
sub | s15 | AUTO | AUTO | あ、ダメだ、思い出せないや | 啊，不行，想不起来了 | high | AUTO | 「思い出世代や」经$correction_basis修正
sub | s16 | AUTO | AUTO | レミ! | 露米！ | median | AUTO |
sub | s17 | AUTO | AUTO | メミはなんか違う妖精が混ざり込んできてる | 咪咪好像是别的某个妖精 | median | AUTO |
# 复读串「ルビルビルビルビ」无实词、与前后语义脱节、gap 孤立（前 {calc:gap:s17}s 后 {calc:gap:s18}s），三特征叠加判定幻觉
discard | s18 | 复读幻觉
sub | s19 | AUTO | AUTO | なんかリンネはさ | 说起来莉奈娅啊…… | median | AUTO | 本行 gap=1.1s，拒绝追并
sub | s20 | AUTO | AUTO | 一回ストーリーかなんかで使った時 | 之前在剧情里用过她的时候…… | high | AUTO |
sub | s21 | AUTO | AUTO | 全然使い方…あ、お試しかなんかかな? | 完全不知道用法……啊，是试用吗？ | median | AUTO |
sub | s22 | AUTO | AUTO | あれだ、イベントのお試しで使った時全然使い方わかってなかったんだけど | 在活动试用里完全不知道怎么玩 | high | AUTO |
sub | s23 | AUTO | AUTO | あの連打してポイポポポイって食べさせるやつ | 就是那个连打着喂食的动作 | high | AUTO |
sub | s24 | AUTO | AUTO | マジで可愛いよな、あれ元素爆発含めて | 真的好可爱啊，连元素爆发也算上 | high | AUTO |
sub | s25 | AUTO | AUTO | 最高モーションの一つと言ってもいいぐらい好き | 这简直是我最喜欢的顶级动作之一 | high | AUTO |
discard | s26 | 孤立填充词（无残值、不衔接），丢弃
sub | s27 | AUTO | AUTO | よし | 好 | high | AUTO | 独立语气词（有语用功能），完整保留
# 单源自身 {calc:dur:s28}s/{calc:cc:s28}字越过硬门槛；上游已组成完整一句，单源如实输出
sub | s28 | AUTO | AUTO | そうしてるのはみんなと同じじゃないといじめられるから | 之所以那样，是因为不和大家保持一致就会被欺负 | high | AUTO | 上游单行长句原样输出，单源不受20字门槛
sub | s29 | AUTO | AUTO | ごめんごめん、怖がらないで | 抱歉抱歉，别害怕 | high | AUTO | 「怖どらないで」修正
sub | s30 | AUTO | AUTO | 座っておしゃべりしようよ、ねっ | 坐下来聊聊天吧，呐 | high | AUTO |
# 单源自身 {calc:dur:s31}s 越过硬门槛但必须保留；若与 {ref:s32} 合并约 {calc:span:s31,s32}s，更越过 {thr:absolute_seconds}s 绝对门槛
sub | s31 | AUTO | AUTO | スネージナヤには自分の子供と人の赤ん坊を | 在至冬国……把自己的孩子与人类婴儿 | high | AUTO | 与32合并约7.9秒超7秒绝对门槛，拒绝合并
sub | s32 | AUTO | AUTO | こっそりすり替えるフェイがいるって言われてるの | 据说有会偷偷调包的谶鸟 | median | AUTO |
sub | s33 | AUTO | AUTO | フェイの子は、見た目は人間にそっくりだけど、 | 谶鸟的孩子虽然外表和人类一模一样 | high | AUTO | 与34合并约30字且可在「けど」处自然分行，拒绝越过硬门槛
sub | s34 | AUTO | AUTO | すぐに自分が人じゃないって気づく。 | 但很快就会意识到自己并非人类 | high | AUTO |
# 单源自身 {calc:dur:s35}s 越过硬门槛；句意完整且没有可用源内边界，仍须如实输出
sub | s35 | AUTO | AUTO | でも、生きるためには正体を隠さないといけない。 | 但是，为了活下去必须隐藏真身 | high | AUTO |
sub | s36 | AUTO | AUTO | おい、見てみろよ!変な髪! | 喂，快看！怪头发！ | high | AUTO |
sub | s37 | AUTO | AUTO | うわぁ、ほんとだ、翼みたい! | 哇，真的像翅膀一样！ | high | AUTO | 另一人的完整反应
# ASR「どれさん」无法从上下文定锤且音义可疑，保留原文并降为 low，建议人工核对
sub | s38 | AUTO | AUTO | どれさんなの? | 那是哪位？ | low | AUTO | ASR可疑，建议人工核对
sub | s39 | AUTO | AUTO | うっ…ち、違うよ… | 呜……不、不是的…… | high | AUTO |
sub | s40 | AUTO | AUTO | ママが新しい髪型に… | 妈妈给我弄了新发型…… | median | AUTO | 后句为主播插话（说话轮替）
sub | s41 | AUTO | AUTO | でも赤いの排斥でもないんだよな | 也不是在恶意排斥吧 | median | AUTO | 发生说话轮替
sub | s42 | AUTO | AUTO | してくれただけ | 只是帮我弄的而已 | median | AUTO | 与40被41隔开，源不连续不得合并
sub | s43 | AUTO | AUTO | うん | 嗯 | high | AUTO |
</output>

<output variant=basic tag=singles lead="两阶段完整 oneshot（先 `<singles>` 一一对应，再 `<translated>` 保持保守 1:1；无 `plan`）。第一块（示例输入恰有 43 条，所以 singles 恰好 43 行——示例行数只由示例输入决定，与你的任务窗口无关；每行单源 $output_column_count 列；char_count 独立；note 用具体方向/数值描述 gap）：">
sub | s01 | AUTO | AUTO | かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた | 曾几何时她也以为，只要收敛羽翼，就能融入那片温暖的光芒 | high | AUTO | 因为本窗首条旁白长句已完整，宜独立
sub | s02 | AUTO | AUTO | 焚火の向かい側で紙袋を破り | 在篝火对面撕开纸袋 | high | AUTO | 因为前一行 gap=0.3s，但两句各自完整，宜独立
sub | s03 | AUTO | AUTO | 懸命に自分を普通に見せようとしているジャック・フロストのように | 拼命想让自己看起来像个普通人，就像杰克霜精一样 | high | AUTO | 「ジャックフロースト」按音修正；上游已并成单行的长句照常输出，宜独立
sub | s04 | AUTO | AUTO | 炎は揺らめき影が次第に重なり合う | 火焰摇曳，影子渐渐重叠 | high | AUTO | 因为前一行 gap=0.6s 且本句完整，宜独立
sub | s05 | AUTO | AUTO | 普通ではない者同士が互いに絆を見出した | 非同类者彼此发现了羁绊 | high | AUTO | 因为前一行 gap=0.0s，但本句是完整陈述，宜独立
sub | s06 | AUTO | AUTO | さあありのままの姿で旅立とう | 来吧，以原本的姿态踏上旅途 | high | AUTO | 因为前一行 gap=0.6s 且本句是祈使收束，宜独立
sub | s07 | AUTO | AUTO | 世界の彩りは違いによって織りなされているのだから | 因为这世界的色彩，正是由差异所交织而成 | high | AUTO | 因为前一行 gap=0.0s，但两句都完整，宜独立
sub | s08 | AUTO | AUTO | あの、なんだっけ? | 那个……叫什么来着？ | median | AUTO | 因为本行 gap=0.3s 且话语悬缺，视情况可向后合并
sub | s09 | AUTO | AUTO | メミじゃなくて、ユメじゃなくて | 不是咪咪也不是悠咩…… | median | AUTO | 因为前一行 gap=0.3s 且同一枚举未完，宜与前一句合并
sub | s10 | AUTO | AUTO | えっと… | 那个…… | median | AUTO | 因为本行 gap=0.7s，仍是等待后句补全的填充词，视情况可向后合并
sub | s11 | AUTO | AUTO | メメみたいな… | 好像是咩咩的…… | median | AUTO | 因为前一行 gap=0.7s 且本行补全填充词，宜与前一句合并
sub | s12 | AUTO | AUTO | ヤッホーとの出会いのシーンか | 和“呀吼”相遇的场景吗？ | median | AUTO | 因为前一行 gap=2.8s 且本句疑问已完结，宜独立
sub | s13 | AUTO | AUTO | ユミ | 悠米…… | median | AUTO | 因为本行 gap=1.3s，是独立试名改口，宜独立
sub | s14 | AUTO | AUTO | ユメ、や、ミ | 悠咩……不对……米…… | median | AUTO | 因为前一行 gap=1.3s，另起一次试名，宜独立
sub | s15 | AUTO | AUTO | あ、ダメだ、思い出せないや | 啊，不行，想不起来了 | high | AUTO | 「思い出世代や」经$correction_basis修正为「思い出せないや」；前一行 gap=6.5s 且本句完整，宜独立
sub | s16 | AUTO | AUTO | レミ! | 露米！ | median | AUTO | 因为前一行 gap=0.6s，但这是独立呼名定锤，宜独立
sub | s17 | AUTO | AUTO | メミはなんか違う妖精が混ざり込んできてる | 咪咪好像是别的某个妖精 | median | AUTO | 因为前一行 gap=1.6s 且本句完整，宜独立
sub | s18 | AUTO | AUTO | ルビルビルビルビ | ルビルビルビルビ | low | AUTO | 因为复读串疑似幻觉，宜丢弃
sub | s19 | AUTO | AUTO | なんかリンネはさ | 说起来莉奈娅啊…… | median | AUTO | 因为本行 gap=1.1s，不应跨长停顿追并后句，宜独立
sub | s20 | AUTO | AUTO | 一回ストーリーかなんかで使った時 | 之前在剧情里用过她的时候…… | high | AUTO | 因为本行 gap=0.5s 且话语未完，视情况可向后合并
sub | s21 | AUTO | AUTO | 全然使い方…あ、お試しかなんかかな? | 完全不知道用法……啊，是试用吗？ | median | AUTO | 因为前一行 gap=0.5s，但本行自我修正后已收束，宜独立
sub | s22 | AUTO | AUTO | あれだ、イベントのお試しで使った時全然使い方わかってなかったんだけど | 在活动试用里完全不知道怎么玩 | high | AUTO | 因为本句是完整回忆，宜独立
sub | s23 | AUTO | AUTO | あの連打してポイポポポイって食べさせるやつ | 就是那个连打着喂食的动作 | high | AUTO | 因为前一行 gap=0.4s，但本句另述动作且完整，宜独立
sub | s24 | AUTO | AUTO | マジで可愛いよな、あれ元素爆発含めて | 真的好可爱啊，连元素爆发也算上 | high | AUTO | 上游已并入的完整赞叹，宜独立
sub | s25 | AUTO | AUTO | 最高モーションの一つと言ってもいいぐらい好き | 这简直是我最喜欢的顶级动作之一 | high | AUTO | 因为前一行 gap=1.2s 且本句可独立成立，宜独立
sub | s26 | AUTO | AUTO | んー | 嗯…… | low | AUTO | 因为孤立填充词无残值，即使与后句 gap=0.0 也不并入，宜丢弃
sub | s27 | AUTO | AUTO | よし | 好 | high | AUTO | 因为本行 gap=5.1s 且是独立语气词（有语用功能），完整保留，宜独立
sub | s28 | AUTO | AUTO | そうしてるのはみんなと同じじゃないといじめられるから | 之所以那样，是因为不和大家保持一致就会被欺负 | high | AUTO | 上游已把因果长句并成单行（行内空格去除），单源不受20字软门槛限制，宜独立
sub | s29 | AUTO | AUTO | ごめんごめん、怖がらないで | 抱歉抱歉，别害怕 | high | AUTO | 「怖どらないで」修正为「怖がらないで」；本句是完整安抚，宜独立
sub | s30 | AUTO | AUTO | 座っておしゃべりしようよ、ねっ | 坐下来聊聊天吧，呐 | high | AUTO | 因为前一行 gap=1.3s 且本句是完整邀约，宜独立
sub | s31 | AUTO | AUTO | スネージナヤには自分の子供と人の赤ん坊を | 在至冬国……把自己的孩子与人类婴儿 | high | AUTO | 因为本行 gap=0.6s 且宾语未完，但与后句合并约7.9秒超7秒硬上限，宜独立
sub | s32 | AUTO | AUTO | こっそりすり替えるフェイがいるって言われてるの | 据说有会偷偷调包的谶鸟 | median | AUTO | 因为前一行 gap=0.6s 且补全同一句传闻，但合并超7秒，宜独立
sub | s33 | AUTO | AUTO | フェイの子は、見た目は人間にそっくりだけど、 | 谶鸟的孩子虽然外表和人类一模一样 | high | AUTO | 因为本行 gap=0.3s 且让步句未完，视情况可向后合并
sub | s34 | AUTO | AUTO | すぐに自分が人じゃないって気づく。 | 但很快就会意识到自己并非人类 | high | AUTO | 因为前一行 gap=0.3s 且补全同一句，宜与前一句合并
sub | s35 | AUTO | AUTO | でも、生きるためには正体を隠さないといけない。 | 但是，为了活下去必须隐藏真身 | high | AUTO | 上游已把转折词并入主句，宜独立
sub | s36 | AUTO | AUTO | おい、見てみろよ!変な髪! | 喂，快看！怪头发！ | high | AUTO | 因为前一行 gap=1.8s 且同一轮呼喊已并成单行，宜独立
sub | s37 | AUTO | AUTO | うわぁ、ほんとだ、翼みたい! | 哇，真的像翅膀一样！ | high | AUTO | 因为前一行 gap=0.6s，但这是另一人的完整反应，宜独立
sub | s38 | AUTO | AUTO | どれさんなの? | 那是哪位？ | low | AUTO | 因为本句 ASR 可疑且是独立问句，宜独立
sub | s39 | AUTO | AUTO | うっ…ち、違うよ… | 呜……不、不是的…… | high | AUTO | 因为前一行 gap=0.3s 且是独立答句，宜独立
sub | s40 | AUTO | AUTO | ママが新しい髪型に… | 妈妈给我弄了新发型…… | median | AUTO | 因为本行 gap=0.0s，但后句是主播插话（说话轮替），宜独立
sub | s41 | AUTO | AUTO | でも赤いの排斥でもないんだよな | 也不是在恶意排斥吧 | median | AUTO | 因为前一行 gap=0.0s，但发生说话轮替且本句完整，宜独立
sub | s42 | AUTO | AUTO | してくれただけ | 只是帮我弄的而已 | median | AUTO | 因为与39同为角色台词、被40的主播插话隔开，源不连续不得合并，宜独立
sub | s43 | AUTO | AUTO | うん | 嗯 | high | AUTO | 因为本行 gap=4.3s 且是独立应答，宜独立
</output>

<output variant=basic lead="第二块——终稿（保守 1:1，仅词中接回可两源）：">
sub | s01 | AUTO | AUTO | かつて彼女も翼を隠せばあの温かな光に溶け込めると思っていた | 曾几何时她也以为，只要收敛羽翼，就能融入那片温暖的光芒 | high | AUTO |
sub | s02 | AUTO | AUTO | 焚火の向かい側で紙袋を破り | 在篝火对面撕开纸袋 | high | AUTO |
sub | s03 | AUTO | AUTO | 懸命に自分を普通に見せようとしているジャック・フロストのように | 拼命想让自己看起来像个普通人，就像杰克霜精一样 | high | AUTO |
sub | s04 | AUTO | AUTO | 炎は揺らめき影が次第に重なり合う | 火焰摇曳，影子渐渐重叠 | high | AUTO |
sub | s05 | AUTO | AUTO | 普通ではない者同士が互いに絆を見出した | 非同类者彼此发现了羁绊 | high | AUTO |
sub | s06 | AUTO | AUTO | さあありのままの姿で旅立とう | 来吧，以原本的姿态踏上旅途 | high | AUTO |
sub | s07 | AUTO | AUTO | 世界の彩りは違いによって織りなされているのだから | 因为这世界的色彩，正是由差异所交织而成 | high | AUTO |
sub | s08 | AUTO | AUTO | あの、なんだっけ? | 那个……叫什么来着？ | median | AUTO | 1:1 模式，不执行判断型合并
sub | s09 | AUTO | AUTO | メミじゃなくて、ユメじゃなくて | 不是咪咪也不是悠咩…… | median | AUTO | 1:1 模式，不执行 singles 的合并倾向
sub | s10 | AUTO | AUTO | えっと… | 那个…… | median | AUTO | 填充词不附着，保持独立
sub | s11 | AUTO | AUTO | メメみたいな… | 好像是咩咩的…… | median | AUTO | 1:1 模式，不执行 singles 的合并倾向
sub | s12 | AUTO | AUTO | ヤッホーとの出会いのシーンか | 和”呀吼”相遇的场景吗？ | median | AUTO |
sub | s13 | AUTO | AUTO | ユミ | 悠米…… | median | AUTO |
sub | s14 | AUTO | AUTO | ユメ、や、ミ | 悠咩……不对……米…… | median | AUTO |
sub | s15 | AUTO | AUTO | あ、ダメだ、思い出せないや | 啊，不行，想不起来了 | high | AUTO |
sub | s16 | AUTO | AUTO | レミ! | 露米！ | median | AUTO |
sub | s17 | AUTO | AUTO | メミはなんか違う妖精が混ざり込んできてる | 咪咪好像是别的某个妖精 | median | AUTO |
discard | s18 | 复读幻觉
sub | s19 | AUTO | AUTO | なんかリンネはさ | 说起来莉奈娅啊…… | median | AUTO | 本行 gap=1.1s，拒绝追并
sub | s20 | AUTO | AUTO | 一回ストーリーかなんかで使った時 | 之前在剧情里用过她的时候…… | high | AUTO |
sub | s21 | AUTO | AUTO | 全然使い方…あ、お試しかなんかかな? | 完全不知道用法……啊，是试用吗？ | median | AUTO |
sub | s22 | AUTO | AUTO | あれだ、イベントのお試しで使った時全然使い方わかってなかったんだけど | 在活动试用里完全不知道怎么玩 | high | AUTO |
sub | s23 | AUTO | AUTO | あの連打してポイポポポイって食べさせるやつ | 就是那个连打着喂食的动作 | high | AUTO |
sub | s24 | AUTO | AUTO | マジで可愛いよな、あれ元素爆発含めて | 真的好可爱啊，连元素爆发也算上 | high | AUTO |
sub | s25 | AUTO | AUTO | 最高モーションの一つと言ってもいいぐらい好き | 这简直是我最喜欢的顶级动作之一 | high | AUTO |
discard | s26 | 孤立填充词（无残值、不衔接），丢弃
sub | s27 | AUTO | AUTO | よし | 好 | high | AUTO | 独立语气词，完整保留
sub | s28 | AUTO | AUTO | そうしてるのはみんなと同じじゃないといじめられるから | 之所以那样，是因为不和大家保持一致就会被欺负 | high | AUTO | 上游单行长句原样输出，单源不受20字门槛
sub | s29 | AUTO | AUTO | ごめんごめん、怖がらないで | 抱歉抱歉，别害怕 | high | AUTO |
sub | s30 | AUTO | AUTO | 座っておしゃべりしようよ、ねっ | 坐下来聊聊天吧，呐 | high | AUTO |
sub | s31 | AUTO | AUTO | スネージナヤには自分の子供と人の赤ん坊を | 在至冬国……把自己的孩子与人类婴儿 | high | AUTO | 与32合并约7.9秒超7秒硬上限，拒绝合并
sub | s32 | AUTO | AUTO | こっそりすり替えるフェイがいるって言われてるの | 据说有会偷偷调包的谶鸟 | median | AUTO |
sub | s33 | AUTO | AUTO | フェイの子は、見た目は人間にそっくりだけど、 | 谶鸟的孩子虽然外表和人类一模一样 | high | AUTO | 与34合并约30字且可在「けど」处自然分行，拒绝越过软门槛
sub | s34 | AUTO | AUTO | すぐに自分が人じゃないって気づく。 | 但很快就会意识到自己并非人类 | high | AUTO |
sub | s35 | AUTO | AUTO | でも、生きるためには正体を隠さないといけない。 | 但是，为了活下去必须隐藏真身 | high | AUTO |
sub | s36 | AUTO | AUTO | おい、見てみろよ!変な髪! | 喂，快看！怪头发！ | high | AUTO |
sub | s37 | AUTO | AUTO | うわぁ、ほんとだ、翼みたい! | 哇，真的像翅膀一样！ | high | AUTO |
sub | s38 | AUTO | AUTO | どれさんなの? | 那是哪位？ | low | AUTO | ASR可疑，建议人工核对
sub | s39 | AUTO | AUTO | うっ…ち、違うよ… | 呜……不、不是的…… | high | AUTO |
sub | s40 | AUTO | AUTO | ママが新しい髪型に… | 妈妈给我弄了新发型…… | median | AUTO |
sub | s41 | AUTO | AUTO | でも赤いの排斥でもないんだよな | 也不是在恶意排斥吧 | median | AUTO |
sub | s42 | AUTO | AUTO | してくれただけ | 只是帮我弄的而已 | median | AUTO | 与40被41隔开，源不连续不得合并
sub | s43 | AUTO | AUTO | うん | 嗯 | high | AUTO |
</output>

<notes variant=nosingles>
对照要点：
1. 输入 43 条 → 终稿直接输出 translated；局部序号 {ref:s18}、{ref:s26} 以 `discard` 显式丢弃，其余每个序号都被覆盖，合并行用逗号连接。任务窗口必须按窗口自己的条数覆盖。
2. gap 永远指向下一句：如局部序号 {ref:s19} 判断后句时引用「本行 gap={calc:gap:s19}s」；判断前句时引用「前一行 gap=Xs」。所有 gap 均取末源到下一源。
3. 局部序号 {ref:s18} 的复读串为疑似幻觉→以 discard 显式丢弃。{ref:s10}、{ref:s11} 的填充词无独立残值→合并成一句；{ref:s26}「んー」孤立无残值→丢弃；{ref:s27}「よし」有独立语用功能→完整保留。
4. 局部序号 {ref:s19}、{ref:s24} 之后不因「语义相关」跨 {calc:gap:s19}s/{calc:gap:s24}s gap 合并。
5. 新上游已把多数同句碎片预并成单行（{ref:s03}、{ref:s24}、{ref:s28}、{ref:s35}、{ref:s36}）：单源长句原样输出，{thr:hard_chars} 字/{thr:hard_seconds} 秒硬门槛只约束**合并行**；合并最多两个连续源；三源仅限 filler 三明治（见下方专例）。
6. 主例 {ref:s08}、{ref:s09} 展示 {calc:cc:s08} 字<{thr:hard_chars} 硬门槛；下方字数否决例展示硬门槛否决；主例 {ref:s31}、{ref:s32} 展示 {thr:absolute_seconds} 秒绝对门槛否决，{ref:s33}、{ref:s34} 展示可自然分行的句子拒绝越过硬门槛，{ref:s40}–{ref:s42} 展示说话轮替交错时源不连续禁止合并。
7. `<preceding_context>` 中的 -1、0 为只读前文：未出现在终稿中，也未跨边界合并。
8. conf 只用 high/median/low；`<void>` 仅用于 translated。
</notes>

<notes variant=basic>
对照要点：
1. singles 与 translated 一一对应，局部序号逐条覆盖；本档不做判断型合并，唯一的多源行是词中切断的接回。
2. 局部序号 {ref:s18} 的复读串、{ref:s26}「んー」为疑似幻觉/无残值填充：singles 照常输出并在 note 标注，translated 以 `discard` 丢弃。
3. 单源长句（{ref:s01}、{ref:s03}、{ref:s28}、{ref:s35}）原样输出，不因 {thr:hard_chars} 字/{thr:hard_seconds} 秒门槛丢弃或强拆——门槛只约束合并行。
4. `<preceding_context>` 中的 -1、0 为只读前文：未出现在 singles 或终稿中，也未跨边界合并。
5. conf 只用 high/median/low；void 只在 translated；终稿可润色 singles，但不改变纠错事实。
</notes>
