认人与定形状（先做这一步，再按本提示其余的规则$shape_work）：
写 `<translated>` 之前，先把 `<asr_result>` 的行从头到尾过一遍，判断每一行是谁在说、相邻两行之间的每处接缝是不是同一个人接着说、哪些行整行不要。这一步在内部思考里做：$shape_hands_off，结果体现在 `<tracks>` 轨人表、后面每一行的 `spk` 与合并、丢弃、note 上。每一行的 `spk` 和$work_verbs一样是要交付的结果：多人对话里声学层常把几个人并在一条轨上，这样的轨要逐行认是谁，不能一律照抄。
$promo

${tracks_intro}`<asr_result>` 的 `track` 列就是它给这一行挂的轨号，每条轨的画像（占多少语音、$profile_examples）在用户消息的「轨迹画像」里。$tracks_cells
$marks$tracks_errors
$tracks_trust$turn_clue

$who$voice_clause

$spk_answers

每处接缝判成 `join`、`may_join` 或其余（不并），整行不要的行判成 `drop`（这几个词只是你判断时的叫法，不写进输出）。

`join`：这一行与上一行是同一个人接着说、同一句话被切成了碎片，要接成同一条字幕。`<asr_result>` 的第一行不往前 join：它前面是 `<preceding_context>` 的行，不能并。
$join_criteria

`may_join`：这一行是同一个人接着上一行往下说，只是上一行已经是完整的一句，或并起来会比短字幕长；并不并到下面「短片段合并策略」第 3 条再定。
$may_criteria

其余的接缝（换了人、判不出是同一个人、转折另起、换了话题、只是相关）不并。

`drop`：这一行整行不进字幕，在 `<translated>` 里写 `discard|<源序号>`。认人这一步只判下面${single_drop_kinds}（一行里无意义的复读这类幻觉，纠错时按「字幕节奏与排版要求」第 4 条处理）：
$drop_criteria$drop_listen
${shape_drop_tail}凭空出现的套话只在上面${hallu_item_ref}的三点同时满足时丢弃：别处说的「高度疑似幻觉」不放宽这一条。
夹在同一个人一句话中间的单独垫词行不 drop：它前后是同一句的两段，连同它三行并成一条（filler 三明治）。合并与 discard 都只认连续的源：合并行的 position 只能是相邻的源，不能跳过中间的行把前后两行并起来，也不能 discard 中间的行再把两边并成一条。
