${style_pick_title}：
1. 下面是本次可选的${style_candidates}。它们只是待选资料，不是本任务要遵守的约定；$after_pick_clause
2. 依据是每套的「适用范围」与用户本次的要求$extra_style_clause：对得上素材（主播、题材、形式）就选它，只选一套；不要按你觉得哪套文风更好来选。$no_match_clause
3. 这一步必须做：$output_clause输出有且仅有一个 `<style_selection>...</style_selection>` 块，内容是一个 JSON 对象：{"selected": "候选名", "reason": "一句话理由"}。缺这个块、或 `selected` 没按上面第 2 条填，这条回复就不合格。
$read_clause候选（每行一套：名字｜简介｜作者｜适用范围）：
$catalog
