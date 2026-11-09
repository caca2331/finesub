${style_channel_title}：
1. 本任务的第一条回复只做一件事：按下面「${style_pick_title}」选定一套风格。这一条回复只输出 `<window_notes>` 和 `<style_selection>`，然后结束；harness 回填选中那一套的完整约定后，你在同一对话里按 system 与首条 user 消息的全部要求交付终稿。
2. 本任务没有可请求的外部检索或知识库词条：不要输出 `<search_queries>`、`<requested_entries>` 或 `<keep_entries>`。回填只有一次，之后不能改选。
请求回复的输出块：
<window_notes>
请求前对本窗口的简短要点（800 token 以内）：话题/剧情位置、说话状态、可疑 ASR 点及推断的正确候选。写于回填之前，未经证实的判断标「待定」；没有要点时输出空块。
</window_notes>
