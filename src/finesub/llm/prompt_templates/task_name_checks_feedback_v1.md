核实记录（知识库中文定名核查的输入）：
1. 本次任务开启了知识库写入。$position_sentence额外输出一个 `<task_update_feedback>...</task_update_feedback>` 块。
2. 你的记录不会直接写入知识库：任务结束后由单独的核查调用从这些记录出发、自己读页面核实后再落库。你的职责是把搜到的写法、页面地址和原文留住。
$feedback_schema
没有一条可记时输出 `{"name_checks":[]}`，不要编造。
