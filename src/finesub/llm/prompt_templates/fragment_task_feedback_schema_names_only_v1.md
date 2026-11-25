`<task_update_feedback>` 内容必须是一个紧凑 JSON 对象（不要 Markdown 代码块），schema 如下：
{"name_checks":[{"source":"知识库那一行第一列的原文","kb_zh":"那一行的定名","found":"搜到的中文写法，没搜到写空字符串","url":"写着它的页面地址","quote":"那条搜索结果里写着这个写法的原文","query":"你搜的是什么"}]}
字段规则：
1. $name_checks_rule
