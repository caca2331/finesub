素材：$title

人物表：$cast

参考字幕，格式 `开始–结束|说话人|文字`：

$refs

ASR 字幕（连续），格式 `id|开始–结束|$label_head文字`；▶ 标出的是要定真值的行：

$rows

共 $count 个 ▶ 行。每行按 `id|cls|人|prio|src|asr|note` 作答；`one`、`multi` 必须填 `main` 或 `side`，其余类别的 `prio` 留空；`asr` 除 `halluc` 外填 `ok` 或 `miss`。请依次输出 `<people>`、`<rows>`、`<missing>`、`<tracks>` 四个块。
