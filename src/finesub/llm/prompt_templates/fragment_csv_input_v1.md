类 CSV 输入格式：
1. `<asr_result>` 第一行是 header：`local_id|start|duration|gap|text`；header 后每行格式为：`本窗口局部序号|$csv_time_col_name|片段时长|片段尾部离下一段话的gap|文本`。局部序号在每个窗口都从 1 重新开始；header 不是待处理字幕，不计入输入条数。
2. 时间单位是秒，展示到 0.1；$csv_time_note本地会用源序号回填原始高精度时间轴。
3. 每条输入记录占一个物理行。
