类 CSV 输入格式：
1. `<asr_result>` 第一行是 header：`$asr_result_header`；header 后每行格式为：`本窗口局部序号|$csv_time_col_name|片段时长|与上一行之间的停顿gap|${turn_cell}文本`；gap 是上一行结束到本行开始的秒数，与上一行重叠时记 0，首行留空。局部序号在每个窗口都从 1 重新开始；header 不是待处理字幕，不计入输入条数。
2. 时间单位是秒，展示到 0.1；$csv_time_note本地会用源序号回填原始高精度时间轴。
3. 每条输入记录占一个物理行。$turn_column_item
