translated 产出纪律（JSONL，与合并策略无关）：
1. 本行 `gap` 只表示本条结束到后一句开始的间隔；与前一句的间隔来自前一行 `gap`。type=reasoning 对象引用 gap 时写清方向和数值。
2. 加权字数写入数值字段 `char_count`；note 不重复字数。
3. 每写完一个对象，核对 start、duration、末源 gap、conf、char_count 都在对应键中；不要依赖字段位置猜含义。
4. 每个非空物理行都是独立 JSON object；局部推理也必须写成 type=reasoning 对象。不得夹入 header、`#` 注释或 Markdown。
