<requested_entries>
每行一个索引中的 key 或别名，按重要性从高到低排列；这一条回复不请求条目时，输出空块。单独上限 $max_requested_entries 条
</requested_entries>
<keep_entries>
每行一个 $keep_source_block 中实际可见、要留到交付时的词条 key 或别名；没有需要保留的条目时输出空块。单独上限 $max_keep_entries 条；与 requested 合计至多 $max_total_entries 条，超出时先丢 requested 末尾
</keep_entries>
