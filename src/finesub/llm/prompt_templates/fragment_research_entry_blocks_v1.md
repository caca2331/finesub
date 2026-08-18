<requested_entries>
每行一个索引中的 key 或别名，按重要性从高到低排列；没有需要的条目时输出空块。单独上限 $max_requested_entries 条
</requested_entries>
<keep_entries>
每行一个 `<preinjected_entries>` 中实际可见的词条 key 或别名；没有需要保留的条目时输出空块。单独上限 $max_keep_entries 条；harness canonicalize、去重后与 requested 共享 $max_total_entries 条总上限，keep 优先于 requested，超出部分会从 requested 尾部开始丢弃
</keep_entries>
