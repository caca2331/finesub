# session_replay

冻结既有 run 的上游注入、重打某个 harness 会话的 prompt 迭代工具
（correction R2 优先；复用已渲染的 search+extract 正文，不重新调搜索代理）。
从仓库根目录运行：

```powershell
python -m tools.session_replay correction --dry-run --label dry1
python -m tools.session_replay --list-sessions
```

默认会调用生成 API（`--dry-run` 只重建 prompt）。行为细节见
`docs/session_replay.md`。

## 维护策略

**按需维护，不随主程序自动更新。**

- 不在默认测试套件内；harness 接口、fixture schema 变化时不要求同步本工具。
- 只有用户明确要求做 prompt 迭代或修复本工具时才更新。
- 本目录的测试按需运行：`python -m pytest tools/session_replay -n 0`。
