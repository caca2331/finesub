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

## 钉本机 agent 目标（2026-08-25）

`--model` 除了 FREE Gemini 端点，也接受 catalog 里的 **local-agent fact id**，例如：

```powershell
python -m tools.session_replay correction --model local-agy-gemini-3_7-flash -n 5 --max-attempts 8
```

加这条是因为免费档会成小时地整体不可用（2026-08-25 实测 3.7-flash 与 3.6-flash 连续 503
`UNAVAILABLE`），而跑不动的对照等于没有对照；订阅额度这条路不受它影响。两处细节：

- 端点的 `backend` 必须钉成 `local_agent`，否则调用会走 REST 传输，然后因为「本机 agent 本来
  就没有 API key」而失败；
- 会话档位强制 `api`（逐调用新会话）。attempt 之间必须互相独立，否则「N 次里成功几次」这个
  计数就不成立了。

## 金标准与重切窗口的对齐（2026-08-25）

`benchmark.py` 先按 `source_count` + 指纹要求金标准与 fixture **精确匹配**；不匹配且金标准带
`sources` 时，改走 `alignment.py` 按**时间**对齐——金标准里的一条边界是音频上的一个时刻，新切分
若仍在那儿断开，判断就转到此刻相邻的那两行上（容差 0.15 秒，一一配对）。

**对齐只答「窗口被重切了」这一种失配**（`BenchmarkWindowMismatchError`），而且要先确认真的重切
过：指纹连 ASR 文本一起哈希，解码变了、边界一条没动也会失配，那种情况对齐是空重映射、只会把
「金标准已过期」这个唯一信号删掉，所以 `same_cut()` 判定切分未变时直接拒绝。金标准自身坏了
（默认值写错、`must_*`/`may_*` 不互斥）是 `BenchmarkMalformedError`，任何时候都不走对齐。

**新切分多出来的边界归「未审」**，两个方向都不计分。金标准的 `must_not_merge` 默认是靠审完自己
窗口里每一条边界挣来的，而多出来的那些正是把短语切碎的细分——照默认罚下去等于惩罚正确行为。
实现上未审边界/源行被塞进中性类（`may_merge` / `may_drop`），所以打分代码不必认识第三类。

**重切把「该丢」和「该留」并进同一新行时，那行同样归未审**：继承 `must_drop` 会让打分奖励删掉
金标准要求保留的内容。行的归属要求实质重叠（0.05 秒，或短行的一半），相邻两行漂移一两毫秒不算。

报告首行打覆盖率，分**两个数**（例：`223/302 boundaries and 298/303 sources are covered by a
verdict, of which 180 and 260 can change the score`）：**covered** 是金标准够得着的行/边界，
**discriminating** 是分数真能动的那些——`may_merge` / `may_drop` 在打分器里被从两侧同时减掉，
所以拿着宽松裁决的行怎么做都不扣分，把它算进「已评分」等于把下界读松了。混合来源的行数单列，
它们是被重切**制造**出来的宽松裁决，和人手写的那种值得分开看。丢失的四类判断也单列。
**对齐后的分数是下界**：软/硬长度附加费只对并了非中性边界的行触发，完全由未审边界拼出的
超长行不会被罚。

`--json` 的顶层是 `{"alignment": …|null, "scores": [...]}`——`alignment` 为 `null` 表示金标准精确
匹配，否则它带着上面那份覆盖率。自动化比对拿不到首行 prose，正是最容易把下界读成定论的那个读者。
