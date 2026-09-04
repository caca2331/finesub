# `tools/bench/` —— 性能与探测器的测量工具

**状态：活跃。** `docs/bench-baselines.md` 里几乎每一个数字都是这里的某个脚本跑出来的，
那份文档里 20+ 处引用本目录。

⚠ **动性能之前先读 [`docs/bench-discipline.md`](../../docs/bench-discipline.md)**——
一个数字要满足什么条件才算数。本目录的 `discipline.py` 就是那六条的代码强制项，
绕开它测出来的数不进基准文档。

## 三层

| 层 | 文件 | 说明 |
| --- | --- | --- |
| **纪律** | `discipline.py`、`gpu.py` | `repeat`/`paired_ratio`/`scale_check`/`refuse_empty`/`RunConfig` 六条强制项；`gpu.py` 的 `idle_baseline()` 记每次测量的空闲读数（本机不是干净测量环境） |
| **探针** `probe_*.py` | 24 个 | 一个探针回答一个问题，跑完把数写进 `docs/bench-baselines.md` 的对应节。命名即问题：`probe_wddm`（时钟状态）、`probe_ct2_cold`（冷启动 JIT）、`probe_asr_decode_batch`（组批）、`probe_referee_accel`（裁判编译解码）… |
| **打分/导出** `score_*.py` `run_*.py` `export_*.py` `make_*.py` | 8 个 | 跑完之后把产物折成可判读的表；`*_corpus.jsonl` / `p9_titles.json` 是它们的语料清单 |

## 怎么用

```powershell
# 探针一律 python -m，因为它们 import 生产代码
python -m tools.bench.probe_wddm --chunks 120 --gap-ms 40
python -m tools.bench.probe_asr_decode_batch --help
```

跑之前确认素材在位：语料清单里的路径指向 `data/` 与 `assets/`，这两个目录**不入库**
（见 [`docs/data-index.md`](../../docs/data-index.md)）。

## ⚠ 这里的测试默认不跑

`pyproject.toml` 的 `testpaths` 只含 `test/`，**不含 `tools/`**。
所以本目录的 5 个 `test_*.py`（`test_discipline` / `test_sbd_split` / `test_bilingual` /
`test_detector_answer_rate` / `test_ghost_threshold`）**在根套件里永远不会执行**：

```powershell
python -m pytest tools/bench -q      # 要跑就显式指定路径
```

这不是疏忽而是取舍——它们要么慢、要么依赖不入库的素材。但改了 `discipline.py`
或任何 `probe_*` 的判据，请顺手跑一次上面那条；`test/test_import_boundaries.py`
会扫 `tools/` 的导入方向，但不会替你验这些契约。

## 相关

- [`docs/bench-discipline.md`](../../docs/bench-discipline.md)——测量纪律（先读这份）
- [`docs/bench-baselines.md`](../../docs/bench-baselines.md)——量到了什么（按节号引用）
- [`docs/data-index.md`](../../docs/data-index.md)——素材从哪来
- [`../README.md`](../README.md)——`tools/` 全目录索引
