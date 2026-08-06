# WT refine port oracle

本目录是完整 WT refine 移植的开发期 oracle，不属于生产入口，也不随其他改动顺手维护。

当前只冻结 `whisper-timestamped==1.15.9` 使用的 DTW step pattern。生成的固定路径进入
CTranslate2 C++ 单测，C++ 测试运行时不依赖 Python、SciPy 或 `dtw-python`。

```powershell
python -m tools.wt_refine_port.oracle
python -m pytest tools/wt_refine_port -n 0
```

`teacher_force_probe.py` 会先用 patched CT2 做 FW greedy decode 与 WT-mode alignment，释放
模型后再用 OpenAI Whisper 对同一 tokens、同一扩展窗口做 reference alignment。默认只跑一个
segment，产物写入显式指定的 `out/` 路径：

```powershell
python -m tools.wt_refine_port.teacher_force_probe `
  --audio assets/hello.flac `
  --fw-model C:/Users/Carl/Documents/Carl/models/faster-whisper-large-v3-turbo `
  --language en --max-segments 1 `
  --ct2-python C:/Users/Carl/Documents/Carl/projects/CTranslate2/python/build/wt-refine-runtime-wide `
  --ct2-bin C:/Users/Carl/Documents/Carl/projects/CTranslate2/install-cu-wide/bin `
  --cuda-bin "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin" `
  --output out/wt-refine/hello.json
```

`full_window_probe.py` 则保留 FW 原始起止 timestamp tokens，在同一个 30s encoder window 上依次
对齐多个正常 segment；缺少边界 timestamp 的 segment 会明确列入 `rejected`，留给下一阶段的
repair state machine，而不会静默拿后处理后的 segment.start/end 代替。

`state_machine.py` 将 WT 1.15.9 的 decoder-limit 判断、连续 timestamp flush、末 token/end
timestamp 修复、attention row 选择和 confidence 分片冻结为无 Torch/CT2 依赖的纯策略。后续
CT2 generator 只负责导出这些策略需要的事件和张量，避免在 C++ 内重写一套容易漂移的 policy。

`one_pass_probe.py` 使用 patched CT2 的 greedy trace：一次 decode 返回 6 个 timing heads 的
逐步 cross-attention（含 terminal EOT query）、selected-token logprobs 和末两步完整 logits，再按
decoded timestamp spans 切片，直接与同一 encoder output 的 teacher-force `align` 比较。
