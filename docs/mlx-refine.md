# MLX refine 后端

Apple Silicon 的实验性本地 Whisper 后端使用 `mlx-whisper==0.4.3`、`mlx==0.32.2`。
安装仅走源码安装流程；Windows 托管运行时和 checkpoint 兼容键保持上游定义。

解码时保存选中 token 的概率、alignment heads cross-attention、终止状态和真实帧边界，
交给共享 WT refine 逻辑产生词、confidence 与 alignment_events。
trace 与分段对账失败时，仅该窗口改用 teacher-force 对齐，并记录次数与原因。
未闭合尾段必须保留并整窗前进，避免在相同音频上重复解码。

当前移植版本仅声明已实现的解码能力；未知选项在入口拒绝。
CPU 运行使用 patched CTranslate2；缺少其 trace 扩展必须报错。
运行版本从已安装包读取，模型 revision 和 trace 来源只进入 metadata。

开发对照可用 `FINESUB_ASR_BACKEND=auto|fw-refine|mlx-refine`；显式 `--device cpu`
或 `--gpu-tier cpu` 优先走 CT2 CPU。该开关供开发者验收使用，用户入口仍是既有 `--asr`。
默认模型固定为 `mlx-community/whisper-large-v3-turbo` 的
`a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` revision。
源码适配对象为 MIT 许可证的 mlx-whisper 0.4.3 `decoding.py`、`timing.py`、
`transcribe.py`；seek loop 的有界替换与内部接口均在启动时核查，不追踪上游 main。

## 时间戳契约

零时长词的起点不得越过所属段边界并造成跨段倒序。
上游 `normalize_segment_timeline` 将越界词约束到所属段内，
同时保留文本、confidence 和 alignment_events，并在 segment_time_repairs 记录修复。
aligned 与 stable 的输出路径均使用该契约，回归测试使用合成坐标验证边界行为。

## 验收状态

入场验收还需同机 patched CT2 CPU 与 MLX 的差异报告，包含至少一段 30 分钟素材、
实际软件版本、词覆盖与边界误差及 refine 回退率。完成这些数据前保持 PR 为待验收状态。
