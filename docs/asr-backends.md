# ASR 后端契约

对齐循环（`src/finesub/speech/recognition/transcribe.py`）通过一个很小的面驱动解码：
`model.transcribe_wt(audio, **options)`，以及可选的批解码函数。本文是这个面的契约，也是**新 ASR 后端
（例如在 Apple Silicon 上移植 refine 的 MLX 后端）进入仓库的条件**。实现是
[`asr_backend.py`](../src/finesub/speech/recognition/asr_backend.py)，由
`test/test_asr_backend_contract.py` 钉住。

现在只有一个后端：打过补丁的 CTranslate2（`fw-refine`，`fw_refine_backend.RefinedWhisperModel`）。
云端转写（`--asr`，`speech/joint/`）不走这个面：它不进对齐循环，直接把云端的词或段送进分句 DP，写同样格式的 `-aligned.json`
（[`docs/architecture.md`](architecture.md) 的 `speech/joint/` 一行）。

## 为什么要契约

失败模式是**静默**的：一个后端只用 `options.get` 读它认识的选项、其余一律忽略，那么对齐循环
每加一个解码选项，这个后端都会悄悄走另一条路——而除 CTranslate2 之外的后端跑在 CI 没有的硬件上，
没有任何测试会红。所以契约有两半：

1. **选项只在一处声明**：`DECODE_OPTIONS`。对齐循环只能经 `decode_options(...)` 拼参数，
   `transcribe.py` 里不得内联写选项名（守卫扫描字典字面量与下标赋值）。
2. **后端拒绝未知选项**：收到声明集合之外、或自己能力之外的选项，运行时抛 `TypeError`。

## 选项的两种身份

| 身份 | 含义 | 当前选项 |
| --- | --- | --- |
| 必须支持 | 每个后端都实现 | `language`、`temperature`、`detect_disfluencies`、`collect_refine_signals`、`collect_attention_signals`、`force_teacher_force` |
| 挂能力标记 | 后端可以不支持，不支持时循环走写明的降级路径、不发这个选项 | `beam_size`、`best_of`（能力 `beam`：不支持时低覆盖救援跳过 beam 那一级，直接剥离，计数 `beam_rescue_skipped`） |

没有「声明但拒绝」这一态：循环每次都发的选项被拒，等于那个后端每次都报错。所以贪心解码的写法是**不发**
beam 选项，而不是 `beam_size=None`（测试钉着：普通解码与重试发出的键，无能力的后端必须全部接受）。

另有一项不对应选项的能力 `batch_decode`：后端是否提供批量首解码（`--asr-decode-batch` 大于 1 时的预取层）。
没有它，循环逐窗解码，批设置不起作用。

## 一个后端要做什么

- 类上声明 `capabilities`（只能用 `asr_backend.CAPABILITIES` 里的名字）与
  `accepted_decode_options = asr_backend.accepted_for(capabilities)`。
- 在 `transcribe_wt`（与批解码函数，如果有）入口调用 `asr_backend.reject_unknown(...)`。
- 在 `asr_backend.BACKENDS` 里登记 `module:Class`。`test_every_registered_backend_holds_the_contract`
  逐个检查：少接受一个必须的选项、或没有能力却接受挂标记的选项，都会红。
- **不可用时在加载模型前报错**，不退回别的实现：CTranslate2 后端对未打补丁的 wheel 就是这样
  （缺 refine 轨迹或语气词权重支持时，构造模型即抛错）。退回一个没有 refine 信号的实现，会跑完一整趟、
  交出质量不同的字幕，而没有人被告知。

## 新后端的入场条件

以 MLX 为例（讨论见 PR #15），一个移植同一套算法的新后端要带着这些进来：

1. **接在上面的契约上**，过 `test_asr_backend_contract.py`。
2. **自身的已知问题查清修好**：例如按窗口退回 teacher-force 的比例与原因（MLX 在 PR #15 里的 one-pass
   命中率只有 50% / 80%）、检测不到语言时 `language` 不能写成字符串 `"None"`。
3. **一份同机差异刻画**：新后端对同一台机器上的 CTranslate2 CPU 路径，用
   [社区验收套件](../tools/hw_acceptance/README.md) 跑，**至少含一段 ≥30 分钟的素材**（连锁变化出在长
   group 上），带版本；越过复核门槛的项附三问复核结论。完整的三段素材记录可以之后交，用于升支持等级。
4. **不改 checkpoint 的兼容键**（修订号不同不是形状变化，只进来源记录）、**不动 Windows 的锁**（锁内容一变，
   所有 Windows 安装升级时都会重建运行时）、不带会话记录式的文档。

移植后端按 `README_DEV.md` 开发原则的 **A 类（执行形态）** 处理：合入即默认开、标「实验」。它自己的回退
路径不是归约顺序带来的，所以验收另报 refine 回退率；复核判出系统性差异时，结论是**修移植的实现**，
不是改按 B 类——修好之前要不要临时关（让 `auto` 退回 CTranslate2 那条路），由维护者按严重程度定。
