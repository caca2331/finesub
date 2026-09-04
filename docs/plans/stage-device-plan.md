# 逐阶段设备解析 · 拆分 `entry` 与 CPU（设计稿）

> **状态：台账（已全部实施）。** 五步都在代码里。「现状」一节记的是动手前的代码，其中
> §2.6 已被实施时的自查更正（见 §9.2）；§3 的规则与 §4 的顺序各被复审推翻过一次，改动
> 台账在 §9。**读这份文档是为取舍依据；现行行为以代码与 `gpu-profiles.md` 为准。**
>
> ⚠ 动这一族代码时**两个方向都要跑**（§6.2）：`pytest -q` 与
> `CUDA_VISIBLE_DEVICES=-1 pytest -q`。只跑前者会漏掉 CI 会红的那些。
>
> 送审目的：请审 ①目标语义对不对；②五条约束有没有漏；③五步的顺序与风险定价。
>
> **v5（2026-09-02）**：**全部五步已落地**。4b 与第 3 步原判「本机验不了」是**错的**——
> 抬高 torch 的 arch 下界就能在进程内劈开两个后端（§6.1），所以它们有真验收面。
> 顺带用 `CUDA_VISIBLE_DEVICES` 跑无卡回归，查出第 2 步引入的一个真 bug（§6.2）。
>
> **v4**：第 1、2、4a 步落地；第 3 步经查是空操作，推迟到 4b 之后。
>
> **v3（第二轮复审后）**：4a 从「阶段入口建模型」改成「建模型 + 一次热身
> encode」——v2 说「真实探测与照常建是同一件事」是错的，见 §3。
>
> **v2（第一轮复审后）**：顺序改成 1 → 3 → 4 → 2 并把第 2 步拆成两半；裁判那步补上
> 「用户要的 CPU」与「机器只能 CPU」的区分；§3 第二行规则的**判据**被推翻并重写。
> 改动理由逐条记在 §9。

## 1. 起因

档位（`--gpu-tier`）今天既表达「这张卡有多大」，也隐含「这台机器用不用 GPU」——无 CUDA 时
`detect_gpu_tier` 直接返回 `DEFAULT_GPU_TIER = "entry"`
（[`resources.py:108,308`](../../src/finesub/speech/runtime/resources.py)）。于是一次纯 CPU 运行
的产物上写着「基础 GPU 档、显存上限 3GB」。

owner 要的目标语义（2026-09-02，原话）：

> 有些步骤可能需要特定 N 卡如 patched ct2，有些则是更通用的如 torch。最好的语义是，
> **非 cpu 的 profile，能用硬件加速的步骤尽量用 GPU，否则这个步骤降级到 cpu。**

即：**档位是策略（准不准用 GPU），后端能力是能力（这一步用不用得上）**，两者今天混在一起。

## 2. 现状（已核）

### 2.1 逐阶段降级的骨架已经有了

`resolve_device(requested, *, context)` 就是**逐阶段**调的，每次各自打一条 `cpu-fallback`
警告：[`silero_ghost.py:84`](../../src/finesub/speech/preprocessing/silero_ghost.py)、
[`transcribe.py:2940`](../../src/finesub/speech/recognition/transcribe.py)、
[`vad_asr_stage.py:940`](../../src/finesub/speech/recognition/vad_asr_stage.py)。

分离器不走它，有自己的 seam（直接 `cuda_usable()`，
[`separation.py`](../../src/finesub/speech/preprocessing/separator/separation.py) 多处）。
**分离器其实已经是目标语义的参考实现**：自己判、自己降级、还自己调整并行度。

### 2.2 但它只会问一个后端，而 ASR 问错了

`cuda_usable()` 读的是 `torch.cuda.get_arch_list()`
（[`device.py:99`](../../src/finesub/speech/runtime/device.py)）——纯 torch 的答案。各阶段真正
依赖的东西不同：

| 阶段 | 后端 | 真实要求 | 今天问谁 |
| --- | --- | --- | --- |
| 人声分离 BS-RoFormer | torch | torch CUDA kernel | `cuda_usable()` ✓ |
| silero assist | torch | torch CUDA | `resolve_device` ✓ |
| Qwen 裁判 | torch / transformers | torch CUDA | ✓（`qwen_referee` 里那两处裸 `is_available()` 是编译缓存目录名与 `empty_cache()`，**不是**放置判断） |
| **Whisper ASR** | **patched CT2** | **CT2 的 CUDA 构建** | ✗ **问的是 torch** |

全仓**没有任何 CT2 能力探针**。于是两个方向都错：

- **torch 能用、CT2 不能** → 不降级，**直接报错**：`resolve_device` 返回 `"cuda"`，字符串交给
  CTranslate2，失败在第一次 encode 冒出来，由
  [`fw_refine_backend.py:41`](../../src/finesub/speech/recognition/fw_refine_backend.py) 的
  `_missing_gemm_backend` 翻译成人话。
- **CT2 能用、torch 没有这张卡的 kernel** → **静默把 ASR 跑在 CPU 上**。这正是 `device.py`
  文档里说它存在就是要消灭的「静默 10× 变慢」，只是用错了库。

### 2.3 CT2 有探针，且很便宜——但 import 很贵

本机实测（patched `ctranslate2 4.8.1+finesub0.4.0.cu128`）：

```
get_cuda_device_count()                 -> 1          （25 ms）
get_supported_compute_types('cuda')     -> {float16, bfloat16, int8, ...}
get_supported_compute_types('cpu')      -> {float32, int8, int8_float32}
import ctranslate2                      -> 9.0 s      （冷启）
```

**探针本身 25 ms，但 import 要 9 秒。** 所以「启动时探一次、存进 `ResourceProfile`」这个方案
是错的：纯 CPU 运行、或只跑 LLM 阶段的运行会白付 9 秒。探针只能在 ASR 阶段入口懒调。

### 2.4 探针必要不充分

patched CT2 **按名字**在运行时加载 cuBLAS，靠
[`cuda_libs.py`](../../src/finesub/speech/runtime/cuda_libs.py) 在建模型前把目录加进搜索路径。
实测：在**没有** `cuda_libs` 设置的裸进程里 `get_cuda_device_count()` 照样返回 1。

**所以探针通过了，真正建模型时仍可能在 cuBLAS 的 `LoadLibrary` 上炸。** 探针覆盖 CUDA 设备
与 compute type，覆盖不了 cuBLAS。那条错误应当保持响亮。

> **2026-09-03 补：探针现在问两句，不是一句。** `get_cuda_device_count()` 是
> `cudaGetDeviceCount`——它数驱动枚举出的设备，不知道这个 wheel 编了哪些 cubin，所以一张
> 算力低于 `CUDA_ARCH_LIST` 下界的卡（7.0 以下，即 GTX 10 系及更早）它照样答 1，而解码会死
> 在第一次 encode 的 `no kernel image`。`ct2_cuda_unusable_reason` 因此在设备数之后再比一次
> 算力（`device.py` 的 `CT2_MIN_COMPUTE_CAPABILITY`），低于下界回退 CPU。⚠ 这**不改变**本节
> 的结论：加的是一条必要条件，探针仍然不充分——cuBLAS 那一条只有建模型时才知道。

约束：**探针要在 `cuda_libs` 加完搜索路径之后跑**，这样它测的是建模型时的进程状态而不是
另一套。⚠ 但别指望这条能把探针变成充分条件——上面那次实测正说明 `get_cuda_device_count()`
在没有 cuBLAS 的进程里照样返回 1，也就是说**换个顺序它的答案不变**。这条约束买的是「不必
再怀疑顺序」，不是「探针从此可信」。

### 2.5 降级会移动产物边界

分离器在 CPU 上不只是换设备，`separator_instances` 被强制为 1、且不共享模型
（`separation.py`）。而 worker 数决定分块计划（块数 = 轮数 × worker 数）→ **产物边界移动**。

于是「torch 不能用但 CT2 能用」的机器产出的 `-vocal.ogg` 与两者都能用的机器**不同**。
今天就成立；拆开之后它会变成常态，必须写进用户向文档。

### 2.6 产物里的设备记录只缺一处（v3 更正）

> ⚠ **v1/v2 在这里写的是「没有任何字段记解析出来的设备」，那是错的。** 实际查下来五个
> 阶段里四个都记了，第 1 步因此比原先说的小得多。更正记在 §9.2。

逐阶段现状：

| 阶段 | 记在哪 | 记的是什么 |
| --- | --- | --- |
| 人声分离 | `separation.py` 的 metadata sink | `"device": "cuda"/"cpu"` ✓ |
| energy VAD | `energy.py` | 恒 `"cpu"`（本来就只有 CPU 实现）✓ |
| silero assist | `silero_ghost.py` | `"device": device` ✓ |
| **Whisper ASR** | `align_meta["device"]` | **解析后**的设备 ✓——`asr_align_metadata` 在 `resolve_device` **之后**构造（`vad_asr_stage.py:940` → `948`），实测产物里是 `"cuda"` |
| inline 语言重解裁判 | `align_meta["lang_redecode"]["device"]` | ✓ |
| **尾部 `--qwen-verify` 裁判** | `align_meta["qwen_verify"]` | ✗ **只有 `model` / `suspects` / `gaps_probed`，没有设备** |

所以缺口只有最后一行。它恰好是最值得记的一处——裁判的放置是**算出来**的
（`referee_device` 拿档位减常驻），不是用户给的，所以事后完全看不出它落在哪。

### 2.7 顺带确认：两处不用动

- `checkpoint.build_key`（[`checkpoint.py:33`](../../src/finesub/speech/recognition/checkpoint.py)）
  **不含设备** → ASR partial 跨设备 resume。符合仓库「provenance 不是兼容键」的规矩
  （设备差异是数值差异不是形状差异）。**不改**，但要写明是有意的。
- `warn_if_vram_is_short`（[`resources.py:386`](../../src/finesub/speech/runtime/resources.py)）
  里「没有 CUDA 就不告警」是一条硬编码特判。拆出 `cpu` 档后它应该变成
  「`gpu=False` 的档没有显存要求」的自然结果。

## 3. 一条与 `device.py` 哲学冲突的诱惑，先否掉

`device.py` 的原则写着：

> A wrong fallback costs a silent 10x slowdown; a wrong attempt costs a loud error,
> **which is the better way to be wrong**.

所以「CT2 不能用 CUDA → 回退 CPU」**不能无条件做**：

- 装了 CPU-only CT2 的用户今天拿到的是一条指向 `ct2-patches/README` 的明确错误，而
  [`manual/ct2-wheel.md`](../manual/ct2-wheel.md) 整篇存在的理由就是「patched wheel 是必需的」。
  把装错 wheel 变成「安静地慢十倍但跑完」，正是那份文档要防的事。
- 而且 `_missing_gemm_backend` 自己的注释说：**CUDA-only 的构建对 CPU 也报 float32，但它没有
  CPU SGEMM**。所以那个「回退」会再炸一次，更晚、更难懂。

**定案规则（有意与 torch 不对称）**：

| 情况 | 动作 |
| --- | --- |
| CT2 有 CUDA 设备（无论 torch 行不行） | **用 GPU** |
| CT2 无 CUDA 设备 | 回退 CPU，打 `cpu-fallback`；**若 CPU 侧随后没有 GEMM，让那条错在阶段入口抛出**，不要等到第一次 encode |

⚠ **v1 曾把「`get_supported_compute_types('cpu')` 含所需类型」当成回退的门，那是错的**，
而且错得就在同一节：`_missing_gemm_backend` 的注释说 **CUDA-only 的构建对 CPU 也报
float32、却没有 CPU SGEMM**，所以那个查询根本不构成证据。可达路径也存在——CUDA-only 构建
装在一台没有显卡的机器上，`get_cuda_device_count()` 返回 0，落进第二行，查询说 float32
可用，回退过去照样炸。

**为什么仍然回退，而不是像复审建议的那样直接拒绝**：今天 CPU-only 的用户是**能跑的**
（`resolve_device` 给 `"cpu"`，CT2 拿到 CPU 就工作），拒绝回退会拿走这批本来正常的运行。
真正缺 CPU GEMM 的只有**把 patch 用错编译选项构出来的 CUDA-only wheel**——那是开发者的
构建事故，不是用户的安装形态。所以正确的形状是「照常回退，但把那条本来就存在的错误**提前**
到阶段入口」：能跑的照跑，构建错的立刻拿到指向 `ct2-patches/README` 的明确报错，而不是
分离跑完之后才炸。

**要把错误真的提前，必须做一次热身 encode——v2 在这里错过一次。** v2 写的是「阶段入口
本来就要建模型，所以真实探测和照常建是同一件事」。代码不是这样：

```python
# fw_refine_backend.py:181-183
        try:
            with phase_timing.phase("asr.encode"):
                output = super().encode(features)
        except RuntimeError as exc:
            raise _missing_gemm_backend(exc, self.model.device) from exc
```

翻译住在 `encode()` 里，**建模型不经过它**（CUDA-only 构建在 CPU 上建模型是能成功的），
`_missing_gemm_backend` 自己的 docstring 也写着 *"the first encode is where it surfaces"*。
所以「只在阶段入口建模型」一步都不会前移，那会把 4a 做成空操作。

正确形状：**建完模型后主动跑一次热身 `encode`**（一段 1 秒静音即可）。这就是第一轮复审
说的「小 GEMM 探测」，只是**载体是模型本身**而不是另一个入口——那一条意见比 v2 的反驳更对。

⚠ **它买到的不是「更早」，而是「有」（v5 复查更正）。** 早先这里写「让错误在阶段入口发生，
省下分离与 VAD」，两半都不准：

- **省不下分离与 VAD。** `stages.py` 先跑分离才进识别，而识别阶段内 `run_vad_prefix`
  又在建池之前。热身 encode 只可能早于**解码循环**，那点提前量不值一句宣传。
- **但它把「没有错误」变成了「有错误」**，这才是重点。`encode` 的失败发生在
  `transcribe_wt` 之下，而 `transcribe.py` 的分组循环用 `except Exception` 包着它：先
  重试 teacher-force，再**丢掉这一组继续**。于是每组都以同样方式失败、各记一条
  `asr-group-dropped`，任务**跑完**并交出一个空字幕文件——`_missing_gemm_backend` 那段
  精心写的提示只出现在逐组警告里。热身 encode 把失败挪到那个循环**外面**。

落点是现成的：`FwRefineModelPool.warm()`（`fw_refine_backend.py:829`）今天**只建实例、
从不 encode**，热身 encode 正该加在那里。顺带它在 CUDA 路上本来也是真热身（CUDA 上下文
与 kernel），不是纯开销。

## 4. 方案（五步，v2 顺序）

| 步 | 做什么 | 状态 |
| --- | --- | --- |
| **1** | 补上尾部校验裁判的设备记录（§2.6——其余四个阶段已经记了） | **已实施** |
| **2** | `TierSpec` 加 `gpu: bool`；新增 `cpu` 档；`detect_gpu_tier` 无 CUDA 时返回 `"cpu"`；删 `warn_if_vram_is_short` 的特判；**让分离器认这条策略**（见下） | **已实施** |
| **4a** | CT2 探针 + **热身 encode**：阶段入口问 CT2 要不要回退，`warm()` 跑一次 1 秒静音的 `encode`，让 `_missing_gemm_backend` 在这里翻译。**只建模型不 encode 等于没做**（§3） | **已实施**；⚠ 2026-09-03 起探针**多问一句算力下界**（§2.4 的补注）——设备数答得出不代表这个 wheel 有这张卡的 cubin |
| **4b** | 另一半：ASR 的设备**只**问 CT2，所以 torch 认不出的卡也能用 | **已实施**——并且**验得了**，见 §6.1 |
| **3** | 裁判分四问：意图 / 策略 / **它自己后端的**能力 / 有没有 pool 要挤 | **已实施**（与 4b 同一次，见下）。⚠ 「意图」读的是**请求**，`None`（桌面「自动」）等于默认请求 cuda，**不等于**解析后的 ASR 设备——那个值在 CT2 回退后也是 `cpu`（§9.5） |

#### 第 3 步与 4b 是同一件事的两半

第 3 步单独做曾经是空操作（v4 查出的），理由如下；而 4b 一旦落地，它不只是有了收益，**它
是 4b 的必要补丁**：ASR 的设备改由 CT2 决定之后，「ASR 在 CUDA 上」不再蕴含「torch 能用
这张卡」。裁判是 transformers 模型，跟着 CT2 上卡就会被塞到一张 torch 没有 kernel 的卡上
——比 4b 修的那个问题更糟。所以两者必须同一次改，而且裁判必须问**它自己的**后端。

以下是 v4 当时查出的、第 3 步单独做为什么没有收益：

实施前先查了它的前提，结果**不成立**。裁判的放置读的是 `resolve_device` 之后的 ASR 设备，
而那个值等于 `"cpu"` 只有两种可能：

| ASR 落 CPU 的原因 | 裁判能不能用 GPU |
| --- | --- |
| 用户显式 `--device cpu` / `--gpu-tier cpu` | **不该用**——那是意图 |
| `cuda_unusable_reason()` 非空（torch 用不了这张卡） | **也用不了**——裁判就是 torch/transformers |

两条都不给裁判上 GPU 的机会，所以第 3 步今天**一行收益也没有**。它要能成立，得先有一种
「ASR 在 CPU 而 torch 的 CUDA 是好的」的情形——那正是 **4b**。

所以第 3 步与 4b 是同一件事的两半，顺序改成 4b → 3——**v5 里两者一起做了**。

**当初为什么把 4b 排最后**：以为它的收益无法在本机验收——本机 torch 与 CT2 都能用 CUDA。
**那个判断是错的**，§6.1 给出了在进程内劈开两个后端的办法，分支本身现在有真验收。
排序仍然合理（先做能验收的），但理由从「验不了」降级成「收益依赖一台我们没有的机器
是否真的存在」。

⚠ v4 还写过「今天那个错误是一条正确的响亮报错」——**也是错的**，见 §3 的更正：它根本不
让任务停下来。

**第 2 步的连带面**：

- `--gpu-tier cpu` 会与既有的 `--device cpu` 看起来在说同一件事。**`--gpu-tier cpu
  --device cuda` 直接报错**——那是矛盾请求，静默选一边就是下一个静默错配。
  ⚠ **每个前端各要一个检查点**，因为只有前端分得清「用户选了 cuda」和「默认是 cuda」：
  CLI 在 `pipeline.py` 的 `_pipeline_kwargs`，桌面在 worker 调 `run_pipeline` 之前。
  桌面那处 2026-09-02 复查时才补上，而且**修了三次**：先是后端
  （`TaskRequest.device` 默认写死 `"cuda"`——正是仓库禁止的「第二份默认值」），随后发现
  **前端仍然永远发 `device: "cuda"`**，后端那个 `None` 从 UI 根本到不了。当时不炸只是因为
  档位下拉里还没有 `cpu`；一旦有人把它加进去，每个桌面 `cpu` 档任务都会被拒。所以
  `TaskRequest.device`、`types.ts` 的 `device`、`state.ts` / `bridge.ts` 的默认一起改成
  「不选就不发」，两个新档位也同时进了下拉与两份翻译。**第三次**才修到真正发请求的那一行：
  `page.tsx` 无条件写 `device: processing.device`，而 `processingDevice.AUTOMATIC` 是
  `device: "cuda"`——存储层早就把「自动」存成 `null`，**读取层又映射回了 `"cuda"`**。于是
  「自动 + `cpu` 档」（也就是绝大多数人）会被 worker 以矛盾请求拒掉，而下拉与拒绝是同一个
  提交交付的，等于亲手造出那个回归。
  **教训（这条比结论值钱）**：「默认值只有一份」要一路查到**发请求的那一行**，不是查到状态
  初值为止。中间每一层都可以正确地存 `null`，而最后一层把它翻译回默认值——三层里有两层对，
  合起来仍然是错的。为此把那一行提成了 `requestDeviceFields()`：它之所以是个函数，就是因为
  内联在 `page.tsx` 里的版本没有任何测试够得着。
- ✱ **分离器今天绕过 `resolve_device`，自己判 `cuda_usable()`**（`separation.py` 六处）。
  一个管不到它的 `cpu` 档不是策略，只是一个标签。第 2 步必须列出 `TierSpec.gpu` 的**全部
  消费者**并逐一接上，分离器那六处是其中最容易漏的。
- 同步 desktop 的 `GpuTier`（py + ts）、`test_resource_profiles.py` 的表断言、
  `gpu_tier_help()`、以及 §2.5 那条产物边界的用户向说明。

**第 3 步（做的时候）：必须区分「用户要的 CPU」与「机器只能 CPU」**。`referee_device`
（[`lang_redecode.py:147`](../../src/finesub/speech/recognition/lang_redecode.py)）第一件事是
「`asr_device` 不是 cuda 就返回 cpu」，所以 ASR 一旦落 CPU，裁判也被钉死在 CPU——哪怕显卡
完全空着。但**放开它的条件不是「ASR 在 CPU 上」**：

| ASR 为什么在 CPU 上 | 裁判该怎样 |
| --- | --- |
| 用户显式 `--device cpu` / `--gpu-tier cpu` | **也留在 CPU**。用户是要把卡让给别的程序，裁判跳上去就是违背意图 |
| 能力回退（这台机器的 CT2/torch 用不了 GPU） | 用整档预算上 GPU——那是机器事实，不是意图 |

所以传给 `referee_device` 的不能只是一个设备字符串，得带上「这是请求还是回退」。第 2 步的
`cpu` 档正好提供了前者的表达式，这也是 3 排在 2 之后的原因。

## 5. 否掉的替代方案

- **不加 `cpu` 档，改让 `ResourceProfile` 可为 `None`**（「没有 GPU」= 「没有档位」）。语义更
  干净，但 `resource_profile` 到处传且假定非空，触及面比 `gpu: bool` 大一个量级，收益只是
  概念整洁。**不做**——除非审计认为「档位」这个词本来就不该套在 CPU 上。
- **启动时探一次所有后端、存进 profile。** 被 §2.3 的 9 秒 import 否掉。
- **给 `resolve_device` 一个通用 backend 枚举而不是两个具名分支。** 两个值不值得一个枚举，
  且 CLAUDE.md 的不变量是「`能不能用 GPU` 只住在 `device.py`」——具名分支同样满足。

## 6. 验收怎么做到的（v5 新增）

这份计划两次写下「本机验不了」，两次都太早。记在这里，因为下一个同形的判断值得先试一下
再认输。

### 6.1 在进程内劈开 torch 与 CT2

4b 的前提是「CT2 能用这张卡而 torch 不能」，本机两者都能用——听上去只能换机器。但
`cuda_usable()` 判负的条件是**卡的 compute capability 低于 `torch.cuda.get_arch_list()`
的最小值**，而那个列表是可以替换的：

```python
monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_130"])
```

于是 torch 如实地认为它没有这张卡的 kernel，而 CTranslate2——自带 arch 列表与 CUDA
运行时——**照样看得见真实设备**。这不是打桩：`ct2_cuda_unusable_reason()` 跑的是真的
CT2 查询，卡也是真的。`TestTorchAndCt2Disagree` 里第一条测试专门守着这个前提本身
（`cuda_usable()` False 且 `ct2_cuda_unusable_reason()` None），前提一旦不再成立，
后面两条就不再是它们自称的那个东西。

### 6.2 用 `CUDA_VISIBLE_DEVICES` 跑一遍无卡回归

`CUDA_VISIBLE_DEVICES=-1 pytest -q` 是最接近 CI 的本机复现。第一次跑就红了 9 条，其中
**一条是真 bug，八条是测试假设了宿主有卡**：

- **真 bug**：`cuda_device_present()` 用了 `torch.cuda.is_available()`。实测两种写法：

  | `CUDA_VISIBLE_DEVICES` | `is_available()` | `device_count()` | CT2 count |
  | --- | --- | --- | --- |
  | `-1` | False | 0 | 0 |
  | `""`（空串） | **True** | **0** | **1** |

  空串正是 `device.py` 模块 docstring 里点名的「没有可用 CUDA」情形之一，而
  `is_available()` 对它答 True——于是一台零可见设备的机器被放进 GPU 档。改成
  `device_count() > 0`。（CT2 在空串下仍报 1：它的运行时把空串读成「未设置」。两个库在
  这里真的不一致，而这恰好是「档位问 torch、ASR 问 CT2」的又一个理由——谁也别替谁回答。）
- **八条测试假设了宿主有卡**：裁判放置自 4b 起要问 `cuda_usable()`，所以那些钉**算术**的
  测试在无卡机器上全红。它们本就不该依赖宿主硬件，加了 autouse fixture 把能力钉成 True，
  能力为假的情形由 `TestTorchAndCt2Disagree` 单独钉。**这八条在 CI 上本来会红**。

⚠ 所以「本机绿」对这一族改动是不够的，**两个方向都要跑**：

```bash
pytest -q                          # 有卡
CUDA_VISIBLE_DEVICES=-1 pytest -q  # 无卡（= CI 的形状）
```

### 6.3 为什么没有借 GitHub CI

考虑过建临时分支借 CI 的无卡环境，结论是不值得：CI 装的是 `[harness,dev]` + CPU torch，
**没有 `[asr]`，也就没有 ctranslate2**，所以它测不了 4b 关心的那个不对称——它连 CT2 都
没有。它能测的是「完全没有 GPU」，而那一格 6.2 已经在本机复现了，且 `ci-gate` 下一次推
的时候本来就会跑到。加上公开远端只承载 `main` 的快照线（`publish-main.ps1`），为一次验证
往那里推临时分支的代价与收益不成比例。

## 7. 怎么算做对了

按 §4 的步号（v5：五步全部已实施，所以这些是**回归**面而不是待办）：

- **第 1 步**：`align_meta["qwen_verify"]["device"]` 存在；一次 CPU 裁判与一次 GPU 裁判的
  产物该字段不同且都正确。（其余四个阶段已有记录，回归由现有测试覆盖。）
- **第 2 步**：`--gpu-tier cpu` 的运行不再出现任何显存告警，`align_meta.gpu_tier == "cpu"`，
  `gpu_tier_names()` 的棘轮测试更新；**并且分离器真的落 CPU**——这一条要单独验，因为它走的
  是自己的 seam，不接上就只是个标签。`CUDA_VISIBLE_DEVICES=""` 覆盖「机器没有 GPU」这一格。
- **第 3 步**：两格分别验——`--device cpu` 时裁判**仍在 CPU**（意图），能力回退时
  `referee_vram_budget` 返回整档值而不是减去常驻（事实）。
- **第 4a 步**：两条都要——①构造一个「CT2 无 CUDA」的假替身，断言回退到 CPU 并打
  `cpu-fallback`；②构造一个**`encode` 抛 GEMM `RuntimeError`** 的假替身，断言错误从
  `warm()` 抛到调用方、文案指向 `ct2-patches/README`。第二条是这一步真正的验收：少了热身
  encode 它会绿着通过第一条却什么也没做。
  ⚠ **不要写成「断言分离阶段没有被跑过」**——那条验不过，见 §3 的更正：分离与 VAD 都在
  识别阶段之前，热身 encode 只可能早于**解码循环**。
- **第 4b 步**：~~本机验不了~~——**验得了**，抬高 torch 的 arch 下界即可（§6.1）。
  三条测试：前提本身（`cuda_usable()` False 而 CT2 正常）、ASR 保住那张卡、以及**裁判不跟着
  上去**。最后一条是 4b 单独做会引入的 bug，也是第 3 步存在的理由。
- **两个方向都要跑**：`pytest -q` 与 `CUDA_VISIBLE_DEVICES=-1 pytest -q`（§6.2）。第一次跑
  后者红了 9 条，其中一条是真 bug。

## 8. 已知留白

- **4b 的分支已经验过，未验的是它在现实中有没有用武之地。** 两件事要分开说：分支行为
  （torch 说不行、CT2 说行时 ASR 保住卡、裁判不跟上去）在进程内验过了，§6.1；**没验过的
  是「现实中是否存在一张 CT2 跑得动而这套 torch 跑不动的卡」**——那是从两个库各有 arch
  列表推出来的，只能等一台真机器来证实或证伪。分支正确不等于收益存在。
- CPU 上分离器 worker 数固定为 1，**没有量过**。拆出 `cpu` 档只是让「CPU 是一档」可表达，
  不等于标定了它；真要拿收益还得在 CPU 上跑一次 worker 阶梯。

## 9. 复审改动台账（v1 → v2）

第一轮复审（2026-09-02）提了三条，两条接受、一条换了补法：

| 意见 | 处置 |
| --- | --- |
| §3 第二行规则依赖它自己说不可信的探针 | **接受这个发现**（`get_supported_compute_types('cpu')` 确实不构成证据，且可达路径存在），但**两个建议的补法都不采纳**：真实小 GEMM 探测需要先建模型、与「照常建并翻译错误」是同一件事；「不自动回退、直接报错」会拿走今天正常工作的 CPU-only 运行。改成「照常回退 + 把错误提前到阶段入口」，理由写在 §3 |
| 第 4 步没区分「ASR 为什么落 CPU」 | **接受**。用户显式要 CPU 时裁判也留在 CPU，只有能力回退才放它上 GPU。这条同时决定了它必须排在 `cpu` 档之后（那才是「意图」的表达式），见 §4 |
| 顺序改成 1 → 3 → 4 → 2，并把第 2 步拆两半 | **接受**。v1 把 CT2 探针排第二的理由是「它今天就在制造错误」，但那个错误是**正确的响亮报错**；真正静默的那一半恰恰本机验不了。新顺序 = 先做能验收的 |

两条补充约束也已并入：探针在 `cuda_libs` 之后跑（§2.4，附一条更正——它不会让探针变充分）；
`TierSpec.gpu` 要列全部消费者，分离器那六处 `cuda_usable()` 是最容易漏的（§4 第 2 步）。

一条只记录不处置的：`standard_large_vram` 是四个档位名里唯一带下划线的多词名，CLI 上
打起来别扭。名字是 owner 定的，不改。

### 9.1 第二轮复审（v2 → v3）

| 意见 | 处置 |
| --- | --- |
| §3 否掉小 GEMM 探测的理由（「与照常建模型是同一件事」）不成立：翻译在 `encode()` 里，建模型不经过它，4a 会做成空操作 | **接受，已核**：`fw_refine_backend.py:183` 确实在 `encode` 调用外接住，`FwRefineModelPool.warm()` 也只建不 encode。§3 重写、4a 改成「建模型 + 热身 encode」、§6 加了「假替身 encode 抛错时阶段入口就报错且分离没跑」这一条验收。**第一轮那条建议比 v2 的反驳更对**，v2 的否定理由是错的 |

另两处确认无异议：探针在 `cuda_libs` 之后跑但不因此变充分；第 3 步需要显式的「请求 vs
回退」表达式而不是设备字符串，因此排在 `cpu` 档之后。

### 9.2 实现前自查（v3）

动手做第 1 步时先查了「哪些阶段已经记了设备」，结果推翻了 §2.6 自己的前提：**五个阶段里
四个都记了**，`align_meta["device"]` 尤其记的就是 `resolve_device` **之后**的值。v1 写那条
时只 grep 了 `align_meta[` 的赋值行，没看 `asr_align_metadata()` 的构造参数——**一个只查了
一半的 grep 变成了一条断言**。

第 1 步因此从「给五个阶段加记录」缩成「给尾部裁判加一个字段」。这不改变顺序（它仍是零风险
的第一步），但把它的份量说准。

### 9.3 实施记录（v4）

- **第 1 步**：`apply_verification` 的 stats 加 `device`。同步 `docs/vad-asr.md` 的字段契约。
- **第 2 步**：`cpu` 档落地。两处自查发现的问题一并修了：①`warn_if_vram_is_short` 的
  `smallest = gpu_tier_names()[0]` 在加了 `cpu` 之后会把它当「更小的档」推荐，改成
  `gpu_backed_tiers()[0]`；②我写的断言 `"0GB free VRAM" not in help_text` 本身有 bug
  （`0GB` 是 `10GB` 的子串），改成查 `cpu` 档自己的 summary。
  分离器那条策略接线**做了突变检验**：把 `profile.gpu and cuda_usable()` 改回
  `cuda_usable()`，新测试红。
- **第 4a 步**：`device.ct2_cuda_unusable_reason()`（懒 import）+ `_asr_device_for_ct2` +
  `FwRefineModelPool.warm()` 里的 `_warm_up_encode`。**两条无效化都做了突变检验**：
  「`warm()` 只建不 encode」红 2 条，「ASR 只问 torch」红 1 条——正是第二轮复审指出的那个
  空操作，现在有测试挡着它回来。
- **第 3 步**：查出今天是空操作，推迟（见 §4）。

### 9.5 第二位 reviewer 的三条（v5 之后）

| 意见 | 处置 |
| --- | --- |
| [P1] `--device cpu` 没有让整条任务离开 GPU：`stages.py` 调分离只传 `gpu_tier`，分离器按档位与 `cuda_usable()` 自己决定，`--gpu-tier standard --device cpu` 仍先在卡上分离 | **成立，已修**。`run_vocal_separation` 加 `device`，折叠变成「请求 ∧ 策略 ∧ 能力」，管线把请求传下去；显式 `cpu` 与 `cpu` 档不再打 `cpu-fallback`（那是选择不是回退）。测试：用**能用的卡 + GPU 档**钉住显式 `cpu` 不上卡（`test_an_explicit_cpu_request_keeps_separation_off_a_working_card`），管线层钉住 `device` 真的传到了分离（`test_pipeline_hands_the_requested_device_to_separation`） |
| [P2] 桌面「自动」丢失了「没选」：worker 传 `device=None`，`referee_device` 把 `requested_device=None` 解释成解析后的 ASR 设备，于是 CT2 回退 CPU 时裁判也被锁在 CPU | **成立，已修**。`None` 现在只有一个含义——默认请求 cuda；`run_pipeline` 与 `run_vad_asr` 各归一化一次，`referee_device` 不再回落到 `asr_device`。测试：`test_the_desktop_automatic_puts_the_referee_on_the_idle_card`（CT2 不可用、torch 可用、无人选设备 → 裁判上卡且拿整档预算）、`test_an_unchosen_device_is_the_default_not_the_resolved_asr_device` |
| [P1] 3.8 Flash 未验收就提升为默认纠错模型，应先跑同素材 replay 或先退到 3.7 之后 | **owner 驳回**（2026-09-02 原话大意：3.8 比 3.7 好基本不用想，没什么好保守的，也测不出显著差别）。曾按建议做过一版退后并写了验收协议，随 owner 决定整体撤回；决定记在 `model_routes.toml` 的组注释与 `llm_local_agent_agy.md` §1，别再当待办 |

### 9.4 第三轮复审（v5 复查）

| 意见 | 处置 |
| --- | --- |
| 「错误提前到任务开始 / 分离没跑过」不成立：分离与 VAD 都在识别阶段之前 | **接受，已核**（`stages.py` 先分离，`run_vad_prefix` 又在建池之前）。但改的时候发现它**两边都不准**：省不下分离与 VAD 是对的，而真正的收益比原话大得多——那个错误以前**根本不让任务停下来**（分组循环 `except Exception` → 丢组继续 → 跑完交空字幕）。§3 与 CHANGELOG 按这个重写，§7 那条验不过的验收也改了 |
| 桌面绕过了 `--gpu-tier cpu --device cuda` 的检查 | **接受**。根因是 `TaskRequest.device` 带了第二份默认值（`= "cuda"`），那一层因此分不清选择与默认。改成 `None` 并在 worker 里调同一个检查——两个前端现在答得一样（都报错），而不是「结果一样、响应不同」 |
| 桌面**前端**仍永远发 `device: "cuda"`，后端的 `None` 到不了；下拉里也没有两个新档位，等于给下一个人埋雷 | **接受**。`types.ts` 的 `device` 改成可选、`state.ts` / `bridge.ts` 的默认改成不发；`cpu` 与 `standard_large_vram` 进下拉与 en/zh 两份翻译。前端 58 test + `tsc --noEmit` 均过（借主 checkout 的 node_modules） |
| 前端仍在 `page.tsx` 无条件发 `device`，且下拉与拒绝同一提交交付 = 真回归 | **接受**。`AUTOMATIC.device` 改 `null`、`Settings.tsx` 的「自动」用 `AUTOMATIC`、发请求那一行提成 `requestDeviceFields()`。前端测试补一条走完整条路的，并修掉 `processing-device.test.ts` 里那条把 `readProcessingDevice()` 和自己比较的同义反复——**那正是这个回归没被发现的原因**。突变检验：把 `AUTOMATIC.device` 改回 `"cuda"` 红 2 条 |
| worker 的拒绝没有测试 | **接受**。补两条：显式 `cuda` + `cpu` 档被拒且管线没启动；裸 `cpu` 档照常放行且 `device` 传下去是 `None`。两条都做了突变检验——去掉 worker 里的检查红一条，把 `TaskRequest.device` 默认改回 `"cuda"` 也红一条 |
| `CLAUDE.md` 与设计稿 §4/§8 留着实施前的说法 | **接受**。前者那句「无显卡/不支持/显存不足一律落 entry」与同行新加的内容自相矛盾，已拆成三种情形；§4/§8 改成「分支已验，未验的是现实中有没有这样的卡」 |
