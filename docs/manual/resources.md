# 资源与大文件：位置、迁移与删除

FineSub 安装完成后约占十几个 GB，其中绝大部分为可再生内容（Python 运行环境、AI 依赖、模型权重
与下载缓存）。本文介绍各类数据的位置、迁移方式，以及卸载时的删除范围。

## 两类数据，两个去处

| 数据 | 位置 | 大小 | 说明 |
| --- | --- | --- | --- |
| 个人数据 `user-data` | `%LOCALAPPDATA%\FineSub\user-data` | 十几 MB | 设置(`config.toml`)、API Key(`.env`)、自定义模型表（可选的 `model_catalog.psv`，见[模型路由配置](model-routing.md)）、知识库、任务历史、日志。**所有安装形式共用同一份**。桌面安装版、便携版与 `pip` 安装的 CLI 均读取该目录，知识库不会因使用入口不同而各自独立 |
| 大文件 `models` / `cache` / `tasks` / `agent-capsules` | 默认在**安装目录**下，可整体搬走 | 约 12 GB | 模型权重、下载缓存、任务产物，以及本地 Agent 失败现场（有界的文本与 JSONL） |
| 运行环境 `runtime` | **永远**在安装目录下 | 约 5 GB | Python 3.12 + 锁定的 AI 依赖 |

删除安装目录后，仅保留那十几 MB 的个人数据。

### 为什么 `runtime` 不能单独搬

当缓存与运行环境位于**同一磁盘**时，uv 会通过硬链接将 wheel 从缓存直接链入环境，两侧看起来
各占 5 GB，实际为同一份数据。若将两者分置不同磁盘，硬链接会退化为真实复制，**总占用反而增加
约 5 GB**。

因此，当系统盘空间不足时，应将**整个 FineSub 目录迁移到其他磁盘**（见下文），而非仅迁移大文件
目录；桌面版安装时也可直接选择安装到其他磁盘。

## 第一次安装时选盘（仅托管 CLI）

用安装脚本装 `finesub` 时，会在装完后问一次大文件放哪：

```text
FineSub 将在这里保存模型、下载缓存和任务产物,建议预留至少 20 GB。
直接回车使用:C:\Users\<你>\AppData\Local\FineSub
也可以输入其他绝对路径,例如 D:\FineSub
大文件位置:
```

回车就是用默认位置。这一步只登记位置、**不下载任何东西**，安装仍只要几秒钟；模型和运行环境
在第一次真正处理文件时才下载。

不方便交互时（CI、脚本、重定向的终端）不会等待输入，直接用默认位置，也可以事先指定：

```powershell
finesub setup --dirs-only --data-dir D:\FineSub   # 只登记位置
$env:FINESUB_BIG_DATA_DIR = "D:\FineSub"          # 或者用环境变量
```

安装后如需更改位置，请使用下文的 `finesub relocate`。`--data-dir` 仅在首次安装时生效，不会
迁移已存在的数据。

桌面版不走这一步：它的位置由安装目录决定，装好后同样可以用 `finesub relocate` 调整。

## 搬到别的盘

```powershell
finesub relocate --show              # 先看看现在在哪、各占多少
finesub relocate D:\FineSub          # 把 models/cache/tasks/agent-capsules 搬过去
finesub relocate --reset             # 搬回安装目录
```

同盘迁移仅需重命名，瞬时完成；跨盘迁移按 复制 → 校验 → 删除源 的顺序执行。迁移前要求当前无
任务运行、无安装进行；被拒绝时会说明占用方是哪个前端或任务（也可通过 `finesub doctor` 的
`activity` 行查看）。旧版本遗留的占用可能没有名称，此时仅报告数量。

**中途断电或强制终止也不会丢失数据**：新位置在**迁移开始前**即已登记，旧位置在迁移完成确认前
始终可查。无论何时崩溃，每个目录都位于新位置或旧位置之一，启动时均可找到；程序仅提示「上次
搬迁未完成」，再次运行 `finesub relocate` 即可完成剩余迁移。

**两个安装可共用一份数据**：在第二个安装中同样执行一次 `finesub relocate D:\FineSub` 即可。
若目标已是完整的数据目录，则仅登记、不复制。桌面版与 CLI 安装在同一台机器时，可借此避免重复
下载两份模型。

**你自己做的目录联接(junction)会原样保留**：如果你用 `mklink /J` 把 `models` 或它下面的某个
子目录指到了别处，搬迁只会把这个链接搬过去、重新指向同一个位置，不会把目标里的数据复制一份
到新盘，也不会去动那个位置。它不属于 FineSub，可能还有别的程序在用。卸载同理。

### 用资源管理器搬也可以

- **将整个 FineSub 目录移动到其他位置** → 下次启动时程序会自动就近查找，无需额外处理。
- **仅移动大文件目录（如 `D:\FineSub`）** → 在新位置双击其中的 `register-location.cmd` 即可完成
  位置登记。该脚本与 `.finesub-store.json` 标记随目录一同迁移，无需使用命令行。

登记文件位于 `%LOCALAPPDATA%\FineSub\locations.json`，内容仅为若干路径记录，即使损坏也无碍：
找不到记录时自动回退到安装目录，不会报错。

## 设置文件 `config.toml`

`.env` 存 API Key,`config.toml` 存其余设置——用哪些供应商、模型预设、分句参数之类。
它在**用户数据目录**里（就是上表 `user-data` 那一行的位置），和 `.env`、知识库同级。

**多数情况下无需手动修改。** 桌面端设置页编辑的即是该文件；文件会被自动创建，同时保留手动添加
的其他内容。仅当需要配置设置页未提供的项目（如 [`agent.md`](agent.md) 中的 `preset = "agy"`、
[`model-routing.md`](model-routing.md) 中接入自有模型）时才需手动编辑。

### 它默认不存在，要自己建

**所有设置均为可选项**，缺少该文件也能正常运行；文件不存在仅表示尚未修改任何默认值。
要手写时：

1. 打开用户数据目录（上表那个位置）。
2. 新建一个纯文本文件，文件名**精确**是 `config.toml`。
   ⚠ Windows 记事本默认会加 `.txt` 后缀，存成 `config.toml.txt` 就不会被读到——保存时把
   「保存类型」选成「所有文件」，或存好后在资源管理器里打开「文件扩展名」显示确认一遍。
3. 写你要改的那几行就行，没写到的照旧用默认值：

```toml
[providers]
gemini_free = true

[llm]
preset = "agy"
```

⚠ 请使用 UTF-8 编码（记事本默认）。修改后无需重启，下次运行即生效。

⚠ 源码仓库提供带注释的完整模板 `config.example.toml` 可供参考；该模板**不随安装包发布**，桌面端
与 CLI 用户无法直接获得——上文的几行已足以开始使用，其余配置键在各对应文档中均有示例。

### 有哪些节

模板不随安装包发布，所以这里列一遍**有哪些节**，每一节的键去它的 owner 文档查：

| 节 | 管什么 | 细节在哪 |
| --- | --- | --- |
| `[providers]` | 哪几家供应商可用（Gemini 免费/付费、Exa、Tavily） | [`env.md`](env.md) |
| `[pools]` | 同一家配多把 key 时的池子 | [`env.md`](env.md) |
| `[llm]` | 预设、优先模型、思考档位、代理、风格默认值、本机 agent 的几项 | [`model-routing.md`](model-routing.md)、[`agent.md`](agent.md)、[`knowledge.md`](knowledge.md) |
| `[chunking]` | 单个纠错窗口的输入上限 | [`../llm_harness_behavior.md`](../llm_harness_behavior.md) |
| `[segmentation]` | `length_scale`：字幕长短偏好 | [`tuning.md`](tuning.md) |
| `[vad]` | `silero_assist`：语音检测的二次校正开关 | [`tuning.md`](tuning.md) |
| `[separator]` | `enabled`：要不要跑人声分离（输入已是纯人声时设 `false`） | [`tuning.md`](tuning.md) |
| `[cli]` | `update_check`：新版本提醒 | 本页末节 |

每一项都是可选的，没写的照旧用默认值；命令行给了同一项就以命令行为准。

## 下载走哪条线：镜像与手动覆盖

装模型和运行环境要下几个 GB。FineSub 会**自动**判断从官方源还是国内镜像取，判断结果只影响
「从哪儿取」，不影响取到什么——每个文件都有固定的哈希校验，镜像换不了内容。

看当前判断：

```powershell
finesub doctor
```

里面有一行：

```text
download   cn (自动检测，缓存)  pypi=…  huggingface=…  github=…
```

- 第一段是判定结果(`cn` / `global`)和它的来源（自动检测 / 你设的环境变量）。
- 后面每一类要么是正在用的镜像入口，要么是「官方源」，要么是「已停用（连续失败）」——某类
  镜像连续失败几次后会**自动**放弃、回到官方源，只影响那一类，不影响其它。

### 什么时候需要手动覆盖

使用 VPN 或公司代理时，自动检测的结果可能与实际出口不一致（例如判定为 `cn` 却无法连接镜像，或
判定为 `global` 却走了较慢的线路）。此时可通过环境变量**强制**指定区域，该设置始终优先于自动
检测：

| 环境变量 | 作用 |
| --- | --- |
| `FINESUB_DOWNLOAD_REGION` | `cn` / `global` / `auto`（默认）。非法值当没设 |
| `FINESUB_PYPI_INDEX` | 换 pypi 入口；**设成空值 = 这一类不用镜像，回官方源** |
| `FINESUB_HF_ENDPOINT` | 同上，模型权重 |
| `FINESUB_GITHUB_FILE_PROXY` | 同上，GitHub 上的固定文件 |

从镜像下载模型时，程序会自动设置 `HF_HUB_DISABLE_XET=1`——公共镜像不代理 Hugging Face 的
Xet 传输，若不关闭该选项，下载会在第一个数据块处以 `401` 失败，导致模型无法安装。这与速度无关：
镜像本身仅提供普通 HTTP 下载；官方源不受影响，仍可使用 Xet。当你自行设置 `HF_ENDPOINT` 时，
该变量同样会被关闭（该入口同样不代理 Xet）；若确有支持 Xet 的网关，可自行设置
`HF_HUB_DISABLE_XET=0`，该显式设置不会被覆盖。

```powershell
# 这次运行强制走官方源
$env:FINESUB_DOWNLOAD_REGION = "global"
finesub doctor
```

设完再跑一次 `doctor` 确认那一行变了。要长期生效就设成系统环境变量。

## 清理

```powershell
finesub uninstall                    # 删运行环境、模型、下载缓存
finesub uninstall --purge-tasks      # 连成品字幕一起删
finesub uninstall --purge-user-data  # 连设置、API Key、知识库一起删
```

按「是否可再生」区分：运行环境、模型与缓存删除后会自动重新下载；成品字幕与个人数据删除后无法
恢复，因此需要显式指定。

**若大文件目录已迁移或与其他安装共用**，默认不会被删除（其他安装可能仍在使用）；如需删除，须
添加 `--purge-big-data`。

### 本地 Agent 失败现场

本地 Agent 每次调用会在专用目录中建立一个一次性 episode。transport 成功后立即删除；transport
失败，或成功后的目录清理失败时才保留。保留项是有界的 prompt、事件流、stderr 与 staging 文本，
用于定位失败，不承担 resume 正确性。

Agy 媒体后端还会在同一协调域的 `.agents` 下保存 FineSub 生成的 project id、hook、路径 guard
与 custom agent 定义；它们是可再生的执行配置，随当前域的 `agent-clean` 一起删除。Agy CLI 自己
维护的用户级 project 注册表属于供应商状态，FineSub 不在普通清理时改写它。

打包安装把这些现场放在当前大文件根的 `agent-capsules` 下，随 `finesub relocate` 搬迁，并按
其他大文件相同的卸载规则处理。若大文件根已搬走或被其他安装共用，普通卸载会明确提示保留位置；
确认没有本地 Agent 运行后可清理：

```powershell
finesub agent-clean                    # 只清当前协调域
finesub agent-clean --locate '<JSON>'  # 从日志中的稳定 locator 找到现场
finesub agent-clean --all-domains      # 列出全机目标并要求输入 DELETE
finesub agent-clean --all-domains --force  # 非交互确认
```

`--all-domains` 无法证明其他 checkout 或旧版本已经停止；停掉所有 Agent 是操作者的责任。默认
命令只持有并检查当前域的 activity barrier，不会扫描或删除其他域。

### 关于下载缓存

`cache\uv` 占用数个 GB，但**单独删除几乎不会释放空间**：其中的数据块被运行环境以硬链接引用，
只有最后一个引用消失后才会真正释放。若要彻底回收空间，须**先删除运行环境、再清理缓存**，顺序
相反则无效。每次成功安装新的运行环境后，FineSub 会自动执行一次 `uv cache prune`（仅清理不可达
对象，安全）。

下载目录中偶尔会残留 `<文件名>.part.bad` 文件，表示一次**未通过校验**的下载：FineSub 不会将其
当作有效文件，而是原地改名保留，便于排查是断线还是数据源问题，随后会重新下载。该文件对程序运行
没有任何作用，**可随时删除**；保留它也仅占用该次下载的空间。

## 显卡支持范围与档位

### 哪些显卡能用

| | 型号 |
| --- | --- |
| 支持 | RTX 50 / 40 / 30 / 20 系（含 Ti、SUPER、笔记本版），GTX 1660、GTX 1650，以及数据中心的 V100 / A100 / H100 —— 显存需 ≥4GB |
| 不支持 | GTX 10 系及更早（1080 / 1070 / 1060 / 1050、GTX 9 系等），以及 AMD、Intel 核显 |

不支持的显卡**不会报错，而是自动回退到 CPU**，并在 stderr 输出一条 `Warning:`。回退后仍能生成
正确的字幕，但速度明显变慢（人声分离尤甚），长音频不建议采用此方式；如需从一开始就避免这些警告，
可直接显式指定 `--device cpu`。回退并非整体发生，各阶段的回退情况见下文「这张卡能不能用不是
一个问题，是两个」。

显存自 4GB 起即可使用（`entry` 档要求 3GB **空闲**显存）；更大的显存仅在人声分离阶段带来并行
加速，边际收益有限——档位默认自动检测，详见下文。

### 显卡档位与 CPU 回退

`--gpu-tier` 决定同时开多少个人声分离 worker、以及第二模型校验能用多少显存，
**默认 `auto`，一般不用管**。五档：

| 档位 | 需要**空闲**显存 | 内存 | 分离 worker | `auto` 在多大的卡上选它 |
| --- | ---: | ---: | ---: | ---: |
| `cpu` | 不用显卡 | 8GB | 1 | 机器上没有显卡 |
| `entry` | 3GB | 8GB | 1 | <8GB |
| `standard` | 6.5GB | 8GB | 2 | 8–11GB |
| `standard_large_vram` | 10GB | 8GB | **2** | 12–23GB |
| `high` | 10GB | 8GB | 3 | ≥24GB |

显存那一栏是**空闲**量，不是卡的容量：桌面和驱动本来就占着一部分。

**`cpu` 档**（2026-09-02 自 `entry` 档拆出）表示「本次不使用显卡」。无显卡的机器会自动选择该档；
手动指定的常见场景是将整张显卡让给其他程序。⚠ 该档与 `--device cpu` 并非同一设置的两种写法——
**`--gpu-tier cpu --device cuda` 会直接报错**，因为二者互相矛盾。反之，「GPU 档位 +
`--device cpu`」是正常组合，仅表示本次不使用显卡。

**`standard_large_vram` 的 worker 数与 `standard` 相同，增加的是显存预算**，该预算仅用于第二
模型校验(Qwen)。该校验模型存在更快的编译版本，要求主识别模型之外另有 3.5GB 空闲显存——当主
模型更换为 `large-v3` 或日语微调等大型模型时，`standard` 档不再满足条件，而本档可以。分离
worker 数量不随之增加是刻意设计：第三个 worker **实测反而更慢**，因此 `auto` 仅在显卡 ≥24GB
时才选择 `high`。手动指定 `high` 始终允许，但通常不会带来更高的速度。

**`entry` 档表示「基础 GPU 加速」，并非「弱卡专用」。** 以下三种情况都会选择该档，但结果不同
（无显卡的机器不在此列——它会选择 `cpu` 档）:

| 情况 | 会怎样 |
| --- | --- |
| 显卡支持、空闲显存 ≥3GB | 拿到基础 GPU 加速，正常跑 |
| 当前 PyTorch 版本不支持该显卡 | **人声分离、第二模型校验、silero 辅助回退到 CPU** 并输出 `Warning:`；识别不随之回退，由 CTranslate2 自行判断（见下） |
| 显卡可用但空闲显存不足 | **不会自动切换 CPU**（那会改变产物切分），而是输出警告说明差距，随后继续运行 |

第三种情况之所以继续运行，是因为「不足」不等于「无法运行」:`entry` 档实测峰值占用 2.26GB,
而档位要求为 3GB，略低于要求的显卡仍可完成多数素材。若确实遇到显存不足，请按警告中的提示处理：
关闭占用显存的程序、降低档位，或显式指定 `--device cpu`。

### 「这张卡能不能用」不是一个问题，是两个

人声分离和第二模型校验用 PyTorch，而**识别用的是 CTranslate2**——两个独立的程序库，各有
各的显卡支持列表。所以从 2026-09 起这两件事分开问：识别问 CTranslate2，其余问 PyTorch。
一张 PyTorch 认不出的卡，CTranslate2 可能照样跑得动，以前这种机器的识别会白白落到 CPU
（慢十倍），现在不会。

反过来也成立：装了 CPU-only 的 CTranslate2 时，**识别会回退 CPU 并说明原因**，哪怕分离
用得上显卡。它连 CPU 都没有可用后端时，任务**直接报错**并指向
[`ct2-wheel.md`](ct2-wheel.md)——以前这种情况会安静地交出一个空字幕文件。

**两边都不支持的老显卡（算力低于 7.0，即 GTX 10 系及更早）仍可完整运行**：识别环节除「驱动是否
识别显卡」外，还会检查该卡算力是否在 CTranslate2 的编译范围内，不在则回退 CPU 并说明原因。
因此这类机器上的三个语音阶段均在 CPU 上执行，仅速度较慢——不会报错，也不会生成空字幕。如需
避免这些警告，请从一开始使用 `--device cpu`。

⚠ 反之，**比编译范围更新的显卡不会被拦截**：驱动会进行即时编译，当前的 RTX 50 系列即属此类；
被拦截的只有低于下界的旧显卡。

## 新版本提醒（只有全局 CLI 有）

`finesub` 命令运行结束后，若 PyPI 存在更新的**正式版**，会在 stderr 输出一行提示，并给出
`uv tool upgrade finesub`。该机制仅提醒、从不自动升级；每天至多检查一次，检查与命令在后台并行
进行，失败时完全静默，安装后的首次运行不检查。检查结果缓存在共享数据根目录的
`update-check.json`（与 `tasks/` 的索引同级），**可随时删除**——删除后仅会使下一次重新检查。

关掉：`FINESUB_NO_UPDATE_CHECK=1`，或在 `config.toml` 里写 `[cli] update_check = false`。
桌面端有自己的更新通道，不受这一项影响。

## 从仓库源码运行时

检测到源码 checkout 时，FineSub 使用**仓库自身的数据**(`<repo>/knowledge`、`<repo>/.env`、
`<repo>/.state`)，完全不访问 `%LOCALAPPDATA%`，开发过程中的产物不会混入日常使用的知识库。
如需仓库运行同样接入共享数据，请设置 `FINESUB_CHECKOUT_DATA=0`。

git worktree 会共用**主仓**的知识库和配置。为避免在 worktree 里做的实验提交进主仓知识库，
worktree 中的知识库自动更新默认跳过；确实要写就设 `FINESUB_KNOWLEDGE_WRITE=1`。

源码 checkout 的 Agent episode 不放进仓库：主 checkout 与其 worktree 共用
`%TEMP%\finesub-agent-runtime\<完整域 digest>`，两个独立 clone 则彼此隔离。这里同样只有
transport 失败或成功后的清理失败会留下现场；用 `finesub agent-clean`（源码 checkout 里是
`python -m finesub.llm.agent.agent_cleanup`）清理当前域。设置 `FINESUB_CHECKOUT_DATA=0` 后改用
上面的托管 `agent-capsules` 位置。
