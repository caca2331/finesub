# 资源与大文件：装在哪、怎么搬、怎么删

FineSub 装完之后占十几个 GB，绝大部分是可以再生的（Python 运行环境、AI 依赖、模型权重、
下载缓存）。这篇讲它们各自在哪、怎么挪到别的盘、以及卸载时哪些会被删。

## 两类数据，两个去处

| 数据 | 位置 | 大小 | 说明 |
| --- | --- | --- | --- |
| 个人数据 `user-data` | `%LOCALAPPDATA%\FineSub\user-data` | 十几 MB | 设置（`config.toml`）、API Key（`.env`）、自定义模型表（可选的 `model_catalog.psv`，见[模型路由配置](model-routing.md)）、知识库、任务历史、日志。**所有安装形式共用同一份**——桌面安装版、便携版、`pip` 安装的 CLI 都读它，所以你的知识库不会因为换个入口就变成另一个 |
| 大文件 `models` / `cache` / `tasks` / `agent-capsules` | 默认在**安装目录**下，可整体搬走 | 约 12 GB | 模型权重、下载缓存、任务产物，以及本地 Agent 失败现场（有界的文本与 JSONL） |
| 运行环境 `runtime` | **永远**在安装目录下 | 约 5 GB | Python 3.12 + 锁定的 AI 依赖 |

删掉安装目录，剩下的就只有那十几 MB 的个人数据。

### 为什么 `runtime` 不能单独搬

uv 在缓存与运行环境处于**同一个磁盘**时，会用硬链接把 wheel 从缓存直接链进环境——两边看起来
各占 5 GB，实际上是同一份数据。把它们分到两个盘，硬链接就变成了真复制，**总占用反而多出约 5 GB**。

所以：**系统盘不够，就把整个 FineSub 文件夹搬到别的盘**（见下），而不是只搬大文件目录。
桌面版安装时也可以直接选装到别的盘。

## 第一次安装时选盘（仅托管 CLI）

用安装脚本装 `finesub` 时，会在装完后问一次大文件放哪：

```text
FineSub 将在这里保存模型、下载缓存和任务产物，建议预留至少 20 GB。
直接回车使用：C:\Users\<你>\AppData\Local\FineSub
也可以输入其他绝对路径，例如 D:\FineSub
大文件位置：
```

回车就是用默认位置。这一步只登记位置、**不下载任何东西**，所以安装还是几秒钟；模型和运行环境
在第一次真正处理文件时才下载。

不方便交互时（CI、脚本、重定向的终端）不会等待输入，直接用默认位置，也可以事先指定：

```powershell
finesub setup --dirs-only --data-dir D:\FineSub   # 只登记位置
$env:FINESUB_BIG_DATA_DIR = "D:\FineSub"          # 或者用环境变量
```

装好之后再想换位置，用下面的 `finesub relocate`——`--data-dir` 只在第一次有效，它不会搬动
已经存在的数据。

桌面版不走这一步：它的位置由安装目录决定，装好后同样可以用 `finesub relocate` 调整。

## 搬到别的盘

```powershell
finesub relocate --show              # 先看看现在在哪、各占多少
finesub relocate D:\FineSub          # 把 models/cache/tasks/agent-capsules 搬过去
finesub relocate --reset             # 搬回安装目录
```

同一个磁盘内是改名，秒完成；跨盘是复制→校验→删源。搬之前要求当前没有任务在跑、也没有安装在进行。

**中途断电或强制结束也不会丢数据**：新位置在**开始搬之前**就已登记，旧位置在搬完确认前一直保持
可查。所以任何时刻崩溃，每个目录要么在新位置、要么在旧位置，启动时都找得到——只是会提示
「上次搬迁未完成」，再跑一次 `finesub relocate` 把剩下的搬完即可。

**两个安装共用一份**：在第二个安装上也执行一次 `finesub relocate D:\FineSub` 即可——目标已经是
一份完整的数据目录时它只登记、不复制。桌面版和 CLI 装在同一台机器上时，这样就不用下两份模型。

**你自己做的目录联接（junction）会原样保留**：如果你用 `mklink /J` 把 `models` 或它下面的某个
子目录指到了别处，搬迁只会把这个链接搬过去、重新指向同一个位置，不会把目标里的数据复制一份到
新盘，也不会去动那个位置——它不属于 FineSub，可能还有别的程序在用。卸载同理。

### 用资源管理器搬也可以

- **整个 FineSub 文件夹被你拖到别处** → 下次启动自动就近找到，不用管。
- **只把大文件目录（如 `D:\FineSub`）拖到别处** → 到新位置双击里面的 `register-location.cmd`，
  它会把新位置登记好。这个脚本和 `.finesub-store.json` 标记一起随目录走，不需要命令行。

登记文件在 `%LOCALAPPDATA%\FineSub\locations.json`，只有一行内容，坏了也不要紧：找不到就自动
回落到安装目录，不会报错。

## 清理

```powershell
finesub uninstall                    # 删运行环境、模型、下载缓存
finesub uninstall --purge-tasks      # 连成品字幕一起删
finesub uninstall --purge-user-data  # 连设置、API Key、知识库一起删
```

按"能不能再生"分的：运行环境、模型、缓存删了会重新下载；成品字幕和个人数据删了就没了，
所以要显式指定。

**大文件目录如果已经搬走或与别的安装共用**，默认不会被删（另一个安装多半还在用它），
要删得加 `--purge-big-data`。

### 本地 Agent 失败现场

本地 Agent 每次调用会在专用目录中建立一个一次性 episode。transport 成功后立即删除；transport
失败，或成功后的目录清理失败时才保留。保留项是有界的 prompt、事件流、stderr 与 staging 文本，
用于定位失败，不承担 resume 正确性。

Agy 媒体后端还会在同一协调域的 `.agents` 下保存 FineSub 生成的 project id、hook、路径 guard 与
custom agent 定义；它们是可再生的执行配置，随当前域的 `agent-clean` 一起删除。Agy CLI 自己维护的
用户级 project 注册表属于供应商状态，FineSub 不在普通清理时改写它。

打包安装把这些现场放在当前大文件根的 `agent-capsules` 下，随 `finesub relocate` 搬迁，并按其他
大文件相同的卸载规则处理。若大文件根已搬走或被其他安装共用，普通卸载会明确提示保留位置；可在
确认没有本地 Agent 运行后清理：

```powershell
finesub agent-clean                    # 只清当前协调域
finesub agent-clean --locate '<JSON>'  # 从日志中的稳定 locator 找到现场
finesub agent-clean --all-domains      # 列出全机目标并要求输入 DELETE
finesub agent-clean --all-domains --force  # 非交互确认
```

`--all-domains` 无法证明其他 checkout 或旧版本已经停止；停掉所有 Agent 是操作者的责任。默认命令
只持有并检查当前域的 activity barrier，不会扫描或删除其他域。

### 关于下载缓存

`cache\uv` 有好几个 GB，但**单独删它几乎不会释放空间**——里面的数据块被运行环境硬链接着，
只有最后一个引用消失时才真正释放。想彻底回收，顺序是**先删运行环境，再清缓存**，反过来无效。
每次成功装好新运行环境后 FineSub 会自动跑一次 `uv cache prune`（只清不可达对象，安全）。

下载目录里偶尔会留下 `<文件名>.part.bad`。那是一次**校验没通过**的下载：FineSub 不会
拿它冒充成品，而是原地改名留着，好让你（或我们）看一眼是断线还是源出了问题，然后重新
下载。它对程序没有任何用处，**随时可以删**；不删也只是占着那一次下载的空间。

## 从仓库源码运行时

检测到源码 checkout 时，FineSub 用**仓库自己的**数据（`<repo>/knowledge`、`<repo>/.env`、
`<repo>/.state`），完全不碰 `%LOCALAPPDATA%`——开发跑出来的东西不会混进你日常用的知识库。
想让仓库运行也接到共享数据，设 `FINESUB_CHECKOUT_DATA=0`。

git worktree 会共用**主仓**的知识库和配置。为避免在 worktree 里做的实验提交进主仓知识库，
worktree 中的知识库自动更新默认跳过；确实要写就设 `FINESUB_KNOWLEDGE_WRITE=1`。

源码 checkout 的 Agent episode 不放进仓库：主 checkout 与其 worktree 共用
`%TEMP%\finesub-agent-runtime\<完整域 digest>`，两个独立 clone 则彼此隔离。这里同样只有 transport
失败或成功后的清理失败会留下现场；用 `finesub agent-clean`（源码 checkout 里是
`python -m finesub.llm.agent.agent_cleanup`）清理当前域。设置 `FINESUB_CHECKOUT_DATA=0` 后改用上面的托管
`agent-capsules` 位置。
