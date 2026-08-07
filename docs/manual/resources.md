# 资源与大文件：装在哪、怎么搬、怎么删

FineSub 装完之后占十几个 GB，绝大部分是可以再生的（Python 运行环境、AI 依赖、模型权重、
下载缓存）。这篇讲它们各自在哪、怎么挪到别的盘、以及卸载时哪些会被删。

## 两类数据，两个去处

| 数据 | 位置 | 大小 | 说明 |
| --- | --- | --- | --- |
| 个人数据 `user-data` | `%LOCALAPPDATA%\FineSub\user-data` | 十几 MB | 设置、API Key、知识库、任务历史、日志。**所有安装形式共用同一份**——桌面安装版、便携版、`pip` 安装的 CLI 都读它，所以你的知识库不会因为换个入口就变成另一个 |
| 大文件 `models` / `cache` / `tasks` | 默认在**安装目录**下，可整体搬走 | 约 12 GB | 模型权重、下载缓存、任务产物（人声、对齐数据、成品字幕） |
| 运行环境 `runtime` | **永远**在安装目录下 | 约 5 GB | Python 3.12 + 锁定的 AI 依赖 |

删掉安装目录，剩下的就只有那十几 MB 的个人数据。

### 为什么 `runtime` 不能单独搬

uv 在缓存与运行环境处于**同一个磁盘**时，会用硬链接把 wheel 从缓存直接链进环境——两边看起来
各占 5 GB，实际上是同一份数据。把它们分到两个盘，硬链接就变成了真复制，**总占用反而多出约 5 GB**。

所以：**系统盘不够，就把整个 FineSub 文件夹搬到别的盘**（见下），而不是只搬大文件目录。
桌面版安装时也可以直接选装到别的盘。

## 搬到别的盘

```powershell
finesub relocate --show              # 先看看现在在哪、各占多少
finesub relocate D:\FineSub          # 把 models/cache/tasks 搬过去
finesub relocate --reset             # 搬回安装目录
```

同一个磁盘内是改名，秒完成；跨盘会复制→校验→改登记→删源，任何时刻都不会只剩半份。
搬之前要求当前没有任务在跑、也没有安装在进行。

**两个安装共用一份**：在第二个安装上也执行一次 `finesub relocate D:\FineSub` 即可——目标已经是
一份完整的数据目录时它只登记、不复制。桌面版和 CLI 装在同一台机器上时，这样就不用下两份模型。

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

### 关于下载缓存

`cache\uv` 有好几个 GB，但**单独删它几乎不会释放空间**——里面的数据块被运行环境硬链接着，
只有最后一个引用消失时才真正释放。想彻底回收，顺序是**先删运行环境，再清缓存**，反过来无效。
每次成功装好新运行环境后 FineSub 会自动跑一次 `uv cache prune`（只清不可达对象，安全）。

## 从仓库源码运行时

检测到源码 checkout 时，FineSub 用**仓库自己的**数据（`<repo>/knowledge`、`<repo>/.env`、
`<repo>/.state`），完全不碰 `%LOCALAPPDATA%`——开发跑出来的东西不会混进你日常用的知识库。
想让仓库运行也接到共享数据，设 `FINESUB_CHECKOUT_DATA=0`。

git worktree 会共用**主仓**的知识库和配置。为避免在 worktree 里做的实验提交进主仓知识库，
worktree 中的知识库自动更新默认跳过；确实要写就设 `FINESUB_KNOWLEDGE_WRITE=1`。
