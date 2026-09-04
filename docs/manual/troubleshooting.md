# 故障排查：按症状定位对应文档

本手册各篇末尾都设有故障排查小节，本页将这些内容汇总为**总索引**：按观察到的现象定位对应文档。
先介绍两条通用方法：

- **`finesub doctor`** 检查运行环境状态与各路径，怀疑安装异常时优先运行（源码安装对应的自检见
  [`repo-install.md`](repo-install.md)）。
- **`--log-level verbose`** 输出更详细的日志，记录各阶段耗时、资源占用与恢复细节。运行结束后，
  `<名字>-metadata.json` 中也包含各阶段耗时，LLM 阶段的汇总位于
  `<名字>.llm-artifacts/task-report.md`。

## 安装与启动

| 现象 | 去哪 |
| --- | --- |
| 运行时报缺少 CTranslate2，或版本号仅为 `4.8.1`（无 `+finesub…` 后缀） | [`ct2-wheel.md`](ct2-wheel.md)：本项目必须使用打过补丁的版本 |
| `torch.__version__` 不带 `+cu128`（装成了 CPU 版） | [`repo-install.md`](repo-install.md):torch 三件套必须从 PyTorch 官方索引装 |
| URL 输入报「找不到 yt-dlp」 | 源码安装需自行执行 `uv pip install yt-dlp`；托管 CLI 已内置 |
| 下载模型或运行环境时卡住或极慢 | [`resources.md`](resources.md)「下载走哪条线」：镜像判定与手动覆盖 |

## 显卡与显存

| 现象 | 意思与去处 |
| --- | --- |
| `Warning: ... falling back to CPU` | 当前阶段无法使用显卡，自动改用 CPU：仍能生成正确字幕，仅速度较慢。各阶段的回退情况见 [`resources.md`](resources.md)「显卡档位与 CPU 回退」 |
| `Warning: ... the <档位> tier expects N GiB of free VRAM, but only M GiB is free` | **仅为提醒，不会自动降档**：可关闭占用显存的程序、将 `--gpu-tier` 降低一档，或使用 `--device cpu`。详见上一节 |
| 报错： `--gpu-tier cpu` 与 `--device cuda` 不能同时指定 | 二者互相矛盾；`--gpu-tier cpu` 即表示「本次不使用显卡」 |
| GTX 10 系及更早的老显卡上出现多条回退 CPU 的警告 | 属正常现象：这类显卡的各阶段均在 CPU 上运行，速度较慢但结果正确。如需避免这些警告，可直接使用 `--device cpu`，见 [`resources.md`](resources.md)「这张卡能不能用不是一个问题，是两个」 |
| 运行中因 CUDA 显存不足而中止 | 处理方法同上：关闭占用显存的程序、降低档位，或使用 `--device cpu`。⚠ 显存不足属进程级中止，程序无法捕获 |

## 字幕内容

| 现象 | 去哪 |
| --- | --- |
| **字幕文件为空**，或日志中所有段落均被丢弃 | 多为 CTranslate2 安装异常：[`ct2-wheel.md`](ct2-wheel.md) |
| 字幕太长/太碎 | `--split-length-scale`，见 [`tuning.md`](tuning.md) |
| 专有名词、人名反复识别错误 | 通过 `--extra-info` 提供背景信息；将其收录进知识库([`knowledge.md`](knowledge.md))；识别侧还可使用 `--asr-context terms` |
| 语言判错（整段被当成另一种语言） | 直接给 `--language`；混合语言素材试 `--lang-redecode on`，见 [`tuning.md`](tuning.md) |
| 幻觉、复读没被清掉 | 稳定化档位 `--asr-stabilize-profile`，见 [`tuning.md`](tuning.md) |
| 需了解 LLM 修改了哪些行、哪些行它自身也不确定 | 查看 `-annotated.csv`，见 [`outputs.md`](outputs.md) |
| **修改参数后重跑，结果完全相同** | 阶段按产物是否存在跳过，不比较参数：需删除对应产物，见 [`outputs.md`](outputs.md) |

## LLM 阶段（纠错翻译）

| 现象 | 去哪 |
| --- | --- |
| 提示没有可用的 API key | [`env.md`](env.md):`.env` 的写法与 `finesub keys` 的用法 |
| 429 / 配额耗尽 | 免费档存在每分钟与每日上限。可更换后端或降低 `--llm-difficulty`（见 [`model-routing.md`](model-routing.md)）；使用本机订阅见 [`agent.md`](agent.md)（第 7 节说明如何确认配额确实已耗尽） |
| 启动时打了几条路由告警 | [`model-routing.md`](model-routing.md) 第 6 节逐条解释怎么读 |
| 本机 agent 无法启动、卡住或遗留大量现场文件 | 见 [`agent.md`](agent.md) 的异常处理与 `finesub agent-clean` |
| 某窗口反复失败导致任务停止 | 单句过长时，程序最多将窗口对半拆分一次，若仍失败则停止任务；可缩短素材或更换更强的模型档，见 [`tuning.md`](tuning.md) |

## 知识库与风格

| 现象 | 去哪 |
| --- | --- |
| 修改了 `rendered/` 中的 markdown，本次运行未见生效 | 改动会在**下一次纠错运行**时被收录，见 [`knowledge.md`](knowledge.md) |
| 运行结束后知识库未更新 | 默认 `--knowledge collect` 只读不写；如需写入须传 `--knowledge update` |
| `--style` 提示找不到条目 | 需先创建风格条目，见 [`knowledge.md`](knowledge.md)「记住一套翻译口味」 |

## 批量、中断与续跑

| 现象 | 去哪 |
| --- | --- |
| 需要停止运行 | 按一次 Ctrl-C：不再启动新任务，等待当前任务完成后退出一并结束；再按一次：立即退出。已完成的阶段均保存在磁盘上 |
| `--resume-batch` 被拒绝（超过 7 天 / 仍在运行 / 更换了目录） | [`batch.md`](batch.md)：三种拒绝情况各自的处理方式 |
| 批次中某项失败，其余任务是否继续 | 单项失败不影响其他任务；重新运行即续跑，见 [`batch.md`](batch.md) |

## 配置未生效

| 现象 | 去哪 |
| --- | --- |
| 已写 `config.toml` 但似乎未被读取 | 多为文件名被存成了 `config.toml.txt`，或文件放错了目录，见 [`resources.md`](resources.md)「设置文件 `config.toml`」 |
| 更换磁盘后模型被重新下载 | [`resources.md`](resources.md)「搬到别的盘」：哪些目录可迁移，以及迁移后缓存为何需留在原处 |

## 仍未解决

提交 issue 时附上以下内容，可减少沟通往返：

1. 完整命令行（对路径中的隐私信息进行脱敏）。
2. `finesub doctor` 的输出。
3. `<名字>-metadata.json`；若运行涉及 LLM 阶段，还需附上
   `<名字>.llm-artifacts/task-report.md`。
4. 相关 `Warning:` 或报错的**完整**一行——其括号内通常已说明处理方法。
