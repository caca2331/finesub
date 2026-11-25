# 使用手册

写给使用者的文档都在这个目录里。按你现在要做的事挑一份读，不必从头看到尾。

实现层的算法、契约与实验记录在 `docs/` 根下（开发者参考，用户可跳过），入口是 [开发者说明](../../README_DEV.md)；两边的分界是读者而不是主题，约定见 [docs/README.md](../README.md)。

## 第一次上手

| 想做什么 | 看哪份 |
| --- | --- |
| 用 `uv tool install finesub` 装安装版（自带运行环境与依赖，只支持 Windows x64） | [README「快速开始」](../../README.md#快速开始) |
| 配 Gemini／Exa／Tavily 的 API key，写进 `.env` | [环境变量与 API Key](env.md) |
| 升级安装版（`uv tool upgrade finesub`），或卸载它（要按顺序跑两条命令） | [安装版命令说明「卸载」](../../cli/README.md#卸载) |

从源码装（Linux / macOS 只能这样）：clone 仓库、装依赖、装完自检见 [仓库安装](repo-install.md)；ASR 必需的补丁版 CTranslate2（识别引擎），或怀疑装成了原版，见 [安装补丁版 CTranslate2](ct2-wheel.md)。

## 跑起来之后

| 想做什么 | 看哪份 |
| --- | --- |
| 弄清任务目录里每个文件是什么、`--stage` 停在哪一档、低置信度的行去哪看 | [运行产物](outputs.md) |
| 字幕太长或太短；调识别侧与 LLM 侧的参数；想知道大概要跑多久 | [调参](tuning.md) |
| 一次跑多个输入：manifest 怎么写、运行期加任务与插队、中断后续跑 | [一次跑多个输入](batch.md) |
| 判出每行字幕是谁说的（`--diarize`）：怎么开、⚠ 模型许可、GPU 与 CPU、多出来的产物；⚠ 成品译文不标说话人，能直接按人分轨导出的只有识别原文 | [说话人分离](diarization.md) |
| 选转写后端（`--asr`）：本机 Whisper、本机显卡上的开源 MOSS，或直接给出时间与说话人的云端 MAI2、Scribe、MOSS Pro、Gemini 3.5 Transcribe，以及云端的文字配本机 MOSS 说话人的 `moss-diarization-aligner`——几家怎么比、要跑多久、隐私与费用、额度与 key；续跑用的是谁 | [转写后端](asr-backend-choice.md) |
| 素材是中文：只纠错不翻译、统一简体、推荐怎么转写 | [中文素材](chinese.md) |
| 让它记住专有名词与译名，或沿用一套翻译口味（`--style`） | [知识库](knowledge.md) |
| 自己开一个共享知识库给别人投稿、拉取 | [自建共享库](share-server.md) |

## 模型、额度与硬件

| 想做什么 | 看哪份 |
| --- | --- |
| 选哪个 Whisper；分离器和第二模型各起什么作用；几家 LLM 后端实际好不好用 | [模型选择](models.md) |
| 改某个任务用哪个模型、换预设、调思考档位、走代理 | [模型路由配置](model-routing.md) |
| 接自己的 API 端点（DeepSeek、OpenRouter 等），或改出厂模型的限额、输出上限 | [接自己的模型](custom-models.md) |
| 用本机已登录的 AI 订阅（或自备 key 的本机 agent）代替 Gemini API 额度：Codex／Claude Code／Antigravity，以及 DeepSeek Harness、CodeBuddy | [本机 Agent 后端](agent.md) |
| 数据存在哪、怎么搬到别的盘、卸载时各项删什么留什么；显卡支持到哪、档位与 CPU 回退；哪些平台支持到什么等级（Linux / AMD / macOS） | [资源与大文件](resources.md) |

## 出问题了

先查 [故障排查](troubleshooting.md)。它按症状（回退 CPU、显存告警、空字幕、429、设置没生效……）指到对应的那一页，本身不放正文。

## 交给 agent 做

手边有 Claude Code 这类编码 agent、又有一份源码仓库的话（这几件事要在仓库里做，只装了安装版不行），审一次运行结果、把参考资料收进知识库、打一个反馈包这些事可以直接交代给它——说什么话它就去读哪一份，见 [agent 任务](agent-tasks.md)。⚠ 它收资料写进的是仓库自己的知识库，安装版日常用的那份收不到，见同一页开头。
