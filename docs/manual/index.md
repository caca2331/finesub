# 使用手册

写给使用者的文档都在这个目录里。按你现在要做的事挑一份读，不必从头看到尾。

实现层的算法、契约与实验记录在 `docs/` 根下，入口是 [开发者说明](../../README_DEV.md)；两边的分界是读者而不是主题，约定见 [docs/README.md](../README.md)。

## 第一次上手

| 想做什么 | 看哪份 |
| --- | --- |
| 从源码装：clone 仓库、装依赖、装完自检 | [仓库安装](repo-install.md) |
| 装 ASR 必需的 patched CTranslate2，或怀疑装成了原版 | [CT2 wheel](ct2-wheel.md) |
| 配 Gemini／Exa／Tavily 的 API key，写进 `.env` | [环境变量与 API Key](env.md) |

托管版 CLI（`finesub`）自带运行环境与依赖，装它不需要前两份——但它**只支持 Windows x64**，
Linux / macOS 请走[仓库安装](repo-install.md)。

## 跑起来之后

| 想做什么 | 看哪份 |
| --- | --- |
| 弄清 `out/<名字>/` 里每个文件是什么、`--stage` 停在哪一档、低置信度的行去哪看 | [运行产物](outputs.md) |
| 字幕太长或太短；调识别侧与 LLM 侧的参数；想知道大概要跑多久 | [调参](tuning.md) |
| 一次跑多个输入：manifest 怎么写、运行期加任务与插队、中断后续跑 | [一次跑多个输入](batch.md) |
| 给字幕标上是谁说的（`--diarize`）：怎么开、⚠ 模型许可、GPU 与 CPU、多出来的产物 | [说话人分离](diarization.md) |
| 把转写交给一个直接给出时间与说话人的模型（`--asr`）：云端的 MAI2、MOSS Pro、Gemini 3.5 Transcribe（隐私与费用、额度与 key），或本机显卡上的开源 MOSS；续跑用的是谁 | [一体转写](joint-asr.md) |
| 素材是中文：只纠错不翻译、统一简体、推荐怎么转写 | [中文素材](chinese.md) |
| 让它记住专有名词与译名，或沿用一套翻译口味（`--style`） | [知识库](knowledge.md) |
| 自己开一个共享知识库给别人投稿、拉取 | [自建共享库](share-server.md) |

## 模型、额度与硬件

| 想做什么 | 看哪份 |
| --- | --- |
| 选哪个 Whisper；分离器和第二模型各起什么作用；几家 LLM 后端实际好不好用 | [模型选择](models.md) |
| 改某个任务用哪个模型、调思考档位、接自己的 API endpoint | [模型路由配置](model-routing.md) |
| 用本机已登录的 AI 订阅额度代替 Gemini API 额度：Codex／Claude Code／Antigravity，以及 DeepSeek Harness、CodeBuddy | [本机 Agent 后端](agent.md) |
| 数据存在哪、怎么搬到别的盘、怎么卸干净；显卡支持到哪、档位与 CPU 回退；哪些平台支持到什么等级（Linux / AMD / macOS） | [资源与大文件](resources.md) |

## 出问题了

先查 [故障排查](troubleshooting.md)。它按症状（回退 CPU、显存告警、空字幕、429、设置没生效……）指到对应的那一页，本身不放正文。

## 交给 agent 做

手边有 Claude Code 这类编码 agent 的话，审一次运行结果、把参考资料收进知识库、打一个反馈包这些事可以直接交代给它——说什么话它就去读哪一份，见 [agent 任务](agent-tasks.md)。
