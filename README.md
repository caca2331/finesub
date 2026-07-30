# finesub

把长音频变成精修级中文字幕。

```text
音频/视频 → 人声分离 → VAD + ASR → 稳定化 → LLM 纠错翻译 → 成品 SRT
```

## 效果

以主播游戏实况（日语）为例：

**时间轴**——原生 Whisper 输出 vs 稳定化后的 raw 轴：

```
# 原生 Whisper（幻觉 + 磎片 + 超长句）
00:00:24.500 --> 00:00:27.800  ご視聴ありがとうござい          ← ASR幻觉
00:00:27.800 --> 00:00:29.100  あの、なんだっけ?              
00:00:29.100 --> 00:00:30.200  メミじゃなくて、                ← 碎片
00:00:30.200 --> 00:00:31.500  ユメじゃなくて                  ← 碎片
00:00:34.800 --> 00:00:42.000  じゃあなんかがなんかして 最後の遺産が時が来てそれを得て  ← 超长句

# 稳定化后（幻觉丢弃、碎片合并、超长句拆分、时间精准）
00:00:27.800 --> 00:00:29.000  あの、なんだっけ?             
00:00:29.600 --> 00:00:31.200  メミじゃなくて、ユメじゃなくて
00:00:34.800 --> 00:00:37.850  じゃあなんかがなんかして
00:00:38.144 --> 00:00:42.067  最後の遺産が時が来てそれを得て
```

**纠错翻译**——结合音频/画面语境：


| raw ASR                 | 纠错翻译后             |
| ----------------------- | ----------------- |
| `ほんまに?新書に変わってる。あ、ほんとだ。` | 真的吗？换成新衣服了。啊，真的耶。 |
| `ネジがやっぱ分かりやすいな`         | 发条果然很明显呢。         |
| `ごめんごめん 怖どらないで`         | 抱歉抱歉，别害怕。         |


核心亮点：

1. **精准时间轴**——稳定化后的 raw 轴低幻觉、高召回：BGM/静默段不会产生幽灵字幕，真实语音不会被吞，时间边界精确到帧。
2. **翻译harness**——高度优化的prompt，包含自维护的知识库，主播常用术语、角色名、游戏专名会自动积累并应用到后续窗口，越跑越准。



## 快速开始

Windows 用户也可以使用可选的 [FineSub Desktop](desktop/README.md) 图形客户端来创建任务、管理资源和查看日志；它复用同一套 pipeline，不取代命令行。
>**desktop版尚不完善，追求稳定的话仍建议使用命令行工具。不会用的话可以问ai agent。**

需要 **Python 3.12+**。

```powershell
# 安装（ASR 全栈 + LLM 层）
pip install -e ".[asr,harness]"
```

一条命令出字幕：

```powershell
# 音频，视频输入都可
asr-pipeline data/<输入名>.mp4 --language ja --extra-info "主播四月一日，原神直播切片" --stage final-srt --knowledge update

# URL 也行
asr-pipeline "https://www.bilibili.com/video/BVxxxx" --stage final-srt --name "四月一看PV"
```

其中：
- 不传`--language`时自动检测语言；
- `--extra-info`提供背景信息（主播名、游戏名、关键专名等），能显著提升纠错准确率，非必须。
- 不传 `--stage` 则默认停在 raw SRT（ASR结果，不调 API）；加 `--stage final-srt` 跑 LLM 纠错翻译（需要 `.env` 中配好 Gemini API key，推荐再配上Exa API key；都是免费的，见 [环境配置](docs/manual/env.md)）。
- 传 `--knowledge update` 可在纠错后自动更新本地知识库（主播术语、角色名等），下次跑同一主播时自动注入。不加则不更新。
- 传 `--name` 以指定和覆盖输入名。
- 显存够的话可以额外传 `--gpu-budget-gb 8`，语音识别阶段会并行提速。如果卡比较好可传12或16，但边际收益有限。

跑完后去 `out/<输入名>/` 里找字幕：`<输入名>.srt`（成品）和 `<输入名>-raw.srt`（未纠错原文）。



## 它做了什么

1. **人声分离**——去掉 BGM 和音效，只留人声。
2. **VAD + ASR 对齐**——切分语音段、跑 Whisper、输出带时间戳的逐句转写。
3. **ASR 稳定化**——去噪、合并碎片、丢弃幻觉，输出干净且时间精准的 raw 轴。
4. **LLM 纠错翻译**——结合音频/画面语境纠正误听、翻译成中文、进一步合并和丢弃，输出成品字幕。
- 多模态纠错：结合音频/画面纠正 ASR 误听（专名、同音词、口误）
- 翻译成自然中文（不是机翻味）
- 合并碎片成完整句（严守时长/字数门槛）
- 丢弃复读幻觉、套话、无意义填充词
- 输出置信度标注，低置信行建议人工核对
- 自动积累知识库：主播术语、角色名、常用表达会写入本地知识库，下次跑同一主播时自动注入，越用越准

> LLM路线/档位、知识库、搜索代理、token 预算等细节见 [LLM Harness 行为](docs/llm_harness_behavior.md) 和 [知识库说明](docs/knowledge.md)。

## 批量运行

```powershell
# 多个输入
python -m asr_playground.batch data/a.wav data/b.mp4 --stage final-srt --language ja

# JSONL manifest
python -m asr_playground.batch --manifest tasks.jsonl --knowledge update
```

单项失败不影响其余，重跑即续跑。

## 输出文件

以 `data/input.mp4` 跑到 `--stage final-srt` 为例：


| 文件              | 说明                   |
| --------------- | -------------------- |
| `input-raw.srt` | 未纠错原文 SRT            |
| `input.srt`     | **成品 SRT**（纠错翻译+后处理） |


全部产物归到 `out/input/` 一个目录下。完整产物树见 [README_DEV.md](README_DEV.md)。

## 环境要求


| 阶段         | 需要                            |
| ---------- | ----------------------------- |
| 人声分离 + ASR | NVIDIA GPU（≥4GB 显存）、≥8GB 内存   |
| LLM 纠错翻译   | 无需 GPU；≥4GB 内存；PATH 上有 ffmpeg |


无 GPU 时 ASR 回退 CPU（慢很多）。URL 输入另需 `pip install yt-dlp`。

## 文档

- [环境配置](docs/manual/env.md)——API key 配置
- [开发者说明](README_DEV.md)——架构、产物与调试
- [LLM Harness 行为](docs/llm_harness_behavior.md)——LLM 运行时行为
- [知识库说明](docs/knowledge.md)——知识库
- [ASR 稳定化](docs/asr-stabilize.md)——ASR 稳定化规则
- [测试说明](docs/testing.md)——测试
- [知识库样板](examples/knowledge/)——迷你骨架条目

---

代码 [MIT](LICENSE)；`src/llm/prompt_templates/` 下的 prompt 明文 [CC BY-SA 4.0](src/llm/prompt_templates/LICENSE.md)。

如果觉得有用，欢迎点个 [Star](https://github.com/caca2331/finesub) ⭐

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=caca2331/finesub&type=Date)](https://www.star-history.com/#caca2331/finesub&Date)
