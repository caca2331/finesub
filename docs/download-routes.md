# 下载路线、镜像与模型校验

`src/finesub_bootstrap/` 里下载那一族的 owner 文档：`download_routes` / `download_sources` /
`downloader` / `asset_resolve` / `model_fetch` / `model_ensure` / `hf_verify` / `model_manifest`。

一句话立场：**公共镜像是「更快地取到我们已经知道形状的字节」，不是新的信任根**。lock 文件的
哈希、固定资源的 SHA-256 和模型清单决定内容是否可用；镜像只决定从哪儿取。这条贯穿全文，
下面每个设计都是它的推论。

设计取舍与被否决的做法在本地 `docs/archive/cli-bootstrap-logging-download-plan.md` §5
——它不随仓库发布。面向用户的说明在 [`manual/resources.md`](manual/resources.md)。

## 1. 路线解析

结果只有 `cn` 或 `global`，并记录来源：

```text
FINESUB_DOWNLOAD_REGION=cn|global|auto
  > 近期缓存的 auto 结果（TTL 24h）
  > 公共 IP country endpoint
  > global（超时、离线、响应非法）
```

- 默认 `auto`；**显式环境变量永远优先**，便于 VPN、公司代理和故障排查。非法值当没设。
- 不部署自有 geo 服务。一主一备两个公共 country endpoint，单连接超时 ≤1.5s，**总预算 ≤3s**
  ——一个花掉比它省下的还多的路线判定已经没有意义。失败即 `global`，从不报错。
- 探测走 `network_routes()` 的**首选**路由，所以地区代表下载出口而不是本机物理位置。已知近似：
  `network_routes()` 是「代理路由 + 直连兜底」的列表，探测走了代理而实际下载落到直连是可能的；
  判定以首选路由为准，某次下载实际走了另一条路由时**不拿它的成败更新缓存结论**。
- ✱ **只缓存 `region`、判定时间和 endpoint 名称，绝不保存 IP。** 地址是我们没有理由存的个人
  数据。状态文件放在 `%LOCALAPPDATA%\FineSub` 的小文件里，而不是可能尚未选择的 big-data root。
- 失败的探测**不写缓存**——否则一次离线会把 `global` 钉住 24 小时。

## 2. 镜像表与逐类降级

公共 endpoint 与镜像地址在随版本发布的 `download-sources.json`，不散落在 Python 里。
环境变量提供紧急覆盖，**空值表示禁用该类 cn 加速并回到官方源**：

```text
FINESUB_PYPI_INDEX   FINESUB_HF_ENDPOINT   FINESUB_GITHUB_FILE_PROXY
FINESUB_DOWNLOAD_SOURCES   # 指向另一张表，供演练或分支自定义
```

表随版本发布，意味着某个入口挂掉要等下次发版才能换，而环境变量覆盖只对读文档的人有效。
所以**每类资源自带本机降级**：连续失败达阈值（3 次）后把「本机禁用该类 cn 加速」写进同一个
状态文件，后续运行直接走官方源；一次成功清零计数。降级是**按资源类**的——pypi 停用不影响
huggingface。

`active_mirror()` 是唯一权衡这四个理由（地区不对、没配、被有意清空、本机已放弃）的地方；
每个调用方各自问一遍，是新资源类只兑现其中三个的由来。

`finesub doctor` 打印 `download   cn (自动检测，缓存)  pypi=…  huggingface=…  github=…`，
每类要么是已配置的入口、要么是「官方源」、要么是「已停用（连续失败）」；**不打印 IP**，
也不靠下载大文件去验证镜像。

## 3. 依赖包与 managed runtime

字节分布决定优先级（`desktop/runtime/pylock.win-py312.toml`）：

| 来源 | 体量 |
| --- | --- |
| `files.pythonhosted.org` 的全部 wheel | 约 257 MB |
| `download-r2.pytorch.org` 的 torch / torchaudio / torchvision cu128 | 约 2.75 GB |
| 自建 GitHub Release 上的 patched CT2 | 数十 MB |

普通 PyPI wheel 占依赖下载不到十分之一，**torch 三件套的镜像可行性是这一节的前置判定**。

**不能只设 `UV_DEFAULT_INDEX`**：lock 已锁定 wheel 的绝对 URL，uv 会按这些 URL 下载。构建时
从 canonical global lock 生成 `pylock.win-py312.cn.toml`（`desktop/scripts/make_cn_lock.py`）：

- 只改 artifact URL，不重新解析版本；包名、版本、marker、文件名和 SHA-256 **必须逐项一致**；
- torch 三件套优先于普通 wheel 处理；patched CT2 走与 §5 相同的 GitHub file proxy；
- 两者都只在存在通过验证的加速入口时才改，否则保留原地址——允许「部分加速」而不降低可复现性，
  但「只有普通 wheel 被加速」按前置判定不算达标；
- ✱ 生成器**自带门禁**：生成后立刻自检，不等价就删掉产物并报错，不留一份「看起来生成成功了」
  的 lock。测试对**入库的那两份文件**再跑一次同样的比对（`desktop/backend/tests/test_cn_lock.py`），
  所以重新生成后漂移了也走不到用户机器上。
- ✱ **marker 始终哈希 canonical lock**：两份 lock 只差在谁发货，若按实际安装用的那份计算，
  跨地区就会被判成「依赖变了」而重建一个本来就正确的 5 GB 环境。

`cli/install.ps1` 只认显式的 `FINESUB_PYPI_INDEX`：它跑在 FineSub 装好**之前**，自动判定就在
它即将安装的那份 Python 里，此处用不上。从第一条 `finesub` 命令起路线正常解析。

**哈希不符要按「谁发的货」分方向**：cn lock 保留 canonical 的 sha256，所以 uv 会校验；镜像发错
字节时哈希不符，而这恰恰是官方源能救的情况。把它判为不可重试等于把唯一的补救路径堵死。所以
**从 cn lock 安装时哈希不符 → 回退 canonical 并降级该镜像；从 canonical 安装时哈希不符 →
直接失败**（那时它确实说明文件本身有问题，换台主机也没用）。磁盘错误和解压错误不伪装成网络故障。

managed Python 本体只有在找到兼容 uv 路径契约、可验证且稳定的公共镜像后才设
`UV_PYTHON_INSTALL_MIRROR`；没有合格候选时继续官方源，不把 runtime 安装和 wheel 加速绑一起。

## 4. Hugging Face 模型

faster-whisper 与 Qwen referee 共用 `HF_ENDPOINT`，它设在
`RuntimeEnvironment.worker_context()`——那里已经在设 `HF_HOME`/`TORCH_HOME`/`FINESUB_MODEL_DIR`，
而且这是**两个前端唯一的公共通路**：CLI 根本没有预取阶段，模型是 pipeline 跑起来之后惰性下载的，
只在桌面 prefetch 里设置等于漏掉一半用户。global 不设置；cn 设为表里的 mirror；
✱ **用户已显式设置 `HF_ENDPOINT` 时不覆盖**——他们是有意指向那里的，一个地区猜测不足以推翻它。

为避免公共镜像把可变的 `main` 解析成不同内容，三个生产模型都**固定 revision**。
`model-manifest.json` 记录 repo、revision、必需相对路径、大小和 SHA-256。

> **不许先编哈希占位。** 这份表最初是空的：真实大小与摘要要等实测才有，而编造的哈希比没有
> 哈希更糟——一个永远过不了的校验最后会被关掉，而不是被修好。2026-08-10 填入实测值（摘要取自
> 官方源重下比对与 Hub 的 file-metadata API），2026-08-21 的验收里逐个下载核对过。再加新模型时
> 同样适用。未登记的模型走老路：由拥有它的库自己下载，不声称有额外校验。

**回退必须换进程**：HF mirror 不可用时不能只在同一进程里晚改环境变量，`huggingface_hub` 可能
已在 import 时缓存 endpoint。所以按 endpoint 启动独立子进程，且**粒度是每个模型一个子进程**
——三个模型在同一进程里顺序下载时，整批重试会让已经拿到的 whisper 陪着失败的 qwen 重下 1.6 GB。

**helper 必须住在 `finesub_bootstrap`**，不能留在 `desktop/backend/worker/prefetch.py`：
`finesub` 不允许 import `desktop`。桌面 prefetch 与 pipeline 都从那里调用，前者只保留自己的进度上报。

### 4.1 校验与 marker

**轮询很频繁**（桌面按定时器问「模型在不在」），哈希三个 GB 来回答不是选项。所以完整校验
**只在本进程刚下载完之后跑一次**并写 marker，之后每次只比 marker。

四态（`hf_verify.marker_state`）：

| 状态 | 含义 | 要不要重取 |
| --- | --- | --- |
| `absent` | 没人校验过——早于本机制、或来自别处的缓存 | **不要**。缺席不是损坏的证据，为证明这点重下几个 GB 是另一个 bug |
| `current` | marker 指向当前 manifest 摘要 | 不要 |
| `stale` | 校验过，但 manifest 此后重钉了 | 要 |
| `failed` | 上次校验没通过 | 要 |

`stale` 压过 `failed`：重钉的 manifest 把旧失败一起作废。

✱ **marker 绝不能写进 `snapshots/<revision>/`**：`model_caches._hf_repo_complete` 判完整的最后
一条正是「revision 目录非空」，marker 写进去会让一个中断在建链接之前的空 snapshot 看起来完整，
恰好废掉那个判据。它落在 **repo 目录层**（`models--<org>--<name>/`），跟着权重走——搬盘或删模型
自动失效，不需要另写清理；`existing_hf_home()` 切到 `~/.cache/huggingface` 时 marker 也在那边。

**校验失败要连带删文件**：失败也留 marker（否则留下的非空 snapshot 与健康的旧缓存无从区分），
并把失败的文件删掉，让重试真的重取。snapshot 条目可能是指向 `blobs/` 的**符号链接**——
`huggingface_hub` 能建就建、建不了才整份复制（它**从不硬链接**），所以链接与它指向的 blob
要一起删，否则下次下载会把同一份坏字节重新链回来。两种布局都有测试；符号链接那支在 Windows
上按 skip 处理，CI 的 ubuntu 作业真跑。

✱ **校验在 fallback attempt 之内，不在其后。** 镜像发回「长度对、内容错」的文件时 HTTP 层是
成功的，只有 manifest 校验知道；校验若在 `fetch_with_fallback` 返回之后才跑，发现问题时回退
已经结束，镜像不被记失败，下次还去同一个镜像。现在「下载 + 校验」是同一个 attempt：mismatch
抛 `hf_verify.VerificationMismatch`，`is_mirror_failure` 认它（进程内认异常类型，桌面 prefetch
子进程只有文本穿回来、按 `MISMATCH_MARKER` 短语认），于是记镜像失败、转官方源重取，
**官方源的字节照样再校验一次**——它不比镜像的更可信。

**快路径的三个条件**（`ensure_hf_model`，在位时是一次目录检查，因此可以坐在热路径上）：
repo 完整、**manifest 钉住的那个 revision 在位**（不是任意 revision——加载器拿的就是这个 pin，
缺了会绕开镜像回退去惰性下载）、marker 是 `absent` 或 `current`。三条缺一即走完整路径。

**钉住的 revision 要交到真正的加载器手里**：`RefinedWhisperModel`（经 `FwRefineModelPool`）与
referee 的两个 `from_pretrained` 都带上它，否则 HF 仍可能把 `main` 重解析到别的提交——刚校验过
的是一个 snapshot，装进显存的是另一个。

**门只对 manifest 描述的那个模型开**：`--model` 指了别的模型（小模型、自定义仓库、本地路径）
时预取直接跳过，不先下默认模型的 1.6 GB；qwen referee 侧同理。

两个 stage 入口挂着它：`vad_asr_stage.run_vad_asr` 开头取 whisper，`QwenReferee._ensure_model`
取 referee。两处都**尽力而为**——取不到就让原本的 loader 去报它自己的错，别把前置失败伪装成
阶段失败。

> **为什么不是「先下再跑」**：提前准备好（桌面资源面板、预取）与「跑到那一步才下」是同一条
> 代码路径的两种时机。在**需要该模型的阶段开始时**做一次「确保就位」，已就位就是毫秒级空操作，
> 没就位就在那里下。用户可感知的行为不变，没有新的交互式提示。

## 5. BS-Roformer 固定模型

audio-separator 自己从 GitHub 下 checkpoint，既不受 `HF_ENDPOINT` 控制，也无法复用地区回退。
所以把「下载」和「加载」拆开：`model-manifest.json` 登记固定 URL + 大小 + SHA-256，用
`finesub_bootstrap.downloader` 下载、续传、校验后放进 separator model dir，
`Separator.load_model()` 只加载已存在且校验过的文件。

**需要的是三个文件而不是一个**（2026-08-10 实测，把 `requests.get` 换成一调用就抛异常）：

| 文件 | 大小 | 来源 | 可否钉哈希 |
| --- | --- | --- | --- |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | 639 MB | GitHub release（不可变） | 可 |
| `model_bs_roformer_ep_317_sdr_12.9755.yaml` | 2.3 KB | 同上，模型配置 | 可 |
| `download_checks.json` | 28 KB | **`raw.githubusercontent.com` 的 `main` 分支** | **不可** |

1. `download_checks.json` 是 `list_supported_model_files()` 取的上游模型索引，是这条链路上
   唯一真正来自 `raw.githubusercontent.com` 的文件，**也就是大陆最难连的那个**；预取必须把它
   一起放好，否则 ckpt 和 yaml 都在也照样卡住。
2. 它在上游**可变**，所以 manifest **不能钉它的 SHA-256**——钉了会在上游更新当天全体校验失败。
   按「取一次、不校验」处理，cn 路线下同样可以走 GitHub file proxy。

三者齐备时 `load_model()` **零网络调用**。`run_vocal_separation` 会并发创建多个 Separator、
每个都 `load_model()` 一次；文件已在本地，这只是重复的本地读取，不会放大成并发网络请求。

**哈希不符时隔离 `.bad`，从官方源重新完整下载**，不拼接两个来源的未验证字节。

## 6. 会动的上游：ffmpeg 与 `digest_from`

`runtime-manifest.json` 的五个资产默认按 `url` + `size` + `sha256` 钉死。✱ **ffmpeg 是唯一的
例外**，带 `digest_from: "github-release-api"` 而不带 `size`/`sha256`。

原因和 `download_checks.json` 同类：上游字节会动。BtbN/FFmpeg-Builds 的 `autobuild-<日期>` tag
只保留几天的滚动窗口，而 `latest` tag 下的资产**每次构建都被删了重传**。所以这个位置上
「钉一个哈希」不是选项。

但这里能比「取一次、不校验」做得更好：**GitHub release API 会给出每个资产的 `digest`**，
于是把「哪个版本」和「哪些字节」分开——装机时先问 API 拿到当前资产的 size 与 sha256
（`asset_resolve.resolve_asset`），把它变成普通的 pinned `DownloadAsset` 再交给 `downloader`，
下游的续传边界、进度总量、大小与摘要校验全都不需要知道「有会动的资产」这回事。答案走 API
自己的 TLS 连接，要伪造得同时改掉 API 响应。

**换来的与放弃的**：完整性和续传安全都保住了，放弃的是**可复现性**——隔一天装的两台机器会拿到
不同的 ffmpeg 构建，且都不是我们测过的那个。这笔交易只对「接口面极小且稳定」的工具成立：
管线对 ffmpeg 只用 `-i` / `-ss` / `-t` 和 `ffprobe -show_entries`。✱ 凡是我们依赖其细节行为的
一律继续钉死，`test_runtime_manifest_pins_every_asset_it_can` 把「例外只有 ffmpeg」钉成红线
——再加一个名字进去，得先让那条测试变红。

`version` 是**静态标签而不是构建号**（`n9.0-latest`）：`status()` 比的是 installed == spec.version，
所以装过一次不会每天重下；要把所有人推到新 ffmpeg，改这个标签，用户看到 `outdated`
（提示升级，不阻塞）。

失败面比钉死的多一处：`api.github.com` 得能连上（未认证按 IP 每小时 60 次；一次装机花一次，
只有共用出口才可能撞到，撞到时报的是「稍后重试」而不是下载损坏）。

## 7. 续传与 `.part`

`downloader.download_asset` 的 `.part` 是**可续传的缓存产物**，旁边的 `.part.expect` 记着这份
部分下载是冲着哪个摘要去的：

- 不一致（pin 被 bump，或上游重编了）**直接丢掉重来**，而不是把新构建的尾巴接到旧构建的头上
  ——那种文件只能在下载完之后校验失败，白花一整次。
- **没有 `.expect` 的 `.part` 同样丢弃**：不知道它冲着什么去的，信它可能白花一整次，丢它只
  损失已经花掉的字节。
- 下载锁（`<dest>.lock`）走 OS 文件锁，所以**进程被杀后锁自动释放**，重试不会被尸体卡住；
  `.part` 与 `.expect` 都在，续传照常发 `Range`。有测试真起子进程、写盘途中 kill 来钉这条。
- 服务端忽略 `Range` 返回 200 时，从零重写而不是往后追加。

## 8. 隐私与可观测性

normal 日志每类下载只输出一次选择结果：

```text
下载路线：中国大陆（自动检测，缓存 24h）
Python 依赖：TUNA PyPI；Torch / patched CT2：官方源
模型：HF mirror；BS-Roformer：官方源
```

切换和回退要说明**资源类别与原因**，但不打印完整代理凭据、IP、带 token 的 URL 或 HF token。

## 9. 验证

```powershell
python -m pytest -q test/bootstrap/test_download_routes.py test/bootstrap/test_downloader.py `
  test/bootstrap/test_hf_verify.py test/bootstrap/test_model_ensure.py `
  test/bootstrap/test_model_caches.py test/bootstrap/test_asset_resolve.py
python -m pytest -q desktop/backend/tests/test_cn_lock.py desktop/backend/tests/test_runtime_regional_lock.py
```

默认单测使用 fake HTTP、fake subprocess 和小文件，**不访问公共镜像、不加载模型**。

**实机验收（2026-08-21，境外出口）**：cn lock 的 174/174 artifact 整份下载后逐个哈希全部一致
（3.23 GB，零重试）；地区判定、global 全新安装、破坏哈希、中断续传逐项通过。清单与逐项结果在
本地 `docs/archive/cli-bootstrap-logging-download-plan.md` 文首与 §7.3。**仍未验**：大陆出口才能做的 cn 安装、中途切源与两条路线耗时对比（后者是
「cn 该不该当默认」的唯一判据，代理下没有意义），以及真实拔网线。owner 判定这两项不重要
（2026-08-21）。
