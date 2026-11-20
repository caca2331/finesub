# 自己开一个共享知识库

知识库可以在多人之间共享：一方当维护者，其余人投稿与拉取。这一份写给**想自己开一个**的人——
只用别人的库，看 [知识库](knowledge.md) 的「和别人共用一个知识库」就够了。

先说清楚这件事的性质：**审核是人工的**。别人投稿之后要你亲自过目、亲自拍板才会进库，
没有"设好就不用管"这条路。所以在开之前值得先想清楚，你愿不愿意定期花这个时间。

开一个库要先 clone 一份源码仓库（`git clone https://github.com/caca2331/finesub.git`）：信箱的 Worker 与整台服务端的部署文件都在里面，
审核时在本机起服务（`python -m finesub.llm.knowledge.share.server`）也只能从源码跑，环境按[仓库安装](repo-install.md)装。
其余命令下文按安装版的 `finesub knowledge-share …` 写；从源码运行时换成 `python -m finesub.llm.knowledge.share …`，参数一样。
⚠ 两个入口读的 `config.toml` 与 `.env` 不是同一份：安装版读 `user-data` 目录那份，源码运行读仓库根那份。同一个库的收发与审核**统一用一种入口**，
免得一边读到 `[share]` 的设置与 key、另一边读不到。

## 两种搭法，先选一种

|  | **信箱**（推荐） | **整台服务端** |
| --- | --- | --- |
| 公网上放什么 | 一个 Cloudflare Worker，只收投稿、发快照 | 完整的服务端 |
| 需要常开的机器 | 不需要 | 需要一台（VPS，或本机 + 内网穿透） |
| 内容在哪被审 | 你本机 | 那台机器上 |
| 花钱 | 免费档大概率够 | VPS 的钱 |
| 你要做的额外事 | 每隔一阵跑一条 `relay` 命令收发 | 无（但机器要一直活着） |

两种形态**对投稿者完全一样**——他们敲的命令、用的协议没有区别。以后换形态时库文件跟着搬即可（见下面的
「搬家与备份」）；只是换形态通常也换了地址，要通知使用者第一次指向新地址时加一次 `--accept-move`。

信箱形态的取舍是：公网上那个 Worker **不理解投稿内容**，它只负责收下和转发；判断内容好不好、
要不要收，全部在你本机做。所以你的 API key 一步都不会离开本机，公网上那部分被攻破也拿不到审核
判断。**发布快照**则靠签名守着：快照在你本机用签名密钥签过名才发出去，拉取的人只认这把钥匙
签的快照。所以拿到 Worker 或维护者口令的人能让拉取失败，却发不出别人会收的内容——前提是你按下面
「[签名密钥](#签名密钥)」一节设好了钥匙。

## 信箱形态：开起来

需要一个 Cloudflare 账号（免费注册）和 Node。

```bash
cd deploy/share-worker
npx wrangler login
```

浏览器会弹出来让你授权。然后生成一个**维护者口令**——它是你之后收发投稿的凭据。下面这行（PowerShell 里也能跑）
把它写进一个只有一行、UTF-8 编码的纯文本文件（下例是 `D:\share-keys\mailbox.token`，目录要先建好），同时打印出来；后面的 `--mailbox-token-file` 指的就是这个文件：

```powershell
python -c "import secrets, pathlib; t = secrets.token_urlsafe(32); pathlib.Path(r'D:\share-keys\mailbox.token').write_text(t, encoding='utf-8'); print(t)"
```

⚠ 别用 PowerShell 的 `>` 或 `Out-File` 自己存口令：Windows PowerShell 5 会存成 UTF-16 或带 BOM 的文件，FineSub 读出来对不上。
**Cloudflare 上看不到它**（写进去就读不出来了），这个文件（或你另存进密码管理器的那份）是唯一的副本。丢了也不是灾难：
重新设一个覆盖掉即可，队列与内容都不受影响。

```bash
npx wrangler secret put MAINTAINER_TOKEN   # 它会停下来等你输入：把刚才打印出的口令粘贴进去，回车
npx wrangler deploy
```

第一次部署时 Cloudflare 会让你选一个 `*.workers.dev` 子域。⚠ **这个子域基本改不掉**，想好再定。
部署完你会拿到一个地址，形如 `https://<worker 名>.<你的子域>.workers.dev`。

⚠ `wrangler deploy` 说成功之后，全球生效还要几十秒。太早去访问会看到"旧的样子"，不是出错。

## 签名密钥

快照要带你的签名，别人拉取时才分得清「维护者发布的」和「拿到口令的人塞进去的」。在你本机生成一次：

```powershell
finesub knowledge-share keygen --out D:\share-keys\signing.key
```

它写出私钥文件，并打印一行**公钥**。

- **私钥留在本机、另外备份**。它不上传到任何地方，也不放进 Cloudflare。丢了的话：生成一把新的，
  并告诉拉取的人：在 `config.toml` 里写了公钥的（见下一条）要换成新公钥；没写、靠第一次拉取记住钥匙的，
  加一次 `--accept-new-key`。
- **公钥给拉取的人**：他们在 `config.toml` 的 `[share]` 下写 `signing_key = "<公钥>"`，之后只认这把钥匙
  签的快照。不写也能用：第一次拉到带签名的快照时，客户端会记住它的钥匙，从此只认这一把——和记住
  「库的身份」是同一种「第一次见到的就当真」。
- 整台服务端形态也能签（启动服务时加 `--signing-key-file`），但钥匙就放在那台机器上：它防的是口令
  泄露，挡不住那台机器本身被人拿下。

## 信箱形态：日常怎么用

你本机保留一份服务端数据（**它才是真相**，公网上那份只是副本）。三步一循环：

```powershell
# 1. 把信箱里的投稿收进本机队列，顺便把已有的裁定和签好名的最新快照送回去
finesub knowledge-share relay --remote https://<你的 Worker 地址> `
  --server-root D:\share-data --mailbox-token-file D:\share-keys\mailbox.token `
  --signing-key-file D:\share-keys\signing.key

# 2. 审核（另开一个终端起本机服务，队列在本机，所以地址指向本机）。本机服务的口令自己编一串存进
#    文件即可（和信箱口令无关），起服务和审核用同一个文件
python -m finesub.llm.knowledge.share.server --root D:\share-data --port 8787 `
  --maintainer-token-file <本机服务的口令文件，格式同上：一行、UTF-8>
finesub knowledge-share review --remote http://127.0.0.1:8787 `
  --maintainer-token-file <同上>

# 3. 再收发一次，把裁定和新内容送出去
finesub knowledge-share relay --remote https://<你的 Worker 地址> `
  --server-root D:\share-data --mailbox-token-file <同上> --signing-key-file <同上>
```

⚠ `relay` 一定要写 `--remote`（或在 `config.toml` 的 `[share] remote` 里固定成你的 Worker 地址）：不写时它连的是
内置的共享库信箱，不是你的。固定在 `config.toml` 里时注意是哪一份（见页首）；
本机服务 `python -m finesub.llm.knowledge.share.server` 不读 `config.toml`。拿不准就每条都写 `--remote`。

`relay` 不带 `--signing-key-file` 时直接报错退出，什么都不搬；只想搬投稿、不发快照时加 `--no-publish`。

第一步可以挂成计划任务每天跑一次——它不花钱也不做判断，只是搬运。**第三步之前的审核不要自动化**：
那一步是把内容写进公开的库，值得你亲自点头。

`review` 默认只看不动（列出待审内容：每个条目叫什么、是新建还是改已有的，以及证据要求）；
`--execute` 才会真的起一次 AI 会话去核实，`--post` 才会把裁定提交。`--post` **一次只交一项**
（要带 `--queue-id`），提交前把 AI 的结论与它给的出处链接列出来、等你答 `y`——你点头就等于你替这些
链接作保，它们会计入证据门槛，所以先点开看一眼。已经看过、不想再被问时加 `--yes`。它会要求每条主张
有外部出处，达不到就拒绝——你确认没问题的话，可以用 `--override-thresholds "理由"` 放行（同样只对
这一项），理由会记在服务端。

⚠ AI 审核读的正是投稿人写的内容，投稿里夹带的话可能影响它的结论——这是提交前必须由你过目的原因。

⚠ **投稿是不可信输入**。别人写进条目里的东西会经过清洗，但清洗防的是格式与注入，不防内容错误或
不该公开的信息——那是审核那一步的事，也是这套流程为什么不肯替你自动点头。

## 整台服务端的搭法

步骤在开发者向的 [`deploy/share-server/README.md`](../../deploy/share-server/README.md)：装包、建数据目录、
systemd 拉起、Caddy 反代加 TLS、litestream 持续备份。日常审核用的是同一条 `review` 命令，
只是 `--remote` 指向那台机器。

## 搬家与备份

**真相只有一个文件**：`--server-root` 目录下的 `share-server.sqlite`（信箱形态在你本机，
整台形态在那台机器上）。里面有全部内容、待审队列、贡献者凭据和完整性链（每次发布都记一笔、前后串起来的校验记录，客户端靠它发现库被改写过）。

- **备份** = 拷走这个文件（服务停下时拷，或用 `sqlite3` 的 `.backup`）。
- **搬家** = 把它放到新位置，照常起服务。⚠ **必须搬这个文件本身**——拿公开快照重建一个新库会得到
  一个新身份，所有人的客户端都会报「这是另一个库」并拒绝拉取，重新发布也改不回来，只能让每个人加
  `--accept-move` 认下它——那正是防冒名的那道检查，别拿它来掩盖重建。
- **换地址**（换域名、从 Cloudflare 搬到 VPS）不影响任何人的凭据与合并记录，只是他们第一次指向
  新地址时要加一次 `--accept-move`。所以**换地址时要主动通知使用者**，别指望他们照旧敲命令。

## 别人怎么接你的库

给他们地址就行。⚠ 地址必须是 `https://`（口令在请求头里，走明文 http 谁都看得见）；只有本机
测试可以用 `http://127.0.0.1`。服务端也不能靠重定向把人引到新地址——客户端遇到重定向会报错而不是
跟过去，换地址按上面的「[搬家与备份](#搬家与备份)」来。

```powershell
finesub knowledge-share register --remote <你的地址>
finesub knowledge-share pull --remote <你的地址>
```

⚠ 不写 `--remote` 时命令会连到**这个项目内置的那个库**，不是你的。要让别人默认连你的，得让他们在
`config.toml` 里写一行：

```toml
[share]
remote = "https://你的地址"
signing_key = "<你的公钥>"   # 可选：写了就从第一次拉取起只认你签的快照
```

## 会碰到的限制

匿名注册每天 20 个（信箱形态另限同一 IP 每天 3 个）、总共 500 个，注册满 7 天仍没有一份待审或已批准投稿的口令会被收回；每人最多 5 份待审、全局最多 200 份；30 天没人处理的投稿自动过期。
这些是防滥用的，够正常使用用很久。真被刷了，在 Cloudflare 那边加一条限速规则比改这些数字有效。

更细的（端点契约、配额常量、部署时的坑）在开发者向的
[`deploy/share-worker/README.md`](../../deploy/share-worker/README.md)。
