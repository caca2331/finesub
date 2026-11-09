# 共享知识库的信箱部署（Cloudflare Worker + Durable Object）

`deploy/share-server/` 那套是**整台服务端**（VPS 或本机 + Tunnel）。这一套是另一种形态：
公网上只放一个**信箱 + 快照 CDN**，审核、apply、发链仍然在维护者本机。两种形态怎么选见
[`docs/manual/share-server.md`](../../docs/manual/share-server.md) 的「两种搭法，先选一种」；
更细的取舍依据在维护者的本地计划里（不随公开快照发布）。

```text
贡献者 ──push──▶ Worker（信箱）        维护者本机
        ◀─status─┘   ▲   │              ShareService（真相库）
                     │   └──/inbox────▶ share relay ──▶ 本机队列 ──▶ share review
                     └──/publish, /outbox──────────────┘
```

## 这个 Worker 不做什么

**它不理解 bundle 语义**：不清洗、不校验 anchors、不判按槽位的审核证据门槛、不 parse 快照
（快照按原字节存取，metadata 走 query string）。它只读 bundle 里的一个字段——
`idempotency_key`。判定全部在本机 `ShareService.push()` 上发生，那正是它当初被写出来防的事。

客户端本来就把服务端当不可信：拉取侧逐项验哈希链，再走 `snapshot_content_problems` 的结构准入，
把服务端做薄只是缩小攻击面。但**哈希链谁都能续**，所以能写 `/publish` 的人（拿到
`MAINTAINER_TOKEN`，或拿下 Cloudflare 账号）原本就能发布任意快照。守住发布权的是**签名**：relay
在维护者本机用 `share keygen` 生成的私钥签快照头（`share/signing.py`），客户端只认预置或第一次见到的
那把公钥（`sync.check_signature`）。口令泄露因此只够让拉取失败，发不出会被接受的内容；私钥不上
Cloudflare。

⚠ 一处与自托管服务端的**行为差异**：信箱不校验「同一个幂等键配了不同内容」——它没有语义
摘要可比。它的做法是**第一份 body 永远不被覆盖**（同 key 同贡献者直接回旧的 `queue_id`），
所以偷换已入队 bundle 这条路是堵死的；真正换了内容的客户端本来就会换新 key
（`cli._cmd_push` 按内容摘要找旧 key）。

## 部署

需要：Cloudflare 账号（免费档够——这个 Worker 只搬字节）、`npx wrangler`、一个 maintainer token。

```bash
cd deploy/share-worker
npx wrangler secret put MAINTAINER_TOKEN     # 自己生成一串随机值，存进密码管理器
npx wrangler deploy
```

**只有一把维护者口令，是有意的**：收件、回写、发布都用它。发布权不靠口令守——客户端只认维护者签名的
快照——所以口令泄露最多让拉取失败（发一份没签名或签错的快照），发不出会被接受的内容；再拆一把
发布口令只多一个要保管的秘密，防不住更多东西（2026-09-25 评估后不拆）。

**部署新版本前后**：

- ⚠ 贡献者口令在 Durable Object 里只存哈希，旧部署里的明文在新版本首次启动时就地换成哈希
  （`hashStoredTokens`）。**这一步回不去**：换回更早的 Worker 版本，已注册的口令会全部失效。
- 部署之后跑一次带 `--signing-key-file` 的 `relay`，让线上快照换成签过名的那份——内置公钥的客户端
  不收未签名的快照。
- `wrangler.jsonc` 开着 observability：Workers Logs 每条请求存来访的原始 IP 与大致位置（免费档留 3 天；
  `X-Share-Token`、`X-Maintainer-Token` 的值 Cloudflare 自己打码，2026-11-08 实测）。对公众开放之后在 `observability` 里加
  `"logs": {"invocation_logs": false}` 关掉逐条请求日志，代码自己打的日志照留；小事，晚关也无妨（owner 2026-11-08）。

部署前先跑一遍[单元测试](#测试)，再对着本地 dev Worker 跑一遍下面的 `e2e_check.py`（长路径的坑见下一段：
把 `src/` 与 `wrangler.jsonc` 拷到 `C:\cfsw` 这类短路径再 `wrangler dev`）。两样都要：单元测试跑在替身上，
真 workerd 只有 e2e 碰得到。

本地跑（不需要账号）：把 `MAINTAINER_TOKEN=...` 写进 `.dev.vars`，然后 `npx wrangler dev`。
⚠ **目录路径不能太深**：本地 miniflare 的 SQLite 在长路径下起不来，症状是毫无细节的
`internal error`（见 `tools/cf_share_poc/README.md` 的坑记）。

跑完一整条链路（注册 → 投稿 → 中继 → 本机审核 → 裁定回传 → 拉取验链 → 两条越权检查）：

```bash
python deploy/share-worker/e2e_check.py --remote http://127.0.0.1:8810 --mailbox-token <同一个值>
```

⚠ 它**只对着 dev Worker 跑**：脚本用一个临时的空库，对着线上会把空快照发布上去。默认的
`python -m pytest` 里没有它：那里的 Python 测试把传输打了桩，Worker 的单元测试跑在替身上，
Python 客户端与真 workerd 上的 Worker 只在这一处对接过。

⚠ **边缘会按 User-Agent 拦人**。首次线上部署时，每一个请求（连不存在的路由）都被 Cloudflare 挡成
`403 / error code: 1010`，因为 urllib 的默认 UA 是 `Python-urllib/3.x`——回复里没有任何线索指向真因。
客户端因此显式声明 `User-Agent`（`share/client.py` 的 `USER_AGENT`，有测试钉着）。**别把它改回默认**，
也别改成浏览器串：任何诚实的标识都能过。

⚠ **token 用 `--…-token-file`，或者写成 `--flag=value`**。`secrets.token_urlsafe` 生成的串可能以
`-` 开头，那时 `--maintainer-token <值>` 会被 argparse 当成"又一个选项"，报一句与真因无关的
`expected one argument`（首次实跑就撞上了）。三个入口都有文件版：`relay --mailbox-token-file`、
`review --maintainer-token-file`、`e2e_check.py --mailbox-token-file`。

## 测试

Worker 的单元测试在 `test/`，只用 Node 自带的东西（`node:test`、`node:sqlite`），**不用 `npm install`**：

```bash
cd deploy/share-worker
node --test "test/*.test.mjs"    # Node ≥ 22.13；22.5–22.12（及 23.0–23.3）要加 --experimental-sqlite
```

默认的 `python -m pytest` 经 `test/test_llm_knowledge_share_worker.py` 跑同一组，JS 测试红了它就红；
没有 `node`、或 Node 旧到没有 `node:sqlite`（早于 22.5）时那一条是 skip。

- `regressions.test.mjs` 钉着 2026-09-09 复审在这个文件里找出的六处缺陷：push 上限按 UTF-16 码元量、
  快照切片劈开代理对、已中继的项被信箱过期、一条循环用光注册额度、`last_used` 让垃圾投稿保住 token、
  条件请求的 ETag 分支没人用。每条都验过：把缺陷放回去，对应的测试会红。ETag 那条的客户端一半
  （`fetch_snapshot` 发的 `If-None-Match` 恰好是 Worker 比较的那个串）在上面那个 Python 文件里。
- `routes.test.mjs` 是各路由的契约：注册 → 投稿 → 中继 → 裁定 → 查状态、快照跨切片原样往返、
  越权一律拒、旧部署的明文 token 在第一个请求之前换成哈希。

**它们怎么跑**：`src/index.js` 原样载入，`cloudflare:workers` 由模块解析钩子（`test/hooks.mjs`）换成
`test/workerd.mjs`——一个只按本 Worker 用到的那部分 Durable Object 接口写的替身。`ctx.storage.sql`
落在内存里的 `node:sqlite` 上，`env.INBOX` 按名字给出同一个对象，RPC 的参数与返回值走结构化克隆，
每次调用都记下来（测试据此数往返）。请求从默认导出的 `fetch` 进、`Response` 出，与边缘一样。

**替身照着做的与没做的**（逐条写在 `test/workerd.mjs` 开头，那里是准的）：

- 照着做：`exec` 的游标（`toArray`、`one`、迭代）、绑定类型、孤立代理存进 SQLite 后读回的字节变了、
  单个值 2 MB 上限、`transactionSync` 出错回滚、`blockConcurrencyWhile` 挡住它之后的调用、
  自定义错误类过不了 RPC、`crypto.subtle.timingSafeEqual`。
- 没做：输入 / 输出门与写合并、游标的惰性、alarm 与休眠、单值大小以外的各项上限（CPU 时间、请求大小、
  语句长度）、边缘本身（真实的 `CF-Connecting-IP`、缓存、按 User-Agent 拦截）。

所以单元测试绿**代替不了**部署前对着 `wrangler dev` 跑 `e2e_check.py`：那是唯一跑真 workerd 的地方。
替身写错了会悄悄藏住 bug，所以改它时同步改那段开头注释，并照着 workerd 的文档核对。

## 维护者的日常

本机保留一份 `share-server.sqlite`（就是自托管那套的 `--root`），三步循环：

```powershell
# 1. 把信箱里的投稿收进本机队列，并把已有裁定与最新快照发回去
python -m finesub.llm.knowledge.share relay `
  --remote https://finesub-share.<你的子域>.workers.dev `
  --server-root D:\share-data --mailbox-token-file <存着 MAINTAINER_TOKEN 的文件> `
  --signing-key-file <share keygen 写出的私钥>

# 2. 照常审核（队列在本机，所以 --remote 指本机；API key 从不离开本机）
python -m finesub.llm.knowledge.share.server --root D:\share-data --port 8787 --maintainer-token-file <本机服务的口令文件>   # 另一个终端
python -m finesub.llm.knowledge.share review --remote http://127.0.0.1:8787 --maintainer-token-file …

# 3. 再 relay 一次，把裁定和新快照送出去
python -m finesub.llm.knowledge.share relay --remote … --server-root D:\share-data `
  --mailbox-token-file … --signing-key-file …
```

`relay` **可重入**：崩在「本机入队了、还没告诉信箱」这一步，下次重跑会重推同一条，本机的
幂等键把它折回同一个队列项而不是排两次。`--no-publish` 只收不发。

它顺带清理队列存档：**裁定超过 90 天**（`--purge-days N`，0 关闭）的项里，被拒/过期的整行删掉，
已批准的**保留行、只清空 bundle 正文**（内容已经在语料里）。⚠ 已批准的行不能删——digest 的
跨贡献者计票正是从这些行数的（`DISTINCT contributor WHERE status='approved'`），删了会把已被
佐证的 claim 悄悄变回一票。计龄从**裁定时刻**算，不是收件时刻：等了三个月才审的投稿不该一裁定
就被扫走。⚠ 同理，**裁定时刻未知（旧库升级上来的行）不等于"很久以前"**：升级会把它们记成升级
那一刻，保留期从那时起算——早先把空值当成早于任何截止时间，第一次清理就会删掉几分钟前刚做的裁定。

## 端点

| 谁 | 端点 | 说明 |
| --- | --- | --- |
| 贡献者 | `GET /identity` | corpus id + 已发布快照的 rev / 链头。还没发布过快照时 **503** |
| 贡献者 | `GET /snapshot` | 维护者发布的那份，原样返回。带条件请求契约：**ETag = 链头**，`If-None-Match` 命中回 **304**（客户端把防回滚锚点当标签发，所以「已是最新」的 pull 两边都不花钱）。⚠ `Cache-Control` 是 **`no-cache`** 而不是 `max-age`——客户端对 `server_rev` 倒退是硬失败，允许任何中间层发旧副本等于把正常操作变成故障 |
| 贡献者 | `POST /register` | 匿名 contributor token |
| 贡献者 | `POST /push`、`GET /push/<id>` | 投稿与查状态（状态只有推送者本人或维护者可读） |
| 维护者 | `GET /inbox`、`GET /tracked` | 待中继的投稿 / 还没有终局裁定的项。**两个都按 `after` 游标分页**——它们都是最早的排在前面，不带游标读会一直重读队头，积压时后面的项永远轮不到 |
| 维护者 | `POST /outbox` | 回写状态、裁定与 handle→canonical 映射 |
| 维护者 | `POST /publish` | 上传快照（body 是快照字节，identity 走 query string） |

客户端看到的 `queue_id` 是**信箱的** id；本机队列 id 只在 `/outbox` 的记录里出现。

## 配额

`MAX_CONTRIBUTORS` 500、每贡献者 pending ≤5、全局 pending ≤200、30 天过期——常量在
`src/index.js` 顶部，与 `share/server.py` 同源。⚠ **改一边要改另一边**：本机 relay 会把同一批
内容再过一次本机配额，两边不一致时以更严的那边为准，症状是信箱收下了、本机却拒绝
（贡献者会看到一条 rejected + 原因，不会静默丢失）。

⚠ **总额 500 防的是表膨胀，不是刷取**：地址随包发布，人人有，一条 `for` 循环就能占满，之后新人
永远只看到 429。所以另有三道——每天新注册 ≤20（`MAX_REGISTRATIONS_PER_DAY`）；同一 IP 每天 ≤3
（`MAX_REGISTRATIONS_PER_IP_PER_DAY`，按 `CF-Connecting-IP` 的哈希计，一天后删），挡的是一个人每天
先把那 20 个用完；以及回收「发出去 7 天、且**没有待审或已批准投稿**」的 token（`UNUSED_TOKEN_DAYS`）。
⚠ 本地 `wrangler dev` 没有 `CF-Connecting-IP`，所有请求算同一个 IP，一天只能注册 3 次——e2e 反复跑
撞上 429 时清掉 `.wrangler/state` 重来。

⚠ 回收的判据看**投稿的结局**，别换成「用过没有」那类标记：早先的版本在 `requireContributor` 里盖
`last_used`，而那一步跑在 parse bundle **之前**——一条格式错误的投稿就能让 token 变永久（复审
2026-09-09）；后来改成「信箱里有没有它的行」，一条格式正确的垃圾投稿又能做到同一件事。被拒和过期
的投稿因此都不占座。

**信箱不无限长**：裁定回传（`/outbox`）时，已批准和被拒的行清掉 bundle 正文（本机有副本，信箱这边
再没人读）；被拒的行在裁定送回 90 天后、过期的行在收件 90 天后整行删（`PURGE_SETTLED_DAYS`）。已批准的行留着：贡献者的
`share status` 要从它取 handle→canonical 映射。

**贡献者 token 只存哈希**：`route()` 先算 SHA-256 再交给 Durable Object，对象里、`/inbox` 发给 relay 的、
本机队列里存的都是哈希。旧部署里的明文 token 在对象首次启动时就地换成哈希（`hashStoredTokens`）。
维护者口令用常数时间比较（`sameSecret`）。

⚠ **过期只碰没被中继走的项**（`relayed_at=''`）。两边计龄的起点不同——信箱按收件、本机按中继——
所以一旦交给了维护者，命运就归本机；否则信箱这边把它标 expired 之后它就退出 `tracked()`，那段
时间里做出的裁定（包括内容已经进语料的批准）永远送不回去，贡献者看到的是「过期」，
handle→canonical 映射丢失，他下次 pull 会把自己的条目当新的再建一遍。

## 备份

真相在**本机**的 `share-server.sqlite`，照 `deploy/share-server/README.md` 的备份段做即可。
信箱里只有待处理队列与一份快照副本——丢了不影响真相，重跑一次 `relay` 就回来了
（除了那些还没被中继走的投稿，所以别在积压时删 Worker）。
