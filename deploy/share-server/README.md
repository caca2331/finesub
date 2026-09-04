# 共享知识库服务端部署（plan §6 服务端形态）

服务本体是 stdlib-only 的单进程 HTTP（`python -m finesub.llm.knowledge.share.server`），
**只绑定 127.0.0.1**；TLS、域名与对外暴露一律交给前面的 Caddy（终态）或 Cloudflare
Tunnel（过渡），服务自身永远不直接见公网。服务器上没有任何 LLM API key——审核在
维护者本机跑（`share review`），服务器只存队列与语料。

## 终态：轻量 Linux VPS

1. 装 Python ≥ 3.10 与本包（只需要 `[harness]` 之下更少的东西——实际上
   `finesub.llm.knowledge.share.server` 是 stdlib-only，`pip install finesub` 即可，或
   直接把仓库 `src/` 放上去用 `PYTHONPATH` 跑）。
2. 建数据目录（仓库外）：`/var/lib/finesub-share`。
3. 首次手动起一次拿 maintainer token（只打印一次，存进密码管理器）：

   ```bash
   python -m finesub.llm.knowledge.share.server --root /var/lib/finesub-share --port 8787
   ```

4. 装 systemd 单元（`finesub-share.service`，把 token 写进 override 的 Environment 或
   `--maintainer-token` 参数）：

   ```bash
   sudo cp finesub-share.service /etc/systemd/system/
   sudo systemctl enable --now finesub-share
   ```

5. Caddy 反代 + 自动 TLS（`Caddyfile`，改成你的域名）：

   ```bash
   sudo cp Caddyfile /etc/caddy/Caddyfile && sudo systemctl reload caddy
   ```

6. litestream 把 SQLite 连续复制到对象存储（`litestream.yml`，填桶与密钥）：

   ```bash
   sudo cp litestream.yml /etc/litestream.yml
   sudo systemctl enable --now litestream
   ```

## 过渡：本机 + Cloudflare Tunnel（与终态同构）

服务照常本机起（同一条命令、同一个 `--root`）；对外用：

```bash
cloudflared tunnel --url http://127.0.0.1:8787
```

免费、无需公网 IP；迁移到 VPS = 拷 `--root` 目录 + 同一份 systemd 单元。
（或 Oracle/GCP 免费档 VM 上直接走终态步骤。）

## 运维要点

- **滥用防护分两层**：应用层内建硬配额（每贡献者 pending ≤5、全局 pending ≤200、注册
  数 ≤500、pending 30 天自动过期——常量在 `share/server.py` 顶部，改了重启生效）；代理层
  IP 限速见 Caddyfile 注释（需 caddy-ratelimit 插件）。公开部署前两层都要在。

- **备份/恢复**：真相只有 `--root` 下的 `share-server.sqlite`（含队列、贡献者、哈希链）。
  litestream 之外，凡本地手动备份直接拷文件即可（服务停下或用 sqlite3 `.backup`）。
  恢复 = 放回文件重启。哈希链带防回滚：**不要**用旧备份覆盖新库后继续服务——客户端
  会因 server_rev 倒退拒绝拉取；确需回滚要同时通知所有客户端清锚点（meta 里
  `share:<remote>:server_rev`/`chain_hash` 两键）。
- **token**：contributor token 匿名自助（`POST /register`）；maintainer token 泄露 =
  队列可被清空，换 token 只需改 systemd 单元重启（已有队列数据不受影响）。
- **GitHub snapshot 镜像**（可选，未内建）：定期把 `GET /snapshot` 的 JSON 推到公开
  仓库当匿名拉取源与异地备份；客户端验的是哈希链，不信任镜像本身。
- 升级：换包重启即可；store schema 迁移在进程启动时就地执行。
