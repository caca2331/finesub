# 游戏专名列表的构建与发布

维护者用：从上游词表做出一份游戏专名列表文件，审读、发布成公开仓库 Release 的资产，再写进随包发的清单。
名单要满足什么、为什么那样挑，见 [`docs/knowledge.md`「一份名单要满足什么」](../../docs/knowledge.md#一份名单要满足什么)；
用户那一侧怎么取回、怎么进窗，见同一份文档的「游戏专名列表」。

## 步骤

1. **备齐四份输入**，格式写在 [`build.py`](build.py) 开头：上游抽成的词表（每行一个名字）、每个名字的日期、
   「有名」的闭卷探针结果、一批与游戏无关的日语录音的列表（录音在维护者本机，怎么清见[要满足什么](../../docs/knowledge.md#一份名单要满足什么)的「背景误碰率」）。产出它们的脚本还是维护者在 `explore/` 下的实验脚本，
   清单与怎么重跑（其中几个要在归档 tag 上跑）见那个目录的 README；收进这个目录是
   [`docs/llm_followups.md`「游戏专名列表的余项」](../../docs/llm_followups.md)里的一条。记下抽取时上游的 commit。
   「有名」的探针按下面做，不依赖那些脚本：
   - **池子**：常驻名单的候选池，即核心名字里近两年加入的与读音短的。拿一个只有 `{}` 的 `--famous` 文件跑第 2 步，构建会停下并
     逐行列出池里的 `日文=中文`（原神 2026-11 这份是 106 个）；
   - **出题**：把池里的日文名乱序编号，让一个不联网、不读文件的模型凭记忆写出每个名字的简体中文官方译名，不确定写 `?`、
     不许自己音译；换三种乱序各问一遍（2026-11 用的是 3.5 Flash Lite：知识截止较早，判出的有名对更新的模型也成立）；
   - **记分**：每个名字答对官方 `zh`（去掉空格、间隔号与「」『』后逐字相同）的次数，写成 `{中文名: 0–3}` 的 JSON 交给 `--famous`；
     三次全对算有名（`FAMOUS_PASSES`）；
   - **补测**：审读改了日期或修正之后池子变大时，构建照样停下列出缺的那几个，只给它们出题，分数并进同一个文件。
2. **构建**（CPU，几分钟；要 `kb-phonetic` extra，从 checkout 根运行）：

   ```bash
   hw status && PYTHONPATH=src python tools/game_glossary/build.py <游戏键> --terms <词表> --dates <日期> \
       --famous <探针结果> --background <录音列表> --source-commit <上游 commit> --out <输出文件> --built <构建日期>
   ```

   它打印条目数、常驻名单与核心名字的个数、误碰对数、文件的字节数与 sha256。`--built` 是「近两年」的起算日，也是文件名与
   Release tag 里的日期。
3. **审读**：把全表导成几份制表符分隔的文件（行号、`ja`、`zh`、`en`、标签、是否常驻、简注），常驻名单连同它在索引里的
   那一行单独一份，派给另一个模型读（2026-11 用的是 DeepSeek，一份全表约 7000 条分 5 份）。任务里写清名单的用途、要找的
   问题（[要满足什么](../../docs/knowledge.md#一份名单要满足什么)里的每一条），只报有把握的、逐条给行号与理由。
   按那一节的判法核实之后，修正写进 `GAMES` 里这款游戏的 `fixes`（改 `ja` / `zh` / 标签）、`drop`（整条去掉）、
   `released`（实装日期）、`ordinary`（日常词，移出常驻名单），重新构建。
4. **冒烟**（换了挑法或第一次做某款游戏时）：把文件装进一个临时知识库，

   ```bash
   PYTHONPATH=src python -c "import sys, pathlib; from finesub.llm import glossary_store as gs; gs.install(sys.argv[1], pathlib.Path(sys.argv[2]).read_bytes())" <临时库> <输出文件>
   ```

   再拿一段讲这款游戏的录音跑一次纠错（`--knowledge-root <临时库> --extra-info <游戏名>`，模型用 Luna 即可），看 prompt 里那一块与运行的告警。
5. **挂 Release**（挂之前先问 owner）：tag 是 `glossary-<游戏键>-<构建日期>`，资产就是构建出的那个文件，说明里写上游、上游 commit 与许可：

   ```bash
   gh release create glossary-<游戏键>-<构建日期> <输出文件> --repo caca2331/finesub --target <最近一个发版 tag 的提交> \
       --latest=false --title "..." --notes "..."
   ```

   `--latest=false` 让产品版本留在 Latest 上。`--target` 让新 tag 落在已有的发版提交上：不给时 tag 打在公开 `main` 的最新提交上，
   那可能是下次发布本该替换掉的 checkpoint，被 tag 钉住就替换不掉了。同一份文件不重传：要改就换构建日期、出新的 tag。
6. **写清单**：在 `src/finesub/llm/glossary-manifest.json` 的 `glossaries` 里加一行（同一款游戏的旧行换掉）：
   `game` 与 `aliases`（与 `GAMES` 里的 `name`、`aliases` 相同：取回认清单里的，选中认名单文件里的）、`url`（Release 资产的下载地址）、
   `size`、`sha256`（第 2 步打印的）。
7. **验证取回**：在临时知识库上真下载一次，看装上的哈希与清单相同：

   ```bash
   PYTHONPATH=src python -m finesub.llm.knowledge --root <临时库> glossary fetch <游戏名>
   ```

   然后提交清单。用户要等下一个带这份清单的版本才取得到。

## 注意

- **重建**：常驻名单的「近两年」从 `--built` 算，离构建超过半年任务就告警过时，所以大约半年重建一次，重走全部步骤，
  审读也照做：上游每次更新都可能带进新的笔误。旧的 Release 留着，已经装上的用户不依赖它。
- **新游戏**：在 `GAMES` 里加一项，核心标签、称呼标签、修正表都按那个上游的标签体系重新定；常驻名单的规则先与 owner 定
  （原神的是 owner 2026-11-12 定的）。
- **撤回**：上游要求停止再分发时，删掉 Release 资产与清单那一行，下一个版本起取不到；已装的副本留在用户本地。
