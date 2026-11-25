# 游戏专名列表的构建与发布

维护者用：从上游词表做出一份游戏专名列表文件，审读、发布成公开仓库 Release 的资产，再写进随包发的清单。
名单要满足什么、为什么那样挑，见 [`docs/knowledge.md`「一份名单要满足什么」](../../docs/knowledge.md#一份名单要满足什么)；
用户那一侧怎么取回、怎么进窗，见同一份文档的「游戏专名列表」。

维护者留下的每一版构建在 `builds/<游戏键>/`：`glossary.json`（构建出的名单，一个名字一行）、`famous.json`（有名探针的得分）、
`build.txt`（这一版的每份输入、它读的上游 commit 与读音库版本），以及重取不回来的小输入（绝区零的 `dates-enka.json`）。
它在私有历史里跟踪、公开快照里没有：名单里的标签与标记是我们的判断，改坏了要能按提交往回找；从上游按 commit
重取得到的输入（词表、日期、版本、提及数）不跟踪，靠 `build.txt` 记下的 commit 重做。

## 步骤

1. **备齐输入**，格式写在 [`build.py`](build.py) 开头，都用这个目录的脚本做（`python -I` 跑，不读 checkout 的包，`build.py` 除外）。
   记下每份输入读的上游 commit（词表、日期与版本用的仓库、数提及的文本表），第 2 步写进 `build.txt`。
   - **词表**：[`extract.py`](extract.py) 把上游抽成每行一个名字（只取名字，不取描述与剧情文本，上游的代码只当数据读）。
     原神是 genshin-langdata；星穹铁道是 StarRailRes 加 turnbasedgamedata 的文本表（阵营、星神、世界、敌人阵营 StarRailRes 不收）；
     绝区零是 ZenlessData 的文本表。用法与每个上游取哪些表写在它开头。
   - **日期**：[`dates.py`](dates.py)，每款游戏的来源不同（原神用上游的 git 历史；星铁用 StarRailRes 的历史加文本表各版本的快照；
     绝区零用快照加 Enka 的角色表），用法写在它开头。几份日期都交给 `--dates`，取最晚的。实装前就进了数据的角色，
     把实装日期写进 `GAMES` 的 `released`。
   - **有名**：下面的闭卷探针，用 [`famous.py`](famous.py)。
   - **剧情信号**：[`signals.py`](signals.py) 数每个名字在游戏文本表里被提到的次数（`--signals`，键是词表里的 `id|ja`，所以要和同一份词表配套：配错了提及数会悄悄对不上）；台词数由 `extract.py` 从说话人表写进词表
     （星铁的 TalkSentenceConfig、绝区零对话键的 `_Name`）。文本表只在本地用。两者定重要 NPC，规则见
     [要满足什么](../../docs/knowledge.md#一份名单要满足什么)的「标签怎么定一个名字的去处」。
   - **版本**（可选，`--versions`）：数据仓库每个版本的第一个提交 `{版本: [提交, 日期]}`，给 `version:` 标签。
   - **背景录音**：一批与游戏无关的日语录音的列表（录音在维护者本机，怎么清见[要满足什么](../../docs/knowledge.md#一份名单要满足什么)的「背景误碰率」；
     清理与量法的脚本还在 `explore/game_glossary/form_bg_*.py`）。不在录音旁边时：重建沿用上一版的误碰率（`--noise-from`），
     第一次做的游戏只能不带（`--no-background`），有了录音再重建。

   「有名」的探针：
   - **池子**：常驻名单的候选池（核心名字里近两年加入的与读音短的、没被标记移出的），加上重要、却因短名字与普通词规则缺证据
     落成不常用的名字（若、ピーク、石門：探针三次全对就成了证据；名字的别称缺证据时不替它探本名，探的是写出来的那个名字）。拿一个只有 `{}` 的 `--famous` 文件跑第 2 步，
     构建会停下并逐行列出这些 `日文=中文`（原神 2026-11 的候选池是 106 个，星铁 68 个，绝区零 78 个；缺证据的另有十几个），存成一个文件；
   - **出题**：`famous.py <游戏键> --pool <那个文件> --famous <探针结果> --cache <缓存目录>`。它把池里的日文名乱序编号，
     让一个不联网、不读文件的模型凭记忆写出每个名字的简体中文官方译名，不确定写 `?`、不许自己音译；换三种乱序各问一遍。
     模型是 3.5 Flash Lite（`gemini-3.5-flash-lite`，免费额度，钥匙在 `GEMINI_API_KEY`）：知识截止较早，判出的有名对更新的模型也成立。
     一批最多 120 个名字，一款游戏三遍只要几次调用；每次的问答按题缓存，`<缓存目录>/calls.json` 记每天的调用数，
     到 `--max-calls` 就停（免费额度一天几百次）；
   - **记分**：每个名字答对官方 `zh`（去掉空格、间隔号与「」『』“”后逐字相同）的次数，写成 `{中文名: 0–3}` 并进 `--famous` 那个文件；
     三次全对算有名（`FAMOUS_PASSES`）；
   - **补测**：审读改了日期或修正之后池子变大时，构建照样停下列出缺的那几个，只给它们出题，分数并进同一个文件。
2. **构建**（CPU，几分钟；要 `kb-phonetic` extra，从 checkout 根运行）：

   ```bash
   hw status && PYTHONPATH=src python tools/game_glossary/build.py <游戏键> --terms <词表> --dates <日期> [--dates <日期>…] \
       --signals <提及数> [--versions <版本>] --famous tools/game_glossary/builds/<游戏键>/famous.json --background <录音列表> \
       --readings-from tools/game_glossary/builds/<游戏键>/glossary.json --source-commit "<词表的上游 commit>" \
       --out tools/game_glossary/builds/<游戏键>/glossary.json --built <构建日期>
   # 或者 --noise-from <上一版的名单文件>、--no-background
   ```

   每款游戏上一次用的输入与参数在它的 `build.txt`。`--out` 直接覆盖 `builds/` 里的那份（构建先读完 `--readings-from` 再写）。
   要留的这一版连同 `famous.json` 与改过的 `build.txt`（输入、上游 commit、本机读音库版本）提交，提交说明写改了什么；
   试着玩的构建不提交，`git restore` 回去即可。

   它打印条目数、标记的个数、常驻名单与池子的大小、误碰对数、文件的字节数与 sha256；`GAMES` 里手写的覆盖
   （`fixes`、`drop`、`ordinary`、`tiers`、`released`、`extra_variants`、`variant_keep`）有键这次一个名字都没对上时，逐张表列出
   `unused override:`：多半是上游改了写法，上游自己修掉笔误的就删掉那条，否则按新写法改键。读音（`kana`、`short-reading`）与普通词的判断
   取决于本机 pyopenjtalk-plus 与 Sudachi 的词典，同样的输入换一台机器会出不同的文件（2026-11-20：Windows 上 sudachipy 0.6.10、
   pyopenjtalk-plus 0.4.1.post7 把 六分街 读成ロクブガイ、若陀竜王 读成ワカオー，读不全 飛霄，另一台机器读的是ロップンガイ、ワカダリューオー）；
   文件的 `provenance.readers` 记着构建时的版本。所以重建时加 `--readings-from <上一版>`：不全是假名的名字沿用上一版的 `kana`
   （与本机读的不同时打 `reading:carried` 标签），本机只读上一版没有的新名字与上一版也没读出的名字，这些读音在审读时看。
   每次重建后先和上一版逐条比（换了机器或升了这几个库尤其要比）；上一版就是最近一次提交的那份，
   往回查哪一次改坏了标签也是这样拿两个提交比：

   ```bash
   git show HEAD:tools/game_glossary/builds/<游戏键>/glossary.json > tmp/<游戏键>-prev.json
   python -I tools/game_glossary/compare.py tmp/<游戏键>-prev.json tools/game_glossary/builds/<游戏键>/glossary.json
   ```

   它按 `(ja, zh)` 列出换了层级、进出常驻池或折叠的名字与标签的增减。2026-11-20 之前构建的文件没有 `provenance.readers`；星铁、绝区零与原神的格式 2 名单以维护者本机那次重建为基线，
   `builds/` 里的第一版就是 2026-11-13 发布的那三份（还是整份一行，之后的构建一个名字一行，第一次 diff 读不出东西，用 `compare.py`）。
   `--built` 是「近两年」的起算日，也是 Release 资产的文件名与 tag 里的日期。
3. **审读**：把全表导成几份制表符分隔的文件（行号、`ja`、`zh`、`en`、标签、是否常驻、简注），常驻名单连同它在索引里的
   那一行单独一份，派给另一个模型读（2026-11 用的是 DeepSeek，一份全表约 7000 条分 5 份）。任务里写清名单的用途、要找的
   问题（[要满足什么](../../docs/knowledge.md#一份名单要满足什么)里的每一条），只报有把握的、逐条给行号与理由。
   审读时看每条的 `tags`：它为什么常用 / 不常用 / 只认原样（`exact`）/ 进常驻池都写在那里。常用与不常用判错的，改规则（`Game.kinds`、`story_lines`、
   `story_mentions`）或在 `tiers` 里逐条覆盖并写上理由（成为 `override:` 标签；`sound` 把一个只认原样的名字放回读音匹配）。
   短名字与普通词同形的名字按风险与证据标签分（`risk_tags`，规则见
   [要满足什么](../../docs/knowledge.md#一份名单要满足什么)的「短名字与普通词同形的名字」，开关 `Game.risk_rule`，门槛 `NOISE_RISK`，
   折叠 `FOLD_MIN`）。只改了覆盖或这条规则时不必从头构建：

   ```bash
   PYTHONPATH=src python tools/game_glossary/retag.py <游戏键> <名单文件> --out <新文件>
   ```

   它按文件里的事实标签重写 `risk:` / `evidence:` / `override:` 与标记，打印各层挪动的条数；新进常驻池的名字要补探针、走完整构建。
   `fold` 例外，它要本机的词典重判平假名是不是普通词：本机读音库与上次判它的那套（`provenance.fold_readers`，retag 写进去；没有时是构建的 `readers`）不同时告警，
   `fold+` / `fold-` 是由此变了的条数。
   改了 `build.py` 的规则先跑这里的测试：`python -m pytest tools/game_glossary -n 0`（不进默认套件）。改短名字与普通词同形那条规则要另一个 agent 同意才生效（2026-11-20 定它时就是这样做的，见 [`llm_followups.md`](../../docs/llm_followups.md)「游戏专名列表的余项」）。
   retag 只管 `tiers`、`risk_rule` 与 `risk_tags` 用到的常量：`kinds`、`story_lines`、`story_mentions`、`ordinary`、`drop`、`fixes`
   改的是构建时算好的事实标签，要重新构建（retag 会把新的 `story_lines` 只用到 `evidence:lines` 上，文件就前后不一致）。
   按那一节的判法核实之后，修正写进 `GAMES` 里这款游戏的 `fixes`（改 `ja` / `zh` / `en` / 标签）、`drop`（标 `skip`，写上理由）、
   `released`（实装日期）、`ordinary`（日常词，标 `core_skip`、移出常驻名单），重新构建。名单偏全：排除都是标记，拿不准的先不标。
   发布前拿游戏自己的文本表核对构建出的名单：列出的名字带出去的 `ja` / `zh` / `en` 会被纠错模型照抄、之后不再校验，
   每个都得是游戏写的字符串。

   ```bash
   python -I tools/game_glossary/textmap_check.py <游戏键> <输出文件> <文本表目录> <核对.tsv> [--readables <原神 Readable 目录>]
   ```

   它按「同一个 key 的日中英是同一句话」逐条对、比写出来的字，列出对不上的与游戏在那里写的字（各类别与比法在它开头）；
   只有 `cooc-ok:ja-missing`（游戏只在台词或书里把三者写在一起的名字）与 `variant-ja-missing`（拼写变体，它的 zh 游戏写过）不用读，它打印的
   「names to read」是其余的条数。逐条看：确认了的写进 `fixes`（原神差两个字以内的也可以收进
   [`genshin-textmap-fixes.json`](genshin-textmap-fixes.json)，`fixes` 优先），重新构建再核对；看过、照留的下次还会列出来
   （2026-11-21 原神留了 11 条，像 天空の島 那条的中文是一整句）。名字有自己的 key 时以那条为准。审读模型对 `ja` / `zh` / `en`
   提的改法也先过这一步：游戏写过的就是对的。文本表只在本地用，不随名单发。
4. **冒烟**（换了挑法或第一次做某款游戏时）：把文件装进一个临时知识库，

   ```bash
   PYTHONPATH=src python -c "import sys, pathlib; from finesub.llm import glossary_store as gs; gs.install(sys.argv[1], pathlib.Path(sys.argv[2]).read_bytes())" <临时库> <输出文件>
   ```

   再拿一段讲这款游戏的录音跑一次纠错（`--knowledge-root <临时库> --extra-info <游戏名>`，模型用 Luna 即可），看 prompt 里那一块与运行的告警。
5. **挂 Release**（挂之前先问 owner）：tag 是 `glossary-<游戏键>-<构建日期>`，资产是 `builds/` 里那份拷成 `glossary-<游戏键>-<构建日期>.json`，
   说明里写上游与上游 commit：

   ```bash
   cp tools/game_glossary/builds/<游戏键>/glossary.json tmp/glossary-<游戏键>-<构建日期>.json
   gh release create glossary-<游戏键>-<构建日期> tmp/glossary-<游戏键>-<构建日期>.json --repo caca2331/finesub --target <最近一个发版 tag 的提交> \
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

   然后把清单和 `builds/` 里发布的那份放进同一个提交、说明写上 tag，提交前看那份的 sha256 与清单里的相同
   （之后又重建过就是错的那份）。用户要等下一个带这份清单的版本才取得到。

## 注意

- **重建**：常驻名单的「近两年」从 `--built` 算，离构建超过半年任务就告警过时，所以大约半年重建一次，重走全部步骤，
  审读也照做：上游每次更新都可能带进新的笔误。旧的 Release 留着，已经装上的用户不依赖它。
- **新游戏**：在 `extract.py` 里加一个抽取，在 `GAMES` 里加一项，核心标签、称呼标签、修正表都按那个上游的标签体系重新定；
  常驻名单收谁由 agent 按它的目标与设计定，不另问 owner（owner 2026-11-21；原神的核心标签是 owner 2026-11-12 定的，
  星铁与绝区零的是 2026-11-13，见
  [要满足什么](../../docs/knowledge.md#一份名单要满足什么)），定了在汇报里写一行。
- **上游**：只提专名做成对照词典，不带剧情文本、图片与代码，上游的许可不另作限制（owner 2026-11-13）；名单单独下载、不进 wheel。
- **撤回**：有人要求停止时，删掉 Release 资产与清单那一行，下一个版本起取不到；已装的副本留在用户本地。
