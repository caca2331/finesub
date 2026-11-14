"""Settle `splittable` by asking a model, because a character count could not settle it.

⚠⚠ **Its labelling role is SUPERSEDED -- read `turn_output_judge.py` instead** (owner 2026-09-20:
「感觉是裁判或裁判 prompt 的问题」). It asked the judge to label the TRUTH ("does this row's text
already hold two people's words?"), and the judge answered that well; the question was wrong. It
said NO on every row where the second person spoke but was never transcribed -- 51 of the
character rule's 59 positives -- and those are the rows v2 exists to fix, so a truth built this way
can never credit v2. The replacement grades each model's SPLIT against the truth instead.
It stays because `turn_output_judge` imports its target pin (`pinned_roles`, `JUDGE_TARGET`) and
its truth presentation (`people_in`, `format_people`), so both judges read one format.
Its 207 labels (`judge-labels.jsonl`) are kept as data, not as a truth anything is scored against.

Owner 2026-09-20: 「量法是调用模型让其打分。判断模型用 workbuddy 的 ds 41f 即可。」

`turn_shape_bed` decides whether a row holds two people's words by counting characters, and that
constant turned out to be justified against the wrong language -- onetonton's refined axis is a
Chinese fansub, not a Japanese transcript -- with the class count swinging 1428 / 2121 / 2872 /
3377 as the threshold moves 4 / 3 / 2 / 1. No value of it is defensible, because the quantity it
counts is a translation's length, not the row's content.

So the question goes to a model instead, phrased as what it actually is:

    given that these people spoke during this row, does the row's OWN Japanese text carry
    meaningful content from more than one of them?

⚠⚠ **The judge sees only the truth side.** The row's text, and the refined lines inside it with
their speakers and Chinese content. It never sees what `audio-agy` or `text-agy` answered, or
that a comparison exists -- a judge that knew the hypothesis would be scoring it.

⚠ **A different family from the model under test** (DeepSeek via CodeBuddy against agy's Gemini
front), so the label cannot inherit the tested model's blind spots.

⚠ Rows where only one person speaks inside at all are not judged: they cannot hold two people's
words whatever a model says, and spending a call to be told so would only add noise.

    python tools/diarizen_export/turn_truth_judge.py --dry-run
    python tools/diarizen_export/turn_truth_judge.py          # spends CodeBuddy credits
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "diarization_explore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = REPO / "tmp" / "diarizen-bed" / "turn-llm"

#: The judge (owner 2026-09-20). ⚠ Deliberately not the family under test. Labels land in
#: `judge-labels.jsonl`; `--recheck` writes `judge-recheck.jsonl` so the two passes stay apart.
JUDGE_TARGET = "local-codebuddy-deepseek-v4_1-flash"

#: Rows per call. Small enough that the reply stays one line per row and parses.
BATCH = 15

#: A person on for less than this inside the row is not listed at all. ⚠ The first version
#: rounded to one decimal and printed 「小日向美香（0.0s）」 next to a sentence that was spoken in
#: the PREVIOUS row -- a speaker who by the bed's own numbers said nothing here, offered to the
#: judge as the second person. Whatever the judge did with that, it was not the question.
MIN_PRESENCE = 0.15

#: A refined line with less than this share of itself inside the row is shown as 部分, because the
#: sentence the judge reads is then mostly somebody else's row. Nothing is silently trimmed.
INSIDE_SHARE = 0.5

#: Characters of a refined line shown before truncating, with an explicit ellipsis. ⚠ The first
#: version cut at 70 with no mark, so a truncated sentence read as a short one -- which is the
#: single cue the judge is being asked to weigh.
SHOW_CHARS = 90

BLOCK = re.compile(r"<judge>(.*?)</judge>", re.S)

SYSTEM = """你在核对一份**日语**对谈节目的字幕。

**这个判断的用途**：字幕里有些行把两个人的话并进了一句，需要拆成两行。你要判断的就是
**这一行到底该不该拆**——只有当这一行的日语文字里**真的同时装着两个人各自有实际意义的话**时，
才有一刀可切。

每一条给你四样：

- `#N`　　　编号。**回答时的 `id` 就写这个数字**（从 0 开始）。
- `字幕`　　这一行的**日语**原文，自动转写的结果。⚠ **它可能漏字、可能只记下了一个人的话。**
- `时长`　　这一行有多长。
- `期间发言`　人工精修时间轴说，在这一行的时间跨度内有谁说了话。**按开口先后排列**，给出
  　　　　　每个人的**中文译文**、他在本行内占了多少秒、以及这句话是整句落在本行内（`完整`）
  　　　　　还是只有一部分落进来（`部分`，那多半是相邻行的话延伸过来的）。

⚠ **中文是译文，不是原文。** 不要去做逐字对应，只用它判断「这个人在这一行里说了什么意思」。

---

**问题：这一行的日语文字里，是不是装着两个（或更多）人各自有实际意义的话？**

答 `YES` 要同时满足两条：

1. **至少两个人说了有实际意义的内容。** 下列都**不算**有实际意义：
   附和（嗯、对、是啊、没错、我也是）、语气词（诶、哦、啊、欸）、笑声、
   以及只是把对方刚说的话重复一遍。
2. **他们的话在这一行的日语原文里都能找到**——看意思对得上就行，不要求字面对应，
   一部分也算。

答 `NO` 的常见情形：

- 只有一个人说了有意义的话，其他人只是附和或出声；
- 第二个人确实说了有意义的话，**但日语原文里根本没写到**（转写漏了）
  ⇒ 这一行的文字仍然只属于一个人，**没有第二个人的字可拆**；
- 两个人同时说，原文只记下了其中一个；
- 标着 `部分` 的那句话，其实主要发生在相邻的行里。

答 `UNSURE`：**拿不准就答这个，不要猜。** 比如原文太短或明显残缺、
译文和原文对不上、或者两个人的话你分不出哪句属于谁。⚠ 这一项不会被当成错，
猜错才会。

⚠ **不要因为「期间发言里出现了两个名字」就答 YES。** 名字出现只说明那时有人出声，
真正的判据是**这一行的日语文字里有没有两个人的话**。

---

按下面的格式回答，**每条一行，顺序随意但 `id` 必须对**，不要有别的内容：

<judge>
{"id": 0, "answer": "YES", "who": ["甲", "乙"], "why": "一句话理由"}
{"id": 1, "answer": "NO", "why": "一句话理由"}
{"id": 2, "answer": "UNSURE", "why": "一句话理由"}
</judge>

`who` 只在答 YES 时给，写是哪两个人的话被并进了这一行。"""


def pinned_roles(target_id: str):
    """`{role: config}` whose whole endpoint chain is `target_id`, whatever preset binds it.

    ⚠ Built from the routing tables rather than from one preset's group: the judge is chosen for
    being a DIFFERENT family from everything the presets bind for this task, so requiring it to
    already sit in the bound group would have made "pick an independent judge" impossible.
    """
    from dataclasses import replace

    from finesub.llm.routing.config import LLMRole, ModelEndpoint, role_config_for
    from finesub.llm.routing.model_routes import default_model_routes

    routes = default_model_routes()
    target = routes.targets[target_id]
    fact = routes.target_fact(target_id)
    endpoint = ModelEndpoint(
        provider_tier=fact.provider_tier,
        api_model_id=fact.api_model_id,
        target_id=target.id,
        fact_id=target.fact_id,
        backend=target.backend,
        native_search_tool=routes.targets[target_id].native_search_tool,
    )
    base = role_config_for("research", role=LLMRole.GENERAL_CAPABLE)
    # ⚠⚠ `model_group_id=""` or the pin is decorative: `ModelRouter.plan` takes the group path
    # whenever that id is set and never reads `endpoint_chain` there.
    return {LLMRole.GENERAL_CAPABLE: replace(base, endpoint_chain=(endpoint,),
                                             test_endpoint=endpoint, model_group_id="")}


def judgeable(row, events):
    """What the truth judge is shown for one row, or None when it cannot hold two people's words."""
    people = people_in(row, events)
    return people if len(people) >= 2 else None


def people_in(row, events):
    """Everyone audible in the row above the floor, as the judges are shown them -- even ONE.

    ⚠ Split out of `judgeable` because the output judge needs the single-speaker case too: a row
    where only one person really speaks is exactly the row a split model should NOT have cut, and
    showing the grader nothing there would hide the one verdict that punishes a wrong cut.

    ⚠ Ordered by when each person first opens their mouth INSIDE the row, not by how long they
    hold it. Sorting by duration puts the main speaker first every time, which is a cue the judge
    can learn instead of reading; the sequence is also the thing that distinguishes a handover
    from simultaneous speech, so it is information rather than a hint.
    """
    from turn_shape_bed import speech_inside

    start, end = float(row["start"]), float(row["end"])
    seconds, _ = speech_inside(start, end, events)
    # ⚠ Anyone below the floor is dropped here, so the two-person gate in `judgeable` counts only
    # people who are really audible in the row.
    present = {who: value for who, value in seconds.items() if value >= MIN_PRESENCE}

    lines, first_at, already = {}, {}, set()
    for line_start, line_end, who, text in events:
        overlap = min(line_end, end) - max(line_start, start)
        if overlap <= 0 or who not in present:
            continue
        whole = overlap / max(line_end - line_start, 1e-6) >= INSIDE_SHARE
        shown = text if len(text) <= SHOW_CHARS else text[:SHOW_CHARS] + "…"
        lines.setdefault(who, []).append((shown, whole))
        # ⚠ Two different facts, kept apart. Clamping "was already talking" to the row start
        # printed 「第 0.0 秒开口」 for almost everybody, which destroyed the sequencing the judge
        # is meant to read AND contradicted the prompt's own claim about the ordering.
        if line_start < start:
            already.add(who)
        else:
            first_at[who] = min(first_at.get(who, 1e9), line_start - start)

    def opened(who):
        return None if who in already else first_at.get(who, 0.0)

    # Already-talking speakers come first; the rest by when they cut in.
    return [(who, present[who], opened(who), lines.get(who, []))
            for who in sorted(present, key=lambda name: -1.0 if name in already
                              else first_at.get(name, 0.0))]


def format_people(people) -> list[str]:
    """The truth block, one line per person. Shared by both judges so they read one format."""
    if not people:
        return ["     （真值里这一行的时间跨度内没有任何人的台词）"]
    spoke = []
    for who, seconds, offset, lines in people:
        body = "；".join(f"{text}（{'完整' if whole else '部分'}）" for text, whole in lines)
        when = "本行开始前就在说" if offset is None else f"第 {offset:.1f} 秒开口"
        spoke.append(f"     - {who}　{when}，占 {seconds:.1f} 秒：{body}")
    return spoke


def prompt_for(batch) -> str:
    parts = []
    for index, (_, row, people) in enumerate(batch):
        spoke = format_people(people)
        span = float(row["end"]) - float(row["start"])
        head = [f"#{index}",
                f"  字幕：{str(row.get('text', '')).strip()}",
                f"  时长：{span:.1f} 秒",
                "  期间发言："]
        parts.append("\n".join(head + spoke))
    return "\n\n".join(parts) + f"\n\n共 {len(batch)} 条，逐条判断。"


def record_people(people):
    """`people` as JSON-safe rows. ⚠ Its own function because it is the ONLY code path a call
    has to survive after the money is spent: the first run crashed here unpacking a 3-tuple that
    had grown a fourth field, after the batch had already been sent and answered. `--dry-run`
    exercises it now, so the shape is checked before anything is billed."""
    return [{"who": who, "seconds": round(seconds, 2),
             "opened": None if offset is None else round(offset, 2),
             "lines": [{"text": text, "whole": whole} for text, whole in lines]}
            for who, seconds, offset, lines in people]


def parse(text: str) -> dict:
    match = BLOCK.search(text or "")
    if not match:
        return {}
    found = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            found[int(payload["id"])] = (str(payload.get("answer", "")).upper(),
                                         str(payload.get("why", ""))[:120])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return found


def guarded():
    from finesub.llm.client import RoleClient
    from finesub.llm.routing.config import LLMRole

    client = RoleClient(role_configs=pinned_roles(JUDGE_TARGET))

    def complete(system: str, user: str):
        reply = client.complete(LLMRole.GENERAL_CAPABLE,
                                [{"role": "system", "content": system},
                                 {"role": "user", "content": user}])
        target = str(getattr(reply, "target_id", ""))
        if target != JUDGE_TARGET:
            raise SystemExit(f"⚠ the judge call landed on {target!r}, not {JUDGE_TARGET!r}. "
                             f"Stopped after ONE call.")
        return reply.content, target

    return complete


def sampled_rows(episodes):
    """Every judgeable row in the 25 windows C.5d sampled, in a stable order."""
    import numpy as np

    from turn_llm_bed import NATIVE, candidates, windows
    from turn_shape_bed import episode_rows

    pool = []
    for episode in episodes:
        loaded = episode_rows(episode)
        path = NATIVE / f"onetonton-{episode:02d}.npz"
        if loaded is None or not path.exists():
            continue
        stable, _, _, events = loaded
        marked = candidates(stable, np.load(path)["probs"])
        for start, _, rows in windows(stable, marked):
            pool.append((episode, start, rows, stable, events))
    generator = np.random.default_rng(20260920)
    chosen = [pool[i] for i in
              generator.choice(len(pool), size=min(25, len(pool)), replace=False)]

    found = []
    for episode, _, rows, stable, events in chosen:
        for index in rows:
            people = judgeable(stable[index], events)
            if people is not None:
                found.append((f"{episode}-{round(float(stable[index]['start']), 1)}",
                              stable[index], people))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    # ⚠ Without this the judge is just another unvalidated ruler -- which is how this line got
    # here. A second pass over a subset, SHUFFLED so the rows land in different batches with
    # different neighbours, says how much of the label is the row and how much is the context it
    # happened to be graded in.
    parser.add_argument("--recheck", type=int, default=0,
                        help="re-judge this many random rows in a different batch composition")
    arguments = parser.parse_args()

    rows = sampled_rows(arguments.episodes)
    if arguments.recheck:
        import random

        shuffled = list(rows)
        random.Random(31415).shuffle(shuffled)
        rows = shuffled[: arguments.recheck]
    labels = OUT / ("judge-recheck.jsonl" if arguments.recheck else "judge-labels.jsonl")
    done = {json.loads(line)["key"] for line in labels.read_text(encoding="utf-8").splitlines()} \
        if labels.exists() else set()
    todo = [row for row in rows if row[0] not in done]
    print(f"{len(rows)} judgeable rows in the sampled windows; {len(todo)} still to judge "
          f"({(len(todo) + BATCH - 1) // BATCH} calls)")
    if arguments.dry_run:
        print(prompt_for(todo[:3]))
        # ⚠ Serialise one record too. The write path is the only thing a spent call cannot be
        # protected from, and leaving it out of the rehearsal is what threw away a batch.
        key, row, people = todo[0]
        json.dumps({"key": key, "answer": "YES", "why": "", "text": str(row.get("text", "")),
                    "people": record_people(people)}, ensure_ascii=False)
        print(f"\n(write path rehearsed on {key}: ok)")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    complete = guarded()
    for offset in range(0, len(todo), BATCH):
        batch = todo[offset:offset + BATCH]
        answers = {}
        for attempt in range(4):
            try:
                text, target = complete(SYSTEM, prompt_for(batch))
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"batch {offset}: {type(error).__name__} {str(error)[:90]}; "
                      f"retry {attempt + 1}")
                time.sleep(15 * (attempt + 1))
                continue
            answers = parse(text)
            if answers:
                break
            print(f"batch {offset}: no <judge> rows parsed; retry {attempt + 1}")
        if not answers:
            print(f"batch {offset}: gave up")
            continue
        with labels.open("a", encoding="utf-8") as handle:
            for index, (key, row, people) in enumerate(batch):
                answer, why = answers.get(index, ("", ""))
                if not answer:
                    continue
                handle.write(json.dumps({"key": key, "answer": answer, "why": why,
                                         "text": str(row.get("text", "")),
                                         "people": record_people(people)},
                                        ensure_ascii=False) + "\n")
        said = sum(1 for value, _ in answers.values() if value == "YES")
        print(f"  batch {offset:3d} [{target}] {len(answers)}/{len(batch)} judged, "
              f"{said} YES")


if __name__ == "__main__":
    main()
