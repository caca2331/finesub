"""Production's correction with the merge rules relaxed to the owner's principles (docs/reports/2026-10-17-three-pass.md
「三个 pass 的分工」 merge principles): same person first; no merge over a long pause; length around the hard threshold,
a bit longer for one sentence, never over the absolute one. Replaces the merge-rule fragment (relaxed_merge_rules.md),
the variant clauses that repeat the hard threshold, the user reminder's "most sources stay single", and drops the two
length-veto examples. merge_trial.py runs it when TRIAL_RULES=relaxed; two_pass.py's pass 2 installs it with its own
note in front of the merge rules.

    python tools/diarization_explore/relaxed_rules.py <correction_translation args>
"""
from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path
from string import Template

RULES = Path(__file__).with_name("relaxed_merge_rules.md")
VETOES = {"length-veto-span", "length-veto-chars"}
#: the other places that tell most sources to stay single or teach the hard-threshold veto
REWRITES = [
    ("可以合并语义连贯的短片段，但多数源应保持独立；", "同一个人的同一句话被切开时合并；"),
    ("很短、间隔很近且语义连贯的碎片按中央合并规则处理，多数源应保持独立。", "同一个人的同一句话被切开时，按中央合并规则合并。"),
    ("多数源保持独立，仅在合并规则明确允许时合并。", "按合并规则合并。"),
    ("且可在「けど」处自然分行，拒绝越过硬门槛", "，可在「けど」处自然分成两条"),
    ("下方字数否决例展示硬门槛否决；", ""),
    ("展示可自然分行的句子拒绝越过硬门槛", "展示较长的句子在自然断点处分成两条"),
]


def clauses(slot):
    from finesub.llm.prompt_constants import (ABSOLUTE_MERGE_CHARS as A, ABSOLUTE_MERGE_SECONDS as AS,
                                              HARD_MERGE_CHARS as H, HARD_MERGE_SECONDS as HS,
                                              SANDWICH_MAX_CHARS, SANDWICH_MAX_SECONDS)
    return {
        "translated_merge_rule": (
            f"合并后 `char_count` 一般在 {H} 字、跨度 {HS} 秒左右以内；同一个人的同一句话确实连着时可以稍长，"
            f"并把理由{slot}。`char_count` >{A} 或跨度 >{AS} 秒是**绝对门槛**，任何情况都必须拒绝。"
            f"filler 三明治三源行更严：合计 ≤{SANDWICH_MAX_SECONDS} 秒且合并后 ≤{SANDWICH_MAX_CHARS} 字。"),
        "pacing_merge_clause": (
            f"1. 拟合并时先估算合并后的字数和跨度：一般在 {H} 字/{HS} 秒左右以内，同一句话确实连着可以稍长；"
            f"字数 >{A} 或跨度 >{AS} 秒是**绝对门槛**，任何多源合并都不得越过；单个源自身超限时如实输出，"
            "不因该门槛丢弃。源数遵守 position 规则。\n"
            "2. 多源 position 用英文逗号连接两个连续源序号。本地时间轴取首源开始到末源结束；gap 取末源到下一源的间隔。"),
        "merge_reminder": (
            f"同一个人的同一句话被切开、中间没有长停顿时合并；并完一般在 {H} 字/{HS} 秒左右，"
            f"{A} 字/{AS} 秒为绝对门槛，任何情况不得越过。"),
        "mid_reminder_merge_rule": (
            f"合并后一般在 {H} 字/{HS} 秒左右以内，同一句话确实连着可以稍长；超 {A} 字或超 {AS} 秒即越绝对门槛，"
            "任何情况禁止（单源豁免）。"),
    }


def rewrite(text):
    for old, new in REWRITES:
        text = text.replace(old, new)
    # a single source over the usual length still gets its `#` note, so the gate says 超过, not 并完超过
    text = re.sub(r"越过硬门槛（字数 >(\d+) 或跨度 >(\d+)s）", r"超过 \1 字或 \2 秒", text)
    return text.replace("硬门槛", "一般长度")


def mark_example_merges(text, cell):
    """Each example's `<asr_result>` with its `turn` column reading `cell` on exactly the rows its `<translated>`
    merges into the row before (voided merges too), so the examples obey 「merge only where marked」."""
    def fix(match):
        merged = {i for line in match.group(4).splitlines() if re.match(r"\s*sub\|[\d,]+\|", line)
                  for i in line.split("|")[1].split(",")[1:]}
        rows = []
        for line in match.group(2).splitlines():
            cells = line.split("|", 5)
            if len(cells) == 6 and cells[0].strip().isdigit():
                cells[4] = cell if cells[0].strip() in merged else ""
            rows.append("|".join(cells))
        return match.group(1) + "\n".join(rows) + "\n" + match.group(3) + match.group(4)
    return re.sub(r"(<asr_result>\n)(.*?)\n?(</asr_result>)(.*?<translated>.*?</translated>)", fix, text, flags=re.S)


#: a may_join run's rewrites of production text that speaks of judging the speaker in pass 2
MAY_REWRITES = [("；无法确认是否同一人时同样按不同人处理、不合并。", "；第二行没标「可并」，本来就不并。")]


def install(prefix="", turns=None, turn_cell=None, rules=RULES, turn_starts=()):
    """Patch production's prompt assembly in this process; `prefix` goes in front of the merge rules in `rules`.
    `turns` replaces the turn fragments by name, and `turn_cell` flips the input's `turn` column: it then reads
    `turn_cell` on the lines *without* a turn and nothing on the lines with one (a may_join run marks the mergeable
    seams; the examples get the same marks on the rows they merge). Of the other lines, those starting at
    `turn_starts` (seconds, 3 decimals) read the change cell: the judged speaker changes there."""
    from finesub.llm import chunking, example_builder, prompt_compose, prompt_variants

    original_template, original_examples = prompt_compose.load_prompt_template, example_builder.load_examples
    original_built = prompt_compose.build_examples

    def template(name, **values):
        if name == "fragment_merge_rules_nosingles_v1.md":
            return prefix + Template(rules.read_text(encoding="utf-8")).safe_substitute(
                {"prompt_version": prompt_compose.PROMPT_VERSION, **{k: str(v) for k, v in values.items()}})
        if turns and name in turns:
            return turns[name]
        text = rewrite(original_template(name, **values))
        return may_text(text) if turn_cell else text

    def may_text(text):
        for old, new in MAY_REWRITES:
            text = text.replace(old, new)
        return mark_example_merges(text, turn_cell)

    def examples(*a, **k):
        return tuple(e for e in original_examples(*a, **k) if e.id not in VETOES)

    def built(*a, **k):
        text = rewrite(original_built(*a, **k))
        return may_text(text) if turn_cell else text

    for name in ("capableB", "capableC", "basicB"):
        variant = prompt_variants.VARIANTS[name]
        prompt_variants.VARIANTS[name] = dataclasses.replace(variant, **clauses(variant.approval_reason_slot))
    prompt_compose.load_prompt_template, example_builder.load_examples = template, examples
    prompt_compose.build_examples = built
    if turn_cell:
        original_render = chunking.render_segments_as_csv

        starts = set(turn_starts)

        def render(segments, **kwargs):
            text = original_render(segments, **kwargs)
            if not kwargs.get("turn_column"):
                return text
            rows = [line.split("|", 5) for line in text.splitlines()]
            for n, (cells, segment) in enumerate(zip(rows, segments)):
                if len(cells) == 6:
                    if cells[4] != chunking.Seam.CHANGE.value and n > 0:
                        cells[4] = turn_cell
                    else:  # the window's first line has no line before it here to merge into
                        cells[4] = chunking.Seam.CHANGE.value if round(segment.start, 3) in starts else ""
            return "\n".join("|".join(cells) for cells in rows) + "\n"

        chunking.render_segments_as_csv = render
        # a repair round in this process: its turn notes speak of the marks this prompt shows
        from finesub.llm import output_protocol
        original_validate = output_protocol.validate_correction_output_text

        def validate(*a, **k):
            result = original_validate(*a, **k)
            notes = tuple(note.replace("的 turn 列标了换人", f"的 turn 列没标「{turn_cell}」")
                          for note in result.turn_notes)
            return dataclasses.replace(result, turn_notes=notes)

        output_protocol.validate_correction_output_text = validate


def main(args):
    from finesub.llm import correction_translation

    install()
    sys.argv = ["correction_translation", *args]
    sys.exit(correction_translation.main())


if __name__ == "__main__":
    main(sys.argv[1:])
