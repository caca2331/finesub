"""A correct-only task (a Chinese source, docs/llm_prompts.md「只纠错」) is never told to translate.

The guard renders what such a task's model can be shown -- the correction prompt over the switch grid, and
the paths the grid does not reach (the request turn and its fill, the agentic acquire note, task feedback,
the phonetic hints, the style picks and a style written for the task) -- and bans the words of translating in the whole text. Its scan surface is part of the
guard: a path left out here never goes red. A second scan holds what such a task is really shown (no knowledge
base) to no kana outside the worked examples, which come from Japanese material by the owner's call: the cue words
the rules quote are Chinese.
"""

from __future__ import annotations

import dataclasses
import itertools
import re

import pytest

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.output_protocol import correction_output_header
from finesub.llm import prompt_compose, prompt_tables
from finesub.llm.prompt_compose import compose_correction_system, compose_correction_user
from finesub.llm.prompt_variants import grouped_variant, resolve_variant
from finesub.llm.prepare_prompts import build_prepare_task_messages, render_style_selection_section
from finesub.llm.prompts import ContextPack
from finesub.llm.request_prompts import build_correction_fill_message, build_window_task_messages
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.speaker_judge.prompt import SOURCE_WORDS, shape_section

BANNED = re.compile(r"译|translation", re.IGNORECASE)
# Literals that name something rather than ask for a translation: the knowledge base's own marker for a
# transliterated name, and an example of a query topic in the search-query rules.
EXEMPT = ("「音译」", "中日互译")
KANA = re.compile(r"[\u3040-\u30ff]")

MEDIA = ("text", "audio", "video")
RETRIEVAL = ("none", "local", "native")
DIFFICULTY = ("quality", "intermediate", "efficiency")
SHAPES = (("capableC-polish", False), ("capableC-polish", True), ("basicA-polish", False))


def _banned_lines(text: str) -> list[str]:
    for literal in EXEMPT:
        text = text.replace(literal, "")
    return [line for line in text.splitlines() if BANNED.search(line)]


def _window():
    return dataclasses.replace(
        plan_correction_windows([SubtitleSegment("1", 0.0, 1.0, "大家好。"), SubtitleSegment("2", 1.5, 2.5, "开始吧。")])[0],
        length_scale=1.0,
    )


def _grid(knowledge_states=(True, False)):
    # speakers: none, labels (`turn` column), or keyed lines from a joint transcription (`同时` / `换回`; the
    # route recommended for Chinese sources)
    for media, retrieval, difficulty, (variant, grouped), knowledge, speakers in itertools.product(
        MEDIA, RETRIEVAL, DIFFICULTY, SHAPES, knowledge_states, ("none", "turns", "keyed")
    ):
        turns, keyed = speakers != "none", speakers == "keyed"
        if grouped and not turns:
            continue
        profile = resolve_profile(media, retrieval, difficulty)
        system = compose_correction_system(
            profile, variant=variant, grouped=grouped, turns=turns, keyed=keyed, cloud=keyed,
            knowledge_enabled=knowledge, length_scale=1.0,
        )
        user = compose_correction_user(
            profile,
            variant=variant,
            knowledge_enabled=knowledge,
            general_context_json="{}",
            window_context="（无）",
            entry_details="（无）",
            previous_advice="（无）",
            pre_round_notes="",
            search_results="（无）",
            preceding_context_csv="-1|-1.0|0.5||前文",
            current_asr_csv="1|0.0|1.0|||大家好\n2|1.1|1.0|0.1|可并|开始吧" if turns else "1|0.0|1.0||大家好",
            current_asr_row_count=2 if turns else 1,
            turns=turns,
            keyed=keyed,
            grouped=grouped,
            length_scale=1.0,
        )
        yield f"{media}/{retrieval}/{difficulty}/{variant}{'/grouped' if grouped else ''}/kb={knowledge}/{speakers}", (
            variant,
            grouped,
            system + "\n" + user,
        )


GRID = dict(_grid())


def _other_paths():
    """What the grid does not reach: the request turn and its fill, the agentic acquire note, the
    feedback block, the phonetic hints."""

    base = dict(
        window=_window(),
        context_pack=ContextPack(),
        variant="capableC-polish",
        profile=resolve_profile("text", "local", "quality"),
        knowledge_enabled=True,
        task_update_feedback=True,
        phonetic_hints="1|原神 / 角色|桑多涅|桑多涅||木偶",
    )
    indexes = {"streamer": "- 主播A | 测试主播"}
    yield "direct", build_window_task_messages(**base)
    yield "request", build_window_task_messages(request_channel=True, knowledge_indexes=indexes, **base)
    yield "agentic", build_window_task_messages(agentic=True, knowledge_indexes=indexes, **base)
    yield "fill", [
        build_correction_fill_message(
            entry_details="# 主播A", search_results="wiki", requested_entries=True, requested_search=True,
            task_update_feedback=True,
        )
    ]


    # `--style auto` among the styles written for a correct-only task: each shape a pick is asked in
    catalog = "- 甲组｜简介｜作者：未记｜适用范围：中文直播"
    for shape, request_channel, agentic in (("style_only", True, False), ("request", True, False),
                                            ("agentic", False, True)):
        profile = resolve_profile("text", "none" if shape == "style_only" else "local", "quality")
        yield f"pick-{shape}", build_window_task_messages(
            **{**base, "profile": profile, "knowledge_enabled": shape != "style_only",
               "task_update_feedback": shape != "style_only"},
            request_channel=request_channel, agentic=agentic, style_catalog=catalog, extra_style_hint="多用短句",
            knowledge_indexes=indexes if shape != "style_only" else None,
        )
    yield "pick-prepare", [{"content": render_style_selection_section(catalog, extra_style="多用短句", read=read,
                                                                     correct_only=True)}
                           for read in ("", "optional", "required")]
    # the whole preparation task, not only its pick: its duties and background are worded by the task too
    for agentic, queries in ((False, True), (False, False), (True, True)):
        yield f"prepare-{'agentic' if agentic else 'batched'}-{queries}", build_prepare_task_messages(
            transcript="--- window 1 ---\n1|大家好", knowledge_indexes=indexes, emits_queries=queries,
            agentic=agentic, style_catalog=catalog, speakers=True, correct_only=True,
        )


OTHER = {name: "\n".join(str(message["content"]) for message in messages) for name, messages in _other_paths()}


@pytest.mark.parametrize("name", sorted(GRID))
def test_the_correction_prompt_never_asks_for_a_translation(name):
    assert _banned_lines(GRID[name][2]) == []


@pytest.mark.parametrize("name", sorted(OTHER))
def test_the_other_window_paths_never_ask_for_a_translation(name):
    assert _banned_lines(OTHER[name]) == []


#: A mode-text key left as `$key` / `${key}` in a rendered prompt: a caller that renders a template without the
#: mode texts (`load_prompt_template` is not strict) sends the slot to the model. An ASCII boundary: a slot is
#: often followed by Chinese (`$may_join_basis决定`), which `\b` would miss.
SLOT = re.compile(r"\$\{?(%s)(?![A-Za-z0-9_])" % "|".join(
    sorted({key for table in prompt_tables._mode_table().values() for key in table}, key=len, reverse=True)))


@pytest.mark.parametrize("name", sorted(GRID) + sorted(OTHER))
def test_no_mode_text_slot_is_left_unfilled(name):
    text = GRID[name][2] if name in GRID else OTHER[name]
    assert SLOT.findall(text) == []


#: The other paths a correct-only task really takes: it never reads the knowledge base (`_with_task_mode`).
WITHOUT_KNOWLEDGE = ("pick-style_only", "pick-prepare", "prepare-")


def test_the_rules_quote_no_japanese_outside_the_worked_examples(monkeypatch):
    monkeypatch.setattr(prompt_compose, "build_examples", lambda **_: "")
    texts = {name: text for name, (_, _, text) in _grid(knowledge_states=(False,))}
    texts.update({name: "\n".join(str(message["content"]) for message in messages)
                  for name, messages in _other_paths() if name.startswith(WITHOUT_KNOWLEDGE)})
    texts.update({f"pass1-{source}": shape_section(source, correct_only=True) for source in SOURCE_WORDS})
    assert "下面的示范" not in "".join(texts.values()), "the examples are left out"
    kana = {name: line for name, text in texts.items() for line in text.splitlines()
            if KANA.search(line) or "假名" in line}
    assert kana == {}


def test_the_other_paths_really_carry_their_blocks():
    """The scan above is only worth what it reaches."""

    assert "<task_update_feedback>" in OTHER["direct"]
    assert "<kb_phonetic_hints>" in OTHER["direct"]
    assert "kb_search" in OTHER["agentic"]
    assert "<requested_entries>" in OTHER["request"]


_LITERAL_ROW = re.compile(r"`(sub\|[^`]*)`")


@pytest.mark.parametrize("name", sorted(GRID))
def test_sample_rows_have_the_header_cell_count(name):
    """A worked row written into a fragment (`sub|5,6|...`) has as many cells as this task's header; the
    example blocks are rendered from the header and need no check."""

    variant, grouped, text = GRID[name]
    v = resolve_variant(variant)
    header = correction_output_header(grouped_variant(v) if grouped else v)
    width = len(header.split("|"))
    wrong = [row for row in _LITERAL_ROW.findall(text) if len(row.split("|")) != width]
    assert wrong == [], (header, wrong)


def test_the_keyed_cells_really_explain_the_keyed_values():
    """Same: the keyed cells are only worth scanning if they render the `同时` / `换回` wording."""

    assert all("换回" in GRID[name][2] for name in GRID if name.endswith("/keyed"))


def test_the_pick_paths_really_ask_for_a_pick():
    assert all("<style_selection>" in OTHER[name] for name in OTHER if name.startswith("pick-"))
    assert all("多用短句" not in OTHER[name] for name in OTHER if name.startswith("pick-")), "no --extra-style"


def test_a_style_written_for_the_task_renders_without_a_translation_word(tmp_path):
    """The block's own wording (the user's lines are theirs to write)."""
    from finesub.llm.knowledge.maintain import main as maintain_main
    from finesub.llm.knowledge.node.edit import edit_subject
    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.knowledge.style import render_style_block

    root = tmp_path / "knowledge"
    assert maintain_main(["--root", str(root), "new", "style", "甲组", "--intro", "简介"]) == 0
    repo = KnowledgeRepo.open(root)
    edit_subject(repo, repo.subjects("style")[0], (
        "# 甲组\n\n简介\n\n## 档案\n\n[任务] 中文纠错\n\n## 约定\n\n- [加注] 谐音梗加括注\n\n"
        "## 正例\n\n- [加注] 好家伙 → 好家伙（谐音梗）\n"
    ))
    assert _banned_lines(render_style_block(root, ("甲组",), correct_only=True)) == []


def test_only_a_window_with_audio_is_told_what_it_can_hear():
    """Cold read P735 (owner 2026-11-07) and P728: a text window cannot tell a word ASR missed from one never
    said, nor crying turned to gibberish from a line that was spoken."""

    for name, (_variant, _grouped, text) in GRID.items():
        heard = not name.startswith("text/")
        assert ("漏转" in text) == heard, name
        assert ("哭声、喘息这类高噪声片段除外" in text) == heard, name
