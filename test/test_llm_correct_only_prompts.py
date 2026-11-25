"""A correct-only task (a Chinese source, docs/llm_prompts.md「只纠错」) is never told to translate.

The guard renders what such a task's model can be shown -- the correction prompt over the switch grid, and
the paths the grid does not reach (the request turn and its fill, the agentic acquire note, task feedback,
the phonetic hints, the style picks, a style written for the task and the set that also names who speaks) -- and bans the words of translating in the whole text. Its scan surface is part of the
guard: a path left out here never goes red. A second scan holds what such a task is really shown (the knowledge
base read, no feedback asked, no phonetic hints) to no kana outside the worked examples, which come from Japanese
material by the owner's call: the cue words the rules quote are Chinese. The entries themselves are data: what a
term line carries in its first column is the library's, not the templates'.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import re
import tempfile
from pathlib import Path

import pytest

from finesub.llm.chunking import (
    CorrectionBudget, SubtitleSegment, SubtitleWindow, load_product_segments, plan_correction_windows,
)
from finesub.llm.knowledge.node.name_trust import UNCHECKED_MARK
from finesub.llm.output_protocol import correction_output_header
from finesub.llm import prompt_compose, prompt_tables
from finesub.llm.prompt_compose import compose_correction_system, compose_correction_user
from finesub.llm.prompt_variants import grouped_variant, resolve_variant
from finesub.llm.prepare_prompts import build_prepare_task_messages, render_style_selection_section
from finesub.llm.prompts import ContextPack, build_correction_csv_messages
from finesub.llm.request_prompts import build_correction_fill_message, build_window_task_messages
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.speaker_judge.prompt import SOURCE_WORDS, shape_section
from finesub.llm.window_speakers import speaker_material, with_speakers

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


def _other_paths(*, taken: bool = False):
    """What the grid does not reach: the request turn and its fill, the agentic acquire note, the
    feedback block, the phonetic hints. ``taken`` leaves out the two blocks no correct-only task is ever
    sent (`_with_task_mode` asks for no feedback; `kb_phonetic_hints.enabled` is off for it): the paths
    as such a task really takes them."""

    base = dict(
        window=_window(),
        context_pack=ContextPack(),
        variant="capableC-polish",
        profile=resolve_profile("text", "local", "quality"),
        knowledge_enabled=True,
        task_update_feedback=not taken,
        phonetic_hints="" if taken else "1|原神 / 角色|桑多涅|桑多涅||木偶",
    )
    indexes = {"streamer": "- 主播A | 测试主播"}
    yield "direct", build_window_task_messages(**base)
    yield "request", build_window_task_messages(request_channel=True, knowledge_indexes=indexes, **base)
    yield "agentic", build_window_task_messages(agentic=True, knowledge_indexes=indexes, **base)
    yield "fill", [
        build_correction_fill_message(
            entry_details="# 主播A", search_results="wiki", requested_entries=True, requested_search=True,
            task_update_feedback=not taken,
        )
    ]


    # `--style auto` among the styles written for a correct-only task: each shape a pick is asked in
    catalog = "- 甲组｜简介｜作者：未记｜适用范围：中文直播"
    for shape, request_channel, agentic in (("style_only", True, False), ("request", True, False),
                                            ("agentic", False, True)):
        profile = resolve_profile("text", "none" if shape == "style_only" else "local", "quality")
        yield f"pick-{shape}", build_window_task_messages(
            **{**base, "profile": profile, "knowledge_enabled": shape != "style_only",
               "task_update_feedback": shape != "style_only" and not taken},
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


def _tracked_window(directory: Path, *, marks: bool) -> SubtitleWindow:
    """A Chinese window carrying speaker material (a single-judged task's), its rows on two tracks; ``marks``: one
    row the acoustic layer is unsure of, which brings the marks' note in."""

    def speaker(track, *others, mark=None):
        payload = {"track": track, "tracks": [track, *others], "block": 0, "coverage": {str(track): 0.9},
                   "labels": [{"role": "unknown", "conf": "high"}]}
        return payload | ({"mark": mark} if mark else {})

    low = {"tier": "low", "alternative": 0} if marks else None
    rows = [(speaker(0), "然后昨天直播的时候"), (speaker(0), "被那个boss打败了"), (speaker(1, mark=low), "啊真的假的"),
            (speaker(1, 0), "嗯真的")]
    stable = directory / f"clip-{marks}-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"id": str(i), "start": 2.0 * i, "end": 2.0 * i + 1.5, "text": text, "speaker": sp}
        for i, (sp, text) in enumerate(rows, start=1)]}, ensure_ascii=False), encoding="utf-8")
    segments, _layer = load_product_segments(stable)
    window = SubtitleWindow("0001", segments, [], "even", CorrectionBudget(0, 0, 0, "test"), clip_start=0.0,
                            clip_end=10.0, length_scale=1.0)
    return with_speakers([window], speaker_material(stable, ContextPack(), knowledge_root=None, title="测试直播"))[0]


def _tracked_paths():
    """The correct-only set that also names who speaks (an intermediate task with the speaker judge on): system and
    user, with and without the confidence marks, over the media. Built here without the knowledge base: its lines are scanned on the other paths."""

    with tempfile.TemporaryDirectory() as directory:
        for marks, media in itertools.product((False, True), MEDIA):
            window = _tracked_window(Path(directory), marks=marks)
            yield f"tracked-{media}-marks={marks}", build_correction_csv_messages(
                window=window, profile=resolve_profile(media, "none", "intermediate"),
                variant="capableC-tracked-polish", knowledge_enabled=False)


OTHER = {name: "\n".join(str(message["content"]) for message in messages)
         for name, messages in itertools.chain(_other_paths(), _tracked_paths())}


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


def test_the_rules_quote_no_japanese_outside_the_worked_examples(monkeypatch):
    """Every path a correct-only task takes, the knowledge base read or not (`--knowledge read` is its default)."""

    monkeypatch.setattr(prompt_compose, "build_examples", lambda **_: "")
    texts = {name: text for name, (_, _, text) in _grid()}
    texts.update({name: "\n".join(str(message["content"]) for message in messages)
                  for name, messages in itertools.chain(_other_paths(taken=True), _tracked_paths())})
    texts.update({f"pass1-{source}": shape_section(source, correct_only=True) for source in SOURCE_WORDS})
    assert "下面的示范" not in "".join(texts.values()), "the examples are left out"
    kana = {name: line for name, text in texts.items() for line in text.splitlines()
            if KANA.search(line) or "假名" in line}
    assert kana == {}


def test_the_term_lines_are_explained_wherever_entries_are_first_shown():
    """A term line's first column is mostly a foreign word: wherever a correct-only task is first shown entries
    (the window's first message, the preparation task) it is told once to write the second column. The fill turn
    continues a conversation that was told; a task that reads no knowledge base has no entries to explain. A
    translation is told how to read the marks too (owner 2026-11-20), in its own words. A window also takes a name
    the background survey settled; the preparation task is that survey, so its note does not name one."""

    mode = prompt_tables.task_mode_texts(True)
    note, prepare_note = (prompt_compose.kb_term_lines_note(mode, window=w).strip() for w in (True, False))
    translation_note = prompt_compose.kb_term_lines_note(prompt_tables.task_mode_texts(False), window=True).strip()
    assert note and translation_note and note != translation_note
    assert UNCHECKED_MARK in note and UNCHECKED_MARK in translation_note, "it says what the mark means"
    assert "背景调查" in note and "背景调查" not in prepare_note
    for name, (_, _, text) in GRID.items():
        assert text.count(note) == ("kb=True" in name), name
    for name, text in OTHER.items():
        shown = name in ("direct", "request", "agentic", "pick-request", "pick-agentic")
        assert text.count(note) == shown, name
        assert text.count(prepare_note) == name.startswith("prepare-"), name


def test_the_preparation_task_keeps_a_spoken_name_the_library_writes_differently():
    """The preparation task's rule takes the library's name over the transcript's. A correct-only task writes what
    was said, so its rule excepts a row whose name nobody checked (the `（未核）` mark the projection puts on it: a
    sourced spelling from the search results comes first, else what was said) and a name said repeatedly that
    sounds clearly different; a translation's rule excepts only the unchecked row (owner 2026-11-20)."""

    exception = prompt_tables.task_mode_texts(True)["kb_conflict_spoken"]
    assert exception and UNCHECKED_MARK in exception
    assert UNCHECKED_MARK in prompt_tables.task_mode_texts(False)["kb_conflict_spoken"]
    for name, text in OTHER.items():
        assert text.count(exception) == name.startswith("prepare-"), name
    assert all(exception not in text for _, _, text in GRID.values())


def test_the_other_paths_really_carry_their_blocks():
    """The scan above is only worth what it reaches."""

    assert "<task_update_feedback>" in OTHER["direct"]
    assert "<kb_phonetic_hints>" in OTHER["direct"]
    assert "kb_search" in OTHER["agentic"]
    assert "<requested_entries>" in OTHER["request"]
    tracked = [OTHER[name] for name in OTHER if name.startswith("tracked-")]
    assert len(tracked) == 6 and all("`spk`" in text and "轨迹画像" in text for text in tracked)
    assert all(("（低，备选" in OTHER[name]) == name.endswith("marks=True") for name in OTHER
               if name.startswith("tracked-"))


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
