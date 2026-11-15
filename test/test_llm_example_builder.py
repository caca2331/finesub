"""Builder-layer guarantees from docs/llm_prompts.md.

Each test here corresponds to a row of that plan's mechanism table: the point
is not that the current examples happen to be right, but that a class of
inconsistency can no longer be expressed.
"""

from __future__ import annotations

import re

import pytest

from finesub.subtitles.metrics import weighted_char_count
from finesub.llm.example_builder import (
    EXAMPLES_DIR,
    EXAMPLES_ZH_DIR,
    ExampleMaterialError,
    build_examples,
    load_examples,
    parse_example,
    render_example,
)
from finesub.llm.output_protocol import output_csv_header
from finesub.llm.prompt_constants import BASE, Thresholds
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.prompt_compose import compose_correction_system

STANDARD = Thresholds.at(1.0)

_MINIMAL = """---
id: demo
kind: mini
applies: [capableC]
headers: true
output_headers: true
teach: 测试用
---
引子 {thr:hard_chars}：

<input>
a | 10.0 | 1.0 | 一行
b | 12.0 | 2.0 | 二行
</input>

<output>
sub | a | 一行 | 第一句 | high | 跨度 {calc:span:a,b}
sub | b | 二行 | 第二句 | high |
</output>

尾注引用 {ref:b}。
"""


def test_derived_numbers_are_computed_not_transcribed() -> None:
    """Mechanism 2: a number the prose cites cannot be wrong, because nobody types it."""

    material = _MINIMAL.replace(
        "跨度 {calc:span:a,b}", "跨度 {calc:span:a,b} 间隔 {calc:gap:b} 字数 {calc:cc:a,b}"
    )
    rendered = render_example(
        parse_example(material, source="demo.md"), variant="capableC", with_start=False, thresholds=STANDARD
    )
    # gap = start - (previous start + duration) = 12.0 - 11.0
    assert "间隔 1.0" in rendered
    # the char count goes through weighted_char_count, summed over the rows starting at each label
    assert f"字数 {weighted_char_count('第一句') + weighted_char_count('第二句'):g}" in rendered
    inputs = [line.split("|") for line in rendered.splitlines() if line[:2] in ("1|", "2|")]
    # the input's gap cell: the first row has no row before it, so empty rather than invented
    assert [row[3] for row in inputs] == ["", "1.0"]


def test_local_ids_are_assigned_and_references_resolved() -> None:
    """Mechanism 3 / class B: material has no line numbers to go stale."""

    example = parse_example(_MINIMAL, source="demo.md")
    rendered = render_example(
        example, variant="capableC", with_start=False, thresholds=STANDARD
    )
    assert "尾注引用 2。" in rendered
    assert "{ref:" not in rendered and "{calc:" not in rendered
    assert "{thr:" not in rendered
    # The teaching aside is computed from the rows it talks about.
    assert "跨度 4.0" in rendered


def test_unknown_placeholders_fail_loudly() -> None:
    broken = _MINIMAL.replace("{ref:b}", "{ref:nope}")
    with pytest.raises(ExampleMaterialError, match="nope"):
        render_example(
            parse_example(broken, source="demo.md"),
            variant="capableC",
            with_start=False,
            thresholds=STANDARD,
        )
    broken_thr = _MINIMAL.replace("{thr:hard_chars}", "{thr:nope}")
    with pytest.raises(ExampleMaterialError, match="nope"):
        render_example(
            parse_example(broken_thr, source="demo.md"),
            variant="capableC",
            with_start=False,
            thresholds=STANDARD,
        )


def test_a_gap_on_the_blocks_first_row_fails_loudly() -> None:
    broken = _MINIMAL.replace("{calc:span:a,b}", "{calc:gap:a}")
    with pytest.raises(ExampleMaterialError, match="first row"):
        render_example(parse_example(broken, source="demo.md"), variant="capableC", with_start=False,
                       thresholds=STANDARD)


def test_malformed_material_is_rejected() -> None:
    with pytest.raises(ExampleMaterialError, match="frontmatter"):
        parse_example("no frontmatter here", source="demo.md")
    short_row = _MINIMAL.replace("a | 10.0 | 1.0 | 一行", "a | 10.0 | 一行")
    with pytest.raises(ExampleMaterialError, match="input row"):
        parse_example(short_row, source="demo.md")
    bad_number = _MINIMAL.replace("a | 10.0 | 1.0", "a | ten | 1.0")
    with pytest.raises(ExampleMaterialError, match="not a number"):
        parse_example(bad_number, source="demo.md")


def test_column_layout_follows_the_csv_header_constants() -> None:
    """Mechanism 4 / class E: no string surgery inserts the start column."""

    example = parse_example(_MINIMAL, source="demo.md")
    plain = render_example(
        example, variant="capableC", with_start=False, thresholds=STANDARD
    )
    with_start = render_example(
        example, variant="capableC", with_start=True, thresholds=STANDARD
    )
    assert output_csv_header() in plain
    assert output_csv_header(start=True) in with_start
    start_row = [
        line for line in with_start.splitlines() if line.startswith("sub|")
    ][0]
    assert start_row.split("|")[2] == "10.0"  # the row's own start, not a splice


def test_rendering_is_deterministic() -> None:
    """Mechanism 6: the fingerprint and replay both assume this."""

    first = build_examples(variant="capableC", with_start=False, thresholds=STANDARD)
    second = build_examples(variant="capableC", with_start=False, thresholds=STANDARD)
    assert first == second
    assert first.strip()


@pytest.mark.parametrize("directory", (EXAMPLES_DIR, EXAMPLES_ZH_DIR), ids=("examples", "examples_zh"))
def test_material_lint(directory) -> None:
    """Mechanism 7: labels unique, timeline monotonic, applies names known."""

    known_variants = {"capableC", "basicA"}
    ids = set()
    for example in load_examples(directory):
        assert example.id not in ids, f"duplicate example id {example.id}"
        ids.add(example.id)
        assert example.kind in {"main", "mini", "bad-output"}
        assert set(example.applies) <= known_variants, example.id
        for block in example.blocks:
            labels = [row.label for row in block.inputs]
            assert len(labels) == len(set(labels)), example.id
            starts = [row.start for row in block.inputs]
            assert starts == sorted(starts), f"{example.id}: timeline not monotonic"
            for row in block.inputs:
                assert row.duration > 0, example.id
            for row in block.outputs:
                for label in row.labels:
                    assert label in labels, f"{example.id}: unknown label {label}"


@pytest.mark.parametrize("correct_only", (False, True), ids=("translate", "correct-only"))
def test_built_examples_satisfy_the_output_contract(correct_only: bool) -> None:
    """Mechanism 1 / class F: the examples go through the real validator -- a correct-only task's Chinese ones
    as its two columns (`polished` may be empty on a line that is only filler).

    Nothing used to feed the worked examples to ``output_protocol``, so "the example
    itself violates the output contract" was undetectable.
    """

    from finesub.llm.chunking import SubtitleSegment
    from finesub.llm.output_protocol import validate_translated_csv_text

    from finesub.llm.prompt_variants import resolve_variant

    for name, grouped, scale in [(name, grouped, scale) for name, grouped in _SETS for scale in TIERS]:
        thresholds = Thresholds.at(scale)
        variant = resolve_variant(f"{name}-polish" if correct_only else name)
        rendered = build_examples(
            variant=name, with_start=variant.output_has_start,
            thresholds=thresholds, grouped=grouped, polished=correct_only,
            directory=EXAMPLES_ZH_DIR if correct_only else EXAMPLES_DIR,
        )
        pairs = re.findall(
            r"(?ms)^<asr_result>\n(.*?)^</asr_result>.*?^<translated>\n(.*?)^</translated>$",
            rendered,
        )
        assert pairs, "no example blocks were built"
        for asr_block, translated in pairs:
            sources = []
            for line in asr_block.splitlines():
                if not line.strip() or line.startswith("local_id|"):
                    continue
                local_id, start, duration, _gap, text = line.split("|", 4)
                text = text.split("|")[-1]  # past the `turn` cell, where the input has one
                sources.append(
                    SubtitleSegment(
                        local_id, float(start), float(start) + float(duration), text
                    )
                )
            if any(row.startswith("sub|") and row.count(",") >= 4 for row in
                   translated.splitlines()):
                continue  # kind: bad-output -- it exists to violate the contract
            result = validate_translated_csv_text(
                f"<translated>\n{translated}</translated>",
                sources,
                require_headers=translated.lstrip().startswith("type|"),
                require_start_column=variant.output_has_start,
                forbid_start_column=not variant.output_has_start,
                polished=correct_only,
                long_merge_limits=thresholds if variant.rejects_long_merges else None,
            )
            assert not result.errors, (name, grouped, scale, result.errors)


@pytest.mark.parametrize("name, grouped", (("capableC-polish", False), ("capableC-polish", True), ("basicA-polish", False)))
def test_a_polish_sets_examples_are_chinese_and_show_the_local_edits(name: str, grouped: bool) -> None:
    """A Chinese source's set reads the Chinese examples (`EXAMPLES_ZH_DIR`): headed `polished`, no kana, the
    line that is only filler kept with its `polished` cell empty, and `polished` showing the local edits (filler
    out, a stutter compressed, a measure word fixed) while most lines leave it equal to the corrected text."""

    from finesub.llm.output_protocol import correction_output_header
    from finesub.llm.prompt_variants import resolve_variant

    variant = resolve_variant(name)
    rendered = build_examples(
        variant=variant.examples_name, with_start=variant.output_has_start,
        polished=True, thresholds=Thresholds.at(1.0), grouped=grouped, directory=EXAMPLES_ZH_DIR,
    )
    header = correction_output_header(variant)
    assert header in rendered and "|polished|" in header
    assert not re.search(r"[\u3040-\u30ff]", rendered)
    width = header.count("|") + 1
    text = header.split("|").index("corrected_text")
    rows = {row[text]: row[text + 1] for row in (line.split("|") for line in rendered.splitlines())
            if row[0] == "sub" and len(row) == width}
    assert rows["呃"] == ""
    assert rows["就是那个那个连点喂它吃东西的动作"] == "就是那个连点喂它吃东西的动作"
    assert rows["这个动作配一个歌，我能循环看一下午"] == "这个动作配一首歌，我能循环看一下午"
    assert sum(corrected == polished for corrected, polished in rows.items()) > len(rows) / 2


def test_thresholds_appear_once_in_python() -> None:
    """Mechanism 4 / class D: the numbers live in prompt_constants only."""

    import finesub.llm.prompt_variants as variants_module
    from finesub.llm.prompt_tables import PHRASES_FILE, PROMPT_TEMPLATE_DIR
    from pathlib import Path

    # The variant clauses live in phrases_v1.toml (`[variant.*]`); prompt_variants builds the sets from them.
    source = Path(variants_module.__file__).read_text(encoding="utf-8") + (
        PROMPT_TEMPLATE_DIR / PHRASES_FILE
    ).read_text(encoding="utf-8")
    for name in ("hard_chars", "absolute_chars"):
        value = str(BASE[name])
        # The variant clauses reference the constants through placeholders now;
        # a bare literal here means a threshold got a second home.
        assert f"{value} 字" not in source, f"literal {value} in the variant clauses"

    # The threshold-bearing fragments reference ${thr_*} rather than literals
    # (the constants-table promise held for prompt_variants and the example
    # material but not for these two files until the third review round).
    template_dir = Path(variants_module.__file__).parent / "prompt_templates"
    for fragment in (
        "fragment_merge_rules_grouped_v1.md",
        "fragment_output_contract_grouped_v1.md",
    ):
        text = (template_dir / fragment).read_text(encoding="utf-8")
        assert "${thr_" in text, f"{fragment} lost its threshold placeholders"
        for phrase in (">20", ">36", "20 字", "36 字", "≤16", "≤3 字"):
            assert phrase not in text, f"literal threshold {phrase!r} in {fragment}"
    # The sandwich's bounds follow the length ratio, so the grouped set's clause on them is a template too.
    assert "${thr_sandwich_max_seconds}" in source and "${thr_sandwich_max_chars}" in source


def test_migrated_minis_reach_the_prompts() -> None:
    profile = resolve_profile("audio", "local", "quality")
    for variant, grouped in (("capableC", False), ("capableC", True)):
        system = compose_correction_system(profile, variant=variant, grouped=grouped, length_scale=1.0)
        assert "跨说话人反例" in system
        # The filler sandwich teaches the one three-source shape; capableC merges up to four lines and
        # does not see it (`single_pass_skip`), three passes' pass 2 under the same name does.
        assert ("filler 三明治三源特例" in system) is (variant != "capableC" or grouped), (variant, grouped)
        # And nothing unresolved leaked through the seam.
        assert "{ref:" not in system and "{thr:" not in system and "{calc:" not in system


def test_requires_selects_on_the_switch_vector_not_the_variant_name() -> None:
    """Material applies by switch predicate (docs/llm_prompts.md)."""

    from finesub.llm.example_builder import parse_example, profile_satisfies

    material = _MINIMAL.replace("teach: 测试用", "requires: media>=audio\nteach: 测试用")
    example = parse_example(material, source="demo.md")

    # media is a ladder, so "needs audio" is satisfied by video.
    assert profile_satisfies(example, resolve_profile("video", "local", "quality"))
    assert profile_satisfies(example, resolve_profile("audio", "local", "quality"))
    assert not profile_satisfies(example, resolve_profile("text", "local", "quality"))

    exact = parse_example(
        _MINIMAL.replace("teach: 测试用", "requires: retrieval=local\nteach: 测试用"),
        source="demo.md",
    )
    assert profile_satisfies(exact, resolve_profile("text", "local", "quality"))
    assert not profile_satisfies(exact, resolve_profile("text", "none", "quality"))

    # No profile in hand: the variant filter still applies, the switch one does not.
    assert profile_satisfies(example, None)


def test_malformed_requires_is_rejected() -> None:
    from finesub.llm.example_builder import parse_example, profile_satisfies

    unknown_axis = parse_example(
        _MINIMAL.replace("teach: 测试用", "requires: nope=x\nteach: 测试用"),
        source="demo.md",
    )
    with pytest.raises(ExampleMaterialError, match="unknown switch axis"):
        profile_satisfies(unknown_axis, resolve_profile("text", "none", "quality"))

    ladder_on_wrong_axis = parse_example(
        _MINIMAL.replace("teach: 测试用", "requires: retrieval>=local\nteach: 测试用"),
        source="demo.md",
    )
    with pytest.raises(ExampleMaterialError, match="media ladder"):
        profile_satisfies(ladder_on_wrong_axis, resolve_profile("text", "none", "quality"))


def test_examples_obey_the_injection_reality_principle() -> None:
    """An example may not cite an input its vector never receives (docs/llm_prompts.md).

    This is the same rule the slot registry enforces for fragments, applied to
    the generated example layer.
    """

    from finesub.llm.prompt_compose import PromptContext

    # (marker, the injected block it presupposes)
    MARKERS = {
        "搜索结果": "search_results",
        "<search_results>": "search_results",
        "重听": "audio",
        "画面": "video",
        "<entry_details>": "entry_details",
    }
    for switches in (
        ("text", "none", "efficiency"),
        ("text", "none", "quality"),
        ("text", "local", "quality"),
        ("audio", "none", "quality"),
        ("audio", "local", "quality"),
        ("video", "local", "quality"),
    ):
        profile = resolve_profile(*switches)
        injected = PromptContext(profile=profile).injected_blocks
        for variant in ("capableC", "basicA"):
            rendered = build_examples(
                variant=variant,
                with_start=variant == "basicA",
                profile=profile,
                params={
                    "noisy_span_handling": "（占位）",
                    "output_column_count": "9",
                },
                thresholds=STANDARD,
            )
            for marker, block in MARKERS.items():
                if marker in rendered:
                    assert block in injected, (
                        f"{switches}/{variant}: an example says {marker!r}, "
                        f"but this vector never receives {block}"
                    )


def test_placeholder_residue_is_an_error_not_a_leak() -> None:
    """Plan §5.3: a *malformed* placeholder must fail assembly, not ship.

    ``{calc span}`` and ``{thr_hard_chars}`` do not match the resolver's
    pattern, so before this guard they passed straight into the prompt.
    """

    from finesub.llm.example_builder import parse_example

    for broken in ("{calc span a}", "{thr_hard_chars}", "{ ref b }"):
        material = _MINIMAL.replace("{ref:b}", broken)
        with pytest.raises(ExampleMaterialError, match="residue"):
            render_example(
                parse_example(material, source="demo.md"),
                variant="capableC",
                with_start=False,
                thresholds=STANDARD,
            )


def test_window_overlap_pointer_has_the_oneshot_it_points_at() -> None:
    """Plan §6 class two: the one cross-fragment reference to oneshot content.

    ``fragment_window_overlap_v1`` tells the model "see the preceding_context
    block in the oneshot above, lines -1 and 0". That pointer dangles silently
    if a combination ever renders without the oneshot or renumbers its
    read-only context, so it is checked rather than reviewed.
    """

    for switches in (
        ("text", "none", "efficiency"),
        ("text", "local", "quality"),
        ("audio", "local", "quality"),
        ("video", "local", "quality"),
    ):
        for variant in ("capableC", "basicA"):
            system = compose_correction_system(
                resolve_profile(*switches), variant=variant, length_scale=1.0
            )
            if "完整示例见上方 oneshot" not in system:
                continue
            blocks = re.findall(
                r"(?ms)^<preceding_context>\n(.*?)^</preceding_context>$", system
            )
            assert blocks, f"{switches}/{variant}: the pointer has no oneshot to point at"
            ids = [
                line.split("|", 1)[0]
                for line in blocks[0].splitlines()
                if re.match(r"^-?\d+\|", line)
            ]
            assert ids == ["-1", "0"], (
                f"{switches}/{variant}: the pointer names -1 and 0 but the "
                f"oneshot renders {ids}"
            )


def test_missing_overlay_group_is_an_error_not_an_empty_block() -> None:
    """A variant whose overlay group is absent must fail, not ship a blank example.

    The fallback used to emit ``<translated></translated>`` while the
    surrounding 对照要点 still described a 43-row worked example.
    """

    from finesub.llm.example_builder import EXAMPLES_DIR, parse_example

    material = (EXAMPLES_DIR / "00-main-oneshot.md").read_text(encoding="utf-8")
    start = material.index("<output variant=basic ")
    without_basic = material[:start] + material[material.index("<notes variant=grouped>"):]

    with pytest.raises(ExampleMaterialError, match="no output overlay for group"):
        render_example(
            parse_example(without_basic, source="x.md"),
            variant="basicA",
            with_start=True,
            thresholds=STANDARD,
        )
    # The intact material still renders for that variant.
    assert render_example(
        parse_example(material, source="x.md"),
        variant="basicA",
        with_start=True,
        thresholds=STANDARD,
    ).strip()


def test_plain_output_block_inside_an_overlay_file_is_not_a_missing_group() -> None:
    """The missing-overlay check must not fire on a block with its own outputs.

    An overlay file may also carry a plain <input>/<output> pair (a
    variant-independent mini alongside the shared scene); only a block that
    *needs* an overlay and finds none for the group is an error.
    """

    material = """---
id: demo-mixed
kind: mini
applies: [capableC, basicA]
teach: 测试用
---
<input>
a | 10.0 | 1.0 | 一行
</input>

<output>
sub | a | 一行 | 第一句 | high |
</output>

<input>
c | 20.0 | 1.0 | 三行
</input>

<output variant=nosingles>
sub | c | 三行 | 第三句 | high |
</output>
"""
    rendered = render_example(
        parse_example(material, source="mixed.md"),
        variant="capableC",
        with_start=False,
        thresholds=STANDARD,
    )
    assert "第一句" in rendered and "第三句" in rendered
    # A variant whose group is really absent still fails.
    with pytest.raises(ExampleMaterialError, match="no output overlay for group"):
        render_example(
            parse_example(material, source="mixed.md"),
            variant="basicA",
            with_start=True,
            thresholds=STANDARD,
        )


def test_prose_between_input_and_output_renders_before_the_output() -> None:
    """The bad-output example's 「错误输出：」 labels the wrong output, so it sits between the two blocks.

    Prose between blocks used to be folded into the trailer: the label rendered after the wrong output with an
    empty line under it. Prose anywhere else between blocks has no place to go and is a material error.
    """

    examples = {example.id: example for example in load_examples()}
    bad = examples["noise-burst-bad-output"]
    for variant in bad.applies:
        for grouped in (False, True):
            rendered = render_example(bad, variant=variant, with_start=False, grouped=grouped, thresholds=STANDARD)
            if not rendered:
                continue
            assert "</asr_result>\n错误输出：\n<translated>\n" in rendered, (variant, grouped)
            assert "</translated>\n错在：" in rendered, (variant, grouped)

    second_pair = (
        "\n\n中间一句\n\n<input>\nc | 20.0 | 1.0 | 三行\n</input>\n\n"
        "<output>\nsub | c | 三行 | 第三句 | high |\n</output>"
    )
    misplaced = _MINIMAL.replace("</output>", "</output>" + second_pair)
    with pytest.raises(ExampleMaterialError, match="no place to render"):
        parse_example(misplaced, source="demo.md")


def _example_blocks(name: str, grouped: bool, thresholds: Thresholds):
    """(sources' spans by local id, the translated rows split into cells by column name) per example block."""

    from finesub.llm.output_protocol import output_row_spec
    from finesub.llm.prompt_variants import resolve_variant

    variant = resolve_variant(name)
    columns = output_row_spec(start=variant.output_has_start)
    rendered = build_examples(variant=name, with_start=variant.output_has_start,
                              thresholds=thresholds, grouped=grouped)
    blocks = re.findall(r"(?ms)^<asr_result>\n(.*?)^</asr_result>.*?^<translated>\n(.*?)^</translated>$", rendered)
    assert blocks
    for asr_block, translated in blocks:
        spans = {}
        for line in asr_block.splitlines():
            if not line.strip() or line.startswith("local_id|"):
                continue
            local_id, start, duration, _rest = line.split("|", 3)
            spans[local_id] = (float(start), float(start) + float(duration))
        rows = [dict(zip(columns, row.split("|", len(columns) - 1))) for row in translated.splitlines()
                if row.startswith("sub|")]
        yield spans, rows


#: The three tiers' ratios (`docs/llm_harness_behavior.md`「合并门槛与任务的长度比例」), and the sets whose examples they render in.
TIERS = (0.8, 1.0, 1.25)
_SETS = (("capableC", False), ("capableC", True), ("basicA", False))


@pytest.mark.parametrize("scale", TIERS)
@pytest.mark.parametrize("name, grouped", _SETS)
def test_every_merged_example_row_is_within_its_tiers_usual_length(name: str, grouped: bool, scale: float) -> None:
    """The examples are written once and shown at every tier: their characters do not scale, their durations
    do. So a merged row is within the tier's usual length (hard chars and seconds) at all three tiers, or its
    note says why it runs long; never past the absolute threshold, and a three-source row of three passes'
    pass 2 (the filler sandwich) within the sandwich's bounds. Cold read round 2, P172, is why a note that
    calls an over-long merge "在界内" fails. A voided row (`<void>`) is a merge taken back; the bad-output
    example's over-cap merge exists to break the rules."""

    thresholds = Thresholds.at(scale)
    for spans, rows in _example_blocks(name, grouped, thresholds):
        if any(len(row["position"].split(",")) > 4 for row in rows):
            continue  # kind: bad-output
        for row in rows:
            ids = row["position"].split(",")
            if len(ids) < 2 or "<void>" in row["note"]:
                continue
            span = round(max(spans[i][1] for i in ids) - min(spans[i][0] for i in ids), 1)
            chars = weighted_char_count(row["translation"])
            where = (name, grouped, scale, row["position"], span, chars)
            assert span <= thresholds.absolute_seconds and chars <= thresholds.absolute_chars, where
            if grouped and len(ids) == 3:
                assert span <= thresholds.sandwich_max_seconds and chars <= thresholds.sandwich_max_chars, where
            if span > thresholds.hard_seconds or chars > thresholds.hard_chars:
                assert "长" in row["note"] and "界内" not in row["note"], where


@pytest.mark.parametrize("scale", TIERS)
def test_the_main_examples_refusals_hold_at_every_tier(scale: float) -> None:
    """The main example refuses three merges for their length, and its notes say so at every tier: s08's four
    lines run 「偏长」 (past the tier's hard seconds), s31 with s32 「越过绝对门槛」 (past its absolute seconds), s33
    with s34 too many characters (past its hard chars). Characters do not scale and durations do, so each refusal
    has to hold at the short and the long tier alike."""

    from finesub.llm.example_builder import _scaled_example

    thresholds = Thresholds.at(scale)
    main = next(example for example in load_examples() if example.id.startswith("main"))
    scaled = _scaled_example(main, scale)
    block = scaled.blocks[0]
    rows = {row.label: row for row in block.inputs}
    assert rows["s11"].end - rows["s08"].start > thresholds.hard_seconds
    assert rows["s32"].end - rows["s31"].start > thresholds.absolute_seconds
    translations = {}
    for overlay in scaled.overlays.values():
        for overlay_block in overlay:
            for row in overlay_block.rows:
                if row.labels in (("s33",), ("s34",)):
                    translations.setdefault(row.labels[0], set()).add(row.translation)
    assert set(translations) == {"s33", "s34"}
    for first in translations["s33"]:
        for second in translations["s34"]:
            assert weighted_char_count(first) + weighted_char_count(second) > thresholds.hard_chars


def test_the_pause_into_the_window_is_kept_at_every_tier() -> None:
    """The preceding context and the window's rows are one timeline: the pause from the last preceding row to the
    window's first row does not scale, as no pause does."""

    from finesub.llm.example_builder import _scaled_example

    main = next(example for example in load_examples() if example.preceding)
    gap = round(main.blocks[0].inputs[0].start - main.preceding[-1].end, 1)
    for scale in TIERS:
        scaled = _scaled_example(main, scale)
        assert round(scaled.blocks[0].inputs[0].start - scaled.preceding[-1].end, 1) == gap, scale
        # anchored on the window's first row: it starts where it does at every tier, the context moves instead
        assert scaled.blocks[0].inputs[0].start == main.blocks[0].inputs[0].start, scale


def test_the_context_sits_at_one_place_whichever_block_follows_it() -> None:
    """The preceding context is rendered once: it is anchored on its own last row's end, not on a block, so two
    blocks opening at different times each keep their own pause from it; a block that would open before the
    context's last row is a material error."""

    from finesub.llm.example_builder import Example, ExampleBlock, InputRow, _scaled_example

    def example(*starts: float) -> Example:
        return Example(id="t", kind="main", applies=(), requires_switches=(), teach="",
                       preceding=(InputRow("p1", -6.0, 2.0, "x"), InputRow("p2", -3.5, 3.0, "x")),
                       blocks=tuple(ExampleBlock(inputs=(InputRow(f"b{i}", start, 2.0, "y"),), outputs=())
                                    for i, start in enumerate(starts)))

    for scale in TIERS:
        scaled = _scaled_example(example(0.0, 1.2), scale)
        assert scaled.preceding[-1].end == -0.5, scale
        assert [block.inputs[0].start for block in scaled.blocks] == [0.0, 1.2], scale
    # p2 lasts 3.0 s, 3.8 s at 1.25 and still ends at -0.5: it opens at -4.3, and b0 at -4.5 opens before it
    with pytest.raises(ExampleMaterialError, match="would start before the preceding 'p2'"):
        _scaled_example(example(0.0, -4.5), 1.25)


def test_scaled_rows_that_would_fall_out_of_order_are_a_material_error() -> None:
    from finesub.llm.example_builder import InputRow, scaled_rows

    # b opens 1.0 s into a's 4.0 s: at 0.5 a ends at 2.0, so b would open at -1.0
    rows = (InputRow("a", 0.0, 4.0, "x"), InputRow("b", 1.0, 1.0, "y"))
    with pytest.raises(ExampleMaterialError, match="would start before"):
        scaled_rows(rows, 0.5)


def test_the_example_durations_follow_the_ratio_and_the_pauses_do_not() -> None:
    from finesub.llm.example_builder import InputRow, scaled_rows

    rows = (InputRow("a", 1.4, 4.6, "x"), InputRow("b", 6.3, 2.0, "y"), InputRow("c", 8.5, 1.0, "z"))
    assert scaled_rows(rows, 1.0) == rows
    short = scaled_rows(rows, 0.8)
    # 4.6 x 0.8 = 3.68 -> 3.7; 2.0 x 0.8 = 1.6; the pauses 0.3 and 0.2 stay
    assert [(row.start, row.duration) for row in short] == [(1.4, 3.7), (5.4, 1.6), (7.2, 0.8)]
    long = scaled_rows(rows, 1.25)
    # 4.6 x 1.25 = 5.75 -> 5.8 (half up), 2.0 -> 2.5, 1.0 -> 1.25 -> 1.3
    assert [(row.start, row.duration) for row in long] == [(1.4, 5.8), (7.5, 2.5), (10.2, 1.3)]


@pytest.mark.parametrize("directory", (EXAMPLES_DIR, EXAMPLES_ZH_DIR), ids=("examples", "examples_zh"))
def test_the_examples_state_no_number_by_hand(directory) -> None:
    """A judgment in the examples that cites a time or a merged length cites a computed one (`{calc:}`) or a
    threshold (`{thr:}`): the durations follow the ratio and the thresholds with them, so a typed number would
    be right at one tier only. The input rows (their numbers are the data) are left out."""

    for path in sorted(directory.glob("*.md")):
        text = re.sub(r"(?s)<(input|preceding)>.*?</\1>", "", path.read_text(encoding="utf-8"))
        text = re.sub(r"\{(?:ref|thr|calc):[^}]*\}", "", text)
        typed = re.findall(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:s\b|秒)|约\s*\d+(?:\.\d+)?\s*字", text)
        assert not typed, (path.name, typed)


def test_a_chinese_example_shows_its_headers_where_the_japanese_one_does():
    """Cold read P456: the Chinese set mirrors the Japanese one file by file; a front-matter switch set on one
    side only renders the two sets differently (the correct-only minis lost their header rows)."""

    def switch(path):
        return "\noutput_headers: true\n" in path.read_text(encoding="utf-8").split("\n---", 1)[0]

    for zh in sorted(EXAMPLES_ZH_DIR.glob("*.md")):
        ja = EXAMPLES_DIR / zh.name
        if ja.is_file():
            assert switch(zh) == switch(ja), zh.name


def test_every_clause_table_belongs_to_a_registered_set():
    """`phrases_v1.toml`'s `[variant.<name>]` tables are the registered base sets' clauses, one each: a table
    left behind by a deleted set would be read by nothing and say nothing."""

    from finesub.llm import prompt_tables, prompt_variants

    registered = set(prompt_variants.VARIANTS) | {variant.name for variant in prompt_variants._GROUPED.values()}
    registered = {name for name in registered if not name.endswith(prompt_variants.CORRECT_ONLY_SUFFIX)}
    assert set(prompt_tables.prompt_phrases("variant")) == registered
