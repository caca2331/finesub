"""Builder-layer guarantees from docs/llm_prompts.md.

Each test here corresponds to a row of that plan's mechanism table: the point
is not that the current examples happen to be right, but that a class of
inconsistency can no longer be expressed.
"""

from __future__ import annotations

import re

import pytest

from finesub.subtitles.metrics import weighted_char_count
from finesub.llm.output_protocol import OUTPUT_CSV_HEADER, OUTPUT_CSV_HEADER_WITH_START
from finesub.llm.example_builder import (
    ExampleMaterialError,
    build_examples,
    load_examples,
    parse_example,
    render_example,
)
from finesub.llm.prompt_constants import THRESHOLDS
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.prompt_compose import compose_correction_system

_MINIMAL = """---
id: demo
kind: mini
applies: [capableB]
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
sub | a | AUTO | AUTO | 一行 | 第一句 | high | AUTO | 跨度 {calc:span:a,b}
sub | b | AUTO | AUTO | 二行 | 第二句 | high | AUTO |
</output>

尾注引用 {ref:b}。
"""


def test_derived_columns_are_computed_not_transcribed() -> None:
    """Mechanism 2: AUTO columns cannot be wrong, because nobody types them."""

    example = parse_example(_MINIMAL, source="demo.md")
    rendered = render_example(
        example, variant="capableB", with_start=False, with_comments=False
    )
    rows = [
        line for line in rendered.splitlines() if line.startswith("sub|")
    ]
    # gap = next.start - (start + duration) = 12.0 - 11.0
    assert rows[0].split("|")[3] == "1.0"
    # duration comes from the source row, char_count from weighted_char_count
    assert rows[0].split("|")[2] == "1.0"
    assert rows[0].split("|")[7] == str(int(weighted_char_count("第一句")))
    # the last row has no successor, so its gap is 0 rather than invented
    assert rows[1].split("|")[3] == "0.0"


def test_local_ids_are_assigned_and_references_resolved() -> None:
    """Mechanism 3 / class B: material has no line numbers to go stale."""

    example = parse_example(_MINIMAL, source="demo.md")
    rendered = render_example(
        example, variant="capableB", with_start=False, with_comments=False
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
            variant="capableB",
            with_start=False,
            with_comments=False,
        )
    broken_thr = _MINIMAL.replace("{thr:hard_chars}", "{thr:nope}")
    with pytest.raises(ExampleMaterialError, match="nope"):
        render_example(
            parse_example(broken_thr, source="demo.md"),
            variant="capableB",
            with_start=False,
            with_comments=False,
        )


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
        example, variant="capableB", with_start=False, with_comments=False
    )
    with_start = render_example(
        example, variant="capableB", with_start=True, with_comments=False
    )
    assert OUTPUT_CSV_HEADER in plain
    assert OUTPUT_CSV_HEADER_WITH_START in with_start
    start_row = [
        line for line in with_start.splitlines() if line.startswith("sub|")
    ][0]
    assert start_row.split("|")[2] == "10.0"  # the row's own start, not a splice


def test_rendering_is_deterministic() -> None:
    """Mechanism 6: the fingerprint and replay both assume this."""

    first = build_examples(variant="capableB", with_start=False, with_comments=False)
    second = build_examples(variant="capableB", with_start=False, with_comments=False)
    assert first == second
    assert first.strip()


def test_material_lint(tmp_path) -> None:
    """Mechanism 7: labels unique, timeline monotonic, applies names known."""

    known_variants = {"capableB", "capableC", "basicA", "basicB"}
    ids = set()
    for example in load_examples():
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


def test_built_examples_satisfy_the_output_contract() -> None:
    """Mechanism 1 / class F: the examples go through the real validator.

    Nothing used to feed the worked examples to ``output_protocol``, so "the example
    itself violates the output contract" was undetectable.
    """

    from finesub.llm.chunking import SubtitleSegment
    from finesub.llm.output_protocol import validate_translated_csv_text

    rendered = build_examples(
        variant="capableB", with_start=False, with_comments=False
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
            sources.append(
                SubtitleSegment(
                    local_id, float(start), float(start) + float(duration), text
                )
            )
        if any(row.startswith("sub|") and row.count(",") >= 3 for row in
               translated.splitlines()):
            continue  # kind: bad-output -- it exists to violate the contract
        result = validate_translated_csv_text(
            f"<translated>\n{translated}</translated>",
            sources,
            require_singles=False,
            require_headers=translated.lstrip().startswith("type|"),
        )
        assert not result.errors, result.errors


def test_thresholds_appear_once_in_python() -> None:
    """Mechanism 4 / class D: the numbers live in prompt_constants only."""

    import finesub.llm.prompt_variants as variants_module
    from pathlib import Path

    source = Path(variants_module.__file__).read_text(encoding="utf-8")
    for name in ("hard_chars", "absolute_chars"):
        value = str(THRESHOLDS[name])
        # The variant clauses reference the constants through placeholders now;
        # a bare literal here means a threshold got a second home.
        assert f"{value} 字" not in source, f"literal {value} in prompt_variants"

    # The threshold-bearing fragments reference ${thr_*} rather than literals
    # (the constants-table promise held for prompt_variants and the example
    # material but not for these two files until the third review round).
    template_dir = Path(variants_module.__file__).parent / "prompt_templates"
    for fragment in (
        "fragment_merge_rules_nosingles_v1.md",
        "fragment_output_contract_nosingles_reasoning_v1.md",
    ):
        text = (template_dir / fragment).read_text(encoding="utf-8")
        assert "${thr_" in text, f"{fragment} lost its threshold placeholders"
        for phrase in (">20", ">36", "20 字", "36 字", "≤16", "≤3 字"):
            assert phrase not in text, f"literal threshold {phrase!r} in {fragment}"


def test_migrated_minis_reach_the_capable_b_prompt() -> None:
    system = compose_correction_system(
        resolve_profile("audio", "local", "quality"), variant="capableB"
    )
    assert "filler 三明治三源正例" in system
    assert "跨说话人反例" in system
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
        for variant in ("capableB", "capableC", "basicA", "basicB"):
            rendered = build_examples(
                variant=variant,
                with_start=variant in ("basicA", "basicB"),
                with_comments=variant == "capableC",
                profile=profile,
                params={
                    "noisy_span_handling": "（占位）",
                    "merge_connect_basis": "（占位）",
                    "output_column_count": "9",
                },
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
                variant="capableB",
                with_start=False,
                with_comments=False,
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
        for variant in ("capableB", "capableC", "basicA", "basicB"):
            system = compose_correction_system(
                resolve_profile(*switches), variant=variant
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
    start = material.index("<output variant=basic tag=singles")
    without_basic = material[:start] + material[material.index("<notes variant=nosingles>"):]

    with pytest.raises(ExampleMaterialError, match="no output overlay for group"):
        render_example(
            parse_example(without_basic, source="x.md"),
            variant="basicA",
            with_start=True,
            with_comments=False,
        )
    # The intact material still renders for that variant.
    assert render_example(
        parse_example(material, source="x.md"),
        variant="basicA",
        with_start=True,
        with_comments=False,
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
applies: [capableB, basicA]
teach: 测试用
---
<input>
a | 10.0 | 1.0 | 一行
</input>

<output>
sub | a | AUTO | AUTO | 一行 | 第一句 | high | AUTO |
</output>

<input>
c | 20.0 | 1.0 | 三行
</input>

<output variant=nosingles>
sub | c | AUTO | AUTO | 三行 | 第三句 | high | AUTO |
</output>
"""
    rendered = render_example(
        parse_example(material, source="mixed.md"),
        variant="capableB",
        with_start=False,
        with_comments=False,
    )
    assert "第一句" in rendered and "第三句" in rendered
    # A variant whose group is really absent still fails.
    with pytest.raises(ExampleMaterialError, match="no output overlay for group"):
        render_example(
            parse_example(material, source="mixed.md"),
            variant="basicA",
            with_start=True,
            with_comments=False,
        )
