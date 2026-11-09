"""Fast-mode planning: the fused whole-input window and the enable decision.

The fused *round* (research + query in one call) left with the task-flow
rework (2026-09); fast mode is now one whole-input window
task, so only the planner and its gates remain under test here.
"""

from __future__ import annotations
import finesub.llm.correction_translation  # noqa: F401

import json
from pathlib import Path

import pytest

from finesub.llm.chunking import SubtitleSegment
from finesub.llm.correction_translation import _fast_execute_kwargs
from finesub.llm.routing.profiles import resolve_profile, window_output_budget
from finesub.llm.stages.plan import FAST_WINDOW_CHUNK_ID, FastDecision, decide_fast_mode, plan_fast_window


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


class HugePromptCounter(FakeTokenCounter):
    """Prompt-level counts blow the input budget; per-text counts stay tiny."""

    def count_texts(self, texts) -> int:
        list(texts)
        return 300_000


def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "一。"),
        SubtitleSegment("2", 100.0, 102.5, "二。"),
    ]


def _stable_json(tmp_path):
    path = tmp_path / "clip-stable.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 100.0, "end": 102.5, "text": "二。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path



def test_plan_fast_window_covers_everything_with_edge_pads() -> None:
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("audio", "local", "quality")
    )

    assert window.chunk_id == FAST_WINDOW_CHUNK_ID
    assert [segment.id for segment in window.segments] == ["1", "2"]
    assert window.overlap_segments == []
    assert window.boundary_reason == "fast_single_window"
    # Both edges get the 60s global pad; start clamps to 0.
    assert window.clip_start == 0.0
    assert window.clip_end == 162.5

    clamped = plan_fast_window(
        _segments(),
        audio_duration=120.0,
        counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
    )
    assert clamped.clip_end == 120.0




def test_decide_fast_mode_with_knowledge_off_never_reads_the_base(tmp_path, monkeypatch) -> None:
    """The fast gate must not grow with a knowledge base the run will not see."""

    import finesub.llm.stages.plan as plan_module

    monkeypatch.setattr(
        plan_module,
        "load_index_texts",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("knowledge base was read with knowledge off")
        ),
    )
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        knowledge_enabled=False,
        token_counter=FakeTokenCounter(),
    )
    assert decision.enabled


def test_decide_fast_mode_auto_enables_small_input(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
    )

    assert decision.enabled
    assert decision.reason == "fits fast budgets"
    assert decision.window is not None
    assert decision.output_budget == window_output_budget(fast=True)
    assert 0 < decision.expected_output_tokens <= decision.output_budget
    assert 0 < decision.opening_input_tokens <= decision.input_budget
    assert decision.to_metadata()["enabled"] is True


def test_decide_fast_mode_off_and_bad_mode(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="off",
        token_counter=FakeTokenCounter(),
    )
    assert not decision.enabled
    assert decision.window is None

    with pytest.raises(ValueError, match="auto/on/off"):
        decide_fast_mode(
            stable_json=_stable_json(tmp_path),
            fast="fastest",
            token_counter=FakeTokenCounter(),
        )


def test_decide_fast_mode_auto_falls_back_when_output_over_budget(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality", output_scale=10_000),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
    )

    assert not decision.enabled
    assert "expected output" in decision.reason


def test_decide_fast_mode_on_raises_when_over_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="--fast on"):
        decide_fast_mode(
            stable_json=_stable_json(tmp_path),
            fast="on",
            profile=resolve_profile("audio", "local", "quality", output_scale=10_000),
            knowledge_root=tmp_path / "kb",
            token_counter=FakeTokenCounter(),
        )


def test_decide_fast_mode_auto_falls_back_when_over_subtitle_cap(tmp_path) -> None:
    """The fast window is the whole input, so it is the most exposed to the
    quality guardrail the multi-window planner enforces."""
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
        max_window_subtitle_tokens=1,
    )

    assert not decision.enabled
    assert "max_window_subtitle_tokens" in decision.reason
    meta = decision.to_metadata()
    assert meta["max_window_subtitle_tokens"] == 1
    assert meta["subtitle_input_tokens"] > 1


def test_decide_fast_mode_subtitle_cap_zero_disables_the_gate(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
        max_window_subtitle_tokens=0,
    )

    assert decision.enabled
    assert decision.to_metadata()["max_window_subtitle_tokens"] == 0


def test_decide_fast_mode_on_raises_when_over_subtitle_cap(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_window_subtitle_tokens"):
        decide_fast_mode(
            stable_json=_stable_json(tmp_path),
            fast="on",
            profile=resolve_profile("audio", "local", "quality"),
            knowledge_root=tmp_path / "kb",
            token_counter=FakeTokenCounter(),
            max_window_subtitle_tokens=1,
        )


def test_decide_fast_mode_checks_opening_input_reserve(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        token_counter=HugePromptCounter(),
    )

    assert not decision.enabled
    assert "opening input" in decision.reason




def test_the_gate_weighs_the_style_catalog_it_will_send(tmp_path) -> None:
    """`--style auto` puts the candidate list in the fused window's OPENING
    message (it picks in its own request turn), not in the fill the 56k reserve
    covers. A gate that measured the opening without it let a long catalog
    through and the real call hit the ceiling instead."""

    kwargs = dict(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("text", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
    )
    plain = decide_fast_mode(**kwargs)
    with_catalog = decide_fast_mode(
        **kwargs, style_catalog="- 甲组｜口味｜作者：甲组｜适用范围：主播A 的全部投稿"
    )

    assert with_catalog.opening_input_tokens > plain.opening_input_tokens
    assert with_catalog.input_budget == plain.input_budget  # the reserve is the fill's


def test_fast_gate_names_a_too_small_envelope_instead_of_a_negative_budget(
    tmp_path,
) -> None:
    """The fill reserve is an absolute token count calibrated against
    Gemini's 194k prompt limit. A bound group with a smaller envelope made
    ``prompt_limit - reserve`` negative, so every input "exceeded" a negative
    budget -- true, and unreadable. The gate now says the actual thing.

    ``--fast on`` still errors rather than downgrading silently, same as the
    budget gates it sits next to.
    """

    from dataclasses import replace as dc_replace

    import pytest

    from finesub.llm.routing.config import DEFAULT_LIMITS
    from finesub.llm.routing.profiles import FAST_FILL_INPUT_RESERVE_TOKENS
    from finesub.llm.stages.plan import decide_fast_mode

    small = dc_replace(DEFAULT_LIMITS, prompt_input_limit=23_000)
    # No stable JSON is read: the envelope decides before any input does.
    decision = decide_fast_mode(
        stable_json=tmp_path / "absent.json", fast="auto", limits=small
    )

    assert decision.enabled is False
    assert "23000" in decision.reason
    assert str(FAST_FILL_INPUT_RESERVE_TOKENS) in decision.reason
    assert "-" not in decision.reason.split("envelope")[1].split("<=")[0]

    with pytest.raises(ValueError, match="does not fit fast mode"):
        decide_fast_mode(
            stable_json=tmp_path / "absent.json", fast="on", limits=small
        )

    # off still short-circuits first.
    assert (
        decide_fast_mode(
            stable_json=tmp_path / "absent.json", fast="off", limits=small
        ).reason
        == "fast mode disabled"
    )


def test_every_fast_mode_call_site_passes_the_group_envelope() -> None:
    """A missing `limits=` here is invisible, and was wrong for a long time.

    `decide_fast_mode` defaults `limits` to `DEFAULT_LIMITS`, so a call site
    that forgets the keyword silently budgets the fused window against
    Gemini's 194000/65536 whatever group is bound. That is what `main()` did:
    the CLI path decided fast mode on numbers no candidate had declared, and
    the function's own "this group cannot hold a fused window" guard could
    never fire because it always saw 194000 (fixed 2026-09-04).

    A source guard rather than a behavioural one because the defect has no
    symptom to assert on -- both readings run, and the wrong one only shows up
    as windows sized for the wrong model.
    """

    import ast

    import finesub.llm.correction_translation as ct

    source = Path(ct.__file__)
    tree = ast.parse(source.read_text(encoding="utf-8"))
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "decide_fast_mode"
    ]
    assert sites, "the fast planner is still called from this module"
    missing = [
        node.lineno
        for node in sites
        if not any(keyword.arg == "limits" for keyword in node.keywords)
    ]
    assert not missing, (
        f"{source.name} calls decide_fast_mode without limits= at lines "
        f"{missing}; that silently plans against DEFAULT_LIMITS"
    )



def test_every_window_planning_call_site_passes_the_group_envelope() -> None:
    """The same invisible default for `plan_correction_windows`: a dry run that forgot `limits=` previewed windows
    cut against `DEFAULT_LIMITS` -- 10k subtitles whatever the correction group's rows say."""

    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "finesub"
    missing, seen = [], 0
    for path in src.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "plan_correction_windows":
                seen += 1
                if not any(keyword.arg == "limits" for keyword in node.keywords):
                    missing.append(f"{path.relative_to(src)}:{node.lineno}")
    assert seen >= 4 and not missing, missing



def test_fast_execute_kwargs_is_the_whole_input_window() -> None:
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("text", "local", "quality")
    )
    decision = FastDecision(mode="auto", enabled=True, reason="", window=window)
    assert _fast_execute_kwargs(decision) == {"windows_override": [window]}



def test_research_only_never_runs_under_fast_mode() -> None:
    """`--research-only` asks for the preparation task; fast mode has none, so
    `auto` resolves to `off` for that run and `on` is a stated conflict (it used
    to skip the preparation and fall through into the correction windows)."""
    import pytest

    from finesub.llm.correction_request import fast_mode_for_research_only

    assert fast_mode_for_research_only("auto", research_only=True) == "off"
    assert fast_mode_for_research_only("off", research_only=True) == "off"
    assert fast_mode_for_research_only("auto", research_only=False) == "auto"
    assert fast_mode_for_research_only("on", research_only=False) == "on"
    with pytest.raises(ValueError, match="research-only"):
        fast_mode_for_research_only("on", research_only=True)


def test_the_gate_weighs_the_index_a_none_window_will_request_from(tmp_path) -> None:
    """window-kb-request-channel plan: under `retrieval=none` a readable base
    opens the entries-only channel, so its index is in the opening message."""

    root = tmp_path / "kb"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text("- 主播A | エーちゃん | 测试主播\n", encoding="utf-8")
    (root / "streamer" / "主播A.md").write_text(
        "# 主播A\n资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    (root / "common" / "index.md").write_text("", encoding="utf-8")
    kwargs = dict(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("text", "none", "quality"),
        knowledge_root=root,
        token_counter=FakeTokenCounter(),
    )
    with_base = decide_fast_mode(**kwargs, knowledge_enabled=True)
    without = decide_fast_mode(**kwargs, knowledge_enabled=False)
    assert with_base.opening_input_tokens > without.opening_input_tokens
