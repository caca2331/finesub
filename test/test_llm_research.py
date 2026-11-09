from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.client import LLMCallResult
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.prompts import ContextPack
from finesub.llm.general_context import GeneralChunk, select_general_chunks, whole_chunk_ids
import finesub.llm.general_context as gc
from finesub.llm.research import (
    check_research_input_limit,
    parse_round1_output,
    parse_round2_output,
    plan_research_chunks,
    render_research_transcript,
    resolve_round1_entries,
)
from finesub.llm.token_budget import default_token_counter
from finesub.llm.web_search import QuerySearchResult


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))



def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "第一句。"),
        SubtitleSegment("2", 1.2, 2.0, "第二句还没完"),
        SubtitleSegment("3", 2.0, 3.0, "现在结束。"),
        SubtitleSegment("4", 3.2, 4.0, "第四句。"),
        SubtitleSegment("5", 4.2, 5.0, "第五句。"),
    ]


class FakeSearchClient:
    def __init__(self, results: list[QuerySearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[tuple[str, ...], int | None]] = []

    def search_many(self, queries, *, max_queries=None):
        requested = [getattr(item, "query", item) for item in queries]
        self.calls.append((tuple(requested), max_queries))
        if not self.results:
            return []
        # Like the real client, echo each request's query into its result
        # (priority decrement matches rendered labels against request queries).
        return [
            replace(self.results[index % len(self.results)], query=query)
            for index, query in enumerate(requested)
        ]

    def extract_many(self, requests, *, max_urls=None):
        return []


def test_render_research_transcript_marks_windows_without_duplicates() -> None:
    segments = _segments()
    windows = plan_correction_windows(
        segments,
        counter=FakeTokenCounter(),
    )
    transcript = render_research_transcript(segments, windows)

    for window in windows:
        assert f"--- window {window.chunk_id} ---" in transcript
    for segment in segments:
        assert transcript.count(f"{segment.id}|") == 1


def test_parse_round1_output_reads_tag_blocks() -> None:
    text = (
        "好的，以下是结果。\n"
        "<requested_entries>\n主播A\n- 游戏B\n</requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n1. 游戏B 剧情\n游戏B 角色名\n</search_queries>\n"
    )

    result = parse_round1_output(text)

    assert result.requested_entries == ("主播A", "游戏B")
    assert result.keep_entries == ()
    assert result.search_queries == ("游戏B 剧情", "游戏B 角色名")
    assert result.analysis_notes == ""


def test_resolve_round1_entries_caps_channels_and_merges_keep_first(
    tmp_path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text(
        "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
    )
    (knowledge_root / "common" / "index.md").write_text(
        "- 游戏B [游戏] | Game B | 测试游戏\n", encoding="utf-8"
    )
    (knowledge_root / "streamer" / "主播A.md").write_text(
        "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    (knowledge_root / "common" / "游戏B.md").write_text(
        "# 游戏B\n游戏资料。\n\n## 档案\n本名: 游戏B\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=["Game B", "不存在"],
        keep_names=["エーちゃん", "Game B"],
        visible_keep_keys=["主播A"],
        max_requested_entries=1,
        max_keep_entries=1,
        max_total_entries=1,
    )

    assert list(selected) == ["主播A"]  # keep channel wins shared cap
    assert missing == ["不存在"]
    assert ignored == ["Game B"]  # indexed, but not visible as a preinjection
    assert dropped == ["游戏B"]


def test_resolve_round1_entries_enforces_eight_each_and_twelve_total(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    streamer = knowledge_root / "streamer"
    common = knowledge_root / "common"
    streamer.mkdir(parents=True)
    common.mkdir(parents=True)
    kept_keys = [f"保留{i}" for i in range(10)]
    requested_keys = [f"请求{i}" for i in range(10)]
    (streamer / "index.md").write_text(
        "".join(f"- {key} | keep-{i} | 测试\n" for i, key in enumerate(kept_keys)),
        encoding="utf-8",
    )
    (common / "index.md").write_text(
        "".join(
            f"- {key} [游戏] | request-{i} | 测试\n"
            for i, key in enumerate(requested_keys)
        ),
        encoding="utf-8",
    )
    for key in [*kept_keys, *requested_keys]:
        target = streamer if key.startswith("保留") else common
        (target / f"{key}.md").write_text(
            f"# {key}\n内容。\n\n## 档案\n本名: {key}\n\n## 元数据\n最近更新日期: 2026-08-01\n",
            encoding="utf-8",
        )

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=requested_keys,
        keep_names=kept_keys,
        visible_keep_keys=kept_keys,
    )

    assert list(selected) == [*kept_keys[:8], *requested_keys[:4]]
    assert missing == []
    assert ignored == []
    assert dropped == [*kept_keys[8:], *requested_keys[4:]]


def test_resolve_round1_entries_does_not_report_cross_channel_duplicate_as_dropped(
    tmp_path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    streamer = knowledge_root / "streamer"
    common = knowledge_root / "common"
    streamer.mkdir(parents=True)
    common.mkdir(parents=True)
    (streamer / "index.md").write_text("- 主播A | エーちゃん | 测试\n", encoding="utf-8")
    (common / "index.md").write_text("", encoding="utf-8")
    (streamer / "主播A.md").write_text(
        "# 主播A\n内容。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=["エーちゃん"],
        keep_names=["主播A"],
        visible_keep_keys=["主播A"],
        max_requested_entries=0,
        max_keep_entries=1,
        max_total_entries=1,
    )

    assert list(selected) == ["主播A"]
    assert missing == []
    assert ignored == []
    assert dropped == []


def test_parse_round1_output_reads_optional_analysis_notes() -> None:
    text = (
        "<analysis_notes>\n主播在玩游戏B（待定）。\n</analysis_notes>\n"
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>"
    )

    result = parse_round1_output(text)

    assert result.analysis_notes == "主播在玩游戏B（待定）。"


def test_parse_round1_output_requires_both_blocks() -> None:
    with pytest.raises(ValueError, match="search_queries"):
        parse_round1_output(
            "<requested_entries>\n主播A\n</requested_entries>\n"
            "<keep_entries></keep_entries>"
        )


def test_parse_round1_output_accepts_empty_blocks() -> None:
    result = parse_round1_output(
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n</search_queries>"
    )

    assert result.requested_entries == ()
    assert result.search_queries == ()


def test_parse_round2_output_reads_context_pack_tag() -> None:
    payload = {
        "general_context": {"global_summary": "摘要"},
        "window_contexts": [{"window_ids": ["0001"], "context": "背景一"}],
    }
    text = f"<context_pack>\n{json.dumps(payload, ensure_ascii=False)}\n</context_pack>"

    pack = parse_round2_output(text)

    assert pack.general_context["global_summary"] == "摘要"
    assert pack.unbound_window_contexts == {"0001": "背景一"}


def _context_pack_reply(window_contexts: list) -> str:
    payload = {"general_context": {}, "window_contexts": window_contexts}
    return f"<context_pack>\n{json.dumps(payload, ensure_ascii=False)}\n</context_pack>"


def test_parse_round2_output_copies_a_shared_note_to_each_listed_window() -> None:
    pack = parse_round2_output(_context_pack_reply([
        {"window_ids": ["0001"], "context": "一"},
        {"window_ids": ["0001", "0002"], "context": "共享"},
        {"window_ids": ["0002", "0002"], "context": "共享"},
    ]))

    assert pack.unbound_window_contexts == {"0001": "一\n\n共享", "0002": "共享"}


def test_parse_round2_output_refuses_the_single_window_field() -> None:
    with pytest.raises(ValueError, match="window_ids"):
        parse_round2_output(_context_pack_reply([{"window_id": "0001", "context": "一"}]))
    with pytest.raises(ValueError, match="window_ids"):
        parse_round2_output(_context_pack_reply([{"window_ids": [], "context": "一"}]))


def test_a_shared_note_reads_once_in_a_window_spanning_both_copies() -> None:
    pack = ContextPack.from_dict({"window_contexts": [
        {"window_id": "0001", "first_source_id": "1", "last_source_id": "2", "context": "一\n\n共享"},
        {"window_id": "0002", "first_source_id": "3", "last_source_id": "4", "context": "二\n\n共享"},
    ]}).with_source_order(["1", "2", "3", "4"])

    assert pack.window_context_for_source_ids(["1", "2", "3", "4"]) == "一\n\n共享\n\n二"


def test_parse_round2_output_rejects_bare_json() -> None:
    payload = {"general_context": {"global_summary": "摘要"}, "window_contexts": []}

    with pytest.raises(ValueError, match="exactly one <context_pack>"):
        parse_round2_output(json.dumps(payload, ensure_ascii=False))


def test_check_research_input_limit_raises_on_overflow() -> None:
    messages = [{"role": "user", "content": "字" * 10_000}]

    with pytest.raises(ValueError, match="Split the audio"):
        check_research_input_limit(
            messages,
            round_name="round 1",
            limits=type("L", (), {"prompt_input_limit": 10})(),
            counter=FakeTokenCounter(),
        )


def _knowledge_root(tmp_path, *, with_index: bool):
    root = tmp_path / "knowledge"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    if with_index:
        (root / "streamer" / "index.md").write_text(
            "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
        )
        (root / "streamer" / "主播A.md").write_text(
            "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
        )
        (root / "common" / "index.md").write_text("", encoding="utf-8")
    return root


def test_a_note_for_a_window_outside_the_plan_warns_instead_of_crashing(tmp_path, monkeypatch) -> None:
    """The preparation task may name a window the plan does not have; the
    note is dropped with a warning and the run goes on (the warning call once
    passed a keyword the reporter does not take and killed the whole task)."""

    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.research import run_research_stage

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    responses = [
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>",
        '<context_pack>{"general_context": {}, "window_contexts": '
        '[{"window_ids":["9999"],"context":"背景"}]}</context_pack>',
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0), role=role, model="fake", fallback_used=False, raw_response={}
            )

    monkeypatch.setattr("finesub.llm.research.RoleClient", FakeClient)
    context_path = tmp_path / "clip-research-context.json"
    run_research_stage(
        stable_json=stable_json,
        context_path=context_path,
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
    )

    payload = json.loads(context_path.read_text(encoding="utf-8"))
    assert payload["window_context_bind_report"]["unplaceable_window_ids"] == ["9999"]


def test_run_research_stage_writes_planning_metadata(tmp_path, monkeypatch) -> None:
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.prompt_compose import PROMPT_VERSION
    from finesub.llm.routing.config import WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    from finesub.llm.research import run_research_stage

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)

    responses = [
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>",
        '<context_pack>{"general_context": {}, "window_contexts": '
        '[{"window_ids":["0001"],"context":"背景"}]}</context_pack>',
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    monkeypatch.setattr("finesub.llm.research.RoleClient", FakeClient)

    context_path = tmp_path / "clip-research-context.json"
    run_research_stage(
        stable_json=stable_json,
        context_path=context_path,
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
    )

    payload = json.loads(context_path.read_text(encoding="utf-8"))
    planning = payload["planning"]
    assert planning["prompt_version"] == PROMPT_VERSION
    assert planning["context_reserve_tokens"] == WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    assert planning["profile_id"] == (
        "correction_media=audio,retrieval=local,"
        "difficulty=quality,continuity=serial"
    )
    assert planning["output_scale"] == 1.0
    assert planning["stable_json_hash"].startswith("sha256:")
    assert planning["extra_info_hash"].startswith("sha256:")
    assert planning["knowledge_inputs_hash"].startswith("sha256:")
    assert planning["knowledge_enabled"] is True
    assert planning["collect_task_feedback"] is False
    assert payload["context_pack"]["window_contexts"] == [
        {
            "window_id": "0001",
            "first_source_id": "1",
            "last_source_id": "1",
            "context": "背景",
        }
    ]


def test_knowledge_tri_state_remains_in_research_audit_metadata(tmp_path) -> None:
    """Generation settings remain observable even though they do not gate reuse."""

    from finesub.llm.research import planning_metadata

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    knowledge_root = _knowledge_root(tmp_path, with_index=True)
    common = dict(
        stable_json=stable,
        knowledge_root=knowledge_root,
        extra_info="",
        max_window_subtitle_tokens=None,
    )
    off = planning_metadata(
        resolve_profile("text", "none", "quality"),
        knowledge_enabled=False,
        collect_task_feedback=False,
        **common,
    )
    on = planning_metadata(
        resolve_profile("text", "none", "quality"),
        knowledge_enabled=True,
        collect_task_feedback=True,
        **common,
    )

    assert off != on
    assert off["knowledge_enabled"] is False and on["knowledge_enabled"] is True
    assert off["knowledge_inputs_hash"] == ""
    assert on["knowledge_inputs_hash"].startswith("sha256:")


def test_difficulty_is_normalized_in_research_audit_metadata(tmp_path) -> None:
    """Difficulty is not a research input; other generation axes stay auditable."""

    from finesub.llm.research import planning_metadata, research_reuse_key

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    common = dict(
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )
    high = planning_metadata(resolve_profile("audio", "local", "quality"), **common)
    med = planning_metadata(resolve_profile("audio", "local", "intermediate"), **common)
    other_axis = planning_metadata(
        resolve_profile("text", "local", "quality"), **common
    )

    # The full vector stays recorded, so a mixed-difficulty run is auditable...
    assert high["profile_id"] != med["profile_id"]
    # ...but difficulty is not a reuse key: it picks the correction prompt
    # variant and the thinking knob, neither of which the research rounds read.
    assert research_reuse_key(high) == research_reuse_key(med)
    # Correction media changes geometry, not what research means.
    assert research_reuse_key(other_axis) == research_reuse_key(high)


def test_every_planning_field_is_classified_as_gate_or_audit(tmp_path) -> None:
    """L3's answer to `_task_fingerprint`'s include-list assertion.

    `research_reuse_key` picks named fields, so a field added to
    `planning_metadata` without a decision would be silently ignored -- an old
    context reused past something that should have invalidated it. Classifying
    is the whole point of the whitelist; forgetting must fail here, loudly,
    instead of in someone's subtitles.
    """

    from finesub.llm.research import (
        RESEARCH_REUSE_AUDIT_ONLY,
        RESEARCH_REUSE_GATES,
        planning_metadata,
        research_reuse_key,
    )

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    planning = planning_metadata(
        resolve_profile("audio", "local", "quality"),
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )

    classified = set(RESEARCH_REUSE_GATES) | set(RESEARCH_REUSE_AUDIT_ONLY)
    assert set(planning) == classified
    assert not set(RESEARCH_REUSE_GATES) & set(RESEARCH_REUSE_AUDIT_ONLY)
    assert set(research_reuse_key(planning)) == set(RESEARCH_REUSE_GATES)


def test_retrieval_mode_gates_research_reuse(tmp_path, monkeypatch) -> None:
    """`research_semantics_id` (= retrieval) is what separates a context whose
    preparation could fetch nothing from one that searched."""

    from finesub.llm.research import planning_metadata, research_reuse_key

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    common = dict(
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )
    none = planning_metadata(resolve_profile("audio", "none", "quality"), **common)
    native = planning_metadata(resolve_profile("audio", "native", "quality"), **common)
    local = planning_metadata(resolve_profile("audio", "local", "quality"), **common)

    assert research_reuse_key(none) != research_reuse_key(native)
    assert research_reuse_key(none) != research_reuse_key(local)


def test_native_sources_are_flagged_when_the_backend_reports_no_urls() -> None:
    """A `sources` list that nothing can check must not read like a citation.

    The native prompt asks the model to cite what it verified. On a backend
    that reports no result URLs the reply comes back with site-level "sources"
    the harness cannot corroborate -- and in the 2026-08-15 A/B the one
    checkable claim carried that way was a wrong official title.
    """

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "global_summary": "x",
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}],
        }
    )
    agy = [
        {
            "backend": "local_agent",
            "search_events": [{"tool": "search_web", "query": "q", "urls": []}],
        }
    ]

    marked = _mark_unverified_sources(pack, agy)

    assert marked.general_context["sources"][0]["verified"] is False
    assert "无法核对" in marked.general_context["sources_note"]


def test_native_sources_are_left_alone_when_urls_were_recorded() -> None:
    """Codex/Claude keep a per-call URL ledger, so there is something to check."""

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}]
        }
    )
    grounded = [
        {
            "backend": "local_agent",
            "search_events": [{"tool": "web_search", "urls": ["https://example.org/a"]}],
        }
    ]

    marked = _mark_unverified_sources(pack, grounded)

    assert marked is pack
    assert "verified" not in marked.general_context["sources"][0]
    assert "sources_note" not in marked.general_context


def test_marking_skips_rounds_that_are_not_context_packs() -> None:
    from finesub.llm.research import _mark_unverified_sources

    round1 = parse_round1_output(
        "<requested_entries></requested_entries>"
        "<keep_entries></keep_entries>"
        "<search_queries></search_queries>"
    )
    assert _mark_unverified_sources(round1, []) is round1


def test_marking_reaches_a_pack_wrapped_in_a_preparation_reply() -> None:
    """The preparation task parses into its own reply type, not a bare pack --
    and the mark is what stops a native agent's self-reported `verified=true`
    from reaching the windows unchallenged. A request turn carries no pack and
    is handed back untouched."""

    from finesub.llm.research import _mark_unverified_sources
    from finesub.llm.stages.prepare_task import PrepareReply

    agy = [{"backend": "local_agent", "search_events": [{"tool": "search_web", "urls": []}]}]
    reply = PrepareReply(
        raw="",
        context_pack=ContextPack(
            general_context={"sources": [{"title": "Some Wiki", "verified": True}]}
        ),
    )

    marked = _mark_unverified_sources(reply, agy)

    assert marked is not reply
    assert marked.context_pack.general_context["sources"][0]["verified"] is False
    assert "无法核对" in marked.context_pack.general_context["sources_note"]
    assert _mark_unverified_sources(PrepareReply(raw="x"), agy).context_pack is None


def test_gemini_native_sources_are_not_stamped_unverified() -> None:
    """The shipped default's native path is Gemini `google_search`, not an agent.

    Since the client parses `groundingMetadata`, this path exits at the URL
    check with something real to show. The backend list still has to hold: an
    older Gemini reply with no grounding row must not be stamped either --
    saying "the backend reported nothing" about a provider that does report is
    untrue. `test_llm_switch_matrix` asserts that route exists.
    """

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}]
        }
    )
    grounded = [
        {
            "backend": "gemini_rest",
            "search_events": [
                {
                    "tool": "google_search",
                    "queries": ["q"],
                    "urls": ["https://example.org/a"],
                }
            ],
        }
    ]
    assert _mark_unverified_sources(pack, grounded) is pack

    bare = [{"backend": "gemini_rest", "return_code": 200}]
    marked = _mark_unverified_sources(pack, bare)

    assert marked is pack
    assert "sources_note" not in marked.general_context


# --- chunked research for ultra-long material (plan W7 / P8) -----------------


def _many_windows():
    segments = [
        SubtitleSegment(str(n), float(n), float(n) + 0.8, f"第{n}句的内容文本。")
        for n in range(1, 13)
    ]
    windows = plan_correction_windows(
        segments,
        counter=FakeTokenCounter(),
        max_window_subtitle_tokens=30,
    )
    assert len(windows) >= 3, "fixture must yield several research windows"
    return segments, windows


def test_plan_research_chunks_packs_whole_windows() -> None:
    segments, windows = _many_windows()
    counter = FakeTokenCounter()
    chunks = plan_research_chunks(
        windows, count_tokens=counter.count_text, max_tokens=60
    )
    assert len(chunks) >= 2
    # Every window lands in exactly one chunk, order preserved, never split.
    flattened = [window.chunk_id for chunk in chunks for window in chunk]
    assert flattened == [window.chunk_id for window in windows]
    # 0 disables chunking outright.
    assert plan_research_chunks(
        windows, count_tokens=counter.count_text, max_tokens=0
    ) == [list(windows)]
    # A cap smaller than any single window still never splits one.
    tiny = plan_research_chunks(windows, count_tokens=counter.count_text, max_tokens=1)
    assert [len(chunk) for chunk in tiny] == [1] * len(windows)


def _spans(*sizes: int) -> list[tuple[str, int, int]]:
    """Chunks c01, c02... of `sizes` rows each, back to back."""
    spans, low = [], 0
    for number, size in enumerate(sizes, start=1):
        spans.append((f"c{number:02d}", low, low + size - 1))
        low += size
    return spans


def _measure(sizes: dict[str, int]):
    return lambda shown, own: sum(sizes[chunk_id] for chunk_id in shown)


def test_a_window_reads_its_own_chunk_then_the_nearest_that_fit() -> None:
    spans = _spans(10, 10, 10, 10, 10)
    sizes = dict.fromkeys(("c01", "c02", "c03", "c04", "c05"), 5)
    picked = select_general_chunks(spans, ["c03"], measure=_measure(sizes), max_tokens=15)
    # c02 and c04 are both one row off: the earlier goes first
    assert picked.order == ("c03", "c02", "c04") and picked.shown == ("c02", "c03", "c04")
    assert picked.dropped == ("c01", "c05") and not picked.own_over_cap
    # the first that does not fit stops the adding: no smaller one further off slips in
    sizes = {"c01": 1, "c02": 50, "c03": 5, "c04": 1, "c05": 1}
    picked = select_general_chunks(spans, ["c01"], measure=_measure(sizes), max_tokens=20)
    assert picked.order == ("c01",) and picked.dropped == ("c02", "c03", "c04", "c05")


def test_an_own_chunk_past_the_ceiling_is_still_read() -> None:
    picked = select_general_chunks(_spans(10, 10), ["c02"], measure=_measure({"c01": 1, "c02": 99}),
                                   max_tokens=20)
    assert picked.order == ("c02",) and picked.own_over_cap and picked.dropped == ("c01",)


def test_a_window_across_two_chunks_owns_both() -> None:
    pack = ContextPack(
        general_context={"c01": {"global_summary": "一"}, "c02": {"global_summary": "二"}},
        general_chunks=(GeneralChunk("c01", "1", "2"), GeneralChunk("c02", "3", "4")),
    ).with_source_order(["1", "2", "3", "4"])
    assert pack.general_selection(["2", "3"]).own == ("c01", "c02")
    assert pack.general_selection(["3"]).own == ("c02",)
    # a row the source order leaves out (the judge keeps empty and zero-length rows) does not unplace the window
    assert pack.general_selection(["3", "3b"]).own == ("c02",) and not pack.general_selection(["3", "3b"]).fallback
    assert "本窗口位于 c01、c02 段" in pack.general_text_for_source_ids(["2", "3"])


def test_the_ceiling_counts_the_text_as_sent(monkeypatch) -> None:
    """Indented, without the judge's `speakers`, with the note and the own-chunk sentence."""
    from finesub.llm import general_context as gc

    blocks = {"c01": {"global_summary": "甲" * 50}, "c02": {"global_summary": "乙", "speakers": [{"name": "x" * 4000}]}}
    pack = ContextPack(general_context=blocks, general_chunks=(
        GeneralChunk("c01", "1", "1"), GeneralChunk("c02", "2", "2"))).with_source_order(["1", "2"])
    text = pack.general_text_for_source_ids(["1"])
    assert text.index(gc.NOTE_KEY) < text.index('"c01"') < text.index('"c02"') < text.index(gc.OWN_CHUNK_KEY)
    assert "speakers" not in text and text.startswith('{\n  "')
    monkeypatch.setattr(gc, "GENERAL_MAX_TOKENS", default_token_counter().count_text(text))
    assert ContextPack.from_dict(pack.to_dict()).with_source_order(["1", "2"]).general_selection(["1"]).shown \
        == ("c01", "c02"), "the judge's list never pushes a chunk out of the correction's context"
    monkeypatch.setattr(gc, "GENERAL_MAX_TOKENS", default_token_counter().count_text(text) - 1)
    again = ContextPack.from_dict(pack.to_dict()).with_source_order(["1", "2"])
    assert again.general_selection(["1"]).dropped == ("c02",)


def test_whole_view_keeps_the_leading_chunks() -> None:
    assert whole_chunk_ids(["c01", "c02", "c03"], measure=lambda ids: 10 * len(ids), max_tokens=25) == ("c01", "c02")
    assert whole_chunk_ids(["c01", "c02"], measure=lambda ids: 99, max_tokens=1) == ("c01",)


def test_an_unplaceable_window_reads_the_whole_view_with_a_warning(monkeypatch) -> None:
    import finesub.llm.prompts as prompts_mod

    warnings = []
    monkeypatch.setattr(prompts_mod, "current_reporter", lambda: SimpleNamespace(
        warning=lambda code, *args, **kwargs: warnings.append(code)))
    pack = ContextPack(
        general_context={"c01": {"global_summary": "一"}, "c02": {"global_summary": "二"}},
        general_chunks=(GeneralChunk("c01", "1", "2"), GeneralChunk("c02", "3", "gone")),
    ).with_source_order(["1", "2", "3", "4"])
    selection = pack.general_selection(["1"])
    assert selection.fallback and selection.shown == ("c01", "c02")
    assert pack.general_text_for_source_ids(["1"]) == pack.general_text_whole()
    assert warnings == ["research-general-unplaced"], "warned once per pack"


def test_the_chunked_shape_survives_every_rebuild() -> None:
    chunks = (GeneralChunk("c01", "1", "2"), GeneralChunk("c02", "3", "4"))
    pack = ContextPack(general_context={"c01": {"a": 1}, "c02": {"b": 2}}, general_chunks=chunks)
    windows = plan_correction_windows(_segments(), max_window_subtitle_tokens=10_000)
    for rebuilt in (pack.with_source_order(["1"]), ContextPack.from_dict(pack.to_dict()),
                    ContextPack.from_dict(pack.to_dict()).bind_window_ranges(windows)):
        assert rebuilt.general_chunks == chunks
    single = ContextPack(general_context={"global_summary": "s"})
    assert "general_chunks" not in single.to_dict()
    assert single.general_blocks() == single.general_blocks(["1"]) == [{"global_summary": "s"}]
    ordered = pack.with_source_order(["1", "2", "3", "4"])
    assert ordered.general_blocks() == [{"a": 1}, {"b": 2}]
    assert ordered.general_blocks(["4"]) == [{"b": 2}, {"a": 1}], "own chunk first"


def test_the_users_requests_live_in_their_own_field_and_reach_every_view(monkeypatch) -> None:
    from finesub.llm import general_context as gc

    single = ContextPack.from_dict({"general_context": {"global_summary": "s", "user_requests": ["人名用片假名"]}})
    assert single.user_requests == ("人名用片假名",) and "user_requests" not in single.general_context
    assert ContextPack.from_dict(single.to_dict()).user_requests == single.user_requests
    # a string where a list belongs is not split into one request per character
    assert ContextPack.from_dict({"general_context": {}, "user_requests": "人名"}).user_requests == ()
    assert '"user_requests": [\n    "人名用片假名"' in single.general_text_for_source_ids(["1"])
    big = "x" * 4000
    chunked = ContextPack(
        general_context={"c01": {"global_summary": "一"}, "c02": {"global_summary": "二"}},
        general_chunks=(GeneralChunk("c01", "1", "1"), GeneralChunk("c02", "2", "2")),
        user_requests=(big,),
    ).with_source_order(["1", "2"])
    # outside the ceiling: a long request pushes no chunk out
    monkeypatch.setattr(gc, "GENERAL_MAX_TOKENS", default_token_counter().count_text(
        gc.render_chunked_general(chunked.general_context, ("c01", "c02"), ("c01",))))
    assert chunked.general_selection(["1"]).shown == ("c01", "c02")
    for text in (chunked.general_text_for_source_ids(["1"]), chunked.general_text_whole(),
                 chunked.with_source_order(["9"]).general_text_for_source_ids(["9"])):
        assert big in text
    windows = plan_correction_windows(_segments(), max_window_subtitle_tokens=10_000)
    for rebuilt in (ContextPack.from_dict(chunked.to_dict()), chunked.bind_window_ranges(windows)):
        assert rebuilt.user_requests == (big,)


def test_the_old_merged_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="old merged shape"):
        ContextPack.from_dict({"general_context": {"合并说明": "m", "c01": {"a": 1}}})


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("", 50_000),  # unset: the planning limits' cap (owner 2026-11-08)
        ("research_transcript_max_tokens = 0\n", 0),
        ("research_transcript_max_tokens = 100000\n", 100_000),
    ],
)
def test_research_transcript_cap_reads_config(tmp_path, monkeypatch, body, expected) -> None:
    from finesub import config as app_config
    from finesub.llm.research import research_transcript_cap
    from finesub.llm.routing.profiles import DEFAULT_PROFILE

    path = tmp_path / "config.toml"
    path.write_text("[chunking]\n" + body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()

    assert research_transcript_cap(DEFAULT_PROFILE) == expected
    assert research_transcript_cap(replace(DEFAULT_PROFILE, retrieval="native")) == 0


@pytest.mark.parametrize(
    "value",
    ["-1", "true", "2.5", '"50000"'],
    ids=["negative", "bool", "float", "string"],
)
def test_a_malformed_chunking_cap_is_an_error_not_a_guess(tmp_path, monkeypatch, value) -> None:
    """`int(raw)` used to turn `true` into a 1-token cap, `2.5` into 2 and
    `"50000"` into 50000, all silently. Each is now an error naming the key."""

    from finesub import config as app_config
    from finesub import settings

    path = tmp_path / "config.toml"
    path.write_text(
        f"[chunking]\nresearch_transcript_max_tokens = {value}\n"
        f"max_window_subtitle_tokens = {value}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    app_config.clear_config_cache()

    with pytest.raises(ValueError, match="research_transcript_max_tokens"):
        settings.RESEARCH_TRANSCRIPT_MAX_TOKENS.resolve()
    with pytest.raises(ValueError, match="max_window_subtitle_tokens"):
        settings.MAX_WINDOW_SUBTITLE_TOKENS.resolve()


def test_chunked_research_merges_packs_and_replays_finished_chunks(
    tmp_path, monkeypatch
) -> None:
    """Plan W7: every chunk runs the full rounds over its own slice, notes
    bind to disjoint source intervals, keep entries union in chunk order, and
    a rerun replays finished chunks off their per-chunk files."""

    import re

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import DEFAULT_PROFILE

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:3]]
    calls: list[str] = []

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        calls.append(chunk_label)
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        return {
            "context_pack": {
                "general_context": {"专名": f"来自{chunk_label}",
                                    "user_requests": ["共有要求", f"要求{chunk_label}"]},
                "window_contexts": [
                    {"window_id": window_id, "context": f"note-{chunk_label}"}
                ],
            },
            "keep_entries": ["共有词条", f"词条{chunk_label}"],
            "rounds": {"round1": True, "round2": True},
            "injected_entries": [f"词条{chunk_label}"],
            "token_report": {"totals": {"total_tokens": 10}},
        }

    kwargs = dict(
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=DEFAULT_PROFILE,
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        research_runner=fake_run_research,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
    )
    from finesub.llm import general_context as gc

    warnings = []
    monkeypatch.setattr(research_mod, "current_reporter", lambda: SimpleNamespace(
        warning=lambda code, message, **kwargs: warnings.append((code, message)), debug=lambda *args: None))
    # room for two of the three blocks: the middle chunk's tie goes to the earlier neighbour
    blocks = {f"c0{n}": {"专名": f"来自c0{n}"} for n in (1, 2, 3)}
    monkeypatch.setattr(gc, "GENERAL_MAX_TOKENS", default_token_counter().count_text(
        gc.render_chunked_general(blocks, ("c01", "c02"), ("c01",))))
    payload = research_mod._run_chunked_research(chunk_plan, **kwargs)
    dropped = [message for code, message in warnings if code == "research-general-chunks-dropped"]
    assert [message.split(" 的通用背景")[0] for message in dropped] == [
        "准备任务分块 c01 的窗口放不下 c03", "准备任务分块 c02 的窗口放不下 c03", "准备任务分块 c03 的窗口放不下 c01"]
    monkeypatch.undo()

    assert calls == ["c01", "c02", "c03"]
    pack = ContextPack.from_dict(payload["context_pack"])
    assert len(pack.window_contexts) == 3
    assert all(note.first_source_id and note.last_source_id for note in pack.window_contexts)
    # every chunk's block kept by chunk id, with the source ids its windows cover
    assert list(pack.general_context) == ["c01", "c02", "c03"]
    assert pack.user_requests == ("共有要求", "要求c01", "要求c02", "要求c03"), "union, by text"
    bodies = [[segment.id for segment in window.segments[len(window.overlap_segments):]] for window in windows[:3]]
    assert [(c.chunk_id, c.first_source_id, c.last_source_id) for c in pack.general_chunks] == [
        (f"c0{number}", body[0], body[-1]) for number, body in enumerate(bodies, start=1)]
    # Union in chunk order, deduplicated.
    assert payload["keep_entries"][:2] == ["共有词条", "词条c01"]
    assert len(payload["keep_entries"]) == len(set(payload["keep_entries"]))
    for record in payload["research_chunks"]:
        assert (tmp_path / record["file"]).exists()
        assert record["replayed"] is False
    assert payload["token_report"]["totals"]["total_tokens"] == 30

    # A rerun must replay every finished chunk without a single LLM call.
    def exploding_run_research(**_kwargs):
        raise AssertionError("a finished chunk was re-run")

    replayed = research_mod._run_chunked_research(
        chunk_plan, **{**kwargs, "research_runner": exploding_run_research}
    )
    assert [record["replayed"] for record in replayed["research_chunks"]] == [True] * 3
    assert replayed["keep_entries"] == payload["keep_entries"]
    assert replayed["context_pack"] == payload["context_pack"]


def _vote(key: str) -> dict:
    return {"intent": "auto", "outcome": "selected" if key else "none", "key": key, "reason": "r", "rev": 3}


def _chunked_with_votes(tmp_path, votes, *, knowledge_root=None, catalog=None, seen=None, context_name="research-context.json"):
    import finesub.llm.research as research_mod
    from finesub.llm.knowledge.style import StyleCatalog
    from finesub.llm.routing.profiles import DEFAULT_PROFILE

    segments, windows = _many_windows()
    order = iter(votes)

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        if seen is not None:
            seen.append({"chunk_label": chunk_label, **kwargs})
        return {
            "context_pack": {"general_context": {}, "window_contexts": []},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {"total_tokens": 1}},
            "style_selection": _vote(next(order)),
        }

    catalog = catalog or StyleCatalog.from_keys(["甲组", "乙组", "default_style"], rev=3)
    return research_mod._run_chunked_research(
        [[window] for window in windows[: len(votes)]],
        segments=segments,
        context_file=tmp_path / context_name,
        extra_info="",
        knowledge_root=knowledge_root if knowledge_root is not None else tmp_path,
        knowledge_enabled=False,
        profile=DEFAULT_PROFILE,
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        research_runner=fake_run_research,
        research_runner_kwargs={"style_catalog": catalog, "extra_style": "口语一些"},
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
    )


def test_chunked_research_picks_the_style_by_votes(tmp_path) -> None:
    """auto-style-gaps plan B (owner 2026-10-01): every chunk sees the catalog
    and votes with its own forced pick; the most votes win, and the reason
    lists the tally."""

    seen: list[dict] = []
    payload = _chunked_with_votes(tmp_path, ["甲组", "乙组", "甲组"], seen=seen)

    assert seen and all(kwargs["style_catalog"] for kwargs in seen)
    assert all(kwargs["extra_style"] == "口语一些" for kwargs in seen)
    choice = payload["style_selection"]
    assert choice["outcome"] == "selected" and choice["key"] == "甲组" and choice["rev"] == 3
    assert "甲组 2 票（c01、c03）" in choice["reason"] and "乙组 1 票（c02）" in choice["reason"]


def test_a_chunk_saved_without_a_vote_runs_again(tmp_path) -> None:
    """A chunk finished before chunks voted (or under a run without auto) has
    no vote; replaying it would lose one, so that chunk -- only that one --
    runs again. Chunks that carry a vote replay."""

    seen: list[dict] = []
    _chunked_with_votes(tmp_path, ["甲组", "乙组"], seen=seen)
    stale = tmp_path / "research-context-c01.json"
    saved = json.loads(stale.read_text(encoding="utf-8"))
    del saved["style_selection"]
    stale.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    seen.clear()
    payload = _chunked_with_votes(tmp_path, ["乙组"], seen=seen)
    assert [kwargs["chunk_label"] for kwargs in seen] == ["c01"]
    assert payload["style_selection"]["key"] == "乙组"  # c01 now votes 乙组, c02 replays 乙组



def test_a_chunk_this_code_cannot_read_runs_again(tmp_path) -> None:
    """The prompt version no longer gates a chunk, so a saved chunk in a shape `ContextPack` refuses (here the
    pre-interval id->text notes) is redone -- that chunk only -- instead of failing the stage on every resume."""

    seen: list[dict] = []
    _chunked_with_votes(tmp_path, ["甲组", "乙组"], seen=seen)
    stale = tmp_path / "research-context-c01.json"
    saved = json.loads(stale.read_text(encoding="utf-8"))
    saved["context_pack"]["window_contexts"] = {"w001": "旧形状"}
    stale.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")

    seen.clear()
    _chunked_with_votes(tmp_path, ["乙组"], seen=seen)
    assert [kwargs["chunk_label"] for kwargs in seen] == ["c01"]

def test_a_tie_breaks_by_usage_then_revision_then_name() -> None:
    from finesub.llm.knowledge.style import StyleCatalog, StyleChoice, StyleVoteFacts, tally_style_votes

    catalog = StyleCatalog.from_keys(["甲组", "乙组", "丙组"], rev=9)

    def votes(*keys):
        return [(f"c{i + 1:02d}", StyleChoice(outcome="selected", key=k, rev=9)) for i, k in enumerate(keys)]

    by_usage = tally_style_votes(
        votes("甲组", "乙组"), catalog, StyleVoteFacts(usage={"甲组": 1, "乙组": 4}, last_rev={"甲组": 8, "乙组": 2})
    )
    assert by_usage.key == "乙组" and "使用率：乙组 4、甲组 1" in by_usage.reason

    by_rev = tally_style_votes(
        votes("甲组", "乙组"), catalog, StyleVoteFacts(usage={"甲组": 2, "乙组": 2}, last_rev={"甲组": 8, "乙组": 2})
    )
    assert by_rev.key == "甲组" and "最近修订（rev）：乙组 2、甲组 8" in by_rev.reason

    unreadable = tally_style_votes(
        votes("乙组", "甲组"), catalog, StyleVoteFacts(usage=None, last_rev={"甲组": 5, "乙组": 5})
    )
    assert unreadable.key == "乙组" and "使用率读不到，跳过" in unreadable.reason  # 乙 < 甲 by code point
    assert "按名字字典序" in unreadable.reason
    # Deterministic: the same votes and facts, the same answer, in any order.
    again = tally_style_votes(
        votes("甲组", "乙组"), catalog, StyleVoteFacts(usage=None, last_rev={"甲组": 5, "乙组": 5})
    )
    assert again.key == unreadable.key


def test_a_style_beats_none_on_a_tie() -> None:
    from finesub.llm.knowledge.style import StyleCatalog, StyleChoice, StyleVoteFacts, tally_style_votes

    catalog = StyleCatalog.from_keys(["甲组"], rev=1)  # no default_style: "none" is a legal vote
    choice = tally_style_votes(
        [("c01", StyleChoice(outcome="none", rev=1)), ("c02", StyleChoice(outcome="selected", key="甲组", rev=1))],
        catalog,
        StyleVoteFacts(usage=None),
    )
    assert choice.outcome == "selected" and choice.key == "甲组"
    assert "与「不用风格」打平时选风格" in choice.reason
    none_wins = tally_style_votes(
        [("c01", StyleChoice(outcome="none", rev=1)), ("c02", StyleChoice(outcome="none", rev=1)),
         ("c03", StyleChoice(outcome="selected", key="甲组", rev=1))],
        catalog,
        StyleVoteFacts(usage=None),
    )
    assert none_wins.outcome == "none" and none_wins.key == ""


def test_the_tie_break_reads_usage_and_revision_off_the_store(tmp_path) -> None:
    """The facts come from the store as of the catalog's revision (a later
    edit does not change a replay's answer) and from the `style_injected`
    usage ledger; both are written into the reason."""

    from finesub.llm.knowledge.maintain import main as maintain_main
    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.knowledge.node.signals import book_style_injected
    from finesub.llm.knowledge.style import load_style_catalog

    root = tmp_path / "kb"
    maintain_main(["--root", str(root), "new", "style", "甲组", "--intro", "甲"])
    maintain_main(["--root", str(root), "new", "style", "乙组", "--intro", "乙"])
    KnowledgeRepo.forget(root)
    repo = KnowledgeRepo.open(root)
    for task in ("t1", "t2"):
        book_style_injected(repo, ["style/乙组"], task_id=task)
    book_style_injected(repo, ["style/甲组"], task_id="t1")
    catalog = load_style_catalog(root)

    payload = _chunked_with_votes(tmp_path, ["甲组", "乙组"], knowledge_root=root, catalog=catalog)
    assert payload["style_selection"]["key"] == "乙组"
    assert "使用率：乙组 2、甲组 1" in payload["style_selection"]["reason"]


def _named_chunks_run(tmp_path, monkeypatch, deliveries, *, resume=True):
    """Run three one-window chunks whose packs are ``deliveries[i]`` (general context dicts); returns the
    payload, what each chunk was handed, and the warnings."""

    import re

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:len(deliveries)]]
    handed: dict[str, str] = {}
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(research_mod, "current_reporter", lambda: SimpleNamespace(
        warning=lambda code, message, **kwargs: warnings.append((code, message)), debug=lambda *a, **k: None))

    def fake_run_research(*, transcript, chunk_label, earlier_chunks="", **kwargs):
        handed[chunk_label] = earlier_chunks
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        return {
            "context_pack": {
                "general_context": deliveries[int(chunk_label[1:]) - 1],
                "window_contexts": [{"window_id": window_id, "context": f"note-{chunk_label}"}],
            },
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    payload = research_mod._run_chunked_research(
        chunk_plan,
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=resolve_profile(continuity="parallel"),
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        research_runner=fake_run_research,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=resume,
        client=object(),
        transcript_cap=1000,
    )
    return payload, handed, warnings


def test_later_chunks_get_the_earlier_names_and_the_first_name_wins(tmp_path, monkeypatch) -> None:
    """Chunks run one after another, also under `continuity=parallel`; from the second on each is handed the
    names and summaries before it; the merged pack keeps one first-wins name table, and the disagreements --
    said in `name_disputes`, or re-listed with another spelling -- reach the user, not the windows."""

    first = {"global_summary": "前段", "entities": [{"name": "ヒロ", "zh": "希罗", "type": "人物", "note": "主角"}],
             "asr_risks": [{"heard_as": "成れ果て", "should_be": "成れの果て", "reason": "r"}]}
    second = {"global_summary": "中段",
              "entities": [{"name": "ヒロ", "zh": "广", "type": "人物", "note": "另一说"},
                           {"name": "ミナ", "zh": "米娜", "type": "人物", "note": "新人"}],
              "asr_risks": [{"heard_as": "成れ果て", "should_be": "なれはて", "reason": "r2"}],
              "name_disputes": [{"name": "ヒロ", "suggest": "广", "reason": "第 3 行自称"}]}
    third = {"global_summary": "后段", "entities": [{"name": "ヒロ", "zh": "希罗", "type": "人物", "note": "措辞不同"}]}
    payload, handed, warnings = _named_chunks_run(tmp_path, monkeypatch, [first, second, third])

    assert handed["c01"] == ""
    assert '"希罗"' in handed["c02"] and '"前段"' in handed["c02"] and "米娜" not in handed["c02"]
    assert '"希罗"' in handed["c03"] and "米娜" in handed["c03"] and '"广"' not in handed["c03"]
    general = payload["context_pack"]["general_context"]
    table = general[gc.NAME_TABLE_KEY]
    assert [(item["name"], item["zh"]) for item in table["entities"]] == [("ヒロ", "希罗"), ("ミナ", "米娜")]
    assert len(table["asr_risks"]) == 2, "differing fixes of one mishearing both stay"
    assert all(set(general[chunk]) == {"global_summary"} for chunk in ("c01", "c02", "c03"))
    records = {record["chunk_id"]: record for record in payload["research_chunks"]}
    assert records["c02"]["name_disputes"] == [{"name": "ヒロ", "suggest": "广", "reason": "第 3 行自称"}]
    assert [item["dropped"] for item in records["c02"]["relisted"]] == [{"zh": "广", "type": "人物"}]
    assert "relisted" not in records["c03"], "a note worded another way is no disagreement"
    disputed = [message for code, message in warnings if code == "research-name-disputes"]
    assert len(disputed) == 1 and "写成 中文写法 广、类型 人物，保留前面的 中文写法 希罗、类型 人物" in disputed[0]
    pack = ContextPack.from_dict(payload["context_pack"])
    assert not pack.name_disputes and "name_disputes" not in json.dumps(payload["context_pack"], ensure_ascii=False)
    # a resumed run replays the chunk files and finds the same disagreements
    replayed, handed_again, _warnings = _named_chunks_run(tmp_path, monkeypatch, [first, second, third])
    assert not handed_again and replayed["research_chunks"][1]["name_disputes"] == records["c02"]["name_disputes"]


def test_a_later_chunk_reruns_when_an_earlier_one_delivers_differently(tmp_path, monkeypatch) -> None:
    """The carry is part of a chunk's input hash: the first chunk's hash is unchanged by it, a later one's
    follows what came before."""

    one = {"global_summary": "前段", "entities": [{"name": "ヒロ", "zh": "希罗", "type": "人物"}]}
    two = {"global_summary": "中段"}
    payload, _handed, _warnings = _named_chunks_run(tmp_path, monkeypatch, [one, two])
    hashes = [record["input_hash"] for record in payload["research_chunks"]]
    for chunk in ("c01", "c02"):
        (tmp_path / f"research-context-{chunk}.json").unlink()
    again, _handed, _warnings = _named_chunks_run(
        tmp_path, monkeypatch, [{**one, "entities": [{"name": "ヒロ", "zh": "广", "type": "人物"}]}, two])
    changed = [record["input_hash"] for record in again["research_chunks"]]
    assert changed[0] == hashes[0] and changed[1] != hashes[1]


def test_the_name_table_and_carry_keep_every_entry_past_the_ceiling() -> None:
    blocks = [("c01", {"global_summary": "甲" * 50, "entities": [{"name": f"n{i}", "zh": "名", "note": "说明" * 20}
                                                               for i in range(30)]}),
              ("c02", {"global_summary": "乙" * 50})]
    count = len
    table, _drift = gc.merge_name_tables(blocks)
    fitted, over = gc.fit_name_table(table, count=count, max_tokens=1500)
    assert len(fitted["entities"]) == 30 and "note" not in fitted["entities"][0] and over
    text, over = gc.carry_text(blocks, count=count, max_tokens=1500)
    carried = json.loads(text)
    assert len(carried[gc.NAME_TABLE_KEY]["entities"]) == 30 and gc.EARLIER_SUMMARIES_KEY not in carried and over
    assert gc.carry_text(blocks, count=count, max_tokens=1500) == (text, over), "the same blocks, the same text"


def test_readers_lead_with_the_name_table() -> None:
    from finesub.llm.stages.speaker_judge import run as judge_run

    pack = ContextPack(
        general_context={gc.NAME_TABLE_KEY: {"entities": [{"name": "ヒロ", "zh": "希罗"}],
                                              "speakers": [{"name": "ヒロ", "role": "主播"}]},
                         "c01": {"global_summary": "一"}, "c02": {"global_summary": "二"}},
        general_chunks=(GeneralChunk("c01", "1", "2"), GeneralChunk("c02", "3", "4")),
    ).with_source_order(["1", "2", "3", "4"])
    text = pack.general_text_for_source_ids(["3"])
    assert text.index(gc.NOTE_KEY) < text.index(gc.NAME_TABLE_KEY) < text.index('"c01"') and "全片名字表」为准" in text
    assert judge_run.entities_of(pack) == [{"name": "ヒロ", "zh": "希罗"}]
    assert judge_run.speakers_of(pack) == [{"name": "ヒロ", "role": "主播"}]
    assert judge_run.summary_of(pack, ["3"]) == "二\n一"
    assert gc.NAME_TABLE_KEY in pack.general_text_whole()
    # a chunked pack from before the table: no table sentence in the note
    bare = ContextPack(general_context={"c01": {"global_summary": "一"}},
                       general_chunks=(GeneralChunk("c01", "1", "4"),)).with_source_order(["1", "2", "3", "4"])
    assert "全片名字表" not in bare.general_text_for_source_ids(["1"])


def test_name_disputes_stay_out_of_every_reader() -> None:
    pack = ContextPack.from_dict({"general_context": {"global_summary": "s", "name_disputes": [
        {"name": "ヒロ", "suggest": "广", "reason": "r"}]}})
    assert pack.name_disputes == ({"name": "ヒロ", "suggest": "广", "reason": "r"},)
    assert "name_disputes" not in pack.general_context and "广" not in pack.general_text_for_source_ids(["1"])
    assert ContextPack.from_dict(pack.to_dict()).name_disputes == pack.name_disputes


def test_the_carry_reaches_the_prepare_prompt_only_for_a_later_chunk() -> None:
    from finesub.llm.prepare_prompts import build_prepare_task_messages

    plain = build_prepare_task_messages(transcript="t")
    later = build_prepare_task_messages(transcript="t", earlier_chunks='{"x": 1}')
    for text in (plain[0]["content"], plain[-1]["content"]):
        assert "earlier_chunks" not in text and "name_disputes" not in text and "分段接续" not in text
    assert "分段接续" in later[0]["content"] and '"name_disputes"' in later[0]["content"]
    user = later[-1]["content"]
    assert user.index("<earlier_chunks>\n{\"x\": 1}\n</earlier_chunks>") < user.index("<transcript>")


def test_chunk_resume_honours_the_stage_level_reuse_gates(tmp_path, monkeypatch) -> None:
    """Reviewer 2026-08-30 P1-2: the chunk files are an L3-adjacent cache, so
    they obey the same gates -- test_profile changing must re-run the chunk;
    difficulty changing must NOT (it is deliberately exempt: switching to
    intermediate exists to stop burning quota, not to redo it), and neither
    must a bump of the correction's prompt version (audit-only)."""

    import re
    from dataclasses import replace as dc_replace

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[windows[0]]]
    calls: list[str] = []

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        calls.append(chunk_label)
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        return {
            "context_pack": {"general_context": {},
                             "window_contexts": [{"window_id": window_id, "context": "n"}]},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    profile = resolve_profile()

    def run(**overrides):
        kwargs = dict(
            segments=segments,
            context_file=tmp_path / "research-context.json",
            extra_info="",
            knowledge_root=tmp_path,
            knowledge_enabled=False,
            profile=profile,
            test_profile=True,
            task_artifact_dir=None,
            task_id="t",
            research_runner=fake_run_research,
            token_counter=FakeTokenCounter(),
            collect_task_feedback=False,
            resume=True,
            client=object(),
            transcript_cap=1000,
        )
        kwargs.update(overrides)
        return research_mod._run_chunked_research(chunk_plan, **kwargs)

    run()
    assert calls == ["c01"]
    run()  # unchanged inputs: replayed off the chunk file
    assert calls == ["c01"]
    run(test_profile=False)  # a smoke chunk must never serve a real run
    assert calls == ["c01", "c01"]
    monkeypatch.setattr(research_mod, "PROMPT_VERSION", "test-bumped-version")
    run(test_profile=False)  # the correction's prompt version is not the chunk's: replayed
    assert calls == ["c01", "c01"]
    # Difficulty is exempt from the gates: the finished chunk replays.
    run(test_profile=False, profile=dc_replace(profile, difficulty="intermediate"))
    assert calls == ["c01", "c01"]
