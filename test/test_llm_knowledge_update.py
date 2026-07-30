from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.client import LLMCallResult
from llm.config import LLMRole
from llm.knowledge.base import append_task_artifact
from llm.knowledge.update import (
    CHUNK_LEDGER_FILENAME,
    derive_task_paths,
    run_knowledge_update,
)
from llm.prompts import (
    build_fast_round1_messages,
    build_research_round2_messages,
)
from llm.research import extract_round_task_feedback
from asr_playground.subtitles.model import SrtSegment, render_srt


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    def complete(self, role, messages, **kwargs):
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        return LLMCallResult(
            content=self.content,
            role=LLMRole.GENERAL_CAPABLE,
            model="fake",
            fallback_used=False,
            raw_response={},
        )


class SequenceFakeClient:
    """Returns successive contents for parse-retry tests."""

    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.calls: list[list[dict]] = []
        self.kwargs: list[dict] = []

    def complete(self, role, messages, **kwargs):
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return LLMCallResult(
            content=self.contents[index],
            role=LLMRole.GENERAL_CAPABLE,
            model="fake",
            fallback_used=False,
            raw_response={},
        )


def _feedback_json() -> str:
    return json.dumps(
        {
            "knowledge_hints": [
                {
                    "category": "common",
                    "entry": "游戏B",
                    "direction": "new_entry",
                    "focus": "新游戏值得建条目",
                    "reason": "窗口证据",
                    "confidence": 7,
                }
            ],
            "asr_corrections": [],
            "uncertainties": ["剧情线待确认"],
        },
        ensure_ascii=False,
    )


def _proposal_response(*, with_mistakes: bool) -> str:
    proposal = json.dumps(
        {
            "category": "common",
            "entry": "游戏B",
            "entry_type": "游戏",
            "intro": "测试游戏",
            "op": "replace_section",
            "section": "简介",
            "content": "A 固定译为甲。",
            "reason": "feedback hint + final_csv 差异",
        },
        ensure_ascii=False,
    )
    text = f"<knowledge_proposals>\n{proposal}\n</knowledge_proposals>"
    if with_mistakes:
        mistake = json.dumps(
            {
                "op": "add_mistake",
                # wrong must be findable in the chunk's material text
                # (anti-fabrication check) — "你好" is the final translation.
                "source": "hello",
                "wrong": "你好",
                "correct": "您好",
                "note": "打招呼场景",
                "prompt_version": "v9",
                "reason": "refined 对照",
            },
            ensure_ascii=False,
        )
        text += f"\n<mistake_proposals>\n{mistake}\n</mistake_proposals>"
    return text


def _write_task_outputs(tmp_path: Path, *, with_refined: bool = False):
    final_srt = tmp_path / "x.srt"
    final_srt.write_text(
        render_srt(
            [
                SrtSegment(index=1, start=0.0, end=1.0, text="你好"),
                SrtSegment(index=2, start=2.0, end=3.0, text="再见"),
            ]
        ),
        encoding="utf-8",
    )
    paths = derive_task_paths(final_srt)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    paths["stable_json"].write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "hello"},
                    {"id": "2", "start": 2.0, "end": 3.0, "text": "bye"},
                ]
            }
        ),
        encoding="utf-8",
    )
    paths["annotated_csv"].write_text(
        "# type|position|duration|corrected|translation|conf|note\n"
        "sub|1|1.0|hello|你好|8|\n"
        "sub|2|1.0|bye|再见|7|\n",
        encoding="utf-8",
    )
    paths["research_context"].write_text(
        json.dumps(
            {
                "context_pack": {
                    "general_context": {"global_summary": "整体摘要"},
                    "window_contexts": {"0001": "第一窗背景"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    append_task_artifact(
        paths["artifact_dir"],
        kind="correction_window_response",
        task_id="t",
        payload={
            "chunk_id": "0001",
            "validation_ok": True,
            "output_limited": False,
            "window": {"source_ids": ["1", "2"]},
        },
    )
    append_task_artifact(
        paths["artifact_dir"],
        kind="correction_window_task_feedback",
        task_id="t",
        payload={"chunk_id": "0001", "feedback": _feedback_json()},
    )
    refined = None
    if with_refined:
        refined = tmp_path / "refined.srt"
        refined.write_text(
            render_srt([SrtSegment(index=1, start=0.0, end=1.0, text="你好呀")]),
            encoding="utf-8",
        )
    return final_srt, paths, refined


def test_run_knowledge_update_artifacts_only_applies_and_ignores_mistakes(
    tmp_path,
) -> None:
    final_srt, paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    # The model disobeys and emits a mistake block anyway: harness must ignore it.
    client = FakeClient(_proposal_response(with_mistakes=True))

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=client,
    )

    assert report["mode"] == "artifacts_only"
    assert len(client.calls) == 1
    system = client.calls[0][0]["content"]
    user = client.calls[0][1]["content"]
    assert "无精修模式" in system
    assert "mistake_proposals" not in system
    assert "--- window 0001" in user
    assert "游戏B" in user  # entry excerpt prefetched from the feedback hint
    assert "库中暂无" in user
    assert "第一窗背景" in user
    assert "<refined_csv>" not in user
    assert "<common_mistakes>" not in user
    assert "<good_examples>" not in user
    # Knowledge applied; mistake ledger untouched (design F/G).
    assert "A 固定译为甲" in (knowledge_root / "common" / "游戏B.md").read_text(
        encoding="utf-8"
    )
    assert not (knowledge_root / "translation" / "common-mistake.md").exists()
    assert report["chunks"][0]["mistake_report"] is None
    # Ledger written; artifacts retained.
    ledger = paths["artifact_dir"] / CHUNK_LEDGER_FILENAME
    assert ledger.exists()
    kinds = [
        json.loads(line)["kind"]
        for line in (paths["artifact_dir"] / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "knowledge_update_response" in kinds
    assert "knowledge_update_apply_report" in kinds


def test_run_knowledge_update_retries_invalid_jsonl_then_applies(tmp_path) -> None:
    final_srt, paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    bad = "<knowledge_proposals>\n{not-json\n</knowledge_proposals>"
    good = _proposal_response(with_mistakes=False)
    client = SequenceFakeClient([bad, good])

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-retry",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=client,
    )

    assert len(client.calls) == 2
    assert client.kwargs[0].get("temperature") == 1.0
    assert client.kwargs[1].get("temperature") == 0.99
    assert "A 固定译为甲" in (knowledge_root / "common" / "游戏B.md").read_text(
        encoding="utf-8"
    )
    assert report["chunks"][0]["executed"] is True
    artifacts = [
        json.loads(line)
        for line in (paths["artifact_dir"] / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    ku_responses = [
        row for row in artifacts if row.get("kind") == "knowledge_update_response"
    ]
    assert len(ku_responses) == 2
    assert ku_responses[0]["payload"]["parse_error"]
    assert not ku_responses[1]["payload"].get("parse_error")


def test_run_knowledge_update_reruns_skip_applied_chunks(tmp_path) -> None:
    final_srt, _, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    client = FakeClient(_proposal_response(with_mistakes=False))
    common_kwargs = dict(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=client,
    )

    run_knowledge_update(**common_kwargs)
    entry_text = (knowledge_root / "common" / "游戏B.md").read_text(encoding="utf-8")
    report = run_knowledge_update(**common_kwargs)

    # Second run hits the ledger: no new LLM call, no double apply.
    assert len(client.calls) == 1
    assert report["chunks"][0]["skipped"] == "already_applied"
    assert (knowledge_root / "common" / "游戏B.md").read_text(
        encoding="utf-8"
    ) == entry_text


def test_run_knowledge_update_recovers_commit_before_ledger_crash(tmp_path) -> None:
    final_srt, paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    client = FakeClient(_proposal_response(with_mistakes=False))
    common_kwargs = dict(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=client,
    )

    run_knowledge_update(**common_kwargs)
    ledger_path = paths["artifact_dir"] / CHUNK_LEDGER_FILENAME
    records = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    intents = [record for record in records if record.get("status") == "intent"]
    assert intents
    # Simulate a crash after the unified git commit but before the applied
    # ledger record was appended.
    ledger_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in intents),
        encoding="utf-8",
    )

    report = run_knowledge_update(**common_kwargs)

    assert len(client.calls) == 1
    assert report["chunks"][0]["skipped"] == "already_applied"
    assert "recovered_after_commit" in ledger_path.read_text(encoding="utf-8")


def test_run_knowledge_update_refined_mode_applies_mistakes(tmp_path) -> None:
    final_srt, _, refined = _write_task_outputs(tmp_path, with_refined=True)
    knowledge_root = tmp_path / "knowledge"
    client = FakeClient(_proposal_response(with_mistakes=True))

    report = run_knowledge_update(
        final_srt=final_srt,
        refined_srt=refined,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=client,
    )

    assert report["mode"] == "refined_aligned"
    system = client.calls[0][0]["content"]
    user = client.calls[0][1]["content"]
    assert "精修对照模式" in system
    assert "<mistake_proposals>" in system
    assert "<refined_csv>" in user
    assert "你好呀" in user
    ledger_text = (knowledge_root / "translation" / "common-mistake.md").read_text(
        encoding="utf-8"
    )
    assert "您好" in ledger_text
    assert report["chunks"][0]["mistake_report"]["applied"]


def test_run_knowledge_update_dry_run_writes_prompts_only(tmp_path) -> None:
    final_srt, _, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    prompt_dir = tmp_path / "prompts"

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        execute=False,
        apply=False,
        prompt_dir=prompt_dir,
        token_counter=FakeTokenCounter(),
    )

    assert report["chunks"][0]["executed"] is False
    prompt_text = (prompt_dir / "knowledge-update-chunk01.txt").read_text(
        encoding="utf-8"
    )
    assert "<raw_csv>" in prompt_text
    assert not (knowledge_root / "common").exists()


def test_run_knowledge_update_dry_run_does_not_prepare_git_when_apply_is_true(
    tmp_path,
) -> None:
    final_srt, _, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"

    run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        execute=False,
        apply=True,
        prompt_dir=tmp_path / "prompts",
        token_counter=FakeTokenCounter(),
    )

    assert not (knowledge_root / ".git").exists()


def test_run_knowledge_update_splits_over_limit_chunks(tmp_path, monkeypatch) -> None:
    # Two executed windows so the over-limit chunk can split on the boundary.
    final_srt = tmp_path / "x.srt"
    final_srt.write_text(
        render_srt(
            [
                SrtSegment(index=1, start=0.0, end=1.0, text="你好"),
                SrtSegment(index=2, start=2.0, end=3.0, text="再见"),
            ]
        ),
        encoding="utf-8",
    )
    paths = derive_task_paths(final_srt)
    paths["artifact_dir"].mkdir(parents=True, exist_ok=True)
    paths["stable_json"].write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "hello"},
                    {"id": "2", "start": 2.0, "end": 3.0, "text": "bye"},
                ]
            }
        ),
        encoding="utf-8",
    )
    paths["annotated_csv"].write_text(
        "# type|position|duration|corrected|translation|conf|note\n"
        "sub|1|1.0|hello|你好|8|\n"
        "sub|2|1.0|bye|再见|7|\n",
        encoding="utf-8",
    )
    for chunk_id, ids in (("0001", ["1"]), ("0002", ["2"])):
        append_task_artifact(
            paths["artifact_dir"],
            kind="correction_window_response",
            task_id="t",
            payload={
                "chunk_id": chunk_id,
                "validation_ok": True,
                "output_limited": False,
                "window": {"source_ids": ids},
            },
        )
    client = FakeClient(_proposal_response(with_mistakes=False))

    class TinyLimits:
        prompt_input_limit = 100  # forces the two-window chunk to split
        output_limit = 65_536

    monkeypatch.setattr("llm.knowledge.update.DEFAULT_LIMITS", TinyLimits())

    class WordCounter(FakeTokenCounter):
        def count_text(self, text: str) -> int:
            # Weight by rendered pack headers ("[...s] ---", absent from the
            # static prompt text) so the two-window prompt (~140) exceeds the
            # tiny 100-token limit but a one-window prompt (~80) fits.
            return (text or "").count("s] ---") * 60 + 10

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=tmp_path / "knowledge",
        token_counter=WordCounter(),
        client=client,
    )

    assert len(report["chunks"]) == 2
    assert [chunk["window_ids"] for chunk in report["chunks"]] == [["0001"], ["0002"]]
    assert len(client.calls) == 2
    # The chunk notice marks a multi-chunk run once splitting happened.
    assert "材料分块说明" in client.calls[0][1]["content"]


# ---------------------------------------------------------------------------
# feedback collection prompt wiring (research round 2 / fast round 1)


def test_research_round2_feedback_block_only_when_collecting() -> None:
    base = dict(transcript="--- window 0001 ---\n1|hello\n")
    off = build_research_round2_messages(**base)
    on = build_research_round2_messages(**base, collect_task_feedback=True)

    assert "task_update_feedback" not in off[0]["content"]
    assert "task_update_feedback" not in off[1]["content"]
    assert "<task_update_feedback>" in on[0]["content"]
    assert "knowledge_hints" in on[0]["content"]
    assert "task_update_feedback" in on[1]["content"]  # closing reminder


def test_extract_round_task_feedback_is_best_effort() -> None:
    text = (
        "<context_pack>{}</context_pack>\n"
        f"<task_update_feedback>{_feedback_json()}</task_update_feedback>"
    )
    assert "游戏B" in extract_round_task_feedback(text, count_tokens=len)
    assert extract_round_task_feedback("no block here", count_tokens=len) == ""
    # Duplicated blocks degrade to empty instead of raising.
    assert (
        extract_round_task_feedback(
            "<task_update_feedback>a</task_update_feedback>"
            "<task_update_feedback>b</task_update_feedback>",
            count_tokens=len,
        )
        == ""
    )


def test_fast_round1_feedback_block_only_when_collecting() -> None:
    from llm.chunking import SubtitleSegment, plan_correction_windows

    window = plan_correction_windows(
        [
            SubtitleSegment("1", 0.0, 1.0, "えっと。"),
            SubtitleSegment("2", 1.2, 2.0, "やばい。"),
        ],
        counter=FakeTokenCounter(),
    )[0]
    off = build_fast_round1_messages(window=window)
    on = build_fast_round1_messages(window=window, collect_task_feedback=True)

    assert "task_update_feedback" not in off[0]["content"]
    assert "<task_update_feedback>" in on[0]["content"]
    assert "task_update_feedback" in on[1]["content"]


def test_fast_round1_parse_collects_feedback() -> None:
    from llm.stages.fast_session import parse_fast_round1_output

    text = (
        "<analysis_notes>要点</analysis_notes>\n"
        "<requested_entries>星野灯</requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>游戏B 剧情</search_queries>\n"
        f"<task_update_feedback>{_feedback_json()}</task_update_feedback>"
    )
    off = parse_fast_round1_output(text, count_tokens=len)
    on = parse_fast_round1_output(text, collect_task_feedback=True, count_tokens=len)

    assert off.task_update_feedback == ""
    assert "游戏B" in on.task_update_feedback
