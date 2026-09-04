from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.client import LLMCallResult
from finesub.llm.routing.config import LLMRole
from finesub.llm.knowledge import update as update_module
from finesub.llm.knowledge.base import append_task_artifact
from finesub.llm.knowledge.update import (
    CHUNK_LEDGER_FILENAME,
    derive_task_paths,
    run_knowledge_update,
)
from finesub.llm.prompts import (
    build_fast_round1_messages,
    build_research_round2_messages,
)
from finesub.llm.research import extract_round_task_feedback
from finesub.subtitles.model import SrtSegment, render_srt


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


def _proposal_response(*, stray_block: bool = False) -> str:
    create = json.dumps(
        {
            "op": "create_entry",
            "category": "common",
            "entry": "游戏B",
            "entry_type": "游戏",
            "intro": "测试游戏",
            "reason": "库中没有母词条",
        },
        ensure_ascii=False,
    )
    append = json.dumps(
        {
            "op": "append_lines",
            "category": "common",
            "entry": "游戏B",
            "section": "术语",
            "content": "A|甲|||A 固定译为甲。",
            "reason": "feedback hint + final_csv 差异",
        },
        ensure_ascii=False,
    )
    text = f"<knowledge_proposals>\n{create}\n{append}\n</knowledge_proposals>"
    if stray_block:
        # A model that emits the block deleted in step 3 (or any block the
        # harness does not parse): the run must ignore it, not choke on it.
        text += (
            "\n<mistake_proposals>\n"
            '{"op":"add_mistake","source":"hello","wrong":"你好","correct":"您好"}\n'
            "</mistake_proposals>"
        )
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
        "sub|1|1.0|0.0|hello|你好|8|2|\n"
        "sub|2|1.0|0.0|bye|再见|7|2|\n",
        encoding="utf-8",
    )
    paths["research_context"].write_text(
        json.dumps(
            {
                "context_pack": {
                    "general_context": {"global_summary": "整体摘要"},
                    "window_contexts": [
                        {
                            "window_id": "0001",
                            "first_source_id": "1",
                            "last_source_id": "2",
                            "context": "第一窗背景",
                        }
                    ],
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


@pytest.mark.requires_main_checkout
def test_run_knowledge_update_artifacts_only_ignores_a_stray_block(
    tmp_path,
) -> None:
    """The artifacts_only variant has no style section and no second output
    block; a model that emits one anyway must change nothing."""

    final_srt, paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    # The model disobeys and emits a mistake block anyway: harness must ignore it.
    client = FakeClient(_proposal_response(stray_block=True))

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
    assert "mistake_proposals" not in system  # the block is gone from both variants
    assert "翻译风格条目" not in system  # ...and so is the style section
    assert "--- window 0001" in user
    assert "游戏B" in user  # entry excerpt prefetched from the feedback hint
    assert "库中暂无" in user
    assert "第一窗背景" in user
    assert "<refined_csv>" not in user
    assert "<common_mistakes>" not in user
    assert "<good_examples>" not in user
    # Knowledge applied; mistake ledger untouched (design F/G).
    assert "A 固定译为甲" in (knowledge_root / "rendered" / "common" / "游戏B.md").read_text(
        encoding="utf-8"
    )
    # No ledger, and no report field for one: both left with step 3.
    assert not (knowledge_root / "translation").exists()
    assert "mistake_report" not in report["chunks"][0]
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


@pytest.mark.requires_main_checkout
def test_run_knowledge_update_retries_invalid_jsonl_then_applies(tmp_path) -> None:
    final_srt, paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    bad = "<knowledge_proposals>\n{not-json\n</knowledge_proposals>"
    good = _proposal_response()
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
    assert "A 固定译为甲" in (knowledge_root / "rendered" / "common" / "游戏B.md").read_text(
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


@pytest.mark.requires_main_checkout
def test_run_knowledge_update_reruns_skip_applied_chunks(tmp_path) -> None:
    final_srt, _, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    client = FakeClient(_proposal_response())
    common_kwargs = dict(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=client,
    )

    run_knowledge_update(**common_kwargs)
    entry_text = (knowledge_root / "rendered" / "common" / "游戏B.md").read_text(encoding="utf-8")
    report = run_knowledge_update(
        **{
            **common_kwargs,
            "task_summary": "参数变化后的任务说明",
            "difficulty": "intermediate",
        }
    )

    # Execution metadata may change; applied materials remain committed.
    assert len(client.calls) == 1
    assert report["chunks"][0]["skipped"] == "already_applied"
    assert (knowledge_root / "rendered" / "common" / "游戏B.md").read_text(
        encoding="utf-8"
    ) == entry_text


@pytest.mark.requires_main_checkout
def test_run_knowledge_update_recovers_commit_before_ledger_crash(tmp_path) -> None:
    final_srt, paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    client = FakeClient(_proposal_response())
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
    # Simulate a crash after the store revision committed but before the applied
    # ledger record was appended.
    ledger_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in intents),
        encoding="utf-8",
    )

    report = run_knowledge_update(**common_kwargs)

    assert len(client.calls) == 1
    assert report["chunks"][0]["skipped"] == "already_applied"
    assert "recovered_after_commit" in ledger_path.read_text(encoding="utf-8")


@pytest.mark.requires_main_checkout
def test_run_knowledge_update_refined_mode_reads_the_refined_material(tmp_path) -> None:
    """What the refined variant is FOR: the refined lines reach the prompt.

    It used to also write the mistake ledger; that half became a style entry
    (`docs/plans/translation-style-plan.md`), which this run names none of — so the
    prompt carries no style section either."""

    final_srt, _, refined = _write_task_outputs(tmp_path, with_refined=True)
    knowledge_root = tmp_path / "knowledge"
    client = FakeClient(_proposal_response(stray_block=True))

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
    assert "<mistake_proposals>" not in system
    assert "<refined_csv>" in user
    assert "你好呀" in user
    # the stray block changed nothing: no ledger file, no entry from it
    assert not (knowledge_root / "translation").exists()
    assert "A 固定译为甲" in (knowledge_root / "rendered" / "common" / "游戏B.md").read_text(
        encoding="utf-8"
    )


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


def test_run_knowledge_update_dry_run_writes_no_revision_when_apply_is_true(
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

    from finesub.llm.knowledge.base import knowledge_version

    assert knowledge_version(knowledge_root) == "rev:0"


@pytest.mark.requires_main_checkout
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
        "sub|1|1.0|0.0|hello|你好|8|2|\n"
        "sub|2|1.0|0.0|bye|再见|7|2|\n",
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
    client = FakeClient(_proposal_response())

    class TinyLimits:
        prompt_input_limit = 100  # forces the two-window chunk to split
        output_limit = 65_536

    monkeypatch.setattr(
        "finesub.llm.knowledge.update.planning_limits_for",
        lambda task_group, difficulty: TinyLimits(),
    )

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

    class LargeLimits:
        prompt_input_limit = 1_000_000
        output_limit = 65_536

    monkeypatch.setattr(
        "finesub.llm.knowledge.update.planning_limits_for",
        lambda task_group, difficulty: LargeLimits(),
    )
    resumed = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="换模型后分块边界改变",
        knowledge_root=tmp_path / "knowledge",
        token_counter=WordCounter(),
        client=client,
        difficulty="intermediate",
    )

    assert len(client.calls) == 2
    assert resumed["chunks"][0]["skipped"] == "already_applied"


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
    from finesub.llm.chunking import SubtitleSegment, plan_correction_windows

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
    from finesub.llm.stages.fast_session import parse_fast_round1_output

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




def test_a_worktree_asks_before_writing_the_main_knowledge_base(
    tmp_path, monkeypatch
) -> None:
    # Worktrees share the main checkout's knowledge base on purpose, so an
    # experiment in one would otherwise commit into the real thing. Skipping
    # costs nothing: no quota is spent and the ledger does not advance.
    final_srt, _, _ = _write_task_outputs(tmp_path)
    client = FakeClient(_proposal_response())
    monkeypatch.setattr(update_module, "is_linked_worktree", lambda: True)
    monkeypatch.delenv("FINESUB_KNOWLEDGE_WRITE", raising=False)

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-worktree",
        knowledge_root=tmp_path / "knowledge",
        token_counter=FakeTokenCounter(),
        client=client,
    )

    assert report["skipped"] == "worktree_readonly"
    assert client.calls == []


def test_a_worktree_writes_when_the_developer_says_so(tmp_path, monkeypatch) -> None:
    final_srt, _, _ = _write_task_outputs(tmp_path)
    monkeypatch.setattr(update_module, "is_linked_worktree", lambda: True)
    monkeypatch.setenv("FINESUB_KNOWLEDGE_WRITE", "1")

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-worktree-allowed",
        knowledge_root=tmp_path / "knowledge",
        token_counter=FakeTokenCounter(),
        client=FakeClient(_proposal_response()),
    )

    assert report.get("skipped") != "worktree_readonly"


@pytest.mark.requires_main_checkout
def test_another_process_holding_the_knowledge_lock_skips_applying(
    tmp_path, monkeypatch
) -> None:
    # One embedded git repository, several front ends: a second process
    # committing between our apply and our commit would fold our uncommitted
    # files into its commit. Waiting it out, then giving up, is the same
    # degradation as a dirty repository -- warn, keep the proposal, do not
    # advance the ledger.
    from finesub_bootstrap.locks import holding_lock
    from finesub.llm.knowledge import base as knowledge_base

    final_srt, _, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(knowledge_base, "KNOWLEDGE_LOCK_TIMEOUT_SECONDS", 0.1)

    with holding_lock(knowledge_base.knowledge_lock_path(knowledge_root)):
        report = run_knowledge_update(
            final_srt=final_srt,
            task_id="task-locked",
            knowledge_root=knowledge_root,
            token_counter=FakeTokenCounter(),
            client=FakeClient(_proposal_response()),
        )

    applied = [
        chunk.get("knowledge_report") for chunk in report["chunks"]
    ]
    assert all(entry is None for entry in applied)
    ledgers = list(tmp_path.rglob("knowledge-update-chunks.jsonl"))
    assert ledgers == [] or all(
        path.read_text(encoding="utf-8").strip() == "" for path in ledgers
    )


def test_the_knowledge_lock_sits_outside_the_git_worktree(tmp_path) -> None:
    from finesub.llm.knowledge.base import knowledge_lock_path

    # Inside the knowledge directory it would be swept into the auto-apply
    # commits; keyed on an install root it would not be the same file for two
    # front ends sharing one knowledge base.
    lock = knowledge_lock_path(tmp_path / "knowledge")

    assert lock == tmp_path / "knowledge.lock"



# --- B': the conflict repair round -------------------------------------------


def _competitor_commit(repo) -> None:
    """A real foreign revision, as the concurrent task B' exists for."""

    from finesub.llm.knowledge.node.proposals import (
        apply_model_proposals as engine_apply,
    )

    foreign = json.dumps(
        {
            "op": "create_entry",
            "category": "common",
            "entry": "竞争者游戏",
            "entry_type": "游戏",
            "intro": "另一个任务写的",
            "reason": "并发写入",
        },
        ensure_ascii=False,
    )
    engine_apply(
        f"<knowledge_proposals>\n{foreign}\n</knowledge_proposals>",
        repo=repo,
        task_id="competitor",
        knowledge_read_rev=repo.rev,
    )


def _conflicting_apply(monkeypatch, *, competitor: bool) -> list[int]:
    """Make the FIRST apply report one CAS-dropped line; returns the read revs.

    ``competitor`` lands a real foreign revision alongside it, which is what
    the repair guard now checks for: without one the drops are self-inflicted
    and no repair round runs.
    """

    real_apply = update_module.apply_model_proposals
    reads: list[int] = []

    def patched(text, **kwargs):
        report = real_apply(text, **kwargs)
        reads.append(kwargs["knowledge_read_rev"])
        if len(reads) == 1:
            if competitor:
                _competitor_commit(kwargs["repo"])
            data = report.to_dict()
            data["conflicts"] = [
                {"entity": "node", "id": "n1", "reason": "标记 [A] 已被占用", "dropped_ops": [1]}
            ]
            data["skipped"] = [
                {
                    "category": "common",
                    "entry": "游戏B",
                    "op": "append_lines",
                    "section": "术语",
                    "reason": "dropped by CAS conflict",
                }
            ]
            report.conflicts = data["conflicts"]
            report.skipped = []
            report.to_dict = lambda data=data: data  # type: ignore[method-assign]
        return report

    monkeypatch.setattr(update_module, "apply_model_proposals", patched)
    return reads


def test_conflicted_entries_only_picks_what_a_repair_round_can_fix() -> None:
    """Rejections are not conflicts: a malformed op or a missing parent will be
    rejected again no matter how fresh the entry is."""

    from finesub.llm.knowledge.update import MAX_REPAIR_ENTRIES, conflicted_entries

    report = {
        "conflicts": [{"entity": "node", "id": "N1", "reason": "x"}],
        "skipped": [
            {"category": "common", "entry": "甲", "reason": "dropped by CAS conflict"},
            {"category": "common", "entry": "乙", "reason": "标记 [出道] 在小节 '档案' 内已被占用"},
            {"category": "common", "entry": "丙", "reason": "create_entry: without a parent"},
            {"category": "common", "entry": "甲", "reason": "dropped by CAS conflict"},
        ],
    }
    assert conflicted_entries(report) == [("common", "甲"), ("common", "乙")]
    # A rolled-back envelope is an authoring error, not a race.
    assert conflicted_entries({**report, "rolled_back": True}) == []
    assert conflicted_entries({"skipped": report["skipped"]}) == []
    many = {
        "conflicts": [{"entity": "node"}],
        "skipped": [
            {"category": "common", "entry": f"e{n}", "reason": "dropped by CAS conflict"}
            for n in range(MAX_REPAIR_ENTRIES + 3)
        ],
    }
    assert len(conflicted_entries(many)) == MAX_REPAIR_ENTRIES


@pytest.mark.requires_main_checkout
def test_a_conflict_is_fed_back_with_the_winner_s_version(tmp_path, monkeypatch) -> None:
    """B' (plan §4.2): the dropped lines are re-decided against what the winner
    wrote, at the new revision and with the handles of that revision."""

    final_srt, paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    applies = _conflicting_apply(monkeypatch, competitor=True)
    client = SequenceFakeClient(
        [_proposal_response(), _proposal_response()]
    )

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=client,
    )

    assert len(client.calls) == 2, "the repair round never ran"
    repair_user = client.calls[1][1]["content"]
    assert "游戏B" in repair_user and "dropped by CAS conflict" in repair_user
    # The dropped proposal VERBATIM: an op name and an entry name are not
    # something a model can re-decide from (reviewer 2026-08-31 P1).
    assert "A 固定译为甲。" in repair_user
    assert "你原来提的" in repair_user
    assert "只处理上面列出的这几条" in repair_user
    # It reads the CURRENT revision, not the one the first round read.
    assert applies[1] > applies[0]
    assert f"rev {applies[1]}" in repair_user
    chunk = report["chunks"][0]
    assert chunk["knowledge_repair_report"]["attempted"] is True
    assert chunk["knowledge_repair_report"]["current_rev"] == applies[1]
    kinds = [
        json.loads(line)["kind"]
        for line in (paths["artifact_dir"] / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert "knowledge_conflict_repair" in kinds


@pytest.mark.requires_main_checkout
def test_a_failing_repair_round_leaves_the_run_alone(tmp_path, monkeypatch) -> None:
    """Everything else is already committed, so B' is additive by construction:
    a repair that errors must not turn a successful update into a failure."""

    final_srt, _paths, _ = _write_task_outputs(tmp_path)
    knowledge_root = tmp_path / "knowledge"
    _conflicting_apply(monkeypatch, competitor=True)

    class ExplodingSecondCall(FakeClient):
        def complete(self, role, messages, **kwargs):
            if self.calls:
                raise RuntimeError("repair round exploded")
            return super().complete(role, messages, **kwargs)

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=knowledge_root,
        token_counter=FakeTokenCounter(),
        client=ExplodingSecondCall(_proposal_response()),
    )

    assert report["chunks"][0]["knowledge_report"]["committed"] is True
    assert "error" in report["chunks"][0]["knowledge_repair_report"]


@pytest.mark.requires_main_checkout
def test_self_inflicted_conflicts_do_not_buy_a_repair_round(tmp_path, monkeypatch) -> None:
    """No foreign revision since the read -- the only commit is the main
    apply's own -- so the drops collided with the model's own lines, and a
    repair round would only show the model what it just wrote."""

    final_srt, _paths, _ = _write_task_outputs(tmp_path)
    _conflicting_apply(monkeypatch, competitor=False)
    client = FakeClient(_proposal_response())

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=tmp_path / "knowledge",
        token_counter=FakeTokenCounter(),
        client=client,
    )

    assert len(client.calls) == 1, "a repair round ran with no competitor to repair against"
    assert "knowledge_repair_report" not in report["chunks"][0]


@pytest.mark.requires_main_checkout
def test_a_repair_round_that_fails_validation_still_leaves_its_exchange(
    tmp_path, monkeypatch
) -> None:
    """The round that fails to validate is precisely the one someone reads
    back, so the exchange is logged before validation, not only on success."""

    final_srt, paths, _ = _write_task_outputs(tmp_path)
    _conflicting_apply(monkeypatch, competitor=True)
    client = SequenceFakeClient(
        [
            _proposal_response(),
            "<knowledge_proposals>\n{not json\n</knowledge_proposals>",
        ]
    )

    report = run_knowledge_update(
        final_srt=final_srt,
        task_id="task-1",
        task_summary="测试任务",
        knowledge_root=tmp_path / "knowledge",
        token_counter=FakeTokenCounter(),
        client=client,
    )

    assert "error" in report["chunks"][0]["knowledge_repair_report"]
    from finesub.llm.exchange_log import EXCHANGE_DIR_NAME

    exchanges = list(
        (paths["artifact_dir"] / EXCHANGE_DIR_NAME).glob("*knowledge-conflict-repair*")
    )
    assert exchanges, "the failed repair round left no exchange record"
