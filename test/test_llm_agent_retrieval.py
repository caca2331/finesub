from __future__ import annotations

import json

from finesub.llm.agent.agent_retrieval import AgentRetrievalAccess
from finesub.llm.agent.agent_task_runtime import (
    DEFAULT_LEASE_TTL_SECONDS,
    AgentTaskRuntime,
    AgentTaskSpec,
    AssignmentConflictError,
)
from finesub.llm.web_search import QueryExtractResult, QuerySearchResult, SearchResultItem
import pytest


class _SearchClient:
    def __init__(self) -> None:
        self.search_calls = 0
        self.fetch_calls = 0

    def search(self, query: str, guided_query: str = "") -> QuerySearchResult:
        self.search_calls += 1
        return QuerySearchResult(
            query=query,
            guided_query=guided_query,
            provider="test",
            items=(
                SearchResultItem(
                    title="FineSub",
                    url="https://example.test/finesub",
                    snippet="public result",
                ),
            ),
        )

    def extract(self, url: str, guided_query: str = "") -> QueryExtractResult:
        self.fetch_calls += 1
        return QueryExtractResult(
            url=url,
            guided_query=guided_query,
            provider="test",
            title="FineSub",
            content="public page",
        )


def _runtime(tmp_path, *, max_queries: int = 2) -> tuple[AgentTaskRuntime, dict]:
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="research",
        tasks=[
            AgentTaskSpec(
                task_id="task-1",
                session_type="research",
                input_hash="sha256:input",
                goal="research",
                retrieval_mode="local",
                retrieval_budget={"max_queries": max_queries},
            )
        ],
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    return runtime, claimed


def _call_args(claimed: dict) -> dict:
    return {
        "assignment_id": "assignment-1",
        "task_id": "task-1",
        "worker_id": "worker-1",
        "lease_generation": claimed["task"]["lease_generation"],
    }


def test_local_retrieval_is_durable_idempotent_and_fetch_is_search_bounded(
    tmp_path,
) -> None:
    runtime, claimed = _runtime(tmp_path)
    client = _SearchClient()
    access = AgentRetrievalAccess(
        runtime, client=client, count_tokens=lambda text: len(text) // 4
    )

    searched = access.search(
        **_call_args(claimed), request_id="search-1", query="FineSub"
    )
    repeated = access.search(
        **_call_args(claimed), request_id="search-1", query="FineSub"
    )
    assert searched["status"] == "completed"
    assert repeated["result"] == searched["result"]
    assert client.search_calls == 1
    assert searched["budget"]["used"]["queries_used"] == 1

    fetched = access.fetch(
        **_call_args(claimed),
        request_id="fetch-1",
        url="https://example.test/finesub",
    )
    assert fetched["status"] == "completed"
    assert fetched["result"]["content"] == "public page"
    assert client.fetch_calls == 1

    with pytest.raises(AssignmentConflictError, match="completed search result"):
        access.fetch(
            **_call_args(claimed),
            request_id="fetch-private",
            url="https://attacker.test/not-shown",
        )


def test_local_retrieval_budget_does_not_reset_or_double_charge(tmp_path) -> None:
    runtime, claimed = _runtime(tmp_path, max_queries=1)
    client = _SearchClient()
    access = AgentRetrievalAccess(runtime, client=client, count_tokens=lambda _text: 1)
    args = _call_args(claimed)

    assert access.search(**args, request_id="search-1", query="one")["status"] == "completed"
    exhausted = access.search(**args, request_id="search-2", query="two")
    repeated = access.search(**args, request_id="search-2", query="two")
    assert exhausted["status"] == repeated["status"] == "budget_exhausted"
    assert client.search_calls == 1


def test_reservations_from_an_expired_lease_stop_holding_parallel_slots(
    tmp_path,
) -> None:
    now = [100.0]
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="research",
        tasks=[
            AgentTaskSpec(
                task_id="task-1",
                session_type="research",
                input_hash="sha256:input",
                goal="research",
                retrieval_mode="local",
                retrieval_budget={"max_parallel": 1},
            )
        ],
        clock=lambda: now[0],
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    stranded = runtime.begin_retrieval_call(
        **_call_args(claimed),
        request_id="search-lost",
        operation="search",
        request={"query": "one"},
    )
    assert stranded["status"] == "in_progress"

    now[0] += DEFAULT_LEASE_TTL_SECONDS + 1
    reclaimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="reclaim",
        expected_control_generation=stranded["control_generation"],
    )
    args = _call_args(reclaimed)
    assert args["lease_generation"] != claimed["task"]["lease_generation"]
    access = AgentRetrievalAccess(
        runtime, client=_SearchClient(), count_tokens=lambda _text: 1
    )

    assert (
        access.search(**args, request_id="search-next", query="two")["status"]
        == "completed"
    )
    settled = runtime.complete_retrieval_call(
        **args,
        request_id="search-lost",
        result={"items": []},
        result_count=0,
        response_tokens=1,
        wall_seconds=1.0,
    )
    assert settled["status"] == "abandoned"


def test_native_retrieval_is_recorded_as_a_soft_limit(tmp_path) -> None:
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="research",
        tasks=[
            AgentTaskSpec(
                task_id="task-1",
                session_type="research",
                input_hash="sha256:input",
                goal="research",
                retrieval_mode="native",
                retrieval_budget={"max_queries": 1},
            )
        ],
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    events = [
        {"query": "one", "urls": ["https://example.test/a"]},
        {"query": "two", "urls": ["https://example.test/b", "https://example.test/c"]},
    ]

    recorded = runtime.record_native_retrieval(
        **_call_args(claimed), request_id="native-1", search_events=events
    )
    assert recorded["status"] == "recorded"
    assert recorded["enforcement"] == "soft"
    assert recorded["searches"] == 2
    assert recorded["violations"] == ["max_queries"]
    assert recorded["budget"]["used"]["queries_used"] == 2
    assert recorded["budget"]["used"]["results_returned"] == 3
    assert (
        runtime.record_native_retrieval(
            **_call_args(claimed), request_id="native-1", search_events=[]
        )["searches"]
        == 2
    )
    assert (
        json.loads(runtime.read_artifact(recorded["result_ref"]))["searches"][0]["query"]
        == "one"
    )


def test_local_retrieval_rejects_non_local_task(tmp_path) -> None:
    runtime = AgentTaskRuntime.start_assignment(
        tmp_path / "assignment",
        assignment_id="assignment-1",
        worker_goal="finish",
        tasks=[
            AgentTaskSpec(
                task_id="task-1",
                session_type="correction",
                input_hash="sha256:input",
                goal="correct",
            )
        ],
    )
    claimed = runtime.next_task(
        assignment_id="assignment-1",
        worker_id="worker-1",
        request_id="claim",
        expected_control_generation=1,
    )
    access = AgentRetrievalAccess(runtime, client=_SearchClient(), count_tokens=lambda _text: 1)
    with pytest.raises(AssignmentConflictError, match="retrieval_mode=local"):
        access.search(
            **_call_args(claimed), request_id="search-1", query="should fail"
        )
