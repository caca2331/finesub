"""Budgeted harness-owned web retrieval for durable Agent tasks."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from typing import Any, Callable

from .agent_task_runtime import AgentTaskRuntime
from ..web_search import QueryExtractResult, QuerySearchResult, WebSearchClient


class AgentRetrievalAccess:
    """Run local search/fetch calls through the runtime's fenced budget ledger."""

    def __init__(
        self,
        runtime: AgentTaskRuntime,
        *,
        client: WebSearchClient | None = None,
        count_tokens: Callable[[str], int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.client = client or WebSearchClient()
        if count_tokens is None:
            from ..token_budget import default_token_counter

            count_tokens = default_token_counter().count_text
        self.count_tokens = count_tokens
        self.clock = clock

    def search(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        query: str,
        guided_query: str = "",
    ) -> dict[str, Any]:
        request = {"query": query, "guided_query": guided_query}
        return self._run(
            assignment_id=assignment_id,
            task_id=task_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
            request_id=request_id,
            operation="search",
            request=request,
            invoke=lambda: self._search_result(
                self.client.search(query, guided_query=guided_query)
            ),
        )

    def fetch(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        url: str,
        guided_query: str = "",
    ) -> dict[str, Any]:
        request = {"url": url, "guided_query": guided_query}
        return self._run(
            assignment_id=assignment_id,
            task_id=task_id,
            worker_id=worker_id,
            lease_generation=lease_generation,
            request_id=request_id,
            operation="fetch",
            request=request,
            invoke=lambda: self._extract_result(
                self.client.extract(url, guided_query=guided_query)
            ),
        )

    def _run(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        operation: str,
        request: dict[str, Any],
        invoke: Callable[[], tuple[dict[str, Any], int]],
    ) -> dict[str, Any]:
        common = {
            "assignment_id": assignment_id,
            "task_id": task_id,
            "worker_id": worker_id,
            "lease_generation": lease_generation,
            "request_id": request_id,
        }
        reserved = self.runtime.begin_retrieval_call(
            **common, operation=operation, request=request
        )
        if reserved["status"] != "in_progress":
            return reserved
        started = self.clock()
        try:
            result, result_count = invoke()
        except Exception as exc:
            # Closing the reservation must not be able to replace the reason
            # the call failed. If the ledger write itself fails, the transport
            # error is still what the caller needs to see -- the reservation
            # is reclaimed later by the next claim under a fresh lease.
            failure = f"{type(exc).__name__}: {exc}"
            try:
                return self.runtime.fail_retrieval_call(
                    **common,
                    error=failure,
                    wall_seconds=max(0.0, self.clock() - started),
                )
            except Exception:
                return {"status": "failed", "error": failure}
        serialized = json.dumps(
            result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return self.runtime.complete_retrieval_call(
            **common,
            result=result,
            result_count=result_count,
            response_tokens=max(0, int(self.count_tokens(serialized))),
            wall_seconds=max(0.0, self.clock() - started),
        )

    @staticmethod
    def _search_result(result: QuerySearchResult) -> tuple[dict[str, Any], int]:
        payload = {
            "query": result.query,
            "guided_query": result.guided_query,
            "provider": result.provider,
            "items": [asdict(item) for item in result.items],
            "answer": result.answer,
            "error": result.error,
            "fallbacks": [event.to_dict() for event in result.fallbacks],
            "metadata": dict(result.metadata),
        }
        return payload, len(result.items)

    @staticmethod
    def _extract_result(result: QueryExtractResult) -> tuple[dict[str, Any], int]:
        return (
            {
                "url": result.url,
                "guided_query": result.guided_query,
                "provider": result.provider,
                "title": result.title,
                "content": result.content,
                "error": result.error,
                "fallbacks": [event.to_dict() for event in result.fallbacks],
                "metadata": dict(result.metadata),
            },
            0,
        )
