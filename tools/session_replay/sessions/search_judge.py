"""Search-judge round session adapter for prompt iteration.

Freezes one round of the search loop's judge call and replays it with a
modified prompt. The fixture is extracted from ``search-loop-round-<N>.json``
(dumped by ``search_loop.py`` since the 补中间态 change).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from llm.config import CapabilityTier, LLMRole
from llm.prompts import build_search_loop_messages, build_search_loop_v2_messages
from .base import (
    ReplayResult,
    reject_unsupported_variant,
    run_text_replay,
    validate_session_contract,
)


def _load_judge_fixture(artifact_dir: Path, round_index: int | None) -> Dict[str, Any]:
    """Load a search-loop round fixture from the artifact directory."""

    if round_index is not None:
        path = artifact_dir / f"search-loop-round-{round_index}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(
            f"No search-loop fixture for round {round_index} in {artifact_dir}. "
            f"Expected search-loop-round-{round_index}.json."
        )
    # Auto-pick the latest round file.
    candidates = sorted(artifact_dir.glob("search-loop-round-*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No search-loop-round-*.json fixtures in {artifact_dir}. "
            "Run the production search loop once to generate them."
        )
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


class SearchJudgeSessionAdapter:
    name = "search-judge"

    def build_messages(
        self,
        fixture: Dict[str, Any],
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
        loop_version: str = "v1",
    ) -> List[Dict[str, Any]]:
        reject_unsupported_variant(self.name, variant=variant)
        if loop_version == "v2":
            return build_search_loop_v2_messages(
                round_index=int(fixture.get("round_index", 0)),
                max_rounds=int(fixture.get("max_rounds", 3)),
                is_final_round=bool(fixture.get("is_final_round", False)),
                background=fixture.get("background", ""),
                contract_json=fixture.get("contract_json", ""),
                executed_queries=fixture.get("executed_queries", []),
                previous_evidence_pack=fixture.get("previous_evidence_pack", fixture.get("progress_log", "")),
                search_results=fixture.get("search_results", ""),
                streamer_index=fixture.get("streamer_index", ""),
                common_index=fixture.get("common_index", ""),
                knowledge_entries=fixture.get("knowledge_entries", ""),
                previous_requested_entries=fixture.get("previous_requested_entries", []),
                previous_kept_entries=fixture.get("previous_kept_entries", []),
                previous_contract_json=fixture.get("previous_contract_json", ""),
                previous_search_queries=fixture.get("previous_search_queries", []),
                previous_extract_urls=fixture.get("previous_extract_urls", []),
                followup_query_cap=int(fixture.get("followup_query_cap", 4)),
            )
        return build_search_loop_messages(
            round_index=int(fixture.get("round_index", 0)),
            max_rounds=int(fixture.get("max_rounds", 3)),
            is_final_round=bool(fixture.get("is_final_round", False)),
            background=fixture.get("background", ""),
            contract_json=fixture.get("contract_json", ""),
            executed_queries=fixture.get("executed_queries", []),
            progress_log=fixture.get("progress_log", ""),
            search_results=fixture.get("search_results", ""),
            streamer_index=fixture.get("streamer_index", ""),
            common_index=fixture.get("common_index", ""),
            knowledge_entries=fixture.get("knowledge_entries", ""),
            previous_requested_entries=fixture.get("previous_requested_entries", []),
            previous_kept_entries=fixture.get("previous_kept_entries", []),
            previous_contract_json=fixture.get("previous_contract_json", ""),
            previous_search_queries=fixture.get("previous_search_queries", []),
            previous_extract_urls=fixture.get("previous_extract_urls", []),
            followup_query_cap=int(fixture.get("followup_query_cap", 4)),
        )

    def validate_reply(self, content: str) -> List[str]:
        return validate_session_contract(content, "search_loop")

    def run(
        self,
        *,
        run: Path,
        chunk_id: str = "",
        out_dir: Path,
        n: int = 3,
        max_attempts: int = 9,
        label: str = "baseline",
        note: str = "",
        dry_run: bool = False,
        test_profile: bool = False,
        force_extract: bool = False,
        thinking_level: str | None = None,
        temperature: float = 1.0,
        round_index: int | None = None,
        variant: str | None = None,
        force_tier: str | None = None,
        loop_version: str = "v1",
        **_kwargs: Any,
    ) -> ReplayResult:
        reject_unsupported_variant(self.name, variant=variant, force_tier=force_tier)
        from ..fixture import resolve_run_layout

        layout = resolve_run_layout(run)
        fixture = _load_judge_fixture(layout["artifact_dir"], round_index)
        messages = self.build_messages(fixture, loop_version=loop_version)
        return run_text_replay(
            session_name="search-judge",
            messages=messages,
            validate_reply=self.validate_reply,
            out_dir=out_dir,
            n=n,
            max_attempts=max_attempts,
            label=label,
            note=note,
            dry_run=dry_run,
            test_profile=test_profile,
            temperature=temperature,
            thinking_level=thinking_level,
            role=LLMRole.LIGHTWEIGHT,
        )
