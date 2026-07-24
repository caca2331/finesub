"""Research round (R1/R2) session adapters for prompt iteration.

Each round is replayed independently — R1 emits queries/notes, R2 emits the
background context. Fixtures are extracted from research-stage artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from llm.config import CapabilityTier, LLMRole
from llm.prompts import build_research_round1_messages, build_research_round2_messages
from .base import (
    ReplayResult,
    reject_unsupported_variant,
    run_text_replay,
    validate_session_contract,
)


# ---------------------------------------------------------------------------
# Fixture loading (from research-stage artifacts)
# ---------------------------------------------------------------------------


def _load_research_fixture(
    artifact_dir: Path,
    round_name: str,
    *,
    research_context: Path | None = None,
) -> Dict[str, Any]:
    """Load a research round fixture from the artifact directory.

    Looks for ``research-round{1,2}-input.json`` (dumped by the harness) or
    falls back to the research context JSON.
    """

    direct = artifact_dir / f"research-{round_name}-input.json"
    if direct.exists():
        return json.loads(direct.read_text(encoding="utf-8"))
    # Fallback: build from the research context file.
    context_files = (
        [research_context]
        if research_context is not None and research_context.exists()
        else sorted(artifact_dir.glob("*-research-context.json"))
        or sorted(artifact_dir.parent.glob("*-research-context.json"))
    )
    if context_files:
        ctx = json.loads(context_files[0].read_text(encoding="utf-8"))
        return {"_from_context": True, **ctx}
    raise FileNotFoundError(
        f"No research fixture found in {artifact_dir}. "
        f"Expected research-{round_name}-input.json or *-research-context.json."
    )


# ---------------------------------------------------------------------------
# Research R1
# ---------------------------------------------------------------------------

class ResearchR1SessionAdapter:
    name = "research-r1"

    def build_messages(
        self,
        fixture: Dict[str, Any],
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
    ) -> List[Dict[str, Any]]:
        reject_unsupported_variant(self.name, variant=variant)
        return build_research_round1_messages(
            transcript=fixture.get("transcript", ""),
            extra_info=fixture.get("extra_info", ""),
            note_url_extracts=fixture.get("note_url_extracts", ""),
            streamer_index=fixture.get("streamer_index", ""),
            common_index=fixture.get("common_index", ""),
            preinjected_entries=fixture.get("preinjected_entries", ""),
            max_search_queries=int(fixture.get("max_search_queries", 8)),
            use_search_contract=bool(fixture.get("use_search_contract", False)),
        )

    def validate_reply(self, content: str) -> List[str]:
        return validate_session_contract(content, "research_round1")

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
        variant: str | None = None,
        force_tier: str | None = None,
        **_kwargs: Any,
    ) -> ReplayResult:
        reject_unsupported_variant(self.name, variant=variant, force_tier=force_tier)
        from ..fixture import resolve_run_layout

        layout = resolve_run_layout(run)
        fixture = _load_research_fixture(
            layout["artifact_dir"],
            "round1",
            research_context=layout["research_context"],
        )
        messages = self.build_messages(fixture)
        return run_text_replay(
            session_name="research-r1",
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
            role=LLMRole.GENERAL_CAPABLE,
        )


# ---------------------------------------------------------------------------
# Research R2
# ---------------------------------------------------------------------------

class ResearchR2SessionAdapter:
    name = "research-r2"

    def build_messages(
        self,
        fixture: Dict[str, Any],
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
    ) -> List[Dict[str, Any]]:
        reject_unsupported_variant(self.name, variant=variant)
        return build_research_round2_messages(
            transcript=fixture.get("transcript", ""),
            extra_info=fixture.get("extra_info", ""),
            round1_notes=fixture.get("round1_notes", ""),
            entry_details_text=fixture.get("entry_details_text", ""),
            search_results=fixture.get("search_results", ""),
            use_evidence_pack=bool(fixture.get("use_evidence_pack", False)),
            collect_task_feedback=bool(fixture.get("collect_task_feedback", False)),
        )

    def validate_reply(self, content: str) -> List[str]:
        return validate_session_contract(content, "research_round2")

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
        variant: str | None = None,
        force_tier: str | None = None,
        **_kwargs: Any,
    ) -> ReplayResult:
        reject_unsupported_variant(self.name, variant=variant, force_tier=force_tier)
        from ..fixture import resolve_run_layout

        layout = resolve_run_layout(run)
        fixture = _load_research_fixture(
            layout["artifact_dir"],
            "round2",
            research_context=layout["research_context"],
        )
        messages = self.build_messages(fixture)
        return run_text_replay(
            session_name="research-r2",
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
            role=LLMRole.GENERAL_CAPABLE,
        )
