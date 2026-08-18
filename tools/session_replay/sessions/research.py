"""Research round (R1/R2) session adapters for prompt iteration.

Each round is replayed independently — R1 emits queries/notes, R2 emits the
background context. Fixtures are extracted from research-stage artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from finesub.llm.routing.config import CapabilityTier, LLMRole
from finesub.llm.prompts import build_research_round1_messages, build_research_round2_messages
from .base import (
    replay_retrieval,
    replay_wants_native_search,
    ReplayResult,
    reject_unsupported_variant,
    run_text_replay,
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


def _r1_axis_flags(
    fixture: Dict[str, Any], retrieval: str = ""
) -> Dict[str, Any]:
    """R1's per-axis halves, defaulting to the ``retrieval=local`` full shape.

    Production dumps these into ``research-round1-input.json``, so a fixture
    frozen from a none/native run carries them; every fixture frozen before the
    round-structure split predates them and is local, which is what the
    defaults reproduce. ``validate_reply`` derives its contract from the same
    flags, so a none/native fixture is checked against the shape it was told to
    produce rather than the frozen local one.

    ``retrieval`` re-shapes a *local* fixture into another axis so the two can
    be compared from one frozen run. Production derives ``emits_queries`` from
    ``profile.external_injection`` (``retrieval == "local"``), so only a local
    run asks for search queries -- without this, a ``--profile retrieval=native``
    replay silently keeps the local prompt and only bolts the search tool onto
    the dispatch, and the two arms differ by nothing that matters.
    """

    flags = {
        "emits_queries": bool(fixture.get("emits_queries", True)),
        "emits_entries": bool(fixture.get("emits_entries", True)),
        "emits_notes": bool(fixture.get("emits_notes", True)),
        "max_requested_entries": int(fixture.get("max_requested_entries", 8)),
    }
    if retrieval:
        flags["emits_queries"] = retrieval == "local"
        flags["emits_notes"] = retrieval != "none"
    return flags


class ResearchR1SessionAdapter:
    name = "research-r1"

    def build_messages(
        self,
        fixture: Dict[str, Any],
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
        retrieval: str = "",
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
            **_r1_axis_flags(fixture, retrieval),
        )

    def validate_reply(
        self,
        content: str,
        fixture: Dict[str, Any] | None = None,
        retrieval: str = "",
    ) -> List[str]:
        """Validate against the shape this replay's switches actually asked for.

        A none/native fixture legitimately omits blocks the local contract
        demands; checking it against the frozen local shape would fail correct
        replies and burn the retry budget doing it.
        """

        from finesub.llm.session_contract import research_round1_contract

        flags = _r1_axis_flags(fixture or {}, retrieval)
        return research_round1_contract(
            emits_queries=flags["emits_queries"],
            emits_entries=flags["emits_entries"],
            emits_notes=flags["emits_notes"],
        ).validate(content)

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
        retrieval = replay_retrieval(fixture, _kwargs.get("profile"))
        messages = self.build_messages(fixture, retrieval=retrieval)
        return run_text_replay(
            session_name="research-r1",
            messages=messages,
            validate_reply=lambda content: self.validate_reply(
                content, fixture, retrieval
            ),
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
            native_search=replay_wants_native_search(
                fixture, _kwargs.get("profile")
            ),
        )


# ---------------------------------------------------------------------------
# Research R2
# ---------------------------------------------------------------------------

def _r2_axis_flags(fixture: Dict[str, Any], retrieval: str = "") -> Dict[str, Any]:
    """R2's retrieval-shaped inputs, defaulting to what the fixture froze.

    ``retrieval`` re-shapes a local fixture into another axis, mirroring what
    production does: only ``retrieval=local`` runs the harness's own search, so
    only it has an evidence pack to inject and only it emits ``keep_entries``
    (``round2_emits_keep = local_search`` in ``finesub.llm.research``). A native run
    reaches the same facts through the model's own tool and must therefore be
    given an *empty* injection -- handing it the local pack and calling it
    "native" compares two arms that saw the same evidence.
    """

    if not retrieval:
        return {
            "search_results": str(fixture.get("search_results", "") or ""),
            "use_evidence_pack": bool(fixture.get("use_evidence_pack", False)),
            "native_search": bool(fixture.get("native_search", False)),
            "emits_keep": bool(fixture.get("emits_keep", True)),
        }
    local = retrieval == "local"
    return {
        "search_results": str(fixture.get("search_results", "") or "") if local else "",
        "use_evidence_pack": bool(fixture.get("use_evidence_pack", False)) and local,
        "native_search": retrieval == "native",
        "emits_keep": local,
    }


class ResearchR2SessionAdapter:
    name = "research-r2"

    def build_messages(
        self,
        fixture: Dict[str, Any],
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
        retrieval: str = "",
    ) -> List[Dict[str, Any]]:
        reject_unsupported_variant(self.name, variant=variant)
        flags = _r2_axis_flags(fixture, retrieval)
        return build_research_round2_messages(
            transcript=fixture.get("transcript", ""),
            extra_info=fixture.get("extra_info", ""),
            round1_notes=fixture.get("round1_notes", ""),
            entry_details_text=fixture.get("entry_details_text", ""),
            search_results=flags["search_results"],
            use_evidence_pack=flags["use_evidence_pack"],
            collect_task_feedback=bool(fixture.get("collect_task_feedback", False)),
            native_search=flags["native_search"],
            emits_keep=flags["emits_keep"],
        )

    def validate_reply(
        self,
        content: str,
        fixture: Dict[str, Any] | None = None,
        retrieval: str = "",
    ) -> List[str]:
        """R2's contract likewise follows the switches this replay ran under."""

        from finesub.llm.session_contract import research_round2_contract

        return research_round2_contract(
            emits_keep=_r2_axis_flags(fixture or {}, retrieval)["emits_keep"]
        ).validate(content)

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
        retrieval = replay_retrieval(fixture, _kwargs.get("profile"))
        messages = self.build_messages(fixture, retrieval=retrieval)
        return run_text_replay(
            session_name="research-r2",
            messages=messages,
            validate_reply=lambda content: self.validate_reply(
                content, fixture, retrieval
            ),
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
            native_search=replay_wants_native_search(
                fixture, _kwargs.get("profile")
            ),
        )
