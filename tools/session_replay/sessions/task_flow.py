"""Task-flow session adapters (docs/llm_harness_research.md, 「准备任务」).

Two replays of the opt-in flow's *first* messages:

* ``prepare-task``: the preparation task's opening (``research-prepare-input.json``
  is the exact builder input the harness dumped). A reply is either a request
  batch or a delivery; both shapes validate.
* ``correction-request``: a window task's opening with the request channel
  (reuses the correction fixture). A reply is either a request batch or the
  window's delivery (CSV, validated by ``output_protocol`` like production).

Neither replays the fill turn: that needs the harness to execute the batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from finesub.llm.output_protocol import validate_correction_window_output
from finesub.llm.output_tags import find_top_level_tag_blocks
from finesub.llm.prepare_prompts import build_prepare_task_messages
from finesub.llm.prompt_variants import DEFAULT_VARIANT_FOR_TIER, resolve_variant
from finesub.llm.prompts import ContextPack
from finesub.llm.request_prompts import build_window_task_messages
from finesub.llm.routing.config import CapabilityTier, LLMRole
from finesub.llm.session_contract import SESSION_CONTRACTS, correction_request_contract
from finesub.llm.stages.task_graph import window_requests
from finesub.llm.stages.correction import correction_role_for_profile
from ..fixture import (
    CorrectionFixture,
    apply_profile_override,
    build_window_from_fixture,
    ensure_correction_fixture,
    indexes_from_fixture,
    resolve_run_layout,
)
from .base import ReplayResult, reject_unsupported_variant, run_text_replay

_PREPARE_BUILDER_KEYS = (
    "transcript",
    "extra_info",
    "note_url_extracts",
    "knowledge_indexes",
    "preinjected_entries",
    "max_search_queries",
    "emits_queries",
    "emits_entries",
    "native_search",
    "max_requested_entries",
    "collect_task_feedback",
    "emits_keep",
    "agentic",
    "style_catalog",
    "extra_style",
    "earlier_chunks",
)


def _load_prepare_fixture(artifact_dir: Path) -> Dict[str, Any]:
    path = artifact_dir / "research-prepare-input.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No preparation fixture in {artifact_dir}: expected research-prepare-input.json "
            "(dumped by every run since stage D)"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class PrepareTaskSessionAdapter:
    name = "prepare-task"

    def build_messages(
        self,
        fixture: Dict[str, Any],
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
    ) -> List[Dict[str, Any]]:
        reject_unsupported_variant(self.name, variant=variant)
        kwargs = {key: fixture[key] for key in _PREPARE_BUILDER_KEYS if key in fixture}
        if "knowledge_indexes" in kwargs and not kwargs["knowledge_indexes"]:
            kwargs["knowledge_indexes"] = indexes_from_fixture(fixture)
        return build_prepare_task_messages(**kwargs)

    def validate_reply(self, content: str) -> List[str]:
        """A delivery or a request batch -- whichever the reply is."""

        if find_top_level_tag_blocks(content, "context_pack"):
            return SESSION_CONTRACTS["prepare_task_delivery"].validate(content)
        return SESSION_CONTRACTS["prepare_task_request"].validate(content)

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
        model: str | None = None,
        variant: str | None = None,
        force_tier: str | None = None,
        agentic: bool = False,
        **_kwargs: Any,
    ) -> ReplayResult:
        reject_unsupported_variant(self.name, variant=variant, force_tier=force_tier)
        layout = resolve_run_layout(run)
        fixture = _load_prepare_fixture(layout["artifact_dir"])
        if agentic:
            # The dump recorded the strategy the run resolved; the flag replays
            # the tool-session shape of the same opening.
            fixture = {**fixture, "agentic": True}
        return run_text_replay(
            session_name="prepare-task",
            messages=self.build_messages(fixture),
            validate_reply=self.validate_reply,
            out_dir=out_dir,
            n=n,
            max_attempts=max_attempts,
            label=label,
            note=note,
            dry_run=dry_run,
            test_profile=test_profile,
            temperature=temperature,
            model=model,
            thinking_level=thinking_level,
            role=LLMRole.GENERAL_CAPABLE,
            native_search=bool(fixture.get("native_search")),
        )


class WindowRequestSessionAdapter:
    name = "correction-request"

    def ensure_fixture(
        self, *, run: Path, chunk_id: str, force_extract: bool = False, **_kwargs: Any
    ) -> tuple[CorrectionFixture, Path]:
        return ensure_correction_fixture(run=run, chunk_id=chunk_id, force_extract=force_extract)

    def build_messages(
        self,
        fixture: CorrectionFixture,
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
        agentic: bool = False,
    ) -> List[Dict[str, Any]]:
        profile = fixture.profile()
        window = build_window_from_fixture(fixture)
        indexes = indexes_from_fixture(fixture.context_pack or {})
        # The channel follows the grant like production (`task_graph.window_flow`):
        # harness search under ``retrieval=local``, entry requests wherever the
        # fixture carries a knowledge index.
        readable = any((text or "").strip() for text in indexes.values())
        requests = window_requests(profile, knowledge_readable=readable)
        return build_window_task_messages(
            request_channel=requests and not agentic,
            agentic=agentic and requests,
            knowledge_indexes=indexes,
            carried_entries=fixture.entry_details,
            window=window,
            context_pack=ContextPack.from_dict(fixture.context_pack).with_source_order(
                [segment.id for segment in window.segments]
            ),
            previous_advice=fixture.previous_advice,
            entry_details=fixture.entry_details,
            extra_style=fixture.extra_style,
            style_block=fixture.style_block,
            task_update_feedback=fixture.task_update_feedback,
            profile=profile,
            knowledge_enabled=readable,
            tier=tier,
            variant=variant,
        )

    def validate_reply(self, content: str, fixture: CorrectionFixture, *, variant: str | None = None) -> List[str]:
        if find_top_level_tag_blocks(content, "translated"):
            window = build_window_from_fixture(fixture)
            return list(
                validate_correction_window_output(
                    content, window, variant=resolve_variant(variant, CapabilityTier.CAPABLE)
                ).errors
            )
        return correction_request_contract(
            search=fixture.profile().external_injection
        ).validate(content)

    def run(
        self,
        *,
        run: Path,
        chunk_id: str,
        out_dir: Path,
        n: int = 3,
        max_attempts: int = 9,
        label: str = "baseline",
        note: str = "",
        dry_run: bool = False,
        test_profile: bool = False,
        force_extract: bool = False,
        thinking_level: str | None = None,
        profile: str | None = None,
        temperature: float = 1.0,
        model: str | None = None,
        force_tier: str | None = None,
        variant: str | None = None,
        agentic: bool = False,
        **_kwargs: Any,
    ) -> ReplayResult:
        fixture, _ = self.ensure_fixture(run=run, chunk_id=chunk_id, force_extract=force_extract)
        fixture = apply_profile_override(fixture, profile)
        # Same rule as the correction adapter: --variant names a variant,
        # --force-tier names a tier and maps to its default variant; validated
        # eagerly so a typo never silently serves the default prompt.
        if variant is None and force_tier is not None:
            variant = DEFAULT_VARIANT_FOR_TIER[CapabilityTier(force_tier)]
        if variant is not None:
            resolve_variant(variant)
        messages = self.build_messages(fixture, variant=variant, agentic=agentic)
        return run_text_replay(
            session_name="correction-request",
            messages=messages,
            validate_reply=lambda content: self.validate_reply(content, fixture, variant=variant),
            out_dir=out_dir,
            n=n,
            max_attempts=max_attempts,
            label=label,
            note=note,
            dry_run=dry_run,
            test_profile=test_profile,
            temperature=temperature,
            model=model,
            thinking_level=thinking_level,
            role=correction_role_for_profile(fixture.profile()),
        )
