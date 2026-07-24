"""Query round session adapter for prompt iteration.

Reuses the correction fixture (shared window + context) and replays
``build_correction_query_messages`` — the per-window query round that emits
search queries, window notes, and entry requests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from llm.config import CapabilityTier, LLMRole
from llm.prompts import ContextPack, build_correction_query_messages
from ..fixture import (
    CorrectionFixture,
    apply_profile_override,
    build_window_from_fixture,
    ensure_correction_fixture,
    extract_indexes_from_exchange,
    find_query_exchange,
    fixture_path,
    resolve_run_layout,
)
from .base import (
    ReplayResult,
    reject_unsupported_variant,
    run_text_replay,
    validate_session_contract,
)


class QuerySessionAdapter:
    name = "query"

    def ensure_fixture(
        self,
        *,
        run: Path,
        chunk_id: str,
        force_extract: bool = False,
        **_kwargs: Any,
    ) -> tuple[CorrectionFixture, Path]:
        """Load the correction fixture (query shares the same frozen inputs)."""

        layout = resolve_run_layout(run)
        path = fixture_path(layout["artifact_dir"], chunk_id)
        if path.exists() and not force_extract:
            from ..fixture import load_fixture

            fixture = load_fixture(path)
            # Backward compat: older fixtures lack indexes in context_pack.
            if "streamer_index" not in fixture.context_pack:
                query_exch = find_query_exchange(layout["artifact_dir"], chunk_id)
                if query_exch:
                    for key, value in extract_indexes_from_exchange(query_exch).items():
                        fixture.context_pack.setdefault(key, value)
            return fixture, path
        return ensure_correction_fixture(
            run=run, chunk_id=chunk_id, force_extract=force_extract
        )

    def build_messages(
        self,
        fixture: CorrectionFixture,
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
    ) -> List[Dict[str, Any]]:
        reject_unsupported_variant(self.name, variant=variant)
        profile = fixture.profile()
        window = build_window_from_fixture(fixture)
        return build_correction_query_messages(
            window=window,
            context_pack=ContextPack.from_dict(fixture.context_pack),
            previous_advice=fixture.previous_advice,
            streamer_index=fixture.context_pack.get("streamer_index", "")
            if fixture.context_pack
            else "",
            common_index=fixture.context_pack.get("common_index", "")
            if fixture.context_pack
            else "",
            carried_entries=fixture.entry_details,
            profile=profile,
        )

    def validate_reply(self, content: str) -> List[str]:
        return validate_session_contract(content, "query")

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
        **_kwargs: Any,
    ) -> ReplayResult:
        reject_unsupported_variant(self.name, variant=variant, force_tier=force_tier)
        fixture, _ = self.ensure_fixture(
            run=run, chunk_id=chunk_id, force_extract=force_extract
        )
        fixture = apply_profile_override(fixture, profile)
        messages = self.build_messages(fixture, variant=variant)
        return run_text_replay(
            session_name="query",
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
            model=model,
            thinking_level=thinking_level,
            role=LLMRole.LIGHTWEIGHT_MULTIMODAL,
        )
