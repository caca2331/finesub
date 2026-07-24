"""Fast round-1 (query+research fused) session adapter for prompt iteration.

Freezes the fast-mode round-1 inputs and replays the fused query+research call.
The fixture is extracted from ``fast-round-input.json`` (dumped by
``fast_session.py`` since the 补中间态 change).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from llm.config import CapabilityTier, LLMRole
from llm.profiles import DEFAULT_PROFILE
from llm.prompts import build_fast_round1_messages
from .base import (
    ReplayResult,
    reject_unsupported_variant,
    run_text_replay,
    validate_session_contract,
)


def _load_fast_fixture(artifact_dir: Path) -> Dict[str, Any]:
    """Load the fast round-1 fixture from the artifact directory."""

    path = artifact_dir / "fast-round-input.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No fast-round-input.json in {artifact_dir}. "
        "Run the production fast session once to generate it."
    )


class FastRound1SessionAdapter:
    name = "fast-round1"

    def build_messages(
        self,
        fixture: Dict[str, Any],
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
    ) -> List[Dict[str, Any]]:
        reject_unsupported_variant(self.name, variant=variant)
        # The fast round needs a SubtitleWindow; for replay we reconstruct a
        # minimal one from the fixture metadata + stable.json. For text-only
        # replay (no media), we pass a placeholder window with the CSV text.
        from llm.chunking import CorrectionBudget, SubtitleWindow, SubtitleSegment

        window_meta = fixture.get("window", {})
        # Build a minimal window for prompt assembly. The actual segments come
        # from stable.json in production; for replay the frozen CSV is embedded
        # in the fixture's note_url_extracts or we reconstruct from stable.
        segments = [
            SubtitleSegment(
                id=i,
                start=window_meta.get("clip_start", 0.0),
                end=window_meta.get("clip_end", 0.0),
                text=f"[replay placeholder segment {i}]",
            )
            for i in range(max(1, window_meta.get("segment_count", 1)))
        ]
        window = SubtitleWindow(
            chunk_id=window_meta.get("chunk_id", "fast"),
            segments=segments,
            overlap_segments=[],
            boundary_reason="replay",
            budget=CorrectionBudget(
                max_input_tokens=194_000,
                max_output_tokens=65_536,
                safety_margin=1_000,
            ),
        )
        return build_fast_round1_messages(
            window=window,
            audio_file_label=fixture.get("audio_file_label", ""),
            extra_info=fixture.get("extra_info", ""),
            note_url_extracts=fixture.get("note_url_extracts", ""),
            streamer_index=fixture.get("streamer_index", ""),
            common_index=fixture.get("common_index", ""),
            preinjected_entries=fixture.get("preinjected_entries", ""),
            max_search_queries=int(fixture.get("max_search_queries", 8)),
            use_search_contract=bool(fixture.get("use_search_contract", False)),
            collect_task_feedback=bool(fixture.get("collect_task_feedback", False)),
            profile=DEFAULT_PROFILE,
        )

    def validate_reply(self, content: str) -> List[str]:
        return validate_session_contract(content, "fast_round1")

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
        fixture = _load_fast_fixture(layout["artifact_dir"])
        messages = self.build_messages(fixture)
        return run_text_replay(
            session_name="fast-round1",
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
