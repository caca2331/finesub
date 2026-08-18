"""Research R2 where the model drives retrieval and the harness executes it.

A third retrieval arm beside ``retrieval=local`` (harness picks the queries)
and ``retrieval=native`` (agy searches for itself). The prompt is the *native*
shape -- no injected evidence pack, because the model is expected to go and
find things -- with the tool protocol from ``agent_tools`` appended, and the
dispatch deliberately does **not** entitle agy's own search: the model must ask
the harness, so every query and URL lands in the ledger.

See ``agent_tools`` for why the tool is a prompt protocol rather than a
registered function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from finesub.llm.routing.config import CapabilityTier, LLMRole
from ..agent_tools import (
    DEFAULT_MAX_CALLS_PER_ROUND,
    DEFAULT_MAX_ROUNDS,
    TOOL_PROTOCOL_TEXT,
    ToolRound,
    append_ledger,
    execute_tool_calls,
    ledger_urls,
    parse_tool_calls,
)
from .base import (
    ReplayResult,
    ReplaySample,
    replay_seed,
    replay_temperature,
    reject_unsupported_variant,
    sample_call_meta,
    _write_summary,
)
from .research import ResearchR2SessionAdapter, _load_research_fixture


class ResearchR2AgentSearchAdapter:
    """R2 with model-chosen, harness-executed retrieval."""

    name = "research-r2-tools"

    def build_messages(
        self, fixture: Dict[str, Any], *, tier: CapabilityTier = CapabilityTier.CAPABLE
    ) -> List[Dict[str, Any]]:
        # Native shape: no injected pack, no keep_entries. The model is told to
        # retrieve; only the route to the web differs from the native arm.
        messages = ResearchR2SessionAdapter().build_messages(
            fixture, retrieval="native"
        )
        out: List[Dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "user":
                message = {
                    **message,
                    "content": f"{message.get('content', '')}\n\n{TOOL_PROTOCOL_TEXT}",
                }
            out.append(message)
        return out

    def validate_reply(self, content: str, fixture: Dict[str, Any] | None = None):
        return ResearchR2SessionAdapter().validate_reply(
            content, fixture or {}, "native"
        )

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
        thinking_level: str | None = None,
        temperature: float = 1.0,
        variant: str | None = None,
        force_tier: str | None = None,
        **_kwargs: Any,
    ) -> ReplayResult:
        reject_unsupported_variant(self.name, variant=variant, force_tier=force_tier)
        from finesub.llm.client import RoleClient
        from ..fixture import resolve_run_layout

        layout = resolve_run_layout(run)
        fixture = _load_research_fixture(
            layout["artifact_dir"], "round2", research_context=layout["research_context"]
        )
        messages = self.build_messages(fixture)

        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        system_text = next(
            (str(m.get("content") or "") for m in messages if m.get("role") == "system"),
            "",
        )
        user_text = next(
            (str(m.get("content") or "") for m in messages if m.get("role") == "user"), ""
        )
        (out_dir / "prompt.system.txt").write_text(system_text, encoding="utf-8")
        (out_dir / "prompt.user.txt").write_text(user_text, encoding="utf-8")

        successes: List[ReplaySample] = []
        failures: List[ReplaySample] = []
        if dry_run:
            summary_path = _write_summary(
                out_dir, self.name, label, note or "dry-run（未调用 API）",
                successes, failures, temperature, dry_run=True,
            )
            return ReplayResult(
                out_dir=out_dir,
                prompt_system_path=out_dir / "prompt.system.txt",
                prompt_user_path=out_dir / "prompt.user.txt",
                successes=successes,
                failures=failures,
                summary_path=summary_path,
                dry_run=True,
            )

        client = RoleClient(test_profile=test_profile)
        search_client = _search_client()
        thinking_kwargs: Dict[str, Any] = {}
        if thinking_level:
            thinking_kwargs["thinking_level"] = thinking_level

        sample_idx = 0
        attempt = 0
        while sample_idx < n and attempt < max_attempts:
            attempt += 1
            temp = replay_temperature(temperature)
            turn = list(messages)
            rounds: List[ToolRound] = []
            content = ""
            call = None
            try:
                for round_index in range(DEFAULT_MAX_ROUNDS + 1):
                    call = client.complete(
                        LLMRole.GENERAL_CAPABLE,
                        turn,
                        temperature=temp,
                        seed=replay_seed(attempt) + round_index,
                        # Never entitle agy's own search: the whole point is
                        # that retrieval goes through the ledger.
                        native_search=False,
                        **thinking_kwargs,
                        **({"max_tokens": 65_536}),
                    )
                    content = call.content
                    calls = parse_tool_calls(
                        content, max_calls=DEFAULT_MAX_CALLS_PER_ROUND
                    )
                    if not calls or round_index >= DEFAULT_MAX_ROUNDS:
                        break
                    executed = execute_tool_calls(calls, search_client)
                    executed.index = round_index
                    rounds.append(executed)
                    turn = turn + [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": executed.rendered},
                    ]
            except Exception as exc:
                failures.append(ReplaySample(
                    ok=False, index=sample_idx, attempt=attempt,
                    content=f"[call error] {type(exc).__name__}: {exc}",
                    temperature=temp,
                    validation_errors=[f"API call failed: {exc}"],
                ))
                continue

            errors = list(self.validate_reply(content, fixture))
            # A reply that still asks for tools after the round cap never
            # produced an answer; failing it here keeps a truncated
            # conversation from being scored as a sample.
            if parse_tool_calls(content):
                errors.append(
                    f"reply still requests tools after {DEFAULT_MAX_ROUNDS} rounds"
                )
            meta = sample_call_meta(call) if call is not None else {}
            meta["tool_rounds"] = len(rounds)
            meta["tool_urls"] = len(ledger_urls(rounds))
            sample = ReplaySample(
                ok=not errors, index=sample_idx, attempt=attempt, content=content,
                temperature=temp, validation_errors=errors,
                model=meta.get("model", ""), call_meta=meta,
            )
            if not errors:
                sample_idx += 1
                path = out_dir / f"reply-{sample_idx:02d}.md"
                path.write_text(content, encoding="utf-8")
                sample.path = path
                append_ledger(out_dir, rounds, sample_idx)
                successes.append(sample)
            else:
                failures.append(sample)

        summary_path = _write_summary(
            out_dir, self.name, label, note, successes, failures, temperature,
            dry_run=False,
        )
        return ReplayResult(
            out_dir=out_dir,
            prompt_system_path=out_dir / "prompt.system.txt",
            prompt_user_path=out_dir / "prompt.user.txt",
            successes=successes,
            failures=failures,
            summary_path=summary_path,
            dry_run=False,
        )


def _search_client():
    from finesub.llm.web_search import WebSearchClient

    return WebSearchClient()
