"""Shared protocol and validation helpers for replay session adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

from llm.client import (
    LiteLLMRoleClient,
    VALIDATION_BASE_TEMPERATURE,
    VALIDATION_TEMPERATURE_STEP,
    extract_token_distribution,
)
from llm.config import CapabilityTier


# ---------------------------------------------------------------------------
# Structural validation — a session's reply must satisfy its output contract
# (top-level sibling blocks, non-empty vs may-be-empty). The contract is the
# single source of truth, shared with production (llm.session_contract).
# ---------------------------------------------------------------------------


def validate_session_contract(content: str, session_name: str) -> List[str]:
    """Return structural errors against the named session's output contract."""

    from llm.session_contract import SESSION_CONTRACTS

    return SESSION_CONTRACTS[session_name].validate(content)


def reject_unsupported_variant(
    session_name: str,
    *,
    variant: str | None = None,
    force_tier: str | None = None,
) -> None:
    """Fail loudly if a prompt-variant override is requested for a round that
    has no variant set registered yet.

    The named-variant system (``llm.prompt_variants``) is correction-CSV
    specific: each variant bundles merge fragments and ``<singles>``
    requirements that only the correction round emits. The query / research /
    judge / fast rounds each have a single fixed prompt, so ``--variant`` /
    ``--force-tier`` cannot select anything for them. Silently serving the
    baseline would make an A/B run look like it varied when it did not, so we
    raise instead — mirroring ``resolve_variant``'s hard error on an unknown
    name. Adding variants for one of these rounds means registering a per-round
    variant set and threading it through that round's builder.
    """

    requested = variant or force_tier
    if requested:
        raise NotImplementedError(
            f"session '{session_name}' has no registered prompt variants; "
            f"--variant/--force-tier apply only to the 'correction' round "
            f"(got {requested!r}). See docs/session_replay.md for how to add "
            f"a per-round variant set."
        )


def pin_client_role_to_free_model(client: Any, role: Any, requested: str) -> str:
    """Pin one role to exactly one FREE model and return its LiteLLM id.

    Exact short ids (for example ``3.5-flash``) take precedence over substring
    matching so they cannot accidentally include ``3.5-flash-lite``.
    """

    from llm.config import ModelEndpoint
    from llm.model_catalog import default_model_catalog

    needle = requested.strip().lower()
    if not needle:
        raise ValueError("--model cannot be empty")

    def canonical(model_id: str) -> str:
        return model_id.lower().removeprefix("gemini/").removeprefix("gemini-")

    canonical_needle = canonical(needle)

    def is_exact(model_id: str) -> bool:
        return canonical(model_id) == canonical_needle

    base_config = client.role_configs[role]
    role_free = [
        ep for ep in base_config.endpoint_chain if "FREE" in ep.provider_tier
    ]
    exact = [ep for ep in role_free if is_exact(ep.litellm_model)]
    if exact:
        selected = exact[0]
    else:
        catalog_free = [
            entry
            for entry in default_model_catalog()
            if "FREE" in entry.provider_tier
        ]
        exact_catalog = [e for e in catalog_free if is_exact(e.litellm_model)]
        if exact_catalog:
            entry = exact_catalog[0]
            selected = ModelEndpoint(entry.provider_tier, entry.litellm_model)
        else:
            fuzzy = {
                e.litellm_model: e
                for e in catalog_free
                if needle in e.litellm_model.lower()
            }
            if not fuzzy:
                available = sorted(e.litellm_model for e in catalog_free)
                raise RuntimeError(
                    f"--model '{requested}' matches no FREE model: {available}"
                )
            if len(fuzzy) > 1:
                matches = sorted(fuzzy)
                raise RuntimeError(
                    f"--model '{requested}' is ambiguous; use one exact model id: "
                    f"{matches}"
                )
            entry = next(iter(fuzzy.values()))
            selected = ModelEndpoint(entry.provider_tier, entry.litellm_model)

    client.role_configs[role] = replace(base_config, endpoint_chain=(selected,))
    return selected.litellm_model


# ---------------------------------------------------------------------------
# Sample result (shared across all sessions)
# ---------------------------------------------------------------------------

_USAGE_KEYS = (
    "prompt_tokens",
    "total_input_tokens",
    "thinking_tokens",
    "output_tokens",
    "total_output_tokens",
    "total_tokens",
)


@dataclass
class ReplaySample:
    ok: bool
    index: int
    attempt: int
    content: str
    temperature: float = VALIDATION_BASE_TEMPERATURE
    validation_errors: List[str] = field(default_factory=list)
    model: str = ""
    path: Path | None = None
    call_meta: Dict[str, Any] = field(default_factory=dict)


def sample_call_meta(call: Any) -> Dict[str, Any]:
    """Extract durable call metadata from an LLMCallResult."""

    raw = getattr(call, "raw_response", None) or {}
    dist = extract_token_distribution(raw)
    usage = {key: int(dist.get(key) or 0) for key in _USAGE_KEYS}
    return {
        "model": str(getattr(call, "model", "") or ""),
        "api_key_label": str(getattr(call, "api_key_label", "") or ""),
        "thinking_level": str(getattr(call, "thinking_level", "") or ""),
        "capability_tier": str(
            getattr(getattr(call, "capability_tier", None), "value", "") or ""
        ),
        "fallback_used": bool(getattr(call, "fallback_used", False)),
        "usage": usage,
    }


def replay_temperature(base_temperature: float, attempt: int) -> float:
    """Temperature for a one-based replay attempt, decreasing every call."""

    return max(
        0.0,
        round(
            float(base_temperature)
            - VALIDATION_TEMPERATURE_STEP * max(0, int(attempt) - 1),
            2,
        ),
    )


# ---------------------------------------------------------------------------
# Session protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ReplaySession(Protocol):
    """Protocol that every replay session adapter implements."""

    name: str

    def build_messages(
        self,
        fixture: Any,
        *,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Assemble the prompt messages from frozen fixture inputs."""
        ...

    def validate_reply(self, content: str) -> List[str]:
        """Return structural validation errors (empty = pass)."""
        ...

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
        temperature: float = VALIDATION_BASE_TEMPERATURE,
        model: str | None = None,
        force_tier: str | None = None,
        variant: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the replay loop and return a result object."""
        ...


# ---------------------------------------------------------------------------
# Generic replay loop (shared by non-media sessions)
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    """Generic replay result for text-only sessions."""

    out_dir: Path
    prompt_system_path: Path
    prompt_user_path: Path
    successes: List[ReplaySample]
    failures: List[ReplaySample]
    summary_path: Path
    dry_run: bool


def run_text_replay(
    *,
    session_name: str,
    messages: List[Dict[str, Any]],
    validate_reply: Any,
    out_dir: Path,
    n: int = 3,
    max_attempts: int = 9,
    label: str = "baseline",
    note: str = "",
    dry_run: bool = False,
    test_profile: bool = False,
    temperature: float = VALIDATION_BASE_TEMPERATURE,
    model: str | None = None,
    thinking_level: str | None = None,
    role: Any = None,
) -> ReplayResult:
    """Generic replay loop for text-only sessions (no media upload).

    Builds prompt dumps, optionally calls the API in a validation-gated loop,
    and writes per-sample replies + a summary.
    """

    from llm.config import LLMRole

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dump prompts.
    system_text = user_text = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_text = str(msg.get("content") or "")
        elif msg.get("role") == "user":
            user_text = str(msg.get("content") or "")
    prompt_system_path = out_dir / "prompt.system.txt"
    prompt_user_path = out_dir / "prompt.user.txt"
    prompt_system_path.write_text(system_text, encoding="utf-8")
    prompt_user_path.write_text(user_text, encoding="utf-8")

    successes: List[ReplaySample] = []
    failures: List[ReplaySample] = []

    if dry_run:
        summary_path = _write_summary(
            out_dir, session_name, label, note or "dry-run（未调用 API）",
            successes, failures, temperature, dry_run=True,
        )
        return ReplayResult(
            out_dir=out_dir,
            prompt_system_path=prompt_system_path,
            prompt_user_path=prompt_user_path,
            successes=successes,
            failures=failures,
            summary_path=summary_path,
            dry_run=True,
        )

    client = LiteLLMRoleClient(test_profile=test_profile)
    call_role = role or LLMRole.LIGHTWEIGHT
    if model:
        pin_client_role_to_free_model(client, call_role, model)
    thinking_kwargs: Dict[str, Any] = {}
    if thinking_level:
        thinking_kwargs["thinking_level"] = thinking_level

    sample_idx = 0
    attempt = 0
    while sample_idx < n and attempt < max_attempts:
        attempt += 1
        temp = replay_temperature(temperature, attempt)
        try:
            call = client.complete(
                call_role,
                messages,
                temperature=temp,
                **thinking_kwargs,
                **({"max_tokens": 65_536}),
            )
        except Exception as exc:
            failures.append(ReplaySample(
                ok=False, index=sample_idx, attempt=attempt,
                content=f"[call error] {type(exc).__name__}: {exc}",
                temperature=temp,
                validation_errors=[f"API call failed: {exc}"],
            ))
            continue

        content = call.content
        errors = validate_reply(content)
        ok = len(errors) == 0
        meta = sample_call_meta(call)

        sample = ReplaySample(
            ok=ok,
            index=sample_idx,
            attempt=attempt,
            content=content,
            temperature=temp,
            validation_errors=errors,
            model=meta.get("model", ""),
            call_meta=meta,
        )

        if ok:
            sample_idx += 1
            path = out_dir / f"reply-{sample_idx:02d}.md"
            path.write_text(content, encoding="utf-8")
            sample.path = path
            successes.append(sample)
        else:
            failures.append(sample)

    summary_path = _write_summary(
        out_dir, session_name, label, note,
        successes, failures, temperature, dry_run=False,
    )
    return ReplayResult(
        out_dir=out_dir,
        prompt_system_path=prompt_system_path,
        prompt_user_path=prompt_user_path,
        successes=successes,
        failures=failures,
        summary_path=summary_path,
        dry_run=False,
    )


def _write_summary(
    out_dir: Path,
    session_name: str,
    label: str,
    note: str,
    successes: List[ReplaySample],
    failures: List[ReplaySample],
    base_temperature: float,
    *,
    dry_run: bool,
) -> Path:
    lines = [
        f"# {session_name} replay summary",
        "",
        f"- label: {label}",
        f"- time: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- base_temperature: {base_temperature}",
        f"- dry_run: {dry_run}",
        f"- successes: {len(successes)}",
        f"- failures: {len(failures)}",
    ]
    if note:
        lines.append(f"- note: {note}")
    if successes:
        lines.append("")
        lines.append("## Usage")
        for s in successes:
            usage = s.call_meta.get("usage", {})
            lines.append(
                f"- reply-{s.index:02d}: model={s.model} "
                f"in={usage.get('total_input_tokens', 0)} "
                f"out={usage.get('total_output_tokens', 0)} "
                f"think={usage.get('thinking_tokens', 0)}"
            )
    if failures:
        lines.append("")
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- attempt {f.attempt}: {'; '.join(f.validation_errors)}")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
