"""Shared protocol and validation helpers for replay session adapters."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable

from finesub.llm.client import (
    RoleClient,
    VALIDATION_BASE_TEMPERATURE,
    extract_token_distribution,
)
from finesub.llm.routing.config import CapabilityTier


# ---------------------------------------------------------------------------
# Structural validation — a session's reply must satisfy its output contract
# (top-level sibling blocks, non-empty vs may-be-empty). The contract is the
# single source of truth, shared with production (finesub.llm.session_contract).
# ---------------------------------------------------------------------------


def validate_session_contract(content: str, session_name: str) -> List[str]:
    """Return structural errors against the named session's output contract."""

    from finesub.llm.session_contract import SESSION_CONTRACTS

    return SESSION_CONTRACTS[session_name].validate(content)


def reject_unsupported_variant(
    session_name: str,
    *,
    variant: str | None = None,
    force_tier: str | None = None,
) -> None:
    """Fail loudly if a prompt-variant override is requested for a round that
    has no variant set registered yet.

    The named-variant system (``finesub.llm.prompt_variants``) is correction-CSV
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
    """Pin one role to exactly one model and return its API model id.

    FREE Gemini endpoints and **local-agent targets** are both pinnable: the
    free tier goes unavailable for hours at a time (2026-08-25: 503 across
    3.7-flash and 3.6-flash), and a prompt A/B that cannot be run is not an
    A/B. A local-agent pin is matched by catalog `fact_id` as well as
    `api_model_id`, because that is the name a person reads in the catalog,
    and it forces `agent_session_mode="api"` -- one fresh session per attempt
    is what makes attempts independent samples, which is the whole premise of
    counting successes out of tries.

    Exact short ids (for example ``3.5-flash``) take precedence over substring
    matching so they cannot accidentally include ``3.5-flash-lite``.
    """

    from finesub.llm.routing.config import ModelEndpoint
    from finesub.llm.routing.model_catalog import default_model_catalog

    needle = requested.strip().lower()
    if not needle:
        raise ValueError("--model cannot be empty")

    def canonical(model_id: str) -> str:
        return model_id.lower().removeprefix("gemini/").removeprefix("gemini-")

    canonical_needle = canonical(needle)

    def is_exact(model_id: str) -> bool:
        return canonical(model_id) == canonical_needle

    base_config = client.role_configs[role]
    agent_entries = {
        entry.fact_id: entry
        for entry in default_model_catalog()
        if entry.provider_kind == "local_agent"
    }
    agent_hit = next(
        (
            entry
            for entry in agent_entries.values()
            if needle in {entry.fact_id.lower(), entry.api_model_id.lower()}
        ),
        None,
    )
    if agent_hit is not None:
        # `backend` is what sends the call down the local-agent path instead
        # of the REST one; leaving it at the default routed an agy pin into
        # the Gemini transport, which then failed for want of an API key a
        # local agent never has.
        client.role_configs[role] = replace(
            base_config,
            endpoint_chain=(
                ModelEndpoint(
                    agent_hit.provider_tier,
                    agent_hit.api_model_id,
                    fact_id=agent_hit.fact_id,
                    backend="local_agent",
                ),
            ),
            model_group_id="",
            agent_session_mode="api",
        )
        return agent_hit.api_model_id

    role_free = [
        ep for ep in base_config.endpoint_chain if "FREE" in ep.provider_tier
    ]
    exact = [ep for ep in role_free if is_exact(ep.api_model_id)]
    if exact:
        selected = exact[0]
    else:
        catalog_free = [
            entry
            for entry in default_model_catalog()
            if "FREE" in entry.provider_tier
        ]
        exact_catalog = [e for e in catalog_free if is_exact(e.api_model_id)]
        if exact_catalog:
            entry = exact_catalog[0]
            selected = ModelEndpoint(entry.provider_tier, entry.api_model_id)
        else:
            fuzzy = {
                e.api_model_id: e
                for e in catalog_free
                if needle in e.api_model_id.lower()
            }
            if not fuzzy:
                available = sorted(e.api_model_id for e in catalog_free)
                raise RuntimeError(
                    f"--model '{requested}' matches no FREE model: {available}; "
                    f"local-agent targets are pinnable too, by fact id: "
                    f"{sorted(agent_entries)}"
                )
            if len(fuzzy) > 1:
                matches = sorted(fuzzy)
                raise RuntimeError(
                    f"--model '{requested}' is ambiguous; use one exact model id: "
                    f"{matches}"
                )
            entry = next(iter(fuzzy.values()))
            selected = ModelEndpoint(entry.provider_tier, entry.api_model_id)

    # A cell config normally carries a model_group_id, which would make the
    # router ignore endpoint_chain and expand the whole group again.  Pinning
    # is an adapter plan by definition: retain the cell's prompt/thinking
    # settings but clear group expansion and expose exactly one endpoint.
    client.role_configs[role] = replace(
        base_config,
        endpoint_chain=(selected,),
        model_group_id="",
    )
    return selected.api_model_id


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


def replay_retrieval(
    fixture: Any = None, profile_override: str | None = None
) -> str:
    """The retrieval axis this replay runs under ("" when nothing names one).

    The `--profile` override wins; otherwise the profile the fixture was
    captured under.
    """

    def _retrieval_of(text: str) -> str:
        for part in str(text or "").split(","):
            key, _, value = part.partition("=")
            if key.strip() == "retrieval":
                return value.strip()
        return ""

    if profile_override:
        return _retrieval_of(profile_override)
    profile_id = ""
    if isinstance(fixture, dict):
        profile_id = str(
            fixture.get("profile_id")
            or (fixture.get("profile") or {}).get("profile_id")
            or ""
        )
    return _retrieval_of(profile_id)


def replay_wants_native_search(
    fixture: Any = None, profile_override: str | None = None
) -> bool:
    """Whether this replay should dispatch with the model's own search tool.

    Read from the `--profile` override when there is one, otherwise from the
    profile the fixture was captured under. Getting this wrong is silent and
    one-directional: a `retrieval=native` prompt has no evidence pack, so a
    call dispatched without the tool answers from memory and the arm looks bad
    for a reason that has nothing to do with retrieval.
    """

    return replay_retrieval(fixture, profile_override) == "native"


def sample_call_meta(call: Any) -> Dict[str, Any]:
    """Extract durable call metadata from an LLMCallResult."""

    raw = getattr(call, "raw_response", None) or {}
    dist = extract_token_distribution(raw)
    usage = {key: int(dist.get(key) or 0) for key in _USAGE_KEYS}
    return {
        "model": str(getattr(call, "model", "") or ""),
        # Two targets can carry the same model and differ only in what their
        # controlled project entitles -- agy's media front has view_file alone,
        # the native one adds search_web. Without the target id an artifact
        # cannot answer "did this arm have a search tool at all", which is
        # exactly the question a retrieval comparison turns on.
        "target_id": str(getattr(call, "target_id", "") or ""),
        "backend": str(getattr(call, "backend", "") or ""),
        "api_key_label": str(getattr(call, "api_key_label", "") or ""),
        "thinking_level": str(getattr(call, "thinking_level", "") or ""),
        "capability_tier": str(
            getattr(getattr(call, "capability_tier", None), "value", "") or ""
        ),
        "fallback_used": bool(getattr(call, "fallback_used", False)),
        "usage": usage,
    }


# Replay samples are supposed to be draws from one distribution. Walking the
# temperature down per attempt made each draw come from a *different* one, so
# a set of n replies mixed sampling regimes and the spread across them measured
# the ramp as much as the prompt. The re-roll now comes from the seed alone:
# same temperature every call, different seed every call.
REPLAY_SEED_BASE = 20260815


def replay_temperature(base_temperature: float) -> float:
    """The sampling temperature for every attempt in a replay.

    Takes no attempt number on purpose: it used to walk down 0.01 per call, and
    a parameter that no longer changes the answer is how a reader concludes the
    ramp is still there.
    """

    return max(0.0, float(base_temperature))


def replay_seed(attempt: int) -> int:
    """A fresh seed per attempt, so retries re-roll without moving temperature."""

    return REPLAY_SEED_BASE + max(0, int(attempt))


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
    native_search: bool = False,
) -> ReplayResult:
    """Generic replay loop for text-only sessions (no media upload).

    Builds prompt dumps, optionally calls the API in a validation-gated loop,
    and writes per-sample replies + a summary.

    ``native_search`` has to be passed through rather than inferred from the
    rebuilt prompt: under ``retrieval=native`` the prompt *drops* the evidence
    pack because the model is supposed to search for itself. Rebuilding that
    prompt while dispatching an ordinary call would hand the model a research
    round with neither injected evidence nor a search tool -- a strictly
    handicapped arm that would read as "native is worse".
    """

    from finesub.llm.routing.config import LLMRole

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

    client = RoleClient(test_profile=test_profile)
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
        temp = replay_temperature(temperature)
        try:
            call = client.complete(
                call_role,
                messages,
                temperature=temp,
                seed=replay_seed(attempt),
                native_search=native_search,
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
            target = s.call_meta.get("target_id") or ""
            extra = "".join(
                f" {key}={s.call_meta[key]}"
                for key in ("tool_rounds", "tool_urls")
                if key in s.call_meta
            )
            lines.append(
                f"- reply-{s.index:02d}: model={s.model}"
                f"{f' target={target}' if target else ''} "
                f"in={usage.get('total_input_tokens', 0)} "
                f"out={usage.get('total_output_tokens', 0)} "
                f"think={usage.get('thinking_tokens', 0)}{extra}"
            )
    if failures:
        lines.append("")
        lines.append("## Failures")
        for f in failures:
            lines.append(f"- attempt {f.attempt}: {'; '.join(f.validation_errors)}")
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
