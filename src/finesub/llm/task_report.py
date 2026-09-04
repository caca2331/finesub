"""Human-readable task report built from retained task artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

from .exchange_metadata import (
    AGENT_SESSION_USAGE_FILENAME,
    infer_session_name,
    is_session_response_record,
    normalize_session_usage,
)
from .knowledge.base import TASK_ARTIFACT_FILENAME

TASK_REPORT_FILENAME = "task-report.md"

_TOKEN_REPORT_KEYS = (
    "uncached_input_tokens",
    "cached_input_tokens",
    "total_input_tokens",
    "thinking_tokens",
    "output_tokens",
    "total_output_tokens",
)


def write_task_report(
    artifact_dir: str | Path,
    *,
    task_id: str = "",
    outputs: Mapping[str, str] | None = None,
    run_metadata_path: str | Path | None = None,
) -> Path:
    """Write the user-facing task report and return its path."""

    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = _read_records(root / TASK_ARTIFACT_FILENAME)
    run_metadata = _read_json(Path(run_metadata_path)) if run_metadata_path else {}
    text = render_task_report(
        records,
        task_id=task_id,
        outputs=outputs or {},
        run_metadata=run_metadata,
        agent_sessions=_read_json(root / AGENT_SESSION_USAGE_FILENAME).get("sessions"),
    )
    path = root / TASK_REPORT_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def render_task_report(
    records: list[dict[str, Any]],
    *,
    task_id: str = "",
    outputs: Mapping[str, str] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
    agent_sessions: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    outputs = dict(outputs or {})
    run_metadata = dict(run_metadata or {})
    fallback_lines: list[str] = []
    ip_warning_lines: list[str] = []
    file_access_lines: list[str] = []
    provider_counts: Counter[str] = Counter()
    search_error_count = 0
    retry_count = 0
    split_count = 0
    token_phase_lines: list[str] = []
    token_total_lines: list[str] = []
    postprocess_lines: list[str] = []
    knowledge_lines: list[str] = []
    window_plan_lines: list[str] = []
    api_call_counts: Counter[str] = Counter()
    token_totals: Counter[str] = Counter()
    session_rows: list[dict[str, Any]] = []
    # (provider tier, model) -> token counters. Keyed on the pair because that
    # is what a bill is keyed on; see `_answering_target`.
    provider_usage: dict[tuple[str, str], Counter[str]] = {}
    # fact_id -> per-call {returned input tokens / local estimate} samples.
    calibration_samples: dict[str, list[Mapping[str, Any]]] = {}

    for record in records:
        kind = str(record.get("kind", ""))
        payload = record.get("payload") or {}
        if not isinstance(payload, Mapping):
            continue

        if payload.get("fallback_used"):
            fallback_lines.append(
                _bullet(
                    f"LLM model fallback in `{kind}`"
                    + _context_suffix(payload, ("chunk_id", "attempt", "model"))
                )
            )

        if kind.endswith("_call_error") or "error_type" in payload:
            error_type = str(payload.get("error_type", ""))
            error_text = str(payload.get("error", ""))
            if "LLMIPRiskError" in error_type or "IP risk warning" in error_text:
                ip_warning_lines.append(
                    _bullet(
                        f"`{kind}` detected likely IP/proxy risk control"
                        + _context_suffix(payload, ("chunk_id", "attempt"))
                    )
                )
            if (
                kind == "correction_window_call_error"
                and ("403" in error_text or "PERMISSION_DENIED" in error_text)
                and ("file" in error_text.lower() or "File" in error_text)
            ):
                file_access_lines.append(
                    _bullet(
                        f"window `{payload.get('chunk_id', '?')}` Gemini File access denied"
                        + _context_suffix(payload, ("attempt",))
                    )
                )

        if kind == "api_call":
            category = str(payload.get("category", "unknown"))
            api_call_counts[category] += 1
            if category == "web_extract":
                for item in payload.get("executed", []) or []:
                    if isinstance(item, Mapping):
                        provider = str(item.get("provider") or "error")
                        provider_counts[f"extract:{provider}"] += 1

        if kind in {"research_search_results", "correction_search_results", "search_loop_round"}:
            for item in payload.get("executed", []) or []:
                if not isinstance(item, Mapping):
                    continue
                provider = str(item.get("provider") or "error")
                provider_counts[f"search:{provider}"] += 1
                api_call_counts["web_search"] += 1
                if item.get("error"):
                    search_error_count += 1
                for event in item.get("fallbacks", []) or []:
                    if isinstance(event, Mapping):
                        fallback_lines.append(
                            _bullet(
                                "Search fallback: "
                                f"{event.get('provider', '')} {event.get('reason', '')}"
                                + (
                                    f" ({event.get('key_id')})"
                                    if event.get("key_id")
                                    else ""
                                )
                            )
                        )
            extract_urls = payload.get("extract_urls") or []
            if isinstance(extract_urls, list):
                api_call_counts["web_extract"] += len(extract_urls)

        if kind == "research_round1_response":
            api_call_counts["llm_research_round1"] += 1
        elif kind == "fast_round1_response":
            api_call_counts["llm_fast_round1"] += 1
        elif kind == "research_round2_response":
            api_call_counts["llm_research_round2"] += 1
        elif kind == "search_loop_round" and is_session_response_record(
            kind, payload
        ):
            api_call_counts["llm_search_loop"] += 1
        elif kind == "correction_query_response":
            api_call_counts["llm_correction_query"] += 1
        elif kind == "correction_window_response":
            api_call_counts["llm_correction"] += 1
        elif kind == "knowledge_update_response":
            api_call_counts["llm_knowledge_update"] += 1

        if kind == "correction_window_retry":
            retry_count += 1
            if payload.get("reason") == "output_limited_split_in_half":
                split_count += 1

        if kind == "token_distribution_report":
            phase = str(payload.get("phase", ""))
            totals = payload.get("totals") or {}
            if isinstance(totals, Mapping):
                token_phase_lines.append(_bullet(_format_token_totals(phase, totals)))
                for key in _TOKEN_REPORT_KEYS:
                    value = totals.get(key)
                    if isinstance(value, (int, float)):
                        token_totals[key] += int(value)

        route_decision = payload.get("route_decision")
        if isinstance(route_decision, Mapping):
            for candidate in route_decision.get("candidates") or []:
                if not isinstance(candidate, Mapping):
                    continue
                sample = candidate.get("estimate_calibration")
                if isinstance(sample, Mapping) and sample.get("ratio"):
                    calibration_samples.setdefault(
                        str(sample.get("fact_id") or "?"), []
                    ).append(sample)

        if is_session_response_record(kind, payload):
            usage = payload.get("usage") or {}
            if isinstance(usage, Mapping):
                row = normalize_session_usage(usage)
                row["session"] = infer_session_name(kind, payload)
                session_rows.append(row)
                # Read from `usage` rather than the normalized row: cost needs
                # cached input separated from uncached, and normalizing folds
                # them into one total.
                counters = provider_usage.setdefault(
                    _answering_target(payload), Counter()
                )
                counters["calls"] += 1
                for key in _TOKEN_REPORT_KEYS:
                    value = usage.get(key)
                    if isinstance(value, (int, float)):
                        counters[key] += int(value)

        if kind == "final_srt":
            outputs.setdefault("final_srt", str(payload.get("path", "")))
            outputs.setdefault("translated_srt", str(payload.get("translated_path", "")))
            outputs.setdefault("corrected_srt", str(payload.get("corrected_path", "")))
            outputs.setdefault("raw_srt", str(payload.get("raw_path", "")))
            postprocess = payload.get("postprocess") or {}
            if isinstance(postprocess, Mapping):
                applied_profiles = postprocess.get("applied_profiles") or []
                if isinstance(applied_profiles, (list, tuple)):
                    steps = "→".join(str(item) for item in applied_profiles) or "none"
                else:
                    steps = str(applied_profiles)
                postprocess_lines.append(
                    _bullet(
                        f"profile {postprocess.get('profile')}: "
                        f"steps {steps}, "
                        f"{postprocess.get('segment_count', 0)} segments, "
                        f"overlaps {postprocess.get('overlaps_fixed', 0)}, "
                        f"duration {postprocess.get('duration_extended', 0)}, "
                        f"flash {postprocess.get('flash_extended', 0)}, "
                        f"punctuation {postprocess.get('punctuation_replacements', 0)}, "
                        f"trimmed {postprocess.get('trimmed_lines', 0)}"
                    )
                )

        if kind == "window_plan_report":
            window_plan_lines.append(
                _bullet(
                    f"{payload.get('phase', '?')} planning: input over budget raised "
                    f"window count {payload.get('estimated_windows', '?')} -> "
                    f"{payload.get('planned_windows', '?')} "
                    f"({payload.get('replan_attempts', '?')} replan(s); last error: "
                    f"{payload.get('last_over_budget_error', '') or 'n/a'})"
                )
            )

        if kind == "knowledge_update_apply_report":
            report = payload.get("knowledge_report") or {}
            if isinstance(report, Mapping):
                conflicts = report.get("conflicts") or []
                conflict_note = ""
                if report.get("rolled_back"):
                    conflict_note = (
                        f"; ROLLED BACK ({report.get('rollback_reason', '') or 'no reason'})"
                    )
                elif conflicts:
                    # Concurrency losses are per-op under task parallelism
                    # (plan W2); a chunk that lost lines must say so here.
                    conflict_note = f"; {len(conflicts)} conflict(s) dropped/reverted"
                knowledge_lines.append(
                    _bullet(
                        f"knowledge update chunk {payload.get('chunk', '?')}: "
                        f"{len(report.get('applied', []) or [])} applied, "
                        f"{len(report.get('skipped', []) or [])} skipped"
                        + conflict_note
                    )
                )
                for conflict in conflicts[:8]:
                    if isinstance(conflict, Mapping):
                        knowledge_lines.append(
                            _bullet(
                                f"conflict: {conflict.get('entity', '?')} "
                                f"{conflict.get('id', '?')}: {conflict.get('reason', '')}",
                                indent=1,
                            )
                        )

    if token_totals:
        token_total_lines.append(_bullet(_format_token_totals("task total", dict(token_totals))))

    lines = [
        "# Task Report",
        "",
        f"- Task id: {task_id or '（未指定）'}",
        f"- Retained artifact records: {len(records)}",
    ]
    if outputs:
        lines.extend(["", "## Outputs"])
        for key, value in outputs.items():
            if value:
                lines.append(_bullet(f"{key}: `{value}`"))

    timing = run_metadata.get("timing") or {}
    rounds = run_metadata.get("llm_rounds") or []
    workers = run_metadata.get("workers") or {}
    if isinstance(timing, Mapping) and timing:
        lines.extend(["", "## Timing"])
        stages = timing.get("stages") or {}
        if isinstance(stages, Mapping):
            # Every stage the pipeline times, or the bullets do not account for
            # total_sec and the reader is left subtracting.
            for key, label in (
                ("download", "Download"),
                ("vocal_separation", "Vocal separation"),
                ("asr", "ASR"),
                ("stabilize", "Stabilization"),
                ("raw_srt", "Raw SRT"),
                ("llm_harness", "LLM harness"),
            ):
                item = stages.get(key)
                if not isinstance(item, Mapping):
                    continue
                elapsed = item.get("elapsed_sec")
                suffix = (
                    f"{float(elapsed):.3f}s"
                    if isinstance(elapsed, (int, float))
                    else "n/a"
                )
                lines.append(
                    _bullet(f"{label}: {suffix} ({item.get('status', 'unknown')})")
                )
        total = timing.get("total_sec")
        if isinstance(total, (int, float)):
            lines.append(_bullet(f"Pipeline total: {float(total):.3f}s"))

    if isinstance(workers, Mapping) and workers:
        lines.extend(["", "## Workers"])
        batch = workers.get("batch")
        if isinstance(batch, Mapping):
            lines.append(
                _bullet(
                    "batch pools: "
                    + ", ".join(
                        f"{key}={value}" for key, value in sorted(batch.items())
                    )
                )
            )
        separator = workers.get("vocal_separation")
        if isinstance(separator, Mapping):
            lines.append(
                _bullet(
                    "vocal separation: "
                    f"effective={separator.get('effective', '?')}, "
                    f"profile limit={separator.get('profile_limit', '?')}"
                )
            )
        asr = workers.get("asr")
        if isinstance(asr, Mapping):
            lines.append(
                _bullet(
                    "ASR WT: "
                    f"effective={asr.get('effective', '?')}, "
                    f"requested={asr.get('requested', '?')}, "
                    f"profile limit={asr.get('profile_limit', '?')}"
                )
            )

    if isinstance(rounds, list) and rounds:
        lines.extend(
            [
                "",
                "## LLM Round Timing",
                "| Round | Wall | API | Attempts | Retries | Status |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in rounds:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "| {round} | {elapsed:.3f}s | {api:.3f}s | {attempts} | "
                "{retries} | {status} |".format(
                    round=row.get("round", "?"),
                    elapsed=float(row.get("elapsed_sec") or 0.0),
                    api=float(row.get("api_sec") or 0.0),
                    attempts=int(row.get("api_attempts") or 0),
                    retries=int(row.get("retries") or 0),
                    status=row.get("status", "unknown"),
                )
            )

    lines.extend(["", "## API Call Counts"])
    if api_call_counts:
        for category, count in sorted(api_call_counts.items()):
            lines.append(_bullet(f"{category}: {count}"))
    else:
        lines.append("- No retained API call records.")
    if provider_counts:
        provider_summary = ", ".join(
            f"{provider}: {count}" for provider, count in sorted(provider_counts.items())
        )
        lines.append(_bullet(f"web providers: {provider_summary}"))
        lines.append(_bullet(f"search errors: {search_error_count}"))

    lines.extend(["", "## LLM Token Usage"])
    lines.extend(token_phase_lines or ["- No retained token phase report."])
    lines.extend(token_total_lines or ["- No aggregate LLM token totals."])
    lines.append(_bullet(f"correction retries: {retry_count}; split retries: {split_count}"))

    lines.extend(["", "## Session Token Totals"])
    lines.extend(_session_token_lines(session_rows))

    lines.extend(["", "## Provider Token Totals"])
    _add_agent_session_usage(provider_usage, agent_sessions)
    lines.extend(_provider_token_lines(provider_usage))

    lines.extend(["", "## Token Estimate Calibration"])
    lines.extend(_calibration_lines(calibration_samples))

    lines.extend(["", "## Fallbacks And Warnings"])
    if fallback_lines:
        lines.append("- Fallback occurred during this task.")
        lines.extend(fallback_lines)
    else:
        lines.append("- No fallback was recorded in retained artifacts.")
    if ip_warning_lines:
        lines.append("- LLM IP/proxy risk-control warning was detected separately from quota/provider errors.")
        lines.extend(ip_warning_lines)
    if file_access_lines:
        lines.extend(["", "### Gemini File Access"])
        lines.extend(file_access_lines)
        lines.append(
            _bullet(
                "Likely cause (not key rotation): a background-prefetched clip upload "
                "was referenced before the Gemini Files entry finished processing to "
                "ACTIVE, or a stale file URI was reused across windows. Resume with "
                "a fresh upload usually succeeds."
            )
        )

    if window_plan_lines:
        lines.extend(["", "## Window Planning"])
        lines.extend(window_plan_lines)

    lines.extend(["", "## Postprocess"])
    lines.extend(postprocess_lines or ["- No retained postprocess summary."])

    lines.extend(["", "## Knowledge"])
    lines.extend(knowledge_lines or ["- No retained knowledge update summary."])

    lines.extend(
        [
            "",
            "## Suggested Review",
            "- Review lines around correction retries, split windows, search failures, and any fallback noted above.",
            "- If IP/proxy risk-control warnings appear, retry from a clean network path before treating them as quota failures.",
            "- If Gemini File 403 errors appear, delete downstream artifacts and rerun, or use resume after the prefetch upload completes.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _format_token_totals(label: str, totals: Mapping[str, Any]) -> str:
    call_count = totals.get("call_count", "?")
    parts = [f"{key}={totals.get(key, 0)}" for key in _TOKEN_REPORT_KEYS if key in totals]
    if not parts:
        parts = [
            f"total={totals.get('total_tokens', 0)}",
            f"text={totals.get('prompt_text_tokens', 0)}",
            f"audio={totals.get('prompt_audio_tokens', 0)}",
            f"thinking={totals.get('thinking_tokens', 0)}",
            f"output={totals.get('output_tokens', 0)}",
        ]
    return f"{label} ({call_count} calls): " + ", ".join(parts)


def _session_token_lines(session_rows: list[dict[str, Any]]) -> list[str]:
    if not session_rows:
        return ["- No retained per-session LLM usage."]

    lines = [
        "| Session | Total input | Total output | Thinking | Visible output |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    totals = Counter()
    for row in session_rows:
        lines.append(
            "| {session} | {total_input_tokens} | {total_output_tokens} | "
            "{thinking_tokens} | {output_tokens} |".format(**row)
        )
        for key in (
            "total_input_tokens",
            "total_output_tokens",
            "thinking_tokens",
            "output_tokens",
        ):
            totals[key] += int(row.get(key, 0))
    lines.append(
        "| **task total** | {total_input_tokens} | {total_output_tokens} | "
        "{thinking_tokens} | {output_tokens} |".format(**dict(totals))
    )
    lines.append("")
    lines.append(
        _bullet(
            "Each row is one LLM API session (one exchange file). "
            "Input/output totals come from provider usage metadata, not from "
            "summing component estimates in the exchange header."
        )
    )
    return lines


def _answering_target(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Which (provider tier, model) actually answered this call.

    A bill is keyed on the pair, not on the model: the same model on the free
    and the paid Gemini tier costs differently, and a local-agent tier costs
    nothing per token at all. `model` alone cannot say which, and only the route
    decision records which candidate answered -- so the pair is joined there.
    """

    route = payload.get("route_decision")
    model = str(payload.get("model") or "")
    if not isinstance(route, Mapping):
        return ("unknown", model or "unknown")
    answered = ""
    for candidate in route.get("candidates") or ():
        if isinstance(candidate, Mapping) and candidate.get("outcome") == "success":
            answered = str(candidate.get("target_id") or "")
            break
    for entry in route.get("effective_chain") or ():
        if not isinstance(entry, Mapping):
            continue
        if answered:
            if str(entry.get("target_id") or "") != answered:
                continue
        # No recorded winner (an older artifact): the model name is the only
        # handle left, and it is unambiguous unless one chain lists the same
        # model on two tiers.
        elif str(entry.get("model") or "") != model:
            continue
        return (
            str(entry.get("provider_tier") or "unknown"),
            str(entry.get("model") or model or "unknown"),
        )
    return ("unknown", model or "unknown")


def _add_agent_session_usage(
    usage: dict[tuple[str, str], Counter[str]],
    sessions: Sequence[Mapping[str, Any]] | None,
) -> None:
    """Fold a run's agent session totals into the per-provider table.

    A pseudo-conversational session is metered per CLI invocation, not per
    task, so every call of such a session reports no tokens at all and the
    tokens arrive here instead (docs/llm_local_agent.md §12.1.3). Only the
    token columns are touched: the call count still comes from the windows,
    which is what it means.
    """

    for session in sessions or []:
        if not isinstance(session, Mapping):
            continue
        tokens = session.get("usage")
        if not isinstance(tokens, Mapping):
            continue
        # Keyed exactly as the per-call rows are (`_answering_target` passes
        # the tier through as recorded): case-folding here split one agent
        # into two rows, one with the calls and one with the tokens.
        target = (
            str(session.get("provider_tier") or "unknown"),
            str(session.get("model") or "unknown"),
        )
        counters = usage.setdefault(target, Counter())
        for key in _TOKEN_REPORT_KEYS:
            value = tokens.get(key)
            if isinstance(value, (int, float)):
                counters[key] += int(value)


def _provider_token_lines(
    usage: Mapping[tuple[str, str], Counter],
) -> list[str]:
    """Token cost grouped the way it is charged.

    The session table answers "which round spent this"; this one answers "who is
    billing for it". Both are needed and neither derives from the other -- one
    session can fall back across tiers, and one tier serves many sessions.
    """

    if not usage:
        return ["- No retained per-provider LLM usage."]

    lines = [
        "| Provider tier | Model | Calls | Input (total) | of which cached | "
        "Visible output | Thinking |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    totals: Counter[str] = Counter()
    for (tier, model), counts in sorted(usage.items()):
        lines.append(
            f"| {tier} | {model} | {counts['calls']} | "
            f"{counts['total_input_tokens']} | {counts['cached_input_tokens']} | "
            f"{counts['output_tokens']} | {counts['thinking_tokens']} |"
        )
        totals.update(counts)
    lines.append(
        f"| **task total** | | {totals['calls']} | "
        f"{totals['total_input_tokens']} | {totals['cached_input_tokens']} | "
        f"{totals['output_tokens']} | {totals['thinking_tokens']} |"
    )
    lines.append("")
    lines.append(
        _bullet(
            "Input is the full prompt side with the cached part called out, "
            "because the two are priced differently -- and because a provider "
            "that reports only a total then reads as a total rather than as "
            "zero. Local-agent tiers are metered by subscription, not per "
            "token; their rows are for cache and context accounting."
        )
    )
    return lines


# Below this many samples a median is noise, so the report shows the
# observation without turning it into a suggestion.
_CALIBRATION_MIN_SAMPLES = 5
# Suggest a change only once the observed ratio is far enough from the
# configured scale to matter for window planning.
_CALIBRATION_MIN_DRIFT = 0.10


def _calibration_lines(
    samples: Mapping[str, list[Mapping[str, Any]]]
) -> list[str]:
    """``token_scale`` reference: returned input tokens over local estimate.

    The local counter speaks Gemini's vocabulary, so any other model's
    estimate carries a systematic bias (plan v2 D14/§5.6). This measures it
    from the provider's own numbers -- the only ones that know the truth.

    Deliberately advisory, never written back: ``token_scale`` changes the
    window geometry and therefore the routing identity, so adopting it must be
    an explicit config change rather than something a run does to its own
    checkpoints. Samples are also biased by content shape (dense CSV vs prose
    context packs), which is the second reason a human decides.
    """

    if not samples:
        return ["- No calibration samples were retained."]
    lines = [
        "| Fact | Samples | Median returned/estimated | Configured token_scale | Suggestion |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for fact_id in sorted(samples):
        rows = samples[fact_id]
        ratios = sorted(
            float(row["ratio"]) for row in rows if isinstance(row.get("ratio"), (int, float))
        )
        if not ratios:
            continue
        median = ratios[len(ratios) // 2]
        configured = 1.0
        for row in rows:
            value = row.get("token_scale")
            if isinstance(value, (int, float)):
                configured = float(value)
        if len(ratios) < _CALIBRATION_MIN_SAMPLES:
            suggestion = f"too few samples (need {_CALIBRATION_MIN_SAMPLES})"
        elif abs(median - configured) / max(configured, 1e-6) < _CALIBRATION_MIN_DRIFT:
            suggestion = "keep"
        else:
            suggestion = f"consider token_scale = {median:.2f}"
        lines.append(
            f"| {fact_id} | {len(ratios)} | {median:.3f} | {configured:.2f} | {suggestion} |"
        )
    lines.append("")
    lines.append(
        _bullet(
            "Ratios compare the provider's returned input tokens against the "
            "*unscaled* local estimate. Set token_scale in the catalog by hand "
            "when you accept a suggestion: it changes new window geometry; "
            "saved plans and completed windows remain reusable."
        )
    )
    return lines


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _bullet(text: str, indent: int = 0) -> str:
    return f"{'  ' * indent}- {text}"


def _context_suffix(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    parts = [f"{key}={payload[key]}" for key in keys if payload.get(key) is not None]
    return f" ({', '.join(parts)})" if parts else ""
