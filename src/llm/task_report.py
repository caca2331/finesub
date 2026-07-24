"""Human-readable task report built from retained task artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any, Mapping

from .exchange_metadata import (
    SESSION_RESPONSE_KINDS,
    infer_session_name,
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
) -> Path:
    """Write the user-facing task report and return its path."""

    root = Path(artifact_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = _read_records(root / TASK_ARTIFACT_FILENAME)
    text = render_task_report(records, task_id=task_id, outputs=outputs or {})
    path = root / TASK_REPORT_FILENAME
    path.write_text(text, encoding="utf-8")
    return path


def render_task_report(
    records: list[dict[str, Any]],
    *,
    task_id: str = "",
    outputs: Mapping[str, str] | None = None,
) -> str:
    outputs = dict(outputs or {})
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
        elif kind == "search_loop_round":
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

        if kind in SESSION_RESPONSE_KINDS:
            usage = payload.get("usage") or {}
            if isinstance(usage, Mapping):
                row = normalize_session_usage(usage)
                row["session"] = infer_session_name(kind, payload)
                session_rows.append(row)

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
            mistakes = payload.get("mistake_report") or {}
            if isinstance(report, Mapping):
                mistake_note = ""
                if isinstance(mistakes, Mapping) and mistakes:
                    mistake_note = (
                        f"; mistakes {len(mistakes.get('applied', []) or [])} applied"
                    )
                knowledge_lines.append(
                    _bullet(
                        f"knowledge update chunk {payload.get('chunk', '?')}: "
                        f"{len(report.get('applied', []) or [])} applied, "
                        f"{len(report.get('skipped', []) or [])} skipped"
                        + mistake_note
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


def _bullet(text: str) -> str:
    return f"- {text}"


def _context_suffix(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    parts = [f"{key}={payload[key]}" for key in keys if payload.get(key) is not None]
    return f" ({', '.join(parts)})" if parts else ""
