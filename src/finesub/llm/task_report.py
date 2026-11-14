"""Human-readable task report built from retained task artifacts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any, Mapping, Sequence

from finesub import jsonl
from finesub_bootstrap.fsops import write_atomic
from .exchange_metadata import AGENT_SESSION_USAGE_FILENAME, is_session_response_record
from .knowledge.base import TASK_ARTIFACT_FILENAME
from . import token_books

TASK_REPORT_FILENAME = "task-report.md"

# LLM response record -> its line under "API Call Counts".
_CALL_COUNTS = {
    "prepare_task_request": "llm_prepare_request",
    "prepare_task_response": "llm_prepare_delivery",
    "speaker_judge_response": "llm_speaker_judge",
    "correction_query_response": "llm_correction_query",
    "correction_window_response": "llm_correction",
    "proofread_response": "llm_proofread",
    "knowledge_update_response": "llm_knowledge_update",
    "knowledge_proposal_repair_response": "llm_knowledge_proposal_repair",
}

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
    records = list(jsonl.iter_records(root / TASK_ARTIFACT_FILENAME))
    run_metadata = _read_json(Path(run_metadata_path)) if run_metadata_path else {}
    text = render_task_report(
        records,
        task_id=task_id,
        outputs=outputs or {},
        run_metadata=run_metadata,
        agent_sessions=_read_json(root / AGENT_SESSION_USAGE_FILENAME).get("sessions"),
    )
    path = root / TASK_REPORT_FILENAME
    write_atomic(path, text)
    return path


def _last(records: Sequence[Mapping[str, Any]], kind: str) -> Mapping[str, Any] | None:
    payloads = [record.get("payload") or {} for record in records if record.get("kind") == kind]
    return payloads[-1] if payloads else None


def _speaker_judge_section(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """The last judge report (a resumed run reports the sidecar as it stands now), or nothing; as three passes'
    pass 1, also the rows it dropped, each with the reason given."""
    report = _last(records, "speaker_judge_report")
    if report is None:
        return []
    lines = ["", "## Speaker Judge", _speaker_judge_line(report)]
    if report.get("shape"):
        acts = report.get("acts") or {}
        text = (f"pass 1 ({report['shape']}): {acts.get('join', 0)} join, {acts.get('may_join', 0)} may_join, "
                f"{acts.get('drop', 0)} drop")
        if report.get("without_pass1"):
            text += f"; judge windows without a pass 1 answer: {', '.join(report['without_pass1'])}"
        if "fills" in report:
            text += (f"; heard the ASR miss: {report['fills']} fill(s), {report['inserts']} insert(s) "
                     f"({report.get('deferred', 0)} deferred for want of room)")
        lines.append(_bullet(text))
        lines.extend(_bullet(f"row {item.get('row')} dropped: {item.get('why') or '(no reason given)'}", 1)
                     for item in report.get("dropped") or ())
    return lines


def _passes_section(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """How the task ran (`correction_mode`) and what pass 3 did (`proofread`), or nothing for a task that
    recorded neither."""
    mode, proofread = _last(records, "correction_mode"), _last(records, "proofread")
    if mode is None and proofread is None:
        return []
    lines = ["", "## Passes"]
    if mode is not None:
        if mode.get("mode") == "three-pass":
            text = f"three passes, pass 1 {mode.get('shape') or 'none (no speaker labels)'}: {mode.get('windows', 0)} window(s)"
            without = mode.get("without_pass1") or []
            text += f", {len(without)} on the single pass prompt for want of pass 1" + (
                f" ({', '.join(without)})" if without else "")
            if mode.get("skipped"):
                text += f"; {len(mode['skipped'])} left out, every core line dropped ({', '.join(mode['skipped'])})"
            text += f"; {mode.get('joined', 0)} line(s) joined, {mode.get('dropped', 0)} dropped"
            if mode.get("refused"):
                text += "; joins kept apart: " + ", ".join(f"{why} {count}" for why, count in mode["refused"].items())
        else:
            text = f"one pass ({mode.get('reason') or 'no reason recorded'})"
        lines.append(_bullet(text + f"; pass 3 {'on' if mode.get('proofread') else 'off'}"))
    if proofread is not None:
        text = f"pass 3: {proofread.get('status') or 'unknown'}"
        if "windows" in proofread:
            text += (f", {proofread.get('windows', 0)} window(s), {proofread.get('reused', 0)} reused, "
                     f"{len(proofread.get('changes') or [])} line(s) changed")
        if proofread.get("failed"):
            text += f"; not answered (pass 2's lines kept): {', '.join(proofread['failed'])}"
        if proofread.get("stopped"):
            text += f"; stopped early: {proofread['stopped']}"
        if proofread.get("error"):
            text += f"; did not run (pass 2's subtitles are the result): {proofread['error']}"
        lines.append(_bullet(text))
    return lines


def _speaker_judge_line(payload: Mapping[str, Any]) -> str:
    if payload.get("skipped"):
        return _bullet(f"skipped: {payload['skipped']}")
    tracks = ", ".join(payload.get("new_tracks") or []) or "none"
    text = (f"{payload.get('windows', 0)} window(s): judged {payload.get('judged', 0)}, reused "
            f"{payload.get('reused', 0)}, failed {payload.get('failed', 0)} and pending {payload.get('pending', 0)} "
            f"(those keep the acoustic labels); {payload.get('moved', 0)} line(s) moved; new tracks: {tracks}")
    if payload.get("cut"):
        text += f"; {payload['cut']} window(s) halved (the answer or the prompt did not fit)"
    if payload.get("stopped"):
        text += f"; stopped early: {payload['stopped']}"
    return _bullet(text)


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
    call_error_lines: list[str] = []
    ip_warning_lines: list[str] = []
    file_access_lines: list[str] = []
    provider_counts: Counter[str] = Counter()
    search_error_count = 0
    retry_count = 0
    split_count = 0
    postprocess_lines: list[str] = []
    knowledge_lines: list[str] = []
    window_plan_lines: list[str] = []
    api_call_counts: Counter[str] = Counter()
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
            call_error_lines.append(_failed_call_line(kind, payload, error_type, error_text))
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

        if kind in {"research_search_results", "correction_search_results"}:
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

        # A request turn that delivered at once IS the window call counted
        # below -- one request, counted once (`is_session_response_record`).
        if kind in _CALL_COUNTS and not (kind == "correction_query_response" and not is_session_response_record(kind, payload)):
            api_call_counts[_CALL_COUNTS[kind]] += 1

        if kind == "correction_window_retry":
            retry_count += 1
            if payload.get("reason") == "output_limited_split_in_half":
                split_count += 1

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

    bookings = token_books.book(records, agent_sessions)
    lines.extend(["", "## LLM Token Usage"])
    lines.extend(token_books.phase_lines(bookings))
    lines.append(_bullet(f"correction retries: {retry_count}; split retries: {split_count}"))

    lines.extend(["", "## Session Token Totals"])
    lines.extend(token_books.session_lines(bookings))

    lines.extend(["", "## Provider Token Totals"])
    lines.extend(token_books.provider_lines(bookings))

    lines.extend(["", "## Token Estimate Calibration"])
    lines.extend(_calibration_lines(calibration_samples))

    lines.extend(["", "## Fallbacks And Warnings"])
    if fallback_lines:
        lines.append("- Fallback occurred during this task.")
        lines.extend(fallback_lines)
    else:
        lines.append("- No fallback was recorded in retained artifacts.")
    if call_error_lines:
        lines.append("- Calls failed during this task:")
        lines.extend(call_error_lines)
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
    return "\n".join([*lines, *_speaker_judge_section(records), *_passes_section(records)]).rstrip() + "\n"


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


def _clip(text: str, limit: int) -> str:
    """One line of at most `limit` characters, for quoting an error in a bullet."""

    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _failed_call_line(
    kind: str, payload: Mapping[str, Any], error_type: str, error_text: str
) -> str:
    """One bullet naming a failed call, for "Fallbacks And Warnings".

    Every failed call gets one -- not just the IP-risk and File-403 classes,
    which also have sections of their own. Those used to be the only failures
    this report named, so a run whose call failed any other way (an agent that
    left without a submit, say) read "No fallback was recorded" and nothing
    else: accurate about fallbacks, silent about the failure the reader came
    here to find.
    """

    return _bullet(
        f"`{kind}` failed: {error_type or 'error'}"
        + (f": {_clip(error_text, 200)}" if error_text else "")
        + _context_suffix(payload, ("chunk_id", "attempt", "call"))
    )


def _context_suffix(payload: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    parts = [f"{key}={payload[key]}" for key in keys if payload.get(key) is not None]
    return f" ({', '.join(parts)})" if parts else ""
