"""Correction R2 session adapter for prompt iteration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from llm.audio_clips import (
    default_clip_path,
    default_video_clip_path,
    extract_window_clip,
    extract_window_video_clip,
)
from llm.client import (
    LiteLLMRoleClient,
    UploadedFileRef,
    VALIDATION_BASE_TEMPERATURE,
    VALIDATION_TEMPERATURE_STEP,
    extract_finish_reason,
    extract_token_distribution,
    upload_gemini_file,
)
from llm.config import DEFAULT_LIMITS, CapabilityTier
from llm.csv_utils import validate_correction_window_output
from llm.exchange_metadata import extract_tagged_block
from llm.prompts import ContextPack, build_correction_csv_messages
from llm.prompt_variants import DEFAULT_VARIANT_FOR_TIER, resolve_variant
from llm.stages.correction_loop import correction_role_for_profile
from ..fixture import (
    CorrectionFixture,
    apply_profile_override,
    build_window_from_fixture,
    ensure_correction_fixture,
    find_correction_exchange,
    fixture_path,
    resolve_media_path,
    resolve_run_layout,
    save_fixture,
)
from .base import pin_client_role_to_free_model

# Compact usage keys written into reply meta / summary (provider-normalized).
_USAGE_KEYS = (
    "prompt_tokens",
    "prompt_text_tokens",
    "prompt_audio_tokens",
    "uncached_input_tokens",
    "cached_input_tokens",
    "total_input_tokens",
    "thinking_tokens",
    "output_tokens",
    "total_output_tokens",
    "total_tokens",
)


@dataclass
class SampleResult:
    ok: bool
    index: int
    attempt: int
    content: str
    temperature: float = VALIDATION_BASE_TEMPERATURE
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    model: str = ""
    path: Path | None = None
    translated_path: Path | None = None
    call_meta: Dict[str, Any] = field(default_factory=dict)


def call_result_meta(call: Any) -> Dict[str, Any]:
    """Extract durable call metadata from ``LLMCallResult`` (or test doubles).

    ``LLMCallResult`` has no ``.usage`` field — tokens live on ``raw_response``.
    """

    raw = getattr(call, "raw_response", None) or {}
    dist = extract_token_distribution(raw)
    usage = {key: int(dist.get(key) or 0) for key in _USAGE_KEYS}
    tier = getattr(call, "capability_tier", None)
    return {
        "model": str(getattr(call, "model", "") or ""),
        "api_key_label": str(getattr(call, "api_key_label", "") or ""),
        "thinking_level": str(getattr(call, "thinking_level", "") or ""),
        "thinking_budget": int(getattr(call, "thinking_budget", 0) or 0),
        "capability_tier": str(getattr(tier, "value", tier) or ""),
        "fallback_used": bool(getattr(call, "fallback_used", False)),
        "finish_reason": extract_finish_reason(raw) if raw else "",
        "usage": usage,
    }


@dataclass
class CorrectionReplayResult:
    fixture_path: Path
    out_dir: Path
    prompt_system_path: Path
    prompt_user_path: Path
    successes: List[SampleResult]
    failures: List[SampleResult]
    summary_path: Path
    dry_run: bool


def _thinking_override_kwargs(
    profile,
    thinking_level: str | None = None,
) -> Dict[str, Any]:
    """CLI ``thinking_level`` wins over profile ``thinking_override``."""

    level = (thinking_level or profile.thinking_override or "").strip()
    if level:
        return {"thinking_level": level}
    return {}


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


class CorrectionSessionAdapter:
    name = "correction"

    def ensure_fixture(
        self,
        *,
        run: Path,
        chunk_id: str,
        force_extract: bool = False,
        allow_oneshot_r1: bool = False,
        **_kwargs: Any,
    ) -> tuple[CorrectionFixture, Path]:
        layout = resolve_run_layout(run)
        path = fixture_path(layout["artifact_dir"], chunk_id)
        if path.exists() and not force_extract:
            from ..fixture import load_fixture

            return load_fixture(path), path
        if find_correction_exchange(layout["artifact_dir"], chunk_id) is not None:
            return ensure_correction_fixture(
                run=run, chunk_id=chunk_id, force_extract=force_extract
            )
        if allow_oneshot_r1:
            raise NotImplementedError(
                "One-shot R1 (live query round + search/extract) is not wired in "
                "this revision; provide an R2 exchange or a pre-built "
                f"session-fixtures/correction-{chunk_id}.json under {layout['artifact_dir']}."
            )
        raise FileNotFoundError(
            f"No fixture and no R2 exchange for chunk {chunk_id} in "
            f"{layout['artifact_dir']}/exchanges. Re-run the original correction "
            "window once, or pass a prepared fixture."
        )

    def build_messages(
        self,
        fixture: CorrectionFixture,
        tier: CapabilityTier = CapabilityTier.CAPABLE,
        variant: str | None = None,
    ) -> List[Dict[str, Any]]:
        profile = fixture.profile()
        window = build_window_from_fixture(fixture)
        audio_path = resolve_media_path(fixture, "audio_path")
        video_path = resolve_media_path(fixture, "video_path")
        if profile.use_video and video_path is not None:
            audio_label = str(video_path)
        elif audio_path is not None:
            audio_label = str(audio_path)
        else:
            audio_label = ""
        return build_correction_csv_messages(
            window=window,
            context_pack=ContextPack.from_dict(fixture.context_pack),
            audio_file_label=audio_label,
            previous_advice=fixture.previous_advice,
            query_round_notes=fixture.window_notes,
            search_results=fixture.search_results,
            entry_details=fixture.entry_details,
            extra_style=fixture.extra_style,
            common_mistakes_block=fixture.common_mistakes_block,
            task_update_feedback=fixture.task_update_feedback,
            evidence_pack_mode=fixture.evidence_pack_mode,
            profile=profile,
            tier=tier,
            variant=variant,
        )

    def prepare_media(
        self,
        fixture: CorrectionFixture,
        *,
        clip_dir: Path,
    ) -> UploadedFileRef | None:
        profile = fixture.profile()
        if not profile.use_audio and not profile.use_video:
            return None
        window = build_window_from_fixture(fixture)
        clip_dir.mkdir(parents=True, exist_ok=True)
        # Extraction is deterministic for a given window; reuse an existing clip
        # in the shared dir. The upload still happens every run (Gemini file
        # refs expire), only the ffmpeg extraction is skipped. Extraction is
        # atomic (temp file + rename) so concurrent replays sharing this dir
        # never read a half-written clip.
        if profile.use_video:
            out = default_video_clip_path(clip_dir, fixture.chunk_id)
            if not out.exists():
                video_src = resolve_media_path(fixture, "video_path")
                if video_src is None:
                    raise FileNotFoundError("mm-high fixture requires video_path")
                tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp{out.suffix}")
                extract_window_video_clip(
                    video_src, window.clip_start, window.clip_end, tmp
                )
                tmp.replace(out)
            return upload_gemini_file(out)
        out = default_clip_path(clip_dir, fixture.chunk_id)
        if not out.exists():
            audio_src = resolve_media_path(fixture, "audio_path")
            if audio_src is None:
                raise FileNotFoundError("audio profile fixture requires audio_path")
            tmp = out.with_name(f".{out.stem}.{os.getpid()}.tmp{out.suffix}")
            extract_window_clip(audio_src, window.clip_start, window.clip_end, tmp)
            tmp.replace(out)
        return upload_gemini_file(out)

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
        **_kwargs: Any,
    ) -> CorrectionReplayResult:
        if model and test_profile:
            raise RuntimeError("--model and --test-profile are mutually exclusive")
        # A forced variant overrides the answering endpoint's tier-derived
        # prompt set. --variant names a variant directly; the legacy
        # --force-tier names a *tier* (capable/basic) and maps to that tier's
        # default variant. --variant wins. Validated eagerly so a typo never
        # silently serves the default prompt.
        forced_variant = variant
        if forced_variant is None and force_tier is not None:
            forced_variant = DEFAULT_VARIANT_FOR_TIER[CapabilityTier(force_tier)]
        if forced_variant is not None:
            resolve_variant(forced_variant)  # raises on unknown name
        forced_variant_config = (
            resolve_variant(forced_variant) if forced_variant is not None else None
        )
        fixture_override = _kwargs.get("fixture_override")
        if fixture_override is not None:
            from ..fixture import load_fixture

            fixture = load_fixture(Path(fixture_override))
            fixture_src = Path(fixture_override)
        else:
            fixture, fixture_src = self.ensure_fixture(
                run=run, chunk_id=chunk_id, force_extract=force_extract
            )
        fixture = apply_profile_override(fixture, profile)
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy fixture into the iterate dir so search/extract text is archived.
        local_fixture = out_dir / "fixture.json"
        save_fixture(local_fixture, fixture)

        # Assemble per capability tier, mirroring production (T.1): the
        # answering endpoint's tier picks its own prompt at call time.
        def messages_for_tier(tier: CapabilityTier) -> List[Dict[str, Any]]:
            return self.build_messages(fixture, tier=tier)

        def _split(msgs: List[Dict[str, Any]]) -> tuple[str, str]:
            system_text = user_text = ""
            for msg in msgs:
                if msg.get("role") == "system":
                    system_text = str(msg.get("content") or "")
                elif msg.get("role") == "user":
                    user_text = str(msg.get("content") or "")
            return system_text, user_text

        def _dump(suffix: str, system_text: str, user_text: str, *, label: str) -> None:
            (out_dir / f"prompt.system{suffix}.txt").write_text(
                system_text, encoding="utf-8"
            )
            (out_dir / f"prompt.user{suffix}.txt").write_text(
                user_text, encoding="utf-8"
            )
            # Sanity: frozen search body must appear in the rebuilt user prompt.
            if (
                fixture.search_results.strip()
                and fixture.search_results.strip() not in user_text
            ):
                raise RuntimeError(
                    "Rebuilt user prompt is missing frozen search_results; refusing "
                    f"to continue ({label}; would silently drop search/extract reuse)."
                )

        if forced_variant is not None:
            # Iteration forces a variant, and the variant name already encodes
            # its target tier — so dump exactly the one prompt that will be
            # sent, named by variant. No per-tier reference dumps (they would be
            # unused noise / the "which file is real?" ambiguity).
            system_text, user_text = _split(
                self.build_messages(fixture, variant=forced_variant)
            )
            _dump(
                f".{forced_variant}", system_text, user_text,
                label=f"variant={forced_variant}",
            )
            prompt_system_path = out_dir / f"prompt.system.{forced_variant}.txt"
            prompt_user_path = out_dir / f"prompt.user.{forced_variant}.txt"
        else:
            # Un-forced: the answering endpoint's tier decides at call time, so
            # dump both tier defaults as references.
            for tier in (CapabilityTier.CAPABLE, CapabilityTier.BASIC):
                system_text, user_text = _split(messages_for_tier(tier))
                suffix = "" if tier is CapabilityTier.CAPABLE else f".{tier.value}"
                _dump(suffix, system_text, user_text, label=f"tier={tier.value}")
            prompt_system_path = out_dir / "prompt.system.txt"
            prompt_user_path = out_dir / "prompt.user.txt"

        successes: List[SampleResult] = []
        failures: List[SampleResult] = []
        if dry_run:
            summary_path = self._write_summary(
                out_dir=out_dir,
                fixture=fixture,
                fixture_src=fixture_src,
                label=label,
                note=note or "dry-run（未调用 API）",
                successes=successes,
                failures=failures,
                dry_run=True,
                base_temperature=temperature,
            )
            return CorrectionReplayResult(
                fixture_path=local_fixture,
                out_dir=out_dir,
                prompt_system_path=prompt_system_path,
                prompt_user_path=prompt_user_path,
                successes=successes,
                failures=failures,
                summary_path=summary_path,
                dry_run=True,
            )

        client = LiteLLMRoleClient(test_profile=test_profile)
        # Media clips are deterministic per (run, chunk, profile) and huge
        # (~40MB each), so share one static dir across all labels of this test
        # bed instead of re-extracting a copy under every label dir.
        clip_dir = out_dir.parent / "_clips"
        file_ref = self.prepare_media(fixture, clip_dir=clip_dir)
        window = build_window_from_fixture(fixture)
        profile = fixture.profile()
        role = correction_role_for_profile(profile)
        if model:
            # Pin exactly one FREE endpoint. In particular, ``3.5-flash``
            # must not also match/fallback to ``3.5-flash-lite``.
            pin_client_role_to_free_model(client, role, model)

        # The factory used for actual calls: when a variant is forced, ignore
        # the answering endpoint's tier and always serve that variant's prompt.
        # The per-tier dumps above stay faithful to both real tiers.
        if forced_variant is not None:
            def answer_factory(_tier: CapabilityTier) -> List[Dict[str, Any]]:
                return self.build_messages(fixture, variant=forced_variant)
        else:
            answer_factory = messages_for_tier

        quota_exhausted: str | None = None
        attempt = 0
        while len(successes) < n and attempt < max_attempts:
            attempt += 1
            call_temperature = replay_temperature(temperature, attempt)
            try:
                call = client.complete(
                    role,
                    answer_factory,
                    max_tokens=DEFAULT_LIMITS.output_limit,
                    file_ref=file_ref,
                    temperature=call_temperature,
                    **_thinking_override_kwargs(profile, thinking_level=thinking_level),
                )
            except RuntimeError as exc:
                # Daily/RPM exhaustion of the (possibly pinned) chain: stop,
                # keep what we have, and report instead of burning attempts.
                quota_exhausted = str(exc)
                break
            response_variant = forced_variant_config or resolve_variant(
                None, call.capability_tier
            )
            validation = validate_correction_window_output(
                call.content,
                window,
                variant=response_variant,
                allow_insert=profile.use_audio,
            )
            meta = call_result_meta(call)
            sample = SampleResult(
                ok=validation.ok,
                index=len(successes) + 1 if validation.ok else len(failures) + 1,
                attempt=attempt,
                content=call.content,
                temperature=call_temperature,
                validation_errors=list(validation.errors),
                validation_warnings=list(validation.warnings),
                model=str(meta.get("model") or getattr(call, "model", "") or ""),
                call_meta=meta,
            )
            if validation.ok:
                idx = len(successes) + 1
                reply_path = out_dir / f"reply-{idx:02d}.md"
                translated_path = out_dir / f"reply-{idx:02d}.translated.csv"
                self._write_reply(reply_path, sample)
                translated = extract_tagged_block(call.content, "translated")
                translated_path.write_text(translated + "\n", encoding="utf-8")
                sample.path = reply_path
                sample.translated_path = translated_path
                sample.index = idx
                successes.append(sample)
            else:
                fail_path = out_dir / f"failed-attempt{attempt:02d}.md"
                self._write_reply(fail_path, sample)
                sample.path = fail_path
                failures.append(sample)

        if forced_variant is not None:
            note = (
                (note + "\n\n" if note else "")
                + f"**强制 prompt variant={forced_variant}**"
                f"（require_singles={forced_variant_config.require_full_singles}；"
                "应答端点实际 tier 见各 "
                "reply meta；用作跨变体对照）"
            )
        if quota_exhausted:
            note = (
                (note + "\n\n" if note else "")
                + f"**配额中断**（model={model or 'chain'}）：{quota_exhausted}"
            )
        summary_path = self._write_summary(
            out_dir=out_dir,
            fixture=fixture,
            fixture_src=fixture_src,
            label=label,
            note=note,
            successes=successes,
            failures=failures,
            dry_run=False,
            base_temperature=temperature,
        )
        if quota_exhausted:
            raise RuntimeError(
                f"Quota exhausted after {len(successes)}/{n} validation-ok replies "
                f"(model={model or 'chain'}; see {out_dir}): {quota_exhausted}"
            )
        if len(successes) < n:
            raise RuntimeError(
                f"Only collected {len(successes)}/{n} validation-ok replies "
                f"after {attempt} attempts (see {out_dir})"
            )
        return CorrectionReplayResult(
            fixture_path=local_fixture,
            out_dir=out_dir,
            prompt_system_path=prompt_system_path,
            prompt_user_path=prompt_user_path,
            successes=successes,
            failures=failures,
            summary_path=summary_path,
            dry_run=False,
        )

    def _write_reply(self, path: Path, sample: SampleResult) -> None:
        call_meta = dict(sample.call_meta or {})
        meta = {
            "ok": sample.ok,
            "attempt": sample.attempt,
            "model": sample.model or call_meta.get("model") or "",
            "api_key_label": call_meta.get("api_key_label") or "",
            "thinking_level": call_meta.get("thinking_level") or "",
            "thinking_budget": call_meta.get("thinking_budget") or 0,
            "temperature": sample.temperature,
            "fallback_used": bool(call_meta.get("fallback_used")),
            "finish_reason": call_meta.get("finish_reason") or "",
            "validation_errors": sample.validation_errors,
            "validation_warnings": sample.validation_warnings,
            "usage": call_meta.get("usage") or {},
        }
        lines = [
            f"# correction replay attempt {sample.attempt}",
            "",
            "```json",
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "## 模型响应",
            "",
            (sample.content or "").strip(),
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_summary(
        self,
        *,
        out_dir: Path,
        fixture: CorrectionFixture,
        fixture_src: Path,
        label: str,
        note: str,
        successes: Sequence[SampleResult],
        failures: Sequence[SampleResult],
        dry_run: bool,
        base_temperature: float = VALIDATION_BASE_TEMPERATURE,
    ) -> Path:
        from llm.prompt_compose import PROMPT_VERSION

        path = out_dir / "summary.md"
        lines = [
            f"# Session replay — {label}",
            "",
            f"- time: {datetime.now(timezone.utc).isoformat()}",
            f"- session: correction (R2)",
            f"- profile: {fixture.profile_id}",
            f"- chunk: {fixture.chunk_id}",
            f"- prompt_version: {PROMPT_VERSION}",
            f"- dry_run: {dry_run}",
            f"- base_temperature: {base_temperature:.2f}",
            f"- fixture: `{fixture_src}`",
            f"- frozen search_results chars: {len(fixture.search_results)}",
            f"- frozen entry_details chars: {len(fixture.entry_details)}",
            "",
        ]
        if not dry_run:
            all_samples = [*successes, *failures]
            lines.extend(
                [
                    "## Token 汇总",
                    "",
                    self._format_token_totals(all_samples),
                    "",
                ]
            )
        lines.extend(
            [
                "## 本轮改动重点",
                "",
                (note or "（未填写 — 用 `--note` 或事后编辑本文件）").strip(),
                "",
                "## 成功回复",
                "",
            ]
        )
        if not successes:
            lines.append("（无 — dry-run 或尚未采满）")
            lines.append("")
        for sample in successes:
            lines.append(f"### reply-{sample.index:02d} (attempt {sample.attempt})")
            lines.append("")
            if sample.path:
                lines.append(f"- file: `{sample.path.name}`")
            if sample.translated_path:
                lines.append(f"- translated: `{sample.translated_path.name}`")
            lines.append(f"- validation_ok: true")
            lines.append(f"- model: `{sample.model or '?'}`")
            lines.append(f"- temperature: {sample.temperature:.2f}")
            usage = (sample.call_meta or {}).get("usage") or {}
            if usage:
                lines.append(
                    "- tokens: "
                    f"in={usage.get('total_input_tokens', 0)} "
                    f"vis={usage.get('output_tokens', 0)} "
                    f"think={usage.get('thinking_tokens', 0)} "
                    f"out_total={usage.get('total_output_tokens', 0)}"
                )
            level = (sample.call_meta or {}).get("thinking_level") or ""
            if level:
                lines.append(f"- thinking_level: {level}")
            if sample.validation_warnings:
                lines.append(f"- warnings: {sample.validation_warnings}")
            translated = extract_tagged_block(sample.content, "translated")
            lines.append("")
            lines.append("<translated>")
            lines.append(translated)
            lines.append("</translated>")
            lines.append("")
        lines.append("## 失败尝试")
        lines.append("")
        if not failures:
            lines.append("（无）")
            lines.append("")
        else:
            for sample in failures:
                name = sample.path.name if sample.path else f"attempt{sample.attempt}"
                usage = (sample.call_meta or {}).get("usage") or {}
                think = usage.get("thinking_tokens", 0)
                lines.append(
                    f"- `{name}`: {'; '.join(sample.validation_errors) or 'unknown'}"
                    f" (model={sample.model or '?'}, temp={sample.temperature:.2f}, think={think})"
                )
            lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def _format_token_totals(samples: Sequence[SampleResult]) -> str:
        totals = {key: 0 for key in _USAGE_KEYS}
        models: Dict[str, int] = {}
        for sample in samples:
            usage = (sample.call_meta or {}).get("usage") or {}
            for key in _USAGE_KEYS:
                totals[key] += int(usage.get(key) or 0)
            model = sample.model or "?"
            models[model] = models.get(model, 0) + 1
        model_bits = ", ".join(f"{m}×{n}" for m, n in sorted(models.items()))
        return (
            f"- calls: {len(samples)}"
            + (f" ({model_bits})" if model_bits else "")
            + "\n"
            f"- input: {totals['total_input_tokens']} "
            f"(uncached={totals['uncached_input_tokens']}, "
            f"cached={totals['cached_input_tokens']}, "
            f"audio={totals['prompt_audio_tokens']})\n"
            f"- output: visible={totals['output_tokens']} "
            f"thinking={totals['thinking_tokens']} "
            f"total={totals['total_output_tokens']}"
        )
