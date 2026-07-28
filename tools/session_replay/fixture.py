"""Correction R2 fixture load / extract / persist.

The fixture freezes the *rendered* ``search_results`` string (search + extract
merged), plus notes/entry_details/advice/context/window/media. Once on disk,
replay must not call the web search agent again.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence

from llm.audio_clips import compute_clip_range, probe_audio_duration
from llm.chunking import (
    SubtitleSegment,
    SubtitleWindow,
    load_segments_from_stable_json,
)
from llm.exchange_metadata import extract_tagged_block
from llm.profiles import TranslationProfile, resolve_profile
from llm.prompts import ContextPack
from llm.search_loop import EVIDENCE_PACK_HEADER
from llm.token_budget import CorrectionBudget
from .exchange_parse import split_exchange_sections

FIXTURE_VERSION = 1
FIXTURES_DIRNAME = "session-fixtures"
EMPTY_MARKERS = frozenset({"", "（无）", "（空）"})


@dataclass
class CorrectionFixture:
    session: str
    version: int
    profile_id: str
    chunk_id: str
    evidence_pack_mode: bool
    task_update_feedback: bool
    context_pack: Dict[str, Any]
    previous_advice: str
    entry_details: str
    query: Dict[str, Any]
    window: Dict[str, Any]
    media: Dict[str, Any]
    extra_style: str = ""
    common_mistakes_block: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    stable_json: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorrectionFixture":
        return cls(
            session=str(data.get("session") or "correction"),
            version=int(data.get("version") or FIXTURE_VERSION),
            profile_id=str(data.get("profile_id") or "mm-med"),
            chunk_id=str(data.get("chunk_id") or "0001"),
            evidence_pack_mode=bool(data.get("evidence_pack_mode")),
            task_update_feedback=bool(data.get("task_update_feedback")),
            context_pack=dict(data.get("context_pack") or {}),
            previous_advice=str(data.get("previous_advice") or ""),
            entry_details=str(data.get("entry_details") or ""),
            query=dict(data.get("query") or {}),
            window=dict(data.get("window") or {}),
            media=dict(data.get("media") or {}),
            extra_style=str(data.get("extra_style") or ""),
            common_mistakes_block=str(data.get("common_mistakes_block") or ""),
            source=dict(data.get("source") or {}),
            stable_json=str(data.get("stable_json") or ""),
        )

    @property
    def search_results(self) -> str:
        return str(self.query.get("search_results") or "")

    @property
    def window_notes(self) -> str:
        return str(self.query.get("window_notes") or "")

    def profile(self) -> TranslationProfile:
        route, _, level = self.profile_id.partition("-")
        if not route or not level:
            raise ValueError(f"Invalid profile_id in fixture: {self.profile_id!r}")
        return resolve_profile(route, level)


def apply_profile_override(
    fixture: CorrectionFixture, profile_id: str | None
) -> CorrectionFixture:
    """Rebuild under a different profile while keeping frozen injections."""

    override = (profile_id or "").strip()
    if not override or override == fixture.profile_id:
        return fixture
    route, sep, level = override.partition("-")
    if not sep or not route or not level:
        raise ValueError(
            f"Invalid --profile {override!r}; expected route-level "
            "(e.g. mm-low, mm-high, text-med)."
        )
    resolve_profile(route, level)  # validate
    return replace(fixture, profile_id=override)


def fixture_path(artifact_dir: Path, chunk_id: str) -> Path:
    return artifact_dir / FIXTURES_DIRNAME / f"correction-{chunk_id}.json"


def load_fixture(path: Path) -> CorrectionFixture:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Fixture is not a JSON object: {path}")
    return CorrectionFixture.from_dict(data)


def save_fixture(path: Path, fixture: CorrectionFixture) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fixture.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _normalize_block(text: str) -> str:
    body = (text or "").strip()
    return "" if body in EMPTY_MARKERS else body


def _find_run_dir(run: Path) -> Path:
    run = run.expanduser().resolve()
    if (run / "llm-artifacts").is_dir():
        return run
    if run.name.startswith("llm-artifacts"):
        return run.parent
    if (run.parent / "llm-artifacts").is_dir() and run.parent.name != "llm-artifacts":
        return run.parent
    return run


def resolve_run_layout(run: Path) -> Dict[str, Path]:
    """Map a run path to ``run_dir`` / ``artifact_dir`` / media / stable / research.

    ``run`` may be the run dir itself or a specific artifacts dir inside it
    (``llm-artifacts``, ``llm-artifacts-<label>``, ``<stem>.llm-artifacts``);
    pointing at the artifacts dir disambiguates runs that keep several."""

    raw = run.expanduser().resolve()
    run_dir = _find_run_dir(run)
    if raw.name.startswith("llm-artifacts") and raw.is_dir():
        artifact_dir = raw
    else:
        artifact_dir = run_dir / "llm-artifacts"
    if not artifact_dir.is_dir():
        # Standalone: ``out/<stem>/<stem>.llm-artifacts``
        candidates = list(run_dir.glob("*.llm-artifacts"))
        # Suffixed layout: ``llm-artifacts-<label>`` (newest wins).
        candidates += sorted(
            run_dir.glob("llm-artifacts-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates = [c for c in candidates if c.is_dir()]
        if candidates:
            artifact_dir = candidates[0]
            run_dir = artifact_dir.parent
    stem = run_dir.name
    stable = run_dir / f"{stem}-stable.json"
    if not stable.exists():
        matches = list(run_dir.glob("*-stable.json"))
        if matches:
            stable = matches[0]
    research = artifact_dir / f"{stem}-research-context.json"
    if not research.exists():
        research = run_dir / f"{stem}-research-context.json"
    if not research.exists():
        matches = []
        if artifact_dir.is_dir():
            matches = list(artifact_dir.glob("*-research-context.json"))
        if not matches:
            matches = list(run_dir.glob("*-research-context.json"))
        research = matches[0] if matches else research
    audio = run_dir / f"{stem}.ogg"
    if not audio.exists():
        for pattern in ("*.ogg", "*.flac", "*.wav", "*.mp3", "*.m4a", "*.aac", "audio.*"):
            hits = list(run_dir.glob(pattern))
            if hits:
                audio = hits[0]
                break
    video = run_dir / f"{stem}.mp4"
    if not video.exists():
        hits = list(run_dir.glob("*.mp4"))
        video = hits[0] if hits else video
    return {
        "run_dir": run_dir,
        "artifact_dir": artifact_dir,
        "stable_json": stable,
        "research_context": research,
        "audio_path": audio,
        "video_path": video,
    }


def find_correction_exchange(artifact_dir: Path, chunk_id: str) -> Path | None:
    exchanges = artifact_dir / "exchanges"
    if not exchanges.is_dir():
        return None
    # Prefer validation-ok attempt files; skip *-query-*
    pattern = f"*-correction-{chunk_id}-attempt*.md"
    candidates = sorted(exchanges.glob(pattern))
    if not candidates:
        # Older naming without -attempt
        candidates = [
            p
            for p in sorted(exchanges.glob(f"*-correction-{chunk_id}.md"))
            if "-query" not in p.name
        ]
    return candidates[0] if candidates else None


def find_query_exchange(artifact_dir: Path, chunk_id: str) -> Path | None:
    """Locate the query-round exchange for a given chunk (if present)."""

    exchanges = artifact_dir / "exchanges"
    if not exchanges.is_dir():
        return None
    candidates = sorted(exchanges.glob(f"*-correction-{chunk_id}-query-*.md"))
    return candidates[0] if candidates else None


def extract_indexes_from_exchange(exchange_path: Path) -> dict[str, str]:
    """Extract streamer_index and common_index blocks from a query exchange."""

    import re as _re

    text = exchange_path.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for name in ("streamer_index", "common_index"):
        m = _re.search(rf"<{name}>(.*?)</{name}>", text, _re.DOTALL)
        if m:
            result[name] = m.group(1).strip()
    return result


def _load_context_pack(research_path: Path) -> ContextPack:
    if not research_path.exists():
        return ContextPack()
    data = json.loads(research_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return ContextPack()
    # research-context may nest context under context_pack / pack
    for key in ("context_pack", "pack", "context"):
        nested = data.get(key)
        if isinstance(nested, Mapping) and (
            "general_context" in nested or "window_contexts" in nested
        ):
            return ContextPack.from_dict(nested)
    if "general_context" in data or "window_contexts" in data:
        return ContextPack.from_dict(data)
    return ContextPack()


def _infer_profile_id(layout: Mapping[str, Path], exchange_meta: Mapping[str, str]) -> str:
    research = layout["research_context"]
    if research.exists():
        try:
            data = json.loads(research.read_text(encoding="utf-8"))
            planning = data.get("planning") if isinstance(data, Mapping) else None
            if isinstance(planning, Mapping) and planning.get("profile_id"):
                return str(planning["profile_id"])
            if isinstance(data, Mapping) and data.get("profile_id"):
                return str(data["profile_id"])
        except (OSError, json.JSONDecodeError):
            pass
    # Heuristic: video present → mm-high
    if layout["video_path"].exists():
        return "mm-high"
    if layout["audio_path"].exists():
        return "mm-med"
    return "mm-low"


def _read_clip_start(artifact_dir: Path, chunk_id: str) -> float | None:
    cache = artifact_dir / "correction-windows.jsonl"
    if not cache.exists():
        return None
    for line in cache.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("chunk_id")) == chunk_id and "clip_start" in row:
            return float(row["clip_start"])
    return None


def _parse_payload_json(user_text: str) -> Dict[str, Any]:
    """Read the legacy pre-v40 dynamic JSON payload, if present."""

    marker = "动态窗口 payload："
    idx = user_text.find(marker)
    if idx < 0:
        return {}
    blob = user_text[idx + len(marker) :]
    # Payload is a JSON object after the marker (and before 最后提醒).
    end_marker = "最后提醒"
    end = blob.find(end_marker)
    if end >= 0:
        blob = blob[:end]
    blob = blob.strip()
    start = blob.find("{")
    if start < 0:
        return {}
    # Walk braces
    depth = 0
    for i, ch in enumerate(blob[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(blob[start : i + 1])
                except json.JSONDecodeError:
                    return {}
                return data if isinstance(data, dict) else {}
    return {}


def _segments_by_ids(
    all_segments: Sequence[SubtitleSegment], ids: Sequence[str]
) -> List[SubtitleSegment]:
    by_id = {seg.id: seg for seg in all_segments}
    missing = [sid for sid in ids if sid not in by_id]
    if missing:
        raise ValueError(f"stable.json missing source ids: {missing[:8]}...")
    return [by_id[sid] for sid in ids]


def _parse_ids_from_csv_block(block: str) -> List[str]:
    ids: List[str] = []
    for line in (block or "").splitlines():
        line = line.strip()
        if not line or line.startswith("<"):
            continue
        # Strip outer <asr_result> wrapping if present is already done by extract
        parts = line.split("|", 1)
        if parts and parts[0].strip().lower() in {"source_id", "local_id"}:
            continue
        if parts and parts[0].strip().isdigit():
            ids.append(parts[0].strip())
        elif parts and parts[0].strip():
            # ids may be non-digit in theory
            ids.append(parts[0].strip())
    return ids


def _extract_direct_input_block(text: str, tag: str) -> str:
    """Extract a prompt input block whose tags occupy their own lines.

    User prompts mention output tags such as ``<reasoning>`` inline before the
    input material. The response-oriented top-level tag parser intentionally
    treats everything after such a mention as nested, so prompt inputs use a
    stricter line-anchored matcher instead.
    """

    match = re.search(
        rf"(?ms)^[ \t]*<{re.escape(tag)}>[ \t]*\r?\n"
        rf"(.*?)^[ \t]*</{re.escape(tag)}>[ \t]*$",
        text or "",
    )
    return match.group(1).strip() if match else ""


def extract_fixture_from_exchange(
    *,
    run: Path,
    chunk_id: str,
    exchange_path: Path | None = None,
) -> CorrectionFixture:
    layout = resolve_run_layout(run)
    artifact_dir = layout["artifact_dir"]
    exchange_path = exchange_path or find_correction_exchange(artifact_dir, chunk_id)
    if exchange_path is None or not exchange_path.exists():
        raise FileNotFoundError(
            f"No correction R2 exchange for chunk {chunk_id} under {artifact_dir}/exchanges"
        )

    sections = split_exchange_sections(exchange_path.read_text(encoding="utf-8"))
    user = sections["user"]
    if not user.strip():
        raise ValueError(f"Exchange has empty user section: {exchange_path}")

    search_results = _normalize_block(extract_tagged_block(user, "search_results"))
    # Evidence pack may be injected under the same slot or a dedicated tag.
    if not search_results:
        search_results = _normalize_block(extract_tagged_block(user, "evidence_pack"))
    window_notes = _normalize_block(extract_tagged_block(user, "pre_round_notes"))
    entry_details = _normalize_block(extract_tagged_block(user, "entry_details"))
    previous_advice = _normalize_block(extract_tagged_block(user, "previous_advice"))
    if not search_results:
        raise ValueError(
            f"R2 exchange user section missing <search_results> body (needed to "
            f"freeze search+extract): {exchange_path}"
        )

    payload = _parse_payload_json(user)
    asr_inner = extract_tagged_block(
        str(payload.get("current_asr_csv") or ""), "asr_result"
    ) or _extract_direct_input_block(user, "asr_result")
    preceding_inner = extract_tagged_block(
        str(payload.get("preceding_context_csv") or ""), "preceding_context"
    ) or _extract_direct_input_block(user, "preceding_context")

    source_ids = [str(x) for x in (payload.get("source_ids") or [])]
    if not source_ids:
        # Prefer correction-windows cache
        cache_path = artifact_dir / "correction-windows.jsonl"
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("chunk_id")) == chunk_id:
                    source_ids = [str(x) for x in (row.get("source_ids") or [])]
                    break
    if not source_ids:
        source_ids = _parse_ids_from_csv_block(asr_inner)

    overlap_ids = [str(x) for x in (payload.get("overlap_source_ids") or [])]
    preceding_ids = _parse_ids_from_csv_block(preceding_inner)

    cached_clip_start = _read_clip_start(artifact_dir, chunk_id)

    stable = layout["stable_json"]
    if not stable.exists():
        raise FileNotFoundError(f"stable.json not found under {layout['run_dir']}")
    all_segments = load_segments_from_stable_json(stable)
    window_segments = _segments_by_ids(all_segments, source_ids)
    # preceding / overlap ids are kept for window rebuild; bodies load from stable.

    audio_path = layout["audio_path"]
    duration = probe_audio_duration(audio_path) if audio_path.exists() else None
    clip_start_calc, clip_end = compute_clip_range(
        window_segments,
        global_first_id=all_segments[0].id,
        global_last_id=all_segments[-1].id,
        audio_duration=duration,
    )
    # Prefer cached clip_start when present (exact original), keep computed end.
    clip_start = (
        float(cached_clip_start) if cached_clip_start is not None else clip_start_calc
    )

    budget_raw = payload.get("budget") if isinstance(payload.get("budget"), Mapping) else {}
    budget = {
        "input_tokens": int(budget_raw.get("input_tokens") or 0),
        "subtitle_input_tokens": int(budget_raw.get("subtitle_input_tokens") or 0),
        "estimated_output_tokens": int(budget_raw.get("estimated_output_tokens") or 0),
        "total_with_margin": int(
            budget_raw.get("total_with_margin")
            or (
                int(budget_raw.get("input_tokens") or 0)
                + int(budget_raw.get("estimated_output_tokens") or 0)
            )
        ),
        "token_counter_source": str(budget_raw.get("token_counter") or "fixture"),
    }

    profile_id = _infer_profile_id(layout, {})
    evidence_pack_mode = search_results.lstrip().startswith(EVIDENCE_PACK_HEADER[:20]) or (
        EVIDENCE_PACK_HEADER.split("\n", 1)[0] in search_results[:200]
    )
    # Feedback was on for the reference sample; detect from system text.
    task_update_feedback = "task_update_feedback" in sections["system"]

    context_pack = _load_context_pack(layout["research_context"])

    # Inject knowledge-base indexes from the query exchange (query-round-only
    # inputs not present in the R2 exchange the fixture is extracted from).
    query_exch = find_query_exchange(artifact_dir, chunk_id)
    if query_exch:
        for key, value in extract_indexes_from_exchange(query_exch).items():
            context_pack.setdefault(key, value)

    run_dir = layout["run_dir"]

    def _rel_or_abs(path: Path) -> str:
        if not path.exists():
            return ""
        try:
            return str(path.resolve().relative_to(run_dir.resolve()))
        except ValueError:
            return str(path.resolve())

    return CorrectionFixture(
        session="correction",
        version=FIXTURE_VERSION,
        profile_id=profile_id,
        chunk_id=chunk_id,
        evidence_pack_mode=evidence_pack_mode,
        task_update_feedback=task_update_feedback,
        context_pack=context_pack.to_dict(),
        previous_advice=previous_advice,
        entry_details=entry_details,
        query={
            "window_notes": window_notes,
            "search_results": search_results,
            "requested_entry_keys": [],
        },
        window={
            "chunk_id": chunk_id,
            "source_ids": source_ids,
            "overlap_source_ids": overlap_ids,
            "preceding_source_ids": preceding_ids,
            "boundary_reason": str(payload.get("boundary_reason") or "fixture"),
            "clip_start": clip_start,
            "clip_end": clip_end,
            "budget": budget,
        },
        media={
            "run_dir": str(run_dir),
            "audio_path": _rel_or_abs(layout["audio_path"]),
            "video_path": _rel_or_abs(layout["video_path"]),
        },
        source={
            "exchange": str(exchange_path),
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "frozen": ["search_results", "window_notes", "entry_details", "previous_advice"],
            "note": (
                "search_results is the full rendered R1 injection (search + extract); "
                "do not re-run web_search when replaying."
            ),
        },
        stable_json=str(stable),
    )


def ensure_correction_fixture(
    *,
    run: Path,
    chunk_id: str,
    force_extract: bool = False,
) -> tuple[CorrectionFixture, Path]:
    """Return fixture from disk, or extract from R2 exchange and persist.

    Does **not** call the search agent. If no exchange exists, raises with a
    message to run a one-shot R1 (handled by the correction adapter).
    """

    layout = resolve_run_layout(run)
    path = fixture_path(layout["artifact_dir"], chunk_id)
    if path.exists() and not force_extract:
        return load_fixture(path), path
    fixture = extract_fixture_from_exchange(run=run, chunk_id=chunk_id)
    save_fixture(path, fixture)
    return fixture, path


def build_window_from_fixture(fixture: CorrectionFixture) -> SubtitleWindow:
    stable = Path(fixture.stable_json)
    if not stable.exists():
        # Fall back relative to media.run_dir
        run_dir = Path(fixture.media.get("run_dir") or ".")
        candidates = list(run_dir.glob("*-stable.json"))
        if not candidates:
            raise FileNotFoundError(f"stable.json missing for fixture chunk {fixture.chunk_id}")
        stable = candidates[0]
    all_segments = load_segments_from_stable_json(stable)
    win = fixture.window
    source_ids = [str(x) for x in (win.get("source_ids") or [])]
    overlap_ids = [str(x) for x in (win.get("overlap_source_ids") or [])]
    preceding_ids = [str(x) for x in (win.get("preceding_source_ids") or [])]
    budget_raw = win.get("budget") if isinstance(win.get("budget"), Mapping) else {}
    budget = CorrectionBudget(
        input_tokens=int(budget_raw.get("input_tokens") or 0),
        subtitle_input_tokens=int(budget_raw.get("subtitle_input_tokens") or 0),
        estimated_output_tokens=int(budget_raw.get("estimated_output_tokens") or 0),
        total_with_margin=int(budget_raw.get("total_with_margin") or 0),
        token_counter_source=str(budget_raw.get("token_counter_source") or "fixture"),
    )
    return SubtitleWindow(
        chunk_id=str(win.get("chunk_id") or fixture.chunk_id),
        segments=_segments_by_ids(all_segments, source_ids),
        overlap_segments=_segments_by_ids(all_segments, overlap_ids)
        if overlap_ids
        else [],
        boundary_reason=str(win.get("boundary_reason") or "fixture"),
        budget=budget,
        clip_start=float(win.get("clip_start") or 0.0),
        clip_end=float(win.get("clip_end") or 0.0),
        preceding_segments=_segments_by_ids(all_segments, preceding_ids)
        if preceding_ids
        else [],
    )


def resolve_media_path(fixture: CorrectionFixture, key: str) -> Path | None:
    raw = str(fixture.media.get(key) or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path
    run_dir = Path(fixture.media.get("run_dir") or ".")
    candidate = run_dir / raw
    return candidate if candidate.is_file() else None
