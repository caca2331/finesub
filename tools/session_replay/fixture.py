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
import sys
from typing import Any, Dict, List, Mapping, Sequence

from finesub.media.clips import compute_clip_range, probe_audio_duration
from finesub.llm.chunking import (
    SubtitleSegment,
    SubtitleWindow,
    load_product_segments,
    load_segments_from_stable_json,
    product_windows,
)
from finesub.llm.exchange_metadata import (
    extract_tagged_block,
    extract_top_level_tagged_blocks,
)
from finesub.llm.routing.profiles import (
    DEFAULT_PROFILE,
    TranslationProfile,
    parse_profile_id,
    resolve_profile,
)
from finesub.llm.prompts import ContextPack
from finesub.llm.token_budget import CorrectionBudget
from .exchange_parse import split_exchange_sections

FIXTURE_VERSION = 1
FIXTURES_DIRNAME = "session-fixtures"
# Placeholders, never results to freeze. The last two are what the opening
# message of a task-flow window held in its search slot before v97 dropped the
# slot from production openings; exchange logs from those runs still carry them.
EMPTY_MARKERS = frozenset({
    "",
    "（无）",
    "（空）",
    "（本轮尚未检索；需要由 harness 代为取得的输入时按 system 指令先提请求）",
    "（本任务由你用授权的工具自行检索，没有回填轮）",
})
# What every registered session reads out of an artifact dir. Used only to pick
# between several candidate dirs left by one run -- not as a validity check.
REPLAY_INPUT_MARKERS = (
    "exchanges",
    "correction-windows.jsonl",
    "research-prepare-input.json",
)


@dataclass
class CorrectionFixture:
    session: str
    version: int
    profile_id: str
    chunk_id: str
    task_update_feedback: bool
    context_pack: Dict[str, Any]
    previous_advice: str
    entry_details: str
    query: Dict[str, Any]
    window: Dict[str, Any]
    media: Dict[str, Any]
    extra_style: str = ""
    style_block: str = ""
    source: Dict[str, Any] = field(default_factory=dict)
    stable_json: str = ""
    # A three-pass task's window: {"shape": pass 1's shape (None: no pass 1)}; the window is regrouped by pass 1's
    # sidecar on every build, as production does after pass 1 (`three_pass.pass2_plan`). Empty for one pass.
    three_pass: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CorrectionFixture":
        return cls(
            session=str(data.get("session") or "correction"),
            version=int(data.get("version") or FIXTURE_VERSION),
            profile_id=str(data.get("profile_id") or DEFAULT_PROFILE.profile_id),
            chunk_id=str(data.get("chunk_id") or "0001"),
            task_update_feedback=bool(data.get("task_update_feedback")),
            context_pack=dict(data.get("context_pack") or {}),
            previous_advice=str(data.get("previous_advice") or ""),
            entry_details=str(data.get("entry_details") or ""),
            query=dict(data.get("query") or {}),
            window=dict(data.get("window") or {}),
            media=dict(data.get("media") or {}),
            extra_style=str(data.get("extra_style") or ""),
            style_block=str(data.get("style_block") or ""),
            source=dict(data.get("source") or {}),
            stable_json=str(data.get("stable_json") or ""),
            three_pass=dict(data.get("three_pass") or {}),
        )

    @property
    def search_results(self) -> str:
        return str(self.query.get("search_results") or "")

    @property
    def window_notes(self) -> str:
        return str(self.query.get("window_notes") or "")

    def profile(self) -> TranslationProfile:
        # parse_profile_id also accepts the retired route-level preset names,
        # so fixtures frozen before the switch refactor keep loading.
        return parse_profile_id(self.profile_id)


def apply_profile_override(
    fixture: CorrectionFixture, profile_id: str | None
) -> CorrectionFixture:
    """Rebuild under a different profile while keeping frozen injections."""

    override = (profile_id or "").strip()
    if not override or override == fixture.profile_id:
        return fixture
    try:
        resolved = parse_profile_id(override)  # validate (accepts legacy names)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --profile {override!r}; expected a switch vector like "
            "'media=audio,retrieval=local,difficulty=quality'."
        ) from exc
    # Normalise: a legacy name is stored as the vector it means.
    return replace(fixture, profile_id=resolved.profile_id)


def fixture_path(artifact_dir: Path, chunk_id: str) -> Path:
    return artifact_dir / FIXTURES_DIRNAME / f"correction-{chunk_id}.json"


def load_fixture(path: Path) -> CorrectionFixture:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Fixture is not a JSON object: {path}")
    fixture = CorrectionFixture.from_dict(data)
    raw_windows = fixture.context_pack.get("window_contexts") or []
    if not isinstance(raw_windows, Mapping):
        return fixture

    # Fixtures predate interval-addressed research notes.  Unlike production,
    # replay knows both the requested frozen chunk and its exact source ids, so
    # it can bind that one note safely.  Keep every other legacy note unbound:
    # ContextPack retains it for audit but never injects it into a window.
    source_ids = [str(item) for item in fixture.window.get("source_ids") or []]
    first = source_ids[0] if source_ids else ""
    last = source_ids[-1] if source_ids else ""
    payload = dict(fixture.context_pack)
    payload["window_contexts"] = [
        {
            "window_id": str(window_id),
            "context": str(context),
            **(
                {"first_source_id": first, "last_source_id": last}
                if str(window_id) == fixture.chunk_id and first and last
                else {}
            ),
        }
        for window_id, context in raw_windows.items()
    ]
    return replace(fixture, context_pack=ContextPack.from_dict(payload).to_dict())


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


def _replay_input_score(path: Path) -> int:
    """How many replay inputs this candidate artifact dir actually holds."""

    return sum(1 for name in REPLAY_INPUT_MARKERS if (path / name).exists())


def resolve_run_layout(run: Path) -> Dict[str, Path]:
    """Map a run path to ``run_dir`` / ``artifact_dir`` / media / stable / research.

    ``run`` may be the run dir itself or a specific artifacts dir inside it
    (``llm-artifacts``, ``llm-artifacts-<label>``, ``<stem>.llm-artifacts``,
    ``<stem>-artifacts``); pointing at the artifacts dir disambiguates runs that
    keep several."""

    raw = run.expanduser().resolve()
    run_dir = _find_run_dir(run)
    if raw.name.startswith("llm-artifacts") and raw.is_dir():
        artifact_dir = raw
    else:
        artifact_dir = run_dir / "llm-artifacts"
    if not artifact_dir.is_dir():
        # Standalone: ``out/<stem>/<stem>.llm-artifacts``, the suffixed layout
        # ``llm-artifacts-<label>``, and what ``--task-artifact-dir`` is usually
        # pointed at in this repo: ``<stem>-artifacts``.
        candidates = [
            path
            for pattern in ("*.llm-artifacts", "llm-artifacts-*", "*-artifacts")
            for path in run_dir.glob(pattern)
            if path.is_dir()
        ]
        if candidates:
            # One run can leave two of them: the dir ``--task-artifact-dir``
            # named (exchanges + per-round input dumps) and the thin one ``-o``
            # derives (research context only). Replay needs the former, so rank
            # by the inputs actually present and only then by recency -- glob
            # order would hand back the thin one and silently degrade every
            # fixture to the research-context fallback.
            artifact_dir = max(
                candidates,
                key=lambda path: (_replay_input_score(path), path.stat().st_mtime),
            )
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
        if not matches:
            # The research context is the one artifact ``-o`` derives on its
            # own, so it commonly sits in the *other* artifact dir -- the thin
            # one the ranking above deliberately passed over.
            matches = sorted(run_dir.glob("*-artifacts/*-research-context.json"))
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
            if "-request" not in p.name
        ]
    return candidates[0] if candidates else None


def find_request_exchange(artifact_dir: Path, chunk_id: str) -> Path | None:
    """Locate the request-turn exchange for a given chunk (if present)."""

    exchanges = artifact_dir / "exchanges"
    if not exchanges.is_dir():
        return None
    candidates = sorted(exchanges.glob(f"*-correction-{chunk_id}-request-*.md"))
    return candidates[0] if candidates else None


def extract_indexes_from_exchange(exchange_path: Path) -> dict[str, str]:
    """The rendered `<...._index>` blocks of a request-turn exchange, by key.

    One block per matchable category, named by the category -- so this
    reads the category list rather than the two names it used to know,
    and a fixture recorded before a category existed simply has no block
    for it.
    """

    import re as _re

    text = exchange_path.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for name in _index_keys():
        m = _re.search(rf"<{name}>(.*?)</{name}>", text, _re.DOTALL)
        if m:
            result[name] = m.group(1).strip()
    return result


def _index_keys() -> tuple[str, ...]:
    from finesub.llm.knowledge.node.presets import MATCHABLE_CATEGORIES

    return tuple(f"{category}_index" for category in MATCHABLE_CATEGORIES)


def indexes_from_fixture(fixture) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """`{category: index text}` for whatever the fixture recorded.

    Two shapes are live, and neither is legacy. The round-input dumps store
    the mapping the builders take (`knowledge_indexes`, keyed by bare
    category); `extract_indexes_from_exchange` reads the rendered
    `<streamer_index>` tags out of an exchange and is flat by construction.
    Reading only the flat one replayed the current dumps with EMPTY indexes and
    said nothing about it (review 2026-09-07, third round).
    """

    from finesub.llm.knowledge.node.presets import MATCHABLE_CATEGORIES

    nested = fixture.get("knowledge_indexes") or {}
    out: dict[str, str] = {}
    for category in MATCHABLE_CATEGORIES:
        body = ""
        if isinstance(nested, Mapping):
            body = str(nested.get(category) or "")
        if not body.strip():
            body = str(fixture.get(f"{category}_index", "") or "")
        if body.strip():
            out[category] = body
    return out


def _load_context_pack(
    research_path: Path,
    *,
    legacy_window_id: str = "",
    legacy_source_ids: Sequence[str] = (),
) -> ContextPack:
    if not research_path.exists():
        return ContextPack()
    data = json.loads(research_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        return ContextPack()
    # research-context may nest context under context_pack / pack.
    # Production deliberately rejects the old id->text map because it cannot
    # place those notes after window geometry changes. Replay has one narrower,
    # safe migration: the requested legacy chunk and its frozen source ids are
    # both known, so bind that one note to this fixture's interval. Preserve
    # other notes as unbound audit data; they are never injected.
    for key in ("context_pack", "pack", "context"):
        nested = data.get(key)
        if isinstance(nested, Mapping) and (
            "general_context" in nested or "window_contexts" in nested
        ):
            payload = dict(nested)
            break
    else:
        payload = (
            dict(data)
            if "general_context" in data or "window_contexts" in data
            else {}
        )
    raw_windows = payload.get("window_contexts") or []
    if isinstance(raw_windows, Mapping):
        first = str(legacy_source_ids[0]) if legacy_source_ids else ""
        last = str(legacy_source_ids[-1]) if legacy_source_ids else ""
        payload["window_contexts"] = [
            {
                "window_id": str(window_id),
                "context": str(context),
                **(
                    {"first_source_id": first, "last_source_id": last}
                    if str(window_id) == legacy_window_id and first and last
                    else {}
                ),
            }
            for window_id, context in raw_windows.items()
        ]
    if payload:
        return ContextPack.from_dict(payload)
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
    # Heuristic from what media the run left behind.
    if layout["video_path"].exists():
        return resolve_profile("video", "local", "quality").profile_id
    if layout["audio_path"].exists():
        return resolve_profile("audio", "local", "quality").profile_id
    return resolve_profile("text", "local", "quality").profile_id


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


def _resolve_preceding_ids(
    parsed: Sequence[str],
    source_ids: Sequence[str],
    all_ids: Sequence[str],
) -> List[str]:
    """Map the prompt's preceding-context numbering back to stable source ids.

    The correction prompt renumbers every window: target rows are ``1..N`` and
    the read-only preceding rows count backwards as ``1-M..0``. Those display
    numbers are not source ids, so taking them literally asks stable.json for
    id ``-9`` and every window but the first fails to extract. Recover the real
    ids by position: the preceding rows are exactly the rows immediately before
    the window's first source.
    """

    parsed = [str(item) for item in parsed]
    known = set(all_ids)
    if parsed and all(item in known for item in parsed):
        # Payload-era exchanges carried real ids here; keep them verbatim.
        return parsed
    if not parsed or not source_ids:
        return []
    try:
        first = list(all_ids).index(str(source_ids[0]))
    except ValueError:
        return []
    return list(all_ids[max(0, first - len(parsed)) : first])


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
    # Task flow: the opening message carries the window's inputs (CSV, read-only
    # context, advice, the *carried* entries) with placeholders in the retrieval
    # slots; the fill turn -- a second user section, present only after a
    # request batch -- carries what the batch produced. A direct delivery has
    # no fill: its retrieval stays the placeholder, i.e. nothing.
    users = sections.get("users") or [user]
    fill = users[-1] if len(users) > 1 else ""

    # Search results come from the fill; openings before v97 carried a
    # placeholder block, and a replayed fixture's opening its frozen results.
    # A window with neither searched nothing.
    search_blocks = extract_top_level_tagged_blocks(user, "search_results")
    fill_search = extract_top_level_tagged_blocks(fill, "search_results") if fill else []
    search_results = _normalize_block(
        fill_search[0] if fill_search else (search_blocks[0] if search_blocks and not fill else "")
    )
    window_notes = _normalize_block(extract_tagged_block(user, "pre_round_notes"))
    if fill and not window_notes:
        # The request turn's notes live in the replayed assistant message.
        window_notes = _normalize_block(
            extract_tagged_block(str(sections.get("assistant") or ""), "window_notes")
        )
    fill_entries = extract_tagged_block(fill, "entry_details") if fill else ""
    entry_details = _normalize_block(fill_entries or extract_tagged_block(user, "entry_details"))
    previous_advice = _normalize_block(extract_tagged_block(user, "previous_advice"))

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
    parsed_preceding = _parse_ids_from_csv_block(preceding_inner)
    three_pass = _three_pass_record(artifact_dir)
    if three_pass:
        # The ledger names a grouped window's lines by each group's first row; the window's own rows are the
        # ones production planned before pass 1 grouped them.
        source_ids, overlap_ids = _planned_rows(artifact_dir, chunk_id, source_ids)

    cached_clip_start = _read_clip_start(artifact_dir, chunk_id)

    stable = layout["stable_json"]
    if not stable.exists():
        raise FileNotFoundError(f"stable.json not found under {layout['run_dir']}")
    # the ledger names the lines the correction read (`chunking.load_product_segments`); a three-pass window's
    # planned rows are the judge's, spelled as those lines when the window is built
    all_segments = load_segments_from_stable_json(stable) if three_pass else load_product_segments(stable)[0]
    window_segments = _segments_by_ids(all_segments, source_ids)
    # preceding / overlap ids are kept for window rebuild; bodies load from stable.
    preceding_ids = _resolve_preceding_ids(
        parsed_preceding, source_ids, [seg.id for seg in all_segments]
    )

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
        "token_counter_source": str(budget_raw.get("token_counter") or "fixture"),
    }

    profile_id = _infer_profile_id(layout, {})
    # Feedback was on for the reference sample; detect from system text.
    task_update_feedback = "task_update_feedback" in sections["system"]

    # The fixture stores the pack as a plain dict: the query replay reads the
    # knowledge indexes off it with ``fixture.context_pack.get(...)``, and
    # ContextPack itself has no slot for them.
    context_pack = _load_context_pack(
        layout["research_context"],
        legacy_window_id=chunk_id,
        legacy_source_ids=source_ids,
    ).to_dict()

    # Inject knowledge-base indexes from the request-turn exchange (the
    # delivery exchange the fixture is extracted from carries none).
    query_exch = find_request_exchange(artifact_dir, chunk_id)
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
        task_update_feedback=task_update_feedback,
        context_pack=context_pack,
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
            "length_scale": _recorded_length_scale(artifact_dir),
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
        three_pass=three_pass,
    )


def _correction_mode(artifact_dir: Path) -> Dict[str, Any]:
    """The task's last `correction_mode` artifact's payload (empty without one)."""
    from finesub.llm.knowledge.base import TASK_ARTIFACT_FILENAME

    path = artifact_dir / TASK_ARTIFACT_FILENAME
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.is_file() else []
    return next((record.get("payload") or {} for record in reversed(records)
                 if record.get("kind") == "correction_mode"), {})


def _three_pass_record(artifact_dir: Path) -> Dict[str, Any]:
    """`{"shape": ...}` when the task ran three passes (its last `correction_mode` artifact), else empty."""
    from finesub.llm.stages.three_pass import THREE_PASS

    mode = _correction_mode(artifact_dir)
    return {"shape": mode.get("shape")} if mode.get("mode") == THREE_PASS else {}


def _recorded_length_scale(artifact_dir: Path) -> float:
    """The length ratio the task recorded with its choice of passes; a task recorded before the ratio was ran
    at 1.0."""
    return float(_correction_mode(artifact_dir).get("length_scale", 1.0))


def _planned_rows(artifact_dir: Path, chunk_id: str, fallback: List[str]) -> tuple[List[str], List[str]]:
    """A window's rows and overlap rows as planned (`correction-window-plan.json`); a fast run plans one
    window and writes no plan, so its rows stay `fallback`."""
    from finesub.llm.stages.correction.commit import WINDOW_PLAN_FILENAME

    path = artifact_dir / WINDOW_PLAN_FILENAME
    if not path.is_file():
        return fallback, []
    windows = (json.loads(path.read_text(encoding="utf-8")).get("plan") or {}).get("windows") or []
    planned = next((window for window in windows if str(window.get("chunk_id")) == chunk_id), None)
    if planned is None:
        raise ValueError(f"window {chunk_id} is not in {path.name} (a half of a split window cannot be replayed "
                         "as three passes' pass 2)")
    return [str(x) for x in planned.get("segment_ids") or []], [str(x) for x in planned.get("overlap_ids") or []]


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
    all_segments = load_segments_from_stable_json(stable) if fixture.three_pass else load_product_segments(stable)[0]
    win = fixture.window
    source_ids = [str(x) for x in (win.get("source_ids") or [])]
    overlap_ids = [str(x) for x in (win.get("overlap_source_ids") or [])]
    preceding_ids = [str(x) for x in (win.get("preceding_source_ids") or [])]
    budget_raw = win.get("budget") if isinstance(win.get("budget"), Mapping) else {}
    budget = CorrectionBudget(
        input_tokens=int(budget_raw.get("input_tokens") or 0),
        subtitle_input_tokens=int(budget_raw.get("subtitle_input_tokens") or 0),
        estimated_output_tokens=int(budget_raw.get("estimated_output_tokens") or 0),
        token_counter_source=str(budget_raw.get("token_counter_source") or "fixture"),
    )
    window = SubtitleWindow(
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
        # the task's recorded ratio (a fixture extracted before it was recorded: 1.0, as its task ran)
        length_scale=float(win.get("length_scale", 1.0)),
    )
    if not fixture.three_pass:
        return window
    from finesub.llm.prompt_constants import Thresholds
    from finesub.llm.stages.three_pass import pass2_plan

    window = product_windows([window], *load_product_segments(stable))[0]
    grouped, _, _ = pass2_plan([window], stable, fixture.three_pass.get("shape"),
                               warn=lambda message: print(f"Warning: {message}", file=sys.stderr),
                               thresholds=Thresholds.at(window.length_scale))
    if not grouped:
        raise ValueError(f"pass 1 dropped every core line of window {window.chunk_id}: nothing to replay")
    return grouped[0]


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
