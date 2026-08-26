"""Disposable ASR checkpoint identity, persistence, and cleanup."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from ...reporting import current_reporter


# Old checkpoint formats are intentionally invalidated rather than migrated.
SCHEMA_VERSION = 2


def path_for_output(aligned_output: str | Path) -> Path:
    """Return the partial path for a completed aligned-output path."""

    return Path(aligned_output).with_suffix(".partial.json")


def _audio_identity(audio_path: str | Path) -> Dict[str, object]:
    path = Path(audio_path)
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path)}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_key(
    *,
    model_name: str,
    language: Optional[str],
    gap_sec: float,
    audio_path: str | Path,
    detect_disfluencies: bool = False,
    lang_redecode: bool = False,
) -> Dict[str, object]:
    """Return parameters a resumed run must agree on.

    ``detect_disfluencies`` changes the decoded word stream (``[*]`` blocks,
    refined leading starts), so partials from the other setting must not
    resume. ``lang_redecode`` changes what ``align_group`` results land in the
    partial (docs/asr-align.md, language-vote-flip redecode) — same rule.
    """

    return {
        "model": str(model_name),
        "language": str(language) if language else "",
        "gap_sec": round(float(gap_sec), 6),
        "audio": _audio_identity(audio_path),
        "detect_disfluencies": bool(detect_disfluencies),
        "lang_redecode": bool(lang_redecode),
    }


def intervals_digest(intervals: List[Dict[str, object]]) -> str:
    spans = [
        (
            round(float(interval.get("start", 0.0)), 3),
            round(float(interval.get("end", 0.0)), 3),
        )
        for interval in intervals
    ]
    payload = json.dumps(spans, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load(
    path: Path,
    fingerprint: Dict[str, object],
) -> Optional[Dict[str, object]]:
    """Return resumable state, or ``None`` for absent/stale/unreadable data."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("version") != SCHEMA_VERSION:
        return None
    if data.get("fingerprint") != fingerprint:
        return None
    if int(data.get("processed_intervals") or 0) <= 0:
        return None
    return data


def write(path: Path, payload: Dict[str, object]) -> None:
    """Atomically replace a partial without letting cache I/O kill the run."""

    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as exc:
        current_reporter().warning(
            "checkpoint-write-failed",
            f"could not write ASR checkpoint {path}: {exc}",
            impact="这次中断后要从头重跑识别",
        )
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def clear(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        current_reporter().warning(
            "checkpoint-remove-failed",
            f"could not remove ASR checkpoint {path}: {exc}",
        )
