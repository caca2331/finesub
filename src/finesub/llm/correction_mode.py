"""The correction task's recorded way of running (`correction-mode.json`), readable without the correction stack.

`stages/three_pass.py` makes the choice; this module only holds the record, so that the speech pipeline can read
which speaker judge a task's LLM stage ran with (`stages.task_diarize`) without importing the LLM layer. The two
window file names live here for the same reason: `started` needs them, and `stages/correction/commit.py` takes
them from here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finesub_bootstrap.fsops import write_atomic

# Mid-loop resume: each successfully committed window's raw response is appended
# here so a rerun replays completed windows instead of re-calling the LLM.
WINDOW_CACHE_FILENAME = "correction-windows.jsonl"
# The persisted window plan (model-routing v2): chunk-id stability across reruns.
WINDOW_PLAN_FILENAME = "correction-window-plan.json"

MODE_FILENAME = "correction-mode.json"
MODE_VERSION = 1
SINGLE = "single"
THREE_PASS = "three-pass"
# What each difficulty does about the passes (owner D1 and D2, `docs/llm_design_notes.md`「三个 pass 的决策记录」).
# Here rather than in `stages/three_pass.py` so that the speech pipeline reads the second without importing the
# correction layer (`stages.default_diarize`).
#: The difficulties that may run the three-pass split: the two for capable models.
THREE_PASS_DIFFICULTIES = frozenset({"quality", "intermediate"})
#: The difficulties a run that reaches the LLM stage turns the speaker judge on at when `--diarize` is unset.
DEFAULT_JUDGE_DIFFICULTIES = frozenset({"quality"})
#: The difficulties `--llm-proofread auto` runs pass 3 at, whichever way the task runs.
PROOFREAD_DIFFICULTIES = frozenset({"quality"})


@dataclass(frozen=True)
class PassChoice:
    """How a task runs: `mode` (`single` / `three-pass`), whether pass 3 follows, why this mode, and the speaker
    judge it runs with (`off` / `text` / `audio`; None in a record written before the judge was recorded).

    ``length_scale``: the task's length ratio, which its merge thresholds follow, and ``length_source`` where it
    came from (`length_tier.TaskLengthScale`). Settled after the task's style, so None in the choice made before
    it and in a record written before the ratio was; a resumed task keeps the recorded one whatever its flags
    and `config.toml` say now (`length_tier.task_length_scale`). A record without a ratio is resolved afresh
    rather than taken as 1.0: the change that added the ratio put it into each window's input hash (`scale=`),
    so such a task's delivered windows are other inputs and redone anyway, and nothing delivered depends on the
    old ratio.

    ``correct_only``: the task only corrects, its source being Chinese (``source_language``, None where it could
    not be told); False in a record written before the field, which translated (`three_pass.task_correct_only`)."""

    mode: str
    proofread: bool
    reason: str = ""
    speaker_judge: str | None = None
    length_scale: float | None = None
    length_source: str = ""
    correct_only: bool = False
    source_language: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {"mode": self.mode, "proofread": self.proofread, "reason": self.reason,
                "speaker_judge": self.speaker_judge, "length_scale": self.length_scale,
                "length_scale_source": self.length_source, "correct_only": self.correct_only,
                "source_language": self.source_language}


def recorded_choice(artifact_dir: str | Path | None) -> PassChoice | None:
    """The choice a task recorded, None when it recorded none (or the record does not read)."""
    if artifact_dir is None:
        return None
    try:
        payload = json.loads((Path(artifact_dir) / MODE_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != MODE_VERSION \
            or payload.get("mode") not in (SINGLE, THREE_PASS) or not isinstance(payload.get("proofread"), bool):
        return None
    judge = payload.get("speaker_judge")
    scale = payload.get("length_scale")
    scale = float(scale) if isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale > 0 else None
    language = payload.get("source_language")
    return PassChoice(payload["mode"], payload["proofread"], str(payload.get("reason") or ""),
                      judge if judge in ("off", "text", "audio") else None,
                      scale, str(payload.get("length_scale_source") or "") if scale is not None else "",
                      payload.get("correct_only") is True, language if isinstance(language, str) else None)


def started(artifact_dir: str | Path) -> bool:
    """Whether the task's correction windows have started (they start right after the choice is recorded)."""
    return any((Path(artifact_dir) / name).exists() for name in (WINDOW_PLAN_FILENAME, WINDOW_CACHE_FILENAME))


def record_choice(artifact_dir: str | Path, choice: PassChoice) -> None:
    """Write the task's choice, unless that very choice is already recorded."""
    if recorded_choice(artifact_dir) == choice:
        return
    # Atomic: a torn record reads as no choice while the windows have started, and a resume would fall back
    # to one pass.
    write_atomic(Path(artifact_dir) / MODE_FILENAME,
                 json.dumps({"version": MODE_VERSION, **choice.to_metadata()}, ensure_ascii=False, indent=2) + "\n")
