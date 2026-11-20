"""A single-judged task's speakers (`correction_mode.SINGLE_JUDGED`): the correction windows' `spk` column written as
the speaker judge's sidecar, and the tracks a serial run hands on.

The windows answer who speaks in their `spk` column (`capableC-tracked`); what comes after them -- the `spk` cell of
`-annotated.csv`, the per-speaker exports, the knowledge update's lines -- reads speakers from the sidecar
(`speech.diarization.judgments`), so each committed window's answer is written there in the shape the judge writes
(`speaker_judge.answers.results`): one unit per correction window (the judge's own cut, `speaker_judge.windows.
split_windows`), marked `source: single-pass`, the whole file `route: single-pass`. The windows themselves never read
it (`window_speakers`: the invariant).

- A merged row's `spk` goes to each of its sources, its note with it as the result's `why` (the prompt asks a row
  that does not copy its track's main for one sentence on who it is and why); a `discard` row and the overlap a
  window re-reads from the one before write nothing for those rows. A source with no answer keeps its acoustic track.
- The reply's `<tracks>` table (who each track and label is) is the unit's `names`, as the judge writes its own
  (`speaker_judge.answers.parse_tracks`, fail-soft: a reply without one names nobody); a half adds to or replaces
  its parent's lines. A serial run shows the next window those names on the tracks it carries.
- A halved window (`-a` / `-b`) writes into its parent's unit as each half commits; the unit is the parent's rows,
  so a half that commits later only adds rows. A serial run hands each half its unit's carried tracks, a half
  split on an earlier run included.
- A `spk` that names no track this row may take, or no speaker at all (a window accepted with its `spk` column
  unfixed, `attempts`), writes nothing for its row; a window whose rows wrote nothing at all is a failed unit.
- Written on every commit, a window replayed from the ledger included: a run that stopped after the ledger took a
  window and before the sidecar did writes it on resume. `media` is what this run's windows hear (`--llm-media`): a
  window replayed after the switch changed is written with the new one -- rare, and only the audit field.
- A serial run hands each window the new tracks the window before opened or carried on (`carried`), as the judge's
  own serial run does (`speaker_judge.run.carried_tracks`, owner 2026-10-09); a parallel run hands none. A serial
  task resumed with `--llm-continuity parallel` replays its windows (a replay checks no `spk` cells), and a cell
  naming a carried track writes nothing for its row, which keeps its acoustic track.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ....speech.diarization import judgments as sidecar
from ....subtitles.stable_rows import read_rows
from ...chunking import SubtitleWindow
from ...output_protocol import SPK_ABSTAIN, SPK_NEW_PREFIX, SpkCheck
from ...window_speakers import SpeakerMaterial

#: What the sidecar says wrote it: the windows of a single-judged task (`route`, and each unit's `source`).
SINGLE_PASS = "single-pass"


class SpeakerBook:
    """The sidecar a single-judged task's windows write, shared by the window workers."""

    def __init__(self, stable_json: str | Path, windows: Sequence[SubtitleWindow], material: SpeakerMaterial, *,
                 media: str = "text") -> None:
        from ..speaker_judge.windows import split_windows

        #: what the windows heard, as the judge records its own (`text` / `audio`; `--llm-media`, video hears too)
        self.media = "text" if media == "text" else "audio"

        self.path = sidecar.sidecar_path(stable_json)
        self.stable_json = stable_json
        self.segments = read_rows(stable_json)
        self.material = material
        self.rows = material.rows
        self.ids = sidecar.segment_ids(self.segments)
        self.tracks = sidecar.profile_tracks([row.speaker for row in self.rows])
        self.units = split_windows(self.ids, windows)
        self.unit_of = {unit.chunk: index for index, unit in enumerate(self.units)}
        self.entries: List[Dict[str, Any]] = [
            {"id": unit.id, "chunk": unit.chunk, "first": self.ids[unit.low], "last": self.ids[unit.high - 1],
             "status": "pending", "source": SINGLE_PASS}
            for unit in self.units
        ]
        #: per unit: row id -> the `spk` cell its window answered and the row's note
        self.answers: Dict[int, Dict[str, tuple[str, str]]] = {}
        #: per unit: the reply's `<tracks>` table, track or label -> who
        self.names: Dict[int, Dict[str, str]] = {}
        #: per unit: the carried tracks its window was shown, which its `spk` may name
        self.shown: Dict[int, tuple] = {}
        self.lock = threading.Lock()

    def _index(self, window: SubtitleWindow) -> int | None:
        return self.unit_of.get(window.chunk_id.split("-", 1)[0])

    def with_carried(self, window: SubtitleWindow) -> SubtitleWindow:
        """``window`` (a serial run's) with the tracks the window before it opened or carried on: its unit's, so a
        half (split in this run or an earlier one) gets its parent's."""
        from ..speaker_judge.run import carried_tracks

        index = self._index(window)
        if index is None or window.speakers is None:
            return window
        with self.lock:
            found = sidecar.units(self.entries, self.ids) or []
            unit = next((unit for unit in found if unit.id == self.units[index].id), None)
            carried = tuple(carried_tracks(self.segments, found, unit)) if unit is not None else ()
            self.shown[index] = carried
        return replace(window, speakers=self.material.for_window(window, carried)) if carried else window

    def record(self, window: SubtitleWindow, rows: Sequence[Any], reply: str = "") -> None:
        """Write a committed window's `spk` answers (``rows``: its validated rows, harness ids) into its unit, the
        cells this window may name only (a replay checks none: `ResumeLedger.replay`), and the names of its
        ``reply``'s `<tracks>` table."""
        from ..speaker_judge.answers import parse_tracks

        index = self._index(window)
        if index is None or window.speakers is None:
            return
        overlap = {segment.id for segment in window.overlap_segments}
        check = SpkCheck(window.speakers.allowed)
        with self.lock:
            answers = self.answers.setdefault(index, {})
            for row in rows:
                cell = str(getattr(row, "spk", "") or "").strip()
                if getattr(row, "kind", "") != "sub" or not cell or check.errors(0, cell, row.source_ids):
                    continue
                note = str(getattr(row, "note", "") or "").strip()
                answers.update((source, (cell, note)) for source in row.source_ids if source not in overlap)
            self.names.setdefault(index, {}).update(parse_tracks(reply))
            self.entries[index] = self._entry(index, window)
            self._write()

    def _entry(self, index: int, window: SubtitleWindow) -> Dict[str, Any]:
        from ...prompt_compose import TRACKED_PROMPT_VERSION
        from ..speaker_judge.answers import results

        unit, answers = self.units[index], self.answers.get(index, {})
        carry = {track.track: (track.window, track.label, track.of) for track in self.shown.get(index, ())}
        verdicts = {f"q{row.id}": _verdict(*answers.get(row.id, (None, ""))) for row in self.rows[unit.low:unit.high]}
        entry: Dict[str, Any] = {
            "id": unit.id, "chunk": unit.chunk, "first": self.ids[unit.low], "last": self.ids[unit.high - 1],
            "source": SINGLE_PASS, "prompt_version": TRACKED_PROMPT_VERSION, "media": self.media,
            "key": sidecar.window_key(self.segments[unit.low:unit.high], self.ids[unit.low:unit.high], self.tracks),
        }
        if not answers:
            return {**entry, "status": "failed"}
        names = self.names.get(index) or {}
        return {**entry, "status": "success", **({"names": dict(names)} if names else {}),
                "results": results(self.rows, unit, verdicts, carry)}

    def report(self, unchecked: int, cut: int, warn: Any) -> Dict[str, Any]:
        """The task report's `speaker_judge_report` for these windows, in the judge's own fields
        (`speaker_judge.task.judge_task`): ``unchecked`` windows were accepted with their `spk` column unfixed,
        ``cut`` halved in this run."""
        from ..speaker_judge.task import new_tracks_of

        judged = sidecar.read_judgments(self.stable_json, warn=warn) if self.path.exists() else None
        status = [entry["status"] for entry in self.entries]
        return {"route": SINGLE_PASS, "windows": len(status), "judged": status.count("success"),
                "failed": status.count("failed"), "pending": status.count("pending"), "cut": cut,
                "moved": len(judged.moved) if judged else 0, "abstained": len(judged.abstained) if judged else 0,
                "new_tracks": new_tracks_of(judged), "spk_unchecked": unchecked}

    def _write(self) -> None:
        payload = {"version": sidecar.SIDECAR_VERSION, "route": SINGLE_PASS,
                   "rows_hash": sidecar.rows_hash(self.segments), "windows": self.entries}
        temporary = self.path.with_name(f".{self.path.name}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(self.path)


def _verdict(cell: str | None, why: str) -> Any:
    """A `spk` cell as the judge's verdict (`speaker_judge.answers.Verdict`), its row's note the reason: no answer
    keeps the row's track."""
    from ..speaker_judge.answers import Verdict

    if cell is None:
        return Verdict("keep")
    if cell == SPK_ABSTAIN:
        return Verdict("abstain", why=why)
    if cell.startswith(SPK_NEW_PREFIX):
        return Verdict("new", why=why, label=cell[len(SPK_NEW_PREFIX):].strip())
    return Verdict("track", why=why, track=int(cell[1:]))


def foreign_sidecar(stable_json: str | Path) -> Path | None:
    """A sidecar beside the stable that a single-judged task's windows did not write (a separate judge's, three
    passes' pass 1 with its splits, fills and inserts, or one that does not read): spent quota, which the task
    must not overwrite (owner Q11). None when there is none, or it is a single-judged task's own."""
    path = sidecar.sidecar_path(stable_json)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path
    return None if isinstance(payload, dict) and payload.get("route") == SINGLE_PASS else path
