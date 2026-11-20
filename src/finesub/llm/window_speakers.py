"""What a single-judged task's windows read about speakers (`capableC-tracked`, `correction_mode.SINGLE_JUDGED`):
the acoustic track cell of each line, each track's profile, and the people the task may hear.

The single pass that also names who speaks is shown what pass 1 of three passes is shown in text mode -- the cells
(`speaker_judge.prompt.cell`, with the marks), the profiles (`profiles`), the candidates (`candidates`) and, in a
serial run, the tracks the window before opened (`speaker_judge.prompt.Carried`, handed on by
`stages/correction/speaker_book.py`) -- built by the judge's own functions, so the two prompts read the same material.
No voice samples (owner 2026-11-10: text first; the audio judge runs three passes).

**The invariant this module keeps**: the rows and their speakers come from the judge's layer, the stable's own rows
with their acoustic labels (`subtitles.stable_rows.read_rows`), never from the product layer
(`speech.diarization.product.read_product`), so no `-speaker-judge.json` sidecar is read -- not the one the task's own
windows write, nor one an earlier run left. The correction's lines are loaded the same way (`chunking.
load_product_segments` with ``verdicts=False``), so the window's `turn` column, its track column and its line ids all
read the acoustic labels, and a resumed window renders as it first did. The sidecar is for what comes after the
windows (the `spk` column, the per-speaker exports).

The material rides on the window (`chunking.SubtitleWindow.speakers`) so every path that renders or validates a
window -- a halved one, a resumed one, a tool session's validator in another process (`agent_validators`) -- reads
the same thing. A halved window keeps its parent's profiles, which rank example lines outside the parent's rows,
until a serial run hands it its unit's carried tracks (`speaker_book.SpeakerBook.with_carried`): its material is then
rebuilt on its own rows, as any window's is.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

#: The correction prompt set that reads this material (`prompt_variants`).
TRACKED_VARIANT = "capableC-tracked"


@dataclass(frozen=True)
class WindowSpeakers:
    """One window's speaker material. ``cells`` and ``tracks`` are by source id: the track cell its `track` column
    shows, and the tracks that cell names (an alternative or a filled track included, `cell_tracks`). ``profile_tracks``
    are the tracks the profiles list; a row's `spk` may name any of them or any its own cells show."""

    cells: Mapping[str, str] = field(default_factory=dict)
    tracks: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    profile_tracks: Tuple[int, ...] = ()
    #: the lines whose cell shows a confidence mark (`（低）` and the rest)
    marks: frozenset[str] = frozenset()
    profiles: str = ""
    cast: str = ""
    title: str = ""

    def cell(self, source_id: str) -> str:
        """The `track` cell of a line: `??` (unassigned, `speaker_judge.prompt.PHRASES`) for one the judge's layer
        does not have, which cannot happen without a sidecar."""
        from .stages.speaker_judge.prompt import PHRASES

        return self.cells.get(source_id, PHRASES["unassigned"])

    def marked(self, source_ids: Sequence[str]) -> bool:
        """Whether a cell of these lines shows a confidence mark, so the prompt explains the brackets (as pass 1's)."""
        return bool(self.marks.intersection(source_ids))

    def allowed(self, source_ids: Sequence[str]) -> frozenset[int]:
        """The tracks a row over these sources may name in `spk`."""
        return frozenset(self.profile_tracks).union(*(self.tracks.get(key, ()) for key in source_ids))

    def to_dict(self) -> Dict[str, Any]:
        return {"cells": dict(self.cells), "tracks": {key: list(value) for key, value in self.tracks.items()},
                "profile_tracks": list(self.profile_tracks), "marks": sorted(self.marks),
                "profiles": self.profiles, "cast": self.cast, "title": self.title}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WindowSpeakers":
        return cls(
            cells={str(key): str(value) for key, value in dict(payload.get("cells") or {}).items()},
            tracks={str(key): tuple(int(track) for track in value)
                    for key, value in dict(payload.get("tracks") or {}).items()},
            profile_tracks=tuple(int(track) for track in payload.get("profile_tracks") or ()),
            marks=frozenset(str(key) for key in payload.get("marks") or ()),
            profiles=str(payload.get("profiles") or ""),
            cast=str(payload.get("cast") or ""),
            title=str(payload.get("title") or ""),
        )


@dataclass(frozen=True)
class SpeakerMaterial:
    """The task's judge-layer rows and candidates, from which each window's `WindowSpeakers` is cut."""

    rows: Tuple[Any, ...]
    cast: str
    title: str

    def for_window(self, window: Any, carried: Sequence[Any] = ()) -> WindowSpeakers:
        """``window``'s material; ``carried``: the tracks an earlier window opened (`speaker_judge.prompt.Carried`),
        listed after the acoustic ones and nameable in `spk`."""
        from ..speech.diarization.judgments import profile_tracks
        from .stages.speaker_judge.prompt import cell, cell_tracks, profiles
        from .stages.speaker_judge.run import marks_shown
        from .stages.speaker_judge.windows import JudgeWindow

        index = {row.id: position for position, row in enumerate(self.rows)}
        positions = [index[segment.id] for segment in window.segments if segment.id in index]
        if not positions:
            raise ValueError(f"window {window.chunk_id}: none of its lines is a row of the stable")
        show = marks_shown()
        unit = JudgeWindow(window.chunk_id, window.chunk_id, min(positions), max(positions) + 1)
        profile = profiles(self.rows, unit, carried)
        shared = {row.tracks[0] for row in self.rows if row.tracks}
        listed = tuple(sorted(shared | set(profile_tracks([row.speaker for row in self.rows]))
                              | {track.track for track in carried}))
        own = [segment.id for segment in window.segments if segment.id in index]
        return WindowSpeakers(
            cells={key: cell(self.rows[index[key]], marks=show) for key in own},
            tracks={key: tuple(cell_tracks(self.rows[index[key]], marks=show)) for key in own},
            profile_tracks=listed,
            marks=frozenset(key for key in own if cell(self.rows[index[key]], marks=show)
                            != cell(self.rows[index[key]], marks=False)),
            profiles=profile.text,
            cast=self.cast,
            title=self.title,
        )


def speaker_material(stable_json: str | Path, context_pack: Any, *, knowledge_root: Any, title: str) -> SpeakerMaterial:
    """The task's material, read from the judge's layer (see the module note). ``knowledge_root`` is None when the run
    does not read the knowledge base (no roster then, as for the judge); ``title`` is the judge's (the extra-info
    block, else the output's name). Raises ValueError where the set cannot run (`tracked_rows`): the task's choice
    keeps such a task off it."""

    from ..subtitles.stable_rows import read_rows
    from .stages.speaker_judge.prompt import PHRASES, candidates, rows_of
    from .stages.speaker_judge.roster import roster_entries
    from .stages.speaker_judge.run import entities_of, speakers_of

    segments = read_rows(stable_json)
    if not isinstance(segments, list) or not tracked_rows(segments):
        raise ValueError(f"{TRACKED_VARIANT} needs acoustic speaker labels, all in one block, on {stable_json}")
    rows = rows_of(segments)
    cast = candidates(entities_of(context_pack), roster_entries(knowledge_root, title), speakers_of(context_pack))
    return SpeakerMaterial(tuple(rows), "、".join(candidate.display for candidate in cast) or PHRASES["no_cast"],
                           title)


def tracked_rows(segments: Any) -> bool:
    """Whether these rows (`read_rows`) can be judged by the single pass: acoustic labels, all in one block
    (`judgments.judgeable`), not a cloud task's speakers (its prompt is written for acoustic tracks)."""

    from ..speech.diarization import judgments
    from .stages.speaker_judge.prompt import label_source, rows_of

    return isinstance(segments, list) and judgments.judgeable(segments) and label_source(rows_of(segments)) != "cloud"


def with_speakers(windows: Sequence[Any], material: SpeakerMaterial | None) -> list[Any]:
    """``windows`` carrying their speaker material (unchanged without one)."""

    if material is None:
        return list(windows)
    return [replace(window, speakers=material.for_window(window)) for window in windows]


def task_speakers(windows: Sequence[Any], stable_json: str | Path, *, tracked: bool, context_pack: Any,
                  knowledge_root: Any, title: str) -> tuple[list[Any], SpeakerMaterial | None]:
    """The task's windows with their material when it is single-judged (``tracked``), else as they are; and the
    material, for the windows a serial run hands carried tracks to (`speaker_book`)."""

    if not tracked:
        return list(windows), None
    material = speaker_material(stable_json, context_pack, knowledge_root=knowledge_root, title=title)
    return with_speakers(windows, material), material
