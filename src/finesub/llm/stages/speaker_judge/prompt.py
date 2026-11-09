"""What the speaker judge is shown.

Ported from the experiment line that measured best, round 9's judge on raw ASR rows
(`tools/diarization_explore/window_judge_r9.py`, with round 3's profiles and round 8's marks), minus
the one sentence that only held for corrected text. What production changed, cold-read and accepted
(`docs/diarization-findings.md`「生产裁判的验收」): a generic scene sentence plus the preparation
task's summary instead of a per-material scene; the task's title instead of 「标题略」; names outside
the list allowed; and, since owner 2026-10-09, the candidate list itself instead of a cast another call
picked from it per window (the preparation task already lists who speaks in the file).

Three passes (`docs/llm_harness_behavior.md`「说话人裁判」) add pass 1's shape section to this prompt: per
row, whether it joins the line before it into one subtitle (`join`), may be merged with it by the correction
(`may_join`), or is left out (`drop`); the rows then carry their duration and the gap before them. Ported from
`tools/diarization_explore/judge_shape.py`, as tried on the test windows.

The shape section's criteria (who is speaking; when a row joins, may join or is dropped) are shared with the
single pass prompt, which makes the same calls within the correction (`finesub.llm.shape_criteria`): this
prompt words them for a step of its own and wraps them in its JSON fields.

The whole sections live in `prompt_templates/speaker_*`; the phrase-level pieces below are the cells,
profile lines and the audio table's rows, rendered per row (the precedent is `prompt_compose`'s phrase layer).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ....speech.diarization.judgments import segment_ids
from ....speech.diarization.labels import SegmentSpeaker, filled_track, speaker_of
from ....subtitles import line_groups
from ...prompt_compose import load_prompt_template, task_mode_texts
from ...shape_criteria import shape_criteria
from .windows import JudgeWindow

SYSTEM_TEMPLATE = "speaker_judge_system_v3.md"
USER_TEMPLATE = "speaker_judge_user_v3.md"
SHAPE_TEMPLATE = "speaker_judge_shape_v1.md"
SHAPE_AUDIO_TEMPLATE = "speaker_judge_shape_audio_v1.md"
#: The opening task sentence's addition when the judge also shapes the subtitles (pass 1).
SHAPE_TASK_TEMPLATE = "speaker_judge_shape_task_v1.md"
AUDIO_NOTE_TEMPLATE = "speaker_judge_audio_note_v1.md"
MARKS_NOTE_TEMPLATE = "speaker_judge_marks_note_v1.md"
SUMMARY_TEMPLATE = "speaker_judge_summary_v1.md"
AUDIO_LAYOUT_TEMPLATE = "speaker_judge_audio_layout_v1.md"
AUDIO_SILENT_TEMPLATE = "speaker_judge_audio_silent_v1.md"

#: Example lines per track profile, voice samples per track, and what qualifies (round 3).
EXAMPLE_LINES = 3
MIN_EXAMPLE_CHARS = 8
SAMPLES_PER_TRACK = 2
SAMPLE_SECONDS = (2.0, 5.0)
#: Silence between the audio file's pieces, and the margin around the window's recording.
SAMPLE_GAP = 1.5
EDGE_PAD = 0.5

PHRASES = {
    "unassigned": "??",
    "low": "（低）",
    "low_alternative": "（低，备选 s{alternative}）",
    "filled": "s{track}（补）",
    "profile_line": "- s{track}：占语音 {share:.0f}%。例句：{examples}",
    "carried_line": "- s{track}：前面的窗口里新建的轨（{who}）。例句：{examples}",
    "carried_named": "前面的窗口猜是{name}，未确认",
    "carried_unknown": "前面的窗口认不出是谁",
    "no_examples": "（没有可靠的例句）",
    "sample_row": "| s{track} 的声音样本 | {start:.1f}–{end:.1f} 秒 |",
    "window_row": "| 这一段字幕的录音 | {start:.1f}–{end:.1f} 秒 |",
    "no_cast": "（无）",
    "unknown_person": "不明",
    "row_format": "开始秒|轨号|文字",
    "shaped_row_format": "开始秒|时长|gap|轨号|文字",
}

#: The judge's field line: alone, with pass 1's shape section, and with the shape section heard (`split` and the rest).
FIELDS_TEMPLATES = {"alone": "speaker_judge_fields_v1.md", "shaped": "speaker_judge_fields_shaped_v1.md",
                    "heard": "speaker_judge_fields_shaped_audio_v1.md"}
SPLIT_HINT_TEMPLATE = "speaker_judge_split_hint_v1.md"
#: The audio note's answer when content and voice disagree: the shape section keeps `abstain` for a possible
#: change of speaker (owner B2), the plain judge's prompt is unchanged.
CONFLICT_TEMPLATES = {"shaped": "speaker_judge_conflict_shaped_v1.md", "alone": "speaker_judge_conflict_v1.md"}
INSERT_HINT_TEMPLATE = "speaker_judge_insert_hint_v1.md"
#: How the shape section tells one person from another: by the judge's own verdicts, worded for where the tracks
#: come from (`SOURCE_WORDS`); a cloud task's tracks are its transcription's speakers, whose typical error is
#: another. The single pass prompt words this sentence its own way (`prompt_compose.SHAPE_SAME_PERSON_FRAGMENT`).
SHAPE_SAME_PERSON_TEMPLATES = {"acoustic": "speaker_judge_shape_same_person_v1.md",
                               "cloud": "speaker_judge_shape_same_person_cloud_v1.md"}

#: What the judge is told about where the tracks come from: the acoustic layer (a local task), or a cloud task's
#: transcription (its lines' `line_speaker`), which splits a person rather than merging people, has no `??`
#: or `s5+s4` cells and no samples mixed from two people. Template files; None leaves the slot empty.
SOURCE_WORDS = {
    "acoustic": {
        "tracks_intro": "speaker_judge_source_acoustic_intro_v1.md",
        "tracks_errors": "speaker_judge_source_acoustic_errors_v1.md",
        "profile_examples": "speaker_judge_source_acoustic_profile_v1.md",
        "tracks_cells": "speaker_judge_source_acoustic_cells_v1.md",
        "tracks_trust": "speaker_judge_source_acoustic_trust_v1.md",
        "keep_cells": "speaker_judge_source_acoustic_keep_v1.md",
        "audio_mix": "speaker_judge_source_acoustic_mix_v1.md",
    },
    "cloud": {
        "tracks_intro": "speaker_judge_source_cloud_intro_v1.md",
        "tracks_errors": "speaker_judge_source_cloud_errors_v1.md",
        "profile_examples": "speaker_judge_source_cloud_profile_v1.md",
        "tracks_cells": None,
        "tracks_trust": "speaker_judge_source_cloud_trust_v1.md",
        "keep_cells": None,
        "audio_mix": None,
    },
}


def label_source(rows: Sequence["Row"]) -> str:
    """`cloud` when the rows carry a cloud transcription's speakers, else `acoustic` (`SOURCE_WORDS`)."""
    return "cloud" if line_groups.speakers([row.group_key for row in rows]) else "acoustic"


def _phrase(template: str | None) -> str:
    """One sentence of the judge's prompt, kept in its own template file."""
    return load_prompt_template(template).strip() if template else ""


#: Candidates the prompt lists at most: a long file researched in chunks can name dozens of people.
MAX_CANDIDATES = 30


@dataclass(frozen=True)
class Row:
    id: str
    start: float
    end: float
    text: str
    speaker: SegmentSpeaker
    #: the group key (`subtitles.line_groups.key_of`): a cloud task's `line_speaker`, or the overlap lane
    group_key: str | None = None

    @property
    def seconds(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def tracks(self) -> list[int]:
        tracks = self.speaker.extras.get("tracks")
        if not tracks and self.speaker.track is not None:
            tracks = [self.speaker.track]
        return [int(track) for track in tracks or ()]

    @property
    def low(self) -> bool:
        mark = self.speaker.extras.get("mark")
        return isinstance(mark, Mapping) and mark.get("tier") == "low"


def rows_of(segments: Sequence[Mapping[str, Any]]) -> list[Row]:
    return [Row(key, float(segment.get("start", 0.0)), float(segment.get("end", 0.0)),
                str(segment.get("text", "")).strip(), speaker_of(segment), line_groups.key_of(segment))
            for key, segment in zip(segment_ids(segments), segments)]


def cell(row: Row, *, marks: bool) -> str:
    """The track cell the judge reads: `s5+s4`, `??` unassigned; with marks, round 8's wording."""
    base = "+".join(f"s{track}" for track in row.tracks) or PHRASES["unassigned"]
    if not marks:
        return base
    filled = filled_track(row.speaker)
    if filled is not None:
        return PHRASES["filled"].format(track=filled)
    if row.low:
        alternative = row.speaker.extras["mark"].get("alternative")
        return base + (PHRASES["low"] if alternative is None
                       else PHRASES["low_alternative"].format(alternative=int(alternative)))
    return base


def cell_tracks(row: Row, *, marks: bool) -> list[int]:
    """The track ids `cell` shows for this row -- the judge may answer any of them, a track that never
    leads a line (an alternative, a filled track, a second track) included."""
    filled = filled_track(row.speaker) if marks else None
    if filled is not None:
        return [int(filled)]
    alternative = row.speaker.extras["mark"].get("alternative") if marks and row.low else None
    return [*row.tracks, *([int(alternative)] if alternative is not None else [])]


@dataclass(frozen=True)
class Candidate:
    """A person the judge may name: the name to copy, and what else they go by (shown in brackets)."""

    name: str
    others: tuple[str, ...] = ()

    @property
    def display(self) -> str:
        return f"{self.name}（{'、'.join(self.others)}）" if self.others else self.name


def candidates(entities: Sequence[Mapping[str, Any]], roster: Sequence[tuple[str, tuple[str, ...], str]],
               speakers: Sequence[Mapping[str, Any]] = ()) -> list[Candidate]:
    """Who the preparation task saw speaking first, then the knowledge base's streamers, then the people
    it found mentioned; a name once, `MAX_CANDIDATES` at most. The judge may still name anyone else."""
    out: dict[str, Candidate] = {}
    for speaker in speakers:
        name = str(speaker.get("name") or "").strip()
        if name:
            out.setdefault(name, Candidate(name))
    for key, others, _ in roster:
        out.setdefault(key, Candidate(key, tuple(others)))
    for entity in entities:
        name = str(entity.get("name") or "").strip()
        if not name or str(entity.get("type") or "") not in ("人物", "主播"):
            continue
        translated = str(entity.get("zh") or "").strip()
        out.setdefault(name, Candidate(name, (translated,) if translated and translated != name else ()))
    return list(out.values())[:MAX_CANDIDATES]


@dataclass(frozen=True)
class Carried:
    """A track an earlier window opened, handed on when windows are judged in order (owner 2026-10-09):
    the number the reader gives it, where it was opened (and that unit's verdict), who the judge said it
    is, and its rows so far."""

    track: int
    window: str
    label: str
    name: str = ""
    rows: tuple[int, ...] = ()
    of: str = ""


@dataclass(frozen=True)
class Profiles:
    text: str
    #: track -> the rows its voice samples are cut from, in row order
    samples: dict[int, list[int]]


def profiles(rows: Sequence[Row], window: JudgeWindow, carried: Sequence[Carried] = ()) -> Profiles:
    """Each track's share of speech over the file and its most certain lines, outside the shown rows first.

    Ranked by the main track's coverage (the experiment ranked by VBx's soft margin, which the stable
    does not keep), then the longer line, then the earlier. A carried track comes after the acoustic
    ones: who it is and its lines so far, the latest first."""
    shown = window.shown(rows)
    share: dict[int, float] = {}
    for row in rows:
        if row.tracks:
            share[row.tracks[0]] = share.get(row.tracks[0], 0.0) + row.seconds
    total = sum(share.values()) or 1.0

    def rank(index: int) -> tuple:
        row = rows[index]
        coverage = row.speaker.extras.get("coverage") or {}
        return (index in shown, -float(coverage.get(str(row.tracks[0]), 0.0)), -row.seconds, index)

    lines, samples = [], {}
    # a voice sample is one person: never a row another speaker talks over, or one that talks over another
    crossed = line_groups.overlapping(rows)
    for track in sorted(share, key=lambda key: (-share[key], key)):
        own = [index for index, row in enumerate(rows) if row.tracks == [track] and not row.low]
        examples = sorted(sorted((index for index in own if len(re.sub(r"[\W_]", "", rows[index].text)) >= MIN_EXAMPLE_CHARS),
                                 key=rank)[:EXAMPLE_LINES])
        samples[track] = sorted(sorted((index for index in own if SAMPLE_SECONDS[0] <= rows[index].seconds <= SAMPLE_SECONDS[1]
                                        and not crossed[index]),
                                       key=rank)[:SAMPLES_PER_TRACK])
        lines.append(PHRASES["profile_line"].format(
            track=track, share=share[track] / total * 100,
            examples=" / ".join(f"「{rows[index].text}」" for index in examples) or PHRASES["no_examples"]))
    for track in carried:
        latest = sorted(track.rows, reverse=True)
        examples = sorted([index for index in latest
                           if len(re.sub(r"[\W_]", "", rows[index].text)) >= MIN_EXAMPLE_CHARS][:EXAMPLE_LINES])
        samples[track.track] = sorted([index for index in latest
                                       if SAMPLE_SECONDS[0] <= rows[index].seconds <= SAMPLE_SECONDS[1]
                                       and not crossed[index]][:SAMPLES_PER_TRACK])
        who = (PHRASES["carried_named"].format(name=track.name)
               if track.name and track.name != PHRASES["unknown_person"] else PHRASES["carried_unknown"])
        lines.append(PHRASES["carried_line"].format(
            track=track.track, who=who,
            examples=" / ".join(f"「{rows[index].text}」" for index in examples) or PHRASES["no_examples"]))
    return Profiles("\n".join(lines), samples)


@dataclass(frozen=True)
class AudioLayout:
    """The audio file the judge hears: each track's samples, then the shown rows' recording, with
    `SAMPLE_GAP` of silence between pieces -- round 9's layout, which its audio note describes."""

    pieces: list[tuple[float, float]]
    table: str
    #: where the recording starts in the file, and the source time it starts at
    offset: float
    origin: float

    @property
    def seconds(self) -> float:
        start, end = self.pieces[-1]
        return self.offset + end - start

    def speech(self, rows: Sequence[Row]) -> list[tuple[float, float]]:
        """Where rows speak in the file (seconds): every sample, and the rows within the recording."""
        spans, cursor = [], 0.0
        for start, end in self.pieces[:-1]:
            spans.append((cursor, cursor + end - start))
            cursor += end - start + SAMPLE_GAP
        start, end = self.pieces[-1]
        spans += [(self.offset + max(row.start, start) - start, self.offset + min(row.end, end) - start)
                  for row in rows if row.end > start and row.start < end]
        return spans


def audio_layout(rows: Sequence[Row], window: JudgeWindow, samples: Mapping[int, Sequence[int]]) -> AudioLayout:
    pieces, table, cursor = [], [], 0.0
    for track in sorted(samples):
        for index in samples[track]:
            row = rows[index]
            pieces.append((row.start, row.end))
            table.append(PHRASES["sample_row"].format(track=track, start=cursor, end=cursor + row.seconds))
            cursor += row.seconds + SAMPLE_GAP
    shown = window.shown(rows)
    origin = max(0.0, rows[shown[0]].start - EDGE_PAD)
    end = line_groups.run_end([rows[index] for index in shown]) + EDGE_PAD
    pieces.append((origin, end))
    table.append(PHRASES["window_row"].format(start=cursor, end=cursor + end - origin))
    silent = [f"s{track}" for track in sorted(samples) if not samples[track]]
    text = "\n\n" + load_prompt_template(AUDIO_LAYOUT_TEMPLATE, strict=True, rows="\n".join(table)).rstrip("\n")
    if silent:
        text += "\n\n" + load_prompt_template(AUDIO_SILENT_TEMPLATE, strict=True, tracks="、".join(silent)).rstrip("\n")
    return AudioLayout(pieces, text, cursor, origin)


def shape_section(source: str = "acoustic", *, correct_only: bool = False) -> str:
    """Pass 1's shape section for tracks from `source` (`SOURCE_WORDS`). Its criteria are the ones the single
    pass prompt shares (`shape_criteria`), worded for a step of their own; what pass 2 weighs follows the
    task's mode (`correct_only`: a Chinese source, no translation)."""
    criteria = shape_criteria("pass1", same_person=_phrase(SHAPE_SAME_PERSON_TEMPLATES[source]),
                              correct_only=correct_only)
    return load_prompt_template(SHAPE_TEMPLATE, strict=True, join_criteria=criteria["join"],
                                may_criteria=criteria["may_join"], drop_criteria=criteria["drop"],
                                **task_mode_texts(correct_only))


def _timing(rows: Sequence[Row], index: int) -> str:
    """A shaped row's `时长|gap` cells: its seconds, and the seconds since the row before it ended (none for
    the file's first row), 0 where the two overlap -- the correction window's gap (`chunking.render_segments_as_csv`)."""
    row = rows[index]
    gap = f"{round(max(0.0, row.start - rows[index - 1].end), 1):.1f}" if index > 0 else ""
    return f"{row.seconds:.1f}|{gap}"


def _summary(summary: str) -> str:
    return "\n" + load_prompt_template(SUMMARY_TEMPLATE, strict=True, global_summary=summary.strip()).rstrip("\n") \
        if summary.strip() else ""


def judge_prompt(
    rows: Sequence[Row],
    window: JudgeWindow,
    *,
    title: str,
    summary: str,
    cast: Sequence[Candidate],
    marks: bool,
    profile: Profiles,
    audio: AudioLayout | None = None,
    shape: bool = False,
    correct_only: bool = False,
) -> tuple[str, str, list[str]]:
    """`(system, user, ids)` for one window; `ids` are the judged rows' `q<row id>`. `profile` is
    `profiles(rows, window)`, and `audio` the layout built from its samples. `shape` adds pass 1's shape
    section and the rows' timing; with audio as well, the shape section adds what only listening finds:
    `split` (where a line changes speaker), `fill` (the rest of a line's sentence the ASR missed) and `insert`
    (a whole line it missed)."""
    row_format = PHRASES["shaped_row_format" if shape else "row_format"]
    split = shape and audio is not None
    source = label_source(rows)
    words = {key: _phrase(name) for key, name in SOURCE_WORDS[source].items()}
    # The marks note explains the brackets only a window that shows one needs.
    marked = marks and any(cell(rows[index], marks=True) != cell(rows[index], marks=False) for index in window.shown(rows))
    system = load_prompt_template(
        SYSTEM_TEMPLATE, strict=True, summary=_summary(summary),
        marks=load_prompt_template(MARKS_NOTE_TEMPLATE) if marked else "",
        audio=load_prompt_template(AUDIO_NOTE_TEMPLATE, strict=True, audio_mix=words["audio_mix"],
                                   insert_hint=_phrase(INSERT_HINT_TEMPLATE) if split else "",
                                   conflict_answer=_phrase(CONFLICT_TEMPLATES["shaped" if shape else "alone"]))
        if audio is not None else "",
        row_format=row_format,
        fields=_phrase(FIELDS_TEMPLATES["heard" if split else "shaped" if shape else "alone"]),
        shape=(shape_section(source, correct_only=correct_only)
               + (load_prompt_template(SHAPE_AUDIO_TEMPLATE, strict=True) if split else ""))
        if shape else "",
        split_hint=_phrase(SPLIT_HINT_TEMPLATE) if split else "",
        shape_task=_phrase(SHAPE_TASK_TEMPLATE) if shape else "",
        **{key: value for key, value in words.items() if key != "audio_mix"},
    )
    when = (lambda row: audio.offset + row.start - audio.origin) if audio is not None else (lambda row: row.start)
    lines, ids = [], []
    for index in window.shown(rows):
        row = rows[index]
        judged = window.low <= index < window.high
        if judged:
            ids.append(f"q{row.id}")
        timing = f"{_timing(rows, index)}|" if shape else ""
        lines.append((f"▶ q{row.id} " if judged else "") + f"{when(row):.1f}|{timing}{cell(row, marks=marks)}|{row.text}")
    user = load_prompt_template(
        USER_TEMPLATE, strict=True, title=title,
        cast="、".join(candidate.display for candidate in cast) or PHRASES["no_cast"],
        profiles=profile.text + (audio.table if audio is not None else ""), rows="\n".join(lines), count=len(ids),
        row_format=row_format,
    )
    return system, user, ids

