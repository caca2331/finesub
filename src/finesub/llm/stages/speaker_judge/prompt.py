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
import unicodedata
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Sequence

from ....speech.diarization.judgments import segment_ids
from ....speech.diarization.labels import SegmentSpeaker, filled_track, speaker_of
from ....speech.joint.model import UNKNOWN_SPEAKER
from ....subtitles import line_groups
from ...prompt_compose import load_prompt_template, task_mode_texts
from ...shape_criteria import shape_criteria
from .windows import JudgeWindow

SYSTEM_TEMPLATE = "speaker_judge_system_v4.md"
USER_TEMPLATE = "speaker_judge_user_v3.md"
#: A cloud judge shown the person table, where every track is a person and is identified only on evidence, instead
#: of candidates and example lines (`people_io`)
PEOPLE_SYSTEM_TEMPLATE = "speaker_judge_system_tracks_v6.md"
#: The same when a candidate carries the preparation task's description: its 「候选」 and 「人物表」 bullets say what
#: the description on a candidate's line is (`Candidate.note`, `people_io.PHRASES["prepared"]`)
PREPARED_SYSTEM_TEMPLATE = "speaker_judge_system_tracks_v7.md"
PEOPLE_USER_TEMPLATE = "speaker_judge_user_people_v1.md"
SHAPE_TEMPLATE = "speaker_judge_shape_v1.md"
SHAPE_AUDIO_TEMPLATE = "speaker_judge_shape_audio_v1.md"
#: The opening task sentence's addition when the judge also shapes the subtitles (pass 1).
SHAPE_TASK_TEMPLATE = "speaker_judge_shape_task_v1.md"
AUDIO_NOTE_TEMPLATE = "speaker_judge_audio_note_v1.md"
MARKS_NOTE_TEMPLATE = "speaker_judge_marks_note_v1.md"
SUMMARY_TEMPLATE = "speaker_judge_summary_v1.md"
#: The person table's voice section, a cut track compared as its latest piece
VOICES_PEOPLE_TEMPLATE = "speaker_judge_voices_people_v2.md"
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
    "relayed_note": "，说这个人是：{note}",
    "moved_out": "；前面的段从这条轨挪走 {moved} 行（占它前面 {rows} 行的 {share:.0f}%）：{where}",
    "moved_to": "{count} 行到 s{track}{named}",
    "moved_to_off": "{count} 行到画像里没有的一条轨{named}",
    "people_profile_line": "- s{track}：占语音 {share:.0f}%，{first:.0f}–{last:.0f} 秒出现{origin}{spread}{relayed}{changed}。",
    # a track on stage whose rows here, from some row on, sound unlike its rows before (`run.voice_split`)
    "voice_changed": "；s{track} 从 q{row} 附近起，声音和它前面的行不像",
    # ... and like one of its earlier pieces' (most alike first)
    "voice_like": "，和旧截 {pieces} 的声音像",
    # a cut track's earlier piece in the voice section
    "old_piece_name": "{handle}（到第{segment}段 q{row} 之前的一截）",
    # a cut track in the voice section is its latest piece
    "piece_name": "s{track}（第{segment}段 q{row} 起的一截）",
    # how many of the track's rows come before this unit, in it and after it (from its latest piece's start)
    "spread": "；转写里这条轨{since}前面各段 {before} 行、这一段的 ▶ 行里 {here} 行、后面各段 {after} 行",
    "spread_since": "从第{segment}段 q{row} 起",
    "relayed_status": "；{status}",
    "moved_person": "（那条轨前面{name}）",
    "moved_rest": "其余 {count} 行到别的轨",
    # the person table numbers the judge's own units 第N段, so the transcription's chunks are 块
    "origin_seam": "；转写分块的第 {part} 块里才有的标签，没和前面各块的标签连上",
    "origin_rescue": "；补转回来的一小段里给的标签，没和主转写的标签连上",
    "nobody_line": "- s{track}：**不是一个人**：转写模型没给说话人的话（多是补转回来的小片段）都挂在这条轨上，占语音 {share:.0f}%。"
                   "这条轨上的行按内容判给别的轨或 `@新轨`，判不了答 `keep`；别的行不要答成它，`<tracks>` 里也不要把别的轨写成它。"
                   "例句：{examples}",
    "carried_linked": "前面的窗口认为是 s{track} 那个人{note}，未确认",
    "voice_pair": "- {a} 与 {b}：{tier}",
    "voice_pair_off": "- {a} 与 {b}（不在这一段，{status}）：{tier}",
    "voice_tiers": {"high": "高", "mid": "中"},
    "no_voice_pairs": "（没有达到「中」的两条轨）",
    "unprinted": "；说话太少、没比声音的轨：{tracks}",
    "carried_line": "- s{track}：前面的窗口里新建的轨（{who}）。例句：{examples}",
    "carried_named": "前面的窗口猜是{name}{note}，未确认",
    "carried_unknown": "前面的窗口认不出是谁{note}",
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
                               "people": "speaker_judge_shape_same_person_tracks_v2.md"}

#: What the judge is told about where the tracks come from: the acoustic layer (a local task), or a cloud task's
#: transcription (its lines' `line_speaker`), which splits a person more often than it merges people, has no
#: `??` or `s5+s4` cells and no samples mixed from two people. A cloud task's judge is always shown the person
#: table (`run.unit_prompt`), so its words are keyed `people`. Template files; None leaves the slot empty.
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
    # a cloud task's judge, with the person table
    "people": {
        "tracks_intro": "speaker_judge_source_cloud_intro_v1.md",
        "tracks_errors": "speaker_judge_source_cloud_errors_people_v2.md",
        "tracks_cells": None,
        "tracks_trust": "speaker_judge_source_cloud_trust_v1.md",
        "keep_cells": None,
        "audio_mix": None,
    },
}


def label_source(rows: Sequence["Row"]) -> str:
    """`cloud` when the rows carry a cloud transcription's speakers, else `acoustic`: a cloud task's judge is shown
    the person table (`run.unit_prompt`), and the single pass takes acoustic rows only (`window_speakers.tracked_rows`)."""
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
    """A person the judge may name: the name to copy, and what else they go by (shown in brackets). `note`: the
    preparation task's description of them (`speakers[].note`), shown on their line of the person table."""

    name: str
    others: tuple[str, ...] = ()
    note: str = ""

    @property
    def display(self) -> str:
        return f"{self.name}（{'、'.join(self.others)}）" if self.others else self.name


def candidates(entities: Sequence[Mapping[str, Any]], roster: Sequence[tuple[str, tuple[str, ...], str]],
               speakers: Sequence[Mapping[str, Any]] = ()) -> list[Candidate]:
    """Who the preparation task saw speaking first, then the knowledge base's streamers, then the people
    it found mentioned; a name once, `MAX_CANDIDATES` at most. The judge may still name anyone else. A speaker brings
    the preparation task's description (`note`); the knowledge base's and `entities`' people none (an entity's
    `note` is the evidence for its name, not a description)."""
    out: dict[str, Candidate] = {}
    for speaker in speakers:
        name = str(speaker.get("name") or "").strip()
        if name:
            out.setdefault(name, Candidate(name, note=str(speaker.get("note") or "").strip()))
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


def joint_label(row: Row) -> str | None:
    """The cloud transcription's own label a row leads (`write_joint_rows` keeps it); None on a local task."""
    label = row.speaker.extras.get("joint_speaker")
    return str(label) if label is not None else None


def no_one(rows: Sequence[Row], track: int) -> bool:
    """Whether a cloud track is the `unknown` one (lines no speaker was found for): no one, it gets no person."""
    return joint_label(next(row for row in rows if row.tracks[:1] == [track])) == UNKNOWN_SPEAKER


#: A cloud label's origin, from the name `speech.joint` gave it: `w<k>:` a label of the k-th window of a split
#: transcription (`stitch`, which keeps each window's labels apart), `c<k>:` a label of the k-th re-asked piece that was not merged
#: (`relink.renumber`).
_SEAM_LABEL = re.compile(r"^w(\d+):")
_RESCUE_LABEL = re.compile(r"(?:^|:)c\d+:")


def origin(label: str | None) -> str:
    """What a cloud track's profile says of where its label came from: nothing for one heard from the first window."""
    if label is None:
        return ""
    if _RESCUE_LABEL.search(label):
        return PHRASES["origin_rescue"]
    seam = _SEAM_LABEL.match(label)
    return PHRASES["origin_seam"].format(part=int(seam.group(1)) + 1) if seam and int(seam.group(1)) > 0 else ""


def voice_pairs(rows: Sequence[Row], tracks: Sequence[int], voices: Mapping[str, Any] | None,
                others: Sequence[int] = (), prints: Mapping[int, Any] | None = None) -> list[tuple[int, int, str]]:
    """The pairs of tracks in a tier (`speech.joint.voices`), `(a, b, tier)`: among `tracks`, then each of `tracks`
    with each of `others` (tracks off stage); empty without a comparison. `prints` (track -> its own
    voiceprint, None when too short) stands for a track's label print (`run.piece_voices`: a cut track's latest
    piece); two labels that talk at once are never in a tier, whatever stands for them (the overlap stays the whole
    labels': a hint, where the merge gate reads each track's own talk, `judgments.talk_spans`). A cut track's
    earlier piece in `prints` under its person key (str, `run.piece_voices`) is paired with each of `tracks` (its
    own track's latest piece included, with no overlap gate: one label), after the tracks' pairs and only `high`;
    never with another earlier piece."""
    if not voices or voices.get("unavailable"):
        return []
    from ....speech.diarization import track_people as T
    from ....speech.joint import voices as V

    labels = _labels(rows)
    prints = prints or {}
    found, pair_of = [], V.pairs(voices)
    pairs = [*((a, b) for index, a in enumerate(tracks) for b in tracks[index + 1:]),
             *((a, b) for a in tracks for b in others if b not in tracks)]
    for a, b in pairs:
        if a not in labels or b not in labels:
            continue
        pair = pair_of(labels[a], labels[b])
        if a in prints or b in prints:
            vector = {track: prints[track] if track in prints else V.label_print(voices, labels[track]) for track in (a, b)}
            pair = {"cos": V.cosine(vector[a], vector[b]), "overlap": pair["overlap"] if pair else 0.0}
        if tier := V.tier(pair, voices["tiers"]):
            found.append((a, b, tier))
    for key, vector in prints.items():
        if not isinstance(key, str) or vector is None or (own := T.piece_track(key)) not in labels:
            continue
        for b in tracks:
            if b not in labels:
                continue
            pair = pair_of(labels[own], labels[b]) if labels[b] != labels[own] else None
            other = prints[b] if b in prints else V.label_print(voices, labels[b])
            pair = {"cos": V.cosine(vector, other), "overlap": pair["overlap"] if pair else 0.0}
            if V.tier(pair, voices["tiers"]) == "high":
                found.append((key, b, "high"))
    return found


def _labels(rows: Sequence[Row]) -> dict[int, str]:
    labels = {}
    for row in rows:
        if row.tracks and (label := joint_label(row)) is not None:
            labels.setdefault(row.tracks[0], label)
    return {track: label for track, label in labels.items() if label != UNKNOWN_SPEAKER}


def voice_lines(rows: Sequence[Row], tracks: Sequence[int], voices: Mapping[str, Any] | None,
                off_stage: Mapping[int, str] | None = None, prints: Mapping[int, Any] | None = None,
                names: Mapping[int, str] | None = None) -> str:
    """The similarity section under a cloud task's profiles: the pairs of tracks in a tier, `high` before `mid`, and
    the tracks too short to compare (`speech.joint.voices`); empty without a comparison. `off_stage` (listed tracks
    off stage -> what the units before identified them as) adds their pairs with the tracks on stage. `prints` and
    `names` (`run.piece_voices`): a cut track compared as its latest piece, and named so, its earlier pieces among
    the pairs (`voice_pairs`)."""
    if not voices or voices.get("unavailable"):
        return ""
    labels = _labels(rows)
    off_stage, prints, names = off_stage or {}, prints or {}, names or {}

    def name(track: int | str) -> str:
        return names.get(track, f"s{track}")

    # `high` first, stable within a tier
    found = sorted(voice_pairs(rows, tracks, voices, sorted(off_stage), prints), key=lambda pair: pair[2] != "high")
    pairs = [PHRASES["voice_pair_off" if b in off_stage else "voice_pair"].format(
                 a=name(a), b=name(b), tier=PHRASES["voice_tiers"][tier], status=off_stage.get(b, ""))
             for a, b, tier in found]
    printed = {label for label, value in (voices.get("labels") or {}).items() if value.get("printed")}
    unprinted = [name(track) for track in tracks
                 if (prints[track] is None if track in prints else track in labels and labels[track] not in printed)]
    return "\n\n" + load_prompt_template(
        VOICES_PEOPLE_TEMPLATE, strict=True,
        pairs="\n".join(pairs) or PHRASES["no_voice_pairs"],
        silent=PHRASES["unprinted"].format(tracks="、".join(unprinted)) if unprinted else "").rstrip("\n")


def voice_hint(track: int, at: str, like: Sequence[str] = ()) -> str:
    """What a hinted track's profile line says (`run.VoiceHint`, the row's id): its voice changes from `at` on, like
    its earlier pieces `like` (their handles, most alike first)."""
    return PHRASES["voice_changed"].format(track=track, row=at) \
        + (PHRASES["voice_like"].format(pieces="、".join(like)) if like else "")


_NOTE = re.compile(r"^(.*?)\s*[（(]([^（）()]*)[）)]\s*$")


def split_note(value: str) -> tuple[str, str]:
    """A `<tracks>` value as (name, note): the judge may say in brackets after the name who the person is
    (`s3 = 不明（店员，向客人介绍菜单）`). The name is NFKC, the note as written."""
    text = str(value).strip()
    found = _NOTE.match(text)
    name, note = (found.group(1), found.group(2).strip()) if found else (text, "")
    return unicodedata.normalize("NFKC", name).strip(), note


def _noted(note: str) -> str:
    return PHRASES["relayed_note"].format(note=note) if note else ""


def _spread(counts: tuple[int, int, int, tuple[int, str] | None] | None) -> str:
    if counts is None:
        return ""
    before, here, after, since = counts
    return PHRASES["spread"].format(
        since=PHRASES["spread_since"].format(segment=since[0], row=since[1]) if since else "",
        before=before, here=here, after=after)


#: A track's moves are said from this many rows moved, naming at most MOVED_PLACES destinations
MOVED_LEAST = 2
MOVED_PLACES = 3


def moved_phrase(moved: tuple[int, int, Sequence[tuple[int, int]]] | None,
                 names: Mapping[int, tuple[str, str]], stage: Collection[int] | None = None) -> str:
    """What a cloud track's line says of the rows the units before moved off it (`run.moved_out`: moved, rows,
    (track, count) most first); a destination the units before identified says as what (`people_io.profile_names`:
    `认作 p3 Coco`, `连到 s5`). With `stage` a destination off stage is said without its number."""
    if not moved or moved[0] < MOVED_LEAST:
        return ""
    count, total, where = moved

    def named(track: int) -> str:
        name = names.get(track, ("", ""))[0]
        return PHRASES["moved_person"].format(name=name) if name else ""

    said = [PHRASES["moved_to_off"].format(count=k, named=named(track)) if stage is not None and track not in stage
            else PHRASES["moved_to"].format(count=k, track=track, named=named(track))
            for track, k in where[:MOVED_PLACES]]
    rest = sum(k for _, k in where[MOVED_PLACES:])
    if rest:
        said.append(PHRASES["moved_rest"].format(count=rest))
    return PHRASES["moved_out"].format(moved=count, rows=total, share=count / total * 100, where="、".join(said))


def stage_tracks(rows: Sequence[Row], window: JudgeWindow, *, marks: bool) -> set[int]:
    """The tracks on stage in a window: every track a shown row's cell names, ▶ rows and context alike."""
    return {track for index in window.shown(rows) for track in cell_tracks(rows[index], marks=marks)}


def profiles(rows: Sequence[Row], window: JudgeWindow, carried: Sequence[Carried] = (),
             voices: Mapping[str, Any] | None = None, *, merging: bool = False,
             relayed: Mapping[int, tuple[str, str]] | None = None,
             moved: Mapping[int, tuple[int, int, Sequence[tuple[int, int]]]] | None = None,
             stage: Collection[int] | None = None, people: bool = False,
             off_stage: Mapping[int, str] | None = None,
             spread: Mapping[int, tuple[int, int, int, tuple[int, str] | None]] | None = None,
             prints: Mapping[int, Any] | None = None, piece_names: Mapping[int, str] | None = None,
             changed: Mapping[int, str] | None = None) -> Profiles:
    """Each track's share of speech over the file and its most certain lines, outside the shown rows first.

    Ranked by the main track's coverage (the experiment ranked by VBx's soft margin, which the stable
    does not keep), then the longer line, then the earlier. A carried track (a local task's) comes after the
    acoustic ones: who it is and its lines so far, the latest first; with `merging` (the judge's own windows, whose
    `<tracks>` may link tracks: not the single-pass window's) one named after another track says so. `people` (a
    cloud task's judge, `run.unit_prompt`) lists a cloud track without example lines (the person table describes
    people), says when it speaks and where its label came from (`origin`), says `unknown` is no one, and `voices`
    (`speech.joint.voices`) adds the similarity section. `stage` (`stage_tracks`) lists only those tracks;
    `relayed` (`people_io.profile_names`) is what the units before identified each track as, short and long (`认作 p3 Coco`, `前面认作 p3 Coco（第2段：…）`), said on those tracks' lines, and `moved`
    (`run.moved_out`) how many of their rows the units before moved where; `off_stage` (listed tracks off stage ->
    `people_io.brief`) adds their voice pairs. `spread` (track -> its rows before this unit, in it, after it, and the
    segment and row its latest piece starts at) is said on those tracks' lines. `prints`, `piece_names` and `changed`
    (`run.piece_voices`): a cut track's voice is its latest piece's, its earlier pieces are in `prints` and
    `piece_names` under their person key (`voice_pairs`), and a track on stage that sounds unlike its earlier rows
    says so, from which row (track -> `voice_hint`'s phrase)."""
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
    listed = [track for track in sorted(share, key=lambda key: (-share[key], key)) if stage is None or track in stage]
    for track in listed:
        own = [index for index, row in enumerate(rows) if row.tracks == [track] and not row.low]
        examples = sorted(sorted((index for index in own if len(re.sub(r"[\W_]", "", rows[index].text)) >= MIN_EXAMPLE_CHARS),
                                 key=rank)[:EXAMPLE_LINES])
        samples[track] = sorted(sorted((index for index in own if SAMPLE_SECONDS[0] <= rows[index].seconds <= SAMPLE_SECONDS[1]
                                        and not crossed[index]),
                                       key=rank)[:SAMPLES_PER_TRACK])
        quoted = " / ".join(f"「{rows[index].text}」" for index in examples) or PHRASES["no_examples"]
        if people and no_one(rows, track):
            lines.append(PHRASES["nobody_line"].format(track=track, share=share[track] / total * 100, examples=quoted))
        elif people:
            led = [row for row in rows if row.tracks[:1] == [track]]
            said = (relayed or {}).get(track, ("", ""))[1]
            lines.append(PHRASES["people_profile_line"].format(
                track=track, share=share[track] / total * 100, first=min(row.start for row in led),
                last=max(row.end for row in led), origin=origin(joint_label(led[0])),
                spread=_spread((spread or {}).get(track)),
                relayed=(PHRASES["relayed_status"].format(status=said) if said else "")
                + moved_phrase((moved or {}).get(track), relayed or {}, stage),
                changed=(changed or {}).get(track, "")))
        else:
            lines.append(PHRASES["profile_line"].format(track=track, share=share[track] / total * 100, examples=quoted))
    for track in carried:
        latest = sorted(track.rows, reverse=True)
        examples = sorted([index for index in latest
                           if len(re.sub(r"[\W_]", "", rows[index].text)) >= MIN_EXAMPLE_CHARS][:EXAMPLE_LINES])
        samples[track.track] = sorted([index for index in latest
                                       if SAMPLE_SECONDS[0] <= rows[index].seconds <= SAMPLE_SECONDS[1]
                                       and not crossed[index]][:SAMPLES_PER_TRACK])
        name, note = split_note(track.name)
        linked = re.fullmatch(r"s(\d+)", name) if merging else None
        who = (PHRASES["carried_linked"].format(track=linked.group(1), note=_noted(note)) if linked
               else PHRASES["carried_named"].format(name=name, note=_noted(note))
               if name and name != PHRASES["unknown_person"] else PHRASES["carried_unknown"].format(note=_noted(note)))
        lines.append(PHRASES["carried_line"].format(
            track=track.track, who=who,
            examples=" / ".join(f"「{rows[index].text}」" for index in examples) or PHRASES["no_examples"]))
    text = "\n".join(lines) + (voice_lines(rows, listed, voices, off_stage, prints, piece_names) if people else "")
    return Profiles(text, samples)


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
    people: str | None = None,
    prepared: bool = False,
) -> tuple[str, str, list[str]]:
    """`(system, user, ids)` for one window; `ids` are the judged rows' `q<row id>`. `profile` is
    `profiles(rows, window)`, and `audio` the layout built from its samples. `shape` adds pass 1's shape
    section and the rows' timing; with audio as well, the shape section adds what only listening finds:
    `split` (where a line changes speaker), `fill` (the rest of a line's sentence the ASR missed) and `insert`
    (a whole line it missed). `people` is the person table's section (`people_io.table_section`), given exactly for a
    cloud task's rows (`run.unit_prompt`): the judge is shown it instead of the candidates, reads the cloud wording of
    where the tracks come from, and writes the people's `<tracks>` and `<people>`; without it the rows are a local
    task's acoustic tracks. `prepared`
    picks its system prompt that says what the preparation task's descriptions in the table are."""
    row_format = PHRASES["shaped_row_format" if shape else "row_format"]
    split = shape and audio is not None
    source = "people" if people is not None else "acoustic"
    words = {key: _phrase(name) for key, name in SOURCE_WORDS[source].items()}
    # The marks note explains the brackets only a window that shows one needs.
    marked = marks and any(cell(rows[index], marks=True) != cell(rows[index], marks=False) for index in window.shown(rows))
    system = load_prompt_template(
        (PREPARED_SYSTEM_TEMPLATE if prepared else PEOPLE_SYSTEM_TEMPLATE) if people is not None else SYSTEM_TEMPLATE,
        strict=True, summary=_summary(summary),
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
        PEOPLE_USER_TEMPLATE if people is not None else USER_TEMPLATE, strict=True, title=title,
        cast="、".join(candidate.display for candidate in cast) or PHRASES["no_cast"], people=people or "",
        profiles=profile.text + (audio.table if audio is not None else ""), rows="\n".join(lines), count=len(ids),
        row_format=row_format,
    )
    return system, user, ids

