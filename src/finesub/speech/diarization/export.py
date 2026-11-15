"""Lossless exports: one SRT per speaker, or one ASS carrying all of them.

The single-track SRT is not touched here -- owner decision, 2026-09-05 (acoustic row splitting, owner
2026-10-14, is the one thing diarization changes in it, upstream of this). These two are *additional* outputs, produced on request, and they exist because one
track has to make a choice about overlapping speech while these do not.

The correctness criterion is the invariant, not the accuracy:

    every source line appears in at least one track

A misassigned line costs one deletion in a subtitle editor; a dropped line has to
be noticed first, then found, then retyped. So a line with two speakers is written
into both tracks, deliberately, and a line with no speaker still goes somewhere --
the `other` track catches everything the acoustic layer declined to place.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

from ...subtitles.model import SrtSegment, has_text, render_srt
from .labels import SegmentSpeaker, cell_name, filled_track

#: Where lines nobody claimed go. Named, not numbered, so it cannot collide with
#: a track id.
FALLBACK_TRACK = "other"

#: What a file name (Windows) or an ASS `Style:` / `Dialogue:` field cannot hold, as full-width look-alikes.
_FILE_RESERVED = str.maketrans({c: chr(ord(c) + 0xFEE0) for c in '\\/:"<>,'})
#: Room for the name inside one path component, beside `<name>-speakers-` and `.srt`.
_NAME_LIMIT = 64
#: Group names a track or the fallback owns, whether or not this export has that group.
_TRACK_KEY = re.compile(rf"speaker[0-9]+|{FALLBACK_TRACK}")


def _keys(speaker: SegmentSpeaker) -> list[tuple[str, object]]:
    """The groups one line goes to: the names a person wrote onto its labels, else its tracks."""
    names = list(dict.fromkeys(cell_name(label.name) for label in speaker.labels if label.name))
    if names:
        return [("name", name) for name in names]
    tracks = speaker.extras.get("tracks")
    if not tracks and speaker.track is not None:
        tracks = [speaker.track]
    if not tracks and filled_track(speaker) is not None:
        tracks = [filled_track(speaker)]
    return [("track", int(track)) for track in dict.fromkeys(tracks or ())]


def _shown(keys: Sequence[tuple[str, object]]) -> dict[tuple[str, object], str]:
    """Each group's name in the file names and the ASS styles.

    A track reads `speaker4`; a name reads as in the `spk` cell (`cell_name`), with what a file name or an
    ASS field cannot hold also turned full-width, and cut to `_NAME_LIMIT`. Track ids and `other` keep
    their names; a name that would land on a name already taken -- ignoring case, as Windows does -- or
    that reads like a track's or the fallback's (`speaker7`, `other`) gets `-2`, `-3`, so no export
    overwrites another and no person's file passes for a track's."""
    shown = {key: FALLBACK_TRACK if key[0] == "other" else f"speaker{key[1]}" for key in keys if key[0] != "name"}
    taken = {value.casefold() for value in shown.values()}
    for key in keys:
        if key[0] != "name":
            continue
        base = cell_name(str(key[1]).translate(_FILE_RESERVED)[:_NAME_LIMIT])
        candidate, suffix = base, 2
        while candidate.casefold() in taken or _TRACK_KEY.fullmatch(candidate.casefold()):
            candidate, suffix = f"{base}-{suffix}", suffix + 1
        taken.add(candidate.casefold())
        shown[key] = candidate
    return {key: shown[key] for key in keys}


def group_by_speaker(
    subtitles: Sequence[Mapping[str, object]],
    speakers: Sequence[SegmentSpeaker],
) -> dict[str, list[Mapping[str, object]]]:
    """Split lines into per-speaker groups, keeping every line somewhere.

    A line assigned to several tracks is repeated in each of them; a line with no
    assignment lands in `other`. Both are deliberate: see the module docstring.
    A line the confidence marks filled (`mark.track`, `marks.py`) goes to that track:
    the per-speaker files may carry some errors but should need almost no hand work
    (owner 2026-10-07). Of filled lines the reference has words on, 54-58% are right
    where `other` always needs a move; about a third of missed lines have no words
    in the reference at all (music, effects) and land in a person's track too.

    A name a person wrote onto a label stands for the line's tracks, as in the `spk` cell; the group
    names are safe as file names and ASS style names (`_shown`).
    """
    if len(subtitles) != len(speakers):
        raise ValueError(f"{len(subtitles)} subtitles but {len(speakers)} speaker records")

    keyed: dict[tuple[str, object], list[Mapping[str, object]]] = {}
    for subtitle, speaker in zip(subtitles, speakers):
        for key in _keys(speaker) or [("other", None)]:
            keyed.setdefault(key, []).append(subtitle)
    shown = _shown(list(keyed))
    return {shown[key]: lines for key, lines in keyed.items()}


def render_speaker_srts(
    subtitles: Sequence[Mapping[str, object]],
    speakers: Sequence[SegmentSpeaker],
    *,
    empty_cues: bool,
) -> dict[str, str]:
    """One SRT per speaker, as `{name: srt text}`. A line with no text is a ``""`` cue with `empty_cues`
    (`--empty-cues`), left out otherwise -- as in the single-track SRT; the invariant counts it either way."""
    rendered = {}
    for name, lines in group_by_speaker(subtitles, speakers).items():
        shown = [line for line in lines if empty_cues or has_text(line.get("text"))]
        segments = [
            SrtSegment(
                index=position + 1,
                start=float(line["start"]),
                end=float(line["end"]),
                text=str(line.get("text") or ""),
            )
            for position, line in enumerate(shown)
        ]
        rendered[name] = render_srt(segments)
    return rendered


def check_completeness(
    subtitles: Sequence[Mapping[str, object]],
    groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[int]:
    """Indices of source lines that reached no track. Must always be empty.

    This is the export's actual acceptance test -- accuracy is a metric, this is
    a bug. Returned rather than asserted so a caller can report *which* lines.
    """
    seen = {id(line) for lines in groups.values() for line in lines}
    return [position for position, line in enumerate(subtitles) if id(line) not in seen]


def _ass_time(seconds: float) -> str:
    """ASS wants `H:MM:SS.cc`, centiseconds and a single-digit hour."""
    if seconds < 0:
        seconds = 0.0
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{int(secs):02d}.{int(round(secs % 1 * 100)):02d}"


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""

#: The four hand-picked track colours, legible in both light and dark players. Not a design
#: system -- just four hues that do not wash out.
_BASE_COLOURS = ("&H00FFFFFF", "&H0080FFFF", "&H00FF8080", "&H0080FF80")
#: Reserved for the fallback track, by NAME rather than by position: a file with more tracks than
#: base colours would otherwise hand somebody else the grey.
_FALLBACK_COLOUR = "&H00C0C0C0"


def _style_colours(count: int) -> list[str]:
    """`count` distinct `&H00BBGGRR` colours, starting with the four hand-picked ones.

    ⚠ The palette used to be five entries indexed modulo its length, which was correct while the
    only backend emitted four tracks and wrong the moment one emitted more: the sixth track
    repeated the first, and two speakers sharing a colour undoes the entire point of this file.
    A clustering backend produced 5 to 23 tracks on our own material, so the palette has to be a
    function of the count rather than a fixed tuple.

    Beyond the four, hues are walked at an angle that does not revisit a previous one for a long
    time, at two brightness levels, skipping anything too close to the reserved grey.
    """
    import colorsys

    colours = list(_BASE_COLOURS[:count])
    index = 0
    while len(colours) < count:
        hue = (0.62 + index * 0.381966) % 1.0  # golden-ratio step: no two neighbours repeat soon
        value = 1.0 if index % 2 == 0 else 0.72
        red, green, blue = (round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.85, value))
        candidate = f"&H00{blue:02X}{green:02X}{red:02X}"
        if candidate != _FALLBACK_COLOUR and candidate not in colours:
            colours.append(candidate)
        index += 1
    return colours


def render_ass(
    subtitles: Sequence[Mapping[str, object]],
    speakers: Sequence[SegmentSpeaker],
) -> str:
    """One ASS file with every line, styled by speaker.

    Overlapping speech needs no special handling here: two events may share a
    time range, which is exactly what a single SRT track cannot express.
    """
    groups = group_by_speaker(subtitles, speakers)
    names = sorted(groups, key=lambda name: (name == FALLBACK_TRACK, name))

    lines = [_ASS_HEADER]
    palette = _style_colours(sum(1 for name in names if name != FALLBACK_TRACK))
    for position, name in enumerate(names):
        colour = _FALLBACK_COLOUR if name == FALLBACK_TRACK else palette[position]
        lines.append(
            f"Style: {name},Arial,72,{colour},&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,3,1,2,60,60,60,1"
        )
    lines.append("\n[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    events = []
    for name in names:
        for line in groups[name]:
            events.append(
                (
                    float(line["start"]),
                    f"Dialogue: 0,{_ass_time(float(line['start']))},{_ass_time(float(line['end']))},"
                    f"{name},,0,0,0,,{str(line.get('text', '')).replace(chr(10), ' ')}",
                )
            )
    events.sort(key=lambda item: item[0])
    lines.extend(event for _, event in events)
    return "\n".join(lines) + "\n"
