"""An ASS file to watch a diarized, translated clip with: one colour per speaker track.

Owner preview 2026-09-12. The pipeline's own `--speaker-tracks ass` renders the stable (source-language)
text; this renders what the LLM stage delivered -- each `-annotated.csv` row with its translation, the
corrected source text under it in small type, and its `spk` cell as the colour and the tag. Times come
from the stable segments a row covers (the CSV has none). Lines of several speakers get a grey style and
a `[s0+s1]` tag; a flagged lead (`?`, `~`, `*`) puts `?` on the tag. A cell the speaker judge changed
(`s3<s2?`) shows its left side, the speakers after the verdict. With `--stable-only`, renders the stable
text itself.

  python preview_ass.py OUT_DIR [--stable-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parents[1] / "examples"))

import spk_cell  # noqa: E402
from finesub.llm.output_protocol import split_annotated_row  # noqa: E402

#: ASS colours are &HBBGGRR. Picked to stay apart on a bright variety-show frame.
PALETTE = ["&H0000D7FF", "&H00FFB25C", "&H0078E36A", "&H00B77BFF", "&H006ADCF5", "&H00F2A6E8"]
MIXED = "&H00C8C8C8"


def ass_time(seconds):
    centis = int(round(max(0.0, seconds) * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, centis = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def clean(text):
    return (text or "").replace("{", "（").replace("}", "）").replace("\n", " ").strip()


def ids_in(position):
    out = []
    for part in position.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            if a.isdigit() and b.isdigit():
                out.extend(str(i) for i in range(int(a), int(b) + 1))
        elif part:
            out.append(part)
    return out


def tracks_in(cell):
    """(track numbers after the judge, whether the lead is flagged); a name has no track and is skipped."""
    final = spk_cell.parse(cell or "")["final"]
    return [int(t[1:]) for t in final["tracks"] if spk_cell.TRACK.fullmatch(t)], final["flag"] is not None


def labels(out_dir, stem):
    """Track names for the preview: none since the role layer went (2026-10-09); tracks show as ids."""
    return {}


def render(out_dir, stable_only=False):
    out_dir = Path(out_dir)
    stem = out_dir.name
    segments = json.loads((out_dir / f"{stem}-stable.json").read_text(encoding="utf-8"))["segments"]
    by_id = {str(s.get("id") or i): s for i, s in enumerate(segments, start=1)}
    names = labels(out_dir, stem)
    events = []
    csv = out_dir / f"{stem}-annotated.csv"
    if not stable_only and csv.exists():
        for line in csv.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            fields = split_annotated_row(line)
            if fields is None:
                continue
            covered = [by_id[i] for i in ids_in(fields.position) if i in by_id]
            if not covered:
                continue
            tracks, flagged = tracks_in(fields.spk)
            events.append((min(s["start"] for s in covered), max(s["end"] for s in covered), tracks, flagged,
                           clean(fields.translation) or clean(fields.corrected), clean(fields.corrected)))
    else:
        for s in segments:
            speaker = s.get("speaker") or {}
            tracks = speaker.get("tracks") or ([speaker["track"]] if speaker.get("track") is not None else [])
            events.append((s["start"], s["end"], tracks, bool(speaker.get("review")), clean(s.get("text")), ""))
    seen = sorted({t for _, _, tracks, _, _, _ in events for t in tracks})
    styles = [f"Style: S{t},Microsoft YaHei,58,{PALETTE[i % len(PALETTE)]},&H000000FF,&H00101010,&H80000000,"
              f"-1,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1" for i, t in enumerate(seen)]
    styles.append(f"Style: Mixed,Microsoft YaHei,58,{MIXED},&H000000FF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,40,40,50,1")
    lines = []
    for start, end, tracks, flagged, main, source in sorted(events):
        # the track id always shows: a name is the role layer's guess and can be wrong
        tag = "+".join(f"s{t}·{names[t]}" if t in names else f"s{t}" for t in tracks) or "?"
        if flagged:
            tag += "?"
        style = f"S{tracks[0]}" if len(tracks) == 1 else "Mixed"
        text = f"[{tag}] {main}" + (f"\\N{{\\fs38}}{source}" if source and source != main else "")
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},{style},{tag},0,0,0,,{text}")
    legend = "  ".join(f"s{t}={names.get(t, '?')}" for t in seen)
    header = "\n".join([
        "[Script Info]", f"Title: {stem} speaker preview", "ScriptType: v4.00+", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "PlayResX: 1920", "PlayResY: 1080", f"; tracks: {legend}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding",
        *styles, "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])
    target = out_dir / f"{stem}-preview{'-stable' if stable_only or not csv.exists() else ''}.ass"
    target.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8-sig")
    multi = sum(1 for e in events if len(e[2]) > 1)
    print(f"{target}  events {len(events)}  multi-speaker {multi}  tracks {legend}")
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--stable-only", action="store_true")
    arguments = parser.parse_args()
    render(arguments.out_dir, arguments.stable_only)


if __name__ == "__main__":
    main()
