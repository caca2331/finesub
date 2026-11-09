"""A local task's lines through every pass the overlap rules touched: a diarized
local sample where Whisper's defect overlaps lines -- a hallucinated stock phrase wraps three real lines, a short
line sits inside another -- and none of them carries a group key. Pinned: the raw SRT, the correction windows'
CSV input (with gap) and read-only lines, the window cuts, ends and clip ranges, a window split in half, the
merged rows' times (the long line merged with one it wraps, the nested line with its host), the final SRT, and
three passes' grouping and grouped windows. The expected values were produced by the code before the rules
(24cc1c1d): without keys there is one group, so nothing may move. The CSV inputs' gap cells were re-pinned
when gap became the pause before a line (csv-v111): the overlapping lines read 0, the first line empty; no
other cell moved."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from finesub.llm import chunking
from finesub.llm.prompt_constants import Thresholds
from finesub.llm.grouping import group_lines, grouped_windows
from finesub.llm.output_protocol import (
    correction_output_header,
    render_translated_segments_as_srt,
    validate_correction_window_output,
)
from finesub.llm.prompt_variants import resolve_variant
from finesub.llm.routing.config import CapabilityTier
from finesub.media.clips import compute_clip_range
from finesub.speech.diarization.judgments import ACT_JOIN, ACT_MAY
from finesub.speech.diarization.labels import different_speakers, speaker_of
from finesub.subtitles import rendering
from finesub.subtitles.postprocess import TIMELINE_POSTPROCESS_PROFILES, postprocess_srt_file, postprocess_srt_text

EXPECTED = Path(__file__).parent / "data" / "overlap_local_golden.json"
AUDIO_END = 40.0
# (start, end, text, track): the stock phrase (3) wraps 4-6; 6 is inside 5, the same voice
LINES = [
    (0.0, 2.0, "おはようございます。", 0),
    (2.5, 4.0, "今日はよろしくお願いします。", 1),
    (4.5, 25.0, "ご視聴ありがとうございました。", 2),
    (5.0, 7.0, "まずは自己紹介から。", 0),
    (7.5, 9.5, "はい、私から行きます。", 1),
    (8.0, 8.8, "そうそう", 1),
    (10.5, 13.0, "趣味はゲームで", 1),
    (13.2, 14.5, "最近は何を", 1),
    (14.6, 16.0, "遊んでいますか。", 0),
    (26.0, 28.0, "パズルゲームです。", 1),
    (28.5, 30.0, "いいですね。", 0),
    (30.5, 33.0, "では次の話題に行きましょう。", 2),
]


class Counter:
    source = "test-fake"

    def count_text(self, text):
        return max(1, len(text or "") // 2)

    def count_texts(self, texts):
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds):
        return max(0, int(seconds * 32))


def rows():
    return [{"id": str(i), "start": s, "end": e, "text": t,
             "speaker": {"track": k, "tracks": [k], "block": 0, "labels": [{"role": "unknown", "conf": "high"}]}}
            for i, (s, e, t, k) in enumerate(LINES, start=1)]


def raw_srt(tmp_path):
    """As `stages._write_raw_srt` writes it."""
    stable = tmp_path / "sample-stable.json"
    stable.write_text(json.dumps({"segments": [{k: v for k, v in r.items() if k != "speaker"} for r in rows()]},
                                 ensure_ascii=False), encoding="utf-8")
    written = rendering.convert_json_to_srt(stable, output_path=tmp_path / "sample-raw.srt", validate=False)
    assert written.keys is None
    for profile in TIMELINE_POSTPROCESS_PROFILES:
        postprocess_srt_file(written.path, profile=profile, validate=False, audio_end=AUDIO_END, keys=written.keys)
    return written.path.read_text(encoding="utf-8")


def reply(window, merges):
    """A basicA answer for `window` (seven columns, no absolute merge threshold, so the stock phrase may merge with
    the long line it wraps): the local id runs in `merges` merged, every other line alone."""
    local = [str(i) for i in range(1, len(window.segments) + 1)]
    by_local = dict(zip(local, window.segments))
    runs, i = [], 0
    while i < len(local):
        run = next((m for m in merges if m[0] == local[i]), (local[i],))
        runs.append(run)
        i += len(run)
    lines = [correction_output_header(resolve_variant("basicA"))]
    for run in runs:
        translation = "译" * (2 + len(run))
        lines.append(f"sub|{','.join(run)}|{by_local[run[0]].start:.1f}|"
                     f"{''.join(by_local[n].text for n in run)}|{translation}|high|")
    return "<translated>\n" + "\n".join(lines) + "\n</translated>"


def produce(tmp_path):
    out = {"raw_srt": raw_srt(tmp_path)}
    speakers = {r["id"]: speaker_of(r) for r in rows()}
    segments = chunking._subtitle_segments(rows(), speakers)
    windows = chunking.plan_correction_windows(segments, counter=Counter(), max_window_subtitle_tokens=80,
                                               overlap_window_seconds=4.0, audio_duration=AUDIO_END)
    windows = [replace(w, length_scale=1.0) for w in windows]
    out["windows"] = json.dumps([
        {"id": w.chunk_id, "ids": [s.id for s in w.segments], "overlap": [s.id for s in w.overlap_segments],
         "reason": w.boundary_reason, "clip": [w.clip_start, w.clip_end], "end": w.end,
         "range": list(compute_clip_range(w.segments, global_first_id=segments[0].id,
                                          global_last_id=segments[-1].id, audio_duration=AUDIO_END))}
        for w in windows], ensure_ascii=False)
    for w in windows:
        out[f"csv/{w.chunk_id}"] = chunking.render_window_segments_as_csv(w)
        out[f"preceding/{w.chunk_id}"] = chunking.render_window_preceding_as_csv(w)
    halves = chunking.split_window_in_half(windows[0], counter=Counter(), global_first_id=segments[0].id,
                                           global_last_id=segments[-1].id, audio_duration=AUDIO_END)
    out["halves"] = json.dumps([[[s.id for s in h.segments], [s.id for s in h.overlap_segments], h.clip_start,
                                 h.clip_end] for h in halves])
    # single pass: the stock phrase merged with the line it wraps (across a turn), the nested line with its host
    result = validate_correction_window_output(reply(windows[0], [("3", "4"), ("5", "6")]), windows[0],
                                               variant=resolve_variant("basicA", CapabilityTier.BASIC))
    assert result.ok, result.errors
    out["merged"] = json.dumps([[list(r.source_ids), r.start, r.end] for r in result.segments])
    out["final_srt"] = postprocess_srt_text(render_translated_segments_as_srt(result.segments), audio_end=AUDIO_END)[0]
    # three passes: pass 1 joins the nested line to its host and marks one line mergeable
    grouping = group_lines(segments, {"6": ACT_JOIN, "8": ACT_MAY},
                           different=lambda a, b: different_speakers(speakers[a], speakers[b]),
                           breaks=[w.segments[len(w.overlap_segments)].id for w in windows], thresholds=Thresholds.at(1.0))
    out["grouping"] = json.dumps([list(grouping.groups), {k: v.value for k, v in grouping.seams.items()}],
                                 ensure_ascii=False)
    for w in grouped_windows(windows, grouping):
        out[f"grouped/{w.chunk_id}"] = chunking.render_window_segments_as_csv(w)
    return out


def test_a_local_task_with_overlapping_lines_comes_out_as_before_the_overlap_rules(tmp_path):
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    produced = produce(tmp_path)
    assert produced.keys() == expected.keys()
    for name, text in expected.items():
        assert produced[name] == text, name
