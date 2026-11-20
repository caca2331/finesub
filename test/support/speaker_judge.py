"""Builders for the speaker judge's LLM step: correction windows, labelled rows, a fake client, one run.

The judge's own tests are `test_speaker_judge_stage.py`; the correction, three-pass and product tests
drive the judge through these too.
"""

from __future__ import annotations

import json
import re

from finesub.llm.chunking import SubtitleSegment, SubtitleWindow
from finesub.llm.client import LLMCallResult
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.speaker_judge import prompt, run as judge_run


def correction(*spans):
    """Correction windows whose core rows are the given 1-based id ranges."""
    out = []
    for number, (first, last) in enumerate(spans, start=1):
        segments = [SubtitleSegment(str(i), float(i), i + 1.0, f"line {i}") for i in range(first, last + 1)]
        out.append(SubtitleWindow(f"{number:04d}", segments, [], "even", None))
    return out


def speaker(track, *others, coverage=0.9, mark=None):
    payload = {"track": track, "tracks": [track, *others], "block": 0, "labels": [{"role": "unknown", "conf": "high"}],
               "coverage": {str(track): coverage, **{str(o): 0.3 for o in others}}}
    if mark:
        payload["mark"] = mark
        payload["review"] = mark.get("tier") == "low"
    return payload


def segments_of(specs):
    return [{"id": str(i), "start": float(2 * i), "end": 2 * i + float(seconds), "text": text,
             **({"speaker": sp} if sp is not None else {})}
            for i, (sp, seconds, text) in enumerate(specs, start=1)]


class FakeClient:
    """Answers the judge by moving every judged row to s1."""

    def __init__(self, *, judge=None, error=None):
        self.calls = []
        self.judge = judge
        self.error = error

    def complete(self, role, messages, **kwargs):
        self.calls.append(kwargs.get("task_group"))
        user = messages[1]["content"]
        if self.error is not None:
            raise self.error
        ids = re.findall(r"▶ (q\d+) ", user)
        content = self.judge(ids) if self.judge else "<judge>\n" + "\n".join(
            json.dumps({"id": i, "track": "s1", "why": "w"}) for i in ids) + "\n</judge>\n<tracks>\ns1 = 佐藤\n</tracks>"
        return LLMCallResult(content=content, role=role, model="fake", fallback_used=False, raw_response={})


def judge_files(tmp_path, rows=8):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": segments_of([(speaker(i % 2), 2, f"line {i}") for i in range(1, rows + 1)])}),
                      encoding="utf-8")
    return stable


def run(stable, client, tmp_path, cut=((1, 4), (5, 8)), media="text", audio=None, **options):
    return judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=correction(*cut), profile=resolve_profile("text", "none", "quality"),
        client=client, inputs=judge_run.JudgeInputs("t", pool=(prompt.Candidate("さとう"),)), media=media,
        audio=audio, task_artifact_dir=tmp_path / "art", task_id="t", **options)


def task_request(tmp_path, stable, **overrides):
    from finesub.llm.correction_request import CorrectionRequest

    return CorrectionRequest(stable_json=stable, output_path=tmp_path / "out" / "clip.srt", postprocess_profile=None,
                             knowledge="none", profile=resolve_profile("text", "none", "quality"), test_profile=True,
                             speaker_judge="text", **overrides)


def fast_whole_file(monkeypatch, orchestration, rows=8):
    window = correction((1, rows))[0]
    monkeypatch.setattr(orchestration, "decide_fast_mode", lambda **kwargs: orchestration.FastDecision(
        mode="auto", enabled=True, reason="test", window=window))
