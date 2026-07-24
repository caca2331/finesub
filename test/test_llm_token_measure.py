from __future__ import annotations

import json

from llm.token_measure import compare_subtitle_token_formats


class FakeCounter:
    source = "fake-counter"

    def count_text(self, text: str) -> int:
        if "-->" in text:
            return 100
        return 40

    def count_texts(self, texts):
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return 0


def test_compare_subtitle_token_formats_reports_srt_vs_csv(tmp_path) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 1.123, "end": 2.123, "text": "一"},
                    {"id": "2", "start": 2.623, "end": 3.123, "text": "二"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = compare_subtitle_token_formats(
        stable_json,
        counter=FakeCounter(),
        model="gemini/test",
    )

    assert result.segments == 2
    assert result.counter_source == "fake-counter"
    assert result.model == "gemini/test"
    assert result.srt_tokens == 100
    assert result.csv_tokens == 40
    assert result.token_reduction == 60
    assert result.token_reduction_pct == 60.0
    assert result.csv_to_srt_token_ratio == 0.4
    assert result.srt_chars > result.csv_chars
