import json
import sys

from tools.split_explorer import __main__ as explorer


def test_no_split_srt_preserves_source_text_and_segments_cache(
    tmp_path, monkeypatch
) -> None:
    stable = tmp_path / "sample-stable.json"
    cache = tmp_path / "sample-vad.json"
    output = tmp_path / "sample.srt"
    stable.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "原 文",
                        "words": [
                            {
                                "start": 0.0,
                                "end": 2.0,
                                "word": "原文",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cache.write_text(
        json.dumps({"segments": [{"start": 0.0, "end": 2.0}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_explorer",
            str(stable),
            "--cache",
            str(cache),
            "--srt",
            str(output),
        ],
    )

    assert explorer.main() == 0
    assert "原 文" in output.read_text(encoding="utf-8")
    assert "原文" not in output.read_text(encoding="utf-8")
