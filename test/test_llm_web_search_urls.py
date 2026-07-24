from __future__ import annotations

from llm.web_search import extract_urls_from_text


def test_extract_urls_from_text_dedupes_and_limits() -> None:
    text = (
        "素材来源: https://www.youtube.com/watch?v=abc 的切片\n"
        "重复 https://www.youtube.com/watch?v=abc\n"
        "更多 https://example.com/one https://example.com/two "
        "https://example.com/three https://example.com/four "
        "https://example.com/five https://example.com/six"
    )
    urls = extract_urls_from_text(text, limit=5)

    assert urls == [
        "https://www.youtube.com/watch?v=abc",
        "https://example.com/one",
        "https://example.com/two",
        "https://example.com/three",
        "https://example.com/four",
    ]
