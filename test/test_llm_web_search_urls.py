from __future__ import annotations

from finesub.llm.web_search import extract_urls_from_text


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


def test_follow_search_redirect_takes_one_hop_on_the_real_client(monkeypatch) -> None:
    """The real API client does not follow redirects (`client_kwargs`); if it did, the hop would land on the page
    and the link would come back unchanged, so the client is built for real here with only its transport stubbed."""

    import httpx

    from finesub.llm import web_search

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.host == "vertexaisearch.cloud.google.com":
            if request.url.path.endswith("/fail"):
                raise httpx.ConnectError("down", request=request)
            if request.url.path.endswith("/plain"):
                return httpx.Response(200, text="no redirect")
            return httpx.Response(302, headers={"location": "https://wiki.example/page"})
        return httpx.Response(200, text="the page")

    real = web_search.api_http_client
    monkeypatch.setattr(web_search, "api_http_client",
                        lambda *a, **kw: real(*a, transport=httpx.MockTransport(handler), **kw))
    base = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
    assert web_search.follow_search_redirect(base + "abc") == "https://wiki.example/page"
    assert seen == [base + "abc"]
    assert web_search.follow_search_redirect("https://wiki.example/other") == "https://wiki.example/other"
    assert web_search.follow_search_redirect(base + "plain") == base + "plain"
    assert web_search.follow_search_redirect(base + "fail") == base + "fail"
