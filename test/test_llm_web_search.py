from __future__ import annotations

import json
import tempfile
from pathlib import Path

from llm.web_search import (
    ExtractRequest,
    GEMMA4_SEARCH_BATCH_QUERY_LIMIT,
    QueryExtractResult,
    QuerySearchResult,
    SearchRequest,
    SearchResultItem,
    WebSearchClient,
    render_extract_results,
    render_search_results,
    search_results_metadata,
)


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeHttpClient:
    def __init__(self, script, calls, *, timeout: float) -> None:
        self.script = script
        self.calls = calls
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url, **kwargs):
        self.calls.append(("post", url, kwargs))
        return self.script.pop(0)

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        return self.script.pop(0)


def _client(script, calls, **kwargs):
    def factory(**factory_kwargs):
        return FakeHttpClient(script, calls, **factory_kwargs)

    defaults = dict(
        exa_keys=[],
        gemma_keys=[],
        tavily_keys=["tvly-key"],
        state_path=Path(tempfile.mkdtemp()) / "state.json",
        client_factory=factory,
        sleep_func=lambda _: None,
        query_interval_seconds=0.0,
        max_retries=0,
    )
    defaults.update(kwargs)
    return WebSearchClient(**defaults)


def test_tavily_search_success_uses_auto_parameters_and_bearer() -> None:
    calls = []
    script = [
        FakeResponse(
            payload={
                "answer": "游戏B 是恐怖游戏",
                "results": [
                    {"title": "wiki", "url": "https://example.test", "content": "恐怖游戏介绍"}
                ],
            }
        )
    ]
    client = _client(script, calls)

    result = client.search("游戏B 剧情")

    assert result.provider == "tavily"
    assert result.answer == "游戏B 是恐怖游戏"
    assert result.items[0].url == "https://example.test"
    method, url, kwargs = calls[0]
    assert method == "post"
    assert "api.tavily.com" in url
    assert kwargs["json"]["auto_parameters"] is True
    assert kwargs["headers"]["Authorization"] == "Bearer tvly-key"


def test_disabled_search_providers_are_skipped_without_network_calls(monkeypatch) -> None:
    from llm import api_keys

    calls = []
    monkeypatch.setattr(
        api_keys,
        "read_config",
        lambda path=None: {
            "providers": {
                "exa": False,
                "gemma4_grounded": False,
                "tavily": False,
                "duckduckgo": False,
            }
        },
    )
    client = _client([], calls)

    result = client.search("disabled providers")

    assert result.error == "no provider available"
    assert calls == []


def test_exa_search_is_primary_and_cleans_summary_highlights_and_guided() -> None:
    calls = []
    script = [
        FakeResponse(
            payload={
                "results": [
                    {
                        "title": "诺姆 wiki",
                        "url": "https://exa.test/nom",
                        "summary": "诺姆是绳匠",
                        "highlights": ["与安比同属", "邦布伙伴"],
                        "image": "https://exa.test/nom.png",
                    }
                ]
            }
        )
    ]
    client = _client(script, calls, exa_keys=["exa-key"])

    result = client.search("绝区零 诺姆", guided_query="诺姆的人际关系")

    assert result.provider == "exa"
    assert result.items[0].url == "https://exa.test/nom"
    # Summary + highlights joined; image url dropped.
    assert "诺姆是绳匠" in result.items[0].snippet
    assert "邦布伙伴" in result.items[0].snippet
    assert "nom.png" not in result.items[0].snippet
    method, url, kwargs = calls[0]
    assert method == "post" and "api.exa.ai/search" in url
    assert kwargs["headers"]["x-api-key"] == "exa-key"
    assert kwargs["json"]["type"] == "deep"
    # Guided query steers highlights without changing the search query itself.
    assert kwargs["json"]["query"] == "绝区零 诺姆"
    assert kwargs["json"]["contents"]["highlights"]["query"] == "诺姆的人际关系"


def test_exa_key_error_falls_back_to_tavily_search() -> None:
    calls = []
    script = [
        FakeResponse(status_code=429, text="exa rate limited"),
        FakeResponse(
            payload={
                "results": [
                    {"title": "t", "url": "https://tv.test", "content": "tavily 命中"}
                ]
            }
        ),
    ]
    client = _client(script, calls, exa_keys=["exa-key"], tavily_keys=["tvly-key"])

    result = client.search("query")

    assert result.provider == "tavily"
    assert result.items[0].snippet == "tavily 命中"
    assert [event.reason for event in result.fallbacks] == ["key_locked"]
    assert [method for method, _, _ in calls] == ["post", "post"]
    assert "api.exa.ai/search" in calls[0][1]
    assert "api.tavily.com/search" in calls[1][1]


def test_exa_key_error_falls_back_to_gemma4_before_tavily() -> None:
    calls = []
    script = [
        FakeResponse(status_code=429, text="exa rate limited"),
        FakeResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '<gemma4_search_results>{"results":[{"id":"q1",'
                                        '"query":"query","summary":"Gemma 摘要",'
                                        '"sources":[{"title":"Gemma Source",'
                                        '"url":"https://gemma.test/a"}]}]}'
                                        "</gemma4_search_results>"
                                    )
                                }
                            ]
                        },
                        "groundingMetadata": {
                            "webSearchQueries": ["query"],
                            "groundingChunks": [
                                {
                                    "web": {
                                        "uri": "https://gemma.test/a",
                                        "title": "Gemma Source",
                                    }
                                }
                            ],
                            "groundingSupports": [
                                {
                                    "segment": {"text": "接地句子"},
                                    "groundingChunkIndices": [0],
                                }
                            ],
                        },
                    }
                ],
                "usageMetadata": {"promptTokenCount": 25},
            }
        ),
    ]
    client = _client(script, calls, exa_keys=["exa-key"], gemma_keys=["gemma-key"])

    result = client.search("query")

    assert result.provider == "gemma4"
    assert result.answer == "Gemma 摘要"
    assert result.items[0].url == "https://gemma.test/a"
    assert result.items[0].snippet == "接地句子"
    assert [event.reason for event in result.fallbacks] == ["key_locked"]
    assert [method for method, _, _ in calls] == ["post", "post"]
    assert "api.exa.ai/search" in calls[0][1]
    assert "gemma-4-31b-it:generateContent" in calls[1][1]
    gemma_payload = calls[1][2]["json"]
    assert gemma_payload["tools"] == [{"google_search": {}}]
    assert "<|think|>" in gemma_payload["contents"][0]["parts"][0]["text"]
    assert "medium-depth thinking" in gemma_payload["contents"][0]["parts"][0]["text"]


def test_gemma4_search_metadata_is_compact() -> None:
    calls = []
    script = [
        FakeResponse(status_code=500, text="exa down"),
        FakeResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '<gemma4_search_results>{"results":[{"id":"q1",'
                                        '"summary":"摘要","sources":[{"url":"https://a.test"}]}]}'
                                        "</gemma4_search_results>"
                                    )
                                }
                            ]
                        },
                        "groundingMetadata": {
                            "webSearchQueries": ["q"],
                            "groundingChunks": [
                                {"web": {"uri": "https://a.test", "title": "A"}}
                            ],
                            "groundingSupports": [
                                {
                                    "segment": {"text": "A support"},
                                    "groundingChunkIndices": [0],
                                }
                            ],
                        },
                    }
                ]
            }
        ),
    ]
    client = _client(script, calls, exa_keys=["exa-key"], gemma_keys=["gemma-key"])

    metadata = search_results_metadata([client.search("q")])

    provider_meta = metadata[0]["provider_metadata"]
    assert provider_meta["web_search_queries"] == ["q"]
    assert provider_meta["grounding_chunks"] == [{"title": "A", "url": "https://a.test"}]
    assert provider_meta["grounding_supports"] == [
        {"text": "A support", "groundingChunkIndices": [0]}
    ]


def test_gemma4_without_grounding_falls_back_to_tavily() -> None:
    calls = []
    script = [
        FakeResponse(status_code=500, text="exa down"),
        FakeResponse(
            payload={
                "candidates": [
                    {"content": {"parts": [{"text": "ungrounded answer"}]}}
                ]
            }
        ),
        FakeResponse(
            payload={
                "candidates": [
                    {"content": {"parts": [{"text": "still ungrounded"}]}}
                ]
            }
        ),
        FakeResponse(
            payload={
                "results": [
                    {"title": "t", "url": "https://tv.test", "content": "tavily 命中"}
                ]
            }
        ),
    ]
    client = _client(
        script,
        calls,
        exa_keys=["exa-key"],
        gemma_keys=["gemma-key"],
        tavily_keys=["tvly-key"],
    )

    result = client.search("query")

    assert result.provider == "tavily"
    assert result.items[0].snippet == "tavily 命中"
    assert [event.provider for event in result.fallbacks] == ["exa", "gemma4"]


def test_gemma4_retries_without_visible_thinking_when_grounding_is_empty() -> None:
    calls = []
    script = [
        FakeResponse(status_code=500, text="exa down"),
        FakeResponse(
            payload={
                "candidates": [
                    {"content": {"parts": [{"text": "thinking but no chunks"}]}}
                ],
                "usageMetadata": {"promptTokenCount": 20},
            }
        ),
        FakeResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '<gemma4_search_results>{"results":[{"id":"q1",'
                                        '"summary":"retry 摘要",'
                                        '"sources":[{"url":"https://retry.test"}]}]}'
                                        "</gemma4_search_results>"
                                    )
                                }
                            ]
                        },
                        "groundingMetadata": {
                            "webSearchQueries": ["q"],
                            "groundingChunks": [
                                {"web": {"uri": "https://retry.test", "title": "Retry"}}
                            ],
                        },
                    }
                ],
            }
        ),
    ]
    client = _client(script, calls, exa_keys=["exa-key"], gemma_keys=["gemma-key"])

    result = client.search("q")

    assert result.provider == "gemma4"
    assert result.items[0].url == "https://retry.test"
    assert result.metadata["retried_without_visible_thinking"] is True
    first_prompt = calls[1][2]["json"]["contents"][0]["parts"][0]["text"]
    retry_prompt = calls[2][2]["json"]["contents"][0]["parts"][0]["text"]
    assert "<|think|>" in first_prompt
    assert "<|think|>" not in retry_prompt


def test_gemma4_search_batches_pending_queries_at_eight_per_pass() -> None:
    def gemma_payload(count: int):
        rows = [
            {
                "id": f"q{idx + 1}",
                "summary": f"摘要 {idx + 1}",
                "sources": [{"url": f"https://gemma.test/{idx + 1}"}],
            }
            for idx in range(count)
        ]
        chunks = [
            {
                "web": {
                    "uri": f"https://gemma.test/{idx + 1}",
                    "title": f"Source {idx + 1}",
                }
            }
            for idx in range(count)
        ]
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "<gemma4_search_results>"
                                    + json.dumps({"results": rows}, ensure_ascii=False)
                                    + "</gemma4_search_results>"
                                )
                            }
                        ]
                    },
                    "groundingMetadata": {
                        "webSearchQueries": [f"query-{idx + 1:02d}" for idx in range(count)],
                        "groundingChunks": chunks,
                    },
                }
            ],
        }

    calls = []
    script = [
        FakeResponse(payload=gemma_payload(GEMMA4_SEARCH_BATCH_QUERY_LIMIT)),
        FakeResponse(payload=gemma_payload(1)),
    ]
    client = _client(script, calls, gemma_keys=["gemma-key"], tavily_keys=[])

    results = client.search_many(
        [f"query-{idx + 1:02d}" for idx in range(GEMMA4_SEARCH_BATCH_QUERY_LIMIT + 1)]
    )

    assert len(results) == GEMMA4_SEARCH_BATCH_QUERY_LIMIT + 1
    assert {result.provider for result in results} == {"gemma4"}
    gemma_calls = [call for call in calls if "gemma-4-31b-it:generateContent" in call[1]]
    assert len(gemma_calls) == 2
    first_prompt = gemma_calls[0][2]["json"]["contents"][0]["parts"][0]["text"]
    second_prompt = gemma_calls[1][2]["json"]["contents"][0]["parts"][0]["text"]
    assert "query-08" in first_prompt
    assert "query-09" not in first_prompt
    assert "query-09" in second_prompt


def test_gemma4_uses_long_provider_specific_timeout() -> None:
    calls = []
    script = [
        FakeResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '<gemma4_search_results>{"results":[{"id":"q1",'
                                        '"summary":"摘要",'
                                        '"sources":[{"url":"https://timeout.test"}]}]}'
                                        "</gemma4_search_results>"
                                    )
                                }
                            ]
                        },
                        "groundingMetadata": {
                            "groundingChunks": [
                                {"web": {"uri": "https://timeout.test", "title": "Timeout"}}
                            ]
                        },
                    }
                ]
            }
        )
    ]
    clients = []

    class RecordingHttpClient(FakeHttpClient):
        def __init__(self, script, calls, *, timeout: float) -> None:
            super().__init__(script, calls, timeout=timeout)
            clients.append(self)

    def factory(**factory_kwargs):
        return RecordingHttpClient(script, calls, **factory_kwargs)

    client = WebSearchClient(
        exa_keys=[],
        gemma_keys=["gemma-key"],
        tavily_keys=[],
        client_factory=factory,
        query_interval_seconds=0.0,
        max_retries=0,
    )

    result = client.search("q")

    assert result.provider == "gemma4"
    assert clients[0].timeout == 1200.0


def test_tavily_search_requests_advanced_answer_and_ten_results() -> None:
    calls = []
    script = [FakeResponse(payload={"results": []})]
    client = _client(script, calls)

    client.search("query")

    payload = calls[0][2]["json"]
    assert payload["include_answer"] == "advanced"
    assert payload["max_results"] == 10
    assert payload["auto_parameters"] is True


def test_exa_extract_is_primary_and_passes_guided_query() -> None:
    calls = []
    script = [
        FakeResponse(
            payload={
                "results": [
                    {
                        "url": "https://exa.test/page",
                        "title": "整页",
                        "summary": "整页摘要",
                        "highlights": ["重点句"],
                    }
                ]
            }
        )
    ]
    client = _client(script, calls, exa_keys=["exa-key"])

    result = client.extract("https://exa.test/page", guided_query="诺姆的阵营")

    assert result.provider == "exa"
    assert "整页摘要" in result.content and "重点句" in result.content
    method, url, kwargs = calls[0]
    assert method == "post" and "api.exa.ai/contents" in url
    assert kwargs["json"]["ids"] == ["https://exa.test/page"]
    assert kwargs["json"]["summary"] is True
    assert kwargs["json"]["highlights"]["query"] == "诺姆的阵营"


def test_extract_falls_back_from_exa_to_tavily() -> None:
    calls = []
    script = [
        FakeResponse(status_code=402, text="exa payment required"),
        FakeResponse(
            payload={"results": [{"url": "https://tv.test", "raw_content": "整页正文"}]}
        ),
    ]
    client = _client(script, calls, exa_keys=["exa-key"], tavily_keys=["tvly-key"])

    result = client.extract("https://tv.test", guided_query="重点")

    assert result.provider == "tavily"
    assert result.content == "整页正文"
    assert [method for method, _, _ in calls] == ["post", "post"]
    assert "api.tavily.com/extract" in calls[1][1]
    assert calls[1][2]["json"]["query"] == "重点"
    assert calls[1][2]["json"]["chunks_per_source"] == 5


def test_extract_falls_back_from_exa_to_gemma4_and_decodes_url_in_prompt() -> None:
    calls = []
    encoded_url = "https://zh.moegirl.org.cn/%E8%87%B3%E5%86%AC%E5%9C%B0%E5%8C%BA"
    decoded_url = "https://zh.moegirl.org.cn/至冬地区"
    script = [
        FakeResponse(status_code=402, text="exa payment required"),
        FakeResponse(
            payload={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": (
                                        '<gemma4_extract_results>{"results":[{"id":"u1",'
                                        f'"url":"{decoded_url}","title":"至冬地区",'
                                        '"content":"整页摘要",'
                                        f'"sources":[{{"title":"至冬地区","url":"{decoded_url}"}}]'
                                        "}]}</gemma4_extract_results>"
                                    )
                                }
                            ]
                        },
                        "groundingMetadata": {
                            "webSearchQueries": [decoded_url],
                            "groundingChunks": [
                                {"web": {"uri": decoded_url, "title": "至冬地区"}}
                            ],
                            "groundingSupports": [
                                {
                                    "segment": {"text": "正文支持句"},
                                    "groundingChunkIndices": [0],
                                }
                            ],
                        },
                    }
                ]
            }
        ),
    ]
    client = _client(script, calls, exa_keys=["exa-key"], gemma_keys=["gemma-key"])

    result = client.extract(encoded_url, guided_query="历史")

    assert result.provider == "gemma4"
    assert result.title == "至冬地区"
    assert "整页摘要" in result.content
    assert "正文支持句" in result.content
    prompt = calls[1][2]["json"]["contents"][0]["parts"][0]["text"]
    assert decoded_url in prompt
    assert encoded_url in prompt
    assert result.metadata["decoded_url"] == decoded_url


def test_extract_reports_error_when_all_providers_fail() -> None:
    calls = []
    script = [FakeResponse(status_code=401, text="exa bad key")]
    client = _client(script, calls, exa_keys=["exa-key"], tavily_keys=[])

    result = client.extract("https://x.test")

    assert not result.ok
    assert "exa" in result.error


def test_search_request_guided_query_is_carried_into_search_many() -> None:
    calls = []
    script = [FakeResponse(payload={"results": []})]
    client = _client(script, calls, exa_keys=["exa-key"])

    client.search_many([SearchRequest(query="q1", guided_query="focus")])

    assert calls[0][2]["json"]["query"] == "q1"
    assert calls[0][2]["json"]["contents"]["highlights"]["query"] == "focus"


def test_extract_many_dedupes_and_carries_guided_query() -> None:
    calls = []
    script = [
        FakeResponse(payload={"results": [{"url": "https://a.test", "summary": "A"}]}),
    ]
    client = _client(script, calls, exa_keys=["exa-key"])

    results = client.extract_many(
        [
            ExtractRequest(url="https://a.test", guided_query="重点A"),
            ExtractRequest(url="https://a.test", guided_query="dup"),
        ]
    )

    assert [r.url for r in results] == ["https://a.test"]
    assert len(calls) == 1
    assert calls[0][2]["json"]["highlights"]["query"] == "重点A"


def test_render_extract_results_groups_and_truncates() -> None:
    results = [
        QueryExtractResult(
            url="https://a.test", provider="exa", title="标题", content="x" * 500
        ),
        QueryExtractResult(url="https://b.test", error="提取失败原因"),
    ]

    block = render_extract_results(
        results, max_content_tokens=100, max_total_tokens=10_000, count_tokens=len
    )
    text = block.text

    assert "--- 深度提取 url: https://a.test ---" in text
    assert "title: 标题" in text
    assert "x" * 80 + "…" in text  # content soft-capped near 100 tokens
    assert "x" * 300 not in text
    assert "提取失败" in text
    assert block.included == ("https://a.test", "https://b.test")


def test_tavily_quota_error_locks_key_and_retries_next_pool_key() -> None:
    calls = []
    script = [
        FakeResponse(status_code=432, text="plan limit exceeded"),
        FakeResponse(
            payload={
                "results": [
                    {"title": "next key", "url": "https://next.test", "content": "第二个 key"}
                ],
            }
        ),
        # Second query prefers the most recently successful key.
        FakeResponse(payload={"results": []}),
    ]
    client = _client(script, calls, tavily_keys=["bad-key", "good-key"])

    first = client.search("query one")
    second = client.search("query two")

    assert first.provider == "tavily"
    assert first.items[0].snippet == "第二个 key"
    assert first.fallbacks[0].reason == "key_locked"
    assert second.provider == "tavily"
    assert [method for method, _, _ in calls] == ["post", "post", "post"]
    assert calls[0][2]["headers"]["Authorization"] == "Bearer bad-key"
    assert calls[1][2]["headers"]["Authorization"] == "Bearer good-key"
    assert calls[2][2]["headers"]["Authorization"] == "Bearer good-key"


_DDG_PAGE = (
    '<a rel="nofollow" class="result__a" '
    'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Fwiki&amp;rut=x">'
    "<b>DDG</b> 结果标题</a>"
    '<a class="result__snippet" href="#">DDG <b>摘要</b>文本</a>'
)


def test_provider_error_falls_back_to_duckduckgo() -> None:
    calls = []
    script = [
        FakeResponse(status_code=500, text="tavily down"),
        FakeResponse(text=_DDG_PAGE),
    ]
    client = _client(script, calls)

    result = client.search("query")

    assert result.provider == "duckduckgo"
    assert result.items[0].title == "DDG 结果标题"
    assert result.items[0].url == "https://example.test/wiki"
    assert result.items[0].snippet == "DDG 摘要文本"
    methods = [(method, url) for method, url, _ in calls]
    assert "html.duckduckgo.com" in methods[1][1]


def test_all_tavily_pool_keys_locked_falls_back_to_duckduckgo() -> None:
    calls = []
    script = [
        FakeResponse(status_code=432, text="limit one"),
        FakeResponse(status_code=432, text="limit two"),
        FakeResponse(text=_DDG_PAGE),
    ]
    client = _client(script, calls, tavily_keys=["k1", "k2"])

    result = client.search("query")

    assert result.provider == "duckduckgo"
    assert [event.reason for event in result.fallbacks] == ["key_locked", "key_locked"]
    assert [method for method, _, _ in calls] == ["post", "post", "get"]


def test_all_providers_failing_reports_error_result() -> None:
    calls = []
    script = [
        FakeResponse(status_code=401, text="bad key"),
        FakeResponse(status_code=403, text="ddg blocked"),
    ]
    client = _client(script, calls)

    result = client.search("query")

    assert not result.ok
    assert "tavily" in result.error and "duckduckgo" in result.error


def test_search_many_dedupes_caps_and_paces() -> None:
    calls = []
    script = [
        FakeResponse(payload={"results": []}),
        FakeResponse(payload={"results": []}),
    ]
    client = _client(script, calls)

    results = client.search_many(
        ["q1", "q1", " q2 ", "q3"],
        max_queries=2,
    )

    assert [result.query for result in results] == ["q1", "q2"]
    assert len(calls) == 2


def test_render_search_results_groups_truncates_and_reports_errors() -> None:
    results = [
        QuerySearchResult(
            query="q1",
            provider="tavily",
            items=(SearchResultItem(title="t1", url="u1", snippet="s" * 1000),),
            answer="回答",
        ),
        QuerySearchResult(query="q2", error="all failed"),
    ]

    block = render_search_results(
        results, max_snippet_tokens=100, max_total_tokens=10_000, count_tokens=len
    )
    text = block.text

    assert "--- query: q1 ---" in text
    assert "answer: 回答" in text
    assert "t1 (u1)" in text
    assert "s" * 80 + "…" in text  # snippet soft-capped near 100 tokens
    assert "s" * 300 not in text
    assert "--- query: q2 ---" in text
    assert "搜索失败" in text
    assert block.included == ("q1", "q2")
    assert not block.truncated and not block.dropped

    # A tight block budget drops the tail query and reports it.
    capped = render_search_results(
        results, max_snippet_tokens=100, max_total_tokens=200, count_tokens=len
    )
    assert "注入预算说明" in capped.text
    assert capped.dropped
    assert capped.tokens <= 200


def test_search_results_metadata_is_compact() -> None:
    results = [
        QuerySearchResult(
            query="q1",
            provider="duckduckgo",
            items=(SearchResultItem(title="t", url="https://a.test", snippet="long " * 100),),
        )
    ]

    metadata = search_results_metadata(results)

    assert metadata[0]["provider"] == "duckduckgo"
    assert metadata[0]["item_count"] == 1
    assert metadata[0]["urls"] == ["https://a.test"]
    assert metadata[0]["fallbacks"] == []
    assert "snippet" not in metadata[0]
