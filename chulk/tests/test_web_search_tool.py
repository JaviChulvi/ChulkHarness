from __future__ import annotations

from chulk.tools.web_search import WebSearchError, tavily_search_tool


def test_tavily_search_returns_bounded_cited_results_without_exposing_key() -> None:
    calls: list[tuple[str, dict[str, object], float]] = []

    def request(url: str, payload: dict[str, object], timeout: float) -> object:
        calls.append((url, payload, timeout))
        return {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com/news",
                    "content": "abcdefgh",
                    "score": 0.9,
                },
                {"title": "Unsafe", "url": "file:///etc/passwd", "content": "ignored"},
            ]
        }

    tool = tavily_search_tool(
        "top-secret",
        max_results=2,
        max_result_content_chars=4,
        request_json=request,
    )
    result = tool.callable({"query": " current news "})

    assert result.success is True
    assert result.value == {
        "query": "current news",
        "results": [
            {
                "title": "Example",
                "url": "https://example.com/news",
                "content": "abcd",
                "score": 0.9,
            }
        ],
    }
    assert "https://example.com/news" in result.observation
    assert "Cite the source URL" in result.observation
    assert "top-secret" not in result.observation
    assert calls[0][1]["api_key"] == "top-secret"
    assert calls[0][1]["search_depth"] == "basic"


def test_tavily_search_returns_sanitized_recoverable_failure() -> None:
    def request(_url: str, _payload: dict[str, object], _timeout: float) -> object:
        raise WebSearchError("Web search request failed")

    tool = tavily_search_tool("top-secret", request_json=request)
    result = tool.callable({"query": "anything"})

    assert result.success is False
    assert result.failure_kind == "environment_failure"
    assert "top-secret" not in result.observation


def test_tavily_search_rejects_invalid_configuration() -> None:
    for kwargs in ({"api_key": ""}, {"api_key": "key", "max_results": 0}):
        try:
            tavily_search_tool(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid configuration should fail")
