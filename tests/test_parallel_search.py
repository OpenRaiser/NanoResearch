"""Exercise provider selection and the real MCP transport without network access."""

import asyncio
import json
import sys
from types import SimpleNamespace

import httpx
import pytest
import respx

from mcp_server.server import handle_tool_call
from mcp_server.tools import parallel_search
from mcp_server.tools.web_search import search_web
from nanoresearch.agents.ideation import IdeationAgent
from nanoresearch.config import ResearchConfig

URL = "https://search.parallel.ai/mcp"
RESULTS = {"results": [
    {"title": "Asyncio", "url": "https://docs.python.org/3/library/asyncio.html",
     "excerpts": ["Concurrent code", "Using async/await"]},
    {"title": None, "url": "https://example.org", "excerpts": []},
]}


@pytest.fixture
def mcp_server():
    requests = []
    tool_result = {"content": [{"type": "text", "text": json.dumps(RESULTS)}]}

    async def respond(request):
        requests.append(request)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {"protocolVersion": message["params"]["protocolVersion"],
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif message["method"] == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {
                "type": "object", "properties": {"objective": {"type": "string"},
                "search_queries": {"type": "array", "items": {"type": "string"}}},
                "required": ["objective", "search_queries"]}}]}
        elif message["method"] == "tools/call":
            result = tool_result
        else:
            raise AssertionError(message)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    with respx.mock(assert_all_called=False) as router:
        router.post(URL).mock(side_effect=respond)
        router.get(URL).respond(405)
        router.delete(URL).respond(200)
        yield requests, tool_result


async def test_native_registry_dispatch(monkeypatch, mcp_server):
    monkeypatch.setenv("NANORESEARCH_WEB_SEARCH_PROVIDER", "parallel")
    # Credentials, including ambient Parallel keys, must not influence this path.
    monkeypatch.setenv("PARALLEL_API_KEY", "unused-fixture-key")
    agent = SimpleNamespace(config=ResearchConfig(literature_sources=["web"]))
    registry = await IdeationAgent._build_search_tools(agent)
    assert "search_web" in registry.names()
    results = await registry.call("search_web", {"query": "Python asyncio documentation", "max_results": 1})
    assert results == [{"title": "Asyncio", "url": RESULTS["results"][0]["url"],
                        "snippet": "Concurrent code\nUsing async/await"}]
    requests, _ = mcp_server
    for request in requests:
        assert str(request.url) == URL
        assert request.headers["user-agent"] == parallel_search._USER_AGENT
        assert "authorization" not in request.headers
    call = next(json.loads(r.content) for r in requests if json.loads(r.content)["method"] == "tools/call")
    assert call["params"]["name"] == "web_search"
    assert call["params"]["arguments"] == {
        "objective": "Python asyncio documentation", "search_queries": ["Python asyncio documentation"]}


async def test_stdio_dispatch_structured_content(monkeypatch, mcp_server):
    monkeypatch.setenv("NANORESEARCH_WEB_SEARCH_PROVIDER", "parallel")
    _, result = mcp_server
    result.update(content=[], structuredContent=RESULTS)
    response = await handle_tool_call("search_web", {"query": "asyncio"})
    assert len(response) == 2
    assert response[1] == {"title": "", "url": "https://example.org", "snippet": ""}


@pytest.mark.parametrize("tool_result", [
    {"isError": True, "content": [{"type": "text", "text": "rate limited"}]},
    {"content": [{"type": "text", "text": "not json"}]},
    {"structuredContent": {"unexpected": []}, "content": []},
])
async def test_failed_tool_is_empty(monkeypatch, mcp_server, tool_result):
    monkeypatch.setenv("NANORESEARCH_WEB_SEARCH_PROVIDER", "parallel")
    _, result = mcp_server
    result.clear()
    result.update(tool_result)
    assert await search_web("asyncio") == []


async def test_http_failure(monkeypatch):
    monkeypatch.setenv("NANORESEARCH_WEB_SEARCH_PROVIDER", "parallel")
    with respx.mock as router:
        router.post(URL).respond(503)
        assert await search_web("asyncio") == []


async def test_timeout_and_cancellation(monkeypatch):
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def waiting(*args):
        try:
            entered.set()
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(parallel_search, "_search", waiting)
    monkeypatch.setattr(parallel_search, "_TIMEOUT", 0.01)
    assert await parallel_search.search_parallel("asyncio") == []
    assert cleaned.is_set()
    cleaned.clear()
    entered.clear()
    monkeypatch.setattr(parallel_search, "_TIMEOUT", 30)
    task = asyncio.create_task(parallel_search.search_parallel("asyncio"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()


async def test_default_and_explicit_duckduckgo(monkeypatch):
    class DDGS:
        def text(self, query, **kwargs):
            return [{"title": "Existing", "href": "https://example.org", "body": "Snippet"}]

    monkeypatch.setitem(sys.modules, "duckduckgo_search", SimpleNamespace(DDGS=DDGS))
    monkeypatch.delenv("NANORESEARCH_WEB_SEARCH_PROVIDER", raising=False)
    default = await search_web("unchanged")
    monkeypatch.setenv("NANORESEARCH_WEB_SEARCH_PROVIDER", "duckduckgo")
    assert await search_web("unchanged") == default == [
        {"title": "Existing", "url": "https://example.org", "snippet": "Snippet"}]
    assert ResearchConfig().literature_sources == ["openalex"]


async def test_invalid_provider_and_arguments(monkeypatch, mcp_server):
    monkeypatch.setenv("NANORESEARCH_WEB_SEARCH_PROVIDER", "typo")
    with pytest.raises(ValueError, match="Unknown web search provider"):
        await search_web("asyncio")
    monkeypatch.setenv("NANORESEARCH_WEB_SEARCH_PROVIDER", "parallel")
    with pytest.raises(ValueError, match="non-empty"):
        await search_web(" ")
    assert await search_web("asyncio", 0) == []
    assert mcp_server[0] == []
