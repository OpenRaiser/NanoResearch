"""Tests for new MCP tools: web_search, paperswithcode, pdf_reader."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# ============================================================
# Web Search Tests
# ============================================================

MOCK_DDG_HTML = """
<html>
<body>
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="https://example.com/page1">First Result Title</a>
  <a class="result__snippet">This is the first snippet.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="https://example.com/page2">Second Result Title</a>
  <a class="result__snippet">This is the second snippet.</a>
</div>
</body>
</html>
"""


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_search_web_parses_results(self):
        from mcp_server.tools.web_search import search_web, _parse_ddg_html

        results = _parse_ddg_html(MOCK_DDG_HTML, max_results=10)
        assert len(results) == 2
        assert results[0]["title"] == "First Result Title"
        assert results[0]["url"] == "https://example.com/page1"
        assert results[0]["snippet"] == "This is the first snippet."

    @pytest.mark.asyncio
    async def test_search_web_max_results(self):
        from mcp_server.tools.web_search import _parse_ddg_html

        results = _parse_ddg_html(MOCK_DDG_HTML, max_results=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_web_empty_html(self):
        from mcp_server.tools.web_search import _parse_ddg_html

        results = _parse_ddg_html("<html><body></body></html>", max_results=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_web_with_mock_http(self):
        from mcp_server.tools.web_search import search_web

        mock_resp = MagicMock()
        mock_resp.text = MOCK_DDG_HTML
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("mcp_server.tools.web_search.get_http_client", return_value=mock_client):
            results = await search_web("test query")
        assert len(results) == 2


# ============================================================
# Papers With Code Tests
# ============================================================

MOCK_PWC_TASKS_RESPONSE = {
    "results": [
        {
            "id": "image-classification",
            "name": "Image Classification",
            "description": "Classifying images into categories.",
        },
        {
            "id": "object-detection",
            "name": "Object Detection",
            "description": "Detecting objects in images.",
        },
    ]
}

MOCK_PWC_DATASETS_RESPONSE = {
    "results": [
        {"id": "imagenet", "name": "ImageNet"},
    ]
}

MOCK_PWC_SOTA_RESPONSE = {
    "rows": [
        {
            "method": "ViT-G/14",
            "paper_title": "Scaling Vision Transformers",
            "paper_url": "https://arxiv.org/abs/2106.04560",
            "metrics": {"Top-1 Accuracy": 90.45},
        },
        {
            "method": "SwinV2-G",
            "paper_title": "Swin Transformer V2",
            "paper_url": "https://arxiv.org/abs/2111.09883",
            "metrics": {"Top-1 Accuracy": 90.17},
        },
    ]
}


class TestPapersWithCode:
    @pytest.mark.asyncio
    async def test_search_tasks(self):
        from mcp_server.tools.paperswithcode import search_tasks

        mock_resp = MagicMock()
        mock_resp.json.return_value = MOCK_PWC_TASKS_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("mcp_server.tools.paperswithcode.get_http_client", return_value=mock_client):
            results = await search_tasks("image classification")

        assert len(results) == 2
        assert results[0]["id"] == "image-classification"
        assert results[0]["name"] == "Image Classification"

    @pytest.mark.asyncio
    async def test_get_sota(self):
        from mcp_server.tools.paperswithcode import get_sota

        async def mock_get(url, **kwargs):
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            if "/datasets/" in url and "/sota/" not in url:
                mock_resp.json.return_value = MOCK_PWC_DATASETS_RESPONSE
            else:
                mock_resp.json.return_value = MOCK_PWC_SOTA_RESPONSE
            return mock_resp

        # Create a mock client that supports both async context manager usages
        def make_mock_client():
            mock_client = AsyncMock()
            mock_client.get = mock_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            return mock_client

        with patch("mcp_server.tools.paperswithcode.get_http_client", side_effect=lambda **kw: make_mock_client()):
            results = await get_sota("image-classification")

        assert len(results) >= 1
        assert results[0]["method"] == "ViT-G/14"


# ============================================================
# PDF Reader Tests
# ============================================================

class TestPdfReader:
    def test_extract_text_from_bytes_without_pymupdf(self):
        """Test graceful fallback when PyMuPDF is not installed."""
        from mcp_server.tools.pdf_reader import extract_text_from_bytes
        import importlib
        import sys

        # Test with actual empty/invalid pdf bytes (should handle gracefully)
        # This tests the code path, not real PDF extraction
        result = extract_text_from_bytes(b"not a real pdf")
        # If PyMuPDF is installed, it will fail to open; if not, returns empty
        assert "full_text" in result
        assert "sections" in result
        assert "method_text" in result
        assert "experiment_text" in result

    def test_split_sections(self):
        from mcp_server.tools.pdf_reader import _split_sections

        text = """Some preamble text.

1 Introduction
This is the introduction section.

2 Methods
This describes the methods.

3 Experiments
We ran these experiments.

4 Conclusion
We conclude that..."""

        sections = _split_sections(text)
        assert "Introduction" in sections
        assert "Methods" in sections
        assert "Experiments" in sections
        assert "Conclusion" in sections
        assert "methods" in sections["Methods"].lower()

    def test_split_sections_no_headings(self):
        from mcp_server.tools.pdf_reader import _split_sections

        text = "Just plain text without any headings."
        sections = _split_sections(text)
        assert "Preamble" in sections

    @pytest.mark.asyncio
    async def test_download_and_extract_mock(self):
        from mcp_server.tools.pdf_reader import download_and_extract

        # Mock the HTTP response with empty bytes
        mock_resp = MagicMock()
        mock_resp.content = b"fake pdf bytes"
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("mcp_server.tools.pdf_reader.get_http_client", return_value=mock_client), \
             patch("mcp_server.tools.pdf_reader.extract_text_from_bytes") as mock_extract:
            mock_extract.return_value = {
                "full_text": "test",
                "sections": {},
                "method_text": "",
                "experiment_text": "",
                "page_count": 1,
            }
            result = await download_and_extract("https://example.com/paper.pdf")
            assert result["full_text"] == "test"


# ============================================================
# Server Registration Tests
# ============================================================

class TestServerToolRegistration:
    def test_new_tools_registered(self):
        from mcp_server.server import TOOLS
        assert "search_web" in TOOLS
        assert "search_pwc_tasks" in TOOLS
        assert "get_pwc_sota" in TOOLS
        assert "read_pdf" in TOOLS

    @pytest.mark.asyncio
    async def test_handle_search_web(self):
        from mcp_server.server import handle_tool_call

        with patch("mcp_server.server.search_web", new_callable=AsyncMock) as mock:
            mock.return_value = [{"title": "Test", "url": "http://test.com", "snippet": "..."}]
            result = await handle_tool_call("search_web", {"query": "test"})
            assert len(result) == 1
            mock.assert_called_once_with("test", 10)

    @pytest.mark.asyncio
    async def test_handle_search_web_empty_query(self):
        from mcp_server.server import handle_tool_call
        with pytest.raises(ValueError, match="non-empty"):
            await handle_tool_call("search_web", {"query": ""})

    @pytest.mark.asyncio
    async def test_handle_read_pdf(self):
        from mcp_server.server import handle_tool_call

        with patch("mcp_server.server.pdf_download_and_extract", new_callable=AsyncMock) as mock:
            mock.return_value = {"full_text": "content", "sections": {}}
            result = await handle_tool_call("read_pdf", {"pdf_url": "https://example.com/p.pdf"})
            assert result["full_text"] == "content"
