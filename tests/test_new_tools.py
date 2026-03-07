"""Tests for new MCP tools: web_search, paperswithcode, pdf_reader."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

# ============================================================
# Web Search Tests
# ============================================================

MOCK_DDGS_RESULTS = [
    {"title": "First Result Title", "href": "https://example.com/page1", "body": "This is the first snippet."},
    {"title": "Second Result Title", "href": "https://example.com/page2", "body": "This is the second snippet."},
]


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_search_web_parses_results(self):
        from mcp_server.tools.web_search import search_web

        mock_ddgs_cls = MagicMock()
        mock_ddgs_inst = MagicMock()
        mock_ddgs_inst.text = MagicMock(return_value=MOCK_DDGS_RESULTS)
        mock_ddgs_cls.return_value = mock_ddgs_inst

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            with patch("mcp_server.tools.web_search._limiter") as mock_limiter:
                mock_limiter.acquire = AsyncMock()
                results = await search_web("test query")
        assert len(results) == 2
        assert results[0]["title"] == "First Result Title"
        assert results[0]["url"] == "https://example.com/page1"
        assert results[0]["snippet"] == "This is the first snippet."

    @pytest.mark.asyncio
    async def test_search_web_empty_results(self):
        from mcp_server.tools.web_search import search_web

        mock_ddgs_cls = MagicMock()
        mock_ddgs_inst = MagicMock()
        mock_ddgs_inst.text = MagicMock(return_value=[])
        mock_ddgs_cls.return_value = mock_ddgs_inst

        mock_module = MagicMock()
        mock_module.DDGS = mock_ddgs_cls

        with patch.dict("sys.modules", {"duckduckgo_search": mock_module}):
            with patch("mcp_server.tools.web_search._limiter") as mock_limiter:
                mock_limiter.acquire = AsyncMock()
                results = await search_web("test query")
        assert results == []


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

        mock_web_results = [
            {"title": "Image Classification", "url": "https://paperswithcode.com/task/image-classification", "snippet": "Classify images."},
            {"title": "Object Detection", "url": "https://paperswithcode.com/task/object-detection", "snippet": "Detect objects."},
            {"title": "Unrelated", "url": "https://example.com/other", "snippet": "Not PwC."},
        ]

        with patch("mcp_server.tools.web_search.search_web", new_callable=AsyncMock) as mock_sw:
            mock_sw.return_value = mock_web_results
            results = await search_tasks("image classification")

        assert len(results) == 2  # only paperswithcode.com URLs
        assert results[0]["name"] == "Image Classification"
        assert "paperswithcode.com" in results[0]["url"]

    @pytest.mark.asyncio
    async def test_get_sota(self):
        from mcp_server.tools.paperswithcode import get_sota

        # PwC API is defunct, should return empty list
        results = await get_sota("image-classification")
        assert results == []


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
