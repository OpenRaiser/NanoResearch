"""Tests for MCP tools (with mocked HTTP)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


# --- arXiv search tests ---

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>GNN-Fold: Graph Neural Network for Protein Structure Prediction</title>
    <summary>We propose GNN-Fold, a novel approach to protein folding using graph neural networks.</summary>
    <published>2024-01-01T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <category term="cs.LG"/>
    <category term="q-bio.BM"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1" type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Equivariant Transformers for Molecular Dynamics</title>
    <summary>An equivariant transformer architecture for molecular simulations.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Carol White</name></author>
    <category term="cs.LG"/>
  </entry>
</feed>"""


class TestArxivSearch:
    @pytest.mark.asyncio
    async def test_parse_results(self):
        from mcp_server.tools.arxiv_search import _parse_atom_feed

        papers = _parse_atom_feed(SAMPLE_ARXIV_XML)
        assert len(papers) == 2
        assert papers[0]["paper_id"] == "2401.00001v1"
        assert papers[0]["title"] == "GNN-Fold: Graph Neural Network for Protein Structure Prediction"
        assert "Alice Smith" in papers[0]["authors"]
        assert papers[0]["year"] == 2024
        assert papers[0]["venue"] == "arXiv"

    @pytest.mark.asyncio
    async def test_search_arxiv_mock(self):
        from mcp_server.tools.arxiv_search import search_arxiv

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_ARXIV_XML
        mock_response.raise_for_status = lambda: None

        with patch("mcp_server.tools.arxiv_search.get_http_client") as mock_client_factory:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_factory.return_value = mock_client

            results = await search_arxiv("protein folding GNN", max_results=5)
            assert len(results) == 2
            assert results[0]["title"].startswith("GNN-Fold")


# --- Semantic Scholar tests ---

SAMPLE_S2_RESPONSE = {
    "data": [
        {
            "paperId": "abc123",
            "title": "Deep Learning for Proteins",
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
            "year": 2023,
            "abstract": "A survey of deep learning for protein tasks.",
            "venue": "Nature Methods",
            "citationCount": 100,
            "url": "https://semanticscholar.org/paper/abc123",
            "externalIds": {"ArXiv": "2301.00001"},
        }
    ]
}


class TestSemanticScholar:
    @pytest.mark.asyncio
    async def test_search_mock(self):
        from mcp_server.tools.semantic_scholar import search_semantic_scholar

        mock_response = AsyncMock()
        mock_response.json = lambda: SAMPLE_S2_RESPONSE
        mock_response.raise_for_status = lambda: None

        with patch("mcp_server.tools.semantic_scholar.get_http_client") as mock_factory:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_factory.return_value = mock_client

            results = await search_semantic_scholar("deep learning protein")
            assert len(results) == 1
            assert results[0]["title"] == "Deep Learning for Proteins"
            assert results[0]["arxiv_id"] == "2301.00001"
            assert results[0]["citation_count"] == 100


# --- Figure generation tests ---

class TestFigureGen:
    def test_placeholder(self, tmp_path: Path):
        from mcp_server.tools.figure_gen import generate_figure

        output = tmp_path / "test_fig.png"
        path = generate_figure("placeholder", {"text": "Test"}, output, "Test Figure")
        assert Path(path).is_file()

    def test_bar_chart(self, tmp_path: Path):
        from mcp_server.tools.figure_gen import generate_figure

        output = tmp_path / "bar.png"
        data = {"labels": ["A", "B", "C"], "values": [1, 2, 3], "xlabel": "X", "ylabel": "Y"}
        path = generate_figure("bar_chart", data, output, "Bar Chart")
        assert Path(path).is_file()

    def test_line_chart(self, tmp_path: Path):
        from mcp_server.tools.figure_gen import generate_figure

        output = tmp_path / "line.png"
        data = {
            "series": [{"x": [1, 2, 3], "y": [1, 4, 9], "label": "Quadratic"}],
            "xlabel": "X",
            "ylabel": "Y",
        }
        path = generate_figure("line_chart", data, output)
        assert Path(path).is_file()

    def test_table(self, tmp_path: Path):
        from mcp_server.tools.figure_gen import generate_figure

        output = tmp_path / "table.png"
        data = {"headers": ["Method", "Acc"], "rows": [["Ours", "95.0"], ["Baseline", "90.0"]]}
        path = generate_figure("table", data, output)
        assert Path(path).is_file()


# --- PDF compile tests ---

class TestPdfCompile:
    @pytest.mark.asyncio
    async def test_file_not_found(self):
        from mcp_server.tools.pdf_compile import compile_pdf

        result = await compile_pdf("/nonexistent/paper.tex")
        assert "error" in result
        assert "not found" in result["error"].lower()
