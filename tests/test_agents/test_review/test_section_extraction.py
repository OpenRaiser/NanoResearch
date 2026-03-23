"""Tests for nanoresearch.agents.review.section_extraction."""

from __future__ import annotations

import pytest

from nanoresearch.agents.review.section_extraction import _SectionExtractionMixin


class TestExtractSections:
    """Tests for _extract_sections."""

    def test_no_sections_returns_full_paper(self) -> None:
        tex = "Just some text without section commands."
        sections = _SectionExtractionMixin._extract_sections(tex)
        assert len(sections) == 1
        assert sections[0][0] == "Full Paper"
        assert sections[0][1] == tex
        assert sections[0][2] == 0

    def test_extracts_section(self) -> None:
        tex = r"\section{Introduction}This is intro.\section{Method}This is method."
        sections = _SectionExtractionMixin._extract_sections(tex)
        assert len(sections) == 2
        assert sections[0][0] == "Introduction"
        assert "intro" in sections[0][1]
        assert sections[1][0] == "Method"
        assert "method" in sections[1][1]

    def test_extracts_subsection_level(self) -> None:
        tex = r"\subsection{Details}Some details here."
        sections = _SectionExtractionMixin._extract_sections(tex)
        assert len(sections) == 1
        assert sections[0][0] == "Details"
        assert sections[0][2] == 1


class TestGetFullSectionContent:
    """Tests for _get_full_section_content."""

    def test_returns_content_for_heading(self) -> None:
        sections = [
            ("Introduction", "Intro text.", 0),
            ("Method", "Method text.", 0),
        ]
        content = _SectionExtractionMixin._get_full_section_content(
            sections, "Method"
        )
        assert content == "Method text."

    def test_merges_subsections_into_parent(self) -> None:
        sections = [
            ("Introduction", "Intro.", 0),
            ("Background", "Background text.", 1),
            ("Related", "Related text.", 1),
            ("Method", "Method only.", 0),
        ]
        content = _SectionExtractionMixin._get_full_section_content(
            sections, "Introduction"
        )
        assert "Intro." in content
        assert "Background" in content
        assert "Related" in content
