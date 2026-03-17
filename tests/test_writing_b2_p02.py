"""Tests for B2 (baseline→cite key mapping) and P0-2 (component count fix)."""

import pytest

from nanoresearch.agents.writing.context_builder import _ContextBuilderMixin
from nanoresearch.agents.writing.section_writer import _SectionWriterMixin


# ── B2: _match_baselines_to_cite_keys ──


class TestMatchBaselinesToCiteKeys:
    """Test fuzzy matching of baseline names to paper cite keys."""

    def _match(self, baselines, papers, cite_keys):
        return _ContextBuilderMixin._match_baselines_to_cite_keys(
            baselines, papers, cite_keys,
        )

    def test_exact_name_in_title(self):
        baselines = [{"name": "BERT"}]
        papers = [
            {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "year": 2019},
            {"title": "GPT-2: Language Models are Unsupervised Multitask Learners", "year": 2019},
        ]
        cite_keys = {0: "devlin2019", 1: "radford2019"}
        result = self._match(baselines, papers, cite_keys)
        assert result == {"BERT": "devlin2019"}

    def test_no_match_returns_empty(self):
        baselines = [{"name": "SomeObscureModel"}]
        papers = [{"title": "Attention Is All You Need", "year": 2017}]
        cite_keys = {0: "vaswani2017"}
        result = self._match(baselines, papers, cite_keys)
        assert result == {}

    def test_multiple_baselines_matched(self):
        baselines = [
            {"name": "ResNet"},
            {"name": "VGG"},
        ]
        papers = [
            {"title": "Deep Residual Learning (ResNet)", "year": 2016},
            {"title": "Very Deep Convolutional Networks (VGG)", "year": 2015},
            {"title": "Unrelated Paper About NLP", "year": 2020},
        ]
        cite_keys = {0: "he2016", 1: "simonyan2015", 2: "smith2020"}
        result = self._match(baselines, papers, cite_keys)
        assert result["ResNet"] == "he2016"
        assert result["VGG"] == "simonyan2015"

    def test_word_boundary_match(self):
        """MuLOT should match title containing 'MuLOT' as a word."""
        baselines = [{"name": "MuLOT"}]
        papers = [
            {"title": "MuLOT: Multi-Level Optimal Transport for Multimodal", "year": 2022},
        ]
        cite_keys = {0: "pramanick2022"}
        result = self._match(baselines, papers, cite_keys)
        assert result == {"MuLOT": "pramanick2022"}

    def test_empty_inputs(self):
        assert self._match([], [], {}) == {}
        assert self._match([{"name": "X"}], [], {}) == {}
        assert self._match([], [{"title": "Y"}], {0: "k"}) == {}

    def test_non_dict_baselines_skipped(self):
        result = self._match(["not_a_dict"], [{"title": "X"}], {0: "k"})
        assert result == {}

    def test_baseline_without_name_skipped(self):
        result = self._match([{"description": "no name"}], [{"title": "X"}], {0: "k"})
        assert result == {}

    def test_case_insensitive(self):
        baselines = [{"name": "bert"}]
        papers = [{"title": "BERT: Pre-training of Deep Bidirectional Transformers"}]
        cite_keys = {0: "devlin2019"}
        result = self._match(baselines, papers, cite_keys)
        assert result == {"bert": "devlin2019"}

    def test_multi_word_token_overlap(self):
        """Multi-word baseline name matched by token overlap."""
        baselines = [{"name": "Contrastive Learning Framework"}]
        papers = [
            {"title": "A Simple Framework for Contrastive Learning of Visual Representations"}
        ]
        cite_keys = {0: "chen2020"}
        result = self._match(baselines, papers, cite_keys)
        # "contrastive", "learning", "framework" all overlap → should match
        assert result.get("Contrastive Learning Framework") == "chen2020"


class TestBaselineCiteBlock:
    """Test _baseline_cite_block formatting."""

    def test_empty_map(self):
        assert _ContextBuilderMixin._baseline_cite_block({}) == ""

    def test_format(self):
        block = _ContextBuilderMixin._baseline_cite_block({"BERT": "devlin2019"})
        assert "BERT" in block
        assert "\\cite{devlin2019}" in block
        assert "BASELINE" in block


# ── P0-2: _fix_component_count_mismatch ──


class TestFixComponentCountMismatch:
    """Test deterministic component count fixer."""

    def _fix(self, content):
        return _SectionWriterMixin._fix_component_count_mismatch(content)

    def test_four_claimed_five_listed_itemize(self):
        content = (
            "Our method consists of four key components:\n"
            "\\begin{itemize}\n"
            "\\item Module A\n"
            "\\item Module B\n"
            "\\item Module C\n"
            "\\item Module D\n"
            "\\item Module E\n"
            "\\end{itemize}"
        )
        fixed = self._fix(content)
        assert "five key components" in fixed
        assert "four" not in fixed.lower().split("key")[0].split("\n")[-1]

    def test_five_claimed_three_listed_itemize(self):
        content = (
            "We identify five main modules in our architecture.\n"
            "\\begin{itemize}\n"
            "\\item First\n"
            "\\item Second\n"
            "\\item Third\n"
            "\\end{itemize}"
        )
        fixed = self._fix(content)
        assert "three main modules" in fixed

    def test_correct_count_unchanged(self):
        content = (
            "Our method has three components:\n"
            "\\begin{itemize}\n"
            "\\item A\n"
            "\\item B\n"
            "\\item C\n"
            "\\end{itemize}"
        )
        fixed = self._fix(content)
        assert fixed == content

    def test_parenthetical_enumeration(self):
        content = (
            "The framework comprises four stages: "
            "(1) encoding, (2) alignment, (3) fusion, (4) decoding, and (5) prediction."
        )
        fixed = self._fix(content)
        assert "five stages" in fixed

    def test_subsection_count(self):
        content = (
            "We introduce three core blocks below.\n"
            "\\subsection{Block A}\nText.\n"
            "\\subsection{Block B}\nText.\n"
            "\\subsection{Block C}\nText.\n"
            "\\subsection{Block D}\nText.\n"
        )
        fixed = self._fix(content)
        assert "four core blocks" in fixed

    def test_capitalized_word_preserved(self):
        content = (
            "Four primary elements form the backbone.\n"
            "\\begin{itemize}\n"
            "\\item A\n\\item B\n\\item C\n"
            "\\end{itemize}"
        )
        fixed = self._fix(content)
        assert "Three primary elements" in fixed

    def test_no_enumeration_unchanged(self):
        content = "Our method has four components that work together seamlessly."
        fixed = self._fix(content)
        assert fixed == content  # No items to count → no change

    def test_number_out_of_range_unchanged(self):
        """Numbers > 10 not in _NUM_WORDS, should be untouched."""
        content = (
            "eleven modules are:\n"
            "\\begin{itemize}\n\\item A\n\\end{itemize}"
        )
        fixed = self._fix(content)
        assert fixed == content
