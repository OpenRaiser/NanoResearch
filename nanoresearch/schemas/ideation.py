"""Ideation stage data models: literature search results, gap analysis, hypotheses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from nanoresearch.schemas.evidence import EvidenceBundle


class PaperReference(BaseModel):
    """A single paper from literature search."""

    paper_id: str = Field(description="arXiv ID or Semantic Scholar ID")
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    venue: str = ""
    citation_count: int = 0
    url: str = ""
    bibtex: str = ""
    method_text: str = Field(default="", description="Method section text from full PDF reading")
    experiment_text: str = Field(default="", description="Experiment section text from full PDF reading")
    relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="LLM-assessed relevance to the topic"
    )

    @field_validator(
        "paper_id", "title", "abstract", "venue", "url", "bibtex",
        "method_text", "experiment_text", mode="before",
    )
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class GapAnalysis(BaseModel):
    """An identified research gap from surveying the literature."""

    gap_id: str = Field(description="Unique identifier, e.g. GAP-001")
    description: str = Field(description="What is missing or underexplored")
    supporting_refs: list[str] = Field(
        default_factory=list, description="paper_id list that supports this gap"
    )
    severity: Literal["low", "medium", "high"] = Field(
        default="medium", description="How important is closing this gap"
    )
    quantitative_evidence: str = Field(
        default="", description="e.g. 'Only 2/15 papers address X'"
    )
    future_work_mention: str = Field(
        default="", description="Which paper(s) mention this as future work"
    )

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, v):
        """Normalize severity to lowercase to prevent Literal validation failures."""
        if isinstance(v, str):
            v = v.lower()
            if v not in ("low", "medium", "high"):
                return "medium"
            return v
        return v

    @field_validator("gap_id", "description", "quantitative_evidence", "future_work_mention", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class Hypothesis(BaseModel):
    """A research hypothesis generated from gap analysis."""

    hypothesis_id: str = Field(description="Unique identifier, e.g. HYP-001")
    statement: str = Field(description="Concise hypothesis statement")
    gap_refs: list[str] = Field(
        default_factory=list, description="gap_id list this hypothesis addresses"
    )
    novelty_justification: str = ""
    feasibility_notes: str = ""
    closest_existing_work: str = Field(
        default="", description="Most similar published paper and how this idea differs"
    )

    @field_validator("hypothesis_id", "statement", "feasibility_notes", "closest_existing_work", "novelty_justification", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class IdeationOutput(BaseModel):
    """Complete output of the ideation stage."""

    topic: str = Field(description="Original user-provided research topic")
    search_queries: list[str] = Field(
        default_factory=list, description="Actual search queries used"
    )
    papers: list[PaperReference] = Field(
        default_factory=list, description="Retrieved papers (target ≥ 20)"
    )
    survey_summary: str = Field(default="", description="Narrative literature survey")
    gaps: list[GapAnalysis] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    selected_hypothesis: str = Field(
        default="", description="hypothesis_id of the chosen hypothesis"
    )
    rationale: str = Field(default="", description="Why this hypothesis was selected")
    evidence: EvidenceBundle = Field(
        default_factory=EvidenceBundle,
        description="Quantitative evidence extracted from paper abstracts",
    )
    reference_repos: list[dict] = Field(
        default_factory=list,
        description="GitHub reference repos (full_name, file_tree, readme_excerpt, stars)",
    )
    must_cites: list[str] = Field(
        default_factory=list,
        description="Must-cite paper titles extracted from surveys",
    )
    must_cite_matches: list[dict] = Field(
        default_factory=list,
        description="Must-cite titles matched to paper indices (title, paper_index, matched)",
    )
    # Survey-specific fields (only used when paper_mode is survey_*)
    theme_clusters: list[str] = Field(
        default_factory=list,
        description="Topic categories/themes discovered in literature (for surveys)",
    )
    key_challenges: list[str] = Field(
        default_factory=list,
        description="Open problems identified across papers (for surveys, replaces hypotheses)",
    )
    future_directions: list[str] = Field(
        default_factory=list,
        description="Future research directions extracted from paper future work sections (for surveys)",
    )
    search_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings from search APIs that failed (non-fatal, for downstream awareness)",
    )

    @field_validator("topic", "survey_summary", "selected_hypothesis", "rationale", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""
