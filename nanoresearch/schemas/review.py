"""Review stage data models: section reviews, consistency issues, revision output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SectionReview(BaseModel):
    """Review of a single paper section."""

    section: str = Field(description="Section heading (e.g. 'Introduction')")
    score: int = Field(ge=1, le=10, description="Quality score 1-10")
    issues: list[str] = Field(default_factory=list, description="Identified issues")
    suggestions: list[str] = Field(default_factory=list, description="Improvement suggestions")
    strengths: list[str] = Field(default_factory=list, description="Section strengths to preserve during revision")
    score_justification: str = Field(default="", description="Brief justification for the score")

    @field_validator("section", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class ConsistencyIssue(BaseModel):
    """A cross-section consistency issue found in the paper."""

    issue_type: str = Field(description="Type: ref_mismatch, cite_missing, symbol_inconsistency, env_mismatch, etc.")
    description: str = Field(description="Human-readable description of the issue")
    locations: list[str] = Field(default_factory=list, description="Where in the paper this occurs")
    severity: Literal["low", "medium", "high"] = Field(default="medium")

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, v):
        """Normalize severity to lowercase to prevent Literal validation failures."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("issue_type", "description", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class ReviewOutput(BaseModel):
    """Complete output of the review stage."""

    overall_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Average quality score")
    section_reviews: list[SectionReview] = Field(default_factory=list)
    consistency_issues: list[ConsistencyIssue] = Field(default_factory=list)
    major_revisions: list[str] = Field(default_factory=list, description="Required major changes")
    minor_revisions: list[str] = Field(default_factory=list, description="Suggested minor changes")
    revised_sections: dict[str, str] = Field(
        default_factory=dict,
        description="Sections that were revised: heading -> new content",
    )
    revision_rounds: int = Field(default=0, description="Number of revision rounds performed")
