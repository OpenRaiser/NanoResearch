"""Evidence extraction data models: metrics extracted from published papers."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ExtractedMetric(BaseModel):
    """A single quantitative metric extracted from a published paper."""

    paper_id: str = Field(
        min_length=1,
        description="arXiv ID or Semantic Scholar ID of the source paper",
    )
    paper_title: str = ""
    dataset: str = Field(description="e.g. 'QM9', 'CASP14', 'ImageNet'")
    metric_name: str = Field(description="e.g. 'MAE', 'GDT-TS', 'Top-1 Accuracy'")
    value: float | str = Field(description="Numeric value or string if range/qualifier")
    unit: str = ""
    context: str = Field(default="", description="Exact quote from abstract containing the number")
    method_name: str = Field(default="", description="Name of the method that achieved this result")
    higher_is_better: bool | None = None

    @field_validator("paper_title", "unit", "context", "method_name", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class EvidenceBundle(BaseModel):
    """Collection of quantitative evidence extracted from literature."""

    extracted_metrics: list[ExtractedMetric] = Field(default_factory=list)
    extraction_notes: str = ""
    coverage_warnings: list[str] = Field(default_factory=list)

    @field_validator("extraction_notes", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""
