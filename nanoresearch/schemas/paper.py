"""Paper structure data models: skeleton, sections, figures."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Section(BaseModel):
    """A single section of the paper."""

    heading: str = Field(description="Section title, e.g. 'Introduction'")
    label: str = Field(default="", description="LaTeX label, e.g. 'sec:intro'")
    content: str = Field(default="", description="LaTeX body text for this section")
    subsections: list[Section] = Field(default_factory=list)

    @field_validator("heading", "label", "content", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class FigurePlaceholder(BaseModel):
    """A placeholder for a figure to be generated."""

    figure_id: str = Field(description="Unique identifier, e.g. fig:overview")
    caption: str = ""
    figure_type: str = Field(
        default="placeholder",
        description="Type: placeholder | bar_chart | line_chart | diagram | table",
    )
    data: dict = Field(
        default_factory=dict,
        description="Data for figure generation (varies by figure_type)",
    )
    width: str = "\\textwidth"
    filename: str = ""

    @field_validator("figure_id", "caption", "figure_type", "width", "filename", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""


class PaperSkeleton(BaseModel):
    """Complete paper structure for LaTeX generation."""

    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    sections: list[Section] = Field(default_factory=list)
    figures: list[FigurePlaceholder] = Field(default_factory=list)
    references_bibtex: str = Field(default="", description="BibTeX entries as a string")
    template_format: str = Field(
        default="arxiv",
        description="Template format (auto-discovered from templates directory)",
    )

    @field_validator("title", "abstract", "references_bibtex", "template_format", mode="before")
    @classmethod
    def _coerce_to_str(cls, v):
        if isinstance(v, list):
            return "; ".join(str(x) for x in v) if v else ""
        if isinstance(v, dict):
            return str(v)
        return v if isinstance(v, str) else str(v) if v is not None else ""

    @field_validator("template_format")
    @classmethod
    def _check_template(cls, v: str) -> str:
        from nanoresearch.templates import get_available_formats

        available = get_available_formats()
        if v not in available:
            raise ValueError(
                f"Unknown template_format {v!r}. Available: {available}"
            )
        return v
