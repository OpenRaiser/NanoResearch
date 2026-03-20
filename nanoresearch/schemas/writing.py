"""Writing stage output schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class WritingOutput(BaseModel):
    """Output of the writing stage."""

    title: str = ""
    abstract: str = ""
    sections: dict[str, str] = Field(default_factory=dict, description="section_name -> LaTeX content")
    bibtex: str = ""
    paper_tex_path: str = ""
    references_bib_path: str = ""
