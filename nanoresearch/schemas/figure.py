"""Figure generation stage output schema."""
from __future__ import annotations

from pydantic import BaseModel, Field


class FigureRecord(BaseModel):
    """A single generated figure."""

    figure_id: str = ""
    title: str = ""
    path: str = ""
    chart_type: str = ""
    description: str = ""


class FigureOutput(BaseModel):
    """Output of the figure generation stage."""

    figures: list[FigureRecord] = Field(default_factory=list)
    figure_count: int = 0
    status: str = "pending"
