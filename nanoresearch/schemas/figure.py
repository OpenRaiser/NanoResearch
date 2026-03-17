"""Figure generation stage output schema."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def _sync_figure_count(self) -> "FigureOutput":
        """Keep figure_count in sync with len(figures)."""
        if self.figure_count != len(self.figures):
            self.figure_count = len(self.figures)
        return self
