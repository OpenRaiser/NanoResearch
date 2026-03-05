"""Figure generation tool using matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_figure(
    figure_type: str,
    data: dict[str, Any],
    output_path: str | Path,
    title: str = "",
    figsize: tuple[float, float] = (8, 5),
) -> str:
    """Generate a figure and save it as PNG.

    Args:
        figure_type: One of "bar_chart", "line_chart", "table", "placeholder".
        data: Figure-specific data.
        output_path: Where to save the PNG.
        title: Figure title.
        figsize: Figure size in inches.

    Returns:
        Absolute path of the saved figure.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    if figure_type == "bar_chart":
        _draw_bar_chart(ax, data)
    elif figure_type == "line_chart":
        _draw_line_chart(ax, data)
    elif figure_type == "table":
        _draw_table(ax, data)
    else:
        _draw_placeholder(ax, data)

    if title:
        ax.set_title(title, fontsize=12)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _draw_bar_chart(ax, data: dict) -> None:
    labels = data.get("labels", [])
    values = data.get("values", [])
    colors = data.get("colors", None)
    ax.bar(labels, values, color=colors)
    ax.set_xlabel(data.get("xlabel", ""))
    ax.set_ylabel(data.get("ylabel", ""))


def _draw_line_chart(ax, data: dict) -> None:
    for series in data.get("series", []):
        ax.plot(
            series.get("x", []),
            series.get("y", []),
            label=series.get("label", ""),
            marker=series.get("marker", "o"),
        )
    ax.set_xlabel(data.get("xlabel", ""))
    ax.set_ylabel(data.get("ylabel", ""))
    if data.get("series"):
        ax.legend()


def _draw_table(ax, data: dict) -> None:
    ax.axis("off")
    headers = data.get("headers", [])
    rows = data.get("rows", [])
    if headers and rows:
        table = ax.table(
            cellText=rows,
            colLabels=headers,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)


def _draw_placeholder(ax, data: dict) -> None:
    ax.text(
        0.5, 0.5,
        data.get("text", "[Figure Placeholder]"),
        ha="center", va="center",
        fontsize=14, color="gray",
        transform=ax.transAxes,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
