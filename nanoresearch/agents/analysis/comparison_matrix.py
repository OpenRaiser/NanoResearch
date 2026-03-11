"""Build structured comparison matrix for the Experiments section."""
from __future__ import annotations


def build_comparison_matrix(
    baselines: list[dict],
    proposed: dict,
    metrics: list[dict],
) -> dict:
    """Build method comparison matrix with best/second-best annotations.

    Args:
        baselines: [{"name": str, "metrics": {metric_name: value}}]
        proposed: {"name": str, "metrics": {metric_name: value}}
        metrics: [{"name": str, "higher_is_better": bool}]

    Returns:
        {"headers": [...], "rows": [...], "annotations": {...}}
    """
    all_methods = baselines + [proposed]
    headers = ["Method"] + [m["name"] for m in metrics]

    rows = []
    for method in all_methods:
        row = {
            "method": method.get("name", "Unknown"),
            "is_proposed": method is proposed,
        }
        for m in metrics:
            val = method.get("metrics", {}).get(m["name"])
            row[m["name"]] = val
        rows.append(row)

    # Find best and second-best per metric
    # Use "row_idx:metric_name" string keys so the dict is JSON-serializable.
    annotations: dict[str, str] = {}
    for m in metrics:
        vals = [
            (i, row.get(m["name"]))
            for i, row in enumerate(rows)
            if isinstance(row.get(m["name"]), (int, float))
        ]
        if not vals:
            continue
        higher = m.get("higher_is_better", True)
        vals.sort(key=lambda x: x[1], reverse=higher)
        if len(vals) >= 1:
            annotations[f"{vals[0][0]}:{m['name']}"] = "best"
        if len(vals) >= 2:
            annotations[f"{vals[1][0]}:{m['name']}"] = "second"

    return {
        "headers": headers,
        "rows": rows,
        "annotations": annotations,
        "proposed_method_name": proposed.get("name", "Ours"),
    }


def _latex_escape_cell(text: str) -> str:
    """Escape LaTeX-special chars in table cells (headers and method names)."""
    text = text.replace("_", "\\_")
    text = text.replace("%", "\\%")
    text = text.replace("&", "\\&")
    text = text.replace("#", "\\#")
    text = text.replace("$", "\\$")
    return text


def comparison_matrix_to_latex(matrix: dict) -> str:
    r"""Render comparison matrix as LaTeX tabular.

    Best values are \textbf{bold}, second-best are \underline{underlined}.
    Metric names and method names with underscores are escaped automatically.
    """
    headers = matrix["headers"]
    rows = matrix["rows"]
    annotations = matrix["annotations"]

    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(
            f"\\textbf{{{_latex_escape_cell(h)}}}" for h in headers
        ) + " \\\\",
        "\\midrule",
    ]

    for i, row in enumerate(rows):
        cells = []
        method_name = _latex_escape_cell(row["method"])
        if row.get("is_proposed"):
            method_name = f"\\textbf{{{method_name}}} (Ours)"
        cells.append(method_name)

        for h in headers[1:]:
            val = row.get(h)
            if val is None:
                cells.append("--")
                continue
            if isinstance(val, float):
                formatted = f"{val:.2f}" if val < 1 else f"{val:.1f}"
            else:
                formatted = str(val)
            ann = annotations.get(f"{i}:{h}")
            if ann == "best":
                formatted = f"\\textbf{{{formatted}}}"
            elif ann == "second":
                formatted = f"\\underline{{{formatted}}}"
            cells.append(formatted)

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
