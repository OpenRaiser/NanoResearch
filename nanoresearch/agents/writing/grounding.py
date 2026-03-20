"""Evidence grounding: experiment normalization, grounding packet construction."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ._types import GroundingPacket
from . import _escape_latex_text
from .grounding_tables import _GroundingTablesMixin

logger = logging.getLogger(__name__)


class _GroundingMixin(_GroundingTablesMixin):
    """Mixin — grounding and table methods."""

    @staticmethod
    def _normalize_experiment_results(
        experiment_results: dict,
        blueprint: dict,
        experiment_analysis: dict,
    ) -> dict:
        """Coerce raw execution/analysis metrics into the main_results schema."""
        normalized = dict(experiment_results) if isinstance(experiment_results, dict) else {}
        analysis_payload = experiment_analysis if isinstance(experiment_analysis, dict) else {}
        main_results = normalized.get("main_results")
        if isinstance(main_results, list) and main_results:
            if not normalized.get("ablation_results") and isinstance(
                analysis_payload.get("ablation_results"), list
            ):
                normalized["ablation_results"] = analysis_payload.get("ablation_results", [])
            return normalized

        metric_snapshot = analysis_payload.get("final_metrics", {})
        if not isinstance(metric_snapshot, dict) or not metric_snapshot:
            metric_snapshot = {
                key: value
                for key, value in normalized.items()
                if isinstance(value, (int, float, str, bool))
            }
        if not metric_snapshot:
            return normalized

        datasets = blueprint.get("datasets", [])
        dataset_name = "Unknown Dataset"
        if isinstance(datasets, list) and datasets:
            first_dataset = datasets[0]
            if isinstance(first_dataset, dict):
                dataset_name = str(first_dataset.get("name", dataset_name)) or dataset_name
            else:
                dataset_name = str(first_dataset) or dataset_name

        method_name = (
            (blueprint.get("proposed_method") or {}).get("name")
            or "Proposed Method"
        )
        normalized["main_results"] = [
            {
                "method_name": method_name,
                "dataset": dataset_name,
                "is_proposed": True,
                "metrics": [
                    {"metric_name": key, "value": value}
                    for key, value in metric_snapshot.items()
                ],
            }
        ]
        if not normalized.get("ablation_results") and isinstance(
            analysis_payload.get("ablation_results"), list
        ):
            normalized["ablation_results"] = analysis_payload.get("ablation_results", [])
        return normalized

    # ---- grounding packet construction ----------------------------------------

    @classmethod
    def _classify_completeness(
        cls,
        experiment_status: str,
        main_results: list[dict],
        experiment_analysis: dict,
    ) -> ResultCompleteness:
        """Classify how complete the experiment results are."""
        status_lower = (experiment_status or "").lower()
        if status_lower in ("pending", "failed", "error", "unknown", ""):
            return "none"
        if not main_results:
            return "none"
        # Check for quick-eval markers
        is_quick = (
            "quick" in status_lower
            or experiment_analysis.get("is_quick_eval", False)
            or "quick-eval" in experiment_analysis.get("summary", "").lower()
            or "quick_eval" in status_lower
        )
        if is_quick:
            return "quick_eval"
        # Check for partial results (e.g., only 1 dataset out of planned N)
        converged = experiment_analysis.get("converged")
        if converged is False:
            return "partial"
        return "full"

    @classmethod
    def _build_grounding_packet(
        cls,
        experiment_results: dict,
        experiment_status: str,
        experiment_analysis: dict,
        experiment_summary: str,
        blueprint: dict,
    ) -> GroundingPacket:
        """Build a GroundingPacket from all available evidence sources."""
        normalized = cls._normalize_experiment_results(
            experiment_results or {}, blueprint, experiment_analysis or {}
        )
        analysis = experiment_analysis if isinstance(experiment_analysis, dict) else {}
        main_results = normalized.get("main_results", [])
        if not isinstance(main_results, list):
            main_results = []
        ablation_results = normalized.get("ablation_results", [])
        if not isinstance(ablation_results, list):
            ablation_results = []
        comparison = analysis.get("comparison_with_baselines", {})
        if not isinstance(comparison, dict):
            comparison = {}
        final_metrics = analysis.get("final_metrics", {})
        if not isinstance(final_metrics, dict):
            final_metrics = {}

        completeness = cls._classify_completeness(
            experiment_status, main_results, analysis,
        )

        # Identify evidence gaps
        gaps: list[str] = []
        if completeness == "none":
            gaps.append("No experiment results available")
        elif completeness == "quick_eval":
            gaps.append("Results are from quick-eval only (limited epochs/data)")
        if not ablation_results:
            gaps.append("No ablation study results")
        if not comparison:
            gaps.append("No baseline comparison data from analysis")

        packet = GroundingPacket(
            experiment_status=experiment_status,
            result_completeness=completeness,
            main_results=main_results,
            ablation_results=ablation_results,
            comparison_with_baselines=comparison,
            final_metrics=final_metrics,
            key_findings=analysis.get("key_findings", []) or [],
            limitations=analysis.get("limitations", []) or [],
            training_dynamics=str(analysis.get("training_dynamics", "")),
            analysis_summary=str(analysis.get("summary", "")),
            experiment_summary_md=experiment_summary or "",
            evidence_gaps=gaps,
        )

        # Pre-build deterministic tables when data is available
        if packet.has_real_results:
            packet.main_table_latex = cls._build_main_table_latex(
                main_results, comparison, blueprint,
            )
            if ablation_results:
                packet.ablation_table_latex = cls._build_ablation_table_latex(
                    ablation_results, blueprint,
                )
        else:
            # No real results — build scaffold tables from blueprint so the
            # Experiments section still has Table 1/Table 2 structure.
            # The LLM fills cells with literature-reported baseline numbers
            # and marks the proposed method row with "--" (to be updated later).
            packet.main_table_latex = cls._build_scaffold_main_table(blueprint)
            packet.ablation_table_latex = cls._build_scaffold_ablation_table(blueprint)

        return packet

    @staticmethod
    def _build_main_table_latex(
        main_results: list[dict],
        comparison: dict,
        blueprint: dict,
    ) -> str:
        """Build a deterministic LaTeX main-results table from structured data.

        Returns empty string if data is insufficient.
        """
        if not main_results:
            return ""

        # Collect all metric names across all entries
        MAX_TABLE_COLS = 6
        all_metrics: list[str] = []
        seen: set[str] = set()
        for entry in main_results:
            for m in entry.get("metrics", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("metric_name", "")
                if name and name not in seen:
                    all_metrics.append(name)
                    seen.add(name)
        if not all_metrics:
            return ""
        # Cap columns to prevent table overflow
        if len(all_metrics) > MAX_TABLE_COLS:
            all_metrics = all_metrics[:MAX_TABLE_COLS]

        # Build rows: first from comparison_with_baselines, then main_results
        rows: list[tuple[str, bool, dict[str, str]]] = []  # (method, is_proposed, {metric: val_str})
        proposed_name = ""

        # Rows from main_results
        for entry in main_results:
            method = entry.get("method_name", "?")
            is_proposed = entry.get("is_proposed", False)
            if is_proposed:
                proposed_name = method
            metric_vals: dict[str, str] = {}
            for m in entry.get("metrics", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("metric_name", "")
                val = m.get("value")
                std = m.get("std")
                if val is not None:
                    val_str = f"{val}"
                    if std is not None:
                        val_str += f" $\\pm$ {std}"
                    metric_vals[name] = val_str
            rows.append((method, is_proposed, metric_vals))

        # Add baseline rows from comparison_with_baselines that aren't already in rows
        existing_methods = {r[0].lower() for r in rows}
        for method_name, method_metrics in comparison.items():
            if method_name.lower() in existing_methods:
                continue
            if method_name.lower() in ("our_method", "proposed", "ours"):
                continue
            if not isinstance(method_metrics, dict):
                continue
            metric_vals = {}
            for metric_name in all_metrics:
                val = method_metrics.get(metric_name)
                if val is not None:
                    metric_vals[metric_name] = str(val)
            if metric_vals:  # only add if has any values
                rows.append((method_name, False, metric_vals))

        if len(rows) < 1:
            return ""

        # Sort: baselines first, proposed method last
        baseline_rows = [r for r in rows if not r[1]]
        proposed_rows = [r for r in rows if r[1]]
        sorted_rows = baseline_rows + proposed_rows

        # Build LaTeX
        n_metrics = len(all_metrics)
        col_spec = "@{}l" + "c" * n_metrics + "@{}"
        header_cells = " & ".join(all_metrics)
        use_resizebox = n_metrics >= 5

        lines = [
            "\\begin{table}[t!]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\caption{{Main experimental results. Best results are in \\textbf{{bold}}.}}",
            "\\label{tab:main_results}",
        ]
        if use_resizebox:
            lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Method & {header_cells} \\\\",
            "\\midrule",
        ])

        # Determine which metrics are lower-is-better
        _LOWER_KW = (
            "loss", "error", "perplexity", "mse", "mae", "rmse", "cer", "wer",
            "fid", "distance", "divergence", "latency", "regret",
            "miss_rate", "false_positive", "eer",
        )
        lower_is_better_metrics: set[str] = {
            mn for mn in all_metrics
            if any(kw in mn.lower().replace(" ", "_").replace("-", "_")
                   for kw in _LOWER_KW)
        }

        # Find best value per metric (for bolding)
        _NUM_RE = re.compile(r'[+-]?(?:\d+\.?\d*|\.\d+)')

        def _extract_leading_number(s: str) -> float | None:
            """Extract leading numeric value from a metric string like '87.58 +/- 2.99'."""
            m = _NUM_RE.match(s.strip())
            return float(m.group(0)) if m else None

        best_vals: dict[str, float] = {}
        for _, _, mv in sorted_rows:
            for metric_name in all_metrics:
                val_str = mv.get(metric_name, "")
                val_num = _extract_leading_number(val_str)
                if val_num is None:
                    continue
                lower = metric_name in lower_is_better_metrics
                if metric_name not in best_vals:
                    best_vals[metric_name] = val_num
                elif lower and val_num < best_vals[metric_name]:
                    best_vals[metric_name] = val_num
                elif not lower and val_num > best_vals[metric_name]:
                    best_vals[metric_name] = val_num

        for method, is_proposed, metric_vals in sorted_rows:
            cells = []
            for metric_name in all_metrics:
                val_str = metric_vals.get(metric_name, "--")
                # Bold best value
                val_num = _extract_leading_number(val_str)
                if val_num is not None and metric_name in best_vals:
                    if abs(val_num - best_vals[metric_name]) < 1e-9:
                        val_str = f"\\textbf{{{val_str}}}"
                cells.append(val_str)
            method_display = f"{_escape_latex_text(method)} (Ours)" if is_proposed else _escape_latex_text(method)
            lines.append(f"{method_display} & {' & '.join(cells)} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
        ])
        if use_resizebox:
            lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    # Methods moved to grounding_tables.py: _build_ablation_table_latex,
    # _build_scaffold_main_table, _build_scaffold_ablation_table,
    # _build_real_results_context,
    # _build_experiment_analysis_context, _build_baseline_comparison_context,
    # _build_grounding_status_context, _find_table_span, _verify_and_inject_tables,
    # _table_metrics_match, _build_figure_blocks, _resolve_figure_include,
    # _TOOL_SECTIONS

