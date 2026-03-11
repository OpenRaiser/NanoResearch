"""Evidence grounding: experiment normalization, tables, figure blocks."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ._types import GroundingPacket
from . import _escape_latex_text

logger = logging.getLogger(__name__)

class _GroundingMixin:
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
            blueprint.get("proposed_method", {}).get("name")
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

        lines = [
            "\\begin{table}[t!]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\caption{{Main experimental results. Best results are in \\textbf{{bold}}.}}",
            "\\label{tab:main_results}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Method & {header_cells} \\\\",
            "\\midrule",
        ]

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
            "\\end{table}",
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_ablation_table_latex(
        ablation_results: list[dict],
        blueprint: dict,
    ) -> str:
        """Build a deterministic LaTeX ablation table from structured data."""
        if not ablation_results:
            return ""

        # Collect metric names
        all_metrics: list[str] = []
        seen: set[str] = set()
        for entry in ablation_results:
            for m in entry.get("metrics", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("metric_name", "")
                if name and name not in seen:
                    all_metrics.append(name)
                    seen.add(name)
        if not all_metrics:
            return ""

        n_metrics = len(all_metrics)
        col_spec = "@{}l" + "c" * n_metrics + "@{}"
        header_cells = " & ".join(all_metrics)

        lines = [
            "\\begin{table}[t!]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\caption{Ablation study. Each row removes or replaces one component.}",
            "\\label{tab:ablation}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Variant & {header_cells} \\\\",
            "\\midrule",
        ]

        for entry in ablation_results:
            variant = _escape_latex_text(entry.get("variant_name", "?"))
            cells = []
            for metric_name in all_metrics:
                val_str = "--"
                for m in entry.get("metrics", []):
                    if isinstance(m, dict) and m.get("metric_name") == metric_name:
                        val = m.get("value")
                        if val is not None:
                            val_str = str(val)
                        break
                cells.append(val_str)
            lines.append(f"{variant} & {' & '.join(cells)} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_scaffold_main_table(blueprint: dict) -> str:
        """Build a table scaffold from blueprint when no real results exist.

        The scaffold contains baseline method names and metric columns
        extracted from the blueprint.  Cell values are left as ``--``
        for baselines (LLM will fill from literature) and the proposed
        method row is clearly marked.  This ensures the Experiments
        section always has Table 1 structure even without real data.
        """
        baselines = blueprint.get("baselines", [])
        if not isinstance(baselines, list):
            baselines = []
        metrics_spec = blueprint.get("metrics", [])
        if not isinstance(metrics_spec, list):
            metrics_spec = []

        # Extract metric names from blueprint
        metric_names: list[str] = []
        for m in metrics_spec:
            if isinstance(m, dict):
                name = m.get("name", "") or m.get("metric_name", "")
            elif isinstance(m, str):
                name = m
            else:
                continue
            if name and name not in metric_names:
                metric_names.append(name)

        if not metric_names:
            # Fallback — can't build meaningful table without metric columns
            return ""

        # Baseline names
        baseline_names: list[str] = []
        for b in baselines:
            if isinstance(b, dict):
                name = b.get("name", "") or b.get("method", "")
            elif isinstance(b, str):
                name = b
            else:
                continue
            if name:
                baseline_names.append(name)

        if not baseline_names:
            baseline_names = ["Baseline 1", "Baseline 2"]

        # Proposed method name
        proposed = (
            blueprint.get("proposed_method", {}).get("name", "")
            or blueprint.get("method_name", "")
            or "Ours"
        )

        n = len(metric_names)
        col_spec = "@{}l" + "c" * n + "@{}"
        header = " & ".join(metric_names)

        lines = [
            "\\begin{table}[t!]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\caption{Main experimental results. Best results are in \\textbf{bold}. "
            "Results for the proposed method are pending due to execution issues.}",
            "\\label{tab:main_results}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Method & {header} \\\\",
            "\\midrule",
        ]

        for bname in baseline_names:
            cells = " & ".join(["--"] * n)
            lines.append(f"{_escape_latex_text(bname)} & {cells} \\\\")

        lines.append("\\midrule")
        cells = " & ".join(["--"] * n)
        lines.append(f"{_escape_latex_text(proposed)} (Ours) & {cells} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_scaffold_ablation_table(blueprint: dict) -> str:
        """Build an ablation table scaffold from blueprint (no real data).

        Rows are derived from blueprint ``contributions`` or ``components``.
        """
        metrics_spec = blueprint.get("metrics", [])
        if not isinstance(metrics_spec, list):
            metrics_spec = []

        metric_names: list[str] = []
        for m in metrics_spec:
            if isinstance(m, dict):
                name = m.get("name", "") or m.get("metric_name", "")
            elif isinstance(m, str):
                name = m
            else:
                continue
            if name and name not in metric_names:
                metric_names.append(name)

        if not metric_names:
            return ""

        # Derive ablation variants from contributions / components
        contributions = blueprint.get("contributions", [])
        if not isinstance(contributions, list):
            contributions = []

        variants: list[str] = []
        for c in contributions:
            if isinstance(c, str) and len(c) < 60:
                variants.append(f"w/o {c}")
            elif isinstance(c, dict):
                name = c.get("name", "") or c.get("component", "")
                if name:
                    variants.append(f"w/o {name}")

        if not variants:
            variants = ["w/o Component A", "w/o Component B"]

        proposed = (
            blueprint.get("proposed_method", {}).get("name", "")
            or blueprint.get("method_name", "")
            or "Full Model"
        )

        n = len(metric_names)
        col_spec = "@{}l" + "c" * n + "@{}"
        header = " & ".join(metric_names)

        lines = [
            "\\begin{table}[t!]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\caption{Ablation study. Each row removes one component. "
            "Results are pending due to execution issues.}",
            "\\label{tab:ablation}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\toprule",
            f"Variant & {header} \\\\",
            "\\midrule",
        ]

        cells = " & ".join(["--"] * n)
        for v in variants[:5]:
            lines.append(f"{_escape_latex_text(v)} & {cells} \\\\")

        lines.append("\\midrule")
        lines.append(f"{_escape_latex_text(proposed)} (Full) & {cells} \\\\")

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_real_results_context(
        experiment_results: dict, experiment_status: str
    ) -> str:
        """Build context block from real experiment results for writing prompts."""
        main_results = experiment_results.get("main_results", [])
        if not isinstance(main_results, list):
            main_results = []
        normalized_status = (experiment_status or "").lower()
        has_real = bool(
            experiment_results
            and normalized_status not in ("pending", "failed", "error", "unknown")
            and main_results
        )

        if has_real:
            lines = [
                "=== REAL EXPERIMENT RESULTS (MUST USE THESE EXACT NUMBERS) ===",
                "The following numbers come from actual experiments. Use them EXACTLY",
                "in tables, text, and analysis. Do NOT round, adjust, or fabricate.",
                "",
            ]
            for entry in main_results:
                if not isinstance(entry, dict):
                    continue
                method = entry.get("method_name", "?")
                dataset = entry.get("dataset", "?")
                is_proposed = entry.get("is_proposed", False)
                tag = " [PROPOSED]" if is_proposed else ""
                for metric in entry.get("metrics", []):
                    if not isinstance(metric, dict):
                        continue
                    val = metric.get("value", "?")
                    std = metric.get("std")
                    std_str = f" $\\pm$ {std}" if std is not None else ""
                    lines.append(
                        f"  {method} on {dataset}: "
                        f"{metric.get('metric_name', '?')} = {val}{std_str}{tag}"
                    )

            ablation = experiment_results.get("ablation_results", [])
            if not isinstance(ablation, list):
                ablation = []
            if ablation:
                lines.append("")
                lines.append("--- Ablation Results (real) ---")
                for entry in ablation:
                    if not isinstance(entry, dict):
                        continue
                    variant = entry.get("variant_name", "?")
                    for metric in entry.get("metrics", []):
                        if not isinstance(metric, dict):
                            continue
                        val = metric.get("value", "?")
                        lines.append(
                            f"  {variant}: {metric.get('metric_name', '?')} = {val}"
                        )

            lines.append("=== END REAL EXPERIMENT RESULTS ===")
            return "\n".join(lines)
        else:
            return (
                "=== EXPERIMENT RESULTS: NOT AVAILABLE ===\n"
                "The proposed method's experiment did not produce results due to execution issues.\n\n"
                "RULES FOR EXPERIMENTS SECTION:\n"
                "- Do NOT fabricate results for the PROPOSED METHOD. Use '--' in its table cells.\n"
                "- For BASELINE methods, you SHOULD fill in numbers from their original papers\n"
                "  (cite the source). This is standard practice — reproducing published numbers.\n"
                "- Use the PRE-BUILT TABLE scaffold provided below. Fill baseline cells with\n"
                "  literature-reported numbers and keep proposed method cells as '--'.\n"
                "- Write a full Experiments section: datasets, metrics, baselines, setup,\n"
                "  implementation details, and analysis of baseline landscape.\n"
                "- Add one sentence noting that proposed method results are pending:\n"
                "  'Due to technical issues during execution, results for our method could not\n"
                "   be obtained in this version. We report baseline results from the literature\n"
                "   for reference.'\n"
                "- Do NOT skip the tables. Do NOT omit the ablation table.\n"
                "=== END EXPERIMENT RESULTS ==="
            )

    @staticmethod
    def _build_experiment_analysis_context(
        experiment_analysis: dict,
        experiment_summary: str,
        experiment_status: str,
    ) -> str:
        """Build a compact narrative summary from execution analysis artifacts."""
        if not experiment_analysis and not experiment_summary:
            return ""

        lines = [
            "=== EXPERIMENT ANALYSIS SUMMARY ===",
            f"Status: {experiment_status}",
        ]

        summary = str(experiment_analysis.get("summary", "")).strip()
        if summary:
            lines.append(f"Summary: {summary}")

        converged = experiment_analysis.get("converged")
        if converged is not None:
            lines.append(f"Converged: {converged}")

        final_metrics = experiment_analysis.get("final_metrics", {})
        if isinstance(final_metrics, dict) and final_metrics:
            lines.append("Final metrics snapshot:")
            for key, value in final_metrics.items():
                lines.append(f"- {key}: {value}")

        key_findings = experiment_analysis.get("key_findings", [])
        if isinstance(key_findings, list) and key_findings:
            lines.append("Key findings:")
            for item in key_findings[:6]:
                lines.append(f"- {item}")

        limitations = experiment_analysis.get("limitations", [])
        if isinstance(limitations, list) and limitations:
            lines.append("Limitations:")
            for item in limitations[:6]:
                lines.append(f"- {item}")

        training_dynamics = experiment_analysis.get("training_dynamics")
        if training_dynamics:
            lines.append(f"Training dynamics: {training_dynamics}")

        cleaned_summary = experiment_summary.strip()
        if cleaned_summary:
            lines.append("Markdown experiment summary:")
            lines.append(cleaned_summary[:4000])

        lines.append("=== END EXPERIMENT ANALYSIS SUMMARY ===")
        return "\n".join(lines)

    @staticmethod
    def _build_baseline_comparison_context(grounding: "GroundingPacket | None") -> str:
        """Build context block from comparison_with_baselines analysis data."""
        if not grounding or not grounding.comparison_with_baselines:
            return ""
        comp = grounding.comparison_with_baselines
        lines = [
            "=== BASELINE COMPARISON (from experiment analysis) ===",
            "The following comparison data was extracted from actual experiment analysis.",
            "Use these numbers for comparison tables and discussion.",
            "",
        ]
        for method_name, metrics in comp.items():
            if not isinstance(metrics, dict):
                continue
            tag = " [PROPOSED]" if method_name.lower() in ("our_method", "proposed", "ours") else ""
            metric_strs = [f"{k}={v}" for k, v in metrics.items() if v is not None]
            if metric_strs:
                lines.append(f"  {method_name}{tag}: {', '.join(metric_strs)}")
        lines.append("=== END BASELINE COMPARISON ===")
        return "\n".join(lines)

    @staticmethod
    def _build_grounding_status_context(grounding: "GroundingPacket | None") -> str:
        """Build a brief context block informing the LLM about evidence completeness."""
        if not grounding:
            return ""
        completeness_desc = {
            "full": "FULL — complete experiment results are available. Use exact numbers.",
            "partial": "PARTIAL — experiment ran but did not fully converge. Use available numbers with caveats.",
            "quick_eval": "QUICK-EVAL ONLY — results are from a shortened evaluation run. "
                          "Use these numbers but note they may not reflect full training.",
            "none": "NONE — no experiment results available. Do NOT fabricate any numbers.",
        }
        desc = completeness_desc.get(grounding.result_completeness, "UNKNOWN")
        lines = [
            f"=== RESULT COMPLETENESS: {grounding.result_completeness.upper()} ===",
            desc,
        ]
        if grounding.evidence_gaps:
            lines.append("Evidence gaps:")
            for gap in grounding.evidence_gaps:
                lines.append(f"  - {gap}")
        lines.append("=== END RESULT COMPLETENESS ===")
        return "\n".join(lines)

    # ---- figure/table blocks ------------------------------------------------

    def _verify_and_inject_tables(
        self,
        content: str,
        grounding: GroundingPacket,
        heading: str,
    ) -> str:
        """Verify Experiments section has correct tables; inject if missing.

        If the LLM omitted the main results or ablation table, or built them
        with wrong numbers, replace/inject the deterministic pre-built versions.
        """
        has_main_table = bool(re.search(
            r'\\label\{tab:main_results\}', content
        ))
        has_ablation_table = bool(re.search(
            r'\\label\{tab:ablation\}', content
        ))

        if not has_main_table and grounding.main_table_latex:
            self.log(f"  {heading}: LLM omitted main results table, injecting pre-built")
            # Find a good insertion point — after first mention of "main results"
            # or after first paragraph
            insert_match = re.search(
                r'(?:main results|overall performance|comparison)',
                content, re.IGNORECASE,
            )
            if insert_match:
                # Insert after the paragraph containing the match
                para_end = content.find('\n\n', insert_match.end())
                if para_end == -1:
                    para_end = len(content)
                content = (
                    content[:para_end]
                    + "\n\n" + grounding.main_table_latex + "\n"
                    + content[para_end:]
                )
            else:
                # Append at end
                content += "\n\n" + grounding.main_table_latex

        if not has_ablation_table and grounding.ablation_table_latex:
            self.log(f"  {heading}: LLM omitted ablation table, injecting pre-built")
            insert_match = re.search(
                r'(?:ablation|component analysis)',
                content, re.IGNORECASE,
            )
            if insert_match:
                para_end = content.find('\n\n', insert_match.end())
                if para_end == -1:
                    para_end = len(content)
                content = (
                    content[:para_end]
                    + "\n\n" + grounding.ablation_table_latex + "\n"
                    + content[para_end:]
                )
            else:
                content += "\n\n" + grounding.ablation_table_latex

        return content

    def _build_figure_blocks(self, blueprint: dict, figure_output: dict | None = None) -> dict[str, str]:
        """Pre-build LaTeX figure/table blocks to embed inline.

        Dynamically builds blocks from whatever figures the FigureAgent produced.
        Falls back to scanning the figures/ directory when figure_output is
        not available (BUG-2 fix: no more hardcoded filenames).
        """
        blocks: dict[str, str] = {}
        figures = (figure_output or {}).get("figures", {})

        # Resolve figures directory for file-existence checks (BUG-1 fix)
        figures_dir = self.workspace.path / "figures" if hasattr(self, "workspace") else None

        if figures:
            # Dynamic: iterate over all figures produced by the FigureAgent.
            # Each figure gets exactly ONE key (label_suffix) — no aliases,
            # to prevent the same block being placed twice.
            for fig_key, fig_data in figures.items():
                if "error" in fig_data and "png_path" not in fig_data:
                    logger.warning(
                        "Skipping failed figure %s: %s",
                        fig_key, fig_data.get("error", "unknown error"),
                    )
                    continue  # skip failed figures with no output

                caption = _escape_latex_text(fig_data.get("caption", f"Figure: {fig_key}"))
                # Derive a LaTeX-friendly label from fig_key
                # e.g., "fig1_architecture" -> "fig:architecture"
                #        "fig2_results"     -> "fig:results"
                parts = fig_key.split("_", 1)
                label_suffix = parts[1] if len(parts) > 1 else fig_key
                label = f"fig:{label_suffix}"

                # BUG-1 fix: verify file existence before choosing format.
                # Prefer PDF > PNG > JPG, but only if the file actually exists.
                include_name = self._resolve_figure_include(
                    fig_key, fig_data, figures_dir,
                )
                if include_name is None:
                    logger.warning(
                        "Figure %s: no valid file found on disk, skipping", fig_key,
                    )
                    continue

                # Architecture/framework figures use full width;
                # result/ablation/chart figures use 0.75 width for better layout
                _full_width_kws = ("overview", "framework", "pipeline",
                                   "architecture", "model", "workflow", "diagram")
                suffix_lower = label_suffix.lower()
                if any(kw in suffix_lower for kw in _full_width_kws):
                    fig_width = r"\textwidth"
                else:
                    fig_width = r"0.75\textwidth"

                block = (
                    "\\begin{figure}[t!]\n"
                    "\\centering\n"
                    f"\\includegraphics[width={fig_width}, "
                    f"height=0.32\\textheight, keepaspectratio]"
                    f"{{{include_name}}}\n"
                    f"\\caption{{{caption}}}\n"
                    f"\\label{{{label}}}\n"
                    "\\end{figure}"
                )

                blocks[label_suffix] = block
        else:
            # BUG-2 fix: scan actual figures/ directory instead of
            # hardcoding 3 filenames that may not exist.
            if figures_dir and figures_dir.exists():
                for img in sorted(figures_dir.iterdir()):
                    if img.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg"):
                        stem = img.stem
                        readable = stem.replace("_", " ").replace("-", " ").title()
                        blocks[stem] = (
                            "\\begin{figure}[t!]\n"
                            "\\centering\n"
                            f"\\includegraphics[width=0.75\\textwidth]{{{img.name}}}\n"
                            f"\\caption{{{_escape_latex_text(readable)}.}}\n"
                            f"\\label{{fig:{stem}}}\n"
                            "\\end{figure}"
                        )
            if not blocks:
                logger.warning("No figures available — paper will have no figure blocks")

        return blocks

    @staticmethod
    def _resolve_figure_include(
        fig_key: str, fig_data: dict, figures_dir: Path | None,
    ) -> str | None:
        """Resolve the actual filename to use in \\includegraphics.

        BUG-1 fix: checks physical file existence, preferring PDF > PNG > JPG.
        Returns None if no valid file is found.
        """
        # Candidate paths ordered by preference
        candidates = [
            (fig_data.get("pdf_path"), f"{fig_key}.pdf"),
            (fig_data.get("png_path"), f"{fig_key}.png"),
            (None, f"{fig_key}.jpg"),
        ]
        for meta_path, default_name in candidates:
            # 1. Check metadata path
            if meta_path:
                p = Path(meta_path)
                if p.exists():
                    return p.name
            # 2. Check figures_dir
            if figures_dir and (figures_dir / default_name).exists():
                return default_name
        return None

    # ---- tool-augmented search for writing -----------------------------------

    # Sections that benefit from tool-augmented search during writing
    _TOOL_SECTIONS = frozenset({"Introduction", "Related Work", "Method", "Experiments"})

