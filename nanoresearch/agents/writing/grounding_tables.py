"""Grounding tables mixin: table/figure block building, table verification."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ._types import GroundingPacket
from . import _escape_latex_text

logger = logging.getLogger(__name__)


class _GroundingTablesMixin:
    """Mixin — table/figure block building and table verification methods."""

    @staticmethod
    def _build_ablation_table_latex(
        ablation_results: list[dict],
        blueprint: dict,
    ) -> str:
        """Build a deterministic LaTeX ablation table from structured data."""
        if not ablation_results:
            return ""
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
            "\\begin{table}[t!]", "\\centering", "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            "\\caption{Ablation study. Each row removes or replaces one component.}",
            "\\label{tab:ablation}",
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            f"Variant & {header_cells} \\\\", "\\midrule",
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
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        return "\n".join(lines)

    @staticmethod
    def _build_scaffold_main_table(blueprint: dict) -> str:
        """Build a table scaffold from blueprint when no real results exist."""
        baselines = blueprint.get("baselines", [])
        if not isinstance(baselines, list):
            baselines = []
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
        # Cap at 6 metric columns to avoid table overflow
        MAX_TABLE_COLS = 6
        if len(metric_names) > MAX_TABLE_COLS:
            metric_names = metric_names[:MAX_TABLE_COLS]
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
        proposed = (
            (blueprint.get("proposed_method") or {}).get("name", "")
            or blueprint.get("method_name", "") or "Ours"
        )
        datasets = blueprint.get("datasets", [])
        dataset_names = []
        for d in datasets:
            if isinstance(d, dict):
                name = d.get("name", "")
            elif isinstance(d, str):
                name = d
            else:
                continue
            if name:
                dataset_names.append(name)
        dataset_str = ", ".join(dataset_names[:3]) if dataset_names else "the benchmark"
        n = len(metric_names)
        col_spec = "@{}l" + "c" * n + "@{}"
        header = " & ".join(metric_names)
        lines = [
            "\\begin{table}[t!]", "\\centering", "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\caption{{Main experimental results on {_escape_latex_text(dataset_str)}. "
            "Best results are in \\textbf{bold}. "
            "'--' indicates that the method was not evaluated in our experiments.}",
            "\\label{tab:main_results}",
        ]
        use_resizebox = n >= 5
        if use_resizebox:
            lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            f"Method & {header} \\\\", "\\midrule",
        ])
        for bname in baseline_names:
            cells = " & ".join(["--"] * n)
            lines.append(f"{_escape_latex_text(bname)} & {cells} \\\\")
        lines.append("\\midrule")
        cells = " & ".join(["--"] * n)
        lines.append(f"{_escape_latex_text(proposed)} (Ours) & {cells} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        if use_resizebox:
            lines.append("}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    @staticmethod
    def _build_scaffold_ablation_table(blueprint: dict) -> str:
        """Build an ablation table scaffold from blueprint (no real data)."""
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
        # Cap at 6 metric columns to avoid table overflow
        MAX_TABLE_COLS = 6
        if len(metric_names) > MAX_TABLE_COLS:
            metric_names = metric_names[:MAX_TABLE_COLS]
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
            (blueprint.get("proposed_method") or {}).get("name", "")
            or blueprint.get("method_name", "") or "Full Model"
        )
        datasets = blueprint.get("datasets", [])
        dataset_names = []
        for d in datasets:
            if isinstance(d, dict):
                name = d.get("name", "")
            elif isinstance(d, str):
                name = d
            else:
                continue
            if name:
                dataset_names.append(name)
        dataset_str = ", ".join(dataset_names[:3]) if dataset_names else "the benchmark"
        n = len(metric_names)
        col_spec = "@{}l" + "c" * n + "@{}"
        header = " & ".join(metric_names)
        lines = [
            "\\begin{table}[t!]", "\\centering", "\\small",
            "\\setlength{\\tabcolsep}{4pt}",
            f"\\caption{{Ablation study on {_escape_latex_text(dataset_str)}. Each row removes one component. "
            "Results are pending due to execution issues.}",
            "\\label{tab:ablation}",
        ]
        use_resizebox = n >= 5
        if use_resizebox:
            lines.append("\\resizebox{\\textwidth}{!}{%")
        lines.extend([
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            f"Variant & {header} \\\\", "\\midrule",
        ])
        cells = " & ".join(["--"] * n)
        for v in variants[:5]:
            lines.append(f"{_escape_latex_text(v)} & {cells} \\\\")
        lines.append("\\midrule")
        lines.append(f"{_escape_latex_text(proposed)} (Full) & {cells} \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        if use_resizebox:
            lines.append("}")
        lines.append("\\end{table}")
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
                "ABSOLUTE RULES FOR WRITING WITHOUT RESULTS:\n"
                "- Do NOT fabricate results for the PROPOSED METHOD. Use '--' in its table cells.\n"
                "- Do NOT write ANY specific numbers or quantitative claims about the proposed method.\n"
                "- For BASELINE methods, you MAY fill in numbers from their original papers (cite source).\n"
                "- Do NOT generate \\begin{table} environments yourself. Tables are auto-injected.\n"
                "  Reference them as Table~\\ref{tab:main_results} and Table~\\ref{tab:ablation}.\n"
                "- Write a full Experiments section: datasets, metrics, baselines, setup, implementation.\n"
                "- Include: 'Due to technical issues during execution, quantitative results for our "
                "method are not available in this version.'\n"
                "=== END EXPERIMENT RESULTS ==="
            )

    @staticmethod
    def _build_experiment_analysis_context(
        experiment_analysis: dict, experiment_summary: str, experiment_status: str,
    ) -> str:
        """Build a compact narrative summary from execution analysis artifacts."""
        if not experiment_analysis and not experiment_summary:
            return ""
        lines = ["=== EXPERIMENT ANALYSIS SUMMARY ===", f"Status: {experiment_status}"]
        summary = str(experiment_analysis.get("summary", "")).strip()
        if summary:
            lines.append(f"Summary: {summary}")
        converged = experiment_analysis.get("converged")
        if converged is not None:
            lines.append(f"Converged: {converged}")
        for section, key in [("Final metrics snapshot:", "final_metrics"),
                              ("Key findings:", "key_findings"),
                              ("Limitations:", "limitations")]:
            data = experiment_analysis.get(key, {} if key == "final_metrics" else [])
            if isinstance(data, dict) and data:
                lines.append(section)
                for k, v in data.items():
                    lines.append(f"- {k}: {v}")
            elif isinstance(data, list) and data:
                lines.append(section)
                for item in data[:6]:
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
        lines = ["=== BASELINE COMPARISON (from experiment analysis) ===",
                  "Use these numbers for comparison tables and discussion.", ""]
        for method_name, metrics in grounding.comparison_with_baselines.items():
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
            "full": "FULL -- complete experiment results are available. Use exact numbers.",
            "partial": (
                "PARTIAL -- experiment ran but did not fully converge. "
                "ONLY use numbers from REAL EXPERIMENT RESULTS provided below. "
                "For methods NOT run, use '--'. Do NOT use blueprint projected values. "
                "You MAY cite numbers from original published papers of baselines."
            ),
            "quick_eval": (
                "QUICK-EVAL ONLY -- results from a shortened evaluation run. "
                "Use these numbers but note they may not reflect full training. "
                "For methods NOT evaluated, use '--' in table cells."
            ),
            "none": (
                "NONE -- no experiment results available. "
                "ABSOLUTE BAN: Do NOT write ANY specific numbers or quantitative comparisons "
                "for the proposed method. Use ONLY qualitative language. "
                "For baselines, you MAY cite numbers from their original papers."
            ),
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

    @staticmethod
    def _find_table_span(content: str, label: str) -> tuple[int, int] | None:
        """Find span of ``\\begin{table}...\\end{table}`` enclosing *label*."""
        label_match = re.search(re.escape(label), content)
        if not label_match:
            return None
        before = content[:label_match.start()]
        tbl_start = before.rfind("\\begin{table}")
        if tbl_start < 0:
            return None
        after_match = re.search(r"\\end\{table\}", content[label_match.end():])
        if not after_match:
            return None
        tbl_end = label_match.end() + after_match.end()
        return (tbl_start, tbl_end)

    def _verify_and_inject_tables(self, content: str, grounding: GroundingPacket, heading: str) -> str:
        """Verify Experiments section has correct tables; inject if missing or wrong."""
        for label_key, table_latex, kw_pattern in [
            (r"\label{tab:main_results}", grounding.main_table_latex,
             r'(?:main results|overall performance|comparison)'),
            (r"\label{tab:ablation}", grounding.ablation_table_latex,
             r'(?:ablation|component analysis)'),
        ]:
            if not table_latex:
                continue
            span = self._find_table_span(content, label_key)
            if span:
                if label_key == r"\label{tab:main_results}":
                    llm_table = content[span[0]:span[1]]
                    if self._table_metrics_match(llm_table, grounding):
                        self.log(f"  {heading}: LLM main table metrics match grounding")
                        continue
                    self.log(f"  {heading}: LLM table metrics MISMATCH, replacing with pre-built")
                else:
                    self.log(f"  {heading}: replacing LLM ablation table with pre-built")
                content = content[:span[0]] + table_latex + content[span[1]:]
            else:
                self.log(f"  {heading}: LLM omitted table {label_key}, injecting pre-built")
                insert_match = re.search(kw_pattern, content, re.IGNORECASE)
                if insert_match:
                    para_end = content.find('\n\n', insert_match.end())
                    if para_end == -1:
                        para_end = len(content)
                    content = content[:para_end] + "\n\n" + table_latex + "\n" + content[para_end:]
                else:
                    content += "\n\n" + table_latex
        return content

    @staticmethod
    def _table_metrics_match(
        llm_table: str, grounding: GroundingPacket,
    ) -> bool:
        """Check if an LLM-generated table's metric columns match the grounding packet."""
        expected_metrics: set[str] = set()
        for entry in grounding.main_results:
            for m in entry.get("metrics", []):
                if isinstance(m, dict):
                    name = m.get("metric_name", "")
                    if name:
                        expected_metrics.add(name.lower().strip())

        if not expected_metrics:
            return True

        llm_table_lower = llm_table.lower()
        found = sum(1 for m in expected_metrics if m in llm_table_lower)
        threshold = max(1, len(expected_metrics) // 2)
        return found >= threshold

    def _build_figure_blocks(self, blueprint: dict, figure_output: dict | None = None) -> dict[str, str]:
        """Pre-build LaTeX figure/table blocks to embed inline."""
        blocks: dict[str, str] = {}
        figures = (figure_output or {}).get("figures", {})
        figures_dir = self.workspace.path / "figures" if hasattr(self, "workspace") else None
        _full_kws = ("overview", "framework", "pipeline", "architecture", "model", "workflow", "diagram")
        if figures:
            for fig_key, fig_data in figures.items():
                if "error" in fig_data and "png_path" not in fig_data:
                    logger.warning("Skipping failed figure %s: %s", fig_key, fig_data.get("error", "?"))
                    continue
                caption = _escape_latex_text(fig_data.get("caption", f"Figure: {fig_key}"))
                parts = fig_key.split("_", 1)
                label_suffix = parts[1] if len(parts) > 1 else fig_key
                include_name = self._resolve_figure_include(fig_key, fig_data, figures_dir)
                if include_name is None:
                    logger.warning("Figure %s: no valid file found on disk, skipping", fig_key)
                    continue
                fw = r"\textwidth" if any(kw in label_suffix.lower() for kw in _full_kws) else r"0.75\textwidth"
                blocks[label_suffix] = (
                    f"\\begin{{figure}}[t!]\n\\centering\n"
                    f"\\includegraphics[width={fw}, height=0.32\\textheight, keepaspectratio]{{{include_name}}}\n"
                    f"\\caption{{{caption}}}\n\\label{{fig:{label_suffix}}}\n\\end{{figure}}"
                )
        else:
            if figures_dir and figures_dir.exists():
                for img in sorted(figures_dir.iterdir()):
                    if img.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg"):
                        stem = img.stem
                        readable = stem.replace("_", " ").replace("-", " ").title()
                        blocks[stem] = (
                            f"\\begin{{figure}}[t!]\n\\centering\n"
                            f"\\includegraphics[width=0.75\\textwidth]{{{img.name}}}\n"
                            f"\\caption{{{_escape_latex_text(readable)}.}}\n"
                            f"\\label{{fig:{stem}}}\n\\end{{figure}}"
                        )
            if not blocks:
                logger.warning("No figures available -- paper will have no figure blocks")
        return blocks

    @staticmethod
    def _resolve_figure_include(
        fig_key: str, fig_data: dict, figures_dir: Path | None,
    ) -> str | None:
        """Resolve the actual filename to use in \\includegraphics."""
        candidates = [
            (fig_data.get("pdf_path"), f"{fig_key}.pdf"),
            (fig_data.get("png_path"), f"{fig_key}.png"),
            (None, f"{fig_key}.jpg"),
        ]
        for meta_path, default_name in candidates:
            if meta_path:
                p = Path(meta_path)
                if p.exists():
                    return p.name
            if figures_dir and (figures_dir / default_name).exists():
                return default_name
        return None

    _TOOL_SECTIONS = frozenset({"Introduction", "Related Work", "Method", "Experiments"})
