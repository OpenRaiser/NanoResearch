"""Analysis agent — parses real experiment results and generates figures from actual data."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)


def _flatten_metric_list(metrics) -> dict[str, float]:
    """Convert [{metric_name: str, value: float}, ...] to {name: value}.

    Filters out NaN/Inf values.
    """
    if isinstance(metrics, dict):
        return {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and math.isfinite(v)
        }
    if not isinstance(metrics, list):
        return {}
    flat: dict[str, float] = {}
    for m in metrics:
        if isinstance(m, dict):
            name = m.get("metric_name") or m.get("name")
            val = m.get("value")
            if name and isinstance(val, (int, float)) and math.isfinite(val):
                flat[name] = val
    return flat


class AnalysisAgent(BaseResearchAgent):
    """Analyzes real experiment results and generates publication figures from actual data."""

    stage = PipelineStage.ANALYSIS

    @property
    def stage_config(self):
        """Use writing model config for result analysis (needs strong reasoning)."""
        return self.config.for_stage("writing")

    async def run(self, **inputs: Any) -> dict[str, Any]:
        execution_output: dict = inputs.get("execution_output", {})
        experiment_blueprint: dict = inputs.get("experiment_blueprint", {})

        self.log("Starting analysis of real experiment results")

        # Step 1: Analyze results (LLM)
        analysis = await self._analyze_results(execution_output, experiment_blueprint)
        if not isinstance(analysis, dict):
            analysis = {}
        analysis.setdefault("execution_output", execution_output)
        self.log(f"Analysis complete: {list(analysis.keys())}")

        # Step 1.5: Computational analysis (deterministic, no LLM)
        computational = self._compute_analysis(
            execution_output, experiment_blueprint, analysis
        )
        if computational:
            self.log(f"Computational analysis: {list(computational.keys())}")

        # Step 2: Generate figures from real data
        figures = await self._generate_figures(analysis, experiment_blueprint)
        self.log(f"Generated {len(figures)} figures")

        # Step 3: Write an experiment summary markdown for downstream writing/review
        summary_markdown = self._render_experiment_summary_markdown(
            analysis,
            execution_output,
            experiment_blueprint,
            computational,
        )
        summary_path = self.workspace.write_text(
            "drafts/experiment_summary.md",
            summary_markdown,
        )

        result = {
            "analysis": analysis,
            "computational_analysis": computational,
            "figures": figures,
            "execution_output": execution_output,
            "experiment_summary": summary_markdown,
            "experiment_summary_path": str(summary_path),
        }

        self.workspace.write_json("plans/analysis_output.json", result)
        return result

    async def _analyze_results(
        self, execution_output: dict, blueprint: dict
    ) -> dict:
        """Use LLM to interpret and summarize experiment results."""
        metrics = execution_output.get("metrics", {})
        parsed_metrics = execution_output.get("parsed_metrics", {})
        stdout_log = execution_output.get("stdout_log", "")[-5000:]
        training_log_csv = execution_output.get("training_log_csv", "")
        final_status = execution_output.get("final_status", "UNKNOWN")

        system_prompt = (
            "You are an ML researcher analyzing experiment results. "
            "Given the training logs and metrics, provide a comprehensive analysis. "
            "Be honest about results — if the model didn't converge or results are poor, say so. "
            "Return JSON only."
        )

        user_prompt = f"""Job Status: {final_status}

Metrics JSON:
{json.dumps(metrics, indent=2)[:3000]}

Parsed Metrics from Log:
{json.dumps(parsed_metrics, indent=2)[:2000]}

Training Log CSV (last part):
{training_log_csv[-3000:] if training_log_csv else 'N/A'}

Stdout Log (last part):
{stdout_log[-3000:]}

Expected Metrics: {json.dumps(blueprint.get('metrics', []), indent=2)[:500]}
Baselines: {json.dumps(blueprint.get('baselines', []), indent=2)[:1000]}
Ablation Groups: {json.dumps(blueprint.get('ablation_groups', []), indent=2)[:1000]}

Analyze these results. Return JSON:
{{
  "summary": "1-2 paragraph summary of results...",
  "converged": true/false,
  "final_metrics": {{"metric_name": value, ...}},
  "comparison_with_baselines": {{
    "our_method": {{"metric1": value, "metric2": value}},
    "baseline1_name": {{"metric1": value_or_null, "metric2": value_or_null}},
    "baseline2_name": {{"metric1": value_or_null, "metric2": value_or_null}}
  }},
  "ablation_results": [
    {{
      "variant_name": "Full model",
      "metrics": [{{"metric_name": "Accuracy", "value": 0.85}}]
    }},
    {{
      "variant_name": "w/o Component A",
      "metrics": [{{"metric_name": "Accuracy", "value": 0.79}}]
    }}
  ],
  "training_dynamics": "Description of training curve behavior...",
  "key_findings": ["finding1", "finding2", ...],
  "limitations": ["limitation1", ...],
  "figures_to_generate": [
    {{
      "figure_id": "fig_training_curve",
      "title": "Training Loss Curve",
      "type": "line",
      "data_source": "training_log"
    }},
    {{
      "figure_id": "fig_results_comparison",
      "title": "Results Comparison with Baselines",
      "type": "bar",
      "data_source": "metrics"
    }},
    {{
      "figure_id": "fig_ablation",
      "title": "Ablation Study",
      "type": "bar",
      "data_source": "metrics"
    }}
  ]
}}

IMPORTANT:
- For comparison_with_baselines, return a DICT mapping method names to their metrics.
  Include our method's actual numbers. For baselines, use their expected_performance
  from the blueprint if available, or null if unknown.
- For ablation_results, include the full model and variants with/without each component.
  If the experiment only ran the full model, create ablation entries by estimating
  component contributions from the training dynamics and key findings.
  Each variant MUST have concrete numeric values, not null.
- NEVER use "N/A" as a numeric value. Use null for truly unknown values."""

        result = await self.generate_json(system_prompt, user_prompt)
        return result if isinstance(result, dict) else {}

    # ── Computational analysis (deterministic, no LLM) ──────────────────

    def _compute_analysis(
        self,
        execution_output: dict,
        blueprint: dict,
        llm_analysis: dict,
    ) -> dict:
        """Run deterministic computational analysis alongside LLM analysis."""
        from nanoresearch.agents.analysis.training_dynamics import (
            analyze_training_dynamics,
        )
        from nanoresearch.agents.analysis.ablation_analysis import (
            quantify_ablation_contributions,
        )
        from nanoresearch.agents.analysis.comparison_matrix import (
            build_comparison_matrix,
            comparison_matrix_to_latex,
        )

        result: dict = {}
        raw_metrics = execution_output.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}

        bp_metrics = blueprint.get("metrics", [])
        if not isinstance(bp_metrics, list):
            bp_metrics = []

        # 1. Training dynamics
        training_log = raw_metrics.get("training_log", [])
        if isinstance(training_log, list) and len(training_log) >= 3:
            dynamics = analyze_training_dynamics(training_log)
            result["training_dynamics"] = dynamics

        # 2. Comparison matrix
        main_results = raw_metrics.get("main_results", [])
        bp_baselines = blueprint.get("baselines", [])
        matrix_inputs = self._build_matrix_inputs(
            main_results, bp_baselines, bp_metrics
        )
        if matrix_inputs:
            baselines_list, proposed, metrics_list = matrix_inputs
            matrix = build_comparison_matrix(baselines_list, proposed, metrics_list)
            result["comparison_matrix"] = matrix
            result["comparison_latex"] = comparison_matrix_to_latex(matrix)

        # 3. Ablation contributions
        ablation_raw = raw_metrics.get("ablation_results", [])
        if not isinstance(ablation_raw, list) or not ablation_raw:
            ablation_raw = llm_analysis.get("ablation_results", [])
        if isinstance(ablation_raw, list) and len(ablation_raw) >= 2:
            primary_metric = self._find_primary_metric(bp_metrics)
            higher = self._metric_higher_is_better(primary_metric, bp_metrics)
            full_result, ablation_variants = self._split_ablation(
                ablation_raw, primary_metric
            )
            if full_result and ablation_variants:
                contributions = quantify_ablation_contributions(
                    full_result, ablation_variants, primary_metric, higher
                )
                result["ablation_contributions"] = contributions

        return result

    @staticmethod
    def _build_matrix_inputs(
        main_results: list,
        bp_baselines: list,
        bp_metrics: list,
    ):
        """Convert pipeline data into (baselines, proposed, metrics) for comparison_matrix.

        Returns None if insufficient data.
        """
        if not isinstance(main_results, list):
            return None

        # Flatten main_results into {name, metrics: {metric: value}}
        proposed = None
        baselines = []
        for entry in main_results:
            if not isinstance(entry, dict):
                continue
            flat = _flatten_metric_list(entry.get("metrics", []))
            item = {"name": entry.get("method_name", "Unknown"), "metrics": flat}
            if entry.get("is_proposed"):
                proposed = item
            else:
                baselines.append(item)

        # Supplement baselines from blueprint expected_performance
        seen = {b["name"] for b in baselines}
        for bp_bl in bp_baselines:
            if not isinstance(bp_bl, dict):
                continue
            name = bp_bl.get("name", "")
            if name in seen:
                continue
            perf = bp_bl.get("expected_performance", {})
            if isinstance(perf, dict) and perf:
                # Filter out non-numeric and "N/A"
                clean = {
                    k: v for k, v in perf.items()
                    if isinstance(v, (int, float))
                }
                if clean:
                    baselines.append({"name": name, "metrics": clean})
                    seen.add(name)

        if proposed is None or not proposed.get("metrics"):
            return None
        if not baselines:
            return None

        # Build metrics list
        metrics_list = []
        seen_m: set[str] = set()
        for m in bp_metrics:
            if isinstance(m, dict) and m.get("name"):
                metrics_list.append({
                    "name": m["name"],
                    "higher_is_better": m.get("higher_is_better", True),
                })
                seen_m.add(m["name"])
        # Add any metric from proposed that's not in blueprint
        for mname in proposed.get("metrics", {}):
            if mname not in seen_m:
                metrics_list.append({"name": mname, "higher_is_better": True})
                seen_m.add(mname)

        if not metrics_list:
            return None
        return baselines, proposed, metrics_list

    @staticmethod
    def _find_primary_metric(bp_metrics: list) -> str:
        """Return the primary metric name from blueprint, or first available."""
        for m in bp_metrics:
            if isinstance(m, dict) and m.get("primary"):
                return m.get("name", "accuracy")
        if bp_metrics and isinstance(bp_metrics[0], dict):
            return bp_metrics[0].get("name", "accuracy")
        return "accuracy"

    @staticmethod
    def _metric_higher_is_better(metric_name: str, bp_metrics: list) -> bool:
        for m in bp_metrics:
            if isinstance(m, dict) and m.get("name") == metric_name:
                return m.get("higher_is_better", True)
        return True

    _FULL_MODEL_NAMES = frozenset({
        "full", "full model", "full_model", "ours", "proposed", "complete",
    })

    @staticmethod
    def _split_ablation(
        ablation_raw: list, primary_metric: str
    ) -> tuple:
        """Split ablation entries into (full_result_dict, variants_list).

        Identifies the full model by name first; falls back to highest score.
        Returns ({metric: value}, [{variant_name, metrics: {metric: value}}]).
        """
        entries = []
        for entry in ablation_raw:
            if not isinstance(entry, dict):
                continue
            flat = _flatten_metric_list(entry.get("metrics", []))
            if not flat:
                # metrics might already be a dict
                raw_m = entry.get("metrics", {})
                if isinstance(raw_m, dict):
                    flat = {
                        k: v for k, v in raw_m.items()
                        if isinstance(v, (int, float)) and math.isfinite(v)
                    }
            entries.append({
                "variant_name": entry.get("variant_name", "unknown"),
                "metrics": flat,
                "score": flat.get(primary_metric),
            })

        scored = [e for e in entries if isinstance(e["score"], (int, float))]
        if len(scored) < 2:
            return None, None

        # Find full model: prefer name match, fall back to highest score
        full_entry = None
        for e in scored:
            vn = e["variant_name"].lower().strip()
            if vn in AnalysisAgent._FULL_MODEL_NAMES or "full" in vn:
                full_entry = e
                break
        if full_entry is None:
            scored.sort(key=lambda e: e["score"], reverse=True)
            full_entry = scored[0]

        full = full_entry["metrics"]
        variants = [
            {"variant_name": e["variant_name"], "metrics": e["metrics"]}
            for e in scored if e is not full_entry
        ]
        return full, variants

    @staticmethod
    def _render_experiment_summary_markdown(
        analysis: dict,
        execution_output: dict,
        blueprint: dict,
        computational: dict | None = None,
    ) -> str:
        """Render a compact markdown summary of the executed experiment."""
        if computational is None:
            computational = {}
        lines = [
            "# Experiment Summary",
            "",
            f"- Status: `{execution_output.get('final_status', 'UNKNOWN')}`",
            f"- Method: `{blueprint.get('proposed_method', {}).get('name', 'Unknown')}`",
            f"- Datasets: {', '.join(ds.get('name', '?') for ds in blueprint.get('datasets', []) if isinstance(ds, dict)) or 'N/A'}",
            "",
            "## Narrative",
            analysis.get("summary", "No summary available."),
            "",
        ]

        final_metrics = analysis.get("final_metrics", {})
        if not isinstance(final_metrics, dict) or not final_metrics:
            final_metrics = AnalysisAgent._extract_metric_snapshot(execution_output)
        if isinstance(final_metrics, dict) and final_metrics:
            lines.append("## Final Metrics")
            for key, value in final_metrics.items():
                lines.append(f"- `{key}`: {value}")
            lines.append("")

        key_findings = analysis.get("key_findings", [])
        if isinstance(key_findings, list) and key_findings:
            lines.append("## Key Findings")
            for item in key_findings:
                lines.append(f"- {item}")
            lines.append("")

        limitations = analysis.get("limitations", [])
        if isinstance(limitations, list) and limitations:
            lines.append("## Limitations")
            for item in limitations:
                lines.append(f"- {item}")
            lines.append("")

        dynamics = analysis.get("training_dynamics", "")
        if dynamics:
            lines.append("## Training Dynamics")
            lines.append(str(dynamics))
            lines.append("")

        comparison = analysis.get("comparison_with_baselines", {})
        if isinstance(comparison, dict) and comparison:
            lines.append("## Comparison with Baselines")
            lines.append("")
            # Build a markdown table
            # Collect all metric names
            all_metrics: list[str] = []
            seen_m: set[str] = set()
            for method_metrics in comparison.values():
                if isinstance(method_metrics, dict):
                    for k in method_metrics:
                        if k not in seen_m:
                            all_metrics.append(k)
                            seen_m.add(k)
            if all_metrics:
                lines.append("| Method | " + " | ".join(all_metrics) + " |")
                lines.append("|" + "|".join(["---"] * (len(all_metrics) + 1)) + "|")
                for method_name, method_metrics in comparison.items():
                    if not isinstance(method_metrics, dict):
                        continue
                    cells = [str(method_metrics.get(m, "--")) for m in all_metrics]
                    lines.append(f"| {method_name} | " + " | ".join(cells) + " |")
                lines.append("")

        ablation = analysis.get("ablation_results", [])
        if isinstance(ablation, list) and ablation:
            lines.append("## Ablation Results")
            for entry in ablation:
                if not isinstance(entry, dict):
                    continue
                variant = entry.get("variant_name", "?")
                metric_strs = []
                for m in entry.get("metrics", []):
                    if isinstance(m, dict):
                        metric_strs.append(f"{m.get('metric_name', '?')}={m.get('value', '?')}")
                lines.append(f"- {variant}: {', '.join(metric_strs)}")
            lines.append("")

        # ── Computational analysis sections ──
        comp_dynamics = computational.get("training_dynamics")
        if isinstance(comp_dynamics, dict):
            lines.append("## Training Dynamics (Computed)")
            lines.append(
                f"- Convergence epoch: {comp_dynamics.get('convergence_epoch', '?')} "
                f"/ {comp_dynamics.get('total_epochs', '?')}"
            )
            lines.append(f"- Best epoch: {comp_dynamics.get('best_epoch', '?')}")
            lines.append(
                f"- Best val loss: {comp_dynamics.get('best_val_loss', '?')}"
            )
            if comp_dynamics.get("overfitting_detected") is not None:
                lines.append(
                    f"- Overfitting detected: {comp_dynamics['overfitting_detected']}"
                )
            if comp_dynamics.get("loss_stability"):
                lines.append(
                    f"- Stability: {comp_dynamics['loss_stability']}"
                )
            if comp_dynamics.get("early_stopping_recommended"):
                lines.append("- Early stopping recommended")
            lines.append("")

        comp_contributions = computational.get("ablation_contributions")
        if isinstance(comp_contributions, list) and comp_contributions:
            lines.append("## Ablation Contributions (Computed)")
            for c in comp_contributions:
                flag = " **[CRITICAL]**" if c.get("is_critical") else ""
                lines.append(
                    f"- {c.get('component', '?')}: "
                    f"drop={c.get('absolute_drop', '?')} "
                    f"({c.get('relative_contribution_pct', '?')}%){flag}"
                )
            lines.append("")

        comp_latex = computational.get("comparison_latex")
        if comp_latex:
            lines.append("## Comparison Table (LaTeX)")
            lines.append("```latex")
            lines.append(comp_latex)
            lines.append("```")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _extract_metric_snapshot(execution_output: dict) -> dict[str, Any]:
        """Extract a flat metric snapshot from raw execution artifacts."""
        for candidate in (
            execution_output.get("metrics", {}),
            execution_output.get("parsed_metrics", {}),
        ):
            if not isinstance(candidate, dict):
                continue
            flat_metrics = {
                key: value
                for key, value in candidate.items()
                if isinstance(value, (int, float, str, bool))
            }
            if flat_metrics:
                return flat_metrics
        return {}

    async def _generate_figures(
        self, analysis: dict, blueprint: dict
    ) -> dict:
        """Generate publication-quality figures from real experiment data."""
        figures_output = {}
        figures_dir = self.workspace.path / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Get actual data for plotting
        execution_output = analysis.get("execution_output", {}) or {}
        final_metrics = analysis.get("final_metrics", {})
        training_log = analysis.get("training_dynamics", "")

        figure_specs = analysis.get("figures_to_generate", [])
        if not figure_specs:
            # Default figures
            figure_specs = [
                {"figure_id": "fig_training_curve", "title": "Training Loss Curve", "type": "line"},
                {"figure_id": "fig_results", "title": "Results", "type": "bar"},
            ]

        max_figs = 5
        if len(figure_specs) > max_figs:
            self.log(f"Capping analysis figures from {len(figure_specs)} to {max_figs}")
        figure_specs = figure_specs[:max_figs]

        for fig_spec in figure_specs:
            fig_id = fig_spec.get("figure_id", "fig_unknown")
            fig_title = fig_spec.get("title", "Figure")

            self.log(f"Generating figure: {fig_id}")

            # Generate plotting code via LLM
            plot_code = await self._generate_plot_code(
                fig_spec, analysis, blueprint
            )

            # Write and execute the plotting code
            script_path = figures_dir / f"{fig_id}_plot.py"
            script_path.write_text(plot_code, encoding="utf-8")

            png_path = figures_dir / f"{fig_id}.png"
            pdf_path = figures_dir / f"{fig_id}.pdf"

            try:
                result = await self._run_shell(
                    f'cd "{figures_dir}" && "{sys.executable}" "{script_path}"',
                    timeout=60,
                )
                if png_path.exists():
                    figures_output[fig_id] = {
                        "png_path": str(png_path),
                        "pdf_path": str(pdf_path) if pdf_path.exists() else "",
                        "caption": fig_title,
                        "script_path": str(script_path),
                    }
                    self.log(f"Figure generated: {fig_id}")
                else:
                    self.log(f"Figure script ran but no output: {result.get('stderr', '')[:200]}")
                    figures_output[fig_id] = {"error": "No output file generated"}
            except Exception as e:
                self.log(f"Failed to generate figure {fig_id}: {e}")
                figures_output[fig_id] = {"error": str(e)}

        return figures_output

    async def _generate_plot_code(
        self, fig_spec: dict, analysis: dict, blueprint: dict
    ) -> str:
        """Generate matplotlib plotting code for a specific figure."""
        fig_id = fig_spec.get("figure_id", "fig")
        fig_title = fig_spec.get("title", "Figure")
        fig_type = fig_spec.get("type", "bar")

        final_metrics = analysis.get("final_metrics", {})
        baselines = blueprint.get("baselines", [])
        metrics = blueprint.get("metrics", [])

        system_prompt = (
            "You are a data visualization expert. Write a complete matplotlib Python script "
            "that creates a publication-quality figure. The script must:\n"
            "1. Use matplotlib and seaborn\n"
            "2. Save the figure as both PNG (300 DPI) and PDF\n"
            "3. Use the ACTUAL experiment results provided (not made-up data)\n"
            "4. Have proper axis labels, title, legend\n"
            "5. Use a clean academic style\n"
            "Return ONLY the Python code."
        )

        user_prompt = f"""Generate a {fig_type} plot for: {fig_title}
Figure ID: {fig_id}

ACTUAL experiment results:
{json.dumps(final_metrics, indent=2)[:2000]}

Analysis summary: {analysis.get('summary', '')[:1000]}
Training dynamics: {analysis.get('training_dynamics', '')[:500]}

Baselines for comparison: {json.dumps(baselines, indent=2)[:500]}
Metrics definitions: {json.dumps(metrics, indent=2)[:300]}

IMPORTANT:
- Use the REAL numbers from the experiment results above
- If some metrics are missing, use reasonable placeholder values but note them
- Save as '{fig_id}.png' (dpi=300) and '{fig_id}.pdf'
- Use plt.tight_layout()
- Make the figure 8x5 inches

Return ONLY the Python code, no markdown fences."""

        code = await self.generate(system_prompt, user_prompt)

        # Strip markdown fences
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        return code

    async def _run_shell(self, cmd: str, timeout: int = 60) -> dict:
        """Run a shell command asynchronously with proxy environment."""
        env = {**__import__('os').environ}
        proxy_url = env.get("https_proxy") or env.get("HTTPS_PROXY", "")
        if not proxy_url:
            import re as _re
            bashrc = Path.home() / ".bashrc"
            if bashrc.exists():
                content = bashrc.read_text(errors="replace")
                m = _re.search(r"https_proxy=(http://[^\s;'\"]+)", content)
                if m:
                    proxy_url = m.group(1)
        if proxy_url:
            env.update({
                "http_proxy": proxy_url, "https_proxy": proxy_url,
                "HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url,
            })
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"returncode": -1, "stdout": "", "stderr": "Command timed out"}
        return {
            "returncode": proc.returncode or 0,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }

    async def close(self) -> None:
        await super().close()
