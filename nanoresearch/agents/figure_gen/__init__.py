"""Figure generation agent — dynamic figure planning + hybrid AI/code charts."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.manifest import PipelineStage

from ._constants import (  # noqa: F401 — re-exported
    AI_FIGURE_TEMPLATES,
    CHART_EXEC_TIMEOUT,
    FIGURE_PLAN_SYSTEM,
    _run_chart_subprocess,
    _clean_ai_image_caption,
)
from .evidence import _EvidenceMixin
from .ai_figure import _AiFigureMixin
from .code_figure import _CodeFigureMixin
from .trim import _TrimMixin
from .save_figure import _SaveFigureMixin

__all__ = ["FigureAgent"]

logger = logging.getLogger(__name__)


class FigureAgent(
    _EvidenceMixin,
    _AiFigureMixin,
    _CodeFigureMixin,
    _TrimMixin,
    _SaveFigureMixin,
    BaseResearchAgent,
):
    stage = PipelineStage.FIGURE_GEN

    async def run(self, **inputs: Any) -> dict[str, Any]:
        blueprint: dict = inputs.get("experiment_blueprint", {})
        if not blueprint:
            logger.warning("No experiment_blueprint provided; using empty dict")
            blueprint = {}
        ideation_output: dict = inputs.get("ideation_output", {})
        experiment_results: dict = inputs.get("experiment_results", {})
        experiment_status: str = inputs.get("experiment_status", "pending")
        survey_blueprint: dict = inputs.get("survey_blueprint", {})
        # ANALYSIS no longer generates figures; FIGURE_GEN owns all 4 figures.
        existing_figures: dict = {}

        # Detect survey mode: no experiment blueprint but survey_blueprint exists
        is_survey = bool(survey_blueprint) and not blueprint
        if is_survey:
            self.log("Survey mode detected: using survey-specific figure planning")
        else:
            self.log("Starting figure generation (dynamic planning + hybrid)")

        if experiment_results:
            self.log(f"Using REAL experiment results (status: {experiment_status})")
        else:
            self.log(f"No real experiment results available (status: {experiment_status})")

        # Build context and plan figures based on mode
        if is_survey:
            # Survey context from ideation_output
            theme_clusters = ideation_output.get("theme_clusters", [])
            key_challenges = ideation_output.get("key_challenges", [])
            future_directions = ideation_output.get("future_directions", [])
            survey_size = survey_blueprint.get("survey_size", "standard")

            survey_context = (
                f"Survey topic: {blueprint.get('title', 'Survey Paper')}\n"
                f"Survey size: {survey_size}\n"
                f"Theme clusters ({len(theme_clusters)}): {', '.join(theme_clusters[:10])}\n"
                f"Key challenges ({len(key_challenges)}): {', '.join(key_challenges[:5])}\n"
                f"Future directions ({len(future_directions)}): {', '.join(future_directions[:5])}\n"
            )
            evidence_block = self._build_evidence_block(
                ideation_output, blueprint, experiment_results, experiment_status
            )
            # Plan survey figures
            figure_plan = await self._plan_survey_figures(survey_context, evidence_block, survey_size)
            context = survey_context
            # Survey figure generation uses these with survey-specific defaults
            method_name = "Surveyed Methods"
            baselines = "N/A"
            metrics = "Benchmark metrics"
            ablation_groups = "N/A"
            primary_metric = "Performance"
        else:
            method = blueprint.get("proposed_method") or {}
            method_name = method.get("name", "Proposed Method")
            components = ", ".join(method.get("key_components") or [])
            baselines_list = blueprint.get("baselines") or []
            baselines = ", ".join(b.get("name", "") for b in baselines_list)
            metrics_list = blueprint.get("metrics") or []
            metrics = ", ".join(m.get("name", "") for m in metrics_list)
            ablation_groups = ", ".join(
                a.get("group_name", "") for a in (blueprint.get("ablation_groups") or [])
            )
            primary_metric = next(
                (m.get("name", "") for m in metrics_list if m.get("primary")),
                metrics_list[0].get("name", "Score") if metrics_list else "Score",
            )
            datasets = ", ".join(d.get("name", "") for d in (blueprint.get("datasets") or []))

            context = (
                f"Research title: {blueprint.get('title', '')}\n"
                f"Method: {method_name}\n"
                f"Components: {components}\n"
                f"Datasets: {datasets}\n"
                f"Baselines: {baselines}\n"
                f"Metrics: {metrics}\n"
                f"Ablation groups: {ablation_groups}\n"
                f"Primary metric: {primary_metric}\n"
            )

            # Build evidence block for chart prompts
            evidence_block = self._build_evidence_block(
                ideation_output, blueprint, experiment_results, experiment_status
            )

            # Step 1: LLM plans which figures to generate
            figure_plan = await self._plan_figures(context, evidence_block)
        self.log(f"Figure plan: {len(figure_plan)} figures")

        figure_results = {}

        # Step 2: Generate each planned figure (skip those already from ANALYSIS)
        # Build coroutines for all figures, then run concurrently
        async def _gen_one(fig_spec: dict) -> tuple[str, dict | None]:
            """Generate one figure; returns (fig_key, result_or_None)."""
            if not isinstance(fig_spec, dict) or "fig_key" not in fig_spec:
                logger.warning("Skipping invalid fig_spec: %s", fig_spec)
                return ("", None)
            fig_key = fig_spec["fig_key"]
            fig_type = fig_spec.get("fig_type", "code_chart")
            chart_type = fig_spec.get("chart_type", "grouped_bar")
            description = fig_spec.get("description", "")
            caption = fig_spec.get("caption", description)
            title = fig_spec.get("title", "")

            self.log(f"Generating {fig_key} ({fig_type}/{chart_type})")
            try:
                if fig_type == "ai_image":
                    ai_image_type = fig_spec.get("ai_image_type", "generic")
                    result = await self._generate_ai_figure(
                        context, fig_key, fig_key, description, ai_image_type,
                        caption=caption,
                    )
                else:
                    output_path = str(
                        self.workspace.path / "figures" / f"{fig_key}.png"
                    )
                    chart_prompt = self._build_chart_prompt(
                        chart_type=chart_type,
                        title=title,
                        description=description,
                        method_name=method_name,
                        baselines=baselines,
                        metrics=metrics,
                        ablation_groups=ablation_groups,
                        primary_metric=primary_metric,
                        evidence_block=evidence_block,
                        output_path=output_path,
                        context=context,
                    )
                    result = await self._generate_code_figure(
                        fig_key, output_path, chart_prompt, caption,
                    )
                return (fig_key, result)
            except Exception as exc:
                logger.warning(
                    "Figure generation failed for %s: %s",
                    fig_key, exc, exc_info=True,
                )
                self.log(f"Figure failed for {fig_key}, skipping: {exc}")
                return (fig_key, None)

        # Generate all planned figures
        new_specs = [spec for spec in figure_plan if isinstance(spec, dict)]

        results = await asyncio.gather(
            *(_gen_one(spec) for spec in new_specs),
            return_exceptions=False,
        )
        for fig_key, result in results:
            if fig_key and result is not None:
                figure_results[fig_key] = result

        self.log(f"Figure generation complete: {len(figure_results)} new figures")

        # All figures come from FIGURE_GEN only (exactly 4: 2 ai_image + 2 code_chart)
        merged = figure_results
        self.log(f"Total figures: {len(merged)}")

        # Persist output so that resume can reload it
        output = {"figures": merged}
        self.workspace.write_json("drafts/figure_output.json", output)

        return output

    # -----------------------------------------------------------------------
    # Figure planning
    # -----------------------------------------------------------------------

    async def _plan_figures(self, context: str, evidence_block: str) -> list[dict]:
        """Ask LLM to plan which figures to generate."""
        prompt = (
            f"Plan the figures for this research paper.\n\n"
            f"Research context:\n{context}\n\n"
            f"{evidence_block}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. First, identify the research domain (nlp/cv/llm/multimodal/general_ml)\n"
            f"2. Follow the domain-specific figure convention from the system prompt\n"
            f"3. Select EXACTLY 4 figures: 2 ai_image + 2 code_chart. No more, no fewer.\n"
            f"   - Fig 1 & Fig 2: fig_type='ai_image' (Gemini-generated conceptual figures)\n"
            f"   - Fig 3 & Fig 4: fig_type='code_chart' (matplotlib-generated data charts)\n"
            f"4. Choose the most appropriate ai_image_type for architecture diagrams\n"
            f"5. Every figure must use a DIFFERENT chart_type — NO duplicates\n\n"
            f"Return the figure plan as JSON with 'domain' and 'figures' fields."
        )

        try:
            # Use figure_prompt config (text model), NOT figure_gen (Gemini image model)
            figure_prompt_config = self.config.for_stage("figure_prompt")
            result = await self.generate_json(
                FIGURE_PLAN_SYSTEM, prompt, stage_override=figure_prompt_config
            )
            figures = result.get("figures", [])
            if not figures:
                self.log("Figure plan returned empty, using default plan")
                return self._default_figure_plan()
            # Validate each figure spec
            validated = []
            seen_chart_types: set[str] = set()
            for fig in figures:
                if "fig_key" not in fig:
                    continue
                fig.setdefault("fig_type", "code_chart")
                fig.setdefault("chart_type", "grouped_bar")
                fig.setdefault("caption", fig.get("description", ""))
                # Validate ai_image_type for AI figures & clean verbose captions
                if fig["fig_type"] == "ai_image":
                    img_type = fig.get("ai_image_type", "generic")
                    if img_type not in AI_FIGURE_TEMPLATES:
                        logger.warning(
                            "Unknown ai_image_type %r, falling back to 'generic'",
                            img_type,
                        )
                        fig["ai_image_type"] = "generic"
                    # Clean caption: LLM sometimes returns generation-prompt-
                    # length text as the caption.  Keep it short & academic.
                    fig["caption"] = _clean_ai_image_caption(
                        fig["caption"], fig.get("title", ""),
                    )
                # Deduplicate chart_type for code_chart figures
                if fig["fig_type"] == "code_chart":
                    ct = fig["chart_type"]
                    if ct in seen_chart_types:
                        logger.warning(
                            "Duplicate chart_type %r in figure plan, skipping %s",
                            ct, fig.get("fig_key"),
                        )
                        continue
                    seen_chart_types.add(ct)
                validated.append(fig)
            if not validated:
                return self._default_figure_plan()

            # Enforce exactly 4 figures: 2 ai_image + 2 code_chart
            ai_figs = [f for f in validated if f.get("fig_type") == "ai_image"]
            code_figs = [f for f in validated if f.get("fig_type") == "code_chart"]

            # Take exactly 2 of each; if not enough, default plan fills the gap
            ai_figs = ai_figs[:2]
            code_figs = code_figs[:2]

            if len(ai_figs) < 2 or len(code_figs) < 2:
                self.log(
                    f"Figure plan has {len(ai_figs)} ai_image + {len(code_figs)} code_chart, "
                    f"need 2+2; falling back to default plan"
                )
                return self._default_figure_plan()

            validated = ai_figs + code_figs
            return validated
        except Exception as e:
            logger.warning("Figure planning failed: %s", e, exc_info=True)
            self.log(f"Figure planning failed ({e}), using default plan")
            return self._default_figure_plan()

    def _default_figure_plan(self) -> list[dict]:
        """Fallback figure plan — exactly 2 ai_image + 2 code_chart."""
        return [
            {
                "fig_key": "fig1_framework_overview",
                "fig_type": "ai_image",
                "ai_image_type": "system_overview",
                "chart_type": None,
                "title": "Framework Overview",
                "description": "Framework overview showing all key components and data flow.",
                "caption": "Architecture of the proposed framework showing key components and data flow.",
            },
            {
                "fig_key": "fig2_qualitative_examples",
                "fig_type": "ai_image",
                "ai_image_type": "qualitative_comparison",
                "chart_type": None,
                "title": "Qualitative Examples",
                "description": "Qualitative comparison of representative examples showing model behavior.",
                "caption": "Qualitative examples illustrating how the proposed method processes inputs.",
            },
            {
                "fig_key": "fig3_results_comparison",
                "fig_type": "code_chart",
                "chart_type": "grouped_bar",
                "title": "Main Results",
                "description": "Comparison of baselines vs proposed method across benchmark datasets.",
                "caption": "Performance comparison across benchmark datasets.",
            },
            {
                "fig_key": "fig4_ablation",
                "fig_type": "code_chart",
                "chart_type": "horizontal_bar",
                "title": "Ablation Study",
                "description": "Component contribution analysis showing the impact of removing each module.",
                "caption": "Ablation study showing contribution of each component.",
            },
        ]

    # -----------------------------------------------------------------------
    # Survey figure planning
    # -----------------------------------------------------------------------

    async def _plan_survey_figures(
        self, context: str, evidence_block: str, survey_size: str = "standard"
    ) -> list[dict]:
        """Ask LLM to plan which figures to generate for a survey paper."""
        # Size-based figure count: short=2, standard=3, long=4
        size_fig_counts = {"short": 2, "standard": 3, "long": 4}
        target_figs = size_fig_counts.get(survey_size, 3)

        prompt = (
            f"Plan the figures for this survey paper.\n\n"
            f"Survey context:\n{context}\n\n"
            f"{evidence_block}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Identify the research domain (nlp/cv/llm/multimodal/general_ml)\n"
            f"2. Select EXACTLY {target_figs} figures appropriate for a survey paper:\n"
            f"   - One taxonomy/overview diagram (ai_image, ai_image_type='system_overview')\n"
            f"   - One benchmark comparison chart (code_chart, e.g. grouped_bar or line)\n"
            f"   - Additional figures as appropriate for the survey size\n"
            f"3. For ai_image: use 'system_overview' or 'qualitative_comparison' types\n"
            f"4. For code_chart: use meaningful chart types (grouped_bar, line, heatmap)\n"
            f"5. Every code_chart must use a DIFFERENT chart_type — NO duplicates\n\n"
            f"Return the figure plan as JSON with 'domain' and 'figures' fields."
        )

        try:
            figure_prompt_config = self.config.for_stage("figure_prompt")
            result = await self.generate_json(
                FIGURE_PLAN_SYSTEM, prompt, stage_override=figure_prompt_config
            )
            figures = result.get("figures", [])
            if not figures:
                self.log("Survey figure plan returned empty, using default plan")
                return self._default_survey_figure_plan(survey_size)
            # Validate and normalize
            validated = []
            seen_chart_types: set[str] = set()
            for fig in figures:
                if "fig_key" not in fig:
                    continue
                fig.setdefault("fig_type", "ai_image")
                fig.setdefault("chart_type", None)
                fig.setdefault("caption", fig.get("description", ""))
                if fig["fig_type"] == "ai_image":
                    img_type = fig.get("ai_image_type", "system_overview")
                    if img_type not in AI_FIGURE_TEMPLATES:
                        fig["ai_image_type"] = "system_overview"
                    fig["caption"] = _clean_ai_image_caption(
                        fig["caption"], fig.get("title", ""),
                    )
                elif fig["fig_type"] == "code_chart":
                    ct = fig.get("chart_type", "grouped_bar")
                    if ct in seen_chart_types:
                        logger.warning(
                            "Duplicate chart_type %r in survey figure plan, skipping %s",
                            ct, fig.get("fig_key"),
                        )
                        continue
                    seen_chart_types.add(ct)
                validated.append(fig)
            if not validated:
                return self._default_survey_figure_plan(survey_size)
            return validated[:target_figs]
        except Exception as e:
            logger.warning("Survey figure planning failed: %s", e, exc_info=True)
            self.log(f"Survey figure planning failed ({e}), using default plan")
            return self._default_survey_figure_plan(survey_size)

    def _default_survey_figure_plan(self, survey_size: str = "standard") -> list[dict]:
        """Fallback figure plan for survey papers."""
        # Size-based figures: short=2, standard=3, long=4
        if survey_size == "short":
            return [
                {
                    "fig_key": "fig1_taxonomy",
                    "fig_type": "ai_image",
                    "ai_image_type": "system_overview",
                    "chart_type": None,
                    "title": "Taxonomy of Methods",
                    "description": "Taxonomy diagram showing the categorization of methods in the field.",
                    "caption": "Taxonomy of surveyed methods organized by approach and methodology.",
                },
                {
                    "fig_key": "fig2_benchmark",
                    "fig_type": "code_chart",
                    "chart_type": "grouped_bar",
                    "title": "Benchmark Comparison",
                    "description": "Comparison of methods on key benchmarks.",
                    "caption": "Performance comparison of representative methods on standard benchmarks.",
                },
            ]
        elif survey_size == "long":
            return [
                {
                    "fig_key": "fig1_taxonomy",
                    "fig_type": "ai_image",
                    "ai_image_type": "system_overview",
                    "chart_type": None,
                    "title": "Taxonomy of Methods",
                    "description": "Comprehensive taxonomy diagram showing the categorization of methods in the field.",
                    "caption": "Taxonomy of surveyed methods organized by approach and methodology.",
                },
                {
                    "fig_key": "fig2_benchmark",
                    "fig_type": "code_chart",
                    "chart_type": "grouped_bar",
                    "title": "Benchmark Comparison",
                    "description": "Comparison of methods on key benchmarks.",
                    "caption": "Performance comparison of representative methods on standard benchmarks.",
                },
                {
                    "fig_key": "fig3_evolution",
                    "fig_type": "ai_image",
                    "ai_image_type": "qualitative_comparison",
                    "chart_type": None,
                    "title": "Field Evolution",
                    "description": "Timeline showing how the field evolved over time.",
                    "caption": "Evolution of the field over time, highlighting key developments.",
                },
                {
                    "fig_key": "fig4_method_comparison",
                    "fig_type": "code_chart",
                    "chart_type": "heatmap",
                    "title": "Method Comparison Matrix",
                    "description": "Comparison matrix showing methods vs characteristics.",
                    "caption": "Method comparison matrix showing capabilities across different dimensions.",
                },
            ]
        else:  # standard (default)
            return [
                {
                    "fig_key": "fig1_taxonomy",
                    "fig_type": "ai_image",
                    "ai_image_type": "system_overview",
                    "chart_type": None,
                    "title": "Taxonomy of Methods",
                    "description": "Taxonomy diagram showing the categorization of methods in the field.",
                    "caption": "Taxonomy of surveyed methods organized by approach and methodology.",
                },
                {
                    "fig_key": "fig2_benchmark",
                    "fig_type": "code_chart",
                    "chart_type": "grouped_bar",
                    "title": "Benchmark Comparison",
                    "description": "Comparison of methods on key benchmarks.",
                    "caption": "Performance comparison of representative methods on standard benchmarks.",
                },
                {
                    "fig_key": "fig3_evolution",
                    "fig_type": "ai_image",
                    "ai_image_type": "qualitative_comparison",
                    "chart_type": None,
                    "title": "Field Evolution",
                    "description": "Timeline showing how the field evolved over time.",
                    "caption": "Evolution of the field over time, highlighting key developments.",
                },
            ]

    # -----------------------------------------------------------------------
    # Chart prompt builder
    # -----------------------------------------------------------------------
