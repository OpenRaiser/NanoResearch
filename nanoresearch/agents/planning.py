"""Planning agent — generates experiment blueprint from ideation output."""

from __future__ import annotations

import json
import logging
from typing import Any

from nanoresearch.agents.base import BaseResearchAgent
from nanoresearch.schemas.experiment import ExperimentBlueprint
from nanoresearch.schemas.manifest import PipelineStage

logger = logging.getLogger(__name__)

# Configurable limits
MAX_PAPERS_IN_SUMMARY = 15

PLANNING_SYSTEM_PROMPT = """You are a research experiment planner. Your task is to design a rigorous experiment plan based on the selected hypothesis from the ideation phase.

CRITICAL RULES — evidence grounding:
- NEVER invent baseline performance numbers. Use ONLY numbers from the published evidence provided.
- If a metric value is not available in the evidence, set it to "N/A" — do NOT guess.
- For the proposed method, describe expected improvements as a "projected improvement range" (e.g. "5-10% improvement over best baseline"), NOT as exact numbers.
- Every baseline number MUST have a "performance_provenance" entry citing which paper it came from.
- Mark proposed-method values as projected: set "is_projected" to true for each metric.

Design experiments that are:
- Reproducible with standard ML frameworks
- Include proper baselines for comparison (using REAL published numbers)
- Have clear evaluation metrics
- Include ablation studies to validate each component

Always respond in valid JSON format."""


class PlanningAgent(BaseResearchAgent):
    stage = PipelineStage.PLANNING

    async def run(self, **inputs: Any) -> dict[str, Any]:
        if "ideation_output" not in inputs:
            raise ValueError("PlanningAgent requires 'ideation_output' in inputs")
        ideation_data: dict = inputs["ideation_output"]
        if not isinstance(ideation_data, dict):
            raise TypeError(
                f"Expected dict for ideation_output, got {type(ideation_data).__name__}"
            )
        self.log("Starting experiment planning")

        topic = ideation_data.get("topic", "")
        selected_hyp = ideation_data.get("selected_hypothesis", "")
        rationale = ideation_data.get("rationale", "")

        # Find the selected hypothesis details
        hyp_detail = ""
        hypotheses = ideation_data.get("hypotheses", [])
        for h in hypotheses:
            if h.get("hypothesis_id") == selected_hyp:
                hyp_detail = json.dumps(h, indent=2)
                break
        if not hyp_detail and hypotheses:
            logger.warning(
                "Selected hypothesis %r not found among %d hypotheses, using first",
                selected_hyp, len(hypotheses),
            )
            hyp_detail = json.dumps(hypotheses[0], indent=2)
            selected_hyp = hypotheses[0].get("hypothesis_id", selected_hyp)

        # Summarize related gaps (limit to 10 to avoid prompt overflow)
        gaps_text = json.dumps(ideation_data.get("gaps", [])[:10], indent=2)

        # Summarize key papers
        papers = ideation_data.get("papers", [])
        papers_summary = "\n".join(
            f"- {p.get('title', '?')} ({p.get('year', '?')})"
            for p in papers[:MAX_PAPERS_IN_SUMMARY]
        )

        # Build evidence block from extracted metrics
        evidence_block = self._build_evidence_block(ideation_data)

        prompt = f"""Research Topic: "{topic}"

Selected Hypothesis: {selected_hyp}
{hyp_detail}

Selection Rationale: {rationale}

Identified Research Gaps:
{gaps_text}

Key Related Papers:
{papers_summary}

{evidence_block}

Design a comprehensive experiment blueprint as JSON with:
1. "title": Experiment title
2. "hypothesis_ref": "{selected_hyp}"
3. "datasets": Array of datasets, each with:
   - "name", "description", "source_url", "size_info", "preprocessing_notes"
4. "baselines": Array of baseline methods, each with:
   - "name", "description", "reference_paper_id"
   - "expected_performance": dict of metric→value (use ONLY values from the evidence above, or "N/A")
   - "performance_provenance": dict of metric→source (e.g. "Abstract of arxiv:2401.00001")
   - "is_projected": dict of metric→bool (true only for proposed method projections)
5. "proposed_method": Object describing the proposed method with:
   - "name", "description", "key_components" (list), "architecture" (text description)
6. "metrics": Array of evaluation metrics, each with:
   - "name", "description", "higher_is_better" (bool), "primary" (bool)
7. "ablation_groups": Array of ablation studies, each with:
   - "group_name", "description", "variants" (array of dicts describing each variant)
8. "compute_requirements": Object with "gpu_type", "num_gpus", "estimated_hours", "memory_gb"
9. "evidence_summary": Brief summary of which published numbers were used
10. "data_provenance_note": Statement distinguishing published baseline numbers from projected improvements

IMPORTANT: Use ONLY the numbers from the PUBLISHED QUANTITATIVE EVIDENCE section.
If evidence is missing for a baseline-metric pair, use "N/A".

Return ONLY valid JSON."""

        # Retry loop: if blueprint validation fails, retry with error feedback
        max_planning_retries = 2
        last_validation_error = ""
        for planning_attempt in range(max_planning_retries + 1):
            retry_prompt = prompt
            if last_validation_error:
                retry_prompt += (
                    f"\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
                    f"Your previous JSON output failed schema validation:\n"
                    f"{last_validation_error}\n"
                    f"Please fix the issues and return valid JSON.\n"
                    f"--- END ERROR ---"
                )

            result = await self.generate_json(PLANNING_SYSTEM_PROMPT, retry_prompt)

            # Guard: generate_json may return a list instead of a dict
            if isinstance(result, list):
                if len(result) == 1 and isinstance(result[0], dict):
                    result = result[0]
                else:
                    last_validation_error = (
                        f"Expected JSON object, got list of length {len(result)}"
                    )
                    if planning_attempt < max_planning_retries:
                        logger.warning(
                            "Planning attempt %d/%d: %s",
                            planning_attempt + 1, max_planning_retries + 1,
                            last_validation_error,
                        )
                        continue
                    raise RuntimeError(last_validation_error)

            # Coerce fields the LLM may return as structured objects but schema expects as strings
            result = self._coerce_blueprint_fields(result)

            try:
                blueprint = ExperimentBlueprint.model_validate(result)
                break  # Validation succeeded
            except Exception as exc:
                last_validation_error = str(exc)
                if planning_attempt < max_planning_retries:
                    logger.warning(
                        "Planning attempt %d/%d validation failed: %s",
                        planning_attempt + 1, max_planning_retries + 1,
                        last_validation_error,
                    )
                    continue
                logger.error("Blueprint validation failed after %d attempts: %s",
                             max_planning_retries + 1, exc)
                raise RuntimeError(
                    f"LLM output does not match ExperimentBlueprint schema: {exc}"
                ) from exc

        # Save output
        output_path = self.workspace.write_json(
            "plans/experiment_blueprint.json",
            blueprint.model_dump(mode="json"),
        )
        self.workspace.register_artifact(
            "experiment_blueprint", output_path, self.stage
        )
        logger.info("[%s] Blueprint generated: %s", self.stage.value, blueprint.title)
        return blueprint.model_dump(mode="json")

    @staticmethod
    def _build_evidence_block(ideation_data: dict) -> str:
        """Format extracted evidence for inclusion in the planning prompt."""
        evidence = ideation_data.get("evidence", {})
        metrics = evidence.get("extracted_metrics", [])

        if not metrics:
            return (
                "=== PUBLISHED QUANTITATIVE EVIDENCE ===\n"
                "No quantitative evidence was extracted from the literature.\n"
                "Use 'N/A' for all baseline expected_performance values.\n"
                "=== END EVIDENCE ==="
            )

        lines = ["=== PUBLISHED QUANTITATIVE EVIDENCE ==="]
        for m in metrics[:100]:  # cap to avoid prompt overflow
            if not isinstance(m, dict):
                continue
            value = m.get("value", "?")
            unit = m.get("unit", "")
            unit_str = f" {unit}" if unit else ""
            lines.append(
                f"- {m.get('method_name', '?')} on {m.get('dataset', '?')}: "
                f"{m.get('metric_name', '?')} = {value}{unit_str} "
                f"(source: {m.get('paper_id', '?')} — \"{m.get('context', '')}\")"
            )

        notes = evidence.get("extraction_notes", "")
        if notes:
            lines.append(f"\nExtraction notes: {notes}")

        warnings = evidence.get("coverage_warnings", [])
        for w in warnings:
            lines.append(f"WARNING: {w}")

        lines.append("=== END EVIDENCE ===")
        return "\n".join(lines)

    @staticmethod
    def _coerce_blueprint_fields(data: dict) -> dict:
        """Coerce LLM output fields that should be strings but came as dicts/lists.

        The LLM sometimes returns structured objects for fields like size_info,
        preprocessing_notes, evidence_summary. Convert them to JSON strings.
        """
        # Top-level string fields
        for key in ("evidence_summary", "data_provenance_note"):
            val = data.get(key)
            if val is not None and not isinstance(val, str):
                data[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)

        # Dataset-level string fields
        for ds in data.get("datasets", []):
            if not isinstance(ds, dict):
                continue
            for key in ("size_info", "preprocessing_notes", "description", "source_url"):
                val = ds.get(key)
                if val is not None and not isinstance(val, str):
                    ds[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)

        # Baseline-level string fields
        for bl in data.get("baselines", []):
            if not isinstance(bl, dict):
                continue
            for key in ("description",):
                val = bl.get(key)
                if val is not None and not isinstance(val, str):
                    bl[key] = str(val)

        # ComputeRequirements string fields
        cr = data.get("compute_requirements")
        if isinstance(cr, dict):
            for key in ("gpu_type", "notes"):
                val = cr.get(key)
                if val is not None and not isinstance(val, str):
                    cr[key] = str(val)
            # Numeric fields — LLM may return "4" or "8.0" as strings
            # Schema fields: num_gpus (int), estimated_hours (float), memory_gb (float)
            for key in ("num_gpus",):
                val = cr.get(key)
                if isinstance(val, str):
                    try:
                        cr[key] = int(val) if val.isdigit() else int(float(val))
                    except (ValueError, TypeError):
                        pass
            for key in ("estimated_hours", "memory_gb"):
                val = cr.get(key)
                if isinstance(val, str):
                    try:
                        cr[key] = float(val)
                    except (ValueError, TypeError):
                        pass

        return data
