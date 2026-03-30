"""Stage reflection — evaluate results after key stages and adjust the plan.

Inspired by EvoScientist's PLAN/REFLECTION dual-mode planner.
Reflection triggers after: PLANNING, EXECUTION, ANALYSIS.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_REFLECTION_SYSTEM = """\
You are a research pipeline reflection agent. After a stage completes,
evaluate the results and suggest adjustments.

Return JSON:
{
  "quality_score": <1-10>,
  "completed_signals": ["<what was achieved>"],
  "unmet_signals": ["<what is still missing or weak>"],
  "suggestions": ["<concrete actionable improvement>"],
  "should_continue": true,
  "reason": "<1-sentence rationale>"
}

Rules:
- Be specific and actionable in suggestions
- quality_score < 5 means significant issues
- should_continue=false only if the pipeline cannot produce useful output
- Do NOT suggest changes outside the current pipeline's scope"""

# Stages that trigger reflection after success
REFLECTION_STAGES = {"planning", "execution", "analysis"}

_FAILURE_REFLECTION_SYSTEM = """\
You are a research pipeline failure analyst. A stage has failed on retry.
Analyze the error and accumulated context to produce an actionable recovery plan.

Return JSON:
{
  "root_cause": "<1-sentence diagnosis of why the stage failed>",
  "category": "<one of: data_issue, model_issue, config_issue, code_bug, infra_issue, timeout>",
  "suggestions": ["<concrete change to make before retrying>"],
  "param_adjustments": {
    "<key>": "<new_value>"
  },
  "should_retry": true,
  "reason": "<1-sentence rationale>"
}

Rules:
- Be SPECIFIC: "reduce batch size from 32 to 8" not "try smaller batch size"
- param_adjustments should map to blueprint or config keys that agents understand
- should_retry=false only if the error is fundamentally unrecoverable
- Consider: was it a timeout? OOM? bad data? missing dependency? LLM parsing error?"""


async def reflect_on_stage(
    stage_name: str,
    stage_result: dict[str, Any],
    accumulated: dict[str, Any],
    dispatcher,
    stage_config,
) -> dict[str, Any]:
    """Reflect on a completed stage and return structured assessment.

    Args:
        stage_name: Name of the completed stage
        stage_result: Output of the completed stage
        accumulated: All accumulated results so far
        dispatcher: ModelDispatcher for LLM calls
        stage_config: StageModelConfig to use

    Returns:
        Reflection dict with quality_score, suggestions, etc.
        Empty dict if reflection is not needed for this stage.
    """
    if stage_name not in REFLECTION_STAGES:
        return {}

    context = _build_reflection_context(stage_name, stage_result, accumulated)
    if not context:
        return {}

    try:
        raw = await dispatcher.generate(stage_config, _REFLECTION_SYSTEM, context, json_mode=True)
        reflection = json.loads(raw)
        if not isinstance(reflection, dict):
            return {}

        # Clamp score
        score = reflection.get("quality_score", 5)
        if isinstance(score, (int, float)):
            reflection["quality_score"] = max(1, min(10, int(score)))

        logger.info(
            "[REFLECTION] %s: score=%d, unmet=%d, suggestions=%d, continue=%s",
            stage_name,
            reflection.get("quality_score", 0),
            len(reflection.get("unmet_signals", [])),
            len(reflection.get("suggestions", [])),
            reflection.get("should_continue", True),
        )
        return reflection

    except Exception as e:
        logger.warning("Reflection failed for %s: %s", stage_name, e)
        return {}


async def reflect_on_failure(
    stage_name: str,
    error: str,
    attempt: int,
    accumulated: dict[str, Any],
    dispatcher,
    stage_config,
) -> dict[str, Any]:
    """Reflect on a FAILED stage attempt and suggest recovery strategy.

    Called between retries so the next attempt can adapt instead of
    blindly repeating the same approach.

    Returns:
        Reflection dict with root_cause, suggestions, param_adjustments, etc.
        Empty dict if reflection fails or is not applicable.
    """
    parts = [
        f"Stage FAILED: {stage_name} (attempt {attempt})",
        f"Error: {error}",
    ]

    topic = accumulated.get("topic", "")
    if topic:
        parts.append(f"Research topic: {topic}")

    # Include blueprint context if available
    bp = accumulated.get("experiment_blueprint", {})
    if bp and isinstance(bp, dict):
        method = (bp.get("proposed_method") or {}).get("name", "")
        if method:
            parts.append(f"Method: {method}")
        compute = bp.get("compute_requirements", {})
        if compute:
            parts.append(f"Compute: {json.dumps(compute)}")

    context = "\n".join(parts)

    try:
        raw = await dispatcher.generate(
            stage_config, _FAILURE_REFLECTION_SYSTEM, context, json_mode=True,
        )
        reflection = json.loads(raw)
        if not isinstance(reflection, dict):
            return {}

        logger.info(
            "[FAILURE REFLECTION] %s attempt %d: cause=%s, category=%s, %d suggestions, retry=%s",
            stage_name, attempt,
            reflection.get("root_cause", "?")[:80],
            reflection.get("category", "?"),
            len(reflection.get("suggestions", [])),
            reflection.get("should_retry", True),
        )
        return reflection

    except Exception as e:
        logger.warning("Failure reflection failed for %s: %s", stage_name, e)
        return {}


def _build_reflection_context(
    stage_name: str,
    stage_result: dict[str, Any],
    accumulated: dict[str, Any],
) -> str:
    """Build a concise context string for the reflection LLM call."""
    parts = [f"Stage completed: {stage_name}\n"]

    topic = accumulated.get("topic", "")
    if topic:
        parts.append(f"Research topic: {topic}")

    if stage_name == "planning":
        bp = stage_result.get("experiment_blueprint", stage_result)
        method = (bp.get("proposed_method") or {}).get("name", "")
        metrics = [m.get("name", "") for m in (bp.get("metrics") or [])]
        datasets = [d.get("name", "") for d in (bp.get("datasets") or [])]
        baselines = [b.get("name", "") for b in (bp.get("baselines") or [])]
        parts.append(f"Method: {method}")
        parts.append(f"Metrics: {', '.join(metrics)}")
        parts.append(f"Datasets: {', '.join(datasets)}")
        parts.append(f"Baselines: {', '.join(baselines)}")
        parts.append(f"Ablation groups: {len(bp.get('ablation_groups', []))}")

    elif stage_name == "execution":
        exp = stage_result.get("experiment_output", stage_result)
        results = (exp.get("experiment_results") or {}).get("main_results", [])
        status = exp.get("status", "unknown")
        parts.append(f"Execution status: {status}")
        parts.append(f"Main results entries: {len(results)}")
        if results:
            # Show first result summary
            first = results[0] if results else {}
            metrics = first.get("metrics", [])
            metric_strs = [f"{m.get('metric_name', '?')}={m.get('value', '?')}" for m in metrics[:5]]
            parts.append(f"Sample metrics: {', '.join(metric_strs)}")

    elif stage_name == "analysis":
        analysis = stage_result.get("experiment_analysis", stage_result)
        summary = analysis.get("summary", "")
        if summary:
            parts.append(f"Analysis summary: {summary[:300]}")

    # Include ideation context for all reflections
    ideation = accumulated.get("ideation_output", {})
    if isinstance(ideation, dict):
        hyps = ideation.get("hypotheses", [])
        if hyps:
            hyp_titles = [h.get("title", "") for h in hyps[:3]]
            parts.append(f"Hypotheses: {'; '.join(hyp_titles)}")

    return "\n".join(parts)
