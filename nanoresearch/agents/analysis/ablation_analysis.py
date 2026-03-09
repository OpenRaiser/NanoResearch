"""Quantify contribution of each ablated component."""
from __future__ import annotations


def quantify_ablation_contributions(
    full_result: dict,
    ablation_results: list[dict],
    primary_metric: str,
    higher_is_better: bool = True,
) -> list[dict]:
    """Compute contribution of each component to overall performance.

    Args:
        full_result: {"metric_name": value, ...} for the full model.
        ablation_results: [{"variant_name": str, "metrics": {metric: value}}]
        primary_metric: Which metric to rank by.
        higher_is_better: Direction of the metric.

    Returns:
        Sorted list of contribution dicts (largest drop first).
    """
    full_score = full_result.get(primary_metric)
    if full_score is None or not isinstance(full_score, (int, float)):
        return []

    contributions = []
    for ablation in ablation_results:
        variant = ablation.get("variant_name", "unknown")
        metrics = ablation.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        ablated_score = metrics.get(primary_metric)
        if ablated_score is None or not isinstance(ablated_score, (int, float)):
            continue

        if higher_is_better:
            drop = full_score - ablated_score
        else:
            drop = ablated_score - full_score  # lower is better → increase = drop

        relative = (
            (drop / abs(full_score) * 100)
            if abs(full_score) > 1e-8
            else 0.0
        )

        contributions.append({
            "component": variant,
            "full_model_score": round(full_score, 4),
            "without_component_score": round(ablated_score, 4),
            "absolute_drop": round(drop, 4),
            "relative_contribution_pct": round(relative, 2),
            "is_critical": relative > 10.0,  # >10% drop = critical
        })

    contributions.sort(key=lambda x: x["absolute_drop"], reverse=True)
    return contributions
