"""Blueprint semantic validation — checks internal consistency after PLANNING."""

from __future__ import annotations

import logging
from typing import Any

from nanoresearch.constants import LOWER_IS_BETTER_PATTERNS

logger = logging.getLogger(__name__)


def validate_blueprint(blueprint: dict[str, Any]) -> list[str]:
    """Return a list of semantic issues found in the blueprint.

    An empty list means the blueprint passed all checks.
    """
    issues: list[str] = []

    # 1. Metrics list must be non-empty
    metrics = blueprint.get("metrics", [])
    if not metrics:
        issues.append("Blueprint has no evaluation metrics defined.")

    # 2. At least one primary metric
    has_primary = any(
        m.get("primary", False) for m in metrics if isinstance(m, dict)
    )
    if metrics and not has_primary:
        issues.append("No metric is marked as primary=True.")

    # 3. Metric direction consistency
    for m in metrics:
        if not isinstance(m, dict):
            continue
        name = m.get("name", "").lower()
        higher = m.get("higher_is_better", True)
        for pattern in LOWER_IS_BETTER_PATTERNS:
            if pattern in name and higher:
                issues.append(
                    f"Metric '{m.get('name')}' contains '{pattern}' but "
                    f"higher_is_better=True — likely should be False."
                )
                break

    # 4. Proposed method must have key_components
    pm = blueprint.get("proposed_method", {})
    if isinstance(pm, dict):
        kc = pm.get("key_components", [])
        if not kc:
            issues.append("proposed_method.key_components is empty.")

    # 5. Ablation variable names should reference key_components
    key_components_lower = set()
    if isinstance(pm, dict):
        for comp in pm.get("key_components", []):
            if isinstance(comp, str):
                key_components_lower.add(comp.lower())
    method_desc = ""
    if isinstance(pm, dict):
        method_desc = (
            pm.get("description", "") + " " +
            pm.get("architecture", "") + " " +
            " ".join(str(c) for c in pm.get("key_components", []))
        ).lower()

    for ag in blueprint.get("ablation_groups", []):
        if not isinstance(ag, dict):
            continue
        group_name = ag.get("group_name", "")
        for variant in ag.get("variants", []):
            if not isinstance(variant, dict):
                continue
            var_name = variant.get("name", variant.get("variant_name", ""))
            # Check if ablation variant references something in the method
            if var_name and method_desc and var_name.lower() not in method_desc:
                # Soft check: only warn if name doesn't overlap at all
                words = set(var_name.lower().split())
                method_words = set(method_desc.split())
                if not words & method_words:
                    issues.append(
                        f"Ablation variant '{var_name}' in group "
                        f"'{group_name}' doesn't reference anything "
                        f"in the proposed method description."
                    )

    # 6. Baseline expected_performance metric names must match metrics list
    metric_names = {
        m.get("name", "") for m in metrics if isinstance(m, dict)
    }
    for bl in blueprint.get("baselines", []):
        if not isinstance(bl, dict):
            continue
        perf = bl.get("expected_performance", {})
        if isinstance(perf, dict):
            for metric_name in perf:
                if metric_names and metric_name not in metric_names:
                    issues.append(
                        f"Baseline '{bl.get('name')}' has performance for "
                        f"metric '{metric_name}' which is not in the "
                        f"metrics list: {metric_names}"
                    )

    return issues
