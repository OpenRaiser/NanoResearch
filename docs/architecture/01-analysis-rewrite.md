# 2. P0: ANALYSIS Module Rewrite

## Problem

`analysis.py` (420 lines) is the weakest link. It takes execution results, sends them
to an LLM prompt, and returns the LLM's JSON summary. Zero computational analysis.
No statistical tests, no curve fitting, no contribution quantification.

A research paper that says "our method achieves 92.3% accuracy" without significance
testing is unacceptable at any top venue.

## What to Build

Create `nanoresearch/agents/analysis/` package:

```
agents/analysis/
├── __init__.py              # AnalysisAgent entry (existing run() method)
├── statistics.py            # Statistical significance tests
├── training_dynamics.py     # Training curve analysis
├── ablation_analysis.py     # Component contribution quantification
├── comparison_matrix.py     # Auto comparison table builder
├── claim_verifier.py        # Result-claim consistency check
```

## 2.1 Statistical Significance Testing (`statistics.py`)

```python
"""Statistical significance testing for experiment results."""
import math
from typing import Optional


def welch_t_test(sample_a: list[float], sample_b: list[float]) -> dict:
    """Welch's t-test (unequal variance) for two independent samples.

    Use this instead of scipy to avoid adding a heavy dependency.
    Returns t_statistic, p_value (two-tailed), degrees_of_freedom.
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return {"t_statistic": None, "p_value": None, "df": None,
                "reason": "need at least 2 samples per group"}

    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return {"t_statistic": 0.0, "p_value": 1.0, "df": n_a + n_b - 2,
                "reason": "zero variance"}

    t_stat = (mean_a - mean_b) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else n_a + n_b - 2

    # Approximate two-tailed p-value via normal for df > 30,
    # otherwise use a conservative lookup table.
    p_value = _approx_two_tailed_p(t_stat, df)

    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "df": round(df, 1),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


def cohens_d(sample_a: list[float], sample_b: list[float]) -> Optional[float]:
    """Effect size (Cohen's d) between two groups."""
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return None
    mean_a = sum(sample_a) / n_a
    mean_b = sum(sample_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in sample_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in sample_b) / (n_b - 1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-12:
        return 0.0
    return round((mean_a - mean_b) / pooled_std, 4)


def bootstrap_ci(samples: list[float], n_bootstrap: int = 1000,
                 confidence: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap confidence interval (no scipy needed)."""
    import random
    rng = random.Random(seed)
    n = len(samples)
    if n < 2:
        return {"lower": None, "upper": None, "mean": None}

    boot_means = []
    for _ in range(n_bootstrap):
        boot = [rng.choice(samples) for _ in range(n)]
        boot_means.append(sum(boot) / n)

    boot_means.sort()
    alpha = 1 - confidence
    lo_idx = int(n_bootstrap * alpha / 2)
    hi_idx = int(n_bootstrap * (1 - alpha / 2))
    return {
        "mean": round(sum(samples) / n, 6),
        "lower": round(boot_means[lo_idx], 6),
        "upper": round(boot_means[min(hi_idx, n_bootstrap - 1)], 6),
        "confidence": confidence,
    }


def compute_significance_report(proposed_runs: list[float],
                                baseline_runs: list[float],
                                metric_name: str,
                                higher_is_better: bool = True) -> dict:
    """Full significance report for one metric comparison."""
    t_result = welch_t_test(proposed_runs, baseline_runs)
    effect = cohens_d(proposed_runs, baseline_runs)
    ci_proposed = bootstrap_ci(proposed_runs)
    ci_baseline = bootstrap_ci(baseline_runs)

    mean_p = sum(proposed_runs) / len(proposed_runs) if proposed_runs else 0
    mean_b = sum(baseline_runs) / len(baseline_runs) if baseline_runs else 0
    improvement = mean_p - mean_b
    if not higher_is_better:
        improvement = -improvement

    interpretation = "not enough data"
    if t_result["p_value"] is not None:
        if t_result["p_value"] < 0.05 and effect is not None and abs(effect) > 0.2:
            interpretation = "statistically significant improvement"
        elif t_result["p_value"] < 0.05:
            interpretation = "statistically significant but small effect"
        else:
            interpretation = "not statistically significant"

    return {
        "metric": metric_name,
        "proposed_mean": round(mean_p, 6),
        "baseline_mean": round(mean_b, 6),
        "improvement": round(improvement, 6),
        "t_test": t_result,
        "cohens_d": effect,
        "proposed_ci": ci_proposed,
        "baseline_ci": ci_baseline,
        "interpretation": interpretation,
    }


def _approx_two_tailed_p(t: float, df: float) -> float:
    """Approximate two-tailed p-value. Uses normal approx for df > 30.

    For df <= 30, uses a conservative lookup table indexed by df ranges.
    The lookup is intentionally conservative (overestimates p-value) to
    avoid false significance claims.
    """
    import math
    abs_t = abs(t)
    if df > 30:
        # Normal approximation (accurate for large df)
        p = math.erfc(abs_t / math.sqrt(2))
        return min(p, 1.0)
    # Conservative lookup table — uses df=10 critical values for all df <= 30.
    # df=10 has wider tails than df=30, so this overestimates p-values (safe).
    # Critical values for df=10, two-tailed:
    thresholds = [(4.587, 0.001), (3.169, 0.01), (2.764, 0.02),
                  (2.228, 0.05), (1.812, 0.10), (1.372, 0.20)]
    for threshold, p in thresholds:
        if abs_t >= threshold:
            return p
    return 0.50
```

> **Note**: Pure Python implementation to avoid requiring scipy. If scipy is available
> in the environment, you can optionally use `scipy.stats.ttest_ind` for exact p-values.
> The fallback is intentionally conservative (overestimates p-values).

## 2.2 Training Dynamics Analyzer (`training_dynamics.py`)

```python
"""Automated training curve analysis."""
from typing import Optional


def analyze_training_dynamics(training_log: list[dict]) -> dict:
    """Analyze training curve for convergence, overfitting, stability.

    Args:
        training_log: List of dicts with keys like "epoch", "train_loss",
                      "val_loss", plus optional metric keys.

    Returns:
        Dict with convergence_epoch, overfitting_detected, stability, etc.
    """
    val_losses = [e["val_loss"] for e in training_log
                  if isinstance(e.get("val_loss"), (int, float))
                  and _is_finite_val(e["val_loss"])]
    train_losses = [e["train_loss"] for e in training_log
                    if isinstance(e.get("train_loss"), (int, float))
                    and _is_finite_val(e["train_loss"])]

    result: dict = {
        "total_epochs": len(training_log),
        "has_val_loss": len(val_losses) > 0,
        "has_train_loss": len(train_losses) > 0,
    }

    if len(val_losses) < 3:
        result["analysis_skipped"] = "insufficient data (need >= 3 val_loss entries)"
        return result

    # 1. Convergence speed
    initial = val_losses[0]
    final_best = min(val_losses)
    target_90 = initial - 0.9 * (initial - final_best)
    convergence_epoch = len(val_losses)  # default: never
    for i, loss in enumerate(val_losses):
        if loss <= target_90:
            convergence_epoch = i
            break
    result["convergence_epoch"] = convergence_epoch
    result["convergence_ratio"] = round(convergence_epoch / len(val_losses), 3)

    # 2. Best epoch
    best_epoch = int(_argmin(val_losses))
    result["best_epoch"] = best_epoch
    result["best_val_loss"] = round(val_losses[best_epoch], 6)
    result["early_stopping_recommended"] = best_epoch < len(val_losses) * 0.7

    # 3. Overfitting detection (last 1/3 of training)
    split = max(1, len(val_losses) * 2 // 3)
    if len(val_losses[split:]) >= 2:
        val_tail = val_losses[split:]
        val_trend = _linear_slope(val_tail)
        result["val_loss_tail_trend"] = round(val_trend, 6)

        overfitting = False
        if len(train_losses) >= len(val_losses) and len(train_losses[split:]) >= 2:
            train_tail = train_losses[split:]
            train_trend = _linear_slope(train_tail)
            result["train_loss_tail_trend"] = round(train_trend, 6)
            overfitting = val_trend > 0.001 and train_trend < -0.001
        else:
            overfitting = val_trend > 0.001
        result["overfitting_detected"] = overfitting

    # 4. Train-val gap (at last epoch)
    if train_losses and val_losses:
        idx = min(len(train_losses), len(val_losses)) - 1
        result["final_train_val_gap"] = round(train_losses[idx] - val_losses[idx], 6)

    # 5. Loss stability (std of epoch-to-epoch differences)
    diffs = [val_losses[i + 1] - val_losses[i] for i in range(len(val_losses) - 1)]
    mean_loss = sum(val_losses) / len(val_losses)
    if mean_loss > 1e-8:
        stability = _std(diffs) / mean_loss
        result["loss_stability_ratio"] = round(stability, 4)
        result["loss_stability"] = (
            "stable" if stability < 0.1
            else "noisy" if stability < 0.5
            else "unstable"
        )

    return result


def _is_finite_val(v) -> bool:
    import math
    return isinstance(v, (int, float)) and math.isfinite(v)


def _argmin(xs: list[float]) -> int:
    return min(range(len(xs)), key=lambda i: xs[i])


def _linear_slope(ys: list[float]) -> float:
    """Least-squares slope (no numpy needed)."""
    n = len(ys)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(ys) / n
    num = sum((i - x_mean) * (ys[i] - y_mean) for i in range(n))
    denom = sum((i - x_mean) ** 2 for i in range(n))
    return num / denom if denom > 1e-12 else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    return (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
```

## 2.3 Ablation Contribution Analyzer (`ablation_analysis.py`)

```python
"""Quantify contribution of each ablated component."""


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
        Sorted list of contribution dicts.
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
            drop = ablated_score - full_score  # lower is better, so increase = drop

        relative = (drop / abs(full_score) * 100) if abs(full_score) > 1e-8 else 0.0

        contributions.append({
            "component": variant,
            "full_model_score": round(full_score, 4),
            "without_component_score": round(ablated_score, 4),
            "absolute_drop": round(drop, 4),
            "relative_contribution_pct": round(relative, 2),
            "is_critical": relative > 10.0,  # >10% drop = critical component
        })

    contributions.sort(key=lambda x: x["absolute_drop"], reverse=True)
    return contributions
```

## 2.4 Comparison Matrix Builder (`comparison_matrix.py`)

```python
"""Build structured comparison matrix for WRITING's Experiments section."""


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
        {"headers": [...], "rows": [...], "best_cells": [...]}
    """
    all_methods = baselines + [proposed]
    headers = ["Method"] + [m["name"] for m in metrics]

    rows = []
    for method in all_methods:
        row = {"method": method.get("name", "Unknown"),
               "is_proposed": method is proposed}
        for m in metrics:
            val = method.get("metrics", {}).get(m["name"])
            row[m["name"]] = val
        rows.append(row)

    # Find best and second-best per metric
    annotations = {}
    for m in metrics:
        vals = [(i, row.get(m["name"])) for i, row in enumerate(rows)
                if isinstance(row.get(m["name"]), (int, float))]
        if not vals:
            continue
        higher = m.get("higher_is_better", True)
        vals.sort(key=lambda x: x[1], reverse=higher)
        if len(vals) >= 1:
            annotations[(vals[0][0], m["name"])] = "best"
        if len(vals) >= 2:
            annotations[(vals[1][0], m["name"])] = "second"

    return {
        "headers": headers,
        "rows": rows,
        "annotations": annotations,
        "proposed_method_name": proposed.get("name", "Ours"),
    }


def _latex_escape_cell(text: str) -> str:
    """Escape underscores and other LaTeX-special chars in table cells.

    Used for BOTH headers and row labels (method names like 'Our_Method').
    """
    text = text.replace("_", "\\_")
    text = text.replace("%", "\\%")
    text = text.replace("&", "\\&")
    return text


def comparison_matrix_to_latex(matrix: dict) -> str:
    """Render comparison matrix as LaTeX tabular.

    Best values are \\textbf{bold}, second-best are \\underline{underlined}.
    Note: metric names AND method names with underscores are escaped automatically.
    """
    headers = matrix["headers"]
    rows = matrix["rows"]
    annotations = matrix["annotations"]

    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{_latex_escape_cell(h)}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]

    for i, row in enumerate(rows):
        cells = []
        # FIX: escape method names too, not just headers
        method_name = _latex_escape_cell(row["method"])
        if row.get("is_proposed"):
            method_name = f"\\textbf{{{method_name}}} (Ours)"
        cells.append(method_name)

        for h in headers[1:]:
            val = row.get(h)
            if val is None:
                cells.append("--")
                continue
            # Format number
            if isinstance(val, float):
                formatted = f"{val:.2f}" if val < 1 else f"{val:.1f}"
            else:
                formatted = str(val)
            # Apply annotation
            ann = annotations.get((i, h))
            if ann == "best":
                formatted = f"\\textbf{{{formatted}}}"
            elif ann == "second":
                formatted = f"\\underline{{{formatted}}}"
            cells.append(formatted)

        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)
```

## 2.5 Integration into Existing `analysis.py`

In the existing `run()` method, **after** the LLM analysis call, add:

```python
# --- NEW: computational analysis ---
from .analysis.statistics import compute_significance_report
from .analysis.training_dynamics import analyze_training_dynamics
from .analysis.ablation_analysis import quantify_ablation_contributions
from .analysis.comparison_matrix import build_comparison_matrix

computational_analysis = {}

# Statistical significance (requires multi-run data)
if multi_run_results:  # list of per-run metric dicts
    sig_reports = {}
    for metric in blueprint.metrics:
        proposed_runs = [r[metric.name] for r in multi_run_results
                         if metric.name in r and isinstance(r[metric.name], (int, float))]
        best_baseline_runs = _get_best_baseline_runs(metric.name, baselines)
        if proposed_runs and best_baseline_runs:
            sig_reports[metric.name] = compute_significance_report(
                proposed_runs, best_baseline_runs,
                metric.name, metric.higher_is_better)
    computational_analysis["significance"] = sig_reports

# Training dynamics
if training_log:
    computational_analysis["training_dynamics"] = analyze_training_dynamics(training_log)

# Ablation contributions
if ablation_results and final_metrics:
    primary = next((m for m in blueprint.metrics if m.primary), None)
    if primary:
        computational_analysis["ablation_contributions"] = quantify_ablation_contributions(
            final_metrics, ablation_results, primary.name, primary.higher_is_better)

# Comparison matrix
if baselines_with_results and proposed_result:
    computational_analysis["comparison_matrix"] = build_comparison_matrix(
        baselines_with_results, proposed_result, blueprint.metrics)
```

> **IMPORTANT**: These are additive changes. Do NOT remove the existing LLM analysis call.
> The computational analysis enriches the LLM's qualitative analysis with quantitative backing.
> The WRITING module should consume both `analysis` (LLM) and `computational_analysis` (code).
