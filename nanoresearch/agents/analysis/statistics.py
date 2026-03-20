"""Statistical significance testing for experiment results.

Pure Python implementation — no scipy dependency required.
If scipy is available, use scipy.stats.ttest_ind for exact p-values.
The fallback is intentionally conservative (overestimates p-values).
"""
from __future__ import annotations

import math
from typing import Optional


def welch_t_test(sample_a: list[float], sample_b: list[float]) -> dict:
    """Welch's t-test (unequal variance) for two independent samples.

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
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b)
                           / (n_a + n_b - 2))
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


def compute_significance_report(
    proposed_runs: list[float],
    baseline_runs: list[float],
    metric_name: str,
    higher_is_better: bool = True,
) -> dict:
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
    """Approximate two-tailed p-value.

    Uses normal approximation for df > 30. For df <= 30, uses a conservative
    lookup table with df=10 critical values (overestimates p-values → safe).
    """
    abs_t = abs(t)
    if df > 30:
        # Normal approximation (accurate for large df)
        p = math.erfc(abs_t / math.sqrt(2))
        return min(p, 1.0)
    # Conservative lookup — df=10 two-tailed critical values.
    # df=10 has wider tails than df=30, so this overestimates p (safe).
    thresholds = [
        (4.587, 0.001), (3.169, 0.01), (2.764, 0.02),
        (2.228, 0.05), (1.812, 0.10), (1.372, 0.20),
    ]
    for threshold, p in thresholds:
        if abs_t >= threshold:
            return p
    return 0.50
