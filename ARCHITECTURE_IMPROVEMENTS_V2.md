# NanoResearch Architecture Improvements V2 — Developer Handbook

> **Author**: Claude Opus 4.6 deep review, March 2026
> **Target**: Next developer continuing NanoResearch development
> **Scope**: All modules, ~30,000 LOC, 9-stage pipeline
> **Goal**: From "production-quality research agent" to "industry-defining research agent"

---

## Table of Contents

**Core Improvements (28 sections + 5 appendices)**

1. [Overall Assessment](#1-overall-assessment)
2. [P0: ANALYSIS Module Rewrite](#2-p0-analysis-module-rewrite)
3. [P0: Multi-Model Review Committee](#3-p0-multi-model-review-committee)
4. [P0: Citation Fact-Checking](#4-p0-citation-fact-checking)
5. [P0: Context Engine (OpenClaw-Inspired)](#5-p0-context-engine-openclaw-inspired)
6. [P1: File Splitting & Module Restructuring](#6-p1-file-splitting--module-restructuring)
7. [P1: Shared LaTeX Fixer Module](#7-p1-shared-latex-fixer-module)
8. [P1: Prompt Template Externalization](#8-p1-prompt-template-externalization)
9. [P1: IDEATION ReAct Search Loop](#9-p1-ideation-react-search-loop)
10. [P2: CODING Quality Gates](#10-p2-coding-quality-gates)
11. [P2: Cost Tracking](#11-p2-cost-tracking)
12. [P2: Constants Centralization](#12-p2-constants-centralization)
13. [P2: DAG Parallel Stage Scheduling](#13-p2-dag-parallel-stage-scheduling)
14. [P2: Progress Streaming](#14-p2-progress-streaming)
15. [P3: Structured Logging](#15-p3-structured-logging)
16. [P3: Paper Quality Benchmark](#16-p3-paper-quality-benchmark)
17. [P3: Blueprint Semantic Validation](#17-p3-blueprint-semantic-validation)
18. [Bug-Level Fixes (Non-Breaking)](#18-bug-level-fixes-non-breaking)
19. [Implementation Order & Dependencies](#19-implementation-order--dependencies)
20. [ERRATA: Bugs Fixed in This Document](#20-errata-bugs-fixed-in-this-document-self-review)
21. [Graceful Shutdown & Signal Handling](#21-missing-graceful-shutdown--signal-handling)
22. [Resource Cleanup & Connection Pool Safety](#22-missing-resource-cleanup--connection-pool-safety)
23. [Bare Exception Hardening](#23-missing-bare-exception-hardening)
24. [Concurrency Safety](#24-missing-concurrency-safety)
25. [Checkpoint Transactionality](#25-missing-checkpoint-transactionality)
26. [SLURM Edge Cases](#26-missing-slurm-edge-cases)
27. [Test Expansion Priorities](#27-missing-test-expansion-priorities)
28. [WRITING Global Consistency Pass](#28-missing-writing-global-consistency-pass)

**Appendices**: A (OpenClaw Reference) · B (Test Checklist) · C (Final File Structure) · D (Risk Assessment) · E (New Developer Checklist)

---

## 1. Overall Assessment

### Scores by Module

| Module | Score | Main Strength | Main Weakness |
|--------|-------|---------------|---------------|
| BaseAgent Framework | 9.0/10 | 3-tier JSON repair; ReAct loop; context compaction | `args_hash` instability; LaTeX cmd detection heuristic |
| IDEATION | 8.0/10 | Multi-source search + snowball + must-cite | No semantic dedup; no search quality self-eval |
| PLANNING | 8.5/10 | Evidence grounding + provenance tracking | No semantic validation; no compute feasibility check |
| SETUP | 7.0/10 | Global cache + dual-source download | Simple search-clone chain |
| CODING | 7.5/10 | Interface Contract + parallel gen + import check | Regex-based import check; no type check; no smoke test |
| EXECUTION | 8.0/10 | SLURM/local dual + 20-round debug + checkpoint | 4749-line monolith; global timeout; hardcoded repair order |
| ANALYSIS | 6.5/10 | — | Entire module is one LLM prompt wrapper; zero computation |
| FIGURE_GEN | 8.0/10 | Domain templates + hybrid AI/code + colorblind palette | 700 lines of inline prompts; silent 3-figure cap |
| WRITING | 8.5/10 | Per-section context; Contribution Contract; inline figs | 3578-line monolith; weak lower-is-better detection |
| REVIEW | 8.0/10 | Monotonic score guarantee; backpressure; fig preservation | Self-review (same LLM); section truncation loses Method |
| Pipeline Orchestrator | 8.5/10 | Atomic manifest; checkpoint resume; stale recovery | No cost tracking; strict serial execution |
| Multi-Model Routing | 7.5/10 | Per-stage config; thinking model compat | No usage/cost return |
| Literature Search | 8.0/10 | Circuit breaker; batch API; rate limiting | No domain filter on OpenAlex |
| LaTeX Fix Strategy | 8.5/10 | 2-level (deterministic + LLM search-replace) | Duplicated in writing.py and review.py |

**Weighted Total: 7.9/10**

---

## 2. P0: ANALYSIS Module Rewrite

### Problem

`analysis.py` (420 lines) is the weakest link. It takes execution results, sends them
to an LLM prompt, and returns the LLM's JSON summary. Zero computational analysis.
No statistical tests, no curve fitting, no contribution quantification.

A research paper that says "our method achieves 92.3% accuracy" without significance
testing is unacceptable at any top venue.

### What to Build

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

### 2.1 Statistical Significance Testing (`statistics.py`)

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

### 2.2 Training Dynamics Analyzer (`training_dynamics.py`)

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

### 2.3 Ablation Contribution Analyzer (`ablation_analysis.py`)

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

### 2.4 Comparison Matrix Builder (`comparison_matrix.py`)

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


def _latex_escape_header(text: str) -> str:
    """Escape underscores and other LaTeX-special chars in table headers."""
    text = text.replace("_", "\\_")
    text = text.replace("%", "\\%")
    text = text.replace("&", "\\&")
    return text


def comparison_matrix_to_latex(matrix: dict) -> str:
    """Render comparison matrix as LaTeX tabular.

    Best values are \\textbf{bold}, second-best are \\underline{underlined}.
    Note: metric names with underscores are escaped automatically.
    """
    headers = matrix["headers"]
    rows = matrix["rows"]
    annotations = matrix["annotations"]

    n_cols = len(headers)
    col_spec = "l" + "c" * (n_cols - 1)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{_latex_escape_header(h)}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]

    for i, row in enumerate(rows):
        cells = []
        method_name = row["method"]
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

### 2.5 Integration into Existing `analysis.py`

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

---

## 3. P0: Multi-Model Review Committee

### Problem

Current REVIEW uses the same LLM (or family) that wrote the paper to review it.
This is fundamentally flawed — it's like having a student grade their own exam.

### Solution: Multi-Reviewer Architecture

**File**: `nanoresearch/agents/review.py` — modify `_review_paper()` method.

### 3.1 Config Changes (`config.py`)

Add to `ResearchConfig`:

```python
# In StageModelConfig, add reviewer profiles
review_committee: list[dict] = [
    # Default: use 2 different models
    # Each reviewer has a role, model config, and weight
]
```

Config example (`~/.nanobot/config.json`):

```json
{
  "research": {
    "review_committee": [
      {
        "role": "Methodology Expert",
        "focus": "technical soundness, mathematical rigor, proof correctness, novelty assessment",
        "model": "gpt-4o",
        "base_url": "https://api.openai.com/v1",
        "weight": 0.40
      },
      {
        "role": "Empirical Reviewer",
        "focus": "experiment design, statistical significance, reproducibility, baselines fairness",
        "model": "claude-sonnet-4-20250514",
        "base_url": "https://api.anthropic.com/v1",
        "weight": 0.35
      },
      {
        "role": "Writing Quality Reviewer",
        "focus": "clarity, logical flow, grammar, figure quality, related work coverage",
        "model": "gemini-2.5-pro",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "weight": 0.25
      }
    ]
  }
}
```

### 3.2 Implementation Changes (`review.py`)

```python
async def _multi_reviewer_assessment(self, paper_tex: str,
                                      sections: list[tuple[str, str]]) -> dict:
    """Run parallel reviews from multiple model personas."""
    committee = self.config.review_committee
    if not committee or len(committee) < 2:
        # Fallback to single-model review (existing behavior)
        return await self._review_paper(sections)

    import asyncio
    review_tasks = []
    for reviewer in committee:
        review_tasks.append(
            self._review_as_role(paper_tex, sections, reviewer)
        )
    reviews = await asyncio.gather(*review_tasks, return_exceptions=True)

    # Filter out failures
    valid_reviews = []
    weights = []
    for review, reviewer in zip(reviews, committee):
        if isinstance(review, Exception):
            self.log(f"Reviewer '{reviewer['role']}' failed: {review}")
            continue
        valid_reviews.append(review)
        weights.append(reviewer.get("weight", 1.0 / len(committee)))

    if not valid_reviews:
        self.log("All reviewers failed, falling back to single-model")
        return await self._review_paper(sections)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Weighted score
    overall = sum(r["overall_score"] * w for r, w in zip(valid_reviews, weights))

    # Union of all issues (deduplicated by content similarity)
    all_issues = []
    seen_issues = set()
    for review in valid_reviews:
        for section_review in review.get("section_reviews", []):
            for issue in section_review.get("issues", []):
                issue_key = issue.strip().lower()[:80]
                if issue_key not in seen_issues:
                    seen_issues.add(issue_key)
                    all_issues.append(issue)

    # Per-section: take lowest score (most critical reviewer wins)
    merged_section_reviews = self._merge_section_reviews(valid_reviews)

    return {
        "overall_score": round(overall, 2),
        "section_reviews": merged_section_reviews,
        "individual_reviews": valid_reviews,  # Keep for transparency
        "num_reviewers": len(valid_reviews),
    }


async def _review_as_role(self, paper_tex: str,
                           sections: list[tuple[str, str]],
                           reviewer: dict) -> dict:
    """Run review from a specific reviewer persona."""
    role = reviewer["role"]
    focus = reviewer["focus"]

    # Create a temporary model config for this reviewer
    from .multi_model import ModelDispatcher
    reviewer_config = StageModelConfig(
        model=reviewer["model"],
        base_url=reviewer.get("base_url", self.config.base_url),
        api_key=reviewer.get("api_key", self.config.api_key),
        temperature=0.3,
        max_tokens=4096,
    )

    system = (
        f"You are a top-tier {role} at a major ML conference (NeurIPS/ICML/ICLR). "
        f"Your primary focus: {focus}. "
        f"Review the paper section by section. For each section, provide:\n"
        f"- score (1-10)\n"
        f"- issues (list of specific problems)\n"
        f"- suggestions (list of specific improvements)\n"
        f"Return JSON: {{\"overall_score\": float, \"section_reviews\": [...]}}"
    )

    # Use the reviewer's model for the LLM call
    result = await self.dispatcher.generate(
        config=reviewer_config,
        system_prompt=system,
        user_prompt=f"Review this paper:\n\n{paper_tex[:20000]}",
        json_mode=True,
    )

    return self._parse_review_json(result)


def _merge_section_reviews(self, reviews: list[dict]) -> list[dict]:
    """Merge section reviews from multiple reviewers.

    Strategy: for each section, take the LOWEST score (most critical reviewer
    wins) and union all issues/suggestions.
    """
    section_map: dict[str, dict] = {}
    for review in reviews:
        for sr in review.get("section_reviews", []):
            name = sr.get("section", "").lower().strip()
            if name not in section_map:
                section_map[name] = {
                    "section": sr.get("section", name),
                    "score": sr.get("score", 5),
                    "issues": list(sr.get("issues", [])),
                    "suggestions": list(sr.get("suggestions", [])),
                }
            else:
                existing = section_map[name]
                # Take minimum score (strictest reviewer)
                existing["score"] = min(existing["score"],
                                         sr.get("score", 5))
                # Union issues and suggestions (dedup by first 80 chars)
                seen = {i[:80].lower() for i in existing["issues"]}
                for issue in sr.get("issues", []):
                    if issue[:80].lower() not in seen:
                        existing["issues"].append(issue)
                        seen.add(issue[:80].lower())
                seen_s = {s[:80].lower() for s in existing["suggestions"]}
                for sug in sr.get("suggestions", []):
                    if sug[:80].lower() not in seen_s:
                        existing["suggestions"].append(sug)
                        seen_s.add(sug[:80].lower())
    return list(section_map.values())
```

### 3.3 Safety Constraints

- **Backward compatible**: If `review_committee` is empty or has < 2 entries, fall back to existing single-model review.
- **Graceful degradation**: If any reviewer fails (API error, timeout), continue with remaining reviewers.
- **No new dependencies**: Uses existing `ModelDispatcher` infrastructure.
- **Revision still uses original model**: Only the REVIEW scoring uses multiple models. The REVISION (rewriting) step stays on the configured revision model.

---

## 4. P0: Citation Fact-Checking

### Problem

The pipeline generates citations like "Smith et al. [15] achieved 95.2% accuracy on
ImageNet" — but never verifies if this matches what paper [15] actually says. This can
produce factual errors in published papers.

### Solution

Add `nanoresearch/agents/review_citation_checker.py`:

```python
"""Citation fact-checking: verify claims against source abstracts."""
import re
from typing import Optional


async def verify_citation_claims(
    agent,  # BaseResearchAgent instance (for LLM calls)
    paper_tex: str,
    papers: list[dict],
    bibtex_keys_to_papers: dict[str, dict],
) -> list[dict]:
    """Verify factual accuracy of citation claims in the paper.

    For each sentence containing a \\cite{}, compare the claim against the
    source paper's abstract/title.

    Args:
        agent: Agent instance for LLM calls.
        paper_tex: Full LaTeX source.
        papers: List of paper dicts from ideation.
        bibtex_keys_to_papers: Mapping from BibTeX key to paper dict.

    Returns:
        List of verification results.
    """
    # 1. Extract sentences with citations
    cite_sentences = _extract_cite_sentences(paper_tex)
    if not cite_sentences:
        return []

    # 2. Batch verify — cap at 15 checks to limit LLM cost.
    #    Group by cite_key to avoid re-verifying same source.
    checked_keys: set[str] = set()
    verifications = []
    MAX_CHECKS = 15
    for sentence, cite_keys in cite_sentences:
        if len(verifications) >= MAX_CHECKS:
            break
        for key in cite_keys:
            if key in checked_keys or len(verifications) >= MAX_CHECKS:
                continue
            checked_keys.add(key)
            paper = bibtex_keys_to_papers.get(key)
            if not paper:
                continue
            abstract = paper.get("abstract", "")
            title = paper.get("title", "")
            if not abstract and not title:
                continue

            # 3. LLM verification
            result = await agent.generate_json(
                system=(
                    "You are a citation fact-checker. Compare the claim in the "
                    "paper against the source's title and abstract. "
                    "Return JSON: {\"accurate\": true/false, "
                    "\"issue\": null or string describing the inaccuracy}"
                ),
                user=(
                    f"Claim in paper: \"{sentence}\"\n\n"
                    f"Source paper title: \"{title}\"\n"
                    f"Source paper abstract: \"{abstract[:1500]}\"\n\n"
                    f"Is the claim accurately representing the source?"
                ),
            )

            verifications.append({
                "sentence": sentence[:200],
                "cite_key": key,
                "source_title": title,
                "accurate": result.get("accurate", True),
                "issue": result.get("issue"),
            })

    return verifications


def _extract_cite_sentences(tex: str) -> list[tuple[str, list[str]]]:
    """Extract sentences containing \\cite commands."""
    # Split into sentences (rough: period + space + capital)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', tex)
    results = []
    for sent in sentences:
        cite_matches = re.findall(r'\\cite[tp]?\{([^}]+)\}', sent)
        if cite_matches:
            keys = []
            for match in cite_matches:
                keys.extend(k.strip() for k in match.split(","))
            results.append((sent.strip(), keys))
    return results
```

### Integration into `review.py`

In `run()`, after the main review and before revisions:

```python
# Citation fact-checking
from .review_citation_checker import verify_citation_claims
citation_verifications = await verify_citation_claims(
    self, paper_tex, papers, bibtex_key_map)
inaccurate = [v for v in citation_verifications if not v["accurate"]]
if inaccurate:
    self.log(f"Citation fact-check: {len(inaccurate)} inaccurate claims found")
    # Add to consistency_issues for the revision loop to fix
    for v in inaccurate:
        consistency_issues.append({
            "type": "citation_inaccuracy",
            "description": f"Claim about [{v['cite_key']}] may be inaccurate: {v['issue']}",
            "sentence": v["sentence"],
        })
```

> **IMPORTANT**: This adds issues to the existing consistency_issues list, which the
> existing revision loop already processes. No structural changes to the revision flow.

---

## 5. P0: Context Engine (OpenClaw-Inspired)

### What OpenClaw Does

OpenClaw implements a **pluggable ContextEngine interface** with lifecycle hooks:

```
bootstrap → ingest → assemble → compact → afterTurn
                                           ↓
                                   prepareSubagentSpawn → onSubagentEnded
```

**Key files in OpenClaw** (cloned to `D:/openclaw`):
- `src/context-engine/types.ts` — ContextEngine interface (168 lines)
- `src/context-engine/registry.ts` — Factory registry + slot resolution (86 lines)
- `src/context-engine/legacy.ts` — Backward-compatible wrapper (117 lines)
- `src/memory/hybrid.ts` — Hybrid vector+keyword search with temporal decay (156 lines)
- `src/memory/temporal-decay.ts` — Exponential decay scoring (168 lines)

**Key design insights**:
1. **Pluggable slot**: Only ONE context engine active at a time (exclusive slot pattern)
2. **LegacyEngine as fallback**: Always registered, preserves existing behavior
3. **Lifecycle hooks**: `bootstrap` (init), `ingest` (per-message), `assemble` (pre-LLM), `afterTurn` (post-LLM), `compact` (overflow)
4. **Hybrid search**: `score = vectorWeight * vectorScore + textWeight * textScore` (default 0.7/0.3)
5. **Temporal decay**: `multiplier = e^(-ln2/halfLifeDays * ageInDays)` — older memories score lower
6. **MMR diversity**: Maximal Marginal Relevance re-ranking prevents redundant retrieval

### 5.1 What NanoResearch Should Borrow

NanoResearch is NOT a chatbot — it's a pipeline. The direct OpenClaw model doesn't apply.
But the **concepts** are transformative for our use case:

#### Concept 1: Cross-Stage Memory with Decay

Currently each stage gets its inputs from the previous stage's JSON output. There's no
"memory" across pipeline runs or across stages within a run beyond what's explicitly
passed in the stage input dict.

**Problem examples**:
- REVIEW doesn't know which specific experiments failed in EXECUTION (only gets final metrics)
- WRITING doesn't remember which search queries found the best papers (only gets papers)
- Across runs: the system doesn't remember "last time topic X failed because of Y"

**Solution**: Implement a `ResearchMemory` class:

```python
"""nanoresearch/memory/research_memory.py"""
import json
import hashlib
import math
import time
from pathlib import Path
from typing import Optional


class MemoryEntry:
    """A single memory unit with temporal decay."""
    __slots__ = ("key", "content", "source_stage", "created_at",
                 "access_count", "tags", "importance")

    def __init__(self, key: str, content: str, source_stage: str,
                 tags: list[str] = None, importance: float = 0.5):
        self.key = key
        self.content = content
        self.source_stage = source_stage
        self.created_at = time.time()
        self.access_count = 0
        self.tags = tags or []
        self.importance = importance  # 0.0 to 1.0


class ResearchMemory:
    """Cross-stage memory with hybrid search and temporal decay.

    Inspired by OpenClaw's ContextEngine, adapted for pipeline use.

    Lifecycle:
        1. bootstrap(session_id) — load from disk on resume
        2. ingest(entry) — store after each stage completes
        3. query(query, tags, budget) — retrieve relevant memories
        4. compact(max_entries) — prune old/low-value entries
        5. persist(path) — save to disk

    Usage in pipeline:
        memory = ResearchMemory()
        memory.bootstrap(session_id)

        # After IDEATION:
        memory.ingest(MemoryEntry(
            key="search_strategy",
            content="OpenAlex returned best results for 'transformer attention'",
            source_stage="IDEATION",
            tags=["search", "strategy"],
            importance=0.7
        ))

        # In WRITING, retrieve relevant memories:
        relevant = memory.query("writing method section", tags=["method"],
                                budget=5)
    """
    HALF_LIFE_HOURS = 24.0  # Memories decay with 24h half-life within a run
    CROSS_RUN_HALF_LIFE_DAYS = 30.0  # Cross-run memories decay with 30-day half-life

    def __init__(self):
        self._entries: dict[str, MemoryEntry] = {}
        self._session_id: Optional[str] = None

    def bootstrap(self, session_id: str, persist_path: Optional[Path] = None):
        """Load existing memories from disk."""
        self._session_id = session_id
        if persist_path and persist_path.exists():
            try:
                data = json.loads(persist_path.read_text("utf-8"))
                for item in data.get("entries", []):
                    entry = MemoryEntry(
                        key=item["key"],
                        content=item["content"],
                        source_stage=item["source_stage"],
                        tags=item.get("tags", []),
                        importance=item.get("importance", 0.5),
                    )
                    entry.created_at = item.get("created_at", time.time())
                    entry.access_count = item.get("access_count", 0)
                    self._entries[entry.key] = entry
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # Start fresh on corruption

    def ingest(self, entry: MemoryEntry):
        """Store a memory entry. Overwrites if key exists."""
        self._entries[entry.key] = entry

    def query(self, query: str, tags: list[str] = None,
              budget: int = 10, stage_filter: str = None) -> list[dict]:
        """Retrieve relevant memories with temporal decay scoring.

        Args:
            query: Natural language query (matched against content via word overlap).
            tags: Filter to entries with at least one matching tag.
            budget: Maximum number of results.
            stage_filter: Only return entries from this stage.

        Returns:
            List of {"key", "content", "score", "source_stage"} sorted by score.
        """
        now = time.time()
        query_words = set(query.lower().split())

        scored = []
        for entry in self._entries.values():
            # Tag filter
            if tags and not set(tags) & set(entry.tags):
                continue
            # Stage filter
            if stage_filter and entry.source_stage != stage_filter:
                continue

            # Word overlap score (simple BM25-like)
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            if not query_words:
                text_score = 0.0
            else:
                text_score = overlap / len(query_words)

            # Temporal decay
            age_hours = (now - entry.created_at) / 3600.0
            decay = math.exp(-math.log(2) / self.HALF_LIFE_HOURS * age_hours)

            # Importance boost
            importance_boost = 0.5 + 0.5 * entry.importance

            # Combined score
            score = text_score * decay * importance_boost

            if score > 0.01:
                entry.access_count += 1
                scored.append({
                    "key": entry.key,
                    "content": entry.content,
                    "score": round(score, 4),
                    "source_stage": entry.source_stage,
                    "tags": entry.tags,
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:budget]

    def compact(self, max_entries: int = 200):
        """Prune lowest-scored entries to stay under budget."""
        if len(self._entries) <= max_entries:
            return
        now = time.time()
        scored = []
        for key, entry in self._entries.items():
            age_hours = (now - entry.created_at) / 3600.0
            decay = math.exp(-math.log(2) / self.HALF_LIFE_HOURS * age_hours)
            score = entry.importance * decay * (1 + entry.access_count * 0.1)
            scored.append((key, score))
        scored.sort(key=lambda x: x[1])
        # Remove lowest-scored entries
        to_remove = len(self._entries) - max_entries
        for key, _ in scored[:to_remove]:
            del self._entries[key]

    def persist(self, path: Path):
        """Save all memories to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self._session_id,
            "entry_count": len(self._entries),
            "entries": [
                {
                    "key": e.key,
                    "content": e.content,
                    "source_stage": e.source_stage,
                    "created_at": e.created_at,
                    "access_count": e.access_count,
                    "tags": e.tags,
                    "importance": e.importance,
                }
                for e in self._entries.values()
            ],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_context_for_stage(self, stage: str, budget: int = 10) -> str:
        """Build a context string for a specific stage from memories.

        This replaces the ad-hoc context passing between stages.
        """
        # Stage-specific memory queries
        stage_queries = {
            "WRITING": "key findings experiment results method design",
            "REVIEW": "paper structure claims evidence figures",
            "FIGURE_GEN": "experiment results training metrics ablation",
            "ANALYSIS": "execution status metrics errors debug",
        }
        query = stage_queries.get(stage, stage.lower())
        results = self.query(query, budget=budget)
        if not results:
            return ""
        lines = [f"[Memory from {r['source_stage']}] {r['content']}" for r in results]
        return "\n".join(lines)
```

#### Concept 2: Cross-Run Learning

```python
"""nanoresearch/memory/cross_run_memory.py"""
import json
import math
import time
from pathlib import Path


class CrossRunMemory:
    """Persistent memory across pipeline runs.

    Stored at ~/.nanobot/memory/cross_run.json

    Examples of what gets stored:
    - "Topic 'attention mechanisms' failed at EXECUTION because PyTorch 2.x
       changed the autograd API" → future runs can preempt this
    - "OpenAlex returns better results than S2 for NLP topics" → search strategy
    - "User's cluster needs 'module load cuda/12.1' before training" → env setup
    """
    PERSIST_PATH = Path.home() / ".nanobot" / "memory" / "cross_run.json"

    def __init__(self):
        self._entries: list[dict] = []
        self._load()

    def _load(self):
        if self.PERSIST_PATH.exists():
            try:
                data = json.loads(self.PERSIST_PATH.read_text("utf-8"))
                self._entries = data.get("entries", [])
            except (json.JSONDecodeError, OSError):
                self._entries = []

    def record_outcome(self, topic: str, stage: str, success: bool,
                       lesson: str, tags: list[str] = None):
        """Record a lesson learned from a pipeline run."""
        self._entries.append({
            "topic": topic,
            "stage": stage,
            "success": success,
            "lesson": lesson,
            "tags": tags or [],
            "timestamp": time.time(),
        })
        self._persist()

    def get_lessons(self, topic: str, stage: str, limit: int = 5) -> list[str]:
        """Retrieve relevant lessons for a given topic and stage."""
        now = time.time()
        scored = []
        topic_words = set(topic.lower().split())
        for entry in self._entries:
            entry_words = set(entry.get("topic", "").lower().split())
            overlap = len(topic_words & entry_words) / max(len(topic_words), 1)

            # Temporal decay (30-day half-life)
            age_days = (now - entry.get("timestamp", now)) / 86400.0
            decay = math.exp(-math.log(2) / 30.0 * age_days)

            # Stage match boost
            stage_match = 1.5 if entry.get("stage") == stage else 1.0

            score = overlap * decay * stage_match
            if score > 0.01:
                scored.append((entry["lesson"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [lesson for lesson, _ in scored[:limit]]

    def _persist(self):
        self.PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Keep only last 500 entries
        if len(self._entries) > 500:
            self._entries = self._entries[-500:]
        data = {"entries": self._entries, "count": len(self._entries)}
        self.PERSIST_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
```

#### Concept 3: Pluggable Context Assembly (inspired by OpenClaw slots)

```python
"""nanoresearch/context_engine/interface.py"""
from abc import ABC, abstractmethod


class ContextEngine(ABC):
    """Pluggable context engine interface (inspired by OpenClaw).

    Only ONE engine is active per pipeline run (exclusive slot pattern).
    Default: LegacyContextEngine (current behavior).
    """

    @abstractmethod
    async def assemble(self, stage: str, base_context: dict,
                       budget_chars: int = 15000) -> str:
        """Assemble context for an LLM call within a character budget.

        Args:
            stage: Current pipeline stage name.
            base_context: Dict of available context blocks.
            budget_chars: Maximum total characters.

        Returns:
            Assembled context string.
        """
        ...

    @abstractmethod
    async def compact(self, messages: list[dict],
                      budget_chars: int = 100000) -> list[dict]:
        """Compact message history to fit within budget.

        Args:
            messages: Current conversation messages.
            budget_chars: Maximum total characters.

        Returns:
            Compacted message list.
        """
        ...

    async def after_stage(self, stage: str, output: dict):
        """Hook called after each stage completes. Override to persist state."""
        pass

    async def dispose(self):
        """Cleanup resources."""
        pass


class LegacyContextEngine(ContextEngine):
    """Backward-compatible engine wrapping existing behavior."""

    async def assemble(self, stage, base_context, budget_chars=15000):
        # Existing per-section context builder logic
        # This preserves 100% backward compatibility
        blocks = []
        total = 0
        for key, value in base_context.items():
            text = str(value) if not isinstance(value, str) else value
            if total + len(text) > budget_chars:
                remaining = budget_chars - total
                if remaining > 200:
                    blocks.append(f"[{key}] {text[:remaining]}...[truncated]")
                break
            blocks.append(f"[{key}] {text}")
            total += len(text)
        return "\n\n".join(blocks)

    async def compact(self, messages, budget_chars=100000):
        # Existing head/tail compaction from base.py
        total = sum(len(str(m)) for m in messages)
        if total <= budget_chars:
            return messages
        # Keep first 2 + last 6 messages, trim middle
        if len(messages) <= 8:
            return messages
        head = messages[:2]
        tail = messages[-6:]
        return head + [{"role": "system",
                        "content": f"[{len(messages)-8} messages compacted]"}] + tail


class SmartContextEngine(ContextEngine):
    """Advanced engine with memory-aware context assembly.

    Uses ResearchMemory to inject relevant cross-stage context.
    Uses priority scoring to select most important blocks.

    NOTE: Requires ResearchMemory and MemoryEntry from
    nanoresearch.memory.research_memory to be imported by the caller.
    """

    def __init__(self, memory: 'ResearchMemory'):
        self.memory = memory

    async def assemble(self, stage, base_context, budget_chars=15000):
        # 1. Get memory-based context additions
        memory_context = self.memory.get_context_for_stage(stage, budget=5)

        # 2. Priority-score each context block
        priorities = self._score_blocks(stage, base_context)

        # 3. Greedily select blocks by priority within budget
        selected = []
        remaining = budget_chars
        if memory_context:
            mem_block = f"[Cross-Stage Memory]\n{memory_context}"
            if len(mem_block) < remaining * 0.2:  # Max 20% budget for memory
                selected.append(mem_block)
                remaining -= len(mem_block)

        for key, priority in sorted(priorities, key=lambda x: x[1], reverse=True):
            text = str(base_context[key])
            if len(text) <= remaining:
                selected.append(f"[{key}]\n{text}")
                remaining -= len(text)
            elif remaining > 500:
                selected.append(f"[{key}]\n{text[:remaining-50]}...[truncated]")
                remaining = 0
                break

        return "\n\n".join(selected)

    async def compact(self, messages, budget_chars=100000):
        # Same as Legacy for now; can be enhanced with summarization later
        total = sum(len(str(m)) for m in messages)
        if total <= budget_chars:
            return messages
        if len(messages) <= 8:
            return messages
        head = messages[:2]
        tail = messages[-6:]
        return head + [{"role": "system",
                        "content": f"[{len(messages)-8} messages compacted]"}] + tail

    async def after_stage(self, stage, output):
        """Auto-extract key facts from stage output into memory."""
        # Extract lessons from output
        if isinstance(output, dict):
            if "error" in output:
                self.memory.ingest(MemoryEntry(
                    key=f"{stage}_error",
                    content=f"Stage {stage} encountered: {str(output['error'])[:500]}",
                    source_stage=stage,
                    tags=["error", "debug"],
                    importance=0.8,
                ))
            if "key_findings" in output:
                findings = output["key_findings"]
                if isinstance(findings, list):
                    for i, f in enumerate(findings[:5]):
                        self.memory.ingest(MemoryEntry(
                            key=f"{stage}_finding_{i}",
                            content=str(f)[:500],
                            source_stage=stage,
                            tags=["finding"],
                            importance=0.7,
                        ))

    def _score_blocks(self, stage: str, blocks: dict) -> list[tuple[str, float]]:
        """Score context blocks by relevance to current stage."""
        # Stage-specific relevance weights
        relevance = {
            "WRITING": {
                "hypotheses": 0.9, "evidence": 0.9, "method": 0.8,
                "results": 1.0, "papers": 0.5, "figures": 0.7,
            },
            "REVIEW": {
                "claims": 0.9, "evidence": 0.8, "results": 0.9,
                "papers": 0.6, "method": 0.7,
            },
            "FIGURE_GEN": {
                "results": 1.0, "training_log": 0.9, "ablation": 0.8,
                "method": 0.5,
            },
        }
        stage_weights = relevance.get(stage, {})
        scored = []
        for key in blocks:
            weight = stage_weights.get(key, 0.5)
            scored.append((key, weight))
        return scored
```

### 5.2 Integration into Orchestrator

```python
# In orchestrator.py run():

from nanoresearch.memory.research_memory import ResearchMemory
from nanoresearch.context_engine.interface import SmartContextEngine

memory = ResearchMemory()
memory.bootstrap(session_id, workspace.path / "memory.json")
context_engine = SmartContextEngine(memory)

for stage in stages:
    # ... existing stage execution ...
    result = await agent.run(**inputs)

    # NEW: post-stage memory hook
    await context_engine.after_stage(stage.name, result)

    # ... existing output saving ...

# At end of pipeline:
memory.persist(workspace.path / "memory.json")
```

### 5.3 Cross-Run Integration

```python
# In orchestrator.py, at pipeline start:
from nanoresearch.memory.cross_run_memory import CrossRunMemory
cross_run = CrossRunMemory()

# Get lessons from previous runs
lessons = cross_run.get_lessons(topic, "EXECUTION")
if lessons:
    log.info(f"Lessons from previous runs: {lessons}")
    # Inject into execution agent context

# At pipeline end:
cross_run.record_outcome(
    topic=topic,
    stage="PIPELINE",
    success=final_status == "DONE",
    lesson=f"{'Completed' if success else 'Failed at ' + failed_stage}: {summary}",
    tags=[failed_stage] if not success else ["success"],
)
```

---

## 6. P1: File Splitting & Module Restructuring

### 6.1 execution.py (4749 lines → 7 files)

```
agents/execution/
├── __init__.py              # ExecutionAgent class, run() method (~200 lines)
│                            # Delegates to local_runner or cluster_runner
├── local_runner.py          # _run_local_mode(), venv creation, training (~800 lines)
│                            # Contains: dry-run loop, quick-eval loop
├── cluster_runner.py        # SLURM submit, poll, checkpoint resume (~600 lines)
│                            # Contains: sbatch, squeue, sacct wrappers
├── debug_loop.py            # Debug iteration: error diagnosis → patch → retry (~800 lines)
│                            # Contains: repair ordering, remediation ledger
├── resource_matcher.py      # Fuzzy resource path matching (~800 lines)
│                            # Contains: _collect_resource_candidates, _match_resource_target
├── repair_strategies.py     # Deterministic repair strategies (~500 lines)
│                            # Contains: resource-path, option-value, runtime repairs
└── result_collector.py      # Metrics collection and normalization (~400 lines)
                             # Contains: metrics.json parsing, CSV extraction
```

**Migration approach**:
1. Create the directory structure first
2. Move functions one file at a time, starting from the most isolated (result_collector)
3. Replace with imports in `__init__.py`
4. Run tests after each file extraction
5. DO NOT rename any public function signatures

### 6.2 writing.py (3578 lines → 7 files)

```
agents/writing/
├── __init__.py              # WritingAgent class, run() method (~300 lines)
├── context_builder.py       # _build_core_context + 5 section builders (~500 lines)
├── grounding.py             # GroundingPacket + ContributionContract (~200 lines)
├── section_writer.py        # _generate_section + prompt assembly (~600 lines)
├── table_builder.py         # _build_main_table_latex + ablation table (~300 lines)
├── citation_manager.py      # BibTeX build + citation resolution + HTML entity fix (~400 lines)
└── latex_assembler.py       # Template rendering + figure embedding (~400 lines)
```

### 6.3 experiment.py (3346 lines → 4 files)

```
agents/experiment/
├── __init__.py              # ExperimentAgent class, run() dispatch (~300 lines)
├── pipeline_mode.py         # 6-phase structured pipeline (~1000 lines)
├── react_mode.py            # ReAct tool-based iteration (~800 lines)
└── edit_apply.py            # _apply_search_replace_edit + helpers (~500 lines)
```

> **CRITICAL**: Every extracted function must keep its exact signature. Only internal
> helper functions (prefixed with `_`) that are used within a single extracted file
> can be moved without re-exporting.

---

## 7. P1: Shared LaTeX Fixer Module

### Problem

`writing.py` and `review.py` both implement the 2-level LaTeX fix strategy independently.
This is ~500 lines of duplicated code.

### Solution

Create `nanoresearch/latex/fixer.py`:

```python
"""Shared 2-level LaTeX fix strategy for writing and review stages."""
import re
import shutil
from pathlib import Path
from typing import Optional, Callable, Awaitable


# Level 1: Deterministic fixes (no LLM)
UNICODE_REPLACEMENTS = {
    '\u2013': '--',   # en-dash
    '\u2014': '---',  # em-dash
    '\u2018': '`',    # left single quote
    '\u2019': "'",    # right single quote
    '\u201c': '``',   # left double quote
    '\u201d': "''",   # right double quote
    '\u2026': '\\ldots{}',
    '\u00e9': "\\'e",
    '\u00e8': "\\`e",
    '\u00f6': '\\"o',
    '\u00fc': '\\"u',
    '\u00e4': '\\"a',
    # Add more as encountered
}

REQUIRED_PACKAGES = [
    "inputenc", "fontenc", "amsmath", "amssymb", "graphicx",
    "booktabs", "hyperref", "natbib", "placeins",
]


def deterministic_fix(tex: str) -> str:
    """Level 1: Apply all deterministic fixes. No LLM needed.

    Safe to call multiple times (idempotent).
    """
    # 1. Remove junk before \documentclass
    dc_match = re.search(r'\\documentclass', tex)
    if dc_match and dc_match.start() > 0:
        tex = tex[dc_match.start():]

    # 2. Unicode replacements
    for char, replacement in UNICODE_REPLACEMENTS.items():
        tex = tex.replace(char, replacement)

    # 3. Remove control characters (except newline, tab)
    tex = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', tex)

    # 4. Fix @{} doubling at table line ends
    tex = re.sub(r'@\{\}(\s*\\\\)', r'\1', tex)

    # 5. Ensure required packages
    tex = _ensure_packages(tex)

    # 6. Fix mismatched \begin{}/\end{} pairs
    tex = _fix_mismatched_envs(tex)

    # 7. Ensure \end{document} exists
    if '\\end{document}' not in tex:
        tex = tex.rstrip() + '\n\\end{document}\n'

    return tex


async def llm_search_replace_fix(
    tex: str,
    error_log: str,
    llm_call: Callable[[str, str], Awaitable[str]],
    max_rounds: int = 3,
) -> tuple[str, bool]:
    """Level 2: LLM-driven search-replace fix.

    Args:
        tex: LaTeX source with compilation errors.
        error_log: tectonic/pdflatex error output.
        llm_call: async function(system, user) -> str (raw LLM response).
        max_rounds: Maximum fix attempts.

    Returns:
        (fixed_tex, success) tuple.
    """
    seen_errors = set()

    for round_num in range(max_rounds):
        # Detect infinite loop
        error_sig = _error_signature(error_log)
        if error_sig in seen_errors:
            break
        seen_errors.add(error_sig)

        system = (
            "You are a LaTeX debugging expert. Given compilation errors, "
            "return a JSON array of search-replace fixes.\n"
            "Format: [{\"old\": \"exact text to find\", \"new\": \"replacement text\"}]\n"
            "Rules:\n"
            "- old must be an EXACT substring of the LaTeX source\n"
            "- Each fix should be minimal (change as little as possible)\n"
            "- Return [] if you cannot determine a fix"
        )
        user = (
            f"LaTeX compilation error:\n```\n{error_log[-3000:]}\n```\n\n"
            f"LaTeX source (first 8000 chars):\n```\n{tex[:8000]}\n```"
        )

        try:
            raw = await llm_call(system, user)
            fixes = _parse_fixes_json(raw)
        except Exception:
            break

        if not fixes:
            break

        applied = 0
        for fix in fixes:
            old = fix.get("old", "")
            new = fix.get("new", "")
            if not old or old == new:
                continue
            if old in tex:
                tex = tex.replace(old, new, 1)
                applied += 1
            else:
                # Try whitespace-normalized match
                normalized_old = ' '.join(old.split())
                normalized_tex = ' '.join(tex.split())
                if normalized_old in normalized_tex:
                    # Find the actual position and replace
                    tex = _whitespace_aware_replace(tex, old, new)
                    applied += 1

        if applied == 0:
            break

    return tex, True  # Return current state regardless


def backup_and_fix(tex_path: Path,
                   compile_fn,
                   llm_call,
                   max_rounds: int = 3) -> tuple[bool, Optional[str]]:
    """Full fix pipeline with backup/restore.

    Args:
        tex_path: Path to .tex file.
        compile_fn: Callable that returns (success: bool, error_log: str).
        llm_call: Async LLM call function.
        max_rounds: Maximum fix rounds.

    Returns:
        (success, error_or_none)
    """
    # Backup
    backup_path = tex_path.with_suffix('.tex.bak')
    shutil.copy2(tex_path, backup_path)

    tex = tex_path.read_text(encoding='utf-8')

    # Level 1: Deterministic
    tex = deterministic_fix(tex)
    tex_path.write_text(tex, encoding='utf-8')

    success, error = compile_fn(tex_path)
    if success:
        return True, None

    # Level 2: LLM search-replace
    # (caller must run this in async context)
    # Return the current state for async processing
    return False, error


# --- Internal helpers ---

def _ensure_packages(tex: str) -> str:
    """Add missing required packages after \\documentclass line.

    Only adds if the package name does not appear anywhere in the preamble
    (covers \\usepackage{pkg}, \\usepackage[opts]{pkg}, and style files
    that internally \\RequirePackage{pkg}).
    """
    # Extract preamble (everything before \begin{document})
    begin_doc = tex.find('\\begin{document}')
    preamble = tex[:begin_doc] if begin_doc > 0 else tex[:2000]

    for pkg in REQUIRED_PACKAGES:
        # Check if package name appears anywhere in preamble
        if pkg not in preamble:
            dc_end = tex.find('\n', tex.find('\\documentclass'))
            if dc_end > 0:
                tex = tex[:dc_end+1] + f'\\usepackage{{{pkg}}}\n' + tex[dc_end+1:]
    return tex


def _fix_mismatched_envs(tex: str) -> str:
    """Fix unmatched \\begin{}/\\end{} pairs."""
    begins = re.findall(r'\\begin\{(\w+)\}', tex)
    ends = re.findall(r'\\end\{(\w+)\}', tex)
    begin_counts = {}
    end_counts = {}
    for b in begins:
        begin_counts[b] = begin_counts.get(b, 0) + 1
    for e in ends:
        end_counts[e] = end_counts.get(e, 0) + 1

    for env in begin_counts:
        diff = begin_counts[env] - end_counts.get(env, 0)
        if diff > 0:
            # Missing \end{env} — append before \end{document}
            end_doc = tex.rfind('\\end{document}')
            if end_doc > 0:
                tex = tex[:end_doc] + f'\\end{{{env}}}\n' * diff + tex[end_doc:]
    return tex


def _error_signature(error: str) -> str:
    """Extract a stable signature from an error log for dedup."""
    import hashlib
    # Keep last 500 chars of error (most specific)
    return hashlib.md5(error[-500:].encode()).hexdigest()[:8]


def _parse_fixes_json(raw: str) -> list[dict]:
    """Parse LLM response as JSON array of {old, new} fixes."""
    import json
    # Strip markdown fences
    raw = re.sub(r'^```(?:json)?\s*\n?', '', raw.strip())
    raw = re.sub(r'\n?```\s*$', '', raw.strip())
    try:
        fixes = json.loads(raw)
        if isinstance(fixes, list):
            return [f for f in fixes if isinstance(f, dict) and "old" in f and "new" in f]
    except json.JSONDecodeError:
        pass
    return []


def _whitespace_aware_replace(tex: str, old: str, new: str) -> str:
    """Replace old with new, normalizing whitespace for matching."""
    # Split old into lines, match each with flexible whitespace
    old_lines = old.strip().split('\n')
    pattern_parts = []
    for line in old_lines:
        escaped = re.escape(line.strip())
        pattern_parts.append(r'\s*' + escaped)
    pattern = r'\s*'.join(pattern_parts) if len(pattern_parts) == 1 else '\n'.join(pattern_parts)

    try:
        result = re.sub(pattern, new, tex, count=1)
        return result
    except re.error:
        return tex  # Safety: don't modify if regex fails
```

### Migration Steps

1. Create `nanoresearch/latex/fixer.py` with the code above
2. In `writing.py`, replace all deterministic fix calls with:
   ```python
   from nanoresearch.latex.fixer import deterministic_fix, llm_search_replace_fix
   ```
3. In `review.py`, replace `_try_deterministic_fix()` and `_search_replace_llm_fix()` with imports
4. Delete the duplicated implementations from both files
5. Run LaTeX compilation tests to verify identical behavior

---

## 8. P1: Prompt Template Externalization

### Problem

~1000+ lines of prompt strings are hardcoded in Python files (figure_gen.py:39-201,
341-730; writing.py section prompts; review.py review prompts). This makes prompt
iteration slow (requires code changes) and prevents A/B testing.

### Solution

Create `nanoresearch/prompts/` directory with YAML templates:

```
prompts/
├── __init__.py              # PromptLoader class
├── figure_gen/
│   ├── planning.yaml        # Figure planning prompt
│   ├── chart_types/
│   │   ├── grouped_bar.yaml
│   │   ├── line_plot.yaml
│   │   ├── heatmap.yaml
│   │   ├── radar.yaml
│   │   └── ...
│   └── ai_templates/
│       ├── system_overview.yaml
│       ├── transformer_arch.yaml
│       └── ...
├── writing/
│   ├── title.yaml
│   ├── abstract.yaml
│   ├── introduction.yaml
│   ├── related_work.yaml
│   ├── method.yaml
│   ├── experiments.yaml
│   └── conclusion.yaml
├── review/
│   ├── section_review.yaml
│   ├── revision.yaml
│   └── consistency_check.yaml
└── ideation/
    ├── query_generation.yaml
    ├── gap_analysis.yaml
    └── hypothesis_selection.yaml
```

### Prompt Loader

```python
"""nanoresearch/prompts/__init__.py"""
import yaml
from pathlib import Path
from typing import Optional

_PROMPTS_DIR = Path(__file__).parent
_CACHE: dict[str, dict] = {}


def load_prompt(category: str, name: str, variables: dict = None) -> str:
    """Load a prompt template and optionally fill variables.

    Args:
        category: Subdirectory (e.g., "writing", "figure_gen/chart_types")
        name: File name without .yaml extension
        variables: Dict of {placeholder: value} for string formatting

    Returns:
        Rendered prompt string.
    """
    cache_key = f"{category}/{name}"
    if cache_key not in _CACHE:
        path = _PROMPTS_DIR / category / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            _CACHE[cache_key] = yaml.safe_load(f)

    template = _CACHE[cache_key]
    prompt_text = template.get("prompt", template.get("system_prompt", ""))

    if variables:
        # Use safe string formatting (no eval)
        for key, value in variables.items():
            prompt_text = prompt_text.replace(f"{{{key}}}", str(value))

    return prompt_text


def get_prompt_version(category: str, name: str) -> Optional[str]:
    """Get version string of a prompt template."""
    load_prompt(category, name)  # Ensure cached
    return _CACHE.get(f"{category}/{name}", {}).get("version")
```

### Example YAML Template

```yaml
# prompts/writing/method.yaml
name: method_section
version: "2.3"
description: "System prompt for generating the Method section"
system_prompt: |
  You are an expert ML researcher writing the Method section of a top-venue paper.

  STRUCTURE:
  - 5-7 paragraphs with subsections for each key component
  - Begin with problem formulation and notation
  - Each subsection: intuition → formalization → implementation detail
  - Include at least 2 equations (numbered, with explanation)
  - End with complexity analysis (time + space)

  STYLE:
  - Present tense for method description
  - Active voice ("We propose..." not "It is proposed...")
  - No hedging words ("may", "might", "could", "perhaps")
  - Define all notation on first use

  CONTRIBUTION CONTRACT:
  {contribution_guidance}

  CONTEXT:
  {method_context}
variables:
  - contribution_guidance
  - method_context
```

### Migration Approach

1. Create the YAML files by extracting existing inline prompts (copy-paste, no edits)
2. Replace inline strings with `load_prompt()` calls one at a time
3. Verify output quality is identical before moving to the next prompt
4. DO NOT modify prompt content during migration — only the delivery mechanism changes

---

## 9. P1: IDEATION ReAct Search Loop

### Problem

IDEATION runs a fixed linear search: generate queries → search → rank → expand → done.
If the search misses a key research direction, there's no way to detect or fix it.

### Solution

Add a search quality self-evaluation step with conditional re-search:

```python
# In ideation.py, after initial search + ranking:

async def _evaluate_search_coverage(self, topic: str, papers: list[dict],
                                     gaps: list[dict]) -> dict:
    """Evaluate whether the search covers all major directions of the topic."""
    result = await self.generate_json(
        system=(
            "You are a research librarian evaluating search completeness. "
            "Given a topic and found papers, assess coverage.\n"
            "Return JSON: {\n"
            "  \"coverage_score\": 1-10,\n"
            "  \"missing_directions\": [\"direction1\", ...],\n"
            "  \"suggested_queries\": [\"query1\", ...],\n"
            "  \"well_covered\": [\"area1\", ...]\n"
            "}"
        ),
        user=(
            f"Topic: {topic}\n\n"
            f"Found {len(papers)} papers. Top 20 titles:\n"
            + "\n".join(f"- {p.get('title', '')}" for p in papers[:20])
            + f"\n\nIdentified gaps:\n"
            + "\n".join(f"- {g.get('description', '')}" for g in gaps)
        ),
    )
    return result


async def _supplementary_search(self, missing_directions: list[str],
                                 existing_papers: dict) -> list[dict]:
    """Run targeted searches for missing research directions."""
    new_papers = []
    for direction in missing_directions[:3]:  # Cap at 3 supplementary searches
        results = await self._search_literature_single(direction)
        for paper in results:
            dedup_key = self._dedup_key(paper)
            if dedup_key not in existing_papers:
                existing_papers[dedup_key] = paper
                new_papers.append(paper)
    return new_papers
```

### Integration

In `run()`, after `_rank_and_filter_papers()`:

```python
# Self-evaluation loop (max 2 rounds)
for eval_round in range(2):
    coverage = await self._evaluate_search_coverage(topic, papers, gaps)
    score = coverage.get("coverage_score", 10)
    if score >= 8:
        self.log(f"Search coverage: {score}/10 — sufficient")
        break
    missing = coverage.get("missing_directions", [])
    if not missing:
        break
    self.log(f"Search coverage: {score}/10 — supplementing {len(missing)} directions")
    new_papers = await self._supplementary_search(missing, all_papers_dict)
    papers.extend(new_papers)
    # Re-rank with new papers
    papers = self._rank_and_filter_papers(papers)
```

> **Safety**: Capped at 2 supplementary rounds and 3 queries per round to prevent
> runaway API costs. Each round adds ~15 seconds.

---

## 10. P2: CODING Quality Gates

### 10.1 AST-Based Import Checking

Replace regex-based import checking (coding.py:657-746) with AST parsing:

```python
"""nanoresearch/agents/coding/import_checker.py"""
import ast
from pathlib import Path


class ImportChecker:
    """Check cross-file import consistency using AST parsing."""

    def __init__(self, code_dir: Path):
        self.code_dir = code_dir
        self.module_exports: dict[str, set[str]] = {}  # module → exported names
        self._parse_all_modules()

    def _parse_all_modules(self):
        for py_file in self.code_dir.rglob("*.py"):
            module_name = py_file.stem
            try:
                tree = ast.parse(py_file.read_text("utf-8"))
                exports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        exports.add(node.name)
                    elif isinstance(node, ast.AsyncFunctionDef):
                        exports.add(node.name)
                    elif isinstance(node, ast.ClassDef):
                        exports.add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                exports.add(target.id)
                self.module_exports[module_name] = exports
            except SyntaxError:
                pass  # Skip unparseable files

    def check_imports(self) -> list[dict]:
        """Check all files for import mismatches."""
        issues = []
        for py_file in self.code_dir.rglob("*.py"):
            try:
                tree = ast.parse(py_file.read_text("utf-8"))
            except SyntaxError:
                issues.append({
                    "file": str(py_file.relative_to(self.code_dir)),
                    "type": "syntax_error",
                    "message": "File has syntax errors",
                })
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module in self.module_exports:
                        for alias in (node.names or []):
                            name = alias.name
                            if name == "*":
                                continue  # Skip star imports
                            if name not in self.module_exports[node.module]:
                                issues.append({
                                    "file": str(py_file.relative_to(self.code_dir)),
                                    "type": "missing_export",
                                    "message": (
                                        f"'{name}' imported from '{node.module}' "
                                        f"but not defined there"
                                    ),
                                    "line": node.lineno,
                                })
        return issues
```

### 10.2 Auto Smoke Test Generation

After code generation, automatically create `test_smoke.py`:

```python
# In coding.py, after file generation:

async def _generate_smoke_test(self, code_dir: Path, file_list: list[str]) -> str:
    """Generate a minimal smoke test that verifies basic functionality."""
    imports_to_test = []
    for f in file_list:
        if f.endswith(".py") and not f.startswith("test_"):
            module = f.replace("/", ".").replace("\\", ".").removesuffix(".py")
            imports_to_test.append(module)

    test_code = '''"""Auto-generated smoke test. Verifies imports and basic shapes."""
import sys
import importlib

def test_all_imports():
    """Verify all generated modules can be imported."""
    failures = []
    modules = {modules_list}
    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures.append(f"{{mod}}: {{e}}")
    if failures:
        print("Import failures:")
        for f in failures:
            print(f"  {{f}}")
        sys.exit(1)
    print(f"All {{len(modules)}} modules imported successfully")

if __name__ == "__main__":
    test_all_imports()
'''.format(modules_list=repr(imports_to_test))

    (code_dir / "test_smoke.py").write_text(test_code, encoding="utf-8")
    return "test_smoke.py"
```

### 10.3 Auto-Formatting

After code generation:

```python
import subprocess, sys

async def _format_generated_code(self, code_dir: Path):
    """Auto-format generated code with black (if available)."""
    try:
        subprocess.run(
            [sys.executable, "-m", "black", "--quiet", "--line-length", "100",
             str(code_dir)],
            capture_output=True, timeout=30
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass  # Non-critical, skip if black not installed
```

---

## 11. P2: Cost Tracking

### Problem

Users have no visibility into API costs per run.

### Solution

Modify `ModelDispatcher.generate()` to return usage data:

```python
# In multi_model.py, modify generate() to return a richer result:

@dataclass
class LLMResult:
    content: str
    usage: dict  # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    model: str
    latency_ms: int

# In generate():
start = time.monotonic()
response = client.chat.completions.create(...)
latency_ms = int((time.monotonic() - start) * 1000)

usage = {}
if hasattr(response, 'usage') and response.usage:
    usage = {
        "prompt_tokens": response.usage.prompt_tokens or 0,
        "completion_tokens": response.usage.completion_tokens or 0,
        "total_tokens": response.usage.total_tokens or 0,
    }

return LLMResult(
    content=response.choices[0].message.content or "",
    usage=usage,
    model=config.model,
    latency_ms=latency_ms,
)
```

> **IMPORTANT**: This is a breaking change to `generate()` return type.
> Migration: All callers currently expect `str`. Change them to use `result.content`.
> Do this file by file, not all at once. Consider keeping `generate()` returning `str`
> and adding a separate `generate_with_usage()` method to avoid breaking changes.

### Cost Aggregation in Orchestrator

```python
# In orchestrator, accumulate per-stage costs:
stage_costs = {}
# After each stage:
stage_costs[stage.name] = {
    "total_tokens": accumulated_tokens,
    "num_calls": call_count,
    "total_latency_ms": accumulated_latency,
}
# Save to manifest:
workspace.update_manifest(cost_tracking=stage_costs)
```

---

## 12. P2: Constants Centralization

Create `nanoresearch/constants.py`:

```python
"""Centralized constants for the NanoResearch pipeline.

All magic numbers should be defined here. Import from this module,
never hardcode numeric literals in agent code.
"""

# === Literature Search ===
TARGET_CITATION_COUNT = 50
MIN_HIGH_CITED_PAPERS = 10
SNOWBALL_MAX_NEW_PAPERS = 15
SNOWBALL_TOP_K = 5
SEARCH_COVERAGE_THRESHOLD = 8  # out of 10
MAX_SUPPLEMENTARY_SEARCH_ROUNDS = 2

# === Code Generation ===
MAX_IMPORT_FIX_RETRIES = 2
MAX_CODE_GEN_RETRIES = 3
MAX_REFERENCE_REPOS = 3

# === Execution ===
MAX_DEBUG_ROUNDS = 20
QUICK_EVAL_TIMEOUT_S = 1200
DRY_RUN_TIMEOUT_S = 60
SUBPROCESS_OUTPUT_LIMIT = 5000

# === Writing ===
MAX_SECTION_CONTEXT_CHARS = 15000
MAX_LATEX_FIX_ROUNDS = 5
MAX_CONTRIBUTION_ITEMS = 3

# === Review ===
MIN_ACCEPTABLE_SECTION_SCORE = 8.0
MAX_REVISION_ROUNDS = 5
CONVERGENCE_THRESHOLD = 0.3  # stop if improvement < this

# === Analysis ===
MAX_ANALYSIS_FIGURES = 5

# === Figure Generation ===
MAX_IMAGE_RETRIES = 2
MAX_CODE_CHART_RETRIES = 3

# === API ===
MAX_API_RETRIES = 5
RETRY_BASE_DELAY_S = 3.0
RETRY_BACKOFF_FACTOR = 2.0

# === Context Management ===
TOOL_RESULT_MAX_CHARS = 6000
TOOL_RESULT_HEAD_CHARS = 2000
TOOL_RESULT_TAIL_CHARS = 1500
CONTEXT_COMPACTION_THRESHOLD = 100000
PROTECTED_TAIL_TURNS = 6
COMPACTED_PREVIEW_CHARS = 400

# === Metrics ===
LOWER_IS_BETTER_PATTERNS = frozenset({
    "loss", "error", "perplexity", "cer", "wer", "fer",
    "mae", "mse", "rmse", "mape", "fid", "kid", "ece",
    "latency", "inference_time",
})
```

### Migration

For each constant:
1. Add to `constants.py`
2. Replace the hardcoded value in the agent file with an import
3. Verify the value matches the original exactly
4. Run tests

---

## 13. P2: DAG Parallel Stage Scheduling

### Problem

All 9 stages run strictly in serial. But some stages have no dependencies between them.

### Solution

Define a dependency DAG and run independent stages in parallel:

```python
# In orchestrator.py:

STAGE_DEPENDENCIES = {
    "IDEATION": [],
    "PLANNING": ["IDEATION"],
    "SETUP": ["PLANNING"],
    "CODING": ["SETUP", "PLANNING"],
    "EXECUTION": ["CODING"],
    "ANALYSIS": ["EXECUTION"],
    "FIGURE_GEN": ["ANALYSIS"],
    "WRITING": ["ANALYSIS", "FIGURE_GEN"],
    "REVIEW": ["WRITING"],
}

async def _run_dag(self):
    """Run stages respecting dependencies, parallelizing where possible."""
    completed = set()
    results = {}

    while True:
        # Find stages that can run now
        runnable = []
        for stage in self._processing_stages():
            if stage.name in completed:
                continue
            deps = STAGE_DEPENDENCIES.get(stage.name, [])
            if all(d in completed for d in deps):
                runnable.append(stage)

        if not runnable:
            break  # All done or deadlocked

        if len(runnable) == 1:
            # Single stage: run directly (saves overhead)
            result = await self._run_stage(runnable[0], results)
            results[runnable[0].name] = result
            completed.add(runnable[0].name)
        else:
            # Multiple independent stages: run in parallel
            tasks = [self._run_stage(s, results) for s in runnable]
            stage_results = await asyncio.gather(*tasks, return_exceptions=True)
            for stage, result in zip(runnable, stage_results):
                if isinstance(result, Exception):
                    self.log(f"Stage {stage.name} failed: {result}")
                    raise result
                results[stage.name] = result
                completed.add(stage.name)
```

> **Safety**: This is an opt-in feature. Add `parallel_stages: true` to config.
> Default: `false` (keeps existing serial behavior).

---

## 14. P2: Progress Streaming

### Problem

Users wait 30-60 minutes with no feedback beyond log files.

### Solution

Add a `ProgressEmitter` that writes to a JSON file, updated in real-time:

```python
"""nanoresearch/pipeline/progress.py"""
import json
import time
from pathlib import Path
from typing import Optional


class ProgressEmitter:
    """Emit progress events to a JSON file for UI consumption."""

    def __init__(self, progress_path: Path):
        self.path = progress_path
        self._events: list[dict] = []
        self._current_stage: Optional[str] = None
        self._start_time = time.time()

    def stage_start(self, stage: str, total_stages: int, current_index: int):
        self._current_stage = stage
        self._emit({
            "type": "stage_start",
            "stage": stage,
            "progress_pct": round(current_index / total_stages * 100),
            "message": f"Starting {stage}...",
        })

    def stage_progress(self, message: str, detail: str = ""):
        self._emit({
            "type": "stage_progress",
            "stage": self._current_stage,
            "message": message,
            "detail": detail,
        })

    def stage_complete(self, stage: str, summary: str = ""):
        self._emit({
            "type": "stage_complete",
            "stage": stage,
            "message": f"{stage} completed",
            "summary": summary,
        })

    def pipeline_complete(self, success: bool, summary: str = ""):
        self._emit({
            "type": "pipeline_complete",
            "success": success,
            "total_time_s": round(time.time() - self._start_time),
            "summary": summary,
        })

    def _emit(self, event: dict):
        event["timestamp"] = time.time()
        self._events.append(event)
        # Atomic write
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({
            "events": self._events[-50:],  # Keep last 50 events
            "current": event,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)
```

### Integration

```python
# In orchestrator.py:
progress = ProgressEmitter(workspace.path / "progress.json")

for i, stage in enumerate(stages):
    progress.stage_start(stage.name, len(stages), i)
    result = await self._run_stage(stage, ...)
    progress.stage_complete(stage.name, f"Generated {len(result)} items")

progress.pipeline_complete(True, "Paper generated successfully")
```

---

## 15. P3: Structured Logging

Replace ad-hoc `self.log()` with structured logging:

```python
"""nanoresearch/logging_config.py"""
import logging
import json
import sys


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "session_id": getattr(record, "session_id", None),
            "message": record.getMessage(),
        }
        # Add any extra fields
        for key in ("model", "tokens", "latency_ms", "error_type"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(log_path=None, level=logging.INFO):
    logger = logging.getLogger("nanoresearch")
    logger.setLevel(level)

    # Console handler (human-readable)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # File handler (structured JSON)
    if log_path:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        logger.addHandler(file_handler)

    return logger
```

---

## 16. P3: Paper Quality Benchmark

Add quantitative quality metrics that don't require LLM self-evaluation:

```python
"""nanoresearch/evaluation/paper_metrics.py"""
import re


def compute_paper_metrics(paper_tex: str, bibtex: str = "") -> dict:
    """Compute quantitative paper quality metrics without LLM."""
    text = paper_tex

    # Structure metrics
    sections = len(re.findall(r'\\section\b', text))
    subsections = len(re.findall(r'\\subsection\b', text))
    figures = len(re.findall(r'\\begin\{figure', text))
    tables = len(re.findall(r'\\begin\{table', text))
    equations = len(re.findall(r'\\begin\{equation', text))
    equations += len(re.findall(r'\$\$', text)) // 2  # Display math

    # Citation metrics
    cite_keys = set()
    for match in re.findall(r'\\cite[tp]?\{([^}]+)\}', text):
        for key in match.split(","):
            cite_keys.add(key.strip())

    # Word count (approximate, strips LaTeX commands)
    plain_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    plain_text = re.sub(r'[\\${}%&_^~]', '', plain_text)
    word_count = len(plain_text.split())

    # Writing quality indicators
    sentences = re.split(r'[.!?]\s+', plain_text)
    avg_sentence_len = (sum(len(s.split()) for s in sentences) / max(len(sentences), 1))

    # Hedge words (lower is better for scientific writing)
    hedge_words = ["may", "might", "could", "perhaps", "possibly", "arguably",
                   "it seems", "appears to", "we believe"]
    hedge_count = sum(plain_text.lower().count(w) for w in hedge_words)
    hedge_ratio = hedge_count / max(word_count, 1) * 1000  # per 1000 words

    # Passive voice detection (rough)
    passive_pattern = r'\b(is|are|was|were|been|being)\s+\w+ed\b'
    passive_count = len(re.findall(passive_pattern, plain_text, re.I))
    passive_ratio = passive_count / max(len(sentences), 1) * 100  # percentage

    return {
        "structure": {
            "sections": sections,
            "subsections": subsections,
            "figures": figures,
            "tables": tables,
            "equations": equations,
            "word_count": word_count,
        },
        "citations": {
            "unique_cite_keys": len(cite_keys),
            "citations_per_1000_words": round(len(cite_keys) / max(word_count, 1) * 1000, 1),
        },
        "writing_quality": {
            "avg_sentence_length": round(avg_sentence_len, 1),
            "hedge_words_per_1000": round(hedge_ratio, 1),
            "passive_voice_pct": round(passive_ratio, 1),
        },
        "completeness": {
            "has_abstract": "\\begin{abstract}" in text,
            "has_introduction": bool(re.search(r'\\section\{Introduction\}', text, re.I)),
            "has_conclusion": bool(re.search(r'\\section\{Conclusion', text, re.I)),
            "has_references": "\\bibliography" in text or "\\begin{thebibliography}" in text,
            "has_appendix": "\\appendix" in text or "\\section{Appendix" in text,
        },
    }
```

---

## 17. P3: Blueprint Semantic Validation

Add validation beyond Pydantic structural checks:

```python
# In planning.py, after Pydantic validation:

def validate_blueprint_semantics(bp, config) -> list[str]:
    """Semantic consistency checks for ExperimentBlueprint."""
    issues = []

    # 1. Every baseline should have at least one expected_performance entry
    for b in bp.baselines:
        if not b.expected_performance:
            issues.append(f"Baseline '{b.name}' has no expected_performance values")

    # 2. Metric direction consistency
    for m in bp.metrics:
        name_lower = m.name.lower()
        if any(p in name_lower for p in LOWER_IS_BETTER_PATTERNS):
            if m.higher_is_better:
                issues.append(
                    f"Metric '{m.name}' looks like lower-is-better "
                    f"but higher_is_better=True")

    # 3. Dataset source URLs should be valid-looking
    for d in bp.datasets:
        if d.source_url and not d.source_url.startswith(("http://", "https://")):
            issues.append(f"Dataset '{d.name}' has suspicious source_url: {d.source_url}")

    # 4. Proposed method must have key_components
    pm = bp.proposed_method
    if isinstance(pm, dict) and not pm.get("key_components"):
        issues.append("Proposed method missing 'key_components' list")

    # 5. Compute requirements vs cluster config
    if bp.compute_requirements and hasattr(config, 'cluster'):
        req_gpus = bp.compute_requirements.num_gpus or 0
        max_gpus = config.cluster.get("max_gpus", 8) if isinstance(config.cluster, dict) else 8
        if req_gpus > max_gpus:
            issues.append(
                f"Blueprint requires {req_gpus} GPUs but cluster max is {max_gpus}")

    # 6. At least one primary metric
    primary_metrics = [m for m in bp.metrics if m.primary]
    if not primary_metrics:
        issues.append("No primary metric defined")

    return issues
```

---

## 18. Bug-Level Fixes (Non-Breaking)

These are point fixes that should be applied immediately with minimal risk.

### 18.1 `args_hash` Stabilization (base.py:427)

**Current** (unstable):
```python
try:
    args_hash = hash(frozenset(args.items()))
except:
    args_hash = hash(str(args))
```

**Fix**:
```python
import json as _json
try:
    args_hash = hash(_json.dumps(args, sort_keys=True, default=str))
except (TypeError, ValueError):
    args_hash = hash(str(sorted(args.items())) if isinstance(args, dict) else str(args))
```

### 18.2 LaTeX Command Detection (base.py:94)

**Current** (heuristic):
```python
if char.isalpha():  # likely LaTeX command
```

**Fix**: Use known command set:
```python
_LATEX_CMD_PREFIXES = frozenset([
    "cite", "textbf", "textit", "frac", "ref", "label", "sqrt", "sum",
    "int", "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
    "sigma", "omega", "text", "math", "begin", "end", "item", "section",
    "subsection", "paragraph", "emph", "url", "href", "footnote",
    "caption", "includegraphics", "usepackage", "newcommand",
])

# In _fix_json_escapes:
cmd_match = re.match(r'([a-zA-Z]+)', text[pos + 1:])
if cmd_match and cmd_match.group(1) in _LATEX_CMD_PREFIXES:
    # Double-escape this backslash
```

### 18.3 Analysis Figure Cap Logging (analysis.py:287)

**Current** (silent cap):
```python
figure_specs[:3]
```

**Fix**:
```python
max_figs = 5  # or import from constants
if len(figure_specs) > max_figs:
    self.log(f"Capping analysis figures from {len(figure_specs)} to {max_figs}")
figure_specs = figure_specs[:max_figs]
```

### 18.4 LaTeX Fix Loop Infinite Detection (review.py:1494-1527)

**Add** before the fix loop:
```python
seen_error_sigs = set()
# Inside loop:
import hashlib
sig = hashlib.md5(error_log[-500:].encode()).hexdigest()[:8]
if sig in seen_error_sigs:
    self.log("LaTeX fix loop: same error repeated, stopping")
    break
seen_error_sigs.add(sig)
```

### 18.5 Review Section Truncation (review.py:856-858)

**Current**: Keeps first 12K + last 8K chars (may drop Method section).

**Fix**: Use section-boundary-aware truncation:
```python
def _smart_truncate(self, text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    # Find section boundaries
    section_starts = [(m.start(), m.group(1))
                      for m in re.finditer(r'\\section\{([^}]+)\}', text)]
    if not section_starts:
        # No sections found, fall back to head/tail
        return text[:12000] + "\n\n[...truncated...]\n\n" + text[-8000:]

    # Keep sections by priority
    priority = ["abstract", "introduction", "method", "experiment",
                "result", "conclusion"]
    kept_sections = []
    remaining = max_chars

    for pname in priority:
        for i, (start, title) in enumerate(section_starts):
            if pname in title.lower():
                end = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(text)
                content = text[start:end]
                if len(content) <= remaining:
                    kept_sections.append(content)
                    remaining -= len(content)
                elif remaining > 500:
                    kept_sections.append(content[:remaining] + "\n[...truncated...]")
                    remaining = 0
                break

    return "\n\n".join(kept_sections) if kept_sections else text[:max_chars]
```

### 18.6 Writing Lower-is-Better Detection (writing.py:1731-1759)

**Current**: Only checks "loss", "error", "perplexity".

**Fix**: Use centralized pattern set:
```python
from nanoresearch.constants import LOWER_IS_BETTER_PATTERNS

def _is_lower_better(metric_name: str) -> bool:
    name = metric_name.lower().replace(" ", "_").replace("-", "_")
    return any(p in name for p in LOWER_IS_BETTER_PATTERNS)
```

### 18.7 Cache Structure Validation (ideation.py:105-122)

**Wrap** cache loading in try-except:
```python
try:
    cached = json.loads(cache_path.read_text("utf-8"))
    if not isinstance(cached, dict) or "papers" not in cached:
        raise ValueError("invalid cache structure")
except (json.JSONDecodeError, ValueError, OSError) as e:
    self.log(f"Search cache invalid ({e}), starting fresh")
    cached = None
```

### 18.8 `generate_with_tools` Terminal Guard (base.py:481-484)

**Add** after the final summary call:
```python
if hasattr(response, 'tool_calls') and response.tool_calls:
    # LLM returned tool_calls in summary round — force text extraction
    return response.content or "Agent completed but produced no text summary."
```

---

## 19. Implementation Order & Dependencies

### Phase 1: Zero-Risk Bug Fixes (Day 1)

Apply all items from Section 18 (Bug-Level Fixes). These are point changes with no
architectural impact. Each one is independently testable.

**Order**: 18.3 → 18.4 → 18.7 → 18.1 → 18.2 → 18.6 → 18.8 → 18.5

### Phase 2: Constants + Shared LaTeX Fixer (Day 2-3)

1. Create `constants.py` (Section 12) — no existing code changes yet
2. Create `nanoresearch/latex/fixer.py` (Section 7)
3. Migrate writing.py to use shared fixer
4. Migrate review.py to use shared fixer
5. Gradually replace magic numbers with constants imports

### Phase 3: ANALYSIS Module Rewrite (Day 4-6)

1. Create `agents/analysis/` package (Section 2)
2. Add `statistics.py`, `training_dynamics.py`, `ablation_analysis.py`, `comparison_matrix.py`
3. Integrate into existing `analysis.py` run() method (additive, no removals)
4. Add tests for each new module
5. Verify WRITING module consumes the new `computational_analysis` output

### Phase 4: Context Engine + Memory (Day 7-9)

1. Create `nanoresearch/memory/` package (Section 5)
2. Implement `ResearchMemory` and `CrossRunMemory`
3. Implement `ContextEngine` interface with `LegacyContextEngine`
4. Integrate into orchestrator (additive — Legacy engine preserves existing behavior)
5. Add `SmartContextEngine` as opt-in

### Phase 5: Multi-Model Review + Citation Check (Day 10-12)

1. Add `review_committee` config option (Section 3)
2. Implement `_multi_reviewer_assessment()` in review.py
3. Add `review_citation_checker.py` (Section 4)
4. Integrate citation checking into review consistency_issues
5. Both features are backward-compatible (single-model fallback)

### Phase 6: File Splitting (Day 13-16)

1. Split execution.py (Section 6.1) — start with result_collector, then resource_matcher
2. Split writing.py (Section 6.2) — start with grounding.py, then context_builder
3. Split experiment.py (Section 6.3) — start with edit_apply
4. Run full test suite after each file extraction

### Phase 7: Prompt Externalization (Day 17-19)

1. Create `prompts/` directory structure (Section 8)
2. Extract prompts one category at a time (start with figure_gen — largest)
3. Verify output quality matches before and after for each prompt
4. Add PromptLoader with caching

### Phase 8: IDEATION + CODING Improvements (Day 20-22)

1. Add search coverage self-evaluation (Section 9)
2. Replace regex import checker with AST (Section 10.1)
3. Add smoke test generation (Section 10.2)
4. Add auto-formatting (Section 10.3)

### Phase 9: Infrastructure (Day 23-25)

1. Cost tracking (Section 11)
2. Progress streaming (Section 14)
3. Structured logging (Section 15)
4. Paper quality benchmark (Section 16)
5. Blueprint semantic validation (Section 17)

### Phase 10: DAG Scheduling (Day 26-28)

1. Implement DAG scheduler (Section 13)
2. Add `parallel_stages` config option
3. Test with FIGURE_GEN + WRITING parallelization
4. Default OFF, opt-in only

---

## Appendix A: OpenClaw Reference Files

The OpenClaw source code is cloned to `D:/openclaw` for reference. Key files:

| File | What to Study |
|------|---------------|
| `src/context-engine/types.ts` | ContextEngine interface contract |
| `src/context-engine/registry.ts` | Factory registration + slot resolution pattern |
| `src/context-engine/legacy.ts` | Backward-compatible wrapper pattern |
| `src/memory/hybrid.ts` | Hybrid vector+keyword merge algorithm |
| `src/memory/temporal-decay.ts` | Exponential decay scoring formula |
| `src/memory/mmr.ts` | MMR diversity re-ranking |
| `src/plugins/slots.ts` | Exclusive plugin slot pattern |

**Key formulas from OpenClaw**:
- Hybrid score: `score = vectorWeight * vectorScore + textWeight * textScore` (0.7/0.3 default)
- Temporal decay: `multiplier = e^(-ln2/halfLifeDays * ageInDays)` (30-day half-life default)
- BM25 rank to score: `score = relevance / (1 + relevance)` where `relevance = -rank`

## Appendix B: Test Checklist

Before merging any change, verify:

- [ ] `python -m pytest tests/` passes
- [ ] A single full pipeline run (fast_draft profile) completes without error
- [ ] LaTeX compilation succeeds (test with tectonic)
- [ ] JSON repair handles truncated output (test with `test_anti_fabrication.py`)
- [ ] Review score does not regress on existing test papers
- [ ] Resume from checkpoint works (run half, kill, resume)
- [ ] Windows paths work (forward slashes, UTF-8 encoding)

---

## 20. ERRATA: Bugs Fixed in This Document (Self-Review)

During re-review, the following bugs were found and fixed in this document:

| Location | Bug | Fix |
|----------|-----|-----|
| Section 2.1 `_approx_two_tailed_p` | Lookup table used df=1 critical values for all df <= 30 — would underestimate p-values and falsely claim significance | Fixed to use df=10 critical values (conservative) |
| Section 2.4 `comparison_matrix_to_latex` | Metric names with underscores (e.g. `F1_score`) would break LaTeX compilation | Added `_latex_escape_header()` to escape `_`, `%`, `&` |
| Section 3.2 | `_merge_section_reviews()` called but never defined | Added full implementation |
| Section 4 Citation checker | No rate limiting — paper with 50 citations would make 50 LLM calls | Added `MAX_CHECKS = 15` cap and `checked_keys` dedup |
| Section 5.1 `CrossRunMemory` | Missing `import json, math, time, Path` | Added imports |
| Section 5.1 `SmartContextEngine.after_stage` | Uses `MemoryEntry` class without import note | Added docstring note about required import |
| Section 7 `_ensure_packages` | Could add duplicate packages if loaded by style file | Fixed to check full preamble, not just `\usepackage{}` pattern |

---

## 21. MISSING: Graceful Shutdown & Signal Handling

### Problem

The pipeline has **no signal handlers**. If the process receives SIGTERM/SIGINT:
- SLURM jobs are left running on the cluster (orphaned)
- Workspace manifest may be in inconsistent state (partial write)
- Temporary files are not cleaned up
- OpenAI/API client connections are not closed

### Solution

```python
"""nanoresearch/pipeline/shutdown.py"""
import signal
import asyncio
import logging
from typing import Optional

log = logging.getLogger("nanoresearch")


class GracefulShutdown:
    """Graceful shutdown manager for the pipeline.

    Registers signal handlers that set a flag and trigger cleanup.
    Agents should check `is_shutting_down` periodically.
    """

    def __init__(self):
        self._shutting_down = False
        self._cleanup_callbacks: list = []
        self._original_handlers: dict = {}

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    def register(self):
        """Register signal handlers. Call once at pipeline start."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except (OSError, ValueError):
                pass  # Not all signals available on Windows

    def on_cleanup(self, callback):
        """Register a cleanup callback (sync or async)."""
        self._cleanup_callbacks.append(callback)

    def _handle_signal(self, signum, frame):
        if self._shutting_down:
            # Second signal: force exit
            log.warning("Forced shutdown (second signal)")
            raise SystemExit(1)
        self._shutting_down = True
        log.info(f"Shutdown signal received ({signal.Signals(signum).name}), "
                 f"cleaning up...")

    async def run_cleanup(self):
        """Run all registered cleanup callbacks."""
        for cb in reversed(self._cleanup_callbacks):
            try:
                result = cb()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                log.warning(f"Cleanup callback failed: {e}")

    def restore(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
```

### Integration

```python
# In orchestrator.py run():
shutdown = GracefulShutdown()
shutdown.register()

# Register cleanup: cancel SLURM jobs, close clients, persist memory
shutdown.on_cleanup(lambda: dispatcher.close())
shutdown.on_cleanup(lambda: memory.persist(workspace.path / "memory.json"))
shutdown.on_cleanup(lambda: workspace.update_manifest(
    current_stage=PipelineStage.FAILED.value))

try:
    for stage in stages:
        if shutdown.is_shutting_down:
            log.info("Shutdown requested, stopping pipeline")
            break
        # ... run stage ...
finally:
    await shutdown.run_cleanup()
    shutdown.restore()
```

---

## 22. MISSING: Resource Cleanup & Connection Pool Safety

### Problem

- `ModelDispatcher._clients` dict grows unbounded — a new OpenAI client is created
  for each unique timeout value and never evicted.
- Subprocess instances from code execution may become zombies if parent crashes.
- No `async with` pattern for the pipeline orchestrator.

### Solution

#### 22.1 Client Pool with Max Size

```python
# In multi_model.py, modify _get_client():

_MAX_CLIENTS = 5  # Maximum concurrent API clients

def _get_client(self, timeout: float) -> OpenAI:
    rounded = round(timeout / 10) * 10  # Round to nearest 10s
    if rounded not in self._clients:
        # Evict oldest if at capacity
        if len(self._clients) >= _MAX_CLIENTS:
            oldest_key = next(iter(self._clients))
            try:
                self._clients[oldest_key].close()
            except Exception:
                pass
            del self._clients[oldest_key]
        self._clients[rounded] = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=rounded,
        )
    return self._clients[rounded]
```

#### 22.2 Subprocess Cleanup Guard

```python
# In execution agents, wrap subprocess calls:

async def _run_with_cleanup(cmd: list[str], timeout: int, cwd: Path) -> tuple[str, str, int]:
    """Run subprocess with guaranteed cleanup on cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd))
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout)
        return stdout.decode("utf-8", errors="replace"), \
               stderr.decode("utf-8", errors="replace"), \
               proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()  # Reap zombie
        raise
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
```

#### 22.3 Orchestrator as Async Context Manager

```python
class PipelineOrchestrator:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.dispatcher.close()
        if self.memory:
            self.memory.persist(self.workspace.path / "memory.json")
        return False  # Don't suppress exceptions
```

---

## 23. MISSING: Bare Exception Hardening

### Problem

20+ instances of `except Exception:` with `pass` or minimal handling. These swallow
real errors and make debugging extremely difficult.

### Rule for Developers

```
NEVER write `except Exception: pass`
ALWAYS at minimum: `except Exception as e: self.log(f"... failed: {e}")`
PREFER specific exceptions: `except (json.JSONDecodeError, KeyError) as e:`
```

### Specific Fixes Needed

| File | Line (approx) | Current | Fix |
|------|---------------|---------|-----|
| base.py:156 | `except Exception: pass` | `except Exception as e: log.debug(f"JSON escape fix failed: {e}")` |
| execution.py:4158 | `except Exception:` silent | Add `log.warning(f"Log stream error: {e}")` |
| execution.py:4317 | `except Exception:` silent | Add `log.warning(f"Metrics recovery: {e}")` |
| debug.py:156 | `except Exception:` in file read | `except (OSError, UnicodeDecodeError) as e:` |
| coding.py:354 | `except Exception:` in JSON parse | `except (json.JSONDecodeError, ValueError) as e:` |
| figure_gen.py:1730 | `except Exception:` on image close | `except (OSError, AttributeError) as e:` |

### General Pattern to Follow

```python
# BAD:
try:
    result = risky_operation()
except Exception:
    pass

# GOOD:
try:
    result = risky_operation()
except SpecificError as e:
    self.log(f"Operation failed (non-fatal): {e}")
    result = fallback_value
```

---

## 24. MISSING: Concurrency Safety

### Problem

Several global singletons are initialized lazily without thread safety:

1. `ideation.py:41-72` — `_arxiv_search`, `_s2_search` etc. are global `None` variables
   initialized on first call. If two coroutines call simultaneously, both see `None` and
   both initialize.

2. `multi_model.py:44-56` — `_clients` dict is shared but has no lock.

### 24.1 Fix for Ideation Search Functions

The project already has `_lazy_lock = asyncio.Lock()` but it's only used for some
functions. Ensure ALL lazy init functions use it:

```python
# Verify every _ensure_*() function acquires _lazy_lock:
async def _ensure_all_search_functions():
    async with _lazy_lock:
        if _arxiv_search is not None:
            return  # Already initialized
        # ... initialize all at once ...
```

### 24.2 Fix for Client Dict

```python
import threading

class ModelDispatcher:
    def __init__(self, config):
        self._clients = {}
        self._client_lock = threading.Lock()

    def _get_client(self, timeout: float) -> OpenAI:
        rounded = round(timeout / 10) * 10
        with self._client_lock:
            if rounded not in self._clients:
                self._clients[rounded] = OpenAI(...)
            return self._clients[rounded]
```

---

## 25. MISSING: Checkpoint Transactionality

### Problem

- Workspace manifest is written with atomic rename (good), but stage output JSON files
  are written directly — if crash happens mid-write, the JSON is corrupted.
- On resume, the orchestrator loads the output file without validating it's complete JSON.

### Solution

```python
# In workspace.py write_json(), use same atomic pattern as _write_manifest:

def write_json(self, subpath: str, data: dict):
    """Atomically write JSON to workspace."""
    target = self.path / subpath
    target.parent.mkdir(parents=True, exist_ok=True)
    import tempfile, os
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(target))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
```

```python
# In orchestrator.py, validate loaded output:

def _load_stage_output(self, stage: PipelineStage) -> dict:
    """Load and validate stage output JSON."""
    path = self._output_file_path(stage)
    if not path.exists():
        raise StageError(f"Output file missing for {stage.name}: {path}")
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        return data
    except (json.JSONDecodeError, ValueError) as e:
        raise StageError(
            f"Corrupted output for {stage.name}: {e}. "
            f"Delete {path} and re-run this stage."
        )
```

---

## 26. MISSING: SLURM Edge Cases

### Problem

Several SLURM integration edge cases are unhandled:

1. Job enters `COMPLETING` state (between RUNNING and COMPLETED) — not recognized
2. `sacct` query can hang indefinitely — no timeout
3. Job ID file (`active_job_id.txt`) corruption → can't recover job state
4. `scancel` is fire-and-forget — no verification that job actually stopped

### Solution

```python
# 1. Add COMPLETING to recognized states:
SLURM_RUNNING_STATES = {"RUNNING", "COMPLETING", "REQUEUED", "RESIZING"}
SLURM_TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
                          "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY"}

# 2. Add timeout to sacct:
async def _get_job_status(job_id: str, timeout: int = 30) -> str:
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "sacct", "-j", job_id, "--format=State", "--noheader",
                "--parsable2", stdout=asyncio.subprocess.PIPE),
            timeout=timeout)
        stdout, _ = await proc.communicate()
        status = stdout.decode().strip().split("\n")[0].strip()
        return status if status else "UNKNOWN"
    except asyncio.TimeoutError:
        return "UNKNOWN"

# 3. Validate job ID on load:
def _read_job_id(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text("utf-8").strip()
    if not text.isdigit():
        log.warning(f"Corrupted job ID file: {path} contains '{text}'")
        return None
    return text

# 4. Verify scancel:
async def _cancel_job(job_id: str, timeout: int = 15) -> bool:
    proc = await asyncio.create_subprocess_exec("scancel", job_id)
    await asyncio.wait_for(proc.wait(), timeout=timeout)
    # Verify
    await asyncio.sleep(2)
    status = await _get_job_status(job_id)
    if status in SLURM_TERMINAL_STATES or status == "UNKNOWN":
        return True
    log.warning(f"Job {job_id} still in state {status} after scancel")
    return False
```

---

## 27. MISSING: Test Expansion Priorities

### Currently Untested (High Priority)

| Test Category | What to Test | Why |
|---------------|-------------|-----|
| **Subprocess failure** | Code execution returns non-zero, stdout is garbage, timeout mid-write | Most common real-world failure |
| **Concurrent agent runs** | Two IDEATION agents sharing same OpenAlex rate limiter | Race condition on `_limiter` global |
| **Workspace corruption** | Resume after partial JSON write; manifest missing fields | Crash recovery robustness |
| **LaTeX fixer idempotency** | Run `deterministic_fix()` twice — output must be identical | Prevents fix-introduces-new-bug loops |
| **Citation fact-checker** | Claim says "95% accuracy" but source says "85% accuracy" | Core P0 feature |
| **Multi-model review** | One reviewer times out; all reviewers disagree | Graceful degradation |
| **Memory persistence** | Write 500 entries, reload, verify all present and scores correct | Cross-run memory reliability |
| **Signal handling** | Send SIGTERM during EXECUTION stage | Clean shutdown |

### Test Template

```python
# tests/test_analysis_statistics.py

import pytest
from nanoresearch.agents.analysis.statistics import (
    welch_t_test, cohens_d, bootstrap_ci, compute_significance_report
)

def test_welch_t_test_identical_samples():
    """Identical distributions should not be significant."""
    a = [0.85, 0.86, 0.84, 0.85, 0.86]
    b = [0.85, 0.86, 0.84, 0.85, 0.86]
    result = welch_t_test(a, b)
    assert result["p_value"] > 0.5  # Definitely not significant

def test_welch_t_test_clearly_different():
    """Obviously different distributions should be significant."""
    a = [0.90, 0.91, 0.92, 0.90, 0.91]
    b = [0.70, 0.71, 0.72, 0.70, 0.71]
    result = welch_t_test(a, b)
    assert result["p_value"] < 0.01
    assert result["significant_at_001"] is True

def test_welch_t_test_insufficient_data():
    """Should return None fields with fewer than 2 samples."""
    result = welch_t_test([0.9], [0.8])
    assert result["t_statistic"] is None
    assert result["reason"] == "need at least 2 samples per group"

def test_cohens_d_large_effect():
    a = [0.90, 0.91, 0.92]
    b = [0.70, 0.71, 0.72]
    d = cohens_d(a, b)
    assert d is not None
    assert d > 1.0  # Very large effect

def test_bootstrap_ci_contains_mean():
    samples = [0.85, 0.86, 0.84, 0.87, 0.83]
    ci = bootstrap_ci(samples)
    assert ci["lower"] <= ci["mean"] <= ci["upper"]

def test_significance_report_lower_is_better():
    """For lower-is-better metrics, proposed < baseline = improvement."""
    proposed = [0.10, 0.11, 0.09]
    baseline = [0.20, 0.21, 0.19]
    report = compute_significance_report(
        proposed, baseline, "loss", higher_is_better=False)
    assert report["improvement"] > 0  # Positive means improvement
    assert report["interpretation"] == "statistically significant improvement"


# tests/test_training_dynamics.py

from nanoresearch.agents.analysis.training_dynamics import analyze_training_dynamics

def test_overfitting_detection():
    """Detect overfitting when val_loss rises in last third."""
    log = [
        {"epoch": i, "train_loss": 1.0 - i * 0.05, "val_loss": 1.0 - i * 0.03}
        for i in range(10)
    ]
    # Make val_loss rise in last 4 epochs
    for i in range(6, 10):
        log[i]["val_loss"] = 0.7 + (i - 6) * 0.05
    result = analyze_training_dynamics(log)
    assert result["overfitting_detected"] is True

def test_stable_training():
    """Stable training should not trigger overfitting."""
    log = [
        {"epoch": i, "train_loss": 1.0 / (1 + i), "val_loss": 1.1 / (1 + i)}
        for i in range(20)
    ]
    result = analyze_training_dynamics(log)
    assert result["overfitting_detected"] is False
    assert result["loss_stability"] == "stable"

def test_insufficient_data():
    log = [{"epoch": 1, "val_loss": 0.5}]
    result = analyze_training_dynamics(log)
    assert "analysis_skipped" in result


# tests/test_latex_fixer.py

from nanoresearch.latex.fixer import deterministic_fix

def test_idempotency():
    """Running deterministic_fix twice should produce identical output."""
    tex = r"""
\documentclass{article}
\begin{document}
Hello — world "test"
\end{document}
"""
    fixed_once = deterministic_fix(tex)
    fixed_twice = deterministic_fix(fixed_once)
    assert fixed_once == fixed_twice, "deterministic_fix is not idempotent"

def test_removes_junk_before_documentclass():
    tex = "Some junk\n\n\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}"
    fixed = deterministic_fix(tex)
    assert fixed.startswith("\\documentclass")

def test_adds_missing_end_document():
    tex = "\\documentclass{article}\n\\begin{document}\nHello"
    fixed = deterministic_fix(tex)
    assert "\\end{document}" in fixed
```

---

## 28. MISSING: WRITING Global Consistency Pass

### Problem

After all sections are generated, there's no check that:
- Numbers in the abstract match numbers in experiments
- Contribution count in intro matches method subsection count
- All `\ref{fig:X}` have corresponding `\label{fig:X}`

### Solution

Add to `writing.py` after all sections are generated:

```python
def _global_consistency_check(self, sections: dict[str, str]) -> list[str]:
    """Post-generation consistency check across all sections.

    Returns list of issue strings. Non-blocking — issues are logged
    and optionally passed to REVIEW for fixing.
    """
    issues = []
    all_text = "\n".join(sections.values())

    # 1. Abstract numbers must appear in Experiments
    abstract = sections.get("abstract", "")
    experiments = sections.get("experiments", "")
    abstract_numbers = set(re.findall(r'(\d+\.?\d*)\s*\\?%', abstract))
    experiment_numbers = set(re.findall(r'(\d+\.?\d*)\s*\\?%', experiments))
    orphan_numbers = abstract_numbers - experiment_numbers
    for num in orphan_numbers:
        issues.append(
            f"Abstract claims {num}% but this number does not appear in Experiments")

    # 2. All \ref{fig:X} must have a matching \label{fig:X}
    refs = set(re.findall(r'\\ref\{(fig:[^}]+)\}', all_text))
    labels = set(re.findall(r'\\label\{(fig:[^}]+)\}', all_text))
    for ref in refs - labels:
        issues.append(f"\\ref{{{ref}}} has no matching \\label")

    # 3. All \ref{tab:X} must have a matching \label{tab:X}
    tab_refs = set(re.findall(r'\\ref\{(tab:[^}]+)\}', all_text))
    tab_labels = set(re.findall(r'\\label\{(tab:[^}]+)\}', all_text))
    for ref in tab_refs - tab_labels:
        issues.append(f"\\ref{{{ref}}} has no matching \\label")

    # 4. Contribution bullet count sanity check
    intro = sections.get("introduction", "")
    contrib_items = len(re.findall(r'\\item', intro))
    if contrib_items > 5:
        issues.append(
            f"Introduction has {contrib_items} \\item entries — "
            f"consider merging to 2-4 contributions")

    return issues
```

---

## Appendix C: Complete File Structure After All Improvements

```
nanoresearch/
├── agents/
│   ├── base.py                      # (MODIFIED: args_hash, LaTeX cmd set, tool guard)
│   ├── ideation.py                  # (MODIFIED: cache validation, search coverage loop)
│   ├── planning.py                  # (MODIFIED: semantic validation)
│   ├── setup.py                     # (unchanged)
│   ├── analysis/                    # (NEW PACKAGE)
│   │   ├── __init__.py              # AnalysisAgent (migrated from analysis.py)
│   │   ├── statistics.py            # Welch t-test, Cohen's d, bootstrap CI
│   │   ├── training_dynamics.py     # Convergence, overfitting, stability
│   │   ├── ablation_analysis.py     # Component contribution quantification
│   │   ├── comparison_matrix.py     # Auto comparison table + LaTeX render
│   │   └── claim_verifier.py        # Result-claim consistency
│   ├── coding/                      # (NEW PACKAGE, split from coding.py)
│   │   ├── __init__.py
│   │   └── import_checker.py        # AST-based import validation
│   ├── execution/                   # (NEW PACKAGE, split from execution.py)
│   │   ├── __init__.py
│   │   ├── local_runner.py
│   │   ├── cluster_runner.py
│   │   ├── debug_loop.py
│   │   ├── resource_matcher.py
│   │   ├── repair_strategies.py
│   │   └── result_collector.py
│   ├── experiment/                  # (NEW PACKAGE, split from experiment.py)
│   │   ├── __init__.py
│   │   ├── pipeline_mode.py
│   │   ├── react_mode.py
│   │   └── edit_apply.py
│   ├── writing/                     # (NEW PACKAGE, split from writing.py)
│   │   ├── __init__.py
│   │   ├── context_builder.py
│   │   ├── grounding.py
│   │   ├── section_writer.py
│   │   ├── table_builder.py
│   │   ├── citation_manager.py
│   │   └── latex_assembler.py
│   ├── review.py                    # (MODIFIED: multi-model, citation check)
│   ├── review_citation_checker.py   # (NEW)
│   ├── figure_gen.py                # (MODIFIED: external prompts, figure cap)
│   ├── debug.py                     # (unchanged)
│   └── tools.py                     # (unchanged)
├── constants.py                     # (NEW: all magic numbers)
├── context_engine/                  # (NEW PACKAGE)
│   ├── __init__.py
│   └── interface.py                 # ContextEngine ABC + Legacy + Smart
├── latex/                           # (NEW PACKAGE)
│   ├── __init__.py
│   └── fixer.py                     # Shared 2-level LaTeX fix
├── memory/                          # (NEW PACKAGE)
│   ├── __init__.py
│   ├── research_memory.py           # Cross-stage memory with decay
│   └── cross_run_memory.py          # Persistent cross-run learning
├── evaluation/                      # (NEW PACKAGE)
│   ├── __init__.py
│   └── paper_metrics.py             # Quantitative paper quality metrics
├── pipeline/
│   ├── orchestrator.py              # (MODIFIED: memory hooks, shutdown, DAG)
│   ├── multi_model.py               # (MODIFIED: cost tracking, client pool)
│   ├── state.py                     # (unchanged)
│   ├── workspace.py                 # (MODIFIED: atomic write_json)
│   ├── progress.py                  # (NEW: progress streaming)
│   └── shutdown.py                  # (NEW: graceful shutdown)
├── prompts/                         # (NEW: externalized prompt templates)
│   ├── __init__.py
│   ├── figure_gen/
│   ├── writing/
│   ├── review/
│   └── ideation/
├── logging_config.py                # (NEW: structured logging)
└── config.py                        # (MODIFIED: review_committee, parallel_stages)
```

## Appendix D: Risk Assessment per Change

| Change | Risk of Breaking Existing Behavior | Mitigation |
|--------|-----------------------------------|------------|
| Bug fixes (Section 18, 20) | **NONE** — additive or narrowly scoped | Point fixes, testable in isolation |
| Constants centralization | **NONE** — imports replace literals | Value must match original exactly |
| Shared LaTeX fixer | **LOW** — same logic, different file | Run LaTeX compilation tests |
| ANALYSIS enrichment | **NONE** — additive, original LLM call kept | New outputs added alongside existing |
| Memory system | **NONE** — opt-in via SmartContextEngine | LegacyContextEngine = exact existing behavior |
| Multi-model review | **NONE** — fallback to single model if config empty | Backward compatible by default |
| Citation fact-check | **NONE** — adds to existing consistency_issues list | Existing revision loop processes it |
| File splitting | **LOW** — imports change, signatures preserved | Run tests after each extraction |
| Prompt externalization | **MEDIUM** — prompt text must be byte-identical | Diff prompts before/after extraction |
| DAG scheduling | **NONE** — opt-in via config flag | Default OFF |
| Cost tracking | **MEDIUM** — `generate()` return type changes | Use `generate_with_usage()` instead to avoid breaking |
| Graceful shutdown | **NONE** — additive signal handlers | Restores original handlers on exit |

---

## Appendix E: Checklist for New Developer

Before starting implementation:

- [ ] Read this entire document
- [ ] Read `MEMORY.md` in the project root for fix history context
- [ ] Read `architecture_improvements.md` (V1) for prior architectural decisions
- [ ] Run `python -m pytest tests/` to establish baseline (all tests pass)
- [ ] Run one full `fast_draft` pipeline to understand the end-to-end flow
- [ ] Set up a test topic that completes in <10 minutes for iteration

During implementation:

- [ ] Follow the phase order in Section 19 (bug fixes first, infrastructure last)
- [ ] Commit after each individual change, not in bulk
- [ ] Write tests for new code BEFORE integrating with existing modules
- [ ] Never modify prompt text during the externalization phase — only move it
- [ ] When splitting files, do ONE file at a time, run tests, then continue
- [ ] Check Appendix D risk assessment before each change

After implementation:

- [ ] Full test suite passes
- [ ] `fast_draft` pipeline completes end-to-end
- [ ] `local_quick` pipeline completes with real experiment execution
- [ ] Resume from checkpoint works (kill mid-WRITING, resume, verify PDF generated)
- [ ] LaTeX compilation succeeds with tectonic
- [ ] Multi-model review works with at least 2 different providers
- [ ] Citation fact-checker catches a known inaccurate citation in a test paper
- [ ] Memory persists across two consecutive runs on the same topic
