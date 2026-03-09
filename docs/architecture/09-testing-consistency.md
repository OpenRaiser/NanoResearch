# 27. Test Expansion Priorities

## Currently Untested (High Priority)

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

## Test Template

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

# 28. WRITING Global Consistency Pass

## Problem

After all sections are generated, there's no check that:
- Numbers in the abstract match numbers in experiments
- Contribution count in intro matches method subsection count
- All `\ref{fig:X}` have corresponding `\label{fig:X}`

## Solution

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

## Test Checklist

Before merging any change, verify:

- [ ] `python -m pytest tests/` passes
- [ ] A single full pipeline run (fast_draft profile) completes without error
- [ ] LaTeX compilation succeeds (test with tectonic)
- [ ] JSON repair handles truncated output (test with `test_anti_fabrication.py`)
- [ ] Review score does not regress on existing test papers
- [ ] Resume from checkpoint works (run half, kill, resume)
- [ ] Windows paths work (forward slashes, UTF-8 encoding)

---

## Complete File Structure After All Improvements

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
