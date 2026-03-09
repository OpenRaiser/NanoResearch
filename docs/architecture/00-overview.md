# NanoResearch Architecture Improvements V2 — Overview

> **Author**: Claude Opus 4.6 deep review, March 2026
> **Target**: Next developer continuing NanoResearch development
> **Scope**: All modules, ~30,000 LOC, 9-stage pipeline
> **Goal**: From "production-quality research agent" to "industry-defining research agent"

---

## Document Index

This handbook is split into 10 files for easier navigation:

| File | Sections | Priority |
|------|----------|----------|
| [00-overview.md](00-overview.md) | Assessment, Implementation Plan, Risk, Checklists | — |
| [01-analysis-rewrite.md](01-analysis-rewrite.md) | Section 2: ANALYSIS Module Rewrite | P0 |
| [02-review-citations.md](02-review-citations.md) | Sections 3-4: Multi-Model Review + Citation Fact-Checking | P0 |
| [03-context-engine.md](03-context-engine.md) | Section 5: Context Engine + Memory System | P0 |
| [04-file-splitting-latex.md](04-file-splitting-latex.md) | Sections 6-7: File Splitting + Shared LaTeX Fixer | P1 |
| [05-prompts-search-coding.md](05-prompts-search-coding.md) | Sections 8-10: Prompt Externalization, IDEATION, CODING | P1-P2 |
| [06-infrastructure.md](06-infrastructure.md) | Sections 11-15: Cost, Constants, DAG, Progress, Logging | P2-P3 |
| [07-quality-validation.md](07-quality-validation.md) | Sections 16-18, 20: Quality Metrics, Validation, Bug Fixes, Errata | P3 |
| [08-robustness.md](08-robustness.md) | Sections 21-26: Shutdown, Cleanup, Exceptions, Concurrency, Checkpoints, SLURM | P1-P2 |
| [09-testing-consistency.md](09-testing-consistency.md) | Sections 27-28 + Appendices: Tests, Consistency, File Structure | — |

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

## Implementation Order & Dependencies

### Phase 1: Zero-Risk Bug Fixes (Day 1)

Apply all items from Section 18 in [07-quality-validation.md](07-quality-validation.md). These are point changes with no
architectural impact. Each one is independently testable.

**Order**: 18.3 → 18.4 → 18.7 → 18.1 → 18.2 → 18.6 → 18.8 → 18.5

### Phase 2: Constants + Shared LaTeX Fixer (Day 2-3)

1. Create `constants.py` (Section 12 in [06-infrastructure.md](06-infrastructure.md)) — no existing code changes yet
2. Create `nanoresearch/latex/fixer.py` (Section 7 in [04-file-splitting-latex.md](04-file-splitting-latex.md))
3. Migrate writing.py to use shared fixer
4. Migrate review.py to use shared fixer
5. Gradually replace magic numbers with constants imports

### Phase 3: ANALYSIS Module Rewrite (Day 4-6)

1. Create `agents/analysis/` package ([01-analysis-rewrite.md](01-analysis-rewrite.md))
2. Add `statistics.py`, `training_dynamics.py`, `ablation_analysis.py`, `comparison_matrix.py`
3. Integrate into existing `analysis.py` run() method (additive, no removals)
4. Add tests for each new module
5. Verify WRITING module consumes the new `computational_analysis` output

### Phase 4: Context Engine + Memory (Day 7-9)

1. Create `nanoresearch/memory/` package ([03-context-engine.md](03-context-engine.md))
2. Implement `ResearchMemory` and `CrossRunMemory`
3. Implement `ContextEngine` interface with `LegacyContextEngine`
4. Integrate into orchestrator (additive — Legacy engine preserves existing behavior)
5. Add `SmartContextEngine` as opt-in

### Phase 5: Multi-Model Review + Citation Check (Day 10-12)

1. Add `review_committee` config option ([02-review-citations.md](02-review-citations.md))
2. Implement `_multi_reviewer_assessment()` in review.py
3. Add `review_citation_checker.py`
4. Integrate citation checking into review consistency_issues
5. Both features are backward-compatible (single-model fallback)

### Phase 6: File Splitting (Day 13-16)

1. Split execution.py (Section 6.1) — start with result_collector, then resource_matcher
2. Split writing.py (Section 6.2) — start with grounding.py, then context_builder
3. Split experiment.py (Section 6.3) — start with edit_apply
4. Run full test suite after each file extraction

### Phase 7: Prompt Externalization (Day 17-19)

1. Create `prompts/` directory structure ([05-prompts-search-coding.md](05-prompts-search-coding.md))
2. Extract prompts one category at a time (start with figure_gen — largest)
3. Verify output quality matches before and after for each prompt
4. Add PromptLoader with caching

### Phase 8: IDEATION + CODING Improvements (Day 20-22)

1. Add search coverage self-evaluation (Section 9)
2. Replace regex import checker with AST (Section 10.1)
3. Add smoke test generation (Section 10.2)
4. Add auto-formatting (Section 10.3)

### Phase 9: Infrastructure (Day 23-25)

1. Cost tracking (Section 11 in [06-infrastructure.md](06-infrastructure.md))
2. Progress streaming (Section 14)
3. Structured logging (Section 15)
4. Paper quality benchmark (Section 16 in [07-quality-validation.md](07-quality-validation.md))
5. Blueprint semantic validation (Section 17)

### Phase 10: DAG Scheduling (Day 26-28)

1. Implement DAG scheduler (Section 13 in [06-infrastructure.md](06-infrastructure.md))
2. Add `parallel_stages` config option
3. Test with FIGURE_GEN + WRITING parallelization
4. Default OFF, opt-in only

---

## Risk Assessment per Change

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

## Checklist for New Developer

Before starting implementation:

- [ ] Read all 10 architecture docs in order
- [ ] Read `MEMORY.md` in the project root for fix history context
- [ ] Read `architecture_improvements.md` (V1) for prior architectural decisions
- [ ] Run `python -m pytest tests/` to establish baseline (all tests pass)
- [ ] Run one full `fast_draft` pipeline to understand the end-to-end flow
- [ ] Set up a test topic that completes in <10 minutes for iteration

During implementation:

- [ ] Follow the phase order above (bug fixes first, infrastructure last)
- [ ] Commit after each individual change, not in bulk
- [ ] Write tests for new code BEFORE integrating with existing modules
- [ ] Never modify prompt text during the externalization phase — only move it
- [ ] When splitting files, do ONE file at a time, run tests, then continue
- [ ] Check risk assessment table before each change

After implementation:

- [ ] Full test suite passes
- [ ] `fast_draft` pipeline completes end-to-end
- [ ] `local_quick` pipeline completes with real experiment execution
- [ ] Resume from checkpoint works (kill mid-WRITING, resume, verify PDF generated)
- [ ] LaTeX compilation succeeds with tectonic
- [ ] Multi-model review works with at least 2 different providers
- [ ] Citation fact-checker catches a known inaccurate citation in a test paper
- [ ] Memory persists across two consecutive runs on the same topic
