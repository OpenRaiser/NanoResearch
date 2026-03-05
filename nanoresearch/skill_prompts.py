"""Condensed skill guidance extracted from K-Dense scientific skills.

Each constant provides focused, actionable instructions for a specific
pipeline stage.  They are injected into agent system prompts to improve
output quality without blowing up token budgets (~500-800 tokens each).
"""

# ---------------------------------------------------------------------------
# IDEATION — from literature-review + scientific-brainstorming
# ---------------------------------------------------------------------------
IDEATION_SKILL = """
=== LITERATURE REVIEW STANDARDS ===

SEARCH STRATEGY:
- Use 2-4 main concepts per query with synonyms and abbreviations.
- Boolean operators: combine concepts with AND, use OR for synonyms.
- Search at least 2 complementary databases (Semantic Scholar + arXiv).
- Use citation chaining: trace forward citations of key papers.

PAPER QUALITY TIERS (prioritize higher tiers):
- Tier 1: Nature, Science, Cell, PNAS, NeurIPS, ICML, ICLR, CVPR, ACL
- Tier 2: High-impact specialized journals (IF>10), top workshops
- Tier 3: Specialized journals (IF 5-10)
- Tier 4: Lower-impact peer-reviewed (use sparingly)

CITATION COUNT SIGNIFICANCE (by age):
- 0-3 years: 20+ noteworthy, 100+ highly influential
- 3-7 years: 100+ significant, 500+ landmark
- 7+ years: 500+ seminal, 1000+ foundational

SYNTHESIS RULES:
- Organize findings thematically, NOT study-by-study.
- Identify consensus areas AND controversies.
- Extract specific quantitative comparisons (acc, F1, latency).
- Note methodological variations and their impact on results.
- Identify seminal/foundational papers that MUST be cited.

HYPOTHESIS GENERATION:
- Each hypothesis must be testable with concrete predictions.
- State expected quantitative improvements over baselines.
- Identify which components drive the improvement (for ablation).
- Consider failure modes and boundary conditions.
"""

# ---------------------------------------------------------------------------
# WRITING — from scientific-writing + citation-management
# ---------------------------------------------------------------------------
WRITING_SKILL = """
=== SCIENTIFIC WRITING STANDARDS ===

TWO-STAGE PROCESS (apply for each section):
Stage 1 — OUTLINE: Identify key points, arguments, data to present, papers to cite.
Stage 2 — PROSE: Convert outline into flowing paragraphs. Every sentence must connect
    logically to the next. Add transitions (however, moreover, in contrast).

SECTION-SPECIFIC RULES:
- Introduction: Importance → Literature gaps → Research questions → Novelty + contributions
- Related Work: Thematic synthesis (NOT study-by-study listing). Compare approaches,
  highlight limitations, position your method.
- Methods: Reproducible detail. State design choices with justification.
  Include equations for all non-trivial operations.
- Experiments: Datasets, baselines, metrics, implementation details (lr, batch, epochs).
  Present results with analysis — don't just state numbers.
- Discussion/Conclusion: Relate to questions, compare with literature, acknowledge
  limitations honestly, suggest future directions.

CITATION INTEGRATION:
- Embed citations naturally within sentences, not as disconnected lists.
- GOOD: "\\citet{smith2023} first demonstrated X, which \\citet{jones2024} extended to Y."
- BAD: "Several studies have shown this \\citep{smith2023, jones2024, lee2022}."
- Use \\citet when authors are grammatical subjects, \\citep for parenthetical.
- Balance citations: every claim needs support, but don't over-cite (3-5 refs per paragraph max).

ANTI-PATTERNS (never do these):
- Bullet points in any final section (only allowed in Methods for criteria lists)
- Labeled abstract sub-sections (Background:, Methods:, etc.)
- Mixing tenses: past for methods/results, present for established facts/discussion
- Vague filler: "it is well known", "has attracted attention", "in recent years"
- Study-by-study literature summaries: "X did A. Y did B. Z did C."

BIBTEX RULES:
- Citation key format: FirstAuthorYYYYkeyword (e.g., smith2023attention)
- Use -- for page ranges (not single dash)
- Always include DOI when available
- Protect capitalization with braces: title = {{BERT}: Pre-Training of ...}
"""

# ---------------------------------------------------------------------------
# REVIEW — from peer-review + scholar-evaluation (ScholarEval framework)
# ---------------------------------------------------------------------------
REVIEW_SKILL = """
=== SCHOLAREVAL REVIEW FRAMEWORK ===

EVALUATE ACROSS 8 DIMENSIONS (assess each):
1. Problem Formulation — clarity, significance, novelty of research questions
2. Literature Review — comprehensiveness, critical synthesis, gap identification
3. Methodology — rigor, reproducibility, appropriateness for research questions
4. Data/Evidence — quality, sample size, source credibility
5. Analysis — method appropriateness, logical coherence, alternatives considered
6. Results — clarity of presentation, statistical rigor, visualization quality
7. Writing Quality — organization, logical flow, accessibility, notation consistency
8. Citations — completeness, accuracy, balance of perspectives

SCORING SCALE (per section):
  9-10: Publication-ready at top venue, only minor polishing
  7-8: Solid contribution with identifiable but fixable weaknesses
  5-6: Significant issues that need substantial addressing
  3-4: Major rewrite needed, fundamental gaps in one or more dimensions
  1-2: Fundamentally flawed, would require complete rethinking

STRUCTURED FEEDBACK FORMAT:
For each section produce:
- Score (integer 1-10)
- Major Issues: each stating (a) the problem, (b) why it matters, (c) specific fix
- Minor Issues: with location and suggestion
- DO NOT give vague criticism — every issue must be actionable

RED FLAGS TO CATCH:
- Overstated conclusions not supported by experimental evidence
- Missing ablation for claimed contributions
- Inconsistent notation between sections
- Missing error bars, std, or confidence intervals in result tables
- Causal claims from correlational evidence
- Selective reporting (only showing favorable metrics/datasets)
- Claims of SOTA without proper baselines or statistical testing
- Related work missing seminal papers in the field

REVISION GUIDANCE:
- Each issue must map to a concrete edit (add equation, add citation, rewrite paragraph)
- Prioritize: correctness > completeness > clarity > style
- For low-score sections (<=5): provide a structural outline of what the section should contain
"""

# ---------------------------------------------------------------------------
# FIGURE_GEN — from scientific-visualization (supplements existing CHART_CODE_SYSTEM)
# ---------------------------------------------------------------------------
FIGURE_SKILL = """
=== PUBLICATION FIGURE QUALITY CHECKLIST ===

MANDATORY FOR EVERY FIGURE:
- All axes labeled with descriptive name AND units: "Accuracy (%)", "Latency (ms)"
- Error bars present when data has variance (define SD/SEM/CI in caption)
- Sample size (n) stated in caption or on figure
- No title inside figure — LaTeX caption serves as title

CHART TYPE SELECTION:
- Bar charts: comparing discrete categories/methods (main results, ablation)
- Line plots: trends over continuous variable (training curves, scaling)
- Heatmaps: correlation matrices, attention maps, confusion matrices
- Box/violin plots: distribution comparison across groups
- Scatter plots: relationship between two continuous variables

ACCESSIBILITY (non-negotiable):
- Use Okabe-Ito palette (already in rcParams)
- Add redundant encoding: different line styles (-, --, :, -.) AND markers (o, s, D, ^)
- Add hatching patterns (/, \\, x, o) for bar charts
- Ensure figure is readable in grayscale
- Never use jet/rainbow colormaps

MULTI-PANEL LAYOUT:
- Label panels (a), (b), (c) in upper-left, bold, fontsize=12
- Consistent styling across panels (same font, colors, line widths)
- Shared legend below figure when panels share the same categories
- Reserve space: plt.subplots_adjust(bottom=0.18) for bottom legend

COMMON MISTAKES TO AVOID:
- Font too small (unreadable at print size)
- Truncated axis that exaggerates differences
- 3D effects or chart junk
- Missing units on axes
- Inconsistent color assignment across figures (same method = same color everywhere)
"""
