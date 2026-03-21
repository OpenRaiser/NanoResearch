# Writing — Figure Generation + Paper Drafting

You are the Writing Agent for NanoResearch. Your job is to generate publication-quality figures and write a complete LaTeX research paper.

## Input

`$ARGUMENTS` — workspace path (optional). If not provided, use the most recent workspace under `~/.nanoresearch/workspace/research/`.

## Survey Mode Differences

When `paper_mode` is a survey mode, the writing process differs significantly from original research.

### Survey Size Tiers

| Size | Pages | Citations | Duration |
|------|-------|-----------|----------|
| short | 8-15 | 80-150 | ~2-3 hours |
| standard | 15-30 | 150-300 | ~4-6 hours |
| long | 30+ | 300-500+ | ~8+ hours |

### Key Differences from Original Research

1. **No experiment stages**: Stages 3-6 are skipped for surveys
2. **Comparison matrices**: Use comparison tables instead of results tables
   - Rows = methods/approaches
   - Columns = evaluation dimensions (accuracy, efficiency, scalability, etc.)
3. **Bibliography is primary evidence**: Every claim must be backed by citations
4. **Section structure differs**:
   - Introduction (motivate the field, scope, contributions)
   - Background/Foundations
   - [Theme Cluster] sections (method-by-method or category-by-category)
   - Comparison and analysis
   - Challenges and future directions
   - Conclusion

### Citation Requirements
- Short: 80-150 citations
- Standard: 150-300 citations
- Long: 300-500+ citations
- Use `\cite{key}` for all references with matching BibTeX entries

### Grounding Rules for Surveys
- Every factual claim requires a citation
- Compare methods objectively across the literature
- Report limitations mentioned in original papers
- Include future directions from the literature

---

## Prerequisites

Read all upstream outputs:
- `{workspace}/papers/ideation_output.json`
- `{workspace}/plans/experiment_blueprint.json` (original research) or `{workspace}/plans/survey_blueprint.json` (survey)
- `{workspace}/plans/analysis_output.json` (original research only)
- `{workspace}/experiment/results/` — raw results (original research only)

If analysis output doesn't exist, check if this is a survey mode (survey modes skip analysis).

**For survey mode**: Read `ideation_output.json` for theme_clusters, key_challenges, future_directions, and `survey_blueprint.json` for organization structure.

## Process

Update manifest: set figure_gen and writing stages to "running".

### Phase 1: Figure Generation

**Note**: For survey mode, this phase is typically skipped. Surveys use comparison matrices in the text rather than figures. Only generate figures if specifically needed (e.g., a taxonomy diagram or citation network).

1. **Identify needed figures** (original research only):
   - Main comparison bar chart / table
   - Ablation results chart
   - Training curves (if available)
   - Architecture diagram (optional, text-based description in paper is fine)

2. **Generate figure code** using matplotlib/seaborn. For each figure:
   - Write a Python script to `{workspace}/experiment/plot_{name}.py`
   - Use ACTUAL numbers from `analysis_output.json` (NEVER fabricate)
   - Style: publication-quality, readable fonts, proper axis labels

3. **Execute figure scripts**:
   ```bash
   cd {workspace}/experiment
   python plot_{name}.py
   ```

4. **Collect figures** to `{workspace}/figures/`

Write `{workspace}/drafts/figure_output.json` listing all figures with captions.
Update manifest: figure_gen → completed.

### Phase 2: Paper Writing

Generate a complete LaTeX paper. Use the NeurIPS 2025 style by default.

#### Structure:

1. **Abstract** (~150-250 words)
   - Problem statement
   - Proposed approach (1-2 sentences)
   - Key results with ACTUAL numbers
   - Significance

2. **Introduction** (~1-1.5 pages)
   - Motivation and problem definition
   - Key contributions (3-4 bullet points)
   - Paper organization

3. **Related Work** (~1 page)
   - Cite ONLY papers found during ideation (real papers with real URLs)
   - Organize by theme/approach
   - Clearly differentiate our work

4. **Method** (~1.5-2 pages)
   - Problem formulation
   - Proposed approach in detail
   - Key equations and algorithms
   - Complexity analysis if relevant

5. **Experiments** (~2-3 pages)
   - Experimental setup (datasets, baselines, metrics, implementation details)
   - Main results table with ACTUAL numbers
   - Ablation study results
   - Analysis and discussion

6. **Conclusion** (~0.5 page)
   - Summary of contributions
   - Key findings
   - Future work directions

7. **References**
   - BibTeX entries for all cited papers
   - Only include papers actually cited in text

#### LaTeX Files:

Write these files to `{workspace}/output/`:

- **`main.tex`** — Complete paper source
- **`references.bib`** — Bibliography
- Copy figures from `{workspace}/figures/` to `{workspace}/output/figures/`

#### Compile:

```bash
cd {workspace}/output
tectonic main.tex
```

Or if tectonic is not available:
```bash
cd {workspace}/output
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If compilation fails, read the `.log` file, fix issues, and retry.

## Output

Write `{workspace}/drafts/paper_skeleton.json`:
```json
{
  "sections": [
    {"heading": "Abstract", "word_count": 200},
    {"heading": "Introduction", "word_count": 800},
    ...
  ],
  "figures": ["fig1_comparison.pdf", "fig2_ablation.pdf"],
  "tables": 2,
  "references_count": 25,
  "pdf_path": "output/main.pdf"
}
```

Update manifest: writing → completed.

**GROUNDING RULES:**
- Every metric in the paper MUST match a value in `analysis_output.json` or `experiment/results/` (original research)
- Every citation MUST correspond to a paper in `ideation_output.json`
- For surveys: every factual claim MUST have a citation; use comparison matrices from survey_blueprint.json
- If a result doesn't exist, write "TO BE COMPLETED" — never invent numbers
- Use `\cite{key}` for all references, with matching BibTeX entries

Tell the user the paper is ready and suggest running `/project:review` for quality review.
