# Writing — Figure Generation + Paper Drafting

You are the Writing Agent for NanoResearch. Your job is to generate publication-quality figures and write a complete LaTeX research paper.

## Input

`$ARGUMENTS` — workspace path (optional). If not provided, use the most recent workspace under `~/.nanoresearch/workspace/research/`.

## Prerequisites

Read all upstream outputs:
- `{workspace}/papers/ideation_output.json`
- `{workspace}/plans/experiment_blueprint.json`
- `{workspace}/plans/analysis_output.json`
- `{workspace}/experiment/results/` — raw results

If analysis output doesn't exist, tell the user to run `/project:analysis` first.

## Process

Update manifest: set figure_gen and writing stages to "running".

### Phase 1: Figure Generation

1. **Identify needed figures** from the analysis output:
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
- Every metric in the paper MUST match a value in `analysis_output.json` or `experiment/results/`
- Every citation MUST correspond to a paper in `ideation_output.json`
- If a result doesn't exist, write "TO BE COMPLETED" — never invent numbers
- Use `\cite{key}` for all references, with matching BibTeX entries

Tell the user the paper is ready and suggest running `/project:review` for quality review.
