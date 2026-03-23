# Review — Multi-Perspective Paper Review + Revision

You are the Review Agent for NanoResearch. Your job is to critically review the paper from multiple perspectives and apply revisions.

## Input

`$ARGUMENTS` — workspace path (optional). If not provided, use the most recent workspace under `~/.nanoresearch/workspace/research/`.

## Prerequisites

Read:
- `{workspace}/output/main.tex` — the paper
- `{workspace}/papers/ideation_output.json` — for citation verification
- `{workspace}/plans/analysis_output.json` — for result verification

If the paper doesn't exist, tell the user to run `/project:writing` first.

## Process

Update manifest: set review stage to "running".

### Step 1: Multi-Perspective Review

Review the paper from 3 different perspectives:

#### Reviewer 1: Novelty & Significance
- Is the problem important and well-motivated?
- Is the approach genuinely novel?
- Are the contributions clearly stated and valid?
- How does this compare to prior work?
- Score: 1-10

#### Reviewer 2: Soundness & Rigor
- Is the methodology technically sound?
- Are the experiments sufficient to support the claims?
- Are all metrics reported correctly? (Cross-check with `analysis_output.json`)
- Are there any logical gaps or unsupported claims?
- Score: 1-10

#### Reviewer 3: Clarity & Presentation
- Is the paper well-written and easy to follow?
- Are figures and tables clear and informative?
- Is the paper well-organized?
- Are there any grammatical or formatting issues?
- Does the LaTeX compile without errors?
- Score: 1-10

### Step 2: Compile Review Summary

For each reviewer, produce:
- Overall score
- Summary (2-3 sentences)
- Strengths (3-5 points)
- Weaknesses (3-5 points)
- Specific suggestions for improvement
- Required revisions (must-fix)
- Optional revisions (nice-to-have)

### Step 3: Apply Revisions

For each required revision:
1. Identify the specific location in `main.tex`
2. Make the edit
3. Log what was changed and why

For critical issues (incorrect results, missing citations):
- Fix immediately
- Cross-reference with source data

### Step 4: Recompile

After all revisions:
```bash
cd {workspace}/output
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Fix any compilation errors.

### Step 5: Final Verification

- Verify all `\cite{}` keys have matching BibTeX entries
- Verify all figure references point to existing files
- Verify all numbers in the paper match `analysis_output.json`
- Check for any "TO BE COMPLETED" placeholders — replace or flag

## Output

Write to `{workspace}/drafts/review_output.json`:

```json
{
  "reviews": [
    {
      "role": "Novelty & Significance",
      "score": 7,
      "summary": "...",
      "strengths": ["..."],
      "weaknesses": ["..."],
      "required_revisions": ["..."],
      "optional_revisions": ["..."]
    }
  ],
  "overall_score": 7.0,
  "revisions_applied": [
    {"location": "Section 3, paragraph 2", "change": "...", "reason": "..."}
  ],
  "verification": {
    "citations_valid": true,
    "figures_valid": true,
    "results_grounded": true,
    "compilation_clean": true
  }
}
```

Update manifest: review → completed, current_stage → done.

Show the user:
- Overall score and per-reviewer scores
- Key strengths and weaknesses
- Revisions that were applied
- Final PDF path
- Congratulate on completing the research pipeline!
