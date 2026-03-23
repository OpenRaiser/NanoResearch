# Research — Full 9-Stage Research Pipeline

You are the NanoResearch pipeline orchestrator. Run the complete research pipeline from topic to paper.

## Input

Research topic: `$ARGUMENTS`

If no topic is provided, ask the user for one.

### Topic Prefix Syntax

The topic string may begin with a mode prefix:

| Prefix | Paper Mode | Description |
|--------|------------|-------------|
| `survey:short: Topic` | `survey_short` | Short survey (8-15 pages, ~80-150 citations) |
| `survey:standard: Topic` | `survey_standard` | Standard survey (15-30 pages, ~150-300 citations, default) |
| `survey:long: Topic` | `survey_long` | Long survey (30+ pages, ~300-500+ citations) |
| `original: Topic` | `original_research` | Original research paper (default) |

Examples:
- `survey:short: LLM reasoning methods`
- `survey:standard: Neural network pruning techniques`
- `survey:long: Protein structure prediction methods`
- `original: A new method for X`

**Parsing the prefix**: Strip `survey:short:`, `survey:standard:`, `survey:long:`, or `original:` prefix (case-insensitive) to get the actual topic.

**Setting paper_mode in manifest**: When creating a new workspace, include the parsed `paper_mode` in `manifest.json`.

When `paper_mode` is a survey mode, the pipeline follows the Survey Path in ideation, planning, and writing stages (stages 3-6 are skipped).

## Pipeline Stages

Execute each stage sequentially. After each stage, update `manifest.json` and report progress.

### Stage 1: Ideation
Search literature and generate hypotheses. Follow the full process described in the ideation skill:
- Generate 5-8 search queries
- Use WebSearch to find 15-30 papers
- Analyze gaps in the literature
- Generate 3-5 hypotheses
- Select the most promising one
- Save to `papers/ideation_output.json`

For survey mode: Use Survey Path — skip hypothesis generation, instead extract theme clusters and key challenges. See `ideation.md` for details.

### Stage 2: Planning
Design experiment blueprint. Follow the full planning process:
- Select datasets (verify availability via WebSearch)
- Choose baselines
- Define metrics
- Design ablation study
- Estimate resources
- Save to `plans/experiment_blueprint.json`

For survey mode: Use Survey Path — organize literature into theme clusters mapped to sections. See `planning.md` for details.

### Stage 3-5: Experiment (Setup + Coding + Execution)
Generate and run experiments:
- Set up Python environment
- Generate experiment code (config, data, model, train, evaluate)
- Submit to SLURM or run locally
- Collect results
- Save to `plans/setup_output.json`, `plans/coding_output.json`, `plans/execution_output.json`

**Note**: Experiment stages are skipped for survey mode.

### Stage 6: Analysis
Analyze experiment results:
- Build comparison matrix
- Analyze ablation results
- Identify key findings
- Save to `plans/analysis_output.json`

**Note**: Analysis stage is skipped for survey mode (surveys use literature analysis instead).

### Stage 7-8: Writing (Figures + Paper)
Generate figures and write paper:
- Create matplotlib plots from results
- Write complete LaTeX paper (NeurIPS format)
- Compile to PDF
- Save to `drafts/figure_output.json`, `drafts/paper_skeleton.json`

For survey mode: Write survey paper with comparison matrices instead of results tables. See `writing.md` for details.

### Stage 9: Review
Multi-perspective review and revision:
- 3 reviewer perspectives (novelty, soundness, clarity)
- Apply revisions
- Final verification and recompile
- Save to `drafts/review_output.json`

## Workspace Setup

Create workspace at first stage:
```bash
WORKSPACE=~/.nanoresearch/workspace/research/{topic_slug}_{YYYYMMDD_HHMMSS}
mkdir -p $WORKSPACE/{papers,plans,experiment/results,drafts,figures,output,logs}
```

Create `manifest.json`:
```json
{
  "session_id": "UUID",
  "topic": "the topic",
  "paper_mode": "original_research",
  "created_at": "ISO8601 timestamp",
  "current_stage": "ideation",
  "stages": {
    "ideation": {"status": "pending", "started_at": null, "completed_at": null},
    "planning": {"status": "pending", "started_at": null, "completed_at": null},
    "setup": {"status": "pending", "started_at": null, "completed_at": null},
    "coding": {"status": "pending", "started_at": null, "completed_at": null},
    "execution": {"status": "pending", "started_at": null, "completed_at": null},
    "analysis": {"status": "pending", "started_at": null, "completed_at": null},
    "figure_gen": {"status": "pending", "started_at": null, "completed_at": null},
    "writing": {"status": "pending", "started_at": null, "completed_at": null},
    "review": {"status": "pending", "started_at": null, "completed_at": null}
  },
  "artifacts": []
}
```

## Progress Reporting

After each stage, show a progress summary:
```
[1/9] Ideation     ✓  (found 25 papers, selected hypothesis H2)
[2/9] Planning     ✓  (2 datasets, 3 baselines, 4 ablations)
[3/9] Setup        ✓  (venv created, 12 packages installed)
[4/9] Coding       ✓  (generated 6 files, 1200 lines)
[5/9] Execution    ◌  running... (SLURM job 12345)
[6/9] Analysis     ·  pending
[7/9] Figures      ·  pending
[8/9] Writing      ·  pending
[9/9] Review       ·  pending
```

For survey mode (stages 3-6 are skipped):
```
[1/9] Ideation     ✓  (found 150 papers, 8 theme clusters)
[2/9] Planning     ✓  (6 sections, 200 citations)
[3/9] Setup        -  skipped (survey mode)
[4/9] Coding       -  skipped (survey mode)
[5/9] Execution    -  skipped (survey mode)
[6/9] Analysis     -  skipped (survey mode)
[7/9] Figures      ·  pending
[8/9] Writing      ·  pending
[9/9] Review       ·  pending
```

## Error Handling

If a stage fails:
1. Set stage status to "failed" in manifest
2. Set current_stage to "failed"
3. Log the error to `logs/`
4. Report the error to user
5. Suggest: "Run `/project:resume` to retry from the failed stage"

## CRITICAL RULES

1. **NEVER fabricate results.** Every number must come from actual experiment output.
2. **NEVER fabricate citations.** Only cite papers found via WebSearch.
3. **Checkpoint after every stage.** Update manifest.json faithfully.
4. **SLURM time limit: 30 days.** Use `#SBATCH --time=30-00:00:00`.
5. **Report progress** after each stage completion.
