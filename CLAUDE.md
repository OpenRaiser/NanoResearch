# NanoResearch — Claude Code Integration Mode

NanoResearch is an end-to-end autonomous AI research engine. In Claude Code integration mode, **you ARE the research engine** — no external API keys needed.

## How It Works

You drive a 9-stage research pipeline entirely through Claude Code's native capabilities:
- **WebSearch** for literature retrieval (arXiv, Semantic Scholar, Google Scholar)
- **Bash** for code execution, SLURM job submission, LaTeX compilation
- **File read/write** for generating code, papers, and structured artifacts

## Workspace Convention

All research sessions live under `~/.nanoresearch/workspace/research/`. Each session has:

```
{session_dir}/
├── manifest.json          # Pipeline state tracker
├── papers/
│   └── ideation_output.json
├── plans/
│   ├── experiment_blueprint.json
│   ├── setup_output.json
│   ├── coding_output.json
│   ├── execution_output.json
│   └── analysis_output.json
├── experiment/            # Generated code + results
│   ├── *.py
│   ├── requirements.txt
│   └── results/
├── drafts/
│   ├── paper_skeleton.json
│   ├── figure_output.json
│   └── review_output.json
├── figures/
├── output/                # Final export
│   ├── main.tex
│   ├── references.bib
│   ├── main.pdf
│   └── figures/
└── logs/
```

## Available Commands

| Command | Description |
|---------|-------------|
| `/project:research` | Run the full 9-stage pipeline (topic as argument) |
| `/project:ideation` | Stage 1: Literature search + hypothesis generation |
| `/project:planning` | Stage 2: Experiment blueprint design |
| `/project:experiment` | Stages 3-5: Setup + code generation + execution |
| `/project:analysis` | Stage 6: Results analysis |
| `/project:writing` | Stages 7-8: Figure generation + paper writing |
| `/project:review` | Stage 9: Multi-perspective review + revision |
| `/project:status` | Show current pipeline status |
| `/project:resume` | Resume pipeline from last checkpoint |

## Critical Rules

1. **NEVER fabricate results.** Every number, metric, and claim must come from actual experiment output files. If results don't exist yet, say so — don't invent them.
2. **NEVER fabricate citations.** Only cite papers you found via WebSearch with real titles, authors, and years. Use placeholder `\cite{tbd}` if unsure.
3. **All code must be runnable.** Generated Python code should include proper imports, error handling, and be testable.
4. **Checkpoint after each stage.** Update `manifest.json` after completing each stage so the pipeline can resume.
5. **Use SLURM for GPU jobs.** Submit via `sbatch` with `--time=30-00:00:00` (30 days). Check GPU availability first.

## SLURM Convention

- Partition: auto-detect (or set in manifest)
- Time limit: `#SBATCH --time=30-00:00:00` (30 days, NEVER use short limits)
- Check availability: `sinfo -p belt_road -o "%P %a %D %C"` before submitting

## Manifest Format

Two manifest schemas exist in `~/.nanoresearch/workspace/research/`. **All commands must handle both.**

### New schema (Claude Code commands — use this for new workspaces)

```json
{
  "session_id": "uuid",
  "topic": "...",
  "created_at": "ISO8601",
  "current_stage": "ideation|planning|setup|coding|execution|analysis|figure_gen|writing|review|done|failed",
  "stages": {
    "ideation": {"status": "pending|running|completed|failed", "started_at": null, "completed_at": null},
    ...
  },
  "artifacts": ["papers/ideation_output.json", "plans/experiment_blueprint.json"]
}
```

### Old schema (Python pipeline, `schema_version: "1.1"`)

- Stage keys are UPPERCASE: `IDEATION`, `PLANNING`, `SETUP`, `CODING`, `EXECUTION`, `ANALYSIS`, `FIGURE_GEN`, `WRITING`, `REVIEW`
- `current_stage` is UPPERCASE: `DONE`, `FAILED`, etc.
- May have extra stages: `INIT`, `FORMAT_FIX` — skip these
- Stage objects have extra fields: `stage`, `retries`, `error_message`, `output_path`
- `artifacts` is `[{name, path, stage, created_at, checksum}]`
- Has `config_snapshot`, `pipeline_mode`, `updated_at`

### Normalization rules (for status/resume/all reads)

1. Convert all stage keys and `current_stage` to **lowercase**
2. Canonical stage order: `ideation`, `planning`, `setup`, `coding`, `execution`, `analysis`, `figure_gen`, `writing`, `review`
3. Skip stages not in canonical list (`init`, `format_fix`)
4. For artifacts: if entry is an object, use its `path` field; if a string, use directly
5. Status comparison is case-insensitive: `"completed"` = `"COMPLETED"`
