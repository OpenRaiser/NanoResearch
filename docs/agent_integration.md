# NanoResearch Agent Integration Reference

NanoResearch is an end-to-end autonomous AI research engine. Agent integrations should drive the existing pipeline rather than inventing a second workflow.

## Core Goal

Given a research topic, NanoResearch should produce a resumable research workspace containing:
- literature artifacts
- planning artifacts
- runnable experiment code when needed
- execution or literature-analysis evidence
- figures
- a LaTeX paper draft
- review output and final exported assets

## Pipeline

NanoResearch uses a 9-stage pipeline:

```text
IDEATION -> PLANNING -> SETUP -> CODING -> EXECUTION -> ANALYSIS -> FIGURE_GEN -> WRITING -> REVIEW
```

Stage meanings:
- `ideation`: literature search, gap finding, hypothesis or theme extraction
- `planning`: experiment blueprint or survey blueprint generation
- `setup`: environment and resource preparation
- `coding`: runnable experiment generation
- `execution`: local or SLURM-backed experiment execution
- `analysis`: structured evidence extraction from outputs
- `figure_gen`: figure generation for paper assets
- `writing`: LaTeX paper drafting
- `review`: critique, verification, and revision

## Workspace Convention

Workspaces live under `~/.nanoresearch/workspace/research/`.
A typical workspace contains:

```text
{session_dir}/
├── manifest.json
├── papers/
├── plans/
├── experiment/
├── drafts/
├── figures/
├── output/
└── logs/
```

Agents should prefer reusing an existing workspace when the user asks to continue, inspect status, resume, or revise a prior run.

## Paper Modes

NanoResearch supports both original research and survey papers.

Topic prefixes:
- `original: Topic` -> `original_research`
- `survey:short: Topic` -> `survey_short`
- `survey:standard: Topic` -> `survey_standard`
- `survey:long: Topic` -> `survey_long`

Behavior:
- original research follows the full 9-stage pipeline
- survey modes skip experiment-heavy stages and use literature-grounded planning, writing, and review
- the prefix is parsed by the existing CLI and manifest logic; integrations should reuse that behavior

## Entry Mapping

Agent integrations should map user intent onto the existing NanoResearch workflow:

| User intent | Preferred repo behavior | Expected primary outputs |
| --- | --- | --- |
| `research` | run the full pipeline for a topic | full workspace, `manifest.json`, paper outputs |
| `ideation` | create or continue a workspace and generate literature output | `papers/ideation_output.json` |
| `planning` | read ideation output and produce an experiment or survey blueprint | `plans/experiment_blueprint.json` or `plans/survey_blueprint.json` |
| `experiment` | perform setup, coding, and execution for original research | `plans/setup_output.json`, `plans/coding_output.json`, `plans/execution_output.json` |
| `analysis` | convert experiment outputs into structured findings | `plans/analysis_output.json` |
| `writing` | generate figures and the paper draft | `drafts/figure_output.json`, `drafts/paper_skeleton.json`, `output/main.tex`, `output/main.pdf` |
| `review` | critique, revise, and verify the paper | `drafts/review_output.json` |
| `status` | inspect the most relevant workspace and normalize manifest state | manifest summary + artifact list |
| `resume` | continue from the first non-completed stage | updated workspace state and downstream artifacts |

## Grounding Rules

All integrations must preserve NanoResearch's grounding guarantees:
- never fabricate experiment results
- never fabricate citations
- prefer existing CLI / orchestrator behavior over ad hoc scripts
- keep paper claims tied to actual outputs or verified literature
- preserve checkpoint and resume semantics via `manifest.json`

## Environment and Safety

- Python CLI remains the source of truth for API-driven runs
- Agent integrations may use Bash, SLURM, file reads/writes, and web search if the host agent platform supports them
- For GPU jobs, prefer existing repo conventions and execution profiles rather than inventing new scheduler logic
- Do not add a separate `codex` or `claude` execution mode at the pipeline layer unless the product intentionally changes
