# NanoResearch — Claude Code Integration Mode

NanoResearch is an end-to-end autonomous AI research engine. In Claude Code integration mode, Claude Code should drive the existing pipeline rather than inventing a second workflow.

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

Reuse an existing workspace when the user asks to continue, inspect status, resume, or revise a prior run.

## Paper Modes

Topic prefixes:
- `original: Topic` -> `original_research`
- `survey:short: Topic` -> `survey_short`
- `survey:standard: Topic` -> `survey_standard`
- `survey:long: Topic` -> `survey_long`

Behavior:
- original research follows the full 9-stage pipeline
- survey modes skip experiment-heavy stages and use literature-grounded planning, writing, and review
- the prefix is parsed by the existing CLI and manifest logic; Claude Code should reuse that behavior

## Available Commands

| Command | Description |
| --- | --- |
| `/project:research` | Run the full 9-stage pipeline for a topic |
| `/project:ideation` | Run stage 1 literature search and idea generation |
| `/project:planning` | Run stage 2 planning |
| `/project:experiment` | Run setup, coding, and execution for original research |
| `/project:analysis` | Run experiment analysis |
| `/project:writing` | Run figure generation and writing |
| `/project:review` | Run review and revision |
| `/project:status` | Show workspace status |
| `/project:resume` | Resume from the last checkpoint |

## Grounding Rules

- never fabricate experiment results
- never fabricate citations
- prefer existing CLI / orchestrator behavior over ad hoc scripts
- keep paper claims tied to actual outputs or verified literature
- preserve checkpoint and resume semantics via `manifest.json`

## Claude Code Role

In Claude Code integration mode, Claude Code acts as the research engine using its native tools:
- **WebSearch** for literature retrieval
- **Bash** for code execution, SLURM submission, and LaTeX compilation
- **File read/write** for workspace artifacts, code, and paper drafts

## Claude Code Rules

1. Use the existing workspace and manifest conventions above.
2. Reuse the existing topic prefix syntax for survey modes.
3. Prefer NanoResearch's existing pipeline and outputs over custom one-off scripts.
4. Never fabricate results or citations.
5. Keep workspaces compatible with the Python CLI.
