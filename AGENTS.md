# NanoResearch — Codex Integration Mode

NanoResearch is an end-to-end autonomous AI research engine. In Codex integration mode, Codex should drive the existing pipeline rather than inventing a second workflow.

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
- the prefix is parsed by the existing CLI and manifest logic; Codex should reuse that behavior

## Codex Intent Mapping

| If the user asks for... | Codex should interpret it as... | Preferred repo entry / behavior |
| --- | --- | --- |
| full research run | topic-to-paper pipeline | `nanoresearch run --topic "..."` or the equivalent workspace-driven pipeline flow |
| ideation or literature survey | stage 1 ideation | produce `papers/ideation_output.json` in a workspace |
| planning | stage 2 planning | produce `plans/experiment_blueprint.json` or `plans/survey_blueprint.json` |
| experiment execution | setup + coding + execution | use the existing experiment path for original research |
| analysis | stage 6 evidence extraction | produce `plans/analysis_output.json` |
| writing | figure generation + paper drafting | produce or update paper outputs in the workspace |
| review | review + revision pass | produce or update `drafts/review_output.json` and revised paper assets |
| status | inspect the active workspace | read and normalize `manifest.json` |
| resume | continue an interrupted run | restart from the first non-completed stage |

## Grounding Rules

- never fabricate experiment results
- never fabricate citations
- prefer existing CLI / orchestrator behavior over ad hoc scripts
- keep paper claims tied to actual outputs or verified literature
- preserve checkpoint and resume semantics via `manifest.json`

## Codex Operating Rules

1. Prefer the existing NanoResearch CLI or Python entrypoints when driving the system.
2. Do not introduce a separate Codex-specific pipeline mode.
3. Keep outputs compatible with existing workspaces and manifests.
4. Never fabricate results or citations.
5. When a user asks for pipeline work, operate through the existing workspace artifacts and stage boundaries described above.
