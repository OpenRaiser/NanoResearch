# NanoResearch — Codex Integration Mode

This file is the Codex entrypoint for NanoResearch.

For shared project workflow, paper modes, workspace conventions, and stage semantics, read [`docs/agent_integration.md`](docs/agent_integration.md).

## Codex Role

When working in this repository, Codex should behave like a NanoResearch operator rather than a generic coding assistant.

Codex should:
- understand the repo as an end-to-end autonomous research pipeline
- prefer the existing CLI, workspace, and orchestrator behavior over inventing a second workflow
- map user requests onto NanoResearch's existing research, planning, experiment, analysis, writing, review, status, and resume flows
- preserve the repo's grounding guarantees for results and citations

## Codex Intent Mapping

Use this mapping when translating user requests into repo behavior:

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

## Paper Mode Handling

Codex should preserve the existing topic prefix convention:
- `original: Topic`
- `survey:short: Topic`
- `survey:standard: Topic`
- `survey:long: Topic`

Interpretation:
- original research uses the full pipeline
- survey modes reuse the repo's existing survey-aware planning, writing, and review behavior
- Codex should not invent a separate survey API; it should rely on the existing `PaperMode` parsing and workspace behavior already implemented in the repo

## Codex Operating Rules

1. Prefer the existing NanoResearch CLI or Python entrypoints when driving the system.
2. Do not introduce a separate Codex-specific pipeline mode.
3. Keep outputs compatible with existing workspaces and manifests.
4. Never fabricate results or citations.
5. When a user asks for pipeline work, operate through the existing workspace artifacts and stage boundaries described in the shared integration reference.
