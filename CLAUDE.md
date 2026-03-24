# NanoResearch — Claude Code Integration Mode

This file is the Claude Code entrypoint for NanoResearch.

For shared project workflow, paper modes, workspace conventions, and stage semantics, read [`docs/agent_integration.md`](docs/agent_integration.md).

## Claude Code Role

In Claude Code integration mode, Claude Code acts as the research engine using its native tools:
- **WebSearch** for literature retrieval
- **Bash** for code execution, SLURM submission, and LaTeX compilation
- **File read/write** for workspace artifacts, code, and paper drafts

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

## Claude Code Rules

1. Use the existing workspace and manifest conventions from the shared integration reference.
2. Reuse the existing topic prefix syntax for survey modes:
   - `survey:short:`
   - `survey:standard:`
   - `survey:long:`
   - `original:`
3. Prefer NanoResearch's existing pipeline and outputs over custom one-off scripts.
4. Never fabricate results or citations.
5. Keep workspaces compatible with the Python CLI.
