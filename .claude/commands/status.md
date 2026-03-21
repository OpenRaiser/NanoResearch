# Status — Check Research Pipeline Status

Show the current status of a NanoResearch workspace.

## Instructions

1. Find the most recent workspace under `~/.nanoresearch/workspace/research/`. If `$ARGUMENTS` is provided, treat it as a workspace path.

2. Read `manifest.json` from the workspace directory.

3. **Normalize the schema** before display (manifests come in two variants):
   - **Old schema** (`schema_version: "1.1"`): stage keys are UPPERCASE (`IDEATION`, `PLANNING`...), `current_stage` is UPPERCASE (`DONE`, `FAILED`...), may have extra stages (`INIT`, `FORMAT_FIX`), artifacts are `[{name, path, stage, checksum}]`
   - **New schema**: stage keys are lowercase, artifacts are `["path"]`
   - **Always display in lowercase.** Convert stage keys and `current_stage` to lowercase.
   - The canonical stage order is: `ideation`, `planning`, `setup`, `coding`, `execution`, `analysis`, `figure_gen`, `writing`, `review`. Skip any stages not in this list (e.g., `INIT`, `FORMAT_FIX`).
   - For artifacts: if an entry is an object, use its `path` field; if a string, use it directly.

4. Display a summary table:
   - Session ID
   - Topic
   - Schema version (old v1.1 / new)
   - Current stage
   - For each canonical stage: status (pending/running/completed/failed), timestamps
   - List of produced artifacts with file paths

5. If no workspace exists, say so and suggest running `/project:research "topic"` to start.

## Output Format

```
Session: {session_id}
Topic: {topic}
Schema: {v1.1 | claude-code}
Current Stage: {current_stage}

Stage          Status      Started              Completed
─────          ──────      ───────              ─────────
ideation       completed   2026-03-18 10:00:00  2026-03-18 10:05:00
planning       completed   2026-03-18 10:05:01  2026-03-18 10:10:00
setup          running     2026-03-18 10:10:01  -
...

Artifacts:
  - papers/ideation_output.json (12.3 KB)
  - plans/experiment_blueprint.json (8.1 KB)
```
