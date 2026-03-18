# Resume — Resume Pipeline from Last Checkpoint

Resume a NanoResearch pipeline that was interrupted or failed.

## Input

`$ARGUMENTS` — workspace path (optional). If not provided, use the most recent workspace under `~/.nanobot/workspace/research/`.

## Schema Compatibility

Manifests come in two schemas — **normalize before processing**:

1. **Old schema** (Python pipeline, `schema_version: "1.1"`):
   - Stage keys are UPPERCASE: `IDEATION`, `PLANNING`, `SETUP`, `CODING`, `EXECUTION`, `ANALYSIS`, `FIGURE_GEN`, `WRITING`, `REVIEW`
   - `current_stage` is UPPERCASE: `DONE`, `FAILED`, `IDEATION`, etc.
   - May contain extra stages (`INIT`, `FORMAT_FIX`) — ignore these
   - Stage objects have extra fields: `retries`, `error_message`, `output_path`
   - `artifacts` is an array of objects `{name, path, stage, checksum}`

2. **New schema** (Claude Code commands):
   - Stage keys are lowercase: `ideation`, `planning`, etc.
   - `current_stage` is lowercase: `done`, `failed`, `ideation`, etc.
   - `artifacts` is a simple array of path strings

**Normalization rules:**
- Convert all stage keys and `current_stage` to **lowercase** for comparison
- The canonical stage order is: `ideation`, `planning`, `setup`, `coding`, `execution`, `analysis`, `figure_gen`, `writing`, `review`
- Skip any stages not in the canonical list (e.g., `INIT`, `FORMAT_FIX`)
- For artifacts, if an entry is an object, use its `path` field; if it's a string, use it directly

## Process

1. Read `{workspace}/manifest.json`

2. Normalize the manifest (see Schema Compatibility above).

3. Find the current state (use lowercase comparison):
   - If `current_stage` is "done": Tell the user the pipeline is already complete.
   - If `current_stage` is "failed": Find the failed stage, reset it to "pending", and continue from there. Check `error_message` field if present (old schema).
   - Otherwise: Continue from `current_stage`.

4. Identify which stages are already completed by checking each stage's status (case-insensitive: "completed" or "COMPLETED" both count).

5. Resume execution from the first non-completed stage, following the same process as `/project:research`:
   - Read all existing artifacts from completed stages
   - Execute the remaining stages sequentially
   - Update manifest after each stage (use lowercase keys for new entries)

6. Show progress:
   ```
   Schema: {old v1.1 | new}
   Resuming from stage: {stage_name}
   Completed stages: ideation, planning, setup
   Remaining stages: coding, execution, analysis, figure_gen, writing, review
   ```

## Error Recovery

If a stage previously failed:
- Check `error_message` field (old schema) or read from `logs/` for details
- Attempt to diagnose the issue
- Fix if possible (e.g., missing package → install it)
- Re-run the stage

If the same stage fails again:
- Report the error clearly
- Ask the user for guidance
