"""Iteration helpers: checkpoint, hypothesis, changes, history, imports, syntax."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from nanoresearch.agents._code_utils import _strip_code_fences
from nanoresearch.agents.repair_journal import (
    append_snapshot_journal,
    capture_repair_snapshot,
    rollback_snapshot,
)
from nanoresearch.schemas.iteration import (
    ExperimentHypothesis,
    FeedbackAnalysis,
    IterationState,
    RoundResult,
)
from nanoresearch.agents.experiment._iteration_helpers import _IterationHelpersMixin

logger = logging.getLogger(__name__)


class _IterationMixin(_IterationHelpersMixin):
    """Mixin — iteration checkpoint, hypothesis, changes, history."""

    def _save_iteration_checkpoint(
        self,
        state: IterationState,
        checkpoint_path: str = "logs/iteration_checkpoint.json",
    ) -> None:
        """Save iteration state checkpoint for crash recovery."""
        self.workspace.write_json(
            checkpoint_path,
            state.model_dump(),
        )

    def _load_iteration_checkpoint(
        self,
        default_state: IterationState,
        checkpoint_path: str = "logs/iteration_checkpoint.json",
    ) -> tuple[IterationState, int]:
        """Load iteration checkpoint if available.

        Returns (state, start_round) where start_round is the round to
        resume from (1 if no checkpoint exists).
        """
        try:
            data = self.workspace.read_json(checkpoint_path)
            if isinstance(data, dict) and data.get("rounds"):
                state = IterationState.model_validate(data)
                completed_rounds = len(state.rounds)
                start_round = completed_rounds + 1
                if start_round <= state.max_rounds:
                    logger.info(
                        "Resuming experiment from round %d (checkpoint has %d completed rounds)",
                        start_round, completed_rounds,
                    )
                    return state, start_round
                else:
                    logger.info(
                        "Checkpoint shows all %d rounds completed, starting fresh",
                        completed_rounds,
                    )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to load iteration checkpoint: %s", exc)
        return default_state, 1

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    async def _generate_iteration_hypothesis(
        self,
        analysis: FeedbackAnalysis | None,
        history_summary: str,
        blueprint: str,
        preflight_error_ctx: str = "",
        code_dir: Path | None = None,
    ) -> ExperimentHypothesis:
        """LLM generates the next iteration hypothesis from feedback."""
        analysis_text = ""
        if analysis:
            analysis_text = (
                f"Attribution: {analysis.attribution}\n"
                f"Recommended action: {analysis.recommended_action}\n"
                f"Metrics: {json.dumps(analysis.metric_summary)}\n"
                f"Training dynamics: convergence={analysis.training_dynamics.convergence_speed}, "
                f"overfitting={analysis.training_dynamics.overfitting_detected}, "
                f"stability={analysis.training_dynamics.loss_stability}\n"
                f"Error categories: {analysis.error_categories}"
        )

        # Collect actual file list from code_dir for the LLM
        code_dir = code_dir or (self.workspace.path / "code")
        actual_files = []
        if code_dir.exists():
            for f in sorted(code_dir.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f) and ".pyc" not in str(f):
                    actual_files.append(str(f.relative_to(code_dir)).replace("\\", "/"))
        file_list = "\n".join(f"  - {f}" for f in actual_files) if actual_files else "  (no files yet)"

        # Build list of previously tried hypotheses to prevent repetition
        prev_hypotheses = []
        if history_summary:
            for line in history_summary.split("\n"):
                if line.strip():
                    prev_hypotheses.append(line.strip())
        prev_hyp_block = "\n".join(prev_hypotheses) if prev_hypotheses else "None"

        prompt = f"""Based on the previous experiment round's feedback, generate a hypothesis for the next improvement iteration.
{preflight_error_ctx}
== Previous Analysis ==
{analysis_text or "No analysis available."}

== History ==
{history_summary or "No previous rounds."}

== PREVIOUSLY TRIED HYPOTHESES (DO NOT REPEAT) ==
{prev_hyp_block}

== Experiment Blueprint ==
{blueprint[:2000]}

== Actual Project Files ==
{file_list}

IMPORTANT RULES:
1. Only reference files that exist in the list above. Do NOT invent new file paths.
2. Use the EXACT paths shown above in your planned_changes.
3. The `--quick-eval` mode HARDCODES a small model and 3-5 epochs regardless of config.
   Changing epochs/batch_size/num_runs in config/default.yaml has NO EFFECT on quick-eval.
   DO NOT suggest increasing epochs or changing hyperparameters in config — it is USELESS.
4. Instead, focus on changes that actually affect quick-eval behavior:
   - Fix bugs in model architecture (src/model.py)
   - Fix bugs in training loop (src/trainer.py)
   - Fix evaluation/metrics collection (src/evaluate.py, src/utils.py)
   - Fix data loading/preprocessing (src/dataset.py)
   - Fix the quick-eval code path in main.py directly
   - Improve model architecture (e.g., add batch norm, better init, residual connections)
5. DO NOT repeat any hypothesis from the list above. Each round must try something DIFFERENT.
   If you cannot think of a genuinely new improvement, set "no_new_ideas": true.

Output a JSON object with:
{{
  "hypothesis": "<what you will change and why>",
  "planned_changes": ["<EXACT_FILE_PATH: specific change>", ...],
  "expected_signal": "<what metric improvement you expect>",
  "rationale": "<reasoning>",
  "no_new_ideas": false
}}"""

        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML experiment iteration planner. Generate a focused hypothesis for the next improvement round. Output ONLY valid JSON.",
                prompt,
                json_mode=True,
            )
            data = self._parse_llm_json_payload(raw)

            # If LLM says no new ideas, signal early stop
            if data.get("no_new_ideas"):
                logger.info("LLM reports no new ideas — will signal early stop")
                return ExperimentHypothesis(
                    round_number=0,
                    hypothesis="__NO_NEW_IDEAS__",
                    planned_changes=[],
                    expected_signal="",
                    rationale="LLM exhausted improvement ideas",
                )

            return ExperimentHypothesis(
                round_number=0,  # caller sets this
                hypothesis=data.get("hypothesis", "Iterative improvement"),
                planned_changes=data.get("planned_changes", []),
                expected_signal=data.get("expected_signal", ""),
                rationale=data.get("rationale", ""),
            )
        except Exception as exc:
            logger.warning("Failed to generate hypothesis: %s", exc)
            return ExperimentHypothesis(
                round_number=0,
                hypothesis="Retry with general improvements based on error feedback",
                planned_changes=["Fix errors from previous round"],
                expected_signal="Successful execution",
                rationale="Fallback hypothesis after LLM generation failure",
            )

    async def _apply_iteration_changes(
        self,
        hypothesis: ExperimentHypothesis,
        code_dir: Path,
        venv_python: str,
    ) -> list[str]:
        """LLM modifies specific files using search-replace edits (OpenClaw style).

        Uses precise search-replace blocks instead of full file rewrites to:
        1. Reduce token usage (LLM only outputs the diff, not entire files)
        2. Avoid accidental deletion of unchanged code
        3. Make changes auditable
        """
        self._remember_mutation_snapshot_entry(None)
        # Collect current file contents for context
        file_contents: dict[str, str] = {}
        for py_file in code_dir.rglob("*.py"):
            parts = py_file.relative_to(code_dir).parts
            if any(p.startswith(".") or p == "__pycache__" for p in parts):
                continue
            try:
                rel = str(py_file.relative_to(code_dir)).replace("\\", "/")
                content = py_file.read_text(encoding="utf-8", errors="replace")
                file_contents[rel] = content
            except OSError:
                continue

        # Also include config and other non-py files
        for pattern in ("config/*.yaml", "config/*.yml", "*.txt", "*.sh"):
            for f in code_dir.glob(pattern):
                try:
                    rel = str(f.relative_to(code_dir)).replace("\\", "/")
                    file_contents[rel] = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    pass

        files_summary = "\n".join(
            f"--- {path} ---\n{content[:2000]}\n"
            for path, content in file_contents.items()
        )

        prompt = f"""Apply the following changes to the experiment code project using SEARCH-REPLACE edits.

== Hypothesis ==
{hypothesis.hypothesis}

== Planned Changes ==
{json.dumps(hypothesis.planned_changes, indent=2)}

== Rationale ==
{hypothesis.rationale}

== Current Files ==
{files_summary[:15000]}

Output a JSON array of edit operations. Two types are supported:

1. **Search-replace edit** (preferred for modifying existing files):
{{
  "path": "relative/path.py",
  "action": "edit",
  "edits": [
    {{"old": "exact text to find", "new": "replacement text"}}
  ]
}}

2. **Full file write** (only for NEW files that don't exist yet):
{{
  "path": "relative/new_file.py",
  "action": "write",
  "content": "full file content"
}}

IMPORTANT RULES:
- "old" must be an EXACT substring of the current file content (including whitespace/indentation)
- Each "old" string must be unique within its file
- Use search-replace for ALL modifications to existing files
- Only use "write" action for creating brand new files
- Multiple edits per file are fine — they are applied sequentially

Output ONLY valid JSON array."""

        modified_files: list[str] = []
        snapshot_batch: list[dict[str, Any]] = []
        try:
            code_gen_config = self.config.for_stage("code_gen")
            raw = await self._dispatcher.generate(
                code_gen_config,
                "You are an ML code editor. Apply precise search-replace edits to implement the hypothesis. Output ONLY a JSON array.",
                prompt,
            )
            changes = self._parse_llm_json_payload(raw)
            if not isinstance(changes, list):
                changes = [changes]

            for change in changes:
                if not isinstance(change, dict) or "path" not in change:
                    continue
                file_path = change["path"]
                # Security: prevent directory traversal
                try:
                    (code_dir / file_path).resolve().relative_to(code_dir.resolve())
                except ValueError:
                    logger.warning("Skipping unsafe iteration path: %s", file_path)
                    continue

                action = change.get("action", "write")  # backwards compat

                if action == "edit":
                    # Search-replace mode
                    edits = change.get("edits", [])
                    if not edits:
                        continue
                    # Read current content
                    target = code_dir / file_path
                    if not target.exists():
                        logger.warning("Edit target does not exist: %s", file_path)
                        continue
                    try:
                        current = target.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        continue

                    applied = 0
                    for edit in edits:
                        if not isinstance(edit, dict):
                            continue
                        old = edit.get("old", "")
                        new = edit.get("new", "")
                        if not old:
                            continue
                        current, matched, match_strategy = self._apply_search_replace_edit(
                            current, old, new,
                        )
                        if matched:
                            applied += 1
                            self.log(f"  Matched edit in {file_path} via {match_strategy}")
                        else:
                            logger.warning(
                                "Edit old text not found in %s: %s",
                                file_path, old[:80],
                            )

                    if applied > 0:
                        target_path = code_dir / file_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        snapshot = capture_repair_snapshot(
                            self.workspace.path, target_path,
                            namespace="iteration_changes",
                            root_dir=self.workspace.path, operation="rewrite",
                        )
                        target_path.write_text(current, encoding="utf-8")
                        if target_path.suffix.lower() == ".py" and not self._check_syntax(target_path):
                            self.log(f"  Edited file became invalid Python in {file_path}, rolling back")
                            rollback_snapshot(self.workspace.path, target_path, snapshot)
                            snapshot["rolled_back"] = True
                            snapshot["rollback_reason"] = "syntax_error"
                            snapshot_batch.append(snapshot)
                            continue

                        modified_files.append(file_path)
                        snapshot_batch.append(snapshot)
                        self.log(f"  Edited: {file_path} ({applied}/{len(edits)} edits applied)")
                else:
                    # Full write mode (new files or backwards compat)
                    content = change.get("content", "")
                    if not content:
                        continue
                    target_path = code_dir / file_path
                    existed_before = target_path.exists()
                    snapshot = capture_repair_snapshot(
                        self.workspace.path, target_path,
                        namespace="iteration_changes",
                        root_dir=self.workspace.path,
                        existed_before=existed_before,
                        operation="rewrite" if existed_before else "create",
                    )
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    target_path.write_text(content, encoding="utf-8")
                    if target_path.suffix.lower() == ".py" and not self._check_syntax(target_path):
                        self.log(f"  Wrote invalid Python in {file_path}, rolling back")
                        rollback_snapshot(self.workspace.path, target_path, snapshot)
                        snapshot["rolled_back"] = True
                        snapshot["rollback_reason"] = "syntax_error"
                        snapshot_batch.append(snapshot)
                        continue

                    modified_files.append(file_path)
                    snapshot_batch.append(snapshot)
                    self.log(f"  Wrote: {file_path}")

        except Exception as exc:
            logger.warning("Failed to apply iteration changes: %s", exc)

        if snapshot_batch:
            entry = append_snapshot_journal(
                self.workspace.path,
                agent=self.__class__.__name__,
                mutation_kind="iteration_changes",
                scope="legacy_iteration_search_replace",
                snapshots=snapshot_batch,
                metadata={"modified_files": list(modified_files)},
            )
            self._remember_mutation_snapshot_entry(entry)
        return modified_files
